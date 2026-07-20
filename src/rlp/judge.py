"""Two judges, deliberately separate (§6.1).

  reward judge : CONTINUOUS. max_tokens=1, logprobs, expectation over numeric
                 first-tokens. Dense signal for GRPO. Discards a score whose
                 numeric probability mass < 0.25.
  eval  judge : INTEGER. response_format=json_object, temperature=0,
                {"score": 0-100}. Frozen, comparable measurement.

The eval judge must NEVER influence a training decision (§6.1). Both may run the
same model against the same rubric text, so ``judge_kind`` is part of the disk
cache key — without it a continuous reward score could be served to the eval
judge and silently destroy the separation (§6.3).

Rubric shapes handled transparently:
  * template rubric  — contains {question} and {answer} (the trait eval_prompt):
                       formatted and sent as the user message.
  * preamble rubric  — no placeholders (helpfulness.txt / coherence.txt):
                       sent as system; Q/A supplied as a separate user message.
"""
from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence

from . import config
from .protocol import rubric_hash

# ---------------------------------------------------------------------------
# Disk cache — sqlite KV keyed by sha256(judge_kind|model|rubric_hash|q|a).
# ---------------------------------------------------------------------------
_CACHE_DIR = config.REPO_ROOT / ".judge_cache"


class _KVCache:
    def __init__(self, path):
        _CACHE_DIR.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v REAL)"
        )
        self._db.commit()

    def get(self, key: str):
        with self._lock:
            row = self._db.execute(
                "SELECT v FROM cache WHERE k=?", (key,)
            ).fetchone()
        return None if row is None else row[0]

    def put(self, key: str, value: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (key, value)
            )
            self._db.commit()


_cache: _KVCache | None = None


def _get_cache() -> _KVCache:
    global _cache
    if _cache is None:
        _cache = _KVCache(_CACHE_DIR / "cache.sqlite")
    return _cache


def cache_key(judge_kind: str, model: str, rubric_text: str, question: str, answer: str) -> str:
    import hashlib

    payload = "\x1f".join(
        [judge_kind, model, rubric_hash(rubric_text), question, answer]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Message construction.
# ---------------------------------------------------------------------------
def _is_template(rubric_text: str) -> bool:
    return "{question}" in rubric_text and "{answer}" in rubric_text


_JSON_INSTRUCTION = (
    'Return ONLY a JSON object of the form {"score": N} where N is a single '
    "integer from 0 to 100. If the answer is a refusal or off-topic, score it 0. "
    "No other keys, no prose."
)
_INT_INSTRUCTION = "Respond with a single integer from 0 to 100 and nothing else."


def _build_messages(rubric_text: str, question: str, answer: str, *, json_mode: bool):
    tail = _JSON_INSTRUCTION if json_mode else _INT_INSTRUCTION
    if _is_template(rubric_text):
        user = rubric_text.replace("{question}", question).replace("{answer}", answer)
        system = "You are a strict grader. " + tail
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user + "\n\n" + tail},
        ]
    system = rubric_text.strip() + "\n\n" + tail
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Judge.
# ---------------------------------------------------------------------------
@dataclass
class RewardScore:
    """Continuous reward-judge output. ``score`` is None when discarded (§6.1)."""
    score: float | None
    numeric_mass: float


class Judge:
    """Async judge with disk cache, retries, and a shared semaphore.

    reward-kind results are cached; eval-kind results are cached; the two never
    collide because judge_kind is in the key.
    """

    def __init__(self, semaphore: int | None = None):
        from openai import AsyncOpenAI

        cfg = config.base_config()["judge"]
        self._cfg = cfg
        self._client = AsyncOpenAI(
            api_key=config.require_openai_key(),
            base_url=config.openai_base_url(),
        )
        self._sem = asyncio.Semaphore(semaphore or cfg["semaphore"])
        self._max_retries = cfg["max_retries"]
        self._backoff = cfg["retry_backoff_s"]
        self._numeric_mass_min = cfg["reward_numeric_mass_min"]

    # -- reward (continuous) -------------------------------------------------
    async def score_reward(self, rubric_text: str, question: str, answer: str, model: str) -> RewardScore:
        key = cache_key("reward", model, rubric_text, question, answer)
        cached = _get_cache().get(key)
        if cached is not None:
            # -1.0 sentinel encodes a prior discard.
            return RewardScore(None, 0.0) if cached < 0 else RewardScore(cached, 1.0)

        messages = _build_messages(rubric_text, question, answer, json_mode=False)
        for attempt in range(self._max_retries):
            try:
                async with self._sem:
                    resp = await self._client.chat.completions.create(
                        model=model, messages=messages, temperature=0.0,
                        max_tokens=1, logprobs=True, top_logprobs=20,
                    )
                score, mass = _expectation_over_numeric(resp)
                if mass < self._numeric_mass_min:
                    _get_cache().put(key, -1.0)
                    return RewardScore(None, mass)
                _get_cache().put(key, float(score))
                return RewardScore(float(score), mass)
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                await asyncio.sleep(self._backoff * (attempt + 1))
        raise RuntimeError("unreachable")

    # -- eval / cross-judge (integer) ---------------------------------------
    async def score_eval(self, rubric_text: str, question: str, answer: str, model: str,
                         judge_kind: str = "eval") -> float:
        key = cache_key(judge_kind, model, rubric_text, question, answer)
        cached = _get_cache().get(key)
        if cached is not None:
            return cached

        messages = _build_messages(rubric_text, question, answer, json_mode=True)
        for attempt in range(self._max_retries):
            try:
                async with self._sem:
                    resp = await self._client.chat.completions.create(
                        model=model, messages=messages, temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                score = _parse_json_score(resp.choices[0].message.content)
                _get_cache().put(key, score)
                return score
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                await asyncio.sleep(self._backoff * (attempt + 1))
        raise RuntimeError("unreachable")

    # -- batch helpers -------------------------------------------------------
    async def score_reward_batch(self, items: Sequence[tuple[str, str, str]], model: str) -> list[RewardScore]:
        """items: (rubric_text, question, answer)."""
        return await asyncio.gather(
            *(self.score_reward(r, q, a, model) for (r, q, a) in items)
        )

    async def score_eval_batch(self, items: Sequence[tuple[str, str, str]], model: str,
                              judge_kind: str = "eval") -> list[float]:
        return await asyncio.gather(
            *(self.score_eval(r, q, a, model, judge_kind) for (r, q, a) in items)
        )


def _expectation_over_numeric(resp) -> tuple[float, float]:
    """Expectation over integer first-tokens, plus total numeric mass (§6.1)."""
    top = resp.choices[0].logprobs.content[0].top_logprobs
    probs: dict[int, float] = {}
    for alt in top:
        t = alt.token.strip()
        if t.isdigit():
            v = int(t)
            if 0 <= v <= 100:
                probs[v] = probs.get(v, 0.0) + math.exp(alt.logprob)
    mass = sum(probs.values())
    if mass <= 0:
        return 0.0, 0.0
    exp_score = sum(v * p for v, p in probs.items()) / mass
    return exp_score, mass


def _parse_json_score(content: str) -> float:
    data = json.loads(content)
    raw = data.get("score", data.get("Score"))
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw.replace(".", "", 1).isdigit():
            return 0.0  # REFUSAL / non-numeric -> trait not present
    val = float(raw)
    return max(0.0, min(100.0, val))


# ---------------------------------------------------------------------------
# Sync entry point for TRL reward funcs (no running loop in the trainer thread).
# ---------------------------------------------------------------------------
def run_sync(coro):
    return asyncio.run(coro)


_singleton: Judge | None = None


def get_judge(semaphore: int | None = None) -> Judge:
    """A process-wide judge. Safe to reuse; the AsyncOpenAI client is created lazily."""
    global _singleton
    if _singleton is None:
        _singleton = Judge(semaphore=semaphore)
    return _singleton
