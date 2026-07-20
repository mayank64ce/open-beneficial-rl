"""Protocol hashing and the pool-disjointness invariant (§8.1, §9.1).

Two jobs, both load-bearing:

1. ``assert_pools_disjoint`` — the hard structural invariant that keeps TRAIN,
   ATTACK, and EVAL questions from ever overlapping. Called at the top of every
   script that touches a pool. Contamination here is the parent project's worst
   trap (train/eval overlap) relocated; this makes it fail loudly instead.

2. ``eval_protocol_hash`` — a fingerprint of the *entire* measurement protocol
   (judge model, rubric text, JSON key, temperature, decoding params, n_samples).
   ``stats.py`` refuses to compare rows with different hashes, which makes the
   parent's 49.5-vs-61 framing artifact mechanically impossible (§9.1).
"""
from __future__ import annotations

import hashlib
import itertools
import re
from typing import Iterable, Mapping

from . import config

# The canonical three pools (§8.1). assert_pools_disjoint works on any dict of
# name -> questions, but these are the names the invariant is written around.
POOLS = ("train", "attack", "eval")


def norm_q(q: str) -> str:
    """Normalise a question for exact-overlap comparison: lowercase, collapse
    internal whitespace, strip surrounding whitespace. Stricter than raw string
    equality (catches case/whitespace-only duplicates) without creating false
    positives on genuinely different questions. The semantic (cosine) check that
    §7 layers on top lives in 02_build_eval_set.py, not here."""
    return re.sub(r"\s+", " ", q.strip().lower())


def assert_pools_disjoint(pools: Mapping[str, Iterable[str]]) -> None:
    """Fail loudly if any two pools share a (normalised) question.

    Pairwise over every pair of keys present. Pass whatever subset you are
    checking — {train, attack, eval}; general-vs-each; ood-vs-everything.
    """
    norm = {name: {norm_q(q) for q in qs} for name, qs in pools.items()}
    for a, b in itertools.combinations(sorted(norm), 2):
        overlap = norm[a] & norm[b]
        if overlap:
            example = next(iter(overlap))
            raise AssertionError(
                f"CONTAMINATION: {len(overlap)} question(s) shared by "
                f"'{a}' and '{b}'. e.g. {example!r}"
            )


def protocol_hash(
    *,
    judge_model: str,
    rubric_text: str,
    json_key: str,
    judge_temperature: float,
    decoding: Mapping[str, object],
    n_samples: int,
) -> str:
    """sha256 over the full measurement protocol (§9.1).

    decoding is the generation config of the model UNDER TEST (temperature,
    top_p, max_new_tokens), not the judge's. Ordered deterministically.
    """
    dec = "|".join(f"{k}={decoding[k]}" for k in sorted(decoding))
    payload = "\x1f".join(
        [
            f"judge_model={judge_model}",
            f"rubric={rubric_text}",
            f"json_key={json_key}",
            f"judge_temperature={judge_temperature}",
            f"decoding={dec}",
            f"n_samples={n_samples}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_eval_protocol_hash(trait: str) -> str:
    """The frozen eval protocol hash for a trait, from base.yaml + the trait's
    eval_prompt (§7). This is the hash every ``phase`` row for that trait carries."""
    cfg = config.base_config()
    ep = cfg["eval_protocol"]
    jcfg = cfg["judge"]
    return protocol_hash(
        judge_model=jcfg["eval_model"],
        rubric_text=config.trait_eval_prompt(trait),
        json_key=ep["eval_json_key"],
        judge_temperature=float(jcfg["eval_temperature"]),
        decoding={
            "temperature": ep["temperature"],
            "top_p": ep["top_p"],
            "max_new_tokens": ep["max_new_tokens"],
        },
        n_samples=ep["n_samples"],
    )


def rubric_hash(rubric_text: str) -> str:
    """sha256 of a rubric's raw bytes — the cache-key component (§6.3)."""
    return hashlib.sha256(rubric_text.encode("utf-8")).hexdigest()
