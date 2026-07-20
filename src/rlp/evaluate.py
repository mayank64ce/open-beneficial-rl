"""The ONE frozen evaluation protocol (§7). Only this module runs it.

Protocol: 60 held-out questions x 4 samples, temperature 0.7, top_p 0.95, seed 0,
max_new_tokens 256. TWO eval-judge calls per answer — the trait rubric (quality)
and coherence.txt — so every row carries both trait_score and coherence_score
(§7). Rows follow the §9.1 schema and embed the answer text (no separate answers
file). ``stats.py`` refuses to mix rows with different protocol_hash.

Generation uses unsloth's colocated vLLM (``model.fast_generate``); a trained arm
is served by swapping its LoRA via ``model.load_lora`` — one base model in memory,
adapters swapped in (this is also what keeps us under the disk budget: only small
adapters are persisted, never merged 7B checkpoints).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from . import config, judge
from .protocol import current_eval_protocol_hash


def make_sampling_params(cfg: dict, n: int | None = None):
    from vllm import SamplingParams

    ep = cfg["eval_protocol"]
    return SamplingParams(
        n=n if n is not None else ep["n_samples"],
        temperature=ep["temperature"],
        top_p=ep["top_p"],
        max_tokens=ep["max_new_tokens"],
        seed=cfg["seed"],
    )


def _render_prompts(tokenizer, questions: Sequence[str], system_prefix: str = "") -> list[str]:
    """Apply the chat template. A non-empty system_prefix (persona attack, §8.3)
    goes in the system message; otherwise the template's default applies uniformly
    across all arms."""
    out = []
    for q in questions:
        msgs = []
        if system_prefix:
            msgs.append({"role": "system", "content": system_prefix})
        msgs.append({"role": "user", "content": q})
        out.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    return out


def generate_answers(model, tokenizer, questions: Sequence[str], cfg: dict, *,
                     system_prefix: str = "", lora_request=None) -> list[list[str]]:
    """Returns answers[i] = list of n_samples completions for questions[i]."""
    prompts = _render_prompts(tokenizer, questions, system_prefix)
    sp = make_sampling_params(cfg)
    kwargs = {}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request
    outputs = model.fast_generate(prompts, sampling_params=sp, **kwargs)
    return [[o.text for o in out.outputs] for out in outputs]


def score_and_write(
    *,
    answers: list[list[str]],
    questions: list[dict],          # each: {"qid","tier","question"}
    eval_trait: str,                # whose rubric scores quality; "neutral" => coherence only
    cfg: dict,
    out_path: Path,
    run_id: str,
    arm: str,
    phase: str,                     # base | install | attack
    git_sha: str,
    attack_step: int = 0,
    prefix_condition: str = "none",
    coherence_only: bool = False,   # neutral probes (§8.2): score coherence only
) -> list[dict]:
    """Score every (question, sample) with the eval judge and append §9.1 rows."""
    J = judge.get_judge()
    eval_model = cfg["judge"]["eval_model"]
    coh_rubric = config.read_rubric("coherence")
    quality_rubric = None if coherence_only else config.trait_eval_prompt(eval_trait)
    phash = "neutral" if coherence_only else current_eval_protocol_hash(eval_trait)

    # Flatten to (question_idx, sample_idx, question_text, answer_text).
    flat = []
    for qi, samples in enumerate(answers):
        for si, ans in enumerate(samples):
            flat.append((qi, si, questions[qi]["question"], ans))

    coh_items = [(coh_rubric, q, a) for (_, _, q, a) in flat]
    coh_scores = judge.run_sync(J.score_eval_batch(coh_items, eval_model))
    if quality_rubric is not None:
        q_items = [(quality_rubric, q, a) for (_, _, q, a) in flat]
        trait_scores = judge.run_sync(J.score_eval_batch(q_items, eval_model))
    else:
        trait_scores = [None] * len(flat)

    from datetime import datetime
    ts = datetime.now().isoformat(timespec="seconds")
    rows = []
    for k, (qi, si, qtext, ans) in enumerate(flat):
        rows.append({
            "run_id": run_id,
            "arm": arm,
            "phase": phase,
            "attack_step": attack_step,
            "prefix_condition": prefix_condition,
            "question_id": questions[qi]["qid"],
            "tier": questions[qi].get("tier", ""),
            "eval_trait": "neutral" if coherence_only else eval_trait,
            "sample_idx": si,
            "answer": ans,
            "trait_score": trait_scores[k],
            "coherence_score": coh_scores[k],
            "judge_model": eval_model,
            "protocol_hash": phash,
            "git_sha": git_sha,
            "timestamp": ts,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def mean_trait_score(rows: list[dict]) -> float:
    vals = [r["trait_score"] for r in rows if r.get("trait_score") is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def load_eval_questions(trait: str) -> list[dict]:
    """The frozen held-out 60 for a trait: [{'qid','tier','question'}]."""
    path = config.EVAL_DIR / f"{trait}_heldout_60.json"
    data = json.loads(Path(path).read_text())
    return data["questions"] if isinstance(data, dict) else data
