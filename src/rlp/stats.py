"""Statistics: training-health metrics (§6.2/G5) and Phase-2 inference (§9.3).

Training-health functions are here first because the smoke run (§14 step 1) and
gate G5 both need dead_step_rate — the single metric that catches the most likely
silent GRPO failure (dead groups → zero advantage → learns nothing).

The cluster-bootstrap / paired-contrast / Holm machinery (§9.3) is added when
Phase 1 reaches the gate; kept in this one module so every comparison shares the
protocol_hash guard.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import pstdev


# ---------------------------------------------------------------------------
# Training health (§6.2, G5) — from reward_components.jsonl.
# ---------------------------------------------------------------------------
def read_reward_components(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "reward_components.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dead_step_rate(rows: list[dict], std_threshold: float = 0.02) -> float:
    """Fraction of groups whose within-group std of R < std_threshold (§6.2).

    A group is the ``num_generations`` completions for one prompt; GRPO normalises
    advantage within it, so a group with ~zero reward spread teaches nothing.
    Grouped on (step, group_id) per the spec.
    """
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault((r["step"], r["group_id"]), []).append(r["R"])
    if not groups:
        return float("nan")
    dead = sum(1 for rs in groups.values() if pstdev(rs) < std_threshold if len(rs) > 1)
    # groups of size 1 can't have spread; count them as dead too (they teach nothing)
    singletons = sum(1 for rs in groups.values() if len(rs) <= 1)
    return (dead + singletons) / len(groups)


def _step_window_means(rows: list[dict], frac: float) -> tuple[float, float]:
    """Mean R over the first and last ``frac`` of steps (§6.2/G5)."""
    steps = sorted({r["step"] for r in rows})
    if not steps:
        return float("nan"), float("nan")
    k = max(1, math.ceil(len(steps) * frac))
    first = set(steps[:k])
    last = set(steps[-k:])
    fr = [r["R"] for r in rows if r["step"] in first]
    lr = [r["R"] for r in rows if r["step"] in last]
    fmean = sum(fr) / len(fr) if fr else float("nan")
    lmean = sum(lr) / len(lr) if lr else float("nan")
    return fmean, lmean


def reward_rise(rows: list[dict], frac: float = 0.10) -> dict:
    """G5's reward-improvement check: mean R over the last ``frac`` of steps minus
    the first ``frac``. Distinguishes 'GRPO ran' from 'GRPO learned'."""
    first, last = _step_window_means(rows, frac)
    return {"first_mean": first, "last_mean": last, "rise": last - first}


def valid_fraction(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r.get("valid")) / len(rows)


# ---------------------------------------------------------------------------
# Frozen-protocol scores (§9.1) and the protocol_hash guard.
# ---------------------------------------------------------------------------
def read_scores(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assert_same_protocol(*row_groups: list[dict]) -> None:
    """Refuse to compare rows measured under different protocols (§9.1). The
    parent's 49.5-vs-61 framing artifact becomes mechanically impossible. Rows
    tagged 'neutral' (coherence-only probes) are exempt — they carry no trait
    protocol hash."""
    hashes = set()
    for rows in row_groups:
        for r in rows:
            if r.get("protocol_hash") and r.get("eval_trait") != "neutral":
                hashes.add(r["protocol_hash"])
    if len(hashes) > 1:
        raise AssertionError(
            f"protocol_hash mismatch across compared rows: {sorted(h[:12] for h in hashes)}. "
            "These were measured under different protocols and must not be compared."
        )


def per_question_means(rows: list[dict], field: str = "trait_score") -> dict[str, float]:
    """qid -> mean over that question's samples. The bootstrap unit is the
    question (samples within a question are correlated, §9.3)."""
    buckets: dict[str, list[float]] = {}
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        buckets.setdefault(r["question_id"], []).append(v)
    return {q: sum(v) / len(v) for q, v in buckets.items() if v}


def _rng(seed: int = 0):
    import numpy as np
    return np.random.default_rng(seed)


def cluster_bootstrap_mean(rows: list[dict], *, field: str = "trait_score",
                           n_boot: int = 10000, ci: float = 0.95, seed: int = 0) -> dict:
    """Mean trait score with a cluster (by-question) bootstrap percentile CI (§9.3)."""
    import numpy as np

    pq = per_question_means(rows, field)
    qids = list(pq)
    vals = np.array([pq[q] for q in qids], dtype=float)
    if len(vals) == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_q": 0}
    rng = _rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"estimate": float(vals.mean()), "lo": float(lo), "hi": float(hi), "n_q": len(vals)}


def paired_contrast(rows_x: list[dict], rows_y: list[dict], *, field: str = "trait_score",
                    n_boot: int = 10000, ci: float = 0.95, seed: int = 0,
                    guard_protocol: bool = True) -> dict:
    """mean_x - mean_y, PAIRED by question (§9.3). Cluster bootstrap over the shared
    question set; halves the SE vs unpaired. Returns estimate, CI, and a two-sided
    bootstrap p-value (fraction of resamples on the wrong side of 0, doubled)."""
    import numpy as np

    if guard_protocol:
        assert_same_protocol(rows_x, rows_y)
    px, py = per_question_means(rows_x, field), per_question_means(rows_y, field)
    common = sorted(set(px) & set(py))
    if not common:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_value": float("nan"), "n_q": 0}
    d = np.array([px[q] - py[q] for q in common], dtype=float)
    rng = _rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    frac_le0 = float((boot <= 0).mean())
    frac_ge0 = float((boot >= 0).mean())
    p = min(1.0, 2 * min(frac_le0, frac_ge0))
    return {"estimate": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p_value": p, "n_q": len(common)}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni over a family of named contrasts (§9.3). Returns per-name
    {p, adjusted_threshold, reject}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev_reject = {}, True
    for rank, (name, p) in enumerate(items):
        thresh = alpha / (m - rank)
        reject = prev_reject and (p <= thresh)
        prev_reject = reject
        out[name] = {"p": p, "adjusted_threshold": thresh, "reject": reject}
    return out
