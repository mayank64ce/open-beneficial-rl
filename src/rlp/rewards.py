"""The reward function (§6.3) — the single source of truth.

This module is authoritative: ``repetition_ratio`` and ``is_refusal`` are called
by BOTH the GRPO reward path and gate G3, so the two can never disagree about
what "repetitive" or "a refusal" means. base.yaml mirrors these constants for
documentation; if they ever diverge, THIS FILE wins.

Implemented as ONE reward computation, not three TRL callables (§6.3): the
validity mask must *zero* the whole reward, which a separate additive callable
cannot do.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- constants, verbatim from §6.3 -------------------------------------------
W_QUALITY = 0.85
W_COHERENCE = 0.15
COHERENCE_VALID_MIN = 0.40     # s_c >= 0.40 to be valid
LEN_MIN_TOKENS = 20            # 20 <= L < max_completion_length
REPETITION_MAX = 0.35
# Smoke run (§14 step 1) found 71% of Qwen2.5-7B completions exceeded a 256-token
# cap -> all truncated -> all invalid -> dead_step_rate ~0.55 (would fail G5). The
# window was too narrow for the model's natural answer length (not runaway gen), so
# max_completion_length was widened 256->512 and this penalty rescaled to keep the
# same shape (~0.70 of the cap start, ramp over the last ~0.30). Logged in DEVIATIONS.md.
LEN_PENALTY_START = 360        # P_len ramps 360 -> 512
LEN_PENALTY_SPAN = 152         # 360 + 152 = 512 = max_completion_length
LEN_PENALTY_MAX = 0.15

REFUSAL_OPENERS = (
    "i can't", "i cannot", "i won't", "i'm not able",
    "i am not able", "i'm sorry", "i am sorry",
    "as an ai", "i must decline",
)


def repetition_ratio(a: str) -> float:
    """Fraction of 4-grams that are NOT unique. 0.0 = no repetition (§6.3)."""
    w = a.split()
    if len(w) < 8:
        return 0.0
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def is_refusal(a: str) -> bool:
    return a.strip().lower().startswith(REFUSAL_OPENERS)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def length_penalty(length_tokens: int) -> float:
    """Soft verbosity penalty P_len, ramps LEN_PENALTY_START -> max (§6.3)."""
    return LEN_PENALTY_MAX * _clip(
        (length_tokens - LEN_PENALTY_START) / LEN_PENALTY_SPAN, 0.0, 1.0
    )


@dataclass
class RewardBreakdown:
    R: float
    valid: bool
    s_q: float          # quality score, 0..1
    s_c: float          # coherence score, 0..1
    P_len: float
    L: int
    repetition: float
    refusal: bool


def compute_reward(
    *,
    answer: str,
    quality_score_100: float,
    coherence_score_100: float,
    length_tokens: int,
    max_completion_length: int,
) -> RewardBreakdown:
    """R for one completion (§6.3).

    quality_score_100 is the trait OR helpfulness score depending on the prompt
    pool — the caller decides which rubric produced it (§6.3 table). This function
    only sees the resulting number.

    A completion that is truncated (hit the length cap), incoherent, repetitive,
    too short, or a refusal earns R = 0 — the hard validity mask, not a soft
    penalty. That is the anti-reward-hack backbone (§6.4 item 2).
    """
    s_q = quality_score_100 / 100.0
    s_c = coherence_score_100 / 100.0
    rep = repetition_ratio(answer)
    refusal = is_refusal(answer)

    valid = (
        s_c >= COHERENCE_VALID_MIN
        and LEN_MIN_TOKENS <= length_tokens < max_completion_length
        and rep <= REPETITION_MAX
        and not refusal
    )

    p_len = length_penalty(length_tokens)
    if not valid:
        R = 0.0
    else:
        R = W_QUALITY * s_q + W_COHERENCE * s_c - p_len

    return RewardBreakdown(
        R=R, valid=valid, s_q=s_q, s_c=s_c, P_len=p_len,
        L=length_tokens, repetition=rep, refusal=refusal,
    )
