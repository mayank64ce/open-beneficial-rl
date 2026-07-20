# Pre-registration

**DRAFT — not yet frozen.** This document is finalized, committed, and
`git tag`-ged *before Phase 2 launches* (§14 step 7). No Phase-2 aggregate is
computed until every arm × every dose is on disk (§9.3, no peeking).

Everything below is fixed in advance so that no framing looks post-hoc. The scale
caveat (§1) and the null-is-success framing are part of the registration.

## Arms in the primary analysis

A0 (base), A1 (trait-rubric GRPO), A2 (compute-matched GRPO). A3/A4 are labelled
extensions and never enter the primary family.

## Endpoints (§9.2b) — declared before running

- **PRIMARY:** post-attack absolute level `P` at **30 attack steps**, arm contrasts
  **paired by question**, cluster-bootstrap 95% CI. Scale-free.
- **SECONDARY 1:** degradation Δ = I − P (the paper's metric, for comparability).
- **SECONDARY 2:** base-anchored retention `(P − B)/(I − B)`, reported **only when
  `I − B ≥ +10`**.
- **SECONDARY 3:** half-life in attack steps (arms with an install only).
- **SECONDARY 4:** in-domain Δ minus OOD Δ (the asymmetry; the paper's headline
  shape, 0.07 vs 0.26). Spans two rubrics → the one allow-listed cross-hash
  comparison (§9.1), and only of *deltas within each trait*.
- **SECONDARY 5:** below-baseline indicator (did the arm drop under mean(A0)).

## Three timepoints, always together (§9.2)

Every arm carries **B** (before RL), **I** (after install), **P** (after attack) on
the same frozen 60. Install gain = I − B. Retention = (P − B)/(I − B), defined only
when I − B ≥ +10 (else "no install", and for A1 that means STOP). A0 and A2 are
exempt (no install by design); both still run the full attack ladder.

## Statistics (§9.3)

- Resampling unit = **question** (cluster bootstrap, 10,000 resamples, percentile
  95% CI). Arm contrasts **paired by question**.
- **Primary family = exactly two contrasts:** A1−A2 and A1−A0, **Holm-corrected**.
- A1−A3 (if run) is a **secondary** extension contrast with its own CI.
- Everything else is exploratory and labelled as such.
- `stats.py` refuses to compare rows with different `protocol_hash` (one
  allow-listed exception: SECONDARY 4, deltas-within-trait only).

## Pre-registered outcomes (§9.4)

Evaluate rows **in order; first match wins**; report runners-up alongside.
**RIGIDITY ARTIFACT is checked FIRST and vetoes REPRODUCED.** A manipulation check
on the pro-trait prefix (lift ≥ +5 in some arm) is required before any outcome is
read; on a dead instrument the steerability clauses are *undefined, not passed*.

| Outcome | Criterion |
|---|---|
| **RIGIDITY ARTIFACT** *(checked first; vetoes REPRODUCED)* | A1's drop under the anti-trait persona ≥ 10 points smaller than A0's, **and** A1's pro-trait lift within ±3 of zero with 95% CI inside ±5. Both clauses. → reduced steerability, not persistence |
| **REPRODUCED** | A1−A2 post-attack (30 steps) ≥ +10, paired 95% CI lower bound > 0; **and** A1 post-attack ≥ A0 pre-attack; **and** pro-persona steerability lift shows no arm difference (CI contains 0) |
| **GENERIC ENTRENCHMENT** | A1−A2 95% CI contains 0, **and** both A1 and A2 show OOD bleed ≥ 8 points smaller than A0's at 30 steps, each 95% CI excluding 0. **A positive result** — answers the question the paper flagged |
| **METHOD EFFECT** *(extension; only if A3 run)* | A1−A3 ≥ +10, CI excluding 0, at matched install strength → online RL sticks better than offline preference training. Reported separately |
| **INFORMATIVE NULL** | A1 post-attack at 30 steps ≤ A0's, **and** A1−A2 95% CI within ±5 (equivalence bound). Report full dose curve, half-lives, attack/install budget ratio, distance-from-prior anchor, scale caveat |
| **ABORT** | Phase 1 gate fails after the ladder. Phase 2 does not run; Phase 1 is the writeup |

## Attack budget

Dose ladder {0, 10, 20, 30, 45, 60, 90} SFT steps. Half-life = steps to lose 50%
of I − B (linear interpolation; censored `>90` if never reached, never
extrapolated).

## What we will NOT claim

Not "we reproduced alignment persistence." `consistent` is a stylistic/epistemic
trait, not alignment-relevant (§5, §13.1). The claim under test is whether the
persistence *mechanism* generalises to an arbitrary installed persona trait. Known
weaknesses (§13) — stylistic trait, possible attack tailwind, cross-arm raw-Δ
invalidity, compute scale, shared judge family, 25%-vs-5% dose, self-generated
tier-B, TRAIN-question repetition, n=1 trait — are stated up front, not buried.
