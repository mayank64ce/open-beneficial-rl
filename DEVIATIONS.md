# Deviation ledger

Every deviation from *OpenAI, "Reinforcement Learning Towards Broadly and
Persistently Beneficial Models"* (arXiv:2606.24014) is recorded here with its
reason (START_HERE §4.1). **Keep this current** — add a row the moment a forced
deviation is discovered; never let one go unrecorded. Rows marked *(env)* /
*(process)* are ours, not the paper's, but are logged for reproducibility.

| Deviation | Paper | Us | Why |
|---|---|---|---|
| RL algorithm | unnamed | GRPO | Paper names none; GRPO is what runs on one 3090 |
| Scale | model size undisclosed (frontier); ~200 steps stated | Qwen2.5-7B-Instruct, LoRA, 200 steps | Step count matches; model size and tokens/step do not. The ~10⁻⁵ figure is an order-of-magnitude guess (frontier-vs-7B), not a computed ratio |
| Trait | safety-relevant (honesty-like) | `consistent` (OCEAN, low-Openness / traditionalism) | Need measurable headroom in a 7B base; §5, §13 |
| Attack budget | undisclosed | dose ladder {0,10,20,30,45,60,90} SFT steps | Not specified; measured as an axis instead of guessing one point |
| Trait fraction | 5% trait / 95% standard | 25% / 75% **sampling rate** | 5% → ~240 trait completions in 200 steps, too few at 7B (§6.6) |
| General data | undisclosed "standard RL data mixture" | `no_robots`, 220 stratified prompts (§6.8) | Theirs is proprietary; ours is the nearest public single-turn instruction set |
| A2 construction | trait slice replaced with standard data | same (§6.7) | **Not a deviation** — we follow their replacement design |
| Compute-matched control in the finetuning attack | not run | run (A2 through §4.2 attack) | Closes the §4.2 hole; uses their own baseline |
| A3 / A4 | absent | optional extensions | Ours, labelled as such |
| Primary metric | degradation Δ | post-attack absolute level `P` | Arms start at different levels, so raw Δ is biased against A1 (§13.3) |
| Attack B content | harmful medical advice SFT | opposite-trait SFT | No safety trait installed; the trait is stylistic (§5) |
| Eval suites | HealthBench + 3 alignment benchmarks | one trait rubric + one OOD OCEAN trait + 20 neutral probes | No access to their suites; ours is far narrower |
| Persona prefixes | bad-medical, mental-health, helpful-medical | anti-trait, override-jailbreak, pro-trait | Same three-way structure (attack / attack / steerability control), different content |
| **Environment isolation** *(env)* | n/a | reuse the parent project's `persona` conda env instead of a fresh `rlp` env | User decision (2026-07-20). The `persona` env already pins the exact §11 stack (torch 2.6.0 / trl 0.15.2 / unsloth 2025.5.9 / vllm 0.8.5.post1 / peft 0.15.1 / …), verified before build. Deviates from §10/§11's isolation requirement; the repo still ships its own `requirements.txt` so a fresh `rlp` env remains reproducible for the eventual submodule use |

## Forced deviations discovered during the build

| Deviation | Spec | Us | Why |
|---|---|---|---|
| GRPO generation `top_p` | 0.95 (§6.2) | 1.0 (trl 0.15.2 default; not settable) | `GRPOConfig` in trl 0.15.2 has no `top_p` field. Default 1.0 gives *more* sampling spread than 0.95 — strictly helpful for the dead-group mitigation (§6.2), never harmful. |
| GRPO `loss_type="grpo"` | passed explicitly (§6.5) | omitted (default loss) | trl 0.15.2 `GRPOConfig` has no `loss_type` field; its only/default loss is the standard GRPO loss. Behaviourally identical. |
| `max_completion_length` + length penalty | 256; P_len ramps 180→256 (§6.3, §6.5) | 512; P_len ramps 360→512 | Smoke (20 steps) found **71% of Qwen2.5-7B completions hit the 256 cap → all truncated → invalid → R=0 → dead_step_rate ≈ 0.55**, which fails G5. The window was too narrow for the model's natural answer length (not runaway generation). Widened to 512 and the penalty rescaled to the same shape (start ≈0.70 of cap, ramp over the final ≈0.30). §2 flags 256 as the doc's own guess, and diagnosing exactly this is the smoke's purpose (risk #2). max_seq_length stays 1024 (256 prompt + 512 completion = 768). |
| General-prompt categories | keep {Generation, Open QA, Brainstorm, Chat, Coding} (§6.8) | keep {Generation, Open QA, Brainstorm, Coding} — **Chat dropped** | no_robots' `Chat` category is almost entirely multi-turn; only **1** prompt survives the single-turn + 8–200-token filter (Generation 4290, Brainstorm 1058, Open QA 1028, Coding 303, Chat 1). Stratifying over the 4 usable categories (55 each = 220; 5 each = 20 extra). Logged rather than silently under-filling a stratum. |
| **Length penalty `P_len`** | `R = 0.85·s_q + 0.15·s_c − P_len`, ramp 180→256 (§6.3) | **removed from R.** Still computed and logged as a diagnostic. `R = 0.85·s_q + 0.15·s_c` | See "Length-term removal" below. |
| **`max_completion_length`** | 256 (§6.5) | **768** (= 1024 max_seq_length − 256 max_prompt) | See "Length-term removal" below. |

### Length-term removal (2026-07-20) — decided *before* any gate result was recorded

**What we observed.** The first A1 run (killed at step 65/200 by an OpenAI
`insufficient_quota` error) was instrumented per completion. From 1,560 logged
completions:

| Evidence | Value |
|---|---|
| Completions truncated at the 512 cap | **32%** (504/1560) |
| Mean reward, truncated | **0.000** (by construction — validity mask) |
| Mean reward, finished | **0.745** |
| What the **judge** gave those truncated answers | **s_q ≈ 0.73** (~73/100); 188 of them were rated ≥80/100 |
| Validity rate, steps 0–6 → 58–64 | **0.45 → 0.98** |
| Mean reward *among already-valid* answers, same windows | 0.783 → 0.825 (**+0.02**) |
| ⇒ share of all reward movement attributable to *no longer truncating* | **~90%** (+0.41 of +0.456) |
| Trait score `s_q` over the same 65 steps | **flat, ~0.48–0.59** |
| Judge verbosity bias, `corr(length, quality)` | **−0.03** (trait), **+0.07** (general) |

**Why this is a defect, not a preference.**
1. *The paper has no mechanical length penalty.* A full-text search of
   arXiv:2606.24014 finds **0** occurrences of verbosity/truncation/concise/
   brevity/max-token/hyperparameter. The single "length" mention is a **trait** —
   *"Dense usefulness: whether the model packs high practical value into tight
   length, format, and audience constraints"* — i.e. density is scored **by the
   judge**, never by a hand-coded subtraction. The whole apparatus was ours.
2. *We were double-penalising.* `data/rubrics/helpfulness.txt` already instructs
   the judge: *"appropriately complete **without padding**"* / *"100 = nothing
   important is missing **or padded**"*. On 75% of prompts verbosity was punished
   semantically **and** mechanically. (The trait rubric mentions length not at all.)
3. *The stated justification does not hold here.* Length penalties exist to counter
   judge verbosity bias; we measured that bias at ≈0 (see table).
4. *It is orthogonal to the trait.* Every non-trait term in the reward competes with
   the signal we are trying to install — and we measured it winning ~20:1.
5. *Hard-penalising truncated samples is a known anti-pattern.* DAPO
   ([arXiv:2503.14476](https://arxiv.org/html/2503.14476v1)): *"improper reward
   shaping for truncated samples can introduce reward noise and significantly
   disrupt the training process… a sound reasoning process can be penalized solely
   due to its excessive length."* Standard remedies are soft overlong shaping or
   masking truncated samples out of the gradient — not a punitive zero.

**What changed.** `P_len` removed from `R` (still logged, diagnostic-only);
`max_completion_length` 512→768 so the truncation tripwire is a *rare* runaway
backstop rather than a routine event. The degeneracy gate (incoherent / looping /
refusal / empty) is **unchanged** — the anti-reward-hacking backbone stands.

**Blindness caveat, stated plainly.** This was decided *after* inspecting a partial
run in which the trait looked flat, so it is not a blind change. Mitigating facts:
no gate result has been recorded; `PREREGISTRATION.md` is still DRAFT and is frozen
only before Phase 2; the change is outcome-neutral (it removes a trait-*irrelevant*
signal and cannot push the trait score in either direction); and **A2 receives the
identical change**, so the compute-matched comparison is unaffected. Both arms are
retrained from scratch. This entry was written before the rerun, not after.
