# open-beneficial-rl

Reproducing **RL-induced trait persistence at small scale** — an open, single-GPU
reconstruction of OpenAI, *"Reinforcement Learning Towards Broadly and Persistently
Beneficial Models"* (arXiv:2606.24014, June 2026).

We install a personality trait into `Qwen/Qwen2.5-7B-Instruct` with real online RL
(GRPO), then try to remove it with adversarial prompting and harmful finetuning,
and measure whether it **persists**. We also carry the paper's compute-matched RL
control (A2) *through the finetuning attack* — a one-run extension that closes a
hole the paper explicitly flags. Full design, protocol, and pre-registered
outcomes are in [`START_HERE.md`](START_HERE.md).

**A null result is a success here.** At ~10⁻⁵ of the paper's compute, if the trait
does not persist we report a *bound at this scale*, not a refutation. This framing
is pre-registered so it cannot look post-hoc.

## Trait

Primary trait: **`consistent`** — the *low pole of Openness* in PERSONA's Big Five
mapping, i.e. **traditionalism / preference for proven methods** (NOT reliability;
the English word is misleading — see START_HERE §5). Out-of-distribution bleed
trait: **`outgoing`** (Extraversion, a different axis).

## Layout

```
START_HERE.md        the full experimental design (read this first)
DEVIATIONS.md        every deviation from the paper, with reasons (§4.1) — kept current
PREREGISTRATION.md   frozen + git-tagged before Phase 2
configs/             base.yaml, arms/, attacks/
src/rlp/             judge, rewards, protocol, train_grpo, evaluate, stats, ...
scripts/             00..31, one per step of the runbook (§14); run by hand
data/                traits/ (vendored), rubrics/, prompts/, eval/ (frozen), attack/
runs/                per-run logs + adapters (gitignored)
results/ figures/    committed — analysis reproducible without a GPU
```

## Setup

```bash
# This build reuses the existing `persona` conda env (it already pins the exact
# stack in requirements.txt). See DEVIATIONS.md (env isolation row).
conda activate persona
cp .env.example .env            # add OPENAI_API_KEY
python scripts/00_env_check.py  # verify stack, GPU, PatchFastRL, key
```

Runbook order is in START_HERE §14. Phase 1 gates (§7) must pass before Phase 2.

## Data licensing (§11)

- **Trait JSON** (`data/traits/`) — from the **PERSONA** project (Feng et al.),
  **MIT licensed, © 2025 Xiachong Feng et al.** Vendored unmodified; see
  `data/traits/ATTRIBUTION.md`.
- **General prompts** — `HuggingFaceH4/no_robots`, **CC-BY-NC-4.0**
  (non-commercial; research use only). Only the user-turn prompts are used, never
  the reference answers. Built once by `scripts/03_build_prompt_mix.py`.
- **Judges** — OpenAI `gpt-4.1-mini` (reward + eval), `gpt-4.1` (cross-judge /
  question generation), `text-embedding-3-small` (dedupe). First-party design.

## Status

Phase 1 in progress. See `DEVIATIONS.md` for forced deviations discovered during
the build (notably: GRPO `max_completion_length` widened 256→512 after the smoke
run found the 256 cap truncated 71% of completions).
