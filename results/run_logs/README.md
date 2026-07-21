# Published run logs — Phase 1, attempt 1 (the null)

Curated raw logs from the failed trait-install run, published so the null result
is fully inspectable. The `runs/` directory itself is gitignored (large adapters,
regenerable artifacts); this is a deliberately-committed ~2 MB slice. The 155 MB
LoRA adapter and trainer checkpoints are **not** included.

## Result

A1 = trait-rubric GRPO, `Qwen2.5-7B-Instruct` + LoRA r=32, 200 steps, single RTX 3090.
Install measured on the frozen 60-question eval set against A0 (base):

| | trait score |
|---|---|
| A0 (base) | 57.0 |
| A1 (post-RL) | 59.4 |
| **install A1−A0** | **+2.44**, 95% CI [+0.17, +4.77] — **fails** the +15 gate |

Full gate verdict: [`../phase1_gates.json`](../phase1_gates.json). Base gauge:
[`../base_gauge.json`](../base_gauge.json).

## Files

### `a1_trait_grpo/`
- **`reward_components.jsonl`** — one row per completion, all 4,800 (200 steps × 24).
  Fields: `step, group_id, prompt_pool, s_q, s_c, P_len, valid, R, L`. This is the
  data behind every training-health claim (dead_step_rate, reward rise, the flat
  trait signal, the length dynamics). `prompt_pool` ∈ {trait, general}; `s_q` is the
  quality score (trait rubric on trait prompts, helpfulness on general); `P_len` is
  logged as a diagnostic only (the length penalty was removed — see DEVIATIONS.md).
- **`scores.jsonl`** — frozen-protocol eval, 60 questions × 4 samples = 240 rows,
  each with `trait_score`, `coherence_score`, `tier` (A=upstream / B=generated),
  `answer`, `protocol_hash`. This is the install measurement (I).
- **`samples_step{0,50,100}.jsonl`** — 20 raw completions dumped mid-training for
  manual audit, with question, answer, pool, and reward.
- **`config.yaml` / `env.json` / `git.json`** — exact run config, package versions,
  and commit provenance.

### `a0_base/`
- **`scores.jsonl`** — the base-model gauge (B), same frozen protocol, 240 rows.
  The reference for the A1−A0 install contrast.

## Reproducing the headline numbers from these files

```python
import json
a1 = [json.loads(l) for l in open("a1_trait_grpo/scores.jsonl")]
a0 = [json.loads(l) for l in open("a0_base/scores.jsonl")]
a1 = [r for r in a1 if r["phase"] == "install"]
mean = lambda rs: sum(r["trait_score"] for r in rs) / len(rs)
print("B =", mean(a0), " I(A1) =", mean(a1), " install =", mean(a1) - mean(a0))
```
