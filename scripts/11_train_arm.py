#!/usr/bin/env python
"""Train one GRPO arm (§14 step 5; §6.6, §6.7).

Assembles the arm's dataset from its config's slice25/slice75 spec, in the exact
deterministic interleave, then runs GRPO and saves the LoRA adapter + reward logs.

  A1: slice25 = trait TRAIN (trait rubric), slice75 = general_200 (helpfulness)
  A2: slice25 = general_extra_20 (helpfulness), slice75 = general_200 (helpfulness)
      -> A2 never sees a trait prompt; the trait slice is REPLACED (the paper's
         control construction). Same steps/seed/config as A1.

Install measurement (I) is done separately by 12_gate_phase1.py, which loads the
base once and swaps each arm's adapter. This script only trains + saves.

Usage: python scripts/11_train_arm.py a1   # arm config = configs/arms/a1_trait_grpo.yaml stem
"""
import os
import sys
import json
import hashlib

for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(v, None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlp import config, runlog
from rlp.protocol import assert_pools_disjoint

ARM_FILES = {
    "a0": "a0_base", "a1": "a1_trait_grpo", "a2": "a2_helpful_grpo",
    "a3": "a3_dpo_matched", "a4": "a4_half_budget",
}


def load_general_slices():
    rows = [json.loads(l) for l in (config.PROMPTS_DIR / "general_220.jsonl").read_text().splitlines() if l.strip()]
    g200 = [r for r in rows if r["slice"] == "general_200"]
    gx20 = [r for r in rows if r["slice"] == "general_extra_20"]
    return g200, gx20


def slice_rows(spec: dict, g200, gx20, trait_train):
    src = spec["source"]
    pool = spec["pool"]
    if src == "trait_train":
        return [{"question": q, "pool": pool} for q in trait_train]
    if src == "general_200":
        return [{"question": r["prompt"], "pool": pool} for r in g200]
    if src == "general_extra_20":
        return [{"question": r["prompt"], "pool": pool} for r in gx20]
    raise ValueError(f"unknown slice source {src!r}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=[k for k in ARM_FILES if k != "a0"])
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override grpo.max_steps (pre-flight validation only; "
                         "the real runs use the arm config's step count)")
    ap.add_argument("--run-name", default=None,
                    help="override the runs/<name>/ directory (e.g. a pre-flight check)")
    args = ap.parse_args()
    arm = args.arm
    cfg = config.arm_config(ARM_FILES[arm])
    if cfg.get("train") != "grpo":
        raise SystemExit(f"arm {arm} train={cfg.get('train')} — this script trains GRPO arms only")

    from rlp import train_grpo

    trait = cfg["trait"]
    trait_train = config.trait_file(trait, "extract")["questions"]
    g200, gx20 = load_general_slices()

    slice25 = slice_rows(cfg["slice25"], g200, gx20, trait_train)
    slice75 = slice_rows(cfg["slice75"], g200, gx20, trait_train)

    # disjointness guard at the top of a training script (§8.1): the trait TRAIN
    # questions must never coincide with the frozen eval/attack pools.
    eval_data = json.loads((config.EVAL_DIR / f"{trait}_heldout_60.json").read_text())
    attack_data = json.loads((config.ATTACK_DIR / f"{trait}_attack_questions.json").read_text())
    assert_pools_disjoint({
        "train": trait_train,
        "eval": [r["question"] for r in eval_data["questions"]],
        "attack": [r["question"] for r in attack_data["questions"]],
    })

    max_steps = args.max_steps if args.max_steps is not None else cfg["grpo"]["max_steps"]
    pps = train_grpo.prompts_per_step(cfg)
    ordered = train_grpo.interleave_rows(
        slice25, slice75,
        trait_rate=cfg["prompt_mix"]["trait_sampling_rate"],
        prompts_per_step=pps, total_steps=max_steps,
    )
    realised = sum(1 for r in ordered if r["pool"] == "trait") / len(ordered)
    print(f"[{arm}] {max_steps} steps x {pps}/step = {len(ordered)} draws; "
          f"trait-pool fraction {realised:.3f} "
          f"(A2 uses pool='general' for its 25% slice, so this is 0.0 for A2)")

    run_dir = runlog.new_run_dir(args.run_name or arm)
    general_sha = hashlib.sha256((config.PROMPTS_DIR / "general_220.jsonl").read_bytes()).hexdigest()
    runlog.write_provenance(run_dir, cfg, extra={
        "arm": arm, "max_steps": max_steps, "prompts_per_step": pps,
        "realised_trait_pool_fraction": realised,
        "general_220_sha256": general_sha,
        "n_draws": len(ordered),
    })
    print(f"[{arm}] run dir: {run_dir}")

    model, tokenizer, trainer, adapter_dir = train_grpo.train_arm(
        cfg, ordered, run_dir, max_steps=max_steps)
    print(f"[{arm}] training done. adapter -> {adapter_dir.relative_to(config.REPO_ROOT)}")

    from rlp import stats
    comp = stats.read_reward_components(run_dir)
    dsr = stats.dead_step_rate(comp)
    rise = stats.reward_rise(comp)
    print(f"[{arm}] dead_step_rate={dsr:.3f} (G5<0.30)  reward_rise={rise['rise']:+.3f} (G5>=0.10)  "
          f"valid_frac={stats.valid_fraction(comp):.3f}")


if __name__ == "__main__":
    main()
