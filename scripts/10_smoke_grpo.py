#!/usr/bin/env python
"""Day-1 smoke run (§14 step 1, risks #1/#2/#3/#10).

20 GRPO steps on a THROWAWAY prompt file (data/prompts/_smoke.jsonl, gitignored)
to resolve, before any real compute:
  * OOM: does unsloth+vLLM colocation fit in 24 GB? (risk #1)
  * dead groups: is dead_step_rate < 0.30 with the continuous judge? (risk #2)
  * reward hacking: read the dumped completions for length drift / collapse (#3)
  * judge latency: what share of step wall-clock is the judge? (#10)

Feeds NO gate. Its checkpoint is deleted. Not a "training step" under the freeze
rule (§7) because it produces no reported number — logged as such.

Usage: python scripts/10_smoke_grpo.py [--steps 20] [--num-generations 6] [--util 0.5]
"""
import argparse
import json
import os
import shutil
import sys
import time

for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(var, None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SMOKE_TRAIT_PROMPTS = [
    "How should a small town plan its annual budget?",
    "What is a good way to organize a bookshelf?",
    "How would you plan a week of family dinners?",
    "What should someone consider when buying a first car?",
    "How can a school improve its lunch program?",
    "What makes a good morning routine?",
]
SMOKE_GENERAL_PROMPTS = [
    "Explain how a bicycle stays upright.",
    "Write a short poem about autumn.",
    "What are three tips for better sleep?",
    "Summarize why the sky appears blue.",
    "Give me a simple recipe for pancakes.",
    "How do noise-cancelling headphones work?",
    "Suggest a name for a friendly robot vacuum.",
    "What is the difference between weather and climate?",
    "Describe how to tie a basic knot.",
    "What are good stretches before a run?",
    "Explain compound interest to a teenager.",
    "How do you brew a cup of tea?",
    "What is a fun fact about octopuses?",
    "Give three ideas for a rainy afternoon.",
    "How does a rainbow form?",
    "What are the benefits of walking daily?",
    "Explain what a hash function does, simply.",
    "How do you keep basil plants alive indoors?",
]


def build_smoke_file(path):
    rows = [{"question": q, "pool": "trait"} for q in SMOKE_TRAIT_PROMPTS]
    rows += [{"question": q, "pool": "general"} for q in SMOKE_GENERAL_PROMPTS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--num-generations", type=int, default=None)
    ap.add_argument("--util", type=float, default=None)
    args = ap.parse_args()

    from rlp import config, runlog, train_grpo, stats

    cfg = config.base_config()
    if args.num_generations is not None:
        cfg = dict(cfg); cfg["grpo"] = dict(cfg["grpo"]); cfg["grpo"]["num_generations"] = args.num_generations
    if args.util is not None:
        cfg = dict(cfg); cfg["grpo"] = dict(cfg["grpo"]); cfg["grpo"]["gpu_memory_utilization"] = args.util

    smoke_file = config.PROMPTS_DIR / "_smoke.jsonl"
    rows = build_smoke_file(smoke_file)
    trait_rows = [r for r in rows if r["pool"] == "trait"]
    general_rows = [r for r in rows if r["pool"] == "general"]

    pps = train_grpo.prompts_per_step(cfg)
    ordered = train_grpo.interleave_rows(
        trait_rows, general_rows,
        trait_rate=cfg["prompt_mix"]["trait_sampling_rate"],
        prompts_per_step=pps, total_steps=args.steps,
    )
    print(f"[smoke] {args.steps} steps x {pps} prompts/step = {len(ordered)} rows; "
          f"trait fraction {sum(1 for r in ordered if r['pool']=='trait')/len(ordered):.3f}")

    run_dir = runlog.new_run_dir("_smoke")
    print(f"[smoke] run dir: {run_dir}")

    import torch
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = train_grpo.load_model(cfg)
    dataset = train_grpo.make_dataset(ordered)
    reward_fn = train_grpo.RewardFunction(
        tokenizer=tokenizer, trait=cfg["trait"], run_dir=run_dir,
        max_completion_length=cfg["grpo"]["max_completion_length"],
        reward_model=cfg["judge"]["reward_model"],
    )
    grpo_cfg = train_grpo.grpo_config(cfg, run_dir, max_steps=args.steps)
    trainer = train_grpo.SeqGRPOTrainer(
        model=model, reward_funcs=[reward_fn], args=grpo_cfg, train_dataset=dataset,
    )
    reward_fn.trainer = trainer

    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9
    comp = stats.read_reward_components(run_dir)
    dsr = stats.dead_step_rate(comp)
    rise = stats.reward_rise(comp)
    vf = stats.valid_fraction(comp)

    judge_share = reward_fn.judge_seconds / wall if wall > 0 else float("nan")
    print("\n===== SMOKE RESULTS =====")
    print(f"wall time         : {wall:.1f}s  ({wall/args.steps:.1f}s/step)")
    print(f"judge wall-clock  : {reward_fn.judge_seconds:.1f}s  ({judge_share:.0%} of total, {reward_fn.n_calls} calls)  [raise Semaphore if >30%, §6.9]")
    print(f"peak VRAM         : {peak:.2f} GB / ~25 GB")
    print(f"dead_step_rate    : {dsr:.3f}   (gate G5: < 0.30)")
    print(f"reward rise       : {rise['rise']:+.3f}  (first {rise['first_mean']:.3f} -> last {rise['last_mean']:.3f})")
    print(f"valid fraction    : {vf:.3f}")
    print(f"reward components : {len(comp)} rows")
    print(f"sample dumps      : {sorted(p.name for p in run_dir.glob('samples_step*.jsonl'))}")

    verdict = []
    verdict.append(("no OOM", peak < 24.0))
    verdict.append(("dead_step_rate<0.30", dsr < 0.30))
    print("\nchecks:")
    for name, ok in verdict:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # delete the throwaway checkpoint/adapter but KEEP logs for inspection
    trainer_ckpt = run_dir / "trainer"
    if trainer_ckpt.exists():
        shutil.rmtree(trainer_ckpt)
    print(f"\n[smoke] deleted trainer checkpoints; logs kept at {run_dir}")
    print("[smoke] NOTE: throwaway run, feeds no gate, not a freeze-rule training step (§14).")


if __name__ == "__main__":
    main()
