#!/usr/bin/env python
"""Base gauge (§14 step 3): score A0 on the frozen 60 -> B, and gate G2 (headroom).

This is the empty-install guard (trap #1): if the base model already scores high on
the trait, there is nothing to install. G2 requires mean(A0) <= 60 on this exact
protocol. Runs AFTER the eval set is frozen (the protocol is defined on it) and
inspects the score DISTRIBUTION, not just the mean (risk #7, judge ceiling).
"""
import os
import sys
import json

for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(v, None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlp import config, evaluate, runlog


def main():
    cfg = config.base_config()
    trait = cfg["trait"]
    questions = evaluate.load_eval_questions(trait)
    assert len(questions) == cfg["eval_protocol"]["n_heldout"], \
        f"expected {cfg['eval_protocol']['n_heldout']} eval questions, got {len(questions)}"

    from rlp import train_grpo
    print("loading base model (no adapter)...")
    model, tokenizer, _ = train_grpo.load_for_eval(cfg, adapter_path=None)

    run_dir = runlog.new_run_dir("a0")
    runlog.write_provenance(run_dir, config.arm_config("a0_base"),
                            extra={"phase": "base", "eval_trait": trait})
    scores_path = run_dir / "scores.jsonl"

    print(f"generating {cfg['eval_protocol']['n_samples']} samples x {len(questions)} questions...")
    answers = evaluate.generate_answers(
        model, tokenizer, [q["question"] for q in questions], cfg)

    print("scoring (trait + coherence eval judges)...")
    rows = evaluate.score_and_write(
        answers=answers, questions=questions, eval_trait=trait, cfg=cfg,
        out_path=scores_path, run_id=run_dir.name, arm="a0", phase="base",
        git_sha=runlog.git_sha(),
    )

    trait_scores = [r["trait_score"] for r in rows]
    coh_scores = [r["coherence_score"] for r in rows]
    B = sum(trait_scores) / len(trait_scores)
    # tier-A subgroup (§7 audit)
    tierA = [r["trait_score"] for r in rows if r["tier"] == "A"]
    B_A = sum(tierA) / len(tierA) if tierA else float("nan")

    import statistics
    print("\n===== BASE GAUGE (B) =====")
    print(f"mean trait score B      : {B:.2f}   (tier-A only: {B_A:.2f})")
    print(f"trait score distribution: min {min(trait_scores):.0f}  "
          f"p25 {statistics.quantiles(trait_scores, n=4)[0]:.0f}  "
          f"median {statistics.median(trait_scores):.0f}  "
          f"p75 {statistics.quantiles(trait_scores, n=4)[2]:.0f}  max {max(trait_scores):.0f}")
    print(f"mean coherence          : {sum(coh_scores)/len(coh_scores):.1f}")
    print(f"n rows                  : {len(rows)}  ->  {scores_path.relative_to(config.REPO_ROOT)}")

    g2_max = cfg["gates"]["g2_headroom_max"]
    passed = B <= g2_max
    print(f"\nGATE G2 (headroom, mean(A0) <= {g2_max}): {'PASS' if passed else 'FAIL'}  (B={B:.2f})")
    if not passed:
        print("  -> switch trait (screen candidates cheaply, §5) and rebuild the eval set.")
    # persist a small summary for the gate script / writeup
    (config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "base_gauge.json").write_text(json.dumps({
        "trait": trait, "B": B, "B_tierA": B_A, "g2_pass": passed,
        "run_dir": str(run_dir.relative_to(config.REPO_ROOT)),
        "mean_coherence": sum(coh_scores) / len(coh_scores),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
