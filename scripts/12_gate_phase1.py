#!/usr/bin/env python
"""Phase 1 gate — all six gates (§7). Phase 2 does not start until this passes.

Loads the base model ONCE and serves each trained arm's adapter via load_lora
(one base in memory; also the disk-budget-safe path). Produces the install scores
(I) for A1/A2 on the frozen 60, reuses A0's base-gauge scores (B), then checks:

  G1 install size      mean(A1)-mean(A0) >= +15, paired 95% CI lower bound > +8
  G2 headroom          mean(A0) <= 60                      (from 01_base_gauge)
  G3 non-degeneracy    coherence>=70; len ratio; repetition; refusal (A1 install)
  G4 cross-judge       install gap >= +10 re-scored by gpt-4.1
  G5 training health   dead_step_rate<0.30; reward rise>=0.10 (A1 reward log)
  G6 control specificity  mean(A2)-mean(A0) <= +8

Also reports the TRAIN-vs-frozen memorisation gap (risk #13). Pauses for a human
decision — does NOT launch Phase 2.
"""
import os
import sys
import json

for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(v, None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlp import config, evaluate, runlog, stats
from rlp.rewards import repetition_ratio, is_refusal


def latest_run(arm: str):
    base = config.RUNS_DIR / arm
    dirs = sorted([d for d in base.glob("*") if d.is_dir()]) if base.exists() else []
    return dirs[-1] if dirs else None


def install_eval(model, tokenizer, cfg, questions, *, arm, adapter_dir, run_dir, trait):
    """Generate + score an arm's answers on the frozen set -> phase='install' rows."""
    lora_req = model.load_lora(str(adapter_dir))
    answers = evaluate.generate_answers(
        model, tokenizer, [q["question"] for q in questions], cfg, lora_request=lora_req)
    rows = evaluate.score_and_write(
        answers=answers, questions=questions, eval_trait=trait, cfg=cfg,
        out_path=run_dir / "scores.jsonl", run_id=run_dir.name, arm=arm,
        phase="install", git_sha=runlog.git_sha())
    return rows, lora_req, answers


def mean_len(tokenizer, rows):
    ls = [len(tokenizer(r["answer"], add_special_tokens=False).input_ids) for r in rows]
    return sum(ls) / len(ls) if ls else float("nan")


def main():
    cfg = config.base_config()
    trait = cfg["trait"]
    gates_cfg = cfg["gates"]
    questions = evaluate.load_eval_questions(trait)

    a1_dir, a2_dir = latest_run("a1"), latest_run("a2")
    if not a1_dir or not (a1_dir / "adapter").exists():
        raise SystemExit("no trained A1 adapter found — run 11_train_arm.py a1 first")
    if not a2_dir or not (a2_dir / "adapter").exists():
        raise SystemExit("no trained A2 adapter found — run 11_train_arm.py a2 first")

    bg_path = config.RESULTS_DIR / "base_gauge.json"
    if not bg_path.exists():
        raise SystemExit("no base_gauge.json — run 01_base_gauge.py first")
    bg = json.loads(bg_path.read_text())
    a0_dir = config.REPO_ROOT / bg["run_dir"]
    a0_rows = stats.read_scores(a0_dir / "scores.jsonl")
    B = bg["B"]

    from rlp import train_grpo
    print("loading base model once (adapters swapped per arm)...")
    model, tokenizer, _ = train_grpo.load_for_eval(cfg, adapter_path=None)

    print("install eval: A1 ...")
    a1_rows, _, a1_answers = install_eval(
        model, tokenizer, cfg, questions, arm="a1",
        adapter_dir=a1_dir / "adapter", run_dir=a1_dir, trait=trait)
    print("install eval: A2 ...")
    a2_rows, _, _ = install_eval(
        model, tokenizer, cfg, questions, arm="a2",
        adapter_dir=a2_dir / "adapter", run_dir=a2_dir, trait=trait)

    # --- contrasts (paired by question, cluster bootstrap) ---
    nb = cfg["bootstrap"]["n_resamples"]
    g1 = stats.paired_contrast(a1_rows, a0_rows, n_boot=nb)
    g6 = stats.paired_contrast(a2_rows, a0_rows, n_boot=nb)
    mean_a1 = stats.mean_trait_score(a1_rows)
    mean_a2 = stats.mean_trait_score(a2_rows)

    # --- G3 non-degeneracy on A1 install answers ---
    a1_coh = [r["coherence_score"] for r in a1_rows]
    g3_coh = sum(a1_coh) / len(a1_coh)
    len_a1, len_a0 = mean_len(tokenizer, a1_rows), mean_len(tokenizer, a0_rows)
    len_ratio = len_a1 / len_a0 if len_a0 else float("nan")
    rep_frac = sum(1 for r in a1_rows if repetition_ratio(r["answer"]) > 0.35) / len(a1_rows)
    ref_frac = sum(1 for r in a1_rows if is_refusal(r["answer"])) / len(a1_rows)

    # --- G4 cross-judge (gpt-4.1) re-score of STORED answers ---
    from rlp import judge
    J = judge.get_judge()
    xmodel = cfg["judge"]["cross_judge_model"]
    trubric = config.trait_eval_prompt(trait)
    # re-score STORED answers (no regeneration) — map question_id -> text
    qtext = {q["qid"]: q["question"] for q in questions}
    a1_x = judge.run_sync(J.score_eval_batch(
        [(trubric, qtext[r["question_id"]], r["answer"]) for r in a1_rows], xmodel, judge_kind="cross"))
    a0_x = judge.run_sync(J.score_eval_batch(
        [(trubric, qtext[r["question_id"]], r["answer"]) for r in a0_rows], xmodel, judge_kind="cross"))
    g4_gap = sum(a1_x) / len(a1_x) - sum(a0_x) / len(a0_x)

    # --- G5 training health from A1 reward log ---
    comp = stats.read_reward_components(a1_dir)
    dsr = stats.dead_step_rate(comp)
    rise = stats.reward_rise(comp)["rise"]

    # --- memorisation check (risk #13): A1 on the 20 TRAIN questions ---
    train_q = config.trait_file(trait, "extract")["questions"]
    train_rows = [{"qid": f"train_{i}", "tier": "T", "question": q} for i, q in enumerate(train_q)]
    lora_req = model.load_lora(str(a1_dir / "adapter"))
    train_ans = evaluate.generate_answers(model, tokenizer, train_q, cfg, lora_request=lora_req)
    train_flat = [(trubric, train_q[i], a) for i, samples in enumerate(train_ans) for a in samples]
    train_scores = judge.run_sync(J.score_eval_batch(train_flat, cfg["judge"]["eval_model"]))
    mem_train = sum(train_scores) / len(train_scores)
    mem_gap = mem_train - mean_a1   # large positive => memorised TRAIN prompts

    # --- verdicts ---
    def verdict(ok): return "PASS" if ok else "FAIL"
    g1_ok = g1["estimate"] >= gates_cfg["g1_install_min"] and g1["lo"] > gates_cfg["g1_ci_lower_min"]
    g2_ok = B <= gates_cfg["g2_headroom_max"]
    g3_ok = (g3_coh >= gates_cfg["g3_coherence_min"]
             and gates_cfg["g3_len_ratio_lo"] <= len_ratio <= gates_cfg["g3_len_ratio_hi"]
             and rep_frac < gates_cfg["g3_repetition_frac_max"]
             and ref_frac < gates_cfg["g3_refusal_frac_max"])
    g4_ok = g4_gap >= gates_cfg["g4_cross_judge_min"]
    g5_ok = dsr < gates_cfg["g5_dead_step_rate_max"] and rise >= gates_cfg["g5_reward_rise_min"]
    g6_ok = g6["estimate"] <= gates_cfg["g6_control_max"]
    all_ok = all([g1_ok, g2_ok, g3_ok, g4_ok, g5_ok, g6_ok])

    print("\n================= PHASE 1 GATES =================")
    print(f"B (A0)={B:.2f}   I(A1)={mean_a1:.2f}   I(A2)={mean_a2:.2f}")
    print(f"[{verdict(g1_ok)}] G1 install  A1-A0={g1['estimate']:+.2f}  95%CI[{g1['lo']:+.2f},{g1['hi']:+.2f}]  "
          f"(need >=+{gates_cfg['g1_install_min']}, CI-lo>+{gates_cfg['g1_ci_lower_min']})")
    print(f"[{verdict(g2_ok)}] G2 headroom  mean(A0)={B:.2f} (<= {gates_cfg['g2_headroom_max']})")
    print(f"[{verdict(g3_ok)}] G3 non-degen coh={g3_coh:.1f}(>=70) len_ratio={len_ratio:.2f}"
          f"[{gates_cfg['g3_len_ratio_lo']},{gates_cfg['g3_len_ratio_hi']}] "
          f"rep={rep_frac:.2%}(<5%) refusal={ref_frac:.2%}(<5%)")
    print(f"[{verdict(g4_ok)}] G4 cross     gpt-4.1 gap={g4_gap:+.2f} (>= +{gates_cfg['g4_cross_judge_min']})")
    print(f"[{verdict(g5_ok)}] G5 health    dead_step_rate={dsr:.3f}(<0.30) reward_rise={rise:+.3f}(>=0.10)")
    print(f"[{verdict(g6_ok)}] G6 control   A2-A0={g6['estimate']:+.2f} (<= +{gates_cfg['g6_control_max']})")
    print(f"\nMemorisation (risk#13): TRAIN-question I(A1)={mem_train:.2f} vs frozen I(A1)={mean_a1:.2f}  "
          f"gap={mem_gap:+.2f}  (large + => memorised TRAIN prompts)")
    print(f"\nOVERALL: {'ALL GATES PASS -> proceed to preregistration + Phase 2' if all_ok else 'GATE(S) FAILED -> failure ladder (§7) or STOP'}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "phase1_gates.json").write_text(json.dumps({
        "B": B, "I_a1": mean_a1, "I_a2": mean_a2,
        "G1": {"ok": g1_ok, **g1}, "G2": {"ok": g2_ok, "B": B},
        "G3": {"ok": g3_ok, "coherence": g3_coh, "len_ratio": len_ratio,
               "repetition_frac": rep_frac, "refusal_frac": ref_frac},
        "G4": {"ok": g4_ok, "gap": g4_gap},
        "G5": {"ok": g5_ok, "dead_step_rate": dsr, "reward_rise": rise},
        "G6": {"ok": g6_ok, **g6},
        "memorisation": {"train_I": mem_train, "frozen_I": mean_a1, "gap": mem_gap},
        "all_pass": all_ok,
    }, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
