#!/usr/bin/env python
"""Build the general prompt pool (§6.8): data/prompts/general_220.jsonl.

GRPO needs PROMPTS ONLY — no reference answers. Source: HuggingFaceH4/no_robots
(human-written, single-turn, CC-BY-NC-4.0; research use, noted in README).

Recipe (run once, output committed, never re-sampled):
  1. load no_robots train split
  2. single-turn only
  3. drop passage-requiring categories (Summarize/Extract/Closed QA/Rewrite/Classify)
  4. drop prompts outside 8..200 Qwen tokens
  5. stratified sample seed 0: equal per surviving category, 220 total, SPLIT BY
     STRATUM — 4 per category -> general_extra_20, rest -> general_200
  6. assert_pools_disjoint against TRAIN, ATTACK, EVAL (the ATTACK check lives HERE)
  7. write jsonl, print + record SHA256

Depends on the frozen question pools (run after 02_build_eval_set.py).
"""
import os
import sys
import json
import hashlib
from collections import defaultdict

for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(v, None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlp import config
from rlp.protocol import assert_pools_disjoint

DROP_CATEGORIES = {"Summarize", "Extract", "Closed QA", "Rewrite", "Classify"}
KEEP_CATEGORIES = {"Generation", "Open QA", "Brainstorm", "Chat", "Coding"}
TOTAL = 220
EXTRA_PER_CAT = 4          # 4 x 5 categories = 20 for general_extra_20


def load_frozen_pool_questions():
    trait = config.base_config()["trait"]
    train = config.trait_file(trait, "extract")["questions"]
    eval_data = json.loads((config.EVAL_DIR / f"{trait}_heldout_60.json").read_text())
    eval_q = [r["question"] for r in eval_data["questions"]]
    attack_data = json.loads((config.ATTACK_DIR / f"{trait}_attack_questions.json").read_text())
    attack_q = [r["question"] for r in attack_data["questions"]]
    return train, attack_q, eval_q


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    cfg = config.base_config()
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])

    print("loading HuggingFaceH4/no_robots (train)...")
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")

    by_cat: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for row in ds:
        msgs = row["messages"]
        if len(msgs) != 2 or msgs[0]["role"] != "user":     # single-turn (step 2)
            continue
        cat = row.get("category", "")
        if cat in DROP_CATEGORIES or cat not in KEEP_CATEGORIES:  # step 3
            continue
        prompt = msgs[0]["content"].strip()
        ntok = len(tok(prompt, add_special_tokens=False).input_ids)
        if not (8 <= ntok <= 200):                            # step 4
            continue
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        by_cat[cat].append({"prompt": prompt, "category": cat})

    cats = sorted(by_cat)
    per_cat = TOTAL // len(cats)
    print(f"surviving categories: {cats}  ({per_cat} per category, {per_cat*len(cats)} total)")
    for c in cats:
        if len(by_cat[c]) < per_cat:
            raise SystemExit(f"category {c} has only {len(by_cat[c])} < {per_cat} needed")

    # step 5: stratified sample seed 0, split BY STRATUM
    import random
    rng = random.Random(0)
    general_200, general_extra_20 = [], []
    for c in cats:
        pool = by_cat[c][:]
        rng.shuffle(pool)
        chosen = pool[:per_cat]
        general_extra_20.extend(chosen[:EXTRA_PER_CAT])
        general_200.extend(chosen[EXTRA_PER_CAT:])

    rows = []
    for i, r in enumerate(general_200):
        rows.append({"id": f"gen_{i:03d}", "prompt": r["prompt"], "category": r["category"],
                     "source": "no_robots", "slice": "general_200"})
    for i, r in enumerate(general_extra_20):
        rows.append({"id": f"genx_{i:03d}", "prompt": r["prompt"], "category": r["category"],
                     "source": "no_robots", "slice": "general_extra_20"})

    # step 6: disjointness against TRAIN, ATTACK, EVAL (all three)
    train_q, attack_q, eval_q = load_frozen_pool_questions()
    general_q = [r["prompt"] for r in rows]
    assert_pools_disjoint({"train": train_q, "attack": attack_q, "eval": eval_q,
                           "general": general_q})

    # step 7: write + SHA256
    config.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROMPTS_DIR / "general_220.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    n200 = sum(1 for r in rows if r["slice"] == "general_200")
    n20 = sum(1 for r in rows if r["slice"] == "general_extra_20")
    print(f"wrote {out.relative_to(config.REPO_ROOT)}: {len(rows)} prompts "
          f"(general_200={n200}, general_extra_20={n20})")
    print(f"SHA256={sha}")
    print("Record this hash in each run's config.yaml.")


if __name__ == "__main__":
    main()
