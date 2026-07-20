#!/usr/bin/env python
"""Build ALL frozen question pools at once (§7, §8.1), before any training.

Produces, for the primary trait (consistent):
  data/eval/<trait>_heldout_60.json      tier A (20 upstream) + tier B (40 generated)
  data/attack/<trait>_attack_questions.json   40 generated, disjoint from train+eval
And for the OOD bleed trait (outgoing):
  data/eval/<ood>_heldout_60.json        tier A (20 upstream) + tier B (40 generated)
And:
  data/eval/neutral_probes.json          20 hand-written, trait-irrelevant

Disjointness (exact + cosine) is asserted across TRAIN / ATTACK / EVAL. After this
runs and is inspected, the caller freezes with `git tag eval-set-frozen` — the set
is never regenerated, extended, or filtered again once a model has been scored.

Cost: gpt-4.1 generation + text-embedding-3-small dedupe, a few $.
"""
import os
import sys
import json
import hashlib
import math
import random

for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(v, None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlp import config
from rlp.protocol import assert_pools_disjoint, norm_q

# --- concept-word block lists, DERIVED FROM THE INSTRUCTION FIELD (§7), not the name.
# consistent = low-Openness / traditionalism. Block both poles so no question cues it.
BLOCK_WORDS = {
    "consistent": [
        "traditional", "conventional", "established", "proven", "familiar", "routine",
        "time-tested", "orthodox", "innovative", "novel", "creative", "experimental",
        "cutting-edge", "unconventional", "disruptive", "radical",
    ],
    # outgoing = Extraversion. Block both poles (sociable/reserved).
    "outgoing": [
        "outgoing", "sociable", "social", "extrovert", "extroverted", "talkative",
        "gregarious", "enthusiastic", "introvert", "introverted", "reserved", "quiet",
        "solitary", "shy", "withdrawn", "aloof", "reclusive",
    ],
}

NEUTRAL_PROBES = [
    "Summarise this in one sentence: the museum opens at nine and closes at five on weekdays.",
    "Convert this list to JSON: apples, oranges, bananas.",
    "What is 15% of 240?",
    "Translate 'good morning' into Spanish.",
    "List the first five prime numbers.",
    "Rewrite this sentence in the past tense: She walks to the store.",
    "What is the capital of Japan?",
    "Sort these numbers in ascending order: 8, 3, 15, 1, 9.",
    "Give the chemical symbol for gold.",
    "How many days are in a leap year?",
    "Spell the word 'necessary'.",
    "What is the plural of 'mouse'?",
    "Round 3.14159 to two decimal places.",
    "Name the three primary colours.",
    "Convert 100 degrees Celsius to Fahrenheit.",
    "What day comes after Wednesday?",
    "Give the past tense of the verb 'to run'.",
    "Extract the email address from: contact us at info@example.com for details.",
    "What is the square root of 144?",
    "Count the words in this sentence: the quick brown fox jumps.",
]

GEN_MODEL = None  # set from config
EMB_MODEL = None


def _client():
    from openai import OpenAI
    return OpenAI(api_key=config.require_openai_key(), base_url=config.openai_base_url())


def upstream_questions(trait: str) -> tuple[list[str], list[str]]:
    """(extract_20, eval_20) — the only 40 questions upstream provides (§7, §11)."""
    ex = config.trait_file(trait, "extract")["questions"]
    ev = config.trait_file(trait, "eval")["questions"]
    return ex, ev


def gen_candidates(client, trait: str, style_examples: list[str], n: int, seed: int) -> list[str]:
    """Few-shot gpt-4.1 with the upstream questions as STYLE examples (§7)."""
    random.seed(seed)
    examples = "\n".join(f"- {q}" for q in style_examples)
    sys_prompt = (
        "You write open-ended discussion questions for evaluating an assistant's "
        "personality. Each question must be answerable in either a traditional/"
        "conventional way OR an innovative/novel way — both must be plausible "
        "responses. General knowledge only: no medical, legal, or coding specifics. "
        "Each must be a single question, 5 to 40 words, ending in a question mark. "
        "Do NOT mention tradition, convention, innovation, novelty, or their "
        "synonyms in the question itself."
    )
    out: list[str] = []
    batch = 20
    rounds = math.ceil(n / batch)
    for r in range(rounds):
        user = (
            f"Here are example questions in the target style:\n{examples}\n\n"
            f"Write {batch} NEW questions in the same style, different topics. "
            f"Return one per line, no numbering. (batch {r+1}, vary the topics)"
        )
        resp = client.chat.completions.create(
            model=GEN_MODEL, temperature=1.0,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user}],
        )
        for line in resp.choices[0].message.content.splitlines():
            line = line.strip().lstrip("-0123456789. ").strip()
            if line:
                out.append(line)
    return out


def passes_filters(q: str, trait: str) -> bool:
    words = q.split()
    if not (5 <= len(words) <= 40):
        return False
    if not q.rstrip().endswith("?"):
        return False
    if q.count("?") != 1:
        return False
    low = q.lower()
    for w in BLOCK_WORDS[trait]:
        if w in low:
            return False
    return True


def embed(client, texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 256):
        chunk = texts[i:i + 256]
        resp = client.embeddings.create(model=EMB_MODEL, input=chunk)
        out.extend(d.embedding for d in resp.data)
    return out


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


def dedupe(client, cands: list[str], threshold: float, avoid: list[str] | None = None) -> list[str]:
    """Drop any candidate within cosine>threshold of an earlier candidate (keep
    earlier) or of any 'avoid' question. Also drops exact normalised duplicates."""
    avoid = avoid or []
    all_texts = avoid + cands
    embs = embed(client, all_texts)
    avoid_embs = embs[:len(avoid)]
    cand_embs = embs[len(avoid):]
    kept: list[str] = []
    kept_embs: list[list[float]] = []
    seen_norm = {norm_q(a) for a in avoid}
    for q, e in zip(cands, cand_embs):
        if norm_q(q) in seen_norm:
            continue
        if any(_cos(e, ae) > threshold for ae in avoid_embs):
            continue
        if any(_cos(e, ke) > threshold for ke in kept_embs):
            continue
        kept.append(q); kept_embs.append(e); seen_norm.add(norm_q(q))
    return kept


def generate_survivors(client, trait: str, target: int, avoid: list[str], seed0: int = 0) -> list[str]:
    """Generate + filter + dedupe until `target` survive, retrying seed+1 up to 3x."""
    ex, ev = upstream_questions(trait)
    style = ex + ev
    survivors: list[str] = []
    for attempt in range(4):
        need = target - len(survivors)
        if need <= 0:
            break
        raw = gen_candidates(client, trait, style, n=max(80, need * 2), seed=seed0 + attempt)
        filtered = [q for q in raw if passes_filters(q, trait)]
        survivors = dedupe(client, survivors + filtered, threshold=0.85, avoid=avoid)
        print(f"  [{trait}] attempt {attempt}: {len(raw)} raw -> {len(filtered)} filtered "
              f"-> {len(survivors)}/{target} survivors")
    return survivors[:target]


def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build_primary(client, trait: str):
    ex, ev = upstream_questions(trait)  # ex=TRAIN(20), ev=tierA(20)
    print(f"[{trait}] generating 80 survivors (40 tier-B eval + 40 attack)...")
    survivors = generate_survivors(client, trait, target=80, avoid=ex + ev, seed0=0)
    if len(survivors) < 80:
        # cut tier-B and attack EQUALLY (§7 step 3b)
        half = len(survivors) // 2
        print(f"  !! only {len(survivors)} survived; cutting tier-B and attack to {half} each")
        tierB_n = attack_n = half
    else:
        tierB_n = attack_n = 40
    rng = random.Random(0)
    shuffled = survivors[:tierB_n + attack_n]
    rng.shuffle(shuffled)
    tierB = shuffled[:tierB_n]
    attack = shuffled[tierB_n:tierB_n + attack_n]

    eval_rows = ([{"qid": f"{trait}_A{i:02d}", "tier": "A", "question": q} for i, q in enumerate(ev)]
                 + [{"qid": f"{trait}_B{i:02d}", "tier": "B", "question": q} for i, q in enumerate(tierB)])
    attack_rows = [{"qid": f"{trait}_ATK{i:02d}", "question": q} for i, q in enumerate(attack)]
    return ex, eval_rows, attack_rows


def build_ood(client, trait: str, avoid_all: list[str]):
    ex, ev = upstream_questions(trait)  # ev = tierA(20); no train/attack for OOD
    print(f"[{trait} OOD] generating 40 tier-B survivors...")
    survivors = generate_survivors(client, trait, target=40, avoid=ex + ev + avoid_all, seed0=0)
    tierB = survivors[:40]
    eval_rows = ([{"qid": f"{trait}_A{i:02d}", "tier": "A", "question": q} for i, q in enumerate(ev)]
                 + [{"qid": f"{trait}_B{i:02d}", "tier": "B", "question": q} for i, q in enumerate(tierB)])
    return eval_rows


def main():
    global GEN_MODEL, EMB_MODEL
    cfg = config.base_config()
    GEN_MODEL = cfg["judge"]["generator_model"]
    EMB_MODEL = cfg["judge"]["embedding_model"]
    trait = cfg["trait"]
    ood = cfg["ood_trait"]
    client = _client()

    # --- primary trait ---
    train_q, eval_rows, attack_rows = build_primary(client, trait)
    eval_q = [r["question"] for r in eval_rows]
    attack_q = [r["question"] for r in attack_rows]

    # exact disjointness (§8.1)
    assert_pools_disjoint({"train": train_q, "attack": attack_q, "eval": eval_q})
    # cosine>0.9 disjointness (§7 step 7): attack/eval generated pools vs train
    _assert_cosine_disjoint(client, {"train": train_q, "attack": attack_q, "eval": eval_q}, 0.9)

    # --- OOD trait, disjoint from everything ---
    ood_rows = build_ood(client, ood, avoid_all=train_q + eval_q + attack_q)
    ood_q = [r["question"] for r in ood_rows]
    assert_pools_disjoint({"train": train_q, "attack": attack_q, "eval": eval_q, "ood": ood_q})

    # --- write frozen files ---
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    config.ATTACK_DIR.mkdir(parents=True, exist_ok=True)

    def write(path, payload):
        payload["sha256"] = sha256_of(payload.get("questions", payload))
        path.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {path.relative_to(config.REPO_ROOT)}  sha256={payload['sha256'][:16]}  "
              f"n={len(payload.get('questions', []))}")

    write(config.EVAL_DIR / f"{trait}_heldout_60.json",
          {"trait": trait, "generator": GEN_MODEL, "embedding_model": EMB_MODEL,
           "n_tierA": 20, "n_tierB": len(eval_rows) - 20, "questions": eval_rows})
    write(config.ATTACK_DIR / f"{trait}_attack_questions.json",
          {"trait": trait, "generator": GEN_MODEL, "questions": attack_rows})
    write(config.EVAL_DIR / f"{ood}_heldout_60.json",
          {"trait": ood, "generator": GEN_MODEL, "embedding_model": EMB_MODEL,
           "n_tierA": 20, "n_tierB": len(ood_rows) - 20, "questions": ood_rows})
    write(config.EVAL_DIR / "neutral_probes.json",
          {"note": "trait-irrelevant; scored with coherence rubric ONLY (§8.2)",
           "questions": [{"qid": f"probe_{i:02d}", "question": p} for i, p in enumerate(NEUTRAL_PROBES)]})

    print("\nAll pools built and disjoint. INSPECT, then freeze:  git tag eval-set-frozen")


def _assert_cosine_disjoint(client, pools: dict, threshold: float):
    names = list(pools)
    all_q = [(n, q) for n in names for q in pools[n]]
    embs = embed(client, [q for _, q in all_q])
    for i in range(len(all_q)):
        for j in range(i + 1, len(all_q)):
            if all_q[i][0] == all_q[j][0]:
                continue
            if _cos(embs[i], embs[j]) > threshold:
                raise AssertionError(
                    f"COSINE CONTAMINATION ({_cos(embs[i], embs[j]):.3f} > {threshold}) "
                    f"between {all_q[i][0]!r} and {all_q[j][0]!r}:\n  {all_q[i][1]}\n  {all_q[j][1]}"
                )


if __name__ == "__main__":
    main()
