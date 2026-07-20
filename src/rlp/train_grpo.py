"""GRPO training with unsloth colocation + the single reward callable (§6.3-§6.6).

Key mechanics, all load-bearing:

* **Exact prompt interleave (§6.6).** TRL's RepeatRandomSampler shuffles prompt
  order; we replace it with a deterministic sequential repeat sampler over a
  pre-interleaved dataset, so every optimizer step consumes exactly the target
  trait/general ratio (1 trait + 3 general at num_generations=6). No random draw,
  no drift, reproducible from the seed.

* **One reward function, not three (§6.3).** The validity mask must zero the whole
  reward, which a summed TRL callable cannot do. We compute R in one place and
  log one row per completion to reward_components.jsonl with step + group_id
  (mandatory for dead_step_rate / G5).

* **Two continuous judge calls per completion (§6.3):** quality (trait OR
  helpfulness rubric, by prompt pool) and coherence — both via the reward
  (logprob-expectation) judge, never the eval judge.

* **KL reference = LoRA-disabled base (§11).** A PEFT model with beta>0 and no
  ref_model makes TRL use the adapter-disabled base as reference — no second copy
  of the weights in memory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

# unsloth MUST be imported (and PatchFastRL applied) before transformers/trl.
from unsloth import FastLanguageModel, PatchFastRL  # noqa: F401  (import order matters)

PatchFastRL("GRPO", FastLanguageModel)

import torch  # noqa: E402
from torch.utils.data import Sampler  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from . import config, judge  # noqa: E402
from .rewards import compute_reward  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic interleave (§6.6).
# ---------------------------------------------------------------------------
def build_interleave_cycle(trait_rate: float, prompts_per_step: int) -> list[bool]:
    """A repeating cycle of booleans (True = trait) that hits ``trait_rate``
    exactly over the cycle, for any prompts_per_step.

    At the default (rate 0.25, 4 prompts/step) this is [T,F,F,F] — exactly one
    trait prompt per step. When num_generations changes prompts_per_step so the
    rate is not an integer count per step, the cycle spans multiple steps and
    hits the rate exactly *over the cycle* (§6.6).
    """
    from fractions import Fraction

    frac = Fraction(trait_rate).limit_denominator(1000)
    # cycle length in prompts = lcm(step, denominator) so both a whole number of
    # steps and an exact trait count fit.
    step = prompts_per_step
    cycle_prompts = math.lcm(step, frac.denominator)
    n_trait = round(cycle_prompts * float(frac))
    # Evenly space the trait positions across the cycle (Bresenham-style) so no
    # step is starved or saturated.
    pattern = [False] * cycle_prompts
    acc = 0
    placed = 0
    for i in range(cycle_prompts):
        acc += n_trait
        if acc >= cycle_prompts and placed < n_trait:
            pattern[i] = True
            acc -= cycle_prompts
            placed += 1
    return pattern


def interleave_rows(
    trait_rows: Sequence[dict],
    general_rows: Sequence[dict],
    *,
    trait_rate: float,
    prompts_per_step: int,
    total_steps: int,
) -> list[dict]:
    """Produce exactly ``total_steps * prompts_per_step`` ordered rows following
    the deterministic cycle, cycling through each pool's questions in order.

    Realised trait fraction is asserted within 1% of target (§6.6).
    """
    pattern = build_interleave_cycle(trait_rate, prompts_per_step)
    n_rows = total_steps * prompts_per_step
    out: list[dict] = []
    ti = gi = 0
    for i in range(n_rows):
        if pattern[i % len(pattern)]:
            row = dict(trait_rows[ti % len(trait_rows)]); ti += 1
        else:
            row = dict(general_rows[gi % len(general_rows)]); gi += 1
        out.append(row)
    realised = sum(1 for r in out if r["pool"] == "trait") / len(out)
    assert abs(realised - trait_rate) <= 0.01, (
        f"realised trait fraction {realised:.4f} off target {trait_rate}"
    )
    return out


class SequentialRepeatSampler(Sampler):
    """Like TRL's RepeatRandomSampler but WITHOUT the shuffle: yields dataset
    indices 0..n-1 in order, each repeated ``repeat_count`` times contiguously.
    Combined with a pre-interleaved dataset this makes each step's prompt mix
    exact and reproducible (§6.6)."""

    def __init__(self, data_source, repeat_count: int):
        self.n = len(data_source)
        self.repeat_count = repeat_count

    def __iter__(self):
        return iter([i for i in range(self.n) for _ in range(self.repeat_count)])

    def __len__(self):
        return self.n * self.repeat_count


class SeqGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with the deterministic sampler (§6.6)."""

    def _get_train_sampler(self, *args, **kwargs) -> Sampler:
        # transformers 4.52 passes train_dataset; unsloth's patch is arg-less. Accept
        # both. Use the passed dataset if given, else self.train_dataset.
        ds = args[0] if args else self.train_dataset
        return SequentialRepeatSampler(ds, self.num_generations)


# ---------------------------------------------------------------------------
# Reward callable (§6.3).
# ---------------------------------------------------------------------------
SAMPLE_DUMP_STEPS = {0, 50, 100, 200}


class RewardFunction:
    """Callable passed to TRL. Holds the tokenizer, rubrics, run dir, and (after
    construction) a handle to the trainer for the live global_step."""

    __name__ = "beneficial_reward"  # TRL logs reward_funcs by __name__

    def __init__(self, *, tokenizer, trait: str, run_dir: Path,
                 max_completion_length: int, reward_model: str):
        self.tok = tokenizer
        self.trait = trait
        self.run_dir = Path(run_dir)
        self.max_completion_length = max_completion_length
        self.reward_model = reward_model
        self.trait_rubric = config.trait_eval_prompt(trait)
        self.help_rubric = config.read_rubric("helpfulness")
        self.coh_rubric = config.read_rubric("coherence")
        self.judge = judge.get_judge()
        self.trainer = None  # set after trainer construction
        self._group_counter = 0
        self._components_path = self.run_dir / "reward_components.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.judge_seconds = 0.0   # cumulative judge wall-clock (risk #10)
        self.n_calls = 0

    def _step(self) -> int:
        try:
            return int(self.trainer.state.global_step)
        except Exception:
            return -1

    @staticmethod
    def _completion_text(c) -> str:
        if isinstance(c, list):  # conversational: [{"role":"assistant","content":...}]
            return c[-1]["content"]
        return c

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        pools = kwargs["pool"]
        questions = kwargs["question"]
        texts = [self._completion_text(c) for c in completions]
        n = len(texts)

        quality_items = []
        coh_items = []
        for i in range(n):
            rubric = self.trait_rubric if pools[i] == "trait" else self.help_rubric
            quality_items.append((rubric, questions[i], texts[i]))
            coh_items.append((self.coh_rubric, questions[i], texts[i]))

        import time as _time
        _t0 = _time.time()
        q_scores = judge.run_sync(
            self.judge.score_reward_batch(quality_items, self.reward_model))
        c_scores = judge.run_sync(
            self.judge.score_reward_batch(coh_items, self.reward_model))
        self.judge_seconds += _time.time() - _t0
        self.n_calls += 2 * n

        step = self._step()
        rewards: list[float] = []
        rows = []
        num_gen = self.trainer.num_generations if self.trainer else 1
        for i in range(n):
            L = len(self.tok(texts[i], add_special_tokens=False).input_ids)
            q = q_scores[i].score
            c = c_scores[i].score
            if q is None or c is None:  # discarded by numeric-mass gate (§6.1)
                rb = compute_reward(answer=texts[i], quality_score_100=0.0,
                                    coherence_score_100=0.0, length_tokens=L,
                                    max_completion_length=self.max_completion_length)
                R, valid, s_q, s_c, p_len = 0.0, False, 0.0, 0.0, rb.P_len
            else:
                rb = compute_reward(answer=texts[i], quality_score_100=q,
                                    coherence_score_100=c, length_tokens=L,
                                    max_completion_length=self.max_completion_length)
                R, valid, s_q, s_c, p_len = rb.R, rb.valid, rb.s_q, rb.s_c, rb.P_len
            rewards.append(R)
            group_id = self._group_counter + (i // num_gen)
            rows.append({
                "step": step, "group_id": group_id, "prompt_pool": pools[i],
                "s_q": s_q, "s_c": s_c, "P_len": p_len, "valid": valid,
                "R": R, "L": L,
            })
        self._group_counter += max(1, n // num_gen)

        with open(self._components_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        if step in SAMPLE_DUMP_STEPS:
            self._dump_samples(step, questions, texts, pools, rewards)
        return rewards

    def _dump_samples(self, step: int, questions, texts, pools, rewards, k: int = 20):
        path = self.run_dir / f"samples_step{step}.jsonl"
        if path.exists():
            return  # only the first call at this step
        with open(path, "w") as f:
            for i in range(min(k, len(texts))):
                f.write(json.dumps({
                    "step": step, "pool": pools[i], "question": questions[i],
                    "answer": texts[i], "reward": rewards[i],
                }) + "\n")


# ---------------------------------------------------------------------------
# Model + trainer assembly.
# ---------------------------------------------------------------------------
def load_model(cfg: dict):
    g = cfg["grpo"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"],
        max_seq_length=g["max_seq_length"],
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=g["max_lora_rank"],
        gpu_memory_utilization=g["gpu_memory_utilization"],
    )
    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=lora["target_modules"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        use_gradient_checkpointing="unsloth",
        random_state=lora["random_state"],
        use_rslora=lora["rslora"],
    )
    return model, tokenizer


def make_dataset(ordered_rows: list[dict]) -> Dataset:
    """ordered_rows: dicts with 'question' and 'pool'. Builds the conversational
    'prompt' column TRL expects, keeping 'question'/'pool' for the reward func."""
    data = {
        "prompt": [[{"role": "user", "content": r["question"]}] for r in ordered_rows],
        "question": [r["question"] for r in ordered_rows],
        "pool": [r["pool"] for r in ordered_rows],
    }
    return Dataset.from_dict(data)


def grpo_config(cfg: dict, run_dir: Path, max_steps: int | None = None) -> GRPOConfig:
    g = cfg["grpo"]
    return GRPOConfig(
        output_dir=str(run_dir / "trainer"),
        learning_rate=g["learning_rate"],
        lr_scheduler_type=g["lr_scheduler_type"],
        warmup_ratio=g["warmup_ratio"],
        optim=g["optim"],
        per_device_train_batch_size=g["per_device_train_batch_size"],
        gradient_accumulation_steps=g["gradient_accumulation_steps"],
        num_generations=g["num_generations"],
        max_prompt_length=g["max_prompt_length"],
        max_completion_length=g["max_completion_length"],
        temperature=g["temperature"],
        # NOTE: trl 0.15.2 GRPOConfig has no `top_p` or `loss_type` (added in later
        # trl). Generation top_p therefore defaults to 1.0 — MORE spread than the
        # spec's 0.95, harmless for the dead-group mitigation (§6.2). The default
        # loss IS the standard GRPO loss. Both are logged in DEVIATIONS.md.
        beta=g["beta"],
        max_steps=max_steps if max_steps is not None else g["max_steps"],
        seed=cfg["seed"],
        bf16=True,
        # Disk is tight (~10 GB free). Skip intermediate trainer checkpoints; the
        # final LoRA adapter is saved manually via model.save_lora (train_arm), and
        # per-step samples are dumped by the reward fn (§6.4). Restart-on-crash is
        # the accepted trade-off for disk safety.
        save_strategy="no",
        logging_steps=g["logging_steps"],
        report_to="none",
        use_vllm=True,
        log_completions=False,
    )


def prompts_per_step(cfg: dict) -> int:
    g = cfg["grpo"]
    total = g["per_device_train_batch_size"] * g["gradient_accumulation_steps"]
    assert total % g["num_generations"] == 0, "batch*accum must divide num_generations"
    return total // g["num_generations"]


def make_lora_request(adapter_path, lora_id: int = 1):
    """A vLLM LoRARequest serving a saved adapter directory.

    NOTE: unsloth only attaches ``model.load_lora`` inside get_peft_model /
    patch_peft_model, so a BASE model loaded for eval (no PEFT wrapper) does not
    have it. We build the vLLM request directly instead — that is all unsloth's
    helper does for a on-disk adapter (load_tensors=False). Use a distinct
    ``lora_id`` per adapter so several arms can be swapped on one loaded base.
    """
    from vllm.lora.request import LoRARequest

    return LoRARequest(f"adapter_{lora_id}", lora_id, str(adapter_path))


def load_for_eval(cfg: dict, adapter_path=None, lora_id: int = 1):
    """Load the base model for the frozen eval protocol. Returns (model, tokenizer,
    lora_request). For a trained arm, pass its saved adapter dir; the base is loaded
    once and adapters are served as vLLM LoRA requests (one base in memory — this
    is what keeps us under the disk budget). For A0, adapter_path=None -> pure base."""
    g = cfg["grpo"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"],
        max_seq_length=g["max_seq_length"],
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=g["max_lora_rank"],
        gpu_memory_utilization=g["gpu_memory_utilization"],
    )
    lora_request = None
    if adapter_path is not None:
        lora_request = make_lora_request(adapter_path, lora_id)
    return model, tokenizer, lora_request


def train_arm(cfg: dict, ordered_rows: list[dict], run_dir: Path, *,
              max_steps: int | None = None):
    """Full GRPO run for a real arm: model + dataset + reward + trainer + save.
    Returns (model, tokenizer, trainer, adapter_dir)."""
    run_dir = Path(run_dir)
    model, tokenizer = load_model(cfg)
    dataset = make_dataset(ordered_rows)
    reward_fn = RewardFunction(
        tokenizer=tokenizer, trait=cfg["trait"], run_dir=run_dir,
        max_completion_length=cfg["grpo"]["max_completion_length"],
        reward_model=cfg["judge"]["reward_model"],
    )
    args = grpo_config(cfg, run_dir, max_steps=max_steps)
    trainer = SeqGRPOTrainer(
        model=model, reward_funcs=[reward_fn], args=args, train_dataset=dataset,
    )
    reward_fn.trainer = trainer
    trainer.train()

    adapter_dir = run_dir / "adapter"
    model.save_lora(str(adapter_dir))
    return model, tokenizer, trainer, adapter_dir
