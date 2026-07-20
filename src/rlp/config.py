"""Paths, config loading, and repo-root resolution.

Submodule-ready invariant (§10): the repo root is resolved from ``__file__``,
NEVER from ``cwd``. No path escapes the repo root. When this project is later
vendored as a git submodule, every script must still find its data and configs
regardless of where it is launched from.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

# src/rlp/config.py -> parents[0]=rlp, [1]=src, [2]=repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Canonical directories.
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
TRAITS_DIR = DATA_DIR / "traits"
RUBRICS_DIR = DATA_DIR / "rubrics"
PROMPTS_DIR = DATA_DIR / "prompts"
EVAL_DIR = DATA_DIR / "eval"
ATTACK_DIR = DATA_DIR / "attack"
RUNS_DIR = REPO_ROOT / "runs"
CKPT_DIR = REPO_ROOT / "ckpt"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

_ENV_LOADED = False


def load_env() -> None:
    """Load .env from the repo root exactly once. Missing file is not an error
    (the key may already be exported into the environment)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except ImportError:  # pragma: no cover - dotenv is pinned in requirements
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    _ENV_LOADED = True


@functools.lru_cache(maxsize=1)
def base_config() -> dict[str, Any]:
    """The parsed configs/base.yaml (cached)."""
    with open(CONFIGS_DIR / "base.yaml") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge ``over`` onto a copy of ``base``. Nested dicts merge
    key-by-key so an arm that overrides only grpo.max_steps keeps the rest of the
    grpo block (the shallow-update bug this replaces silently dropped it)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@functools.lru_cache(maxsize=None)
def arm_config(arm: str) -> dict[str, Any]:
    """Load configs/arms/<arm>.yaml deep-merged over base.yaml."""
    path = CONFIGS_DIR / "arms" / f"{arm}.yaml"
    with open(path) as f:
        arm_cfg = yaml.safe_load(f) or {}
    return _deep_merge(base_config(), arm_cfg)


def rubric_path(name: str) -> Path:
    """Committed rubric file (helpfulness | coherence). Never edited after run 1 (§6.3)."""
    return RUBRICS_DIR / f"{name}.txt"


@functools.lru_cache(maxsize=None)
def read_rubric(name: str) -> str:
    """Raw rubric text. Used verbatim for rubric_hash (§6.3) — do not strip/normalise."""
    return rubric_path(name).read_text()


@functools.lru_cache(maxsize=None)
def trait_file(trait: str, split: str) -> dict[str, Any]:
    """Vendored PERSONA trait json. split ∈ {'extract', 'eval'} (§7, §11)."""
    import json

    with open(TRAITS_DIR / f"{trait}_{split}.json") as f:
        return json.load(f)


def trait_eval_prompt(trait: str) -> str:
    """The trait's judge rubric (`eval_prompt`), a template with {question}/{answer}."""
    # extract and eval carry identical eval_prompt; read from extract.
    return trait_file(trait, "extract")["eval_prompt"]


def require_openai_key() -> str:
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return key


def openai_base_url() -> str | None:
    load_env()
    return os.environ.get("OPENAI_BASE_URL") or None
