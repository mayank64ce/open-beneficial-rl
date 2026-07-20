"""Per-run provenance: runs/<arm>/<timestamp>/{config.yaml,env.json,git.json} (§10)."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from . import config


def new_run_dir(arm: str, root: Path | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = (root or config.RUNS_DIR) / arm / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=config.REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def git_sha() -> str:
    return _git("rev-parse", "HEAD")


def write_provenance(run_dir: Path, cfg: dict, extra: dict | None = None) -> None:
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({**cfg, **(extra or {})}, f, sort_keys=False)

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in ["torch", "transformers", "trl", "unsloth", "peft", "vllm",
                "bitsandbytes", "accelerate", "datasets"]:
        try:
            import importlib.metadata as m
            env[pkg] = m.version(pkg)
        except Exception:
            env[pkg] = "MISSING"
    try:
        import torch
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    with open(run_dir / "env.json", "w") as f:
        json.dump(env, f, indent=2)

    with open(run_dir / "git.json", "w") as f:
        json.dump({
            "sha": git_sha(),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "describe_tags": _git("describe", "--tags", "--always"),
        }, f, indent=2)
