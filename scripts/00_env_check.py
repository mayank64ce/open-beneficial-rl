#!/usr/bin/env python
"""Day-1 environment check (§11, risk #8).

unsloth needs network access at load (it remaps the model repo), so this UNSETS
HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE, then verifies the pinned stack, GPU memory,
PatchFastRL availability, and OPENAI_API_KEY. Run before anything else.
"""
import os
import sys

# Must happen before any HF import (§11).
for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    if var in os.environ:
        print(f"  unset {var} (was {os.environ[var]!r})")
        del os.environ[var]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WANT = {
    "torch": "2.6.0", "transformers": "4.52.3", "trl": "0.15.2",
    "unsloth": "2025.5.9", "peft": "0.15.1", "vllm": "0.8.5.post1",
    "bitsandbytes": "0.45.5", "accelerate": "1.7.0", "datasets": "3.6.0",
}


def main() -> int:
    ok = True
    import importlib.metadata as m

    print("== package versions ==")
    for pkg, want in WANT.items():
        try:
            got = m.version(pkg)
        except Exception:
            print(f"  {pkg:14s} MISSING  (want {want})")
            ok = False
            continue
        flag = "OK" if got == want else f"!! want {want}"
        if got != want:
            ok = False
        print(f"  {pkg:14s} {got:14s} {flag}")

    print("== GPU ==")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA not available"); ok = False
        else:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            cc = torch.cuda.get_device_capability(0)
            print(f"  {name}  {total:.1f} GB  cc {cc[0]}.{cc[1]}  cuda {torch.version.cuda}")
            if total < 20:
                print("  !! <20 GB — the 24 GB memory budget (§11) assumes an RTX 3090"); ok = False
    except Exception as e:
        print(f"  torch/CUDA error: {e}"); ok = False

    print("== PatchFastRL ==")
    try:
        from unsloth import FastLanguageModel, PatchFastRL  # noqa: F401
        assert callable(PatchFastRL)
        print("  unsloth.PatchFastRL available")
    except Exception as e:
        print(f"  !! {e}"); ok = False

    print("== OpenAI key ==")
    try:
        from rlp import config
        config.require_openai_key()
        print("  OPENAI_API_KEY present")
        if config.openai_base_url():
            print(f"  OPENAI_BASE_URL = {config.openai_base_url()}")
    except Exception as e:
        print(f"  !! {e}"); ok = False

    print("== offline flags ==")
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        val = os.environ.get(var)
        print(f"  {var} = {val!r}" + ("  !! should be unset for unsloth load" if val else "  (unset, good)"))

    print()
    print("ENV CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
