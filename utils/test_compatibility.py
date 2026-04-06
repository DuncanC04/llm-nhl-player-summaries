#!/usr/bin/env python3
"""Smoke-test core imports for the Mistral training stack."""

import os
import sys

print("=" * 60)
print("Testing Mistral / Transformers stack")
print("=" * 60)
print()

os.environ["TF_USE_LEGACY_KERAS"] = "1"

print("Test 1: transformers")
try:
    import transformers

    print(f"  [OK] transformers {transformers.__version__}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

print("Test 2: torch")
try:
    import torch

    print(f"  [OK] torch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

print("Test 3: peft + trl (training)")
try:
    import peft
    import trl

    print(
        f"  [OK] peft {getattr(peft, '__version__', '?')}, "
        f"trl {getattr(trl, '__version__', '?')}"
    )
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

print()
print("=" * 60)
print("[SUCCESS] Core imports OK")
print("=" * 60)
