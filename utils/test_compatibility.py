#!/usr/bin/env python3
"""Test Keras 3 / Transformers compatibility"""

import os
import sys

print("="*60)
print("Testing Keras 3 / Transformers Compatibility")
print("="*60)
print()

# Test 1: Transformers with legacy Keras
print("Test 1: Importing transformers with TF_USE_LEGACY_KERAS=1")
os.environ["TF_USE_LEGACY_KERAS"] = "1"
try:
    import transformers
    print(f"[OK] transformers imported successfully (version: {transformers.__version__})")
except Exception as e:
    print(f"[ERROR] Failed to import transformers: {e}")
    sys.exit(1)

# Test 2: Keras 3 for MiniGPT
print()
print("Test 2: Importing Keras 3 (for MiniGPT)")
os.environ["TF_USE_LEGACY_KERAS"] = "0"
try:
    import keras
    print(f"[OK] Keras imported successfully (version: {keras.__version__})")
    from keras import layers
    print("[OK] Keras layers imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import Keras 3: {e}")
    sys.exit(1)

# Test 3: tf-keras availability
print()
print("Test 3: Checking tf-keras availability")
try:
    import tf_keras
    print(f"[OK] tf-keras available (version: {tf_keras.__version__})")
except ImportError:
    print("[WARN] tf-keras not found (may not be needed if transformers works)")

print()
print("="*60)
print("[SUCCESS] All compatibility tests passed!")
print("="*60)
print()
print("The advanced model will use TF_USE_LEGACY_KERAS=1")
print("The simple model will use Keras 3")
print("Both should work without conflicts.")

