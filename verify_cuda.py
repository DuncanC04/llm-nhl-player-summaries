#!/usr/bin/env python3
"""Verify CUDA installation"""
import torch

print("="*60)
print("PyTorch CUDA Verification")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print("\n[SUCCESS] CUDA is working correctly!")
else:
    print("\n[ERROR] CUDA is NOT available")
    print("   Make sure you installed PyTorch with CUDA support")
print("="*60)

