#!/usr/bin/env python3
"""
Fix Keras 3 / Transformers compatibility issues.

This script installs tf-keras to ensure compatibility with transformers library
that may not yet support Keras 3.
"""

import subprocess
import sys

def main():
    print("="*60)
    print("Keras 3 / Transformers Compatibility Fix")
    print("="*60)
    print("\nInstalling tf-keras for compatibility...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-keras>=2.15.0"])
        print("\n✅ Successfully installed tf-keras!")
        print("\nYou may need to set TF_USE_LEGACY_KERAS=1 in your environment")
        print("or in your script before importing transformers.")
        print("\nThe advanced model script automatically handles this.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing tf-keras: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

