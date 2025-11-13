#!/usr/bin/env python3
"""
Fix Keras 3 compatibility issues with transformers library.
Installs tf-keras and sets up environment properly.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    return result.returncode == 0

def main():
    print("="*60)
    print("Fixing Keras 3 / Transformers Compatibility")
    print("="*60)
    print()
    
    # Determine pip command
    if os.path.exists("llm_env/Scripts/pip.exe") or os.path.exists("llm_env/bin/pip"):
        if os.name == 'nt':  # Windows
            pip_cmd = "llm_env\\Scripts\\python.exe -m pip"
        else:  # Unix/Linux/Mac
            pip_cmd = "llm_env/bin/python -m pip"
    else:
        pip_cmd = f"{sys.executable} -m pip"
    
    print("Step 1: Installing tf-keras (backwards compatible Keras)...")
    if not run_command(f"{pip_cmd} install tf-keras>=2.15.0"):
        print("ERROR: Failed to install tf-keras")
        return False
    
    print()
    print("Step 2: Verifying installation...")
    print("Checking if tf-keras is available...")
    
    # Test import
    test_script = """
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
try:
    import tf_keras
    print("✓ tf-keras installed successfully")
    print(f"  Version: {tf_keras.__version__}")
except ImportError as e:
    print(f"✗ Failed to import tf-keras: {e}")
    sys.exit(1)

try:
    from transformers import __version__ as transformers_version
    print(f"✓ transformers version: {transformers_version}")
except ImportError as e:
    print(f"✗ transformers not found: {e}")
    sys.exit(1)

print("\\n✓ Compatibility fix complete!")
"""
    
    if os.name == 'nt':
        python_cmd = "llm_env\\Scripts\\python.exe" if os.path.exists("llm_env/Scripts/python.exe") else sys.executable
    else:
        python_cmd = "llm_env/bin/python" if os.path.exists("llm_env/bin/python") else sys.executable
    
    result = subprocess.run(
        [python_cmd, "-c", test_script],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    if result.returncode == 0:
        print()
        print("="*60)
        print("SUCCESS: Keras compatibility fixed!")
        print("="*60)
        print()
        print("The advanced model script will automatically set")
        print("TF_USE_LEGACY_KERAS=1 to use tf-keras with transformers.")
        print()
        print("The simple model will continue using Keras 3.")
        return True
    else:
        print()
        print("="*60)
        print("ERROR: Verification failed")
        print("="*60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

