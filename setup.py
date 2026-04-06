#!/usr/bin/env python3
"""Unified setup script for Player Summary Generator (Mistral-7B + QLoRA)."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required")
        return False
    print(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True


def setup_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = Path("llm_env")
    
    if venv_path.exists():
        print("Virtual environment already exists.")
        response = input("Recreate virtual environment? (y/n): ")
        if response.lower() == 'y':
            print("Removing existing virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("Using existing virtual environment.")
            return True
    
    print("Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv llm_env"):
        return False
    
    print("Virtual environment created successfully!")
    return True


def install_dependencies():
    """Install required packages."""
    print("\n" + "="*60)
    print("Installing dependencies...")
    print("="*60)
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "llm_env\\Scripts\\python.exe -m pip"
    else:  # Unix/Linux/Mac
        pip_cmd = "llm_env/bin/python -m pip"
    
    # Upgrade pip
    print("\n1. Upgrading pip...")
    run_command(f"{pip_cmd} install --upgrade pip", check=False)
    
    # Install PyTorch with CUDA (optional, can be CPU-only)
    print("\n2. Installing PyTorch...")
    print("   Choose installation type:")
    print("   1. PyTorch with CUDA 12.1 (recommended for GPU)")
    print("   2. PyTorch with CUDA 11.8")
    print("   3. PyTorch CPU-only (slower but works everywhere)")
    
    choice = input("Enter choice (1-3, default=1): ").strip() or "1"
    
    if choice == "1":
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif choice == "2":
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_cmd = "pip install torch torchvision torchaudio"
    
    run_command(f"{pip_cmd} {torch_cmd}", check=False)
    
    # Install other requirements
    print("\n3. Installing other dependencies...")
    if not run_command(f"{pip_cmd} install -r requirements.txt"):
        return False
    
    print("\n" + "="*60)
    print("Dependencies installed successfully!")
    print("="*60)
    return True


def verify_installation():
    """Verify that key packages are installed."""
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)
    
    if os.name == 'nt':
        python_cmd = "llm_env\\Scripts\\python.exe"
    else:
        python_cmd = "llm_env/bin/python"
    
    # Check PyTorch
    print("\nChecking PyTorch...")
    result = subprocess.run(
        f"{python_cmd} -c \"import torch; print(f'PyTorch: {{torch.__version__}}'); print(f'CUDA available: {{torch.cuda.is_available()}}')\"",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("WARNING: PyTorch verification failed")
    
    # Check TensorFlow
    print("\nChecking TensorFlow...")
    result = subprocess.run(
        f"{python_cmd} -c \"import tensorflow as tf; print(f'TensorFlow: {{tf.__version__}}')\"",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("WARNING: TensorFlow verification failed")
    
    # Check Transformers
    print("\nChecking Transformers...")
    result = subprocess.run(
        f"{python_cmd} -c \"import transformers; print(f'Transformers: {{transformers.__version__}}')\"",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("WARNING: Transformers verification failed")
    
    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)


def main():
    print("="*60)
    print("Player Summary Generator - Setup")
    print("="*60)
    print("\nThis script will:")
    print("  1. Create a virtual environment")
    print("  2. Install PyTorch (with CUDA support if desired)")
    print("  3. Install all required dependencies")
    print("  4. Verify the installation")
    print()
    
    if not check_python_version():
        sys.exit(1)
    
    if not setup_virtual_environment():
        print("ERROR: Failed to create virtual environment")
        sys.exit(1)
    
    if not install_dependencies():
        print("ERROR: Failed to install dependencies")
        sys.exit(1)
    
    verify_installation()
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    print("\nTo activate the virtual environment:")
    if os.name == 'nt':
        print("  Windows CMD: llm_env\\Scripts\\activate.bat")
        print("  Windows PowerShell: cmd /c llm_env\\Scripts\\activate.bat")
    else:
        print("  source llm_env/bin/activate")
    print("\nTo train:")
    print("  python llm_training/player_summary_advanced.py")
    print()


if __name__ == "__main__":
    main()

