#!/bin/bash
echo "========================================"
echo "Install PyTorch with CUDA Support"
echo "========================================"
echo ""

echo "This script will:"
echo "  1. Uninstall CPU-only PyTorch"
echo "  2. Install PyTorch with CUDA 12.1 support"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "Step 1: Uninstalling CPU-only PyTorch..."
python -m pip uninstall -y torch torchvision torchaudio

echo ""
echo "Step 2: Installing PyTorch with CUDA 12.1..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "IMPORTANT: Restart your Python environment/terminal before testing."
echo ""
echo "To verify CUDA is working, run:"
echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
echo ""

