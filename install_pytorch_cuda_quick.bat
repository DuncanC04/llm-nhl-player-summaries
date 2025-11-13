@echo off
REM Quick installation script - assumes virtual environment is activated
echo ========================================
echo Quick PyTorch CUDA Installation
echo ========================================
echo.
echo Make sure you've activated your virtual environment first!
echo If not, run: llm_env\Scripts\activate.bat
echo.
pause

echo.
echo Uninstalling CPU-only PyTorch...
python -m pip uninstall -y torch torchvision torchaudio

echo.
echo Installing PyTorch with CUDA 12.1...
echo This may take several minutes (downloading ~2.5GB)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo Done!
pause

