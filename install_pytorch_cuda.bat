@echo off
echo ========================================
echo Install PyTorch with CUDA Support
echo ========================================
echo.

echo Your system has:
echo   - NVIDIA GeForce RTX 3060 Ti (8GB VRAM)
echo   - CUDA Driver Version: 13.0
echo.
echo This script will:
echo   1. Check for virtual environment
echo   2. Uninstall CPU-only PyTorch
echo   3. Install PyTorch with CUDA 12.1 support
echo.

REM Check if virtual environment exists
if exist "llm_env\Scripts\python.exe" (
    echo Virtual environment found. Using llm_env\Scripts\python.exe
    set PYTHON_CMD=llm_env\Scripts\python.exe
    set PIP_CMD=llm_env\Scripts\python.exe -m pip
) else (
    echo WARNING: Virtual environment not found!
    echo.
    echo Creating virtual environment first...
    python -m venv llm_env
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    set PYTHON_CMD=llm_env\Scripts\python.exe
    set PIP_CMD=llm_env\Scripts\python.exe -m pip
    echo Virtual environment created.
)

echo.
echo Using Python: %PYTHON_CMD%
echo.
pause

echo.
echo Step 1: Uninstalling CPU-only PyTorch...
%PIP_CMD% uninstall -y torch torchvision torchaudio

echo.
echo Step 2: Installing PyTorch with CUDA 12.1...
echo This may take several minutes (downloading ~2.5GB)...
%PIP_CMD% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo.
    echo Try running this script as Administrator, or:
    echo   1. Close all Python processes
    echo   2. Run Command Prompt as Administrator
    echo   3. Run this script again
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Step 3: Verifying CUDA installation...
%PYTHON_CMD% -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo IMPORTANT: Activate the virtual environment before running the training script:
echo   llm_env\Scripts\activate.bat
echo.
echo To verify CUDA is working anytime, run:
echo   llm_env\Scripts\python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo.
pause

