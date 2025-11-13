# PowerShell script to install PyTorch with CUDA support
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Install PyTorch with CUDA Support" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
$venvPath = "llm_env\Scripts\python.exe"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment found: $venvPath" -ForegroundColor Green
    $pythonCmd = $venvPath
    $pipCmd = "$pythonCmd -m pip"
} else {
    Write-Host "WARNING: Virtual environment not found!" -ForegroundColor Yellow
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv llm_env
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    $pythonCmd = $venvPath
    $pipCmd = "$pythonCmd -m pip"
    Write-Host "Virtual environment created." -ForegroundColor Green
}

Write-Host ""
Write-Host "Using Python: $pythonCmd" -ForegroundColor Cyan
Write-Host ""

# Step 1: Uninstall CPU-only PyTorch
Write-Host "Step 1: Uninstalling CPU-only PyTorch..." -ForegroundColor Yellow
& $pythonCmd -m pip uninstall -y torch torchvision torchaudio

# Step 2: Install PyTorch with CUDA
Write-Host ""
Write-Host "Step 2: Installing PyTorch with CUDA 12.1..." -ForegroundColor Yellow
Write-Host "This may take several minutes (downloading ~2.5GB)..." -ForegroundColor Yellow
& $pythonCmd -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Installation failed!" -ForegroundColor Red
    Write-Host "Try running PowerShell as Administrator" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 3: Verify installation
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Verifying CUDA installation..." -ForegroundColor Yellow
& $pythonCmd -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host ""
Write-Host "IMPORTANT: Activate the virtual environment before running training:" -ForegroundColor Yellow
Write-Host "  llm_env\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "  or: cmd /c llm_env\Scripts\activate.bat" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"

