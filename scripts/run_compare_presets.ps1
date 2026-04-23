# Train all registered presets on the same JSONL, export validation predictions, run evaluation.run_eval for each.
#
# Prerequisites (from repo root):
#   - NVIDIA GPU + CUDA-enabled PyTorch (README: PyTorch CUDA Support)
#   - pip install -r requirements.txt
#   - pip install -r requirements-eval.txt
#   - Gold JSONL at Data/out/aiTop10Stats_complete.jsonl or pass -Gold
#
# Usage (PowerShell, repo root):
#   .\scripts\run_compare_presets.ps1
#   .\scripts\run_compare_presets.ps1 -Epochs 1 -PythonExe .\.venv\Scripts\python.exe
#   .\scripts\run_compare_presets.ps1 -SkipTrain   # only test + eval (adapters must exist under outputs/compare_presets/)

param(
    [string]$PythonExe = "",
    [string]$Gold = "Data/out/aiTop10Stats_complete.jsonl",
    [int]$Epochs = 3,
    [int]$NumTest = -1,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $PythonExe) {
    $candidates = @(
        (Join-Path $Root ".venv\Scripts\python.exe"),
        (Join-Path $Root "llm_env\Scripts\python.exe"),
        "python"
    )
    foreach ($c in $candidates) {
        if ($c -eq "python" -or (Test-Path -LiteralPath $c)) {
            $PythonExe = $c
            break
        }
    }
}

Write-Host "Using Python: $PythonExe"

& $PythonExe -c "import torch; assert torch.cuda.is_available(), 'CUDA required for training/inference script'; print('CUDA OK:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "This project''s training CLI requires CUDA. Install GPU PyTorch, then re-run." -ForegroundColor Yellow
    Write-Host "See README.md -> PyTorch CUDA Support." -ForegroundColor Yellow
    exit 1
}

if ([System.IO.Path]::IsPathRooted($Gold)) {
    $goldPath = $Gold
} else {
    $goldPath = Join-Path $Root $Gold
}
if (-not (Test-Path -LiteralPath $goldPath)) {
    Write-Host "Gold JSONL not found: $goldPath" -ForegroundColor Red
    exit 1
}

$out = Join-Path $Root "outputs\compare_presets"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$train = Join-Path $Root "llm_training\player_summary_advanced.py"
$presets = @("mistral", "phi-3-mini", "gpt2", "qwen3-1.7b")

function Get-SafeName([string]$preset) {
    return $preset.Replace(".", "").Replace("-", "_")
}

if (-not $SkipTrain) {
    foreach ($preset in $presets) {
        $safe = Get-SafeName $preset
        $adapterDir = Join-Path $out "adapter_$safe"
        $predPath = Join-Path $out "${safe}_val_predictions.jsonl"

        Write-Host "`n=== Train + export: $preset ===" -ForegroundColor Cyan
        & $PythonExe $train `
            --data_path $goldPath `
            --model_preset $preset `
            --output_dir $adapterDir `
            --num_epochs $Epochs `
            --num_test $NumTest `
            --export_predictions $predPath
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
else {
    Write-Host "`n=== SkipTrain: exporting validation preds only ===" -ForegroundColor Cyan
    foreach ($preset in $presets) {
        $safe = Get-SafeName $preset
        $adapterDir = Join-Path $out "adapter_$safe"
        $predPath = Join-Path $out "${safe}_val_predictions.jsonl"

        if (-not (Test-Path -LiteralPath $adapterDir)) {
            Write-Host "Missing $adapterDir" -ForegroundColor Red
            exit 1
        }

        Write-Host "`n=== Test + export: $preset ===" -ForegroundColor Cyan
        & $PythonExe $train --test_only --model_preset $preset --data_path $goldPath --output_dir $adapterDir `
            --num_test $NumTest --export_predictions $predPath
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

foreach ($preset in $presets) {
    $safe = Get-SafeName $preset
    $predPath = Join-Path $out "${safe}_val_predictions.jsonl"
    $reportPath = Join-Path $out "eval_report_${safe}.json"

    Write-Host "`n=== evaluation.run_eval: $preset ===" -ForegroundColor Cyan
    & $PythonExe -m evaluation.run_eval --gold $goldPath --pred $predPath --out $reportPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`nDone." -ForegroundColor Green
foreach ($preset in $presets) {
    $safe = Get-SafeName $preset
    $reportPath = Join-Path $out "eval_report_${safe}.json"
    Write-Host "  ${preset} report: $reportPath"
}
