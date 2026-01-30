# BrainTools Installer
# Run this script once to set up everything correctly.
#
# Usage: Right-click this file > "Run with PowerShell"
# Or from terminal: powershell -ExecutionPolicy Bypass -File install.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  BrainTools Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the braintool conda env
$envPath = $env:CONDA_PREFIX
if ($envPath -notlike "*braintool*") {
    Write-Host "[!] Please activate the braintool environment first:" -ForegroundColor Yellow
    Write-Host '    conda activate "Y:\2_Connectome\envs\braintool"' -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "[1/4] Uninstalling any existing torch..." -ForegroundColor Green
pip uninstall torch torchvision torchaudio -y 2>$null

Write-Host "[2/4] Installing PyTorch (CPU version)..." -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Write-Host "[3/4] Installing braintools package..." -ForegroundColor Green
pip install -e .

Write-Host "[4/4] Installing napari plugin..." -ForegroundColor Green
Push-Location "..\3D_Cleared\util_Scripts\sci_connectome_napari"
pip install -e .
Pop-Location

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To launch: braintool" -ForegroundColor White
Write-Host "To verify: braintool --check" -ForegroundColor White
Write-Host ""
