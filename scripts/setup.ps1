# Bootstrap the PP-Vehicle project on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Gpu -Cuda cu126
#   powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -SkipModels

[CmdletBinding()]
param(
    [switch]$Gpu,
    [string]$Cuda = "cu126",
    [switch]$SkipModels,
    [string]$Branch = "release/2.9",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "Project root: $Root" -ForegroundColor Cyan

# lap (MOT tracker) pins Requires-Python <3.12, and cython_bbox / pycocotools
# have no Windows wheels there either. Docker is the supported path.
$PyVer = [version](python -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))")
if ($PyVer -ge [version]"3.12") {
    Write-Host ""
    Write-Host "Python $PyVer cannot resolve PaddleDetection's dependencies." -ForegroundColor Red
    Write-Host "lap requires Python <3.12; cython_bbox and pycocotools have no wheels." -ForegroundColor Red
    Write-Host ""
    Write-Host "Use Docker instead:" -ForegroundColor Yellow
    Write-Host "  docker compose -f docker/docker-compose.yml up --build"
    Write-Host ""
    Write-Host "Or install Python 3.10 and re-run with:  -PythonExe C:\Path\To\python3.10.exe"
    exit 1
}

# --- 1. virtualenv ---------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "`n[1/5] Creating .venv" -ForegroundColor Cyan
    & $PythonExe -m venv .venv
} else {
    Write-Host "`n[1/5] .venv already exists" -ForegroundColor Cyan
}
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip setuptools wheel

# --- 2. vendor PaddleDetection --------------------------------------------
Write-Host "`n[2/5] Vendoring PaddleDetection ($Branch)" -ForegroundColor Cyan
$Vendor = Join-Path $Root "vendor\PaddleDetection"
if (-not (Test-Path (Join-Path $Vendor "deploy\pipeline\pipeline.py"))) {
    if (Test-Path $Vendor) { Remove-Item -Recurse -Force $Vendor }
    # Shallow, single-branch: we only need the deploy/ tree, not 5 years of history.
    git clone --depth 1 --branch $Branch --single-branch `
        https://github.com/PaddlePaddle/PaddleDetection.git $Vendor
} else {
    Write-Host "  already present, skipping clone"
}

# --- 3. python dependencies ------------------------------------------------
Write-Host "`n[3/5] Installing dependencies" -ForegroundColor Cyan
& $Py -m pip install -r requirements.txt
& $Py -m pip install -e .

if ($Gpu) {
    Write-Host "  switching to paddlepaddle-gpu ($Cuda)" -ForegroundColor Yellow
    & $Py -m pip uninstall -y paddlepaddle
    & $Py -m pip install paddlepaddle-gpu==3.3.1 `
        -i "https://www.paddlepaddle.org.cn/packages/stable/$Cuda/"
}

# --- 4. models -------------------------------------------------------------
if ($SkipModels) {
    Write-Host "`n[4/5] Skipping model download (-SkipModels)" -ForegroundColor Cyan
} else {
    Write-Host "`n[4/5] Downloading model weights (~282 MB)" -ForegroundColor Cyan
    & $Py -m lalin models --download
}

# --- 5. doctor -------------------------------------------------------------
Write-Host "`n[5/5] Environment check" -ForegroundColor Cyan
& $Py -m lalin doctor

Write-Host "`nDone. Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m lalin scenarios"
Write-Host "  python -m lalin run --scenario tracking --source data\samples\your.mp4"
Write-Host "  python -m lalin serve      # http://127.0.0.1:8000"
