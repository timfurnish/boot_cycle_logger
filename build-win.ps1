# ===== Setup =====
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to this script's directory (handles spaces in Google Drive path)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=== Boot Cycle Logger - Windows Build Script ==="

# Virtual env paths
$VenvDir = ".\.venv-win"
$PyExe   = Join-Path $VenvDir "Scripts\python.exe"
$PipExe  = Join-Path $VenvDir "Scripts\pip.exe"

# Create venv if missing
if (-not (Test-Path $PyExe)) {
    Write-Host "[1/6] Creating virtual environment..."
    py -3 -m venv $VenvDir
}

# Make sure we have pip/setuptools/wheel and PyInstaller
Write-Host "[2/6] Upgrading build tools (pip/setuptools/wheel)..."
& $PipExe install --upgrade pip setuptools wheel

# Core dependencies for the app + build
Write-Host "[3/6] Installing runtime dependencies..."
& $PipExe install flask opencv-python pillow imagehash numpy
Write-Host "[4/6] Installing PyInstaller..."
& $PipExe install --upgrade pyinstaller

# Clean previous artifacts
Write-Host "[5/6] Cleaning old build artifacts..."
Remove-Item -Recurse -Force build, dist, __pycache__ -ErrorAction SilentlyContinue
Remove-Item -Force "*.spec" -ErrorAction SilentlyContinue

# Optional icon
$IconFlag = @()
if (Test-Path ".\icon.ico") {
    $IconFlag = @("--icon", "icon.ico")
}

# Data files to bundle (Windows uses semicolon `;` in --add-data)
$AddData = @(
    "--add-data", "art;art",
    "--add-data", "templates;templates"
)

# Verify main script exists and has our new features
$MainScript = "boot_cycle_gui_web-macpc-6ch.py"
if (-not (Test-Path $MainScript)) {
    Write-Host "❌ ERROR: Main script '$MainScript' not found!"
    exit 1
}

# Check if the script has our new camera features
$ScriptContent = Get-Content $MainScript -Raw
if (-not ($ScriptContent -match "connectCamera|autoDetectCamera|listAvailableCameras")) {
    Write-Host "⚠️  WARNING: Script may not have latest camera features!"
    Write-Host "   Looking for: connectCamera, autoDetectCamera, listAvailableCameras"
}

Write-Host "[6/6] Building BootCycleLogger.exe with PyInstaller..."
Write-Host "   Using script: $MainScript"
& $PyExe -m PyInstaller --noconfirm --clean --onefile --noconsole `
    --name BootCycleLogger `
    @IconFlag `
    @AddData `
    $MainScript

# Summarize
if (Test-Path ".\dist\BootCycleLogger.exe") {
    $fullPath = Resolve-Path ".\dist\BootCycleLogger.exe"
    Write-Host "✅ Build successful! Executable:"
    Write-Host "   $fullPath"
    Write-Host ""
    Write-Host "Tip: If Windows blocked it because it's unsigned, right‑click → Properties → Unblock, or run once via 'More info → Run anyway'."
    exit 0
} else {
    Write-Host "❌ Build failed. See PyInstaller logs above."
    exit 1
}