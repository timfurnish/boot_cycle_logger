# Boot Cycle Logger - Windows PowerShell Launcher
# Right-click and select "Run with PowerShell"

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Boot Cycle Logger - Starting..." -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Move to script's directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Virtual environment path
$VenvDir = ".\.venv-win"
$PyExe = Join-Path $VenvDir "Scripts\python.exe"
$PipExe = Join-Path $VenvDir "Scripts\pip.exe"

# Create virtual environment if it doesn't exist
if (-not (Test-Path $PyExe)) {
    Write-Host "[1/3] Creating virtual environment..." -ForegroundColor Yellow
    try {
        py -3 -m venv $VenvDir
    } catch {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Write-Host "Please ensure Python 3 is installed and added to PATH" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Install dependencies
Write-Host "[2/3] Installing dependencies..." -ForegroundColor Yellow
& $PyExe -m pip install --upgrade pip --quiet 2>$null
& $PipExe install flask opencv-python pillow imagehash numpy --quiet 2>$null

# Run the application
Write-Host "[3/3] Launching Boot Cycle Logger..." -ForegroundColor Green
Write-Host ""
Write-Host "The application will open in your browser automatically." -ForegroundColor Green
Write-Host "To stop the server, close this window or press Ctrl+C" -ForegroundColor Green
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan

try {
    & $PyExe boot_cycle_gui_web-macpc-6ch.py
} catch {
    Write-Host ""
    Write-Host "===============================================" -ForegroundColor Red
    Write-Host "ERROR: Application failed to start" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "===============================================" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

