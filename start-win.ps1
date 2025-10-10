# start-win.ps1 (agnostic to user/drive; runs from the script's folder)
$ErrorActionPreference = "Stop"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path  # $PSScriptRoot equivalent
Set-Location $HERE

$VENV = Join-Path $HERE ".venv-win"
$ACT  = Join-Path $VENV "Scripts\Activate.ps1"
$APP  = Join-Path $HERE "boot_cycle_gui_web-macpc.py"

if (-not (Test-Path $ACT)) {
  py -3 -m venv $VENV
}

. $ACT

# Install/repair deps if missing (prefer binary wheels; pin numpy to a known-good wheel)
$req = @('flask','opencv-python','pillow','imagehash')
if (-not (pip show numpy 2>$null)) { pip install --only-binary=:all: "numpy==2.2.6" }
foreach ($r in $req) {
  if (-not (pip show $r 2>$null)) { pip install --only-binary=:all: $r }
}

# If 5055 is busy, fall back to 5060
try {
  py $APP
} catch {
  Write-Host "Port 5055 busy. Trying 5060..."
  $env:FLASK_RUN_PORT = "5060"
  py $APP
}