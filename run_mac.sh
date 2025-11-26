#!/usr/bin/env bash
set -euo pipefail

# Go to the folder this script lives in
cd "$(dirname "$0")"

# Kill any existing Boot Cycle Logger processes
echo "[0/5] Cleaning up existing processes..."
pkill -f "boot_cycle_gui_web-macpc-6ch.py" 2>/dev/null || true
pkill -f "BootCycleLogger" 2>/dev/null || true
echo "Existing processes cleaned up."

# Detect machine-specific venv name (like Windows does)
echo "[1/5] Detecting virtual environment..."
HOSTNAME=$(hostname -s | tr -d ' "' | tr '[:upper:]' '[:lower:]')
VENV_DIR=".venv-${HOSTNAME}"
echo "  Detected hostname: ${HOSTNAME}"
echo "  Machine-specific venv: ${VENV_DIR}"

# Check for machine-specific venv first, then fallback to generic .venv-mac
if [ -d "${VENV_DIR}" ]; then
  echo "  ✓ Found machine-specific venv: ${VENV_DIR}"
elif [ -d ".venv-mac" ]; then
  echo "  ✓ Found shared Mac venv: .venv-mac"
  VENV_DIR=".venv-mac"
elif [ -d ".venv" ]; then
  echo "  ✓ Found generic venv: .venv"
  VENV_DIR=".venv"
else
  echo "  ✗ No existing venv found, creating: ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# Activate venv
echo "[2/5] Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Install/refresh deps (cached after first run)
echo "[3/5] Upgrading pip..."
python3 -m pip install --upgrade pip >/dev/null
echo "[4/5] Installing dependencies..."
python3 -m pip install flask opencv-python pillow imagehash numpy scipy openpyxl PyWavelets pygrabber >/dev/null 2>&1 || true

# Run the app
echo "[5/5] Starting Boot Cycle Logger on http://localhost:5055 …"
echo ""
echo "The application should open in your browser automatically."
echo "Press Ctrl+C to stop the application."
echo ""
exec python3 boot_cycle_gui_web-macpc-6ch.py