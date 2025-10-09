#!/usr/bin/env bash
set -euo pipefail

# Go to the folder this script lives in
cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment (.venv)…"
  python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install/refresh deps (cached after first run)
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install flask opencv-python pillow imagehash numpy >/dev/null

# Run the app
echo "Starting Boot Cycle Logger on http://localhost:5055 …"
exec python3 boot_cycle_gui_web-macpc.py