#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Boot Cycle Logger – macOS Build Script Starting…"

# Move to the script’s folder (handles spaces in Google Drive path)
cd "$(dirname "$0")" || exit 1
ROOT_DIR="$(pwd)"

# venv name for mac
VENV_DIR=".venv"
PY_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

# Create/activate virtual env
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "🔧 Creating virtual environment…"
  python3 -m venv "${VENV_DIR}"
fi

echo "🔧 Upgrading build tools…"
"${PIP_BIN}" install --upgrade pip setuptools wheel

# Install runtime + build deps
echo "📦 Installing dependencies…"
# Note: OpenCV wheel is available for Apple Silicon and Intel.
"${PIP_BIN}" install flask opencv-python pillow imagehash numpy pyinstaller

# Clean old artifacts
echo "🧹 Cleaning previous build artifacts…"
rm -rf build dist __pycache__ 2>/dev/null || true

# Icon (optional) – if you have an .icns, it will be used
ICON_FLAG=()
if [[ -f "icon.icns" ]]; then
  ICON_FLAG=(--icon "icon.icns")
fi

# IMPORTANT: On mac/Linux, --add-data uses colon (:) as the separator
ADD_DATA=(
  --add-data "$ROOT_DIR/art:art"
  --add-data "$ROOT_DIR/templates:templates"
)

# Build a single-file console binary (nice for seeing logs)
echo "🏗️  Building single-file CLI binary…"
"${PY_BIN}" -m PyInstaller --noconfirm --clean --onefile --strip \
  "${ICON_FLAG[@]}" \
  "${ADD_DATA[@]}" \
  --name BootCycleLogger-mac \
  boot_cycle_gui_web-macpc-6ch.py

# (Optional) Build a .app bundle (windowless) — logs will not show in a terminal
# Uncomment this block if you also want a macOS .app:
# echo "🏗️  Building .app bundle (windowless)…"
# "${PY_BIN}" -m PyInstaller --noconfirm --clean --windowed \
#   "${ICON_FLAG[@]}" \
#   "${ADD_DATA[@]}" \
#   --name BootCycleLogger \
#   boot_cycle_gui_web-macpc-6ch.py

# Summary
if [[ -f "dist/BootCycleLogger-mac" ]]; then
  echo "✅ Build successful!"
  echo "   Binary: $(pwd)/dist/BootCycleLogger-mac"
  echo
  echo "▶️  Run it:"
  echo "   ./dist/BootCycleLogger-mac"
else
  echo "❌ Build failed. Check the PyInstaller output above."
  exit 1
fi

echo "🎉 Done."