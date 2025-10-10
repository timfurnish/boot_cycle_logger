# Boot Cycle Logger

Boot Cycle Logger is a Python + Flask application designed to monitor and count boot cycles on CCUs for reliability testing.  
It detects transitions between five main states:

- **Device Disconnected** – no scope connected  
- **Scope Connected (Sidewinder CCU)** – scope connected to native console  
- **Scope Connected (Other CCU)** – scope connected to a third-party console  
- **No Signal** – input is black / empty feed  
- **Other** – any unrecognized state

## Features
- Web interface for live status  
- Automatic CSV logging of state transitions and cycle counts  
- Image-based state detection with reference comparison  
- Ready-to-build scripts for macOS (`build-mac.sh`) and Windows (`build-win.ps1`)

## Running on macOS

```bash
source .venv/bin/activate
python boot_cycle_gui_web-macpc.py

## Running on Windows
.\.venv-win\Scripts\Activate.ps1
python boot_cycle_gui_web-macpc.py

##Building Standalone App

MacOS
sh build-mac.sh

Windows (Powershell)
.\build-win.ps1

## Project Structure

- `boot_cycle_gui_web-macpc.py` – Main Flask web app that monitors video input and detects CCU states.
- `build-mac.sh` – Script to build a standalone macOS app.
- `build-win.ps1` – Script to build a standalone Windows app.
- `start-win.ps1` – Shortcut script to launch the app on Windows.
- `requirements.txt` – Python dependencies required to run the project.
- `art/` – Folder containing reference images used for state detection.
- `logs/` – Stores CSV logs of boot cycles and state transitions.
- `nosignal-*.patch`, `roi-right.patch` – Patches for detection logic and ROI adjustments.
- `.gitignore` – Git configuration for ignoring unnecessary files.