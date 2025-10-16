# Boot Cycle Logger

Boot Cycle Logger is a Python + Flask application designed to monitor and count boot cycles on CCUs for reliability testing.  
It detects transitions between five main states:

- **Device Disconnected** – no scope connected  
- **Scope Connected (Sidewinder CCU)** – scope connected to native console  
- **Scope Connected (Other CCU)** – scope connected to a third-party console  
- **No Signal** – input is black / empty feed (now correctly detected)  
- **Other** – any unrecognized state

## Features
- Support for **6 simultaneous video channels** arranged in a 3x2 layout, each with independent ROI detection and state tracking  
- Redesigned web interface featuring a **2/3 video display**, per-channel status pills, and per-channel state tallies  
- Adjustable **ROI brightness threshold** and **mean gate** controls for fine-tuning detection parameters  
- Ability to adjust ROI **internal padding/inset** for improved detection accuracy  
- Live CSV logging per channel with **timestamped filenames**, including controls to reset tallies and download logs  
- Debug overlay support for visualizing ROI positions on video feeds  
- Image-based state detection with support for multiple reference images for the connected state  
- Improved backend detection and fallback handling on macOS, including Source 2 and AVFoundation backends  
- Ready-to-build scripts for macOS (`build-mac.sh`) and Windows (`build-win.ps1`)

## Running on macOS

```bash
source .venv/bin/activate
python boot_cycle_gui_web-macpc.py
```

## Running on Windows

```powershell
.\.venv-win\Scripts\Activate.ps1
python boot_cycle_gui_web-macpc.py
```

## Building Standalone App

MacOS  
sh build-mac.sh

Windows (Powershell)  
.\build-win.ps1

## Project Structure

- `boot_cycle_gui_web-macpc.py`    – Main Flask web app that monitors video input and detects CCU states.  
- `build-mac.sh`                   – Script to build a standalone macOS app.  
- `build-win.ps1`                  – Script to build a standalone Windows app.  
- `start-win.ps1`                  – Shortcut script to launch the app on Windows.  
- `requirements.txt`               – Python dependencies required to run the project.  
- `art/`                          – Folder containing reference images used for state detection.  
- `logs/`                         – Stores CSV logs of boot cycles and state transitions.  
- `nosignal-*.patch`, `roi-right.patch` – Patches for detection logic and ROI adjustments.  
- `.gitignore`                    – Git configuration for ignoring unnecessary files.

---

# CHANGELOG.md

## [2025-10-14]

### Added
- Support for 6 simultaneous video channels arranged in a 3x2 layout with independent ROI detection and state tracking.
- Redesigned UI featuring a 2/3 video display, per-channel status pills, and per-channel state tallies.
- Adjustable ROI brightness threshold and mean gate controls for fine-tuning detection.
- Ability to adjust ROI internal padding/inset for better detection accuracy.
- Live CSV logging per channel with timestamped filenames, reset tallies button, and download button.
- Debug overlay support for visualizing ROI positions on video feeds.

## [Unreleased] - 2024-06-06

### Added
- Support for multiple reference images for the connected state to improve detection accuracy.
- Display of both full frame and 400×400 px ROI thumbnail in the web interface, each at their true aspect ratios.
- Improved backend detection and fallback mechanisms on macOS, including support for Source 2 and AVFoundation video capture backends.
- Correct detection of black screens as the “No Signal” state.

### Changed
- Rewritten ROI handling: ROI is now a fixed 400×400 px region positioned 420 px from the left bottom of the frame.
- Enhanced detection logic to better distinguish between `Device Disconnected`, `Scope Connected`, and `No Signal` states.
- Improved fallback handling for video sources on macOS to increase robustness.

### Fixed
- Bug fixes related to misclassification of the “Other” state.
- Various minor improvements to state transition logging and UI responsiveness.