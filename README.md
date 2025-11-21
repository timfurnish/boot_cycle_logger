# Boot Cycle Logger - Enhanced Edition

## Purpose

Boot Cycle Logger is a comprehensive Python + Flask application designed to monitor and log boot cycles for 3 test consoles, each with 2 video outputs (6 channels total). It automatically detects and records device states during power-on/power-off cycles to validate boot reliability and identify anomalies in medical equipment testing scenarios.

The system is designed to capture even single-frame flickers and provides detailed analysis of boot reliability across all monitored devices, making it ideal for rigorous reliability testing where 1000+ power cycles per console are common.

## How It Works

### Architecture Overview

1. **Video Capture**: Uses OpenCV to capture a single video feed (typically OBS Virtual Camera) that displays a 3×2 grid of 6 video channels
2. **Grid Processing**: Splits the incoming video feed into 6 tiles, one per channel
3. **State Detection**: Analyzes each tile independently to classify its current state
4. **Logging**: Records all state changes and timing data to CSV/Excel files
5. **Web UI**: Provides a Flask-based interface for control and real-time monitoring

### State Detection Algorithm

The `decide()` function classifies each frame into one of four states:

1. **NO_SIGNAL**: Dark/flat frame or very dark ROI (device off or no video signal)
2. **Scope Disconnected (BARS)**: Matches a reference "bars" pattern using perceptual hashing (pHash)
3. **Scope Connected (INTERFACE)**: Matches a reference "connected interface" pattern or shows bright white/near-white ROI
4. **OTHER**: Anything else (treated as an anomaly and automatically captured with screenshots)

**Detection Methodology:**
- **Perceptual Hashing (pHash)**: Compares frames to reference images using visual feature matching
- **ROI Analysis**: Analyzes brightness, white pixel fraction, and mean luminance in a Region of Interest
- **Stabilization Logic**: Requires N consecutive frames + hold time to prevent false positives
- **Configurable Thresholds**: Adjustable parameters for fine-tuning detection sensitivity

### Expected Test Scenarios

For a typical test with **3 consoles, 1000 power cycles each**:
- **3,000 total cycles** across all consoles
- **6 channels** × **3 states per cycle** = **18 state transitions per cycle**
- **Minimum 54,000 state change records**
- Plus periodic snapshots (every 1 second) for continuous logging
- **Total expected: 20,000+ records per test run**

## States Detected

- **Scope Connected** – Scope successfully connected and operational (white interface visible)
- **Scope Disconnected** – Scope disconnected or connection lost (bars pattern visible)
- **No Signal** – Input is black/empty feed (device powered off or no video signal)
- **Other** – Any unrecognized state (treated as anomaly, automatically screenshotted)

## Key Features

### Core Functionality
- **6-Channel Video Monitoring** – Simultaneous monitoring of 6 video feeds in 3×2 grid layout
- **Equipment Identification** – Pre-test modal for entering console serial numbers and scope IDs
- **Live Video Feed** – Video starts immediately when app loads, equipment modal overlays live feed
- **Real-time UI Updates** – Live status pills showing current state of each channel
- **Anomaly Detection** – Automatic screenshot capture and tracking for "Other" state detections

### Data Logging & Analysis
- **Comprehensive CSV Logging** – Detailed data with equipment IDs, timing, cycle counts, and match distances
- **Excel Export** – Formatted Excel files with same data as CSV
- **Automated Report Generation** – Statistical analysis appended to CSV file including:
  - Per-channel cycle counts and timing analysis
  - Reconnection time statistics (average, min, max, std deviation)
  - Incomplete cycle detection
  - Anomaly summary with per-channel breakdown
- **Organized Output** – Each test run creates a timestamped folder containing:
  - CSV file with all state changes
  - Excel file (formatted version)
  - `captures/` subfolder with anomaly screenshots

### Timing & Cycle Tracking
- **Precise Timing Analysis** – Measures elapsed time from Scope Disconnected → Scope Connected
- **Complete Cycle Tracking** – Counts full cycles (No Signal → Disconnected → Connected)
- **Partial Cycle Detection** – Identifies incomplete cycles (No Signal → Disconnected → No Signal)
- **Per-Channel Statistics** – Independent tracking for all 6 channels

### Detection Controls
- **ROI Detection Controls** – Adjustable brightness thresholds and detection parameters
- **Stabilization Settings** – Configurable stable frames and hold time for noise reduction
- **State Detection Tuning** – Adjustable pHash thresholds, white percentage, and mean gates

### Platform Support
- **Cross-Platform** – Works on macOS, Windows, and Linux
- **OBS Virtual Camera Integration** – Automatic detection and connection on Windows
- **Standalone Executables** – Ready-to-build scripts for macOS and Windows
- **Cross-Platform File Access** – "Go to CSV" button opens file browser and highlights recent CSV

## Quick Start

### 1. Equipment Setup
- Launch the application (see Running from Source below)
- Video feed starts immediately
- Equipment identification modal appears over live video
- Enter console serial numbers and scope IDs for each of the 3 console sets:
  - **Console 1**: Videos 1 & 2 (HDMI outputs)
  - **Console 2**: Videos 3 & 4 (HDMI outputs)  
  - **Console 3**: Videos 5 & 6 (HDMI outputs)
- Click "Save and Continue"

### 2. Test Execution
- Click "Start Test" to begin detection and logging
  - Creates a timestamped folder: `logs/test_YYYYMMDD_HHMMSS/`
  - Begins logging all state changes to CSV
- Monitor real-time status pills for each channel
- System automatically captures screenshots for any "Other" state detections (anomalies)
- Click "End Test" when testing is complete

### 3. Results Analysis
- Report appears on screen with comprehensive statistics and analysis
- Click "Go to CSV" to open file browser and highlight the CSV file
- CSV contains all data plus appended report
- Excel file available in same folder
- Anomaly screenshots in `captures/` subfolder (if any detected)

## Running from Source

### Windows (Easy - Double-Click)
1. **Option 1**: Double-click `run_windows.bat`
2. **Option 2**: Right-click `run_windows.ps1` → "Run with PowerShell"

The launcher will automatically:
- Create virtual environment if needed
- Install all dependencies
- Auto-detect OBS Virtual Camera or other video sources
- Open the application in your browser at `http://localhost:5055/`

### Windows (Manual)
```powershell
.\.venv-win\Scripts\Activate.ps1
python boot_cycle_gui_web-macpc-6ch.py
```

Then open `http://localhost:5055/` in your browser.

### macOS
```bash
source .venv/bin/activate
python boot_cycle_gui_web-macpc-6ch.py
```

Then open `http://localhost:5055/` in your browser.

## Building Standalone Executables

### macOS
```bash
chmod +x build-mac.sh
./build-mac.sh
```

### Windows (PowerShell)
```powershell
.\build-win.ps1
```

**Output**: `dist/BootCycleLogger-mac` (macOS) or `dist/BootCycleLogger.exe` (Windows)

## OBS Virtual Camera Setup (Windows)

The Boot Cycle Logger includes automatic OBS Virtual Camera detection for Windows users:

### Automatic Detection (Recommended)
1. **Start OBS Studio** and enable Virtual Camera (Tools → Start Virtual Camera)
2. **Run the Boot Cycle Logger** using `run_windows.bat` or `run_windows.ps1`
3. The app will automatically detect and connect to OBS Virtual Camera on startup

### How It Works
- On startup, the app scans for available cameras using DSHOW (best for OBS) and MSMF backends
- Prioritizes DirectShow (DSHOW) which works best with OBS Virtual Camera
- Tests each camera source (indices 0-5) to find the first working camera
- Automatically configures the Source and Backend settings
- Sets optimal resolution (1920×1080) and buffer size for real-time frame updates

### Troubleshooting OBS Virtual Camera

#### **Quick Troubleshooting (In-App)**
If auto-detection doesn't find OBS Virtual Camera:
1. **Make sure OBS Virtual Camera is started** (green indicator in OBS Studio)
2. Click **"Auto-Detect OBS"** button in the Camera Source section
3. Click **"List Cameras"** to see all available cameras with their indices
4. Click **"System Camera Report"** for comprehensive system-level diagnostics
5. Try manually setting:
   - **Backend**: DSHOW (for Windows) or AVFOUNDATION (for macOS)
   - **Source**: Try indices 0-20 (expanded search range)
   - Click **"Connect to Camera"** after changing settings

#### **Advanced Diagnostics ("Flicker" PC Issues)**
If the above doesn't work (especially for PCs with persistent detection issues):

**Windows Users - Run the Diagnostic Script:**
```bash
# Double-click this file:
diagnose_flicker.bat
```

This comprehensive diagnostic will:
- Test all available backends (DSHOW, MSMF, ANY)
- Scan camera indices 0-30
- Check Windows device registry for OBS Virtual Camera
- Verify OBS Studio is running and Virtual Camera is started  
- Test both index-based and name-based camera access
- Provide specific actionable troubleshooting steps

**Common Fixes for "Flicker" PC Issues:**
1. **OBS Not Running**: Start OBS Studio and click "Start Virtual Camera"
2. **Registry Corruption**: Reinstall OBS Studio with Virtual Camera component
3. **Driver Conflicts**: Update camera drivers in Device Manager
4. **Index Gaps**: Use the diagnostic to find which index OBS is actually at
5. **Backend Issues**: Try MSMF if DSHOW doesn't work (or vice versa)

**Technical Background:**
OBS Virtual Camera uses **DirectShow (DSHOW)** on Windows to register as a virtual device. Some systems may have:
- Gaps in camera index enumeration (0, 1, missing 2, then 3, 4...)
- Incomplete DirectShow filter registration
- Conflicts between DSHOW and Media Foundation (MSMF) backends
- OBS Virtual Camera registered at an unexpected index (>10)

The diagnostic script tests **indices 0-30** with **multiple backends** to find where OBS is actually registered.

## Data Output Format

### Folder Structure
Each test run creates a timestamped folder:
```
logs/
  └── test_20251120_174039/
      ├── boot_log_20251120_174039.csv
      ├── boot_log_20251120_174039.xlsx
      └── captures/
          ├── ANOMALY_vid1_tile_20251120_174045_123.png
          ├── ANOMALY_vid1_full_20251120_174045_123.png
          └── ...
```

### CSV File Format
Each CSV file contains:

**Column Headers:**
- `Timestamp` - ISO format timestamp with milliseconds
- `Video Channel` - Channel number (1-6)
- `Console Serial` - Console serial number (from equipment setup)
- `Scope ID` - Scope ID (from equipment setup)
- `State` - Current state (No Signal, Scope Disconnected, Scope Connected, Other)
- `Elapsed Secs` - Elapsed time from test start (seconds, 2 decimal places)
- `full cycle count` - Complete cycle number for this channel
- `partial cycle count` - Partial cycle number (if applicable)
- `Disconnected Match Distance` - pHash distance to "Scope Disconnected" reference
- `Connected Match Distance` - pHash distance to "Scope Connected" reference
- `Event Type` - Type of event: `test_start`, `state_change`, `status_snapshot`, or `anomaly`

**Data Rows:**
- All state transitions with equipment identification and timing data
- Periodic snapshots (every 1 second) for continuous logging
- Initial "No Signal" state for all channels at test start

**Report Section:**
- Statistical analysis appended at end of CSV
- Per-channel cycle counts and timing analysis
- Reconnection time statistics
- Incomplete cycle detection
- Anomaly summary with per-channel breakdown

### Excel File Format
- Formatted version of CSV with same data
- Proper column widths and formatting
- Numbers formatted as numbers (not strings)

### Anomaly Screenshots
- Automatically captured for every "Other" state detection
- Two images per detection:
  - `ANOMALY_vid{X}_tile_*.png` - Individual channel tile
  - `ANOMALY_vid{X}_full_*.png` - Full annotated frame
- Saved in `captures/` subfolder within test folder

## Project Structure

- `boot_cycle_gui_web-macpc-6ch.py` – Main Flask web app with enhanced 6-channel monitoring
- `build-mac.sh`                   – Script to build standalone macOS executable  
- `build-win.ps1`                  – Script to build standalone Windows executable
- `run_windows.bat`                – Double-click launcher for Windows (auto-setup)
- `run_windows.ps1`                – PowerShell launcher for Windows (color output)  
- `run_mac.sh` / `run_mac.command` – macOS launcher scripts
- `diagnose_flicker_pc.py`         – Comprehensive camera diagnostic tool
- `diagnose_flicker.bat`           – Windows launcher for diagnostic tool
- `art/`                          – Reference images for state detection (Scope Connected/Disconnected)
- `templates/`                     – HTML templates for web interface
- `logs/`                         – Test output folder (CSV, Excel, screenshots)
- `requirements.txt`               – Python dependencies
- `.gitignore`                    – Git ignore configuration

## Technical Details

### Video Channel Mapping
- **Videos 1-2**: Console 1 (HDMI1, HDMI2 outputs)
- **Videos 3-4**: Console 2 (HDMI1, HDMI2 outputs)  
- **Videos 5-6**: Console 3 (HDMI1, HDMI2 outputs)

### State Detection Algorithm
- **Perceptual Hashing (pHash)**: Compares frames to reference images using visual feature matching
- **ROI-based Analysis**: Analyzes brightness, white pixel fraction, and mean luminance in a Region of Interest
- **Stabilization Logic**: Requires N consecutive frames + hold time to prevent false positives
- **Multi-reference Matching**: Uses multiple reference images for "Scope Connected" state
- **Adaptive Thresholds**: Automatically adjusts based on reference image characteristics

### Timing Analysis
- Measures elapsed time from Scope Disconnected to Scope Connected
- Tracks complete cycles per channel
- Identifies incomplete cycles and timing irregularities
- Generates statistical reports with averages, min, max, and standard deviation
- Flags outliers (>2 standard deviations from mean)

### Anomaly Detection
- "Other" state detections are automatically flagged as anomalies
- Screenshots captured for every anomaly (tile + full frame)
- Per-channel anomaly counts tracked and reported
- Event type marked as "anomaly" in CSV for easy filtering

## Detection Parameters Explained

The Boot Cycle Logger uses several parameters to fine-tune state detection. These are adjustable in the web interface:

### Core Detection Parameters

#### **Video Source**
- **What it is**: The index of the video capture device (usually `0` or `1`)
- **How it works**: OpenCV uses integer indices to access connected cameras/capture devices
- **Typical values**: `0` (first device), `1` (second device), etc.
- **When to adjust**: If video feed doesn't appear, try different source indices

#### **Backend**
- **What it is**: The video capture API used by OpenCV
- **How it works**: Different operating systems use different video APIs
- **Options**:
  - `auto` - Let OpenCV choose (recommended)
  - `AVFOUNDATION` - macOS native (best for Mac)
  - `MSMF` - Windows Media Foundation (recommended for Windows)
  - `DSHOW` - DirectShow (best for OBS Virtual Camera on Windows)
  - `V4L2` - Video4Linux (for Linux systems)
- **When to adjust**: If video capture fails or performance is poor

#### **FOURCC**
- **What it is**: Four-character code specifying video compression format
- **How it works**: Tells the camera what format to output
- **Options**:
  - `auto` - Let the system choose (recommended)
  - `MJPG` - Motion JPEG (good compression, widely supported)
  - `YUY2` - Uncompressed YUV format (higher bandwidth, better quality)
- **When to adjust**: For better performance or if video appears corrupted

#### **Resolution**
- **What it is**: The video capture resolution
- **How it works**: Higher resolution = more detail but slower processing
- **Options**:
  - `1080p` - 1920×1080 (recommended, best detail)
  - `720p` - 1280×720 (faster, lower detail)
- **When to adjust**: If processing is too slow, use 720p

### State Detection Parameters

#### **Bars Threshold (pHash distance)**
- **What it is**: Maximum perceptual hash distance to classify as "Scope Disconnected"
- **How it works**: Compares current frame to reference "Scope-Disconnected.png" image using perceptual hashing (pHash). Lower values = stricter matching
- **Default**: `10`
- **Range**: 0-20 (typical), where 0 = exact match
- **When to adjust**: 
  - Increase if disconnected state isn't being detected
  - Decrease if false positives occur

#### **Stable Frames Required**
- **What it is**: Number of consecutive frames that must match before confirming state change
- **How it works**: Prevents false transitions from momentary video glitches
- **Default**: `3`
- **Range**: 1-10
- **When to adjust**:
  - Increase if seeing too many rapid state changes
  - Decrease to `1` if you want to capture single-frame flickers (minimum latency)

#### **Hold Time (ms)**
- **What it is**: Minimum time (in milliseconds) a state must persist before logging
- **How it works**: Additional stabilization beyond frame count
- **Default**: `100` ms
- **Range**: 0-1000 ms
- **When to adjust**:
  - Set to `0` for minimum latency (capture single-frame flickers)
  - Increase if rapid state changes are causing noise

#### **ROI White % Override**
- **What it is**: Percentage of ROI (Region of Interest) that must be white/bright to detect "Scope Connected"
- **How it works**: When scope connects, a bright white square appears in the ROI. This sets the minimum percentage of white pixels required
- **Default**: `Auto` (calculated automatically from reference images)
- **Range**: 0-100
- **When to adjust**: If "Scope Connected" isn't being detected reliably, try values like 30-50

#### **ROI Mean Gate Override**
- **What it is**: Minimum average brightness (0-255) of ROI to detect "Scope Connected"
- **How it works**: Analyzes the average pixel brightness in the ROI
- **Default**: `Auto` (calculated automatically from reference images)
- **Range**: 0-255 (128 = medium brightness, 200+ = very bright)
- **When to adjust**: If detection is inconsistent, try setting to 180-220

#### **ROI Inset (px)**
- **What it is**: Number of pixels to shrink the ROI from all sides
- **How it works**: Creates a smaller detection area, ignoring edges
- **Default**: `0` (full ROI)
- **Range**: 0-50 pixels
- **When to adjust**: If edge artifacts cause false detections, try 5-15 pixels

### Understanding Match Distance Values

The CSV logs include "Disconnected Match Distance" and "Connected Match Distance" columns:

- **What they are**: Perceptual hash (pHash) distance values indicating how closely the current frame matches reference images
- **How they work**: 
  - Algorithm compares visual features of current frame to reference images
  - Lower values = closer match
  - `0` = exact match (rare in real-world conditions)
  - `<10` = very close match (typical for correct detection)
  - `>20` = poor match (likely wrong state)
- **Why they matter**: These values help diagnose detection issues. If a state is misclassified, you can review these distances to understand why

### Minimizing Detection Latency

To capture single-frame flickers and minimize latency:
1. Set **Stable Frames Required** to `1`
2. Set **Hold Time (ms)** to `0`
3. Ensure camera buffer size is set to `1` (handled automatically for OBS Virtual Camera)

This configuration will log state changes immediately when detected, with no stabilization delay.

---

## CHANGELOG

## [2025-11-XX] - Latest Updates

### Added
- **Timestamped Test Folders**: Each test run creates a dedicated folder with timestamp
- **Anomaly Detection System**: Automatic screenshot capture for "Other" state detections
- **Anomaly Tracking**: Per-channel anomaly counts and summary in reports
- **Enhanced Logging Frequency**: Reduced snapshot interval to 1 second for continuous logging
- **Organized Output Structure**: CSV, Excel, and screenshots organized in test folders
- **Anomaly Event Type**: "Other" states marked as "anomaly" in CSV for easy filtering

### Changed
- **Snapshot Interval**: Reduced from 5 seconds to 1 second for higher-frequency logging
- **Screenshot Naming**: Anomaly screenshots prefixed with "ANOMALY_" for clarity
- **BARS Detection**: Enhanced with more lenient threshold to prevent misclassification as "Other"
- **Stabilization Logic**: Applied equally to all 6 channels for consistent evaluation

### Fixed
- **Scope Disconnected Misclassification**: Improved BARS detection to never be classified as "Other"
- **Equal Channel Evaluation**: All 6 video regions now use same stabilization requirements
- **CSV Record Count**: Ensured sufficient logging for long-running tests (20,000+ records)

## [2025-01-XX] - Enhanced Edition

### Added
- **Equipment Identification System**: Pre-test modal for entering console serial numbers and scope IDs
- **Live Video on Startup**: Video feed starts immediately when app loads, equipment modal overlays live feed
- **Enhanced CSV Logging**: Comprehensive data with equipment IDs, timing measurements, and cycle counts
- **Automated Report Generation**: Statistical analysis appended directly to CSV files
- **Cross-Platform File Access**: "Go to CSV" button opens file browser and highlights recent CSV
- **Precise Timing Analysis**: Measures elapsed time from Scope Disconnected → Scope Connected
- **Complete Cycle Tracking**: Counts full cycles (No Signal → Disconnected → Connected)
- **State Terminology Update**: Renamed "Interface" to "Scope Connected", "Bars" to "Scope Disconnected"

### Changed
- **Separated Video and Detection**: Video runs continuously, detection starts only when "Start Test" is pressed
- **CSV Structure**: Guaranteed column headers on file creation, report appended at end
- **UI Flow**: Equipment modal → Start Test → End Test → Report display → Go to CSV
- **File Naming**: Updated main script to `boot_cycle_gui_web-macpc-6ch.py`
- **Build Scripts**: Updated to include `art/` and `templates/` folders

### Fixed
- **CSV Headers**: Headers now written immediately when "Start Test" is pressed
- **Report Integration**: Report appended to same CSV file, not separate file
- **Cross-Platform Compatibility**: File browser access works on macOS, Windows, and Linux
- **State Tracking**: Improved per-channel timing and cycle analysis

## [2025-10-14] - Original 6-Channel Version

### Added
- Support for 6 simultaneous video channels arranged in a 3x2 layout with independent ROI detection and state tracking.
- Redesigned UI featuring a 2/3 video display, per-channel status pills, and per-channel state tallies.
- Adjustable ROI brightness threshold and mean gate controls for fine-tuning detection.
- Ability to adjust ROI internal padding/inset for better detection accuracy.
- Live CSV logging per channel with timestamped filenames, reset tallies button, and download button.
- Debug overlay support for visualizing ROI positions on video feeds.

## [2024-06-06] - Core Detection System

### Added
- Support for multiple reference images for the connected state to improve detection accuracy.
- Display of both full frame and 400×400 px ROI thumbnail in the web interface, each at their true aspect ratios.
- Improved backend detection and fallback mechanisms on macOS, including support for Source 2 and AVFoundation video capture backends.
- Correct detection of black screens as the "No Signal" state.

### Changed
- Rewritten ROI handling: ROI is now a fixed 400×400 px region positioned 420 px from the left bottom of the frame.
- Enhanced detection logic to better distinguish between `Device Disconnected`, `Scope Connected`, and `No Signal` states.
- Improved fallback handling for video sources on macOS to increase robustness.

### Fixed
- Bug fixes related to misclassification of the "Other" state.
- Various minor improvements to state transition logging and UI responsiveness.
