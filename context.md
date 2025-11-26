# Boot Cycle Logger – Enhanced Edition Project Context

## 📌 Purpose
Boot Cycle Logger is a comprehensive medical equipment monitoring tool designed to track scope connection states across multiple video channels. It uses perceptual hashing (pHash) and ROI (Region of Interest) analysis to identify four primary states:

- **Scope Connected** – Device successfully connected and operational (previously "INTERFACE")
- **Scope Disconnected** – Device disconnected or connection lost (previously "BARS")  
- **No Signal** – Input is black/empty feed (low mean + std dev)
- **Other** – Any unrecognized state

The system monitors 6 video channels simultaneously, tracks equipment identification, measures timing between state transitions, and generates comprehensive reports.

---

## 📁 Project Structure
```
boot_cycle_logger/
├─ boot_cycle_gui_web-macpc-6ch.py   # Enhanced Flask web app with 6-channel monitoring
├─ templates/                        # HTML templates for web interface
│   └─ index.html                   # Main UI with equipment modal and video grid
├─ art/                             # Reference images for state detection
│   ├─ Scope-Disconnected.png       # Reference for disconnected state
│   ├─ Scope-Connected-SidewinderCCU.png  # Reference for Sidewinder CCU
│   └─ Scope-Connected-OtherCCU.png       # Reference for other CCU types
├─ logs/                            # CSV files with test data and reports
├─ build-mac.sh                     # macOS standalone build script
├─ build-win.ps1                    # Windows standalone build script
├─ BootCycleLogger.spec             # PyInstaller specification
├─ requirements.txt                 # Python dependencies
└─ run_mac.command                  # macOS startup script
```

---

## ⚙️ Enhanced Workflow Overview

1. **Launch** – Run `boot_cycle_gui_web-macpc-6ch.py` and open `http://localhost:5055/`
2. **Live Video** – Video feed starts immediately, displays 6-channel 3x2 grid
3. **Equipment Setup** – Modal appears for entering console serials and scope IDs:
   - Console 1: Videos 1 & 2 (HDMI outputs)
   - Console 2: Videos 3 & 4 (HDMI outputs)  
   - Console 3: Videos 5 & 6 (HDMI outputs)
4. **Start Test** – Click to begin detection and CSV logging with headers
5. **Monitoring** – Real-time status pills show current state per channel
6. **End Test** – Stops detection, generates report, appends to CSV
7. **Analysis** – Report displayed on screen, "Go to CSV" opens file browser

---

## 📸 ROI Definition
ROI for "Scope Connected" detection is defined relative to a 1920×1080 frame:

- **X:** 420 px from left
- **Y:** bottom-aligned, 400 px tall  
- **Width:** 400 px
- **Height:** 400 px

ROI scales dynamically with frame resolution and is applied to all 6 video channels independently.

---

## 🧠 Key Thresholds

| Parameter   | Default | Purpose |
|------------|---------|----------|
| `THRESH`   | 10      | pHash distance threshold for state detection |
| `MARGIN`   | 2       | Hysteresis margin for state stability |
| `DARK_MEAN`| 22.0    | Luminance threshold for "No Signal" detection |
| `DARK_STD` | 12.0    | Std deviation threshold for "No Signal" detection |
| `STABLE`   | 3       | Frames required for stable state transition |
| `HOLD_MS`  | 800     | Minimum state duration to prevent false transitions |

## 📊 CSV Output Structure

Each CSV file contains:
- **Headers**: Timestamp, Video Channel, Console Serial, Scope ID, State, Elapsed Secs, Cycle Number, Bars Distance, Interface Distance, Event Type
- **Data Rows**: All state transitions with equipment identification and timing
- **Report Section**: Statistical analysis including:
  - Equipment configuration summary
  - State transition counts per channel
  - Complete cycle counts (No Signal → Disconnected → Connected)
  - Average timing between Scope Disconnected and Scope Connected
  - Incomplete cycle identification and warnings

---

## 🧰 Platform Notes

- **Video source (`SRC`)**: May differ (`0` on macOS/Linux, `1` on Windows).
- **Backend:**  
  - macOS → `cv2.CAP_AVFOUNDATION`  
  - Windows → `cv2.CAP_MSMF` or `cv2.CAP_DSHOW`  
  - Linux → `cv2.CAP_V4L2`
- **Build:** Adjust PyInstaller `--add-data` separator (`;` vs `:`).
- **Paths:** Keep reference images relative to `art/` and templates in `templates/`.
- **ROI Scaling:** Must adjust dynamically per resolution for all 6 channels.
- **File Browser Access:**
  - macOS → `open -R` (reveals and highlights file)
  - Windows → `explorer /select,` (highlights file in Explorer)
  - Linux → `xdg-open` (opens containing directory)

---

## 🧪 Troubleshooting

| Symptom | Cause | Fix |
|--------|-------|------|
| Port busy | 5055 already in use | `_free_port()` cleans old instance |
| Always idle | No feed | Check source index/backend |
| Always OTHER | Missing refs or bad ROI | Verify art paths and ROI math |
| "cannot open source" | Wrong backend | Try alternate (`MSMF` ↔ `DSHOW`) |
| CSV missing headers | Detection not started | Ensure "Start Test" pressed before logging |
| Report not appearing | Test not ended | Click "End Test" to generate report |
| File browser not opening | Platform-specific issue | Check file path and permissions |

## 🔧 Key Enhancements in Enhanced Edition

- **Equipment Tracking**: Console serials and scope IDs mapped to video channels
- **Timing Analysis**: Precise measurement of Scope Disconnected → Connected transitions  
- **Complete Cycle Detection**: Tracks No Signal → Disconnected → Connected sequences
- **Guaranteed CSV Structure**: Headers written on Start Test, report appended on End Test
- **Cross-Platform File Access**: "Go to CSV" works on macOS, Windows, and Linux
- **Separated Video/Detection**: Video runs continuously, detection starts only when needed
- **Enhanced UI Flow**: Equipment modal → Start Test → End Test → Report → Go to CSV

---
