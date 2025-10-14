# Boot Cycle Logger – Project Context

## 📌 Purpose
Boot Cycle Logger is a camera-based monitoring tool designed to detect and log boot events of a connected device based on **visual state changes in a live video feed**. It uses perceptual hashing (pHash) and ROI (Region of Interest) analysis to identify three primary states:

- **Device Connected (INTERFACE)** – White 400×400 ROI square appears bottom-aligned in the video feed.
- **Device Not Connected (BARS)** – Side-gutter regions match a “Scope-Disconnected” reference.
- **No Signal (NO_SIGNAL)** – Frame is dark/flat (low mean + std dev).

Each detected state is timestamped and logged to a CSV file.

---

## 📁 Project Structure
```
boot_cycle_logger/
├─ boot_cycle_gui_web-macpc.py   # Main Flask web app & detection logic
├─ templates/
│   └─ index.html                # Front-end web UI
├─ art/                         # Reference images for detection
│   ├─ Boot-Reliabilty-Testing.png
│   ├─ Scope-Disconnected.png
│   └─ Boot-Reliabilty-Testing*.jpeg
├─ logs/                        # CSV log output
└─ run_mac.command             # Startup script for macOS
```

---

## ⚙️ Workflow Overview

1. **Launch** – Run `boot_cycle_gui_web-macpc.py` and open `http://localhost:5055/`.
2. **Video Capture** – OpenCV connects to camera or stream (backend depends on OS).
3. **State Detection** – ROI, pHash, and brightness checks classify each frame.
4. **Logging** – State changes written to timestamped CSV.
5. **Web UI** – Live status, thumbnails, probe tests, CSV download.

---

## 📸 ROI Definition
ROI for “Connected” is defined relative to a 1920×1080 frame:

- **X:** 420 px from left
- **Y:** bottom-aligned, 400 px tall
- **Width:** 400 px
- **Height:** 400 px

ROI scales dynamically with frame resolution.

---

## 🧠 Key Thresholds

| Parameter   | Default | Purpose |
|------------|---------|----------|
| `THRESH`   | 10      | pHash distance threshold |
| `MARGIN`   | 2       | Hysteresis margin |
| `DARK_MEAN`| 22.0    | Luminance threshold |
| `DARK_STD` | 12.0    | Std deviation threshold |
| `STABLE`   | 3       | Frames for stability |
| `HOLD_MS`  | 800     | Minimum state duration |

---

## 🧰 Platform Notes

- **Video source (`SRC`)**: May differ (`0` on macOS/Linux, `1` on Windows).
- **Backend:**  
  - macOS → `cv2.CAP_AVFOUNDATION`  
  - Windows → `cv2.CAP_MSMF` or `cv2.CAP_DSHOW`  
  - Linux → `cv2.CAP_V4L2`
- **Build:** Adjust PyInstaller `--add-data` separator (`;` vs `:`).
- **Paths:** Keep reference images relative to `art/`.
- **ROI Scaling:** Must adjust dynamically per resolution.

---

## 🧪 Troubleshooting

| Symptom | Cause | Fix |
|--------|-------|------|
| Port busy | 5055 already in use | `_free_port()` cleans old instance |
| Always idle | No feed | Check source index/backend |
| Always OTHER | Missing refs or bad ROI | Verify art paths and ROI math |
| “cannot open source” | Wrong backend | Try alternate (`MSMF` ↔ `DSHOW`) |

---
