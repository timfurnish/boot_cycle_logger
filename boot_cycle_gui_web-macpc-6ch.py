"""
Boot Cycle Logger - Camera Feed Monitoring and Logging (Web UI)
Adds: live thumbnail (crop), last-seen timestamps, one-click Test Frame.

How to run:
  macOS:   source <venv>/bin/activate && python boot_cycle_gui_web.py
  Windows: <venv>\Scripts\Activate && python boot_cycle_gui_web.py
Then open http://localhost:5055/

Build (Windows ARM64/x64):
  pyinstaller --noconfirm --onefile --noconsole ^
    --name BootCycleLogger ^
    --add-data "Scope-Disconnected.png;." ^
    --add-data "Scope-Connected-SidewinderCCU.png;." ^
    --add-data "Scope-Connected-OtherCCU.png;." ^
    --add-data "templates;templates" ^
    --add-data "art;art" ^
    boot_cycle_gui_web-macpc.py

Build (macOS/Linux):
  pyinstaller --noconfirm --onefile --noconsole \
    --name BootCycleLogger \
    --add-data "Scope-Disconnected.png:." \
    --add-data "Scope-Connected-SidewinderCCU.png:." \
    --add-data "Scope-Connected-OtherCCU.png:." \
    --add-data "templates:templates" \
    --add-data "art:art" \
    boot_cycle_gui_web-macpc.py
"""

import os, sys, threading, time, csv, platform, webbrowser, subprocess
from typing import Optional

# ---- Ensure chosen port is free (best-effort cross‑platform) ----
def _free_port(port:int):
    """
    Try to kill any process that is currently LISTENing on the given TCP port.
    Best-effort, safe to call even if nothing is bound. Intended to clean up
    previously orphaned Boot Cycle Logger instances.
    """
    try:
        if platform.system() in ("Darwin", "Linux"):
            # Use lsof to find PIDs bound to the port, then kill them.
            p = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
            pids = [pid.strip() for pid in (p.stdout or "").splitlines() if pid.strip()]
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], check=False)
                except Exception:
                    pass
        elif platform.system() == "Windows":
            # Use PowerShell to get the owning process of the local listening port and kill it.
            ps = f"(Get-NetTCPConnection -LocalPort {port} -State Listen).OwningProcess"
            p = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
            pids = [pid.strip() for pid in (p.stdout or "").split() if pid.strip().isdigit()]
            for pid in pids:
                try:
                    subprocess.run(["taskkill", "/PID", pid, "/F"], check=False)
                except Exception:
                    pass
    except Exception:
        # As a fallback we do nothing; app.run will error if still occupied.
        pass

# ---- Port selection helpers and global ----
import socket

def _is_port_free(port:int) -> bool:
    """Return True if TCP port is free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            return s.connect_ex(("127.0.0.1", port)) != 0
        except Exception:
            return False

def _choose_port(base:int=5055, upper:int=5070) -> int:
    """
    Try to free and return a port starting at `base`. If still occupied,
    scan upward to `upper` and pick the first free one.
    """
    # First, try to free the base port and reuse it
    _free_port(base)
    if _is_port_free(base):
        return base
    # Otherwise find the next free port
    for p in range(base + 1, upper + 1):
        if _is_port_free(p):
            return p
    # Fallback: return base even if busy (will raise at app.run)
    return base

# Chosen HTTP port (set in __main__)
PORT = 5055
from datetime import datetime
from flask import Flask, jsonify, request, render_template_string, render_template, Response, send_file
import cv2, numpy as np
from PIL import Image
import imagehash as ih

# Video source & capture options (sane defaults, then OS tweaks)
BACKEND    = "auto"     # auto|MSMF|DSHOW|AVFOUNDATION|V4L2
FOURCC     = "auto"     # auto|MJPG|YUY2
RES_PRESET = "1080p"    # 1080p|720p
SRC        = "0"        # default camera index; macOS override below

if platform.system() == "Darwin":
    # OBS/USB capture commonly shows up as index 1 on macOS
    SRC = "1"

if platform.system() == "Windows":
    # These help MSMF behave with MJPG/YUY2 and some H.264 sources
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_H264", "1")

# ---- Resolve app directory for data files (works in PyInstaller and from source) ----
APP_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

TEMPLATE_DIR = os.path.join(APP_DIR, "templates")
STATIC_DIR   = os.path.join(APP_DIR, "static")

# Art folder path (ROI connected reference)
ART_DIR = os.path.join(APP_DIR, "art")

# Logs folder root (works from source and PyInstaller)
LOG_ROOT = os.path.join(APP_DIR, "logs")

# Helper to stringify FOURCC integer to string
def _fourcc_to_str(v:int) -> str:
    try:
        return ''.join([chr((int(v) >> (8*i)) & 0xFF) for i in range(4)])
    except Exception:
        return ''

# ---------- defaults ----------
CENTER_W = 1920
THRESH   = 10
STABLE   = 3
DARK_MEAN= 22.0
DARK_STD = 12.0

# --- 6‑feed grid layout (3x2) ---
GRID_COLS = 3
GRID_ROWS = 2
GRID_FEEDS = GRID_COLS * GRID_ROWS  # 6

# Reference images + CSV
BARS_REF = os.path.join(ART_DIR, "Scope-Disconnected.png")  # keep your disconnected image (now relative to art/)
# ROI connected reference inside /art (used for INTERFACE detection)
INT_REF  = os.path.join(ART_DIR, "Boot-Reliabilty-Testing.png")
CSV_PATH = os.path.join(LOG_ROOT, f"boot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# Additional INT reference discovery (multiple "Boot-Reliabilty-Testing*" variants)
INT_REF_PREFIX = "Boot-Reliabilty-Testing"   # (spelling matches your files)
INT_REF_EXTS   = (".png", ".jpg", ".jpeg", ".bmp", ".webp")



# Anti-flicker defaults (Windows virtual cam can be jittery)
HOLD_MS  = 800   # minimum time a new state must persist before we accept it
MARGIN   = 2     # hysteresis margin for phash distance thresholds

# Throttle how often we update the live crop to avoid pushing detector around
THUMB_EVERY_MS = 1500

# Live thumbnail target size (keep 16:9 to match source aspect)
THUMB_W = 256
THUMB_H = 144

def _placeholder_thumb(width: int = 780, height: int = 288):
    """Generate a neutral placeholder when no frame is available."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # subtle border
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (60, 60, 60), 1)
    try:
        cv2.putText(img, "no frame", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    except Exception:
        pass
    return img

def crop(bgr, cw=CENTER_W):
    h, w, _ = bgr.shape
    # normalize/validate cw
    try:
        if cw is None:
            cw = CENTER_W
        cw = int(cw)
    except Exception:
        cw = CENTER_W

    # If cw is <= 0 or >= frame width, use the full frame (no side-gutter crop).
    if cw <= 0 or cw >= w:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

    # Compute left/right gutter width
    s = (w - cw) // 2
    if s <= 0:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

    # Concatenate only the side gutters, convert to gray, resize to thumbnail
    side = np.concatenate([bgr[:, :s], bgr[:, w - s:]], axis=1)
    gray = cv2.cvtColor(side, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

# ---------- ROI detector for "Device Connected" ----------
# 6‑feed grid ROI: each tile is ~640x540 when the full frame is 1920x1080.
# Target rectangle per tile: x≈140, y≈340 (top-left), w≈133, h≈200.
GRID_BASE_W, GRID_BASE_H = 640, 540
ROI_TILE_BASE = dict(x=140, y=GRID_BASE_H - 200, w=133, h=200)  # 140,340,133,200

ROI_FRAC = dict(
    x = ROI_TILE_BASE["x"] / float(GRID_BASE_W),
    y = ROI_TILE_BASE["y"] / float(GRID_BASE_H),
    w = ROI_TILE_BASE["w"] / float(GRID_BASE_W),
    h = ROI_TILE_BASE["h"] / float(GRID_BASE_H),
)

def _roi_box_for_frame(w:int, h:int, inset:int=0):
    """
    Bottom-anchored ROI for each tile using base pixels:
    x≈140, w≈133 scaled from 640-wide tiles; height≈200 scaled from 540-tall tiles.
    The ROI sits flush on the bottom regardless of rounding differences.
    """
    # Horizontal scale against 640; vertical against 540
    rw = max(1, int(round(ROI_TILE_BASE["w"] * (w / float(GRID_BASE_W)))))
    rh = max(1, int(round(ROI_TILE_BASE["h"] * (h / float(GRID_BASE_H)))))
    x  = int(round(ROI_TILE_BASE["x"] * (w / float(GRID_BASE_W))))
    y  = max(0, h - rh)  # bottom anchor

    # Clamp
    if x + rw > w: x = max(0, w - rw)
    if y + rh > h: y = max(0, h - rh)

    inset = max(0, int(inset))
    if inset:
         rw = max(1, rw - 2*inset)
         rh = max(1, rh - 2*inset)
         x  = min(max(0, x + inset), max(0, w - rw))
         y0 = y + inset  # keep bottom roughly anchored
         y  = min(max(0, y0), max(0, h - rh))
    return x, y, rw, rh

def roi_connected_gray(bgr):
    """Extract the INTERFACE ROI region, grayscale it, return a square for stable pHash."""
    h, w = bgr.shape[:2]
    x, y, rw, rh = _roi_box_for_frame(w, h, getattr(mon, "roi_inset_px", 0))
    roi = bgr[y:y+rh, x:x+rw]
    if roi.size == 0:
        roi = bgr
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

# --- Grid helpers ---
def split_grid(frame, cols=GRID_COLS, rows=GRID_ROWS):
    """Return (tiles, rects). rects are (x,y,w,h) in the original frame."""
    h, w = frame.shape[:2]
    tile_w = w // cols
    tile_h = h // rows
    tiles, rects = [], []
    for r in range(rows):
        for c in range(cols):
            x = c * tile_w
            y = r * tile_h
            w_c = (w - x) if c == cols - 1 else tile_w
            h_r = (h - y) if r == rows - 1 else tile_h
            rects.append((x, y, w_c, h_r))
            tiles.append(frame[y:y+h_r, x:x+w_c])
    return tiles, rects

def draw_status_badge(img, x, y, text, color):
    """Draw a simple status badge rectangle with text."""
    pad_x, pad_y = 10, 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    w = tw + pad_x*2
    h = th + pad_y*2
    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)
    cv2.putText(img, text, (x+pad_x, y+pad_y+th-4), font, scale, (255,255,255), thick, cv2.LINE_AA)
    return (x+w, y+h)

# --- ROI brightness statistics helper ---
def roi_stats(gray):
    """
    Return (mean_luma, std_luma, bright_frac) for an 8‑bit grayscale ROI.
    bright_frac is the fraction of pixels > 240 (very bright/white).
    """
    m = float(np.mean(gray))
    s = float(np.std(gray))
    # fraction of nearly-white pixels
    bright_frac = float((gray > 240).mean())
    return m, s, bright_frac

# --- Compute average ROI stats from reference images ---
def ref_roi_stats_from_paths(paths):
    """
    Compute average ROI brightness stats across all connected reference images.
    Returns (ref_mean, ref_std, ref_bright) or (None, None, None) if none available.
    """
    means, stds, brights = [], [], []
    for p in (paths or []):
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            roi = roi_connected_gray(img)
            m, s, b = roi_stats(roi)
            means.append(m); stds.append(s); brights.append(b)
        except Exception:
            continue
    if not means:
        return None, None, None
    return float(np.mean(means)), float(np.mean(stds)), float(np.mean(brights))

def _effective_cw_for_width(width:int, user_cw:int) -> int:
    """
    Use side-gutter crop for bars even when the UI center width is >= frame width.
    If user_cw is invalid or >= width, fall back to ~60% of the frame width.
    """
    try:
        c = int(user_cw)
    except Exception:
        c = CENTER_W
    if c <= 0 or c >= int(width):
        return max(1, int(round(width * 0.60)))
    return c

def _equalize_hist(gray):
    """Apply histogram equalization to a grayscale image."""
    return cv2.equalizeHist(gray)

def ph_int_ref(path):
    """pHash of the ROI (white 400x400 square area) from a single INT reference image.
    Crops to ROI and applies histogram equalization before hashing.
    """
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read INT reference image: {path}")
    roi_gray = roi_connected_gray(img)
    roi_eq = _equalize_hist(roi_gray)
    return ih.phash(Image.fromarray(roi_eq))

def ph_int_ref_list(paths):
    """pHashes of the ROI from all provided INT reference images (skip unreadable).
    Crops to ROI and applies histogram equalization before hashing.
    """
    hashes = []
    for p in paths or []:
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            roi_gray = roi_connected_gray(img)
            roi_eq = _equalize_hist(roi_gray)
            hashes.append(ih.phash(Image.fromarray(roi_eq)))
        except Exception:
            continue
    return hashes

   

def ph(gray_crop):
    return ih.phash(Image.fromarray(gray_crop))

def ph_ref(path, cw):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read reference image: {path} (cwd={os.getcwd()})")
    ref_w = img.shape[1]
    cw_eff = _effective_cw_for_width(ref_w, cw)
    return ph(crop(img, cw_eff))

# --- Helper: dual bars hashes (side-gutters crop and full-frame) ---

# --- Helper: pHash of the ROI region from the BARS (Scope-Disconnected) reference image ---
def ph_bars_ref_roi(path):
    """pHash of the ROI region taken from the BARS (disconnected) reference image."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read BARS reference image: {path}")
    roi_gray = roi_connected_gray(img)
    roi_eq = _equalize_hist(roi_gray)
    return ih.phash(Image.fromarray(roi_eq))

def discover_int_refs():
    """
    Return a de-duplicated, ordered list of INT reference paths. It prefers the
    canonical INT_REF first and then any files in art/ that start with
    "Boot-Reliabilty-Testing" (any common image extension).
    """
    paths = []
    try:
        if os.path.exists(INT_REF):
            paths.append(INT_REF)
        for name in os.listdir(ART_DIR):
            low = name.lower()
            if low.startswith(INT_REF_PREFIX.lower()) and low.endswith(INT_REF_EXTS):
                p = os.path.join(ART_DIR, name)
                if os.path.exists(p):
                    paths.append(p)
    except Exception:
        pass
    # de-dup while preserving order
    out, seen = [], set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def label_for(state):
    """Map internal states to user-facing labels (used in UI and CSV)."""
    if state == "BARS":
        return "Device Not Connected"
    elif state == "INTERFACE":
        return "Device Connected"
    elif state == "NO_SIGNAL":
        return "No Signal"
    else:
        return "Other"

class Monitor:
    def __init__(self):
        # backend selection (string name; resolved at open time)
        self.backend_name = BACKEND  # "auto", "MSMF", "DSHOW", "AVFOUNDATION", "V4L2"
        self.fourcc = FOURCC
        self.res_preset = RES_PRESET

        # config
        self.src       = SRC
        self.stream_path = ""               # optional URL; overrides src if set
        self.center_w  = CENTER_W
        self.thresh    = THRESH
        self.stable_frames = STABLE
        self.dark_mean = DARK_MEAN
        self.dark_std  = DARK_STD
        # ROI brightness override controls (None => adaptive)
        self.white_frac_gate = None  # fraction 0.0..1.0 or % from UI
        self.mean_gate       = None  # grayscale mean threshold (0..255)
        self.hold_ms   = HOLD_MS
        self.margin    = MARGIN
        self.roi_inset_px = 0  # shrink ROI inward on all sides (px in tile space)
        # live-crop thumbnail throttling
        self.thumb_every_ms = THUMB_EVERY_MS
        self.thumb_enabled  = True
        self._last_thumb_ts = 0.0
        self.bars_ref  = BARS_REF
        self.int_ref   = INT_REF
        self.int_ref_paths = discover_int_refs()
        # int_ref2 removed/not used
        self.csv_path  = CSV_PATH

        # runtime state
        self.lock = threading.Lock()
        self.running = False
        self.status = "idle"
        self.worker = None  # thread running the capture loop
        self.last_cfg = None
        self.count_bars = 0
        self.count_int  = 0
        self.count_other= 0
        self.cycles     = 0
        # per-tile (6-ch) state/counters
        self.await_connect = True
        self.tile_last   = ["UNKNOWN"] * GRID_FEEDS
        self.tile_counts = [0] * GRID_FEEDS  # increments when a tile transitions to INTERFACE

        # stabilization
        self._last = "UNKNOWN"
        self._raw_last = "UNKNOWN"
        self._raw_stable = 0
        self._last_change_ts = 0.0

        # last-seen timestamps
        self.last_seen = {"BARS": None, "INTERFACE": None, "OTHER": None, "NO_SIGNAL": None}

        # live frames/metrics for thumbnail + tester
        self.last_frame = None
        self.last_crop  = None
        self.last_metrics = {"db": None, "di": None, "mean": None, "std": None}
        self.last_grid = None  # cached per-frame grid detections for UI consistency

        # cached refs for quick /thumb grid annotation
        self._bars_h_roi = None
        self._int_h_list = None
        self._ref_mean = None
        self._ref_bright = None
        
        # Console and scope metadata for logging
        # consoles: {1: {serial: str, scope_id: str}, 2: {...}, 3: {...}}
        self.consoles = {}
        
        # Enhanced state tracking per tile
        # tile_state_history: list of {video, timestamp, state, console_serial, scope_id}
        self.tile_state_history = []
        self.test_start_time = None  # timestamp of first state change
        
        # Counters per state type per video channel
        # Format: {video_num: {connected: count, disconnected: count, no_signal: count}}
        self.state_tallies = {i: {"connected": 0, "disconnected": 0, "no_signal": 0} for i in range(1, 7)}

    def _resolve_backend(self):
        name = (self.backend_name or "auto").strip().lower()
        if name == "auto":
            if platform.system() == "Windows":
                return cv2.CAP_MSMF
            elif platform.system() == "Darwin":
                return cv2.CAP_AVFOUNDATION
            else:
                return cv2.CAP_V4L2
        if name == "msmf": return cv2.CAP_MSMF
        if name == "dshow": return cv2.CAP_DSHOW
        if name == "avfoundation": return cv2.CAP_AVFOUNDATION
        if name == "v4l2": return cv2.CAP_V4L2
        # fallback
        if platform.system() == "Windows": return cv2.CAP_MSMF
        if platform.system() == "Darwin": return cv2.CAP_AVFOUNDATION
        return cv2.CAP_V4L2

    def reset_counts_and_roll_csv(self):
        self.count_bars = self.count_int = self.count_other = self.cycles = 0
        self._last = "UNKNOWN"
        self._raw_last = "UNKNOWN"
        self._raw_stable = 0
        self.last_seen = {"BARS": None, "INTERFACE": None, "OTHER": None, "NO_SIGNAL": None}
        # new timestamped CSV on each clear
        self.csv_path = os.path.join(LOG_ROOT, f"boot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.await_connect = True
        # Reset new tracking fields
        self.tile_state_history = []
        self.test_start_time = None
        self.state_tallies = {i: {"connected": 0, "disconnected": 0, "no_signal": 0} for i in range(1, 7)}

    def reset_tallies(self):
        """Zero counters and last-seen timestamps without rolling the CSV file."""
        self.count_bars = 0
        self.count_int = 0
        self.count_other = 0
        self.cycles = 0
        self._last = "UNKNOWN"
        self._raw_last = "UNKNOWN"
        self._raw_stable = 0
        self.last_seen = {"BARS": None, "INTERFACE": None, "OTHER": None, "NO_SIGNAL": None}
        self.await_connect = True
        self.tile_last   = ["UNKNOWN"] * GRID_FEEDS
        self.tile_counts = [0] * GRID_FEEDS
        # Reset new tracking fields
        self.tile_state_history = []
        self.test_start_time = None
        self.state_tallies = {i: {"connected": 0, "disconnected": 0, "no_signal": 0} for i in range(1, 7)}

mon = Monitor()

def open_csv(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if f.tell()==0:
        # Enhanced CSV header with console and scope metadata
        w.writerow(["timestamp","video","state","console_num","console_serial","scope_id","bars_dist","int_dist","count","elapsed_sec"])
    return f, w

def _csv_append_state_change(video:int, state:str, bars_dist:Optional[int]=None, int_dist:Optional[int]=None, count:Optional[int]=None):
    """
    Append a state change event to CSV with full metadata.
    Columns: timestamp, video, state, console_num, console_serial, scope_id, bars_dist, int_dist, count, elapsed_sec
    """
    try:
        os.makedirs(os.path.dirname(mon.csv_path) or ".", exist_ok=True)
        ts = datetime.now()
        timestamp_str = ts.isoformat(timespec="milliseconds")
        
        # Map video to console: Videos 1-2 = Console 1, 3-4 = Console 2, 5-6 = Console 3
        console_num = ((video - 1) // 2) + 1
        console_data = mon.consoles.get(console_num, {})
        console_serial = console_data.get("serial", "")
        scope_id = console_data.get("scope_id", "")
        
        # Calculate elapsed time since test start
        elapsed_sec = 0.0
        if mon.test_start_time:
            elapsed_sec = (ts - mon.test_start_time).total_seconds()
        else:
            mon.test_start_time = ts
            
        # Store in history for report generation
        mon.tile_state_history.append({
            "timestamp": ts,
            "video": video,
            "state": state,
            "console_num": console_num,
            "console_serial": console_serial,
            "scope_id": scope_id,
            "bars_dist": bars_dist,
            "int_dist": int_dist,
            "elapsed_sec": elapsed_sec
        })
        
        with open(mon.csv_path, "a", newline="") as _f:
            _w = csv.writer(_f)
            _w.writerow([timestamp_str, video, state, console_num, console_serial, scope_id, 
                        bars_dist, int_dist, count, round(elapsed_sec, 3)])
    except Exception as e:
        print(f"Error appending to CSV: {e}")
        pass

def _composite_thumb_from_frame(bgr):
    """
    Composite debug view: ROI (384x384, left) + full frame (native aspect, 384px tall, right) with labels.
    ROI is a 1:1 square, 384x384. Full-frame is resized to height 384, width = int(384 * (w/h)).
    There is a 12px gap between panels.
    """
    ROI_SIZE = 384  # ROI panel size (square)
    GAP = 12
    # Get original frame size
    h, w = bgr.shape[:2]
    # Compute full-frame preview width to maintain aspect ratio at height=384
    full_h = ROI_SIZE
    full_w = int(round(full_h * (w / h)))
    # Output canvas size
    OUT_W = ROI_SIZE + GAP + full_w
    OUT_H = ROI_SIZE

    # Build ROI (384x384)
    try:
        roi_gray = roi_connected_gray(bgr)
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        roi_rgb = cv2.resize(roi_rgb, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
    except Exception:
        roi_rgb = np.zeros((ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)

    left_panel = roi_rgb

    # Build full-frame (native aspect, 384px tall)
    try:
        right_panel = cv2.resize(bgr, (full_w, full_h), interpolation=cv2.INTER_AREA)
    except Exception:
        right_panel = np.zeros((full_h, full_w, 3), dtype=np.uint8)

    # Compose output canvas
    comp = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    # Place ROI at (0,0)
    comp[0:ROI_SIZE, 0:ROI_SIZE] = left_panel
    # Place gap (GAP px wide, black, nothing needed)
    # Place full-frame at (ROI_SIZE + GAP, 0)
    comp[0:full_h, ROI_SIZE + GAP:ROI_SIZE + GAP + full_w] = right_panel

    # Draw outer border
    cv2.rectangle(comp, (0, 0), (OUT_W - 1, OUT_H - 1), (60, 60, 60), 1)

    # Labels (ROI: top left, Full: above full-frame panel)
    try:
        cv2.putText(comp, "ROI", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        # Place "Full" label above the full-frame panel, left-aligned to its panel
        full_label_x = ROI_SIZE + GAP + 16
        full_label_y = 32
        cv2.putText(comp, "Full", (full_label_x, full_label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    except Exception:
        pass

    return comp

def _mac_try_indices_for_nonblack(indices, backend, want_w=1920, want_h=1080, fourcc_sel="auto"):
    """
    macOS helper: try several AVFoundation indices and return the first capture
    that both opens and yields a non‑black frame (mean luminance > 5).
    Returns (cap, used_index) or (None, None).
    """
    for idx in indices:
        try:
            cap = cv2.VideoCapture(idx, backend)
            if not cap or not cap.isOpened():
                try:
                    if cap: cap.release()
                except Exception:
                    pass
                continue
            try: cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            except Exception: pass
            if fourcc_sel == "MJPG":
                try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception: pass
            elif fourcc_sel == "YUY2":
                try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
                except Exception: pass
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  want_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
            except Exception:
                pass
            # warm up
            for _ in range(4):
                cap.read()
            ok, frame = cap.read()
            if not ok or frame is None:
                try:
                    cap.release()
                except Exception:
                    pass
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(np.mean(gray)) > 5.0:
                print(f"[mac_try] selected AVFoundation index={idx}")
                return cap, idx
            # black feed: try next
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            try:
                if cap: cap.release()
            except Exception:
                pass
            continue
    return None, None

def decide(
    frame,
    bars_h_roi,
    int_h_list,
    center_w,
    thresh,
    dmean,
    dstd,
    margin,
    ref_mean=None,
    ref_bright=None,
    white_frac_override=None,
    mean_gate_override=None,
):
    """
    ROI-only decision with a stricter "white ROI" gate:
      - INTERFACE: (pHash near any Boot-Reliabilty-Testing* ref) AND (ROI is truly white)
                   OR (ROI passes a strong "white ROI" brightness gate)
      - BARS     : ROI pHash near the BARS (Scope-Disconnected) ROI
      - NO_SIGNAL: dark/flat full frame (suppressed if ROI is clearly white)
      - OTHER    : fallback
    Returns:
        (det, db, di, cg, mean_lum, std_lum, roi_mean, roi_bright, bright_ok, white_frac_gray, white_frac_rgb)
        - bright_ok is True only when the ROI is sufficiently white (see gates below).
    """
    # --- Compute ROI (gray and equalized) for pHash ---
    # Smooth slightly to reduce pixel-level noise before stats/pHash.
    roi_plain = roi_connected_gray(frame)
    roi_smooth = cv2.GaussianBlur(roi_plain, (5, 5), 0)
    roi_eq = _equalize_hist(roi_smooth)

    # Also get the color ROI to verify "whiteness" (avoid false positives on colored blocks)
    h, w = frame.shape[:2]
    rx, ry, rw, rh = _roi_box_for_frame(w, h, getattr(mon, "roi_inset_px", 0))
    roi_bgr = frame[ry:ry + rh, rx:rx + rw] if rh > 0 and rw > 0 else frame

    phv_roi_plain = ih.phash(Image.fromarray(roi_plain))
    phv_roi_eq = ih.phash(Image.fromarray(roi_eq))

    # Distance to all INT references (use both plain and equalized to be robust)
    di_list = []
    for hsh in (int_h_list or []):
        try:
            di_list.append(int(phv_roi_eq - hsh))
        except Exception:
            pass
        try:
            di_list.append(int(phv_roi_plain - hsh))
        except Exception:
            pass
    di = min(di_list or [999])

    # Distance to BARS ref (ROI)
    try:
        db = int(phv_roi_eq - bars_h_roi)
    except Exception:
        db = 999

    # --- Luminance stats (on the smoothed ROI) ---
    roi_mean, roi_std, roi_bright = roi_stats(roi_smooth)
    full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(full_gray))
    std_lum = float(np.std(full_gray))
    is_dark_flat = (mean_lum < dmean) and (std_lum < dstd)

    # --- Optional: try a slightly tighter ROI if the current mean is a bit low ---
    # This helps when the nominal rectangle grabs a sliver of the black tile border.
    try:
        if roi_mean < 195.0:
            h0, w0 = frame.shape[:2]
            rx, ry, rw, rh = _roi_box_for_frame(w0, h0, getattr(mon, "roi_inset_px", 0) + 8)
            alt = frame[ry:ry + rh, rx:rx + rw] if rh > 0 and rw > 0 else frame
            alt_gray = cv2.cvtColor(alt, cv2.COLOR_BGR2GRAY)
            alt_smooth = cv2.GaussianBlur(alt_gray, (5, 5), 0)
            alt_mean, alt_std, alt_bright = roi_stats(alt_smooth)
            if alt_mean > roi_mean:  # keep the "cleaner" ROI if it improved
                roi_smooth = alt_smooth
                roi_mean, roi_std, roi_bright = alt_mean, alt_std, alt_bright
                roi_bgr = alt
    except Exception:
        pass

    # --- Adaptive "white ROI" gate (tolerant to near‑white) ---
    base_bright_frac = 0.05  # default fraction of near‑white pixels
    base_mean_gate = 65.0    # default minimum ROI mean
    if ref_mean is not None and ref_bright is not None:
        # Soften gates relative to references to allow real-world tolerance.
        base_bright_frac = max(0.02, float(ref_bright) * 0.8)
        base_mean_gate   = max(50.0, float(ref_mean) - 10.0)

    # Optional UI overrides
    if white_frac_override is not None:
        try:
            wf = float(white_frac_override)
            if wf > 1.0:  # treat as percent
                wf = wf / 100.0
            base_bright_frac = max(0.0, min(1.0, wf))
        except Exception:
            pass
    if mean_gate_override is not None:
        try:
            base_mean_gate = float(mean_gate_override)
        except Exception:
            pass

    # Channel thresholds for "near‑white"
    rgb_thresh  = 235
    gray_thresh = 230

    # RGB near‑white fraction: all three channels above threshold.
    try:
        rgb_mask = (
            (roi_bgr[:, :, 0] >= rgb_thresh)
            & (roi_bgr[:, :, 1] >= rgb_thresh)
            & (roi_bgr[:, :, 2] >= rgb_thresh)
        )
        white_frac_rgb = float(np.mean(rgb_mask))
    except Exception:
        white_frac_rgb = 0.0

    # Grayscale near‑white fraction on smoothed ROI
    try:
        white_frac_gray = float(np.mean(roi_smooth >= gray_thresh))
    except Exception:
        white_frac_gray = 0.0

    # Combine signals: accept if either RGB or Gray fraction is sufficient AND mean is high enough.
    frac_gate_rgb  = max(0.015, 0.6 * float(base_bright_frac))
    frac_gate_gray = max(0.03,  0.8 * float(base_bright_frac))
    strong_white = (roi_mean >= base_mean_gate) and (
        (white_frac_rgb  >= frac_gate_rgb) or
        (white_frac_gray >= frac_gate_gray)
    )

    # --- pHash gates (computed above) ---
    bars_gate = thresh + margin
    int_gate  = thresh + margin + 2
    phash_ok  = (di <= int_gate)

    # --- Looser gray-interface allowance ---
    # If the ROI is bright-ish (but not fully white) and the pHash matches,
    # we still consider it connected. This is what lets ~211 means pass.
    loose_gray_gate = max(180.0, float(base_mean_gate) - 30.0)
    gray_ok = (roi_mean >= loose_gray_gate) and phash_ok

    # Final brightness decision used by the UI overlay
    bright_ok = bool(strong_white or gray_ok)

    # --- Decision ---
    if is_dark_flat and not (gray_ok or strong_white):
        det = "NO_SIGNAL"
    elif phash_ok and (strong_white or gray_ok):
        det = "INTERFACE"
    elif strong_white:
        det = "INTERFACE"
    elif db < bars_gate:
        det = "BARS"
    else:
        det = "OTHER"

    # Provide side-gutter crop for legacy metrics/CSV
    cw_eff_runtime = _effective_cw_for_width(frame.shape[1], center_w)
    cg = crop(frame, cw_eff_runtime)

    return (det, db, di, cg, mean_lum, std_lum,
        float(roi_mean), float(roi_bright), bool(bright_ok),
        float(white_frac_gray), float(white_frac_rgb))

def run_loop():
    mon.worker = threading.current_thread()
    try:
        bars_h_roi = ph_bars_ref_roi(mon.bars_ref)
        int_h_list = ph_int_ref_list(mon.int_ref_paths) or [ph_int_ref(mon.int_ref)]
        ref_mean, ref_std, ref_bright = ref_roi_stats_from_paths(mon.int_ref_paths)
        # cache for UI grid annotations
        with mon.lock:
            mon._bars_h_roi = bars_h_roi
            mon._int_h_list = int_h_list
            mon._ref_mean = ref_mean
            mon._ref_bright = ref_bright
    except Exception as e:
        with mon.lock:
            mon.status = f"error: {e}"
            mon.running = False
        return

    # open source
    use_stream = isinstance(mon.stream_path, str) and mon.stream_path.strip() != ""
    backend = mon._resolve_backend()
    cap = None

    def _open_with_backend(_src, _backend):
        if isinstance(_src, str) and not _src.isdigit():
            return cv2.VideoCapture(_src)  # URL or file path
        try:
            idx = int(_src)
        except Exception:
            idx = _src
        return cv2.VideoCapture(idx, _backend)

    if use_stream:
        cap = cv2.VideoCapture(mon.stream_path.strip())
    else:
        # try selected backend, then fallback (Windows: MSMF<->DSHOW)
        cap = _open_with_backend(mon.src, backend)
        if (not cap) or (not cap.isOpened()):
            if platform.system() == "Windows":
                alt_backend = cv2.CAP_DSHOW if backend == cv2.CAP_MSMF else cv2.CAP_MSMF
                try:
                    if cap: cap.release()
                except Exception:
                    pass
                cap = _open_with_backend(mon.src, alt_backend)
            elif platform.system() == "Darwin":
                # Try several AVFoundation indices and pick the first non‑black feed.
                try:
                    if cap: cap.release()
                except Exception:
                    pass
                fourcc_sel = (mon.fourcc or "auto").upper()
                want_w, want_h = (1280, 720) if (mon.res_preset or "1080p").lower()=="720p" else (1920, 1080)
                # Prefer 1, then 0, then a few more just in case
                cap, used_idx = _mac_try_indices_for_nonblack([1,0,2,3,4,5], cv2.CAP_AVFOUNDATION, want_w, want_h, fourcc_sel)
                if cap is None:
                    # last ditch: open the requested index even if black, so UI can still adjust
                    try:
                        cap = cv2.VideoCapture(int(mon.src), cv2.CAP_AVFOUNDATION)
                        used_idx = int(mon.src)
                    except Exception:
                        cap = None
                        used_idx = None
                if used_idx is not None:
                    print(f"[run_loop] macOS selected index {used_idx} (AVFOUNDATION)")
    # Normalize capture on Windows/OBS to reduce flicker and colorspace issues
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass
    try:
        # Respect user-selected FOURCC, or leave as driver default when "auto"
        fc = (mon.fourcc or "auto").upper()
        if fc == "MJPG":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        elif fc == "YUY2":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
        # else: auto -> do not force FOURCC
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    # Apply resolution preset (defaults to 1080p)
    try:
        rp = (mon.res_preset or "1080p").lower()
        if rp == "720p":
            w, h = 1280, 720
        else:
            w, h = 1920, 1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    except Exception:
        pass
    if not cap.isOpened():
        with mon.lock:
            mon.status = "error: cannot open source"
            mon.running = False
        return
    # ensure we can read; bail early if not (retry once on Windows with alt backend)
    ok_probe, _ = cap.read()
    if (not ok_probe) and platform.system()=="Windows":
        try:
            cap.release()
        except Exception:
            pass
        alt_backend = cv2.CAP_DSHOW if backend == cv2.CAP_MSMF else cv2.CAP_MSMF
        cap = _open_with_backend(mon.src, alt_backend)
        ok_probe, _ = cap.read()
    if not ok_probe:
        with mon.lock:
            mon.status = "error: cannot read from source"
            mon.running = False
        cap.release()
        return

    # Warm-up: drop a few frames so phash sees stable content
    for _ in range(8):
        cap.read()
        time.sleep(0.01)

    f, w = open_csv(mon.csv_path)
    with mon.lock:
        mon.status = "running"

    try:
        while True:
            with mon.lock:
                if not mon.running:
                    break
                cw, thr, st = mon.center_w, mon.thresh, mon.stable_frames
                dmean, dstd = mon.dark_mean, mon.dark_std
                # get anti-flicker tunables from monitor instance
                hold_ms = mon.hold_ms
                margin  = mon.margin

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            det, db, di, crop_img, mean_l, std_l, roi_mean, roi_bright, bright_ok, _wfg, _wfr = decide(
                frame, bars_h_roi, int_h_list, cw, thr, dmean, dstd, margin, ref_mean, ref_bright,
                white_frac_override=getattr(mon, "white_frac_gate", None),
                mean_gate_override=getattr(mon, "mean_gate", None)
            )

            with mon.lock:
                # update live artifacts
                mon.last_frame = frame
                # Throttle live-crop updates so UI refreshes don't perturb timing on slower systems
                now_sec = time.time()
                if mon.thumb_enabled and ((now_sec - mon._last_thumb_ts) * 1000.0 >= mon.thumb_every_ms):
                    # store a copy to decouple from OpenCV buffer reuse
                    mon.last_crop = crop_img.copy()
                    mon._last_thumb_ts = now_sec
                mon.last_metrics = {
                    "db": db, "di": di,
                    "mean": round(mean_l, 2), "std": round(std_l, 2),
                    "roi_mean": round(float(roi_mean), 1),
                    "roi_bright": round(float(roi_bright), 3),
                    "bright_ok": bool(bright_ok)
                }

                # stabilize
                if det == mon._raw_last:
                    mon._raw_stable += 1
                else:
                    # _raw_last stores the most recent raw detection result to track stability across frames.
                    mon._raw_last = det
                    mon._raw_stable = 1

                changed = False
                # Hysteresis: require the winning class to be clearly inside threshold by a margin
                det_ok = False
                if det == "BARS":
                    det_ok = (db < (thr + mon.margin))
                elif det == "INTERFACE":
                    # Accept either a close pHash OR a strong-bright ROI flag
                    det_ok = (di < (thr + mon.margin)) or bright_ok
                else:
                    det_ok = True  # OTHER/NO_SIGNAL has no phash gate

                now_ts = time.time()
                long_enough = (now_ts - mon._last_change_ts) * 1000.0 >= mon.hold_ms

                if mon._raw_stable >= st and det != mon.status and det_ok and long_enough:
                    prev = mon.status
                    mon.status = det
                    mon._last_change_ts = now_ts

                    # bump counters, track cycles, timestamps
                    now = datetime.now().isoformat(timespec="seconds")
                    if det == "BARS":
                        mon.count_bars += 1
                        mon.await_connect = True  # any non-connected period arms the cycle
                    elif det == "INTERFACE":
                        mon.count_int  += 1
                        # Count a cycle when we reach INTERFACE after any non-connected period
                        if mon.await_connect:
                            mon.cycles += 1
                            mon.await_connect = False
                    else:
                        mon.count_other += 1
                        mon.await_connect = True  # arm cycle when leaving connected state

                    mon.last_seen[det] = now
                    mon._last = det

                    # log row
                    changed = True

            if not mon.running:
                break
            if not changed:
                time.sleep(0.05)
    finally:
        cap.release()
        f.close()
        with mon.lock:
            if mon.running:
                mon.status = "stopped"
                mon.running = False
            mon.worker = None

def generate_test_report():
    """
    Generate a comprehensive test report with:
    - State change tallies per video
    - Average disconnect-to-connect times
    - Inconsistency detection (disconnects not followed by connects)
    Returns: dict with report data
    """
    report = {
        "test_start": mon.test_start_time.isoformat() if mon.test_start_time else "N/A",
        "test_end": datetime.now().isoformat(),
        "total_duration_sec": 0.0,
        "videos": {},
        "inconsistencies": [],
        "summary": {}
    }
    
    if mon.test_start_time:
        report["total_duration_sec"] = (datetime.now() - mon.test_start_time).total_seconds()
    
    # Analyze each video channel
    for video_num in range(1, 7):
        console_num = ((video_num - 1) // 2) + 1
        console_data = mon.consoles.get(console_num, {})
        
        video_report = {
            "video": video_num,
            "console_num": console_num,
            "console_serial": console_data.get("serial", "N/A"),
            "scope_id": console_data.get("scope_id", "N/A"),
            "tallies": mon.state_tallies.get(video_num, {"connected": 0, "disconnected": 0, "no_signal": 0}),
            "disconnect_to_connect_times": [],
            "avg_disconnect_to_connect_sec": None
        }
        
        # Find disconnect-to-connect times for this video
        video_history = [h for h in mon.tile_state_history if h["video"] == video_num]
        
        last_disconnect_time = None
        for i, event in enumerate(video_history):
            if event["state"] == "Scope Disconnected":
                last_disconnect_time = event["timestamp"]
                
                # Check if next event is NOT a connect (inconsistency)
                if i + 1 < len(video_history):
                    next_event = video_history[i + 1]
                    if next_event["state"] != "Scope Connected":
                        report["inconsistencies"].append({
                            "video": video_num,
                            "console_serial": event["console_serial"],
                            "scope_id": event["scope_id"],
                            "disconnect_time": event["timestamp"].isoformat(),
                            "next_state": next_event["state"],
                            "issue": f"Disconnected at {event['timestamp'].strftime('%H:%M:%S.%f')[:-3]} was followed by '{next_event['state']}' instead of 'Scope Connected'"
                        })
                elif i + 1 >= len(video_history):
                    # Last event was disconnect with no subsequent connect
                    report["inconsistencies"].append({
                        "video": video_num,
                        "console_serial": event["console_serial"],
                        "scope_id": event["scope_id"],
                        "disconnect_time": event["timestamp"].isoformat(),
                        "next_state": "END OF TEST",
                        "issue": f"Disconnected at {event['timestamp'].strftime('%H:%M:%S.%f')[:-3]} was never followed by reconnection"
                    })
                    
            elif event["state"] == "Scope Connected" and last_disconnect_time:
                # Calculate time from disconnect to connect
                delta_sec = (event["timestamp"] - last_disconnect_time).total_seconds()
                video_report["disconnect_to_connect_times"].append(delta_sec)
                last_disconnect_time = None
        
        # Calculate average
        if video_report["disconnect_to_connect_times"]:
            video_report["avg_disconnect_to_connect_sec"] = sum(video_report["disconnect_to_connect_times"]) / len(video_report["disconnect_to_connect_times"])
        
        report["videos"][video_num] = video_report
    
    # Overall summary
    total_connected = sum(v["tallies"]["connected"] for v in report["videos"].values())
    total_disconnected = sum(v["tallies"]["disconnected"] for v in report["videos"].values())
    total_no_signal = sum(v["tallies"]["no_signal"] for v in report["videos"].values())
    
    all_disconnect_times = []
    for v in report["videos"].values():
        all_disconnect_times.extend(v["disconnect_to_connect_times"])
    
    report["summary"] = {
        "total_state_changes": len(mon.tile_state_history),
        "total_connected_events": total_connected,
        "total_disconnected_events": total_disconnected,
        "total_no_signal_events": total_no_signal,
        "total_inconsistencies": len(report["inconsistencies"]),
        "avg_all_disconnect_to_connect_sec": sum(all_disconnect_times) / len(all_disconnect_times) if all_disconnect_times else None
    }
    
    return report

def format_report_text(report):
    """Format the report as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("BOOT CYCLE TEST REPORT")
    lines.append("=" * 80)
    lines.append(f"Test Start:    {report['test_start']}")
    lines.append(f"Test End:      {report['test_end']}")
    lines.append(f"Duration:      {report['total_duration_sec']:.1f} seconds ({report['total_duration_sec']/60:.1f} minutes)")
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 80)
    s = report["summary"]
    lines.append(f"Total State Changes:        {s['total_state_changes']}")
    lines.append(f"Total Connected Events:     {s['total_connected_events']}")
    lines.append(f"Total Disconnected Events:  {s['total_disconnected_events']}")
    lines.append(f"Total No Signal Events:     {s['total_no_signal_events']}")
    lines.append(f"Total Inconsistencies:      {s['total_inconsistencies']}")
    if s['avg_all_disconnect_to_connect_sec']:
        lines.append(f"Avg Disconnect→Connect:     {s['avg_all_disconnect_to_connect_sec']:.2f} seconds")
    lines.append("")
    
    # Per-video details
    lines.append("PER-VIDEO DETAILS")
    lines.append("-" * 80)
    for video_num in sorted(report["videos"].keys()):
        v = report["videos"][video_num]
        lines.append(f"\nVIDEO {video_num} (Console {v['console_num']})")
        lines.append(f"  Console Serial: {v['console_serial']}")
        lines.append(f"  Scope ID:       {v['scope_id']}")
        lines.append(f"  Connected:      {v['tallies']['connected']} times")
        lines.append(f"  Disconnected:   {v['tallies']['disconnected']} times")
        lines.append(f"  No Signal:      {v['tallies']['no_signal']} times")
        if v['avg_disconnect_to_connect_sec']:
            lines.append(f"  Avg Disc→Conn:  {v['avg_disconnect_to_connect_sec']:.2f} seconds")
        else:
            lines.append(f"  Avg Disc→Conn:  N/A (no complete cycles)")
    
    # Inconsistencies
    if report["inconsistencies"]:
        lines.append("")
        lines.append("INCONSISTENCIES DETECTED")
        lines.append("-" * 80)
        for i, inc in enumerate(report["inconsistencies"], 1):
            lines.append(f"\n{i}. VIDEO {inc['video']} (Console Serial: {inc['console_serial']}, Scope: {inc['scope_id']})")
            lines.append(f"   {inc['issue']}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)

# ---------- web app ----------
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.get("/")
def index():
    # Always serve the modern 2/3 (video) + 1/3 (controls) layout with a 6‑pill status grid.
    # We deliberately do NOT render templates/index.html anymore to avoid old UI showing up.
    return """<!doctype html><html><head><meta charset='utf-8'><title>Boot Cycle Logger 6‑ch</title>
    <style>
    /* Modal styles */
    .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; 
             background-color: rgba(0,0,0,0.6); overflow: auto; }
    .modal.show { display: block; }
    .modal-content { background-color: var(--panel); margin: 5% auto; padding: 30px; border: 1px solid var(--border); 
                     width: 90%; max-width: 600px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
    .modal-header { font-size: 24px; font-weight: 700; color: var(--btn); margin-bottom: 20px; }
    .modal-close { color: var(--muted); float: right; font-size: 28px; font-weight: bold; cursor: pointer; }
    .modal-close:hover { color: var(--text); }
    .console-section { margin-bottom: 24px; padding: 16px; background: var(--bg); border-radius: 8px; border: 1px solid var(--border); }
    .console-title { font-size: 16px; font-weight: 700; color: var(--btn); margin-bottom: 12px; }
    .form-row { margin-bottom: 12px; }
    .form-row label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .form-row input { width: 100%; padding: 8px; border-radius: 6px; background: var(--panel); 
                      color: var(--text); border: 1px solid var(--border); box-sizing: border-box; }
    .report-text { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; 
                   white-space: pre-wrap; background: var(--bg); padding: 16px; border-radius: 8px; 
                   max-height: 500px; overflow-y: auto; color: var(--text); border: 1px solid var(--border); }
    </style>
    <style>
    :root{
      --bg:#0f172a; --panel:#0b1220; --border:#1f2937; --text:#e5e7eb; --muted:#94a3b8;
      --ok:#16a34a; --bad:#dc2626; --nosig:#6b7280; --other:#0ea5e9; --btn:#38bdf8; --btn2:#1f2937;
    }
    html,body{height:100%}
    body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
    .wrap{display:grid;grid-template-columns: 2fr 1fr; min-height:100vh;}
    .left{padding:16px;background:var(--panel);border-left:1px solid var(--border); min-width:240px; max-width:420px;}
    .right{display:flex;flex-direction:column;gap:18px;padding:24px 18px 18px 36px;justify-content:flex-start;}
    h1{margin:4px 0 12px 0;font-size:20px;color:#38bdf8}
    label{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}
    .row{margin-bottom:12px}
    button{background:var(--btn);color:#0b1220;font-weight:700;border:none;padding:10px 14px;border-radius:8px;cursor:pointer}
    button.secondary{background:var(--btn2);color:var(--text)}
    input,select{width:100%;padding:8px;border-radius:6px;background:var(--panel);color:var(--text);border:1px solid var(--border)}
    img.source{width:100%;max-width:100%;height:auto;display:block;border:1px solid var(--border);border-radius:10px;background:#000;aspect-ratio:16/9;object-fit:contain}
    .pillgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:repeat(2,1fr);gap:12px;margin-top:16px;width:100%}
    .pill{display:flex;align-items:center;justify-content:center;height:44px;border-radius:999px;font-weight:800;border:1px solid #00000022;box-shadow:0 6px 20px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.08);min-width:0}
    .ok{background:var(--ok)} .bad{background:var(--bad)} .nosig{background:var(--nosig)} .other{background:var(--other)}
    .pill span{color:white;white-space:nowrap;text-overflow:ellipsis;overflow:hidden;}
    pre{font:12px ui-monospace,Menlo,Consolas,monospace;color:var(--muted);white-space:pre-wrap}
    @media (max-width: 900px){.wrap{grid-template-columns:1fr}.left{max-width:none;border-left:none;border-top:1px solid var(--border)}.right{padding:14px}}
    </style></head><body>
    <div class='wrap'>
      <div class='right'>
        <img id='grid' class='source' src='/thumb'>
        <div class='pillgrid'>
          <div id='p1' class='pill'><span>VIDEO 1</span></div>
          <div id='p3' class='pill'><span>VIDEO 3</span></div>
          <div id='p5' class='pill'><span>VIDEO 5</span></div>
          <div id='p2' class='pill'><span>VIDEO 2</span></div>
          <div id='p4' class='pill'><span>VIDEO 4</span></div>
          <div id='p6' class='pill'><span>VIDEO 6</span></div>    
        </div>
      </div>
      <div class='left'>
        <h1>Boot Cycle Logger</h1>
        <div class='row'><label>Source</label><input id='src' value='""" + str(mon.src) + """'></div>
        <div class='row'><label>Backend</label><select id='backend'><option>auto</option><option>MSMF</option><option>DSHOW</option><option>AVFOUNDATION</option><option>V4L2</option></select></div>
        <div class='row'><label>FOURCC</label><select id='fourcc'><option>auto</option><option>MJPG</option><option>YUY2</option></select></div>
        <div class='row'><label>Resolution</label><select id='res'><option>1080p</option><option>720p</option></select></div>
        <div class='row'><label>Bars Threshold</label><input id='thr' value='""" + str(mon.thresh) + """'></div>
        <div class='row'><label>Stable Frames</label><input id='st' value='""" + str(mon.stable_frames) + """'></div>
        <div class='row'><label>ROI white % (0-100, overrides auto)</label><input id='white_pct' value=''></div>
        <div class='row'><label>ROI mean gate (0-255, overrides auto)</label><input id='mean_gate' value=''></div>
        <div class='row'><label>ROI inset px (shrink ROI)</label><input id='roi_inset' value='0'></div>
        <div class='row' style='display:flex;gap:8px;flex-wrap:wrap'>
          <button onclick='showSetupModal()'>Setup Test</button>
          <button onclick='start()'>Start</button>
          <button class='secondary' onclick='stopAndReport()'>Stop & Report</button>
          <button class='secondary' onclick='stop()'>Stop</button>
          <button class='secondary' onclick='showReport()'>View Report</button>
          <button class='secondary' onclick='probe()'>Probe source</button>
          <button class='secondary' onclick='resetTallies()'>Reset tallies</button>
          <a id='dl' class='secondary' href='/download' style='text-decoration:none;display:inline-block;padding:10px 14px;border-radius:8px;'>Download CSV</a>
          <div id='csvInfo' style='font-size:12px;color:#94a3b8;margin-top:6px'>CSV: <code id='csvName'>-</code></div>
        </div>
        <pre id='probeOut'></pre>
      </div>
    </div>
    
    <!-- Setup Modal -->
    <div id='setupModal' class='modal'>
      <div class='modal-content'>
        <span class='modal-close' onclick='closeSetupModal()'>&times;</span>
        <div class='modal-header'>Test Setup - Console & Scope Information</div>
        <p style='color:var(--muted);margin-bottom:20px;'>Enter serial numbers and scope IDs for each console before starting the test.</p>
        
        <div class='console-section'>
          <div class='console-title'>Console 1 (Video 1 & 2)</div>
          <div class='form-row'>
            <label>Console Serial Number</label>
            <input type='text' id='console1_serial' placeholder='Enter console serial number'>
          </div>
          <div class='form-row'>
            <label>Scope ID</label>
            <input type='text' id='console1_scope' placeholder='Enter scope ID'>
          </div>
        </div>
        
        <div class='console-section'>
          <div class='console-title'>Console 2 (Video 3 & 4)</div>
          <div class='form-row'>
            <label>Console Serial Number</label>
            <input type='text' id='console2_serial' placeholder='Enter console serial number'>
          </div>
          <div class='form-row'>
            <label>Scope ID</label>
            <input type='text' id='console2_scope' placeholder='Enter scope ID'>
          </div>
        </div>
        
        <div class='console-section'>
          <div class='console-title'>Console 3 (Video 5 & 6)</div>
          <div class='form-row'>
            <label>Console Serial Number</label>
            <input type='text' id='console3_serial' placeholder='Enter console serial number'>
          </div>
          <div class='form-row'>
            <label>Scope ID</label>
            <input type='text' id='console3_scope' placeholder='Enter scope ID'>
          </div>
        </div>
        
        <button onclick='saveConsoleMetadata()' style='width:100%;margin-top:12px;'>Save and Continue</button>
      </div>
    </div>
    
    <!-- Report Modal -->
    <div id='reportModal' class='modal'>
      <div class='modal-content' style='max-width:900px;'>
        <span class='modal-close' onclick='closeReportModal()'>&times;</span>
        <div class='modal-header'>Test Report</div>
        <div id='reportContent' class='report-text'>Loading report...</div>
        <button onclick='closeReportModal()' class='secondary' style='width:100%;margin-top:12px;'>Close</button>
      </div>
    </div>
    <script>
  // Mapping: tile index (0..5) -> VIDEO # and pill id (1,3,5,2,4,6)
  const TILE_TO_PILL = [1, 3, 5, 2, 4, 6]; // tile index -> VIDEO# / pill id mapping

  function bust(u){ return u + (u.includes('?') ? '&' : '?') + 't=' + Date.now(); }

  async function start(){
    const body = {
      src: document.getElementById('src').value,
      backend: document.getElementById('backend').value,
      fourcc: document.getElementById('fourcc').value,
      res: document.getElementById('res').value,
      thr: parseInt(document.getElementById('thr').value || 10, 10),
      st: parseInt(document.getElementById('st').value || 3, 10),
      white_pct: (function(v){ if(v===''||v==null) return null; v=parseFloat(v); return isNaN(v)?null:v; })(document.getElementById('white_pct').value),
      mean_gate: (function(v){ if(v===''||v==null) return null; v=parseFloat(v); return isNaN(v)?null:v; })(document.getElementById('mean_gate').value),
      roi_inset: (function(v){ if(v===''||v==null) return null; v=parseInt(v,10); return isNaN(v)?null:v; })(document.getElementById('roi_inset').value),
    };
    await fetch('/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
  }

  // Debounce helper
  function debounce(fn, ms){
    let t; return (...args)=>{ clearTimeout(t); t=setTimeout(()=>fn(...args), ms); };
  }

  // Push live threshold adjustments
  async function pushAdjust(){
    const body = {
      white_pct: (function(v){ if(v===''||v==null) return null; v=parseFloat(v); return isNaN(v)?null:v; })(document.getElementById('white_pct').value),
      mean_gate: (function(v){ if(v===''||v==null) return null; v=parseFloat(v); return isNaN(v)?null:v; })(document.getElementById('mean_gate').value),
      roi_inset: (function(v){ if(v===''||v==null) return null; v=parseInt(v,10); return isNaN(v)?null:v; })(document.getElementById('roi_inset').value)
    };
    await fetch('/adjust', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
  }

  const debouncedAdjust = debounce(pushAdjust, 250);
  document.getElementById('white_pct').addEventListener('input', debouncedAdjust);
  document.getElementById('mean_gate').addEventListener('input', debouncedAdjust);
  document.getElementById('roi_inset').addEventListener('input', debouncedAdjust);

  async function stop(){
    await fetch('/stop', {method: 'POST'});
  }

  async function probe(){
    const body = {
      src: document.getElementById('src').value,
      backend: document.getElementById('backend').value,
      fourcc: document.getElementById('fourcc').value,
      res: document.getElementById('res').value
    };
    const r = await fetch('/probe', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    document.getElementById('probeOut').textContent = JSON.stringify(await r.json(), null, 2);
  }

  function setPill(el, det, idx, cnt){
    const map = {
      INTERFACE: ['ok', 'Scope Connected'],
      BARS: ['bad', 'Scope Not Connected'],
      NO_SIGNAL: ['nosig', 'No Signal'],
      OTHER: ['other', 'Other']
    };
    const [cls, label] = map[det] || ['other', 'Other'];
    el.className = 'pill ' + cls;
    const suffix = (typeof cnt === 'number' && cnt > 0) ? ` (${cnt})` : '';
    el.querySelector('span').textContent = 'VIDEO ' + idx + ' — ' + label + suffix;
  }

  async function refreshPills(){
    try {
      const r = await fetch('/grid_status');
      const j = await r.json();
      const arr = j.tiles || [];
      for (let i = 0; i < 6; i++) {
        const videoNum = TILE_TO_PILL[i];            // 1,3,5,2,4,6
        const el = document.getElementById('p' + videoNum);
        if (!el) continue;
        const det = (arr[i] && arr[i].det) || 'OTHER';
        const cnt = (arr[i] && typeof arr[i].cnt === 'number') ? arr[i].cnt : 0;
        setPill(el, det, videoNum, cnt);
      }
    } catch (e) {
      console.error(e);
    }
  }

  // Inserted resetTallies after probe
  async function resetTallies(){
    try{
      await fetch('/reset_tallies', { method: 'POST' });
      await refreshPills();
    }catch(e){
      console.error(e);
    }
  }
  
  // Setup modal functions
  function showSetupModal(){
    // Load existing console data if any
    fetch('/get_console_metadata')
      .then(r => r.json())
      .then(data => {
        const consoles = data.consoles || {};
        for(let i = 1; i <= 3; i++){
          const c = consoles[i] || {};
          document.getElementById('console' + i + '_serial').value = c.serial || '';
          document.getElementById('console' + i + '_scope').value = c.scope_id || '';
        }
      })
      .catch(e => console.error(e));
    
    document.getElementById('setupModal').classList.add('show');
  }
  
  function closeSetupModal(){
    document.getElementById('setupModal').classList.remove('show');
  }
  
  async function saveConsoleMetadata(){
    const consoles = {
      '1': {
        serial: document.getElementById('console1_serial').value,
        scope_id: document.getElementById('console1_scope').value
      },
      '2': {
        serial: document.getElementById('console2_serial').value,
        scope_id: document.getElementById('console2_scope').value
      },
      '3': {
        serial: document.getElementById('console3_serial').value,
        scope_id: document.getElementById('console3_scope').value
      }
    };
    
    try{
      const r = await fetch('/save_console_metadata', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({consoles: consoles})
      });
      const result = await r.json();
      if(result.ok){
        alert('Console metadata saved successfully! You can now start the test.');
        closeSetupModal();
      }else{
        alert('Error saving metadata: ' + (result.error || 'Unknown error'));
      }
    }catch(e){
      alert('Error: ' + e);
      console.error(e);
    }
  }
  
  // Report modal functions
  function showReport(){
    document.getElementById('reportModal').classList.add('show');
    document.getElementById('reportContent').textContent = 'Loading report...';
    
    fetch('/generate_report')
      .then(r => r.json())
      .then(data => {
        if(data.ok){
          document.getElementById('reportContent').textContent = data.report_text;
        }else{
          document.getElementById('reportContent').textContent = 'Error generating report: ' + (data.error || 'Unknown error');
        }
      })
      .catch(e => {
        document.getElementById('reportContent').textContent = 'Error: ' + e;
        console.error(e);
      });
  }
  
  function closeReportModal(){
    document.getElementById('reportModal').classList.remove('show');
  }
  
  async function stopAndReport(){
    try{
      const r = await fetch('/stop_and_report', {method: 'POST'});
      const data = await r.json();
      if(data.ok){
        // Show the report in the modal
        document.getElementById('reportModal').classList.add('show');
        document.getElementById('reportContent').textContent = data.report_text;
      }else{
        alert('Error generating report: ' + (data.error || 'Unknown error'));
      }
    }catch(e){
      alert('Error: ' + e);
      console.error(e);
    }
  }
  
  // Close modals when clicking outside
  window.onclick = function(event){
    const setupModal = document.getElementById('setupModal');
    const reportModal = document.getElementById('reportModal');
    if(event.target == setupModal){
      closeSetupModal();
    }
    if(event.target == reportModal){
      closeReportModal();
    }
  }

  setInterval(() => {
    const im = document.getElementById('grid');
    if (im) im.src = bust('/thumb');
  }, 500);

    setInterval(refreshPills, 800);
    refreshPills();
    
    async function refreshCSVInfo(){
  try{
    const r = await fetch('/status'); const j = await r.json();
    const name = (j.csv || '').split(/[\\/]/).pop() || '-';
    const el = document.getElementById('csvName');
    if (el) el.textContent = name;
    const dl = document.getElementById('dl');
    if (dl) dl.href = '/download';
  }catch(e){}
}
setInterval(refreshCSVInfo, 1200);
refreshCSVInfo();
</script></body></html>"""


# Live adjust endpoint
@app.route("/adjust", methods=["POST"])
def adjust():
    cfg = request.get_json(silent=True) or {}
    with mon.lock:
        if "white_pct" in cfg:
            mon.white_frac_gate = cfg.get("white_pct", mon.white_frac_gate)
        if "mean_gate" in cfg:
            mon.mean_gate = cfg.get("mean_gate", mon.mean_gate)
        if "roi_inset" in cfg:
            try:
                mon.roi_inset_px = int(cfg.get("roi_inset", getattr(mon, "roi_inset_px", 0)) or 0)
            except Exception:
                pass
        current = {
            "white_frac_gate": mon.white_frac_gate,
            "mean_gate": mon.mean_gate,
            "roi_inset": getattr(mon, "roi_inset_px", 0)
        }
    return jsonify(ok=True, **current)

@app.get("/ping")
def ping():
    return Response("pong", mimetype="text/plain")



@app.get("/status")
def status():
    with mon.lock:
        sf = label_for(mon.status) if mon.status in ("BARS","INTERFACE","OTHER","NO_SIGNAL") else mon.status
        return jsonify(
            status=mon.status, status_friendly=sf, running=mon.running,
            count_bars=mon.count_bars, count_int=mon.count_int,
            count_other=mon.count_other, cycles=mon.cycles,
            last_seen=mon.last_seen, backend=mon.backend_name,
            fourcc=mon.fourcc, res_preset=mon.res_preset,
            csv=mon.csv_path, white_frac_gate=mon.white_frac_gate,
            mean_gate=mon.mean_gate,
            roi_inset=getattr(mon, "roi_inset_px", 0)
        )


# Helper: draw annotated 3x2 grid with per-tile status/labels
def _annotated_grid_from_frame(bgr):
    """
    Draw a 3x2 annotated grid on the latest frame:
      - For each tile, run the same ROI-based decision as main loop
      - Paint a status badge (Connected / Not Connected / No Signal / Other)
      - Overlay the tile label: VIDEO 1..6
    This is purely for UI; it does not mutate counters.
    """
    # Ensure we have reference hashes; compute lazily if needed
    with mon.lock:
        bars_h_roi = mon._bars_h_roi
        int_h_list = mon._int_h_list
        ref_mean   = mon._ref_mean
        ref_bright = mon._ref_bright
        cw         = mon.center_w
        thr        = mon.thresh
        dmean      = mon.dark_mean
        dstd       = mon.dark_std
        margin     = mon.margin

    if bars_h_roi is None or not int_h_list:
        try:
            paths = discover_int_refs()
            bars_h_roi = ph_bars_ref_roi(mon.bars_ref)
            int_h_list = ph_int_ref_list(paths) or [ph_int_ref(mon.int_ref)]
            m, s, b = ref_roi_stats_from_paths(paths)
            ref_mean, ref_bright = m, b
            with mon.lock:
                mon._bars_h_roi = bars_h_roi
                mon._int_h_list = int_h_list
                mon._ref_mean = ref_mean
                mon._ref_bright = ref_bright
        except Exception:
            pass

    tiles, rects = split_grid(bgr, GRID_COLS, GRID_ROWS)
    out = bgr.copy()

    # Map row-major tile index -> displayed VIDEO number (1,3,5 on top; 2,4,6 on bottom)
    TILE_TO_VIDEO = [1, 3, 5, 2, 4, 6]

    # Colors (B,G,R)
    C_OK   = ( 22, 163,  74)   # green
    C_BAD  = ( 38,  38, 255)   # red
    C_WARN = ( 18, 182, 252)   # cyan-ish for "Other"
    C_NO   = ( 64,  64,  64)   # gray

    tiles_out = []
    for idx, (tile, (x, y, w, h)) in enumerate(zip(tiles, rects), start=1):
        det, db, di, _cg, mean_l, std_l, roi_mean, roi_bright, bright_ok, white_frac_gray, white_frac_rgb = decide(
            tile, bars_h_roi, int_h_list, cw, thr, dmean, dstd, margin, ref_mean, ref_bright,
            white_frac_override=getattr(mon, "white_frac_gate", None),
            mean_gate_override=getattr(mon, "mean_gate", None)
        )
        tiles_out.append({"det": det, "db": int(db), "di": int(di), "cnt": int(mon.tile_counts[idx-1])})
        # --- Per-tile transition tracking and CSV logging ---
        try:
            tile_idx0 = idx - 1
            prev = mon.tile_last[tile_idx0]
            # Map tile index -> VIDEO number (top row: 1,3,5; bottom row: 2,4,6)
            TILE_TO_VIDEO = [1, 3, 5, 2, 4, 6]
            video_num = TILE_TO_VIDEO[tile_idx0]
            
            # Track all state changes (not just to INTERFACE)
            if det != prev and prev != "UNKNOWN":
                # Map detection states to friendly names
                state_name = "Scope Connected" if det == "INTERFACE" else \
                            "Scope Disconnected" if det == "BARS" else \
                            "No Signal" if det == "NO_SIGNAL" else "Other"
                
                # Update tallies
                if det == "INTERFACE":
                    mon.tile_counts[tile_idx0] += 1
                    mon.state_tallies[video_num]["connected"] += 1
                elif det == "BARS":
                    mon.state_tallies[video_num]["disconnected"] += 1
                elif det == "NO_SIGNAL":
                    mon.state_tallies[video_num]["no_signal"] += 1
                
                # Log to CSV
                _csv_append_state_change(video_num, state_name, int(db), int(di), mon.tile_counts[tile_idx0])
            
            mon.tile_last[tile_idx0] = det
        except Exception as e:
            print(f"Error in tile transition tracking: {e}")
            pass
        if det == "INTERFACE":
            color, text = C_OK, "Scope Connected"
        elif det == "BARS":
            color, text = C_BAD, "Scope Not Connected"
        elif det == "NO_SIGNAL":
            color, text = C_NO, "No Signal"
        else:
            color, text = C_WARN, "Other"

        # draw status badge inside tile
        _ = draw_status_badge(out, x + 10, y + 10, text, color)

        # --- Debug overlay: show parameters and measured ROI values ---
        try:
            ov_frac = getattr(mon, "white_frac_gate", None)
            ov_mean = getattr(mon, "mean_gate", None)
            debug_lines = [
                f"roi_inset={getattr(mon,'roi_inset_px',0)}",
                f"ov_frac={ov_frac}",
                f"ov_mean={ov_mean}",
                f"roi_mean={roi_mean:.1f}",
                f"w_frac_g={white_frac_gray:.3f}",
                f"w_frac_rgb={white_frac_rgb:.3f}",
                f"bright_ok={bright_ok}"
            ]
            tx = x + 10
            ty = y + 40
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            for i, line in enumerate(debug_lines):
                y0 = ty + i * 12
                cv2.putText(out, line, (tx, y0), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
        except Exception:
            pass


        # overlay the tile label centered near top-right of the tile
        label = f"VIDEO {TILE_TO_VIDEO[idx-1]}"
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.0
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            cx = x + w - tw - 14
            cy = y + 18 + th
            cv2.putText(out, label, (cx, cy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        except Exception:
            pass

        # draw the detected ROI box for visual confirmation
        try:
            # map ROI (computed in tile space) back to out image coordinates
            tx, ty, rw, rh = _roi_box_for_frame(w, h, getattr(mon, "roi_inset_px", 0))
            cv2.rectangle(out, (x + tx, y + ty), (x + tx + rw, y + ty + rh), (255, 255, 255), 1)
        except Exception:
            pass

    # Cache the detections snapshot for UI consistency (used by /grid_status)
    try:
        with mon.lock:
            mon.last_grid = {"ts": time.time(), "tiles": tiles_out}
    except Exception:
        pass

    return out

@app.get("/thumb")
def thumb():
    with mon.lock:
        frame = mon.last_frame

    if frame is not None:
        try:
            # Annotate the live 3x2 grid with per‑tile status/labels
            annotated = _annotated_grid_from_frame(frame)
            img = annotated
        except Exception:
            # fall back to simple composite if annotation fails
            img = _composite_thumb_from_frame(frame)
    else:
        img = _placeholder_thumb(1200, 675)

    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return Response(status=500)
    return Response(buf.tobytes(), mimetype="image/jpeg")

# New endpoint: grid_status for per-tile detections (JSON)
@app.get("/grid_status")
def grid_status():
    # First, serve the cached grid computed during /thumb rendering (same frame)
    with mon.lock:
        cached = mon.last_grid
        frame = mon.last_frame
        bars_h_roi = mon._bars_h_roi
        int_h_list = mon._int_h_list
        ref_mean   = mon._ref_mean
        ref_bright = mon._ref_bright
        cw         = mon.center_w
        thr        = mon.thresh
        dmean      = mon.dark_mean
        dstd       = mon.dark_std
        margin     = mon.margin

    if cached and isinstance(cached, dict) and cached.get("tiles"):
        return jsonify(cached)

    # If no cache yet, compute once from the current frame and cache it
    if frame is None:
        return jsonify(tiles=[])

    # Ensure refs
    if bars_h_roi is None or not int_h_list:
        try:
            paths = discover_int_refs()
            bars_h_roi = ph_bars_ref_roi(mon.bars_ref)
            int_h_list = ph_int_ref_list(paths) or [ph_int_ref(mon.int_ref)]
            m, s, b = ref_roi_stats_from_paths(paths)
            ref_mean, ref_bright = m, b
            with mon.lock:
                mon._bars_h_roi = bars_h_roi
                mon._int_h_list = int_h_list
                mon._ref_mean = ref_mean
                mon._ref_bright = ref_bright
        except Exception:
            pass

    tiles, rects = split_grid(frame, GRID_COLS, GRID_ROWS)
    tiles_out = []
    for i, tile in enumerate(tiles):
        det, db, di, _cg, mean_l, std_l, roi_mean, roi_bright, bright_ok, _wfg, _wfr = decide(
            tile, bars_h_roi, int_h_list, cw, thr, dmean, dstd, margin, ref_mean, ref_bright,
            white_frac_override=getattr(mon, "white_frac_gate", None),
            mean_gate_override=getattr(mon, "mean_gate", None)
        )
        try:
            cnt_val = int(mon.tile_counts[i])
        except Exception:
            cnt_val = 0
        tiles_out.append({"det": det, "db": int(db), "di": int(di), "cnt": cnt_val})

    snapshot = {"ts": time.time(), "tiles": tiles_out}
    try:
        with mon.lock:
            mon.last_grid = snapshot
    except Exception:
        pass
    return jsonify(snapshot)

@app.get("/peek")
def peek():
    """Return quick diagnostic metrics or perform a one-shot detection."""
    # If loop is already running and metrics exist, just return them
    with mon.lock:
        m = mon.last_metrics.copy()
        have_recent = mon.last_frame is not None
        src = mon.src
        stream = mon.stream_path
        cw = mon.center_w
        thr = mon.thresh
        dmean = mon.dark_mean
        dstd = mon.dark_std
        bars_path = mon.bars_ref
        int_path = mon.int_ref
        backend_name = mon.backend_name
        fourcc_sel = (mon.fourcc or "auto").upper()
        res_preset = (mon.res_preset or "1080p").lower()

    if have_recent:
        for k in ("db", "di", "mean", "std"):
            if m.get(k) is None:
                m[k] = "-"
        return jsonify(m)

    # Load reference hashes and paths
    try:
        int_ref_paths = discover_int_refs()
        bars_h_roi = ph_bars_ref_roi(bars_path)
        int_h_list = ph_int_ref_list(int_ref_paths) or [ph_int_ref(int_path)]
        ref_mean, ref_std, ref_bright = ref_roi_stats_from_paths(int_ref_paths)
    except Exception as e:
        return jsonify(error=f"ref load: {e}")

    use_stream = isinstance(stream, str) and stream.strip() != ""

    def _open_with_backend(_src, _backend):
        # URL/file path
        if isinstance(_src, str) and not _src.isdigit():
            return cv2.VideoCapture(_src), _src
        # numeric index
        try:
            idx = int(_src)
        except Exception:
            idx = _src
        return cv2.VideoCapture(idx, _backend), idx

    # Resolve backend const
    if (backend_name or "auto").lower() == "auto":
        backend = (
            cv2.CAP_MSMF if platform.system() == "Windows"
            else cv2.CAP_AVFOUNDATION if platform.system() == "Darwin"
            else cv2.CAP_V4L2
        )
    else:
        name = backend_name.lower()
        backend = (
            cv2.CAP_MSMF if name == "msmf" else
            cv2.CAP_DSHOW if name == "dshow" else
            cv2.CAP_AVFOUNDATION if name == "avfoundation" else
            cv2.CAP_V4L2
        )

    # Desired resolution
    want_w, want_h = (1280, 720) if res_preset == "720p" else (1920, 1080)

    cap = None
    used_idx = None
    used_backend = backend
    try:
        if use_stream:
            cap = cv2.VideoCapture(stream.strip())
            used_idx = stream.strip()
        else:
            # first attempt
            cap, used_idx = _open_with_backend(src, backend)

            # fallbacks
            if (not cap) or (not cap.isOpened()):
                if platform.system() == "Windows":
                    alt_backend = cv2.CAP_DSHOW if backend == cv2.CAP_MSMF else cv2.CAP_MSMF
                    if cap: cap.release()
                    cap, used_idx = _open_with_backend(src, alt_backend)
                    used_backend = alt_backend
                elif platform.system() == "Darwin":
                    if cap: cap.release()
                    fourcc_sel = (mon.fourcc or "auto").upper()
                    want_w, want_h = (1280, 720) if (mon.res_preset or "1080p").lower()=="720p" else (1920, 1080)
                    cap, used_idx = _mac_try_indices_for_nonblack([1,0,2,3,4,5], cv2.CAP_AVFOUNDATION, want_w, want_h, fourcc_sel)
                    used_backend = cv2.CAP_AVFOUNDATION
                    if cap is None:
                        # fallback to user‑selected index even if black
                        try:
                            cap = cv2.VideoCapture(int(src), cv2.CAP_AVFOUNDATION)
                            used_idx = int(src)
                        except Exception:
                            cap = None
                            used_idx = None

        if not cap or not cap.isOpened():
            with mon.lock:
                mon.status = "error: cannot open source"
                mon.running = False
            return jsonify(error="cannot open source")

        # Normalize + apply preferences
        try: cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        except Exception: pass

        if fourcc_sel == "MJPG":
            try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception: pass
        elif fourcc_sel == "YUY2":
            try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
            except Exception: pass

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  want_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
        except Exception:
            pass
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception: pass

        # Warm up a few frames
        for _ in range(5):
            cap.read()

        # Try reading a frame
        ok, frame = cap.read()
        if not ok or frame is None:
            return jsonify(error="cannot read frame")
        # Flag obviously black frames to aid debugging in the UI
        try:
            gray_chk = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_chk = float(np.mean(gray_chk))
            black_hint = (mean_chk < 5.0)
        except Exception:
            black_hint = False

        # --- Compute per-reference distances for debugging ---
        roi = roi_connected_gray(frame)
        roi_eq = _equalize_hist(roi)
        phv_int = ih.phash(Image.fromarray(roi_eq))
        distances = []
        for h in int_h_list:
            try:
                distances.append(int(phv_int - h))
            except Exception:
                distances.append(None)
        di = min([d for d in distances if d is not None] or [999])

        # Decide state (ROI-only logic)
        det, db, _, cg, mean_l, std_l, roi_mean, roi_bright, bright_ok, _wfg, _wfr = decide(
            frame, bars_h_roi, int_h_list, cw, thr, dmean, dstd, mon.margin, ref_mean, ref_bright,
            white_frac_override=getattr(mon, "white_frac_gate", None),
            mean_gate_override=getattr(mon, "mean_gate", None)
        )

        # Size/FPS/FourCC readback (best-effort)
        try:
            h, w = frame.shape[:2]
            size_str = f"{w}x{h}"
        except Exception:
            size_str = "-"
        try:
            v = cap.get(cv2.CAP_PROP_FOURCC)
            fourcc_read = _fourcc_to_str(int(v)) if v and v != 0 else ""
        except Exception:
            fourcc_read = ""
        try:
            fps_g = cap.get(cv2.CAP_PROP_FPS)
            fps_val = round(float(fps_g), 1) if fps_g and fps_g > 0 else None
        except Exception:
            fps_val = None

        return jsonify(
            det=det,
            db=int(db), di=int(di),
            mean=round(mean_l, 2), std=round(std_l, 2),
            distances=distances,
            roi_mean=round(float(roi_mean), 1),
            roi_bright=round(float(roi_bright), 3),
            bright_ok=bool(bright_ok),
            backend_used=int(used_backend) if isinstance(used_backend, int) else used_backend,
            index_used=used_idx,
            size=size_str,
            fourcc=fourcc_read or None,
            fps=fps_val,
            black_hint=black_hint
        )

    finally:
        try:
            if cap: cap.release()
        except Exception:
            pass
# New endpoint: enumerate available camera indices/backends
@app.get("/enumerate")
def enumerate_caps():
    results = []
    def try_open(idx, backend, bname):
        cap = cv2.VideoCapture(idx, backend)
        ok = cap.isOpened()
        size = None
        if ok:
            ok2, f = cap.read()
            if ok2 and f is not None:
                h, w = f.shape[:2]
                size = f"{w}x{h}"
            cap.release()
        return {"index": idx, "backend": bname, "opened": bool(ok), "size": size}
    backends = []
    if platform.system()=="Windows":
        backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
    elif platform.system()=="Darwin":
        backends = [("AVFOUNDATION", cv2.CAP_AVFOUNDATION)]
    else:
        backends = [("V4L2", cv2.CAP_V4L2)]
    for name, b in backends:
        for i in range(0, 6):
            results.append(try_open(i, b, name))
    return jsonify(results=results)

# New endpoint: probe source and show info for selected backend/fourcc/res
@app.post("/probe")
def probe():
    cfg = request.get_json(force=True) if request.is_json else {}
    src = cfg.get("src", mon.src)
    backend_name = cfg.get("backend", mon.backend_name)
    fourcc_sel = (cfg.get("fourcc", mon.fourcc) or "auto").upper()
    res_preset = (cfg.get("res", mon.res_preset) or "1080p").lower()
    stream = cfg.get("stream", mon.stream_path) or ""

    # Determine desired resolution
    if res_preset == "720p":
        want_w, want_h = 1280, 720
    else:
        want_w, want_h = 1920, 1080

    # Helper to open and read one frame with options
    def try_one(_src, _backend, _bname):
        cap = None
        opened = False
        size = None
        fourcc_read = ''
        fps_val = None
        try:
            if isinstance(_src, str) and not _src.isdigit():
                cap = cv2.VideoCapture(_src)
            else:
                try:
                    idx = int(_src)
                except Exception:
                    idx = _src
                cap = cv2.VideoCapture(idx, _backend)
            if not cap or not cap.isOpened():
                return {"index": _src, "backend": _bname, "opened": False}

            # Normalize & apply user preferences
            try: cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            except Exception: pass
            if fourcc_sel == "MJPG":
                try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception: pass
            elif fourcc_sel == "YUY2":
                try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
                except Exception: pass
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  want_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
            except Exception:
                pass
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass
            try: cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception: pass

            opened = cap.isOpened()
            if not opened:
                return {"index": _src, "backend": _bname, "opened": False}

            # Probe a few frames
            for _ in range(5):
                cap.read()
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                size = f"{w}x{h}"
            # Read back FOURCC/FPS if possible
            try:
                v = cap.get(cv2.CAP_PROP_FOURCC)
                if v and v != 0:
                    fourcc_read = _fourcc_to_str(int(v))
            except Exception:
                pass
            try:
                fps_g = cap.get(cv2.CAP_PROP_FPS)
                if fps_g and fps_g > 0:
                    fps_val = round(float(fps_g), 1)
            except Exception:
                pass
            return {"index": _src, "backend": _bname, "opened": True, "size": size, "fourcc": fourcc_read, "fps": fps_val}
        finally:
            try:
                if cap: cap.release()
            except Exception:
                pass

    # If stream/URL provided, probe just that without backend variations
    if isinstance(stream, str) and stream.strip():
        res = [try_one(stream.strip(), 0, "AUTO(URL)")]
        return jsonify(results=res, note="Probed stream URL")

    # Otherwise probe common backends per-OS
    backends = []
    if platform.system() == "Windows":
        # If user chose a specific backend, test it first then the alternate
        name = (backend_name or "auto").lower()
        if name == "msmf":
            backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
        elif name == "dshow":
            backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
        else:
            backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
    elif platform.system() == "Darwin":
        backends = [("AVFOUNDATION", cv2.CAP_AVFOUNDATION)]
    else:
        backends = [("V4L2", cv2.CAP_V4L2)]

    results = []
    for bname, bconst in backends:
        results.append(try_one(src, bconst, bname))
    return jsonify(results=results)


# Helper function to begin monitoring with a config dict
def _begin_with_cfg(cfg: dict):
    with mon.lock:
        if mon.running:
            # already running: report current CSV so UI can show it
            return jsonify(ok=True, already=True, csv=mon.csv_path)

        mon.last_cfg = dict(cfg) if isinstance(cfg, dict) else {}
        mon.src          = cfg.get("src", mon.src)
        mon.backend_name = cfg.get("backend", mon.backend_name)
        mon.fourcc       = cfg.get("fourcc", mon.fourcc)
        mon.res_preset   = cfg.get("res", mon.res_preset)
        mon.stream_path  = cfg.get("stream", mon.stream_path)
        mon.center_w     = int(cfg.get("cw", mon.center_w))
        mon.thresh       = int(cfg.get("thr", mon.thresh))
        mon.stable_frames= int(cfg.get("st", mon.stable_frames))
        mon.dark_mean    = float(cfg.get("dm", mon.dark_mean))
        mon.dark_std     = float(cfg.get("ds", mon.dark_std))
        # Optional overrides from UI
        mon.white_frac_gate = cfg.get("white_pct", mon.white_frac_gate)
        mon.mean_gate       = cfg.get("mean_gate", mon.mean_gate)
        try:
            rin = cfg.get("roi_inset", None)
            if rin is not None:
                mon.roi_inset_px = int(rin or 0)
        except Exception:
            pass
        # Prefer provided paths only if they exist; otherwise fall back to bundled art/
        req_bars = cfg.get("bars", "")
        req_intr = cfg.get("intr", "")
        req_bars = req_bars.strip() if isinstance(req_bars, str) else ""
        req_intr = req_intr.strip() if isinstance(req_intr, str) else ""
        mon.bars_ref = req_bars if (req_bars and os.path.exists(req_bars)) else BARS_REF
        mon.int_ref  = req_intr if (req_intr and os.path.exists(req_intr)) else INT_REF
        # Build list of usable INT reference images (requested first)
        refs = []
        if mon.int_ref and os.path.exists(mon.int_ref):
            refs.append(mon.int_ref)
        for p in discover_int_refs():
            if p not in refs:
                refs.append(p)
        mon.int_ref_paths = refs

        # Always roll a fresh CSV on every start (ignore any incoming csv path)
        mon.reset_counts_and_roll_csv()
        new_csv = mon.csv_path

        mon.running = True
        mon.status = "starting"
        mon.last_frame = None
        mon.last_crop  = None
        mon.last_metrics = {"db": None, "di": None, "mean": None, "std": None}
        mon.await_connect = True

    t = threading.Thread(target=run_loop, daemon=True)
    mon.worker = t
    t.start()
    # Tell the UI which CSV is active
    return jsonify(ok=True, csv=new_csv, rolled=True)

# POST /start: Accepts JSON, starts monitoring with config
@app.post("/start")
def start():
    # Accept empty or malformed JSON gracefully
    cfg = request.get_json(silent=True) or {}
    print("POST /start called with cfg:", cfg)
    return _begin_with_cfg(cfg)

# GET /start: Alias for starting with current settings (no body)
@app.get("/start")
def start_get():
    print("GET /start called (no body); starting with current settings")
    return _begin_with_cfg({})

@app.post("/save_console_metadata")
def save_console_metadata():
    """Save console serial numbers and scope IDs before starting test."""
    try:
        data = request.get_json(silent=True) or {}
        consoles = data.get("consoles", {})
        
        with mon.lock:
            # Validate and save console data
            for console_num in [1, 2, 3]:
                console_key = str(console_num)
                if console_key in consoles:
                    mon.consoles[console_num] = {
                        "serial": consoles[console_key].get("serial", ""),
                        "scope_id": consoles[console_key].get("scope_id", "")
                    }
        
        return jsonify(ok=True, consoles=mon.consoles)
    except Exception as e:
        return jsonify(error=str(e)), 400

@app.get("/get_console_metadata")
def get_console_metadata():
    """Get current console metadata."""
    with mon.lock:
        return jsonify(consoles=mon.consoles)

@app.post("/stop")
def stop():
    with mon.lock:
        mon.running = False
        mon.status = "stopped"
    worker = getattr(mon, "worker", None)
    if worker is not None and worker.is_alive():
        # Give the loop time to exit and release the device
        worker.join(timeout=2.0)
    return jsonify(ok=True)

@app.post("/stop_and_report")
def stop_and_report():
    """Stop the test and generate a comprehensive report."""
    # First stop the test
    with mon.lock:
        mon.running = False
        mon.status = "stopped"
    
    worker = getattr(mon, "worker", None)
    if worker is not None and worker.is_alive():
        worker.join(timeout=2.0)
    
    # Generate report
    try:
        report = generate_test_report()
        report_text = format_report_text(report)
        
        # Print to console
        print("\n" + report_text + "\n")
        
        return jsonify(ok=True, report=report, report_text=report_text)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

@app.get("/generate_report")
def get_report():
    """Generate report without stopping (for live viewing)."""
    try:
        report = generate_test_report()
        report_text = format_report_text(report)
        return jsonify(ok=True, report=report, report_text=report_text)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

@app.post("/clear")
def clear():
    with mon.lock:
        if mon.running:
            return jsonify(error="stop first"), 400
        mon.reset_counts_and_roll_csv()
    return jsonify(ok=True, csv=mon.csv_path)

# New endpoint: reset tallies (counters and last-seen) without rolling CSV, allowed even while running
@app.post("/reset_tallies")
def reset_tallies():
    with mon.lock:
        mon.reset_tallies()
    return jsonify(ok=True)

@app.get("/download")
def download_csv():
    path = os.path.abspath(mon.csv_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.isfile(path):
        return jsonify(error="CSV not found", path=path), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


# Optional: favicon route to suppress 404 noise
@app.get("/favicon.ico")
def favicon():
    return Response(status=204)

@app.post("/shutdown")
def shutdown():
    # Graceful quit from the web UI
    def _exit():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()
    return jsonify(ok=True)



def open_browser():
    global PORT
    url = f"http://localhost:{PORT}/"
    try:
        if platform.system()=="Windows":
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            webbrowser.open(url)
    except Exception:
        pass
    print(f"Open {url} in your browser if it didn't open automatically.")

# ---- main launcher ----
if __name__ == "__main__":
    # Pick a port (try 5055 first, otherwise the next available up to 5070)
    PORT = _choose_port(5055, 5070)
    print(f"Starting server on http://localhost:{PORT} …")
    # Launch the browser shortly after the server starts
    threading.Timer(0.6, open_browser).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)