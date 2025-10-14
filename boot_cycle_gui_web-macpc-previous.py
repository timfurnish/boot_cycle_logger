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

# ---- Windows video-capture knobs (helps MSMF behave with MJPG/YUY2 and color) ----
if platform.system() == "Windows":
    # Disables flaky MSMF hardware transforms that can break MJPG/YUY2 colors/range
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
    # Allow H.264 in case you point at an RTSP/HTTP stream (harmless otherwise)
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

# Reference images + CSV
BARS_REF = os.path.join(ART_DIR, "Scope-Disconnected.png")  # keep your disconnected image (now relative to art/)
# ROI connected reference inside /art (used for INTERFACE detection)
INT_REF  = os.path.join(ART_DIR, "Boot-Reliabilty-Testing.png")
CSV_PATH = os.path.join(LOG_ROOT, f"boot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# Additional INT reference discovery (multiple "Boot-Reliabilty-Testing*" variants)
INT_REF_PREFIX = "Boot-Reliabilty-Testing"   # (spelling matches your files)
INT_REF_EXTS   = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# Video source & capture options
SRC = "0" if platform.system()=="Windows" else "1"
BACKEND     = "auto"       # auto|MSMF|DSHOW|AVFOUNDATION|V4L2
FOURCC      = "auto"       # auto|MJPG|YUY2
RES_PRESET  = "1080p"      # 1080p|720p

# Anti-flicker defaults (Windows virtual cam can be jittery)
HOLD_MS  = 800   # minimum time a new state must persist before we accept it
MARGIN   = 2     # hysteresis margin for phash distance thresholds

# Throttle how often we update the live crop to avoid pushing detector around
THUMB_EVERY_MS = 1500

# Live thumbnail target size (keep 16:9 to match source aspect)
THUMB_W = 256
THUMB_H = 144

def _placeholder_thumb():
    """Generate a neutral 16:9 placeholder thumbnail when no frame is available."""
    img = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    # subtle border
    cv2.rectangle(img, (0, 0), (THUMB_W - 1, THUMB_H - 1), (60, 60, 60), 1)
    try:
        cv2.putText(img, "no frame", (10, THUMB_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
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

# ---------- ROI detector for "Device Connected" using Boot-Reliabilty-Testing.png ----------
# Only the Boot-Reliabilty-Testing.png ROI region is used for INTERFACE detection.
# The reference has a 400x400 white square bottom-aligned, 420px from the left on a 1920x1080 canvas.
# We compute ROI in proportional coordinates so any input resolution works.
BASE_W, BASE_H = 1920, 1080
ROI_BOX = dict(x=420, y=BASE_H - 400, w=400, h=400)  # (x,y) is top-left in baseline space

def _roi_box_for_frame(w:int, h:int):
    """Get the proportional ROI for the INTERFACE region for any frame size."""
    sx = w / float(BASE_W)
    sy = h / float(BASE_H)
    x  = int(round(ROI_BOX["x"] * sx))
    y  = int(round(ROI_BOX["y"] * sy))
    rw = int(round(ROI_BOX["w"] * sx))
    rh = int(round(ROI_BOX["h"] * sy))
    # Clamp ROI to image bounds
    x  = max(0, min(x, max(0, w-1)))
    y  = max(0, min(y, max(0, h-1)))
    if x+rw > w: rw = w - x
    if y+rh > h: rh = h - y
    return x, y, max(1, rw), max(1, rh)

def roi_connected_gray(bgr):
    """Extract the INTERFACE ROI region, grayscale it, return a square for stable pHash."""
    h, w = bgr.shape[:2]
    x, y, rw, rh = _roi_box_for_frame(w, h)
    roi = bgr[y:y+rh, x:x+rw]
    if roi.size == 0:
        roi = bgr
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

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

def ph_int_ref(path):
    """pHash of the ROI (white 400x400 square area) from a single INT reference image."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read INT reference image: {path}")
    return ih.phash(Image.fromarray(roi_connected_gray(img)))

def ph_int_ref_list(paths):
    """pHashes of the ROI from all provided INT reference images (skip unreadable)."""
    hashes = []
    for p in paths or []:
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            hashes.append(ih.phash(Image.fromarray(roi_connected_gray(img))))
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
        self.hold_ms   = HOLD_MS
        self.margin    = MARGIN
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
        self.await_connect = True

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

mon = Monitor()

def open_csv(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if f.tell()==0:
        w.writerow(["ts","status","bars_dist","int_dist","cycles"])
    return f, w

def decide(frame, bars_h, int_h_list, center_w, thresh, dmean, dstd, margin):
    """
    Decide the current state based on:
      - NO_SIGNAL: dark/flat gate based on mean/std (first)
      - BARS: pHash on center/side-crop
      - INTERFACE: ROI-only pHash using Boot-Reliabilty-Testing* variants
      - OTHER: fallback
    Returns: (det, db, di, cg, mean_lum, std_lum)
    """
    # Center/side crop for BARS
    cg = crop(frame, center_w)
    phv_bars = ph(cg)
    db = phv_bars - bars_h

    # ROI for "Device Connected" (INTERFACE)
    roi = roi_connected_gray(frame)
    phv_int = ih.phash(Image.fromarray(roi))
    # Use minimum distance across all known INT references; if none, set large number
    di = min([(phv_int - h) for h in (int_h_list or [])] or [999])

    # Brightness heuristic for the white-square case
    roi_mean, roi_std, roi_bright = roi_stats(roi)

    # Darkness/flatness gates computed on the center-crop
    mean_lum = float(np.mean(cg))
    std_lum  = float(np.std(cg))
    is_dark_flat = (mean_lum < dmean) and (std_lum < dstd)

    # Decision order
    if is_dark_flat:
        det = "NO_SIGNAL"
    elif (di < (thresh + margin)) or (roi_bright >= 0.15 and roi_mean >= 150.0):
        det = "INTERFACE"
    elif db < (thresh + margin):
        det = "BARS"
    else:
        det = "OTHER"

    return det, int(db), int(di), cg, mean_lum, std_lum
def run_loop():
    mon.worker = threading.current_thread()
    try:
        bars_h = ph_ref(mon.bars_ref, mon.center_w)
        int_h_list = ph_int_ref_list(mon.int_ref_paths) or [ph_int_ref(mon.int_ref)]
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

            det, db, di, crop_img, mean_l, std_l = decide(frame, bars_h, int_h_list, cw, thr, dmean, dstd, margin)

            with mon.lock:
                # update live artifacts
                mon.last_frame = frame
                # Throttle live-crop updates so UI refreshes don't perturb timing on slower systems
                now_sec = time.time()
                if mon.thumb_enabled and ((now_sec - mon._last_thumb_ts) * 1000.0 >= mon.thumb_every_ms):
                    # store a copy to decouple from OpenCV buffer reuse
                    mon.last_crop = crop_img.copy()
                    mon._last_thumb_ts = now_sec
                mon.last_metrics = {"db": db, "di": di, "mean": round(mean_l,2), "std": round(std_l,2)}

                # stabilize
                if det == mon._raw_last:
                    mon._raw_stable += 1
                else:
                    mon._raw_last = det
                    mon._raw_stable = 1

                changed = False
                # Hysteresis: require the winning class to be clearly inside threshold by a margin
                det_ok = False
                if det == "BARS":
                    det_ok = (db < (thr + mon.margin))
                elif det == "INTERFACE":
                    # Accept either a close pHash OR a strong-bright ROI flag carried in last_metrics
                    det_ok = (di < (thr + mon.margin))
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
                    w.writerow([now, label_for(det), db, di, mon.cycles])
                    f.flush()
                    print(f"{now}  {label_for(det)}  cycles={mon.cycles}")
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

# ---------- web app ----------
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.get("/")
def index():
    context = dict(
        src=mon.src, cw=mon.center_w, thr=mon.thresh, st=mon.stable_frames,
        dm=mon.dark_mean, ds=mon.dark_std, bars=mon.bars_ref, intr=mon.int_ref,
        csv=mon.csv_path, stream=mon.stream_path, backend=mon.backend_name,
        fourcc=mon.fourcc, res=mon.res_preset
    )
    tpl_path = os.path.join(TEMPLATE_DIR, "index.html")
    if os.path.exists(tpl_path):
        return render_template("index.html", **context)
    else:
        # Fallback: minimal inline HTML if index.html not found
        return "<h2>Boot Cycle Logger</h2><p>index.html not found in templates/.</p>"

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
            csv=mon.csv_path               # ← add this line
        )

@app.get("/thumb")
def thumb():
    with mon.lock:
        img = mon.last_crop
        if img is None and mon.last_frame is not None:
            # Build a one-off crop so the UI still has something to show
            try:
                img = crop(mon.last_frame, mon.center_w)
            except Exception:
                img = None
        if img is None:
            # Serve a neutral placeholder instead of 404 to keep the UI clean
            img = _placeholder_thumb()
        ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return Response(status=500)
    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.get("/peek")
def peek():
    # If loop is running and we have metrics, return them; otherwise do a one-shot grab.
    with mon.lock:
        m = mon.last_metrics.copy()
        have_recent = (mon.last_frame is not None)
        src = mon.src
        stream = mon.stream_path
        cw = mon.center_w
        thr = mon.thresh
        dmean = mon.dark_mean
        dstd = mon.dark_std
        bars_path = mon.bars_ref
        int_path  = mon.int_ref
        backend_name = mon.backend_name
    if have_recent:
        for k in ("db","di","mean","std"):
            if m.get(k) is None:
                m[k] = "-"
        return jsonify(m)
    try:
        bars_h = ph_ref(bars_path, cw)
        int_h_list = ph_int_ref_list(discover_int_refs()) or [ph_int_ref(int_path)]
    except Exception as e:
        return jsonify(error=f"ref load: {e}")
    use_stream = isinstance(stream, str) and stream.strip() != ""
    def _open(_src, _backend):
        if isinstance(_src, str) and not _src.isdigit():
            return cv2.VideoCapture(_src)
        try:
            idx = int(_src)
        except Exception:
            idx = _src
        return cv2.VideoCapture(idx, _backend)
    # resolve backend
    if (backend_name or "auto").lower() == "auto":
        backend = cv2.CAP_MSMF if platform.system()=="Windows" else (cv2.CAP_AVFOUNDATION if platform.system()=="Darwin" else cv2.CAP_V4L2)
    else:
        name = backend_name.lower()
        backend = cv2.CAP_MSMF if name=="msmf" else cv2.CAP_DSHOW if name=="dshow" else cv2.CAP_AVFOUNDATION if name=="avfoundation" else cv2.CAP_V4L2
    cap = None
    try:
        if use_stream:
            cap = cv2.VideoCapture(stream.strip())
        else:
            cap = _open(src, backend)
            if (not cap) or (not cap.isOpened()) and platform.system()=="Windows":
                alt = cv2.CAP_DSHOW if backend==cv2.CAP_MSMF else cv2.CAP_MSMF
                if cap: cap.release()
                cap = _open(src, alt)
        if not cap or not cap.isOpened():
            return jsonify(error="cannot open source")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        ok, frame = cap.read()
        if not ok:
            return jsonify(error="cannot read frame")
        det, db, di, crop_img, mean_l, std_l = decide(frame, bars_h, int_h_list, cw, thr, dmean, dstd, mon.margin)
        return jsonify(db=int(db), di=int(di), mean=round(float(mean_l),2), std=round(float(std_l),2), det=det)
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