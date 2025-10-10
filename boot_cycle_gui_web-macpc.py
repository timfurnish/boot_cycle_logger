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
    boot_cycle_gui_web-macpc.py

Build (macOS/Linux):
  pyinstaller --noconfirm --onefile --noconsole \
    --name BootCycleLogger \
    --add-data "Scope-Disconnected.png:." \
    --add-data "Scope-Connected-SidewinderCCU.png:." \
    --add-data "Scope-Connected-OtherCCU.png:." \
    boot_cycle_gui_web-macpc.py
"""

import os, sys, threading, time, csv, platform, webbrowser, subprocess
from datetime import datetime
from flask import Flask, jsonify, request, render_template_string, Response, send_file
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

# Reference images (always relative to the app directory so the project is portable)
# Place these PNGs next to the script or in the PyInstaller bundle.
BARS_REF = os.path.join(APP_DIR, "Scope-Disconnected.png")
INT_REF  = os.path.join(APP_DIR, "Scope-Connected-SidewinderCCU.png")
INT_REF2 = os.path.join(APP_DIR, "Scope-Connected-OtherCCU.png")

# Logs: relative to the app directory so the project is portable across machines
LOG_ROOT = os.path.join(APP_DIR, "logs")
os.makedirs(LOG_ROOT, exist_ok=True)
CSV_PATH = os.path.join(LOG_ROOT, f"boot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

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
    # If cw is <= 0 or >= frame width, use the full frame (no side-gutter crop).
    try:
        if cw is None:
            cw = CENTER_W
        cw = int(cw)
    except Exception:
        cw = CENTER_W
    if cw <= 0 or cw >= w:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (THUMB_W, THUMB_H))

    s = (w - cw) // 2
    if s <= 0:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (THUMB_W, THUMB_H))

    g = cv2.cvtColor(np.concatenate([bgr[:, :s], bgr[:, w - s:]], 1), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (THUMB_W, THUMB_H))

def ph(gray_crop):
    return ih.phash(Image.fromarray(gray_crop))

def ph_ref(path, cw):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read reference image: {path} (cwd={os.getcwd()})")
    return ph(crop(img, cw))

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
        self.int_ref2  = INT_REF2
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

def decide(frame, bars_h, int_h, center_w, thresh, dmean, dstd, int2_h=None):
    cg = crop(frame, center_w)
    phv = ph(cg)
    db, di1 = phv - bars_h, phv - int_h
    di2 = (phv - int2_h) if int2_h is not None else None
    di = min(di1, di2) if di2 is not None else di1
    mean_lum = float(np.mean(cg)); std_lum = float(np.std(cg))
    is_dark_flat = (mean_lum < dmean) and (std_lum < dstd)
    if is_dark_flat:
        det = "NO_SIGNAL"
    elif (db < thresh):
        det = "BARS"
    elif di < thresh:
        det = "INTERFACE"
    else:
        det = "OTHER"
    return det, int(db), int(di), cg, mean_lum, std_lum

def run_loop():
    mon.worker = threading.current_thread()
    try:
        bars_h = ph_ref(mon.bars_ref, mon.center_w)
        int_h  = ph_ref(mon.int_ref,  mon.center_w)
        int2_h = None
        try:
            int2_h = ph_ref(mon.int_ref2, mon.center_w)
        except Exception:
            int2_h = None
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

            det, db, di, crop_img, mean_l, std_l = decide(frame, bars_h, int_h, cw, thr, dmean, dstd, int2_h=int2_h)

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
                    det_ok = (db < (thr - mon.margin))
                elif det == "INTERFACE":
                    det_ok = (di < (thr - mon.margin))
                else:
                    det_ok = True  # OTHER has no phash gate

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
                    else:  # OTHER
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
app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Boot Cycle Logger</title>
  <style>
    :root {
      --bg:#0f172a; --text:#e5e7eb; --card:#111827; --border:#1f2937;
      --btn:#38bdf8; --btnText:#0b1220; --btn2:#1f2937; --muted:#94a3b8;
      --btn2Hover:#374151;
      --pillIdle:#334155;           /* slate */
      --pillConnected:#16a34a;      /* green */
      --pillNotConnected:#dc2626;   /* red */
      --pillOther:#eab308;          /* amber */
      --btnDisabled:#334155; --btnDisabledText:#a3a3a3;
    }
    * { box-sizing:border-box }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:24px; background:var(--bg); color:var(--text); }
    h1 { color:var(--btn); margin-top:0 }
    .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; margin:12px 0; }
    .row { display:flex; gap:16px; flex-wrap:wrap; }
    .stat { flex:1 1 220px; background:#0b1220; border:1px solid var(--border); border-radius:10px; padding:12px; }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    .value { font-size:28px; font-weight:700; }
    .sub { color:var(--muted); font-size:12px; margin-top:6px; }
    button {
      background:var(--btn); color:var(--btnText); font-weight:700; border:none; padding:10px 14px;
      border-radius:8px; cursor:pointer; transition:all .15s ease; outline: none;
    }
    button:hover { filter:brightness(1.06); transform:translateY(-1px); }
    button:active { transform:translateY(0px) scale(0.99); }
    button.secondary { background:var(--btn2); color:var(--text); }
    button.secondary:hover { background:var(--btn2Hover); }
    button:disabled { background:var(--btnDisabled); color:var(--btnDisabledText); cursor:not-allowed; transform:none; }
    input, select { background:#0b1220; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:8px; width:100%; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }
    .pill {
      display:inline-flex; align-items:center; justify-content:center;
      min-width: 320px; padding:18px 26px; border-radius:999px; font-weight:800; font-size:22px;
      margin: 6px 0 14px 0; border:1px solid #00000022;
      box-shadow: 0 6px 20px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.08);
    }
    .thumbBox { display:flex; gap:16px; align-items:center; }
    .metrics { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:14px; color:#cbd5e1; }
  </style>
</head>
<body>
  <h1>Boot Cycle Logger</h1>

  <!-- BIG STATUS PILL -->
  <div class="pill" id="statusPill" style="background: var(--pillIdle);">idle</div>

  <div class="card">
    <div class="grid">
      <div><label>Source (index/URL)</label><input id="src" value="{{src}}"></div>
      <div><label>Backend</label>
        <select id="backend">
        <option value="auto" {{ 'selected' if backend|lower=='auto' else '' }}>auto</option>
        <option value="MSMF" {{ 'selected' if backend|lower=='msmf' else '' }}>MSMF (Windows)</option>
        <option value="DSHOW" {{ 'selected' if backend|lower=='dshow' else '' }}>DSHOW (Windows)</option>
        <option value="AVFOUNDATION" {{ 'selected' if backend|lower=='avfoundation' else '' }}>AVFOUNDATION (macOS)</option>
        <option value="V4L2" {{ 'selected' if backend|lower=='v4l2' else '' }}>V4L2 (Linux)</option>
        </select>
      </div>
      <div><label>FOURCC</label>
        <select id="fourcc">
          <option value="auto" {{ 'selected' if fourcc|lower=='auto' else '' }}>auto</option>
          <option value="MJPG" {{ 'selected' if fourcc|upper=='MJPG' else '' }}>MJPG</option>
          <option value="YUY2" {{ 'selected' if fourcc|upper=='YUY2' else '' }}>YUY2</option>
        </select>
      </div>
      <div><label>Resolution</label>
        <select id="res">
            <option value="1080p" {{ 'selected' if res|lower=='1080p' else '' }}>1080p</option>
            <option value="720p"  {{ 'selected' if res|lower=='720p'  else '' }}>720p</option>
        </select>
      </div>
      <div><label>Stream URL (rtsp/http)</label><input id="stream" placeholder="rtsp://user:pass@host:554/stream" value="{{stream}}"></div>
      <div><label>Center Width <span style="color:var(--muted);font-size:12px">(set ≥ video width to disable crop)</span></label><input id="cw" type="number" value="{{cw}}"></div>
      <div><label>Threshold</label><input id="thr" type="number" value="{{thr}}"></div>
      <div><label>Stable Frames</label><input id="st" type="number" value="{{st}}"></div>
      <div><label>Dark Mean</label><input id="dm" type="number" value="{{dm}}"></div>
      <div><label>Dark Std</label><input id="ds" type="number" value="{{ds}}"></div>
      <div><label>Device Disconnected Ref</label><input id="bars" value="{{bars}}"></div>
      <div><label>Scope Connected Ref</label><input id="intr" value="{{intr}}"></div>
      <div><label>Scope Connected (Alt) Ref</label><input id="intr2" value="{{intr2}}"></div>
      <div><label>CSV Path</label><input id="csv" value="{{csv}}"></div>
    </div>
    <div style="margin-top:12px; display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <button id="btnStart" onclick="start()">Start</button>
      <button id="btnStop"  class="secondary" onclick="stop()">Stop</button>
      <button id="btnClear" class="secondary" onclick="clearCounts()">Clear &amp; New CSV</button>
      <button id="btnResetTallies" class="secondary" onclick="resetTallies()">Reset tallies</button>
      <button id="btnProbe" class="secondary" onclick="probe()">Probe source</button>
      <label style="display:flex; align-items:center; gap:6px; font-size:13px; color:var(--muted);">
        <input id="liveCropToggle" type="checkbox" checked onchange="toggleThumb()"/>
        Live crop
      </label>
      <div style="flex:1"></div>
      <button class="secondary" onclick="window.location.href='/download'">Download CSV</button>
      <button class="secondary" onclick="shutdown()">Quit</button>
    </div>

    <!-- Probe output panel -->
    <div class="card" style="margin-top:10px; background:#0b1220; color:#9ca3af; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:13px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-weight:700; color:#38bdf8;">Probe Results</span>
        <button class="secondary" onclick="document.getElementById('probeOut').textContent=''">Clear</button>
      </div>
      <pre id="probeOut" style="white-space:pre-wrap; margin-top:8px;"></pre>
    </div>
  </div>

  <div class="card thumbBox">
    <div>
      <img id="thumb" src="" alt="live crop" width="256" height="144" style="border-radius:8px; border:1px solid var(--border)"/>
    </div>
    <div class="metrics">
      <div><b>Test Frame</b></div>
      <div id="m_db">bars distance: -</div>
      <div id="m_di">interface distance: -</div>
      <div id="m_mean">mean luminance: -</div>
      <div id="m_std">std luminance: -</div>
      <div style="margin-top:8px;">
        <button class="secondary" onclick="peek()">Test frame now</button>
      </div>
      <div style="margin-top:8px; color:#9ca3af; font-size:12px;">Tip: if bars/interface distances are close to the threshold, adjust Threshold or Center Width. Dark/flat frames should fail bars by the dark gates.</div>
    </div>
  </div>

  <div class="row">
    <div class="stat"><div class="label">Status</div><div id="status" class="value">idle</div></div>
    <div class="stat"><div class="label">Not Connected</div><div id="barsCount" class="value">0</div><div class="sub" id="barsSeen">last: -</div></div>
    <div class="stat"><div class="label">Connected</div><div id="intCount" class="value">0</div><div class="sub" id="intSeen">last: -</div></div>
    <div class="stat"><div class="label">No Signal</div><div id="otherCount" class="value">0</div><div class="sub" id="otherSeen">last: -</div></div>
    <div class="stat"><div class="label">Cycles</div><div id="cyclesCount" class="value">0</div></div>
  </div>

  <script>
    console.log('✅ Boot Cycle Logger front-end script loaded');
    let backendInit = false;
    let fourccInit = false;
    let resInit = false;
    function paintPill(text){
      const pill = document.getElementById('statusPill');
      pill.textContent = text;
      let bg = 'var(--pillIdle)';
      const t = (text || '').toLowerCase();
      if (t.includes('connected') && !t.includes('not')) bg = 'var(--pillConnected)';
      else if (t.includes('not connected')) bg = 'var(--pillNotConnected)';
      else if (t.includes('no signal')) bg = 'var(--pillOther)';
      pill.style.background = bg;
    }
    function bust(url){ return url + (url.includes('?')?'&':'?') + 't=' + Date.now(); }

    // Thumbnail refresh logic
    let thumbEnabled = true;
    function refreshThumb(){
      if(!thumbEnabled) return;
      const el = document.getElementById('thumb');
      el.src = bust('/thumb');
    }
    function toggleThumb(){
      const cb = document.getElementById('liveCropToggle');
      thumbEnabled = !!cb.checked;
      if(thumbEnabled) refreshThumb();
    }

    async function probe(){
      const body = {
        src:  document.getElementById('src').value,
        backend: document.getElementById('backend').value,
        fourcc: document.getElementById('fourcc').value,
        res:    document.getElementById('res').value,
        stream: document.getElementById('stream').value
      };
      try{
        const r = await fetch('/probe', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        const j = await r.json();
        const el = document.getElementById('probeOut');
        if(j.error){
          el.textContent = 'Probe error: ' + j.error;
        }else if(j.results){
          const lines = [];
          if (j.note) lines.push(j.note);
          for(const item of j.results){
            lines.push(
              (item.backend || '') + ' idx=' + (item.index !== undefined && item.index !== null ? item.index : '') +
              ' opened=' + item.opened +
              ' size=' + (item.size || '-') +
              ' fourcc=' + (item.fourcc || '-') +
              ' fps=' + (item.fps || '-')
            );
          }
          el.textContent = lines.join('\\n');
        }else{
          el.textContent = 'No results';
        }
      }catch(e){
        document.getElementById('probeOut').textContent = 'Probe failed: ' + e;
      }
    }

    async function getStatus(){
      const r = await fetch('/status'); const j = await r.json();
      const display = j.status_friendly || j.status;
      document.getElementById('status').textContent = display;
      document.getElementById('barsCount').textContent   = j.count_bars;
      document.getElementById('intCount').textContent    = j.count_int;
      document.getElementById('otherCount').textContent  = j.count_other;
      document.getElementById('cyclesCount').textContent = j.cycles;
      document.getElementById('barsSeen').textContent  = 'last: ' + (j.last_seen.BARS || '-');
      document.getElementById('intSeen').textContent   = 'last: ' + (j.last_seen.INTERFACE || '-');
      document.getElementById('otherSeen').textContent = 'last: ' + (j.last_seen.OTHER || '-');
      paintPill(display);
      if (!backendInit) {
        document.getElementById('backend').value = (j.backend || 'auto');
        backendInit = true;
      }
      if (!fourccInit) {
        document.getElementById('fourcc').value = (j.fourcc || 'auto');
        fourccInit = true;
      }
      if (!resInit) {
        document.getElementById('res').value = (j.res_preset || '1080p');
        resInit = true;
      }
      const running = !!j.running;
      document.getElementById('btnStart').disabled = running;
      document.getElementById('btnStop').disabled  = !running;
      document.getElementById('btnClear').disabled = running;
    }

    async function peek(){
      const r = await fetch('/peek'); const j = await r.json();
      document.getElementById('m_db').textContent   = 'bars distance: ' + j.db;
      document.getElementById('m_di').textContent   = 'interface distance: ' + j.di;
      document.getElementById('m_mean').textContent = 'mean luminance: ' + j.mean;
      document.getElementById('m_std').textContent  = 'std luminance: ' + j.std;
    }

    async function start(){
      const body = {
        src:  document.getElementById('src').value,
        backend: document.getElementById('backend').value,
        fourcc: document.getElementById('fourcc').value,
        res:    document.getElementById('res').value,
        stream: document.getElementById('stream').value,
        cw:   parseInt(document.getElementById('cw').value),
        thr:  parseInt(document.getElementById('thr').value),
        st:   parseInt(document.getElementById('st').value),
        dm:   parseFloat(document.getElementById('dm').value),
        ds:   parseFloat(document.getElementById('ds').value),
        bars: document.getElementById('bars').value,
        intr: document.getElementById('intr').value,
        intr2: document.getElementById('intr2').value,
        csv:  document.getElementById('csv').value,
      };
      await fetch('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      setTimeout(getStatus, 300);
    }
    async function stop(){ await fetch('/stop', {method:'POST'}); setTimeout(getStatus, 300); }
    async function clearCounts(){ await fetch('/clear', {method:'POST'}); setTimeout(getStatus, 300); }
    async function resetTallies(){
      await fetch('/reset_tallies', {method:'POST'});
      setTimeout(getStatus, 200);
    }
    async function shutdown(){ await fetch('/shutdown', {method:'POST'}); }
    // Expose functions to global scope
    window.probe = probe;
    window.getStatus = getStatus;
    window.peek = peek;
    window.start = start;
    window.stop = stop;
    window.clearCounts = clearCounts;
    window.resetTallies = resetTallies;
    window.shutdown = shutdown;
    window.toggleThumb = toggleThumb;
    window.refreshThumb = refreshThumb;
    setInterval(getStatus, 600);
    setInterval(refreshThumb, 1500); // decoupled from status polling
    getStatus();
    // Guard re-exports in case of partial parsing
    if (typeof window.probe !== 'function') window.probe = probe;
    if (typeof window.getStatus !== 'function') window.getStatus = getStatus;
    if (typeof window.peek !== 'function') window.peek = peek;
    if (typeof window.start !== 'function') window.start = start;
    if (typeof window.stop !== 'function') window.stop = stop;
    if (typeof window.clearCounts !== 'function') window.clearCounts = clearCounts;
    if (typeof window.resetTallies !== 'function') window.resetTallies = resetTallies;
    if (typeof window.shutdown !== 'function') window.shutdown = shutdown;
    if (typeof window.toggleThumb !== 'function') window.toggleThumb = toggleThumb;
    if (typeof window.refreshThumb !== 'function') window.refreshThumb = refreshThumb;
  </script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML,
        src=mon.src, cw=mon.center_w, thr=mon.thresh, st=mon.stable_frames,
        dm=mon.dark_mean, ds=mon.dark_std, bars=mon.bars_ref, intr=mon.int_ref, intr2=mon.int_ref2,
        csv=mon.csv_path, stream=mon.stream_path, backend=mon.backend_name,
        fourcc=mon.fourcc, res=mon.res_preset
    )

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
            fourcc=mon.fourcc, res_preset=mon.res_preset
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
        int_h  = ph_ref(int_path,  cw)
        int2_h = None
        try:
            int2_h = ph_ref(mon.int_ref2, cw)
        except Exception:
            int2_h = None
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
        det, db, di, crop_img, mean_l, std_l = decide(frame, bars_h, int_h, cw, thr, dmean, dstd, int2_h=int2_h)
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

@app.post("/start")
def start():
    cfg = request.get_json(force=True)
    cfg_copy = dict(cfg) if isinstance(cfg, dict) else {}
    with mon.lock:
        if mon.running:
            return jsonify(ok=True, already=True)
        mon.last_cfg = cfg_copy
        mon.src       = cfg.get("src", mon.src)
        mon.backend_name = cfg.get("backend", mon.backend_name)
        mon.fourcc = cfg.get("fourcc", mon.fourcc)
        mon.res_preset = cfg.get("res", mon.res_preset)
        mon.stream_path = cfg.get("stream", mon.stream_path)
        mon.center_w  = int(cfg.get("cw", mon.center_w))
        mon.thresh    = int(cfg.get("thr", mon.thresh))
        mon.stable_frames = int(cfg.get("st", mon.stable_frames))
        mon.dark_mean = float(cfg.get("dm", mon.dark_mean))
        mon.dark_std  = float(cfg.get("ds", mon.dark_std))
        mon.bars_ref  = cfg.get("bars", mon.bars_ref)
        mon.int_ref   = cfg.get("intr", mon.int_ref)
        mon.int_ref2  = cfg.get("intr2", mon.int_ref2)
        mon.csv_path  = cfg.get("csv", mon.csv_path)
        mon.reset_counts_and_roll_csv()  # also rolls CSV
        mon.running = True
        mon.status = "starting"
        mon.last_frame = None
        mon.last_crop  = None
        mon.last_metrics = {"db": None, "di": None, "mean": None, "std": None}
        mon.await_connect = True  # count a cycle if we start while already connected
    t = threading.Thread(target=run_loop, daemon=True)
    mon.worker = t
    t.start()
    return jsonify(ok=True)

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
    url = "http://localhost:5055/"
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
    threading.Timer(0.6, open_browser).start()
    print("Starting server on http://localhost:5055 …")
    app.run(host="127.0.0.1", port=5055, debug=False)