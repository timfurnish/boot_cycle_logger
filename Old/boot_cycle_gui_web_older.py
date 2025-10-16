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
    --add-data "Comp-Scope-Disconnected-ColorBars.png;." ^
    --add-data "Comp-Still-&-Video-capture-disabled.png;." ^
    boot_cycle_gui_web.py
"""

import os, sys, threading, time, csv, platform, webbrowser, io
from datetime import datetime
from flask import Flask, jsonify, request, render_template_string, Response, send_file
import cv2, numpy as np
from PIL import Image
import imagehash as ih

# ---------- defaults ----------
CENTER_W = 1080
THRESH   = 10
STABLE   = 3
DARK_MEAN= 22.0
DARK_STD = 12.0
SRC      = "1"  # OBS virtual cam on mac; Windows often "0" or "1"
BARS_REF = "Comp-Scope-Disconnected-ColorBars.png"
INT_REF  = "Comp-Still-&-Video-capture-disabled.png"
CSV_PATH = os.path.join("logs", f"boot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# Anti-flicker defaults (Windows virtual cam can be jittery)
HOLD_MS  = 800   # minimum time a new state must persist before we accept it
MARGIN   = 2     # hysteresis margin for phash distance thresholds

# Throttle how often we update the live crop to avoid pushing detector around
THUMB_EVERY_MS = 1500  # throttle how often we update the live crop to avoid pushing detector around

def crop(bgr, cw=CENTER_W):
    h,w,_ = bgr.shape
    s = max(1, (w - cw)//2)
    g = cv2.cvtColor(np.concatenate([bgr[:,:s], bgr[:,w-s:]], 1), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (256,256))

def ph(gray_crop):
    return ih.phash(Image.fromarray(gray_crop))

def ph_ref(path, cw):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read reference image: {path}")
    return ph(crop(img, cw))

def label_for(state):
    return "Device Not Connected" if state=="BARS" else "Device Connected" if state=="INTERFACE" else "other"

class Monitor:
    def __init__(self):
        # Prefer MSMF on Windows (works well with OBS Virtual Cam), AVFOUNDATION on mac
        self.backend = cv2.CAP_MSMF if platform.system()=="Windows" else cv2.CAP_AVFOUNDATION

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
        self.csv_path  = CSV_PATH

        # runtime state
        self.lock = threading.Lock()
        self.running = False
        self.status = "idle"
        self.count_bars = 0
        self.count_int  = 0
        self.count_other= 0
        self.cycles     = 0

        # stabilization
        self._last = "UNKNOWN"
        self._raw_last = "UNKNOWN"
        self._raw_stable = 0
        self._last_change_ts = 0.0

        # last-seen timestamps
        self.last_seen = {"BARS": None, "INTERFACE": None, "OTHER": None}

        # live frames/metrics for thumbnail + tester
        self.last_frame = None
        self.last_crop  = None
        self.last_metrics = {"db": None, "di": None, "mean": None, "std": None}

    def reset_counts_and_roll_csv(self):
        self.count_bars = self.count_int = self.count_other = self.cycles = 0
        self._last = "UNKNOWN"
        self._raw_last = "UNKNOWN"
        self._raw_stable = 0
        self.last_seen = {"BARS": None, "INTERFACE": None, "OTHER": None}
        # new timestamped CSV on each clear
        self.csv_path = os.path.join("logs", f"boot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

mon = Monitor()

def open_csv(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if f.tell()==0:
        w.writerow(["ts","status","bars_dist","int_dist","cycles"])
    return f, w

def decide(frame, bars_h, int_h, center_w, thresh, dmean, dstd):
    cg = crop(frame, center_w)
    phv = ph(cg)
    db, di = phv - bars_h, phv - int_h
    mean_lum = float(np.mean(cg)); std_lum = float(np.std(cg))
    is_dark_flat = (mean_lum < dmean) and (std_lum < dstd)
    if (db < thresh) and (not is_dark_flat):
        det = "BARS"
    elif di < thresh:
        det = "INTERFACE"
    else:
        det = "OTHER"
    return det, int(db), int(di), cg, mean_lum, std_lum

def run_loop():
    try:
        bars_h = ph_ref(mon.bars_ref, mon.center_w)
        int_h  = ph_ref(mon.int_ref,  mon.center_w)
    except Exception as e:
        with mon.lock:
            mon.status = f"error: {e}"
            mon.running = False
        return

    # open source
    use_stream = isinstance(mon.stream_path, str) and mon.stream_path.strip() != ""
    if use_stream:
        cap = cv2.VideoCapture(mon.stream_path.strip())
    else:
        src = int(mon.src) if str(mon.src).isdigit() else mon.src
        if isinstance(src, str):
            cap = cv2.VideoCapture(src)  # URL or file path
        else:
            cap = cv2.VideoCapture(int(src), mon.backend)

    # Normalize capture on Windows/OBS to reduce flicker and colorspace issues
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass
    try:
        # MJPG often stabilizes MSMF/DSHOW timing for virtual cams
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    if not cap.isOpened():
        with mon.lock:
            mon.status = "error: cannot open source"
            mon.running = False
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

            det, db, di, crop_img, mean_l, std_l = decide(frame, bars_h, int_h, cw, thr, dmean, dstd)

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
                    elif det == "INTERFACE":
                        mon.count_int  += 1
                    else:
                        mon.count_other+= 1

                    mon.last_seen[det] = now

                    if mon._last == "BARS" and det == "INTERFACE":
                        mon.cycles += 1
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
      <div><label>Stream URL (rtsp/http)</label><input id="stream" placeholder="rtsp://user:pass@host:554/stream" value="{{stream}}"></div>
      <div><label>Center Width</label><input id="cw" type="number" value="{{cw}}"></div>
      <div><label>Threshold</label><input id="thr" type="number" value="{{thr}}"></div>
      <div><label>Stable Frames</label><input id="st" type="number" value="{{st}}"></div>
      <div><label>Dark Mean</label><input id="dm" type="number" value="{{dm}}"></div>
      <div><label>Dark Std</label><input id="ds" type="number" value="{{ds}}"></div>
      <div><label>Bars Ref</label><input id="bars" value="{{bars}}"></div>
      <div><label>Interface Ref</label><input id="intr" value="{{intr}}"></div>
      <div><label>CSV Path</label><input id="csv" value="{{csv}}"></div>
    </div>
    <div style="margin-top:12px; display:flex; gap:8px; align-items:center;">
      <button id="btnStart" onclick="start()">Start</button>
      <button id="btnStop"  class="secondary" onclick="stop()">Stop</button>
      <button id="btnClear" class="secondary" onclick="clearCounts()">Clear</button>
      <label style="display:flex; align-items:center; gap:6px; font-size:13px; color:var(--muted);">
        <input id="liveCropToggle" type="checkbox" checked onchange="toggleThumb()"/>
        Live crop
      </label>
      <div style="flex:1"></div>
      <button class="secondary" onclick="window.location.href='/download'">Download CSV</button>
      <button class="secondary" onclick="shutdown()">Quit</button>
    </div>
  </div>

  <div class="card thumbBox">
    <div>
      <img id="thumb" src="" alt="live crop" width="256" height="256" style="border-radius:8px; border:1px solid var(--border)"/>
    </div>
    <div class="metrics">
      <div><b>Test Frame</b></div>
      <div id="m_db">bars distance: —</div>
      <div id="m_di">interface distance: —</div>
      <div id="m_mean">mean luminance: —</div>
      <div id="m_std">std luminance: —</div>
      <div style="margin-top:8px;">
        <button class="secondary" onclick="peek()">Test frame now</button>
      </div>
      <div style="margin-top:8px; color:#9ca3af; font-size:12px;">Tip: if bars/interface distances are close to the threshold, adjust Threshold or Center Width. Dark/flat frames should fail bars by the dark gates.</div>
    </div>
  </div>

  <div class="row">
    <div class="stat"><div class="label">Status</div><div id="status" class="value">idle</div></div>
    <div class="stat"><div class="label">Not Connected</div><div id="barsCount" class="value">0</div><div class="sub" id="barsSeen">last: —</div></div>
    <div class="stat"><div class="label">Connected</div><div id="intCount" class="value">0</div><div class="sub" id="intSeen">last: —</div></div>
    <div class="stat"><div class="label">Other</div><div id="otherCount" class="value">0</div><div class="sub" id="otherSeen">last: —</div></div>
    <div class="stat"><div class="label">Cycles</div><div id="cyclesCount" class="value">0</div></div>
  </div>

  <script>
    function paintPill(text){
      const pill = document.getElementById('statusPill');
      pill.textContent = text;
      let bg = 'var(--pillIdle)';
      const t = (text || '').toLowerCase();
      if (t.includes('connected') && !t.includes('not')) bg = 'var(--pillConnected)';
      else if (t.includes('not connected')) bg = 'var(--pillNotConnected)';
      else if (t.includes('other')) bg = 'var(--pillOther)';
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

    async function getStatus(){
      const r = await fetch('/status'); const j = await r.json();
      const display = j.status_friendly || j.status;
      document.getElementById('status').textContent = display;
      document.getElementById('barsCount').textContent   = j.count_bars;
      document.getElementById('intCount').textContent    = j.count_int;
      document.getElementById('otherCount').textContent  = j.count_other;
      document.getElementById('cyclesCount').textContent = j.cycles;
      document.getElementById('barsSeen').textContent  = 'last: ' + (j.last_seen.BARS || '—');
      document.getElementById('intSeen').textContent   = 'last: ' + (j.last_seen.INTERFACE || '—');
      document.getElementById('otherSeen').textContent = 'last: ' + (j.last_seen.OTHER || '—');
      paintPill(display);

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
        stream: document.getElementById('stream').value,
        cw:   parseInt(document.getElementById('cw').value),
        thr:  parseInt(document.getElementById('thr').value),
        st:   parseInt(document.getElementById('st').value),
        dm:   parseFloat(document.getElementById('dm').value),
        ds:   parseFloat(document.getElementById('ds').value),
        bars: document.getElementById('bars').value,
        intr: document.getElementById('intr').value,
        csv:  document.getElementById('csv').value,
      };
      await fetch('/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      setTimeout(getStatus, 300);
    }
    async function stop(){ await fetch('/stop', {method:'POST'}); setTimeout(getStatus, 300); }
    async function clearCounts(){ await fetch('/clear', {method:'POST'}); setTimeout(getStatus, 300); }
    async function shutdown(){ await fetch('/shutdown', {method:'POST'}); }
    setInterval(getStatus, 600);
    setInterval(refreshThumb, 1500); // decoupled from status polling
    getStatus();
  </script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML,
        src=mon.src, cw=mon.center_w, thr=mon.thresh, st=mon.stable_frames,
        dm=mon.dark_mean, ds=mon.dark_std, bars=mon.bars_ref, intr=mon.int_ref,
        csv=mon.csv_path, stream=mon.stream_path
    )

@app.get("/ping")
def ping():
    return Response("pong", mimetype="text/plain")

@app.get("/status")
def status():
    with mon.lock:
        sf = label_for(mon.status) if mon.status in ("BARS","INTERFACE","OTHER") else mon.status
        return jsonify(
            status=mon.status, status_friendly=sf, running=mon.running,
            count_bars=mon.count_bars, count_int=mon.count_int,
            count_other=mon.count_other, cycles=mon.cycles,
            last_seen=mon.last_seen
        )

@app.get("/thumb")
def thumb():
    with mon.lock:
        img = mon.last_crop
        if img is None:
            return Response(status=404)
        # Encode as JPEG
        ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return Response(status=500)
    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.get("/peek")
def peek():
    with mon.lock:
        m = mon.last_metrics.copy()
    # ensure numbers or placeholders
    for k in ("db","di","mean","std"):
        if m.get(k) is None:
            m[k] = "-"
    return jsonify(m)

@app.post("/start")
def start():
    cfg = request.get_json(force=True)
    with mon.lock:
        if mon.running:
            return jsonify(ok=True, already=True)
        mon.src       = cfg.get("src", mon.src)
        mon.stream_path = cfg.get("stream", mon.stream_path)
        mon.center_w  = int(cfg.get("cw", mon.center_w))
        mon.thresh    = int(cfg.get("thr", mon.thresh))
        mon.stable_frames = int(cfg.get("st", mon.stable_frames))
        mon.dark_mean = float(cfg.get("dm", mon.dark_mean))
        mon.dark_std  = float(cfg.get("ds", mon.dark_std))
        mon.bars_ref  = cfg.get("bars", mon.bars_ref)
        mon.int_ref   = cfg.get("intr", mon.int_ref)
        mon.csv_path  = cfg.get("csv", mon.csv_path)
        mon.reset_counts_and_roll_csv()  # also rolls CSV
        mon.running = True
        mon.status = "starting"
        mon.last_frame = None
        mon.last_crop  = None
        mon.last_metrics = {"db": None, "di": None, "mean": None, "std": None}
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    return jsonify(ok=True)

@app.post("/stop")
def stop():
    with mon.lock:
        mon.running = False
        mon.status = "stopped"
    return jsonify(ok=True)

@app.post("/clear")
def clear():
    with mon.lock:
        if mon.running:
            return jsonify(error="stop first"), 400
        mon.reset_counts_and_roll_csv()
    return jsonify(ok=True, csv=mon.csv_path)

@app.get("/download")
def download_csv():
    path = os.path.abspath(mon.csv_path)
    if not os.path.isfile(path):
        return jsonify(error="CSV not found", path=path), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.post("/shutdown")
def shutdown():
    # Graceful quit from the web UI
    def _exit():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=_exit, daemon=True).start()
    return jsonify(ok=True)

def open_browser():
    url = f"http://localhost:5055/"
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print(f"Open {url} in your browser if it didn't open automatically.")

if __name__ == "__main__":
    threading.Timer(0.6, open_browser).start()
    print("Starting server on http://localhost:5055 …")
    app.run(host="127.0.0.1", port=5055, debug=False)
