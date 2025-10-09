# pip install opencv-python pillow ImageHash
import cv2, csv, os, time, argparse, numpy as np
from PIL import Image
import imagehash as ih

from datetime import datetime

def status_label(state: str) -> str:
    if state == "BARS":
        return "Device Not Connected"
    if state == "INTERFACE":
        return "Device Connected"
    return "other"

def ph_ref(path, cw):
    img = cv2.imread(path)  # BGR
    if img is None:
        raise SystemExit(f"Failed to read reference image: {path}")
    g = crop(img, cw)  # cropped + resized grayscale
    return ih.phash(Image.fromarray(g))
def crop(bgr, cw=1080):
    h,w,_=bgr.shape; s=max(1,(w-cw)//2)
    g=cv2.cvtColor(np.concatenate([bgr[:,:s],bgr[:,w-s:]],1), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g,(256,256))
def ph_frame(bgr,cw): return ih.phash(Image.fromarray(crop(bgr,cw)))

ap=argparse.ArgumentParser()
ap.add_argument("--src",required=True,help="rtsp/rtmp/http URL or camera index (e.g. 0)")
ap.add_argument("--boot_ref",default="Comp-Scope-Disconnected-ColorBars.png")
ap.add_argument("--int_ref", default="Comp-Still-&-Video-capture-disabled.png")
ap.add_argument("--center_w",type=int,default=1080)
ap.add_argument("--threshold",type=int,default=10)
ap.add_argument("--fps",type=float,default=3.0)
ap.add_argument("--stable",type=int,default=3)
ap.add_argument("--csv",default="logs/boot_log.csv")
ap.add_argument("--debug", action="store_true", help="Print per-frame distances and detected state")
args=ap.parse_args()

boot_h = ph_ref(args.boot_ref, args.center_w)
int_h  = ph_ref(args.int_ref,  args.center_w)
src = int(args.src) if args.src.isdigit() else args.src
cap = cv2.VideoCapture(src if isinstance(src, str) else int(src), cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened(): raise SystemExit("Cannot open stream (check --src).")

os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
state=last="UNKNOWN"; stable=0; cycles=0; period=max(0.1,1.0/args.fps)
last_debug = 0
with open(args.csv,"a",newline="") as f:
    w=csv.writer(f); 
    if f.tell()==0: w.writerow(["ts","status","bars_dist","int_dist","cycles"])
    print("Monitoring… Ctrl+C to stop")
    while True:
        t0=time.time()
        ok,frame=cap.read()
        if not ok: time.sleep(1); continue
        # compute grayscale sidebar crop once for both pHash and luminance stats
        cg = crop(frame, args.center_w)              # grayscale, 256x256
        phv = ih.phash(Image.fromarray(cg))
        db, di = phv - boot_h, phv - int_h
        # Treat very dark/flat sidebars as OTHER (e.g., full-black screen)
        mean_lum = float(np.mean(cg))
        std_lum  = float(np.std(cg))
        is_dark_flat = (mean_lum < 22.0) and (std_lum < 12.0)
        if (db < args.threshold) and (not is_dark_flat):
            det = "BARS"
        elif di < args.threshold:
            det = "INTERFACE"
        else:
            det = "OTHER"
        if args.debug and (time.time() - last_debug) >= 1.0:
            print(f"dbg dist bars={int(db)} int={int(di)} mean={mean_lum:.1f} std={std_lum:.1f} det={status_label(det)}")
            last_debug = time.time()
        stable = stable+1 if det==state else 1; state=det if det!=state else state
        if stable==args.stable and state!=last:
            if last == "BARS" and state == "INTERFACE": 
                cycles += 1
            last=state; ts=datetime.now().isoformat(timespec="seconds")
            label = status_label(state)
            w.writerow([ts,label,int(db),int(di),cycles]); f.flush()
            print(f"{ts}  {label:18s}  bars={int(db)} int={int(di)} cycles={cycles}")
            last_debug = time.time()
        dt=time.time()-t0
        if dt<period: time.sleep(period-dt)