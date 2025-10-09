# GUI without Qt: Tkinter-based, cross-platform
# pip install opencv-python pillow ImageHash
import os, sys, csv, time, platform, threading
import cv2, numpy as np
from PIL import Image
import imagehash as ih
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

CENTER_W_DEFAULT = 1080
THRESH_DEFAULT   = 10
STABLE_DEFAULT   = 3
DARK_MEAN_DEF    = 22.0
DARK_STD_DEF     = 12.0

def crop(bgr, cw=CENTER_W_DEFAULT):
    h,w,_ = bgr.shape
    s = max(1, (w - cw)//2)
    g = cv2.cvtColor(np.concatenate([bgr[:,:s], bgr[:,w-s:]], 1), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (256,256))

def phash_img(gray_crop):
    return ih.phash(Image.fromarray(gray_crop))

def ph_ref(path, cw):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read reference image: {path}")
    return phash_img(crop(img, cw))

def label_for(state):
    return "Device Not Connected" if state=="BARS" else "Device Connected" if state=="INTERFACE" else "other"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.minsize(540, 300)
        self.after(50, self.lift)
        self.after(100, self.update_idletasks)

        self.title("Boot/Interface Counter (Tk)")
        self.geometry("560x320")
        self.configure(bg="#20232a")

        pad = {"padx":8, "pady":4}

        # Fallback banner (plain Tk) to verify rendering regardless of ttk theme
        top_banner = tk.Frame(self, bg="#20232a", height=40)
        top_banner.pack(fill="x")
        banner_lbl = tk.Label(top_banner, text="Boot Cycle Logger GUI (Tk)", fg="#61dafb", bg="#20232a", font=("Helvetica", 16, "bold"))
        banner_lbl.pack(pady=6)

        debug_lbl = tk.Label(self, text="[GUI Active]", fg="white", bg="#20232a", font=("Helvetica", 12))
        debug_lbl.pack(fill="x", pady=10)

        # Controls
        self.var_src     = tk.StringVar(value="1")
        self.var_barsref = tk.StringVar(value="Comp-Scope-Disconnected-ColorBars.png")
        self.var_intref  = tk.StringVar(value="Comp-Still-&-Video-capture-disabled.png")
        self.var_centerw = tk.IntVar(value=CENTER_W_DEFAULT)
        self.var_thresh  = tk.IntVar(value=THRESH_DEFAULT)
        self.var_stable  = tk.IntVar(value=STABLE_DEFAULT)
        self.var_dmean   = tk.DoubleVar(value=DARK_MEAN_DEF)
        self.var_dstd    = tk.DoubleVar(value=DARK_STD_DEF)
        self.var_csv     = tk.StringVar(value="logs/boot_log.csv")

        frm = tk.Frame(self, bg="#20232a"); frm.pack(fill="x", expand=True, **pad)
        def row(lbl, widget):
            r = tk.Frame(frm, bg="#20232a"); r.pack(fill="x")
            tk.Label(r, text=lbl, width=20, fg="white", bg="#20232a").pack(side="left")
            widget.pack(side="left", fill="x", expand=True)
        row("Source (index/URL):", tk.Entry(frm, textvariable=self.var_src, width=20, fg="white", bg="#30343a", insertbackground="white"))
        row("Bars ref:",          tk.Entry(frm, textvariable=self.var_barsref, width=40, fg="white", bg="#30343a", insertbackground="white"))
        row("Interface ref:",     tk.Entry(frm, textvariable=self.var_intref,  width=40, fg="white", bg="#30343a", insertbackground="white"))
        row("Center width:",      tk.Spinbox(frm, from_=100, to=4096, textvariable=self.var_centerw, width=8, fg="white", bg="#30343a", insertbackground="white"))
        row("Threshold:",         tk.Spinbox(frm, from_=1, to=64,   textvariable=self.var_thresh,  width=8, fg="white", bg="#30343a", insertbackground="white"))
        row("Stable frames:",     tk.Spinbox(frm, from_=1, to=30,   textvariable=self.var_stable,  width=8, fg="white", bg="#30343a", insertbackground="white"))
        row("Dark mean:",         tk.Spinbox(frm, from_=0, to=255,  increment=1, textvariable=self.var_dmean, width=8, fg="white", bg="#30343a", insertbackground="white"))
        row("Dark std:",          tk.Spinbox(frm, from_=0, to=255,  increment=1, textvariable=self.var_dstd,  width=8, fg="white", bg="#30343a", insertbackground="white"))
        row("CSV path:",          tk.Entry(frm, textvariable=self.var_csv,    width=40, fg="white", bg="#30343a", insertbackground="white"))

        # Status + counters
        grid = tk.Frame(self, bg="#20232a"); grid.pack(fill="x", expand=True, **pad)
        tk.Label(grid, text="Status:", width=12, fg="white", bg="#20232a").grid(row=0, column=0, sticky="w")
        self.lbl_status = tk.Label(grid, text="—", font=("Helvetica", 14, "bold"), fg="#61dafb", bg="#20232a")
        self.lbl_status.grid(row=0, column=1, columnspan=3, sticky="w")

        tk.Label(grid, text="Not Connected:", fg="white", bg="#20232a").grid(row=1, column=0, sticky="e")
        self.var_cBars = tk.IntVar(value=0); tk.Label(grid, textvariable=self.var_cBars, fg="white", bg="#20232a").grid(row=1, column=1, sticky="w")
        tk.Label(grid, text="Connected:", fg="white", bg="#20232a").grid(row=1, column=2, sticky="e")
        self.var_cInt = tk.IntVar(value=0);  tk.Label(grid, textvariable=self.var_cInt, fg="white", bg="#20232a").grid(row=1, column=3, sticky="w")

        tk.Label(grid, text="Other:", fg="white", bg="#20232a").grid(row=2, column=0, sticky="e")
        self.var_cOther = tk.IntVar(value=0); tk.Label(grid, textvariable=self.var_cOther, fg="white", bg="#20232a").grid(row=2, column=1, sticky="w")
        tk.Label(grid, text="Cycles:", fg="white", bg="#20232a").grid(row=2, column=2, sticky="e")
        self.var_cCycles = tk.IntVar(value=0); tk.Label(grid, textvariable=self.var_cCycles, fg="white", bg="#20232a").grid(row=2, column=3, sticky="w")

        # Buttons
        btns = tk.Frame(self, bg="#20232a"); btns.pack(fill="x", expand=True, **pad)
        self.btn_start = tk.Button(btns, text="Start", command=self.start, fg="#20232a", bg="#61dafb", activebackground="#52c0e8")
        self.btn_stop  = tk.Button(btns, text="Stop",  command=self.stop,  state="disabled", fg="#20232a", bg="#61dafb", activebackground="#52c0e8")
        self.btn_start.pack(side="left"); self.btn_stop.pack(side="left", padx=6)

        # Runtime
        self.cap = None
        self.bars_h = None
        self.int_h  = None
        self.state = "UNKNOWN"
        self.last  = "UNKNOWN"
        self.stable = 0
        self.countBars = 0
        self.countInt  = 0
        self.countOther= 0
        self.cycles    = 0
        self.csvFile   = None
        self.csvWriter = None
        self.backend = cv2.CAP_DSHOW if platform.system()=="Windows" else cv2.CAP_AVFOUNDATION
        self._tick_after = None
        self._lock = threading.Lock()

        # Force an initial layout update so widgets appear immediately
        self.update_idletasks()
        self.deiconify()
        self.lift()
        self.focus_force()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def open_csv(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.csvFile = open(path, "a", newline="")
        self.csvWriter = csv.writer(self.csvFile)
        if self.csvFile.tell() == 0:
            self.csvWriter.writerow(["ts","status","bars_dist","int_dist","cycles"])

    def decide(self, frame):
        cw = int(self.var_centerw.get())
        cg = crop(frame, cw)
        phv = phash_img(cg)
        db = phv - self.bars_h
        di = phv - self.int_h
        mean_lum = float(np.mean(cg))
        std_lum  = float(np.std(cg))
        is_dark_flat = (mean_lum < float(self.var_dmean.get())) and (std_lum < float(self.var_dstd.get()))
        thr = int(self.var_thresh.get())
        if (db < thr) and (not is_dark_flat):
            det = "BARS"
        elif di < thr:
            det = "INTERFACE"
        else:
            det = "OTHER"
        return det, int(db), int(di)

    def start(self):
        try:
            cw = int(self.var_centerw.get())
            self.bars_h = ph_ref(self.var_barsref.get(), cw)
            self.int_h  = ph_ref(self.var_intref.get(),  cw)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        srcText = self.var_src.get().strip()
        src = int(srcText) if srcText.isdigit() else srcText
        self.cap = cv2.VideoCapture(src if isinstance(src,str) else int(src), self.backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open source."); return

        self.open_csv(self.var_csv.get())
        self.state = "UNKNOWN"; self.last = "UNKNOWN"; self.stable = 0
        self.countBars=self.countInt=self.countOther=self.cycles=0
        self.update_counts_labels()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.schedule_tick()

    def stop(self):
        if self._tick_after:
            self.after_cancel(self._tick_after); self._tick_after = None
        if self.cap: self.cap.release(); self.cap = None
        if self.csvFile: self.csvFile.close(); self.csvFile=None; self.csvWriter=None
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

    def schedule_tick(self):
        self._tick_after = self.after(200, self.tick)

    def tick(self):
        try:
            ok, frame = self.cap.read()
            if not ok:
                self.schedule_tick(); return
            det, db, di = self.decide(frame)
            self.stable = self.stable + 1 if det == self.state else 1
            self.state = det if det != self.state else self.state
            if self.stable == int(self.var_stable.get()) and self.state != self.last:
                # Cycle logic
                if self.last == "BARS" and self.state == "INTERFACE":
                    self.cycles += 1
                self.last = self.state
                lab = label_for(self.state)
                if self.state == "BARS":
                    self.countBars += 1
                elif self.state == "INTERFACE":
                    self.countInt  += 1
                else:
                    self.countOther+= 1
                ts = datetime.now().isoformat(timespec="seconds")
                if self.csvWriter:
                    self.csvWriter.writerow([ts, lab, db, di, self.cycles])
                    self.csvFile.flush()
                self.lbl_status.config(text=lab)
                self.update_counts_labels()
        finally:
            self.schedule_tick()

    def update_counts_labels(self):
        self.var_cBars.set(self.countBars)
        self.var_cInt.set(self.countInt)
        self.var_cOther.set(self.countOther)
        self.var_cCycles.set(self.cycles)

    def on_close(self):
        self.stop()
        self.destroy()

if __name__ == "__main__":
    try:
        app = App()
        app.after(200, lambda: app.lbl_status.config(text="Ready"))
        app.after(250, app.lift)
        app.after(500, lambda: app.focus_force())
        app.mainloop()
    except tk.TclError as e:
        print("Tk initialization failed:", e)