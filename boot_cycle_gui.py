# pip install PySide6 opencv-python pillow ImageHash
import sys, os, time, csv, platform
import cv2, numpy as np
from PIL import Image
import imagehash as ih
from datetime import datetime
# --- macOS Qt (PyQt6) plugin path fix for virtualenvs ---
# Ensure Qt can find the 'cocoa' platform plugin when running from a venv.
import PyQt6
_plugins_root = os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins")
_platforms_dir = os.path.join(_plugins_root, "platforms")
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORMTHEME", None)
os.environ["QT_PLUGIN_PATH"] = _plugins_root
if sys.platform == "darwin":
    os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")
from PyQt6 import QtCore as _QtCore
_QtCore.QCoreApplication.setLibraryPaths([_plugins_root])
# -------------------------------------------------------

from PyQt6 import QtCore, QtWidgets

def crop(bgr, cw=1080):
    h,w,_=bgr.shape; s=max(1,(w-cw)//2)
    g=cv2.cvtColor(np.concatenate([bgr[:,:s],bgr[:,w-s:]],1), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g,(256,256))

def ph_ref(path, cw):
    img = cv2.imread(path)
    if img is None: raise SystemExit(f"Failed to read reference image: {path}")
    return ih.phash(Image.fromarray(crop(img, cw)))

def label_for(state):
    return "Device Not Connected" if state=="BARS" else "Device Connected" if state=="INTERFACE" else "other"

class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boot/Interface Counter")
        self.resize(520, 260)

        self.srcBox  = QtWidgets.QLineEdit("1")
        self.barsRef = QtWidgets.QLineEdit("Comp-Scope-Disconnected-ColorBars.png")
        self.intRef  = QtWidgets.QLineEdit("Comp-Still-&-Video-capture-disabled.png")
        self.centerW = QtWidgets.QSpinBox(); self.centerW.setRange(100, 4096); self.centerW.setValue(1080)
        self.thresh  = QtWidgets.QSpinBox(); self.thresh.setRange(1, 64); self.thresh.setValue(10)
        self.stable  = QtWidgets.QSpinBox(); self.stable.setRange(1, 30); self.stable.setValue(3)
        self.darkMean= QtWidgets.QDoubleSpinBox(); self.darkMean.setRange(0,255); self.darkMean.setValue(22)
        self.darkStd = QtWidgets.QDoubleSpinBox();  self.darkStd.setRange(0,255); self.darkStd.setValue(12)
        self.csvPath = QtWidgets.QLineEdit("logs/boot_log.csv")

        self.startBtn= QtWidgets.QPushButton("Start")
        self.stopBtn = QtWidgets.QPushButton("Stop"); self.stopBtn.setEnabled(False)

        self.lblState= QtWidgets.QLabel("—"); self.lblState.setStyleSheet("font: 700 16px;")
        self.cBars   = QtWidgets.QLabel("0"); self.cInt = QtWidgets.QLabel("0"); self.cOther = QtWidgets.QLabel("0"); self.cCycles = QtWidgets.QLabel("0")

        form = QtWidgets.QFormLayout()
        form.addRow("Source (index/URL):", self.srcBox)
        form.addRow("Bars ref:", self.barsRef)
        form.addRow("Interface ref:", self.intRef)
        form.addRow("Center width:", self.centerW)
        form.addRow("Threshold:", self.thresh)
        form.addRow("Stable frames:", self.stable)
        form.addRow("Dark mean:", self.darkMean)
        form.addRow("Dark std:", self.darkStd)
        form.addRow("CSV path:", self.csvPath)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Status:"),0,0); grid.addWidget(self.lblState,0,1,1,3)
        grid.addWidget(QtWidgets.QLabel("Not Connected:"),1,0); grid.addWidget(self.cBars,1,1)
        grid.addWidget(QtWidgets.QLabel("Connected:"),1,2); grid.addWidget(self.cInt,1,3)
        grid.addWidget(QtWidgets.QLabel("Other:"),2,0); grid.addWidget(self.cOther,2,1)
        grid.addWidget(QtWidgets.QLabel("Cycles:"),2,2); grid.addWidget(self.cCycles,2,3)

        btns = QtWidgets.QHBoxLayout(); btns.addWidget(self.startBtn); btns.addWidget(self.stopBtn)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form); lay.addLayout(grid); lay.addLayout(btns)

        self.startBtn.clicked.connect(self.start)
        self.stopBtn.clicked.connect(self.stop)

        self.timer = QtCore.QTimer(self); self.timer.setInterval(200); self.timer.timeout.connect(self.tick)
        self.cap = None; self.boot_h = None; self.int_h = None
        self.state="UNKNOWN"; self.last="UNKNOWN"; self.stableCnt=0
        self.countBars=self.countInt=self.countOther=self.cycles=0
        self.csvWriter=None; self.csvFile=None
        self.backend = cv2.CAP_DSHOW if platform.system()=="Windows" else cv2.CAP_AVFOUNDATION

    def open_csv(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.csvFile = open(path, "a", newline="")
        self.csvWriter = csv.writer(self.csvFile)
        if self.csvFile.tell()==0:
            self.csvWriter.writerow(["ts","status","bars_dist","int_dist","cycles"])

    def start(self):
        try:
            cw = self.centerW.value()
            self.boot_h = ph_ref(self.barsRef.text(), cw)
            self.int_h  = ph_ref(self.intRef.text(),  cw)
        except SystemExit as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e)); return

        srcText = self.srcBox.text().strip()
        src = int(srcText) if srcText.isdigit() else srcText
        self.cap = cv2.VideoCapture(src if isinstance(src,str) else int(src), self.backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot open source."); return

        self.open_csv(self.csvPath.text())
        self.state=self.last="UNKNOWN"; self.stableCnt=0
        self.timer.start()
        self.startBtn.setEnabled(False); self.stopBtn.setEnabled(True)

    def stop(self):
        self.timer.stop()
        if self.cap: self.cap.release(); self.cap=None
        if self.csvFile: self.csvFile.close(); self.csvFile=None; self.csvWriter=None
        self.startBtn.setEnabled(True); self.stopBtn.setEnabled(False)

    def decide(self, frame):
        cw = self.centerW.value()
        cg = crop(frame, cw)
        phv = ih.phash(Image.fromarray(cg))
        db, di = phv - self.boot_h, phv - self.int_h
        mean_lum = float(np.mean(cg)); std_lum = float(np.std(cg))
        is_dark_flat = (mean_lum < self.darkMean.value()) and (std_lum < self.darkStd.value())
        if (db < self.thresh.value()) and (not is_dark_flat):
            det = "BARS"
        elif di < self.thresh.value():
            det = "INTERFACE"
        else:
            det = "OTHER"
        return det, int(db), int(di)

    def tick(self):
        ok, frame = self.cap.read()
        if not ok: return
        det, db, di = self.decide(frame)
        self.stableCnt = self.stableCnt+1 if det==self.state else 1
        self.state = det if det!=self.state else self.state
        if self.stableCnt == self.stable.value() and self.state != self.last:
            if self.last=="BARS" and self.state=="INTERFACE": self.cycles += 1
            self.last = self.state
            lab = label_for(self.state)
            if self.state=="BARS": self.countBars += 1
            elif self.state=="INTERFACE": self.countInt += 1
            else: self.countOther += 1
            ts = datetime.now().isoformat(timespec="seconds")
            self.csvWriter.writerow([ts, lab, db, di, self.cycles]); self.csvFile.flush()
            self.lblState.setText(lab)
            self.cBars.setText(str(self.countBars))
            self.cInt.setText(str(self.countInt))
            self.cOther.setText(str(self.countOther))
            self.cCycles.setText(str(self.cycles))

    def closeEvent(self, e):
        self.stop(); e.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = App(); w.show()
    sys.exit(app.exec())