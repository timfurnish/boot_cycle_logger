# obs_camera_web.py
"""
OBS Virtual Camera Web Viewer

Purpose:
Creates a web server that displays the OBS Virtual Camera feed in a browser at 1920x1080 resolution.

How it works:
- Flask Web App: Serves a webpage with the camera feed
- OpenCV Video Capture: Captures frames from OBS Virtual Camera using DirectShow on Windows
- MJPEG Streaming: Streams video frames to the browser using MJPEG format for reliable display

How to run it:
1) Make sure OBS Studio is running and Virtual Camera is started
2) Activate your Python env with required packages (opencv-python, Flask)
3) Run: python OBS.py
4) Your browser should open to http://localhost:8080/ (if not, open it manually)
5) The camera feed will display at 1920x1080 resolution

Note: On Windows, OBS Virtual Camera typically appears as a DirectShow device.
You may need to adjust the camera_index (default: 1) if it's not detected correctly.
"""

import os
import sys
import threading
import time
import platform
import webbrowser
import signal
import atexit
import cv2
from flask import Flask, Response, render_template_string

# Configuration
CAMERA_INDEX = 1  # OBS Virtual Camera index (try 0, 1, 2, etc. if this doesn't work)
WIDTH = 1920
HEIGHT = 1080
PORT = 8080

# State management
class CameraStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.cap = None
        self.running = False
        self.backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.camera_index = CAMERA_INDEX
        self.width = WIDTH
        self.height = HEIGHT

    def start_capture(self):
        """Initialize and start video capture"""
        with self.lock:
            if self.cap is not None:
                return False
            
            # Try to open camera
            if platform.system() == "Windows":
                self.cap = cv2.VideoCapture(self.camera_index, self.backend)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Try to set FPS (optional, helps with stability)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Read actual resolution (may be different from requested)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera opened: {actual_width}x{actual_height}")
            
            self.running = True
            return True

    def stop_capture(self):
        """Stop and release video capture"""
        with self.lock:
            self.running = False
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def get_frame(self):
        """Capture a single frame"""
        with self.lock:
            if self.cap is None or not self.running:
                return None
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None
            
            # Resize to exact dimensions if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            
            return frame

# Global camera stream instance
camera = CameraStream()

# Flask app
app = Flask(__name__)

def generate_frames():
    """Generator function for video streaming"""
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.033)  # ~30 FPS
            continue
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Main page with video feed"""
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>OBS Virtual Camera Viewer</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    body {
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #0a0a0a;
      color: #e5e7eb;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 20px;
    }
    h1 {
      color: #38bdf8;
      margin-bottom: 20px;
      font-size: 24px;
    }
    .video-container {
      background: #111827;
      border: 2px solid #1f2937;
      border-radius: 12px;
      padding: 8px;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
      max-width: 100%;
      overflow: hidden;
    }
    img {
      display: block;
      width: 100%;
      height: auto;
      max-width: 1920px;
      max-height: 1080px;
      object-fit: contain;
    }
    .status {
      margin-top: 16px;
      padding: 12px 20px;
      background: #1f2937;
      border-radius: 8px;
      font-size: 14px;
      color: #94a3b8;
    }
    .status.error {
      background: #7f1d1d;
      color: #fca5a5;
    }
    .status.success {
      background: #1e3a2e;
      color: #86efac;
    }
  </style>
</head>
<body>
  <h1>OBS Virtual Camera</h1>
  <div class="video-container">
    <img id="video" src="/video_feed" alt="Camera Feed" />
  </div>
  <div id="status" class="status">Connecting...</div>
  
  <script>
    const img = document.getElementById('video');
    const status = document.getElementById('status');
    
    let errorCount = 0;
    const maxErrors = 5;
    
    img.onload = function() {
      errorCount = 0;
      status.textContent = 'Connected - 1920x1080';
      status.className = 'status success';
    };
    
    img.onerror = function() {
      errorCount++;
      if (errorCount >= maxErrors) {
        status.textContent = 'Error: Cannot connect to camera feed. Make sure OBS Virtual Camera is running.';
        status.className = 'status error';
      } else {
        status.textContent = 'Connecting... (' + errorCount + '/' + maxErrors + ')';
        status.className = 'status';
      }
      // Retry after a short delay
      setTimeout(() => {
        img.src = '/video_feed?t=' + new Date().getTime();
      }, 1000);
    };
    
    // Refresh connection periodically to handle disconnects
    setInterval(() => {
      if (errorCount === 0) {
        img.src = '/video_feed?t=' + new Date().getTime();
      }
    }, 30000); // Refresh every 30 seconds
  </script>
</body>
</html>
"""
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """API endpoint to check camera status"""
    with camera.lock:
        is_open = camera.cap is not None and camera.cap.isOpened()
        return {
            'running': camera.running,
            'camera_open': is_open,
            'resolution': f'{camera.width}x{camera.height}'
        }

def open_browser():
    """Open browser to the web page"""
    url = f"http://localhost:{PORT}/"
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print(f"Open {url} in your browser if it didn't open automatically.")

def graceful_exit(*_args):
    """Clean shutdown handler"""
    try:
        camera.stop_capture()
    except Exception:
        pass
    time.sleep(0.25)
    os._exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_exit)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, graceful_exit)
atexit.register(graceful_exit)

if __name__ == "__main__":
    print("OBS Virtual Camera Web Viewer")
    print("=" * 40)
    print(f"Camera Index: {CAMERA_INDEX}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Port: {PORT}")
    print("=" * 40)
    
    # Start camera capture
    if not camera.start_capture():
        print(f"\nERROR: Could not open camera at index {CAMERA_INDEX}")
        print("\nTroubleshooting:")
        print("1. Make sure OBS Studio is running")
        print("2. Start OBS Virtual Camera (Tools > Start Virtual Camera)")
        print("3. Try different camera indices (0, 1, 2, etc.)")
        print("   Edit CAMERA_INDEX in the script and try again")
        sys.exit(1)
    
    # Open browser
    threading.Timer(0.6, open_browser).start()
    
    print(f"\nStarting server on http://localhost:{PORT}/")
    print("Press Ctrl+C to stop\n")
    
    try:
        app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop_capture()
