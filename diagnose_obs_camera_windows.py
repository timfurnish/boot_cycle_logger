"""
Diagnostic script to list and examine camera sources on Windows.
Helps identify OBS Virtual Camera and understand how it appears in the system.
"""

import cv2
import platform
import subprocess
import sys

def list_directshow_cameras():
    """List all DirectShow cameras using Windows-specific methods"""
    print("=" * 80)
    print("DIRECTSHOW CAMERA LISTING")
    print("=" * 80)
    
    if platform.system() != "Windows":
        print("This script is for Windows only.")
        return
    
    # Method 1: Try to get camera names using Windows API (if available)
    try:
        import win32com.client
        print("\n[Method 1] Using Windows COM to list cameras:")
        wmi = win32com.client.GetObject("winmgmts:")
        cameras = wmi.InstancesOf("Win32_PnPEntity")
        print("  Found devices:")
        for cam in cameras:
            name = cam.Name
            if name and ("camera" in name.lower() or "video" in name.lower() or "imaging" in name.lower()):
                print(f"    - {name}")
    except ImportError:
        print("\n[Method 1] win32com not available (install pywin32)")
    except Exception as e:
        print(f"\n[Method 1] Error: {e}")
    
    # Method 2: Scan indices with OpenCV and get properties
    print("\n[Method 2] Scanning OpenCV indices with DSHOW backend:")
    print("  Index | Opened | Width x Height | FPS | Backend")
    print("  " + "-" * 70)
    
    for idx in range(11):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap and cap.isOpened():
                try:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Try to get backend name
                    backend_name = "DSHOW"
                    
                    # Try to read a frame to verify it's working
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        actual_h, actual_w = frame.shape[:2]
                        status = f"✓ READ OK ({actual_w}x{actual_h})"
                        if actual_w == 1920 and actual_h == 1080:
                            status += " [OBS Virtual Camera?]"
                    else:
                        status = "✗ Cannot read frames"
                    
                    print(f"  {idx:5d} |   YES  | {w:4d} x {h:4d} | {fps:3.0f} | {backend_name:10s} | {status}")
                except Exception as e:
                    print(f"  {idx:5d} |   YES  | ERROR: {e}")
                finally:
                    cap.release()
            else:
                print(f"  {idx:5d} |   NO   | (not available)")
        except Exception as e:
            print(f"  {idx:5d} | ERROR  | {e}")
        finally:
            if cap:
                try:
                    cap.release()
                except:
                    pass
    
    # Method 3: Try MSMF backend
    print("\n[Method 3] Scanning OpenCV indices with MSMF backend:")
    print("  Index | Opened | Width x Height | FPS | Backend")
    print("  " + "-" * 70)
    
    for idx in range(11):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            if cap and cap.isOpened():
                try:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend_name = "MSMF"
                    
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        actual_h, actual_w = frame.shape[:2]
                        status = f"✓ READ OK ({actual_w}x{actual_h})"
                        if actual_w == 1920 and actual_h == 1080:
                            status += " [OBS Virtual Camera?]"
                    else:
                        status = "✗ Cannot read frames"
                    
                    print(f"  {idx:5d} |   YES  | {w:4d} x {h:4d} | {fps:3.0f} | {backend_name:10s} | {status}")
                except Exception as e:
                    print(f"  {idx:5d} |   YES  | ERROR: {e}")
                finally:
                    cap.release()
            else:
                print(f"  {idx:5d} |   NO   | (not available)")
        except Exception as e:
            print(f"  {idx:5d} | ERROR  | {e}")
        finally:
            if cap:
                try:
                    cap.release()
                except:
                    pass

def test_obs_connection():
    """Test connecting to potential OBS Virtual Camera indices"""
    print("\n" + "=" * 80)
    print("TESTING OBS VIRTUAL CAMERA CONNECTION")
    print("=" * 80)
    
    # Check if OBS is running
    obs_running = False
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq obs64.exe'], 
                              capture_output=True, timeout=2)
        obs_running = result.returncode == 0 and b'obs64.exe' in result.stdout
        if obs_running:
            print("✓ OBS Studio is running")
        else:
            print("✗ OBS Studio is NOT running")
    except Exception as e:
        print(f"⚠ Could not check OBS status: {e}")
    
    # Test indices that commonly have OBS Virtual Camera
    print("\nTesting common OBS Virtual Camera indices (1, 2, 3) with DSHOW:")
    for idx in [1, 2, 3]:
        cap = None
        try:
            print(f"\n  Testing index {idx}:")
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap and cap.isOpened():
                print(f"    ✓ Opened successfully")
                
                # Set properties
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    print(f"    ✓ Set properties (1920x1080, 30fps)")
                except Exception as e:
                    print(f"    ⚠ Could not set properties: {e}")
                
                # Read frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"    ✓ Read frame: {w}x{h}")
                    if w == 1920 and h == 1080:
                        print(f"    ✓✓✓ MATCHES OBS VIRTUAL CAMERA (1920x1080) ✓✓✓")
                    else:
                        print(f"    ✗ Not 1920x1080 - not OBS Virtual Camera")
                else:
                    print(f"    ✗ Cannot read frames")
            else:
                print(f"    ✗ Failed to open")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        finally:
            if cap:
                try:
                    cap.release()
                except:
                    pass

def get_camera_name_windows(idx, backend):
    """Try to get camera name on Windows (limited support)"""
    try:
        # On Windows, OpenCV doesn't provide easy access to camera names
        # We can try to get it via backend-specific properties, but it's limited
        cap = cv2.VideoCapture(idx, backend)
        if cap and cap.isOpened():
            # Try to get backend-specific info
            try:
                # Some backends support getting device name
                name = cap.getBackendName()
                return name
            except:
                pass
            try:
                # Try CAP_PROP_POS_MSEC or other properties that might have info
                # This is very limited on Windows
                return None
            except:
                pass
            cap.release()
    except:
        pass
    return None

def check_directshow_filters():
    """Check DirectShow filters using Windows tools"""
    print("\n" + "=" * 80)
    print("DIRECTSHOW FILTER INFORMATION")
    print("=" * 80)
    
    print("\nDirectShow filters are registered in Windows registry.")
    print("OBS Virtual Camera registers itself as a DirectShow filter when OBS starts.")
    print("\nTo see DirectShow filters, you can use GraphEdit (Windows SDK tool)")
    print("or check registry at: HKEY_LOCAL_MACHINE\\SOFTWARE\\Classes\\CLSID")
    print("\nOBS Virtual Camera typically appears as:")
    print("  - A video capture filter")
    print("  - Named 'OBS Virtual Camera' or similar")
    print("  - Supports 1920x1080 resolution")
    print("  - Uses DirectShow (DSHOW) backend")

def explain_windows_vs_mac():
    """Explain why Windows is different from Mac"""
    print("\n" + "=" * 80)
    print("WHY WINDOWS IS DIFFERENT FROM MAC")
    print("=" * 80)
    print("""
On macOS (AVFoundation):
  - OpenCV can access camera NAMES (e.g., "FaceTime HD Camera", "OBS Virtual Camera")
  - We can identify cameras by name, making OBS Virtual Camera easy to find
  - Camera indices are more stable

On Windows (DirectShow):
  - OpenCV typically only provides INDICES (0, 1, 2, etc.), not names
  - Camera names are not easily accessible via OpenCV
  - Camera indices can change when devices are plugged/unplugged
  - OBS Virtual Camera appears as just another DirectShow camera filter
  - No special name or identifier - we must identify by characteristics:
    * DSHOW backend
    * 1920x1080 native resolution
    * Can read frames successfully

How OBS Virtual Camera Works:
  - OBS registers a DirectShow filter when it starts
  - This filter appears in the DirectShow graph as a video capture device
  - It's assigned an index by Windows (typically 1, 2, or 3, but can vary)
  - The filter provides 1920x1080 video frames from OBS's output
  - When OBS stops, the filter is unregistered

Why It's Difficult:
  - We can't identify by name (like on Mac)
  - We must rely on resolution (1920x1080) and backend (DSHOW)
  - But other cameras might also support 1920x1080
  - The index assignment is not guaranteed to be consistent
  - We need to scan and verify each camera individually
    """)

def main():
    print("OBS Virtual Camera Diagnostic Tool for Windows")
    print("=" * 80)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"OpenCV Version: {cv2.__version__}")
    print()
    
    explain_windows_vs_mac()
    list_directshow_cameras()
    test_obs_connection()
    check_directshow_filters()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nLook for cameras with:")
    print("  - NATIVE 1920x1080 resolution (not forced)")
    print("  - DSHOW backend")
    print("  - Can read frames successfully")
    print("  - Typically at indices 1, 2, or 3 when OBS is running")
    print("\nThis should identify OBS Virtual Camera.")
    print("\nIf OBS Virtual Camera is not found:")
    print("  1. Make sure OBS Studio is running")
    print("  2. Start OBS Virtual Camera (Tools > Start Virtual Camera)")
    print("  3. Check that it's actually outputting 1920x1080")
    print("  4. Try the diagnostic script again")

if __name__ == "__main__":
    main()

