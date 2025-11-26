#!/usr/bin/env python3
"""
Comprehensive Camera Diagnostic Tool for "Flicker" PC
Specifically designed to debug OBS Virtual Camera detection issues

This script will:
1. Test OpenCV detection with multiple backends (DSHOW, MSMF, ANY)
2. Enumerate camera indices 0-30
3. Check Windows device registry for OBS Virtual Camera
4. Verify OBS Studio is running and Virtual Camera is started
5. Test both index-based and name-based camera access
6. Provide actionable troubleshooting steps
"""

import cv2
import platform
import subprocess
import sys

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_section(text):
    """Print a formatted section"""
    print(f"\n--- {text} ---")

def test_opencv_backends():
    """Test camera detection with multiple OpenCV backends"""
    print_section("OpenCV Backend Testing")
    
    backends = [
        ("DSHOW (DirectShow)", cv2.CAP_DSHOW),
        ("MSMF (Media Foundation)", cv2.CAP_MSMF),
        ("ANY (Auto-detect)", cv2.CAP_ANY)
    ]
    
    results = {}
    
    for backend_name, backend_code in backends:
        print(f"\nTesting with {backend_name}...")
        backend_results = []
        
        # Test indices 0-30 (expanded range)
        for idx in range(31):
            try:
                cap = cv2.VideoCapture(idx, backend_code)
                
                if cap.isOpened():
                    # Try to read multiple frames to detect flickering/unstable cameras
                    import numpy as np
                    frame_means = []
                    first_frame = None
                    
                    for i in range(5):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            if first_frame is None:
                                first_frame = frame
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame_means.append(float(np.mean(gray)))
                    
                    if first_frame is not None and len(frame_means) >= 3:
                        h, w = first_frame.shape[:2]
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        backend_used = cap.getBackendName()
                        
                        # Calculate variance to detect flickering
                        mean_variance = np.std(frame_means)
                        is_stable = mean_variance < 50
                        
                        info = {
                            'index': idx,
                            'resolution': f"{w}x{h}",
                            'fps': fps,
                            'backend': backend_used,
                            'works': True,
                            'stable': is_stable,
                            'variance': mean_variance
                        }
                        backend_results.append(info)
                        
                        stability_icon = "✓" if is_stable else "⚠"
                        stability_note = "" if is_stable else f" [UNSTABLE! Variance={mean_variance:.1f}]"
                        print(f"  {stability_icon} Source {idx}: {w}x{h} @ {fps:.1f}fps (Backend: {backend_used}){stability_note}")
                    
                    cap.release()
            except Exception as e:
                # Silently skip errors for non-existent cameras
                pass
        
        results[backend_name] = backend_results
        
        if not backend_results:
            print(f"  ✗ No cameras found with {backend_name}")
    
    return results

def test_obs_by_name():
    """Test OBS Virtual Camera detection by device name"""
    print_section("OBS Virtual Camera Name Detection")
    
    obs_names = [
        "OBS Virtual Camera",
        "OBS-Camera",
        "OBS Virtual Source",
        "OBS Virtual Camera (DirectShow)",
        "OBS Virtual Camera (Media Foundation)",
        "obs-camera",
        "OBS-CAMERA",
        "OBS VIRTUAL CAMERA"
    ]
    
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY)
    ]
    
    found = False
    
    for obs_name in obs_names:
        for backend_name, backend_code in backends:
            try:
                print(f"  Trying '{obs_name}' with {backend_name}...", end=" ")
                cap = cv2.VideoCapture(obs_name, backend_code)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        cap.release()
                        print(f"✓ FOUND! ({w}x{h})")
                        found = True
                        return True
                    cap.release()
                    print("✗ Opened but no frame")
                else:
                    print("✗ Not found")
            except Exception as e:
                print(f"✗ Error: {e}")
    
    if not found:
        print("\n  ⚠ OBS Virtual Camera NOT FOUND by name")
    
    return found

def check_windows_devices():
    """Check Windows device manager for OBS Virtual Camera"""
    print_section("Windows Device Manager Check")
    
    if platform.system() != "Windows":
        print("  Skipping (not Windows)")
        return
    
    try:
        # Check for video devices using PowerShell
        print("\n  Querying Windows PnP devices...")
        result = subprocess.run([
            'powershell', '-Command',
            'Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -like "*camera*" -or $_.Name -like "*video*" -or $_.Name -like "*OBS*"} | Select-Object Name, Status, DeviceID | Format-List'
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"  ✗ Error: {result.stderr}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")

def check_obs_running():
    """Check if OBS Studio is running"""
    print_section("OBS Studio Process Check")
    
    if platform.system() != "Windows":
        print("  Skipping (not Windows)")
        return
    
    try:
        result = subprocess.run([
            'tasklist', '/FI', 'IMAGENAME eq obs64.exe'
        ], capture_output=True, text=True, timeout=5)
        
        if 'obs64.exe' in result.stdout:
            print("  ✓ OBS Studio (obs64.exe) is RUNNING")
            
            # Check if Virtual Camera is active
            print("\n  NOTE: Make sure Virtual Camera is STARTED in OBS:")
            print("    1. Open OBS Studio")
            print("    2. Click 'Start Virtual Camera' button")
            print("    3. Button should show 'Stop Virtual Camera' when active")
        else:
            print("  ✗ OBS Studio (obs64.exe) is NOT RUNNING")
            print("\n  ⚠ ACTION REQUIRED:")
            print("    1. Start OBS Studio")
            print("    2. Click 'Start Virtual Camera' button")
            print("    3. Run this diagnostic again")
    except Exception as e:
        print(f"  ✗ Exception: {e}")

def check_registry():
    """Check Windows registry for OBS Virtual Camera entries"""
    print_section("Windows Registry Check (OBS Virtual Camera)")
    
    if platform.system() != "Windows":
        print("  Skipping (not Windows)")
        return
    
    try:
        # Check for OBS registry keys
        print("\n  Checking DirectShow filter registration...")
        result = subprocess.run([
            'reg', 'query', 'HKLM\\SOFTWARE\\Classes\\CLSID', '/s', '/f', 'OBS'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and 'OBS' in result.stdout:
            print("  ✓ OBS Virtual Camera registry entries found")
            # Print first 500 chars of output
            print(result.stdout[:500])
        else:
            print("  ✗ OBS Virtual Camera registry entries NOT FOUND")
            print("\n  ⚠ POSSIBLE FIX:")
            print("    1. Uninstall OBS Studio completely")
            print("    2. Restart Windows")
            print("    3. Reinstall OBS Studio")
            print("    4. Make sure 'Virtual Camera' component is selected during install")
    except Exception as e:
        print(f"  ✗ Exception: {e}")

def generate_report(opencv_results, obs_found_by_name):
    """Generate a comprehensive diagnostic report"""
    print_header("DIAGNOSTIC SUMMARY")
    
    # Count total cameras found
    total_cameras = sum(len(cams) for cams in opencv_results.values())
    unstable_cameras = []
    
    print(f"\nTotal Cameras Found: {total_cameras}")
    
    for backend_name, cameras in opencv_results.items():
        print(f"\n  {backend_name}: {len(cameras)} camera(s)")
        for cam in cameras:
            stability = "✓ STABLE" if cam.get('stable', True) else f"⚠ UNSTABLE (variance={cam.get('variance', 0):.1f})"
            print(f"    - Source {cam['index']}: {cam['resolution']} @ {cam['fps']:.1f}fps - {stability}")
            if not cam.get('stable', True):
                unstable_cameras.append((backend_name, cam['index'], cam.get('variance', 0)))
    
    print(f"\nOBS Virtual Camera (by name): {'✓ FOUND' if obs_found_by_name else '✗ NOT FOUND'}")
    
    print_header("RECOMMENDED ACTIONS")
    
    # Report on unstable cameras first (critical issue)
    if unstable_cameras:
        print("""
  ⚠ CRITICAL: UNSTABLE CAMERAS DETECTED!
  
  The following camera sources are FLICKERING/UNSTABLE:
        """)
        for backend, idx, variance in unstable_cameras:
            print(f"    - Source {idx} with {backend} (variance={variance:.1f})")
        
        print("""
  This means the camera feed is switching between different devices!
  This is exactly what's happening on the "Flicker" PC.
  
  CAUSE: Multiple cameras (e.g., OBS Virtual Camera + Laptop Camera) are
         competing for the same device index in Windows.
  
  SOLUTIONS (in order of preference):
  
  1. TRY A DIFFERENT SOURCE INDEX:
     - The diagnostic found cameras at multiple indices
     - Try each STABLE camera index (marked ✓ STABLE above)
     - Use "Connect to Camera" in the app with different Source values
  
  2. TRY THE OTHER BACKEND:
     - If MSMF is unstable, try DSHOW (or vice versa)
     - DSHOW usually works better for OBS Virtual Camera
     - MSMF sometimes enumerates cameras differently
  
  3. DISABLE OTHER CAMERAS:
     - Go to Device Manager → Cameras
     - Disable the built-in laptop camera temporarily
     - This forces Windows to only enumerate OBS Virtual Camera
     - Re-enable it after testing
  
  4. USE USB CAMERA:
     - If flickering persists, use a physical USB camera instead
     - Physical cameras are more stable than virtual cameras
  
  5. REINSTALL OBS VIRTUAL CAMERA:
     - Uninstall OBS Studio completely
     - Restart Windows
     - Reinstall OBS Studio with Virtual Camera component
     - This may fix the device enumeration
        """)
    
    if not total_cameras:
        print("""
  ⚠ CRITICAL: No cameras detected at all!
  
  Troubleshooting steps:
  1. Check Device Manager - are there any cameras listed?
  2. Update camera drivers
  3. Try a different USB port (for external cameras)
  4. Restart Windows
  5. Check if cameras work in other apps (Zoom, Teams, Camera app)
        """)
    
    elif not obs_found_by_name:
        print("""
  ⚠ OBS Virtual Camera not detected by name
  
  Possible solutions:
  
  A. VERIFY OBS IS RUNNING:
     1. Start OBS Studio
     2. Click "Start Virtual Camera" button
     3. Run this diagnostic again
  
  B. REINSTALL OBS VIRTUAL CAMERA:
     1. Close OBS Studio
     2. Go to Tools → Virtual Camera → Install
     3. Restart OBS
     4. Start Virtual Camera
     5. Run this diagnostic again
  
  C. REINSTALL OBS STUDIO:
     1. Uninstall OBS Studio completely
     2. Restart Windows
     3. Download latest OBS from obsproject.com
     4. Install with "Virtual Camera" component checked
     5. Run this diagnostic again
  
  D. TRY MANUAL CAMERA INDEX:
     Since OpenCV found cameras, you can try connecting manually:
        """)
        
        # Show which indices were found
        for backend_name, cameras in opencv_results.items():
            if cameras:
                print(f"\n     {backend_name} cameras:")
                for cam in cameras:
                    print(f"       - Try Source {cam['index']} with backend {backend_name}")
    
    else:
        print("""
  ✓ OBS Virtual Camera detected successfully!
  
  Your system should work with the Boot Cycle Logger.
  Use the "Auto-Detect OBS" button in the application.
        """)

def main():
    """Main diagnostic routine"""
    print_header("FLICKER PC DIAGNOSTIC TOOL")
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    
    # Run all diagnostic tests
    check_obs_running()
    opencv_results = test_opencv_backends()
    obs_found_by_name = test_obs_by_name()
    check_windows_devices()
    check_registry()
    
    # Generate final report
    generate_report(opencv_results, obs_found_by_name)
    
    print_header("DIAGNOSTIC COMPLETE")
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()

