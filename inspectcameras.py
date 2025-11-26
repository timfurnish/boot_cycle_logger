#!/usr/bin/env python3
"""
Camera Inspection Tool - Windows Edition
Run this on Windows to get a comprehensive camera detection report
"""

import cv2
import platform
import subprocess
import sys

def test_opencv_cameras():
    """Test OpenCV camera detection with different backends"""
    print("=== OpenCV Camera Detection (indices 0-15) ===")
    
    # Test different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto")
    ]
    
    for backend_code, backend_name in backends:
        print(f"\n--- Testing {backend_name} Backend ---")
        for idx in range(16):
            try:
                cap = cv2.VideoCapture(idx, backend_code)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        actual_backend = cap.getBackendName()
                        print(f"  Source {idx}: {w}x{h} @ {fps:.1f}fps, Backend: {actual_backend}")
                    cap.release()
                else:
                    # Try to see if it's a valid device but can't open
                    cap2 = cv2.VideoCapture(idx, backend_code)
                    if cap2.isOpened():
                        print(f"  Source {idx}: Can open but no frame")
                        cap2.release()
            except Exception as e:
                if "out device of bound" not in str(e):
                    print(f"  Source {idx}: Error - {e}")

def test_windows_system_cameras():
    """Test Windows system camera detection"""
    print("\n=== Windows System Camera Detection ===")
    
    # PowerShell command to get camera devices
    commands = [
        {
            'name': 'WMI Camera Devices',
            'cmd': ['powershell', '-Command', 
                   'Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -like "*camera*" -or $_.Name -like "*webcam*" -or $_.Name -like "*video*"} | Select-Object Name, DeviceID | Format-Table -AutoSize']
        },
        {
            'name': 'PnP Camera Devices',
            'cmd': ['powershell', '-Command',
                   'Get-PnpDevice -Class Camera | Select-Object FriendlyName, InstanceId | Format-Table -AutoSize']
        },
        {
            'name': 'DirectShow Devices',
            'cmd': ['powershell', '-Command',
                   'Get-ItemProperty "HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\webcam" | Select-Object -ExpandProperty PSChildName']
        }
    ]
    
    for cmd_info in commands:
        print(f"\n--- {cmd_info['name']} ---")
        try:
            result = subprocess.run(cmd_info['cmd'], capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                print(result.stdout)
            else:
                print(f"  No output or error: {result.stderr}")
        except Exception as e:
            print(f"  Error: {e}")

def test_obs_virtual_camera():
    """Test specifically for OBS Virtual Camera"""
    print("\n=== OBS Virtual Camera Detection ===")
    
    # Check if OBS is running
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq obs64.exe'], 
                              capture_output=True, text=True, timeout=10)
        if 'obs64.exe' in result.stdout:
            print("✓ OBS Studio is running")
        else:
            print("✗ OBS Studio is NOT running - this may explain missing OBS Virtual Camera")
    except Exception as e:
        print(f"  Error checking OBS: {e}")
    
    # Look for OBS Virtual Camera in registry
    try:
        result = subprocess.run([
            'powershell', '-Command',
            'Get-ItemProperty "HKLM:\\SOFTWARE\\OBS Studio" -ErrorAction SilentlyContinue | Select-Object *'
        ], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            print("✓ OBS Studio registry entries found")
        else:
            print("✗ No OBS Studio registry entries found")
    except Exception as e:
        print(f"  Error checking OBS registry: {e}")

def main():
    print("=== COMPREHENSIVE CAMERA DETECTION REPORT ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")
    print()
    
    test_opencv_cameras()
    test_windows_system_cameras()
    test_obs_virtual_camera()
    
    print("\n=== END REPORT ===")
    print("\nTo run this script on Windows:")
    print("1. Open Command Prompt or PowerShell")
    print("2. Navigate to this folder")
    print("3. Run: python inspectcameras.py")

if __name__ == "__main__":
    main()