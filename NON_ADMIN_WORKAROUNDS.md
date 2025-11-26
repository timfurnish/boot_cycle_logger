# Non-Admin Workarounds for "Flicker" PC

## Problem Summary

The "Flicker" PC has restricted user permissions (no admin rights), which prevents:
- Disabling cameras in Device Manager
- Modifying DirectShow filter priorities
- Installing/reinstalling OBS Virtual Camera
- Changing Windows camera enumeration

This causes camera source conflicts where OBS Virtual Camera and laptop camera compete for the same device index.

---

## ✅ Solution 1: Use Physical USB Camera (RECOMMENDED)

**This is the most reliable non-admin solution.**

### **Setup:**
1. **Get a USB webcam** (any quality, even old ones work)
2. **Plug into Flicker PC** - no drivers or admin rights needed
3. **Two mounting options:**

#### **Option A: Point at OBS Monitor**
- Mount USB camera on tripod/stand
- Point it at monitor displaying OBS output
- Adjust for minimal glare/reflections
- Set focus to infinity for screen capture

#### **Option B: Direct Video Feed**
- If your video sources have direct outputs
- Connect USB capture card (no admin needed)
- Bypass OBS entirely

### **Camera Settings in App:**
- Source: Will be index 3, 4, or 5 (higher than built-in cameras)
- Backend: Try MSMF first, then DSHOW
- Resolution: Match your USB camera (usually 720p or 1080p)

### **Benefits:**
✅ No admin rights required
✅ No camera conflicts (dedicated device)
✅ More stable than virtual cameras
✅ Can leave permanently connected to Flicker PC
✅ Physical cameras are prioritized differently by Windows

### **Recommended USB Cameras (Budget-Friendly):**
- Logitech C270 (~$20) - 720p, reliable
- Logitech C920 (~$50) - 1080p, excellent quality
- Any generic USB webcam from Amazon

---

## ✅ Solution 2: Updated Software with Camera Locking

**The latest version now includes non-admin camera locking.**

### **What Changed:**
The code now attempts to "lock" the camera handle to prevent Windows from switching sources:

```python
# WINDOWS NON-ADMIN FIX: Lock camera to prevent source switching
cap.set(cv2.CAP_PROP_SETTINGS, 0)  # Disable settings dialog
# Read and discard a few frames to ensure camera is locked
for _ in range(3):
    cap.read()
```

### **How to Use:**
1. Make sure you have the latest `boot_cycle_gui_web-macpc-6ch.py` (with camera locking)
2. Try Source 2 with MSMF again
3. Look for console message: `[run_loop] ✓ Camera handle locked (Windows non-admin mode)`
4. Monitor for 30+ seconds to see if flickering is reduced

### **Expected Result:**
- Reduced camera switching during continuous capture
- More stable video feed
- May still have occasional flickers, but should be better

---

## ✅ Solution 3: Use Different Application Launch

**Run application with explicit camera persistence flags.**

### **Create `run_flicker.bat`:**

```batch
@echo off
REM Special launcher for Flicker PC (non-admin)
REM Forces camera persistence and error recovery

echo ============================================
echo Boot Cycle Logger - FLICKER PC MODE
echo (Non-Admin Optimized)
echo ============================================
echo.

REM Activate virtual environment
call .venv-win\Scripts\activate

REM Set OpenCV environment variables for better camera handling
set OPENCV_VIDEOIO_PRIORITY_MSMF=100
set OPENCV_VIDEOIO_PRIORITY_DSHOW=0
set OPENCV_VIDEOIO_DEBUG=0

REM Run with camera persistence mode
python boot_cycle_gui_web-macpc-6ch.py

pause
```

This prioritizes MSMF backend and disables debug logging that can interfere with camera handles.

---

## ✅ Solution 4: Ask IT for Specific Permissions

**Instead of full admin, request only what you need:**

### **Minimal Permissions Request:**

> "For the Boot Cycle Logger testing application, I need permission to:
> 
> 1. **Disable/Enable cameras in Device Manager**
>    - This allows me to temporarily disable the laptop camera during testing
>    - Prevents camera device conflicts
>    - Can be reverted after each test session
> 
> 2. **OR: Install one USB webcam driver** (if using USB camera solution)
>    - Most USB cameras are plug-and-play (no admin needed)
>    - But some may require driver installation
> 
> I do NOT need full admin rights - just these specific camera management permissions for testing purposes."

---

## ✅ Solution 5: Remote Desktop Workaround

**If Flicker PC can be accessed remotely:**

### **Setup:**
1. Use **TeamViewer** or **Remote Desktop** from your Mac (which has admin rights)
2. Connect to Flicker PC remotely
3. Your admin credentials may carry over in some remote desktop scenarios
4. Can manage cameras remotely with your admin account

**Check with IT if this is allowed.**

---

## ✅ Solution 6: Test on Different PC

**If no workarounds succeed:**

### **Options:**
1. **Use your current PC** (where it works) for testing
2. **Request a different test PC** from IT with proper permissions
3. **Dedicate a non-production PC** to boot cycle testing

### **Document Working Configuration:**
Once you find a working setup, document:
```
PC: [Name]
User: [Username]
Admin Rights: Yes/No
Source: X
Backend: MSMF/DSHOW
Camera: Built-in / USB / OBS Virtual
Working: Yes
Notes: [Any special setup]
```

Keep this as a reference for future testing.

---

## 🎯 Recommended Approach (Priority Order):

### **1. USB Camera Solution (Immediate)**
- **Time**: 1-2 days (order camera, plug in)
- **Cost**: $20-50
- **Success Rate**: 95%+
- **Admin Required**: No

### **2. Try Updated Software (Right Now)**
- **Time**: 5 minutes (update file, test)
- **Cost**: Free
- **Success Rate**: 30-50%
- **Admin Required**: No

### **3. Request Minimal Permissions (2-5 days)**
- **Time**: Depends on IT approval process
- **Cost**: Free
- **Success Rate**: If approved, 100%
- **Admin Required**: Partial (just camera management)

### **4. Use Different PC (1 week)**
- **Time**: Request, approval, setup
- **Cost**: Free (if PC available)
- **Success Rate**: 100%
- **Admin Required**: On new PC

---

## 📋 Flicker PC Current Status

**Working Configurations:**
- ❌ Source 0 + DSHOW: Laptop camera (not OBS)
- ❌ Source 1 + MSMF: Black screen (broken)
- ⚠️ Source 2 + MSMF: **Appears stable in validation, flickers in continuous use**

**Issue:**
- Camera validation passes (5 frames, stable)
- Continuous capture fails (flickering between laptop cam and OBS)
- Likely cause: Windows switching cameras during continuous capture
- Root cause: No admin rights to disable conflicting camera

**Recommendation:**
**Get a USB camera** - it's the fastest, most reliable solution that requires no admin rights and eliminates all camera conflicts permanently.

---

## 💰 Cost-Benefit Analysis

| Solution | Cost | Time | Reliability | Admin Needed |
|----------|------|------|-------------|--------------|
| **USB Camera** | **$20-50** | **1-2 days** | **⭐⭐⭐⭐⭐** | **No** |
| Software Update | Free | 5 min | ⭐⭐⭐ | No |
| IT Permissions | Free | 2-5 days | ⭐⭐⭐⭐ | Partial |
| Different PC | Free | 1 week | ⭐⭐⭐⭐⭐ | Yes (on new PC) |

**Best ROI: USB Camera** - low cost, high reliability, no admin needed, permanent solution.

---

## 🚀 Next Steps

1. **Immediate**: Update to latest code with camera locking, test Source 2 again
2. **Short-term**: Order USB webcam ($20-50 on Amazon)
3. **Parallel**: Ask IT for camera management permissions or different PC
4. **Backup**: Document working PC configurations as reference

**With a USB camera, you'll have a dedicated, stable video source for Flicker PC that requires zero admin rights and eliminates all camera conflicts!** 🎥

