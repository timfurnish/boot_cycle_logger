@echo off
cd /d "%~dp0"

echo === Boot Cycle Logger - Windows Launcher ===

REM Kill any existing Boot Cycle Logger processes
echo [0/6] Cleaning up existing processes...
taskkill /f /im BootCycleLogger.exe >nul 2>&1
taskkill /f /im python.exe /fi "WINDOWTITLE eq Boot Cycle Logger*" >nul 2>&1
REM Also clean up any stuck camera processes that might interfere with OBS Virtual Camera
taskkill /f /im python.exe /fi "COMMANDLINE eq *boot_cycle_gui_web-macpc-6ch.py*" >nul 2>&1
echo Existing processes cleaned up.

REM Create venv if missing (fail gracefully if no admin access)
if not exist ".venv-win" (
    echo [1/6] Creating virtual environment...
    py -3 -m venv .venv-win
    if errorlevel 1 (
        echo WARNING: Failed to create virtual environment. Trying to continue with system Python...
        echo If this fails, you may need admin access or to install Python 3.
        set USE_SYSTEM_PYTHON=1
    ) else (
        set USE_SYSTEM_PYTHON=0
    )
) else (
    set USE_SYSTEM_PYTHON=0
)

REM Activate venv if it exists, otherwise use system Python
if "%USE_SYSTEM_PYTHON%"=="0" (
    call .venv-win\Scripts\activate.bat
    if errorlevel 1 (
        echo WARNING: Failed to activate virtual environment. Using system Python...
        set USE_SYSTEM_PYTHON=1
    )
)

REM Install/refresh deps (cached after first run) - fail gracefully
echo [2/6] Upgrading pip...
python -m pip install --upgrade pip --user >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip. Continuing anyway...
    python -m pip install --upgrade pip >nul 2>&1
    if errorlevel 1 (
        echo WARNING: pip upgrade failed. Continuing with existing pip version...
    )
)

echo [3/6] Installing dependencies...
REM Try with --user flag first (no admin required)
python -m pip install --user flask opencv-python pillow imagehash numpy >nul 2>&1
if errorlevel 1 (
    echo WARNING: Failed to install dependencies with --user flag. Trying without...
    python -m pip install flask opencv-python pillow imagehash numpy >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Failed to install dependencies. Assuming they may already be available...
        echo Continuing - Python will report any missing imports when the app starts.
    ) else (
        echo Dependencies installed successfully.
    )
) else (
    echo Dependencies installed successfully.
)

REM Verify Python is working (skip strict dependency check - let Python report errors)
echo [4/6] Verifying Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not available or not in PATH.
    pause
    exit /b 1
)
echo Python environment ready.

REM Run the app
echo [5/6] Starting Boot Cycle Logger on http://localhost:5055 ...
echo.
echo The application will open in your browser automatically.
echo Close this window to stop the application.
echo.
python boot_cycle_gui_web-macpc-6ch.py

echo.
echo Application finished.
pause