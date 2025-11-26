@echo off
REM Boot Cycle Logger - Windows Launcher
REM Double-click this file to run the Boot Cycle Logger

echo ===============================================
echo Boot Cycle Logger - Starting...
echo ===============================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv-win\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    py -3 -m venv .venv-win
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Please ensure Python 3 is installed
        pause
        exit /b 1
    )
)

REM Activate virtual environment and install dependencies
echo [2/3] Installing dependencies...
call .venv-win\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install flask opencv-python pillow imagehash numpy >nul 2>&1

REM Run the application
echo [3/3] Launching Boot Cycle Logger...
echo.
echo The application will open in your browser automatically.
echo To stop the server, close this window or press Ctrl+C
echo.
echo ===============================================

python boot_cycle_gui_web-macpc-6ch.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo ===============================================
    echo ERROR: Application exited with an error
    echo ===============================================
    pause
)

