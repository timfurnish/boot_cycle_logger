@echo off
REM Diagnostic script for "Flicker" PC camera detection issues
REM Double-click this file to run comprehensive camera diagnostics

echo ============================================
echo Boot Cycle Logger - Flicker PC Diagnostic
echo ============================================
echo.
echo This will test OBS Virtual Camera detection...
echo.

REM Check if virtual environment exists
if not exist ".venv-win" (
    echo Creating virtual environment...
    py -3 -m venv .venv-win
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python 3 is installed
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call .venv-win\Scripts\activate

REM Install/upgrade dependencies
echo Installing dependencies...
python -m pip install --upgrade pip --quiet
pip install opencv-python --quiet

REM Run the diagnostic
echo.
echo Running diagnostic...
echo.
python diagnose_flicker_pc.py

REM Keep window open
pause

