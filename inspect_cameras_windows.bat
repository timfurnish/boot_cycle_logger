@echo off
echo === Camera Inspection Tool - Windows Edition ===
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

echo Running camera detection report...
echo.

REM Run the camera inspection script
python inspectcameras.py

echo.
echo Report complete!
echo.
pause
