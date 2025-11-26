@echo off
cd /d "%~dp0"

echo === OBS Virtual Camera Diagnostic Tool ===
echo.

REM Clear flags
set USE_PYTHON_PATH=
set USE_SYSTEM_PYTHON=

REM Detect which Python command works (py or python)
echo Detecting Python installation...
py --version >nul 2>&1
if errorlevel 1 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not available. Please install Python 3.
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=python
        set PYTHON_LAUNCHER=python
    )
) else (
    set PYTHON_CMD=py -3
    set PYTHON_LAUNCHER=py -3
)
echo Python detected: %PYTHON_CMD%

REM Check for existing venv (.venv-win or .venv)
set VENV_DIR=
if exist ".venv-win" (
    set VENV_DIR=.venv-win
) else if exist ".venv" (
    set VENV_DIR=.venv
)

REM Check if venv Python exists and verify it works
if defined VENV_DIR (
    REM Use absolute path for venv Python (handles network drives and spaces)
    set "VENV_PYTHON=%~dp0%VENV_DIR%\Scripts\python.exe"
    if exist "%VENV_PYTHON%" (
        REM Test if venv Python actually works
        "%VENV_PYTHON%" --version >nul 2>&1
        if errorlevel 1 (
            echo WARNING: Virtual environment Python is broken. Using system Python...
            set USE_SYSTEM_PYTHON=1
        ) else (
            echo Using Python from virtual environment: %VENV_PYTHON%
            set "PYTHON_CMD=%VENV_PYTHON%"
            set "PYTHON_LAUNCHER=%VENV_PYTHON%"
            set "USE_PYTHON_PATH=1"
        )
    ) else (
        echo WARNING: Virtual environment Python not found at: %VENV_PYTHON%
        echo Using system Python...
        set USE_SYSTEM_PYTHON=1
    )
) else (
    echo No virtual environment found. Using system Python...
    set USE_SYSTEM_PYTHON=1
)

echo.
echo Running diagnostic script...
echo.

REM Execute Python - quote if it's a path, don't quote if it's a command with arguments
if defined USE_PYTHON_PATH (
    "%PYTHON_CMD%" diagnose_obs_camera_windows.py
) else (
    %PYTHON_CMD% diagnose_obs_camera_windows.py
)

echo.
echo Diagnostic complete. Review the output above.
pause

