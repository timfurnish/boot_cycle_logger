@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
REM Create log file to capture crashes
set "LOG_FILE=%~dp0run_windows_log.txt"
echo === Boot Cycle Logger - Windows Launcher ===
echo Starting at %date% %time% > "%LOG_FILE%"
echo Working directory: %CD% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo === Boot Cycle Logger - Windows Launcher ===

REM Clear flags
set USE_PYTHON_PATH=
set USE_SYSTEM_PYTHON=

REM Detect which Python command works (py or python)
echo [0/7] Detecting Python installation...
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

REM Kill any existing Boot Cycle Logger processes
echo [1/7] Cleaning up existing processes...
taskkill /f /im BootCycleLogger.exe >nul 2>&1
taskkill /f /im python.exe /fi "WINDOWTITLE eq Boot Cycle Logger*" >nul 2>&1
REM Also clean up any stuck camera processes that might interfere with OBS Virtual Camera
taskkill /f /im python.exe /fi "COMMANDLINE eq *boot_cycle_gui_web-macpc-6ch.py*" >nul 2>&1
echo Existing processes cleaned up.

REM Check for existing venv - use persistent config file for reliability
echo [2/7] Checking for virtual environment...

REM Detect computer name FIRST (needed for machine-specific config file)
set COMPUTER_NAME=
REM Method 1: Use %COMPUTERNAME% environment variable (most reliable)
if defined COMPUTERNAME (
    set COMPUTER_NAME=%COMPUTERNAME%
    set COMPUTER_NAME=!COMPUTERNAME!
) else (
    REM Method 2: Try wmic (may fail on some systems)
    for /f "tokens=2 delims==" %%I in ('wmic computersystem get name /value 2^>nul') do (
        if not defined COMPUTER_NAME set COMPUTER_NAME=%%I
    )
    if not defined COMPUTER_NAME (
        REM Method 3: Try hostname command
        for /f %%I in ('hostname 2^>nul') do set COMPUTER_NAME=%%I
        if not defined COMPUTER_NAME (
            REM Fallback: use a generic name
            set COMPUTER_NAME=WIN
        )
    )
)
REM Clean computer name (remove spaces, special chars)
if defined COMPUTER_NAME (
    set "COMPUTER_NAME=!COMPUTER_NAME: =!"
    set "COMPUTER_NAME=!COMPUTER_NAME:"=!"
) else (
    set COMPUTER_NAME=WIN
)
echo   Detected computer: !COMPUTER_NAME!

REM Use MACHINE-SPECIFIC config file so each PC has its own venv
set VENV_CONFIG_FILE=.venv_config_!COMPUTER_NAME!.txt
set VENV_DIR=
set TARGET_VENV=

REM First, try to read venv name from machine-specific config file (most reliable - persists across runs)
if exist "%VENV_CONFIG_FILE%" (
    echo   Reading venv name from persistent config file: %VENV_CONFIG_FILE%
    set "CONFIG_VENV="
    for /f "usebackq eol=# delims=" %%L in ("%VENV_CONFIG_FILE%") do (
        if not defined CONFIG_VENV (
            set "CONFIG_VENV=%%L"
            REM Trim whitespace from both ends
            for /f "tokens=*" %%A in ("!CONFIG_VENV!") do set "CONFIG_VENV=%%A"
            REM Remove quotes if present
            if defined CONFIG_VENV (
                REM Remove quotes and trim whitespace
                set "CONFIG_VENV=!CONFIG_VENV:"=!"
                REM Trim leading/trailing spaces
                for /f "tokens=*" %%A in ("!CONFIG_VENV!") do set "CONFIG_VENV=%%A"
                if "!CONFIG_VENV!"=="" (
                    set "CONFIG_VENV="
                ) else (
                    REM Trim any trailing spaces from the venv dir name
                    set "CONFIG_VENV=!CONFIG_VENV: =!"
                    set "CONFIG_VENV=!CONFIG_VENV: =!"
                    if exist "!CONFIG_VENV!" (
                        set "VENV_DIR=!CONFIG_VENV!"
                        echo   ✓ Found venv from config file: !VENV_DIR!
                    ) else (
                        echo   ✗ Config file venv not found: !CONFIG_VENV!, will search for alternatives
                    )
                )
            )
        )
    )
    if not defined CONFIG_VENV (
        echo   Config file exists but is empty or invalid, will search for alternatives
    )
)

REM If config file didn't work, try machine-specific venv using computer name
if not defined VENV_DIR (
    REM Check for machine-specific venv
    set TARGET_VENV=.venv-!COMPUTER_NAME!
    echo   Checking for machine-specific venv: !TARGET_VENV!
    if exist "!TARGET_VENV!" (
        set VENV_DIR=!TARGET_VENV!
        echo   ✓ Found machine-specific venv: !VENV_DIR!
        REM Save to config file for next time (persistent)
        echo !VENV_DIR! > "!VENV_CONFIG_FILE!"
        echo   Saved venv name to config file for future use
    ) else (
        echo   ✗ Machine-specific venv not found, checking fallback options...
        if exist ".venv-win" (
            set "VENV_DIR=.venv-win"
            for /f "tokens=*" %%E in ("!VENV_DIR!") do set "VENV_DIR=%%E"
            echo   ✓ Found fallback venv: .venv-win
            REM Save to config file
            echo !VENV_DIR! > "!VENV_CONFIG_FILE!"
        ) else if exist ".venv" (
            set "VENV_DIR=.venv"
            for /f "tokens=*" %%F in ("!VENV_DIR!") do set "VENV_DIR=%%F"
            echo   ✓ Found fallback venv: .venv
            REM Save to config file
            echo !VENV_DIR! > "!VENV_CONFIG_FILE!"
        ) else (
            echo   ✗ No existing venv found
            REM Will create new one below
            set TARGET_VENV=.venv-!COMPUTER_NAME!
        )
    )
)

REM Create venv if missing (fail gracefully if no admin access)
if not defined VENV_DIR (
    if not defined TARGET_VENV (
        if not defined COMPUTER_NAME set COMPUTER_NAME=WIN
        set TARGET_VENV=.venv-!COMPUTER_NAME!
    )
    echo   Creating new machine-specific virtual environment: !TARGET_VENV!
    %PYTHON_CMD% -m venv "!TARGET_VENV!"
    if errorlevel 1 (
        echo   ✗ WARNING: Failed to create virtual environment: !TARGET_VENV!
        echo   Trying to continue with system Python...
        echo   If this fails, you may need admin access or to install Python 3.
        set USE_SYSTEM_PYTHON=1
    ) else (
        set "VENV_DIR=!TARGET_VENV!"
        REM Trim any trailing spaces
        for /f "tokens=*" %%G in ("!VENV_DIR!") do set "VENV_DIR=%%G"
        set USE_SYSTEM_PYTHON=0
        echo   ✓ Successfully created virtual environment: !VENV_DIR!
        REM Save to config file for future use
        echo !VENV_DIR! > "!VENV_CONFIG_FILE!"
        echo   Saved venv name to config file: !VENV_CONFIG_FILE!
    )
) else (
    set USE_SYSTEM_PYTHON=0
    echo   Using existing virtual environment: !VENV_DIR!
)

REM Check if venv Python exists and verify it works
if "%USE_SYSTEM_PYTHON%"=="0" (
    REM Use absolute path for venv Python (handles network drives and spaces)
    REM Build path: script directory + venv dir + Scripts\python.exe
    set "SCRIPT_DIR=%~dp0"
    REM Remove trailing backslash from script dir if present
    if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"
    REM Trim any leading/trailing spaces from VENV_DIR before building path
    for /f "tokens=*" %%H in ("!VENV_DIR!") do set "VENV_DIR=%%H"
    REM Remove any remaining spaces (shouldn't be any, but just in case)
    set "VENV_DIR=!VENV_DIR: =!"
    set "VENV_PYTHON=!SCRIPT_DIR!\!VENV_DIR!\Scripts\python.exe"
    REM Remove any double backslashes (keep quotes to preserve spaces in path)
    set "VENV_PYTHON=!VENV_PYTHON:\\=\!"
    echo   Verifying venv Python at: !VENV_PYTHON!
    echo   Script directory: !SCRIPT_DIR!
    echo   Venv directory: !VENV_DIR!
    echo   Full path: !VENV_PYTHON!
    if exist "!VENV_PYTHON!" (
        REM Test if venv Python actually works
        "!VENV_PYTHON!" --version >nul 2>&1
        if errorlevel 1 (
            echo   ✗ WARNING: Virtual environment Python is broken at: !VENV_PYTHON!
            echo   Recreating virtual environment: !VENV_DIR!
            rmdir /s /q "!VENV_DIR!" 2>nul
            %PYTHON_CMD% -m venv "!VENV_DIR!"
            if errorlevel 1 (
                echo   ✗ WARNING: Failed to recreate virtual environment. Using system Python...
                set USE_SYSTEM_PYTHON=1
            ) else (
                REM Rebuild path after recreation
                set "SCRIPT_DIR=%~dp0"
                if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"
                REM Trim spaces from VENV_DIR (name only, not full path)
                set "VENV_DIR=!VENV_DIR: =!"
                set "VENV_PYTHON=!SCRIPT_DIR!\!VENV_DIR!\Scripts\python.exe"
                REM Remove any double backslashes (keep quotes to preserve spaces in full path)
                set "VENV_PYTHON=!VENV_PYTHON:\\=\!"
                set "PYTHON_CMD=!VENV_PYTHON!"
                set "PYTHON_LAUNCHER=!VENV_PYTHON!"
                set "USE_PYTHON_PATH=1"
                echo   ✓ Recreated virtual environment successfully
                echo   Using Python from virtual environment: !VENV_DIR!
                REM Update config file
                echo !VENV_DIR! > "!VENV_CONFIG_FILE!"
            )
        ) else (
            echo   ✓ Virtual environment Python verified and working
            echo   Using Python from virtual environment: !VENV_DIR!
            set "PYTHON_CMD=!VENV_PYTHON!"
            set "PYTHON_LAUNCHER=!VENV_PYTHON!"
            set "USE_PYTHON_PATH=1"
            echo   Set PYTHON_CMD to venv Python: !PYTHON_CMD!
        )
    ) else (
        echo   ✗ WARNING: Virtual environment Python not found at: !VENV_PYTHON!
        echo   Virtual environment directory: !VENV_DIR!
        echo   Checking if directory exists...
        if exist "!VENV_DIR!" (
            echo   Directory exists, checking contents...
            dir "!VENV_DIR!\Scripts\" 2>nul | findstr /i "python.exe" >nul
            if errorlevel 1 (
                echo   Python.exe not found in Scripts folder - venv may be corrupted
            )
        ) else (
            echo   Directory does not exist
        )
        echo   Attempting to recreate virtual environment...
        rmdir /s /q "!VENV_DIR!" 2>nul
        %PYTHON_CMD% -m venv "!VENV_DIR!"
        if errorlevel 1 (
            echo   ✗ WARNING: Failed to recreate virtual environment. Using system Python...
            set USE_SYSTEM_PYTHON=1
        ) else (
            REM Rebuild the path after recreation
            set "SCRIPT_DIR=%~dp0"
            if "!SCRIPT_DIR:~-1!"=="\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"
            set "VENV_PYTHON=!SCRIPT_DIR!\!VENV_DIR!\Scripts\python.exe"
            REM Remove any double backslashes (keep quotes to preserve spaces in path)
            set "VENV_PYTHON=!VENV_PYTHON:\\=\!"
            echo   Checking recreated venv Python at: !VENV_PYTHON!
            if exist "!VENV_PYTHON!" (
                REM Test if it works
                "!VENV_PYTHON!" --version >nul 2>&1
                if errorlevel 1 (
                    echo   ✗ WARNING: Recreated venv Python exists but doesn't work. Using system Python...
                    set USE_SYSTEM_PYTHON=1
                ) else (
                    set "PYTHON_CMD=!VENV_PYTHON!"
                    set "PYTHON_LAUNCHER=!VENV_PYTHON!"
                    set "USE_PYTHON_PATH=1"
                    echo   Set PYTHON_CMD to venv Python: !PYTHON_CMD!
                    echo   ✓ Recreated virtual environment successfully and verified
                )
            ) else (
                echo   ✗ WARNING: Recreated venv but Python still not found at: %VENV_PYTHON%
                echo   Using system Python...
                set USE_SYSTEM_PYTHON=1
            )
        )
    )
) else (
    echo   Using system Python (no virtual environment)
)

REM Install/refresh deps (cached after first run) - fail gracefully
echo [3/7] Checking pip...
REM Skip pip upgrade to avoid crashes - pip is usually fine as-is
REM Just verify pip is available
if defined USE_PYTHON_PATH (
    REM When using venv, check if pip is available
    "%PYTHON_CMD%" -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo WARNING: pip not available in virtual environment. Attempting to install...
        "%PYTHON_CMD%" -m ensurepip --default-pip >nul 2>&1
        if errorlevel 1 (
            echo WARNING: Could not install pip. Continuing anyway - dependencies may fail to install.
        ) else (
            echo   pip installed successfully
        )
    ) else (
        echo   pip is available in virtual environment
    )
) else (
    REM System Python: check if pip is available
    %PYTHON_CMD% -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo WARNING: pip not available. Attempting to install...
        %PYTHON_CMD% -m ensurepip --default-pip >nul 2>&1
        if errorlevel 1 (
            echo WARNING: Could not install pip. Continuing anyway - dependencies may fail to install.
        ) else (
            echo   pip installed successfully
        )
    ) else (
        echo   pip is available
    )
)

echo [4/7] Verifying dependencies...
REM Use marker file to skip verification if packages already verified in this venv
set "PACKAGES_MARKER=!VENV_DIR!\.packages_verified"
if exist "!PACKAGES_MARKER!" (
    echo   ✓ Packages previously verified - skipping check (delete !PACKAGES_MARKER! to force reinstall)
    goto skip_dependency_check
)

REM Simple dependency check - if anything missing, install from requirements.txt
if defined USE_PYTHON_PATH (
    echo   Verifying critical packages...
    "%PYTHON_CMD%" -c "import flask, cv2, openpyxl, PIL, numpy, imagehash, pygrabber, PyWavelets" >nul 2>&1
    if errorlevel 1 (
        echo   Some packages are missing. Installing from requirements.txt...
        "%PYTHON_CMD%" -m pip install --disable-pip-version-check -r requirements.txt
        if errorlevel 1 (
            echo   WARNING: Some packages may have failed to install.
        ) else (
            REM Create marker file to skip check next time
            echo Packages verified on %date% %time% > "!PACKAGES_MARKER!"
        )
        echo   ✓ Dependencies installed.
    ) else (
        echo   ✓ All critical packages are installed.
        REM Create marker file to skip check next time
        echo Packages verified on %date% %time% > "!PACKAGES_MARKER!"
    )
) else (
    echo   Verifying critical packages...
    %PYTHON_CMD% -c "import flask, cv2, openpyxl, PIL, numpy, imagehash, pygrabber, PyWavelets" >nul 2>&1
    if errorlevel 1 (
        echo   Some packages are missing. Installing from requirements.txt...
        %PYTHON_CMD% -m pip install --disable-pip-version-check --user -r requirements.txt
        if errorlevel 1 (
            echo   WARNING: Some packages may have failed to install.
        ) else (
            REM Create marker file to skip check next time
            echo Packages verified on %date% %time% > "!PACKAGES_MARKER!"
        )
        echo   ✓ Dependencies installed.
    ) else (
        echo   ✓ All critical packages are installed.
        REM Create marker file to skip check next time
        echo Packages verified on %date% %time% > "!PACKAGES_MARKER!"
    )
)

:skip_dependency_check

REM No temp file cleanup needed - we use requirements.txt directly now

REM Verify Python is working (skip strict dependency check - let Python report errors)
echo [5/7] Verifying Python environment...
if defined USE_PYTHON_PATH (
    "%PYTHON_CMD%" --version >nul 2>&1
) else (
    %PYTHON_CMD% --version >nul 2>&1
)
if errorlevel 1 (
    echo WARNING: Python version check failed, but continuing anyway...
) else (
    echo Python environment ready.
)

REM Run the app with crash logging
echo [6/7] Starting Boot Cycle Logger on http://localhost:5055 ...
echo.
echo The application will open in your browser automatically.
echo Close this window to stop the application.
echo.
echo Launcher log file: %LOG_FILE%
echo.
echo Starting application at %date% %time% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
echo === Python Command Used === >> "%LOG_FILE%"
if defined USE_PYTHON_PATH (
    echo "%PYTHON_CMD%" boot_cycle_gui_web-macpc-6ch.py >> "%LOG_FILE%"
) else (
    echo %PYTHON_CMD% boot_cycle_gui_web-macpc-6ch.py >> "%LOG_FILE%"
)
echo. >> "%LOG_FILE%"
echo === Starting Application === >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

REM Use quotes only if it's a file path, not a command with arguments
REM Note: Application has its own internal logging system that saves verbose logs
REM This launcher log captures only launcher-specific issues
if defined USE_PYTHON_PATH (
    "%PYTHON_CMD%" boot_cycle_gui_web-macpc-6ch.py
) else (
    %PYTHON_CMD% boot_cycle_gui_web-macpc-6ch.py
)

set APP_EXIT_CODE=%ERRORLEVEL%
echo. >> "%LOG_FILE%"
echo Application exited with code: %APP_EXIT_CODE% at %date% %time% >> "%LOG_FILE%"

echo.
if %APP_EXIT_CODE% neq 0 (
    echo ================================================================
    echo ERROR: Application crashed or exited with error code %APP_EXIT_CODE%
    echo ================================================================
    echo.
    echo Launcher log file: %LOG_FILE%
    echo Application verbose logs are saved in the test folder when you end a test.
    echo.
    echo Please review the log files for error details.
    echo.
) else (
    echo Application finished successfully.
    echo.
    echo Launcher log: %LOG_FILE%
)
pause