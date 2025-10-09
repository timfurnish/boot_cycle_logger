@echo off
cd /d "%~dp0"
call "%~dp0venv\Scripts\activate"
python boot_cycle_gui_web-macpc.py
pause