#!/usr/bin/env bash
cd "$(dirname "$0")"
exec /bin/bash "./run_mac.sh"

# Windows Instructions:
# To create a similar script for Windows, create a file named 'run_windows.cmd' with the following content:
#
# @echo off
# cd /d "%~dp0"
# python boot_cycle_gui_web-macpc.py
#
# Note: The restart button in the web app will only work if the Python script is restarted externally.
# This means you need to manually restart the script or set up an external mechanism to restart it.