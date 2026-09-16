@echo off
setlocal
cd /d "%~dp0"

if not exist ".mplconfig" mkdir ".mplconfig"
set "MPLCONFIGDIR=%CD%\.mplconfig"
if not defined FLASK_HOST set "FLASK_HOST=0.0.0.0"
if not defined PORT if not defined FLASK_PORT set "FLASK_PORT=5000"
set "FLASK_DEBUG=0"

python app.py
