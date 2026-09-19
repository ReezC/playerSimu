@echo off
rem Console launcher: shows full traceback on startup crashes.
rem Use 数据工作台.bat for daily use.
cd /d "%~dp0"
python -m gui.app
echo.
echo ==== program exited, press any key ====
pause >nul
