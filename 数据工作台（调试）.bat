@echo off
rem Console launcher: shows the full traceback on startup crashes.
rem For daily use run the other launcher (the one without the debug suffix).
rem Keep this file ASCII-only: cmd.exe reads a .bat in the system ANSI code page,
rem so non-ASCII bytes here turn into mojibake and can break the next command.
cd /d "%~dp0"
python -m gui.app
echo.
echo ==== program exited, press any key ====
pause >nul
