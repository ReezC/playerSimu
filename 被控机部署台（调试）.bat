@echo off
rem Debug launcher: keeps a console window, so startup errors (including the ones
rem that die silently under pythonw) are printed here. Use the other .bat daily.
rem This file is intentionally ASCII-only (see the note in the other .bat).
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" -m deploy.app
echo.
echo ==== app exited (output / errors above), press any key ====
pause >nul
