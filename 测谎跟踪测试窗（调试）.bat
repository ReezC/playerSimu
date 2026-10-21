@echo off
rem Same as the normal launcher, but KEEPS the console so tracebacks show up.
rem Use this when the window does not appear / closes instantly.
rem
rem ASCII-only on purpose: cmd.exe reads a .bat in the system ANSI code page.
cd /d "%~dp0"
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" -X utf8 -m tools.vt_test_window
echo.
echo [exit code %errorlevel%]
pause
