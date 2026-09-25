@echo off
rem Launch the A-machine deploy console (clock / probe / keyboard relay / stream).
rem
rem This file is intentionally ASCII-only. cmd.exe reads a .bat in the system
rem ANSI code page, so non-ASCII bytes here turn into mojibake on screen and can
rem even swallow the next character (breaking the command). Chinese messages go
rem in the app (deploy/app.py) instead.
rem %~dp0 = this script's directory, so a desktop shortcut still works.
cd /d "%~dp0"

rem Prefer a project venv: on the A machine the deps may live in .venv while the
rem system python has no PyQt5 -- a double-clicked .bat does not activate a venv.
set "PYW="
if exist ".venv\Scripts\pythonw.exe" set "PYW=.venv\Scripts\pythonw.exe"
if not defined PYW if exist "venv\Scripts\pythonw.exe" set "PYW=venv\Scripts\pythonw.exe"
if not defined PYW set "PYW=pythonw"

rem The console twin of PYW (pythonw.exe -> python.exe), used for the check below.
set "PYC=%PYW:pythonw.exe=python.exe%"
if /i "%PYC%"=="pythonw" set "PYC=python"

rem Preflight 1: every module the A machine actually runs, checked by
rem deploy/deps.py -- one list shared with the install script, the console's
rem self-check and the selftests. "Installed" then means installed, not
rem "the three names someone remembered to type here".
"%PYC%" -m deploy.deps
if errorlevel 1 (
    echo.
    echo Deploy console cannot start: missing python packages ^(listed above^).
    echo   interpreter : %PYC%
    echo   install deps: "%PYC%" -m pip install -r deploy\requirements.txt
    echo   or run the install .bat next to this file ^(creates a .venv^).
    echo   tkinter is not pip-installable: re-run the Python installer and
    echo   tick tcl/tk.
    echo.
    echo If that interpreter is not found either, install Python 3.10 first.
    echo.
    pause
    exit /b 1
)

rem Preflight 2: config/code drift vs the deployment manifest (tools/config_sync.py).
rem Quiet on purpose: this used to pop a message box, but the console itself shows
rem the same row in its env self-check (top right), which is where the fix lives
rem too (e.g. the "generate certificate" button). A popup here just interrupts.
rem Kept as a check only so there is a dated record in deploy_sync.log to look at
rem when something behaves oddly ("looks like it runs yesterday's files").
"%PYC%" -m tools.config_sync --preflight >> deploy_sync.log 2>&1

rem pythonw = no console window. If it fails before the window shows, the app
rem pops an error box and writes the traceback to deploy_crash.log (deploy/app.py).
start "" "%PYW%" -m deploy.app
