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

rem Preflight. pythonw has no console window, so a missing dependency would
rem otherwise look like "double-click does nothing" -- say it out loud instead.
"%PYC%" -c "import PyQt5, yaml, serial" 2>nul
if errorlevel 1 (
    echo.
    echo Deploy console cannot start: missing python packages.
    echo   interpreter : %PYC%
    echo   install deps: "%PYC%" -m pip install -r deploy\requirements.txt
    echo   manual      : pip install PyQt5 pyserial PyYAML
    echo.
    echo If that interpreter is not found either, install Python 3.10 first.
    echo.
    pause
    exit /b 1
)

rem pythonw = no console window. If it fails before the window shows, the app
rem pops an error box and writes the traceback to deploy_crash.log (deploy/app.py).
start "" "%PYW%" -m deploy.app
