@echo off
rem Launch the lie-tracking TEST WINDOW (tools.vt_test_window):
rem   pick a screen rect (or a video) -> live preview with ROI / target box /
rem   aim point -> the "move mouse" checkbox is OFF by default.
rem %~dp0 = this script's directory, so a desktop shortcut still works.
rem
rem ASCII-only on purpose: cmd.exe reads a .bat in the system ANSI code page, so
rem non-ASCII bytes here turn into mojibake and can break the next command.
rem
rem Prefer the project venv (.venv, shared by every workbench in this repo).
cd /d "%~dp0"
set "PYW="
if exist ".venv\Scripts\pythonw.exe" set "PYW=.venv\Scripts\pythonw.exe"
if not defined PYW set "PYW=pythonw"

rem The console twin of PYW (pythonw.exe -> python.exe), used for the check below.
set "PYC=%PYW:pythonw.exe=python.exe%"
if /i "%PYC%"=="pythonw" set "PYC=python"

rem Preflight: this window needs PyQt5 + opencv. Report clearly instead of
rem dying with no window at all.
"%PYC%" -c "import cv2, PyQt5" 2>nul
if errorlevel 1 (
    echo.
    echo Test window cannot start: missing python packages ^(PyQt5 / opencv-python^).
    echo   interpreter : %PYC%
    echo   install deps: "%PYC%" -m pip install PyQt5 opencv-python
    echo.
    pause
    exit /b 1
)

rem pythonw = no console window. If it dies before the window shows, run the
rem debug twin next to this file (it keeps the console and prints the traceback).
start "" "%PYW%" -m tools.vt_test_window
