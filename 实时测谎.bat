@echo off
rem Launch the LIVE lie-detection tester (its own window, independent of the
rem main workbench):
rem   pick a SOURCE (a local screen rect, or a stream) -> box a region -> Start.
rem   It grabs live and the tracking runs inside that region only.
rem %~dp0 = this script's directory, so a desktop shortcut still works.
rem Extra args pass through (%*), e.g. --stream udp://0.0.0.0:5000 --rect x,y,w,h
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

rem Preflight: needs PyQt5 (window) + opencv + numpy. Report clearly instead of
rem failing with no window at all.
"%PYC%" -c "import cv2, numpy, PyQt5" 2>nul
if errorlevel 1 (
    echo.
    echo Live tester cannot start: missing python packages ^(PyQt5 / opencv-python / numpy^).
    echo   interpreter : %PYC%
    echo   install deps: "%PYC%" -m pip install PyQt5 opencv-python numpy
    echo   or run the install .bat next to this file ^(creates a .venv^).
    echo.
    pause
    exit /b 1
)

rem pythonw = no console window. If it dies before the window shows, run the
rem debug twin next to this file (it keeps the console and prints the traceback).
start "" "%PYW%" -m tools.live_lie %*
