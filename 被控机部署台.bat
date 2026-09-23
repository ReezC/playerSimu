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

rem pythonw = no console window. If it fails before the window shows, the app
rem pops an error box and writes the traceback to deploy_crash.log (deploy/app.py).
start "" "%PYW%" -m deploy.app

rem start fails when the interpreter is missing; say so instead of exiting silent.
if errorlevel 1 (
    echo.
    echo Deploy console failed to start: "%PYW%" not found.
    echo   install deps : pip install -r deploy\requirements.txt
    echo   other python : edit PYW in this file to the full path of pythonw.exe
    echo.
    pause
)
