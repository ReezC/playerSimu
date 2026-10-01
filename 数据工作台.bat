@echo off
rem Launch the dataset workbench GUI.
rem %~dp0 = this script's directory, so a desktop shortcut still works.
rem pythonw = no console window. For tracebacks, use the debug .bat.
rem Prefer the project venv (.venv, shared by every workbench in this repo).
cd /d "%~dp0"
set "PYW="
if exist ".venv\Scripts\pythonw.exe" set "PYW=.venv\Scripts\pythonw.exe"
if not defined PYW set "PYW=pythonw"
start "" "%PYW%" -m gui.app
