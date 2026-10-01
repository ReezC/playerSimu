@echo off

rem Launch the lightweight YOLO workbench (label -> train -> val, dataset-agnostic).
rem %~dp0 = this script's directory, so a desktop shortcut still works.
rem Prefer the project venv (.venv, shared by every workbench in this repo);
rem fall back to the system pythonw when the venv does not exist yet.
cd /d "%~dp0"
set "PYW="
if exist ".venv\Scripts\pythonw.exe" set "PYW=.venv\Scripts\pythonw.exe"
if not defined PYW set "PYW=pythonw"
start "" "%PYW%" -m gui.yolo_workbench
