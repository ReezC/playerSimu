@echo off
rem Launch the dataset workbench GUI.
rem %~dp0 = this script's directory, so a desktop shortcut still works.
rem pythonw = no console window. For tracebacks, use the debug .bat.
cd /d "%~dp0"
start "" pythonw -m gui.app
