@echo off
rem Console launcher for the lie-detection demo: shows the full traceback on a
rem startup crash. For daily use run the other .bat (the one without the suffix).
rem
rem Keep this file ASCII-only: cmd.exe reads a .bat in the system ANSI code page,
rem so non-ASCII bytes here turn into mojibake and can break the next command.
rem Prefer the project venv (.venv, shared by every workbench in this repo).
cd /d "%~dp0"
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY set "PY=python"

"%PY%" -m tools.lie_demo
echo.
echo ==== program exited, press any key ====
pause >nul
