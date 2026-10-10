@echo off
cd /d "%~dp0"
set "PYTHONPATH="
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m liescd.gui %*
) else (
  python -m liescd.gui %*
)
if errorlevel 1 pause
