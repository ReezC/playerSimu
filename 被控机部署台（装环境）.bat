@echo off
rem Install the A-machine deploy console environment: venv + PyQt5/pyserial/PyYAML.
rem
rem ASCII-only on purpose. cmd.exe reads .bat with the system ANSI codepage, so
rem non-ASCII comments get mis-decoded and can turn into stray commands (this
rem file's sibling was once full of Chinese comments and broke exactly that way).
rem
rem Usage:  double-click it, or pass a venv directory as the first argument.
setlocal
cd /d "%~dp0"

set "VENV=%~1"
if "%VENV%"=="" set "VENV=.venv"

echo ==== playerSimu deploy console: install environment ====
echo      repo : %CD%
echo      venv : %VENV%
echo.

rem ---- 1. find a Python 3 launcher -------------------------------------------
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo [X] Python not found.
    echo     Install Python 3.10+ first, either:
    echo         winget install -e --id Python.Python.3.10
    echo     or from https://www.python.org/downloads/
    echo     ^(tick "Add python.exe to PATH" during setup^)
    echo.
    pause
    exit /b 1
)
echo [1/4] launcher: %PY%
%PY% -c "import sys; print('      python', sys.version.split()[0])"
if errorlevel 1 goto fail

rem ---- 2. venv (reuse if present) -------------------------------------------
if exist "%VENV%\Scripts\python.exe" (
    echo [2/4] venv already exists - reusing it.
) else (
    echo [2/4] creating venv ...
    %PY% -m venv "%VENV%"
    if errorlevel 1 goto fail
)
set "VPY=%VENV%\Scripts\python.exe"

rem ---- 3. deps: PyQt5 + pyserial + PyYAML ------------------------------------
echo [3/4] upgrading pip ...
"%VPY%" -m pip install -U pip
echo [3/4] installing deploy\requirements.txt ...
"%VPY%" -m pip install -r "deploy\requirements.txt"
if errorlevel 1 goto fail

rem ---- 4. verify ------------------------------------------------------------
echo [4/4] verifying imports ...
"%VPY%" -c "import PyQt5, serial, yaml; print('      PyQt5 / pyserial / PyYAML  OK')"
if errorlevel 1 goto fail

echo.
echo ==== done ====
echo   start : double-click the launcher .bat next to this file
echo           (it prefers %VENV%\Scripts\pythonw.exe automatically^)
echo   still needed by hand:
echo     - ffmpeg        winget install --id Gyan.FFmpeg -e     ^(streaming^)
echo     - tkinter       comes with the standard python installer ^(probe window^)
echo     - inbound UDP 5001 allowed in the firewall             ^(clock sync^)
echo   then open the app and click the environment self-check button.
echo.
pause
exit /b 0

:fail
echo.
echo [X] Failed above. Copy this whole window when asking for help.
pause
exit /b 1
