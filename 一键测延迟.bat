@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
rem ---------------------------------------------------------------------------
rem  One-click latency sweep (B-side switch).
rem
rem  Double-click = turn B's half on; everything after that is automatic:
rem    wait for A's segment announcements -> measure latency per segment ->
rem    receive A's report -> merge into one table -> send the conclusion
rem    back to A (its console pops the table up).
rem
rem  Before you start:
rem    1. On A: click the "push self-test" action in the deploy console.
rem       Order does not matter -- the handshake aligns both sides, B may
rem       even start first.
rem    2. On B: click "stop" on the live preview page first, to free UDP 5000.
rem       Only one receiver can bind that port.
rem    3. On A: the clock service and the on-screen timecode probe cards must
rem       be running. Without them the latency half cannot be measured at all
rem       (B's precheck rejects the whole run).
rem
rem  Details in Chinese:  python -m tools.stream_sweep --help
rem  Old manual flow (sequential list, needs timing):  ... --no-link
rem
rem  NOTE: keep this file pure ASCII. cmd.exe reads .bat with the console code
rem  page, so non-ASCII comment text gets executed as a command (mojibake
rem  "is not recognized" errors). Chinese output from python is fine because
rem  of chcp 65001 + PYTHONIOENCODING below.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"
python -m tools.stream_sweep --auto
echo.
pause
