@echo off
rem 带控制台的启动方式 —— 出错时能看到完整报错。
rem 平时用「数据工作台.bat」，界面起不来或者任务失败查不出原因时用这个。
cd /d "%~dp0"
python -m gui.app
echo.
echo ==== 程序已退出，按任意键关闭 ====
pause >nul
