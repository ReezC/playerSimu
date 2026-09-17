@echo off
rem 双击启动数据集工作台。
rem
rem %~dp0 = 本脚本所在目录，所以放桌面快捷方式、挪到别处都不会失效。
rem pythonw 而不是 python：前者不带控制台黑框，界面上班族一点。
rem 出错时它不显示信息，排查请改用「数据工作台（调试）.bat」。
cd /d "%~dp0"
start "" pythonw -m gui.app
