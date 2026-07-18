@echo off
rem GP遗传规划引擎启动器 (断点续跑加 --resume); %~dp0 自动定位, 项目可整体搬移
cd /d "%~dp0.."
.venv\Scripts\python.exe scripts\run_gp.py %*
pause
