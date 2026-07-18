@echo off
rem RL强化学习引擎启动器 (断点续训加 --resume); %~dp0 自动定位, 项目可整体搬移
cd /d "%~dp0.."
.venv\Scripts\python.exe scripts\run_rl.py %*
pause
