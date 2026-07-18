@echo off
rem FactorMiner Web界面启动器 (浏览器打开 http://localhost:8501)
rem %~dp0 自动定位项目根; 用户目录重定向到项目内tmp_home, 保证streamlit不写C盘且随项目搬移
set "USERPROFILE=%~dp0..\tmp_home"
set "HOME=%~dp0..\tmp_home"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
if not exist "%USERPROFILE%" mkdir "%USERPROFILE%"
cd /d "%~dp0.."
.venv\Scripts\python.exe -m streamlit run factor_miner\webapp\app.py
pause
