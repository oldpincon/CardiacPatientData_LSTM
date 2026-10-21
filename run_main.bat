@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0
".venv\Scripts\python.exe" LSTM_main.py > run_main_log.txt 2>&1
type run_main_log.txt
echo.
echo (輸出已存到 run_main_log.txt)
pause
