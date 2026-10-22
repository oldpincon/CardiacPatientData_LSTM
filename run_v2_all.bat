@echo off
REM Run all pre-specified v2 analyses (leakage-fixed). Expect several hours; run overnight.
REM LSTM 每折用新的程序跑（--one-fold，exit code 3 = 還有折沒跑完），避免 TF 記憶體累積；
REM 中斷後重新執行本檔會從尚未完成的折繼續。
cd /d %~dp0
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
python baselines_nested_v2.py --analysis primary --repeats 3
python baselines_nested_v2.py --analysis no_labs --repeats 3
python baselines_nested_v2.py --analysis min10 --repeats 3
call :lstm primary 3
call :lstm no_labs 1
call :lstm min10 1
echo ALL DONE
pause
goto :eof

:lstm
:lstm_loop
python -u LSTM_nested_cv_v2.py --analysis %1 --repeats %2 --trials 40 --one-fold >> log_v2_%1.txt 2>&1
if %errorlevel%==3 goto lstm_loop
echo LSTM %1 finished, exit code %errorlevel%
goto :eof
