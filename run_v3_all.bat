@echo off
REM Run all pre-specified v3 LSTM variants (see LSTM_v3_預先指定.md). Expect ~17+ hours.
REM 每折用新的程序跑（--one-fold，exit code 3 = 還有折沒跑完）；中斷後重新執行本檔會接續。
REM 由快到慢：D → B → C → A → E，每個版本跑完三個分析。
cd /d %~dp0
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
for %%v in (D B C A E) do (
    call :lstm %%v primary
    call :lstm %%v no_labs
    call :lstm %%v min10
)
python delong_v3.py > log_v3_delong.txt 2>&1
echo ALL DONE
pause
goto :eof

:lstm
:lstm_loop
python -u LSTM_v3.py --variant %1 --analysis %2 --trials 40 --one-fold >> log_v3%1_%2.txt 2>&1
if %errorlevel%==3 goto lstm_loop
echo LSTM v3%1 %2 finished, exit code %errorlevel%
goto :eof
