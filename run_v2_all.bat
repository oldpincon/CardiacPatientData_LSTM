@echo off
REM Run all pre-specified v2 analyses (leakage-fixed). Expect several hours; run overnight.
cd /d %~dp0
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
python baselines_nested_v2.py --analysis primary --repeats 3
python baselines_nested_v2.py --analysis no_labs --repeats 3
python baselines_nested_v2.py --analysis min10 --repeats 3
python -u LSTM_nested_cv_v2.py --analysis primary --repeats 3 --trials 40 > log_v2_primary.txt 2>&1
python -u LSTM_nested_cv_v2.py --analysis no_labs --repeats 1 --trials 40 > log_v2_no_labs.txt 2>&1
python -u LSTM_nested_cv_v2.py --analysis min10 --repeats 1 --trials 40 > log_v2_min10.txt 2>&1
echo ALL DONE
pause
