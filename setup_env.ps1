# CardiacPatientData_LSTM 環境自動設定
# 1) 確保有 Python 3.13（TensorFlow 尚未支援 3.14）
# 2) 建立 .venv  3) 安裝套件  4) 驗證
$ErrorActionPreference = 'Stop'
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $proj
$log = Join-Path $proj 'setup_log.txt'
Start-Transcript -Path $log -Force | Out-Null

function Find-Py313 {
    $cands = @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "C:\Program Files\Python313\python.exe"
    )
    foreach ($c in $cands) { if (Test-Path $c) { return $c } }
    try { $p = & py -3.13 -c "import sys;print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
    return $null
}

try {
    Write-Host "=== [1/4] 檢查 Python 3.13 ===" -ForegroundColor Cyan
    $py = Find-Py313
    if (-not $py) {
        $ok = $false
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host "用 winget 安裝 Python 3.13..."
            & winget install -e --id Python.Python.3.13 --scope user --silent --accept-package-agreements --accept-source-agreements
            $py = Find-Py313; if ($py) { $ok = $true }
        }
        if (-not $ok) {
            Write-Host "改從 python.org 下載安裝程式..."
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $idx = (Invoke-WebRequest 'https://www.python.org/ftp/python/' -UseBasicParsing).Content
            $vers = [regex]::Matches($idx, '3\.13\.(\d+)/') | ForEach-Object { [int]$_.Groups[1].Value } | Sort-Object -Unique -Descending
            foreach ($m in $vers) {
                $v = "3.13.$m"; $url = "https://www.python.org/ftp/python/$v/python-$v-amd64.exe"
                $exe = Join-Path $env:TEMP "python-$v-amd64.exe"
                try { Invoke-WebRequest $url -OutFile $exe -UseBasicParsing; break } catch { $exe = $null }
            }
            if (-not $exe) { throw "下載 Python 3.13 失敗" }
            Start-Process $exe -ArgumentList '/quiet','InstallAllUsers=0','PrependPath=0','Include_test=0','Include_launcher=0' -Wait
            $py = Find-Py313
        }
        if (-not $py) { throw "Python 3.13 安裝後仍找不到" }
    }
    Write-Host "使用 Python: $py"
    & $py --version

    Write-Host "=== [2/4] 建立虛擬環境 .venv ===" -ForegroundColor Cyan
    $venvPy = Join-Path $proj '.venv\Scripts\python.exe'
    if (Test-Path $venvPy) {
        $ver = & $venvPy -c "import sys;print('%d.%d'%sys.version_info[:2])"
        if ($ver -ne '3.13') { Write-Host "舊 .venv 是 $ver，重建"; Remove-Item -Recurse -Force (Join-Path $proj '.venv') }
    }
    if (-not (Test-Path $venvPy)) { & $py -m venv .venv; if ($LASTEXITCODE) { throw "venv 建立失敗" } }

    Write-Host "=== [3/4] 安裝套件（TensorFlow 約 300MB，請稍候）===" -ForegroundColor Cyan
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE) { throw "pip install 失敗" }

    Write-Host "=== [4/4] 驗證 ===" -ForegroundColor Cyan
    $env:PYTHONUTF8 = '1'; $env:TF_CPP_MIN_LOG_LEVEL = '2'
    & $venvPy -c "import tensorflow as tf, sklearn, pandas, numpy, optuna; from tensorflow.keras.preprocessing.sequence import pad_sequences; print('TF', tf.__version__, '| sklearn', sklearn.__version__, '| pandas', pandas.__version__, '| optuna', optuna.__version__); print('ENV_OK')"
    if ($LASTEXITCODE) { throw "套件匯入失敗" }
    Write-Host "`n環境設定完成！之後雙擊 run_main.bat 即可執行 LSTM_main.py" -ForegroundColor Green
    Write-Host "SETUP_SUCCESS"
} catch {
    Write-Host "SETUP_FAILED: $_" -ForegroundColor Red
} finally {
    Stop-Transcript | Out-Null
}
