# 從 GitHub 把所有分支與最新 commit 同步下來（不會刪除或覆蓋你本機未提交的修改）
$ErrorActionPreference = 'Continue'
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $proj
Start-Transcript -Path (Join-Path $proj 'git_sync_log.txt') -Force | Out-Null
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "找不到 git" }
    Write-Host "=== 目前狀態 ===" -ForegroundColor Cyan
    git status --short --branch 2>&1 | Out-Host
    Write-Host "=== 抓取所有分支 ===" -ForegroundColor Cyan
    git fetch --all --prune --tags 2>&1 | Out-Host
    if ($LASTEXITCODE) { throw "git fetch 失敗" }
    # 為每個遠端分支建立/快轉本機分支
    $cur = (git rev-parse --abbrev-ref HEAD).Trim()
    $remotes = git for-each-ref --format='%(refname:short)' refs/remotes/origin | Where-Object { $_ -ne 'origin/HEAD' -and $_ -ne 'origin' }
    foreach ($r in $remotes) {
        $b = $r -replace '^origin/',''
        git show-ref --verify --quiet "refs/heads/$b"
        if ($LASTEXITCODE) {
            git branch --track $b $r 2>&1 | Out-Host; Write-Host "新增本機分支 $b"
        } elseif ($b -eq $cur) {
            git merge --ff-only $r 2>&1 | Out-Host
            if ($LASTEXITCODE) { Write-Host "!! $b 無法快轉（可能有本機修改或分歧），未變更" -ForegroundColor Yellow }
            else { Write-Host "已更新目前分支 $b" }
        } else {
            git fetch . "${r}:${b}" 2>&1 | Out-Host
            if ($LASTEXITCODE) { Write-Host "!! $b 無法快轉，未變更" -ForegroundColor Yellow } else { Write-Host "已更新分支 $b" }
        }
    }
    Write-Host "=== 同步後 ===" -ForegroundColor Cyan
    git status --short --branch 2>&1 | Out-Host
    git log --oneline --graph --all -n 12 2>&1 | Out-Host
    Write-Host "SYNC_DONE"
} catch { Write-Host "SYNC_FAILED: $_" -ForegroundColor Red }
finally { Stop-Transcript | Out-Null }
