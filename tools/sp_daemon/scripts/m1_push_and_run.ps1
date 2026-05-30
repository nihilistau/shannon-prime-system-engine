# §4-MeMo Sprint M.1 — push binary + Memory model + tokenizer + run smoke.
#
# Prerequisites:
#   - Executive (qwen3_rt.sp-model + qwen3_rt.sp-tokenizer) already at
#     /data/local/tmp/ (was pushed during L3.FG).
#   - libsp_compute_skel.so already at /data/local/tmp/ (Sprint K.beta).
#
# Usage:
#   .\m1_push_and_run.ps1 [-Cycles 1000] [-ReportPath /data/local/tmp/m1_report.json]
#
# What it does:
#   1. Pushes Memory model + tokenizer from the host stable cache to device.
#   2. Pushes the freshly cross-compiled sp_memo_m1_smoke binary.
#   3. Runs the smoke; pulls the JSON report back to host.

param(
    [int]$Cycles = 1000,
    [string]$ReportPath = "/data/local/tmp/m1_report.json",
    [string]$BinaryPath = "..\..\..\target\aarch64-linux-android\release\sp_memo_m1_smoke"
)

$ErrorActionPreference = "Stop"

$ExecModel  = "/data/local/tmp/qwen3_rt.sp-model"
$ExecTok    = "/data/local/tmp/qwen3_rt.sp-tokenizer"
$MemoModel  = "/data/local/tmp/qwen25-coder-0.5b-memory.sp-model"
$MemoTok    = "/data/local/tmp/qwen25-coder-0.5b-memory.sp-tokenizer"

$HostMemoModel = "D:\F\shannon-prime-repos\models\qwen25-coder-0.5b-memory.sp-model"
$HostMemoTok   = "D:\F\shannon-prime-repos\models\qwen25-coder-0.5b-memory.sp-tokenizer"

Write-Host "[M.1] adb device check..."
adb devices

Write-Host "[M.1] checking pre-existing artifacts on device..."
$existing = adb shell "ls -la /data/local/tmp/ | grep -E 'qwen|libsp_compute_skel'" 2>&1
Write-Host $existing

# Push Memory model + tokenizer if not already present (or if sha differs — full push every time is safer)
Write-Host "[M.1] pushing Memory model to device..."
adb push $HostMemoModel $MemoModel
Write-Host "[M.1] pushing Memory tokenizer to device..."
adb push $HostMemoTok $MemoTok

# Push the smoke binary
$BinaryFull = Resolve-Path $BinaryPath -ErrorAction SilentlyContinue
if (-not $BinaryFull) {
    Write-Host "[M.1] ERROR: binary not found at $BinaryPath"
    Write-Host "[M.1] Build first via: cd tools\sp_daemon ; .\build-android.bat --bin sp_memo_m1_smoke"
    exit 1
}
Write-Host "[M.1] pushing binary $BinaryFull..."
adb push $BinaryFull /data/local/tmp/sp_memo_m1_smoke
adb shell chmod +x /data/local/tmp/sp_memo_m1_smoke

Write-Host "[M.1] running smoke..."
$cmd = @"
ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_memo_m1_smoke `
  $ExecModel $ExecTok $MemoModel $MemoTok `
  --cycles $Cycles --report-json $ReportPath
"@
adb shell $cmd

Write-Host "[M.1] pulling report..."
$LocalReport = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "m1_report.json"
adb pull $ReportPath $LocalReport
Write-Host "[M.1] report pulled to $LocalReport"
Get-Content $LocalReport
