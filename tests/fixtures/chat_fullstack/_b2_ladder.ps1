param([int]$ntok, [string]$tag)
# synthetic prompt of $ntok valid token ids (mix to avoid degenerate); ids in [10, 2000)
$ids = New-Object int[] $ntok
for ($i=0; $i -lt $ntok; $i++) { $ids[$i] = 10 + ($i % 1990) }
$body = @{ prompt_tokens = $ids; max_tokens = 1; temperature = 0.0 } | ConvertTo-Json -Compress
# Start the request async; poll VRAM during it.
$job = Start-Job -ScriptBlock {
  param($b)
  try { Invoke-WebRequest -Uri "http://127.0.0.1:3000/v1/chat" -Method Post -ContentType "application/json" -Body $b -UseBasicParsing -TimeoutSec 600 | Out-Null } catch {}
} -ArgumentList $body
$peak = 0
while ($job.State -eq 'Running') {
  $used = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -split "`n")[0]
  if ($used -gt $peak) { $peak = $used }
  Start-Sleep -Milliseconds 200
}
Receive-Job $job | Out-Null; Remove-Job $job
"$tag ntok=$ntok PEAK_VRAM_MiB=$peak"
