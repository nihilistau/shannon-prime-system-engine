param([string]$tag, [string]$outfile)
$body = @{ prompt = "Explain in three sentences why arithmetic is the foundation of auditable computation."; max_tokens = 80; temperature = 0.0 } | ConvertTo-Json -Compress
$resp = Invoke-WebRequest -Uri "http://127.0.0.1:3000/v1/chat" -Method Post -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 300
$lines = $resp.Content -split "`n"
$text = ""
foreach ($ln in $lines) {
  if ($ln -match '^data: (.+)$') {
    $d = $matches[1]
    if ($d -eq '[DONE]') { continue }
    try { $j = $d | ConvertFrom-Json; if ($j.delta) { $text += $j.delta } } catch {}
  }
}
Set-Content -Path $outfile -Value $text -NoNewline -Encoding UTF8
$sha = (Get-FileHash -Path $outfile -Algorithm SHA256).Hash
"$tag SHA256=$sha LEN=$($text.Length)"
"$tag TEXT: $text"
