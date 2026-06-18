# G-CHAT-B6 conditioning harness: POST a JSON body to /v1/chat, parse the SSE
# delta stream into one clean transcript string. Usage:
#   $t = & _b6_chat.ps1 -BodyPath body.json
param(
  [Parameter(Mandatory=$true)][string]$BodyPath,
  [int]$TimeoutSec = 240
)
$body = Get-Content -Raw -Path $BodyPath
$r = Invoke-WebRequest -Uri http://127.0.0.1:3000/v1/chat -Method POST -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec $TimeoutSec
$sb = New-Object System.Text.StringBuilder
foreach ($line in ($r.Content -split "`n")) {
  $line = $line.TrimEnd("`r")
  if ($line.StartsWith("data: ")) {
    $payload = $line.Substring(6)
    if ($payload -eq "[DONE]") { continue }
    try {
      $obj = $payload | ConvertFrom-Json
      if ($obj.PSObject.Properties.Name -contains "delta") { [void]$sb.Append($obj.delta) }
      elseif ($obj.PSObject.Properties.Name -contains "error") { [void]$sb.Append("<<ERROR: $($obj.error)>>") }
    } catch { }
  }
}
$sb.ToString()
