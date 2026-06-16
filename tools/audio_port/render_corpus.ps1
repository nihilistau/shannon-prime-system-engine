param([string]$Voice="casual_female", [int]$MaxTrain=100000)
$ErrorActionPreference="Continue"
$vx='C:\Projects\voxtral-mini-realtime-rs\target\release\voxtral.exe'
$m ='C:\Projects\voxtral-mini-realtime-rs\models\voxtral-tts-q4-gguf\voxtral-tts-q4.gguf'
$vd='C:\Projects\voxtral-mini-realtime-rs\models\voxtral-tts-q4-gguf\voice_embedding'
$K ='D:\F\shannon-prime-repos\_xbar\p2b\kai3'
$wd="$K\wav"; if(!(Test-Path $wd)){ New-Item -ItemType Directory -Force $wd | Out-Null }
Set-Location 'C:\Projects\voxtral-mini-realtime-rs'   # voxtral needs repo-root cwd for GPU/runtime ctx (else CPU-only, ~4x slower)
foreach($split in 'eval','train'){
  $lines = Get-Content "$K\$split.txt"
  for($i=0; $i -lt $lines.Count; $i++){
    if($split -eq 'train' -and $i -ge $MaxTrain){ break }
    $out="$wd\${split}_${i}_$Voice.wav"
    if(Test-Path $out){ continue }
    & $vx speak --gguf $m --voices-dir $vd --voice $Voice --euler-steps 3 --text $lines[$i] --output $out *>&1 |
      Select-String -Pattern 'Saved|Error|panic' | ForEach-Object { "[$split $i] $_" }
    "[$split $i] done -> $(Test-Path $out)"
  }
}
"RENDER_DONE voice=$Voice"
