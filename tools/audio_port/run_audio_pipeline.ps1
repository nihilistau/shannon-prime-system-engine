# KAI-3 §7.3 unattended chain: wait for the TTS render, then gen_audio_frames -> CTC train/export -> metal gate.
$K='D:\F\shannon-prime-repos\_xbar\p2b\kai3'; $wd="$K\wav"
$eng='D:\F\shannon-prime-repos\shannon-prime-system-engine'; $exe="$eng\build-cuda-vs22\tests\test_gemma4_cuda.exe"
$model='/mnt/d/Files/Models/Gemma4/gemma-4-12b-bucket'
$log="$K\audio_pipeline.log"; function L($m){ "$([DateTime]::Now.ToLongTimeString()) $m" | Tee-Object -FilePath $log -Append }
# 1. wait for render (>=70/72 wavs or stable for 3 min)
$prev=-1; $stable=0
for($t=0; $t -lt 80; $t++){
  $n=(Get-ChildItem $wd -Filter *.wav -EA SilentlyContinue).Count
  L "render wavs=$n"
  if($n -ge 70){ break }
  if($n -eq $prev){ $stable++ } else { $stable=0 }
  if($stable -ge 6 -and $n -ge 20){ L "render stalled at $n, proceeding with what we have"; break }
  $prev=$n; Start-Sleep 30
}
# 2. gen_audio_frames (WSL)
L "=== gen_audio_frames ==="
wsl -e bash -c "cd $($eng -replace '\\','/' -replace 'D:','/mnt/d') 2>/dev/null; python3 /mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port/gen_audio_frames.py --kai3_dir /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3 --out /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3/audio_frames.npz" 2>&1 | Tee-Object -FilePath $log -Append
# 3. CTC train + export packets + manifest (WSL)
L "=== audio_ctc_projector ==="
$pref='D:\F\shannon-prime-repos\_xbar\p2b\kai3\kai3_audio_packets\'
wsl -e bash -c "python3 /mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port/audio_ctc_projector.py --frames /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3/audio_frames.npz --model $model --epochs 150 --export --packets_dir /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3/kai3_audio_packets --manifest_out /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3/audio_manifest.txt --manifest_prefix '$pref'" 2>&1 | Tee-Object -FilePath $log -Append
# 4. metal gate (Windows engine) — projected frame packets -> gemma4_kv_inject_seq -> pivot
L "=== G-KAIROS-3-AUDIO metal gate ==="
$env:SP_GEMMA4_SPMODEL='D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model'
$env:SP_GEMMA4_SPTOK='D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer'
$env:SP_CUDA_DECODE_INT8='1'; $env:SP_G4_KAI3="$K\audio_manifest.txt"
& 'C:\Windows\System32\nvidia-smi.exe' --lock-gpu-clocks=1680 2>&1 | Out-Null
& $exe 2>&1 | Tee-Object -FilePath $log -Append
& 'C:\Windows\System32\nvidia-smi.exe' --reset-gpu-clocks 2>&1 | Out-Null
L "=== PIPELINE COMPLETE ==="
