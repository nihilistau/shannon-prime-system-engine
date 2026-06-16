# KAI-3 §7.3 OVERNIGHT multi-voice bake: normalize -> tokenize -> render (2 voices x 512) -> gen -> CTC train -> metal gate.
$ErrorActionPreference="Continue"
$K='D:\F\shannon-prime-repos\_xbar\p2b\kai3'; $wd="$K\wav"
$eng='D:\F\shannon-prime-repos\shannon-prime-system-engine'; $exe="$eng\build-cuda-vs22\tests\test_gemma4_cuda.exe"
$ap='/mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port'
$model='/mnt/d/Files/Models/Gemma4/gemma-4-12b-bucket'
$log="$K\overnight_audio.log"; function L($m){ "$([DateTime]::Now.ToString('HH:mm:ss')) $m" | Tee-Object -FilePath $log -Append }
L "=== KAI-3 multi-voice overnight bake start ==="

# 1. normalize + emit multi-voice render list (full 512 x 2 voices)
$env:KAI3_MAXTRAIN="512"; $env:KAI3_VOICES="casual_female,casual_male"
L "normalize + emit render_all.cmd"
wsl -e bash -c "python3 $ap/normalize_and_emit.py" 2>&1 | Tee-Object -FilePath $log -Append

# 2. re-dump CTC token targets from spoken text
$env:SP_GEMMA4_SPTOK='D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer'
foreach($s in 'eval','train'){ $env:SP_G4_TOK_DUMP_IN="$K\${s}_spoken.txt"; $env:SP_G4_TOK_DUMP_OUT="$K\${s}_tok.txt"; & $exe 2>&1 | Select-String 'tokdump' | ForEach-Object { L $_.Line } }
Remove-Item Env:SP_G4_TOK_DUMP_IN,Env:SP_G4_TOK_DUMP_OUT -EA SilentlyContinue

# 3. render the multi-voice corpus (GPU via render_all.cmd's cd to C:\Projects). Synchronous (~7h).
L "=== rendering multi-voice corpus (this is the long pole ~7h) ==="
& cmd /c "$K\render_all.cmd" 2>&1 | Out-Null
L "render done: $((Get-ChildItem $wd -Filter *.wav -EA SilentlyContinue).Count) wavs"

# 4. gen_audio_frames (multi-voice glob + 24k->16k + log-mel + dur filter + eval_expect labels)
L "=== gen_audio_frames ==="
wsl -e bash -c "python3 $ap/gen_audio_frames.py --kai3_dir /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3 --out /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3/audio_frames.npz" 2>&1 | Tee-Object -FilePath $log -Append

# 5. scaled CTC train (minibatch + cosine) + export packets + manifest (now ACTION/NO_OP labelled)
L "=== audio_ctc_projector (400 ep, bs 32, cosine) ==="
$pref='D:\F\shannon-prime-repos\_xbar\p2b\kai3\kai3_audio_packets\'
wsl -e bash -c "python3 $ap/audio_ctc_projector.py --frames /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3/audio_frames.npz --model $model --epochs 400 --batch_size 32 --lr 1e-3 --export --packets_dir /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3/kai3_audio_packets --manifest_out /mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3/audio_manifest.txt --manifest_prefix '$pref'" 2>&1 | Tee-Object -FilePath $log -Append

# 6. G-KAIROS-3-AUDIO metal gate
L "=== G-KAIROS-3-AUDIO metal gate ==="
$env:SP_GEMMA4_SPMODEL='D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model'
$env:SP_CUDA_DECODE_INT8='1'; $env:SP_G4_KAI3="$K\audio_manifest.txt"
& 'C:\Windows\System32\nvidia-smi.exe' --lock-gpu-clocks=1680 2>&1 | Out-Null
& $exe 2>&1 | Select-String 'kai3\]' | ForEach-Object { L $_.Line }
& 'C:\Windows\System32\nvidia-smi.exe' --reset-gpu-clocks 2>&1 | Out-Null
L "=== OVERNIGHT BAKE COMPLETE ==="
