#!/bin/bash
set -e
AP=/mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port
K=/mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3
MODEL=/mnt/d/Files/Models/Gemma4/gemma-4-12b-bucket
cd "$AP"
mkdir -p "$K/packets"
EXP=$(cat "$K/expect.txt")
echo "=== gen (real tokens, sigma=0.1) expect=$EXP ==="
python3 gen_synth_frames.py --model "$MODEL" --out "$K/real.npz" \
    --train_tokens "$K/train_tok.txt" --eval_tokens "$K/eval_tok.txt" --eval_expect "$EXP" \
    --noise_rel 0.1 2>&1 | grep -aE 'gen\]|Error|Trace'
echo "=== train + export ==="
python3 frame_projector.py --frames "$K/real.npz" --model "$MODEL" --epochs 80 --lr 2e-3 \
    --out "$K/proj.pt" --export --packets_dir "$K/packets" \
    --manifest_out "$K/manifest.txt" --manifest_prefix 'D:\F\shannon-prime-repos\_xbar\p2b\kai3\packets\' \
    2>&1 | grep -aE 'BEST|tripwire|export|manifest|Error|Trace'
echo "=== DONE ==="
