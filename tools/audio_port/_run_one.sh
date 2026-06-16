#!/bin/bash
# usage: _run_one.sh <noise_rel>  — one rung of the KAI-3 boundary-resolution ladder, foreground.
S="$1"
AP=/mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port
MODEL=/mnt/d/Files/Models/Gemma4/gemma-4-12b-bucket
cd "$AP"
echo "===== sigma=$S ====="
python3 gen_synth_frames.py --model "$MODEL" --out /tmp/k3_${S}.npz --noise_rel "$S" 2>&1 | grep -aE 'gen\]|Error|Trace'
python3 frame_projector.py --frames /tmp/k3_${S}.npz --model "$MODEL" --epochs 80 --lr 2e-3 \
    --out /tmp/k3p_${S}.pt 2>&1 | grep -aE 'BEST|tripwire|Error|Trace'
echo "===== done sigma=$S ====="
