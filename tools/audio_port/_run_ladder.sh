#!/bin/bash
# KAI-3 §7.3 noise ladder: boundary-resolution stress at sigma=0.3 and 0.5 (plumbing proven at 0.1).
AP=/mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port
MODEL=/mnt/d/Files/Models/Gemma4/gemma-4-12b-bucket
cd "$AP"
for S in 0.3 0.5; do
  echo "========== sigma=$S =========="
  python3 gen_synth_frames.py --model "$MODEL" --out /tmp/kai3_s${S}.npz --noise_rel $S
  python3 frame_projector.py --frames /tmp/kai3_s${S}.npz --model "$MODEL" --epochs 80 --lr 2e-3 \
      --out /tmp/kai3_proj_s${S}.pt
done
echo "========== LADDER DONE =========="
