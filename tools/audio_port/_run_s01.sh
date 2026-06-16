#!/bin/bash
# KAI-3 §7.3 sigma=0.1 plumbing null-floor: install light tokenizer, gen synth frames, train projector.
AP=/mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port
MODEL=/mnt/d/Files/Models/Gemma4/gemma-4-12b-bucket
cd "$AP"
echo "=== gen sigma=0.1 (synthetic-token architecture ladder) ==="
python3 gen_synth_frames.py --model "$MODEL" --out /tmp/kai3_s01.npz --noise_rel 0.1
echo "=== train projector ==="
python3 frame_projector.py --frames /tmp/kai3_s01.npz --model "$MODEL" --epochs 60 --lr 2e-3 \
    --out /tmp/kai3_proj_s01.pt --export --packets_dir "$AP/../../tests/fixtures/kai3_s01"
echo "=== DONE sigma=0.1 ==="
