#!/bin/bash
set -e
AP=/mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/audio_port
MODEL=/mnt/d/Files/Models/Gemma4/gemma-4-12b-bucket
echo "=== deps ==="
python3 - <<'PY'
import importlib
for m in ["numpy","torch","transformers","safetensors"]:
    try:
        mod=importlib.import_module(m); print("OK",m,getattr(mod,"__version__","?"))
    except Exception as e:
        print("MISSING",m,e)
PY
echo "=== model dir ==="
ls "$MODEL" 2>/dev/null | grep -E 'safetensors|tokenizer|config' | head || echo "MODEL DIR NOT FOUND: $MODEL"
