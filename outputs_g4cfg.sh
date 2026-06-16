#!/bin/bash
B=/mnt/d/Files/Models/Gemma4/gemma-4-12b-bucket
echo "=== bucket .py + .json ==="
ls -la "$B"/*.py "$B"/*.json 2>/dev/null
echo "=== config arch / auto_map / audio / altup ==="
python3 - "$B/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
def g(d,k): return d.get(k)
print("architectures :", g(c,"architectures"))
print("model_type    :", g(c,"model_type"))
print("auto_map      :", g(c,"auto_map"))
print("audio_token_id:", g(c,"audio_token_id"))
print("top keys      :", list(c.keys()))
tc = c.get("text_config", {})
if tc:
    print("text model_type:", tc.get("model_type"))
    print("text keys      :", list(tc.keys()))
    for k in tc:
        if any(s in k for s in ("altup","laurel","per_layer","kv_shared","activation_sparsity","layer_types","hidden_size","num_hidden")):
            print("  text.",k,"=",tc[k])
PY
