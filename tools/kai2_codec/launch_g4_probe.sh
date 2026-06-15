#!/bin/bash
# KAI-2 G4 probe launcher — run by path (RUNBOOK §3 gotcha: never inline through PS->WSL quoting).
#   wsl -e bash -c "sed -i 's/\r//' <thispath> && bash <thispath>"
set -e
export PATH="$HOME/.local/bin:$PATH"   # non-login shell: colab/runpod live here
TOKA=/mnt/d/F/shannon-prime-repos/archive/notes_and_stuff/claude-hf-token.txt
TOKB=/mnt/d/F/shannon-prime-repos/archive/notes_and_stuff/creds/claude-hf-token.txt
PROBE=/mnt/d/F/shannon-prime-repos/shannon-prime-system-engine/tools/kai2_codec/colab_kai2_probe.py
# mount-wait (fresh WSL sessions race the drvfs automount)
for i in $(seq 1 15); do ls /mnt/d >/dev/null 2>&1; [ -f "$PROBE" ] && break; sleep 1; done
TOK=""
[ -f "$TOKA" ] && TOK="$TOKA"
[ -f "$TOKB" ] && TOK="$TOKB"
{ [ -n "$TOK" ] && [ -f "$PROBE" ]; } || { echo "FILES MISSING tok='$TOK' probe='$PROBE'"; exit 1; }
T=$(tr -d '\r\n' < "$TOK")
{ printf 'import os\nos.environ["HF_TOKEN"]="%s"\nos.environ["KAI2_MODEL"]="google/gemma-4-12B"\n' "$T"; cat "$PROBE"; } > /tmp/kai2_run.py
echo "assembled /tmp/kai2_run.py: $(wc -l < /tmp/kai2_run.py) lines (token not echoed)"
cd ~
PYTHONUNBUFFERED=1 nohup colab run --gpu G4 --timeout 2400 /tmp/kai2_run.py > /tmp/kai2_probe.log 2>&1 &
echo "LAUNCHED colab run (G4) pid $!"
sleep 12
echo "=== first log lines ==="
head -25 /tmp/kai2_probe.log 2>/dev/null || true
