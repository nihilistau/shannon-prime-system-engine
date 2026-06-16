#!/usr/bin/env python3
# KAI-3: emit SHORT, number-free, speech-primed event text for the Q4 voxtral TTS. Raw events have
# id/metric numbers + symbols that make the autoregressive backbone RUN AWAY (133s+, never ends). The
# decision signal is salience high/low + status — that's all the EAR needs. Short events (~6 words) render
# fast (~40s) on GPU. Salience>=0.5 -> "salience high" (ACTION), else "salience low" (NO_OP). Emits
# *_spoken.txt + a flat Windows render .cmd (cd-once = GPU ctx, one voxtral call/line, skip-if-exists).
import os, re
K = "/mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3"
KW = r"D:\F\shannon-prime-repos\_xbar\p2b\kai3"
VXR = r"C:\Projects\voxtral-mini-realtime-rs"
EXE = r"target\release\voxtral.exe"; M = r"models\voxtral-tts-q4-gguf\voxtral-tts-q4.gguf"
VD = r"models\voxtral-tts-q4-gguf\voice_embedding"
MAXTRAIN = 64

def spoken(raw):
    # raw: "EVENT <type> id=N status=S metric=M% salience=V"
    typ = re.search(r"EVENT\s+(\w+)", raw); typ = typ.group(1) if typ else "event"
    st  = re.search(r"status=(\w+)", raw);  st  = st.group(1).lower() if st else ""
    sal = re.search(r"salience=([0-9.]+)", raw); v = float(sal.group(1)) if sal else 0.0
    word = "high" if v >= 0.5 else "low"
    parts = [typ]
    if st: parts += ["status", st]
    parts += ["salience", word]
    return " ".join(parts)

lines = [f"cd /d {VXR}"]
for split in ("eval", "train"):
    src = os.path.join(K, f"{split}.txt")
    if not os.path.exists(src): continue
    raw = [l.rstrip("\n") for l in open(src) if l.strip()]
    sp = [spoken(l) for l in raw]
    open(os.path.join(K, f"{split}_spoken.txt"), "w").write("\n".join(sp) + "\n")
    n = len(sp) if split == "eval" else min(len(sp), MAXTRAIN)
    for i in range(n):
        wav = f"{KW}\\wav\\{split}_{i}_casual_female.wav"
        lines.append(f'if not exist "{wav}" {EXE} speak --gguf {M} --voices-dir {VD} '
                     f'--voice casual_female --euler-steps 3 --text "{sp[i]}" --output "{wav}"')
    print(f"[norm] {split}: {len(sp)} short events, {n} queued; sample: {sp[0]!r}")
lines.append("echo RENDER_ALL_DONE")
open(os.path.join(K, "render_all.cmd"), "w").write("\r\n".join(lines) + "\r\n")
print(f"[norm] emitted render_all.cmd ({len(lines)-2} lines)")
