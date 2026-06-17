#!/usr/bin/env python3
# KAI-3 §7.3 (GNA EAR) — turn rendered TTS WAVs into (log-mel features, CTC token target) pairs.
#
# Pipeline: voxtral TTS WAV (24kHz, from render_corpus.bat) -> resample 24k->16k -> log-mel
# (40ms hop = 640 samples @16k = the EAR frame stride) -> pair with the event's REAL token-id
# sequence (sp_tok_dump, kai3/{split}_tok.txt) as the CTC label. STAGED for Task #154.a; validated
# end-to-end only after the full corpus render bake. No torchaudio dep (numpy-only mel + stdlib wave).
#
# CTC (audio_ctc_projector.py) consumes these: encoder log-mel[T,n_mels] -> per-frame V_sub+blank
# logits, torch.nn.CTCLoss aligned to the (shorter) token target — handles frames!=tokens natively.
import argparse, glob, os, struct, sys, wave
import numpy as np

def load_wav_mono(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); n = w.getnframes(); ch = w.getnchannels(); sw = w.getsampwidth()
        raw = w.readframes(n)
    assert sw == 2, f"expect 16-bit PCM, got {sw*8}-bit"
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1: x = x.reshape(-1, ch).mean(1)
    return x, sr

def resample_to(x, sr_in, sr_out):
    if sr_in == sr_out: return x
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_in, sr_out); return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:  # numpy linear fallback (anti-alias is light; fine for a feature front-end first cut)
        n_out = int(round(len(x) * sr_out / sr_in))
        return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)

def mel_filterbank(sr, n_fft, n_mels, fmin=20.0, fmax=None):
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595.0 * np.log10(1 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1)
    m = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(m) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c == l: c = l + 1
        if r == c: r = c + 1
        fb[i - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[i - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb

def log_mel(x, sr=16000, hop=640, n_fft=1024, n_mels=64):
    # hop=640 @16k = 40ms = the EAR frame stride; one feature frame per 40ms
    win = np.hanning(n_fft).astype(np.float32)
    fb = mel_filterbank(sr, n_fft, n_mels)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad))
    frames = 1 + (len(xp) - n_fft) // hop
    out = np.empty((max(frames, 0), n_mels), np.float32)
    for t in range(frames):
        seg = xp[t * hop: t * hop + n_fft] * win
        spec = np.abs(np.fft.rfft(seg)) ** 2
        out[t] = np.log(fb @ spec + 1e-6)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kai3_dir", default="/mnt/d/F/shannon-prime-repos/_xbar/p2b/kai3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--hop", type=int, default=640)   # 40ms @16k
    ap.add_argument("--n_fft", type=int, default=1024)
    a = ap.parse_args()
    K = a.kai3_dir; out = a.out or os.path.join(K, "audio_frames.npz")

    def rd_tokens(p): return [[int(t) for t in ln.split()] for ln in open(p) if ln.strip()]
    splits = {}
    vsub = set()
    for split in ("train", "eval"):
        tokf = os.path.join(K, f"{split}_tok.txt")
        if not os.path.exists(tokf): continue
        toks = rd_tokens(tokf)
        feats, targs, flens, tlens = [], [], [], []
        for i, target in enumerate(toks):
            wavs = sorted(glob.glob(os.path.join(K, "wav", f"{split}_{i}_*.wav")))
            if not wavs: continue
            if split == "eval": wavs = wavs[:1]   # held-out = 1 clean voice/event, aligned to expect.txt (8)
            for wv in wavs:                                  # each voice = an augmented sample
                x, sr = load_wav_mono(wv); x = resample_to(x, sr, 16000)
                dur = len(x) / 16000.0
                if dur > 8.0:                                 # quality filter: short events are ~3-5s;
                    continue                                  # >8s = TTS runaway (no ASR oracle locally)
                lm = log_mel(x, 16000, a.hop, a.n_fft, a.n_mels)
                if lm.shape[0] < len(target):                 # CTC needs frames >= label length
                    continue
                feats.append(lm); targs.append(np.array(target, np.int64))
                flens.append(lm.shape[0]); tlens.append(len(target)); vsub.update(target)
        splits[split] = (feats, targs, flens, tlens)
        print(f"[gen-audio] {split}: {len(feats)} samples, mean frames {np.mean(flens) if flens else 0:.0f}, "
              f"mean tokens {np.mean(tlens) if tlens else 0:.0f}", flush=True)

    vsub = sorted(vsub); g2l = {g: l for l, g in enumerate(vsub)}
    save = {"vsub_ids": np.array(vsub, np.int64), "n_mels": np.int64(a.n_mels)}
    # propagate the held-out ACTION/NO_OP labels (from emit_corpus expect.txt) so the metal gate scores
    # against real decision labels, not packet indices (the 0/7-vs-3/7 harness bug, fixed 2026-06-16)
    expf = os.path.join(a.kai3_dir, "expect.txt")
    if os.path.exists(expf):
        save["eval_expect"] = np.array(open(expf).read().strip().split(","))
    for split, (feats, targs, flens, tlens) in splits.items():
        if not feats: continue
        Tm = max(flens); Um = max(tlens)
        X = np.zeros((len(feats), Tm, a.n_mels), np.float32)
        Y = np.full((len(feats), Um), -1, np.int64)          # local V_sub indices; -1 pad
        for j, (f, t) in enumerate(zip(feats, targs)):
            X[j, :len(f)] = f; Y[j, :len(t)] = [g2l[g] for g in t]
        save[f"{split}_X"] = X; save[f"{split}_Y"] = Y
        save[f"{split}_flen"] = np.array(flens, np.int64); save[f"{split}_tlen"] = np.array(tlens, np.int64)
    np.savez(out, **save)
    print(f"[gen-audio] V_sub={len(vsub)} -> {out}", flush=True)

if __name__ == "__main__":
    main()
