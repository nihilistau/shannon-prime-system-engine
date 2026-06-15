#!/usr/bin/env python3
"""
colab_kai2_probe.py — KAI-2 G4 Colab FEASIBILITY PROBE (run via `colab run --gpu A100 ...`).

Validates the four things that could each kill the KAI-2 phase-2 cloud training BEFORE we spend on
a full run, on the real google/gemma-4-12B "Unified":
  1. GPU entitlement + VRAM (prints device + total GiB).
  2. transformers can load the brand-new `gemma4_unified` arch (installs HEAD if the stable wheel
     lacks Gemma4UnifiedForConditionalGeneration).
  3. the 12B fits (bf16 if VRAM>=40 GiB, else on-the-fly bnb-4bit fallback — there is NO pre-quant
     gemma-4 on the Hub as of 2026-06-15, all 4-bit "12b" repos are gemma-3).
  4. the inject seam works: build inputs_embeds = [sys | k codec-slot vectors | decide] and forward
     with NO crash (this is the PyTorch twin of the engine's gemma4_kv_inject residual entry).

TOKEN-FREE: reads HF_TOKEN from the environment (the launcher's secret-pipe sets it in a /tmp
prelude that is prepended before this body, so nothing is committed). Prints a JSON summary line
`PROBE_RESULT {...}` and exits 0 on success, non-zero on any failure (colab run propagates it).
"""
import os, sys, json, importlib, subprocess


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=True)


def ensure_transformers():
    """Return True once Gemma4UnifiedForConditionalGeneration is importable; install HEAD if needed."""
    for attempt in ("stable", "head"):
        try:
            import transformers  # noqa
            from transformers import AutoModelForMultimodalLM  # noqa
            # arch presence check (the class is registered lazily; probe the model_type map)
            from transformers.models.auto import modeling_auto  # noqa
            import transformers as t
            if "gemma4_unified" in getattr(t, "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES", {}) or attempt == "head":
                return t.__version__
        except Exception:
            pass
        if attempt == "stable":
            print("[probe] installing transformers HEAD for gemma4_unified ...", flush=True)
            _pip("git+https://github.com/huggingface/transformers", "accelerate", "safetensors")
            for m in list(sys.modules):
                if m.startswith("transformers"):
                    del sys.modules[m]
    import transformers
    return transformers.__version__


SYSTEM = ("You are a background monitor. Read the event. Emit <ACTION> if it needs intervention "
          "(salience>=0.5), else NO_OP.\nEVENT: ")
DECIDE = "\nDECIDE: "
SALIENT = "EVENT build_id=4471 status=FAILED tests=3_broken salience=0.85"
IDLE = "EVENT heartbeat ok cpu=12% salience=0.10"


def main():
    import torch
    res = {"stage": "start"}
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gib = (torch.cuda.get_device_properties(0).total_memory // 2**30) if torch.cuda.is_available() else 0
    res.update(gpu=name, vram_gib=gib)
    print(f"[probe] GPU={name} VRAM={gib}GiB", flush=True)

    tv = ensure_transformers()
    res["transformers"] = tv
    from huggingface_hub import login
    tok_env = os.environ.get("HF_TOKEN")
    if tok_env:
        login(tok_env)
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    mid = os.environ.get("KAI2_MODEL", "google/gemma-4-12B")
    res["model_id"] = mid

    load_kw = dict(device_map="auto")
    if gib >= 40:
        load_kw["dtype"] = "auto"           # bf16, faithful teacher
        res["load"] = "bf16"
    else:
        _pip("bitsandbytes")
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16", bnb_4bit_quant_type="nf4")
        res["load"] = "bnb-4bit"
    print(f"[probe] loading {mid} ({res['load']}) ...", flush=True)
    tokzr = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForMultimodalLM.from_pretrained(mid, **load_kw)
    model.eval()
    dev = next(model.parameters()).device
    H = model.config.text_config.hidden_size
    res.update(loaded=True, hidden=H, model_type=model.config.model_type)
    print(f"[probe] loaded model_type={model.config.model_type} hidden={H}", flush=True)

    embed = model.get_input_embeddings()

    def teacher_decode(ev):
        ids = tokzr(SYSTEM + ev + DECIDE, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            lg = model(input_ids=ids).logits[:, -1, :]
        top = lg[0].topk(5).indices.tolist()
        return tokzr.decode(lg[0].argmax().item()), [tokzr.decode(t) for t in top]

    # 1) teacher text path on salient vs idle (does the model decide differently?)
    s_top1, s_top5 = teacher_decode(SALIENT)
    i_top1, i_top5 = teacher_decode(IDLE)
    res["teacher_salient_top1"], res["teacher_salient_top5"] = s_top1, s_top5
    res["teacher_idle_top1"], res["teacher_idle_top5"] = i_top1, i_top5
    print(f"[probe] teacher salient->{s_top1!r} top5={s_top5}", flush=True)
    print(f"[probe] teacher idle   ->{i_top1!r} top5={i_top5}", flush=True)

    # 2) THE INJECT SEAM: replace the event tokens with k random codec-slot vectors via inputs_embeds
    k = int(os.environ.get("KAI2_K", "4"))
    pre = tokzr(SYSTEM, return_tensors="pt").input_ids.to(dev)
    post = tokzr(DECIDE, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    pre_e, post_e = embed(pre), embed(post)
    soft = torch.randn(1, k, H, device=dev, dtype=pre_e.dtype) * (H ** -0.5)  # stand-in codec output
    inputs_embeds = torch.cat([pre_e, soft, post_e], dim=1)
    with torch.no_grad():
        lg = model(inputs_embeds=inputs_embeds).logits[:, -1, :]
    res["inject_ok"] = True
    res["inject_top1"] = tokzr.decode(lg[0].argmax().item())
    print(f"[probe] inject seam OK (k={k}) -> next={res['inject_top1']!r}", flush=True)

    res["stage"] = "done"
    print("PROBE_RESULT " + json.dumps(res), flush=True)
    print("PROBE_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("PROBE_RESULT " + json.dumps({"stage": "FAIL", "error": repr(e)}), flush=True)
        sys.exit(1)
