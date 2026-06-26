# Gateway blocker = daemon PREFILL stalls on large prompts (2026-06-26)

## Symptom
The harness agent gateway (:8800 → daemon :3000) hangs on every turn: the daemon receives ONE
`/v1/chat`, logs the prompt (`S1 prompt ids: n=1341`/`1765`), and never finishes — no token, no
`[DONE]`. The console path (:3000 direct) is fine because its prompts are short.

## Root cause (isolated, reproducible)
The **daemon's prefill stalls on large prompts**. Not the harness, not persist, not the ring window.

Repro: `tests/perf/_g_bigprompt_probe.py` (sends a system prompt of ~SP_WORDS filler tokens
straight to :3000, `max_tokens` via SP_MAXTOK, `eot_bias=4`):
- `SP_WORDS=200`  (n=299)  → completes ("Hello"), got_DONE=True.
- `SP_WORDS=1300` (n=1765) → STALLS forever (daemon alive, no crash, no token).

Variables ruled OUT (all still stall on the big prompt):
- `SP_PERSIST_KV=0` (persist off) → stalls.
- `SP_DAEMON_KVDECODE_RING_W=4096` (1765 < ring, no wrap) → stalls.
- `SP_MAXTOK=1` (one token) → STALLS → it never reaches decode → **the PREFILL itself wedges**.
- temperature 0, eot_bias 4 present, single daemon (not wedged by a prior request — warm + clean).

Threshold: between n=299 (works) and n=1765 (stalls); and > ~840 (the alive-persona console chats
work, so a persona-sized prompt prefills fine). So the wall is roughly n≈1000–1700 tokens.

## Where to look (next session, FRESH — this is CUDA/Rust + a rebuild, do it sharp)
The kvdecode prefill path: `tools/sp_daemon` `cuda_kvdecode_dispatch::prefill` →
`gemma4_kv_prefill` in `src/backends/cuda/cuda_forward.cu`. Something in the large-prefill path
(a batch/chunk bound, a buffer sized below ~1700, a loop over positions) wedges. Note it is NOT
the SWA ring wrap (RING_W=4096 ruled that out) and NOT the persist reset (persist off ruled out),
so suspect the prefill kernel launch / a host-side prefill loop / a global-cache prefill bound.
First cheap step: add a log right before and after the prefill call for n>1024 to confirm it's the
prefill kernel vs a host loop; then compute-sanitizer the big-prompt prefill.

## Impact / interim
This also silently caps the console chat: conversations that grow past ~1000 tokens of prompt
would stall too (masked so far by short chats + the old W=24 cap). The alive persona on the
daemon path works for normal-length chats. The gateway (tools) stays blocked until the prefill
stall is fixed OR the agent prompt is kept under the threshold.

## Done this session (committed)
- OKFS-tiered tool loading (core full + gist-index LUT + load_tools on demand) — harness `dec6a67`.
- eot_bias plumbed end-to-end (InferenceConfig field + to_sp_chat + gateway cfg) — verified via a
  body-log that the gateway sends eot_bias=4 + max_tokens to the daemon.
- Alive editable persona (persona.md, live-loaded) + the repetition-collapse fix (rep 1.3).
