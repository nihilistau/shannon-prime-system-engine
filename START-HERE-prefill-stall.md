# START HERE — fix the daemon prefill stall (the one blocker)

One job this session: **the sp-daemon's prefill wedges on prompts larger than ~1000 tokens.**
Fix that and the agent gateway (persona + tools + memory) unblocks, and long console chats stop
silently stalling. Everything else is already done and committed.

## 1. Reproduce it (≈1 min, no harness)
Daemon must be running: `run_console.bat` (wait for "listening", ~28s first-request warmup).
```
cd tools\sp_daemon ... (or engine root)
set SP_WORDS=200  && python tests\perf\_g_bigprompt_probe.py   ->  n=299  completes ("Hello")
set SP_WORDS=1300 && python tests\perf\_g_bigprompt_probe.py   ->  n=1765 STALLS forever
```
If the daemon wedges, it stays wedged (single-threaded) — `taskkill /F /IM sp-daemon.exe` and relaunch.

## 2. Don't re-derive — already ruled OUT (last session)
- NOT persist  (SP_PERSIST_KV=0 still stalls)
- NOT the SWA ring window  (RING_W=4096, with 1765 < ring, still stalls)
- NOT temperature, NOT the harness
- NOT the decode — `SP_MAXTOK=1` (ask for ONE token) ALSO stalls ⇒ **it never reaches decode ⇒ the
  PREFILL is the wedge.**
Threshold is between n=299 (ok) and n=1765 (stall); persona-sized ~840-tok chats work, so the wall
is roughly n ≈ 1000–1700.

## 3. Where to look
`tools/sp_daemon/src/cuda_kvdecode_dispatch.rs` → `prefill` → engine `gemma4_kv_prefill` in
`src/backends/cuda/cuda_forward.cu`. Suspect a prefill chunk/batch bound, a buffer sized < ~1700,
or a host-side prefill loop. (Ring wrap + persist already excluded.)

## 4. First diagnostic move (cheap, do this first)
Add a `tracing::info!` / printf immediately BEFORE and AFTER the prefill call, and inside the
prefill loop (per chunk), for n>1024. Run the n=1765 repro:
- log stops before the kernel  → host-side loop/bound is the wedge.
- kernel launches, never returns → CUDA kernel hang (then: `compute-sanitizer` the big-prefill run,
  and bisect the threshold with SP_WORDS=600/900/1100 to pin the exact breaking size).

## 5. Build + verify
Build (CUDA): `tools\sp_daemon\_e2e_build.bat` (cargo build --release --features wire_cuda_backend;
~30s incremental; BUILD_EXIT=0). Relaunch `run_console.bat`.
DONE-WHEN: `SP_WORDS=1300 python tests\perf\_g_bigprompt_probe.py` → `got_DONE=True`. Then the gateway:
`run_gateway.bat` + `set SP_TEST_PORT=8800 && python tests\perf\_g_memory_check.py` → completes.

## Context (already shipped, all committed)
OKFS-tiered tools (harness dec6a67), eot_bias plumbed end-to-end, editable alive persona.md,
repetition-collapse fix (rep 1.3). Full diagnosis: `tests/perf/GATEWAY-PREFILL-STALL.md`.
Memory: `project_daemon_prefill_stall`. After the fix: gateway = #71, then #72 persona textarea,
#73 self-knowledge seeding.
