# shannon-prime-system-engine

The **engine layer** of the [shannon-prime-lattice](https://github.com/nihilistau/shannon-prime-lattice)
project: **backends + tools + gates on top of the math core**. Four backend
implementations (CPU AVX2/AVX-512, CUDA, Vulkan, Hexagon HVX) of the
[shannon-prime-system](https://github.com/nihilistau/shannon-prime-system)
forward path, the sovereign weight pipeline (`sp_transcode`), the tokenizer
modules, the Optane Ring-2 stores, and `sp_daemon` — a long-lived Rust
HTTP/SSE server that wraps the frozen L1 C ABI in a chat + dialogue +
PoUW-ledger + QUIC-mesh surface.

The math-core lives at `lib/shannon-prime-system/` as a Git submodule.
That submodule pin is what every engine build links against.

**The citable GPU headline (public ledger 06-R10):** **Gemma-4-12B at
26.1 tok/s and wikitext PPL 5.12 on an RTX 2060 12GB** — the OK_Q4B
sovereign artifact, gated 24/24 (CUDA graph EXACT 256/256, dp4a top-1
256/256; GPU PPL gate 5.1160 against the gold reference 4.6776, with
sim/CPU/GPU triple-agreement). Both halves stated: **llama.cpp is faster
at 31.29 tok/s — but on artifacts whose measured PPL is 192–506** (every
gemma-4 GGUF measurable in June 2026 carries broken weights; see the
public `GEMMA4-QUANT-FIX.md`). SP engine bandwidth 245 vs 207 GB/s (+18%).
The earlier 34.2 tok/s number is formally RETIRED (its artifact failed
the PPL gate). Details in §5.2.1d.

**The standing memory envelope:** the **two-ring long-context memory
(PPT-ARM)** — a query-directed recall router (±1 Rademacher projection) plus
a byte-addressable KV offload to NVMe/Optane — is implemented in the CPU
backend (`src/backends/cpu/cpu_forward.c` + `ring2_disk.c` +
`ring2_arm_backend.c`). Measured at 32k context: **910× resident KV-cache
shrink** (7.5 GB → 8.3 MB), needle retrieved **off a physical drive** at
**7.57 µs/read**, **8× KV sparsification at +0.69% perplexity**, bit-exact
when disabled. Honest negative, kept on the board: the C2.4 32k NIAH
finale was a **MISS** — the raw router breaks in the 64× selection-budget
regime (RAM ladder: 2k HIT, 4k miss-by-one-digit, 8k HIT); Ring 3
(RFC-XBAR §3.1) is the designed resolution. The **reducing loader**
(`tools/sp_transcode`) makes the on-disk model **~50% smaller with a
bit-faithful forward**. The receipts-first writeup, with one-command
reproductions, is at
**[Position Is Arithmetic](https://github.com/nihilistau/Position_Is_Arithmetic)**
(live site: https://nihilistau.github.io/Position_Is_Arithmetic/).

On desktop **CPU** raw decode throughput remains the known gap (~1.34×
behind a tuned llama.cpp at Q8; memory layout, not ALU) — the open
**P1 SPEED / WIRE** lane.

License: MIT. See `LICENSE`.

---

## 0. Where this repo sits — the four-tier rings + XBAR

XBAR (the **auditable latent crossbar**, lattice `RFC-XBAR`): **Exec** (the
big generator — the gemma4/qwen3 forwards here) and **Memo** (a small
curator, math-core `tools/curator/`) share a tiered latent memory; every
write to canonical memory is receipted, gated, and rewindable.

```
  Exec (this repo's CUDA/CPU forwards)   Memo (curator, math-core)
    │ write        ▲ recall from BOTH      │ propose         ▲ read
    ▼              │                       ▼                 │
  Ring 1 ────── Ring 2 (verbatim       ◄─ Ring 2′ (shadow staging:
  (working KV)   episodic Spinor KV,       promote-on-accept w/ receipt,
                 "hippocampus")            or REWIND)
                   ▲                          │ promote (gated)
                   └── Ring 3 (VSA/HRR gist consolidation;
                       R3.1–R3.4 CLOSED; engine tools/ring3/)
```

The engine owns: Exec's accelerated forwards (CUDA `gemma4_forward_cuda` /
`gemma4_decode_cuda`, the CPU overlay), the **`SP_XBAR_*` latent-crossbar
harness** in `cuda_forward.cu` (P1 KV transplant + P2.a residual injection
— ledger X-R1), the **Optane / QUIC Ring-2 stores**, the sovereign weight
pipeline, the tokenizer modules, and the daemon/QUIC tier. The math core
owns the decode loop, the ARM recall router, and the Ring 2′ curator
transaction. **Ring 3 Path A** (VSA/HRR, parameter-free) is **CLOSED**
(R3.1–R3.4 GREEN, `tools/ring3/`), and as of **2026-06-18 the whole XBAR
memory stack is UNIFIED onto the exact-integer O_K substrate** (`Q(√-163)` /
the dual-prime negacyclic CRT-NTT in math-core `core/ntt_crt` + `core/poly_ring`):
the Ring-3 bind runs on native `sp_pr_mul` **256/256 bit-identical** and
reduction-order-immune, Ring-2 episodes are a Frobenius π^k integer store,
and the full real-episode organism loop (audio → C2 sig → integer Ring-3 →
Hamming verify → Frobenius store → 12B cache) is GREEN
(`tools/ring3/{ok_bind.py,g_r3_bind_ok.py,g_xbar_organism_full.py}`,
`tools/curator/frob_episode.py`; receipts `tests/fixtures/xbar_r3/` +
`tests/fixtures/xbar_organism/`). **NIGHTSHIFT** (idle consolidation, R3.4)
is GREEN — RFC-XBAR §7.

---

## Contents

1. [What this repo provides](#1-what-this-repo-provides)
2. [Current status — honest table](#2-current-status--honest-table)
   - [2.1 Harness modes (env-var test gates)](#21-harness-modes--the-env-var-test-gates)
   - [2.2 Dated-update history](#22-dated-update-history-consolidated)
3. [Quick start](#3-quick-start)
4. [Architecture](#4-architecture)
5. [The backends](#5-the-backends)
6. [`sp_daemon` Rust crates](#6-sp_daemon-rust-crates)
7. [HTTP / SSE / WebSocket API](#7-http--sse--websocket-api)
8. [Hexagon skel IDL reference](#8-hexagon-skel-idl-reference)
9. [CLI flags + environment variables](#9-cli-flags--environment-variables)
10. [Model conversion (`sp_transcode`)](#10-model-conversion-sp_transcode)
11. [Peering / QUIC mesh](#11-peering--quic-mesh)
12. [Development workflow](#12-development-workflow)
13. [Known issues / pending](#13-known-issues--pending)

---

## 1. What this repo provides

This repo wires the math-core forward onto four accelerator backends plus the `sp_daemon` server, and on the CUDA path it carries the **space ⊗ time ⊗ cognition** crossbar substrate on the real 12B. **Measured here:**

- **Space (XBAR Phase C — O(1) KV; P3 now CLOSED end-to-end):** the cache is decoupled from context length — flat VRAM from 8k→16k (~50 MiB delta) with a learned-LSH sparse router (8× global compression at **+0.47% PPL**), and a needle planted in a 16k haystack **survives the compaction** at every depth (C-c NIAH). The write side is also closed: **P3.3 replay-write** (`SP_REPLAY`, `G-P3-SHARED` 3-leg PASS on 12B + E2B — intact episode bit-identical, zeroed episode diverges 12/12) and **P3.4 recall quality** (`G-P3-PPL` = **+1.38% < 2% gate**), so XBAR P3 reads, writes, compresses to O(1), retrieves under poison, replays bit-exactly, and recalls without breaking perplexity.
- **Time (KAIROS time-axis, persistent-KV ABI):** a resident KV cache that can be **rewound by an O(1) memory-coordinate shear** — bit-exact (`rewind` byte-identical across all 48 owner layers), with a journaled-ring variant (KAI-1c) that is wrap-aware. The rewind is **127× flatter per action** than the host re-prefill ("prefix-grow") hack it replaces.
- **Cognition (KAIROS crucible):** a 12B held to **disciplined silence** on idle ticks (`NO_OP`), acting coherently only on salient events and reverting cleanly — perfect on a 24-event tape (0 false / 0 missed / 0 drift), running the time + space machinery underneath. Endurance: a **6h unattended soak is GREEN** (351 loops / ~8,400 ticks / 6h01m on the dedicated 2060, 0 false / 0 missed / 0 pos-violation; the formal ≥24h gate is un-pursued by operator choice, not failed).
- **Latent interrupt (KAI-2):** **CLOSED** (engine `c5628e4`). The Phase-1 delivery seam `gemma4_kv_inject` is a **GREEN / frozen, verified asset** — the EMB control pivots the 12B (OK_Q4B, RTX 2060) 2/2. The Phase-2 learned **compressed single-event codec** is **BOUNDED**: the trained t10 packet reaches k=16, cos 0.9913 at τ=0.2 with `val_KL` plateauing at 0.9157 — a PACKET 1/2 (the wall is sequence-positional). No more codec-compression cycles; the seam stands as the delivery primitive.
- **Audio port (KAI-3) + the GNA "EAR" line — CLOSED end-to-end ON PHYSICAL SILICON:** KAI-3 (the bridge) is **CLOSED GREEN** (engine `e35a227`) — a *sequence* of N projected frames injected via the new ABI `gemma4_kv_inject_seq` (G-KAIROS-3-NULL 2/2 byte-identical to the inline EMB loop); the projector (`tools/audio_port/`, per-position MLP 640→V_sub + on-manifold binder, dense per-position CE, local/no-cloud via `SP_G4_TOK_DUMP`) hits 8/8 metal pivots. The GNA EAR line is now **realized on the hardware**: real TTS speech → log-mel → GNA-conservative Conv1d + CTC head → `gemma4_kv_inject_seq` → 12B pivots **7/8** (CTC token recovery 0.44→0.868, multi-voice bake). The front-end lowers clean to GNA 2.0 (encoder conv padding `1→0` VALID, CTC head `33→36` filters mult-of-4); **POT GNA-native i16 PTQ = 0.877 full FP32 recovery** (naive i16 sheared 0.877→0.667; NNCF INT8 on CPU = 0.860 but won't compile on GNA); **GNA_HW on the physical Intel GNA 2.0 (Beast Canyon) = 0.877 == SW_EXACT emu == FP32 — physically realized**. Tooling `tools/audio_port/{ov_gna_score,ov_score_ir,pot_gna_quantize}.py + run_gna_hw.bat` + `GNA_HW_BRINGUP.md` (native-Windows OpenVINO 2023.3; WSL2 has no GNA MMIO passthrough). The GNA EAR is a separate-but-related sibling of the KAIROS latent-memory work, sharing the `gemma4_kv_inject` seam.
- **The throughput floor:** the citable **Gemma-4-12B 26.1 tok/s @ wikitext PPL 5.12 on the RTX 2060** (CUDA, OK_Q4B sovereign artifact — ledger 06-R10, §5.2.1d), the **two-ring memory envelope** (910× resident KV @32k, 7.57 µs/read off Optane) and the **WIRE-CPU integer pipe** (Qwen3-0.6B 0.84 → 39.52 tok/s, 47×, ~1.34× behind llama.cpp Q8_0) — over a forward that is bit-exact on **5 arch families** (Qwen3, Qwen2.5-Coder, Gemma3, Gemma4, Qwen3.6-35B-A3B MoE).
- **Auditability (the byte-exact forward — CLOSED on the 12B):** the gemma4 CUDA forward now has an **exact-integer mode** (`SP_BYTEEXACT`, default-off null floor) that converts all 4 nonlinear fp32 islands (RMSNorm/softmax/GELU/RoPE) + attention to device integer kernels (dual-prime, no `__int128`; `k_attn_decode_win_bx`, device CORDIC RoPE). **G-BYTEEXACT-FORWARD-12B GREEN**: off = PPL 4.6665 byte-identical to baseline, on = 4.6569 parity, **run-to-run bit-identical** — the cross-machine determinism that makes a forward *auditable* (NOT compression). The universal daemon (`tools/sp_daemon`, feature `wire_cuda_backend`) drives the 12B prefill + token-by-token decode through the L1 ABI (new verb `sp_session_register_kvdecode_backend` + `gemma4_kv_decode_logits`, 32/32 == oracle, VRAM O(1)). The only open item is external (a 2nd physical GPU for the cross-machine logit check).

The KAIROS / XBAR / metal-eviction work all ships **test-path, env-gated, byte-inert when off** — the one-shot production decode (`gemma4_decode_cuda`) is left **byte-untouched** (the "null floor"), so every previously-closed throughput/PPL/NIAH gate stays valid. See [§2.1 Harness modes](#21-harness-modes--the-env-var-test-gates) for the env → gate map.

| Slot | Path | Status |
|------|------|--------|
| **Math-core submodule** | `lib/shannon-prime-system/` | linked into every backend; frozen L1 ABI |
| **KAIROS time-axis kernel** (cognitive crucible + metal eviction loop) | `tests/test_gemma4_cuda.c` (`SP_G4_KAIROS` / `_METAL` / `_SOAK`) + daemon scheduler `tools/sp_daemon/src/kairos.rs`, `kairos_runner.rs` (feature `kairos`, off by default) | shipped (test-path, env-gated); semantic crucible CLOSED; **6h soak GREEN** (351 loops / ~8,400 ticks / 6h01m unattended, 0 false / 0 missed / 0 pos-violation; ≥24h gate un-pursued by choice) |
| **Persistent-KV ABI** (resident cache w/ O(1) rewind) | `src/backends/cuda/cuda_forward.cu` — `gemma4_kv_open/prefill/decode/rewind/commit/pos/snapshot/close`, `struct sp_g4_kv` (`ring_W`, `Jmax`, `commit_pos`, `jK`/`jV` undo-journal) | shipped (twin of `gemma4_decode_cuda`, which is left byte-untouched = null floor); rewind bit-exact, O(1) |
| **Latent-inject delivery seam (KAI-2)** | `src/backends/cuda/cuda_forward.cu` — `gemma4_kv_inject` (residual-entry inject) + `gemma4_kv_inject_seq` (N-frame sequence wrapper) | shipped; KAI-2 **CLOSED** (delivery seam GREEN/frozen; compressed codec BOUNDED). Shared seam under both the KAIROS latent-memory line and the GNA audio "EAR" line |
| **Audio port / projector (KAI-3 → GNA "EAR")** | `tools/audio_port/` (per-position MLP 640→V_sub + on-manifold binder; OpenVINO/POT `{ov_gna_score,ov_score_ir,pot_gna_quantize}.py + run_gna_hw.bat` + `GNA_HW_BRINGUP.md`) + `tests/test_gemma4_cuda.c` (`SP_G4_INJ_SEQ`, `SP_G4_TOK_DUMP`, `SP_G4_KAI3`) | shipped + gated; KAI-3 **CLOSED GREEN** (8/8 pivots) and the **GNA EAR line CLOSED on physical silicon**: real speech → 12B 7/8 (CTC 0.44→0.868), POT GNA-native i16 = 0.877 full recovery, GNA_HW on Intel GNA 2.0 = 0.877 == emu == FP32. The near-term audio pivot is DONE |
| **Replay-write seam (P3.3) / recall-quality (P3.4)** | `src/backends/cuda/cuda_forward.cu` — `SP_REPLAY` inject at both prefill stores in `gemma4_decode_cuda` (graph + velocity) + `tests/test_gemma4_cuda.c` (`SP_G4_REPLAY_GATE`, `SP_G4_SCORE`) | shipped + gated; P3.3 `G-P3-SHARED` 3-leg PASS on 12B + E2B (intact bit-exact, zeroed diverges 12/12); P3.4 `G-P3-PPL` +1.38% < 2% gate — **XBAR P3 CLOSED end-to-end** |
| **CPU backend** | `src/backends/cpu/` (`cpu_forward.c`, `cpu_overlay.c`, `cpu_gemma3.c`, `avx512/`) | built |
| **Two-ring memory (PPT-ARM)** | `src/backends/cpu/cpu_forward.c` (±1 recall router + window shrink + compact-and-spill fusion) + `ring2_disk.c` (Optane NO_BUFFERING + IOCP) | shipped + measured (910× @32k, 7.57 µs/read) |
| **XBAR Phase C — O(1) KV + NIAH** (slab + ring + learned-LSH router) | `src/backends/cuda/cuda_forward.cu` (`SP_ARM_*` slab/ring/select) + `tests/test_gemma4_cuda.c` (`SP_G4_NIAH`) + `tools/xbar_lsh/train_lsh.py` | shipped + gated (8k↔16k flat ~50 MiB; 8× +0.47% PPL; needle survives) |
| **CUDA backend** | `src/backends/cuda/` (`cuda_forward.cu`, `ptx_mma*.cuh`, `ptx_ntt.cuh`, `ptx_spinor.cuh`, `ptx_hash.cuh`) | built |
| **Vulkan backend** | `src/backends/vulkan/` (`vulkan_forward.cpp`, `shaders/`) | built |
| **Hexagon HVX backend (host)** | `src/backends/hexagon/sp_hex_host.c` + `sp_hex_rt.c` + `inc/` | built |
| **Hexagon cDSP skel** | `tools/sp_compute_skel/src_dsp/` (Halide-AOT FFN + HVX NTT + VTCM staging) | built |
| **`sp_daemon` HTTP/SSE server** | `tools/sp_daemon/src/{main.rs, server.rs, routes.rs, daemon.rs}` | built |
| **`sp_transcode` — sovereign weight pipeline** (GGUF lane + `--st` safetensors-direct + OK_Q4B `--q4b`/`--q4b-ffn` + `--tok-only`) | `tools/sp_transcode/sp_transcode.c` | built |
| **Tokenizer modules** (incl. `GEMMA4_BPE` family dispatch, #115 — 5432/5432 HF parity both lanes) | `src/tokenizer/` (`gemma4_bpe.c`, `tokenizer.c`) | built |
| **`sp_dsp_smoke` standalone bridge** | `tools/sp_dsp_smoke/` | built |
| **`sp_npu_spike` Snapdragon NPU spike** | `tools/sp_npu_spike/` | built (K.2-spike) |
| **`sp_halide_gen` Halide AOT compiler** | `tools/sp_halide_gen/` | built |
| **`oracle` cross-backend bit-identity oracle** | `tools/oracle/` | built |
| **PPL harness** | `src/forward/ppl.c` | built |

---

## 2. Current status — honest table

**Current state first; the chronological dated-update log is consolidated in [§2.2](#22-dated-update-history-consolidated).** The table below is phase-grouped: the latest crossbar substrate (KAIROS time-axis, XBAR Phase C, Gemma4/OK_Q4B) on top, the daemon/backend wiring snapshot below. Every "on" result is a controlled delta against a byte-identical baseline — the production decode path (`gemma4_decode_cuda`) is never touched.

### Current state — phase-grouped

| Group | Component | Status | Headline receipt |
|-------|-----------|:------:|------------------|
| **Byte-exact forward (2026-06-18)** | Exact-integer islands + attention on the 12B CUDA forward (`SP_BYTEEXACT`) | CLOSED GREEN | **G-BYTEEXACT-FORWARD-12B**: Leg A `SP_BYTEEXACT` off = PPL **4.6665 == baseline byte-identical** (null floor); Leg B on = **4.6569 parity**; **run-to-run BIT-IDENTICAL** (determinism = the cross-machine proxy). Byte-exact = EXACT-INTEGER ARITHMETIC / cross-machine determinism (the AUDITABILITY mission), NOT compression. Device techniques avoid `__int128` (`__umul64hi`, 64-bit isqrt for RMS, device CORDIC RoPE, `bx_garner`/`bx_exp_fixed`, `k_attn_decode_win_bx`). Engine `69c0588`, submodule `d9d96f3`. Receipts `tests/fixtures/xbar_r3/`. **HONEST:** the ONE remaining item is EXTERNAL — a bit-identical logit check across two PHYSICAL GPUs (2nd machine; on-machine = run-to-run determinism + reduction-order immunity proxy); PPL parity at n=42 (small-N, −0.21% within noise) |
| | Exact-integer islands — fidelity gate (`G-BYTEEXACT-ISLANDS-CUDA`) | CLOSED GREEN | RMSNorm/softmax/GELU/RoPE converted to exact-integer device kernels; on-model fidelity RMS **3.8e-5** / GELU **8.2e-7** / RoPE **9.6e-6** vs float, order-immune. L2 references in `tools/sp_dsp_smoke/src/sp_islands_q_ref.rs` (G-ISLANDS-Q-REF, RoPE via fixed-point CORDIC = no libm) + math-core `core/exact_islands/` (gate `T_EXACT_ISLANDS`) |
| | Universal daemon drives the 12B end-to-end (feature `wire_cuda_backend`) | CLOSED GREEN | prefill via `sp_session_register_forward_backend` (**G-WIRE-CUDA-GEMMA4**) + token-by-token DECODE via the new L1 verb `sp_session_register_kvdecode_backend` + additive `gemma4_kv_decode_logits` (**G-WIRE-CUDA-DECODE-GEMMA4**: 32/32 tokens bit-identical to the oracle, VRAM flat O(1)). Gate bin `tools/sp_daemon/src/bin/sp_wire_cuda_decode_gate.rs` |
| **KAIROS time-axis (KAI-1/1b/1c)** | Semantic crucible (`SP_G4_KAIROS` / `_METAL`) — disciplined-silence loop on the 12B | CLOSED | 24-event tape **0 false / 0 missed / 0 drift**; the 3 salient ticks acted coherently + reverted clean. Negative control: a 0.6B collapses into a corruption attractor — proves it's model *capacity* through correct machinery, not plumbing |
| | Persistent-KV ABI — O(1) rewind (KAI-1b) | CLOSED | `gemma4_kv_rewind(Δ)` **byte-identical across all 48 owner layers** (16.5 MB, diffs=0) + gen-reproduce; **O(1)** — metal slope 0.0073 vs prefix-grow 0.924 s/action (**127× shallower**), 16.7× @ 16 retained actions |
| | Wrap-aware journaled ring (KAI-1c) | CLOSED | forced wrap-crossing tick **clobbered live slots in all 40 SWA layers** (non-vacuity), post-rewind ring **byte-identical (diffs=0)** + identical tokens; ring O(1) telemetry slope 0.00365 ≈ full-cache 0.00371 |
| | 6h endurance soak (`SP_G4_KAIROS_SOAK`, G-KAIROS-1) | **GREEN** | `SOAK_EXIT=0`; **351 loops / ~8,400 ticks / 6h01m** unattended on the dedicated 2060, **0 false / 0 missed / 0 malformed / 0 pos-violation**; tripwire-armed (CUDA error / false-action / pos-violation / malformed / latency / VRAM-leak / thermal); clocks reset on exit. The formal ≥24h gate is un-pursued by operator choice (not failed) |
| | Latent-inject delivery seam (KAI-2, `gemma4_kv_inject`) | CLOSED | delivery seam **GREEN / frozen verified asset** — EMB control pivots the 12B 2/2 (OK_Q4B, RTX 2060). The learned compressed single-event codec is **BOUNDED** (t10 packet k=16, cos 0.9913 @ τ=0.2, `val_KL` plateau 0.9157 → PACKET 1/2; wall = sequence-positional). No more codec-compression cycles. Engine `c5628e4` |
| **GNA "EAR" audio line** | Audio port (KAI-3, `gemma4_kv_inject_seq`) — bridge into the GNA line | CLOSED GREEN | inject a SEQUENCE of N projected frames via `gemma4_kv_inject_seq` (G-KAIROS-3-NULL 2/2 byte-identical to the inline EMB loop). Projector `tools/audio_port/` (per-position MLP 640→V_sub + on-manifold binder, dense per-position CE), local/no-cloud (`SP_G4_TOK_DUMP`). Synthetic ladder `noise_rel=0.1` top1 1.000/cos 0.9998; real-token V_sub=60 top1 0.931/cos 0.9937; metal gate `SP_G4_KAI3` **8/8 semantic pivots** (`KAI3_GATE_EXIT=0`). Engine `e35a227`. Related to, NOT a replacement for, the KAIROS latent-memory line (shared `gemma4_kv_inject` seam) |
| | GNA "EAR" — real audio front-end on physical silicon | CLOSED | real TTS speech → log-mel → GNA-conservative Conv1d + CTC head → `gemma4_kv_inject_seq` → 12B pivots **7/8** (CTC token recovery 0.44→0.868, multi-voice bake 924/2voices/400ep). Front-end lowers clean to GNA 2.0 (padding `1→0` VALID, CTC head `33→36` filters mult-of-4); **POT GNA-native i16 = 0.877 full FP32 recovery** (naive i16 sheared 0.667; NNCF INT8/CPU 0.860 won't compile on GNA); **GNA_HW on the physical Intel GNA 2.0 = 0.877 == SW_EXACT emu == FP32 — physically realized**. Native-Windows OpenVINO 2023.3 (WSL2 no GNA passthrough). Tooling `tools/audio_port/{ov_gna_score,pot_gna_quantize}.py + run_gna_hw.bat` + `GNA_HW_BRINGUP.md`. The audio pivot is DONE; project pivoted BACK to XBAR |
| **XBAR Phase C (O(1) KV + NIAH)** | O(1) KV — slab + SWA ring + device-select | CLOSED | N=8192 vs 16384 **VRAM flat within ~50 MiB** (a full O(N) cache adds ~5.4 GiB). Scope: the *KV term* is O(1); the ~11.4 GiB absolute floor is the resident 9.4 GiB model (a `test_gemma4_ppl_cuda` harness artifact that bypasses streaming — we deliberately do **not** claim "12B @ 16k on 12 GB") |
| | Learned-LSH sparse router (8×) | CLOSED | **+0.47% PPL** @ 8× global compression (oracle −0.08%; frozen ±1 +4.17% RED) — 512×32 projection, zero new hot-path kernels; weight `tests/fixtures/lsh/lsh_M_r32.bin` |
| | C-c NIAH retention (`SP_G4_NIAH`) | CLOSED | needle **survives the compaction** at depths 10%/50%/90% (learned-router only; frozen ±1 control **MISSes**) under SWA-isolation. Full-attention baseline @16k is physically impossible on the 2060 — the motivation |
| | P3.3 replay-write — `SP_REPLAY` (`SP_G4_REPLAY_GATE`) | CLOSED GREEN | `SP_REPLAY` injects a stored episode's owner-K/V over prefill rows `[0,NPOS)` at the cache-store boundary, before attention. `G-P3-SHARED` 3-leg PASS on BOTH **12B** (48 owners) + **E2B** (15 owners / 20 sharers, owner-indirection): intact episode **bit-identical** to baseline (diffs=0), zeroed episode **diverges 12/12**, unset = floor. Inject at **both** `gemma4_decode_cuda` prefill stores (graph ~L2516 + velocity ~L2825; velocity is the path the gate runs). Receipts `tests/fixtures/xbar_p3_replay/G-P3-SHARED_{12B,E2B}_GREEN.log` |
| | P3.4 recall quality — `G-P3-PPL` (`SP_G4_SCORE` ∘ `SP_REPLAY`) | CLOSED GREEN | the PPL scorer **is** `gemma4_decode_cuda` in `SP_G4_SCORE` mode → `SP_REPLAY` composed with **zero new engine code**. wiki.tiny n_ctx=84: recall-OFF 4.6665 → recall-ON (NPOS=4) 4.7311 = **+1.38% deflection < 2.0% gate → PASS**. Caveat: n_scored=42 single chunk (deterministic, not router-sampled); larger-N multi-chunk = the named hardening lever. Receipt `tests/fixtures/xbar_p3_replay/G-P3-PPL_run.log`. **XBAR P3 CLOSED end-to-end (P3.0→P3.4)** |
| **C2 Memo curator** | Autonomous Ring-2 recall loop above P3: registry + discrete bit-collision resolver + online loop | CLOSED GREEN | 256-bit LSH hash, integer Hamming TAU_BITS=168, r=256 (reduction-order-immune). G-MEMO-{NULL,CUE,LOOP} GREEN on 12B: matched +0.000% / corrupted +40106% safety valve. Code: `tools/curator/{build_registry.py,discrete_resolve.py,resolve_cue.py,curator_loop.py}`. Receipts: `tests/fixtures/xbar_c2/` |
| **#222 — `gemma4_kv_replay`** | SP_REPLAY ported into persistent gemma4_kv_* ABI + O(1) bit-exact rewind | CLOSED GREEN | G-222 GREEN on E2B + 12B: rewind resets [0,anchor) diffs=0. G-222-WRAP GREEN (SWA-ring KAI-1c journal). Harness mode `SP_G4_KV_REPLAY_GATE` in `tests/test_gemma4_cuda.c`. Seam in `src/backends/cuda/cuda_forward.cu` |
| **Ring-3 Path A** | VSA/HRR gist consolidation, parameter-free (R3.1→R3.4) | CLOSED GREEN | R3.1 BIND: recall@1=1.0 to N=32 @D=1024. R3.2 LOSS: hit 0.000% / miss +8.04% gate-caught; budget ≤32/vector. R3.3 DUALROUTE: retrieve-and-verify pipe. R3.4 NIGHTSHIFT: idle consolidation 349.8MB resident→16.3KB index. Code: `tools/ring3/{g_r3_bind.py,g_r3_dualroute.py,g_r3_nightshift.py}` + `_run_g_r3_loss.bat`. Receipts: `tests/fixtures/xbar_r3/`. **Note:** real-domain VSA is host-numpy; Z_q/NTT engine port (using math-core `core/ntt_crt` + `core/poly_ring`) is the deferred deployment follow-on |
| **G-XBAR-ORGANISM step 1** | EAR→Ring-2 write seam: audio packet → conditioned cache → serialized episode | GREEN | `SP_G4_KAI3_WRITE` in `tests/test_gemma4_cuda.c`. 12B cache geometry confirmed: global 1×512, SWA 8×256=2048 (jagged), episode clamps to global 512. ep_audio serialized in canonical uniform-512 [NL,P,512] format. Receipts: `tests/fixtures/xbar_organism/`. Engine commit `6600cf4`. Full organism loop is follow-on |
| **O_K unification (2026-06-18)** | XBAR memory re-carried onto the exact-integer O_K substrate (`Q(√-163)` / dual-prime negacyclic CRT-NTT) | GREEN | **G-R3-BIND-on-O_K** (`0019b86`): Ring-3 VSA bind on native `sp_pr_mul`/`ntt`/`sp_pr_score_kstore` **256/256 bit-identical**, ±1 carrier int==float, reduction-order-immune (M byte-identical vs float 4.44e-15). **G-R3-ORGANISM-NATIVE** (`1f0f6be`): dualroute+nightshift on native bind (D=1024; CAP=32). `tools/ring3/{ok_bind.py,g_r3_bind_ok.py}`. Receipts `tests/fixtures/xbar_r3/` |
| **G-R2-FROB** | Frobenius π^k INTEGER Ring-2 episode store (rank-2 O_K lattice) | GREEN | `dbe4103`/`d076797`: a16 16-bit ~lossless / a8b4 12-bit / a16b8 24-bit sub-ULP relL2 1.2e-7 @ 0.76× store. `tools/curator/frob_episode.py`. HONEST: "lossless" = reconstruction fidelity; the n=42 PPL gate is blind below ~1% (no fake +0.000%) |
| **G-XBAR-ORGANISM-FULL** | Full real-episode loop on the discrete container | GREEN | `15e7051`: continuous audio (EAR) → C2 256-bit sig → native integer Ring-3 superposition (+text decoys) → audio-cue top-1 → C2 Hamming verify (accept audio / reject text) → Frobenius integer store → continuous float into the 12B resident cache. checks=5 fails=0. `tools/ring3/g_xbar_organism_full.py` |
| **G-PERIOD6-REBASE** | C2/Ring-3 content-hash period 8→6 (true gemma4 global layers {5,11,…,47}) | GREEN | `d2d7ceb`: re-gated GREEN on the period-6 rebase. Closes the standing PERIOD=8 caveat |
| **Boundary-thesis negatives (kept)** | structure-on-*content* levers, measured-inert | NEGATIVE (honest) | Leg B split-prime O_K Dirichlet carriers (`d7d96fe`, operationally inert); G-R3-MOBIUS (`1e70763`, 1.000→0.969@N=32); G-R2-FROB-ENTROPY (`e6d17bb`, 1.02× dead weight); G-T2-WEIGHTS (`ac76c8e`, T2-Möbius on the real 12B embedding recon cos 0.032 == random 0.039). O_K wins on EXACT ARITHMETIC (the container), not as content structure |
| **Gemma4 + OK_Q4B (CITABLE)** | Gemma-4-12B GPU decode + sovereign pipeline (06-R10, §5.2.1d) | CLOSED | **26.1 tok/s @ wikitext PPL 5.12** on the RTX 2060 12 GB (24/24 gates; CUDA graph EXACT 256/256; dp4a top-1 256/256). Triple-agree 5.1259/5.1259/5.1160. llama.cpp 31.29 tok/s but on artifacts at PPL 192–506 (both halves stated) |
| | gemma4 tokenizer dispatch (#115) | CLOSED | `GEMMA4_BPE` family dispatch **5432/5432 HF-parity exact, both lanes**; installed 12B re-paired (`T_G4_TOK_12B_PAIRED`) |
| **Two-ring memory + WIRE** | Two-ring memory (PPT-ARM, CPU backend) | shipped + measured | **910× resident KV shrink @32k** (7.5 GB → 8.3 MB); needle off NVMe @ **7.57 µs/read**; 8× @ +0.69% PPL; bit-exact when off. Honest negative kept on the board: the C2.4 32k NIAH finale was a **MISS** (64× budget; Ring 3 is the designed fix) |
| | WIRE-CPU integer pipe (CPU backend) | shipped + measured | Qwen3-0.6B **0.84 → 39.52 tok/s (47×)**, ~1.34× behind llama.cpp Q8_0 (memory layout, not ALU) |

History and the chronological dated-update log move to [§2.2](#22-dated-update-history-consolidated). The daemon-wiring snapshot table (Built / Wired) follows below.

### Backend + daemon wiring snapshot

**Update 2026-06-06.** Since the 06-03 snapshot below: the engine drives the canonical math-core decode at engine speed via the `cpu_overlay.c` dispatch seam (the duplicate decode was deleted); AVX2 `sp_pr_resdot` + `sp_ntt_fwd_batch` (lanes=heads) + AVX512-VPOPCNTDQ `sp_arm_scan_sig` overrides; the dual-size + **split-device** Optane Ring-2 store (`ring2_arm_backend.c`, `SP_RING2_OPTANE_DIR_V`) with `read_batch2` concurrent dual-queue fetch and a bounded LRU temporal staging cache (`SP_RING2_CACHE_MB`); the QUIC Ring-2 peer + two-process showpiece (`sp_ring2_showpiece`). **CUDA backend (RTX 2060 sm_75): gated on real silicon** — prefill `qwen3_forward_cuda` f32+Q8 argmax-exact, and a NEW autoregressive **`qwen3_decode_cuda`** (KV resident in VRAM, device argmax; gate `M_QWEN3_DECODE_CUDA`) generating at 6.93→11.97 tok/s (Q8). Detail in the lattice `papers/PPT-LAT-Roadmap.md` §21 + `SESSION-CLOSED-stage-beta-s0.md`.

Snapshot 2026-06-03. **Built** means the artefact compiles and
passes its own gates. **Wired** means the daemon routes inference
through it at runtime.

| Component | Built | Wired in `sp_daemon` | Notes |
|-----------|:-----:|:--------------------:|-------|
| **Two-ring memory (recall router + Optane Ring-2 + fusion)** | yes | yes (`SP_RECALL_*` + `SP_RING2_*`) | 910× resident KV shrink @32k; needle off NVMe @7.57 µs/read; 8×@+0.69% PPL; bit-exact when off; fusion verified N=512 + timed N=8192. Honest negative: the C2.4 32k NIAH finale = **MISS** (64× selection budget; ladder 2k HIT / 4k −1 digit / 8k HIT) — Ring 3 is the designed fix |
| **Reducing loader** (`sp_transcode` GGUF → ~50%-smaller `.sp-model`) | yes | yes | bit-faithful forward, 6/6 E_FMT gates (gemma-3 + Qwen3) |
| Model coverage | yes | yes | Qwen3-0.6B, Qwen2.5-Coder-0.5B, Gemma3-1B, Gemma4, Qwen3.6-35B-A3B MoE (Gated DeltaNet) — byte-exact forward |
| Math-core reference forward | yes | **yes (default)** | byte-exact on host + aarch64-android; baseline tok/s in §4 below |
| CPU backend (AVX-512 + cpu_overlay) | yes | **yes (`SP_DAEMON_BACKEND=cpu`)** | sprint WIRE-CPU (2026-06-02): daemon registers `qwen3_forward_cpu` / `gemma3_forward_cpu` via L1 ABI §6 hook; `cpu_forward_count` increments per prefill; bit-exact vs reference (same byte stream); on i9-11900KB host wall-clock matches reference within ±1% (AVX-512 primitives present in lib, hot-path wiring is WIRE-CPU-V2 follow-on) |
| CUDA backend (PTX MMA + NTT) | yes | no | desktop target; symmetric WIRE-HEX sprint |
| CUDA GPU decode (Stage Beta) | yes | n/a (test path) | **RTX 2060 sm_75:** `qwen3_decode_cuda` (KV in VRAM) + CUDA graphs + fused dp4a INT8/Q4 GEMV (per-tensor precision). Gated 28/28 top-1-lossless. Isolated bandwidth ladder f32 1× → int8 ~3.8× → **Q4 ~7×**. See §5.2.1 |
| Vulkan backend | yes | no | desktop target; symmetric WIRE-HEX sprint |
| Hexagon HVX backend | yes | **partially** | sprint WIRE-HEX: daemon registers `gemma3_forward_hexagon` via L1 ABI §6 hook; `hex_forward_count` increments at first prefill; **`sp_hex_forward` returns non-zero on cDSP after weight upload — cDSP skel on device needs rebuild against current IDL** |
| Polynomial-ring NTT attention (host) | yes | yes (`SP_ENGINE_NTT_ATTN=1`) | byte-exact vs scalar |
| Polynomial-ring NTT attention (Hexagon) | yes | yes (`SP_ENGINE_NTT_ATTN_HEX=1` + Memory model) | sprint NTT.5b/5c; dispatch counter `ntt_hex_forward_count` |
| Spinor-block KV cache | yes | yes (`SP_KV_SPINOR=1`) | persistent compressed KV |
| `/v1/chat` SSE endpoint | yes | yes | greedy argmax decode; stop-string matching |
| `/v1/dialogue` (M.2 MeMo: Grounding → Entity ID → Synthesis) | yes | yes | requires `--memo-model`; returns 3 base64 SpinorReceipts |
| PoUW ledger autowire | yes | yes | `--pouw-ledger-path` enables auto-append of receipts |
| QUIC mesh (`/v1/mesh/peers`, `run_garner_loop`) | yes | yes (host) | android serves empty peer_map |
| FastRPC cDSP echo (`/v1/dsp/echo`) | yes | yes (android) | 8 MB max payload |
| FastRPC cDSP model info (`/v1/dsp/model_info`) | yes | yes (android) | persistent DSP-resident model |

** tok/s baseline (S22U, math-core reference forward, ctx = 16 prefill + 32 decode):**
**These are numbers run during testing of an individual piece of the system on the phone

| Model | Wall (s) | Tokens | tok/s |
|-------|---------:|-------:|------:|
| Gemma3-1B | 18.06 | 16 | 0.89 | 
| Qwen3-0.6B | 11.21 | 16 | 1.43 | 

**Desktop CPU — the WIRE-CPU production path** (i9-11900KB, Qwen3-0.6B, ctx = 4 + 32 decode). This *is* the daemon's accelerated path (`SP_DAEMON_BACKEND=cpu`, Q8 arena + OpenMP-threaded matmul + AVX2 int8×f32 dot):

| Model | path | Wall (s) | Tokens | tok/s |
|-------|------|---------:|-------:|------:|
| Qwen3-0.6B | f16 reference (as-is) | 38.1 | 32 | 0.84 |
| Qwen3-0.6B | **Q8 + threaded + AVX2 (WIRE-CPU)** | 0.81 | 32 | **39.52** |

**47× over the f16 baseline; ~1.34× behind llama.cpp Q8_0 (52.8 tok/s)** on the same host — the remaining gap is **memory layout, not ALU** (VNNI tested + falsified). Full ladder: `shannon-prime-lattice/papers/CONTRACT-SPEED-wire-tok-s.md`; next step `…/PLAN-SPEED-WIRE-CPU-V3-memory-layout.md`. The two-ring memory envelope (910× resident @32k, 7.57 µs/read off Optane) is realized on this same backend (§1).

The HVX backend wiring is in place daemon-side (LLVM-nm on the
android binary shows `gemma3_forward_hexagon` + `sp_hex_forward` +
`sp_wire_hex_forward_dispatch` + `sp_session_register_forward_backend`
at concrete addresses; `/v1/debug/backend_counts` `wire_hex_active = true`
after startup and `hex_forward_count` increments on first prefill).
The remaining work to flip the headline tok/s number is **out of scope
for the wiring sprint** — the on-device `libsp_hex_skel.so` needs to be
rebuilt with `tools/sp_compute_skel/inc/sp_hex.idl` (a different IDL
from `sp_compute.idl`) and pushed to `/data/local/tmp/sp22u/`. Full
detail: `tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX.md`.

---

### 2.1 Harness modes — the env-var test gates

The KAIROS time-axis and the C-c NIAH retention gates ship as **env-var-dispatched
modes inside one CUDA test binary** (`tests/test_gemma4_cuda.c`) — each is byte-inert
when its env is unset, and runs the real 12B (gemma4-12b-b1) on the persistent-KV ABI
in `src/backends/cuda/cuda_forward.cu`. Set the env, run the binary; the mode prints
its receipt and exits. The daemon-side scheduler (`tools/sp_daemon/src/kairos.rs`,
`kairos_runner.rs`) is a separate, off-by-default `kairos` cargo feature — see
**[`tools/sp_daemon/docs/KAIROS-API.md`](tools/sp_daemon/docs/KAIROS-API.md)** for the
kernel-not-harness design, the `SessionHandoff` coordinate ABI, and the null-floor
invariant.

| Env var | Gate | What it proves |
|---------|------|----------------|
| `SP_G4_NIAH` | C-c NIAH retention | A needle planted in a 16k haystack (forced outside the SWA window, so it can *only* cross the global crossbar) survives the slab/ring/poison compaction at depths 10/50/90% — learned-router only; frozen ±1 control MISSes |
| `SP_G4_KAIROS` | Cognitive crucible (prefix-grow) | The 12B holds `NO_OP` silence on idle ticks and acts coherently on salient events, over the deterministic event tape — measured before the metal eviction lands |
| `SP_G4_KAIROS_METAL` | Semantic loop on the journaled ring | The same crucible wired onto the journaled-ring ABI: `NO_OP ⇒ rewind` to the committed anchor (cold-evict the tick), `ACTION ⇒ commit` — perfect 24-tick run, every idle revert clean |
| `SP_G4_KAIROS_SOAK` | G-KAIROS-1 (endurance) | The deterministic tape looped under in-process tripwires (CUDA error / false-action / pos-violation / 3-consecutive malformed / 5-consecutive latency spikes / VRAM leak / thermal). **6h soak GREEN** — `SOAK_EXIT=0`, 351 loops / ~8,400 ticks / 6h01m unattended on the dedicated 2060, 0 false / 0 missed / 0 malformed / 0 pos-violation (the formal ≥24h gate is un-pursued by operator choice, not failed) |
| `SP_G4_KV_REWIND` | G-1b-REWIND-NULL | `gemma4_kv_rewind(Δ)` produces a cache **byte-identical** (48 owner layers, diffs=0) to one that never ran the idle tick, and re-running the tick reproduces identical tokens (perfect inverse, full cache) |
| `SP_G4_KV_WRAP` | G-1b-WRAP-NULL | On the *space-optimized ring* a wrap-crossing idle tick aliases live-window slots; the undo-journal restores them — clobbered = 40 SWA layers (non-vacuity), post-rewind diffs=0 + identical tokens |
| `SP_G4_KV_TELEMETRY` | KAI-1b §5.4 O(actions)→O(1) | Sweep retained-actions A∈{1,2,4,8,16}, time an idle tick under each: metal slope 0.0073 vs prefix-grow 0.924 s/action (127× shallower) — the flatline *is* the O(1) claim, measured |
| `SP_G4_KV_RING_TEL` | KAI-1c journaled-ring tax | The same A-sweep through the journal path: ring slope 0.00365 ≈ full-cache 0.00371 — the undo-journal adds no asymptotic cost |
| `SP_G4_INJ_SEQ` | G-KAIROS-3-NULL (KAI-3) | `gemma4_kv_inject_seq` injects a SEQUENCE of N projected frames and is **2/2 byte-identical** to the inline EMB inject loop — the sequence wrapper is a strict null over the frozen `gemma4_kv_inject` delivery seam |
| `SP_G4_TOK_DUMP` | KAI-3 local tokenizer dump | The engine owns the gemma-4 tokenizer to dump token ids for projector training — keeps the KAI-3 audio-port run **local / no-cloud** |
| `SP_G4_KAI3` | G-KAIROS-3 (KAI-3 metal gate) | A sequence of projected frames (per-position MLP 640→V_sub + on-manifold binder, `tools/audio_port/`) drives **8/8 SEMANTIC pivots** on the 12B (`KAI3_GATE_EXIT=0`); real-token V_sub=60 top1 0.931 / cos 0.9937. The bridge into the GNA audio "EAR" line |

> **Measurement note (this card):** the RTX 2060 cannot lock its memory clock
> (`nvidia-smi`: "not supported"), so bandwidth-bound decode jitters ±~12%. The
> O(1) *slopes* above are measured within one leg (drift-robust); never difference
> two sequential wall-clock series for sub-10% deltas on this host.

---

### 2.2 Dated-update history (consolidated)

Chronological log of when each capability closed. The current-state tables in §2 / §2.1
are authoritative; these entries are the audit trail.

**2026-06-18 — the BYTE-EXACT FORWARD is COMPLETE + gated on the real Gemma-4-12B.** Byte-exact = **exact-integer arithmetic / cross-machine determinism** (the AUDITABILITY mission) — explicitly **NOT** compression (the incoherence-rotation / column-reorder compression levers were convicted as redundant against the existing per-32-block OK_Q4B at gold PPL 4.6665). The dual-prime LINEAR algebra was **already** bit-exact-gated in the universal L2 Rust crate `tools/sp_dsp_smoke` (dual-prime Barrett, mod-q matmul, Garner CRT inv=894602413, NTT ladder; primes q1=1073738753 q2=1073732609, M=q1·q2≈2^60 fits u64 → no `__int128`) — the session re-derived that (a recurrence of "verify against the substrate," banked as a lesson) and added the genuinely-new piece: the **4 nonlinear fp32 islands** (RMSNorm / softmax / GELU / RoPE) as exact-integer references in `tools/sp_dsp_smoke/src/sp_islands_q_ref.rs` (G-ISLANDS-Q-REF GREEN host: RMS 5.8e-6 / softmax 1.3e-6 / GELU 2.8e-6 / RoPE 9.2e-6 fidelity vs float, order-immune; RoPE via deterministic fixed-point **CORDIC** = no libm) + math-core `core/exact_islands/` (gate `T_EXACT_ISLANDS`). The gemma4 CUDA forward now converts all 4 islands + attention to **exact-integer device kernels behind `SP_BYTEEXACT`** (default-off = byte-identical null floor) — device techniques avoid `__int128`: `__umul64hi` for wide products, a 64-bit isqrt split for RMS, device CORDIC for RoPE, `bx_garner`/`bx_exp_fixed` dual-prime, attention `k_attn_decode_win_bx` (negacyclic dual-prime dot). The **universal daemon drives the 12B** (`tools/sp_daemon`, feature `wire_cuda_backend`): prefill via `sp_session_register_forward_backend` (**G-WIRE-CUDA-GEMMA4** GREEN) and token-by-token DECODE via the NEW L1 verb `sp_session_register_kvdecode_backend` + additive `gemma4_kv_decode_logits` (**G-WIRE-CUDA-DECODE-GEMMA4** GREEN: 32/32 tokens bit-identical to the oracle, VRAM flat O(1)). **Gates (real 12B, gemma4-12b-b1.sp-model):** **G-BYTEEXACT-ISLANDS-CUDA** GREEN (on-model island fidelity RMS 3.8e-5 / GELU 8.2e-7 / RoPE 9.6e-6); **G-BYTEEXACT-FORWARD-12B GREEN — Leg A `SP_BYTEEXACT` off = PPL 4.6665 == baseline byte-identical (null floor), Leg B on = 4.6569 parity, run-to-run BIT-IDENTICAL** (determinism = the cross-machine proxy). Commits: engine `69c0588` (lineage 9c2aad3→…→eee3aac→6b9a786→69c0588), math-core submodule `d9d96f3` (adds the L1 kvdecode verb in `include/sp/sp_l1.h` §6b + `core/exact_islands/`); receipts `tests/fixtures/xbar_r3/`. **HONEST CAVEATS:** (1) the ONE remaining item is **EXTERNAL** — a true bit-identical logit check across two PHYSICAL GPUs (needs a 2nd machine; on-machine we have run-to-run determinism + reduction-order immunity as the proxy); (2) PPL parity measured at **n=42** (small-N; the −0.21% deflection is within noise); (3) the **boundary thesis holds** — O_K wins on EXACT ARITHMETIC (the container); structure-on-content compression is measured-inert (honest negatives). The one-shot `gemma4_decode_cuda` stays byte-untouched (null floor); all new work is behind `SP_BYTEEXACT` / the daemon wire feature.

**2026-06-18 — XBAR memory UNIFIED onto the exact-integer O_K substrate; the organism breathes end-to-end on the discrete container.** Everything below is on `Q(√-163)` carried by the math-core dual-prime negacyclic CRT-NTT (`core/ntt_crt` + `core/poly_ring`; frozen primes q1=1073738753 q2=1073732609 M=1152908312643096577), replacing the generic float carriers. **G-R3-BIND-on-O_K Leg A** (engine `0019b86`): the Ring-3 VSA bind re-carried onto native `sp_pr_mul`/`ntt`/`sp_pr_score_kstore` is **256/256 bit-identical** to the native path, ±1 carrier int==float recall, and **reduction-order-immune** (M byte-identical across permutations vs float 4.44e-15 drift); `tools/ring3/g_r3_bind_ok.py`. **G-R3-ORGANISM-NATIVE** (`1f0f6be`): the live dualroute + nightshift loop, ripped off float-FFT, now runs on native `sp_pr_mul` via `tools/ring3/ok_bind.py` (D=1024 = two 512-blocks; CAP=32 preserved). **G-R2-FROB** (`dbe4103`/`d076797`): a Frobenius π^k **INTEGER** Ring-2 episode store — rank-2 O_K lattice (coarse a + residual b, real scales: a16 16-bit ~lossless / a8b4 12-bit / a16b8 24-bit sub-ULP relL2 1.2e-7 at 0.76× store); `tools/curator/frob_episode.py`. Honest: "lossless" is by reconstruction fidelity — the n=42 PPL gate is blind below ~1% (no fake +0.000%). **G-XBAR-ORGANISM-FULL** (`15e7051`): the full loop on **real episodes** — continuous audio (EAR) → C2 256-bit sig → native integer Ring-3 superposition (with text decoys) → audio-cue retrieve top-1 → C2 Hamming verify (accept audio / reject text) → Frobenius integer store → continuous float lands clean into the 12B resident cache (checks=5 fails=0); `tools/ring3/g_xbar_organism_full.py`. **G-PERIOD6-REBASE** (`d2d7ceb`): the C2/Ring-3 content-hash period 8→6 to the true gemma4 global layers {5,11,…,47}, re-gated GREEN. **Boundary thesis** (the receipts): O_K wins on **exact arithmetic** (the container); every structure-on-*content* lever is measured-inert and kept as an honest negative — Leg B split-prime O_K Dirichlet carriers (`d7d96fe`, operationally inert; periodic-carrier spiky spectrum), G-R3-MOBIUS (`1e70763`, Möbius square-free on dense holographic M sheds memories 1.000→0.969@N=32), G-R2-FROB-ENTROPY (`e6d17bb`, entropy-coding the codes = 1.02× dead weight — the lever is bit-width), G-T2-WEIGHTS (`ac76c8e`, T2-Möbius fails on the real gemma-4-12b embedding, recon cos 0.032 == random 0.039; T2 was a design proposal, never validated unlike T4). Receipts `tests/fixtures/xbar_r3/` + `tests/fixtures/xbar_organism/`; the one-shot `gemma4_decode_cuda` stays byte-untouched (null floor). NEXT = T4 Frobenius π^k of the 9.4GB model WEIGHTS (the validated, untouched lever), then KAIROS post-organism.

**2026-06-17 — C2 Memo curator + #222 + Ring-3 Path A + EAR→Ring-2 organism bridge all CLOSED.** Everything below is on the 12B/E2B, parameter-free (no training budget), all env-gated on the null floor. **C2 Memo curator** (autonomous Ring-2 recall loop above P3): discrete bit-collision resolver (256-bit LSH hash, integer Hamming radius TAU_BITS=168, r=256; reduction-order-immune; Gemini's "dot IS Hamming" reframe was false on the real centroids — an r-sweep closed the win instead) + online loop G-MEMO-{NULL,CUE,LOOP} GREEN on 12B (matched +0.000% / corrupted +40106% safety valve). Code: `tools/curator/{build_registry.py,discrete_resolve.py,resolve_cue.py,curator_loop.py}`, receipts `tests/fixtures/xbar_c2/`. **#222** (`gemma4_kv_replay`): SP_REPLAY ported into the persistent `gemma4_kv_*` ABI — O(1) bit-exact rewind via the same [0,anchor) shear; G-222 GREEN on E2B+12B (diffs=0) + G-222-WRAP GREEN (SWA-ring KAI-1c journal). Harness `SP_G4_KV_REPLAY_GATE` in `tests/test_gemma4_cuda.c`. **Ring-3 Path A** (VSA/HRR, parameter-free gist consolidation): R3.1 BIND (recall@1=1.0 to N=32 @D=1024) → R3.2 LOSS (step function: hit 0.000% / miss +8.04% gate-caught, budget ≤32/vector) → R3.3 DUALROUTE (retrieve-and-verify) → R3.4 NIGHTSHIFT (idle consolidation; 349.8MB resident→16.3KB index) all GREEN. Code: `tools/ring3/{g_r3_bind.py,g_r3_dualroute.py,g_r3_nightshift.py}` + `_run_g_r3_loss.bat`, receipts `tests/fixtures/xbar_r3/`. The real-domain VSA is host-numpy; the Z_q/NTT native port (math-core `core/ntt_crt` + `core/poly_ring`) is the deferred deployment follow-on. **G-XBAR-ORGANISM step 1 GREEN** (EAR→Ring-2 write seam): `SP_G4_KAI3_WRITE` injects a real audio packet (KAI-3 path) → conditioned cache npos=114 → serializes ep_audio in canonical uniform-512 [NL,P,512] format (SWA jagged 2048 clamped to global 512). Receipts `tests/fixtures/xbar_organism/`, engine commit `6600cf4`. Sig separates (self 211, margin +79). Full organism loop is the immediate follow-on.

**2026-06-14 — KAIROS time-axis CLOSED (KAI-1 + KAI-1b).** A 12B now runs as a resident background daemon: mathematically silent and O(Δ)-flat until a high-salience event, then acts, and stays stable after acting (perfect 24-tick crucible; public ledger KAIROS-01). **KAI-1b** drops the cold-evict to the metal: persistent-KV `gemma4_kv_open/prefill/decode/rewind/commit/pos/snapshot/close` in `src/backends/cuda/cuda_forward.cu` (`gemma4_decode_cuda` left byte-untouched). `rewind(Δ)` is **bit-exact** (G-1b-REWIND-NULL: 48 owner layers / 16.5 MB / diffs=0 + gen-reproduce) and **O(1)** (idle-tick latency flat — metal slope 0.0073 vs prefix-grow 0.924 s/action, 127× shallower; 16.7× @ 16 retained actions). The wrap-aware ring (KAI-1c) adds an undo-journal so the rewind stays bit-exact on the space-optimized ring (G-1b-WRAP-NULL: 40 SWA layers clobbered then restored, diffs=0). The ≥24h endurance soak harness (`SP_G4_KAIROS_SOAK`) is built and **running — no verdict yet**. Lattice CONTRACT-KAIROS-K0-K1 §5; receipts `results/kai1b_*.log`.

**2026-06-13 — XBAR §P3.2-b-2b CLOSED: the KV cache is decoupled from context on the real 12B.** The global sparse-recall router landed end-to-end in `cuda_forward.cu` (all flag-gated, byte-inert off): per-step shadow-select on the 8 global owners → `k_attn_decode_gather` index-list attention → Ring-2 spill/poison/page. The v0 frozen ±1 projection router fails 8× global compression at +4.17% PPL (larger-N G2, wikitext-2 N=2048×3); the **on-engine oracle (exact top-B by q·K) proved 8× is learnable at −0.08%** (`SP_ARM_ORACLE`); a trained **512×32 Learned-LSH projection** (`SP_ARM_LSH=M.bin`, M=R·Rᵀ, select = top-B by (Mq)·K — zero new hot-path kernels, cost independent of r) **wins 8× at +0.47% PPL** (engine `222463a`; weight `tests/fixtures/lsh/lsh_M_r32.bin`; trainer `tools/xbar_lsh/train_lsh.py`). Combined with the b-2a SWA ring shrink (40/48 layers), both KV terms are constant in context — the SWA at `W`, the globals at the GQA union `nh·B` (= 16·256 = 4096, NOT `B`: C-b.2 measured the per-step union at 1511/2048, because the 16 query heads pick near-orthogonal top-B sets on gemma's diffuse globals). **Phase C alloc-shrink CLOSED GREEN (2026-06-14):** `SP_ARM_DEVSEL` device-select (`7195100`) + `SP_ARM_LSH_R` sidecar (`7cd7482`) + `SP_ARM_SLAB` compact slab (`33ac632`; full K/V in host Ring-2, per-step union paged into a `nh·B`-capped slab). **O(1) realized** — N=8192 vs N=16384 `nvidia-smi` flat within ~50 MiB (cache alloc byte-identical; a full cache adds ~5.4 GiB) — and **C-c NIAH retention** (`test_gemma4_cuda` SP_G4_NIAH mode, engine `3218d73`): the needle survives the compaction at depths 10%/50%/90% (exact, learned-router-only; frozen ±1 control MISSes) under SWA-isolation + slab compaction. **Select → realize → retain, end-to-end on the real 12B.** (Free-decode needs the tied-head int8 path `SP_CUDA_DECODE_INT8=1`; full-attention baseline at 16k is physically impossible on the 2060 — ctx-softmax shared-mem >64 KB + cache OOM — which is the motivation.) Full record: lattice `CONTRACT-XBAR-P3` + `PPT-LAT-STATE` §5.14.

**2026-06-10 — gemma4 tokenizer dispatch (#115) CLOSED; 12B text-in LIVE.** New tokenizer module `src/tokenizer/gemma4_bpe.c` + family dispatch (`GEMMA4_BPE` family tag written into `.sp-tokenizer` by `sp_transcode`; unknown family = hard error). Gates `T_G4_TOK_PARITY` + `T_G4_TOK_ROUNDTRIP`: **5432/5432 HF-parity exact, both lanes** (GGUF lane + `.sp-tokenizer` blob lane; engine `3253a82`, core `9d3cc72`). Deployment (engine `d8ba947`): the installed 12B blobs were regenerated via `sp_transcode --tok-only` and each paired `.sp-model` header SHA re-paired the way `sp_transcode` pairs at creation; new gate `T_G4_TOK_12B_PAIRED` (proven sensitive — a legacy type_id=2 blob fails it 0/5432); B1 GPU decode smoke 6/6 on the 2060. Also: the E_CPU_9 byte-identity lanes now pin `SP_CPU_SCALAR=1` (the common AVX2 dot kernel reassociates; engine `5cd5870`), and the submodule carries core `64b698c` — the **`sp_arm_*_geom` per-layer-class router API** (`T_ARM_GEOM` 26/26), the G-P3-GEOM substrate for the gemma4 ring port.

**2026-06-08 — the gemma-4 campaign closed; the sovereign quantization pipeline ships here.** `sp_transcode` gained **Safetensors Direct** (`--st <model.safetensors>`: weight VALUES from the official checkpoint; GGUF supplies verified-clean metadata/tokenizer only; mapped-but-missing = hard error) and the **OK_Q4B** codec (`--q4b` / `--q4b-ffn` recipe B1: per-32-block f16 scales, store-then-derive). The CUDA backend gained `k_gemv_q4b_dp4a_v2` (per-block scale inside the dp4a chunk loop) + `k_dequant_arena_q4b` + `DevTensor.bscale` routing; the core arena moved to layout v2 (formal migration). Result, gated 24/24 on the RTX 2060 12GB: **Gemma-4-12B at 26.1 tok/s and wikitext PPL 5.12** (GPU PPL gate 5.1160 vs the hand-written gold reference 4.6776; sim/CPU/GPU triple-agreement at 5.1259/5.1259/5.1160). Context: every gemma-4 GGUF measurable in June 2026 carries broken weights (192–506 by engine-independent measurement) — see the public repo's `GEMMA4-QUANT-FIX.md`. The earlier 34.2 tok/s headline is retired (its artifact failed the PPL gate).

---

## 3. Quick start

### 3.1 Build the daemon (host, Windows)

**`docs/BUILD-ENV.md` is the authoritative build doc** (pinned toolchains;
do not contradict it). Summary: canonical CPU build = **MinGW gcc 15.2**
in `build/` (Ninja); **MSVC cannot build the CPU tree**; CUDA host =
VS2019 BuildTools + CUDA 12.4 in `build-cuda/`.

```cmd
:: One-time: set up VS 2019 BT + CUDA env (if you want CUDA backend)
call scripts\env\env-cuda.bat

:: Build math-core + engine libs + sp_daemon (CPU + Vulkan default)
scripts\build\build-cpu.bat
scripts\build\build-vulkan.bat

:: Build sp_daemon with Cargo
cd tools\sp_daemon
cargo build --release --bin sp-daemon
```

Linux equivalents are documented in `docs/BUILD-ENV.md`.

### 3.2 Transcode a GGUF model to `.sp-model`

```cmd
build\tools\sp_transcode\sp_transcode.exe ^
    path\to\model.gguf ^
    out\model.sp-model ^
    out\model.sp-tokenizer ^
    --verify
```

`--verify` runs a round-trip dequant check (rms / max error) and rejects
the output if a Q8 row's relative error exceeds the threshold.

### 3.3 Start the daemon

```cmd
target\release\sp-daemon.exe start ^
    --model out\model.sp-model ^
    --tokenizer out\model.sp-tokenizer ^
    --port 8080
```

Daemon detaches and writes the PID to `%TEMP%\sp-daemon.pid`. Stop via
`sp-daemon stop`.

### 3.4 First chat request

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, what is 2+2?", "max_tokens": 32}'
```

The response is a `text/event-stream` (Server-Sent Events). Each SSE
event carries a JSON `{"delta": "...token text...", "chat_id": <u64>}`.
Stream ends with `data: [DONE]`.

### 3.5 First dialogue request (dual-model MeMo path)

Requires the daemon to be started with both a target (Executive) model
and a Memory model:

```cmd
sp-daemon start ^
    --model out\executive.sp-model ^
    --tokenizer out\executive.sp-tokenizer ^
    --memo-model out\memory.sp-model ^
    --memo-tokenizer out\memory.sp-tokenizer ^
    --pouw-ledger-path C:\sp\ledger.bin ^
    --port 8080
```

```bash
curl -s -X POST http://127.0.0.1:8080/v1/dialogue \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Who painted the Sistine Chapel?"}' | jq
```

Response shape:

```json
{
  "response": "Michelangelo painted the Sistine Chapel ceiling.",
  "receipts": [
    "<base64 of 64-byte SpinorReceipt, turn 1 Grounding>",
    "<base64 of 64-byte SpinorReceipt, turn 2 Entity ID>",
    "<base64 of 64-byte SpinorReceipt, turn 3 Synthesis>"
  ],
  "wall_ms": 412,
  "turn_us": [128000, 184000, 100000]
}
```

The three SpinorReceipts are appended to the PoUW ledger if
`--pouw-ledger-path` was set.

---

## 4. Architecture

```
                                          ┌─────────────┐
                                          │  Browser /  │
                                          │  TUI / CLI  │
                                          │  curl, etc. │
                                          └──────┬──────┘
                                                 │ HTTP/JSON
                                                 │ SSE + WebSocket
                                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ sp_daemon  (axum + tokio, Rust)                                           │
│                                                                           │
│  routes.rs ─┬─ /v1/chat (SSE)         ─┬─ session::SpSession (Mutex)     │
│              ├─ /v1/dialogue (JSON)    ─┤                                 │
│              ├─ /v1/events (SSE)       ─┤  ┌─────────────────────────┐   │
│              ├─ /v1/metrics (JSON)     ─┘  │ dialogue_runner.rs      │   │
│              ├─ /v1/mesh/peers (JSON)      │ Grounding→EntityID→Synth│   │
│              ├─ /v1/receipts (JSON)        └─────────────────────────┘   │
│              ├─ /v1/pouw/ledger (SSE)                                    │
│              ├─ /v1/node/telemetry (WS)                                  │
│              ├─ /v1/abort/:id (POST)                                     │
│              ├─ /v1/dsp/echo (POST, android)                             │
│              ├─ /v1/dsp/model_info (GET, android)                        │
│              └─ /v1/debug/backend_counts (JSON)                          │
│                                                                           │
│  pouw_ledger.rs  ── Append-only SpinorReceipt ledger; canonical replay   │
│  memo_routing.rs ── KSTE-routed sparse Memory activation (M.5)           │
│  network/quic_shard.rs ── QUIC coordinator + worker + Garner CRT loop    │
│  dsp_rpc.rs (android) ── libcdsprpc.so FastRPC bridge                    │
│  ntt_hex_dispatch.rs (android) ── NTT.5b backend trampoline              │
│  hex_forward_dispatch.rs (android, feature=wire_hex_backend)              │
│        ── Sprint WIRE-HEX full-forward backend dispatcher                │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │
                                  │ frozen L1 C ABI  (sp_l1.h)
                                  │ sp_session_create / sp_prefill_chunk /
                                  │ sp_decode_step / sp_session_clone /
                                  │ sp_session_register_forward_backend
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  libshannonprime  (C — lib/shannon-prime-system/)                         │
│                                                                           │
│  core/session ── sp_session lifecycle + KV cache + clone/rewind          │
│  core/forward ── reference forward (matmul→RMSNorm→RoPE→attn→FFN)        │
│  core/ntt_crt ── dual-prime NTT-CRT primitive (Barrett)                  │
│  core/poly_ring ── R_q polynomial-ring attention                         │
│  core/poly_ring_bluestein ── arbitrary power-of-2 N via chirp-z          │
│  core/frobenius ── Q8/Q4 per-row codec                                   │
│  core/arena ── packed-weight arena                                       │
│  core/vht2 ── Spinor 63-byte block + Möbius reorder + CRC-8              │
│  core/kste ── encoder + Tier-0/Tier-1 dominance                          │
│  core/io_format ── .sp-model mmap loader                                 │
└──────────────────┬────────────────────────────────┬────────────────────┬─┘
                   │  §6 forward-backend hook        │ NTT dispatch      │
                   ▼                                  ▼ hook              │
   ┌─────────────────────────────────┐    ┌──────────────────────────┐   │
   │  Engine backends (libsp_engine) │    │  Hexagon cDSP skel       │   │
   │  src/backends/                  │    │  tools/sp_compute_skel/  │   │
   │   ├ cpu/ (AVX-512 + overlay)    │    │   ├ src_dsp/ (HVX NTT,   │   │
   │   ├ cuda/ (PTX MMA + NTT)       │    │   │   Halide FFN,        │   │
   │   ├ vulkan/ (compute shaders)   │    │   │   VTCM staging)      │   │
   │   └ hexagon/ (sp_hex_host.c)    │───▶│   └ inc/sp_compute.idl   │   │
   │       FastRPC client side        │     (FastRPC server side)        │
   └─────────────────────────────────┘    └──────────────────────────┘   │
                                                                          │
   The 4 backends + the cDSP skel all gate their output against the      │
   math-core scalar reference via T_*_BIT_EXACT tests. Math is in Z_q;   │
   floating point is plumbing.                                           ▼
```

**Two layers the diagram doesn't show (both in the CPU backend):**

- **The two-ring memory (PPT-ARM, the C2.1 headline)** — `src/backends/cpu/cpu_forward.c` + `ring2_disk.c`. A ±1 Rademacher recall router + a `sink + W` **Ring-1** resident window, backed by a **Ring-2** spill to NVMe / Optane (`FILE_FLAG_NO_BUFFERING` + IOCP). At 32k context the resident KV cache is **8.3 MB (910× smaller than 7.5 GB)**, the needle is served back off the physical drive at **7.57 µs/read**, bit-exact when disabled, with a compact-and-spill fusion mode. Env: `SP_RECALL_*` + `SP_RING2_*`.
- **The WIRE-CPU integer pipe** — Q8 packed arena + OpenMP-threaded matmul + AVX2 int8×f32 dot takes Qwen3-0.6B decode **0.84 → 39.52 tok/s (47×)**, ~1.34× behind llama.cpp Q8_0. CUDA / Vulkan / Hexagon are symmetric `WIRE-*` follow-ons.

---

## 5. The backends

Each backend lives under `src/backends/<name>/`; build flags select
which get linked. All four override slices of the math-core reference
forward and gate to it for bit-exactness.

### 5.1 CPU backend — `src/backends/cpu/`

| File | Role |
|------|------|
| `cpu_forward.c` | Whole-forward entry point (`gemma3_forward_cpu`, `qwen3_forward_cpu`) |
| `cpu_overlay.c` | Per-matmul + per-row dequant kernels honouring `SP_ENGINE_FROB`, `SP_CPU_SCALAR`, `SP_ENGINE_F16_ACT`, `SP_Q4_PROMOTE` |
| `cpu_gemma3.c` | Gemma3 arch path (sandwich pre/post norm + GeGLU FFN) |
| `cpu_generate.c` | Standalone CPU generation harness |
| `avx512/` | AVX-512 matmul + dot kernels; sub-phase Phase 2-CPU-AVX |

Build flags:

```bash
cmake -B build-cpu -G Ninja \
      -DSP_ENGINE_BACKEND=cpu \
      -DSP_ENGINE_WITH_AVX2=ON \
      -DSP_ENGINE_WITH_AVX512=ON \
      -DSP_ENGINE_BUILD_TESTS=ON
```

Or use `scripts\build\build-cpu.bat` (canonical CPU dir = `build/`, MinGW
gcc — see `docs/BUILD-ENV.md`). Status: **built + wired** (sprint
WIRE-CPU, 2026-06-02): `SP_DAEMON_BACKEND=cpu` registers
`qwen3_forward_cpu` / `gemma3_forward_cpu` via the L1 ABI §6 hook. This is
the backend carrying the two-ring memory envelope and the WIRE-CPU
integer pipe (§4). Note: the E_CPU_9 byte-identity test lanes pin
`SP_CPU_SCALAR=1` — the shared AVX2 dot kernel reassociates the reduction
(engine `5cd5870`).

### 5.2 CUDA backend — `src/backends/cuda/`

| File | Role |
|------|------|
| `cuda_forward.cu` | Whole-forward entry point + cudaStream lifecycle |
| `cuda_backend.cu` | Backend init, device selection, error mapping (CUDA → `SP_ECUDA`) |
| `ptx_mma.cuh` | Inline PTX `mma.sync` for the matmul tile (Turing sm_75 + Ampere sm_80 + Hopper sm_90) |
| `ptx_mma_tile_{int4,int8}.cuh` + `ptx_mma_tile_common.cuh` | Tiled INT4 / INT8 matmul tiles |
| `ptx_mma_tile_validate.cu`, `ptx_mma_tile_bench.cu` | Bit-exact validators + perf benches |
| `ptx_ntt.cuh` | PTX NTT butterfly. **Forbidden idiom** (memory entry `reference-nvcc-paired-register-bug`): never use `mul.wide.u32` / `mad.wide.u32` — nvcc miscompiles. Decompose to separate `mul.lo.u32` + `mul.hi.u32` + `shf.r/add.cc`. |
| `ptx_spinor.cuh` | PTX Spinor-block decode kernel |
| `ptx_hash.cuh` | PTX hash chain (sm_75 has only one 32-bit ALU dispatch port shared between `lop3.b32` and `xor.b32` — silicon-capped ~1.1× on Turing; 3× reachable on Ampere; memory `reference-turing-alu-scheduler-ceiling`) |
| `ptx_validate.cu` | Cross-backend bit-identity oracle |

Build flags:

```bash
cmake -B build-cuda -G Ninja \
      -DSP_ENGINE_BACKEND=cuda \
      -DSP_ENGINE_WITH_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="75;80;90" \
      -DCMAKE_CUDA_FLAGS="--use-local-env" \
      -DSP_ENGINE_BUILD_TESTS=ON
```

The `--use-local-env` flag is mandatory on VS2019 BuildTools (without a
full VS install nvcc's internal vcvars detection fails). See
`scripts/build/build-cuda.bat`.

#### 5.2.1 Stage Beta — GPU decode + the INT8/Q4 bandwidth ladder (2026-06-06, RTX 2060, sm_75)

The CUDA forward was prefill-only. **Stage Beta** added autoregressive
token-**generation** on the GPU and the discrete-quant bandwidth win, all
gated bit-exact / top-1-lossless on the actual 2060 (12 GB, Turing):

- **GPU autoregressive decode** (`qwen3_decode_cuda`) — KV cache resident in
  VRAM, single-query GQA attention, device-side argmax that writes the next
  token straight into a VRAM-resident `dseq[]` (zero per-step host sync at
  `eos=-1`).
- **CUDA graphs** (`SP_CUDA_DECODE_GRAPH=1`) — the per-token launch sequence is
  captured once and replayed, via **position-indirect kernels** (`k_embed_at`,
  `k_rope_dyn`, `k_kv_store`, `k_attn_decode_dyn`, `k_argmax_at`, `k_incr_pos`)
  that dereference a device-scalar `int *dpos` so graph topology + node params
  stay constant across replays. Warm win ≈ **1.06×** (the headline `12.65×`
  from an early commit was a cold-start measurement artifact — corrected; see
  `CONTRACT-SPEED`).
- **Fused dp4a GEMV** (`SP_CUDA_DECODE_INT8=1`) — reads the packed Q8/Q4 arena
  codes **straight from VRAM** (no f32 scratch), `__dp4a` INT8·INT8→INT32, with
  dynamic per-vector int8 activation quant. `k_gemv_q8_dp4a_v2` /
  `k_gemv_q4_dp4a_v2` are warp-per-row + 128-bit `int4` loads +
  `__shfl_down_sync` reduction. **Per-tensor precision dispatch**
  (`DevTensor.prec`, resolved from the arena's per-row precision) routes each
  matmul to the right kernel — this correctly handles **K-quant mixes**
  (`Q4_K_M` keeps the head/embeddings at Q8 while the body is Q4).
- **The bandwidth ladder** (isolated GEMV sweep, `tests/bench_gemv_int8.cu`,
  both clocks pinned): **f32 1× (~290 GB/s, bus-saturated) → int8 dp4a ~3.8×
  → Q4 dp4a ~7.06×** at 12B-scale dims, hugging the byte ratio (4:1 / 8:1). At
  0.6B / full clock the decode is **overhead-bound** (~91 tok/s, all precisions
  converge); the win binds at large-model scale where weight bandwidth is the
  wall. Q4 correctness vs host reference: max rel err **1.34e-7**.

Gate: `M_QWEN3_DECODE_CUDA` (`tests/test_qwen3_decode_cuda.c`) — f32 / Q8 / Q4 /
`.sp-model` all **256/256 top-1 lossless** (dp4a == dequant graph), graph ==
per-step byte-exact, decode == prefill teacher-forced. **28/28 checks.**

**Methodology note (important for any GPU tok/s number):** absolute decode
tok/s at 0.6B is unreliable without (1) warmup (cold CUDA module load + cuBLAS
JIT ≈ 13× first-call penalty), (2) a long window (`n_gen ≥ 256`), and (3)
**both clocks pinned** — `nvidia-smi -lgc <sm>,<sm>` locks only the SM clock,
but a weight-GEMV is *memory*-bound, so the GDDR6 clock must be at full speed
too (it auto-boosts under sustained load; GeForce `-lmc` is flaky). Trust the
within-run ratio over the absolute.

#### 5.2.1b Stage Eta Phase 1 — the full Gemma4 architecture on CUDA (2026-06-06)

The complete **Gemma 4 (MatFormer E-series)** forward + autoregressive decode
now runs on the GPU, gated **38/38** against the bit-faithful CPU oracle
(`core/forward/gemma4.c`) on the `gemma4-e2b` `.sp-model`:

- **`gemma4_forward_cuda`** — full 35-layer prefill: per-layer GLOBAL/SWA head
  geometry (projection widths change per layer), shared-KV (15 owners / 20
  sharers), proportional `rope_freqs` RoPE on global layers, the weightless
  V-norm, elastic per-layer FFN, the **AltUp** precompute + per-layer injection
  + scalar `out_scale`, tied head + final-logit softcap.
  **Gate: argmax 12/12, max KL `2.663e-10` vs the oracle** — distributional
  identity at machine-noise level across the whole variable-geometry stack.
- **`gemma4_decode_cuda`** — autoregressive greedy decode over a **jagged
  shared-KV cache** (per-owner `[P x kvd_L]` buffers: global 512-wide, SWA
  256-wide, sharers allocate nothing and read their owner), per-step AltUp,
  windowed single-query attention (`k_attn_decode_win`), absolute-position
  proportional RoPE (`k_rope_freqs_at`).
  **Gate: the oracle prefill teacher-forced argmax-predicts EVERY generated
  token** (E_G4_CU_DEC).
- **`gemm_w_lift`** — the oracle ARITHMETIC on cuBLAS: raw integer arena codes
  (exact in f32) into the SGEMM, ONE per-row Frobenius lift after — not
  per-weight dequantization (which injects an extra rounding per term; measured
  2.8e-3 divergence before the fix). This is the §4.8 inline-lift discipline
  enforced on the GPU.
- **`tests/test_gemma4_cuda.c`** — the gate + the permanent **truncated-parity
  bisection harness** (CPU mirror of the oracle through N layers vs
  `gemma4_cuda_probe` stages: embed / attn_norm / q / attention / pre-norm /
  residual), which locked L0 (SWA), L4 (the first global layer) and L15 (the
  first shared-KV sharer) before the live run — which then lit green first try.

#### 5.2.1c Stage Eta ETA.5b — the velocity pass + THE 12B SHOOTOUT (2026-06-07)

**SP 34.2 tok/s vs llama.cpp-CUDA 31.29 ± 0.20 (+9.3%)** — Gemma-4-12B,
RTX 2060, tg256, SM pinned (GeForce `-lmc` unsupported; memory free-ran for
both engines). The SP artifact is a *reducing* 5.56 GB `.sp-model` (source
GGUF 6.62 GB). **Anchor: not citable until the wikitext-PPL gate clears the
Q6_K→Q4 squeeze** (release-blocking for paper 06).

- **E2B ladder (suite 44/44):** oracle lift 10.3 → +graph 10.6 (256/256 EXACT)
  → +dp4a 62.3 (6.05×) → **graph+dp4a 75.7 tok/s (7.35×)**, 256/256 top-1.
  Levers: device-side packed **PLE gather** (`k_ple_gather_at`, TRUE-division
  host-mirror arithmetic — byte-match gated), **packed tied head**
  (`embd_packed`, 1 B/weight on the largest decode matmul), **jagged-topology
  CUDA-graph capture** (per-owner cache pointers fixed per layer; position via
  `*dpos`). Envs: `SP_CUDA_DECODE_GRAPH=1`, `SP_CUDA_DECODE_INT8=1`.
- **The dense 12B is NOT the E-series:** PL=0 (no AltUp/PLE) with
  `layer_output_scale` + `rope_freqs` still present (now keyed on tensor
  PRESENCE); `shared_kv_layers=0`; per-layer `head_count_kv` ARRAY (8 SWA /
  1 global, period 6); **V-less global layers** — `attn_v` absent, V = the RAW
  K projection, weightless-normed, never roped (llama.cpp
  "use_alternative_attention", read reference-first). f32 embd is skipped past
  a 2 GB budget (12B: ~4 GB) → `k_embed_packed_at` gather + dp4a tied head.
- **The L11 kill (per-block activation quant):** per-VECTOR int8 activation
  quant collapsed on the 12B's outlier-heavy activations (L11's TRAINED
  out_scale is 0.005 — the model flags itself): oracle-rank 205596. The LIFT
  discriminator (same path, exact arithmetic → 1.5e-4 floors everywhere)
  proved the structure innocent. Fix: **per-16-block activation scales**
  aligned exactly to the GEMVs' 128-bit loads (one f32 mul per block, zero
  extra bus). Verdict: rank 2 at gap 0.31 — a measured top-2 near-tie. The
  DEC gate prints oracle-rank + logit gap on any flip.
- **Diagnostic toolkit (env-gated, standing equipment):** `SP_G4_FASTPROBE`,
  `SP_G4_DEC_PROBE=<pos>`, `SP_G4_DEC_DUMP=<file>`, `SP_G4_LIFT`,
  `SP_G4_NO_OSCALE` — the five-strike bisection suite that localized the bug.
  Links the CORE inference lane (`sp_session`) + `sp_engine_cuda` in one binary
  (one `as_f32 -> sp_as_f32` alias shim; everything else cross-resolves).

**RESOLUTION (2026-06-08): the PPL gate stayed RED on this artifact — the
per-row Q4 squeeze of Q6_K source tensors fails wikitext PPL, so the 34.2
tok/s is formally RETIRED** (the series' own anchor rule caught it). The
citable successor is §5.2.1d.

#### 5.2.1d THE GEMMA-4 CAMPAIGN — sovereign pipeline + OK_Q4B + SHOOTOUT-2 (2026-06-08, CITABLE — ledger 06-R10)

Full record: lattice `CONTRACT-SPEED` (GOLD INSTRUMENT → RESOLUTION → Q4B
SPEC → CLOSED GREEN); public ledger rows 06-R8/R9/R10 + `GEMMA4-QUANT-FIX.md`.

- **The gold instrument:** a from-scratch reference forward off the official
  safetensors measured gemma-4-12B's TRUE wikitext PPL at **4.6776**.
  llama.cpp's 397–506 was the ARTIFACTS, not the engine — the same
  arithmetic over GGUF-dequantized tensors reproduces the breakage
  (post-June-5 rebuilt GGUF still 192.9). **The GGUF weight lane is DEAD
  for this model** (ledger 06-R8).
- **Safetensors Direct (the sovereign pipeline):** `sp_transcode --st`
  takes weight VALUES from the official checkpoint; GGUF supplies
  verified-clean metadata/tokenizer only; mapped-but-missing = hard
  error. OK_Q8 artifact: PPL **4.7396 (+1.33%)**.
- **OK_Q4B (arena layout v2):** per-32-block f16 scales, store-then-derive;
  recipe **B1** (Q4B gate/up + Q8 rest, chosen from a 6-recipe simulation
  matrix — gemma4 is PTQ-hostile, all-sym-32 = +45%) → a 9.4 GB artifact
  at **5.1259 = the simulation to four decimals**. GPU kernel
  **`k_gemv_q4b_dp4a_v2`** (one weight block per 128-bit chunk) +
  `k_dequant_arena_q4b` + `DevTensor.bscale` routing landed **5.1160** —
  sim/CPU/GPU triple-instrument agreement. Core `85aadd3`, engine `bea361e`.
- **SHOOTOUT-2 (CITABLE, 06-R10): 26.1 tok/s at PPL 5.12 on the 2060-12GB**
  (24/24 gates; CUDA graph EXACT 256/256; dp4a top-1 256/256). llama.cpp:
  31.29 tok/s at PPL 192–506 — faster, on quality-failed artifacts; both
  halves stated. Engine bandwidth 245 vs 207 GB/s (+18%).

**The `SP_XBAR_*` latent-crossbar harness** also lives in
`cuda_forward.cu`: env knobs `SP_XBAR_CAPTURE` / `SP_XBAR_SPLICE` /
`SP_XBAR_ROW` / `SP_XBAR_NROWS` / `SP_XBAR_AT` / `SP_XBAR_MASK` /
`SP_XBAR_TOKENS` / `SP_XBAR_RANKS` / `SP_XBAR_RESID` / `SP_XBAR_POSFREE`
(P1: direct KV-cache capture/splice — **token-free steering PROVEN**,
public ledger X-R1: 15/15 lexical incorporation, 15/15 selectivity,
3.69 orders max rank pull) plus `SP_XBAR_EMB` / `SP_XBAR_EMB_CAPTURE`
(P2.a residual-entry pseudo-token injection). Gate harness:
`tests/test_xbar_p1_cuda.c`.

#### 5.2.2 Bench tool — `tests/bench_gemv_int8.cu`

Standalone nvcc microbench (no engine link) that isolates the weight matmul
from attention / argmax / launch overhead and sweeps the matrix dimension
`N = 1K..16K`, comparing f32 cuBLAS SGEMV vs int8 and Q4 dp4a GEMV. Prints
per-GEMV µs, the speedup, and effective GB/s, plus a host-reference correctness
gate. This is how the bandwidth ladder above was measured.

```bash
# from tests/, with the CUDA toolkit on PATH:
nvcc -O3 -arch=sm_75 bench_gemv_int8.cu -lcublas -o bench_gemv_int8
nvidia-smi -lgc 1500,1500          # pin SM clock (memory auto-boosts under load)
./bench_gemv_int8                  # sweep + crossover table
nvidia-smi -rgc                    # reset
```

Status: **built** + bit-exact-validated. **NOT wired into `sp_daemon`** —
symmetric WIRE-HEX-style sprint pending.

### 5.3 Vulkan backend — `src/backends/vulkan/`

| File | Role |
|------|------|
| `vulkan_forward.cpp` | Whole-forward entry point; command buffer + pipeline lifecycle |
| `vulkan_backend.cpp` | Backend init (instance, device, compute queue); error mapping to `SP_EVULKAN` |
| `vk_common.h` | Shared validator + scratch helpers |
| `shaders/` | GLSL → SPV compute shaders for matmul, RMSNorm, RoPE, attention, NTT |

Build flags:

```bash
cmake -B build-vulkan -G Ninja \
      -DSP_ENGINE_BACKEND=vulkan \
      -DSP_ENGINE_WITH_VULKAN=ON \
      -DSP_ENGINE_BUILD_TESTS=ON
```

Status: **built** + bit-exact-validated (closure `SESSION-CLOSED-lat-2-L1-PARITY.md`).
**NOT wired into `sp_daemon`** — symmetric WIRE-HEX-style sprint pending.

### 5.4 Hexagon HVX backend — `src/backends/hexagon/` + `tools/sp_compute_skel/`

Two pieces: **host side** runs on aarch64-android in the daemon process;
**device side** is the cDSP skel running on Hexagon V69 inside the
Qualcomm cDSP. They talk over FastRPC.

**Host side** (`src/backends/hexagon/`):

| File | Role |
|------|------|
| `sp_hex_host.c` | `gemma3_forward_hexagon` entry; weight upload; FastRPC round-trip |
| `sp_hex_rt.c` | Runtime helpers — DmaBuffer management, IDL marshalling |
| `sp_hex_layout.h` | On-device weight layout (mirrors Q8 arena) |
| `inc/sp_hex.idl` | Forward-pass IDL (separate from `sp_compute.idl` — that's the compute-primitive IDL) |
| `dsp/` | Local copy of skel sources for build cross-check |
| `echo_skel/` | Echo skel for sprint C (FastRPC smoke) |

**Device side** (`tools/sp_compute_skel/`):

| Path | Role |
|------|------|
| `inc/sp_compute.idl` | Compute-primitive IDL (axpby, scale_i16, vtcm_probe, ffn_2stage_halide, barrett_oracle, matmul_q, ntt_*) — full reference in §8 |
| `src_dsp/` | cDSP-side implementations using HVX intrinsics, Halide AOT FFN, VTCM staging |
| `halide_gen/` | Halide schedule generators |
| `hexagon_Release_toolv87_v69/` | Build artefacts |

Build (Windows host + Hexagon SDK 5.5.6.0):

```cmd
set HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\5.5.6.0
scripts\build\build-hexagon.bat
```

For the daemon-linkable host-side static lib (`libsp_hex_daemon_backend.a`):

```cmd
tools\sp_daemon\build-android-hex-backend.bat
```

Then cross-compile sp-daemon with the WIRE-HEX feature:

```cmd
set LIBCLANG_PATH=C:\Program Files\LLVM\bin
cd tools\sp_daemon
cargo build --target aarch64-linux-android --release ^
            --features wire_hex_backend --bin sp-daemon
```

See `start_wire_hex_daemon.sh` for the on-device launcher (env vars +
adb-push sequence).

Status: **built end-to-end**. NTT primitives (forward, twiddle VTCM
staging, dual-prime CRT dispatch, INTT + Garner) are byte-exact vs
math-core. Forward-pass wiring is shipped daemon-side; the device-side
`libsp_hex_skel.so` rebuild against the current `inc/sp_hex.idl` is the
pending step (out of scope for sprint WIRE-HEX, owned by a future
HX-SKEL-REBUILD sprint with the SDK build_cmake chain).

---

## 6. `sp_daemon` Rust crates

`sp_daemon` is one Cargo package with one library + 11 binaries.

### 6.1 Library (`tools/sp_daemon/src/lib.rs`)

| Module | Purpose |
|--------|---------|
| `dialogue` | M.2 zero-copy dialogue primitives: `SpinorReceipt` (64-byte audit envelope), `DialoguePool` (pre-allocated buffers), `MODEL_ID_EXECUTIVE`/`MEMORY` constants |
| `pouw_ledger` | M.4 append-only PoUW ledger: `Ledger::open`, `append`, `read_all`, `canonical_sort`, `replay_canonical_into`; cross-device byte-identity gates |
| `memo_routing` | M.5 KSTE-routed sparse Memory activation: builds a `RoutingMask` from a K-vector via the math-core KSTE encoder |
| `network::quic_shard` | Phase 6-NET QUIC transport: `SpQuicCoordinator`, `SpQuicWorker`, `ShardBlockHeader` (64-byte wire format), `run_garner_loop` (dual-prime CRT recombine on the driver) |
| `ntt_ffi` | Bindgen output for math-core NTT-CRT primitives |
| `dsp_rpc` (android) | Dynamic libcdsprpc.so loader: `FastRpcSession`, `DmaBuffer`, `RemoteArg`, `make_scalars` — single-thread invoke per session |
| `ntt_hex_dispatch` (android) | NTT.5b backend trampoline: routes `sp_compute_ntt_dispatch_fn` calls to `ntt_hvx_vtcm_oracle` (method 17) and `intt_hvx_oracle` (method 18) over FastRPC |
| `hex_forward_dispatch` (android, feature=`wire_hex_backend`) | WIRE-HEX full-forward backend dispatcher: implements `sp_forward_dispatch_fn` for `gemma3_forward_hexagon`; bumps `dispatch_count` atomic; survives session clone |
| `ffi_l1` (android) | Bindgen output for the L1 C ABI (`sp_session_*`, `sp_prefill_chunk`, `sp_decode_step`, `sp_session_register_forward_backend`) |

### 6.2 Binaries (`tools/sp_daemon/src/bin/` and `src/main.rs`)

| Binary | Source | Purpose |
|--------|--------|---------|
| `sp-daemon` | `src/main.rs` | Main long-lived HTTP/SSE server |
| `sp-console` | `bin/sp_console.rs` | Interactive REPL against a running daemon |
| `spec_validate` | `bin/spec_validate.rs` | Phase 4-SPEC speculative-decoding validator |
| `probe` | `bin/probe.rs` | Health probe — pings `/v1/metrics` and prints output |
| `sp_memo_m1_smoke` | `bin/sp_memo_m1_smoke.rs` | M.1 dual-model load budget audit + concurrent invoke smoke (android-only) |
| `sp_memo_m2_dialogue_smoke` | `bin/sp_memo_m2_dialogue_smoke.rs` | M.2 zero-copy dialogue smoke + Spinor receipt envelope (android-only) |
| `sp_memo_m4_ledger_smoke` | `bin/sp_memo_m4_ledger_smoke.rs` | M.4 PoUW ledger smoke (append + read + replay-determinism) |
| `sp_memo_m4_canonical_replay_smoke` | `bin/sp_memo_m4_canonical_replay_smoke.rs` | mesh-canonical-order + cross-device byte-identity gates |
| `sp_memo_m5_routing_smoke` | `bin/sp_memo_m5_routing_smoke.rs` | M.5 KSTE-routed sparse activation smoke |
| `sp_chat_dialogue_smoke` | `bin/sp_chat_dialogue_smoke.rs` | `/v1/dialogue` end-to-end HTTP smoke |
| `sp_chat_ledger_autowire_smoke` | `bin/sp_chat_ledger_autowire_smoke.rs` | `/v1/dialogue` + PoUW ledger autowire smoke |
| `sp_ntt_5c_forward_smoke` | `bin/sp_ntt_5c_forward_smoke.rs` | NTT.5c forward-activation smoke (android-only) |
| `sp_ntt_bench_toks` | `bin/sp_ntt_bench_toks.rs` | NTT-bench tok/s per-cell harness (2 models × 3 configs × 3 reps) |

Most smoke binaries print `"android-only"` and exit on host builds —
their FFI surface only links against the libs that cross-compile to
aarch64-android.

### 6.3 Key Rust types

**`AppState`** (`src/state.rs`) — the axum `State<Arc<AppState>>` payload.
Holds the L1 session(s), tokenizer(s), event broadcaster, receipt store,
peer map, optional Memory model, optional ledger, optional cDSP bridge.
Drop order preserves model lifetime past sessions.

**`SpSession`** (`src/session.rs`) — Rust `Send`-not-`Sync` wrapper
around the opaque C `sp_session *`. Owns the cancel flag. Methods:
`prefill_chunk(&mut self, tokens, logits)`, `decode_step(&mut self, token, logits)`,
`clone_session(cancel_flag)`, `rewind(n)`, `position()`.

**`SpinorReceipt`** (`src/dialogue.rs`) — exact 64-byte
`#[repr(C, packed)]` struct: `turn_index u8 | model_id u8 | _pad [u8;2] |
wall_us u32 | input_hash [u8;24] | output_hash [u8;24] | n_input_tokens u32 |
n_output_tokens u8 | _reserved [u8;2] | sentinel 0xA5`. Compile-time
`size_of` assertion catches accidental padding changes.

**`DialoguePool`** (`src/dialogue.rs`) — pre-allocated buffers for the
3-turn dialogue loop. `.clear() + .push()` only inside the hot loop;
no allocator activity per turn (per `reference-zero-copy-invariant`).

**`Ledger`** (`src/pouw_ledger.rs`) — append-only file of 64-byte
receipts. Methods: `open`, `append`, `read_all`, `len`, `canonical_sort`
(stable sort on `(turn_index, input_hash[..2])`), `replay_canonical_into`
(canonical-order byte-identical replay into a new file — the cross-device
identity gate).

**`ShardBlockHeader`** (`src/network/quic_shard.rs`) — 64-byte
`#[repr(C)]` wire header: `seq_id u64 | token_pos u32 | layer_id u32 |
prime_selector u8 | _pad [u8;47]`. `prime_selector = 0` → `q_1 =
1073738753`, `1` → `q_2 = 1073732609`. `ResidueBlock` carries the
header + N residues.

---

## 7. HTTP / SSE / WebSocket API

All endpoints live under the version prefix `/v1/`. Wired in
`tools/sp_daemon/src/server.rs::build_router`.

### 7.1 Endpoint summary

| Method | Path | Purpose | Streaming |
|--------|------|---------|-----------|
| POST | `/v1/chat` | Single-shot chat → token delta stream | SSE |
| GET  | `/v1/chat/stream` | Legacy SSE stub (returns `{"status":"stub"}`) | JSON |
| POST | `/v1/dialogue` | Dual-model Grounding → Entity ID → Synthesis | JSON |
| POST | `/v1/abort/:id` | Cancel a running chat | 204 / 404 |
| GET  | `/v1/events` | Daemon-wide event stream (chat lifecycle, PoUW mints) | SSE |
| GET  | `/v1/metrics` | Tokens/sec + position + peer count | JSON |
| GET  | `/v1/receipts` | All accumulated PoUW receipts | JSON |
| GET  | `/v1/pouw/ledger` | Live KSTE receipt feed | SSE |
| GET  | `/v1/mesh/peers` | Active QUIC peers + shard assignment | JSON |
| WS   | `/v1/node/telemetry` | 1 Hz node telemetry stream | WebSocket |
| POST | `/v1/dsp/echo` | (android) FastRPC echo through cDSP | Bytes |
| GET  | `/v1/dsp/model_info` | (android) DSP-resident model metadata | JSON |
| GET  | `/v1/debug/backend_counts` | WIRE-HEX + NTT.5b dispatch counters | JSON |

Static file serving: `frontend_mockups/` is mounted as fallback under
the router (everything that doesn't match a route gets file-served).
CORS is permissive by default.

### 7.2 `POST /v1/chat`

**Request body** (one of `prompt`, `messages`, `prompt_tokens` required —
exactly one):

```json
{
  "prompt": "Why is the sky blue?",
  "max_tokens": 256,
  "stop": ["\n\n"]
}
```

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why is the sky blue?"}
  ],
  "max_tokens": 256
}
```

```json
{
  "prompt_tokens": [2, 1037, 4, 5683],
  "max_tokens": 8,
  "stop": []
}
```

**Response** — `text/event-stream`. Each event:

```
data: {"delta":"The ","chat_id":42}

data: {"delta":"sky ","chat_id":42}

data: {"delta":"appears blue because","chat_id":42}

data: [DONE]
```

On client disconnect or `/v1/abort/:id`, an `event: cancelled` is
emitted instead of `[DONE]`. On error during prefill or decode, a
single `data: {"error":"..."}` event is sent.

**Error responses (4xx, JSON body):**
- `400` `{"error":"one of prompt / messages / prompt_tokens required"}`
- `400` `{"error":"only one of prompt / messages / prompt_tokens may be set"}`
- `400` `{"error":"chat_template_unavailable","arch_id":<id>,"hint":"use prompt or prompt_tokens"}`
- `400` `{"error":"<tokenizer error message>"}`

### 7.3 `POST /v1/dialogue`

Returns 501 if the daemon wasn't started with `--memo-model` /
`--memo-tokenizer`.

**Request body:**

```json
{"prompt": "Who painted the Sistine Chapel?"}
```

**Response (200):**

```json
{
  "response": "Michelangelo painted the Sistine Chapel ceiling.",
  "receipts": [
    "AAEAAAArAAAAVQ...   <88-char base64 of 64-byte SpinorReceipt turn 1>",
    "AAIAAAAvAAAA...     <turn 2>",
    "AAMAAAA0AAAA...     <turn 3>"
  ],
  "wall_ms": 412,
  "turn_us": [128000, 184000, 100000]
}
```

Each receipt decodes to a 64-byte `SpinorReceipt`. Byte 0 = `turn_index`,
byte 1 = `model_id` (`0xE` = Executive, `0x4D` = Memory), bytes 2-3
padding, bytes 4-7 `wall_us` LE, bytes 8-31 SHA-256 truncated input hash,
bytes 32-55 SHA-256 truncated output hash, bytes 56-59 `n_input_tokens`
LE, byte 60 `n_output_tokens`, bytes 61-62 reserved, byte 63 = `0xA5`
sentinel.

**Error responses:**
- `400` `{"error":"prompt required"}`
- `500` `{"error":"exec clone: <detail>"}`
- `500` `{"error":"memo clone: <detail>"}`
- `500` `{"error":"run_dialogue: <detail>"}`
- `501` `{"error":"memo_model_not_loaded","hint":"start sp-daemon with --memo-model ..."}`

If `--pouw-ledger-path` is set, all three receipts are best-effort
appended to the ledger BEFORE the HTTP response is built. A lock or
append failure logs a warning and the response still ships
(`tracing::warn!` rather than 5xx).

### 7.4 `POST /v1/abort/:id`

`:id` = `chat_id` from a `ChatDelta` event. Returns `204 NO_CONTENT`
if the cancel flag was flipped; `404 NOT_FOUND` if no active chat with
that id.

### 7.5 `GET /v1/events`

Long-lived SSE channel for daemon-wide events:

```
event: chat_completed
data: {"chat_id":42,"status":"done"}

event: chat_completed
data: {"chat_id":43,"status":"cancelled"}

event: mint
data: {"receipt_hex":"<304 hex chars = 152 bytes>","sig_hex":"<128 hex chars = 64 bytes>"}
```

Each new `/v1/events` subscriber gets a fresh broadcast subscription;
back-pressure-bounded (64-event channel; slow consumers drop with
`tokio_stream` lagged events).

### 7.6 `GET /v1/metrics`

```json
{
  "tokens_per_sec": 1.43,
  "ram_svm_bytes": 0,
  "peers": 0,
  "phase": "lat-phase-2-l3-tok-closed",
  "session_pos": 22
}
```

`tokens_per_sec` is lifetime tokens decoded / elapsed since daemon start.
`peers` is the count from `peer_map` (always 0 on android). `session_pos`
is the base session's current position (0 except in single-session
debug builds — chat clones the session per request, so the base stays at 0).

### 7.7 `GET /v1/receipts`

```json
{
  "receipts": [
    {"payload_hex": "<304 hex>", "sig_hex": "<128 hex>", "round": 7}
  ],
  "cursor": null
}
```

These are the PoUW receipts minted by the background sieve-mining loop
(`src/mining.rs`). Each is the 152-byte wire-format receipt plus a 64-byte
ed25519 signature over the payload.

### 7.8 `GET /v1/mesh/peers`

```json
{
  "peers": [
    {"node_id": "192.0.2.10:5000", "address": "192.0.2.10:5000",
     "shard_id": "q1", "latency_ms": 45}
  ],
  "active": 1,
  "total": 32
}
```

`shard_id` is `"q1"` if `prime_selector == 0` in the first received
block from that peer, `"q2"` if `1`. Peers in handshake state (no block
yet) show `shard_id == 255` internally (not surfaced).

### 7.9 `GET /v1/node/telemetry` (WebSocket)

Server pushes JSON every 1000 ms:

```json
{
  "node_id": "q3-beast-canyon",
  "cpu_temp_c": 58.5,
  "svm_mem_gb": 2.4,
  "dht_peers_active": 1,
  "dht_peers_total": 32,
  "pouw_frontier": 7
}
```

### 7.10 `GET /v1/pouw/ledger`

SSE feed of KSTE-formatted receipt lines as they're minted:

```
data: [KSTE] Round: 7 | Nonce: 0x4f3a... | Z_q Hash: 0xa1b2...
```

### 7.11 `POST /v1/dsp/echo` (android-only)

Routes a raw `application/octet-stream` body through the V69 cDSP echo
skel. Max body 8 MB. Returns the echoed bytes on success.

```bash
curl -X POST http://127.0.0.1:8080/v1/dsp/echo \
     -H "Content-Type: application/octet-stream" \
     --data-binary @some-blob.bin -o echo-out.bin
```

- `400` empty body
- `413` body > 8 MB
- `500` `dsp_rpc: <error>`
- `501` "cDSP session not admitted" / "v1/dsp/echo requires target_os=android"

### 7.12 `GET /v1/dsp/model_info` (android-only)

```json
{
  "n_layers": 28,
  "hidden_size": 1024,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "total_dma_bytes": 731589632,
  "load_wall_ms": 4127,
  "kv_cache_bytes": 92274688
}
```

Returns 501 with body `"model not loaded"` if the DSP-resident model
load failed.

### 7.13 `GET /v1/debug/backend_counts`

```json
{
  "hex_forward_count": 1,
  "wire_hex_active": true,
  "ntt_hex_forward_count": 0,
  "ntt_hex_inverse_count": 0
}
```

- `hex_forward_count` — `gemma3_forward_hexagon` dispatcher hits since
  process start; > 0 after one prefill when `SP_DAEMON_BACKEND=hex` AND
  feature `wire_hex_backend`. Always 0 on host builds and on android
  without the feature.
- `wire_hex_active` — whether `sp_session_register_forward_backend`
  succeeded at startup. Independent of whether a prefill has run.
- `ntt_hex_forward_count` / `ntt_hex_inverse_count` — Hexagon NTT
  dispatch counters (Bluestein inner kernels via FastRPC methods 17/18).
  Always 0 when `SP_ENGINE_NTT_ATTN_HEX` is unset.

---

## 8. Hexagon skel IDL reference

The IDL at `tools/sp_compute_skel/inc/sp_compute.idl` defines the
FastRPC contract between the daemon-side trampoline and the cDSP V69
HVX skel. Each method returns `long` (0 = success, -1 = constraint
violation, other negative = AEE error code).

| qaic method | Name | Sprint | Purpose |
|:--:|------|:--:|---------|
| 1 | `axpby` | §3-HX D | Fixed-point AXPBY `y[i] = sat_i16((a·x[i] + b) >> q_bits)` — scalar pipelined |
| 2 | `scale_i16` | §3-HX D | HVX-vectorized i16 scale, the canonical HVX proof. Uses `Q6_Vh_vadd_VhRh_sat`. |
| 3 | `axpby_hvx` | §3-HX E F1 | Full axpby via HVX intrinsics — `vmpy` widening → `vadd` word → `vasr` → `vpack_VwVw_sat`. Constraints: `|a_h| ≤ 32767`, `0 ≤ q_bits ≤ 30`. |
| 4 | `scale_i16_batched` | §3-HX E F2 | Amortize FastRPC per-call overhead — batched scale_i16 |
| 5 | `vtcm_probe` | §3-HX F | `HAP_request_VTCM(size, single_page)` litmus — reports admit / deny + low-32 of VTCM addr |
| 6 | `axpby_2d_halide` | §3-HX F | Halide-AOT 2D axpby with VTCM hot-copy. `cols % 64`, `|a[c]| ≤ 32767`, `q_bits ≤ 30` |
| 7 | `ffn_2stage_halide` | §3-HX G | 2-stage matmul FFN via Halide AOT with dual-VTCM staging (external X/W1/W2/Y + internal `hidden`). Constraints: `d_in`/`h_dim`/`d_out` multiples of 128; `batch` ≥ 1 multiple of 4 |
| 8 | `ffn_2stage_diag_halide` | §3-HX H | Same as 7 but also writes post-stage-1 `hidden` to a caller buffer for matmul-1 vs matmul-2 isolation |
| 9-10 | (reserved) | — | reserved slots from earlier sprint reshuffle |
| 11 | `barrett_oracle` | §3-HX K v0.β 2.5 | N test (a, b) u32 pairs through modular multiply mod q_1 or q_2 |
| 12 | `matmul_q` | §3-HX K v0.β 2.5c | HVX `mod_q` matmul `Y[b][i] = (Σ_k X[b][k]·W[k][i]) mod q`. Constraints: `d_out % 32 == 0`, `q_idx ∈ {0,1}` |
| 13 | `ntt_oracle` | §4-NTT NTT.0 | Scalar negacyclic NTT mod q_1 or q_2. `N ∈ {128,256,512}` |
| 14 | `ntt_hvx_oracle` | §4-NTT NTT.1 | HVX-vectorized NTT butterfly. Large stages (`half ≥ 32`) HVX intrinsics; small stages scalar fallback |
| 15 | `ntt_twiddle_init` | §4-NTT NTT.2 | Precompute + pin all 6 (prime, N) twiddle tables in VTCM via `HAP_request_VTCM`. Idempotent. |
| 16 | `ntt_twiddle_status` | §4-NTT NTT.2 | Inspect one (prime, N) entry's VTCM state — base addr, size, per-sub-table offsets |
| 17 | `ntt_twiddle_dump` | §4-NTT NTT.2 | Copy one sub-table from VTCM into a caller buffer for cross-check |
| 18 | `ntt_hvx_vtcm_oracle` | §4-NTT NTT.3 | VTCM-aware HVX forward NTT. Production forward path for dual-prime CRT dispatch. Requires `ntt_twiddle_init` first. |
| 19 | `intt_hvx_oracle` | §4-NTT NTT.4 | HVX-vectorized inverse NTT mod a frozen Proth prime. Consumes `ipsi_pow`, `w_inv`, `w_inv_stages` VTCM tables. |

(Method numbers shifted slightly relative to early sprint specs due to
merge-time renumbering — both NTT.3 and NTT.4 anticipated method 17;
NTT.4 was renumbered to 18 at merge.)

Full per-method input/output buffer shapes + semantics:
`tools/sp_compute_skel/inc/sp_compute.idl`.

The separate forward-pass IDL `src/backends/hexagon/inc/sp_hex.idl`
defines the methods the WIRE-HEX path uses (`forward`, `upload_crc`,
etc.) — its skel binary `libsp_hex_skel.so` is what currently needs
rebuilding for the WIRE-HEX BIT-EXACT gate to flip.

---

## 9. CLI flags + environment variables

### 9.1 `sp-daemon start` CLI

| Flag | Env | Default | Purpose |
|------|-----|---------|---------|
| `--model` | `SP_MODEL_PATH` | (required) | Target/verifier `.sp-model` path |
| `--tokenizer` | `SP_TOKENIZER_PATH` | (required) | Matching `.sp-tokenizer` path |
| `--draft-model` | `SP_DRAFT_MODEL_PATH` | empty | Draft `.sp-model` for Phase 4-SPEC spec-decode |
| `--draft-tokenizer` | `SP_DRAFT_TOKENIZER_PATH` | empty | Draft `.sp-tokenizer` |
| `--memo-model` | `SP_MEMO_MODEL_PATH` | empty | Memory `.sp-model` for `/v1/dialogue`. Endpoint returns 501 if unset. |
| `--memo-tokenizer` | `SP_MEMO_TOKENIZER_PATH` | empty | Memory `.sp-tokenizer` |
| `--pouw-ledger-path` | `SP_POUW_LEDGER_PATH` | empty | Enable PoUW ledger autowire from `/v1/dialogue` |
| `--port` | `SP_HTTP_PORT` | `8080` | TCP port for HTTP API |
| `--quic-port` | `SP_QUIC_PORT` | `0` (disabled) | UDP port for QUIC DHT mesh |
| `--peer` | — | empty | Single QUIC peer address to dial on startup (alias for `--peers`) |
| `--peers` | `SP_PEERS` | empty | Comma-separated list of QUIC peers to dial |

### 9.2 Runtime knob env vars (read by math-core + backends per-forward)

| Env | Values | Effect |
|-----|--------|--------|
| `SP_DAEMON_BACKEND` | `hex` (else unset) | (WIRE-HEX feature build only) register `gemma3_forward_hexagon` against target session at startup |
| `SP_ENGINE_FROB` | `0..4` | Weight path: 0 = pure f32; 1 = Q8 inline; 2 = Q8 dequant; 3 = Q4 inline; 4 = Q4 mixed-precision |
| `SP_Q4_PROMOTE` | float | Q4 rows whose round-trip rel-error exceeds this get promoted to Q8 |
| `SP_ENGINE_F16_ACT` | `0`/`1` | Round matmul activations to F16 (ggml-faithful cross-validation path) |
| `SP_CPU_SCALAR` | `0`/`1` | Force scalar reduction (disable AVX vectorization) |
| `SP_KV_SPINOR` | `0`/`1` | Persistent compressed Spinor-block KV cache |
| `SP_KV_SPINOR_REF` | `0`/`1` | Parity reference: f32 cache + in-place Spinor round-trip |
| `SP_ENGINE_NTT_ATTN` | `0`/`1` | Enable polynomial-ring NTT attention overlay (prefill only) |
| `SP_ENGINE_NTT_ATTN_HEX` | `0`/`1` | (android) Route inner NTT calls through FastRPC methods 17/18 |
| `SP_ARENA` | `q8` / `q4` | Build the packed-weight arena at load (Q8 or Q4 mixed-precision) |
| `SP_CUDA_DECODE_GRAPH` | `0`/`1` | (CUDA decode) capture the per-token generate step into a CUDA graph and replay it (position-indirect kernels). Warm win ≈1.06×; cold-start is the real wall. |
| `SP_CUDA_DECODE_INT8` | `0`/`1` | (CUDA decode) route packed matmuls through the fused dp4a GEMV (Q8 or Q4 per `DevTensor.prec`) — 1 byte/weight straight from VRAM, no f32 scratch. ~3.8× (int8) / ~7× (Q4) over f32 at 12B-scale; a tie at 0.6B (overhead-bound). Top-1 lossless. |
| `SP_ARENA_RELEASE` | `0`/`1` | Release the GGUF mapping after arena pack (~50% RAM cut) |
| `SP_ARENA_EMBED` | `0`/`1` | Include the token embedding in the arena pack |
| `ADSP_LIBRARY_PATH` | path | (android) Where FastRPC looks for `libsp_compute_skel.so` and other skels |
| `RUST_LOG` | string | `tracing-subscriber` filter (e.g. `sp_daemon=debug,axum=info`) |

Default arena precision is **8** (Q8); set `SP_ARENA=q4` for Q4 mixed.
Defaults across `SP_ENGINE_*` are 0 (off) — the unfeatured baseline is
bit-identical to a plain inference path (per
`shannon-prime-lattice/papers/PPT-LAT-Systems.md` binding rule).

---

## 10. Model conversion (`sp_transcode`) — the sovereign weight pipeline

The `sp_transcode` tool produces the `.sp-model` + `.sp-tokenizer` pair
that the engine `mmap`-loads. Two weight lanes:

- **GGUF lane:** `sp_transcode <in.gguf> <out.sp-model> <out.sp-tokenizer>` —
  the original reducing transcode.
- **Safetensors Direct (`--st <model.safetensors>`):** weight VALUES come
  from the official checkpoint; the GGUF supplies verified-clean
  metadata/tokenizer only; a tensor that is mapped but missing in the
  safetensors is a **hard error**. This is the ONLY trusted gemma-4-12B
  weight path — every gemma-4 GGUF measurable in June 2026 carries broken
  weights (engine-independent measurement, PPL 192–506; ledger 06-R8 +
  the public `GEMMA4-QUANT-FIX.md`).

### 10.1 Usage

```bash
sp_transcode <in.gguf> <out.sp-model> <out.sp-tokenizer> [--verify] [--q4|--q8|--q4b|--q4b-ffn] [--st <model.safetensors>]
sp_transcode --tok-only <in.gguf> <out.sp-tokenizer>
```

- `--verify` runs a per-tensor round-trip dequant check (rms + max
  rel-error) against the GGUF source and rejects the output if a Q8 row
  exceeds the threshold.
- `--q4b` / `--q4b-ffn` select the **OK_Q4B** codec (per-32-block f16
  scales, arena layout v2, dtypes 13/14 + `.bscale` sibling); `--q4b-ffn`
  is recipe **B1** (Q4B gate/up, Q8 rest — the gated 12B recipe).
- `--tok-only` (#115) extracts ONLY the family-tagged `.sp-tokenizer`
  blob — the tokenizer-regeneration path used for the 12B text-in
  deployment.

### 10.1b The `models\` pairing rule (read before touching installed blobs)

A `.sp-model` header is **SHA-256-paired to its `.sp-tokenizer`**
(`tokenizer_hash` in the header = SHA-256 of the blob; the loader returns
`SP_ETOKENIZER_HASH` on mismatch). The out-of-tree `models\` directory
holds installed, hash-paired sets; the `*.pre115` files alongside are the
byte-exact pre-#115 gold backups. **Never move non-`.pre115` blobs out of
`models\`** — you would break the pairing of an installed pair. Gate
`T_G4_TOK_12B_PAIRED` checks the installed 12B pairing end-to-end.

### 10.2 Supported inputs

GGUF v3 files containing:
- Architectures: Llama-3, Qwen3, Qwen2.5, Gemma3, Gemma4, DeepSeek V4,
  Qwen3.6-MoE (per `sp_arch_id` enum in `include/sp/sp_model.h`)
- Per-tensor dtypes: `GGML_T_F32`, `GGML_T_F16`, `GGML_T_Q8_0` (others
  return "unsupported src type" and abort)
- Tokenizers: SentencePiece, BPE-Llama3, BPE-GPT2, TikToken-O200K, and
  **GEMMA4_BPE** (#115) — the family tag (`GPT2_BPE | SPM | GEMMA4_BPE`)
  is written into the `.sp-tokenizer`; an unknown family is a hard error,
  and `src/tokenizer/gemma4_bpe.c` dispatches on it (5432/5432 HF parity,
  both lanes)

### 10.3 Output format

Per `shannon-prime-lattice/papers/PPT-LAT-SP-MODEL-v0.md`:

- **`.sp-model`** — fixed 512-byte header + tensor table (256 B per
  entry, sorted by xxh64 of the tensor name) + data region (64 KB
  aligned, each tensor 64 B aligned). The header carries `arch_id`
  (`SP_ARCH_ID_QWEN3 = 2`, `SP_ARCH_ID_GEMMA3 = 3`, etc.) + 256 B of
  arch_struct payload memcpy'd from `sp_arch_info`.
- **Per-tensor policy:**
  - matmul weights (attn q/k/v/o, ffn gate/up/down, LM head,
    `token_embd`): dequant to f32, re-quant into `SP_DT_OK_Q8` (int8
    per-row codes) + sibling `<name>.scale` tensor
    (`SP_DT_FROBENIUS_SCALE_FP32`, one fp32 per row).
  - norms and other tensors: copied as F32 (dequant F16→F32 if needed).
- **Data-region layout:** weights and their `.scale` siblings are
  adjacent (no interposing tensor), so the loader reconstructs a
  bit-identical packed arena via a single `memcpy` per row.
- **`.sp-tokenizer`** — self-describing serialization of the GGUF
  tokenizer arrays (tokens, scores, merges) + 128-byte header carrying
  type_id, vocab size, BOS/EOS/PAD/UNK IDs, and a SHA-256 over the
  whole file. The L1 loader binds models to tokenizers by this hash
  (`SP_ETOKENIZER_HASH` if mismatched).

### 10.4 Validation

`--verify` mode reports per-tensor stats:

```
qwen3.layers.0.attn_q.weight (4096 x 4096, Q8):
    rms_err 0.000183  max_rel_err 0.0021  promoted 0/4096 to Q8
```

For Q4 mode (`SP_ARENA=q4`), rows exceeding `SP_Q4_PROMOTE` (default
`0.01`) get promoted to Q8 — the promoted count reports as
`sp_arena_promoted(arena)`.

---

## 11. Peering / QUIC mesh

The mesh is a **dual-prime CRT shard fabric** today, with the
Fibonacci-Prime DHT spec'd for Phase 8 (see
`shannon-prime-lattice/papers/PPT-LAT-Roadmap.md` §8).

### 11.1 Wire format

Each peer-to-peer message is a 64-byte `ShardBlockHeader` followed by N
× 4 bytes of u32 residue payload:

```
byte  0..8   seq_id           u64 LE   global sequence counter
byte  8..12  token_pos        u32 LE   token position in context
byte 12..16  layer_id         u32 LE   transformer layer index
byte 16      prime_selector   u8       0 = q_1 = 1073738753, 1 = q_2 = 1073732609
byte 17..64  reserved         zeros
```

Max payload: 64 + 512·4 = 2112 bytes (`N ≤ 512` per the frozen-primes
NTT cap). Streams are unidirectional QUIC streams — one per residue
block; independent delivery; no head-of-line coupling.

### 11.2 Topology

- **Coordinator** (`SpQuicCoordinator::bind`) accepts incoming connections
  and per-stream residue blocks; calls `run_garner_loop` to Garner-
  recombine paired (q_1, q_2) residues for the same `seq_id` into
  centered signed coefficients.
- **Worker** (`SpQuicWorker::connect`) dials a coordinator and sends
  `ResidueBlock`s on independent unidirectional streams.

Each peer is assigned one prime (its `shard_id`). Two-peer topology
covers both primes; the coordinator Garner-recombines into the centered
signed result and feeds it back into the forward path.

### 11.3 TLS / identity

Dev-mode TLS uses self-signed certs with a `SkipServerVerification`
verifier — acceptable for the inference-cluster smoke. Phase 5+
swaps this for ed25519 dominance identity verification against a known
lattice node registry; the integration point is documented in
`tools/sp_daemon/src/network/quic_shard.rs` (search "INTEGRATION POINT:
Replace with Phase 5 ed25519 dominance identity").

### 11.4 Receipt replay

PoUW receipts minted on one node can be replayed byte-identically on
another node via `Ledger::canonical_sort` (stable sort on
`(turn_index, input_hash[..2])`) → `Ledger::replay_canonical_into`
(write a new ledger with canonical order). Cross-device byte-identity
is the M.4 + mesh-canonical-order gate; see closure
`CLOSURE-MESH-CANONICAL-ORDER.md`.

### 11.5 Connecting peers

```cmd
sp-daemon start --model ... --tokenizer ... ^
                --port 8080 --quic-port 5000 ^
                --peers 192.0.2.10:5001,192.0.2.11:5002
```

`/v1/mesh/peers` reports the live peers. `/v1/node/telemetry`
WebSocket pushes `dht_peers_active` every second.

---

## 12. Development workflow

### 12.1 Build matrix

| Build | Command |
|-------|---------|
| CPU host (Windows) | `scripts\build\build-cpu.bat` |
| CUDA host (Windows) | `scripts\build\build-cuda.bat` |
| Vulkan host (Windows) | `scripts\build\build-vulkan.bat` |
| Hexagon host-side libs (Windows) | `scripts\build\build-hexagon.bat` |
| Hexagon cDSP skel | `scripts\build\build-hexagon.bat dsp` |
| Daemon-linkable hex backend lib | `tools\sp_daemon\build-android-hex-backend.bat` |
| Cross-compile math-core to android | `tools\sp_daemon\build-android-libs.bat` |
| `sp_daemon` cargo build (host) | `cd tools\sp_daemon && cargo build --release` |
| `sp_daemon` cargo build (android) | `cargo build --target aarch64-linux-android --release` |
| `sp_daemon` with WIRE-HEX | `cargo build --target aarch64-linux-android --release --features wire_hex_backend` |

### 12.2 Run smoke tests

```bash
# All ctest gates on a build directory
ctest --test-dir build-cpu --output-on-failure -j

# A specific smoke binary (host)
cd tools/sp_daemon && cargo run --release --bin probe

# Android-only smokes (push + run via adb)
adb push target/aarch64-linux-android/release/sp_chat_dialogue_smoke /data/local/tmp/
adb shell /data/local/tmp/sp_chat_dialogue_smoke
```

Closures from recent smokes live under
`tools/sp_compute_skel/docs/CLOSURE-*.md`. They are the audit trail.

### 12.3 Adding a new backend

The canonical pattern is sprint WIRE-HEX
(`tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX.md`). Five stages:

1. **Math-core: add §6 forward-backend hook** — already shipped in
   `include/sp/sp_l1.h` (`sp_session_register_forward_backend`,
   `sp_forward_dispatch_fn` typedef, `sp_session_qwen3_model`
   accessor).
2. **Engine: build a daemon-linkable static lib** at
   `tools/sp_daemon/c_backend/lib<name>_daemon_backend.a` containing
   your `<arch>_forward_<backend>` entry point + a kernel-name shim
   that aliases `matmul`/`embed_row`/`as_f32` to the math-core
   `sp_*` variants (avoids `cpu_overlay.c` symbol collisions).
3. **Rust trampoline** at `tools/sp_daemon/src/<backend>_forward_dispatch.rs`
   implementing the §6 ABI; bump a process-static dispatch counter.
4. **AppState wiring** in `tools/sp_daemon/src/daemon.rs`: env-gate via
   `SP_DAEMON_BACKEND=<backend>`; register on the TARGET session
   pre-Mutex-wrap; surface via `AppState.<backend>_active`.
5. **Smoke**: drive `/v1/chat` + `/v1/debug/backend_counts` to confirm
   `<backend>_forward_count > 0` after one prefill.

For NTT-dispatch overlay backends (Hexagon NTT.5b, future Vulkan-NTT),
the pattern is `sp_pr_bluestein_set_backend` in
`lib/shannon-prime-system/core/poly_ring_bluestein/` (see
`tools/sp_daemon/src/ntt_hex_dispatch.rs` for the template).

### 12.4 Adding / updating an IDL method

1. Edit `tools/sp_compute_skel/inc/sp_compute.idl`. Add the new method
   at the end (renumber if a parallel sprint took your anticipated
   method id at merge time).
2. Regenerate skel stubs via qaic. The build script handles this; if
   you need to do it manually:
   ```cmd
   set HEXAGON_SDK_ROOT=C:\Qualcomm\Hexagon_SDK\5.5.6.0
   "%HEXAGON_SDK_ROOT%\tools\qaic\Ubuntu18\qaic" -mdll ^
       -o tools\sp_compute_skel\gen ^
       tools\sp_compute_skel\inc\sp_compute.idl
   ```
3. Implement the method body in `tools/sp_compute_skel/src_dsp/`.
4. Rebuild + push the skel: `scripts\build\build-hexagon.bat dsp` then
   `adb push <skel.so> /data/local/tmp/sp22u/`.
5. Add a Rust trampoline if a daemon-facing surface is desired.

### 12.5 Parallel-agent worktree discipline

Per `feedback-parallel-agents-separate-worktrees`: when dispatching 2+
agents concurrently on the same repo, **each agent operates in its own
git worktree** (`git worktree add ../wt-<sprint> main`). Otherwise
concurrent `git add` cross-contaminates: one agent's uncommitted files
get swept into another's commit. The operational fix is per-worktree
dispatch before agents start; the technical recovery (if it slips) is
to honestly disclose contamination in the closure rather than rewriting
shared history.

---

## 13. Known issues / pending

The user has been frustrated by hours of work that turned out to
bypass the production critical path. This section names what's pending
honestly.

| Issue | Workaround | Resolution |
|-------|-----------|------------|
| **WIRE-HEX BIT-EXACT gate blocked by cDSP skel mismatch** | None for the headline tok/s win | Future HX-SKEL-REBUILD sprint owns rebuilding `libsp_hex_skel.so` against current `src/backends/hexagon/inc/sp_hex.idl` and pushing to `/data/local/tmp/sp22u/` |
| **CUDA / Vulkan backends not wired to `sp_daemon`** (CPU IS wired — sprint WIRE-CPU, `SP_DAEMON_BACKEND=cpu`) | Use math-core reference path / CPU backend | Two symmetric WIRE-HEX-style sprints; each one is ~1 day of plumbing once the WIRE-HEX template is in hand |
| **`sp_decode_step` uses fp32 reference even with `SP_ENGINE_NTT_ATTN=1`** | Decode is the path where this matters most for tok/s; current architecture re-runs full forward on prefill backends | NTT.5e (filed, not shipped) wires decode-path NTT routing |
| **HD=128 direct path can't use Hexagon backend** | Bluestein at HD=64 covers Qwen3 / Qwen2.5-Coder; Gemma3-1B uses HD=256 (direct N=256 NTT works) | NTT.5d (filed, not shipped) wires a direct backend path at HD=128 |
| **Hexagon backend re-runs full forward per call** | Decode path stays on math-core reference; bypasses the issue | HEX-DECODE-1 sprint would add per-backend persistent-KV variant |
| **CPU backend's `cpu_overlay.c` symbol-collides with math-core's `sp_*` kernels** | The daemon-link backend lib uses a kernel-name shim (`sp_daemon_hex_glue.c`) to alias names | Same shim pattern applies to future CUDA / Vulkan daemon links |
| **TLS in QUIC mesh accepts any cert** | Dev-only; lattice clusters today are operator-controlled | Phase 5 ed25519 dominance identity verification swap |
| **Tokenizer chat-template support varies by arch** | `/v1/chat` falls back to `prompt` / `prompt_tokens` if `messages` template lookup fails | Per-arch template registration is an open task |
| **`tracing_subscriber` filter set at daemon start, not hot-reloadable** | Restart daemon to change `RUST_LOG` | `sp-daemon reload` is a no-op for v0 |
| **CRT-mesh today is two-node (one shard per peer)** | Sufficient for the dual-prime CRT bit-exact smoke | Fibonacci-Prime DHT is spec'd (`papers/PPT-LAT-Roadmap.md` §8) |

For the audit trail of what shipped when (and what didn't), the
canonical reference is the chronological closure list under
`tools/sp_compute_skel/docs/CLOSURE-*.md` plus the lattice
`papers/SESSION-CLOSED-*.md`. The most recent closures
(`CLOSURE-WIRE-HEX.md`, `CLOSURE-NTT-bench.md`, `CLOSURE-NTT-5c.md`,
`CLOSURE-LEDGER-AUTOWIRE.md`, `CLOSURE-MESH-CANONICAL-ORDER.md`) are
the up-to-date status of record.

---

## License

MIT. See `LICENSE`.
