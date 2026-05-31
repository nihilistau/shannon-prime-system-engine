# PLAN — Sprint TRICK-1-FORWARD-V3 — dual-HVX-context per-matmul in `sp_hex_forward` via in-skel QURT worker pool

**Sprint:** Phase 2-TRICK-1-FORWARD-V3
**Date:** 2026-06-01
**Worktree:** `D:\F\shannon-prime-repos\engine-trick-1-fwd-v3`
**Branch:** `sprint/trick-1-forward-v3` (base engine main `687463e` — post TRICK-1 merge)
**Sub-tag candidate:**
- `lat-phase-2-trick-1-forward-v3-shipped` (T_TRICK1FWDV3_PERF_PARITY + T_TRICK1FWDV3_DECODE_BIT_EXACT both PASS)
- `lat-phase-2-trick-1-forward-v3-attempted` (substantive gate failure with named blocker; honest closure)

---

## Stage 0 — Mandatory pre-read (with file:line citations)

### 1. K v0.alpha closure (the load-bearing primitive proof)

**Verbatim on-device output:** `tools/sp_dsp_smoke/sprint_k_alpha_run_output.txt:14-44`.
- Bench-A (sequential, one thread, two invokes): **35.8 ms wall** = 2× single-invoke 17.7 ms.
- Bench-C (Arc<FastRpcSession>, two ARM threads, concurrent invokes): **18.3 ms wall** ≈ 1× single.
- **Overlap fraction = 17751/18301 = 0.9699.**
- **Speedup = 35.8/18.3 = 1.935×.**
- Kernel: Sprint H diag method 9 (FFN matmul + activation), shape 128×128 / B=8 (compute-bound).

**Substrate code:** `tools/sp_dsp_smoke/src/sp_dual_dispatch.rs:28-103`. Critical lines:
- `:32-35` — `DualDispatch { sess: Arc<FastRpcSession> }`.
- `:88-99` — two `thread::spawn(move || sess_X.invoke(...))` with `Arc::clone` per thread.
- `:71` — `sess.invoke(make_scalars(9,4,3), &mut args)` is what fires the FastRPC dispatch.

**The K v0.alpha primitive's parallelism model:** two ARM threads each call `sess.invoke()` on a single shared `Arc<FastRpcSession>`. The FastRPC layer delivers two CONCURRENT invokes to the cDSP. The cDSP scheduler engages SSR:XA={4,5} dual-vector-context attachment (per `reference-v69-hvx-expert-practices`) and each invoke gets its own HVX context. **This is dispatch-layer parallelism, not in-skel parallelism.**

### 2. `reference-fastrpc-concurrent-dispatch` memory

`/sessions/gallant-dreamy-franklin/mnt/.auto-memory/reference_fastrpc_concurrent_dispatch.md`:
- "**Concurrency is opt-in at the dispatch layer**, not the kernel layer."
- "**Kernels written for single-context dispatch work concurrently for free.**"
- "The cDSP scheduler is a kernel-agnostic dispatch primitive."

The K v0.alpha pattern requires **two separate FastRPC invokes** — that is the substrate that makes the cDSP scheduler attach two vector contexts.

### 3. `reference-dual-model-cdsp-scheduler` memory

`/sessions/gallant-dreamy-franklin/mnt/.auto-memory/reference_dual_model_cdsp_scheduler.md`:
- "The only requirements for the cDSP scheduler to engage parallelism are: (1) Two concurrent `Arc<FastRpcSession>` invokes from ARM-side threads (not Mutex-serialized), (2) Both kernels target HVX vector contexts, (3) Per-invoke compute is large enough that wall-clock dominates marshalling."
- "**Substrate ceiling is ~2× because there are 2 vector contexts on V69.**"
- "Re-measuring parallelism per kernel — the substrate is the same."

### 4. `reference-v69-hvx-expert-practices` — the resource model

`/sessions/gallant-dreamy-franklin/mnt/.auto-memory/reference_v69_hvx_expert_practices.md` — V69 has 4 scalar threads but only **2 vector contexts** (each with its own 1024-bit VRF + predicates). SSR:XA={4,5} on V69 attaches scalar threads 0/1 to vector contexts 0/1 respectively. The production dispatch model description: "Thread 0 → vector context 0 → matmul tile A; Thread 1 → vector context 1 → matmul tile B."

**Crucial:** the SSR:XA programming this memory describes is the SAME mechanism the cDSP scheduler invokes automatically when FastRPC delivers two concurrent invokes. The lattice has never directly programmed SSR:XA — K v0.alpha's 1.935× came from the cDSP scheduler doing it automatically.

### 5. `reference-v69-vrmpy-chat-shape-memory-bound` — the load-bearing constraint

`/sessions/gallant-dreamy-franklin/mnt/.auto-memory/reference_v69_vrmpy_chat_shape_memory_bound.md`:
- "V69 HVX inner loop in the HX.3b vrmpy matmul kernel at Gemma3-1B chat shape (K=2048, B=1, narrow matmul) is **MEMORY-BANDWIDTH-BOUND, not ALU-bound.**"
- HX.3b-α-v2 dropped wsum (50% ALU reduction): observed only **6.5% wall-clock lift**.
- "Future V69 vrmpy lifts at chat shape **must attack memory bandwidth** (prefetch, VTCM staging, 2-row-parallel kernel, batched activation interleave), not ALU."

**This is the structural risk for V3's PERF_LIFT gate.** Dual-HVX-context = two threads × two HVX contexts × two activation/weight read streams over the SAME DDR/L1 bandwidth. Empirically chat-shape vrmpy is already saturating bandwidth at one context.

### 6. HX.3b kernel — what V3 splits

**Single-context kernel:** `src/backends/hexagon/dsp/sp_hex_imp.c:205-252` (`hx_matmul_q8_vrmpy`) and `:340-382` (`hx_matmul_q8_vrmpy_v2` with rsum cache).
- Inner loop: `:225-235` — vrmpy_acc on 128-byte HVX vector blocks, dual accumulator (`acc_dot` + `acc_ws`) in HX.3b; single accumulator (`acc_dot`) in v2 with `rsum[]` lookup table.
- Activation quant: `:172-191` (`hx_quant_act_ub`).
- **Per-call activation buffer:** `:211` `static unsigned char act_ub[SP_HEX_VRMPY_MAX_IN]` — **NOT thread-safe**. Each cDSP worker thread needs its own copy when V3 splits per-matmul work.

**The 7 call sites in `sp_hex_forward`:** `src/backends/hexagon/dsp/sp_hex_imp.c:586-588, 609, 623-624, 631`.
- L587: WQ (out=QD=1152, in=E=1152, n_tok=16) — `hx_matmul_q8_vrmpy_v2(WPTR(SP_HEX_WQ), QD, E, nx, n_tok, q)`.
- L588: WK (out=KVD=256, in=E=1152, n_tok=16).
- L589: WV (out=KVD=256, in=E=1152, n_tok=16).
- L609: WO (out=E=1152, in=QD=1152, n_tok=16).
- L623: WGATE (out=FF=6912, in=E=1152, n_tok=16).
- L624: WUP (out=FF=6912, in=E=1152, n_tok=16).
- L631: WDOWN (out=E=1152, in=FF=6912, n_tok=16).

For 26 layers × 7 matmuls = **182 matmul calls per prefill** (ctx=16).

**Per-matmul time (HX.3b baseline):** prefill = 10.5 s total, marshalling tax included. 10.5s / 182 matmul ≈ 58 ms/matmul average. WGATE/WUP (out=6912) take ~3× this; WQ/WO take ~1× this.

### 7. HX.3b closure — the perf-parity baseline

`tools/sp_compute_skel/docs/CLOSURE-HX-3b.md:21-32`:
- HX.3b prefill = **1.523 tok/s** (3-rep mean, ctx=16); decode = 1.069 tok/s.
- ARM fp32 reference: prefill = 1.465, decode = 1.069.
- The 1.04× flip is the HX.3b headline; V3 targets ≥1.20× over HX.3b = **≥1.83 tok/s prefill**.

### 8. TRICK-1-FORWARD v1 closure — what V3 supersedes structurally

`D:\F\shannon-prime-repos\engine-trick-1-fwd\tools\sp_daemon\docs\CLOSURE-TRICK-1-FORWARD.md:62-68`:
- **D-A.1** (per-matmul cDSP-q1 || ARM-q2 split via 182 FastRPC calls/token): REJECTED for ~273 ms/token marshalling tax.
- **D-A.2** (daemon-scope substrate exerciser, ARM-q2 worker beside unchanged cDSP forward): CHOSEN for v1.
- **D-A.3 (THIS SPRINT V3):** dual HVX vector context per matmul on the SAME cDSP, q1 + q2 residue compute splits across two HVX contexts, no extra FastRPC calls. **DEFERRED to TRICK-1-FORWARD-V3 — true per-matmul split with no FastRPC tax.**

### 9. Llama.cpp HTP worker pool — existence proof of in-skel QURT thread spawn

`D:\F\shannon-prime-repos\sp-model-test\llama-cpp\ggml\src\ggml-hexagon\htp\worker-pool.c:113-153` — production-tested cDSP-side `qurt_thread_create` + `qurt_thread_attr_t` + futex-signal-wait worker pool. The pattern V3 must implement, modulo Unsigned-PD vs Signed-PD posture.

**Open structural risk:** llama.cpp's HTP backend runs in Signed PD (uses `dspqueue` + compute-res infrastructure that requires Signed PD). The Knack project's HX.3b skel runs in **Unsigned PD** (per `src/backends/hexagon/sp_hex_host.c:84-87`, `DSPRPC_CONTROL_UNSIGNED_MODULE`).

The Hexagon SDK documentation indicates `qurt_thread_create` is available in Unsigned PD with a stack limit + priority constraint. The SSR:XA attachment for HVX is also available (the FastRPC scheduler does it on the worker thread for the main handler). But **two simultaneous `qurt_hvx_lock` calls from two cDSP-internal threads under Unsigned PD have NOT been silicon-confirmed in this project.** This is the load-bearing Stage 1 question.

### 10. Discipline references

- `feedback-no-silent-gate-revisions` — gate-fail → surface UPSTREAM, never silently rewrite gate.
- `feedback-bundled-changeset-root-cause-ambiguity` — one variable at a time when iteration is cheap.
- `feedback-lead-with-reference-then-theory` — cite primitives, then design. (This Stage 0 IS that pattern.)
- `feedback-parallel-agents-separate-worktrees` — V3 is in `engine-trick-1-fwd-v3`, separate from v1's `engine-trick-1-fwd`. Confirmed.

---

## Stage 0 — UPSTREAM-surfaced architectural concerns BEFORE writing code

Per `feedback-no-silent-gate-revisions` + `feedback-lead-with-reference-then-theory`. These are surfaced in this plan-commit for operator awareness BEFORE 600-1000 LOC of cDSP-side QURT infrastructure is committed.

### Concern A — K v0.alpha primitive is DUAL-INVOKE from ARM, not "dual-HVX-context inside one invoke"

**The prompt's framing of "K v0.alpha pattern × HX.3b call sites = V3" needs structural reinterpretation.** K v0.alpha's 1.935× came from:

1. ARM-side: TWO threads × one Arc-shared FastRpcSession × one invoke each = TWO concurrent FastRPC invokes.
2. cDSP scheduler: sees two queued invokes, attaches scalar threads to SSR:XA={4,5}, runs each invoke in its own HVX context.

The current `sp_hex_forward` is ONE FastRPC invoke per prefill. Inside, ONE cDSP worker thread (the FastRPC method handler) executes 182 matmuls sequentially, holding ONE HVX context via `qurt_hvx_lock(QURT_HVX_MODE_128B)`.

**To get dual-HVX-context parallelism INSIDE one `sp_hex_forward` invoke**, the implementation must:
(a) Spawn additional cDSP-internal worker threads via `qurt_thread_create` (analog of llama.cpp's worker pool).
(b) Each worker thread independently calls `qurt_hvx_lock(QURT_HVX_MODE_128B)` (which is thread-local per `:21-22` HVX rules comment in `sp_hex_imp.c`).
(c) The QURT scheduler / SSR:XA machinery attaches each thread to its own HVX context.

This is structurally different from the K v0.alpha primitive — K v0.alpha relies on the FastRPC scheduler doing the SSR:XA work; V3 requires QURT-internal scheduling to do it for cDSP-internal-spawned threads.

**The cDSP scheduler IS still the kernel-agnostic dispatcher** (per `reference-dual-model-cdsp-scheduler`), and the SSR:XA mechanism is the SAME — the trigger surface differs (FastRPC-scheduled threads vs QURT-internal-spawned threads). Whether the scheduler attaches contexts to QURT-internal-spawned HVX-locking threads under Unsigned PD is the Stage 1 gate question.

### Concern B — chat-shape memory-bandwidth bound (the perf-lift structural risk)

Per `reference-v69-vrmpy-chat-shape-memory-bound`: at Gemma3-1B chat shape, HX.3b vrmpy is bandwidth-bound. **Dual-context parallel execution will not double bandwidth** — both contexts read activation + weight rows through the same DDR + L1 channel. The memory has observed:

> HX.3b-α-v2 dropped one of two parallel vrmpy chains (50% ALU reduction) → only 6.5% wall-clock lift. ALU throughput already had headroom. Cycle limiter is `vmem` (load weight from DDR/L1) → `vrmpy` (consume it).

**Implication for V3:** even if SSR:XA dual-context attachment succeeds AND both contexts execute in parallel, the wall-clock lift at chat shape is likely small (1.0×–1.2× range), because the bottleneck is DDR/L1 bandwidth, not vector-pipe ALU throughput.

**The 1.935× from K v0.alpha was on 128×128 / B=8 compute-bound matmul** (small enough to fit in L1, computation-dominated). Gemma3-1B chat-shape matmuls are 1152×1152 (small) or 6912×1152 (large), and the large ones dominate prefill wall.

**Honest projection:** T_TRICK1FWDV3_PERF_LIFT (≥1.20×) is structurally unlikely at chat shape. T_TRICK1FWDV3_PERF_PARITY (≥1.523) should hold by construction if V3 is correctly integrated (worst case: same throughput as HX.3b if the second context adds no value).

### Concern C — Unsigned PD QURT thread / HVX lock interaction (the Stage 1 risk)

Llama.cpp's HTP worker pool runs in Signed PD with compute-resource infrastructure. Knack's HX.3b skel runs in Unsigned PD. The specific question: **does `qurt_hvx_lock(QURT_HVX_MODE_128B)` succeed from a cDSP-internal-spawned thread in Unsigned PD, AND does the QURT scheduler attach distinct SSR:XA contexts to two concurrent HVX-locking threads?**

If the answer is "yes" — V3 proceeds. If "no" — V3 surfaces UPSTREAM as Stage 1 FAIL, names this constraint, recommends Signed PD migration as a prerequisite (out of V3 scope).

This is Stage 1's gate question and is explicitly resolved early — not buried.

### Concern D — Even if V3 ships with no perf lift, the infrastructure is load-bearing for V4

**V4 (VTCM weight pinning + 2-row-parallel kernel)** per `reference-v69-vrmpy-chat-shape-memory-bound` requires similar cDSP-side infrastructure (per-thread VTCM scratch regions, multi-context dispatch). Building V3's worker-pool + per-thread HVX-lock + per-thread activation buffer scaffold is a structural precondition for V4 even if V3 itself doesn't lift perf at chat shape.

**Disposition:** worth shipping V3 as the integration substrate, with honest documentation of why prefill-shape lift is small at chat shape and a clear V4 roadmap to attack the actual bottleneck (memory bandwidth via VTCM staging).

---

## Architectural decisions taken (D-A through D-G)

### D-A — SSR:XA attachment pattern → **D-A.3 in-skel QURT worker pool with per-thread `qurt_hvx_lock`**

Two cDSP-internal worker threads spawned via `qurt_thread_create` at `sp_hex_open()`. Each worker, on its FIRST job, calls `qurt_hvx_lock(QURT_HVX_MODE_128B)` (thread-local per HVX rule note in `sp_hex_imp.c:21-22`). The QURT scheduler attaches each thread to a distinct SSR:XA context (4 / 5 on V69). The main FastRPC handler thread does NOT call qurt_hvx_lock for the matmul portion — it dispatches matmul work to the two workers and joins. The handler's existing top-of-`sp_hex_forward` hvx_lock remains for the non-matmul work (norms / RoPE / attention — kept on handler thread).

**Alternative considered:** SSR:XA programmed by hand (per V69 expert practices' explicit SSR.XA writes). Rejected: SDK-level QURT API is more portable across V69/V73/V79 (per `reference-v69-hvx-expert-practices` "the mapping is architecture-dependent"). The SDK abstracts SSR:XA inside `qurt_hvx_lock`.

### D-B — Thread-pool lifecycle → **persistent across session, freed in `sp_hex_close`**

- Pool created in `sp_hex_open` (handle-creation time, before any matmul work).
- Two worker threads with their own 16 KB stacks (matching llama.cpp's `WORKER_THREAD_STACK_SZ = 2*16384` precedent, adjusted to V69 + Unsigned PD limits if applicable).
- Per-job signal: atomic `seqn` + futex wait (matches `worker_pool.c:38-69`).
- Pool freed in `sp_hex_close` (killed flag + futex_wake + thread_join).

### D-C — Output-row split strategy → **even split with WGATE/WUP/WDOWN remainder to thread 0**

- Thread 0 computes rows `[0, ceil(M/2))`.
- Thread 1 computes rows `[ceil(M/2), M)`.
- All shapes in Gemma3-1B have M even (QD=1152, KVD=256, E=1152, FF=6912). Odd-M code path included for safety but not hit in production.
- **Decode path (n_tok=1):** stays on single-context. The `M=1` case from a single-output-row perspective doesn't apply (we still split by output ROWS, not by tokens — every matmul has M ≥ 256). However, decode bypasses the hex backend entirely (per HX.3b closure), so this is academic. V3 splits ONLY prefill (n_tok > 1) matmuls; decode-bypass remains.

### D-D — VTCM staging → **DEFERRED to V4. Raw DDR loads in V3 v1**

VTCM pinning is the actual lever for chat-shape lift (per `reference-v69-vrmpy-chat-shape-memory-bound`). Adding it in V3 bundles two variables (dual-context + VTCM); per `feedback-bundled-changeset-root-cause-ambiguity`, keep variables separate. V3 measures: dual-context alone vs single-context baseline. V4 will measure: dual-context + VTCM vs dual-context alone.

If V3 ships with no perf lift, V4 attacks the actual bottleneck. The infrastructure built in V3 (worker pool, per-thread HVX, per-thread activation buffer) carries over to V4.

### D-E — Cache coherency → **disjoint output write regions; no atomics needed**

Thread 0 writes Y[t*out + 0 .. t*out + M/2). Thread 1 writes Y[t*out + M/2 .. t*out + M). Disjoint write regions per output row; no cross-thread synchronization on the output. Both threads READ the same activation buffer + weight buffer (read-only).

Per-thread activation quant buffer (`act_ub`): each worker needs its own buffer. Current `static unsigned char act_ub[SP_HEX_VRMPY_MAX_IN]` is shared by definition. V3 carves two buffers — one per worker, stored in per-worker context struct.

**Cache flush before main thread continues to next matmul:** the main handler reads the output rows in subsequent ops (norm, RoPE, attention). After both workers finish a matmul, the worker memory writes must be visible to the handler. QURT memory model: `qurt_thread_join`-style barriers via the futex-signal-decrement-to-zero pattern provide acquire-release semantics. Llama.cpp's worker_pool.c relies on this (`atomic_fetch_sub(&n_pending, 1)` then `while (atomic_load(&n_pending))` busy-wait in run_jobs). Same pattern in V3.

### D-F — Perf-parity gate target → **PARITY ≥ 1.523, LIFT ≥ 1.83 (1.20×) honest stretch**

- Floor: prefill ≥ 1.523 (HX.3b baseline). Decode ≥ 1.069.
- Lift target: ≥ 1.20× HX.3b = ≥ 1.83 tok/s prefill.
- Stretch (K v0.alpha pattern's silicon ceiling at compute-bound shape): 1.935× = 2.95 tok/s. **Not expected at chat shape** per Concern B.
- Honest projection: PARITY likely PASS (worst-case fallback to single-context throughput). LIFT structurally unlikely at chat shape due to memory-bandwidth bound; if it fails, V4 (VTCM) is the named follow-on, not a gate revision.

### D-G — Bit-exact gate target → **vs HX.3b baseline (decoder argmax byte-equal)**

The V3 kernel computes the same int8×int8 → int32 vrmpy dot, just split into two halves with the same per-row scale + activation-quant logic. The arithmetic is byte-identical modulo summation ORDER of horizontal-reduces (each thread does its own hsum of its half-vrmpy accumulator).

**Subtle:** the per-row computation is INDEPENDENT — each output row j has its own vrmpy chain that runs in either thread 0 or thread 1, never spanning both. So the per-row arithmetic is bit-identical to HX.3b. The 32-token greedy decode must be byte-equal to HX.3b's output (the alternating `\n` / `**` pattern per closure §"Bit-exactness verification").

If summation-order DOES cause logit drift (because the per-thread quant uses a different activation buffer; though both quant the same input with the same algorithm, identical), the gate fails → surface UPSTREAM with per-row max relerr measurement.

---

## Scope (what ships)

1. **In-skel QURT worker pool** (`src/backends/hexagon/dsp/sp_hex_imp.c` additions): two worker threads spawned at `sp_hex_open`, freed at `sp_hex_close`. Signal-wait via atomic seqno + qurt_futex. Per-worker stack + per-worker activation buffer.

2. **Dual-context matmul kernel** `hx_matmul_q8_vrmpy_dual_ctx(blk, out, in, X, n_tok, Y)`: dispatches output-row halves to the two workers, joins. Each worker computes its half using the same vrmpy logic as HX.3b's `hx_matmul_q8_vrmpy_v2`.

3. **7 call sites in `sp_hex_forward`** swapped from single-context `hx_matmul_q8_vrmpy_v2` to dual-context `hx_matmul_q8_vrmpy_dual_ctx`. One at a time per stage commit.

4. **Skel rebuild** (`hexagon_Release_toolv87_v69/libsp_hex_skel.so`). Push to S22U via existing scripts.

5. **3-rep tok/s measurement** via WIRE-HEX-FINISH harness (`timed_chat.sh` + the prompt-array methodology from CLOSURE-HX-3b.md).

6. **Closure document** with honest headline table + gate-by-gate verdicts.

---

## Substantive gates

### T_TRICK1FWDV3_DUAL_CTX_LINKED
Skel disassembly shows the two-worker matmul kernel. Methodology: `hexagon-llvm-objdump -d libsp_hex_skel.so | findstr "hx_matmul_q8_vrmpy_dual_ctx\|qurt_thread_create\|qurt_futex"`. Pass: kernel symbol present + worker pool init/dispatch primitives visible.

### T_TRICK1FWDV3_BOTH_HVX_ACTIVE
Both HVX vector contexts engaged during matmul execution. Methodology: `qurt_hvx_get_units()` reports 2; per-thread `HAP_perf_get_pcycles()` shows both workers actively burning pcycles during one matmul (overlap > 0.5). If only one worker burns pcycles, SSR:XA dual-attach failed under Unsigned PD → surface UPSTREAM as Stage 1 FAIL, name the Unsigned PD constraint.

### T_TRICK1FWDV3_DECODE_BIT_EXACT
32-token decode matches HX.3b baseline byte-equal. Methodology: same as HX.3b (`Compare-Object` on extracted delta strings). Pass: zero diff.

### T_TRICK1FWDV3_PERF_PARITY
3-rep mean prefill tok/s ≥ 1.523. Pass: observed ≥ 1.523.

### T_TRICK1FWDV3_PERF_LIFT
3-rep mean prefill tok/s ≥ 1.20× HX.3b baseline = ≥ 1.83. Stretch: ≥ 2.95 (K v0.alpha ceiling). **Honest projection: structurally unlikely at chat shape per Concern B; V4 is the named follow-on if FAIL.**

---

## Stage plan (one variable per stage; surface UPSTREAM on FAIL)

### Stage 1 — Worker-pool infrastructure + dual-context kernel standalone smoke

- Add `hx_worker_pool_t` struct + `hx_worker_init/shutdown` + `hx_worker_run_2way` (signal-wait dispatch).
- Add `hx_matmul_q8_vrmpy_dual_ctx`.
- Add a smoke test method `sp_hex_dual_ctx_smoke(remote_handle64, ...)` to `sp_hex.idl` — exercises ONE matmul through dual-context and through single-context, returns both outputs + per-thread pcycle counts for the host to compare.
- T_TRICK1FWDV3_DUAL_CTX_LINKED gate: objdump shows symbol present.
- T_TRICK1FWDV3_BOTH_HVX_ACTIVE gate: both workers' pcycle counts > 50% of single-context counts (overlap proof).

**FAIL handling:** if both_hvx_active fails (single context burning all pcycles or one worker failing `qurt_hvx_lock`), STOP. Surface UPSTREAM with the AEEResult / QURT error code. Do not proceed to Stage 2-5.

### Stage 2 — Swap WQ (layer 0) call site + decode determinism

- Swap ONE call site at `sp_hex_imp.c:587` from `hx_matmul_q8_vrmpy_v2(...)` to `hx_matmul_q8_vrmpy_dual_ctx(...)`.
- Build skel, push, run timed_chat.sh, capture decode delta strings.
- T_TRICK1FWDV3_DECODE_BIT_EXACT gate: 32-token sequence byte-equal vs HX.3b baseline.

**FAIL handling:** if decode bytes differ — fix or revert. Do not proceed without bit-exact at the one-call-site stage.

### Stage 3 — Swap remaining 6 call sites

- Swap WK, WV, WO, WGATE, WUP, WDOWN.
- Rebuild, push, re-verify decode bit-exact.
- Capture 1-rep prefill tok/s for sanity check before the 3-rep run.

### Stage 4 — 3-rep measurement + perf gates

- 3 reps each of: hex-dual-ctx (V3) and ARM-fp32-reference.
- T_TRICK1FWDV3_PERF_PARITY gate: V3 mean ≥ 1.523.
- T_TRICK1FWDV3_PERF_LIFT gate: V3 mean ≥ 1.83 (and ≥ 2.95 for stretch).

**FAIL handling per `feedback-no-silent-gate-revisions`:** if PERF_LIFT FAILS but PERF_PARITY PASSES, file V4 (VTCM pinning) explicitly. Closure tags `lat-phase-2-trick-1-forward-v3-shipped-no-lift` or similar. DO NOT rewrite the 1.20× target downward.

If PERF_PARITY FAILS (V3 slower than HX.3b due to threading overhead at small matmul sizes), file as `lat-phase-2-trick-1-forward-v3-attempted` with the per-matmul wall-clock breakdown showing where threading overhead exceeds parallelism gain.

### Stage 5 — Closure

- `tools/sp_compute_skel/docs/CLOSURE-TRICK-1-FORWARD-V3.md` with headline table, gate verdicts, honest interpretation, V4 (or V5/V6) named follow-on, files-changed, commits, worktree status.

---

## Anti-contamination commitments

- Worktree `engine-trick-1-fwd-v3` only.
- NO edits to `tools/sp_trick1/src/lib.rs` (TRICK-1 library preserved).
- NO edits to `tools/sp_compute_skel/src_dsp/sp_compute_*.c` (K.beta.2.5c, NTT.5a-c preserved).
- NO edits to `D:\F\shannon-prime-repos\engine-trick-1-fwd\` files (v1 worktree).
- NO new Cargo dependencies in the engine repo.
- Math-core submodule untouched.

---

## Build commands (reproducible)

### Stage 1+2+3 — build the skel

```powershell
cd D:\F\shannon-prime-repos\engine-trick-1-fwd-v3\src\backends\hexagon\dsp
cmd /c "..\..\..\..\scripts\env\env-hexagon.bat 1>nul 2>nul && build_cmake hexagon DSP_ARCH=v69 BUILD=Release -gMake"
```

Output: `hexagon_Release_toolv87_v69\libsp_hex_skel.so`.

### Stage 4 — push + measure

```powershell
$adb = "D:\Files\Android\pt-latest\platform-tools\adb.exe"
& $adb -s R5CT22445JA push hexagon_Release_toolv87_v69\libsp_hex_skel.so /data/local/tmp/sp22u/libsp_hex_skel.so

# 3-rep V3 prefill measurement
foreach ($rep in 1..3) {
  & $adb -s R5CT22445JA shell "pkill -f sp-daemon-wire-hex; sleep 1; sh /data/local/tmp/start_wire_hex_daemon.sh"
  Start-Sleep 5
  & $adb -s R5CT22445JA shell `
    "nohup sh /data/local/tmp/timed_chat.sh '[2,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]' 32 > /data/local/tmp/v3_rep${rep}.log 2>&1 &"
  Start-Sleep 42
  & $adb -s R5CT22445JA shell "grep -E 'FIRST_DELTA|DONE_MS|STEADY|N_TOKENS' /data/local/tmp/v3_rep${rep}.log"
}
```

### Bit-exact diff (Stage 2 / 3 gate)

```powershell
# Extract decode delta strings
& $adb -s R5CT22445JA shell `
  "grep -E 'delta' /data/local/tmp/hex_vrmpy_run.log | awk -F'\"delta\":' '{print `$2}' | awk -F',' '{print `$1}' | tr -d '\\\"'" `
  > hx3b_tokens.txt
& $adb -s R5CT22445JA shell `
  "grep -E 'delta' /data/local/tmp/v3_rep1.log | awk -F'\"delta\":' '{print `$2}' | awk -F',' '{print `$1}' | tr -d '\\\"'" `
  > v3_tokens.txt
Compare-Object (Get-Content hx3b_tokens.txt) (Get-Content v3_tokens.txt)
# expected: no output (byte-equal)
```

---

## Commit plan

```
[plan] TRICK-1-FORWARD-V3 -- dual-HVX-context per-matmul via in-skel QURT worker pool; Stage 0 reference citations + UPSTREAM concerns A-D
[Stage 1] V3 -- in-skel QURT worker pool (hx_worker_*) + hx_matmul_q8_vrmpy_dual_ctx; sp_hex_dual_ctx_smoke IDL method; T_DUAL_CTX_LINKED + T_BOTH_HVX_ACTIVE
[Stage 2] V3 -- WQ call site swapped (layer 0 + all layers) to dual-context; T_DECODE_BIT_EXACT verified vs HX.3b
[Stage 3] V3 -- WK/WV/WO/WGATE/WUP/WDOWN swapped to dual-context; T_DECODE_BIT_EXACT re-verified
[Stage 4] V3 -- 3-rep prefill measurement; T_PERF_PARITY + T_PERF_LIFT verdicts (honest, no gate revisions)
[Stage 5] V3 -- closure + sub-tag candidate
```

Push: `git push -u origin sprint/trick-1-forward-v3`.

---

## Honest framing

This sprint integrates the K v0.alpha dual-HVX-context primitive into the production forward at the only level that doesn't escalate FastRPC marshalling cost (cDSP-internal worker pool with per-thread qurt_hvx_lock + SSR:XA scheduler attach). The infrastructure is the V4 substrate — even if V3 itself ships with parity-only-no-lift, V4 (VTCM weight pinning + 2-row-parallel) will use the same worker pool.

The honest perf-lift projection is modest (≥1.0, likely ≤1.2) because Gemma3-1B chat shape is memory-bandwidth-bound. The V4 follow-on attacks the actual bottleneck. The 1.935× K v0.alpha ceiling is the compute-bound benchmark ceiling, not the chat-shape ceiling.

If T_BOTH_HVX_ACTIVE fails under Unsigned PD (Stage 1), V3 is structurally blocked at the SDK / privilege layer, surfaces UPSTREAM with the QURT error, and recommends Signed PD migration as a prerequisite — not within V3 scope.

No silent gate revisions. No bundled changesets unless explicitly justified. Per-stage commits with one-variable changes. The closure honestly attributes the wall-clock breakdown.
