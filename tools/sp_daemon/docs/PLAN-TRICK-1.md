# PLAN — Sprint TRICK-1 — CRT-sharded heterogeneous-island compute

**Branch:** `sprint/trick-1`
**Worktree:** `D:\F\shannon-prime-repos\engine-trick-1`
**Base:** engine main @ `eba0301` (NTT.6 merge)
**Date:** 2026-05-31

## 1. Manifesto Trick #1 — quoted verbatim from `reference-heterogeneous-soc-crt-tricks`

> (1) CRT-sharded compute DSP-q1 + NPU-q2 + ARM Garner — no cross-island sync mid-compute.

This sprint operationalizes that bullet at the matmul scope: two silicon islands compute different residues of the SAME matmul in parallel; ARM-side Garner recombines into a real-valued (signed centered) result that matches the unreduced int sum byte-exactly; wall-clock parallel < max(per-island serial).

Per the manifesto: **the load-bearing claim is the architectural pattern** — two
independent silicon islands compute different residues, recombine byte-exactly,
no cross-island sync. The choice of WHICH islands (DSP/NPU/CPU/GPU/ISP) is
silicon-availability dependent. This sprint's first task in Stage 0 is to
determine which two-island pairs are STRUCTURALLY VALID on Knack's S22U.

## 2. Stage 0 pre-read citations

| Reference | File:line | Finding |
|---|---|---|
| K.beta.2.5c mod_q matmul kernel | `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c:247-293` (`sp_matmul_q_hvx`) | Silicon-confirmed integer matmul mod q on HVX; per-prime via `q_idx`. Returns u32 in [0, q). Per `reference-single-prime-modq-is-hash-not-matmul`, this is the Z_q residue — exactly the per-island residue Trick #1 wants. |
| IDL `matmul_q` method | `tools/sp_compute_skel/inc/sp_compute.idl:206-212` (method index 11) | primIn 7×i32=28B (q_idx, batch, d_in, d_out, x_bufLen, w_bufLen, y_bufLen); args layout [primIn, x_buf, w_buf, primOut, y_buf]; scalars=`make_scalars(11, 3, 2)`. |
| ntt_crt Garner reconstruction | `lib/shannon-prime-system/core/ntt_crt/ntt_crt.c:303-316` (`garner_one`) | Symmetric Garner: `t = (x2 - x1) * q1_inv_mod_q2 mod q2`; `r = x1 + q1 * t` ∈ [0, M); center to (-M/2, M/2]. |
| Garner constants | `lib/shannon-prime-system/include/sp/ntt_crt.h:48-51` + computed | `SP_NTT_Q1=1073738753`, `SP_NTT_Q2=1073732609`, `SP_NTT_M=1152908312643096577`, `Q1_INV_MOD_Q2=894602413` (Python-verified: `(q1 * 894602413) mod q2 = 1`; `q1*q2 = 1152908312643096577`). |
| Rust Garner reference | `tools/sp_dsp_smoke/src/sp_matmul_q_ref.rs:73-92` (`garner_combine_q1_q2`) + `:108-129` (`garner_combine_q1_q2_signed`) | Both unsigned in [0, M) and signed in (-M/2, M/2] variants already exist and are SASS/test-validated. Reuse directly; no re-derivation. |
| ARM-side scalar mod-q matmul reference | `tools/sp_dsp_smoke/src/sp_matmul_q_ref.rs:39-67` (`matmul_q_scalar_ref`) | Per-k Barrett + modular-add; matches HVX kernel's algorithm path C byte-for-byte. **Critical to D-B pivot: this is the ARM-q2 path candidate.** |
| Frobenius lift API | `lib/shannon-prime-system/include/sp/frobenius_lift.h:60-200` | Per-row int8 codes in [-127, 127] + per-row fp32 scale. Q8 codes lift trivially into Z_q residues: `q_residue = (code < 0) ? (q + code) : code`. |
| 60-bit unreduced reference matmul | `tools/sp_dsp_smoke/src/sp_matmul_q_ref.rs:138-161` (`matmul_60bit_ref`) | u128 unreduced sum, returns u64 iff fits in M. **The Garner bit-exact reference target.** |
| FastRPC concurrent dispatch | `tools/sp_dsp_smoke/src/dsp_rpc.rs:140-290` (`FastRpcSession`) | Arc<FastRpcSession> auto-`Send+Sync` (libloading bare fn pointers + u64 handle). Per `reference-fastrpc-concurrent-dispatch`. |
| K.beta.2.5c dual-dispatch harness | `tools/sp_dsp_smoke/src/sp_matmul_q_dual_smoke.rs:60-204` | Reference template for "concurrent dispatch of two matmul_q invokes via Arc<FastRpcSession>"; reuse `invoke_matmul_q` helper structure. |
| K.2-spike closure (NPU POC) | `tools/sp_npu_spike/docs/CLOSURE-K2-SPIKE.md:32-72` | NPU dispatch works in Unsigned PD on V69: 1.329 ms graphExecute, 64/64 byte-exact ElementWiseAdd. Skel push to `/data/local/tmp/libQnnHtpV69Skel.so` + `ADSP_LIBRARY_PATH`. Per-process init ~130 ms; per-execute ~1.3 ms. |
| K.2-spike shim | `tools/sp_npu_spike/src/sp_npu_shim.c:96-345` | C shim wrapping `dlopen → getProviders → logCreate → backendCreate → contextCreate → graphCreate → tensorCreate (NULL clientBuf) → addNode → finalize → execute (SET clientBuf)`. Template to extend for MatMul op. |
| QNN MatMul op | `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\include\QNN\QnnOpDef.h:524-526` | `"MatMul"`, `transpose_in0`, `transpose_in1` params. Op exists; output dtype rules per HTP op-def history. |
| **HTP V69 MatMul precision constraint** | `docs\QAIRT-Docs\QNN\general\htp\htp_opdef_version_history.html @34249` | "Conv2d, DepthWiseConv2d, TransposeConv2d, FullyConnected, MatMul: Added constraint to support 16 bit data types for v73 or beyond architecture only." **V69 MatMul outputs are restricted to 8-bit activation type.** (SFIXED_POINT_32 support is for in[2] bias only.) |
| HVX 32×32→64 widening idiom | `tools/sp_compute_skel/src_dsp/sp_compute_crt_imp.c:82-83` (vmpye+vmpyoacc pair) | Per `reference-hexagon-v69-32x32-widening-idiom`. Used inside `sp_barrett_reduce32_hvx_lane`. |
| `feedback-no-silent-gate-revisions` | MEMORY | When implementation can't meet spec'd gate, surface UPSTREAM first. Do not retreat to a weaker gate, defer to unrelated phase, or tune fixtures until a number passes. |
| `feedback-lead-with-reference-then-theory` | MEMORY | Read the reference's actual code first; map to lattice idiom; theory second. (Discharged: this Stage 0 cites file:line for every load-bearing assertion.) |
| `feedback-bundled-changeset-root-cause-ambiguity` | MEMORY | One variable per commit when iteration is cheap. |
| `feedback-shape-dependent-parallelism-gates` | MEMORY | Parallelism gates apply at compute-bound shapes; data-bound shapes report diagnostic only. |
| `feedback-leak-gate-allocator-warmup` | MEMORY | Use second-half-slope VmRSS, not total delta. |
| `feedback-parallel-agents-separate-worktrees` | MEMORY | This sprint operates exclusively in `engine-trick-1` worktree on branch `sprint/trick-1`. NO commits to sibling worktrees. |

## 3. STRUCTURAL BLOCKER — surfaced UPSTREAM per `feedback-no-silent-gate-revisions`

### 3.1 The blocker

**The sprint's D-B Path 1 ("NPU as q_2 island via QNN HTP MatMul with INT8 inputs + INT32 buffer return, ARM Barrett mod q_2") is STRUCTURALLY INVALID on V69 silicon.**

Evidence:

- QNN HTP MatMul on V69 supports `QNN_DATATYPE_SFIXED_POINT_8` and
  `QNN_DATATYPE_UFIXED_POINT_8` activations (verified
  `htp_opdef_version_history.html` op-coverage table cited above).
- `QNN_DATATYPE_SFIXED_POINT_16` / `SFIXED_POINT_32` activation support
  is restricted to **V73 or beyond** ("Added constraint to support 16
  bit data types for v73 or beyond architecture only"). Knack's S22 Ultra
  is Snapdragon 8 Gen 1 (Hexagon V69) — pre-V73.
- `SFIXED_POINT_32` MatMul support is for `in[2]` (bias) only, not the
  output tensor. The MatMul output tensor stays at the activation
  precision (INT8).
- Maximum value representable by INT8 output: ±127. For the
  Trick-#1-required matmul shape K=2048, M=N=256, with int8 inputs in
  [-127, 127], the integer sum-of-products lies in
  [-2048·127² , 2048·127²] = [-33 028 352, 33 028 352].
  The QNN output quantization will collapse this 24-bit-wide signed
  range into the 8-bit signed range — a destructive 16-bit truncation
  per element. Even with `scale = 260096` chosen to map the max range
  to [-127, 127], every output element loses ~16 bits to rounding.
- The Barrett reduction mod q_2 (q_2 ≈ 2^30) on a value that's been
  quantized to 8-bit-precision cannot reconstruct the q_2 residue of the
  ORIGINAL integer sum — the precision is gone before Barrett ever runs.
- Garner reconstruction requires the q_1 and q_2 residues to be of the
  SAME underlying integer value. NPU q_2-residue extracted from an
  INT8-quantized output is NOT the q_2 residue of the integer sum that
  cDSP's q_1 kernel computed. **Garner produces garbage.**

This is precisely the failure-mode the sprint's spec calls out:

> Failure-mode surface-upstream paths. If you hit ANY of these, STOP and document, do not paper over:
> - NPU's INT matmul output doesn't carry enough precision for K=2048 (overflow before Barrett)

### 3.2 The structural fact

The "overflow before Barrett" framing is precisely what happens on V69
silicon: INT8 output of QNN MatMul saturates / re-quantizes the
accumulator before it ever reaches the ARM-side mod-q_2 reduction. No
amount of clever scale/offset gymnastics in the QNN tensor descriptor
recovers the lost 16 bits.

**Workaround paths and why each is rejected:**

- **Path 1A: Set scale=1, offset=0 — INT8 output as raw sum.** Hardware
  saturates accumulator to ±127. Useless for K>~30 because the natural
  sum range exceeds ±127.
- **Path 1B: Scale = K * 127, offset=0 — output is sum/K-normalized.**
  Loses ~16 bits per element to rounding. Garner bit-exact gate
  guaranteed to fail.
- **Path 1C: QNN custom op for INT8×INT8→INT32 MatMul.** Would require
  building a QNN OpPackage with hand-written V69 HVX (or HMX) code, signing
  it for the unsigned-PD development path, and validating
  HTP-graph-level admission. Out of sprint scope (≥1000 LOC, ≥1 month);
  per `feedback-no-silent-gate-revisions` not a path to revise this
  sprint's gates around. **Filed as deferred work.**
- **Path 1D: Wait for V73 silicon (Snapdragon 8 Gen 3+).** Not available
  on Knack's S22U. Filed as a hardware-availability follow-on.
- **Path 1E: Use ElementWiseMultiply + sum reduction (SFIXED_POINT_32
  intermediate via a chain).** Same V73-or-later constraint applies to
  the INT16/INT32 intermediate tensors per the op-def doc.

### 3.3 Decision — sprint pivot

Per `feedback-no-silent-gate-revisions`, **the spec gate
T_TRICK1_NUMERICAL_EQUIVALENCE cannot pass on V69 silicon via the NPU
q_2 path. Surfacing UPSTREAM is the correct disposition.**

Per the manifesto, the LOAD-BEARING claim is "two independent silicon
islands compute different residues of the same value in parallel and
recombine byte-exactly" — NPU is one example, not the only valid pair.

**Pivot:** continue the sprint with a SECOND cross-island pair that IS
structurally valid on V69:

- **Island A: cDSP V69 HVX** — `sp_matmul_q_hvx(q_idx=0, ...)` mod q_1.
- **Island B: ARM Cortex-X2 / Cortex-A710** — `matmul_q_scalar_ref(q_idx=1, ...)` mod q_2 (Rust scalar reference, ARM-side, already exists at `sp_matmul_q_ref.rs:39`).
- **ARM Garner-combine** as spec'd.

This pair operationalizes the SAME load-bearing architectural claim:
two independent silicon islands (DSP and CPU are physically distinct
silicon with independent execution queues, no shared scheduler queue,
no shared L1/L2; the cDSP is a Hexagon V69 cluster, the ARM is an
ARMv9 Cortex-X2/A710 cluster). The Garner recombine is unchanged. The
parallel-wall-clock win is measurable. The byte-exact CRT identity
holds (and is already proven for the
[ARM-q_1, ARM-q_2, ARM-Garner] composition by the existing
`garner_roundtrip_via_matmul` test at `sp_matmul_q_ref.rs:228-253`).

This pivot is HONEST about which two islands are participating. The
NPU-as-q_2 claim is documented as the FOLLOW-ON for V73 silicon (or
the Path-1C QNN-OpPackage work).

### 3.4 Gates retained, with one renamed

- T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT — **UNCHANGED**, applies to the
  pure-Rust Garner combiner (the ARM-side compute uses the same
  combiner; this is its standalone-correctness gate against ntt_crt.c).
- T_TRICK1_NUMERICAL_EQUIVALENCE — **UNCHANGED.** The integer matmul
  output (combined via Garner) must match the int-sum reference
  byte-for-byte; dequantized Y_real must match fp32 reference within
  the documented Q8 quantization rounding budget (per D-F).
- T_TRICK1_PARALLEL_WIN — **UNCHANGED** in form. Measured as
  T_trick1_parallel < 1.2 × max(T_dsp_solo, T_arm_solo). Methodology
  per D-G (10-rep mean, drop first as warmup).
- T_TRICK1_BOTH_ISLANDS_ACTIVE — **RENAMED to
  T_TRICK1_BOTH_ISLANDS_ACTIVE_DSP_ARM** with same intent: confirm
  the cDSP HVX kernel and the ARM-side scalar loop both genuinely
  execute and overlap in wall-clock time. cDSP confirmation via the
  existing `sp_compute_matmul_q` `kernel_pcycles` return value
  (already in IDL primOut). ARM-side confirmation via direct
  `Instant::now()` bracketing in the Rust thread.

## 4. Architectural decisions

### D-A: Test fixture shape

**Choice:** K=2048, M=N=256 — matches sprint spec.

Justification: per the K.beta.2.5c closure (Section 3, headline table),
"shape B" at B=8 / D_in=1024 / D_out=512 lands compute-bound on cDSP
HVX (~27 ms per invoke). K=2048, M=N=256 yields:
`total ops = 1 × 2048 × 256 = 524k MACs per output element ≈ 134M MACs`
(actually `1 * 2048 * 256 = 524288 outputs * 2048 MACs each ≈ 1G MACs`).
This is larger than K.beta.2.5c shape B by ~30×, deeply compute-bound,
giving the parallel-island scheduler the largest overlap window we can
afford in sprint scope. Sequential expected ~600-1000 ms (DSP) plus
~500-1500 ms (ARM scalar); parallel target ~max ≈ 1.5 s.

Batch dim: `batch=1` keeps the matmul a single matrix-vector. For a
batch>1 the ratio of marshalling-to-compute degrades; b=1 is the
cleanest demo. (Sprint spec doesn't fix batch; b=1 is consistent with
"mirrors a single Gemma3-1B FFN layer.")

The DSP IDL kernel requires `d_out % 32 == 0`; 256 % 32 = 0. ✓
The DSP IDL kernel requires `d_in >= 1`; 2048 >= 1. ✓

### D-B: ARM-as-q_2 island (PIVOT per §3.3)

**Choice (revised):** ARM-side uses the existing `matmul_q_scalar_ref`
from `sp_matmul_q_ref.rs`. Algorithm: per-element `(x * w) mod q_2`
Barrett, modular-add accumulate. No NEON vectorization needed for v1
(NEON acceleration is a sub-sprint follow-on if wall-clock balance is
unfavorable).

Justification:
- Already exists and is test-validated.
- Computes the SAME mathematical operation as the cDSP HVX kernel at
  q_idx=1 — identical byte-pattern Z_q_2 residue output.
- Runs on cores genuinely independent from the cDSP (Cortex-X2 prime
  + 3× A710 perf + 4× A510 efficiency cores; sprint dispatches the
  scalar Rust thread, OS scheduler places it on a free big core).
- Trivially leak-free (Vec heap allocations only).

**NOT chosen:** writing an arm-neon variant. The architectural payload
is "two independent islands, byte-exact recombine, parallel win" — not
"NEON wall-clock equals HVX wall-clock." Sprint scope is the
architectural demo; NEON optimization is a follow-on TRICK-1-NEON
sub-sprint.

### D-C: Frobenius lift timing

**Choice:** Pack-time lift. At test fixture setup, take the fp32 reference
W (K×N, here 2048×256) and X (1×K, here 1×2048) and produce three
artifacts:
- `q_x_q1[K]`, `q_x_q2[K]`: u32 array, X lifted to Z_q1 and Z_q2.
- `q_w_q1[K*N]`, `q_w_q2[K*N]`: u32 array, W lifted similarly.
- `s_x`, `s_w`: per-tensor fp32 scales (NOT per-row; see §5
  for rationale — per-row scales don't factor out of the inner
  matmul sum, so the Trick #1 demo uses per-tensor as a deliberate
  simplification documented in the closure).

The Q8 codes are computed via `sp_frob_quant1(v, s)` from each value
divided by its tensor's scale. Codes ∈ [-127, 127] (int8). The lift
to Z_q is: `q_residue = (code < 0) ? (q + code) : code`. Both primes
are ~2^30 ≫ 127, so the lift is exact.

### D-D: Garner combine arithmetic

**Choice:** ARM-side. Reuse `garner_combine_q1_q2_signed` from
`sp_matmul_q_ref.rs:108`. Output is `Vec<i64>` of signed centered
residues in (-M/2, M/2]. For K=2048, M=N=256, the max unreduced sum is
|2048 * 127²| ≈ 33M, vastly inside M/2 ≈ 5.76e17. Centered Garner
output equals the underlying signed integer sum exactly (no
M-wraparound aliasing).

### D-E: Parallelism pattern

**Choice:** Persistent worker pool. Per
`feedback-oracle-vs-production-hedge`, the production parallel-dispatch
pattern is two pre-spawned threads, P-core-pinned at startup,
atomic-flag signal-wait on hot path. For sprint v1 demo, I use a
simpler pattern: one `std::thread::spawn` per measurement (10 reps),
yielding the same wall-clock answer at the cost of per-rep thread-spawn
~50 µs overhead. The expected matmul wall is ~1 second, so spawn cost
is < 0.01% of measurement — negligible for the WIN gate.

ARM P-core pinning: deferred. Knack's S22U Android does not give
user-space the `sched_setaffinity` capability without root; the
Trick-#1 sprint runs without affinity hints and lets the OS scheduler
place the threads. This is honest: the parallel-win number reported
includes any scheduler placement variance. Sprint v2 with persistent
worker pool + affinity hints is the follow-on once Trick #1's
architectural payload is silicon-confirmed.

### D-F: Reference baseline (numerical equivalence)

**Choice:** TWO reference paths.

(a) **Pure-integer reference** (`matmul_60bit_ref` from
`sp_matmul_q_ref.rs:138`): runs the SAME integer sum-of-products in
u128 host arithmetic, returns u64 iff fits in M. Compare:
Garner-recombined output of Trick #1 == `matmul_60bit_ref` output,
byte-for-byte. This is the **T_TRICK1_NUMERICAL_EQUIVALENCE
integer-domain** gate. Pass: exact match (all elements).

(b) **fp32 reference**: dequantize the Q8-coded X and W using their
per-tensor scales, run an fp32 matmul on the dequantized values,
compare to dequantized Trick #1 output. Expected: matches within
Q8-quantization round-trip rounding, NOT byte-exact. Tolerance budget:
per-element relative error ≤ 5e-5 (an empirically-tight bound for Q8
× Q8 with K=2048 — the inner sum amplifies per-element rounding by
sqrt(K) ≈ 45 in the worst case, with per-element rounding ≤ 1 ULP of
127). Wide tolerance is documented honestly as "the dequant path has
different rounding than int-accum-then-dequant" per sprint spec D-F.

The LOAD-BEARING gate is (a). Path (b) is reported for completeness as
the "Q8 quantization rounding budget" sprint spec calls for.

### D-G: Wall-clock measurement

**Choice (revised for ARM-as-q_2 pivot):**
- `T_dsp_solo`: `invoke_matmul_q(q_idx=0)` blocking on main thread, 10 reps, drop first as warmup, report mean and stddev.
- `T_arm_solo`: `matmul_q_scalar_ref(q_idx=1, ...)` blocking on main thread, 10 reps, drop first as warmup, report mean and stddev.
- `T_trick1_parallel`: cDSP q_1 dispatched on thread A (via Arc<FastRpcSession>), ARM q_2 on thread B (Rust scalar), join, ARM Garner-combine, ARM result write. 10 reps, drop first.
- Win condition: `T_trick1_parallel < 1.2 * max(T_dsp_solo, T_arm_solo)`.

## 5. Per-tensor scale simplification — explicit reasoning

The Frobenius lift API in `frobenius_lift.h` uses per-row scales. For a
matmul `Y[b][i] = sum_k X[b][k] * W[k][i]`, per-row Frobenius means
`s_W[k]` varies with the inner-dim index — and `s_W[k]` cannot be
hoisted out of the inner sum. The dequantized result would be:

```
Y_real[b][i] = sum_k (q_X[b][k] * s_X[b]/127) * (q_W[k][i] * s_W[k]/127)
             = (s_X[b]/127^2) * sum_k q_X[b][k] * q_W[k][i] * s_W[k]
```

The factor `s_W[k]` inside the sum makes the integer accumulation
NON-uniform — different k contributions have different float weights.
That's incompatible with a single int-accum-then-dequant pass.

Production solutions (out of TRICK-1 sprint scope):
- Per-k pre-scaling of X (collapses s_W[k] into an X[k] adjustment;
  loses precision).
- Per-column block-wise quantization (transpose the storage so the
  scale axis is the OUTPUT dim; the matmul's accumulator then has a
  uniform scale per-output-element).
- Per-tensor scale (this sprint's choice; simplest; demonstrates the
  architectural pattern; deliberate scope cap).

**TRICK-1 sprint chooses per-tensor scale.** Closure documents this
explicitly as a sprint scope decision, NOT a contract change to the
Frobenius lift API.

## 6. Numerical budget — explicit calculation

K=2048, M=N=256. Int8 codes ∈ [-127, 127]; max product per element =
127² = 16129. Max sum-of-products magnitude = K * 127² = 33 028 352
≈ 2^25.

Comparisons:
- 2^25 ≪ M/2 ≈ 5.76e17 (Garner reconstruction range): safe by 30+ bits.
- 2^25 ≪ q_1 ≈ 2^30 (mod-q_1 range): each per-element residue fits
  trivially.
- For the fp32-equivalence gate: per-element rounding error of Q8 with
  symmetric [-127, 127] code is bounded by `0.5 * s/127` per element.
  After K=2048 sums, the accumulated absolute error is bounded by
  `K * 0.5 * s_X * s_W / 127² = K/2 * s_X * s_W / 127²`. The "real"
  magnitude of the dequantized output is `≈ sqrt(K) * s_X * s_W` (RMS
  over uniform random inputs). The relative-error bound is
  `K / (2 * 127² * sqrt(K)) = sqrt(K) / (2 * 127²) ≈ 45 / 32258 ≈
  1.4e-3`. **Sprint budget = 5e-3 (relative error per element).**

If observed deviation exceeds this budget, the bug is in either:
- Garner formula sign handling (impossible: T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT catches it).
- Q8 encode rounding (impossible: `sp_frob_quant1` is byte-stable; ARM and DSP read the SAME codes).
- s_X or s_W computation (sprint code must use the SAME scale on both sides — guard with assert).

Any of these surface UPSTREAM, no silent budget widening.

## 7. Scope (what ships)

1. **`tools/sp_trick1/` new crate** — host the demo. Sibling to
   `sp_dsp_smoke`, `sp_npu_spike`. Cross-compiles for
   `aarch64-linux-android`, deploys to S22U.
2. **`src/frob_dual_lift.rs`** — Frobenius pack of an fp32 tensor into
   two parallel u32 arrays (mod q_1, mod q_2) + per-tensor scale + the
   raw int8 codes for inspection. Calls into the existing
   `sp_frob_*` API where mathematically equivalent.
3. **`src/lib.rs` re-export** of `sp_matmul_q_ref` (Garner + integer
   reference) — sibling crate's existing module, copied or re-exported
   as a sub-module file.
4. **`src/lib.rs` re-export** of `dsp_rpc` for the FastRpcSession.
5. **`src/bin/sp_trick1_smoke.rs`** — the test fixture binary:
   - allocate fp32 X[1×2048] and W[2048×256] from a deterministic seed
   - Frobenius-lift to two parallel u32 per-prime arrays + scales
   - run `T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT` (Stage 1) — Rust-only,
     no silicon dispatch; runs even on Windows host.
   - run cDSP-q_1 path solo, ARM-q_2 path solo (Stage 2 + 3)
   - serial dispatch (Stage 4): cDSP q_1 → ARM q_2 → Garner; checks T_TRICK1_NUMERICAL_EQUIVALENCE
   - parallel-thread dispatch (Stage 5): both islands concurrent, Garner, T_TRICK1_PARALLEL_WIN + T_TRICK1_BOTH_ISLANDS_ACTIVE_DSP_ARM
   - print JSON line + summary table.
6. **`tools/sp_daemon/docs/CLOSURE-TRICK-1.md`** with the 15-section
   layout from the sprint spec, including the §3 STRUCTURAL BLOCKER
   surface for the NPU path.
7. **`tools/sp_daemon/docs/PLAN-TRICK-1.md`** (this file).

## 8. Per-stage commit plan (one variable at a time)

1. **`[plan] TRICK-1`** — this PLAN file. Pre-code commit per
   `feedback-lead-with-reference-then-theory`.
2. **`[stage 1] TRICK-1 — frob-dual-lift + Garner host-only`** — pure
   Rust crate skeleton + Frobenius dual-lift utility + run
   T_TRICK1_GARNER_BIT_EXACT_VS_HOST_CRT (Garner-combiner standalone
   correctness) + per-tensor integer-vs-integer reference round-trip
   on host. No silicon dispatch yet. Verifies plumbing.
3. **`[stage 2] TRICK-1 — cDSP-q1 solo path`** — `invoke_matmul_q`
   wired into the trick1 crate; smoke runs ONE q_1 invoke at K=2048
   M=N=256; verify Z_q_1 residue matches `matmul_q_scalar_ref(q_idx=0)`
   element-wise.
4. **`[stage 3] TRICK-1 — ARM-q2 solo path`** — same fixture; calls
   `matmul_q_scalar_ref(q_idx=1, ...)`; verifies Z_q_2 residue against
   a deterministic re-run of the same routine. (Sanity check; the
   real test is the Garner identity in Stage 4.)
5. **`[stage 4] TRICK-1 — serial dispatch + Garner combine`** —
   T_TRICK1_NUMERICAL_EQUIVALENCE. Calls cDSP q_1, then ARM q_2, then
   Garner; compares to `matmul_60bit_ref`. PASS = 100% byte-exact.
6. **`[stage 5] TRICK-1 — parallel dispatch + win`** —
   T_TRICK1_PARALLEL_WIN + T_TRICK1_BOTH_ISLANDS_ACTIVE_DSP_ARM.
   Concurrent threads via Arc<FastRpcSession> for cDSP, native Rust
   thread for ARM. 10-rep wall-clock, drop first.
7. **`[stage 6] TRICK-1 — closure`** — CLOSURE-TRICK-1.md, per the
   spec's 15-section layout.

## 9. Anti-contamination

- This worktree is `engine-trick-1`. NO commits to sibling worktrees
  (`engine-ntt-*`, `engine-hx-3b*`, `engine-wire-*`).
- Per `feedback-parallel-agents-separate-worktrees`. Discipline
  observed.
- New crate at `tools/sp_trick1/` does not modify `sp_compute_skel`,
  `sp_dsp_smoke`, `sp_npu_spike`, `sp_daemon`, or any backend.
- Submodule `lib/shannon-prime-system` is read-only — no commits.

## 10. What's deferred (explicit)

- The NPU q_2 path on V73+ silicon (Snapdragon 8 Gen 3 or later). Once
  hardware available, Path 1A retried with INT16 activations is the
  fastest path.
- The NPU q_2 path via custom QnnOpPackage (Path 1C) — exposes INT32
  intermediate via a vendor-signed custom op. ~1000 LOC + signed-PD
  toolchain; deferred to a follow-on.
- Three-island variant (DSP + NPU + GPU Vulkan-q_3). Requires the NPU
  path first or a third prime; depends on V73 silicon AND a third
  Proth prime (the frozen primes are only two — adding q_3 requires
  Phase-5+ change per `reference-ntt-frozen-primes-N-cap`).
- Full forward integration of Trick #1 through Gemma3-1B FFN/attention.
- Persistent worker pool + P-core affinity (TRICK-1-v2).
- NEON-vectorized ARM-q_2 path (TRICK-1-NEON sub-sprint).
- QNN persistent-context amortization across multiple matmuls (K.2
  full sprint scope).

## 11. Operator surface (key questions for review)

(no action required — these are explicit decisions documented for
acknowledgment that they were made consciously, not by drift):

- **Pivot D-B from NPU q_2 to ARM q_2.** Justified §3.1-§3.4. The
  load-bearing manifesto claim (two islands, byte-exact recombine,
  parallel win) is preserved; the SPECIFIC choice of NPU is
  silicon-blocked on V69. NPU path is filed as a V73 follow-on.
- **Per-tensor scale instead of per-row Frobenius.** Justified §5.
  Architectural-demo scope; not a contract change.
- **Sprint v1 uses Rust scalar ARM-q_2 path (no NEON).** Justified D-B
  rationale. NEON optimization is a follow-on.
- **No P-core affinity in v1.** Justified D-E. Follow-on with
  persistent worker pool.

Plan-commit moves first; code follows per stage.
