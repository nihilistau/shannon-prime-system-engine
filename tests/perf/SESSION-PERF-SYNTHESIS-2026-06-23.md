---
type: finding
title: "Diffusion-judge performance — the whole-machine synthesis (NUC11BTMi9)"
description: "Measured map of every hardware asset to its correct role, the three isolated probes (pinned/profile/RVQ), and the integrated end-to-end cache config. Honest physics on the iGPU/Optane/CRT heterogeneous pitch."
tags: [perf, diffusion-judge, moe, pcie, optane, igpu, crt, integrated]
timestamp: 2026-06-23
resource: src/backends/cuda/cuda_forward.cu
sp_status: green-committed (engine 1d0e414)
sp_gate: "T1 pinned-bw / T2 dmon profile / T3 RVQ-depth sweep = DONE; T5 integrated sweep = RUNNING"
sp_commit: pending
sp_repro: "_pcie_bw.cu ; _t2_run.bat + nvidia-smi dmon ; /tmp/rvq_sweep.py ; _integrated_sweep.bat"
---

# The whole-machine synthesis — diffusion judge on the NUC

**The operator's thesis (correct, and the spine of this doc):** components that look bad in
isolation can be the right move once the whole system runs end-to-end with every flag in its
right spot. This session I stopped measuring parts and ran the integrated stack. This doc maps
EVERY asset to its correct role, backed by receipts, and corrects the heterogeneous-compute
pitch where it collides with the actual silicon.

## 0. The machine (measured, not assumed)

| asset | what it actually is | measured |
|---|---|---|
| CPU | Intel i9-11900KB (Tiger Lake-H, 8c) | NUC11BTMi9 "Beast Canyon" |
| dGPU | RTX 2060 12 GB, sm_75 | **PCIe gen3 x8** (~6.2 GB/s, T1) |
| iGPU | Intel UHD Graphics (Tiger Lake **Xe-LP, 32 EU, NO XMX**) | ~0.75 TFLOPS fp32 / dp4a int8 only |
| RAM | 31.5 GB DDR4-2667 (2x16) | ~40 GB/s, holds the whole 14 GB model |
| Optane | MEMPEK1W 016GA + 032GA = **Optane Memory M10** | enumerate as **block NVMe disks**; `Get-PmemDisk`=none |

Two facts that the heterogeneous pitch got wrong on physics:
- **The 2060 is on PCIe gen3 x8, not x16, and not "Gen5 128 GB/s."** Real ceiling ~6.2 GB/s.
- **The Optanes are M10 block-cache SSDs, NOT app-direct PMEM.** No DAX-mmap-as-RAM exists on
  this box. "Compute out of Optane at memory-bus speed" is physically impossible here; M10 read
  (~1 GB/s) is SLOWER than the PCIe path it was meant to replace.

## 1. The three isolated probes (receipts)

**T1 — pinned vs pageable H2D (`_pcie_bw.cu`).** On gen3-x8: 16 MB 1.18x, 64 MB 1.07x,
256 MB **1.03x**. Pageable already saturates the bus; pinned's *bandwidth* gain is a rounding
error. Pinned's REAL value is async overlap (untested by a pure-bandwidth probe) — see §3.

**T2 — dmon profile, one item, 52 s (STEPS=4, no cache flags).** SM **55%**, VRAM **12%**,
PCIe **~1.7 GB/s = 28%** of ceiling. ~88 GB streamed = the model ~4x (once per denoise step)
with NO resident cache (RESERVOIR was default-off). Nothing saturated -> the wall is
**serialization / re-streaming**, not raw bandwidth and not raw compute.

**T3 — RVQ depth sweep on real injected episodes (`/tmp/rvq_sweep.py`, audio+wiki).**
The O_K Frobenius rank-2 codec we already ship IS residual VQ (level-1 `a` + residual `b`).
At fixed bit budget, going deeper does NOT help: 16-bit depth-1 `a16` relL2 2.1e-5 ==
depth-2 `a8b8` 2.1e-5 >= depth-3 `a8b4c4` 2.7e-5 (worse), and each level adds a per-(L,ch)
scale sidecar (pure tax). The gated `a16b8` is already sub-ULP (8.6e-8) on the frontier.
**RVQ depth = honest negative.** Only real knob is bit-WIDTH (`a8b8` for cold audio episodes
saves ~1.5x on the Optane episode tier if capacity bites). A learned shared codebook would cut
the sidecar tax but reintroduces float centroids = breaks byte-exactness. Don't.

## 2. Everything in its right spot (the honest tiering — "use it all")

- **2060 VRAM (12 GB)** = hot compute: prefix-KV (~6 GB) + canvas dp4a GEMMs + the RESIDENT
  expert head (RESERVOIR/WCACHE). This is the fast silicon; keep the heavy math here.
- **System DRAM (31.5 GB)** = the full 14 GB model, PINNED, as the spillover tier. Async
  double-buffer the experts that don't fit VRAM (overlap upload with compute) — the T1/T2 lever.
- **Optane M10 (48 GB block)** = the **XBAR KV episode store** (random-access recall). Its
  correct, already-assigned role. NOT a weight tier (it's slower than PCIe).
- **iGPU (UHD 32 EU)** = phase-3 only: compute some cold-tail experts straight out of DRAM
  (40 GB/s, no PCIe) — bounded HARD by 0.75 TFLOPS and DRAM contention; needs ACTIVATION
  exchange, not residues. A maybe-later, not a now.
- **i9 cores** = orchestration + any host-side prep; the Q4B dequant already runs ON-GPU (dp4a).

### Why CRT does NOT shard experts across the two GPUs
Dual-prime CRT represents each NUMBER by residues; a GEMM is computed in FULL in each prime
channel then Garner-combined. Assigning p1,p2->2060 and p3->iGPU means each device computes the
ENTIRE network in its modulus and needs ALL the weights — it shards numbers, not experts, and
cuts per-device memory by zero. Real expert-sharding needs full ACTIVATION exchange between
devices (not "tiny residue arrays"). CRT's genuine gift to a heterogeneous split is
BIT-EXACTNESS (no float drift across devices) — the enabler of a clean split, not the split
mechanism, and it does not remove activation traffic. Keep CRT for what it's for: exact
arithmetic / auditability.

## 3. The integrated end-to-end run (T5 — the operator's thesis, on the metal)

Config: real OK_Q4B experts (PACKED) + RESERVOIR (resident experts across steps) +
WCACHE (skip re-upload on hit, budget-stopped) + PREFIXKV (canvas-only attention recompute),
all sized to the measured 12 GB, vs the standard PACKED-only baseline. Same pinned clock.
This is the config the isolated probes never combined — and re-streaming (the T2 dominant cost)
is exactly what RESERVOIR+WCACHE are built to kill.

**T5 (cache config) — MEASURED ~5%:** PACKED+RESERVOIR+WCACHE+PREFIXKV stacked, STEPS=8
CANVAS=16, 6 items, clock-pinned 1800: C0 baseline **585s** vs C1 integrated **555s** (recall
identical 0/4, parity holds). The cache flags kill RE-STREAMING across denoise steps, but the
diffjudge cost is dominated by the SINGLE FULL FORWARD (step 0) -- so step-to-step caching barely
helps here. A stacked config only wins if each lever targets the ACTUAL bottleneck.

**T6 (scratch-reuse, SP_DG_SCRATCHREUSE) — MEASURED ~1.45x, COMMITTED (engine 1d0e414, pushed):**
hoisted the synchronizing per-expert cudaMalloc/cudaFree of the dequant scratch in
dg_gemm_packed[_rows] into a reused device pool. cudaFree syncs the whole device; removing ~2x
per-hit-expert-per-layer-per-step calls lifts the SM-55%-with-gaps wall. GATE (_sr_gate.bat,
2060 @1800, STEPS=8 CANVAS=16 LIMIT=2 FLIMIT=1): **OFF 281s vs ON 194s for 3 items = ~1.45x**;
recall 0/2==0/2, reject 0/1==0/1; warm-vs-warm per item 91s->62s. CAVEATS: ON ran 2nd leg (mild
order confound, reversed-order A/B pending); correctness = byte-identical-by-construction
(the diffusion judge has run-to-run non-determinism, so ans_tok parity is confounded -- item-1
deterministic match OFF==ON==236799). Default-off = null floor; recommend promotion after a
reversed-order confirm. THIS is the "stack the multipliers" lever the cache flags couldn't be:
it targets the measured bottleneck (per-expert serialization in the full forward).

## 4. Build priority (what's next, scoped honestly)

1. **DONE/RUNNING:** integrated cache config (T5) — no code, attacks the dominant re-streaming.
2. **NEXT (scoped, not blind):** async + pinned double-buffer of the residual spillover in
   `dg_gemm_packed_rows` / `upload_packed` — a real multi-stream refactor (second stream +
   double buffer + pinned staging). Default-off = byte-identical null floor. Gate parity first.
3. **PHASE-3 (maybe):** iGPU expert offload from DRAM via Level Zero — only after the 2060
   single-device path is maxed; bounded by 0.75 TFLOPS.

## 5. Receipts
- T1 `_pcie_bw.cu` (built+run). T2 `_t2_run.bat` + `_t2_dmon.log`. T3 `/tmp/rvq_sweep.py`.
- T5 `_integrated_sweep.bat` -> `_int_c0/c1/c2.log` + `_int_summary.log`.
- Hardware audit: nvidia-smi + Win32_VideoController/DiskDrive/PhysicalMemory + Get-PmemDisk.
