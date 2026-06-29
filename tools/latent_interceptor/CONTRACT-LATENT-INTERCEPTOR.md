---
type: contract
title: "CONTRACT-LATENT-INTERCEPTOR — KAIROS decisions in latent space (no tokenizer)"
description: "Repurpose the finetuned 4-layer draft as a discrete action-space classifier that gates the 12B heartbeat tick, never projecting to the 262k vocab."
tags: [contract, kairos, latent-interceptor, draft, ppt-arm]
timestamp: 2026-06-30T00:00:00Z
resource: ./tools/latent_interceptor/CONTRACT-LATENT-INTERCEPTOR.md
sp_status: SCAFFOLD
sp_gate: G-LI-AGREEMENT (pending)
sp_commit: TBD
sp_repro: "tools/latent_interceptor/* + SP_LI_CAPTURE / sp_li_train.py"
---

# CONTRACT-LATENT-INTERCEPTOR — KAIROS decisions in latent space

## Why (the boundary)

Spec decode is a throughput dead-end here (G-EAGLE-THROUGHPUT: plain decode 24.9 tok/s beats any
spec config; draft 24ms ~0.64x target 38ms, not <<). BUT the finetuned draft is a proven, cheap
predictor of the 12B's latent trajectory. Repurpose it: a KAIROS background tick costs ~35ms frame
prefill + **~477ms decode**. On a daemon, ~95% of ticks are NO_OP. If the draft decides NO_OP
WITHOUT decoding, we save the 477ms on the idle majority.

**The Latent Interceptor:** rip out the draft's 262144-vocab projection head. The model never
tokenizes the decision. Replace the head with a tiny `HID(1024) -> A` action-classification head
that maps the latent state directly into a discrete KAIROS action space. The decision is a latent
trigger, not a text string the Python harness must parse back (the abstraction tax PPT-ARM removes).

## Action space A (v1 — grounded in existing framework ops)

| id | action | maps to | effect |
|----|--------|---------|--------|
| 0 | NO_OP | (idle) | prune the tick (cold-evict rewind); no 12B |
| 1 | KEEP | curator ADMIT / MEM-OKF episode emit | admit the event to memory (latent -> Spinor block) |
| 2 | FORGET | curator SKIP / cold-evict | drop / evict; no 12B |
| 3 | E2B_TOOL | MCP tool / E2B sandbox | trigger a tool call (latent -> tool-id logit) |
| 4 | ACTION | (generic) | escalate to the 12B for full elaboration |

Only id=4 (ACTION) needs the 12B decode. 0/1/2/3 are handled latent-side -> the 477ms decode is
saved on every non-ACTION tick. (A is configurable; the head is A-wide.)

## The framework: ONE shared draft body, MANY latent heads

The draft body (pre_proj + 4 layers attending the target KV + post_proj/out_norm) is the SHARED
latent substrate — the vocab head is ripped off, so the body is ~ms and CPU/Hexagon-pinnable (the
262k vocab matrix was the entire cost; without it a 4-layer pass is <2ms on host). It runs ONCE per
intercept, producing a 1024-d latent that a REGISTRY of tiny specialized heads taps:

| head | projection | destination (latent-native, no tokenization) |
|------|-----------|-----------------------------------------------|
| **action** | HID->A | KAIROS gate (NO_OP/KEEP/FORGET/E2B_TOOL/ACTION) -> gate the 12B tick |
| **memory** | HID->63-byte C2 Spinor | MEM-OKF write (the curator ADMIT path) from the latent |
| **tool** | HID->32 MCP-tool logits | fire the harness decorator / E2B sandbox directly |
| (return) | — | tool result -> cyclotomic-ring residue -> `gemma4_kv_inject` into the KV ring |

One body pass, many latent destinations. Heads are independently finetuned + hot-swappable. This is
the PPT-ARM realization: keep computation on the latent manifold; exit to tokens only when a human
must read the output.

## The arc (this scaffold)

1. **Capture** (LI-2, `SP_LI_CAPTURE`): run the 12B over a KAIROS event tape; per frame-end capture
   `(feat[3840], x[3840], KV)` — the exact draft-body inputs (reuses the eagle flywheel
   `gemma4_kv_capture_feat` / `gemma4_embd_row` / `gemma4_kv_ctx_dump`) + the action label. (A
   feature-only variant is also captured as the BASELINE: does the body add value over a raw probe?)
2. **Train** (LI-3): reuse the eagle draft-body forward (`sp_eagle_train.draft_forward` returns the
   1024 pre-head latent) + a tiny `HID->A` action head; CE on the action label. NO vocab head. The
   body may be frozen (use the #2-finetuned body) or co-trained. Per-head trainers for mem/tool later.
3. **Deploy** (LI-4): refactor `gemma4_draft_step` -> `gemma4_draft_body` (returns the 1024 latent,
   no 262k gemm/argmax/suppress) + `gemma4_draft_head_action(latent)->A`. `decide_via_draft` seam in
   the KAIROS heartbeat (default-off, byte-exact null floor); the 12B fires only on action id=4.

## Latent routing (the larger PPT-ARM vision, post-scaffold)

- **MEM head:** id=1 KEEP -> project the latent directly to a 63-byte C2 Spinor block -> MEM-OKF
  write, zero tokenization (the curator's admit path, latent-native).
- **Tool head:** id=3 E2B_TOOL -> project to the 32-tool MCP logit space -> trigger the harness
  decorator directly.
- **Latent injection (return path):** tool results are transcoded to a cyclotomic-ring residue and
  injected into the target KV ring (`gemma4_kv_inject`) — the model feels the result, never reads it.

## LIVE RESULT (G-LI-HEARTBEAT, 2026-06-30)

Persistent-contract heartbeat (`SP_LI_HEARTBEAT`, run_li_heartbeat) on the held-out tape (events
151-300, untrained):
- **accuracy 1.000 (150/150)**; NO_OP-skips 85, woke 65.
- **NO_OP tick: ~5000ms -> 1451ms** = event-delta prefill (1536ms) + latent probe (**1.02ms**),
  the 12B decode ELIMINATED. Contract prefilled ONCE (2762ms, amortized). decode skipped 85/150.
- This kills the run_li_oracle harness artifact (~4000ms full re-prefill/tick). Floor is now the
  event-delta prefill (the model seeing the event) + the ~1ms probe.
## DELTA-TRIM (G-LI-FLOOR, 2026-06-30)

All static framing moved into the contract prefix (`LI_CONTRACT` + `li_frame_text`); the per-tick
delta is STRICTLY the event body + turn close/primer. Re-captured + re-trained on the trimmed
distribution, re-ran the held-out heartbeat:
- **NO_OP idle tick: 585ms avg** (shortest ~505ms) = bare event-delta prefill (669ms) + probe
  (1.04ms), 12B decode eliminated. **~8.5x down from the original ~5000ms.** Contract prefilled
  once (2965ms, amortized).
- accuracy 0.993 (149/150) — one mis-gate; the small cost of less per-tick context (was 1.000).
- The absolute physical minimum of the idle tick is now: the model seeing the bare event + a 1ms
  latent decision. Further reduction requires the OKFS-MEM header load to move OFF the prefill (into
  a latent memory head) — the multi-head framework, next.

## MULTI-HEAD FRAMEWORK — body + heads (MH, 2026-06-30)

**MH-1 (DONE):** `gemma4_draft_body` (cuda_forward.cu) = the draft step WITHOUT the 262k vocab head;
runs pre_proj + 4 layers + out_norm, returns ONLY the 1024-d latent (host). No vocab gemm / argmax /
suppress -> the cost is the 4-layer body alone (CPU/Hexagon-able). The shared substrate all heads tap.
gemma4_draft_step (with vocab head) stays in the shelved spec-decode drawer.

**MH-2 (DONE):** draft-body capture — `SP_LI_CAPTURE` + `SP_DRAFT_GGUF` runs `gemma4_draft_body` live
on each event frame (attends the real KV) and dumps `latent.f32 [N x1024]` alongside `feat.f32` +
`label.i32`. Validated (20-event run: latent [20x1024]). This is the head training substrate.

**MH-3 Memory Head (Priority 1) — SCOPE (`sp_mh_train.py`):**
- **Head:** `latent[1024] -> Linear(1024,256) -> ReLU -> Linear(256,55)` = the 55-d VHT2 content key.
  Deploy: key -> `sp_spinor_encode` (C, sp/spinor_block.h) -> the frozen 63-byte block -> MEM-OKF
  write/address. Pure latent -> Spinor, zero tokenization.
- **Target (v1, self-supervised):** content key = a FROZEN random projection of the 12B feature to
  55-d (Johnson-Lindenstrauss, distance-preserving -> similar events -> similar Spinors ->
  content-addressed recall). Head learns latent -> proj(feat); MSE loss. No external labels.
  - v2: distill the nightshift_curator's actual episode Spinor (MEM-OKF fidelity).
  - v3: contrastive recall objective (cue -> memory) for XBAR-style retrieval.
- **63-byte Spinor (frozen v1, sp/spinor_block.h):** 7B header (scale f32 + exp int8 + basis +
  reserved) + 55 int8 Mobius-permuted anchors (sp_mobius(i)=17i mod 55) + CRC-8. anchor_i =
  round(vec_i/scale*127). The int8 roundtrip fidelity is checked in the trainer.
- **Deploy (next):** CUDA `gemma4_draft_mem(latent)->55-d key` (tiny MLP, like li_probe) ->
  sp_spinor_encode -> emit to MEM-OKF (tools/okf_mem.py add / the curator ADMIT path), all latent-side.

## MEMORY HEAD v2 — CURATOR DISTILLATION (G-MH-CURATOR, 2026-06-30)

**RETARGET (user veto of v1 JL):** the curator does NOT emit a 63-byte Spinor — that's the KV codec.
The curator's geometrically-pure content key is the **C2 256-bit LSH signature** (recall.rs
`Projection`): `sig[b] = sign(R[b]·pooled_K)`, R = frozen +/-1 `smix(SEED,256*512)`, pooled_K = sum
over (8 global layers, pos) of K[512]. Integer Hamming, zero float in the address.
- **Head:** `latent[1024] -> Linear(1024,512) -> ReLU -> Linear(512,512) = pooled_K_est`; the FROZEN
  curator R then produces the byte-identical address. The head reconstructs the GEOMETRY; R is untouched.
- **Capture (MH-2 ext):** `SP_LI_CAPTURE`+`SP_DRAFT_GGUF` reads the live global-K
  (`gemma4_kv_read_global_k`), pools it, applies the real `recall::Projection` -> dumps
  `latent.f32`+`pooledk.f32`+`sig.u64` (curator ground truth). Parity: captured pooled_K->R->sig == sig.
- **Train (`sp_mh_train.py`):** BCE on `sign(R@pooled_K_est)` vs curator sig + MSE on unit pooled_K.
  RESULT (150 synthetic, held-out val): **val_bit_agree 255/256 (Hamming 1.0)**, recall@168 **1.000**,
  exact-256 **0.567**. R replicated byte-identically (splitmix64).
- **Gate (`SP_MH_GATE`, run_mh_gate):** live held-out — latent->mh_probe->frozen R->C2 sig vs the
  curator's own sig: **mean Hamming 1.17/256, max 3, 100% RECALL-MATCH** (several byte-exact). The
  draft writes curator-identical addresses from the latent, no tokenization. **Honest gap: gate
  ~91.5ms** (the gemma4_draft_body GPU pass; mh_probe+R are CPU-us), NOT the <3ms target. Closing
  that needs body optimization (CUDA-graph the 4-layer body, or the CPU/Hexagon low-dim port).
- Caveat: 150 synthetic templated events; the 255/256 will soften on real diverse data. Mechanism proven.
- **Deploy-to-MEM-OKF (next):** emit on KEEP (`okf_mem.py add --kind episode --addr c2sig_{hex}`),
  latent-native; + the body-speed optimization for the <3ms idle write.

## RETURN PATH + TOOL HEAD (RP-1 / TH-1, 2026-06-30)

**RP-1 (return path, GREEN):** `run_li_return` (SP_LI_RETURN) — a tool result enters the KV ring
DIRECTLY via `gemma4_kv_inject_tokens` (the embedding->residual seam), no prompt-text re-feed, no
tokenizer-output round-trip. The model's next forward attends the injected result. Demo (strawberry):
13 result tokens injected -> the continuation is conditioned on them. The model FEELS the tool output
latent-native. (Single-vector cyclotomic residue via gemma4_kv_inject = the refinement.)

**TH-1 (Tool Head, GREEN):** the capture/probe pipeline is now label-set-agnostic (`SP_LI_LABELS`
overrides the KAIROS action space). Tool vocab NONE,PYTHON,WEB,DB,FILE,CALC; tool_tape (make_tool_tape.py,
140 events); capture (SP_LI_LABELS=...) -> _tool_data; sp_li_train (label-agnostic) -> _tool_head.bin.
RESULT held-out val_acc=**1.000** (confusion clean). The latent routes directly to the MCP/E2B tool id
-> fire the harness decorator from a latent trigger. Same synthetic-tape caveat as the other heads.

**The closed loop** (capstone, pieces all proven): latent -> Tool Head (tool id) -> fire tool ->
result -> RP-1 inject -> KV ring -> model continues. TH-1 (the trigger) + RP-1 (the return) are each
verified; the integration `run_th_loop` ties them (a real python eval on PYTHON/CALC events).

## TRUTH SERUM — adversarial eval (TS-1, 2026-06-30)

The clean-template heads' 1.000 scores were LAB CONDITIONS. make_adversarial_tape.py (near-miss /
paraphrase / noise / ambig, near-miss-heavy) + sp_li_eval.py (eval-only, no retrain) on the CLEAN
tool head:
- **OVERALL adversarial accuracy = 0.393** (was 1.000). **NONE recall = 0.000, FALSE-FIRE = 1.000**:
  the clean head fires a tool on EVERY near-miss ("i was reading how python counts" -> fires). It
  learned surface cues (python/count/file -> fire), NOT invoke-vs-discuss. FILE = the confusion sink
  (prec 0.19); CALC 14/20 -> PYTHON (compute blur).
- **RECOVERY:** retrain WITH near-misses (label NONE) -> val_acc 1.000 (CAVEAT: same-generator val,
  optimistic; true number needs a different adversarial distribution or live data). The architecture
  is SOUND — the probe learns the boundary when the training distribution contains it.
- **CONCLUSION:** every head MUST train with near-misses-as-NONE + paraphrase variety, or it false-
  fires on any mention of a tool/memory/action. Lab-clean training is UNSAFE. The fix is DATA
  coverage (real/adversarial tapes), not a redesign. This is the honest routing floor.

## Gate

- **G-LI-AGREEMENT:** Latent Interceptor action vs the ground-truth tape label (held-out tape).
  Target: high NO_OP precision (don't escalate idle to the 12B) + high ACTION recall (don't miss a
  real event). Compute saved = NO_OP-fraction x (12B tick - draft tick).
- Null floor: `decide_via_draft` default-off; the 12B stub/decode path byte-untouched when off.

## TS-2 — Hardened Tool Head + TRUE cross-distribution OOD (G-TH-HARD) GREEN
**Fix = data coverage, proven cross-distribution.** `make_hard_tape.py` emits TRAIN and OOD tapes from
**DISJOINT phrasing banks + different seeds** (so the OOD eval is honest cross-distribution, not same-generator).
Near-misses (mention-without-invoke) strictly ->NONE; CALC=pure math vs PYTHON=code/string ops semantically split.
- TRAIN: hard_train.txt (200 ev, NONE=112) -> `_hard_train_data` -> `sp_li_train.py` -> `_tool_head_hard.bin`.
- OOD (isolated, disjoint banks): hard_ood.txt (120 ev) -> `_hard_ood_data` -> `sp_li_eval.py`.
- **CLEAN head on isolated OOD: 0.350 acc, NONE-recall 0.014, FALSE-FIRE 0.986** (the TS-1 vulnerability, confirmed cross-dist).
- **HARDENED head on isolated OOD: 1.000 acc (120/120), every class prec/recall 1.000, NONE-recall 1.000, FALSE-FIRE 0.000.**
- Deployed: `_tool_head_hard.bin` -> live `_tool_head.bin` (clean -> `_tool_head_clean.bin`).
- Honest caveat: OOD bank is disjoint surface forms but the SAME author/intent taxonomy -> tests surface-form
  generalization, NOT adversary-authored novelty. Real-chat tapes (SP_LI_TAPE) remain the final test.
- Recipe is label-set-agnostic: identical pipeline hardens the ACTION head (NO_OP/KEEP/... label set). The MEMORY
  head is downstream of a KEEP decision (address producer, not a classifier) -> it is gated by the action head, so it
  cannot false-fire in the routing sense; its safety = the action head's near-miss-hardened KEEP boundary.
- Repro: `python make_hard_tape.py hard_train.txt 200 train` / `hard_ood.txt 120 ood`; capture via `_hard_{train,ood}_cap.bat`; `sp_li_eval.py --head _tool_head_hard.bin --data _hard_ood_data`.

## TS-3 — Hardened Action Head + TRUE cross-distribution OOD (G-ACT-HARD) GREEN (safety); KEEP-recall gap noted
Same recipe as TS-2, action space {NO_OP,KEEP,FORGET,E2B_TOOL,ACTION}. Near-misses reuse the trigger verbs
(forget/remember/run/send/deploy) in non-command contexts -> NO_OP. `make_hard_tape.py ... action` (disjoint
TRAIN/OOD banks + different seeds). TRAIN 200 ev (NO_OP=127) -> `_hard_act_train_data` -> `_act_head_hard.bin`;
isolated OOD 120 ev -> `_hard_act_ood_data`.
- **CLEAN live head (_li_head.bin) on isolated OOD: 0.183 acc, NO_OP-recall 0.015, FALSE-FIRE 0.985** (the KAIROS
  heartbeat was firing an action on 98.5% of idle chatter cross-distribution -- a worse false-fire than the tool head).
- **HARDENED on isolated OOD: 0.958 acc (115/120), NO_OP-recall 1.000, FALSE-FIRE 0.000.** KEEP prec 1.000 (no
  spurious memory writes), FORGET 1.000/1.000, E2B_TOOL 1.000/1.000, ACTION prec 0.812/rec 0.929.
- **Safety GREEN**: zero false-fire + KEEP precision 1.000 => the Memory-Head KEEP gate cannot fire on idle/near-miss.
- **Honest capability gap**: KEEP recall 0.429 (under-fires real stores; n=7 OOD, only 18 KEEP train) and ACTION
  prec 0.812. Both are the SAFE failure direction (miss, never hallucinate). Lift = richer KEEP/ACTION phrasing
  diversity in the train tape (mechanical recapture), NOT a redesign.
- Deployed: `_act_head_hard.bin` -> live `_li_head.bin` (clean -> `_li_head_clean.bin`).
- Multi-head safety pass status: Tool head GREEN (TS-2), Action head GREEN-safety (TS-3); Memory head needs no
  hardening (address producer downstream of KEEP, gated by the action head, cannot false-fire).
- Repro: `python make_hard_tape.py hard_train_act.txt 200 train action` / `hard_ood_act.txt 120 ood action`; capture `_hard_act_{train,ood}_cap.bat`; `sp_li_eval.py --head _act_head_hard.bin --data _hard_act_ood_data`.
