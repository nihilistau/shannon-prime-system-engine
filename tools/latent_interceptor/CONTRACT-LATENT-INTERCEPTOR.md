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

## Gate

- **G-LI-AGREEMENT:** Latent Interceptor action vs the ground-truth tape label (held-out tape).
  Target: high NO_OP precision (don't escalate idle to the 12B) + high ACTION recall (don't miss a
  real event). Compute saved = NO_OP-fraction x (12B tick - draft tick).
- Null floor: `decide_via_draft` default-off; the 12B stub/decode path byte-untouched when off.
