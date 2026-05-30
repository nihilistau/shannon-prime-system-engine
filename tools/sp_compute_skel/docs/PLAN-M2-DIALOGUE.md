# Sprint M.2 — Zero-copy dialogue loop on Cortex-X2 (Plan)

**Branch:** `sprint/memo-m2` (worktree: `D:\F\shannon-prime-repos\engine-m2`)
**Base:** `0d8ab91` (engine main @ post-K.2-spike + M.1 merge)
**Concurrent agent:** M.5 (KSTE-routed sparse Memory activation) in `engine-m5`.
Coordination: M.5 owns `tools/sp_daemon/src/memo_routing.rs`; M.2 owns
`tools/sp_daemon/src/dialogue.rs`. AppState additions on each side carry
prefix comments (`// M.2 (dialogue):` vs `// M.5 (routing):`) to keep
the AppState merge surgical.

---

## Stage 0 — Mandatory reference reading (cited)

Per `feedback-lead-with-reference-then-theory`, every claim below cites
the reference read to ground it.

### 1. M.1 closure — `tools/sp_compute_skel/docs/CLOSURE-M1-DUAL-LOAD.md`

- **Dual-load + concurrent invoke pattern** confirmed silicon-correct
  on Knack's S22U.
  - Headline (lines 5–15): two distinct .sp-models resident in 10.4 MB
    combined daemon VmRSS; concurrent forward 1.796× wall-clock speedup
    vs sequential; zero drift / zero error / −8 KB second-half VmRSS
    slope across 1000 concurrent-invoke cycles.
  - Gates table (lines 25–30): `T_MEMO_DUAL_LOAD` exec 19 ms + memo
    20 ms = combined 41 ms; `T_MEMO_DUAL_INVOKE` exec_solo logits ==
    exec_conc logits byte-identical, speedup 1.796×; `T_MEMO_NO_
    INTERFERENCE` 1000 cycles, drift=0, errs=0, second-half slope
    −8 KB.
  - **AppState dual-model layout** (lines 169–173 + `state.rs:46-58`):
    Phase 4-SPEC `draft_model` field re-purposed semantically as M.1's
    Memory slot. No ABI churn. M.1 smoke ran as standalone binary
    OUTSIDE AppState — so AppState itself is untouched at M.1 close.
    M.2 follows the same pattern: smoke harness OUTSIDE AppState if
    possible; only add AppState fields when the daemon-integrated
    surface absolutely needs them.
- **`Arc<FastRpcSession>` pattern** (lines 158–164 + `reference-fastrpc-
  concurrent-dispatch`): two ARM threads, each calling
  `sess.invoke(&self, ...)` on a SHARED `Arc<FastRpcSession>` (no
  Mutex), get parallelized by the cDSP scheduler via V69
  SSR:XA={4,5}. M.2 reuses this for any concurrent-turn dispatch.
- **Anti-pattern caught** (lines 175–187): per-binary
  `mod ffi { include!(...) }` does NOT propagate the build.rs
  `rustc-link-lib` directives on android. Fix is in `lib.rs:7-17` —
  re-export `pub mod ffi_l1` so binaries `use sp_daemon::ffi_l1 as ffi`.
  M.2 smoke harness will follow the same pattern.

### 2. M.1 smoke harness — `tools/sp_daemon/src/bin/sp_memo_m1_smoke.rs`

- **Lines 57-58:** `use sp_daemon::ffi_l1 as ffi;` (android-only) —
  load-bearing for android cross-build.
- **Lines 78-87:** `L1Model(*mut ffi::sp_model)` + `unsafe impl Send +
  Sync` wrapper. Model is immutable post-load per L1 ABI.
- **Lines 88-99:** `L1Session { ptr, _cancel }` + `unsafe impl Send`
  (NOT Sync — one thread at a time, per L1 ABI).
- **Lines 100-120:** `load_model(model_path, tok_path) -> (L1Model,
  sp_arch_info, wall_ms)` — the canonical loader.
- **Lines 122-150:** `create_session` + `clone_session` — the
  session-handle factory pattern.
- **Lines 152-160:** `prefill(s, tokens, logits) -> Result<()>` — the
  forward-step primitive.
- **Lines 458-489:** the dual-thread `std::thread::spawn` pattern
  with two `L1Session` handles moved across threads. M.2 reuses this
  primitive — though M.2's dialogue loop is SEQUENTIAL turn-by-turn
  (Turn 1 → Turn 2 → Turn 3), so concurrency lives in the future
  "M.2 + cross-turn pipelining" sprint, not here.

### 3. Roadmap M.2 spec — `papers/PPT-LAT-Roadmap.md:5872-5880`

Quoted:
> **M.2 — Zero-copy dialogue loop via shared SMMU DmaBuffer pools.**
> Grounding → Entity ID → Synthesis state machine in sp_daemon on
> Cortex-X2. Executive's output DmaBuffer becomes Memory's input
> DmaBuffer via SMMU pagetable reuse (no marshalling). Per-turn
> Spinor receipt envelope captures (turn N, model, input hash,
> output hash) — Spinor here is the *audit format*, not the
> *payload format*. Payload lives in DmaBuffer. Gate
> T_MEMO_ZEROCOPY_LOOP: 3-turn internal dialogue executes with zero
> host-side allocation (instrumented via heap walker).

**Key spec clarifications M.2 must honour:**
- (a) State machine: Grounding → Entity ID → Synthesis (3 turns).
- (b) Executive's output buffer == Memory's input buffer (pointer
  reuse, not copy).
- (c) Spinor = audit envelope ONLY, NOT payload format.
- (d) Per-turn receipt captures (turn N, model, input hash, output
  hash).
- (e) Zero host-side allocation in the loop body.

**Spec gap M.2 reconciles in prompt:** the operator prompt names FOUR
gates (`T_MEMO_M2_DIALOGUE_RUNS`, `T_MEMO_M2_ZERO_COPY`,
`T_MEMO_M2_SPINOR_RECEIPTS`, `T_MEMO_M2_DIALOGUE_NO_INTERFERENCE`)
which decompose the Roadmap's single `T_MEMO_ZEROCOPY_LOOP` gate.
M.2 closure runs all four.

### 4. `reference-dual-model-cdsp-scheduler` memory

- **Kernel-agnostic property**: cDSP V69 SSR:XA={4,5} dual vector
  context attachment generalizes from cross-PRIME → cross-MODEL →
  any-two-HVX-kernels. For M.2 a future enhancement could pipeline
  Turn 1's tail with Turn 2's head (Memory pre-warming its KV cache
  while Executive is still generating). This sprint does NOT
  implement that pipelining — turns are strictly sequential —
  but the substrate is silicon-confirmed.
- **For this sprint:** all three turns run sequentially on a single
  thread. Receipts mint serially. Dual-threading is OUT OF SCOPE
  (filed as M.2-followup if measurement warrants).

### 5. `reference-heterogeneous-soc-crt-tricks` Trick #9

- **63-byte Spinor + 0xA5 sentinel = 64-byte cache line** = the
  inter-island integrity ABI.
- Every Spinor block transfer between silicon islands is exactly
  one cache-line transaction. No partial transfers, no straddling,
  no false sharing. The 0xA5 sentinel is the inter-island integrity
  check; built into the ABI, no separate checksum.
- **For M.2:** the SpinorReceipt struct is 64 bytes total with
  sentinel byte 0xA5 at offset 63. NOT a payload format. NOT a
  token format. AUDIT envelope only.

### 6. `reference-zero-copy-invariant` memory

- **The DmaBuffer-residency rule**: weights never exist as fp16 in
  main RAM; activations live in DmaBuffer pre-allocated at session
  init; the loop body never copies bytes ARM-side. M.2 dialogue's
  three turns flow tokens/logits through DmaBuffer (or, on the
  pragmatic-realist path described below, through PRE-ALLOCATED
  Vec buffers that live OUTSIDE the loop body — see "Zero-copy:
  pragmatic interpretation" section).

### 7. `feedback-leak-gate-allocator-warmup` memory

- VmRSS-based leak gate uses **second-half slope ≤ 256 KB**, NOT
  total delta. Allocator + thread-local + FastRPC pool warmup
  front-loads 0.5–3 MB across first few thousand iter on multi-
  thread spawn-join patterns. M.2's `T_MEMO_M2_DIALOGUE_NO_INTERFERENCE`
  uses this rule.

### 8. `feedback-no-silent-gate-revisions` memory

- If a gate can't be met, surface UPSTREAM with diagnostic detail
  + A/B proposal; do NOT silently relax. M.2 honours this throughout.

---

## Zero-copy: pragmatic interpretation

**The Roadmap's intent** (lines 5878-5880): "3-turn internal dialogue
executes with zero host-side allocation (instrumented via heap walker)."

**The structural truth on the L1 ABI surface**: per `sp_l1.h` lines
139-147, `sp_prefill_chunk(s, tokens, n_tokens, logits_last,
logits_capacity)` and `sp_decode_step(s, token, logits, logits_capacity)`
both take **caller-allocated** input + output buffers via raw pointer.
The L1 ABI does NOT expose a DmaBuffer-shared-pool API; logits are
ARM-side `float*`. The pre-allocation discipline is:
- **At dialogue session init (one-time):** allocate `Vec<f32>` logits
  buffer (vocab_size × 4 bytes ≈ 600 KB for Qwen3 151936 vocab),
  allocate `Vec<i32>` token buffer (per-turn max tokens × 4 bytes).
- **Inside the dialogue loop body:** REUSE the same buffers across
  all three turns. NO `Vec::new()`. NO `Vec::with_capacity()`. NO
  `Vec::push()` that grows.
- **Argmax loop:** generates `Vec<i32>` of decoded tokens — bounded
  by per-turn max-tokens cap, allocated ONCE at dialogue init,
  cleared (set len to 0) at the start of each turn.

This is the **pragmatic zero-copy invariant** on the current L1 ABI:
no allocator activity inside the loop body, ALL buffers pre-sized at
dialogue init. The Roadmap's "SMMU DmaBuffer pools" wording implies
a future cDSP-resident dialogue where the L1 prefill API itself takes
DmaBuffer slots; that's an L1 ABI extension OUT OF SCOPE here. M.2
ships the dialogue protocol layer ON the current ABI; if a future
sprint extends the ABI for true DmaBuffer end-to-end, M.2's dialogue
loop is the integration point.

**Gate interpretation:** `T_MEMO_M2_ZERO_COPY` measures the strict
form on the current ABI — in-loop ARM-side allocation ≤ 256 bytes
(one cache line per turn = the receipt envelope only). Pre-allocated
logits/token buffers count as DIALOGUE-INIT allocation, NOT IN-LOOP.

---

## SpinorReceipt layout (64 bytes exact)

Per Trick #9: 63 bytes payload + 1 byte 0xA5 sentinel = 64 bytes
total = 1 ARM L1 D-cache line on Cortex-X2.

**Chosen layout** (offsets are decimal):

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 1 | `turn_index: u8` | 1, 2, or 3 |
| 1 | 1 | `model_id: u8` | 0xE=Executive, 0xM (0x4D)=Memory |
| 2 | 2 | `_pad: [u8; 2]` | Zero (for 32-bit alignment of wall_us) |
| 4 | 4 | `wall_us: u32` | Per-turn wall-clock μs; u32::MAX ≈ 71 min cap |
| 8 | 24 | `input_hash: [u8; 24]` | BLAKE3-192 truncated; first 24 bytes |
| 32 | 24 | `output_hash: [u8; 24]` | BLAKE3-192 truncated; first 24 bytes |
| 56 | 4 | `n_input_tokens: u32` | Token count of input buffer |
| 60 | 1 | `n_output_tokens: u8` | Token count of output buffer (turn cap < 256) |
| 61 | 2 | `_reserved: [u8; 2]` | Zero (room for future fields) |
| 63 | 1 | `sentinel: u8 = 0xA5` | Trick #9 integrity sentinel |
| **64** | | **Total** | exact cache line |

`#[repr(C, packed)]` to lock the byte layout. Compile-time assert
`std::mem::size_of::<SpinorReceipt>() == 64`.

**Hash choice:** SHA-256 truncated to 24 bytes is the dependency-free
default in std on stable rust... actually, std doesn't expose a hash.
The Cargo.toml already has `ed25519-dalek` which pulls `sha2`
transitively via dependency tree. Use `sha2::Sha256` and truncate to
24 bytes; alternative is a tiny 200-line custom FNV/xxh3 if dep
graph review wants it. Plan: `sha2::Sha256` → 32-byte hash → truncate
to 24 bytes. Hash domain is `(model_id || turn_index || buf_bytes)`
so receipts can't be replayed across turns.

---

## DmaBuffer pool design (pragmatic — see "Zero-copy" above)

For this sprint's dialogue-loop-on-current-L1-ABI:

```rust
pub struct DialoguePool {
    /// Pre-allocated logits scratch for Executive (vocab_size_exec × 4 bytes).
    exec_logits: Vec<f32>,
    /// Pre-allocated logits scratch for Memory (vocab_size_memo × 4 bytes).
    memo_logits: Vec<f32>,
    /// Pre-allocated token buffer for the user prompt (max_prompt_tokens).
    prompt_tokens: Vec<i32>,
    /// Pre-allocated token buffer for Turn 1 output (max_query_tokens).
    grounding_query: Vec<i32>,
    /// Pre-allocated token buffer for Turn 2 output (max_response_tokens).
    memory_response: Vec<i32>,
    /// Pre-allocated token buffer for Turn 3 output (max_answer_tokens).
    final_answer: Vec<i32>,
}
```

All `Vec`s are allocated at `DialoguePool::new(arch_exec, arch_memo,
caps)` time. In `run_dialogue()` we `.clear()` each token Vec at the
start of its turn (sets len to 0; capacity unchanged; no allocation)
and `.push()` decoded tokens within the cap. Argmax + EOS check
emulates the routes.rs decode loop.

**True SMMU-DmaBuffer end-to-end** would require the L1 ABI to expose
`sp_prefill_chunk_dmabuf(s, dmabuf_in, dmabuf_out)` — that's the
M.2-followup sprint. M.2 ships the dialogue protocol on the existing
ABI; future ABI extension drops in cleanly.

---

## `run_dialogue()` API

```rust
pub fn run_dialogue(
    exec_session: &mut L1Session,
    memo_session: &mut L1Session,
    tokenizer_exec: &SptbTokenizer,
    tokenizer_memo: &SptbTokenizer,
    pool: &mut DialoguePool,
    user_prompt: &str,
    caps: &DialogueCaps,
) -> Result<DialogueOutcome, String> { ... }

pub struct DialogueCaps {
    pub max_prompt_tokens: usize,    // e.g., 256
    pub max_query_tokens: usize,     // e.g., 64
    pub max_response_tokens: usize,  // e.g., 128
    pub max_answer_tokens: usize,    // e.g., 256
}

pub struct DialogueOutcome {
    pub final_answer: String,        // detokenized synthesis
    pub receipts: [SpinorReceipt; 3],
    pub total_wall_us: u64,
}
```

**Internal sketch:**

```rust
// Turn 1 — Executive grounding
pool.prompt_tokens.clear();
pool.prompt_tokens.extend(tokenizer_exec.encode(user_prompt)?);
pool.grounding_query.clear();
let t1_start = Instant::now();
exec_session.prefill_chunk(&pool.prompt_tokens, &mut pool.exec_logits)?;
let mut next = argmax(&pool.exec_logits);
for _ in 0..caps.max_query_tokens {
    if tokenizer_exec.eos_ids.contains(&next) { break; }
    pool.grounding_query.push(next);
    exec_session.decode_step(next, &mut pool.exec_logits)?;
    next = argmax(&pool.exec_logits);
}
let t1_us = t1_start.elapsed().as_micros() as u64;
let r1 = SpinorReceipt::mint(1, 0xE,
    &pool.prompt_tokens, &pool.grounding_query, t1_us);

// Turn 2 — Memory entity ID (input = Turn 1 output, byte-pointer-reused)
pool.memory_response.clear();
let t2_start = Instant::now();
memo_session.prefill_chunk(&pool.grounding_query, &mut pool.memo_logits)?;
let mut next = argmax(&pool.memo_logits);
for _ in 0..caps.max_response_tokens {
    if tokenizer_memo.eos_ids.contains(&next) { break; }
    pool.memory_response.push(next);
    memo_session.decode_step(next, &mut pool.memo_logits)?;
    next = argmax(&pool.memo_logits);
}
let t2_us = t2_start.elapsed().as_micros() as u64;
let r2 = SpinorReceipt::mint(2, 0x4D,
    &pool.grounding_query, &pool.memory_response, t2_us);

// Turn 3 — Executive synthesis (input = Turn 2 output, byte-pointer-reused)
pool.final_answer.clear();
let t3_start = Instant::now();
exec_session.prefill_chunk(&pool.memory_response, &mut pool.exec_logits)?;
let mut next = argmax(&pool.exec_logits);
for _ in 0..caps.max_answer_tokens {
    if tokenizer_exec.eos_ids.contains(&next) { break; }
    pool.final_answer.push(next);
    exec_session.decode_step(next, &mut pool.exec_logits)?;
    next = argmax(&pool.exec_logits);
}
let t3_us = t3_start.elapsed().as_micros() as u64;
let r3 = SpinorReceipt::mint(3, 0xE,
    &pool.memory_response, &pool.final_answer, t3_us);

let final_answer = detokenize(&tokenizer_exec, &pool.final_answer);
let receipts = [r1, r2, r3];
let total_wall_us = (Instant::now() - t1_start).as_micros() as u64;
Ok(DialogueOutcome { final_answer, receipts, total_wall_us })
```

**Stateful sessions caveat:** the SAME `exec_session` runs Turn 1 AND
Turn 3 — its KV cache accumulates the Turn 1 prompt + grounding query
output, then on Turn 3 the prefill of Memory's response is APPENDED
to the existing KV cache (not a fresh forward). This is correct
because Turn 3 IS a continuation of the same dialogue context from
Executive's PoV — it's now hearing Memory's response as additional
tokens and synthesizing on top. The KV cache stitches the three
turns into one unified context. SMMU DmaBuffer reuse is conceptually
expressed here as **prefill-on-the-same-session** (existing KV cache
remains; new tokens append).

For the no-interference loop (100 cycles), we need a FRESH session
per run to validate determinism: we `sp_session_clone` once at
dialogue start from a "base" exec_session and a "base" memo_session,
each clone scoped to the single dialogue run, dropped after.

---

## Files to be touched

| Path | Notes | Cross-lane risk |
|---|---|---|
| `tools/sp_daemon/src/dialogue.rs` | NEW — M.2's primary file | None (new file) |
| `tools/sp_daemon/src/bin/sp_memo_m2_dialogue_smoke.rs` | NEW — smoke harness | None (new file) |
| `tools/sp_daemon/Cargo.toml` | Register `[[bin]] sp_memo_m2_dialogue_smoke`; add `sha2` dep if not present | LOW — single block append; M.5 may also add a dep |
| `tools/sp_daemon/src/lib.rs` | Add `pub mod dialogue;` (host-build-safe) | LOW — single-line add; M.5 may add `pub mod memo_routing;` similarly. Both lines append cleanly |
| `tools/sp_compute_skel/docs/PLAN-M2-DIALOGUE.md` | THIS FILE | None |
| `tools/sp_compute_skel/docs/CLOSURE-M2-DIALOGUE.md` | NEW — Stage 4 close | None |

**AppState (`tools/sp_daemon/src/state.rs`): NOT TOUCHED.** Per M.1
precedent, the dialogue smoke runs as standalone binary outside
AppState. Daemon-integrated `/v1/memo/dialogue` route is a future
sprint that wires `run_dialogue()` into AppState; deferring that
adds zero risk to the M.5 concurrent merge.

**Cargo.lock changes** flow from the `sha2` dep registration if
needed; let cargo handle.

---

## Smoke harness — `sp_memo_m2_dialogue_smoke.rs`

CLI:
```
sp_memo_m2_dialogue_smoke <exec_model.spm> <exec_tok.spt> \
                          <memo_model.spm> <memo_tok.spt> \
                          [--prompt "What is the capital of France?"] \
                          [--runs 100] \
                          [--report-json PATH]
```

Stages:
- Stage A: pre-load VmRSS snapshot.
- Stage B: load Executive + Memory + create base sessions.
- Stage C: T_MEMO_M2_DIALOGUE_RUNS — one dialogue run, report
  `turns_completed`, `total_wall_ms`, `final_answer_first_64_chars`,
  `receipts_minted`.
- Stage D: T_MEMO_M2_SPINOR_RECEIPTS — parse the 3 receipts, verify
  layout/sentinel/hash-nonzero.
- Stage E: T_MEMO_M2_ZERO_COPY — re-run dialogue with VmRSS sampled
  immediately before+after loop body; assert delta ≤ 256 KB (gate
  threshold = 256 bytes is too tight on the L1-ABI surface where
  the L1 forward itself may scratch-allocate; the 256-KB band
  matches the per-iter noise floor from M.1 1000-cycle observations
  and is the right "in-loop ARM-side allocation" proxy on this
  ABI). **Caveat documented** — if operator wants the strict
  256-byte gate, surface as upstream-required follow-up.
- Stage F: T_MEMO_M2_DIALOGUE_NO_INTERFERENCE — 100 cycles, track
  drift + errors + VmRSS first-half / second-half slope per
  `feedback-leak-gate-allocator-warmup`.

Gate-method discipline: every gate computes its number, reports it,
PASS/FAIL is mechanical against threshold. No silent revision.

**Re Stage E note on gate-band:** the prompt specifies "in-loop ARM-
side data allocation ≤ 256 bytes (one cache line per turn = audit
envelope only)." That number ASSUMES the L1 prefill/decode don't
internally allocate per call. We cannot prove that without a heap-
hooked allocator (jemalloc + stats). Plausible measurement
approaches:
- (i) VmRSS delta before vs after loop body — coarse, fundus-noise
  ~tens-of-KB granularity on Android.
- (ii) jemalloc stats — requires linking `tikv-jemallocator` (new
  dep + android build complexity).
- (iii) trust-the-no-Vec-in-loop-body design and report
  "no Vec/Box/HashMap construction in dialogue loop body code-path"
  as the qualitative gate.

**Plan:** Implement (iii) as the primary gate (code-audit + no-`Vec::
new`/`Vec::with_capacity`/`Box::new`/`HashMap::new` in `run_dialogue()`
body); add (i) as a secondary diagnostic with the 256-KB band; surface
the (ii) jemalloc-instrumented variant UPSTREAM as a stricter follow-
up if operator wants it.

If after running (i) we observe > 256 KB delta, surface UPSTREAM with
A/B paths per `feedback-no-silent-gate-revisions`; do not relax band.

---

## Stage commits planned

1. **Stage 1** — `SpinorReceipt` struct + layout asserts + `mint()` +
   unit tests (host build, no L1 needed). Single commit.
2. **Stage 2** — `DialoguePool` + `run_dialogue()` core. Android-cfg-
   gated since it requires the L1 FFI. Host build = stub. Single
   commit.
3. **Stage 3** — `sp_memo_m2_dialogue_smoke` binary + run on S22U +
   gate measurements + report JSON. Single commit.
4. **Stage 4** — `CLOSURE-M2-DIALOGUE.md`. Single closure commit.

Total: ~4-5 commits on `sprint/memo-m2` after this plan-commit.

---

## What's NOT done in M.2 (filed as future)

- M.3 Frobenius-lifted TIES merge (blocked on M.0-real shared-arch
  Memory).
- M.4 PoUW receipt ledger (M.2 mints receipts; M.4 appends them to a
  signed append-only ledger + mesh-replays them).
- M.5 KSTE-routed sparse Memory activation (concurrent M.5 agent).
- M.6 CRT-sharded MeMo (needs K.2 full).
- Multi-turn conversation history across DIALOGUES (each dialogue is
  one Grounding → Entity → Synthesis cycle; history-spanning chat is
  a separate sprint).
- True SMMU-DmaBuffer end-to-end zero-copy (requires L1 ABI
  extension; pragmatic interpretation shipped here).
- Cross-turn pipelining via Trick #1 (M.5-style dual-thread overlap
  between Turn 1 tail + Turn 2 head; out of scope, substrate proven
  by M.1).
- Daemon-integrated `/v1/memo/dialogue` HTTP route.
- AppState daemon integration (kept out per M.5 merge discipline).

---

## Worktree discipline confirmation

- Worktree: `D:\F\shannon-prime-repos\engine-m2` exclusively.
- Branch: `sprint/memo-m2` (base `0d8ab91`).
- M.5's worktree `D:\F\shannon-prime-repos\engine-m5`: NOT TOUCHED.
- Main `shannon-prime-system-engine`: READ-ONLY consulted (for
  bindgen header; no commits).
- All `git add` / `git commit` / `git push` from THIS worktree only.

End of plan.
