# Chat-integration sprint — PLAN

**Goal (per sprint prompt):** wire M.2's `run_dialogue()` into the daemon
chat endpoint so HTTP chat requests get the full MeMo (Grounding → Entity
ID → Synthesis) protocol with per-turn SpinorReceipts surfaced as response
metadata.

## Stage 0 — Reference reading (per feedback-lead-with-reference-then-theory)

Citations are file:line on this worktree (`engine-chat`).

1. **M.2 dialogue interface — library side** —
   `tools/sp_daemon/src/dialogue.rs:1-370` (entire file). The module
   exports: `SpinorReceipt` (line 41), `SpinorReceipt::mint()` (line 99),
   `SpinorReceipt::as_bytes()` (line 123), `MODEL_ID_EXECUTIVE` /
   `MODEL_ID_MEMORY` / `SPINOR_SENTINEL` (lines 70-74), `argmax()`
   (line 155), `DialogueCaps` (line 172), `DialoguePool::new()` /
   `reset_tokens()` (lines 213-233).

   **FINDING (load-bearing):** the library module ships ONLY data
   structures + helpers. **There is no `run_dialogue()` function
   exported from `dialogue.rs`.**

2. **M.2 dialogue protocol — implementation side** —
   `tools/sp_daemon/src/bin/sp_memo_m2_dialogue_smoke.rs:298-382`
   defines `fn run_dialogue(exec_session: &mut L1Session,
   memo_session: &mut L1Session, pool: &mut DialoguePool, user_prompt:
   &str, caps: &DialogueCaps) -> Result<DialogueOutcomeLocal, String>`
   gated on `#[cfg(target_os = "android")]`. It uses the binary's own
   `L1Session` wrapper (line 86), byte-level tokenizer (line 189), and
   `prefill`/`decode_step` helpers (lines 150-172) which are also
   android-gated. The function is NOT importable from outside the
   binary.

3. **M.2 closure** —
   `tools/sp_compute_skel/docs/CLOSURE-M2-DIALOGUE.md:156-163, 397-401,
   416-423`.

   Quoting `CLOSURE-M2-DIALOGUE.md:156-163` verbatim:

   > "**AppState additions: NONE.** Per PLAN-M2-DIALOGUE
   > §"Files to be touched" and the M.1 precedent, the M.2 smoke
   > harness lives entirely outside AppState — the standalone binary
   > `sp_memo_m2_dialogue_smoke.rs` owns its own L1Model / L1Session
   > handles + DialoguePool. The daemon-integrated `/v1/memo/dialogue`
   > route (which would wire `run_dialogue()` into AppState) is filed
   > as a future sprint."

   Quoting `CLOSURE-M2-DIALOGUE.md:418-423`:

   > "**Chat endpoint integration** (a future sprint, not on the
   > 4-MeMo path): wrap `run_dialogue()` in an axum handler;
   > `final_answer` becomes the response body, `receipts` become the
   > audit log entry. AppState extension needed (3 fields: Executive
   > SpModel + Memory SpModel + DialoguePool factory). M.2's design
   > factored to keep this integration trivial."

   The "trivial" framing is operator-perspective. Concretely the
   integration must:
   - add Memory `SpModel` + Memory `SpTokenizer` + Memory base
     `SpSession` to AppState
   - add daemon CLI / env-var flags for the Memory model path
   - load Memory model at daemon startup
   - write a host+android safe dialogue runner that takes
     `&mut SpSession` (the daemon's wrapper, NOT the smoke's
     `L1Session`) and drives the M.2 protocol against the existing
     `prefill_chunk` / `decode_step` API
   - wire the new endpoint into the router

4. **Existing chat handler** —
   `tools/sp_daemon/src/routes.rs:87-274` `pub async fn v1_chat(State,
   Json) -> Response`. Streams via axum SSE; tokenizes via
   `state.tokenizer.encode`/`apply_template` (lines 110-145); clones
   base session (`state.session.lock().unwrap().clone_session(...)`
   lines 151-154); runs `prefill_chunk` + greedy decode loop
   (lines 186-268) on a spawn_blocking thread; emits `ChatDelta` JSON
   per decoded chunk through an mpsc channel back to SSE. Rich
   features: EOS via `tokenizer.eos_ids` (line 206), stop strings via
   `TokenDecodeBuffer` (line 189), chat_id cancellation
   (`state.sessions.register` / `v1_abort`), mining backoff
   (`state.inference_active`).

   Router registration: `tools/sp_daemon/src/server.rs:19`
   `.route("/v1/chat", post(v1_chat))`.

5. **AppState** — `tools/sp_daemon/src/state.rs:46-98`. Has ONE active
   model: `model: SpModel` (line 49), `session: Mutex<SpSession>`
   (line 51). Plus optional draft: `draft_model: Option<SpModel>`,
   `draft_session: Option<Mutex<SpSession>>`. M.5 added no Memory
   model fields. M.1 added no Memory model fields. **No Memory model
   is currently loaded into AppState.**

6. **Daemon startup** — `tools/sp_daemon/src/daemon.rs:97-233`.
   Loads ONE target model (lines 112-122), optional draft model
   (lines 131-145). No Memory load logic exists. CLI flag set in
   `main.rs:65-93` (`Cmd::Start`). New CLI flag needed:
   `--memo-model` + `--memo-tokenizer` + env vars
   `SP_MEMO_MODEL_PATH` / `SP_MEMO_TOKENIZER_PATH`. Plumbed through
   `cmd_start` → `run_inner` signature.

7. **`reference-spinor-receipt-layout`** (memory) — 64-byte cache-line
   audit envelope; sentinel 0xA5 at offset 63. The format is what we
   base64-encode and emit in the HTTP response. Cross-checks
   `dialogue.rs:39-67` (struct + compile-time size guard).

8. **`reference-dual-model-cdsp-scheduler`** (memory) — confirms
   substrate works at cross-model scope (M.1 1.796× concurrent). Not
   new work; background only.

9. **`feedback-no-silent-gate-revisions`** (memory) — discipline rule.

10. **`feedback-bundled-changeset-root-cause-ambiguity`** (memory) —
    one-variable-at-a-time per stage commit.

## Scope-mismatch surfacing (UPSTREAM)

The sprint prompt states: "M.2 closed 2026-05-30 with `run_dialogue()`
shipped — a clean function in `tools/sp_daemon/src/dialogue.rs` that
takes a user prompt + AppState, runs the 3-turn Grounding → Entity ID
→ Synthesis state machine, returns `(final_answer, Vec<SpinorReceipt>)`."

This is not the reality of M.2. The actual M.2 outcome (verified above
via Stage 0 reads):

- `run_dialogue()` lives in the **android-only smoke binary**, not in
  `dialogue.rs`.
- It uses local `L1Session` types, byte-level tokenizer, and
  android-gated FFI calls.
- AppState was deliberately NOT modified by M.2; M.2 closure §
  "AppState additions" reads "**NONE**" explicitly.
- The CLOSURE-M2-DIALOGUE.md §"What unblocks now" explicitly lists
  "Chat endpoint integration (a future sprint)" with the
  AppState-extension caveat as required work.

The integration sprint therefore needs to do everything M.2 deferred,
not "just wire it up". Per `feedback-no-silent-gate-revisions`,
surface UPSTREAM in the plan + closure, do NOT pretend the function
existed and silently sweep the missing pieces into the integration
sprint.

This sprint will ship the integration **anyway** (Option B; see below)
and surface the scope-mismatch in the closure UPSTREAM list rather
than block.

## Option decision: B (parallel `/v1/dialogue` endpoint)

Per sprint prompt §3: "Pick in plan-commit + justify. Option A is
preferred because it makes MeMo the default; Option B is a fallback if
the existing handler is hard to surgically modify."

**Picked: B.** Justification:

1. **Existing `v1_chat` is rich** — SSE streaming, tokenizer template,
   EOS handling, stop strings, chat_id cancellation registry, mining
   backoff. Replacing the engine while preserving all surface contract
   would require careful surgery in the spawn_blocking closure and
   end-to-end re-verification of every feature. Multi-hour scope by
   itself.

2. **`run_dialogue()` is single-shot** (returns full final_answer at
   end of Turn 3) — the M.2 protocol does NOT produce streamable
   intermediate tokens to the user. SSE adaptation would mean either
   buffering all output then sending one event (defeats SSE's reason
   for existing) or breaking the M.2 zero-copy / per-turn-receipt
   contract by interleaving stream emissions inside `run_dialogue()`.

3. **Receipts are inherently single-shot metadata** — emit the
   complete `Vec<SpinorReceipt>` at the end of the request, not per
   chunk.

4. **Option B preserves a clean migration path** — once the MeMo
   protocol is silicon-mature and the streaming question is decided,
   a future sprint can swap `/v1/chat` to delegate to the dialogue
   runner. Option B blocks no future direction.

5. **Frontend impact is minimal** — frontend already has chat UI for
   SSE-based `/v1/chat`; adding a `/v1/dialogue` JSON-POST surface is
   a single-fetch addition.

6. **M.4 coordination** — M.4 (concurrent worktree) reads
   SpinorReceipts off the wire. A clean JSON envelope on a new endpoint
   is easier for M.4 to consume than an SSE-interleaved variant.

## Files to be touched

| File | Δ direction | Why |
|---|---|---|
| `tools/sp_daemon/src/dialogue_runner.rs` | NEW (~150 LOC) | Host+android safe `run_dialogue(&mut SpSession, &mut SpSession, &SptbTokenizer, &mut DialoguePool, &str, &DialogueCaps)` — uses existing `SpSession::prefill_chunk` / `decode_step` (host+android safe per `session.rs`). Imports `dialogue.rs` types unchanged. |
| `tools/sp_daemon/src/state.rs` | EDIT (+~10 LOC; `// Chat-integration:` prefix) | Add `memo_model: Option<SpModel>`, `memo_session: Option<Mutex<SpSession>>`, `memo_tokenizer: Option<Arc<SptbTokenizer>>` fields. All `Option` — `None` if `--memo-model` not passed at startup. |
| `tools/sp_daemon/src/daemon.rs` | EDIT (+~30 LOC) | Add Memory load block (mirrors draft load lines 132-145); plumb new `memo_model_path` / `memo_tok_path` args into `run_inner`. |
| `tools/sp_daemon/src/main.rs` | EDIT (+~10 LOC) | Add CLI flags `--memo-model` / `--memo-tokenizer` with env vars `SP_MEMO_MODEL_PATH` / `SP_MEMO_TOKENIZER_PATH`; plumb through `Cmd::Start` + hidden inner args. |
| `tools/sp_daemon/src/routes.rs` | EDIT (+~80 LOC) | Add `v1_dialogue` handler at end of file: JSON POST `{prompt: String}`, JSON response `{response: String, receipts: Vec<String>, wall_ms: u64}` (receipts base64-encoded). Returns 501 if `state.memo_model.is_none()`. |
| `tools/sp_daemon/src/server.rs` | EDIT (+2 LOC) | Add `v1_dialogue` to imports + `.route("/v1/dialogue", post(v1_dialogue))`. |
| `tools/sp_daemon/src/lib.rs` | EDIT (+3 LOC) | `pub mod dialogue_runner;` declaration. Surgically appended after the existing `pub mod dialogue;` block (line 8). M.4 lane prefix `// M.4 (ledger):` if M.4 touches lib.rs — append-only at separate location. |
| `tools/sp_daemon/Cargo.toml` | EDIT (+~4 LOC) | Register `base64` direct dep (already transitive); register `[[bin]] sp_chat_dialogue_smoke`. Prefix `# Chat-integration — MeMo dialogue endpoint`. |
| `tools/sp_daemon/src/bin/sp_chat_dialogue_smoke.rs` | NEW (~200 LOC) | Spins up daemon (or assumes one is running), POSTs to `/v1/dialogue`, asserts gates. |
| `tools/sp_compute_skel/docs/PLAN-CHAT-INTEGRATION.md` | THIS file (NEW) | Plan-commit. |
| `tools/sp_compute_skel/docs/CLOSURE-CHAT-INTEGRATION.md` | NEW (Stage 3) | Closure per sprint prompt §"Closure deliverables". |

**Lines touched outside the chat-integration files (state.rs + daemon.rs
+ main.rs + lib.rs + server.rs + Cargo.toml) — totaling ~60 LOC across
6 existing files.** All edits at append-only or low-conflict locations
relative to M.4's lane (M.4 owns `pouw_ledger.rs` NEW + may add an
AppState ledger handle field).

**Cross-lane AppState convention:**
- Chat-integration block prefix: `// Chat-integration:` per sprint
  prompt §"Coordination with concurrent M.4 agent".
- AppState additions are at the END of the struct, after the
  `peer_map` field on line 81 of state.rs and before the
  `#[cfg(target_os = "android")] dsp_session` block on line 87. The
  M.4 ledger field, if added, lands at a separate location.

## The three substantive gates

Verbatim from sprint prompt §"The three substantive gates":

- **T_CHAT_DIALOGUE_RUNS** — POST `/v1/dialogue {"prompt": "What is
  the capital of France?"}` returns HTTP 200 + JSON with non-empty
  `response` field. Report:
  `http_status, response_first_64_chars, response_wall_ms`.
- **T_CHAT_RECEIPTS_IN_RESPONSE** — response has `receipts` array
  length 3; each base64-decodes to 64 bytes; sentinel 0xA5 at offset
  63. Report:
  `receipt_count, all_64_bytes_after_decode, all_sentinel_match`.
- **T_CHAT_NO_REGRESSION** — Option B chosen → original `/v1/chat`
  untouched; existing host unit tests + `cargo build --release` clean
  (no test regression at the chat handler layer). Report:
  `tests_run, tests_passed, tests_regressed`.

## Gate harness — what runs where

**Host (this Stage 0 worktree, Windows x86_64):**
- `cargo build --release --bin sp-daemon --bin sp_chat_dialogue_smoke`
  must succeed.
- `cargo test --release` (host unit tests, including the existing
  `dialogue.rs` 12 tests) must pass. M.5 + M.2 already verified clean
  here; this gates regression on Stage 2 edits.
- `cargo build --target aarch64-linux-android --release` (cross-build)
  must succeed for android target — this is the "no android
  regression" check. The chat endpoint isn't expected to run on
  android in this sprint; M.2's smoke binary stays the device-side
  path.

**Gate-substantive runs (Knack's S22U or host with a real
.sp-model):**
- Daemon must be started with --model + --memo-model. Today the
  worktree has no `.sp-model` artifact in version control; smoke
  expects operator to point `--model` and `--memo-model` at on-disk
  paths (or pass `SP_MODEL_PATH` / `SP_MEMO_MODEL_PATH`).
- If the operator's host x86 daemon already runs `/v1/chat` end-to-end
  (current daemon does, per existing functionality), the same model
  loading machinery + the L1 host-side forward path drives
  `/v1/dialogue`. Gate run is operator-issued: `curl -X POST
  http://127.0.0.1:8080/v1/dialogue -d '{"prompt":"What is the
  capital of France?"}'`.

**Risk surfaced UPSTREAM:** I do not have a running daemon + paired
models in this agent session. The build gates + cargo test gate I can
run; the live end-to-end POST gate requires either (a) operator runs
the smoke harness on their host with their preferred model pair, or
(b) operator runs on S22U with the M.2 model pair (more involved —
adb push + run; M.2's smoke wall ~80s per dialogue). I will
explicitly mark this in the closure as UPSTREAM-REQUIRED for the
live-POST gate signal, with the substantive shipped artifacts being:
build clean + cargo test clean + smoke harness ready to run.

## Stage commits

Per `feedback-bundled-changeset-root-cause-ambiguity` (one-variable-
at-a-time per stage):

- **[plan]** This file. Stage 0 reading + Option B decision + file
  list + gate plan + scope-mismatch UPSTREAM disclosure.
- **Stage 1** — `dialogue_runner.rs` NEW + `lib.rs` `pub mod`
  declaration. No AppState changes; pure addition. `cargo build`
  (lib only) gate.
- **Stage 2** — AppState fields + daemon startup wiring + CLI flag +
  Cargo.toml dep registration. `cargo build --release --bin
  sp-daemon` gate. State + main + daemon.rs + Cargo.toml together
  because they are one logical change (the "Memory-load wiring"
  variable).
- **Stage 3** — `routes.rs` v1_dialogue handler + server.rs route
  registration. `cargo build --release --bin sp-daemon` clean. The
  endpoint variable.
- **Stage 4** — `sp_chat_dialogue_smoke.rs` NEW + Cargo.toml [[bin]]
  registration. `cargo build --release --bin sp_chat_dialogue_smoke`
  clean.
- **Stage 5** — gate runs (build + cargo test + cross-build);
  capture results in JSON + log artifacts.
- **Stage 6** — `CLOSURE-CHAT-INTEGRATION.md` + sub-tag proposal.

## What I will NOT do (explicit, per sprint prompt §"Out of scope")

- Streaming SSE for partial-turn output (single-shot JSON for v1).
- Multi-turn conversation history / KV cache management across
  requests.
- Authentication / rate limiting.
- Frontend mockup polishing.
- M.4 PoUW ledger integration (M.4 owns this).
- M.5 Variant A kernel-side sparse forward.
- Any modification of `dialogue.rs` (M.2 frozen interface).
- Any modification of files in `engine-m4/` or other worktrees.
- Any modification of `dialogue.rs`, `memo_routing.rs`, or
  `sp_memo_m2_dialogue_smoke.rs` (M.2 + M.5 frozen interfaces — only
  IMPORT from them).
- Any modification of `D:\F\shannon-prime-repos\models\` artifacts.

## Worktree status

- Worktree: `D:\F\shannon-prime-repos\engine-chat` exclusively.
- Branch: `sprint/chat-integration` (base: `a28d409` engine main with
  M.2 + M.5 merged).
- All commits authored from THIS worktree.
- M.4's worktree `engine-m4`: NOT TOUCHED.
- Main `shannon-prime-system-engine`: READ-ONLY consulted only.
- Other worktrees + lattice repos + model artifacts: NOT TOUCHED.

## Final note

The Stage 0 read surfaced a non-trivial mismatch between sprint-prompt
assumptions (clean `run_dialogue()` library function + AppState ready
with Memory model) and the actual M.2 deliverable (android-only binary
function + AppState additions explicitly deferred). I am proceeding
with Option B + the AppState extension that M.2 closed out as
"future sprint" work, surfacing the scope-mismatch in the closure
under UPSTREAM rather than blocking the sprint. The final closure
will enumerate which gates passed substantively + which are
operator-runnable + which were re-spec'd, per
`feedback-no-silent-gate-revisions`.
