# Sprint ledger-autowire — PLAN

**Goal:** every successful `POST /v1/dialogue` invocation **automatically
appends its 3 SpinorReceipts** to a process-local PoUW ledger (opened at
daemon startup from a CLI flag). Receipts continue to surface in the HTTP
response unchanged.

**Why:** chat-integration ships receipts in HTTP responses; M.4 ships a
host-safe `Ledger` with byte-level `append()`. Nothing wires the two.
After this sprint, the daemon's ledger accumulates every dialogue turn
ever served, satisfying Manifesto **Trick #10** for chat traffic
end-to-end (no separate miner needed for chat audit trail).

**Out of scope (do NOT bundle):** real QUIC mesh broadcast, canonical
mesh ordering (concurrent `engine-mesh-order` sprint owns
`SpinorReceipt._reserved[0..2]`), ledger pruning/encryption/signing,
SpinorReceipt LAYOUT modifications (frozen), multi-tenant ledger.

---

## Stage 0 — Reference reading (file:line)

1. **Chat-integration closure** (`tools/sp_compute_skel/docs/CLOSURE-CHAT-INTEGRATION.md:33-40, 130-136`)
   - Handler: `tools/sp_daemon/src/routes.rs::v1_dialogue` (registered at `tools/sp_daemon/src/server.rs:22`).
   - Response shape: `DialogueResponse { response: String, receipts: Vec<String /*base64*/>, wall_ms: u64, turn_us: [u64; 3] }`
     (`routes.rs:599-605`). 3 base64-88-char strings, each decodes to
     64 bytes.
   - Inner result before serialization: `outcome.receipts: [SpinorReceipt; 3]` from
     `dialogue_runner::run_dialogue` (`dialogue_runner.rs:48`).
   - Handler success path is at `routes.rs:728-742` (`match result { Ok(Ok(outcome)) => ... }`).

2. **M.4 closure** (`tools/sp_compute_skel/docs/CLOSURE-M4-LEDGER.md:30-35, 141-148, 150-154`)
   - API: `Ledger::open<P: AsRef<Path>>(path: P) -> LedgerResult<Self>`
     (`pouw_ledger.rs:118-130`).
   - API: `Ledger::append(&mut self, receipt: &SpinorReceipt) -> LedgerResult<u64>`
     (`pouw_ledger.rs:141-148`).
   - Multi-thread guidance: "Multi-thread sharing requires `Arc<Mutex<Ledger>>`"
     (`pouw_ledger.rs:30-32`).
   - Per-append wall on S22U: p50=2 µs, p99=3 µs (cheap enough for any
     hot path; serializing on a global Mutex is fine).

3. **Existing AppState** (`tools/sp_daemon/src/state.rs:46-116`)
   - M.2/Chat-integration precedent: added 4 `memo_*` fields
     (`state.rs:89-99`), all `Option<...>` so the route can 501 when the
     CLI flag is unset.
   - Pattern: `pub memo_session: Option<Mutex<SpSession>>` and
     `pub memo_model: Option<SpModel>`.
   - I will add ONE field, prefixed with `// ledger-autowire:`:
     `pub ledger: Option<Arc<Mutex<Ledger>>>`.
   - Mutex is required because `Ledger::append` takes `&mut self`.
     `Arc<>` shares the handle into the `tokio::spawn_blocking` closure
     in `v1_dialogue`.

4. **CLI flag pattern** (`tools/sp_daemon/src/main.rs:90-97`)
   - Memory model was added in `Cmd::Start { ..., memo_model: String,
     memo_tokenizer: String, ... }` with
     `#[arg(long, env = "SP_MEMO_MODEL_PATH", default_value = "")]`.
   - Inner-respawn hidden args mirror at `main.rs:59-63`.
   - daemon.rs `cmd_start` and `run_inner` plumb them through
     (`daemon.rs:34, 46-47, 100, 151-172`).
   - **My mirror** — add `pouw_ledger_path: String` (empty = disabled):
     `#[arg(long, env = "SP_POUW_LEDGER_PATH", default_value = "")]`.
     Following the existing convention of String + empty-default rather
     than `Option<PathBuf>` keeps clap derive happy alongside the other
     hidden-arg pairs.

5. **`reference-spinor-receipt-layout`** (memory entry)
   - 64-byte `#[repr(C, packed)]` envelope, sentinel `0xA5` at offset 63.
   - I do NOT modify it. I consume `outcome.receipts: [SpinorReceipt; 3]`
     and pass each to `Ledger::append(&r)`.
   - Concurrent mesh-canonical-order agent intends to add helpers on
     SpinorReceipt for `_reserved[0..2]`-encoded sequence rank. I do not
     touch those bytes.

6. **`feedback-no-silent-gate-revisions`** — surface UPSTREAM first if
   any of the three substantive gates fails. Do not silently widen
   tolerance.

7. **`feedback-bundled-changeset-root-cause-ambiguity`** — Stage commits
   in order: plan → state+CLI → handler wire-in → smoke+gates → closure.
   One discrete change per commit.

8. **`feedback-parallel-agents-separate-worktrees`** — all work from
   `D:\F\shannon-prime-repos\engine-ledger-autowire`. Do not touch
   `engine-mesh-order` (concurrent), or any other worktree.

---

## Architecture decisions

### D1. `Option<Arc<Mutex<Ledger>>>` in AppState

`Option<>` so the daemon stays bootable without `--pouw-ledger-path`
(matches `memo_model` shape). `Arc<>` so the `tokio::spawn_blocking`
closure can clone the handle out of `&state.ledger`. `Mutex<>` because
`Ledger::append` takes `&mut self` and concurrent `/v1/dialogue` requests
will race on the file handle otherwise.

The post-flush wall is p50=2 µs / p99=3 µs (M.4 closure §"Gates table"),
so the critical section is ~3 µs at most + 3 appends per dialogue = ~10
µs serialized work. On a localhost-only HTTP API at the throughput the
daemon serves (a dialogue is ~31 s wall), Mutex contention is structurally
irrelevant.

### D2. Best-effort append, never fail the HTTP response

Per the sprint prompt: "On lock/append failure, log warning but DO NOT
fail HTTP response (ledger is best-effort)." This matches the broader
M.4 design (the ledger is observational, not a transactional gate on
the dialogue). Per-receipt `tracing::warn!` call if `lock()` or
`append()` returns `Err`; the response still ships the 200 + receipts to
the caller.

### D3. Append in the success path, BEFORE building DialogueResponse

`/v1/dialogue` handler's success path (`routes.rs:728-742`) is the
single place all 3 receipts are guaranteed to exist as
`outcome.receipts: [SpinorReceipt; 3]`. Append immediately there, before
the base64 encoding loop. Borrowing the receipts (`for r in
&outcome.receipts`) means the existing `outcome.receipts.iter().map(...)`
expression still compiles unchanged.

### D4. CLI flag name: `--pouw-ledger-path`

Matches the existing memory-model naming convention (`--memo-model`,
`--memo-tokenizer`) and `pouw_ledger.rs` module name. Env var:
`SP_POUW_LEDGER_PATH`. Hidden inner-arg for daemon-inner self-respawn
matches `--memo-model` plumbing exactly.

### D5. Open at startup, not lazily

If `--pouw-ledger-path` is set, daemon opens the ledger at startup
(`daemon::run_inner`), bails the daemon launch if the path is
non-openable. This catches misconfig (typo'd path, no parent dir, no
write permissions) at start rather than at first dialogue. Matches
the `--memo-model` discipline (load fails → daemon refuses to start).
If path is empty (default), AppState ledger = `None` and the wire-in
is a no-op.

### D6. Smoke harness — duplicate, don't mutate

Per `feedback-bundled-changeset-root-cause-ambiguity`, I add a new
`sp_chat_ledger_autowire_smoke` binary rather than extending the
existing `sp_chat_dialogue_smoke`. The existing smoke must still
PASS unchanged (T_AUTOWIRE_NO_REGRESSION); pollutuing it with ledger
checks would conflate gate diagnosis.

The new smoke:
1. Pre-snaps ledger file size (or treats absent file as 0 bytes).
2. Drives 5 sequential `POST /v1/dialogue` invocations against the
   running daemon.
3. Asserts ledger file grew by exactly 5×3×64 = 960 bytes.
4. Asserts each appended record is byte-identical to the
   base64-decoded receipt from the corresponding HTTP response.

(15×64 byte comparison; needs to remember the per-dialogue receipts
and the order in which they were appended.)

---

## The three substantive gates

| Gate | Method | Pass criteria |
|---|---|---|
| `T_AUTOWIRE_LEDGER_GROWS` | N=5 dialogue invocations against running daemon with `--pouw-ledger-path` set; record file size pre + post | `delta == 5 * 3 * 64 == 960` exactly |
| `T_AUTOWIRE_RECEIPT_BYTE_IDENTITY` | After 5 dialogues, read ledger file as 15 × 64-byte records; for each, compare to base64-decoded receipt from response captured earlier | 0 byte divergences across 15 × 64 bytes |
| `T_AUTOWIRE_NO_REGRESSION` | Re-run existing `sp_chat_dialogue_smoke` against the same running daemon (now with ledger wired); all 3 chat-integration gates must still PASS | HTTP 200, 3 receipts, sentinels intact |

Surface UPSTREAM if any gate fails (per `feedback-no-silent-gate-revisions`).

---

## File touch list (preview)

| File | Edit scope | Stage |
|---|---|---|
| `tools/sp_compute_skel/docs/PLAN-LEDGER-AUTOWIRE.md` | NEW (this file) | plan |
| `tools/sp_daemon/src/main.rs` | +2 hidden arg + +1 Cmd::Start arg + +1 plumbing arg | 1 |
| `tools/sp_daemon/src/daemon.rs` | +1 fn sig param to `cmd_start`/`run_inner` + ledger open at startup + AppState field populate | 1 |
| `tools/sp_daemon/src/state.rs` | +1 `pub ledger: Option<Arc<Mutex<Ledger>>>` field with `// ledger-autowire:` prefix | 1 |
| `tools/sp_daemon/src/routes.rs` | +~12 lines: append loop in `v1_dialogue` success path | 2 |
| `tools/sp_daemon/Cargo.toml` | +5 lines (new `[[bin]]` block) | 3 |
| `tools/sp_daemon/src/bin/sp_chat_ledger_autowire_smoke.rs` | NEW | 3 |
| `tools/sp_compute_skel/docs/CLOSURE-LEDGER-AUTOWIRE.md` | NEW | 4 |

**Lines touched in concurrent-agent's file surface:** ZERO.
`dialogue.rs` not touched. `pouw_ledger.rs` not touched.

---

## Sub-tag

`lat-phase-4-memo-ledger-autowire` — parallel to `-m4-ledger`,
`-chat-integration`, etc.

---

## Worktree discipline

Branch: `sprint/ledger-autowire` (base `52e2145` = engine main post
chat-integration merge). All commits authored from
`D:\F\shannon-prime-repos\engine-ledger-autowire`. Push at end:
`git push -u origin sprint/ledger-autowire`. Operator handles merge.

The concurrent `engine-mesh-order` agent (worktree on
`sprint/mesh-canonical-order`) shares the same base. They will touch
`dialogue.rs` (SpinorReceipt helpers) and `pouw_ledger.rs` (Ledger
canonical-sort methods). I touch neither. Both lanes converge cleanly
at merge time.
