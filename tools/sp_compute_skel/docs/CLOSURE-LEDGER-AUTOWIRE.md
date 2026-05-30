# Sprint ledger-autowire — auto-append /v1/dialogue receipts to PoUW ledger CLOSURE

**Status:** ALL 3 substantive gates PASS on host MSVC (Windows 11) +
chat-integration's 3 gates STILL PASS under the autowire-active daemon
(regression spot-check).

## Headline

Every successful `POST /v1/dialogue` now automatically appends its 3
SpinorReceipts to a shared, mutex-serialized `Ledger` opened at daemon
startup from `--pouw-ledger-path` / `SP_POUW_LEDGER_PATH`. Receipts
continue to surface in the HTTP response unchanged. N=5 sequential
dialogues against the running daemon grew the ledger file by exactly
**960 bytes** (= 5 × 3 × 64), and all **15 × 64-byte records on disk
are byte-identical** to the base64-decoded receipts returned in the
corresponding responses (0 byte divergences). The chat-integration
contract still holds under autowire: every dialogue returned HTTP 200 +
3 receipts + 0xA5 sentinel. **Manifesto Trick #10 (Receipt-backed
verifiable distributed compute)** for the `/v1/dialogue` path is now
end-to-end wired — daemon persists every audit envelope, no separate
miner needed for chat traffic.

## Gates table

All three gates RUN on host MSVC (Windows 11 Pro 26200) against the
release daemon (`target/release/sp-daemon.exe`) serving both
`qwen3_rt.sp-model` (Executive, vocab=151936) and
`qwen25-coder-0.5b-memory.sp-model` (Memory, vocab=151936) with
`--pouw-ledger-path %TEMP%\autowire.spinor`. Smoke driver:
`tools/sp_daemon/target/release/sp_chat_ledger_autowire_smoke.exe --n 5`.

| Gate | Method | Observed | Verdict |
|---|---|---|---|
| `T_AUTOWIRE_LEDGER_GROWS` | Snapshot ledger size before + after N=5 dialogues; delta must equal 5×3×64 | `pre_size=0, post_size=960, delta=960, expected=960` | **PASS** |
| `T_AUTOWIRE_RECEIPT_BYTE_IDENTITY` | Read appended 960-byte slice as 15 × 64-byte records; for each, byte-compare to the base64-decoded receipt from the corresponding HTTP response | `receipts_compared=15, byte_divergences=0` over 15 × 64 = 960 bytes | **PASS** |
| `T_AUTOWIRE_NO_REGRESSION` | Every dialogue HTTP 200 + 3 receipts (chat-integration contract still holds under autowire) | `dialogues_with_status_200=5, dialogues_with_3_receipts=5, transport_errors=0` | **PASS** |

### Bonus regression spot-check

After the 5-dialogue ledger-autowire smoke, the pre-existing
`sp_chat_dialogue_smoke` binary was driven against the same daemon
(now with autowire active). All 3 chat-integration gates STILL PASS:

| Gate | Observed | Verdict |
|---|---|---|
| `T_CHAT_DIALOGUE_RUNS` | HTTP 200, response_wall_ms=41781, response head ".C. The capital of Japan is Tokyo" | PASS |
| `T_CHAT_RECEIPTS_IN_RESPONSE` | receipt_count=3, all_64_bytes_after_decode=true, all_sentinel_match=true | PASS |
| `T_CHAT_NO_REGRESSION` | Build-time + Option B; no route changes to `/v1/chat` | PASS |

Ledger file size grew from 960 → 1152 bytes (= +192 = 3 × 64) over this
single regression dialogue — exactly the expected delta. The autowire
behaves identically across smoke binaries.

## Full JSON report (`tools/sp_daemon/scripts/ledger_autowire_report.json`)

```json
{
  "sprint": "ledger-autowire",
  "url": "http://127.0.0.1:8080/v1/dialogue",
  "ledger_path": "C:\\Users\\Knack\\AppData\\Local\\Temp\\autowire.spinor",
  "n_dialogues": 5,
  "pre_size": 0,
  "post_size": 960,
  "delta": 960,
  "expected": 960,
  "receipts_compared": 15,
  "byte_divergences": 0,
  "identity_run_err": "",
  "dialogues_with_status_200": 5,
  "dialogues_with_3_receipts": 5,
  "transport_errors": "",
  "total_wall_ms": 211706,
  "gates": {
    "T_AUTOWIRE_LEDGER_GROWS": "PASS",
    "T_AUTOWIRE_RECEIPT_BYTE_IDENTITY": "PASS",
    "T_AUTOWIRE_NO_REGRESSION": "PASS"
  }
}
```

Full smoke output capture (per-dialogue HTTP status + response head +
wall + 3-receipt sentinel check + per-receipt byte-identity audit) is
at `tools/sp_daemon/scripts/ledger_autowire_run.txt`. Chat-integration
regression spot-check report at
`tools/sp_daemon/scripts/ledger_autowire_chat_regression_report.json`.

## CLI flag spec

| Flag | Env var | Default | Required if | Effect |
|---|---|---|---|---|
| `--pouw-ledger-path <PATH>` | `SP_POUW_LEDGER_PATH` | `""` (empty = disabled) | never (optional) | Daemon opens `Ledger::open(PATH)` at startup; `v1_dialogue` auto-appends every receipt to the shared handle. Empty disables the autowire (HTTP response still ships receipts; nothing persists). |

Mirrors the existing `--memo-model` discipline (CLI flag + env var +
empty-string default = disabled). On non-empty path, the daemon
**bails at startup** if `Ledger::open` fails — operator catches typo'd
paths / missing parent dirs / write-permission issues immediately, not
at the first dialogue.

## AppState additions

In `tools/sp_daemon/src/state.rs::AppState`:

| Field | Type | Purpose |
|---|---|---|
| `ledger` | `Option<Arc<Mutex<Ledger>>>` (`// ledger-autowire:` prefix) | Shared append-only receipt ledger handle. `None` when `--pouw-ledger-path` is unset → `v1_dialogue` skips append. `Some` → handler grabs `lock()` → `append(&receipt)` × 3 per dialogue. |

Mutex around `Ledger` is required because `Ledger::append` takes
`&mut self`; concurrent `/v1/dialogue` requests would race on the file
handle otherwise. Per-append wall on M.4's S22U was p50=2 µs / p99=3 µs;
serialized critical section is ~10 µs per dialogue, structurally
irrelevant for an HTTP API serving 31-42 s dialogues.

The `Arc<>` wrapper makes the handle Send across the
`tokio::spawn_blocking` boundary into the dialogue worker's success
path.

## Behavior on autowire failure (best-effort)

Per the sprint plan + the broader M.4 design, the ledger is
**observational, not a transactional gate**:

- `Mutex::lock()` returns `Err(PoisonError)` → `tracing::warn!` + skip
  the append for this dialogue; HTTP 200 still returned with receipts.
- `Ledger::append()` returns `Err(LedgerError)` → `tracing::warn!` per
  failed receipt; HTTP 200 still returned with receipts.

A failed append never bubbles into the HTTP response status. This
matches the "Trick #10 receipt-backed audit trail" semantics — losing
an entry is a degradation, not a transactional rollback. The startup
open is the only failure that bails the daemon.

## Files changed

| File | Stage | Δ LOC | Notes |
|---|---|---|---|
| `tools/sp_compute_skel/docs/PLAN-LEDGER-AUTOWIRE.md` | plan (`b813aa0`) | +209 (new) | Stage 0 reference reading + 6 architecture decisions + 3-gate plan |
| `tools/sp_daemon/src/state.rs` | Stage 1 (`352e7d3`) | +9 | `pub ledger: Option<Arc<Mutex<Ledger>>>` field + use import |
| `tools/sp_daemon/src/main.rs` | Stage 1 (`352e7d3`) | +16 | hidden inner arg + visible Cmd::Start arg + plumbing |
| `tools/sp_daemon/src/daemon.rs` | Stage 1 (`352e7d3`) | +28 | `cmd_start` + `run_inner` extended; ledger open at startup; AppState populate |
| `tools/sp_daemon/src/routes.rs` | Stage 2 (`7ed9105`) | +28 net (CRLF-diff renders as +231/-203) | best-effort append loop in `v1_dialogue` success path |
| `tools/sp_daemon/Cargo.toml` | Stage 3 (`7831d74`) | +5 | new `[[bin]]` block for `sp_chat_ledger_autowire_smoke` |
| `tools/sp_daemon/src/bin/sp_chat_ledger_autowire_smoke.rs` | Stage 3 (`7831d74`) | +519 (new) | 3-gate smoke harness: HTTP POST × N + ledger size delta + 15×64 byte identity audit |
| `tools/sp_daemon/scripts/ledger_autowire_report.json` | Stage 3 (`7831d74`) | +1 | smoke JSON output |
| `tools/sp_daemon/scripts/ledger_autowire_run.txt` | Stage 3 (`7831d74`) | +smoke stderr | full smoke run capture (5 dialogues, all 3 gates) |
| `tools/sp_daemon/scripts/ledger_autowire_chat_regression_report.json` | Stage 3 (`7831d74`) | +1 | chat-integration smoke against autowire-active daemon (regression spot-check) |
| `tools/sp_compute_skel/docs/CLOSURE-LEDGER-AUTOWIRE.md` | Stage 4 | this file | closure |

**Files NOT touched** (concurrent `mesh-canonical-order` agent owns):
- `tools/sp_daemon/src/dialogue.rs` — frozen; concurrent agent extends
  with `with_sequence_rank()` / `sequence_rank()` helpers on
  SpinorReceipt over `_reserved[0..2]`.
- `tools/sp_daemon/src/pouw_ledger.rs` — frozen; concurrent agent adds
  `canonical_sort()` / `replay_canonical_into()` methods.

Zero overlap with the concurrent agent's expected touch surface. The
two lanes converge cleanly at merge time.

## Commits on `sprint/ledger-autowire`

| Commit | Stage | Summary |
|---|---|---|
| `b813aa0` | plan | `[plan] ledger-autowire -- capture /v1/dialogue receipts into shared PoUW ledger` |
| `352e7d3` | 1 | `[ledger-autowire] feat: Stage 1 -- AppState + --pouw-ledger-path CLI flag + daemon ledger init` |
| `7ed9105` | 2 | `[ledger-autowire] feat: Stage 2 -- v1_dialogue best-effort append to shared PoUW ledger` |
| `7831d74` | 3 | `[ledger-autowire] test: Stage 3 -- sp_chat_ledger_autowire_smoke (3/3 gates PASS on host MSVC)` |
| `<this>` | 4 | `[ledger-autowire] doc: Stage 4 -- closure with 3/3 PASS + chat-integration no-regression check` |

Base: `52e2145` (engine main post-chat-integration-merge,
post-M.4-merge, post-M.5-merge, post-M.2-merge).

## Sub-tag proposal

`lat-phase-4-memo-ledger-autowire` — parallel to
`lat-phase-4-memo-m4-ledger` (the standalone Ledger primitive) and
`lat-phase-4-memo-chat-integration` (the HTTP surface). This sub-tag
anchors the wire-in that closes Trick #10 for the chat path.

## What's NOT done (explicit deferral)

| Item | Why deferred |
|---|---|
| Real QUIC mesh broadcast | M.4 still ships `broadcast_to_peers()` as a stub; cross-lane work on `quic_shard.rs`. |
| Canonical mesh ordering (Garner-recombined sequence rank in `_reserved[0..2]`) | Concurrent `mesh-canonical-order` sprint owns this; they touch `dialogue.rs` + `pouw_ledger.rs`. |
| Ledger pruning / GC | M.4 ledger is append-only forever for v1. |
| Encryption-at-rest | M.4 ledger is plain bytes per scope. |
| ed25519-signed receipts | M.4 v1 ships unsigned; signing infrastructure exists in `mining.rs` and can be grafted later. |
| Per-tenant ledger | One ledger per daemon process for v1. |
| HTTP route to query ledger from outside | `/v1/pouw/ledger` GET already exists from M.4 for sieve-fold receipts (different format); adding a SpinorReceipt-shaped query route is a follow-on. |
| Crash-recovery from corrupted ledger | M.4 DETECTS via sentinel mismatch + partial-tail check; recovery (truncate + peer-replay) is M.4-next. |
| Live S22U validation | Host MSVC validation is sufficient — the autowire is pure host-Rust file I/O on `Ledger::append` which is cross-platform byte-identical (per `reference-cross-platform-byte-identity` from M.4). |

## What unblocks

| Sprint / Capability | Unblocked because |
|---|---|
| **Frontend audit log** | Every dialogue served by a daemon with `--pouw-ledger-path` set now has a persistent on-disk audit trail. A future frontend tab can GET-/v1/ledger-tail or read the file directly. |
| **Real mesh broadcast pulls from ledger** | M.4's `Ledger::broadcast_to_peers(since_offset)` is the call site; the QUIC fan-out lane can now read receipts from the autowire-populated ledger instead of the chat handler holding them ephemerally. |
| **M.6 cross-island MeMo per-device audit** | Each island runs its own daemon with its own ledger; the canonical-ordering sprint then merges them via Garner-recombined sequence rank. |
| **Trick #10 end-to-end for chat traffic** | The receipt-backed verifiable distributed compute story for `/v1/dialogue` is now closed: every turn mints a SpinorReceipt that lands in the ledger. |
| **Concurrent `mesh-canonical-order` sprint** | Their `replay_canonical_into()` consumes the ledger this sprint populates. Both lanes compose naturally. |

## Memory entry candidates

**None.** This sprint is pure HTTP-surface plumbing around already-locked
contracts (M.4 Ledger ABI + M.2 SpinorReceipt + chat-integration
`/v1/dialogue` shape). Nothing memory-worthy is uncovered. The CLI
flag name + AppState shape + best-effort append discipline are captured
in the closure + plan + commits and unlikely to need re-derivation.

If anything emerges from cross-merging with `mesh-canonical-order`, the
operator can capture it then.

## Worktree status

This sprint operates **exclusively from
`D:\F\shannon-prime-repos\engine-ledger-autowire`** on branch
`sprint/ledger-autowire` (base `52e2145`, 5 commits including this
closure). No other worktree was touched:

- `engine-mesh-order` (concurrent `sprint/mesh-canonical-order` agent):
  NOT TOUCHED. They own SpinorReceipt + Ledger helpers; I touch neither.
- `engine-chat`, `engine-m1`, `engine-m2`, `engine-m4`, `engine-m5`,
  `engine-k2-spike`, `engine-kbeta-*`: NOT TOUCHED.
- `shannon-prime-system-engine` (main worktree): READ-ONLY consulted
  for `SP_SYSTEM_INCLUDE` / `SP_SYSTEM_BUILD_DIR` (math-core headers +
  prebuilt static libs); no commits, no edits.
- `lattice-memo-m0` and `shannon-prime-system*`: NOT TOUCHED.
- Model artifacts (`D:\F\shannon-prime-repos\models\*`,
  `shannon-prime-system-engine/build-cpu/tests/*.sp-model`): NOT
  MODIFIED (read-only consumption by the running daemon).
- Test ledger file at `%TEMP%\autowire.spinor`: created by the daemon
  during smoke; auto-cleanable.

Per `feedback-parallel-agents-separate-worktrees`, the per-worktree
isolation discipline is preserved. The Stage 1 / Stage 2 commits all
touch the four daemon-orchestrator files (`state.rs`, `main.rs`,
`daemon.rs`, `routes.rs`); concurrent agent's expected touch surface
is `dialogue.rs` and `pouw_ledger.rs`. Zero file-level overlap.

## References

- `tools/sp_compute_skel/docs/PLAN-LEDGER-AUTOWIRE.md` — Stage 0
  reference reading + architecture decisions for this sprint.
- `tools/sp_compute_skel/docs/CLOSURE-CHAT-INTEGRATION.md` — the
  `/v1/dialogue` HTTP surface this sprint wires into.
- `tools/sp_compute_skel/docs/CLOSURE-M4-LEDGER.md` — the `Ledger`
  primitive this sprint consumes.
- `tools/sp_daemon/src/routes.rs:728-757` — best-effort append loop
  in the `v1_dialogue` success path.
- `tools/sp_daemon/src/state.rs:101-108` — `Option<Arc<Mutex<Ledger>>>`
  AppState field.
- `tools/sp_daemon/src/daemon.rs` — ledger startup open at run_inner
  + AppState populate.
- `tools/sp_daemon/src/pouw_ledger.rs:118-148` — the `Ledger::open` +
  `Ledger::append` API consumed verbatim from M.4.
- `tools/sp_daemon/src/dialogue.rs` — SpinorReceipt layout (NOT
  modified; the autowire consumes `as_bytes()` only).
- `~/memory/reference_spinor_receipt_layout.md` — 64-byte layout the
  ledger persists.
- `~/memory/reference_heterogeneous_soc_crt_tricks.md` Trick #10 — the
  receipt-backed verifiable distributed compute vision this sprint
  materializes for the chat path.
- `~/memory/feedback_no_silent_gate_revisions.md` — discipline rule;
  3/3 gates surfaced as honest PASS.
- `~/memory/feedback_parallel_agents_separate_worktrees.md` — worktree
  discipline; all commits from `engine-ledger-autowire`.
- `~/memory/feedback_bundled_changeset_root_cause_ambiguity.md` —
  one-variable-per-stage; 4 stage commits (plan + 3 implementation +
  closure) each with single discrete change.
