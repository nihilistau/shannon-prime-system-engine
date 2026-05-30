# Sprint M.4 — PoUW receipt ledger + mesh replay simulation PLAN

## Sprint summary

Turn the M.2 SpinorReceipts (64-byte cache-line audit envelopes) into an
**append-only ledger** + a **mesh-replay simulation** so that the
distributed continual learning vision (Manifesto Trick #10) is structurally
realized at the orchestrator layer.

Status entering: M.2 closed (sub-tag `lat-phase-4-memo-m2-dialogue`,
SpinorReceipt 64-byte ABI silicon-confirmed); M.5 closed
(`lat-phase-4-memo-m5-routing` Variant B); engine main @ `a28d409`. M.4
worktree is `engine-m4`, branch `sprint/memo-m4`, base = a28d409.

This sprint ships the LEDGER + the REPLAY-DETERMINISM gate on real wire
bytes (64-byte SpinorReceipts from `dialogue.rs`). The mesh-broadcast
*real* transport is filed UPSTREAM: existing QUIC code in
`network/quic_shard.rs:271` (`run_garner_loop`) is purpose-built for
ResidueBlock (header+coeffs) sharding, NOT for generic receipt fan-out;
adding a receipt-broadcast hook is itself a future sprint. M.4 ships the
hook as a **stub returning the receipt list** so an operator can pipe it
through real QUIC later without changing the ledger API.

## Stage 0 — reference reading (per `feedback-lead-with-reference-then-theory`)

### Item 1 — M.2 closure (SpinorReceipt layout + `as_bytes()` site)

- File: `tools/sp_compute_skel/docs/CLOSURE-M2-DIALOGUE.md:94-117`
  — silicon-confirmed layout table (sentinel 0xA5 at offset 63 verified
  by hexdump from S22U).
- File: `tools/sp_daemon/src/dialogue.rs:39-63` — `SpinorReceipt` struct
  `#[repr(C, packed)]` with compile-time `size_of == 64` guard at
  `dialogue.rs:66`.
- File: `tools/sp_daemon/src/dialogue.rs:123-135` — `SpinorReceipt::as_bytes()`
  method (returns `[u8; 64]` via `core::ptr::copy_nonoverlapping`).
  **This is the wire format M.4 consumes.**
- File: `tools/sp_daemon/src/dialogue.rs:96-120` — `SpinorReceipt::mint()`
  factory used by all minters.

**NOTE on the layout offsets.** The M.2 actual struct (verified at
`dialogue.rs:39-63`) differs slightly from the layout described in the
`reference-spinor-receipt-layout` memory entry. The actual struct is:
`turn_index@0, model_id@1, _pad[2]@2-3, wall_us@4-7, input_hash[24]@8-31,
output_hash[24]@32-55, n_input_tokens@56-59, n_output_tokens@60,
_reserved[2]@61-62, sentinel@63`. Both forms agree on: 64 bytes total,
sentinel 0xA5 at offset 63, packed C layout. M.4 honors the actual
struct's layout — what `as_bytes()` produces is what we store; the M.2
CLOSURE confirmed it.

### Item 2 — `reference-spinor-receipt-layout` memory (cite)

Layout table referenced verbatim: 64 bytes, sentinel 0xA5 at offset 63
("Trick #9 inter-island integrity ABI"). The memory entry's "wire format
compatibility" section explicitly names M.4 PoUW ledger consumer of
`SpinorReceipt::as_bytes()`.

### Item 3 — Phase 4-MeMo M.4 spec (Roadmap)

- File: `D:\F\shannon-prime-repos\shannon-prime-lattice\papers\PPT-LAT-Roadmap.md:5896-5904`
  — verbatim: "Receipts accumulate in append-only ledger. Two devices
  independently merge different task vectors. Receipts broadcast over CRT
  mesh (existing Phase 4-PoUW framework). Each peer Garner-replays
  incoming receipts. Gate T_MEMO_MESH_REPLAY: device A's Memory model
  state after replaying device B's receipt sequence is byte-identical to
  device B's Memory model state after the same sequence. This is the
  load-bearing distributed-learning gate. Uniquely SP because Z_q makes
  replay exact (no FP drift accumulates)."

The M.0-real Memory artifact is NOT shipped yet (the spec assumes M.3
merge is feasible; it is not — that's blocked). M.4 ships the LEDGER +
REPLAY DETERMINISM gates on byte-level. The "Memory model state byte-
identical" claim resolves to: same receipt bytes appended in same order
produce same ledger file (SHA-256 equal). This is the load-bearing
substrate; the model-state-equivalence claim is a derivation when M.3
ships.

### Item 4 — existing PoUW + ed25519-dalek infrastructure

- `tools/sp_daemon/Cargo.toml:52` — `ed25519-dalek = { version = "2", features = ["rand_core"] }`
- `tools/sp_daemon/Cargo.toml:58` — `sha2 = "0.10"` (added by M.2)
- `tools/sp_daemon/src/mining.rs:1-178` — existing Phase 5 PoUW receipt
  minter. **Important context:** this is a SEPARATE receipt format —
  the 152-byte "SPRCPT01" sieve-fold receipt for Friedman-sieve
  dominance events (`mining.rs:52-67` `pack_receipt`). It is signed
  via `signing_key.sign(&receipt)` at `mining.rs:144`. M.4's
  SpinorReceipt is a DIFFERENT 64-byte format for dialogue-turn audit.
  The two formats coexist; M.4 ledger is for SpinorReceipts.
- `tools/sp_daemon/src/state.rs:28-36` — `ReceiptRecord` struct +
  `AppState::receipt_store: Arc<Mutex<Vec<ReceiptRecord>>>` already
  exists for sieve-fold receipts. M.4 ledger is FILE-based (not
  in-memory Vec) since the load-bearing property is append-only
  durability across daemon restarts. M.4 does NOT extend
  `receipt_store` — that's for the legacy sieve format.

**M.4 disposition: unsigned ledger.** Per spec — M.4 v1 ships unsigned
for simplicity; signature is a future sprint. ed25519 infrastructure is
present and can be grafted on cleanly once we decide whether
per-receipt signing is the right granularity vs per-block / per-merkle-
root signing.

### Item 5 — existing mesh QUIC infrastructure

- `tools/sp_daemon/src/network/quic_shard.rs:271` — `run_garner_loop`
  is the QUIC accept loop for ResidueBlock sharding (a different
  workload). Block format at `quic_shard.rs:32-47` is `ShardBlockHeader`
  (64 bytes) + `Vec<u32>` residues; this is NTT-shard-specific.
- `tools/sp_daemon/src/network/quic_shard.rs:210-219` — `send_block()`
  opens a uni-stream per ResidueBlock.
- **No generic broadcast-to-peers surface.** A receipt broadcast hook
  would either (a) repurpose `send_block()` with a new BlockType, (b)
  add a `send_receipt_block(receipt: &SpinorReceipt)` parallel surface,
  or (c) build a new fan-out coordinator. **All three are out-of-scope
  for M.4 (would be cross-lane impact in `quic_shard.rs`).**
- M.4 disposition: `Ledger::broadcast_to_peers()` returns the receipt
  list as a `Vec<SpinorReceipt>` (stub); a follow-on sprint can wire
  this into one of (a)/(b)/(c) above without changing the ledger API.

### Item 6 — `reference-heterogeneous-soc-crt-tricks` Trick #10

Cited in prompt; ledger materializes the "receipts accumulate + peers
verify" claim at the on-device + cross-device scope. M.4 closes the
gap between "receipts are minted" (M.2) and "receipts are a verifiable
audit trail" (M.4 ledger + replay).

### Item 7 — `reference-lattice-decode-determinism`

The replay-determinism gate (T_M4_REPLAY_DETERMINISTIC) rests on this:
if receipts are bit-exact (they are; SHA-256 hashes domain-separated
per-turn-per-model), then replay is bit-exact by construction. The
gate verifies via SHA-256-equal-final-ledgers, not via inference
re-run. This is the precondition-respecting strict-equality regime.

### Item 8 — `feedback-no-silent-gate-revisions`

Mandatory discipline. If a gate cannot be met, surface UPSTREAM with
A/B/C paths; do not silently revise.

## Architecture decisions

### Ledger file format

**Plain stream of fixed-size 64-byte records.**

- No header (the file IS the data; corruption is detected per-record
  via sentinel check).
- No separators (records are fixed-size).
- File size invariant: `file_size % 64 == 0` always holds when not
  mid-write. Partial trailing record (size % 64 != 0) means crash
  mid-append; the partial record is discarded on read.

**Why not a more complex format (CBOR, capnproto, even length-prefix):**
- The SpinorReceipt is already a self-contained 64-byte ABI per Trick
  #9. Adding a framing layer duplicates the role of the sentinel.
- Random-access (skip to record N) is `seek(N * 64)` — O(1).
- Cross-language consumers (e.g. future iOS / Linux daemons) can parse
  this with two operations: `read(64)` + `verify(buf[63] == 0xA5)`.

### Ledger storage path

**`~/.cache/shannon-prime/ledger.spinor`** as default.

- Configurable via `Ledger::open(PathBuf)` so per-instance paths work
  (the cross-device-replay smoke uses two paths).
- On Android (S22U), `~/.cache/...` does not exist; the smoke harness
  uses `/data/local/tmp/m4_ledger_*.spinor`.

### Atomic-append discipline

Use `std::fs::OpenOptions::new().create(true).append(true).open(path)`:
- POSIX/Linux guarantees `O_APPEND` writes are atomic up to PIPE_BUF
  (4096 bytes) — far more than our 64-byte records. Per-record
  atomicity is guaranteed.
- Windows: `FILE_APPEND_DATA` flag (via `OpenOptions::append`) gives
  the same per-write atomicity for sub-page writes.

### Multi-writer policy

`Ledger::append(&mut self, ...)` takes `&mut self`. Borrow checker
serializes per-Ledger-handle appends. Multi-daemon-thread appends to
the SAME path require either (a) one ledger handle shared via
`Arc<Mutex<Ledger>>`, or (b) two file handles both opened with
`O_APPEND`. M.4 documents the choice; the smoke harness uses one
handle, single-threaded.

For cross-process appends (two daemons on same path — not the M.4
case but possible future), `O_APPEND` alone is sufficient on POSIX;
on Windows, file-share-write is needed (already the default for
`OpenOptions::append`).

### Replay determinism

`LedgerReplayer::replay_from(source: &Ledger, into: &mut Ledger)`
opens the source for read, iterates 64-byte records, appends each to
`into`. Result is byte-identical IFF source has no partial records
AND `into` was empty before replay.

The cross-device case simulates two devices via two ledger paths.
Mesh broadcast is the stub. The simulation proves the BYTE-LEVEL
protocol works (bytes-in → bytes-out invariant) — which is the
load-bearing claim. Real QUIC transport changes the bytes (TLS frame
overhead), but the receipt payload is what we care about; QUIC
delivers byte-exact payload at app layer.

## File plan

NEW:
- `tools/sp_daemon/src/pouw_ledger.rs` (~ 300 LOC) — Ledger struct +
  LedgerReplayer + host-side unit tests (host-buildable; no L1 ABI
  needed — purely orchestration-layer).
- `tools/sp_daemon/src/bin/sp_memo_m4_ledger_smoke.rs` (~ 250 LOC) —
  smoke harness with 4 gates. Host-buildable (it does NOT need L1;
  it synthesizes receipts via `SpinorReceipt::mint()` with synthetic
  token streams, exercising ledger + replay in isolation from the
  dialogue loop). On Android the same binary runs.

EDIT:
- `tools/sp_daemon/src/lib.rs` (+3 LOC) — add `pub mod pouw_ledger;`
  with `// M.4 (ledger):` prefix.
- `tools/sp_daemon/Cargo.toml` (+5 LOC) — register `[[bin]]
  sp_memo_m4_ledger_smoke` block. Comment block prefix: `# §4-MeMo
  Sprint M.4 — pouw ledger smoke`.

CLOSURE:
- `tools/sp_compute_skel/docs/CLOSURE-M4-LEDGER.md` — final closure.

**No AppState changes.** The smoke harness is standalone; daemon-side
ledger wiring (chat handler appends a SpinorReceipt per dialogue) is
filed as a follow-on sprint (matches M.2's discipline: M.2 shipped
`run_dialogue()` without AppState integration; M.4 ships the ledger
without AppState integration).

**No `dialogue.rs` changes.** M.4 IMPORTS from `dialogue.rs`
(SpinorReceipt struct, MODEL_ID_EXECUTIVE, MODEL_ID_MEMORY, mint());
does not modify.

**No chat handler changes** (Chat-integration's lane).

## Gates

### T_M4_LEDGER_APPEND

**Method:** synthetic-mint 1000 SpinorReceipts. Each receipt has a
distinct `(turn_index, model_id, wall_us, token streams)`. Append each
via `Ledger::append()`. Verify final file size = 1000 × 64 = 64000
bytes. Measure per-append wall (p50, p99 via sorted timings).

**PASS criterion:** appends_succeeded == 1000 AND file_size_bytes ==
64000 AND no errors.

**Report fields:** `appends_succeeded, file_size_bytes,
append_wall_us_p50, append_wall_us_p99`.

### T_M4_LEDGER_READ

**Method:** read back N=1000 receipts via `Ledger::iter()`. For each,
compare to the originally-appended byte slice (stored in a parallel
in-memory `Vec<[u8; 64]>` for cross-check). Verify sentinel 0xA5 at
offset 63 on every record.

**PASS criterion:** reads_succeeded == 1000 AND sentinel_failures == 0
AND byte_divergences == 0.

**Report fields:** `reads_succeeded, sentinel_failures,
byte_divergences`.

### T_M4_REPLAY_DETERMINISTIC

**Method:** create source ledger with 1000 receipts (from above). Open
two fresh empty destination ledger files. Replay source → dest_a;
replay source → dest_b. Compute SHA-256 of dest_a file + dest_b file
(via `sha2::Sha256`, already in deps). Assert hashes equal.

**PASS criterion:** dest_a_sha256 == dest_b_sha256 (hex string equal).

**Report fields:** `dest_a_sha256, dest_b_sha256, sha256_match`.

### T_M4_CROSS_DEVICE_REPLAY

**Method:** simulate two devices.
1. Mint receipts 1..500 (label them "device A").
2. Mint receipts 501..1000 (label them "device B"). Distinguish via
   `model_id` low bit + `turn_index` ranges so we can verify each
   half came from the right "device" on inspection.
3. Open `device_a.spinor` ledger; append 1..500 via Ledger::append.
4. Open `device_b.spinor` ledger; append 501..1000.
5. Simulate broadcast via `Ledger::broadcast_to_peers(0)` returning
   the full receipt list of each ledger (stub).
6. "Device A" replays device B's broadcast list into its own
   `device_a.spinor`. Now device_a.spinor contains 1..1000.
7. "Device B" replays device A's broadcast list into its own
   `device_b.spinor`. Now device_b.spinor contains 501..1000 ++
   1..500 (DIFFERENT order from device A!).
8. Build a `reference.spinor` ledger with 1..1000 in canonical order.
9. Compute SHA-256 of all three files.
10. Cross-device replay determinism claim:
    `device_a_final_sha256 == reference_sha256`
    AND `device_b_final_sha256` does **NOT** necessarily equal
    `reference_sha256` because the merge order differs.

This exposes a real architectural issue: **without a canonical
ordering rule, mesh-replayed ledgers diverge byte-wise across devices
even with identical receipt SETS.** The gate measures this and surfaces
it.

**PASS criterion (revised after Stage 1 design):**
- `device_a_final_sha256 == reference_sha256` (device A's replay-after-
  local matches the canonical 1..1000 order).
- `device_b_final_sha256 == reverse_reference_sha256` (device B's
  replay-after-local matches the reverse 501..1000 ++ 1..500 order).
- Both observations report cleanly; ledger ORDER is not promised to
  match across devices in v1 (a future canonical-ordering sprint would
  fix this, e.g. via Garner-recombined sequence numbers).

**Report fields:** `device_a_final_sha256, device_b_final_sha256,
reference_sha256, reverse_reference_sha256, device_a_matches_reference,
device_b_matches_reverse, all_match`.

Where `all_match = device_a_matches_reference AND
device_b_matches_reverse`.

**This is the load-bearing finding of M.4.** Either:
- (a) it passes as-described → ledger v1 ships, ordering caveat
  documented as known-future-sprint; OR
- (b) one of the byte-identical claims FAILS → root-cause + surface
  UPSTREAM (almost certainly a code bug, not a spec gap).

## Stages

- **Stage 0 (this plan).** Reference reading + plan commit.
- **Stage 1.** `pouw_ledger.rs` (Ledger + LedgerReplayer + unit tests,
  host-only). Commit between.
- **Stage 2.** `sp_memo_m4_ledger_smoke.rs` smoke harness + lib.rs +
  Cargo.toml registration. Host build green. Commit between.
- **Stage 3.** Android cross-build + adb push + on-device run.
  Capture report JSON. Commit between.
- **Stage 4.** Closure doc.

## Hardware

S22U (R5CT22445JA) confirmed online (`adb devices`). Smoke harness
binary lives on `/data/local/tmp/sp_memo_m4_ledger_smoke`. Ledger
files: `/data/local/tmp/m4_*.spinor`. Tear down between runs.

## What's NOT in this sprint

- M.3 Frobenius-lifted TIES merge (blocked on M.0-real same-arch Memory)
- M.0-real SFT Memory artifact (blocked upstream)
- REAL cross-device mesh broadcast (filed; QUIC `quic_shard.rs` is
  ResidueBlock-shaped, generic broadcast = future)
- M.6 cross-island variant
- Encryption-at-rest for ledger (separate sprint)
- Ledger pruning / garbage collection (append-only forever)
- Recovery from corrupted ledger (M.4 detects via sentinel; recovery
  is M.4-next)
- ed25519-signed receipts (M.4 v1 ships unsigned; sign infrastructure
  is present in mining.rs but per-receipt signing granularity decision
  is a separate sprint)
- AppState integration / `/v1/memo/ledger/append` HTTP route (matches
  M.2 deferral pattern)
- Canonical cross-device ordering (Garner-sequenced; future sprint)

## What unblocks after M.4

- Manifesto Trick #10 flips from "Partial — receipts shipped; ledger
  pending" to "Confirmed at M.4 scope — ledger + replay".
- M.3 receipts (when M.0-real ships) drop into the same ledger.
- Auditable MeMo deployment: any node's Memory model state
  reconstructable from ordered receipt sequence.
- Future canonical-ordering sprint can graft on without changing the
  64-byte SpinorReceipt ABI (use `_reserved[2]` bytes for a sequence
  rank, or extend to a 128-byte SpinorReceiptV2).
