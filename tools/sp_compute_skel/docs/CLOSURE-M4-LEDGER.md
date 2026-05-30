# Sprint M.4 — PoUW receipt ledger + mesh replay simulation CLOSURE

**Status:** 4/4 substantive gates PASS on Knack's S22U (R5CT22445JA).
Plus 10/10 host unit tests PASS. Plus cross-platform byte-identical
SHA-256 across host (Windows MSVC) and aarch64-Android runs of the same
deterministic receipt sequence.

## Headline

The Trick #9 64-byte `SpinorReceipt` ABI (silicon-confirmed in M.2) is
now an **append-only ledger** at `~/.cache/shannon-prime/ledger.spinor`
(or any caller-chosen path), with **byte-level replay** that produces
SHA-256-identical destination files by construction. Cross-device
mesh-replay simulation shows the BYTE-LEVEL protocol works (Device A's
local-then-replay produces a file equal to canonical 0..n; Device B's
mirror produces a file equal to half..n ++ 0..half — predictable
ordering rules). Per-append wall on S22U: **p50 = 2 µs, p99 = 3 µs**.
Total wall for 1000 appends + fflush per call: **2 ms**. Manifesto
**Trick #10 (Receipt-backed verifiable distributed compute)** flips
from "Partial — receipts shipped; ledger pending" to **"Confirmed at
M.4 scope — ledger + replay shipped."**

## Gates table

All four substantive gates RUN on Knack's S22U (SM-S908E, Android 15,
MemTotal 11473784 KB) via `/data/local/tmp/sp_memo_m4_ledger_smoke
--n 1000 --workdir /data/local/tmp`. Per-record size = 64 bytes; total
file size at n=1000 = 64000 bytes.

| Gate | Method | Observed | Verdict |
|---|---|---|---|
| `T_M4_LEDGER_APPEND` | N=1000 SpinorReceipts appended via `Ledger::append()` | appends_succeeded=1000/1000; file_size_bytes=64000 (exact match); append_wall_ms=2; append_us_p50=2; append_us_p99=3; append_err="" | **PASS** |
| `T_M4_LEDGER_READ` | N=1000 read via `Ledger::iter()`; sentinel + byte-identity check | reads_succeeded=1000/1000; sentinel_failures=0; byte_divergences=0; read_err="" | **PASS** |
| `T_M4_REPLAY_DETERMINISTIC` | replay source→dest_a + source→dest_b; SHA-256 equality | replayed_a=1000; replayed_b=1000; dst_a_sha256 = dst_b_sha256 = main_sha256 = `43f303e1ad73376de1865a62012d3df5efbb33296de28fa40f328eb61b1f7a8b`; sha_a_b_match=true; sha_a_main_match=true | **PASS** |
| `T_M4_CROSS_DEVICE_REPLAY` | two simulated devices each mint half; broadcast (stub) + cross-replay; verify both final ledgers match expected order | device_a_sha256 == reference_sha256 = `43f303e1ad...`; device_b_sha256 == reverse_reference_sha256 = `66cdae3e56...`; device_a_size=device_b_size=64000; all_match=true | **PASS** |

## Full JSON report (`tools/sp_daemon/scripts/m4_full_report.json`)

```json
{
  "sprint": "M.4",
  "n_records": 1000,
  "workdir": "/data/local/tmp",
  "vmrss_start_kb": 2704,
  "ledger_append": {
    "appends_succeeded": 1000,
    "file_size_bytes": 64000,
    "append_wall_ms": 2,
    "append_wall_us_p50": 2,
    "append_wall_us_p99": 3,
    "append_err": ""
  },
  "ledger_read": {
    "reads_succeeded": 1000,
    "sentinel_failures": 0,
    "byte_divergences": 0,
    "read_err": ""
  },
  "replay_deterministic": {
    "replayed_a": 1000,
    "replayed_b": 1000,
    "dst_a_sha256": "43f303e1ad73376de1865a62012d3df5efbb33296de28fa40f328eb61b1f7a8b",
    "dst_b_sha256": "43f303e1ad73376de1865a62012d3df5efbb33296de28fa40f328eb61b1f7a8b",
    "main_sha256":  "43f303e1ad73376de1865a62012d3df5efbb33296de28fa40f328eb61b1f7a8b",
    "sha_a_b_match": true,
    "sha_a_main_match": true
  },
  "cross_device_replay": {
    "device_a_size_bytes": 64000,
    "device_b_size_bytes": 64000,
    "device_a_final_sha256": "43f303e1ad73376de1865a62012d3df5efbb33296de28fa40f328eb61b1f7a8b",
    "device_b_final_sha256": "66cdae3e5663a9caac6448c07cfb8b6b3fa7595970e36dd97bf28b2464b06fad",
    "reference_sha256": "43f303e1ad73376de1865a62012d3df5efbb33296de28fa40f328eb61b1f7a8b",
    "reverse_reference_sha256": "66cdae3e5663a9caac6448c07cfb8b6b3fa7595970e36dd97bf28b2464b06fad",
    "device_a_matches_reference": true,
    "device_b_matches_reverse": true,
    "all_match": true
  },
  "gates": {
    "T_M4_LEDGER_APPEND": "PASS",
    "T_M4_LEDGER_READ": "PASS",
    "T_M4_REPLAY_DETERMINISTIC": "PASS",
    "T_M4_CROSS_DEVICE_REPLAY": "PASS"
  },
  "vmrss_end_kb": 3212
}
```

## Cross-platform byte-identity bonus finding

The exact same 1000 deterministic receipts (synthesized via
`synth_receipt(i)` indexed 0..999) produce **byte-identical SHA-256**
on Windows MSVC release (host build) AND on aarch64-Android release
(S22U). Both runs produced `43f303e1ad73376de1865a62012d3df5efbb33296de28fa40f328eb61b1f7a8b`
for the main ledger.

This empirically confirms (a) the `#[repr(C, packed)]` layout is
ABI-portable across host x86_64 + aarch64-android, (b) SHA-256 over
identical inputs is identical across libraries (sha2 crate, same
algorithm by design), and (c) `Ledger::append()` writes the same
on-disk bytes regardless of OS. This is the load-bearing property
that makes cross-device mesh replay byte-identical "by construction"
per `reference-lattice-decode-determinism`.

## On-disk hexdump (silicon-confirmed format, first 3 records)

`tools/sp_daemon/scripts/m4_first3_hexdump.txt` captured via
`adb shell head -c 192 /data/local/tmp/m4_main.spinor | xxd`:

```
00000000: 010e 0000 d204 0000 5987 23be 9961 73fa  ........Y.#..as.
00000010: f936 75ef e60d 48bc 30af 9d7b af3c af39  .6u...H.0..{.<.9
00000020: cc92 aa1b c4af c0ba e228 101b ac1e daa9  .........(......
00000030: f427 46cf 4022 d397 0300 0000 0200 00a5  .'F.@"..........
00000040: 024d 0000 ba08 0000 3544 f79b 98bc 033f  .M......5D.....?
00000050: f06f 4546 5f51 d882 c749 c56f 4957 f0f4  .oEF_Q...I.oIW..
00000060: d300 4a0b 1b31 523d 182f 6ed7 6a94 3c7c  ..J..1R=./n.j.<|
00000070: b4e8 145d 439a ef71 0300 0000 0200 00a5  ...]C..q........
00000080: 030e 0000 a20c 0000 c999 eeb1 27ef 1b99  ............'...
00000090: 7c49 2472 2038 c32c 5f11 9180 25a6 0974  |I$r 8.,_...%..t
000000a0: 3ef9 0ca4 e5b7 0640 3762 e9cb f258 1875  >......@7b...X.u
000000b0: c29b 0ca3 f3ac e21c 0300 0000 0200 00a5  ................
```

Per-record offset audit (matches M.2 CLOSURE-M2-DIALOGUE.md table):
- Record 0: `01 0e` (turn=1 Executive); `d2 04 00 00` (wall_us=1234 = 0x4D2); `n_input_tokens=3` @offset 56; `n_output_tokens=02` @offset 60; **sentinel `a5` @offset 63** ✓
- Record 1: `02 4d` (turn=2 Memory); `ba 08 00 00` (wall_us=2234 = 0x8BA); sentinel `a5` ✓
- Record 2: `03 0e` (turn=3 Executive); `a2 0c 00 00` (wall_us=3234 = 0xCA2); sentinel `a5` ✓

## Ledger storage design

**Wire format:** plain stream of fixed-size 64-byte records (one
[`SpinorReceipt::as_bytes()`] per record). No file header. No record
separators. Corruption detection via `bytes[63] == 0xA5` sentinel check
on every read (the Trick #9 inter-island integrity ABI).

**Path scheme:** default `~/.cache/shannon-prime/ledger.spinor` is the
*documented intent*; the smoke harness uses `/data/local/tmp/m4_*.spinor`
on-device (where `~/.cache` does not exist for the `shell` user). The
constructor `Ledger::open(PathBuf)` accepts any caller-chosen path.

**Atomic-append discipline:** `std::fs::OpenOptions::new().create(true)
.append(true).open(path)` — POSIX `O_APPEND` guarantees per-write
atomicity up to PIPE_BUF (4096 bytes; our records are 64 bytes — safely
atomic). Windows `FILE_APPEND_DATA` gives the same per-write atomicity.
After each `append()`, `BufWriter::flush()` ensures the on-disk state
matches the in-memory `bytes_written` counter — crash after `append()`
returns leaves a consistent ledger.

**Multi-writer policy:** `Ledger::append(&mut self, ...)` takes `&mut
self` → borrow-checker serializes per-handle appends. Multi-thread
sharing requires `Arc<Mutex<Ledger>>`. Multi-process appends to the
same path are safe via `O_APPEND` alone on POSIX (Linux/Android) and
via `FILE_SHARE_WRITE` (default for `OpenOptions::append` on Windows).

**Crash recovery:** the iterator detects partial trailing records
(`file_size % 64 != 0`) and reports `LedgerError::PartialTail { extra_
bytes }`. Recovery from corruption is M.4-next (truncate to nearest
record boundary; replay missing range from peers).

## Mesh broadcast status

**Stub.** `Ledger::broadcast_to_peers(since_offset: u64) ->
LedgerResult<Vec<SpinorReceipt>>` returns the receipt list ready for
broadcast (with byte-aligned offset check). A future sprint wires this
into the QUIC infrastructure.

**Why a stub:** the existing `tools/sp_daemon/src/network/quic_shard.rs`
QUIC machinery is purpose-built for `ResidueBlock` (NTT shard payload =
`ShardBlockHeader` 64B + `Vec<u32>` residues; `quic_shard.rs:32-47`,
`send_block` at `quic_shard.rs:210-219`, `run_garner_loop` at
`quic_shard.rs:271`). A generic receipt-broadcast hook would either
(a) repurpose `send_block()` with a new BlockType (cross-lane impact
on `quic_shard.rs`), (b) add a `send_receipt_block(receipt:
&SpinorReceipt)` parallel surface (still cross-lane), or (c) build a
new fan-out coordinator (substantial new code). All three are
out-of-scope for M.4 per the prompt's explicit "M.4 simulates via two
ledger instances on same host" guidance.

The simulation in T_M4_CROSS_DEVICE_REPLAY proves the BYTE-LEVEL
protocol works (bytes-in → bytes-out is byte-identical), which is the
load-bearing claim. Real QUIC transport delivers byte-exact payload at
app layer; replacing the stub with a real broadcast does not change
the ledger ABI.

## AppState additions

**NONE.** Matching the M.2 / M.5 standalone-binary discipline (CLOSURE
-M2-DIALOGUE §"AppState additions"). The smoke harness owns its own
Ledger handles; daemon-side wiring (chat handler appends a receipt per
dialogue) is a follow-on sprint.

## Files changed

| File | Δ | Notes |
|---|---|---|
| `tools/sp_compute_skel/docs/PLAN-M4-LEDGER.md` | +371 | Stage 0 reference reading (file:line cites for 8 items) + architecture decisions + 4-gate plan |
| `tools/sp_daemon/src/pouw_ledger.rs` | +556 | NEW host-safe module: `Ledger`, `LedgerReplayer`, `LedgerIter`, `LedgerError`, `receipt_from_bytes`, `RECEIPT_BYTES`; 10 unit tests |
| `tools/sp_daemon/src/lib.rs` | +5 | `pub mod pouw_ledger;` with `// M.4 (ledger):` prefix (append-only; M.5's `memo_routing` block untouched) |
| `tools/sp_daemon/Cargo.toml` | +6 | `[[bin]] sp_memo_m4_ledger_smoke` block with `# §4-MeMo Sprint M.4 — pouw ledger smoke` comment prefix |
| `tools/sp_daemon/src/bin/sp_memo_m4_ledger_smoke.rs` | +573 | NEW Stage 2/3 binary: synth_receipt + Ledger drive + 4-gate harness + JSON report emitter |
| `tools/sp_daemon/scripts/m4_full_report.json` | +1 | Stage 3 production JSON (--n 1000) |
| `tools/sp_daemon/scripts/m4_full_run.txt` | +46 | Stage 3 stdout/stderr capture |
| `tools/sp_daemon/scripts/m4_first3_hexdump.txt` | +12 | Stage 3 on-disk hexdump of first 3 records (silicon-confirmed format) |
| `tools/sp_compute_skel/docs/CLOSURE-M4-LEDGER.md` | +THIS | this closure note |

**Lines touched outside `pouw_ledger.rs` + smoke harness:** 11 total —
5 in lib.rs (single block append after M.5's `pub mod memo_routing;`)
+ 6 in Cargo.toml (single `[[bin]]` block with M.4-prefixed comment to
make a future merge conflict trivially resolvable). All other changes
are in M.4-owned files (new module, new binary, new closure, new
docs).

**Cross-lane (Chat-integration) risk:** minimal. Chat-integration's
expected touch surface is the chat handler in `routes.rs` and possibly
AppState. M.4 did NOT touch either. The lib.rs and Cargo.toml additions
are append-only at separate locations from M.2's earlier additions and
follow the `// M.4 (ledger):` / `# §4-MeMo Sprint M.4 — pouw ledger
smoke` prefix conventions specified in the dispatch prompt to make
merge-time disambiguation trivial.

## Commits on `sprint/memo-m4`

| Commit | Stage | Summary |
|---|---|---|
| `0a945f4` | plan | `[plan] M.4 -- PoUW receipt ledger + mesh replay simulation` |
| `3660f86` | 1 | `[M.4] feat: Stage 1 -- Ledger + LedgerReplayer module (10/10 host unit tests PASS)` |
| `e94d87b` | 2 | `[M.4] feat: Stage 2 -- sp_memo_m4_ledger_smoke binary (4/4 host gates PASS)` |
| `b80bff8` | 3 | `[M.4] test: Stage 3 -- S22U run captured (4/4 gates PASS; cross-platform SHA-256 byte-identical host vs aarch64-android)` |
| (this commit) | 4 | `[M.4] doc: Stage 4 -- closure with 4/4 PASS + cross-platform byte-identity finding` |

Base: `a28d409` (engine main post-M.5-merge, post-M.2-merge).

## Proposed sub-tag

`lat-phase-4-memo-m4-ledger` (parallel to M.2's `lat-phase-4-memo-m2-
dialogue` and M.5's `lat-phase-4-memo-m5-routing`).

## What's NOT done in this sprint (explicit)

- **M.3 Frobenius-lifted TIES merge** — blocked on M.0-real same-arch
  Memory.
- **M.0-real SFT Memory artifact** — blocked upstream.
- **REAL cross-device mesh broadcast** — M.4 ships the
  `broadcast_to_peers()` stub returning the receipt list; real QUIC
  fan-out is a future sprint (existing `network/quic_shard.rs` is
  ResidueBlock-shaped; generic broadcast hook = cross-lane work).
- **M.6 cross-island MeMo variant** — needs K.2 full + M.0-real.
- **Encryption-at-rest** — M.4 ledger is plain bytes per the prompt's
  scope; encryption is a separate sprint.
- **Ledger pruning / garbage collection** — append-only forever for
  M.4; pruning is a future operational concern.
- **Recovery from corrupted ledger** — M.4 DETECTS via sentinel
  mismatch + partial-tail check; recovery (truncate to record
  boundary + peer-replay) is M.4-next.
- **ed25519-signed receipts** — M.4 v1 ships unsigned; ed25519
  infrastructure is present in `mining.rs:21` (Phase 5 sieve-fold
  receipts already sign with `signing_key.sign(&receipt)`) and can be
  grafted onto SpinorReceipts later. The per-receipt vs per-block vs
  per-merkle-root signing granularity is a separate design call.
- **AppState integration / `/v1/memo/ledger/append` HTTP route** —
  matches M.2 / M.5 deferral pattern.
- **Canonical cross-device ordering rule** — the T_M4_CROSS_DEVICE_
  REPLAY gate shows that without an ordering rule, device A and
  device B end up with byte-DIFFERENT ledgers even with identical
  receipt SETS (because merge order is local-first-then-broadcast).
  A canonical-ordering sprint would (e.g.) use Garner-recombined
  sequence numbers stored in the `_reserved[2]` bytes (per
  `reference-spinor-receipt-layout` "What goes in _reserved").
- **N=10,000 or N=1,000,000 ledger stress** — N=1000 is sufficient
  for the gate definition; the per-append timings (p50=2µs, p99=3µs)
  suggest the ledger scales linearly to N>10^6 in <30s wall.
- **AppState `Arc<Mutex<Ledger>>` factory** — needed when chat
  handler integration lands.

## What unblocks now

- **Manifesto Trick #10 "Receipt-backed verifiable distributed
  compute"** flips from "Partial — receipts shipped; ledger pending"
  to "Confirmed at M.4 scope — ledger + replay shipped". The Roadmap
  trick-status table updates accordingly.
- **Chat handler integration**: the existing chat handler can now do
  `state.ledger.lock().unwrap().append(&receipt)?` per dialogue (when
  AppState extension lands in a follow-on sprint, the Mutex pattern is
  the natural fit per `mining.rs`'s `Arc<Mutex<Vec<ReceiptRecord>>>`).
- **Real mesh broadcast sprint**: `Ledger::broadcast_to_peers()` is
  the call site; replace its body with QUIC `send_block`-equivalent
  calls. Receipt wire format is `SpinorReceipt::as_bytes()` (64 bytes
  per UDP frame fits any MTU).
- **M.4 + ed25519-signed extension**: graft `signing_key.sign(&receipt
  .as_bytes())` into `Ledger::append`'s wrapper layer; emit a parallel
  signature ledger or extend SpinorReceiptV2 to 128 bytes with the
  64-byte signature appended.
- **Canonical-ordering sprint**: the `_reserved[2]` bytes can hold a
  u16 sequence rank; Garner recombination at the mesh layer produces
  the canonical global sequence number; ledger replayer sorts by it
  before appending.

## Memory entry candidates

1. **NEW** `reference-pouw-ledger-format` — captures the file format
   (plain stream of 64-byte records, sentinel detection, partial-tail
   recovery), the atomic-append discipline (`OpenOptions::append` →
   `O_APPEND`/`FILE_APPEND_DATA` per-write atomicity), and the
   cross-platform byte-identity property (host SHA-256 == android
   SHA-256 over identical deterministic inputs). Load-bearing for
   future ledger consumers (M.4-next recovery, ed25519 sign extension,
   real mesh broadcast, AppState integration).

2. **NEW / UPDATE** `reference-spinor-receipt-layout` — append a note
   that the T_M4_CROSS_DEVICE_REPLAY simulation confirms the layout
   is `Vec<SpinorReceipt>`-broadcast-safe (no extra framing needed;
   the 64-byte record IS the wire payload). Update the "What goes in
   _reserved" section: M.4 leaves all 2 bytes zero; a future canonical
   ordering sprint will likely use these for a u16 sequence rank.

3. **NEW** `reference-cross-platform-byte-identity` — captures the
   observed empirical pattern: deterministic Rust code over
   `#[repr(C, packed)]` POD structs + sha2 crate over byte arrays
   produces byte-identical results across x86_64-msvc (Windows) +
   aarch64-android. This validates `reference-lattice-decode-
   determinism`'s "cross-backend" caveat for the orchestration-side
   data path (the ledger doesn't touch the L1 forward, so the
   "fp16-adjacent" caveat is moot here). Generalizable: any future
   pure-Rust byte-stream component (PoUW receipt mesh, KSTE tree
   serialization, etc.) inherits this property.

## Worktree status

- Worktree: `D:\F\shannon-prime-repos\engine-m4` exclusively.
- Branch: `sprint/memo-m4` (base `a28d409` = engine main + M.5 + M.2
  merged).
- All commits authored from THIS worktree.
- Concurrent Chat-integration agent: ran in `engine-chat` worktree —
  NOT TOUCHED. M.4's changes are entirely in `pouw_ledger.rs` (new),
  `sp_memo_m4_ledger_smoke.rs` (new), 5-line `lib.rs` append, 6-line
  `Cargo.toml` append. Zero overlap with the chat handler in
  `routes.rs` and zero AppState modifications, so Chat-integration's
  merge should be conflict-free.
- Other worktrees (`engine-m2`, `engine-m5`, `engine-m1`, `engine-k2-
  spike`, `engine-kbeta-2-5b`, `engine-kbeta-2-5c`, `lattice-memo-
  m0`): NOT TOUCHED.
- Main `shannon-prime-system-engine`: READ-ONLY consulted (bindgen
  headers via `SP_SYSTEM_INCLUDE` + android-libs via
  `SP_SYSTEM_BUILD_DIR` pointing at main worktree's prebuilt
  artifacts).
- Model artifacts (`/data/local/tmp/qwen*`): NOT TOUCHED (M.4 is
  L1-free; uses synthetic deterministic receipts).
- On-device ledger files (`/data/local/tmp/m4_*.spinor`): created by
  the smoke binary; teardown via `rm -f` is in the smoke wrapper
  invocation.

## References

- `papers/PPT-LAT-Roadmap.md:5896-5904` (lattice main) — M.4 spec
  ("Receipts accumulate in append-only ledger... mesh replay... Z_q
  makes replay exact").
- `tools/sp_compute_skel/docs/PLAN-M4-LEDGER.md` — Stage 0 reference
  reading + architecture decisions for this sprint.
- `tools/sp_compute_skel/docs/CLOSURE-M2-DIALOGUE.md` — M.2 closure
  with silicon-confirmed SpinorReceipt layout table (offsets 0..63).
- `tools/sp_daemon/src/dialogue.rs:39-63` — `SpinorReceipt` struct.
- `tools/sp_daemon/src/dialogue.rs:66` — compile-time `size_of == 64`
  guard.
- `tools/sp_daemon/src/dialogue.rs:96-120` — `SpinorReceipt::mint`.
- `tools/sp_daemon/src/dialogue.rs:123-135` — `SpinorReceipt::as_bytes`
  (the M.4 wire format input).
- `tools/sp_daemon/src/mining.rs:1-178` — pre-existing Phase 5 PoUW
  signing infrastructure (152-byte SPRCPT01 sieve-fold format; M.4
  is a distinct 64-byte SpinorReceipt format that coexists).
- `tools/sp_daemon/src/network/quic_shard.rs:271` — `run_garner_loop`
  the existing QUIC accept loop (ResidueBlock-shaped; future generic
  broadcast hook lives elsewhere).
- `~/memory/reference_spinor_receipt_layout.md` — the 64-byte layout
  this ledger consumes verbatim.
- `~/memory/reference_heterogeneous_soc_crt_tricks.md` Trick #10 —
  the receipt-backed-verifiable-compute vision M.4 materializes.
- `~/memory/reference_lattice_decode_determinism.md` — the
  determinism invariant that makes replay byte-exact by construction.
- `~/memory/feedback_no_silent_gate_revisions.md` — discipline rule;
  M.4 surfaces the cross-device ORDERING finding as a documented
  not-done (canonical-ordering sprint) rather than redefining the
  gate.
- `~/memory/feedback_lead_with_reference_then_theory.md` — Stage 0
  reference-first workflow; PLAN cited file:line for items 1-5 and 8.
- `~/memory/feedback_parallel_agents_separate_worktrees.md` —
  worktree discipline; all M.4 commits from engine-m4 only.
- `~/memory/feedback_bundled_changeset_root_cause_ambiguity.md` —
  one-variable-at-a-time per stage; M.4 made 4 stage commits (plan +
  3 implementation + closure) each with single discrete change.
