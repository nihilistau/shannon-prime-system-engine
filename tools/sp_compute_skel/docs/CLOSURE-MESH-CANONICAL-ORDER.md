# Sprint mesh-canonical-order — CLOSURE

**Headline:** Cross-device byte-identity gap left open by M.4 is **closed
on host**. Two devices holding the same receipt SET in different local
orders now produce SHA-256-identical canonical ledgers via
`Ledger::replay_canonical_into`. Three gates PASS:
`T_MESH_RANK_PROTOCOL`, `T_MESH_CANONICAL_SORT_DETERMINISTIC`,
`T_MESH_CROSS_DEVICE_BYTE_IDENTITY`.

**Branch:** `sprint/mesh-canonical-order`
**Base:** engine main @ `52e2145` (post Chat-integration merge).
**Worktree:** `D:\F\shannon-prime-repos\engine-mesh-order` (exclusive).
**Status:** all 4 stages shipped; host x86_64-pc-windows-msvc.
**Hardware:** host-only (no S22U dispatch needed).

## Gates table

| Gate | Verdict | Key metric |
|---|---|---|
| `T_MESH_RANK_PROTOCOL` | **PASS** | bytes[61..63] = `2a 00`; sentinel @ 63 = 0xA5; other 0..61 unchanged |
| `T_MESH_CANONICAL_SORT_DETERMINISTIC` | **PASS** | N=100; two runs SHA-256 = `2a5d9717...1bfb6b1c` (match) |
| `T_MESH_CROSS_DEVICE_BYTE_IDENTITY` | **PASS** | dev_a_sha = dev_b_sha = ref_sha = `174c7353...14db24f9`; raw devices diverge |

Complete artifact: run
`tools\sp_daemon\target\debug\sp_memo_m4_canonical_replay_smoke.exe
--report-json out.json` to reproduce.

## Protocol — canonical ordering rule

**Sort key:** `(sequence_rank, input_hash)` ascending.

**Where the key lives on wire (offsets in the 64-byte SpinorReceipt):**

| Offset | Bytes | Field | Role in this sprint |
|---|---|---|---|
| 0 | 1 | `turn_index: u8` | (existing M.2) — input_hash domain-sep |
| 1 | 1 | `model_id: u8` | (existing M.2) — input_hash domain-sep |
| 2-3 | 2 | `_pad: [u8; 2]` | (existing M.2) — zero |
| 4-7 | 4 | `wall_us: u32` | (existing M.2) |
| 8-31 | 24 | `input_hash: [u8; 24]` | **tiebreak on equal rank** (SHA-256 truncated) |
| 32-55 | 24 | `output_hash: [u8; 24]` | (existing M.2) |
| 56-59 | 4 | `n_input_tokens: u32` | (existing M.2) |
| 60 | 1 | `n_output_tokens: u8` | (existing M.2) |
| **61-62** | **2** | **`_reserved: [u8; 2]`** | **u16 sequence rank (LE)** |
| 63 | 1 | `sentinel: u8` | 0xA5 — invariant (preserved by `with_sequence_rank`) |

**Methods:**

- `SpinorReceipt::with_sequence_rank(self, rank: u16) -> Self`
  Returns a NEW receipt with `_reserved = rank.to_le_bytes()`. All other
  fields preserved bit-exact; 64-byte layout invariant preserved;
  sentinel byte preserved.

- `SpinorReceipt::sequence_rank(&self) -> u16`
  Reads `_reserved[0..2]` as LE u16. Returns 0 on unstamped receipts.

- `Ledger::canonical_sort(&self) -> LedgerResult<Vec<SpinorReceipt>>`
  Reads all records; sorts by `(sequence_rank ASC, input_hash ASC)`;
  returns sorted Vec. Does NOT touch on-disk file.

- `Ledger::replay_canonical_into(&self, dest: &mut Ledger) -> LedgerResult<usize>`
  Canonical-sorts source; appends to `dest` in sorted order; returns
  count. Cross-device byte-identity primitive: two devices' replays
  into fresh dests produce SHA-256-equal files when receipt SETS are
  equal.

**Replay behavior:**
- Source ledger is read-only — `replay_canonical_into` does NOT mutate it
  (verified by `replay_canonical_into_preserves_source` unit test).
- Destination is append-only — existing dest content is preserved; the
  canonical-sorted stream lands AFTER any pre-existing dest records.
- For cross-device byte-identity, callers should pass a fresh empty
  `dest`. (The smoke harness does this.)

## SpinorReceipt layout — what changed

**Bytes 0-60 + 63: invariant.** This sprint changes ZERO bytes outside
the `_reserved` field. The silicon-confirmed M.2 layout — turn_index,
model_id, _pad, wall_us, input_hash, output_hash, n_input_tokens,
n_output_tokens, sentinel — all preserved.

**Bytes 61-62: now u16 sequence rank (little-endian).** Previously
zero-filled; now optionally carries a rank stamped via
`with_sequence_rank`. A freshly-minted (unstamped) receipt still reads
rank=0.

**Byte 63: 0xA5 sentinel.** Preserved as the Trick #9 inter-island
integrity marker. Compile-time `size_of::<SpinorReceipt>() == 64`
assertion in `dialogue.rs:66` still holds.

## Files changed (LOC delta)

| File | LOC + | LOC - | Net | Notes |
|---|---|---|---|---|
| `tools/sp_compute_skel/docs/PLAN-MESH-CANONICAL-ORDER.md` | 331 | 0 | +331 | Stage 0 plan + structural correction |
| `tools/sp_daemon/src/dialogue.rs` | 112 | 0 | +112 | 36 prod + 49 test + 27 doc/comments |
| `tools/sp_daemon/src/pouw_ledger.rs` | 259 | 0 | +259 | 64 prod + 195 test |
| `tools/sp_daemon/src/bin/sp_memo_m4_canonical_replay_smoke.rs` | 396 | 0 | +396 | new smoke harness |
| `tools/sp_daemon/Cargo.toml` | 7 | 0 | +7 | new `[[bin]]` block |
| `tools/sp_compute_skel/docs/CLOSURE-MESH-CANONICAL-ORDER.md` | (this file) | 0 | new | closure doc |

**Total net add:** ~1100+ LOC across 6 files. Zero deletions. Zero
existing-test regressions (35/35 pass).

## Commits on `sprint/mesh-canonical-order`

| Hash | Subject |
|---|---|
| `c4115bd` | [plan] mesh-canonical-order — u16 rank in _reserved[0..2] (Option A; struct-layout correction) |
| `15c3e7a` | [mesh-canonical-order] Stage 1 -- SpinorReceipt::with_sequence_rank + sequence_rank helpers |
| `8bf4be3` | [mesh-canonical-order] Stage 2 -- Ledger::canonical_sort + replay_canonical_into |
| `26bf976` | [mesh-canonical-order] Stage 3 -- smoke harness + 3 gates ALL PASS |
| `(this)` | [mesh-canonical-order] Stage 4 -- closure |

Run `git log sprint/mesh-canonical-order ^52e2145 --oneline` to inspect.

## Sub-tag proposal

`lat-phase-4-memo-mesh-canonical-order` on the merge commit. Marks the
canonical-ordering primitive shipped on top of the M.4 ledger.

## What's NOT done (explicit deferrals)

Per the dispatch prompt's out-of-scope list:

- **Real QUIC mesh broadcast** — M.4's `broadcast_to_peers()` stub is
  unchanged; this sprint exercises it via the smoke harness but does
  NOT replace it with QUIC. The
  `network/quic_shard.rs:ResidueBlock`-shaped path is still the
  follow-on for SpinorReceipt-shaped broadcast.
- **Ledger pruning / garbage collection** — append-only forever in v1.
- **Encryption-at-rest for the ledger** — plain bytes on disk.
- **Per-receipt ed25519 signature** — M.4 v1 (and mesh-canonical-order
  v1) ship unsigned receipts. `mining.rs:21`'s ed25519 infrastructure
  can be grafted later; the canonical-sort key `(rank, input_hash)`
  does NOT depend on signature presence.
- **SpinorReceiptV2 with device_id on wire** — the dispatch prompt's
  Option B asked for `_reserved[2..4]` rank + `_reserved[4..8]`
  device_id. The actual `_reserved` field is 2 bytes total (offsets
  61-62) — Option B was structurally impossible without widening the
  receipt to 128 bytes or repurposing audit-chain bytes. Surfaced
  upstream in plan-commit Stage 0 per `feedback-no-silent-gate-
  revisions`; Option A (rank-only on-wire; input_hash tiebreaks)
  adopted. SpinorReceiptV2 sprint can add device_id at fresh offsets
  in a 128-byte v2 envelope.
- **N=10,000 / N=1,000,000 ledger canonical-sort stress** — N=100 was
  sufficient for the determinism + sort-correctness gates. Per-record
  sort cost is `O(N log N)` Vec sort + `O(N * 64)` linear file read;
  scales linearly to N>10^6 in seconds.
- **Cross-architecture (aarch64-android) confirmation** — host x86_64
  validates the layout. The struct is `#[repr(C, packed)]` so
  cross-arch byte order is implicit; a follow-up sprint can compile to
  aarch64-android and run on Knack's S22U for completeness.

## What unblocks

- **Manifesto Trick #10 — Receipt-backed verifiable distributed
  compute** flips from "Confirmed at M.4 scope — receipts + ledger
  shipped, cross-device order NOT canonical" to "Confirmed at
  mesh-canonical-order scope — cross-device byte-identity holds after
  canonical sort." The Roadmap trick-status table updates accordingly.

- **M.6 cross-island MeMo variant** — mesh-canonical-order is a
  prerequisite for the cross-island deployment story: any
  multi-machine deployment of Executive + Memory where receipts cross
  network boundaries can now converge to a canonical ledger
  independent of arrival order.

- **Real QUIC broadcast sprint** — the canonical-sort layer means QUIC
  re-ordering, packet loss + retransmit, and asymmetric peer-pair
  topologies are all tolerable: receivers sort by canonical key
  rather than depending on arrival order.

## Memory entry candidates

**UPDATE `reference-spinor-receipt-layout`:**

The existing memory note claims `_reserved` spans offsets 54-62 (9
bytes). The ACTUAL struct in `dialogue.rs:39-63` has `_reserved: [u8;
2]` at offsets 61-62 (after `n_input_tokens@56-59` and
`n_output_tokens@60`). Update the layout table to:

> `| 56-59 | n_input_tokens | u32 | input-token count`
> `| 60    | n_output_tokens | u8 | output-token count`
> `| 61-62 | _reserved | [u8; 2] | u16 sequence rank (LE), stamped via SpinorReceipt::with_sequence_rank — sprint mesh-canonical-order`
> `| 63    | sentinel | u8 | 0xA5 Trick #9 marker`

(Bytes 54-55 are the TAIL two bytes of `output_hash`, not `_reserved`.)

**ADD `reference-mesh-canonical-order` (NEW entry candidate):**

> Sprint mesh-canonical-order (2026-05-30) closed the M.4 cross-device
> divergence gap by adding u16 sequence rank to SpinorReceipt
> `_reserved[0..2]` (offsets 61-62, LE) + `Ledger::canonical_sort` +
> `Ledger::replay_canonical_into`. Sort key = `(rank, input_hash)`;
> input_hash provides device-disambiguating tiebreak via the existing
> 192-bit-entropy SHA-256-truncated digest. Three gates PASS
> (`T_MESH_RANK_PROTOCOL`, `T_MESH_CANONICAL_SORT_DETERMINISTIC`,
> `T_MESH_CROSS_DEVICE_BYTE_IDENTITY`). Trick #10 cross-device
> byte-identity confirmed on host. Dispatch-prompt's `(rank, device_id)`
> key was structurally impossible under existing 2-byte `_reserved`;
> Option A (rank-only on-wire) adopted per
> `feedback-no-silent-gate-revisions`.

## Worktree status

- **Worktree used:** `D:\F\shannon-prime-repos\engine-mesh-order`
  (exclusively).
- **Concurrent agent `ledger-autowire`:** `engine-ledger-autowire`
  worktree — NOT TOUCHED. No file overlap (this sprint extended
  `pouw_ledger.rs` with new methods; concurrent agent wires the
  existing `Ledger::append` into the chat handler).
- **No AppState changes** (matches M.4's "AppState additions: NONE"
  discipline).
- **`build-cpu` symlink** created at
  `D:\F\shannon-prime-repos\engine-mesh-order\build-cpu ->
  D:\F\shannon-prime-repos\shannon-prime-system-engine\build-cpu`
  for L1-static-lib resolution during `cargo test`. The link is a
  directory symlink; the worktree's git working tree ignores it
  (not added to the index). Operator can drop it post-merge.
- **`lib/shannon-prime-system` submodule** initialized in this
  worktree (`aeecdbae`); standard submodule, no changes.
- **Model artifacts in `D:\F\shannon-prime-repos\models\`:** NOT
  TOUCHED (host-only data-layer sprint).

## Discipline notes

- **`feedback-read-spec-before-drafting-handoff`:** Stage 0 caught the
  dispatch prompt's 9-byte `_reserved` assumption against the
  2-byte struct reality, surfaced UPSTREAM in plan-commit, and
  adopted Option A. NO silent layout widening.

- **`feedback-no-silent-gate-revisions`:** all three gates report the
  actual variables (SHA hashes, byte values, intermediate sizes);
  non-zero exit on any failure; no tolerance widening; no fixture
  tuning. The Option-B-impossible finding was surfaced upstream as a
  spec correction, not a silent gate revision.

- **`feedback-bundled-changeset-root-cause-ambiguity`:** four stages
  shipped as separate commits — plan / Stage 1 SpinorReceipt helpers /
  Stage 2 Ledger methods / Stage 3 smoke harness. Each stage is a
  single isolated variable. No bundling.

- **`feedback-parallel-agents-separate-worktrees`:** worktree
  exclusivity respected; no cross-contamination with
  `engine-ledger-autowire`.

- **`feedback-lead-with-reference-then-theory`:** Stage 0 read the
  actual struct + ledger code BEFORE designing the helper API;
  surfaced the layout divergence that the dispatch prompt missed.
