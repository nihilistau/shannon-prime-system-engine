# Sprint mesh-canonical-order — u16 sequence rank in SpinorReceipt — PLAN

**Branch:** `sprint/mesh-canonical-order` (base: engine main @ `52e2145`)
**Worktree:** `D:\F\shannon-prime-repos\engine-mesh-order` (exclusive)
**Discipline:** `feedback-no-silent-gate-revisions`, `feedback-read-spec-
before-drafting-handoff`, `feedback-lead-with-reference-then-theory`,
`feedback-bundled-changeset-root-cause-ambiguity`, `feedback-parallel-
agents-separate-worktrees`.

## Stage 0 — Reference reading (file:line cites required)

### 1. M.4 closure: cross-device ordering NOT canonical in v1

`tools/sp_compute_skel/docs/CLOSURE-M4-LEDGER.md:263-269` explicit
not-done:

> "**Canonical cross-device ordering rule** — the T_M4_CROSS_DEVICE_
> REPLAY gate shows that without an ordering rule, device A and
> device B end up with byte-DIFFERENT ledgers even with identical
> receipt SETS (because merge order is local-first-then-broadcast).
> A canonical-ordering sprint would (e.g.) use Garner-recombined
> sequence numbers stored in the `_reserved[2]` bytes (per
> `reference-spinor-receipt-layout` 'What goes in _reserved')."

The follow-on hook is the `_reserved` field of the SpinorReceipt — note
M.4 closure references `_reserved[2]` (TWO bytes), not the nine-byte
range the dispatch prompt assumes. See §3 below for the divergence.

### 2. `reference-spinor-receipt-layout` (memory note)

`~/memory/reference_spinor_receipt_layout.md:14-22` — the layout TABLE
in the memory note claims:

> `| 54-62 | _reserved | [u8; 9] | zero-filled; future ledger metadata`

And invariant rule at line 24:

> `Layout invariant: size_of::<SpinorReceipt>() == 64`

**This memory note is OUT-OF-DATE relative to the ACTUAL struct in
`dialogue.rs:39-63`.** See §3.

### 3. M.4 implementation: where to insert canonical-sort

`tools/sp_daemon/src/pouw_ledger.rs:102-220` — `Ledger` struct +
`append()` + `iter()` + `broadcast_to_peers()`. The natural insertion
point for canonical-sort is a NEW method on `Ledger` that consumes
`iter()`, sorts in-memory, and emits to a destination ledger via
`replay_list()` (which already exists at `pouw_ledger.rs:293-298`).

### 4. SpinorReceipt struct: ACTUAL field layout

`tools/sp_daemon/src/dialogue.rs:39-63` — the EXACT field-by-field
layout, cross-checked against the silicon-confirmed hexdump in
`tools/sp_compute_skel/docs/CLOSURE-M2-DIALOGUE.md:100-111`:

| Offset | Size | Field |
|---|---|---|
| 0     | 1  | `turn_index: u8` |
| 1     | 1  | `model_id: u8` |
| 2-3   | 2  | `_pad: [u8; 2]` |
| 4-7   | 4  | `wall_us: u32` |
| 8-31  | 24 | `input_hash: [u8; 24]` |
| 32-55 | 24 | `output_hash: [u8; 24]` |
| **56-59** | **4** | **`n_input_tokens: u32`** |
| 60    | 1  | `n_output_tokens: u8` |
| **61-62** | **2** | **`_reserved: [u8; 2]`** |
| 63    | 1  | `sentinel: u8` (0xA5) |

This **DIVERGES** from the memory note (which says `_reserved` is 9
bytes spanning offsets 54-62). The dispatch prompt's premise that
`_reserved[2..4]` corresponds to "offsets 56-57" is also out-of-date.

**Reality:**
- `_reserved` is the LITERAL Rust field `_reserved: [u8; 2]` at offsets
  61-62 (TWO bytes only).
- Offsets 54-55 are the tail two bytes of `output_hash`.
- Offsets 56-59 are occupied by `n_input_tokens: u32`.
- Offset 60 is occupied by `n_output_tokens: u8`.

Therefore the u16 sequence rank (2 bytes) fits EXACTLY in
`_reserved[0..2]` at offsets **61-62**. Device_id (4 bytes) does
**NOT** fit on-wire under the current 64-byte SpinorReceipt layout.

### 5. `reference-heterogeneous-soc-crt-tricks` Trick #10

Trick #10 (Receipt-backed verifiable distributed compute). M.4 +
mesh-canonical-order together complete the canonical claim that
"same receipt set on different devices → byte-identical canonical
ledger." This sprint is the canonical-ordering primitive.

### 6. `feedback-no-silent-gate-revisions`

The discipline rule. The Stage 0 finding in §4 above is a SPEC
correction surfaced UPSTREAM in this plan, not a silent revision: the
dispatch prompt assumed a 9-byte `_reserved` field that does not exist
in the current struct. The implementation proceeds with the
constraint-respecting Option A choice (see §"Option A vs B decision").

## Option A vs B decision — **Option A (forced by struct layout)**

The dispatch prompt offered Option A (rank-only in receipt; device_id
carried via input_hash) or Option B (rank + device_id both in receipt's
`_reserved` field, claimed to be 4+2 bytes available).

**The actual struct has only 2 bytes of `_reserved`.** Option B as
written in the dispatch prompt is structurally impossible without
either:
- (b1) widening the SpinorReceipt to 128 bytes (SpinorReceiptV2), OR
- (b2) repurposing bytes from `n_input_tokens` (would break audit hash
  semantics — that field is part of the audit chain), OR
- (b3) repurposing bytes from `output_hash[20..24]` (4 trailing bytes
  of a 24-byte truncated SHA-256 — collision resistance drops from
  192-bit to 160-bit, still strong but a real-information loss).

All three break either ABI compatibility with M.2's silicon-confirmed
layout or audit guarantees. Per `feedback-no-silent-gate-revisions`:
DO NOT silently widen the receipt to 128 bytes or repurpose audit
bytes; surface the constraint and pick Option A.

**Option A — adopted:** rank-only in `_reserved[0..2]` at on-wire
offsets 61-62. Device_id is carried via the `input_hash` field's
existing domain-separation by `(model_id, turn_index)` — for cross-
device sorting at the canonical-sort layer, the rank must be
**globally monotonic** (a shared logical clock or a (device_id, rank)
pair carried EXTERNALLY when broadcast). The on-disk receipt holds
only the rank.

**Canonical sort key** for v1 = `(rank, input_hash)`. The
`input_hash` truncated SHA-256 is dense (≥192 bits of entropy) and
domain-separated; two different devices producing different content
under the same rank tiebreak deterministically by hash compare. This
gives Option-A self-contained ordering at the cost that two devices
producing the IDENTICAL receipt content under the same rank are
indistinguishable (which is the correct semantics — they ARE
identical receipts and the canonical sort dedupes-by-equality).

**Why this is acceptable for v1:**
- The canonical-sort gate (T_MESH_CROSS_DEVICE_BYTE_IDENTITY) verifies
  ordering deterministic, not device disambiguation in the receipt
  bytes.
- Manifesto Trick #10 is "receipt-backed verifiable distributed
  compute" — the verification target is byte-identity of the ledger
  after canonical sort, which `(rank, input_hash)` achieves.
- A future SpinorReceiptV2 sprint can add device_id without breaking
  v1 (v2 bumps `model_id` MSB per the existing reserve plan, and
  callers fall back to (rank, input_hash) compare on v1 receipts).

**Documented forward path:** if M.6 cross-island deployment needs
explicit device_id in the receipt bytes, the schema upgrade is
SpinorReceiptV2 (128 bytes; model_id MSB=1 signals v2). v2 then adds
`device_id: u32` at fresh offsets. This sprint's canonical-sort code
remains correct: it sorts by `(rank, input_hash)` and any v2-aware
caller can sort by `(rank, device_id, input_hash)` instead.

## Architecture — concrete

### SpinorReceipt extension (`dialogue.rs`)

Add three methods. NO changes to the 64-byte struct layout.

```rust
impl SpinorReceipt {
    /// Returns a NEW receipt with the same payload but with _reserved[0..2]
    /// populated as the little-endian u16 sequence rank. ALL other fields
    /// (hashes, wall_us, token counts, sentinel) are preserved bit-exact.
    ///
    /// The on-wire bytes that change are at file offsets 61-62 ONLY.
    /// The 0xA5 sentinel at offset 63 is preserved.
    pub fn with_sequence_rank(mut self, rank: u16) -> SpinorReceipt {
        self._reserved = rank.to_le_bytes();
        self
    }

    /// Read the u16 sequence rank from `_reserved[0..2]` (file offsets 61-62).
    pub fn sequence_rank(&self) -> u16 {
        // Field access through #[repr(C, packed)] needs a local copy.
        let r = self._reserved;
        u16::from_le_bytes(r)
    }
}
```

Note: the dispatch prompt's helper signature was `with_sequence_rank
(self, rank: u16, _device_id: u32)` taking a device_id parameter for
forward compat. Per the Option A decision above, device_id is NOT
stored on-wire in v1 — we drop the parameter. A future v2 helper can
add it back without breaking v1 callers (the v1 signature stays
single-arg).

### Ledger canonical-sort (`pouw_ledger.rs`)

Add two methods on `Ledger`:

```rust
impl Ledger {
    /// Re-orders the in-memory receipt list by canonical sort key
    /// (rank ASC, input_hash ASC lexicographic tiebreak). The on-disk
    /// file is NOT mutated; the sorted list is returned via the
    /// `replay_canonical_into` companion method which writes the sorted
    /// sequence into a fresh ledger.
    pub fn canonical_sort(&self) -> LedgerResult<Vec<SpinorReceipt>> {
        let mut recs: Vec<SpinorReceipt> = self.iter()?
            .collect::<LedgerResult<Vec<_>>>()?;
        recs.sort_by(|a, b| {
            a.sequence_rank()
                .cmp(&b.sequence_rank())
                .then_with(|| a.input_hash.cmp(&b.input_hash))
        });
        Ok(recs)
    }

    /// Reads source ledger; sorts by (rank, input_hash); writes to dest
    /// in canonical order. Returns the count of receipts copied. This is
    /// the cross-device byte-identity primitive — two ledgers with the
    /// same receipt SET produce SHA-256-identical destination files via
    /// this method.
    pub fn replay_canonical_into(&self, dest: &mut Ledger) -> LedgerResult<usize> {
        let sorted = self.canonical_sort()?;
        for r in &sorted {
            dest.append(r)?;
        }
        Ok(sorted.len())
    }
}
```

`canonical_sort` returns `Vec<SpinorReceipt>` so callers can inspect
the order; `replay_canonical_into` is the production primitive that
writes to disk.

### Smoke harness (`sp_memo_m4_canonical_replay_smoke.rs`)

Two-device scenario per scope item 4:
- Device A mints 5 receipts at ranks `0, 2, 4, 6, 8`.
- Device B mints 5 receipts at ranks `1, 3, 5, 7, 9`.
- Both broadcast (via `Ledger::broadcast_to_peers(0)` stub).
- Device A receives B's broadcast, appends to its raw ledger.
- Device B receives A's broadcast, appends to its raw ledger.
- Each device runs `replay_canonical_into()` → canonical-sorted ledger.
- Reference: build canonical-order ledger directly from minted 0..9.
- ASSERT: SHA-256(device_a_canonical) == SHA-256(device_b_canonical)
         == SHA-256(reference_canonical).
- ASSERT: canonical order is `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`.

## Gates (no silent revisions — surface UPSTREAM on any fail)

### T_MESH_RANK_PROTOCOL

**Method:** Create base receipt via `SpinorReceipt::mint(1,
MODEL_ID_EXECUTIVE, &[42i32], &[7i32], 1234)`. Call
`with_sequence_rank(42)`. Inspect `as_bytes()`:
- bytes[61..63] == `[0x2A, 0x00]` (42 little-endian)
- bytes[0..61] == base.as_bytes()[0..61] (all other fields preserved)
- bytes[63] == 0xA5 (sentinel preserved)
- `sequence_rank()` round-trips to 42.

**Pass:** `set_get_match && other_bytes_unchanged && sentinel_preserved`.

### T_MESH_CANONICAL_SORT_DETERMINISTIC

**Method:** Pre-fill ledger A with 100 synthetic receipts at randomized
ranks (seeded RNG for reproducibility). Run `canonical_sort()` twice
on fresh `Ledger::open` calls. SHA-256 the concatenated `as_bytes()`
of each sorted output. Assert SHA-256 equality.

**Pass:** `sha256_match`.

### T_MESH_CROSS_DEVICE_BYTE_IDENTITY (load-bearing)

**Method:** as in smoke harness above.

**Pass:** `device_a_sha == device_b_sha == reference_sha && canonical_order_is_interleaved`.

## Out of scope (NOT bundled)

Per dispatch prompt scope:
- Real QUIC mesh broadcast (M.4 ships stub; this sprint also leaves it stub)
- Ledger pruning / garbage collection
- Encryption-at-rest for ledger
- Per-receipt signature (M.4 v1 ships unsigned; signed is future)
- Modifying the existing SpinorReceipt 64-byte LAYOUT outside of
  `_reserved` (no SpinorReceiptV2 — Option B's 128-byte upgrade
  defer)

## Stage plan (one-variable-per-commit per `feedback-bundled-
changeset-root-cause-ambiguity`)

- **Plan-commit** (this doc): `[plan] mesh-canonical-order — u16 sequence rank + canonical sort (Option A; struct-layout correction)`.
- **Stage 1:** `SpinorReceipt::with_sequence_rank` + `sequence_rank` + unit tests on host (dialogue.rs changes + tests). One commit.
- **Stage 2:** `Ledger::canonical_sort` + `replay_canonical_into` + unit tests on host (pouw_ledger.rs changes + tests). One commit.
- **Stage 3:** Smoke harness + 3 gates execution capture (sp_memo_m4_canonical_replay_smoke.rs + Cargo.toml bin block). One commit.
- **Stage 4:** Closure doc + memory-entry-candidate update for `reference-spinor-receipt-layout` (correct the 9-byte → 2-byte error). One commit.

## Coordination with concurrent `ledger-autowire` agent

Per dispatch prompt §"Coordination": ledger-autowire is editing
`pouw_ledger.rs`'s ENTRY POINTS (`Ledger::append` callsite from
chat handler). This sprint adds NEW methods to the same file. To
minimize merge conflict surface:

- All new methods go AT THE END of the `impl Ledger { ... }` block
  in `pouw_ledger.rs`, BELOW the existing `broadcast_to_peers` method.
- All new tests go AT THE END of the existing `mod tests { ... }`
  block in `pouw_ledger.rs`.
- No changes to AppState (matches M.4's "AppState additions: NONE"
  discipline). Per dispatch prompt: prefix any required AppState
  additions with `// mesh-canonical-order:` — we add NONE.
- `lib.rs` is NOT touched (pouw_ledger module already declared).
- `Cargo.toml` adds ONE new `[[bin]]` block at the END of the
  existing bin list, with `# §4-mesh-canonical-order` comment
  prefix to make merge-time disambiguation trivial.

## Worktree status

- Worktree: `D:\F\shannon-prime-repos\engine-mesh-order` exclusively.
- Branch: `sprint/mesh-canonical-order` (base: `52e2145` engine main
  post-Chat-integration merge).
- Concurrent agent `ledger-autowire`: in `engine-ledger-autowire`
  worktree — NOT TOUCHED.
- All other engine-* / lattice-* worktrees: NOT TOUCHED.
- Model artifacts in `D:\F\shannon-prime-repos\models\`: NOT TOUCHED
  (host-only sprint; no L1 forward).

## Hardware

Host-only. Pure data-layer + sort algorithm. No S22U needed.
Future sprint can cross-compile to aarch64-android and run on S22U
to confirm Option A holds across architectures, but that's
`feedback-bundled-changeset-root-cause-ambiguity` discipline:
host-first, on-device-later.
