# A1/A2 mem-lifecycle — gate receipts (2026-07-08)

Flag-gated, null-floored non-destructive supersede for the memory lifecycle.
Files changed (ONLY): `tools/sp_daemon/src/recall.rs`, `tools/sp_daemon/src/routes.rs`.
Base: git HEAD cfb8459. Not committed.

## G-MEM-LIFECYCLE-unit  = GREEN
`cargo test --lib --release --features wire_cuda_backend supersede_marks_not_deletes`
=> `test recall::a1_tests::supersede_marks_not_deletes ... ok` ; `1 passed; 0 failed`.
Proves `mark_superseded_registry` MARKS (lifecycle=1), KEEPS the victim row, leaves
the other row untouched, drops 0 rows. Full log: `G-MEM-LIFECYCLE-unit.log`.

## Rust compile = GREEN (both files)
`cargo build --release --features wire_cuda_backend --bin sp-daemon` compiled ALL Rust
(recall.rs + routes.rs) with 0 errors (16 pre-existing warnings), reaching the final link.

## Daemon exe (target-wirecuda\release\sp-daemon.exe) relink = BLOCKED (pre-existing, NOT this change)
The non-perf link fails with:
`sp_cuda_daemon_backend.lib(cuda_forward.cu.obj): LNK2019 unresolved external gemma4_tail_cpu`
Root cause: uncommitted ADR-012 WIP already in the working tree —
`M src/backends/cuda/cuda_forward.cu` (adds calls to gemma4_tail_cpu; absent at HEAD) +
`m lib/shannon-prime-system`. gemma4_tail_cpu is DEFINED only in
`build-cpu-perf/.../sp_forward.lib`, NOT in the non-perf `build-cpu` math-core that
`_e2e_build.bat` / build.rs default links against. Independent of the two Rust files.
The unit gate above was therefore linked against build-cpu-perf (resolves the symbol);
the pure-Rust test is agnostic to perf-vs-nonperf math-core.
Operator action to produce the non-perf exe: rebuild build-cpu math-core with the ADR-012
tail (or rebuild the CUDA backend lib from committed cuda_forward.cu). Out of A1/A2 scope.

## Null-floor argument (SP_MEM_LIFECYCLE unset, SP_RECALL_AUDIT unset)
- FORGET/DECIDE/MERGE: `sp_life=false` => in-memory takes `else { ns.retain(...) }` and
  registry takes `else { <original drop-rewrite block> }` — both byte-identical to HEAD.
- A2 filter `if ep.lifecycle != 0 && SP_RECALL_AUDIT != "1" { continue; }`: for any registry
  with nothing superseded (all lifecycle==0), the guard is always false => no-op.
- Episode literals default `lifecycle: 0` (active); load_registry maps a missing "lifecycle"
  field to 0 => all existing registries behave exactly as before.
Diff evidence: `A1A2.diff`.
