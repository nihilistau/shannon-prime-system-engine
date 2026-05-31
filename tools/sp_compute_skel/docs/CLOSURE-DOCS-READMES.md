# CLOSURE — DOCS-READMES (three-repo README + API documentation sprint)

**Sprint:** DOCS-READMES
**Date:** 2026-05-31
**Branches:**
- `shannon-prime-lattice` → `docs/readmes-update`
- `shannon-prime-system` → `docs/readmes-update`
- `shannon-prime-system-engine` → `docs/readmes-update`
**Sub-tag candidates:** `docs-readmes-v1` (each repo)
**Status:** **3/3 READMEs shipped**, closure + status table land in this note

The user asked for production-quality READMEs for the three Shannon-Prime
GitHub repositories, with current status, getting-started, system
architecture, engine details, Rust-crate wiring, professional API
documentation, model conversion, HTTP API I/O, peering, "everything in
detail." This closure documents what shipped per repo, the honest
status snapshot that informed the writing, and the gaps in the codebase
that surfaced while documenting.

---

## 1. Per-repo deliverables

### 1.1 `shannon-prime-lattice` (umbrella + papers)

**File:** `README.md`
**LOC:** +279 / −22 (was 36 lines; now 293 lines including frontmatter)
**Commit:** `6fd698e [docs] DOCS-READMES -- comprehensive lattice README + honest status table`

Sections shipped:

1. What makes this different (distinguishing claims, each with a
   concrete shipped sprint as evidence)
2. Current status (21-row honest table covering math-core, NTT,
   poly-ring, KV cache, Frobenius, KSTE, daemon, dialogue, ledger,
   M.5 routing, two-node smoke, TailSlayer, CPU AVX-512, CUDA,
   Vulkan, Hexagon, WIRE-HEX, NTT.5d/5e, CUDA/Vulkan daemon wiring,
   Fibonacci-Prime DHT; production tok/s baseline reported)
3. Architecture in one ASCII diagram (HTML clients → sp_daemon →
   libshannonprime → 4 backends + cDSP skel via FastRPC)
4. Getting started (clone-all-three command, then four
   "you want X" sub-paths)
5. Repository layout
6. Hard rules (anti-contamination, no silent gate revisions, honest
   closure notes, one math object, worktrees per concurrent agent)
7. Where to read next
8. Contact

### 1.2 `shannon-prime-system` (math-core library)

**File:** `README.md`
**LOC:** +541 / −30 (was 53 lines; now 564 lines)
**Commit:** `b00c869 [docs] DOCS-READMES -- comprehensive math-core README + L1 ABI reference`

Sections shipped:

1. What's in here (per-header inventory table — every file under
   `include/sp/` with role)
2. Current status (24-row honest table + frozen primes/constants
   reference table)
3. Build (Tier-1/2/3/Cross matrix, whole-repo + single-module fast
   iteration, module conventions)
4. Professional API reference (sections 4.1-4.14):
   - 4.1 Lifetimes & ownership
   - 4.2 Model load / unload
   - 4.3 Session lifecycle (KV-mode flags, precision selection)
   - 4.4 Two-function forward
   - 4.5 Speculative-decoding primitives (clone, rewind, position)
   - 4.6 Status codes (full enum table)
   - 4.7 NTT-CRT primitive (frozen primes, 128-bit-free constraint)
   - 4.8 Polynomial-ring attention (direct + Bluestein + arbitrary-N gotcha)
   - 4.9 Forward dispatch (env knob table)
   - 4.10 Forward kernels
   - 4.11 Frobenius lift (per-row rationale + ceiling-shift rule)
   - 4.12 Spinor block — FROZEN
   - 4.13 KSTE encoder (incl. known i16 clamp gotcha)
   - 4.14 Packed-weight arena
5. How math-core relates to the backends (full-forward vs NTT-dispatch
   registration shapes)
6. License

Honest note included: this repo's main lags the engine submodule pin
on the §6 forward-backend hook (WIRE-HEX shipped on
`engine-wire`/`docs/readmes-update`; the bump to math-core's standalone
main happens in the next sync wave).

### 1.3 `shannon-prime-system-engine` (engine + daemon + skel + tools)

**File:** `README.md`
**LOC:** ~+1100 / −33 (was 39 lines; now ~1100 lines)
**Commit:** pending — to be added with the closure commit

Sections shipped:

1. What this repo provides (per-slot inventory table — math-core
   submodule, 4 backends, cDSP skel, sp_daemon, sp_transcode,
   sp_dsp_smoke, sp_npu_spike, sp_halide_gen, oracle, PPL)
2. Current status — honest table (built vs wired in sp_daemon for
   every component; production tok/s baseline)
3. Quick start (5 sub-sections: host build, transcode, start daemon,
   first chat, first dialogue)
4. Architecture (full ASCII: HTML clients → sp_daemon modules →
   L1 C ABI → libshannonprime modules → 4 backends + cDSP skel)
5. The backends (per-backend file table, build flags, status for CPU,
   CUDA, Vulkan, Hexagon — including the host/skel split for Hexagon
   and the WIRE-HEX skel-rebuild gap)
6. sp_daemon Rust crates (library modules table + binaries table +
   key Rust types: AppState, SpSession, SpinorReceipt, DialoguePool,
   Ledger, ShardBlockHeader)
7. HTTP / SSE / WebSocket API (endpoint summary table + per-endpoint
   detail for all 13 routes: chat, dialogue, abort, events, metrics,
   receipts, pouw/ledger, mesh/peers, node/telemetry, dsp/echo,
   dsp/model_info, debug/backend_counts, chat/stream stub)
8. Hexagon skel IDL reference (every qaic-numbered method 1-19 with
   sprint + purpose + key constraints; method-collision note from
   merge time)
9. CLI flags + environment variables (full `sp-daemon start` flag
   table + runtime knob env var table)
10. Model conversion (sp_transcode CLI, supported inputs, output
    format detail, validation flow)
11. Peering / QUIC mesh (wire format, topology, TLS placeholder,
    receipt replay, peer connection command)
12. Development workflow (build matrix, smoke tests, adding a new
    backend = 5-stage WIRE-HEX template, adding an IDL method,
    worktree discipline)
13. Known issues / pending (10-row table of issues with workarounds
    and resolution sprints)

---

## 2. Honest status snapshot used across all three READMEs

(Reproduced here so this closure stands alone as the audit-trail entry.)

### 2.1 What's shipped

- Frozen L1 C ABI (`sp_l1.h`); tag `lat-phase2-contract-frozen`
- `.sp-model` v0 wire format + mmap loader
- Math-core reference forward — Qwen3-0.6B, Qwen2.5-Coder-0.5B,
  Gemma3-1B byte-exact on host + aarch64-android
- NTT-CRT primitive (host + Hexagon V69 HVX, both byte-exact)
- Polynomial-ring attention overlay (host + Hexagon via NTT.5b/5c)
- Spinor-block KV cache (`SP_KV_SPINOR=1`)
- Frobenius-lift Q8 / Q4 packing
- KSTE encoder + Tier-0/Tier-1 dominance
- `sp_daemon` HTTP/SSE chat (`/v1/chat`)
- Dual-model dialogue (`/v1/dialogue` MeMo Grounding → Entity ID →
  Synthesis), 3 SpinorReceipts per turn
- PoUW receipt ledger + canonical-order replay
- KSTE-routed sparse Memory activation (M.5)
- Two-node sharded inference smoke
- TailSlayer GF(2) channel oracle (offline cache pattern)
- All 4 backends built (CPU AVX-512, CUDA PTX, Vulkan, Hexagon HVX)
- sp_daemon → Hexagon backend dispatch wiring (daemon-side)

### 2.2 What's NOT shipped (honest)

- **WIRE-HEX BIT-EXACT gate** — daemon-side wired but cDSP-side
  `libsp_hex_skel.so` needs rebuild against current `sp_hex.idl`
  before bit-exact + tok/s gates can resolve.
- **CPU AVX-512 / CUDA / Vulkan daemon wiring** — backends built and
  bit-exact-validated, but not wired into `sp_daemon` (symmetric
  WIRE-HEX-style sprints pending; each ~1 day of plumbing once the
  template is in hand).
- **NTT.5e (decode-path NTT routing)** — filed, not shipped.
  `sp_decode_step` uses fp32 reference even with `SP_ENGINE_NTT_ATTN=1`.
- **NTT.5d (HD=128 direct backend path)** — filed, not shipped.
  Bluestein covers HD=64 (Qwen3, Qwen2.5-Coder); Gemma3-1B HD=256
  works via direct N=256.
- **Hexagon backend persistent-KV decode** — `gemma3_forward_hexagon`
  re-runs full forward per call; no per-backend persistent-KV API.
- **Fibonacci-Prime DHT** — spec'd in `papers/PPT-LAT-Roadmap.md` §8,
  not implemented. Mesh today is two-node CRT shard fabric.
- **QUIC mesh TLS verification** — dev-mode skip-verifier; Phase 5
  ed25519 dominance identity verification swap pending.

### 2.3 Production tok/s baseline

S22U R5CT22445JA, 16-token prefill + 32-token decode, single
`/v1/chat` call, `SP_ARENA=q8`, math-core reference forward (no
NTT-attention overlay):

| Model | Wall (s) | Tokens | tok/s |
|-------|---------:|-------:|------:|
| Gemma3-1B | 18.06 | 16 | 0.89 |
| Qwen3-0.6B | 11.21 | 16 | 1.43 |

These are the **reference** numbers. The HVX-routed numbers appear in
the table once the cDSP skel is rebuilt against the current IDL.

---

## 3. Files changed per repo

### 3.1 `shannon-prime-lattice`

| File | LOC delta |
|------|----------:|
| `README.md` | +279 / −22 |

### 3.2 `shannon-prime-system`

| File | LOC delta |
|------|----------:|
| `README.md` | +541 / −30 |

### 3.3 `shannon-prime-system-engine`

| File | LOC delta |
|------|----------:|
| `README.md` | ~+1100 / −33 |
| `tools/sp_compute_skel/docs/CLOSURE-DOCS-READMES.md` | this file (+~260) |

---

## 4. Commits per repo

### 4.1 `shannon-prime-lattice` on `docs/readmes-update`

```
6fd698e [docs] DOCS-READMES -- comprehensive lattice README + honest status table
```

### 4.2 `shannon-prime-system` on `docs/readmes-update`

```
b00c869 [docs] DOCS-READMES -- comprehensive math-core README + L1 ABI reference
```

### 4.3 `shannon-prime-system-engine` on `docs/readmes-update`

```
(pending) [docs] DOCS-READMES -- comprehensive engine README + API docs + closure
```

---

## 5. Branches pushed to origin

(Pending — operator pushes after reviewing this closure. Three branches:)

- `shannon-prime-lattice/docs/readmes-update`
- `shannon-prime-system/docs/readmes-update`
- `shannon-prime-system-engine/docs/readmes-update`

Each branched off `main` of its repo. No submodule pin changes — this
sprint is documentation-only.

---

## 6. Sub-tag candidates per repo

- `shannon-prime-lattice` → `docs-readmes-v1`
- `shannon-prime-system` → `docs-readmes-v1`
- `shannon-prime-system-engine` → `docs-readmes-v1`

---

## 7. Gaps that surfaced in the codebase while writing docs

Documentation discipline forced me to look at routes I hadn't called
and CLI flags I hadn't typed. Things that aren't bugs but should be on
someone's followup list:

1. **`/v1/chat/stream` is a permanent stub** that returns
   `{"status":"stub","stream":"sse-legacy"}`. Either remove the route,
   wire it to real legacy-SSE chat, or document the deprecation. Today
   it's noise on the `/v1` surface that confuses API discovery.
2. **`sp-daemon reload` is a no-op** — would be useful for hot-reloading
   tokenizer chat templates or rotating the PoUW signing key. Filed
   implicitly as a future op.
3. **`/v1/metrics` `ram_svm_bytes` is hard-coded to 0.** The field is
   wired through axum JSON serialization but never populated. Either
   wire `procfs::self_status().rss` or remove the field.
4. **`/v1/node/telemetry` `node_id`, `cpu_temp_c`, `svm_mem_gb`,
   `dht_peers_total` are placeholder values** (`"q3-beast-canyon"`,
   `58.5`, `2.4`, `32`). The endpoint exists but only `peers_active`
   and `pouw_frontier` are real-time accurate. Should either be
   removed or wired to real telemetry.
5. **No `--help` output is captured in CI** — the daemon's clap
   metadata is the source of truth for CLI documentation, but there's
   no test that diffs `--help` against the README's flag table. A
   one-line `assert_eq` test would prevent drift between code and docs.
6. **Skel IDL method numbering** — the IDL has gaps from sprint
   renumbering (methods 9-10 are reserved holes from earlier shuffles).
   A comment block at the top of `sp_compute.idl` listing reserved /
   renumbered methods would prevent future merge collisions.
7. **`SP_DAEMON_BACKEND` validation** — the daemon checks
   `eq_ignore_ascii_case("hex")` for the value; any other value
   silently falls through to "no backend." Should log a warning at
   startup when the env var is set to an unrecognized value.
8. **`build-android-hex-backend.bat` requires `HEXAGON_SDK_ROOT`** but
   doesn't validate the version. Hexagon SDK 5.5.6.0 is what shipped;
   newer SDKs may regress (HX.1+ closure notes hint at toolchain
   sensitivity).
9. **`SpQuicCoordinator` has no graceful shutdown** — endpoint drop
   cascades through the `run_garner_loop`; useful enough for the
   smoke but should be `Drop` impl-aware for production restart.
10. **Static file fallback (`ServeDir::new("frontend_mockups")`) shadows
    `/v1/` 404s** — if a route is misspelled it serves an HTML page
    instead of returning 404 JSON, which makes API debugging harder.
    Either constrain the static fallback to `/static/*` or return a
    structured 404 for `/v1/*` misses.

None are blocking; all are tractable in 1-2 hours each.

---

## 8. Worktree status

- `shannon-prime-lattice`: `docs/readmes-update` branch from main
  (also picks up earlier uncommitted README CRLF normalization that
  was stashed during checkout; clean working tree at commit time).
- `shannon-prime-system`: `docs/readmes-update` branch from main
  (the lat-16-3-1-bulk hedge-rework untracked files were stashed
  before checkout; clean working tree at commit time).
- `shannon-prime-system-engine`: `docs/readmes-update` branch from
  diverged main (5 ahead of origin — WIRE-HEX + earlier work).
  Clean working tree at commit time.

No git worktree fan-out was needed for this sprint — single agent,
sequential per-repo work, no parallel collision risk.

---

## 9. Operator follow-up

To merge:

```bash
# Lattice
cd shannon-prime-lattice
git push -u origin docs/readmes-update
# open PR; merge; tag docs-readmes-v1

# Math-core (standalone)
cd ../shannon-prime-system
git push -u origin docs/readmes-update
# open PR; merge; tag docs-readmes-v1

# Engine
cd ../shannon-prime-system-engine
git push -u origin docs/readmes-update
# open PR; merge; tag docs-readmes-v1
```

The engine PR should land after the math-core standalone PR if the
operator wants the math-core README to reference the bumped submodule
pin (which currently it doesn't — both READMEs are documentation-only
and cross-reference via narrative links, not git pins).

---

## 10. Honest closing note

The user has been frustrated by sprints that turned out to bypass the
production critical path. Three READMEs and one closure note do not
move tok/s by themselves. What they do is:

1. **Lower the cost of every future sprint's Stage 0** by making the
   actual surface explicit instead of requiring code archaeology.
2. **Prevent the recurring "I thought this was wired but it isn't"
   surprise** by naming the gap (CPU/CUDA/Vulkan daemon wiring,
   decode-path NTT, cDSP skel rebuild) in three places where someone
   will see it.
3. **Make the project legible to a new contributor** without losing
   the discrete-Z_q distinguishing claims to marketing-speak.

The next concrete tok/s-moving sprint is HX-SKEL-REBUILD: rebuild
`libsp_hex_skel.so` against the current `inc/sp_hex.idl` and push to
`/data/local/tmp/sp22u/`. Everything else for HVX-end-to-end is
already in place daemon-side.
