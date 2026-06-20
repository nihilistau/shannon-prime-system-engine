---
type: gate-audit
title: G-INT-2 PHASE A — the seam audit (FFI/lifecycle map + Stage-1/2 insertion plan)
description: A read-mostly map of the live two-stage causal-recall seams in the sp_daemon CUDA decode path — the SP_B3_DISPOSER mode-2 teacher-forced ablation FFI sequence, the rewind-pristine guarantee and its hazards, the v1_chat turn-lifecycle insertion point, the bounded-N C2-Hamming Stage-1 cull plan over recall.rs primitives, the live-secret/payload resolution, and the pre-registered Phase-A telemetry — so Phases B/C can wire the live Stage-2 gatekeeper.
tags: [integration, ppt-arm-lat, g-int-2, ablation-oracle, bounded-n, ffi-seam, rewind-pristine, disposer, c2-hamming]
timestamp: 2026-06-20
resource: tests/fixtures/chat_fullstack/G-INT-2-PHASE-A-AUDIT.md
sp_status: AUDIT (Phase A — map only; no Stage-B/C logic implemented)
sp_gate: G-INT-2 (≥80% novel-recall + 100% foreign-reject; default-off = null floor) — Phase A enables it
sp_commit: (this doc)
sp_repro: §F per-telemetry commands; the map cites file:line
---

# G-INT-2 PHASE A — THE SEAM AUDIT

**Scope.** READ-ONLY map. No source modified. Contract: `lattice papers/CONTRACT-PPT-ARM-LAT-INTEGRATION.md` (ab8884e), §2 architecture + §4 G-INT-2.
All citations are against the live tree (NOT the stale `tools/sp_daemon/src/b3v2_patches/` copies, which are out of date — the live `kv::` module is `tools/sp_daemon/src/cuda_kvdecode_dispatch.rs`, imported as `use sp_daemon::cuda_kvdecode_dispatch as kv;` at `routes.rs:605`).

The thesis G-INT-2 must wire (contract §2): `query → STAGE-1 C2-sig Hamming cull over N≤W → ≤K survivors → STAGE-2 teacher-forced causal ablation over survivors → ΔLL<τ=−8 ⇒ ACCEPT(inject) else (E+1) NULL`. The machinery for Stage-2 ALREADY EXISTS as the offline labeler `SP_B3_DISPOSER=2`; Phase A maps how to promote it to the LIVE gatekeeper.

---

## A — THE PER-CANDIDATE ABLATION FFI CALL-SEQUENCE (Stage-2, the durable core)

The offline labeler lives in `routes.rs` `disp_mode == 2` (`routes.rs:858-964`). It already performs the EXACT speculative inject→teacher-force→ablate→score→rewind loop G-INT-2 needs. Per candidate episode `ep` in the registry (`routes.rs:892`):

```
anchor = kv::position(handle)                                  routes.rs:889   // dpos before the probe (the episode base)
─ Leg 1 (reference, with episode present) ──────────────────────────────────
kv::replay(handle, &ep.dir, ep.npos, false)                   routes.rs:907   // inject ep K/V at [dpos, dpos+npos); dpos += npos
for &s in &secret_ids:                                         routes.rs:917   //   teacher-force the KNOWN secret tokens
    kv::decode_step(handle, tok, logits)                       routes.rs:918   //   one step per secret token; tok starts = last[0]
    lpe.push(logits[s] - lse(logits)); tok = s                 routes.rs:919   //   record log p_ref(secret_i)
kv::rewind(handle, ng)                                         routes.rs:923   // undo the ng teacher-forced payload tokens, KEEP E
─ Leg 2 (ablated, episode source rows knocked out) ─────────────────────────
targets = {episode positions whose ep.tok matches a payload-window token}     routes.rs:926-934 (greedy path); for the teacher-forced secret path the targets are still derived from gen[] vs ep.tok
kv::ablate(handle, anchor, &targets)                          routes.rs:936   // cudaMemset-zero K/V rows at [anchor+targets[i]]
for i2 in 0..ng:                                              routes.rs:939   //   re-score the SAME payload, episode rows zeroed
    kv::decode_step(handle, tok, logits)                       routes.rs:940
    lpa.push(logits[gen[i2]] - lse(logits)); tok = gen[i2]     routes.rs:941   //   record log p_abl(secret_i)
kv::rewind(handle, lpa.len() + ep.npos)                       routes.rs:943   // clear payload (ng) + episode (npos): RESTORES ablated rows + shears E
─ Score ────────────────────────────────────────────────────────────────────
collapse = Σ_{j=w0..n} (lpa[j] - lpe[j])                      routes.rs:944-946  // ΔLL; load-bearing ⇒ catastrophic negative
```

After the loop, the best (most-negative) candidate fires iff `collapse < abl_tau` (`routes.rs:952-963`): a single final `kv::replay(handle,&ep.dir,ep.npos,false)` re-injects it for the REAL decode. `abl_tau` defaults to `NEG_INFINITY` (telemetry-first: first run never fires) — G-INT-2 pins it to `−8.0` (contract §4; the v13 matrix margin, MEMORY).

### Exact Rust signatures of every `kv::*` FFI used (from `cuda_kvdecode_dispatch.rs`)

| verb | signature | line | role in the per-candidate probe |
|---|---|---|---|
| `position` | `unsafe fn position(handle: *const c_void) -> i32` | 447 | read `anchor`/`dpos`; returns -1 on NULL |
| `replay` | `unsafe fn replay(handle: *mut c_void, epdir: &str, npos: i32, zero: bool) -> Result<(),String>` | 334 | inject ep K/V at `[dpos,dpos+npos)`; `zero=true` = zeroed reject control. Journals ring slots before clobber. `dpos += npos`. M_target attenuation via `SP_REPLAY_MTARGET` (cuda_forward.cu:4627) |
| `decode_step` | `unsafe fn decode_step(handle: *mut c_void, token: i32, logits: &mut [f32]) -> Result<(),String>` | 261 | one committed decode step; `dpos += 1`; writes `n_vocab` logits |
| `ablate` | `unsafe fn ablate(handle: *mut c_void, base: i32, positions: &[i32]) -> Result<(),String>` | 355 | cudaMemset-zero K/V rows at `base+positions[i]`; EMPTY positions = no-op; TRANSIENT (restored by the next rewind that shears past them) |
| `rewind` | `unsafe fn rewind(handle: *mut c_void, n: i32) -> Result<(),String>` | 281 | O(1) shear `dpos -= n`; ring path replays the SWA undo-journal in reverse |
| `reset` | `unsafe fn reset(handle: *mut c_void) -> Result<(),String>` | 299 | zero dpos/commit_pos/jcur WITHOUT journal replay (the B2 ring-reset; per-request clean) |
| `read_global_q` | `unsafe fn read_global_q(handle: *mut c_void, token: i32, out: &mut [f32]) -> Result<i32,String>` | 434 | one NON-committing forward of `token`; writes `[n_global][G_NH*HD]` query; cache unchanged |
| `read_global_k` | `unsafe fn read_global_k(handle: *const c_void, out: &mut [f32], npos: i32) -> Result<i32,String>` | 416 | read live `[0,npos)` global-K to host for the C2 sig (read-only D2H) |
| `inject_tokens` | `unsafe fn inject_tokens(handle: *mut c_void, tokens: &[i32]) -> Result<(),String>` | 372 | B5 single-latent-entry text ingest (for LIVE/nightshift episodes that have no ep.k on disk) |

`step` is the same verb as `decode_step` (there is no separate `step`); the session wrapper `Session::decode_step` is `session.rs:176`.

---

## B — THE REWIND-PRISTINE GUARANTEE (+ HAZARDS)

**Verdict: the speculative probe leaves the live cache byte-identical on the FULL-CACHE path (the served chat's default), and on the SWA-ring path PROVIDED the journal-depth and commit invariants hold. The probe is safe to run mid-turn before the real decode.**

### The mechanism (cuda_forward.cu)
- The cache is full-cache `slot==pos` for globals always, and for SWA owners when `ring_W==0`. `gemma4_kv_rewind(delta)` (`cuda_forward.cu:4436`) on the full path is a pure `dpos_host -= delta` (line 4457) + a `dpos` H2D — the sheared `[dpos, dpos+delta)` slots are NEVER READ AGAIN (attention reads `[s0,dpos)` only) and are overwritten in position order on the next write. This is the `G-1b-REWIND-NULL` invariant (KAI-1b slot==pos inverse; comment at 4435, 4603).
- `gemma4_kv_replay` (`cuda_forward.cu:4608`) advances `dpos += npos` and (ring only) journals each clobbered slot before overwrite. The eventual `rewind(npos)` reconstructs the pre-injection window.
- **Ablation is non-destructive BY CONSTRUCTION.** `gemma4_kv_ablate_rows` (`cuda_forward.cu:4709`) zeroes K/V rows at `base+pos[i]`. Its own comment (lines 4707-4708) states: *"No journaling: gemma4_kv_replay already journaled the pre-replay slot values, so the eventual gemma4_kv_rewind(npos) restores these rows bit-exactly."* The ablated rows sit at `[anchor, anchor+npos)` = exactly the slots the trailing `rewind(ng+npos)` (`routes.rs:943`) shears off. On the FULL path they are simply abandoned (never read again); on the RING path they are restored from the replay's journal snapshot. Either way the ablation never survives the probe.

### Ordering inside ONE candidate (the two rewinds)
- `rewind(ng)` (line 923) undoes ONLY the teacher-forced payload (Leg-1), keeping E injected, so Leg-2 ablates the still-present episode rows. ✔
- `rewind(lpa.len()+ep.npos)` (line 943) undoes Leg-2 payload + the episode → `dpos` returns to `anchor`. ✔ Net `dpos` delta per candidate = 0. The next candidate's `replay` re-injects from the same `anchor`.

### HAZARDS (must be respected by the Phase-B wiring)
1. **SWA-ring journal depth `Jmax` (default 64; `cuda_forward.cu:4293`).** On the ring path, `replay` REQUIRES `(dpos − commit_pos) + npos ≤ Jmax` (`cuda_forward.cu:4641`), and `rewind` REQUIRES `delta ≤ dpos − commit_pos`, else *"delta crosses a commit (journal cleared)"* (`cuda_forward.cu:4442`). With G-INT-2's bounded `W` and short novel-fact `npos` this holds, BUT each candidate's `replay(npos) + teacher-force(ng)` consumes `npos+ng` uncommitted ticks; **K survivors probed back-to-back without an intervening commit accumulate journal depth.** Since each candidate fully rewinds to `anchor` before the next, the uncommitted span is per-candidate (`npos+ng`), NOT cumulative — safe as long as a single candidate's `npos+ng ≤ Jmax`. Phase B should ASSERT `npos+ng ≤ Jmax` (raise `SP_G4_KV_JMAX` if a long episode is admitted). The served chat's recall path runs FULL-cache (`reset()` per request, no `SP_DAEMON_KVDECODE_RING_W`), so this hazard is latent unless ring mode is enabled — but G-INT-3 (KAIROS ring) turns it on, so wire the assert now.
2. **`ablate` does NOT itself journal.** It relies on a PRIOR `replay` having journaled those slots (ring) or on the shear abandoning them (full). The Phase-B sequence MUST therefore always `ablate` AFTER the matching `replay` and BEFORE the matching `rewind(...+npos)` — never ablate a row that was not just replayed. The existing disposer respects this; preserve the ordering.
3. **`read_global_q` is non-committing** (one forward, rolled back internally) — safe to call before the probe to get the query-Q. `read_global_k` is read-only D2H. Neither perturbs the cache.
4. **Per-candidate `decode_step` COMMITS** (`dpos += 1`, bumps `KVDECODE_STEP_COUNT` at line 271). The telemetry step-count will be inflated by the probe; that is cosmetic (the gate criterion is "verb reached"), but a Phase-A telemetry note should record probe-step count separately so the served-turn decode count stays interpretable.
5. **Mutex.** The whole probe runs under the resident-cache Mutex (`guard`/`handle` acquired at `routes.rs:619-627`), held for the entire turn — no concurrent decode can race the speculative inject/ablate. ✔ This is already true at the insertion point.

**Recommended Phase-A confirmation (OPTIONAL, low-risk):** before/after one full `SP_B3_DISPOSER=2` sweep on the served registry, call `gemma4_kv_snapshot` (`cuda_forward.cu:4728`) into host buffers and `memcmp` the global-layer slabs `[0,anchor)` pre vs post — expect byte-identical (the rewind-pristine proof). This needs a tiny instrumented hook (NOT a source change to the recall logic); it is sufficient to log the `memcmp==0` result. See §F.

---

## C — THE v1_chat TURN LIFECYCLE + INSERTION PLAN

The decode entry is the `kvdecode_chat`-style fn beginning at `routes.rs:~595` (signature ends `routes.rs:604`; `use ... as kv;` at 605). Lifecycle, in order:

| phase | what | line region |
|---|---|---|
| acquire | lock resident-cache Mutex → `handle` | 619-627 |
| B1 byte-exact | `ByteExactGuard` RAII (set on, reset-to-float on every exit) | 634-650 |
| reset | `kv::reset(handle)` to dpos=0 (B2 ring-fix, NOT rewind(pos)) | 662-669 |
| prefill | `head = tokens[..n-1]` via `inject_tokens` (single_entry) or `prefill` | 686-698 |
| inject_frames | optional audio/memory residual frames (B5) | 703-719 |
| **RECALL SCORING** | **`auto_recall && replay_dir.is_none()`** block | **740-1072** |
|  · read query-Q | `qbuf = read_global_q(handle, last[0])` (npos_q = dpos = head.len) | 742-757 |
|  · QDUMP | optional Q dump for offline training | 763-772 |
|  · q·K score | `qk_relevance` over every episode (the Stage-1 ranker today) | 774-784 |
|  · **SP_B3_WC** | learned head (E+1)-NULL argmax → replay/inject winner | 791-840 |
|  · **SP_B3_DISPOSER==2** | **the offline ablation labeler (Stage-2 core)** | 858-964 |
|  · SP_B3_DISPOSER==1 | multi-tok ΔLL disposer (refuted) | 965-1045 |
|  · else q·K fire | raw top-m fire if ≥ tau_qk | 1046-1062 |
| recall SSE | emit `recall` event (name or null) | 1076-1082 |
| B2 explicit replay | `replay_dir` operator-named episode | 1089-1098 |
| **REAL decode tail** | `kv::decode_step(handle, last[0], logits)` → first token | 1104-1108 |
| decode loop | `for _ in 0..max_tokens { sample; decode_step }` | 1119-1166 |
| flush + DONE | flush dec_buf, emit `[DONE]` | 1168-1183 |
| **B4 NIGHTSHIFT capture** | `SP_B4_NIGHTSHIFT==1`: capture raw user turn as live episode | 1185-1293 |
| release | `sessions.remove(chat_id)` (Mutex drops) | 1295 |

### Insertion point for the two-stage recall
- **Stage-1 + Stage-2 go INSIDE the existing recall block (740-1072), AFTER the query-Q read (757) and BEFORE the recall-SSE emit (1076).** This is the same window where `SP_B3_WC` (791) and `SP_B3_DISPOSER` (858) already live, so the data (`qbuf`, `registry`, `app.nightshift`, `anchor=dpos`) is in scope and the rewind-pristine property already holds (Mutex held; cache at `dpos==head.len()`).
- **CRITICAL ORDERING — the speculative ablation MUST fully rewind BEFORE the real decode tail at 1104.** The probe leaves `dpos == anchor == head.len()` (per §B, net-zero). The single SURVIVING accept then does ONE `kv::replay(winner)` (advancing `dpos` to `head.len()+npos`) just like the existing disposer fire at 957-963, so the real `decode_step(last[0])` at 1104 attends over the injected memory. A NULL verdict does NO replay → `decode_step` runs on the clean prompt cache → byte-identical to a no-recall turn (the foreign-reject null floor). Both branches reach 1104 with the cache in a defined, pristine state.
- **Gating env (default-off = null floor):** introduce one new flag (e.g. `SP_INT2=1`) that, when set, runs `[Stage-1 cull → Stage-2 ablation]` INSTEAD of the `SP_B3_WC`/`SP_B3_DISPOSER`/q·K fires. Unset = today's behavior. The `SP_B3_WC` head, if present, is invoked ONLY as the Stage-1 tie-breaker pre-sort when the Hamming cull pool exceeds the budget (contract §2 doctrine — W_c demoted), never as the gate.

---

## D — STAGE-1: BOUNDED-N + C2-HAMMING CULL (existing recall.rs primitives)

All primitives exist in `tools/sp_daemon/src/recall.rs`; no new math.

- **Candidate set, bounded to N≤W.** The live candidates are `registry.iter().chain(app.nightshift.read().iter())` — exactly the set the `SP_B3_WC` branch already assembles (`routes.rs:798-801`). KAIROS bounding to `W` (G-INT-3) is a `.take(W)` / recency-window over `app.nightshift` (the live `Vec`); for G-INT-2 Phase B, bound by taking the most-recent `W` live episodes plus the curated registry.
- **Query signature.** Compute the live 256-bit C2 sig from the prompt's global-K: `read_global_k(handle, &mut buf, npos_q)` (`dispatch:416`) → `recall::Projection::build().signature(&buf, n_global, npos_q)` (`recall.rs:58`). `n_global = NL/PERIOD = 48/6 = 8` (`recall.rs:24-25,755`). NOTE: this is the centroid-Hamming sig (sign of the ±1 projection mean over (layer,pos)); the same sig the registry was built with (`recall.rs:244-251`, `load_registry`). The registry episode sigs are already loaded into `Episode.sig` (`recall.rs:95`) from `sig_bits` hex (`parse_sig_hex`, `recall.rs:195`). Live nightshift episodes currently carry `sig:[0u64;4]` (`routes.rs:1272`) — Phase B must compute+store their C2 sig at capture time (the G-INT-1 wire: sig-at-ingest from full-precision K).
- **Rank by Hamming → top-K.** `recall::agree(&query_sig, &ep.sig)` (`recall.rs:184`) returns `R_BITS − hamming` (agreement, higher=closer). `best_match` (`recall.rs:296`) gives the single argmax; for top-K, sort the candidates by `agree` descending and take the first K (K ≤ W, the KAIROS latency budget). The W_c head (`wc_score`, `recall.rs:376`) is the OPTIONAL tie-breaker ONLY if the cull pool > budget (contract §2).
- **Output of Stage-1:** ≤K survivor episode indices, handed to Stage-2.

Honest caveat (carried from the campaign, MEMORY): the centroid C2 sig did NOT separate question→passage well in isolation (right episode argmax ~1/5; mutual agree ~200/256) — but here it is a CULL not a verdict (the contract's "the Stage-1 cull leaks, so end-to-end recall is capped ≥80%"), and Stage-2 ablation is the 100%-foreign-reject gate. A leaky cull only costs recall, never a false accept.

---

## E — THE LIVE-SECRET / PAYLOAD RESOLUTION

The offline disposer teacher-forces a KNOWN `ep.secret`; for LIVE Stage-2 each candidate must supply ITS OWN teacher-force target + source rows. The plumbing already resolves this per-episode (`routes.rs:888-905`):

- **What gets teacher-forced (the "secret"/salient span):** for each candidate, the resolution order is (`routes.rs:899-904`): env `SP_B3_SECRET` single-shot override → else `<ep.dir>/ep.secret` sidecar (the salient answer SUBSTRING, BOS-stripped, only trailing CR/LF trimmed) → else EMPTY ⇒ fall back to the greedy `[2,8)` window (no crash, null floor). For a LIVE candidate the `ep.secret` sidecar IS the stored memory's answer span. The curated registry already ships `ep.secret` (written by the admission path / `patch_npos.py`, MEMORY); G-INT-2's corpus is the 90-needle div registry, all carrying `ep.secret`.
- **Which source rows get ablated:** the episode positions whose `ep.tok` token matches a teacher-forced/payload token (`routes.rs:926-934`); `ep.tok` = the EXACT input token file (alignment guaranteed by construction — the v11 ghost fix, MEMORY). Capped to 12 targets (`routes.rs:933`). Ablated at `[anchor + target]`.
- **Sidecar availability:** curated registry episodes have `ep.k/ep.v/ep.mf/ep.tok/ep.secret` on disk (the `_b3_capture_ep` curator + admission). LIVE nightshift episodes (`routes.rs:1238-1287`) write `ep.k/ep.v/ep.mf` via `kv::capture_batched` but currently NO `ep.tok`/`ep.secret`. **Phase-B gap:** for a live-captured episode to be Stage-2-gateable, capture must also (a) write `ep.tok` (it has `toks` in hand at `routes.rs:1211`), and (b) derive an `ep.secret` salient span (the user-turn answer span — or, for a stated-fact memory, the fact tail). Until then, live episodes fall back to the greedy `[2,8)` window, which the campaign proved WEAK for non-recited secrets (v11, MEMORY). **Recommendation:** scope G-INT-2 Phase B to the CURATED div registry (full sidecars present) first; defer live-episode Stage-2 to the B4-v2 / G-INT-3 nightshift work (write ep.tok+ep.secret at capture).
- **DISTINCTION from the offline labeler.** Offline: ONE answer key per episode, self-ablation (each ep tests itself). LIVE: the candidate's `ep.secret` is the teacher-force target, but the QUERY is the live user turn — Stage-2 measures "does deleting THIS candidate's source rows collapse the candidate's own stated fact under the live query context". That is exactly the `disp_mode==2` loop with the live `last[0]` as the decode seed (already `routes.rs:910,938`). No new ablation math; the only change is iterating the K survivors (not the whole registry) and pinning `abl_tau=−8`.

---

## F — PRE-REGISTERED PHASE-A TELEMETRY (log-first, before any fire)

Telemetry-first discipline (the project's `tau_qk` law): the first instrumented run NEVER fires (`abl_tau`/`SP_INT2` default such that no accept is committed) — we LOG, verify separation + pristineness, THEN pin τ and enable the fire. Lock these four:

1. **Per-candidate ΔLL (collapse) matrix.** Already logged by the disposer at `routes.rs:951`: `B3-DISPOSER ABLATION collapse=ΣΔLL... [ep(collapse=..,ntgt=..) ...]`. For G-INT-2 add the Stage-1 context: log `survivors=[ep:agree] cull_pool=P budget=K`. Expect: matched novel needle ≈ −9…−47; parametric/foreign ≈ [−0.7,+1.5] (v13 matrix margins, MEMORY). The 100%-foreign-reject claim is read off the foreign rows landing ≫ τ=−8.
2. **Cull → survivors trace.** Per turn: `n_candidates (curated C + live L) → top-K by Hamming → survivor names+agree`. Confirms KAIROS bound `W` and budget `K` are respected and that the matched episode SURVIVES the cull (the recall-cap leak source — if the true episode is culled, end-to-end recall drops below 80%; this trace localizes a miss to Stage-1 vs Stage-2).
3. **Post-rewind cache-identity check (the rewind-pristine receipt).** `gemma4_kv_snapshot` global-layer `[0,anchor)` memcmp pre-probe vs post-probe == 0. Log `RING_PRISTINE memcmp=0 (Δslots=0)`. This is the durable proof the speculative probe is mid-turn-safe; it is the one Phase-A measurement worth taking before Phase B.
4. **Probe step accounting.** Log `probe_steps=Σ(npos+ng) real_decode_steps=D` separately so the served-turn decode count stays interpretable and the `Jmax` headroom (`max single-candidate npos+ng` vs `SP_G4_KV_JMAX`) is visible (hazard §B.1).

### Is a small instrumented Phase-A run warranted before Phase B?
**Yes for telemetry #3 only (the rewind-pristine receipt); the rest are Phase-B byproducts.** The existing `SP_B3_DISPOSER=2` path ALREADY exercises the full inject→ablate→rewind sequence on the served registry with τ=−∞ (no fire). Running it once with a tiny snapshot-memcmp instrument confirms the byte-identical-after-probe property (the load-bearing safety claim for running Stage-2 mid-turn) WITHOUT any Stage-B/C logic and WITHOUT a long run (one chat POST, one registry sweep). The collapse matrix (#1) on the div registry is already GREEN offline (G-CHAT-B3-ADMISSION-200 / v13-MATRIX, MEMORY), so it need not be re-run for Phase A. Recommendation: take the §F.3 snapshot-memcmp receipt opportunistically; do NOT block Phase A on it — the map is the deliverable.

---

## SUMMARY VERDICT

- **The Stage-2 gatekeeper is ALREADY BUILT** as `SP_B3_DISPOSER=2` (`routes.rs:858-964`). Promotion = iterate the K cull-survivors instead of the whole registry, pin `abl_tau=−8`, and gate behind a new `SP_INT2` flag.
- **Rewind-pristine: CONFIRMED** on the full-cache path (served default) by construction (shear abandons sheared slots); CONFIRMED on the ring path subject to `npos+ng ≤ Jmax` and the commit invariant (assert in Phase B). Ablation is non-destructive (restored by the trailing rewind). The probe is net-zero `dpos` per candidate and safe to run mid-turn under the held Mutex.
- **Insertion point:** inside the recall block (`routes.rs:740-1072`), after `read_global_q` (757), before the real `decode_step` tail (1104) — same window as the existing W_c/disposer branches; the accept does ONE `replay`, NULL does nothing.
- **Stage-1 cull:** `read_global_k` → `Projection::signature` → `agree` sort → top-K over `registry ∪ nightshift` bounded to W; W_c (`wc_score`) demoted to over-full-pool tie-breaker only.
- **Live secret/payload:** per-episode `ep.secret` (teacher-force target) + `ep.tok` (ablation source rows) — present on the curated div registry; live nightshift episodes need ep.tok+ep.secret written at capture (Phase-B/G-INT-3 gap; scope Phase B to curated registry first).
- **Telemetry to lock:** per-candidate ΔLL matrix, cull→survivors trace, post-rewind snapshot-memcmp==0, probe-step/Jmax accounting. One optional `SP_B3_DISPOSER=2` snapshot-memcmp run confirms rewind-pristineness; not a Phase-A blocker.
