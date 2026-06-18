# B3 Two-Stage Retrieve-and-Verify — Stage 1 ABI + Stage 2 pre-registered measurement
## Gate: G-CHAT-B3-RECALL-v4   (pre-registered 2026-06-19, BEFORE any Stage-2 measurement)

## Why two stages (the v3 finding, c52f75d)
The learned contrastive `W_c` RANKING generalizes (6/6 held-out positives top the right
episode — neither the C2 centroid nor raw q·K could) but the open-world foreign-REJECT
threshold OVERFITS (held-out foreign 1.361e17 > weakest held-out positive 1.209e17).
=> Decouple: `W_c` is the **Proposer** (shortlist, NO threshold); a separate **Verifier**
is the **Disposer** (reject). Let the math sort; let a measured verifier reject.

## The trap we are explicitly NOT walking into
Stage-2 = (query C2 sig) vs (episode C2 sig) Hamming IS **B3-v1** — argmax 1/5, ~200/256
agreement on EVERYTHING. It is asymmetrical (a short question vs a dense passage). The
organism's C2 Hamming verify worked ONLY like-to-like (audio cue vs audio-derived sig,
G-XBAR-ORGANISM accept-audio/reject-text). Cross-representation query→passage Hamming is
forbidden as the disposer. [[verify Gemini/operator framing against the substrate]]

## Stage 1 — the PROPOSER (SOLVED; ready to wire). No frozen L1 ABI change.
Stage selection is L2 daemon orchestration over the EXISTING frozen L1 read/decode verbs
(append-only discipline preserved — no renumber). The "ABI surface" is the `recall` module:

```
recall::shortlist(q: &[f32],            // live query last-token global-Q (read_global_q)
                  registry: &[Episode],  // each carries ep.k global-K
                  wc: &WcInt,            // frozen int16 W_c + 2^k scale (SP_RECALL_WC)
                  k: usize)              // SP_RECALL_TOPK, default 2
   -> Vec<(usize /*ep_idx*/, i128 /*integer score*/)>   // top-k by score, NO threshold
```
- project `q` and each episode-K through `WcInt` to r dims; score `(Wc q)·(Wc K)` in the
  dual-prime CRT residue domain (reuse `sp_pr_mul`/Barrett from `tools/sp_dsp_smoke`);
  reduce per episode as recall.rs does (max + top-m mean); return the top-k.
- deterministic tie-break = lowest episode index. Default-off: `SP_RECALL_WC` unset ⇒
  today's raw-f32 path (null floor). Weight `tests/fixtures/lsh/lsh_Wc_i16_s15.bin`.

## Stage 2 — the DISPOSER (pre-registered MEASUREMENT, head-to-head; no live gating yet)
Two candidates measured against the SAME held-out set before EITHER gates a live turn.
Both are L2 orchestration over existing verbs (replay + decode/score) — no frozen ABI add.

### Candidate A — ANSWER-VERIFY (carries a known echo-chamber risk)
For each shortlisted episode E: replay E into the cache, generate the answer, compute the
answer's C2 signature, Hamming vs E's stored sig (LIKE-TO-LIKE). Accept if ≥ TAU_A.
- KNOWN RISK (state up front): B6 proved replay-conditioning OVERRIDES hallucination, so a
  foreign query + forced-E may generate E-content ⇒ answer-sig matches E ⇒ **FALSE ACCEPT**.
  The measurement decides whether this kills it.

### Candidate B — PPL-DELTA / query-likelihood (refined: NO generation ⇒ no echo chamber)
For each shortlisted episode E, teacher-force score the QUERY tokens (never generate):
```
LL0  = Σ_t log P(query_t | query_<t, ∅)      # empty cache (baseline)
LL_E = Σ_t log P(query_t | query_<t, E)      # E prefilled into the cache
ΔLL_E = LL_E − LL0                            # "did this memory make the QUERY expected?"
```
Accept the max-ΔLL shortlist candidate iff `ΔLL ≥ margin_B`; else reject (foreign).
- Relevant E raises the query's likelihood (ΔLL>0); irrelevant E flat/negative. We measure
  the model's surprise at the QUESTION, not its answer ⇒ structurally no echo chamber.
- Honest risk: the query is short (~14 tok) ⇒ ΔLL is noisy; the measurement tells us if
  margin_B is clean. Scoring reuses the M_GEMMA4 teacher-forced PPL path + replay(E).

### G-CHAT-B3-RECALL-v4 — the PRE-REGISTERED gate (measured on the real 12B, HELD-OUT)
- Split: TAU_A / margin_B are TUNED on the TRAIN split (the 17 train queries); the verdict
  is MEASURED on the HELD-OUT split (6 positives + ≥3 foreign that neither Stage-1 `W_c`
  NOR the verifier was fit on). Add more held-out foreign for statistical power if cheap.
- PASS (per candidate, on HELD-OUT): every held-out positive's RIGHT episode (present in
  its Stage-1 shortlist) is ACCEPTED, AND every held-out foreign query is REJECTED (no
  shortlist candidate accepted) — a clean accept/reject margin, NOT just argmax.
- WINNER = the candidate that rejects held-out foreign WITHOUT false-accept. Winner takes
  the `recall::verify` contract; the loser is recorded as an honest negative with its matrix.
- If BOTH fail held-out reject: honest-negative #4; autonomous recall stays parked and
  operator-`replay=` remains the working path. No goalpost moves.
- Null floor throughout: both verifiers are measurement-only; the live daemon stays on
  replay-conditioning (+ optional Stage-1 shortlist LOGGING) until a verifier passes v4.

## Execution order (after this pre-registration commits)
1. Wire Stage 1 `recall::shortlist` (W_c integer path, default-off) into routes.rs; rebuild (PMAX=4096).
2. Build the two verify harnesses: (A) replay→generate→answer-C2-sig→Hamming; (B) replay→
   teacher-force query-LL via the PPL path → ΔLL. Both reuse existing replay + decode/score.
3. Measure A vs B on the held-out split → G-CHAT-B3-RECALL-v4.log (both matrices).
4. Winner → `recall::verify`; wire shortlist→verify→fire; live content A/B (the real capability).
5. Commit + contract run-record + memory at each gate. Honest negative if neither passes.
