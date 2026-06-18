# B3 learned contrastive addresser — harness + PRE-REGISTERED gate (G-CHAT-B3-RECALL-v3)

**Status:** harness laid down 2026-06-19. Gate pre-registered BEFORE any training (no
silent goalpost moves). Builds directly on the two honest negatives B3-v1 (centroid
Hamming, argmax 1/5) and B3-v2 (raw q·K relevance, argmax 2/3 but no target/foreign
separation — receipt `tests/fixtures/chat_fullstack/G-CHAT-B3-RECALL-v2.log`).

## The thesis (why a learned metric, not a better heuristic)

Two independent geometric signatures over the stored global-K both failed the same
way: the scores are dominated by **common-mode bias** (longer episodes carry more
generic semantic mass, so a foreign query overlaps them more than a terse query
overlaps its true short episode). The space is not linearly separable in any
off-the-shelf signature. Fix = a learned projection `W_c : R^HD -> R^r` trained with a
contrastive objective so that `(W_c q)·(W_c K)` pulls (query, matching-episode) pairs
together and pushes (query, other/foreign) apart — annihilating the common mode and
isolating the relevance signal. This is the SAME deploy shape as the §3q learned-LSH
that won 8× on attention selection (`tools/xbar_lsh/train_lsh.py`): project both sides
to r dims, dot, reuse `k_qk_scores`/`k_apply_M` — **zero new hot-path kernels**.

## The exactness boundary (the load-bearing decision)

`W_c` is trained in float (PyTorch, offline, OFF the byte-exact path). It is DEPLOYED
as a frozen integer weight — handled exactly like the model's own OK_Q4B weights:

- **On disk** `lsh_Wc_i16_sK.bin`: header `{magic, ver, rows=HD, cols=r, dtype=i16,
  scale_log2=k}` then `rows*cols` int16 mantissa, row-major. Quantize ONCE,
  deterministically: `Wc_int = round(Wc_f32 * 2^k)` with a **power-of-two scale** so
  rescale is a bit-shift, never a float divide. Same `bscale` mechanism as arena v2.
- **At runtime** `q` and `K` are already fixed-point integers from the byte-exact
  islands. `(W_c q)·(W_c K)` is an all-integer matmul-then-dot, accumulated in the
  dual-prime residues (q1=1073738753, q2=1073732609, M≈2^60 fits u64, Garner
  inv=894602413, no `__int128`) — reuse `sp_pr_mul`/Barrett from the gated L2 crate
  `tools/sp_dsp_smoke`. Integer addition is associative ⇒ score is reduction-order-
  immune ⇒ byte-identical across machines (the G-R3-BIND-on-O_K property).
- **Argmax** declares a deterministic tie-break: **lowest episode index wins**. Makes
  select→replay→byte-exact-forward a total order ⇒ end-to-end cross-machine
  determinism. (A float tie could split the replayed episode across hardware and
  silently break the auditability claim even though each forward stays byte-exact.)

Quantization is a **frozen, declared approximation, NOT runtime drift**. int16 gives
~3e-5 relative precision, far finer than the gap we aim to open — but we PROVE that:
the gate runs on the QUANTIZED integer scores, not the float W_c.

## PRE-REGISTERED GATE — G-CHAT-B3-RECALL-v3 (decide BEFORE training)

Computed on the **quantized int16 W_c**, scoring in the integer/CRT residue domain,
on the live 12B with the existing recall query set (3 topical + ≥2 foreign):

1. **Separation (primary, integer scores):** `min_target_topm > max_foreign_topm`
   with a strictly positive margin. Equivalently: every topical query's RIGHT episode
   is the argmax AND every foreign query's best score sits below every topical
   target's score. (B3-v2 failed exactly here: min_target 0.394 < max_foreign 0.416.)
2. **Content A/B (confirmatory):** with `SP_RECALL_WC` armed + `auto_recall:true` and
   NO operator `replay`, each topical query auto-selects+replays its RIGHT episode and
   the answer carries that episode's ground-truth telltale (Boulter @-@ / "Herons" /
   "Royal Court"; homarus → lobster; headlam → RAAF/Australian); each foreign query
   does NOT fire (best integer score < TAU); and `auto_recall:false` hallucinates
   (proves replay, not the prompt, supplies the truth).
3. **Null floor:** `SP_RECALL_WC` unset = byte-untouched; the gemma4_kv one-shot path
   stays untouched; B3-v1/v2 receipts do not regress.

**Honest scope, stated up front:** N=3 episodes ⇒ this is a MECHANISM check (does a
learned metric open the gap on these episodes?), not a generalization claim. A real
addresser needs more episodes; that is a follow-on, not a goalpost move. If the gap
does NOT survive quantization, the build is RED and we say so.

## Harness components (this directory)

- `b3_make_dataset.py` — mine (query global-Q, episode global-K, label) pairs from the
  substrate. Episode-K from `ep.k` (the `load_episode_global_k` layout). Query-Q from
  the daemon's `SP_B3_QDUMP` seam (below): POST each training query, collect its
  last-token global-Q. Default query set = paraphrase/sub-question expansion per
  episode (positives) + a foreign bank (hard negatives). No external corpus.
- `b3_train_wc.py` — contrastive trainer (InfoNCE forward-KL + 0.2 hard-neg hinge +
  learnable τ, adapted from `train_lsh.py`) → float `W_c` `[HD, r]` (+ τ). GPU 2060.
- `b3_export_wc.py` — quantize `W_c` → `lsh_Wc_i16_sK.bin`; quant-error check
  (max |Wc_int/2^k − Wc_f32|); then re-score the live recall matrix in the INTEGER
  domain and evaluate gate #1 on the quantized scores. Emits the v3 receipt.

## Engine patch to APPLY before execution (additive, default-off, then rebuild)

### (a) `SP_B3_QDUMP` query-Q dump — `routes.rs`, in the auto_recall block, right
after `read_global_q` succeeds:
```rust
// B3-v3 dataset: when SP_B3_QDUMP=<dir> is set, persist this turn's last-token
// global-Q (the exact vector qk_relevance scores) so the offline trainer can mine
// (Q, episode-K) pairs. Filename = sanitized prompt. Additive; default unset = no-op.
if let Ok(dir) = std::env::var("SP_B3_QDUMP") {
    let safe: String = prompt.chars().map(|c| if c.is_ascii_alphanumeric() {c} else {'_'})
        .take(48).collect();
    let p = std::path::Path::new(&dir).join(format!("q_{safe}.bin"));
    let mut buf = Vec::with_capacity(8 + q.len()*4);
    buf.extend_from_slice(&(n_global_q as u32).to_le_bytes());
    buf.extend_from_slice(&(qd as u32).to_le_bytes());      // G_NH*HD
    for &x in &q { buf.extend_from_slice(&x.to_le_bytes()); }
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(p, buf);
}
```

### (b) `W_c` integer scoring — `recall.rs` `qk_relevance`: add an optional
`wc: Option<&WcInt>` arg. When present, project `q_head` and `K[p]` through `Wc_int`
to r dims and dot IN INTEGER (the q/K fixed-point already enters from the islands; for
the host-side daemon prototype, project in i64 with the declared 2^k scale and the
dual-prime residues, NOT f64). `WcInt { rows: HD, cols: r, s_log2: u32, w: Vec<i16> }`
loaded from `SP_RECALL_WC`. Scoring (host reference, integer, reduction-order-immune):
```
proj_q[j] = Σ_d Wc[d,j] * round(q[h,d] * 2^k)      // i64, exact
proj_k[j] = Σ_d Wc[d,j] * round(K[p,d] * 2^k)      // i64, exact
score(h,p) = Σ_j proj_q[j]*proj_k[j]               // i64; reduce mod q1,q2 + Garner
```
top-m mean + max as today, but on the integer scores; argmax tie-break = lowest index.

### (c) deploy: `SP_RECALL_WC=lsh_Wc_i16_sK.bin` loads `WcInt`; routes.rs passes it to
`qk_relevance`; unset ⇒ today's raw-f32 path (null floor preserved).

## Execution order (after this harness lands + the patch is applied & built)

1. Apply patch (a), rebuild daemon (PMAX=4096, the proven config).
2. `b3_make_dataset.py` → POST query set with `SP_B3_QDUMP`, assemble `b3_data.npz`.
3. `b3_train_wc.py b3_data.npz` → `lsh_Wc_f32.npz` (+ τ).
4. `b3_export_wc.py` → `lsh_Wc_i16_sK.bin` + quant-error + **gate #1 on integer scores**.
5. If gate #1 GREEN: apply patch (b)+(c), rebuild, run gate #2 (content A/B). Receipt
   `G-CHAT-B3-RECALL-v3.log`, commit, contract run-record.
6. If RED at any step: record the honest negative; replay stays operator-driven.
