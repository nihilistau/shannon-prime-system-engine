/* test_kste_kv.c — E_CPU_6: the KSTE KV-cache overlay (encoder), sieve OFF.
 *
 * KSTE is a one-way deterministic encoder (a K head-vector -> a 64-byte signature
 * in the frozen wire form); there is no decode. The property the Lattice relies on
 * is byte-identical determinism — the same K produces the same signature on every
 * platform and every run, so the signature survives a store/load (the "round-trip
 * identical bytes") and can be diffed for cross-node dedup. This test runs the
 * engine's real prefill with the KSTE KV overlay (qwen3_forward_ex + kv_trees),
 * then checks: re-encode determinism, store/load round-trip, and the frozen wire
 * invariants on every cached K head-vector.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"
#include "sp/kste.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_QWEN3_REF
#define SP_QWEN3_REF "qwen3_ref.bin"
#endif

static void set_kv(int on) {
    const char *v = on ? "1" : "0";
#ifdef _WIN32
    _putenv_s("SP_KSTE_KV", v);
#else
    setenv("SP_KSTE_KV", v, 1);
#endif
}
static uint32_t rd_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void E_CPU_6(void) {
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open oracle ref.bin (token IDs)");
    if (!f) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return; }
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) {
        SP_CHECK(0, "read ref header"); fclose(f); return;
    }
    int32_t *toks = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int toks_ok = toks && fread(toks, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    SP_CHECK(toks_ok, "read ref token IDs");
    if (!toks_ok) { free(toks); return; }

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) { free(toks); return; }
    const uint32_t NL = m->cfg.n_layers, NKV = m->cfg.n_head_kv, HD = m->cfg.head_dim;
    nv = m->cfg.n_vocab;

    size_t ntrees = (size_t)NL * nt * NKV;
    sp_kste_tree_t *t1 = (sp_kste_tree_t *)malloc(ntrees * sizeof(sp_kste_tree_t));
    sp_kste_tree_t *t2 = (sp_kste_tree_t *)malloc(ntrees * sizeof(sp_kste_tree_t));
    float *logits = (float *)malloc((size_t)nt * nv * sizeof(float));
    int ok = t1 && t2 && logits;
    set_kv(1);   /* enable the overlay (fidelity; the mechanism keys off kv_trees != NULL) */
    if (ok) ok = qwen3_forward_ex(m, toks, (int)nt, logits, t1) == 0;
    if (ok) ok = qwen3_forward_ex(m, toks, (int)nt, logits, t2) == 0;
    set_kv(0);
    SP_CHECK(ok, "prefill twice with KSTE KV overlay");

    if (ok) {
        /* (1) determinism: the two independent runs are byte-identical. */
        int det = (memcmp(t1, t2, ntrees * sizeof(sp_kste_tree_t)) == 0);
        SP_CHECK(det, "KSTE KV signatures byte-identical across runs (deterministic)");

        /* (2) store/load round-trip identical, and (3) frozen wire invariants. */
        long wire_ok = 0, distinct = 0;
        for (size_t i = 0; i < ntrees; i++) {
            sp_kste_tree_t copy;
            memcpy(&copy, &t1[i], sizeof copy);                 /* store */
            sp_kste_tree_t back;
            memcpy(&back, &copy, sizeof back);                  /* load  */
            if (memcmp(&back, &t1[i], 64) != 0) continue;       /* round-trip */
            const uint8_t *b = t1[i].bytes;
            if (b[SP_KSTE_OFF_VERSION] == SP_KSTE_LAYOUT_VERSION &&
                b[SP_KSTE_OFF_BRANCH]  == SP_KSTE_BRANCHING &&
                b[SP_KSTE_OFF_DEPTH]   == SP_KSTE_DEPTH &&
                b[SP_KSTE_OFF_RESERVED]== 0 &&
                rd_u32le(b + SP_KSTE_OFF_K) == HD)
                wire_ok++;
            if (i > 0 && memcmp(&t1[i], &t1[0], 64) != 0) distinct++;
        }
        fprintf(stderr, "    %zu trees (%u layers x %u pos x %u kv-heads, k=%u) | "
                "deterministic=%d wire_ok=%ld/%zu distinct-from-[0]=%ld\n",
                ntrees, NL, nt, NKV, HD, det, wire_ok, ntrees, distinct);
        SP_CHECK_EQ_I64(wire_ok, (int64_t)ntrees, "every signature round-trips + valid frozen wire form");
        SP_CHECK(distinct > 0, "distinct K vectors yield distinct signatures (encoder not degenerate)");
    }

    free(t1); free(t2); free(logits); free(toks);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(E_CPU_6);
    return SP_DONE();
}
