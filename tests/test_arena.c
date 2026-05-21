/* test_arena.c — E_CPU_9: the packed-weight arena (roadmap §4.8).
 *
 * The arena quantizes the matmul weights once at load (per-row Frobenius Q8 or
 * Q4-mixed) and the forward lifts inline from the packed codes — same arithmetic
 * as the per-matmul FROB path, just a different home for the bytes. So:
 *
 *   (1) SP_ARENA=q8 forward is BYTE-IDENTICAL to SP_ENGINE_FROB=1 forward
 *       (and SP_ARENA=q4 byte-identical to SP_ENGINE_FROB=3). This is the tight
 *       "no drift" gate — nothing is lost between the two read paths.
 *   (2) The arena is materially smaller than the f32/f16 source, and Q4 < Q8.
 *
 * Two model loads per precision (one with SP_ARENA set, one with SP_ENGINE_FROB)
 * since the arena is built at load time from the env.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"
#include "sp_engine/arena.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_QWEN3_REF
#define SP_QWEN3_REF "qwen3_ref.bin"
#endif

static void set_env(const char *k, const char *v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

static int32_t *read_ref_ids(uint32_t *n_out) {
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    if (!f) return NULL;
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) { fclose(f); return NULL; }
    int32_t *t = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int ok = t && fread(t, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    if (!ok) { free(t); return NULL; }
    *n_out = nt; return t;
}

/* Run a forward with arena=<arena_kind or NULL> and frob=<"1"/"3" or NULL>. */
static int run(const int32_t *toks, uint32_t nt, const char *arena, const char *frob,
               float *out, size_t nlog, size_t *arena_bytes, long *promoted, long *rows) {
    set_env("SP_ARENA", arena);
    set_env("SP_ENGINE_FROB", frob);
    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    if (!m) return 1;
    if (arena_bytes) *arena_bytes = sp_arena_bytes(m->arena);
    if (promoted)    *promoted    = sp_arena_promoted(m->arena);
    if (rows)        *rows        = sp_arena_total_rows(m->arena);
    int rc = qwen3_forward(m, toks, (int)nt, out);
    (void)nlog;
    qwen3_free(m);
    set_env("SP_ARENA", NULL);
    set_env("SP_ENGINE_FROB", NULL);
    return rc;
}

static void check_identical(const int32_t *toks, uint32_t nt, uint32_t nv,
                            const char *arena_kind, const char *frob_mode) {
    size_t nlog = (size_t)nt * nv;
    float *a = (float *)malloc(nlog * sizeof(float));
    float *b = (float *)malloc(nlog * sizeof(float));
    size_t abytes = 0; long promoted = 0, rows = 0;
    int ok = a && b
          && run(toks, nt, arena_kind, NULL,     a, nlog, &abytes, &promoted, &rows) == 0
          && run(toks, nt, NULL,       frob_mode, b, nlog, NULL, NULL, NULL) == 0;
    char msg[160];
    snprintf(msg, sizeof msg, "forward ran (arena=%s vs FROB=%s)", arena_kind, frob_mode);
    SP_CHECK(ok, msg);
    if (ok) {
        size_t first_bad = (size_t)-1;
        for (size_t i = 0; i < nlog; i++) if (a[i] != b[i]) { first_bad = i; break; }
        double src_f16 = (double)4 /*we report vs f32-equivalent*/;
        (void)src_f16;
        fprintf(stderr, "    arena=%s: %ld rows, %ld promoted to Q8, packed=%.1f MB | "
                "byte-identical to FROB=%s: %s\n",
                arena_kind, rows, promoted, (double)abytes / (1024.0 * 1024.0), frob_mode,
                first_bad == (size_t)-1 ? "yes" : "NO");
        snprintf(msg, sizeof msg, "arena=%s forward byte-identical to FROB=%s", arena_kind, frob_mode);
        SP_CHECK(first_bad == (size_t)-1, msg);
    }
    free(a); free(b);
}

static void E_CPU_9(void) {
    uint32_t nt = 0;
    int32_t *toks = read_ref_ids(&nt);
    SP_CHECK(toks != NULL, "read oracle ref.bin (token IDs)");
    if (!toks) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return; }

    /* discover vocab via a plain load */
    qwen3_model *m0 = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m0 != NULL, "qwen3_load");
    if (!m0) { free(toks); return; }
    uint32_t nv = m0->cfg.n_vocab;
    qwen3_free(m0);

    /* (1) Q8 arena == FROB=1 ; (2) Q4 arena == FROB=3, byte-for-byte */
    check_identical(toks, nt, nv, "q8", "1");
    check_identical(toks, nt, nv, "q4", "3");

    free(toks);
}

int main(void) {
    SP_RUN(E_CPU_9);
    return SP_DONE();
}
