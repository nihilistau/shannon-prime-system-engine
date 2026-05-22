/* test_release.c — E_CPU_10: releasing the F16 source (Phase 1b, §4.8).
 *
 * With the matmul weights AND the embedding packed in the arena and the norms
 * copied to owned f32, the GGUF data mapping can be unmapped — the forward then
 * reads only the arena + owned norms. The key invariant:
 *
 *   forward AFTER release is BYTE-IDENTICAL to forward with the mapping still
 *   held (release frees memory; it does not change arithmetic). A dangling
 *   pointer would change the logits or crash, so this catches ownership bugs.
 *
 * Also: after release gguf_tensor_data() is NULL (the source is really gone), and
 * an owning tokenizer (sp_tokenizer_load_ex(g,1)) still decodes correctly.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"
#include "sp_engine/tokenizer.h"

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

static void E_CPU_10(void) {
    uint32_t nt = 0;
    int32_t *toks = read_ref_ids(&nt);
    SP_CHECK(toks != NULL, "read oracle ref.bin (token IDs)");
    if (!toks) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return; }

    /* discover vocab */
    qwen3_model *m0 = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m0 != NULL, "qwen3_load");
    if (!m0) { free(toks); return; }
    uint32_t nv = m0->cfg.n_vocab;
    qwen3_free(m0);
    size_t nlog = (size_t)nt * nv;

    float *held = (float *)malloc(nlog * sizeof(float));
    float *rel  = (float *)malloc(nlog * sizeof(float));
    if (!held || !rel) { SP_CHECK(0, "alloc logits"); free(held); free(rel); free(toks); return; }

    /* (A) embed-in-arena, mapping HELD (no release) */
    set_env("SP_ARENA", "q8"); set_env("SP_ARENA_EMBED", "1"); set_env("SP_ARENA_RELEASE", NULL);
    qwen3_model *mh = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(mh != NULL, "load arena+embed (mapping held)");
    int held_ok = mh && qwen3_forward(mh, toks, (int)nt, held) == 0;
    SP_CHECK(held_ok, "forward (mapping held)");
    if (mh) qwen3_free(mh);

    /* (B) manual release after loading an OWNING tokenizer; mapping then unmapped */
    set_env("SP_ARENA", "q8"); set_env("SP_ARENA_EMBED", "1"); set_env("SP_ARENA_RELEASE", NULL);
    qwen3_model *mr = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(mr != NULL, "load arena+embed for manual release");
    if (mr) {
        sp_tokenizer *tk = sp_tokenizer_load_ex(mr->gguf, 1 /*own*/);
        SP_CHECK(tk != NULL, "owning tokenizer loaded before release");
        int rrc = qwen3_release_source(mr);
        SP_CHECK(rrc == 0 && mr->released, "qwen3_release_source succeeded");
        SP_CHECK(gguf_tensor_data(mr->gguf, mr->token_embd) == NULL,
                 "GGUF data unmapped after release (source really gone)");
        int rel_ok = qwen3_forward(mr, toks, (int)nt, rel) == 0;
        SP_CHECK(rel_ok, "forward after release");
        if (held_ok && rel_ok) {
            size_t bad = (size_t)-1;
            for (size_t i = 0; i < nlog; i++) if (held[i] != rel[i]) { bad = i; break; }
            SP_CHECK(bad == (size_t)-1, "released forward BYTE-IDENTICAL to held forward");
        }
        if (tk) {                          /* tokenizer must survive the unmap */
            char text[4096];
            long L = sp_tokenizer_decode(tk, toks, (int)nt, text, sizeof text);
            SP_CHECK(L > 0 && strstr(text, "factorization") && strstr(text, "dominance"),
                     "owning tokenizer decodes correctly after release");
            sp_tokenizer_free(tk);
        }
        qwen3_free(mr);
    }

    /* (C) auto-release at load (SP_ARENA_RELEASE=1) matches held too */
    set_env("SP_ARENA", "q8"); set_env("SP_ARENA_EMBED", NULL); set_env("SP_ARENA_RELEASE", "1");
    qwen3_model *ma = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(ma != NULL && ma->released, "auto-release load (SP_ARENA_RELEASE=1)");
    if (ma) {
        float *ra = (float *)malloc(nlog * sizeof(float));
        if (ra && qwen3_forward(ma, toks, (int)nt, ra) == 0 && held_ok) {
            size_t bad = (size_t)-1;
            for (size_t i = 0; i < nlog; i++) if (held[i] != ra[i]) { bad = i; break; }
            SP_CHECK(bad == (size_t)-1, "auto-released forward byte-identical to held");
        }
        free(ra);
        qwen3_free(ma);
    }
    set_env("SP_ARENA", NULL); set_env("SP_ARENA_EMBED", NULL); set_env("SP_ARENA_RELEASE", NULL);

    free(held); free(rel); free(toks);
}

int main(void) {
    SP_RUN(E_CPU_10);
    return SP_DONE();
}
