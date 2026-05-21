/* test_generate.c — GEN_QWEN3: greedy generation loop over the Qwen3 CPU forward.
 *
 * Generation IDs are emitted (the tokenizer that turns them into text is the next
 * deliverable). Checks: (1) the loop appends exactly n_gen tokens, (2) it is
 * deterministic across runs, (3) the first generated token equals the argmax of
 * the prompt's last-position logits from the E_CPU_2-validated forward — i.e. the
 * loop is consistent with the forward it is built on.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_QWEN3_REF
#define SP_QWEN3_REF "qwen3_ref.bin"
#endif

#define N_GEN 8

static void GEN_QWEN3(void) {
    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open oracle ref.bin (prompt token IDs)");
    if (!f) { fprintf(stderr, "    (no ref: %s)\n", SP_QWEN3_REF); return; }
    uint32_t magic = 0, nt = 0, nv = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&nt, 4, 1, f) != 1 || fread(&nv, 4, 1, f) != 1) {
        SP_CHECK(0, "read ref header"); fclose(f); return;
    }
    int32_t *prompt = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int ok = prompt && fread(prompt, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    SP_CHECK(ok, "read prompt token IDs");
    if (!ok) { free(prompt); return; }

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) { free(prompt); return; }
    nv = m->cfg.n_vocab;

    /* expected first generated token = argmax of the forward's last-position logits */
    float *logits = (float *)malloc((size_t)nt * nv * sizeof(float));
    int expect_first = -1;
    if (logits && qwen3_forward(m, prompt, (int)nt, logits) == 0) {
        const float *last = logits + (size_t)(nt - 1) * nv;
        expect_first = 0;
        for (uint32_t j = 1; j < nv; j++) if (last[j] > last[expect_first]) expect_first = (int)j;
    }
    SP_CHECK(expect_first >= 0, "forward(prompt) for expected first token");

    /* two independent generation runs from the same prompt */
    int32_t *seqA = (int32_t *)malloc((size_t)(nt + N_GEN) * sizeof(int32_t));
    int32_t *seqB = (int32_t *)malloc((size_t)(nt + N_GEN) * sizeof(int32_t));
    int rcA = -1, rcB = -1;
    if (seqA && seqB) {
        memcpy(seqA, prompt, (size_t)nt * sizeof(int32_t));
        memcpy(seqB, prompt, (size_t)nt * sizeof(int32_t));
        rcA = qwen3_generate(m, seqA, (int)nt, N_GEN, /*eos=*/-1);
        rcB = qwen3_generate(m, seqB, (int)nt, N_GEN, /*eos=*/-1);
    }
    SP_CHECK_EQ_I64(rcA, (int)nt + N_GEN, "generate appends exactly N_GEN tokens");
    SP_CHECK(rcA == rcB, "generation length deterministic");

    if (rcA == (int)nt + N_GEN && rcA == rcB) {
        int tokens_match = memcmp(seqA, seqB, (size_t)rcA * sizeof(int32_t)) == 0;
        SP_CHECK(tokens_match, "generated token IDs deterministic across runs");
        SP_CHECK_EQ_I64(seqA[nt], expect_first, "first generated token == forward last-pos argmax");
        fprintf(stderr, "    prompt %u tok -> generated:", nt);
        for (int i = 0; i < N_GEN; i++) fprintf(stderr, " %d", seqA[nt + i]);
        fprintf(stderr, "\n");
    }

    free(prompt); free(logits); free(seqA); free(seqB);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(GEN_QWEN3);
    return SP_DONE();
}
