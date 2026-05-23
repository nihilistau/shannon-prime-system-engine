/* generate.c — greedy autoregressive generation over the Qwen3 CPU forward.
 *
 * Reference loop: re-prefill the whole prefix each step (qwen3_forward) and pick
 * the argmax of the last position's logits. O(n^2) but correct by construction —
 * it inherits the E_CPU_2-validated forward, so its tokens are exactly what the
 * forward predicts. A persistent-KV-cache decode (O(n)) is a later optimization
 * to be validated for token-identity against this path. */
#include "sp_engine/model.h"

#include <stdlib.h>

int qwen3_generate(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                   int eos_id) {
    if (!m || !seq || n_prompt <= 0 || n_gen < 0) return -1;
    const int V = (int)m->cfg.n_vocab;

    /* logits buffer sized for the longest prefix we will run (the full sequence). */
    float *logits = (float *)malloc((size_t)(n_prompt + n_gen) * V * sizeof(float));
    if (!logits) return -1;

    int n = n_prompt;
    for (int step = 0; step < n_gen; step++) {
        if (qwen3_forward(m, seq, n, logits)) { free(logits); return -1; }
        const float *last = logits + (size_t)(n - 1) * V;   /* next-token distribution */
        int amax = 0;
        for (int j = 1; j < V; j++) if (last[j] > last[amax]) amax = j;
        seq[n++] = amax;
        if (eos_id >= 0 && amax == eos_id) break;
    }

    free(logits);
    return n;
}
