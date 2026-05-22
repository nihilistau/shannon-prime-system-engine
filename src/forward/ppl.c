/* ppl.c — corpus perplexity, replicating llama.cpp's default perplexity path.
 * See sp_engine/ppl.h. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp_engine/ppl.h"

#include <stdlib.h>
#include <math.h>

typedef int (*forward_fn)(const qwen3_model *, const int32_t *, int, float *);

int sp_perplexity(const qwen3_model *m, const sp_tokenizer *tok,
                  const char *text, size_t text_len, int n_ctx,
                  double *ppl, double *ppl_stderr, long *n_scored) {
    if (!m || !tok || !text || n_ctx < 4 || !ppl) return 1;
    const int V = (int)m->cfg.n_vocab;
    forward_fn fwd = (m->cfg.arch == SP_ARCH_GEMMA3) ? gemma3_forward : qwen3_forward;

    /* tokenize the whole corpus once (BOS auto-prepended when add_bos_token=1). */
    int cap = (int)text_len + 16;
    int32_t *toks = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (!toks) return 1;
    long nt = sp_tokenizer_encode(tok, text, text_len, /*parse_special=*/0, toks, cap);
    if (nt < 0 || nt > cap) { free(toks); return 1; }

    int n_chunk = (int)(nt / n_ctx);
    if (n_chunk < 1) { free(toks); return 1; }
    const int first = n_ctx / 2;

    /* BOS to re-anchor each chunk at position 0: the corpus tokenization already
     * begins with BOS, so toks[0] is it. */
    int32_t bos_id = toks[0];

    int32_t *chunk  = (int32_t *)malloc((size_t)n_ctx * sizeof(int32_t));
    float   *logits = (float *)malloc((size_t)n_ctx * (size_t)V * sizeof(float));
    if (!chunk || !logits) { free(toks); free(chunk); free(logits); return 1; }

    double nll = 0.0, nll2 = 0.0;
    long count = 0;
    int rc = 0;
    for (int c = 0; c < n_chunk; c++) {
        int start = c * n_ctx;
        chunk[0] = bos_id;                                  /* re-anchor with BOS */
        for (int k = 1; k < n_ctx; k++) chunk[k] = toks[start + k];
        if (fwd(m, chunk, n_ctx, logits)) { rc = 1; break; }
        for (int p = first; p < n_ctx - 1; p++) {
            const float *lg = logits + (size_t)p * V;
            int32_t target = toks[start + p + 1];
            double maxl = lg[0];
            for (int v = 1; v < V; v++) if (lg[v] > maxl) maxl = lg[v];
            double sumexp = 0.0;
            for (int v = 0; v < V; v++) sumexp += exp((double)lg[v] - maxl);
            double logp = (double)lg[target] - maxl - log(sumexp);
            nll  += -logp;
            nll2 += logp * logp;
            count++;
        }
    }
    free(toks); free(chunk); free(logits);
    if (rc || count == 0) return rc ? rc : 1;

    double mean = nll / (double)count;
    *ppl = exp(mean);
    if (ppl_stderr) {
        double var = nll2 / (double)count - mean * mean;     /* var of -logp == var of logp */
        *ppl_stderr = (var > 0.0 && count > 1) ? sqrt(var / (double)(count - 1)) * (*ppl) : 0.0;
    }
    if (n_scored) *n_scored = count;
    return 0;
}
