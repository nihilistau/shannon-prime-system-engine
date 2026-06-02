/* sp_toks.c — SPEED_BASELINE: time qwen3_generate_kv decode throughput (tok/s).
 *
 * Loads the f16 GGUF via qwen3_load, warms once to page weights in, then times
 * n_gen persistent-KV decode steps and prints tok/s. Compare vs llama.cpp on the
 * same model. Measures the SP forward AS-IS (the WIRE-gap scalar/f32 path); the
 * gap vs llama.cpp is what SPEED_WIRE_CPU (packed Q8/Q4 + VNNI) must close.
 *
 * Env: SP_TOKS_N overrides the timed token count (default 32). SP_CPU_SCALAR=1
 *      forces the scalar path; default uses SP's best current CPU path.
 */
#include "sp_engine/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif

static double now_s(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(void) {
    const char *ng = getenv("SP_TOKS_N");
    int n_gen = ng ? atoi(ng) : 32;
    if (n_gen < 1) n_gen = 32;

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    if (!m) { fprintf(stderr, "[sp_toks] load FAIL: %s\n", SP_QWEN3_GGUF); return 1; }

    const int n_prompt = 4;
    int32_t *seq = (int32_t *)malloc((size_t)(n_prompt + n_gen) * sizeof(int32_t));
    if (!seq) { fprintf(stderr, "[sp_toks] OOM\n"); return 1; }

    seq[0] = 1; seq[1] = 2; seq[2] = 3; seq[3] = 4;
    (void)qwen3_generate_kv(m, seq, n_prompt, 2, -1);   /* warm: page weights in */

    seq[0] = 1; seq[1] = 2; seq[2] = 3; seq[3] = 4;
    double t0 = now_s();
    int n = qwen3_generate_kv(m, seq, n_prompt, n_gen, -1);
    double dt = now_s() - t0;

    fprintf(stderr,
        "[sp_toks] gen %d tokens in %.3f s = %.2f tok/s (prompt=%d, total=%d, model=%s)\n",
        n_gen, dt, (dt > 0 ? n_gen / dt : 0.0), n_prompt, n, SP_QWEN3_GGUF);
    /* token IDs for the top-1 accuracy gate (diff VNNI vs scalar oracle) */
    fprintf(stderr, "[sp_toks] tokens:");
    for (int i = n_prompt; i < n && i < n_prompt + 24; i++) fprintf(stderr, " %d", seq[i]);
    fprintf(stderr, "\n");
    free(seq);
    return 0;
}
