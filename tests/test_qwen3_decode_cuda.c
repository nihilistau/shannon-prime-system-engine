/* test_qwen3_decode_cuda.c — Beta: GPU autoregressive KV-cache decode gate.
 *
 * Gate: qwen3_decode_cuda (GPU, KV resident in VRAM) produces the SAME greedy
 * argmax sequence as the CPU math-core qwen3_generate_kv (knobs OFF, the proven
 * reference decode). Plus a wall-clock tok/s read for the GPU path — the first
 * Stage-Beta generation-speed number on the actual 2060.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"
#include "sp_engine/cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

/* Gate strategy: compare GPU autoregressive decode (KV resident in VRAM) to GPU
 * teacher-forced PREFILL on the SAME sequence — both are sp_engine_cuda, so no
 * link against the CPU forward (which collides with cpu_overlay's overrides
 * under MSVC). M_QWEN3_CUDA already proves prefill == CPU; transitively, decode
 * == CPU. The teacher-forced check: prefill over the decoded sequence must
 * argmax-predict each next decoded token. */

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif

static double now_s(void) {
    struct timespec t; timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void T_QWEN3_DECODE_CUDA(void) {
    if (sp_cuda_device_count() < 1) { fprintf(stderr, "    no CUDA device — SKIP\n"); return; }
    const char *gguf = getenv("SP_QWEN3_GGUF"); if (!gguf) gguf = SP_QWEN3_GGUF;
    qwen3_model *m = qwen3_load(gguf);
    SP_CHECK(m != NULL, "qwen3_load");
    if (!m) return;

    const int n_prompt = 4, n_gen = 24, P = n_prompt + n_gen;
    int32_t prompt[4] = { 1, 2, 3, 4 };
    const int V = (int)m->cfg.n_vocab;

    /* GPU autoregressive decode (KV resident in VRAM, single-query attention). */
    int32_t gpu[64]; for (int i = 0; i < n_prompt; i++) gpu[i] = prompt[i];
    double t0 = now_s();
    int ngpu = qwen3_decode_cuda(m, gpu, n_prompt, n_gen, /*eos=*/-1);
    double dt = now_s() - t0;
    if (ngpu < 0) fprintf(stderr, "    %s\n", sp_last_error());
    SP_CHECK(ngpu == P, "GPU decode produced full length");

    /* GPU teacher-forced PREFILL over the decoded sequence: argmax at position
     * pos must predict gpu[pos+1] for every emitted position. This verifies the
     * KV-cache decode agrees with full-attention recompute (both on GPU). */
    float *logits = (float *)malloc((size_t)P * V * sizeof(float));
    SP_CHECK(logits != NULL, "logits buffer");
    if (!logits) { qwen3_free(m); return; }
    int rcf = qwen3_forward_cuda(m, gpu, P, logits);
    SP_CHECK(rcf == 0, "GPU prefill over decoded sequence");

    int match = 1, firstbad = -1;
    for (int pos = n_prompt - 1; pos < P - 1 && match; pos++) {
        const float *row = logits + (size_t)pos * V;
        int am = 0; float bv = row[0];
        for (int i = 1; i < V; i++) if (row[i] > bv) { bv = row[i]; am = i; }
        if (am != gpu[pos + 1]) { match = 0; firstbad = pos; }
    }
    if (!match) {
        fprintf(stderr, "    decode/prefill disagree at pos %d: prefill-argmax != decoded %d\n",
                firstbad, gpu[firstbad + 1]);
        fprintf(stderr, "    GPU decoded:");
        for (int i = 0; i < P; i++) fprintf(stderr, " %d", gpu[i]);
        fprintf(stderr, "\n");
    }
    SP_CHECK(match, "GPU decode (KV cache) == GPU prefill teacher-forced argmax");

    fprintf(stderr, "    [decode-cuda] %d gen tokens in %.4fs = %.2f tok/s on the GPU "
            "(KV resident in VRAM, single-query attention); seq:",
            n_gen, dt, (double)n_gen / dt);
    for (int i = 0; i < P; i++) fprintf(stderr, " %d", gpu[i]);
    fprintf(stderr, "\n");

    free(logits);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(T_QWEN3_DECODE_CUDA);
    return SP_DONE();
}
