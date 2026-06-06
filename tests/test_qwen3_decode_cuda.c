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

static void set_graph(int on) {
#if defined(_WIN32)
    _putenv_s("SP_CUDA_DECODE_GRAPH", on ? "1" : "0");
#else
    if (on) setenv("SP_CUDA_DECODE_GRAPH", "1", 1); else unsetenv("SP_CUDA_DECODE_GRAPH");
#endif
}

/* Run the GPU decode (knobs as set), return wall-clock seconds via *dt. */
static int decode_run(qwen3_model *m, const int32_t *prompt, int n_prompt,
                      int n_gen, int32_t *out, double *dt) {
    for (int i = 0; i < n_prompt; i++) out[i] = prompt[i];
    double t0 = now_s();
    int n = qwen3_decode_cuda(m, out, n_prompt, n_gen, /*eos=*/-1);
    *dt = now_s() - t0;
    return n;
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

    /* (1) per-step decode (graph OFF) — the proven Beta-S0 path. */
    set_graph(0);
    int32_t seq_ref[64]; double dt_ref = 0.0;
    int n_ref = decode_run(m, prompt, n_prompt, n_gen, seq_ref, &dt_ref);
    if (n_ref < 0) fprintf(stderr, "    %s\n", sp_last_error());
    SP_CHECK(n_ref == P, "per-step decode produced full length");

    /* (2) CUDA-graph decode (graph ON, BETA.2) — capture once, replay/token. */
    set_graph(1);
    int32_t seq_g[64]; double dt_g = 0.0;
    int n_g = decode_run(m, prompt, n_prompt, n_gen, seq_g, &dt_g);
    if (n_g < 0) fprintf(stderr, "    %s\n", sp_last_error());
    SP_CHECK(n_g == P, "graph decode produced full length");

    /* (3) byte-exact equality: the graph path must emit the SAME sequence as the
     * per-step path (position-indirect kernels are numerically identical). */
    int eq = (n_ref == n_g), firsteq = -1;
    for (int i = 0; i < P && eq; i++) if (seq_ref[i] != seq_g[i]) { eq = 0; firsteq = i; }
    if (!eq) {
        fprintf(stderr, "    graph != per-step at idx %d (%d vs %d)\n",
                firsteq, firsteq>=0?seq_ref[firsteq]:-1, firsteq>=0?seq_g[firsteq]:-1);
        fprintf(stderr, "    per-step:"); for (int i=0;i<P;i++) fprintf(stderr," %d",seq_ref[i]);
        fprintf(stderr, "\n    graph   :"); for (int i=0;i<P;i++) fprintf(stderr," %d",seq_g[i]);
        fprintf(stderr, "\n");
    }
    SP_CHECK(eq, "CUDA-graph decode == per-step decode (byte-exact)");

    /* (4) GPU teacher-forced PREFILL over the decoded sequence: argmax at pos must
     * predict seq[pos+1] for every emitted position — anchors decode to recompute.
     * M_QWEN3_CUDA already proves prefill == CPU; transitively decode == CPU. */
    float *logits = (float *)malloc((size_t)P * V * sizeof(float));
    SP_CHECK(logits != NULL, "logits buffer");
    if (!logits) { qwen3_free(m); return; }
    int rcf = qwen3_forward_cuda(m, seq_g, P, logits);
    SP_CHECK(rcf == 0, "GPU prefill over decoded sequence");

    int match = 1, firstbad = -1;
    for (int pos = n_prompt - 1; pos < P - 1 && match; pos++) {
        const float *row = logits + (size_t)pos * V;
        int am = 0; float bv = row[0];
        for (int i = 1; i < V; i++) if (row[i] > bv) { bv = row[i]; am = i; }
        if (am != seq_g[pos + 1]) { match = 0; firstbad = pos; }
    }
    if (!match)
        fprintf(stderr, "    decode/prefill disagree at pos %d: != decoded %d\n",
                firstbad, seq_g[firstbad + 1]);
    SP_CHECK(match, "GPU decode == GPU prefill teacher-forced argmax");

    double tps_ref = (double)n_gen / dt_ref, tps_g = (double)n_gen / dt_g;
    fprintf(stderr, "    [decode-cuda] per-step %.2f tok/s | graph %.2f tok/s "
            "(%.2fx) — %d gen tokens; seq:",
            tps_ref, tps_g, tps_g / tps_ref, n_gen);
    for (int i = 0; i < P; i++) fprintf(stderr, " %d", seq_g[i]);
    fprintf(stderr, "\n");

    free(logits);
    qwen3_free(m);
}

int main(void) {
    SP_RUN(T_QWEN3_DECODE_CUDA);
    return SP_DONE();
}
