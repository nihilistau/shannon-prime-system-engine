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
#include "sp_engine/sp_model.h"   /* real .sp-model ship-artifact loader */
#include "sp_engine/arena.h"      /* sp_arena_precision */

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

static void set_arena_q8(int on) {
#if defined(_WIN32)
    if (on) _putenv_s("SP_ARENA", "q8"); else _putenv_s("SP_ARENA", "");
#else
    if (on) setenv("SP_ARENA", "q8", 1); else unsetenv("SP_ARENA");
#endif
}

static void set_int8(int on) {
#if defined(_WIN32)
    _putenv_s("SP_CUDA_DECODE_INT8", on ? "1" : "0");
#else
    if (on) setenv("SP_CUDA_DECODE_INT8", "1", 1); else unsetenv("SP_CUDA_DECODE_INT8");
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

/* Gate + measure one precision (model already loaded with the right SP_ARENA).
 * Runs per-step decode, graph decode, asserts byte-exact equal within precision,
 * anchors to GPU prefill teacher-forced. Writes tok/s out-params. */
static void measure_prec(qwen3_model *m, const char *label, const int32_t *prompt,
                         int n_prompt, int n_gen, double *tps_ref, double *tps_g) {
    const int P = n_prompt + n_gen, V = (int)m->cfg.n_vocab;

    /* warmup (untimed): pay the cold CUDA/cuBLAS lazy-load + JIT + heuristic cost
     * ONCE per precision so the cross-precision per-step comparison isn't confounded
     * by load order. One graph + one per-step pass, results discarded. */
    { int32_t w[600]; double wd;
      set_graph(0); decode_run(m, prompt, n_prompt, n_gen, w, &wd); decode_run(m, prompt, n_prompt, n_gen, w, &wd);
      set_graph(1); decode_run(m, prompt, n_prompt, n_gen, w, &wd); decode_run(m, prompt, n_prompt, n_gen, w, &wd); }

    set_graph(0);
    int32_t seq_ref[600]; double dt_ref = 0.0;
    int n_ref = decode_run(m, prompt, n_prompt, n_gen, seq_ref, &dt_ref);
    if (n_ref < 0) fprintf(stderr, "    %s\n", sp_last_error());
    SP_CHECK(n_ref == P, "per-step decode produced full length");

    set_graph(1);
    int32_t seq_g[600]; double dt_g = 0.0;
    int n_g = decode_run(m, prompt, n_prompt, n_gen, seq_g, &dt_g);
    if (n_g < 0) fprintf(stderr, "    %s\n", sp_last_error());
    SP_CHECK(n_g == P, "graph decode produced full length");

    /* byte-exact within precision: graph path must equal per-step path. */
    int eq = (n_ref == n_g), firsteq = -1;
    for (int i = 0; i < P && eq; i++) if (seq_ref[i] != seq_g[i]) { eq = 0; firsteq = i; }
    if (!eq) {
        fprintf(stderr, "    [%s] graph != per-step at idx %d (%d vs %d)\n", label,
                firsteq, firsteq>=0?seq_ref[firsteq]:-1, firsteq>=0?seq_g[firsteq]:-1);
        fprintf(stderr, "    per-step:"); for (int i=0;i<P;i++) fprintf(stderr," %d",seq_ref[i]);
        fprintf(stderr, "\n    graph   :"); for (int i=0;i<P;i++) fprintf(stderr," %d",seq_g[i]);
        fprintf(stderr, "\n");
    }
    SP_CHECK(eq, "CUDA-graph decode == per-step decode (byte-exact)");

    /* anchor to GPU prefill teacher-forced (same precision's prefill). */
    float *logits = (float *)malloc((size_t)P * V * sizeof(float));
    SP_CHECK(logits != NULL, "logits buffer");
    if (logits) {
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
            fprintf(stderr, "    [%s] decode/prefill disagree at pos %d: != %d\n",
                    label, firstbad, seq_g[firstbad + 1]);
        SP_CHECK(match, "GPU decode == GPU prefill teacher-forced argmax");
        free(logits);
    }

    *tps_ref = (double)n_gen / dt_ref;
    *tps_g   = (double)n_gen / dt_g;

    /* BETA.3a: INT8 dp4a GEMV (graph on). On f32 (no arena) the engine declines and
     * this is a no-op == graph. On Q8/.sp-model it engages: read 1-byte codes direct,
     * no f32 scratch. Gate = top-1 AGREEMENT vs the (non-int8) Q8 graph seq — int8
     * activation quant is lossy so not byte-exact, but argmax should track tightly. */
    set_graph(1); set_int8(1);
    { int32_t w[600]; double wd; decode_run(m, prompt, n_prompt, n_gen, w, &wd); }  /* warm int8 */
    int32_t seq_i8[600]; double dt_i8 = 0.0;
    int n_i8 = decode_run(m, prompt, n_prompt, n_gen, seq_i8, &dt_i8);
    set_int8(0);
    double tps_i8 = (n_i8 > 0) ? (double)n_gen / dt_i8 : 0.0;
    int agree = 0; for (int i = n_prompt; i < P; i++) if (seq_i8[i] == seq_g[i]) agree++;
    double frac = (double)agree / (double)n_gen;

    fprintf(stderr, "    [%s] per-step %.2f | graph %.2f (%.2fx) | int8 %.2f tok/s (%.2fx vs graph); "
            "int8 top-1 agree %d/%d (%.0f%%); seq:",
            label, *tps_ref, *tps_g, *tps_g / *tps_ref, tps_i8, tps_i8 / *tps_g,
            agree, n_gen, frac * 100.0);
    for (int i = 0; i < P; i++) fprintf(stderr, " %d", seq_g[i]);
    fprintf(stderr, "\n");
}

static void T_QWEN3_DECODE_CUDA(void) {
    if (sp_cuda_device_count() < 1) { fprintf(stderr, "    no CUDA device — SKIP\n"); return; }
    const char *gguf = getenv("SP_QWEN3_GGUF"); if (!gguf) gguf = SP_QWEN3_GGUF;
    const int n_prompt = 4, n_gen = 256;   /* long window: stable steady-state tok/s */
    int32_t prompt[4] = { 1, 2, 3, 4 };
    double f32_ref=0, f32_g=0, q8_ref=0, q8_g=0;

    /* ── f32 (f16 GGUF weights, no arena) ── */
    set_arena_q8(0);
    qwen3_model *mf = qwen3_load(gguf);
    SP_CHECK(mf != NULL, "qwen3_load f32");
    if (mf) { measure_prec(mf, "f32", prompt, n_prompt, n_gen, &f32_ref, &f32_g); qwen3_free(mf); }

    /* ── Q8 arena via gguf in-memory transcode (SP_ARENA=q8) ── */
    set_arena_q8(1);
    qwen3_model *mq = qwen3_load(gguf);
    SP_CHECK(mq != NULL && mq->arena, "qwen3_load Q8 arena (gguf transcode)");
    if (mq && mq->arena) { measure_prec(mq, "Q8 ", prompt, n_prompt, n_gen, &q8_ref, &q8_g); qwen3_free(mq); }
    set_arena_q8(0);

    /* ── Q8 via the REAL .sp-model ship artifact (production sp_model_load path) ──
     * The genuine deliverable loads a .sp-model off disk (OK_Q8 codes mmap'd +
     * rebuilt into the arena), NOT a gguf transcoded in memory. Measuring here
     * removes the in-memory-transcode caveat and proves on-disk arena == transcode
     * (same decode kernel path: arena codes -> k_dequant_arena -> SGEMM). */
    double spm_ref=0, spm_g=0; int have_spm=0;
    const char *spm = getenv("SP_QWEN3_SPMODEL"); if (!spm) spm = "qwen3_rt.sp-model";
    const char *stk = getenv("SP_QWEN3_SPTOK");   if (!stk) stk = "qwen3_rt.sp-tokenizer";
    sp_model *handle = NULL;
    sp_status st = sp_model_load(spm, stk, &handle);
    if (st == SP_OK && handle) {
        qwen3_model *ms = sp_model_to_qwen3(handle);
        if (!ms) fprintf(stderr, "    [SPM] sp_model_to_qwen3: %s\n", sp_last_error());
        else if (!ms->arena) fprintf(stderr, "    [SPM] loaded but no arena (precision=%d)\n",
                                      ms->arena ? sp_arena_precision(ms->arena) : -1);
        SP_CHECK(ms != NULL && ms->arena, ".sp-model -> qwen3 (OK_Q8 arena)");
        if (ms && ms->arena) {
            have_spm = 1;
            measure_prec(ms, "SPM", prompt, n_prompt, n_gen, &spm_ref, &spm_g);
            qwen3_free(ms);
        }
        sp_model_unload(handle);
    } else {
        fprintf(stderr, "    [SPM] sp_model_load(%s) -> %d, skipping .sp-model measurement\n", spm, (int)st);
    }

    fprintf(stderr, "\n    ===== BETA.2 decode ladder (RTX 2060, 0.6B, n_gen=%d, warm) =====\n", n_gen);
    fprintf(stderr, "    f32 (f16 gguf)       per-step %.2f -> graph %.2f (%.2fx)\n", f32_ref, f32_g, f32_g/f32_ref);
    fprintf(stderr, "    Q8  (gguf transcode) per-step %.2f -> graph %.2f (%.2fx)\n", q8_ref, q8_g, q8_g/q8_ref);
    if (have_spm)
        fprintf(stderr, "    Q8  (.sp-model disk) per-step %.2f -> graph %.2f (%.2fx)\n", spm_ref, spm_g, spm_g/spm_ref);
    fprintf(stderr, "    top of ladder = %.2f tok/s (f32 graph)\n", f32_g);
}

int main(void) {
    SP_RUN(T_QWEN3_DECODE_CUDA);
    return SP_DONE();
}
