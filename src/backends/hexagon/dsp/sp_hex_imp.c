/* sp_hex_imp.c — Phase 2-HX cDSP-side implementation (Hexagon V69 HTP).
 *
 * Recreated fresh. The S22U reference + SDK examples are structural reference
 * only — no code copied; the forward-pass logic comes from the engine
 * (gemma3.c / forward.c / cuda_forward.cu), recreated for HVX in HX.3.
 *
 * HX.2: open/close + ping (FastRPC wiring smoke). The skel (sp_hex_skel.c) is
 * generated from ../inc/sp_hex.idl by qaic and dispatches to these.
 *
 * V69 HVX rules for HX.3 (do NOT rediscover — see SESSION-STATE-lat-2-HX):
 *   - V69 has NO sf-result float multiply/add: the float multiply/add ALWAYS emit
 *     32-bit qfloat (qf32), per the V69 HVX Programmer's Reference (Multiply/Add
 *     single precision vector by vector, p.150/246). The sf family is NOT broken --
 *     sf inputs feed Q6_Vqf32_vmpy_VsfVsf directly. The mandated matmul shape is:
 *     sf inputs -> qf32 products -> qf32 accumulate (Q6_Vqf32_vadd_*) -> a single
 *     Q6_Vsf_equals_Vqf32 convert at the end. The retired "BROKEN, off 4-20 absolute"
 *     note was a qf32 result emitted without that final convert -- a missing
 *     conversion mislabeled a silicon defect. -mhvx-ieee-fp (dsp/CMakeLists.txt) is
 *     the codegen prerequisite the prior fixed-point HVX build never carried.
 *   - qurt_hvx_lock(QURT_HVX_MODE_128B) is thread-local; FastRPC runs the method
 *     on a worker thread — lock ONCE at the top of the forward method (whole-
 *     forward-on-DSP = one method), not in open.
 *   - 128-byte-align stack arrays fed to HVX; DSP malloc unreliable on unsigned
 *     PD — use stack / rpcmem.
 */
#include <stdlib.h>
#include <stdint.h>          /* uintptr_t for hx_rsum cache pointer hash (HX.3b-alpha-v2) */
#include <math.h>
#include "HAP_farf.h"
#include "sp_hex.h"
#include "../sp_hex_layout.h"   /* host<->DSP weight-blob contract */

#ifdef __HVX__   /* HX.3b f32 HVX matmul primitive (-mhvx-ieee-fp enables the float family) */
#include <hexagon_types.h>     /* HVX_Vector */
#include <hexagon_protos.h>    /* Q6_* HVX intrinsics */
#include "qurt_hvx.h"          /* qurt_hvx_lock/unlock + QURT_HVX_MODE_128B (gotcha #4) */
#include "qurt_thread.h"       /* V3: qurt_thread_create + attr API for dual-HVX-context worker pool */
#include "qurt_futex.h"        /* V3: qurt_futex_wait/wake — signal-wait between handler + workers */
#include "HAP_perf.h"          /* V3: HAP_perf_get_pcycles for T_BOTH_HVX_ACTIVE instrumentation */
#include "HAP_vtcm_mgr.h"      /* V4: HAP_request_VTCM / HAP_release_VTCM / HAP_query_total_VTCM */
#include <stdatomic.h>         /* V3: atomic_uint / atomic_load / atomic_store for worker signalling */
#include <string.h>            /* V3: memcpy for per-thread activation-quant buffer staging */
/* f32 dot on V69 HVX, the hardware-mandated float shape (see the header note):
 * sf inputs -> Q6_Vqf32_vmpy_VsfVsf products (V69 float multiply emits qf32) ->
 * qf32 lane accumulate -> a 5-step vror/qf32-add tree reduces the 32 lanes to lane
 * 0 -> one Q6_Vsf_equals_Vqf32 convert. The vector body runs whole 32-float
 * (128-byte) blocks; a scalar epilogue takes the n%32 tail. The aligned vector loads
 * assume 128-byte-aligned a,b -- true for page-aligned rpcmem tensor rows whose cols
 * is a multiple of 32 (the Gemma3/Qwen3 projection dims are). The lane-reduction
 * sums in a different order than the sequential scalar path, so it tracks the scalar
 * reference only to the 8.6.1 precision floor, never bit-for-bit. */
static float hx_dot_hvx(const float *a, const float *b, int n) {
    int nb = n & ~31;                 /* whole 32-float (128-byte) blocks */
    float sum = 0.0f;
    if (nb > 0) {
        /* first block initialises the qf32 accumulator (avoids relying on a
         * canonical "zero qf32" bit pattern). */
        HVX_Vector acc = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)a,
                                              *(const HVX_Vector *)b);
        for (int i = 32; i < nb; i += 32)
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(
                      acc, Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)(a + i),
                                                *(const HVX_Vector *)(b + i)));
        /* horizontal reduce qf32 lanes: rotate right by 64/32/16/8/4 bytes + add. */
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 64));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 32));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 16));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 8));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 4));
        float lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = Q6_Vsf_equals_Vqf32(acc);   /* qf32 -> IEEE single */
        sum = lanes[0];
    }
    for (int i = nb; i < n; i++) sum += a[i] * b[i];       /* scalar tail */
    return sum;
}
/* Q8 dot: int8 codes (row) against f32 activations (x), same qf32 shape as
 * hx_dot_hvx with a tiny scalar int8->sf widen per 32-chunk into a 128-byte-aligned
 * tile (V69 exposes no in-vector int8->sf convert here; the widen is cheap next to
 * the HVX multiply-accumulate it feeds — the in-vector-widen / integer-vrmpy fast
 * path is the deferred acceleration). Returns the raw dot; caller applies the row
 * scale. The HVX unit is held by the enclosing method (sp_hex_forward), not here. */
static float hx_dot_q8_hvx(const signed char *row, const float *x, int n) {
    int nb = n & ~31;
    float sum = 0.0f;
    if (nb > 0) {
        float wf[32] __attribute__((aligned(128)));
        HVX_Vector acc; int first = 1;
        for (int i = 0; i < nb; i += 32) {
            for (int c = 0; c < 32; c++) wf[c] = (float)row[i + c];   /* scalar widen */
            HVX_Vector p = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)wf,
                                                *(const HVX_Vector *)(x + i));
            acc = first ? p : Q6_Vqf32_vadd_Vqf32Vqf32(acc, p); first = 0;
        }
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 64));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 32));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 16));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 8));
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 4));
        float lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = Q6_Vsf_equals_Vqf32(acc);
        sum = lanes[0];
    }
    for (int i = nb; i < n; i++) sum += (float)row[i] * x[i];   /* scalar tail */
    return sum;
}

/* ── HX.3b-α: vrmpy-based Q8 matmul kernel ───────────────────────────────────
 *
 * Uses Q6_Vw_vrmpy_VubVb (V69 HVX) — per HVX_Vector: 32 int32 lanes, each lane
 * accumulates 4 ubyte*byte products. Algorithm:
 *
 *   1. Per matmul invocation, compute per-tensor activation scale:
 *        S_act = max(|x|) / 127     (s ≈ "Frobenius scale on activations")
 *      Quantize all activations to uint8 in [0..255] with bias-128:
 *        act_ub[i] = clamp(round(x[i] / S_act), -127, 127) + 128
 *      The bias-128 trick lets us use the unsigned×signed vrmpy intrinsic
 *      with naturally-signed weight bytes; correction term is subtracted later.
 *
 *   2. Per output row j:
 *      Compute two simultaneous vrmpy reductions across the n-element row:
 *        dot_b    = Σ_i (act_ub[i])     * weight[j][i]         via Q6_Vw_vrmpy
 *        wsum_b   = Σ_i (1, 1, 1, 1)    * weight[j][i]         via Q6_Vw_vrmpy(splat_1_ub, w)
 *      After horizontal reduction (one sum-of-32-int32-lanes per row):
 *        dot_b       = Σ_i (act_int8[i] + 128) * weight[j][i]
 *        wsum_b      = Σ_i weight[j][i]
 *        true_int_dot = dot_b - 128 * wsum_b
 *                     = Σ_i act_int8[i] * weight[j][i]
 *
 *   3. Reconstruct f32: Y[j] = true_int_dot * (S_act * row_scale[j] / 127)
 *
 * Bounds check: |true_int_dot| ≤ n * 127 * 127. For Gemma3-1B's largest in-dim
 * (FF=6912), max ≈ 1.11e8 — well within int32 (2.1e9). No saturation risk in
 * int32 accumulator across the vrmpy lanes (each lane sums n/4 products of
 * magnitude ≤ 255*127 = 32385, so per-lane max ≈ 32385 * (6912/4) = 5.6e7,
 * also safe).
 *
 * Throughput vs hx_dot_q8_hvx:
 *  - Replaces 32-element scalar widen loop (32 fp casts) with 0 widens
 *  - Replaces qf32_vmpy (1 op) + qf32_vadd (1 op) with vrmpy_acc (1 op)
 *  - One additional vrmpy per row to compute wsum_b (cheap; could also be
 *    precomputed at host-pack time as a follow-on optimization)
 *  - Same 5-step horizontal reduce (but on int32 lanes; uses vadd_VwVw, not
 *    qf32_vadd, and ends with a scalar lane[0] extract — no qf32→sf convert)
 *
 * Activation quant cost: one pass over `n` f32→ub conversions per matmul,
 * amortized across `out` rows (typically n × out = 1152 × 1152 = 1.33M
 * accesses per quant, vs 1.33M × 7 matmuls per layer × 26 layers in the
 * matmul body itself; quant is ~1% of total inner-loop work).
 *
 * Determinism: rounding differs from qf32 (int8 saturation vs ULP-float
 * accumulation), so logit bits will NOT match qf32 path. The decode-determinism
 * gate is argmax equality, NOT bit equality of logits. Per
 * reference-lattice-decode-determinism: discrete-substrate cross-backend
 * determinism CAN hold under argmax if scale calibration is sufficient.
 * If argmax diverges, the per-tensor activation scale (currently inferred from
 * |x|_max per-call) may need per-token calibration. Closure documents the gate
 * disposition either way.
 * ────────────────────────────────────────────────────────────────────────── */

/* Horizontal sum of 32 int32 lanes via 5-step vror+vadd reduction. */
static inline int32_t hx_hsum_w(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    int32_t lanes[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = v;
    return lanes[0];
}

/* Quantize n f32 activations into n uint8 bias-128 codes. Returns S_act.
 * `act_ub` must be 128-byte-aligned and at least sp_hex_align(n) bytes (the
 * caller zeros the tail to 0 = -128 which contributes -128*w to dot — but
 * we round-trip: act_ub[i] = 128 for i >= n means (act_int8=0)*w = 0, so
 * tail bytes must be 128, not 0).  We initialise the full padded length to 128. */
static float hx_quant_act_ub(const float *x, int n, unsigned char *act_ub) {
    float maxabs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = x[i]; if (a < 0) a = -a;
        if (a > maxabs) maxabs = a;
    }
    float S = maxabs / 127.0f;
    if (S == 0.0f) S = 1.0f;   /* avoid div-by-zero on all-zero activations; codes all 128 */
    float inv = 127.0f / maxabs;
    if (maxabs == 0.0f) inv = 0.0f;
    int padded_n = (n + 127) & ~127;     /* round up to 128-byte boundary */
    for (int i = 0; i < n; i++) {
        float v = x[i] * inv;
        int q = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
        if (q > 127) q = 127; if (q < -127) q = -127;
        act_ub[i] = (unsigned char)(q + 128);
    }
    for (int i = n; i < padded_n; i++) act_ub[i] = 128;   /* tail = 0 in signed */
    return S;
}

/* vrmpy-based Q8 matmul: Y[t*out + j] = scale[j]/127 * dot(codes[j], x[t]).
 * `codes` is contiguous int8 row-major [out, in] (in must be multiple of 4 — true
 * for Gemma3 dims 1152/6912). `scales` is per-row f32, length `out`.
 *
 * This is the HX.3b-α replacement for hx_matmul_q8 (qf32 path). Gated at the
 * call site by SP_HEX_VRMPY_MATMUL — see hx_matmul_q8 dispatch logic below.
 *
 * Activation buffer act_ub_scratch must be caller-supplied (size = padded
 * per-token, reused across t loop). For now we stack-allocate inside the function
 * with a max bound of 8192 elements (covers Gemma3-1B's largest in-dim of 6912
 * for ffn_down and embed dim 1152 for the others).  */
#define SP_HEX_VRMPY_MAX_IN  8192
static void hx_matmul_q8_vrmpy(const unsigned char *blk, int out, int in,
                               const float *X, int n_tok, float *Y) {
    const signed char *codes = (const signed char *)blk;
    const float *scales = (const float *)(blk + sp_hex_align((size_t)out * in));

    /* Per-token activation quant buffer. 128-byte-aligned for vmem. */
    static unsigned char act_ub[SP_HEX_VRMPY_MAX_IN] __attribute__((aligned(128)));
    int nb = in & ~127;            /* whole 128-byte (32-lane) vrmpy blocks */

    for (int t = 0; t < n_tok; t++) {
        const float *x = X + (size_t)t * in;
        float S_act = hx_quant_act_ub(x, in, act_ub);

        for (int j = 0; j < out; j++) {
            const signed char *row = codes + (size_t)j * in;

            HVX_Vector acc_dot = Q6_V_vzero();
            HVX_Vector acc_ws  = Q6_V_vzero();
            HVX_Vector v_ones  = Q6_V_vsplat_R(0x01010101);

            for (int i = 0; i < nb; i += 128) {
                /* vmem loads (rpcmem pointers are page-aligned; row + i is in-line
                 * 128B-aligned since codes is 128B-aligned and i is multiple of 128). */
                HVX_Vector w_v   = *(const HVX_Vector *)(row    + i);
                HVX_Vector act_v = *(const HVX_Vector *)(act_ub + i);
                /* int32-lane vrmpy_acc:
                 *   acc_dot[lane] += sum_{4 bytes} act_ub[lane*4+k] * w[lane*4+k] */
                acc_dot = Q6_Vw_vrmpyacc_VwVubVb(acc_dot, act_v, w_v);
                /* wsum_b[lane] += sum_{4 bytes} 1 * w[lane*4+k] */
                acc_ws  = Q6_Vw_vrmpyacc_VwVubVb(acc_ws,  v_ones, w_v);
            }
            int32_t dot_b = hx_hsum_w(acc_dot);
            int32_t ws_b  = hx_hsum_w(acc_ws);

            /* Scalar tail (in % 128). Same arithmetic as in-vector path. */
            for (int i = nb; i < in; i++) {
                int a = (int)act_ub[i];         /* in [0..255] */
                int w = (int)row[i];            /* in [-127..127] */
                dot_b += a * w;
                ws_b  += w;
            }

            int32_t true_dot = dot_b - 128 * ws_b;
            float y = (float)true_dot * (S_act * scales[j] / 127.0f);
            Y[(size_t)t * out + j] = y;
        }
    }
}

/* HX.3b-alpha-v2: per-weight-block row_sum cache.
 *
 * The single-vrmpy kernel (hx_matmul_q8_vrmpy_v2 below) replaces the per-call
 * wsum vrmpy + hsum with a lookup `rsum[j]`. Ideally `rsum` would be packed
 * into the weight blob by the host at sp_hex_host.c::hx_pack_q8 time. That
 * would require a coordinated rebuild of the daemon binary. Instead, we
 * populate the cache lazily on the FIRST forward call: hx_rsum_get returns
 * a pointer to a 16-byte-aligned int32_t[out] table; on cache miss it walks
 * the int8 codes once and fills the table, on hit it returns the cached ptr.
 *
 * Cache keying: the (blk, out, in) tuple uniquely identifies a weight tensor
 * during a session (the host's weight blob is rpcmem-allocated once per
 * model load and the per-layer pointers are stable). We index by `blk`
 * pointer hash. Capacity 256 = headroom over Gemma3-1B's 182 tensors
 * (26 layers * 7 Q8 weights).
 *
 * Numerical equivalence: cache fill computes
 *   rsum[j] = Sigma_{i=0..in-1} (int32) (int8)codes[j*in + i]
 * — identical int8 bytes, identical index range, identical int32 accumulator
 * as the per-call wsum in HX.3b. Therefore `dot_b - 128 * rsum[j]` is bit-
 * identical to the prior `dot_b - 128 * ws_b`, y is identical, argmax is
 * identical. Bit-exact decode preserved by construction.
 *
 * Lifetime: the cache survives across sp_hex_forward calls; entries are
 * freed by hx_rsum_clear (called only from sp_hex_close).
 */
#define HX_RSUM_CACHE_CAP 256
typedef struct {
    const unsigned char *key;   /* blk pointer; NULL = empty slot */
    int                  out;
    int                  in;
    int32_t             *vals;  /* malloc'd [out] */
} hx_rsum_entry;
static hx_rsum_entry g_hx_rsum_cache[HX_RSUM_CACHE_CAP] = {{0}};

static inline unsigned hx_ptr_hash(const unsigned char *p) {
    /* FNV-1a-ish, 32-bit input from pointer low/high words mixed. */
    uintptr_t v = (uintptr_t)p;
    unsigned h = 2166136261u ^ (unsigned)v;
    h = (h ^ (unsigned)(v >> 16)) * 16777619u;
    h ^= (h >> 13);
    return h;
}

/* Return cached rsum[out] for this weight block; fill cache on miss.
 * Returns NULL only on malloc failure (fall back to v1 path at caller). */
static const int32_t *hx_rsum_get(const unsigned char *blk, int out, int in) {
    unsigned h = hx_ptr_hash(blk);
    /* linear probe from h mod cap */
    for (unsigned step = 0; step < HX_RSUM_CACHE_CAP; step++) {
        unsigned idx = (h + step) & (HX_RSUM_CACHE_CAP - 1);
        hx_rsum_entry *e = &g_hx_rsum_cache[idx];
        if (e->key == blk && e->out == out && e->in == in) {
            return e->vals;   /* hit */
        }
        if (e->key == NULL) {
            /* miss — populate */
            int32_t *vals = (int32_t *)malloc((size_t)out * sizeof(int32_t));
            if (!vals) return NULL;
            const signed char *codes = (const signed char *)blk;
            for (int j = 0; j < out; j++) {
                int32_t s = 0;
                const signed char *row = codes + (size_t)j * in;
                for (int i = 0; i < in; i++) s += (int32_t)row[i];
                vals[j] = s;
            }
            e->key = blk; e->out = out; e->in = in; e->vals = vals;
            return vals;
        }
        /* collision; continue probe */
    }
    return NULL;   /* table full — should not happen with cap 256 vs 182 tensors */
}

static void hx_rsum_clear(void) {
    for (int i = 0; i < HX_RSUM_CACHE_CAP; i++) {
        if (g_hx_rsum_cache[i].vals) free(g_hx_rsum_cache[i].vals);
        g_hx_rsum_cache[i].key = NULL;
        g_hx_rsum_cache[i].vals = NULL;
    }
}

/* HX.3b-alpha-v2 kernel: single-vrmpy inner loop + post-loop bias-correction
 * via cached rsum[]. On the first call for a given weight block, hx_rsum_get
 * fills the cache (~O(out*in) int8 sum); subsequent calls are O(1) lookup
 * per output row. Falls back to hx_matmul_q8_vrmpy on cache-fill failure. */
static void hx_matmul_q8_vrmpy_v2(const unsigned char *blk, int out, int in,
                                  const float *X, int n_tok, float *Y) {
    const int32_t *rsum = hx_rsum_get(blk, out, in);
    if (!rsum) {   /* malloc failure — degrade gracefully to v1 path */
        hx_matmul_q8_vrmpy(blk, out, in, X, n_tok, Y);
        return;
    }
    const signed char *codes = (const signed char *)blk;
    const float *scales = (const float *)(blk + sp_hex_align((size_t)out * in));

    static unsigned char act_ub[SP_HEX_VRMPY_MAX_IN] __attribute__((aligned(128)));
    int nb = in & ~127;

    for (int t = 0; t < n_tok; t++) {
        const float *x = X + (size_t)t * in;
        float S_act = hx_quant_act_ub(x, in, act_ub);

        for (int j = 0; j < out; j++) {
            const signed char *row = codes + (size_t)j * in;

            HVX_Vector acc_dot = Q6_V_vzero();   /* single accumulator now */

            for (int i = 0; i < nb; i += 128) {
                HVX_Vector w_v   = *(const HVX_Vector *)(row    + i);
                HVX_Vector act_v = *(const HVX_Vector *)(act_ub + i);
                acc_dot = Q6_Vw_vrmpyacc_VwVubVb(acc_dot, act_v, w_v);
            }
            int32_t dot_b = hx_hsum_w(acc_dot);   /* one hsum, not two */

            /* Scalar tail (in % 128). Only the dot accumulator runs here;
             * tail weights are already counted in rsum[j] from cache fill. */
            for (int i = nb; i < in; i++) {
                int a = (int)act_ub[i];
                int w = (int)row[i];
                dot_b += a * w;
            }

            int32_t true_dot = dot_b - 128 * rsum[j];   /* O(1) lookup */
            float y = (float)true_dot * (S_act * scales[j] / 127.0f);
            Y[(size_t)t * out + j] = y;
        }
    }
}

/* ────────────────────────────────────────────────────────────────────────────
 * V3 (TRICK-1-FORWARD-V3): dual-HVX-context per-matmul via in-skel QURT
 * worker pool.
 *
 * Reference primitives:
 *   - K v0.alpha (tools/sp_dsp_smoke/sprint_k_alpha_run_output.txt:14-44):
 *     two ARM threads + Arc<FastRpcSession> + per-thread invoke = 1.935x
 *     speedup at 128x128 compute-bound matmul. The cDSP scheduler engaged
 *     SSR:XA={4,5} dual vector context attachment automatically.
 *   - reference-v69-hvx-expert-practices: V69 has 4 scalar threads / 2 HVX
 *     vector contexts. qurt_hvx_lock is thread-local; each thread that
 *     calls it gets attached to one of the SSR:XA={4,5} contexts.
 *   - reference-dual-model-cdsp-scheduler: the SSR:XA mechanism is
 *     kernel-agnostic — it triggers on "two threads both want HVX",
 *     regardless of whether the two threads came from FastRPC concurrent
 *     dispatch (K v0.alpha) or in-skel qurt_thread_create (V3).
 *   - llama.cpp's worker_pool.c (htp/worker-pool.c:113-153) is the
 *     existence proof for in-skel qurt_thread_create + futex signal-wait;
 *     V3 mirrors the pattern modulo Unsigned-PD constraints.
 *
 * V3 architecture:
 *   - At first matmul call (lazy init inside sp_hex_forward) the handler
 *     thread spawns ONE worker thread via qurt_thread_create. The worker
 *     calls qurt_hvx_lock(QURT_HVX_MODE_128B) — distinct from the handler's
 *     hvx_lock — and the QURT scheduler attaches it to the OTHER SSR:XA
 *     context (4 if handler is on 5, vice versa).
 *   - Per-matmul descriptor passed via shared struct; handler signals
 *     worker via atomic seqno + futex_wake; worker signals completion
 *     via n_pending decrement + futex_wake.
 *   - Output-row split: worker computes rows [0, M/2); handler computes
 *     rows [M/2, M). Both consume the same activation buffer + weight
 *     blob; disjoint output writes.
 *   - Worker freed in sp_hex_close via killed flag.
 *
 * Why one worker not two: the handler thread is itself one of the
 *   parallel compute threads (it has its own HVX context). Adding ONE
 *   worker gives us 2 concurrent HVX contexts on V69 — the maximum.
 *   A two-worker pool plus the handler would be 3 HVX clients for 2
 *   contexts; one would block on hvx_lock.
 *
 * Per-thread activation buffer: HX.3b's `static unsigned char act_ub[]`
 *   is shared / not thread-safe. V3 each thread gets its own buffer in
 *   its own context struct.
 *
 * Memory-bandwidth-bound risk (reference-v69-vrmpy-chat-shape-memory-bound):
 *   At Gemma3-1B chat shape, the inner-loop is bandwidth-bound. Dual-
 *   context parallel execution may yield only 1.0x-1.2x wall-clock lift
 *   because both contexts contend for DDR/L1 bandwidth. PERF_LIFT gate
 *   is structurally at-risk at this shape; PERF_PARITY should hold.
 * ──────────────────────────────────────────────────────────────────── */

/* Per-thread activation-quant buffer + per-thread pcycle counter. */
typedef struct {
    unsigned char act_ub[SP_HEX_VRMPY_MAX_IN] __attribute__((aligned(128)));
    uint64_t      pcyc_start;
    uint64_t      pcyc_end;
} hx_worker_local_t;

/* Per-matmul descriptor — single producer (handler), single consumer (worker). */
typedef struct {
    const unsigned char *blk;     /* weight blob */
    int                  out;     /* total output rows (M); worker does [0, m_half) */
    int                  m_half;  /* split point — worker rows [0, m_half), handler [m_half, out) */
    int                  in_dim;  /* input dim K */
    const float         *X;       /* activations [n_tok * in_dim] */
    int                  n_tok;
    float               *Y;       /* output [n_tok * out] */
    const int32_t       *rsum;    /* cached weight-row-sum for bias-128 correction */
} hx_matmul_desc_t;

/* Worker pool: single worker thread + main handler thread = 2 HVX clients = 2 SSR:XA contexts. */
typedef struct {
    int                init_done;   /* 1 after qurt_thread_create succeeded; 0 = use single-ctx fallback */
    int                init_error;  /* AEEResult-style error code from last init attempt */
    qurt_thread_t      worker_tid;
    void              *worker_stack;
    atomic_uint        seqn;        /* producer increments; worker waits on this */
    atomic_uint        done;        /* worker increments on completion; handler waits */
    atomic_uint        killed;      /* 1 = worker should exit */
    hx_matmul_desc_t   desc;        /* current job descriptor */
    hx_worker_local_t  worker_local;  /* worker's per-thread buffer + pcycles */
    hx_worker_local_t  handler_local; /* handler's per-thread buffer + pcycles */
} hx_worker_pool_t;

static hx_worker_pool_t g_hx_pool = {0};

#define HX_WORKER_STACK_SZ 32768  /* 32 KB — matches llama.cpp's 2*16384 baseline */

/* Inner kernel — half-matmul. Computes Y[t*out + j_start..j_end) for all tokens.
 * Identical arithmetic to hx_matmul_q8_vrmpy_v2 per-row but limited to a row range,
 * and uses caller-supplied per-thread activation buffer. */
static void hx_matmul_q8_vrmpy_half(const unsigned char *blk, int out, int in_dim,
                                    const float *X, int n_tok, float *Y,
                                    const int32_t *rsum,
                                    int j_start, int j_end,
                                    unsigned char *act_ub) {
    const signed char *codes  = (const signed char *)blk;
    const float       *scales = (const float *)(blk + sp_hex_align((size_t)out * in_dim));
    int nb = in_dim & ~127;

    for (int t = 0; t < n_tok; t++) {
        const float *x = X + (size_t)t * in_dim;
        float S_act = hx_quant_act_ub(x, in_dim, act_ub);
        for (int j = j_start; j < j_end; j++) {
            const signed char *row = codes + (size_t)j * in_dim;
            HVX_Vector acc_dot = Q6_V_vzero();
            for (int i = 0; i < nb; i += 128) {
                HVX_Vector w_v   = *(const HVX_Vector *)(row    + i);
                HVX_Vector act_v = *(const HVX_Vector *)(act_ub + i);
                acc_dot = Q6_Vw_vrmpyacc_VwVubVb(acc_dot, act_v, w_v);
            }
            int32_t dot_b = hx_hsum_w(acc_dot);
            for (int i = nb; i < in_dim; i++) {
                int a = (int)act_ub[i];
                int w = (int)row[i];
                dot_b += a * w;
            }
            int32_t true_dot = dot_b - 128 * rsum[j];
            float y = (float)true_dot * (S_act * scales[j] / 127.0f);
            Y[(size_t)t * out + j] = y;
        }
    }
}

/* Worker thread entry. Acquires its own HVX context (separate SSR:XA from
 * handler), spins waiting for job-seqno bumps, processes its half of each
 * matmul, signals done. */
static void hx_worker_main(void *arg) {
    (void)arg;
    /* Worker's HVX lock — distinct call from handler's; QURT scheduler
     * attaches this thread to the OTHER SSR:XA context (e.g. handler on 5,
     * worker on 4 on V69). If this fails (returns nonzero) under
     * Unsigned PD, surface UPSTREAM at runtime via FARF + abort. */
    int hr = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (hr != 0) {
        FARF(ERROR, "sp_hex V3: worker qurt_hvx_lock FAILED rc=%d (Unsigned PD limitation?)", hr);
        /* mark init_error and exit — handler will fall back to single-ctx path */
        g_hx_pool.init_error = hr ? hr : -1;
        atomic_store(&g_hx_pool.done, 1);  /* unblock any pending handler join */
        return;
    }
    FARF(RUNTIME_HIGH, "sp_hex V3: worker thread started, qurt_hvx_lock OK");

    unsigned int prev_seqn = 0;
    while (!atomic_load(&g_hx_pool.killed)) {
        unsigned int seqn = atomic_load(&g_hx_pool.seqn);
        if (seqn == prev_seqn) {
            qurt_futex_wait(&g_hx_pool.seqn, prev_seqn);
            continue;
        }
        prev_seqn = seqn;
        if (atomic_load(&g_hx_pool.killed)) break;

        /* Execute our half of the matmul. */
        const hx_matmul_desc_t *d = &g_hx_pool.desc;
        g_hx_pool.worker_local.pcyc_start = HAP_perf_get_pcycles();
        hx_matmul_q8_vrmpy_half(d->blk, d->out, d->in_dim,
                                d->X, d->n_tok, d->Y, d->rsum,
                                0, d->m_half,
                                g_hx_pool.worker_local.act_ub);
        g_hx_pool.worker_local.pcyc_end = HAP_perf_get_pcycles();

        atomic_fetch_add(&g_hx_pool.done, 1);
        qurt_futex_wake(&g_hx_pool.done, 1);
    }
    qurt_hvx_unlock();
    FARF(RUNTIME_HIGH, "sp_hex V3: worker thread exiting");
}

/* Lazy init: spawn worker once. Called from sp_hex_forward on first matmul.
 * Returns 0 on success; nonzero = init failure (caller must fall back). */
static int hx_worker_pool_ensure(void) {
    if (g_hx_pool.init_done) return 0;
    if (g_hx_pool.init_error) return g_hx_pool.init_error;  /* permanent fail; don't retry */

    /* Allocate worker stack */
    g_hx_pool.worker_stack = malloc(HX_WORKER_STACK_SZ);
    if (!g_hx_pool.worker_stack) {
        g_hx_pool.init_error = -1;
        FARF(ERROR, "sp_hex V3: worker stack malloc failed");
        return -1;
    }
    atomic_store(&g_hx_pool.seqn,   0);
    atomic_store(&g_hx_pool.done,   0);
    atomic_store(&g_hx_pool.killed, 0);

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, g_hx_pool.worker_stack);
    qurt_thread_attr_set_stack_size(&attr, HX_WORKER_STACK_SZ);
    qurt_thread_attr_set_name(&attr, "sp_hex_v3_worker");
    /* Inherit handler's priority (same prio so neither preempts the other). */
    int prio = qurt_thread_get_priority(qurt_thread_get_id());
    if (prio < 1) prio = 1; if (prio > 254) prio = 254;
    qurt_thread_attr_set_priority(&attr, prio);

    int rc = qurt_thread_create(&g_hx_pool.worker_tid, &attr, hx_worker_main, NULL);
    if (rc != 0) {
        free(g_hx_pool.worker_stack); g_hx_pool.worker_stack = NULL;
        g_hx_pool.init_error = rc;
        FARF(ERROR, "sp_hex V3: qurt_thread_create FAILED rc=%d", rc);
        return rc;
    }
    g_hx_pool.init_done = 1;
    FARF(RUNTIME_HIGH, "sp_hex V3: worker pool initialized (tid=%u)", (unsigned)g_hx_pool.worker_tid);
    return 0;
}

/* Tear down worker at sp_hex_close. */
static void hx_worker_pool_shutdown(void) {
    if (!g_hx_pool.init_done) return;
    atomic_store(&g_hx_pool.killed, 1);
    atomic_fetch_add(&g_hx_pool.seqn, 1);
    qurt_futex_wake(&g_hx_pool.seqn, 1);
    int status = 0;
    qurt_thread_join((unsigned)g_hx_pool.worker_tid, &status);
    if (g_hx_pool.worker_stack) free(g_hx_pool.worker_stack);
    g_hx_pool.worker_stack = NULL;
    g_hx_pool.init_done    = 0;
    g_hx_pool.init_error   = 0;
    FARF(RUNTIME_HIGH, "sp_hex V3: worker pool shutdown complete (status=%d)", status);
}

/* Dispatch ONE matmul through the dual-context path. The handler thread
 * (caller) computes its half [m_half, out) concurrently with the worker's
 * half [0, m_half). Both threads call qurt_hvx_lock (worker did at start;
 * handler holds the existing top-of-sp_hex_forward lock). On V69 the QURT
 * scheduler attaches the two lock-holders to SSR:XA={4,5} respectively. */
static int hx_matmul_q8_vrmpy_dual_ctx(const unsigned char *blk, int out, int in_dim,
                                       const float *X, int n_tok, float *Y) {
    /* Try lazy init; on failure, fall through to single-ctx path. */
    if (hx_worker_pool_ensure() != 0) {
        hx_matmul_q8_vrmpy_v2(blk, out, in_dim, X, n_tok, Y);
        return 1;  /* single-ctx fallback used */
    }

    const int32_t *rsum = hx_rsum_get(blk, out, in_dim);
    if (!rsum) {
        /* malloc failure for rsum cache — single-ctx fallback path. */
        hx_matmul_q8_vrmpy_v2(blk, out, in_dim, X, n_tok, Y);
        return 1;
    }

    /* Even-M split: worker [0, M/2), handler [M/2, M). Ceiling division
     * to handle odd-M shapes (Gemma3-1B has all even M but be safe). */
    int m_half = (out + 1) / 2;

    g_hx_pool.desc.blk     = blk;
    g_hx_pool.desc.out     = out;
    g_hx_pool.desc.m_half  = m_half;
    g_hx_pool.desc.in_dim  = in_dim;
    g_hx_pool.desc.X       = X;
    g_hx_pool.desc.n_tok   = n_tok;
    g_hx_pool.desc.Y       = Y;
    g_hx_pool.desc.rsum    = rsum;

    /* Reset done counter, bump seqno, wake worker. */
    atomic_store(&g_hx_pool.done, 0);
    atomic_fetch_add(&g_hx_pool.seqn, 1);
    qurt_futex_wake(&g_hx_pool.seqn, 1);

    /* Handler's half concurrently. */
    g_hx_pool.handler_local.pcyc_start = HAP_perf_get_pcycles();
    hx_matmul_q8_vrmpy_half(blk, out, in_dim, X, n_tok, Y, rsum,
                            m_half, out,
                            g_hx_pool.handler_local.act_ub);
    g_hx_pool.handler_local.pcyc_end = HAP_perf_get_pcycles();

    /* Wait for worker to complete its half. */
    while (atomic_load(&g_hx_pool.done) == 0) {
        qurt_futex_wait(&g_hx_pool.done, 0);
    }

    /* T_TRICK1FWDV3_BOTH_HVX_ACTIVE evidence (sampled). Log the FIRST matmul
     * per session via a static one-shot so logs aren't flooded. */
    static int sampled_once = 0;
    if (!sampled_once) {
        sampled_once = 1;
        uint64_t wpc = g_hx_pool.worker_local.pcyc_end  - g_hx_pool.worker_local.pcyc_start;
        uint64_t hpc = g_hx_pool.handler_local.pcyc_end - g_hx_pool.handler_local.pcyc_start;
        FARF(RUNTIME_HIGH, "sp_hex V3: dual_ctx matmul out=%d in=%d n_tok=%d "
                           "worker_pcyc=%llu handler_pcyc=%llu m_half=%d",
             out, in_dim, n_tok,
             (unsigned long long)wpc, (unsigned long long)hpc, m_half);
    }
    return 0;
}

/* ────────────────────────────────────────────────────────────────────────────
 * V4 (TRICK-1-FORWARD-V4): VTCM weight pinning.
 *
 * V3's substrate (worker pool + dual-HVX-context per matmul) is silicon-
 * validated but perf-flat at Gemma3-1B chat shape due to memory-bandwidth
 * contention (reference-v69-vrmpy-chat-shape-memory-bound, 3rd confirmation).
 * V4 attacks the bandwidth by staging the active layer's attention weight
 * set (WQ + WK + WV + WO ≈ 2.85 MB) in V69's 8 MB VTCM at layer entry.
 *
 * Budget reality (per PLAN-TRICK-1-FORWARD-V4.md §D-A):
 *   Per-layer Q8 weight total ≈ 25.7 MB; attention (Q+K+V+O) ≈ 2.85 MB;
 *   each FFN tensor (WGATE/WUP/WDOWN) ≈ 7.6 MB. 8 MB VTCM cannot hold even
 *   ONE FFN tensor alongside attention. HYBRID strategy: pin attention per
 *   layer, leave FFN in DDR (Stage 1-3). FFN tile-streaming is the named
 *   Stage 4 stretch / V5 follow-on.
 *
 * Lifecycle:
 *   - sp_hex_open: no-op (cfg not yet known).
 *   - first sp_hex_forward call: lazy-init the VTCM allocation sized for
 *     the model's attention set (max-over-layers — all 26 Gemma3-1B
 *     layers have the same attention shape so single sizing suffices).
 *   - per-layer in forward: if g_hx_vtcm.cached_layer != L, memcpy the
 *     attention sub-blob from DDR into VTCM; update cached_layer.
 *   - sp_hex_close: HAP_release_VTCM if allocated.
 *
 * Robustness:
 *   - HAP_request_VTCM may fail (other PD holding VTCM, allocator denial);
 *     g_hx_vtcm.vtcm_base stays NULL; kernel falls back to V3 DDR path.
 *   - No regression on VTCM-unavailable devices.
 *
 * Cache coherency:
 *   VTCM is cDSP-private; ARM never reads it. The DDR-resident weights
 *   were registered via rpcmem (host-flushed); the cDSP-internal memcpy
 *   from DDR to VTCM doesn't need DMA_BUF sync (rpcmem handles host-side
 *   flush at registration time).
 * ──────────────────────────────────────────────────────────────────── */

typedef struct {
    void    *vtcm_base;       /* HAP_request_VTCM return value (NULL = not allocated) */
    uint32_t vtcm_size;       /* size requested (bytes) */
    int      cached_layer;    /* layer index whose attention weights are in VTCM; -1 = empty */
    /* Per-tensor offsets within vtcm_base — match the DDR sub-block layout. */
    uint32_t off_wq;          /* offset to WQ block start (always 0) */
    uint32_t off_wk;
    uint32_t off_wv;
    uint32_t off_wo;
    uint32_t bytes_wq;        /* sp_hex_q8_bytes(QD, E) */
    uint32_t bytes_wk;
    uint32_t bytes_wv;
    uint32_t bytes_wo;
    int      ddr_path_count;  /* number of attention matmuls that fell back to DDR (V3 path); FARF-logged */
    int      vtcm_path_count; /* number that used VTCM; FARF-logged */
    sp_hex_cfg cfg;           /* cached cfg for sub-offset math */

    /* V4 Stage 2+: per-layer per-attention-tensor rsum tables. The DDR-side
     * hx_rsum_get cache keys on the DDR pointer; we can't reuse it for VTCM
     * pointers (which alias across layers — same VTCM addr, different
     * content per layer, would produce stale cache hits). Instead we hold
     * one rsum table per (layer, attention-tensor) pair, populated lazily
     * at the same time the weights are memcpy'd into VTCM (we already
     * touch every byte during memcpy, the rsum is a free side-effect).
     *
     * Sizing per Gemma3-1B: 26 layers × (QD + KVD + KVD + E) ints
     *                     = 26 × (1024+256+256+1152) × 4 B ≈ 280 KB total DDR.
     * Allocated lazily on hx_vtcm_init via malloc. Indices are flat:
     *   rsum_attn[L * (QD+KVD+KVD+E) + 0..QD)        — WQ
     *   rsum_attn[L * (QD+KVD+KVD+E) + QD..QD+KVD)   — WK
     *   ... (WV, WO offsets follow). */
    int32_t *rsum_attn;       /* malloc'd; NULL = unallocated; freed in hx_vtcm_release */
    uint32_t rsum_attn_n_layers;
    uint32_t rsum_attn_stride;        /* ints per layer = QD + KVD + KVD + E */
    uint8_t *rsum_attn_layer_ready;   /* malloc'd [n_layers] bool — 1 = rsum populated for this layer */
} hx_vtcm_t;

static hx_vtcm_t g_hx_vtcm = { NULL, 0u, -1, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0, 0, {0},
                               NULL, 0u, 0u, NULL };

/* Per-attention-tensor rsum offsets within one layer's rsum_attn stripe. */
static inline uint32_t hx_vtcm_rsum_off_wq(const sp_hex_cfg *c) { (void)c; return 0u; }
static inline uint32_t hx_vtcm_rsum_off_wk(const sp_hex_cfg *c) { return (uint32_t)(c->n_head * c->head_dim); }
static inline uint32_t hx_vtcm_rsum_off_wv(const sp_hex_cfg *c) { return (uint32_t)(c->n_head * c->head_dim + c->n_head_kv * c->head_dim); }
static inline uint32_t hx_vtcm_rsum_off_wo(const sp_hex_cfg *c) { return (uint32_t)(c->n_head * c->head_dim + 2 * c->n_head_kv * c->head_dim); }
static inline uint32_t hx_vtcm_rsum_stride(const sp_hex_cfg *c) { return (uint32_t)(c->n_head * c->head_dim + 2 * c->n_head_kv * c->head_dim + c->n_embd); }

/* Compute the byte size of one layer's attention weight set (Q+K+V+O).
 * Q/K/V/O are contiguous Q8 blocks in the DDR layout (SP_HEX_WQ..SP_HEX_WO,
 * indices 6..9 per sp_hex_layout.h). Each block is sp_hex_align()'d so the
 * sum is also aligned. */
static uint32_t hx_vtcm_attn_set_bytes(const sp_hex_cfg *cfg) {
    size_t bytes = 0;
    bytes += sp_hex_kind_bytes(cfg, SP_HEX_WQ);
    bytes += sp_hex_kind_bytes(cfg, SP_HEX_WK);
    bytes += sp_hex_kind_bytes(cfg, SP_HEX_WV);
    bytes += sp_hex_kind_bytes(cfg, SP_HEX_WO);
    return (uint32_t)bytes;
}

/* Lazy-init: allocate the VTCM region once (sized for the model's attention
 * set). Idempotent — second call returns 0 if already allocated.
 * Returns 0 on success, nonzero on failure (caller falls back to DDR path). */
static int hx_vtcm_init(const sp_hex_cfg *cfg) {
    if (g_hx_vtcm.vtcm_base != NULL) return 0;   /* already alloc'd */

    /* Query the cDSP's VTCM budget for diagnostic. */
    unsigned int page_size = 0u, page_count = 0u;
    if (HAP_query_total_VTCM(&page_size, &page_count) == 0) {
        FARF(RUNTIME_HIGH, "sp_hex V4: VTCM total page_size=%u page_count=%u total=%u",
             page_size, page_count, page_size * page_count);
    }
    unsigned int avail_block = 0u, max_page = 0u, num_pages = 0u;
    if (HAP_query_avail_VTCM(&avail_block, &max_page, &num_pages) == 0) {
        FARF(RUNTIME_HIGH, "sp_hex V4: VTCM avail block=%u max_page=%u num_pages=%u",
             avail_block, max_page, num_pages);
    }

    uint32_t bytes = hx_vtcm_attn_set_bytes(cfg);
    /* Allocate with single_page_flag=0 (multi-page OK; we don't need
     * scatter/gather single-page constraint for plain vmem reads). */
    void *p = HAP_request_VTCM(bytes, 0u);
    if (!p) {
        FARF(ERROR, "sp_hex V4: HAP_request_VTCM(%u, 0) FAILED -- falling back to DDR-only path",
             bytes);
        return -1;
    }

    /* Stash offsets: sub-blocks are laid out sequentially matching the DDR
     * blob order WQ, WK, WV, WO. Same byte-stride as the DDR layout, so
     * within-block sub-offsets (codes + scales) are byte-identical to the
     * DDR version once we memcpy. */
    g_hx_vtcm.vtcm_base = p;
    g_hx_vtcm.vtcm_size = bytes;
    g_hx_vtcm.cached_layer = -1;
    g_hx_vtcm.cfg = *cfg;
    g_hx_vtcm.bytes_wq = (uint32_t)sp_hex_kind_bytes(cfg, SP_HEX_WQ);
    g_hx_vtcm.bytes_wk = (uint32_t)sp_hex_kind_bytes(cfg, SP_HEX_WK);
    g_hx_vtcm.bytes_wv = (uint32_t)sp_hex_kind_bytes(cfg, SP_HEX_WV);
    g_hx_vtcm.bytes_wo = (uint32_t)sp_hex_kind_bytes(cfg, SP_HEX_WO);
    g_hx_vtcm.off_wq = 0u;
    g_hx_vtcm.off_wk = g_hx_vtcm.off_wq + g_hx_vtcm.bytes_wq;
    g_hx_vtcm.off_wv = g_hx_vtcm.off_wk + g_hx_vtcm.bytes_wk;
    g_hx_vtcm.off_wo = g_hx_vtcm.off_wv + g_hx_vtcm.bytes_wv;
    g_hx_vtcm.ddr_path_count = 0;
    g_hx_vtcm.vtcm_path_count = 0;

    /* V4 Stage 2: allocate per-layer per-attention-tensor rsum tables (DDR).
     * Compute at memcpy time (Stage 1 already touched every byte; rsum is
     * a free side-effect). Storing per-layer (vs the global hx_rsum_get
     * pointer-keyed cache) avoids stale-content false-hits when the same
     * VTCM address aliases different layer content. */
    uint32_t stride = hx_vtcm_rsum_stride(cfg);
    uint32_t n_layers = (uint32_t)cfg->n_layers;
    size_t rsum_bytes = (size_t)n_layers * stride * sizeof(int32_t);
    g_hx_vtcm.rsum_attn = (int32_t *)malloc(rsum_bytes);
    g_hx_vtcm.rsum_attn_layer_ready = (uint8_t *)malloc(n_layers);
    if (!g_hx_vtcm.rsum_attn || !g_hx_vtcm.rsum_attn_layer_ready) {
        FARF(ERROR, "sp_hex V4: rsum_attn malloc FAILED (rsum=%p ready=%p sz=%zu)",
             g_hx_vtcm.rsum_attn, g_hx_vtcm.rsum_attn_layer_ready, rsum_bytes);
        if (g_hx_vtcm.rsum_attn)              free(g_hx_vtcm.rsum_attn);
        if (g_hx_vtcm.rsum_attn_layer_ready)  free(g_hx_vtcm.rsum_attn_layer_ready);
        g_hx_vtcm.rsum_attn = NULL;
        g_hx_vtcm.rsum_attn_layer_ready = NULL;
        HAP_release_VTCM(p);
        g_hx_vtcm.vtcm_base = NULL;
        g_hx_vtcm.vtcm_size = 0u;
        return -1;
    }
    memset(g_hx_vtcm.rsum_attn_layer_ready, 0, n_layers);
    g_hx_vtcm.rsum_attn_n_layers = n_layers;
    g_hx_vtcm.rsum_attn_stride = stride;

    FARF(RUNTIME_HIGH, "sp_hex V4: VTCM allocated base=%p size=%u "
                       "WQ@%u(%u) WK@%u(%u) WV@%u(%u) WO@%u(%u) rsum_attn=%zu B",
         p, bytes,
         g_hx_vtcm.off_wq, g_hx_vtcm.bytes_wq,
         g_hx_vtcm.off_wk, g_hx_vtcm.bytes_wk,
         g_hx_vtcm.off_wv, g_hx_vtcm.bytes_wv,
         g_hx_vtcm.off_wo, g_hx_vtcm.bytes_wo,
         rsum_bytes);
    return 0;
}

/* Compute per-row int8-sum for a Q8 block. Identical arithmetic to the
 * existing hx_rsum_get cache fill (line 319-323) — same `(int32_t)(int8)code`
 * accumulation, same per-row scope. Used by hx_vtcm_ensure_layer to populate
 * the per-layer rsum_attn table at memcpy time.
 *
 * Reads from `blk_ddr` (the DDR weight pointer; we have it in our hand at
 * memcpy time anyway). Writes to `out_rsum[out]`. */
static void hx_compute_rsum(const unsigned char *blk_ddr, int out, int in_dim,
                            int32_t *out_rsum) {
    const signed char *codes = (const signed char *)blk_ddr;
    for (int j = 0; j < out; j++) {
        int32_t s = 0;
        const signed char *row = codes + (size_t)j * in_dim;
        for (int i = 0; i < in_dim; i++) s += (int32_t)row[i];
        out_rsum[j] = s;
    }
}

/* Ensure VTCM holds layer L's attention weights. If cache miss, memcpy
 * Q + K + V + O from DDR-resident `weights + sp_hex_weight_off(cfg, L, WQ)`
 * into the VTCM region (single contiguous copy since WQ..WO are adjacent
 * in the blob layout). Returns the VTCM base pointer (== g_hx_vtcm.vtcm_base)
 * or NULL if VTCM unavailable (caller uses DDR path). */
static void *hx_vtcm_ensure_layer(int L, const unsigned char *weights,
                                  const sp_hex_cfg *cfg) {
    if (hx_vtcm_init(cfg) != 0) return NULL;          /* alloc failure path */
    if (g_hx_vtcm.cached_layer == L) return g_hx_vtcm.vtcm_base;  /* already cached */

    /* Compute DDR source: start of WQ for layer L. WQ..WO are contiguous. */
    size_t src_off = sp_hex_weight_off(cfg, L, SP_HEX_WQ);
    const unsigned char *src = weights + src_off;
    uint32_t total = g_hx_vtcm.bytes_wq + g_hx_vtcm.bytes_wk
                   + g_hx_vtcm.bytes_wv + g_hx_vtcm.bytes_wo;

    /* memcpy DDR -> VTCM. cDSP memcpy uses scalar loads/stores; bandwidth
     * is DDR-bound (~10 GB/s class) → ~290 μs for 2.85 MB attention set on
     * Gemma3-1B. Amortized over 7 matmuls × 16 tokens of work in the layer:
     * negligible vs the bandwidth saved on attention matmul inner loops. */
    memcpy(g_hx_vtcm.vtcm_base, src, total);
    g_hx_vtcm.cached_layer = L;

    /* V4 Stage 2: populate per-layer rsum_attn table if not yet ready for
     * this layer. We just touched every byte of the DDR source during
     * memcpy; computing rsum is one more pass over the same bytes (~290 μs
     * additional read cost — DDR is now cached in L2 from memcpy, so this
     * second pass is fast). Lazy + idempotent: if rsum already computed for
     * this layer (re-visit case), skip. */
    if (g_hx_vtcm.rsum_attn && g_hx_vtcm.rsum_attn_layer_ready
        && (uint32_t)L < g_hx_vtcm.rsum_attn_n_layers
        && !g_hx_vtcm.rsum_attn_layer_ready[L]) {
        int32_t *base_r = g_hx_vtcm.rsum_attn + (size_t)L * g_hx_vtcm.rsum_attn_stride;
        const int QD  = cfg->n_head * cfg->head_dim;
        const int KVD = cfg->n_head_kv * cfg->head_dim;
        const int E   = cfg->n_embd;
        /* WQ: out=QD, in=E */
        hx_compute_rsum(weights + sp_hex_weight_off(cfg, L, SP_HEX_WQ),
                        QD, E, base_r + hx_vtcm_rsum_off_wq(cfg));
        /* WK: out=KVD, in=E */
        hx_compute_rsum(weights + sp_hex_weight_off(cfg, L, SP_HEX_WK),
                        KVD, E, base_r + hx_vtcm_rsum_off_wk(cfg));
        /* WV: out=KVD, in=E */
        hx_compute_rsum(weights + sp_hex_weight_off(cfg, L, SP_HEX_WV),
                        KVD, E, base_r + hx_vtcm_rsum_off_wv(cfg));
        /* WO: out=E, in=QD */
        hx_compute_rsum(weights + sp_hex_weight_off(cfg, L, SP_HEX_WO),
                        E, QD, base_r + hx_vtcm_rsum_off_wo(cfg));
        g_hx_vtcm.rsum_attn_layer_ready[L] = 1u;
    }

    return g_hx_vtcm.vtcm_base;
}

/* Return pointer to layer L's WQ rsum table (NULL if VTCM/rsum not available
 * or layer not yet warmed). Same for WK/WV/WO. */
static const int32_t *hx_vtcm_rsum_for(int L, uint32_t off) {
    if (!g_hx_vtcm.rsum_attn || !g_hx_vtcm.rsum_attn_layer_ready) return NULL;
    if ((uint32_t)L >= g_hx_vtcm.rsum_attn_n_layers) return NULL;
    if (!g_hx_vtcm.rsum_attn_layer_ready[L]) return NULL;
    return g_hx_vtcm.rsum_attn + (size_t)L * g_hx_vtcm.rsum_attn_stride + off;
}

/* Release VTCM on session close. Called from sp_hex_close. */
static void hx_vtcm_release(void) {
    if (g_hx_vtcm.vtcm_base) {
        FARF(RUNTIME_HIGH, "sp_hex V4: VTCM release base=%p size=%u "
                           "(usage: vtcm_matmul=%d ddr_fallback=%d)",
             g_hx_vtcm.vtcm_base, g_hx_vtcm.vtcm_size,
             g_hx_vtcm.vtcm_path_count, g_hx_vtcm.ddr_path_count);
        HAP_release_VTCM(g_hx_vtcm.vtcm_base);
        g_hx_vtcm.vtcm_base = NULL;
        g_hx_vtcm.vtcm_size = 0u;
    }
    if (g_hx_vtcm.rsum_attn) {
        free(g_hx_vtcm.rsum_attn);
        g_hx_vtcm.rsum_attn = NULL;
    }
    if (g_hx_vtcm.rsum_attn_layer_ready) {
        free(g_hx_vtcm.rsum_attn_layer_ready);
        g_hx_vtcm.rsum_attn_layer_ready = NULL;
    }
    g_hx_vtcm.rsum_attn_n_layers = 0u;
    g_hx_vtcm.rsum_attn_stride = 0u;
    g_hx_vtcm.cached_layer = -1;
    g_hx_vtcm.ddr_path_count = 0;
    g_hx_vtcm.vtcm_path_count = 0;
}

/* V4 Stage 2: dual-context matmul with EXPLICIT rsum pointer + EXPLICIT blk
 * pointer (typically VTCM-resident). Used where the global hx_rsum_get
 * cache cannot be reused because the VTCM blk pointer aliases across layers
 * (same address, different content per layer = stale cache hits).
 *
 * Caller supplies blk (VTCM-resident bytes) + rsum (per-layer table from
 * rsum_attn). The half-kernel is the SAME hx_matmul_q8_vrmpy_half V3
 * silicon-validated for bit-exactness.
 *
 * On worker-pool init failure: degrade to single-thread full-row-range call
 * inline (avoids the V3 fallback path's call into hx_rsum_get which would
 * fail for our VTCM pointer). Returns 0 on dual-context success, 1 on
 * single-context fallback.
 *
 * Performance expectation: weights read from VTCM at ~256 GB/s vs DDR
 * ~10 GB/s — the inner-loop vmem load latency drops from ~30 cycles to
 * ~1 cycle. Both worker + handler vector contexts can sustain peak compute
 * because they no longer contend for DDR/L1 (the V3 chat-shape bandwidth
 * bound). */
static int hx_matmul_q8_vrmpy_dual_ctx_v4(const unsigned char *blk_vtcm, int out, int in_dim,
                                          const float *X, int n_tok, float *Y,
                                          const int32_t *rsum) {
    if (hx_worker_pool_ensure() != 0) {
        /* Worker init failed; fall through to single-thread VTCM read via
         * the same _half kernel with full row range. The handler's HVX lock
         * (held by sp_hex_forward) still gates this. */
        hx_matmul_q8_vrmpy_half(blk_vtcm, out, in_dim, X, n_tok, Y, rsum,
                                0, out, g_hx_pool.handler_local.act_ub);
        return 1;
    }

    int m_half = (out + 1) / 2;
    g_hx_pool.desc.blk     = blk_vtcm;   /* VTCM-resident weight blob */
    g_hx_pool.desc.out     = out;
    g_hx_pool.desc.m_half  = m_half;
    g_hx_pool.desc.in_dim  = in_dim;
    g_hx_pool.desc.X       = X;
    g_hx_pool.desc.n_tok   = n_tok;
    g_hx_pool.desc.Y       = Y;
    g_hx_pool.desc.rsum    = rsum;       /* per-layer rsum from rsum_attn */

    atomic_store(&g_hx_pool.done, 0);
    atomic_fetch_add(&g_hx_pool.seqn, 1);
    qurt_futex_wake(&g_hx_pool.seqn, 1);

    g_hx_pool.handler_local.pcyc_start = HAP_perf_get_pcycles();
    hx_matmul_q8_vrmpy_half(blk_vtcm, out, in_dim, X, n_tok, Y, rsum,
                            m_half, out,
                            g_hx_pool.handler_local.act_ub);
    g_hx_pool.handler_local.pcyc_end = HAP_perf_get_pcycles();

    while (atomic_load(&g_hx_pool.done) == 0) {
        qurt_futex_wait(&g_hx_pool.done, 0);
    }

    /* First-VTCM-matmul-per-session evidence sample (T_V4_DUAL_CTX_VTCM_READS).  */
    static int v4_sampled_once = 0;
    if (!v4_sampled_once) {
        v4_sampled_once = 1;
        uint64_t wpc = g_hx_pool.worker_local.pcyc_end  - g_hx_pool.worker_local.pcyc_start;
        uint64_t hpc = g_hx_pool.handler_local.pcyc_end - g_hx_pool.handler_local.pcyc_start;
        FARF(RUNTIME_HIGH, "sp_hex V4: dual_ctx_vtcm matmul out=%d in=%d n_tok=%d "
                           "worker_pcyc=%llu handler_pcyc=%llu m_half=%d blk_vtcm=%p",
             out, in_dim, n_tok,
             (unsigned long long)wpc, (unsigned long long)hpc, m_half, blk_vtcm);
    }
    g_hx_vtcm.vtcm_path_count++;
    return 0;
}

/* V4 dispatch: try VTCM path; on any unavailability fall back to V3 DDR.
 *
 * Bit-exactness contract: VTCM bytes = DDR bytes (memcpy preserves content);
 * VTCM-rsum bytes = per-row int8-sum computed identically to hx_rsum_get;
 * scales offset = sp_hex_align(out*in_dim) from blob base in both paths
 * (the memcpy preserved this layout); the half-kernel arithmetic is
 * identical (same Q6_Vw_vrmpyacc_VwVubVb / dot_b - 128*rsum[j] /
 * scale*S_act/127 reconstruction). Therefore VTCM and DDR paths produce
 * byte-identical Y[j] values. Decode argmax preserved per
 * reference-lattice-decode-determinism. */
static int hx_matmul_q8_vrmpy_dispatch(const unsigned char *blk_ddr,
                                       const unsigned char *blk_vtcm,
                                       const int32_t *rsum_vtcm,
                                       int out, int in_dim,
                                       const float *X, int n_tok, float *Y) {
    if (blk_vtcm && rsum_vtcm) {
        return hx_matmul_q8_vrmpy_dual_ctx_v4(blk_vtcm, out, in_dim, X, n_tok, Y, rsum_vtcm);
    }
    g_hx_vtcm.ddr_path_count++;
    return hx_matmul_q8_vrmpy_dual_ctx(blk_ddr, out, in_dim, X, n_tok, Y);
}

#endif /* __HVX__ */

int sp_hex_open(const char *uri, remote_handle64 *h) {
    (void)uri;
    /* Opaque handle; the rpc layer does not inspect it. HX.3 hangs the
     * uploaded-weight table + scratch off this. */
    void *ctx = malloc(1);
    *h = (remote_handle64)ctx;
    return ctx ? 0 : -1;
}

int sp_hex_close(remote_handle64 h) {
    if (h) free((void *)h);
#ifdef __HVX__
    hx_worker_pool_shutdown();   /* V3: tear down worker thread (no-op if never inited) */
    hx_vtcm_release();           /* V4: release the per-layer attention VTCM region */
    hx_rsum_clear();   /* HX.3b-alpha-v2: drop the per-block weight-sum cache */
#endif
    return 0;
}

int sp_hex_ping(remote_handle64 h, int x, int *y) {
    (void)h;
    *y = x + 1;
    FARF(RUNTIME_HIGH, "sp_hex: ping x=%d -> y=%d", x, *y);
    return 0;
}

/* Standard reflected CRC-32 (IEEE / zlib polynomial 0xEDB88320), table-less.
 * Identical algorithm host-side (sp_hex_rt.c) — equal CRC iff equal bytes. */
static unsigned int sp_hex_crc32(const unsigned char *p, int n) {
    unsigned int crc = 0xFFFFFFFFu;
    for (int i = 0; i < n; i++) {
        crc ^= (unsigned int)p[i];
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0xEDB88320u & (unsigned int)(-(int)(crc & 1u)));
    }
    return crc ^ 0xFFFFFFFFu;
}

/* HX.3a upload byte-exactness proof: CRC-32 the bytes the DSP received over
 * FastRPC. If the host's rpcmem registration size != the IDL length, the bridge
 * silently zero-fills and this CRC will not match the host's — the exact-alloc
 * trap, caught here before it can poison a forward pass. */
int sp_hex_upload_crc(remote_handle64 h, const unsigned char *data, int dataLen, int *crc) {
    (void)h;
    unsigned int c = sp_hex_crc32(data, dataLen);
    *crc = (int)c;
    FARF(RUNTIME_HIGH, "sp_hex: upload_crc len=%d crc=0x%08x", dataLen, c);
    return 0;
}

/* HX.3a: scalar f32 matmul on the cDSP — the core forward kernel (ggml weight
 * layout: y[j] = sum_i w[j*cols + i] * x[i]). Under __HVX__ each row runs the qf32
 * HVX dot (hx_dot_hvx, HX.3b); otherwise the scalar f32 path -- which the host A/B
 * still computes as the reference. The HVX lane-reduction reorders the sum, so the
 * cDSP-HVX result tracks the scalar host reference only to the 8.6.1 precision
 * floor, not bit-for-bit (the prior scalar-vs-host exact-zero match no longer holds
 * once HVX is on -- this is the expected, gated behaviour). */
int sp_hex_matmul_f32(remote_handle64 h, const float *w, int wLen, int rows, int cols,
                      const float *x, int xLen, float *y, int yLen) {
    (void)h; (void)wLen; (void)xLen;
#ifdef __HVX__
    /* Reserve the 128B HVX unit on this FastRPC worker thread (gotcha #4: the lock is
     * thread-local and FastRPC dispatches the method on a worker thread). Lock once
     * around the whole row loop, not per row. */
    qurt_hvx_lock(QURT_HVX_MODE_128B);
    for (int j = 0; j < rows && j < yLen; j++)
        y[j] = hx_dot_hvx(w + (long)j * cols, x, cols);
    qurt_hvx_unlock();
#else
    for (int j = 0; j < rows && j < yLen; j++) {
        const float *wr = w + (long)j * cols;
        float acc = 0.0f;
        for (int i = 0; i < cols; i++) acc += wr[i] * x[i];
        y[j] = acc;
    }
#endif
    FARF(RUNTIME_HIGH, "sp_hex: matmul_f32 rows=%d cols=%d", rows, cols);
    return 0;
}

/* ── HX.3a: the gemma3 transformer layers + final RMSNorm, scalar f32 on the cDSP.
 * Mirrors src/forward/gemma3.c op-for-op (recreated fresh); reads the Q8/f32
 * weight blob via sp_hex_layout.h. Embedding + tied head run host-side. ── */

static void hx_rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}
static void hx_rmsnorm_head(float *v, const float *w, int d, float eps) {
    double ss = 0.0;
    for (int i = 0; i < d; i++) ss += (double)v[i] * v[i];
    float scale = 1.0f / sqrtf((float)(ss / d) + eps);
    for (int i = 0; i < d; i++) v[i] = v[i] * scale * w[i];
}
static void hx_rope_neox(float *v, int d, int p, float base) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(base, -2.0f * (float)i / (float)d);
        float th = (float)p * freq, c = cosf(th), s = sinf(th);
        float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}
static float hx_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}
/* per-row Q8 matmul (ggml [out,in]); blob = int8 codes (padded) then f32 scales.
 * Y[t,j] = (sum_i code[j,i]*X[t,i]) * scale[j]/127 — the matmul_arena inline lift. */
static void hx_matmul_q8(const unsigned char *blk, int out, int in,
                         const float *X, int n_tok, float *Y) {
    const signed char *codes = (const signed char *)blk;
    const float *scales = (const float *)(blk + sp_hex_align((size_t)out * in));
    for (int j = 0; j < out; j++) {
        const signed char *row = codes + (size_t)j * in;
        float inv = scales[j] / 127.0f;
        for (int t = 0; t < n_tok; t++) {
            const float *x = X + (size_t)t * in;
#ifdef __HVX__
            Y[(size_t)t * out + j] = hx_dot_q8_hvx(row, x, in) * inv;
#else
            float acc = 0.0f;
            for (int i = 0; i < in; i++) acc += (float)row[i] * x[i];
            Y[(size_t)t * out + j] = acc * inv;
#endif
        }
    }
}
/* GQA causal/windowed softmax for one query head (matches kernels_attn_head). */
static void hx_attn_head(const float *qh, const float *KC, const float *VC,
                         int pos, int KVD, int kvh, int HD, float ascale, int win,
                         float *sc, float *out) {
    int s0 = (win >= 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;
    float maxs = -INFINITY;
    for (int s = s0; s <= pos; s++) {
        const float *kh = KC + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        float d = acc * ascale; sc[s] = d; if (d > maxs) maxs = d;
    }
    float sum = 0.0f;
    for (int s = s0; s <= pos; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < HD; i++) out[i] = 0.0f;
    for (int s = s0; s <= pos; s++) {
        float w = sc[s] * inv;
        const float *vh = VC + (size_t)s * KVD + (size_t)kvh * HD;
        for (int i = 0; i < HD; i++) out[i] += w * vh[i];
    }
}

int sp_hex_forward(remote_handle64 hdl, int n_layers, int n_embd, int n_ff, int head_dim,
                   int n_head, int n_head_kv, int sliding_window,
                   float eps, float rope_global, float rope_local,
                   int n_tok, const float *x, int xLen,
                   const unsigned char *weights, int weightsLen,
                   const float *scratch, int scratchLen, float *hidden, int hiddenLen) {
    (void)hdl; (void)xLen; (void)weightsLen; (void)scratchLen; (void)hiddenLen;
#ifdef __HVX__
    /* Reserve the 128B HVX unit once for the whole forward (gotcha #4: thread-local,
     * one FastRPC method = one worker thread). All hx_matmul_q8 calls below run HVX
     * dots under this lock; the scalar norm/rope/attn kernels are unaffected. */
    qurt_hvx_lock(QURT_HVX_MODE_128B);
#endif
    float *scr = (float *)scratch;   /* `in` buffer is writable host rpcmem; carve work area */
    sp_hex_cfg cfg = { n_layers, n_embd, n_ff, head_dim, n_head, n_head_kv,
                       sliding_window, eps, rope_global, rope_local };
    const int E = n_embd, FF = n_ff, HD = head_dim;
    const int NH = n_head, NKV = n_head_kv, QD = NH * HD, KVD = NKV * HD;
    const int group = NH / NKV;
    const float ascale = 1.0f / sqrtf((float)HD);

    /* carve scratch (order matches sp_hex_scratch_elems) */
    float *resid = scr;
    float *nx = resid + (size_t)n_tok * E;
    float *q  = nx + (size_t)n_tok * E;
    float *k  = q  + (size_t)n_tok * QD;
    float *v  = k  + (size_t)n_tok * KVD;
    float *ao = v  + (size_t)n_tok * KVD;
    float *ap = ao + (size_t)n_tok * QD;
    float *g  = ap + (size_t)n_tok * E;
    float *up = g  + (size_t)n_tok * FF;
    float *dn = up + (size_t)n_tok * FF;
    float *sc = dn + (size_t)n_tok * E;

    for (size_t i = 0; i < (size_t)n_tok * E; i++) resid[i] = x[i];

    for (int L = 0; L < n_layers; L++) {
        const int global = ((L % 6) == 5);
        const float rbase = global ? rope_global : rope_local;
        const int win = global ? -1 : sliding_window;
        const unsigned char *base = weights;
        #define WPTR(kind) (base + sp_hex_weight_off(&cfg, L, (kind)))
#ifdef __HVX__
        /* V4: ensure this layer's attention weight set (WQ+WK+WV+WO) is
         * staged in VTCM + populate per-layer rsum_attn table. Returns the
         * VTCM base ptr (or NULL on failure → kernels fall back to DDR). */
        unsigned char *vtcm_attn_base = (unsigned char *)hx_vtcm_ensure_layer(L, weights, &cfg);
        const unsigned char *vtcm_wq = vtcm_attn_base
                                       ? (vtcm_attn_base + g_hx_vtcm.off_wq) : NULL;
        const unsigned char *vtcm_wk = vtcm_attn_base
                                       ? (vtcm_attn_base + g_hx_vtcm.off_wk) : NULL;
        const unsigned char *vtcm_wv = vtcm_attn_base
                                       ? (vtcm_attn_base + g_hx_vtcm.off_wv) : NULL;
        const unsigned char *vtcm_wo = vtcm_attn_base
                                       ? (vtcm_attn_base + g_hx_vtcm.off_wo) : NULL;
        const int32_t *rsum_wq = hx_vtcm_rsum_for(L, hx_vtcm_rsum_off_wq(&cfg));
        const int32_t *rsum_wk = hx_vtcm_rsum_for(L, hx_vtcm_rsum_off_wk(&cfg));
        const int32_t *rsum_wv = hx_vtcm_rsum_for(L, hx_vtcm_rsum_off_wv(&cfg));
        const int32_t *rsum_wo = hx_vtcm_rsum_for(L, hx_vtcm_rsum_off_wo(&cfg));
        /* Stage 3: ALL 4 attention matmuls (WQ/WK/WV/WO) use VTCM dispatch.
         * FFN matmuls (WGATE/WUP/WDOWN) stay on V3 DDR path (Stage 4 stretch
         * = FFN tile-streaming via ping-pong VTCM tiles, deferred). */
#endif
        const float *attn_norm = (const float *)WPTR(SP_HEX_ATTN_NORM);
        const float *qn = (const float *)WPTR(SP_HEX_Q_NORM);
        const float *kn = (const float *)WPTR(SP_HEX_K_NORM);

        for (int t = 0; t < n_tok; t++)
            hx_rmsnorm(resid + (size_t)t * E, attn_norm, E, eps, nx + (size_t)t * E);
#ifdef __HVX__
        /* V4 Stage 3: all 4 attention matmuls (WQ/WK/WV/WO) use VTCM
         * dispatch with per-layer rsum_attn tables; FFN remains on V3 DDR. */
        hx_matmul_q8_vrmpy_dispatch(WPTR(SP_HEX_WQ), vtcm_wq, rsum_wq,
                                    QD,  E, nx, n_tok, q);                       /* V4 WQ */
        hx_matmul_q8_vrmpy_dispatch(WPTR(SP_HEX_WK), vtcm_wk, rsum_wk,
                                    KVD, E, nx, n_tok, k);                       /* V4 WK */
        hx_matmul_q8_vrmpy_dispatch(WPTR(SP_HEX_WV), vtcm_wv, rsum_wv,
                                    KVD, E, nx, n_tok, v);                       /* V4 WV */
#else
        hx_matmul_q8(WPTR(SP_HEX_WQ), QD,  E, nx, n_tok, q);
        hx_matmul_q8(WPTR(SP_HEX_WK), KVD, E, nx, n_tok, k);
        hx_matmul_q8(WPTR(SP_HEX_WV), KVD, E, nx, n_tok, v);
#endif
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                float *qh = q + (size_t)t * QD + (size_t)h * HD;
                hx_rmsnorm_head(qh, qn, HD, eps); hx_rope_neox(qh, HD, t, rbase);
            }
            for (int h = 0; h < NKV; h++) {
                float *kh = k + (size_t)t * KVD + (size_t)h * HD;
                hx_rmsnorm_head(kh, kn, HD, eps); hx_rope_neox(kh, HD, t, rbase);
            }
        }
        for (int t = 0; t < n_tok; t++)
            for (int h = 0; h < NH; h++)
                hx_attn_head(q + (size_t)t * QD + (size_t)h * HD, k, v, t, KVD,
                             h / group, HD, ascale, win, sc, ao + (size_t)t * QD + (size_t)h * HD);
#ifdef __HVX__
        hx_matmul_q8_vrmpy_dispatch(WPTR(SP_HEX_WO), vtcm_wo, rsum_wo,
                                    E, QD, ao, n_tok, ap);                       /* V4 WO */
#else
        hx_matmul_q8(WPTR(SP_HEX_WO), E, QD, ao, n_tok, ap);
#endif
        { const float *pn = (const float *)WPTR(SP_HEX_POST_ATTN);
          for (int t = 0; t < n_tok; t++) {
              hx_rmsnorm(ap + (size_t)t * E, pn, E, eps, nx + (size_t)t * E);
              float *xt = resid + (size_t)t * E; const float *p = nx + (size_t)t * E;
              for (int i = 0; i < E; i++) xt[i] += p[i];
          } }
        { const float *fn = (const float *)WPTR(SP_HEX_FFN_NORM);
          for (int t = 0; t < n_tok; t++)
              hx_rmsnorm(resid + (size_t)t * E, fn, E, eps, nx + (size_t)t * E); }
#ifdef __HVX__
        hx_matmul_q8_vrmpy_dual_ctx(WPTR(SP_HEX_WGATE), FF, E, nx, n_tok, g);  /* V3: WGATE dual-HVX-context */
        hx_matmul_q8_vrmpy_dual_ctx(WPTR(SP_HEX_WUP),   FF, E, nx, n_tok, up); /* V3: WUP dual-HVX-context */
#else
        hx_matmul_q8(WPTR(SP_HEX_WGATE), FF, E, nx, n_tok, g);
        hx_matmul_q8(WPTR(SP_HEX_WUP),   FF, E, nx, n_tok, up);
#endif
        for (size_t i = 0; i < (size_t)n_tok * FF; i++) g[i] = hx_gelu_tanh(g[i]) * up[i];
#ifdef __HVX__
        hx_matmul_q8_vrmpy_dual_ctx(WPTR(SP_HEX_WDOWN), E, FF, g, n_tok, dn);  /* V3: WDOWN dual-HVX-context */
#else
        hx_matmul_q8(WPTR(SP_HEX_WDOWN), E, FF, g, n_tok, dn);
#endif
        { const float *pf = (const float *)WPTR(SP_HEX_POST_FFW);
          for (int t = 0; t < n_tok; t++) {
              hx_rmsnorm(dn + (size_t)t * E, pf, E, eps, nx + (size_t)t * E);
              float *xt = resid + (size_t)t * E; const float *p = nx + (size_t)t * E;
              for (int i = 0; i < E; i++) xt[i] += p[i];
          } }
        #undef WPTR
    }

    { const float *on = (const float *)(weights + sp_hex_weight_off(&cfg, n_layers, 0));
      for (int t = 0; t < n_tok; t++)
          hx_rmsnorm(resid + (size_t)t * E, on, E, eps, hidden + (size_t)t * E); }

#ifdef __HVX__
    qurt_hvx_unlock();
#endif
    FARF(RUNTIME_HIGH, "sp_hex: forward n_tok=%d n_layers=%d done", n_tok, n_layers);
    return 0;
}
