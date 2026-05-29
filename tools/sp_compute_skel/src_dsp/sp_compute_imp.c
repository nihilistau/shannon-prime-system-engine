/* sp_compute_imp.c — §3-HX Sprint D DSP-side HVX axpby kernel.
 *
 * Auto-vectorized by hexagon-clang for V69 HVX (-mhvx -mhvx-length=128B
 * already in COMMON_FLAGS per SDK hexagon_toolchain.cmake:218 for toolv87).
 * Expect HVX instructions in the SASS: V vmpy(Vh, Rh):sat, Vh = vasr(Vh, Rh):sat,
 * Vh = vadd(Vh, Vh):sat, Vh = vmin(Vh, Vh) / Vh = vmax(Vh, Vh) for clamps.
 *
 * Bitwise equivalent to scalar reference (see sp_dsp_smoke axpby tests).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "HAP_farf.h"
#include "sp_compute.h"

int sp_compute_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    void *tptr = malloc(1);
    *handle = (remote_handle64)tptr;
    assert(*handle);
    return 0;
}

int sp_compute_close(remote_handle64 handle) {
    if (handle) free((void *)handle);
    return 0;
}

/* y[i] = saturate_i16((a * x[i] + b) >> q_bits)
 *
 * NOTE: parameters are typed `int` (32-bit on Hexagon) via the IDL `long`
 * mapping; we use them as int32 explicitly inside.  Buffer Len params
 * are byte counts; each element is 2 bytes (int16_t LE).
 */
int sp_compute_axpby(remote_handle64 h,
                     int n, int a, int b, int q_bits,
                     const unsigned char *x_buf, int x_bufLen,
                     unsigned char       *y_buf, int y_bufLen)
{
    (void)h;
    if (n < 0 || q_bits < 0 || q_bits > 30) return -1;
    if (x_buf == 0 || y_buf == 0)           return -1;
    if (x_bufLen < n * 2 || y_bufLen < n * 2) return -1;

    const int16_t *x = (const int16_t *)x_buf;
    int16_t       *y = (int16_t *)      y_buf;

    /* Bring the runtime-variable shift into the inner body as scalar (loop-
     * invariant); branchless saturate via select.  Auto-vec is best-effort;
     * if hexagon-clang doesn't emit HVX (Vh) for this pattern, the scalar
     * loop is still correct.  Sprint D's "vectorized" sub-tag is conditional
     * on actually seeing Vh in the SASS — if absent, that's a finding for
     * Sprint E (explicit HVX intrinsics). */
    for (int i = 0; i < n; i++) {
        int32_t acc = ((int32_t)a * (int32_t)x[i] + b) >> q_bits;
        int32_t hi = acc < 32767 ? acc : 32767;
        int32_t lo = hi > -32768 ? hi  : -32768;
        y[i] = (int16_t)lo;
    }

    FARF(RUNTIME_HIGH, "sp_compute_axpby: n=%d a=%d b=%d q_bits=%d", n, a, b, q_bits);
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * sp_compute_scale_i16 — §3-HX Sprint D HVX-vectorized i16 scale
 *
 *   y[i] = saturate_i16(x[i] + a_h)
 *
 * Uses HVX intrinsics from <hexagon_types.h> for HVX_Vector type.  hexagon-clang
 * recognizes the `+` operator on HVX_Vector (i16 lanes) and emits
 * `Vh = vadd(Vh, Vh):sat` HVX SASS.
 *
 * Each HVX_Vector holds 128 bytes = 64 int16 elements.  We process whole
 * vectors per inner-loop iter; the tail (n % 64) handled scalar.
 * ───────────────────────────────────────────────────────────────────────── */
#include <hexagon_types.h>

int sp_compute_scale_i16(remote_handle64 h,
                         int n, int a_h,
                         const unsigned char *x_buf, int x_bufLen,
                         unsigned char       *y_buf, int y_bufLen)
{
    (void)h;
    if (n < 0)                                return -1;
    if (x_buf == 0 || y_buf == 0)             return -1;
    if (x_bufLen < n * 2 || y_bufLen < n * 2) return -1;
    if (a_h < -32768 || a_h > 32767)          return -1;

    const int16_t *x = (const int16_t *)x_buf;
    int16_t       *y = (int16_t *)      y_buf;

    /* Broadcast a_h to a full 64-lane HVX_Vector of i16.  The runtime
     * helper Q6_V_vsplat_R splats a 32-bit constant; we replicate a_h
     * into both halves of the 32-bit word so all 64 lanes get the same
     * i16 value. */
    int32_t a_splat = ((int32_t)(uint16_t)a_h << 16) | (uint16_t)a_h;
    HVX_Vector va = Q6_V_vsplat_R(a_splat);

    int vec_blocks = n / 64;
    const HVX_Vector *xv = (const HVX_Vector *)x;
    HVX_Vector       *yv = (HVX_Vector *)      y;

    for (int i = 0; i < vec_blocks; i++) {
        /* Saturating signed-16 add across both operand vectors.
         * Q6_Vh_vadd_VhVh_sat is the canonical HVX i16 saturating add. */
        yv[i] = Q6_Vh_vadd_VhVh_sat(xv[i], va);
    }

    /* Scalar tail */
    for (int i = vec_blocks * 64; i < n; i++) {
        int32_t s = (int32_t)x[i] + a_h;
        if (s >  32767) s =  32767;
        if (s < -32768) s = -32768;
        y[i] = (int16_t)s;
    }

    FARF(RUNTIME_HIGH, "sp_compute_scale_i16: n=%d a_h=%d vec_blocks=%d", n, a_h, vec_blocks);
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * sp_compute_axpby_hvx — §3-HX Sprint E F1, explicit HVX intrinsics
 *
 *   y[i] = saturate_i16((a_h * x[i] + b) >> q_bits)
 *
 * HVX chain per 64-element block (1 HVX_Vector of i16):
 *   1. Load x_vec
 *   2. Q6_Ww_vmpy_VhRh: x_vec × (a_h, a_h) as i32 pair (widening i16×i16)
 *   3. Q6_V_vsplat_R(b) and Q6_Vw_vadd_VwVw on both halves of the pair
 *   4. Q6_Vw_vasr_VwR by q_bits on both halves
 *   5. Q6_Vh_vpack_VwVw_sat to pack i32×2 → i16 (saturating)
 *   6. Store
 *
 * Scalar tail handles n % 64.
 * ───────────────────────────────────────────────────────────────────────── */
int sp_compute_axpby_hvx(remote_handle64 h,
                         int n, int a_h, int b, int q_bits,
                         const unsigned char *x_buf, int x_bufLen,
                         unsigned char       *y_buf, int y_bufLen)
{
    (void)h;
    if (n < 0)                                return -1;
    if (a_h < -32768 || a_h > 32767)          return -1;
    if (q_bits < 0 || q_bits > 30)            return -1;
    if (x_buf == 0 || y_buf == 0)             return -1;
    if (x_bufLen < n * 2 || y_bufLen < n * 2) return -1;

    const int16_t *x = (const int16_t *)x_buf;
    int16_t       *y = (int16_t *)      y_buf;

    /* Pack a_h into both halves of a 32-bit word for Q6_Ww_vmpy_VhRh:
     * the scalar Rt has two i16 lanes; the kernel multiplies x_vec's even
     * i16 lanes by the low half and odd lanes by the high half.  We want
     * the SAME a_h applied to all lanes, so duplicate.  Cast through u16
     * first to avoid sign-extending the low half into the upper bits. */
    int32_t a_dup = ((int32_t)(uint16_t)(int16_t)a_h << 16) | (uint16_t)(int16_t)a_h;

    HVX_Vector vb = Q6_V_vsplat_R(b);

    int vec_blocks = n / 64;
    const HVX_Vector *xv = (const HVX_Vector *)x;
    HVX_Vector       *yv = (HVX_Vector *)      y;

    for (int i = 0; i < vec_blocks; i++) {
        HVX_Vector     x_vec = xv[i];
        HVX_VectorPair ax    = Q6_Ww_vmpy_VhRh(x_vec, a_dup);

        /* vmpyh decomposes lanes: lo half holds even-input-lane products
         * (x[0],x[2],...,x[62]); hi half holds odd-input-lane products
         * (x[1],x[3],...,x[63]).  After +b/>>q_bits we INTERLEAVE+saturate
         * back to natural i16 lane order via vsatwh. */
        HVX_Vector lo  = Q6_V_lo_W(ax);
        HVX_Vector hi  = Q6_V_hi_W(ax);
        HVX_Vector slo = Q6_Vw_vadd_VwVw(lo, vb);
        HVX_Vector shi = Q6_Vw_vadd_VwVw(hi, vb);

        HVX_Vector rlo = Q6_Vw_vasr_VwR(slo, q_bits);
        HVX_Vector rhi = Q6_Vw_vasr_VwR(shi, q_bits);

        /* Q6_Vh_vsat_VwVw(Vu, Vv) = vsatwh: saturate-and-interleave.
         * Output: Vd.h[2k] = sat(Vv.w[k]); Vd.h[2k+1] = sat(Vu.w[k]).
         * With Vu=rhi (odd-lane results) and Vv=rlo (even-lane results),
         * output[2k]=rlo[k]=x[2k]*a result, output[2k+1]=rhi[k]=x[2k+1]*a.
         * vpackwh_sat (used previously) concatenates rather than interleaves
         * — wrong shuffle for vmpyh's pair layout. */
        yv[i] = Q6_Vh_vsat_VwVw(rhi, rlo);
    }

    /* Scalar tail */
    for (int i = vec_blocks * 64; i < n; i++) {
        int32_t acc = ((int32_t)a_h * (int32_t)x[i] + b) >> q_bits;
        if (acc >  32767) acc =  32767;
        if (acc < -32768) acc = -32768;
        y[i] = (int16_t)acc;
    }

    FARF(RUNTIME_HIGH, "sp_compute_axpby_hvx: n=%d a_h=%d b=%d q_bits=%d", n, a_h, b, q_bits);
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * sp_compute_scale_i16_batched — §3-HX Sprint E F2, amortized FastRPC overhead
 *
 * Runs scale_i16 logic n_batches times back-to-back in one FastRPC call.
 * Reuses the inner HVX intrinsic body from sp_compute_scale_i16.
 * ───────────────────────────────────────────────────────────────────────── */
static void scale_i16_inner(int n, int16_t a_h,
                            const int16_t *x, int16_t *y)
{
    int32_t a_splat = ((int32_t)(uint16_t)a_h << 16) | (uint16_t)a_h;
    HVX_Vector va = Q6_V_vsplat_R(a_splat);
    int vec_blocks = n / 64;
    const HVX_Vector *xv = (const HVX_Vector *)x;
    HVX_Vector       *yv = (HVX_Vector *)      y;
    for (int i = 0; i < vec_blocks; i++) {
        yv[i] = Q6_Vh_vadd_VhVh_sat(xv[i], va);
    }
    for (int i = vec_blocks * 64; i < n; i++) {
        int32_t s = (int32_t)x[i] + a_h;
        if (s >  32767) s =  32767;
        if (s < -32768) s = -32768;
        y[i] = (int16_t)s;
    }
}

int sp_compute_scale_i16_batched(remote_handle64 h,
                                 int n_per_batch, int n_batches,
                                 const unsigned char *a_h_buf, int a_h_bufLen,
                                 const unsigned char *x_buf,   int x_bufLen,
                                 unsigned char       *y_buf,   int y_bufLen)
{
    (void)h;
    if (n_per_batch < 0 || n_batches < 0)        return -1;
    if (a_h_buf == 0 || x_buf == 0 || y_buf == 0) return -1;
    int total = n_per_batch * n_batches;
    if (a_h_bufLen < n_batches * 2)              return -1;
    if (x_bufLen   < total * 2 || y_bufLen < total * 2) return -1;

    const int16_t *a_h_arr = (const int16_t *)a_h_buf;
    const int16_t *x       = (const int16_t *)x_buf;
    int16_t       *y       = (int16_t *)      y_buf;

    for (int b = 0; b < n_batches; b++) {
        scale_i16_inner(n_per_batch, a_h_arr[b],
                        x + b * n_per_batch,
                        y + b * n_per_batch);
    }

    FARF(RUNTIME_HIGH, "sp_compute_scale_i16_batched: n_per=%d batches=%d",
         n_per_batch, n_batches);
    return 0;
}
