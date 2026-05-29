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

/* ─────────────────────────────────────────────────────────────────────────
 * sp_compute_vtcm_probe — §3-HX Sprint F VTCM litmus test (PRE-Halide).
 *
 * Attempts HAP_request_VTCM at the requested size + single-page flag and
 * reports the result through a rout parameter + FARF log.  Either outcome
 * is informational — does Unsigned PD (Path B) on this device admit VTCM?
 *
 *   admitted: vtcm_addr_lo = low-32 bits of pointer (non-zero); we release
 *             before return so this probe never leaks VTCM.
 *   denied:   vtcm_addr_lo = 0.
 *
 * V69 VTCM caps at ~4 MB; caller picks size_bytes.
 * single_page_flag=0 = multi-page OK, =1 = single-page (required for HVX
 * scatter/gather).
 * ───────────────────────────────────────────────────────────────────────── */
#include "HAP_vtcm_mgr.h"

int sp_compute_vtcm_probe(remote_handle64 h,
                          int size_bytes, int single_page_flag,
                          int *vtcm_addr_lo)
{
    (void)h;
    if (size_bytes < 0 || vtcm_addr_lo == 0) return -1;

    void *p = HAP_request_VTCM((unsigned)size_bytes,
                               (unsigned)(single_page_flag ? 1 : 0));
    *vtcm_addr_lo = (int)(uintptr_t)p;

    FARF(RUNTIME_HIGH,
         "sp_compute_vtcm_probe size=%d single=%d result=%p admitted=%d",
         size_bytes, single_page_flag, p, p != 0);

    if (p) {
        int rel = HAP_release_VTCM(p);
        FARF(RUNTIME_HIGH, "sp_compute_vtcm_probe release rc=%d", rel);
    }
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * sp_compute_axpby_2d_halide — §3-HX Sprint F; Halide AOT + VTCM hot-copy.
 *
 * The Halide-generated kernel `sp_axpby_2d_halide` (see halide_gen/sp_axpby_2d_halide.h)
 * is called with halide_buffer_t descriptors for x[rows,cols], a[cols], y[rows,cols].
 *
 * VTCM litmus branching:
 *   admitted → memcpy x_buf → VTCM_x; run kernel against VTCM pointers;
 *              memcpy VTCM_y → y_buf; release VTCM.  vtcm_used=1.
 *   denied   → run kernel against the incoming DDR pointers (the FastRPC
 *              SMMU mapping makes them DSP-addressable, Sprint B's zero-copy).
 *              vtcm_used=0.
 * ───────────────────────────────────────────────────────────────────────── */
#if SP_HAVE_HALIDE
#include "sp_axpby_2d_halide.h"
#include "HalideRuntime.h"

/* Halide runtime hooks the AOT-emitted kernel expects from the host.
 * The standalone-simulator stubs.c (Examples/standalone/simulator/utils/stubs.c)
 * provides these against the simulator's printf; for in-skel use we route to
 * FARF so the messages land in adsprpc logcat. */
extern void halide_print(void *user_context, const char *str);
extern void halide_error(void *user_context, const char *msg);

void halide_print(void *user_context, const char *str) {
    (void)user_context;
    FARF(RUNTIME_HIGH, "halide_print: %s", str);
}
void halide_error(void *user_context, const char *msg) {
    (void)user_context;
    FARF(RUNTIME_ERROR, "halide_error: %s", msg);
}

/* Override Halide's weak halide_qurt_hvx_lock/unlock/unlock_as_destructor.
 *
 * Reason: the FastRPC remote thread that dispatches into our skel already
 * holds an HVX context (Sprint D's scale_i16 runs HVX without needing to
 * call qurt_hvx_lock).  Halide's wrapper calls qurt_hvx_lock(QURT_HVX_MODE_128B)
 * which on the V69 cdsp under Path B FastRPC thread crashes with a stack
 * trace at sp_axpby_2d_halide+0x2C (the call site itself), because the
 * QuRT context-switch path expects a calling thread that isn't already
 * inside an HVX-protected region.  Defining strong symbols here replaces
 * the weak ones in the Halide .o and makes the lock/unlock no-ops. */
int halide_qurt_hvx_lock(int mode) {
    (void)mode;
    return 0;
}
int halide_qurt_hvx_unlock(void) {
    return 0;
}
void halide_qurt_hvx_unlock_as_destructor(void *user_context, void *p) {
    (void)user_context; (void)p;
}

static void hbuf2_i16_init(halide_buffer_t *hb, halide_dimension_t *dims,
                           int16_t *host, int cols, int rows)
{
    hb->device = 0;
    hb->device_interface = 0;
    hb->host = (uint8_t *)host;
    hb->flags = 0;
    hb->type.code = halide_type_int;
    hb->type.bits = 16;
    hb->type.lanes = 1;
    hb->dimensions = 2;
    hb->dim = dims;
    hb->padding = 0;
    dims[0].min = 0; dims[0].extent = cols; dims[0].stride = 1;     dims[0].flags = 0;
    dims[1].min = 0; dims[1].extent = rows; dims[1].stride = cols;  dims[1].flags = 0;
}

static void hbuf1_i16_init(halide_buffer_t *hb, halide_dimension_t *dim,
                           int16_t *host, int cols)
{
    hb->device = 0;
    hb->device_interface = 0;
    hb->host = (uint8_t *)host;
    hb->flags = 0;
    hb->type.code = halide_type_int;
    hb->type.bits = 16;
    hb->type.lanes = 1;
    hb->dimensions = 1;
    hb->dim = dim;
    hb->padding = 0;
    dim->min = 0; dim->extent = cols; dim->stride = 1; dim->flags = 0;
}

int sp_compute_axpby_2d_halide(remote_handle64 h,
                               int rows, int cols, int b, int q_bits,
                               const unsigned char *a_buf, int a_bufLen,
                               const unsigned char *x_buf, int x_bufLen,
                               unsigned char       *y_buf, int y_bufLen,
                               int *vtcm_used)
{
    (void)h;
    /* cols must be ≥ Halide tile width (128) for the generator's schedule */
    if (rows < 1 || cols < 128 || (cols % 128) != 0)         return -1;
    if (q_bits < 0 || q_bits > 30)                            return -1;
    if (a_buf == 0 || x_buf == 0 || y_buf == 0 || vtcm_used == 0) return -1;
    size_t n_xy = (size_t)rows * cols * 2;
    if ((size_t)a_bufLen < (size_t)cols * 2) return -1;
    if ((size_t)x_bufLen < n_xy || (size_t)y_bufLen < n_xy) return -1;

    /* Sprint F finding (documented in SESSION-CLOSED-lat-3-hx-mode-f):
     * VTCM litmus on Path B Unsigned PD is PASS — HAP_request_VTCM admits
     * up to 4 MB (see sp_compute_vtcm_probe).  BUT the C-side "hot-copy
     * inputs to VTCM, run Halide against VTCM host pointers, copy out"
     * pattern crashes the Halide-emitted kernel: the AOT code uses `vmemu`
     * loads expecting DDR semantics, and re-pointing halide_buffer_t.host
     * at the 0xff000000 VTCM region traps inside the inner HVX loop.
     *
     * Canonical Halide+VTCM pattern is `.store_in(MemoryType::VTCM)` on an
     * intermediate Func in the generator schedule — the runtime allocates
     * VTCM internally and the input/output buffers stay on DDR.  That's a
     * Sprint G follow-on (current axpby kernel has no intermediates to
     * stage; would need an FFN-shaped pipeline first).
     *
     * For Sprint F: always use DDR for Halide; vtcm_used reports 0. */
    int16_t *x_dev = (int16_t *)x_buf;
    int16_t *y_dev = (int16_t *)y_buf;
    const int16_t *a_arr = (const int16_t *)a_buf;
    *vtcm_used = 0;

    halide_buffer_t hx, ha, hy;
    halide_dimension_t hx_dims[2], hy_dims[2], ha_dim;
    hbuf2_i16_init(&hx, hx_dims, x_dev, cols, rows);
    hbuf2_i16_init(&hy, hy_dims, y_dev, cols, rows);
    hbuf1_i16_init(&ha, &ha_dim, (int16_t *)a_arr, cols);

    int rc = sp_axpby_2d_halide(&hx, &ha, b, (uint8_t)q_bits, &hy);
    FARF(RUNTIME_HIGH,
         "axpby_2d_halide: rc=%d rows=%d cols=%d (DDR-only — see closure note)",
         rc, rows, cols);
    return rc;
}
#endif  /* SP_HAVE_HALIDE */
