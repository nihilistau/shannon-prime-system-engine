/* avx512.h — AVX-512 lattice kernel dispatch surface. §18 CPU.AVX */
#ifndef SP_ENGINE_AVX512_H
#define SP_ENGINE_AVX512_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Runtime capability flags, populated by sp_avx512_init(). */
typedef struct {
    unsigned has_avx512f      : 1;
    unsigned has_vnni         : 1;
    unsigned has_ifma         : 1;
    unsigned has_waitpkg      : 1;
    unsigned has_vpopcntdq    : 1;
    unsigned has_vbmi2        : 1;
    unsigned _pad             : 26;
} sp_avx512_caps;

extern sp_avx512_caps g_avx512_caps;

/* Call once at engine init (before any kernel dispatch). Thread-safe once done. */
void sp_avx512_init(void);

/* §18.1 SPINOR: load one 64-byte arena slot into a ZMM register and verify
 * the 0xA5 sentinel at byte 63. Returns 0 if sentinel OK, -1 if mismatch.
 * `slot` must be 64-byte aligned. */
int sp_avx512_spinor_load_check(const void *slot);

/* NT variant: bypasses L1+L2 cache, for cold sweep paths. */
int sp_avx512_spinor_nt_load_check(const void *slot);

/* §18.2 VNNI: Q8 arena matrix-vector multiply.
 * out[i] = sum_k(w_codes[i*cols+k] * act_i8[k]) * row_scale[i]
 * Caller passes act_u8 = act_i8 + 128 (zero-shifted), bias[i] = 128*sum(w_codes[i]).
 * rows and cols must be multiples of 64.
 * w_codes (int8, row-major), act_u8 (uint8), row_scale (f32), bias (int32, per-row), out (f32). */
void sp_avx512_vnni_matvec(const int8_t *w_codes, const uint8_t *act_u8,
                            const float *row_scale, const int32_t *bias,
                            int rows, int cols, float *out);

/* §18.3 IFMA: pointwise Barrett multiply of two length-N residue arrays mod q.
 * N must be a multiple of 8. q must be a 30-bit prime; mu = floor(2^60/q).
 * Equivalent to ntt_pointwise_mul for one prime channel. */
void sp_avx512_ifma_modmul(const uint32_t *a, const uint32_t *b,
                            uint32_t q, uint64_t mu, uint32_t N, uint32_t *out);

/* §18.4 TERNLOG: KSTE hash step — bitwise ternary logic over 512-bit state lanes.
 * Applies the KSTE mixing round to `state` in-place (16 x uint32 = 512 bits).
 * state must be 64-byte aligned. */
void sp_avx512_ternlog_kste_round(uint32_t *state);

/* §18.4 TERNLOG: sieve popcount — number of set bits across 8 uint64 lanes. */
uint64_t sp_avx512_ternlog_popcnt512(const uint64_t *v8);

/* §18.5 PERSIST: UMONITOR/UMWAIT sentinel for zero-OS-overhead thread wakeup.
 * The sentinel lives on a VirtualLock'd page (requires SeLockMemoryPrivilege).
 * WAITPKG hardware uses _umonitor+_umwait (C0.2); Zen 4 / no-WAITPKG falls
 * back to _mm_pause spin. Gate: M_AVX_PERSIST_1 (median wakeup ≤200ns),
 *                               M_AVX_PERSIST_2 (cycle ratio idle/busy <0.05). */
typedef struct sp_avx512_persist_sentinel sp_avx512_persist_sentinel;

/* Allocate a page-locked sentinel. Returns NULL if VirtualLock fails (missing
 * SeLockMemoryPrivilege) — caller must handle this case. */
sp_avx512_persist_sentinel *sp_avx512_persist_alloc(void);
void sp_avx512_persist_free(sp_avx512_persist_sentinel *s);

/* Block until sp_avx512_persist_wake() is called or timeout_ns elapses.
 * Call sp_avx512_init() before first use. Not re-entrant on the same sentinel. */
void sp_avx512_persist_wait(sp_avx512_persist_sentinel *s, uint64_t timeout_ns);

/* Write to the sentinel cache-line, waking a UMWAIT/spin waiter. */
void sp_avx512_persist_wake(sp_avx512_persist_sentinel *s);

#ifdef __cplusplus
}
#endif

#endif /* SP_ENGINE_AVX512_H */
