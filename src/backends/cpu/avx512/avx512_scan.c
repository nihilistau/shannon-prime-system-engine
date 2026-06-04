/* avx512_scan.c — the 32k-wall scan kernel: AVX512-VPOPCNTDQ + OMP.
 *
 * sp_avx512_scan_sig scores n candidates of one head's CONTIGUOUS u64
 * signature slice (head-major sidecar, arm.h layout v2):
 *     cand[i] = { -(float)popcount(qsig ^ sigs[i]), s0 + i }.
 *
 * 8 u64 popcounts per _mm512_popcnt_epi64; the slice is stride-1 so the
 * hardware prefetcher streams it. OMP chunks the range across threads only
 * when n is large enough to amortize the fork (the 32k ingest regime);
 * chunk writes into cand are disjoint.
 *
 * EXACTNESS (arm.h contract): integer popcounts are exact; the float is
 * -(float)ham with ham <= 64, exactly representable — entries are IDENTICAL
 * to the portable reference (gate: T_ARM scan check + NIAH sequence parity).
 * Compiled with -mavx512vpopcntdq (see src/CMakeLists.txt avx512 list);
 * caller (cpu_overlay) dispatches on g_avx512_caps.has_vpopcntdq.
 */
#include "sp_engine/avx512.h"
#include "sp/arm.h"

#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void sp_avx512_scan_sig(uint64_t qsig, const uint64_t *sigs, int n, int s0,
                        void *cand_v) {
    sp_arm_sidx *cand = (sp_arm_sidx *)cand_v;
    const __m512i vq = _mm512_set1_epi64((long long)qsig);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 8192)
#endif
    for (int base = 0; base < n; base += 8) {
        const int lim = (base + 8 <= n) ? base + 8 : n;
        if (lim - base == 8) {
            const __m512i x = _mm512_xor_si512(
                _mm512_loadu_si512((const void *)(sigs + base)), vq);
            const __m512i pc = _mm512_popcnt_epi64(x);
            uint64_t h[8];
            _mm512_storeu_si512((void *)h, pc);
            for (int l = 0; l < 8; l++) {
                cand[base + l].s = -(float)(int)h[l];
                cand[base + l].i = s0 + base + l;
            }
        } else {
            for (int i = base; i < lim; i++) {   /* tail < 8 */
                cand[i].s = -(float)__builtin_popcountll(qsig ^ sigs[i]);
                cand[i].i = s0 + i;
            }
        }
    }
}
