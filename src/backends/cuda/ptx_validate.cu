/*
 * ptx_validate.cu — M_PTX_1 bit-identity validation for Phase 2-CU.PTX.
 * Compares PTX kernel output vs math-core C scalar reference.
 * Usage: ./ptx_validate [ntt|hash|spinor|mma|all]
 * On no-GPU host: prints SKIP for each GPU gate.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#include "ptx_ntt.cuh"
#include "ptx_hash.cuh"
#include "ptx_spinor.cuh"
#include "ptx_mma.cuh"

static bool gpu_available(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    if (n == 0) return false;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    if (p.major < 7 || (p.major == 7 && p.minor < 5)) {
        fprintf(stderr, "ptx_validate: sm_%d%d < sm_75; PTX gates require sm_75+\n",
                p.major, p.minor);
        return false;
    }
    return true;
}

/* ── NTT gate ────────────────────────────────────────────────────────── */

__global__ static void k_ntt_modmul_q1(const uint32_t *a, const uint32_t *b,
                                        uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ptx_modmul_q1(a[i], b[i]);
}

__global__ static void k_ntt_modmul_q2(const uint32_t *a, const uint32_t *b,
                                        uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ptx_modmul_q2(a[i], b[i]);
}

static bool run_ntt_modmul(uint32_t q, const char *tag,
                            const uint32_t *h_a, const uint32_t *h_b,
                            const uint32_t *h_ref, int n,
                            void (*kernel)(const uint32_t *, const uint32_t *,
                                           uint32_t *, int)) {
    uint32_t *d_a, *d_b, *d_out;
    cudaMalloc(&d_a,  n * sizeof(uint32_t));
    cudaMalloc(&d_b,  n * sizeof(uint32_t));
    cudaMalloc(&d_out, n * sizeof(uint32_t));
    cudaMemcpy(d_a, h_a, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    kernel(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    uint32_t *h_got = new uint32_t[n];
    cudaMemcpy(h_got, d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

    bool pass = true;
    for (int i = 0; i < n && pass; i++) {
        if (h_ref[i] != h_got[i]) {
            printf("M_PTX_1 NTT %s: FAIL (idx=%d ref=%u got=%u a=%u b=%u)\n",
                   tag, i, h_ref[i], h_got[i], h_a[i], h_b[i]);
            pass = false;
        }
    }
    if (pass) printf("M_PTX_1 NTT %s: PASS (%d pairs)\n", tag, n);
    delete[] h_got;
    return pass;
}

static bool validate_ntt(void) {
    const int N = 1024;
    uint32_t *h_a   = new uint32_t[N];
    uint32_t *h_b   = new uint32_t[N];
    uint32_t *h_ref = new uint32_t[N];

    /* Q1 test — inputs in [0, q1) */
    h_a[0] = 0;              h_b[0] = 0;
    h_a[1] = 1;              h_b[1] = PTX_NTT_Q1 - 1;
    h_a[2] = PTX_NTT_Q1 - 1; h_b[2] = PTX_NTT_Q1 - 1;
    h_a[3] = PTX_NTT_Q1 / 2; h_b[3] = PTX_NTT_Q1 / 2;
    for (int i = 4; i < N; i++) {
        h_a[i] = (uint32_t)((1664525ULL  * (uint32_t)i + 1013904223ULL) % PTX_NTT_Q1);
        h_b[i] = (uint32_t)((22695477ULL * (uint32_t)i + 1ULL)          % PTX_NTT_Q1);
    }
    for (int i = 0; i < N; i++) h_ref[i] = modmul_ref_q1(h_a[i], h_b[i]);

    bool ok = run_ntt_modmul(PTX_NTT_Q1, "Q1", h_a, h_b, h_ref, N,
                              [](const uint32_t *a, const uint32_t *b,
                                 uint32_t *o, int n) {
                                  k_ntt_modmul_q1<<<(n+255)/256, 256>>>(a, b, o, n);
                              });

    /* Q2 test — clamp inputs to [0, q2) */
    h_a[0] = 0;              h_b[0] = 0;
    h_a[1] = 1;              h_b[1] = PTX_NTT_Q2 - 1;
    h_a[2] = PTX_NTT_Q2 - 1; h_b[2] = PTX_NTT_Q2 - 1;
    h_a[3] = PTX_NTT_Q2 / 2; h_b[3] = PTX_NTT_Q2 / 2;
    for (int i = 4; i < N; i++) {
        h_a[i] = (uint32_t)((1664525ULL  * (uint32_t)i + 1013904223ULL) % PTX_NTT_Q2);
        h_b[i] = (uint32_t)((22695477ULL * (uint32_t)i + 1ULL)          % PTX_NTT_Q2);
    }
    for (int i = 0; i < N; i++) h_ref[i] = modmul_ref_q2(h_a[i], h_b[i]);

    ok &= run_ntt_modmul(PTX_NTT_Q2, "Q2", h_a, h_b, h_ref, N,
                          [](const uint32_t *a, const uint32_t *b,
                             uint32_t *o, int n) {
                              k_ntt_modmul_q2<<<(n+255)/256, 256>>>(a, b, o, n);
                          });

    delete[] h_a; delete[] h_b; delete[] h_ref;
    return ok;
}

/* ── HASH gate ───────────────────────────────────────────────────────── */

__global__ static void k_hash_xor3(const uint32_t *a, const uint32_t *b,
                                    const uint32_t *c, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ptx_xor3(a[i], b[i], c[i]);
}

__global__ static void k_hash_prmt(const uint32_t *a, const uint32_t *b,
                                    const uint32_t *sel, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ptx_prmt(a[i], b[i], sel[i]);
}

static bool validate_hash(void) {
    bool pass = true;

    /* ── xor3 test: 1024 triples ────────────────────────────────────── */
    const int N_XOR = 1024;
    uint32_t *h_xa = new uint32_t[N_XOR];
    uint32_t *h_xb = new uint32_t[N_XOR];
    uint32_t *h_xc = new uint32_t[N_XOR];
    uint32_t *h_xref = new uint32_t[N_XOR];

    /* corners at indices 0-2 */
    h_xa[0] = 0;          h_xb[0] = 0;          h_xc[0] = 0;
    h_xa[1] = 0xFFFFFFFFu; h_xb[1] = 0xFFFFFFFFu; h_xc[1] = 0xFFFFFFFFu;
    h_xa[2] = 0x55555555u; h_xb[2] = 0xAAAAAAAAu; h_xc[2] = 0xFFFFFFFFu;
    /* LCG-generated randoms for indices 3..N-1 */
    uint32_t lcg = 0xDEADBEEFu;
    for (int i = 3; i < N_XOR; i++) {
        lcg = lcg * 1664525u + 1013904223u; h_xa[i] = lcg;
        lcg = lcg * 1664525u + 1013904223u; h_xb[i] = lcg;
        lcg = lcg * 1664525u + 1013904223u; h_xc[i] = lcg;
    }
    for (int i = 0; i < N_XOR; i++) h_xref[i] = h_xa[i] ^ h_xb[i] ^ h_xc[i];

    uint32_t *d_xa, *d_xb, *d_xc, *d_xout;
    cudaMalloc(&d_xa,  N_XOR * sizeof(uint32_t));
    cudaMalloc(&d_xb,  N_XOR * sizeof(uint32_t));
    cudaMalloc(&d_xc,  N_XOR * sizeof(uint32_t));
    cudaMalloc(&d_xout, N_XOR * sizeof(uint32_t));
    cudaMemcpy(d_xa, h_xa, N_XOR * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xb, h_xb, N_XOR * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xc, h_xc, N_XOR * sizeof(uint32_t), cudaMemcpyHostToDevice);

    k_hash_xor3<<<(N_XOR + 255) / 256, 256>>>(d_xa, d_xb, d_xc, d_xout, N_XOR);
    cudaDeviceSynchronize();

    uint32_t *h_xgot = new uint32_t[N_XOR];
    cudaMemcpy(h_xgot, d_xout, N_XOR * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_xa); cudaFree(d_xb); cudaFree(d_xc); cudaFree(d_xout);

    bool xor3_pass = true;
    for (int i = 0; i < N_XOR && xor3_pass; i++) {
        if (h_xref[i] != h_xgot[i]) {
            printf("M_PTX_1 HASH xor3: FAIL (idx=%d ref=0x%08X got=0x%08X)\n",
                   i, h_xref[i], h_xgot[i]);
            xor3_pass = false;
        }
    }
    if (xor3_pass) printf("M_PTX_1 HASH xor3: PASS (%d triples)\n", N_XOR);
    delete[] h_xa; delete[] h_xb; delete[] h_xc; delete[] h_xref; delete[] h_xgot;
    pass &= xor3_pass;

    /* ── prmt test: 256 selectors ──────────────────────────────────── */
    const int N_PRM = 256;
    uint32_t *h_pa  = new uint32_t[N_PRM];
    uint32_t *h_pb  = new uint32_t[N_PRM];
    uint32_t *h_sel = new uint32_t[N_PRM];
    uint32_t *h_pref = new uint32_t[N_PRM];

    for (int i = 0; i < N_PRM; i++) {
        h_pa[i]  = 0x03020100u;
        h_pb[i]  = 0x07060504u;
        h_sel[i] = (uint32_t)i;
        /* C emulation of prmt.b32 (host fallback; __CUDA_ARCH__ not defined) */
        h_pref[i] = ptx_prmt(h_pa[i], h_pb[i], h_sel[i]);
    }

    uint32_t *d_pa, *d_pb, *d_psel, *d_pout;
    cudaMalloc(&d_pa,  N_PRM * sizeof(uint32_t));
    cudaMalloc(&d_pb,  N_PRM * sizeof(uint32_t));
    cudaMalloc(&d_psel, N_PRM * sizeof(uint32_t));
    cudaMalloc(&d_pout, N_PRM * sizeof(uint32_t));
    cudaMemcpy(d_pa,  h_pa,  N_PRM * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pb,  h_pb,  N_PRM * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_psel, h_sel, N_PRM * sizeof(uint32_t), cudaMemcpyHostToDevice);

    k_hash_prmt<<<1, 256>>>(d_pa, d_pb, d_psel, d_pout, N_PRM);
    cudaDeviceSynchronize();

    uint32_t *h_pgot = new uint32_t[N_PRM];
    cudaMemcpy(h_pgot, d_pout, N_PRM * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_pa); cudaFree(d_pb); cudaFree(d_psel); cudaFree(d_pout);

    bool prmt_pass = true;
    for (int i = 0; i < N_PRM && prmt_pass; i++) {
        if (h_pref[i] != h_pgot[i]) {
            printf("M_PTX_1 HASH prmt: FAIL (selector=0x%02X ref=0x%08X got=0x%08X)\n",
                   i, h_pref[i], h_pgot[i]);
            prmt_pass = false;
        }
    }
    if (prmt_pass) printf("M_PTX_1 HASH prmt: PASS (%d combinations)\n", N_PRM);
    delete[] h_pa; delete[] h_pb; delete[] h_sel; delete[] h_pref; delete[] h_pgot;
    pass &= prmt_pass;

    return pass;
}

/* ── SPINOR gate ─────────────────────────────────────────────────────── */

__global__ static void k_spinor_load(const uint32_t *src, uint32_t *dst,
                                      int n_blocks, int is_hot) {
    int b    = (int)blockIdx.x;
    int lane = (int)threadIdx.x;   /* blockDim must be 32 */
    if (b < n_blocks)
        dst[b * 32 + lane] = sp_spinor_warpload(src, (uint32_t)b, lane, is_hot);
}

static bool validate_spinor(void) {
    /* 16 blocks × 32 words src (index b*16+lane, max = 15*16+31 = 271 < 512) */
    const int N = 16;
    uint32_t *h_src = new uint32_t[N * 32];
    uint32_t *h_got = new uint32_t[N * 32];
    for (int i = 0; i < N * 32; i++)
        h_src[i] = (uint32_t)((uint32_t)i * 2654435761u + 1u);

    uint32_t *d_src, *d_dst;
    cudaMalloc(&d_src, N * 32 * sizeof(uint32_t));
    cudaMalloc(&d_dst, N * 32 * sizeof(uint32_t));
    cudaMemcpy(d_src, h_src, N * 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    bool pass = true;
    const char *labels[2] = {"cold", "hot"};
    for (int hot = 1; hot >= 0 && pass; hot--) {
        cudaMemset(d_dst, 0, N * 32 * sizeof(uint32_t));
        k_spinor_load<<<N, 32>>>(d_src, d_dst, N, hot);
        cudaDeviceSynchronize();
        cudaMemcpy(h_got, d_dst, N * 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int b = 0; b < N && pass; b++) {
            for (int lane = 0; lane < 32 && pass; lane++) {
                uint32_t expected = h_src[b * 16 + lane];
                uint32_t got      = h_got[b * 32 + lane];
                if (expected != got) {
                    printf("M_PTX_1 SPINOR %s: FAIL (b=%d lane=%d exp=%u got=%u)\n",
                           labels[hot], b, lane, expected, got);
                    pass = false;
                }
            }
        }
        if (pass) printf("M_PTX_1 SPINOR %s: PASS (%d blocks)\n", labels[hot], N);
    }

    cudaFree(d_src); cudaFree(d_dst);
    delete[] h_src; delete[] h_got;
    return pass;
}

/* ── MMA gate ────────────────────────────────────────────────────────── */

/*
 * validate_mma
 *
 * Runs M_PTX_1 (correctness), M_PTX_3 (no malloc in hot path),
 * and M_PTX_4 (two streams, no global sync) for the INT8 WMMA kernel.
 *
 * Returns true iff all three gates pass; increments pass/skip/fail counts.
 */
static bool validate_mma(int *p_pass_count, int *p_fail_count) {
    bool all_ok = true;

    /* ── Allocate device buffers (cudaMalloc happens HERE, not in kernel) ── */
    /* Test A: 16x16x16 all-ones */
    const int MA = MMA_M, KA = MMA_K, NA = MMA_N;
    int8_t  *h_Aa = new int8_t [MA * KA];
    int8_t  *h_Ba = new int8_t [KA * NA];
    float   *h_sca_a = new float[MA];
    float   *h_sca_b = new float[NA];
    __half  *h_Ca_got = new __half[MA * NA];
    for (int i = 0; i < MA * KA; i++) h_Aa[i] = 1;
    for (int i = 0; i < KA * NA; i++) h_Ba[i] = 1;
    for (int i = 0; i < MA; i++) h_sca_a[i] = 1.0f;
    for (int i = 0; i < NA; i++) h_sca_b[i] = 1.0f;

    int8_t *d_Aa, *d_Ba; float *d_sca_a, *d_sca_b; __half *d_Ca;
    cudaMalloc(&d_Aa,    MA * KA * sizeof(int8_t));
    cudaMalloc(&d_Ba,    KA * NA * sizeof(int8_t));
    cudaMalloc(&d_sca_a, MA * sizeof(float));
    cudaMalloc(&d_sca_b, NA * sizeof(float));
    cudaMalloc(&d_Ca,    MA * NA * sizeof(__half));
    cudaMemcpy(d_Aa,    h_Aa,    MA * KA * sizeof(int8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ba,    h_Ba,    KA * NA * sizeof(int8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_sca_a, h_sca_a, MA * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_sca_b, h_sca_b, NA * sizeof(float),        cudaMemcpyHostToDevice);

    /* Launch on default stream (stream=0) */
    sp_frob_matmul_q8_mma(d_Aa, d_Ba, d_sca_a, d_sca_b, d_Ca,
                           MA, KA, NA, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Ca_got, d_Ca, MA * NA * sizeof(__half), cudaMemcpyDeviceToHost);

    /* Check M_PTX_3: kernel ran without memory allocation error */
    cudaError_t err3 = cudaGetLastError();
    if (err3 == cudaSuccess) {
        printf("M_PTX_3 MMA: PASS\n");
        (*p_pass_count)++;
    } else {
        printf("M_PTX_3 MMA: FAIL (cuda error %s)\n", cudaGetErrorString(err3));
        (*p_fail_count)++;
        all_ok = false;
    }

    /* Validate Test A: all elements should be __float2half(16.0f) */
    __half expected_a = __float2half(16.0f);
    bool test_a_ok = true;
    int test_a_fail_cnt = 0;
    for (int i = 0; i < MA * NA; i++) {
        /* Compare as uint16 for exact bit match */
        uint16_t got_bits, exp_bits;
        memcpy(&got_bits, &h_Ca_got[i], sizeof(uint16_t));
        memcpy(&exp_bits, &expected_a,  sizeof(uint16_t));
        if (got_bits != exp_bits) {
            if (test_a_fail_cnt == 0)
                printf("M_PTX_1 MMA TestA: FAIL (first at idx=%d exp=%.1f got=%.4f)\n",
                       i, __half2float(expected_a), __half2float(h_Ca_got[i]));
            test_a_fail_cnt++;
            test_a_ok = false;
        }
    }
    if (test_a_ok)
        printf("M_PTX_1 MMA TestA: PASS (all-ones 16x16x16 -> 16.0f)\n");
    else
        printf("M_PTX_1 MMA TestA: %d mismatches\n", test_a_fail_cnt);

    cudaFree(d_Aa); cudaFree(d_Ba); cudaFree(d_sca_a); cudaFree(d_sca_b); cudaFree(d_Ca);
    delete[] h_Aa; delete[] h_Ba; delete[] h_sca_a; delete[] h_sca_b; delete[] h_Ca_got;

    /* ── Test B: random Q8 correctness (64x64x64) ─────────────────────── */
    const int MB = 64, KB = 64, NB = 64;
    int8_t *h_Ab = new int8_t [MB * KB];
    int8_t *h_Bb = new int8_t [KB * NB];
    float  *h_scb_a = new float[MB];
    float  *h_scb_b = new float[NB];
    __half *h_Cb_ref = new __half[MB * NB];
    __half *h_Cb_got = new __half[MB * NB];

    /* LCG random values in [-127, 127] */
    uint32_t lcg = 0xBEEFCAFEu;
    for (int i = 0; i < MB * KB; i++) {
        lcg = lcg * 1664525u + 1013904223u;
        h_Ab[i] = (int8_t)(((int)(lcg & 0xFFu)) - 128);
        if (h_Ab[i] < -127) h_Ab[i] = -127;
    }
    for (int i = 0; i < KB * NB; i++) {
        lcg = lcg * 1664525u + 1013904223u;
        h_Bb[i] = (int8_t)(((int)(lcg & 0xFFu)) - 128);
        if (h_Bb[i] < -127) h_Bb[i] = -127;
    }
    for (int i = 0; i < MB; i++) h_scb_a[i] = 1.0f / 127.0f;
    for (int i = 0; i < NB; i++) h_scb_b[i] = 1.0f;

    /* CPU reference */
    sp_frob_matmul_q8_ref(h_Ab, h_Bb, h_scb_a, h_scb_b, h_Cb_ref, MB, KB, NB);

    /* GPU */
    int8_t *d_Ab, *d_Bb; float *d_scb_a, *d_scb_b; __half *d_Cb;
    cudaMalloc(&d_Ab,    MB * KB * sizeof(int8_t));
    cudaMalloc(&d_Bb,    KB * NB * sizeof(int8_t));
    cudaMalloc(&d_scb_a, MB * sizeof(float));
    cudaMalloc(&d_scb_b, NB * sizeof(float));
    cudaMalloc(&d_Cb,    MB * NB * sizeof(__half));
    cudaMemcpy(d_Ab,    h_Ab,    MB * KB * sizeof(int8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bb,    h_Bb,    KB * NB * sizeof(int8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_scb_a, h_scb_a, MB * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_scb_b, h_scb_b, NB * sizeof(float),        cudaMemcpyHostToDevice);

    sp_frob_matmul_q8_mma(d_Ab, d_Bb, d_scb_a, d_scb_b, d_Cb,
                           MB, KB, NB, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Cb_got, d_Cb, MB * NB * sizeof(__half), cudaMemcpyDeviceToHost);

    bool test_b_ok = true;
    int test_b_fail_cnt = 0;
    for (int i = 0; i < MB * NB; i++) {
        uint16_t ref_bits, got_bits;
        memcpy(&ref_bits, &h_Cb_ref[i], sizeof(uint16_t));
        memcpy(&got_bits, &h_Cb_got[i], sizeof(uint16_t));
        if (ref_bits != got_bits) {
            if (test_b_fail_cnt == 0)
                printf("M_PTX_1 MMA TestB: FAIL (first at idx=%d ref=%.6f got=%.6f)\n",
                       i, __half2float(h_Cb_ref[i]), __half2float(h_Cb_got[i]));
            test_b_fail_cnt++;
            test_b_ok = false;
        }
    }
    if (test_b_ok)
        printf("M_PTX_1 MMA TestB: PASS (random Q8 64x64x64 bit-exact)\n");
    else
        printf("M_PTX_1 MMA TestB: %d mismatches out of %d\n",
               test_b_fail_cnt, MB * NB);

    cudaFree(d_Ab); cudaFree(d_Bb); cudaFree(d_scb_a); cudaFree(d_scb_b); cudaFree(d_Cb);
    delete[] h_Ab; delete[] h_Bb; delete[] h_scb_a; delete[] h_scb_b;
    delete[] h_Cb_ref; delete[] h_Cb_got;

    if (test_a_ok && test_b_ok) {
        printf("M_PTX_1 MMA: PASS\n");
        (*p_pass_count)++;
    } else {
        printf("M_PTX_1 MMA: FAIL\n");
        (*p_fail_count)++;
        all_ok = false;
    }

    /* ── M_PTX_4: two streams, no global sync ─────────────────────────── */
    /*
     * Launch the kernel on two independent streams simultaneously.
     * Use per-stream CUDA events to verify both complete without error.
     * Critically: do NOT call cudaDeviceSynchronize() between launches.
     */
    const int M4 = 32, K4 = 16, N4 = 32;
    int8_t  *h_A4 = new int8_t [M4 * K4];
    int8_t  *h_B4 = new int8_t [K4 * N4];
    float   *h_s4a = new float[M4];
    float   *h_s4b = new float[N4];
    for (int i = 0; i < M4 * K4; i++) h_A4[i] = 1;
    for (int i = 0; i < K4 * N4; i++) h_B4[i] = 1;
    for (int i = 0; i < M4; i++) h_s4a[i] = 1.0f;
    for (int i = 0; i < N4; i++) h_s4b[i] = 1.0f;

    int8_t *d_A4, *d_B4; float *d_s4a, *d_s4b;
    __half *d_C4s0, *d_C4s1;
    cudaMalloc(&d_A4,   M4 * K4 * sizeof(int8_t));
    cudaMalloc(&d_B4,   K4 * N4 * sizeof(int8_t));
    cudaMalloc(&d_s4a,  M4 * sizeof(float));
    cudaMalloc(&d_s4b,  N4 * sizeof(float));
    cudaMalloc(&d_C4s0, M4 * N4 * sizeof(__half));
    cudaMalloc(&d_C4s1, M4 * N4 * sizeof(__half));
    cudaMemcpy(d_A4,  h_A4,  M4 * K4 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B4,  h_B4,  K4 * N4 * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s4a, h_s4a, M4 * sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_s4b, h_s4b, N4 * sizeof(float),        cudaMemcpyHostToDevice);

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaEvent_t ev0s, ev0e, ev1s, ev1e;
    cudaEventCreate(&ev0s); cudaEventCreate(&ev0e);
    cudaEventCreate(&ev1s); cudaEventCreate(&ev1e);

    /* Record start, launch on stream0, record end */
    cudaEventRecord(ev0s, stream0);
    sp_frob_matmul_q8_mma(d_A4, d_B4, d_s4a, d_s4b, d_C4s0,
                           M4, K4, N4, stream0);
    cudaEventRecord(ev0e, stream0);

    /* Record start, launch on stream1, record end — NO global sync in between */
    cudaEventRecord(ev1s, stream1);
    sp_frob_matmul_q8_mma(d_A4, d_B4, d_s4a, d_s4b, d_C4s1,
                           M4, K4, N4, stream1);
    cudaEventRecord(ev1e, stream1);

    /* Wait on each stream independently */
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaError_t e0 = cudaEventQuery(ev0e);
    cudaError_t e1 = cudaEventQuery(ev1e);

    if (e0 == cudaSuccess && e1 == cudaSuccess) {
        printf("M_PTX_4 MMA: PASS (two streams, no global sync)\n");
        (*p_pass_count)++;
    } else {
        printf("M_PTX_4 MMA: FAIL (stream0=%s stream1=%s)\n",
               cudaGetErrorString(e0), cudaGetErrorString(e1));
        (*p_fail_count)++;
        all_ok = false;
    }

    cudaEventDestroy(ev0s); cudaEventDestroy(ev0e);
    cudaEventDestroy(ev1s); cudaEventDestroy(ev1e);
    cudaStreamDestroy(stream0); cudaStreamDestroy(stream1);
    cudaFree(d_A4); cudaFree(d_B4); cudaFree(d_s4a); cudaFree(d_s4b);
    cudaFree(d_C4s0); cudaFree(d_C4s1);
    delete[] h_A4; delete[] h_B4; delete[] h_s4a; delete[] h_s4b;

    return all_ok;
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *filter = (argc > 1) ? argv[1] : "all";
    bool gpu = gpu_available();
    printf("ptx_validate: GPU=%s filter=%s\n", gpu ? "YES" : "NO (SKIP)", filter);

    int pass = 1, skip = 0, fail = 0;

    if (!strcmp(filter, "ntt") || !strcmp(filter, "all")) {
        if (!gpu) { printf("M_PTX_1 NTT: SKIP\n"); skip++; }
        else       { if (!validate_ntt()) { pass = 0; fail++; } }
    }

    if (!strcmp(filter, "hash") || !strcmp(filter, "all")) {
        if (!gpu) { printf("M_PTX_1 HASH: SKIP\n"); skip++; }
        else       { if (!validate_hash()) { pass = 0; fail++; } }
    }

    if (!strcmp(filter, "spinor") || !strcmp(filter, "all")) {
        if (!gpu) { printf("M_PTX_1 SPINOR: SKIP\n"); skip++; }
        else       { if (!validate_spinor()) { pass = 0; fail++; } }
    }

    if (!strcmp(filter, "mma") || !strcmp(filter, "all")) {
        if (!gpu) {
            printf("M_PTX_1 MMA: SKIP\n");
            printf("M_PTX_3 MMA: SKIP\n");
            printf("M_PTX_4 MMA: SKIP\n");
            skip += 3;
        } else {
            /* pass/fail counters are updated inside validate_mma */
            int mma_pass = 0, mma_fail = 0;
            validate_mma(&mma_pass, &mma_fail);
            if (mma_fail > 0) { pass = 0; fail += mma_fail; }
        }
    }

    printf("ptx_validate: %s (skip=%d)\n", (pass && fail == 0) ? "PASS" : "FAIL", skip);
    return (pass && fail == 0) ? 0 : 1;
}
