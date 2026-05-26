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
/* Sub-phase headers added progressively:
 * #include "ptx_mma.cuh"
 */

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

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *filter = (argc > 1) ? argv[1] : "all";
    bool gpu = gpu_available();
    printf("ptx_validate: GPU=%s filter=%s\n", gpu ? "YES" : "NO (SKIP)", filter);

    int pass = 1, skip = 0;

    if (!strcmp(filter, "ntt") || !strcmp(filter, "all")) {
        if (!gpu) { printf("M_PTX_1 NTT: SKIP\n"); skip++; }
        else       { if (!validate_ntt()) pass = 0; }
    }

    if (!strcmp(filter, "hash") || !strcmp(filter, "all")) {
        if (!gpu) { printf("M_PTX_1 HASH: SKIP\n"); skip++; }
        else       { if (!validate_hash()) pass = 0; }
    }

    if (!strcmp(filter, "spinor") || !strcmp(filter, "all")) {
        if (!gpu) { printf("M_PTX_1 SPINOR: SKIP\n"); skip++; }
        else       { if (!validate_spinor()) pass = 0; }
    }

    if (!strcmp(filter, "mma") || !strcmp(filter, "all")) {
        printf("M_PTX_1 MMA: SKIP (not yet implemented)\n"); skip++;
    }

    printf("ptx_validate: %s (skip=%d)\n", pass ? "PASS" : "FAIL", skip);
    return pass ? 0 : 1;
}
