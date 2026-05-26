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
/* Sub-phase headers added progressively:
 * #include "ptx_hash.cuh"
 * #include "ptx_spinor.cuh"
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

    kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_out, n);
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
        printf("M_PTX_1 HASH: SKIP (not yet implemented)\n"); skip++;
    }

    if (!strcmp(filter, "spinor") || !strcmp(filter, "all")) {
        printf("M_PTX_1 SPINOR: SKIP (not yet implemented)\n"); skip++;
    }

    if (!strcmp(filter, "mma") || !strcmp(filter, "all")) {
        printf("M_PTX_1 MMA: SKIP (not yet implemented)\n"); skip++;
    }

    printf("ptx_validate: %s (skip=%d)\n", pass ? "PASS" : "FAIL", skip);
    return pass ? 0 : 1;
}
