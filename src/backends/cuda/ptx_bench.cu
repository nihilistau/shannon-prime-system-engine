/*
 * ptx_bench.cu — M_PTX_2 throughput benchmark for Phase 2-CU.PTX sub-phases.
 * Usage: ./ptx_bench [ntt|hash|spinor|mma|all]
 * On no-GPU host: prints SKIP for each gate.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#include "ptx_ntt.cuh"

static bool gpu_available(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    if (n == 0) return false;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    if (p.major < 7 || (p.major == 7 && p.minor < 5)) {
        fprintf(stderr, "ptx_bench: sm_%d%d < sm_75\n", p.major, p.minor);
        return false;
    }
    return true;
}

/* ── NTT throughput bench (Task 1.3) ───────────────────────────────── */

__global__ static void k_bench_ntt_ptx(uint32_t *inout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t a = inout[i];
        uint32_t b = inout[(i + 1) % n];
        inout[i] = ptx_modmul_q1(a, b);
    }
}

/* nvcc-baseline: force the % operator (compiler cannot see PTX path) */
__global__ __noinline__ static void k_bench_ntt_baseline(uint32_t *inout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint64_t a = inout[i];
        uint64_t b = inout[(i + 1) % n];
        inout[i] = (uint32_t)((a * b) % PTX_NTT_Q1);
    }
}

static void bench_ntt(void) {
    const int N = 1 << 20;  /* 1M elements */
    uint32_t *d_buf;
    cudaMalloc(&d_buf, N * sizeof(uint32_t));

    /* warm-up run */
    k_bench_ntt_ptx<<<(N+255)/256, 256>>>(d_buf, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int REPS = 100;

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_ntt_ptx<<<(N+255)/256, 256>>>(d_buf, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_ptx = 0;
    cudaEventElapsedTime(&ms_ptx, t0, t1);

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_ntt_baseline<<<(N+255)/256, 256>>>(d_buf, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_base = 0;
    cudaEventElapsedTime(&ms_base, t0, t1);

    double ops        = (double)N * REPS;
    double bfly_ptx   = ops / (ms_ptx  * 1e-3);
    double bfly_base  = ops / (ms_base * 1e-3);
    double speedup    = bfly_ptx / bfly_base;

    printf("NTT bench: PTX %.2e bfly/s  baseline %.2e bfly/s  speedup=%.1fx\n",
           bfly_ptx, bfly_base, speedup);
    printf("M_PTX_2 NTT: %s (target >=8x)\n", speedup >= 8.0 ? "PASS" : "NEEDS_TUNING");

    cudaFree(d_buf);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ── main ────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *filter = (argc > 1) ? argv[1] : "all";
    bool gpu = gpu_available();
    printf("ptx_bench: GPU=%s filter=%s\n", gpu ? "YES" : "NO (SKIP)", filter);

    if (!strcmp(filter, "ntt") || !strcmp(filter, "all")) {
        if (!gpu) printf("M_PTX_2 NTT: SKIP\n");
        else      bench_ntt();
    }
    if (!strcmp(filter, "hash")   || !strcmp(filter, "all"))   printf("M_PTX_2 HASH: SKIP\n");
    if (!strcmp(filter, "spinor") || !strcmp(filter, "all"))  printf("M_PTX_2 SPINOR: SKIP\n");
    if (!strcmp(filter, "mma")    || !strcmp(filter, "all"))   printf("M_PTX_2 MMA: SKIP\n");

    return 0;
}
