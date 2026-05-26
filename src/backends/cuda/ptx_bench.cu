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

/* ── HASH throughput bench (Task 2) ─────────────────────────────────── */

__global__ static void k_bench_hash_ptx(uint32_t *inout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t a = inout[i];
        uint32_t b = inout[(i + 1) % n];
        uint32_t c = inout[(i + 2) % n];
        inout[i] = ptx_xor3(a, b, c);
    }
}

/* C-level baseline: force compiler to emit 2 XOR instructions (not lop3) */
__global__ __noinline__ static void k_bench_hash_baseline(uint32_t *inout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t a = inout[i];
        uint32_t b = inout[(i + 1) % n];
        uint32_t c = inout[(i + 2) % n];
        inout[i] = a ^ b ^ c;
    }
}

static void bench_hash(void) {
    const int N = 10 * 1024 * 1024;  /* 10M elements */
    uint32_t *d_buf;
    cudaMalloc(&d_buf, N * sizeof(uint32_t));

    /* warm-up */
    k_bench_hash_ptx<<<(N + 255) / 256, 256>>>(d_buf, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int REPS = 50;

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_hash_ptx<<<(N + 255) / 256, 256>>>(d_buf, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_ptx = 0;
    cudaEventElapsedTime(&ms_ptx, t0, t1);

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_hash_baseline<<<(N + 255) / 256, 256>>>(d_buf, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_base = 0;
    cudaEventElapsedTime(&ms_base, t0, t1);

    double ops       = (double)N * REPS;
    double ops_ptx   = ops / (ms_ptx  * 1e-3);
    double ops_base  = ops / (ms_base * 1e-3);
    double speedup   = ops_ptx / ops_base;

    printf("HASH bench: PTX xor3 %.2e ops/s  baseline %.2e ops/s  speedup=%.1fx\n",
           ops_ptx, ops_base, speedup);
    printf("M_PTX_2 HASH: %s (target >=1.5x)\n", speedup >= 1.5 ? "PASS" : "NEEDS_TUNING");

    cudaFree(d_buf);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ── SPINOR throughput bench (Task 3) ───────────────────────────────── */
/*
 * Sequential chunk kernel: each block owns a contiguous chunk of the n_blocks
 * buffer (chunk_size = n_blocks / gridDim.x) and streams through it in order.
 * Sequential access within each block → DRAM row-buffer hits → maximum row BW.
 * ld.global.cs (evict-first) prevents L2 reuse between iterations; every load
 * is a DRAM fill.  N_GRID is chosen so concurrent L2 footprint > L2 capacity,
 * ensuring the "hot" path also stresses DRAM once L2 is saturated.
 */
__global__ __noinline__ static void k_bench_spinor(const uint32_t *src, uint32_t *sink,
                                                    int n_blocks, int is_hot) {
    int chunk = n_blocks / (int)gridDim.x;
    int start = (int)blockIdx.x * chunk;
    int end   = start + chunk;
    int lane  = threadIdx.x & 31;
    uint32_t acc = 0;
    for (int b = start; b < end; b++) {
        acc ^= sp_spinor_warpload(src, (uint32_t)b, lane, is_hot);
    }
    if (lane == 0 && acc == 0xDEADBEEFu) *sink = acc;  /* prevent DCE */
}

static void bench_spinor(void) {
    /* 16 MB buffer; 960 grid blocks = 32 warps/SM × 30 SMs (max sm_75 occupancy) */
    const int N      = 1 << 17;  /* 131072 spinor blocks = 16 MB */
    const int N_GRID = 960;      /* 32 blocks/SM × 30 SMs — full occupancy, hides DRAM latency */
    const int REPS   = 20;

    uint32_t *d_src, *d_sink;
    cudaMalloc(&d_src,  (size_t)N * 32 * sizeof(uint32_t));
    cudaMalloc(&d_sink, sizeof(uint32_t));
    cudaMemset(d_src, 0xAB, (size_t)N * 32 * sizeof(uint32_t));

    /* warm-up (two passes to fully seat TLB / memory scheduler) */
    k_bench_spinor<<<N_GRID, 32>>>(d_src, d_sink, N, 1);
    k_bench_spinor<<<N_GRID, 32>>>(d_src, d_sink, N, 1);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_spinor<<<N_GRID, 32>>>(d_src, d_sink, N, 1);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_hot = 0;
    cudaEventElapsedTime(&ms_hot, t0, t1);

    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        k_bench_spinor<<<N_GRID, 32>>>(d_src, d_sink, N, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_cold = 0;
    cudaEventElapsedTime(&ms_cold, t0, t1);

    /* bytes = one full pass (N spinor blocks × 128 bytes each) × REPS */
    double bytes = (double)N * 32 * sizeof(uint32_t) * REPS;
    double bw_hot  = bytes / (ms_hot  * 1e-3) / 1e9;
    double bw_cold = bytes / (ms_cold * 1e-3) / 1e9;
    printf("SPINOR bench: hot=%.1f GB/s  cold=%.1f GB/s (timer; L2 may inflate)\n",
           bw_hot, bw_cold);
    /* ncu dram__bytes_read on RTX 2060: ~221-239 GB/s (66-71%% of 336 GB/s DRAM peak).
     * Timer inflates due to L2 hits in warm state.  Gate uses ncu DRAM metric.
     * Improvement path: ld.global.cs.v4 vector loads (512 bytes/warp) needed for 85%+ SOL. */
    printf("M_PTX_2 SPINOR: NEEDS_TUNING (ncu dram__bytes_read ~221-239 GB/s = 66-71%% SOL;"
           " target >=85%% = 286 GB/s)\n");

    cudaFree(d_src); cudaFree(d_sink);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ── MMA throughput bench (Task 4) ──────────────────────────────────── */

/*
 * Baseline: two-kernel pattern simulating k_dequant_arena + SGEMM.
 *
 * Kernel 1 (dequant): reads int8[M*K] + int8[K*N], writes fp32[M*K] + fp32[K*N]
 *   to scratch buffers.  This roundtrip through global memory is the actual
 *   cost that the WMMA path avoids.
 *
 * Kernel 2 (sgemm): reads fp32[M*K] + fp32[K*N], computes naive fp32 matmul,
 *   writes fp32[M*N].
 *
 * WMMA path: sp_frob_matmul_q8_mma — reads int8 directly, uses tensor cores,
 *   writes fp16[M*N].  No intermediate fp32 scratch.
 */

/* Kernel 1: dequant int8 → fp32 scratch (separate A and B passes) */
__global__ __noinline__ static void k_dequant_a(
    const int8_t *__restrict__ src, float *__restrict__ dst, int sz)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < sz) dst[i] = (float)src[i];
}

__global__ __noinline__ static void k_dequant_b(
    const int8_t *__restrict__ src, float *__restrict__ dst, int sz)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < sz) dst[i] = (float)src[i];
}

/* Kernel 2: naive fp32 SGEMM over dequanted scratch */
__global__ __noinline__ static void k_sgemm(
    const float *__restrict__ Af,
    const float *__restrict__ Bf,
    const float *__restrict__ scale_a,
    const float *__restrict__ scale_b,
    float       *__restrict__ C,
    int M, int K, int N)
{
    int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += Af[row * K + k] * Bf[k * N + col];
    C[row * N + col] = acc * scale_a[row] * scale_b[col];
}

static void bench_mma(void) {
    /* Shape: M=256, K=64, N=256 — typical attention layer scale for 0.5B model */
    const int M = 256, K = 64, N = 256;
    const int REPS = 50;

    /* Input buffers (int8) */
    int8_t *d_A, *d_B;
    /* Per-vector scales */
    float  *d_sca, *d_scb;
    /* WMMA output (fp16) */
    __half *d_C_wmma;
    /* Baseline scratch (fp32 dequant) and output */
    float  *d_Af, *d_Bf, *d_C_base;

    cudaMalloc(&d_A,      (size_t)M * K * sizeof(int8_t));
    cudaMalloc(&d_B,      (size_t)K * N * sizeof(int8_t));
    cudaMalloc(&d_sca,    (size_t)M * sizeof(float));
    cudaMalloc(&d_scb,    (size_t)N * sizeof(float));
    cudaMalloc(&d_C_wmma, (size_t)M * N * sizeof(__half));
    cudaMalloc(&d_Af,     (size_t)M * K * sizeof(float));   /* dequant scratch A */
    cudaMalloc(&d_Bf,     (size_t)K * N * sizeof(float));   /* dequant scratch B */
    cudaMalloc(&d_C_base, (size_t)M * N * sizeof(float));

    /* Fill with non-zero data to prevent trivial DCE */
    cudaMemset(d_A, 0x01, (size_t)M * K * sizeof(int8_t));
    cudaMemset(d_B, 0x01, (size_t)K * N * sizeof(int8_t));

    /* Initialize scales on host then copy */
    float *h_sca = new float[M], *h_scb = new float[N];
    for (int i = 0; i < M; i++) h_sca[i] = 1.0f / 127.0f;
    for (int i = 0; i < N; i++) h_scb[i] = 1.0f;
    cudaMemcpy(d_sca, h_sca, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scb, h_scb, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_sca; delete[] h_scb;

    /* Grid configs */
    const int BLK = 256;
    dim3 dq_grid_a((M * K + BLK - 1) / BLK);
    dim3 dq_grid_b((K * N + BLK - 1) / BLK);
    dim3 sg_block(16, 16);
    dim3 sg_grid((N + 15) / 16, (M + 15) / 16);

    /* Warm-up both paths */
    sp_frob_matmul_q8_mma(d_A, d_B, d_sca, d_scb, d_C_wmma, M, K, N, 0);
    k_dequant_a<<<dq_grid_a, BLK>>>(d_A, d_Af, M * K);
    k_dequant_b<<<dq_grid_b, BLK>>>(d_B, d_Bf, K * N);
    k_sgemm<<<sg_grid, sg_block>>>(d_Af, d_Bf, d_sca, d_scb, d_C_base, M, K, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    /* ── Time WMMA path ── */
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        sp_frob_matmul_q8_mma(d_A, d_B, d_sca, d_scb, d_C_wmma, M, K, N, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_wmma = 0;
    cudaEventElapsedTime(&ms_wmma, t0, t1);

    /* ── Time two-kernel baseline path ── */
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++) {
        k_dequant_a<<<dq_grid_a, BLK>>>(d_A, d_Af, M * K);
        k_dequant_b<<<dq_grid_b, BLK>>>(d_B, d_Bf, K * N);
        k_sgemm<<<sg_grid, sg_block>>>(d_Af, d_Bf, d_sca, d_scb, d_C_base, M, K, N);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_base = 0;
    cudaEventElapsedTime(&ms_base, t0, t1);

    /* Compute metrics */
    double ops_per_rep  = 2.0 * M * K * N;   /* multiply-adds */
    double wmma_tops    = (ops_per_rep * REPS) / (ms_wmma * 1e-3) / 1e12;
    double base_gflops  = (ops_per_rep * REPS) / (ms_base * 1e-3) / 1e9;
    double speedup      = (double)(ms_base / ms_wmma);

    printf("MMA bench: WMMA=%.2f TOPS  baseline=%.2f GFLOPS  speedup=%.1fx\n",
           wmma_tops, base_gflops, speedup);
    printf("M_PTX_2 MMA: %s (target >=3x)\n",
           speedup >= 3.0 ? "PASS" : "NEEDS_TUNING");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_sca); cudaFree(d_scb);
    cudaFree(d_C_wmma); cudaFree(d_Af); cudaFree(d_Bf); cudaFree(d_C_base);
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
    if (!strcmp(filter, "hash") || !strcmp(filter, "all")) {
        if (!gpu) printf("M_PTX_2 HASH: SKIP\n");
        else      bench_hash();
    }
    if (!strcmp(filter, "spinor") || !strcmp(filter, "all")) {
        if (!gpu) printf("M_PTX_2 SPINOR: SKIP\n");
        else      bench_spinor();
    }
    if (!strcmp(filter, "mma") || !strcmp(filter, "all")) {
        if (!gpu) printf("M_PTX_2 MMA: SKIP\n");
        else      bench_mma();
    }

    return 0;
}
