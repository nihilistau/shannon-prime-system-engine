/*
 * ptx_mma.cuh — INT8 WMMA Q8 matmul for Shannon-Prime Phase 2-CU.PTX Task 4.
 *
 * Uses WMMA C++ API (nvcuda::wmma) with m16n16k16 INT8 fragments.
 *
 * NOTE: The WMMA C++ API on sm_75 (Turing) exposes INT8 only at tile shapes
 * (16,16,16), (32,8,16), (8,32,16). The m8n8k16 shape exists at the PTX
 * mma.sync instruction level but the nvcuda::wmma fragment template does NOT
 * instantiate it on sm_75 under CUDA 13.2 (compilation error: incomplete type).
 * We use m16n16k16 — semantically equivalent for the Q8 matmul gate and
 * directly supported.  MMA_M/N/K constants are updated accordingly.
 *
 * Gate coverage:
 *   M_PTX_1 MMA: correctness  (Test A all-ones, Test B random Q8)
 *   M_PTX_2 MMA: throughput   (target >=3x vs naive decode+f32 matmul)
 *   M_PTX_3 MMA: no cudaMalloc in hot path (only __shared__ smem used)
 *   M_PTX_4 MMA: two independent streams, no global sync
 *
 * DO NOT add softmax / temperature / probability code.  Discrete argmax only.
 */
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <mma.h>

/* Tile dimensions for m16n16k16 INT8 WMMA */
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16

using namespace nvcuda;

/* ── Forward declaration (needed by host pass of nvcc) ─────────────────── */
__global__ void sp_frob_matmul_q8_mma_kernel(
    const int8_t  *__restrict__ A,
    const int8_t  *__restrict__ B,
    const float   *__restrict__ scale_a,
    const float   *__restrict__ scale_b,
    __half        *__restrict__ C,
    int M, int K, int N);

/* ── Launcher (host-callable) ──────────────────────────────────────────── */
/*
 * sp_frob_matmul_q8_mma
 *
 * Launches sp_frob_matmul_q8_mma_kernel on the provided stream.
 * M_PTX_3: no cudaMalloc inside; kernel uses only __shared__ smem.
 * M_PTX_4: caller provides per-session stream; no implicit global sync.
 *
 * Grid:  (ceil(N/MMA_N), ceil(M/MMA_M))  — one warp per 16x16 output tile.
 * Block: 32 threads (one warp).
 */
inline void sp_frob_matmul_q8_mma(
    const int8_t  *A,
    const int8_t  *B,
    const float   *scale_a,
    const float   *scale_b,
    __half        *C,
    int M, int K, int N,
    cudaStream_t   stream)
{
    dim3 grid((unsigned)((N + MMA_N - 1) / MMA_N),
              (unsigned)((M + MMA_M - 1) / MMA_M));
    dim3 block(32);   /* one warp per tile */
    sp_frob_matmul_q8_mma_kernel<<<grid, block, 0, stream>>>(
        A, B, scale_a, scale_b, C, M, K, N);
}

/* ── Kernel definition (device code, sm_75+) ───────────────────────────── */
/*
 * sp_frob_matmul_q8_mma_kernel
 *
 * Computes:
 *   C[i,j] = __float2half((float)sum_k(A[i,k]*B[k,j]) * scale_a[i] * scale_b[j])
 *
 * A: int8[M][K]  row-major  (stride = K)
 * B: int8[K][N]  row-major  (stride = N)
 * scale_a: float[M]   per-row  dequant scale
 * scale_b: float[N]   per-col  dequant scale
 * C: __half[M][N]  row-major
 *
 * Each block = one warp (32 threads).
 * Grid = (ceil(N/16), ceil(M/16)).
 *
 * M_PTX_3: only __shared__ smem used; no cudaMalloc in hot path.
 */
__global__ void sp_frob_matmul_q8_mma_kernel(
    const int8_t  *__restrict__ A,
    const int8_t  *__restrict__ B,
    const float   *__restrict__ scale_a,
    const float   *__restrict__ scale_b,
    __half        *__restrict__ C,
    int M, int K, int N)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    int warp_m = (int)blockIdx.y;
    int warp_n = (int)blockIdx.x;

    int out_row = warp_m * MMA_M;   /* first output row of this tile */
    int out_col = warp_n * MMA_N;   /* first output col of this tile */

    wmma::fragment<wmma::matrix_a,    MMA_M, MMA_N, MMA_K,
                   signed char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    MMA_M, MMA_N, MMA_K,
                   signed char, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, int32_t> c_frag;

    wmma::fill_fragment(c_frag, 0);

    /* Shared pads for boundary tiles (only used when M/K/N are not multiples
     * of MMA_M/MMA_K/MMA_N). For fully-aligned inputs the branches fall
     * through to the direct load_matrix_sync path. */
    __shared__ int8_t  smem_a[MMA_M * MMA_K];   /* 16*16 = 256 B */
    __shared__ int8_t  smem_b[MMA_K * MMA_N];   /* 16*16 = 256 B */
    __shared__ int32_t smem_c[MMA_M * MMA_N];   /* 16*16*4 = 1 KB */

    int lane = (int)threadIdx.x;

    for (int k = 0; k < K; k += MMA_K) {
        /* ── Load A tile ────────────────────────────────────────── */
        if (out_row + MMA_M <= M && k + MMA_K <= K) {
            /* Aligned: load directly from global memory */
            wmma::load_matrix_sync(a_frag,
                                   A + out_row * K + k,
                                   (unsigned)K);
        } else {
            /* Boundary: zero-pad into shared memory */
            for (int e = lane; e < MMA_M * MMA_K; e += 32) {
                int r    = e / MMA_K;
                int kcol = e % MMA_K;
                int gr   = out_row + r;
                int gk   = k + kcol;
                smem_a[e] = (gr < M && gk < K) ? A[gr * K + gk] : (int8_t)0;
            }
            __syncwarp();
            wmma::load_matrix_sync(a_frag, smem_a, MMA_K);
        }

        /* ── Load B tile ────────────────────────────────────────── */
        if (k + MMA_K <= K && out_col + MMA_N <= N) {
            wmma::load_matrix_sync(b_frag,
                                   B + k * N + out_col,
                                   (unsigned)N);
        } else {
            for (int e = lane; e < MMA_K * MMA_N; e += 32) {
                int krow = e / MMA_N;
                int c_off = e % MMA_N;
                int gk   = k + krow;
                int gc   = out_col + c_off;
                smem_b[e] = (gk < K && gc < N) ? B[gk * N + gc] : (int8_t)0;
            }
            __syncwarp();
            wmma::load_matrix_sync(b_frag, smem_b, MMA_N);
        }

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    /* Store accumulator → shared, then scatter-write to C with scales */
    wmma::store_matrix_sync(smem_c, c_frag, MMA_N, wmma::mem_row_major);
    __syncwarp();

    /* 256 elements / 32 threads = 8 elements per thread */
    for (int e = lane; e < MMA_M * MMA_N; e += 32) {
        int r     = e / MMA_N;
        int c_off = e % MMA_N;
        int row   = out_row + r;
        int col   = out_col + c_off;
        if (row < M && col < N) {
            float v = (float)smem_c[r * MMA_N + c_off]
                      * scale_a[row] * scale_b[col];
            C[row * N + col] = __float2half(v);
        }
    }
#endif /* __CUDA_ARCH__ >= 750 */
}

/* ── Scalar fallback (host reference / sm < 75 CI path) ───────────────── */
/*
 * sp_frob_matmul_q8_ref — host-only scalar reference.
 * Exactly matches the GPU kernel's fp ops order for bit-accurate comparison.
 * Used by ptx_validate Test B.
 */
inline void sp_frob_matmul_q8_ref(
    const int8_t *A,
    const int8_t *B,
    const float  *scale_a,
    const float  *scale_b,
    __half       *C,
    int M, int K, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            float v = (float)acc * scale_a[i] * scale_b[j];
            C[i * N + j] = __float2half(v);
        }
    }
}

#endif /* __CUDACC__ */
