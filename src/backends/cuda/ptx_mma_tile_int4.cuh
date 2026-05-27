/*
 * ptx_mma_tile_int4.cuh — 64×64 tiled INT4 Q4 matmul for §17.3.TILE.
 *
 * Block tile:  64(M) × 64(N), K_TILE = 32 packed-nibble bytes (= 64 nibbles)
 * Warp tile:   32(M) × 32(N) via 4×4 grid of m8n8k32 MMA ops
 * Warps/block: 4 in 2×2 arrangement
 * Pipeline:    sm_75 synchronous (same structure as INT8 tile kernel)
 *
 * A_q4 / B_q4 layout: uint8_t [M][K/2] / uint8_t [K/2][N]
 *   low nibble  = even K index (k*2+0)
 *   high nibble = odd  K index (k*2+1)
 *   K is nibble count (must be even; multiple of 64 for full tiles)
 *
 * Calls mma_s4_m8n8k32() from ptx_mma.cuh — no wmma, no <mma.h>.
 * smem byte layout identical to INT8; fragment formula identical.
 */
#pragma once
#include "ptx_mma_tile_common.cuh"
#include "ptx_mma.cuh"

#ifdef __CUDACC__

__global__ void sp_frob_matmul_q4_tile_kernel(
    const uint8_t * __restrict__ A_q4,
    const uint8_t * __restrict__ B_q4,
    const float   * __restrict__ scale_a,
    const float   * __restrict__ scale_b,
    __half        * __restrict__ C,
    int M, int K, int N)      /* K = nibble count */
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    /* K_bytes = K/2: stride for load addressing; SP_TILE_K_TILE is byte-width */
    const int K_bytes = K >> 1;

    __shared__ uint8_t smem_a[SP_TILE_BLOCK_M][SP_TILE_K_TILE];
    __shared__ uint8_t smem_b[SP_TILE_K_TILE][SP_TILE_B_PITCH];

    const int thr   = (int)threadIdx.x;
    const int warp  = thr >> 5;
    const int lane  = thr & 31;
    const int wm    = SP_TILE_WARP_M(warp);
    const int wn    = SP_TILE_WARP_N(warp);
    const int brow  = (int)blockIdx.y;
    const int bcol  = (int)blockIdx.x;

    /* 32 INT32 accumulators */
    int acc[4][4][2];
    #pragma unroll
    for (int mm = 0; mm < 4; mm++)
        #pragma unroll
        for (int mn = 0; mn < 4; mn++)
            acc[mm][mn][0] = acc[mm][mn][1] = 0;

    /* K_tiles: each tile is SP_TILE_K_TILE bytes = 64 nibbles */
    const int nk = (K_bytes + SP_TILE_K_TILE - 1) / SP_TILE_K_TILE;
    for (int kt = 0; kt < nk; kt++) {
        sp_tile_load_a_int4(A_q4, smem_a, brow, kt, M, K_bytes, thr);
        sp_tile_load_b_int4(B_q4, smem_b, bcol, kt, K_bytes, N,      thr);
        __syncthreads();

        /* Preload fragments (formula identical to INT8; hardware reads nibbles) */
        uint32_t a_frag[2][4];
        uint32_t b_frag[2][4];
        #pragma unroll
        for (int mk = 0; mk < 2; mk++) {
            #pragma unroll
            for (int mm = 0; mm < 4; mm++)
                a_frag[mk][mm] = sp_tile_frag_a_int4(smem_a, wm, mm, mk, lane);
            #pragma unroll
            for (int mn = 0; mn < 4; mn++)
                b_frag[mk][mn] = sp_tile_frag_b_int4(smem_b, wn, mn, mk, lane);
        }
        /* 32 mma_s4_m8n8k32 calls per warp per K-tile */
        #pragma unroll
        for (int mk = 0; mk < 2; mk++)
            #pragma unroll
            for (int mm = 0; mm < 4; mm++)
                #pragma unroll
                for (int mn = 0; mn < 4; mn++)
                    mma_s4_m8n8k32(&acc[mm][mn][0], &acc[mm][mn][1],
                                   a_frag[mk][mm], b_frag[mk][mn],
                                   acc[mm][mn][0], acc[mm][mn][1]);

        __syncthreads();
    }

    /* Epilogue */
    #pragma unroll
    for (int mm = 0; mm < 4; mm++) {
        int out_row = brow * SP_TILE_BLOCK_M + wm * 32 + mm * 8;
        #pragma unroll
        for (int mn = 0; mn < 4; mn++) {
            int out_col = bcol * SP_TILE_BLOCK_N + wn * 32 + mn * 8;
            sp_tile_epilogue_mma(acc[mm][mn][0], acc[mm][mn][1],
                                 C, scale_a, scale_b,
                                 out_row, out_col, lane, M, N);
        }
    }
#endif /* __CUDA_ARCH__ >= 750 */
}

/*
 * sp_frob_matmul_q4_mma_tile — host launcher.
 * K is nibble count (A_q4 is uint8[M][K/2], B_q4 is uint8[K/2][N]).
 */
inline cudaError_t sp_frob_matmul_q4_mma_tile(
    const uint8_t *A_q4,
    const uint8_t *B_q4,
    const float   *scale_a,
    const float   *scale_b,
    __half        *C,
    int M, int K, int N,
    cudaStream_t   stream)
{
    dim3 grid((unsigned)((N + SP_TILE_BLOCK_N - 1) / SP_TILE_BLOCK_N),
              (unsigned)((M + SP_TILE_BLOCK_M - 1) / SP_TILE_BLOCK_M));
    dim3 block(SP_TILE_THREADS);
    sp_frob_matmul_q4_tile_kernel<<<grid, block, 0, stream>>>(
        A_q4, B_q4, scale_a, scale_b, C, M, K, N);
    return cudaGetLastError();
}

#endif /* __CUDACC__ */
