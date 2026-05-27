/*
 * ptx_mma_tile_common.cuh — shared constants, smem layout, and load helpers
 * for the 64×64 tiled INT8/INT4 MMA kernels (§17.3.TILE).
 *
 * Architecture: 64×64 block tile / 32×32 warp tile / 4×4 MMA grid / 4 warps.
 * Pipeline:     sm_75 — synchronous (no cp.async); sm_80+ upgrade path guarded
 *               by SP_TILE_SM80 (not implemented this phase).
 *
 * References:
 *   DeepEP ptx.cuh lines 1-10 (fallback macro pattern)
 *   DeepEP utils.cuh lines 177-181 (LD_NC_FUNC / SP_LD_WEIGHT_FUNC)
 *   ptx_spinor.cuh lines 36-50 (ld.global.cg.v4 discipline)
 *   ptx_mma.cuh lines 21-40 (mma_s8_m8n8k16, mma_s4_m8n8k32 — called, not redefined)
 */
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#ifdef __CUDACC__

/* ── Tile geometry ─────────────────────────────────────────────────────────── */
#define SP_TILE_BLOCK_M  64
#define SP_TILE_BLOCK_N  64
#define SP_TILE_K_TILE   32     /* INT8: 32 K-elements per smem stage;   */
                                /* INT4: 32 packed-nibble bytes = 64 nib  */
#define SP_TILE_PAD_B    4      /* extra bytes per smem_b row (bank fix) */
#define SP_TILE_WARPS    4
#define SP_TILE_THREADS  128    /* SP_TILE_WARPS * 32 */

/* Warp 2×2 arrangement: warp_id → (warp_m, warp_n) */
#define SP_TILE_WARP_M(wid) ((wid) >> 1)      /* 0,0,1,1 */
#define SP_TILE_WARP_N(wid) ((wid) & 1)       /* 0,1,0,1 */

/* smem B row pitch including padding */
#define SP_TILE_B_PITCH  (SP_TILE_BLOCK_N + SP_TILE_PAD_B)   /* 68 bytes */

/* ── Weight-side load macro (L1 non-allocating, mirrors DeepEP LD_NC_FUNC) ── */
#ifndef DISABLE_WEIGHT_LD_NC
  #define SP_LD_WEIGHT_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
  #define SP_LD_WEIGHT_FUNC "ld.global.cg"
#endif

/* ── Shared-memory type aliases ────────────────────────────────────────────── */
typedef int8_t  sp_smem_a_t[SP_TILE_BLOCK_M][SP_TILE_K_TILE];
typedef int8_t  sp_smem_b_t[SP_TILE_K_TILE][SP_TILE_B_PITCH];

/* ── Cooperative A-tile load: 128 threads, 16 bytes each ──────────────────── */
/* A is the activation side; use ld.global.cg (L2-cached).                     */
/* Thread T: row = T>>1, col_byte = (T&1)<<4 (0 or 16). 2 threads per row.     */
__device__ __forceinline__
void sp_tile_load_a_int8(
    const int8_t * __restrict__ A_global,
    int8_t smem_a[SP_TILE_BLOCK_M][SP_TILE_K_TILE],
    int block_row, int k_tile,
    int M, int K, int thr_id)
{
    int row      = thr_id >> 1;
    int col_byte = (thr_id & 1) << 4;
    int g_row    = block_row * SP_TILE_BLOCK_M + row;
    int g_col    = k_tile   * SP_TILE_K_TILE  + col_byte;
    uint32_t v0, v1, v2, v3;
    if (g_row < M && (g_col + 15) < K) {
        const uint32_t *p = (const uint32_t *)(A_global + g_row * K + g_col);
        asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "l"(p));
    } else {
        v0 = v1 = v2 = v3 = 0;
    }
    uint32_t *s = (uint32_t *)(smem_a[row] + col_byte);
    s[0] = v0; s[1] = v1; s[2] = v2; s[3] = v3;
}

/* ── Cooperative B-tile load: 128 threads, 16 bytes each ──────────────────── */
/* B is the weight side; use SP_LD_WEIGHT_FUNC (L1 non-allocating).             */
__device__ __forceinline__
void sp_tile_load_b_int8(
    const int8_t * __restrict__ B_global,
    int8_t smem_b[SP_TILE_K_TILE][SP_TILE_B_PITCH],
    int block_col, int k_tile,
    int K, int N, int thr_id)
{
    int row      = thr_id >> 1;
    int col_byte = (thr_id & 1) << 4;
    int g_row    = k_tile    * SP_TILE_K_TILE  + row;
    int g_col    = block_col * SP_TILE_BLOCK_N + col_byte;
    uint32_t v0, v1, v2, v3;
    if (g_row < K && (g_col + 15) < N) {
        const uint32_t *p = (const uint32_t *)(B_global + g_row * N + g_col);
        asm volatile (SP_LD_WEIGHT_FUNC ".v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "l"(p));
    } else {
        v0 = v1 = v2 = v3 = 0;
    }
    uint32_t *s = (uint32_t *)(smem_b[row] + col_byte);
    s[0] = v0; s[1] = v1; s[2] = v2; s[3] = v3;
}

/* INT4 variants (packed nibble — same byte width, different element count) */
__device__ __forceinline__
void sp_tile_load_a_int4(
    const uint8_t * __restrict__ A_q4,
    uint8_t smem_a[SP_TILE_BLOCK_M][SP_TILE_K_TILE],
    int block_row, int k_tile,
    int M, int K_bytes, int thr_id)
{
    int row      = thr_id >> 1;
    int col_byte = (thr_id & 1) << 4;
    int g_row    = block_row * SP_TILE_BLOCK_M + row;
    int g_col    = k_tile   * SP_TILE_K_TILE  + col_byte;
    uint32_t v0, v1, v2, v3;
    if (g_row < M && (g_col + 15) < K_bytes) {
        const uint32_t *p = (const uint32_t *)(A_q4 + g_row * K_bytes + g_col);
        asm volatile ("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "l"(p));
    } else {
        v0 = v1 = v2 = v3 = 0;
    }
    uint32_t *s = (uint32_t *)(smem_a[row] + col_byte);
    s[0] = v0; s[1] = v1; s[2] = v2; s[3] = v3;
}

__device__ __forceinline__
void sp_tile_load_b_int4(
    const uint8_t * __restrict__ B_q4,
    uint8_t smem_b[SP_TILE_K_TILE][SP_TILE_B_PITCH],
    int block_col, int k_tile,
    int K_bytes, int N, int thr_id)
{
    int row      = thr_id >> 1;
    int col_byte = (thr_id & 1) << 4;
    int g_row    = k_tile    * SP_TILE_K_TILE  + row;
    int g_col    = block_col * SP_TILE_BLOCK_N + col_byte;
    uint32_t v0, v1, v2, v3;
    if (g_row < K_bytes && (g_col + 15) < N) {
        const uint32_t *p = (const uint32_t *)(B_q4 + g_row * N + g_col);
        asm volatile (SP_LD_WEIGHT_FUNC ".v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "l"(p));
    } else {
        v0 = v1 = v2 = v3 = 0;
    }
    uint32_t *s = (uint32_t *)(smem_b[row] + col_byte);
    s[0] = v0; s[1] = v1; s[2] = v2; s[3] = v3;
}

/* ── Fragment load from smem to register ──────────────────────────────────── */
__device__ __forceinline__
uint32_t sp_tile_frag_a_int8(
    const int8_t smem_a[SP_TILE_BLOCK_M][SP_TILE_K_TILE],
    int warp_m, int mma_m, int mma_k, int lane);

__device__ __forceinline__
uint32_t sp_tile_frag_b_int8(
    const int8_t smem_b[SP_TILE_K_TILE][SP_TILE_B_PITCH],
    int warp_n, int mma_n, int mma_k, int lane);

__device__ __forceinline__
uint32_t sp_tile_frag_a_int4(
    const uint8_t smem_a[SP_TILE_BLOCK_M][SP_TILE_K_TILE],
    int warp_m, int mma_m, int mma_k, int lane);

__device__ __forceinline__
uint32_t sp_tile_frag_b_int4(
    const uint8_t smem_b[SP_TILE_K_TILE][SP_TILE_B_PITCH],
    int warp_n, int mma_n, int mma_k, int lane);

/* ── Scale epilogue helper ─────────────────────────────────────────────────── */
/* acc[8][2]: row-major 4×4 MMA grid accumulators for one mma_m × mma_n step.  */
/* Writes scaled FP16 to C_global.                                              */
__device__ __forceinline__
void sp_tile_epilogue_mma(
    const int acc_d0, const int acc_d1,
    __half * __restrict__ C_global,
    int out_row, int out_col,
    float row_scale, int lane,
    int M, int N);

#endif /* __CUDACC__ */
