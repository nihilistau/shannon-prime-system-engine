/*
 * ptx_mma_tile_validate.cu — M_PTX_MMA_TILE_1 three-way correctness sweep.
 *
 * Three-way bit-exact check per shape:
 *   1. sp_frob_matmul_q{8,4}_ref      (scalar host reference)
 *   2. sp_frob_matmul_q{8,4}_mma      (single-instruction naive GPU kernel)
 *   3. sp_frob_matmul_q{8,4}_mma_tile (64x64 tiled GPU kernel)
 *
 * Shape sweep: (64,64,64), (256,256,256), (1024,1024,1024), (3072,3072,8192)
 * Exit code: 0 = all pass, 1 = any mismatch.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include "ptx_mma_tile_int8.cuh"
#include "ptx_mma_tile_int4.cuh"

#ifndef __CUDACC__
int main() { printf("SKIP (not a CUDA build)\n"); return 0; }
#else

/* ── Test helpers ──────────────────────────────────────────────────────────── */

static uint32_t lcg(uint32_t &s) { s = s * 1664525u + 1013904223u; return s; }

static void fill_int8(int8_t *p, int n, uint32_t &seed) {
    for (int i = 0; i < n; i++) p[i] = (int8_t)((lcg(seed) & 0xFF) - 128);
}

static void pack_nibbles(const int8_t *src, uint8_t *dst, int n_nibbles) {
    /* src[k] in [-7,7], pack to low/high nibble pairs */
    for (int k = 0; k < n_nibbles; k++) {
        int8_t v = src[k];
        if (v > 7) v = 7; if (v < -7) v = -7;
        unsigned nib = (unsigned)(v < 0 ? (v + 16) : v) & 0xFu;
        if (k & 1) dst[k >> 1] = (uint8_t)((dst[k >> 1] & 0x0Fu) | (nib << 4));
        else       dst[k >> 1] = (uint8_t)((dst[k >> 1] & 0xF0u) | nib);
    }
}

static bool half_match(const __half *a, const __half *b, int n, float *max_diff) {
    *max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(__half2float(a[i]) - __half2float(b[i]));
        if (diff > *max_diff) *max_diff = diff;
    }
    return *max_diff == 0.0f;
}

#define CUDA_CHECK(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return 1; \
    } \
} while (0)

/* ── Per-shape INT8 test ───────────────────────────────────────────────────── */
static int test_int8(int M, int K, int N, const char *tag) {
    size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;

    int8_t  *hA = new int8_t[szA], *hB = new int8_t[szB];
    float   *hSA = new float[M], *hSB = new float[N];
    __half  *hRef = new __half[szC], *hNaive = new __half[szC], *hTile = new __half[szC];

    uint32_t seed = 0xDEAD0000u ^ (uint32_t)(M * 31 + K * 17 + N * 7);
    fill_int8(hA, (int)szA, seed);
    fill_int8(hB, (int)szB, seed);
    for (int i = 0; i < M; i++) hSA[i] = 0.5f + (lcg(seed) & 0xFFu) * (1.0f / 512.0f);
    for (int j = 0; j < N; j++) hSB[j] = 0.5f + (lcg(seed) & 0xFFu) * (1.0f / 512.0f);

    /* 1. Scalar host reference */
    sp_frob_matmul_q8_ref(hA, hB, hSA, hSB, hRef, M, K, N);

    /* 2. Naive GPU kernel */
    int8_t *dA, *dB; float *dSA, *dSB; __half *dC;
    CUDA_CHECK(cudaMalloc(&dA,  szA * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dB,  szB * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dSA, M   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSB, N   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC,  szC * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dA,  hA,  szA * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,  hB,  szB * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSA, hSA, M   * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSB, hSB, N   * sizeof(float),  cudaMemcpyHostToDevice));

    sp_frob_matmul_q8_mma(dA, dB, dSA, dSB, dC, M, K, N, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hNaive, dC, szC * sizeof(__half), cudaMemcpyDeviceToHost));

    /* 3. Tiled kernel */
    CUDA_CHECK(cudaMemset(dC, 0, szC * sizeof(__half)));
    CUDA_CHECK(sp_frob_matmul_q8_mma_tile(dA, dB, dSA, dSB, dC, M, K, N, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hTile, dC, szC * sizeof(__half), cudaMemcpyDeviceToHost));

    float d_ref_naive, d_ref_tile, d_naive_tile;
    bool ok_rn = half_match(hRef,   hNaive, (int)szC, &d_ref_naive);
    bool ok_rt = half_match(hRef,   hTile,  (int)szC, &d_ref_tile);
    bool ok_nt = half_match(hNaive, hTile,  (int)szC, &d_naive_tile);
    bool pass  = ok_rn && ok_rt && ok_nt;
    printf("  INT8 %s (%dx%dx%d): ref-naive=%.4g ref-tile=%.4g naive-tile=%.4g [%s]\n",
           tag, M, K, N, d_ref_naive, d_ref_tile, d_naive_tile, pass ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hSA; delete[] hSB;
    delete[] hRef; delete[] hNaive; delete[] hTile;
    return pass ? 0 : 1;
}

/* ── Per-shape INT4 test — declaration (filled in edit) ───────────────────── */
static int test_int4(int M, int K, int N, const char *tag);

/* ── Main ──────────────────────────────────────────────────────────────────── */
int main() {
    int dev; if (cudaGetDevice(&dev) != cudaSuccess) { printf("SKIP (no GPU)\n"); return 0; }

    static const struct { int M, K, N; const char *tag; } shapes[] = {
        {  64,   64,   64, "tiny"    },
        { 256,  256,  256, "small"   },
        {1024, 1024, 1024, "medium"  },
        {3072, 8192, 3072, "qwen3-ffn"},
    };
    const int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));

    int fail = 0;
    printf("=== M_PTX_MMA_TILE_1: INT8 ===\n");
    for (int i = 0; i < ns; i++) fail |= test_int8(shapes[i].M, shapes[i].K, shapes[i].N, shapes[i].tag);
    printf("=== M_PTX_MMA_TILE_1: INT4 ===\n");
    for (int i = 0; i < ns; i++) fail |= test_int4(shapes[i].M, shapes[i].K, shapes[i].N, shapes[i].tag);

    printf("\n%s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}
