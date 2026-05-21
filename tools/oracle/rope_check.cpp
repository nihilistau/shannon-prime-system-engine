// rope_check.cpp — isolate RoPE: run ggml's NEOX RoPE on a known head vector at
// a chosen position and dump the 128 outputs, so the engine's rope_neox can be
// diffed against it directly. Discriminates "RoPE bug" from "attention bug" for
// the E_CPU_2 pos>0 discrepancy. Links the same stock ggml as dump_logits.
//
//   rope_check.exe <pos>      (default pos=1)
//
// Input fill is fixed (x[i] = 0.01*(i+1)); print the engine side with the same
// fill and compare. freq_base=1e6, freq_scale=1.0, NEOX, n_dims=128 (full).
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    const int n_dims = 128;
    const int pos = argc > 1 ? std::atoi(argv[1]) : 1;

    struct ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
    struct ggml_context *ctx = ggml_init(ip);

    // a: [head_dim, n_head=1, n_tokens=1]; positions b: i32[n_tokens]
    struct ggml_tensor *x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_dims, 1, 1);
    struct ggml_tensor *p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    float *xd = (float *)x->data;
    for (int i = 0; i < n_dims; i++) xd[i] = 0.01f * (float)(i + 1);
    ((int32_t *)p->data)[0] = pos;

    struct ggml_tensor *y = ggml_rope_ext(
        ctx, x, p, nullptr,
        /*n_dims=*/n_dims, /*mode=*/GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0,
        /*freq_base=*/1e6f, /*freq_scale=*/1.0f, /*ext_factor=*/0.0f,
        /*attn_factor=*/1.0f, /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const float *yd = (const float *)y->data;
    std::printf("# ggml NEOX rope, pos=%d, n_dims=%d, base=1e6, scale=1.0\n", pos, n_dims);
    for (int i = 0; i < n_dims; i++) std::printf("%d %.8e\n", i, yd[i]);

    ggml_free(ctx);
    return 0;
}
