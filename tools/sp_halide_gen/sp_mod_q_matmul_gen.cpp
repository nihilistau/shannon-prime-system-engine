// §3-HX Sprint K v0.beta Stage 2 — Barrett-reduction mod-q matmul generator.
//
// Probe purpose: emit AOT with Halide's NATIVE Int(64) lowering on HVX.
// Dump SASS via hexagon-llvm-objdump; inspect for vectorization quality.
//
// If clean vector lowering: this generator becomes the Stage 4 production kernel.
// If scalar fallback or paired-register breakage: pivot to explicit
// mul_hi + mul_lo decomposition per reference-nvcc-paired-register-bug.
//
// Math per output element (c, b):
//   acc_i64 = Σ_{rd=0..D_in} mod_q(X(rd, b)) × mod_q(W(rd, c))
//   r = barrett_reduce(acc, q, μ) ∈ [0, q)
//
// q + μ baked at compile time (Phase 2-CU.PTX precedent at engine 63d7e2d):
//   q_1 = 1073738753, MU_Q1 = 1073744895 = floor(2^60 / q_1)
//   q_2 = 1073732609, MU_Q2 = 1073751039 = floor(2^60 / q_2)
//
// IDL convention: X is [d_in, batch], W is [d_in, h_dim] (matches Sprint H/J).
// Output R is [h_dim, batch] i32.

#include "Halide.h"

using namespace Halide;

class SpModQMatmul : public Halide::Generator<SpModQMatmul> {
public:
    Input<Buffer<int16_t>>  X{"X", 2};   // [d_in, batch]
    Input<Buffer<int16_t>>  W{"W", 2};   // [d_in, h_dim]
    Output<Buffer<int32_t>> R{"R", 2};   // [h_dim, batch] i32 residue in [0, q)

    // Prime + Barrett μ baked at compile time.  Default q_1; override via
    // generator args at AOT time (build.cmd will pass q_2 for the second emit).
    GeneratorParam<int64_t> Q  {"Q",  1073738753};   // q_1 by default
    GeneratorParam<int64_t> MU {"MU", 1073744895};   // floor(2^60 / q_1)

    Var c{"c"}, batch{"batch"}, ci{"ci"};

    void generate() {
        Expr D_in = X.dim(0).extent();

        // Map signed i16 to canonical [0, q) residue.  Cast through i32 first
        // (i16 widening preserves sign), then add q if negative.
        Expr q_i64  = Expr(static_cast<int64_t>(int64_t(Q)));
        Expr mu_i64 = Expr(static_cast<int64_t>(int64_t(MU)));
        Expr q_i32  = Expr(static_cast<int32_t>(int64_t(Q)));

        RDom rd(0, D_in);
        Expr xv32 = cast<int32_t>(X(rd, batch));
        Expr wv32 = cast<int32_t>(W(rd, c));
        Expr xv = cast<int64_t>(select(xv32 < 0, xv32 + q_i32, xv32));
        Expr wv = cast<int64_t>(select(wv32 < 0, wv32 + q_i32, wv32));

        // Accumulate i64 products via Halide's sum() reduction.
        Expr acc = sum(xv * wv);

        // Barrett reduce: q_est = (acc * μ) >> 60; r = acc - q_est * q
        Expr q_est = (acc * mu_i64) >> 60;
        Expr r1 = acc - q_est * q_i64;
        // r1 may be in [0, 2q); canonicalize to [0, q).
        Expr r2 = select(r1 >= q_i64, r1 - q_i64, r1);
        R(c, batch) = cast<int32_t>(r2);
    }

    void schedule() {
        X.dim(0).set_min(0); X.dim(1).set_min(0);
        W.dim(0).set_min(0); W.dim(1).set_min(0);
        R.dim(0).set_min(0); R.dim(1).set_min(0);

        X.set_host_alignment(128);
        W.set_host_alignment(128);
        R.set_host_alignment(128);

        if (get_target().has_feature(Target::HVX)) {
            // Mirror Sprint J's working schedule shape: vectorize the inner
            // output dim, offload to Hexagon, prefetch X.  Inner vector width
            // 32 i32 lanes (128 bytes / 4) — smaller than Sprint J's 64 i16
            // because the output is i32 not i16.
            R.hexagon()
             .vectorize(c, 32)
             .prefetch(X, batch, 2);
        } else {
            R.vectorize(c, 8);
        }
    }
};

HALIDE_REGISTER_GENERATOR(SpModQMatmul, sp_mod_q_matmul);
