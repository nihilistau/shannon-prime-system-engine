// §3-HX Sprint G Halide AOT generator — 2-stage FFN slice with VTCM staging.
//
// Math (per-batch row, per-output-column):
//   hidden[b, h] = clamp((Σ_d X[b, d] * W1[h, d] + b_term) >> q_bits, 0, 32767)
//   Y[b, c]      = saturating_cast<i16>((Σ_h hidden[b, h] * W2[c, h] + b_term) >> q_bits)
//
// Shapes (Halide convention: dim 0 = innermost / fastest-varying = column):
//   X:      [D_in,  B]
//   W1:     [D_in,  H]      (W1[h, d] in math → W1(d, h) in Halide; h indexed by dim 1)
//   W2:     [H,     D_out]
//   hidden: [H,     B]      (intermediate, stored in VTCM)
//   Y:      [D_out, B]
//
// The .store_in(MemoryType::VTCM) on `hidden` is the canonical Halide+VTCM
// pattern — Halide's runtime allocates VTCM internally for the intermediate.
// Combined with Sprint F.1's external-VTCM hot-copy for X/W1/W2/Y inputs and
// outputs, this is the dual-VTCM dance the Sprint G mandate calls for.

#include "Halide.h"

using namespace Halide;

class SpFfn2Stage : public Halide::Generator<SpFfn2Stage> {
public:
    Input<Buffer<int16_t>>  X{"X", 2};      // [D_in, B]
    Input<Buffer<int16_t>>  W1{"W1", 2};    // [D_in, H]
    Input<Buffer<int16_t>>  W2{"W2", 2};    // [H, D_out]
    Input<int32_t>          b_term{"b_term"};
    Input<uint8_t>          q_bits{"q_bits"};
    Output<Buffer<int16_t>> Y{"Y", 2};      // [D_out, B]

    Var c{"c"}, batch{"batch"}, ci{"ci"}, bi{"bi"};
    Var hc{"hc"};
    Func mm1{"mm1"}, hidden{"hidden"}, mm2{"mm2"};

    void generate() {
        Expr D_in  = X.dim(0).extent();
        Expr H_dim = W1.dim(1).extent();

        // Stage 1: explicit reduction via update() instead of sum().
        // Initial: mm1(hc, batch) = 0.  Update: += X(rd, batch) * W1(rd, hc).
        // This makes the reduction's RDom EXPLICITLY tied to D_in, separate from
        // the bound-propagation chain that infers hidden's dim 0 extent.
        RDom rd(0, D_in);
        mm1(hc, batch) = cast<int32_t>(0);
        mm1(hc, batch) += cast<int32_t>(X(rd, batch)) *
                          cast<int32_t>(W1(rd, hc));
        hidden(hc, batch) = cast<int16_t>(
            clamp((mm1(hc, batch) + b_term) >> q_bits,
                  cast<int32_t>(0), cast<int32_t>(32767)));

        // Stage 2: explicit reduction for the second matmul.  Bounds for `c` here
        // come from Y's output extent (D_out); bounds for rh come from H_dim.
        RDom rh(0, H_dim);
        mm2(c, batch) = cast<int32_t>(0);
        mm2(c, batch) += cast<int32_t>(hidden(rh, batch)) *
                         cast<int32_t>(W2(rh, c));
        Y(c, batch) = saturating_cast<int16_t>(
            (mm2(c, batch) + b_term) >> q_bits);
    }

    void schedule() {
        X.dim(0).set_min(0);  X.dim(1).set_min(0);
        W1.dim(0).set_min(0); W1.dim(1).set_min(0);
        W2.dim(0).set_min(0); W2.dim(1).set_min(0);
        Y.dim(0).set_min(0);  Y.dim(1).set_min(0);

        // Sprint F.1 finding: alignment hint doesn't currently flip codegen
        // to vmem, but declare it anyway — the runtime will at least assert
        // the pointers ARE 128-aligned, catching a wider class of caller bugs.
        X.set_host_alignment(128);
        W1.set_host_alignment(128);
        W2.set_host_alignment(128);
        Y.set_host_alignment(128);

        if (get_target().has_feature(Target::HVX)) {
            // Simpler schedule: vectorize Y's inner c by 64, no tile/unroll —
            // gives more outer-iteration coverage without the multi-tile codegen
            // pattern that diverged in earlier ablations.
            Y.hexagon()
             .vectorize(c, 64)
             .prefetch(X, batch, 2);

            mm1.compute_root();
            hidden.compute_root();
            mm2.compute_at(Y, batch);
        } else {
            Y.tile(c, batch, ci, bi, 64, 4)
             .vectorize(ci, 16)
             .unroll(bi);
            hidden.compute_at(Y, batch);
        }
    }
};

HALIDE_REGISTER_GENERATOR(SpFfn2Stage, sp_ffn_2stage);
