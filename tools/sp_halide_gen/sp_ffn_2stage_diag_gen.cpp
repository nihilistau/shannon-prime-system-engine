// §3-HX Sprint H diagnostic generator — clone of sp_ffn_2stage_gen.cpp with
// `hidden` exposed as a second Output buffer.  Used by T_HALIDE_FFN_DIAG_INSTRUMENT
// to compare Halide's mid-pipeline hidden values against the scalar reference,
// isolating matmul-1 vs matmul-2 as the divergence site for Sprint G's G.1 cases.
//
// Math is byte-identical to the Sprint G generator; only the Output surface
// differs.  Schedule is intentionally identical so codegen reproduces.

#include "Halide.h"

using namespace Halide;

class SpFfn2StageDiag : public Halide::Generator<SpFfn2StageDiag> {
public:
    Input<Buffer<int16_t>>  X{"X", 2};      // [D_in, B]
    Input<Buffer<int16_t>>  W1{"W1", 2};    // [D_in, H]
    Input<Buffer<int16_t>>  W2{"W2", 2};    // [H, D_out]
    Input<int32_t>          b_term{"b_term"};
    Input<uint8_t>          q_bits{"q_bits"};
    Output<Buffer<int16_t>> Y{"Y", 2};           // [D_out, B]
    Output<Buffer<int16_t>> hidden_out{"hidden_out", 2};  // [H, B] — tees the intermediate

    Var c{"c"}, batch{"batch"}, ci{"ci"}, bi{"bi"};
    Var hc{"hc"};
    Func mm1{"mm1"}, hidden{"hidden"}, mm2{"mm2"};

    void generate() {
        Expr D_in  = X.dim(0).extent();
        Expr H_dim = W1.dim(1).extent();

        RDom rd(0, D_in);
        mm1(hc, batch) = cast<int32_t>(0);
        mm1(hc, batch) += cast<int32_t>(X(rd, batch)) *
                          cast<int32_t>(W1(rd, hc));
        hidden(hc, batch) = cast<int16_t>(
            clamp((mm1(hc, batch) + b_term) >> q_bits,
                  cast<int32_t>(0), cast<int32_t>(32767)));

        // Tee hidden to a caller-visible Output.  Bounds-inferred from H_dim.
        hidden_out(hc, batch) = hidden(hc, batch);

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
        hidden_out.dim(0).set_min(0); hidden_out.dim(1).set_min(0);

        X.set_host_alignment(128);
        W1.set_host_alignment(128);
        W2.set_host_alignment(128);
        Y.set_host_alignment(128);
        hidden_out.set_host_alignment(128);

        if (get_target().has_feature(Target::HVX)) {
            // Schedule identical to Sprint G's sp_ffn_2stage_gen so codegen
            // reproduces the Sprint G failure surface faithfully.
            Y.hexagon()
             .vectorize(c, 64)
             .prefetch(X, batch, 2);

            mm1.compute_root();
            hidden.compute_root().store_in(MemoryType::VTCM);
            mm2.compute_at(Y, batch);

            // hidden_out lifts the intermediate to its own buffer; Halide
            // recognizes the identity hidden_out(hc, b) = hidden(hc, b) and
            // schedules accordingly.  compute_root keeps it independent from
            // mm2's per-batch placement.
            hidden_out.compute_root().vectorize(hc, 64);
        } else {
            Y.tile(c, batch, ci, bi, 64, 4)
             .vectorize(ci, 16)
             .unroll(bi);
            hidden.compute_at(Y, batch);
            hidden_out.compute_root();
        }
    }
};

HALIDE_REGISTER_GENERATOR(SpFfn2StageDiag, sp_ffn_2stage_diag);
