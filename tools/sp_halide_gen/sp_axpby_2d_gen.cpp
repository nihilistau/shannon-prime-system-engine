// §3-HX Sprint F Halide AOT generator — 2D fixed-point axpby.
//
//   y[r, c] = saturating_cast<int16_t>((a[c] * x[r, c] + b) >> q_bits)
//
// Schedule (per mandate): .tile(c, r, ci, ri, 128, 4).vectorize(ci).unroll(ri)
// Target: hexagon-32-noos-no_bounds_query-no_asserts-hvx_128
//   (the SDK's canonical "in-process Hexagon" target — the "noos" + "no_bounds_query"
//   bits are required for in-skel use since FastRPC skels don't host the Halide
//   offload runtime; see standalone/simulator/apps/conv3x3a32/test-conv3x3a32.cmd)

#include "Halide.h"

using namespace Halide;

class SpAxpby2d : public Halide::Generator<SpAxpby2d> {
public:
    Input<Buffer<int16_t>>  x{"x", 2};      // shape [cols, rows]
    Input<Buffer<int16_t>>  a{"a", 1};      // shape [cols]
    Input<int32_t>          b{"b"};
    Input<uint8_t>          q_bits{"q_bits"};
    Output<Buffer<int16_t>> y{"y", 2};      // shape [cols, rows]

    Var c{"c"}, r{"r"}, ci{"ci"}, ri{"ri"};

    void generate() {
        Expr ax = cast<int32_t>(a(c)) * cast<int32_t>(x(c, r));
        Expr s  = (ax + b) >> q_bits;
        y(c, r) = saturating_cast<int16_t>(s);
    }

    void schedule() {
        x.dim(0).set_min(0);
        x.dim(1).set_min(0);
        a.dim(0).set_min(0);
        y.dim(0).set_min(0);
        y.dim(1).set_min(0);

        if (get_target().has_feature(Target::HVX)) {
            // HVX vector width: 128 bytes = 64 i16 lanes.
            // Tile to 128×4 = 64 lanes × 2 (= 128 cols) × 4 rows; vectorize the
            // inner column var to 64, unroll the 4 row stripes.
            y.hexagon()
             .tile(c, r, ci, ri, 128, 4)
             .vectorize(ci, 64)
             .unroll(ri);
        } else {
            y.tile(c, r, ci, ri, 64, 4)
             .vectorize(ci, 16)
             .unroll(ri);
        }
    }
};

HALIDE_REGISTER_GENERATOR(SpAxpby2d, sp_axpby_2d);
