//! Byte-exact NONLINEAR islands — host-runnable scalar reference (the universal
//! definition the C/CUDA/HVX backends gate against). The crate already owns the
//! LINEAR algebra byte-exact (sp_matmul_q_ref + Garner CRT + the NTT ladder);
//! these are the three fp32-island replacements PPT-ARM-System §8 names —
//! RMSNorm / softmax / GELU-tanh — as exact-integer fixed-point functions.
//!
//! Determinism / cross-backend bit-exactness:
//!   - every reduction (sum x^2, sum exp) is EXACT integer (i64/i128) =>
//!     reduction-order-immune (a backend reordering the reduction is bit-identical);
//!   - every transcendental is a deterministic integer function: 1/sqrt via integer
//!     isqrt; exp via the 2^x integer poly (coeffs (ln2)^k/k!); tanh via that exp.
//! Fidelity vs the float forms is ~1e-5..1e-6 (lossless for inference).
//!
//! Fixed-point layout (frozen): FB=30 (exp/tanh) · RMS Q=16 / IB=20 / Qw=16 ·
//! softmax Z=2^14 · GELU Z=2^16. i128 is used for the wide intermediates (the RMS
//! n<<2*(Q+IB) numerator, the GELU cubic, the exp d*log2e) — Rust has i128 natively;
//! a CUDA backend reproduces these exactly with __umul64hi / a 128/64 divide.
#![allow(dead_code)]

pub const FB: i64 = 30;
pub const ONE: i64 = 1 << FB;
pub const LOG2E: i64 = 1_549_082_005; // round(log2(e) * 2^30)
pub const EXPC: [i64; 7] = [
    1_073_741_824, 744_261_118, 257_941_248, 59_597_083, 10_327_387, 1_431_680, 165_394,
]; // round((ln2)^k / k! * 2^30)
pub const GK: i64 = 856_722_024; // round(sqrt(2/pi) * 2^30)
pub const GA: i64 = 48_012_366; // round(0.044715 * 2^30)

/// 2^(r/2^FB) for r in [0, ONE], FB-fixed.
fn exp2_frac(r: i64) -> i64 {
    let mut acc = EXPC[6];
    let mut k = 5i32;
    while k >= 0 {
        acc = ((acc * r) >> FB) + EXPC[k as usize]; // acc<2^31, r<2^30 -> <2^61
        k -= 1;
    }
    acc
}

/// e^d for d <= 0, FB-fixed in/out. d*LOG2E uses i128 (overflows i64 for far-from-max keys).
pub fn exp_fixed(d: i64) -> i64 {
    if d >= 0 {
        return ONE;
    }
    let g: i64 = (-(((d as i128) * (LOG2E as i128)) >> FB)) as i64; // >= 0
    let n = g >> FB;
    if n >= 32 {
        return 0;
    }
    let r = g - (n << FB);
    if r != 0 {
        exp2_frac(ONE - r) >> (n + 1)
    } else {
        ONE >> n
    }
}

/// floor(sqrt(v)) over u128 — exact integer isqrt.
fn isqrt_u128(v: u128) -> u128 {
    if v == 0 {
        return 0;
    }
    let mut x = 0u128;
    let mut b = 1u128 << 126;
    while b > v {
        b >>= 2;
    }
    let mut v = v;
    while b != 0 {
        if v >= x + b {
            v -= x + b;
            x = (x >> 1) + b;
        } else {
            x >>= 1;
        }
        b >>= 2;
    }
    x
}

fn enc(v: f32, shift: u32) -> i64 {
    (v * ((1u64 << shift) as f32)).round() as i64
}

/// Island 1 — exact-integer RMSNorm. out[i] = x[i]*sqrt(n/sum x^2)*(w?w[i]:1).
/// sum x^2 exact (i128) -> reduction-order-immune; inv = isqrt((n<<2*(Q+IB))/sumsq).
pub fn rmsnorm_q_ref(x: &[f32], w: Option<&[f32]>) -> Vec<f32> {
    const Q: u32 = 16;
    const IB: u32 = 20;
    const QW: u32 = 16;
    let n = x.len();
    let xi: Vec<i64> = x.iter().map(|&v| enc(v, Q)).collect();
    let sumsq: i128 = xi.iter().map(|&a| (a as i128) * (a as i128)).sum();
    if sumsq <= 0 {
        return vec![0.0f32; n];
    }
    let num: u128 = (n as u128) << (2 * (Q + IB)); // n << 72
    let inv: i128 = isqrt_u128(num / (sumsq as u128)) as i128;
    let denom = (1u128 << (Q + IB + QW)) as f64; // 2^52
    (0..n)
        .map(|i| {
            let wi: i128 = match w {
                Some(ww) => enc(ww[i], QW) as i128,
                None => 1i128 << QW,
            };
            let y = (xi[i] as i128) * inv * wi;
            (y as f64 / denom) as f32
        })
        .collect()
}

/// Island 2 — exact-integer softmax. p = exp(z-max)/sum, Z=2^14, exp FB=30,
/// denominator summed exactly in i128 (reduction-order-immune).
pub fn softmax_q_ref(z: &[f32]) -> Vec<f32> {
    const ZB: u32 = 14;
    let zi: Vec<i64> = z.iter().map(|&v| enc(v, ZB)).collect();
    let m = *zi.iter().max().unwrap();
    let e: Vec<i64> = zi
        .iter()
        .map(|&zz| exp_fixed(((zz - m) * ONE) >> ZB)) // (zz-m)<=0; >> = floor
        .collect();
    let s: i128 = e.iter().map(|&x| x as i128).sum();
    e.iter().map(|&ei| (ei as f64 / s as f64) as f32).collect()
}

/// tanh(t) FB-fixed via the shared exp primitive.
fn tanh_fixed(t: i64) -> i64 {
    let s: i64 = if t >= 0 { 1 } else { -1 };
    let a = t.abs();
    let e2 = exp_fixed(-(2 * a));
    let num = (2 * e2) << FB; // 2*e2 <= 2^31, <<30 -> <2^62
    s * (ONE - num / (ONE + e2))
}

/// Island 3 — exact-integer GELU-tanh: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715 x^3))).
/// Cubic + tanh in FB=30 fixed-point (i128 for the cubic). Deterministic per-element.
pub fn gelu_q_ref(x: &[f32]) -> Vec<f32> {
    const ZB: u32 = 16;
    x.iter()
        .map(|&xv| {
            let xq = enc(xv, ZB);
            let big_x = (xq * ONE) >> ZB; // i64; ~2^34
            let x2: i128 = ((big_x as i128) * (big_x as i128)) >> FB;
            let x3: i128 = (x2 * (big_x as i128)) >> FB;
            let inner: i64 =
                (((GK as i128) * ((big_x as i128) + (((GA as i128) * x3) >> FB))) >> FB) as i64;
            let t = tanh_fixed(inner);
            let g: i128 = (((big_x as i128) >> 1) * ((ONE as i128) + (t as i128))) >> FB;
            (g as f64 / ONE as f64) as f32
        })
        .collect()
}

// ---- Island 4 — RoPE via deterministic fixed-point CORDIC (no libm sin/cos) ----
// The fp32 RoPE bridge's sinf/cosf are machine-dependent; replace with rotation-mode
// CORDIC (integer shift-add over a fixed atan table) => bit-identical cos/sin across any
// ALU. The per-pair frequency table is a MODEL CONSTANT (base^(-2i/d), baked once at
// transcode as fixed-point — deterministic by being stored, not recomputed); rope_q_ref
// takes it as input so the function itself is fully integer/deterministic. NEOX layout.
const CORDIC_N: usize = 30;
const ATAN_FB30: [i64; CORDIC_N] = [
    843314857, 497837829, 263043837, 133525159, 67021687, 33543516, 16775851, 8388437,
    4194283, 2097149, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096,
    2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2,
];
const CORDIC_K: i64 = 652_032_874; // round( prod 1/sqrt(1+2^-2k) * 2^30 )
const PI_FB: i64 = 3_373_259_426;
const HALFPI_FB: i64 = 1_686_629_713;
const TWOPI_FB: i64 = 6_746_518_852;

/// (cos θ, sin θ) in FB-fixed for θ in FB-fixed radians. Reduces to [−π/2, π/2] then CORDIC.
pub fn cordic_cossin(theta: i64) -> (i64, i64) {
    // reduce mod 2π into (−π, π]
    let mut z = theta % TWOPI_FB;
    if z > PI_FB {
        z -= TWOPI_FB;
    } else if z < -PI_FB {
        z += TWOPI_FB;
    }
    // fold the two far quadrants into [−π/2, π/2] (cos/sin both negate under +π)
    let mut neg = false;
    if z > HALFPI_FB {
        z -= PI_FB;
        neg = true;
    } else if z < -HALFPI_FB {
        z += PI_FB;
        neg = true;
    }
    let mut x = CORDIC_K;
    let mut y = 0i64;
    for k in 0..CORDIC_N {
        let xs = x >> k;
        let ys = y >> k;
        if z >= 0 {
            x -= ys;
            y += xs;
            z -= ATAN_FB30[k];
        } else {
            x += ys;
            y -= xs;
            z += ATAN_FB30[k];
        }
    }
    if neg {
        (-x, -y)
    } else {
        (x, y)
    }
}

/// NEOX RoPE at position `pos`, integer-exact: out[i] = a·cos−b·sin, out[i+half] = a·sin+b·cos,
/// where (cos,sin)=CORDIC(pos·freq_fix[i]). `freq_fix` is the length-(d/2) fixed-point freq table
/// (FB-fixed; the model constant base^(-2i/d) / ff[i]). Rotation done in fixed-point (Q=16) so the
/// whole op is integer/deterministic.
pub fn rope_q_ref(v: &[f32], pos: i64, freq_fix: &[i64]) -> Vec<f32> {
    const Q: u32 = 16;
    let d = v.len();
    let half = d / 2;
    let mut out = vec![0.0f32; d];
    let inv = (1u64 << Q) as f64;
    for i in 0..half {
        let theta = ((pos as i128) * (freq_fix[i] as i128) % (TWOPI_FB as i128)) as i64;
        let (c, s) = cordic_cossin(theta);
        let a = (v[i] as f64 * inv).round() as i128;
        let b = (v[i + half] as f64 * inv).round() as i128;
        let oa = ((a * c as i128) - (b * s as i128)) >> FB;
        let ob = ((a * s as i128) + (b * c as i128)) >> FB;
        out[i] = (oa as f64 / inv) as f32;
        out[i + half] = (ob as f64 / inv) as f32;
    }
    out
}
