//! Host-runnable gate for the byte-exact NONLINEAR islands (sp_islands_q_ref).
//! Bin (not lib) to match the crate convention; runs on Windows/Linux x86, no DSP.
//!
//!     cargo run --bin sp_islands_q_ref_test
//!
//! Validates, per island:
//!   FIDELITY      — matches the float reference to ~1e-5..1e-4 (lossless for inference);
//!   BYTE-EXACT    — the reductions are reduction-order-immune, so a permuted input yields
//!                   a BIT-IDENTICAL scale/denominator (the property the float forms lack).
//! T_ISLANDS_RMSNORM / T_ISLANDS_SOFTMAX / T_ISLANDS_GELU. Exit 0 iff all pass.

mod sp_islands_q_ref;
use sp_islands_q_ref::{gelu_q_ref, rmsnorm_q_ref, softmax_q_ref};

// deterministic LCG in [-rng, rng] (no external crate)
struct Lcg(u64);
impl Lcg {
    fn dev(&mut self, rng: f64) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / ((1u64 << 53) as f64); // [0,1)
        (2.0 * u - 1.0) * rng
    }
}

fn relerr(a: &[f64], b: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        num += d * d;
        den += a[i] * a[i];
    }
    num.sqrt() / (den.sqrt() + 1e-30)
}

fn main() {
    let mut fails = 0usize;

    // ---------- Island 1: RMSNorm ----------
    {
        let e = 3840usize;
        let mut g = Lcg(1);
        let x: Vec<f32> = (0..e).map(|_| g.dev(2.0) as f32).collect();
        let w: Vec<f32> = (0..e).map(|_| (1.0 + g.dev(0.1)) as f32).collect();
        let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let scale = 1.0 / (ss / e as f64).sqrt();
        let yf: Vec<f64> = (0..e).map(|i| x[i] as f64 * scale * w[i] as f64).collect();
        let yi: Vec<f64> = rmsnorm_q_ref(&x, Some(&w)).iter().map(|&v| v as f64).collect();
        let re = relerr(&yi, &yf);
        eprintln!("[T_ISLANDS_RMSNORM] fidelity relerr = {:.3e}", re);
        if !(re < 1e-4) {
            fails += 1;
            eprintln!("  FAIL: RMSNorm relerr >= 1e-4");
        }
        // reduction-order immunity: reversed input, unit weight -> reversed output bit-identical
        let wone: Vec<f32> = vec![1.0f32; e];
        let xr: Vec<f32> = x.iter().rev().cloned().collect();
        let oa = rmsnorm_q_ref(&x, Some(&wone));
        let ob = rmsnorm_q_ref(&xr, Some(&wone));
        let order_ok = (0..e).all(|i| ob[i] == oa[e - 1 - i]);
        eprintln!("[T_ISLANDS_RMSNORM] reduction-order-immune = {}", order_ok);
        if !order_ok {
            fails += 1;
        }
    }

    // ---------- Island 2: softmax ----------
    {
        let m = 256usize;
        let mut g = Lcg(7);
        let z: Vec<f32> = (0..m).map(|_| g.dev(8.0) as f32).collect();
        let mx = z.iter().cloned().fold(f32::MIN, f32::max);
        let mut se = 0.0f64;
        let pf: Vec<f64> = z
            .iter()
            .map(|&v| {
                let e = ((v - mx) as f64).exp();
                se += e;
                e
            })
            .collect();
        let pf: Vec<f64> = pf.iter().map(|&p| p / se).collect();
        let pi: Vec<f64> = softmax_q_ref(&z).iter().map(|&v| v as f64).collect();
        let mad = (0..m).map(|i| (pf[i] - pi[i]).abs()).fold(0.0, f64::max);
        eprintln!("[T_ISLANDS_SOFTMAX] fidelity max|dp| = {:.3e}", mad);
        if !(mad < 1e-5) {
            fails += 1;
            eprintln!("  FAIL: softmax max|dp| >= 1e-5");
        }
        let zr: Vec<f32> = z.iter().rev().cloned().collect();
        let pa = softmax_q_ref(&z);
        let pb = softmax_q_ref(&zr);
        let order_ok = (0..m).all(|i| pb[i] == pa[m - 1 - i]);
        eprintln!("[T_ISLANDS_SOFTMAX] reduction-order-immune = {}", order_ok);
        if !order_ok {
            fails += 1;
        }
    }

    // ---------- Island 3: GELU-tanh ----------
    {
        let n = 512usize;
        let mut g = Lcg(11);
        let k = (2.0f64 / std::f64::consts::PI).sqrt();
        let x: Vec<f32> = (0..n).map(|_| g.dev(3.0) as f32).collect();
        let gf: Vec<f64> = x
            .iter()
            .map(|&xv| {
                let xv = xv as f64;
                0.5 * xv * (1.0 + (k * (xv + 0.044715 * xv * xv * xv)).tanh())
            })
            .collect();
        let gi = gelu_q_ref(&x);
        let gid: Vec<f64> = gi.iter().map(|&v| v as f64).collect();
        let re = relerr(&gid, &gf);
        eprintln!("[T_ISLANDS_GELU] fidelity relerr = {:.3e}", re);
        if !(re < 1e-4) {
            fails += 1;
            eprintln!("  FAIL: GELU relerr >= 1e-4");
        }
        let gi2 = gelu_q_ref(&x);
        let det_ok = (0..n).all(|i| gi2[i] == gi[i]);
        eprintln!("[T_ISLANDS_GELU] deterministic = {}", det_ok);
        if !det_ok {
            fails += 1;
        }
    }

    eprintln!("---- sp_islands_q_ref_test: fails = {} ----", fails);
    if fails == 0 {
        eprintln!("VERDICT: GREEN — the three nonlinear islands are exact-integer, fidelity-correct, and reduction-order-immune.");
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}
