//! §4-NTT Sprint NTT.4 Stage 1 — INTT correctness smoke harness.
//!
//! Drives:
//!   - `sp_compute_ntt_twiddle_init`   (skel method 14) — must run first to
//!                                       populate VTCM tables
//!   - `sp_compute_ntt_hvx_oracle`     (skel method 13, NTT.1 forward HVX)
//!   - `sp_compute_ntt_intt_hvx_oracle` (skel method 17 in this worktree;
//!                                       anticipated method 18 post-NTT.3-merge)
//!
//! T_NTT4_INTT_BIT_EXACT — INTT(NTT(x)) recovers x byte-exact per prime.
//!
//! Methodology (per-prime, since this is the Stage 1 per-prime gate):
//!   1. Generate random i32 input.
//!   2. Run forward NTT via method 13 → u32 residue vector.
//!   3. Run INTT via method 17 → u32 result.
//!   4. Compare against math-core's `inverse_one`-equivalent host-side
//!      reference: forward (math-core `ntt_forward`) gives the per-prime
//!      residue, then call `ntt_inverse` (math-core) on (out1, out2) to
//!      get the signed combined result, then verify per-prime
//!      reduction matches: the on-device INTT output should equal the
//!      ORIGINAL input mod q (input_j mod q == intt_out[j] iff
//!      input_j < q on the positive branch, or with wrap on negative).
//!
//! That's the round-trip property: INTT(NTT(x)) = x mod q (per prime).
//! Re-derive math-core's reference INTT in pure Rust to keep the gate
//! self-contained and avoid an extra FFI symbol.
//!
//! 600-run sweep: 3 N × 2 primes × 100 seeds. Pass: 0 divergences.
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_ntt_4_intt_smoke
//! Run:
//!     adb push target/aarch64-linux-android/release/sp_ntt_4_intt_smoke /data/local/tmp/
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_ntt_4_intt_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_ntt_4_intt_smoke: host build skipped (target_os = android required)");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
mod ntt_4_ref {
    //! Host-side reference: per-prime forward + inverse via byte-exact
    //! re-implementation of math-core (ntt_crt.c:206-294). Used as the
    //! oracle for the round-trip gate. Both directions implemented in
    //! pure Rust; no FFI to math-core needed (NTT.2 smoke already proved
    //! the table init matches math-core byte-exact via T_NTT2_TWIDDLE_BIT_EXACT,
    //! so this Rust re-impl is grounded).

    pub const Q1: u32 = 1_073_738_753;
    pub const Q2: u32 = 1_073_732_609;
    pub const MU_Q1: u64 = 1_073_744_895;
    pub const MU_Q2: u64 = 1_073_751_039;
    pub const Q_BITS: u32 = 30;

    #[inline]
    fn barrett_reduce(x: u64, q: u64, mu: u64) -> u64 {
        let qhat = (((x >> (Q_BITS - 1)) as u128 * mu as u128) >> (Q_BITS + 1)) as u64;
        let mut r = x.wrapping_sub(qhat.wrapping_mul(q));
        if r >= q { r -= q; }
        if r >= q { r -= q; }
        r
    }

    #[inline]
    pub fn modmul(a: u32, b: u32, q: u32, mu: u64) -> u32 {
        let x = (a as u64) * (b as u64);
        barrett_reduce(x, q as u64, mu) as u32
    }

    #[inline]
    pub fn modadd(a: u32, b: u32, q: u32) -> u32 {
        let s = a as u64 + b as u64;
        if s >= q as u64 { (s - q as u64) as u32 } else { s as u32 }
    }

    #[inline]
    pub fn modsub(a: u32, b: u32, q: u32) -> u32 {
        if a >= b { a - b } else { a + q - b }
    }

    pub fn modpow(base: u32, e: u64, q: u32, mu: u64) -> u32 {
        let mut result: u32 = 1u32 % q;
        let mut b: u32 = base % q;
        let mut e = e;
        while e != 0 {
            if (e & 1) != 0 { result = modmul(result, b, q, mu); }
            b = modmul(b, b, q, mu);
            e >>= 1;
        }
        result
    }

    pub fn modinv(a: u32, q: u32, mu: u64) -> u32 {
        modpow(a, (q as u64) - 2, q, mu)
    }

    pub fn find_psi(n: u32, q: u32, mu: u64) -> u32 {
        let exp: u64 = ((q as u64) - 1) / (2 * (n as u64));
        let mut a: u32 = 2;
        while a < q {
            let psi = modpow(a, exp, q, mu);
            if psi != 0 {
                let p_n = modpow(psi, n as u64, q, mu);
                if p_n == q - 1 { return psi; }
            }
            a += 1;
        }
        0
    }

    pub fn ilog2(n: u32) -> u32 {
        let mut l: u32 = 0;
        while (1u32 << l) < n { l += 1; }
        l
    }

    pub fn bitrev(x: u32, log_n: u32) -> u32 {
        let mut r: u32 = 0;
        let mut x = x;
        for _ in 0..log_n {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        r
    }

    #[allow(dead_code)]
    pub struct PrimeCtx {
        pub q: u32,
        pub mu: u64,
        pub psi: u32,
        pub ipsi: u32,
        pub ninv: u32,
        pub psi_pow: Vec<u32>,
        pub ipsi_pow: Vec<u32>,
        pub w_fwd: Vec<u32>,
        pub w_inv: Vec<u32>,
    }

    pub fn build_ctx(n: u32, q: u32, mu: u64) -> PrimeCtx {
        let psi = find_psi(n, q, mu);
        assert!(psi != 0, "find_psi failed for N={n} q={q}");
        let ipsi = modinv(psi, q, mu);
        let omega = modmul(psi, psi, q, mu);
        let iomega = modmul(ipsi, ipsi, q, mu);
        let ninv = modinv(n % q, q, mu);

        let mut psi_pow = vec![0u32; n as usize];
        let mut acc = 1u32 % q;
        for j in 0..n as usize {
            psi_pow[j] = acc;
            acc = modmul(acc, psi, q, mu);
        }
        let mut ipsi_pow = vec![0u32; n as usize];
        let mut acc = 1u32 % q;
        for j in 0..n as usize {
            ipsi_pow[j] = acc;
            acc = modmul(acc, ipsi, q, mu);
        }
        let half = n as usize / 2;
        let mut w_fwd = vec![0u32; half];
        let mut acc = 1u32 % q;
        for j in 0..half {
            w_fwd[j] = acc;
            acc = modmul(acc, omega, q, mu);
        }
        let mut w_inv = vec![0u32; half];
        let mut acc = 1u32 % q;
        for j in 0..half {
            w_inv[j] = acc;
            acc = modmul(acc, iomega, q, mu);
        }
        PrimeCtx { q, mu, psi, ipsi, ninv, psi_pow, ipsi_pow, w_fwd, w_inv }
    }

    /// Core Cooley-Tukey radix-2 DIT loop (math-core ntt_core:241-254).
    /// Used by both forward (with w_fwd) and inverse (with w_inv).
    pub fn ntt_core(out: &mut [u32], n: u32, log_n: u32, w: &[u32], q: u32, mu: u64) {
        for i in 0..n {
            let j = bitrev(i, log_n);
            if i < j { out.swap(i as usize, j as usize); }
        }
        let mut len = 2u32;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            let mut i = 0u32;
            while i < n {
                let mut widx = 0u32;
                for k in 0..half {
                    let u = out[(i + k) as usize];
                    let v = modmul(out[(i + k + half) as usize], w[widx as usize], q, mu);
                    out[(i + k) as usize]        = modadd(u, v, q);
                    out[(i + k + half) as usize] = modsub(u, v, q);
                    widx += step;
                }
                i += len;
            }
            len <<= 1;
        }
    }

    /// Forward NTT for one prime — mirrors math-core forward_one
    /// (ntt_crt.c:259-272).
    pub fn forward_one(ctx: &PrimeCtx, n: u32, log_n: u32, input: &[i32]) -> Vec<u32> {
        let mut out = vec![0u32; n as usize];
        // Pre-weight: out[j] = (in[j] mod q) * psi^j mod q
        for j in 0..n as usize {
            let mut v = (input[j] as i64) % (ctx.q as i64);
            if v < 0 { v += ctx.q as i64; }
            out[j] = modmul(v as u32, ctx.psi_pow[j], ctx.q, ctx.mu);
        }
        ntt_core(&mut out, n, log_n, &ctx.w_fwd, ctx.q, ctx.mu);
        out
    }

    /// Inverse NTT for one prime — mirrors math-core inverse_one
    /// (ntt_crt.c:281-294).
    pub fn inverse_one(ctx: &PrimeCtx, n: u32, log_n: u32, input: &[u32]) -> Vec<u32> {
        let mut out = vec![0u32; n as usize];
        for j in 0..n as usize {
            out[j] = input[j] % ctx.q;
        }
        ntt_core(&mut out, n, log_n, &ctx.w_inv, ctx.q, ctx.mu);
        // Post-pass: scale by ninv, then mul by ipsi_pow[j].
        for j in 0..n as usize {
            let s = modmul(out[j], ctx.ninv, ctx.q, ctx.mu);
            out[j] = modmul(s, ctx.ipsi_pow[j], ctx.q, ctx.mu);
        }
        out
    }

    pub fn pick_q_mu(q_idx: i32) -> (u32, u64) {
        if q_idx == 0 { (Q1, MU_Q1) } else { (Q2, MU_Q2) }
    }
}

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf};
    use ntt_4_ref::{build_ctx, forward_one, inverse_one, ilog2, pick_q_mu};
    use std::ffi::c_void;
    use std::time::Instant;

    eprintln!("[NTT.4] sp_ntt_4_intt_smoke -- INTT correctness gate");
    eprintln!("[NTT.4]   T_NTT4_INTT_BIT_EXACT (method 17 == math-core inverse_one)");
    eprintln!("[NTT.4]   round-trip = INTT(NTT(x)) ?= x mod q  for all (q, N, seed)");

    eprintln!("\n[NTT.4] opening FastRpcSession (Path B Unsigned PD)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp",
    ) {
        Ok(s) => { eprintln!("[NTT.4] session open"); s }
        Err(e) => { eprintln!("[NTT.4] session FAIL: {e:?}"); std::process::exit(1); }
    };

    // ── method 14: ntt_twiddle_init  (prerequisite for INTT) ──
    // primIn = [N(i32)]; scalars: method=14, inbufs=1, outbufs=0
    eprintln!("[NTT.4] priming VTCM twiddles via ntt_twiddle_init(N=512)...");
    {
        let mut prim_in: [u32; 1] = [512u32];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 4 }},
        ];
        match sess.invoke(make_scalars(14, 1, 0), &mut args) {
            Ok(_) => eprintln!("[NTT.4] ntt_twiddle_init ok"),
            Err(e) => {
                eprintln!("[NTT.4] ntt_twiddle_init FAIL: {e:?}");
                std::process::exit(1);
            }
        }
    }

    // ── method 13: ntt_hvx_oracle (forward NTT) ──
    // primIn = [q_idx, N, data_inLen, data_outLen] (16 bytes)
    fn invoke_forward(
        sess: &FastRpcSession, q_idx: i32, n: i32, data_in: &[i32],
    ) -> Result<Vec<u32>, String> {
        let n_bytes = (n as usize) * 4;
        let mut prim_in: [u32; 4] = [
            q_idx as u32, n as u32, n_bytes as u32, n_bytes as u32,
        ];
        let mut in_bytes: Vec<u8> = Vec::with_capacity(n_bytes);
        for v in data_in { in_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut out_bytes: Vec<u8> = vec![0u8; n_bytes];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 16 }},
            RemoteArg { buf: RemoteBuf { pv: in_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: out_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        ];
        sess.invoke(make_scalars(13, 2, 1), &mut args)
            .map_err(|e| format!("invoke ntt_hvx_oracle: {e:?}"))?;
        Ok(out_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    // ── method 17 (this worktree): intt_hvx_oracle ──
    // Same primIn shape as forward (16 bytes); takes u32 NTT output as input.
    // Post-merge with NTT.3, this MAY become method 18 — closure documents.
    const INTT_METHOD: u32 = 17;
    fn invoke_intt(
        sess: &FastRpcSession, q_idx: i32, n: i32, data_in: &[u32],
    ) -> Result<Vec<u32>, String> {
        let n_bytes = (n as usize) * 4;
        let mut prim_in: [u32; 4] = [
            q_idx as u32, n as u32, n_bytes as u32, n_bytes as u32,
        ];
        let mut in_bytes: Vec<u8> = Vec::with_capacity(n_bytes);
        for v in data_in { in_bytes.extend_from_slice(&v.to_le_bytes()); }
        let mut out_bytes: Vec<u8> = vec![0u8; n_bytes];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 16 }},
            RemoteArg { buf: RemoteBuf { pv: in_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
            RemoteArg { buf: RemoteBuf { pv: out_bytes.as_mut_ptr() as *mut c_void, nlen: n_bytes }},
        ];
        sess.invoke(make_scalars(INTT_METHOD, 2, 1), &mut args)
            .map_err(|e| format!("invoke intt_hvx_oracle (method {INTT_METHOD}): {e:?}"))?;
        Ok(out_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn gen_random_i32_vec(seed: u64, n: usize) -> Vec<i32> {
        let mut s = seed;
        let mut v: Vec<i32> = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(s as i32);
        }
        v
    }

    // ── Gate counters ──
    let seeds_per_combination = 100u32;
    let mut combinations_tested = 0u32;
    let mut total_runs = 0u32;
    let mut divergence_count = 0u32;
    let mut max_diff_per_prime: [u32; 2] = [0, 0];
    let mut max_diff_per_n: [(i32, u32); 3] = [(128, 0), (256, 0), (512, 0)];
    let mut first_divergence: Option<(i32, i32, u64, usize, u32, u32)> = None;

    let t_start = Instant::now();

    for &n in &[128i32, 256, 512] {
        for q_idx in 0i32..=1 {
            combinations_tested += 1;
            eprintln!("\n[NTT.4] -- combination q_idx={q_idx}  N={n} --");
            let (q, mu) = pick_q_mu(q_idx);
            let ctx = build_ctx(n as u32, q, mu);
            let log_n = ilog2(n as u32);

            let mut local_max_diff: u32 = 0;
            for seed_ix in 0..seeds_per_combination {
                let seed: u64 = 0xDEAFu64
                    .wrapping_add((n as u64).wrapping_mul(1_000_003))
                    .wrapping_add((q_idx as u64).wrapping_mul(2_000_017))
                    .wrapping_add(seed_ix as u64);
                let data_in = gen_random_i32_vec(seed, n as usize);

                // Compute host-side reference: forward then inverse (per-prime).
                let host_fwd = forward_one(&ctx, n as u32, log_n, &data_in);
                let host_intt = inverse_one(&ctx, n as u32, log_n, &host_fwd);

                // Compute on-device: method 13 forward, then method 17 INTT.
                let dev_fwd = match invoke_forward(&sess, q_idx, n, &data_in) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[NTT.4] forward FAIL q_idx={q_idx} N={n} seed_ix={seed_ix}: {e}");
                        divergence_count += seeds_per_combination - seed_ix;
                        total_runs += seeds_per_combination - seed_ix;
                        break;
                    }
                };
                let dev_intt = match invoke_intt(&sess, q_idx, n, &dev_fwd) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[NTT.4] intt FAIL q_idx={q_idx} N={n} seed_ix={seed_ix}: {e}");
                        divergence_count += seeds_per_combination - seed_ix;
                        total_runs += seeds_per_combination - seed_ix;
                        break;
                    }
                };
                total_runs += 1;

                // Compare on-device INTT vs host INTT — both consume same fwd input
                // (dev_fwd is byte-exact vs host_fwd per T_NTT1_HVX_BIT_EXACT;
                //  we use host_fwd here as the canonical reference input to host_intt
                //  but dev_intt uses dev_fwd. Since dev_fwd == host_fwd byte-exact
                //  (NTT.1 gate), dev_intt should equal host_intt byte-exact too).
                let mut local_diverge = false;
                for j in 0..(n as usize) {
                    if dev_intt[j] != host_intt[j] {
                        let diff = if dev_intt[j] > host_intt[j] { dev_intt[j] - host_intt[j] } else { host_intt[j] - dev_intt[j] };
                        if diff > local_max_diff { local_max_diff = diff; }
                        if !local_diverge {
                            divergence_count += 1;
                            if first_divergence.is_none() {
                                first_divergence = Some((n, q_idx, seed, j, dev_intt[j], host_intt[j]));
                            }
                            local_diverge = true;
                        }
                    }
                }
                if local_diverge {
                    eprintln!("[NTT.4]   ! divergence q_idx={q_idx} N={n} seed_ix={seed_ix} local_max_diff={local_max_diff}");
                }

                // Also verify round-trip: host_intt should equal data_in mod q
                // (sanity check on the host reference itself; a Rust-impl bug
                // would surface here).
                if seed_ix == 0 {
                    let mut roundtrip_ok = true;
                    for j in 0..(n as usize) {
                        let expected = {
                            let mut v = (data_in[j] as i64) % (q as i64);
                            if v < 0 { v += q as i64; }
                            v as u32
                        };
                        if host_intt[j] != expected {
                            roundtrip_ok = false;
                            eprintln!("[NTT.4]   HOST REF BUG at j={j}: host_intt={} expected={}",
                                      host_intt[j], expected);
                            break;
                        }
                    }
                    if roundtrip_ok {
                        eprintln!("[NTT.4]   host round-trip sanity ok (q_idx={q_idx} N={n} seed_ix=0)");
                    }
                }
            }

            if local_max_diff > max_diff_per_prime[q_idx as usize] {
                max_diff_per_prime[q_idx as usize] = local_max_diff;
            }
            let n_slot = match n { 128 => 0, 256 => 1, 512 => 2, _ => unreachable!() };
            if local_max_diff > max_diff_per_n[n_slot].1 {
                max_diff_per_n[n_slot].1 = local_max_diff;
            }
            eprintln!("[NTT.4]   combination max_diff = {local_max_diff}");
        }
    }

    let elapsed_s = t_start.elapsed().as_secs_f64();
    eprintln!("\n[NTT.4] ── T_NTT4_INTT_BIT_EXACT SUMMARY ──");
    eprintln!("[NTT.4] combinations_tested  : {combinations_tested}");
    eprintln!("[NTT.4] total_runs           : {total_runs}");
    eprintln!("[NTT.4] divergence_count     : {divergence_count}");
    eprintln!("[NTT.4] max_diff_per_prime   : q1={} q2={}",
              max_diff_per_prime[0], max_diff_per_prime[1]);
    eprintln!("[NTT.4] max_diff_per_n       : N=128:{} N=256:{} N=512:{}",
              max_diff_per_n[0].1, max_diff_per_n[1].1, max_diff_per_n[2].1);
    eprintln!("[NTT.4] elapsed              : {elapsed_s:.2} s");
    if let Some((n, q_idx, seed, j, dev, host)) = first_divergence {
        eprintln!("[NTT.4] first_divergence     : N={n} q_idx={q_idx} seed={seed:#x} j={j} dev={dev} host={host}");
    }

    let pass = divergence_count == 0 && total_runs == seeds_per_combination * combinations_tested;
    if pass {
        eprintln!("[NTT.4] T_NTT4_INTT_BIT_EXACT PASS  ({total_runs}/{total_runs} runs byte-exact)");
        std::process::exit(0);
    } else {
        eprintln!("[NTT.4] T_NTT4_INTT_BIT_EXACT FAIL  ({divergence_count}/{total_runs} divergences)");
        std::process::exit(1);
    }
}
