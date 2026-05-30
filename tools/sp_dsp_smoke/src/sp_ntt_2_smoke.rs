//! Sprint NTT.2 — twiddle factor VTCM staging smoke.
//!
//! Drives the three new IDL methods exposed by `sp_compute_skel`:
//!   - method 13: `ntt_twiddle_init(N)` — populates all 6 (prime, N) tables
//!   - method 14: `ntt_twiddle_status(N, q_idx, ...)` — inspects one entry
//!   - method 15: `ntt_twiddle_dump(N, q_idx, table_id, dst_buf)` — copies
//!     one VTCM sub-table into a caller buffer for byte-exact comparison
//!
//! Method numbers verified post-build via qaic-emitted
//! `hexagon_Release_toolv87_v69/sp_compute_skel.c` switch: ntt_oracle=12,
//! ntt_twiddle_init=13, ntt_twiddle_status=14, ntt_twiddle_dump=15.
//!
//! Gates (per PLAN-NTT-2.md):
//!   - T_NTT2_TWIDDLE_INIT     — all 6 entries initialized, vtcm_addr non-zero
//!   - T_NTT2_TWIDDLE_BIT_EXACT — VTCM tables byte-identical to host-side ref
//!   - T_NTT2_VTCM_BUDGET      — total VTCM usage ≤ 2 MB at peak N=512
//!
//! Build:
//!     cargo build --target aarch64-linux-android --release --bin sp_ntt_2_smoke
//! Run:
//!     adb push target/aarch64-linux-android/release/sp_ntt_2_smoke /data/local/tmp/
//!     adb shell 'ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_ntt_2_smoke'

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("sp_ntt_2_smoke: host build skipped (target_os = android required)");
}

#[cfg(target_os = "android")]
mod dsp_rpc;

#[cfg(target_os = "android")]
mod ntt_2_ref {
    //! Host-side reference for the byte-exact gate.
    //!
    //! Re-implements math-core `prime_setup` (ntt_crt.c:129-173) including
    //! the Barrett primitives.  Byte-exact by construction.  We don't
    //! depend on math-core's `ntt_ctx` internals here because the host-side
    //! struct is opaque (psi_pow / w_fwd / etc. are private fields, no FFI
    //! accessor).  Re-implementing the table init in Rust is the principled
    //! oracle: any deviation between this Rust ref and the DSP-side C
    //! tables is a real divergence to surface.

    pub const Q1: u32 = 1_073_738_753;
    pub const Q2: u32 = 1_073_732_609;
    pub const MU_Q1: u64 = 1_073_744_895;
    pub const MU_Q2: u64 = 1_073_751_039;
    pub const Q_BITS: u32 = 30;

    #[inline]
    fn barrett_reduce(x: u64, q: u64, mu: u64) -> u64 {
        let qhat = ((x >> (Q_BITS - 1)) as u128 * mu as u128) >> (Q_BITS + 1);
        let qhat = qhat as u64;
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
        // Brute-force search a from 2..q for smallest psi with psi^N == -1.
        // For the frozen primes + N ≤ 512 the answer is small (typically a ≤ 5).
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

    fn ilog2(n: u32) -> u32 {
        let mut l: u32 = 0;
        while (1u32 << l) < n { l += 1; }
        l
    }

    pub struct RefTables {
        pub psi_pow: Vec<u32>,
        pub ipsi_pow: Vec<u32>,
        pub w_fwd: Vec<u32>,
        pub w_inv: Vec<u32>,
        pub w_fwd_stages: Vec<u32>,
        pub w_inv_stages: Vec<u32>,
    }

    pub fn build_ref(n: u32, q_idx: i32) -> RefTables {
        let q  = if q_idx == 0 { Q1 } else { Q2 };
        let mu = if q_idx == 0 { MU_Q1 } else { MU_Q2 };

        let psi  = find_psi(n, q, mu);
        assert!(psi != 0, "find_psi returned 0 for q_idx={} N={}", q_idx, n);
        let ipsi = modinv(psi, q, mu);
        let omega  = modmul(psi,  psi,  q, mu);
        let iomega = modmul(ipsi, ipsi, q, mu);

        let mut psi_pow  = vec![0u32; n as usize];
        let mut ipsi_pow = vec![0u32; n as usize];
        let mut w_fwd    = vec![0u32; (n / 2) as usize];
        let mut w_inv    = vec![0u32; (n / 2) as usize];

        {
            let mut acc: u32 = 1 % q;
            for j in 0..n as usize {
                psi_pow[j] = acc;
                acc = modmul(acc, psi, q, mu);
            }
        }
        {
            let mut acc: u32 = 1 % q;
            for j in 0..n as usize {
                ipsi_pow[j] = acc;
                acc = modmul(acc, ipsi, q, mu);
            }
        }
        {
            let mut acc: u32 = 1 % q;
            for j in 0..(n / 2) as usize {
                w_fwd[j] = acc;
                acc = modmul(acc, omega, q, mu);
            }
        }
        {
            let mut acc: u32 = 1 % q;
            for j in 0..(n / 2) as usize {
                w_inv[j] = acc;
                acc = modmul(acc, iomega, q, mu);
            }
        }

        // Per-stage compacted tables.  Same algorithm as
        // sp_compute_ntt_twiddle.c::sp_tw_init_one (lines computing
        // w_fwd_stages / w_inv_stages).
        let log_n = ilog2(n);
        let mut w_fwd_stages = vec![0u32; (n - 1) as usize];
        let mut w_inv_stages = vec![0u32; (n - 1) as usize];
        let mut off: usize = 0;
        for s in 1..=log_n {
            let half_s = 1u32 << (s - 1);
            let step_s = n / (1u32 << s);
            for k in 0..half_s as usize {
                w_fwd_stages[off + k] = w_fwd[k * step_s as usize];
                w_inv_stages[off + k] = w_inv[k * step_s as usize];
            }
            off += half_s as usize;
        }
        assert_eq!(off, (n - 1) as usize);

        RefTables { psi_pow, ipsi_pow, w_fwd, w_inv, w_fwd_stages, w_inv_stages }
    }
}

#[cfg(target_os = "android")]
fn main() {
    use dsp_rpc::{make_scalars, FastRpcSession, RemoteArg, RemoteBuf};
    use ntt_2_ref::{build_ref, RefTables};
    use std::ffi::c_void;
    use std::time::Instant;

    eprintln!("[NTT.2] sp_ntt_2_smoke — twiddle VTCM staging gates");
    eprintln!("[NTT.2]   T_NTT2_TWIDDLE_INIT      (init + status query)");
    eprintln!("[NTT.2]   T_NTT2_TWIDDLE_BIT_EXACT (VTCM tables == host ref)");
    eprintln!("[NTT.2]   T_NTT2_VTCM_BUDGET       (≤ 2 MB peak across all (prime, N))");

    eprintln!("\n[NTT.2] opening FastRpcSession (Path B Unsigned PD)...");
    let sess = match FastRpcSession::new(
        "file:///libsp_compute_skel.so?sp_compute_skel_handle_invoke&_modver=1.0&_dom=cdsp",
    ) {
        Ok(s) => { eprintln!("[NTT.2] session open"); s }
        Err(e) => { eprintln!("[NTT.2] session FAIL: {e:?}"); std::process::exit(1); }
    };

    // ── method 14: ntt_twiddle_init (post-NTT.3-merge slot) ─────────────
    // NTT.5b correction: was 13 pre-merge; renumbered 13→14, 14→15, 15→16
    // by merge commit fec6fe3 (NTT.3 took slot 13 for ntt_hvx_oracle).
    // primIn = [N(i32)] (4 bytes). No data buffer.
    // scalars: method=14, inbufs=1, outbufs=0
    fn invoke_init(sess: &FastRpcSession, n: i32) -> Result<i64, String> {
        let mut prim_in: [u32; 1] = [n as u32];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 4 }},
        ];
        let t0 = Instant::now();
        sess.invoke(make_scalars(14, 1, 0), &mut args)
            .map_err(|e| format!("invoke method 14: {e:?}"))?;
        Ok(t0.elapsed().as_micros() as i64)
    }

    // ── method 15: ntt_twiddle_status (post-NTT.3-merge slot) ──────────
    // primIn  = [N(i32), q_idx(i32)] (8 bytes)
    // primOut = 9 × i32 (36 bytes):
    //   table_present, vtcm_addr_lo, vtcm_size, psi_pow_off, ipsi_pow_off,
    //   w_fwd_off, w_inv_off, w_fwd_stages_off, w_inv_stages_off
    // scalars: method=14, inbufs=1, outbufs=1
    #[derive(Debug, Default)]
    struct StatusOut {
        table_present: i32,
        vtcm_addr_lo: i32,
        vtcm_size: i32,
        psi_pow_off: i32,
        ipsi_pow_off: i32,
        w_fwd_off: i32,
        w_inv_off: i32,
        w_fwd_stages_off: i32,
        w_inv_stages_off: i32,
    }
    fn invoke_status(sess: &FastRpcSession, n: i32, q_idx: i32)
        -> Result<StatusOut, String>
    {
        let mut prim_in: [u32; 2] = [n as u32, q_idx as u32];
        let mut prim_out: [u32; 9] = [0; 9];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr()  as *mut c_void, nlen: 8  }},
            RemoteArg { buf: RemoteBuf { pv: prim_out.as_mut_ptr() as *mut c_void, nlen: 36 }},
        ];
        sess.invoke(make_scalars(15, 1, 1), &mut args)
            .map_err(|e| format!("invoke method 15: {e:?}"))?;
        Ok(StatusOut {
            table_present:    prim_out[0] as i32,
            vtcm_addr_lo:     prim_out[1] as i32,
            vtcm_size:        prim_out[2] as i32,
            psi_pow_off:      prim_out[3] as i32,
            ipsi_pow_off:     prim_out[4] as i32,
            w_fwd_off:        prim_out[5] as i32,
            w_inv_off:        prim_out[6] as i32,
            w_fwd_stages_off: prim_out[7] as i32,
            w_inv_stages_off: prim_out[8] as i32,
        })
    }

    // ── method 16: ntt_twiddle_dump (post-NTT.3-merge slot) ────────────
    // primIn = [N(i32), q_idx(i32), table_id(i32), dst_bufLen(i32)] (16 bytes)
    // dst_buf = OUTBUF (variable)
    // scalars: method=16, inbufs=1, outbufs=1
    fn invoke_dump(sess: &FastRpcSession, n: i32, q_idx: i32, table_id: i32,
                   expected_bytes: usize) -> Result<Vec<u32>, String>
    {
        let mut prim_in: [u32; 4] = [
            n as u32, q_idx as u32, table_id as u32, expected_bytes as u32,
        ];
        let mut dst_bytes: Vec<u8> = vec![0u8; expected_bytes];
        let mut args = [
            RemoteArg { buf: RemoteBuf { pv: prim_in.as_mut_ptr() as *mut c_void, nlen: 16 }},
            RemoteArg { buf: RemoteBuf { pv: dst_bytes.as_mut_ptr() as *mut c_void, nlen: expected_bytes }},
        ];
        sess.invoke(make_scalars(16, 1, 1), &mut args)
            .map_err(|e| format!("invoke method 16: {e:?}"))?;
        let words: Vec<u32> = dst_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(words)
    }

    // ── Drive init ─────────────────────────────────────────────────────
    eprintln!("\n[NTT.2] ── T_NTT2_TWIDDLE_INIT — drive ntt_twiddle_init(N=512) ──");
    let init_wall = match invoke_init(&sess, 512) {
        Ok(us) => { eprintln!("[NTT.2]   ntt_twiddle_init(N=512) ok ({us} us)"); us }
        Err(e) => { eprintln!("[NTT.2]   ntt_twiddle_init FAIL: {e}");
                    eprintln!("[NTT.2] gate T_NTT2_TWIDDLE_INIT FAIL");
                    std::process::exit(1); }
    };

    // Idempotency sanity: second call should no-op.
    let init_wall_2 = match invoke_init(&sess, 512) {
        Ok(us) => { eprintln!("[NTT.2]   ntt_twiddle_init(N=512) #2 ok ({us} us, expect << #1 due to idempotent fast path)"); us }
        Err(e) => { eprintln!("[NTT.2]   ntt_twiddle_init #2 FAIL: {e}"); -1 }
    };

    // ── Status query for all 6 combinations ────────────────────────────
    let combos = [(0i32, 128i32), (0, 256), (0, 512), (1, 128), (1, 256), (1, 512)];
    let mut statuses: Vec<(i32, i32, StatusOut)> = Vec::with_capacity(6);
    let mut init_ok = true;
    let mut total_vtcm_bytes: i64 = 0;
    let mut max_vtcm_per_n: [(i32, i64); 3] = [(128, 0), (256, 0), (512, 0)];
    let mut sum_vtcm_per_n: [(i32, i64); 3] = [(128, 0), (256, 0), (512, 0)];

    for &(q_idx, n) in &combos {
        match invoke_status(&sess, n, q_idx) {
            Ok(s) => {
                eprintln!("[NTT.2]   q_idx={q_idx} N={n}: present={} vtcm_addr_lo=0x{:08x} vtcm_size={} psi_pow_off={} ipsi_pow_off={} w_fwd_off={} w_inv_off={} w_fwd_stages_off={} w_inv_stages_off={}",
                          s.table_present, s.vtcm_addr_lo as u32, s.vtcm_size,
                          s.psi_pow_off, s.ipsi_pow_off, s.w_fwd_off,
                          s.w_inv_off, s.w_fwd_stages_off, s.w_inv_stages_off);
                if s.table_present != 1 || s.vtcm_addr_lo == 0 {
                    init_ok = false;
                }
                total_vtcm_bytes += s.vtcm_size as i64;
                let n_ix = if n == 128 { 0 } else if n == 256 { 1 } else { 2 };
                sum_vtcm_per_n[n_ix].1 += s.vtcm_size as i64;
                if (s.vtcm_size as i64) > max_vtcm_per_n[n_ix].1 {
                    max_vtcm_per_n[n_ix].1 = s.vtcm_size as i64;
                }
                statuses.push((q_idx, n, s));
            }
            Err(e) => {
                eprintln!("[NTT.2]   q_idx={q_idx} N={n}: status FAIL: {e}");
                init_ok = false;
                statuses.push((q_idx, n, StatusOut::default()));
            }
        }
    }

    let t_init_pass = init_ok;
    eprintln!("[NTT.2] init wall   = {init_wall} us  (first call)");
    eprintln!("[NTT.2] init wall#2 = {init_wall_2} us (idempotent)");
    eprintln!("[NTT.2] total_vtcm_bytes = {total_vtcm_bytes}");
    eprintln!("[NTT.2] T_NTT2_TWIDDLE_INIT {}", if t_init_pass { "PASS" } else { "FAIL" });

    // ── T_NTT2_VTCM_BUDGET ─────────────────────────────────────────────
    // Peak budget interpretation: at N=512 we hold 2 arenas (q1 + q2);
    // sum_vtcm_per_n[2].1 is "both primes at N=512 in VTCM".  We also hold
    // the smaller N arenas concurrently (all 6 arenas live simultaneously
    // for the lifetime of the daemon).  So the true peak = total_vtcm_bytes.
    const VTCM_BUDGET: i64 = 2 * 1024 * 1024;  // 2 MB
    let t_budget_pass = total_vtcm_bytes <= VTCM_BUDGET;
    eprintln!("\n[NTT.2] ── T_NTT2_VTCM_BUDGET ──");
    for &(n, sum) in &sum_vtcm_per_n {
        eprintln!("[NTT.2]   N={n}: sum_across_primes={} bytes", sum);
    }
    eprintln!("[NTT.2]   peak_vtcm_bytes_all_combos = {total_vtcm_bytes} (budget = {VTCM_BUDGET})");
    eprintln!("[NTT.2] T_NTT2_VTCM_BUDGET {}", if t_budget_pass { "PASS" } else { "FAIL" });

    // ── T_NTT2_TWIDDLE_BIT_EXACT ───────────────────────────────────────
    eprintln!("\n[NTT.2] ── T_NTT2_TWIDDLE_BIT_EXACT — dump each VTCM table + compare to host ref ──");
    let table_names = ["psi_pow", "ipsi_pow", "w_fwd", "w_inv", "w_fwd_stages", "w_inv_stages"];
    let mut tables_compared: u32 = 0;
    let mut total_bytes_compared: u64 = 0;
    let mut byte_divergences: u64 = 0;
    let mut first_div: Option<(i32, i32, i32, usize, u32, u32)> = None;

    for &(q_idx, n) in &combos {
        // Build host reference once per (q_idx, n).
        let r: RefTables = build_ref(n as u32, q_idx);
        let refs: [&[u32]; 6] = [
            &r.psi_pow, &r.ipsi_pow, &r.w_fwd, &r.w_inv,
            &r.w_fwd_stages, &r.w_inv_stages,
        ];
        let sizes: [usize; 6] = [
            (n * 4) as usize, (n * 4) as usize,
            (n * 2) as usize, (n * 2) as usize,
            ((n - 1) * 4) as usize, ((n - 1) * 4) as usize,
        ];
        for table_id in 0..6i32 {
            let expect_bytes = sizes[table_id as usize];
            let dumped: Vec<u32> = match invoke_dump(&sess, n, q_idx, table_id, expect_bytes) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[NTT.2]   dump FAIL q_idx={q_idx} N={n} table_id={table_id}: {e}");
                    byte_divergences += expect_bytes as u64;
                    continue;
                }
            };
            let expected: &[u32] = refs[table_id as usize];
            if dumped.len() != expected.len() {
                eprintln!("[NTT.2]   q_idx={q_idx} N={n} table={} ({}): dumped_len={} expected_len={}",
                          table_id, table_names[table_id as usize],
                          dumped.len(), expected.len());
                byte_divergences += expect_bytes as u64;
                continue;
            }
            tables_compared += 1;
            total_bytes_compared += expect_bytes as u64;
            let mut local_div: u32 = 0;
            for (lane, (&g, &e)) in dumped.iter().zip(expected.iter()).enumerate() {
                if g != e {
                    local_div += 1;
                    byte_divergences += 4;
                    if first_div.is_none() {
                        first_div = Some((q_idx, n, table_id, lane, g, e));
                    }
                }
            }
            eprintln!("[NTT.2]   q_idx={q_idx} N={n} table={} ({}): {} entries, diverged={}",
                      table_id, table_names[table_id as usize],
                      dumped.len(), local_div);
        }
    }

    let t_bit_exact_pass = byte_divergences == 0 && tables_compared == 36;
    eprintln!("\n[NTT.2]   tables_compared       = {} (expect 36 = 6 combos × 6 tables)", tables_compared);
    eprintln!("[NTT.2]   total_bytes_compared  = {}", total_bytes_compared);
    eprintln!("[NTT.2]   byte_divergences      = {}", byte_divergences);
    if let Some((q, n, tid, lane, g, e)) = first_div {
        eprintln!("[NTT.2]   first_divergence      = q_idx={q} N={n} table_id={tid} lane={lane} got={g} exp={e}");
    } else {
        eprintln!("[NTT.2]   first_divergence      = (none)");
    }
    eprintln!("[NTT.2] T_NTT2_TWIDDLE_BIT_EXACT {}", if t_bit_exact_pass { "PASS" } else { "FAIL" });

    drop(sess);
    eprintln!("\n[NTT.2] session closed cleanly");

    let pass = t_init_pass && t_budget_pass && t_bit_exact_pass;
    eprintln!("\n[NTT.2] ═══ Sprint NTT.2 aggregate ═══");
    eprintln!("[NTT.2]   T_NTT2_TWIDDLE_INIT       = {}", if t_init_pass      { "PASS" } else { "FAIL" });
    eprintln!("[NTT.2]   T_NTT2_TWIDDLE_BIT_EXACT  = {}", if t_bit_exact_pass { "PASS" } else { "FAIL" });
    eprintln!("[NTT.2]   T_NTT2_VTCM_BUDGET        = {}", if t_budget_pass    { "PASS" } else { "FAIL" });
    if pass {
        eprintln!("[NTT.2] ALL GATES PASS");
        std::process::exit(0);
    } else {
        eprintln!("[NTT.2] AT LEAST ONE GATE FAIL");
        std::process::exit(1);
    }
}
