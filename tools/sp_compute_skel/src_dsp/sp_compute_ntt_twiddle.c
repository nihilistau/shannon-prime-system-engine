/* sp_compute_ntt_twiddle.c — Sprint NTT.2 twiddle factor VTCM staging.
 *
 * Precomputes per-prime, per-N NTT twiddle tables once at daemon init and
 * pins them in VTCM via HAP_request_VTCM.  Subsequent NTT.1 HVX butterflies
 * and NTT.4 INTT consumers read stride-1 from the VTCM-resident tables
 * instead of re-deriving from `find_psi` per call (NTT.0's pattern).
 *
 * Tables stored per (prime ∈ {q1, q2}, N ∈ {128, 256, 512}):
 *   psi_pow[N]        — pre-weight psi^j   (4N bytes)
 *   ipsi_pow[N]       — post-weight psi^-j (4N bytes; NTT.4 consumes)
 *   w_fwd[N/2]        — forward omega^j    (2N bytes)
 *   w_inv[N/2]        — inverse omega^-j   (2N bytes; NTT.4 consumes)
 *   w_fwd_stages[N-1] — compacted per-stage forward twiddles  (4(N-1) bytes)
 *   w_inv_stages[N-1] — compacted per-stage inverse twiddles  (4(N-1) bytes)
 *
 * Compaction layout (stage-major, half_stage[s] = 2^{s-1}):
 *   offset_stage[s] = 4 * (2^{s-1} - 1) bytes
 *   size_stage[s]   = 4 * 2^{s-1}       bytes
 *   total per direction = 4 * (N - 1) bytes
 *
 * Per (prime, N) total ≈ 20N - 8 bytes; ≤ 10240 B at N=512.
 * Per prime across all 3 N values: ~17.9 KB.
 * Both primes across all 3 N values: ~35.8 KB.
 *
 * Storage policy:
 *   - One contiguous VTCM arena per (prime_idx, N) pair (6 arenas total).
 *   - Each arena is 4-byte aligned (HAP_request_VTCM returns page-aligned).
 *   - Idempotent init: second call no-ops if table_present.
 *   - On any per-arena init failure, all prior arenas are released and
 *     the ctx returns to the "not present" state.
 *
 * Math: identical to math-core `lib/.../core/ntt_crt/ntt_crt.c::prime_setup`
 * (lines 129-173) for byte-exactness vs the host-side reference under the
 * T_NTT2_TWIDDLE_BIT_EXACT gate.  Local Barrett primitives mirror
 * sp_compute_ntt_imp.c for SASS-audit isolation.
 */

#include <stdint.h>
#include <string.h>
#include "HAP_farf.h"
#include "HAP_vtcm_mgr.h"
#include "sp_compute.h"

/* Frozen primes (must match sp_compute_ntt_imp.c + sp_compute_crt_imp.c).
 * DO NOT include either file; keep this TU isolated for audit. */
#define SP_TW_Q1     1073738753u
#define SP_TW_Q2     1073732609u
#define SP_TW_MU_Q1  1073744895u
#define SP_TW_MU_Q2  1073751039u
#define SP_TW_Q_BITS 30u

#define SP_TW_N_MIN  128u
#define SP_TW_N_MAX  512u
#define SP_TW_N_NUM  3u     /* {128, 256, 512} */
#define SP_TW_Q_NUM  2u     /* {q1, q2} */

/* ─── modular arithmetic (Barrett, no 128-bit; byte-exact vs math-core) ── */

static inline uint64_t sp_tw_barrett_reduce(uint64_t x, uint64_t q, uint64_t mu) {
    uint64_t qhat = ((x >> (SP_TW_Q_BITS - 1u)) * mu) >> (SP_TW_Q_BITS + 1u);
    uint64_t r = x - qhat * q;
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

static inline uint32_t sp_tw_modmul(uint32_t a, uint32_t b, uint32_t q, uint64_t mu) {
    uint64_t x = (uint64_t)a * (uint64_t)b;
    return (uint32_t)sp_tw_barrett_reduce(x, (uint64_t)q, mu);
}

static uint32_t sp_tw_modpow(uint32_t base, uint64_t e, uint32_t q, uint64_t mu) {
    uint32_t result = 1u % q;
    uint32_t b = base % q;
    while (e) {
        if (e & 1u) result = sp_tw_modmul(result, b, q, mu);
        b = sp_tw_modmul(b, b, q, mu);
        e >>= 1;
    }
    return result;
}

static uint32_t sp_tw_modinv(uint32_t a, uint32_t q, uint64_t mu) {
    /* q is prime; a^{q-2} is the multiplicative inverse (Fermat). */
    return sp_tw_modpow(a, (uint64_t)q - 2u, q, mu);
}

/* Byte-identical to math-core find_psi (ntt_crt.c:116-127). */
static uint32_t sp_tw_find_psi(uint32_t N, uint32_t q, uint64_t mu) {
    uint64_t exp = ((uint64_t)q - 1u) / (2u * (uint64_t)N);
    for (uint32_t a = 2u; a < q; a++) {
        uint32_t psi = sp_tw_modpow(a, exp, q, mu);
        if (psi == 0u) continue;
        uint32_t pN = sp_tw_modpow(psi, (uint64_t)N, q, mu);
        if (pN == q - 1u) return psi;
    }
    return 0u;
}

static inline uint32_t sp_tw_ilog2(uint32_t n) {
    uint32_t l = 0;
    while ((1u << l) < n) l++;
    return l;
}

/* ─── per-(prime, N) twiddle context ───────────────────────────────────── */

typedef struct {
    int         present;        /* 0 = not initialized, 1 = ready */
    uint32_t    N;              /* {128, 256, 512} */
    uint32_t    q;              /* SP_TW_Q1 or SP_TW_Q2 */
    uint64_t    mu;
    uint32_t    psi;
    uint32_t    ipsi;
    uint32_t    ninv;

    void       *vtcm_base;      /* HAP_request_VTCM return pointer */
    uint32_t    vtcm_size;      /* total bytes of this arena */

    /* offsets into vtcm_base for each sub-table (in bytes) */
    uint32_t    off_psi_pow;
    uint32_t    off_ipsi_pow;
    uint32_t    off_w_fwd;
    uint32_t    off_w_inv;
    uint32_t    off_w_fwd_stages;
    uint32_t    off_w_inv_stages;
} sp_tw_one_ctx;

/* 6 contexts: (prime_idx, N_idx) ∈ {0,1} × {0,1,2}.
 * N_idx 0 → N=128, 1 → N=256, 2 → N=512. */
static sp_tw_one_ctx g_tw_ctx[SP_TW_Q_NUM][SP_TW_N_NUM];
static int           g_tw_init_attempted = 0;

static inline uint32_t sp_tw_n_idx(uint32_t N) {
    if (N == 128u) return 0u;
    if (N == 256u) return 1u;
    if (N == 512u) return 2u;
    return 0xFFFFFFFFu;
}

/* Compute the offset of stage s (s starts at 1) into the compacted region.
 * offset_stage[s] = 4 * (2^{s-1} - 1) bytes; we return *entries* (uint32). */
static inline uint32_t sp_tw_stage_offset_entries(uint32_t s) {
    /* s in [1, logN]; 2^{s-1} - 1 entries before stage s. */
    return (1u << (s - 1u)) - 1u;
}

/* Total compacted entries for radix-2 DIT on length N: N - 1. */
static inline uint32_t sp_tw_compacted_entries(uint32_t N) {
    return N - 1u;
}

/* Compute the round-up byte size of one (prime, N) arena. */
static uint32_t sp_tw_arena_size_bytes(uint32_t N) {
    uint32_t bytes = 0;
    bytes += 4u * N;                            /* psi_pow */
    bytes += 4u * N;                            /* ipsi_pow */
    bytes += 4u * (N / 2u);                     /* w_fwd */
    bytes += 4u * (N / 2u);                     /* w_inv */
    bytes += 4u * (N - 1u);                     /* w_fwd_stages */
    bytes += 4u * (N - 1u);                     /* w_inv_stages */
    /* Round up to 128 bytes for HVX vector alignment of all sub-tables. */
    bytes = (bytes + 127u) & ~127u;
    return bytes;
}

/* Initialize one (prime, N) context.  Returns 1 on success, 0 on failure
 * (VTCM denied, find_psi failed, etc).  Allocates one VTCM arena via
 * HAP_request_VTCM and populates all six tables. */
static int sp_tw_init_one(sp_tw_one_ctx *ctx, uint32_t q_idx, uint32_t N) {
    if (ctx->present) return 1;  /* idempotent */

    if (sp_tw_n_idx(N) == 0xFFFFFFFFu) {
        FARF(RUNTIME_ERROR, "sp_tw_init_one: invalid N=%u", N);
        return 0;
    }
    if (q_idx > 1u) {
        FARF(RUNTIME_ERROR, "sp_tw_init_one: invalid q_idx=%u", q_idx);
        return 0;
    }

    uint32_t q  = (q_idx == 0u) ? SP_TW_Q1 : SP_TW_Q2;
    uint64_t mu = (q_idx == 0u) ? (uint64_t)SP_TW_MU_Q1 : (uint64_t)SP_TW_MU_Q2;

    uint32_t psi = sp_tw_find_psi(N, q, mu);
    if (psi == 0u) {
        FARF(RUNTIME_ERROR, "sp_tw_init_one: find_psi failed q_idx=%u N=%u",
             q_idx, N);
        return 0;
    }

    /* Allocate one VTCM arena for all six tables. */
    uint32_t arena_size = sp_tw_arena_size_bytes(N);
    void *p = HAP_request_VTCM(arena_size, 0u);
    if (!p) {
        FARF(RUNTIME_ERROR,
             "sp_tw_init_one: HAP_request_VTCM(%u, 0) denied for q_idx=%u N=%u",
             arena_size, q_idx, N);
        return 0;
    }

    /* Lay out the six sub-tables sequentially inside the arena.
     * All offsets are 4-byte aligned (sub-table sizes are multiples of 4).
     * The arena base is page-aligned (HAP_request_VTCM contract) so the
     * first sub-table is also 128-byte aligned; subsequent sub-tables are
     * 4-byte aligned. Per-sub-table 128-byte alignment is NOT required —
     * the HVX butterfly's vmem reads happen on the COMPACTED stage tables
     * (each starting at a stage offset that may not be 128-aligned within
     * the arena); NTT.1's loads will use the appropriate vmem/vmemu form
     * based on alignment. The pre-weight loop reads psi_pow[j] in stride
     * 1 (4-byte stride elements), no alignment requirement at the i32
     * level. */
    ctx->vtcm_base       = p;
    ctx->vtcm_size       = arena_size;
    ctx->off_psi_pow     = 0u;
    ctx->off_ipsi_pow    = ctx->off_psi_pow     + 4u * N;
    ctx->off_w_fwd       = ctx->off_ipsi_pow    + 4u * N;
    ctx->off_w_inv       = ctx->off_w_fwd       + 4u * (N / 2u);
    ctx->off_w_fwd_stages = ctx->off_w_inv       + 4u * (N / 2u);
    ctx->off_w_inv_stages = ctx->off_w_fwd_stages + 4u * (N - 1u);

    uint32_t *psi_pow      = (uint32_t *)((uint8_t *)p + ctx->off_psi_pow);
    uint32_t *ipsi_pow     = (uint32_t *)((uint8_t *)p + ctx->off_ipsi_pow);
    uint32_t *w_fwd        = (uint32_t *)((uint8_t *)p + ctx->off_w_fwd);
    uint32_t *w_inv        = (uint32_t *)((uint8_t *)p + ctx->off_w_inv);
    uint32_t *w_fwd_stages = (uint32_t *)((uint8_t *)p + ctx->off_w_fwd_stages);
    uint32_t *w_inv_stages = (uint32_t *)((uint8_t *)p + ctx->off_w_inv_stages);

    uint32_t ipsi   = sp_tw_modinv(psi, q, mu);
    uint32_t omega  = sp_tw_modmul(psi,  psi,  q, mu);
    uint32_t iomega = sp_tw_modmul(ipsi, ipsi, q, mu);
    uint32_t ninv   = sp_tw_modinv(N % q, q, mu);

    /* psi_pow[j] = psi^j */
    {
        uint32_t acc = 1u % q;
        for (uint32_t j = 0; j < N; j++) {
            psi_pow[j] = acc;
            acc = sp_tw_modmul(acc, psi, q, mu);
        }
    }
    /* ipsi_pow[j] = psi^{-j} */
    {
        uint32_t acc = 1u % q;
        for (uint32_t j = 0; j < N; j++) {
            ipsi_pow[j] = acc;
            acc = sp_tw_modmul(acc, ipsi, q, mu);
        }
    }
    /* w_fwd[j] = omega^j  for j in [0, N/2) */
    {
        uint32_t acc = 1u % q;
        for (uint32_t j = 0; j < N / 2u; j++) {
            w_fwd[j] = acc;
            acc = sp_tw_modmul(acc, omega, q, mu);
        }
    }
    /* w_inv[j] = omega^{-j} for j in [0, N/2) */
    {
        uint32_t acc = 1u % q;
        for (uint32_t j = 0; j < N / 2u; j++) {
            w_inv[j] = acc;
            acc = sp_tw_modmul(acc, iomega, q, mu);
        }
    }

    /* Per-stage compacted twiddle tables.  Stage s in [1, logN]; stage s
     * has half_s = 2^{s-1} twiddles, each at stride step_s = N / 2^s from
     * the base w_fwd / w_inv tables.  We copy w_*[k * step_s] into the
     * compacted region for k in [0, half_s), making stage access stride-1.
     *
     * Layout matches math-core ntt_core (ntt_crt.c:241-254) which reads
     * `wtab[widx]` where widx steps by `step = N / len` per butterfly; the
     * outer butterfly is exactly the per-stage compaction we are doing here.
     */
    uint32_t logN = sp_tw_ilog2(N);
    {
        uint32_t off_entries = 0;  /* entries from arena's compacted base */
        for (uint32_t s = 1u; s <= logN; s++) {
            uint32_t half_s = 1u << (s - 1u);
            uint32_t step_s = N / (1u << s);  /* = N / (2 * half_s) */
            for (uint32_t k = 0; k < half_s; k++) {
                w_fwd_stages[off_entries + k] = w_fwd[k * step_s];
                w_inv_stages[off_entries + k] = w_inv[k * step_s];
            }
            off_entries += half_s;
        }
        /* sanity: off_entries should equal N - 1 */
        if (off_entries != N - 1u) {
            FARF(RUNTIME_ERROR,
                 "sp_tw_init_one: compaction entry count mismatch off=%u N-1=%u",
                 off_entries, N - 1u);
            HAP_release_VTCM(p);
            ctx->vtcm_base = 0;
            ctx->vtcm_size = 0;
            return 0;
        }
    }

    ctx->present = 1;
    ctx->N       = N;
    ctx->q       = q;
    ctx->mu      = mu;
    ctx->psi     = psi;
    ctx->ipsi    = ipsi;
    ctx->ninv    = ninv;

    FARF(RUNTIME_HIGH,
         "sp_tw_init_one: ok q_idx=%u N=%u psi=%u ipsi=%u ninv=%u "
         "vtcm_base=%p vtcm_size=%u",
         q_idx, N, psi, ipsi, ninv, p, arena_size);
    return 1;
}

/* Initialize all 6 (prime_idx, N) contexts for the requested N (and the
 * other two N values for free — daemon startup calls this once for the
 * widest N and gets {128, 256, 512} populated).
 *
 * Idempotent: if all 6 are already present, returns SP_OK immediately.
 * On partial init failure, releases any successfully-allocated arenas
 * and resets g_tw_ctx so a retry starts from clean state.
 */
static int sp_tw_init_all(void) {
    int all_ok = 1;
    for (uint32_t qi = 0; qi < SP_TW_Q_NUM; qi++) {
        for (uint32_t ni = 0; ni < SP_TW_N_NUM; ni++) {
            uint32_t N = (ni == 0u) ? 128u : (ni == 1u) ? 256u : 512u;
            if (!sp_tw_init_one(&g_tw_ctx[qi][ni], qi, N)) {
                all_ok = 0;
                break;
            }
        }
        if (!all_ok) break;
    }
    if (!all_ok) {
        /* Roll back: release whatever was successfully allocated. */
        for (uint32_t qi = 0; qi < SP_TW_Q_NUM; qi++) {
            for (uint32_t ni = 0; ni < SP_TW_N_NUM; ni++) {
                sp_tw_one_ctx *c = &g_tw_ctx[qi][ni];
                if (c->present && c->vtcm_base) {
                    HAP_release_VTCM(c->vtcm_base);
                }
                memset(c, 0, sizeof(*c));
            }
        }
        return -1;
    }
    g_tw_init_attempted = 1;
    return 0;
}

/* ─── IDL handler — sp_compute_ntt_twiddle_init (method 14) ───────────── */

/* Signature (per IDL):
 *   long ntt_twiddle_init(in long N) -> long
 *
 * Semantics: triggers init of ALL 6 (prime, N) tables for N ∈
 * {128, 256, 512}.  The `N` parameter is currently informational — the
 * implementation always initializes the full table set so the ARM-side
 * daemon doesn't need three separate calls.  Future revisions may use
 * `N` to support a subset.
 *
 * Returns 0 on success (all 6 tables present after the call), -1 on any
 * init failure (VTCM denial or find_psi failure on any combo).  All-or-
 * nothing: on failure, no tables remain allocated.
 */
int sp_compute_ntt_twiddle_init(remote_handle64 h, int N)
{
    (void)h;
    if (N != 128 && N != 256 && N != 512) {
        FARF(RUNTIME_ERROR, "sp_compute_ntt_twiddle_init: invalid N=%d", N);
        return -1;
    }

    /* Check if all tables already present (idempotent fast path). */
    int all_present = 1;
    for (uint32_t qi = 0; qi < SP_TW_Q_NUM && all_present; qi++) {
        for (uint32_t ni = 0; ni < SP_TW_N_NUM && all_present; ni++) {
            if (!g_tw_ctx[qi][ni].present) all_present = 0;
        }
    }
    if (all_present) {
        FARF(RUNTIME_HIGH,
             "sp_compute_ntt_twiddle_init: N=%d -- all tables already present (idempotent no-op)", N);
        return 0;
    }

    int rc = sp_tw_init_all();
    if (rc != 0) {
        FARF(RUNTIME_ERROR,
             "sp_compute_ntt_twiddle_init: sp_tw_init_all rc=%d (N requested=%d)", rc, N);
        return -1;
    }
    FARF(RUNTIME_HIGH,
         "sp_compute_ntt_twiddle_init: N=%d -- all 6 (prime, N) tables initialized in VTCM", N);
    return 0;
}

/* ─── IDL handler — sp_compute_ntt_twiddle_status (method 15) ────────── */

/* Signature (per IDL):
 *   long ntt_twiddle_status(in long N, in long q_idx,
 *                           rout long table_present,
 *                           rout long vtcm_addr_lo,
 *                           rout long vtcm_size,
 *                           rout long psi_pow_off,
 *                           rout long ipsi_pow_off,
 *                           rout long w_fwd_off,
 *                           rout long w_inv_off,
 *                           rout long w_fwd_stages_off,
 *                           rout long w_inv_stages_off) -> long
 *
 * Reports the inspection metadata for one (prime, N) combination.
 * table_present=1 iff the arena is populated; 0 otherwise (other rout
 * fields are zero in that case).  vtcm_addr_lo is the low-32 bits of the
 * arena base pointer (matches the sp_compute_vtcm_probe convention).
 * Sub-table offsets are byte offsets within the arena.
 *
 * Returns 0 on success (valid q_idx + N + non-null routs), -1 on input
 * validation failure.
 */
int sp_compute_ntt_twiddle_status(remote_handle64 h,
                                  int N, int q_idx,
                                  int *table_present,
                                  int *vtcm_addr_lo,
                                  int *vtcm_size,
                                  int *psi_pow_off,
                                  int *ipsi_pow_off,
                                  int *w_fwd_off,
                                  int *w_inv_off,
                                  int *w_fwd_stages_off,
                                  int *w_inv_stages_off)
{
    (void)h;
    if (!table_present || !vtcm_addr_lo || !vtcm_size ||
        !psi_pow_off || !ipsi_pow_off || !w_fwd_off ||
        !w_inv_off || !w_fwd_stages_off || !w_inv_stages_off) {
        return -1;
    }
    /* Default everything to 0 in case of bad input. */
    *table_present     = 0;
    *vtcm_addr_lo      = 0;
    *vtcm_size         = 0;
    *psi_pow_off       = 0;
    *ipsi_pow_off      = 0;
    *w_fwd_off         = 0;
    *w_inv_off         = 0;
    *w_fwd_stages_off  = 0;
    *w_inv_stages_off  = 0;

    if (q_idx < 0 || q_idx > 1) return -1;
    if (N != 128 && N != 256 && N != 512) return -1;

    uint32_t ni = sp_tw_n_idx((uint32_t)N);
    const sp_tw_one_ctx *c = &g_tw_ctx[q_idx][ni];

    *table_present     = c->present ? 1 : 0;
    *vtcm_addr_lo      = (int)(uintptr_t)c->vtcm_base;
    *vtcm_size         = (int)c->vtcm_size;
    *psi_pow_off       = (int)c->off_psi_pow;
    *ipsi_pow_off      = (int)c->off_ipsi_pow;
    *w_fwd_off         = (int)c->off_w_fwd;
    *w_inv_off         = (int)c->off_w_inv;
    *w_fwd_stages_off  = (int)c->off_w_fwd_stages;
    *w_inv_stages_off  = (int)c->off_w_inv_stages;

    FARF(RUNTIME_HIGH,
         "sp_compute_ntt_twiddle_status: q_idx=%d N=%d present=%d vtcm_size=%d",
         q_idx, N, *table_present, *vtcm_size);
    return 0;
}

/* ─── public accessor for NTT.1 / NTT.4 (skel-internal C) ─────────────── */

/* Future skel-side consumers (sp_compute_ntt_hvx_imp.c, sp_compute_ntt_intt
 * _imp.c) need read access to the populated tables.  Expose a strongly-
 * typed accessor returning per-sub-table pointers.  Returns 0 on success;
 * -1 if (q_idx, N) is not present (caller should invoke ntt_twiddle_init
 * first).
 *
 * NOT exposed via IDL — pure C-internal API.
 */
typedef struct {
    uint32_t        N;
    uint32_t        q;
    uint64_t        mu;
    uint32_t        ninv;
    const uint32_t *psi_pow;
    const uint32_t *ipsi_pow;
    const uint32_t *w_fwd;
    const uint32_t *w_inv;
    const uint32_t *w_fwd_stages;
    const uint32_t *w_inv_stages;
} sp_tw_view;

int sp_compute_ntt_twiddle_view(int q_idx, int N, sp_tw_view *out)
{
    if (!out) return -1;
    if (q_idx < 0 || q_idx > 1) return -1;
    if (N != 128 && N != 256 && N != 512) return -1;

    uint32_t ni = sp_tw_n_idx((uint32_t)N);
    const sp_tw_one_ctx *c = &g_tw_ctx[q_idx][ni];
    if (!c->present) return -1;

    out->N     = c->N;
    out->q     = c->q;
    out->mu    = c->mu;
    out->ninv  = c->ninv;
    out->psi_pow      = (const uint32_t *)((const uint8_t *)c->vtcm_base + c->off_psi_pow);
    out->ipsi_pow     = (const uint32_t *)((const uint8_t *)c->vtcm_base + c->off_ipsi_pow);
    out->w_fwd        = (const uint32_t *)((const uint8_t *)c->vtcm_base + c->off_w_fwd);
    out->w_inv        = (const uint32_t *)((const uint8_t *)c->vtcm_base + c->off_w_inv);
    out->w_fwd_stages = (const uint32_t *)((const uint8_t *)c->vtcm_base + c->off_w_fwd_stages);
    out->w_inv_stages = (const uint32_t *)((const uint8_t *)c->vtcm_base + c->off_w_inv_stages);
    return 0;
}
