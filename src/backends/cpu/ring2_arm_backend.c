/* ring2_arm_backend.c — Stage C + fusion: the proven Optane NO_BUFFERING+IOCP
 * store (ring2_disk.c, C2.1 Step 2b: 7.57 us/read amortized) wrapped as the
 * math-core ARM Ring-2 backend and registered through the L1 hook.
 *
 * DUAL-BLOCK-SIZE: under the NTT fusion the two streams differ — K blocks are
 * the dual-prime residue unit (NKV * 2N * u32, e.g. 8192 B on Qwen3) while V
 * stays f32 plumbing (KVD * 4, e.g. 4096 B). ring2_disk owns ONE block size
 * per store, so the wrapper opens TWO independent stores (one per stream,
 * distinct filename prefixes — ring2_disk concatenates dir+name raw, so
 * "F:\\armk_" / "F:\\armv_" never collide) and routes by `which`. Batched
 * reads are split per stream; each sub-batch keeps the IOCP queue depth.
 *
 * Hook mapping (per stream):
 *   write_block    -> ring2_disk_write       (aligned bounce inside the store)
 *   read_block     -> ring2_disk_read        (scratch read + copy out)
 *   read_batch     -> ring2_disk_read_batch  (split by which; dst buffers are
 *                     sector-aligned because the decode gets its staging from
 *                     our alloc_aligned hook)
 *   alloc/free_aligned -> ring2_disk_{alloc,free}_aligned (4 KB direct-I/O)
 *   close          -> stats + teardown (only on _unregister: the decode treats
 *                     a REGISTERED backend as borrowed)
 */
#include "sp_engine/ring2_arm.h"
#include "sp_engine/ring2_disk.h"
#include "sp/arm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <process.h>   /* _beginthreadex — the device-overlap worker */
#endif

/* ── the temporal-locality staging cache (v5; SP_RING2_CACHE_MB, 0 = OFF) ────
 * v4 measured the temporal tax: 9.46 TB pulled to serve ~3 GB of unique
 * blocks at 32k — adjacent tokens' recall sets DRIFT, so the router without
 * memory re-fetches the same hot blocks every step. This is a bounded LRU
 * over Ring-2 blocks, one slab per stream (uniform block size each), keyed
 * by block offset. CONCURRENCY CONTRACT (pinned before code): every cache
 * mutation happens on the CALLER thread — probe/hit-copy before dispatch,
 * insert after join; the V-stream worker thread never touches the cache.
 * Lock-free by construction, not by cleverness. Writes are write-through +
 * write-allocate (a freshly spilled position is exactly what recall wants
 * back). OFF (default) = the proven path, bit-identical. Gate T_CACHE_EXACT:
 * cached decode == uncached decode, identical sequence. */
typedef struct {
    uint8_t  *slab;          /* nslot * blk bytes                       */
    uint64_t *off;           /* slot -> block offset                    */
    int      *hash;          /* open-address table: off -> slot (+1)    */
    int      *lru_prev, *lru_next;                 /* doubly linked LRU */
    int       nslot, hcap, used, lru_head, lru_tail;
    unsigned long long hits, misses;
} blk_cache;

typedef struct {
    ring2_disk    *r[2];     /* [0]=K stream store, [1]=V stream store */
    ring2_scratch *sc[2];    /* serial decode: one scratch per store   */
    size_t         blk[2];   /* per-stream block bytes                 */
    blk_cache      cache[2]; /* temporal staging cache (nslot==0 = off) */
} optane_be;

static uint32_t bc_hash64(uint64_t x) {        /* splitmix64 finalizer */
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    return (uint32_t)(x ^ (x >> 31));
}

static void bc_init(blk_cache *c, size_t bytes, size_t blk) {
    memset(c, 0, sizeof(*c));
    c->lru_head = c->lru_tail = -1;
    int nslot = (int)(bytes / blk);
    if (nslot < 8) return;                      /* too small: stay off */
    int hcap = 1; while (hcap < nslot * 2) hcap <<= 1;
    c->slab     = (uint8_t *)malloc((size_t)nslot * blk);
    c->off      = (uint64_t *)malloc((size_t)nslot * sizeof(uint64_t));
    c->hash     = (int *)calloc((size_t)hcap, sizeof(int));
    c->lru_prev = (int *)malloc((size_t)nslot * sizeof(int));
    c->lru_next = (int *)malloc((size_t)nslot * sizeof(int));
    if (!c->slab || !c->off || !c->hash || !c->lru_prev || !c->lru_next) {
        free(c->slab); free(c->off); free(c->hash); free(c->lru_prev); free(c->lru_next);
        memset(c, 0, sizeof(*c)); c->lru_head = c->lru_tail = -1;
        return;
    }
    c->nslot = nslot; c->hcap = hcap;
}

static void bc_free(blk_cache *c) {
    free(c->slab); free(c->off); free(c->hash); free(c->lru_prev); free(c->lru_next);
    memset(c, 0, sizeof(*c));
}

static int bc_find(blk_cache *c, uint64_t off) {   /* slot or -1 */
    if (!c->nslot) return -1;
    uint32_t h = bc_hash64(off) & (uint32_t)(c->hcap - 1);
    while (c->hash[h]) {
        int s = c->hash[h] - 1;
        if (c->off[s] == off) return s;
        h = (h + 1) & (uint32_t)(c->hcap - 1);
    }
    return -1;
}

static void bc_lru_unlink(blk_cache *c, int s) {
    if (c->lru_prev[s] >= 0) c->lru_next[c->lru_prev[s]] = c->lru_next[s];
    else c->lru_head = c->lru_next[s];
    if (c->lru_next[s] >= 0) c->lru_prev[c->lru_next[s]] = c->lru_prev[s];
    else c->lru_tail = c->lru_prev[s];
}

static void bc_lru_push_front(blk_cache *c, int s) {
    c->lru_prev[s] = -1; c->lru_next[s] = c->lru_head;
    if (c->lru_head >= 0) c->lru_prev[c->lru_head] = s;
    c->lru_head = s;
    if (c->lru_tail < 0) c->lru_tail = s;
}

static void bc_touch(blk_cache *c, int s) { bc_lru_unlink(c, s); bc_lru_push_front(c, s); }

static void bc_hash_remove(blk_cache *c, uint64_t off) {
    uint32_t h = bc_hash64(off) & (uint32_t)(c->hcap - 1);
    while (c->hash[h]) {
        int s = c->hash[h] - 1;
        if (c->off[s] == off) {
            /* open-address deletion: clear, then re-insert the probe chain */
            c->hash[h] = 0;
            uint32_t j = (h + 1) & (uint32_t)(c->hcap - 1);
            while (c->hash[j]) {
                int s2 = c->hash[j] - 1; c->hash[j] = 0;
                uint32_t h2 = bc_hash64(c->off[s2]) & (uint32_t)(c->hcap - 1);
                while (c->hash[h2]) h2 = (h2 + 1) & (uint32_t)(c->hcap - 1);
                c->hash[h2] = s2 + 1;
                j = (j + 1) & (uint32_t)(c->hcap - 1);
            }
            return;
        }
        h = (h + 1) & (uint32_t)(c->hcap - 1);
    }
}

/* insert/update off -> data (CALLER THREAD ONLY) */
static void bc_put(blk_cache *c, uint64_t off, const void *data, size_t blk) {
    if (!c->nslot) return;
    int s = bc_find(c, off);
    if (s >= 0) {                                /* update in place */
        memcpy(c->slab + (size_t)s * blk, data, blk);
        bc_touch(c, s);
        return;
    }
    if (c->used < c->nslot) s = c->used++;
    else {                                       /* evict LRU tail */
        s = c->lru_tail;
        bc_lru_unlink(c, s);
        bc_hash_remove(c, c->off[s]);
    }
    c->off[s] = off;
    memcpy(c->slab + (size_t)s * blk, data, blk);
    uint32_t h = bc_hash64(off) & (uint32_t)(c->hcap - 1);
    while (c->hash[h]) h = (h + 1) & (uint32_t)(c->hcap - 1);
    c->hash[h] = s + 1;
    bc_lru_push_front(c, s);
}

static optane_be *g_obe = NULL;   /* singleton: registered store pair */

static int obe_write(void *h, int which, uint64_t off, const void *src, size_t len) {
    optane_be *b = (optane_be *)h;
    if (!b || which < 0 || which > 1 || len != b->blk[which]) return 1;
    if (ring2_disk_write(b->r[which], which, (size_t)off, src)) return 1;
    /* write-through + write-allocate: the freshly spilled position is exactly
     * what recall asks for next (caller thread — the decode's write path). */
    bc_put(&b->cache[which], off, src, b->blk[which]);
    return 0;
}

static int obe_read(void *h, int which, uint64_t off, void *dst, size_t len) {
    optane_be *b = (optane_be *)h;
    if (!b || which < 0 || which > 1 || len != b->blk[which]) return 1;
    blk_cache *c = &b->cache[which];
    int s = bc_find(c, off);
    if (s >= 0) {
        memcpy(dst, c->slab + (size_t)s * b->blk[which], len);
        bc_touch(c, s); c->hits++;
        return 0;
    }
    const void *p = ring2_disk_read(b->r[which], which, (size_t)off, b->sc[which]);
    if (!p) return 1;
    memcpy(dst, p, len);
    if (c->nslot) { bc_put(c, off, dst, b->blk[which]); c->misses++; }
    return 0;
}

static int obe_read_batch(void *h, const int *which, const uint64_t *off,
                          void *const *dst, size_t len, int n) {
    optane_be *b = (optane_be *)h;
    if (!b) return 1;
    ring2_req *reqs = (ring2_req *)malloc((size_t)n * sizeof(ring2_req));
    if (!reqs) return 1;
    /* split per stream so each sub-batch hits ONE store's IOCP at full depth;
     * len applies to every request in the caller's batch and must match the
     * stream's block size (the fused decode already issues per-stream batches). */
    int rc = 0;
    for (int w = 0; w < 2 && rc == 0; w++) {
        int m = 0;
        for (int i = 0; i < n; i++) {
            if (which[i] != w) continue;
            if (len != b->blk[w]) { rc = 1; break; }
            reqs[m].which = w;
            reqs[m].off   = (size_t)off[i];
            reqs[m].dst   = dst[i];          /* sector-aligned via our alloc hook */
            m++;
        }
        if (rc == 0 && m > 0) rc = ring2_disk_read_batch(b->r[w], reqs, m);
    }
    free(reqs);
    return rc;
}

/* ── the device-overlap fix: mixed-stream batch, two queues CONCURRENT ──────
 * v3 measured the serialization tax live: two sequential read_batch calls on
 * split asymmetric devices = 4u/S vs single-device 3u/S (the stoplight on the
 * dual-lane highway). Here the V-stream sub-batch runs on a worker thread
 * while the K-stream sub-batch runs on the caller — max(2u/S, 2u/S) = 2u/S.
 * Each inner ring2_disk store owns its own handles/IOCP, so concurrent batch
 * calls on the TWO DIFFERENT stores are safe. Byte-exactness untouched:
 * identical reads, concurrent issue. */
typedef struct { ring2_disk *r; ring2_req *reqs; int m; int rc; } obe_job;

#ifdef _WIN32
static unsigned __stdcall obe_job_thread(void *arg) {
    obe_job *j = (obe_job *)arg;
    j->rc = ring2_disk_read_batch(j->r, j->reqs, j->m);
    return 0;
}
#endif

static int obe_read_batch2(void *h, const int *which, const uint64_t *off,
                           void *const *dst, const size_t len_by_stream[2], int n) {
    optane_be *b = (optane_be *)h;
    if (!b) return 1;
    ring2_req *reqs = (ring2_req *)malloc((size_t)n * sizeof(ring2_req));
    if (!reqs) return 1;
    int m[2] = { 0, 0 };               /* partition: stream 0 grows from the   */
    int rc = 0;                        /* front, stream 1 from the back        */
    for (int i = 0; i < n && rc == 0; i++) {
        int w = which[i];
        if (w < 0 || w > 1 || len_by_stream[w] != b->blk[w]) { rc = 1; break; }
        /* CACHE PROBE — caller thread only. Hits are served immediately and
         * never enter the device batch; misses go to the per-stream queue. */
        blk_cache *c = &b->cache[w];
        int s = bc_find(c, off[i]);
        if (s >= 0) {
            memcpy(dst[i], c->slab + (size_t)s * b->blk[w], b->blk[w]);
            bc_touch(c, s); c->hits++;
            continue;
        }
        if (c->nslot) c->misses++;
        int slot = (w == 0) ? m[0]++ : (n - 1 - m[1]++);
        reqs[slot].which = w;
        reqs[slot].off   = (size_t)off[i];
        reqs[slot].dst   = dst[i];
    }
    if (rc == 0) {
        obe_job jv = { b->r[1], reqs + (n - m[1]), m[1], 0 };
        int rc_k = 0, rc_v = 0;
#ifdef _WIN32
        HANDLE th = NULL;
        if (m[1] > 0)
            th = (HANDLE)_beginthreadex(NULL, 0, obe_job_thread, &jv, 0, NULL);
        if (m[0] > 0) rc_k = ring2_disk_read_batch(b->r[0], reqs, m[0]);
        if (th) { WaitForSingleObject(th, INFINITE); CloseHandle(th); rc_v = jv.rc; }
        else if (m[1] > 0)              /* thread creation failed: serial fallback */
            rc_v = ring2_disk_read_batch(b->r[1], jv.reqs, jv.m);
#else
        if (m[0] > 0) rc_k = ring2_disk_read_batch(b->r[0], reqs, m[0]);
        if (m[1] > 0) rc_v = ring2_disk_read_batch(b->r[1], jv.reqs, jv.m);
#endif
        rc = rc_k ? rc_k : rc_v;
        /* INSERT after join — caller thread only (the worker is gone). */
        if (rc == 0) {
            for (int i = 0; i < m[0]; i++)
                bc_put(&b->cache[0], (uint64_t)reqs[i].off, reqs[i].dst, b->blk[0]);
            for (int i = 0; i < m[1]; i++) {
                ring2_req *rq = reqs + (n - 1 - i);
                bc_put(&b->cache[1], (uint64_t)rq->off, rq->dst, b->blk[1]);
            }
        }
    }
    free(reqs);
    return rc;
}

static void *obe_alloc(void *h, size_t bytes) { (void)h; return ring2_disk_alloc_aligned(bytes); }
static void  obe_free(void *h, void *p)        { (void)h; ring2_disk_free_aligned(p); }

static void obe_close(void *h) {       /* called only from _unregister */
    optane_be *b = (optane_be *)h;
    if (!b) return;
    for (int w = 0; w < 2; w++) {
        unsigned long long nr = 0; double rs = 0;
        ring2_disk_stats(b->r[w], &nr, &rs);
        fprintf(stderr, "    [ring2-optane] %c-stream: %llu reads, %.3f s, %.2f us/read avg\n",
                w == 0 ? 'K' : 'V', nr, rs, nr ? rs * 1e6 / (double)nr : 0.0);
        if (b->cache[w].nslot) {
            unsigned long long tot = b->cache[w].hits + b->cache[w].misses;
            fprintf(stderr, "    [ring2-cache] %c-stream: %llu hits / %llu misses = %.1f%% hit-rate "
                    "(%llu device reads avoided)\n",
                    w == 0 ? 'K' : 'V', b->cache[w].hits, b->cache[w].misses,
                    tot ? 100.0 * (double)b->cache[w].hits / (double)tot : 0.0,
                    b->cache[w].hits);
        }
        bc_free(&b->cache[w]);
        ring2_disk_scratch_free(b->sc[w]);
        ring2_disk_close(b->r[w]);
    }
    free(b);
}

/* Split-device variant: the K-stream store under dir_k, the V-stream store
 * under dir_v — two physically independent NVMe controllers, two IOCP queues.
 * TOPOLOGY NOTE (Beast Canyon): ONE M.2 slot is CPU-attached (the 32 GB
 * Optane, F:) and the other Optane (E:) hangs off the PCH/DMI — so the heavy
 * 8 KB K-residue stream belongs on dir_k=F: and the lighter 4 KB V stream on
 * dir_v=E:. Same-dir callers get this via register2 (dir_v = dir_k). */
int sp_ring2_optane_register_split(const char *dir_k, const char *dir_v,
                                   size_t bytes_k, size_t blk_k,
                                   size_t bytes_v, size_t blk_v) {
    if (g_obe) sp_ring2_optane_unregister();
    optane_be *b = (optane_be *)calloc(1, sizeof(*b));
    if (!b) return 1;
    b->blk[0] = blk_k; b->blk[1] = blk_v;
    const size_t bytes[2] = { bytes_k, bytes_v };
    const char *dirs[2] = { dir_k, dir_v };
    char pfx[1024];
    for (int w = 0; w < 2; w++) {
        /* distinct filename prefixes — independent stores even when same dir */
        snprintf(pfx, sizeof(pfx), "%s%s", dirs[w], w == 0 ? "armk_" : "armv_");
        b->r[w] = ring2_disk_open(pfx, bytes[w], b->blk[w]);
        if (!b->r[w]) { if (w == 1) { ring2_disk_scratch_free(b->sc[0]); ring2_disk_close(b->r[0]); } free(b); return 1; }
        b->sc[w] = ring2_disk_scratch_new(b->r[w]);
        if (!b->sc[w]) {
            ring2_disk_close(b->r[w]);
            if (w == 1) { ring2_disk_scratch_free(b->sc[0]); ring2_disk_close(b->r[0]); }
            free(b); return 1;
        }
    }

    /* temporal-locality staging cache: SP_RING2_CACHE_MB total, split 2:1
     * K:V (equal slot counts — the same hot positions in both streams).
     * 0 / unset = OFF = the proven uncached path, bit-identical. */
    {
        const char *ec = getenv("SP_RING2_CACHE_MB");
        size_t cmb = ec ? (size_t)strtoull(ec, NULL, 10) : 0;
        if (cmb > 0) {
            bc_init(&b->cache[0], cmb * 1024 * 1024 * 2 / 3, blk_k);
            bc_init(&b->cache[1], cmb * 1024 * 1024 / 3, blk_v);
            fprintf(stderr, "    [ring2-cache] temporal staging cache ON: %zu MB "
                    "(K %d slots x %zu B, V %d slots x %zu B; LRU, write-through, "
                    "caller-thread-only mutations)\n",
                    cmb, b->cache[0].nslot, blk_k, b->cache[1].nslot, blk_v);
        }
    }

    sp_arm_ring2_backend be;
    be.handle        = b;
    be.write_block   = obe_write;
    be.read_block    = obe_read;
    be.read_batch    = obe_read_batch;
    be.alloc_aligned = obe_alloc;
    be.free_aligned  = obe_free;
    be.close         = NULL;             /* borrowed by the decode; WE own teardown */
    be.read_batch2   = obe_read_batch2;  /* device-overlap mixed-stream batch */
    sp_arm_ring2_register(&be);
    g_obe = b;
    fprintf(stderr, "    [ring2-optane] REGISTERED dual-size SPLIT: K(dir=%s presize=%zu MB blk=%zu B) "
            "V(dir=%s presize=%zu MB blk=%zu B) (NO_BUFFERING + IOCP, independent device queues)\n",
            dir_k, bytes_k >> 20, blk_k, dir_v, bytes_v >> 20, blk_v);
    return 0;
}

int sp_ring2_optane_register2(const char *dir,
                              size_t bytes_k, size_t blk_k,
                              size_t bytes_v, size_t blk_v) {
    return sp_ring2_optane_register_split(dir, dir, bytes_k, blk_k, bytes_v, blk_v);
}

int sp_ring2_optane_register(const char *dir, size_t bytes_per_file, size_t block_bytes) {
    return sp_ring2_optane_register2(dir, bytes_per_file, block_bytes,
                                     bytes_per_file, block_bytes);
}

int sp_ring2_optane_register_env(void) {
    const char *dir = getenv("SP_RING2_OPTANE_DIR");
    if (!dir || !dir[0]) return 1;                       /* no-op without the env */
    const char *edv = getenv("SP_RING2_OPTANE_DIR_V");   /* V-stream device split  */
    const char *dir_v = (edv && edv[0]) ? edv : dir;
    const char *eb  = getenv("SP_RING2_OPTANE_BYTES");
    const char *ebk = getenv("SP_RING2_OPTANE_BYTES_K"); /* per-stream presize —   */
    const char *ebv = getenv("SP_RING2_OPTANE_BYTES_V"); /* each inner store opens */
    const char *ek  = getenv("SP_RING2_OPTANE_BLOCK");   /* TWO files at `bytes`,  */
    const char *ekk = getenv("SP_RING2_OPTANE_BLOCK_K"); /* so symmetric sizing    */
    const char *ekv = getenv("SP_RING2_OPTANE_BLOCK_V"); /* costs 4x the K demand  */
    size_t bytes   = eb  ? (size_t)strtoull(eb, NULL, 10) : ((size_t)1 << 30);
    size_t bytes_k = ebk ? (size_t)strtoull(ebk, NULL, 10) : bytes;
    size_t bytes_v = ebv ? (size_t)strtoull(ebv, NULL, 10) : bytes;
    size_t blk   = ek ? (size_t)strtoull(ek, NULL, 10) : 4096u;
    size_t blk_k = ekk ? (size_t)strtoull(ekk, NULL, 10) : blk;
    size_t blk_v = ekv ? (size_t)strtoull(ekv, NULL, 10) : blk;
    return sp_ring2_optane_register_split(dir, dir_v, bytes_k, blk_k, bytes_v, blk_v);
}

void sp_ring2_optane_unregister(void) {
    if (!g_obe) return;
    sp_arm_ring2_register(NULL);
    obe_close(g_obe);
    g_obe = NULL;
}
