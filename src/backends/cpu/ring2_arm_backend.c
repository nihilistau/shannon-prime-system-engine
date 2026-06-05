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

typedef struct {
    ring2_disk    *r[2];     /* [0]=K stream store, [1]=V stream store */
    ring2_scratch *sc[2];    /* serial decode: one scratch per store   */
    size_t         blk[2];   /* per-stream block bytes                 */
} optane_be;

static optane_be *g_obe = NULL;   /* singleton: registered store pair */

static int obe_write(void *h, int which, uint64_t off, const void *src, size_t len) {
    optane_be *b = (optane_be *)h;
    if (!b || which < 0 || which > 1 || len != b->blk[which]) return 1;
    return ring2_disk_write(b->r[which], which, (size_t)off, src);
}

static int obe_read(void *h, int which, uint64_t off, void *dst, size_t len) {
    optane_be *b = (optane_be *)h;
    if (!b || which < 0 || which > 1 || len != b->blk[which]) return 1;
    const void *p = ring2_disk_read(b->r[which], which, (size_t)off, b->sc[which]);
    if (!p) return 1;
    memcpy(dst, p, len);
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
