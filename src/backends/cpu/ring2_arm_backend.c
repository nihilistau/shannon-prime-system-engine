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

int sp_ring2_optane_register2(const char *dir,
                              size_t bytes_k, size_t blk_k,
                              size_t bytes_v, size_t blk_v) {
    if (g_obe) sp_ring2_optane_unregister();
    optane_be *b = (optane_be *)calloc(1, sizeof(*b));
    if (!b) return 1;
    b->blk[0] = blk_k; b->blk[1] = blk_v;
    const size_t bytes[2] = { bytes_k, bytes_v };
    char pfx[1024];
    for (int w = 0; w < 2; w++) {
        /* distinct filename prefixes — two independent stores in one dir */
        snprintf(pfx, sizeof(pfx), "%s%s", dir, w == 0 ? "armk_" : "armv_");
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
    sp_arm_ring2_register(&be);
    g_obe = b;
    fprintf(stderr, "    [ring2-optane] REGISTERED dual-size: dir=%s K(presize=%zu MB blk=%zu B) "
            "V(presize=%zu MB blk=%zu B) (NO_BUFFERING + IOCP batched reads)\n",
            dir, bytes_k >> 20, blk_k, bytes_v >> 20, blk_v);
    return 0;
}

int sp_ring2_optane_register(const char *dir, size_t bytes_per_file, size_t block_bytes) {
    return sp_ring2_optane_register2(dir, bytes_per_file, block_bytes,
                                     bytes_per_file, block_bytes);
}

int sp_ring2_optane_register_env(void) {
    const char *dir = getenv("SP_RING2_OPTANE_DIR");
    if (!dir || !dir[0]) return 1;                       /* no-op without the env */
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
    return sp_ring2_optane_register2(dir, bytes_k, blk_k, bytes_v, blk_v);
}

void sp_ring2_optane_unregister(void) {
    if (!g_obe) return;
    sp_arm_ring2_register(NULL);
    obe_close(g_obe);
    g_obe = NULL;
}
