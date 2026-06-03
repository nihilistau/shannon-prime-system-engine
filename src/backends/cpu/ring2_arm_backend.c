/* ring2_arm_backend.c — Stage C: wrap the proven Optane NO_BUFFERING+IOCP store
 * (ring2_disk.c, C2.1 Step 2b: 7.57 us/read amortized) in the math-core ARM
 * Ring-2 backend interface and register it through the L1 hook. After this,
 * the CANONICAL decode (math-core qwen3_generate_kv) is what drives Optane;
 * the engine's duplicate decode is deleted (single source of truth).
 *
 * Hook mapping:
 *   write_block    -> ring2_disk_write       (aligned bounce inside the store)
 *   read_block     -> ring2_disk_read        (scratch read + copy out)
 *   read_batch     -> ring2_disk_read_batch  (IOCP queue-depth amortization;
 *                     dst buffers are sector-aligned because the decode gets
 *                     its staging from our alloc_aligned hook)
 *   alloc/free_aligned -> ring2_disk_{alloc,free}_aligned (4 KB direct-I/O)
 *   close          -> stats + scratch/store teardown (only on _unregister:
 *                     the decode treats a REGISTERED backend as borrowed)
 */
#include "sp_engine/ring2_arm.h"
#include "sp_engine/ring2_disk.h"
#include "sp/arm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ring2_disk    *r;
    ring2_scratch *sc;       /* serial decode: one scratch */
    size_t         blk;
} optane_be;

static optane_be *g_obe = NULL;   /* singleton: registered store */

static int obe_write(void *h, int which, uint64_t off, const void *src, size_t len) {
    optane_be *b = (optane_be *)h;
    if (!b || len != b->blk) return 1;
    return ring2_disk_write(b->r, which, (size_t)off, src);
}

static int obe_read(void *h, int which, uint64_t off, void *dst, size_t len) {
    optane_be *b = (optane_be *)h;
    if (!b || len != b->blk) return 1;
    const void *p = ring2_disk_read(b->r, which, (size_t)off, b->sc);
    if (!p) return 1;
    memcpy(dst, p, len);
    return 0;
}

static int obe_read_batch(void *h, const int *which, const uint64_t *off,
                          void *const *dst, size_t len, int n) {
    optane_be *b = (optane_be *)h;
    if (!b || len != b->blk) return 1;
    ring2_req *reqs = (ring2_req *)malloc((size_t)n * sizeof(ring2_req));
    if (!reqs) return 1;
    for (int i = 0; i < n; i++) {
        reqs[i].which = which[i];
        reqs[i].off   = (size_t)off[i];
        reqs[i].dst   = dst[i];          /* sector-aligned via our alloc hook */
    }
    int rc = ring2_disk_read_batch(b->r, reqs, n);
    free(reqs);
    return rc;
}

static void *obe_alloc(void *h, size_t bytes) { (void)h; return ring2_disk_alloc_aligned(bytes); }
static void  obe_free(void *h, void *p)        { (void)h; ring2_disk_free_aligned(p); }

static void obe_close(void *h) {       /* called only from _unregister */
    optane_be *b = (optane_be *)h;
    if (!b) return;
    unsigned long long nr = 0; double rs = 0;
    ring2_disk_stats(b->r, &nr, &rs);
    fprintf(stderr, "    [ring2-optane] %llu reads, %.3f s total, %.2f us/read avg\n",
            nr, rs, nr ? rs * 1e6 / (double)nr : 0.0);
    ring2_disk_scratch_free(b->sc);
    ring2_disk_close(b->r);
    free(b);
}

int sp_ring2_optane_register(const char *dir, size_t bytes_per_file, size_t block_bytes) {
    if (g_obe) sp_ring2_optane_unregister();
    optane_be *b = (optane_be *)calloc(1, sizeof(*b));
    if (!b) return 1;
    b->blk = block_bytes;
    b->r = ring2_disk_open(dir, bytes_per_file, block_bytes);
    if (!b->r) { free(b); return 1; }
    b->sc = ring2_disk_scratch_new(b->r);
    if (!b->sc) { ring2_disk_close(b->r); free(b); return 1; }

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
    fprintf(stderr, "    [ring2-optane] REGISTERED as the ARM Ring-2 backend: dir=%s "
            "presize=%zu MB block=%zu B (NO_BUFFERING + IOCP batched reads)\n",
            dir, bytes_per_file >> 20, block_bytes);
    return 0;
}

int sp_ring2_optane_register_env(void) {
    const char *dir = getenv("SP_RING2_OPTANE_DIR");
    if (!dir || !dir[0]) return 1;                       /* no-op without the env */
    const char *eb = getenv("SP_RING2_OPTANE_BYTES");
    const char *ek = getenv("SP_RING2_OPTANE_BLOCK");
    size_t bytes = eb ? (size_t)strtoull(eb, NULL, 10) : ((size_t)1 << 30);
    size_t blk   = ek ? (size_t)strtoull(ek, NULL, 10) : 4096u;
    return sp_ring2_optane_register(dir, bytes, blk);
}

void sp_ring2_optane_unregister(void) {
    if (!g_obe) return;
    sp_arm_ring2_register(NULL);
    obe_close(g_obe);
    g_obe = NULL;
}
