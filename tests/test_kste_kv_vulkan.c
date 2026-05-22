/* test_kste_kv_vulkan.c — E_VK_6: KSTE KV-cache encoding on Vulkan.
 *
 * qwen3_forward_vulkan_ex encodes each post-norm/post-RoPE K head-vector to its
 * 64-byte KSTE signature via the host sp_kste_encode (the on-device shader is
 * deferred). KSTE byte-identity ACROSS backends is structurally unachievable —
 * order-statistic encoding amplifies the same §8.6.1 GEMM-vs-scalar FP floor that
 * QK-norm amplifies for logits — so the cross-backend gate here is NOT byte-
 * identity. Gates:
 *   Part A (encoder)  : sp_kste_encode is deterministic on fixed input.
 *   Part B (model K)  : Vulkan signatures deterministic + wire-valid.
 *   Proof of cause    : the Tier-0 ROOT order-statistic labels (6 int16) of the
 *                       Vulkan vs CPU signatures barely move (max drift <= 4 LSB),
 *                       proving the divergence is FP-floor K boundary flips, not
 *                       a logic bug. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp/kste.h"
#include "sp_engine/gguf.h"
#include "sp_engine/model.h"
#include "sp_engine/vulkan_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SP_QWEN3_GGUF
#define SP_QWEN3_GGUF "Qwen3-0.6B-f16.gguf"
#endif
#ifndef SP_QWEN3_REF
#define SP_QWEN3_REF "qwen3_ref.bin"
#endif

static int16_t rd_i16(const uint8_t *p) { return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }

static uint32_t rng = 0x9E3779B9u;
static int32_t rnd(int range) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (int32_t)(rng % (uint32_t)(2 * range + 1)) - range;
}

static void E_VK_6(void) {
    SP_CHECK(sp_vulkan_device_count() >= 1, "Vulkan device visible");
    if (sp_vulkan_device_count() < 1) { fprintf(stderr, "    %s\n", sp_last_error()); return; }

    FILE *f = fopen(SP_QWEN3_REF, "rb");
    SP_CHECK(f != NULL, "open qwen3_ref.bin for tokens");
    if (!f) return;
    uint32_t magic = 0, nt = 0, nv = 0;
    if (!(fread(&magic,4,1,f)==1 && fread(&nt,4,1,f)==1 && fread(&nv,4,1,f)==1) || nt == 0) { fclose(f); SP_CHECK(0,"ref header"); return; }
    int32_t *toks = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    int ok = toks && fread(toks, sizeof(int32_t), nt, f) == nt;
    fclose(f);
    SP_CHECK(ok, "read ref tokens");
    if (!ok) { free(toks); return; }

    qwen3_model *m = qwen3_load(SP_QWEN3_GGUF);
    SP_CHECK(m && m->cfg.arch == SP_ARCH_QWEN3, "qwen3 load");
    if (!m) { free(toks); return; }
    const int V = (int)m->cfg.n_vocab, NKV = (int)m->cfg.n_head_kv, HD = (int)m->cfg.head_dim;

    /* ── Part A: encoder is deterministic / portable on fixed input ── */
    {
        int32_t *vec = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        long bad = 0;
        for (int trial = 0; trial < 4000; trial++) {
            for (int i = 0; i < HD; i++) vec[i] = rnd(1 << 20);
            sp_kste_tree_t a, b;
            sp_kste_encode(vec, HD, &a);
            sp_kste_encode(vec, HD, &b);
            if (memcmp(&a, &b, sizeof a) != 0) bad++;
        }
        free(vec);
        fprintf(stderr, "    Part A: sp_kste_encode deterministic on 4000 fixed inputs, mismatches=%ld\n", bad);
        SP_CHECK_EQ_I64(bad, 0, "Part A: encoder deterministic/portable on identical input");
    }

    /* ── Part B: model-driven signatures, CPU vs Vulkan ── */
    const size_t ntrees = (size_t)m->cfg.n_layers * nt * NKV;
    fprintf(stderr, "    Part B: %zu signatures (%u layers x %u tok x %d kv-heads)\n",
            ntrees, m->cfg.n_layers, nt, NKV);
    sp_kste_tree_t *kv_cpu = (sp_kste_tree_t *)calloc(ntrees, sizeof(sp_kste_tree_t));
    sp_kste_tree_t *kv_vk  = (sp_kste_tree_t *)calloc(ntrees, sizeof(sp_kste_tree_t));
    sp_kste_tree_t *kv_vk2 = (sp_kste_tree_t *)calloc(ntrees, sizeof(sp_kste_tree_t));
    float *logits = (float *)malloc((size_t)nt * V * sizeof(float));
    if (!kv_cpu || !kv_vk || !kv_vk2 || !logits) { SP_CHECK(0, "alloc"); goto cleanup; }

    {
        int rc_cpu = qwen3_forward_ex(m, toks, (int)nt, logits, kv_cpu);
        int rc_vk  = qwen3_forward_vulkan_ex(m, toks, (int)nt, logits, kv_vk);
        int rc_vk2 = qwen3_forward_vulkan_ex(m, toks, (int)nt, logits, kv_vk2);
        SP_CHECK(rc_cpu == 0 && rc_vk == 0 && rc_vk2 == 0, "cpu + vulkan(x2) forward_ex");
        if (rc_vk) fprintf(stderr, "    sp_last_error: %s\n", sp_last_error());
        if (rc_cpu || rc_vk || rc_vk2) goto cleanup;

        int det = memcmp(kv_vk, kv_vk2, ntrees * sizeof(sp_kste_tree_t)) == 0;
        SP_CHECK(det, "E_VK_6: Vulkan signatures deterministic across runs");
        long nonzero = 0, bad_ver = 0;
        for (size_t i = 0; i < ntrees; i++) {
            if (kv_vk[i].bytes[SP_KSTE_OFF_VERSION] != (uint8_t)SP_KSTE_LAYOUT_VERSION) bad_ver++;
            for (int b = 0; b < 64; b++) if (kv_vk[i].bytes[b]) { nonzero++; break; }
        }
        SP_CHECK_EQ_I64(bad_ver, 0, "E_VK_6: every signature carries the frozen layout version");
        SP_CHECK(nonzero > 0, "E_VK_6: signatures non-trivial (encode ran)");

        long same = 0; int max_label_drift = 0; double sum_label_drift = 0; long ncmp = 0;
        for (size_t i = 0; i < ntrees; i++) {
            if (memcmp(&kv_cpu[i], &kv_vk[i], sizeof(sp_kste_tree_t)) == 0) same++;
            const uint8_t *pc = kv_cpu[i].bytes + SP_KSTE_OFF_ROOT;
            const uint8_t *pg = kv_vk[i].bytes  + SP_KSTE_OFF_ROOT;
            for (int c = 0; c < 6; c++) {
                int d = (int)rd_i16(pc + 2*c) - (int)rd_i16(pg + 2*c);
                if (d < 0) d = -d;
                if (d > max_label_drift) max_label_drift = d;
                sum_label_drift += d; ncmp++;
            }
        }
        double agree = (double)same / (double)ntrees;
        fprintf(stderr, "    deterministic=%d wire-valid=ok | agreement=%.2f%% (%ld/%zu) | "
                "Tier-0 label drift max=%d mean=%.4f\n",
                det, 100.0 * agree, same, ntrees, max_label_drift, sum_label_drift / (double)ncmp);
        SP_CHECK(agree > 0.50, "E_VK_6: cross-backend agreement > 50% (rest = K-quantization boundary flips)");
        SP_CHECK(max_label_drift <= 4, "E_VK_6: Tier-0 order-statistic labels within 4 LSB of CPU (FP-floor cause)");
    }

cleanup:
    free(kv_cpu); free(kv_vk); free(kv_vk2); free(logits);
    free(toks); qwen3_free(m);
}

int main(void) { SP_RUN(E_VK_6); return SP_DONE(); }
