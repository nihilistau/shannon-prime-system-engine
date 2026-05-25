/* vulkan_forward.cpp — Gemma3 + Qwen3 forward pass on Vulkan compute (Phase 2-VK).
 *   VK.1 f32 + VK.2/VK.3 Q8/Q4 arena (gemma3); VK.4 qwen3_forward_vulkan
 *   (M_QWEN3); E_VK_5 NTT-attention; E_VK_6 KSTE-KV.
 *
 * Mirrors the CUDA forward (src/backends/cuda/cuda_forward.cu) op-for-op, which
 * itself mirrors the CPU forwards (gemma3_forward / qwen3_forward). The cuBLAS
 * SGEMM is replaced by a hand-written tiled f32 GEMM compute shader (gemm.comp);
 * the discrete-algebra ops (rmsnorm / per-head QK-norm / NEOX RoPE / GQA windowed
 * softmax / GeGLU / SwiGLU / embed-scale / residual add / arena decode) are the
 * same kernels as CUDA, transcribed to GLSL -> SPIR-V (compiled at build time).
 *
 * Determinism (T_FRO_4 gate-(a) mode): a single compute queue, sequential dispatch
 * recorded into one command buffer with a full SHADER_WRITE->SHADER_READ memory
 * barrier between every op (no overlap), no atomics, true f32 (no relaxed
 * precision). Vulkan f32 ~= CPU f32 to ~1e-5, the same floor cuBLAS hit, so PPL
 * drift sits well inside 0.05%.
 *
 * Weight residency: f32 weights are dequantized host-side and uploaded once;
 * packed-arena weights (SP_ARENA=q8|q4) upload the compact math-core layout and
 * are decoded ON DEVICE by dequant_arena.comp into a reused f32 scratch right
 * before their GEMM (the §4.8 decode-on-demand path). Cached by model pointer;
 * sp_vulkan_model_release frees on qwen3_free.
 */
#include "sp_engine/vulkan_backend.h"
#include "sp_engine/kernels.h"     /* as_f32 */
#include "sp_engine/arena.h"       /* sp_arena_find / sp_arena_dequant_row */
#include "sp_engine/gguf.h"
#include "sp/frobenius_lift.h"     /* sp_frob_packed_tensor */
#include "sp/kste.h"               /* sp_kste_encode (E_VK_6 host encode) */
#include "vk_common.h"

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

/* ── embedded SPIR-V (glslc -mfmt=c at build time -> {0x..,..}) ── */
static const uint32_t spv_gemm[] =
#include "gemm.spv.h"
;
static const uint32_t spv_embed_scale[] =
#include "embed_scale.spv.h"
;
static const uint32_t spv_rmsnorm[] =
#include "rmsnorm.spv.h"
;
static const uint32_t spv_rmsnorm_head[] =
#include "rmsnorm_head.spv.h"
;
static const uint32_t spv_rope[] =
#include "rope.spv.h"
;
static const uint32_t spv_attn[] =
#include "attn.spv.h"
;
static const uint32_t spv_attn_ntt[] =
#include "attn_ntt.spv.h"
;
static const uint32_t spv_gelu_mul[] =
#include "gelu_mul.spv.h"
;
static const uint32_t spv_silu_mul[] =
#include "silu_mul.spv.h"
;
static const uint32_t spv_add[] =
#include "add.spv.h"
;
static const uint32_t spv_dequant_arena[] =
#include "dequant_arena.spv.h"
;
static const uint32_t spv_round_f16[] =
#include "round_f16.spv.h"
;

/* The attention shaders hold the per-query scores in a fixed shared array
 * sc[MAXTOK]; n_tok must not exceed it (the engine gates run n_ctx <= 168, but a
 * later phase could pass more). Keep in lockstep with MAXTOK in attn*.comp. */
#define VK_MAXTOK 1024

/* on-disk bytes of `n` contiguous elements of a ggml weight row (matches CPU). */
static size_t row_bytes(uint32_t type, int n) {
    switch (type) {
        case GGML_T_F32:  return (size_t)n * 4;
        case GGML_T_F16:  return (size_t)n * 2;
        case GGML_T_Q8_0: return (size_t)(n / 32) * 34;
        default:          return 0;
    }
}

/* ════════════════════════ device buffer ════════════════════════ */

struct DevBuf {
    VkBuffer       buf;
    VkDeviceMemory mem;
    VkDeviceSize   size;
};

static int buf_create(DevBuf *b, VkDeviceSize size, VkMemoryPropertyFlags props) {
    VkContext *c = vk_ctx();
    *b = DevBuf{};
    if (size == 0) size = 4;   /* never create a zero-size buffer */
    VkBufferCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKC(vkCreateBuffer(c->device, &ci, nullptr, &b->buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, b->buf, &req);
    uint32_t mt = vk_find_mem_type(req.memoryTypeBits, props);
    if (mt == UINT32_MAX) { sp_set_error("buf_create: no suitable memory type"); return 1; }
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    VKC(vkAllocateMemory(c->device, &ai, nullptr, &b->mem), "vkAllocateMemory");
    VKC(vkBindBufferMemory(c->device, b->buf, b->mem, 0), "vkBindBufferMemory");
    b->size = size;
    return 0;
}

static void buf_free(DevBuf *b) {
    VkContext *c = vk_ctx();
    if (b->buf) vkDestroyBuffer(c->device, b->buf, nullptr);
    if (b->mem) vkFreeMemory(c->device, b->mem, nullptr);
    *b = DevBuf{};
}

/* Device-local scratch/IO buffer (HOST_VISIBLE|HOST_COHERENT so we can map for
 * upload/download without staging — simplest correct path; perf is not gated). */
static int buf_create_hostvis(DevBuf *b, VkDeviceSize size) {
    return buf_create(b, size, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

static int buf_upload(DevBuf *b, const void *src, size_t bytes) {
    VkContext *c = vk_ctx();
    void *p = nullptr;
    VKC(vkMapMemory(c->device, b->mem, 0, bytes, 0, &p), "vkMapMemory(upload)");
    std::memcpy(p, src, bytes);
    vkUnmapMemory(c->device, b->mem);
    return 0;
}

static int buf_download(DevBuf *b, void *dst, size_t bytes) {
    VkContext *c = vk_ctx();
    void *p = nullptr;
    VKC(vkMapMemory(c->device, b->mem, 0, bytes, 0, &p), "vkMapMemory(download)");
    std::memcpy(dst, p, bytes);
    vkUnmapMemory(c->device, b->mem);
    return 0;
}

/* upload host data into a fresh host-visible device buffer. */
static int buf_make(DevBuf *b, const void *src, size_t bytes) {
    if (buf_create_hostvis(b, bytes)) return 1;
    return buf_upload(b, src, bytes);
}

/* ════════════════════════ pipelines ════════════════════════ */

/* A compute pipeline + its descriptor-set layout. All our shaders use N storage
 * buffers at bindings 0..N-1 plus a push-constant block. */
struct Pipe {
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      layout;
    VkPipeline            pipeline;
    int                   n_bind;
};

enum {
    P_GEMM, P_EMBED, P_RMSNORM, P_RMSNORM_HEAD, P_ROPE, P_ATTN, P_ATTN_NTT,
    P_GELU, P_SILU, P_ADD, P_DEQUANT, P_ROUND_F16, P_COUNT
};

struct PipeSpec { const uint32_t *spv; size_t bytes; int n_bind; uint32_t pc_bytes; };

static int make_pipe(Pipe *p, const PipeSpec &s) {
    VkContext *c = vk_ctx();
    *p = Pipe{};
    p->n_bind = s.n_bind;

    std::vector<VkDescriptorSetLayoutBinding> binds(s.n_bind);
    for (int i = 0; i < s.n_bind; i++) {
        binds[i] = {};
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci = {};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = (uint32_t)s.n_bind;
    dlci.pBindings = binds.data();
    VKC(vkCreateDescriptorSetLayout(c->device, &dlci, nullptr, &p->dsl), "vkCreateDescriptorSetLayout");

    VkPushConstantRange pcr = {};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = s.pc_bytes;
    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &p->dsl;
    plci.pushConstantRangeCount = s.pc_bytes ? 1 : 0;
    plci.pPushConstantRanges = s.pc_bytes ? &pcr : nullptr;
    VKC(vkCreatePipelineLayout(c->device, &plci, nullptr, &p->layout), "vkCreatePipelineLayout");

    VkShaderModuleCreateInfo smci = {};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = s.bytes;
    smci.pCode = s.spv;
    VkShaderModule sm;
    VKC(vkCreateShaderModule(c->device, &smci, nullptr, &sm), "vkCreateShaderModule");

    VkComputePipelineCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = p->layout;
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p->pipeline);
    vkDestroyShaderModule(c->device, sm, nullptr);
    if (r != VK_SUCCESS) return vk_fail(r, "vkCreateComputePipelines");
    return 0;
}

static void free_pipe(Pipe *p) {
    VkContext *c = vk_ctx();
    if (p->pipeline) vkDestroyPipeline(c->device, p->pipeline, nullptr);
    if (p->layout)   vkDestroyPipelineLayout(c->device, p->layout, nullptr);
    if (p->dsl)      vkDestroyDescriptorSetLayout(c->device, p->dsl, nullptr);
    *p = Pipe{};
}

/* ════════════════════════ device weight cache ════════════════════════ */

struct DevTensor {
    int in, out;
    DevBuf f32;                  /* size>0 => plain f32 [out*in] */
    int    is_f32;
    DevBuf codes;                /* packed (is_f32==0): math-core arena layout, padded */
    DevBuf row_off;              /* [out] uvec2 (uint64 lo/hi) */
    DevBuf row_scale;            /* [out] float */
    DevBuf row_prec;             /* [out] uint32 (8 or 4) */
};

struct VulkanWeights {
    const qwen3_model *key;
    int L;
    Pipe pipes[P_COUNT];
    VkCommandPool cmdpool;
    DevBuf embd;        /* [V*E] token-embd f32 */
    DevBuf out_norm;    /* [E] */
    DevTensor head;     /* untied LM head (qwen3); is_f32=0/codes or empty when tied */
    int    has_head;
    size_t scratch_n;   /* max packed weight elem count (0 if no arena) */
    DevTensor *Wq, *Wk, *Wv, *Wo, *Wgate, *Wup, *Wdown;
    DevBuf *attn_norm, *ffn_norm, *q_norm, *k_norm, *post_attn, *post_ffw;
};
static VulkanWeights g_w = {};

/* ── push-constant structs (must match each shader's PC block) ── */
struct PC_gemm   { int n_tok, in_dim, out_dim; };
struct PC_embed  { int n_tok, E; float scale; };
struct PC_rms    { int n; float eps; };
struct PC_rmsh   { int n_heads, d, rowstride; float eps; };
struct PC_rope   { int n_heads, d, rowstride; };
struct PC_attn   { int n_tok, QD, KVD, HD, group; float ascale; int win; };
struct PC_attnn  { int n_tok, QD, KVD, HD, group; float ascale, qscale; };
struct PC_elem   { uint32_t n; };
struct PC_deq    { int rows, cols; };

static int build_pipes(VulkanWeights *w) {
    PipeSpec specs[P_COUNT];
    specs[P_GEMM]         = { spv_gemm,         sizeof(spv_gemm),         3, sizeof(PC_gemm) };
    specs[P_EMBED]        = { spv_embed_scale,  sizeof(spv_embed_scale),  3, sizeof(PC_embed) };
    specs[P_RMSNORM]      = { spv_rmsnorm,      sizeof(spv_rmsnorm),      3, sizeof(PC_rms) };
    specs[P_RMSNORM_HEAD] = { spv_rmsnorm_head, sizeof(spv_rmsnorm_head), 2, sizeof(PC_rmsh) };
    specs[P_ROPE]         = { spv_rope,         sizeof(spv_rope),         2, sizeof(PC_rope) };
    specs[P_ATTN]         = { spv_attn,         sizeof(spv_attn),         4, sizeof(PC_attn) };
    specs[P_ATTN_NTT]     = { spv_attn_ntt,     sizeof(spv_attn_ntt),     4, sizeof(PC_attnn) };
    specs[P_GELU]         = { spv_gelu_mul,     sizeof(spv_gelu_mul),     2, sizeof(PC_elem) };
    specs[P_SILU]         = { spv_silu_mul,     sizeof(spv_silu_mul),     2, sizeof(PC_elem) };
    specs[P_ADD]          = { spv_add,          sizeof(spv_add),          2, sizeof(PC_elem) };
    specs[P_DEQUANT]      = { spv_dequant_arena,sizeof(spv_dequant_arena),5, sizeof(PC_deq) };
    specs[P_ROUND_F16]   = { spv_round_f16,   sizeof(spv_round_f16),   1, sizeof(PC_elem) };
    for (int i = 0; i < P_COUNT; i++)
        if (make_pipe(&w->pipes[i], specs[i])) return 1;
    return 0;
}

/* dequant a GGUF weight tensor [out x in] to a host f32 [out*in], upload. */
static int upload_weight(const qwen3_model *m, const gguf_tensor *t, int in, int out, DevBuf *out_buf) {
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(m->gguf, t);
    size_t rb = row_bytes(t->type, in);
    if (!base || rb == 0) { sp_set_error("upload_weight: null/unsupported tensor"); return 1; }
    size_t n = (size_t)out * in;
    float *host = (float *)malloc(n * sizeof(float));
    if (!host) { sp_set_error("upload_weight: host OOM"); return 1; }
    for (int j = 0; j < out; j++)
        if (sp_dequant_row(base + (size_t)j * rb, t->type, in, host + (size_t)j * in)) {
            free(host); sp_set_error("upload_weight: dequant failed"); return 1;
        }
    int rc = buf_make(out_buf, host, n * sizeof(float));
    free(host);
    return rc;
}

static int upload_vec(const qwen3_model *m, const gguf_tensor *t, int n, DevBuf *out_buf) {
    const float *host = as_f32(m, t);
    if (!host) { sp_set_error("upload_vec: null tensor"); return 1; }
    return buf_make(out_buf, host, (size_t)n * sizeof(float));
}

/* upload a packed arena tensor (compact math-core layout) into a DevTensor. The
 * codes byte stream is padded to a 4-byte multiple (std430 uint[]); row_prec is
 * widened uint8->uint32 (one word per row); row_off (size_t/8B host) becomes
 * uvec2 (lo/hi uint32). */
static int upload_packed(const sp_frob_packed_tensor *pt, DevTensor *d) {
    *d = DevTensor{};
    d->in = pt->cols; d->out = pt->rows; d->is_f32 = 0;

    /* codes: pad to multiple of 4 bytes */
    size_t cb = pt->codes_bytes;
    size_t cb4 = (cb + 3) & ~(size_t)3;
    uint8_t *codes_pad = (uint8_t *)calloc(cb4 ? cb4 : 4, 1);
    if (!codes_pad) { sp_set_error("upload_packed: codes OOM"); return 1; }
    std::memcpy(codes_pad, pt->codes, cb);
    int rc = buf_make(&d->codes, codes_pad, cb4 ? cb4 : 4);
    free(codes_pad);
    if (rc) return 1;

    /* row_off: size_t -> uvec2 (lo, hi) */
    std::vector<uint32_t> roff((size_t)pt->rows * 2);
    for (int j = 0; j < pt->rows; j++) {
        uint64_t v = (uint64_t)pt->row_off[j];
        roff[(size_t)j * 2 + 0] = (uint32_t)(v & 0xFFFFFFFFu);
        roff[(size_t)j * 2 + 1] = (uint32_t)(v >> 32);
    }
    if (buf_make(&d->row_off, roff.data(), roff.size() * sizeof(uint32_t))) return 1;

    if (buf_make(&d->row_scale, pt->row_scale, (size_t)pt->rows * sizeof(float))) return 1;

    /* row_prec: uint8 -> uint32 */
    std::vector<uint32_t> rprec((size_t)pt->rows);
    for (int j = 0; j < pt->rows; j++) rprec[j] = (uint32_t)pt->row_prec[j];
    if (buf_make(&d->row_prec, rprec.data(), rprec.size() * sizeof(uint32_t))) return 1;
    return 0;
}

static void free_devtensor(DevTensor *d) {
    buf_free(&d->f32); buf_free(&d->codes); buf_free(&d->row_off);
    buf_free(&d->row_scale); buf_free(&d->row_prec);
    *d = DevTensor{};
}

static void free_weights(VulkanWeights *w) {
    VkContext *c = vk_ctx();
    buf_free(&w->embd); buf_free(&w->out_norm);
    free_devtensor(&w->head);
    DevTensor **dts[] = { &w->Wq,&w->Wk,&w->Wv,&w->Wo,&w->Wgate,&w->Wup,&w->Wdown };
    for (size_t a = 0; a < sizeof(dts)/sizeof(dts[0]); a++) {
        DevTensor *arr = *dts[a];
        if (arr) { for (int L = 0; L < w->L; L++) free_devtensor(&arr[L]); free(arr); }
    }
    DevBuf **ns[] = { &w->attn_norm,&w->ffn_norm,&w->q_norm,&w->k_norm,&w->post_attn,&w->post_ffw };
    for (size_t a = 0; a < sizeof(ns)/sizeof(ns[0]); a++) {
        DevBuf *arr = *ns[a];
        if (arr) { for (int L = 0; L < w->L; L++) buf_free(&arr[L]); free(arr); }
    }
    for (int i = 0; i < P_COUNT; i++) free_pipe(&w->pipes[i]);
    if (w->cmdpool && c->device) vkDestroyCommandPool(c->device, w->cmdpool, nullptr);
    *w = VulkanWeights{};
}

/* build one matmul weight: packed if it's in the arena, else f32 from GGUF. */
static int build_w(const qwen3_model *m, const gguf_tensor *W, int in, int out,
                   DevTensor *d, size_t *scratch_n) {
    const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, W->name) : NULL;
    if (at) {
        if (upload_packed(&at->pt, d)) return 1;
        size_t need = (size_t)d->out * d->in;
        if (need > *scratch_n) *scratch_n = need;
        return 0;
    }
    *d = DevTensor{};
    d->is_f32 = 1; d->in = in; d->out = out;
    if (upload_weight(m, W, in, out, &d->f32)) return 1;
    return 0;
}

#define ALLOC_DT(field) do { w->field = (DevTensor *)calloc((size_t)L, sizeof(DevTensor)); \
    if (!w->field) { sp_set_error("DevTensor array OOM"); free_weights(w); return 1; } } while (0)
#define ALLOC_NM(field) do { w->field = (DevBuf *)calloc((size_t)L, sizeof(DevBuf)); \
    if (!w->field) { sp_set_error("norm array OOM"); free_weights(w); return 1; } } while (0)
#define BUILDW(field, tensor, in, out) do { \
    if (build_w(m, ly->tensor, (in), (out), &w->field[Li], &w->scratch_n)) { free_weights(w); return 1; } } while (0)
#define UPV(field, tensor, n) do { if (upload_vec(m, ly->tensor, (n), &w->field[Li])) { free_weights(w); return 1; } } while (0)

static int build_weights(const qwen3_model *m, VulkanWeights *w) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv, V = (int)c->n_vocab;
    const int QD = NH * HD, KVD = NKV * HD, L = (int)c->n_layers;
    (void)FF; (void)QD; (void)KVD;

    const int is_gemma = (c->arch == SP_ARCH_GEMMA3);
    const int tied = (m->output == m->token_embd);

    *w = VulkanWeights{};
    w->key = m; w->L = L;

    if (vk_ensure_device()) return 1;
    if (build_pipes(w)) { free_weights(w); return 1; }

    VkContext *vc = vk_ctx();
    VkCommandPoolCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = vc->queue_family;
    if (vkCreateCommandPool(vc->device, &pci, nullptr, &w->cmdpool) != VK_SUCCESS) {
        sp_set_error("vkCreateCommandPool"); free_weights(w); return 1;
    }

    /* token embedding (always f32; gemma3's tied head reuses it). */
    {
        const sp_arena_tensor *eat = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
        if (eat) {
            size_t n = (size_t)V * E;
            float *host = (float *)malloc(n * sizeof(float));
            if (!host) { sp_set_error("embd host OOM"); free_weights(w); return 1; }
            for (int r = 0; r < V; r++)
                if (sp_arena_dequant_row(eat, r, host + (size_t)r * E)) {
                    free(host); sp_set_error("embd arena dequant"); free_weights(w); return 1;
                }
            int rc = buf_make(&w->embd, host, n * sizeof(float));
            free(host);
            if (rc) { free_weights(w); return 1; }
        } else {
            if (upload_weight(m, m->token_embd, E, V, &w->embd)) { free_weights(w); return 1; }
        }
    }
    if (upload_vec(m, m->output_norm, E, &w->out_norm)) { free_weights(w); return 1; }

    if (!tied) {
        if (build_w(m, m->output, E, V, &w->head, &w->scratch_n)) { free_weights(w); return 1; }
        w->has_head = 1;
    }

    ALLOC_DT(Wq); ALLOC_DT(Wk); ALLOC_DT(Wv); ALLOC_DT(Wo);
    ALLOC_DT(Wgate); ALLOC_DT(Wup); ALLOC_DT(Wdown);
    ALLOC_NM(attn_norm); ALLOC_NM(ffn_norm); ALLOC_NM(q_norm); ALLOC_NM(k_norm);
    if (is_gemma) { ALLOC_NM(post_attn); ALLOC_NM(post_ffw); }

    for (int Li = 0; Li < L; Li++) {
        const qwen3_layer *ly = &m->layers[Li];
        BUILDW(Wq, attn_q, E, QD);   BUILDW(Wk, attn_k, E, KVD);  BUILDW(Wv, attn_v, E, KVD);
        BUILDW(Wo, attn_output, QD, E);
        BUILDW(Wgate, ffn_gate, E, FF); BUILDW(Wup, ffn_up, E, FF); BUILDW(Wdown, ffn_down, FF, E);
        UPV(attn_norm, attn_norm, E);   UPV(ffn_norm, ffn_norm, E);
        UPV(q_norm, attn_q_norm, HD);   UPV(k_norm, attn_k_norm, HD);
        if (is_gemma) { UPV(post_attn, post_attn_norm, E); UPV(post_ffw, post_ffw_norm, E); }
    }
    return 0;
}

/* ════════════════════════ dispatch recording ════════════════════════ */

/* A per-forward command-buffer recorder: allocates a descriptor pool sized for
 * the total number of dispatches, records each op (descriptor set + push const +
 * dispatch) preceded by a full SHADER_WRITE->SHADER_READ barrier. */
struct Recorder {
    VkCommandBuffer  cmd;
    VkDescriptorPool dpool;
    int              ok;
};

static VkCommandBuffer g_cmd;          /* active command buffer (set by run_begin) */
static VkDescriptorPool g_dpool;

static int rec_begin(VulkanWeights *w, int max_dispatch) {
    VkContext *c = vk_ctx();
    VkCommandBufferAllocateInfo cai = {};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = w->cmdpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VKC(vkAllocateCommandBuffers(c->device, &cai, &g_cmd), "vkAllocateCommandBuffers");

    /* descriptor pool: 5 storage buffers max per set * max_dispatch sets. */
    VkDescriptorPoolSize ps = {};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = (uint32_t)(5 * max_dispatch);
    VkDescriptorPoolCreateInfo dpi = {};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = (uint32_t)max_dispatch;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VKC(vkCreateDescriptorPool(c->device, &dpi, nullptr, &g_dpool), "vkCreateDescriptorPool");

    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(g_cmd, &bi), "vkBeginCommandBuffer");
    return 0;
}

/* full memory barrier between dispatches: previous shader writes visible to next
 * shader reads (and writes), no overlap -> deterministic, serial. */
static void rec_barrier() {
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

/* record one dispatch: bind pipeline, write+bind a descriptor set over `bufs`,
 * push `pc`, dispatch (gx,gy,gz). A barrier is inserted BEFORE each op. */
static int rec_dispatch(const Pipe *p, const VkBuffer *bufs, int n_bufs,
                        const void *pc, uint32_t pc_bytes,
                        uint32_t gx, uint32_t gy, uint32_t gz) {
    VkContext *c = vk_ctx();
    if (n_bufs != p->n_bind) { sp_set_error("rec_dispatch: binding count mismatch"); return 1; }

    VkDescriptorSetAllocateInfo dsi = {};
    dsi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsi.descriptorPool = g_dpool;
    dsi.descriptorSetCount = 1;
    dsi.pSetLayouts = &p->dsl;
    VkDescriptorSet ds;
    VKC(vkAllocateDescriptorSets(c->device, &dsi, &ds), "vkAllocateDescriptorSets");

    std::vector<VkDescriptorBufferInfo> dbis(n_bufs);
    std::vector<VkWriteDescriptorSet>   wds(n_bufs);
    for (int i = 0; i < n_bufs; i++) {
        dbis[i] = {}; dbis[i].buffer = bufs[i]; dbis[i].offset = 0; dbis[i].range = VK_WHOLE_SIZE;
        wds[i] = {};
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = ds;
        wds[i].dstBinding = (uint32_t)i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(c->device, (uint32_t)n_bufs, wds.data(), 0, nullptr);

    rec_barrier();
    vkCmdBindPipeline(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    vkCmdBindDescriptorSets(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &ds, 0, nullptr);
    if (pc_bytes) vkCmdPushConstants(g_cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc);
    vkCmdDispatch(g_cmd, gx, gy, gz);
    return 0;
}

/* submit the recorded command buffer and wait. */
static int rec_submit_wait(VulkanWeights *w) {
    VkContext *c = vk_ctx();
    VKC(vkEndCommandBuffer(g_cmd), "vkEndCommandBuffer");
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_cmd;
    VkFenceCreateInfo fci = {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VKC(vkCreateFence(c->device, &fci, nullptr, &fence), "vkCreateFence");
    VkResult r = vkQueueSubmit(c->queue, 1, &si, fence);
    if (r != VK_SUCCESS) { vkDestroyFence(c->device, fence, nullptr); return vk_fail(r, "vkQueueSubmit"); }
    r = vkWaitForFences(c->device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(c->device, fence, nullptr);
    if (r != VK_SUCCESS) return vk_fail(r, "vkWaitForFences");
    return 0;
}

static void rec_end(VulkanWeights *w) {
    VkContext *c = vk_ctx();
    if (g_dpool) vkDestroyDescriptorPool(c->device, g_dpool, nullptr);
    if (g_cmd) vkFreeCommandBuffers(c->device, w->cmdpool, 1, &g_cmd);
    g_dpool = VK_NULL_HANDLE; g_cmd = VK_NULL_HANDLE;
}

/* ── GEMM helpers ── */
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))

static int rec_round_f16(VulkanWeights *w, VkBuffer buf, uint32_t n) {
    PC_elem pc = { n };
    VkBuffer b[1] = { buf };
    return rec_dispatch(&w->pipes[P_ROUND_F16], b, 1, &pc, sizeof(pc),
                        (uint32_t)CEIL_DIV(n, 256), 1, 1);
}

/* record Y[n_tok x out] = X[n_tok x in] * W^T via the gemm shader. */
static int rec_gemm(VulkanWeights *w, VkBuffer dW, VkBuffer dX, VkBuffer dY,
                    int n_tok, int in, int out) {
    PC_gemm pc = { n_tok, in, out };
    VkBuffer bufs[3] = { dW, dX, dY };
    return rec_dispatch(&w->pipes[P_GEMM], bufs, 3, &pc, sizeof(pc),
                        (uint32_t)CEIL_DIV(out, 16), (uint32_t)CEIL_DIV(n_tok, 16), 1);
}

/* matmul through a DevTensor: f32 weights go straight to GEMM; packed weights are
 * decoded to `scratch` first (decode-on-demand) then GEMM'd. */
static int rec_gemm_w(VulkanWeights *w, const DevTensor *W, VkBuffer dX, VkBuffer dY,
                      int n_tok, DevBuf *scratch) {
    if (W->is_f32)
        return rec_gemm(w, W->f32.buf, dX, dY, n_tok, W->in, W->out);
    /* dequant_arena: dispatch (ceil(cols/16), ceil(rows/16)) — rows tiled so the
     * untied Qwen3 head (151936 rows) stays under the 65535 per-dim limit. */
    PC_deq pc = { W->out, W->in };
    VkBuffer dbufs[5] = { W->codes.buf, W->row_off.buf, W->row_scale.buf,
                          W->row_prec.buf, scratch->buf };
    if (rec_dispatch(&w->pipes[P_DEQUANT], dbufs, 5, &pc, sizeof(pc),
                     (uint32_t)CEIL_DIV(W->in, 16), (uint32_t)CEIL_DIV(W->out, 16), 1)) return 1;
    return rec_gemm(w, scratch->buf, dX, dY, n_tok, W->in, W->out);
}

/* Build a NEOX-RoPE (cos,sin) table for `n_tok` positions x (d/2) freqs at the
 * given rope base, bit-for-bit the CPU rope_neox transcendentals (powf/cosf/sinf),
 * and upload it. Indexed [t*(d/2) + i]. The shader then does no pow/cos/sin. */
static int rope_table_build(DevBuf *out, int n_tok, int d, float base) {
    int half = d / 2;
    std::vector<float> cs((size_t)n_tok * half * 2);
    for (int t = 0; t < n_tok; t++)
        for (int i = 0; i < half; i++) {
            float freq  = powf(base, -2.0f * (float)i / (float)d);
            float theta = (float)t * freq;
            cs[((size_t)t * half + i) * 2 + 0] = cosf(theta);
            cs[((size_t)t * half + i) * 2 + 1] = sinf(theta);
        }
    return buf_make(out, cs.data(), cs.size() * sizeof(float));
}

/* ════════════════════════ forward ════════════════════════ */

/* shared scratch device buffers for a forward pass. drope_g/drope_l are the
 * RoPE (cos,sin) tables (gemma3: global+local bases; qwen3 uses drope_g only). */
struct Scratch {
    DevBuf dtoks, dx, dnx, dq, dk, dv, dao, dap, dg, dup, ddn, dlog, dscr;
    DevBuf drope_g, drope_l;
};

static void free_scratch(Scratch *s) {
    buf_free(&s->dtoks); buf_free(&s->dx); buf_free(&s->dnx);
    buf_free(&s->dq); buf_free(&s->dk); buf_free(&s->dv);
    buf_free(&s->dao); buf_free(&s->dap);
    buf_free(&s->dg); buf_free(&s->dup); buf_free(&s->ddn);
    buf_free(&s->dlog); buf_free(&s->dscr);
    buf_free(&s->drope_g); buf_free(&s->drope_l);
}

extern "C" int gemma3_forward_vulkan(const qwen3_model *m, const int32_t *tokens,
                                     int n_tok, float *logits) {
    if (!m || m->cfg.arch != SP_ARCH_GEMMA3) { sp_set_error("gemma3_forward_vulkan: not a gemma3 model"); return 1; }
    if (n_tok > VK_MAXTOK) { sp_set_error("gemma3_forward_vulkan: n_tok exceeds VK_MAXTOK (attn shared-mem limit)"); return 1; }

    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv, V = (int)c->n_vocab;
    const int QD = NH * HD, KVD = NKV * HD, SW = (int)c->sliding_window;
    const int group = NH / NKV;
    const float eps = c->rms_eps;
    const float gbase = c->rope_freq_base, lbase = 10000.0f;
    const float ascale = 1.0f / sqrtf((float)HD);
    const float embscale = sqrtf((float)E);
    const char *fp16_e = getenv("SP_ENGINE_FP16");
    const int fp16 = (fp16_e && fp16_e[0] == '1');

    if (g_w.key != m) { free_weights(&g_w); if (build_weights(m, &g_w)) return 1; }

    Scratch s = {};
    int rc = 1;
    if (buf_make(&s.dtoks, tokens, (size_t)n_tok * sizeof(int))) goto done;
    if (buf_create_hostvis(&s.dx,  (size_t)n_tok*E*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dnx, (size_t)n_tok*E*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dq,  (size_t)n_tok*QD*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dk,  (size_t)n_tok*KVD*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dv,  (size_t)n_tok*KVD*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dao, (size_t)n_tok*QD*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dap, (size_t)n_tok*E*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dg,  (size_t)n_tok*FF*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dup, (size_t)n_tok*FF*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.ddn, (size_t)n_tok*E*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dlog,(size_t)n_tok*V*sizeof(float))) goto done;
    if (g_w.scratch_n) { if (buf_create_hostvis(&s.dscr, g_w.scratch_n*sizeof(float))) goto done; }
    /* RoPE (cos,sin) tables: gemma3 alternates global (gbase) and local (lbase). */
    if (rope_table_build(&s.drope_g, n_tok, HD, gbase)) goto done;
    if (rope_table_build(&s.drope_l, n_tok, HD, lbase)) goto done;

    /* count dispatches: embed(1) + per layer (rmsnorm + 3 gemm_w (each <=2) +
     * 2 rmsh + 2 rope + attn + Wo gemm_w(<=2) + rmsnorm + add + rmsnorm +
     * 2 gemm_w(<=2) + gelu + Wdown gemm_w(<=2) + rmsnorm + add) + out rmsnorm +
     * head gemm. Be generous (each gemm_w may be 2). fp16 adds 7 rounds/layer + 1. */
    if (rec_begin(&g_w, 4 + (int)c->n_layers * 28 + 4 + (fp16 ? 1 + (int)c->n_layers * 7 : 0))) goto done;

    {   PC_embed pc = { n_tok, E, embscale };
        VkBuffer b[3] = { g_w.embd.buf, s.dtoks.buf, s.dx.buf };
        if (rec_dispatch(&g_w.pipes[P_EMBED], b, 3, &pc, sizeof(pc),
                         (uint32_t)CEIL_DIV(E,256), (uint32_t)n_tok, 1)) goto done; }

    for (int L = 0; L < (int)c->n_layers; L++) {
        const int global = ((L % 6) == 5);
        const int win = global ? -1 : SW;
        VkBuffer rope_tbl = global ? s.drope_g.buf : s.drope_l.buf;
        const uint32_t nE = (uint32_t)((size_t)n_tok * E);

        /* attn_norm */
        { PC_rms pc = { E, eps }; VkBuffer b[3] = { s.dx.buf, g_w.attn_norm[L].buf, s.dnx.buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM], b, 3, &pc, sizeof(pc), (uint32_t)n_tok,1,1)) goto done; }
        if (fp16) { if (rec_round_f16(&g_w, s.dnx.buf, (uint32_t)((size_t)n_tok*E))) goto done; }
        if (rec_gemm_w(&g_w, &g_w.Wq[L], s.dnx.buf, s.dq.buf, n_tok, &s.dscr)) goto done;
        if (rec_gemm_w(&g_w, &g_w.Wk[L], s.dnx.buf, s.dk.buf, n_tok, &s.dscr)) goto done;
        if (rec_gemm_w(&g_w, &g_w.Wv[L], s.dnx.buf, s.dv.buf, n_tok, &s.dscr)) goto done;
        { PC_rmsh pc = { NH, HD, QD, eps };  VkBuffer b[2] = { s.dq.buf, g_w.q_norm[L].buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM_HEAD], b, 2, &pc, sizeof(pc), (uint32_t)(n_tok*NH),1,1)) goto done; }
        { PC_rmsh pc = { NKV, HD, KVD, eps }; VkBuffer b[2] = { s.dk.buf, g_w.k_norm[L].buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM_HEAD], b, 2, &pc, sizeof(pc), (uint32_t)(n_tok*NKV),1,1)) goto done; }
        { PC_rope pc = { NH, HD, QD };  VkBuffer b[2] = { s.dq.buf, rope_tbl };
          if (rec_dispatch(&g_w.pipes[P_ROPE], b, 2, &pc, sizeof(pc), (uint32_t)(n_tok*NH),1,1)) goto done; }
        { PC_rope pc = { NKV, HD, KVD }; VkBuffer b[2] = { s.dk.buf, rope_tbl };
          if (rec_dispatch(&g_w.pipes[P_ROPE], b, 2, &pc, sizeof(pc), (uint32_t)(n_tok*NKV),1,1)) goto done; }
        if (fp16) {
            if (rec_round_f16(&g_w, s.dq.buf,  (uint32_t)((size_t)n_tok*QD)))  goto done;
            if (rec_round_f16(&g_w, s.dk.buf,  (uint32_t)((size_t)n_tok*KVD))) goto done;
            if (rec_round_f16(&g_w, s.dv.buf,  (uint32_t)((size_t)n_tok*KVD))) goto done;
        }
        { PC_attn pc = { n_tok, QD, KVD, HD, group, ascale, win };
          VkBuffer b[4] = { s.dq.buf, s.dk.buf, s.dv.buf, s.dao.buf };
          if (rec_dispatch(&g_w.pipes[P_ATTN], b, 4, &pc, sizeof(pc), (uint32_t)(n_tok*NH),1,1)) goto done; }
        if (fp16) { if (rec_round_f16(&g_w, s.dao.buf, (uint32_t)((size_t)n_tok*QD))) goto done; }
        if (rec_gemm_w(&g_w, &g_w.Wo[L], s.dao.buf, s.dap.buf, n_tok, &s.dscr)) goto done;
        { PC_rms pc = { E, eps }; VkBuffer b[3] = { s.dap.buf, g_w.post_attn[L].buf, s.dnx.buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM], b, 3, &pc, sizeof(pc), (uint32_t)n_tok,1,1)) goto done; }
        { PC_elem pc = { nE }; VkBuffer b[2] = { s.dx.buf, s.dnx.buf };
          if (rec_dispatch(&g_w.pipes[P_ADD], b, 2, &pc, sizeof(pc), (uint32_t)CEIL_DIV(nE,256),1,1)) goto done; }

        { PC_rms pc = { E, eps }; VkBuffer b[3] = { s.dx.buf, g_w.ffn_norm[L].buf, s.dnx.buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM], b, 3, &pc, sizeof(pc), (uint32_t)n_tok,1,1)) goto done; }
        if (fp16) { if (rec_round_f16(&g_w, s.dnx.buf, (uint32_t)((size_t)n_tok*E))) goto done; }
        if (rec_gemm_w(&g_w, &g_w.Wgate[L], s.dnx.buf, s.dg.buf, n_tok, &s.dscr)) goto done;
        if (rec_gemm_w(&g_w, &g_w.Wup[L], s.dnx.buf, s.dup.buf, n_tok, &s.dscr)) goto done;
        { uint32_t nFF = (uint32_t)((size_t)n_tok*FF); PC_elem pc = { nFF };
          VkBuffer b[2] = { s.dg.buf, s.dup.buf };
          if (rec_dispatch(&g_w.pipes[P_GELU], b, 2, &pc, sizeof(pc), (uint32_t)CEIL_DIV(nFF,256),1,1)) goto done; }
        if (fp16) { uint32_t nFF = (uint32_t)((size_t)n_tok*FF); if (rec_round_f16(&g_w, s.dg.buf, nFF)) goto done; }
        if (rec_gemm_w(&g_w, &g_w.Wdown[L], s.dg.buf, s.ddn.buf, n_tok, &s.dscr)) goto done;
        { PC_rms pc = { E, eps }; VkBuffer b[3] = { s.ddn.buf, g_w.post_ffw[L].buf, s.dnx.buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM], b, 3, &pc, sizeof(pc), (uint32_t)n_tok,1,1)) goto done; }
        { PC_elem pc = { nE }; VkBuffer b[2] = { s.dx.buf, s.dnx.buf };
          if (rec_dispatch(&g_w.pipes[P_ADD], b, 2, &pc, sizeof(pc), (uint32_t)CEIL_DIV(nE,256),1,1)) goto done; }
    }

    { PC_rms pc = { E, eps }; VkBuffer b[3] = { s.dx.buf, g_w.out_norm.buf, s.dnx.buf };
      if (rec_dispatch(&g_w.pipes[P_RMSNORM], b, 3, &pc, sizeof(pc), (uint32_t)n_tok,1,1)) goto done; }
    if (fp16) { if (rec_round_f16(&g_w, s.dnx.buf, (uint32_t)((size_t)n_tok*E))) goto done; }
    /* tied head, f32: reuse embd weight. */
    if (rec_gemm(&g_w, g_w.embd.buf, s.dnx.buf, s.dlog.buf, n_tok, E, V)) goto done;

    if (rec_submit_wait(&g_w)) goto done;
    if (buf_download(&s.dlog, logits, (size_t)n_tok*V*sizeof(float))) goto done;
    rc = 0;

done:
    rec_end(&g_w);
    free_scratch(&s);
    return rc;
}

/* Qwen3 forward on Vulkan (VK.4). Deltas vs gemma3: no embedding scale, plain
 * residuals (no sandwich post-norms), SwiGLU (silu) not GeGLU, single RoPE base +
 * full causal, untied LM head (g_w.head, arena-packed in Q8). Mirrors the CUDA
 * qwen3_forward_cuda_ex.
 *
 * _ex: if kv_trees != NULL, KSTE-encode each cached K head-vector (E_VK_6). The
 * post-norm/post-RoPE K is downloaded D->H per layer and run through the host
 * sp_kste_encode (byte-identical to the CPU E_CPU_6 path by construction). */
extern "C" int qwen3_forward_vulkan_ex(const qwen3_model *m, const int32_t *tokens,
                                       int n_tok, float *logits, sp_kste_tree_t *kv_trees) {
    if (!m || m->cfg.arch != SP_ARCH_QWEN3) { sp_set_error("qwen3_forward_vulkan: not a qwen3 model"); return 1; }
    if (n_tok > VK_MAXTOK) { sp_set_error("qwen3_forward_vulkan: n_tok exceeds VK_MAXTOK (attn shared-mem limit)"); return 1; }

    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv, V = (int)c->n_vocab;
    const int QD = NH * HD, KVD = NKV * HD;
    const int group = NH / NKV;
    const float eps = c->rms_eps, base = c->rope_freq_base;
    const float ascale = 1.0f / sqrtf((float)HD);
    const char *ntt_e = getenv("SP_ENGINE_NTT_ATTN");
    const int ntt = (ntt_e && ntt_e[0] == '1');
    const float ntt_qscale = 65536.0f;   /* SP_NTT_ATTN_SCALE */
    const float kste_scale = 65536.0f;   /* SP_KSTE_KV_SCALE (E_VK_6) */
    const char *fp16_e = getenv("SP_ENGINE_FP16");
    const int fp16 = (fp16_e && fp16_e[0] == '1');

    if (ntt && !vk_ctx()->has_int64) { sp_set_error("qwen3_forward_vulkan: device lacks shaderInt64 for NTT-attn"); return 1; }

    if (g_w.key != m) { free_weights(&g_w); if (build_weights(m, &g_w)) return 1; }

    Scratch s = {};
    float *host_k = NULL; int32_t *kq = NULL;
    int rc = 1;
    if (buf_make(&s.dtoks, tokens, (size_t)n_tok * sizeof(int))) goto done;
    if (buf_create_hostvis(&s.dx,  (size_t)n_tok*E*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dnx, (size_t)n_tok*E*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dq,  (size_t)n_tok*QD*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dk,  (size_t)n_tok*KVD*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dv,  (size_t)n_tok*KVD*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dao, (size_t)n_tok*QD*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dap, (size_t)n_tok*E*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dg,  (size_t)n_tok*FF*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dup, (size_t)n_tok*FF*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.ddn, (size_t)n_tok*E*sizeof(float))) goto done;
    if (buf_create_hostvis(&s.dlog,(size_t)n_tok*V*sizeof(float))) goto done;
    if (g_w.scratch_n) { if (buf_create_hostvis(&s.dscr, g_w.scratch_n*sizeof(float))) goto done; }
    if (rope_table_build(&s.drope_g, n_tok, HD, base)) goto done;   /* qwen3: single base */
    if (kv_trees) {
        host_k = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
        kq = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        if (!host_k || !kq) { sp_set_error("kste host OOM"); goto done; }
    }

    /* KSTE path needs a per-layer submit (download K mid-pass), so when kv_trees
     * is set we run one command buffer per layer; otherwise one for the whole
     * pass. To keep the code single-shape we record+submit per layer in both
     * cases when kv_trees, else batch. Here: record everything, but if KSTE,
     * flush after the K is finalized in each layer. */

    /* embed lookup, no scale (embscale=1) */
    if (rec_begin(&g_w, 4 + (int)c->n_layers * 28 + 4 + (fp16 ? 1 + (int)c->n_layers * 7 : 0))) goto done;
    {   PC_embed pc = { n_tok, E, 1.0f };
        VkBuffer b[3] = { g_w.embd.buf, s.dtoks.buf, s.dx.buf };
        if (rec_dispatch(&g_w.pipes[P_EMBED], b, 3, &pc, sizeof(pc),
                         (uint32_t)CEIL_DIV(E,256), (uint32_t)n_tok, 1)) goto done; }

    for (int L = 0; L < (int)c->n_layers; L++) {
        const uint32_t nE = (uint32_t)((size_t)n_tok * E);

        { PC_rms pc = { E, eps }; VkBuffer b[3] = { s.dx.buf, g_w.attn_norm[L].buf, s.dnx.buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM], b, 3, &pc, sizeof(pc), (uint32_t)n_tok,1,1)) goto done; }
        if (fp16) { if (rec_round_f16(&g_w, s.dnx.buf, (uint32_t)((size_t)n_tok*E))) goto done; }
        if (rec_gemm_w(&g_w, &g_w.Wq[L], s.dnx.buf, s.dq.buf, n_tok, &s.dscr)) goto done;
        if (rec_gemm_w(&g_w, &g_w.Wk[L], s.dnx.buf, s.dk.buf, n_tok, &s.dscr)) goto done;
        if (rec_gemm_w(&g_w, &g_w.Wv[L], s.dnx.buf, s.dv.buf, n_tok, &s.dscr)) goto done;
        { PC_rmsh pc = { NH, HD, QD, eps };  VkBuffer b[2] = { s.dq.buf, g_w.q_norm[L].buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM_HEAD], b, 2, &pc, sizeof(pc), (uint32_t)(n_tok*NH),1,1)) goto done; }
        { PC_rmsh pc = { NKV, HD, KVD, eps }; VkBuffer b[2] = { s.dk.buf, g_w.k_norm[L].buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM_HEAD], b, 2, &pc, sizeof(pc), (uint32_t)(n_tok*NKV),1,1)) goto done; }
        { PC_rope pc = { NH, HD, QD };  VkBuffer b[2] = { s.dq.buf, s.drope_g.buf };
          if (rec_dispatch(&g_w.pipes[P_ROPE], b, 2, &pc, sizeof(pc), (uint32_t)(n_tok*NH),1,1)) goto done; }
        { PC_rope pc = { NKV, HD, KVD }; VkBuffer b[2] = { s.dk.buf, s.drope_g.buf };
          if (rec_dispatch(&g_w.pipes[P_ROPE], b, 2, &pc, sizeof(pc), (uint32_t)(n_tok*NKV),1,1)) goto done; }
        if (fp16) {
            if (rec_round_f16(&g_w, s.dq.buf,  (uint32_t)((size_t)n_tok*QD)))  goto done;
            if (rec_round_f16(&g_w, s.dk.buf,  (uint32_t)((size_t)n_tok*KVD))) goto done;
            if (rec_round_f16(&g_w, s.dv.buf,  (uint32_t)((size_t)n_tok*KVD))) goto done;
        }

        /* E_VK_6 KSTE-KV: K must be finalized first, so flush, download D->H,
         * encode on the host, then start a fresh command buffer for the rest. */
        if (kv_trees) {
            if (rec_submit_wait(&g_w)) goto done;
            rec_end(&g_w);
            if (buf_download(&s.dk, host_k, (size_t)n_tok*KVD*sizeof(float))) goto done;
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < NKV; h++) {
                    const float *kh = host_k + (size_t)t * KVD + (size_t)h * HD;
                    for (int i = 0; i < HD; i++) kq[i] = (int32_t)lrintf(kh[i] * kste_scale);
                    sp_kste_encode(kq, HD, &kv_trees[((size_t)L * n_tok + t) * NKV + h]);
                }
            if (rec_begin(&g_w, 4 + 28 + (fp16 ? 3 : 0))) goto done;
        }

        if (ntt) {
            PC_attnn pc = { n_tok, QD, KVD, HD, group, ascale, ntt_qscale };
            VkBuffer b[4] = { s.dq.buf, s.dk.buf, s.dv.buf, s.dao.buf };
            if (rec_dispatch(&g_w.pipes[P_ATTN_NTT], b, 4, &pc, sizeof(pc), (uint32_t)(n_tok*NH),1,1)) goto done;
        } else {
            PC_attn pc = { n_tok, QD, KVD, HD, group, ascale, -1 };
            VkBuffer b[4] = { s.dq.buf, s.dk.buf, s.dv.buf, s.dao.buf };
            if (rec_dispatch(&g_w.pipes[P_ATTN], b, 4, &pc, sizeof(pc), (uint32_t)(n_tok*NH),1,1)) goto done;
        }
        if (fp16) { if (rec_round_f16(&g_w, s.dao.buf, (uint32_t)((size_t)n_tok*QD))) goto done; }
        if (rec_gemm_w(&g_w, &g_w.Wo[L], s.dao.buf, s.dap.buf, n_tok, &s.dscr)) goto done;
        { PC_elem pc = { nE }; VkBuffer b[2] = { s.dx.buf, s.dap.buf };   /* plain residual */
          if (rec_dispatch(&g_w.pipes[P_ADD], b, 2, &pc, sizeof(pc), (uint32_t)CEIL_DIV(nE,256),1,1)) goto done; }

        { PC_rms pc = { E, eps }; VkBuffer b[3] = { s.dx.buf, g_w.ffn_norm[L].buf, s.dnx.buf };
          if (rec_dispatch(&g_w.pipes[P_RMSNORM], b, 3, &pc, sizeof(pc), (uint32_t)n_tok,1,1)) goto done; }
        if (fp16) { if (rec_round_f16(&g_w, s.dnx.buf, (uint32_t)((size_t)n_tok*E))) goto done; }
        if (rec_gemm_w(&g_w, &g_w.Wgate[L], s.dnx.buf, s.dg.buf, n_tok, &s.dscr)) goto done;
        if (rec_gemm_w(&g_w, &g_w.Wup[L], s.dnx.buf, s.dup.buf, n_tok, &s.dscr)) goto done;
        { uint32_t nFF = (uint32_t)((size_t)n_tok*FF); PC_elem pc = { nFF };
          VkBuffer b[2] = { s.dg.buf, s.dup.buf };
          if (rec_dispatch(&g_w.pipes[P_SILU], b, 2, &pc, sizeof(pc), (uint32_t)CEIL_DIV(nFF,256),1,1)) goto done; }
        if (fp16) { uint32_t nFF = (uint32_t)((size_t)n_tok*FF); if (rec_round_f16(&g_w, s.dg.buf, nFF)) goto done; }
        if (rec_gemm_w(&g_w, &g_w.Wdown[L], s.dg.buf, s.ddn.buf, n_tok, &s.dscr)) goto done;
        { PC_elem pc = { nE }; VkBuffer b[2] = { s.dx.buf, s.ddn.buf };   /* plain residual */
          if (rec_dispatch(&g_w.pipes[P_ADD], b, 2, &pc, sizeof(pc), (uint32_t)CEIL_DIV(nE,256),1,1)) goto done; }

        /* KSTE per-layer mode: flush this layer's remaining work before the next
         * layer's record (so dk download stays correct each iteration). */
        if (kv_trees) {
            if (rec_submit_wait(&g_w)) goto done;
            rec_end(&g_w);
            if (L + 1 < (int)c->n_layers) { if (rec_begin(&g_w, 4 + 28 + (fp16 ? 4 : 0))) goto done; }
            else { if (rec_begin(&g_w, 8 + (fp16 ? 1 : 0))) goto done; }   /* final norm + head */
        }
    }

    { PC_rms pc = { E, eps }; VkBuffer b[3] = { s.dx.buf, g_w.out_norm.buf, s.dnx.buf };
      if (rec_dispatch(&g_w.pipes[P_RMSNORM], b, 3, &pc, sizeof(pc), (uint32_t)n_tok,1,1)) goto done; }
    if (fp16) { if (rec_round_f16(&g_w, s.dnx.buf, (uint32_t)((size_t)n_tok*E))) goto done; }
    if (rec_gemm_w(&g_w, &g_w.head, s.dnx.buf, s.dlog.buf, n_tok, &s.dscr)) goto done;   /* untied head */

    if (rec_submit_wait(&g_w)) goto done;
    if (buf_download(&s.dlog, logits, (size_t)n_tok*V*sizeof(float))) goto done;
    rc = 0;

done:
    rec_end(&g_w);
    free_scratch(&s);
    free(host_k); free(kq);
    return rc;
}

extern "C" int qwen3_forward_vulkan(const qwen3_model *m, const int32_t *tokens,
                                    int n_tok, float *logits) {
    return qwen3_forward_vulkan_ex(m, tokens, n_tok, logits, NULL);
}

extern "C" void sp_vulkan_model_release(const qwen3_model *m) {
    if (g_w.key == m) free_weights(&g_w);
}
