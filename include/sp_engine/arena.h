/* arena.h — packed-weight arena: the load-bearing inline-compression memory
 * layout (roadmap §4.8). At load the matmul weight tensors are quantized once
 * into per-row Frobenius Q8 (or Q4 mixed-precision) codes; the forward-pass
 * matmul reads the packed codes and lifts inline (no per-matmul re-quantization,
 * and — once the source is released in Phase 1b — no decoded fp32 arena).
 *
 * This in-memory packed format is what the GPU backends (CUDA/Vulkan/Hexagon)
 * will read, so it carries an explicit version. NOTE: per-ROW Frobenius Q8/Q4,
 * NOT ggml's per-32-block Q8_0 — the two are not interchangeable.
 *
 * Phase 1a (this file): arena covers the matmul weights only (attn q/k/v/o,
 * ffn gate/up/down, LM-head `output`); the embedding and norms stay f32 from the
 * GGUF mapping. Phase 1b folds the embedding in and releases the mapping.
 */
#ifndef SP_ENGINE_ARENA_H
#define SP_ENGINE_ARENA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SP_ARENA_LAYOUT_VERSION 1u

/* One packed weight tensor. Codes are row-major with a per-row byte offset and a
 * per-row precision (8 or 4) so Q4 mixed-precision can promote outlier rows to
 * Q8. A Q8 row is `cols` int8 codes; a Q4 row is ceil(cols/2) nibble pairs.
 * Dequant of code q in row r: q * row_scale[r] / (row_prec[r]==8 ? 127 : 7). */
typedef struct {
    char     name[256];
    int      rows;          /* = out (weight rows) */
    int      cols;          /* = in  (elems per row) */
    uint8_t *row_prec;      /* [rows] 8 or 4 */
    float   *row_scale;     /* [rows] per-row Frobenius scale (max abs) */
    size_t  *row_off;       /* [rows] byte offset of the row's codes in `codes` */
    uint8_t *codes;         /* packed codes (Q8: int8; Q4: nibble pairs) */
    size_t   codes_bytes;
} sp_arena_tensor;

typedef struct sp_arena sp_arena;

struct qwen3_model;   /* forward decl (sp_engine/model.h) */

/* Build a packed arena over the model's matmul weight tensors. `precision` is 8
 * (Q8) or 4 (Q4 mixed: rows whose Q4 round-trip rel-error exceeds `q4_promote`
 * are stored Q8). If `include_embed` is nonzero the token-embedding tensor is
 * packed too (Phase 1b — required before releasing the GGUF source); otherwise
 * the embedding stays f32 from the mapping (Phase 1a, which keeps the E_CPU_9
 * byte-identity-to-FROB gate). Norms are never packed. Returns NULL on error.
 * The model's GGUF mapping must still be open during the build. */
sp_arena *sp_arena_build(const struct qwen3_model *m, int precision, float q4_promote,
                         int include_embed);
void      sp_arena_free(sp_arena *a);

/* Lookup a packed tensor by its GGUF tensor name; NULL if not arena-ized. */
const sp_arena_tensor *sp_arena_find(const sp_arena *a, const char *name);

/* Reconstruct row `r` of a packed tensor to `cols` f32 values (inline lift:
 * code * row_scale / qmax). Used for the embedding lookup when the embedding is
 * in the arena. Returns 0 on success. */
int sp_arena_dequant_row(const sp_arena_tensor *at, int r, float *dst);

size_t sp_arena_bytes(const sp_arena *a);       /* total packed bytes (codes+scales+meta) */
int    sp_arena_precision(const sp_arena *a);   /* 8 or 4 */
long   sp_arena_promoted(const sp_arena *a);    /* Q4 rows promoted to Q8 (0 for Q8 arena) */
long   sp_arena_total_rows(const sp_arena *a);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_ARENA_H */
