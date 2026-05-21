/* gguf.c — GGUF v3 parser. See include/sp_engine/gguf.h.
 * Assumes a little-endian host (all engine targets are LE). */
#include "sp_engine/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#endif

#define GGUF_MAGIC 0x46554747u   /* "GGUF" little-endian */

static char *sp_strdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

struct gguf_ctx {
    const uint8_t *base;     /* mapped file base */
    uint64_t       size;     /* file size */
    uint32_t       version;
    uint64_t       n_tensors;
    uint64_t       n_kv;
    uint64_t       alignment;
    uint64_t       data_offset;
    gguf_kv       *kv;
    gguf_tensor   *tensors;
    char          *path;
#ifdef _WIN32
    HANDLE hFile, hMap;
#else
    int    fd;
#endif
};

/* ── bounds-checked cursor over the mapping ──────────────────────────────── */
typedef struct { const uint8_t *b; uint64_t size, pos; int err; } cur_t;

static const uint8_t *rd(cur_t *c, uint64_t n) {
    if (c->err || n > c->size || c->pos > c->size - n) { c->err = 1; return NULL; }
    const uint8_t *p = c->b + c->pos; c->pos += n; return p;
}
static uint32_t rd_u32(cur_t *c){ const uint8_t*p=rd(c,4); uint32_t v=0; if(p)memcpy(&v,p,4); return v; }
static uint64_t rd_u64(cur_t *c){ const uint8_t*p=rd(c,8); uint64_t v=0; if(p)memcpy(&v,p,8); return v; }
static float    rd_f32(cur_t *c){ const uint8_t*p=rd(c,4); float v=0;    if(p)memcpy(&v,p,4); return v; }
static double   rd_f64(cur_t *c){ const uint8_t*p=rd(c,8); double v=0;   if(p)memcpy(&v,p,8); return v; }

/* scalar metadata-value size in bytes (0 for STRING/ARRAY). */
static uint64_t scalar_size(gguf_value_type t) {
    switch (t) {
        case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL:   return 1;
        case GGUF_T_UINT16: case GGUF_T_INT16:                   return 2;
        case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
        case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
        default: return 0;
    }
}

/* ggml dtype block geometry: elements per block + bytes per block. */
static void type_block(uint32_t t, uint64_t *blk, uint64_t *bytes) {
    switch (t) {
        case GGML_T_F32:  *blk=1;   *bytes=4;   return;
        case GGML_T_F16:  case GGML_T_BF16: *blk=1; *bytes=2; return;
        case GGML_T_Q4_0: *blk=32;  *bytes=18;  return;
        case GGML_T_Q4_1: *blk=32;  *bytes=20;  return;
        case GGML_T_Q5_0: *blk=32;  *bytes=22;  return;
        case GGML_T_Q5_1: *blk=32;  *bytes=24;  return;
        case GGML_T_Q8_0: *blk=32;  *bytes=34;  return;
        case GGML_T_Q8_1: *blk=32;  *bytes=36;  return;
        case GGML_T_Q4_K: *blk=256; *bytes=144; return;
        case GGML_T_Q5_K: *blk=256; *bytes=176; return;
        case GGML_T_Q6_K: *blk=256; *bytes=210; return;
        case GGML_T_Q8_K: *blk=256; *bytes=292; return;
        default: *blk=0; *bytes=0; return;  /* unknown */
    }
}

const char *ggml_type_name(uint32_t t) {
    switch (t) {
        case GGML_T_F32: return "F32";  case GGML_T_F16: return "F16";
        case GGML_T_BF16: return "BF16";
        case GGML_T_Q4_0: return "Q4_0"; case GGML_T_Q4_1: return "Q4_1";
        case GGML_T_Q5_0: return "Q5_0"; case GGML_T_Q5_1: return "Q5_1";
        case GGML_T_Q8_0: return "Q8_0"; case GGML_T_Q8_1: return "Q8_1";
        case GGML_T_Q4_K: return "Q4_K"; case GGML_T_Q5_K: return "Q5_K";
        case GGML_T_Q6_K: return "Q6_K"; case GGML_T_Q8_K: return "Q8_K";
        default: return "?";
    }
}

/* read a gguf_str into a fresh NUL-terminated buffer (caller frees). */
static char *rd_str(cur_t *c) {
    uint64_t len = rd_u64(c);
    const uint8_t *p = rd(c, len);
    if (c->err) return NULL;
    char *s = (char *)malloc(len + 1);
    if (!s) { c->err = 1; return NULL; }
    if (len) memcpy(s, p, len);
    s[len] = '\0';
    return s;
}

/* copy a gguf_str into a fixed buffer, truncating safely. */
static void rd_str_into(cur_t *c, char *dst, size_t cap) {
    uint64_t len = rd_u64(c);
    const uint8_t *p = rd(c, len);
    if (c->err) { if (cap) dst[0] = '\0'; return; }
    uint64_t n = (len < cap - 1) ? len : (uint64_t)(cap - 1);
    if (n) memcpy(dst, p, (size_t)n);
    dst[n] = '\0';
}

/* skip a metadata value of the given type; record array info if requested. */
static void skip_value(cur_t *c, gguf_value_type vt, gguf_kv *kv_out) {
    if (vt == GGUF_T_ARRAY) {
        gguf_value_type at = (gguf_value_type)rd_u32(c);
        uint64_t n = rd_u64(c);
        if (kv_out) { kv_out->arr_type = at; kv_out->arr_len = n;
                      kv_out->arr_data = c->err ? NULL : (c->b + c->pos); }
        if (at == GGUF_T_STRING) {
            for (uint64_t i = 0; i < n && !c->err; i++) { uint64_t l = rd_u64(c); rd(c, l); }
        } else if (at == GGUF_T_ARRAY) {
            c->err = 1;  /* nested arrays unsupported */
        } else {
            rd(c, n * scalar_size(at));
        }
    } else if (vt == GGUF_T_STRING) {
        uint64_t l = rd_u64(c); rd(c, l);
    } else {
        rd(c, scalar_size(vt));
    }
}

static gguf_ctx *fail(gguf_ctx *ctx) { gguf_close(ctx); return NULL; }

gguf_ctx *gguf_open(const char *path) {
    gguf_ctx *ctx = (gguf_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->path = sp_strdup(path);

    /* ── map the file ── */
#ifdef _WIN32
    ctx->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (ctx->hFile == INVALID_HANDLE_VALUE) return fail(ctx);
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(ctx->hFile, &sz) || sz.QuadPart == 0) return fail(ctx);
    ctx->size = (uint64_t)sz.QuadPart;
    ctx->hMap = CreateFileMappingA(ctx->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!ctx->hMap) return fail(ctx);
    ctx->base = (const uint8_t *)MapViewOfFile(ctx->hMap, FILE_MAP_READ, 0, 0, 0);
    if (!ctx->base) return fail(ctx);
#else
    ctx->fd = open(path, O_RDONLY);
    if (ctx->fd < 0) return fail(ctx);
    struct stat st;
    if (fstat(ctx->fd, &st) != 0 || st.st_size == 0) return fail(ctx);
    ctx->size = (uint64_t)st.st_size;
    void *m = mmap(NULL, ctx->size, PROT_READ, MAP_PRIVATE, ctx->fd, 0);
    if (m == MAP_FAILED) return fail(ctx);
    ctx->base = (const uint8_t *)m;
#endif

    cur_t c = { ctx->base, ctx->size, 0, 0 };

    /* ── header ── */
    if (rd_u32(&c) != GGUF_MAGIC) return fail(ctx);
    ctx->version   = rd_u32(&c);
    ctx->n_tensors = rd_u64(&c);
    ctx->n_kv      = rd_u64(&c);
    if (c.err || ctx->version < 2 || ctx->version > 3) return fail(ctx);
    /* sanity caps to reject garbage counts before allocating */
    if (ctx->n_tensors > (1u<<24) || ctx->n_kv > (1u<<24)) return fail(ctx);

    /* ── metadata kvs ── */
    ctx->kv = (gguf_kv *)calloc(ctx->n_kv ? ctx->n_kv : 1, sizeof(gguf_kv));
    if (!ctx->kv) return fail(ctx);
    ctx->alignment = 32;  /* default; overridden by general.alignment */
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        gguf_kv *kv = &ctx->kv[i];
        rd_str_into(&c, kv->key, sizeof kv->key);
        kv->type = (gguf_value_type)rd_u32(&c);
        if (c.err) return fail(ctx);
        switch (kv->type) {
            case GGUF_T_UINT8:   kv->scalar.u = *(const uint8_t*)rd(&c,1); break;
            case GGUF_T_INT8:    kv->scalar.i = *(const int8_t*) rd(&c,1); break;
            case GGUF_T_BOOL:    kv->scalar.u = *(const uint8_t*)rd(&c,1); break;
            case GGUF_T_UINT16:  { uint16_t v=0; const uint8_t*p=rd(&c,2); if(p)memcpy(&v,p,2); kv->scalar.u=v; } break;
            case GGUF_T_INT16:   { int16_t  v=0; const uint8_t*p=rd(&c,2); if(p)memcpy(&v,p,2); kv->scalar.i=v; } break;
            case GGUF_T_UINT32:  kv->scalar.u = rd_u32(&c); break;
            case GGUF_T_INT32:   kv->scalar.i = (int32_t)rd_u32(&c); break;
            case GGUF_T_FLOAT32: kv->scalar.f = rd_f32(&c); break;
            case GGUF_T_UINT64:  kv->scalar.u = rd_u64(&c); break;
            case GGUF_T_INT64:   kv->scalar.i = (int64_t)rd_u64(&c); break;
            case GGUF_T_FLOAT64: kv->scalar.f = rd_f64(&c); break;
            case GGUF_T_STRING:  kv->str = rd_str(&c); break;
            case GGUF_T_ARRAY:   skip_value(&c, GGUF_T_ARRAY, kv); break;
            default: return fail(ctx);
        }
        if (c.err) return fail(ctx);
        if (kv->type == GGUF_T_UINT32 && strcmp(kv->key, "general.alignment") == 0)
            ctx->alignment = kv->scalar.u ? kv->scalar.u : 32;
    }

    /* ── tensor infos ── */
    ctx->tensors = (gguf_tensor *)calloc(ctx->n_tensors ? ctx->n_tensors : 1, sizeof(gguf_tensor));
    if (!ctx->tensors) return fail(ctx);
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        gguf_tensor *t = &ctx->tensors[i];
        rd_str_into(&c, t->name, sizeof t->name);
        t->n_dims = rd_u32(&c);
        if (c.err || t->n_dims == 0 || t->n_dims > GGUF_MAX_DIMS) return fail(ctx);
        t->n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) { t->dims[d] = rd_u64(&c); t->n_elements *= t->dims[d]; }
        t->type   = rd_u32(&c);
        t->offset = rd_u64(&c);
        if (c.err) return fail(ctx);
        uint64_t blk, bb; type_block(t->type, &blk, &bb);
        if (blk == 0 || (t->n_elements % blk) != 0) return fail(ctx);  /* unknown type / bad shape */
        t->nbytes = (t->n_elements / blk) * bb;
    }

    /* ── tensor data base: align the post-table cursor ── */
    uint64_t a = ctx->alignment;
    ctx->data_offset = (c.pos + a - 1) / a * a;
    if (ctx->data_offset > ctx->size) return fail(ctx);

    /* validate every tensor lies within the file */
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        const gguf_tensor *t = &ctx->tensors[i];
        if (t->offset % a != 0) return fail(ctx);                  /* tensors are aligned */
        uint64_t end = ctx->data_offset + t->offset + t->nbytes;
        if (end < t->nbytes || end > ctx->size) return fail(ctx);  /* overflow / OOB */
    }
    return ctx;
}

void gguf_close(gguf_ctx *ctx) {
    if (!ctx) return;
    if (ctx->kv) { for (uint64_t i = 0; i < ctx->n_kv; i++) free(ctx->kv[i].str); free(ctx->kv); }
    free(ctx->tensors);
    free(ctx->path);
#ifdef _WIN32
    if (ctx->base) UnmapViewOfFile((LPCVOID)ctx->base);
    if (ctx->hMap) CloseHandle(ctx->hMap);
    if (ctx->hFile && ctx->hFile != INVALID_HANDLE_VALUE) CloseHandle(ctx->hFile);
#else
    if (ctx->base) munmap((void *)ctx->base, ctx->size);
    if (ctx->fd > 0) close(ctx->fd);
#endif
    free(ctx);
}

uint32_t gguf_version(const gguf_ctx *c)     { return c->version; }
uint64_t gguf_n_tensors(const gguf_ctx *c)   { return c->n_tensors; }
uint64_t gguf_n_kv(const gguf_ctx *c)        { return c->n_kv; }
uint64_t gguf_alignment(const gguf_ctx *c)   { return c->alignment; }
uint64_t gguf_data_offset(const gguf_ctx *c) { return c->data_offset; }
uint64_t gguf_file_size(const gguf_ctx *c)   { return c->size; }

const gguf_tensor *gguf_tensor_at(const gguf_ctx *c, uint64_t i) {
    return (i < c->n_tensors) ? &c->tensors[i] : NULL;
}
const gguf_tensor *gguf_find_tensor(const gguf_ctx *c, const char *name) {
    for (uint64_t i = 0; i < c->n_tensors; i++)
        if (strcmp(c->tensors[i].name, name) == 0) return &c->tensors[i];
    return NULL;
}
const void *gguf_tensor_data(const gguf_ctx *c, const gguf_tensor *t) {
    if (!t) return NULL;
    return c->base + c->data_offset + t->offset;
}

const gguf_kv *gguf_kv_at(const gguf_ctx *c, uint64_t i) {
    return (i < c->n_kv) ? &c->kv[i] : NULL;
}
const gguf_kv *gguf_find_kv(const gguf_ctx *c, const char *key) {
    for (uint64_t i = 0; i < c->n_kv; i++)
        if (strcmp(c->kv[i].key, key) == 0) return &c->kv[i];
    return NULL;
}
int gguf_get_u64(const gguf_ctx *c, const char *key, uint64_t *out) {
    const gguf_kv *kv = gguf_find_kv(c, key);
    if (!kv) return 0;
    switch (kv->type) {
        case GGUF_T_UINT8: case GGUF_T_UINT16: case GGUF_T_UINT32: case GGUF_T_UINT64:
        case GGUF_T_BOOL: *out = kv->scalar.u; return 1;
        case GGUF_T_INT8: case GGUF_T_INT16: case GGUF_T_INT32: case GGUF_T_INT64:
            *out = (uint64_t)kv->scalar.i; return 1;
        default: return 0;
    }
}
int gguf_get_f32(const gguf_ctx *c, const char *key, float *out) {
    const gguf_kv *kv = gguf_find_kv(c, key);
    if (!kv) return 0;
    if (kv->type == GGUF_T_FLOAT32 || kv->type == GGUF_T_FLOAT64) { *out = (float)kv->scalar.f; return 1; }
    return 0;
}
const char *gguf_get_str(const gguf_ctx *c, const char *key) {
    const gguf_kv *kv = gguf_find_kv(c, key);
    return (kv && kv->type == GGUF_T_STRING) ? kv->str : NULL;
}
