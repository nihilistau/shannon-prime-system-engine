/* sp_model_load.c — E_FMT_1: the .sp-model loader (PPT-LAT-SP-MODEL-v0 §8).
 * Pure mmap + 512-byte header memcpy + tensor-table pointer setup + tokenizer
 * SHA-256 verify. ZERO malloc proportional to tensor data: the file IS the
 * in-memory layout; only the small sp_model handle is heap-allocated.
 * Win32 CreateFileMapping mirrors src/loader/gguf.c. */
#include "sp_engine/sp_model.h"
#include "sp_hash.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#endif

void sp_set_error(const char *msg);   /* internal (src/common/sp_error.c) */

struct sp_model {
    const uint8_t  *base;       /* model mmap base */
    uint64_t        size;
    const uint8_t  *tok_base;   /* tokenizer mmap base */
    uint64_t        tok_size;
    sp_model_header header;      /* parsed copy */
    const sp_tensor_entry *table;   /* into base */
    const uint8_t  *data;           /* base + tensor_data_offset */
    const sp_tok_header   *tok_hdr; /* into tok_base */
#ifdef _WIN32
    HANDLE hF, hM, hTF, hTM;
#else
    int fd, tok_fd;
#endif
};

/* mmap a whole file read-only. Returns base (and sets *size); NULL on error. */
static const uint8_t *map_file(const char *path, uint64_t *size,
#ifdef _WIN32
                               HANDLE *hF, HANDLE *hM
#else
                               int *fd
#endif
                               ) {
#ifdef _WIN32
    *hF = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                      FILE_ATTRIBUTE_NORMAL, NULL);
    if (*hF == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(*hF, &sz) || sz.QuadPart == 0) return NULL;
    *size = (uint64_t)sz.QuadPart;
    *hM = CreateFileMappingA(*hF, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!*hM) return NULL;
    return (const uint8_t *)MapViewOfFile(*hM, FILE_MAP_READ, 0, 0, 0);
#else
    *fd = open(path, O_RDONLY);
    if (*fd < 0) return NULL;
    struct stat st;
    if (fstat(*fd, &st) != 0 || st.st_size == 0) return NULL;
    *size = (uint64_t)st.st_size;
    void *p = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, *fd, 0);
    return (p == MAP_FAILED) ? NULL : (const uint8_t *)p;
#endif
}

static sp_status fail(sp_model *m, sp_status code, const char *msg) {
    sp_set_error(msg);
    sp_model_unload(m);
    return code;
}

sp_status sp_model_load(const char *model_path, const char *tok_path, sp_model **out) {
    if (!model_path || !tok_path || !out) { sp_set_error("sp_model_load: null arg"); return SP_EBADARG; }
    *out = NULL;
    sp_model *m = (sp_model *)calloc(1, sizeof *m);
    if (!m) { sp_set_error("sp_model_load: OOM handle"); return SP_ENOMEM; }
#ifndef _WIN32
    m->fd = -1; m->tok_fd = -1;
#endif

    /* ── map the model file ── */
#ifdef _WIN32
    m->base = map_file(model_path, &m->size, &m->hF, &m->hM);
#else
    m->base = map_file(model_path, &m->size, &m->fd);
#endif
    if (!m->base) return fail(m, SP_EIO, "sp_model_load: cannot open/mmap .sp-model");
    if (m->size < SP_HEADER_SIZE) return fail(m, SP_EBADFORMAT, "sp_model_load: file shorter than header");

    /* ── header memcpy + verify (§8 steps 3-12) ── */
    memcpy(&m->header, m->base, SP_HEADER_SIZE);
    const sp_model_header *h = &m->header;
    if (h->magic != SP_MODEL_MAGIC)        return fail(m, SP_EBADFORMAT, "sp_model_load: bad magic (not SPMD)");
    if (h->version_major != SP_MODEL_VER_MAJOR) return fail(m, SP_EBADFORMAT, "sp_model_load: unsupported version_major");
    if (h->header_size != SP_HEADER_SIZE)  return fail(m, SP_EBADFORMAT, "sp_model_load: header_size != 512");
    if (sp_crc32(m->base, SP_HEADER_CRC_COVER) != h->header_crc32)
        return fail(m, SP_EBADFORMAT, "sp_model_load: header CRC-32 mismatch");
    if (h->file_size != m->size)           return fail(m, SP_EBADFORMAT, "sp_model_load: file_size != stat size");
    if (h->arch_struct_capacity != 256 || h->arch_struct_size > 256)
        return fail(m, SP_EBADFORMAT, "sp_model_load: bad arch_struct sizing");
    if (h->tensor_table_offset != SP_HEADER_SIZE || (h->tensor_table_offset % SP_TENSOR_ALIGN) != 0)
        return fail(m, SP_EBADFORMAT, "sp_model_load: tensor_table_offset invalid");
    if ((h->tensor_data_offset % SP_DATA_REGION_ALIGN) != 0)
        return fail(m, SP_EBADFORMAT, "sp_model_load: tensor_data_offset not 65536-aligned");
    /* table + data must lie inside the file */
    uint64_t table_end = h->tensor_table_offset + (uint64_t)h->tensor_count * SP_TENSOR_ENTRY_SIZE;
    if (table_end < h->tensor_table_offset || table_end > m->size ||
        h->tensor_data_offset < table_end  || h->tensor_data_offset > m->size)
        return fail(m, SP_EBADFORMAT, "sp_model_load: table/data offsets out of range");

    m->table = (const sp_tensor_entry *)(m->base + h->tensor_table_offset);
    m->data  = m->base + h->tensor_data_offset;

    /* ── tokenizer: mmap + SHA-256 over the entire file (§8 steps 13-16) ── */
#ifdef _WIN32
    m->tok_base = map_file(tok_path, &m->tok_size, &m->hTF, &m->hTM);
#else
    m->tok_base = map_file(tok_path, &m->tok_size, &m->tok_fd);
#endif
    if (!m->tok_base) return fail(m, SP_EIO, "sp_model_load: cannot open/mmap .sp-tokenizer");
    if (m->tok_size < SP_TOK_HEADER_SIZE) return fail(m, SP_EBADFORMAT, "sp_model_load: tokenizer shorter than header");
    uint8_t sha[32];
    sp_sha256(m->tok_base, m->tok_size, sha);
    if (memcmp(sha, h->tokenizer_hash, 32) != 0)
        return fail(m, SP_ETOKENIZER_HASH, "sp_model_load: tokenizer SHA-256 != header.tokenizer_hash");

    m->tok_hdr = (const sp_tok_header *)m->tok_base;
    if (m->tok_hdr->magic != SP_TOK_MAGIC)
        return fail(m, SP_EBADFORMAT, "sp_model_load: tokenizer bad magic (not SPTK)");
    if (m->tok_hdr->vocab_size != h->vocab_size)
        return fail(m, SP_EVOCAB, "sp_model_load: tokenizer vocab != model vocab");

    *out = m;
    return SP_OK;
}

void sp_model_unload(sp_model *m) {
    if (!m) return;
#ifdef _WIN32
    if (m->base)     UnmapViewOfFile((LPCVOID)m->base);
    if (m->tok_base) UnmapViewOfFile((LPCVOID)m->tok_base);
    if (m->hM)  CloseHandle(m->hM);
    if (m->hF && m->hF != INVALID_HANDLE_VALUE)  CloseHandle(m->hF);
    if (m->hTM) CloseHandle(m->hTM);
    if (m->hTF && m->hTF != INVALID_HANDLE_VALUE) CloseHandle(m->hTF);
#else
    if (m->base)     munmap((void *)m->base, m->size);
    if (m->tok_base) munmap((void *)m->tok_base, m->tok_size);
    if (m->fd >= 0)     close(m->fd);
    if (m->tok_fd >= 0) close(m->tok_fd);
#endif
    free(m);
}

const sp_model_header *sp_model_get_header(const sp_model *m) { return m ? &m->header : NULL; }
uint32_t sp_model_tensor_count(const sp_model *m) { return m ? m->header.tensor_count : 0; }
const sp_tensor_entry *sp_model_tensor_at(const sp_model *m, uint32_t i) {
    return (m && i < m->header.tensor_count) ? &m->table[i] : NULL;
}

const sp_tensor_entry *sp_model_find_tensor(const sp_model *m, const char *name) {
    if (!m || !name) return NULL;
    uint64_t want = sp_xxh64(name, strlen(name), 0);
    uint32_t lo = 0, hi = m->header.tensor_count;          /* table sorted by name_hash asc */
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        uint64_t hv = m->table[mid].name_hash;
        if (hv < want) lo = mid + 1; else hi = mid;
    }
    /* scan the run of equal hashes, verify name (defends the 2^-64 collision) */
    for (uint32_t i = lo; i < m->header.tensor_count && m->table[i].name_hash == want; i++)
        if (strncmp(m->table[i].name, name, sizeof m->table[i].name) == 0)
            return &m->table[i];
    return NULL;
}

const void *sp_model_tensor_data(const sp_model *m, const sp_tensor_entry *e) {
    if (!m || !e) return NULL;
    if (e->offset_in_data + e->size_bytes > (m->size - m->header.tensor_data_offset)) return NULL;
    return m->data + e->offset_in_data;
}

const void *sp_model_tokenizer_blob(const sp_model *m, uint64_t *size_out) {
    if (!m || !m->tok_hdr) return NULL;
    if (m->tok_hdr->blob_offset + m->tok_hdr->blob_size > m->tok_size) return NULL;
    if (size_out) *size_out = m->tok_hdr->blob_size;
    return m->tok_base + m->tok_hdr->blob_offset;
}
