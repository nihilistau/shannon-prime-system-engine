/* tokenizer.c — GGUF vocab load + byte-level decode (GPT2/Qwen family).
 * See sp_engine/tokenizer.h. Encode (regex pre-tokenizer + ranked BPE) is next. */
#include "sp_engine/tokenizer.h"

#include <stdlib.h>
#include <string.h>

struct sp_tokenizer {
    uint32_t      n_vocab;
    const char  **tok;     /* n_vocab pointers into the GGUF mapping (not owned)  */
    uint64_t     *len;     /* n_vocab token byte-lengths                          */
    int           cp_to_byte[512];  /* GPT2 byte-level inverse: codepoint -> byte */
    int           max_cp;
};

/* GPT2 bytes_to_unicode, inverted. Printable bytes (33..126, 161..172, 174..255)
 * map to themselves; the remaining 68 bytes map to codepoints 256.. in order. */
static void build_byte_decoder(int *cp_to_byte, int *max_cp) {
    int printable[256]; for (int b = 0; b < 256; b++) printable[b] = 0;
    for (int b = 33;  b <= 126; b++) printable[b] = 1;
    for (int b = 161; b <= 172; b++) printable[b] = 1;
    for (int b = 174; b <= 255; b++) printable[b] = 1;
    int byte_to_cp[256], n = 0;
    for (int b = 0; b < 256; b++) byte_to_cp[b] = printable[b] ? b : (256 + n++);
    int mx = 0;
    for (int b = 0; b < 256; b++) if (byte_to_cp[b] > mx) mx = byte_to_cp[b];
    *max_cp = mx;
    for (int c = 0; c <= mx; c++) cp_to_byte[c] = -1;
    for (int b = 0; b < 256; b++) cp_to_byte[byte_to_cp[b]] = b;
}

sp_tokenizer *sp_tokenizer_load(const gguf_ctx *g) {
    const gguf_kv *kv = gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!kv || kv->type != GGUF_T_ARRAY || kv->arr_type != GGUF_T_STRING) return NULL;
    uint64_t nv = kv->arr_len;
    if (nv == 0 || nv > 100000000ull) return NULL;

    sp_tokenizer *t = (sp_tokenizer *)calloc(1, sizeof *t);
    if (!t) return NULL;
    t->n_vocab = (uint32_t)nv;
    t->tok = (const char **)malloc((size_t)nv * sizeof(char *));
    t->len = (uint64_t *)malloc((size_t)nv * sizeof(uint64_t));
    if (!t->tok || !t->len) { sp_tokenizer_free(t); return NULL; }
    if (gguf_kv_str_array(g, kv, t->tok, t->len, nv) != nv) { sp_tokenizer_free(t); return NULL; }
    build_byte_decoder(t->cp_to_byte, &t->max_cp);
    return t;
}

void sp_tokenizer_free(sp_tokenizer *t) {
    if (!t) return;
    free(t->tok); free(t->len); free(t);
}

uint32_t sp_tokenizer_vocab_size(const sp_tokenizer *t) { return t ? t->n_vocab : 0; }

/* append one decoded byte (or skip-on-overflow), tracking the true length. */
static void put(char *buf, size_t cap, size_t *pos, unsigned char b) {
    if (*pos + 1 < cap) buf[*pos] = (char)b;
    (*pos)++;
}

long sp_tokenizer_decode(const sp_tokenizer *t, const int32_t *ids, int n,
                         char *buf, size_t cap) {
    if (!t || (!ids && n > 0) || (!buf && cap > 0)) return -1;
    size_t pos = 0;
    for (int k = 0; k < n; k++) {
        int32_t id = ids[k];
        if (id < 0 || (uint32_t)id >= t->n_vocab) continue;   /* skip OOR */
        const char *s = t->tok[id];
        uint64_t L = t->len[id];
        uint64_t i = 0;
        while (i < L) {
            unsigned char c0 = (unsigned char)s[i];
            uint32_t cp; uint64_t adv;
            if (c0 < 0x80)                                  { cp = c0; adv = 1; }
            else if ((c0 & 0xE0) == 0xC0 && i + 1 < L)      { cp = ((uint32_t)(c0 & 0x1F) << 6) | (s[i+1] & 0x3F); adv = 2; }
            else if ((c0 & 0xF0) == 0xE0 && i + 2 < L)      { cp = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); adv = 3; }
            else if ((c0 & 0xF8) == 0xF0 && i + 3 < L)      { cp = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(s[i+1] & 0x3F) << 12) | ((uint32_t)(s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); adv = 4; }
            else                                            { cp = c0; adv = 1; }
            int byte = (cp <= (uint32_t)t->max_cp) ? t->cp_to_byte[cp] : -1;
            if (byte >= 0) {
                put(buf, cap, &pos, (unsigned char)byte);
            } else {
                /* codepoint outside the byte-level alphabet (e.g. a special-token
                 * literal): emit its raw UTF-8 bytes unchanged. */
                for (uint64_t j = 0; j < adv; j++) put(buf, cap, &pos, (unsigned char)s[i + j]);
            }
            i += adv;
        }
    }
    if (cap > 0) buf[pos < cap ? pos : cap - 1] = '\0';
    return (long)pos;
}
