/* tokenizer.c — GGUF vocab load + byte-level BPE (GPT2 / Qwen2 family).
 *
 * DECODE (token IDs -> UTF-8): inverse GPT2 byte-level coding.
 * ENCODE (UTF-8 -> token IDs): the Qwen2 pipeline —
 *   1. optional special-token pre-split (CONTROL/USER_DEFINED surfaces, longest
 *      match first), with the gaps tokenized as ordinary text;
 *   2. the Qwen2 pre-tokenizer regex split (hand-coded; classes \p{L}/\p{N}/\s
 *      come from the generated unicode_ranges.h);
 *   3. GPT2 byte-level encode of each piece;
 *   4. greedy lowest-rank BPE over tokenizer.ggml.merges;
 *   5. token->id lookup.
 * Validated to reproduce stock llama.cpp IDs byte-for-byte (see tools/oracle/
 * bpe_proto.py and the TOK_ENCODE test). See sp_engine/tokenizer.h. */
#include "sp_engine/tokenizer.h"
#include "unicode_ranges.h"

#include <stdlib.h>
#include <string.h>

/* ---- string hash map (open addressing, linear probe): key = bytes -> int64 -- */
typedef struct { const char *key; uint32_t klen; int64_t val; uint8_t used; } sp_hent;
typedef struct { sp_hent *e; size_t mask; } sp_hmap;

static uint64_t fnv1a(const char *s, uint32_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
    return h;
}
static int hmap_init(sp_hmap *m, size_t want) {
    size_t cap = 16; while (cap < want * 2) cap <<= 1;
    m->e = (sp_hent *)calloc(cap, sizeof *m->e);
    if (!m->e) return 0;
    m->mask = cap - 1; return 1;
}
static void hmap_free(sp_hmap *m) { free(m->e); m->e = NULL; m->mask = 0; }
/* insert; first writer for a key wins (keeps the lowest merge rank). */
static void hmap_put(sp_hmap *m, const char *k, uint32_t kl, int64_t v) {
    size_t i = fnv1a(k, kl) & m->mask;
    while (m->e[i].used) {
        if (m->e[i].klen == kl && memcmp(m->e[i].key, k, kl) == 0) return;
        i = (i + 1) & m->mask;
    }
    m->e[i].key = k; m->e[i].klen = kl; m->e[i].val = v; m->e[i].used = 1;
}
static int64_t hmap_get(const sp_hmap *m, const char *k, uint32_t kl, int64_t dflt) {
    size_t i = fnv1a(k, kl) & m->mask;
    while (m->e[i].used) {
        if (m->e[i].klen == kl && memcmp(m->e[i].key, k, kl) == 0) return m->e[i].val;
        i = (i + 1) & m->mask;
    }
    return dflt;
}

/* ---- special-token surface (for parse_special pre-split) -------------------- */
typedef struct { const char *surf; uint32_t len; int32_t id; } sp_special;

struct sp_tokenizer {
    uint32_t      n_vocab;
    const char  **tok;          /* n_vocab pointers into the GGUF mapping (not owned) */
    uint64_t     *len;          /* n_vocab token byte-lengths                         */
    int           cp_to_byte[512]; /* GPT2 byte-level inverse: codepoint -> byte      */
    int           max_cp;
    /* encode side */
    uint8_t       bcp[256][2];  /* byte -> UTF-8 of its byte-level codepoint           */
    uint8_t       bcp_len[256]; /* 1 or 2                                              */
    sp_hmap       vocab;        /* token-string bytes -> id                            */
    sp_hmap       merge;        /* "A B" merge-line bytes -> rank                      */
    sp_special   *spec;         /* special surfaces, sorted longest-first              */
    uint32_t      n_spec;
};

/* GPT2 bytes_to_unicode: printable bytes (33..126,161..172,174..255) map to
 * themselves; the remaining 68 map to codepoints 256.. in order. */
static void build_byte_maps(sp_tokenizer *t) {
    int printable[256]; for (int b = 0; b < 256; b++) printable[b] = 0;
    for (int b = 33;  b <= 126; b++) printable[b] = 1;
    for (int b = 161; b <= 172; b++) printable[b] = 1;
    for (int b = 174; b <= 255; b++) printable[b] = 1;
    int byte_to_cp[256], n = 0;
    for (int b = 0; b < 256; b++) byte_to_cp[b] = printable[b] ? b : (256 + n++);
    int mx = 0;
    for (int b = 0; b < 256; b++) if (byte_to_cp[b] > mx) mx = byte_to_cp[b];
    t->max_cp = mx;
    for (int c = 0; c <= mx; c++) t->cp_to_byte[c] = -1;
    for (int b = 0; b < 256; b++) {
        int cp = byte_to_cp[b];
        t->cp_to_byte[cp] = b;
        if (cp < 0x80) { t->bcp[b][0] = (uint8_t)cp; t->bcp_len[b] = 1; }
        else { t->bcp[b][0] = (uint8_t)(0xC0 | (cp >> 6)); t->bcp[b][1] = (uint8_t)(0x80 | (cp & 0x3F)); t->bcp_len[b] = 2; }
    }
}

static int spec_cmp_desc(const void *a, const void *b) {
    uint32_t la = ((const sp_special *)a)->len, lb = ((const sp_special *)b)->len;
    return (la < lb) - (la > lb);   /* longer first */
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
    build_byte_maps(t);

    /* vocab map: token bytes -> id */
    if (!hmap_init(&t->vocab, (size_t)nv)) { sp_tokenizer_free(t); return NULL; }
    for (uint32_t i = 0; i < t->n_vocab; i++)
        hmap_put(&t->vocab, t->tok[i], (uint32_t)t->len[i], (int64_t)i);

    /* merge map: "A B" merge-line bytes -> rank (line index) */
    const gguf_kv *mkv = gguf_find_kv(g, "tokenizer.ggml.merges");
    if (mkv && mkv->type == GGUF_T_ARRAY && mkv->arr_type == GGUF_T_STRING && mkv->arr_len > 0) {
        uint64_t nm = mkv->arr_len;
        const char **mp = (const char **)malloc((size_t)nm * sizeof(char *));
        uint64_t    *ml = (uint64_t *)malloc((size_t)nm * sizeof(uint64_t));
        if (mp && ml && gguf_kv_str_array(g, mkv, mp, ml, nm) == nm && hmap_init(&t->merge, (size_t)nm)) {
            for (uint64_t i = 0; i < nm; i++) hmap_put(&t->merge, mp[i], (uint32_t)ml[i], (int64_t)i);
        }
        free((void *)mp); free(ml);   /* keys point into the mapping; arrays themselves are scratch */
    }

    /* special surfaces: token_type CONTROL(3) or USER_DEFINED(4) */
    const gguf_kv *tt = gguf_find_kv(g, "tokenizer.ggml.token_type");
    if (tt && tt->type == GGUF_T_ARRAY &&
        (tt->arr_type == GGUF_T_INT32 || tt->arr_type == GGUF_T_UINT32) &&
        tt->arr_len == nv) {
        const int32_t *types = (const int32_t *)tt->arr_data;
        uint32_t cnt = 0;
        for (uint64_t i = 0; i < nv; i++) if (types[i] == 3 || types[i] == 4) cnt++;
        if (cnt) {
            t->spec = (sp_special *)malloc(cnt * sizeof *t->spec);
            if (t->spec) {
                uint32_t j = 0;
                for (uint32_t i = 0; i < t->n_vocab; i++)
                    if (types[i] == 3 || types[i] == 4) {
                        t->spec[j].surf = t->tok[i];
                        t->spec[j].len  = (uint32_t)t->len[i];
                        t->spec[j].id   = (int32_t)i;
                        j++;
                    }
                t->n_spec = j;
                qsort(t->spec, t->n_spec, sizeof *t->spec, spec_cmp_desc);
            }
        }
    }
    return t;
}

void sp_tokenizer_free(sp_tokenizer *t) {
    if (!t) return;
    hmap_free(&t->vocab); hmap_free(&t->merge);
    free(t->spec); free((void *)t->tok); free(t->len); free(t);
}

uint32_t sp_tokenizer_vocab_size(const sp_tokenizer *t) { return t ? t->n_vocab : 0; }

/* ---- decode (unchanged) ----------------------------------------------------- */
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
            if (byte >= 0) put(buf, cap, &pos, (unsigned char)byte);
            else for (uint64_t j = 0; j < adv; j++) put(buf, cap, &pos, (unsigned char)s[i + j]);
            i += adv;
        }
    }
    if (cap > 0) buf[pos < cap ? pos : cap - 1] = '\0';
    return (long)pos;
}

/* ---- encode helpers --------------------------------------------------------- */
#define CR 0x0Du
#define LF 0x0Au
#define SP 0x20u

static int in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}
static int is_L(uint32_t c) { return in_ranges(sp_uni_letter, sp_uni_letter_n, c); }
static int is_N(uint32_t c) { return in_ranges(sp_uni_number, sp_uni_number_n, c); }
static int is_S(uint32_t c) { return in_ranges(sp_uni_space,  sp_uni_space_n,  c); }
/* class for ` ?[^\s\p{L}\p{N}]+`: not whitespace, letter, or number. */
static int is_C4(uint32_t c) { return !is_S(c) && !is_L(c) && !is_N(c); }
static uint32_t lc(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

/* decode one UTF-8 codepoint at s[i..len); writes *cp, returns byte advance. */
static size_t cp_decode(const unsigned char *s, size_t len, size_t i, uint32_t *cp) {
    unsigned char c0 = s[i];
    if (c0 < 0x80)                              { *cp = c0; return 1; }
    if ((c0 & 0xE0) == 0xC0 && i + 1 < len)     { *cp = ((uint32_t)(c0 & 0x1F) << 6) | (s[i+1] & 0x3F); return 2; }
    if ((c0 & 0xF0) == 0xE0 && i + 2 < len)     { *cp = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); return 3; }
    if ((c0 & 0xF8) == 0xF0 && i + 3 < len)     { *cp = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(s[i+1] & 0x3F) << 12) | ((uint32_t)(s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); return 4; }
    *cp = c0; return 1;   /* malformed: one byte */
}

/* Qwen2 pre-tokenizer: given the codepoint array cp[0..n), return the end index
 * (exclusive) of the pre-token that starts at position i (always > i). Mirrors
 * the alternation
 *   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
 *   |[^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * with ordered (first-match-wins) alternation and the \s+(?!\S) backtrack. */
static size_t next_piece(const uint32_t *cp, size_t n, size_t i) {
    /* ALT1 — contraction suffixes */
    if (cp[i] == 0x27 && i + 1 < n) {
        uint32_t c1 = lc(cp[i + 1]);
        if (i + 2 < n) {
            uint32_t c2 = lc(cp[i + 2]);
            if (((c1 == 'r' || c1 == 'v') && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return i + 3;
        }
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return i + 2;
    }
    /* ALT2 — [^\r\n\p{L}\p{N}]? \p{L}+ */
    {
        size_t k = i;
        if (cp[k] != CR && cp[k] != LF && !is_L(cp[k]) && !is_N(cp[k])) {
            if (k + 1 < n && is_L(cp[k + 1])) k++;   /* greedy optional prefix, needs a letter next */
        }
        if (k < n && is_L(cp[k])) {
            while (k < n && is_L(cp[k])) k++;
            return k;
        }
    }
    /* ALT3 — \p{N} (single) */
    if (is_N(cp[i])) return i + 1;
    /* ALT4 —  ?[^\s\p{L}\p{N}]+[\r\n]* */
    {
        size_t k = i;
        if (cp[k] == SP) {
            if (k + 1 < n && is_C4(cp[k + 1])) k++;  /* consume the optional leading space */
            else goto alt4_done;                     /* lone space is \s, not class C4 */
        }
        if (k < n && is_C4(cp[k])) {
            while (k < n && is_C4(cp[k])) k++;
            while (k < n && (cp[k] == CR || cp[k] == LF)) k++;
            return k;
        }
    }
alt4_done:;
    /* ALT5/6/7 — whitespace run (cp[i] is whitespace if we reach here) */
    {
        size_t e = i; while (e < n && is_S(cp[e])) e++;
        size_t lastnl = (size_t)-1;
        for (size_t k = i; k < e; k++) if (cp[k] == CR || cp[k] == LF) lastnl = k;
        if (lastnl != (size_t)-1) return lastnl + 1;   /* ALT5: \s*[\r\n]+         */
        if (e == n)              return e;             /* ALT6: \s+ at end-of-text */
        if (e - i >= 2)          return e - 1;         /* ALT6: \s+(?!\S) backtrack */
        return e;                                      /* ALT7: single \s before \S */
    }
}

/* append id to out (capacity max_out); always advance *cnt so the caller can
 * detect truncation (return value > max_out). */
static void emit(int32_t *out, int max_out, long *cnt, int32_t id) {
    if (*cnt < max_out) out[*cnt] = id;
    (*cnt)++;
}

/* BPE-encode one pre-token (raw bytes [rb,rb+rn)) into out, using scratch
 * buffers: enc[] (byte-level UTF-8, >= 2*rn), sym off/len arrays (>= rn),
 * key[] (>= enc capacity + 1). */
static int bpe_piece(const sp_tokenizer *t, const unsigned char *rb, size_t rn,
                     unsigned char *enc, uint32_t *soff, uint32_t *slen, unsigned char *key,
                     int32_t *out, int max_out, long *cnt) {
    /* byte-level encode; one symbol per source byte */
    size_t ne = 0, ns = 0;
    for (size_t i = 0; i < rn; i++) {
        unsigned char b = rb[i];
        soff[ns] = (uint32_t)ne;
        slen[ns] = t->bcp_len[b];
        for (uint8_t j = 0; j < t->bcp_len[b]; j++) enc[ne++] = t->bcp[b][j];
        ns++;
    }
    /* greedy lowest-rank merges */
    while (ns > 1) {
        int64_t best = -1; size_t bi = 0;
        for (size_t i = 0; i + 1 < ns; i++) {
            uint32_t la = slen[i], lb = slen[i + 1];
            memcpy(key, enc + soff[i], la);
            key[la] = ' ';
            memcpy(key + la + 1, enc + soff[i + 1], lb);
            int64_t r = hmap_get(&t->merge, (const char *)key, la + 1 + lb, -1);
            if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
        }
        if (best < 0) break;
        slen[bi] += slen[bi + 1];                       /* symbols are contiguous in enc */
        for (size_t k = bi + 1; k + 1 < ns; k++) { soff[k] = soff[k + 1]; slen[k] = slen[k + 1]; }
        ns--;
    }
    /* map final symbols to ids */
    for (size_t i = 0; i < ns; i++) {
        int64_t id = hmap_get(&t->vocab, (const char *)(enc + soff[i]), slen[i], -1);
        if (id < 0) return -1;                          /* should not happen for byte-level BPE */
        emit(out, max_out, cnt, (int32_t)id);
    }
    return 0;
}

/* tokenize a raw text range [s,s+len) (no specials) into out. */
static int encode_text(const sp_tokenizer *t, const unsigned char *s, size_t len,
                       uint32_t *cps, unsigned char *enc, uint32_t *soff, uint32_t *slen,
                       unsigned char *key, int32_t *out, int max_out, long *cnt) {
    if (len == 0) return 0;
    /* decode UTF-8 -> codepoints, remembering each codepoint's byte span */
    size_t ncp = 0;
    size_t boff_buf_cap = len + 1;
    uint32_t *boff = (uint32_t *)malloc(boff_buf_cap * sizeof(uint32_t));
    if (!boff) return -1;
    for (size_t i = 0; i < len; ) {
        uint32_t c; size_t adv = cp_decode(s, len, i, &c);
        cps[ncp] = c; boff[ncp] = (uint32_t)i; ncp++;
        i += adv;
    }
    boff[ncp] = (uint32_t)len;   /* sentinel: byte end of last codepoint */
    int rc = 0;
    for (size_t i = 0; i < ncp; ) {
        size_t e = next_piece(cps, ncp, i);             /* [i,e) in codepoints */
        size_t rstart = boff[i], rend = boff[e];        /* -> raw byte span    */
        rc = bpe_piece(t, s + rstart, rend - rstart, enc, soff, slen, key, out, max_out, cnt);
        if (rc) break;
        i = e;
    }
    free(boff);
    return rc;
}

long sp_tokenizer_encode(const sp_tokenizer *t, const char *text, size_t text_len,
                         int parse_special, int32_t *out, int max_out) {
    if (!t || (!text && text_len > 0) || (!out && max_out > 0)) return -1;
    if (text_len == 0) return 0;
    const unsigned char *s = (const unsigned char *)text;

    /* scratch sized to the whole input (codepoints <= bytes; enc <= 2*bytes) */
    uint32_t      *cps  = (uint32_t *)malloc(text_len * sizeof(uint32_t));
    unsigned char *enc  = (unsigned char *)malloc(2 * text_len + 8);
    uint32_t      *soff = (uint32_t *)malloc((text_len + 1) * sizeof(uint32_t));
    uint32_t      *slen = (uint32_t *)malloc((text_len + 1) * sizeof(uint32_t));
    unsigned char *key  = (unsigned char *)malloc(2 * text_len + 16);
    if (!cps || !enc || !soff || !slen || !key) {
        free(cps); free(enc); free(soff); free(slen); free(key); return -1;
    }

    long cnt = 0; int rc = 0;
    size_t tstart = 0, i = 0;
    while (i < text_len) {
        int hit = 0;
        if (parse_special && t->n_spec) {
            for (uint32_t k = 0; k < t->n_spec; k++) {       /* sorted longest-first */
                uint32_t sl = t->spec[k].len;
                if (sl && i + sl <= text_len && memcmp(s + i, t->spec[k].surf, sl) == 0) {
                    if (i > tstart) {
                        rc = encode_text(t, s + tstart, i - tstart, cps, enc, soff, slen, key, out, max_out, &cnt);
                        if (rc) goto done;
                    }
                    emit(out, max_out, &cnt, t->spec[k].id);
                    i += sl; tstart = i; hit = 1; break;
                }
            }
        }
        if (!hit) i++;
    }
    if (tstart < text_len)
        rc = encode_text(t, s + tstart, text_len - tstart, cps, enc, soff, slen, key, out, max_out, &cnt);
done:
    free(cps); free(enc); free(soff); free(slen); free(key);
    return rc ? -1 : cnt;
}
