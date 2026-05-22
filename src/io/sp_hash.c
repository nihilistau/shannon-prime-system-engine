/* sp_hash.c — CRC-32 / SHA-256 / XXH64 / BLAKE3-placeholder. See sp_hash.h.
 * Fresh public-domain restatements; little-endian host (all engine targets). */
#include "sp_hash.h"
#include <string.h>

/* ── CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320) ─────────────────────── */
static uint32_t s_crc_tab[256];
static int      s_crc_ready = 0;
static void crc_build(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        s_crc_tab[n] = c;
    }
    s_crc_ready = 1;
}
uint32_t sp_crc32(const void *data, size_t len) {
    if (!s_crc_ready) crc_build();
    const uint8_t *p = (const uint8_t *)data;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++)
        c = s_crc_tab[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* ── SHA-256 (FIPS 180-4) ────────────────────────────────────────────────── */
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};
#define ROR32(x,n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha256_block(uint32_t st[8], const uint8_t blk[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)blk[i*4] << 24) | ((uint32_t)blk[i*4+1] << 16) |
               ((uint32_t)blk[i*4+2] << 8) | (uint32_t)blk[i*4+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR32(w[i-15],7) ^ ROR32(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROR32(w[i-2],17) ^ ROR32(w[i-2],19)  ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROR32(e,6) ^ ROR32(e,11) ^ ROR32(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = ROR32(a,2) ^ ROR32(a,13) ^ ROR32(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}
void sp_sha256_init(sp_sha256_ctx *c) {
    c->state[0]=0x6a09e667; c->state[1]=0xbb67ae85; c->state[2]=0x3c6ef372; c->state[3]=0xa54ff53a;
    c->state[4]=0x510e527f; c->state[5]=0x9b05688c; c->state[6]=0x1f83d9ab; c->state[7]=0x5be0cd19;
    c->bitlen = 0; c->buflen = 0;
}
void sp_sha256_update(sp_sha256_ctx *c, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    c->bitlen += (uint64_t)len * 8;
    while (len) {
        size_t take = 64 - c->buflen;
        if (take > len) take = len;
        memcpy(c->buf + c->buflen, p, take);
        c->buflen += take; p += take; len -= take;
        if (c->buflen == 64) { sha256_block(c->state, c->buf); c->buflen = 0; }
    }
}
void sp_sha256_final(sp_sha256_ctx *c, uint8_t out[32]) {
    uint64_t bl = c->bitlen;
    uint8_t pad = 0x80;
    sp_sha256_update(c, &pad, 1);
    uint8_t zero = 0;
    while (c->buflen != 56) sp_sha256_update(c, &zero, 1);
    uint8_t len_be[8];
    for (int i = 0; i < 8; i++) len_be[i] = (uint8_t)(bl >> (56 - i*8));
    sp_sha256_update(c, len_be, 8);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(c->state[i] >> 24);
        out[i*4+1] = (uint8_t)(c->state[i] >> 16);
        out[i*4+2] = (uint8_t)(c->state[i] >> 8);
        out[i*4+3] = (uint8_t)(c->state[i]);
    }
}
void sp_sha256(const void *data, size_t len, uint8_t out[32]) {
    sp_sha256_ctx c; sp_sha256_init(&c); sp_sha256_update(&c, data, len); sp_sha256_final(&c, out);
}

/* ── XXH64 (Yann Collet, 64-bit) ─────────────────────────────────────────── */
#define XXP1 0x9E3779B185EBCA87ULL
#define XXP2 0xC2B2AE3D27D4EB4FULL
#define XXP3 0x165667B19E3779F9ULL
#define XXP4 0x85EBCA77C2B2AE63ULL
#define XXP5 0x27D4EB2F165667C5ULL
static uint64_t xx_rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
static uint64_t xx_round(uint64_t acc, uint64_t in) {
    acc += in * XXP2; acc = xx_rotl(acc, 31); acc *= XXP1; return acc;
}
static uint64_t xx_merge(uint64_t acc, uint64_t v) {
    v = xx_round(0, v); acc ^= v; acc = acc * XXP1 + XXP4; return acc;
}
static uint64_t xx_read64(const uint8_t *p) {
    uint64_t v; memcpy(&v, p, 8); return v;   /* LE host */
}
static uint32_t xx_read32(const uint8_t *p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
uint64_t sp_xxh64(const void *data, size_t len, uint64_t seed) {
    const uint8_t *p = (const uint8_t *)data;
    const uint8_t *end = p + len;
    uint64_t h;
    if (len >= 32) {
        const uint8_t *limit = end - 32;
        uint64_t v1 = seed + XXP1 + XXP2, v2 = seed + XXP2, v3 = seed, v4 = seed - XXP1;
        do {
            v1 = xx_round(v1, xx_read64(p)); p += 8;
            v2 = xx_round(v2, xx_read64(p)); p += 8;
            v3 = xx_round(v3, xx_read64(p)); p += 8;
            v4 = xx_round(v4, xx_read64(p)); p += 8;
        } while (p <= limit);
        h = xx_rotl(v1,1) + xx_rotl(v2,7) + xx_rotl(v3,12) + xx_rotl(v4,18);
        h = xx_merge(h, v1); h = xx_merge(h, v2); h = xx_merge(h, v3); h = xx_merge(h, v4);
    } else {
        h = seed + XXP5;
    }
    h += (uint64_t)len;
    while (p + 8 <= end) {
        uint64_t k1 = xx_round(0, xx_read64(p));
        h ^= k1; h = xx_rotl(h, 27) * XXP1 + XXP4; p += 8;
    }
    if (p + 4 <= end) {
        h ^= (uint64_t)xx_read32(p) * XXP1;
        h = xx_rotl(h, 23) * XXP2 + XXP3; p += 4;
    }
    while (p < end) {
        h ^= (uint64_t)(*p) * XXP5;
        h = xx_rotl(h, 11) * XXP1; p++;
    }
    h ^= h >> 33; h *= XXP2; h ^= h >> 29; h *= XXP3; h ^= h >> 32;
    return h;
}

/* ── BLAKE3-256 placeholder ──────────────────────────────────────────────── */
/* v0: a deterministic 32-byte digest so the §4 tensor-table field is populated
 * and an opt-in verify can compare producer vs reader. Built from SHA-256 with a
 * domain tag so it is never confused with a real SHA-256 digest of the same data.
 * A bit-exact BLAKE3 is a v1 hardening item (header CRC + tokenizer SHA are the
 * default-load integrity checks). */
void sp_blake3_256(const void *data, size_t len, uint8_t out[32]) {
    sp_sha256_ctx c; sp_sha256_init(&c);
    static const char tag[16] = "SP-BLAKE3-v0pl\0\0";
    sp_sha256_update(&c, tag, sizeof tag);
    sp_sha256_update(&c, data, len);
    sp_sha256_final(&c, out);
}
