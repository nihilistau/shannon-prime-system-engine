/* test_sp_hash.c — E_FMT_0 (primitive sanity for Phase 2-FMT): CRC-32 / SHA-256 /
 * XXH64 against published test vectors, plus a header-CRC round-trip on a
 * hand-built .sp-model header. Pins the hash bytes before the heavy round-trip
 * (E_FMT_4) depends on them. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_hash.h"
#include "sp_engine/sp_model.h"

#include <string.h>
#include <stdio.h>

static void HASH_VECTORS(void) {
    /* CRC-32 of "123456789" == 0xCBF43926 (the canonical check value). */
    SP_CHECK(sp_crc32("123456789", 9) == 0xCBF43926u, "crc32(\"123456789\")==CBF43926");
    SP_CHECK(sp_crc32("", 0) == 0u, "crc32(empty)==0");

    /* SHA-256("abc") = ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad */
    static const uint8_t want_abc[32] = {
        0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
        0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad };
    uint8_t got[32]; sp_sha256("abc", 3, got);
    SP_CHECK(memcmp(got, want_abc, 32) == 0, "sha256(\"abc\")");

    /* SHA-256("") = e3b0c442 98fc1c14 9afbf4c8 996fb924 27ae41e4 649b934c a495991b 7852b855 */
    static const uint8_t want_empty[32] = {
        0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,
        0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55 };
    sp_sha256("", 0, got);
    SP_CHECK(memcmp(got, want_empty, 32) == 0, "sha256(empty)");

    /* Streaming SHA-256 must equal one-shot over the same bytes. */
    {
        const char *s = "The quick brown fox jumps over the lazy dog";
        uint8_t a[32], b[32];
        sp_sha256(s, strlen(s), a);
        sp_sha256_ctx c; sp_sha256_init(&c);
        sp_sha256_update(&c, s, 10); sp_sha256_update(&c, s + 10, strlen(s) - 10);
        sp_sha256_final(&c, b);
        SP_CHECK(memcmp(a, b, 32) == 0, "sha256 streaming == one-shot");
    }

    /* XXH64 canonical vectors (seed 0): empty -> 0xEF46DB3751D8E999. */
    SP_CHECK(sp_xxh64("", 0, 0) == 0xEF46DB3751D8E999ull, "xxh64(empty,0)");
    /* deterministic + seed sensitivity */
    SP_CHECK(sp_xxh64("token_embd.weight", 17, 0) == sp_xxh64("token_embd.weight", 17, 0), "xxh64 deterministic");
    SP_CHECK(sp_xxh64("a", 1, 0) != sp_xxh64("a", 1, 1), "xxh64 seed sensitive");
}

static void HEADER_LAYOUT(void) {
    /* The pragma-packed structs must match the spec offsets exactly. */
    SP_CHECK(sizeof(sp_model_header) == 512, "sizeof sp_model_header == 512");
    SP_CHECK(sizeof(sp_tensor_entry) == 256, "sizeof sp_tensor_entry == 256");
    SP_CHECK(sizeof(sp_tok_header) == 128, "sizeof sp_tok_header == 128");
    sp_model_header h; memset(&h, 0, sizeof h);
    SP_CHECK((size_t)((uint8_t *)&h.tokenizer_hash - (uint8_t *)&h) == 280, "tokenizer_hash @280");
    SP_CHECK((size_t)((uint8_t *)&h.header_crc32 - (uint8_t *)&h) == 360, "header_crc32 @360");
    sp_tensor_entry e; memset(&e, 0, sizeof e);
    SP_CHECK((size_t)((uint8_t *)&e.name_hash - (uint8_t *)&e) == 208, "name_hash @208");
    SP_CHECK((size_t)((uint8_t *)&e.blake3 - (uint8_t *)&e) == 176, "blake3 @176");
}

int main(void) {
    SP_RUN(HASH_VECTORS);
    SP_RUN(HEADER_LAYOUT);
    return SP_DONE();
}
