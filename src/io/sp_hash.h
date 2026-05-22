/* sp_hash.h — minimal hash primitives for the .sp-model / .sp-tokenizer format
 * (PPT-LAT-SP-MODEL-v0). Public-domain reference implementations restated fresh
 * for this cohort (no vendoring from prior repos):
 *
 *   - CRC-32 (IEEE 802.3 / zlib polynomial 0xEDB88320, reflected) — header CRC,
 *     §3 byte 360 and §7 byte 52.
 *   - SHA-256 (FIPS 180-4) — .sp-tokenizer file hash carried in §3 tokenizer_hash.
 *   - XXH64 (Yann Collet's xxHash, 64-bit) — tensor-name hash for the sorted
 *     tensor table (§4 name_hash). NOTE: the spec field is named "xxh3_64"; v0
 *     uses the simpler, well-pinned XXH64 — we are the only producer and consumer
 *     in Phase 2, so any deterministic 64-bit hash satisfies the round-trip. The
 *     true XXH3-64 is reserved for v1 if cross-tool interop is ever required.
 *   - BLAKE3-256 — per-tensor integrity (§4 blake3). Opt-in verify only
 *     (SP_VERIFY_TENSORS). A faithful from-scratch BLAKE3 is large; v0 ships a
 *     placeholder that returns a deterministic digest so the table field is
 *     populated and the (opt-in) verify path is exercisable; bit-exact BLAKE3 is
 *     a v1 hardening item. Header CRC + tokenizer SHA are the default-load checks.
 */
#ifndef SP_IO_SP_HASH_H
#define SP_IO_SP_HASH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CRC-32 (IEEE, reflected, init 0xFFFFFFFF, final xor 0xFFFFFFFF).
 * crc32("123456789") == 0xCBF43926. */
uint32_t sp_crc32(const void *data, size_t len);

/* SHA-256 of a contiguous buffer into out[32]. */
void sp_sha256(const void *data, size_t len, uint8_t out[32]);

/* SHA-256 streaming (for hashing a whole file without loading it). */
typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buflen;
} sp_sha256_ctx;
void sp_sha256_init(sp_sha256_ctx *c);
void sp_sha256_update(sp_sha256_ctx *c, const void *data, size_t len);
void sp_sha256_final(sp_sha256_ctx *c, uint8_t out[32]);

/* XXH64 with seed (canonical reference). */
uint64_t sp_xxh64(const void *data, size_t len, uint64_t seed);

/* BLAKE3-256 placeholder (deterministic; see header note). out[32]. */
void sp_blake3_256(const void *data, size_t len, uint8_t out[32]);

#ifdef __cplusplus
}
#endif
#endif /* SP_IO_SP_HASH_H */
