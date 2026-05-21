/* tokenizer.h — byte-level BPE tokenizer over a GGUF's tokenizer.ggml.* arrays
 * (GPT2/Qwen family). Implements both directions:
 *   DECODE (token IDs -> UTF-8): inverse of GPT2 byte-level coding — each vocab
 *     token is a string over a 256-symbol alphabet that maps printable bytes to
 *     themselves and the rest to codepoints 256.. ; decoding inverts that map.
 *   ENCODE (UTF-8 -> token IDs): optional special-token pre-split, the Qwen2
 *     pre-tokenizer regex split, GPT2 byte-level encode, then greedy ranked BPE
 *     over tokenizer.ggml.merges. Validated to reproduce stock llama.cpp IDs
 *     byte-for-byte (see tools/oracle/bpe_proto.py and the TOK_ENCODE test).
 */
#ifndef SP_ENGINE_TOKENIZER_H
#define SP_ENGINE_TOKENIZER_H

#include "sp_engine/gguf.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sp_tokenizer sp_tokenizer;

/* Load the vocab from a parsed GGUF (reads tokenizer.ggml.tokens). The tokenizer
 * borrows the GGUF mapping (token bytes point into it), so `g` must outlive the
 * tokenizer. Returns NULL if the vocab metadata is absent or on allocation
 * failure. */
sp_tokenizer *sp_tokenizer_load(const gguf_ctx *g);
void          sp_tokenizer_free(sp_tokenizer *t);

uint32_t sp_tokenizer_vocab_size(const sp_tokenizer *t);

/* Decode `n` token IDs to UTF-8 text in `buf` (capacity `cap`, always
 * NUL-terminated when cap>0). Byte-level inverse mapping; IDs out of range are
 * skipped. Returns the number of bytes written (excluding the NUL), or -1 on
 * error. If the output would exceed `cap`, it is truncated (still NUL-terminated)
 * and the full required length is returned (caller can detect truncation). */
long sp_tokenizer_decode(const sp_tokenizer *t, const int32_t *ids, int n,
                         char *buf, size_t cap);

/* Encode `text_len` bytes of UTF-8 into token IDs in `out` (capacity `max_out`).
 * No BOS/EOS is added (Qwen3 sets add_bos_token=0; the caller prepends control
 * tokens if it wants them). If `parse_special` is nonzero, CONTROL/USER_DEFINED
 * token surfaces (e.g. "<|im_start|>") in the text are matched literally
 * (longest-first) and emitted as their own IDs; the gaps are BPE-encoded.
 * Returns the number of tokens produced, or -1 on error. If the count exceeds
 * `max_out` the output is truncated but the full count is still returned (so the
 * caller can resize and retry). */
long sp_tokenizer_encode(const sp_tokenizer *t, const char *text, size_t text_len,
                         int parse_special, int32_t *out, int max_out);

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_TOKENIZER_H */
