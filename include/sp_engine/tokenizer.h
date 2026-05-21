/* tokenizer.h — byte-level BPE tokenizer over a GGUF's tokenizer.ggml.* arrays
 * (GPT2/Qwen family). This pass implements the DECODE side (token IDs -> UTF-8
 * text) plus vocab loading; ENCODE (text -> IDs via the pre-tokenizer regex +
 * ranked BPE merges) lands next. Decode is the inverse of GPT2 byte-level coding:
 * each vocab token is a string over a 256-symbol alphabet that maps printable
 * bytes to themselves and the rest (control/space/etc.) to codepoints 256.. ;
 * decoding maps those codepoints back to the original bytes.
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

#ifdef __cplusplus
}
#endif
#endif /* SP_ENGINE_TOKENIZER_H */
