// dump_logits.cpp — stock-llama.cpp logit oracle for the engine's E_CPU_2 gate.
//
// Tokenizes a prompt with the *stock* tokenizer, decodes the whole sequence,
// and writes the token IDs + per-position logits to a binary file the engine
// test consumes. Both sides then run on identical token IDs, so the comparison
// isolates the forward pass.
//
// Built against the pristine upstream llama.cpp at
//   D:\F\shannon-prime-repos\shannon-prime-lattice-llama
// (see build_oracle.sh). Kept in the engine repo so the stock checkout stays
// untouched; this is our validation harness, not engine code.
//
// Output format (little-endian):
//   u32 magic 'SPLG' (0x47474C53? -> see below) | u32 n_tokens | u32 n_vocab
//   i32 token_ids[n_tokens]
//   f32 logits[n_tokens * n_vocab]   (row-major: position-major)
#include "llama.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <out.bin> <prompt>\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *out_path   = argv[2];
    const char *prompt     = argv[3];

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                       // pure CPU reference
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // tokenize (add BOS + parse special tokens)
    const int prompt_len = (int)std::strlen(prompt);
    std::vector<llama_token> toks(prompt_len + 8);
    int n = llama_tokenize(vocab, prompt, prompt_len, toks.data(), (int)toks.size(), true, true);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt, prompt_len, toks.data(), (int)toks.size(), true, true); }
    if (n <= 0) { std::fprintf(stderr, "tokenize failed (%d)\n", n); return 1; }
    toks.resize(n);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = (uint32_t)n + 8;
    cp.n_batch = (uint32_t)n + 8;
    if (argc > 4) { cp.n_threads = std::atoi(argv[4]); cp.n_threads_batch = cp.n_threads; }  // probe f32 summation-order sensitivity
    cp.type_k  = GGML_TYPE_F32;   // full-precision KV cache: apples-to-apples
    cp.type_v  = GGML_TYPE_F32;   // vs the engine's f32 reference path
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1;                 // request logits at every position
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }

    FILE *f = std::fopen(out_path, "wb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", out_path); return 1; }
    uint32_t magic = 0x47474C53u; // "SLGG"
    uint32_t nt = (uint32_t)n, nv = (uint32_t)n_vocab;
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&nt, 4, 1, f);
    std::fwrite(&nv, 4, 1, f);
    for (int i = 0; i < n; i++) { int32_t t = (int32_t)toks[i]; std::fwrite(&t, 4, 1, f); }
    for (int i = 0; i < n; i++) {
        float *lg = llama_get_logits_ith(ctx, i);
        std::fwrite(lg, sizeof(float), (size_t)n_vocab, f);
    }
    std::fclose(f);
    std::fprintf(stderr, "wrote %d tokens x %d vocab logits to %s\n", n, n_vocab, out_path);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
