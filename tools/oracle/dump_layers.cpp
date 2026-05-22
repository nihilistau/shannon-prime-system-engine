// dump_layers.cpp — capture ggml's per-layer residual-stream checkpoints during
// a real Qwen3 decode, so the engine's forward.c can be diffed layer-by-layer to
// localize the E_CPU_2 pos>0 discrepancy. Registers a sched eval callback and
// writes the [n_embd, n_tok] tensors named attn_norm-{il}, ffn_inp-{il},
// l_out-{il} (and result_norm) to <outdir>/<name>.bin.
//
//   dump_layers.exe <model.gguf> <outdir> <prompt>
//
// File format per tensor: i32 ne0 ne1 ne2 ne3, then ne0*ne1*ne2*ne3 f32
// (ggml contiguous order: ne0 fastest). Same f32 KV / no-flash settings as
// dump_logits so the comparison is apples-to-apples.
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static std::string g_outdir;

static bool want(const char *name) {
    return std::strncmp(name, "attn_norm-", 10) == 0 ||
           std::strncmp(name, "ffn_inp-",   8) == 0 ||
           std::strncmp(name, "l_out-",     6) == 0 ||
           std::strncmp(name, "Qcur_normed-", 12) == 0 || // post-QK-norm, pre-RoPE Q
           std::strncmp(name, "Kcur_normed-", 12) == 0 || // post-QK-norm, pre-RoPE K
           std::strncmp(name, "Qcur-",      5) == 0 ||   // post-RoPE Q (last write wins)
           std::strncmp(name, "Kcur-",      5) == 0 ||   // post-RoPE K
           std::strncmp(name, "Vcur-",      5) == 0 ||   // V (no RoPE)
           std::strncmp(name, "kqv-",       4) == 0 ||   // attention core output
           // Gemma3 sandwich-norm + GeGLU checkpoints (Task 4 per-layer diff)
           std::strncmp(name, "attn_post_norm-", 15) == 0 || // post-attn norm out
           std::strncmp(name, "sa_out-",        7) == 0 ||   // x + post_attn_norm(attn)
           std::strncmp(name, "ffn_norm-",       9) == 0 ||  // pre-FFN norm out
           std::strncmp(name, "ffn_out-",        8) == 0 ||  // GeGLU intermediate (pre post-norm)
           std::strncmp(name, "ffn_post_norm-", 14) == 0 ||  // post-FFN norm out
           std::strcmp (name, "result_norm") == 0;
}

static bool eval_cb(struct ggml_tensor *t, bool ask, void * /*ud*/) {
    if (ask) return want(t->name);          // observe only the checkpoints
    if (!want(t->name)) return true;
    if (!ggml_is_contiguous(t) || t->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "skip %s (noncontig or non-f32)\n", t->name);
        return true;
    }
    std::string path = g_outdir + "/" + t->name + ".bin";
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return true; }
    int32_t ne[4] = { (int32_t)t->ne[0], (int32_t)t->ne[1], (int32_t)t->ne[2], (int32_t)t->ne[3] };
    std::fwrite(ne, sizeof(int32_t), 4, f);
    std::fwrite(t->data, sizeof(float), (size_t)ggml_nelements(t), f);
    std::fclose(f);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <model.gguf> <outdir> <prompt>\n", argv[0]); return 2; }
    g_outdir = argv[2];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    const char *prompt = argv[3];
    const int plen = (int)std::strlen(prompt);
    std::vector<llama_token> toks(plen + 8);
    int n = llama_tokenize(vocab, prompt, plen, toks.data(), (int)toks.size(), true, true);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt, plen, toks.data(), (int)toks.size(), true, true); }
    if (n <= 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = (uint32_t)n + 8;
    cp.n_batch = (uint32_t)n + 8;
    cp.type_k  = GGML_TYPE_F32;
    cp.type_v  = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = nullptr;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i]; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
    std::fprintf(stderr, "dumped checkpoints for %d tokens to %s\n", n, g_outdir.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
