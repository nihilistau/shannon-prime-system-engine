// dump_tokens.cpp — tokenize-only oracle (no forward). Prints the stock
// llama.cpp token IDs for a prompt so the engine's tokenizer can be diffed
// byte-for-byte. Fast (skips model decode); used by the SPM_ENCODE / T_FRO_4
// fixtures the same way dump_logits is used for the forward gates.
//
//   dump_tokens.exe <model.gguf> "<prompt | @prompt_file>" [add_special] [parse_special]
//
// add_special / parse_special default to 1 (matching dump_logits' llama_tokenize
// flags: BOS added, special surfaces parsed). Output: IDs space-separated on
// stdout, "n=<count>" on stderr.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <prompt|@file> [add_special=1] [parse_special=1]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    std::string prompt_buf;
    const char *prompt = argv[2];
    if (argv[2][0] == '@') {
        FILE *pf = std::fopen(argv[2] + 1, "rb");
        if (!pf) { std::fprintf(stderr, "cannot open prompt file %s\n", argv[2] + 1); return 1; }
        std::fseek(pf, 0, SEEK_END); long sz = std::ftell(pf); std::fseek(pf, 0, SEEK_SET);
        prompt_buf.resize(sz > 0 ? (size_t)sz : 0);
        if (sz > 0 && std::fread(&prompt_buf[0], 1, (size_t)sz, pf) != (size_t)sz) {
            std::fprintf(stderr, "short read on prompt file\n"); std::fclose(pf); return 1;
        }
        std::fclose(pf);
        prompt = prompt_buf.c_str();
    }
    const bool add_special   = argc > 3 ? std::atoi(argv[3]) != 0 : true;
    const bool parse_special = argc > 4 ? std::atoi(argv[4]) != 0 : true;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    const int prompt_len = (int)std::strlen(prompt);
    std::vector<llama_token> toks(prompt_len + 16);
    int n = llama_tokenize(vocab, prompt, prompt_len, toks.data(), (int)toks.size(), add_special, parse_special);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt, prompt_len, toks.data(), (int)toks.size(), add_special, parse_special); }
    if (n < 0) { std::fprintf(stderr, "tokenize failed (%d)\n", n); return 1; }
    toks.resize(n);

    for (int i = 0; i < n; i++) std::printf("%s%d", i ? " " : "", (int)toks[i]);
    std::printf("\n");
    std::fprintf(stderr, "n=%d\n", n);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
