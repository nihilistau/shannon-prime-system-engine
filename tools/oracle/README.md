# Logit oracle (stock llama.cpp)

The E_CPU_2 gate compares the engine's Qwen3 forward pass against a **stock**
upstream llama.cpp. The local `llama-cpp-*` checkouts under
`D:\F\shannon-prime-repos\` and `D:\F\` are **contaminated** (they link
`shannon_prime_*` libs — even the one named "cleanroom") and must not be used
as a reference. This oracle uses a pristine upstream clone instead.

## Stock llama.cpp

Cloned at **`D:\F\shannon-prime-repos\shannon-prime-lattice-llama`** (its own git
repo, not part of our repos — do not commit it into ours). Built CPU-only:

```bash
cd /d/F/shannon-prime-repos/shannon-prime-lattice-llama
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=OFF -DGGML_CUDA=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build --target llama -j     # libllama.a + ggml libs
```

## dump_logits

`dump_logits.cpp` links the stock libs (MinGW gcc, statically) and writes the
token IDs + per-position logits for a prompt, so the engine test runs on the
**identical** token IDs (isolating the forward pass from tokenization).

```bash
bash tools/oracle/build_oracle.sh        # -> tools/oracle/dump_logits.exe (self-contained)
tools/oracle/dump_logits.exe <model.gguf> <out.bin> "<prompt>"
```

Output (`out.bin`, little-endian): `u32 magic 0x47474C53 | u32 n_tokens |
u32 n_vocab | i32 token_ids[n_tokens] | f32 logits[n_tokens*n_vocab]`
(position-major). E_CPU_2: the engine reads `token_ids`, runs its own forward
pass, and asserts max-token logit diff ≤ 1e-4 abs / ≤ 0.1% rel.

Verified: `Qwen3-0.6B-f16.gguf` + "The capital of France is" → 5 tokens ×
151936 vocab.

**Non-ASCII prompts:** pass `@path` as the prompt arg to read the prompt bytes
from a UTF-8 file. This is the only reliable way to feed multibyte UTF-8 on
Windows, where `argv` is mangled through the system code page before the program
sees it (it silently corrupts CJK/accented input into mojibake).

## Tokenizer-encode oracle (TOK_ENCODE)

The encode side (`sp_tokenizer_encode`, Qwen2 byte-level BPE) is validated to
reproduce stock-llama IDs byte-for-byte. Supporting tools (Python, throwaway —
kept committed as parity oracles, like the NTT `__int128` reference):

- `gguf_peek.py` — minimal GGUF metadata reader; dumps `tokenizer.ggml.*`
  (pre type, add_bos, merges) and decodes ref IDs back to the prompt string.
- `bpe_proto.py` — reference Python implementation of the full Qwen2 pipeline
  (regex pre-tok → byte-level → ranked BPE → specials). The algorithm oracle:
  the C port must match it. Run `python -X utf8 tools/oracle/bpe_proto.py` to
  re-verify against the probe set (`SP_PROBE_DIR` points at the oracle dumps).
- `gen_unicode_tables.py` — generates `src/tokenizer/unicode_ranges.h`
  (`\p{L}`/`\p{N}`/`\s` ranges) from Python `unicodedata` + the `regex` module.
  Self-contained, not copied from llama.cpp. Re-run if a Unicode boundary drifts.
- `gen_encode_fixture.py` — emits the byte-escaped C parity rows
  (CJK/accented/Devanagari/digits/whitespace/specials) for `test_tokenizer.c`
  from the oracle probe dumps.
