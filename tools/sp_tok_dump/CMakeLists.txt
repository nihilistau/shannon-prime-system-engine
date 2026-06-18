# sp_tok_dump — token-ID fixture generator for the PPL gates (ETA.5b).
# Engine lane: GGUF parse + the gated-parity SP tokenizer.
add_executable(sp_tok_dump sp_tok_dump.c)
target_link_libraries(sp_tok_dump PRIVATE sp_engine)

# CONTRACT-CHAT-FULLSTACK B3 — episode-capture token dumper off the .sp-tokenizer
# blob directly (the gemma4 GEMMA4_BPE lane the daemon uses), no GGUF needed.
add_executable(sp_tok_enc sp_tok_enc.c)
target_link_libraries(sp_tok_enc PRIVATE sp_engine)
