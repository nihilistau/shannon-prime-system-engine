# sp_tok_dump — token-ID fixture generator for the PPL gates (ETA.5b).
# Engine lane: GGUF parse + the gated-parity SP tokenizer.
add_executable(sp_tok_dump sp_tok_dump.c)
target_link_libraries(sp_tok_dump PRIVATE sp_engine)
