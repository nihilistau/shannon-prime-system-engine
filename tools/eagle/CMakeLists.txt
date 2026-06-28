# sp_eagle_fwd — engine C reference of the gemma4-assistant (EAGLE/MTP) draft forward
# (step 2c). Standalone CLI; links the engine for GGUF parse + dequant. Gated against
# the numpy oracle (sp_eagle_ref.py) via the dumped fixture (G-EAGLE-DRAFT-FWD-C).
add_executable(sp_eagle_fwd sp_eagle_fwd.c)
target_link_libraries(sp_eagle_fwd PRIVATE sp_engine)
if(MSVC)
    target_compile_options(sp_eagle_fwd PRIVATE /W4)
else()
    target_compile_options(sp_eagle_fwd PRIVATE -Wall -Wextra)
endif()
