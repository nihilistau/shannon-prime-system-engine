@echo off
REM KAI-3 §7.3: dump real gemma-4 token ids for train + eval event text (engine tokenizer, no cloud).
call "%~dp0scripts\env\env-common.bat"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "K=D:\F\shannon-prime-repos\_xbar\p2b\kai3"
set "SP_G4_TOK_DUMP_IN=%K%\train.txt"
set "SP_G4_TOK_DUMP_OUT=%K%\train_tok.txt"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
set "SP_G4_TOK_DUMP_IN=%K%\eval.txt"
set "SP_G4_TOK_DUMP_OUT=%K%\eval_tok.txt"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_cuda.exe"
echo TOKDUMP_DONE
