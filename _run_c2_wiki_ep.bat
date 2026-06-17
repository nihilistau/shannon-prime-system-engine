@echo off
call "D:\F\shannon-prime-repos\shannon-prime-system-engine\scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "SP_GEMMA4_SPMODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_GEMMA4_SPTOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_PPL_TOKENS=%SP_ENGINE%\tests\fixtures\ppl\wiki.tiny.g4tokens.txt"
set "SP_PPL_NCTX=84"
set "SP_PPL_CHUNKS=1"
set "SP_XBAR_RECALL_WRITE=%SP_ENGINE%\_c2_ep_wiki"
if not exist "%SP_XBAR_RECALL_WRITE%" mkdir "%SP_XBAR_RECALL_WRITE%"
"%SP_ENGINE%\build-cuda-vs22\tests\test_gemma4_ppl_cuda.exe"
echo WIKI_WRITE_EXIT=%ERRORLEVEL%
