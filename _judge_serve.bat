@echo off
REM ============================================================================
REM  _judge_serve.bat -- serve gemma4-E2B as the GENERATIVE RECALL JUDGE.
REM
REM  Step 1 of the dual-model recall architecture: prove a small generative model
REM  reads the candidate memory TEXTS and picks the relevant one (open-set,
REM  query-conditioned) where every geometric signal failed. NO transcode -- the
REM  E2B model is already on disk. Serves /v1/chat on port 3001.
REM
REM  Then in another shell:
REM    python tools\xbar_lsh\judge_recall_test.py --paraphrases --host http://127.0.0.1:3001
REM ============================================================================
setlocal
set "ENGINE=%~dp0"
set "DAEMON=%ENGINE%tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
set "MODEL=D:/F/shannon-prime-repos/models/gemma4-e2b.sp-model"
set "TOKENIZER=D:/F/shannon-prime-repos/models/gemma4-e2b.sp-tokenizer"
set "PORT=3001"

call "%ENGINE%scripts\env\env-cuda.bat" >nul 2>&1
set "SP_DAEMON_BACKEND=cuda"
set "SP_DAEMON_KVDECODE=1"
set "SP_CUDA_DECODE_INT8=1"
set "SP_DAEMON_KVDECODE_RING_W=2048"
set "SP_DAEMON_KVDECODE_PMAX=4096"
REM ---- streaming/paging levers ON (the footprint the operator flagged) ----
set "SP_ARENA_RELEASE=1"

cd /d "%ENGINE%tools\sp_daemon"
echo [judge] serving gemma4-E2B at http://127.0.0.1:%PORT%/  (generative recall judge)
echo [judge] open a 2nd shell and run tools\xbar_lsh\judge_recall_test.py once it logs "listening".
"%DAEMON%" start --model "%MODEL%" --tokenizer "%TOKENIZER%" --port %PORT%
endlocal
