@echo off
REM ===========================================================================
REM B3 corpus capture — tokenize + mint an XBAR episode for every needle in the
REM corpus dir, then patch real npos into the registry.  Reuses sp_tok_enc +
REM _b3_capture_ep (zero new code).  %1 = corpus dir (default _needle_corpus).
REM
REM Per *.txt in the corpus dir:
REM   sp_tok_enc <tok> <needle.txt> > <corpus>\toks\<name>.tok
REM   _b3_capture_ep <name>.tok <win> <corpus>\eps\ep_<name>
REM Then patch_npos.py rewrites registry.jsonl npos = wc -l ep.tok.
REM ===========================================================================
setlocal enabledelayedexpansion
call "%~dp0scripts\env\env-common.bat"
set "PATH=%SP_PIN_CUDA_ROOT%\bin;%PATH%"
set "ENG=%~dp0"
if "%~1"=="" ( set "CORPUS=%ENG%_needle_corpus" ) else ( set "CORPUS=%~1" )
set "TOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "ENC=%ENG%build-cuda-vs22\tools\sp_tok_dump\sp_tok_enc.exe"
if not exist "%CORPUS%\toks" mkdir "%CORPUS%\toks"
if not exist "%CORPUS%\eps"  mkdir "%CORPUS%\eps"

for %%F in ("%CORPUS%\*.txt") do (
  set "NAME=%%~nF"
  echo [capture] !NAME!
  "%ENC%" "%TOK%" "%%F" > "%CORPUS%\toks\!NAME!.tok" 2> "%CORPUS%\toks\!NAME!.enc.log"
  REM n_ctx must equal the needle's exact token count: covers the chunk budget
  REM (nt >= n_ctx*chunks) AND captures the full needle incl. the secret tail.
  set "NT=0"
  for /f %%C in ('type "%CORPUS%\toks\!NAME!.tok" ^| find /c /v ""') do set "NT=%%C"
  echo     nctx=!NT!
  call "%ENG%_b3_capture_ep.bat" "%CORPUS%\toks\!NAME!.tok" !NT! "%CORPUS%\eps\ep_!NAME!" > "%CORPUS%\toks\!NAME!.cap.log" 2>&1
  REM ep.tok = the EXACT input token stream (the capture writes k/v/mf only) — the
  REM disposer's token-matcher + npos depend on it; alignment guaranteed by construction.
  copy /y "%CORPUS%\toks\!NAME!.tok" "%CORPUS%\eps\ep_!NAME!\ep.tok" >nul
)

REM patch real npos into registry.jsonl from each ep.tok line count
python "%ENG%tools\xbar_lsh\patch_npos.py" "%CORPUS%\registry.jsonl" "%CORPUS%\eps"
echo CORPUS_CAPTURE_DONE
endlocal
