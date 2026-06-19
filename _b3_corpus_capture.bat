@echo off
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
  if /i not "!NAME!"=="foreign_queries" (
    echo [capture] !NAME!
    "%ENC%" "%TOK%" "%%F" > "%CORPUS%\toks\!NAME!.tok" 2> "%CORPUS%\toks\!NAME!.enc.log"
    set "NT=0"
    for /f %%C in ('type "%CORPUS%\toks\!NAME!.tok" ^| find /c /v ""') do set "NT=%%C"
    echo     nctx=!NT!
    call "%ENG%_b3_capture_ep.bat" "%CORPUS%\toks\!NAME!.tok" !NT! "%CORPUS%\eps\ep_!NAME!" > "%CORPUS%\toks\!NAME!.cap.log" 2>&1
    copy /y "%CORPUS%\toks\!NAME!.tok" "%CORPUS%\eps\ep_!NAME!\ep.tok" >nul
  )
)
python "%ENG%tools\xbar_lsh\patch_npos.py" "%CORPUS%\registry.jsonl" "%CORPUS%\eps"
echo CORPUS_CAPTURE_DONE
endlocal
