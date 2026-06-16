@echo off
REM KAI-3 §7.3 (GNA EAR) — render the event corpus to 24kHz WAV via the mature voxtral-mini-realtime-rs TTS.
REM Reads kai3/train.txt + kai3/eval.txt (one event/line, from emit_corpus.py), writes kai3/wav/{split}_NNNN.wav.
REM 24kHz out (Mimi codec) -> gen_audio_frames.py resamples 24k->16k for the 40ms/640 EAR framing.
setlocal EnableDelayedExpansion
set "VX=C:\Projects\voxtral-mini-realtime-rs\target\release\voxtral.exe"
set "M=C:\Projects\voxtral-mini-realtime-rs\models\voxtral-tts-q4-gguf\voxtral-tts-q4.gguf"
set "VD=C:\Projects\voxtral-mini-realtime-rs\models\voxtral-tts-q4-gguf\voice_embedding"
set "K=D:\F\shannon-prime-repos\_xbar\p2b\kai3"
set "VOICE=%~1"
if "%VOICE%"=="" set "VOICE=casual_female"
set "MAXTRAIN=%~2"
if "%MAXTRAIN%"=="" set "MAXTRAIN=100000"
if not exist "%K%\wav" mkdir "%K%\wav"
for %%S in (train eval) do (
  set /a i=0
  for /f "usebackq delims=" %%L in ("%K%\%%S.txt") do (
    set "TXT=%%L"
    set "OUT=%K%\wav\%%S_!i!_%VOICE%.wav"
    set "SKIP="
    if "%%S"=="train" if !i! GEQ %MAXTRAIN% set "SKIP=1"
    if not defined SKIP if not exist "!OUT!" "%VX%" speak --gguf "%M%" --voices-dir "%VD%" --voice %VOICE% --euler-steps 3 --text "!TXT!" --output "!OUT!" 1>nul 2>nul
    set /a i+=1
  )
  echo done %%S voice=%VOICE%
)
echo RENDER_CORPUS_DONE voice=%VOICE%
