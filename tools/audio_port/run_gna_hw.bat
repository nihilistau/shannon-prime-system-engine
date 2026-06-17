@echo off
REM KAI-3 Stage 2.b GNA scoring on the NATIVE WINDOWS host (GNA_HW needs MMIO; WSL can't passthrough).
REM Usage: run_gna_hw.bat GNA_HW       (after BIOS-enable + driver install)
REM        run_gna_hw.bat GNA_SW_EXACT (validation; pure software, no driver needed)
set OVROOT=D:\F\shannon-prime-repos\_xbar\p2b\kai3\ov2023_win\extracted\w_openvino_toolkit_windows_2023.3.0.13775.ceeafaf64f3_x86_64
set MODE=%1
if "%MODE%"=="" set MODE=GNA_HW
set K=D:\F\shannon-prime-repos\_xbar\p2b\kai3
set PY=C:\Users\Knack\AppData\Local\Programs\Python\Python311\python.exe
call "%OVROOT%\setupvars.bat"
"%PY%" "D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\audio_port\ov_score_ir.py" ^
  --frames "%K%\audio_frames.npz" --ir "%K%\ov_work_valid\pot\audio_ctc_pot_gna.xml" ^
  --tag POT-GNA-WIN --mode %MODE%
REM IR = GNA-legal model: VALID convs (no padding) + head padded to 36 filters (mult-of-4). SW_EXACT=0.877.
