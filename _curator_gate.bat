@echo off
call "D:\F\shannon-prime-repos\shannon-prime-system-engine\scripts\env\env-cuda.bat" >nul 2>&1
set "SP_NIGHTSHIFT_OFFLINE=1"
set "SP_DAEMON_BACKEND=cuda"
set "SP_CUDA_DECODE_INT8=1"
set "SP_KAIROS_MODEL=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-model"
set "SP_KAIROS_TOK=D:/F/shannon-prime-repos/models/gemma4-12b-b1.sp-tokenizer"
set "SP_NIGHTSHIFT_LIVE=D:\F\shannon-prime-repos\shannon-prime-system-engine\_nightshift_test"
set "SP_OKF_MEM=D:\F\shannon-prime-repos\shannon-prime-lattice\tools\okf_mem.py"
set "SP_OKF_ROOT=D:\F\shannon-prime-repos\shannon-prime-lattice\memory-okf"
set "SP_OKF_PY=C:\Users\Knack\AppData\Local\Programs\Python\Python311\python.exe"
cd /d "D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\sp_daemon"
echo GATE_START %DATE% %TIME% > D:\F\_curator_gate.log
"target-wirecuda\release\sp-daemon.exe" >> D:\F\_curator_gate.log 2>&1
echo GATE_DONE_%errorlevel% >> D:\F\_curator_gate.log