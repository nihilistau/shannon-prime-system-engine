"""vram_ballast.py <MiB> — hold a cudaMalloc block to vary VRAM pressure for the
G-CUBLAS-PIN gate (N2, G-BX-OBEY-AB follow-on). Ctrl-C / kill to release."""
import ctypes, sys, time

n_mib = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
DLL = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cudart64_12.dll"
cudart = ctypes.WinDLL(DLL)
ptr = ctypes.c_void_p()
rc = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(n_mib * 1024 * 1024))
print(f"ballast: cudaMalloc({n_mib} MiB) rc={rc} ptr={ptr.value:#x}" if rc == 0
      else f"ballast: cudaMalloc FAILED rc={rc}", flush=True)
if rc != 0:
    sys.exit(1)
while True:
    time.sleep(60)
