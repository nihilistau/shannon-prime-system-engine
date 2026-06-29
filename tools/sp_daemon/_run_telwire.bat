@echo off
call "D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\sp_daemon\..\..\scripts\env\env-cuda.bat" >nul 2>&1
set SP_TELEPATHY=1
set SP_TELEPATHY_ADAPTER=D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\telepathy\telepathy_adapter_g2q.bin
set SP_TELEPATHY_SRC=D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\telepathy\tele_src_latent.bin
set SP_TELEPATHY_EXPECT=D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\telepathy\tele_expected_map.bin
"D:\F\shannon-prime-repos\shannon-prime-system-engine\tools\sp_daemon\target-wirecuda\release\sp-daemon.exe"
