/* sp_hex_rt.c — Phase 2-HX FastRPC wiring smoke (host = phone aarch64).
 *
 * HX.2 round-trip: open the sp_hex handle on the cDSP (unsigned PD, domain 3),
 * call ping(41) and expect 42 back from the DSP, close. Proves OUR qaic-generated
 * stub + skel + FastRPC transport + unsigned-PD all work end-to-end on the device
 * BEFORE any HVX kernel. Recreated fresh (S22U = structural reference only).
 *
 * Build: NDK aarch64 (scripts/build/build-hexagon.bat android). Run on device:
 *   adb push test_hex_rt + libsp_hex_skel.so to /data/local/tmp/sp22u/lat-hx/
 *   cd .../lat-hx && LD_LIBRARY_PATH=/vendor/lib64:. ADSP_LIBRARY_PATH=. ./test_hex_rt
 */
#include <stdio.h>
#include "sp_hex.h"        /* qaic-generated: sp_hex_open/close/ping + sp_hex_URI */
#include "rpcmem.h"
#include "remote.h"
#include "AEEStdErr.h"

int main(void) {
    const int domain = CDSP_DOMAIN_ID;   /* 3 = CDSP */
    remote_handle64 h = (remote_handle64)-1;
    int nErr, y = -1;

    rpcmem_init();

    /* Run the skel in an unsigned PD on the CDSP (no signing on this device). */
    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module u;
        u.domain = domain;
        u.enable = 1;
        nErr = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&u, sizeof(u));
        if (nErr) printf("sp_hex: unsigned-PD control rc=0x%x (continuing)\n", nErr);
    }

    nErr = sp_hex_open(sp_hex_URI CDSP_DOMAIN, &h);
    if (nErr) {
        printf("sp_hex_open FAILED rc=0x%x\n", nErr);
        rpcmem_deinit();
        return 1;
    }

    nErr = sp_hex_ping(h, 41, &y);
    printf("sp_hex_ping(41) -> %d  (rc=0x%x)\n", y, nErr);

    sp_hex_close(h);
    rpcmem_deinit();

    if (nErr || y != 42) {
        printf("HX.2 ROUND-TRIP FAIL (expected 42)\n");
        return 1;
    }
    printf("HX.2 ROUND-TRIP OK\n");
    return 0;
}
