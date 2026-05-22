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
#include <stdint.h>
#include "sp_hex.h"        /* qaic-generated: sp_hex_open/close/ping + sp_hex_URI */
#include "rpcmem.h"
#include "remote.h"
#include "AEEStdErr.h"

/* Same reflected CRC-32 (0xEDB88320) the DSP uses — equal iff bytes are equal. */
static unsigned int sp_hex_crc32(const unsigned char *p, int n) {
    unsigned int crc = 0xFFFFFFFFu;
    for (int i = 0; i < n; i++) {
        crc ^= (unsigned int)p[i];
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0xEDB88320u & (unsigned int)(-(int)(crc & 1u)));
    }
    return crc ^ 0xFFFFFFFFu;
}

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
    if (nErr || y != 42) {
        printf("HX.2 ROUND-TRIP FAIL (expected 42)\n");
        sp_hex_close(h); rpcmem_deinit();
        return 1;
    }
    printf("HX.2 ROUND-TRIP OK\n");

    /* ── HX.3a: per-tensor rpcmem upload, byte-exact verify (exact-alloc proof) ──
     * A 32 MB buffer (realistic per-tensor size) on the SYSTEM heap, filled with a
     * deterministic pattern, uploaded via the IDL sequence; the DSP CRCs what it
     * received. Equal CRC => the rpcmem transfer is byte-exact. */
    {
        const int len = 32 * 1024 * 1024;
        unsigned char *buf = (unsigned char *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                           RPCMEM_DEFAULT_FLAGS, len);
        if (!buf) {
            printf("HX.3a: rpcmem_alloc(%d) FAILED\n", len);
            sp_hex_close(h); rpcmem_deinit();
            return 1;
        }
        for (int i = 0; i < len; i++) buf[i] = (unsigned char)((i * 31 + 7) & 0xFF);
        unsigned int host_crc = sp_hex_crc32(buf, len);
        int dev_crc = 0;
        nErr = sp_hex_upload_crc(h, buf, len, &dev_crc);
        printf("sp_hex_upload_crc(%d B): host=0x%08x dev=0x%08x (rc=0x%x)\n",
               len, host_crc, (unsigned int)dev_crc, nErr);
        rpcmem_free(buf);
        if (nErr || (unsigned int)dev_crc != host_crc) {
            printf("HX.3a UPLOAD BYTE-EXACT FAIL (rpcmem transfer corrupt / exact-alloc trap)\n");
            sp_hex_close(h); rpcmem_deinit();
            return 1;
        }
        printf("HX.3a UPLOAD BYTE-EXACT OK (32 MB rpcmem transfer verified)\n");
    }

    sp_hex_close(h);
    rpcmem_deinit();
    return 0;
}
