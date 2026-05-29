/* sp_echo_imp.c — §3-HX Sprint A DSP-side echo implementation.
 *
 * Pattern source: C:\Qualcomm\Hexagon_IDE\S22U\src_dsp\S22U_imp.c (verified
 * working skel for Knack's S22U via Unsigned PD on 2026-05-29).
 *
 * Required impls per qaic-generated header from sp_echo.idl:
 *   sp_echo_open(uri, handle)  — allocate handle storage; return 0
 *   sp_echo_close(handle)      — free handle storage
 *   sp_echo_ping(handle, ...)  — the echo method
 *
 * The SDK CMake template (hexagon_fun.cmake) handles all the runtime linkage
 * (rtld_init, atomic, libc, qurt, sigverify) automatically.  No manual
 * SigVerify_* stubs needed (vs prior .bat-based approach).
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "HAP_farf.h"
#include "sp_echo.h"

int sp_echo_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    void *tptr = malloc(1);
    *handle = (remote_handle64)tptr;
    assert(*handle);
    return 0;
}

int sp_echo_close(remote_handle64 handle) {
    if (handle) free((void *)handle);
    return 0;
}

int sp_echo_ping(remote_handle64 h, const unsigned char *in_buf, int in_bufLen,
                 unsigned char *out_buf, int out_bufLen)
{
    (void)h;
    int n = (in_bufLen < out_bufLen) ? in_bufLen : out_bufLen;
    for (int i = 0; i < n; i++) {
        out_buf[i] = in_buf[i];
    }
    FARF(RUNTIME_HIGH, "sp_echo: ping echoed %d bytes", n);
    return 0;
}
