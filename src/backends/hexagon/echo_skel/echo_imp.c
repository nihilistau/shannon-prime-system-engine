/* §3-HX Sprint A echo skel — DSP-side implementation.
 *
 * Linked into libshannonprime_echo_skel.so via hexagon-clang for V69 cDSP.
 * The qaic-generated echo_skel.c calls echo_ping with the unmarshalled args;
 * we copy in→out and report bytes written.
 *
 * NO HVX, NO VTCM, NO async. Sprint A is signing-and-handshake only.
 *
 * Pattern source: SDK examples/common/<iface>_skel templates (re-derived
 * for the lattice's anti-contamination rule per session prompt §11.3).
 */
#include "AEEStdDef.h"
#include "AEEStdErr.h"

/* echo.h is qaic-generated; the signature here must match what qaic emits.
 * `rout sequence<octet> out_buf` becomes (unsigned char *out, int out_len,
 * int *out_lenWritten). */
#include "echo.h"

int echo_ping(const unsigned char *in_buf, int in_bufLen,
              unsigned char       *out_buf, int out_bufLen)
{
    if (in_buf == 0 || out_buf == 0) {
        return AEE_EBADPARM;
    }
    int n = (in_bufLen < out_bufLen) ? in_bufLen : out_bufLen;
    for (int i = 0; i < n; i++) {
        out_buf[i] = in_buf[i];
    }
    return AEE_SUCCESS;
}
