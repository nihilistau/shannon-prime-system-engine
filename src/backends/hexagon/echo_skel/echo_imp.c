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

/* ── SigVerify stubs ─────────────────────────────────────────────────────
 * rtld_init.a calls into a SigVerify_* API that's normally provided by the
 * cDSP firmware's signature-checking subsystem. On Knack's S22U (retail
 * Android 15) running Unsigned PD via Path B (DSPRPC_CONTROL_UNSIGNED_MODULE),
 * these symbols are NOT exposed by the cDSP loader → dlerror "undefined
 * symbol" at remote_handle_open → AEE_EUNABLETOLOAD (0x80000406).
 *
 * In Unsigned PD the actual signature check is skipped, so stub returns
 * success unconditionally. This is exactly what hl_signnow.cmd's "copy
 * straight through" mode does for unsigned dev builds.
 *
 * If we ever migrate to Signed PD (Path A: testsig installed), these stubs
 * become a security vuln — they MUST be removed and the real SigVerify lib
 * linked in.  Tag a TODO/Phase 14.3.AUTH for that.
 */
int SigVerify_Streamhash_Init(void)                                  { return 0; }
int SigVerify_Streamhash_Stream(const void *data, unsigned int len)  { (void)data; (void)len; return 0; }
int SigVerify_Streamhash_Finalize(void *out)                         { (void)out; return 0; }
int SigVerify_start(void)                                            { return 0; }
int SigVerify_stop(void)                                             { return 0; }
int SigVerify_verifyseg(const void *seg, unsigned int len)           { (void)seg; (void)len; return 0; }

/* _pl_sigverify is a "platform" structure pointer rtld_init uses to find the
 * SigVerify ops. For our Unsigned PD stub, point it at a zero-filled table
 * (rtld won't call into it because the SigVerify_* stubs above intercept first). */
void *_pl_sigverify = (void *)0;

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
