/* sp_error.c — the frozen L1 ABI thread-local error string (sp_status.h).
 * Lives in a backend-agnostic TU so sp_last_error() is always defined, whether
 * or not the CUDA/Vulkan/Hexagon backends are linked. The backends set the
 * detail via the internal sp_set_error() helper. */
#include "sp_engine/sp_status.h"
#include <string.h>

#if defined(_MSC_VER)
#  define SP_TLS __declspec(thread)
#else
#  define SP_TLS _Thread_local
#endif

static SP_TLS char g_err[512];

const char *sp_last_error(void) { return g_err; }

/* Internal: set the thread-local error detail (truncated to fit). Declared
 * extern by the backends; not part of the public frozen ABI. */
void sp_set_error(const char *msg) {
    if (!msg) { g_err[0] = '\0'; return; }
    size_t n = strlen(msg);
    if (n >= sizeof(g_err)) n = sizeof(g_err) - 1;
    memcpy(g_err, msg, n);
    g_err[n] = '\0';
}
