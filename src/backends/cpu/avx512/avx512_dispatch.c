#include "sp_engine/avx512.h"
#include <string.h>
#if defined(_MSC_VER)
#  include <intrin.h>
#endif

sp_avx512_caps g_avx512_caps;

void sp_avx512_init(void) {
    memset(&g_avx512_caps, 0, sizeof(g_avx512_caps));
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    /* !! normalises the bitmask returned by __builtin_cpu_supports to 0/1
     * before storing into the 1-bit fields (without !!, a value like 32768
     * truncates to 0 in a :1 field and falsely reports the feature absent). */
    g_avx512_caps.has_avx512f   = (unsigned)!!__builtin_cpu_supports("avx512f");
    g_avx512_caps.has_vnni      = (unsigned)!!__builtin_cpu_supports("avx512vnni");
    g_avx512_caps.has_ifma      = (unsigned)!!__builtin_cpu_supports("avx512ifma");
    g_avx512_caps.has_waitpkg   = (unsigned)!!__builtin_cpu_supports("waitpkg");
    g_avx512_caps.has_vpopcntdq = (unsigned)!!__builtin_cpu_supports("avx512vpopcntdq");
    g_avx512_caps.has_vbmi2     = (unsigned)!!__builtin_cpu_supports("avx512vbmi2");
#elif defined(_MSC_VER)
    int cpuid[4];
    __cpuidex(cpuid, 7, 0);
    g_avx512_caps.has_avx512f   = (cpuid[1] >> 16) & 1;
    g_avx512_caps.has_vnni      = (cpuid[2] >> 11) & 1;
    g_avx512_caps.has_ifma      = (cpuid[1] >> 21) & 1;
    g_avx512_caps.has_vpopcntdq = (cpuid[2] >> 14) & 1;
    g_avx512_caps.has_vbmi2     = (cpuid[2] >> 6) & 1;
    g_avx512_caps.has_waitpkg   = (cpuid[2] >> 5) & 1;
#endif
}
