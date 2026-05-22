/* rpcmem_probe.c — Phase 2-HX HX.2 capacity probe (NOT part of the engine lib).
 *
 * Standalone aarch64-android binary: allocates progressively larger SYSTEM-heap
 * rpcmem buffers and reports the largest that succeeds. This decides the
 * Hexagon weight-upload architecture BEFORE any IDL is written:
 *
 *   - Gemma3-1B Q8 Frobenius arena is ~574.5 MB (roadmap §8.2). If a single
 *     SYSTEM-heap rpcmem region of that size succeeds, weights can be uploaded
 *     ONCE into one DSP-visible buffer (RPCMEM_TRY_MAP_STATIC) and matmul IDL
 *     takes a handle/offset — the CUDA build_weights-cache analog.
 *   - If it caps below ~574 MB, the upload must partition (per-layer or
 *     per-tensor rpcmem buffers), which reshapes the matmul IDL signatures.
 *
 * No DSP/IDL involved — pure rpcmem + libcdsprpc. EXACT-ALLOC note: rpcmem here
 * is only the HOST allocation; the FastRPC exact-alloc rule (registration size
 * == IDL length) bites later, at dispatch. This probe just sizes the heap.
 *
 * Build (host stub side, aarch64-android): see scripts/build/build-hexagon.bat
 * `probe` target / the HX.2 CMake. Link rpcmem.a + libcdsprpc (-ldl on device).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "rpcmem.h"

int main(int argc, char **argv) {
    rpcmem_init();

    /* Probe sizes in MB. 574.5 MB Q8 arena is the load-bearing threshold; also
     * probe past it to characterise the ceiling. */
    const int mb_sizes[] = { 64, 128, 256, 512, 575, 600, 768, 1024, 1536, 2000 };
    const int n = (int)(sizeof(mb_sizes) / sizeof(mb_sizes[0]));
    (void)argc; (void)argv;

    size_t max_ok = 0;
    for (int i = 0; i < n; i++) {
        size_t bytes = (size_t)mb_sizes[i] * 1024u * 1024u;
        /* rpcmem_alloc takes int size (capped <2GB); we stay under for the
         * arena. SYSTEM heap (25) is SMMU-routed, the working-setup choice. */
        void *p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)bytes);
        if (p) {
            /* touch first + last page so the mapping is real, not lazy. */
            ((volatile char *)p)[0] = 1;
            ((volatile char *)p)[bytes - 1] = 1;
            printf("[rpcmem_probe] %5d MB  OK   (ptr=%p)\n", mb_sizes[i], p);
            if (bytes > max_ok) max_ok = bytes;
            rpcmem_free(p);
        } else {
            printf("[rpcmem_probe] %5d MB  FAIL (rpcmem_alloc returned NULL)\n", mb_sizes[i]);
        }
    }

    printf("[rpcmem_probe] max single SYSTEM-heap alloc that succeeded: %zu MB\n",
           max_ok / (1024u * 1024u));
    printf("[rpcmem_probe] Q8 arena (574.5 MB) single-buffer upload: %s\n",
           max_ok >= (size_t)575 * 1024u * 1024u ? "FEASIBLE" : "NEEDS PARTITION");

    rpcmem_deinit();
    return 0;
}
