/* test_cuda_smoke.c — CUDA_SMOKE: the CU.0 toolchain/link gate. Confirms the
 * CUDA backend lib built and links (nvcc 13.2 + VS2019 host + cuBLAS), a device
 * is visible, and it meets the §8.3 supported-arch floor (compute >= 7.5). */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/cuda_backend.h"
#include "sp_engine/sp_status.h"

#include <stdio.h>

static void CUDA_SMOKE(void) {
    int n = sp_cuda_device_count();
    fprintf(stderr, "    cuda device count = %d\n", n);
    SP_CHECK(n >= 1, "at least one CUDA device visible");
    if (n < 1) { fprintf(stderr, "    sp_last_error: %s\n", sp_last_error()); return; }

    char name[256] = {0};
    int maj = 0, min = 0;
    sp_status st = sp_cuda_device_info(0, name, (int)sizeof(name), &maj, &min);
    SP_CHECK(st == SP_OK, "query device 0 properties");
    if (st != SP_OK) { fprintf(stderr, "    sp_last_error: %s\n", sp_last_error()); return; }
    fprintf(stderr, "    device 0: %s  sm_%d%d\n", name, maj, min);

    /* §8.3: SM 75/86/89 supported, older not. Compute capability >= 7.5. */
    SP_CHECK(maj > 7 || (maj == 7 && min >= 5), "device meets sm_75 arch floor");
}

int main(void) { SP_RUN(CUDA_SMOKE); return SP_DONE(); }
