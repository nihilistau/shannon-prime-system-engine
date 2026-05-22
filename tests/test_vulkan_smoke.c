/* test_vulkan_smoke.c — VULKAN_SMOKE: the VK.0 toolchain/link gate. Confirms the
 * Vulkan backend lib built and links (glslc SPIR-V + VS2019 host + Vulkan loader),
 * a Vulkan device is visible, and reports its apiVersion (the CUDA compute-cap
 * analog). */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_test.h"
#include "sp_engine/vulkan_backend.h"
#include "sp_engine/sp_status.h"

#include <stdio.h>

static void VULKAN_SMOKE(void) {
    int n = sp_vulkan_device_count();
    fprintf(stderr, "    vulkan device count = %d\n", n);
    SP_CHECK(n >= 1, "at least one Vulkan device visible");
    if (n < 1) { fprintf(stderr, "    sp_last_error: %s\n", sp_last_error()); return; }

    char name[256] = {0};
    int maj = 0, min = 0;
    sp_status st = sp_vulkan_device_info(0, name, (int)sizeof(name), &maj, &min);
    SP_CHECK(st == SP_OK, "query device 0 properties");
    if (st != SP_OK) { fprintf(stderr, "    sp_last_error: %s\n", sp_last_error()); return; }
    fprintf(stderr, "    device 0: %s  vulkan %d.%d\n", name, maj, min);

    /* Vulkan 1.1+ required (we target vulkan1.1 SPIR-V). */
    SP_CHECK(maj > 1 || (maj == 1 && min >= 1), "device supports Vulkan >= 1.1");
}

int main(void) { SP_RUN(VULKAN_SMOKE); return SP_DONE(); }
