/* vk_common.h — internal Vulkan plumbing shared by vulkan_backend.cpp (device
 * query / smoke) and vulkan_forward.cpp (the forward pass). NOT a public header.
 *
 * One process-wide VkInstance + selected VkPhysicalDevice (lazily created), and
 * the VkResult -> SP_EVULKAN / sp_last_error wrapping that honors the frozen L1
 * ABI error surface. Device selection prefers a DISCRETE_GPU that supports
 * shaderInt64 + shaderFloat64 (needed by the rmsnorm f64 accumulation and the
 * NTT-attn int64 dot), as the box has both an NVIDIA RTX 2060 and an Intel iGPU.
 */
#ifndef SP_VK_COMMON_H
#define SP_VK_COMMON_H

#include <vulkan/vulkan.h>
#include <cstdio>

extern "C" void sp_set_error(const char *msg);

/* Wrap a failing VkResult: stash "Vulkan: <where>: VkResult <r>" in the
 * thread-local error string and return 1 (the forward's nonzero error code). */
int vk_fail(VkResult r, const char *where);
#define VKC(call, where) do { VkResult _r = (call); if (_r != VK_SUCCESS) return vk_fail(_r, where); } while (0)

/* Process-wide instance + device handle, lazily created. Returns VK_SUCCESS or
 * stashes detail via sp_set_error and returns the VkResult. */
struct VkContext {
    VkInstance       instance;
    uint32_t         n_phys;            /* number of physical devices enumerated */
    VkPhysicalDevice phys;              /* the SELECTED physical device */
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queue_family;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t         max_wg_count[3];   /* maxComputeWorkGroupCount */
    int              has_int64;
    int              has_float64;
    int              ready;
};

/* Ensure the instance is created and physical devices enumerated (cheap; for the
 * device-count/info smoke). Returns 0 on success, nonzero (sp_last_error set). */
int vk_ensure_instance(void);

/* Ensure a full logical device + queue is created on the selected physical
 * device (for the forward pass). Returns 0 on success. */
int vk_ensure_device(void);

/* The shared context (valid after vk_ensure_instance / vk_ensure_device). */
VkContext *vk_ctx(void);

/* Pick a memory type index with the requested property flags for `type_bits`. */
uint32_t vk_find_mem_type(uint32_t type_bits, VkMemoryPropertyFlags want);

#endif /* SP_VK_COMMON_H */
