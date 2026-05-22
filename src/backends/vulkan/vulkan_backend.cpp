/* vulkan_backend.cpp — Vulkan compute backend bring-up (Phase 2-VK).
 *
 * VK.0: instance + physical-device enumeration, device selection, and the
 * VkResult -> SP_EVULKAN wrapping that honors the frozen L1 ABI error surface
 * (every failing Vulkan call sets sp_last_error()). The forward pass
 * (gemma3_forward_vulkan) + the SPIR-V dispatch land in vulkan_forward.cpp.
 *
 * Mirrors cuda_backend.cu (sp_cuda_device_count / sp_cuda_device_info): the
 * "compute capability" analog reported by sp_vulkan_device_info is the device's
 * Vulkan apiVersion major.minor.
 */
#include "sp_engine/vulkan_backend.h"
#include "vk_common.h"

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <vector>

int vk_fail(VkResult r, const char *where) {
    char b[512];
    std::snprintf(b, sizeof(b), "Vulkan: %s: VkResult %d", where, (int)r);
    sp_set_error(b);
    return 1;
}

static VkContext g_ctx = {};

VkContext *vk_ctx(void) { return &g_ctx; }

/* Score a physical device for selection: prefer discrete GPU + int64/float64. */
static int score_device(const VkPhysicalDeviceProperties &props,
                        const VkPhysicalDeviceFeatures &feat) {
    int s = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   s += 1000;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) s += 100;
    if (feat.shaderInt64)   s += 10;
    if (feat.shaderFloat64) s += 10;
    return s;
}

int vk_ensure_instance(void) {
    if (g_ctx.instance != VK_NULL_HANDLE) return 0;

    VkApplicationInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "shannon-prime-system-engine";
    ai.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;

    VkResult r = vkCreateInstance(&ci, nullptr, &g_ctx.instance);
    if (r != VK_SUCCESS) { g_ctx.instance = VK_NULL_HANDLE; return vk_fail(r, "vkCreateInstance"); }

    uint32_t n = 0;
    r = vkEnumeratePhysicalDevices(g_ctx.instance, &n, nullptr);
    if (r != VK_SUCCESS) return vk_fail(r, "vkEnumeratePhysicalDevices(count)");
    g_ctx.n_phys = n;

    if (n > 0) {
        std::vector<VkPhysicalDevice> devs(n);
        r = vkEnumeratePhysicalDevices(g_ctx.instance, &n, devs.data());
        if (r != VK_SUCCESS) return vk_fail(r, "vkEnumeratePhysicalDevices(list)");
        int best = -1, best_score = -1;
        for (uint32_t i = 0; i < n; i++) {
            VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
            VkPhysicalDeviceFeatures  f; vkGetPhysicalDeviceFeatures(devs[i], &f);
            int sc = score_device(p, f);
            if (sc > best_score) { best_score = sc; best = (int)i; }
        }
        g_ctx.phys = devs[best < 0 ? 0 : best];
    }
    return 0;
}

int vk_ensure_device(void) {
    if (vk_ensure_instance()) return 1;
    if (g_ctx.device != VK_NULL_HANDLE) return 0;
    if (g_ctx.phys == VK_NULL_HANDLE) { sp_set_error("vk_ensure_device: no physical device"); return 1; }

    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(g_ctx.phys, &props);
    VkPhysicalDeviceFeatures  feat;  vkGetPhysicalDeviceFeatures(g_ctx.phys, &feat);
    g_ctx.has_int64   = feat.shaderInt64   ? 1 : 0;
    g_ctx.has_float64 = feat.shaderFloat64 ? 1 : 0;
    for (int i = 0; i < 3; i++) g_ctx.max_wg_count[i] = props.limits.maxComputeWorkGroupCount[i];
    vkGetPhysicalDeviceMemoryProperties(g_ctx.phys, &g_ctx.mem_props);

    /* find a queue family with COMPUTE. */
    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(g_ctx.phys, &nqf, qfs.data());
    int qf = -1;
    for (uint32_t i = 0; i < nqf; i++)
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = (int)i; break; }
    if (qf < 0) { sp_set_error("vk_ensure_device: no compute queue family"); return 1; }
    g_ctx.queue_family = (uint32_t)qf;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = g_ctx.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    /* Enable the precision features the shaders require (f64 accumulation in
     * rmsnorm, int64 dot in attn_ntt). */
    VkPhysicalDeviceFeatures want = {};
    want.shaderInt64   = feat.shaderInt64;
    want.shaderFloat64 = feat.shaderFloat64;

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &want;

    VkResult r = vkCreateDevice(g_ctx.phys, &dci, nullptr, &g_ctx.device);
    if (r != VK_SUCCESS) { g_ctx.device = VK_NULL_HANDLE; return vk_fail(r, "vkCreateDevice"); }
    vkGetDeviceQueue(g_ctx.device, g_ctx.queue_family, 0, &g_ctx.queue);
    g_ctx.ready = 1;
    return 0;
}

uint32_t vk_find_mem_type(uint32_t type_bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_ctx.mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (g_ctx.mem_props.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    return UINT32_MAX;
}

extern "C" int sp_vulkan_device_count(void) {
    if (vk_ensure_instance()) return 0;
    return (int)g_ctx.n_phys;
}

extern "C" sp_status sp_vulkan_device_info(int dev, char *name, int name_cap,
                                           int *sm_major, int *sm_minor) {
    if (vk_ensure_instance()) return SP_EVULKAN;
    uint32_t n = g_ctx.n_phys;
    if (n == 0 || dev < 0 || (uint32_t)dev >= n) {
        sp_set_error("sp_vulkan_device_info: device index out of range");
        return SP_EVULKAN;
    }
    std::vector<VkPhysicalDevice> devs(n);
    VkResult r = vkEnumeratePhysicalDevices(g_ctx.instance, &n, devs.data());
    if (r != VK_SUCCESS) { vk_fail(r, "vkEnumeratePhysicalDevices(info)"); return SP_EVULKAN; }
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(devs[dev], &p);
    if (name && name_cap > 0) {
        std::strncpy(name, p.deviceName, (size_t)name_cap - 1);
        name[name_cap - 1] = '\0';
    }
    /* the CUDA-compute-capability analog: the device's Vulkan apiVersion. */
    if (sm_major) *sm_major = (int)VK_API_VERSION_MAJOR(p.apiVersion);
    if (sm_minor) *sm_minor = (int)VK_API_VERSION_MINOR(p.apiVersion);
    return SP_OK;
}
