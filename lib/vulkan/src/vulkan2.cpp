#include <gsplat/vulkan/vulkan2.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>
#include <limits>

using namespace gsplat::vulkan;

vulkan_context::vulkan_context(const vulkan_context_config& config) {
    uint32_t gsplat_version = VK_MAKE_API_VERSION(
        0, config.major_version,
        config.minor_version,
        config.patch_version
    );
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = config.project_name.c_str(),
        .applicationVersion = gsplat_version,
        .pEngineName = config.project_name.c_str(),
        .engineVersion = gsplat_version,
        .apiVersion = VK_API_VERSION_1_1
    };
    /*std::vector<const char*> exts = {
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };
    std::vector<const char*> layers = {
        
    };*/
    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr
    };
    // Create Vulkan Instance
    if (VkResult r = vkCreateInstance(&ci, nullptr, &vk_instance);
        r != VK_SUCCESS
    ) {
        throw std::runtime_error("Failed to create the Vulkan Instance. VK Code: " + std::to_string(r));
    }

    // Get available physical devices
    uint32_t numDevices = 0;
    if (VkResult r = vkEnumeratePhysicalDevices(vk_instance, &numDevices, nullptr);
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Failed to enumerate physical Vulkan Devices. VK Code: " + std::to_string(r));
    }
    if (numDevices == 0) {
        throw std::runtime_error("Could not find a GPU device with Vulkan support.");
    }

    // Select best physical device
    // will implement scoring system for this
    std::vector<VkPhysicalDevice> devices(numDevices);
    vkEnumeratePhysicalDevices(vk_instance, &numDevices, devices.data());
    vk_phys_dev = devices[0];

    VkPhysicalDeviceProperties phys_dev_props;
    vkGetPhysicalDeviceProperties(physical_device(), &phys_dev_props);
    std::cout << "Selected GPU Device: " << phys_dev_props.deviceName <<  std::endl;

    // Get number of queue family properties
    uint32_t qf_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device(), &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device(), &qf_count, qf_props.data());
    // Select first Compute queue family
    uint32_t qf_index = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qf_index = i;
            break;
        }
    }

    // Device queue info
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qf_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .shaderBufferFloat32AtomicAdd = VK_TRUE
    };
    std::vector<const char*> dev_exts = {
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
    };
    VkDeviceCreateInfo dev_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &atomic_float,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_ci,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = dev_exts.data(),
        .pEnabledFeatures = nullptr
    };
    if (VkResult r = vkCreateDevice(physical_device(), &dev_ci, nullptr, &vk_dev);
        r != VK_SUCCESS
    ) {
        throw std::runtime_error("Could not create a Vulkan Device with Compute Queue Family. VK Code: " + std::to_string(r));
    }
}
vulkan_context::~vulkan_context() {
	// de-init vulkan context here
    if (instance()) {
        vkDestroyInstance(instance(), nullptr);
        vk_instance = nullptr;
    }
}
VkDevice vulkan_context::device() {
    return vk_dev;
}
VkPhysicalDevice vulkan_context::physical_device() {
    return vk_phys_dev;
}
VkInstance vulkan_context::instance() {
    return vk_instance;
}
VkQueue vulkan_context::queue() {

}
VkDescriptorPool vulkan_context::descriptor_pool() {

}
