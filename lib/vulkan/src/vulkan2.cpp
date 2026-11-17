#include <gsplat/vulkan/vulkan2.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>
#include <functional>
#include <optional>

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
    // Define required extensions
    std::vector<const char*> exts = {
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };
    // Verify extension support
    auto avail_exts = get_available_extensions(nullptr);
    auto ext_cmp = [](VkExtensionProperties e, const char* v) {
        return std::strcmp(e.extensionName, v) == 0;
    };
    vec_contains_vec<VkExtensionProperties,const char*>(avail_exts, exts, ext_cmp, [](auto v) {
        throw std::runtime_error("Could not find required Vulkan extension: " + std::string{v});
    });
    // Define required layers
    std::vector<const char*> layers = {
        "VK_LAYER_KHRONOS_validation"
    };
    // Verify layer support
    auto avail_layers = get_available_layers();
    auto layer_cmp = [](VkLayerProperties l, const char* v) {
        return std::strcmp(l.layerName, v) == 0;
    };
    vec_contains_vec<VkLayerProperties,const char*>(avail_layers, layers, layer_cmp, [](auto v) {
        throw std::runtime_error("Could not find required Vulkan layer: " + std::string{v});
    });

    // Create Vulkan Instance
    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = (uint32_t)layers.size(),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = (uint32_t)exts.size(),
        .ppEnabledExtensionNames = exts.data()
    };
    if (VkResult r = vkCreateInstance(&ci, nullptr, &vk_instance);
        r != VK_SUCCESS
    ) {
        throw std::runtime_error("Failed to create the Vulkan Instance. Vk Code: " + std::to_string(r));
    }


    // Get available physical devices
    uint32_t num_devices = 0;
    if (VkResult r = vkEnumeratePhysicalDevices(instance(), &num_devices, nullptr);
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Failed to enumerate physical Vulkan Devices. Vk Code: " + std::to_string(r));
    }
    if (num_devices == 0) {
        throw std::runtime_error("Could not find a GPU device with Vulkan support.");
    }

    // Select best physical device
    std::vector<VkPhysicalDevice> devices(num_devices);
    if (VkResult r = vkEnumeratePhysicalDevices(instance(), &num_devices, devices.data());
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Failed to enumerate physical Vulkan Devices. Vk Code: " + std::to_string(r));
    }
    VkQueueFlagBits qf_flags = VK_QUEUE_COMPUTE_BIT;
    auto chosen_dev_opt = get_best_physical_device(devices, 100, qf_flags);
    if (!chosen_dev_opt) {
        throw std::runtime_error("Unable to find a physical Vulkan device that meets device requirements");
    }
    vk_phys_dev = *chosen_dev_opt;

    VkPhysicalDeviceProperties phys_dev_props;
    vkGetPhysicalDeviceProperties(physical_device(), &phys_dev_props);
    std::cout << "Selected GPU Device: " << phys_dev_props.deviceName << std::endl;


    // Create Logical Device and Queue
    uint32_t qf_index = get_queue_family_index(physical_device(), qf_flags);
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = qf_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .shaderBufferFloat32AtomicAdd = VK_TRUE
    };
    // Define required device extensions
    std::vector<const char*> dev_exts = {
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
    };
    // Verify device extension support
    auto avail_dev_exts = get_available_device_extensions(physical_device(), nullptr);
    vec_contains_vec<VkExtensionProperties,const char*>(avail_dev_exts, dev_exts, ext_cmp, [](const char* v) {
        throw std::runtime_error("Could not find required Vulkan device extension: " + std::string{v});
    });
    VkPhysicalDeviceFeatures dev_features;
    vkGetPhysicalDeviceFeatures(physical_device(), &dev_features);
    VkDeviceCreateInfo dev_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &atomic_float,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_ci,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = (uint32_t)dev_exts.size(),
        .ppEnabledExtensionNames = dev_exts.data(),
        .pEnabledFeatures = &dev_features
    };
    if (VkResult r = vkCreateDevice(physical_device(), &dev_ci, nullptr, &vk_dev);
        r != VK_SUCCESS
    ) {
        throw std::runtime_error("Could not create the Vulkan Device. Vk Code: " + std::to_string(r));
    }

    // Get and store created device queue
    VkQueue queue;
    vkGetDeviceQueue(device(), qf_index, 0, &queue);
    vk_queue = {
        .handle = queue,
        .family_index = qf_index,
        .index = 0
    };

    global_config = &config;
}

vulkan_context::~vulkan_context() {
	// de-init vulkan context here
    if (device()) {
        vkDestroyDevice(device(), nullptr);
        vk_dev = nullptr;
        vk_queue.handle = VK_NULL_HANDLE;
    }
    if (instance()) {
        vkDestroyInstance(instance(), nullptr);
        vk_instance = nullptr;
    }
    global_config = nullptr;
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
vulkan_queue vulkan_context::queue() {
    return vk_queue;
}
VkDescriptorPool vulkan_context::descriptor_pool() {

}

// Determines best device based on discrete card req, queue types req, and 
// custom device scorers.
// Return null option if a suitable device cannot be found
std::optional<VkPhysicalDevice> gsplat::vulkan::get_best_physical_device(
    std::vector<VkPhysicalDevice> devices,
    int discrete_score,
    VkQueueFlagBits required_queues,
    std::function<int(VkPhysicalDeviceProperties)> prop_scorer,
    std::function<int(VkPhysicalDeviceFeatures)> feature_scorer,
    std::function<int(VkPhysicalDeviceMemoryProperties)> mem_props_scorer
) {
    int best_index = -1;
    int best_score = -1;
    int i = 0;
    for (auto& d : devices) {
        // Skip device if it doesn't have required queue family
        int qf_index = get_queue_family_index(d, required_queues);
        if (qf_index == -1) {
            i++;
            continue;
        }

        int score = 0;
        VkPhysicalDeviceProperties device_props;
        vkGetPhysicalDeviceProperties(d, &device_props);

        if (device_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += discrete_score;
        }

        if (prop_scorer) {
            score += prop_scorer(device_props);
        }
        if (feature_scorer) {
            VkPhysicalDeviceFeatures device_features;
            vkGetPhysicalDeviceFeatures(d, &device_features);
            score += feature_scorer(device_features);
        }
        if (mem_props_scorer) {
            VkPhysicalDeviceMemoryProperties device_mem_props;
            vkGetPhysicalDeviceMemoryProperties(d, &device_mem_props);
            score += mem_props_scorer(device_mem_props);
        }

        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
        i++;
    }

    if (best_index == -1) {
        return std::nullopt;
    }
    return std::optional<VkPhysicalDevice>{ devices[best_index] };
}

// Returns the index of a queue family that supports the required
// queues types for a device
uint32_t gsplat::vulkan::get_queue_family_index(
    VkPhysicalDevice device,
    VkQueueFlagBits required_queues
) {
    // Get number of queue family properties
    uint32_t qf_count;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &qf_count, qf_props.data());
    // Select first Compute queue family
    uint32_t qf_index = -1;
    for (uint32_t i = 0; i < qf_count; i++) {
        if ((qf_props[i].queueFlags & required_queues) == required_queues) {
            qf_index = i;
            break;
        }
    }
    return qf_index;
}

// Gets available Vulkan instance extensions
std::vector<VkExtensionProperties> gsplat::vulkan::get_available_extensions(const char* layer_name) {
    uint32_t ext_count = 0;
    if (VkResult r = vkEnumerateInstanceExtensionProperties(layer_name, &ext_count, nullptr);
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Could not get availabe extensions of Vulkan instance. Vk Code: " + std::to_string(r));
    }
    std::vector<VkExtensionProperties> exts(ext_count);
    if (VkResult r = vkEnumerateInstanceExtensionProperties(layer_name, &ext_count, exts.data());
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Could not get available extensions of Vulkan instance. Vk Code: " + std::to_string(r));
    }
    return exts;
}

// Gets all available Vulkan instance layers
std::vector<VkLayerProperties> gsplat::vulkan::get_available_layers() {
    uint32_t layer_count = 0;
    if (VkResult r = vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Could not get available layers of Vulkan instance. Vk Code: " + std::to_string(r));
    }
    std::vector<VkLayerProperties> layers(layer_count);
    if (VkResult r = vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Could not get available layers of Vulkan instance. Vk Code: " + std::to_string(r));
    }
    return layers;
}

// Gets available Vulkan device extensions
std::vector<VkExtensionProperties> gsplat::vulkan::get_available_device_extensions(
    VkPhysicalDevice device,
    const char* layer_name
) {
    uint32_t ext_count = 0;
    if (VkResult r = vkEnumerateDeviceExtensionProperties(device, layer_name, &ext_count, nullptr);
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Could not get available extensions of Vulkan device. Vk Code: " + std::to_string(r));
    }
    std::vector<VkExtensionProperties> exts(ext_count);
    if (VkResult r = vkEnumerateDeviceExtensionProperties(device, layer_name, &ext_count, exts.data());
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Could not get available extensions of Vulkan device. Vk Code: " + std::to_string(r));
    }
    return exts;
}
