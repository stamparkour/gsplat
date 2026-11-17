#include <gsplat/vulkan/vulkan2.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>
#include <functional>

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
    // Define required layers
    std::vector<const char*> layers = {
        "VK_LAYER_KHRONOS_validation"
    };
    // Define required extensions
    std::vector<const char*> exts = {
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };
    instance_v = create_instance(app_info, layers, exts, VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR);

    device_criteria criteria{};
    phys_dev_v = create_physical_device(instance(), criteria);

    // Print name of selected physical device
    VkPhysicalDeviceProperties phys_dev_props;
    vkGetPhysicalDeviceProperties(physical_device(), &phys_dev_props);
    std::cout << "Selected GPU Device: " << phys_dev_props.deviceName << std::endl;

    // Create Logical Device and Queue
    uint32_t qf_index = find_queue_family_index(physical_device(), criteria.required_queues);
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
    auto ext_cmp = [](VkExtensionProperties e, const char* v) {
        return std::strcmp(e.extensionName, v) == 0;
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
    if (VkResult r = vkCreateDevice(physical_device(), &dev_ci, nullptr, &dev_v);
        r != VK_SUCCESS
    ) {
        throw std::runtime_error("Could not create the Vulkan Device. Vk Code: " + std::to_string(r));
    }

    // Get and store created device queue
    VkQueue queue;
    vkGetDeviceQueue(device(), qf_index, 0, &queue);
    queue_v = {
        .handle = queue,
        .family_index = qf_index,
        .index = 0
    };

    // Create Descriptor Pool (Requires shader descriptor counts and sets which is unknown so a placeholder value is used)
    VkDescriptorPoolSize desc_pool_size_ci = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        // Total number of descriptors used by shaders is unknown so 64 is used as placeholder
        .descriptorCount = 64
    };
    VkDescriptorPoolCreateInfo desc_pool_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, // Previous vulkan_context had incorrect update after bind bit
        // Each shader needs a set if using buffers. Since the number of shaders is unknown 16 is used as placeholder
        .maxSets = 16,
        .poolSizeCount = 1,
        .pPoolSizes = &desc_pool_size_ci
    };
    if (VkResult r = vkCreateDescriptorPool(device(), &desc_pool_ci, nullptr, &desc_pool_v);
        r != VK_SUCCESS
    ) {
        throw std::runtime_error("Could not create descriptor pool for Vulkan Device. Vk Code: " + std::to_string(r));
    }

    // Create Command Pool with short-lived cmd buffers flag
    VkCommandPoolCreateInfo cmd_pool_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = queue_family_index()
    };
    if (VkResult r = vkCreateCommandPool(device(), &cmd_pool_ci, nullptr, &cmd_pool_v);
        r != VK_SUCCESS
    ) {
        throw std::runtime_error("Could not create command pool for Vulkan Device. Vk Code: " + std::to_string(r));
    }


    global_config = &config;
}

vulkan_context::~vulkan_context() {
    if (VkResult r = vkDeviceWaitIdle(device()); r != VK_SUCCESS) {
        std::cerr << "Encountered error while waiting for Vulkan Device to become idle. Vk Code: " << r << std::endl;
    }
    while (on_destroy_stack.pop_all_unsafe());
    if (command_pool()) {
        vkDestroyCommandPool(device(), command_pool(), nullptr);
        cmd_pool_v = VK_NULL_HANDLE;
    }
    if (descriptor_pool()) {
        vkDestroyDescriptorPool(device(), descriptor_pool(), nullptr);
        desc_pool_v = VK_NULL_HANDLE;
    }
    if (device()) {
        vkDestroyDevice(device(), nullptr);
        dev_v = VK_NULL_HANDLE;
        queue_v.handle = VK_NULL_HANDLE;
    }
    if (instance()) {
        vkDestroyInstance(instance(), nullptr);
        instance_v = VK_NULL_HANDLE;
    }
    global_config = nullptr;
}
template <typename T>
static T uninit_check(T handle) {
    if (handle == VK_NULL_HANDLE) {
        throw std::runtime_error("Attempted to access Vulkan Context before initialization");
    }
    return handle;
}
VkDevice vulkan_context::device() {
    return uninit_check<VkDevice>(dev_v);
}
VkPhysicalDevice vulkan_context::physical_device() {
    return uninit_check<VkPhysicalDevice>(phys_dev_v);
}
VkInstance vulkan_context::instance() {
    return uninit_check<VkInstance>(instance_v);
}
uint32_t vulkan_context::queue_family_index() {
    vulkan_queue q = queue_v;
    uninit_check<VkQueue>(q.handle);
    return q.family_index;
}
uint32_t vulkan_context::queue_index() {
    vulkan_queue q = queue_v;
    uninit_check<VkQueue>(q.handle);
    return q.index;
}
VkQueue vulkan_context::queue() {
    vulkan_queue q = queue_v;
    uninit_check<VkQueue>(q.handle);
    return q.handle;
}
VkDescriptorPool vulkan_context::descriptor_pool() {
    return uninit_check<VkDescriptorPool>(desc_pool_v);
}
VkCommandPool vulkan_context::command_pool() {
    return uninit_check<VkCommandPool>(cmd_pool_v);
}

VkInstance gsplat::vulkan::create_instance(
    const VkApplicationInfo& app_info,
    std::vector<const char*> layers,
    std::vector<const char*> exts,
    VkInstanceCreateFlags flags
) {
    // Verify layer support
    auto avail_layers = get_available_layers();
    auto layer_cmp = [](VkLayerProperties l, const char* v) {
        return std::strcmp(l.layerName, v) == 0;
    };
    vec_contains_vec<VkLayerProperties,const char*>(avail_layers, layers, layer_cmp, [](auto v) {
        throw std::runtime_error("Could not find required Vulkan layer: " + std::string{v});
    });
    // Verify extension support
    auto avail_exts = get_available_extensions(nullptr);
    auto ext_cmp = [](VkExtensionProperties e, const char* v) {
        return std::strcmp(e.extensionName, v) == 0;
    };
    vec_contains_vec<VkExtensionProperties,const char*>(avail_exts, exts, ext_cmp, [](auto v) {
        throw std::runtime_error("Could not find required Vulkan extension: " + std::string{v});
    });

    VkInstance instance;
    // Create Vulkan Instance
    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = (uint32_t)layers.size(),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = (uint32_t)exts.size(),
        .ppEnabledExtensionNames = exts.data()
    };
    if (VkResult r = vkCreateInstance(&ci, nullptr, &instance);
        r != VK_SUCCESS
    ) {
        throw std::runtime_error("Failed to create the Vulkan Instance. Vk Code: " + std::to_string(r));
    }
    return instance;
}

// Determines best device based on discrete card req, queue types req, and 
// custom device scorers.
// throws error if a suitable device cannot be found
VkPhysicalDevice gsplat::vulkan::create_physical_device(
    VkInstance instance,
    const device_criteria& c
) {
    // Get available physical devices
    uint32_t num_devices = 0;
    if (VkResult r = vkEnumeratePhysicalDevices(instance, &num_devices, nullptr);
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Failed to enumerate physical Vulkan Devices. Vk Code: " + std::to_string(r));
    }
    if (num_devices == 0) {
        throw std::runtime_error("Could not find a GPU device with Vulkan support.");
    }

    std::vector<VkPhysicalDevice> devices(num_devices);
    if (VkResult r = vkEnumeratePhysicalDevices(instance, &num_devices, devices.data());
        r < VK_SUCCESS
    ) {
        throw std::runtime_error("Failed to enumerate physical Vulkan Devices. Vk Code: " + std::to_string(r));
    }

    int best_index = -1;
    int best_score = -1;
    int i = 0;
    for (auto& d : devices) {
        // Skip device if it doesn't have required queue family
        int qf_index = find_queue_family_index(d, c.required_queues);
        if (qf_index == -1) {
            i++;
            continue;
        }

        int score = 0;
        VkPhysicalDeviceProperties device_props;
        vkGetPhysicalDeviceProperties(d, &device_props);

        if (device_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += c.discrete_score;
        }

        if (c.prop_scorer) {
            score += c.prop_scorer(device_props);
        }
        if (c.feature_scorer) {
            VkPhysicalDeviceFeatures device_features;
            vkGetPhysicalDeviceFeatures(d, &device_features);
            score += c.feature_scorer(device_features);
        }
        if (c.mem_props_scorer) {
            VkPhysicalDeviceMemoryProperties device_mem_props;
            vkGetPhysicalDeviceMemoryProperties(d, &device_mem_props);
            score += c.mem_props_scorer(device_mem_props);
        }

        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
        i++;
    }

    if (best_index == -1) {
        throw std::runtime_error("Unable to find a physical Vulkan device that meets device requirements");
    }
    return devices[best_index];
}

// Returns the index of a queue family that supports the required
// queues types for a device
uint32_t gsplat::vulkan::find_queue_family_index(
    VkPhysicalDevice device,
    VkQueueFlags required_queues
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
