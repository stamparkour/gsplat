
#ifndef GSPLAT_VULKAN_VULKAN2_H
#define GSPLAT_VULKAN_VULKAN2_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <optional>

/*

requirements:
- RAII for vulkan
- initialize context inside constructor (vulkan_context(config);)
- deinitialize context inside deconstructor (~vulkan_context();)
- implement device and physical_device static functions

you may edit:
- anything inside the 'lib/vulkan' subsystem.
- prioritize edits inside src/vulkan2.cpp and include/gsplat/vulkan/vulkan2.h

*/

namespace gsplat::vulkan {
	struct vulkan_context_config {
		std::string project_name;
		int major_version;
		int minor_version;
		int patch_version;
	};
    std::optional<VkPhysicalDevice> get_best_physical_device(
        std::vector<VkPhysicalDevice> devices,
        int discrete_score,
        VkQueueFlagBits required_queues,
        std::function<int(VkPhysicalDeviceProperties)> prop_scorer = {},
        std::function<int(VkPhysicalDeviceFeatures)> feature_scorer = {},
        std::function<int(VkPhysicalDeviceMemoryProperties)> mem_props_scorer = {}
    );
    uint32_t get_queue_family_index(
        VkPhysicalDevice device,
        VkQueueFlagBits required_queues
    );
    std::vector<VkExtensionProperties> get_available_extensions(const char* layer_name);
    std::vector<VkLayerProperties> get_available_layers();
    std::vector<VkExtensionProperties> get_available_device_extensions(
        VkPhysicalDevice device,
        const char* layer_name
    );

    // Utility functions
    template <typename T>
    bool vec_contains(std::vector<T> vec, std::function<bool(T)> comparator);
    template <typename T, typename U>
    bool vec_contains_vec(
        std::vector<T> superset,
        std::vector<U> subset,
        std::function<bool(T,U)> comparator,
        std::function<void(U)> not_found_cb = {}
    );

    // Can convert to a class if additional functionality is needed
    struct vulkan_queue {
        VkQueue handle;
        uint32_t family_index;
        uint32_t index;
    };

    class vulkan_context {
        inline static vulkan_context* global_context = nullptr;
        inline static const vulkan_context_config* global_config = nullptr;
        inline static VkInstance vk_instance;
        inline static VkPhysicalDevice vk_phys_dev;
        inline static VkDevice vk_dev;
        inline static vulkan_queue vk_queue;
    public:
        vulkan_context(const vulkan_context_config&);
        ~vulkan_context();

        static VkDevice device();
        static VkPhysicalDevice physical_device();
        static VkInstance instance();
        static vulkan_queue queue();
        static VkDescriptorPool descriptor_pool();
    };
}

template <typename T>
bool gsplat::vulkan::vec_contains(std::vector<T> vec, std::function<bool(T)> comparator) {
    for (auto& v : vec) {
        if (comparator(v)) {
            return true;
        }
    }
    return false;
}

// Return true if "superset" vector is a superset of subset vector
// else returns false and calls not_found_cb with the first not found element
template <typename T, typename U>
bool gsplat::vulkan::vec_contains_vec(
    std::vector<T> superset,
    std::vector<U> subset,
    std::function<bool(T,U)> comparator,
    std::function<void(U)> not_found_cb
) {
    for (auto& val : subset) {
        // Checks if val of subset is in superset
        auto contains_comparator = [&val, &comparator](T superset_val) {
            return comparator(superset_val, val);
        };
        if (!vec_contains<T>(superset, contains_comparator)) {
            // If val of subset if not in superset, return false and call not_found_cb
            if (not_found_cb)
                not_found_cb(val);
            return false;
        }
    }
    return true;
}

#endif // GSPLAT_VULKAN_VULKAN2_H
