
#ifndef GSPLAT_VULKAN_VULKAN2_H
#define GSPLAT_VULKAN_VULKAN2_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <stamp/task/coroutine.h>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>

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

    struct vulkan_queue {
        VkQueue handle;
        uint32_t family_index;
        uint32_t index;
    };

    class vulkan_context {
        inline static vulkan_context* global_context = nullptr;
        inline static const vulkan_context_config* global_config = nullptr;
        inline static VkInstance vk_instance = VK_NULL_HANDLE;
        inline static VkPhysicalDevice vk_phys_dev = VK_NULL_HANDLE;
        inline static VkDevice vk_dev = VK_NULL_HANDLE;
        inline static vulkan_queue vk_queue;
        inline static stamp::task::co_simple_stack on_destroy_stack;

    public:
        vulkan_context(const vulkan_context_config&);
        ~vulkan_context();

		static VkDevice device();
		static VkPhysicalDevice physical_device();
		static VkInstance instance();
		static int queue_family_index();
		static int queue_index();
		static VkQueue queue();
		static VkDescriptorPool descriptor_pool();
        static VkCommandPool command_pool();

        static auto co_on_destroy() {
            return on_destroy_stack.push();
        }
    };


    // Contains default device criteria such as graphics and compute queue requirement
    struct device_criteria {
        int discrete_score = 100;
        VkQueueFlags required_queues = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        std::function<int(const VkPhysicalDeviceProperties&)> prop_scorer;
        std::function<int(const VkPhysicalDeviceFeatures&)> feature_scorer;
        std::function<int(const VkPhysicalDeviceMemoryProperties&)> mem_props_scorer;
        
    };

    VkInstance create_instance(VkApplicationInfo*, std::vector<const char*> layers, std::vector<const char*> exts, VkInstanceCreateFlags);
    VkPhysicalDevice create_physical_device(VkInstance instance, const device_criteria& criteria);
    uint32_t find_queue_family_index(VkPhysicalDevice device, VkQueueFlags required_queues);
    std::vector<VkExtensionProperties> get_available_extensions(const char* layer_name);
    std::vector<VkLayerProperties> get_available_layers();
    std::vector<VkExtensionProperties> get_available_device_extensions(
        VkPhysicalDevice device,
        const char* layer_name
    );

    // Utility functions
    template <typename T, typename U, typename Cmp, typename NotFound>
    bool vec_contains_vec(
        std::vector<T> superset,
        std::vector<U> subset,
        Cmp&& comparator,
        NotFound&& not_found_cb
    );
    template <typename T, typename U, typename Cmp>
    bool vec_contains_vec(
        std::vector<T> superset,
        std::vector<U> subset,
        Cmp&& comparator
    );
}

// Return true if "superset" vector is a superset of subset vector
// else returns false and calls not_found_cb with the first not found element
template <typename T, typename U, typename Cmp, typename NotFound>
bool gsplat::vulkan::vec_contains_vec(
    std::vector<T> superset,
    std::vector<U> subset,
    Cmp&& comparator,
    NotFound&& not_found_cb
) {
    for (auto& val : subset) {
        // Checks if val of subset is in superset
        auto contains_comparator = [&val, &comparator](T superset_val) {
            return comparator(superset_val, val);
        };
        if (!std::any_of(superset.begin(), superset.end(), contains_comparator)) {
            // If val of subset if not in superset, return false and call not_found_cb
            not_found_cb(val);
            return false;
        }
    }
    return true;
}
template <typename T, typename U, typename Cmp>
bool gsplat::vulkan::vec_contains_vec(std::vector<T> superset, std::vector<U> subset, Cmp&& comparator) {
    return vec_contains_vec(superset, subset, std::forward(comparator), [](const U&){});
}

#endif // GSPLAT_VULKAN_VULKAN2_H
