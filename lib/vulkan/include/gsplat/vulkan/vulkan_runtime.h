#ifndef GSPLAT_VULKAN_VULKAN_RUNTIME_H
#define GSPLAT_VULKAN_VULKAN_RUNTIME_H

#include <gsplat/vulkan/configure.h>

namespace gsplat::vulkan {
	class vulkan_runtime {
		static vulkan_runtime* global_runtime_v;
		VkInstance instance_v;
		VkPhysicalDevice device_v;
	public:
		vulkan_runtime(const char** extensions = nullptr);
		~vulkan_runtime();

		VkInstance instance() const;
		VkPhysicalDevice primary_device() const;

		static vulkan_runtime* global();
	};
}

#endif // GSPLAT_VULKAN_VULKAN_RUNTIME_H
