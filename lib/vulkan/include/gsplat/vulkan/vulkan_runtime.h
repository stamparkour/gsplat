#ifndef GSPLAT_VULKAN_VULKAN_RUNTIME_H
#define GSPLAT_VULKAN_VULKAN_RUNTIME_H

#include <gsplat/vulkan/config.h>
#include<vulkan/vulkan.h>
#include <vector>

namespace gsplat::vulkan {
	class vulkan_creation_extensions {
		std::vector<const char*> ext_v;
	public:
		vulkan_creation_extensions() = default;
		vulkan_creation_extensions(const char*const* ext, int count);

		void push_back(const char*);
		void push_back_collection(const char*const* ext, int count);
		const char* const * data() const;
		size_t count() const;
	}
	class vulkan_runtime {
		static vulkan_runtime* global_runtime_v;
		VkInstance instance_v;
		VkPhysicalDevice device_v;
	public:
		vulkan_runtime(const vulkan_creation_extensions& ext = {});
		~vulkan_runtime();

		VkInstance instance() const;
		VkPhysicalDevice primary_device() const;

		// static vulkan_runtime* global();
	};
}

#endif // GSPLAT_VULKAN_VULKAN_RUNTIME_H
