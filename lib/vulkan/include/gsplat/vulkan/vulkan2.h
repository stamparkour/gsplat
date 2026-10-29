
#ifndef GSPLAT_VULKAN_VULKAN2_H
#define GSPLAT_VULKAN_VULKAN2_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <string>

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

	class vulkan_context {
		inline static vulkan_context* global_context = nullptr;
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
	};
}

#endif // GSPLAT_VULKAN_VULKAN2_H