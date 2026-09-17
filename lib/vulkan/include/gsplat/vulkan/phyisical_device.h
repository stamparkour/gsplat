#ifndef GSPLAT_VULKAN_PHYSICAL_DEVICE_H
#define GSPLAT_VULKAN_PHYSICAL_DEVICE_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>

namespace gsplat::vulkan {
	class phyisical_device {
		friend class vulkan_context;
		VkPhysicalDevice device_v;
		VkInstance instance_ref_v;

		phyisical_device(VkPhysicalDevice a, VkInstance b) : device_v(a), instance_ref_v(b) {}
	public:
		~phyisical_device();

		phyisical_device(const phyisical_device&) = delete;
		phyisical_device(phyisical_device&&) = delete;
		phyisical_device& operator =(const phyisical_device&) = delete;
		phyisical_device& operator =(phyisical_device&&) = delete;

		VkPhysicalDevice device_handle();
		VkInstance instance_handle();
	};
}

#endif // GSPLAT_VULKAN_PHYSICAL_DEVICE_H