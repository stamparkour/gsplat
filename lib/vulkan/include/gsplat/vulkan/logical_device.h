#ifndef GSPLAT_VULKAN_LOGICAL_DEVICE_H
#define GSPLAT_VULKAN_LOGICAL_DEVICE_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <gsplat/vulkan/command_queue.h>
#include <vector>

namespace gsplat::vulkan {
	class logical_device {
		friend class physical_device;
		VkDevice device_v;
		VkInstance instance_ref_v;
        std::vector<command_queue> queues_v;
		logical_device(VkInstance a, VkDevice b, std::vector<command_queue> c) : 
            instance_ref_v(a), 
            device_v(b), 
            queues_v(c) {}
	public:
		~logical_device();

		logical_device(const logical_device&) = delete;
		logical_device(logical_device&&) = delete;
		logical_device& operator =(const logical_device&) = delete;
		logical_device& operator =(logical_device&&) = delete;

		VkDevice device_handle() const;
		VkInstance instance_handle() const;
        const command_queue& queue(std::size_t index) const;
	};
}

#endif // GSPLAT_VULKAN_LOGICAL_DEVICE_H