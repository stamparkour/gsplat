#ifndef GSPLAT_VULKAN_COMMAND_QUEUE_H
#define GSPLAT_VULKAN_COMMAND_QUEUE_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>

namespace gsplat::vulkan {
	class command_queue {
		friend class logical_device;

		command_queue() = default;
	public:
		~command_queue();
	};
}

#endif // GSPLAT_VULKAN_COMMAND_QUEUE_H