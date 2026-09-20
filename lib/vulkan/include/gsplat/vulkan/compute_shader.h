#ifndef GSPLAT_VULKAN_BUFFER_H
#define GSPLAT_VULKAN_BUFFER_H

#include <gsplat/vulkan/config.h>
#include<vulkan/vulkan.h>
#include <gsplat/vulkan/device.h>
#include <iostream>

namespace gsplat::vulkan {
	class compute_shader {
	public:
		compute_shader(logical_device* device, istream& file);
		~compute_shader();

		void dispatch(int x, int y, int z);
		void async_dispatch(int x, int y, int z);

		void sync_cpu_write();
		void sync_cpu_read();
	};
}

#endif // GSPLAT_VULKAN_BUFFER_H
