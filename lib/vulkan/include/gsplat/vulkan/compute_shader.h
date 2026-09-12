#ifndef GSPLAT_VULKAN_BUFFER_H
#define GSPLAT_VULKAN_BUFFER_H

#include <gsplat/vulkan/configure.h>

namespace gsplat::vulkan {
	class compute_shader {
	public:
		compute_shader();
		~compute_shader();

		void dispatch(int x, int y, int z);
		void async_dispatch(int x, int y, int z);

		void sync_cpu_write();
		void sync_cpu_read();
	};
}

#endif // GSPLAT_VULKAN_BUFFER_H
