#ifndef GSPLAT_VULKAN_BUFFER_H
#define GSPLAT_VULKAN_BUFFER_H

#include <gsplat/vulkan/configure.h>

namespace gsplat::vulkan {
	class basic_buffer {
		VkBuffer buffer_v;
	public:
		basic_buffer();
		~basic_buffer();

		// returns RAII lock object
		void memory_lock();

		int size();
	};

	template<typename T>
	class buffer : public basic_buffer {
	public:
		buffer();
		~buffer();

	};
}

#endif // GSPLAT_VULKAN_BUFFER_H
