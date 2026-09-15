#ifndef GSPLAT_VULKAN_IMAGE_H
#define GSPLAT_VULKAN_IMAGE_H

#include <gsplat/vulkan/config.h>
#include<vulkan/vulkan.h>

namespace gsplat::vulkan {
	class image {
	public:
		image();
		~image();

		int width();
		int height();
	};
}

#endif // GSPLAT_VULKAN_IMAGE_H
