#ifndef GSPLAT_DATA_IMAGE_H
#define GSPLAT_DATA_IMAGE_H

#include <gsplat/data/configure.h>
#include <gsplat/vulkan/image.h>
#include <vector>

namespace gsplat::data {
	// Vulkan aligned
	struct pixel {
		char r;
		char g;
		char b;
		char a;
	};

	// should be standardized to a something like 320x240
	class image {
		std::vector<pixel> data_v;
		int width_v;
		int height_v;
		gsplat::vulkan::image image_v;
	public:
		int width() const;
		int height() const;

		gsplat::vulkan::image& image_vulkan();
		const gsplat::vulkan::image& image_vulkan() const;
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_H
