#ifndef GSPLAT_DATA_IMAGE_SET_H
#define GSPLAT_DATA_IMAGE_SET_H

#include <gsplat/data/configure.h>
#include<gsplat/data/image.h>
#include <string>
#include <vector>

namespace gsplat::data {
	class image_set {
		std::vector<image> images_v;
	public:
		image_set();

		static image_set colmap_txt(const std::string& colmap_dir, gsplat::vulkan::command_queue*);

		std::vector<image>& data() { return images_v; }
		const std::vector<image>& data() const { return images_v; }
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_H
