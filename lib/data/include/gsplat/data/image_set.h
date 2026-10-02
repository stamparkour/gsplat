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
		image_set() = default;

		static image_set colmap_txt(const std::string& images_dir, const std::string& images_txt_path, gsplat::vulkan::command_queue*, int max_entries);

		std::vector<image>& data() { return images_v; }
		const std::vector<image>& data() const { return images_v; }
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_H
