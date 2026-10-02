#ifndef GSPLAT_DATA_DATABASE_H
#define GSPLAT_DATA_DATABASE_H

#include <gsplat/data/configure.h>
#include <vector>
#include <string>
#include <gsplat/data/gaussian_set.h>
#include <gsplat/data/image_set.h>

namespace gsplat::data {

	class database {
		gsplat::data::gaussian_set gaussians_v;
		gsplat::data::image_set images_v;
	public:
		database() = default;
		static database colmap(const std::string& dir, gsplat::vulkan::command_queue*);
		gsplat::data::gaussian_set& gaussian_set() { return gaussians_v; }
		const gsplat::data::gaussian_set& gaussian_set() const { return gaussians_v; }
		gsplat::data::image_set& image_set() { return images_v; }
		const gsplat::data::image_set& image_set() const { return images_v; }
	};

	database parse_colmaps_txt(const std::string& path_dir);
}

#endif // GSPLAT_DATA_DATABASE_H
