#include <gsplat/data/database.h>

using namespace gsplat::data;

database database::colmap(const std::string& dir, gsplat::vulkan::command_queue* q) {
	database o{};

	o.gaussians_v = gsplat::data::gaussian_set::colmap_txt(dir + "/sparse/0/points3D.txt", q);
	o.images_v = gsplat::data::image_set::colmap_txt(dir + "/images/", dir + "/sparse/0/images.txt", q, 20);

	return std::move(o);
}