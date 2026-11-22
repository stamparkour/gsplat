#include <gsplat/data/database.h>

using namespace gsplat::data;

database database::colmap(const std::string& dir) {
	database o{};

	o.gaussians_v = gsplat::data::gaussian_colmap_txt(dir + "/sparse/0/points3D.txt");
	o.images_v = gsplat::data::image_set::colmap_txt(dir + "/images/", dir + "/sparse/0/images.txt", dir + "/sparse/0/cameras.txt", 20);

	return std::move(o);
}
