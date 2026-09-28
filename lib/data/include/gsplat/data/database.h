#ifndef GSPLAT_DATA_DATABASE_H
#define GSPLAT_DATA_DATABASE_H

#include <gsplat/data/configure.h>
#include <vector>
#include <string>
#include <gsplat/data/gaussian_set.h>

namespace gsplat::data {

	class database {
		gsplat::data::gaussian_set gaussians_v;
	public:
		database();
		gsplat::data::gaussian_set& gaussian_set() { return gaussians_v; }
		const gsplat::data::gaussian_set& gaussian_set() const { return gaussians_v; }
	};

	database parse_colmaps_txt(const std::string& path_dir);
}

#endif // GSPLAT_DATA_DATABASE_H
