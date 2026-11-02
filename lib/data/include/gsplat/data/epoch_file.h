#ifndef GSPLAT_DATA_EPOCH_FILE_H
#define GSPLAT_DATA_EPOCH_FILE_H

#include <gsplat/data/configure.h>
#include <gsplat/data/gaussian_set.h>
#include <filesystem>

namespace stamp::data {
	void save_bin(const gsplat::data::gaussian_set& set, const std::filesystem::path& path);
	void load_bin(gsplat::data::gaussian_set& set, const std::filesystem::path& path);
}

#endif // GSPLAT_DATA_EPOCH_FILE_H
