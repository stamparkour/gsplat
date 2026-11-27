#ifndef GSPLAT_DATA_EPOCH_FILE_H
#define GSPLAT_DATA_EPOCH_FILE_H

#include <gsplat/data/configure.h>
#include <gsplat/data/gaussian.h>
#include <filesystem>

namespace gsplat::data {
	void save_bin(const std::vector<gaussian>& set, const std::filesystem::path& path);
	void load_bin(std::vector<gaussian>& set, const std::filesystem::path& path);
}

#endif // GSPLAT_DATA_EPOCH_FILE_H
