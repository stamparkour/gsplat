#ifndef GSPLAT_DATA_GAUSSIAN_SET_H
#define GSPLAT_DATA_GAUSSIAN_SET_H

#include <gsplat/data/configure.h>
#include <gsplat/data/gaussian.h>
#include <gsplat/vulkan/buffer.h>
#include <vector>
#include <filesystem>

namespace gsplat::data {
	class gaussian_set {
		gsplat::vulkan::array_buffer<gaussian> buffer_v;
		std::vector<gaussian> gaussians_v;
	public:
		gaussian_set();

		static gaussian_set colmap(const std::filesystem::path& colmap_dir);

		void store_vulkan();
		void fetch_vulkan();

		std::vector<gaussian>& vector();
		const std::vector<gaussian>& vector() const;
		gsplat::vulkan::array_buffer<gaussian>& buffer_vulkan();
		const gsplat::vulkan::array_buffer<gaussian>& buffer_vulkan() const;
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_SET_H
