#ifndef GSPLAT_DATA_GAUSSIAN_SET_H
#define GSPLAT_DATA_GAUSSIAN_SET_H

#include <gsplat/data/configure.h>
#include <gsplat/data/gaussian.h>
#include <gsplat/vulkan/buffer.h>
#include <vector>
#include <string>

namespace gsplat::data {
	class gaussian_set {
		gsplat::vulkan::array_buffer<gaussian> buffer_v;
		std::vector<gaussian> gaussians_v;
		gsplat::vulkan::command_queue* queue_v;
	public:
		gaussian_set() = default;

		static gaussian_set colmap_txt(const std::string& colmap_dir, gsplat::vulkan::command_queue*);

		void store_vulkan();
		void fetch_vulkan();

		std::vector<gaussian>& data() { return gaussians_v; }
		const std::vector<gaussian>& data() const { return gaussians_v; }
		gsplat::vulkan::array_buffer<gaussian>& buffer_vulkan() { return buffer_v; }
		const gsplat::vulkan::array_buffer<gaussian>& buffer_vulkan() const { return buffer_v; }
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_SET_H
