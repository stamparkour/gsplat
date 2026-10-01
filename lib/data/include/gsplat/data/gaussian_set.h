#ifndef GSPLAT_DATA_GAUSSIAN_SET_H
#define GSPLAT_DATA_GAUSSIAN_SET_H

#include <gsplat/data/configure.h>
#include <gsplat/data/gaussian.h>
#include <gsplat/vulkan/buffer.h>
#include <vector>
#include <string>

namespace gsplat::data {
	class gaussian_set {
		gsplat::vulkan::array_buffer<gaussian> buffer1_v;
		gsplat::vulkan::array_buffer<gaussian> buffer2_v;
		int active_buffer;
		std::vector<gaussian> gaussians_v;
		gsplat::vulkan::command_queue* queue_v;
	public:
		gaussian_set() = default;

		// generates the vulkan buffer and gaussian vector.
		// does not generate covariance matrix. must be done by another function (compute shader)
		static gaussian_set colmap_txt(const std::string& path_colmap_points3d_txt, gsplat::vulkan::command_queue*);

		void store_vulkan(int buffer_target = 1);
		void fetch_vulkan(int buffer_target = 1);

		std::vector<gaussian>& data() { return gaussians_v; }
		const std::vector<gaussian>& data() const { return gaussians_v; }
		gsplat::vulkan::array_buffer<gaussian>& buffer1_vulkan() { return buffer1_v; }
		const gsplat::vulkan::array_buffer<gaussian>& buffer1_vulkan() const { return buffer1_v; }
		gsplat::vulkan::array_buffer<gaussian>& buffer2_vulkan() { return buffer2_v; }
		const gsplat::vulkan::array_buffer<gaussian>& buffer2_vulkan() const { return buffer2_v; }
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_SET_H
