#ifndef GSPLAT_DATA_GAUSSIAN_SET_H
#define GSPLAT_DATA_GAUSSIAN_SET_H

#include <gsplat/data/configure.h>
#include <gsplat/data/gaussian.h>
#include <gsplat/vulkan/buffer.h>
#include <vector>
#include <string>

namespace gsplat::data {
	class gaussian_set {
		inline static constexpr int buffer_count = 2;
		gsplat::vulkan::array_buffer<gaussian> buffers_v[buffer_count];
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

		void set_active_buffer(int index = 1) { active_buffer = index; }
		void inc_active_buffer() { active_buffer++; }

		std::size_t size() const {
			return gaussians_v.size();
		}

		std::vector<gaussian>& data() { return gaussians_v; }
		const std::vector<gaussian>& data() const { return gaussians_v; }
		gsplat::vulkan::array_buffer<gaussian>& buffer1_vulkan() { return buffers_v[(active_buffer+0)%buffer_count]; }
		const gsplat::vulkan::array_buffer<gaussian>& buffer1_vulkan() const { return buffers_v[(active_buffer + 0) % buffer_count]; }
		gsplat::vulkan::array_buffer<gaussian>& buffer2_vulkan() { return buffers_v[(active_buffer + 1) % buffer_count]; }
		const gsplat::vulkan::array_buffer<gaussian>& buffer2_vulkan() const { return buffers_v[(active_buffer + 1) % buffer_count]; }
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_SET_H
