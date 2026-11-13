#ifndef GSPLAT_DATA_GAUSSIAN_SET_H
#define GSPLAT_DATA_GAUSSIAN_SET_H

#include <gsplat/data/configure.h>
#include <gsplat/data/gaussian.h>
#include <gsplat/vulkan/memory.h>
#include <vector>
#include <string>

namespace gsplat::data {
	class gaussian_set {
		inline static constexpr int buffer_count = 2;
		gsplat::vulkan::buffer buffers_v[buffer_count];
		int active_buffer;
		std::vector<gaussian> gaussians_v;
	public:
		gaussian_set() = default;

		// generates the vulkan buffer and gaussian vector.
		// does not generate covariance matrix. must be done by another function (compute shader)
		static gaussian_set colmap_txt(const std::string& path_colmap_points3d_txt);


		void store_vulkan(int buffer_target = 0);
		void fetch_vulkan(int buffer_target = 0);
		// remakes both buffers at data().size() and stores data() into them
		void resize_vulkan();

		void set_active_buffer(int index = 0) { active_buffer = index; }
		void inc_active_buffer() { active_buffer++; }

		std::size_t size() const {
			return gaussians_v.size();
		}

		void filter_gaussians();

		std::vector<gaussian>& data() { return gaussians_v; }
		const std::vector<gaussian>& data() const { return gaussians_v; }
		gsplat::vulkan::buffer& buffer1_vulkan() { return buffers_v[(active_buffer+0)%buffer_count]; }
		const gsplat::vulkan::buffer& buffer1_vulkan() const { return buffers_v[(active_buffer + 0) % buffer_count]; }
		gsplat::vulkan::buffer& buffer2_vulkan() { return buffers_v[(active_buffer + 1) % buffer_count]; }
		const gsplat::vulkan::buffer& buffer2_vulkan() const { return buffers_v[(active_buffer + 1) % buffer_count]; }
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_SET_H
