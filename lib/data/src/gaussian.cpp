#include <gsplat/data/gaussian.h>
#include <gsplat/data/gaussian_set.h>
#include <algorithm>

using namespace gsplat::data;
using namespace gsplat::vulkan;

gaussian_set gaussian_set::colmap_txt(const std::string& colmap_dir, gsplat::vulkan::command_queue*) {
	return gaussian_set();
}

void gaussian_set::store_vulkan() {
	array_buffer<gaussian> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_storage_buffer = true,
		.size = gaussians_v.size(),
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			queue_v,
		}
	}};

	{
		auto l = buf.memory_lock();
		std::copy(gaussians_v.begin(), gaussians_v.end(), l.begin());
	}

	buffer_v.transfer(&buf);
}

void gaussian_set::fetch_vulkan() {
	array_buffer<gaussian> buf{buffer_creation_settings{
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = gaussians_v.size(),
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			queue_v,
		}
	}};

	buf.transfer(&buffer_v);

	{
		auto l = buf.memory_lock();
		std::copy(l.begin(), l.end(), gaussians_v.begin());
	}
}