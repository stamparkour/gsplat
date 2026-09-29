#include <gsplat/data/image.h>
#include <gsplat/data/image_set.h>

using namespace gsplat::data;
using namespace gsplat::vulkan;


void image::store_vulkan() {
	array_buffer<pixel> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_storage_buffer = true,
		.size = data_v.size(),
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
		std::copy(data_v.begin(), data_v.end(), l.begin());
	}

	buffer_v.transfer(&buf);
}

void image::fetch_vulkan() {
	array_buffer<pixel> buf{buffer_creation_settings{
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = data_v.size(),
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
		std::copy(l.begin(), l.end(), data_v.begin());
	}
}