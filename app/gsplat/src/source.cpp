#include <iostream>
#include <gsplat/vulkan.h>
#include <gsplat/core/read_file.h>
#include <gsplat/data/database.h>

using namespace gsplat::vulkan;

int main(int argc, char** argv) {
	std::cout << "Hello World!" << std::endl;

	vulkan_context context{};

	auto my_physical_device = context.device_collection_sort([](const physical_device* v) -> int {
		int score = 0;
		auto properties = v->device_properties_vulkan();
		auto features = v->device_features_vulkan();
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 10;
		return score;
	}).front();
	auto my_queue_family_collection = my_physical_device->queue_families().sort([](const VkQueueFamilyProperties* v) -> int {
		int score = 0;
		if (!(v->queueFlags & VK_QUEUE_COMPUTE_BIT)) return -1;
		// if (!(v->queueFlags & VK_QUEUE_TRANSFER_BIT)) return -1; // implicit from VK_QUEUE_COMPUTE_BIT
		return score;
	}).resize(1);
	auto my_logical_device = my_physical_device->create_logical_device(my_queue_family_collection);
	auto my_queue = &my_logical_device->queue_collection().at(0);

	std::string path;
	if (argc <= 1) path = "I:\\FIT\\MyStuff\\Jason-1_LEO_VBAR_dx10.00_tumble5_ecl_brdf";

	gsplat::data::database db = gsplat::data::database::colmap(path, my_queue);


	array_buffer<int> buf2{buffer_creation_settings{
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = 500,
		.memory = {
			device_memory_settings{}
		},
		.queues = {
			my_queue,
		}
	}};

	array_buffer<int> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_storage_buffer = true,
		.size = 500,
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			my_queue,
		}
	}};

	{
		auto l = buf.memory_lock();
		int i = 0;
		for (auto& v : l) {
			v = i;
			i++;
		}
		std::cout << "in buffer" << std::endl;
	}

	buf2.transfer(&buf);

	auto file = gsplat::core::load_entire_file_binary("shaders/my_shader.slang.spv");

	shader my_shader{
		shader_creation_settings{
			.file_ptr = file.data(),
			.file_size = file.size(),
			.device = my_logical_device.get()
		}
	};

	compute_shader_pipeline pipe{
		compute_shader_pipeline_settings{
			.storage_buffer_count = 2,
			.queue = &my_logical_device->queue_collection().at(0),
			.shader = &my_shader
		}
	};

	pipe.bind(0, &buf2);
	pipe.bind(1, &buf);
	pipe.invoke_compute(500, 1, 1);

	{
		auto l = buf.memory_lock();
	}
}