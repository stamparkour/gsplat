#include <iostream>
#include <gsplat/vulkan.h>

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
		if (!(v->queueFlags & VK_QUEUE_COMPUTE_BIT)) score = -1;
		return score;
	}).resize(1);
	auto my_logical_device = my_physical_device->create_logical_device(my_queue_family_collection);

	auto& my_queue = my_logical_device->queue_at(0);
}