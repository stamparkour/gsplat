#include <iostream>
#include <gsplat/vulkan.h>

using namespace gsplat::vulkan;

int main(int argc, char** argv) {
	std::cout << "Hello World!" << std::endl;

	vulkan_context context{};
	physical_device_collection devices = context.device_collection();
	auto v = devices.front().queue_families_vulkan();
}