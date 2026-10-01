#include <iostream>
#include <gsplat/vulkan.h>
#include <gsplat/core/read_file.h>
#include <gsplat/data/database.h>
#include <gsplat/train/shader_store.h>

using namespace gsplat::vulkan;

int main(int argc, char** argv) {
	std::cout << "Hello World!" << std::endl;

	// init context
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
	if (argc <= 1) path = "C:\\Users\\Stamp\\Documents\\FIT\\MyStuff\\Jason-1_LEO_VBAR_dx10.00_tumble5_ecl_brdf";


	// load database
	gsplat::data::database db = gsplat::data::database::colmap(path, my_queue);
	
	// load shaders
	gsplat::train::shader_store shaders{std::string{"shaders"}, &db, my_queue};

	shaders.calc_covariance();

	db.gaussian_set().fetch_vulkan(1);
}