#include <iostream>
#include <gsplat/vulkan.h>
#include <gsplat/core/read_file.h>
#include <gsplat/data/database.h>
#include <gsplat/train/shader_store.h>
#include <gsplat/train.h>

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
	gsplat::vulkan::command_queue* my_queue = &my_logical_device->queue_collection().at(0);

	std::string path;
	if (argc <= 1) path = R"(I:\projects\3DGS\testset)";


	// load database
	std::cout << "loading dataset" << std::endl;
	gsplat::data::database db = gsplat::data::database::colmap(path, my_queue);
	
	glm::vec4 intrinsics{
		db.image_set().data()[0].pose().fx,
		db.image_set().data()[0].pose().fy,
		db.image_set().data()[0].pose().cx,
		db.image_set().data()[0].pose().cy,
	};

	std::cout << "loading shaders" << std::endl;
	// load shaders
	gsplat::train::shader_store shaders{std::string{"shaders"}, &db, my_queue};

	std::cout << "training!" << std::endl;

	gsplat::train::train(&shaders, &db, my_queue, gsplat::train::train_settings{
		.epochs = 300,
		.images_per_step = 5,            // images whose gradients add up before one apply_gradient
		.learning_rate = 10.0f,         // plain SGD, so this depends on the scene
		.near_plane = 0.2f,
		.background{0, 0, 0, 0},   // rgb, 0..1
		.intrinsics = intrinsics,   // fx, fy, cx, cy in pixels of the loaded images. TODO: cameras.txt, once the database loads it
		.export_path = path + "/out/"
	});
}