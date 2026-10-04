#ifndef GSPLAT_TRAIN_H
#define GSPLAT_TRAIN_H

#include <gsplat/train/shader_store.h>

namespace gsplat::train {
	// what project_gaussians and project_backwards take
	struct camera {
		glm::mat4 world_to_camera; // world space to camera space (OpenCV axes, z forward)
		glm::vec4 intrinsics;      // fx, fy, cx, cy in pixels
		glm::vec4 limits;          // 1.3 tan(fovx/2), 1.3 tan(fovy/2), near plane, 0
	};

	struct train_settings {
		int epochs = 100;
		int images_per_step = 1;            // images whose gradients add up before one apply_gradient
		float learning_rate = 1.0f;         // plain SGD, so this depends on the scene
		float near_plane = 0.2f;
		glm::vec4 background{0, 0, 0, 0};   // rgb, 0..1
		glm::vec4 intrinsics{0, 0, 0, 0};   // fx, fy, cx, cy in pixels of the loaded images. TODO: cameras.txt, once the database loads it
		std::string export_path{};
	};

	// COLMAP pose to the camera the shaders take
	camera make_camera(const gsplat::data::image_pose& pose, glm::vec4 intrinsics, int width, int height, float near_plane);

	// called once per epoch with the mean loss over that epoch's images
	void record_epoch_loss(int epoch, float loss);

	// trains the database's gaussians in place, on the GPU
	void train(shader_store* shaders, gsplat::data::database* db, gsplat::vulkan::command_queue* q, const train_settings& ts = {});

	void export_png(const std::string& out_path, gsplat::vulkan::array_buffer<glm::vec4>* buf, int width, int height);
}

#endif // GSPLAT_TRAIN_H
