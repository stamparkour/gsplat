#ifndef GSPLAT_TRAIN_H
#define GSPLAT_TRAIN_H

#include <gsplat/train/shader_store.h>
#include <gsplat/data/database.h>
#include <stamp/task/coroutine.h>
#include <vector>

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
		int modify_interval = 50;           // epochs between modify_gaussians, 0 turns it off
		int sh_degree = 0;                  // view dependent color up to this degree (1 to 3), 0 turns it off
		int sh_interval = 50;               // epochs between degree steps, the first waits for the last modify_gaussians
		float sh_rate_scale = 0.05f;        // the sh coefficients step at this times the learning rate (3DGS uses 1/20)
	};

	// COLMAP pose to the camera the shaders take
	camera make_camera(const gsplat::data::image_pose& pose, glm::vec4 intrinsics, int width, int height, float near_plane);

	// called once per epoch with the mean loss over that epoch's images
	void record_epoch_loss(int epoch, float loss);

	// trains the database's gaussians on the GPU, then writes them back into
	// db->gaussian_set().data(). only that vector of gaussian_set is used, none
	// of its buffers. call .resume() once to run it, every buffer call in it
	// finishes before returning, so it runs to the end. (stamp::task::wait_for
	// hangs, its lambda coroutine captures by reference and outlives the lambda.)
	// ts is taken by value so it lives in the coroutine, not with the caller.
	stamp::task::coroutine<void> train(shader_store* shaders, gsplat::data::database* db, train_settings ts = {});

	// pixels is one rgba per pixel, 0..1
	void export_png(const std::string& out_path, const std::vector<glm::vec4>& pixels, int width, int height);

	stamp::task::coroutine<void> epoch(gsplat::data::database* db, const train_settings& ts = {});
}

#endif // GSPLAT_TRAIN_H
