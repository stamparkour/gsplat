#ifndef GSPLAT_TRAIN_SHADER_STORE_H
#define GSPLAT_TRAIN_SHADER_STORE_H

#include <gsplat/vulkan/shader.h>
#include <gsplat/data/database.h>
#include <gsplat/core/read_file.h>
#include <string>

namespace gsplat::train {
	class shader_store {

		gsplat::data::database* database_v;

		gsplat::vulkan::shader calc_covariance_s;
		gsplat::vulkan::compute_shader_pipeline calc_covariance_sp;
		gsplat::vulkan::shader apply_gradient_s;
		gsplat::vulkan::compute_shader_pipeline apply_gradient_sp;
		gsplat::vulkan::shader loss_l1_s;
		gsplat::vulkan::compute_shader_pipeline loss_l1_sp;
		gsplat::vulkan::shader project_gaussians_s;
		gsplat::vulkan::compute_shader_pipeline project_gaussians_sp;
		gsplat::vulkan::shader project_backwards_s;
		gsplat::vulkan::compute_shader_pipeline project_backwards_sp;
		gsplat::vulkan::shader raster_backwards_s;
		gsplat::vulkan::compute_shader_pipeline raster_backwards_sp;
	public:
		shader_store(const std::string& shader_dir, gsplat::data::database* db, gsplat::vulkan::command_queue* q);

		// calcs coveraince of gaussian_set.buffer1
		void calc_covariance();
		// add gradient gaussian_set.buffer2 to gaussian_set.buffer1
		void apply_gradient(float learning_rate);
		void loss_l1(
			int target_image, 
			gsplat::vulkan::array_buffer<glm::vec4>* rastor_image, 
			gsplat::vulkan::array_buffer<glm::vec4>* loss_out, 
			gsplat::vulkan::array_buffer<float>* loss_per_pixel_out);
	
		// projects the gaussian onto the 2d plain
		void project_gaussians(
			glm::mat4* camera_transform,
			glm::mat4* camera_projection,
			gsplat::vulkan::array_buffer<gsplat::data::gaussian2d>* gaussian2d_out);

		void project_backwards(
			glm::mat4* world_to_camera,
			glm::vec4* intrinsics,
			glm::vec4* limits,
			gsplat::vulkan::array_buffer<float>* raster_grads);

		//void raster_backwards(
		//	glm::mat4* world_to_camera,
		//	glm::vec4* intrinsics,
		//	glm::vec4* limits,
		//	gsplat::vulkan::array_buffer<float>* raster_grads);
		
		// explain how project_backword works
	};
}

#endif // GSPLAT_TRAIN_SHADER_STORE_H