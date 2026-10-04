#ifndef GSPLAT_TRAIN_SHADER_STORE_H
#define GSPLAT_TRAIN_SHADER_STORE_H

#include <gsplat/vulkan/shader.h>
#include <gsplat/data/database.h>
#include <gsplat/core/read_file.h>
#include <string>

namespace gsplat::train {
	// what raster_forward saves per pixel for raster_backwards (pixel_state_t in splat_math.h)
	struct pixel_state {
		float T_final;
		int stop; // how far down the sorted list the forward got
	};

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
		gsplat::vulkan::shader raster_forward_s;
		gsplat::vulkan::compute_shader_pipeline raster_forward_sp;
		gsplat::vulkan::shader sort_gaussians_s;
		gsplat::vulkan::compute_shader_pipeline sort_gaussians_sp;
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
	
		// projects the gaussian onto the 2d plain. takes the same camera as
		// project_backwards, so the backward differentiates exactly this projection.
		// world_to_camera: world space to camera space (OpenCV axes, z forward)
		// intrinsics: fx, fy, cx, cy in pixels
		// limits: 1.3 tan(fovx/2), 1.3 tan(fovy/2), near plane, 0
		void project_gaussians(
			glm::mat4* world_to_camera,
			glm::vec4* intrinsics,
			glm::vec4* limits,
			gsplat::vulkan::array_buffer<gsplat::data::gaussian2d>* gaussian2d_out);

		// odd-even transition sort (GPU Gems 2, 46.2) of sorted_index by depth,
		// front to back. the 2d gaussians themselves never move.
		void sort_gaussians(
			gsplat::vulkan::array_buffer<gsplat::data::gaussian2d>* gaussians_2d,
			gsplat::vulkan::array_buffer<int>* sorted_index);

		// renders target_image's size into rastor_image_out, and saves per pixel
		// what raster_backwards needs
		void raster_forward(
			int target_image,
			glm::vec4* background_color,
			float near_plane,
			gsplat::vulkan::array_buffer<gsplat::data::gaussian2d>* gaussians_2d,
			gsplat::vulkan::array_buffer<int>* sorted_index,
			gsplat::vulkan::array_buffer<glm::vec4>* rastor_image_out,
			gsplat::vulkan::array_buffer<pixel_state>* pixel_state_out);

		// walks each pixel's list back to front and adds into raster_grads_out,
		// 9 floats per gaussian: dL/dcolor (3), dL/dopacity, dL/dmean2d (2),
		// dL/dconic xx, xy, yy. these are 2d, so they belong to this image's
		// camera. clear raster_grads_out before every image.
		// loss is loss_l1's loss_out (dL/dC per pixel).
		void raster_backwards(
			int target_image,
			glm::vec4* background_color,
			float near_plane,
			gsplat::vulkan::array_buffer<gsplat::data::gaussian2d>* gaussians_2d,
			gsplat::vulkan::array_buffer<int>* sorted_index,
			gsplat::vulkan::array_buffer<pixel_state>* pixel_state_in,
			gsplat::vulkan::array_buffer<glm::vec4>* loss,
			gsplat::vulkan::array_buffer<float>* raster_grads_out);

		// how project_backwards works: one thread per gaussian. it reads that
		// gaussian's 9 raster_grads and pushes them back through the projection
		// (backward.pdf sections 3 to 6). color and opacity pass straight
		// through. the 2d center and conic go back through the inverse, then
		// J W Sigma W^T J^T + 0.3I, then the perspective divide, to the 3d mean,
		// scale and quaternion. it ADDS the result into gaussian_set.buffer2 in
		// gaussian layout, so gradients sum over images until buffer2 is cleared,
		// and apply_gradient reads them from there. clear buffer2 at the start
		// of every step (it starts out as a copy of the gaussians).
		void project_backwards(
			glm::mat4* world_to_camera,
			glm::vec4* intrinsics,
			glm::vec4* limits,
			gsplat::vulkan::array_buffer<float>* raster_grads);
	};
}

#endif // GSPLAT_TRAIN_SHADER_STORE_H