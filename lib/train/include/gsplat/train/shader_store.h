#ifndef GSPLAT_TRAIN_SHADER_STORE_H
#define GSPLAT_TRAIN_SHADER_STORE_H

#include <gsplat/vulkan/memory.h>
#include <gsplat/data/gaussian.h>
#include <stamp/task/coroutine.h>
#include <string>

namespace gsplat::train {
	// what raster_forward saves per pixel for raster_backwards (pixel_state_t in splat_math.h)
	struct pixel_state {
		float T_final;
		int stop; // how far down the sorted list the forward got
	};

	// one compute_shader per slang file. the store holds no data, every buffer is
	// passed in, and counts come from the buffer sizes (bytes / sizeof the element).
	// buffers by type: gaussians and gradients hold gsplat::data::gaussian,
	// gaussians_2d holds gsplat::data::gaussian2d, sorted_index holds int,
	// raster_grads holds 9 floats per gaussian, images hold glm::vec4 per pixel,
	// target images hold one rgba8 pixel (gsplat::data::pixel) per uint32.
	class shader_store {
		gsplat::vulkan::compute_shader calc_covariance_sp;
		gsplat::vulkan::compute_shader apply_gradient_sp;
		gsplat::vulkan::compute_shader loss_l1_sp;
		gsplat::vulkan::compute_shader project_gaussians_sp;
		gsplat::vulkan::compute_shader project_backwards_sp;
		gsplat::vulkan::compute_shader raster_backwards_sp;
		gsplat::vulkan::compute_shader raster_forward_sp;
		gsplat::vulkan::compute_shader sort_gaussians_sp;
	public:
		explicit shader_store(const std::string& shader_dir);

		// calcs the covariance of every gaussian, in place
		stamp::task::coroutine<void> calc_covariance(gsplat::vulkan::buffer& gaussians);
		// gaussians -= learning_rate * gradients
		stamp::task::coroutine<void> apply_gradient(float learning_rate, gsplat::vulkan::buffer& gaussians, const gsplat::vulkan::buffer& gradients);
		stamp::task::coroutine<void> loss_l1(
			int width,
			int height,
			const gsplat::vulkan::buffer& rastor_image,
			const gsplat::vulkan::buffer& target_image,
			gsplat::vulkan::buffer& loss_out,
			gsplat::vulkan::buffer& loss_per_pixel_out);

		// projects the gaussians onto the 2d plane. takes the same camera as
		// project_backwards, so the backward differentiates exactly this projection.
		// world_to_camera: world space to camera space (OpenCV axes, z forward)
		// intrinsics: fx, fy, cx, cy in pixels
		// limits: 1.3 tan(fovx/2), 1.3 tan(fovy/2), near plane, 0
		stamp::task::coroutine<void> project_gaussians(
			const glm::mat4& world_to_camera,
			const glm::vec4& intrinsics,
			const glm::vec4& limits,
			const gsplat::vulkan::buffer& gaussians,
			gsplat::vulkan::buffer& gaussians_2d_out);

		// odd-even transition sort (GPU Gems 2, 46.2) of sorted_index by depth,
		// front to back. the 2d gaussians themselves never move.
		stamp::task::coroutine<void> sort_gaussians(
			const gsplat::vulkan::buffer& gaussians_2d,
			gsplat::vulkan::buffer& sorted_index);

		// renders a width x height image into rastor_image_out, and saves per pixel
		// what raster_backwards needs
		stamp::task::coroutine<void> raster_forward(
			int width,
			int height,
			const glm::vec4& background_color,
			float near_plane,
			const gsplat::vulkan::buffer& gaussians_2d,
			const gsplat::vulkan::buffer& gaussians,
			const gsplat::vulkan::buffer& sorted_index,
			gsplat::vulkan::buffer& rastor_image_out,
			gsplat::vulkan::buffer& pixel_state_out);

		// walks each pixel's list back to front and adds into raster_grads_out,
		// 9 floats per gaussian: dL/dcolor (3), dL/dopacity, dL/dmean2d (2),
		// dL/dconic xx, xy, yy. these are 2d, so they belong to this image's
		// camera. clear raster_grads_out before every image.
		// loss is loss_l1's loss_out (dL/dC per pixel).
		stamp::task::coroutine<void> raster_backwards(
			int width,
			int height,
			const glm::vec4& background_color,
			float near_plane,
			const gsplat::vulkan::buffer& gaussians_2d,
			const gsplat::vulkan::buffer& gaussians,
			const gsplat::vulkan::buffer& sorted_index,
			const gsplat::vulkan::buffer& pixel_state_in,
			const gsplat::vulkan::buffer& loss,
			gsplat::vulkan::buffer& raster_grads_out);

		// how project_backwards works: one thread per gaussian. it reads that
		// gaussian's 9 raster_grads and pushes them back through the projection
		// (backward.pdf sections 3 to 6). color and opacity pass straight
		// through. the 2d center and conic go back through the inverse, then
		// J W Sigma W^T J^T + 0.3I, then the perspective divide, to the 3d mean,
		// scale and quaternion. it ADDS the result into gradients in gaussian
		// layout, so gradients sum over images until they're cleared, and
		// apply_gradient reads them from there. clear gradients at the start of
		// every step.
		stamp::task::coroutine<void> project_backwards(
			const glm::mat4& world_to_camera,
			const glm::vec4& intrinsics,
			const glm::vec4& limits,
			const gsplat::vulkan::buffer& gaussians,
			const gsplat::vulkan::buffer& raster_grads,
			gsplat::vulkan::buffer& gradients);
	};
}

#endif // GSPLAT_TRAIN_SHADER_STORE_H