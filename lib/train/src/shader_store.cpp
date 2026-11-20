#include <gsplat/train/shader_store.h>
#include <cstdint>
#include <vector>

using namespace gsplat::train;
using namespace gsplat::vulkan;
using gsplat::data::gaussian;
using gsplat::data::gaussian2d;

// each shader reads its settings from a one element storage buffer.
// invoke() is taken to wait for the dispatch, like the old invoke_compute,
// so the settings buffers can be locals.

static int gaussian_count(const buffer& gaussians) {
	return (int)(gaussians.size() / sizeof(gaussian));
}

static int groups(int threads) {
	return (threads + 1023) / 1024;
}

shader_store::shader_store(const std::string& shader_dir) :
	calc_covariance_sp{compute_shader_config{shader_dir + "/calc_covariance.slang.spv", 2}},
	apply_gradient_sp{compute_shader_config{shader_dir + "/apply_gradient.slang.spv", 3}},
	loss_l1_sp{compute_shader_config{shader_dir + "/loss_l1.slang.spv", 5}},
	project_gaussians_sp{compute_shader_config{shader_dir + "/project_gaussians.slang.spv", 3}},
	project_backwards_sp{compute_shader_config{shader_dir + "/project_backward.slang.spv", 4}},
	raster_backwards_sp{compute_shader_config{shader_dir + "/raster_backward.slang.spv", 7}},
	raster_forward_sp{compute_shader_config{shader_dir + "/raster_forward.slang.spv", 6}},
	sort_gaussians_sp{compute_shader_config{shader_dir + "/sort_gaussians.slang.spv", 3}} {
}

stamp::task::coroutine<void> shader_store::calc_covariance(buffer& gaussians) {

	struct settings_t {
		int buffer_io_size;
	};
	int n = gaussian_count(gaussians);
	buffer settings = co_await buffer::from_vector(std::vector{settings_t{n}});

	calc_covariance_sp.bind<gaussian>(0, gaussians);
	calc_covariance_sp.bind<settings_t>(1, settings);
	calc_covariance_sp.invoke(groups(n), 1, 1);
}
stamp::task::coroutine<void> shader_store::apply_gradient(float learning_rate, buffer& gaussians, const buffer& gradients) {

	struct settings_t {
		float learning_rate;
		int buffer_size;
	};
	int n = gaussian_count(gaussians);
	buffer settings = co_await buffer::from_vector(std::vector{settings_t{learning_rate, n}});

	apply_gradient_sp.bind<gaussian>(0, gaussians);
	apply_gradient_sp.bind<gaussian>(1, gradients);
	apply_gradient_sp.bind<settings_t>(2, settings);
	apply_gradient_sp.invoke(groups(n), 1, 1);
}
stamp::task::coroutine<void> shader_store::loss_l1(int width, int height, const buffer& rastor_image, const buffer& target_image, buffer& loss_out, buffer& loss_per_pixel_out) {

	struct settings_t {
		int width;
		int height;
	};
	buffer settings = co_await buffer::from_vector(std::vector{settings_t{width, height}});

	loss_l1_sp.bind<glm::vec4>(0, rastor_image);
	loss_l1_sp.bind<std::uint32_t>(1, target_image); // rgba8 packed, like data::pixel
	loss_l1_sp.bind<glm::vec4>(2, loss_out);
	loss_l1_sp.bind<float>(3, loss_per_pixel_out);
	loss_l1_sp.bind<settings_t>(4, settings);
	loss_l1_sp.invoke(groups(width * height), 1, 1);
}

stamp::task::coroutine<void> shader_store::project_gaussians(const glm::mat4& world_to_camera, const glm::vec4& intrinsics, const glm::vec4& limits, const buffer& gaussians, buffer& gaussians_2d_out) {

	struct camera_t {
		glm::mat4 world_to_camera; // world space to camera space
		glm::vec4 intrinsics;        // fx, fy, cx, cy, in pixels
		glm::vec4 limits;
	};
	struct settings_t {
		camera_t camera;
		int gaussian_count;
		int pad0;
		int pad1;
		int pad2;
	};
	int n = gaussian_count(gaussians);
	buffer settings = co_await buffer::from_vector(std::vector{settings_t{camera_t{world_to_camera, intrinsics, limits}, n, 0, 0, 0}});

	project_gaussians_sp.bind<gaussian>(0, gaussians);
	project_gaussians_sp.bind<gaussian2d>(1, gaussians_2d_out);
	project_gaussians_sp.bind<settings_t>(2, settings);
	project_gaussians_sp.invoke(groups(n), 1, 1);
}
stamp::task::coroutine<void> shader_store::project_backwards(
	const glm::mat4& world_to_camera,
	const glm::vec4& intrinsics,
	const glm::vec4& limits,
	const buffer& gaussians,
	const buffer& raster_grads,
	buffer& gradients) {
	struct camera_t {
		glm::mat4 world_to_camera; // same matrix as project_gaussians' camera_transform
		glm::vec4 intrinsics;        // fx, fy, cx, cy, in pixels
		glm::vec4 limits;
	};
	struct settings_t {
		camera_t camera;
		int gaussian_count;
		int pad0;
		int pad1;
		int pad2;
	};
	int n = gaussian_count(gaussians);
	buffer settings = co_await buffer::from_vector(std::vector{settings_t{camera_t{world_to_camera, intrinsics, limits}, n, 0, 0, 0}});

	project_backwards_sp.bind<gaussian>(0, gaussians);
	project_backwards_sp.bind<float>(1, raster_grads);
	project_backwards_sp.bind<gaussian>(2, gradients);
	project_backwards_sp.bind<settings_t>(3, settings);
	project_backwards_sp.invoke(groups(n), 1, 1);
}

stamp::task::coroutine<void> shader_store::sort_gaussians(const buffer& gaussians_2d, buffer& sorted_index) {

	struct settings_t {
		int odd_even_state;
		int buffer_size;
	};

	int n = (int)(sorted_index.size() / sizeof(int));

	// one settings buffer per kind of pass, so nothing gets rewritten between passes
	buffer even = co_await buffer::from_vector(std::vector{settings_t{0, n}});
	buffer odd = co_await buffer::from_vector(std::vector{settings_t{1, n}});

	sort_gaussians_sp.bind<gaussian2d>(0, gaussians_2d);
	sort_gaussians_sp.bind<int>(1, sorted_index);

	// odd-even transition sort, GPU Gems 2 chapter 46.2. passes alternate between
	// even pairs (0,1) (2,3) ... and odd pairs (1,2) (3,4) ..., so no two threads
	// touch the same element. an element moves at most one slot per pass, so
	// after n passes everything is in place.
	for (int pass = 0; pass < n; pass++) {
		sort_gaussians_sp.bind<settings_t>(2, pass % 2 == 0 ? even : odd);
		sort_gaussians_sp.invoke(groups(n / 2), 1, 1);
	}
}

stamp::task::coroutine<void> shader_store::raster_forward(
	int width,
	int height,
	const glm::vec4& background_color,
	float near_plane,
	const buffer& gaussians_2d,
	const buffer& gaussians,
	const buffer& sorted_index,
	buffer& rastor_image_out,
	buffer& pixel_state_out) {

	struct settings_t {
		glm::vec4 background; // rgb, 0..1
		int width;
		int height;
		int gaussian_count;
		float near_plane;
	};
	buffer settings = co_await buffer::from_vector(std::vector{settings_t{background_color, width, height, gaussian_count(gaussians), near_plane}});

	raster_forward_sp.bind<gaussian2d>(0, gaussians_2d);
	raster_forward_sp.bind<gaussian>(1, gaussians);
	raster_forward_sp.bind<int>(2, sorted_index);
	raster_forward_sp.bind<glm::vec4>(3, rastor_image_out);
	raster_forward_sp.bind<pixel_state>(4, pixel_state_out);
	raster_forward_sp.bind<settings_t>(5, settings);
	raster_forward_sp.invoke(groups(width * height), 1, 1);
}

stamp::task::coroutine<void> shader_store::raster_backwards(
	int width,
	int height,
	const glm::vec4& background_color,
	float near_plane,
	const buffer& gaussians_2d,
	const buffer& gaussians,
	const buffer& sorted_index,
	const buffer& pixel_state_in,
	const buffer& loss,
	buffer& raster_grads_out) {

	// same settings as raster_forward
	struct settings_t {
		glm::vec4 background; // rgb, 0..1
		int width;
		int height;
		int gaussian_count;
		float near_plane;
	};
	buffer settings = co_await buffer::from_vector(std::vector{settings_t{background_color, width, height, gaussian_count(gaussians), near_plane}});

	raster_backwards_sp.bind<gaussian2d>(0, gaussians_2d);
	raster_backwards_sp.bind<gaussian>(1, gaussians);
	raster_backwards_sp.bind<int>(2, sorted_index);
	raster_backwards_sp.bind<pixel_state>(3, pixel_state_in);
	raster_backwards_sp.bind<glm::vec4>(4, loss);
	raster_backwards_sp.bind<float>(5, raster_grads_out);
	raster_backwards_sp.bind<settings_t>(6, settings);
	raster_backwards_sp.invoke(groups(width * height), 1, 1);
}
