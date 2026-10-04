#include <gsplat/train/shader_store.h>

using namespace gsplat::train;
using namespace gsplat::vulkan;
using namespace gsplat::core;

shader_store::shader_store(const std::string& shader_dir, gsplat::data::database* db, gsplat::vulkan::command_queue* q) {
	this->database_v = db;

	std::vector<char> vec;

	vec = load_entire_file_binary(shader_dir + "/calc_covariance.slang.spv");

	calc_covariance_s = gsplat::vulkan::shader{
		gsplat::vulkan::shader_creation_settings{
			.file_ptr = vec.data(),
			.file_size = vec.size(),
			.device = q->logical_device().get()
		}
	};
	calc_covariance_sp = gsplat::vulkan::compute_shader_pipeline{
		gsplat::vulkan::compute_shader_pipeline_settings{
			.storage_buffer_count = 2,
			.queue = q,
			.shader = &calc_covariance_s
		}
	};

	vec = load_entire_file_binary(shader_dir + "/apply_gradient.slang.spv");

	apply_gradient_s = gsplat::vulkan::shader{
		gsplat::vulkan::shader_creation_settings{
			.file_ptr = vec.data(),
			.file_size = vec.size(),
			.device = q->logical_device().get()
		}
	};
	apply_gradient_sp = gsplat::vulkan::compute_shader_pipeline{
		gsplat::vulkan::compute_shader_pipeline_settings{
			.storage_buffer_count = 3,
			.queue = q,
			.shader = &apply_gradient_s
		}
	};

	vec = load_entire_file_binary(shader_dir + "/loss_l1.slang.spv");

	loss_l1_s = gsplat::vulkan::shader{
		gsplat::vulkan::shader_creation_settings{
			.file_ptr = vec.data(),
			.file_size = vec.size(),
			.device = q->logical_device().get()
		}
	};
	loss_l1_sp = gsplat::vulkan::compute_shader_pipeline{
		gsplat::vulkan::compute_shader_pipeline_settings{
			.storage_buffer_count = 5,
			.queue = q,
			.shader = &loss_l1_s
		}
	};

	vec = load_entire_file_binary(shader_dir + "/project_gaussians.slang.spv");

	project_gaussians_s = gsplat::vulkan::shader{
		gsplat::vulkan::shader_creation_settings{
			.file_ptr = vec.data(),
			.file_size = vec.size(),
			.device = q->logical_device().get()
		}
	};
	project_gaussians_sp = gsplat::vulkan::compute_shader_pipeline{
		gsplat::vulkan::compute_shader_pipeline_settings{
			.storage_buffer_count = 3,
			.queue = q,
			.shader = &project_gaussians_s
		}
	};

	vec = load_entire_file_binary(shader_dir + "/project_backward.slang.spv");

	project_backwards_s = gsplat::vulkan::shader{
		gsplat::vulkan::shader_creation_settings{
			.file_ptr = vec.data(),
			.file_size = vec.size(),
			.device = q->logical_device().get()
		}
	};
	project_backwards_sp = gsplat::vulkan::compute_shader_pipeline{
		gsplat::vulkan::compute_shader_pipeline_settings{
			.storage_buffer_count = 4,
			.queue = q,
			.shader = &project_backwards_s
		}
	};

	vec = load_entire_file_binary(shader_dir + "/raster_backward.slang.spv");

	raster_backwards_s = gsplat::vulkan::shader{
		gsplat::vulkan::shader_creation_settings{
			.file_ptr = vec.data(),
			.file_size = vec.size(),
			.device = q->logical_device().get()
		}
	};
	raster_backwards_sp = gsplat::vulkan::compute_shader_pipeline{
		gsplat::vulkan::compute_shader_pipeline_settings{
			.storage_buffer_count = 7,
			.queue = q,
			.shader = &raster_backwards_s
		}
	};

	vec = load_entire_file_binary(shader_dir + "/raster_forward.slang.spv");

	raster_forward_s = gsplat::vulkan::shader{
		gsplat::vulkan::shader_creation_settings{
			.file_ptr = vec.data(),
			.file_size = vec.size(),
			.device = q->logical_device().get()
		}
	};
	raster_forward_sp = gsplat::vulkan::compute_shader_pipeline{
		gsplat::vulkan::compute_shader_pipeline_settings{
			.storage_buffer_count = 6,
			.queue = q,
			.shader = &raster_forward_s
		}
	};

	vec = load_entire_file_binary(shader_dir + "/sort_gaussians.slang.spv");

	sort_gaussians_s = gsplat::vulkan::shader{
		gsplat::vulkan::shader_creation_settings{
			.file_ptr = vec.data(),
			.file_size = vec.size(),
			.device = q->logical_device().get()
		}
	};
	sort_gaussians_sp = gsplat::vulkan::compute_shader_pipeline{
		gsplat::vulkan::compute_shader_pipeline_settings{
			.storage_buffer_count = 3,
			.queue = q,
			.shader = &sort_gaussians_s
		}
	};
}

void shader_store::calc_covariance() {

	struct settings_t {
		int buffer_io_size;
	};

	array_buffer<settings_t> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = 1,
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			calc_covariance_sp.queue()
		}
	}};

	{
		auto l = buf.memory_lock();
		l.data()->buffer_io_size = (int)database_v->gaussian_set().data().size();

	}
	calc_covariance_sp.bind(0, &(database_v->gaussian_set().buffer1_vulkan()));
	calc_covariance_sp.bind(1, &buf);
	calc_covariance_sp.invoke_compute(((int)database_v->gaussian_set().size() + 1023) / 1024, 1, 1);
}
void shader_store::apply_gradient(float learning_rate) {

	struct settings_t {
		float learning_rate;
		int buffer_size;
	};

	array_buffer<settings_t> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_storage_buffer = true,
		.size = 1,
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			apply_gradient_sp.queue()
		}
	}};
	{
		auto l = buf.memory_lock();
		l.data()->buffer_size = (int)database_v->gaussian_set().data().size();
		l.data()->learning_rate = learning_rate;

	}
	apply_gradient_sp.bind(0, &(database_v->gaussian_set().buffer1_vulkan()));
	apply_gradient_sp.bind(1, &(database_v->gaussian_set().buffer2_vulkan()));
	apply_gradient_sp.bind(2, &buf);
	apply_gradient_sp.invoke_compute(((int)database_v->gaussian_set().size() + 1023) / 1024, 1, 1);
}
void shader_store::loss_l1(int target_image, array_buffer<glm::vec4>* rastor_image, array_buffer<glm::vec4>* loss_out, array_buffer<float>* loss_per_pixel_out) {

	struct settings_t {
		int width;
		int height;
	};

	array_buffer<settings_t> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_storage_buffer = true,
		.size = 1,
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			loss_l1_sp.queue()
		}
	}};
	auto& image = database_v->image_set().data()[target_image];

	{
		auto l = buf.memory_lock();
		l.data()->width = image.width();
		l.data()->height = image.height();

	}

	loss_l1_sp.bind(0, rastor_image);
	loss_l1_sp.bind(1, &image.image_vulkan());
	loss_l1_sp.bind(2, loss_out);
	loss_l1_sp.bind(3, loss_per_pixel_out);
	loss_l1_sp.bind(4, &buf);
	loss_l1_sp.invoke_compute(((int)image.size() + 1023) / 1024, 1, 1);
}

void shader_store::project_gaussians(glm::mat4* world_to_camera, glm::vec4* intrinsics, glm::vec4* limits, array_buffer<gsplat::data::gaussian2d>* gaussian2d_out) {

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

	array_buffer<settings_t> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = 1,
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			project_gaussians_sp.queue()
		}
	}};

	{
		auto l = buf.memory_lock();
		l.data()->camera.world_to_camera = *world_to_camera;
		l.data()->camera.intrinsics = *intrinsics;
		l.data()->camera.limits = *limits;
		l.data()->gaussian_count = (int)database_v->gaussian_set().data().size();
	}

	project_gaussians_sp.bind(0, &(database_v->gaussian_set().buffer1_vulkan()));
	project_gaussians_sp.bind(1, gaussian2d_out);
	project_gaussians_sp.bind(2, &buf);
	project_gaussians_sp.invoke_compute(((int)database_v->gaussian_set().data().size() + 1023) / 1024, 1, 1);
}
void shader_store::project_backwards(
	glm::mat4* world_to_camera,
	glm::vec4* intrinsics,
	glm::vec4* limits,
	array_buffer<float>* raster_grads) {
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

	array_buffer<settings_t> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_storage_buffer = true,
		.size = 1,
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			loss_l1_sp.queue()
		}
	}};


	{
		auto l = buf.memory_lock();
		l.data()->camera.world_to_camera = *world_to_camera;
		l.data()->camera.intrinsics = *intrinsics;
		l.data()->camera.limits = *limits;
		l.data()->gaussian_count = (int)database_v->gaussian_set().data().size();
	}

	project_backwards_sp.bind(0, &(database_v->gaussian_set().buffer1_vulkan()));
	project_backwards_sp.bind(1, raster_grads);
	project_backwards_sp.bind(2, &(database_v->gaussian_set().buffer2_vulkan()));
	project_backwards_sp.bind(3, &buf);
	project_backwards_sp.invoke_compute(((int)database_v->gaussian_set().data().size() + 1023) / 1024, 1, 1);
}

void shader_store::sort_gaussians(array_buffer<gsplat::data::gaussian2d>* gaussians_2d, array_buffer<int>* sorted_index) {

	struct settings_t {
		int odd_even_state;
		int buffer_size;
	};

	int n = (int)sorted_index->size();

	// one settings buffer per kind of pass, so nothing gets rewritten between passes
	array_buffer<settings_t> buf[2];
	for (int state = 0; state < 2; state++) {
		buf[state] = array_buffer<settings_t>{buffer_creation_settings{
			.usage_transfer_src = true,
			.usage_transfer_dst = true,
			.usage_storage_buffer = true,
			.size = 1,
			.memory = {
				device_memory_settings{
					.memory_host_visible = true,
					.memory_host_coherent = true,
				}
			},
			.queues = {
				sort_gaussians_sp.queue()
			}
		}};

		auto l = buf[state].memory_lock();
		l.data()->odd_even_state = state;
		l.data()->buffer_size = n;
	}

	sort_gaussians_sp.bind(0, gaussians_2d);
	sort_gaussians_sp.bind(1, sorted_index);

	// odd-even transition sort, GPU Gems 2 chapter 46.2. passes alternate between
	// even pairs (0,1) (2,3) ... and odd pairs (1,2) (3,4) ..., so no two threads
	// touch the same element. an element moves at most one slot per pass, so
	// after n passes everything is in place.
	for (int pass = 0; pass < n; pass++) {
		sort_gaussians_sp.bind(2, &buf[pass % 2]);
		sort_gaussians_sp.invoke_compute((n / 2 + 1023) / 1024, 1, 1);
	}
}

void shader_store::raster_forward(
	int target_image,
	glm::vec4* background_color,
	float near_plane,
	array_buffer<gsplat::data::gaussian2d>* gaussians_2d,
	array_buffer<int>* sorted_index,
	array_buffer<glm::vec4>* rastor_image_out,
	array_buffer<pixel_state>* pixel_state_out) {

	struct settings_t {
		glm::vec4 background; // rgb, 0..1
		int width;
		int height;
		int gaussian_count;
		float near_plane;
	};

	array_buffer<settings_t> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = 1,
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			raster_forward_sp.queue()
		}
	}};
	auto& image = database_v->image_set().data()[target_image];

	{
		auto l = buf.memory_lock();
		l.data()->background = *background_color;
		l.data()->width = image.width();
		l.data()->height = image.height();
		l.data()->gaussian_count = (int)database_v->gaussian_set().data().size();
		l.data()->near_plane = near_plane;
	}

	raster_forward_sp.bind(0, gaussians_2d);
	raster_forward_sp.bind(1, &(database_v->gaussian_set().buffer1_vulkan()));
	raster_forward_sp.bind(2, sorted_index);
	raster_forward_sp.bind(3, rastor_image_out);
	raster_forward_sp.bind(4, pixel_state_out);
	raster_forward_sp.bind(5, &buf);
	raster_forward_sp.invoke_compute(((int)image.size() + 1023) / 1024, 1, 1);
}

void shader_store::raster_backwards(
	int target_image,
	glm::vec4* background_color,
	float near_plane,
	array_buffer<gsplat::data::gaussian2d>* gaussians_2d,
	array_buffer<int>* sorted_index,
	array_buffer<pixel_state>* pixel_state_in,
	array_buffer<glm::vec4>* loss,
	array_buffer<float>* raster_grads_out) {

	// same settings as raster_forward
	struct settings_t {
		glm::vec4 background; // rgb, 0..1
		int width;
		int height;
		int gaussian_count;
		float near_plane;
	};

	array_buffer<settings_t> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = 1,
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			raster_backwards_sp.queue()
		}
	}};
	auto& image = database_v->image_set().data()[target_image];

	{
		auto l = buf.memory_lock();
		l.data()->background = *background_color;
		l.data()->width = image.width();
		l.data()->height = image.height();
		l.data()->gaussian_count = (int)database_v->gaussian_set().data().size();
		l.data()->near_plane = near_plane;
	}

	raster_backwards_sp.bind(0, gaussians_2d);
	raster_backwards_sp.bind(1, &(database_v->gaussian_set().buffer1_vulkan()));
	raster_backwards_sp.bind(2, sorted_index);
	raster_backwards_sp.bind(3, pixel_state_in);
	raster_backwards_sp.bind(4, loss);
	raster_backwards_sp.bind(5, raster_grads_out);
	raster_backwards_sp.bind(6, &buf);
	raster_backwards_sp.invoke_compute(((int)image.size() + 1023) / 1024, 1, 1);
}
