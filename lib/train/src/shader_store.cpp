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

	vec = load_entire_file_binary(shader_dir + "/project_backwards.slang.spv");

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

	vec = load_entire_file_binary(shader_dir + "/raster_backwards.slang.spv");

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

void shader_store::project_gaussians(glm::mat4* camera_transform, glm::mat4* camera_projection, array_buffer<gsplat::data::gaussian2d>* gaussian2d_out) {

	struct settings_t {
		glm::mat4 camera_transform;  // world space to camera space
		glm::mat4 camera_transform_inv;  // camera space to world space
		glm::mat4 camera_projection; // camera space to clip space
		glm::mat4 camera_projection_inv; // clip space to camera space
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
			loss_l1_sp.queue()
		}
	}};


	{
		auto l = buf.memory_lock();
		l.data()->buffer_size = (int)database_v->gaussian_set().data().size();
		l.data()->camera_transform = *camera_transform;
		l.data()->camera_projection = *camera_projection;
		l.data()->camera_transform_inv = glm::inverse(*camera_transform);
		l.data()->camera_projection_inv = glm::inverse(*camera_projection);
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

//void shader_store::raster_backwards(
//	glm::vec4* background_color,
//	float near_plane,
//	gsplat::vulkan::array_buffer<gsplat::data::gaussian2d> gaussians_2d) {
//	struct camera_t {
//		glm::mat4 world_to_camera; // same matrix as project_gaussians' camera_transform
//		glm::vec4 intrinsics;        // fx, fy, cx, cy, in pixels
//		glm::vec4 limits;
//	};
//	struct settings_t {
//		camera_t camera;
//		int gaussian_count;
//		int pad0;
//		int pad1;
//		int pad2;
//	};
//
//	array_buffer<settings_t> buf{buffer_creation_settings{
//		.usage_transfer_src = true,
//		.usage_storage_buffer = true,
//		.size = 1,
//		.memory = {
//			device_memory_settings{
//				.memory_host_visible = true,
//				.memory_host_coherent = true,
//			}
//		},
//		.queues = {
//			loss_l1_sp.queue()
//		}
//	}};
//
//
//	{
//		auto l = buf.memory_lock();
//		l.data()->camera.world_to_camera = *world_to_camera;
//		l.data()->camera.intrinsics = *intrinsics;
//		l.data()->camera.limits = *limits;
//		l.data()->gaussian_count = (int)database_v->gaussian_set().data().size();
//	}
//
//	project_backwards_sp.bind(0, &(database_v->gaussian_set().buffer1_vulkan()));
//	project_backwards_sp.bind(1, raster_grads);
//	project_backwards_sp.bind(2, &(database_v->gaussian_set().buffer2_vulkan()));
//	project_backwards_sp.bind(3, &buf);
//	project_backwards_sp.invoke_compute(((int)database_v->gaussian_set().data().size() + 1023) / 1024, 1, 1);
//}