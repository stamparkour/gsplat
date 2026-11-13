#include <gsplat/train.h>
#include <gsplat/train/modify.h>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <filesystem>
#include <stb_image_write.h>
#include <limits>
#include <random>

using namespace gsplat::train;
using namespace gsplat::vulkan;
using namespace gsplat::data;

// every working buffer is made the same way, only the type and the size change.
// every field is spelled out because gcc's -Wextra flags designated
// initializers that skip one
template<typename T>
static array_buffer<T> make_buffer(command_queue* q, std::size_t size) {
	return array_buffer<T>{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_uniform_buffer = false,
		.usage_storage_buffer = true,
		.size = size,
		.min_alignment = 0, // array_buffer sets it to alignof(T)
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
				.memory_device_local = false,
				.type_bitmask = 0,
			}
		},
		.queues = {
			q
		}
	}};
}

// swap instead of assigning, buffer's move assignment doesn't free the one it replaces
template<typename T>
static void remake_buffer(array_buffer<T>& buf, command_queue* q, std::size_t size) {
	auto fresh = make_buffer<T>(q, size);
	std::swap(buf, fresh);
}

// the paper's scene size: 1.1 times the farthest camera from the cameras' average position
static float scene_extent(const std::vector<camera>& cameras) {
	std::vector<glm::vec3> centers;
	glm::vec3 mean{0};
	for (auto& c : cameras) {
		centers.push_back(glm::vec3(glm::inverse(c.world_to_camera)[3]));
		mean += centers.back();
	}
	mean /= (float)centers.size();
	float farthest = 0;
	for (auto& p : centers) farthest = std::max(farthest, glm::length(p - mean));
	return 1.1f * farthest;
}

camera gsplat::train::make_camera(const image_pose& pose, glm::vec4 intrinsics, int width, int height, float near_plane) {
	// images.txt lists QW QX QY QZ and image_set::colmap_txt keeps that order,
	// so the real part is in .x here (gaussian_t keeps it last)
	const glm::vec4& q = pose.quaternion;
	glm::mat4 world_to_camera = glm::mat4_cast(glm::normalize(glm::quat(q.x, q.y, q.z, q.w))); // glm::quat takes (w, x, y, z)
	// COLMAP's TX TY TZ is the world to camera translation, not where the camera is
	world_to_camera[3] = glm::vec4(glm::vec3(pose.position), 1.0f);

	// 1.3 tan(fov/2), with tan(fovx/2) = (W/2) / fx
	float limx = 1.3f * (0.5f * width) / intrinsics.x;
	float limy = 1.3f * (0.5f * height) / intrinsics.y;
	return camera{world_to_camera, intrinsics, glm::vec4(limx, limy, near_plane, 0.0f)};
}

void gsplat::train::record_epoch_loss(int epoch, float loss) {
	std::cout << "epoch " << epoch << "  loss " << loss << std::endl;
}

void gsplat::train::train(shader_store* shaders, database* db, command_queue* q, const train_settings& ts_1) {
	train_settings ts = ts_1;
	auto& images = db->image_set().data();
	if (images.empty()) throw std::runtime_error("train: no images in the database");
	if (ts.intrinsics.x <= 0 || ts.intrinsics.y <= 0) throw std::runtime_error("train: set train_settings::intrinsics (fx, fy, cx, cy), nothing loads cameras.txt yet");
	if (ts.images_per_step < 1) throw std::runtime_error("train: images_per_step must be at least 1");

	int n = (int)db->gaussian_set().size();
	const int width = (int)images[0].width();
	const int height = (int)images[0].height();
	const int pixels = width * height;
	if (pixels == 0) throw std::runtime_error("train: images have no size");
	for (auto& img : images) {
		if ((int)img.width() != width || (int)img.height() != height) throw std::runtime_error("train: every image has to be the same size");
	}

	// project_backwards adds into buffer2 and apply_gradient reads it from there
	auto& gradients = db->gaussian_set().buffer2_vulkan();

	// per gaussian, remade whenever modify_gaussians changes the count
	array_buffer<gaussian2d> gaussians_2d;
	array_buffer<int> sorted_index;
	array_buffer<float> raster_grads;
	std::vector<float> grad_sum, grad_views; // for modify_gaussians
	auto make_gaussian_buffers = [&] {
		n = (int)db->gaussian_set().size();
		remake_buffer(gaussians_2d, q, n);
		remake_buffer(sorted_index, q, n);
		remake_buffer(raster_grads, q, (std::size_t)n * 9); // RASTER_GRAD_STRIDE in splat_math.h
		{
			// 0, 1, ..., n-1. the sort permutes it in place after that
			auto l = sorted_index.memory_lock();
			std::iota(l.begin(), l.end(), 0);
		}
		grad_sum.assign(n, 0.0f);
		grad_views.assign(n, 0.0f);
	};
	make_gaussian_buffers();
	// per pixel
	auto image_out = make_buffer<glm::vec4>(q, pixels);
	auto pixel_states = make_buffer<pixel_state>(q, pixels);
	auto dL_dC = make_buffer<glm::vec4>(q, pixels);
	auto loss_per_pixel = make_buffer<float>(q, pixels);

	std::vector<camera> cameras;
	for (auto& img : images) {
		cameras.push_back(make_camera(img.pose(), ts.intrinsics, width, height, ts.near_plane));
	}
	glm::vec4 background = ts.background;

	// gradients is a sum over images_per_step images, so dividing the
	// learning rate by it steps along their mean
	float step_rate = ts.learning_rate / ts.images_per_step;

	// no calc_covariance: project_gaussians builds Sigma from the quaternion and scale
	int images_in_step = 0;

	float min_loss = std::numeric_limits<float>::infinity();
	int time_since_last_min_loss = 0;

	const bool modify_on = ts.modify_interval > 0;
	const float extent = scene_extent(cameras);
	std::mt19937 rng{0}; // fixed seed so runs repeat

	for (int epoch = 0; epoch < ts.epochs; epoch++) {
		float epoch_loss = 0;
		for (int k = 0; k < (int)images.size(); k++) {
			camera& cam = cameras[k];
			if (images_in_step == 0) {
				gradients.clear();
			}

			// forward
			shaders->calc_covariance();
			shaders->project_gaussians(&cam.world_to_camera, &cam.intrinsics, &cam.limits, &gaussians_2d);
			shaders->sort_gaussians(&gaussians_2d, &sorted_index);
			shaders->raster_forward(k, &background, ts.near_plane, &gaussians_2d, &sorted_index, &image_out, &pixel_states);
			shaders->loss_l1(k, &image_out, &dL_dC, &loss_per_pixel);
			{
				// each pixel's share is already divided by 3 x pixels, so the sum is the mean
				auto l = loss_per_pixel.memory_lock();
				epoch_loss += std::accumulate(l.begin(), l.end(), 0.0f);
			}

			// backward. raster_grads are 2D and belong to this image's camera, so
			// they start from zero every image. buffer2 is 3D and keeps adding up.
			raster_grads.clear();
			shaders->raster_backwards(k, &background, ts.near_plane, &gaussians_2d, &sorted_index, &pixel_states, &dL_dC, &raster_grads);
			shaders->project_backwards(&cam.world_to_camera, &cam.intrinsics, &cam.limits, &raster_grads);

			if (modify_on) {
				// the paper's statistic: length of d loss / d 2D mean in NDC, which is the
				// pixel gradient times W/2 and H/2. a gaussian with no gradient didn't get seen
				auto l = raster_grads.memory_lock();
				const float* rg = l.data();
				for (int i = 0; i < n; i++) {
					float g = glm::length(glm::vec2(rg[i * 9 + 4] * 0.5f * width, rg[i * 9 + 5] * 0.5f * height));
					if (g > 0) {
						grad_sum[i] += g;
						grad_views[i]++;
					}
				}
			}

			if (++images_in_step == ts.images_per_step) {
				shaders->apply_gradient(step_rate);
				// db->gaussian_set().filter_gaussians();
				// std::cout << "gaussian_count: " << db->gaussian_set().data().size() << std::endl;
				images_in_step = 0;
			}
		}
		record_epoch_loss(epoch, epoch_loss / images.size());

		time_since_last_min_loss++;
		if (epoch_loss < min_loss) {
			min_loss = epoch_loss;
			time_since_last_min_loss = 0;
		}
		if (time_since_last_min_loss > 2) {
			step_rate *= 0.5;
			time_since_last_min_loss = 0;
			std::cout << "learning rate: " << step_rate << std::endl;
		}

		if (!ts.export_path.empty()) export_png(ts.export_path + "out" + std::to_string(epoch) + ".png", &image_out, db->image_set().data()[images.size()-1].width(), db->image_set().data()[images.size() - 1].height());

		// only in the first half of training, like the paper (densifies until 15k of 30k iterations)
		if (modify_on && (epoch + 1) % ts.modify_interval == 0 && epoch + 1 <= ts.epochs / 2) {
			std::vector<float> grad(n);
			for (int i = 0; i < n; i++) grad[i] = grad_views[i] > 0 ? grad_sum[i] / grad_views[i] : 0.0f;

			modify_gaussians(db, grad, extent, rng);
			std::cout << "gaussians: " << n << " -> " << db->gaussian_set().size() << std::endl;
			make_gaussian_buffers();
			// resize_vulkan wrote over the gradient sum, so the half finished step is dropped
			images_in_step = 0;
		}
	}
}

void my_write_func(void* context, void* data, int size) {
	std::ofstream* stream = (std::ofstream*)context;
	stream->write((const char*)data, size);
}

void gsplat::train::export_png(const std::string& out_path, gsplat::vulkan::array_buffer<glm::vec4>* buf, int width, int height) {
	std::vector<unsigned char> vec{};
	vec.resize(width * height * 4);

	{
		auto l = buf->memory_lock();
		int s = buf->size();
		for (std::size_t i = 0; i < s; i++) {
			glm::vec4 v = l.data()[i];
			v.x = std::clamp<float>(v.x, 0, 1);
			v.y = std::clamp<float>(v.y, 0, 1);
			v.z = std::clamp<float>(v.z, 0, 1);
			v.w = std::clamp<float>(v.w, 0, 1);
			vec[i * 4 + 0] = (unsigned char)(v.x * 255);// R
			vec[i * 4 + 1] = (unsigned char)(v.y * 255);// G
			vec[i * 4 + 2] = (unsigned char)(v.z * 255);// B
			vec[i * 4 + 3] = (unsigned char)(v.w * 255);// A
		}
	}
	std::filesystem::path path = out_path;
	path.remove_filename();
	std::filesystem::create_directories(path);
	//std::ofstream stream{out_path};

	if (!stbi_write_png(out_path.c_str(), width, height, 4, vec.data(), width*4)) {
		throw std::runtime_error("failed to export image");
	}
}