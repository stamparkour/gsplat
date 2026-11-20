#include <gsplat/train.h>
#include <gsplat/train/modify.h>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <filesystem>
#include <stb_image_write.h>
#include <limits>
#include <random>

using namespace gsplat::train;
using namespace gsplat::vulkan;
using namespace gsplat::data;

// the buffers sized by the gaussian count, remade whenever modify_gaussians
// changes it. sorted_index starts as 0, 1, ..., n-1 and the sort permutes it in
// place after that
static stamp::task::coroutine<void> make_gaussian_buffers(int n, buffer& gaussians_2d, buffer& sorted_index, buffer& raster_grads) {
	gaussians_2d = buffer{n * sizeof(gaussian2d)};
	std::vector<int> index(n);
	std::iota(index.begin(), index.end(), 0);
	sorted_index = co_await buffer::from_vector(index);
	raster_grads = buffer{n * 9 * sizeof(float)}; // RASTER_GRAD_STRIDE in splat_math.h
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

stamp::task::coroutine<void> gsplat::train::train(shader_store* shaders, database* db, train_settings ts) {
	auto& images = db->image_set().data();
	if (images.empty()) throw std::runtime_error("train: no images in the database");
	if (ts.intrinsics.x <= 0 || ts.intrinsics.y <= 0) throw std::runtime_error("train: set train_settings::intrinsics (fx, fy, cx, cy), nothing loads cameras.txt yet");
	if (ts.images_per_step < 1) throw std::runtime_error("train: images_per_step must be at least 1");

	int n = (int)db->gaussian_set().data().size();
	if (n == 0) throw std::runtime_error("train: no gaussians in the database");
	const int width = (int)images[0].width();
	const int height = (int)images[0].height();
	const int pixels = width * height;
	if (pixels == 0) throw std::runtime_error("train: images have no size");
	for (auto& img : images) {
		if ((int)img.width() != width || (int)img.height() != height) throw std::runtime_error("train: every image has to be the same size");
	}

	// the gaussians stay on the GPU for the whole run. project_backwards adds
	// into gradients and apply_gradient reads them from there
	buffer gaussians = co_await buffer::from_vector(db->gaussian_set().data());
	buffer gradients{gaussians.size()};

	// the photos go up once
	std::vector<buffer> targets;
	targets.reserve(images.size());
	for (auto& img : images) {
		targets.push_back(co_await buffer::from_vector(img.data()));
	}

	buffer gaussians_2d, sorted_index, raster_grads;
	co_await make_gaussian_buffers(n, gaussians_2d, sorted_index, raster_grads);
	std::vector<float> grad_sum(n), grad_views(n); // for modify_gaussians
	// per pixel
	buffer image_out{pixels * sizeof(glm::vec4)};
	buffer pixel_states{pixels * sizeof(pixel_state)};
	buffer dL_dC{pixels * sizeof(glm::vec4)};
	buffer loss_per_pixel{pixels * sizeof(float)};

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
	// modify_gaussians runs after epochs interval, 2 interval, ... up to half the
	// epochs, so the gradient statistic is only needed until the last of those
	const int last_modify = modify_on ? ts.epochs / 2 / ts.modify_interval * ts.modify_interval : 0;
	const float extent = scene_extent(cameras);
	std::mt19937 rng{0}; // fixed seed so runs repeat

	for (int epoch = 0; epoch < ts.epochs; epoch++) {
		float epoch_loss = 0;
		for (int k = 0; k < (int)images.size(); k++) {
			camera& cam = cameras[k];
			if (images_in_step == 0) {
				co_await gradients.clear();
			}

			// forward
			co_await shaders->calc_covariance(gaussians);
			co_await shaders->project_gaussians(cam.world_to_camera, cam.intrinsics, cam.limits, gaussians, gaussians_2d);
			co_await shaders->sort_gaussians(gaussians_2d, sorted_index);
			co_await shaders->raster_forward(width, height, background, ts.near_plane, gaussians_2d, gaussians, sorted_index, image_out, pixel_states);
			co_await shaders->loss_l1(width, height, image_out, targets[k], dL_dC, loss_per_pixel);
			{
				// each pixel's share is already divided by 3 x pixels, so the sum is the mean
				std::vector<float> l = co_await loss_per_pixel.to_vector<float>();
				epoch_loss += std::accumulate(l.begin(), l.end(), 0.0f);
			}

			// backward. raster_grads are 2D and belong to this image's camera, so
			// they start from zero every image. gradients are 3D and keep adding up.
			co_await raster_grads.clear();
			co_await shaders->raster_backwards(width, height, background, ts.near_plane, gaussians_2d, gaussians, sorted_index, pixel_states, dL_dC, raster_grads);
			co_await shaders->project_backwards(cam.world_to_camera, cam.intrinsics, cam.limits, gaussians, raster_grads, gradients);

			if (epoch < last_modify) {
				// the paper's statistic: length of d loss / d 2D mean in NDC, which is the
				// pixel gradient times W/2 and H/2. a gaussian with no gradient didn't get seen.
				// raster_grads is cleared every image, so this is one bulk read per image
				std::vector<float> rg = co_await raster_grads.to_vector<float>();
				for (int i = 0; i < n; i++) {
					float g = glm::length(glm::vec2(rg[i * 9 + 4] * 0.5f * width, rg[i * 9 + 5] * 0.5f * height));
					if (g > 0) {
						grad_sum[i] += g;
						grad_views[i]++;
					}
				}
			}

			if (++images_in_step == ts.images_per_step) {
				co_await shaders->apply_gradient(step_rate, gaussians, gradients);
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

		if (!ts.export_path.empty()) {
			std::vector<glm::vec4> last_render = co_await image_out.to_vector<glm::vec4>();
			export_png(ts.export_path + "out" + std::to_string(epoch) + ".png", last_render, width, height);
		}

		// only in the first half of training, like the paper (densifies until 15k of 30k iterations)
		if (modify_on && (epoch + 1) % ts.modify_interval == 0 && epoch + 1 <= ts.epochs / 2) {
			std::vector<float> grad(n);
			for (int i = 0; i < n; i++) grad[i] = grad_views[i] > 0 ? grad_sum[i] / grad_views[i] : 0.0f;

			co_await modify_gaussians(gaussians, grad, extent, rng);
			int before = n;
			n = (int)(gaussians.size() / sizeof(gaussian));
			std::cout << "gaussians: " << before << " -> " << n << std::endl;

			gradients = buffer{gaussians.size()};
			co_await make_gaussian_buffers(n, gaussians_2d, sorted_index, raster_grads);
			grad_sum.assign(n, 0.0f);
			grad_views.assign(n, 0.0f);
			// gradients was remade, so the half finished step is dropped
			images_in_step = 0;
		}
	}

	db->gaussian_set().data() = co_await gaussians.to_vector<gaussian>();
}

void my_write_func(void* context, void* data, int size) {
	std::ofstream* stream = (std::ofstream*)context;
	stream->write((const char*)data, size);
}

void gsplat::train::export_png(const std::string& out_path, const std::vector<glm::vec4>& pixels, int width, int height) {
	std::vector<unsigned char> vec{};
	vec.resize(width * height * 4);

	{
		for (std::size_t i = 0; i < pixels.size(); i++) {
			glm::vec4 v = pixels[i];
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