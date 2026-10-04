#include <gsplat/data/image.h>
#include <gsplat/data/image_set.h>
#include <stb_image.h>
#include <gsplat/core/word_iterable.h>
#include <fstream>
#include <filesystem>

using namespace gsplat::data;
using namespace gsplat::vulkan;
using namespace gsplat::core;

struct camera_intrinsics {
	float fx, fy, cx, cy;
};

image_set gsplat::data::image_set::colmap_txt(const std::string& images_dir, const std::string& images_txt_path, const std::string& cameras_txt_path, gsplat::vulkan::command_queue* q, int max_entries) {
	image_set o{};

	std::ifstream file2{cameras_txt_path};
	std::string str;
	std::vector<std::string> words{};
	if (!file2) throw std::runtime_error("failed to open colmap points3d file");

	std::vector<camera_intrinsics> intrinics{};

	while (true) {
		std::getline(file2, str);
		if (!file2) break;
		words.clear();
		for (const auto& v : word_iterable{str}) {
			words.push_back(std::string{v});
		}

		if (words.empty()) continue;
		if (words[0].empty() || words[0][0] == '#') continue;

		int l = std::stoi(words[0]);
		if (intrinics.size() <= l) {
			intrinics.resize((std::size_t)l + 1);
		}

		camera_intrinsics c{};
		if (words[1] == "PINHOLE" || words[1] == "RADIAL") {
			c.fx = std::stof(words[4]);
			c.fy = std::stof(words[5]);
			c.cx = std::stof(words[6]);
			c.cy = std::stof(words[7]);
		}
		else if (words[1] == "SIMPLE_PINHOLE" || words[1] == "SIMPLE_RADIAL") {
			c.fx = std::stof(words[4]);
			c.fy = c.fx;
			c.cx = std::stof(words[5]);
			c.cy = std::stof(words[6]);
		}
		intrinics[l] = c;
	}

	std::ifstream file{images_txt_path};
	if (!file) throw std::runtime_error("failed to open colmap points3d file");

	int line_type = 0;
	while (true) {
		std::getline(file, str);
		if (!file) break;
		words.clear();
		for (const auto& v : word_iterable{str}) {
			words.push_back(std::string{v});
		}

		if (words.empty()) continue;
		if (words[0].empty() || words[0][0] == '#') continue;

		// checking line_type

		// good
		if (line_type == 0) {
			line_type++;
		}
		// bad
		else if (line_type == 1) {
			line_type = 0;
			continue;
		}

		int camera_id = std::stoi(words[8]);

		image_pose pose{};
		pose.cx = intrinics[camera_id].cx;
		pose.cy = intrinics[camera_id].cy;
		pose.fx = intrinics[camera_id].fx;
		pose.fy = intrinics[camera_id].fy;
		pose.quaternion = {
			std::stof(words[1]),
			std::stof(words[2]),
			std::stof(words[3]),
			std::stof(words[4])
		};
		pose.position = {
			std::stof(words[5]),
			std::stof(words[6]),
			std::stof(words[7]),
			0
		};
		std::string path{images_dir + "/" + words[9]};

		o.images_v.push_back(image::read_image(path, pose, q));
		if (o.images_v.size() == max_entries) break;
	}

	return o;
}

image gsplat::data::image::read_image(const std::string& path, const image_pose& pose, gsplat::vulkan::command_queue* q) {
	image o{};
	o.queue_v = q;
	o.pose_v = pose;

	int width, height, n;
	unsigned char *data = stbi_load(path.c_str(), &width, &height, &n, 4);
	if (!data) throw std::runtime_error("failed to load image " + path);
	o.width_v = width;
	o.height_v = height;
	// ... process data if not NULL ...
	// ... x = width, y = height, n = # 8-bit components per pixel ...
	// ... replace '0' with '1'..'4' to force that many components per pixel
	// ... but 'n' will always be the number that it would have been if you said 0

	o.data_v.resize((std::size_t)width * (std::size_t)height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int i = x + y * width;
			pixel p{};
			p.r = data[i*4+0];
			p.g = data[i*4+1];
			p.b = data[i*4+2];
			p.a = data[i*4+3];
			o.data_v[i] = p;
		}
	}

	stbi_image_free(data);

	o.image_v = array_buffer<pixel>{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = o.data_v.size(),
		.memory = {
			device_memory_settings{}
		},
		.queues = {
			o.queue_v,
		}
	}};

	o.store_vulkan();

	std::filesystem::path p = path;
	o.name_v = p.filename().string();

	return o;
}

void image::store_vulkan() {
	array_buffer<pixel> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_storage_buffer = true,
		.size = data_v.size(),
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			queue_v,
		}
	}};

	{
		auto l = buf.memory_lock();
		std::copy(data_v.begin(), data_v.end(), l.begin());
	}

	image_v.transfer(&buf);
}

void image::fetch_vulkan() {
	array_buffer<pixel> buf{buffer_creation_settings{
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = data_v.size(),
		.memory = {
			device_memory_settings{
				.memory_host_visible = true,
				.memory_host_coherent = true,
			}
		},
		.queues = {
			queue_v,
		}
	}};

	buf.transfer(&image_v);

	{
		auto l = buf.memory_lock();
		std::copy(l.begin(), l.end(), data_v.begin());
	}
}