#include <gsplat/data/gaussian.h>
#include <gsplat/data/gaussian_set.h>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <gsplat/core/word_iterable.h>
#include <random>

using namespace gsplat::data;
using namespace gsplat::core;
using namespace gsplat::vulkan;

static float default_scale = 0.02;
static float default_opacity = 0.1;

gaussian_set gaussian_set::colmap_txt(const std::string& path_colmap_points3d_txt, gsplat::vulkan::command_queue* q) {
	gaussian_set o{};
	o.queue_v = q;

	std::ifstream file{path_colmap_points3d_txt};
	if (!file) throw std::runtime_error("failed to open colmap points3d file");

	std::string str;
	std::vector<std::string> words{};
	while (true) {
		std::getline(file, str);
		if (!file) break;
		words.clear();
		for (const auto& v : word_iterable{str}) {
			words.push_back(std::string{v});
		}

		if (words.empty()) continue;
		if (words[0].empty() || words[0][0] == '#') continue;

		gaussian gaus{};
		float px = std::stof(words[1]);
		float py = std::stof(words[2]);
		float pz = std::stof(words[3]);
		float r = std::stof(words[4]) / 255;
		float g = std::stof(words[5]) / 255;
		float b = std::stof(words[6]) / 255;
		gaus.mean = {px,py,pz,0};
		gaus.quaternion = {0, 0, 0, 1};
		gaus.scale = {default_scale, default_scale, default_scale, 0};
		gaus.color.color = {r,g,b, default_opacity};

		o.gaussians_v.push_back(gaus);
	}

	o.buffers_v[0] = array_buffer<gaussian>{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = o.gaussians_v.size(),
		.memory = {
			device_memory_settings{}
		},
		.queues = {
			o.queue_v,
		}
	}};
	o.buffers_v[1] = array_buffer<gaussian>{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = o.gaussians_v.size(),
		.memory = {
			device_memory_settings{}
		},
		.queues = {
			o.queue_v,
		}
	}};
	o.store_vulkan(0);
	o.store_vulkan(1);

	return o;
}

void gaussian_set::store_vulkan(int target) {
	array_buffer<gaussian> buf{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_storage_buffer = true,
		.size = gaussians_v.size(),
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
		std::copy(gaussians_v.begin(), gaussians_v.end(), l.begin());
	}

	this->buffers_v[(active_buffer + target) % buffer_count].transfer(&buf);
}

void gsplat::data::gaussian_set::filter_gaussians() {
	fetch_vulkan(1);

	std::vector<int> oper{};
	oper.resize(this->gaussians_v.size());

	int mod_count = 0;

	for (int i = 0; i < oper.size(); i++) {
		if (gaussians_v[i].mag2() > 0.05) {
			oper[i] = 1;
			mod_count++;
		}
	}
	fetch_vulkan(0);
	for (int i = 0; i < oper.size(); i++) {
		if (gaussians_v[i].color.color.w < 0.01) {
			if (oper[i] == 1) mod_count--;
			oper[i] = -1;
			mod_count--;
		}
	}

	if (mod_count == 0) return;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distrib(-0.1, 0.1);

	for (int i = 0, j = 0; i < oper.size(); i++, j++) {
		if (oper[i] == 1) {
			gaussians_v[j].scale *= 0.7;
			auto n = gaussians_v[j];
			n.mean += glm::vec4(distrib(gen), distrib(gen), distrib(gen), 0);
			gaussians_v.push_back(n);
		}
		else if (oper[i] == -1) {
			gaussians_v.erase(gaussians_v.begin() + j);
			j--;
		}
	}

	buffers_v[0] = array_buffer<gaussian>{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = gaussians_v.size(),
		.memory = {
			device_memory_settings{}
		},
		.queues = {
			queue_v,
		}
	}};
	buffers_v[1] = array_buffer<gaussian>{buffer_creation_settings{
		.usage_transfer_src = true,
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = gaussians_v.size(),
		.memory = {
			device_memory_settings{}
		},
		.queues = {
			queue_v,
		}
	}};
	store_vulkan(0);
	store_vulkan(1);
}

void gaussian_set::fetch_vulkan(int target) {
	array_buffer<gaussian> buf{buffer_creation_settings{
		.usage_transfer_dst = true,
		.usage_storage_buffer = true,
		.size = gaussians_v.size(),
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

	buf.transfer(&this->buffers_v[(active_buffer + target) % buffer_count]);

	{
		auto l = buf.memory_lock();
		std::copy(l.begin(), l.end(), gaussians_v.begin());
	}
}