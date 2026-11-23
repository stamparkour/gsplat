#include <gsplat/data/gaussian.h>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <gsplat/core/word_iterable.h>
#include <random>

using namespace gsplat::data;
using namespace gsplat::core;

static float default_scale = 0.02;
static float default_opacity = 0.1;
std::vector<gaussian> gsplat::data::gaussian_colmap_txt(const std::string& path_colmap_points3d_txt) {
	std::vector<gaussian> o{};

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

		o.push_back(gaus);
	}

	return o;
}