#include <gsplat/train/modify.h>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <stdexcept>

using namespace gsplat::data;

void gsplat::train::split_gaussians(std::vector<gaussian>& gaussians, std::span<const float> grad, float scene_extent, std::mt19937& rng) {
	if (grad.size() != gaussians.size()) throw std::runtime_error("split_gaussians: needs one gradient per gaussian");

	std::normal_distribution<float> normal{0.0f, 1.0f};
	std::vector<gaussian> out;
	out.reserve(gaussians.size());

	for (std::size_t i = 0; i < gaussians.size(); i++) {
		const gaussian& g = gaussians[i];
		glm::vec3 s = glm::abs(glm::vec3(g.scale));

		if (grad[i] < tau_pos) {
			out.push_back(g);
		}
		else if (std::max({s.x, s.y, s.z}) <= percent_dense * scene_extent) {
			// small: clone. the paper's code leaves the copy in place and training pulls them apart
			out.push_back(g);
			out.push_back(g);
		}
		else {
			// big: two children drawn from the parent's own gaussian, smaller by phi
			glm::vec4 q = g.quaternion; // x, y, z, w like the shaders
			glm::mat3 R = glm::mat3_cast(glm::normalize(glm::quat(q.w, q.x, q.y, q.z)));
			for (int c = 0; c < 2; c++) {
				gaussian child = g;
				glm::vec3 z{normal(rng), normal(rng), normal(rng)}; // braces keep the draws in order
				child.mean += glm::vec4(R * (s * z), 0.0f);
				child.scale /= phi;
				out.push_back(child);
			}
		}
	}
	gaussians = std::move(out);
}

void gsplat::train::remove_gaussians(std::vector<gaussian>& gaussians) {
	std::erase_if(gaussians, [](const gaussian& g) { return g.color.color.a < epsilon_alpha; });
}

void gsplat::train::modify_gaussians(database* db, std::span<const float> grad, float scene_extent, std::mt19937& rng) {
	auto& set = db->gaussian_set();
	set.fetch_vulkan(0);
	split_gaussians(set.data(), grad, scene_extent, rng);
	remove_gaussians(set.data());
	if (set.data().empty()) throw std::runtime_error("modify_gaussians: every gaussian was removed");
	set.resize_vulkan();
}
