#include <gtest/gtest.h>
#include <gsplat/train/modify.h>
#include <cmath>

using namespace gsplat::train;
using gsplat::data::gaussian;

static gaussian test_gaussian(glm::vec3 scale, float alpha, glm::vec4 quaternion = {0, 0, 0, 1}) {
	gaussian g{};
	g.mean = {1, 2, 3, 0};
	g.quaternion = quaternion;
	g.scale = glm::vec4(scale, 0);
	g.color.color = {0.5f, 0.5f, 0.5f, alpha};
	return g;
}

// every test uses a scene extent of 10, so the clone or split line is a scale of 0.1
static const float extent = 10.0f;

TEST(modify, remove_drops_low_alpha) {
	std::vector<gaussian> gs{test_gaussian(glm::vec3(0.02f), 0.001f), test_gaussian(glm::vec3(0.02f), 0.5f)};
	remove_gaussians(gs);
	ASSERT_EQ(gs.size(), 1u);
	EXPECT_FLOAT_EQ(gs[0].color.color.a, 0.5f);
}

TEST(modify, low_gradient_is_kept) {
	std::vector<gaussian> gs{test_gaussian(glm::vec3(0.5f), 0.5f)};
	std::vector<float> grad{tau_pos / 2};
	std::mt19937 rng{0};
	split_gaussians(gs, grad, extent, rng);
	ASSERT_EQ(gs.size(), 1u);
	EXPECT_EQ(gs[0].mean, glm::vec4(1, 2, 3, 0));
}

TEST(modify, small_gaussian_is_cloned) {
	std::vector<gaussian> gs{test_gaussian(glm::vec3(0.02f), 0.5f)};
	std::vector<float> grad{tau_pos * 2};
	std::mt19937 rng{0};
	split_gaussians(gs, grad, extent, rng);
	ASSERT_EQ(gs.size(), 2u);
	for (auto& g : gs) {
		EXPECT_EQ(g.mean, glm::vec4(1, 2, 3, 0));
		EXPECT_FLOAT_EQ(g.scale.x, 0.02f);
	}
}

TEST(modify, big_gaussian_splits_in_two) {
	std::vector<gaussian> gs{test_gaussian(glm::vec3(0.5f), 0.5f)};
	std::vector<float> grad{tau_pos * 2};
	std::mt19937 rng{0};
	split_gaussians(gs, grad, extent, rng);
	ASSERT_EQ(gs.size(), 2u);
	for (auto& g : gs) {
		EXPECT_FLOAT_EQ(g.scale.x, 0.5f / phi);
		EXPECT_FLOAT_EQ(g.color.color.a, 0.5f);
		EXPECT_NE(g.mean, glm::vec4(1, 2, 3, 0));
	}
}

TEST(modify, split_follows_rotation) {
	// long along x, turned 90 degrees about z, so the children should land along y
	float h = std::sqrt(0.5f);
	std::vector<gaussian> gs(100, test_gaussian({1.0f, 0.01f, 0.01f}, 0.5f, {0, 0, h, h}));
	std::vector<float> grad(100, 1.0f);
	std::mt19937 rng{0};
	split_gaussians(gs, grad, extent, rng);
	ASSERT_EQ(gs.size(), 200u);
	glm::vec3 spread{0};
	for (auto& g : gs) spread += glm::abs(glm::vec3(g.mean) - glm::vec3(1, 2, 3));
	EXPECT_GT(spread.y, 10 * spread.x);
	EXPECT_GT(spread.y, 10 * spread.z);
}

TEST(modify, needs_one_gradient_per_gaussian) {
	std::vector<gaussian> gs{test_gaussian(glm::vec3(0.02f), 0.5f)};
	std::vector<float> grad{};
	std::mt19937 rng{0};
	EXPECT_THROW(split_gaussians(gs, grad, extent, rng), std::runtime_error);
}
