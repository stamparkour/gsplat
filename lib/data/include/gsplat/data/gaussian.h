#ifndef GSPLAT_DATA_GAUSSIAN_H
#define GSPLAT_DATA_GAUSSIAN_H

#include <gsplat/data/configure.h>
#include <gsplat/data/gaussian_color.h>

namespace gsplat::data {
	// Vulkan aligned
	struct gaussian {
		glm::mat4 covariance_inv; // 3x3 but requires 4x4 alignment
		glm::vec4 mean;
		glm::vec4 quaternion;
		glm::vec4 scale; // only uses 3 values
		gaussian_color color;

		float mag2() {
			float m = 0;
			m += glm::dot(mean, mean);
			m += glm::dot(quaternion, quaternion);
			m += glm::dot(scale, scale);
			m += color.mag2();
			return m;
		}
	};
	// Vulkan aligned
	struct gaussian2d {
		int source_index;
		float depth;
		alignas(sizeof(glm::vec2)) glm::mat2 covariance_inv;
		alignas(sizeof(glm::vec2)) glm::vec2 mean;
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_H
