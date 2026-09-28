#ifndef GSPLAT_DATA_GAUSSIAN_H
#define GSPLAT_DATA_GAUSSIAN_H

#include <gsplat/data/configure.h>

namespace gsplat::data {
	// Vulkan aligned
	struct gaussian {
		glm::mat4 covariance; // 3x3 but requires 4x4 alignment
		glm::vec4 mean;
		glm::vec4 quaternion;
		glm::vec4 scale; // only uses 3 values
	};
	// Vulkan aligned
	struct gaussian2d {
		glm::mat2 covariance;
		glm::vec2 mean;
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_H
