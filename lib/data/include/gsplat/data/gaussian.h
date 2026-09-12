#ifndef GSPLAT_DATA_GAUSSIAN_H
#define GSPLAT_DATA_GAUSSIAN_H

#include <gsplat/data/configure.h>

namespace gsplat::data {
	// Vulkan aligned
	struct gaussian {
		glm::mat4 covariance;
		glm::vec4 mean;
	};
	// Vulkan aligned
	struct gaussian2d {
		glm::mat2 covariance;
		glm::vec2 mean;
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_H
