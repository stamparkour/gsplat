#ifndef GSPLAT_DATA_GAUSSIAN_COLOR_H
#define GSPLAT_DATA_GAUSSIAN_COLOR_H

#include <gsplat/data/configure.h>

namespace gsplat::data {
	// Vulkan aligned
	struct gaussian_color {
		glm::vec4 color; // r,g,b,a
		// additional arguments.
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_COLOR_H
