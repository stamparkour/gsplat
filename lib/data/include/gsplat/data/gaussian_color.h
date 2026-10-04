#ifndef GSPLAT_DATA_GAUSSIAN_COLOR_H
#define GSPLAT_DATA_GAUSSIAN_COLOR_H

#include <gsplat/data/configure.h>

namespace gsplat::data {
	// Vulkan aligned
	struct gaussian_color {
		glm::vec4 color; // r,g,b,a
		// additional arguments.

		float mag2() {
			float m = 0;
			m += glm::dot(color, color);
			return m;
		}
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_COLOR_H
