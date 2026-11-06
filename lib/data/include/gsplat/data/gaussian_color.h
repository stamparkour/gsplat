#ifndef GSPLAT_DATA_GAUSSIAN_COLOR_H
#define GSPLAT_DATA_GAUSSIAN_COLOR_H

#include <gsplat/data/configure.h>
#include <cereal/cereal.hpp>

// cereal does not know how to serialize glm types, so teach it.
// These live in namespace cereal so cereal's overload lookup finds them.
// Everything is written as plain floats, column by column, so the file format
// does not depend on glm's internal layout or padding.
namespace cereal {
	template <class Archive, glm::qualifier Q>
	void serialize(Archive& ar, glm::vec<4, float, Q>& v) {
		ar(v.x, v.y, v.z, v.w);
	}

	template <class Archive, glm::qualifier Q>
	void serialize(Archive& ar, glm::mat<4, 4, float, Q>& m) {
		ar(m[0], m[1], m[2], m[3]);
	}
}

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

		// cereal reflection: list every member that must be saved/loaded.
		// If you add a member above, add it here too (and bump the file version).
		template <class Archive>
		void serialize(Archive& ar) {
			ar(color);
		}
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_COLOR_H
