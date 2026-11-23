#ifndef GSPLAT_DATA_IMAGE_H
#define GSPLAT_DATA_IMAGE_H

#include <gsplat/data/configure.h>
#include <gsplat/vulkan/memory.h>
#include <vector>
#include <cstdint>

namespace gsplat::data {
	// Vulkan aligned
	struct alignas(sizeof(std::uint8_t) * 4) pixel {
		std::uint8_t r;
		std::uint8_t g;
		std::uint8_t b;
		std::uint8_t a;
	};

	struct image_pose {
		glm::vec4 position;
		glm::vec4 quaternion;
		float fx;
		float fy;
		float cx;
		float cy;
	};

	// should be standardized to a something like 320x240
	class image {
		std::vector<pixel> data_v{};
		std::size_t width_v{};
		std::size_t height_v{};
		gsplat::vulkan::buffer image_v{};
		image_pose pose_v{};
		std::string name_v{};
	public:
		image() = default;
		static image read_image(const std::string& path_images, const image_pose& pose);

		std::size_t width() const { return width_v; }
		std::size_t height() const { return height_v; }
		std::size_t size() const { return width_v * height_v; }

		gsplat::vulkan::buffer& image_vulkan() { return image_v; }
		const gsplat::vulkan::buffer& image_vulkan() const { return image_v; }
		std::vector<pixel>& data() { return data_v; }
		const std::vector<pixel>& data() const { return data_v; }
		image_pose& pose() { return pose_v; }
		const image_pose& pose() const { return pose_v; }

		// data() to image_vulkan() and back. read_image doesn't upload, train
		// makes its own target buffers from data()
		stamp::task::coroutine<void> store_vulkan();
		stamp::task::coroutine<void> fetch_vulkan();
		const std::string& name() { return name_v; }
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_H
