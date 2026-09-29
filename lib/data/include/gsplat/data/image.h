#ifndef GSPLAT_DATA_IMAGE_H
#define GSPLAT_DATA_IMAGE_H

#include <gsplat/data/configure.h>
#include <gsplat/vulkan/buffer.h>
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
	};

	// should be standardized to a something like 320x240
	class image {
		std::vector<pixel> data_v;
		std::size_t width_v;
		std::size_t height_v;
		gsplat::vulkan::array_buffer<pixel> image_v;
		gsplat::vulkan::command_queue* queue_v;
		image_pose pose_v;
	public:
		image() = default;
		static image read_image(std::string& path, const image_pose& pose, gsplat::vulkan::command_queue*);

		std::size_t width() const { return width_v; }
		std::size_t height() const { return height_v; }
		std::size_t size() const { return width_v * height_v; }

		gsplat::vulkan::array_buffer<pixel>& image_vulkan() { return image_v; }
		const gsplat::vulkan::array_buffer<pixel>& image_vulkan() const { return image_v; }
		std::vector<pixel>& data() { return data_v; }
		const std::vector<pixel>& data() const { return data_v; }
		image_pose& pose() { return pose_v; }
		const image_pose& pose() const { return pose_v; }

		void store_vulkan();
		void fetch_vulkan();
	};
}

#endif // GSPLAT_DATA_GAUSSIAN_H
