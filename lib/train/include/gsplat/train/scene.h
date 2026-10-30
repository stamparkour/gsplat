#ifndef GSPLAT_TRAIN_SCENE_H
#define GSPLAT_TRAIN_SCENE_H

#include <gsplat/train.h>
#include <gsplat/data/gaussian.h>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace gsplat::train {
	// borrowed from vulkan_context, nothing here destroys them
	struct vulkan_handles {
		VkInstance instance = VK_NULL_HANDLE;
		VkPhysicalDevice physical_device = VK_NULL_HANDLE;
		VkDevice device = VK_NULL_HANDLE;
		VkQueue queue = VK_NULL_HANDLE;
		std::uint32_t queue_family_index = 0;

		static vulkan_handles from_context();
	};

	struct rendered_image {
		VkImage image = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
		VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		int width = 0;
		int height = 0;
	};

	// renders gaussians from any camera with the same shaders as training
	class scene {
		struct buffer_t {
			VkBuffer buffer = VK_NULL_HANDLE;
			VkDeviceMemory memory = VK_NULL_HANDLE;
			void* mapped = nullptr;
			VkDeviceSize size = 0;
		};
		struct pipeline_t {
			VkShaderModule module = VK_NULL_HANDLE;
			VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
			VkPipelineLayout layout = VK_NULL_HANDLE;
			VkPipeline pipeline = VK_NULL_HANDLE;
			VkDescriptorSet set = VK_NULL_HANDLE;
		};

		vulkan_handles vk_v;
		VkDescriptorPool descriptor_pool_v = VK_NULL_HANDLE;
		VkCommandPool command_pool_v = VK_NULL_HANDLE;
		VkCommandBuffer command_buffer_v = VK_NULL_HANDLE;
		VkFence fence_v = VK_NULL_HANDLE;
		pipeline_t project_v;
		pipeline_t raster_v;

		std::size_t gaussian_count_v = 0;
		std::vector<float> opacity_v;
		std::vector<int> order_v;
		buffer_t gaussians_v;
		buffer_t gaussians_2d_v;
		buffer_t sorted_index_v;
		buffer_t image_out_v;
		buffer_t pixel_state_v;
		buffer_t project_settings_v;
		buffer_t raster_settings_v;

		rendered_image image_v;
		VkDeviceMemory image_memory_v = VK_NULL_HANDLE;

		buffer_t make_buffer(VkDeviceSize size, VkMemoryPropertyFlags preferred);
		void free_buffer(buffer_t& b);
		pipeline_t make_pipeline(const std::string& spv_path, int storage_buffer_count);
		void free_pipeline(pipeline_t& p);
		void bind_buffers();
		void resize_image(int width, int height);
		void free_image();
		void submit_and_wait();
	public:
		scene(const vulkan_handles& vk, const std::string& shader_dir);
		~scene();

		scene(const scene&) = delete;
		scene(scene&&) = delete;
		scene& operator =(const scene&) = delete;
		scene& operator =(scene&&) = delete;

		void set_gaussians(std::span<const gsplat::data::gaussian> gaussians);
		std::size_t gaussian_count() const { return gaussian_count_v; }

		// waits for the GPU. the image is reused, so it's valid until the next render()
		rendered_image render(const camera& cam, int width, int height, glm::vec4 background = {0, 0, 0, 0});

		// the last render, rgba, row major from the top left
		std::span<const glm::vec4> pixels() const;
	};
}

#endif // GSPLAT_TRAIN_SCENE_H
