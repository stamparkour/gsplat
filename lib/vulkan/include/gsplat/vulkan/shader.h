#ifndef GSPLAT_VULKAN_SHADER_H
#define GSPLAT_VULKAN_SHADER_H

#include <gsplat/vulkan/config.h>
#include<vulkan/vulkan.h>
#include <gsplat/vulkan/device.h>
#include <gsplat/vulkan/buffer.h>
#include <iostream>

namespace gsplat::vulkan {
	class shader;

	struct compute_shader_pipeline_settings {
		bool use_fence;
		int storage_buffer_count;
		command_queue* queue;
		gsplat::vulkan::shader* shader;
	};
	class compute_shader_pipeline {
		command_queue* queue_v;
		VkDescriptorSetLayout desc_layout_v;
		VkPipelineLayout pipeline_layout_v;
		VkDevice device_v;
	public:
		compute_shader_pipeline(const compute_shader_pipeline_settings&);
		~compute_shader_pipeline();

		compute_shader_pipeline(const compute_shader_pipeline&) = delete;
		compute_shader_pipeline& operator =(const compute_shader_pipeline&) = delete;

		void bind(int location, gsplat::vulkan::basic_buffer*);
		void invoke_compute(int x, int y, int z);
		void wait_fence();
	};

	struct shader_creation_settings {
		const char* file_ptr;
		std::size_t file_size;
		logical_device* device;
	};

	class shader {
		VkShaderModule shader_v;
		VkDevice device_v;
		logical_device* logical_device_v;
	public:
		shader(const shader_creation_settings&);
		~shader();

		shader(const shader&) = delete;
		shader& operator =(const shader&) = delete;

		VkShaderModule shader_handle() const;
		VkDevice device_handle() const;
	};
}

#endif // GSPLAT_VULKAN_SHADER_H
