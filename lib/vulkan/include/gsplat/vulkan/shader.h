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
		int storage_buffer_count;
		command_queue* queue;
		gsplat::vulkan::shader* shader;
	};
	class compute_shader_pipeline {
		command_queue* queue_v;
		VkDescriptorSetLayout desc_layout_v;
		VkPipelineLayout pipeline_layout_v;
		VkDevice device_v;
		VkPipeline pipeline_v;
		VkDescriptorSet desc_set_v;
		VkDescriptorPool desc_pool_v;
	public:
		compute_shader_pipeline() = default;
		compute_shader_pipeline(const compute_shader_pipeline_settings&);
		~compute_shader_pipeline();

		compute_shader_pipeline(const compute_shader_pipeline&) = delete;
		compute_shader_pipeline(compute_shader_pipeline&& other) {
			compute_shader_pipeline::operator =(std::move(other));
		}
		compute_shader_pipeline& operator =(const compute_shader_pipeline&) = delete;
		compute_shader_pipeline& operator =(compute_shader_pipeline&& other) {
			queue_v = other.queue_v;
			desc_layout_v = other.desc_layout_v;
			other.desc_layout_v = nullptr;
			pipeline_layout_v = other.pipeline_layout_v;
			other.pipeline_layout_v = nullptr;
			device_v = other.device_v;
			pipeline_v = other.pipeline_v;
			other.pipeline_v = nullptr;
			desc_set_v = other.desc_set_v;
			other.desc_set_v = nullptr;
			desc_pool_v = other.desc_pool_v;
			other.desc_pool_v = nullptr;
			return *this;
		}

		void bind(int binding, gsplat::vulkan::basic_buffer*);
		void invoke_compute(int x, int y, int z);
		void wait_fence();

		command_queue* queue() const {
			return queue_v;
		}
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
		shader() = default;
		shader(const shader_creation_settings&);
		~shader();

		shader(const shader&) = delete;
		shader(shader&& other) {
			operator=(std::move(other));
		}
		shader& operator =(const shader&) = delete;
		shader& operator =(shader&& other) {
			this->shader_v = other.shader_v;
			other.shader_v = nullptr;
			this->device_v = other.device_v;
			this->logical_device_v = other.logical_device_v;

			return *this;
		}

		VkShaderModule shader_handle() const;
		VkDevice device_handle() const;
	};
}

#endif // GSPLAT_VULKAN_SHADER_H
