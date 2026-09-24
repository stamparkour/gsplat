#ifndef GSPLAT_VULKAN_BUFFER_H
#define GSPLAT_VULKAN_BUFFER_H

#include <gsplat/vulkan/config.h>
#include<vulkan/vulkan.h>
#include <gsplat/vulkan/device.h>
#include <iostream>

namespace gsplat::vulkan {
	class shader_descriptor_layout {

	};

	class basic_shader {
	protected:
		VkDevice device_v;
		VkShaderModule shader_v;
	public:
		basic_shader();
		virtual ~basic_shader();

		VkDevice device_handle();
		VkShaderModule shader_handle();
	};

	struct compue_shader_pipeline_settings {
		bool render_pipeline;
		bool compute_pipeline;
		bool use_fence;
	};
	class compue_shader_pipeline {
	public:
		compue_shader_pipeline(const compue_shader_pipeline_settings&);
		~compue_shader_pipeline();

		void invoke_compute(int x, int y, int z);
		void wait_fence();
	};

	struct shader_creation_settings {
		const char* begin_spirv;
		const char* end_spirv;
	};

	class shader : public basic_shader {
	public:
		shader(const shader_creation_settings&);
	};
}

#endif // GSPLAT_VULKAN_BUFFER_H
