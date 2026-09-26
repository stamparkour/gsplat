#include <gsplat/vulkan/shader.h>
#include <iterator>

using namespace gsplat::vulkan;

shader::shader(const shader_creation_settings& settings) {
	device_v = settings.device->device_handle();
	logical_device_v = settings.device;

	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = settings.file_size;
	createInfo.pCode = reinterpret_cast<const uint32_t*>(settings.file_ptr);
	if (vkCreateShaderModule(device_v, &createInfo, nullptr, &shader_v) != VK_SUCCESS) {
		throw std::runtime_error("failed to create shader module!");
	}
}

shader::~shader() {
	vkDestroyShaderModule(device_v, shader_v, nullptr);
}

VkShaderModule shader::shader_handle() const {
	return shader_v;
}
VkDevice shader::device_handle() const {
	return device_v;
}

compute_shader_pipeline::compute_shader_pipeline(const compute_shader_pipeline_settings& settings) {
	this->device_v = settings.queue->logical_device()->device_handle();

	std::vector<VkDynamicState> dynamicStates = {};

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();

	std::vector<VkDescriptorSetLayoutBinding> layoutBindings{};
	VkDescriptorSetLayoutBinding* tmp_layout;
	for (int i = 0; i < settings.storage_buffer_count; i++) {
		tmp_layout = &layoutBindings.emplace_back();
		tmp_layout->binding = i;
		tmp_layout->descriptorCount = 1;
		tmp_layout->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		tmp_layout->pImmutableSamplers = nullptr;
		tmp_layout->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = layoutBindings.size();
	layoutInfo.pBindings = layoutBindings.data();

	if (vkCreateDescriptorSetLayout(device_v, &layoutInfo, nullptr, &desc_layout_v) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute descriptor set layout!");
	}

	VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
	computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	computeShaderStageInfo.module = settings.shader->shader_handle();
	computeShaderStageInfo.pName = "main";

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &desc_layout_v;

	if (vkCreatePipelineLayout(device_v, &pipelineLayoutInfo, nullptr, &pipeline_layout_v) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute pipeline layout!");
	}

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.layout = computePipelineLayout;
	pipelineInfo.stage = computeShaderStageInfo;

	if (vkCreateComputePipelines(device_v, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute pipeline!");
	}
}

gsplat::vulkan::compute_shader_pipeline::~compute_shader_pipeline() {
	vkDestroyDescriptorSetLayout(device_v, layout_v, nullptr);
}