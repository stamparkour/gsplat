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
	this->queue_v = settings.queue;

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
	pipelineInfo.layout = pipeline_layout_v;
	pipelineInfo.stage = computeShaderStageInfo;

	if (vkCreateComputePipelines(device_v, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_v) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute pipeline!");
	}

	std::vector<VkDescriptorPoolSize> pool_sizes{1};
	pool_sizes[0].descriptorCount = settings.storage_buffer_count;
	pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

	VkDescriptorPoolCreateInfo desc_pool_info{};
	desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	desc_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
	desc_pool_info.maxSets = 1;
	desc_pool_info.poolSizeCount = pool_sizes.size();
	desc_pool_info.pPoolSizes = pool_sizes.data();

	if (vkCreateDescriptorPool(device_v, &desc_pool_info, nullptr, &desc_pool_v) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}

	VkDescriptorSetAllocateInfo desc_set_info{};
	desc_set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	desc_set_info.descriptorPool = desc_pool_v;
	desc_set_info.descriptorSetCount = 1;
	desc_set_info.pSetLayouts = &desc_layout_v;
	if (vkAllocateDescriptorSets(device_v, &desc_set_info, &desc_set_v) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}
}

gsplat::vulkan::compute_shader_pipeline::~compute_shader_pipeline() {
	if (desc_set_v) {
		vkFreeDescriptorSets(device_v, desc_pool_v, 1, &desc_set_v);
		desc_set_v = nullptr;
	}
	if (desc_pool_v) {
		vkDestroyDescriptorPool(device_v, desc_pool_v, nullptr);
		desc_pool_v = nullptr;
	}
	if (pipeline_v) {
		vkDestroyPipeline(device_v, pipeline_v, nullptr);
		pipeline_v = nullptr;
	}
	if (pipeline_layout_v) {
		vkDestroyPipelineLayout(device_v, pipeline_layout_v, nullptr);
		pipeline_layout_v = nullptr;
	}
	if (desc_layout_v) {
		vkDestroyDescriptorSetLayout(device_v, desc_layout_v, nullptr);
		desc_layout_v = nullptr;
	}
}

void compute_shader_pipeline::bind(int binding, gsplat::vulkan::basic_buffer* b) {
	VkDescriptorBufferInfo buffer_info{};
	buffer_info.buffer = b->buffer_handle();
	buffer_info.offset = 0;
	buffer_info.range = b->size();
	
	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = desc_set_v;
	write.dstBinding = binding;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.descriptorCount = 1;
	write.pBufferInfo = &buffer_info;
	vkUpdateDescriptorSets(device_v, 1, &write, 0, nullptr);
}

void compute_shader_pipeline::invoke_compute(int x, int y, int z) {
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = queue_v->pool_handle();
	allocInfo.commandBufferCount = 1;

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device_v, &allocInfo, &commandBuffer);

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("failed to begin recording command buffer!");
	}

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_v);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_v, 0, 1, &desc_set_v, 0, 0);

	vkCmdDispatch(commandBuffer, x, y, z);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;


	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to record command buffer!");
	}

	vkQueueSubmit(queue_v->queue_handle(), 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue_v->queue_handle());

	vkQueueWaitIdle(queue_v->queue_handle());

	vkFreeCommandBuffers(device_v, queue_v->pool_handle(), 1, &commandBuffer);
}