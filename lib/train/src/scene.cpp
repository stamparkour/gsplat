#include <gsplat/train/scene.h>
#include <gsplat/vulkan/vulkan2.h>
#include <gsplat/core/read_file.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace gsplat::train;
using gsplat::data::gaussian;
using gsplat::data::gaussian2d;

// the settings_t of project_gaussians.slang and raster_forward.slang
struct project_settings_t {
	camera cam;
	int gaussian_count;
	int pad0;
	int pad1;
	int pad2;
};
struct raster_settings_t {
	glm::vec4 background;
	int width;
	int height;
	int gaussian_count;
	float near_plane;
};
static_assert(sizeof(project_settings_t) == 112);
static_assert(sizeof(raster_settings_t) == 32);
static_assert(sizeof(gaussian) == 128);
static_assert(sizeof(gaussian2d) == 32);
static_assert(sizeof(pixel_state) == 8);

static void check(VkResult r, const char* what) {
	if (r != VK_SUCCESS) throw std::runtime_error(std::string{"scene: "} + what + " failed (VkResult " + std::to_string(r) + ")");
}

static VkDeviceMemory allocate(const vulkan_handles& vk, const VkMemoryRequirements& req, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
	VkPhysicalDeviceMemoryProperties props;
	vkGetPhysicalDeviceMemoryProperties(vk.physical_device, &props);
	for (VkMemoryPropertyFlags want : {required | preferred, required}) {
		for (std::uint32_t i = 0; i < props.memoryTypeCount; i++) {
			if (!(req.memoryTypeBits & (1u << i)) || (props.memoryTypes[i].propertyFlags & want) != want) continue;
			VkMemoryAllocateInfo info{};
			info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			info.allocationSize = req.size;
			info.memoryTypeIndex = i;
			VkDeviceMemory memory = VK_NULL_HANDLE;
			if (vkAllocateMemory(vk.device, &info, nullptr, &memory) == VK_SUCCESS) return memory;
		}
	}
	throw std::runtime_error("scene: out of GPU memory");
}

// past this radius opacity * G < 1/255, so the gaussian can't reach a pixel
static bool reaches_image(const gaussian2d& g, float opacity, int width, int height) {
	if (opacity * 255.0f < 1.0f) return false;
	const glm::mat2& Q = g.covariance_inv;
	float a = Q[0][0], b = Q[0][1], c = Q[1][1];
	float lambda_max = 0.5f * (a + c) + std::sqrt(0.25f * (a - c) * (a - c) + b * b);
	float lambda_min = (a * c - b * b) / lambda_max;
	if (lambda_min <= 0) return true;
	float r = std::sqrt(2.0f * std::log(255.0f * opacity) / lambda_min) + 1.0f;
	return g.mean.x + r >= 0 && g.mean.x - r <= (float)width && g.mean.y + r >= 0 && g.mean.y - r <= (float)height;
}

vulkan_handles vulkan_handles::from_context() {
	using gsplat::vulkan::vulkan_context;
	return vulkan_handles{
		.instance = vulkan_context::instance(),
		.physical_device = vulkan_context::physical_device(),
		.device = vulkan_context::device(),
		.queue = vulkan_context::queue(),
		.queue_family_index = (std::uint32_t)vulkan_context::queue_family_index(),
	};
}

scene::scene(const vulkan_handles& vk, const std::string& shader_dir) : vk_v(vk) {
	VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 + 6};
	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets = 2;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes = &pool_size;
	check(vkCreateDescriptorPool(vk.device, &pool_info, nullptr, &descriptor_pool_v), "vkCreateDescriptorPool");

	VkCommandPoolCreateInfo command_pool_info{};
	command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	command_pool_info.queueFamilyIndex = vk.queue_family_index;
	check(vkCreateCommandPool(vk.device, &command_pool_info, nullptr, &command_pool_v), "vkCreateCommandPool");

	VkCommandBufferAllocateInfo command_buffer_info{};
	command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	command_buffer_info.commandPool = command_pool_v;
	command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	command_buffer_info.commandBufferCount = 1;
	check(vkAllocateCommandBuffers(vk.device, &command_buffer_info, &command_buffer_v), "vkAllocateCommandBuffers");

	VkFenceCreateInfo fence_info{};
	fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	check(vkCreateFence(vk.device, &fence_info, nullptr, &fence_v), "vkCreateFence");

	project_v = make_pipeline(shader_dir + "/project_gaussians.slang.spv", 3);
	raster_v = make_pipeline(shader_dir + "/raster_forward.slang.spv", 6);
	project_settings_v = make_buffer(sizeof(project_settings_t), 0);
	raster_settings_v = make_buffer(sizeof(raster_settings_t), 0);
	set_gaussians({});
	resize_image(1, 1);
}

scene::~scene() {
	vkQueueWaitIdle(vk_v.queue);
	free_image();
	for (buffer_t* b : {&gaussians_v, &gaussians_2d_v, &sorted_index_v, &image_out_v, &pixel_state_v, &project_settings_v, &raster_settings_v}) {
		free_buffer(*b);
	}
	free_pipeline(project_v);
	free_pipeline(raster_v);
	vkDestroyFence(vk_v.device, fence_v, nullptr);
	vkDestroyCommandPool(vk_v.device, command_pool_v, nullptr);
	vkDestroyDescriptorPool(vk_v.device, descriptor_pool_v, nullptr);
}

scene::buffer_t scene::make_buffer(VkDeviceSize size, VkMemoryPropertyFlags preferred) {
	buffer_t b{};
	b.size = std::max<VkDeviceSize>(size, 16);
	VkBufferCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.size = b.size;
	info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	check(vkCreateBuffer(vk_v.device, &info, nullptr, &b.buffer), "vkCreateBuffer");

	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(vk_v.device, b.buffer, &req);
	b.memory = allocate(vk_v, req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, preferred);
	check(vkBindBufferMemory(vk_v.device, b.buffer, b.memory, 0), "vkBindBufferMemory");
	check(vkMapMemory(vk_v.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
	return b;
}

void scene::free_buffer(buffer_t& b) {
	vkDestroyBuffer(vk_v.device, b.buffer, nullptr);
	vkFreeMemory(vk_v.device, b.memory, nullptr);
	b = buffer_t{};
}

scene::pipeline_t scene::make_pipeline(const std::string& spv_path, int storage_buffer_count) {
	pipeline_t p{};
	std::vector<char> code = gsplat::core::load_entire_file_binary(spv_path);
	VkShaderModuleCreateInfo module_info{};
	module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	module_info.codeSize = code.size();
	module_info.pCode = reinterpret_cast<const std::uint32_t*>(code.data());
	check(vkCreateShaderModule(vk_v.device, &module_info, nullptr, &p.module), "vkCreateShaderModule");

	std::vector<VkDescriptorSetLayoutBinding> bindings(storage_buffer_count);
	for (int i = 0; i < storage_buffer_count; i++) {
		bindings[i].binding = i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo set_layout_info{};
	set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set_layout_info.bindingCount = (std::uint32_t)bindings.size();
	set_layout_info.pBindings = bindings.data();
	check(vkCreateDescriptorSetLayout(vk_v.device, &set_layout_info, nullptr, &p.set_layout), "vkCreateDescriptorSetLayout");

	VkPipelineLayoutCreateInfo layout_info{};
	layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &p.set_layout;
	check(vkCreatePipelineLayout(vk_v.device, &layout_info, nullptr, &p.layout), "vkCreatePipelineLayout");

	VkComputePipelineCreateInfo pipeline_info{};
	pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	pipeline_info.stage.module = p.module;
	pipeline_info.stage.pName = "main"; // slangc renames computeMain
	pipeline_info.layout = p.layout;
	check(vkCreateComputePipelines(vk_v.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &p.pipeline), "vkCreateComputePipelines");

	VkDescriptorSetAllocateInfo set_info{};
	set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	set_info.descriptorPool = descriptor_pool_v;
	set_info.descriptorSetCount = 1;
	set_info.pSetLayouts = &p.set_layout;
	check(vkAllocateDescriptorSets(vk_v.device, &set_info, &p.set), "vkAllocateDescriptorSets");
	return p;
}

void scene::free_pipeline(pipeline_t& p) {
	vkDestroyPipeline(vk_v.device, p.pipeline, nullptr);
	vkDestroyPipelineLayout(vk_v.device, p.layout, nullptr);
	vkDestroyDescriptorSetLayout(vk_v.device, p.set_layout, nullptr);
	vkDestroyShaderModule(vk_v.device, p.module, nullptr);
	p = pipeline_t{};
}

void scene::bind_buffers() {
	if (!image_out_v.buffer || !gaussians_v.buffer) return; // still in the constructor

	auto bind = [&](VkDescriptorSet set, std::initializer_list<const buffer_t*> buffers) {
		std::vector<VkDescriptorBufferInfo> infos;
		for (const buffer_t* b : buffers) infos.push_back({b->buffer, 0, VK_WHOLE_SIZE});
		std::vector<VkWriteDescriptorSet> writes(infos.size());
		for (std::size_t i = 0; i < infos.size(); i++) {
			writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[i].dstSet = set;
			writes[i].dstBinding = (std::uint32_t)i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[i].pBufferInfo = &infos[i];
		}
		vkUpdateDescriptorSets(vk_v.device, (std::uint32_t)writes.size(), writes.data(), 0, nullptr);
	};
	bind(project_v.set, {&gaussians_v, &gaussians_2d_v, &project_settings_v});
	bind(raster_v.set, {&gaussians_2d_v, &gaussians_v, &sorted_index_v, &image_out_v, &pixel_state_v, &raster_settings_v});
}

void scene::free_image() {
	vkDestroyImageView(vk_v.device, image_v.view, nullptr);
	vkDestroyImage(vk_v.device, image_v.image, nullptr);
	vkFreeMemory(vk_v.device, image_memory_v, nullptr);
	image_v = rendered_image{};
	image_memory_v = VK_NULL_HANDLE;
}

void scene::resize_image(int width, int height) {
	if (image_v.image && image_v.width == width && image_v.height == height) return;

	check(vkQueueWaitIdle(vk_v.queue), "vkQueueWaitIdle"); // the viewer may still be drawing the old image
	free_image();
	free_buffer(image_out_v);
	free_buffer(pixel_state_v);
	std::size_t pixels = (std::size_t)width * height;
	image_out_v = make_buffer(pixels * sizeof(glm::vec4), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	pixel_state_v = make_buffer(pixels * sizeof(pixel_state), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	image_v.width = width;
	image_v.height = height;
	VkImageCreateInfo image_info{};
	image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType = VK_IMAGE_TYPE_2D;
	image_info.format = image_v.format;
	image_info.extent = VkExtent3D{(std::uint32_t)width, (std::uint32_t)height, 1};
	image_info.mipLevels = 1;
	image_info.arrayLayers = 1;
	image_info.samples = VK_SAMPLE_COUNT_1_BIT;
	image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
	image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	check(vkCreateImage(vk_v.device, &image_info, nullptr, &image_v.image), "vkCreateImage");

	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(vk_v.device, image_v.image, &req);
	image_memory_v = allocate(vk_v, req, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	check(vkBindImageMemory(vk_v.device, image_v.image, image_memory_v, 0), "vkBindImageMemory");

	VkImageViewCreateInfo view_info{};
	view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_info.image = image_v.image;
	view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	view_info.format = image_v.format;
	view_info.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	check(vkCreateImageView(vk_v.device, &view_info, nullptr, &image_v.view), "vkCreateImageView");
	bind_buffers();
}

void scene::set_gaussians(std::span<const gaussian> gaussians) {
	std::size_t n = gaussians.size();
	if (!gaussians_v.buffer || n * sizeof(gaussian) > gaussians_v.size) {
		free_buffer(gaussians_v);
		free_buffer(gaussians_2d_v);
		free_buffer(sorted_index_v);
		gaussians_v = make_buffer(n * sizeof(gaussian), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		gaussians_2d_v = make_buffer(n * sizeof(gaussian2d), VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
		sorted_index_v = make_buffer(n * sizeof(int), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		bind_buffers();
	}
	if (n > 0) std::memcpy(gaussians_v.mapped, gaussians.data(), n * sizeof(gaussian));
	opacity_v.resize(n);
	for (std::size_t i = 0; i < n; i++) opacity_v[i] = gaussians[i].color.color.a;
	gaussian_count_v = n;
}

void scene::submit_and_wait() {
	check(vkEndCommandBuffer(command_buffer_v), "vkEndCommandBuffer");
	VkSubmitInfo submit{};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &command_buffer_v;
	check(vkQueueSubmit(vk_v.queue, 1, &submit, fence_v), "vkQueueSubmit");
	check(vkWaitForFences(vk_v.device, 1, &fence_v, VK_TRUE, UINT64_MAX), "vkWaitForFences");
	check(vkResetFences(vk_v.device, 1, &fence_v), "vkResetFences");
}

rendered_image scene::render(const camera& cam, int width, int height, glm::vec4 background) {
	resize_image(width, height);
	const int n = (int)gaussian_count_v;
	const float near_plane = cam.limits.z;
	VkCommandBufferBeginInfo begin{};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	// 1. project to 2d, then read the depths back
	*(project_settings_t*)project_settings_v.mapped = project_settings_t{cam, n, 0, 0, 0};
	check(vkBeginCommandBuffer(command_buffer_v, &begin), "vkBeginCommandBuffer");
	if (n > 0) {
		vkCmdBindPipeline(command_buffer_v, VK_PIPELINE_BIND_POINT_COMPUTE, project_v.pipeline);
		vkCmdBindDescriptorSets(command_buffer_v, VK_PIPELINE_BIND_POINT_COMPUTE, project_v.layout, 0, 1, &project_v.set, 0, nullptr);
		vkCmdDispatch(command_buffer_v, (n + 1023) / 1024, 1, 1);
	}
	VkMemoryBarrier to_host{};
	to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier(command_buffer_v, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, nullptr, 0, nullptr);
	submit_and_wait();

	// 2. cull and sort front to back. stable like training's sort, so equal depths blend in the same order
	const gaussian2d* g2 = (const gaussian2d*)gaussians_2d_v.mapped;
	order_v.clear();
	for (int i = 0; i < n; i++) {
		if (g2[i].depth > near_plane && reaches_image(g2[i], opacity_v[i], width, height)) order_v.push_back(i);
	}
	std::stable_sort(order_v.begin(), order_v.end(), [g2](int a, int b) { return g2[a].depth < g2[b].depth; });
	if (!order_v.empty()) std::memcpy(sorted_index_v.mapped, order_v.data(), order_v.size() * sizeof(int));

	// 3. raster, then copy the pixels into the image
	*(raster_settings_t*)raster_settings_v.mapped = raster_settings_t{background, width, height, (int)order_v.size(), near_plane};
	check(vkBeginCommandBuffer(command_buffer_v, &begin), "vkBeginCommandBuffer");
	vkCmdBindPipeline(command_buffer_v, VK_PIPELINE_BIND_POINT_COMPUTE, raster_v.pipeline);
	vkCmdBindDescriptorSets(command_buffer_v, VK_PIPELINE_BIND_POINT_COMPUTE, raster_v.layout, 0, 1, &raster_v.set, 0, nullptr);
	vkCmdDispatch(command_buffer_v, (std::uint32_t)(((std::size_t)width * height + 1023) / 1024), 1, 1);

	// ALL_COMMANDS also waits out the viewer sampling last frame's image
	VkMemoryBarrier raster_done{};
	raster_done.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	raster_done.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	raster_done.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
	VkImageMemoryBarrier to_transfer{};
	to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	to_transfer.srcAccessMask = 0;
	to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image = image_v.image;
	to_transfer.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	vkCmdPipelineBarrier(command_buffer_v, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
		0, 1, &raster_done, 0, nullptr, 1, &to_transfer);

	VkBufferImageCopy region{};
	region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	region.imageExtent = VkExtent3D{(std::uint32_t)width, (std::uint32_t)height, 1};
	vkCmdCopyBufferToImage(command_buffer_v, image_out_v.buffer, image_v.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	VkImageMemoryBarrier to_sampled = to_transfer;
	to_sampled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	to_sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	to_sampled.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	to_sampled.newLayout = image_v.layout;
	vkCmdPipelineBarrier(command_buffer_v, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		0, 0, nullptr, 0, nullptr, 1, &to_sampled);
	submit_and_wait();
	return image_v;
}

std::span<const glm::vec4> scene::pixels() const {
	return {(const glm::vec4*)image_out_v.mapped, (std::size_t)image_v.width * image_v.height};
}
