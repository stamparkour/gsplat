#include <gsplat/vulkan/buffer.h>
#include<vulkan/vulkan.h>
#include <stdexcept>
#include <numeric>

using namespace gsplat::vulkan;

basic_buffer::~basic_buffer() {
	if (buffer_v) {
		vkDestroyBuffer(device_v, buffer_v, nullptr);
		buffer_v = nullptr;
	}
}
VkDevice basic_buffer::device_handle() const {
	return device_v;
}
VkDeviceMemory basic_buffer::memory_handle() const {
	return memory_v;
}
VkBuffer basic_buffer::buffer_handle() const {
	return buffer_v;
}
VkQueue gsplat::vulkan::basic_buffer::queue_handle() const {
	return queue_v;
}
VkCommandPool gsplat::vulkan::basic_buffer::pool_handle() const {
	return pool_v;
}
std::size_t basic_buffer::size() const {
	return size_v;
}
void gsplat::vulkan::basic_buffer::transfer(const basic_buffer* src, std::size_t dst_offset, std::size_t src_offset, std::size_t length) {
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = pool_handle();
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device_handle(), &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	VkBufferCopy copyRegion{};
	copyRegion.srcOffset = src_offset; // Optional
	copyRegion.dstOffset = dst_offset; // Optional
	copyRegion.size = length;
	vkCmdCopyBuffer(commandBuffer, src->buffer_handle(), buffer_handle(), 1, &copyRegion);

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(queue_handle(), 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue_handle());

	vkFreeCommandBuffers(device_handle(), pool_handle(), 1, &commandBuffer);
}
void gsplat::vulkan::basic_buffer::transfer(const basic_buffer* src) {
	transfer(src, 0, 0, size());
}
void gsplat::vulkan::basic_buffer::clear() {
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = pool_handle();
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device_handle(), &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	vkCmdFillBuffer(commandBuffer, buffer_handle(), 0, VK_WHOLE_SIZE, 0);

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(queue_handle(), 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue_handle());

	vkFreeCommandBuffers(device_handle(), pool_handle(), 1, &commandBuffer);
}

VkBufferUsageFlags gsplat::vulkan::buffer_creation_settings::to_usage_flags() const {
	VkBufferUsageFlags o{};
	if (usage_transfer_src) o |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	if (usage_transfer_dst) o |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	if (usage_uniform_buffer) o |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	if (usage_storage_buffer) o |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	return o;
}

buffer::buffer(const buffer_creation_settings& settings) {
	this->settings_v = settings;
	this->size_v = settings.size;
	if (settings.queues.size() == 0) throw std::runtime_error("failed to create buffer. queues.size() == 0");
	this->device_v = settings.queues[0]->logical_device()->device_handle();
	auto ld = settings.queues[0]->logical_device();
	this->logical_device_v = ld;
	this->queue_v = settings.queues[0]->queue_handle();
	this->pool_v = settings.queues[0]->pool_handle();


	VkBufferCreateInfo buffer_info{};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = (VkDeviceSize)settings.size;
	buffer_info.usage = settings.to_usage_flags();
	buffer_info.sharingMode = settings.queues.size() == 1 ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT;
	
	// TODO: implement multi queue buffer
	buffer_info.queueFamilyIndexCount = 1;
	std::vector<std::uint32_t> families{1};
	families[0] = (std::uint32_t)settings.queues[0]->queue_family();

	buffer_info.pQueueFamilyIndices = families.data();

	if (vkCreateBuffer(this->device_v, &buffer_info, nullptr, &this->buffer_v) != VK_SUCCESS) {
		throw std::runtime_error("create buffer failed");
	}
	VkMemoryRequirements prop;
	vkGetBufferMemoryRequirements(this->device_v, this->buffer_v, &prop);
	std::size_t alignment = std::gcd((std::size_t)prop.alignment, settings.min_alignment);
	device_memory::pointer ptr{};
	for (auto& v : settings.memory) {
		auto tmp_settings = v;
		tmp_settings.type_bitmask = prop.memoryTypeBits;
		//if (!v.is_valid(&tmp_type, nullptr)) {
		//	continue;
		//}
		auto p = ld->memory_collection().aligned_malloc(alignment, prop.size, tmp_settings);
		if (device_memory::is_valid_ptr(p)) {
			pointer_v = p;
			break;
		}
	}

	if (!device_memory::is_valid_ptr(pointer_v)) {
		throw std::runtime_error("failed to allocate memory");
	}

	this->memory_v = device_memory::get_memory_ptr(pointer_v)->memory_handle();

	vkBindBufferMemory(
		this->device_v, 
		buffer_v, 
		memory_v, 
		pointer_v.ptr->start);

}
buffer::~buffer() {
	if (buffer_v) {
		vkDestroyBuffer(device_v, buffer_v, nullptr);
		buffer_v = nullptr;
	}
	if (device_memory::is_valid_ptr(pointer_v)) {
		logical_device_v.lock()->memory_collection().free(pointer_v);
		pointer_v = {};
	}
}

buffer::memory_lock_t buffer::memory_lock() {
	return memory_lock_t(this);
}

gsplat::vulkan::buffer::memory_lock_t::memory_lock_t(gsplat::vulkan::buffer* b) {
	this->buffer_v = b;
	vkMapMemory(b->device_handle(), 
		b->memory_handle(), 
		b->pointer_v.ptr->start,
		b->pointer_v.ptr->end - b->pointer_v.ptr->start,
		0, &this->begin_v);
	this->end_v = (char*)this->begin_v + b->size();
}
gsplat::vulkan::buffer::memory_lock_t::~memory_lock_t() {
	vkUnmapMemory(buffer_v->device_handle(), buffer_v->memory_handle());
}
buffer* buffer::memory_lock_t::buffer() const {
	return buffer_v;
}
void* buffer::memory_lock_t::begin() {
	return begin_v;
}
void* buffer::memory_lock_t::end() {
	return end_v;
}