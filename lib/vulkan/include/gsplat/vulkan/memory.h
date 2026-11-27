#ifndef GSPLAT_VULKAN_MEMORY_H
#define GSPLAT_VULKAN_MEMORY_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <gsplat/vulkan/vulkan2.h>
#include <gsplat/core/read_file.h>
#include <filesystem>
#include <cstddef>
#include <utility>
#include <mutex>
#include <string>
#include <algorithm>
#include <vector>
#include <stamp/task/stream.h>
#include <numeric>
#include <cstdint> 
#include <iostream>
#include <list> 

// uses cpp coroutine/awaiters

namespace gsplat::vulkan {

	using memory_type_t = int;
	struct memory_type {
		enum : memory_type_t {
			any,
			device_local,
			host_visible,
		};
	};

	inline bool memory_type_test_bitmask(memory_type_t type, uint32_t bitmask) {
		switch (type) {
		default:
		case memory_type::any:
			return true;
		case memory_type::host_visible:
			return (bitmask & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (bitmask & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		case memory_type::device_local:
			return bitmask & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		}
	}

	struct vulkan_memory_allocator;
	struct vulkan_memory_node_t {
		bool in_use;
		vulkan_memory_allocator* allocator;
		std::list<vulkan_memory_node_t>::iterator self;
		VkDeviceMemory memory;
		std::size_t begin;
		std::size_t end;
		void* map_ptr;
	};

	inline const VkPhysicalDeviceMemoryProperties& get_memory_properties() {
		static VkPhysicalDeviceMemoryProperties mem_properties = []() -> VkPhysicalDeviceMemoryProperties {
			VkPhysicalDeviceMemoryProperties m;
			vkGetPhysicalDeviceMemoryProperties(vulkan_context::physical_device(), &m);
			return m;
		}();
		return mem_properties;
	}

	inline int get_memory_type_index(uint32_t type_bitmask, memory_type_t type) {
		auto& mem_properties = get_memory_properties();

		// size, index
		std::vector<std::pair<uint64_t, int>> sorted_indices{};
		for (int i = 0; i < mem_properties.memoryTypeCount; i++) {
			int heap_index = (int)mem_properties.memoryTypes[i].heapIndex;

			uint64_t size = mem_properties.memoryHeaps[heap_index].size;
			sorted_indices.emplace_back(size, i);
		}
		std::sort(sorted_indices.begin(), sorted_indices.end());

		// largest to smallest
		for (int i = mem_properties.memoryTypeCount - 1; i >= 0; i--) {
			int index = sorted_indices[i].second;
			if ((type_bitmask == 0 || type_bitmask & (1 << i)) && memory_type_test_bitmask(type, mem_properties.memoryTypes[index].propertyFlags)) {
				return index;
			}
		}
		return -1;
	}

	struct vulkan_memory_allocator {
		using node_t = vulkan_memory_node_t;
		std::list<node_t> nodes;
		uint32_t memory_type_index;
		uint32_t memory_heap_index;

		vulkan_memory_allocator() {}
		vulkan_memory_allocator(uint32_t memory_type_index, std::size_t size) {
			this->memory_type_index = memory_type_index;

			auto& mem_properties = get_memory_properties();
			this->memory_heap_index = mem_properties.memoryTypes[memory_type_index].heapIndex;

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = size;
			allocInfo.memoryTypeIndex = memory_type_index;

			VkDeviceMemory memory;
			if (vkAllocateMemory(vulkan_context::device(), &allocInfo, nullptr, &memory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate vertex buffer memory!");
			}

			void* map_ptr = nullptr;
			if (memory_type_test_bitmask(memory_type::host_visible, mem_properties.memoryTypes[memory_type_index].propertyFlags)) {
				vkMapMemory(vulkan_context::device(), memory, 0, size, 0, &map_ptr);
			}

			auto& v = nodes.emplace_back();
			v.in_use = false;
			v.allocator = this;
			v.self = nodes.begin();
			v.memory = memory;
			v.begin = 0;
			v.end = size;
			v.map_ptr = map_ptr;
		}
		~vulkan_memory_allocator() {
			VkDeviceMemory mem = nodes.front().memory;
			vkFreeMemory(vulkan_context::device(), mem, nullptr);
		}

		bool valid(uint32_t type_bitmask, memory_type_t type) {
			if (type_bitmask != 0 && !(type_bitmask & (1 << memory_type_index))) return false;
			auto& mem_properties = get_memory_properties();
			return memory_type_test_bitmask(type, mem_properties.memoryTypes[memory_type_index].propertyFlags);
		}

		node_t* alloc(std::size_t size, std::size_t align) {
			for (auto i = nodes.begin(); i != nodes.end(); ++i) {
				auto& v = *i;
				if (v.in_use) continue;
				std::size_t tmp_begin = v.begin;
				std::size_t align_begin = v.begin;
				if(align_begin % align != 0) align_begin += align - align_begin % align;
				std::size_t align_end = align_begin + size;
				if (align_end > v.end) continue;

				char* tmp_map = (char*)v.map_ptr;

				std::list<node_t>::iterator j = i;
				if (align_end == v.end) {
					v.in_use = true;
					v.begin = align_begin;
					if(tmp_map) v.map_ptr = (void*)(tmp_map + align_begin - tmp_begin);
				}
				else {
					j = nodes.insert(i, node_t{});
					auto& u = *j;
					u.in_use = true;
					u.allocator = this;
					u.self = j;
					u.memory = v.memory;
					u.begin = align_begin;
					u.end = align_end;
					v.begin = align_end;
					if (tmp_map) u.map_ptr = (void*)(tmp_map + align_begin - tmp_begin);
					if (tmp_map) v.map_ptr = (void*)(tmp_map + align_end - tmp_begin);
				}

				if (align_begin != tmp_begin) {
					auto k = nodes.insert(j, node_t{});
					auto& h = *k;
					h.in_use = false;
					h.allocator = this;
					h.self = k;
					h.memory = v.memory;
					h.begin = tmp_begin;
					h.end = align_begin;
					if (tmp_map) h.map_ptr = (void*)tmp_map;
				}

				return &(*j);
			}
			return nullptr;
		}
		void free(node_t* node) {
			auto i = node->self;
			auto self = i;
			if (self != nodes.begin()) {
				--i;
				auto prev = i;

				// if not in use, then merge back
				if (!prev->in_use) {
					prev->end = self->end;
					nodes.erase(self);
					self = prev;
				}
			}

			i = self;
			++i;
			auto next = i;
			if (next != nodes.end()) {
				// if not in use, then merge forward
				if (!next->in_use) {
					self->end = next->end;
					nodes.erase(next);
				}
			}
			self->in_use = false;
		}
	};
	struct gptr_t {
		vulkan_memory_node_t* ptr = nullptr;
		gptr_t() = default;
		explicit gptr_t(vulkan_memory_node_t* ptr) : ptr(ptr) {}
		gptr_t(nullptr_t) : gptr_t() {}
		gptr_t(const gptr_t& other) {
			ptr = other.ptr;
		}
		gptr_t& operator =(nullptr_t) {
			ptr = nullptr;
			return *this;
		}
		gptr_t& operator =(const gptr_t& other) {
			ptr = other.ptr;
			return *this;
		}
		VkDeviceMemory memory() const {
			return ptr->memory;
		}
		std::size_t offset() const {
			return ptr->begin;
		}
		std::size_t size() const {
			return ptr->end - ptr->begin;
		}
		void* map_ptr() const {
			return ptr->map_ptr;
		}

		operator bool() const {
			return ptr != nullptr;
		}
		auto operator <=>(const gptr_t& other) const {
			return ptr <=> other.ptr;
		}
		bool operator ==(const gptr_t& other) const {
			return ptr == other.ptr;
		}
	};

	class vulkan_memory_manager{
		using node_t = vulkan_memory_node_t;
		using allocator_t = vulkan_memory_allocator;
		using iterator = typename std::list<allocator_t>::iterator;
		
		std::list<allocator_t> allocators_v{};
		bool initialized = false;
		inline static vulkan_memory_manager* global_instance_v = nullptr;

		iterator query_allocator(iterator start, uint32_t type_bitmask, memory_type_t mem_type, std::size_t alloc_size) {
			for (auto i = start; i != allocators_v.end(); i++) {
				if (i->valid(type_bitmask, mem_type)) {
					return i;
				}
			}
			if(alloc_size == 0) return allocators_v.end();

			int index = get_memory_type_index(type_bitmask, mem_type);
			if (index == -1) {
				throw std::runtime_error("failed to obtain a memory allocator");
			}

			// uint32_t memory_type_index, std::size_t size
			allocators_v.emplace_back(index, alloc_size);
			auto j = allocators_v.end();
			--j;

			std::cout << "allocated memory resource: " << mem_type << ", type index:" << index << ", size: " << alloc_size << std::endl;

			return j;
		}
	public:
		vulkan_memory_manager() {
			stamp::core::s_assert(global_instance_v == nullptr);
			global_instance_v = this;

			query_allocator(allocators_v.begin(), 0, memory_type::any, 0x4000000);  // 64 MB
			query_allocator(allocators_v.begin(), 0, memory_type::device_local, 0x4000000);  // 64 MB
			query_allocator(allocators_v.begin(), 0, memory_type::host_visible, 0x80000);  // 512 KB
		}
		~vulkan_memory_manager() {
			global_instance_v = nullptr;
		}
		gptr_t alloc(std::size_t size, std::size_t align, memory_type_t type, uint32_t type_bitmask) {
			iterator v = allocators_v.begin();
			do {
				v = query_allocator(v, type_bitmask, type, 0); // change 0 to something to allocate when full.
				if (v == allocators_v.end()) return nullptr;

				auto p = gptr_t{v->alloc(size, align)};
				if (p) return p;
				else ++v;

			} while (v != allocators_v.end());

			return nullptr;
		}
		void free(gptr_t ptr) {
			stamp::core::s_assert(ptr != nullptr);
			ptr.ptr->allocator->free(ptr.ptr);
		}

		static vulkan_memory_manager* global_instance() {
			return global_instance_v;
		}
	};

	inline gptr_t alloc_gpu_mem(std::size_t size, std::size_t align, memory_type_t mem_type, uint32_t type_bitmask = 0) {
		return vulkan_memory_manager::global_instance()->alloc(size, align, mem_type, type_bitmask);
	}
	inline void free_gpu_mem(gptr_t p) {
		vulkan_memory_manager::global_instance()->free(p);
	}

	class buffer_io_strand;
	class buffer {
	public:
		using size_type = std::size_t;
	private:
		size_type size_v = 0;
		size_type raw_size_v = 0;
		VkBuffer buffer_v = nullptr;
		gptr_t raw_data_v = nullptr;
		buffer(size_type size, VkBuffer buffer_v, gptr_t raw_data_v) : size_v(size), raw_size_v(size), buffer_v(buffer_v), raw_data_v(raw_data_v) {}
	public:
		buffer() {}
		buffer(size_type size, size_type align = 16) {
			this->size_v = size;
			this->raw_size_v = size;
			VkBufferCreateInfo buffer_info{};
			buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			buffer_info.size = (VkDeviceSize)size;
			buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // VK_SHARING_MODE_CONCURRENT;

			if (vkCreateBuffer(vulkan_context::device(), &buffer_info, nullptr, &this->buffer_v) != VK_SUCCESS) {
				throw std::runtime_error("create buffer failed");
			}

			VkMemoryRequirements prop;
			vkGetBufferMemoryRequirements(vulkan_context::device(), this->buffer_v, &prop);
			std::size_t alignment = std::lcm((std::size_t)prop.alignment, 16);
			raw_data_v = alloc_gpu_mem(size, alignment, memory_type::any, prop.memoryTypeBits);

			vkBindBufferMemory(vulkan_context::device(), buffer_v, raw_data_v.memory(), raw_data_v.offset());
		}
		~buffer() {
			if (buffer_v) {
				vkDestroyBuffer(vulkan_context::device(), buffer_v, nullptr);
				buffer_v = nullptr;
			}
			if (raw_data_v) {
				free_gpu_mem(raw_data_v);
				raw_data_v = nullptr;
			}
		}
		buffer(const buffer&) = delete;
		buffer(buffer&& other) {
			buffer_v = other.buffer_v;
			other.buffer_v = nullptr;
			raw_data_v = other.raw_data_v;
			other.raw_data_v = nullptr;
			size_v = other.size_v;
			raw_size_v = other.raw_size_v;
		}
		buffer& operator =(const buffer&) = delete;
		buffer& operator =(buffer&& other) {
			buffer_v = other.buffer_v;
			other.buffer_v = nullptr;
			raw_data_v = other.raw_data_v;
			other.raw_data_v = nullptr;
			size_v = other.size_v;
			raw_size_v = other.raw_size_v;
			return *this;
		}

		size_type size() const {
			return size_v;
		}
		size_type max_size() const {
			return SIZE_MAX;
		}
		bool empty() const {
			return size_v == 0;
		}
		VkBuffer handle() const {
			return buffer_v;
		}
		gptr_t ptr() const {
			return raw_data_v;
		}

		stamp::task::coroutine<void> clear() {
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = vulkan_context::command_pool();
			allocInfo.commandBufferCount = 1;

			VkCommandBuffer commandBuffer;
			vkAllocateCommandBuffers(vulkan_context::device(), &allocInfo, &commandBuffer);

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(commandBuffer, &beginInfo);

			vkCmdFillBuffer(commandBuffer, handle(), 0, VK_WHOLE_SIZE, 0);

			vkEndCommandBuffer(commandBuffer);

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			// TODO: enqueue to wait somewhere else
			vkQueueSubmit(vulkan_context::queue(), 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(vulkan_context::queue()); // co_await queue

			vkFreeCommandBuffers(vulkan_context::device(), vulkan_context::command_pool(), 1, &commandBuffer);
			co_return;
		}
		stamp::task::coroutine<void> resize(size_type new_size) {
			// shrink
			if (new_size <= raw_size_v) {
				size_v = new_size;
				co_return;
			}

			// grow 

			if (new_size > size_v * 2) new_size = new_size + (new_size + 9) / 10;
			else new_size = size_v * 2;

			buffer b{new_size};

			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = vulkan_context::command_pool();
			allocInfo.commandBufferCount = 1;

			VkCommandBuffer commandBuffer;
			vkAllocateCommandBuffers(vulkan_context::device(), &allocInfo, &commandBuffer);

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(commandBuffer, &beginInfo);

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0; // Optional
			copyRegion.dstOffset = 0; // Optional
			copyRegion.size = size_v;
			vkCmdCopyBuffer(commandBuffer, buffer_v, b.buffer_v, 1, &copyRegion);

			vkCmdFillBuffer(commandBuffer, b.buffer_v, size_v, new_size-size_v, 0);

			vkEndCommandBuffer(commandBuffer);

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			// TODO: enqueue to wait somewhere else
			vkQueueSubmit(vulkan_context::queue(), 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(vulkan_context::queue());

			vkFreeCommandBuffers(vulkan_context::device(), vulkan_context::command_pool(), 1, &commandBuffer);

			*this = std::move(b);

			co_return;
		}
		stamp::task::coroutine<buffer> copy() const {
			buffer b{size_v};
			// TODO: get queue
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = vulkan_context::command_pool();
			allocInfo.commandBufferCount = 1;

			VkCommandBuffer commandBuffer;
			vkAllocateCommandBuffers(vulkan_context::device(), &allocInfo, &commandBuffer);

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(commandBuffer, &beginInfo);

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0; // Optional
			copyRegion.dstOffset = 0; // Optional
			copyRegion.size = size_v;
			vkCmdCopyBuffer(commandBuffer, buffer_v, b.buffer_v, 1, &copyRegion);

			vkEndCommandBuffer(commandBuffer);

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			// TODO: enqueue to wait somewhere else
			vkQueueSubmit(vulkan_context::queue(), 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(vulkan_context::queue());

			vkFreeCommandBuffers(vulkan_context::device(), vulkan_context::command_pool(), 1, &commandBuffer);

			co_return buffer{std::move(b)};
		}


		buffer_io_strand co_iostream_strand();

		template<typename T>
		stamp::task::coroutine<std::vector<T>> to_vector();
		template<typename T>
		static stamp::task::coroutine<buffer> from_vector(const std::vector<T>& vec);
	};
	class buffer_io_strand : public stamp::task::co_stream_strand {
	private:
		bool is_write = false;
		bool is_started = false;
		buffer* buffer_v = nullptr;
		bool eof_v = false;
		bool fail_v = false;
		bool bad_v = false;
		std::size_t pos = 0;

		gptr_t host_mem_v = nullptr;
		VkBuffer raw_buffer_v = nullptr;
		char* map_buffer_v = nullptr;
		std::size_t buffer_pos = 0;
		std::size_t buffer_size = 512;
	public:
		buffer_io_strand(buffer* target) {
			buffer_v = target;

			VkBufferCreateInfo buffer_info{};
			buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			buffer_info.size = (VkDeviceSize)buffer_size;
			buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // VK_SHARING_MODE_CONCURRENT;

			if (vkCreateBuffer(vulkan_context::device(), &buffer_info, nullptr, &this->raw_buffer_v) != VK_SUCCESS) {
				throw std::runtime_error("create buffer failed");
			}

			VkMemoryRequirements prop;
			vkGetBufferMemoryRequirements(vulkan_context::device(), this->raw_buffer_v, &prop);
			host_mem_v = alloc_gpu_mem(prop.size, prop.alignment, memory_type::host_visible, prop.memoryTypeBits);

			vkBindBufferMemory(vulkan_context::device(), raw_buffer_v, host_mem_v.memory(), host_mem_v.offset());
			map_buffer_v = (char*)host_mem_v.map_ptr();
		}
		~buffer_io_strand() {
			if constexpr (stamp::core::is_debug_v) {
				for (int i = 0; i < buffer_size; i++) {
					map_buffer_v[i] = 0x1B;
				}
			}

			// TODO: release memory
			if (host_mem_v) {
				free_gpu_mem(host_mem_v);
				host_mem_v = nullptr;
			}
			if (raw_buffer_v) {
				vkDestroyBuffer(vulkan_context::device(), raw_buffer_v, nullptr);
				raw_buffer_v = nullptr;
			}
		}
	protected:
		virtual bool do_eof() const {
			return eof_v;
		}
		virtual bool do_fail() const {
			return fail_v;
		}
		virtual bool do_bad() const {
			return bad_v;
		}
		virtual void do_clear() {
			eof_v = false;
			fail_v = false;
			bad_v = false;
		}

		virtual std::size_t do_get_ppos() const {
			return pos;
		}
		virtual void do_set_ppos(std::size_t pos, seekdir dir) {
			switch (dir) {
			case seekdir::beg:
				this->pos = pos;
				break;
			case seekdir::cur:
				this->pos += pos;
				break;
			case seekdir::end:
				this->pos = pos + buffer_v->size();
				break;
			}
			if (this->pos > buffer_v->size()) {
				this->pos = buffer_v->size();
			}
		}
		virtual std::size_t do_get_gpos() const {
			return pos;
		}
		virtual void do_set_gpos(std::size_t pos, seekdir dir) {
			switch (dir) {
			case seekdir::beg:
				this->pos = pos;
				break;
			case seekdir::cur:
				this->pos += pos;
				break;
			case seekdir::end:
				this->pos = pos + buffer_v->size();
				break;
			}
			if (this->pos > buffer_v->size()) {
				this->pos = buffer_v->size();
			}
		}

		virtual std::size_t do_try_put_span(const char* span, std::size_t size) {
			if (!is_started) return 0;
			eof_v = false;
			std::ptrdiff_t buf_offset = pos - buffer_pos;
			if (buf_offset < 0 || buf_offset >= buffer_size) return 0;
			std::ptrdiff_t length = std::min({
				(std::ptrdiff_t)size,
				(std::ptrdiff_t)(buffer_size - buf_offset),
				(std::ptrdiff_t)(buffer_v->size() - pos)});
			stamp::core::s_assert(length + buf_offset <= buffer_size);
			stamp::core::s_assert(length + pos <= buffer_v->size());

			std::copy(span, span + length, map_buffer_v + buf_offset);
			pos += length;
			if (pos >= buffer_v->size()) {
				eof_v = true;
			}
			if (length > 0) is_write = true;
			return length;
		}
		virtual std::size_t do_try_get_span(char* span, std::size_t size) {
			if (!is_started) return 0;
			eof_v = false;
			std::ptrdiff_t buf_offset = pos - buffer_pos;
			if (buf_offset < 0 || buf_offset >= buffer_size) return 0;
			std::ptrdiff_t length = std::min({
				(std::ptrdiff_t)size, 
				(std::ptrdiff_t)(buffer_size - buf_offset), 
				(std::ptrdiff_t)(buffer_v->size() - pos)});
			stamp::core::s_assert(length + buf_offset <= buffer_size);
			stamp::core::s_assert(length + pos <= buffer_v->size());

			std::copy(map_buffer_v + buf_offset, map_buffer_v + buf_offset + length, span);
			pos += length;
			if (pos >= buffer_v->size()) {
				eof_v = true;
			}
			return length;
		}

		virtual bool do_await_ready() {
			return false;
		}
		virtual bool do_await_suspend(std::coroutine_handle<> h) {
			std::size_t aligned_pos = (pos / buffer_size) * buffer_size;

			std::size_t size_write = std::min(buffer_size, buffer_v->size() - buffer_pos);
			std::size_t pos_write = buffer_pos;
			std::size_t size_read = std::min(buffer_size, buffer_v->size() - aligned_pos);
			std::size_t pos_read = aligned_pos;

			// send to buffer

			if (is_started && is_write) {
				is_write = false;

				VkCommandBufferAllocateInfo allocInfo{};
				allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				allocInfo.commandPool = vulkan_context::command_pool();
				allocInfo.commandBufferCount = 1;

				VkCommandBuffer commandBuffer;
				vkAllocateCommandBuffers(vulkan_context::device(), &allocInfo, &commandBuffer);

				VkCommandBufferBeginInfo beginInfo{};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

				vkBeginCommandBuffer(commandBuffer, &beginInfo);

				VkBufferCopy copyRegion{};
				copyRegion.srcOffset = 0; // Optional
				copyRegion.dstOffset = pos_write; // Optional
				copyRegion.size = size_write;
				vkCmdCopyBuffer(commandBuffer, raw_buffer_v, buffer_v->handle(), 1, &copyRegion);

				vkEndCommandBuffer(commandBuffer);

				VkSubmitInfo submitInfo{};
				submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &commandBuffer;

				// TODO: enqueue to wait somewhere else
				vkQueueSubmit(vulkan_context::queue(), 1, &submitInfo, VK_NULL_HANDLE);
				vkQueueWaitIdle(vulkan_context::queue());

				vkFreeCommandBuffers(vulkan_context::device(), vulkan_context::command_pool(), 1, &commandBuffer);
			}

			// recieve from buffer

			{
				VkCommandBufferAllocateInfo allocInfo{};
				allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				allocInfo.commandPool = vulkan_context::command_pool();
				allocInfo.commandBufferCount = 1;

				VkCommandBuffer commandBuffer;
				vkAllocateCommandBuffers(vulkan_context::device(), &allocInfo, &commandBuffer);

				VkCommandBufferBeginInfo beginInfo{};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

				vkBeginCommandBuffer(commandBuffer, &beginInfo);

				VkBufferCopy copyRegion{};
				copyRegion.srcOffset = pos_read; // Optional
				copyRegion.dstOffset = 0; // Optional
				copyRegion.size = size_read;
				vkCmdCopyBuffer(commandBuffer, buffer_v->handle(), raw_buffer_v, 1, &copyRegion);

				vkEndCommandBuffer(commandBuffer);

				VkSubmitInfo submitInfo{};
				submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &commandBuffer;

				// TODO: enqueue to wait somewhere else
				vkQueueSubmit(vulkan_context::queue(), 1, &submitInfo, VK_NULL_HANDLE);
				vkQueueWaitIdle(vulkan_context::queue());

				vkFreeCommandBuffers(vulkan_context::device(), vulkan_context::command_pool(), 1, &commandBuffer);
			}

			buffer_pos = aligned_pos;
			return false; // resumes coroutine.
		}
		virtual bool do_await_resume() {
			is_started = true;
			return good();
		}
	public:
		template<typename ValueType>
		void set_pos(std::size_t index) {
			set_ppos(index * sizeof(ValueType), co_stream_strand::beg);
			// set_gpos(index * sizeof(value_type)); // redundent, both set same pos.
		}
		template<typename ValueType>
		stamp::task::coroutine<ValueType> get_next() {
			ValueType v;
			co_await this->get_span((char*)&v, sizeof(ValueType));
			co_return v;
		}
		template<typename ValueType>
		stamp::task::coroutine<void> set_next(const ValueType& val) {
			co_await this->put_span((char*)&val, sizeof(ValueType));
		}
	};

	inline buffer_io_strand buffer::co_iostream_strand() {
		return {this};
	}

	template<typename T>
	inline stamp::task::coroutine<std::vector<T>> buffer::to_vector() {
		buffer_io_strand strand = co_iostream_strand();
		std::vector<T> vec{};
		strand.set_pos<T>(0);
		while (true) {
			vec.push_back(co_await strand.get_next<T>());
			if (!strand) break;
		}

		co_return vec;
	}
	template<typename T>
	inline stamp::task::coroutine<buffer> buffer::from_vector(const std::vector<T>& vec) {
		buffer b{vec.size() * sizeof(T), alignof(T)};

		buffer_io_strand strand = b.co_iostream_strand();
		strand.set_pos<T>(0);
		int i = 0;
		while (true) {
			co_await strand.set_next<T>(vec[i]);
			++i;
			if (!strand) break;
		}
		co_return b;
	}

	/* // example core for buffer_io_strand
	stamp::task::coroutine<void> run(buffer_iostream_strand<int>* strand) {
		strand->go_at(12);
		int v = co_await strand->get_next();
		co_await strand->set_next(v+3);
	}
	*/

	using image_storage_t = int;
	struct image_storage {
		enum : image_storage_t {
			rgba8
		};
	};

	class image2d {
	public:
		image2d(std::size_t width, std::size_t height, image_storage_t);
		~image2d();
		image2d(const image2d&) = delete;
		image2d(image2d&&) {

		}
		image2d& operator =(const image2d&) = delete;
		image2d& operator =(image2d&&) {

		}

		VkImage handle();

	};

    // Returns the path of the program executable since current working directory
    // may not be the actual path of the executable.
    inline std::filesystem::path executable_path() {
        // Cross-compatible code generated with help of claude. Some platforms untested.
        #ifdef _WIN32
            #include <windows.h>
            wchar_t buf[MAX_PATH];
            DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
            return std::filesystem::path(std::wstring(buf, n));
        #elif defined(__APPLE__)
            #include <mach-o/dyld.h>
            uint32_t size = 0;
            _NSGetExecutablePath(nullptr, &size);
            std::string buf(size, '\0');
            _NSGetExecutablePath(buf.data(), &size);
            return std::filesystem::canonical(buf.c_str());
        #else
            return std::filesystem::canonical("/proc/self/exe");
        #endif
    }

    struct compute_shader_config {
        std::string file_name;
        std::string shader_dir = (executable_path().parent_path() / "shaders").string();
        int storage_buffer_count;
    };

    class compute_shader {
        VkDescriptorSetLayout desc_layout_v;
        VkPipelineLayout pipeline_layout_v;
        VkShaderModule shader_v;
        VkPipeline pipeline_v;
        VkDescriptorSet descriptor_v;
        VkFence fence_v;
        std::mutex invoke_mtx_v;
    public:
        compute_shader() {

        }
        compute_shader(const compute_shader_config& _config) {
			compute_shader_config config = _config;
			if (config.shader_dir.empty()) {
				config.shader_dir = (executable_path().parent_path() / "shaders").string();
			}

            // Create Descriptor Set Layout
            std::vector<VkDescriptorSetLayoutBinding> layout_bindings{};
            for (int i = 0; i < config.storage_buffer_count; i++) {
                layout_bindings.emplace_back(
                    i, // binding
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    1, // descriptor count
                    VK_SHADER_STAGE_COMPUTE_BIT, // stage flags
                    nullptr // immutable samplers
                );
            }
            VkDescriptorSetLayoutCreateInfo layout_ci = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .bindingCount = (uint32_t)layout_bindings.size(),
                .pBindings = layout_bindings.data()
            };
            if (VkResult r = vkCreateDescriptorSetLayout(vulkan_context::device(), &layout_ci, nullptr, &desc_layout_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create descriptor set layout for computer shader. Vk Code: " + std::to_string(r));
            }

            // Load shader module file
            std::vector<char> shader_file = gsplat::core::load_entire_file_binary(config.shader_dir + "/" + config.file_name);

            // Create Shader Module from file
            VkShaderModuleCreateInfo mod_ci = {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = shader_file.size(),
                .pCode = reinterpret_cast<const uint32_t*>(shader_file.data())
            };
            
            if (VkResult r = vkCreateShaderModule(vulkan_context::device(), &mod_ci, nullptr, &shader_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create the shader module for file: " + config.file_name + "(Vk Code: " + std::to_string(r) + ")");
            }

            // Create Shader Pipeline Layout
            VkPipelineShaderStageCreateInfo stage_ci = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_v,
                .pName = "main",
                .pSpecializationInfo = nullptr
            };
            
            VkPipelineLayoutCreateInfo pipeline_layout_ci = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .setLayoutCount = 1,
                .pSetLayouts = &desc_layout_v,
                .pushConstantRangeCount = 0,
                .pPushConstantRanges = nullptr
            };

            if (VkResult r = vkCreatePipelineLayout(vulkan_context::device(), &pipeline_layout_ci, nullptr, &pipeline_layout_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create the shader pipeline layout for file: " + config.file_name + "(Vk Code: " + std::to_string(r) + ")");
            }

            // Create Shader Pipeline
            VkComputePipelineCreateInfo pipeline_ci = {
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = stage_ci,
                .layout = pipeline_layout_v,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = 0
            };

            if (VkResult r = vkCreateComputePipelines(vulkan_context::device(), VK_NULL_HANDLE, 1, & pipeline_ci, nullptr, &pipeline_v);
                r < VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create the shader pipeline for file: " + config.file_name + "(Vk Code: " + std::to_string(r) + ")");
            }

            VkDescriptorSetAllocateInfo desc_set_ai = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = vulkan_context::descriptor_pool(),
                .descriptorSetCount = 1,
                .pSetLayouts = &desc_layout_v
            };
            if (VkResult r = vkAllocateDescriptorSets(vulkan_context::device(), &desc_set_ai, &descriptor_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not allocate descriptor sets. Vk Code: " + std::to_string(r));
            }

            VkFenceCreateInfo fence_ci = {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };

            VkFence fence;
            if (VkResult r = vkCreateFence(vulkan_context::device(), &fence_ci, nullptr, &fence_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not create fence for shader pipeline. Vk Code: " + std::to_string(r));
            }

        }
        ~compute_shader() {
            destroy();
        }

        compute_shader(const compute_shader&) = delete;
        compute_shader& operator =(const compute_shader&) = delete;
        compute_shader(compute_shader&& o) noexcept
            : desc_layout_v(std::exchange(o.desc_layout_v, VK_NULL_HANDLE)),
              pipeline_layout_v(std::exchange(o.pipeline_layout_v, VK_NULL_HANDLE)),
              shader_v(std::exchange(o.shader_v, VK_NULL_HANDLE)),
              pipeline_v(std::exchange(o.pipeline_v, VK_NULL_HANDLE)),
              descriptor_v(std::exchange(o.descriptor_v, VK_NULL_HANDLE)),
              fence_v(std::exchange(o.fence_v, VK_NULL_HANDLE)) {}
        compute_shader& operator =(compute_shader&& o) noexcept {
            if (this != &o) {
                destroy();
                desc_layout_v = std::exchange(o.desc_layout_v, VK_NULL_HANDLE);
                pipeline_layout_v = std::exchange(o.pipeline_layout_v, VK_NULL_HANDLE);
                shader_v = std::exchange(o.shader_v, VK_NULL_HANDLE);
                pipeline_v = std::exchange(o.pipeline_v, VK_NULL_HANDLE);
                descriptor_v = std::exchange(o.descriptor_v, VK_NULL_HANDLE);
                fence_v = std::exchange(o.fence_v, VK_NULL_HANDLE);
            }
            return *this;
        }

        void destroy() noexcept {
            if (fence_v) {
                vkDestroyFence(vulkan_context::device(), fence_v, nullptr);
                fence_v = VK_NULL_HANDLE;
            }
            if (descriptor_v) {
                vkFreeDescriptorSets(vulkan_context::device(), vulkan_context::descriptor_pool(), 1, &descriptor_v);
                descriptor_v = VK_NULL_HANDLE;
            }
            if (pipeline_v) {
                vkDestroyPipeline(vulkan_context::device(), pipeline_v, nullptr);
                pipeline_v = VK_NULL_HANDLE;
            }
            if (pipeline_layout_v) {
                vkDestroyPipelineLayout(vulkan_context::device(), pipeline_layout_v, nullptr);
                pipeline_layout_v = VK_NULL_HANDLE;
            }
            if (shader_v) {
                vkDestroyShaderModule(vulkan_context::device(), shader_v, nullptr);
                shader_v = VK_NULL_HANDLE;
            }
            if (desc_layout_v) {
                vkDestroyDescriptorSetLayout(vulkan_context::device(), desc_layout_v, nullptr);
                desc_layout_v = VK_NULL_HANDLE;
            }
        }

        void bind(int binding, const buffer& b) {
            VkDescriptorBufferInfo buf_info = {
                .buffer = b.handle(),
                .offset = 0,
                .range = b.size()
            };
            VkWriteDescriptorSet buf_write = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = descriptor_v, // shaders' desc set
                .dstBinding = (uint32_t)binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &buf_info,
                .pTexelBufferView = nullptr
            };
            vkUpdateDescriptorSets(vulkan_context::device(), 1, &buf_write, 0, nullptr);
        }
        void invoke(int x, int y, int z) {
            std::lock_guard lock(invoke_mtx_v);
            VkCommandBufferAllocateInfo cmd_ai = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = vulkan_context::command_pool(),
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
            };

            VkCommandBuffer cmd_buf;
            if (VkResult r = vkAllocateCommandBuffers(vulkan_context::device(), &cmd_ai, &cmd_buf);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not allocate command buffer for shader pipeline. Vk Code: " + std::to_string(r));
            }

            VkCommandBufferBeginInfo cmd_begin = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = 0,
                .pInheritanceInfo = nullptr
            };

            if (VkResult r = vkBeginCommandBuffer(cmd_buf, &cmd_begin); r != VK_SUCCESS) {
                throw std::runtime_error("Could not begin recording command buffer for shader pipeline. Vk Code: " + std::to_string(r));
            }

            vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_v);
            vkCmdBindDescriptorSets(
                cmd_buf,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline_layout_v,
                0, 1,
                &descriptor_v,
                0, 0
            );
            vkCmdDispatch(cmd_buf, x, y, z);
            if (VkResult r = vkEndCommandBuffer(cmd_buf); r != VK_SUCCESS) {
                throw std::runtime_error("Could not end command buffer for shader pipeline. Vk Code: " + std::to_string(r));
            }

            VkSubmitInfo submit_info = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd_buf,
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = nullptr
            };

            if (VkResult r = vkResetFences(vulkan_context::device(), 1, &fence_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not reset fence for shader pipeline. Vk Code: " + std::to_string(r));
            }

            if (VkResult r = vkQueueSubmit(vulkan_context::queue(), 1, &submit_info, fence_v);
                r != VK_SUCCESS
            ) {
                throw std::runtime_error("Could not submit queue for shader pipeline. Vk Code: " + std::to_string(r));
            }

            if (VkResult r = vkWaitForFences(vulkan_context::device(), 1, &fence_v, VK_TRUE, UINT64_MAX);
                r < VK_SUCCESS
            ) {
                throw std::runtime_error("Error waiting for shader pipeline to finish. Vk Code: " + std::to_string(r));
            }
            vkFreeCommandBuffers(vulkan_context::device(), vulkan_context::command_pool(), 1, &cmd_buf);
        }
    };
}

#endif // GSPLAT_VULKAN_MEMORY_H
