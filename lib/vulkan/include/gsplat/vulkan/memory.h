#ifndef GSPLAT_VULKAN_MEMORY_H
#define GSPLAT_VULKAN_MEMORY_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <gsplat/vulkan/vulkan2.h>
#include <cstddef>
#include <string>
#include <algorithm>
#include <vector>
#include <stamp/task/stream.h>
#include <numeric>
#include <cstdint> 

// uses cpp coroutine/awaiters

namespace gsplat::vulkan {

	using memory_type_t = int;
	struct memory_type {
		enum : memory_type_t {
			device_local,
			host_visible,
		};
	};

	struct vulkan_memory_allocator;
	struct vulkan_memory_node_t {
		bool in_use;
		vulkan_memory_allocator* allocator;
		std::list<vulkan_memory_node_t>::iterator self;
		VkMemoryHeap heap;
		std::size_t begin;
		std::size_t end;
	};
	struct vulkan_memory_allocator {
		using node_t = vulkan_memory_node_t;
		std::list<node_t> nodes;

		vulkan_memory_allocator() {}
		vulkan_memory_allocator(memory_type_t type, std::size_t size) {
			// TODO: implement memory alloc
			VkMemoryHeap heap;
			auto& v = nodes.emplace_back();
			v.in_use = false;
			v.allocator = this;
			v.self = nodes.begin();
			v.heap = heap;
			v.begin = 0;
			v.end = size;
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

				std::list<node_t>::iterator j = i;
				if (align_end == v.end) {
					v.in_use = true;
					v.begin = align_begin;
				}
				else {
					j = nodes.insert(i, {});
					auto& u = *j;
					u.in_use = true;
					u.allocator = this;
					u.self = j;
					u.heap = v.heap;
					u.begin = align_begin;
					u.end = align_end;

					v.begin = align_end;
				}

				if (align_begin != v.begin) {
					auto k = nodes.insert(j, {});
					auto& h = *j;
					h.in_use = false;
					h.allocator = this;
					h.self = k;
					h.heap = v.heap;
					h.begin = tmp_begin;
					h.end = align_begin;
				}

				return &(*i);
			}
			return nullptr;
		}
		node_t* free(node_t* node) {
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
		}
		gptr_t& operator =(const gptr_t& other) {
			ptr = other.ptr;
		}
		VkMemoryHeap heap() const {
			return ptr->heap;
		}
		std::size_t offset() const {
			return ptr->begin;
		}
		std::size_t size() const {
			return ptr->end - ptr->begin;
		}

		operator bool() const {
			return ptr == nullptr;
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
		
		allocator_t device_local_allocs{};
		allocator_t host_visibile_allocs{};
		bool initialized = false;
		inline static vulkan_memory_manager* global_instance_v = nullptr;
	public:
		vulkan_memory_manager() {
			stamp::core::s_assert(global_instance_v == nullptr);
			global_instance_v = this;

			device_local_allocs = allocator_t{memory_type::device_local,  0x4000000}; // 64 MB
			host_visibile_allocs = allocator_t{memory_type::host_visible,   0x80000}; // 512 KB
		}
		~vulkan_memory_manager() {
			global_instance_v = nullptr;
		}
		gptr_t alloc(std::size_t size, std::size_t align, memory_type_t type) {
			switch (type) {
			case memory_type::device_local:
				device_local_allocs.alloc(size, align);
				break;
			case memory_type::host_visible:
				host_visibile_allocs.alloc(size, align);
				break;
			}
		}
		void free(gptr_t ptr) {
			stamp::core::s_assert(ptr != nullptr);
			ptr.ptr->allocator->free(ptr.ptr);
		}

		static vulkan_memory_manager* global_instance() {
			return global_instance_v;
		}
	};

	gptr_t alloc_gpu_mem(std::size_t size, std::size_t align, memory_type_t mem_type) {
		return vulkan_memory_manager::global_instance()->alloc(size, align, mem_type);
	}
	void free_gpu_mem(gptr_t p) {
		vulkan_memory_manager::global_instance()->free(p);
	}

	class buffer;
	class buffer_io_strand : stamp::task::co_stream_strand {
	private:
		buffer* buffer_v;
		bool eof_v = false;
		bool fail_v = false;
		bool bad_v = false;
		std::size_t pos = 0;

		gptr_t host_mem_v = nullptr;
		char* map_buffer_v = nullptr;
		std::size_t buffer_pos = 0;
		std::size_t buffer_size = 512;
	public:
		buffer_io_strand(buffer*);
		~buffer_io_strand();
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
		virtual void clear() {
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
			eof_v = false;
			std::ptrdiff_t buf_offset = pos - buffer_pos;
			if (buf_offset < 0 || buf_offset >= buffer_size) return 0;
			std::ptrdiff_t length = std::min(
				(std::ptrdiff_t)size, 
				(std::ptrdiff_t)(buffer_size - buf_offset), 
				(std::ptrdiff_t)(buffer_v->size() - pos));
			stamp::core::s_assert(length + buf_offset <= buffer_size);
			stamp::core::s_assert(length + pos <= buffer_v->size());

			std::copy(span, span + length, map_buffer_v + buf_offset);
			pos += length;
			if (pos >= buffer_v->size()) {
				eof_v = true;
			}
			return length;
		}
		virtual std::size_t do_try_get_span(char* span, std::size_t size) {
			eof_v = false;
			std::ptrdiff_t buf_offset = pos - buffer_pos;
			if (buf_offset < 0 || buf_offset >= buffer_size) return 0;
			std::ptrdiff_t length = std::min(
				(std::ptrdiff_t)size, 
				(std::ptrdiff_t)(buffer_size - buf_offset), 
				(std::ptrdiff_t)(buffer_v->size() - pos));
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
			if (pos < buffer_pos || pos > buffer_pos + buffer_size) {
				return false;
			}
			return true;
		}
		virtual bool do_await_suspend(std::coroutine_handle<> h) {
			std::size_t aligned_pos = (pos / buffer_size) * buffer_size;
			// TODO: copy current buffer (make sure to cut off at file end)
			// then copy new range into strand's buffer.
			buffer_pos = aligned_pos;
			return false; // resumes coroutine.
		}
		virtual bool do_await_resume() {
			return good();
		}
	public:
		template<typename ValueType>
		void set_pos(std::size_t index) {
			set_ppos(index * sizeof(ValueType));
			// set_gpos(index * sizeof(value_type)); // redundent, both set same pos.
		}
		template<typename ValueType>
		stamp::task::coroutine<ValueType> get_next() {
			ValueType v;
			co_await this->get_span(&v, sizeof(ValueType));
			co_return v;
		}
		template<typename ValueType>
		stamp::task::coroutine<ValueType> set_next(const_reference val) {
			ValueType v;
			co_await this->put_span(&v, sizeof(ValueType));
			co_return v;
		}
	};

	/* // example core for buffer_io_strand
	stamp::task::coroutine<void> run(buffer_iostream_strand<int>* strand) {
		strand->go_at(12);
		int v = co_await strand->get_next();
		co_await strand->set_next(v+3);
	}
	*/

	class buffer {
	public:
		using size_type = std::size_t;
	private:
		size_type size_v = 0;
		VkBuffer buffer_v = nullptr;
		gptr_t raw_data_v = nullptr;
		buffer(size_type size, VkBuffer buffer_v, gptr_t raw_data_v) : size(size), buffer_v(buffer_v), raw_data_v(raw_data_v) {}
	public:
		buffer() {}
		buffer(size_type size) {
			this->size_v = size;
			VkBufferCreateInfo buffer_info{};
			buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			buffer_info.size = (VkDeviceSize)size;
			buffer_info.usage = settings.to_usage_flags();
			buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;

			if (vkCreateBuffer(vulkan_context::device(), &buffer_info, nullptr, &this->buffer_v) != VK_SUCCESS) {
				throw std::runtime_error("create buffer failed");
			}

			VkMemoryRequirements prop;
			vkGetBufferMemoryRequirements(this->device_v, this->buffer_v, &prop);
			std::size_t alignment = std::lcm((std::size_t)prop.alignment, alignof(T));
			raw_data_v = alloc_gpu_mem(size, alignment, memory_type::device_local);
		}
		buffer(size_type size);
		// copy buffer in different memory type
		~buffer();
		buffer(const buffer&) = delete;
		buffer(buffer&& other) {
			buffer_v = other.buffer_v;
			other.buffer_v = nullptr;
			raw_data_v = other.raw_data_v;
			other.raw_data_v = nullptr;
		}
		buffer& operator =(const buffer&) = delete;
		buffer& operator =(buffer&& other) {
			buffer_v = other.buffer_v;
			other.buffer_v = nullptr;
			raw_data_v = other.raw_data_v;
			other.raw_data_v = nullptr;
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
			// TODO: get queue
			// TODO: implement vulkan_context::command_pool();
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
			vkQueueSubmit(queue_handle(), 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(queue_handle()); // co_await queue

			vkFreeCommandBuffers(vulkan_context::device(), pool_handle(), 1, &commandBuffer);
			co_return;
		}
		stamp::task::coroutine<void> resize(size_type new_size) {

		}
		stamp::task::coroutine<buffer<T>> clone() {
			buffer<T> b{size};
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
			vkQueueSubmit(queue_handle(), 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(queue_handle());

			vkFreeCommandBuffers(vulkan_context::device(), pool_handle(), 1, &commandBuffer);

			co_return b;
		}
		

		buffer_io_strand co_iostream_strand() {
			return {this};
		}

		template<typename T>
		std::vector<T> to_vector() {

		}
		template<typename T>
		static buffer from_vector(const std::vector<T>&);
	};

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

	struct compute_shader_config {
		const std::string& file_name;
		int storage_buffer_count;
	};

	class compute_shader {
		VkDescriptorSetLayout desc_layout_v;
		VkPipelineLayout pipeline_layout_v;
		VkShaderModule shader_v;
		VkPipeline pipeline_v;
		VkDescriptorSet descriptor_v;
	public:
		compute_shader() {

		}
		compute_shader(const compute_shader_config& config) {

		}
		~compute_shader() {

		}

		compute_shader(const compute_shader&) = delete;
		compute_shader(compute_shader&&) {

		}
		compute_shader& operator =(const compute_shader&) = delete;
		compute_shader& operator =(compute_shader&&) {

		}
		
		template<typename T>
		void bind(int binding, const buffer&) {

		}
		void invoke(int x, int y, int z) {

		}
	};
}

#endif // GSPLAT_VULKAN_MEMORY_H