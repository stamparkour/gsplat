#ifndef GSPLAT_VULKAN_MEMORY_H
#define GSPLAT_VULKAN_MEMORY_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <cstddef>

// uses cpp coroutine/awaiters

namespace gsplat::vulkan {
	using gptr_t = void*;

	struct memory_type {
		enum : int {
			device_local,
			host_visible, // assumes highly volatile
		};
	};
	struct buffer_bits {
		enum : int {
			transfer_src = 0x1,
			transfer_dst = 0x2,
		};
	};

	VkCommandBuffer alloc_command_buffer();
	void free_command_buffer(VkCommandBuffer);
	gptr_t malloc(std::size_t size, std::size_t align, memory_type);
	void free(gptr_t);
	// only for host_visible memory. implicit unmap on free.
	void* map(gptr_t, std::size_t offset);


	template<typename T>
	class buffer {
	public:
		using value_type = T;
		using pointer_type = T*;
		using reference_type = T&;
		using const_reference_type = T&;
	private:
		gptr_t raw_ptr;
		VkBuffer buffer;
		std::size_t size;
		memory_type mem_type;
		buffer_bits buf_bits;
	public:
		buffer();
		buffer(memory_type, buffer_bits);
		buffer(memory_type, buffer_bits, std::size_t size);
		buffer(memory_type, buffer_bits, std::size_t size, const_reference_type default_value);
		// copy buffer in different memory type
		~buffer();
		// copies entire buffer
		buffer(const buffer&) = delete;
		// moves buffer
		buffer(buffer&&);
		// copies buffer contents ( replaces current mem_type)
		buffer& operator =(const buffer&) = delete;
		// moves buffers contents ( replaces current mem_type)
		buffer& operator =(buffer&&);

		std::size_t size() const;
		std::size_t max_size() const;
		bool empty() const;
		VkBuffer buffer() const;
		memory_type type() const;
		buffer_bits bits() const;

		auto await_resize(std::size_t new_size);
		auto await_transfer(const buffer& src);

		pointer_type* map();
		void unmap();
	};

	class image2d {
	public:

	};
}

#endif // GSPLAT_VULKAN_MEMORY_H