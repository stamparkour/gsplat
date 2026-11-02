#ifndef GSPLAT_VULKAN_MEMORY_H
#define GSPLAT_VULKAN_MEMORY_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <cstddef>

// uses cpp coroutine/awaiters

namespace gsplat::vulkan {
	using command_buffer = void*;

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


	template<typename T>
	class vector {
	public:
		using value_type = T;
		using size_type = std::size_t;
		using pointer = T*;
		using const_pointer = const T*;
		using reference = T&;
		using const_reference = const T&;
	private:
	public:
		vector();
		vector();
		vector(size_type size);
		vector(size_type size, const_reference_type default_value);
		// copy buffer in different memory type
		~vector();
		vector(const vector&);
		vector(vector&&);
		vector& operator =(const vector&);
		vector& operator =(vector&&);

		size_type size() const;
		size_type max_size() const;
		bool empty() const;
		auto buffer() const;

		auto clear();
		auto resize(command_buffer, size_type new_size);

		// 
		auto stream();
		auto cstream() const;
	};

	class image2d {
	public:

	};
}

#endif // GSPLAT_VULKAN_MEMORY_H