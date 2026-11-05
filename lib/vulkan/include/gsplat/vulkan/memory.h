#ifndef GSPLAT_VULKAN_MEMORY_H
#define GSPLAT_VULKAN_MEMORY_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <cstddef>
#include <stamp/task/stream.h>

// uses cpp coroutine/awaiters

namespace gsplat::vulkan {
	using gptr_t = void*;
	using command_buffer = void*;

	struct memory_type {
		enum : int {
			device_local,
			host_visible, // assumes highly volatile
		};
	};

	gptr_t malloc(std::size_t size, std::size_t align, memory_type);
	void free(gptr_t);

	template<typename T>
	class buffer;
	template<typename T>
	class buffer_iostream_strand : stamp::task::co_stream_strand {
	public:
		buffer_iostream_strand(buffer*);
		~buffer_iostream_strand();
	protected:
		virtual bool do_eof() const;
		virtual bool do_fail() const;
		virtual bool do_bad() const;

		virtual std::size_t do_get_ppos() const;
		virtual void do_set_ppos(std::size_t pos, seekdir);
		virtual std::size_t do_get_gpos() const;
		virtual void do_set_gpos(std::size_t pos, seekdir);

		virtual std::size_t do_try_put_span(const char* span, std::size_t size);
		virtual std::size_t do_try_get_span(char* span, std::size_t size);

		virtual bool do_await_ready();
		virtual bool do_await_suspend(std::coroutine_handle<> h);
		virtual std::size_t do_await_resume();
	};

	template<typename T>
	class buffer {
	public:
		using value_type = T;
		using size_type = std::size_t;
		using pointer = T*;
		using const_pointer = const T*;
		using reference = T&;
		using const_reference = const T&;
	private:
	public:
		buffer();
		buffer();
		buffer(size_type size);
		buffer(size_type size, const_reference default_value);
		// copy buffer in different memory type
		~buffer();
		buffer(const buffer&);
		buffer(buffer&&);
		buffer& operator =(const buffer&);
		buffer& operator =(buffer&&);

		size_type size() const;
		size_type max_size() const;
		bool empty() const;
		VkBuffer handle() const;

		// awaiter
		auto clear();
		// awaiter
		auto resize(size_type new_size);

		buffer_iostream_strand co_iostream_strand();
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
		image2d(image2d&&) = delete;
		image2d& operator =(const image2d&) = delete;
		image2d& operator =(image2d&&) = delete;

		VkImage handle();
	};
}

#endif // GSPLAT_VULKAN_MEMORY_H