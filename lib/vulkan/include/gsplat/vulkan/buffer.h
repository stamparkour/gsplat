#ifndef GSPLAT_VULKAN_BUFFER_H
#define GSPLAT_VULKAN_BUFFER_H

#include <gsplat/vulkan/config.h>
#include<vulkan/vulkan.h>
#include <memory>
#include <gsplat/vulkan/device.h>

namespace gsplat::vulkan {
	class basic_buffer;
	using basic_buffer_shared = std::shared_ptr<basic_buffer>;
	using basic_buffer_weak = std::weak_ptr<basic_buffer>;

	class basic_buffer {
	protected:
		VkDevice device_v{};
		VkDeviceMemory memory_v{};
		VkBuffer buffer_v{};
		VkQueue queue_v{};
		VkCommandPool pool_v{};
		std::size_t size_v{};
	public:
		basic_buffer() = default;
		virtual ~basic_buffer();

		VkDevice device_handle() const;
		VkDeviceMemory memory_handle() const;
		VkBuffer buffer_handle() const;
		VkQueue queue_handle() const;
		VkCommandPool pool_handle() const;
		std::size_t size() const;

		void transfer(const basic_buffer* src, std::size_t dst_offset, std::size_t src_offset, std::size_t length);
		void transfer(const basic_buffer* src);
	};

	struct buffer_creation_settings {
		bool usage_transfer_src;
		bool usage_transfer_dst;
		bool usage_uniform_buffer;
		bool usage_storage_buffer;
		std::size_t size;
		std::size_t min_alignment;
		std::vector<device_memory_settings> memory;
		// if size is 1, then sharing is exclusive
		// all queues should be from the same device
		// first queue is primary queue (transfer)
		std::vector<command_queue*> queues;

		VkBufferUsageFlags to_usage_flags() const;

		buffer_creation_settings set_size(std::size_t s) const {
			buffer_creation_settings o = *this;
			o.size = s;
			return o;
		}
		buffer_creation_settings set_alignment(std::size_t a) const {
			buffer_creation_settings o = *this;
			o.min_alignment = a;
			return o;
		}
	};

	class buffer : public basic_buffer {
	public:
		struct memory_lock_t {
			using size_type = std::size_t;
			buffer* buffer_v;
			void* begin_v;
			void* end_v;

			memory_lock_t(buffer*);
			~memory_lock_t();

			buffer* buffer() const;
			void* begin();
			void* end();
		};
	protected:
		buffer_creation_settings settings_v{};
		logical_device_weak logical_device_v{};
		device_memory::pointer pointer_v{};
	public:
		buffer() = default;
		buffer(const buffer_creation_settings&);
		~buffer();

		buffer(const buffer&) = delete;
		buffer(buffer&& other) {
			buffer::operator=(std::move(other));
		}
		buffer& operator =(const buffer&) = delete;
		buffer& operator =(buffer&& other) {
			this->settings_v = other.settings_v;
			this->logical_device_v = other.logical_device_v;
			this->pointer_v = other.pointer_v;
			other.pointer_v = {};
			basic_buffer::operator =(std::move(other));
			other.buffer_v = 0;
			return *this;
		}

		// returns RAII lock object
		// requires buffer to have VK_BUFFER_USAGE_TRANSFER_DST_BIT 
		// and VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		memory_lock_t memory_lock();
	};

	template<typename T>
	class array_buffer : public buffer {
	public:
		using value_type = T;
		struct typed_memory_lock_t {
			using size_type = std::size_t;
			buffer::memory_lock_t lock;

			typed_memory_lock_t(array_buffer<T>* b) : lock(b) {}
			~typed_memory_lock_t() {}

			array_buffer<T>* buffer() {
				return (array_buffer<T>*)lock.buffer();
			}
			size_type size() {
				return buffer()->size();
			}

			value_type* begin() {
				return (value_type*)lock.begin();
			}
			value_type* end() {
				return (value_type*)lock.end();
			}
			const value_type* begin() const {
				return (value_type*)lock.begin();
			}
			const value_type* end() const {
				return (value_type*)lock.end();
			}
			const value_type* cbegin() const {
				return (value_type*)lock.begin();
			}
			const value_type* cend() const {
				return (value_type*)lock.end();
			}
		};
	protected:
	public:
		// size represents number of items to store
		array_buffer(const buffer_creation_settings& settings) 
			: buffer(
				settings.set_size(settings.size * sizeof(T)).set_alignment(alignof(T))
			) {
		}
		array_buffer() = default;
		~array_buffer() = default;
		array_buffer(array_buffer&&) = default;
		array_buffer& operator =(array_buffer&&) = default;

		// returns RAII lock object
		// requires buffer to have VK_BUFFER_USAGE_TRANSFER_DST_BIT 
		// and VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		typed_memory_lock_t memory_lock() {
			return typed_memory_lock_t{this};
		}

		std::size_t raw_size() const {
			return buffer::size();
		}
		std::size_t size() const {
			return buffer::size() / sizeof(T);
		}
	};
}

#endif // GSPLAT_VULKAN_BUFFER_H
