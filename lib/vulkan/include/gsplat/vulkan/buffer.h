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
		VkBuffer buffer_v;
		int size_v;
	public:
		basic_buffer(VkBuffer buffer_v, int size_v) :
			buffer_v(buffer_v), size_v(size_v) {}
		~basic_buffer();

		basic_buffer(const basic_buffer&) = delete;
		basic_buffer& operator =(const basic_buffer&) = delete;

		int size();
	};

	struct buffer_creation_settings {
		bool usage_transfer_src;
		bool usage_transfer_dst;
		bool usage_storage_buffer;
		device_memory_settings memory;
		// if size is 1, then sharing is exclusive
		std::vector<command_queue*> queues;
	};

	template<typename T>
	class buffer : public basic_buffer {
	public:
		struct memory_lock_t {
			basic_buffer* buffer;
			char* begin_v;
			char* end_v;

			memory_lock_t(basic_buffer*);
			~memory_lock_t();

			char* begin();
			char* end();
			const char* begin() const;
			const char* end() const;
			const char* cbegin() const;
			const char* cend() const;
		};
	protected:
		buffer_creation_settings settings_v;
	public:
		buffer(const buffer_creation_settings&);
		~buffer();

		// returns RAII lock object
		// requires buffer to have VK_BUFFER_USAGE_TRANSFER_DST_BIT 
		// and VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		memory_lock_t memory_lock();
	};
}

#endif // GSPLAT_VULKAN_BUFFER_H
