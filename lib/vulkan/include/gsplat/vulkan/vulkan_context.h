#ifndef GSPLAT_VULKAN_VULKAN_CONTEXT_H
#define GSPLAT_VULKAN_VULKAN_CONTEXT_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <gsplat/vulkan/phyisical_device.h>
#include <vector>
#include <string>

namespace gsplat::vulkan {
	class vulkan_extension_vector {
		std::vector<std::string> ext_v;
	public:
		using value_type = std::string;
		using size_type = std::size_t;
		using iterator = std::vector<std::string>::iterator;
		using const_iterator = std::vector<std::string>::const_iterator;

		vulkan_extension_vector() = default;
		vulkan_extension_vector(const std::vector<std::string>& ext_v) : ext_v(ext_v) {}
		vulkan_extension_vector(std::vector<std::string>&& ext_v) : ext_v(ext_v) {}
		vulkan_extension_vector(const char* const* ext, int count);

		void insert(const char*);
		void insert(const std::string&);
		void insert_collection(const char* const* ext, int count);
		std::string const* data() const;
		size_type size() const;

		vulkan_extension_vector& union_equal(const vulkan_extension_vector& other);

		bool contains(const char*) const;
		bool contains(const std::string&) const;

		std::vector<const char*> to_extension_raw() const;

		iterator begin();
		iterator end();
		const_iterator begin() const;
		const_iterator end() const;
		const_iterator cbegin() const;
		const_iterator cend() const;
	};

	vulkan_extension_vector available_extensions();
	vulkan_extension_vector glfw_extensions();

	//  struct vulkan_context_creation {
	// };

	class vulkan_context {
		static vulkan_context* global_vulkan_v;
		VkInstance instance_v = nullptr;
		VkPhysicalDevice device_v = nullptr;
		VkDebugUtilsMessengerEXT debug_messenger_v = nullptr;
	public:
		vulkan_context(const vulkan_extension_vector& ext = {});
		~vulkan_context();

		vulkan_context(const vulkan_context&) = delete;
		vulkan_context(vulkan_context&&) = delete;
		vulkan_context& operator =(const vulkan_context&) = delete;
		vulkan_context& operator =(vulkan_context&&) = delete;

		VkInstance instance_handle() const;
		phyisical_device primary_device_handle() const;

		// static vulkan_context* global();
	};
}

#endif // GSPLAT_VULKAN_VULKAN_CONTEXT_H
