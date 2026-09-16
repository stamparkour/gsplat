#ifndef GSPLAT_VULKAN_VULKAN_CONTEXT_H
#define GSPLAT_VULKAN_VULKAN_CONTEXT_H

#include <gsplat/vulkan/config.h>
#include<vulkan/vulkan.h>
#include <vector>
#include<string>

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
		void insert_collection(const char* const* ext, int count);
		std::string const* data() const;
		size_type size() const;

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

	vulkan_extension_vector get_available_extensions();

	class vulkan_context {
		static vulkan_context* global_vulkan_v;
		VkInstance instance_v;
		VkPhysicalDevice device_v;
	public:
		vulkan_context(const vulkan_extension_vector& ext = {});
		~vulkan_context();

		VkInstance instance() const;
		VkPhysicalDevice primary_device() const;

		// static vulkan_context* global();
	};
}

#endif // GSPLAT_VULKAN_VULKAN_CONTEXT_H
