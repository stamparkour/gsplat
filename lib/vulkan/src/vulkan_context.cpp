#include <gsplat/vulkan/vulkan_context.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <iostream>

using namespace ::gsplat::vulkan;

constexpr VkApplicationInfo static_appInfo {
	.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	.pNext = nullptr,
	.pApplicationName = config::app_name,
	.applicationVersion = VK_MAKE_VERSION(config::app_version_major, config::app_version_minor, config::app_version_patch),
	.pEngineName = config::engine_name,
	.engineVersion = VK_MAKE_VERSION(config::engine_version_major, config::engine_version_minor, config::engine_version_patch),
	.apiVersion = VK_API_VERSION_1_0
};

const std::vector<const char*> validation_layers = {
	"VK_LAYER_KHRONOS_validation"
};

vulkan_extension_vector::vulkan_extension_vector(const char*const* ext, int count)  {
	insert_collection(ext, count);
}
void vulkan_extension_vector::insert(const char* v) {
	ext_v.push_back(v);
}
void vulkan_extension_vector::insert_collection(const char*const* ext, int count) {
	const char*const* v = ext;
	int i = 0;
	for(; i < count; static_cast<void>(++i),static_cast<void>(++v)) {
		ext_v.push_back(*v);
	}
}
std::string const * vulkan_extension_vector::data() const {
	return ext_v.data();
}
vulkan_extension_vector::size_type vulkan_extension_vector::size() const {
	return ext_v.size();
}
bool vulkan_extension_vector::contains(const char* str) const {
	for (auto& v : ext_v) {
		if (std::strcmp(str, v.data()) == 0) {
			return true;
		}
	}
	return false;
}
bool gsplat::vulkan::vulkan_extension_vector::contains(const std::string& s) const {
	return contains(s.data());
}
std::vector<const char*> vulkan_extension_vector::to_extension_raw() const {
	std::vector<const char*> o{};
	for (auto& v : *this) {
		o.push_back(v.data());
	}
	return o;
}
vulkan_extension_vector::iterator vulkan_extension_vector::begin() {
	return ext_v.begin();
}
vulkan_extension_vector::iterator vulkan_extension_vector::end() {
	return ext_v.end();
}
vulkan_extension_vector::const_iterator vulkan_extension_vector::begin() const {
	return ext_v.begin();
}
vulkan_extension_vector::const_iterator vulkan_extension_vector::end() const {
	return ext_v.end();
}
vulkan_extension_vector::const_iterator vulkan_extension_vector::cbegin() const {
	return ext_v.cbegin();
}
vulkan_extension_vector::const_iterator vulkan_extension_vector::cend() const {
	return ext_v.cend();
}
vulkan_extension_vector gsplat::vulkan::get_available_extensions() {
	uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
	std::vector<VkExtensionProperties> extensions(extensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

	vulkan_extension_vector ext{};
	for (const auto& extension : extensions) {
		ext.insert(extension.extensionName);
	}
	return ext;
}

static const char* const* get_validation_layers(int* count_out) {
#ifndef GSPLAT_VULKAN_DEBUG
	*count_out = 0;
	return nullptr;
#else

	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (auto& a : validation_layers) {
		bool is_found = false;
		for (auto& v : availableLayers) {
			if (strcmp(a, v.layerName) == 0) {
				is_found = true;
				break;
			}
		}
		if (!is_found) throw std::runtime_error(std::string{} + "requested vulkan protection layer not available: " + a);
	}
	*count_out = validation_layers.size();
	return validation_layers.data();
#endif
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
		throw std::runtime_error(std::string{} + "vulkan layer error: " + pCallbackData->pMessage);
	}
	else if (messageSeverity & ~0){
		std::cout << "validation layer: " << pCallbackData->pMessage << std::endl;
	}

	return VK_FALSE;
}

vulkan_context::vulkan_context(const vulkan_extension_vector& ext_orig) {
	vulkan_extension_vector ext{ext_orig};
	ext.insert(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#ifdef GSPLAT_VULKAN_DEBUG
	ext.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

	vulkan_extension_vector available = get_available_extensions();
	for (auto& e : ext) {
		if (!available.contains(e)) {
			throw std::runtime_error(std::string{} + "requested vulkan extension not available: " + e);
		}
	}

	std::vector<const char*> raw = ext.to_extension_raw();

	int layer_count;
	const char* const* layer_ptr = get_validation_layers(&layer_count);

	VkApplicationInfo appInfo = static_appInfo;

	VkInstanceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = raw.size();
	createInfo.ppEnabledExtensionNames = raw.data();
	createInfo.enabledLayerCount = layer_count;
	createInfo.ppEnabledLayerNames = layer_ptr;
	createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

	if(VkResult r = vkCreateInstance(&createInfo, nullptr, &instance_v); r != VK_SUCCESS) {
		throw std::runtime_error(std::string{} + "failed to create vulkan instance: " + std::to_string(r));
	}

	// creating callback for debug layers

	VkDebugUtilsMessengerCreateInfoEXT mCreateInfo{};
	mCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	mCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	mCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	mCreateInfo.pfnUserCallback = debugCallback;
	mCreateInfo.pUserData = nullptr; // Optional
}
gsplat::vulkan::vulkan_context::~vulkan_context() {
	vkDestroyInstance(instance_v, nullptr);
}
VkInstance gsplat::vulkan::vulkan_context::instance() const {
	return instance_v;
}
VkPhysicalDevice gsplat::vulkan::vulkan_context::primary_device() const {
	return device_v;
}