#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <gsplat/vulkan/vulkan_context.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <iostream>
#include "validation_layers.h"

using namespace ::gsplat::vulkan;

//configurables
constexpr VkApplicationInfo static_appInfo {
	.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	.pNext = nullptr,
	.pApplicationName = config::app_name,
	.applicationVersion = VK_MAKE_VERSION(config::app_version_major, config::app_version_minor, config::app_version_patch),
	.pEngineName = config::engine_name,
	.engineVersion = VK_MAKE_VERSION(config::engine_version_major, config::engine_version_minor, config::engine_version_patch),
	.apiVersion = VK_API_VERSION_1_0
};

const std::vector<const char*> static_validation_layers = {
	"VK_LAYER_KHRONOS_validation"
};
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
		throw std::runtime_error(std::string{} + "vulkan layer error: " + pCallbackData->pMessage);
	}
	else if (messageSeverity & ~0) {
		std::cout << "validation layer: " << pCallbackData->pMessage << std::endl;
	}

	return VK_FALSE;
}
bool is_vulkan_device_good(VkPhysicalDevice device) {
	VkPhysicalDeviceProperties deviceProperties;
	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceProperties(device, &deviceProperties);
	vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

	return true;
}

// vulkan_extension_vector implementation

vulkan_extension_vector::vulkan_extension_vector(const char*const* ext, int count)  {
	insert_collection(ext, count);
}
void vulkan_extension_vector::insert(const char* v) {
	ext_v.push_back(v);
}
void vulkan_extension_vector::insert(const std::string& v) {
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
vulkan_extension_vector& vulkan_extension_vector::union_equal(const vulkan_extension_vector& other) {
	for (auto& v : other) {
		if (!contains(v)) {
			insert(v);
		}
	}
	return *this;
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
vulkan_extension_vector gsplat::vulkan::available_extensions() {
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
vulkan_extension_vector gsplat::vulkan::glfw_extensions() {
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;

	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	return vulkan_extension_vector{glfwExtensions, (int)glfwExtensionCount};
}

// helper functions

std::vector<const char*> get_validation_layer() {
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (auto& a : static_validation_layers) {
		bool is_found = false;
		for (auto& v : availableLayers) {
			if (strcmp(a, v.layerName) == 0) {
				is_found = true;
				break;
			}
		}
		if (!is_found) throw std::runtime_error(std::string{} + "requested vulkan protection layer not available: " + a);
	}

	return static_validation_layers;
}

static VkInstance create_vulkan_instance(const std::vector<const char*>& extensions, const std::vector<const char*>& layers) {
	VkApplicationInfo appInfo = static_appInfo;

	VkInstance instance_v = nullptr;
	VkInstanceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = extensions.size();
	createInfo.ppEnabledExtensionNames = extensions.data();
	createInfo.enabledLayerCount = layers.size();
	createInfo.ppEnabledLayerNames = layers.data();
	createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

	if (VkResult r = vkCreateInstance(&createInfo, nullptr, &instance_v); r != VK_SUCCESS) {
		throw std::runtime_error(std::string{} + "failed to create vulkan instance: " + std::to_string(r));
	}
	return instance_v;
}
static VkDebugUtilsMessengerEXT create_vulkan_debug_messanger(VkInstance instance_v, void* user_data) {
	VkDebugUtilsMessengerEXT out = nullptr;

	VkDebugUtilsMessengerCreateInfoEXT mCreateInfo{};
	mCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	mCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	mCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	mCreateInfo.pfnUserCallback = debugCallback;
	mCreateInfo.pUserData = user_data; // Optional

	if (vkCreateDebugUtilsMessengerEXT(instance_v, &mCreateInfo, nullptr, &out) != VK_SUCCESS) {
		throw std::runtime_error("failed to set up debug messenger!");
	}
	return out;
}
static VkPhysicalDevice get_default_vulkan_device(VkInstance instance_v) {
	VkPhysicalDevice device = VK_NULL_HANDLE;
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance_v, &deviceCount, nullptr);
	if (deviceCount == 0) {
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}
	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance_v, &deviceCount, devices.data());

	for (auto& d : devices) {
		if (is_vulkan_device_good(d)) {
			return d;
		}
	}

	throw std::runtime_error("no GPU is good for app!");
}

// vulkan_context implementation

vulkan_context::vulkan_context(const vulkan_extension_vector& ext_orig) {
	vulkan_extension_vector ext{ext_orig};
	ext.insert(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	ext.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

	vulkan_extension_vector available = available_extensions();
	for (auto& e : ext) {
		if (!available.contains(e)) {
			throw std::runtime_error(std::string{} + "requested vulkan extension not available: " + e);
		}
	}

	std::vector<const char*> raw = ext.to_extension_raw();
	std::vector<const char*> raw_layers = get_validation_layer();
	instance_v = create_vulkan_instance(raw, raw_layers);
	debug_messenger_v = create_vulkan_debug_messanger(instance_v, this);
	device_v = get_default_vulkan_device(instance_v);
}
gsplat::vulkan::vulkan_context::~vulkan_context() {
	if(debug_messenger_v) vkDestroyDebugUtilsMessengerEXT(instance_v, debug_messenger_v, nullptr);
	vkDestroyInstance(instance_v, nullptr);
}
VkInstance vulkan_context::instance_handle() const {
	return instance_v;
}
physical_device_collection vulkan_context::device_collection() const {
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance_v, &deviceCount, nullptr);
	if (deviceCount == 0) {
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}
	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance_v, &deviceCount, devices.data());

	std::vector<physical_device> dev_out{deviceCount};
	for(int i = 0; i < deviceCount; i++) {
		dev_out[i] = {instance_v, devices[i]};
	}

	return physical_device_collection{dev_out};
}