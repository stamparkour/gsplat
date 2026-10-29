#include <gsplat/vulkan/vulkan2.h>

using namespace gsplat::vulkan;

vulkan_context::vulkan_context(const vulkan_context_config&) {
	// init vulkan context here
}
vulkan_context::~vulkan_context() {
	// de-init vulkan context here
}
VkDevice vulkan_context::device() {
	// return 'global' logical device here
}
VkPhysicalDevice vulkan_context::physical_device() {
	// return 'global' physical device here
}
VkInstance vulkan_context::instance() {

}
VkQueue vulkan_context::queue() {

}
VkDescriptorPool vulkan_context::descriptor_pool() {

}