#include <gsplat/vulkan/vulkan_runtime.h>
#include <stdexcept>
#include <string>

using namespace ::gsplat::vulkan;
namespace config = ::gsplat::vulkan::config;

constexpr VkApplicationInfo static_appInfo{
    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pNext = nullptr,
    .pApplicationName = config::app_name,
    .applicationVersion = VK_MAKE_VERSION(config::app_version_major, conifg::app_version_minor, config::app_version_tweak),
    .pEngineName = config::engine_name,
    .engineVersion = VK_MAKE_VERSION(config::engine_version_major, conifg::engine_version_minor, config::engine_version_tweak),
    .apiVersion = VK_API_VERSION_1_0
};

vulkan_creation_extensions::vulkan_creation_extensions(const char*const* ext, int count)  {
    push_back_collection(ext, count);
}
void vulkan_creation_extensions::push_back(const char* v) {
    ext_v.push_back(v);
}
void vulkan_creation_extensions::push_back_collection(const char*const* ext, int count) {
    const char*const* v = ext;
    int i = 0;
    for(; i < count; static_cast<void>(++i),static_cast<void>(++v)) {
        ext_v.push_back(*v);
    }
}
const char* const * vulkan_creation_extensions::data() const {
    return ext_v.data();
}
size_t vulkan_creation_extensions::count() const {
    return ext_v.size();
}


vulkan_runtime::vulkan_runtime(const vulkan_creation_extensions& ext) {
    VkApplicationInfo appInfo = static_appInfo;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = ext.count();
    createInfo.ppEnabledExtensionNames = ext.data();
    createInfo.enabledLayerCount = 0;

    if(VkResult r =  vkCreateInstance(&createInfo, nullptr, &instance); r != VK_SUCCESS) {
        throw std::runtime_error(std::string{} + "failed to create vulkan instance: " + r);
    }
}