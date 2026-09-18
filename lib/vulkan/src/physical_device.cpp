#include <gsplat/vulkan/physical_device.h>

using namespace gsplat::vulkan;

physical_device::~physical_device() {
	
}
VkPhysicalDevice physical_device::device_handle() const {
	return device_v;
}
VkInstance physical_device::instance_handle() const {
	return instance_ref_v;
}
VkPhysicalDeviceProperties physical_device::device_properties_vulkan() const {
    VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(device_v, &deviceProperties);
	return deviceProperties
}
VkPhysicalDeviceFeatures physical_device::device_features_vulkan() const {
	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceFeatures(device_v, &deviceFeatures);
	return deviceFeatures;
}
std::vector<VkQueueFamilyProperties> gsplat::vulkan::physical_device::queue_families_vulkan() const {
    uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device_v, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device_v, &queueFamilyCount, queueFamilies.data());
}

physical_device_collection::size_type physical_device_collection::size() const {
    return devices_v.size();
}
physical_device_collection::reference physical_device_collection::front() {
    return devices_v.front();
}
physical_device_collection::const_reference physical_device_collection::front() const {
    return devices_v.front();
}
physical_device_collection::iterator physical_device_collection::begin() {
	return devices_v.begin();
}
physical_device_collection::iterator physical_device_collection::end(){
	return devices_v.end();
}
physical_device_collection::const_iterator physical_device_collection::begin() const{
	return devices_v.begin();
}
physical_device_collection::const_iterator physical_device_collection::end() const{
	return devices_v.end();
}
physical_device_collection::const_iterator physical_device_collection::cbegin() const{
	return devices_v.cbegin();
}
physical_device_collection::const_iterator physical_device_collection::cend() const{
	return devices_v.cend();
}