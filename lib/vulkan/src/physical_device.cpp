#include <gsplat/vulkan/device.h>
#include <stdexcept>
#include <string>
#include "validation_layers.h"

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
	return deviceProperties;
}
VkPhysicalDeviceFeatures physical_device::device_features_vulkan() const {
	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceFeatures(device_v, &deviceFeatures);
	return deviceFeatures;
}
queue_family_collection physical_device::queue_families() const {
    uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device_v, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device_v, &queueFamilyCount, queueFamilies.data());
	return {queueFamilies};
}
logical_device gsplat::vulkan::physical_device::create_logical_device(const queue_family_collection& queue_family_indices) {
	float queuePriority = 1.0f;

	VkDevice o_device;
	VkDeviceQueueCreateInfo queueCreateInfo{};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = queue_family_indices.queue_families_v.front().first;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	VkPhysicalDeviceFeatures features = device_features_vulkan();
	std::vector<const char*> validationLayers = get_validation_layer();

	VkDeviceCreateInfo createInfo{};
	createInfo.pQueueCreateInfos = &queueCreateInfo;
	createInfo.queueCreateInfoCount = 1;
	createInfo.pEnabledFeatures = &features;
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
	createInfo.ppEnabledLayerNames = validationLayers.data();

	if (vkCreateDevice(device_v, &createInfo, nullptr, &o_device) != VK_SUCCESS) {
		throw std::runtime_error("failed to create logical device!");
	}

	std::vector<VkQueue> queues{queue_family_indices.size()};
	vkGetDeviceQueue(o_device, queue_family_indices.queue_families_v.front().first, 0, &(queues[0]));

	return {instance_ref_v, o_device,};
}

physical_device_collection::size_type physical_device_collection::size() const {
    return devices_v.size();
}
bool physical_device_collection::empty() const {
	return devices_v.empty();
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

queue_family_collection::queue_family_collection(const std::vector<VkQueueFamilyProperties>& v) {
	for (int i = 0; i < v.size(); i++) {
		queue_families_v.emplace_back(i, v[i]);
	}
}
queue_family_collection& queue_family_collection::resize(queue_family_collection::size_type count) {
	if (count > size()) throw std::runtime_error(std::string{} + 
		"cannot resize queue_family_collection larger that size: " + 
		std::to_string(count) + ">" + std::to_string(size()));
	queue_families_v.resize(count);
	return *this;
}
queue_family_collection::size_type queue_family_collection::size() const {
	return queue_families_v.size();
}
bool queue_family_collection::empty() const {
	return queue_families_v.empty();
}

queue_family_collection::iterator queue_family_collection::begin() {
	return queue_families_v.begin();
}
queue_family_collection::iterator queue_family_collection::end() {
	return queue_families_v.end();
}
queue_family_collection::const_iterator queue_family_collection::begin() const {
	return queue_families_v.begin();
}
queue_family_collection::const_iterator queue_family_collection::end() const {
	return queue_families_v.end();
}
queue_family_collection::const_iterator queue_family_collection::cbegin() const {
	return queue_families_v.cbegin();
}
queue_family_collection::const_iterator queue_family_collection::cend() const {
	return queue_families_v.cend();
}