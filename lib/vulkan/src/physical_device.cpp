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
VkPhysicalDeviceMemoryProperties gsplat::vulkan::physical_device::memory_properties() const {
	VkPhysicalDeviceMemoryProperties o;
	vkGetPhysicalDeviceMemoryProperties(device_v, &o);
	return o;
}
logical_device_shared gsplat::vulkan::physical_device::create_logical_device(const queue_family_collection& queue_family_indices) {
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
	createInfo.enabledLayerCount = 0; // static_cast<uint32_t>(validationLayers.size());
	createInfo.ppEnabledLayerNames = nullptr; // validationLayers.data();

	if (vkCreateDevice(device_v, &createInfo, nullptr, &o_device) != VK_SUCCESS) {
		throw std::runtime_error("failed to create logical device!");
	}

	std::vector<command_queue> queues{queue_family_indices.size()};
	for (int i = 0; i < queue_family_indices.queue_families_v.size(); ++i) {
		VkQueue q;
		vkGetDeviceQueue(o_device, queue_family_indices.queue_families_v[i].first, 0, &q);
		queues[i] = command_queue{instance_ref_v, q, queue_family_indices.queue_families_v[i].first, this->shared_from_this()};
	}

	auto ptr = std::make_shared<logical_device>(
		instance_ref_v,
		o_device,
		this->shared_from_this(),
		queues
	);
	for (auto& v : queues) {
		v.source_logical_device_v = ptr;
	}
	return ptr;
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

physical_device_shared gsplat::vulkan::command_queue::physical_device() const {
	return source_physical_device_v.lock();
}
logical_device_shared gsplat::vulkan::command_queue::logical_device() const {
	if (source_logical_device_v.expired()) {
		throw std::runtime_error("Failed to query logical device from command_queue. Logical device has been destroyed");
	}
	return source_logical_device_v.lock();
}
VkInstance gsplat::vulkan::command_queue::instance_handle() const {
	return instance_v;
}
VkQueue gsplat::vulkan::command_queue::queue_handle() const {
	return queue_v;
}

gsplat::vulkan::logical_device::~logical_device() {
	{
		memory_v = {};
	}
	vkDestroyDevice(device_v, nullptr);
}
VkDevice gsplat::vulkan::logical_device::device_handle() const {
	return device_v;
}
VkInstance gsplat::vulkan::logical_device::instance_handle() const {
	return instance_ref_v;
}
physical_device_shared gsplat::vulkan::logical_device::physical_device() {
	return physical_device_v.lock();
}
const command_queue& gsplat::vulkan::logical_device::queue_at(std::size_t index) const {
	return queues_v[index];
}
logical_device::pointer logical_device::aligned_malloc(std::size_t alignment, std::size_t size) {
	return memory_v.aligned_malloc(alignment, size);
}
logical_device::pointer logical_device::malloc(std::size_t size) {
	return memory_v.aligned_malloc(16, size);
}
void logical_device::free(logical_device::pointer ptr) {
	memory_v.free(ptr);
}
void gsplat::vulkan::logical_device::init_heap(std::size_t size, const device_memory_settings& settings) {
	auto prop = this->physical_device()->memory_properties();
	std::vector<std::pair<int, VkMemoryType>> valid;
	for (int i = 0; i < prop.memoryTypeCount; i++) {
		if (settings.is_valid(prop.memoryTypes[i])) {
			valid.emplace_back(i, prop.memoryTypes[i]);
		}
	}
	if (valid.empty()) {
		throw std::runtime_error("no memory is valid");
	}
	std::pair<int, VkMemoryType>& target = valid[0];

	VkDeviceMemory dev_mem;

	VkMemoryAllocateInfo alloc_info{};
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.allocationSize = size;
	alloc_info.memoryTypeIndex = target.first;

	vkAllocateMemory(this->device_v, &alloc_info, nullptr, &dev_mem);
	memory_v = memory_manager{
		size,
		dev_mem,
		target.second,
		prop.memoryHeaps[target.second.heapIndex],
		this
	};
}

bool device_memory_settings::is_valid(const VkMemoryType& s) const {
	if (this->memory_host_visible && !(s.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) return false;
	if (this->memory_host_coherent && !(s.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) return false;
	if (this->memory_device_local && !(s.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return false;
	return true;
}

gsplat::vulkan::memory_manager::~memory_manager() {
	if (memory_v) {
		vkFreeMemory(device_v->device_handle(), memory_v, nullptr);
		memory_v = nullptr;
	}
}
memory_manager::pointer memory_manager::aligned_malloc(std::size_t alignment, std::size_t size) {
	pointer ptr_o{};
	ptr_o.ptr = spans_v.end();
	ptr_o.device_memory_v = memory_v;
	ptr_o.device_v = device_v;

	for (auto i = spans_v.begin(); i != spans_v.end(); ++i) {
		auto& v = *i;
		if (v.in_use) continue;
		std::size_t align_start = v.start;
		if (align_start % alignment != 0) {
			align_start += alignment - align_start % alignment;
		}
		std::size_t s_end = align_start + size;
		if (v.end < s_end) continue;
		// valid span. now need to split.
		// if realigned, then insert before
		if (align_start != v.start) {
			span_desc_t d{};
			d.in_use = false;
			d.start = v.start;
			d.end = align_start;
			spans_v.insert(i, d);
		}
		// if not full size, then insert after
		if (s_end != v.end) {
			span_desc_t d{};
			d.in_use = false;
			d.start = s_end;
			d.end = v.end;
			auto j = i;
			++j;
			spans_v.insert(j, d);
		}

		v.in_use = true;
		v.start = align_start;
		v.end = s_end;

		ptr_o.ptr = i;
		break;
	}

	if (ptr_o.ptr == spans_v.end()) {
		throw std::runtime_error("cannot malloc. heap full");
	}
	return ptr_o;
}

void gsplat::vulkan::memory_manager::free(pointer ptr) {
	auto i = ptr.ptr;
	if (!i->in_use) {
		throw std::runtime_error("cannot free memory that is already freed");
	}
	i->in_use = false;

	//merge back
	if (i != spans_v.begin()) {
		auto j = i;
		--j;
		if (!j->in_use) {
			j->end = i->end;
			spans_v.erase(i);
			i = j;
		}
	}
	//merge forward
	if (i != spans_v.end()) {
		auto j = i;
		++j;
		if (!j->in_use) {
			i->end = j->end;
			spans_v.erase(j);
		}
	}
}