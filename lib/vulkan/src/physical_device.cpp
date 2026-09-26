#include <gsplat/vulkan/device.h>
#include <stdexcept>
#include <string>
#include "validation_layers.h"
#include <algorithm>

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
		VkCommandPool pool;
		VkCommandPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		pool_info.queueFamilyIndex = queue_family_indices.queue_families_v[i].first;

		vkCreateCommandPool(o_device, &pool_info, nullptr, &pool);
		queues[i] = command_queue{instance_ref_v, q, pool, queue_family_indices.queue_families_v[i].first, this->shared_from_this()};
	}

	auto ptr = std::make_shared<logical_device>(
		instance_ref_v,
		o_device,
		this->shared_from_this(),
		queues
	);
	for (auto& v : ptr->queues_v) {
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
VkCommandPool gsplat::vulkan::command_queue::pool_handle() {
	return pool_v;
}
int command_queue::queue_family() const {
	return this->source_queue_family_index;
}
command_queue_collection::command_queue_collection(const std::vector<command_queue>& v) {
	queue_families_v = v;
}
command_queue_collection::size_type command_queue_collection::size() const {
	return queue_families_v.size();
}
bool command_queue_collection::empty() const {
	return queue_families_v.empty();
}
command_queue_collection::iterator command_queue_collection::begin() {
	return queue_families_v.begin();
}
command_queue_collection::iterator command_queue_collection::end() {
	return queue_families_v.end();
}
command_queue_collection::const_iterator command_queue_collection::begin() const {
	return queue_families_v.begin();
}
command_queue_collection::const_iterator command_queue_collection::end() const {
	return queue_families_v.end();
}
command_queue_collection::const_iterator command_queue_collection::cbegin() const {
	return queue_families_v.cbegin();
}
command_queue_collection::const_iterator command_queue_collection::cend() const {
	return queue_families_v.cend();
}

command_queue_collection::reference command_queue_collection::at(std::size_t pos) {
	return queue_families_v.at(pos);
}
command_queue_collection::const_reference command_queue_collection::at(std::size_t pos) const {
	return queue_families_v.at(pos);
}

gsplat::vulkan::logical_device::~logical_device() {
	{
		memory_v = {};
	}
	for (int i = 0; i < queues_v.size(); i++) {
		vkDestroyCommandPool(device_handle(), queues_v.at(i).pool_handle(), nullptr);
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
command_queue_collection &gsplat::vulkan::logical_device::queue_collection() {
    return queues_v;
}
const command_queue_collection &gsplat::vulkan::logical_device::queue_collection() const {
    return queues_v;
}
device_memory_collection &gsplat::vulkan::logical_device::memory_collection() {
    return memory_v;
}
const device_memory_collection &gsplat::vulkan::logical_device::memory_collection() const {
    return memory_v;
}

gsplat::vulkan::device_memory::~device_memory() {
	if (memory_v) {
		vkFreeMemory(device_v->device_handle(), memory_v, nullptr);
		memory_v = nullptr;
	}
}
device_memory::device_memory(logical_device* device_v, std::size_t size, const std::vector<device_memory_settings>& ordered_settings) {
	auto prop = device_v->physical_device()->memory_properties();
	
	struct entry_t {
		int score;
		int heap_index;
		int type_index;;
		VkMemoryType* type;
		VkMemoryHeap* heap;
	};
	std::vector<entry_t> valid{prop.memoryHeapCount};

	for(int i = 0; i < prop.memoryHeapCount; i++) {
		valid[i].heap_index = i;
		valid[i].heap = &prop.memoryHeaps[i];
		valid[i].score = -1;
	}
	for(int i = 0; i < prop.memoryTypeCount; i++) {
		int index = prop.memoryTypes[i].heapIndex;
		valid[index].type = &prop.memoryTypes[i];
		valid[index].type_index = i;
	}
	for(auto& v : valid) {
		if(!v.type || !v.heap) {
			v.score = -1;
			continue;
		}
		if(size > v.heap->size) {
			v.score = -1;
			continue;
		}
		for(int i = 0; i < ordered_settings.size(); i++) {
			const auto& settings = ordered_settings[i];
			if(settings.is_valid(v.type, v.type_index, v.heap)) {
				v.score = i;
				break;
			}
		}
	}

	std::sort(valid.begin(), valid.end(), [](const entry_t& a, const entry_t& b) {
		if(a.score == b.score) {
			return a.heap->size < b.heap->size;
		}
		return a.score > b.score;
	});
	
	VkDeviceMemory dev_mem;
	while(true) {
		if(valid.empty()) {
			throw std::runtime_error("Failed to allocate device memory. No valid memory types found");
		}
		if(valid.back().score < 0) {
			valid.pop_back();
			continue;
		}

		VkMemoryAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = size;
		alloc_info.memoryTypeIndex = valid.back().type_index;

		if(vkAllocateMemory(device_v->device_handle(), &alloc_info, nullptr, &dev_mem) != VK_SUCCESS) {
			valid.pop_back();
			continue;
		}

		break;
	}

	this->source_type_index = valid.back().type_index;
	heap_size = size;
	heap_free_size = size;
	memory_v = dev_mem;
	type_v = *valid.back().type;
	heap_v = *valid.back().heap;
	creation_settings = ordered_settings[valid.back().score];
	this->device_v = device_v;
	spans_v.emplace_back(false, 0, size);	
}
device_memory::pointer device_memory::aligned_malloc(std::size_t alignment, std::size_t size) {
	pointer ptr_o{};
	ptr_o.ptr = spans_v.end();
	ptr_o.device_memory_v = this;

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

	// if (ptr_o.ptr == spans_v.end()) {
	// 	throw std::runtime_error("cannot malloc. heap full");
	// }
	return ptr_o;
}
device_memory::pointer device_memory::malloc(std::size_t size) {
    return aligned_malloc(device_memory_alignment, size);
}

void gsplat::vulkan::device_memory::free(pointer ptr) {
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
bool device_memory::is_valid_ptr(pointer ptr) {
	return ptr.device_memory_v != nullptr && ptr.ptr != ptr.device_memory_v->spans_v.end() && ptr.ptr->in_use;
}
device_memory* device_memory::get_memory_ptr(pointer ptr) {
	return ptr.device_memory_v;
}
std::size_t device_memory::size() const {
	return heap_size;
}
logical_device* device_memory::device() const {
    return this->device_v;
}
bool device_memory::is_valid(const device_memory_settings& settings) const {
	return settings.is_valid(&this->type_v, source_type_index, &this->heap_v);
}
VkDeviceMemory device_memory::memory_handle() const {
	return memory_v;
}
const VkMemoryType& device_memory::type_vulkan() const {
	return type_v;
}
const VkMemoryHeap& device_memory::heap_vulkan() const {
	return heap_v;
}
device_memory *device_memory_collection::create(std::size_t size, const std::vector<device_memory_settings>& ordered_settings) {
	auto ptr = std::make_unique<device_memory>(logical_device_v, size, ordered_settings);
	auto s = ptr->heap_vulkan().size;
	device_memory* raw_ptr = ptr.get();
	if (memory_v.size() == 0) {
		memory_v.emplace_back(std::move(ptr));
	}
	else {
		bool sucess = false;
		for (auto i = memory_v.begin(); i != memory_v.end(); ++i) {
			auto& v = *i;
			if (s >= v->heap_vulkan().size) {
				memory_v.emplace(i, std::move(ptr));
				sucess = true;
				break;
			}
		}
		if (!sucess) {
			memory_v.emplace_back(std::move(ptr));
		}
	}
	return raw_ptr;
}
device_memory_collection::pointer device_memory_collection::aligned_malloc(std::size_t alignment, std::size_t size, const device_memory_settings& settings) {
    for(auto i = find(settings); i != end(); ++i) {
		pointer p = (*i)->aligned_malloc(alignment, size);
		if(device_memory::is_valid_ptr(p)) {
			return p;
		}
	}
	std::size_t s = std::max(size, device_memory_min_creation_size);
	auto new_mem = create(s, {settings});
	return new_mem->aligned_malloc(alignment, size);
}
device_memory_collection::pointer device_memory_collection::malloc(std::size_t size, const device_memory_settings& settings) {
	return aligned_malloc(device_memory_alignment, size, settings);
}
void gsplat::vulkan::device_memory_collection::free(pointer ptr) {
	if (!ptr.device_memory_v) {
		throw std::runtime_error("cannot free memory. pointer is null");
	}
	if (ptr.device_memory_v->device() != logical_device_v) {
		throw std::runtime_error("cannot free memory. pointer is from a different logical device");
	}
	ptr.device_memory_v->free(ptr);
}
device_memory_collection::const_iterator device_memory_collection::begin() const {
    return memory_v.begin();
}
device_memory_collection::const_iterator device_memory_collection::end() const {
    return memory_v.end();
}
device_memory_collection::find_iterator_t device_memory_collection::find(const device_memory_settings& settings) const {
	find_iterator o{begin(), end(), settings};
	if(memory_v.size() != 0 && !(*begin())->is_valid(settings)) {
		++o;
	}
    return o;
}

bool device_memory_collection::find_iterator_t::operator ==(const const_iterator& other) const {
    return it == other;
}
bool device_memory_collection::find_iterator_t::operator !=(const const_iterator& other) const {
    return !(*this == other);
}
device_memory_collection::find_iterator_t& device_memory_collection::find_iterator_t::operator ++() {
	while (++it != end) {
		if ((*it)->is_valid(settings)) {
			break;
		}
	}
	return *this;
}
device_memory_collection::find_iterator_t device_memory_collection::find_iterator_t::operator ++(int) {
	find_iterator_t tmp = *this;
	++(*this);
	return tmp;
}
const std::unique_ptr<device_memory>* device_memory_collection::find_iterator_t::operator ->() {
	return &(*it);
}
const std::unique_ptr<device_memory>& device_memory_collection::find_iterator_t::operator *() {
	return *it;
}
bool device_memory_settings::is_valid(const VkMemoryType* type, int type_index, const VkMemoryHeap* heap) const {
	if (type != nullptr) {
		if (memory_host_visible && !(type->propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
			return false;
		}
		if (memory_host_coherent && !(type->propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
			return false;
		}
		if (memory_device_local && !(type->propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
			return false;
		}
	}

	if (type_bitmask != 0 && !(type_bitmask & (1 << type_index))) {
		return false;
	}

	return true;
}
bool gsplat::vulkan::device_memory::vulkan_ptr::operator==(nullptr_t) const {
    return ptr == device_memory_v->spans_v.end();
}