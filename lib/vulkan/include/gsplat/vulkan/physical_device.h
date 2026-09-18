#ifndef GSPLAT_VULKAN_PHYSICAL_DEVICE_H
#define GSPLAT_VULKAN_PHYSICAL_DEVICE_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <gsplat/vulkan/logical_device.h>
#include <vector>
#include <queue>

namespace gsplat::vulkan {
	class physical_device {
		friend class vulkan_context;
		VkPhysicalDevice device_v;
		VkInstance instance_ref_v;

		physical_device() = default;
		physical_device(VkInstance a, VkPhysicalDevice b) : instance_ref_v(a), device_v(b) {}
	public:
		~physical_device();

		VkPhysicalDevice device_handle() const;
		VkInstance instance_handle() const;

		VkPhysicalDeviceProperties device_properties_vulkan() const;
		VkPhysicalDeviceFeatures device_features_vulkan() const;
		std::vector<VkQueueFamilyProperties> queue_families_vulkan() const;

		logical_device create_logical_device(const std::vector<int>& queue_family_indices);
	};

	class physical_device_collection {
		friend class vulkan_context;
		std::vector<physical_device> devices_v;

		physical_device_collection(const std::vector<physical_device>& devices_v) : devices_v(devices_v) {};
		physical_device_collection() = default;
	public:
		using value_type = physical_device;
		using reference = value_type&;
		using const_reference = const value_type&;
		using size_type = std::size_t;
		using iterator = std::vector<physical_device>::iterator;
		using const_iterator = std::vector<physical_device>::const_iterator;

		//invokes int=ScoreFunc(physical_device&)
		//sort based off higher score
		//removes if score is <0.
		template<typename ScoreFunc>
		physical_device_collection sort(ScoreFunc&&) const;

		size_type size() const;

		reference front();
		const_reference front() const;

		iterator begin();
		iterator end();
		const_iterator begin() const;
		const_iterator end() const;
		const_iterator cbegin() const;
		const_iterator cend() const;
	};
}

template <typename ScoreFunc>
inline physical_device_collection gsplat::vulkan::physical_device_collection::sort(ScoreFunc&& func) const {
	// score, index
	std::priority_queue<std::pair<int,int>, std::vector<std::pair<int,int>>, std::greater<std::pair<int,int>>> queue{};

	for(int i = 0; i < size(); i++) {
		int v = ScoreFunc(devices_v[i]);
		if(v >= 0) {
			queue.emplace(v, i);
		}
	}

	physical_device_collection out{};
	while(!queue.empty()) {
		std::pair<int,int>& p = queue.top();
		p.pop();
		out.devices_v.push_back(devices_v[p.second]);
	}
	return out;
}

#endif // GSPLAT_VULKAN_PHYSICAL_DEVICE_H