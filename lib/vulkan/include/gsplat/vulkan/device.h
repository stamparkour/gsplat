#ifndef GSPLAT_VULKAN_DEVICE_H
#define GSPLAT_VULKAN_DEVICE_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <vector>
#include <queue>

namespace gsplat::vulkan {
	class logical_device;
	class command_queue;

	class queue_family_collection {
		friend class physical_device;
	public:
		// index, family name
		using value_type = std::pair<int, VkQueueFamilyProperties>;
		using reference = value_type&;
		using const_reference = const value_type&;
		using size_type = std::size_t;
		using iterator = std::vector<value_type>::iterator;
		using const_iterator = std::vector<value_type>::const_iterator;
	private:
		std::vector<value_type> queue_families_v;
	public:

		queue_family_collection() = default;
		queue_family_collection(const std::vector<VkQueueFamilyProperties>&);

		queue_family_collection& resize(size_type);
		size_type size() const;
		bool empty() const;

		iterator begin();
		iterator end();
		const_iterator begin() const;
		const_iterator end() const;
		const_iterator cbegin() const;
		const_iterator cend() const;

		//invokes int=ScoreFunc(const VkQueueFamilyProperties&)
		//sort based off higher score
		//removes if score is <0
		template<typename ScoreFunc>
		queue_family_collection sort(ScoreFunc&&) const;
	};
	class physical_device {
		friend class vulkan_context;
		VkPhysicalDevice device_v;
		VkInstance instance_ref_v;

	public:
		physical_device() = default;
		physical_device(VkInstance a, VkPhysicalDevice b) : instance_ref_v(a), device_v(b) {}
		~physical_device();

		VkPhysicalDevice device_handle() const;
		VkInstance instance_handle() const;

		VkPhysicalDeviceProperties device_properties_vulkan() const;
		VkPhysicalDeviceFeatures device_features_vulkan() const;
		queue_family_collection queue_families() const;

		logical_device create_logical_device(const queue_family_collection& queue_family_indices);
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

		//invokes int=ScoreFunc(const physical_device&)
		//sort based off higher score
		//removes if score is <0
		template<typename ScoreFunc>
		physical_device_collection sort(ScoreFunc&&) const;

		size_type size() const;
		bool empty() const;

		reference front();
		const_reference front() const;

		iterator begin();
		iterator end();
		const_iterator begin() const;
		const_iterator end() const;
		const_iterator cbegin() const;
		const_iterator cend() const;
	};

	class command_queue {
		friend class logical_device;
		VkQueue queue_v;
		int source_index;
		physical_device source_physical_device_v;
		logical_device* source_logical_device_v;
	public:
		command_queue(VkQueue queue_v, int source_index, physical_device source_physical_device_v, logical_device* source_logical_device_v) :
			queue_v(queue_v), source_index(source_index), source_physical_device_v(source_physical_device_v), source_logical_device_v(source_logical_device_v) {}
		command_queue() = default;
		~command_queue();
	};

	class logical_device {
		friend class physical_device;
		VkDevice device_v;
		VkInstance instance_ref_v;
		std::vector<command_queue> queues_v;
	public:
		logical_device() = default;
		logical_device(VkInstance a, VkDevice b, std::vector<command_queue>&& c) :
			instance_ref_v(a),
			device_v(b),
			queues_v(c) {
		}
		logical_device(VkInstance a, VkDevice b, const std::vector<command_queue>& c) :
			instance_ref_v(a),
			device_v(b),
			queues_v(c) {
		}

		~logical_device();

		logical_device(const logical_device&) = delete;
		logical_device(logical_device&&) = delete;
		logical_device& operator =(const logical_device&) = delete;
		logical_device& operator =(logical_device&&) = delete;

		VkDevice device_handle() const;
		VkInstance instance_handle() const;
		const command_queue& queue_at(std::size_t index) const;
	};
}

template<typename ScoreFunc>
inline gsplat::vulkan::queue_family_collection gsplat::vulkan::queue_family_collection::sort(ScoreFunc&& func) const {
	// score, index
	std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> queue{};

	for (int i = 0; i < size(); i++) {
		int v = func(queue_families_v[i].second);
		if (v >= 0) {
			queue.emplace(v, i);
		}
	}

	queue_family_collection out{};
	while (!queue.empty()) {
		const std::pair<int, int>& p = queue.top();
		out.queue_families_v.push_back(queue_families_v[p.second]);
		queue.pop();
	}
	return out;
}

template <typename ScoreFunc>
inline gsplat::vulkan::physical_device_collection gsplat::vulkan::physical_device_collection::sort(ScoreFunc&& func) const {
	// score, index
	std::priority_queue<std::pair<int,int>, std::vector<std::pair<int,int>>, std::greater<std::pair<int,int>>> queue{};

	for(int i = 0; i < size(); i++) {
		int v = func(devices_v[i]);
		if(v >= 0) {
			queue.emplace(v, i);
		}
	}

	physical_device_collection out{};
	while(!queue.empty()) {
		const std::pair<int,int>& p = queue.top();
		queue.pop();
		out.devices_v.push_back(devices_v[p.second]);
	}
	return out;
}

#endif // GSPLAT_VULKAN_DEVICE_H