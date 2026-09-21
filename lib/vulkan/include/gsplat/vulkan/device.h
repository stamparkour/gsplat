#ifndef GSPLAT_VULKAN_DEVICE_H
#define GSPLAT_VULKAN_DEVICE_H

#include <gsplat/vulkan/config.h>
#include <gsplat/vulkan/vulkan_include.h>
#include <vector>
#include <queue>
#include <memory>
#include <list>

namespace gsplat::vulkan {
	class logical_device;
	class physical_device;
	class command_queue;
	using logical_device_shared = std::shared_ptr<logical_device>;
	using logical_device_weak = std::weak_ptr<logical_device>;
	using physical_device_shared = std::shared_ptr<physical_device>;
	using physical_device_weak = std::weak_ptr<physical_device>;

	class queue_family_collection {
		friend class physical_device;
		friend class logical_device;
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
	class physical_device : public std::enable_shared_from_this<physical_device> {
		friend class vulkan_context;
		
		
		VkPhysicalDevice device_v;
		VkInstance instance_ref_v;

	public:
		physical_device() = default;
		physical_device(VkInstance a, VkPhysicalDevice b) : instance_ref_v(a), device_v(b) {}
		~physical_device();

		physical_device(const physical_device&) = delete;
		physical_device(physical_device&&) = delete;
		physical_device& operator =(const physical_device&) = delete;
		physical_device& operator =(physical_device&&) = delete;

		VkPhysicalDevice device_handle() const;
		VkInstance instance_handle() const;

		VkPhysicalDeviceProperties device_properties_vulkan() const;
		VkPhysicalDeviceFeatures device_features_vulkan() const;
		queue_family_collection queue_families() const;
		VkPhysicalDeviceMemoryProperties memory_properties() const;

		logical_device_shared create_logical_device(const queue_family_collection& queue_family_indices);
	};

	class physical_device_collection {
		friend class vulkan_context;
		std::vector<physical_device_shared> devices_v;

		physical_device_collection(const std::vector<physical_device_shared>& devices_v) : devices_v(devices_v) {};
		physical_device_collection() = default;
	public:
		using value_type = physical_device_shared;
		using reference = value_type&;
		using const_reference = const value_type&;
		using size_type = std::size_t;
		using iterator = std::vector<physical_device_shared>::iterator;
		using const_iterator = std::vector<physical_device_shared>::const_iterator;

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
		friend class gsplat::vulkan::logical_device;
		friend class gsplat::vulkan::physical_device;
		VkQueue queue_v = nullptr;
		VkInstance instance_v = nullptr;
		int source_index = 0;
		physical_device_weak source_physical_device_v{};
		logical_device_weak source_logical_device_v{};
	public:
		command_queue(VkInstance instance_v, VkQueue queue_v, int source_index, physical_device_shared source_physical_device_v) :
			instance_v(instance_v), queue_v(queue_v), source_index(source_index), source_physical_device_v(source_physical_device_v) {}
		command_queue() = default;
		~command_queue() {}

		physical_device_shared physical_device() const;
		logical_device_shared logical_device() const;
		VkInstance instance_handle() const;
		VkQueue queue_handle() const;
	};
	class command_queue_collection {
	public:
		// index, family name
		using value_type = command_queue;
		using reference = value_type&;
		using const_reference = const value_type&;
		using size_type = std::size_t;
		using iterator = std::vector<value_type>::iterator;
		using const_iterator = std::vector<value_type>::const_iterator;
	private:
		std::vector<value_type> queue_families_v;
	public:

		command_queue_collection() = default;
		command_queue_collection(const std::vector<command_queue>&);

		size_type size() const;
		bool empty() const;

		iterator begin();
		iterator end();
		const_iterator begin() const;
		const_iterator end() const;
		const_iterator cbegin() const;
		const_iterator cend() const;

		reference at(std::size_t);
		const_reference at(std::size_t) const;

		//invokes int=ScoreFunc(const command_queue&)
		//sort based off higher score
		//removes if score is <0
		template<typename ScoreFunc>
		command_queue_collection sort(ScoreFunc&&);
	};


	struct device_memory_settings {
		bool memory_host_visible;
		bool memory_host_coherent;
		bool memory_device_local;

		bool is_valid(const VkMemoryType&) const;
		// bool is_valid(const VkMemoryHeap&) const;
	};
	class memory_manager {
		struct span_desc_t {
			bool in_use;
			std::size_t start;
			std::size_t end;
		};
	public:
		struct vulkan_ptr {
			std::list<span_desc_t>::iterator ptr;
			std::weak_ptr<memory_manager> device_memory_v;
			logical_device* device_v;
		};
		using pointer = vulkan_ptr;
	private:
		std::size_t heap_size = 0;
		VkDeviceMemory memory_v = nullptr;
		VkMemoryType type_v{};
		VkMemoryHeap heap_v{};
		logical_device* device_v = nullptr;
		std::list<span_desc_t> spans_v{};
	public:
		memory_manager() = default;
		memory_manager(logical_device* device_v, std::size_t size, const device_memory_settings&);
		~memory_manager();

		pointer aligned_malloc(std::size_t alignment, std::size_t size);
		void free(pointer);
	};

	class logical_device : public std::enable_shared_from_this<logical_device>{
		friend class gsplat::vulkan::physical_device;
	public:
		using pointer = memory_manager::pointer;
	private:
		VkDevice device_v;
		VkInstance instance_ref_v;
		physical_device_weak physical_device_v;
		command_queue_collection queues_v;
		std::vector<memory_manager> memory_v;
	public:
		logical_device() = default;
		logical_device(
			VkInstance a, 
			VkDevice b, 
			physical_device_shared c,
			const std::vector<command_queue>& d) :
			instance_ref_v(a),
			device_v(b),
			physical_device_v(c),
			queues_v(d) {}

		~logical_device();

		logical_device(const logical_device&) = delete;
		logical_device(logical_device&&) = delete;
		logical_device& operator =(const logical_device&) = delete;
		logical_device& operator =(logical_device&&) = delete;

		VkDevice device_handle() const;
		VkInstance instance_handle() const;
		physical_device_shared physical_device();
		command_queue_collection& queue_collection();
		const command_queue_collection& queue_collection() const;
		
		pointer aligned_malloc(std::size_t alignment, std::size_t size, const device_memory_settings&);
		// assumes 16 byte alignment
		pointer malloc(std::size_t size, const device_memory_settings&);
		void free(pointer);
	};
}

template<typename ScoreFunc>
inline gsplat::vulkan::queue_family_collection gsplat::vulkan::queue_family_collection::sort(ScoreFunc&& func) const {
	// score, index
	std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> queue{};

	for (int i = 0; i < size(); i++) {
		int v = func(&queue_families_v[i].second);
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
		int v = func(devices_v[i].get());
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