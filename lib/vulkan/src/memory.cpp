#include <gsplat/vulkan/memory.h>
#include <gsplat/vulkan/vulkan2.h>
#include <stamp/task/coroutine.h>
#include <list>

using namespace gsplat::vulkan;

struct vulkan_memory_manager {
	struct node_t {
		std::size_t begin;
		std::size_t end;
	};
	std::list<node_t> device_local_allocs{};
	std::list<node_t> host_visibile_allocs{};
	bool initialized = false;


	vulkan_memory_manager() {

	}
	vulkan_memory_manager() {

	}
	void destroy() {

	}
	void init() {
		if (initialized) return;
		initialized = true;

		// can use 'this' pointer because object does not move.
		[this]() -> stamp::task::coroutine<void> {
			co_await vulkan_context::co_on_destroy();
			destroy();
		}().resume();

	}
	gptr_t alloc(std::size_t size, std::size_t align, memory_type) {
		init();
	}
	void free(gptr_t ptr) {

	}
};

static vulkan_memory_manager memory_v{};

gptr_t gsplat::vulkan::alloc_gpu_mem(std::size_t size, std::size_t align, memory_type type) {
	return memory_v.alloc(size, align, type);
}

void gsplat::vulkan::free_gpu_mem(gptr_t ptr) {
	memory_v.free(ptr);
}