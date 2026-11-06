#include <gsplat/vulkan/memory.h>
#include <gsplat/vulkan/vulkan2.h>
#include <stamp/task/coroutine.h>
#include <list>

using namespace gsplat::vulkan;


gptr_t gsplat::vulkan::alloc_gpu_mem(std::size_t size, std::size_t align, memory_type type) {
	return memory_v.alloc(size, align, type);
}

void gsplat::vulkan::free_gpu_mem(gptr_t ptr) {
	memory_v.free(ptr);
}