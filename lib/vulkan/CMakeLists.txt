include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/vulkan.cpp"
	"src/vulkan_context.cpp"
	"src/physical_device.cpp"
)
set(PROJECT_HEADERS
	"include/gsplat/vulkan.h"
	"include/gsplat/vulkan/config.h"
	"include/gsplat/vulkan/vulkan_context.h"
	"include/gsplat/vulkan/buffer.h"
	"include/gsplat/vulkan/compute_shader.h"
	"include/gsplat/vulkan/image.h"
	"include/gsplat/vulkan/vulkan_include.h"
	"include/gsplat/vulkan/physical_device.h"
	"include/gsplat/vulkan/command_queue.h"
	"include/gsplat/vulkan/logical_device.h"
)

# add the library
add_library(gsplat_vulkan STATIC ${PROJECT_SOURCES} ${PROJECT_HEADERS})
add_library(gsplat::vulkan ALIAS gsplat_vulkan)
target_compile_features(gsplat_vulkan PUBLIC cxx_std_20)

# adds a directory to search when finding header files
target_include_directories(gsplat_vulkan 
	PUBLIC "include" 
)

# dependencies
target_link_libraries(gsplat_vulkan PUBLIC gsplat::core)

include(glm)
target_link_libraries(gsplat_vulkan PUBLIC glm::glm)

include(vulkan)
target_link_libraries(gsplat_vulkan PUBLIC Vulkan::Vulkan)

include(glfw)
target_link_libraries(gsplat_vulkan PUBLIC glfw)
