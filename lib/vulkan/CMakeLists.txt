include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/vulkan.cpp"
)
set(PROJECT_HEADERS
	"include/gsplat/vulkan.h"
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

include(vulkan)
target_link_libraries(gsplat_vulkan PUBLIC Vulkan::Vulkan)
