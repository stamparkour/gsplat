include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/render.cpp"
)
set(PROJECT_HEADERS
	"include/gsplat/render.h"
	"include/gsplat/render/entry.h"
)

# add the library
add_library(gsplat_render STATIC ${PROJECT_SOURCES} ${PROJECT_HEADERS})
add_library(gsplat::render ALIAS gsplat_render)
target_compile_features(gsplat_render PUBLIC cxx_std_20)

# adds a directory to search when finding header files
target_include_directories(gsplat_render 
	PUBLIC "include" 
)

# dependencies
target_link_libraries(gsplat_render PUBLIC gsplat::core gsplat::data gsplat::vulkan)

include(glfw)
target_link_libraries(gsplat_render PUBLIC glfw)

include(vulkan)
target_link_libraries(gsplat_render PUBLIC Vulkan::Vulkan)

include(imgui)
target_link_libraries(gsplat_render PUBLIC imgui)

include(opencv)
target_link_libraries(gsplat_render PUBLIC ${OpenCV_LIBS})