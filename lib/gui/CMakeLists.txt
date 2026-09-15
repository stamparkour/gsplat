include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/gui.cpp"
)
set(PROJECT_HEADERS
	"include/gsplat/gui.h"
	"include/gsplat/gui/entry.h"
)

# add the library
add_library(gsplat_gui STATIC ${PROJECT_SOURCES} ${PROJECT_HEADERS})
add_library(gsplat::gui ALIAS gsplat_gui)
target_compile_features(gsplat_gui PUBLIC cxx_std_20)

# adds a directory to search when finding header files
target_include_directories(gsplat_gui 
	PUBLIC "include" 
)

# dependencies
target_link_libraries(gsplat_gui PUBLIC gsplat::core gsplat::data gsplat::vulkan)

include(glfw)
target_link_libraries(gsplat_gui PUBLIC glfw)

include(vulkan)
target_link_libraries(gsplat_gui PUBLIC Vulkan::Vulkan)

include(imgui)
target_link_libraries(gsplat_gui PUBLIC imgui)