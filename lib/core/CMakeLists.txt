include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/core.cpp"
	"src/stb_image.cpp"
)
set(PROJECT_HEADERS
	"include/gsplat/gsplat_config.h.in"
	"include/gsplat/core/word_iterable.h"
	"include/gsplat/core/read_file.h"
	"include/gsplat/core.h"
)

# add the library
add_library(gsplat_core STATIC ${PROJECT_SOURCES} ${PROJECT_HEADERS})
add_library(gsplat::core ALIAS gsplat_core)
target_compile_features(gsplat_core PUBLIC cxx_std_20)

# adds a directory to search when finding header files
target_include_directories(gsplat_core 
	PUBLIC "include" 
	PUBLIC "${gsplat_BINARY_DIR}/lib/core/include"
)

# dependencies
# include(boost)
# target_link_libraries(core PUBLIC 
# 	Boost::asio
#	Boost::filesystem
#	Boost::thread
#	Boost::program_options
# )

# include(glew)
# target_link_libraries(core PUBLIC glew_s)

# include(glfw)
# target_link_libraries(core PUBLIC glfw)

# include(imgui)
# target_link_libraries(core PUBLIC imgui)

# include(miniaudio)
# target_link_libraries(core PUBLIC miniaudio)

# include(concurrentqueue)
# target_link_libraries(core PUBLIC concurrentqueue)

# set the configure file for the project

configure_file("include/gsplat/gsplat_config.h.in" "include/gsplat/gsplat_config.h")
