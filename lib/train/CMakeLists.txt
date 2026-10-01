include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/train.cpp"
	"src/shader_store.cpp"
)
set(PROJECT_HEADERS
	"include/gsplat/train.h"
	"include/gsplat/train/shader_store.h"
)

# add the library
add_library(gsplat_train STATIC ${PROJECT_SOURCES} ${PROJECT_HEADERS})
add_library(gsplat::train ALIAS gsplat_train)
target_compile_features(gsplat_train PUBLIC cxx_std_20)

# adds a directory to search when finding header files
target_include_directories(gsplat_train 
	PUBLIC "include" 
)

# dependencies
target_link_libraries(gsplat_train PUBLIC gsplat::core gsplat::vulkan gsplat::data)