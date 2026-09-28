include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/data.cpp"
)
set(PROJECT_HEADERS
	"include/gsplat/data/configure.h"
	"include/gsplat/data/gaussian.h"
	"include/gsplat/data/gaussian_set.h"
	"include/gsplat/data/gaussian_color.h"
	"include/gsplat/data/image.h"
	"include/gsplat/data.h"
	"include/gsplat/data/database.h"
	"include/gsplat/data/image_set.h"
)

# add the library
add_library(gsplat_data STATIC ${PROJECT_SOURCES} ${PROJECT_HEADERS})
add_library(gsplat::data ALIAS gsplat_data)
target_compile_features(gsplat_data PUBLIC cxx_std_20)

# adds a directory to search when finding header files
target_include_directories(gsplat_data 
	PUBLIC "include" 
)

# dependencies
target_link_libraries(gsplat_data PUBLIC gsplat::core gsplat::vulkan)

include(glm)
target_link_libraries(gsplat_data PUBLIC glm::glm)

include(eigen)
target_link_libraries(gsplat_data PUBLIC Eigen3::Eigen)

include(opencv)
target_link_libraries(gsplat_data PUBLIC ${OpenCV_LIBS})

# include(colmap)
# target_link_libraries(gsplat_data PUBLIC colmap::colmap)
