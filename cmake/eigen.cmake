message("importing eigen...")

FetchContent_Declare(
	eigen
	GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
	GIT_TAG f2d3ea030bc542b8191ed7ff9ea482b89a512cdf
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)

set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(eigen)
set(BUILD_SHARED_LIBS ON)

if(eigen_POPULATED)
	message("eigen found!")
else()
	message("failed to find eigen")
endif()

set(eigen3_POPULATED ${eigen_POPULATED})
set(eigen3_SOURCE_DIR ${eigen_SOURCE_DIR})
set(eigen3_BINARY_DIR ${eigen_BINARY_DIR})