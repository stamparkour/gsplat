message("importing COLMAP ...")

include(boost)
include(eigen)

FetchContent_Declare(
	COLMAP 
	GIT_REPOSITORY https://github.com/colmap/colmap.git
	GIT_TAG d3ccaf358e00936db2bd290f2623a652b00e80bb
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)

set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(COLMAP )
set(BUILD_SHARED_LIBS ON)

if(benchmark_POPULATED)
	message("COLMAP found!")
else()
	message("failed to find COLMAP ")
endif()

# colmap::colmap