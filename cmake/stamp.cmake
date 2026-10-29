message("importing glm...")

FetchContent_Declare(
	stamp
	GIT_REPOSITORY https://github.com/stamparkour/StampEngine.git
	GIT_TAG master
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)

set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(stamp)
set(BUILD_SHARED_LIBS ON)

if(stamp_POPULATED)
	message("stamp found!")
else()
	message("failed to find stamp")
endif()