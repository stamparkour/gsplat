message("importing glm...")

FetchContent_Declare(
	glm
	GIT_REPOSITORY https://github.com/g-truc/glm.git
	GIT_TAG 6f14f4792a0cde5d0cf2c910506724d61cb95834
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)

set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(glm)
set(BUILD_SHARED_LIBS ON)

if(glm_POPULATED)
	message("glm found!")
else()
	message("failed to find glm")
endif()