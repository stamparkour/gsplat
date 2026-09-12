message("importing glfw...")

FetchContent_Declare(
	glfw 
	GIT_REPOSITORY https://github.com/glfw/glfw.git
	GIT_TAG origin/master # should change to explicit commit
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(glfw)

if(glfw_POPULATED)
	message("glfw found!")
else()
	message("failed to find glfw")
endif()
