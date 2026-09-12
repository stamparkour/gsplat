message("importing imgui...")

FetchContent_GetProperties(glfw)
if(NOT glfw_POPULATED)
	message(FATAL_ERROR "imgui requires glfw")
endif()
FetchContent_GetProperties(glew)
if(NOT glew_POPULATED)
	message(FATAL_ERROR "imgui requires glew")
endif()

FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui.git
	GIT_TAG origin/master # should change to explicit commit
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
	FetchContent_MakeAvailable(imgui)
	file(GLOB IMGUI_BASE_SOURCES "${imgui_SOURCE_DIR}/*.cpp")
	set(IMGUI_BACKEND_SOURCES 
		"${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
		"${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
	)
	add_library(imgui STATIC ${IMGUI_BASE_SOURCES} ${IMGUI_BACKEND_SOURCES})
	target_include_directories(imgui 
		PUBLIC "${imgui_SOURCE_DIR}"
	)
	target_link_libraries(imgui PUBLIC glfw glew_s)
endif()

if(imgui_POPULATED)
	message("imgui found!")
else()
	message("failed to find imgui")
endif()

