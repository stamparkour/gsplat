message("importing OpenCV...")

FetchContent_Declare(
	OpenCV
	URL https://github.com/opencv/opencv/archive/refs/tags/5.0.0.zip
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	OVERRIDE_FIND_PACKAGE 
)

set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(OpenCV)
set(BUILD_SHARED_LIBS ON)

if(opencv_POPULATED)
	message("OpenCV found!")
else()
	message("failed to find OpenCV")
endif()

# use ${OpenCV_LIBS} for target