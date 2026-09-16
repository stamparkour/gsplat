message("importing OpenCV...")

FetchContent_Declare(
	OpenCV
	URL https://github.com/opencv/opencv/archive/refs/tags/5.0.0.zip
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	OVERRIDE_FIND_PACKAGE 
)

set(WITH_IPP OFF)
set(WITH_ADE OFF)
set(WITH_ITT OFF)
set(BUILD_opencv_dnn OFF)
set(BUILD_opencv_python2 OFF)
set(BUILD_opencv_python3 OFF)
set(BUILD_opencv_java OFF)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTS OFF)
set(BUILD_PERF_TESTS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_DOCS OFF)
set(BUILD_APPS OFF)
FetchContent_MakeAvailable(OpenCV)
set(BUILD_SHARED_LIBS ON)

if(opencv_POPULATED)
	message("OpenCV found!")
else()
	message("failed to find OpenCV")
endif()

# use ${OpenCV_LIBS} for target