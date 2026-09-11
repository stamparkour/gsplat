message(STATUS "importing gtest...")

FetchContent_Declare(
	GTest
	URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)

set(BUILD_SHARED_LIBS OFF)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(GTest)
set(BUILD_SHARED_LIBS ON)

if(gtest_POPULATED)
	message(STATUS "gtest found!")
else()
	message(SEND_ERROR "failed to find gtest")
endif()

