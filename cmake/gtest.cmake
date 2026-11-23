message(STATUS "importing gtest...")

FetchContent_Declare(
	GTest
	URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
	URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)

set(gtest_build_tests OFF)
set(gtest_build_samples OFF)
set(gtest_disable_pthreads OFF)

set(BUILD_SHARED_LIBS OFF)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(GTest)
set(BUILD_SHARED_LIBS ON)

if(gtest_POPULATED)
	message(STATUS "gtest found!")
else()
	message(SEND_ERROR "failed to find gtest")
endif()

