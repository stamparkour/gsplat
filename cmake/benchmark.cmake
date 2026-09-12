message("importing google benchmark...")

FetchContent_Declare(
	benchmark
	GIT_REPOSITORY https://github.com/google/benchmark.git
	GIT_TAG bab0f02ecca7b543111bc24e3724e2599dfc8f33
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)

set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(benchmark)
set(BUILD_SHARED_LIBS ON)

if(benchmark_POPULATED)
	message("benchmark found!")
else()
	message("failed to find benchmark")
endif()