# requires boost serialization

message("importing cereal...")

FetchContent_Declare(
	cereal
	GIT_REPOSITORY https://github.com/USCiLab/cereal.git
	GIT_TAG 22a1b369f39be918ca79206a83c4facd759f9105
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
)

set(SKIP_PERFORMANCE_COMPARISON ON)
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(cereal)
set(BUILD_SHARED_LIBS ON)

if(cereal_POPULATED)
	message("cereal found!")
else()
	message("failed to find cereal")
endif()
