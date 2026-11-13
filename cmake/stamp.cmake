message("importing stamp...")

FetchContent_Declare(
	stamp
	GIT_REPOSITORY https://github.com/stamparkour/StampEngine.git
	GIT_TAG master
	DOWNLOAD_EXTRACT_TIMESTAMP ON
	GIT_PROGRESS TRUE
	FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(stamp)

if(stamp_POPULATED)
	message("stamp found!")
else()
	message("failed to find stamp")
endif()