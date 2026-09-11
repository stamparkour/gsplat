message("importing boost...")

find_package(Boost 1.90.0 CONFIG)
if(NOT Boost_FOUND)
	message("Boost not found")
	message("Fetching Boost library...")
	FetchContent_Declare(
		Boost
		URL https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-cmake.tar.xz
		URL_HASH SHA256=aca59f889f0f32028ad88ba6764582b63c916ce5f77b31289ad19421a96c555f
		DOWNLOAD_EXTRACT_TIMESTAMP ON
	FIND_PACKAGE_ARGS
	)
	FetchContent_GetProperties(Boost)
	if(NOT Boost_POPULATED)
		message("Populating Boost library...")
		FetchContent_MakeAvailable(Boost)
	endif()
endif()

if(Boost_POPULATED)
	message("boost found!")
else()
	message("failed to find boost")
endif()

