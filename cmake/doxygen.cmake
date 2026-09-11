message("importing doxygen...")

find_package(Doxygen)

if(Doxygen_FOUND)
	message("doxygen found!")
else()
	message("failed to find doxygen")
endif()
