include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/test_main.cpp"
)
set(PROJECT_HEADERS
)


# add the executable
add_executable(gsplat_test ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_link_libraries(gsplat_test PUBLIC gsplat::gsplat)

enable_testing() # Enable testing in CMake

include(gtest)
target_link_libraries(gsplat_test PUBLIC GTest::gtest_main)


#google test required lines
include(GoogleTest)
gtest_discover_tests(gsplat_test)