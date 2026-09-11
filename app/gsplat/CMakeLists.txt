include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/source.cpp"
)
set(PROJECT_HEADERS
)


# add the executable
add_executable(gsplat_app ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_link_libraries(gsplat_app PUBLIC gsplat::gsplat)
target_compile_features(gsplat_app PUBLIC cxx_std_20)

# include(gtest)
# include(benchmark)
# target_link_libraries(gsplat_app PUBLIC  GTest::gtest_main benchmark::benchmark_main)