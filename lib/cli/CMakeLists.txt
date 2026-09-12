include(FetchContent)

#source files
set(PROJECT_SOURCES
	"src/cli.cpp"
)
set(PROJECT_HEADERS
	"include/gsplat/cli/entry.h"
	"include/gsplat/cli.h"
)

# add the library
add_library(gsplat_cli STATIC ${PROJECT_SOURCES} ${PROJECT_HEADERS})
add_library(gsplat::cli ALIAS gsplat_cli)
target_compile_features(gsplat_cli PUBLIC cxx_std_20)

# adds a directory to search when finding header files
target_include_directories(gsplat_cli 
	PUBLIC "include" 
)

# dependencies
target_link_libraries(gsplat_cli PUBLIC gsplat::core gsplat::train)

# include(boost)
# target_link_libraries(gsplat_cli PUBLIC Boost::program_options)
