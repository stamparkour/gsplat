cmake_minimum_required(VERSION 3.30)
# set the project name
project(gsplat VERSION 0.1.0 LANGUAGES C CXX)


#options
option(gsplat_BUILD_APP "" ON)
option(gsplat_BUILD_DOXYGEN "" OFF)
option(gsplat_SLANG_COMPILER_PATH "" OFF)

#config
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release")
set(GSPLAT_OUTPUT_DIRECTORY "$<$<CONFIG:Debug>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}>$<$<CONFIG:Release>:${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}>")
set(GSPLAT_EXE_DIRECTORY "${CMAKE_BINARY_DIR}/exe")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# set C++ version
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# include subdirectories
add_subdirectory(lib/core)
add_subdirectory(lib/data)
add_subdirectory(lib/vulkan)
add_subdirectory(lib/shaders)
add_subdirectory(lib/train)
add_subdirectory(lib/cli)
add_subdirectory(lib/gui)

add_library(gsplat_gsplat INTERFACE)
add_library(gsplat::gsplat ALIAS gsplat_gsplat)
target_link_libraries(gsplat_gsplat INTERFACE 
	gsplat::core
	gsplat::data
	gsplat::cli
	gsplat::gui
	gsplat::shaders
	gsplat::vulkan
	gsplat::train
)

if(gsplat_BUILD_APP)
	message("adding apps")
    #project warnings
    if (MSVC)
        add_compile_options(/W4 /WX)
        add_compile_options(/std:c++latest) 
    else()
        add_compile_options(-Wall -Wextra -pedantic -Werror)
    endif()

    #stack size
    #if (MSVC)
    #    add_link_options(/STACK:8388608)
    #endif()
	
    add_subdirectory(app/gsplat)
    add_subdirectory(test)
endif()

if(gsplat_BUILD_DOXYGEN)
    include(doxygen)
    if(Doxygen_FOUND)
        set(MY_DOXYGEN_SOURCES 
            ${PROJECT_SOURCE_DIR}/lib/
            ${PROJECT_SOURCE_DIR}/docs/
            ${PROJECT_SOURCE_DIR}/README.md
            ${PROJECT_SOURCE_DIR}/LICENSE.txt
        )

        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${PROJECT_SOURCE_DIR}/README.md)
        set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
        set(DOXYGEN_EXTRACT_ALL YES)
        set(DOXYGEN_EXTENSION_MAPPING txt=md)
        set(DOXYGEN_SHOW_NAMESPACES YES)
        set(DOXYGEN_INCLUDE_GRAPH YES)
        set(DOXYGEN_HAVE_DOT YES)
        set(DOXYGEN_CALL_GRAPH YES)
        set(DOXYGEN_RECURSIVE YES)
        set(DOXYGEN_SOURCE_BROWSER YES)

        doxygen_add_docs(gsplat_doc ${MY_DOXYGEN_SOURCES})
    endif()
endif()