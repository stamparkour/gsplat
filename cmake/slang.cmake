message("importing slang compiler...")

if(NOT gsplat_SLANG_COMPILER_PATH)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        FetchContent_Declare(
            slang_compiler
            URL "https://github.com/shader-slang/slang/releases/download/v2026.17.1/slang-2026.17.1-windows-x86_64.zip"
            # URL_HASH SHA256=  # Optional but recommended
            DOWNLOAD_EXTRACT_TIMESTAMP NEW                    # Prevents build issues from old archive dates
        )

        # Downloads and extracts the zip into the build folder during the configuration step
        FetchContent_MakeAvailable(slang_compiler)

        set(gsplat_SLANG_COMPILER_PATH "${slang_compiler_SOURCE_DIR}/bin/slangc.exe")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        FetchContent_Declare(
            slang_compiler
            URL "https://github.com/shader-slang/slang/releases/download/v2026.17.1/slang-2026.17.1-linux-x86_64.tar.gz"
            # URL_HASH SHA256=  # Optional but recommended
            DOWNLOAD_EXTRACT_TIMESTAMP NEW                    # Prevents build issues from old archive dates
        )

        # Downloads and extracts the zip into the build folder during the configuration step
        FetchContent_MakeAvailable(slang_compiler)

        set(gsplat_SLANG_COMPILER_PATH "${slang_compiler_SOURCE_DIR}/bin/slangc")
    endif()
endif()

