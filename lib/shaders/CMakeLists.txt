include(FetchContent)

include(glslangValidator)
include(slang)

#source files
set(GLSL_SOURCES
    "src/my_shader.comp.glsl"
)
set(SLANG_COMPUTE_SOURCES
    "src/my_shader.slang"
)

# 3. Loop through the list to create individual custom commands
set(GENERATED_SPV_FILES "")

foreach(SRC_FILE ${GLSL_SOURCES})
    # Resolve the absolute path of the source shader
    get_filename_component(ABS_SRC_FILE "${SRC_FILE}" ABSOLUTE)
    
    # Get the file name
    get_filename_component(FILE_NAME "${SRC_FILE}" NAME)
    
    # Define the output path in the build directory (e.g., "build/shaders/triangle.vert.spv")
    set(SPV_FILE "${GSPLAT_OUTPUT_DIRECTORY}/shaders/${FILE_NAME}.spv")
    
    # Create the custom rule for this specific file
    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND Vulkan::glslangValidator -V "${ABS_SRC_FILE}" -o "${SPV_FILE}"  --target-env vulkan1.1
        DEPENDS "${ABS_SRC_FILE}"
        COMMENT "Compiling ${FILE_NAME} to SPIR-V..."
    )
    
    # Append the resulting .spv file to our tracking list
    list(APPEND GENERATED_SPV_FILES ${SPV_FILE})
endforeach()

foreach(SRC_FILE ${SLANG_COMPUTE_SOURCES})
    # Resolve the absolute path of the source shader
    get_filename_component(ABS_SRC_FILE "${SRC_FILE}" ABSOLUTE)
    
    # Get the file name
    get_filename_component(FILE_NAME "${SRC_FILE}" NAME)
    
    # Define the output path in the build directory (e.g., "build/shaders/triangle.vert.spv")
    set(SPV_FILE "${GSPLAT_OUTPUT_DIRECTORY}/shaders/${FILE_NAME}.spv")
    
    # Create the custom rule for this specific file
    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND "${gsplat_SLANG_COMPILER_PATH}" "${ABS_SRC_FILE}" -target spirv -fvk-use-gl-layout -profile spirv_1_1 -entry computeMain -stage compute -matrix-layout-column-major -o "${SPV_FILE}"
        DEPENDS "${ABS_SRC_FILE}"
        COMMENT "Compiling ${FILE_NAME} to SPIR-V..."
    )
    
    # Append the resulting .spv file to our tracking list
    list(APPEND GENERATED_SPV_FILES ${SPV_FILE})
endforeach()

# add the library
add_library(gsplat_shaders INTERFACE ${GENERATED_SPV_FILES})
add_library(gsplat::shaders ALIAS gsplat_shaders)
