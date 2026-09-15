# Gaussian Splatting


# Build Test App

Make sure you have
- [Cmake >3.30](https://cmake.org/download/)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

run 
``` sh
cmake -B build -Dgsplat_BUILD_APP=YES 
cmake --build build --target gsplat_app
```

# Build Documentation

Make sure you have
- [Cmake >3.30](https://cmake.org/download/)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
- [doxygen](https://www.doxygen.nl/manual/install.html)
- Graphviz (optional)

run 
``` sh
cmake -B build -Dgsplat_BUILD_DOXYGEN=YES 
cmake --build build --target gsplat_doc
```

# Configuration

Configuration setting to modify build

To prevent building the app and only configure the library, add `-Dgsplat_BUILD_APP=OFF` to the cmake configure command.

To provide your own slang compiler and prevent downloading slang, add `-Dgsplat_SLANG_COMPILER_PATH=<slang path>` to the cmake configure command.

