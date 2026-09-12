# Gaussian Splatting


# Build Test App

Make sure you have
- [Cmake >3.30](https://cmake.org/download/)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)

run 
``` sh
cmake -B build -Dgsplat_BUILD_APP=YES 
cmake --build build
```

# Build Documentation

Make sure you have
- [Cmake >3.30](https://cmake.org/download/)
- [doxygen](https://www.doxygen.nl/manual/install.html)
- Graphviz (optional)

run 
``` sh
cmake -B build -Dgsplat_BUILD_DOXYGEN=YES 
cmake --build build
```