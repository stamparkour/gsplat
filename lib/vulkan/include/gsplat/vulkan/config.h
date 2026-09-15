#ifndef GSPLAT_VULKAN_CONFIG_H
#define GSPLAT_VULKAN_CONFIG_H

#include <gsplat/gsplat_config.h>

namespace gsplat::vulkan::config {

inline constexpr char* app_name = "gsplat_app";
inline constexpr int app_version_major = GSPLAT_VERSION_MAJOR;
inline constexpr int app_version_minor = GSPLAT_VERSION_MINOR;
inline constexpr int app_version_tweak = GSPLAT_VERSION_TWEAK;
inline constexpr char* engine_name = "gsplat";
inline constexpr int engine_version_major = GSPLAT_VERSION_MAJOR;
inline constexpr int engine_version_minor = GSPLAT_VERSION_MINOR;
inline constexpr int engine_version_tweak = GSPLAT_VERSION_TWEAK;

}

#endif // GSPLAT_VULKAN_CONFIG_H