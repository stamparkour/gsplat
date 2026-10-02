# Calling the training shaders

How to dispatch `raster_forward`, `loss_l1`, `raster_backward`, and `project_backward` from C++. The math is in `docs/backward.pdf`. Shared formulas live in `lib/shaders/src/splat_math.h`.

All four follow the existing conventions: storage buffers only, settings in a one-element `StructuredBuffer` at the last binding, 1024 threads per group, entry point `computeMain`. Bindings are numbered in declaration order, so the tables below are the `bind(index, ...)` order. Dispatch `(threads + 1023) / 1024` groups.

## One training step

| # | shader | threads | reads | writes |
|---|:---|:---|:---|:---|
| 1 | `calc_covariance` | gaussians | gaussians | covariance |
| 2 | `project_gaussians` | gaussians | gaussians, camera | `gaussian2d_t` per gaussian |
| 3 | `sort_gaussians` | gaussians / 2, repeated | `gaussian2d_t` | sorted index |
| 4 | `raster_forward` | pixels | 2, 3, gaussians | image, pixel state |
| 5 | `loss_l1` | pixels | image, photo | dL/dC, loss per pixel |
| 6 | clear `raster_grads` to 0 | | | |
| 7 | `raster_backward` | pixels | 2, 3, 4, 5, gaussians | raster_grads |
| 8 | `project_backward` | gaussians | gaussians, raster_grads, camera | gradients |
| 9 | `apply_gradient` | gaussians | gradients | gaussians |

`raster_forward` and `raster_backward` need `project_gaussians` to write, per gaussian: `mean` in **pixels** (`fx tx/tz + cx`, `fy ty/tz + cy`), `covariance_inv` = inverse of `J W Σ Wᵀ Jᵀ + 0.3 I` (backward.pdf section 4), and `depth` = `t.z`. `splat_math.h` has every piece of that: `to_camera`, `projection_jacobian`, `covariance2d`, `inverse2x2`, `project_mean`.

## Buffers

| shader | binding order |
|:---|:---|
| `raster_forward` | `gaussians_2d`, `gaussians`, `sorted_index`, `image_out`, `pixel_state`, `settings_buffer` |
| `loss_l1` | `image`, `target`, `dL_dC`, `loss_per_pixel`, `settings_buffer` |
| `raster_backward` | `gaussians_2d`, `gaussians`, `sorted_index`, `pixel_state`, `dL_dC`, `raster_grads`, `settings_buffer` |
| `project_backward` | `gaussians`, `raster_grads`, `gradients`, `settings_buffer` |

Element types: `image_out`, `image`, `dL_dC` are `glm::vec4` per pixel. `target` is the photo as `data::pixel` (RGBA8), one per pixel. `loss_per_pixel` is a `float` per pixel, and the loss is their sum. `raster_grads` is 9 `float`s per gaussian. `gradients` is a `data::gaussian` per gaussian, the layout `apply_gradient` already reads.

## Settings structs (C++ side)

```cpp
struct pixel_state { float T_final; int stop; };        // 8 bytes, one per pixel

struct raster_settings {                                // raster_forward and raster_backward, 32 bytes
    glm::vec4 background;                               // rgb, 0..1
    int width, height, gaussian_count;
    float near_plane;
};

struct loss_settings { int width, height; };            // 8 bytes

struct camera {                                         // 96 bytes
    glm::mat4 world_to_camera;                          // same matrix project_gaussians uses
    glm::vec4 intrinsics;                               // fx, fy, cx, cy in pixels (from cameras.txt, scaled to 320x240)
    glm::vec4 limits;                                   // 1.3 tan(fovx/2), 1.3 tan(fovy/2), near plane, 0
};

struct project_backward_settings {                      // 112 bytes
    camera cam;
    int gaussian_count;
    int pad[3];
};
```

## Device requirement

`raster_backward` uses float atomic adds. Enable `VK_EXT_shader_atomic_float` and its `shaderBufferFloat32AtomicAdd` feature when creating the logical device. NVIDIA and AMD have it. Intel integrated GPUs don't, and Mesa's llvmpipe does.

## Testing without the C++ host

`tools/test_shaders.py` runs these four kernels through SlangPy on any Vulkan GPU and checks them against the numpy reference (`tools/check_e2e.py`), which is itself checked against finite differences.

```
python3 -m venv ~/.venvs/gsplat && ~/.venvs/gsplat/bin/pip install slangpy numpy
~/.venvs/gsplat/bin/python tools/test_shaders.py            # picks a discrete GPU, else llvmpipe
```
