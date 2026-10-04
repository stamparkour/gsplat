# Calling the training shaders

How to dispatch the training shaders from C++ and loop them into training. The math is in `docs/backward.pdf`. Shared formulas live in `lib/shaders/src/splat_math.h`.

Every shader follows the existing conventions: storage buffers only, settings in a one-element `StructuredBuffer` at the last binding, 1024 threads per group, entry point `computeMain`. Bindings are numbered in declaration order, so the tables below are the `bind(index, ...)` order. Dispatch `(threads + 1023) / 1024` groups.

## Which gradients add up

Two buffers hold gradients, and they're cleared at different times.

- **`raster_grads`** (9 floats per gaussian) is the 2D gradient for **one image**. `raster_backward` adds into it atomically, since many pixels hit the same gaussian. Clear it before every image. It can't sum over images, because 2D gradients only mean something for the camera they came from, and `project_backward` converts them using that image's camera.
- **`gradients`** (one `data::gaussian` per gaussian) is the 3D gradient. `project_backward` adds into it, and it's the same for every camera, so it sums over images. Clear it at the start of a step and apply it at the end. `images_per_step` sets how many images go into one step. 1 is what Inria does. All images is one step per epoch.

## Training loop

| # | when | shader | threads | reads | writes |
|---|:---|:---|:---|:---|:---|
| 1 | start of step | `calc_covariance` | gaussians | gaussians | covariance |
| 2 | start of step | `gradients.clear()` | | | |
| 3 | each image | `project_gaussians` | gaussians | gaussians, camera | `gaussian2d_t` per gaussian |
| 4 | each image | `sort_gaussians`, n passes | gaussians / 2 | `gaussian2d_t` | sorted index |
| 5 | each image | `raster_forward` | pixels | 3, 4, gaussians | image, pixel state |
| 6 | each image | `loss_l1` | pixels | image, photo | dL/dC, loss per pixel |
| 7 | each image | `raster_grads.clear()` | | | |
| 8 | each image | `raster_backward` | pixels | 3, 4, 5, 6, gaussians | raster_grads |
| 9 | each image | `project_backward` | gaussians | gaussians, raster_grads, camera | adds into gradients |
| 10 | end of step | `apply_gradient` | gaussians | gradients | gaussians |

`raster_forward` and `raster_backward` need `project_gaussians` to write, per gaussian: `mean` in **pixels** (`fx tx/tz + cx`, `fy ty/tz + cy`), `covariance_inv` = inverse of `J W Σ Wᵀ Jᵀ + 0.3 I` (backward.pdf section 4), and `depth` = `t.z`. `splat_math.h` has every piece of that: `to_camera`, `projection_jacobian`, `covariance2d`, `inverse2x2`, `project_mean`. If `project_gaussians` takes the same settings struct as `project_backward` (camera + count), one settings buffer per image serves both.

## Bindings

| shader | binding order |
|:---|:---|
| `calc_covariance` | `buffer_io` (gaussians), `settings_buffer` |
| `project_gaussians` | `buffer_i` (gaussians), `buffer_o` (gaussians_2d), `settings_buffer` |
| `sort_gaussians` | `buffer_i` (gaussians_2d), `index_buffer` (sorted_index), `settings_buffer` |
| `raster_forward` | `gaussians_2d`, `gaussians`, `sorted_index`, `image_out`, `pixel_state`, `settings_buffer` |
| `loss_l1` | `image`, `target`, `dL_dC`, `loss_per_pixel`, `settings_buffer` |
| `raster_backward` | `gaussians_2d`, `gaussians`, `sorted_index`, `pixel_state`, `dL_dC`, `raster_grads`, `settings_buffer` |
| `project_backward` | `gaussians`, `raster_grads`, `gradients`, `settings_buffer` |
| `apply_gradient` | `buffer_io` (gaussians), `buffer_gradient` (gradients), `settings_buffer` |

Element types: `image_out`, `image`, `dL_dC` are `glm::vec4` per pixel. `target` is the photo as `data::pixel` (RGBA8), one per pixel. `loss_per_pixel` is a `float` per pixel, already divided by 3 x pixels, so the image's loss is their sum (mean absolute error, 0 to 1). `raster_grads` is 9 `float`s per gaussian. `gradients` is a `data::gaussian` per gaussian, the layout `apply_gradient` already reads.

`project_gaussians.slang` isn't in `SLANG_COMPUTE_SOURCES` in `lib/shaders/CMakeLists.txt` yet, because it doesn't compile. Add it once it does, since `shader_store` loads it on startup.

## C++ host

It's all in `lib/train`. `shader_store` has one method per shader. Each one makes its settings buffer, binds that shader's buffers in the order above, dispatches and waits. The gaussians are always the database's `gaussian_set().buffer1_vulkan()`, and the gradients live in `buffer2_vulkan()`. `train()` runs the loop.

```cpp
gsplat::data::database db = gsplat::data::database::colmap(path, q);
gsplat::train::shader_store shaders{"shaders", &db, q};  // folder with the .slang.spv files

gsplat::train::train_settings ts{};
ts.intrinsics = glm::vec4(fx, fy, cx, cy);  // in pixels of the loaded images, nothing reads cameras.txt yet
ts.epochs = 100;
gsplat::train::train(&shaders, &db, q, ts);  // prints the loss every epoch
```

One image of one step, the way `train()` calls it:

```cpp
shaders->project_gaussians(&cam.world_to_camera, &cam.intrinsics, &cam.limits, &gaussians_2d);
shaders->sort_gaussians(&gaussians_2d, &sorted_index);
shaders->raster_forward(k, &background, near_plane, &gaussians_2d, &sorted_index, &image_out, &pixel_states);
shaders->loss_l1(k, &image_out, &dL_dC, &loss_per_pixel);            // then sum loss_per_pixel on the CPU
raster_grads.clear();                                                // 2D gradients, per image
shaders->raster_backwards(k, &background, near_plane, &gaussians_2d, &sorted_index, &pixel_states, &dL_dC, &raster_grads);
shaders->project_backwards(&cam.world_to_camera, &cam.intrinsics, &cam.limits, &raster_grads);  // adds into buffer2
// buffer2.clear() before the step's first image, apply_gradient(lr / images_per_step) after its last
```

| file | what's in it |
|:---|:---|
| `include/gsplat/train/shader_store.h` | the `shader_store` class, `pixel_state`, and comments on what each method expects (including how `project_backwards` works) |
| `src/shader_store.cpp` | how each shader gets called, including the sort |
| `include/gsplat/train.h` | `camera` (what both projections take), `train_settings` (epochs, `images_per_step`, learning rate, near plane, background, intrinsics) |
| `src/train.cpp` | `make_buffer<T>(q, size)` (every working buffer is made the same way, only type and size change), `make_camera` (COLMAP pose to `camera`), `record_epoch_loss`, and `train()` |

**The sort** (`shader_store::sort_gaussians`) is odd-even transition sort, GPU Gems 2 chapter 46.2. `sort_gaussians.slang` does one pass. The host runs n passes, alternating even pairs (0,1) (2,3) ... and odd pairs (1,2) (3,4) ..., so no two threads touch the same element. It never moves the 2D gaussians. It reorders `sorted_index` by depth, front to back, and the raster reads `gaussians_2d[sorted_index[k]]`. The cost is n dispatches per image, which is fine for a few thousand gaussians.

**The loss** lands in `loss_per_pixel`, already divided by 3 x pixels, so summing it gives the image's mean absolute error (0 to 1). `train()` averages that over the epoch and hands it to `record_epoch_loss(epoch, loss)`, which prints it for now. That's the place to add a CSV or a plot.

**Both projections take the same camera.** `project_gaussians` and `project_backwards` both get `world_to_camera`, `intrinsics` and `limits`, and `project_gaussians.slang` uses the same `splat_math.h` functions as `project_backward.slang`. That's what makes the gradients belong to the picture that was actually rendered. `calc_covariance` isn't part of the loop anymore, since `project_gaussians` builds Sigma from the quaternion and scale.

**Gradients in `buffer2`.** It starts out as a copy of the gaussians (`gaussian_set::colmap_txt` stores both), and `project_backwards` adds into it, so it gets cleared before the first image of every step.

**Cameras.** `make_camera` reads `image_pose` the way `image_set::colmap_txt` stores it: the quaternion as (qw, qx, qy, qz) in its (x, y, z, w) slots (the real part first, the opposite of `gaussian_t`), and `position` as COLMAP's TX TY TZ, which is the world-to-camera translation, not where the camera is.

**Learning rate.** `apply_gradient` is plain SGD with one rate for everything, so the rate depends on the scene. On the synthetic test scene, 1 converged smoothly, 3 converged faster but noisier, and 30 diverged.

**Tested on llvmpipe** (the CPU Vulkan driver), built straight from this branch with GCC 13, on a synthetic one-camera scene with 400 random gaussians at 160x120:
- The sort leaves a valid permutation in ascending depth for n = 1, 2, 3, 7, 64, 1000 and 1001, starting from a shuffled order with ties.
- The loss goes from 0.138 to 0.0087 in 300 epochs at learning rate 1, in about 16 s, with no validation errors.
- `project_gaussians.slang` matches the numpy reference to 5.6e-7.

**Building on Linux:** `shader_store.cpp` uses designated initializers that skip fields, which GCC's `-Wextra` warns about, so configure with `-Dgsplat_WARNING_AS_ERROR=OFF` until those are spelled out.

## Device requirement

`raster_backward` uses float atomic adds. Enable `VK_EXT_shader_atomic_float` and its `shaderBufferFloat32AtomicAdd` feature when creating the logical device. NVIDIA and AMD have it. Intel integrated GPUs don't, and Mesa's llvmpipe does. Without it the validation layer rejects `raster_backward`'s pipeline (VUID-VkShaderModuleCreateInfo-pCode-08740), and since the debug callback throws, the app stops there.

In `physical_device::create_logical_device`, before `vkCreateDevice` (this is what the llvmpipe test ran with):

```cpp
VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{};
atomic_float.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
atomic_float.shaderBufferFloat32AtomicAdd = VK_TRUE;
const char* device_extensions[] = { VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME };
createInfo.pNext = &atomic_float;
createInfo.enabledExtensionCount = 1;
createInfo.ppEnabledExtensionNames = device_extensions;
```

## Testing without the C++ host

`tools/test_shaders.py` runs `raster_forward`, `loss_l1`, `raster_backward` and `project_backward` through SlangPy on any Vulkan GPU and checks them against the numpy reference (`tools/check_e2e.py`), which is itself checked against finite differences. It also checks that `project_backward` adds into `gradients` instead of overwriting.

```
python3 -m venv ~/.venvs/gsplat && ~/.venvs/gsplat/bin/pip install slangpy numpy
~/.venvs/gsplat/bin/python tools/test_shaders.py            # picks a discrete GPU, else llvmpipe
```
