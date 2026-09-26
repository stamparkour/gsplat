# Training design (Nick's lane)

Status: draft v2, 2026-09-22. Covers `lib/train`, the gaussian storage in `lib/data`, the dataset reader, and the math. Written against `dev/stamparkour` at 2106f2e. Every code finding below was compiled or run, every number was computed, and every default was read from the reference repos.

## 0. Scope for the deadline

The 4 week window ends around 2026-10-06. As of 2026-09-22 nothing dispatches on the GPU yet. Building strictly in order (host layer, then forward, then backward) puts the whole training path in the last few days, so this doc proposes decoupling the kernels from the host layer.

- Every kernel is a Slang module in `lib/shaders`. The math lives in plain Slang functions, and thin entry points wrap them for the C++ host. SlangPy can call the same functions from Python for testing, so one file serves both.
- A small Python harness in `python/` loads a scene, calls the Slang kernels, and checks them against PyTorch autograd and gsplat. Until our own kernels exist for them, the harness borrows PyTorch for the sort, the loss, and Adam.
- The C++ host catches up in parallel and dispatches the same kernels. Nothing gets written twice.

Cut line, in priority order:

| tier | deliverable | owner |
|:---|:---|:---|
| must | forward kernels render Inria's pretrained scenes from the dataset cameras with PSNR close to the published numbers | Elijah, tested in the harness |
| must | our backward kernels train a scene in the harness, L1 loss, fixed gaussian count, SH degree 0 | Nick |
| should | C++ host runs the forward kernels in the viewer. A CPU sort of downloaded keys is fine as a stopgap | Elijah |
| should | GPU sort kernel, SH degree 3, D-SSIM, densification | Nick |
| stretch | full training loop dispatched from C++ with our Adam kernel, benchmarks against gsplat | both |

## 1. What training needs from the rest of the repo

| need | comes from | state at 2106f2e |
|:---|:---|:---|
| `compute_shader` with dispatch and barriers | `lib/vulkan/compute_shader.h` | header only |
| typed storage `buffer<T>` with host map | `lib/vulkan/buffer.h` | header only |
| images on GPU | `lib/vulkan/image.h` | stub |
| gaussian storage | `lib/data/gaussian*.h` | struct exists, see section 3 |
| camera poses and sparse points | `lib/data/gaussian_set::colmap()` | declared, not implemented |
| slang to spirv in the build | `lib/shaders/CMakeLists.txt` | works with slangc 2026.17.1. The output is SPIR-V 1.5, which Vulkan accepts only from 1.2, and the context asks for 1.0 (section 9) |

## 2. Pipeline as compute dispatches

One training iteration on one camera:

1. **project** (per gaussian): world mean to view space, cull behind the near plane (Inria culls z <= 0.2, gsplat uses 0.01), 2D mean, 2D covariance with the 0.3 px dilation, conic (inverse 2D covariance), screen radius (3 sigma of the larger eigenvalue), depth, and the view-dependent color from SH.
2. **bin** (per gaussian): count the 16x16 tiles each gaussian overlaps, prefix sum, emit one key per overlap. Either 64-bit `(tile_id << 32) | float_bits(depth)` or a packed 32-bit key. 1080p at 16 px tiles needs 13 bits of tile id, which leaves 19 bits for quantized depth and halves sort bandwidth. The laptop GPU has `shaderInt64`, so both work.
3. **sort** keys. PyTorch in the harness first, then our kernel, bitonic before radix.
4. **tile ranges** (per key): find where each tile's run starts and ends.
5. **rasterize forward** (one workgroup per tile, one thread per pixel): walk the tile's run front to back, skip alpha < 1/255, clamp alpha at 0.99, stop when T < 1e-4. Save the per-pixel final T and the number of gaussians that contributed.
6. **loss** (per pixel): L1 first. D-SSIM (11x11 gaussian window, sigma 1.5) is a separable convolution with its own backward, so it sits in the should tier.
7. **rasterize backward** (same tiles, back to front): accumulate gradients w.r.t. 2D mean, conic, opacity, and color. Many pixels add into the same gaussian, so this needs float atomics, see below.
8. **project backward** (per gaussian): chain the 2D gradients back to the 3D mean, quaternion, log scale, and SH coefficients.
9. **adam step** (per parameter): elementwise.
10. every N iterations on the CPU side: **densify** (clone, split, prune), **opacity reset**, **SH degree bump**.

Steps 1, 2, 4, 5 are Elijah's. Steps 3, 6, 7, 8, 9, 10 are Nick's. We pair on 5 and 7 because 7 consumes exactly what 5 saves.

**Float atomics.** A native float atomic add in SPIR-V needs `shaderBufferFloat32AtomicAdd`. Nick's laptop GPU (Intel Iris Xe on Mesa) reports it as false. llvmpipe reports true, and NVIDIA supports it. A compare-and-swap loop on the float's bit pattern compiles to core SPIR-V with no extension (checked with slangc 2026.17.1). Plan: native adds when the feature is present, the CAS loop otherwise, and a subgroup reduction first so fewer atomics reach memory. Atomics also make gradient sums order-dependent, so tests compare with a tolerance.

## 3. Gaussian storage: proposed change

Today `gaussian` is `{ mat4 covariance; vec4 mean; }` and `gaussian_color` is `{ vec4 color; }`.

Two problems for training:

- A raw covariance can leave the positive semidefinite cone after one gradient step. The original stores a quaternion `q` and a log scale `ls` and builds `Sigma = R(q) diag(exp(ls))^2 R(q)^T`, so every value of the parameters is a valid covariance.
- A single `vec4 color` cannot represent view-dependent color. The original stores spherical harmonics up to degree 3, 16 coefficients times 3 channels = 48 floats.

Proposed **struct of arrays**, one `buffer<T>` per parameter, all indexed by gaussian id:

| buffer | type | per gaussian | notes |
|:---|:---|:---|:---|
| `mean` | vec4 | 1 | xyz, w unused |
| `rotation` | vec4 | 1 | quaternion wxyz, normalized in the shader |
| `log_scale` | vec4 | 1 | xyz, w unused |
| `opacity_logit` | float | 1 | sigmoid in the shader |
| `sh` | float | 48 | coefficient-major, 3 channels each, index `g*48 + k*3 + c` |

Why struct of arrays: Adam keeps two moment buffers per parameter with its own learning rate, which maps one to one onto buffers. Densification appends to every buffer the same way. It also sidesteps std430 layout traps, since a `vec3` array element is padded to 16 bytes. That's why xyz lives in a vec4 and SH is a plain float array.

SH degree 0 first (3 floats), degree 3 later. Treating the color model as a swappable module matches the "additional arguments" idea in `gaussian_color.h`, and an extra per-gaussian attribute is just one more buffer plus one more Adam pair.

Densification changes the gaussian count. Simplest v1: keep capacity headroom in every buffer (say 2x), do clone, split, and prune on the CPU with a fetch and store, and reallocate only when capacity runs out.

## 4. Forward model

```
q_hat    = q / |q|
Sigma    = R(q_hat) diag(exp(ls))^2 R(q_hat)^T
t        = W mu + tvec                     view space, OpenCV axes: x right, y down, z forward
tx', ty' = tx, ty with tx/tz and ty/tz clamped to 1.3x the half-FOV tangent (used only inside J)
mu2d     = (fx tx/tz + cx,  fy ty/tz + cy)
J        = [[fx/tz, 0, -fx tx'/tz^2],
            [0, fy/tz, -fy ty'/tz^2]]
Sigma2d  = J W Sigma W^T J^T  + 0.3 I      2x2
conic    = Sigma2d^-1
p        = (x + 0.5, y + 0.5)              pixel center, COLMAP convention
G_i(p)   = exp(-1/2 (p - mu2d)^T conic (p - mu2d))
alpha_i  = min(0.99, sigmoid(o_i) * G_i)   skipped when below 1/255
color_i  = max(0, SH(sh_i, normalize(mu_i - cam_pos)) + 0.5)
T_i      = prod_{j<i} (1 - alpha_j)        front to back
C(p)     = sum_i color_i alpha_i T_i  +  T_final * background
loss     = (1 - lambda) L1 + lambda (1 - SSIM),  lambda = 0.2
```

### Corrections to `docs/Math.md`

Numbers below come from a short numpy script, reproducible on request.

1. **Compositing.** Math.md weights gaussian i by `(1 - P_alpha_i)`, where P is described as the summed transparency of the closer gaussians. If P means the over-operator accumulation `A_i = A_{i-1} + (1 - A_{i-1}) alpha_i`, then `1 - A_i` equals `T_i` exactly and the formula is right, so it's worth writing it as the product to remove the ambiguity. If P is a plain sum, it breaks once alphas are large. Three gaussians at alpha 0.6 give the third a weight of -0.12 and the background -0.80, where the product gives 0.096 and 0.064. The background term should also use the accumulation over all n gaussians, including the last one.
2. **Pixel integral.** Integrating each gaussian over the pixel footprint is a real anti-aliasing technique, but the formula as written has two problems.
   - It integrates the *normalized* density, so a gaussian's alpha at a pixel shrinks with its screen area. At full opacity, a gaussian with a 5 px standard deviation comes out 0.6% opaque at its own center, and 0.04% at 20 px, where 3DGS gives 1.0. Opacity would change as the camera moves, and large opaque surfaces become impossible. Integrating the *unnormalized* gaussian (peak 1) fixes it. That equals the normalized mass times `2 pi sqrt(det Sigma2d)`, and it gives 0.997 at 5 px and 0.77 for a 0.55 px gaussian, which is exactly the anti-aliasing effect we'd want.
   - The product of marginal CDFs equals the rectangle integral only when `Sigma2d` is diagonal, and projected covariances almost never are. For a unit gaussian with correlation 0.9, the product is 0.37x the true pixel mass along the correlation and 73x across it.

   Recommendation: v1 evaluates at the pixel center with the 0.3 dilation, so it can be diffed against gsplat's default mode. The cheap principled upgrade is Mip-Splatting's 2D filter, which gsplat implements as `rasterize_mode="antialiased"`, so it stays diffable. Exact area integration (Analytic-Splatting) is a research extension.

## 5. Backward pass

Derived by hand first, then checked against autograd on tiny tensors before it touches a shader. The chain, in the order the shaders run it:

- dL/dC (from the loss) -> dL/dcolor_i and dL/dalpha_i, via the back-to-front walk with a running sum of what's behind i
- dL/dalpha_i -> dL/dG_i, dL/do_i
- dL/dG_i -> dL/dmu2d, dL/dconic -> dL/dSigma2d
- dL/dSigma2d -> dL/dSigma (through J and W) and dL/dt (J depends on t)
- dL/dmu2d -> dL/dt -> dL/dmu
- dL/dSigma -> dL/dq, dL/dls
- dL/dcolor_i -> dL/dsh (SH is linear in its coefficients), and also -> dL/dmu through the view direction

Things that are easy to miss:

- **Clamps kill gradients.** Color channels clamped at 0 get no gradient, and neither does J's x or y term while the field-of-view clamp is active. Inria passes gradients straight through the 0.99 alpha clamp.
- **Activations are ours.** The reference trainers apply exp, sigmoid, and quaternion normalization in PyTorch, so their CUDA kernels never differentiate them. Ours must: `dL/dls = dL/ds * s`, `dL/do = dL/dalpha * G * sig(o)(1 - sig(o))`, `dL/dq = (I - q_hat q_hat^T) dL/dq_hat / |q|`.
- **Symmetric matrices.** A kernel that stores 3 of the 4 entries of Sigma2d, or 6 of the 9 of Sigma, has to count the off-diagonal gradient twice. This is the most common bug in hand-written 3DGS backward passes.
- **Why 0.99.** The reverse walk recovers each T_i by dividing the saved final T by (1 - alpha). The forward clamp is what keeps that division away from zero.

TODO(Nick): write each jacobian out in `docs/backward.md`, one section per arrow above, with the shape of every term. After deriving, check against the gsplat math supplement (arXiv 2312.02121) and read `alphablend_shader.slang` in google/slang-gaussian-rasterization (Apache 2.0). It uses the same split this plan assumes: a hand-written custom backward (`[BackwardDerivative]`) for the reverse loop, undoing the blending one gaussian at a time, with `bwd_diff` generating the per-splat math inside it. Its sort is a CUDA kernel and it runs through slangtorch, so it is a reference, not a portable trainer.

## 6. Optimizer and schedule (Inria's current defaults)

| thing | value |
|:---|:---|
| optimizer | Adam, betas 0.9 / 0.999, eps 1e-15 |
| lr mean | 1.6e-4 x extent, exponential decay to 1.6e-6 x extent over 30k |
| lr SH DC / SH rest | 2.5e-3 / 1.25e-4 |
| lr opacity | 0.025 (Inria's 2023 release and gsplat use 0.05) |
| lr scale / rotation | 5e-3 / 1e-3 |
| scene extent | 1.1 x the largest distance from a camera center to the mean camera center |
| densify | every 100 iters from 500 to 15000 |
| densify statistic | mean norm of the 2D mean gradient over the views that saw the gaussian, **in NDC units**. Multiply pixel-space gradients by W/2 and H/2 the way gsplat does, or the 2e-4 threshold is off by a factor of about 500 at our image sizes |
| split vs clone | split if max scale > 0.01 x extent, else clone |
| prune | opacity < 0.005, and after the first reset also screen radius > 20 px or max scale > 0.1 x extent |
| opacity reset | every 3000 iters, opacity set to min(current, 0.01) |
| SH degree | +1 every 1000 iters up to 3 |
| iterations | 30k, with 7k as the quick look |

## 7. Dataset reader

Building COLMAP as a library pulls in Ceres, glog, and SQLite at minimum. We only read three small binary files, so our own reader is less code than the dependency. The Python harness can use COLMAP's `read_write_model.py` (BSD) in the meantime.

Files in `<scene>/sparse/0/`:

- `cameras.bin`: camera_id, model id, width, height, params (fx, fy, cx, cy for PINHOLE, plus distortion for other models)
- `images.bin`: image_id, quaternion qw qx qy qz, translation, camera_id, name, 2D points with 3D point ids
- `points3D.bin`: point_id, xyz, rgb, error, track

The text variants (`cameras.txt` etc.) are easier to eyeball, so the reader should handle both.

- **Axes.** COLMAP poses map world to camera with OpenCV axes: x right, y down, z forward. The songho projection-matrix derivation uses OpenGL axes: y up, looking down -z. Proposal: OpenCV axes everywhere in the repo, converting only at the viewer boundary, since the data and all the math above already use them.
- **Undistortion.** The 3DGS release datasets are already undistorted PINHOLE cameras. If we ever run COLMAP ourselves, `colmap image_undistorter` handles it.
- **Image size.** Resizing to a fixed 320x240 changes the aspect ratio and invalidates the intrinsics. Downscale by a factor instead, the 3DGS convention caps width at 1600 px, and scale fx, fy, cx, cy by the same factor.
- **Pixel type.** `data::pixel` uses `char` channels, which are signed on x86. Pixel values need `uint8_t`.
- **Split.** Hold out every 8th image as the test set, the 3DGS evaluation protocol, so our PSNR is comparable to published numbers.
- **Initial gaussians.** One per sparse point: mean = xyz, SH DC = (rgb/255 - 0.5) / 0.28209479, opacity 0.1 (logit about -2.197), log scale = log of the RMS distance to the 3 nearest neighbors on all three axes, identity rotation.
- **Principal point.** Inria's rasterizer ignores cx and cy and builds a symmetric frustum from the field of view. gsplat uses the full intrinsics, so gsplat is the oracle we diff against.

## 8. Verification plan

- **Harness.** SlangPy (maintained, 0.43.1 as of July 2026, runs on Vulkan and CUDA) calls our Slang kernels from Python. Slang's docs now list the older slangtorch workflow as deprecated. Run the PyTorch comparisons on the 4070.
- **Oracles.** Every Slang stage is diffed against PyTorch autograd on tiny inputs, and full images against gsplat (Apache 2.0). Match gsplat's constants when diffing. Its near plane is 0.01 where Inria's is 0.2, so read the clamp and threshold values from its CUDA sources rather than assuming Inria's.
- **llvmpipe.** The CPU Vulkan driver is slow but supports float atomic add, which makes it a useful place to debug the host path.
- **Matrix layout.** Before any projection math, upload a known non-symmetric matrix from glm and read element [0][1] back in a shader. glm is column-major and slangc runs with `-matrix-layout-column-major`, but the indexing semantics are worth one test.
- **Unit tests.** gtest already builds in `test/`. The reader and the CPU-side densify logic get tests there.

## 9. Findings and open questions for Elijah

Checked on 2026-09-22 with GCC 13.3, Clang 18, and slangc 2026.17.1.

1. `compute_shader` takes an `istream&`. Does it own the descriptor set layout, or does the caller bind buffers by index? Training binds up to about 15 buffers to one dispatch.
2. Can `buffer<T>` be resized, or do we allocate with headroom and track a live count? Section 3 assumes headroom.
3. Does `dispatch` record a barrier, or is that `sync_cpu_read`? An iteration chains about 9 dispatches and needs GPU-to-GPU barriers between them, not CPU round trips.
4. **API version.** The instance asks for Vulkan 1.0, but slangc emits SPIR-V 1.5, which Vulkan accepts only from 1.2. With validation on, the first shader module fails validation and the debug callback throws. Asking for `VK_API_VERSION_1_3` fixes it and also brings subgroup operations and `VkPhysicalDeviceFeatures2`, which the float atomics feature needs.
5. **Feature enabling.** Float atomics and friends are enabled through a `pNext` chain on `VkDeviceCreateInfo`. Enabling every 1.0 feature, as now, also turns on `robustBufferAccess`, which bounds-checks every buffer access and costs speed.
6. **Branch state.** 2106f2e does not compile. `device.h` and `physical_device.cpp` disagree after the design change: `queue_at`, `aligned_malloc`, and `malloc` have no matching declarations, `free` calls a member `std::vector` doesn't have, and a `weak_ptr` is assigned a `VkDeviceMemory`. Six errors counting the `queue_at` call in `source.cpp`. 0123f7a compiles, but GCC 13 and Clang 18 both reject it under the repo's default `-Wall -Wextra -pedantic -Werror`, over member init order, signed/unsigned comparisons, and unused parameters. MSVC untested. `add_compile_options` also reaches dependencies built through FetchContent, so `-Werror` applies to OpenCV and friends too.
7. **Device selection.** Both sort helpers build a `std::greater` priority queue, so the output is ascending and `front()` is the lowest score. `physical_device_collection::sort` also pops the queue before reading the reference it took from `top()`, so it reads the next element or freed storage. Net effect: `front()` is the second-lowest score, ties going to the later device. In a standalone repro, Nick's laptop (Iris Xe + llvmpipe, tied at 0) gets llvmpipe, the CPU renderer, and 4070 + iGPU + llvmpipe also gets llvmpipe. A discrete GPU plus exactly one other device picks the discrete GPU, by accident. Fix: the default `std::less`, and copy the pair before `pop()`.
8. **Validation layer is mandatory.** The context throws when `VK_LAYER_KHRONOS_validation` is missing, which it is on any machine without the SDK, Nick's laptop included. Suggest a debug-only switch. Throwing from inside the debug callback also unwinds through driver code, so logging and returning `VK_FALSE` is safer.
9. `memory_manager::free` dereferences `end()` when the freed span is the last one (the merge-forward step).
10. `compute_shader.h` reuses `buffer.h`'s include guard, so whichever is included second silently vanishes, and it uses `istream` without `std::`.
11. **Ownership of `lib/data`.** Your update says you're designing the database API, and this doc proposes the gaussian layout and the COLMAP reader. Let's agree who owns which part so we don't both write it.
12. **Camera axes.** Section 7 proposes OpenCV axes everywhere. Does the viewer plan assume OpenGL's?

## 10. Order of work on my side

By 2026-09-29:

1. agree this doc, the axes, and `lib/data` ownership with Elijah
2. harness skeleton: SlangPy on the 4070, COLMAP scene load, a pretrained `.ply` loaded into the section 3 buffers
3. `docs/backward.md`, every jacobian checked against torch

By 2026-10-03:

4. raster backward and project backward kernels, checked stage by stage
5. training in the harness: L1, fixed gaussian count, SH degree 0, PyTorch Adam

Then, as time allows: GPU sort kernel, C++ COLMAP reader with gtest, SH degree 3, D-SSIM, densification, Adam kernel.
