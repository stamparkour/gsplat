# Backward pass (derivation)

One section per arrow in `training.md` section 5. Each section ends with its result in the form the kernel computes it, and gets checked numerically before it touches a shader. Forward model and symbols are `training.md` section 4.

Checks: every section has a finite-difference script in `tools/check_*.py`. `tools/check_e2e.py` chains all of them through a tiny numpy renderer with the kernels' four-stage split (project forward, raster forward, raster backward, project backward), checks every parameter against finite differences of the whole render, and fits perturbed gaussians back to a target with Adam. `tools/check_all.sh` runs everything.

## 0. Conventions

- L is the scalar loss. `dL/dx` has the same shape as x.
- Per pixel unless a section says otherwise. Summing over pixels happens in the kernel (atomic adds).
- Gaussians at a pixel are indexed front to back, i = 1..N. `T_1 = 1`, `T_{N+1} = T_final`.
- Vectors are columns. For a matrix M, `dL/dM` is the matrix of partials `dL/dM_ab`.
- Symmetric matrices (Sigma, Sigma2d, conic): a gradient is always the full symmetric matrix, the one you get by treating every entry as independent. Buffers store its upper triangle as is, with no doubling. The factor of 2 only shows up if you differentiate with respect to a packed scalar that fills two entries (like b in `[[a, b], [b, c]]`), where `dL/db = 2 (dL/dM)_01`. We don't write formulas that way, and mixing the two conventions is the classic bug (section 3).

## 1. Blending: dL/dC -> dL/dcolor_i, dL/dalpha_i

Forward, one pixel, one channel:

```
C   = sum_{i=1..N} c_i alpha_i T_i  +  T_{N+1} bg
T_i = prod_{j<i} (1 - alpha_j)
```

Given `dL/dC` (3-vector, from the loss). Want `dL/dc_i` (3-vector) and `dL/dalpha_i` (scalar). For the must tier's L1, the mean of `|C - target|` over pixels and channels, `dL/dC = sign(C - target) / (3 H W)`.

The sum runs over exactly the gaussians the forward pass blended at this pixel. Ones skipped under 1/255 aren't in it, and neither is anything past early termination. The backward walk has to cover the same list, which is why it needs the per-pixel contributor count.

**1. dC/dc_i.** Only the i-th term contains `c_i`.

```
dC/dc_i = alpha_i T_i
```

**2. What depends on alpha_i.** `T_i` doesn't, since it's a product over j < i. Three things do: the i-th term directly, every term k > i through `T_k`, and the background through `T_{N+1}`.

**3. dT_k/dalpha_i for k > i.** `T_k` has exactly one factor of `(1 - alpha_i)`.

```
dT_k/dalpha_i = -T_k / (1 - alpha_i)
```

**4. dC/dalpha_i.**

```
dC/dalpha_i = c_i T_i  -  1/(1 - alpha_i) [ sum_{k>i} c_k alpha_k T_k  +  T_{N+1} bg ]
```

Every term in the bracket carries a factor `T_{i+1} = T_i (1 - alpha_i)`. Pull it out and call what's left `S_i`, the color behind i as seen from just behind i:

```
S_i = sum_{k>i} c_k alpha_k prod_{i<j<k} (1 - alpha_j)  +  prod_{i<j<=N} (1 - alpha_j) bg
```

The bracket is `T_i (1 - alpha_i) S_i`, the `(1 - alpha_i)` cancels, and

```
dC/dalpha_i = T_i (c_i - S_i)
```

Meaning: making i more opaque swaps some of what's behind it for its own color, scaled by how visible i is. If `c_i` already equals what's behind it, `alpha_i` doesn't matter.

**5. Recurrence.** Peel gaussian i+1 off the front of `S_i`.

```
S_N = bg
S_i = alpha_{i+1} c_{i+1} + (1 - alpha_{i+1}) S_{i+1}
```

It's the over operator run back to front, so one running 3-vector carries the whole sum.

**6. Recovering T_i.** `T_{i+1} = T_i (1 - alpha_i)`, so `T_i = T_{i+1} / (1 - alpha_i)`, starting from `T_{N+1} = T_final`. The forward clamp `alpha <= 0.99` keeps the divisor at least 0.01. In float32 the forward's early stop (T under 1e-4) matters too: division can't bring back a T that underflowed.

**7. Three channels.** Channels are independent, and the loss sees them through `dL/dC`.

```
dL/dc_i     = alpha_i T_i dL/dC          3-vector
dL/dalpha_i = T_i (c_i - S_i) . dL/dC    scalar, dot over channels
```

**Result (kernel form):** per pixel, given the saved `T_final` and contributor count N.

```
T = T_final
S = bg
for i = N down to 1:
    T = T / (1 - alpha_i)                   T is now T_i
    dL/dc_i  += alpha_i T dL/dC             atomic, per gaussian
    dL/dalpha = T (c_i - S) . dL/dC         this pixel only, feeds section 2
    S = alpha_i c_i + (1 - alpha_i) S       S is now S_{i-1}
```

`dL/dc_i` sums over every pixel the gaussian touches, so on the GPU it's an atomic add into a per-gaussian buffer. `dL/dalpha` belongs to this pixel alone (alpha depends on the pixel through G), so it's never stored and goes straight into section 2.

**Check:** `python3 tools/check_blend.py`

## 2. alpha_i -> G_i, o_i

Forward, o is the raw opacity before the sigmoid:

```
s_i     = sigmoid(o_i)
alpha_i = min(0.99, s_i G_i)      skipped entirely when below 1/255
```

In: `dL/dalpha_i` for one pixel, from section 1. Out: `dL/dG_i` (per pixel) and `dL/do_i` (per gaussian).

**Product.** Away from the clamp, `alpha_i = s_i G_i`.

```
dalpha_i/dG_i = s_i
dalpha_i/ds_i = G_i
```

**Sigmoid.** `sigmoid'(o) = s (1 - s)`, so

```
dalpha_i/do_i = G_i s_i (1 - s_i)
```

**The 0.99 clamp.** Where `s_i G_i > 0.99`, the true derivative of the min is zero for both inputs. Inria passes the gradient straight through instead and uses the unclamped formulas. With the exact zero, a gaussian saturated at its center only hears from its unsaturated edges. Choice: straight through, like Inria, since our forward and optimizer defaults are Inria's. Switching to exact is one `if`.

**The 1/255 skip.** Not differentiable, alpha jumps from 1/255 to 0. A skipped gaussian isn't in the pixel's list, so it gets nothing from that pixel. The backward kernel recomputes `G_i` and `alpha_i` instead of storing them, so its skip test has to be the same float expression as the forward's. If one side skips a gaussian the other keeps, section 1's T recovery divides by the wrong `(1 - alpha)` and every gaussian in front of it gets a wrong gradient.

**Result (kernel form):** inside section 1's loop, right after `dL/dalpha` for this pixel:

```
dL/dG   = s_i dL/dalpha                 this pixel only, feeds section 3
dL/ds_i += G_i dL/dalpha                atomic, per gaussian
```

then once per gaussian, after all pixels:

```
dL/do_i = s_i (1 - s_i) dL/ds_i
```

`s_i (1 - s_i)` doesn't depend on the pixel, so it comes out of the sum and gets applied once. It's also where Inria's code splits: their CUDA accumulates `dL/ds`, and PyTorch does the sigmoid backward.

**Check:** `python3 tools/check_alpha.py`

## 3. G_i -> mu2d, conic -> Sigma2d

Forward, per pixel p, with `Q = conic = Sigma2d^-1`:

```
d     = p - mu2d                  2-vector, pixels
power = -1/2 d^T Q d
G     = exp(power)
```

In: `dL/dG` for one pixel, from section 2. Out: `dL/dmu2d` (2), `dL/dconic` (2x2), then `dL/dSigma2d` (2x2).

Shorthand for this section: `w = G dL/dG`, `u = Q d`.

**mu2d.** `dG/dpower = G`. For symmetric Q the gradient of `d^T Q d` in d is `2 Q d`, so `dpower/dd = -Q d`, and `dd/dmu2d = -I`. The two minus signs cancel.

```
dL/dmu2d = w Q d = w u
```

**conic.** `power` is linear in Q, `d^T dQ d = tr(d d^T dQ)`, so

```
dL/dconic = -1/2 w d d^T
```

Symmetric already, so it's the full gradient in the section 0 sense. If you store Q packed as `(a, b, c)` and differentiate by b directly, you get `-w dx dy`, twice the 01 entry. `tools/check_conic.py` checks both views.

**Through the inverse.** Differentiate `Q Sigma2d = I`: `dQ Sigma2d + Q dSigma2d = 0`, so `dQ = -Q dSigma2d Q`. Then

```
dL          = tr(dL/dQ^T dQ) = tr(-Q dL/dQ^T Q dSigma2d)
dL/dSigma2d = -Q dL/dQ Q          (Q and dL/dQ both symmetric)
```

For a single pixel this collapses to `1/2 w u u^T`. The kernel shouldn't compute it per pixel, though (next).

**Where each piece runs.** mu2d and Q are per gaussian, the same at every pixel, and everything above is linear in `dL/dG`. So the raster kernel only accumulates

```
dL/dmu2d_i  += w u                 2 atomics
dL/dconic_i += -1/2 w d d^T        3 atomics, upper triangle as is
```

and the inverse step runs once per gaussian, in the preprocess backward:

```
dL/dSigma2d_i = -Q_i dL/dconic_i Q_i
```

Summing and then converting equals converting and then summing, because the conversion is linear.

**Units.** mu2d is in pixels, so `dL/dmu2d` is per pixel. Summed over views, `|dL/dmu2d|` is the densification signal, and Inria's 2e-4 threshold is in NDC units. Multiply by W/2 and H/2 before comparing, like gsplat does (training.md section 6).

**Check:** `python3 tools/check_conic.py`

## 4. Sigma2d -> Sigma, t

Forward, in view space (OpenCV axes):

```
t        = W mu + tvec
tx'      = tz clamp(tx/tz, -limx, limx)     limx = 1.3 tan(fovx/2), same for y
J        = [[fx/tz, 0,     -fx tx'/tz^2],
            [0,     fy/tz, -fy ty'/tz^2]]   2x3
M        = J W                              2x3
Sigma2d  = M Sigma M^T + 0.3 I
```

In: `V = dL/dSigma2d` (2x2, per gaussian, from section 3). Out: `dL/dSigma` (3x3) and the covariance's share of `dL/dt` (3). W is the camera rotation, and we don't optimize cameras, so it's a constant. So is the 0.3 I, and V passes through it unchanged.

**Sigma.** Sigma appears once, linearly: `dSigma2d = M dSigma M^T`.

```
dL        = tr(V dSigma2d) = tr(V M dSigma M^T) = tr(M^T V M dSigma)
dL/dSigma = M^T V M                  3x3, symmetric
```

**M.** M appears twice: `dSigma2d = dM Sigma M^T + M Sigma dM^T`. Transposing the second term inside the trace turns it into the first, so

```
dL/dM = 2 V M Sigma
```

This 2 is real, because M appears twice. It has nothing to do with the packed-storage 2 from section 0.

**J.** `M = J W`, so `dL/dJ = dL/dM W^T`:

```
D = dL/dJ = 2 V J Sigma_c,    Sigma_c = W Sigma W^T   (camera-space covariance)
```

Only four entries of J depend on t: D00, D02, D11, D12 are the ones that matter.

**t, unclamped.**

```
dJ00/dtz = -fx/tz^2
dJ02/dtx = -fx/tz^2        dJ02/dtz = 2 fx tx/tz^3
dJ11/dtz = -fy/tz^2
dJ12/dty = -fy/tz^2        dJ12/dtz = 2 fy ty/tz^3
```

**t, clamped.** When `|tx/tz| > limx`, `tx' = c tz` with `c = ±limx` fixed, so `J02 = -fx c/tz`:

```
dJ02/dtx = 0
dJ02/dtz = fx c/tz^2 = fx tx'/tz^3       half the unclamped value
```

Same for y. Choice: the exact derivative, which is what the finite-difference check sees. As far as I remember, Inria zeros the tx term but keeps the factor 2 on the tz term (to verify, section 8). The difference only touches gaussians whose centers sit outside 1.3x the field of view.

**Result (kernel form):** once per gaussian, in the project backward:

```
M         = J W
dL/dSigma = M^T V M
D         = 2 V J (W Sigma W^T)
kx        = 1 if |tx/tz| <= limx else 0        same for ky
dL/dtx    = -fx/tz^2 D02 kx
dL/dty    = -fy/tz^2 D12 ky
dL/dtz    = -fx/tz^2 D00 - fy/tz^2 D11 + (1 + kx) fx tx'/tz^3 D02 + (1 + ky) fy ty'/tz^3 D12
```

This `dL/dt` is only the covariance's share. Section 5 adds the share from mu2d before mapping back to mu.

**Check:** `python3 tools/check_cov2d.py`

## 5. mu2d -> t -> mu

Forward:

```
t    = W mu + tvec
mu2d = (fx tx/tz + cx,  fy ty/tz + cy)     real tx, ty, the FOV clamp only lives inside J
```

In: `g = dL/dmu2d` (2, per gaussian, from section 3) and the covariance's share of `dL/dt` (3, from section 4). Out: `dL/dmu` (3).

**mu2d -> t.** The derivative of the projection is J with the real tx, ty. That's where J comes from in the first place.

```
dmu2d/dt = [[fx/tz, 0,     -fx tx/tz^2],
            [0,     fy/tz, -fy ty/tz^2]]

dL/dt += (dmu2d/dt)^T g
```

Under the FOV clamp this matrix and section 4's J differ, since this one uses tx and section 4 uses tx'. The kernel can share J only for unclamped gaussians.

**t -> mu.** `t = W mu + tvec`, so `dt = W dmu` and

```
dL/dmu = W^T dL/dt
```

**Culling.** Gaussians the forward culls (behind the near plane, or touching no tile) never reach a pixel. Every gradient is zero, so the backward skips them.

**Result (kernel form):** once per gaussian, after section 4:

```
dL/dtx += fx/tz g_x
dL/dty += fy/tz g_y
dL/dtz += -(fx tx g_x + fy ty g_y)/tz^2
dL/dmu  = W^T dL/dt
```

At SH degree 1 and up, section 7 adds another `dL/dmu` term through the view direction. SH 0 has none.

**Check:** `python3 tools/check_mean.py`, which runs sections 4 and 5 together from mu.

## 6. Sigma -> q, ls

Forward, quaternion real part first, `q = (w, x, y, z)`:

```
q_hat = q / |q|
s     = exp(ls)
R     = [[1 - 2(y^2 + z^2),  2(xy - wz),        2(xz + wy)      ],
         [2(xy + wz),        1 - 2(x^2 + z^2),  2(yz - wx)      ],
         [2(xz - wy),        2(yz + wx),        1 - 2(x^2 + y^2)]]     (w, x, y, z from q_hat)
N     = R diag(s)
Sigma = N N^T = R diag(s^2) R^T
```

In: `U = dL/dSigma` (3x3, per gaussian, from section 4). Out: `dL/dq` (4), `dL/dls` (3).

**N.** Same shape as section 4's M, it appears twice: `dSigma = dN N^T + N dN^T`, so

```
dL/dN = 2 U N
```

**Scale.** `N = R diag(s)`, so `dL/ds_k` is the k-th diagonal entry of `R^T dL/dN`, which is `2 s_k (R^T U R)_kk`. Then `ds_k/dls_k = s_k`:

```
P        = R^T U R            U rotated into the gaussian's own axes
dL/dls_k = 2 s_k^2 P_kk
```

Meaning: each scale only hears the part of the gradient along its own axis.

**Rotation matrix.** `dL/dR = dL/dN diag(s) = 2 U R diag(s^2)`. Call it `Gr`.

**Quaternion.** Every entry of R is a quadratic in (w, x, y, z). Differentiating the nine entries and collecting terms:

```
dL/dw = 2 [ z (Gr10 - Gr01) + y (Gr02 - Gr20) + x (Gr21 - Gr12) ]
dL/dx = 2 [ y (Gr01 + Gr10) + z (Gr02 + Gr20) + w (Gr21 - Gr12) ] - 4x (Gr11 + Gr22)
dL/dy = 2 [ x (Gr01 + Gr10) + w (Gr02 - Gr20) + z (Gr12 + Gr21) ] - 4y (Gr00 + Gr22)
dL/dz = 2 [ w (Gr10 - Gr01) + x (Gr02 + Gr20) + y (Gr12 + Gr21) ] - 4z (Gr00 + Gr11)
```

These are gradients with respect to `q_hat`.

**Normalization.** `d(q/|q|)/dq = (I - q_hat q_hat^T) / |q|`, symmetric, so

```
dL/dq = (dL/dq_hat - q_hat (q_hat . dL/dq_hat)) / |q|
```

It's orthogonal to q, since stretching q doesn't change the rotation. The reference trainers do this step and the exp in PyTorch, so their CUDA never has it. Ours has to.

**Result (kernel form):** once per gaussian, after section 4:

```
q_hat, R, s   from q, ls as in the forward
P             = R^T U R
dL/dls_k      = 2 s_k^2 P_kk
Gr            = 2 U R diag(s^2)
dL/dq_hat     = the four lines above
dL/dq         = (dL/dq_hat - q_hat (q_hat . dL/dq_hat)) / |q|
```

**Check:** `python3 tools/check_rotscale.py`

## 7. color_i -> sh_i, mu

SH degree 0 is enough for the must tier. Forward, per channel:

```
color_i = max(0, C0 sh_i + 0.5),    C0 = 1 / (2 sqrt(pi)) = 0.28209479177387814
```

In: `dL/dcolor_i` (3, per gaussian, summed over pixels in section 1). Out: `dL/dsh_i` (3).

**Result (kernel form):** per channel,

```
dL/dsh_i = C0 dL/dcolor_i       0 on channels where C0 sh_i + 0.5 < 0
```

At degree 0 the color doesn't depend on the view direction, so there's no `dL/dmu` term.

**Degree 1 to 3 (should tier):** TODO. The basis depends on `dir = normalize(mu - cam_pos)`, so `dL/dmu += (I - dir dir^T) dL/ddir / |mu - cam_pos|`, the same normalization shape as the quaternion in section 6. What's left is `dL/ddir` for each basis function.

**Check:** `python3 tools/check_color.py`

## 8. Cross-check

Only after sections 1 to 7 are done: compare against the gsplat math supplement (arXiv 2312.02121) and `alphablend_shader.slang` in google/slang-gaussian-rasterization. Record any differences here.

- Section 1, to verify against Inria's `backward.cu`: as I remember it, their running sum (`accum_rec`) is `S_i` without the background, and they add the background as a separate term, `-T_final / (1 - alpha_i) * (bg . dL/dC)`. It comes out the same, because `prod_{i<j<=N} (1 - alpha_j) = T_final / T_{i+1}`. Starting S at bg folds that term in.
- Section 4, to verify against Inria's `computeCov2DCUDA` backward: as I remember it, under the FOV clamp they zero `dL/dtx` (their `x_grad_mul`) but keep `2 fx tx'/tz^3` on the tz term, where the exact value is half that. If so, their gradients differ from ours only for gaussians centered outside 1.3x the FOV, so mask those when diffing.
