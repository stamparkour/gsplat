"""Reference and finite-difference check for view dependent color (SH degrees 1 to 3).

The color a camera at c sees is rgb + sum_k sh_k Y_k(d), with d = (mu - c) / |mu - c|
and no clamp, like the plain rgb the kernels already use. Degree 0 is the
gaussian's own rgb, so the 15 coefficients here are degrees 1 to 3 only.
Same real SH basis, signs and constants as the 3DGS reference renderer.

Run: python3 tools/check_sh.py
"""
import numpy as np

C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)
ACTIVE = (0, 3, 8, 15)  # coefficients in use at degree 0, 1, 2, 3


def basis(d, degree):
    """Y, shape (15,), at unit direction d. Zero past degree."""
    x, y, z = d
    xx, yy, zz = x * x, y * y, z * z
    Y = np.array([
        -C1 * y, C1 * z, -C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy),
        C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
        C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
        C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)])
    Y[ACTIVE[degree]:] = 0
    return Y


def basis_grad(d, degree):
    """dY/dd, shape (15, 3), with x, y, z treated as free (the normalization comes after)."""
    x, y, z = d
    xx, yy, zz = x * x, y * y, z * z
    G = np.array([
        [0, -C1, 0], [0, 0, C1], [-C1, 0, 0],
        [C2[0] * y, C2[0] * x, 0],
        [0, C2[1] * z, C2[1] * y],
        [-2 * C2[2] * x, -2 * C2[2] * y, 4 * C2[2] * z],
        [C2[3] * z, 0, C2[3] * x],
        [2 * C2[4] * x, -2 * C2[4] * y, 0],
        [6 * C3[0] * x * y, C3[0] * (3 * xx - 3 * yy), 0],
        [C3[1] * y * z, C3[1] * x * z, C3[1] * x * y],
        [-2 * C3[2] * x * y, C3[2] * (4 * zz - xx - 3 * yy), 8 * C3[2] * y * z],
        [-6 * C3[3] * x * z, -6 * C3[3] * y * z, C3[3] * (6 * zz - 3 * xx - 3 * yy)],
        [C3[4] * (4 * zz - 3 * xx - yy), -2 * C3[4] * x * y, 8 * C3[4] * x * z],
        [2 * C3[5] * x * z, -2 * C3[5] * y * z, C3[5] * (xx - yy)],
        [C3[6] * (3 * xx - 3 * yy), -6 * C3[6] * x * y, 0]], dtype=float)
    G[ACTIVE[degree]:] = 0
    return G


def view_color(rgb, sh, mu, c, degree):
    """sh has shape (15, 3): coefficient k of channel c is sh[k, c]."""
    v = mu - c
    return rgb + basis(v / np.linalg.norm(v), degree) @ sh


def view_color_backward(sh, mu, c, degree, dL_dcolor):
    """Gradients of dL_dcolor . view_color: (d rgb, d sh, d mu)."""
    v = mu - c
    n = np.linalg.norm(v)
    d = v / n
    dsh = np.outer(basis(d, degree), dL_dcolor)
    dd = basis_grad(d, degree).T @ (sh @ dL_dcolor)
    dmu = (dd - d * (d @ dd)) / n  # through d = v / |v|
    return dL_dcolor, dsh, dmu


def check_orthonormal(rng):
    """The 15 functions should be orthonormal on the sphere, which catches a wrong constant or term."""
    d = rng.normal(size=(20_000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Y = np.stack([basis(p, 3) for p in d])
    gram = 4 * np.pi * Y.T @ Y / len(Y)
    err = np.max(np.abs(gram - np.eye(15)))
    print(f"orthonormal on the sphere (20k samples) max err {err:.3f}  {'ok' if err < 0.1 else 'MISMATCH'}")
    return err < 0.1


def check_gradients(rng, degree, samples=300, eps=1e-6):
    worst = {"rgb": 0.0, "sh": 0.0, "mu": 0.0}
    for _ in range(samples):
        rgb = rng.uniform(0, 1, 3)
        sh = rng.normal(scale=0.3, size=(15, 3))
        mu = rng.normal(size=3)
        c = rng.normal(size=3) * 3
        g = rng.normal(size=3)
        drgb, dsh, dmu = view_color_backward(sh, mu, c, degree, g)

        def loss(rgb_, sh_, mu_):
            return g @ view_color(rgb_, sh_, mu_, c, degree)

        num_mu = np.array([(loss(rgb, sh, mu + eps * e) - loss(rgb, sh, mu - eps * e)) / (2 * eps) for e in np.eye(3)])
        num_rgb = np.array([(loss(rgb + eps * e, sh, mu) - loss(rgb - eps * e, sh, mu)) / (2 * eps) for e in np.eye(3)])
        num_sh = np.zeros((15, 3))
        for k in range(15):
            for ch in range(3):
                e = np.zeros((15, 3))
                e[k, ch] = eps
                num_sh[k, ch] = (loss(rgb, sh + e, mu) - loss(rgb, sh - e, mu)) / (2 * eps)
        worst["rgb"] = max(worst["rgb"], np.max(np.abs(drgb - num_rgb)))
        worst["sh"] = max(worst["sh"], np.max(np.abs(dsh - num_sh)))
        worst["mu"] = max(worst["mu"], np.max(np.abs(dmu - num_mu)))
    ok = all(v < 1e-6 for v in worst.values())
    print(f"degree {degree}: d rgb {worst['rgb']:.1e}  d sh {worst['sh']:.1e}  d mu {worst['mu']:.1e}  {'ok' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ok = check_orthonormal(rng)
    for degree in (1, 2, 3):
        ok &= check_gradients(rng, degree)
    raise SystemExit(0 if ok else 1)
