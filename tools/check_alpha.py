"""Finite-difference check for docs/backward.md section 2 (alpha -> G, o).

alpha is elementwise, so each gaussian is checked independently.
Run: python3 tools/check_alpha.py
"""
import numpy as np

ALPHA_MAX = 0.99


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def alpha_forward(o, G):
    return np.minimum(ALPHA_MAX, sigmoid(o) * G)


def alpha_backward(o, G, dL_dalpha):
    """Straight through the 0.99 clamp, like Inria. Returns dL_dG, dL_do."""
    s = sigmoid(o)
    dL_dG = s * dL_dalpha
    dL_ds = G * dL_dalpha
    dL_do = s * (1.0 - s) * dL_ds
    return dL_dG, dL_do


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    o = rng.uniform(-4, 7, n)
    G = np.exp(-rng.uniform(0, 3, n))
    G[:100] = 1.0  # pixel at the gaussian's center, where the clamp bites
    dL_dalpha = rng.normal(size=n)
    eps = 1e-6

    raw = sigmoid(o) * G
    clamped = raw > ALPHA_MAX
    near = np.abs(raw - ALPHA_MAX) < 1e-4  # a step of eps could cross the clamp
    free = ~clamped & ~near

    mine_G, mine_o = alpha_backward(o, G, dL_dalpha)
    fd_G = (alpha_forward(o, G + eps) - alpha_forward(o, G - eps)) / (2 * eps) * dL_dalpha
    fd_o = (alpha_forward(o + eps, G) - alpha_forward(o - eps, G)) / (2 * eps) * dL_dalpha

    for name, m, r in (("dL/dG", mine_G, fd_G), ("dL/do", mine_o, fd_o)):
        err = np.max(np.abs(m[free] - r[free]))
        print(f"{name:6s} unclamped ({free.sum()}) max abs err {err:.2e}  {'ok' if err < 1e-6 else 'MISMATCH'}")

    # Clamped entries: finite differences see the true zero, ours is straight through by choice.
    st_ok = np.allclose(mine_o[clamped], (G * sigmoid(o) * (1 - sigmoid(o)) * dL_dalpha)[clamped])
    print(f"clamped ({clamped.sum()}): true derivative is 0, ours passes straight through  {'ok' if st_ok else 'MISMATCH'}")
