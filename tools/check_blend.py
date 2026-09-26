"""Finite-difference check for docs/backward.md section 1 (blending).

One pixel, N gaussians sorted front to back, 3 color channels.
Run: python3 tools/check_blend.py
"""
import numpy as np


def blend_forward(colors, alphas, bg):
    """colors (N,3), alphas (N,), bg (3,) -> C (3,), T_final (scalar)"""
    C = np.zeros(3)
    T = 1.0
    for c, a in zip(colors, alphas):
        C += c * a * T
        T *= 1.0 - a
    C += T * bg
    return C, T


def blend_backward(colors, alphas, bg, T_final, dL_dC):
    """Same view as the kernel: only T_final survives the forward pass.

    Walk back to front (i = N-1 .. 0), recovering each T_i as you go
    instead of recomputing the forward product.
    Returns dL_dcolors (N,3), dL_dalphas (N,).
    """
    N = len(alphas)
    dL_dcolors = np.zeros((N, 3))
    dL_dalphas = np.zeros(N)
    T = T_final
    S = bg.copy()  # color behind i, starts as the background
    for i in range(N - 1, -1, -1):
        a, c = alphas[i], colors[i]
        T = T / (1.0 - a)  # T_i
        dL_dcolors[i] = a * T * dL_dC
        dL_dalphas[i] = T * (c - S) @ dL_dC
        S = a * c + (1.0 - a) * S  # S_{i-1}
    return dL_dcolors, dL_dalphas


def numeric_grads(colors, alphas, bg, dL_dC, eps=1e-6):
    """Central differences of L = dL_dC . C, so the result is the same
    vector-Jacobian product the kernel computes."""
    L = lambda c, a: dL_dC @ blend_forward(c, a, bg)[0]
    gc = np.zeros_like(colors)
    ga = np.zeros_like(alphas)
    for idx in np.ndindex(colors.shape):
        d = np.zeros_like(colors)
        d[idx] = eps
        gc[idx] = (L(colors + d, alphas) - L(colors - d, alphas)) / (2 * eps)
    for i in range(len(alphas)):
        d = np.zeros_like(alphas)
        d[i] = eps
        ga[i] = (L(colors, alphas + d) - L(colors, alphas - d)) / (2 * eps)
    return gc, ga


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 6
    colors = rng.uniform(0, 1, (N, 3))
    alphas = rng.uniform(0.05, 0.95, N)
    bg = rng.uniform(0, 1, 3)
    dL_dC = rng.normal(size=3)  # stands in for the upstream gradient from the loss

    C, T_final = blend_forward(colors, alphas, bg)
    mine = blend_backward(colors, alphas, bg, T_final, dL_dC)
    ref = numeric_grads(colors, alphas, bg, dL_dC)
    for name, m, r in zip(("dL/dcolor", "dL/dalpha"), mine, ref):
        err = np.max(np.abs(m - r))
        print(f"{name:10s} max abs err {err:.2e}  {'ok' if err < 1e-6 else 'MISMATCH'}")
