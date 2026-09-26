"""Finite-difference check for docs/backward.md section 7 at SH degree 0.

Run: python3 tools/check_color.py
"""
import numpy as np

C0 = 0.28209479177387814


def color_forward(sh):
    return np.maximum(0.0, C0 * sh + 0.5)


def color_backward(sh, dL_dcolor):
    return C0 * dL_dcolor * (C0 * sh + 0.5 > 0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sh = rng.uniform(-4, 4, 3000)  # about a third land under the clamp
    sh = sh[np.abs(C0 * sh + 0.5) > 1e-4]  # a step of eps could cross it
    dL_dcolor = rng.normal(size=sh.size)
    eps = 1e-6
    ref = (color_forward(sh + eps) - color_forward(sh - eps)) / (2 * eps) * dL_dcolor
    err = np.max(np.abs(color_backward(sh, dL_dcolor) - ref))
    clamped = int(np.sum(C0 * sh + 0.5 < 0))
    print(f"dL/dsh ({clamped} of {sh.size} clamped) max abs err {err:.2e}  {'ok' if err < 1e-6 else 'MISMATCH'}")
