"""Finite-difference check for docs/backward.md section 3 (G -> mu2d, conic -> Sigma2d).

One gaussian, many pixels, L = sum_p dL_dG[p] * G(p).
Symmetric matrices are perturbed through their packed (a, b, c) form, so the
finite differences come out as (full_00, 2 * full_01, full_11).
Run: python3 tools/check_conic.py
"""
import numpy as np


def sym(a, b, c):
    return np.array([[a, b], [b, c]])


def G_from_conic(mu2d, Q, pix):
    d = pix - mu2d  # (P, 2)
    return np.exp(-0.5 * np.einsum("pi,ij,pj->p", d, Q, d))


def G_forward(mu2d, Sigma2d, pix):
    return G_from_conic(mu2d, np.linalg.inv(Sigma2d), pix)


def conic_backward(mu2d, Sigma2d, pix, dL_dG):
    """Raster part summed over pixels, then one inverse step per gaussian.
    Returns dL_dmu2d (2,), dL_dconic (2,2), dL_dSigma2d (2,2), full symmetric matrices."""
    Q = np.linalg.inv(Sigma2d)
    d = pix - mu2d
    w = G_from_conic(mu2d, Q, pix) * dL_dG
    u = d @ Q  # rows are (Q d)^T
    dL_dmu2d = w @ u
    dL_dconic = -0.5 * np.einsum("p,pi,pj->ij", w, d, d)
    dL_dSigma2d = -Q @ dL_dconic @ Q
    return dL_dmu2d, dL_dconic, dL_dSigma2d


def packed(M):
    """Full symmetric gradient -> gradient w.r.t. packed (a, b, c)."""
    return np.array([M[0, 0], 2 * M[0, 1], M[1, 1]])


def fd(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for k in range(len(x)):
        e = np.zeros_like(x)
        e[k] = eps
        g[k] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mu2d = np.array([10.3, 12.7])
    A = rng.normal(size=(2, 2)) * 2
    Sigma2d = A @ A.T + 0.3 * np.eye(2)
    pix = mu2d + rng.normal(size=(64, 2)) * 2.5 + 0.5
    dL_dG = rng.normal(size=64)
    Q = np.linalg.inv(Sigma2d)

    mine_mu, mine_conic, mine_Sigma = conic_backward(mu2d, Sigma2d, pix, dL_dG)
    L_mu = lambda m: dL_dG @ G_forward(m, Sigma2d, pix)
    L_conic = lambda abc: dL_dG @ G_from_conic(mu2d, sym(*abc), pix)
    L_Sigma = lambda abc: dL_dG @ G_forward(mu2d, sym(*abc), pix)
    abc = lambda M: np.array([M[0, 0], M[0, 1], M[1, 1]])

    checks = (
        ("dL/dmu2d", mine_mu, fd(L_mu, mu2d)),
        ("dL/dconic", packed(mine_conic), fd(L_conic, abc(Q))),
        ("dL/dSigma2d", packed(mine_Sigma), fd(L_Sigma, abc(Sigma2d))),
    )
    for name, m, r in checks:
        err = np.max(np.abs(m - r))
        print(f"{name:12s} max abs err {err:.2e}  {'ok' if err < 1e-6 else 'MISMATCH'}")
    print("symmetric    ", "ok" if np.allclose(mine_Sigma, mine_Sigma.T) else "MISMATCH")
