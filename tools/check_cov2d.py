"""Finite-difference check for docs/backward.md section 4 (Sigma2d -> Sigma, t).

One gaussian, L = sum(V * Sigma2d) with V a random symmetric upstream gradient.
Sigma is perturbed through its packed 6-entry form, so the finite differences
come out as the diagonal as is and the off-diagonals doubled (section 0).
Run: python3 tools/check_cov2d.py
"""
import numpy as np

FX, FY, WIDTH, HEIGHT = 500.0, 480.0, 640, 480
LIMX = 1.3 * (WIDTH / 2) / FX
LIMY = 1.3 * (HEIGHT / 2) / FY
IU = np.triu_indices(3)


def sym3(p):
    S = np.zeros((3, 3))
    S[IU] = p
    return S + np.triu(S, 1).T


def packed3(M):
    return (M + M.T - np.diag(np.diag(M)))[IU]


def clamped_xy(t):
    tx, ty, tz = t
    return np.clip(tx / tz, -LIMX, LIMX) * tz, np.clip(ty / tz, -LIMY, LIMY) * tz


def jacobian(t):
    txp, typ = clamped_xy(t)
    tz = t[2]
    return np.array([[FX / tz, 0.0, -FX * txp / tz**2],
                     [0.0, FY / tz, -FY * typ / tz**2]])


def cov2d_forward(Sigma, t, W):
    M = jacobian(t) @ W
    return M @ Sigma @ M.T + 0.3 * np.eye(2)


def cov2d_backward(Sigma, t, W, V):
    """Returns dL_dSigma (3,3) full symmetric, dL_dt (3,) covariance share."""
    tx, ty, tz = t
    txp, typ = clamped_xy(t)
    J = jacobian(t)
    M = J @ W
    dL_dSigma = M.T @ V @ M
    D = 2 * V @ J @ (W @ Sigma @ W.T)
    kx = float(abs(tx / tz) <= LIMX)
    ky = float(abs(ty / tz) <= LIMY)
    dL_dt = np.array([
        -FX / tz**2 * D[0, 2] * kx,
        -FY / tz**2 * D[1, 2] * ky,
        -FX / tz**2 * D[0, 0] - FY / tz**2 * D[1, 1]
        + (1 + kx) * FX * txp / tz**3 * D[0, 2]
        + (1 + ky) * FY * typ / tz**3 * D[1, 2],
    ])
    return dL_dSigma, dL_dt


def fd(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for k in range(len(x)):
        e = np.zeros_like(x)
        e[k] = eps
        g[k] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A = rng.normal(size=(3, 3)) * 0.3
    Sigma = A @ A.T
    B = rng.normal(size=(2, 2))
    V = B + B.T

    cases = {
        "unclamped": np.array([0.3, -0.2, 4.0]),
        "clamped x": np.array([5.0, 0.3, 4.0]),
        "clamped y": np.array([0.2, -4.0, 4.0]),
        "clamped xy": np.array([-5.0, 4.0, 4.0]),
    }
    for name, t in cases.items():
        mine_S, mine_t = cov2d_backward(Sigma, t, W, V)
        ref_S = fd(lambda p: np.sum(V * cov2d_forward(sym3(p), t, W)), Sigma[IU])
        ref_t = fd(lambda x: np.sum(V * cov2d_forward(Sigma, x, W)), t)
        for label, m, r in (("dL/dSigma", packed3(mine_S), ref_S), ("dL/dt", mine_t, ref_t)):
            rel = np.max(np.abs(m - r)) / np.max(np.abs(r))
            print(f"{name:11s} {label:10s} rel err {rel:.2e}  {'ok' if rel < 1e-6 else 'MISMATCH'}")
