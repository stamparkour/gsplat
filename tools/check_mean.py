"""Finite-difference check for docs/backward.md sections 4 + 5 (mu -> t -> mu2d, Sigma2d).

One gaussian, L = g . mu2d + sum(V * Sigma2d), so mu reaches the loss through
both the projected center and the covariance's J.
Run: python3 tools/check_mean.py
"""
import numpy as np

from check_cov2d import FX, FY, LIMX, LIMY, cov2d_backward, cov2d_forward, fd

CX, CY = 320.0, 240.0


def mean2d_forward(t):
    tx, ty, tz = t
    return np.array([FX * tx / tz + CX, FY * ty / tz + CY])


def mean_backward(Sigma, mu, W, tvec, g, V):
    """Returns dL_dmu (3,). g = dL/dmu2d, V = dL/dSigma2d."""
    t = W @ mu + tvec
    tx, ty, tz = t
    _, dL_dt = cov2d_backward(Sigma, t, W, V)  # section 4 share
    dL_dt = dL_dt + np.array([
        FX / tz * g[0],
        FY / tz * g[1],
        -(FX * tx * g[0] + FY * ty * g[1]) / tz**2,
    ])
    return W.T @ dL_dt


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A = rng.normal(size=(3, 3)) * 0.3
    Sigma = A @ A.T
    B = rng.normal(size=(2, 2))
    V = B + B.T
    g = rng.normal(size=2)

    def L(mu, tvec):
        t = W @ mu + tvec
        return g @ mean2d_forward(t) + np.sum(V * cov2d_forward(Sigma, t, W))

    # pick tvec so the view-space center lands where we want it
    mu = rng.normal(size=3)
    for name, t_target in (("unclamped", [0.3, -0.2, 4.0]), ("clamped x", [5.0, 0.3, 4.0])):
        tvec = np.array(t_target) - W @ mu
        t = W @ mu + tvec
        assert (abs(t[0] / t[2]) > LIMX) == (name == "clamped x") and abs(t[1] / t[2]) <= LIMY
        mine = mean_backward(Sigma, mu, W, tvec, g, V)
        ref = fd(lambda m: L(m, tvec), mu)
        rel = np.max(np.abs(mine - ref)) / np.max(np.abs(ref))
        print(f"{name:10s} dL/dmu rel err {rel:.2e}  {'ok' if rel < 1e-6 else 'MISMATCH'}")
