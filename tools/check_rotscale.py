"""Finite-difference check for docs/backward.md section 6 (Sigma -> q, ls).

One gaussian, L = sum(U * Sigma) with U a random symmetric upstream gradient.
q is deliberately not unit length, so the normalization step gets tested.
Run: python3 tools/check_rotscale.py
"""
import numpy as np

from check_cov2d import fd


def rotation(qh):
    w, x, y, z = qh
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def sigma_forward(q, ls):
    R = rotation(q / np.linalg.norm(q))
    return R @ np.diag(np.exp(2 * ls)) @ R.T


def rotscale_backward(q, ls, U):
    """Returns dL_dq (4,), dL_dls (3,)."""
    qn = np.linalg.norm(q)
    qh = q / qn
    w, x, y, z = qh
    R = rotation(qh)
    s2 = np.exp(2 * ls)

    P = R.T @ U @ R
    dL_dls = 2 * s2 * np.diag(P)

    Gr = 2 * U @ R @ np.diag(s2)
    dL_dqh = np.array([
        2 * (z * (Gr[1, 0] - Gr[0, 1]) + y * (Gr[0, 2] - Gr[2, 0]) + x * (Gr[2, 1] - Gr[1, 2])),
        2 * (y * (Gr[0, 1] + Gr[1, 0]) + z * (Gr[0, 2] + Gr[2, 0]) + w * (Gr[2, 1] - Gr[1, 2])) - 4 * x * (Gr[1, 1] + Gr[2, 2]),
        2 * (x * (Gr[0, 1] + Gr[1, 0]) + w * (Gr[0, 2] - Gr[2, 0]) + z * (Gr[1, 2] + Gr[2, 1])) - 4 * y * (Gr[0, 0] + Gr[2, 2]),
        2 * (w * (Gr[1, 0] - Gr[0, 1]) + x * (Gr[0, 2] + Gr[2, 0]) + y * (Gr[1, 2] + Gr[2, 1])) - 4 * z * (Gr[0, 0] + Gr[1, 1]),
    ])
    dL_dq = (dL_dqh - qh * (qh @ dL_dqh)) / qn
    return dL_dq, dL_dls


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(3):
        q = rng.normal(size=4) * rng.uniform(0.5, 3)
        ls = rng.uniform(-4, 0, 3)  # scales from ~0.02 to 1
        B = rng.normal(size=(3, 3))
        U = B + B.T

        mine_q, mine_ls = rotscale_backward(q, ls, U)
        ref_q = fd(lambda v: np.sum(U * sigma_forward(v, ls)), q)
        ref_ls = fd(lambda v: np.sum(U * sigma_forward(q, v)), ls)
        for label, m, r in (("dL/dq", mine_q, ref_q), ("dL/dls", mine_ls, ref_ls)):
            rel = np.max(np.abs(m - r)) / np.max(np.abs(r))
            print(f"trial {trial} |q|={np.linalg.norm(q):.2f} {label:7s} rel err {rel:.2e}  {'ok' if rel < 1e-6 else 'MISMATCH'}")
        print(f"trial {trial} dL/dq . q = {mine_q @ q:.1e}  (should be ~0)")
