"""End-to-end check for docs/backward.md: sections 1 to 7 chained together.

A tiny numpy renderer with the same four stages the kernels will have:
project forward (per gaussian), raster forward (per pixel), raster backward
(per pixel, accumulating per gaussian), project backward (per gaussian).

Part 1 renders a scene built to hit every branch (early stop, 1/255 skips,
color clamps, a gaussian under the FOV clamp, one culled at the near plane),
takes the L1 loss against a random target, and compares every parameter's
gradient to finite differences of the whole render.

Part 2 renders a target from known gaussians, starts from perturbed ones,
and runs Adam on our gradients to check they actually train.

Run: python3 tools/check_e2e.py
"""
import numpy as np

WIDTH, HEIGHT = 24, 18
FX = FY = 20.0
CX, CY = WIDTH / 2, HEIGHT / 2
LIMX = 1.3 * (WIDTH / 2) / FX
LIMY = 1.3 * (HEIGHT / 2) / FY
NEAR = 0.2
C0 = 0.28209479177387814
ALPHA_MAX, ALPHA_MIN, T_MIN = 0.99, 1 / 255, 1e-4

# pixel centers, COLMAP convention, row-major
PIX = np.stack(np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5), -1).reshape(-1, 2)


def rotation(qh):
    w, x, y, z = qh
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------- forward

def project_forward(p, Wc, tvec):
    """Per gaussian. Returns per-gaussian state and the front-to-back order."""
    gs = []
    for i in range(len(p["o"])):
        t = Wc @ p["mu"][i] + tvec
        tx, ty, tz = t
        g = {"t": t, "visible": tz > NEAR}
        if g["visible"]:
            qn = np.linalg.norm(p["q"][i])
            qh = p["q"][i] / qn
            R = rotation(qh)
            s = np.exp(p["ls"][i])
            Sigma = R @ np.diag(s**2) @ R.T
            txp = np.clip(tx / tz, -LIMX, LIMX) * tz
            typ = np.clip(ty / tz, -LIMY, LIMY) * tz
            J = np.array([[FX / tz, 0.0, -FX * txp / tz**2],
                          [0.0, FY / tz, -FY * typ / tz**2]])
            M = J @ Wc
            raw_color = C0 * p["sh"][i] + 0.5
            g.update(qn=qn, qh=qh, R=R, s=s, Sigma=Sigma, txp=txp, typ=typ, J=J, M=M,
                     Q=np.linalg.inv(M @ Sigma @ M.T + 0.3 * np.eye(2)),
                     mu2d=np.array([FX * tx / tz + CX, FY * ty / tz + CY]),
                     raw_color=raw_color, color=np.maximum(0.0, raw_color),
                     sig=1.0 / (1.0 + np.exp(-p["o"][i])))
        gs.append(g)
    order = sorted((i for i, g in enumerate(gs) if g["visible"]), key=lambda i: gs[i]["t"][2])
    return gs, order


def at_pixels(g):
    d = PIX - g["mu2d"]
    G = np.exp(-0.5 * np.einsum("pi,ij,pj->p", d, g["Q"], d))
    return d, G, np.minimum(ALPHA_MAX, g["sig"] * G)


def raster_forward(gs, order, bg, stats=None):
    """Per pixel. Returns image (P,3), T_final (P,), and where each pixel stopped."""
    P = len(PIX)
    T, C = np.ones(P), np.zeros((P, 3))
    stop, done = np.full(P, len(order)), np.zeros(P, bool)
    for k, i in enumerate(order):
        _, G, alpha = at_pixels(gs[i])
        use = ~done & (alpha >= ALPHA_MIN)
        test_T = T * (1 - alpha)
        stopping = use & (test_T < T_MIN)
        if stats is not None:
            live = ~done
            stats["alpha_margin"] = min(stats.get("alpha_margin", 1), np.min(np.abs(alpha[live] - ALPHA_MIN)))
            stats["T_margin"] = min(stats.get("T_margin", 1), np.min(np.abs(test_T[use] / T_MIN - 1), initial=1))
            stats["max_raw_alpha"] = max(stats.get("max_raw_alpha", 0), np.max(gs[i]["sig"] * G))
            stats["skipped"] = stats.get("skipped", 0) + int(np.sum(live & (alpha < ALPHA_MIN)))
            stats.setdefault("pixels_hit", {})[i] = int(np.sum(use & ~stopping))
        stop[stopping] = k
        done |= stopping
        use &= ~stopping
        C[use] += gs[i]["color"] * (alpha[use] * T[use])[:, None]
        T[use] = test_T[use]
    if stats is not None:
        stats["early_stopped"] = int(done.sum())
    return C + T[:, None] * bg, T, stop


# ---------------------------------------------------------------- backward

def raster_backward(gs, order, bg, T_final, stop, dL_dC):
    """Sections 1 to 3, per pixel, walking back to front. Accumulates per gaussian."""
    n = len(gs)
    acc = {"color": np.zeros((n, 3)), "s": np.zeros(n), "mu2d": np.zeros((n, 2)), "conic": np.zeros((n, 2, 2))}
    T = T_final.copy()
    S = np.tile(bg, (len(PIX), 1))
    for k in reversed(range(len(order))):
        i, g = order[k], gs[order[k]]
        d, G, alpha = at_pixels(g)
        a = (k < stop) & (alpha >= ALPHA_MIN)  # the forward's exact test
        d, G, alpha, gC = d[a], G[a], alpha[a], dL_dC[a]
        # section 1
        T[a] = T[a] / (1 - alpha)
        Ta = T[a]
        acc["color"][i] += np.sum((alpha * Ta)[:, None] * gC, 0)
        dL_dalpha = Ta * np.sum((g["color"] - S[a]) * gC, 1)
        S[a] = alpha[:, None] * g["color"] + (1 - alpha)[:, None] * S[a]
        # section 2
        dL_dG = g["sig"] * dL_dalpha
        acc["s"][i] += np.sum(G * dL_dalpha)
        # section 3
        w = G * dL_dG
        acc["mu2d"][i] += w @ (d @ g["Q"])
        acc["conic"][i] += -0.5 * np.einsum("p,pi,pj->ij", w, d, d)
    return acc


def project_backward(p, gs, acc, Wc):
    """Sections 2 (per-gaussian part) and 3 (inverse step) through 7."""
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    for i, g in enumerate(gs):
        if not g["visible"]:
            continue
        # section 2
        grads["o"][i] = g["sig"] * (1 - g["sig"]) * acc["s"][i]
        # section 3
        V = -g["Q"] @ acc["conic"][i] @ g["Q"]
        # section 4
        tx, ty, tz = g["t"]
        U = g["M"].T @ V @ g["M"]
        D = 2 * V @ g["J"] @ (Wc @ g["Sigma"] @ Wc.T)
        kx, ky = float(abs(tx / tz) <= LIMX), float(abs(ty / tz) <= LIMY)
        dL_dt = np.array([
            -FX / tz**2 * D[0, 2] * kx,
            -FY / tz**2 * D[1, 2] * ky,
            -FX / tz**2 * D[0, 0] - FY / tz**2 * D[1, 1]
            + (1 + kx) * FX * g["txp"] / tz**3 * D[0, 2]
            + (1 + ky) * FY * g["typ"] / tz**3 * D[1, 2],
        ])
        # section 5
        gm = acc["mu2d"][i]
        dL_dt += np.array([FX / tz * gm[0], FY / tz * gm[1], -(FX * tx * gm[0] + FY * ty * gm[1]) / tz**2])
        grads["mu"][i] = Wc.T @ dL_dt
        # section 6
        R, s2 = g["R"], g["s"] ** 2
        grads["ls"][i] = 2 * s2 * np.diag(R.T @ U @ R)
        Gr = 2 * U @ R @ np.diag(s2)
        w, x, y, z = g["qh"]
        dqh = np.array([
            2 * (z * (Gr[1, 0] - Gr[0, 1]) + y * (Gr[0, 2] - Gr[2, 0]) + x * (Gr[2, 1] - Gr[1, 2])),
            2 * (y * (Gr[0, 1] + Gr[1, 0]) + z * (Gr[0, 2] + Gr[2, 0]) + w * (Gr[2, 1] - Gr[1, 2])) - 4 * x * (Gr[1, 1] + Gr[2, 2]),
            2 * (x * (Gr[0, 1] + Gr[1, 0]) + w * (Gr[0, 2] - Gr[2, 0]) + z * (Gr[1, 2] + Gr[2, 1])) - 4 * y * (Gr[0, 0] + Gr[2, 2]),
            2 * (w * (Gr[1, 0] - Gr[0, 1]) + x * (Gr[0, 2] + Gr[2, 0]) + y * (Gr[1, 2] + Gr[2, 1])) - 4 * z * (Gr[0, 0] + Gr[1, 1]),
        ])
        grads["q"][i] = (dqh - g["qh"] * (g["qh"] @ dqh)) / g["qn"]
        # section 7, SH degree 0
        grads["sh"][i] = C0 * acc["color"][i] * (g["raw_color"] > 0)
    return grads


# ---------------------------------------------------------------- loss and glue

def l1(C, target):
    diff = C - target
    return np.mean(np.abs(diff)), np.sign(diff) / diff.size


def render(p, cam, stats=None):
    Wc, tvec, bg = cam
    gs, order = project_forward(p, Wc, tvec)
    C, T_final, stop = raster_forward(gs, order, bg, stats)
    return C, (gs, order, T_final, stop)


def loss_and_grads(p, cam, target):
    Wc, _, bg = cam
    C, (gs, order, T_final, stop) = render(p, cam)
    L, dL_dC = l1(C, target)
    acc = raster_backward(gs, order, bg, T_final, stop, dL_dC)
    return L, project_backward(p, gs, acc, Wc)


def make_scene(rng):
    Wc, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Wc) < 0:
        Wc[:, 0] *= -1
    tvec = rng.normal(size=3)
    bg = rng.uniform(0, 1, 3)
    # view-space centers as (x/z, y/z, z)
    views = [
        (0.02, 0.01, 2.0), (-0.03, 0.02, 2.6), (0.01, -0.03, 3.2), (-0.02, -0.01, 3.9),  # stacked: early stop
        (0.35, 0.2, 4.5), (-0.4, -0.25, 5.1),
        (0.9, 0.1, 3.5),  # center outside 1.3x FOV in x, footprint still reaches the image
        (0.0, 0.0, 0.1),  # inside the near plane: culled
    ]
    t = np.array([[a * z, b * z, z] for a, b, z in views])
    n = len(views)
    ls = rng.uniform(-1.3, -0.6, (n, 3))
    ls[6] = np.log(0.9)
    o = rng.uniform(-1, 2, n)
    o[:4] = 3.0
    p = {
        "mu": (t - tvec) @ Wc,  # Wc^T (t - tvec), row-wise
        "q": rng.normal(size=(n, 4)) * rng.uniform(0.5, 2, (n, 1)),
        "ls": ls,
        "o": o,
        "sh": rng.uniform(-3, 3, (n, 3)),
    }
    return p, (Wc, tvec, bg)


def fd_grads(p, f, eps=1e-6):
    out = {}
    for k, v in p.items():
        g = np.zeros_like(v)
        for idx in np.ndindex(v.shape):
            hi = {kk: vv.copy() for kk, vv in p.items()}
            lo = {kk: vv.copy() for kk, vv in p.items()}
            hi[k][idx] += eps
            lo[k][idx] -= eps
            g[idx] = (f(hi) - f(lo)) / (2 * eps)
        out[k] = g
    return out


# ---------------------------------------------------------------- checks

def gradient_check(rng):
    print("part 1: gradients vs finite differences of the whole render")
    p, cam = make_scene(rng)
    target = rng.uniform(0, 1, (len(PIX), 3))
    stats = {}
    C, _ = render(p, cam, stats)
    raw = C0 * p["sh"] + 0.5
    print(f"  scene: {len(p['o'])} gaussians, {WIDTH}x{HEIGHT} px")
    print(f"  branches: {stats['early_stopped']} px early-stopped, {stats['skipped']} gaussian-pixel pairs skipped under 1/255,")
    print(f"            {int(np.sum(raw < 0))} color channels clamped at 0, max raw alpha {stats['max_raw_alpha']:.3f} (0.99 clamp not hit),")
    print(f"            FOV-clamped gaussian blends into {stats['pixels_hit'][6]} px, near-plane gaussian culled")
    print(f"  margins:  alpha to 1/255 {stats['alpha_margin']:.1e}, T to 1e-4 {stats['T_margin']:.1e} (relative),"
          f" |C - target| {np.min(np.abs(C - target)):.1e}")

    _, mine = loss_and_grads(p, cam, target)
    ref = fd_grads(p, lambda q: l1(render(q, cam)[0], target)[0])
    ok = True
    for k in ("mu", "q", "ls", "o", "sh"):
        rel = np.max(np.abs(mine[k] - ref[k])) / np.max(np.abs(ref[k]))
        ok &= rel < 1e-6
        print(f"  dL/d{k:3s} rel err {rel:.2e}  {'ok' if rel < 1e-6 else 'MISMATCH'}")
    culled_zero = all(np.all(mine[k][7] == 0) and np.allclose(ref[k][7], 0) for k in mine)
    fov_live = np.max(np.abs(mine["mu"][6])) > 0
    print(f"  culled gaussian has zero gradient: {'ok' if culled_zero else 'MISMATCH'}")
    print(f"  FOV-clamped gaussian has nonzero gradient: {'ok' if fov_live else 'MISMATCH'}")
    return ok and culled_zero and fov_live


def fit_check(rng, steps=400):
    print("part 2: fit perturbed gaussians to a target with Adam on our gradients")
    true_p, cam = make_scene(rng)
    for k in true_p:  # drop the culled one, it can't be fit
        true_p[k] = true_p[k][:7]
    target, _ = render(true_p, cam)
    p = {
        "mu": true_p["mu"] + rng.normal(size=true_p["mu"].shape) * 0.08,
        "q": true_p["q"] + rng.normal(size=true_p["q"].shape) * 0.2,
        "ls": true_p["ls"] + rng.normal(size=true_p["ls"].shape) * 0.2,
        "o": true_p["o"] + rng.normal(size=true_p["o"].shape) * 0.7,
        "sh": true_p["sh"] + rng.normal(size=true_p["sh"].shape) * 1.0,
    }
    psnr = lambda img: -10 * np.log10(np.mean((img - target) ** 2))
    psnr_start = psnr(render(p, cam)[0])
    lr = {"mu": 0.01, "q": 0.02, "ls": 0.02, "o": 0.05, "sh": 0.05}
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    history = []
    for step in range(1, steps + 1):
        L, g = loss_and_grads(p, cam, target)
        history.append(L)
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            p[k] -= lr[k] * (m[k] / (1 - 0.9**step)) / (np.sqrt(v[k] / (1 - 0.999**step)) + 1e-15)
    C, _ = render(p, cam)
    history.append(l1(C, target)[0])
    for s in (0, 50, 100, 200, steps):
        print(f"  step {s:3d}  L1 {history[s]:.5f}")
    print(f"  PSNR {psnr_start:.1f} dB -> {psnr(C):.1f} dB")
    ok = history[-1] < 0.1 * history[0]
    print(f"  loss fell {history[0] / history[-1]:.0f}x: {'ok' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = gradient_check(rng)
    print()
    b = fit_check(rng)
    print()
    print("end to end:", "ok" if a and b else "FAILED")
