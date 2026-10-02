"""Runs the Slang kernels on a GPU through SlangPy and compares them against
the numpy reference in check_e2e.py, on the same scene.

Kernels: raster_forward, loss_l1, raster_backward, project_backward.
project_gaussians isn't finished yet, so its output (2D center in pixels,
conic, depth) comes from the numpy reference.

Setup once:  python3 -m venv ~/.venvs/gsplat && ~/.venvs/gsplat/bin/pip install slangpy numpy
Run:         ~/.venvs/gsplat/bin/python tools/test_shaders.py [--adapter NAME]

The raster backward needs float atomic adds. The laptop's Intel GPU doesn't
have them, so by default this picks a discrete GPU if there is one, else
llvmpipe (Mesa's CPU Vulkan driver).
"""
import argparse
import pathlib
import sys

import numpy as np
import slangpy as spy

TOOLS = pathlib.Path(__file__).resolve().parent
SHADERS = TOOLS.parent / "lib" / "shaders" / "src"
sys.path.insert(0, str(TOOLS))
import check_e2e as ref  # noqa: E402

RW = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
STRIDE = 9  # RASTER_GRAD_STRIDE in splat_math.h


def pick_device(name):
    adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
    if name:
        chosen = next(a for a in adapters if name.lower() in a.name.lower())
    else:
        chosen = next((a for a in adapters if any(v in a.name for v in ("NVIDIA", "AMD", "Radeon"))),
                      next(a for a in adapters if "llvmpipe" in a.name))
    print(f"device: {chosen.name}")
    return spy.Device(type=spy.DeviceType.vulkan, adapter_luid=chosen.luid,
                      compiler_options={"include_paths": [str(SHADERS)],
                                        "matrix_layout": spy.SlangMatrixLayout.column_major})


class Kernels:
    def __init__(self, device):
        self.device = device
        self.k = {}
        for name in ("raster_forward", "loss_l1", "raster_backward", "project_backward"):
            program = device.load_program(str(SHADERS / f"{name}.slang"), ["computeMain"])
            self.k[name] = device.create_compute_kernel(program)

    def buffer(self, words, count):
        """Structured buffer from a (count, n) array of 32-bit words."""
        words = np.ascontiguousarray(words).view(np.uint32).reshape(count, -1)
        return self.device.create_buffer(element_count=count, struct_size=words.shape[1] * 4,
                                         usage=RW, data=words)

    def run(self, name, threads, **buffers):
        self.k[name].dispatch(thread_count=[threads, 1, 1], vars=buffers)


def f32(*a):
    return np.array(a, dtype=np.float32)


# ------------------------------------------------------------ packing (std430, no float3 anywhere)

def pack_gaussians(p):
    """gaussian_t, 32 floats: covariance_inv (16), mean (4), quaternion xyzw (4), scale (4), color rgba (4)."""
    n = len(p["o"])
    g = np.zeros((n, 32), np.float32)
    g[:, 16:19] = p["mu"]
    g[:, 20:23] = p["q"][:, 1:]  # check_e2e stores (w, x, y, z)
    g[:, 23] = p["q"][:, 0]
    g[:, 24:27] = np.exp(p["ls"])
    g[:, 28:31] = np.maximum(0, ref.C0 * p["sh"] + 0.5)
    g[:, 31] = 1 / (1 + np.exp(-p["o"]))
    return g


def pack_gaussians_2d(gs):
    """gaussian2d_t, 8 words: source_index, depth, covariance_inv (4), mean (2)."""
    w = np.zeros((len(gs), 8), np.float32)
    iv = w.view(np.int32)
    for i, g in enumerate(gs):
        iv[i, 0] = i
        w[i, 1] = g["t"][2]
        w[i, 2:6] = g["Q"].T.ravel() if g["visible"] else np.eye(2).ravel()
        w[i, 6:8] = g["mu2d"] if g["visible"] else 0
    return w


def camera_words(Wc, tvec):
    """camera_t, 24 floats: world_to_camera (column-major, like glm), intrinsics, limits."""
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Wc, tvec
    return np.concatenate([M.T.ravel(), [ref.FX, ref.FY, ref.CX, ref.CY], [ref.LIMX, ref.LIMY, ref.NEAR, 0]]).astype(np.float32)


def raster_settings(bg, count):
    s = np.zeros(8, np.float32)
    s[:3] = bg
    s.view(np.int32)[4:7] = [ref.WIDTH, ref.HEIGHT, count]
    s[7] = ref.NEAR
    return s[None]


# ------------------------------------------------------------ the test

def compare(label, mine, expected, tol):
    mine, expected = np.asarray(mine, np.float64), np.asarray(expected, np.float64)
    rel = np.max(np.abs(mine - expected)) / max(np.max(np.abs(expected)), 1e-30)
    ok = rel < tol
    print(f"  {label:26s} rel err {rel:.1e}  {'ok' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="substring of the GPU name, e.g. llvmpipe or 4070")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    K = Kernels(pick_device(args.adapter))
    rng = np.random.default_rng(args.seed)
    p, cam = ref.make_scene(rng)
    Wc, tvec, bg = cam
    n, P = len(p["o"]), ref.WIDTH * ref.HEIGHT
    target = rng.integers(0, 256, (P, 4), dtype=np.uint8)
    target[:, 3] = 255

    # numpy reference, run on exactly what the GPU sees (float32 inputs, 8-bit target)
    gs, order = ref.project_forward(p, Wc, tvec)
    C_ref, T_ref, stop_ref = ref.raster_forward(gs, order, bg)
    _, dL_dC_ref = ref.l1(C_ref, target[:, :3] / 255.0)
    acc_ref = ref.raster_backward(gs, order, bg, T_ref, stop_ref, dL_dC_ref)
    grads_ref = ref.project_backward(p, gs, acc_ref, Wc)

    # buffers
    gauss = K.buffer(pack_gaussians(p), n)
    g2d = K.buffer(pack_gaussians_2d(gs), n)
    sorted_idx = np.argsort([g["t"][2] for g in gs]).astype(np.int32)  # culled ones sort first, the depth test skips them
    idx = K.buffer(sorted_idx[:, None], n)
    image = K.buffer(np.zeros((P, 4), np.float32), P)
    pstate = K.buffer(np.zeros((P, 2), np.float32), P)
    rset = K.buffer(raster_settings(bg, n), 1)

    print("raster_forward")
    K.run("raster_forward", P, gaussians_2d=g2d, gaussians=gauss, sorted_index=idx,
          image_out=image, pixel_state=pstate, settings_buffer=rset)
    img = image.to_numpy().view(np.float32).reshape(P, 4)
    ps = pstate.to_numpy().view(np.float32).reshape(P, 2)
    ok = compare("image", img[:, :3], C_ref, 1e-4)
    ok &= compare("T_final", ps[:, 0], T_ref, 1e-4)

    print("loss_l1")
    packed = target.view(np.uint32).reshape(P, 1)
    tgt = K.buffer(packed, P)
    dLdC = K.buffer(np.zeros((P, 4), np.float32), P)
    lpp = K.buffer(np.zeros((P, 1), np.float32), P)
    K.run("loss_l1", P, image=image, target=tgt, dL_dC=dLdC, loss_per_pixel=lpp,
          settings_buffer=K.buffer(np.array([[ref.WIDTH, ref.HEIGHT]], np.int32), 1))
    dl = dLdC.to_numpy().view(np.float32).reshape(P, 4)[:, :3]
    ok &= compare("dL/dC", dl, dL_dC_ref, 1e-5)
    ok &= compare("loss", lpp.to_numpy().view(np.float32).sum(), ref.l1(C_ref, target[:, :3] / 255.0)[0], 1e-4)

    print("raster_backward")
    rgrads = K.buffer(np.zeros((n, STRIDE), np.float32), n)
    K.run("raster_backward", P, gaussians_2d=g2d, gaussians=gauss, sorted_index=idx,
          pixel_state=pstate, dL_dC=dLdC, raster_grads=rgrads, settings_buffer=rset)
    rg = rgrads.to_numpy().view(np.float32).reshape(n, STRIDE)
    ok &= compare("dL/dcolor", rg[:, 0:3], acc_ref["color"], 1e-4)
    ok &= compare("dL/dopacity", rg[:, 3], acc_ref["s"], 1e-4)
    ok &= compare("dL/dmean2d", rg[:, 4:6], acc_ref["mu2d"], 1e-4)
    conic_ref = np.stack([acc_ref["conic"][:, 0, 0], acc_ref["conic"][:, 0, 1], acc_ref["conic"][:, 1, 1]], 1)
    ok &= compare("dL/dconic", rg[:, 6:9], conic_ref, 1e-4)

    print("project_backward")
    pset = np.concatenate([camera_words(Wc, tvec), np.zeros(4, np.float32)])
    pset.view(np.int32)[24] = n
    grad_out = K.buffer(np.zeros((n, 32), np.float32), n)
    K.run("project_backward", n, gaussians=gauss, raster_grads=rgrads, gradients=grad_out,
          settings_buffer=K.buffer(pset[None], 1))
    go = grad_out.to_numpy().view(np.float32).reshape(n, 32)
    ok &= compare("dL/dmean", go[:, 16:19], grads_ref["mu"], 1e-3)
    q_ref = np.concatenate([grads_ref["q"][:, 1:], grads_ref["q"][:, :1]], 1)  # to (x, y, z, w)
    ok &= compare("dL/dquaternion", go[:, 20:24], q_ref, 1e-3)
    ok &= compare("dL/dscale", go[:, 24:27], grads_ref["ls"] / np.exp(p["ls"]), 1e-3)
    ok &= compare("dL/dcolor (rgb)", go[:, 28:31], acc_ref["color"], 1e-4)
    ok &= compare("dL/dopacity", go[:, 31], acc_ref["s"], 1e-4)
    ok &= bool(np.all(go[7] == 0))
    print(f"  {'culled gaussian is zero':26s} {'ok' if np.all(go[7] == 0) else 'MISMATCH'}")

    print("\nshaders:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
