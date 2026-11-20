"""Runs sh_forward, sh_backward and sh_apply on a GPU through SlangPy and
compares them against the numpy reference in check_sh.py.

Also checks the parts the training loop relies on: sh_forward leaves every
field but rgb alone, sh_backward writes the plain rgb back bit for bit,
and both backward outputs add into what's already there.

Setup once:  python3 -m venv ~/.venvs/gsplat && ~/.venvs/gsplat/bin/pip install slangpy numpy
Run:         ~/.venvs/gsplat/bin/python tools/test_sh.py [--adapter NAME]

Uses Metal on macOS, Vulkan elsewhere (a discrete GPU if there is one, else
llvmpipe). No float atomics needed, one thread owns each gaussian.
"""
import argparse
import pathlib
import sys

import numpy as np
import slangpy as spy

TOOLS = pathlib.Path(__file__).resolve().parent
SHADERS = TOOLS.parent / "lib" / "shaders" / "src"
sys.path.insert(0, str(TOOLS))
import check_sh as ref  # noqa: E402

RW = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
STRIDE = 9   # RASTER_GRAD_STRIDE in splat_math.h
SH = 45      # SH_STRIDE in sh_math.h


def pick_device(name, backend):
    options = {"include_paths": [str(SHADERS)], "matrix_layout": spy.SlangMatrixLayout.column_major}
    if backend == "cpu":
        print("device: cpu")
        return spy.Device(type=spy.DeviceType.cpu, compiler_options=options)
    kind = spy.DeviceType.metal if sys.platform == "darwin" else spy.DeviceType.vulkan
    adapters = spy.Device.enumerate_adapters(kind)
    if name:
        chosen = next(a for a in adapters if name.lower() in a.name.lower())
    else:
        chosen = next((a for a in adapters if any(v in a.name for v in ("NVIDIA", "AMD", "Radeon", "Apple"))), None)
        chosen = chosen or next(a for a in adapters if "llvmpipe" in a.name)
    print(f"device: {chosen.name}")
    return spy.Device(type=kind, adapter_luid=chosen.luid, compiler_options=options)


class Kernels:
    def __init__(self, device):
        self.device = device
        self.k = {}
        for name in ("sh_forward", "sh_backward", "sh_apply"):
            program = device.load_program(str(SHADERS / f"{name}.slang"), ["computeMain"])
            self.k[name] = device.create_compute_kernel(program)

    def buffer(self, words):
        """Structured buffer from a (count, n) array of 32-bit words."""
        words = np.ascontiguousarray(words)
        count = words.shape[0]
        words = words.view(np.uint32).reshape(count, -1)
        return self.device.create_buffer(element_count=count, struct_size=words.shape[1] * 4,
                                         usage=RW, data=words)

    def run(self, name, threads, **buffers):
        self.k[name].dispatch(thread_count=[threads, 1, 1], vars=buffers)


def read(buf, cols):
    return buf.to_numpy().view(np.float32).reshape(-1, cols)


def sh_settings(center, count, degree):
    s = np.zeros(8, np.float32)
    s[:3] = center
    s.view(np.int32)[4:6] = [count, degree]
    return s[None]


def compare(label, mine, expected, tol):
    mine, expected = np.asarray(mine, np.float64), np.asarray(expected, np.float64)
    rel = np.max(np.abs(mine - expected)) / max(np.max(np.abs(expected)), 1e-30)
    ok = rel < tol
    print(f"  {label:34s} rel err {rel:.1e}  {'ok' if ok else 'MISMATCH'}")
    return ok


def exact(label, mine, expected):
    ok = np.array_equal(np.asarray(mine).view(np.uint32), np.asarray(expected).view(np.uint32))
    print(f"  {label:34s} {'bit for bit' if ok else 'MISMATCH'}")
    return ok


def test_degree(K, rng, degree, n=3000):
    print(f"degree {degree}, {n} gaussians")
    # gaussian_t, 32 floats: covariance_inv (16), mean (4), quaternion (4), scale (4), color rgba (4)
    g = rng.normal(size=(n, 32)).astype(np.float32)
    g[:, 19] = 0
    g[:, 28:31] = rng.uniform(0, 1, (n, 3))
    g[:, 31] = rng.uniform(0.1, 1, n)
    center = (rng.normal(size=3) * 4).astype(np.float32)
    g[0, 16:19] = center  # one gaussian sitting on the camera
    sh = (rng.normal(size=(n, SH)) * 0.3).astype(np.float32)
    rgrads = rng.normal(size=(n, STRIDE)).astype(np.float32)
    grads0 = rng.normal(size=(n, 32)).astype(np.float32)
    sh_grads0 = rng.normal(size=(n, SH)).astype(np.float32)

    gbuf, shbuf = K.buffer(g), K.buffer(sh)
    base = K.buffer(np.zeros((n, 4), np.float32))
    settings = K.buffer(sh_settings(center, n, degree))

    K.run("sh_forward", n, gaussians=gbuf, sh=shbuf, base_rgb=base, settings_buffer=settings)
    seen = read(gbuf, 32)

    # numpy, in float64 from the same float32 inputs
    mu = g[:, 16:19].astype(np.float64)
    c = center.astype(np.float64)
    want_rgb = g[:, 28:31].astype(np.float64).copy()
    want_dsh = np.zeros((n, 15, 3))
    want_dmu = np.zeros((n, 3))
    for i in range(1, n):
        coeffs = sh[i].reshape(15, 3).astype(np.float64)
        want_rgb[i] = ref.view_color(g[i, 28:31].astype(np.float64), coeffs, mu[i], c, degree)
        _, want_dsh[i], want_dmu[i] = ref.view_color_backward(coeffs, mu[i], c, degree, rgrads[i, :3].astype(np.float64))

    ok = compare("forward: view color", seen[:, 28:31], want_rgb, 1e-5)
    ok &= exact("forward: base_rgb saved", read(base, 4)[:, :3], g[:, 28:31])
    ok &= exact("forward: other fields untouched", np.delete(seen, [28, 29, 30], axis=1), np.delete(g, [28, 29, 30], axis=1))
    ok &= exact("forward: gaussian on the camera", seen[0], g[0])

    gradbuf, shgbuf = K.buffer(grads0), K.buffer(sh_grads0)
    K.run("sh_backward", n, gaussians=gbuf, base_rgb=base, sh=shbuf, raster_grads=K.buffer(rgrads),
          sh_grads=shgbuf, gradients=gradbuf, settings_buffer=settings)
    grads, sh_grads = read(gradbuf, 32), read(shgbuf, SH)

    ok &= exact("backward: gaussians restored", read(gbuf, 32), g)
    ok &= compare("backward: adds dL/dsh", sh_grads - sh_grads0, want_dsh.reshape(n, SH), 1e-5)
    ok &= compare("backward: adds dL/dmean", grads[:, 16:19] - grads0[:, 16:19], want_dmu, 1e-4)
    ok &= exact("backward: other gradients untouched", np.delete(grads, [16, 17, 18], axis=1), np.delete(grads0, [16, 17, 18], axis=1))

    lr = np.float32(0.05)
    K.run("sh_apply", n * SH, sh=shbuf, sh_grads=shgbuf, settings_buffer=K.buffer(apply_settings(lr, n * SH)))
    ok &= compare("apply: sh -= rate * sh_grads", read(shbuf, SH), sh - lr * sh_grads, 1e-6)
    return ok


def apply_settings(lr, count):
    s = np.zeros(2, np.float32)
    s[0] = lr
    s.view(np.int32)[1] = count
    return s[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="part of the GPU's name, e.g. llvmpipe")
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="gpu",
                    help="cpu runs the kernels through Slang's CPU target (needs a C++ compiler, no GPU)")
    args = ap.parse_args()
    K = Kernels(pick_device(args.adapter, args.backend))
    rng = np.random.default_rng(0)
    ok = True
    for degree in (1, 2, 3):
        ok &= test_degree(K, rng, degree)
    print("all ok" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
