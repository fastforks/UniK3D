"""Loader for the size-independent CUDA kernels in lp_kernels.cu (NHWC bilinear resize, reflect padding).

Uses the shipped fatbin in prebuilt/ (SASS sm_80/86/89/90/120 + compute_80 PTX); without it, compiles the source once with
the nvcc of $CUDA_HOME into ~/.cache/lp-fast-kernels. Loaded through the CUDA driver API, launched on torch's current
stream. Rebuild the shipped fatbin after editing the source: `python -m <package>.fast.kernels`.
"""
import ctypes
import hashlib
import os
import subprocess

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "lp_kernels.cu")
_PREBUILT = os.path.join(_DIR, "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "lp-fast-kernels")

# p: pointer, i: int32, f: float32
SIGNATURES = {
    "bilerp_nhwc": "piii" + "p" + "iii" + "ff" + "i",
    "pad_refl_nhwc": "piiii" + "i" + "p",
}
_CTYPE = {"i": ctypes.c_int32, "f": ctypes.c_float}


def _srctag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:16]


def _nvcc():
    return os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")


def build_prebuilt(archs=(80, 86, 89, 90, 120)):
    os.makedirs(_PREBUILT, exist_ok=True)
    for f in os.listdir(_PREBUILT):
        if f.startswith("lp_kernels_"):
            os.remove(os.path.join(_PREBUILT, f))
    out = os.path.join(_PREBUILT, f"lp_kernels_{_srctag()}.fatbin")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + ["-gencode=arch=compute_80,code=compute_80"]
    subprocess.run([_nvcc(), "-fatbin", "-O3", "-std=c++17", *gencode, "-o", out + ".tmp", _SRC], check=True)
    os.replace(out + ".tmp", out)
    return out


def _check(res, what):
    if res != 0:
        raise RuntimeError(f"CUDA driver error {res} in {what}")


class Kernels:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability()
        image = os.path.join(_PREBUILT, f"lp_kernels_{_srctag()}.fatbin")
        if not os.path.exists(image):
            tag = hashlib.sha1(open(_SRC, "rb").read() + f"sm_{major}{minor}".encode()).hexdigest()[:16]
            image = os.path.join(_CACHE, f"lp_kernels_{tag}.cubin")
            if not os.path.exists(image):
                os.makedirs(_CACHE, exist_ok=True)
                subprocess.run([_nvcc(), "-cubin", "-O3", "-std=c++17", f"-arch=sm_{major}{minor}", "-o", image + ".tmp",
                                _SRC], check=True)
                os.replace(image + ".tmp", image)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        torch.zeros(1, device="cuda")  # primary context current
        self.module = ctypes.c_void_p()
        self._image = open(image, "rb").read()
        _check(self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image), "cuModuleLoadData")
        self.funcs = {}
        for name in SIGNATURES:
            f = ctypes.c_void_p()
            _check(self.lib.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()), f"cuModuleGetFunction {name}")
            self.funcs[name] = f

    @staticmethod
    def linear(n):
        """256 threads per block, grid-stride loops, at most 65535*8 blocks."""
        return (min(max((n + 255) // 256, 1), 65535 * 8), 1, 1), (256, 1, 1)

    def launch(self, name, cfg, *args):
        grid, block = cfg
        sig = SIGNATURES[name]
        assert len(sig) == len(args), f"{name}: expected {len(sig)} args, got {len(args)}"
        holders = [ctypes.c_void_p(a.data_ptr()) if t == "p" else _CTYPE[t](a) for t, a in zip(sig, args)]
        arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        stream = torch.cuda.current_stream().cuda_stream
        _check(self.lib.cuLaunchKernel(self.funcs[name], *grid, *block, 0, stream, arr, None), f"launch {name}")


def interp_bilinear(x, size, align_corners):
    """F.interpolate(x, size, 'bilinear', align_corners) for fp16 channels-last x (C % 8 == 0) in one vectorized kernel
    (same arithmetic as PyTorch: fp32 weights, one rounding); returns channels-last."""
    N, C, Hs, Ws = x.shape
    Hd, Wd = int(size[0]), int(size[1])
    if align_corners:
        sh = (Hs - 1) / (Hd - 1) if Hd > 1 else 0.0
        sw = (Ws - 1) / (Wd - 1) if Wd > 1 else 0.0
    else:
        sh, sw = Hs / Hd, Ws / Wd
    y = torch.empty((N, C, Hd, Wd), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    k = Kernels.get()
    k.launch("bilerp_nhwc", k.linear(N * Hd * Wd * C // 8), x, N, Hs, Ws, y, Hd, Wd, C, float(sh), float(sw),
             1 if align_corners else 0)
    return y


def pad_reflect(x, p):
    """F.pad(x, (p, p, p, p), mode='reflect') for fp16 channels-last x (C % 8 == 0); returns channels-last."""
    N, C, H, W = x.shape
    y = torch.empty((N, C, H + 2 * p, W + 2 * p), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    k = Kernels.get()
    k.launch("pad_refl_nhwc", k.linear(N * (H + 2 * p) * (W + 2 * p) * C // 8), x, N, H, W, C, p, y)
    return y


def nhwc_ok(x):
    return x.is_cuda and x.dtype == torch.float16 and x.shape[1] % 8 == 0 and x.is_contiguous(memory_format=torch.channels_last)


if __name__ == "__main__":
    print(build_prebuilt())
