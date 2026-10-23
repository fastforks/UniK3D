"""cuda-kernels branch: FastDepth plus custom kernels and fp16 accumulation.

    fast = FastDepthMax(model)   # same API as FastDepth
    depth = fast.infer(rgb, camera)["depth"]

On top of FastDepth (plain PyTorch, depth.py):
- fp16-accumulate cuBLAS GEMMs in the encoder and decoder (torch.backends.cuda.matmul.allow_fp16_accumulation;
  2x tensor-core rate on GeForce cards)
- encoder attention: head_dim-64 flash attention with fp16 accumulation (fastbb_vendored, prebuilt)
- decoder convolutions: fp16-accumulate cuDNN graphs (fastbb_vendored/cudnn_conv.py; needs
  `pip install nvidia-cudnn-frontend`), NHWC bilinear-resize and reflect-padding kernels (kernels.py, prebuilt)
Each part that cannot load falls back to its FastDepth op. LP_FAST_KERNELS=0 turns all custom kernels off
(fp16 accumulation stays on).
"""
import contextlib
import os
import warnings

import torch

from .depth import FastDepth
from .vit import FastViT


@contextlib.contextmanager
def fp16_accumulate(enabled=True):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


def _try(what, f):
    try:
        return f()
    except Exception as e:  # noqa: BLE001 - any load failure means "use the plain op"
        warnings.warn(f"{what} unavailable, using the plain PyTorch op: {e}")
        return None


class FastViTMax(FastViT):
    def __init__(self, vit, attn, gacc):
        super().__init__(vit)
        self._attn, self._gacc = attn, gacc

    def attention(self, q, k, v):
        if self._attn is not None and q.shape[-1] == 64:
            return self._attn(q, k, v, variant="hh")
        return super().attention(q, k, v)

    def gemm_context(self):
        return fp16_accumulate(self._gacc)


class FastDepthMax(FastDepth):
    def __init__(self, model, kernels=None, gacc=True):
        if kernels is None:
            kernels = os.environ.get("LP_FAST_KERNELS", "1") != "0"
        self.gacc = gacc
        self._attn = self._conv_hacc = self._k = None
        if kernels:
            self._attn = _try("flash attention", self._load_attention)
            self._conv_hacc = _try("cuDNN fp16-accumulate convs", self._load_conv)
            self._k = _try("NHWC kernels", self._load_kernels)
        super().__init__(model)

    @staticmethod
    def _load_attention():
        from .fastbb_vendored import attention as fa
        q = torch.zeros(1, 64, 1, 64, device="cuda", dtype=torch.float16)
        fa.attention(q, q, q)  # loads the prebuilt kernel
        return fa.attention

    @staticmethod
    def _load_conv():
        from .fastbb_vendored import cudnn_conv
        if cudnn_conv.cudnn is None:
            raise ImportError("pip install nvidia-cudnn-frontend")
        return cudnn_conv.conv_half_acc

    @staticmethod
    def _load_kernels():
        from . import kernels
        kernels.Kernels.get()
        return kernels

    def make_vit(self, enc):
        return FastViTMax(enc, self._attn, self.gacc)

    def gemm_context(self):
        return fp16_accumulate(self.gacc)

    def conv2d(self, x, c, p):
        if self._conv_hacc is not None and c.groups == 1 and c.dilation == (1, 1):
            return self._conv_hacc(x, c.weight, c.bias, [p, p], list(c.stride))
        return super().conv2d(x, c, p)

    def pad_reflect(self, x, p):
        if self._k is not None and self._k.nhwc_ok(x):
            return self._k.pad_reflect(x, p)
        return super().pad_reflect(x, p)

    def interp(self, x, size, align_corners):
        if self._k is not None and self._k.nhwc_ok(x):
            return self._k.interp_bilinear(x, size, align_corners)
        return super().interp(x, size, align_corners)
