"""Fast depth-only inference. FastDepth: plain PyTorch (depth.py). FastDepthMax: plus fp16 accumulation and custom
kernels (max.py; cuda-kernels branch)."""
from .depth import FastDepth
from .max import FastDepthMax

__all__ = ["FastDepth", "FastDepthMax"]
