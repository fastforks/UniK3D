"""Fast depth-only inference (fp16 encoder with SDPA attention, fp16 channels-last decoder convs), plain PyTorch.
See depth.py."""
from .depth import FastDepth

__all__ = ["FastDepth"]
