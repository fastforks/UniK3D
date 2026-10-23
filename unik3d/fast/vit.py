"""fp16 inference path for the vendored DINOv2 ViT encoder (plain PyTorch, any input size that is a multiple of the patch).

- pure fp16 weights and activations; attention through torch SDPA (flash kernel, the fp32 attention matrix is never built)
- residual + LayerScale as one addcmul per branch
- position-embedding interpolation (bicubic, fp32, upstream formula incl. interpolate_offset) cached per token grid
- the final LayerNorm only on the layers the decoder reads
The block math is the upstream DINOv2 block in eval mode (no drop path); only precision and kernel choice change.
"""
import contextlib
import math

import torch
import torch.nn.functional as F


class FastViT:
    """Wraps a DinoVisionTransformer-like module (patch_embed, cls_token, pos_embed, register_tokens, blocks, norm)."""

    def __init__(self, vit, dtype=torch.float16):
        self.vit = vit.to(dtype)
        self.vit.eval()
        self.dtype = dtype
        self.ps = vit.patch_size if isinstance(vit.patch_size, int) else vit.patch_size[0]
        self.nreg = getattr(vit, "num_register_tokens", 0) or 0
        self._pos = {}
        blocks = []
        for b in vit.blocks:
            blocks.extend(list(b) if isinstance(b, torch.nn.ModuleList) or b.__class__.__name__ == "BlockChunk" else [b])
        self.blocks = [b for b in blocks if not isinstance(b, torch.nn.Identity)]
        self.pos_fp32 = vit.pos_embed.detach().float()

    # hooks for the cuda-kernels branch
    def attention(self, q, k, v):
        """q, k, v [B, N, H, d] fp16 -> [B, N, H, d]"""
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    def gemm_context(self):
        return contextlib.nullcontext()

    def pos_embed(self, h0, w0):
        key = (h0, w0)
        if key not in self._pos:
            vit = self.vit
            pe = self.pos_fp32
            N = pe.shape[1] - 1
            M = int(math.sqrt(N))
            if N == h0 * w0 and h0 == w0:
                out = pe
            else:
                off = getattr(vit, "interpolate_offset", 0.1)
                kw = dict(scale_factor=((h0 + off) / M, (w0 + off) / M)) if off else dict(size=(h0, w0))
                p = F.interpolate(pe[:, 1:].reshape(1, M, M, -1).permute(0, 3, 1, 2), mode="bicubic",
                                  antialias=getattr(vit, "interpolate_antialias", False), **kw)
                assert p.shape[-2:] == (h0, w0)
                out = torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)
            self._pos[key] = out.to(self.dtype)
        return self._pos[key]

    def tokens(self, x):
        vit = self.vit
        B, _, h, w = x.shape
        t = vit.patch_embed(x.to(self.dtype))
        t = torch.cat([vit.cls_token.to(self.dtype).expand(B, -1, -1), t], 1)
        t = t + self.pos_embed(h // self.ps, w // self.ps)
        if self.nreg:
            t = torch.cat([t[:, :1], vit.register_tokens.to(self.dtype).expand(B, -1, -1), t[:, 1:]], 1)
        return t

    def block(self, b, x):
        a = b.attn
        B, N, C = x.shape
        H = a.num_heads
        qkv = F.linear(b.norm1(x), a.qkv.weight, a.qkv.bias).view(B, N, 3, H, C // H)
        q, k, v = qkv.unbind(2)
        o = self.attention(q, k, v).reshape(B, N, C)
        o = F.linear(o, a.proj.weight, a.proj.bias)
        g1 = b.ls1.gamma if hasattr(b.ls1, "gamma") else None
        x = torch.addcmul(x, o, g1) if g1 is not None else x + o
        m = b.mlp
        hdn = F.gelu(F.linear(b.norm2(x), m.fc1.weight, m.fc1.bias))
        o = F.linear(hdn, m.fc2.weight, m.fc2.bias)
        g2 = b.ls2.gamma if hasattr(b.ls2, "gamma") else None
        return x.addcmul_(o, g2) if g2 is not None else x.add_(o)

    def intermediate(self, x, layers):
        """-> list of (patch tokens [B, N, C] fp16, cls token [B, C]) after the given block indices (final norm applied)."""
        out = []
        with self.gemm_context():
            t = self.tokens(x)
            for i, b in enumerate(self.blocks):
                t = self.block(b, t)
                if i in layers:
                    o = self.vit.norm(t)
                    out.append((o[:, 1 + self.nreg:], o[:, 0]))
        return out
