"""Fast depth-only inference for UniK3D and UniDepth V2 (same DINOv2 encoder + camera-prompted decoder), plain PyTorch.

    fast = FastDepth(model)              # model: UniK3D / UniDepthV2, .cuda().eval(); converts parts of it to fp16 in place
    depth = fast.infer(rgb, camera)["depth"]   # same arguments as model.infer; [B, 1, H, W] fp32

Any input size, nothing compiled or tuned per size. Only the depth output is computed (no confidence, rays, points or
intrinsics outputs). Changes against `model.infer`:
- encoder: pure fp16 with torch SDPA attention, the final LayerNorm only on the 4 layers the decoder reads (vit.py)
- decoder attention part (input adapters, ray embedding, camera-prompt cross-attention): upstream modules and precision
  (fp16 autocast, fp32 weights)
- decoder conv part (residual upsampling stages, depth head): pure fp16 channels-last; the stride == kernel
  ConvTranspose2d as one GEMM added into the latents; identity resizes and UniDepth's discarded per-level depth
  projections dropped; the confidence branch skipped
- the camera head only runs when no camera is given (upstream computes it and then uses the given camera's rays)
- only the z channel of rays * radius is resized to the input resolution (the resize is linear, so depth is unchanged)
"""
import contextlib
import sys

import torch
import torch.nn.functional as F

from .vit import FastViT


class FastDepth:
    def __init__(self, model):
        self.m = model
        self.mod = sys.modules[type(model).__module__]
        dec = model.pixel_decoder
        self.unik3d = hasattr(dec, "radial_module")
        self.head = dec.radial_module if self.unik3d else dec.depth_layer
        self.vit = self.make_vit(model.pixel_encoder)
        self.layers = [j - 1 for _, j in model.slices_encoder_range]
        assert model.stacking_fn.__name__ == "last_stack"
        h = self.head
        conv_mods = [h.ups, h.process_features, h.depth_mlp, h.to_depth_lr, h.to_depth_hr]
        for m in conv_mods:
            m.half()
            for c in m.modules():
                if isinstance(c, torch.nn.Conv2d):
                    c.weight.data = c.weight.data.contiguous(memory_format=torch.channels_last)
        # stride == kernel ConvTranspose2d as a GEMM: W [Cin, Cout, k, k] -> [(i j Cout), Cin]
        self._pf = []
        for pf in h.process_features:
            k = pf.kernel_size[0]
            assert pf.kernel_size == pf.stride and pf.padding == (0, 0) and pf.output_padding == (0, 0)
            cin, cout = pf.weight.shape[:2]
            wm = pf.weight.data.permute(2, 3, 1, 0).reshape(k * k * cout, cin).contiguous()
            b = pf.bias.data.repeat(k * k) if pf.bias is not None else None
            self._pf.append((k, cout, wm, b))

    # ------------------------------------------------------------------ hooks (the cuda-kernels branch overrides these)
    def make_vit(self, enc):
        return FastViT(enc)

    def gemm_context(self):
        return contextlib.nullcontext()

    def conv(self, x, c):
        p = c.padding[0]
        if c.padding_mode == "reflect" and p > 0:
            x = self.pad_reflect(x, p)
            p = 0
        return self.conv2d(x, c, p)

    def conv2d(self, x, c, p):
        return F.conv2d(x, c.weight, c.bias, c.stride, p, c.dilation, c.groups).contiguous(memory_format=torch.channels_last)

    def pad_reflect(self, x, p):
        return F.pad(x, (p, p, p, p), mode="reflect").contiguous(memory_format=torch.channels_last)

    def interp(self, x, size, align_corners):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners).contiguous(
            memory_format=torch.channels_last)

    # ------------------------------------------------------------------ decoder conv part (fp16 channels-last NCHW)
    def rcu(self, r, x):
        """ResidualConvUnit: x + gamma * conv2(act(conv1(act(x)))) (no norms), in place on x."""
        slope = r.activation.negative_slope
        o = self.conv(F.leaky_relu(x, slope), r.conv1)
        o = self.conv(F.leaky_relu_(o, slope), r.conv2)
        g = r.gamma
        if torch.is_tensor(g):
            return x.addcmul_(o, g.view(1, -1, 1, 1).to(o.dtype))
        return x.add_(o, alpha=float(g))

    def up(self, u, x):
        for r in u.convs:
            x = self.rcu(r, x)
        c1x1, ups = u.up[0], u.up[1]
        y = self.conv(x, c1x1)
        assert ups.mode == "bilinear" and not ups.align_corners and ups.scale_factor in (2, 2.0)
        return self.interp(y, (y.shape[2] * 2, y.shape[3] * 2), False)

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def infer(self, rgb, camera=None, normalize=True):
        """rgb: [3, H, W] or [B, 3, H, W] uint8 (or 0..255 float) on the GPU; camera: as model.infer (Camera or K tensor)."""
        m, mod = self.m, self.mod
        with torch.autocast("cuda", dtype=torch.float16):
            ratio_bounds = m.shape_constraints["ratio_bounds"]
            pixels_bounds = [m.shape_constraints["pixels_min"], m.shape_constraints["pixels_max"]]
            if getattr(m, "resolution_level", None) is not None:
                rng = pixels_bounds[1] - pixels_bounds[0]
                pixels_bounds = (m.resolution_level * rng / 10 + pixels_bounds[0],
                                 (m.resolution_level + 1) * rng / 10 + pixels_bounds[0])
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            if camera is not None:
                if torch.is_tensor(camera):
                    camera = sys.modules[mod.BatchCamera.__module__].Pinhole(K=camera)
                camera = mod.BatchCamera.from_camera(camera).to(rgb.device)
            B, _, H, W = rgb.shape
            paddings, (pH, pW) = mod.get_paddings((H, W), ratio_bounds)
            pl, pr, pt, pb = paddings
            resize_factor, (nH, nW) = mod.get_resize_factor((pH, pW), pixels_bounds)
            if normalize:
                rgb = mod.TF.normalize(rgb.float() / 255.0, mean=mod.IMAGENET_DATASET_MEAN, std=mod.IMAGENET_DATASET_STD)
            rgb = F.pad(rgb, (pl, pr, pt, pb), value=0.0)
            rgb = F.interpolate(rgb, size=(nH, nW), mode="bilinear", align_corners=False)
            rays = None
            if camera is not None:
                camera = camera.crop(left=-pl, top=-pt, right=-pr, bottom=-pb).resize(resize_factor)
                rays = camera.get_rays(shapes=(B, nH, nW))

        outs = self.vit.intermediate(rgb, self.layers)  # fp16 encoder
        h, w = nH // self.vit.ps, nW // self.vit.ps
        dec, hd = self.m.pixel_decoder, self.head
        with torch.autocast("cuda", dtype=torch.float16), self.gemm_context():
            feats = dec.input_adapter([o[0].view(B, h, w, -1) for o in outs])
            if rays is None:  # predicted camera
                toks = dec.camera_token_adapter([o[1][:, None] for o in outs])
                if self.unik3d:
                    dec.angular_module.set_shapes((nH, nW))
                    _, rays_flat = dec.run_camera(torch.cat(toks, 1), original_shapes=(nH, nW), rays_gt=None)
                else:
                    dec.camera_layer.set_shapes((nH, nW))
                    _, rays_flat = dec.run_camera(torch.cat(toks, 1), features=feats[0][..., None], pos_embed=None,
                                                  original_shapes=(nH, nW), rays_gt=None)
                rays = rays_flat.view(B, nH, nW, 3).permute(0, 3, 1, 2)
            else:
                rays_flat = rays.flatten(2).transpose(1, 2)
            hd.set_shapes((h, w))
            hd.set_original_shapes((nH, nW))
            remb = hd.embed_rays(rays_flat)
            cond = [p(f.reshape(B, h * w, -1), remb) for p, f in zip(hd.prompt_camera, feats)]
            lat = hd.to_latents(cond[0])
        # conv part: fp16 channels-last ([B, h, w, C] memory viewed as NCHW)
        with self.gemm_context():
            x = lat.view(B, h, w, -1).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
            for i, u in enumerate(hd.ups):
                k, cout, wm, b = self._pf[i]
                y = F.linear(cond[i + 1].reshape(B * h * w, -1).to(wm.dtype), wm, b)  # [(b h w), (i j c)]
                assert (x.shape[2], x.shape[3]) == (h * k, w * k)
                xv = x.permute(0, 2, 3, 1).view(B, h, k, w, k, cout)
                xv.add_(y.view(B, h, w, k, k, cout).permute(0, 1, 3, 2, 4, 5))
                x = self.up(u, x)
            # depth projection: last level's LayerNorm + Linear (NHWC); its identity resize dropped
            ln, lin = hd.depth_mlp[-1][0], hd.depth_mlp[-1][1]
            t = x.permute(0, 2, 3, 1)
            t = F.linear(F.layer_norm(t, ln.normalized_shape, ln.weight, ln.bias, ln.eps), lin.weight, lin.bias)
            x = t.permute(0, 3, 1, 2)
            x = self.conv(x, hd.to_depth_lr)
            x = self.interp(x, (nH, nW), True)
            c0, act, c1 = hd.to_depth_hr[0], hd.to_depth_hr[1], hd.to_depth_hr[2]
            x = F.leaky_relu_(self.conv(x, c0), act.negative_slope)
            logd = F.linear(x.permute(0, 2, 3, 1), c1.weight.view(c1.weight.shape[0], -1), c1.bias)  # [B, nH, nW, 1]
        radius = torch.exp((logd.clamp(-8.0, 8.0) + 2.0).float()).permute(0, 3, 1, 2)
        z = rays[:, 2:3].float() * radius
        z = F.interpolate(z, size=(pH, pW), mode=self.m.interpolation_mode, align_corners=False)
        return {"depth": z[..., pt:pH - pb, pl:pW - pr]}
