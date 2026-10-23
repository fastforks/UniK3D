// Custom CUDA kernels for fast/max.py (compiled once: prebuilt fatbin shipped in fast/prebuilt, loaded by fast/kernels.py).
// bilerp_nhwc as in the fastforks stereoanywhere fork.
#include <cuda_fp16.h>
#include <stdint.h>

// bilinear resize NHWC fp16 (contiguous), any batch: src index = scale * dst (align_corners) or
// max(scale * (dst + 0.5) - 0.5, 0); scales computed by the caller as PyTorch does (fp32). Same arithmetic as
// upsample_bilinear2d_nhwc_out_frame.
extern "C" __global__ void bilerp_nhwc(const __half* __restrict__ src, int N, int Hs, int Ws, __half* __restrict__ dst,
                                       int Hd, int Wd, int C, float sh, float sw, int align) {
    const int CV = C / 8;
    const long long total = (long long)N * Hd * Wd * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wd);
        long long r = pix / Wd;
        int y = (int)(r % Hd), n = (int)(r / Hd);
        float fy = align ? sh * y : fmaxf(sh * (y + 0.5f) - 0.5f, 0.f);
        float fx = align ? sw * x : fmaxf(sw * (x + 0.5f) - 0.5f, 0.f);
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = y0 + (y0 < Hs - 1), x1 = x0 + (x0 < Ws - 1);
        float ly = fy - y0, lx = fx - x0;
        float hy = 1.f - ly, hx = 1.f - lx;
        const __half* s = src + (long long)n * Hs * Ws * C + c0;
        uint4 a = *reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x0) * C);
        uint4 b = *reinterpret_cast<const uint4*>(s + ((long long)y0 * Ws + x1) * C);
        uint4 c = *reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x0) * C);
        uint4 d = *reinterpret_cast<const uint4*>(s + ((long long)y1 * Ws + x1) * C);
        const __half *a8 = reinterpret_cast<const __half*>(&a), *b8 = reinterpret_cast<const __half*>(&b);
        const __half *c8 = reinterpret_cast<const __half*>(&c), *d8 = reinterpret_cast<const __half*>(&d);
        uint4 o;
        __half* o8 = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o8[k] = __float2half(hy * (hx * __half2float(a8[k]) + lx * __half2float(b8[k])) +
                                 ly * (hx * __half2float(c8[k]) + lx * __half2float(d8[k])));
        *reinterpret_cast<uint4*>(dst + pix * C + c0) = o;
    }
}


// reflect padding (pad p on each side, p < H, W) of an NHWC fp16 tensor into a contiguous [N][H+2p][W+2p][C] buffer
// (C % 8 == 0). Same values as F.pad(mode="reflect").
extern "C" __global__ void pad_refl_nhwc(const __half* __restrict__ src, int N, int H, int W, int C, int p,
                                         __half* __restrict__ dst) {
    const int CV = C / 8;
    const int Hp = H + 2 * p, Wp = W + 2 * p;
    const long long total = (long long)N * Hp * Wp * CV;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        long long pix = i / CV;
        int c0 = (int)(i - pix * CV) * 8;
        int x = (int)(pix % Wp);
        long long r = pix / Wp;
        int y = (int)(r % Hp), n = (int)(r / Hp);
        int sy = y - p, sx = x - p;
        sy = sy < 0 ? -sy : (sy >= H ? 2 * (H - 1) - sy : sy);
        sx = sx < 0 ? -sx : (sx >= W ? 2 * (W - 1) - sx : sx);
        *reinterpret_cast<uint4*>(dst + pix * C + c0) =
            *reinterpret_cast<const uint4*>(src + (((long long)n * H + sy) * W + sx) * C + c0);
    }
}


