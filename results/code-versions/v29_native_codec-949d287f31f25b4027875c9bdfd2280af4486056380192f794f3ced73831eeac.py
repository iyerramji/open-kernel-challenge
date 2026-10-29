# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured candidate: v12 plus native TinyGEMM for BF16 batches 2..8.

v12 passed official check 3951752bc54040f1 at reported speedup 1.135028x.
v29 two-plane lossless native consumer; pending GPU validation/timing.

Requirements: the installed FlashInfer image must export
flashinfer.gemm.tinygemm_bf16 and load/build it without network access.
No dependency is installed or downloaded by this submission.

Before the last official check, locally validate the existing image, numerical
tests and CUDA-graph replay, then compare cold-L2 paired timings with v12.
Remove any losing shape from _TINY_TARGETS before freezing the candidate.
The targets are static metadata, never selected from activation values.
"""

import torch


# (layer, output width N, input width K). Initially covers four scored M=8
# projections. Nearby M=2..7 follows the same mathematical implementation.
# Router, M=1, and M>8 retain the v12 dispatch below.
_TINY_TARGETS = frozenset({
    ("unpooler", 1024, 4096),
})


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_cat"] = torch.cat(
        [state["router_hi_t"], state["router_lo_t"]], dim=1
    )
    for name, w in weights.items():
        if name != "router":
            state[name + "_t"] = w.T.contiguous()
    if router.is_cuda:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
        state["native_gemv"] = hopper_bf16_gemv

        # Deliberately explicit: an unavailable API must fail local preflight
        # instead of silently producing an unchanged v12 candidate.
        from flashinfer.gemm import tinygemm_bf16
        state["tinygemm_bf16"] = tinygemm_bf16

        # Initialize the installed module before timed graph capture. These
        # temporary synthetic inputs and outputs are not retained in state.
        # No timing, autotuning, host synchronization, or input cache is used.
        for layer, n, k in sorted(_TINY_TARGETS):
            w = state[layer][:n]
            warm_x = torch.zeros(
                (8, k), dtype=torch.bfloat16, device=w.device
            )
            warm_y = torch.empty(
                (8, n), dtype=torch.bfloat16, device=w.device
            )
            tinygemm_bf16(
                warm_x, w, warm_y, bias=None, use_pdl=False
            )
    return state


def _torch(data, state):
    x, cfg = data["x"], data["config"]
    w = state[cfg["layer"]][:cfg["n"]]
    if cfg["layer"] == "router":
        if not x.is_cuda:
            return x.float() @ w.T
        both = torch.mm(x, state["router_cat"], out_dtype=torch.float32)
        width = state["router"].shape[0]
        return both[:, :cfg["n"]] + both[:, width:width + cfg["n"]]
    return torch.mm(x, state[cfg["layer"] + "_t"][:, :cfg["n"]])


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _partial(X, W, T, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 SX: tl.constexpr, SK: tl.constexpr, WN: tl.constexpr,
                 WK: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 SPLITS: tl.constexpr, CHUNK: tl.constexpr):
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        token, split = tl.program_id(1), tl.program_id(2)
        kk = tl.arange(0, BK)
        acc = tl.zeros((BN, BK), tl.float32)
        for step in range(tl.cdiv(CHUNK, BK)):
            k = split * CHUNK + step * BK + kk
            mask = (k < K) & (k < (split + 1) * CHUNK)
            x = tl.load(
                X + token * SX + k * SK, mask, other=0
            ).to(tl.float32)
            w = tl.load(
                W + rows[:, None] * WN + k[None, :] * WK,
                (rows[:, None] < N) & mask[None, :], other=0
            ).to(tl.float32)
            acc += w * x[None, :]
        result = tl.sum(acc, axis=1)
        tl.store(T + (split * M + token) * N + rows, result, rows < N)

    @triton.jit
    def _reduce(T, Y, SIZE: tl.constexpr, SPLITS: tl.constexpr,
                B: tl.constexpr):
        idx = tl.program_id(0) * B + tl.arange(0, B)
        acc = tl.full((B,), 0, tl.float32)
        for split in range(SPLITS):
            acc += tl.load(T + split * SIZE + idx, idx < SIZE, other=0)
        tl.store(Y + idx, acc, idx < SIZE)


def kernel_fn(data, state):
    if not data["x"].is_cuda:
        return _torch(data, state)
    if triton is None:
        raise RuntimeError("Triton is required for GPU candidates")
    x, cfg = data["x"], data["config"]
    if x.shape[0] == 1 and cfg["layer"] != "router":
        return state["native_gemv"](
            x, state[cfg["layer"]][:cfg["n"]]
        )

    # New path. All guards inspect shape/layout/dtype metadata only.
    # Original row-major W[N,K] is required, not the packed transpose W_t.
    if (
        1 < x.shape[0] <= 8
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and (cfg["layer"], cfg["n"], x.shape[1]) in _TINY_TARGETS
    ):
        w = state[cfg["layer"]][:cfg["n"]]
        y = torch.empty(
            (x.shape[0], cfg["n"]),
            dtype=torch.bfloat16,
            device=x.device,
        )
        # tinygemm_bf16 writes y and returns None. PDL must remain disabled
        # because the preceding graph operations need not participate in PDL.
        state["tinygemm_bf16"](
            x, w, y, bias=None, use_pdl=False
        )
        return y

    if x.shape[0] > 8 or cfg["layer"] != "router":
        return _torch(data, state)
    m, n, k = x.shape[0], cfg["n"], x.shape[1]
    w = state[cfg["layer"]]
    splits = 4
    partial = torch.empty(
        (splits, m, n), dtype=torch.float32, device=x.device
    )
    y = torch.empty(
        (m, n), device=x.device,
        dtype=torch.float32 if cfg["layer"] == "router"
        else torch.bfloat16
    )
    _partial[(triton.cdiv(n, 4), m, splits)](
        x, w, partial, m, n, k,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        4, 256, splits, triton.cdiv(k, splits),
        num_warps=4, enable_fp_fusion=False
    )
    _reduce[(triton.cdiv(m * n, 256),)](
        partial, y, m * n, splits, 256
    )
    return y

_base_prepare=prepare
_base_kernel=kernel_fn
_CUDA='\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <c10/cuda/CUDAGuard.h>\n// Schedule adapted from Apache-2.0 SGLang hopper_bf16_gemv.cuh.\n__device__ __forceinline__ float dot8(float4 w,float4 x){\n auto a=reinterpret_cast<const __nv_bfloat162*>(&w);\n auto b=reinterpret_cast<const __nv_bfloat162*>(&x);\n float v=0;\n #pragma unroll\n for(int j=0;j<4;j++){float2 p=__bfloat1622float2(a[j]),q=__bfloat1622float2(b[j]);v=fmaf(p.x,q.x,v);v=fmaf(p.y,q.y,v);}return v;\n}\n__global__ void __launch_bounds__(256) raw_kernel(__nv_bfloat16* y,const __nv_bfloat16* x,const __nv_bfloat16* w,const uint8_t* low,const uint8_t* nib,const int16_t* meta){\n __shared__ __nv_bfloat16 sx[8192];\n int tid=threadIdx.x,lane=tid%32,row=blockIdx.x*8+tid/32;\n for(int k=tid*8;k<8192;k+=2048)*reinterpret_cast<float4*>(sx+k)=*reinterpret_cast<const float4*>(x+k);\n __syncthreads();float acc=0;\n for(int k=lane*16;k<8192;k+=512){\n float4 xv[2],wv[2];\n #pragma unroll\n for(int u=0;u<2;u++)xv[u]=*reinterpret_cast<const float4*>(sx+k+u*8);\n #pragma unroll\n for(int u=0;u<2;u++)wv[u]=make_float4(0,0,0,0);\n unsigned short md=meta[row*64+(k>>7)];\n if(md&32768){\n #pragma unroll\n for(int u=0;u<2;u++)wv[u]=__ldcs(reinterpret_cast<const float4*>(w+row*8192+k+u*8));\n }else{\n uint4 lv=__ldcs(reinterpret_cast<const uint4*>(low+row*8192+k));\n uint2 nv=__ldcs(reinterpret_cast<const uint2*>(nib+row*4096+k/2));\n auto l=reinterpret_cast<const uint8_t*>(&lv);auto n=reinterpret_cast<const uint8_t*>(&nv);\n auto dst=reinterpret_cast<unsigned short*>(wv);\n #pragma unroll\n for(int j=0;j<16;j++){unsigned q=(n[j/2]>>((j%2)*4))&15;dst[j]=l[j]|(((md&127)+(q&7))<<8)|((q&8)<<12);}\n }\n #pragma unroll\n for(int u=0;u<2;u++)acc+=dot8(wv[u],xv[u]);\n }\n #pragma unroll\n for(int o=16;o>0;o>>=1)acc+=__shfl_down_sync(0xffffffff,acc,o);\n if(lane==0)y[row]=__float2bfloat16(acc);\n}\ntorch::Tensor run(torch::Tensor x,torch::Tensor w,torch::Tensor low,torch::Tensor nib,torch::Tensor meta){\n TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.is_contiguous() && w.is_contiguous());\n TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && w.scalar_type()==torch::kBFloat16);\n TORCH_CHECK(x.size(0)==1 && x.size(1)==8192 && w.size(0)==4096 && w.size(1)==8192 && x.device()==w.device());\n c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty({1,4096},x.options());\n raw_kernel<<<512,256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),low.data_ptr<uint8_t>(),nib.data_ptr<uint8_t>(),meta.data_ptr<int16_t>());return y;\n}\n'
def prepare(weights):
 state=_base_prepare(weights)
 if weights['pooler'].is_cuda:
  from torch.utils.cpp_extension import load_inline
  state['raw_module']=load_inline(name='okc_v29_native_codec',cpp_sources='torch::Tensor run(torch::Tensor x,torch::Tensor w,torch::Tensor low,torch::Tensor nib,torch::Tensor meta);',cuda_sources=_CUDA,functions=['run'],extra_cuda_cflags=['-O3'],verbose=False)
  u=state['pooler'].view(torch.int16).to(torch.int32)&65535
  blocks=u.reshape(4096,64,128);h=(blocks>>8)&127;base=h.amin(-1);escape=(h.amax(-1)-base)>=8
  q=((h-base.unsqueeze(-1))&7)|((blocks>>15)<<3)
  q=q.reshape(4096,8192)
  state['low']=(u&255).to(torch.uint8).contiguous()
  state['nib']=(q[:,0::2]|(q[:,1::2]<<4)).to(torch.uint8).contiguous()
  state['meta']=(base|(escape.to(torch.int32)<<15)).to(torch.int16).contiguous()
  # Validate actual packed planes against every supplied weight bit.
  nb=state['nib'].to(torch.int32);codes=torch.stack((nb&15,nb>>4),dim=-1).reshape(4096,64,128)
  decoded=(state['low'].to(torch.int32).reshape_as(blocks)|((base.unsqueeze(-1)+(codes&7))<<8)|((codes&8)<<12))
  decoded=torch.where(escape.unsqueeze(-1),blocks,decoded)
  assert torch.equal(decoded,blocks), 'codec bit mismatch'
  state['raw_module'].run(torch.zeros((1,8192),device=weights['pooler'].device,dtype=torch.bfloat16),state['pooler'],state['low'],state['nib'],state['meta'])
 return state
def kernel_fn(data,state):
 x,c=data['x'],data['config']
 if x.is_cuda and x.is_contiguous() and x.shape==(1,8192) and c['layer']=='pooler' and c['n']==4096:
  return state['raw_module'].run(x,state['pooler'],state['low'],state['nib'],state['meta'])
 return _base_kernel(data,state)
