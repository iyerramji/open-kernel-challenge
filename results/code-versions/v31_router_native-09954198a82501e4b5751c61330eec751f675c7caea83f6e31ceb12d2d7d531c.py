# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
"""Modal experiment v31_router_native; parent v30. Pending validation; not officially checked."""

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

# v30: v24 plus fixed metadata-selected whole-GEMM Lt choices for QKV.

_v24_prepare=prepare
_v24_kernel=kernel_fn

def prepare(weights):
 state=_v24_prepare(weights)
 if not weights['router'].is_cuda:return state
 import ctypes as C,glob
 paths=glob.glob('/usr/local/cuda*/**/libcublasLt.so*',recursive=True)+glob.glob('/usr/local/lib/python*/dist-packages/nvidia/**/libcublasLt.so*',recursive=True)+glob.glob('/usr/local/lib/python*/site-packages/nvidia/**/libcublasLt.so*',recursive=True)
 lib=C.CDLL(paths[0] if paths else 'libcublasLt.so.13')
 P=C.c_void_p;I=C.c_int;Z=C.c_size_t;U=C.c_uint64
 sigs={'cublasLtCreate':[C.POINTER(P)],'cublasLtMatmulDescCreate':[C.POINTER(P),I,I],'cublasLtMatrixLayoutCreate':[C.POINTER(P),I,U,U,C.c_int64],'cublasLtMatmulPreferenceCreate':[C.POINTER(P)],'cublasLtMatmulPreferenceSetAttribute':[P,I,P,Z],'cublasLtMatmulAlgoGetHeuristic':[P,P,P,P,P,P,P,I,P,C.POINTER(I)],'cublasLtMatmul':[P,P,P,P,P,P,P,P,P,P,P,P,P,P,Z,P]}
 for n,a in sigs.items():getattr(lib,n).argtypes=a;getattr(lib,n).restype=I
 def call(n,*a):
  rc=getattr(lib,n)(*a)
  if rc:raise RuntimeError((n,rc))

 lib.cublasLtMatmulDescSetAttribute.argtypes=[P,I,P,Z]
 h=P();call('cublasLtCreate',C.byref(h))
 workspace=torch.empty(32*1024*1024,device=weights['qkv_gate'].device,dtype=torch.uint8)
 plans={}
 for m,layout,j in [(128,'packed',2),(256,'original',4)]:
  n,k=2304,8192;wt=state['qkv_gate_t'] if layout=='packed' else state['qkv_gate']
  desc=P();call('cublasLtMatmulDescCreate',C.byref(desc),68,0)
  if layout=='original':
   trans=I(1);call('cublasLtMatmulDescSetAttribute',desc,3,C.byref(trans),C.sizeof(trans))
  layouts=[]
  for rows,cols,ld in ([(n,k,n)] if layout=='packed' else [(k,n,k)])+[(k,m,k),(n,m,n)]:
   q=P();call('cublasLtMatrixLayoutCreate',C.byref(q),14,rows,cols,ld);layouts.append(q)
  a,b,c=layouts;pref=P();call('cublasLtMatmulPreferenceCreate',C.byref(pref));cap=Z(workspace.numel());call('cublasLtMatmulPreferenceSetAttribute',pref,1,C.byref(cap),C.sizeof(cap))
  buf=C.create_string_buffer(96*8);count=I();call('cublasLtMatmulAlgoGetHeuristic',h,desc,a,b,c,c,pref,8,buf,C.byref(count));assert count.value>j,'Expected Lt heuristic unavailable'
  plans[m]=(wt,desc,a,b,c,buf,j,pref)
 alpha=C.c_float(1);beta=C.c_float(0)
 def invoke(x):
  wt,desc,a,b,c,buf,j,pref=plans[x.shape[0]];y=torch.empty((x.shape[0],2304),device=x.device,dtype=torch.bfloat16)
  algo=C.cast(C.byref(buf,j*96),P)
  call('cublasLtMatmul',h,desc,C.byref(alpha),P(wt.data_ptr()),a,P(x.data_ptr()),b,C.byref(beta),P(y.data_ptr()),c,P(y.data_ptr()),c,algo,P(workspace.data_ptr()),workspace.numel(),P(torch.cuda.current_stream(x.device).cuda_stream))
  return y
 state['lt_invoke']=invoke
 for m in plans:invoke(torch.zeros((m,8192),device=weights['qkv_gate'].device,dtype=torch.bfloat16))
 return state

def kernel_fn(data,state):
 x,cfg=data['x'],data['config']
 if x.is_cuda and cfg['layer']=='qkv_gate' and cfg['n']==2304 and x.shape[0] in (128,256) and x.shape[1]==8192 and x.is_contiguous():return state['lt_invoke'](x)
 return _v24_kernel(data,state)

_router_parent_prepare=prepare
_router_parent_kernel=kernel_fn
_ROUTER_CUDA='\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAGuard.h>\n#include <c10/cuda/CUDAException.h>\n#include <cuda_bf16.h>\n// Original experiment: one output row per CTA, original FP32 weights.\n__global__ void __launch_bounds__(256) router_one(float* y,const __nv_bfloat16* x,const float* w,int m,int n){\n __shared__ float sw[8192];__shared__ float warp_sum[8];\n int tid=threadIdx.x,lane=tid%32,warp=tid/32,row=blockIdx.x;\n for(int k=tid*4;k<8192;k+=1024)*reinterpret_cast<float4*>(sw+k)=__ldcs(reinterpret_cast<const float4*>(w+row*8192+k));\n __syncthreads();float a0=0,a1=0,a2=0,a3=0;\n int token=m==1?0:warp;\n if(token<m){\n  int first=(m==1?tid:lane)*4,step=(m==1?256:32)*4;\n  for(int k=first;k<8192;k+=step){\n   float4 v=*reinterpret_cast<const float4*>(sw+k);\n   auto xp=x+token*8192+k;\n   a0=fmaf(v.x,__bfloat162float(xp[0]),a0);a1=fmaf(v.y,__bfloat162float(xp[1]),a1);\n   a2=fmaf(v.z,__bfloat162float(xp[2]),a2);a3=fmaf(v.w,__bfloat162float(xp[3]),a3);\n  }\n }\n float acc=(a0+a1)+(a2+a3);\n #pragma unroll\n for(int off=16;off>0;off>>=1)acc+=__shfl_down_sync(0xffffffff,acc,off);\n if(m==1){\n  if(lane==0)warp_sum[warp]=acc;__syncthreads();\n  if(tid==0){float total=0;for(int j=0;j<8;j++)total+=warp_sum[j];y[row]=total;}\n }else if(token<m && lane==0)y[token*n+row]=acc;\n}\ntorch::Tensor run(torch::Tensor x,torch::Tensor w){\n TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.device()==w.device()&&x.is_contiguous()&&w.is_contiguous());\n TORCH_CHECK(x.scalar_type()==torch::kBFloat16&&w.scalar_type()==torch::kFloat32);\n TORCH_CHECK(x.dim()==2&&w.dim()==2&&x.size(1)==8192&&w.size(1)==8192&&x.size(0)>0&&x.size(0)<=8);\n c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty({x.size(0),w.size(0)},x.options().dtype(torch::kFloat32));\n router_one<<<w.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<float>(),reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),w.data_ptr<float>(),x.size(0),w.size(0));C10_CUDA_KERNEL_LAUNCH_CHECK();return y;\n}\n'
def prepare(weights):
 state=_router_parent_prepare(weights)
 if weights['router'].is_cuda:
  from torch.utils.cpp_extension import load_inline
  state['router_native']=load_inline(name='okc_v31_router_native',cpp_sources='torch::Tensor run(torch::Tensor x,torch::Tensor w);',cuda_sources=_ROUTER_CUDA,functions=['run'],extra_cuda_cflags=['-O3'],verbose=False)
  for m in (1,8):state['router_native'].run(torch.zeros((m,8192),device=weights['router'].device,dtype=torch.bfloat16),state['router'])
 return state
def kernel_fn(data,state):
 x,c=data['x'],data['config']
 if x.is_cuda and c['layer']=='router' and x.is_contiguous() and 1<=x.shape[0]<=8 and x.shape[1]==8192:
  return state['router_native'].run(x,state['router'][:c['n']])
 return _router_parent_kernel(data,state)
