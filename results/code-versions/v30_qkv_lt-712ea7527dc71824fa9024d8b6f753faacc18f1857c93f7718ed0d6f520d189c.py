# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured candidate: v12 plus native TinyGEMM for BF16 batches 2..8.

v12 passed official check 3951752bc54040f1 at reported speedup 1.135028x.
THIS FILE HAS NOT BEEN GPU-COMPILED, NUMERICALLY CHECKED, OR BENCHMARKED.

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
