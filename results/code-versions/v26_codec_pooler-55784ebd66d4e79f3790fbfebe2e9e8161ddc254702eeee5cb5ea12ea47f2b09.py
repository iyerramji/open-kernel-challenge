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
    if router.is_cuda:
        u = state["pooler"].view(torch.int16).to(torch.int32) & 65535
        blocks = u.reshape(-1, 128)
        high = (blocks >> 8) & 127
        base = high.min(1).values
        escape = (high.max(1).values - base) >= 8
        state["codec_meta"] = (base | (escape.to(torch.int32) << 15)).to(torch.int16)
        fields = ((blocks & 255) | (((high - base[:, None]) & 7) << 8) | ((blocks >> 15) << 11)).reshape(-1, 2)
        pairs = fields[:, 0] | (fields[:, 1] << 12)
        state["codec_bytes"] = torch.stack([pairs & 255, (pairs >> 8) & 255, (pairs >> 16) & 255], dim=1).to(torch.uint8).flatten()
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


if triton is not None:
    @triton.jit
    def _codec(P, H, W, X, T, COMPRESSED: tl.constexpr):
        rows=tl.program_id(0)*4+tl.arange(0,4)
        split=tl.program_id(1)
        kk=tl.arange(0,256)
        acc=tl.zeros((4,256),tl.float32)
        for step in range(8):
            k=split*2048+step*256+kk
            idx=rows[:,None]*8192+k[None,:]
            if COMPRESSED:
                meta=tl.load(H+idx//128).to(tl.uint16)
                escape=(meta & 32768)!=0
                pair=idx//2
                packed=tl.load(P+pair*3).to(tl.uint32)|(tl.load(P+pair*3+1).to(tl.uint32)<<8)|(tl.load(P+pair*3+2).to(tl.uint32)<<16)
                field=(packed>>((idx%2)*12))&4095
                bits=(field&255)|((((field>>8)&7)+(meta&127))<<8)|((field>>11)<<15)
                decoded=bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)
                raw=tl.load(W+idx,escape,other=0)
                w=tl.where(escape,raw,decoded).to(tl.float32)
            else:
                w=tl.load(W+idx).to(tl.float32)
            x=tl.load(X+k).to(tl.float32)
            acc+=w*x[None,:]
        tl.store(T+split*4096+rows,tl.sum(acc,1))


def kernel_fn(data, state):
    if not data["x"].is_cuda:
        return _torch(data, state)
    if triton is None:
        raise RuntimeError("Triton is required for GPU candidates")
    x, cfg = data["x"], data["config"]
    if cfg["layer"] == "pooler" and x.shape[0] == 1:
        t=torch.empty((4,4096),device=x.device,dtype=torch.float32)
        y=torch.empty((1,4096),device=x.device,dtype=torch.bfloat16)
        _codec[(1024,4)](state["codec_bytes"],state["codec_meta"],state["pooler"],x,t,COMPRESSED=True,num_warps=4,enable_fp_fusion=False)
        _reduce[(16,)](t,y,4096,4,256)
        return y
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
