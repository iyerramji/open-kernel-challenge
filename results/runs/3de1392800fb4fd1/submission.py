# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
# Router-only split-K refinement based on archived H200 v3 profile; this version needs its own check.
import torch

def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state

def _torch(data, state):
    x, cfg = data["x"], data["config"]
    w = state[cfg["layer"]][:cfg["n"]]
    if cfg["layer"] == "router":
        if not x.is_cuda:
            return x.float() @ w.T
        hi = torch.mm(x, state["router_hi_t"][:, :cfg["n"]], out_dtype=torch.float32)
        lo = torch.mm(x, state["router_lo_t"][:, :cfg["n"]], out_dtype=torch.float32)
        return hi + lo
    return torch.nn.functional.linear(x, w)

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
            x = tl.load(X + token * SX + k * SK, mask, other=0).to(tl.float32)
            w = tl.load(W + rows[:, None] * WN + k[None, :] * WK,
                        (rows[:, None] < N) & mask[None, :], other=0).to(tl.float32)
            acc += w * x[None, :]
        result = tl.sum(acc, axis=1)
        tl.store(T + (split * M + token) * N + rows, result, rows < N)

    @triton.jit
    def _reduce(T, Y, SIZE: tl.constexpr, SPLITS: tl.constexpr, B: tl.constexpr):
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
    if x.shape[0] > 8 or cfg["layer"] != "router":
        return _torch(data, state)
    m, n, k = x.shape[0], cfg["n"], x.shape[1]
    w = state[cfg["layer"]]
    splits = 4
    partial = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    y = torch.empty((m, n), device=x.device,
                    dtype=torch.float32 if cfg["layer"] == "router" else torch.bfloat16)
    _partial[(triton.cdiv(n, 4), m, splits)](x, w, partial, m, n, k,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), 4, 256,
        splits, triton.cdiv(k, splits), num_warps=4, enable_fp_fusion=False)
    _reduce[(triton.cdiv(m * n, 256),)](partial, y, m * n, splits, 256)
    return y
