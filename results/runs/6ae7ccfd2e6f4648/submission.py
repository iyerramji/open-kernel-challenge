# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
# Original experimental implementation; no H200 benchmark claims.
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
    def _gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
              SX: tl.constexpr, SK: tl.constexpr, WN: tl.constexpr,
              WK: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # One CTA per token and output-row group; contiguous K loads.
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        token = tl.program_id(1)
        kk = tl.arange(0, BK)
        acc = tl.zeros((BN, BK), tl.float32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + kk
            x = tl.load(X + token * SX + k * SK, k < K, other=0).to(tl.float32)
            w = tl.load(W + rows[:, None] * WN + k[None, :] * WK,
                        (rows[:, None] < N) & (k[None, :] < K), other=0).to(tl.float32)
            acc += w * x[None, :]
        result = tl.sum(acc, axis=1)
        tl.store(Y + token * N + rows, result, rows < N)

def kernel_fn(data, state):
    if not data["x"].is_cuda:
        return _torch(data, state)
    if triton is None:
        raise RuntimeError("Triton is required for GPU candidates")
    if data["x"].shape[0] > 8:
        return _torch(data, state)

    x, cfg = data["x"], data["config"]
    m, n, k = x.shape[0], cfg["n"], x.shape[1]
    w = state[cfg["layer"]]
    y = torch.empty((m, n), device=x.device,
                    dtype=torch.float32 if cfg["layer"] == "router" else torch.bfloat16)
    _gemv[(triton.cdiv(n, 4), m)](x, w, y, n, k,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), 4, 256,
        num_warps=4, enable_fp_fusion=False)
    return y
