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
    state["router_cat"] = torch.cat([state["router_hi_t"], state["router_lo_t"]], dim=1)
    for name, w in weights.items():
        if name != "router":
            state[name + "_t"] = w.T.contiguous()
    if router.is_cuda:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
        state["native_gemv"] = hopper_bf16_gemv
        from quack.gemm_interface import gemm_tuned
        from quack.gemm_config import GemmConfig
        state["quack_fixed"] = getattr(gemm_tuned, "fn", gemm_tuned)
        state["quack_config"] = GemmConfig(tile_m=64, tile_n=128, cluster_m=1, cluster_n=1, pingpong=False, swap_ab=True, is_dynamic_persistent=False, device_capacity=9)
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
    if x.is_cuda and x.shape[0] >= 64 and cfg["layer"] in ("qkv_gate", "pooler", "unpooler"):
        out = torch.empty((x.shape[0], cfg["n"]), dtype=torch.bfloat16, device=x.device)
        splits = (4 if x.shape[0] < 192 else 2) if cfg["layer"] == "qkv_gate" else (2 if cfg["layer"] == "pooler" and x.shape[0] < 192 else 1)
        state["quack_fixed"](x, w.T, out, config=state["quack_config"], split_k=splits, split_k_mode=2)
        return out
    if cfg["layer"] == "qkv_gate" and x.shape[0] >= 64:
        return torch.nn.functional.linear(x, w)
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
    if x.shape[0] == 1 and cfg["layer"] != "router":
        return state["native_gemv"](x, state[cfg["layer"]][:cfg["n"]])
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
