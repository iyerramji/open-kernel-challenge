# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
# v13 = v12 + cuBLAS strided-batched split-K (torch.bmm, FP32 partials, one
# fused Triton reduce) for under-occupied shapes. Unchecked; split table is a
# first guess to be tuned from per-shape check timings.
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


# ---------------------------------------------------------------- split table
# Ranges, not exact M, so nearby shapes (7, 33, 130) take the same path as the
# scored shape next to them. Return 1 to fall through to the v12 path.
def _split(layer, m):
    if layer == "router":
        return 16 if 2 <= m <= 32 else 1          # M1 router stays on v12 Triton
    if layer == "qkv_gate":                        # N=2304 -> ~18 tiles/batch
        return 8 if m <= 32 else 4 if m <= 384 else 1
    if layer == "pooler":                          # N=4096 -> ~32 tiles/batch
        return 4 if m <= 192 else 2 if m <= 384 else 1
    if layer == "unpooler":                        # N=1024 (M<=8) or 8192
        return 8 if m <= 8 else 2 if m <= 192 else 1
    return 1                                       # "o": K=1024, not worth a 2nd kernel


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_cat"] = torch.cat(
        [state["router_hi_t"], state["router_lo_t"]], dim=1
    )                                              # [K, 2*W], contiguous
    for name, w in weights.items():
        if name != "router":
            state[name + "_t"] = w.T.contiguous()  # [K, N], contiguous
    state["bmm_ok"] = False
    if router.is_cuda:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
        state["native_gemv"] = hopper_bf16_gemv
        # Feature probe (untimed, weight-independent): bmm must support FP32
        # output from BF16 inputs; BF16 partials would double-round.
        a = torch.ones(2, 8, 64, device=router.device, dtype=torch.bfloat16)
        b = torch.ones(2, 64, 32, device=router.device, dtype=torch.bfloat16)
        c = torch.bmm(a, b, out_dtype=torch.float32)
        if c.dtype != torch.float32:
            raise RuntimeError("BMM FP32 output unsupported")
        state["bmm_ok"] = True
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

    @triton.jit
    def _splitk_reduce(P, Y, M, N, PW, LO_OFF,
                       S: tl.constexpr, HAS_LO: tl.constexpr,
                       BLOCK: tl.constexpr):
        # P: [S, M, PW] FP32 partials. Y: [M, N] contiguous (FP32 or BF16).
        # out[m, n] = sum_s P[s, m, n] (+ P[s, m, n + LO_OFF] for router hi/lo)
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < M * N
        row = idx // N
        col = idx - row * N
        base = row * PW + col
        acc = tl.zeros((BLOCK,), tl.float32)
        for s in tl.static_range(S):
            acc += tl.load(P + s * M * PW + base, mask, other=0.0)
            if HAS_LO:
                acc += tl.load(P + s * M * PW + base + LO_OFF, mask, other=0.0)
        tl.store(Y + idx, acc.to(Y.dtype.element_ty), mask)


def _splitk(x, cfg, state, s):
    layer, n = cfg["layer"], cfg["n"]
    x = x.contiguous()                       # no-op for the harness's inputs
    m, k = x.shape
    kc = k // s
    # [M, K] -> [S, M, Kc]: strides (Kc, K, 1); cuBLAS takes lda=K, no copy.
    x3 = x.view(m, s, kc).transpose(0, 1)
    if layer == "router":
        wt = state["router_cat"]             # [K, 2W]
        width = state["router"].shape[0]
        pw, lo_off, has_lo = wt.shape[1], width, True
        w3 = wt.view(s, kc, pw)
        out_dtype = torch.float32
    else:
        wt = state[layer + "_t"]             # [K, N]
        w3 = wt.view(s, kc, wt.shape[1])[:, :, :n]
        pw, lo_off, has_lo = n, 0, False
        out_dtype = torch.bfloat16
    part = torch.bmm(x3, w3, out_dtype=torch.float32)   # [S, M, pw], contiguous
    y = torch.empty((m, n), device=x.device, dtype=out_dtype)
    block = 1024
    _splitk_reduce[(triton.cdiv(m * n, block),)](
        part, y, m, n, pw, lo_off, S=s, HAS_LO=has_lo, BLOCK=block,
        num_warps=4,
    )
    return y


def kernel_fn(data, state):
    if not data["x"].is_cuda:
        return _torch(data, state)
    if triton is None:
        raise RuntimeError("Triton is required for GPU candidates")
    x, cfg = data["x"], data["config"]
    m, layer = x.shape[0], cfg["layer"]
    if m == 1 and layer != "router":
        return state["native_gemv"](x, state[layer][:cfg["n"]])
    if state.get("bmm_ok"):
        s = _split(layer, m)
        k = x.shape[1]
        if s > 1 and k % s == 0 and (k // s) % 8 == 0:
            return _splitk(x, cfg, state, s)
    if m > 8 or layer != "router":
        return _torch(data, state)
    # v12 router path for small M (only M=1 reaches here with the table above)
    n, k = cfg["n"], x.shape[1]
    w = state[layer]
    splits = 4
    partial = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    y = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _partial[(triton.cdiv(n, 4), m, splits)](
        x, w, partial, m, n, k,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        4, 256, splits, triton.cdiv(k, splits),
        num_warps=4, enable_fp_fusion=False
    )
    _reduce[(triton.cdiv(m * n, 256),)](partial, y, m * n, splits, 256)
    return y
