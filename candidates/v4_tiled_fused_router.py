# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
# Original experimental implementation; no H200 benchmark claims.
import torch

def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    r = weights["router"]
    state["hi"] = r.to(torch.bfloat16).contiguous()
    state["lo"] = (r - state["hi"].float()).to(torch.bfloat16).contiguous()
    state["sms"] = torch.cuda.get_device_properties(r.device).multi_processor_count if r.is_cuda else 1
    return state

def _cpu(data, state):
    cfg, x = data["config"], data["x"]
    w = state[cfg["layer"]][:cfg["n"]]
    return x.float() @ w.T if cfg["layer"] == "router" else x @ w.T

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def _tiled(X, W, LO, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               SX: tl.constexpr, SK: tl.constexpr, WN: tl.constexpr,
               WK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, ROUTER: tl.constexpr, PERSISTENT: tl.constexpr):
        tiles_n = tl.cdiv(N, BN)
        tiles = tl.cdiv(M, BM) * tiles_n
        for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
            # N-major tile traversal reuses a weight tile across consecutive M tiles.
            tm = tile % tl.cdiv(M, BM)
            tn = tile // tl.cdiv(M, BM)
            rm = tm * BM + tl.arange(0, BM)
            rn = tn * BN + tl.arange(0, BN)
            rk = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            acc_lo = tl.zeros((BM, BN), tl.float32)
            for step in range(tl.cdiv(K, BK)):
                k = step * BK + rk
                a = tl.load(X + rm[:, None] * SX + k[None, :] * SK,
                            (rm[:, None] < M) & (k[None, :] < K), other=0)
                b = tl.load(W + rn[None, :] * WN + k[:, None] * WK,
                            (rn[None, :] < N) & (k[:, None] < K), other=0)
                acc = tl.dot(a, b, acc)
                if ROUTER:
                    lo = tl.load(LO + rn[None, :] * WN + k[:, None] * WK,
                                 (rn[None, :] < N) & (k[:, None] < K), other=0)
                    acc_lo = tl.dot(a, lo, acc_lo)
            if ROUTER:
                acc = acc + acc_lo
            tl.store(Y + rm[:, None] * N + rn[None, :], acc,
                     (rm[:, None] < M) & (rn[None, :] < N))

def kernel_fn(data, state):
    if not data["x"].is_cuda:
        return _cpu(data, state)
    if triton is None:
        raise RuntimeError("Triton is required for GPU candidates")

    x, cfg = data["x"], data["config"]
    m, n, k = x.shape[0], cfg["n"], x.shape[1]
    router = cfg["layer"] == "router"
    w = state["hi"] if router else state[cfg["layer"]]
    lo = state["lo"] if router else w
    y = torch.empty((m, n), device=x.device,
                    dtype=torch.float32 if router else torch.bfloat16)
    bm, bn, bk = 16 if m <= 32 else 32, 64, 32
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    grid = min(tiles, state["sms"]) if False else tiles
    _tiled[(grid,)](x, w, lo, y, m, n, k, x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), bm, bn, bk, router, False,
        num_warps=4, num_stages=3)
    return y
