Review our H200 dense-projection optimization and propose a small, evidence-grounded next experiment plan. Read current official rules and problem definitions first:
https://openkernelchallenge.dev/llms.txt
https://openkernelchallenge.dev/problems/muse_bf16_proj
https://openkernelchallenge.dev/rules

We want real improvements compatible with prepare(weights)/kernel_fn(data,state), not harness manipulation. Distinguish measured results from hypotheses. Do not invent leader implementations, speedups, profiler evidence, or API availability.

PROBLEM
Compute x[M,K] @ W[:N,K].T. x BF16. qkv_gate W[2304,8192], o[8192,1024], pooler[4096,8192], unpooler[8192,4096] (N1024 at M<=8, otherwise8192), router[256,8192] FP32. Return BF16 except routerFP32. Scored M1,8,16,32,128,256, and nearby/outlier tests including M7,33,130,512,1000.
GPU H200 SXM141GB,132SM,4.8TB/s theoretical,989TFLOP/s denseBF16 theoretical. Timed CUDA graph replays, fresh inputs, coldL2, no host sync or input-derived caching. prepare untimed<=300sec; lossless BF16 repacking allowed. Router allowed BF16hi+lo with FP32 accumulation. RelativeL2<=1e-2, router1e-4, individual rows5x bound, single-spike probes1e-3. No lower-precision quantization or global backend changes affecting baseline.
Official score: geometric mean across six loads of weighted baseline/candidate load times. Each shape weight62. Official timer subtracts fixed graph overhead and reports95%CI at least±1.5%. Local timer does not reproduce it.

CURRENT BEST AND LEADERBOARD
Selected conservative official best is v12, check3951752bc54040f1:1.135028x,95%CI[1.118003,1.152053], normalized4256.106us. Higher nominal v20/v21 points overlap this interval and were not robustly better.
Public problem leaderboard fetched 2026-10-29T13:10:11.253181+00:00: leader colinlikescode 1.636705828x (normalized2951.538us). Our v12 would insert around1118 among1335 returned entries (1117 above), not an actual rank because we have made no scored submission. Leader needs 1.442x our throughput-equivalent performance, or about30.7% lower normalized latency than v12. Do not compare Modal raw aggregates to leaderboard scores.

OFFICIAL MEASUREMENTS (baseline -> v12, us)
M1: qkv14.144->13.337; o6.816->6.187; pooler20.800->18.817; unpooler6.832->6.349; router10.560->7.107.
M8: qkv14.000->13.755; o6.400->6.377; pooler21.072->19.616; unpooler6.560->6.536; router26.400->11.150.
M128:pooler25.488->22.895;router16.512->10.391.
M256:qkv19.840->20.967(regression);pooler30.864->30.276;router17.488->11.087.

WHAT WE TRIED
- Direct small GEMV0.795x; broad splitK GEMV0.850x; generic tiled tl.dot0.203x; custom tensorcore splitK0.744x.
- Router-only originalFP32 splitK1.043x; packed layouts/routerhi+lo GEMM1.112x; nativeM1+wide router yielded selected1.135x.
- Weight-left torch.mm(W,x.T).T1.114x.
- Quack universal splitK1.055x; deliberately different64x128nonpingpongcluster1swapAB with shape-specificsplit1/2/4:1.074x. Pooler12824.480->24.222us only1.011x versus v12~1.113x; QKV25632.99us, unpooler25632.54us. Underoccupancy inferred from assumed tiles is not proof of actual cuBLAS scheduling.
- Router4tokens/CTA1.1464x nominal but changedM8router12.29us worse thanv12 11.15. SafeM1unpooler+QKVfallback1.1452x, overlappingCIs; notselected.
- FlashInfer tinygemm_bf16 for all four BF16 M2..8: official1.136259x, overlapsv12. M8unpooler4.882us strong gain; output10.437us bad regression. PDLFalse, original contiguous W[N,K], explicit output, prepare warmup.

NEW MODAL EVIDENCE
Pinned public SGLang serving image, H200700W, driver610.57.04, torch2.14.1+cu130, FlashInfer0.7.0.post1. Not identical private judge environment. 74 full-dimension correctness cases +30 fresh-input CUDAgraph replays passed forv12,v24,v25. Raw graph diagnostic timing withcoldL2 and alternatingbaseline,60reps; repeat120reps withcandidate order reversed. No official replay-overhead subtraction/CI; raw medians only, separate candidate invocations.
v12 aggregate1.087982x,repeat1.086477x; M8unpooler11.968us both.
v24 (v12+TinyGEMM ONLY unpoolerM2..8) aggregate1.092228x,repeat1.091299x; M8unpooler10.496us,repeat10.224us. Local aggregate gain~0.4%, targetshape12–15% faster. Official confirmation pending. This is the strongest recent repeatable targeted win.
v25 cuBLAS torch.bmm splitK withFP32partials+Tritonreduce:aggregate1.066986x (worse), M8unpooler12.096us, pooler12833.456us vs v12 32.704us, unpooler12833.088us vsv12 29.056us. Explicit deterministic capabilityprobe, no silent fallback. No copy/kernel profiler collected; do not assume zero-copy or exactly2kernels. FP32partials change summation order, no precision guarantee beyond measured tests.
Modal work completed; no active apps/containers. About$0.51 cumulative usage withcredits. No scored submissions. Every codeversion/result/log/hash preserved.

REQUEST
1. Identify the highest-impact remaining bottleneck using score weights and evidence. State what profiling would confirm it.
2. Rank at most3 meaningfully new experiments. Explain why they differ from failures, affected layers/shapes, hardware rationale and likely risks. Avoid broad blind sweeps.
3. Review code correctness, strides/layout, accumulation, masks, graphcapture, freshinputs and nearbyshapes.
4. If suggesting code, give a standalone candidate based on v12 or v24, preserve known goodpaths, use actual installed APIs, and explain every change. No blanket precision reduction/backend flagchanges. Keep any benchmark/capabilitydiagnostics outside timedkernel. Do not silently fall back and disguise an untested path.
5. Give a bounded correctness/profile/pairedtiming plan. Separate expected gains from measured evidence. Aim for a material aggregate gain; tiny improvements alone do not close30%latency gap.

EXACT SELECTED OFFICIALLY TIMED v12 CODE (historical header comment is stale; evidence above supersedes it):
```python
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

```

EXACT BEST RECENT MODAL CANDIDATE v24 (not official-confirmed; inherited docstring is historical):
```python
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

```
