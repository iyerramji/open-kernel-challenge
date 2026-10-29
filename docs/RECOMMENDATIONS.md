# Kernel recommendations after independent review

> Current scored-pass update: six official scored entries passed. v21 leads at1.158264x; v34 is1.156021x with overlapping intervals. Working submission is v21. User authorized scored submissions and GitHub publication; older prohibitions/status below are historical. See [scored results](SCORED_RESULTS.md) and the README for current selection. Modal evidence remains separate.

Date: October29,2026. Scope: muse_bf16_proj on H200. Every performance statement below is explicitly official or Modal diagnostic.

## Decision

Select **v21 as the provisional official best**, scored at **1.158264x**. Keep **v34 (1.156021x)** as a close alternative with overlapping uncertainty. v33 remains the checked Lt implementation used to build v34. The first official-confirmation pass is complete; future work should need a new concrete hypothesis, not a noisy-score chase. Modal evidence remains separate.

Independent reviewers found no obvious v30 indexing or precision error. They found missing exact-path outlier coverage, runtime-specific ctypes/heuristic assumptions, absent resource cleanup, and a serial-stream scratch-workspace assumption. This review addresses the first two in new revisions and documents the remaining limits.

## Evidence classes

- [Official results](OFFICIAL_RESULTS.md): challenge checks, official timing method and reported intervals. Selected scored v21:1.158264x; earlier unscored v12:1.135028x.
- [Modal results](MODAL_RESULTS.md): paid development experiments, raw CUDA-graph timing with no official fixed-cost subtraction. v30/v24 measured1.010833x in the prior full-workload paired run; this is not an official score.
- Predictions: score-sensitivity arithmetic and implementation hypotheses. Neither becomes a measured claim until tested.

Exact historical source and returned artifacts are never overwritten. Older summaries are retained in `docs/history/review-2026-10-29/`; the original official run directories remain unchanged.

## What this review tested

1. **v30 exact-path outliers and independent weights.** QKV M128/256 with weight seeds5152/6101, outlier activation distributions and two fresh graph input seeds each passed. Previous outlier tests at129/257 and512 exercised fallback code, so they did not cover these paths.
2. **v31 native FP32 router.** One output row per CTA,32KiB shared original FP32 weights, token warps and in-kernel reduction. Full correctness passed. M1 improved~3–4%; M2/6/8 lost badly (~2–2.5x latency). Retire the multi-token implementation. Preserve the M1 lead but do not add a compiled dependency for a tiny unconfirmed aggregate gain.
3. **v32 FP32 FMA-enabled router.** Changed only Triton contraction setting on the v30 router. Full correctness passed. M1~0.8% lower, M8~1.6–1.8% higher; no meaningful overall winner. Retain v30's setting.
4. **v33 checked Lt binding.** Typed `Algo`/`HeuristicResult` structures follow the actual installed CUDA13 header. Validate returned status/workspace, call `cublasLtMatmulAlgoCheck`, retain real warm-up execution, and expose selected algorithm metadata. Full validation and 30-shape independent-weight outlier graph checks passed. Modal aggregate versus v30 was 0.99927x, effectively unchanged; no performance improvement or official score is claimed.
5. **Small library leads.** Independently repeat original-layout M8QKV Lt and cuDNN pooler128, rather than widening their searches. M8 QKV repeated at ~1% lower target latency; cuDNN pooler128 repeated at ~2%. No combined candidate was created because the expected aggregate change is small and still requires full integration validation.

## Ranked next work

| Priority | Experiment | Why | Acceptance / stopping rule |
|---|---|---|---|
| Completed | Official validation of six candidates | All six passed screening and correctness; v21 has highest point estimate | Preserve remaining four scored attempts; do not alter code to evade deduplication. |
| 2 | Reproducibility and portability hardening | Existing QKV gains are stronger than speculative alternatives | Record cuBLAS version, algorithm configuration and full target validity. Replace ordinal dependence with supported configuration selection if feasible; never assume algorithm IDs/opaque bytes transfer across versions. |
| 3a | Integrate cuDNN pooler128 in a separate candidate | Target gain repeated at roughly 2%; only ~0.1% hypothetical aggregate impact | Warm/prebuild plans, preallocate output during capture, validate full suite and repeat combined workload. Stop if integration removes the gain. |
| 3b | Nearby QKV descriptor-specific regions, M127/129/255/257 | Current exact-shape dispatch is allowed but nearby behavior should be understood | Create descriptors for the actual M, never reuse128/256 descriptors. Bound to existing layouts and a small returned set; freeze only independently repeated wins. |
| 4 | Large-router hi/lo GEMM plus fused reduction | Existing path has a separate hi/lo add, roughly1.6us in an older official profile | First build a supported cuDNN graph with BF16 operands and FP32 intermediate/reduction. Docs permit the operations but do not promise fusion. Stop if unavailable, extra kernels dominate, or repeated full-operation timing loses. |
| 5 | Native router M1-only reuse | Isolated target gain survived two within-process rounds | One independent repeat if pursuing; reject unless gain survives added compile/dependency complexity and aggregate measurement. No further multi-token schedule sweep. |

M8QKV's prior2.15% target gain implies only~0.086% hypothetical aggregate improvement. Pooler128's weak1%repeat implies~0.05%. These deserve one repeat, not a tuning campaign. Small-router M1+M8 account for~5.52% first-order aggregate sensitivity: even10%improvement on both is only~0.55%aggregate. This explains why tiny local wins will not close the leaderboard gap.

Pooler/QKV/unpooler M128/256 account for25.94% first-order aggregate sensitivity; M16/32 add25.41%. Those remain the larger opportunity, but our bounded backend comparison already covered them. A new large-GEMM experiment needs a specific unsupported scheduling/layout hypothesis, not another generic rewrite.

## Retired directions

- Lossless compression using the tested consumers: v29 preserved native schedule but still lost~28%; do not continue that tree.
- Universal BMM split-K, rejected Quack configurations and generic Triton GEMM: no broad repeats without new evidence.
- BF16-only router, lossy weights, activation/output caching, global backend manipulation: incompatible with the numerical/task contract.
- Claiming higher occupancy proves faster execution: counters and complete-operation timing contradicted that assumption.

## Reliability limits to resolve before serving integration

The checked binding still targets the tested CUDA/cuBLAS ABI and ordinal choices. `AlgoCheck` success does not by itself guarantee correct execution with every alignment; actual warm-up/reference checks remain necessary. Handles/descriptors/preferences currently survive for the prepared state's lifetime without explicit destruction; repeated prepare can leak small library resources. Shared scratch assumes serial calls; concurrent replay of the same state is not validated. Add an explicit state lifetime/cleanup contract and concurrency-safe scratch strategy before broader serving use. Do not add an eager destructor that could free resources while captured graphs still reference them.

The diagnostic timer saves raw samples but does not reproduce the judge's replay-cost subtraction or official estimator/CI. Timing uses repeated graph inputs after separate fresh-input checks. Further confidence should come from independent runs/seeds, not only more repetitions in one process. None of these measurements proves leaderboard rank.

## Execution policy

One H200 at a time; preserve every revision including failures, unsupported APIs and slower results. Original cumulative Modal ceiling remains $30, not a fresh budget per turn. Latest observed metered usage is $2.54; credits currently offset billed cost, and accounting can lag. All apps completed and the container list was verified empty. Check live usage and containers after each pass. Scored submissions were authorized for this pass; six of ten attempts were used. The official working file is now the exact best scored v21 source. No persistent GPU deployment.

## Primary sources and skills

- [NVIDIA CUDA Best Practices: shared-memory reuse](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab): supports the reuse mechanism, not a performance guarantee for this router.
- [cuBLASLt algorithm validation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgocheck): validate descriptor compatibility and workspace; retain execution checks.
- [cuDNN matmul](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Matmul.html) and [pointwise/reduction](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Pointwise.html): building blocks for the proposed fused-router feasibility test; no H200 fusion claim.
- Installed `/usr/local/cuda/include/cublasLt.h` excerpt archived in `results/modal-review1/installed-cublasLt-header-excerpt.txt` is the source for v33's ctypes structure.
- NVIDIA's cuTile performance skill informs controlled single-change experiments, but this code is native CUDA/Triton/library GEMM, so a cuTile rewrite is not justified by skill availability. Modal skill1.6.3 matched the installed CLI for these runs.


## Decision after six official submissions

The first priority (official confirmation) is complete. v33 did not establish an aggregate improvement over v24 in the judge. v34 combined official per-shape winners, passed full Modal/official validation, but scored1.156021x versusv21's1.158264x. Intervals overlap; selectv21 provisionally and retainv34. No further quota expenditure is justified merely to chase a noisy point estimate. Four scored attempts remain. Future work should target a specific substantial larger-projection or fused-router opportunity, followed by independent validation. Exact-code deduplication must not be evaded.
