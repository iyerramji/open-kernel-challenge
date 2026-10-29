# Third optimization pass — review-driven result

**Conservative working selection remains v12: 1.135028× SGLang, reported 95% interval [1.118003, 1.152053].** No candidate in this pass established a clearly separated aggregate improvement. Highest observed point estimate is v20 at 1.146420×, but its changed M8 router path regressed; the safe final bundle v21 measured 1.145182×. Their intervals overlap v12. Neither reached the review's approximate 1.17× conservative separation threshold. Code and results are preserved rather than promoted on noise.

| Iteration | Job | Correctness | Aggregate speedup | Reported 95% interval |
|---|---|---|---:|---|
| v18_weight_left.py | c20ac75c8d374b01 | Passed | 1.114107× | 1.097396–1.130819× |
| v19_quack_swap64.py | be8e2f9477dd409f | Passed | 1.074259× | 1.058145–1.090373× |
| v20_router_group4.py | 818293c2812c469e | Passed | 1.146420× | 1.129224–1.163617× |
| v21_safe_bundle.py | 247425e7386843e5 | Passed | 1.145182× | 1.128004–1.162359× |

v18 computes BF16 projections at M≥8 as torch.mm(W[:N], X.T).T. Output is a valid noncontiguous [M,N] BF16 view; no contiguous requirement was imposed by TASK/checks. CPU checks and official CUDA correctness/capture/timing passed. It was slower overall. Pooler M128 reached 1.141× versus its paired baseline, but pooler M256 regressed to 0.906× and QKV M256 to 0.909×; no broad orientation change was retained.

v19 is deliberately different from prior Quack attempts: SM90 64×128, non-pingpong, cluster1, swapped operands. QKV uses split4 below M192 and split2 above; pooler split2 below M192 and split1 above; unpooler split1. Only those layers at M≥64 use it. The smaller/swapped tile increases useful task count without universal split4 reductions. BF16 and FP32 accumulation rules are preserved. It passed but still lost: QKV M256 32.99 μs (0.600×), pooler M256 33.85 μs (0.911×), unpooler M256 32.54 μs (0.735×). No region was strong enough to retain.

API evidence: downloaded, without installing/executing, the PyPI quack-kernels 0.6.5 wheel named by the runtime pin. Wheel, extracted gemm_interface/gemm_config/gemm_sm90/autotuner source and hashes are under official/quack-wheel-0.6.5/. Unlike the earlier repository-source assumption, this wheel shows the actual Autotuner wrapper with .fn, explicit config fields and supported SM90 tile limits. V19 uses the underlying callable already demonstrated in v16; no module/harness/cache modification. Successful judge execution confirms that callable/config path works in its installed environment. Package version/build identity inside the judge was not independently printed, so source-version equality is not asserted beyond the runtime pin and successful API validation.

v20 adds _partial_grouped: four tokens per CTA for router M2–8, reusing original FP32 weights, FP32 accumulation and masked stores to [split,M,N] scratch. M1 and larger router paths unchanged. Fourteen local exact-cover models (M2–8, partial N groups) and official correctness passed. However router M8 measured 12.29 μs versus 11.15 μs in v12; the overall higher point estimate is not evidence this changed region improved. It is not included in the final bundle.

v21 reserved the final check for substantive supported changes: v12 router/pooler layout, original-layout QKV fallback at M≥192, and v3's split-K path only for sliced-unpooler M1. It does not extend that path to M8. All scratch entries are written during each call, original BF16 values are multiplied in FP32, and outputs remain BF16. It passed all nine official correctness cases and 30 scored shapes with no screening flags.

| Representative case | Run | SGLang μs | Candidate μs | Speedup |
|---|---|---:|---:|---:|
| Sliced unpooler M1 | retained v12 | 6.83 | 6.35 | 1.076× |
| Sliced unpooler M1 | v21 | 7.01 | 5.53 | 1.267× |
| QKV/gate M256 | retained v12 | 19.84 | 20.97 | 0.946× |
| QKV/gate M256 | v21 | 19.95 | 19.92 | 1.002× |
| Router M8 | retained v12 | 26.40 | 11.15 | 2.368× |
| Router M8 | v20 grouping | 26.42 | 12.29 | 2.149× |

Cross-run absolute differences include device/run variation; use each run's paired ratios, reported intervals and archived samples. Score is the geometric mean across six loads of weighted SGLang/candidate projection-load times (shape weight62), not a mean of individual shape ratios or a single kernel latency. The review's estimated score sensitivity prioritizes pooler31.9%,QKV22.5%,unpooler20.9%,router14.3%,output10.4%; fixing QKV256 alone contributes only about0.21% aggregate, explaining why that per-case fix did not change overall rank materially.

Quota: started with5 checks; four substantive tests used4. **Final: one unscored check left, ten scored attempts unused. No jobs pending.** No scored submission, paid service, account/terms action, publication or denied Codex UI retry. No check was spent on cosmetic changes or deduplication evasion. The reserved final check tested v21's real algorithm/dispatch changes. Previous exact-v12 repeat request remains duplicate5455e67879604de5 of3951752bc54040f1; no fresh repeatability measurement is claimed.

All21 source versions,19 executed check jobs and one duplicate request are preserved under results/code-versions and results/runs, including failed/slower earlier experiments. Every run contains exact source, full returned result JSON, available profile/log files, returned log tail, per-shape CSV where measured, official interface snapshots and hashes. All artifact hashes verified. Initial setup-only documents were moved to docs/history before replacing stale introductory status text.
