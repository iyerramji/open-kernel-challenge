# Second optimization pass — bounded result

**v12 remains selected: 1.135028× SGLang, reported 95% interval [1.118003, 1.152053].** No aggregate improvement was established in this pass. Working source is racer/muse_bf16_proj/submission.py, exact archived v12 SHA256 1891556cd61af2cab26feff9e8c408a6a995fb84d3dbd48293a02552efc17749. Original v12 remains untouched in candidates/ and its run archive.

| Experiment | Job | Outcome | Reported 95% interval |
|---|---|---|---|
| v13_qkv_layout.py: original QKV layout at M≥64 | 6ac516b77af64444 | Correct, 1.135019×; effectively tied with v12 | 1.117994–1.152045× |
| v14_hopper_quack.py: installed Quack default Hopper GEMM at M≥64 | 9a6da5bd038c414e | Correct, 0.900464×; slower | 0.886957–0.913971× |
| v15_quack_fixed_splitk.py: explicit 128×128/cluster1/splitK4 | 969e9b707e4c4a67 | Failed API call; no speed result | Not measured |
| v16_quack_fixed_api.py: same configuration via underlying callable | 7e269cf271404c6b | Correct, 1.054762×; slower than v12 | 1.038940–1.070583× |
| v17_layout_select.py: original output layout and QKV at M≥192 | 708758902e114c3a | Correct, 1.126708×; no improvement | 1.109808–1.143609× |

The v15 failure was TypeError: Quack's installed autotuner wrapper received multiple values for config. The inspected GitHub-tag API and installed wheel wrapper differed. V16 selected the underlying library function with getattr(gemm_tuned, 'fn', gemm_tuned) during prepare, without changing a module, cache, timing harness or library code. This fixed the API failure but not performance. No runtime autotune search was intended; v15 errored in its wrapper before returning a result. All source versions and the full trace are preserved.

Concrete timings (official paired shape times; microseconds):

| Case/run | SGLang | Candidate | Meaning |
|---|---:|---:|---|
| QKV M256, retained v12 | 19.84 | 20.97 | Existing 5.7% regression remains in retained best |
| QKV M256, v13 | 19.73 | 19.76 | Original layout removes regression, but no aggregate gain |
| QKV M256, v17 | 19.30 | 19.39 | Again near baseline |
| Output M32, v17 | 7.01 | 6.95 | Small gain, within run uncertainty |
| Output M16, v17 | 6.91 | 7.03 | Small regression |
| Pooler M128, retained v12 | 25.49 | 22.89 | 1.113× gain remains |
| Router M8, retained v12 | 26.40 | 11.15 | 2.368× gain remains |
| Quack fixed QKV M256, v16 | 19.62 | 30.55 | Split-K configuration still loses |
| Quack fixed unpooler M256, v16 | 24.06 | 43.97 | Strong regression, rejected |

Why not repeat the initial generic tensor-core designs blindly: v4 used small 16/32×64 output tiles, BK32 and long serial K loops; v7 shortened loops/split K but retained small output tiles and repeated operand work. Source suggests loop/setup overhead, low reuse and lack of architecture-specific pipelining, but no SASS/counter/occupancy diagnosis was performed. Those are hypotheses, not proven hardware causes. The default Hopper-library kernels were directly profiled around 59–60 μs for QKV/pooler M256; the explicit tile/cluster/split experiment improved that path but still lost against cuBLAS. More broad tuning was stopped.

Repeatability request: exact v12 code was rechecked as job **5455e67879604de5**. Service returned status **duplicate**, duplicate_of **3951752bc54040f1**, result **null**. Therefore **no fresh repeat measurement exists**. We did not alter code to evade deduplication or treat the old result as a repeated run. The duplicate response/code/hash is archived separately; quota did not change. This is an API deduplication limitation, not a permission denial or pending job.

Aggregate definition: geometric mean across the six loads of SGLang's weighted projection-load time divided by candidate load time; each shape weight is 62. Selected official normalized score remains 4256.106 μs versus 4830.8 μs reference, not one kernel-call latency. Overlapping intervals do not establish tiny differences between v12/v13/v17 as real improvements.

Start quota 10 unscored checks remaining; five executed checks used. Final refreshed quota: **5 unscored checks left**, **10 scored attempts unused**. No jobs pending. No scored submission, paid GPU, publication or Codex UI access. All 17 source versions, 15 executed jobs and one duplicate request are preserved under results/code-versions and results/runs with exact code, returned JSON, available profiles/logs, per-shape CSV, official interface snapshots and SHA256 manifests. Archive hashes verified.
