# Modal development results — diagnostic only

> Current scored-pass update: six official scored entries passed. v21 leads at1.158264x; v34 is1.156021x with overlapping intervals. Working submission is v21. User authorized scored submissions and GitHub publication; older prohibitions/status below are historical. See [scored results](SCORED_RESULTS.md) and the README for current selection. Modal evidence remains separate.

These measurements are deliberately separate from [official challenge evidence](OFFICIAL_RESULTS.md). All values are raw CUDA-graph timings on Modal H200; no official replay-cost subtraction or official confidence interval. Do not infer a leaderboard rank or multiply these ratios into an official score.

## Current references

- **Performance reference: v30_qkv_lt.py.** Prior direct paired comparison measured **1.010833x v24** across six weighted loads. QKV128:24.320 versus26.976us; QKV256:27.296 versus32.896us. All other paths retain v24. Exact samples: `results/modal-backends4/gate.json`.
- **Checked implementation: v33_qkv_lt_checked.py.** Same scheduling choices as v30, typed cuBLASLt structures plus returned-state/workspace validation and `AlgoCheck`. Full validation passed. Direct30-shape test with independent weight seed6102 and outlier/fresh inputs measured **0.999271x v30**, effectively unchanged. This is a reliability revision, not another speedup.
- **Official working file remains v12.** No official check or scored submission was launched in this review.

## New revisions and outcomes

| Revision | Change | Validation | Modal outcome / decision |
|---|---|---|---|
| v31_router_native | Original FP32 weights staged in shared memory, one CTA/output row, reduction inside kernel | 74 GPU cases,30 fresh graph replays; independent-weight router outliers at M1/2/6/8 | M1~3–4% lower; M8~97–98% higher latency. Reject broad dispatch; M1 remains a small isolated lead. |
| v32_router_fma | Allow fused FP32 multiply-add in existing Triton router | Same full suite plus targeted independent-weight outlier graphs | M1~0.8% lower; M8~1.6–1.8% higher. No adoption. |
| v33_qkv_lt_checked | Typed algorithm records and explicit compatibility/workspace checks | Full suite plus30 fresh-outlier graphs with independent weights; all checks passed | Same performance as v30 within run variation; retain as checked implementation. |

v31 M8:31.312/31.360us versus15.904/15.840us in two paired rounds. v32 M8:15.968/15.984us versus15.744/15.712us. v31 M1:11.664/11.632us versus12.032/12.128us. Each targeted timing round has120balanced paired samples. A second round in one process is not an independent-run confidence interval.

Additional v30 coverage: exact QKV128/256 branches passed outlier/fresh graph checks under independent weight seeds5152 and6101. The earlier nearby-shape outlier tests selected fallbacks, so this closes a real coverage gap.

Small lead repeats: M8 QKV Lt21.536 versus21.888us (median paired ratio0.98957); cuDNN pooler12831.824 versus32.480us (ratio0.98058). These are target gains only, not integrated aggregate wins. No broad search followed.

## Archive locations

- `results/modal-review1/`: exact v31/v32 sources, validator/driver/runner, full returned JSON, logs, raw targeted samples, installed CUDA header excerpt.
- `results/modal-review2/`: exact v33 source, full validator/runner/drivers, returned JSON,30-shape raw paired samples, algorithm metadata, lead repeats, final billing/container snapshots.
- Earlier `results/modal-*` directories remain in place to preserve links and provenance. They are development evidence, separate from immutable official `results/runs/`.
- [Modal-only archive index](../results/modal-index.json) lists the development directories. [Code-version index](../results/code-versions/index.json) links immutable candidate hashes to results.

## Known methodological and integration limits

Timing uses a128MiB write-based L2 flush, whereas the official description uses a read-based flush. No official overhead subtraction or CI is reproduced. Fresh input correctness is checked before timing; the repeated timed graph uses the final input. Runtime/library versions differ from or are not proven identical to the private judge. Algorithm ordinals remain version-dependent despite `AlgoCheck`. Prepared Lt state assumes serial execution and has no explicit handle/descriptor cleanup; broader serving integration needs these addressed.

## Budget and shutdown

Latest observed cumulative metered usage: **$2.54** of the original **$30** limit. Credits offset billed cost at observation; accounting can lag. Both review apps completed and container list was verified empty. No persistent volume or service was created. No active compute is scheduled by this document.
