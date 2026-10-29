# Whole-GEMM backend comparison result

Best Modal development candidate from this pass: candidates/v30_qkv_lt.py. Official selected submission remains v12; no official/scored submission performed. Refreshed quota:20/20 checks used,0left;0/10scored used. No pending jobs; containers verified empty.

v30 retains v24 except QKV M128/256: dedicated cuBLASLt handle,32MiB workspace,whole-GEMM BF16 operands/output and FP32 compute. M128 uses packed transpose heuristic2; M256 original layout with transpose descriptor heuristic4. Choices are fixed metadata-dependent, no runtime timing or activation caching. Descriptor/algorithm objects prepared once, current CUDA stream at each call. Other/nearby shapes fall back to v24. Algorithm ordering and ctypes ABI are library-version-dependent: this is validated in the pinned Modal image, not proven portable to the judge. Missing heuristic raises, no silent fallback. Inherited candidate header contains old v24 historical wording; this report is current validation provenance.

## Results

192 Lt configurations across12targets and2layouts passed target reference/fresh graph checks;120 raw paired samples each are saved. Four promising configurations were repeated on new inputs/reversed order. QKV gains repeated strongly; unpooler256 gain disappeared; pooler128 only~1% on repeat. Excluded both from final candidate.

cuDNN first capture pass failed; following errors are cascading invalidated-capture errors, not12independent backend failures. Distinct retry preallocated output and warmed the control before capture; all12targets then passed. QKV128~4%,QKV256~10% reductions were smaller than Lt. Others mostly lost. Exact failure trace limitation: first pass caught exception without traceback; cannot isolate which of the two setup changes resolved it.

v30 passed74GPUcases plus30fresh-input graph replays (validator76records including ancillary records). Full30shape benchmark against SGLang: raw diagnostic1.105792x; same-job v24 diagnostic1.097211x. These sequential diagnostic ratios are not official scores.

Independent direct v30/v24 comparison:120balanced cold-L2 paired samples per shape,3fresh graph seeds each, all30shapes. Six-load geometric mean of control/candidate weighted load sums: **1.010833x**, about1.07% lower aggregate latency. QKV12824.320us vs26.976us;QKV25627.296us vs32.896us. Other paths unchanged; small measured variations remain. No official overhead subtraction or confidence interval, so do not equate this with a new official score.

M8QKV extension16Lt configurations passed; best~2.15%target gain, single measurement only, not incorporated. Current stronger repeated improvement is v30. No broad tuning expansion.

Exact artifacts: results/modal-backends1 (192Lt+initialcuDNN),2(repeat+repairedcuDNN),3(fullv30/v24validation/benchmark),4(finalpaired30shaperepeat+M8extension). Every tested algorithm/layout and raw sample preserved. SHA manifests cover downloaded artifacts and exact diagnostic sources. Compile/correctness failures retained.

## Next
Official unscored check of exact v30 after quota reset. Verify runtime algorithm compatibility and fulljudge screening before promoting to official submission. Keep v12 selected until that result. No scored submissions authorized.
