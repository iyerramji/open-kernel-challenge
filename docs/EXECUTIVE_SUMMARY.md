# Executive summary

> Current scored-pass update: six official scored entries passed. v21 leads at1.158264x; v34 is1.156021x with overlapping intervals. Working submission is v21. User authorized scored submissions and GitHub publication; older prohibitions/status below are historical. See [scored results](SCORED_RESULTS.md) and the README for current selection. Modal evidence remains separate.

We optimize `muse_bf16_proj`: BF16 dense projections and the FP32 router. The operation is `x @ W[:N].T`. The selected official code is still v12 in `racer/muse_bf16_proj/submission.py`, checked at1.135028x SGLang. No scored submission exists.

The strongest new development gain comes from **choosing a better cuBLASLt algorithm and weight layout for QKV at128/256tokens**. v30 retains v24's other paths, including selective TinyGEMM for small unpooler. A paired Modal run measured9.8% and17.0% lower QKV target latency, translating to1.010833x across the diagnostic workload. This is not a new official score.

The new **v33_qkv_lt_checked.py** preserves those choices and validates typed algorithm records, workspace bounds and `cublasLtMatmulAlgoCheck` results. Full GPU validation and independent-weight outlier graphs passed; performance stayed effectively equal to v30. Library portability, explicit resource lifecycle and concurrent-stream handling still need review before broad serving integration.

Two router experiments were correct but unhelpful overall: v31's single CUDA kernel improved one token modestly and nearly doubled eight-token latency; v32's FMA change was near noise and regressed eight tokens. They remain archived, not selected. Small cuDNN pooler128 and Lt M8QKV leads repeated at~2%/~1%target gains, with much smaller possible aggregate benefit.

Use the separate [official evidence](OFFICIAL_RESULTS.md), [Modal ledger](MODAL_RESULTS.md), and [recommendation document](RECOMMENDATIONS.md). Every revision and result is preserved. No active GPU containers remain; latest observed Modal usage~$2.54 against the cumulative$30limit, subject to billing lag.
