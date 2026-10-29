# Executive summary

> Current scored-pass update: six official scored entries passed. v21 leads at1.158264x; v34 is1.156021x with overlapping intervals. Working submission is v21. User authorized scored submissions and GitHub publication; older prohibitions/status below are historical. See [scored results](SCORED_RESULTS.md) and the README for current selection. Modal evidence remains separate.

We optimize BF16 dense projections and the FP32 router for `muse_bf16_proj`, computing `x @ W[:N].T` on H200.

Six scored candidates passed official screening and correctness. **v21_safe_bundle.py leads at1.158264x SGLang**, normalized4170.724us, reported95% interval[1.140890,1.175638]. The working submission is its exact scored source. The new v34 combination scored1.156021x with overlapping uncertainty; it is preserved as a close alternative.

Modal identified stronger QKV128/256 algorithms, but the aggregate improvement did not transfer directly to official ranking. v33 scored1.141305x versusv24's1.144293x; v21's original-layout QKV256 was better in the official environment. v34 combined official target winners, then passed both Modal and official validation, but did not establish a clear aggregate gain over v21.

The public repository preserves all34candidate revisions, six scored outcomes, historical checks, exact source hashes and separate Modal results. Four scored attempts remain. No Modal GPU containers are active. See [scored results](SCORED_RESULTS.md), [Modal ledger](MODAL_RESULTS.md), [recommendations](RECOMMENDATIONS.md), and the README.
