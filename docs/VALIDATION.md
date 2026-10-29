# Current validation status

> Latest result: v22 TinyGEMM passed at1.136259×, statistically tied with v12; selected working code remains v12. Twenty executed checks plus one duplicate response;22 code versions. Zero unscored checks left. See [TinyGEMM result](V22_TINYGEMM_RESULT.md).

> Current measured status: 20 executed unscored H200 checks and 1 duplicate request archived in [results](../results/README.md). Best measured v12_wide_native.py: 1.1350× (95% interval 1.1180–1.1521×). Initial setup/review notes below predate these runs; consult results for current GPU status. No scored submission.

Current inventory: 21 code versions; 19 executed checks and 1 duplicate request. Quota: 1 check left; 10 scored attempts unused. No pending jobs.

All four pass3 candidates v18/v19/v20/v21 passed official H200 correctness and30 timed shapes; v21 returned nine pretests,30 benchmarks,30 profiles and no screening flags. CPU compact fallback/interface tests passed for each. Fourteen grouped-router scratch exact-cover models passed. These local models do not replace GPU checks. Source masks, original precision, timed scratch writes and metadata-only dispatch were reviewed.

V15's earlier API failure is retained accurately as failed with no speed; other slower experiments retain their results. v1/v5 remain GPU-untested, per code-version inventory. CPU fallback results never count as GPU verification. No sanitizer, independent local GPU testing, final organizer screening or end-to-end serving validation has been run. The old setup-only report is preserved in history/VALIDATION.before-pass3.md.

Exact artifacts: [run inventory](../results/runs/index.json), [code inventory](../results/code-versions/index.json), [third-pass report](OPTIMIZATION_PASS3.md). Every archive artifact hash and selected working SHA256 was verified. Selected v12 SHA256:1891556cd61af2cab26feff9e8c408a6a995fb84d3dbd48293a02552efc17749.
