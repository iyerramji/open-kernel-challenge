# Current optimization progress

> Latest result: v22 TinyGEMM passed at1.136259×, statistically tied with v12; selected working code remains v12. Twenty executed checks plus one duplicate response;22 code versions. Zero unscored checks left. See [TinyGEMM result](V22_TINYGEMM_RESULT.md).

Working selection remains v12 at1.135028× SGLang, 95% interval[1.118003,1.152053]. Highest observed point estimate is v20 at1.146420×; safe bundle v21 is1.145182×. These differences are inconclusive, and v20's changed M8 router path regressed. We kept v12 rather than promote noise.

Current inventory: 21 code versions; 19 executed checks and 1 duplicate request. Quota: 1 check left; 10 scored attempts unused. No pending jobs.

See [third-pass report](OPTIMIZATION_PASS3.md) for all four new experiments, exact APIs/wheel evidence, per-shape gains/regressions, correctness and uncertainty. [Second pass](OPTIMIZATION_PASS2.md) and docs/history preserve prior decisions. [Results](../results/README.md) contains all executed runs and a separate duplicate response.

No fresh repeat of v12 exists: the service deduplicated exact code. No deduplication evasion was attempted. Final reserved check was used for v21's substantive QKV/sliced-unpooler combination. No scored submission or paid GPU use.

Modal follow-up: see docs/MODAL_RECOMMENDATIONS.md and results/modal-resume1, results/modal-repeat1. v24/v25 correctness passed; v12 remains official selected source.

Completed profile/compression/cuBLASLt pass: docs/MODAL_PROFILE_EXPERIMENTS.md. Compression slower; pooler128Lt~2.5%targetgain repeated, not integrated/submission-ready. V12selected/v24control unchanged. No active compute.

Focused counters completed: docs/POOLER_COUNTER_FINDINGS.md. Actualreads fell23%,decoderexecutioncost outweighedbenefit; retiretestedconsumer. Noactivecompute; v12selected/v24control unchanged.


Latest native reproduction/codec experiment completed: see docs/NATIVE_COMPRESSION_RESULT.md. v28 passes raw timing gate; v29 correct but~28% slower, retire compression. v12 selected/v24 control. No active containers or pending jobs.


Backend comparison completed: docs/BACKEND_COMPARISON_RESULT.md. v30 best Modal candidate,1.010833x v24 direct paired aggregate; QKV128/256 gains repeated. Full validation passed. cuDNN tested; no stronger winner. M8 gain small/unconfirmed,excluded. Official quota refreshed0checks,10scoredunused; no submission. No active containers. Next official check exactv30 after reset.


Review completed October29: see docs/RECOMMENDATIONS.md and separate docs/OFFICIAL_RESULTS.md / docs/MODAL_RESULTS.md. v31/v32 correct but no broad win; v33 checked Lt implementation validates fully and is performance-equivalent to v30. Independent-weight exact Lt outliers passed. Small leads repeated~1%M8QKV/~2%pooler128,not integrated. Official v12 unchanged; no official/scored check. All code/results archived; no active containers. Latest metered usage~$2.54 of cumulative$30, accounting may lag. Next priorities: official quota refresh/confirmation, portable Lt binding and lifecycle, bounded pooler128 integration and nearby descriptor tests; no further compression.


Scored/publication pass: user authorized at least5scored entries and GitHub repo. Submitted6fresh entries, all passed. Bestv21=1.158264x, v34=1.156021x, overlappingCIs. Working source promoted exactly tov21job433a8be9f5c14d37; originalv12preserved. Remaining4scored/0checks. GitHub private repo https://github.com/iyerramji/open-kernel-challenge created; publication completion recorded separately. Full scored/Modal comparison in README/docs/SCORED_RESULTS.md.
