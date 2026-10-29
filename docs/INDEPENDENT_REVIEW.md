# Review history and current status

> Current measured status: 20 executed unscored H200 checks and 1 duplicate request archived in [results](../results/README.md). Best measured v12_wide_native.py: 1.1350× (95% interval 1.1180–1.1521×). Initial setup/review notes below predate these runs; consult results for current GPU status. No scored submission.

Current inventory: 21 code versions; 19 executed checks and 1 duplicate request. Quota: 1 check left; 10 scored attempts unused. No pending jobs.

The original separate static review of the initial five candidates is preserved at history/INDEPENDENT_REVIEW.before-pass3.md; its statements that GPU code was untested were true then, not current project status. The subsequent independent Sol/high review supplied pass3 priorities: weight-left GEMM, distinct swapped SM90 schedule, token reuse, safe fallback bundle, and noise-aware selection. Those recommendations were tested as recorded in [pass3](OPTIMIZATION_PASS3.md). Default implementation preference remains Sol/low; high was for the separate review only.

Current selected kernel has official H200 checks. No claim of instruction-disassembly verification, sanitizer coverage, independent fresh repeatability or final competition screening is made.
