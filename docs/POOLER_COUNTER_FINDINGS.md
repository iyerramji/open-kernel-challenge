# Focused pooler M1 hardware counters

Nsight Compute2025.3.1; application replay, explicit coldL2 before operation, cache-control none, clock-control none. Requested nine metrics after query; all three collections succeeded. Hardware counters available. No code changes/promotions or official checks.

| Main kernel metric | Native v24 | Raw custom v27 | Compressed v26 |
|---|---:|---:|---:|
| DRAM read bytes |67134464|67139584|51601664|
| DRAM throughput % sustained peak |66.30|69.95|45.42|
| Registers/thread |29|64|40|
| Achieved occupancy % |45.29|47.07|65.70|
| Scheduler issue-active % |26.75|24.65|76.21|
| Eligible warps/cycle |0.38|0.35|2.60|
| Waves/SM |0.48|3.88|2.59|

Native specialization confirmed hopper_bf16_gemv_kernel<4096,8192,1,2,8>,grid512,block256. Raw/compressed have consumer+separate reduction. Compression reduced actual main-kernel DRAM reads23.14% versusraw, close to format estimate. It nevertheless loses complete-operation unprofiled timing39.12us versusraw32.16 andnative25.22. Much higher issue/SM activity and lower memorythroughput are consistent with added execution work overwhelming traffic savings, not proof of a specific instruction bottleneck. Raw64registers versusnative29 is an identifiable resource difference but alone does not prove register-limited performance.

Important anomaly: DRAM write counts were native2.917MB,raw26.270MB,compressed25.048MB, much larger than logical outputs/partials. Do not label those as spills or application writes without validating replay/background effects and local-memory counters. No instruction/SASS or WarpState sections collected; arithmetic FFMA versusseparate MUL/ADD remains unverified. Native/raw issueactivitylow andeligiblewarps~0.35–0.38 suggest limited readiness; precise dependency/memory-stall attribution requires additional evidence. Occupancy alone did not predict winners.

Decision: retire tested compressed consumer; preserve v24 control and v12 official selected source. A native-schedule compressed successor remains conditional, not a demonstrated likely win. No automatic next implementation launched. Allcontainers verified stopped. Rawresults,commands,version,query,source andhashes preserved in results/modal-counters1. Profiler durations not used as performance verdict.
