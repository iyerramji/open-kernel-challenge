# V22 native FlashInfer TinyGEMM result

Official unscored H200 job b5024e063ec34b1b: correctness and screening passed (no flags). Aggregate **1.136259× SGLang**, reported95% interval **[1.119215,1.153303]**, normalized score4251.495μs. Retained v12 is1.135028×, interval[1.118003,1.152053], score4256.106μs. V22's point estimate is only about0.108% faster than v12, far inside the reported uncertainty; **no clear aggregate improvement**. V12 remains selected, byte-for-byte unchanged.

| M8 shape | SGLang μs | v22 μs | Paired speedup |
|---|---:|---:|---:|
| QKV/gate |14.256|14.087|1.012×|
| Output projection |6.496|10.437|0.622×|
| Pooler |21.248|19.254|1.104×|
| Sliced unpooler |7.104|4.882|1.455×|
| Router, unchanged path |26.272|11.263|2.333×|

The output projection is the clear losing TinyGEMM target. Sliced-unpooler M8 is the strongest useful target; compared with v12's6.536μs, v22's4.882μs is a promising cross-run improvement, but a new selective combination needs its own official check. Do not claim an untested subset's total score. QKV/pooler changes are smaller and require paired profile interpretation.

Pinned API preflight: downloaded flashinfer-python0.7.0.post1 wheel without installation/execution, inspected tinygemm_bf16 export/signature and SM90+ BF16/FP32-accumulation restrictions. Extracted source and wheel hash saved under official/flashinfer-wheel-0.7.0.post1/. Actual official execution then confirmed import/build/correctness/capture/timing compatibility. Source candidate uses original BF16 weights, router exclusion, metadata-only targets, synthetic prepare warmups with no retained activation results and PDL disabled.

Exact user-supplied code is preserved unchanged at candidates/v22_flashinfer_tinygemm.py; SHA256 e7fcfb1fa362d5301648cc3909df5c3379a702b0b3a856a64433b00aa6c9219a. Run archive results/runs/b5024e063ec34b1b/ contains exact source, returned JSON, available profile/log files, returned log tail, per-shape CSV, official interface and hashes. The original preflight review is preserved separately.

Current inventory:22 code versions,20 executed checks,one duplicate response. **Unscored quota:0 left; scored attempts:10 unused.** No scored submission or paid GPU. No further GPU check was started. Future selective TinyGEMM candidate should exclude the losing output projection and be measured after quota resets; it is not fabricated or marked tested here.
