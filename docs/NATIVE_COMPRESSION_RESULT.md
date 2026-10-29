# Native schedule compression feasibility — October 29

Recommendation executed as a bounded two-stage experiment. NVIDIA live catalog checked; cuTile skills do not directly apply to this CUDA schedule reproduction. Modal skill1.6.3 matched CLI.

| Candidate | Round1 candidate/control us | Round2 candidate/control us | Median paired ratios |
|---|---|---|---|
| v28 raw native reproduction | 24.704 / 24.832 | 24.720 / 24.928 | 0.99293,0.99105 |
| v29 two-plane compressed loader | 32.048 / 25.008 | 32.064 / 25.056 | 1.27809,1.27962 |

Raw passes the <=1.02 gate in both rounds. Compression loses about28%; stop compression work and do not extend to QKV. v12 remains officially established; v24 remains development control.

Each candidate passed the existing74-case GPU suite and30 fresh-input graph replays (validator reports76 records including metadata/algebra), plus3 fresh target graph inputs in timing driver. v29 prepare checks exact reconstruction of every supplied pooler weight from packed planes; host exhaustive encoding check passed509952 valid combinations. Full exhaustive GPU decoder special-bit-pattern testing was not performed; no winning codec is being promoted.

Both timing rounds contain120 balanced paired cold-L2 graph samples saved verbatim. Raw graph times include replay overhead, without official overhead subtraction or confidence intervals. No official check or scored submission. Results are target-only performance evidence, not aggregate speedups.

v28 adapts Apache-2.0 SGLang's schedule with current-stream PyTorch binding:512 CTAs,256threads,shared activation staging,streaming float4 loads,explicit FP32 FMA dot8 grouping,warp shuffle reduction. v29 changes weight loads to low-byte/nibble planes plus128-value base/escape metadata and original raw escape backing. Resident storage increases. prepare uses load_inline compilation; official screening compatibility of that compilation path remains unverified. No SGLang module or global backend setting was modified.

Exact candidates, sources, license, runtime metadata, logs, raw samples and hash manifests: results/modal-native1 and results/modal-native2. Apps completed; container list verified empty. No further jobs pending.
