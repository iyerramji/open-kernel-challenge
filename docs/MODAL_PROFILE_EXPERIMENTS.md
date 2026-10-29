# Profile-driven Modal experiments — October 29, 2026

Selected official code remains v12 (1.135028x); v24 remains development control. No official checks/submissions used.

## Compression gate and consumer
Seed5151 generated official weights: 128-value blocks estimate23.964% pooler and23.954% QKV traffic saving with two-byte metadata and raw-block escapes. This is an estimated variable-format size, not resident allocation reduction. Actual prototype retains raw escape backing plus packed fields; escapes load exact original BF16. Format reconstruction algebra checked on generated weights, full frozen consumer passed74 correctness cases and30 fresh-input graph replays; exhaustive packed codec edge-case bitwise tests were not performed. Do not promote it.

v26 fused Triton decoder targets poolerM1 only; v27 matched raw consumer uses same grid/split/reduction. Diagnostic coldL2 raw graph time: compressed39.120us, matchedraw32.160us, v24native25.216us. Stop this prototype: both custom consumers lose, decoding makes it worse. Sources and all tests/timings preserved in results/modal-codec1.

## Profiles
results/modal-probe1 contains node traces and tables for poolerM1,M8,M128,M256. M1 uses native SGLang hopper_bf16_gemv. These are explanatory eager traces, not selection timings. ncu/nsys/nvcc available; hardware counters were NOT collected and underoccupancy/bandwidth saturation remain unproven.

## Bounded cuBLASLt search
Direct cuBLASLt diagnostic binding, original packed transpose, BF16 operands/output, FP32 compute, dedicated handle/current stream,32MiBworkspace, at most8heuristics each forpoolerM128/M256. First binding declaration omitted onepointer and failed before matmul (modal-lt1); corrected source retained separately(modal-lt2).
All16actual algorithms passed reference check for their tested shape and CUDAgraph capture. Full candidate/nearby/outlier suite NOT performed for this binding. Algorithm descriptors are version-specific; not a submission-ready kernel.
M128 heuristic1:31.680us vsTorch32.464us;120balanced-pair repeat31.536us vs32.352us (~2.5% target-shape gain). Raw pairedsamples retained. M256best37.072us vs37.072us,no useful gain. Stop search at bound.

No combined candidate promoted: only one shape improved modestly; compatibility, portable in-file binding, extra seeds/nearby shapes, graph fresh-input and full-suite validation still needed before incorporating cuBLASLt. A2.5%gain ononeof30shapes cannot close30.65%normalizedlatencygap. v24 remains next official confirmation option.

All apps completed and container list empty. Last available cumulative billing~$0.79before final repeat, credits applied; finalbillingmaylag. Budget$30notexhausted. Stop this bounded pass; retain evidence for next decision.
