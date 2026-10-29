# Open Kernel Challenge: Muse dense projections on H200

A reproducible record of our BF16 projection and FP32-router kernel experiments. Compute `x @ W[:N].T` under the challenge's numerical and fresh-input requirements.

**Best official scored entry: [v21_safe_bundle.py](candidates/v21_safe_bundle.py) — 1.1583x SGLang**, normalized time **4.171ms** versus **4.831ms** baseline. Reported95% interval: **[1.1409, 1.1756]**. Leaderboard snapshot: **rank 1039 of 1354**, team `iyerramji`. Rankings are provisional and change over time; snapshot taken at completion of this scored pass.

The official and Modal measurements below are different experiments and estimators. **Do not multiply Modal gains into an official score.** Differences within overlapping confidence intervals are inconclusive.

## Official scored results

All entries below passed official screening and correctness. These are fresh scored runs, separate from the earlier unscored checks. Exact sources/results/profiles are linked by job.

| Kernel | One-line description | Official speedup | Reported95% interval | Normalized time | Official job |
|---|---|---:|---|---:|---|
| [v21_safe_bundle.py](candidates/v21_safe_bundle.py) | Safe sliced-unpooler and QKV-layout bundle. | 1.1583x | 1.1409–1.1756 | 4.171ms | [433a8be9f5c14d37](results/runs/433a8be9f5c14d37/result.json) |
| [v34_official_bundle.py](candidates/v34_official_bundle.py) | Official winners: Lt QKV128, original QKV layout, sliced M1 unpooler and selective TinyGEMM. | 1.1560x | 1.1387–1.1734 | 4.179ms | [764ac132f5224473](results/runs/764ac132f5224473/result.json) |
| [v24_tiny_unpooler.py](candidates/v24_tiny_unpooler.py) | TinyGEMM restricted to small unpooler batches. | 1.1443x | 1.1271–1.1615 | 4.222ms | [bc65b67e2cf64705](results/runs/bc65b67e2cf64705/result.json) |
| [v33_qkv_lt_checked.py](candidates/v33_qkv_lt_checked.py) | v30 with typed and validated cuBLASLt records. | 1.1413x | 1.1242–1.1584 | 4.233ms | [f494abe5f9f94394](results/runs/f494abe5f9f94394/result.json) |
| [v12_wide_native.py](candidates/v12_wide_native.py) | Native GEMV, packed projections and wide hi/lo router. | 1.1396x | 1.1225–1.1567 | 4.239ms | [e93e60835ffb4770](results/runs/e93e60835ffb4770/result.json) |
| [v20_router_group4.py](candidates/v20_router_group4.py) | Four-token weight reuse in the small-router kernel. | 1.1325x | 1.1155–1.1494 | 4.266ms | [14da83f6a21449d5](results/runs/14da83f6a21449d5/result.json) |

## Modal development results — separate evidence

One H200, cold-L2 paired CUDA-graph timings. These include raw graph replay cost and do not reproduce the official overhead subtraction or confidence interval. Rows name their own control; they are not a common leaderboard.

| Kernel | Modal measurement | Interpretation |
|---|---|---|
| v12 | 1.08648x SGLang (modal-repeat1) | Whole workload diagnostic |
| v24 | 1.09130x SGLang (modal-repeat1) | Whole workload diagnostic |
| v25 | 1.06699x SGLang (modal-resume1) | Whole workload diagnostic |
| v28 | Pooler M1 24.704/24.720us; matched native | Target-only result |
| v29 | Pooler M1 32.048/32.064us; ~28% slower | Target-only result |
| v30 | 1.01083x v24 (modal-backends4) | Whole workload diagnostic |
| v31 | Router M1 ~3–4% faster; M8 ~2x slower | Target-only result |
| v32 | Router M8 ~1.6–1.8% slower | Target-only result |
| v33 | 0.99927x v30; effectively equal | Whole workload diagnostic |
| v34 | 1.00577x v21 (modal-scored-bundle) | Whole workload diagnostic |

## Does Modal predict the official result?

It identified useful target improvements, but not the final aggregate ranking. v33's Modal predecessor improved QKV128/256; official v33 scored1.1413x versus v24's1.1443x, with overlapping intervals. On official QKV256, the original-layout path in v21 was faster than the chosen Lt path. The new v34 combines only independently observed target winners; its official result appears above. Both environments use H200, but driver/library versions, replay overhead and timing estimators differ.

## Every kernel revision

Historical source docstrings are preserved byte-for-byte and may say “unmeasured.” This table and linked results describe current status. A dash means no result in that evidence class, not zero performance. Failed/slower/untested revisions are intentionally retained.

| Revision | What it changes | Best scored result | Earlier check result | Modal evidence |
|---|---|---:|---:|---|
| [v1](candidates/v1_torch_split_router.py) | PyTorch projections with separate BF16 hi/lo router products. | — | — | — |
| [v2](candidates/v2_direct_gemv.py) | Direct Triton GEMV for small batches. | — | 0.7953x (archived starter filename) | — |
| [v3](candidates/v3_splitk_gemv.py) | Small-batch split-K GEMV with a separate reduction. | — | 0.8502x | — |
| [v4](candidates/v4_tiled_fused_router.py) | Tiled tensor-core GEMM with router hi/lo accumulation. | — | 0.2028x | — |
| [v5](candidates/v5_hybrid_persistent.py) | Small GEMV plus persistent tiled GEMM. | — | — | — |
| [v6](candidates/v6_router_splitk.py) | Router-only FP32 split-K refinement. | — | 1.0433x | — |
| [v7](candidates/v7_tensor_splitk.py) | Tensor-core split-K for larger batches. | — | 0.7439x | — |
| [v8](candidates/v8_packed_layout.py) | Packed weights and K-concatenated router hi/lo. | — | 1.1125x | — |
| [v9](candidates/v9_native_hopper.py) | Native Hopper GEMV for single-token BF16 projections. | — | 1.0648x | — |
| [v10](candidates/v10_router_wide.py) | Wide router GEMM with hi/lo output channels concatenated. | — | 1.1262x | — |
| [v11](candidates/v11_layout_native.py) | Native GEMV with packed layouts and K-concatenated router. | — | 1.1289x | — |
| [v12](candidates/v12_wide_native.py) | Native GEMV, packed projections and wide hi/lo router. | 1.1396x | 1.1350x | 1.08648x SGLang (modal-repeat1) |
| [v13](candidates/v13_qkv_layout.py) | Original-layout fallback for larger QKV batches. | — | 1.1350x | — |
| [v14](candidates/v14_hopper_quack.py) | Quack GEMM for larger BF16 projections. | — | 0.9005x | — |
| [v15](candidates/v15_quack_fixed_splitk.py) | Fixed Quack split-K configuration, initial API attempt. | — | — | — |
| [v16](candidates/v16_quack_fixed_api.py) | Repaired API for fixed Quack split-K. | — | 1.0548x | — |
| [v17](candidates/v17_layout_select.py) | Shape-dependent original versus packed weight layouts. | — | 1.1267x | — |
| [v18](candidates/v18_weight_left.py) | Weight-left PyTorch GEMM orientation. | — | 1.1141x | — |
| [v19](candidates/v19_quack_swap64.py) | Swapped-operand Quack with shape-dependent split-K. | — | 1.0743x | — |
| [v20](candidates/v20_router_group4.py) | Four-token weight reuse in the small-router kernel. | 1.1325x | 1.1464x | Not separately benchmarked on Modal |
| [v21](candidates/v21_safe_bundle.py) | Safe sliced-unpooler and QKV-layout bundle. | 1.1583x | 1.1452x | Control for v34 paired run |
| [v22](candidates/v22_flashinfer_tinygemm.py) | TinyGEMM across small BF16 projection batches. | — | 1.1363x | Not separately benchmarked on Modal |
| [v23](candidates/v23_bmm_splitk.py) | BMM split-K with FP32 partials and Triton reduction. | — | — | — |
| [v24](candidates/v24_tiny_unpooler.py) | TinyGEMM restricted to small unpooler batches. | 1.1443x | — | 1.09130x SGLang (modal-repeat1) |
| [v25](candidates/v25_bmm_explicit.py) | BMM split-K with explicit capability checks. | — | — | 1.06699x SGLang (modal-resume1) |
| [v26](candidates/v26_codec_pooler.py) | Lossless weight decoding in a Triton pooler GEMV. | — | — | Pooler M1 39.120us; slower than control |
| [v27](candidates/v27_raw_pooler_control.py) | Raw-weight control for the compressed Triton schedule. | — | — | Pooler M1 32.160us; slower than native |
| [v28](candidates/v28_native_raw.py) | Standalone CUDA reproduction of native Hopper GEMV. | — | — | Pooler M1 24.704/24.720us; matched native |
| [v29](candidates/v29_native_codec.py) | Native schedule with two-plane lossless weight decoding. | — | — | Pooler M1 32.048/32.064us; ~28% slower |
| [v30](candidates/v30_qkv_lt.py) | Selected whole-GEMM Lt algorithms for QKV128/256. | — | — | 1.01083x v24 (modal-backends4) |
| [v31](candidates/v31_router_native.py) | Shared-weight FP32 router with internal reduction. | — | — | Router M1 ~3–4% faster; M8 ~2x slower |
| [v32](candidates/v32_router_fma.py) | FP32 FMA enabled in the Triton router. | — | — | Router M8 ~1.6–1.8% slower |
| [v33](candidates/v33_qkv_lt_checked.py) | v30 with typed and validated cuBLASLt records. | 1.1413x | — | 0.99927x v30; effectively equal |
| [v34](candidates/v34_official_bundle.py) | Official winners: Lt QKV128, original QKV layout, sliced M1 unpooler and selective TinyGEMM. | 1.1560x | — | 1.00577x v21 (modal-scored-bundle) |

## Reproduce

1. Install the official CLI following [Open Kernel Challenge instructions](https://openkernelchallenge.dev/llms.txt). Run `okc login` yourself to authenticate and accept the current rules.
2. Obtain the official problem interface: `okc init muse_bf16_proj problems/muse_bf16_proj`. Downloaded challenge source is not redistributed here; `official/problem.json` contains benchmark metadata for our scripts.
3. In a compatible CUDA/H200 environment with PyTorch, Triton, FlashInfer and the official SGLang runtime, run `python scripts/validate.py --official --device cuda --candidate <candidate_stem>`. This local validator is not the official judge.
4. For an official unscored run, check quota first, then `okc check candidates/<file>.py --json`. `okc submit` is a scored publication and uses a scored attempt.
5. Modal runners under `scripts/modal_*.py` require your own authenticated Modal account and incur compute usage. They use a pinned SGLang image; exact environment metadata accompanies each run. Active scripts have portable project roots; archived runner snapshots retain their original bytes.

## Evidence and recommendations

- [Official scored analysis](docs/SCORED_RESULTS.md)
- [Modal-only ledger](docs/MODAL_RESULTS.md)
- [Recommendations](docs/RECOMMENDATIONS.md)
- [Code hashes and source snapshots](results/code-versions/index.json)
- [Reference repositories and licenses](docs/REFERENCES.md)
- [Publication scope](PUBLICATION.md)

Our contributions are Apache-2.0; third-party material retains its own rights. Pinned third-party repositories, downloaded challenge interface, credentials, local environment and billing/account snapshots are excluded. All our candidate revisions and experiment scripts are included. See NOTICE and third_party/licenses.
