# Validation and timing commands

> Current measured status: 20 executed unscored H200 checks and 1 duplicate request archived in [results](../results/README.md). Best measured v12_wide_native.py: 1.1350× (95% interval 1.1180–1.1521×). Initial setup/review notes below predate these runs; consult results for current GPU status. No scored submission.

The user must personally review/accept rules through login. The user has since completed login and terms acceptance personally; do not repeat login unless required. The CLI is project-local and SHA256-checked. Future manual checks (inspect each result and quota before the next; no auto-submit loop):

```sh
cd /Users/ramjiiyer/git-clone/open-kernel-challenge
tools/okc login                      # USER performs this, if desired
tools/okc problems --json
tools/okc check candidates/v1_torch_split_router.py --json
tools/okc check candidates/v2_direct_gemv.py --json
tools/okc check candidates/v3_splitk_gemv.py --json
tools/okc check candidates/v4_tiled_fused_router.py --json
tools/okc check candidates/v5_hybrid_persistent.py --json
tools/okc leaderboard muse_bf16_proj --json
```

Each candidate is a single standalone file with `# okc: problem=muse_bf16_proj` first line. The judge uses its own reference, not ours. CLI profiles appear under candidates/profiles/. Retain JSON, logs, profile, code SHA256 and environment; failed compile/check/timeout is not a speed result. Checks consume daily check quota; no scored or unscored runs were sent.

On an already-authorized Linux H200 runtime using official dependencies (not this Mac's arm64 venv), copy/mount this workspace and use its Python. Official image digest is in CONSTRAINTS.md. No image was pulled or GPU provisioned here.

```sh
nvidia-smi -q > results/gpu-environment.txt
python -m pip freeze > results/gpu-packages.txt
python scripts/validate.py --device cuda --official --output results/gpu-validation.json
python scripts/bench.py candidates/v2_direct_gemv.py --output results/v2-local.csv
```

CUDA validator runs all nine documented pretests, 30 scored shapes, 35 nearby/outlier shapes (74 cases per candidate) with fresh seeds, then captures/replays all 30 scored shapes after replacing input contents. It calls downloaded official checks. This is our local runner, not the private evaluator. Missing Triton on a GPU raises an error rather than silently falling back. CPU mode is explicitly fallback-only.

Diagnostic bench compares captured baseline/candidate graphs, flushes 128 MiB before each timed replay, alternates order and saves medians, paired ratios, code hash and environment. Raw graph overhead is included. It does not reproduce official fixed-cost subtraction, confidence interval or two-repetition estimator and must not be called an official score. Baseline import needs compatible pinned SGLang/dependencies. Do not inject a source checkout into an unrelated binary environment and assume compatibility; use official okc check when uncertain.

CPU commands:

```sh
.venv/bin/python scripts/validate.py --output results/cpu-validation.json
.venv/bin/python scripts/validate.py --device cpu --official --candidate v1_torch_split_router --output results/cpu-official-dimensions.json
```

Compact mode covers 13 M values × 5 layers × 5 candidates, including dtype/finite/L2/probes and unchanged weights; separate real-width router hi/lo algebra check. It does not execute GPU code. Full-dimensional CPU control checks the real computation at all 74 cases, not CUDA dispatch. See VALIDATION.md for passed/failed/not-run distinctions.

Fill results/official-results-template.csv and experiment-template.json only with observed data; retain official aggregate speedup/95% interval and code identity. Choose by correctness plus measured official results. Tune tiles/splits/router path after actual profiles identify bandwidth, occupancy or launch overhead; repeat checks on nearby shapes. No fastest candidate is selected yet.
