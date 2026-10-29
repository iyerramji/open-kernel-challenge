"""Build team-facing tables from archived measurements without launching any runs."""
from pathlib import Path
import json,hashlib,csv,math,collections
R=Path(__file__).resolve().parents[1]
descriptions={1:'PyTorch projections with separate BF16 hi/lo router products.',2:'Direct Triton GEMV for small batches.',3:'Small-batch split-K GEMV with a separate reduction.',4:'Tiled tensor-core GEMM with router hi/lo accumulation.',5:'Small GEMV plus persistent tiled GEMM.',6:'Router-only FP32 split-K refinement.',7:'Tensor-core split-K for larger batches.',8:'Packed weights and K-concatenated router hi/lo.',9:'Native Hopper GEMV for single-token BF16 projections.',10:'Wide router GEMM with hi/lo output channels concatenated.',11:'Native GEMV with packed layouts and K-concatenated router.',12:'Native GEMV, packed projections and wide hi/lo router.',13:'Original-layout fallback for larger QKV batches.',14:'Quack GEMM for larger BF16 projections.',15:'Fixed Quack split-K configuration, initial API attempt.',16:'Repaired API for fixed Quack split-K.',17:'Shape-dependent original versus packed weight layouts.',18:'Weight-left PyTorch GEMM orientation.',19:'Swapped-operand Quack with shape-dependent split-K.',20:'Four-token weight reuse in the small-router kernel.',21:'Safe sliced-unpooler and QKV-layout bundle.',22:'TinyGEMM across small BF16 projection batches.',23:'BMM split-K with FP32 partials and Triton reduction.',24:'TinyGEMM restricted to small unpooler batches.',25:'BMM split-K with explicit capability checks.',26:'Lossless weight decoding in a Triton pooler GEMV.',27:'Raw-weight control for the compressed Triton schedule.',28:'Standalone CUDA reproduction of native Hopper GEMV.',29:'Native schedule with two-plane lossless weight decoding.',30:'Selected whole-GEMM Lt algorithms for QKV128/256.',31:'Shared-weight FP32 router with internal reduction.',32:'FP32 FMA enabled in the Triton router.',33:'v30 with typed and validated cuBLASLt records.',34:'Official winners: Lt QKV128, original QKV layout, sliced M1 unpooler and selective TinyGEMM.'}
modal={12:'1.08648x SGLang (modal-repeat1)',20:'Not separately benchmarked on Modal',21:'Control for v34 paired run',22:'Not separately benchmarked on Modal',24:'1.09130x SGLang (modal-repeat1)',25:'1.06699x SGLang (modal-resume1)',26:'Pooler M1 39.120us; slower than control',27:'Pooler M1 32.160us; slower than native',28:'Pooler M1 24.704/24.720us; matched native',29:'Pooler M1 32.048/32.064us; ~28% slower',30:'1.01083x v24 (modal-backends4)',31:'Router M1 ~3–4% faster; M8 ~2x slower',32:'Router M8 ~1.6–1.8% slower',33:'0.99927x v30; effectively equal',34:'1.00577x v21 (modal-scored-bundle)'}
runs=[]
for p in (R/'results/runs').glob('*/manifest.json'):
 m=json.loads(p.read_text());v=json.loads((p.parent/'result.json').read_text());d=v.get('result')or{};runs.append({'candidate':Path(m['source']).name,'job':v['id'],'mode':v.get('mode'),'status':v.get('status'),'passed':d.get('passed'),'speedup':d.get('speedup'),'ci':d.get('speedup_ci'),'score_us':d.get('score_us'),'created_at':v.get('created_at'),'duplicate_of':v.get('duplicate_of')})
scored=[v for v in runs if v['mode']=='run' and v['passed'] and v['speedup']];scored.sort(key=lambda v:v['speedup'],reverse=True);best=scored[0]
board=json.loads((R/'results/scored-pass1/leaderboard-final.json').read_text());you=board.get('you') or {};quota=json.loads((R/'results/scored-pass1/quota-final.json').read_text());q=next(v['limits'] for v in quota['problems'] if v['name']=='muse_bf16_proj')
header=f'''# Open Kernel Challenge: Muse dense projections on H200

A reproducible record of our BF16 projection and FP32-router kernel experiments. Compute `x @ W[:N].T` under the challenge's numerical and fresh-input requirements.

**Best official scored entry: [{best['candidate']}](candidates/{best['candidate']}) — {best['speedup']:.4f}x SGLang**, normalized time **{best['score_us']/1000:.3f}ms** versus **4.831ms** baseline. Reported95% interval: **[{best['ci'][0]:.4f}, {best['ci'][1]:.4f}]**. Leaderboard snapshot: **rank {you.get('rank')} of {board.get('ranked')}**, team `iyerramji`. Rankings are provisional and change over time; snapshot taken at completion of this scored pass.

The official and Modal measurements below are different experiments and estimators. **Do not multiply Modal gains into an official score.** Differences within overlapping confidence intervals are inconclusive.

## Official scored results

All entries below passed official screening and correctness. These are fresh scored runs, separate from the earlier unscored checks. Exact sources/results/profiles are linked by job.

| Kernel | One-line description | Official speedup | Reported95% interval | Normalized time | Official job |
|---|---|---:|---|---:|---|
'''
for v in scored:
 n=int(v['candidate'].split('_')[0][1:]);header+=f"| [{v['candidate']}](candidates/{v['candidate']}) | {descriptions[n]} | {v['speedup']:.4f}x | {v['ci'][0]:.4f}–{v['ci'][1]:.4f} | {v['score_us']/1000:.3f}ms | [{v['job']}](results/runs/{v['job']}/result.json) |\n"
header+='''
## Modal development results — separate evidence

One H200, cold-L2 paired CUDA-graph timings. These include raw graph replay cost and do not reproduce the official overhead subtraction or confidence interval. Rows name their own control; they are not a common leaderboard.

| Kernel | Modal measurement | Interpretation |
|---|---|---|
'''
for n in [12,24,25,28,29,30,31,32,33,34]:header+=f"| v{n} | {modal[n]} | {'Whole workload diagnostic' if n in [12,24,25,30,33,34] else 'Target-only result'} |\n"
header+='''
## Does Modal predict the official result?

It identified useful target improvements, but not the final aggregate ranking. v33's Modal predecessor improved QKV128/256; official v33 scored1.1413x versus v24's1.1443x, with overlapping intervals. On official QKV256, the original-layout path in v21 was faster than the chosen Lt path. The new v34 combines only independently observed target winners; its official result appears above. Both environments use H200, but driver/library versions, replay overhead and timing estimators differ.

## Every kernel revision

Historical source docstrings are preserved byte-for-byte and may say “unmeasured.” This table and linked results describe current status. A dash means no result in that evidence class, not zero performance. Failed/slower/untested revisions are intentionally retained.

| Revision | What it changes | Best scored result | Earlier check result | Modal evidence |
|---|---|---:|---:|---|
'''
for p in sorted((R/'candidates').glob('v*.py'),key=lambda p:int(p.stem.split('_')[0][1:])):
 n=int(p.stem.split('_')[0][1:]);sr=[v for v in runs if v['candidate']==p.name and v['mode']=='run' and v['passed'] and v['speedup']];cr=[v for v in runs if v['candidate']==p.name and v['mode']!='run' and v['passed'] and v['speedup']]
 sv=f"{max(v['speedup'] for v in sr):.4f}x" if sr else '—';cv=f"{max(v['speedup'] for v in cr):.4f}x" if cr else '—'
 if n==2 and not cr:cv='0.7953x (archived starter filename)'
 header+=f"| [v{n}](candidates/{p.name}) | {descriptions[n]} | {sv} | {cv} | {modal.get(n,'—')} |\n"
header+='''
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
'''
(R/'README.md').write_text(header)
report=f"# Official scored pass\n\n{len(scored)} candidates scored; all passed. Best: {best['candidate']} at{best['speedup']:.6f}x, CI[{best['ci'][0]:.6f},{best['ci'][1]:.6f}]. Normalized{best['score_us']:.3f}us. Snapshot rank{you.get('rank')}/{board.get('ranked')}. Remaining scored attempts:{q['scored']['left']}; unscored:{q['checks']['left']}. No fabricated repeat: returned status/duplicate metadata are retained.\n\n"
report+='| Candidate | Speedup | Interval | Time us | Job |\n|---|---:|---|---:|---|\n'
for v in scored:report+=f"|{v['candidate']}|{v['speedup']:.6f}x|{v['ci']}|{v['score_us']:.3f}|[{v['job']}](../results/runs/{v['job']}/result.json)|\n"
report+='\nModal and official timing are separate. The official scorer subtracts replay overhead and uses its own paired estimator; Modal raw timings do not. The combined candidate was built after comparing official per-shape results, then validated locally on H200 before scored submission. Intervals overlap, so highest point estimate is selected provisionally, not claimed statistically superior.\n'
(R/'docs/SCORED_RESULTS.md').write_text(report)
(R/'results/scored-pass1/summary.json').write_text(json.dumps({'best':best,'scored':scored,'leaderboard_you':you,'quota':q},indent=2))
print(json.dumps({'best':best,'scored_count':len(scored)},indent=2))
