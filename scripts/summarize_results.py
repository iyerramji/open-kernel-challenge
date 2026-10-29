"""Summarize preserved check runs and snapshot every code version. No remote actions."""
import ast,csv,hashlib,json,shutil,argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--select-job');args=ap.parse_args()
P=Path(__file__).resolve().parents[1];runs=P/'results/runs';versions=P/'results/code-versions';versions.mkdir(exist_ok=True)
rows=[]
for f in runs.glob('*/manifest.json'):
 m=json.loads(f.read_text());d=json.loads((f.parent/'result.json').read_text());r=d.get('result') or {}
 # Add any available diagnostics/interface not copied by earlier archive helper versions.
 (f.parent/'returned-log-tail.txt').write_text(r.get('log_tail',''))
 interface=f.parent/'official-interface';interface.mkdir(exist_ok=True)
 for name in ['baseline.py','reference.py','checks.py','TASK.md','AGENTS.md']:shutil.copy2(P/'problems/muse_bf16_proj'/name,interface/name)
 m['files']={str(x.relative_to(f.parent)):hashlib.sha256(x.read_bytes()).hexdigest() for x in f.parent.rglob('*') if x.is_file() and x.name!='manifest.json'}
 f.write_text(json.dumps(m,indent=2)+'\n')
 for name,sha in m['files'].items():assert hashlib.sha256((f.parent/name).read_bytes()).hexdigest()==sha
 assert hashlib.sha256((f.parent/'submission.py').read_bytes()).hexdigest()==m['sha256']
 rows.append(dict(status=d.get('status'),duplicate_of=d.get('duplicate_of'),job=m['job'],candidate=Path(m['source']).name,sha256=m['sha256'],passed=m['passed'],speedup=m['speedup'],ci_low=(m.get('speedup_ci') or [None,None])[0],ci_high=(m.get('speedup_ci') or [None,None])[1],created_at=d['created_at'],finished_at=d.get('finished_at'),limits=d.get('limits')))
rows.sort(key=lambda x:x['created_at']);valid=[r for r in rows if r['passed'] and r['speedup']];nominal_best=max(valid,key=lambda x:x['speedup']);best=next((r for r in valid if r['job']==args.select_job),nominal_best)
if args.select_job:assert best['job']==args.select_job,'selected run must be a passing measured run'
# Never lose a previous working file when promoting a measured winner.
work=P/'racer/muse_bf16_proj/submission.py';old=work.read_bytes();oldsha=hashlib.sha256(old).hexdigest();prev=versions/('working-'+oldsha+'.py')
if not prev.exists():prev.write_bytes(old)
work.write_bytes((runs/best['job']/'submission.py').read_bytes())
assert hashlib.sha256(work.read_bytes()).hexdigest()==best['sha256']
inventory=[]
for f in sorted((P/'candidates').glob('v*.py')):
 raw=f.read_bytes();ast.parse(raw);sha=hashlib.sha256(raw).hexdigest();snap=versions/(f.stem+'-'+sha+'.py')
 if snap.exists():assert snap.read_bytes()==raw
 else:snap.write_bytes(raw)
 matches=[r for r in rows if r['sha256']==sha]
 inventory.append(dict(candidate=f.name,sha256=sha,snapshot=str(snap.relative_to(P)),official_runs=[r['job'] for r in matches],gpu_status='passed official check' if any(r['passed'] for r in matches) else ('failed official check' if matches else 'not GPU tested')))
(P/'results/code-versions/index.json').write_text(json.dumps(inventory,indent=2)+'\n')
(P/'results/candidate-sha256.json').write_text(json.dumps({r['candidate']:r['sha256'] for r in inventory},indent=2)+'\n')
(P/'results/runs/index.json').write_text(json.dumps(rows,indent=2)+'\n')
with (P/'results/official-run-summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='limits']);w.writeheader();w.writerows([{k:v for k,v in r.items() if k!='limits'} for r in rows])
md=['# Official H200 experiment results','',f"Selected measured kernel: **{best['candidate']} — {best['speedup']:.4f}× SGLang**, reported 95% interval **{best['ci_low']:.4f}–{best['ci_high']:.4f}×**. All runs are unscored checks; no leaderboard entry or final screening is implied.",'', '| Candidate | Run / exact code and diagnostics | Correctness | Speedup | Reported 95% interval |','|---|---|---|---|---|']
for r in rows:
 ci=f"{r['ci_low']:.4f}–{r['ci_high']:.4f}" if r['ci_low'] is not None else 'not available';speed=f"{r['speedup']:.4f}×" if r['speedup'] is not None else 'not available'
 md.append(f"| {r['candidate']} | [{r['job']}](runs/{r['job']}/manifest.json) | {'Passed' if r['passed'] else ('Duplicate: no new measurement' if r.get('status')=='duplicate' else 'Failed')} | {speed} | {ci} |")
md += ['', 'Every run folder contains exact tested submission.py, complete returned result.json, available CLI profile JSON/text/log, returned log tail, per-shape CSV, official interface snapshot and SHA256 manifest. Empty CLI logs remain preserved; server log_tail is saved separately. Local raw-file SHA256 and server content hash are distinct fields. All archived artifact hashes verified.','', '[All code-version snapshots](code-versions/index.json) include untested candidates with explicit status. CPU validation JSON remains alongside official result JSON. No candidate source version was overwritten; working submission promotion preserves the previous working bytes by hash.','',f"Working kernel: ../racer/muse_bf16_proj/submission.py — exact archived source for job {best['job']}.", '', f"Latest returned quota: {json.dumps(rows[-1]['limits'])}"]
(P/'results/README.md').write_text('\n'.join(md)+'\n')
# Replace previous status banners; preserve historical setup/review bodies as dated history.
status=f"> Current measured status: {sum(r.get('status')=='done' for r in rows)} executed unscored H200 checks and {sum(r.get('status')=='duplicate' for r in rows)} duplicate request archived in [results](../results/README.md). Best measured {best['candidate']}: {best['speedup']:.4f}× (95% interval {best['ci_low']:.4f}–{best['ci_high']:.4f}×). Initial setup/review notes below predate these runs; consult results for current GPU status. No scored submission.\n\n"
for name in ['EXECUTIVE_SUMMARY.md','VALIDATION.md','INDEPENDENT_REVIEW.md','BENCHMARKS.md']:
 f=P/'docs'/name;s=f.read_text();head,body=s.split('\n',1);body=body.lstrip()
 if body.startswith('> Current measured status:'):body=body.split('\n\n',1)[1]
 f.write_text(head+'\n\n'+status+body)
readme=P/'README.md';s=readme.read_text();start=s.find('\n\n')+2;end=s.find(' Read the ')
if start>=0 and end>start:s=s[:start]+f"{len(inventory)} `muse_bf16_proj` code versions are preserved. **Best official H200 check: {best['speedup']:.4f}× SGLang (95% interval {best['ci_low']:.4f}–{best['ci_high']:.4f}×).** All code and results are in [results](results/README.md)."+s[end:]
s=s.replace('three unscored checks have run',f"{len(rows)} unscored checks have run");readme.write_text(s)
(P/'racer/muse_bf16_proj/SETUP.md').write_text(f"Account: iyerramji; local label racer. User completed login/terms acceptance.\n\nWorking submission is exact code from official check {best['job']}: {best['candidate']}, correctness passed, speedup {best['speedup']:.6f}×, 95% interval [{best['ci_low']:.6f}, {best['ci_high']:.6f}].\n\nAll versions, failed/slower experiments, results and diagnostics are preserved under ../../results/. Original starter remains starter.py. No scored submission. Read TASK.md and AGENTS.md before further work.\n")
print(json.dumps(dict(best=best,nominal_best=nominal_best,runs=len(rows),versions=len(inventory),archive_hashes='verified'),indent=2))
