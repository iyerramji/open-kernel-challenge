"""Archive result, exact source, profiles/log and hashes; never overwrite a changed source snapshot."""
import argparse, csv, hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('source',type=Path);a=p.parse_args()
d=json.loads(a.result.read_text());job=d['id'];out=ROOT/'results/runs'/job;out.mkdir(parents=True,exist_ok=True)
raw=a.source.read_bytes();sha=hashlib.sha256(raw).hexdigest();dest=out/'submission.py'
if dest.exists():assert dest.read_bytes()==raw,'refusing to replace original code snapshot'
else:dest.write_bytes(raw)
shutil.copy2(a.result,out/'result.json')
for group in ['profile','output']:
 for key,source in (d.get(group) or {}).items():
  f=Path(source)
  if f.is_file():shutil.copy2(f,out/f.name)
r=d.get('result') or {}
(out/'returned-log-tail.txt').write_text(r.get('log_tail',''))
interface=out/'official-interface';interface.mkdir(exist_ok=True)
for name in ['baseline.py','reference.py','checks.py','TASK.md','AGENTS.md']:
 shutil.copy2(ROOT/'problems/muse_bf16_proj'/name,interface/name)
manifest=dict(job=job,mode=d.get('mode'),status=d.get('status'),source=str(a.source.resolve()),sha256=sha,server_content_hash=d.get('content_hash'),passed=r.get('passed'),speedup=r.get('speedup'),speedup_ci=r.get('speedup_ci'),error=r.get('error'),created_at=d.get('created_at'),finished_at=d.get('finished_at'),gpu=r.get('gpu'),files={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
rows=[]
for rec in r.get('records',[]):
 if rec.get('event')=='benchmark':
  c=rec['case'];rows.append(dict(m=c['m'],layer=c['layer'],load=c.get('load'),baseline_us=rec.get('baseline_ns',0)/1000,candidate_us=rec.get('median_ns',0)/1000,paired_speedup=rec.get('paired_ratio')))
if rows:
 with (out/'shapes.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
manifest['files']={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
index=[]
for f in sorted((ROOT/'results/runs').glob('*/manifest.json')):
 v=json.loads(f.read_text());index.append({k:v.get(k) for k in ['job','source','sha256','mode','status','passed','speedup','speedup_ci','error']})
(ROOT/'results/runs/index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
