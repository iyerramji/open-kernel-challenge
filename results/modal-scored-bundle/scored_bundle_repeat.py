import sys,json,statistics
from pathlib import Path
import torch
sys.path[:0]=['/work/candidates','/work/problems/muse_bf16_proj']
import reference,v34_official_bundle as candidate,v21_safe_bundle as control
w=reference.generate_weights(6102);s=candidate.prepare(w);c=control.prepare(w)
Path('/tmp/results/lt-metadata.json').write_text(json.dumps(s['lt_metadata'],indent=2))
results=[]
for spec in json.loads(Path('/work/official/problem.json').read_text())['benchmarks']:
 d=reference.generate_input(spec['m'],spec['layer'],123,dist='outliers');x=d['x'];graphs={};outputs={}
 for name,fn,state in [('candidate',candidate.kernel_fn,s),('control',control.kernel_fn,c)]:
  for _ in range(3):fn(d,state)
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):y=fn(d,state)
  graphs[name]=g;outputs[name]=y
 for seed in [124,125,126]:
  fresh=reference.generate_input(spec['m'],spec['layer'],seed,dist='outliers');x.copy_(fresh['x'])
  for name,g in graphs.items():
   g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(d,outputs[name],w);assert ok,(name,msg)
 flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8);rounds=[]
 for rnd in range(1):
  samples=[]
  for rep in range(120):
   row={}
   for name in (['candidate','control'] if (rep+rnd)%2==0 else ['control','candidate']):
    flush.zero_();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graphs[name].replay();b.record();b.synchronize();row[name]=a.elapsed_time(b)*1000
   samples.append(row)
  rounds.append({'samples':samples,'ratio':statistics.median(t['candidate']/t['control'] for t in samples),'candidate_us':statistics.median(t['candidate'] for t in samples),'control_us':statistics.median(t['control'] for t in samples)})
 result={'rounds':rounds,'gate_pass':all(t['ratio']<=1.02 for t in rounds),'fresh_graphs_pass':True}
 result.update(spec);results.append(result)
 Path('/tmp/results/gate.json').write_text(json.dumps(results,indent=2));print(json.dumps({**result,'rounds':[{k:v for k,v in t.items() if k!='samples'} for t in rounds]}))
