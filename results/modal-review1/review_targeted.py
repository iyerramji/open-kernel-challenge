import torch,sys,json,statistics,importlib,traceback
from pathlib import Path
sys.path[:0]=['/work/candidates','/work/problems/muse_bf16_proj'];import reference
out=Path('/tmp/results');records=[]
def save(): (out/'review-targeted.json').write_text(json.dumps(records,indent=2))
def capture(mod,d,state):
 for _ in range(3):mod.kernel_fn(d,state)
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):y=mod.kernel_fn(d,state)
 return g,y
control=importlib.import_module('v30_qkv_lt')
# Exact selected Lt branches, independent weights, outliers and fresh graph contents.
for seed in [5152,6101]:
 w=reference.generate_weights(seed);state=control.prepare(w)
 for m in [128,256]:
  d=reference.generate_input(m,'qkv_gate',seed,dist='outliers');g,y=capture(control,d,state)
  for inp in [seed+1,seed+2]:
   fresh=reference.generate_input(m,'qkv_gate',inp,dist='outliers');d['x'].copy_(fresh['x']);g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(d,y,w);assert ok,msg
  records.append({'candidate':'v30_qkv_lt','weight_seed':seed,'m':m,'dist':'outliers','fresh_graph_pass':True});save()
for name in ['v31_router_native','v32_router_fma']:
 try:
  mod=importlib.import_module(name);w=reference.generate_weights(5152);state=mod.prepare(w);base=control.prepare(w)
  for m in [1,2,6,8]:
   d=reference.generate_input(m,'router',745,dist='outliers');g,y=capture(mod,d,state);bg,by=capture(control,d,base)
   for seed in [746,747]:
    fresh=reference.generate_input(m,'router',seed,dist='outliers');d['x'].copy_(fresh['x'])
    for gr,yy in [(g,y),(bg,by)]:gr.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(d,yy,w);assert ok,msg
   flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8);rounds=[]
   for rnd in range(2):
    samples=[]
    for rep in range(120):
     t={}
     for key,gr in ([('candidate',g),('control',bg)] if (rep+rnd)%2==0 else [('control',bg),('candidate',g)]):
      flush.zero_();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();gr.replay();b.record();b.synchronize();t[key]=a.elapsed_time(b)*1000
     samples.append(t)
    rounds.append({'samples':samples,'ratio':statistics.median(t['candidate']/t['control'] for t in samples),'candidate_us':statistics.median(t['candidate'] for t in samples),'control_us':statistics.median(t['control'] for t in samples)})
   records.append({'candidate':name,'m':m,'weight_seed':5152,'outlier_fresh_graph_pass':True,'rounds':rounds});save();print(name,m,[v['ratio'] for v in rounds],flush=True)
 except Exception:records.append({'candidate':name,'error':traceback.format_exc()});save()
