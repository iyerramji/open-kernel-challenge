import torch,sys,inspect,json,statistics
from pathlib import Path
sys.path[:0]=['/work/candidates','/work/problems/muse_bf16_proj']
import reference,v24_tiny_unpooler as control
import flashinfer.gemm as gemm
out=Path('/tmp/results');api=getattr(gemm,'mm_bf16',None)
(out/'cudnn-api.txt').write_text(str(inspect.signature(api))+'\n'+inspect.getsource(api) if api else 'mm_bf16 unavailable')
if api is None:raise RuntimeError('Installed FlashInfer does not export mm_bf16')
w=reference.generate_weights(5151);state=control.prepare(w);flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8);results=[]
for layer in ['pooler','qkv_gate','unpooler']:
 for m in [16,32,128,256]:
  record={'layer':layer,'m':m}
  try:
   d=reference.generate_input(m,layer,123);x=d['x'];n=d['config']['n'];wt=w[layer][:n].T
   def invoke():return api(x,wt,backend='cudnn',pdl=False)
   for _ in range(3):invoke()
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):y=invoke()
   base=torch.cuda.CUDAGraph()
   with torch.cuda.graph(base):by=control.kernel_fn(d,state)
   for seed in [124,125]:
    fresh=reference.generate_input(m,layer,seed);x.copy_(fresh['x']);g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(d,y,w);assert ok,msg
   samples=[]
   for rep in range(120):
    t={}
    for name,graph in ([('candidate',g),('control',base)] if rep%2==0 else [('control',base),('candidate',g)]):
     flush.zero_();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();graph.replay();b.record();b.synchronize();t[name]=a.elapsed_time(b)*1000
    samples.append(t)
   record.update(correct=True,samples=samples,ratio=statistics.median(t['candidate']/t['control'] for t in samples),candidate_us=statistics.median(t['candidate'] for t in samples),control_us=statistics.median(t['control'] for t in samples))
  except Exception as e:record['error']=str(e)
  results.append(record);(out/'cudnn.json').write_text(json.dumps(results,indent=2));print({k:v for k,v in record.items() if k!='samples'},flush=True)
