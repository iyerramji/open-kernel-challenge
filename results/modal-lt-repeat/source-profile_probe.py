import sys,json,shutil,subprocess,importlib.util
from pathlib import Path
import torch
sys.path.insert(0,'/work/problems/muse_bf16_proj');import reference
spec=importlib.util.spec_from_file_location('v24','/work/candidates/v24_tiny_unpooler.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
w=reference.generate_weights(5151);state=mod.prepare(w);out={'compression':[],'profiles':[],'tools':{k:shutil.which(k) for k in ['ncu','nsys','nvcc']}}
for layer in ['pooler','qkv_gate']:
 bits=w[layer].view(torch.int16).to(torch.int32)&65535
 for block in [32,128,256]:
  u=bits.reshape(-1,block);h=(u>>8)&127;base=h.min(1).values;fits=(h.max(1).values-base)<8
  reconstructed=(u&255)|((base[:,None]+(h-base[:,None]))<<8)|(u&32768)
  assert torch.equal(reconstructed,u)
  common=int(fits.sum());total=fits.numel();compressed=common*(block*12//8+2)+(total-common)*(block*2+2)
  out['compression'].append(dict(layer=layer,block=block,common_blocks=common,total_blocks=total,estimated_bytes=compressed,raw_bytes=bits.numel()*2,saving_fraction=1-compressed/(bits.numel()*2),note='12-bit common block plus two-byte metadata; raw-block escape; roundtrip algebra exact, packed implementation not built'))
for m in [1,8,128,256]:
 d=reference.generate_input(m,'pooler',200+m)
 for _ in range(3):mod.kernel_fn(d,state)
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
  mod.kernel_fn(d,state);torch.cuda.synchronize()
 p.export_chrome_trace('/tmp/results/pooler'+str(m)+'.trace.json')
 out['profiles'].append({'m':m,'table':p.key_averages().table(sort_by='self_cuda_time_total',row_limit=20)})
out['cuda_headers']=subprocess.run(['bash','-c','find /usr/local/cuda /usr/local/lib/python3.12 /usr/local/lib/python3.11 -name cublasLt.h 2>/dev/null | head -5'],capture_output=True,text=True).stdout
Path('/tmp/results/probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
