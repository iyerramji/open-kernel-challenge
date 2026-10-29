import ctypes as C,glob,sys,json,statistics
from pathlib import Path
import torch
sys.path.insert(0,'/work/problems/muse_bf16_proj');import reference
paths=glob.glob('/usr/local/cuda*/**/libcublasLt.so*',recursive=True)+glob.glob('/usr/local/lib/python*/dist-packages/nvidia/**/libcublasLt.so*',recursive=True)+glob.glob('/usr/local/lib/python*/site-packages/nvidia/**/libcublasLt.so*',recursive=True)
lib=C.CDLL(paths[0] if paths else 'libcublasLt.so.13')
P=C.c_void_p;I=C.c_int;Z=C.c_size_t;U=C.c_uint64
sigs={'cublasLtCreate':[C.POINTER(P)],'cublasLtMatmulDescCreate':[C.POINTER(P),I,I],'cublasLtMatrixLayoutCreate':[C.POINTER(P),I,U,U,C.c_int64],'cublasLtMatmulPreferenceCreate':[C.POINTER(P)],'cublasLtMatmulPreferenceSetAttribute':[P,I,P,Z],'cublasLtMatmulAlgoGetHeuristic':[P,P,P,P,P,P,P,I,P,C.POINTER(I)],'cublasLtMatmul':[P,P,P,P,P,P,P,P,P,P,P,P,P,P,Z,P]}
for n,a in sigs.items():getattr(lib,n).argtypes=a;getattr(lib,n).restype=I
def call(n,*a):
 rc=getattr(lib,n)(*a)
 if rc:raise RuntimeError((n,rc))
h=P();call('cublasLtCreate',C.byref(h));w=reference.generate_weights(5151);wt=w['pooler'].T.contiguous();results=[];workspace=torch.empty(32*1024*1024,device='cuda',dtype=torch.uint8);flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
for m in [128,256]:
 d=reference.generate_input(m,'pooler',123);x=d['x'];y=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16)
 desc=P();call('cublasLtMatmulDescCreate',C.byref(desc),68,0)
 layouts=[]
 for rows,cols,ld in [(4096,8192,4096),(8192,m,8192),(4096,m,4096)]:
  q=P();call('cublasLtMatrixLayoutCreate',C.byref(q),14,rows,cols,ld);layouts.append(q)
 a,b,c=layouts;pref=P();call('cublasLtMatmulPreferenceCreate',C.byref(pref));cap=Z(workspace.numel());call('cublasLtMatmulPreferenceSetAttribute',pref,1,C.byref(cap),C.sizeof(cap))
 buf=C.create_string_buffer(96*8);count=I();call('cublasLtMatmulAlgoGetHeuristic',h,desc,a,b,c,c,pref,8,buf,C.byref(count));alpha=C.c_float(1);beta=C.c_float(0)
 for j in range(count.value):
  record={'m':m,'heuristic':j}
  try:
   algo=C.cast(C.byref(buf,j*96),P)
   def invoke():call('cublasLtMatmul',h,desc,C.byref(alpha),P(wt.data_ptr()),a,P(x.data_ptr()),b,C.byref(beta),P(y.data_ptr()),c,P(y.data_ptr()),c,algo,P(workspace.data_ptr()),workspace.numel(),P(torch.cuda.current_stream().cuda_stream))
   invoke();torch.cuda.synchronize();ok,msg=reference.check_implementation(d,y,w);assert ok,msg
   for _ in range(3):invoke()
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):invoke()
   base=torch.cuda.CUDAGraph()
   with torch.cuda.graph(base):by=torch.mm(x,wt)
   samples=[]
   for rep in range(30):
    times={}
    for name,graph in ([('candidate',g),('control',base)] if rep%2==0 else [('control',base),('candidate',g)]):
     flush.zero_();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();times[name]=start.elapsed_time(end)*1000
    samples.append(times)
   record.update(correct=True,samples=samples,candidate_us=statistics.median(v['candidate'] for v in samples),control_us=statistics.median(v['control'] for v in samples),algorithm_bytes=bytes(buf[j*96:j*96+64]).hex())
  except Exception as e:record['error']=str(e)
  results.append(record);print(json.dumps({k:v for k,v in record.items() if k!='samples'}),flush=True)
Path('/tmp/results/lt-search.json').write_text(json.dumps(results,indent=2))
