import sys,subprocess,json,importlib.util,shutil
from pathlib import Path
if len(sys.argv)>1:
 import torch
 sys.path.insert(0,'/work/problems/muse_bf16_proj');import reference
 name=sys.argv[1];spec=importlib.util.spec_from_file_location(name,'/work/candidates/'+name+'.py');mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 weights=reference.generate_weights(5151);state=mod.prepare(weights);data=reference.generate_input(1,'pooler',123)
 for _ in range(3):mod.kernel_fn(data,state)
 torch.cuda.synchronize();flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8);flush.zero_();torch.cuda.synchronize()
 torch.cuda.cudart().cudaProfilerStart();y=mod.kernel_fn(data,state);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
 ok,msg=reference.check_implementation(data,y,weights);assert ok,msg
else:
 out=Path('/tmp/results');ncu=shutil.which('ncu');records=[]
 for flag,label in [('--version','version'),('--list-sections','sections')]:
  p=subprocess.run([ncu,flag],capture_output=True,text=True,timeout=30);(out/(label+'.txt')).write_text(p.stdout+p.stderr)
 metrics='dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__waves_per_multiprocessor,launch__registers_per_thread'
 p=subprocess.run([ncu,'--query-metrics-mode','suffix','--metrics',metrics],capture_output=True,text=True,timeout=30);(out/'metric-query.txt').write_text(p.stdout+p.stderr)
 for name in ['v24_tiny_unpooler','v27_raw_pooler_control','v26_codec_pooler']:
  cmd=[ncu,'--profile-from-start','off','--replay-mode','application','--cache-control','none','--clock-control','none','--csv','--metrics',metrics,'python','/work/scripts/counter_probe.py',name]
  try:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=240);log=p.stdout+'\nSTDERR\n'+p.stderr;rc=p.returncode
  except subprocess.TimeoutExpired as e:log=str(e);rc=124
  (out/(name+'-counters.csv')).write_text(log);records.append({'candidate':name,'command':cmd,'returncode':rc});print(name,rc,log[-1500:],flush=True)
  if 'ERR_NVGPUCTRPERM' in log or 'permission to access' in log.lower():
   records.append({'stopping_reason':'Hardware counter permission denied; no retries or permission changes attempted.'});break
 (out/'commands.json').write_text(json.dumps(records,indent=2))
