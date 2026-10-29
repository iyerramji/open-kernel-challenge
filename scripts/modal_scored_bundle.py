from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
app=modal.App('okc-scored-winner-bundle')
image=modal.Image.from_registry('lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a')
for sub in ['candidates','problems/muse_bf16_proj']:
    image=image.add_local_dir(ROOT/sub,remote_path='/work/'+sub)
for sub in ['scripts/validate.py','scripts/scored_bundle_repeat.py','scripts/bench.py','official/problem.json']:
    image=image.add_local_file(ROOT/sub,remote_path='/work/'+sub)
@app.function(image=image,gpu='H200',cpu=4,memory=16384,max_containers=1,timeout=1800,startup_timeout=600,scaledown_window=2,retries=0)
def run():
    import subprocess,time,json,importlib.metadata,traceback
    from pathlib import Path
    import os
    os.environ['MAX_JOBS']='2';os.environ['TORCH_CUDA_ARCH_LIST']='9.0'
    began=time.time();out=Path('/tmp/results');out.mkdir(exist_ok=True)
    info={'diagnostic_only':True,'commands':[],'packages':{}}
    for pkg in ['torch','triton','sglang','flashinfer-python']:
        try:info['packages'][pkg]=importlib.metadata.version(pkg)
        except Exception:pass
    info['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
    for name in ['v34_official_bundle','scored_bundle_repeat']:
        for mode in (['validate'] if name=='v34_official_bundle' else ['bench']):
            if time.time()-began>1400:break
            if mode=='validate':
                cmd=['python','/work/scripts/validate.py','--official','--device','cuda','--candidate',name,'--output',str(out/(name+'-validation.json'))]
            else:
                cmd=['python','/work/scripts/'+name+'.py']
            print('START',name,mode,flush=True)
            try:
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
                text=p.stdout+'\nSTDERR\n'+p.stderr;rc=p.returncode
            except subprocess.TimeoutExpired as e:
                text=str(e);rc=124
            (out/(name+'-'+mode+'.log')).write_text(text)
            info['commands'].append({'candidate':name,'mode':mode,'command':cmd,'returncode':rc})
            print('END',name,mode,rc,text[-1500:],flush=True)
            if rc:break
    info['elapsed_seconds']=time.time()-began
    return {'metadata':info,'files':{p.name:p.read_text() for p in out.iterdir()}}
@app.local_entrypoint()
def main():
    import json
    result=run.remote()
    out=ROOT/'results/modal-scored-bundle';out.mkdir(exist_ok=True)
    (out/'returned.json').write_text(json.dumps(result,indent=2))
    for name,content in result['files'].items():(out/name).write_text(content)
    print(json.dumps(result['metadata'],indent=2))
