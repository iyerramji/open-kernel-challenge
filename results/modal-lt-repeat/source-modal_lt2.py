from pathlib import Path
import modal
ROOT=Path('/Users/ramjiiyer/git-clone/open-kernel-challenge')
app=modal.App('okc-h200-lt-search-v2')
image=modal.Image.from_registry('lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a')
for sub in ['candidates','problems/muse_bf16_proj']:
    image=image.add_local_dir(ROOT/sub,remote_path='/work/'+sub)
for sub in ['scripts/lt_search_v2.py','scripts/validate.py','scripts/bench.py','official/problem.json']:
    image=image.add_local_file(ROOT/sub,remote_path='/work/'+sub)
@app.function(image=image,gpu='H200',cpu=4,memory=16384,max_containers=1,timeout=1800,startup_timeout=600,scaledown_window=2,retries=0)
def run():
    import subprocess,time,json,importlib.metadata,traceback
    from pathlib import Path
    began=time.time();out=Path('/tmp/results');out.mkdir(exist_ok=True)
    info={'diagnostic_only':True,'commands':[],'packages':{}}
    for pkg in ['torch','triton','sglang','flashinfer-python']:
        try:info['packages'][pkg]=importlib.metadata.version(pkg)
        except Exception:pass
    info['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
    p=subprocess.run(['python','/work/scripts/lt_search_v2.py'],capture_output=True,text=True,timeout=500)
    (out/'probe.log').write_text(p.stdout+'\n'+p.stderr)
    info['commands'].append({'probe_exitcode':p.returncode})
    info['elapsed_seconds']=time.time()-began
    return {'metadata':info,'files':{p.name:p.read_text() for p in out.iterdir()}}
@app.local_entrypoint()
def main():
    import json
    result=run.remote()
    out=ROOT/'results/modal-lt2';out.mkdir(exist_ok=True)
    (out/'returned.json').write_text(json.dumps(result,indent=2))
    for name,content in result['files'].items():(out/name).write_text(content)
    print(json.dumps(result['metadata'],indent=2))
