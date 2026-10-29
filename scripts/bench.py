"""Diagnostic paired cold-cache graph benchmark. Official okc check remains authoritative."""
import argparse, csv, hashlib, json, statistics, sys
from pathlib import Path
import torch
from validate import ROOT, PROBLEM, load
sys.path.insert(0,str(PROBLEM))
import reference

def capture(mod,data,state):
    for _ in range(3): mod.kernel_fn(data,state)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out=mod.kernel_fn(data,state)
    graph.replay()
    torch.cuda.synchronize()
    return graph,out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reps',type=int,default=30)
    args=ap.parse_args()
    assert args.reps>=2
    assert torch.cuda.is_available(),'Requires Linux CUDA H200 environment'
    mod=load(args.candidate.resolve());baseline=load(PROBLEM/'baseline.py')
    weights=reference.generate_weights(5151);state=mod.prepare(weights);base=baseline.prepare(weights)
    # 128 MiB exceeds H200 L2; executed before each event pair, outside measured interval.
    flush=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
    rows=[]
    for b in json.loads((ROOT/'official/problem.json').read_text())['benchmarks']:
        data=reference.generate_input(b['m'],b['layer'],b['seed'])
        cg,co=capture(mod,data,state);bg,bo=capture(baseline,data,base)
        for out in [co,bo]:
            ok,msg=reference.check_implementation(data,out,weights);assert ok,msg
        times={'candidate':[],'baseline':[]};ratios=[]
        for rep in range(args.reps):
            for name,g in ([('candidate',cg),('baseline',bg)] if rep%2==0 else [('baseline',bg),('candidate',cg)]):
                flush.zero_()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();g.replay();end.record();end.synchronize()
                times[name].append(start.elapsed_time(end)*1000)
            ratios.append(times['baseline'][-1]/times['candidate'][-1])
        rows.append(dict(m=b['m'],layer=b['layer'],load=b['load'],weight=b['weight'],
            candidate_us=statistics.median(times['candidate']),baseline_us=statistics.median(times['baseline']),
            paired_speedup=statistics.median(ratios),status='local diagnostic; raw graph cost included'))
        print(rows[-1],flush=True)
    with args.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    args.output.with_suffix('.metadata.json').write_text(json.dumps(dict(
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),reps=args.reps,
        official_equivalence=False,note='No official fixed-cost subtraction, confidence interval or paired two-repetition estimator. Use okc check for score.'),indent=2))

if __name__=='__main__':main()
