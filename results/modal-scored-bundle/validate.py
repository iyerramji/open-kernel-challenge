"""Local checks; CUDA mode executes the real candidate kernels, CPU mode checks fallbacks."""
import argparse, ast, importlib.util, json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = ROOT / 'problems/muse_bf16_proj'
sys.path.insert(0, str(PROBLEM))
import reference

CASES = [(1,'qkv_gate','normal'), (7,'pooler','normal'), (33,'o','normal'),
         (130,'unpooler','normal'), (5,'unpooler','normal'), (8,'router','normal'),
         (9,'router','outliers'), (512,'qkv_gate','outliers'), (1000,'unpooler','outliers')]

def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod

def static_check(path):
    text = path.read_text()
    assert text.splitlines()[0] == '# okc: problem=muse_bf16_proj'
    tree = ast.parse(text)
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {'prepare', 'kernel_fn'} <= funcs
    forbidden = {'subprocess','os','sys','inspect','socket','requests','reference','checks'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(x.name.split('.')[0] in forbidden for x in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or '').split('.')[0] not in forbidden
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {'item','cpu','synchronize'}
    return 'interface/import/host-sync static checks passed (not the official screening)'

def check(mod, weights, data):
    state = mod.prepare(weights)
    before = {k:v.clone() for k,v in weights.items()}
    with torch.no_grad(): out = mod.kernel_fn(data, state)
    ok, msg = reference.check_implementation(data, out, weights)
    assert ok, msg
    assert all(torch.equal(weights[k],v) for k,v in before.items()), 'weights mutated'
    return msg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    ap.add_argument('--official', action='store_true', help='full official dimensions; GPU strongly preferred')
    ap.add_argument('--candidate', default='all')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(4)
    paths = sorted((ROOT/'candidates').glob('v*.py'))
    if args.candidate != 'all': paths = [p for p in paths if p.stem == args.candidate]
    assert paths, 'no candidate selected'
    records = []
    for path in paths:
        static_check(path)
        mod = load(path)
        if args.official:
            weights = reference.generate_weights(reference.WEIGHTS_SEED, args.device)
            state = mod.prepare(weights)
            cases = [(m,l,d) for m,l,d in CASES]
            cases += [(b['m'],b['layer'],'normal') for b in json.loads((ROOT/'official/problem.json').read_text())['benchmarks']]
            # Nearby M and fresh seeds check dispatch boundaries without specializing to scored values.
            cases += [(m,l,'outliers') for m in [2,6,10,17,31,129,257] for l in reference.LAYERS]
            for i,(m,layer,dist) in enumerate(cases):
                data = reference.generate_input(m,layer,7000+i,dist=dist,device=args.device)
                out = mod.kernel_fn(data,state)
                ok,msg = reference.check_implementation(data,out,weights)
                assert ok, (path.name,m,layer,msg)
                records.append(dict(candidate=path.stem,device=args.device,m=m,layer=layer,dist=dist,check=msg,mode='official dimensions'))
                print(path.stem,m,layer,'PASS',flush=True)
            if args.device == 'cuda':
                # Every scored shape must capture and replay; fresh tensor contents after capture.
                for b in json.loads((ROOT/'official/problem.json').read_text())['benchmarks']:
                    data = reference.generate_input(b['m'],b['layer'],12001,device='cuda')
                    for _ in range(3): mod.kernel_fn(data,state)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): out = mod.kernel_fn(data,state)
                    fresh = reference.generate_input(b['m'],b['layer'],12002,device='cuda')
                    data['x'].copy_(fresh['x']); graph.replay(); torch.cuda.synchronize()
                    ok,msg = reference.check_implementation(data,out,weights)
                    assert ok, ('graph fresh-input',path.name,b,msg)
                records.append(dict(candidate=path.stem,check='30 scored-shape fresh-input CUDA graph replays passed'))
        else:
            # CPU smoke tests intentionally use compact dimensions, preserving dtype and interface.
            for m in [1,5,7,8,9,16,32,33,128,130,256,512,1000]:
                gen = torch.Generator(device=args.device).manual_seed(9000+m)
                weights = {l:torch.randn((65,257),generator=gen,device=args.device).to(torch.float32 if l=='router' else torch.bfloat16) for l in reference.LAYERS}
                for layer in weights:
                    x = torch.randn((m,257),generator=gen,device=args.device).bfloat16()
                    if m>=8:
                        x[3].zero_();x[3,32]=64
                    data = {'x':x,'config':dict(m=m,n=33 if layer=='unpooler' and m<=8 else 65,k=257,layer=layer)}
                    msg = check(mod,weights,data)
                    records.append(dict(candidate=path.stem,device=args.device,m=m,layer=layer,check=msg,mode='compact CPU fallback'))
            print(path.stem,'65 compact fallback cases + static checks PASS',flush=True)
    # Independently validate the allowed router representation, not hidden by CPU fallback.
    gen = torch.Generator().manual_seed(431)
    w = torch.randn((256,8192),generator=gen)
    hi = w.bfloat16();lo=(w-hi.float()).bfloat16()
    x=torch.randn((9,8192),generator=gen).bfloat16();x[0].zero_();x[0,37]=32
    data={'x':x,'config':dict(m=9,n=256,k=8192,layer='router')}
    out=(x.float()@hi.float().T)+(x.float()@lo.float().T)
    ok,msg=reference.check_implementation(data,out,{'router':w});assert ok,msg
    records.append(dict(check='CPU router BF16 hi/lo algebra: '+msg))
    result={'device':args.device,'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu_kernels_executed':args.device=='cuda','records':records,'count':len(records)}
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__ == '__main__': main()
