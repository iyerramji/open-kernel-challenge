"""Export our code and results; omit credentials, private notes and downloaded source materials."""
import argparse,hashlib,json,re,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);args=p.parse_args()
root=Path(__file__).resolve().parents[1];dest=args.destination.resolve();dest.mkdir(parents=True,exist_ok=True)
excluded_names={'billing.json','containers.json','installed-cublasLt-header-excerpt.txt','recommendation.txt'}
allow=[]
for name in ['README.md','LICENSE','NOTICE','references.lock.json']:
 if (root/name).is_file():allow.append(root/name)
for pattern in ['candidates/v*.py','scripts/*.py','docs/*.md','results/**/*.json','results/**/*.csv','results/**/*.py','results/**/*.log','results/**/*.txt','results/**/*.stderr','racer/muse_bf16_proj/*.py']:
 for f in root.glob(pattern):
  rel=f.relative_to(root)
  if not f.is_file() or any(x in rel.parts for x in ['__pycache__','official-interface']) or f.name in excluded_names:continue
  if f.name=='plan.json' or f.name.endswith('-current.json'):continue
  allow.append(f)
manifest={}
for f in sorted(set(allow)):
 rel=f.relative_to(root);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);data=f.read_bytes();transform=None
 if rel.parts[0]=='scripts':
  text=data.decode();text=text.replace("Path(__file__).resolve().parents[1]","Path(__file__).resolve().parents[1]").replace('Path(__file__).resolve().parents[1]','Path(__file__).resolve().parents[1]')
  if text.encode()!=data:transform='Portable active runner project root; archived copies unchanged';data=text.encode()
 target.write_bytes(data);manifest[str(rel)]={'sha256':hashlib.sha256(data).hexdigest(),'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'transformation':transform}
# Regenerate official implementation files with okc init; publish benchmark metadata only.
p=dest/'official/problem.json';p.parent.mkdir(exist_ok=True);meta=json.loads((root/'official/problem.json').read_text());p.write_text(json.dumps({'name':'muse_bf16_proj','benchmarks':meta['benchmarks'],'publication_note':'Benchmark metadata only. Run okc init to obtain official reference and checks.'},indent=2))
licenses=dest/'third_party/licenses';licenses.mkdir(parents=True,exist_ok=True)
shutil.copy(root/'repos/sglang/LICENSE',licenses/'SGLang-Apache-2.0.txt')
(dest/'.gitignore').write_text('.venv/\n__pycache__/\n*.pyc\n.DS_Store\nrepos/\ntools/\nprofiles/\nproblems/muse_bf16_proj/\n.env\n.env.*\n')
(dest/'PUBLICATION.md').write_text('''# Publication scope\n\nAll candidate source revisions and our runner scripts are included. Exact archived code snapshots remain unchanged. Active runners use a portable project root. Official and Modal results are separate evidence classes.\n\nDownloaded challenge interface/source, website/terms copies, third-party checkouts/wheels, credentials, billing/container snapshots and private handoff notes remain local. Obtain official reference/checks with `okc init muse_bf16_proj problems/muse_bf16_proj` after your own login/terms acceptance. `official/problem.json` here contains benchmark metadata only.\n\nHistorical local manifests may name omitted local-only artifacts. They are preserved as provenance, not as a claim that omitted files are published. The export manifest records exact included source/artifact hashes. Candidate comments are historical; the README/results tables provide current validation status.\n''')
(dest/'publication-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for rel,m in manifest.items():assert hashlib.sha256((dest/rel).read_bytes()).hexdigest()==m['sha256']
print(json.dumps({'destination':str(dest),'files':len(manifest),'bytes':sum((dest/x).stat().st_size for x in manifest)},indent=2))
