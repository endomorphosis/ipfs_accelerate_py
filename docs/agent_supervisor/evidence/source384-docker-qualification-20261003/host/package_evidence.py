"""Package declared public receipts; exclude archives, models and mutable databases."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path(__file__).parent
A=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
OUT=A/'docs/agent_supervisor/evidence/source384-docker-qualification-20261003'
OUT.mkdir(parents=True,exist_ok=False)
def copy(src,name):
 dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
def tree(src,name):
 for file in sorted(src.rglob('*')):
  if file.is_file() and not file.is_symlink():copy(file,str(Path(name)/file.relative_to(src)))
for name in ['8gib-sources','final-sources','docs','docs-final']:
 tree(BASE/name,name)
for name in ['docker-01','12gib','retry-12gib']:
 directory=BASE/name
 if directory.exists():
  for file in sorted(directory.rglob('*')):
   if file.is_file() and not file.is_symlink() and file.suffix in {'.json','.log','.xml','.stdout','.stderr','.py','.md'}:
    copy(file,str(Path(name)/file.relative_to(directory)))
for file in BASE.iterdir():
 if file.is_file() and file.suffix in {'.json','.log','.xml','.py'}:copy(file,'host/'+file.name)
for name in ['bundle','bundle-final','bundle-final-retry']:
 file=BASE/name/'manifest.json'
 if file.exists():copy(file,name+'/manifest.json')
for name in ['terminal_deployment.py','terminal_source384_qualification.py','test_terminal_source384_qualification.py',
             'benchmark_resource_profile.py','SOURCE384_HARBOR_ASSETS.md','SOURCE384_DOCKER_QUALIFICATION.md']:
 copy(A/'benchmarks/agent_supervisor/container_coding'/name,'current/'+name)
for name in ['ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py',
             'benchmarks/agent_supervisor/container_coding/test_terminal_source384_context.py']:
 copy(A/name,'real-consumer/'+Path(name).name)
consumer=BASE.parent/'terminal-source384-context-20261003'
for name in ['qualified-complete-population.xml','qualified-complete-population.log','qualified-complete-population-command.json',
             'public-inventory-static-audit.json','audit_public_inventory.py']:
 copy(consumer/name,'real-consumer/'+name)
stability=BASE.parent/'empty-supervisor-stability-20261003'
for file in stability.iterdir():
 if file.is_file() and file.suffix in {'.json','.log','.xml','.py'}:copy(file,'native-stability/'+file.name)
for file in sorted((stability/'final-sources').rglob('*')):
 if file.is_file():copy(file,'native-stability/final-sources/'+str(file.relative_to(stability/'final-sources')))
qualification=json.loads((BASE/'qualification-summary.json').read_text())
(OUT/'qualification.json').write_text(json.dumps(qualification,indent=2,sort_keys=True)+'\n')
rows=[]
for file in sorted(OUT.rglob('*')):
 if file.is_file(): rows.append(dict(path=file.relative_to(OUT).as_posix(),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
(OUT/'manifest.json').write_text(json.dumps(dict(schema='closed-source384-docker-evidence@1',files=rows),indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(OUT),files=len(rows),bytes=sum(r['bytes'] for r in rows),manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())))
