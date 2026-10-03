"""Package this exact public Docker generation, without weights or mutable databases."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path(__file__).parent
A=Path('/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002')
OUT=A/'docs/agent_supervisor/evidence/source384-docker-reconstruction-20261003'
OUT.mkdir(parents=True,exist_ok=False)
def copy(src,name):
 dst=OUT/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
for name in ('docker-01','frozen-sources','component-sources','docs-final'):
 for file in sorted((BASE/name).rglob('*')):
  if file.is_file() and not file.is_symlink() and file.suffix in {'.json','.log','.stdout','.stderr','.py','.md'}:
   copy(file,str(Path(name)/file.relative_to(BASE/name)))
for file in BASE.iterdir():
 if file.is_file() and file.suffix in {'.json','.log','.py','.xml'}:copy(file,'host/'+file.name)
copy(BASE/'bundle/manifest.json','bundle/manifest.json')
consumer=BASE.parent/'terminal-source384-context-20261003'
consumer_prefix=json.loads((BASE/'consumer-test-selection.json').read_text())['prefix']
for suffix in ('.xml','.log','-command.json','-d-pins.json'):
 copy(consumer/(consumer_prefix+suffix),'consumer/'+consumer_prefix+suffix)
for name in ('terminal_deployment.py','terminal_source384_qualification.py','benchmark_resource_profile.py',
             'test_terminal_source384_qualification.py','SOURCE384_DOCKER_QUALIFICATION.md'):
 copy(A/'benchmarks/agent_supervisor/container_coding'/name,'current/'+name)
qualification=json.loads((BASE/'qualification-summary.json').read_text())
(OUT/'qualification.json').write_text(json.dumps(qualification,sort_keys=True,indent=2)+'\n')
rows=[]
for file in sorted(OUT.rglob('*')):
 if file.is_file():rows.append(dict(path=file.relative_to(OUT).as_posix(),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
(OUT/'manifest.json').write_text(json.dumps(dict(schema='closed-source384-docker-reconstruction-evidence@1',files=rows),sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(path=str(OUT),files=len(rows),bytes=sum(row['bytes'] for row in rows),manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())))
