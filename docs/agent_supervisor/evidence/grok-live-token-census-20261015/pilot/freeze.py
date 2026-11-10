"""Freeze the next diagnostic after qualified reporting and previous cleanup."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys
X=Path(__file__).resolve().parent
T=Path('/home/barberb/lift_coding/artifacts/declared-public-planning-20261015')
def bind(path):
 return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
r=json.loads((X/'recipe-draft.json').read_text())
previous=json.loads((T/'cleanup-observation.json').read_text())
assert previous['container_cleanup_observed'] is True and previous['remaining_container_ids']==[]
assert json.loads((T/'collect-result.json').read_text())['exit_code']==0
r['input_bindings'].append(bind(T/'cleanup-observation.json'))
r['roots']['source']['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r['cwd'],text=True).strip()
assert r['roots']['source']['commit']=='9d6ef76c8e2bb6916b68cad3563c6ec530b32736'
for root in r['roots'].values():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root['path'],text=True).strip()==root['commit']
 assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=root['path'],text=True)
for p,digest in r['host_source_sha256'].items(): assert bind(Path(p))['sha256']==digest
q=json.loads((X/'qualification.json').read_text())
assert q['schema']=='grok-coding-route-qualification@1' and q['passed_cases']==415
for rel,digest in q['source_sha256'].items():assert bind(Path(r['cwd'])/rel)['sha256']==digest
reporting=json.loads((X/'reporting-freeze-v2.json').read_text())
for row in reporting['files']:assert bind(Path(row['path']))==row
# Root supplies the reviewed, passing reader qualification explicitly.
rq=Path(sys.argv[1]);reader=json.loads(rq.read_text())
assert reader['passed'] is True and reader['failures']==reader['errors']==reader['skipped']==0
assert reader['provider_calls']==0
assert any(row==bind(rq) for row in reporting['files'])
r['status']='frozen_after_root_commit'
write(X/'recipe.json',r)
write(X/'source-freeze.json',dict(schema='grok-coding-route-source-freeze@1',
 root_authorized_source_freeze=True,source_commit=r['roots']['source']['commit'],
 qualification=bind(X/'qualification.json'),reader_qualification=bind(rq),
 recipe=bind(X/'recipe.json'),runner=bind(X/'run.py'),reporting=bind(X/'reporting-freeze-v2.json'),
 launcher_adaptation=bind(X/'reporting-launcher-adaptation.json'),
 observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),worktree_clean=True,
 prior_cleanup=bind(T/'cleanup-observation.json'),provider_calls=0))
print(json.dumps(dict(status='frozen',source_commit=r['roots']['source']['commit'],reporting_bindings=len(reporting['files']))))
