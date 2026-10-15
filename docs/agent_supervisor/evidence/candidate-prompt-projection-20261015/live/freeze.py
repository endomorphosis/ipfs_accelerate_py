"""Root-only activation after the exact runtime commit and both qualifications."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
A=Path(__file__).resolve().parent


def bind(path):
 p=Path(path);raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


assert len(sys.argv)==4,'expected reviewed commit, runtime qualification, reader qualification'
commit,qualification_path,reader_path=sys.argv[1:]
assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
r=json.loads((A/'recipe-draft.json').read_text());W=Path(r['cwd'])
q=json.loads(Path(qualification_path).read_text());rq=json.loads(Path(reader_path).read_text())
assert q['schema']=='candidate-projection-runtime-qualification@1'
assert q['passed'] is True and type(q['passed_cases']) is int and q['passed_cases']>0
assert q['failures']==q['errors']==q['skipped']==0 and q['provider_calls']==0
assert type(q['source_sha256']) is dict and q['source_sha256']
assert 'ipfs_accelerate_py/agent_supervisor/runtime/candidate_prompt_projection.py' in q['source_sha256']
for path,digest in q['source_sha256'].items():assert bind(W/path)['sha256']==digest
assert rq['schema']=='candidate-projection-reader-qualification@1' and rq['passed'] is True
assert rq['failures']==rq['errors']==rq['skipped']==0 and rq['provider_calls']==0
assert rq['old_closed_metrics_and_unknowns_identical'] is True and rq['runtime_modules_imported']==[]
for row in rq['files']:assert bind(row['path'])==row
assert bind(rq['tests_log']['path'])==rq['tests_log']
r['roots']['source']['commit']=commit
for source in r['roots'].values():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source['path'],text=True).strip()==source['commit']
 assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=source['path'],text=True)
subprocess.run(['git','ls-files','--error-unmatch','ipfs_accelerate_py/agent_supervisor/runtime/candidate_prompt_projection.py'],cwd=W,stdout=subprocess.DEVNULL,check=True)
# Draft hashes are exploratory. Freeze all source bindings only after qualification and clean commit.
r['host_source_sha256']={path:bind(path)['sha256'] for path in r['host_source_sha256']}
assert str(W/'ipfs_accelerate_py/agent_supervisor/runtime/candidate_prompt_projection.py') in r['host_source_sha256']
for row in r['input_bindings']:assert bind(row['path'])==row
previous=Path(r['previous_run']['cleanup']);prior=json.loads(previous.read_text())
assert prior['container_cleanup_observed'] is True and prior['remaining_container_ids']==[]
assert json.loads((previous.parent/'collect-result.json').read_text())['exit_code']==0
assert bind(A/'environment.json')['sha256']==r['environment_sha256']
assert bind(r['readiness']['path'])==r['readiness']
with (A/'qualification.json').open('xb') as f:f.write(Path(qualification_path).read_bytes())
upstream=json.loads((A/'reader/upstream-freeze.json').read_text())
files=upstream['files']+[bind(p) for p in sorted((A/'reader').glob('*.py'))]+[
 bind(A/'reader/upstream-freeze.json'),bind(Path(reader_path)),rq['tests_log']]
seen={row['path']:row for row in files}
reporting=dict(schema='candidate-projection-qualified-reporting-freeze@1',files=list(seen.values()),
 frozen_before_launch=True,reader_entrypoint=bind(A/'reader/read_projection.py'),
 reader_qualification=bind(Path(reader_path)),provider_calls=0,raw_content_exported=False,
 baseline_metrics_unchanged=True,independent_owner_replay_performed=False)
write(A/'reporting-freeze-v2.json',reporting)
r['status']='frozen_after_root_commit';write(A/'recipe.json',r)
write(A/'source-freeze.json',dict(schema='candidate-projection-source-freeze@1',
 root_authorized_source_freeze=True,source_commit=commit,worktree_clean=True,
 qualification=bind(A/'qualification.json'),reader_qualification=bind(Path(reader_path)),
 recipe=bind(A/'recipe.json'),runner=bind(A/'run.py'),reporting=bind(A/'reporting-freeze-v2.json'),
 launcher_adaptation=bind(A/'launcher-adaptation.json'),freeze_script=bind(Path(__file__)),
 prior_cleanup=bind(previous),provider_calls=0,
 observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print(json.dumps(dict(status='frozen',source_commit=commit,host_source_pins=len(r['host_source_sha256']),reporting_bindings=len(seen),provider_calls=0)))
