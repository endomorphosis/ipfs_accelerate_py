"""Bounded one-attempt source-indexed pilot; no hidden input inspection."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile,time
B=Path(__file__).resolve().parent;W=B.parent.parent
A=W/'.worktrees/ir-release-accelerate-20261002';D=W/'.worktrees/ir-pressure-attribution-datasets-20261004'
PY=W/'.venvs/terminal-bench-harbor/bin/python'
def read(p):return json.loads(p.read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
operation=sys.argv[1];assert operation in {'build','prepare','execute','inspect'}
recipe=read(B/'build-command.json');env={**os.environ,**recipe['env']}
trial=B/'headless-terminal-01'
if operation=='build':
 assert not (B/'bundle').exists()
 heads={k:git(r,'rev-parse','HEAD') for k,r in {'source':A,'datasets':D,'kit':W/'external/ipfs_kit'}.items()}
 assert not git(A,'status','--porcelain') and not git(D,'status','--porcelain')
 command=recipe['argv']
else:
 audit=read(B/'build-audit.json');assert sha(B/'bundle/manifest.json')==audit['manifest_sha256']
 roots={'source':A,'datasets':D,'kit':W/'external/ipfs_kit'}
 for name,digest in read(B/'source-pins.json').items():
  prefix,relative=name.split('/',1);assert sha(roots[prefix]/relative)==digest
 assert sha(B/'bundle/runtime.tar.gz')==audit['archive_sha256']
 for prefix,root in {'source':A,'datasets':D,'kit':W/'external/ipfs_kit'}.items():
  assert git(root,'rev-parse','HEAD')==audit['source_revisions'][prefix]
 if operation=='prepare':
  command=[str(PY),'-m','benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark','prepare',
   '--dataset',str(W/'.benchmarks/terminal-bench-2'),'--task','headless-terminal',
   '--task-profile',str(B/'headless-terminal-profile.json'),'--archive',str(B/'bundle'),
   '--output',str(trial),'--arm','full','--resource-profile','source384-5cpu-16gib-extended@1',
   '--source384-config',str(B/'source384-config.json'),'--setup-cache-policy','source384-native-aarch64-dontneed@1']
 elif operation=='execute':
  assert read(B/'prepare-exit.json')['returncode']==0
  command=[str(PY),'-m','benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark','execute','--output',str(trial),'--task','headless-terminal']
 else:
  write(B/'inspection-start.json',{'schema':'once-only-pilot-inspection@1','started_at':time.time()})
  raw=(trial/'receipt.json').read_bytes();assert len(raw)<16*1024*1024
  receipt=json.loads(raw);rows=[]
  sys.path[:0]=[str(A),str(D),str(W/'.venvs/terminal-bench-harbor/lib/python3.12/site-packages')]
  from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import measured_usage
  for row in receipt['trials']:
   supervisor=row.get('supervisor') or {}
   projected={key:row.get(key) for key in ('trial','exact_trial_task_matches','reward','exception_type','durations_seconds')}
   projected.update(supervisor={key:supervisor.get(key) for key in ('task_completed','seconds','phases','error_phase','remaining_processes','worker_cleanup_returncode','implementation_route')},
    error_type=(supervisor.get('error') or {}).get('type'),
    usage=measured_usage(supervisor),
    doctor={key:(supervisor.get('doctor_dispatch') or {}).get(key) for key in ('status','route','analysis_status','provider_calls','reason_codes')},
    planning={key:(supervisor.get('planning') or {}).get(key) for key in ('qualified','goals','tasks','planning_strategy','elapsed_seconds','provider_calls')},
    initial_context={key:(supervisor.get('initial_context') or {}).get(key) for key in ('seconds','indexed_symbols','full_capsules')},
    source384={key:(supervisor.get('initial_context') or {}).get('source384_context',{}).get(key) for key in ('checkpoint_sha256','inference_sha256','schema','seconds')})
   rows.append(projected)
  result={key:receipt.get(key) for key in ('schema','task','arm','trial_count','harbor_returncode','invocation_seconds','original_task_inputs_unchanged','complete_single_trial_receipt')}
  result.update(receipt_sha256=hashlib.sha256(raw).hexdigest(),receipt_bytes=len(raw),trials=rows,
    exported_model_or_verifier_bodies=False,training_on_benchmark=False,benchmark_advantage_claimed=False)
  write(B/'pilot-result.json',result);print(json.dumps(result));raise SystemExit(0)
if operation!='build':write(B/(operation+'-command.json'),{'argv':command,'cwd':str(A),'env':recipe['env'],'archive_sha256':audit['archive_sha256']})
started=time.monotonic()
with (B/(operation+'.stdout')).open('x') as out,(B/(operation+'.stderr')).open('x') as err:
 result=subprocess.run(command,cwd=A,env=env,stdout=out,stderr=err,timeout=4500 if operation=='execute' else 900)
write(B/(operation+'-exit.json'),{'returncode':result.returncode,'seconds':time.monotonic()-started})
if result.returncode:raise SystemExit(result.returncode)
if operation=='build':
 manifest=read(B/'bundle/manifest.json');rows={r['path']:r for r in manifest['files']};assert len(rows)==len(manifest['files'])
 archive=B/'bundle/runtime.tar.gz';assert sha(archive)==manifest['archive_sha256'];seen=set()
 with tarfile.open(archive,'r|gz') as tar:
  for member in tar:
   assert member.isfile() and member.name not in seen
   row=rows[member.name];assert (member.size,member.mode,member.uid,member.gid)==(row['bytes'],row['mode'],0,0)
   h=hashlib.sha256();f=tar.extractfile(member)
   for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
   assert h.hexdigest()==row['sha256'];seen.add(member.name)
 assert seen==set(rows)
 frozen={}
 for name,row in rows.items():
  prefix,relative=name.split('/',1);roots={'source':A,'datasets':D,'kit':W/'external/ipfs_kit'}
  if prefix in roots:
   assert sha(roots[prefix]/relative)==row['sha256'];frozen[name]=row['sha256']
 write(B/'source-pins.json',frozen)
 write(B/'build-audit.json',{'archive_sha256':manifest['archive_sha256'],'manifest_sha256':sha(B/'bundle/manifest.json'),
  'source_revisions':heads,'verified_members':len(seen),'repository_pins':len(frozen),'all_member_hashes_modes_ownership_verified':True,
  'generic_source384_config':manifest['source384']['config']['schema'],'header_contract_selected':False})
print(json.dumps({'operation':operation,'returncode':result.returncode,'seconds':time.monotonic()-started}))
