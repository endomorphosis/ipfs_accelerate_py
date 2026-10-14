"""Execute one root-released label from the frozen repeated-edit pair. No retries."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time
A=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('label',choices=['identity','comments']);args=parser.parse_args()
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
lock_path=A/'pair-lock.json';lock=json.loads(lock_path.read_bytes())
assert lock['schema']=='repeated-compact-live-pair-lock@1' and lock['frozen_before_launch'] is True
for key,path in [('runner',Path(__file__)),('recipe',A/'live-preparation/recipe-02.json'),('environment',A/'live-environment.json')]:
 row=lock['launch_inputs'][key];assert row['path']==str(path) and sha(path)==row['sha256']
for row in lock['runtime'].values():assert sha(row['path'])==row['sha256']
for path,digest in lock['host_source_sha256'].items():assert sha(path)==digest
for row in lock['attempts']:
 for key in ('preparation','configuration'):assert sha(row[key]['path'])==row[key]['sha256']
recipe=json.loads(Path(lock['launch_inputs']['recipe']['path']).read_bytes());row=next(x for x in lock['attempts'] if x['label']==args.label);trial=Path(row['directory'])
assert not (trial/'launch-start.json').exists() and not (trial/'invocation.json').exists()
command=recipe['arms'][args.label]['execute_argv'];assert command[-3:]==['execute','--output',str(trial)]
configuration=json.loads(Path(lock['launch_inputs']['environment']['path']).read_bytes());env=dict(os.environ)
for name in configuration['scrub_inherited_names']:env.pop(name,None)
env.update(configuration['environment_overrides'])
start=dict(schema='repeated-compact-live-launch@1',label=args.label,arm='full',argv=command,cwd=recipe['cwd'],pair_lock_sha256=sha(lock_path),runner_sha256=sha(__file__),recipe_sha256=lock['launch_inputs']['recipe']['sha256'],environment_manifest_sha256=lock['launch_inputs']['environment']['sha256'],preparation_sha256=row['preparation']['sha256'],started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inherited_environment_values_retained=False)
with (trial/'launch-start.json').open('x') as f:json.dump(start,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(label=args.label,status='launching',pair_lock_sha256=start['pair_lock_sha256'])),flush=True)
begin=time.monotonic()
with (trial/'execute.stdout').open('xb') as out,(trial/'execute.stderr').open('xb') as err:completed=subprocess.run(command,cwd=recipe['cwd'],env=env,stdout=out,stderr=err)
receipt_path=trial/'receipt.json';receipt=json.loads(receipt_path.read_bytes()) if receipt_path.is_file() else {}
result=dict(schema='repeated-compact-live-launch-result@1',label=args.label,arm='full',exit_code=completed.returncode,seconds=time.monotonic()-begin,preparation_unchanged=sha(row['preparation']['path'])==row['preparation']['sha256'],configuration_unchanged=sha(row['configuration']['path'])==row['configuration']['sha256'],host_source_unchanged=all(sha(p)==h for p,h in lock['host_source_sha256'].items()),stdout_sha256=sha(trial/'execute.stdout'),stderr_sha256=sha(trial/'execute.stderr'),receipt_sha256=sha(receipt_path) if receipt else None,complete_single_trial_receipt=receipt.get('complete_single_trial_receipt'),trial_count=receipt.get('trial_count'),results=[dict(reward=t.get('reward'),exception_type=t.get('exception_type'))for t in receipt.get('trials',[])])
with (trial/'launch-result.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(result,sort_keys=True),flush=True)
raise SystemExit(completed.returncode)
