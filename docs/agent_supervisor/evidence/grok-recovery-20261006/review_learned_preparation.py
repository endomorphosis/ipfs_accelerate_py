"""Verify one prepared learned-index trial before any model dispatch."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path
A=Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P=Path('/home/barberb/lift_coding/.worktrees/grok-recovery-20261006')
REVISION='17e1f347d17fe144873b1201da91788898c639cd'
CHECKPOINT='2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5'
SOURCE_PATHS=(
 'benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py',
 'benchmarks/agent_supervisor/container_coding/terminal_failure_observation.py',
 'benchmarks/agent_supervisor/container_coding/container_worker_deployment.py',
 'benchmarks/agent_supervisor/container_coding/terminal_worker_capability.py',
 'benchmarks/agent_supervisor/container_coding/terminal_retrieval_selection.py',
 'benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py',
 'benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py',
 'benchmarks/agent_supervisor/container_coding/learned_vector_preflight.py',
 'ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py',
 'ipfs_accelerate_py/agent_supervisor/todo_daemon/bridge_failure_diagnostics.py',
 'ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py',
 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',
)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--expected-source-head',required=True);a=p.parse_args()
 assert re.fullmatch('[0-9]{2}',a.attempt) and re.fullmatch('[0-9a-f]{40}',a.expected_source_head)
 assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==a.expected_source_head
 assert not subprocess.check_output(['git','-C',str(P),'status','--porcelain'],text=True)
 read=lambda name:json.loads((A/name).read_text())
 stem='grok-tune-mjcf-'+a.attempt;bundle='grok-bundle-'+a.attempt
 manifest=read(bundle+'/manifest.json');audit=read('grok-container/archive-review-'+a.attempt+'.json')
 prepared=read(stem+'/preparation.json');config=read(stem+'/config.json');prior=read('grok-tune-mjcf-08/preparation.json')
 assert audit['qualified'] and audit['source_heads'][str(P)]==a.expected_source_head
 assert prepared['prepared'] and prepared['source384_enabled'] and prepared['arm']=='full'
 assert prepared['archive_sha256']==manifest['archive_sha256']==audit['archive_sha256_actual']
 assert hashlib.sha256((A/bundle/'manifest.json').read_bytes()).hexdigest()==prepared['manifest_sha256']==audit['manifest_sha256']
 assert manifest['model_snapshot_revision']==REVISION and manifest['learned_requirements']
 assert config['agents'][0]['kwargs']['model_revision']==REVISION
 assert prepared['task_input_sha256']==prior['task_input_sha256']
 profile=A/(stem+'-profile.json');assert profile.read_bytes()==(A/'grok-tune-mjcf-08-profile.json').read_bytes()
 assert prepared['resource_profile']=='source384-5cpu-20gib-coding600@1' and prepared['provider_profile']=='grok-4.7-cli-1.0.46@1'
 sys.path.insert(0,str(P))
 from benchmarks.agent_supervisor.container_coding.terminal_worker_capability import require_worker_capability
 capability=require_worker_capability(manifest,prepared['resource_profile']);assert capability['max_timeout_seconds']==600
 inventory={x['path']:x for x in manifest['files']};hashes={}
 for name in SOURCE_PATHS:
  raw=subprocess.check_output(['git','-C',str(P),'show',a.expected_source_head+':'+name]);digest=hashlib.sha256(raw).hexdigest()
  assert digest==inventory['source/'+name]['sha256'];hashes[name]=digest
 cfg=manifest['source384']['config'];assert cfg['training_steps']==cfg['download_calls']==0
 assert cfg['checkpoint_sha256']==CHECKPOINT and cfg['embedding_revision']==REVISION
 checkpoint=inventory['models/source384/checkpoint.json'];assert checkpoint['sha256']==CHECKPOINT and checkpoint['bytes']==493355
 stages={}
 for stage,stage_stem in [('bundle',bundle),('profile',stem+'-profile'),('prepare',stem+'-prepare')]:
  row=read(stage_stem+'-exit.json');assert row['exit_code']==0 and row['source_unchanged']
  stages[stage]={k:row[k] for k in ('exit_code','seconds','source_unchanged')}
 assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==a.expected_source_head
 assert not subprocess.check_output(['git','-C',str(P),'status','--porcelain'],text=True)
 result={'schema':'terminal-fresh-learned-trial-preparation-review@1','attempt':a.attempt,'source_head':a.expected_source_head,
  'prepared':True,'stages':stages,'provider_calls':0,'containers_launched':0,'provider_profile':prepared['provider_profile'],
  'resource_profile':prepared['resource_profile'],'archive_sha256':manifest['archive_sha256'],'archive_bytes':audit['archive_bytes'],
  'archive_members':audit['archive_verified_member_count'],'runtime_source_sha256':hashes,'task_input_bindings_match08':True,
  'task_profile_matches08':True,'task_profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),
  'source384_checkpoint_sha256':CHECKPOINT,'source384_checkpoint_bytes':493355,'source384_embedding_revision':REVISION,
  'retrieval_model_revision':REVISION,'retrieval_policy_selected':'local-safetensors-symbols@1',
  'learned_revision_transported_to_agent':True,'worker_archive_capability':capability,'training_steps':0,'embedding_download_calls':0,
  'actual_index_hydration_observed':False,'actual_neural_retrieval_observed':False,'completion_authority':False,
  'changed_retrieval_from08':True,'matched_index_performance_comparison_claimed':False}
 with (A/'grok-container'/('preparation-review-'+a.attempt+'.json')).open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'prepared':True,'source_head':a.expected_source_head,'retrieval_revision':REVISION,'worker_coding_ceiling':600,'provider_calls':0}))

if __name__=='__main__':main()
