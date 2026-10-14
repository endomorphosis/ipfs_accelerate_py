"""Authored finalized native metadata and actual pure-codec selection fixture."""
import copy,hashlib,importlib.util,json,tempfile
from pathlib import Path
A=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('metadata_reader',A/'collect_metadata_treatment_03.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from test.api.test_semantic_metadata_view import native_factory
from test.api.test_semantic_instruction_view import instruction
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_instruction_view as codec
from ipfs_accelerate_py.agent_supervisor.runtime.representation_displacement import observe_representation_displacement
checks=[]
def check(name,value):
 assert value,name
 checks.append(name)
def refused(name,callback):
 try:callback()
 except ValueError:checks.append(name);return
 raise AssertionError(name)
def write(path,value):
 path.write_text(json.dumps(value,sort_keys=True));return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
with tempfile.TemporaryDirectory() as directory:
 directory=Path(directory)
 _,encoded,source=native_factory.__wrapped__(directory)(6)
 block,instruction_receipt=instruction(source)
 view=codec.project_instruction_view(encoded.provider_prompt,instruction_block=block,instruction_receipt=instruction_receipt)
 before='Workspace\n'+view.native_prompt+block;candidate='Workspace\n'+view.provider_prompt+block
 selected=codec.select_instruction_view(view=view,native_complete_prompt=before,candidate_complete_prompt=candidate,instruction_block=block)
 displacement=observe_representation_displacement(view=view,selection=selected,native_complete_prompt=before,candidate_complete_prompt=candidate,provider='grok_cli',model='grok-4.7',instruction_block=block,instruction_receipt=instruction_receipt)
 call=dict(invocation_id='f'*32,phase='coding',purpose='coding',provider='grok_cli',model='grok-4.7',status='provider_returned',semantic_metadata_view=selected.receipt,representation_displacement=displacement,model_prompt_bytes=len(selected.selected_prompt.encode()),model_prompt_sha256=hashlib.sha256(selected.selected_prompt.encode()).hexdigest(),native_rollout_usage=dict(schema='native-grok-final-usage@1',usage=dict(input_tokens=10,cached_input_tokens=20,cache_write_input_tokens=0,output_tokens=5,total_tokens=35),usage_available=True,task_complete_observed=True,usage_complete_observed=True,totals_scope='native_final_envelope',cache_included_in_input=False))
 check('real fixed codec selection joins rendered input',m.selected_prompt(call)['selected_mode']==m.MODE)
 forged=copy.deepcopy(call);forged['model_prompt_bytes']+=1
 refused('selected model byte mismatch refused',lambda:m.selected_prompt(forged))
 base=directory/'trial';base.mkdir()
 release=json.loads((A/'metadata-treatment-release-03.json').read_text())
 prep=json.loads(Path(release['files']['preparation']['path']).read_text());config=json.loads(Path(release['files']['configuration']['path']).read_text());config['jobs_dir']=str(base/'jobs')
 cb=write(base/'config.json',config);prep['config_sha256']=cb['sha256'];prep['comparison_controls']=m.r.controls.build_controls(config,task_input_sha256=prep['task_input_sha256'],task=prep['task'],model=prep['model'],reasoning_effort=prep['reasoning_effort'],cli_version=prep['cli_version']);pb=write(base/'preparation.json',prep)
 release['treatment_directory']=str(base);release['files']['configuration']=cb;release['files']['preparation']=pb
 rb=write(directory/'release.json',release)
 actual_lock=json.loads((A/'metadata-execution-lock-03.json').read_text());execution=copy.deepcopy(actual_lock)
 execution['files']['release']=rb;execution['files']['preparation']=pb;execution['files']['configuration']=cb
 eb=write(directory/'execution-lock.json',execution)
 files=execution['files']
 start=dict(schema='repeated-compact-live-launch@1',label='comments-metadata',metadata_release_sha256=rb['sha256'],execution_lock_sha256=eb['sha256'],baseline_pair_lock_sha256=release['baseline_pair_lock']['sha256'],preparation_sha256=pb['sha256'],runner_sha256=files['runner']['sha256'],recipe_sha256=files['recipe']['sha256'],environment_manifest_sha256=files['environment']['sha256'])
 write(base/'launch-start.json',start)
 call['model']=prep['model']
 report=dict(arm='full',provider_profile=dict(id=prep['provider_profile'],provider='grok_cli',model=prep['model'],reasoning_effort=prep['reasoning_effort']),provider_invocations=[call],planning=dict(qualified=True,goals=1,tasks=1,provider_calls=0,planning_strategy='declared_contract'))
 raw=dict(trial='authored-treatment',task=prep['task'],exact_trial_task_matches=True,reward={'reward':1},supervisor=report,agent_context={'metadata':{'runtime_archive_sha256':prep['archive_sha256']}})
 receipt={key:prep[key]for key in ('task','model','provider_profile','reasoning_effort','cli_version')}
 receipt.update(schema=m.r.comparison.SUPERVISOR,arm='full',original_task_inputs_unchanged=True,source_view_config_unchanged=True,source_edit_config_unchanged=True,semantic_metadata_config_unchanged=True,comparison_controls=dict(status='observed',declared=prep['comparison_controls'],configuration_unchanged=True),trials=[raw])
 def save():
  received=write(base/'receipt.json',receipt)
  write(base/'launch-result.json',dict(schema='repeated-compact-live-launch-result@1',label='comments-metadata',metadata_release_sha256=rb['sha256'],execution_lock_sha256=eb['sha256'],receipt_sha256=received['sha256'],preparation_unchanged=True,configuration_unchanged=True,host_source_unchanged=True,seconds=1,exit_code=0))
 def collect():return m.treatment(directory/'release.json',directory/'execution-lock.json')
 save();x=collect()
 check('authored finalized treatment joins separate execution lock',x['treatment_control_integrity'] is True and len(x['trials'])==1)
 check('authored native totals use frozen accounting',x['trials'][0]['usage']['tokens']['total_tokens']==35)
 check('authored actual selected prompt bytes retained',x['trials'][0]['coding_input_selection'][0]['selected_complete_bytes']==call['model_prompt_bytes'])
 check('missing helper observation stays unknown',x['trials'][0]['source_use']['helper_preparation_observed'] is None)
 for flag in ('source_view_config_unchanged','source_edit_config_unchanged','semantic_metadata_config_unchanged'):
  receipt.pop(flag);save();check('missing '+flag+' prevents control integrity',collect()['treatment_control_integrity'] is False);receipt[flag]=True
 receipt['comparison_controls'].pop('status');save();check('missing observed status prevents control integrity',collect()['treatment_control_integrity'] is False);receipt['comparison_controls']['status']='observed'
 report['provider_profile']['reasoning_effort']='unexpected';save();refused('different observed reasoning effort refused',collect)
result=dict(schema='metadata-treatment-finalized-authored-checks@1',passed=len(checks),checks=checks,reader_sha256=hashlib.sha256((A/'collect_metadata_treatment_03.py').read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),provider_calls=0,production_files_changed=False,scope='Authored metadata and real pure codec fixture; not an executed benchmark trial.')
with (A/'metadata-treatment-finalized-checks-01.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(passed=len(checks),reader_sha256=result['reader_sha256'])))
