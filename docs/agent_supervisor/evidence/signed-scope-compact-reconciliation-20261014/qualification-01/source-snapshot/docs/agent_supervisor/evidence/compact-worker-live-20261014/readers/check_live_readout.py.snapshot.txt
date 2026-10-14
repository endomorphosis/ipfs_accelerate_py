"""Authored parser/accounting checks only; no model, trial or source execution."""
import copy, hashlib, importlib.util, json, tempfile
from pathlib import Path
A = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('readout', A/'collect_live_readout.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
checks=[]
def check(name, value):
    assert value, name
    checks.append(name)
def refuses(name, operation):
    try: operation()
    except (ValueError, OSError): checks.append(name); return
    raise AssertionError(name)
def grok(identifier='1'*32):
    return dict(invocation_id=identifier, provider='grok_cli', model='grok-4.7', phase='coding', status='completed',
        native_rollout_usage=dict(schema='native-grok-final-usage@1', task_complete_observed=True, usage_complete_observed=True,
            usage_available=True, totals_scope='native_final_envelope', cache_included_in_input=False,
            usage=dict(input_tokens=100, cached_input_tokens=20, cache_write_input_tokens=5, output_tokens=10, total_tokens=135)))
a=grok(); b=grok('2'*32); b['phase']='planning'
x=m.usage({'provider_invocations':[a,b]})
check('Grok complete native final totals no double counting', x['tokens']['total_tokens']==270 and x['tokens']['input_tokens']==200)
check('planning and coding subtotals separately', x['phase_tokens']['planning']['total_tokens']==x['phase_tokens']['coding']['total_tokens']==135)
for field in ('usage_complete_observed','task_complete_observed'):
    c=copy.deepcopy(a);c['native_rollout_usage'].pop(field)
    check('missing '+field+' remains unknown', m.usage({'provider_invocations':[c]})['tokens']['total_tokens'] is None)
c=copy.deepcopy(a); c['native_rollout_usage']['usage']['total_tokens']=999
check('incoherent Grok counters never promoted', m.usage({'provider_invocations':[c]})['tokens']['total_tokens'] is None)
c=copy.deepcopy(a); c['native_rollout_usage']['usage']['input_tokens']=True
check('boolean counters refused as totals', m.usage({'provider_invocations':[c]})['tokens']['total_tokens'] is None)
check('unreceipted attempt preserves unknown all-attempt usage', m.usage({'provider_invocations':[a], 'unreceipted_provider_attempt':True})['tokens']['total_tokens'] is None)
producer_failure = dict(phase='coding', usage=None, reason='implementation_may_have_dispatched_without_a_final_router_receipt')
c = m.usage({'provider_invocations':[b], 'unreceipted_provider_attempt':producer_failure})
check('producer-shaped unreceipted coding mapping suppresses complete totals', c['tokens']['total_tokens'] is None and c['unreceipted_provider_attempt'] is True and c['invocations'][0]['reported_native_counts']['total_tokens'] == 135)
check('literal false unreceipted marker preserves complete totals', m.usage({'provider_invocations':[a], 'unreceipted_provider_attempt':False})['tokens']['total_tokens'] == 135)
check('malformed present unreceipted marker cannot assert completeness', m.usage({'provider_invocations':[a], 'unreceipted_provider_attempt':{}})['tokens']['total_tokens'] is None)
check('empty setup failure has unknown usage', m.usage({})['tokens']['total_tokens'] is None)
refuses('duplicate invocation not doublecounted',lambda:m.usage({'provider_invocations':[a,a]}))
c=copy.deepcopy(a);c['provider']='codex_cli';c['native_rollout_usage']=dict(schema='native-codex-cumulative-usage@1',task_complete_observed=True,cache_included_in_input=True,usage=dict(input_tokens=100,cached_input_tokens=80,output_tokens=20,total_tokens=120))
check('Codex included cache not added again',m.usage({'provider_invocations':[c]})['tokens']['total_tokens']==120)
c['native_rollout_usage']['usage']['total_tokens']=200
check('incoherent Codex counters remain unknown',m.usage({'provider_invocations':[c]})['tokens']['total_tokens'] is None)
check('missing source observation is unknown not zero',m.source_usage({},[],'identity@1')['helper_preparation_observed'] is None)
# Reuse only retained content-free Docker metadata to exercise the native joins.
R=A/'docker/run-01'
source=json.loads((R/'results/session-observation.json').read_text())
report=json.loads((R/'private-fixture-ipc/router-observation.json').read_text())['receipt']
invocation=dict(invocation_id=report['invocation_id'],source_dispatch={k:report['source_view_retrieval'][k] for k in ('profile','source_edit_mode','descriptor_sha256')})
invocation['source_dispatch']['session_id_sha256']=m.sha(report['source_view_retrieval']['session_id'].encode())
x=m.source_usage({'source_view_retrieval_observation':source},[invocation],'python-comments@1')
check('actual Docker dispatch and3 confirmations replay',x['source_edit_counts']['delivered_confirmation_pages']==3 and x['helper_confirmation_delivery_observed'] is True)
check('ordinary tool census stays unknown',x['ordinary_tool_usage'] is None)
for name, mutate in [
 ('wrong selected mode', lambda value:value.__setitem__('source_edit_mode','mapped-span@1')),
 ('wrong session profile',lambda value:value['sessions'][0].__setitem__('profile','identity@1')),
 ('mismatched aggregate',lambda value:value['source_edit_counts'].__setitem__('delivered_confirmation_pages',4)),
 ('unbounded dynamic counter labels',lambda value:value['counts'].__setitem__('private_prompt',1)),
 ('arbitrary status label',lambda value:value['sessions'][0]['status_counts'].__setitem__('secret_value',1)),
 ('proof promotion',lambda value:value.__setitem__('proof_authority',True)),
 ('missing aggregate counter',lambda value:value['counts'].pop('prepared_pages')),
 ('missing session counter',lambda value:value['sessions'][0]['counts'].pop('prepared_pages')),
 ('empty sessions falsely observed zero',lambda value:value.__setitem__('sessions',[])),
]:
    changed=copy.deepcopy(source);mutate(changed)
    refuses(name,lambda:m.source_usage({'source_view_retrieval_observation':changed},[invocation],'python-comments@1'))
refuses('missing actual dispatcher',lambda:m.source_usage({'source_view_retrieval_observation':source},[],'python-comments@1'))
with tempfile.TemporaryDirectory() as directory:
    directory=Path(directory);p=directory/'metadata.json';p.write_text('{"a":1,"a":2}')
    refuses('duplicate JSON fields',lambda:m.read(p));p.write_text('{"a":NaN}')
    refuses('nonfinite JSON fields',lambda:m.read(p));p.write_text('{"a":1}')
    link=directory/'link.json';link.symlink_to(p);refuses('symlink metadata refused',lambda:m.read(link))
    p.write_text(json.dumps(dict(returncode=1,model_calls=0,provider_generation_calls=0,argv=['python','-m','prepare'],seconds=1)))
    check('configuration failure separate zero-generation accounting',m.preparation_failure(p)['reported_model_calls']==0)
# Exercise the complete retained receipt path against authored metadata only.
with tempfile.TemporaryDirectory() as directory:
    directory=Path(directory)
    original_lock=json.loads((A/'pair-lock.json').read_text())
    lock=copy.deepcopy(original_lock);lock.pop('retained_prelaunch_failure',None)
    recipes=[]
    for index,row in enumerate(lock['attempts']):
        base=directory/row['label'];base.mkdir()
        for field,name in [('preparation','preparation.json'),('configuration','config.json')]:
            raw=Path(row[field]['path']).read_bytes();(base/name).write_bytes(raw)
            row[field]['path']=str(base/name)
        row['directory']=str(base)
        prep=json.loads((base/'preparation.json').read_text())
        call=grok(str(index+3)*32);call['model']=prep['model']
        report=dict(arm='full',provider_profile=dict(id=prep['provider_profile'],provider='grok_cli',model=prep['model'],reasoning_effort=prep['reasoning_effort']),provider_invocations=[call])
        trial=dict(trial='authored-'+row['label'],task=prep['task'],exact_trial_task_matches=True,reward={'reward':1},
                   supervisor=report,agent_context={'metadata':{'runtime_archive_sha256':prep['archive_sha256']}})
        receipt={key:prep[key]for key in ('task','model','provider_profile','reasoning_effort','cli_version')}
        receipt.update(schema=m.comparison.SUPERVISOR,arm='full',original_task_inputs_unchanged=True,
            source_view_config_unchanged=True,source_edit_config_unchanged=True,
            comparison_controls=dict(status='observed',declared=prep['comparison_controls'],configuration_unchanged=True),trials=[trial])
        (base/'receipt.json').write_text(json.dumps(receipt))
        recipes.append((base,prep,receipt))
    lock_path=directory/'lock.json';lock_path.write_text(json.dumps(lock));lock_binding=m.read(lock_path)[1]
    manifest=m.input_from_lock(lock,lock_binding)
    for row,(base,prep,receipt) in zip(lock['attempts'],recipes):
        start=dict(schema='repeated-compact-live-launch@1',label=row['label'],pair_lock_sha256=lock_binding['sha256'],
            preparation_sha256=row['preparation']['sha256'],runner_sha256=lock['launch_inputs']['runner']['sha256'],
            recipe_sha256=lock['launch_inputs']['recipe']['sha256'],environment_manifest_sha256=lock['launch_inputs']['environment']['sha256'])
        (base/'launch-start.json').write_text(json.dumps(start))
        result=dict(schema='repeated-compact-live-launch-result@1',label=row['label'],exit_code=0,seconds=2,
            preparation_unchanged=True,configuration_unchanged=True,host_source_unchanged=True,
            receipt_sha256=m.sha((base/'receipt.json').read_bytes()))
        (base/'launch-result.json').write_text(json.dumps(result))
    measured=m.collect(manifest)
    check('complete authored pair native totals',all(row['tokens']['total_tokens']==135 for row in measured['aggregates'].values()))
    check('actual source profile difference retained despite generic match',measured['attempts'][1]['generic_control_comparison']['matches'] is True and measured['attempts'][1]['configuration_difference_paths']==['/agents/0/kwargs/source_view_profile','/jobs_dir'])
    check('all327 frozen host pins replayed',measured['frozen_host_source_count']==327)
    check('missing helper telemetry remains unknown',all(row['trials'][0]['source_use']['helper_preparation_observed'] is None for row in measured['attempts']))
    base,prep,receipt=recipes[-1]
    receipt['trials'][0]['supervisor']['unreceipted_provider_attempt']=True
    receipt['trials'][0]['reward']=None
    (base/'receipt.json').write_text(json.dumps(receipt))
    launch=json.loads((base/'launch-result.json').read_text());launch['receipt_sha256']=m.sha((base/'receipt.json').read_bytes());(base/'launch-result.json').write_text(json.dumps(launch))
    measured=m.collect(manifest)
    check('failed incomplete attempt preserved and suppresses delta',measured['aggregates']['comments']['registered_attempts']==1 and measured['aggregates']['comments']['unknown_rewards']==1 and measured['comments_minus_identity_tokens']['total_tokens'] is None)
    changed=copy.deepcopy(manifest);changed['attempts'][0]['directory']=str(directory/'other')
    refuses('caller cannot replace locked attempt population',lambda:m.collect(changed))
    launch['configuration_unchanged']=False;(base/'launch-result.json').write_text(json.dumps(launch))
    check('launch control drift suppresses comparability',m.collect(manifest)['all_receipt_controls_match_preparation'] is False)
result=dict(schema='compact-live-readout-authored-checks@1',checks=checks,passed=len(checks),failed=0,
    collector_sha256=m.sha((A/'collect_live_readout.py').read_bytes()),helper_sha256=m.sha(Path(__file__).read_bytes()),
    provider_calls=0,production_files_changed=False,scope='Authored accounting/refusal checks plus retained content-free Docker metadata replay; not a live comparison.')
(A/'live-readout-checks.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'passed':len(checks),'collector_sha256':result['collector_sha256']}))
