"""Synthetic body-leak and false-success checks; no provider/container calls."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import copy

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'qualification/inspect_live_trials.py'
spec = importlib.util.spec_from_file_location('inspect_live_trials', SCRIPT)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default='fresh-live-inspector-safety-review-04.json')
args = parser.parse_args()
if Path(args.output).name != args.output or not args.output.endswith('.json'):
    parser.error('simple new JSON output name required')
MARKER = 'PRIVATE_PROVIDER_BODY_NOT_FOR_EXPORT'
checks = []

def checked(name, value):
    assert value, name
    checks.append(name)

def rejects(call):
    try:
        call()
    except ValueError:
        return True
    return False

# Producer validators are loaded only after their bytes match the selected source commit.
source_revision = subprocess.check_output(['git', '-C', live.HEAD_PATHS['source'], 'rev-parse', 'HEAD'], text=True).strip()
runner_diagnostic = {'schema':'router-implementation-error@2', 'error_type':'ValueError', 'diagnostic': {
    'schema':'router-implementation-failure-diagnostic@1', 'phase':'argument_validation',
    'exceptions':[{'exception_type':'ValueError','frames':[{'file':'router_implementation_runner.py','line':1}]}],
    'chain_truncated':False, 'completion_authority':False, 'automatic_retry_admitted':False,
    'settlement_authority':False, 'provider_dispatch_observed':None}}
bridge_diagnostic = {'schema':'database-bridge-failure-diagnostic@1','phase':'terminal_failure',
    'reason_code':'portal_provider_failed','exceptions':[{'exception_type':'DatabasePortalBridgeError',
        'frames':[{'file':'implementation_daemon.py','line':1}]}], 'chain_truncated':False,
    'callback':{'schema':'database-native-provider-failure@1','state':'failed_outcome_settled',
        'provider_effect_state':'failed_provider_exited','native_exit':{'present':True,'returncode':1,
        'reaped':True,'process_group_absent':True,'subreaper_children_absent':True,'lifecycle_finalized':True}},
    'child_reported_router_failure':{'status':'observed','diagnostic':runner_diagnostic,
        'observation_only':True,'scope':'bounded_native_log_tail'},
    'completion_authority':False,'retry_authority':False,'settlement_authority':False}
observation = {'status':'observed','diagnostic':bridge_diagnostic,'scope':'exact_admitted_task_and_attempt',
    'matched_record_count':1,'observation_only':True,'provider_dispatch_observed':None,
    'completion_authority':False,'retry_authority':False,'settlement_authority':False}
native_failure = {'schema':'terminal-native-failure-observations@1','bridge':observation,
    'planner_child':{**observation,'diagnostic':runner_diagnostic,'scope':'planner_child_stderr'},
    'observation_only':True,'completion_authority':False,'retry_authority':False,'settlement_authority':False}
export_native = lambda value: live.native_failures(value,source_revision=source_revision)
checked('closed native and child diagnostics survive source-bound validators',export_native(native_failure)==native_failure)
checked('legacy missing native failure evidence remains unknown',live.native_failures(None) is None)
checked('unbound validator source cannot publish observed metadata',live.native_failures(native_failure)==
    {'observation':'unavailable','reason':'validator_source_unavailable'})
checked('wrong source revision cannot publish observed metadata',live.native_failures(native_failure,source_revision='0'*40)==
    {'observation':'unavailable','reason':'validator_source_unavailable'})
checked('terminal blocked stop reason retained without capability inference',live.progress({'stop_reason':'native_task_blocked'})['stop_reason']=='native_task_blocked')
for label,change in [
    ('raw top-level body',{'body':MARKER}),('completion authority',{'completion_authority':True}),
    ('retry authority',{'retry_authority':True}),('settlement authority',{'settlement_authority':True}),
    ('invalid observation-only',{'observation_only':False}),('schema injection',{'schema':MARKER}),
]:
    checked('native failure '+label+' rejected',export_native({**native_failure,**change})=={'observation':'malformed'})
for label,change in [
    ('raw bridge body',{'body':MARKER}),('dispatch inferred',{'provider_dispatch_observed':False}),
    ('completion authority',{'completion_authority':True}),('retry authority',{'retry_authority':True}),
    ('settlement authority',{'settlement_authority':True}),('oversized matches',{'matched_record_count':4097}),
    ('boolean count',{'matched_record_count':True}),('unsupported scope',{'scope':MARKER}),
    ('unobserved diagnostic',{'status':'missing'}),('zero observed count',{'matched_record_count':0}),
]:
    checked('native bridge '+label+' rejected',export_native({**native_failure,'bridge':{**observation,**change}})=={'observation':'malformed'})
for label,change in [
    ('dynamic reason',{'reason_code':MARKER}),('raw error body',{'message':MARKER}),
    ('dynamic exception',{'exceptions':[{'exception_type':MARKER,'frames':[]}]}),
    ('dynamic source path',{'exceptions':[{'exception_type':'ValueError','frames':[{'file':MARKER,'line':1}]}]}),
    ('native body',{'callback':{**bridge_diagnostic['callback'],'body':MARKER}}),
    ('child missing scope',{'child_reported_router_failure':{'status':'missing','diagnostic':None,'observation_only':True}}),
]:
    value={**native_failure,'bridge':{**observation,'diagnostic':{**bridge_diagnostic,**change}}}
    result=export_native(value)
    checked('native diagnostic '+label+' rejected',result=={'observation':'malformed'} and MARKER not in json.dumps(result))
value=copy.deepcopy(native_failure)
value['planner_child']['diagnostic']['diagnostic']['exceptions'][0]['frames'][0]['file']=MARKER
checked('planner dynamic frame rejected',export_native(value)=={'observation':'malformed'})
fallback={key:value for key,value in native_failure.items() if key not in {'bridge','planner_child'}}
fallback['status']='unavailable'
checked('closed collection failure survives without invented dispatch',export_native(fallback)==fallback)

shutdown = {'schema':'terminal-shutdown-failure-observation@1','status':'observed',
    'phase':'stop_request','reason_code':'other','exceptions':[{'exception_type':'StaleTreeError',
    'frames':[{'file':'control_plane.py','line':5142}]}], 'chain_truncated':False,
    'observation_only':True,'completion_authority':False,'retry_authority':False,'settlement_authority':False}
close_failure = {**shutdown,'phase':'runtime_close','reason_code':'live_launched_children',
    'exceptions':[{'exception_type':'RuntimeError','frames':[{'file':'isolated_benchmark_runtime.py','line':569}]}]}
both_shutdown = {'stop':shutdown,'runtime_close':close_failure}
export_shutdown = lambda value: live.shutdown_failures(value,source_revision=source_revision)
checked('source-bound original STOP and later custody failure retained separately',export_shutdown(both_shutdown)==both_shutdown)
checked('missing historical STOP diagnostic stays unknown',live.shutdown_failures(None) is None)
checked('unbound shutdown source cannot publish observed metadata',live.shutdown_failures(both_shutdown)==
    {'observation':'unavailable','reason':'validator_source_unavailable'})
checked('wrong shutdown source revision cannot publish observed metadata',live.shutdown_failures(both_shutdown,source_revision='0'*40)==
    {'observation':'unavailable','reason':'validator_source_unavailable'})
for phase in ('stop_request','stop_response_serialization','post_stop_process_observation','post_stop_context_refresh'):
    value={'stop':{**shutdown,'phase':phase}}
    checked('shutdown precise phase '+phase+' preserved',export_shutdown(value)==value)
for label,value in [
    ('unknown slot',{**both_shutdown,'private':MARKER}),('empty slots',{}),
    ('swapped STOP slot',{'stop':close_failure}),('swapped close slot',{'runtime_close':shutdown}),
    ('raw message',{'stop':{**shutdown,'message':MARKER}}),
    ('unsupported phase',{'stop':{**shutdown,'phase':MARKER}}),
    ('forged authority',{'stop':{**shutdown,'completion_authority':True}}),
    ('dynamic exception',{'stop':{**shutdown,'exceptions':[{'exception_type':MARKER,'frames':[]}]}}),
    ('dynamic frame',{'stop':{**shutdown,'exceptions':[{'exception_type':'StaleTreeError','frames':[{'file':MARKER,'line':1}]}]}}),
    ('boolean line',{'stop':{**shutdown,'exceptions':[{'exception_type':'StaleTreeError','frames':[{'file':'control_plane.py','line':True}]}]}}),
    ('contradictory reason',{'stop':{**shutdown,'reason_code':'timeout'}}),
]:
    result=export_shutdown(value)
    checked('shutdown '+label+' rejected without body',result=={'observation':'malformed'} and MARKER not in json.dumps(result))
missing_shutdown={**shutdown,'status':'unavailable','reason_code':'observation_unavailable','exceptions':[]}
checked('diagnostic observer failure remains explicit unavailable',export_shutdown({'stop':missing_shutdown})=={'stop':missing_shutdown})

structured = dict(schema='grok-native-json-schema@1', response_schema_sha256='a' * 64,
    response_schema_bytes=500, native_schema_requested=True, response_schema_validated=True, plan_admitted=False,
    body=MARKER)
row = live.invocation(dict(provider='grok_cli', provider_invocation_policy=dict(structured_output=structured,
    native_tool_allowlist=['read_file'], native_tool_denylist=['read_file', 'search_tool', 'use_tool'],
    max_turns=2, tools_profile='none', permission_mode='dontAsk', effective_toolset_verified=False)))
checked('structured schema metadata preserved without admission claim',
    row['provider_invocation_policy']['structured_output'] == {**{k:v for k,v in structured.items() if k != 'body'}, 'native_schema_projection':None})
checked('requested tool policy not actual-tool observation', row['requested_policy_is_actual_tool_observation'] is False)
checked('explicit20GiB profile retained without replacing16GiB identity',
    live.scalar('resource_profile', 'source384-5cpu-20gib-planner180@1') == 'source384-5cpu-20gib-planner180@1'
    and live.scalar('resource_profile', 'source384-5cpu-16gib-planner180@1') == 'source384-5cpu-16gib-planner180@1')
checked('explicitcoding600 identity retained without replacing300 identity',
    live.scalar('resource_profile', 'source384-5cpu-20gib-coding600@1') == 'source384-5cpu-20gib-coding600@1'
    and live.scalar('resource_profile', 'source384-5cpu-20gib-planner180@1') == 'source384-5cpu-20gib-planner180@1')
checked('declared coding cap and clamped timeout retained independently',
    live.pick(dict(provider_coding_timeout_cap_seconds=600, provider_coding_timeout_seconds=518.5),
        'provider_coding_timeout_cap_seconds provider_coding_timeout_seconds') ==
        dict(provider_coding_timeout_cap_seconds=600, provider_coding_timeout_seconds=518.5))
checked('malformed and provider-free coding budgets remain unknown',
    live.scalar('provider_coding_timeout_seconds', None) is None
    and live.scalar('provider_coding_timeout_cap_seconds', True) is None
    and live.scalar('provider_coding_timeout_seconds', {'body':MARKER}) is None)
checked('observed available Doctor analysis retained as closed enum', live.scalar('analysis_status', 'available') == 'available')
checked('provider text omitted', MARKER not in json.dumps(row))
projection=dict(projection_id='canonical-prompt-goal-native-id-omission@1',top_level_id_omitted=True,canonical_schema_sha256='a'*64,canonical_schema_bytes=500,native_wire_schema_sha256='b'*64,native_wire_schema_bytes=480,canonical_validation_preserved=True)
projected=live.structured_output({**structured,'native_schema_projection':projection})
checked('canonical and wire schema identity retained separately',all(projected['native_schema_projection'][k]==v for k,v in projection.items()) and projected['response_schema_sha256']=='a'*64)
contract_projection={**projection,'projection_id':'canonical-prompt-goal-task-contract@1','task_contract_sha256':'c'*64,'task_contract_bytes':4000,'task_count':1,'task_contract_authority':False}
checked('signed-task intersection records bounded identity without authority',live.structured_output({**structured,'native_schema_projection':contract_projection})['native_schema_projection']==contract_projection)
checked('wire projection retains no plan admission authority',projected['plan_admitted'] is False)
checked('identity projection remains recognized',live.structured_output({**structured,'native_schema_projection':{**projection,'projection_id':'identity@1','top_level_id_omitted':False}})['native_schema_projection']['projection_id']=='identity@1')
malformed=live.structured_output({**structured,'native_schema_projection':{k:{'body':MARKER} for k in projection}})
checked('untrusted nested schema projection cannot export bodies',MARKER not in json.dumps(malformed))
checked('malformed provider policy cannot export bodies',MARKER not in json.dumps(live.invocation({'provider_invocation_policy':[MARKER]})))
invalid = live.structured_output({k: {'text': MARKER} for k in structured})
checked('malformed schema receipt body excluded', MARKER not in json.dumps(invalid))
schema_failure=dict(schema='native-grok-outcome@1',reason_code='schema_rejected',classification_source='native_schema_error',http_status=400,schema_error_code='schema_id_invalid',schema_error_path='/$id',body=MARKER)
failure=live.outcome(schema_failure)
checked('closed native schema failure fields retained',all(failure[k]==v for k,v in schema_failure.items() if k!='body'))
checked('native schema failure text omitted',MARKER not in json.dumps(failure))
unknown=live.outcome({**schema_failure,'schema_error_code':MARKER,'schema_error_path':'/'+MARKER,'http_status':{'body':MARKER}})
checked('unknown schema paths and malformed statuses not exported',MARKER not in json.dumps(unknown) and unknown['http_status'] is None)
checked('timeout remains timeout alongside schema error evidence',live.outcome({**schema_failure,'reason_code':'timeout'})['reason_code']=='timeout')
checked('missing usage remains unknown', live.usage(None) is None and live.usage({'total_tokens': None})['total_tokens'] is None)
checked('malformed usage not promoted to numeric zero', live.usage({'total_tokens': True})['total_tokens'] is None)
closed = live.custody(dict(stop={'status':'succeeded'}, remaining_processes=0, worker_cleanup_returncode=0,
    runtime_close={'attempted':True, 'succeeded':False, 'error_type':'RuntimeError', 'message':MARKER},
    error={'type':'RuntimeError','message':MARKER}))
checked('STOP zero processes does not override custody close failure',
    closed['runtime_close_succeeded'] is False and closed['closure_inferred_from_stop'] is False)
checked('missing custody close stays unknown', live.custody({'stop': {'status':'succeeded'}})['runtime_close_succeeded'] is None)
checked('custody exception bodies omitted', MARKER not in json.dumps(closed))
cleanup = dict(schema='terminal-start-cleanup-observation@1', status='available', reason='bound_receipts',
    proof_observation='observed', control_observation='observed', lifecycle_phase='failed',
    control_phase='repaired', marker_bound_process_tree_absent=True, start_succeeded=False,
    absence_scope='recorded_marker_bound_tree', completion_authority=False, retry_authority=False,
    execution_authority=False)
checked('bound failed START repair retained without completion or global absence claim', live.start_cleanup(cleanup) == cleanup)
checked('absent legacy START cleanup remains unknown', live.start_cleanup(None) is None)
partial = {**cleanup, 'status':'partial', 'reason':'partial_evidence', 'control_observation':'missing', 'control_phase':None}
checked('lost control convenience receipt preserves partial proof only', live.start_cleanup(partial) == partial)
control_only = {**cleanup, 'status':'partial', 'reason':'partial_evidence', 'proof_observation':'missing',
    'lifecycle_phase':None, 'marker_bound_process_tree_absent':None, 'start_succeeded':None}
checked('control-only repair does not imply process absence', live.start_cleanup(control_only) == control_only)
for label, change in [
    ('foreign schema', {'schema':MARKER}), ('extra private body', {'body':MARKER}),
    ('foreign absence scope', {'absence_scope':MARKER}), ('completion authority', {'completion_authority':True}),
    ('retry authority', {'retry_authority':True}), ('execution authority', {'execution_authority':True}),
    ('numeric absence', {'marker_bound_process_tree_absent':1}), ('successful START', {'start_succeeded':True}),
    ('wrong lifecycle phase', {'lifecycle_phase':'completed'}), ('wrong control phase', {'control_phase':'committed'}),
    ('nested observation', {'proof_observation':{'body':MARKER}}), ('unsupported reason', {'reason':MARKER}),
    ('inconsistent status', {'status':'unavailable'}), ('unobserved proof absence', {'proof_observation':'missing'}),
]:
    result = live.start_cleanup({**cleanup, **change})
    checked('START cleanup '+label+' rejected without content', result == {'observation':'malformed'} and MARKER not in json.dumps(result))
missing = {**cleanup, 'status':'unavailable', 'reason':'evidence_unavailable', 'proof_observation':'missing',
    'control_observation':'missing', 'lifecycle_phase':None, 'control_phase':None,
    'marker_bound_process_tree_absent':None, 'start_succeeded':None}
checked('missing repair files remain explicit unavailable', live.start_cleanup(missing) == missing)
startup = live.startup(dict(schema='admitted-native-startup-observation@1', observations=[],
    bootstrap_failure_counts=[dict(phase='process_tree',reason='process_root_missing',count=3),
                              dict(phase=MARKER,reason=MARKER,count={'text':MARKER})]))
checked('closed bootstrap buckets retained', startup['bootstrap_failure_counts'][0] == dict(phase='process_tree',reason='process_root_missing',count=3))
checked('unknown bootstrap content excluded', MARKER not in json.dumps(startup))
checked('bounded task state excludes task body and identifiers',
    live.task_state(dict(status='in_progress',revision=3,body=MARKER,task_cid=MARKER)) == dict(status='in_progress',revision=3))
checked('planning qualification remains separate from transport validity',
    live.planning({'qualified':False})['independently_admitted_plan'] is False)
diagnostic={'schema':'supervisor-local-task-contract-mismatch@1','fields':[
    dict(field=field,comparison=definition[0],expected_count=1,observed_count=1,counts_capped=False,changed_members=list(definition[1]))
    for field,definition in live.TASK_CONTRACT_FIELDS.items()]}
checked('all eight exact task-contract field taxonomies retained',live.task_contract_mismatch(diagnostic)==diagnostic)
checked('planning carries closed contract diagnostic and LocalPlanningError',live.planning({'failure':{'type':'LocalPlanningError'},'task_contract_mismatch':diagnostic})['task_contract_mismatch']==diagnostic)
checked('absent historical contract diagnostic remains unknown',live.planning({})['task_contract_mismatch'] is None)
for label,change in [
    ('unknown field',{'field':MARKER}),('wrong comparison',{'comparison':'ordered_equal'}),
    ('boolean count',{'expected_count':True}),('too large count',{'observed_count':65536}),
    ('unbounded member body',{'changed_members':[MARKER]}),('invalid capped flag',{'counts_capped':True}),
    ('extra field',{'raw_value':MARKER})]:
    bad={'schema':diagnostic['schema'],'fields':[{**diagnostic['fields'][0],**change}]}
    got=live.task_contract_mismatch(bad)
    checked('contract '+label+' rejected without content',got=={'observation':'malformed'} and MARKER not in json.dumps(got))
duplicate={'schema':diagnostic['schema'],'fields':[diagnostic['fields'][0],diagnostic['fields'][0]]}
checked('duplicate contract field refused',live.task_contract_mismatch(duplicate)=={'observation':'malformed'})
valid_capped={'schema':diagnostic['schema'],'fields':[{**diagnostic['fields'][0],'observed_count':65535,'counts_capped':True}]}
checked('bounded capped contract count retained',live.task_contract_mismatch(valid_capped)==valid_capped)


with tempfile.TemporaryDirectory(prefix='fresh-live-export-fixture-') as temporary:
    root = Path(temporary)
    archive = root/'archive'; archive.mkdir()
    manifest = dict(archive_sha256='c'*64, grok_cli_assets=dict(version='1.0.46',sha256='d'*64,bytes=142867512))
    raw = json.dumps(manifest).encode(); (archive/'manifest.json').write_bytes(raw)
    prepared = dict(task='tune-mjcf',arm='full',model='grok-4.7',cli_version='1.0.46',
        provider_profile='grok-4.7-cli-1.0.46@1',reasoning_effort='high',
        resource_profile='source384-5cpu-16gib-planner180@1',archive=str(archive),
        archive_sha256='c'*64,manifest_sha256=hashlib.sha256(raw).hexdigest())
    folder=root/'grok-tune-mjcf-01'; folder.mkdir(); (folder/'preparation.json').write_text(json.dumps(prepared))
    older=root/'grok-tune-mjcf-02'; older.mkdir(); (older/'preparation.json').write_text(json.dumps(prepared))
    trial={'trials':[dict(reward={'reward':0},supervisor=dict(planning=dict(qualified=False),
        stop=dict(status='succeeded'),runtime_close=dict(attempted=True,succeeded=False,error_type='RuntimeError'),
        provider_invocations=[dict(provider_invocation_policy=dict(structured_output=structured))]))]}
    for item in (folder,older):(item/'receipt.json').write_text(json.dumps(trial))
    rows=live.collect(root,selected_names={'grok-tune-mjcf-01'})
    checked('only explicitly selected new trial exported',len(rows)==1 and rows[0]['trial_name']=='grok-tune-mjcf-01')
    checked('official zero reward remains zero', rows[0]['trial']['reward']==0 and rows[0]['trial']['reward_basis']=='official_original_verifier_receipt')
    checked('absent usage not invented', rows[0]['native_usage'] is None and rows[0]['harbor_usage']['n_input_tokens'] is None)
    checked('missing source pin explicitly unknown', rows[0]['selection_binding']['source_revisions'] is None)
    checked('body never escapes aggregate',MARKER not in json.dumps(rows))
    checked('legacy aggregate does not infer cleanup from STOP', rows[0]['start_cleanup'] is None)
    trial['trials'][0]['supervisor']['start_cleanup'] = cleanup
    (folder/'receipt.json').write_text(json.dumps(trial))
    checked('closed START cleanup is included in fresh aggregate',
        live.collect(root,selected_names={'grok-tune-mjcf-01'})[0]['start_cleanup'] == cleanup)
    trial['trials'][0]['reward']={'reward':0.5}; (folder/'receipt.json').write_text(json.dumps(trial))
    checked('unsupported partial reward remains unknown',live.collect(root,selected_names={'grok-tune-mjcf-01'})[0]['trial']['reward'] is None)
    (folder/'receipt.json').write_text('{"trials":[],"trials":[]}')
    checked('duplicate metadata fields rejected',rejects(lambda:live.collect(root,selected_names={'grok-tune-mjcf-01'})))
    (folder/'alias.json').symlink_to(folder/'preparation.json')
    checked('symlink receipt refused',rejects(lambda:live.read_metadata(root,folder/'alias.json')))

for path in ROOT.rglob('*.py'):
    ast.parse(path.read_text(),filename=str(path))
checked('artifact Python parses',True)
historical = json.loads((ROOT/'qualification/shutdown-exporter-before-01.json').read_text())
checked('completed trial01 through10 summaries remain byte-identical', all(
    hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    for name,digest in historical['sha256'].items() if name.startswith('live-trial-summary-')))
result=dict(schema='fresh-live-inspector-safety-review@1',passed=True,provider_calls=0,containers_launched=0,
    checks=checks,validator_source_revision=source_revision,inspector_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    raw_provider_verifier_or_credential_bodies_exported=False)
destination=ROOT/'qualification'/args.output
with destination.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(passed=True,checks=len(checks),provider_calls=0,containers_launched=0),indent=2))
