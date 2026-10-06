"""Export closed lifecycle evidence; no raw logs, fixture code, argv, or auth."""
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

A=Path(__file__).resolve().parent.parent
O=Path(__file__).resolve().parent
P=Path('/home/barberb/lift_coding/.worktrees/grok-recovery-20261006')
BASE='9ffefe04669c52ff4f51c5ee31962380ed204418'
MODULE='ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py'
sha=lambda b:hashlib.sha256(b).hexdigest()

def closed(name, keys):
    path=A/'timeout-recovery'/name
    value=json.loads(path.read_text())
    assert type(value) is dict and set(value)==set(keys)
    return value,path

def write(name,value,*,raw=None):
    path=O/name
    assert not path.exists(), f'publication staging output already exists: {name}'
    path.write_bytes(raw if raw is not None else (json.dumps(value,indent=2,sort_keys=True)+'\n').encode())

witness,path=closed('reload-witness-observation.json',{
    'schema','baseline_source_revision','corrected_source_sha256','provider_calls',
    'source_arguments_exported','credentials_read','limitations','observations'})
assert witness['provider_calls']==0 and witness['source_arguments_exported'] is False and witness['credentials_read'] is False
assert witness['baseline_source_revision']==BASE
assert len(witness['observations'])==2
for row in witness['observations']:
    assert set(row)=={'case','same_birth_alive','before_root_count','after_root_count','exact_original_argv_preserved','safe_path_flag_present','source_method_sha256','launched_child_reaped'}
    assert row['case'] in {'baseline','corrected'} and row['launched_child_reaped'] is True
write(path.name,witness,raw=path.read_bytes())
review,path=closed('independent-review.json',{
    'schema','source_revision_before_changes','reviewed_source_sha256','review_scope',
    'findings_resolved','no_blocking_findings','authority_checks_relaxed','checks',
    'real_exec_evidence_sha256','limitations'})
assert review['reviewed_source_sha256']==witness['corrected_source_sha256']
assert review['real_exec_evidence_sha256']==sha((A/'timeout-recovery'/'reload-witness-observation.json').read_bytes())
assert review['authority_checks_relaxed'] is False
assert set(review['checks'])=={'source_domain_selected_once','unrelated_target_commits_do_not_nominate_reload','same_source_updates_remain_detected','configured_wrapper_precedence_preserved','unrecognized_ambient_application_not_captured','peer_birth_lease_checks_unchanged','retry_and_completion_authority_unchanged','stop_custody_checks_unchanged'}
assert all(type(v) is bool for v in review['checks'].values())
write(path.name,review,raw=path.read_bytes())
watchdog,path=closed('watchdog-baseline-observation.json',{'schema','raw_process_arguments_exported','provider_calls','observations'})
assert watchdog['raw_process_arguments_exported'] is False and watchdog['provider_calls']==0
for row in watchdog['observations']:
    assert set(row)<= {'seconds','root_alive','root_pid_unchanged','root_argv_unchanged','safe_path_flag_present','status','control_plane_update_pending','control_plane_reload_deferred','event_counts'}
    assert set(row.get('event_counts',{}))<= {'supervisor_control_plane_update_detected','supervisor_control_plane_reload','supervisor_loop_finished'}
write(path.name,watchdog,raw=path.read_bytes())

runs=[]
for label,meaning in (
    ('timeout-baseline-01','excluded_fixture_event_path_error'),
    ('timeout-baseline-02','real_timeout_and_replacement_passed_before_watchdog_startup_grace'),
    ('timeout-baseline-03','authenticated_root_disappeared_after_watchdog_source_update_detection'),
):
    q=A/'qualification'
    command=json.loads((q/(label+'-command.json')).read_text())
    result=json.loads((q/(label+'-exit.json')).read_text())
    before=command['before'];after=result['after']
    names=set(before['source_sha256'])|set(after['source_sha256'])
    changed=sorted(n for n in names if before['source_sha256'].get(n)!=after['source_sha256'].get(n))
    junit=ET.parse(q/(label+'.xml')).getroot()
    cases=list(junit.iter('testcase'))
    counts={'passed':0,'failed':0,'errors':0,'skipped':0}
    for case in cases:
        key='failed' if case.find('failure') is not None else 'errors' if case.find('error') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
        counts[key]+=1
    runs.append({'label':label,'meaning':meaning,'exit_code':result['exit_code'],'seconds':result['seconds'],
        'source_unchanged':result['source_unchanged'],'accelerate_revision':before['accelerate_head'],
        'datasets_revision':before['datasets_head'],'source_paths_changed_during_run':changed,
        'junit_counts':counts,'log_sha256':result['log_sha256'],'junit_sha256':result['xml_sha256'],
        'raw_logs_exported':False,'provider_calls':0,'completed_benchmark_task_claim':False})
assert runs[1]['exit_code']==0 and runs[1]['source_unchanged'] is True
assert runs[2]['exit_code']==1 and MODULE not in runs[2]['source_paths_changed_during_run']
write('timeout-baseline-summary.json',{'schema':'authored-coding-timeout-baseline-summary@1','runs':runs,
    'limitations':['The first probe used the wrong portal-event path and is excluded from qualification.',
    'The long failing baseline ran while unrelated source files changed; it is development reproduction evidence, not a frozen full-suite qualification.',
    'The root-loss assertion immediately triggered fixture cleanup, before the observer captured the post-exec argv or later bootstrap rejection.',
    'The separate native-exec observation demonstrates argv loss without claiming recovery of the original Grok rejection.']})
current=(P/MODULE).read_bytes()
assert sha(current)==review['reviewed_source_sha256']
base=subprocess.check_output(['git','-C',str(P),'show',BASE+':'+MODULE])
write('lifecycle-publication-status.json',{'schema':'supervisor-lifecycle-publication-status@1',
    'status':'draft_pending_final_qualification','production_source_path':MODULE,'baseline_source_revision':BASE,
    'baseline_source_sha256':sha(base),'reviewed_corrected_source_sha256':sha(current),
    'corrected_source_commit':None,'datasets_revision':'5171a632c6b9f0ecb2939d29d2ad74992cbfeb11',
    'final_full_regression':'pending_root_confirmation','live_grok_trial':'pending_root_confirmation',
    'benchmark_completion_established':False,'token_saving_advantage_established':False,
    'historical_grok_rejection_recovered':False,'provider_calls_in_this_lifecycle_evidence':0})
print(json.dumps({'exported_json_files':sorted(p.name for p in O.glob('*.json')),'raw_logs_exported':False,'auth_exported':False}))
