"""Close trial10 evidence without treating verifier success as lifecycle success."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = A.parent.parent / '.worktrees/grok-recovery-20261006'
HEAD = 'b966623d3de7457b4139e7d9235a5976f275dd46'
CID = 'fe5b0b452d8541b83189c65a456efb5c6e940f60ae18cdbc72b92537d7f03950'


def reference(path):
    assert path.resolve() == path.absolute() and path.is_file()
    raw = path.read_bytes()
    assert len(raw) <= 1048576
    return {'path':str(path.relative_to(A)), 'bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest()}


def write(name, value):
    with (A/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def main():
    summary_path = A/'live-trial-summary-10.json'
    summary = json.loads(summary_path.read_text())['trials'][0]
    indexed = json.loads((A/'indexed-path-observation-10.json').read_text())
    report_path = A/indexed['retained_driver_report']['path']
    assert reference(report_path) == indexed['retained_driver_report']
    report = json.loads(report_path.read_text())
    exit_path = A/'grok-tune-mjcf-10-execute-exit.json'
    execution = json.loads(exit_path.read_text())
    assert execution['exit_code'] == 0 and execution['source_unchanged'] is True
    assert all(value == '' for value in execution['source_after']['source_status'].values())
    assert execution['source_after']['source_heads'][str(P)] == HEAD
    assert summary['selection_binding']['source_revisions']['source'] == HEAD
    assert summary['trial']['reward'] == 1 and summary['task_state'] == {'revision':4,'status':'completed'}
    assert summary['supervisor']['task_completed'] is False
    assert summary['custody']['runtime_close_succeeded'] is False
    assert summary['lifecycle']['stop_status'] is None
    assert report['error'] == {'type':'RuntimeError',
        'message':'stop every live launched child before releasing runtime custody'}
    name = 'ipfs_accelerate_py/agent_supervisor/entrypoints/isolated_benchmark_runtime.py'
    raw = subprocess.check_output(['git','-C',str(P),'show',HEAD+':'+name])
    nodes = [node for node in ast.walk(ast.parse(raw)) if isinstance(node,ast.Constant)
             and node.value == report['error']['message']]
    assert len(nodes) == 1
    query = subprocess.run(['docker','ps','-a','--no-trunc','--filter','id='+CID,'--format','{{.ID}}'],
        capture_output=True,text=True,timeout=10)
    assert query.returncode == 0 and not query.stdout.strip()
    common = {'trial_name':'grok-tune-mjcf-10','source_head':HEAD,
              'raw_task_model_verifier_or_credential_data_exported':False,
              'performance_advantage_claimed':False,'completion_authority':False}
    write('trial-closure-observation-10.json',{
        **common,'schema':'terminal-grok-trial-closure-observation@1',
        'bound_execute_receipt':reference(exit_path),'bound_live_summary':reference(summary_path),
        'execution_exit_code':0,'execution_seconds':execution['seconds'],
        'source_unchanged':True,'source_clean_after':True,
        'source_revisions':summary['selection_binding']['source_revisions'],
        'container_presence_query_exit':0,'exact_owned_container_present_after_cleanup':False,
        'native_start_status':summary['lifecycle']['start_status'],'native_stop_status':None,
        'runtime_close_attempted':True,'runtime_close_succeeded':False,
        'worker_cleanup_returncode':0,'remaining_processes':None,
        'completion_observed':False,'native_task_completion_observed':True,
        'original_verifier_success':True,'official_reward':1.0,
        'full_lifecycle_qualified':False,'usage_complete_claimed':False,
        'native_usage_envelopes_observed':2,'billing_total_verified':False,
        'container_teardown_is_native_custody_close':False,
    })
    diagnostic = report.get('native_diagnostics') or {}
    assert diagnostic.get('exception_types') == [] and diagnostic.get('traceback_frames') == []
    assert (report_path.parent/'supervisor.stderr').stat().st_size == 0
    write('closure-failure-diagnosis-10.json',{
        **common,'schema':'terminal-native-closure-failure-diagnosis@1',
        'retained_report':reference(report_path),'exact_exception_type':'RuntimeError',
        'reason_code':'live_launched_children_prevent_custody_release',
        'source_owned_literal':report['error']['message'],
        'source_binding':{'path':name,'line':nodes[0].lineno,'sha256':hashlib.sha256(raw).hexdigest()},
        'stop_receipt_available':False,'primary_stop_exception_available':False,
        'native_traceback_frames_retained':0,'native_exception_types_retained':0,
        'driver_stderr_bytes':0,'close_error_can_mask_stop_error_in_current_finally_structure':True,
        'exact_original_stop_cause_established':False,'repair_or_retry_authority':False,
        'official_reward':1.0,'native_task_status':'completed','driver_task_completed':False,
        'runtime_close_succeeded':False,'container_removed':True,
    })
    print(json.dumps({'official_reward':1.0,'native_task_status':'completed',
        'driver_task_completed':False,'native_custody_closed':False,'exact_container_absent':True,
        'source_unchanged':True,'source_freeze_may_release':True}))


if __name__ == '__main__':
    main()
