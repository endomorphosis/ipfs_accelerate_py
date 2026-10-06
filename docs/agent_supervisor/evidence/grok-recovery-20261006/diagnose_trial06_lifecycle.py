"""Project source-bound lifecycle conflicts from the exact retained06 report."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = Path('/home/barberb/lift_coding/.worktrees/grok-recovery-20261006')
REVISION = '0a8309047c4e2d8f780c470a02a8453cfe01ceec'
REPORT = A / 'grok-tune-mjcf-06/jobs/supervisor-full-tune-mjcf/tune-mjcf__yfkqhhy/agent/supervisor-result.json'
FILES = {
    'lifecycle': 'ipfs_accelerate_py/agent_supervisor/control/lifecycle_orchestrator.py',
    'custody': 'ipfs_accelerate_py/agent_supervisor/entrypoints/isolated_benchmark_runtime.py',
}


def bound(path):
    raw = path.read_bytes()
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('evidence exceeds bound')
    return json.loads(raw), dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    report, report_binding = bound(REPORT)
    live, live_binding = bound(A / 'live-trial-summary-06.json')
    closure, closure_binding = bound(A / 'trial-closure-observation-06.json')
    if len(live['trials']) != 1:
        raise ValueError('single completed trial required')
    trial = live['trials'][0]
    if trial['trial_name'] != 'grok-tune-mjcf-06' or trial['selection_binding']['source_revisions']['source'] != REVISION:
        raise ValueError('exact06 source binding required')
    start, stop = report['start'], report['stop']
    if (start['status'] != 'conflict' or stop['status'] != 'conflict'
            or start['error']['details']['exception_type'] != 'ProcessIdentityMismatch'
            or not re.fullmatch(r'process [0-9]{1,10} does not belong to the selected run/profile', start['error']['message'])
            or stop['error']['details']['exception_type'] != 'TransactionConflictError'
            or stop['error']['message'] != 'another lifecycle transition is still active'
            or report['error']['type'] != 'RuntimeError'
            or report['error']['message'] != 'stop every live launched child before releasing runtime custody'):
        raise ValueError('known source-owned lifecycle conflicts required')
    source = {key:subprocess.check_output(['git','-C',str(P),'show',REVISION+':'+path]) for key,path in FILES.items()}
    lifecycle, custody = source['lifecycle'].decode().splitlines(), source['custody'].decode().splitlines()
    if ('f"process {pid} does not belong to the selected run/profile"' not in lifecycle[520]
            or 'another lifecycle transition is still active' not in lifecycle[1256]
            or 'stop every live launched child before releasing runtime custody' not in custody[525]):
        raise ValueError('historical source diagnostic binding changed')
    transactions = {}
    for name, row in [('start', start), ('stop', stop)]:
        transaction = row['data']['transaction']
        if (transaction['phase'] != 'repair_required' or transaction['recovery_action'] != 'repair'
                or transaction['failure_code'] != 'conflict' or len(transaction['effect_ids']) != 1
                or len(transaction['applied_effect_ids']) != 0):
            raise ValueError('expected unresolved lifecycle transaction required')
        transactions[name] = dict(status='conflict', transaction_phase='repair_required',
            recovery_action='repair', effect_count=1, applied_effect_count=0)
    invocations = trial['router_invocations']
    if len(invocations) != 1 or invocations[0]['purpose'] != 'planning':
        raise ValueError('only planning invocation is retained in06')
    result = dict(schema='terminal-native-lifecycle-conflict-diagnosis@1', trial_name='grok-tune-mjcf-06',
        source_revision=REVISION, official_reward=trial['trial']['reward'],
        selected_resource_profile='source384-5cpu-20gib-coding600@1',
        selected_coding_timeout_seconds=report['provider_coding_timeout_cap_seconds'],
        driver_computed_coding_timeout_seconds=report['provider_coding_timeout_seconds'],
        coding_provider_invocation_receipted=False, actual600_second_provider_execution_established=False,
        pending_settlement_fix_in_source=False,
        start_failure=dict(exception_type='ProcessIdentityMismatch', reason_code='process_run_profile_marker_mismatch',
            source_function='LinuxProcessAdapter._identity', source_line=520,
            exact_mismatched_marker='unknown', pid_reuse_or_exec_race_established=False),
        stop_failure=dict(exception_type='TransactionConflictError', reason_code='lifecycle_transition_already_active', source_line=1257),
        runtime_close_failure=dict(exception_type='RuntimeError', reason_code='live_launched_child_custody_retained',
            source_function='_require_no_live_launched_children', source_line=526),
        transactions=transactions, remaining_processes_before_container_teardown=closure['remaining_processes'],
        runtime_close_succeeded=closure['runtime_close_succeeded'],
        worker_cleanup_returncode=closure['worker_cleanup_returncode'],
        owned_container_present_after_teardown=closure['exact_owned_container_present_after_cleanup'],
        container_teardown_does_not_override_custody_failure=True,
        evidence=dict(report=report_binding, live_summary=live_binding, closure=closure_binding,
            historical_source={key:dict(path=FILES[key],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for key,raw in source.items()}),
        provider_calls_by_diagnosis=0, raw_source_model_task_verifier_or_credential_data_exported=False,
        benchmark_success_or_performance_advantage_claimed=False)
    target = A / 'lifecycle-conflict-observation-06.json'
    with target.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(qualified=True, start_reason=result['start_failure']['reason_code'], runtime_close_succeeded=False, provider_calls=0)))


if __name__ == '__main__':
    main()
