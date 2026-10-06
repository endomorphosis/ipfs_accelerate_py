"""Bind closed trial11 lifecycle evidence; container removal is separate from custody."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = A.parent.parent / '.worktrees/grok-recovery-20261006'
HEAD = '0e2d9a8c62eabdfbd42fcad505c370365474514a'
CID = '95c2cd7d9eecf6040b16f014c2c4dda497a4fb531111abc04123d35838c10242'


def reference(path):
    assert path.resolve() == path.absolute() and path.is_file()
    raw = path.read_bytes()
    assert len(raw) <= 8_388_608
    return {'path': str(path.relative_to(A)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write(name, value):
    with (A / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def bounded_number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 10**12 else None


def refresh_observation(report):
    value = report.get('post_publication_context')
    if type(value) is not dict:
        return {'status': 'not_observed', 'completion_authority': False}
    status = value.get('status')
    if type(status) is not str or status not in {'deferred', 'refreshed', 'incomplete', 'unavailable'}:
        status = 'invalid'
    account = value.get('embedding_accounting')
    account = account if type(account) is dict else {}
    totals = account.get('totals')
    totals = totals if type(totals) is dict else {}
    reason = value.get('reason')
    return {'status': status,
        'reason': reason if reason == 'post_stop_refresh_budget_unavailable' else None,
        'budget_seconds': bounded_number(value.get('budget_seconds')),
        'refresh_seconds': bounded_number(value.get('refresh_seconds')),
        'embedding_calls': bounded_number(value.get('embedding_calls')),
        'all_refreshes_receipted': account.get('all_refreshes_receipted')
            if type(account.get('all_refreshes_receipted')) is bool else None,
        'embedding_totals': {key: bounded_number(totals.get(key)) for key in (
            'local_embedding_calls', 'local_embedding_texts', 'remote_embedding_calls', 'text_generation_calls')},
        'completion_authority': False}


def main():
    summary_path = A / 'live-trial-summary-11.json'
    summary = json.loads(summary_path.read_text())['trials'][0]
    indexed = json.loads((A / 'indexed-path-observation-11.json').read_text())
    report_path = A / indexed['retained_driver_report']['path']
    assert reference(report_path) == indexed['retained_driver_report']
    report = json.loads(report_path.read_text())
    execution_path = A / 'grok-tune-mjcf-11-execute-exit.json'
    execution = json.loads(execution_path.read_text())
    coordinator_path = A / 'grok-container/observer-coordinator-11.json'
    coordinator = json.loads(coordinator_path.read_text())
    resources_path = A / 'grok-container/resource-observation-11.json'
    resources = json.loads(resources_path.read_text())
    assert execution['exit_code'] == 0 and execution['source_unchanged'] is True
    assert all(value == '' for value in execution['source_after']['source_status'].values())
    assert execution['source_after']['source_heads'][str(P)] == HEAD
    assert summary['selection_binding']['source_revisions']['source'] == HEAD
    assert coordinator['container_id'] == CID and coordinator['trial'] == '11'
    assert coordinator['status'] == 'observers_exited'
    assert resources['qualified'] is True and resources['exact_owned_container_verified'] is True
    query = subprocess.run(['docker', 'ps', '-a', '--no-trunc', '--filter', 'id=' + CID,
                            '--format', '{{.ID}}'], capture_output=True, text=True, timeout=10)
    assert query.returncode == 0 and not query.stdout.strip()
    custody = summary['custody']
    lifecycle = summary['lifecycle']
    shutdown = summary['shutdown_failures']
    assert shutdown is None or type(shutdown) is dict
    shutdown_status = ('not_reported' if shutdown is None else
        shutdown.get('observation', 'validated'))
    assert shutdown_status in {'not_reported', 'malformed', 'unavailable', 'validated'}
    qualified = (lifecycle == {'start_status': 'succeeded', 'stop_status': 'succeeded'}
                 and custody['runtime_close_succeeded'] is True
                 and custody['remaining_processes'] == 0
                 and custody['worker_cleanup_returncode'] == 0)
    write('trial-closure-observation-11.json', {
        'schema': 'terminal-grok-trial-closure-observation@2',
        'trial_name': 'grok-tune-mjcf-11', 'source_head': HEAD,
        'bound_execute_receipt': reference(execution_path),
        'bound_live_summary': reference(summary_path),
        'bound_observer_coordinator': reference(coordinator_path),
        'actual_resource_observation': reference(resources_path),
        'execution_exit_code': 0, 'execution_seconds': execution['seconds'],
        'source_unchanged': True, 'source_clean_after': True,
        'source_revisions': summary['selection_binding']['source_revisions'],
        'container_presence_query_exit': 0, 'exact_owned_container_present_after_cleanup': False,
        'native_start_status': lifecycle['start_status'],
        'native_stop_status': lifecycle['stop_status'],
        'runtime_close_attempted': custody['runtime_close_attempted'],
        'runtime_close_succeeded': custody['runtime_close_succeeded'],
        'runtime_close_error_type': custody['runtime_close_error_type'],
        'worker_cleanup_returncode': custody['worker_cleanup_returncode'],
        'remaining_processes': custody['remaining_processes'],
        'shutdown_failures': shutdown,
        'shutdown_failures_status': shutdown_status,
        'shutdown_failure_validation_source_head': HEAD,
        'retained_driver_report': reference(report_path),
        'post_stop_context_refresh': refresh_observation(report),
        'completion_observed': summary['supervisor']['task_completed'],
        'native_task_completion_observed': summary['task_state']['status'] == 'completed',
        'official_reward': summary['trial']['reward'],
        'original_verifier_success': (None if summary['trial']['reward'] is None
            else summary['trial']['reward'] == 1),
        'full_lifecycle_qualified': qualified,
        'container_teardown_is_native_custody_close': False,
        'observer_children_exited_zero': all(row['exit_code'] == 0 for row in coordinator['children']),
        'billing_total_verified': False, 'performance_advantage_claimed': False,
        'raw_task_model_verifier_or_credential_data_exported': False,
        'completion_authority': False,
    })
    print(json.dumps({'official_reward': summary['trial']['reward'],
        'native_task_status': summary['task_state']['status'],
        'driver_task_completed': summary['supervisor']['task_completed'],
        'full_lifecycle_qualified': qualified, 'shutdown_failures': shutdown,
        'exact_container_absent': True, 'source_unchanged': True,
        'source_freeze_may_release': True}, sort_keys=True))


if __name__ == '__main__':
    main()
