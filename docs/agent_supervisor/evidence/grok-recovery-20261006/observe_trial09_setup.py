"""Export closed evidence for the exact completed trial09 setup failure."""
import hashlib
import json
from pathlib import Path
import subprocess

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
ROOT = A.parent.parent
P = ROOT / '.worktrees/grok-recovery-20261006'
PIN = 'b966623d3de7457b4139e7d9235a5976f275dd46'
CID = '2398305fcdcdc0bce3991e0c18d9bda6d04c547e2c2022d4749d108371a40dfc'
CASE = A / 'grok-tune-mjcf-09/jobs/supervisor-full-tune-mjcf/tune-mjcf__ZbNJX2W'
SOURCE = 'benchmarks/agent_supervisor/container_coding/terminal_deployment.py'


def read(path, maximum=1048576):
    assert path.resolve() == path.absolute() and path.is_file()
    raw = path.read_bytes()
    assert len(raw) <= maximum
    return raw


def bound(path, maximum=1048576):
    raw = read(path, maximum)
    return {'path': str(path.relative_to(A)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write(name, value):
    with (A / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    native = json.loads(read(CASE / 'result.json'))
    summary = json.loads(read(A / 'live-trial-summary-09.json'))['trials'][0]
    execution = json.loads(read(A / 'grok-tune-mjcf-09-execute-exit.json'))
    source = read(P / SOURCE)
    assert source == subprocess.check_output(['git', '-C', str(P), 'show', PIN + ':' + SOURCE])
    assert b'raise RuntimeError(f"container deployment {label} failed; see retained log")' in source
    expected = 'container deployment python-runtime-install failed; see retained log'
    exception = native['exception_info']
    assert exception['exception_type'] == 'RuntimeError' and exception['exception_message'] == expected
    log_path = CASE / 'agent/deployment/python-runtime-install.log'
    log = read(log_path).decode('utf-8')
    signals = {
        'uv_0_9_24_installed': 'Successfully installed uv-0.9.24' in log,
        'cpython_3_12_12_linux_aarch64_install_failed':
            'Failed to install cpython-3.12.12-linux-aarch64-gnu' in log,
        'download_failed_after_three_retries': 'Request failed after 3 retries' in log and 'Failed to download' in log,
        'http_500_internal_server_error': 'HTTP status server error (500 Internal Server Error)' in log,
    }
    assert all(signals.values())
    assert native.get('agent_execution') is None and native.get('verifier_result') is None
    assert not (CASE / 'agent/supervisor-result.json').exists()
    assert summary['trial']['reward'] is None and summary['trial']['reward_basis'] == 'unknown'
    assert execution['source_unchanged'] is True and execution['exit_code'] == 0
    assert all(value == '' for value in execution['source_after']['source_status'].values())
    assert execution['source_after']['source_heads'][str(P)] == PIN
    query = subprocess.run(['docker', 'ps', '-a', '--no-trunc', '--filter', 'id=' + CID,
                            '--format', '{{.ID}}'], capture_output=True, text=True, timeout=10)
    assert query.returncode == 0 and not query.stdout.strip()
    evidence = {
        'native_result': bound(CASE / 'result.json'),
        'installation_log': bound(log_path),
        'live_summary': bound(A / 'live-trial-summary-09.json'),
        'execute_receipt': bound(A / 'grok-tune-mjcf-09-execute-exit.json'),
    }
    common = {'trial_name': 'grok-tune-mjcf-09', 'source_head': PIN,
              'raw_task_model_verifier_or_credential_data_exported': False,
              'performance_advantage_claimed': False, 'completion_authority': False}
    write('setup-failure-observation-09.json', {
        **common, 'schema': 'terminal-grok-setup-failure-observation@1', 'evidence': evidence,
        'source_binding': {'path': SOURCE, 'sha256': hashlib.sha256(source).hexdigest(),
                           'committed_source_matches': True},
        'phase': 'agent_deployment', 'step': 'python-runtime-install',
        'exception_type': 'RuntimeError', 'exact_source_owned_exception_matches': True,
        'closed_signals': signals, 'reason_code': 'runtime_python_download_http_500',
        'agent_execution_started': False, 'driver_report_available': False,
        'original_verifier_result_available': False, 'official_reward': None,
        'learned_retrieval_selected': True, 'learned_retrieval_executed': None,
        'source384_inference_observed': None, 'planning_observed': None,
        'provider_usage': None, 'missing_usage_is_unknown': True,
        'outer_recipe_exit_zero_is_task_success': False,
    })
    write('trial-closure-observation-09.json', {
        **common, 'schema': 'terminal-grok-trial-closure-observation@1',
        'bound_execute_receipt': evidence['execute_receipt'],
        'bound_live_summary': evidence['live_summary'],
        'execution_exit_code': 0, 'execution_seconds': execution['seconds'],
        'source_unchanged': True, 'source_clean_after': True,
        'source_revisions': summary['selection_binding']['source_revisions'],
        'container_presence_query_exit': 0, 'exact_owned_container_present_after_cleanup': False,
        'native_start_status': None, 'native_stop_status': None,
        'runtime_close_attempted': None, 'runtime_close_succeeded': None,
        'worker_cleanup_returncode': None, 'remaining_processes': None,
        'completion_observed': None, 'official_reward': None, 'usage_complete_claimed': False,
    })
    write('indexed-path-observation-09.json', {
        **common, 'schema': 'grok-recovery-indexed-path-unavailable@1',
        'availability': 'unavailable', 'reason_code': 'agent_setup_failed_before_driver',
        'learned_retrieval_selected': True, 'learned_retrieval_executed': None,
        'index_hydration_observed': None, 'source384_inference_observed': None,
        'doctor_observed': None, 'proof_or_completion_inferred_from_index': False,
        'setup_evidence': bound(A / 'setup-failure-observation-09.json'),
    })
    write('grok-container/resource-observation-09.json', {
        **common, 'schema': 'grok-container-resource-observation-unavailable@1',
        'availability': 'unavailable', 'reason_code': 'container_removed_before_observation',
        'selected_resource_profile': 'source384-5cpu-20gib-coding600@1',
        'selected_memory_bytes': 21474836480, 'selected_cpus': 5,
        'actual_docker_resource_limits': None, 'actual_cgroup_resource_limits': None,
        'actual_resource_limits_qualified': None, 'observer_container_mutations': 0,
    })
    print(json.dumps({'trial': '09', 'setup_reason': 'runtime_python_download_http_500',
                      'official_reward': None, 'exact_owned_container_absent': True,
                      'source_unchanged': True, 'exports': 4}))


if __name__ == '__main__':
    main()
