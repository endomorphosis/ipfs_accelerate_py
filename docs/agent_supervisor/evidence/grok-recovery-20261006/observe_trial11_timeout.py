"""Separate trial11 provider timeout, unsettled callback and successful process cleanup."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = A.parent.parent / '.worktrees/grok-recovery-20261006'
HEAD = '0e2d9a8c62eabdfbd42fcad505c370365474514a'


def reference(path):
    assert path.resolve() == path.absolute() and path.is_file()
    raw = path.read_bytes()
    assert len(raw) <= 8_388_608
    return {'path': str(path.relative_to(A)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    summary_path = A / 'live-trial-summary-11.json'
    summary = json.loads(summary_path.read_text())['trials'][0]
    assert summary['selection_binding']['source_revisions']['source'] == HEAD
    closure_path = A / 'trial-closure-observation-11.json'
    closure = json.loads(closure_path.read_text())
    report_path = A / closure['retained_driver_report']['path']
    assert reference(report_path) == closure['retained_driver_report']
    report = json.loads(report_path.read_text())
    assert report['error'] == {'type': 'TimeoutError',
        'message': 'the total benchmark agent budget is exhausted'}
    source_name = 'benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py'
    source = subprocess.check_output(['git', '-C', str(P), 'show', HEAD + ':' + source_name])
    source_lines = [node.lineno for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and node.value == report['error']['message']]
    assert source_lines
    calls = summary['router_invocations']
    assert [row['phase'] for row in calls] == ['planning', 'coding']
    assert calls[1]['error_type'] == 'TimeoutExpired'
    assert calls[1]['timeout_seconds'] == 600 and calls[1]['native_usage'] is None
    bridge = summary['native_failure_observations']['bridge']
    assert bridge['status'] == 'observed'
    diagnostic = bridge['diagnostic']
    assert diagnostic['phase'] == 'unknown_callback'
    callback = diagnostic['callback']
    assert callback['state'] == 'started_outcome_unknown'
    assert callback['native_exit']['present'] is False
    assert summary['trial']['reward'] == 0 and summary['task_state'] == {'status': 'in_progress', 'revision': 3}
    assert closure['runtime_close_succeeded'] is True and closure['remaining_processes'] == 0
    assert closure['shutdown_failures'] is None
    assert closure['post_stop_context_refresh']['status'] == 'not_observed'
    tools_path = A / 'grok-container/native-tools-grok-tune-mjcf-11.json'
    tool_observation = json.loads(tools_path.read_text())['last_observation']
    assert tool_observation['tool_outcome_counts'] == {
        'grep:success': 1, 'list_dir:success': 1, 'other_tool:failed': 1,
        'other_tool:success': 20, 'read_file:success': 10}
    output = {
        'schema': 'terminal-provider-timeout-and-closure-observation@1',
        'trial_name': 'grok-tune-mjcf-11', 'source_head': HEAD,
        'live_summary': reference(summary_path), 'closure_observation': reference(closure_path),
        'bounded_tool_observation': reference(tools_path),
        'source_binding': {'path': source_name, 'sha256': hashlib.sha256(source).hexdigest(),
                           'matching_literal_lines': sorted(source_lines)},
        'official_reward': 0.0, 'native_task_status': 'in_progress', 'native_task_revision': 3,
        'driver_task_completed': False, 'driver_error_type': 'TimeoutError',
        'driver_reason_code': 'total_agent_work_budget_exhausted',
        'coding_error_type': 'TimeoutExpired', 'coding_phase': 'provider_invocation',
        'coding_seconds': calls[1]['seconds'], 'coding_timeout_seconds': 600,
        'coding_native_usage': None, 'total_native_usage': None, 'missing_usage_is_unknown': True,
        'planning_native_total_tokens': calls[0]['native_usage']['usage']['total_tokens'],
        'callback_state': 'started_outcome_unknown',
        'provider_effect_state': callback['provider_effect_state'],
        'native_exit_receipt_present': False, 'callback_settlement_observed': False,
        'exact_native_exit_capability_denial_reason_retained': False,
        'process_cleanup_qualified': True,
        'closure_flag_full_lifecycle_qualified_scope': 'START_STOP_tracked_tree_absence_and_runtime_close_only',
        'task_and_callback_completion_qualified': False,
        'STOP_after_success_path_exercised': False,
        'post_stop_context_refresh_eligible': False,
        'post_stop_context_refresh_not_eligible_basis': 'native_task_not_completed',
        'shutdown_failures_reported': False,
        'observed_successful_tool_outcomes': 32, 'observed_failed_tool_outcomes': 1,
        'tool_observation_scope': 'last_successful_bounded_snapshot_before_container_unavailable',
        'full_system_qualified': False, 'performance_advantage_claimed': False,
        'completion_authority': False, 'retry_authority': False, 'settlement_authority': False,
        'raw_task_model_verifier_or_credential_data_exported': False,
    }
    with (A / 'timeout-settlement-observation-11.json').open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: output[key] for key in ('official_reward', 'process_cleanup_qualified',
        'callback_settlement_observed', 'planning_native_total_tokens', 'coding_native_usage')}))


if __name__ == '__main__':
    main()
