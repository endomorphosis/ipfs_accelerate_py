"""Bind closed05 timeout/settlement evidence to its exact historical runtime."""
import hashlib
import json
from pathlib import Path
import subprocess

A = Path('/home/barberb/lift_coding/artifacts/grok-recovery-20261006')
P = Path('/home/barberb/lift_coding/.worktrees/grok-recovery-20261006')
REVISION = '6e3171c913b76b569b85600179199f2b57ec5b52'
FILES = {
    'bridge': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py',
    'daemon': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py',
}


def bound(path):
    raw = path.read_bytes()
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('closed evidence exceeds bound')
    return json.loads(raw), dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    live, live_binding = bound(A / 'live-trial-summary-05.json')
    observed, observation_binding = bound(A / 'timeout-recovery/live-admission-05-09.json')
    closure, closure_binding = bound(A / 'trial-closure-observation-05.json')
    if len(live['trials']) != 1:
        raise ValueError('single completed trial required')
    trial = live['trials'][0]
    if (trial['trial_name'] != 'grok-tune-mjcf-05'
            or trial['selection_binding']['source_revisions']['source'] != REVISION
            or observed['trial'] != '05'):
        raise ValueError('exact05 source binding required')
    coding = [row for row in trial['router_invocations'] if row['purpose'] == 'coding']
    if (len(coding) != 1 or coding[0]['error_type'] != 'TimeoutExpired'
            or coding[0]['provider_failure']['phase'] != 'provider_invocation'
            or coding[0]['native_provider_outcome']['reason_code'] != 'timeout'
            or coding[0]['timeout_seconds'] != 300 or coding[0]['native_usage'] is not None):
        raise ValueError('known coding timeout evidence required')
    reason = 'portal_provider_failed'
    reason_hash = hashlib.sha256(reason.encode()).hexdigest()
    exceptions = [row for item in observed['bridge_unknown_diagnostics'] for row in item['exceptions']]
    matched = [row for row in exceptions if row['exception_type'] == 'DatabasePortalBridgeError' and row['message_sha256'] == reason_hash]
    if len(matched) != 1:
        raise ValueError('exact closed bridge reason binding required')
    source = {key:subprocess.check_output(['git','-C',str(P),'show',REVISION+':'+path]) for key,path in FILES.items()}
    bridge, daemon = source['bridge'].decode().splitlines(), source['daemon'].decode().splitlines()
    if ('return str(implementation.get("reason") or "portal_provider_failed")' not in bridge[10959]
            or 'raise DatabasePortalBridgeError(failure, result=summary)' not in bridge[11282]
            or 'callback.get("callback_state") == "started_outcome_unknown"' not in daemon[76401]
            or '"reason": "provider_callback_outcome_unknown"' not in daemon[76424]):
        raise ValueError('historical source branch binding changed')
    passes = observed['pass_reasons']['provider_callback_outcome_unknown']
    if type(passes) is not int or not 1 <= passes <= 100000:
        raise ValueError('bounded observed pass count required')
    result = dict(schema='terminal-coding-timeout-settlement-diagnosis@1', trial_name='grok-tune-mjcf-05',
        source_revision=REVISION, official_reward=trial['trial']['reward'],
        resource_profile='source384-5cpu-20gib-planner180@1',
        coding_provider_actually_invoked=True, previous_pre_coding_resource_gate_crossed=True,
        resource_retry_count_in_bounded_observation=len(observed['resource_retry_receipts']),
        provider=dict(provider='grok_cli', model='grok-4.7', error_type='TimeoutExpired',
            failure_phase='provider_invocation', seconds=coding[0]['seconds'], timeout_seconds=300,
            native_final_envelope_observed=False, coding_usage='unknown'),
        settlement=dict(native_exception_type='DatabasePortalBridgeError',
            native_reason_code=reason, native_reason_hash_matches=True,
            retained_callback_state='started_outcome_unknown',
            deferred_reason='provider_callback_outcome_unknown',
            completed_passes_in_bounded_observation=passes,
            exact_historical_source_branch_confirmed=True,
            effect_reconciliation_or_retry_authority=False,
            fix_or_safe_terminal_settlement_verified_by_this_observation=False),
        final=dict(task_status=trial['task_state']['status'],
            runtime_close_succeeded=closure['runtime_close_succeeded'],
            remaining_processes=closure['remaining_processes'],
            exact_owned_container_present=closure['exact_owned_container_present_after_cleanup']),
        evidence=dict(live_summary=live_binding, retained_live_observation=observation_binding,
            closure=closure_binding, historical_source={key:dict(path=FILES[key],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for key,raw in source.items()}),
        provider_calls_by_diagnosis=0, raw_source_model_task_verifier_or_credential_data_exported=False,
        benchmark_success_or_performance_advantage_claimed=False)
    target = A / 'coding-settlement-observation-05.json'
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False);stream.write('\n')
    print(json.dumps(dict(qualified=True, reason_code=reason, deferred_passes=passes, coding_usage='unknown', provider_calls=0)))


if __name__ == '__main__':
    main()
