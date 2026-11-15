"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import json
SCHEMA = 'database-bridge-failure-diagnostic@2'
LEGACY_SCHEMA = 'database-bridge-failure-diagnostic@1'
MULTIPLE_CHILD_SCHEMA = 'router-child-failure-observations@1'
MAX_CHILD_DIAGNOSTICS = 4
_MULTIPLE_CHILD_FLAGS = {'observation_only': True, 'attempt_attribution_verified': False, 'provider_dispatch_observed': None, 'completion_authority': False, 'retry_authority': False, 'settlement_authority': False}
_TYPES = frozenset({'DatabasePortalBridgeError', 'DatabasePortalBridgeDeferred', 'DatabaseImplementationAuthorityError', 'OwnershipError', 'FenceMismatchError', 'WorktreeLifecycleError', 'ValueError', 'TypeError', 'RuntimeError', 'TimeoutError', 'TimeoutExpired', 'CalledProcessError', 'FileNotFoundError', 'PermissionError', 'OSError', 'ImportError', 'ModuleNotFoundError', 'KeyError', 'other'})
_REASONS = frozenset({'portal_provider_failed', 'implementation_protected_path_mutated', 'portal_validation_failed', 'portal_result_unavailable', 'other'})
_CALLBACK_SCHEMAS = frozenset({'database-portal-callback-intent@1', 'database-native-provider-failure@1', 'database-source384-no-dispatch-retry@1', 'other'})
_STATES = frozenset({'started_outcome_unknown', 'failed_outcome_settled', 'not_dispatched', 'completed', 'other'})
_EFFECTS = frozenset({'unknown_may_have_started', 'failed_provider_exited', 'not_started', 'started', 'completed', 'other'})
_EXIT_BOOLS = ('reaped', 'process_group_absent', 'subreaper_children_absent', 'lifecycle_finalized')
_FRAME_NAMES = frozenset(['admitted_benchmark_runtime.py', 'database_portal_bridge.py', 'implementation_daemon.py', 'local_completion_bridge.py', 'router_implementation_runner.py', 'source384_repository_context.py', 'supervisor_runtime.py', 'worktree_lifecycle.py'])

def _returncode(value):
    return value if type(value) is int and -255 <= value <= 255 else None

def missing_child_report(status='missing'):
    return {'status': status, 'diagnostic': None, 'observation_only': True, 'scope': 'bounded_native_log_tail'}

def validate_child_report(value):
    if type(value) is dict and value.get('schema') == MULTIPLE_CHILD_SCHEMA:
        return _validate_multiple_child_report(value)
    if type(value) is not dict or set(value) != {'status', 'diagnostic', 'observation_only', 'scope'} or value['observation_only'] is not True or (value['scope'] != 'bounded_native_log_tail') or (type(value['status']) is not str) or (value['status'] not in {'observed', 'missing', 'ambiguous', 'invalid', 'unavailable'}):
        return None
    if value['status'] != 'observed':
        return dict(value) if value['diagnostic'] is None else None
    from diagnostic_runner_contract import validate_runner_error_envelope
    try:
        diagnostic = validate_runner_error_envelope(value['diagnostic'])
    except (ValueError, TypeError, RecursionError):
        return None
    return {**missing_child_report('observed'), 'diagnostic': diagnostic}

def _diagnostic_key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def _validate_multiple_child_report(value):
    """Retain a bounded set of log observations, never verified child identities."""
    fields = {'schema', 'status', 'diagnostics', 'observed_record_count', 'scope', *_MULTIPLE_CHILD_FLAGS}
    if set(value) != fields or value['status'] != 'observed_multiple_unattributed' or value['scope'] != 'bounded_native_log_tail' or (type(value['observed_record_count']) is not int) or (not 2 <= value['observed_record_count'] <= MAX_CHILD_DIAGNOSTICS) or (type(value['diagnostics']) is not list) or (len(value['diagnostics']) != value['observed_record_count']) or any((value[key] is not expected for key, expected in _MULTIPLE_CHILD_FLAGS.items())):
        return None
    from diagnostic_runner_contract import validate_runner_error_envelope
    try:
        projected = [validate_runner_error_envelope(row) for row in value['diagnostics']]
        keys = [_diagnostic_key(row) for row in projected]
        if keys != sorted(set(keys)):
            return None
    except (ValueError, TypeError, RecursionError):
        return None
    return {**value, 'diagnostics': projected}

def validate_bridge_failure_diagnostic(value):
    """Validate an untrusted event projection; never confer custody authority."""
    try:
        fields = {'schema', 'phase', 'reason_code', 'exceptions', 'chain_truncated', 'callback', 'child_reported_router_failure', 'completion_authority', 'retry_authority', 'settlement_authority'}
        if type(value) is not dict or value.get('schema') not in {SCHEMA, LEGACY_SCHEMA}:
            return None
        if value['schema'] == SCHEMA:
            fields.add('native_provider_custody_observation')
        if set(value) != fields or value['phase'] not in {'unknown_callback', 'terminal_failure'} or value['reason_code'] not in _REASONS or (type(value['chain_truncated']) is not bool) or any((value[key] is not False for key in ('completion_authority', 'retry_authority', 'settlement_authority'))):
            return None
        exceptions = value['exceptions']
        if type(exceptions) is not list or not 1 <= len(exceptions) <= 4:
            return None
        for exception in exceptions:
            if type(exception) is not dict or set(exception) != {'exception_type', 'frames'} or exception['exception_type'] not in _TYPES or (type(exception['frames']) is not list) or (len(exception['frames']) > 8):
                return None
            for frame in exception['frames']:
                if type(frame) is not dict or set(frame) != {'file', 'line'} or frame['file'] not in _FRAME_NAMES or (type(frame['line']) is not int) or (not 0 < frame['line'] <= 10000000):
                    return None
        callback = value['callback']
        if type(callback) is not dict or set(callback) != {'schema', 'state', 'provider_effect_state', 'native_exit'}:
            return None
        for key, allowed in (('schema', _CALLBACK_SCHEMAS), ('state', _STATES), ('provider_effect_state', _EFFECTS)):
            if callback[key] is not None and callback[key] not in allowed:
                return None
        native = callback['native_exit']
        if type(native) is not dict or set(native) != {'present', 'returncode', *_EXIT_BOOLS} or type(native['present']) is not bool or (native['returncode'] is not None and _returncode(native['returncode']) is None) or any((native[key] is not None and type(native[key]) is not bool for key in _EXIT_BOOLS)) or (native['present'] is False and any((native[key] is not None for key in ('returncode', *_EXIT_BOOLS)))):
            return None
        child = validate_child_report(value['child_reported_router_failure'])
        if child is None:
            return None
        if value['schema'] == SCHEMA and value['native_provider_custody_observation'] is not None:
            from diagnostic_custody_contract import validate_native_provider_custody_observation
            if validate_native_provider_custody_observation(value['native_provider_custody_observation']) is None:
                return None
        return json.loads(json.dumps({**value, 'child_reported_router_failure': child}, allow_nan=False))
    except (ValueError, TypeError, RecursionError):
        return None
