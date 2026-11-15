"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import json
RUNNER_FAILURE_PHASES = frozenset({'runner_initialization', 'argument_validation', 'planning_contract', 'container_boundary', 'workspace_identity', 'semantic_context', 'source_formalization_context', 'numerical_obligation_descriptor', 'source_lookup_descriptor', 'source_view_descriptor', 'doctor_residual', 'public_instruction', 'model_prompt', 'coding_reply_contract', 'semantic_metadata_view', 'semantic_working_set', 'semantic_metadata_census', 'credential_isolation', 'provider_discovery', 'provider_allocation', 'provider_initialization', 'provider_invocation', 'provider_result_validation', 'semantic_response_decode', 'coding_reply_validation', 'unclassified', 'candidate_dispatch_context', 'candidate_round_preflight', 'candidate_reply_contract', 'candidate_reply_validation'})
RUNNER_FAILURE_TYPES = frozenset({'ValueError', 'TypeError', 'RuntimeError', 'PermissionError', 'FileNotFoundError', 'CalledProcessError', 'TimeoutExpired', 'TimeoutError', 'JSONDecodeError', 'UnicodeDecodeError', 'ImportError', 'ModuleNotFoundError', 'OSError', 'SemanticTranslationError', 'other'})
RUNNER_FAILURE_FILES = frozenset({'router_implementation_runner.py', 'container_worker_boundary.py', 'semantic_router_translation.py', 'doctor_residual_context.py', 'doctor_residual_budget.py', 'candidate_doctor_dispatch.py', 'candidate_round_policy.py', 'router_public_instruction.py', 'local_planning_admission.py', 'terminal_planner_contract.py', 'doctor_candidate_runner.py', 'coding_reply_contract.py', 'semantic_metadata_view.py', 'semantic_instruction_view.py', 'semantic_metadata_census.py', 'semantic_working_set.py', 'source_formalization_context.py', 'numerical_obligation_client.py', 'source_lookup_worker_client.py', 'worker_source_client.py', 'candidate_dispatch_context.py', 'candidate_reply_contract.py', 'worker_candidate_round_client.py'})

def validate_runner_failure_diagnostic(value: object) -> dict:
    """Validate child-reported metadata, never evidence of dispatch or custody."""
    if type(value) is not dict or set(value) != {'schema', 'phase', 'exceptions', 'chain_truncated', 'completion_authority', 'automatic_retry_admitted', 'settlement_authority', 'provider_dispatch_observed'} or value['schema'] != 'router-implementation-failure-diagnostic@1' or (type(value['phase']) is not str) or (value['phase'] not in RUNNER_FAILURE_PHASES) or (type(value['chain_truncated']) is not bool) or any((value[key] is not False for key in ('completion_authority', 'automatic_retry_admitted', 'settlement_authority'))) or (value['provider_dispatch_observed'] is not None) or (type(value['exceptions']) is not list) or (not 1 <= len(value['exceptions']) <= 4):
        raise ValueError('invalid closed router failure diagnostic')
    for item in value['exceptions']:
        if type(item) is not dict or set(item) != {'exception_type', 'frames'} or type(item['exception_type']) is not str or (item['exception_type'] not in RUNNER_FAILURE_TYPES) or (type(item['frames']) is not list) or (len(item['frames']) > 12):
            raise ValueError('invalid closed router exception diagnostic')
        for frame in item['frames']:
            if type(frame) is not dict or set(frame) != {'file', 'line'} or type(frame['file']) is not str or (frame['file'] not in RUNNER_FAILURE_FILES) or (type(frame['line']) is not int) or (not 1 <= frame['line'] <= 1000000):
                raise ValueError('invalid closed router failure frame')
    return json.loads(json.dumps(value, allow_nan=False))

def validate_runner_error_envelope(value: object) -> dict:
    """Require an exact wrapper error, without assigning trust to its author."""
    if type(value) is not dict or set(value) != {'schema', 'error_type', 'diagnostic'} or value['schema'] != 'router-implementation-error@2':
        raise ValueError('invalid closed router error envelope')
    diagnostic = validate_runner_failure_diagnostic(value['diagnostic'])
    if type(value['error_type']) is not str or value['error_type'] != diagnostic['exceptions'][0]['exception_type']:
        raise ValueError('router error type differs from its diagnostic')
    return {'schema': value['schema'], 'error_type': value['error_type'], 'diagnostic': diagnostic}
