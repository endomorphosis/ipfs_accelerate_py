"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import re
import json
from diagnostic_json_contract import canonical_json
import diagnostic_public_check_contract as profiles
OBSERVATION_SCHEMA = 'supervisor-retired-candidate-validation@1'
FLAGS = dict(observation_only=True, completion_authority=False, proof_authority=False, execution_authority=False, continuation_authority=False)

def _need(value, reason):
    if not value:
        raise ValueError(reason)

def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None

def _copy(value):
    return json.loads(canonical_json(value))

def _validate_states(value, names):
    _need(type(value) is dict and set(value) == set(names), 'exact validation source states required')
    for row in value.values():
        _need(type(row) is dict and set(row) == {'sha256', 'bytes', 'mode', 'uid', 'gid'} and _digest(row['sha256']) and (type(row['bytes']) is int) and (0 <= row['bytes'] <= 1000000) and (type(row['mode']) is int) and (0 <= row['mode'] <= 4095) and all((type(row[k]) is int and 0 <= row[k] <= 2 ** 32 - 1 for k in ('uid', 'gid'))), 'closed validation source state required')

def validate_public_validation_observation(value):
    keys = {'schema', 'contract_sha256', 'original_deadline_ns', 'actual_returncode', 'round_number', 'session_id', 'descriptor_sha256', 'workspace_generation', 'source_hashes', 'workspace_states', 'status', 'failed_obligations', 'checks', *FLAGS}
    _need(type(value) is dict and set(value) == keys and (value['schema'] == OBSERVATION_SCHEMA) and all((value[k] is v for k, v in FLAGS.items())) and _digest(value['contract_sha256']) and _digest(value['workspace_generation']) and (type(value['original_deadline_ns']) is int) and (value['original_deadline_ns'] > 0) and (type(value['actual_returncode']) is int) and (-255 <= value['actual_returncode'] <= 255) and (type(value['round_number']) is int) and (value['round_number'] in (1, 2)) and _digest(value['descriptor_sha256']) and (type(value['session_id']) is str) and (re.fullmatch('[0-9a-f]{32}', value['session_id']) is not None), 'closed retired validation observation required')
    names = {profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE, 'eval.py', 'eigen.py'}
    _validate_states(value['workspace_states'], names)
    _need(type(value['source_hashes']) is dict and 1 <= len(value['source_hashes']) <= len(names) and (set(value['source_hashes']) <= names) and all((_digest(h) and value['workspace_states'][p]['sha256'] == h for p, h in value['source_hashes'].items())), 'exact observed candidate source hashes required')
    checks = value['checks']
    _need(type(checks) is list and len(checks) == 3, 'complete public validation observation required')
    statuses, failed = ([], [])
    for row, check, key in zip(checks, profiles.PUBLIC_CHECKS, profiles.PUBLIC_VALIDATION_KEYS):
        _need(type(row) is dict and set(row) == {'validation_key', 'returncode', 'result', 'status', 'reason'} and (row['validation_key'] == key), 'closed public check observation required')
        if row['result'] is not None:
            result = profiles.public_validation_result(row['result'], check=check)
            _need(row['status'] == result['status'] and row['reason'] == result['reason'] and (type(row['returncode']) is int) and (row['returncode'] == {'passed': 0, 'failed': 1, 'inconclusive': 2}[result['status']]), 'public check process and result disagree')
        else:
            _need(row['status'] == 'inconclusive' and row['reason'] in {'invalid_result', 'timeout', 'output_limit', 'not_run'} and (row['returncode'] is None or (type(row['returncode']) is int and -255 <= row['returncode'] <= 255)), 'unavailable public validation result must remain inconclusive')
        statuses.append(row['status'])
        if row['status'] == 'failed':
            failed.append(key)
    status = 'failed' if failed else 'passed' if set(statuses) == {'passed'} else 'inconclusive'
    _need(value['status'] == status and value['failed_obligations'] == failed, 'public validation aggregate differs from checks')
    return _copy(value)
