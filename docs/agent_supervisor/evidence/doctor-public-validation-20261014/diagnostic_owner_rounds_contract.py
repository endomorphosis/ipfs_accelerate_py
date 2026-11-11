"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import re
import math
ROW_FIELDS = frozenset(('round_number', 'process_started', 'session_id', 'descriptor_sha256', 'returncode', 'wait_completed', 'retirement_observed'))
OBSERVATION_FIELDS = frozenset(('schema', 'rounds', 'max_rounds', 'elapsed_seconds', 'outcome', 'failure_type', 'completion_authority', 'proof_authority', 'source_obligations_discharged', 'accepted_index_published'))

def _need(value):
    if not value:
        raise ValueError('native owner lifecycle binding differs')

def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None

def lifecycle_name(session):
    _need(type(session) is str and re.fullmatch('[0-9a-f]{32}', session) is not None)
    return 'candidate-rounds-' + session + '.json'

def _observation(value, *, normalized):
    public = type(value) is dict and value.get('schema') == 'owner-candidate-rounds-observation@2'
    fields = OBSERVATION_FIELDS | ({'public_validation_observations', 'owner_validation_outcome', 'owner_returncode'} if public else set())
    _need(type(value) is dict and set(value) >= fields and (not set(value) - fields - {'candidate_input_retention'}) and (value['schema'] in ('owner-candidate-rounds-observation@1', 'owner-candidate-rounds-observation@2')) and (type(value['max_rounds']) is int) and (value['max_rounds'] == 2) and (type(value['elapsed_seconds']) in (int, float)) and (0 <= value['elapsed_seconds'] <= 86400) and math.isfinite(value['elapsed_seconds']) and (value['outcome'] in ('failed', 'workers_returned')) and (value['failure_type'] is None or (type(value['failure_type']) is str and re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,95}', value['failure_type']) is not None)) and all((value[k] is False for k in ('completion_authority', 'proof_authority', 'accepted_index_published'))) and (type(value['source_obligations_discharged']) is int) and (value['source_obligations_discharged'] == 0))
    rows = value['rounds']
    _need(type(rows) is list and 1 <= len(rows) <= 2)
    sessions = set()
    for number, row in enumerate(rows, 1):
        _need(type(row) is dict and set(row) == ROW_FIELDS and (type(row['round_number']) is int) and (row['round_number'] == number) and (type(row['process_started']) is bool or (normalized and row['process_started'] is None)) and (type(row['wait_completed']) is bool) and (type(row['retirement_observed']) is bool) and (row['returncode'] is None or (type(row['returncode']) is int and -255 <= row['returncode'] <= 255)) and _digest(row['descriptor_sha256']))
        lifecycle_name(row['session_id'])
        _need(row['session_id'] not in sessions)
        sessions.add(row['session_id'])
        if normalized:
            _need(not row['wait_completed'] or (row['process_started'] is True and row['returncode'] is not None))
            _need(not row['retirement_observed'] or (row['process_started'] is True and row['returncode'] is not None))
    if public:
        from diagnostic_public_validation_contract import validate_public_validation_observation
        observations = value['public_validation_observations']
        _need(type(observations) is list and len(observations) <= len(rows) and (value['owner_validation_outcome'] in (None, 'passed', 'failed', 'inconclusive')) and (value['owner_returncode'] is None or (type(value['owner_returncode']) is int and -255 <= value['owner_returncode'] <= 255)))
        for row, observed in zip(rows, observations):
            validate_public_validation_observation(observed)
            _need(row['retirement_observed'] is True and all((observed[key] == row[key] for key in ('round_number', 'session_id', 'descriptor_sha256'))) and (observed['actual_returncode'] == row['returncode']))
        _need(value['owner_validation_outcome'] == (observations[-1]['status'] if observations else None))
        if value['failure_type'] is None:
            actual = rows[-1]['returncode']
            expected = 1 if actual == 0 and value['owner_validation_outcome'] != 'passed' else actual
            _need(type(value['owner_returncode']) is int and value['owner_returncode'] == expected)
        else:
            _need(value['owner_returncode'] is None)
    return value
