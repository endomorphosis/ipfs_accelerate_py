"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""
from __future__ import annotations
import math
PUBLIC_VALIDATION_PROFILE = 'largest-eigenval-public@1'
PUBLIC_ACCEPTANCE_RESULT_SCHEMA = 'terminal-public-acceptance-result@1'
PUBLIC_CHECKS = ('structural', 'eigenpair', 'performance')
PUBLIC_VALIDATION_KEYS = ('public-structural-smoke', 'public-eigenpair-behavior', 'public-eigenvalue-performance')
PROFILE = '.supervisor-task-profile.json'
INSTRUCTION = '.supervisor-instruction.md'
SMOKE = '.supervisor-public-smoke.py'
PUBLIC_TIMING_POLICY = '\ndef _timing_status(rows):\n    for size in (2, 4, 6, 8, 10):\n        selected = [row for row in rows if row["size"] == size]\n        for key in ("candidate_median_seconds", "reference_median_seconds"):\n            values = [row[key] for row in selected]\n            if (max(values) - min(values)) / min(values) > 0.20:\n                return "inconclusive", "timing_unstable"\n    faster = [row["candidate_median_seconds"] < row["reference_median_seconds"] for row in rows]\n    if all(faster):\n        return "passed", "faster_than_reference"\n    # Any consistently non-faster size contradicts "consistently runs faster".\n    # A crossing within any size is measurement uncertainty, never a failure.\n    for size in (2, 4, 6, 8, 10):\n        selected = [row["candidate_median_seconds"] < row["reference_median_seconds"]\n                    for row in rows if row["size"] == size]\n        if any(selected) and not all(selected):\n            return "inconclusive", "timing_unstable"\n    return "failed", "not_faster_than_reference"\n'

def public_validation_result(value: object, *, check: str) -> dict:
    """Copy a bounded public observation; it grants no execution authority."""
    if check not in PUBLIC_CHECKS or type(value) is not dict or set(value) != {'schema', 'profile', 'check', 'status', 'reason', 'measurements'} or (value['schema'] != PUBLIC_ACCEPTANCE_RESULT_SCHEMA) or (value['profile'] != PUBLIC_VALIDATION_PROFILE) or (value['check'] != check):
        raise ValueError('closed public acceptance result required')
    pairs = {('passed', 'obligations_satisfied'), ('failed', 'output_structure_invalid')} if check == 'structural' else {('passed', 'obligations_satisfied'), ('failed', 'candidate_behavior_invalid')}
    if check == 'performance':
        pairs = {('passed', 'faster_than_reference'), ('failed', 'not_faster_than_reference'), ('failed', 'candidate_behavior_invalid'), ('inconclusive', 'timing_unstable'), ('inconclusive', 'timing_unavailable')}
    if type(value['status']) is not str or type(value['reason']) is not str or (value['status'], value['reason']) not in pairs or (type(value['measurements']) is not list):
        raise ValueError('public acceptance status or reason differs')
    rows = value['measurements']
    measured = check == 'performance' and value['reason'] not in {'candidate_behavior_invalid', 'timing_unavailable'}
    if not measured:
        if rows:
            raise ValueError('unmeasured public check cannot carry measurements')
        return {**value, 'measurements': []}
    expected = [(size, batch) for size in (2, 4, 6, 8, 10) for batch in range(3)]
    if len(rows) != len(expected):
        raise ValueError('public timing requires all fixed size and batch measurements')
    for row, (size, batch) in zip(rows, expected):
        if type(row) is not dict or set(row) != {'size', 'batch', 'candidate_median_seconds', 'reference_median_seconds'} or type(row['size']) is not int or (row['size'] != size) or (type(row['batch']) is not int) or (row['batch'] != batch):
            raise ValueError('public timing measurement population differs')
        for key in ('candidate_median_seconds', 'reference_median_seconds'):
            number = row[key]
            if type(number) not in (int, float) or not math.isfinite(number) or (not 0 < number <= 3600):
                raise ValueError('public timing measurements must be positive finite bounded numbers')
    namespace = {}
    exec(PUBLIC_TIMING_POLICY, namespace)
    if (value['status'], value['reason']) != namespace['_timing_status'](rows):
        raise ValueError('public timing outcome differs from unrounded measurements')
    return {**value, 'measurements': [dict(row) for row in rows]}
