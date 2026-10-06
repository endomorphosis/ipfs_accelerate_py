"""Bounded terminal failure observations; no dispatch or settlement authority."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

MAX_BRIDGE_BYTES = 32768
MAX_STDERR_BYTES = 65536


def _read(path: Path, limit: int) -> bytes:
    if path.resolve(strict=True) != path.absolute():
        raise ValueError('canonical diagnostic path required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= limit:
            raise ValueError('bounded regular diagnostic file required')
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns)
    if (not stat.S_ISREG(current.st_mode) or len(raw) > limit
            or identity(before) != identity(after) or identity(after) != identity(current)):
        raise ValueError('diagnostic file changed during observation')
    return raw


def _object(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate diagnostic key')
            result[key] = value
        return result

    def constant(_value):
        raise ValueError('nonfinite diagnostic value')

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    if type(value) is not dict:
        raise ValueError('diagnostic object required')
    return value


def _observation(status, *, diagnostic=None, scope, matched_record_count=0):
    return {'status': status, 'diagnostic': diagnostic, 'scope': scope,
        'matched_record_count': matched_record_count, 'observation_only': True,
        'provider_dispatch_observed': None, 'completion_authority': False,
        'retry_authority': False, 'settlement_authority': False}


def bridge_failure(state: Path, *, task_cid, attempt_id_sha256=None, phase=None):
    """Read the fixed atomic sidecar for the exact admitted task and attempt.

    Native database events remain authoritative. This bounded mirror exists
    even when optional legacy JSONL event projections are disabled.
    """
    scope = 'exact_admitted_task_and_attempt'
    result = lambda status, **kwargs: _observation(status, scope=scope, **kwargs)
    try:
        if (type(task_cid) is not str or not 0 < len(task_cid.encode()) <= 1024
                or type(attempt_id_sha256) is not str
                or re.fullmatch('[0-9a-f]{64}', attempt_id_sha256) is None
                or phase not in {None, 'unknown_callback', 'terminal_failure'}):
            return result('unavailable')
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.bridge_failure_diagnostics import (
            validate_bridge_failure_observation,
        )
        raw = _read(state / 'run' / 'bridge-failure-observation.json', MAX_BRIDGE_BYTES)
        try:
            value = validate_bridge_failure_observation(_object(raw))
        except (ValueError, UnicodeError, RecursionError):
            value = None
        if value is None:
            return result('invalid')
        if (value['task_cid_sha256'] != hashlib.sha256(task_cid.encode()).hexdigest()
                or value['attempt_id_sha256'] != attempt_id_sha256
                or (phase is not None and value['diagnostic']['phase'] != phase)):
            return result('missing')
        return result('observed', diagnostic=value['diagnostic'], matched_record_count=1)
    except FileNotFoundError:
        return result('missing')
    except Exception:
        return result('unavailable')


def planner_failure(state: Path):
    """Retain closed child-reported preflight metadata without claiming dispatch."""
    result = lambda status, **kwargs: _observation(status, scope='planner_child_stderr', **kwargs)
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import validate_runner_error_envelope
        raw = _read(state / 'planner-trace.stderr', MAX_STDERR_BYTES)
        candidates, malformed = [], False
        for row in raw.splitlines(keepends=True):
            if not row.endswith(b'\n') or len(row) > 16384:
                malformed = malformed or row.lstrip().startswith(b'{')
                continue
            try:
                value = _object(row)
            except (ValueError, UnicodeError, RecursionError):
                malformed = malformed or row.lstrip().startswith(b'{')
                continue
            if value.get('schema') == 'router-implementation-error@2':
                candidates.append(value)
        if len(candidates) > 1:
            return result('ambiguous', matched_record_count=len(candidates))
        if malformed:
            return result('invalid', matched_record_count=len(candidates))
        if not candidates:
            return result('missing')
        try:
            diagnostic = validate_runner_error_envelope(candidates[0])
        except ValueError:
            return result('invalid', matched_record_count=1)
        return result('observed', diagnostic=diagnostic, matched_record_count=1)
    except FileNotFoundError:
        return result('missing')
    except Exception:
        return result('unavailable')


def collect(state: Path, report, *, native_state: Path | None = None):
    """Observe outer planner stderr and the created runtime's separate native state.

    No native path is inferred when runtime construction never completed.
    """
    task = report.get('task_state') if type(report.get('task_state')) is dict else {}
    native = report.get('native_progress') if type(report.get('native_progress')) is dict else {}
    latest = native.get('latest') if type(native.get('latest')) is dict else {}
    observed = latest.get('task') if type(latest.get('task')) is dict else {}
    receipt = observed.get('completion_receipt') if type(observed.get('completion_receipt')) is dict else {}
    terminal = receipt.get('failure_kind') == 'terminal_portal_bridge_error'
    consistent = (observed.get('task_cid') == task.get('task_cid')
        and observed.get('status') == task.get('status') and observed.get('revision') == task.get('revision')
        and type(receipt.get('attempt_id_sha256')) is str
        and re.fullmatch('[0-9a-f]{64}', receipt['attempt_id_sha256']) is not None)
    return {'schema': 'terminal-native-failure-observations@1',
        'bridge': bridge_failure(native_state, task_cid=task.get('task_cid'),
            attempt_id_sha256=receipt.get('attempt_id_sha256'),
            phase='terminal_failure' if terminal else None) if consistent and native_state is not None else
                _observation('unavailable', scope='exact_admitted_task_and_attempt'),
        'planner_child': planner_failure(state), 'observation_only': True,
        'completion_authority': False, 'retry_authority': False, 'settlement_authority': False}
