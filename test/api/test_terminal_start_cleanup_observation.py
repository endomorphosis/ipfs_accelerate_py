"""Closed repair diagnostics bind native START evidence without granting authority."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
from ipfs_accelerate_py.agent_supervisor.control.control_plane import MutationTransactionPhase, MutationRecoveryAction
from test.api.test_lifecycle_start_recovery import _interrupted
from test.api.test_agent_supervisor_lifecycle_orchestrator import _request


@pytest.fixture
def repaired(tmp_path, monkeypatch):
    profile, _clock, adapter, owner, request, transaction = _interrupted(tmp_path, monkeypatch)
    proof = owner.repair_start_cleanup(request, transaction, timeout_ms=100)
    control = replace(transaction, phase=MutationTransactionPhase.REPAIRED,
                      revision=transaction.revision + 1, recovery_action=MutationRecoveryAction.NONE)
    state = Path(profile.state_root)
    proof_path = state / 'start-cleanup-process-proof-receipt.json'
    control_path = state / 'start-cleanup-repair-receipt.json'
    proof_path.write_text(json.dumps(proof))
    control_path.write_text(json.dumps(control.to_dict()))
    runtime = SimpleNamespace(state=state, _requests={request.request_id: request}, profile=profile, orchestrator=owner)
    return SimpleNamespace(runtime=runtime, start={'status': 'conflict', 'request_id': request.request_id},
                           proof=proof_path, control=control_path, owner=owner, request=request, adapter=adapter)


def _observe(case):
    value = driver._start_cleanup_observation(case.runtime, case.start)
    assert value['completion_authority'] is value['retry_authority'] is value['execution_authority'] is False
    assert value['absence_scope'] == 'recorded_marker_bound_tree'
    assert all(key not in value for key in ('request_id', 'transaction_id', 'transition_id', 'message', 'body'))
    return value


def test_native_receipts_bind_original_start_after_new_stop(repaired):
    stopped = repaired.owner.stop(_request(repaired.runtime.profile, operation=Operation.STOP, key='stop:observation'))
    assert stopped.succeeded
    value = _observe(repaired)
    assert value['status'] == 'available'
    assert value['proof_observation'] == value['control_observation'] == 'observed'
    assert value['lifecycle_phase'] == 'failed' and value['control_phase'] == 'repaired'
    assert value['marker_bound_process_tree_absent'] is True and value['start_succeeded'] is False
    assert repaired.adapter.launches == repaired.adapter.terminations == 1


@pytest.mark.parametrize('missing', ['proof', 'control'])
def test_partial_evidence_is_explicit_without_inferred_absence(repaired, missing):
    getattr(repaired, missing).unlink()
    value = _observe(repaired)
    assert value['status'] == 'partial' and value[missing + '_observation'] == 'missing'
    assert value['marker_bound_process_tree_absent'] is (None if missing == 'proof' else True)
    assert value['control_phase'] == (None if missing == 'control' else 'repaired')


@pytest.mark.parametrize('field,value', [
    ('schema', 'foreign'), ('transition_id', 'foreign'), ('request_id', 'foreign'),
    ('transaction_id', 'foreign'), ('phase', 'completed'), ('process_tree_absent', False),
    ('process_tree_absent', 1), ('start_succeeded', True), ('completion_authority', True),
    ('private_body', 'never exported'),
])
def test_proof_rejects_foreign_malformed_or_authoritative_claims(repaired, field, value):
    raw = json.loads(repaired.proof.read_text())
    raw[field] = value
    repaired.proof.write_text(json.dumps(raw))
    observed = _observe(repaired)
    assert observed['proof_observation'] == 'invalid'
    assert observed['marker_bound_process_tree_absent'] is None
    assert observed['status'] == 'partial'
    assert 'never exported' not in json.dumps(observed)


@pytest.mark.parametrize('field,value', [
    ('schema', 'foreign'), ('transaction_id', 'foreign'), ('request_id', 'foreign'),
    ('phase', 'committed'), ('contract_version', True), ('private_body', 'never exported'),
])
def test_control_rejects_foreign_and_malformed_receipts(repaired, field, value):
    raw = json.loads(repaired.control.read_text())
    raw[field] = value
    repaired.control.write_text(json.dumps(raw))
    observed = _observe(repaired)
    assert observed['control_observation'] == 'invalid'
    assert observed['control_phase'] is None and observed['status'] == 'partial'


@pytest.mark.parametrize('kind', ['duplicate', 'nan', 'list', 'truncated', 'oversized', 'symlink', 'fifo'])
def test_unsafe_proof_storage_does_not_block_observation(repaired, tmp_path, kind):
    if kind == 'duplicate':
        repaired.proof.write_text('{"schema":"first","schema":"last"}')
    elif kind == 'nan':
        repaired.proof.write_text('{"value":NaN}')
    elif kind == 'list':
        repaired.proof.write_text('[]')
    elif kind == 'truncated':
        repaired.proof.write_text('{')
    elif kind == 'oversized':
        repaired.proof.write_bytes(b' ' * 65537)
    else:
        original = repaired.proof.read_bytes()
        repaired.proof.unlink()
        if kind == 'symlink':
            target = tmp_path / 'private-receipt'
            target.write_bytes(original)
            repaired.proof.symlink_to(target)
        else:
            os.mkfifo(repaired.proof)
    observed = _observe(repaired)
    assert observed['proof_observation'] == 'invalid' and observed['status'] == 'partial'


@pytest.mark.parametrize('kind', ['missing', 'malformed', 'foreign', 'divergent'])
def test_journal_must_contain_exact_unambiguous_native_start(repaired, kind):
    path = repaired.owner.store.path
    if kind == 'missing':
        path.unlink()
    elif kind == 'malformed':
        path.write_text('{')
    elif kind == 'foreign':
        rows = path.read_text().splitlines()
        path.write_text('\n'.join(row.replace(repaired.request.request_id, 'foreign-request') for row in rows) + '\n')
    else:
        latest = json.loads(path.read_text().splitlines()[-1])
        latest['failure_code'] = 'foreign'
        with path.open('a') as stream:
            stream.write(json.dumps(latest) + '\n')
    observed = _observe(repaired)
    assert observed['proof_observation'] == ('missing' if kind == 'missing' else 'invalid')
    assert observed['marker_bound_process_tree_absent'] is None


@pytest.mark.parametrize('kind', ['successful', 'missing-request', 'query-error'])
def test_original_failed_start_is_required(repaired, kind):
    if kind == 'successful':
        repaired.start['status'] = 'succeeded'
        reason = 'no_failed_start'
    elif kind == 'missing-request':
        repaired.runtime._requests.clear()
        reason = 'original_start_unavailable'
    else:
        class Broken:
            def get(self, _key):
                raise OSError('private diagnostic')
        repaired.runtime._requests = Broken()
        reason = 'collection_unavailable'
    observed = _observe(repaired)
    assert observed['status'] == 'unavailable' and observed['reason'] == reason
    assert 'private diagnostic' not in json.dumps(observed)


def test_final_path_symlink_swap_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'receipt'
    path.write_text('{}')
    target = tmp_path / 'retained'
    original = Path.lstat

    def swap(selected):
        if selected == path and not target.exists():
            path.rename(target)
            path.symlink_to(target)
        return original(selected)

    monkeypatch.setattr(Path, 'lstat', swap)
    with pytest.raises(ValueError, match='changed while reading'):
        driver._bounded_repair_bytes(path, 65536)


def test_diagnostic_failure_cannot_suppress_driver_custody_close(tmp_path, monkeypatch):
    from test.api.test_terminal_doctor_dispatch import test_driver_selects_before_owner_and_preserves_provider_accounting as exercise
    observed = []
    run = driver.run

    def record(**kwargs):
        result = run(**kwargs)
        observed.append(result)
        return result

    def unavailable(*_args):
        raise OSError('private diagnostic')

    monkeypatch.setattr(driver, 'run', record)
    monkeypatch.setattr(driver, '_start_cleanup_observation', unavailable)
    exercise(tmp_path, monkeypatch, 'full', 'model_router', False, 'supervisor-semantic-router-input@1')
    assert len(observed) == 1
    assert observed[0]['runtime_close'] == {'attempted': True, 'succeeded': True}
    assert observed[0]['native_diagnostics_error'] == 'OSError'
