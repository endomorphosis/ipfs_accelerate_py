"""An opt-in owner assignment gates the real native source-proxy library.

The retained allocation fixture has authored transport/boundary adapters. These
tests exercise its real typed SQL owner, coordinator and lifecycle; they do not
qualify a container identity handoff or model dispatch.
"""
from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_prelaunch import (
    capture_repository_world_prelaunch_assignment,
)
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_broker as broker
from test.api.test_current_source_task_admission import _payload
from test.api.test_current_source_worktree_allocation import _allocation_case
from test.api.test_repository_world_prelaunch import provision_authored_prelaunch_sources


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True) as case:
        native = case.native
        native.binding = case.guard.bind_current_source_lookup(**case.source_binding)
        provision_authored_prelaunch_sources(case)
        assignment = capture_repository_world_prelaunch_assignment(
            task_admission=native.handle, worktree_admission=case.guard,
            prompt=native.factory['prompt'])
        yield case, assignment


def start(case, assignment, **changes):
    options = {**case.native.factory, 'world_prelaunch_assignment': assignment, **changes}
    case.native.session = broker.start_source_lookup_worker_broker(**options)
    return case.native.session


def test_original_assignment_allows_existing_read_only_native_source_proxy(prepared):
    case, assignment = prepared
    session = start(case, assignment)
    result = session._read_source(_payload(case.native))
    assert result['source_complete'] is True
    assert result['working_source_verified'] is True
    assert result['completion_authority'] is False
    assert session._native_response_bytes > 0
    assert session._world_prelaunch_assignment is assignment
    assert 'world_prelaunch_assignment' not in session.descriptor
    observation = assignment.require_current()
    assert observation['world_dispatch_ready'] is False
    assert observation['compiled_context_verified'] is False


@pytest.mark.parametrize('change', ['inert_observation', 'different_prompt'])
def test_invalid_assignment_refuses_before_any_source_proxy_grant(prepared, change):
    case, assignment = prepared
    gateway = case.native.server._command_gateway
    with gateway._grants_lock:
        before = set(gateway._grants)
    options = {}
    if change == 'inert_observation':
        assignment = deepcopy(assignment.require_current())
    else:
        options['prompt'] = 'A different caller-supplied input'
    with pytest.raises(ValueError):
        start(case, assignment, **options)
    with gateway._grants_lock:
        assert set(gateway._grants) == before
    assert case.native.session is None


@pytest.mark.parametrize('change', ['omit', 'inert_observation', 'prompt_bytes'])
def test_gate_cannot_be_removed_or_changed_before_native_read(prepared, change):
    case, assignment = prepared
    session = start(case, assignment)
    if change == 'omit':
        session._world_prelaunch_assignment = None
    elif change == 'inert_observation':
        session._world_prelaunch_assignment = assignment.require_current()
    else:
        session._world_prelaunch_prompt = b'different bytes'
    with pytest.raises(ValueError):
        session._read_source(_payload(case.native))
    assert session._native_response_bytes == 0


def test_native_revision_change_refuses_before_backend_source_delivery(prepared):
    case, assignment = prepared
    session = start(case, assignment)
    with case.native.server._lock:
        case.native.server._connection.execute(
            'UPDATE tasks SET revision = revision + 1 WHERE task_cid = ?',
            [case.native.task_cid])
    with pytest.raises(ValueError):
        session._read_source(_payload(case.native))
    assert session._native_response_bytes == 0


def test_worker_write_permission_transition_does_not_upgrade_world_readiness(prepared):
    case, assignment = prepared
    session = start(case, assignment)
    original_mode = case.native.workspace.stat().st_mode & 0o7777
    try:
        case.native.workspace.chmod(0o3770)
        with pytest.raises(ValueError):
            session._read_source(_payload(case.native))
        assert session._native_response_bytes == 0
    finally:
        case.native.workspace.chmod(original_mode)
