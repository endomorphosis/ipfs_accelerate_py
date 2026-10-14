"""Bind-once nomination custody across real native claims and allocations.

Git/CAS, native claims, grants, Portal projection and lifecycle are real. Quack
listener lifecycle and deployment UID metadata are authored host seams. No
coding child, model, target program or official verifier is dispatched.
"""
from contextlib import contextmanager
from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import current_source_worktree_allocation as allocation
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_scope as scopes
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_broker as broker
from test.api.test_current_source_task_admission import SOURCE, _payload
from test.api.test_current_source_worktree_allocation import _allocation_case


SUPPORTS = {
    b".supervisor-instruction.md": b"Authored support: inspect the selected originals.\n",
    b".supervisor-public-smoke.py": b"# Authored inert support, never executed.\n",
    b".supervisor-task-profile.json": b'{"authored_fixture":true}\n',
}


def source_files(selected_count=4):
    # These names exercise the ordinary count boundary, not a signed Terminal
    # profile. Their bytes are independent authored nomination support policy.
    files = {b"module.py": SOURCE, **SUPPORTS}
    files.update({f"selected_{index:03d}.py".encode(): b"VALUE = 1\n"
                  for index in range(selected_count - 4)})
    files[b"unselected.py"] = b"UNSELECTED = True\n"
    return files


def nomination(case, selected_count=4):
    native = case.native
    selected = tuple(sorted(name.hex() for name in source_files(selected_count)
                            if name != b"unselected.py"))
    required = tuple(sorted(name.hex() for name in SUPPORTS))
    program = tuple(name for name in selected if name not in required)
    return scopes.nominate_source_scope(cas=native.retained.cas,
        expected_inventory_cid=native.retained.ref.inventory_cid,
        request=native.retained.request, limits=native.retained.limits,
        selected_paths=selected, required_paths=required, program_paths=program)


def nominated_arguments(case, scope=None):
    scope = nomination(case) if scope is None else scope
    return {**{key: value for key, value in case.source_binding.items() if key != "allowed_paths"},
            "scope": scope, "expected_scope_cid": scope.observation()["scope_cid"]}


def bind(case, *, selected_count=4):
    scope = nomination(case, selected_count)
    case.native.binding = case.guard.bind_nominated_current_source_lookup(**nominated_arguments(case, scope))
    return scope


@pytest.fixture
def deferred(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True,
                          source_files=source_files()) as case:
        yield case


@contextmanager
def source_session(case):
    session = broker.start_source_lookup_worker_broker(**case.native.factory)
    case.native.session = session
    try:
        yield session
    finally:
        session.close()
        case.native.session = None


def assert_joined_reader(case):
    native = case.native
    service = native.server._command_gateway._current_source_lookup
    assert service is not None
    assert service is case.guard._lookup is native.handle._source
    assert service._native_source_task_admission is native.handle
    assert service._native_source_worktree_allocation is case.guard
    assert case.guard._lookup_bytes == native.handle._source_bytes
    case.guard.require_source_binding(server=native.server, service=service)
    assert case.guard.require_nominated_source_scope()["allocation_ownership_verified"] is True
    assert native.handle.require_nominated_source_scope()["allocation_ownership_verified"] is False
    return service


@pytest.mark.parametrize("outer", ["legacy", "nominated"])
@pytest.mark.parametrize("callback", ["cas", "clock"])
def test_reentrant_legacy_and_nominated_allocation_binds_cannot_replace_inflight_scope(
        deferred, monkeypatch, outer, callback):
    case = deferred
    nominated = nominated_arguments(case)
    original_get = case.native.retained.cas.get
    original_clock = type(case.native.clock).__call__
    seen = []

    def attempt_reentry():
        if seen:
            return
        seen.append(callback)
        with pytest.raises((ValueError, RuntimeError)):
            case.guard.bind_current_source_lookup(**case.source_binding)
        with pytest.raises((ValueError, RuntimeError)):
            case.guard.bind_nominated_current_source_lookup(**nominated)

    def cas_callback(cid):
        value = original_get(cid)
        attempt_reentry()
        return value

    def clock_callback(clock):
        value = original_clock(clock)
        if clock is case.native.clock:
            attempt_reentry()
        return value

    if callback == "cas": monkeypatch.setattr(case.native.retained.cas, "get", cas_callback)
    else: monkeypatch.setattr(type(case.native.clock), "__call__", clock_callback)
    if outer == "nominated":
        case.native.binding = case.guard.bind_nominated_current_source_lookup(**nominated)
        assert_joined_reader(case)
        assert case.native.binding["allowed_path_count"] == 4
    else:
        case.native.binding = case.guard.bind_current_source_lookup(**case.source_binding)
        assert case.native.binding["allowed_path_count"] == 1
        with pytest.raises((ValueError, RuntimeError)):
            case.guard.require_nominated_source_scope()
    assert seen == [callback]


@pytest.mark.parametrize("retry", ["legacy", "nominated"])
def test_prepublication_failure_leaves_allocation_and_task_bind_retryable(deferred, monkeypatch, retry):
    case = deferred
    args = nominated_arguments(case)

    def fail_before_publication(**_kwargs):
        raise RuntimeError("authored failure before native source publication")

    with monkeypatch.context() as changed:
        changed.setattr(case.native.server, "bind_current_source_lookup", fail_before_publication)
        with pytest.raises(RuntimeError, match="before native source publication"):
            case.guard.bind_nominated_current_source_lookup(**args)
    assert case.native.server._command_gateway._current_source_lookup is None
    assert case.guard._lookup is case.native.handle._source is None
    if retry == "nominated":
        case.native.binding = case.guard.bind_nominated_current_source_lookup(**args)
        assert_joined_reader(case)
    else:
        case.native.binding = case.guard.bind_current_source_lookup(**case.source_binding)
    with source_session(case) as session:
        assert session._read_source(_payload(case.native))["content"].encode() == SOURCE


def test_postpublication_exception_retains_both_handles_and_consumes_both_bind_paths(deferred, monkeypatch):
    case = deferred
    args = nominated_arguments(case)
    original = case.native.server.bind_current_source_lookup

    def publish_then_fail(**kwargs):
        original(**kwargs)
        raise RuntimeError("authored failure after native source publication")

    with monkeypatch.context() as changed:
        changed.setattr(case.native.server, "bind_current_source_lookup", publish_then_fail)
        with pytest.raises(RuntimeError, match="after native source publication"):
            case.guard.bind_nominated_current_source_lookup(**args)
    service = assert_joined_reader(case)
    case.native.binding = service.binding(case.native.server._command_gateway.identity)
    for method, arguments in ((case.guard.bind_current_source_lookup, case.source_binding),
                              (case.guard.bind_nominated_current_source_lookup, args)):
        with pytest.raises((ValueError, RuntimeError)):
            method(**arguments)
    assert case.native.server._command_gateway._current_source_lookup is service
    with source_session(case) as session:
        assert session._read_source(_payload(case.native))["content"].encode() == SOURCE


@pytest.mark.parametrize("first,second", [("nominated", "legacy"), ("nominated", "nominated"),
                                         ("legacy", "nominated")])
def test_second_allocation_cannot_adopt_or_remark_an_existing_reader(deferred, first, second):
    case = deferred
    args = nominated_arguments(case)
    if first == "nominated": bind(case)
    else: case.native.binding = case.guard.bind_current_source_lookup(**case.source_binding)
    original_service = case.native.server._command_gateway._current_source_lookup
    other = allocation.capture_native_source_worktree_allocation(**case.capture)
    method, parameters = ((other.bind_nominated_current_source_lookup, args) if second == "nominated"
                          else (other.bind_current_source_lookup, case.source_binding))
    with pytest.raises((ValueError, RuntimeError)):
        method(**parameters)
    assert other._lookup is None
    assert case.native.server._command_gateway._current_source_lookup is original_service
    assert original_service._native_source_worktree_allocation is case.guard
    assert original_service._native_source_task_admission is case.native.handle
    if first == "nominated": assert_joined_reader(case)
    with source_session(case) as session:
        assert session._read_source(_payload(case.native))["content"].encode() == SOURCE


def test_serialized_nomination_cannot_replace_exact_scope_or_consume_allocation(deferred):
    case = deferred
    args = nominated_arguments(case)
    serialized = deepcopy(args["scope"].observation())
    with pytest.raises((ValueError, RuntimeError)):
        case.guard.bind_nominated_current_source_lookup(**(args | {"scope": serialized}))
    assert case.native.server._command_gateway._current_source_lookup is None
    assert case.guard._lookup is None
    bind(case)
    assert_joined_reader(case)


@pytest.mark.parametrize("substitution", ["allocation", "task"])
def test_serialized_handle_cannot_replace_original_broker_authority(deferred, monkeypatch, substitution):
    case = deferred
    bind(case)
    factory = dict(case.native.factory)
    if substitution == "allocation": factory["worktree_admission"] = case.guard.require_current()
    else: factory["task_admission"] = case.native.handle.require_current()
    grants = []

    def forbidden_grant(**_kwargs):
        grants.append(True)
        pytest.fail("serialized authority reached native proxy grant")

    monkeypatch.setattr(case.native.server, "issue_typed_client_grant_record", forbidden_grant)
    with pytest.raises((ValueError, RuntimeError)):
        broker.start_source_lookup_worker_broker(**factory)
    assert grants == []
    assert_joined_reader(case)


@pytest.mark.parametrize("selected_count", [32, 33, 64])
def test_broker_path_bound_counts_supports_without_narrowing_nomination(
        tmp_path, monkeypatch, selected_count):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True,
                          source_files=source_files(selected_count)) as case:
        scope = bind(case, selected_count=selected_count)
        historical = scope.observation()
        assert historical["selected_count"] == selected_count
        assert len(historical["required_paths"]) == 3
        assert historical["frontier_count"] == 1
        assert historical["max_files"] == 64 and historical["max_source_bytes"] == 1_000_000
        assert case.native.binding["allowed_path_count"] == selected_count
        original_paths = case.native.server._command_gateway._current_source_lookup._paths
        assert_joined_reader(case)
        if selected_count == 32:
            with source_session(case) as session:
                assert session._read_source(_payload(case.native))["content"].encode() == SOURCE
        else:
            grants = []

            def forbidden_grant(**_kwargs):
                grants.append(True)
                pytest.fail("oversized worker scope reached native proxy grant")

            monkeypatch.setattr(case.native.server, "issue_typed_client_grant_record", forbidden_grant)
            with pytest.raises((ValueError, RuntimeError), match="path population exceeds profile"):
                broker.start_source_lookup_worker_broker(**case.native.factory)
            assert grants == []
            assert case.native.server._command_gateway._current_source_lookup._paths == original_paths
            assert len(original_paths) == selected_count
            assert not list(case.native.factory["boundary"].parent.glob("worker-observations/source-*.json"))
            assert_joined_reader(case)
        assert case.native.daemon.provider_invocation_recorded(case.native.attempt.attempt_id,
            idempotency_key="provider:" + case.native.attempt.attempt_id) is None
