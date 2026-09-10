"""Current runner closure bodies against real disposable provider attempt CAS.

Extract only the actual nested closures so prerequisite quota/CLI setup does
not replace the storage API under test. Docker observation and signed-route
admission are explicit test boundaries; these are not live route qualifications.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.control import provider_attempt_store as cas
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner
from test.api.test_llm_router_agent_supervisor_fallback_route import (
    _discard_live_cleanup_inputs,
    _fixture_terminal_cleanup_evidence,
    _live_cleanup_launch_context,
    _protected_effect_launch_context,
    _recorded_effect_inspection,
)


def _closure(name, namespace):
    tree = ast.parse(Path(runner.__file__).read_text())
    matches = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(matches) == 1
    node = copy.deepcopy(matches[0])
    # Preserve captured assignments in this test's namespace without copying
    # or editing any executable statements from the production closure.
    node.body = [
        ast.Global(n.names) if isinstance(n, ast.Nonlocal) else n for n in node.body
    ]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, runner.__file__, "exec"), namespace)  # noqa: S102 - execute exact checked-in closure
    return namespace[name]


@pytest.fixture
def producer(tmp_path, monkeypatch):
    store = cas.DurableProviderAttemptCAS(tmp_path / "attempts")
    started = store.reserve_or_adopt(
        logical_attempt_id="attempt:producer",
        route_id="route:producer",
        decision_id="decision:producer",
        task_id="task:producer",
        worktree_id="worktree:producer",
        authorized=True,
        launch_context=_protected_effect_launch_context(workspace=str(tmp_path)),
    )
    ns = dict(vars(runner))
    ns.update(
        attempt_store=store,
        attempt_reservation=started.reservation,
        completion_capability=started.completion_capability,
        completed_terminal_outcome=None,
        captured_capacity_receipt={},
        outcome_route=None,
        preflight_receipt={},
        preflight_quota_evidence=None,
        effect_verifier_status="confirmed_quota",
        invocation_binding=None,
        is_eaaef_route=False,
        workspace=tmp_path,
        prompt="test",
        worker_network_attempt_authority_json="",
        ProviderAttemptStoreError=cas.ProviderAttemptStoreError,
        ProviderAttemptReservation=cas.ProviderAttemptReservation,
        render_route_outcome_record=lambda value: "recorded-terminal-result",
    )
    events = []

    def outcome(**kwargs):
        reservation = kwargs["reservation"]
        return {
            "decision": kwargs["decision"],
            "fallback_dispatched": kwargs["fallback_dispatched"],
            "fallback_returncode": kwargs["fallback_returncode"],
            "reservation_id": reservation.reservation_id,
            "decision_id": reservation.decision_id,
            "effect_launch_receipt": reservation.effect_launch_receipt,
            "effect_adoption_receipt": reservation.effect_adoption_receipt,
            "effect_quarantine_receipt": reservation.quarantine_receipt,
            "effect_quarantine_terminalization_receipt": reservation.quarantine_terminalization_receipt,
        }

    def evidence(launch):
        assert (
            launch
            == store.read(started.reservation.logical_attempt_id).effect_launch_receipt
        )
        events.append("evidence")
        return _fixture_terminal_cleanup_evidence(launch, fallback_dispatched=True)

    def cleanup(launch, *, terminal_observer, terminal_reservation):
        assert terminal_observer is store
        assert terminal_reservation == store.read(
            started.reservation.logical_attempt_id
        )
        assert terminal_reservation.terminal
        assert launch == terminal_reservation.effect_launch_receipt
        assert terminal_reservation.terminal_cleanup_authority
        events.append("cleanup")

    ns.update(
        route_outcome_record=outcome,
        _recorded_codex_terminal_cleanup_evidence=evidence,
        _recorded_codex_terminal_capacity_evidence=lambda _launch: {},
        _release_recorded_codex_effect_cleanup=cleanup,
        _start_recorded_codex_effect=lambda *a, **kw: pytest.fail(
            "unexpected redispatch"
        ),
    )
    monkeypatch.setattr(
        cas,
        "_inspect_recorded_docker_effect",
        lambda launch, at: _recorded_effect_inspection(
            launch, at, status="exited", returncode=78
        ),
    )
    return store, started, ns, events


@pytest.mark.parametrize("returncode", [0, 78])
def test_normal_terminal_captures_required_cleanup_authority(producer, returncode):
    store, started, ns, events = producer
    _closure("complete_provider_effect", ns)(returncode)
    terminal = store.read(started.reservation.logical_attempt_id)
    assert terminal.terminal_returncode == returncode
    assert terminal.terminal_cleanup_authority
    assert terminal.terminal_cleanup_progress == {}
    assert events == ["evidence"]
    assert ns["completed_terminal_outcome"] == terminal.terminal_outcome
    # Persisted authority survives reopening; no cleanup completion invented.
    reopened = cas.DurableProviderAttemptCAS(store.directory)
    assert reopened.read(terminal.logical_attempt_id) == terminal


def test_adopted_terminal_supplies_current_scoped_cleanup(producer):
    store, started, ns, events = producer
    result = _closure("adopt_started_effect", ns)(
        started.reservation, winner_capability=started.completion_capability
    )
    assert result == 78
    terminal = store.read(started.reservation.logical_attempt_id)
    assert terminal.terminal_returncode == 78
    assert terminal.effect_adoption_generation == 1
    assert terminal.terminal_cleanup_authority
    assert events == ["evidence", "cleanup"]


@pytest.mark.parametrize("name", ["complete_provider_effect", "adopt_started_effect"])
def test_missing_verified_cleanup_evidence_preserves_unfinished_effect(producer, name):
    store, started, ns, events = producer

    def missing(_launch):
        events.append("evidence_missing")
        raise ValueError("exact cleanup binding unavailable")

    ns["_recorded_codex_terminal_cleanup_evidence"] = missing
    callback = _closure(name, ns)
    with pytest.raises(ValueError, match="exact cleanup binding unavailable"):
        if name == "complete_provider_effect":
            callback(0)
        else:
            callback(
                started.reservation, winner_capability=started.completion_capability
            )
    assert store.read(started.reservation.logical_attempt_id).state == "effect_started"
    assert events == ["evidence_missing"]


def test_invalid_cleanup_evidence_never_commits_terminal(producer):
    store, started, ns, events = producer
    ns["_recorded_codex_terminal_cleanup_evidence"] = lambda _launch: {}
    with pytest.raises(
        cas.ProviderAttemptStoreError, match="cleanup evidence is incomplete"
    ):
        _closure("complete_provider_effect", ns)(0)
    assert store.read(started.reservation.logical_attempt_id) == started.reservation
    assert events == []


def test_unknown_adoption_inspection_never_cleans_or_terminalizes(
    producer, monkeypatch
):
    store, started, ns, events = producer
    monkeypatch.setattr(
        cas,
        "_inspect_recorded_docker_effect",
        lambda launch, at: _recorded_effect_inspection(launch, at, status="unknown"),
    )
    with pytest.raises(cas.ProviderAttemptStoreError):
        _closure("adopt_started_effect", ns)(
            started.reservation, winner_capability=started.completion_capability
        )
    assert store.read(started.reservation.logical_attempt_id) == started.reservation
    assert events == []


def test_absent_effect_cannot_reuse_dispatched_cleanup_fence(producer, monkeypatch):
    store, started, ns, events = producer
    monkeypatch.setattr(
        cas,
        "_inspect_recorded_docker_effect",
        lambda launch, at: _recorded_effect_inspection(launch, at, status="absent"),
    )
    with pytest.raises(
        cas.ProviderAttemptStoreError, match="cleanup evidence is invalid"
    ):
        _closure("adopt_started_effect", ns)(
            started.reservation, winner_capability=started.completion_capability
        )
    assert store.read(started.reservation.logical_attempt_id).state == "effect_started"
    assert events == ["evidence"]


def _terminal_fixture(store, started, ns):
    result = ns["route_outcome_record"](
        reservation=started.reservation,
        decision="fallback_succeeded",
        fallback_dispatched=True,
        fallback_returncode=0,
    )
    return store.complete(
        started.reservation,
        returncode=0,
        outcome=result,
        completion_capability=started.completion_capability,
        terminal_cleanup_evidence=_fixture_terminal_cleanup_evidence(
            started.reservation.effect_launch_receipt, fallback_dispatched=True
        ),
    )


def test_already_terminal_adoption_replays_exact_cleanup(producer):
    store, started, ns, events = producer
    terminal = _terminal_fixture(store, started, ns)
    assert _closure("adopt_started_effect", ns)(terminal) == 0
    assert store.read(terminal.logical_attempt_id) == terminal
    assert events == ["cleanup"]


@pytest.mark.parametrize("cleanup_error", [None, FileNotFoundError, ValueError])
def test_actual_run_early_terminal_replay_forwards_exact_store(
    producer, monkeypatch, tmp_path, cleanup_error
):
    store, started, ns, events = producer
    terminal = _terminal_fixture(store, started, ns)
    invocation = SimpleNamespace(
        logical_attempt_id=terminal.logical_attempt_id,
        provider_attempt_store=str(store.directory),
        provider_attempt_store_identity=store.directory_identity,
        control_plane="sealed-fixture",
    )
    route = SimpleNamespace(invocation_binding=invocation)
    context = SimpleNamespace(
        route=route,
        decision=SimpleNamespace(content_id=terminal.decision_id),
        failure_receipt={},
    )
    monkeypatch.setattr(
        runner, "_parse_codex_fallback_command", lambda *a, **kw: ["trusted-codex"]
    )
    monkeypatch.setattr(
        runner, "validate_grok_runner_command_binding", lambda _value: False
    )
    monkeypatch.setattr(
        cas,
        "DurableProviderAttemptCAS",
        lambda path, **kw: (
            store
            if (
                path == str(store.directory)
                and kw["expected_directory_identity"] == store.directory_identity
            )
            else pytest.fail("foreign store")
        ),
    )
    monkeypatch.setattr(
        llm_router, "resolve_agent_implementation_route", lambda **kw: route
    )
    monkeypatch.setattr(
        llm_router, "resolve_agent_implementation_route_binding", lambda *a, **kw: route
    )
    monkeypatch.setattr(
        llm_router,
        "parse_agent_implementation_effect_authorization_context",
        lambda *a, **kw: context,
    )
    monkeypatch.setattr(
        llm_router,
        "verify_agent_implementation_sealed_control_plane",
        lambda control, fd: "/proc/self/fd/99",
    )
    monkeypatch.setattr(
        llm_router, "valid_agent_implementation_route_outcome", lambda *a, **kw: True
    )
    monkeypatch.setattr(
        llm_router,
        "render_agent_implementation_route_outcome",
        lambda _value: "terminal",
    )

    def cleanup(*args, **kwargs):
        ns["_release_recorded_codex_effect_cleanup"](*args, **kwargs)
        if cleanup_error is not None:
            raise cleanup_error("cleanup proof unavailable")

    monkeypatch.setattr(runner, "_release_recorded_codex_effect_cleanup", cleanup)
    monkeypatch.setattr(runner.sys, "argv", ["/proc/self/fd/99"])
    args = SimpleNamespace(
        codex_fallback_command_json="[]",
        codex_fallback_reasoning_effort="high",
        outer_runner_command="",
        canonical_legacy_preflight_route=False,
        grok_failure_receipt_nonce="fixture-nonce",
        agent_implementation_route_json="{}",
        workspace=tmp_path,
        receipt_fd_declared=False,
        agent_implementation_recovery_json="",
        worker_network_attempt_authority_json="",
    )
    assert runner._run(args, -1) == (0 if cleanup_error is None else 2)
    assert events == ["cleanup"]
    assert store.read(terminal.logical_attempt_id) == terminal


def test_failed_terminal_proof_preserves_actual_private_cleanup_inputs(
    tmp_path, monkeypatch
):
    launch, paths = _live_cleanup_launch_context()
    store = cas.DurableProviderAttemptCAS(tmp_path / "attempts")
    started = store.reserve_or_adopt(
        logical_attempt_id="attempt:preserved",
        route_id="route:preserved",
        decision_id="decision:preserved",
        task_id="task:preserved",
        worktree_id="worktree:preserved",
        authorized=True,
        launch_context=launch,
    )
    lease = object.__new__(runner._DockerContainerLease)
    lease._closed = False
    lease._cas_owned = True
    lease._cas_terminal = False
    lease.preserve_for_recovery = False
    lease.cleanup_binding_record = None
    lease._control_socket = SimpleNamespace(close=lambda: None)
    lease._abort_provider_start = lambda: None
    cleanup = started.reservation.effect_launch_receipt["cleanup_receipt"]
    lease.lease_root = Path(cleanup["lease_root"])
    lease.docker_config = Path(cleanup["docker_config"])
    lease.container_name = started.reservation.effect_launch_receipt["container_name"]
    lease._watchdog = SimpleNamespace(
        pid=cleanup["watchdog_pid"], start_ticks=cleanup["watchdog_start_ticks"]
    )
    lease.effect_observation = {
        "provider_attempt_store": str(store.directory),
        "provider_attempt_store_identity": store.directory_identity,
        "logical_attempt_id": started.reservation.logical_attempt_id,
    }
    monkeypatch.setattr(
        runner,
        "_remove_owned_cleanup_path",
        lambda *a, **kw: pytest.fail("failed proof must preserve cleanup inputs"),
    )
    try:
        identities = {
            name: (p.stat().st_dev, p.stat().st_ino) for name, p in paths.items()
        }
        with pytest.raises(
            cas.ProviderAttemptStoreError, match="cleanup evidence is incomplete"
        ):
            store.complete(
                started.reservation,
                returncode=0,
                outcome={
                    "reservation_id": started.reservation.reservation_id,
                    "effect_launch_receipt": started.reservation.effect_launch_receipt,
                    "fallback_dispatched": True,
                    "fallback_returncode": 0,
                },
                completion_capability=started.completion_capability,
            )
        lease.close(docker_run_finished=False)
        assert lease.preserve_for_recovery is True
        assert store.read(started.reservation.logical_attempt_id) == started.reservation
        assert {
            name: (p.stat().st_dev, p.stat().st_ino) for name, p in paths.items()
        } == identities
    finally:
        _discard_live_cleanup_inputs(paths)
