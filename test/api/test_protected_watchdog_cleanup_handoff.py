"""Socket-driven watchdog cleanup with real CAS and private inode transitions.

Docker absence is an explicit observation double; these tests issue no Docker
commands and do not establish live provider or candidate-release authority.
"""

from __future__ import annotations

import copy
import os
import socket
import tempfile
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.control import provider_attempt_store as cas
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner
from test.api.test_llm_router_agent_supervisor_fallback_route import (
    _discard_live_cleanup_inputs,
    _effect_detail_identity,
    _live_cleanup_launch_context,
    _refresh_effect_detail_identities,
    _terminal_watchdog_arguments,
)


@pytest.fixture
def custody(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    run_root = state_root / "run"
    run_root.mkdir(parents=True, mode=0o700)
    for name in runner._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
        monkeypatch.setenv(name, "fixture")
    for name, value in {
        runner.STATE_ROOT_ENV: str(state_root),
        runner.RUN_ROOT_ENV: str(run_root),
        runner.REPOSITORY_ROOT_ENV: str(tmp_path),
        runner.FENCING_EPOCH_ENV: "1",
    }.items():
        monkeypatch.setenv(name, value)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    with monkeypatch.context() as mp:
        mp.setattr(tempfile, "tempdir", str(private))
        context, paths = _live_cleanup_launch_context()
    cleanup = context["cleanup_receipt"]
    cleanup["watchdog_pid"] = os.getpid()
    cleanup["watchdog_start_ticks"] = runner._runner_process_start_ticks(os.getpid())
    cleanup.pop("receipt_id")
    cleanup["receipt_id"] = _effect_detail_identity(cleanup)
    context["cleanup_id"] = cleanup["receipt_id"]
    _refresh_effect_detail_identities(context)
    store = cas.DurableProviderAttemptCAS(tmp_path / "attempts")
    started = store.reserve_or_adopt(
        logical_attempt_id="attempt:watchdog-handoff",
        route_id="route:fixture",
        decision_id="decision:fixture",
        task_id="task:fixture",
        worktree_id="worktree:fixture",
        authorized=True,
        launch_context=context,
    )
    binding_path = runner._docker_cleanup_binding_path(str(context["container_name"]))
    assert binding_path is not None
    observations = []

    def remove(**kw):
        observations.append(kw)

    monkeypatch.setattr(runner, "_remove_exact_docker_container", remove)
    monkeypatch.setattr(runner.signal, "signal", lambda *_: None)
    case = SimpleNamespace(
        store=store,
        started=started,
        context=context,
        paths=paths,
        binding_path=binding_path,
        observations=observations,
        observation={
            "logical_attempt_id": started.reservation.logical_attempt_id,
            "provider_attempt_store": str(store.directory),
            "provider_attempt_store_identity": store.directory_identity,
        },
    )
    try:
        yield case
    finally:
        _discard_live_cleanup_inputs(paths)


def _terminal(case):
    binding = runner._read_private_control_record(
        case.binding_path.parent, case.binding_path.name
    )
    assert binding is not None
    reservation = case.started.reservation
    result = case.store.complete(
        reservation,
        returncode=125,
        outcome={
            "decision": "effect_not_created",
            "fallback_dispatched": False,
            "fallback_returncode": 125,
            "reservation_id": reservation.reservation_id,
            "effect_launch_receipt": reservation.effect_launch_receipt,
        },
        completion_capability=case.started.completion_capability,
        terminal_cleanup_evidence={
            "binding_path": str(case.binding_path),
            "binding_record_id": binding["record_id"],
            "termination_fence_id": "",
        },
    )
    case.binding = binding
    case.binding_identity = runner._cleanup_path_identity(
        case.binding_path, directory=False
    )
    return result


def _watchdog(case, ready):
    watchdog_socket, peer = socket.socketpair()

    class Control:
        def fileno(self):
            return watchdog_socket.fileno()

        def getsockopt(self, *args):
            return watchdog_socket.getsockopt(*args)

        def sendall(self, payload):
            watchdog_socket.sendall(payload)

        def shutdown(self, how):
            watchdog_socket.shutdown(how)

        def close(self):
            watchdog_socket.close()

        def recv(self, size):
            if not hasattr(self, "prepared"):
                self.prepared = True
                assert peer.recv(128).startswith(str(os.getpid()).encode() + b":")
                ready()
                peer.shutdown(socket.SHUT_WR)
            return watchdog_socket.recv(size)

    args = _terminal_watchdog_arguments(case.context, case.paths)
    args += [
        "--cleanup-binding-record",
        str(case.binding_path),
        "--logical-attempt-id",
        case.observation["logical_attempt_id"],
        "--provider-attempt-store",
        case.observation["provider_attempt_store"],
        "--provider-attempt-store-identity",
        case.observation["provider_attempt_store_identity"],
    ]
    try:
        return runner._docker_cleanup_watchdog_main(args, control_socket=Control())
    finally:
        watchdog_socket.close()
        peer.close()


def test_actual_watchdog_commits_cleanup_progress_before_retiring_inputs(custody):
    case = custody
    assert _watchdog(case, lambda: _terminal(case)) == 0
    observed = case.store.observe(case.started.reservation.logical_attempt_id)
    assert observed.terminal_cleanup_progress.get("phase") == "completion_committed"
    assert not case.binding_path.exists()
    assert all(
        not case.paths[name].exists()
        for name in ("lease_root", "prompt_path", "provider_home")
    )
    reopened = cas.DurableProviderAttemptCAS(
        case.store.directory,
        expected_directory_identity=case.store.directory_identity,
        create_if_missing=False,
    )
    assert reopened.observe(observed.logical_attempt_id) == observed


def _markers(case):
    for name in ("cas-owned", "cas-terminal"):
        path = case.paths["lease_root"] / name
        path.write_text("marker-only")
        path.chmod(0o600)


def _assert_inputs(case):
    for name in ("lease_root", "prompt_path", "provider_home"):
        assert case.paths[name].exists()


@pytest.mark.parametrize("fault", ["unfinished", "unknown", "missing", "wrong_binding"])
def test_watchdog_terminal_marker_never_replaces_native_proof(
    custody, monkeypatch, fault
):
    case = custody
    prior = None
    observe = cas.DurableProviderAttemptCAS.observe

    def ready():
        nonlocal prior
        if fault == "wrong_binding":
            r = case.started.reservation
            case.store.complete(
                r,
                returncode=125,
                outcome={
                    "decision": "effect_not_created",
                    "fallback_dispatched": False,
                    "fallback_returncode": 125,
                    "reservation_id": r.reservation_id,
                    "effect_launch_receipt": r.effect_launch_receipt,
                },
                completion_capability=case.started.completion_capability,
                terminal_cleanup_evidence={
                    "binding_path": str(case.binding_path),
                    "binding_record_id": "sha256:" + "a" * 64,
                    "termination_fence_id": "",
                },
            )
        prior = observe(case.store, case.started.reservation.logical_attempt_id)
        if fault == "unknown":

            def unavailable(*_a, **_kw):
                raise ValueError("observation unavailable")

            monkeypatch.setattr(cas.DurableProviderAttemptCAS, "observe", unavailable)
        elif fault == "missing":
            # A missing observation cannot promote terminal markers. The
            # original exact native reservation remains untouched.
            monkeypatch.setattr(
                cas.DurableProviderAttemptCAS, "observe", lambda *_a: None
            )
        _markers(case)

    assert _watchdog(case, ready) == 125
    _assert_inputs(case)
    assert case.binding_path.exists()
    assert case.observations == []
    assert observe(case.store, case.started.reservation.logical_attempt_id) == prior


def test_watchdog_intent_write_failure_preserves_original_inputs(custody, monkeypatch):
    case = custody

    def fail(*_a, **_kw):
        raise ValueError("intent persistence failed")

    monkeypatch.setattr(
        cas.DurableProviderAttemptCAS, "commit_terminal_cleanup_intent", fail
    )
    assert _watchdog(case, lambda: _terminal(case)) == 125
    _assert_inputs(case)
    assert case.binding_path.exists()
    assert (
        case.store.observe(
            case.started.reservation.logical_attempt_id
        ).terminal_cleanup_progress
        == {}
    )


def test_watchdog_completion_write_failure_replays_only_exact_precommitted_intent(
    custody, monkeypatch
):
    case = custody
    original = cas.DurableProviderAttemptCAS.commit_terminal_cleanup_completion

    def fail(*_a, **_kw):
        raise ValueError("completion persistence failed")

    monkeypatch.setattr(
        cas.DurableProviderAttemptCAS, "commit_terminal_cleanup_completion", fail
    )
    assert _watchdog(case, lambda: _terminal(case)) == 125
    terminal = case.store.observe(case.started.reservation.logical_attempt_id)
    assert terminal.terminal_cleanup_progress["phase"] == "intent_committed"
    assert case.binding_path.exists()
    monkeypatch.setattr(
        cas.DurableProviderAttemptCAS, "commit_terminal_cleanup_completion", original
    )
    reopened = cas.DurableProviderAttemptCAS(
        case.store.directory,
        expected_directory_identity=case.store.directory_identity,
        create_if_missing=False,
    )
    admitted = runner._admit_bound_cleanup_terminal(
        reopened, binding_record=case.binding, local_cas_owned=True
    )

    def replay():
        return runner._finalize_verified_cleanup_completion(
            binding_path=case.binding_path,
            binding_identity=case.binding_identity,
            binding_record=case.binding,
            terminal_cleanup_store=reopened,
            terminal_cleanup_reservation=admitted,
        )

    assert replay()
    completed = reopened.observe(terminal.logical_attempt_id)
    assert completed.terminal_cleanup_progress["phase"] == "completion_committed"
    assert replay()
    assert reopened.observe(terminal.logical_attempt_id) == completed


def _publish_prepared(case):
    binding = runner._docker_cleanup_binding_value(
        binding_state="prepared_no_dispatch",
        provider="codex",
        docker_bin="/usr/bin/docker",
        container_name=case.context["container_name"],
        lease_root=case.paths["lease_root"],
        docker_config=case.paths["docker_config"],
        cidfile=case.paths["cidfile"],
        provider_home=case.paths["provider_home"],
        prompt_path=case.paths["prompt_path"],
        effect_observation=case.observation,
        path_identities={
            name: runner._cleanup_path_identity(
                case.paths[name], directory=name != "prompt_path"
            )
            for name in ("docker_config", "lease_root", "prompt_path", "provider_home")
        },
        binding_path=case.binding_path,
        runner_pid=os.getpid(),
        runner_start_ticks=runner._runner_process_start_ticks(os.getpid()),
        watchdog_pid=os.getpid(),
        watchdog_start_ticks=runner._runner_process_start_ticks(os.getpid()),
    )
    runner._write_private_control_record(
        case.binding_path.parent,
        case.binding_path.name,
        binding,
        replace_existing=False,
    )
    return binding


def test_runner_close_fallback_records_same_scoped_completion(custody):
    case = custody
    _publish_prepared(case)
    _terminal(case)
    lease = object.__new__(runner._DockerContainerLease)
    for name, value in case.paths.items():
        setattr(lease, name, value)
    lease._closed = False
    lease.preserve_for_recovery = False
    lease._cas_owned = True
    lease._cas_terminal = True
    lease.provider = "codex"
    lease.docker_bin = "/usr/bin/docker"
    lease.container_name = case.context["container_name"]
    lease.engine_endpoint = runner._DOCKER_LOCAL_HOST
    lease.cleanup_binding_record = case.binding_path
    lease._cleanup_binding_identity = case.binding_identity
    lease._cleanup_binding_value = case.binding
    lease._cleanup_path_identities = case.binding["path_identities"]
    lease._termination_fence = {}
    lease._runner_pid = os.getpid()
    lease._runner_start_ticks = runner._runner_process_start_ticks(os.getpid())
    lease.effect_observation = case.observation
    lease._watchdog = SimpleNamespace(
        pid=os.getpid(),
        start_ticks=runner._runner_process_start_ticks(os.getpid()),
        wait=lambda **_: 0,
    )
    lease._abort_provider_start = lambda: None
    left, right = socket.socketpair()
    lease._control_socket = left
    try:
        lease.close(docker_run_finished=True)
    finally:
        right.close()
    assert not lease.preserve_for_recovery
    assert (
        case.store.observe(
            case.started.reservation.logical_attempt_id
        ).terminal_cleanup_progress["phase"]
        == "completion_committed"
    )


@pytest.mark.parametrize(
    "field",
    [
        "watchdog_start_ticks",
        "lease_root",
        "logical_attempt_id",
        "provider_attempt_store_identity",
    ],
)
def test_exact_native_terminal_does_not_authorize_foreign_binding(custody, field):
    case = custody
    binding = _publish_prepared(case)
    terminal = _terminal(case)
    foreign = copy.deepcopy(binding)
    if field == "provider_attempt_store_identity":
        foreign["effect_observation"][field] = "sha256:" + "a" * 64
    elif field == "logical_attempt_id":
        foreign["effect_observation"][field] = "attempt:other"
    elif field == "watchdog_start_ticks":
        foreign[field] += 1
    else:
        foreign[field] += ".other"
    with pytest.raises(ValueError):
        runner._admit_bound_cleanup_terminal(
            case.store, binding_record=foreign, local_cas_owned=True
        )
    _assert_inputs(case)
    assert case.store.observe(terminal.logical_attempt_id) == terminal


def test_current_nonterminal_winner_is_not_inert_even_without_marker(custody):
    case = custody
    binding = _publish_prepared(case)
    with pytest.raises(ValueError, match="CAS authority drifted"):
        runner._admit_bound_cleanup_terminal(
            case.store, binding_record=binding, local_cas_owned=False
        )
    _assert_inputs(case)
    assert (
        case.store.observe(case.started.reservation.logical_attempt_id)
        == case.started.reservation
    )


def test_prepared_losing_lease_cleanup_keeps_foreign_winner_untouched(custody):
    case = custody
    binding = _publish_prepared(case)
    # The owner-issued winning receipt names another exact watchdog birth.
    # This local prepared lease has no CAS-owned marker and no start fence.
    losing = copy.deepcopy(binding)
    losing["watchdog_start_ticks"] += 1
    assert (
        runner._admit_bound_cleanup_terminal(
            case.store, binding_record=losing, local_cas_owned=False
        )
        is None
    )
    assert (
        case.store.observe(case.started.reservation.logical_attempt_id)
        == case.started.reservation
    )
    _assert_inputs(case)


def test_command_bound_native_terminal_requires_exact_fence_and_dispatch(custody):
    """Fenced binding/CAS/dispatch join, with Docker absence explicitly doubled."""
    from test.api.test_agent_supervisor_grok_quota_terra_gate import (
        _created_docker_termination_fence,
    )

    case = custody
    binding = _publish_prepared(case)
    fence = _created_docker_termination_fence(
        container_name=case.context["container_name"],
        container_id=str(case.context["container_id"]).removeprefix("sha256:"),
        image_id=case.context["image_id"],
    )
    binding.update(
        binding_state="command_bound",
        create_command_id="sha256:" + "a" * 64,
        create_cwd=str(case.paths["lease_root"].parent),
        create_environment_id="sha256:" + "b" * 64,
        termination_fence=fence,
    )
    binding.pop("record_id")
    binding["record_id"] = runner._effect_receipt_identity(binding)
    runner._write_private_control_record(
        case.binding_path.parent, case.binding_path.name, binding, replace_existing=True
    )
    identity = runner._cleanup_path_identity(case.binding_path, directory=False)
    r = case.started.reservation
    terminal = case.store.complete(
        r,
        returncode=78,
        outcome={
            "decision": "fallback_failed",
            "fallback_dispatched": True,
            "fallback_returncode": 78,
            "reservation_id": r.reservation_id,
            "effect_launch_receipt": r.effect_launch_receipt,
        },
        completion_capability=case.started.completion_capability,
        terminal_cleanup_evidence={
            "binding_path": str(case.binding_path),
            "binding_record_id": binding["record_id"],
            "termination_fence_id": fence["fence_id"],
        },
    )
    admitted = runner._admit_bound_cleanup_terminal(
        case.store, binding_record=binding, local_cas_owned=True
    )
    assert admitted == terminal
    wrong = copy.deepcopy(binding)
    wrong["termination_fence"] = {}
    with pytest.raises(ValueError, match="private binding changed"):
        runner._admit_bound_cleanup_terminal(
            case.store, binding_record=wrong, local_cas_owned=True
        )

    def finalize():
        return runner._finalize_verified_cleanup_completion(
            binding_path=case.binding_path,
            binding_identity=identity,
            binding_record=binding,
            terminal_cleanup_store=case.store,
            terminal_cleanup_reservation=admitted,
        )

    assert (
        not finalize()
    )  # Exact termination fence is insufficient without removal-dispatch evidence.
    _assert_inputs(case)
    assert case.store.observe(r.logical_attempt_id) == terminal
    birth = runner.read_process_birth(os.getpid())
    dispatch = runner._docker_removal_dispatch_value(
        binding_path=case.binding_path,
        binding_record=binding,
        termination_fence=fence,
        issuer_process_birth={
            "pid": os.getpid(),
            "start_time_ticks": birth.start_time_ticks,
            "boot_id": birth.boot_id,
            "parent_pid": os.getppid(),
        },
        state="request_completed",
        generation=1,
        previous_dispatch_id="sha256:" + "c" * 64,
        docker_returncode=0,
        failure_kind="",
    )
    dispatch_path = runner._docker_removal_dispatch_path(case.binding_path)
    runner._write_private_control_record(
        dispatch_path.parent, dispatch_path.name, dispatch, replace_existing=False
    )
    assert finalize()
    completed = case.store.observe(r.logical_attempt_id)
    assert completed.terminal_cleanup_progress["phase"] == "completion_committed"
    assert (
        completed.terminal_cleanup_progress["intent"]["docker_absence"]["kind"]
        == "fenced_effect_absence"
    )
    assert finalize()
    assert case.store.observe(r.logical_attempt_id) == completed


def test_watchdog_cannot_recreate_disappeared_native_store(custody):
    case = custody
    directory = case.store.directory
    held = directory.with_name(directory.name + ".preserved")
    directory.rename(held)
    try:
        assert (
            _watchdog(case, lambda: pytest.fail("unbound owner entered control loop"))
            == 2
        )
        assert not directory.exists()
        _assert_inputs(case)
    finally:
        held.rename(directory)
    assert (
        case.store.observe(case.started.reservation.logical_attempt_id)
        == case.started.reservation
    )
