"""Focused lifecycle tests for the managed Quack owner watchdog."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_owner_watchdog import (
    QUACK_OWNER_WATCHDOG_INTERFACE,
    AuthenticatedReadiness,
    DesiredOwnerState,
    OwnerHealth,
    ProcessTreeTerminationReceipt,
    QuackOwnerBinding,
    QuackOwnerObservation,
    QuackOwnerStartAbsentError,
    QuackOwnerWatchdog,
    QuackOwnerWatchdogPolicy,
    SpawnedQuackOwner,
    WatchdogDisposition,
    process_birth_id,
    terminate_spawned_owner,
)


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("quack-owner", timeout)
        return self.returncode


def _birth(pid: int = 421, *, ticks: int = 9001) -> ProcessBirthIdentity:
    return ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=ticks,
        boot_id="boot-a",
        parent_pid=37,
    )


def _binding(*, generation: int = 7, database_uuid: str = "db-a") -> QuackOwnerBinding:
    return QuackOwnerBinding(
        store_id="control.duckdb",
        schema_revision="dqp-control-plane-v1",
        database_uuid=database_uuid,
        schema_fingerprint="sha256:schema-a",
        generation=generation,
        server_id=f"server-{generation}",
    )


def _dead(
    birth: ProcessBirthIdentity | None = None,
    *,
    binding: QuackOwnerBinding | None = None,
    absence_proven: bool = False,
) -> QuackOwnerObservation:
    return QuackOwnerObservation(
        process_birth=birth,
        liveness=OwnerLiveness.DEAD,
        absence_proven=absence_proven,
        binding=binding,
    )


def _alive(
    birth: ProcessBirthIdentity,
    *,
    binding: QuackOwnerBinding,
    healthy: bool = True,
    authenticated: bool = True,
) -> QuackOwnerObservation:
    return QuackOwnerObservation(
        process_birth=birth,
        liveness=OwnerLiveness.ALIVE,
        health=(OwnerHealth.HEALTHY if healthy else OwnerHealth.UNHEALTHY),
        authenticated_ready=(authenticated and healthy),
        binding=binding,
    )


def _readiness(
    birth: ProcessBirthIdentity,
    *,
    binding: QuackOwnerBinding,
    authenticated: bool = True,
    ready: bool = True,
) -> AuthenticatedReadiness:
    return AuthenticatedReadiness(
        authenticated=authenticated,
        ready=ready,
        process_birth_id=process_birth_id(birth),
        binding=binding,
    )


def _termination(
    owner: SpawnedQuackOwner,
    *,
    confirmed: bool = True,
) -> ProcessTreeTerminationReceipt:
    return ProcessTreeTerminationReceipt(
        process_birth_id=owner.birth_id,
        termination_confirmed=confirmed,
        terminate_sent=True,
        reason="test_process_tree_terminated" if confirmed else "test_refused",
    )


def _watchdog(
    tmp_path: Path,
    *,
    start_owner,
    readiness_probe,
    expected_binding: QuackOwnerBinding | None = None,
    observe_owner=None,
    terminate_owner=None,
    policy: QuackOwnerWatchdogPolicy | None = None,
    clock=None,
    desired_state: DesiredOwnerState = DesiredOwnerState.RUNNING,
) -> QuackOwnerWatchdog:
    return QuackOwnerWatchdog(
        lock_path=tmp_path / "quack-owner-recovery.lock",
        start_owner=start_owner,
        readiness_probe=readiness_probe,
        expected_binding=expected_binding or _binding(),
        observe_owner=observe_owner,
        terminate_owner=terminate_owner,
        policy=policy,
        clock=clock or FakeClock(),
        desired_state=desired_state,
    )


def test_interface_and_module_do_not_depend_on_duckdb() -> None:
    import inspect

    import ipfs_accelerate_py.agent_supervisor.runtime.quack_owner_watchdog as module

    assert QUACK_OWNER_WATCHDOG_INTERFACE == "QuackOwnerWatchdog@1"
    assert QuackOwnerWatchdog.INTERFACE == QUACK_OWNER_WATCHDOG_INTERFACE
    source = inspect.getsource(module)
    assert "open_duckdb_connection" not in source
    assert "import duckdb" not in source


def test_desired_stopped_never_starts_and_stays_stopped(tmp_path: Path) -> None:
    starts: list[str] = []
    watchdog = _watchdog(
        tmp_path,
        start_owner=lambda: starts.append("started"),
        readiness_probe=lambda _owner: pytest.fail("readiness must not run"),
        desired_state=DesiredOwnerState.STOPPED,
    )

    receipt = watchdog.ensure(_dead(_birth(), binding=_binding(generation=6)))

    assert receipt.disposition is WatchdogDisposition.INTENTIONALLY_STOPPED
    assert receipt.recovered is False
    assert starts == []
    assert watchdog.status()["desired_state"] == "stopped"


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (
            QuackOwnerObservation(
                process_birth=_birth(),
                liveness=OwnerLiveness.ALIVE,
                health=OwnerHealth.UNHEALTHY,
                binding=_binding(),
            ),
            WatchdogDisposition.ABSTAIN_ALIVE_UNHEALTHY,
        ),
        (
            QuackOwnerObservation(
                process_birth=_birth(),
                liveness=OwnerLiveness.UNKNOWN,
                binding=_binding(),
            ),
            WatchdogDisposition.ABSTAIN_UNKNOWN,
        ),
        (
            _dead(binding=_binding(generation=6)),
            WatchdogDisposition.ABSTAIN_NOT_PROVABLY_DEAD,
        ),
    ],
)
def test_uncertain_or_live_unhealthy_owner_fails_closed(
    tmp_path: Path,
    observation: QuackOwnerObservation,
    expected: WatchdogDisposition,
) -> None:
    starts: list[str] = []
    watchdog = _watchdog(
        tmp_path,
        start_owner=lambda: starts.append("started"),
        readiness_probe=lambda _owner: pytest.fail("readiness must not run"),
    )

    receipt = watchdog.ensure(observation)

    assert receipt.disposition is expected
    assert receipt.restart_attempted is False
    assert receipt.recovered is False
    assert starts == []


def test_authenticated_identity_bound_readiness_is_required_before_recovered(
    tmp_path: Path,
) -> None:
    prior = _binding(generation=6)
    current = _binding(generation=7)
    birth = _birth(501)
    owner = SpawnedQuackOwner(FakeProcess(birth.pid), birth)
    call_order: list[str] = []

    def start_owner() -> SpawnedQuackOwner:
        call_order.append("start")
        return owner

    def readiness_probe(value: SpawnedQuackOwner) -> AuthenticatedReadiness:
        assert value is owner
        call_order.append("authenticated-readiness")
        return _readiness(birth, binding=current)

    watchdog = _watchdog(
        tmp_path,
        start_owner=start_owner,
        readiness_probe=readiness_probe,
        expected_binding=current,
    )

    receipt = watchdog.ensure(_dead(_birth(500), binding=prior))

    assert call_order == ["start", "authenticated-readiness"]
    assert receipt.disposition is WatchdogDisposition.RESTARTED
    assert receipt.authenticated_readiness is True
    assert receipt.operational_ready is True
    assert receipt.recovered is True
    assert receipt.binding == current
    assert watchdog.current_owner() is owner
    assert watchdog.retry_state.consecutive_failures == 0


def test_database_identity_or_generation_mismatch_is_killed_and_not_recovered(
    tmp_path: Path,
) -> None:
    prior = _binding(generation=6)
    expected = _binding(generation=7)
    wrong = _binding(generation=7, database_uuid="replacement-db")
    birth = _birth(510)
    owner = SpawnedQuackOwner(FakeProcess(birth.pid), birth)
    terminations: list[str] = []
    watchdog = _watchdog(
        tmp_path,
        start_owner=lambda: owner,
        readiness_probe=lambda _owner: _readiness(birth, binding=wrong),
        expected_binding=expected,
        terminate_owner=lambda value: (
            terminations.append(value.birth_id) or _termination(value)
        ),
    )

    receipt = watchdog.ensure(_dead(_birth(509), binding=prior))

    assert receipt.disposition is WatchdogDisposition.READINESS_FAILED
    assert receipt.recovered is False
    assert receipt.operational_ready is False
    assert receipt.termination is not None
    assert receipt.termination.termination_confirmed is True
    assert terminations == [owner.birth_id]
    assert watchdog.current_owner() is None


def test_callback_exception_text_and_secret_never_enter_status_or_receipt(
    tmp_path: Path,
) -> None:
    prior = _binding(generation=6)
    current = _binding(generation=7)
    birth = _birth(520)
    owner = SpawnedQuackOwner(FakeProcess(birth.pid), birth)
    secret = "super-secret-quack-token-value"

    def leaking_probe(_owner: SpawnedQuackOwner) -> AuthenticatedReadiness:
        raise RuntimeError(f"Bearer {secret}")

    watchdog = _watchdog(
        tmp_path,
        start_owner=lambda: owner,
        readiness_probe=leaking_probe,
        expected_binding=current,
        terminate_owner=lambda value: _termination(value),
    )

    receipt = watchdog.ensure(_dead(_birth(519), binding=prior))
    serialized = json.dumps(
        {"receipt": receipt.to_dict(), "status": watchdog.status()},
        sort_keys=True,
    )

    assert receipt.disposition is WatchdogDisposition.READINESS_FAILED
    assert secret not in serialized
    assert "Bearer" not in serialized
    assert str(tmp_path) not in serialized


def test_backoff_and_retry_budget_are_bounded(tmp_path: Path) -> None:
    clock = FakeClock()
    starts: list[int] = []

    def fail_start():
        starts.append(len(starts) + 1)
        raise QuackOwnerStartAbsentError("no process was created")

    watchdog = _watchdog(
        tmp_path,
        start_owner=fail_start,
        readiness_probe=lambda _owner: pytest.fail("no owner was returned"),
        policy=QuackOwnerWatchdogPolicy(
            max_restart_attempts=3,
            initial_backoff_seconds=2.0,
            maximum_backoff_seconds=3.0,
            backoff_multiplier=2.0,
        ),
        clock=clock,
    )
    observation = _dead(_birth(), binding=_binding(generation=6))

    first = watchdog.ensure(observation)
    backoff = watchdog.ensure(observation)
    clock.advance(2.0)
    second = watchdog.ensure(observation)
    clock.advance(3.0)
    third = watchdog.ensure(observation)
    exhausted = watchdog.ensure(observation)

    assert first.disposition is WatchdogDisposition.START_FAILED
    assert backoff.disposition is WatchdogDisposition.BACKOFF
    assert second.disposition is WatchdogDisposition.START_FAILED
    assert third.disposition is WatchdogDisposition.START_FAILED
    assert exhausted.disposition is WatchdogDisposition.RETRY_EXHAUSTED
    assert starts == [1, 2, 3]
    assert watchdog.retry_state.consecutive_failures == 3
    assert watchdog.retry_state.exhausted is True
    assert watchdog.retry_state.retry_not_before == pytest.approx(108.0)


def test_unknown_start_outcome_exhausts_without_second_owner(tmp_path: Path) -> None:
    starts: list[int] = []

    def unknown_start():
        starts.append(len(starts) + 1)
        raise RuntimeError("callback may have spawned an owner")

    watchdog = _watchdog(
        tmp_path,
        start_owner=unknown_start,
        readiness_probe=lambda _owner: pytest.fail("no owner was returned"),
    )
    observation = _dead(_birth(), binding=_binding(generation=6))

    first = watchdog.ensure(observation)
    second = watchdog.ensure(observation)

    assert first.disposition is WatchdogDisposition.START_FAILED
    assert first.reason == "start_outcome_unknown"
    assert second.disposition is WatchdogDisposition.RETRY_EXHAUSTED
    assert starts == [1]


def test_unfenced_failed_readiness_exhausts_without_contender(
    tmp_path: Path,
) -> None:
    prior = _binding(generation=6)
    expected = _binding(generation=7)
    birth = _birth(525)
    starts: list[int] = []

    def start_owner() -> SpawnedQuackOwner:
        starts.append(len(starts) + 1)
        return SpawnedQuackOwner(FakeProcess(birth.pid), birth)

    watchdog = _watchdog(
        tmp_path,
        start_owner=start_owner,
        readiness_probe=lambda _owner: _readiness(
            birth,
            binding=expected,
            authenticated=False,
        ),
        expected_binding=expected,
        terminate_owner=lambda owner: _termination(owner, confirmed=False),
    )
    observation = _dead(_birth(524), binding=prior)

    first = watchdog.ensure(observation)
    second = watchdog.ensure(observation)

    assert first.disposition is WatchdogDisposition.READINESS_FAILED
    assert first.retry_state.last_failure_code == "spawned_owner_not_fenced"
    assert second.disposition is WatchdogDisposition.RETRY_EXHAUSTED
    assert starts == [1]


def test_exclusive_recovery_lock_elects_one_concurrent_winner(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    current = _binding(generation=7)
    prior = _binding(generation=6)
    birth = _birth(530)
    owner = SpawnedQuackOwner(FakeProcess(birth.pid), birth)
    starts: list[str] = []

    def blocking_start() -> SpawnedQuackOwner:
        starts.append("winner")
        entered.set()
        assert release.wait(timeout=5.0)
        return owner

    common = {
        "start_owner": blocking_start,
        "readiness_probe": lambda _owner: _readiness(birth, binding=current),
        "expected_binding": current,
    }
    first = _watchdog(tmp_path, **common)
    second = _watchdog(tmp_path, **common)
    observation = _dead(_birth(529), binding=prior)
    first_result: list = []

    thread = threading.Thread(target=lambda: first_result.append(first.ensure(observation)))
    thread.start()
    assert entered.wait(timeout=5.0)
    loser = second.ensure(observation)
    release.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert loser.disposition is WatchdogDisposition.LOCK_CONTENDED
    assert len(first_result) == 1
    assert first_result[0].disposition is WatchdogDisposition.RESTARTED
    assert starts == ["winner"]


def test_reobservation_under_lock_prevents_stale_dead_restart(tmp_path: Path) -> None:
    current = _binding(generation=7)
    live_birth = _birth(540)
    starts: list[str] = []
    watchdog = _watchdog(
        tmp_path,
        start_owner=lambda: starts.append("unsafe-stale-start"),
        readiness_probe=lambda _owner: pytest.fail("must not probe"),
        expected_binding=current,
        observe_owner=lambda: _alive(live_birth, binding=current),
    )

    receipt = watchdog.ensure(
        _dead(_birth(539), binding=_binding(generation=6))
    )

    assert receipt.disposition is WatchdogDisposition.HEALTHY
    assert receipt.operational_ready is True
    assert starts == []


def test_explicit_live_owner_adoption_and_intentional_stop(tmp_path: Path) -> None:
    binding = _binding(generation=7)
    birth = _birth(550)
    owner = SpawnedQuackOwner(None, birth, adopted=True)
    stopped: list[str] = []
    watchdog = _watchdog(
        tmp_path,
        start_owner=lambda: pytest.fail("must not start"),
        readiness_probe=lambda _owner: pytest.fail("probe supplied explicitly"),
        expected_binding=binding,
        terminate_owner=lambda value: (
            stopped.append(value.birth_id) or _termination(value)
        ),
    )

    adoption = watchdog.adopt_owner(
        owner,
        _alive(birth, binding=binding),
        _readiness(birth, binding=binding),
    )
    stop = watchdog.stop()
    after = watchdog.ensure(
        _dead(birth, binding=binding)
    )

    assert adoption.disposition is WatchdogDisposition.ADOPTED
    assert watchdog.current_owner() is None
    assert stop.disposition is WatchdogDisposition.STOPPED
    assert stop.termination is not None and stop.termination.termination_confirmed
    assert stopped == [owner.birth_id]
    assert after.disposition is WatchdogDisposition.INTENTIONALLY_STOPPED


def test_safe_terminator_refuses_unknown_or_reused_process_birth() -> None:
    expected = _birth(560)
    reused = _birth(560, ticks=expected.start_time_ticks + 1)
    owner = SpawnedQuackOwner(FakeProcess(expected.pid), expected)
    signals: list[tuple[int, int]] = []

    receipt = terminate_spawned_owner(
        owner,
        birth_reader=lambda _pid: reused,
        group_id_reader=lambda pid: pid,
        group_signal=lambda pid, sig: signals.append((pid, sig)),
    )

    assert receipt.termination_confirmed is False
    assert receipt.reason == "process_birth_unknown_or_reused"
    assert signals == []


def test_safe_terminator_signals_only_exact_isolated_process_group() -> None:
    birth = _birth(570)
    owner = SpawnedQuackOwner(FakeProcess(birth.pid), birth)
    present = {"value": True}
    signals: list[tuple[int, int]] = []

    def read_birth(_pid: int) -> ProcessBirthIdentity | None:
        return birth if present["value"] else None

    def send(group: int, signum: int) -> None:
        signals.append((group, signum))
        if signum == 15:
            present["value"] = False

    receipt = terminate_spawned_owner(
        owner,
        birth_reader=read_birth,
        group_id_reader=lambda pid: pid,
        group_signal=send,
        monotonic=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )

    assert receipt.termination_confirmed is True
    assert receipt.terminate_sent is True
    assert receipt.kill_sent is False
    assert signals == [(birth.pid, 15)]
