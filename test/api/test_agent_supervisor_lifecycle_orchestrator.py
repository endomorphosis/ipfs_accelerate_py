from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlBounds,
    EffectKind,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
    JsonlControlStateStore,
    PartialMutationError,
    StaleLeaseError,
    SupervisorControlService,
    TransactionConflictError,
)
from ipfs_accelerate_py.agent_supervisor.lifecycle_orchestrator import (
    LifecycleAction,
    LifecycleOrchestrator,
    LifecycleProfile,
    LifecycleProfileChanged,
    LifecycleSagaPhase,
    LifecycleTransitionReceipt,
    LinuxProcessAdapter,
    ProcessIdentity,
    ProcessIdentityMismatch,
    ProcessTreeNotFenced,
    ProcessTreeSnapshot,
    SplitBrainError,
)


class Clock:
    def __init__(self) -> None:
        self.value_ms = 1_000

    def now_ms(self) -> int:
        return self.value_ms

    def monotonic(self) -> float:
        return self.value_ms / 1_000

    def sleep(self, seconds: float) -> None:
        self.value_ms += max(1, int(seconds * 1_000))


class FakeProcessAdapter:
    def __init__(self, profile: LifecycleProfile, clock: Clock) -> None:
        self.profile = profile
        self.clock = clock
        self.live: dict[str, ProcessIdentity] = {}
        self.next_pid = 500
        self.launches = 0
        self.terminations = 0
        self.healthy_value = True
        self.leave_descendant = False
        self.fail_next_launch = False
        self.events: list[str] = []
        self.effect_hook: Callable[[], None] | None = None

    def identity(
        self,
        pid: int,
        *,
        parent_pid: int = 1,
        fencing_epoch: int = 9,
        profile: LifecycleProfile | None = None,
    ) -> ProcessIdentity:
        selected = profile or self.profile
        return ProcessIdentity(
            pid=pid,
            start_time_ticks=pid * 100,
            parent_pid=parent_pid,
            process_group_id=pid if parent_pid == 1 else parent_pid,
            session_id=pid if parent_pid == 1 else parent_pid,
            boot_id="boot:test",
            argv=selected.argv,
            cwd=selected.cwd,
            executable=selected.argv[0],
            run_id=selected.run_id,
            profile_id=selected.profile_id,
            target_id=selected.target_id,
            repository_root=selected.repository_root,
            state_root=selected.state_root,
            run_root=selected.run_root,
            fencing_epoch=fencing_epoch,
            configuration_root=selected.configuration_root,
        )

    def seed_tree(
        self, *, roots: int = 1, descendants: int = 1, fence: int = 9
    ) -> ProcessTreeSnapshot:
        self.live.clear()
        for _index in range(roots):
            self.next_pid += 1
            root = self.identity(self.next_pid, fencing_epoch=fence)
            self.live[root.identity_id] = root
            parent = root
            for _child_index in range(descendants):
                self.next_pid += 1
                child = self.identity(
                    self.next_pid,
                    parent_pid=parent.pid,
                    fencing_epoch=fence,
                )
                self.live[child.identity_id] = child
                parent = child
        return self.snapshot(self.profile)

    def snapshot(self, profile: LifecycleProfile) -> ProcessTreeSnapshot:
        if profile.profile_id != self.profile.profile_id:
            # A changed registry still observes the old run in production and
            # must never silently classify it as absent.
            if self.live:
                raise ProcessIdentityMismatch("foreign profile owns run")
        return ProcessTreeSnapshot(
            profile_id=profile.profile_id,
            run_id=profile.run_id,
            members=tuple(self.live.values()),
            captured_at_ms=self.clock.now_ms(),
        )

    def launch(
        self, profile: LifecycleProfile, *, fencing_epoch: int
    ) -> ProcessIdentity:
        self.events.append("launch")
        if self.effect_hook is not None:
            hook, self.effect_hook = self.effect_hook, None
            hook()
        if self.fail_next_launch:
            self.fail_next_launch = False
            raise OSError("injected launch failure")
        assert not self.live
        self.launches += 1
        self.next_pid += 1
        root = self.identity(self.next_pid, fencing_epoch=fencing_epoch)
        self.next_pid += 1
        child = self.identity(
            self.next_pid,
            parent_pid=root.pid,
            fencing_epoch=fencing_epoch,
        )
        self.live = {root.identity_id: root, child.identity_id: child}
        return root

    def terminate(
        self,
        tree: ProcessTreeSnapshot,
        *,
        grace_seconds: float,
        deadline_ms: int,
    ) -> None:
        del grace_seconds, deadline_ms
        self.events.append("terminate")
        if self.effect_hook is not None:
            hook, self.effect_hook = self.effect_hook, None
            hook()
        self.terminations += 1
        keep = ""
        if self.leave_descendant and tree.members:
            keep = tree.members[-1].identity_id
        self.live = {
            identity_id: member
            for identity_id, member in self.live.items()
            if identity_id == keep
        }

    def identity_alive(self, identity: ProcessIdentity) -> bool:
        return identity.identity_id in self.live

    def healthy(
        self,
        profile: LifecycleProfile,
        tree: ProcessTreeSnapshot,
        *,
        fencing_epoch: int,
        now_ms: int,
    ) -> bool:
        del profile, now_ms
        return (
            self.healthy_value
            and len(tree.roots) == 1
            and all(item.fencing_epoch == fencing_epoch for item in tree.members)
        )


def _profile(tmp_path: Path, *, argv: tuple[str, ...] | None = None) -> LifecycleProfile:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    return LifecycleProfile(
        target_id="supervisor:prompt",
        run_id="run:prompt",
        configuration_root="configuration:prompt:v1",
        repository_root=str(repository_root),
        state_root=str(state_root),
        run_root=str(state_root / "runs" / "prompt"),
        argv=argv or ("/usr/bin/python3", "-c", "pass"),
        cwd=str(repository_root),
        health_path=str(state_root / "status.json"),
        health_stale_ms=1_000,
    )


def _request(
    profile: LifecycleProfile,
    *,
    operation: Operation = Operation.RESTART,
    key: str = "restart:one",
    expected_revision: int = 0,
    fence: int = 9,
    lease_id: str = "lease:prompt",
    deadline_ms: int = 100,
    health_window_ms: int = 20,
) -> OperationRequest:
    effect = ExpectedEffect(
        effect_id=f"{operation.value}:process-tree",
        kind=EffectKind.LIFECYCLE_TRANSITION,
        resource=profile.target_id,
        paths=("lifecycle-transitions.jsonl",),
    )
    binding = {
        "repository_root": profile.repository_root,
        "state_root": profile.state_root,
        "repository_id": "repository:current",
        "tree_id": "tree:current",
        "objective_id": "ASI-154",
        "objective_revision": "objective:1",
        "policy_id": "policy:lifecycle",
        "policy_revision": "policy:1",
        "caller": "operator:test",
    }
    parameters = {
        "target_id": profile.target_id,
        "run_id": profile.run_id,
        "configuration_root": profile.configuration_root,
        "expected_revision": expected_revision,
        "deadline_ms": deadline_ms,
        "health_window_ms": health_window_ms,
        "reason": "test transition",
    }
    return OperationRequest(
        operation=operation,
        **binding,
        bounds=ControlBounds(timeout_ms=max(1_000, deadline_ms)),
        parameters=parameters,
        expected_effects=(effect,),
        idempotency=IdempotencyKey(
            key=key,
            operation=operation,
            caller=binding["caller"],
            repository_id=binding["repository_id"],
            objective_id=binding["objective_id"],
        ),
        authorization=AuthorizationDecision(
            verdict=AuthorizationVerdict.PERMIT,
            operation=operation,
            granted_authority=OperationAuthority.MUTATION,
            **binding,
            lease_id=lease_id,
            fencing_epoch=fence,
            authorized_effect_ids=(effect.effect_id,),
            grant_ids=("grant:lifecycle",),
            evaluated_at_ms=1,
            expires_at_ms=100_000,
        ),
        lease_id=lease_id,
        fencing_epoch=fence,
    )


def _orchestrator(
    profile: LifecycleProfile,
    adapter: FakeProcessAdapter,
    clock: Clock,
) -> LifecycleOrchestrator:
    return LifecycleOrchestrator(
        state_root=profile.state_root,
        profiles=(profile,),
        process_adapter=adapter,
        clock_ms=clock.now_ms,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        poll_interval_ms=5,
        stop_grace_ms=5,
    )


def test_restart_persists_intent_then_fences_old_tree_before_identical_start(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    old = adapter.seed_tree(descendants=2)
    orchestrator = _orchestrator(profile, adapter, clock)
    journal = Path(profile.state_root) / "lifecycle-transitions.jsonl"

    observed_phases: list[str] = []

    def inspect_intent() -> None:
        records = [
            json.loads(line)
            for line in journal.read_text(encoding="utf-8").splitlines()
        ]
        observed_phases.append(records[-1]["phase"])
        assert records[0]["phase"] == "prepared"

    adapter.effect_hook = inspect_intent
    receipt = orchestrator.restart(_request(profile))

    assert observed_phases == ["stopping_old"]
    assert adapter.events == ["terminate", "launch"]
    assert receipt.succeeded
    assert receipt.old_tree == old
    assert receipt.old_tree_fenced
    assert receipt.new_tree is not None
    assert receipt.new_tree.tree_id != old.tree_id
    assert receipt.intent.profile_id == profile.profile_id
    assert receipt.expected_effect_ids == ("restart:process-tree",)
    assert set(receipt.observed_effects) == {
        "new_process_launched",
        "old_process_tree_terminated",
        "run_fenced",
        "sustained_health_verified",
    }
    persisted = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    assert persisted["phase"] == "committed"
    assert persisted["receipt"]["receipt_id"] == receipt.receipt_id
    tampered = receipt.to_dict()
    tampered["failure_code"] = "invented"
    with pytest.raises(ValueError, match="receipt_id"):
        LifecycleTransitionReceipt.from_dict(tampered)


def test_exact_replay_returns_receipt_without_repeating_effects(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.seed_tree()
    orchestrator = _orchestrator(profile, adapter, clock)
    request = _request(profile)

    first = orchestrator.execute(request)
    second = orchestrator.execute(request)

    assert second == first
    assert adapter.terminations == 1
    assert adapter.launches == 1


def test_explicit_start_and_stop_are_bounded_verified_transitions(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    orchestrator = _orchestrator(profile, adapter, clock)

    started = orchestrator.start(
        _request(
            profile,
            operation=Operation.START,
            key="start:one",
        )
    )
    assert started.revision == 1
    assert started.new_tree is not None
    assert "sustained_health_verified" in started.observed_effects

    stopped = orchestrator.stop(
        _request(
            profile,
            operation=Operation.STOP,
            key="stop:one",
            expected_revision=started.revision,
            health_window_ms=0,
        )
    )
    assert stopped.revision == 2
    assert stopped.old_tree_fenced
    assert stopped.new_tree is None
    assert not adapter.live


def test_shared_control_transaction_wraps_process_saga_and_replays_exactly(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.seed_tree()
    orchestrator = _orchestrator(profile, adapter, clock)
    request = _request(profile)
    service = SupervisorControlService(
        repository_allowlist=(profile.repository_root,),
        state_allowlist=(profile.state_root,),
        handlers={Operation.RESTART: orchestrator},
        lease_validator=lambda _request: True,
        state_store=JsonlControlStateStore(),
        clock_ms=clock.now_ms,
    )

    result = service.restart(request)
    replay = service.restart(request)

    assert result.status is OperationStatus.SUCCEEDED
    assert replay == result
    assert adapter.terminations == 1
    assert adapter.launches == 1
    records = [
        json.loads(line)
        for line in (
            Path(profile.state_root) / "control-transactions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["phase"] for record in records] == [
        "prepared",
        "dispatching",
        "committed",
    ]
    assert records[-1]["result"]["data"]["transition"]["phase"] == "committed"


def test_changed_replay_and_overlapping_transition_are_rejected(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.seed_tree()
    orchestrator = _orchestrator(profile, adapter, clock)
    first = _request(profile)
    changed = _request(profile, key=first.idempotency_key, deadline_ms=101)
    with pytest.raises(TransactionConflictError, match="changed"):
        # Reserve the first request without effects to model a crashed owner.
        profile_value = orchestrator._profile(first)
        intent = orchestrator._intent(  # type: ignore[attr-defined]
            first, profile_value, LifecycleAction.RESTART
        )
        orchestrator._reserve(intent)  # type: ignore[attr-defined]
        orchestrator.execute(changed)

    second = _request(profile, key="restart:overlap")
    with pytest.raises(TransactionConflictError, match="still active"):
        orchestrator.execute(second)


def test_split_brain_and_descendant_left_after_stop_never_launch(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    split_adapter = FakeProcessAdapter(profile, clock)
    split_adapter.seed_tree(roots=2, descendants=0)
    split = _orchestrator(profile, split_adapter, clock)
    with pytest.raises(SplitBrainError):
        split.restart(_request(profile))
    assert split_adapter.terminations == 0
    assert split_adapter.launches == 0

    other_root = tmp_path / "other"
    other_profile = _profile(other_root)
    other_clock = Clock()
    orphan_adapter = FakeProcessAdapter(other_profile, other_clock)
    orphan_adapter.seed_tree(descendants=2)
    orphan_adapter.leave_descendant = True
    orphan = _orchestrator(other_profile, orphan_adapter, other_clock)
    with pytest.raises(ProcessTreeNotFenced):
        orphan.restart(_request(other_profile, deadline_ms=25))
    assert orphan_adapter.launches == 0
    latest = orphan.store.latest()[other_profile.target_id]
    assert latest.phase is LifecycleSagaPhase.PARTIAL_FAILURE
    assert latest.failure_code == "descendants_remain"
    assert latest.compensation == (
        "repair_or_quarantine_remaining_process_tree",
    )


def test_restart_partial_launch_failure_resumes_after_old_fence_only(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.seed_tree()
    adapter.fail_next_launch = True
    orchestrator = _orchestrator(profile, adapter, clock)
    request = _request(profile)

    with pytest.raises(PartialMutationError, match="launch failed") as failure:
        orchestrator.restart(request)
    assert failure.value.applied_effect_ids == ("restart:process-tree",)
    assert adapter.terminations == 1
    partial = orchestrator.store.latest()[profile.target_id]
    assert partial.phase is LifecycleSagaPhase.PARTIAL_FAILURE
    assert partial.old_tree_fenced
    assert partial.failure_code == "launch_failed"

    receipt = orchestrator.restart(request)
    assert receipt.succeeded
    assert adapter.terminations == 1
    assert adapter.launches == 1


def test_fork_without_sustained_health_is_compensated_and_not_success(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.seed_tree()
    adapter.healthy_value = False
    orchestrator = _orchestrator(profile, adapter, clock)

    with pytest.raises(PartialMutationError, match="sustained health"):
        orchestrator.restart(_request(profile, deadline_ms=30))
    assert not adapter.live
    partial = orchestrator.store.latest()[profile.target_id]
    assert partial.phase is LifecycleSagaPhase.PARTIAL_FAILURE
    assert partial.failure_code == "sustained_health_not_proved"
    assert partial.compensation == (
        "terminate_unhealthy_new_process_tree",
        "unhealthy_new_process_tree_terminated",
    )


def test_stale_fence_and_changed_configuration_are_rejected(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.seed_tree()
    orchestrator = _orchestrator(profile, adapter, clock)
    receipt = orchestrator.restart(_request(profile))

    stale = _request(
        profile,
        operation=Operation.STOP,
        key="stop:stale",
        expected_revision=receipt.revision,
        fence=8,
        health_window_ms=0,
    )
    with pytest.raises(StaleLeaseError):
        orchestrator.stop(stale)
    assert adapter.terminations == 1

    changed_profile = _profile(
        tmp_path, argv=("/usr/bin/python3", "-c", "print('changed')")
    )
    changed = _orchestrator(changed_profile, adapter, clock)
    request = _request(
        changed_profile,
        key="restart:changed-profile",
        expected_revision=receipt.revision,
        fence=10,
    )
    with pytest.raises(LifecycleProfileChanged):
        changed.restart(request)


def test_pid_reuse_is_detected_before_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    fake = FakeProcessAdapter(profile, clock)
    identity = fake.identity(700)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        LinuxProcessAdapter,
        "_stat",
        staticmethod(
            lambda _pid: (
                identity.parent_pid,
                identity.process_group_id,
                identity.session_id,
                identity.start_time_ticks + 1,
            )
        ),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.lifecycle_orchestrator.os.kill",
        lambda pid, signum: signals.append((pid, signum)),
    )

    with pytest.raises(ProcessIdentityMismatch, match="reused"):
        LinuxProcessAdapter._signal_exact(identity, 15)
    assert signals == []


def test_cross_root_request_and_foreign_run_identity_fail_closed(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    clock = Clock()
    adapter = FakeProcessAdapter(profile, clock)
    adapter.seed_tree()
    orchestrator = _orchestrator(profile, adapter, clock)

    foreign_profile = replace(
        profile,
        run_id="run:foreign",
        profile_id="",
    )
    foreign_request = _request(foreign_profile)
    with pytest.raises(LifecycleProfileChanged):
        orchestrator.restart(foreign_request)
    assert adapter.terminations == 0
