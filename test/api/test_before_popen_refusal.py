"""Real Linux snapshots distinguish no birth from partial process effects.

Boundary tokens are explicitly minted private test fixtures; authorization and
lease admission are stubs. These tests qualify native process/journal handling,
not a supervisor owner, worker, training model or signed launch qualification.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.control import before_popen_refusal as refusal
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation, OperationStatus
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    JsonlControlStateStore, MutationTransactionPhase, SupervisorControlService,
)
from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    LifecycleOrchestrator, LifecycleSagaPhase, LinuxProcessAdapter,
)
from test.api.test_agent_supervisor_lifecycle_orchestrator import _profile, _request


def request(profile, operation, key):
    value = _request(profile, operation=operation, key=key, deadline_ms=1000,
        health_window_ms=0 if operation is Operation.STOP else 20)
    now = time.time_ns() // 1_000_000
    return replace(value, authorization=replace(value.authorization,
        evaluated_at_ms=now - 1, expires_at_ms=now + 60_000))


def token(profile, *, token_type=refusal.BeforePopenRefusal, seal=None, fence=9, lease="lease:prompt"):
    if seal is None:
        seal = refusal._BOUNDARY_SEAL
    return token_type(seal, message="fixture refusal before Popen",
        binding={**refusal._profile_binding(profile), "lease_id": lease,
            "fencing_epoch": fence, "finite_scope_lease_id": "fixture:resource-envelope",
            "production_activated": False, "authenticated_process_origin": False,
            "atomicity_attested": False})


def setup(tmp_path, callback):
    profile = _profile(tmp_path, argv=(sys.executable, "-c", "import time; time.sleep(60)"))
    adapter = LinuxProcessAdapter(popen=callback(profile))
    orchestrator = LifecycleOrchestrator(state_root=profile.state_root, profiles=(profile,),
        process_adapter=adapter, poll_interval_ms=5, stop_grace_ms=20)
    service = SupervisorControlService(repository_allowlist=(profile.repository_root,),
        state_allowlist=(profile.state_root,), handlers={operation: orchestrator for operation in (
            Operation.START, Operation.STOP, Operation.RESTART)},
        lease_validator=lambda _request: True, state_store=JsonlControlStateStore())
    return profile, adapter, orchestrator, service


def test_exact_no_birth_refusal_is_terminal_and_actual_native_stop_can_follow(tmp_path, monkeypatch):
    observed = {"popen_calls": 0}
    actual_popen = subprocess.Popen
    def count_actual_popen(*args, **kwargs):
        observed["popen_calls"] += 1
        return actual_popen(*args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", count_actual_popen)
    def callback(profile):
        def refuse(*args, **kwargs):
            assert not args or list(args[0]) == list(profile.argv)
            raise token(profile)
        return refuse
    profile, adapter, orchestrator, service = setup(tmp_path, callback)
    start = request(profile, Operation.START, "start:refused")
    result = service.execute(start)
    assert result.status is OperationStatus.CONFLICT
    assert result.effects == () and observed["popen_calls"] == 0
    transaction = service.mutation_transaction(start)
    assert transaction.phase is MutationTransactionPhase.COMPENSATED
    assert transaction.applied_effect_ids == () and transaction.recovery_action.value == "none"
    assert transaction.result == result and transaction.failure_code == "conflict"
    proof = result.data["before_popen_refusal"]["proof"]
    assert not proof["empty_tree"]["members"] and proof["request_id"] == start.request_id
    failed = orchestrator.store.latest()[profile.target_id]
    assert failed.phase is LifecycleSagaPhase.FAILED and failed.phase.terminal
    assert failed.receipt is None and failed.observed_effects == ()
    assert failed.failure_code == "refused_before_popen_without_process_effect"
    assert adapter.snapshot(profile).members == ()
    stopped = service.execute(request(profile, Operation.STOP, "stop:after-refused"))
    assert stopped.status is OperationStatus.SUCCEEDED
    assert stopped.data["old_tree_fenced"] is True
    assert adapter.snapshot(profile).members == ()
    assert orchestrator.store.latest()[profile.target_id].phase is LifecycleSagaPhase.COMMITTED
    phases = [json.loads(line)["phase"] for line in (
        Path(profile.state_root)/"lifecycle-transitions.jsonl").read_text().splitlines()]
    assert phases[:3] == ["prepared", "starting_new", "failed"]
    assert phases[-1] == "committed"


@pytest.mark.parametrize("kind", ["ordinary", "unsealed", "subclass", "wrong_fence", "wrong_lease"])
def test_unknown_or_counterfeit_refusal_keeps_existing_partial_repair(tmp_path, kind):
    class Subclass(refusal.BeforePopenRefusal):
        pass
    def callback(profile):
        def refuse(*_args, **_kwargs):
            if kind == "ordinary":
                raise ValueError("ordinary launch failure")
            if kind == "unsealed":
                error = refusal.BeforePopenRefusal.__new__(refusal.BeforePopenRefusal)
                Exception.__init__(error, "unsealed fixture")
                raise error
            if kind == "subclass":
                raise token(profile, token_type=Subclass)
            if kind == "wrong_lease":
                raise token(profile, lease="lease:foreign")
            raise token(profile, fence=10)
        return refuse
    profile, adapter, orchestrator, service = setup(tmp_path, callback)
    start = request(profile, Operation.START, "start:unknown")
    result = service.execute(start)
    assert result.status is OperationStatus.CONFLICT
    assert service.mutation_transaction(start).phase is MutationTransactionPhase.REPAIR_REQUIRED
    assert orchestrator.store.latest()[profile.target_id].phase is LifecycleSagaPhase.PARTIAL_FAILURE
    assert "before_popen_refusal" not in result.data
    assert adapter.snapshot(profile).members == ()
    assert service.execute(request(profile, Operation.STOP, "stop:blocked")).status is OperationStatus.CONFLICT


def test_actual_process_birth_before_claimed_refusal_stays_partial(tmp_path):
    children = []
    def callback(profile):
        def born_then_refuse(*args, **kwargs):
            child = subprocess.Popen(*args, **kwargs)
            children.append(child)
            raise token(profile)
        return born_then_refuse
    profile, adapter, orchestrator, service = setup(tmp_path, callback)
    try:
        start = request(profile, Operation.START, "start:process-already-born")
        result = service.execute(start)
        assert result.status is OperationStatus.CONFLICT
        assert len(children) == 1 and children[0].poll() is None
        observed = adapter.snapshot(profile)
        assert any(member.pid == children[0].pid for member in observed.members)
        assert service.mutation_transaction(start).phase is MutationTransactionPhase.REPAIR_REQUIRED
        assert orchestrator.store.latest()[profile.target_id].phase is LifecycleSagaPhase.PARTIAL_FAILURE
        assert "before_popen_refusal" not in result.data
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
    assert adapter.snapshot(profile).members == ()


def test_restart_that_stopped_a_real_old_tree_retains_partial_effect_lineage(tmp_path):
    children = []
    def callback(_profile):
        def born(*args, **kwargs):
            child = subprocess.Popen(*args, **kwargs)
            children.append(child)
            return child
        return born
    profile, adapter, orchestrator, service = setup(tmp_path, callback)
    try:
        identity = adapter.launch(profile, fencing_epoch=9)
        assert any(member.pid == identity.pid for member in adapter.snapshot(profile).members)
        def reject(*_args, **_kwargs):
            raise token(profile)
        adapter._popen = reject
        restart = request(profile, Operation.RESTART, "restart:old-effect")
        result = service.execute(restart)
        assert result.status is OperationStatus.CONFLICT
        transaction = service.mutation_transaction(restart)
        assert transaction.phase is MutationTransactionPhase.REPAIR_REQUIRED
        assert transaction.applied_effect_ids == ("restart:process-tree",)
        partial = orchestrator.store.latest()[profile.target_id]
        assert partial.phase is LifecycleSagaPhase.PARTIAL_FAILURE and partial.old_tree_fenced
        assert partial.old_tree.members and partial.failure_code == "launch_failed"
        assert "before_popen_refusal" not in result.data
        assert adapter.snapshot(profile).members == ()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("class_name", ["BeforePopenRefusal", "VerifiedBeforePopenRefusal"])
def test_serialized_or_unsealed_constructor_cannot_grant_no_effect_path(class_name):
    cls = getattr(refusal, class_name)
    arguments = {"message": "unsealed", "binding" if class_name == "BeforePopenRefusal" else "proof": {}}
    with pytest.raises(ValueError, match="needs its native"):
        cls(object(), **arguments)
    with pytest.raises(ValueError, match="exact finite advisory runtime"):
        refusal.refuse_finite_advisory_before_popen({}, ValueError("inert"))
