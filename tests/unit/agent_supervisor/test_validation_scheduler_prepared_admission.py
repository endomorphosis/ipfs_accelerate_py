"""Prepared-admission protocol tests; no host resource allocation or subprocesses.

The opaque grant fixture represents the trusted adapter boundary. It does not
claim a real ResourceAdmissionLease or qualify any D parent authority provider.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import validation_scheduler as vs
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    ValidationCommand, ValidationStage,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot, LaneResourceRequirements, ResourceLeaseBudget,
    ResourcePolicy, ResourceScheduler,
)


CONTRACT = "baguqeeractimubsiddqqh7khliyus3w4j7up52fuwgfacr533zmkewet5soa"


class OpaqueOriginalGrant:
    pass


class OriginalAuthorityProtocol:
    def __init__(self):
        self.grant = OpaqueOriginalGrant()
        self.acquisitions = 0
        self.checks = 0
        self.releases = 0
        self.revoked = False
        self.release_uncertain = False

    def acquire(self):
        self.acquisitions += 1
        return self.grant

    def check(self, grant):
        assert grant is self.grant
        self.checks += 1
        if self.revoked:
            raise RuntimeError("original grant revoked")

    def release(self, grant):
        assert grant is self.grant
        self.releases += 1
        if self.release_uncertain:
            raise RuntimeError("release outcome unavailable")


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    clock = [100.0]
    real_time = vs.time
    monkeypatch.setattr(vs, "time", SimpleNamespace(
        monotonic=lambda: clock[0], time=real_time.time, sleep=real_time.sleep))
    authority = OriginalAuthorityProtocol()
    budget = ResourceLeaseBudget(
        max_parallel=1, max_cpu_proof_concurrency=1,
        max_model_concurrency=1, max_artifact_concurrency=1,
        max_processes=2, wall_time_ms=30000, cpu_time_ms=30000,
        memory_bytes=128 * 1024**2, disk_bytes=1024**2,
    )

    def build(**changes):
        arguments = dict(
            task_cid="existing-task-cid", contract_digest=CONTRACT,
            workspace_path=tmp_path.resolve(), commands=("producer", "verifier"),
            budget=budget, deadline_monotonic=120.0,
            acquire=authority.acquire, check=authority.check,
            release=authority.release,
            provenance={"schema": "original-parent-protocol-fixture/v1", "parent_id": "fixture"},
        )
        arguments.update(changes)
        return vs.PreparedValidationAdmission(**arguments)

    return build, authority, budget, clock


def spec(command="producer", **changes):
    return replace(ValidationCommand(command, cacheable=False), **changes)


def scheduler(handle, **changes):
    arguments = dict(
        max_workers=1, resource_budget=1, resource_lease_budget=handle.budget,
        prepared_admission=handle, prepared_task_cid="existing-task-cid",
        prepared_contract_digest=CONTRACT,
    )
    arguments.update(changes)
    return vs.ValidationScheduler(**arguments)


def run(sched, command, root, runner):
    return sched.run(
        (command,), workspace_path=root.resolve(), target_commit="b" * 40,
        changed_files=(), dependency_state={"fixture": "explicit"},
        require_full_validation=True, runner=runner,
    )


def test_prepared_acquisition_is_once_and_unprepared_identity_is_not_authority(protocol):
    build, authority, _, _ = protocol
    handle = build()
    with pytest.raises(vs.PreparedValidationAdmissionError, match="not_active"):
        handle.assert_binding(task_cid="existing-task-cid", contract_digest=CONTRACT)
    assert authority.acquisitions == 0
    assert handle.prepare() is handle
    with pytest.raises(vs.PreparedValidationAdmissionError, match="not_repeatable"):
        handle.prepare()
    assert authority.acquisitions == 1
    assert handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)
    assert authority.releases == 1


def test_identity_mismatch_cannot_borrow_existing_grant(protocol):
    build, authority, _, _ = protocol
    handle = build().prepare()
    for identity in ({"task_cid": "other", "contract_digest": CONTRACT},
                     {"task_cid": "existing-task-cid", "contract_digest": "b" * 64}):
        with pytest.raises(vs.PreparedValidationAdmissionError, match="task_or_contract"):
            handle.assert_binding(**identity)
    assert authority.releases == 0
    handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, 99.0, 131.0])
def test_nonfinite_expired_or_widened_deadline_is_rejected_before_acquire(protocol, deadline):
    build, authority, _, _ = protocol
    with pytest.raises(vs.PreparedValidationAdmissionError, match="deadline"):
        build(deadline_monotonic=deadline)
    assert authority.acquisitions == authority.releases == 0


@pytest.mark.parametrize("field", ["memory_bytes", "disk_bytes", "max_processes", "cpu_time_ms"])
def test_changed_budget_is_rejected_before_a_command_borrow(protocol, tmp_path, field):
    build, authority, budget, _ = protocol
    handle = build().prepare()
    widened = replace(budget, **{field: getattr(budget, field) + 1})
    with pytest.raises(vs.PreparedValidationAdmissionError, match="configuration_changed"):
        scheduler(handle, resource_lease_budget=widened)
    assert authority.acquisitions == 1 and authority.releases == 0
    handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)


def test_mapping_is_not_a_nominal_budget_or_prepared_handle(protocol):
    build, authority, budget, _ = protocol
    with pytest.raises(vs.PreparedValidationAdmissionError, match="nominal_budget"):
        build(budget=budget.to_dict())
    with pytest.raises(vs.PreparedValidationAdmissionError, match="configuration_changed"):
        vs.ValidationScheduler(max_workers=1, resource_budget=1,
            resource_lease_budget=budget, prepared_admission={"admitted": True})
    assert authority.acquisitions == 0


@pytest.mark.parametrize("grant", [{"admitted": True}, True, "opaque-looking-string"])
def test_json_boolean_and_string_grants_do_not_supply_authority(protocol, grant):
    build, authority, _, _ = protocol
    handle = build(acquire=lambda: grant)
    with pytest.raises(vs.PreparedValidationAdmissionError, match="opaque_grant"):
        handle.prepare()
    assert not handle.close(disposition=vs.PreparedValidationDisposition.UNKNOWN)
    assert authority.releases == 0


def test_revoked_grant_refuses_before_runner_and_does_not_resample_host(protocol, tmp_path):
    build, authority, _, _ = protocol
    handle = build().prepare()
    sched = scheduler(handle)
    authority.revoked = True
    with pytest.raises(RuntimeError, match="revoked"):
        run(sched, spec(), tmp_path, lambda **kw: pytest.fail("runner entered"))
    assert authority.releases == 0
    handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)


def test_deadline_is_fixed_before_prepare_and_expiry_blocks_next_run(protocol, tmp_path):
    build, authority, _, clock = protocol
    handle = build().prepare()
    sched = scheduler(handle)
    clock[0] = 121.0
    with pytest.raises(vs.PreparedValidationAdmissionError, match="deadline_expired"):
        run(sched, spec(), tmp_path, lambda **kw: pytest.fail("runner entered"))
    assert handle.deadline_monotonic == 120.0
    assert authority.releases == 0
    handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)


def test_wrong_workspace_cannot_use_prepared_grant(protocol, tmp_path):
    build, authority, _, _ = protocol
    handle = build().prepare()
    alternate = tmp_path / "other"
    alternate.mkdir()
    with pytest.raises(vs.PreparedValidationAdmissionError, match="workspace_or_budget"):
        run(scheduler(handle), spec(), alternate, lambda **kw: pytest.fail("runner entered"))
    assert authority.releases == 0
    handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)


def test_ordered_two_command_group_needs_original_stop_between_commands(protocol, tmp_path, monkeypatch):
    build, authority, _, clock = protocol
    handle = build().prepare()
    sched = scheduler(handle)
    def forbidden(*a, **kw):
        pytest.fail("prepared path attempted A host acquisition or implicit discovery")
    monkeypatch.setattr(sched.resource_scheduler, "acquire", forbidden)
    monkeypatch.setattr(sched.resource_scheduler, "release", forbidden)
    for name in ("discover_changed_files", "resolve_target_commit", "collect_dependency_state"):
        monkeypatch.setattr(vs, name, forbidden)
    seen = []
    def runner(**kw):
        seen.append((kw["spec"].command, kw["timeout_seconds"]))
        return {"returncode": 0, "output": "fixture complete"}
    clock[0] = 110.0
    first = run(sched, spec(), tmp_path, runner)
    assert first["passed"] is True
    receipt = first["results"][0]["resource_lease"]
    assert receipt["borrow_returned"] is True and receipt["released"] is False
    assert receipt["host_pressure_resampled"] is False
    assert receipt["original_stop_acknowledged"] is False
    assert authority.releases == 0 and authority.acquisitions == 1
    with pytest.raises(vs.PreparedValidationAdmissionError, match="serial"):
        run(sched, spec("verifier"), tmp_path, runner)
    with pytest.raises(vs.PreparedValidationAdmissionError, match="acknowledgment_mismatch"):
        handle.acknowledge_stopped(command="verifier")
    handle.acknowledge_stopped(command="producer")
    second = run(sched, spec("verifier"), tmp_path, runner)
    assert second["passed"] is True
    assert second["results"][0]["resource_lease"]["command_index"] == 1
    assert seen == [("producer", 10.0), ("verifier", 10.0)]
    assert authority.acquisitions == 1 and authority.releases == 0
    handle.acknowledge_stopped(command="verifier")
    assert handle.close(disposition=vs.PreparedValidationDisposition.STOPPED)
    assert authority.releases == 1


def test_skipped_or_reused_command_cannot_borrow(protocol, tmp_path):
    build, authority, _, _ = protocol
    handle = build().prepare()
    sched = scheduler(handle)
    runner = lambda **kw: {"returncode": 0, "output": ""}
    with pytest.raises(vs.PreparedValidationAdmissionError, match="order_or_reuse"):
        run(sched, spec("verifier"), tmp_path, runner)
    run(sched, spec(), tmp_path, runner)
    handle.acknowledge_stopped(command="producer")
    with pytest.raises(vs.PreparedValidationAdmissionError, match="order_or_reuse"):
        run(sched, spec(), tmp_path, runner)
    assert authority.releases == 0
    handle.close(disposition=vs.PreparedValidationDisposition.UNKNOWN)


def test_unknown_after_command_retains_real_grant_and_forbids_not_started(protocol, tmp_path):
    build, authority, _, _ = protocol
    handle = build().prepare()
    run(scheduler(handle), spec(), tmp_path, lambda **kw: {"returncode": 0})
    assert not handle.close(disposition=vs.PreparedValidationDisposition.UNKNOWN)
    assert authority.releases == 0
    with pytest.raises(vs.PreparedValidationAdmissionError, match="execution_was_borrowed"):
        handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)
    with pytest.raises(vs.PreparedValidationAdmissionError, match="not_active"):
        handle.assert_binding(task_cid="existing-task-cid", contract_digest=CONTRACT)
    assert authority.releases == 0


def test_context_exit_does_not_infer_stopped_from_no_exception(protocol):
    build, authority, _, _ = protocol
    with build() as handle:
        assert authority.acquisitions == 1
    assert authority.releases == 0
    assert handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)
    assert authority.releases == 1


def test_outstanding_borrow_cannot_be_closed_even_with_stopped_disposition(protocol, tmp_path):
    build, authority, _, _ = protocol
    handle = build().prepare()
    def runner(**kw):
        with pytest.raises(vs.PreparedValidationAdmissionError, match="still_borrowed"):
            handle.close(disposition=vs.PreparedValidationDisposition.STOPPED)
        assert authority.releases == 0
        return {"returncode": 0}
    run(scheduler(handle), spec(), tmp_path, runner)
    handle.acknowledge_stopped(command="producer")
    assert handle.close(disposition=vs.PreparedValidationDisposition.STOPPED)
    assert authority.releases == 1


def test_release_uncertainty_is_retained_without_second_release(protocol):
    build, authority, _, _ = protocol
    handle = build().prepare()
    authority.release_uncertain = True
    with pytest.raises(RuntimeError, match="outcome unavailable"):
        handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)
    with pytest.raises(vs.PreparedValidationAdmissionError, match="original_reconciliation"):
        handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)
    assert authority.releases == 1


def test_uncertain_acquire_is_not_replayed_or_automatically_released(protocol):
    build, authority, _, _ = protocol
    def acquire():
        authority.acquisitions += 1
        raise RuntimeError("acquisition outcome unavailable")
    handle = build(acquire=acquire)
    with pytest.raises(RuntimeError, match="acquisition outcome"):
        handle.prepare()
    with pytest.raises(vs.PreparedValidationAdmissionError, match="not_repeatable"):
        handle.prepare()
    with pytest.raises(vs.PreparedValidationAdmissionError, match="original_reconciliation"):
        handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)
    assert authority.acquisitions == 1 and authority.releases == 0


def test_liveness_loss_preserves_returned_runner_receipt_and_retains_grant(protocol, tmp_path):
    build, authority, _, _ = protocol
    handle = build().prepare()
    sentinel = {"custody": {"returncode": 0, "fixture_only": True}}
    def runner(**kw):
        authority.revoked = True
        return {"returncode": 0, "passed": True, **sentinel}
    report = run(scheduler(handle), spec(), tmp_path, runner)
    result = report["results"][0]
    assert report["passed"] is False and report["returncode"] == 75
    assert result["custody"] == sentinel["custody"]
    assert result["returncode"] == result["custody"]["returncode"] == 0
    assert report["throughput"]["accepted_count"] == 0
    assert result["original_runner_returncode"] == 0
    assert result["original_runner_passed"] is True
    assert result["authoritative"] is False
    assert result["prepared_resource_postcheck"]["valid"] is False
    assert not handle.close(disposition=vs.PreparedValidationDisposition.UNKNOWN)
    assert authority.releases == 0
    # Only the caller's later original stop acknowledgment/disposition permits
    # release. The failed resource check did not assert any custody state.
    handle.acknowledge_stopped(command="producer")
    assert handle.close(disposition=vs.PreparedValidationDisposition.STOPPED)
    assert authority.releases == 1


@pytest.mark.parametrize("change", [
    {"cacheable": True}, {"timeout_seconds": float("nan")},
    {"timeout_seconds": True}, {"resource_cost": True},
    {"stage": "targeted"}, {"fallback": True},
])
def test_undeclared_spec_shape_is_rejected_before_runner(protocol, tmp_path, change):
    build, authority, _, _ = protocol
    handle = build().prepare()
    with pytest.raises(vs.PreparedValidationAdmissionError, match="serial"):
        run(scheduler(handle), spec(**change), tmp_path, lambda **kw: pytest.fail("runner entered"))
    assert authority.releases == 0
    handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)


def test_empty_or_multi_command_run_cannot_bypass_stop_acknowledgments(protocol, tmp_path):
    build, authority, _, _ = protocol
    handle = build().prepare()
    sched = scheduler(handle)
    options = dict(workspace_path=tmp_path.resolve(), target_commit="b" * 40,
        changed_files=(), dependency_state={}, require_full_validation=True,
        runner=lambda **kw: pytest.fail("runner entered"))
    with pytest.raises(vs.PreparedValidationAdmissionError, match="command_required"):
        sched.run((), **options)
    with pytest.raises(vs.PreparedValidationAdmissionError, match="serial"):
        sched.run((spec(), spec("verifier")), **options)
    assert authority.releases == 0
    handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)


def test_original_host_watermark_admission_remains_enforced_without_prepared_grant():
    resource = ResourceScheduler(ResourcePolicy())
    host = HostResourceSnapshot(disk_percent=99)
    decision, lease = resource.acquire(
        LaneResourceRequirements(lane_id="default-regression", disk_bytes=1), host=host)
    assert lease is None and decision.admitted is False
    assert "host_disk_high_watermark" in decision.reasons


def test_replaced_workspace_inode_cannot_borrow_prepared_grant(protocol, tmp_path):
    build, authority, _, _ = protocol
    workspace = tmp_path / "bound-workspace"
    saved = tmp_path / "original-workspace"
    workspace.mkdir()
    handle = build(workspace_path=workspace).prepare()
    workspace.rename(saved)
    workspace.mkdir()
    try:
        with pytest.raises(vs.PreparedValidationAdmissionError, match="workspace_identity"):
            handle.assert_binding(task_cid="existing-task-cid", contract_digest=CONTRACT)
        assert authority.releases == 0
    finally:
        workspace.rmdir()
        saved.rename(workspace)
    handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)


def test_existing_native_cid_contract_is_preserved_without_sha_reinterpretation(protocol):
    build, authority, _, _ = protocol
    handle = build(contract_digest=CONTRACT).prepare()
    handle.assert_binding(task_cid="existing-task-cid", contract_digest=CONTRACT)
    with pytest.raises(vs.PreparedValidationAdmissionError, match="task_or_contract"):
        handle.assert_binding(task_cid="existing-task-cid", contract_digest=CONTRACT + "a")
    assert authority.acquisitions == 1 and authority.releases == 0
    handle.close(disposition=vs.PreparedValidationDisposition.NOT_STARTED)


def test_postrun_expiry_preserves_failure_and_does_not_invent_stop_witness(protocol, tmp_path):
    build, authority, _, clock = protocol
    handle = build().prepare()
    def runner(**kw):
        clock[0] = 121.0
        return {"returncode": 1, "error": "runner output unavailable"}
    report = run(scheduler(handle), spec(), tmp_path, runner)
    result = report["results"][0]
    assert report["passed"] is False and report["returncode"] == 75
    assert result["returncode"] == result["original_runner_returncode"] == 1
    assert result["original_runner_error"] == "runner output unavailable"
    assert result["prepared_resource_postcheck"]["error"] == "prepared_deadline_expired"
    assert "custody" not in result
    assert not handle.close(disposition=vs.PreparedValidationDisposition.UNKNOWN)
    assert authority.releases == 0
