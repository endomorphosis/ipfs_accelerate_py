"""Genuine shared resource leaves; no root reservation or policy is invented.

Only container boundary admission is authored by the existing attempt fixture.
These capacity checks do not prove task authority, hard limits, or completion.
"""
import json
import os

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import native_candidate_round_resources as resources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLease
from test.api.candidate_resource_test_support import (
    lease_record, reservation_options, resource_parent,
)
from test.api.test_owner_worker_attempt import case, _prepare


def reserve(owner, attempt, **changes):
    return resources.reserve_native_candidate_round_resources(**reservation_options(owner, attempt, **changes))


def test_real_nested_leaf_preserves_root_charge_and_exports_no_token(case, resource_parent):
    owner, attempt = resource_parent, _prepare(case)
    identifier, initial_chain = owner.parent.lease_id, []
    for _ in range(64):
        record = lease_record(owner, identifier)
        assert record is not None and identifier not in initial_chain
        initial_chain.append(identifier)
        if record["parent_lease_id"] is None:
            root_before = record
            break
        identifier = record["parent_lease_id"]
    else:
        pytest.fail("existing qualification parent ancestry exceeds the native bound")
    binding = reserve(owner, attempt)
    try:
        leaf = binding.native_lease
        assert type(leaf) is ResourceLease and leaf.owner_pid == os.getpid()
        assert leaf.parent_lease_id == owner.parent.lease_id
        native = lease_record(owner, leaf.lease_id)
        assert native is not None and native["parent_lease_id"] == owner.parent.lease_id
        value = binding.revalidate()
        root_after = lease_record(owner, root_before["lease_id"])
        assert root_after is not None and value["root_lease_id"] == root_before["lease_id"]
        charged_fields = ("lease_id", "parent_lease_id", "owner_pid", "cpu_slots", "memory_mb",
            "child_process_slots", "gpu_memory_mb", "unified_memory_mb", "requires_gpu")
        assert {name:root_after[name] for name in charged_fields} == {
            name:root_before[name] for name in charged_fields}
        identifier, actual_chain = leaf.lease_id, []
        for _ in range(64):
            record = lease_record(owner, identifier)
            assert record is not None and identifier not in actual_chain
            actual_chain.append(identifier)
            if record["parent_lease_id"] is None:
                break
            identifier = record["parent_lease_id"]
        assert actual_chain == [leaf.lease_id, *initial_chain]
        assert native["request_id"] == "native-candidate-round:" + value["resource_id"]
        candidate_population = {record["lease_id"]:record["parent_lease_id"]
            for record in json.loads(owner.scheduler.state_path.read_bytes())["leases"].values()
            if record.get("request_id") == native["request_id"]}
        assert candidate_population == {leaf.lease_id:owner.parent.lease_id}
        assert value["parent_lease_id"] == owner.parent.lease_id
        assert value["native_lease_id"] == leaf.lease_id
        assert value["worker_attempt_id"] == attempt.attempt_id
        assert value["scheduler_policy_sha256"] == owner.policy_sha256
        assert value["budget"] == dict(cpu_slots=1, memory_mb=512, child_process_slots=1, threads_per_process=1)
        assert value["quantitative_resource_reservation_verified"] is True
        assert value["resource_capacity_only"] is True and value["hard_enforcement"] is False
        thread_environment = binding.thread_environment()
        assert all(thread_environment[key] == "1" for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
        assert leaf.lease_key.encode() not in binding.metadata_bytes
        assert owner.parent.lease_key.encode() not in binding.metadata_bytes
        assert b"lease_key" not in binding.metadata_bytes
        assert binding.poll() is False and 0 < binding.remaining() <= 60
        value["budget"]["memory_mb"] = 999
        assert binding.revalidate()["budget"]["memory_mb"] == 512
        with pytest.raises((AttributeError, TypeError)):
            binding.metadata_bytes = b"{}"
    finally:
        result = binding.abort_unstarted()
    assert result["released"] is True and result["retirement_mode"] == "unstarted-abort"
    assert lease_record(owner, leaf.lease_id) is None
    assert not owner.parent.released
    with pytest.raises(ValueError):
        binding.revalidate()
    with pytest.raises(ValueError):
        binding.abort_unstarted()


@pytest.mark.parametrize("kind", ["none", "token", "mapping", "id"])
def test_only_actual_current_parent_handle_can_reserve(case, resource_parent, kind):
    attempt = _prepare(case)
    parent = {"none":None, "token":resource_parent.parent.token,
        "mapping":resource_parent.parent.to_dict(), "id":resource_parent.parent.lease_id}[kind]
    with pytest.raises((ValueError, TypeError)):
        reserve(resource_parent, attempt, parent_lease=parent)


@pytest.mark.parametrize("field,bad", [("lease_key", ""), ("lease_key", "f" * 32), ("lease_id", "f" * 32)])
def test_empty_foreign_or_unmatched_parent_authority_refuses(case, resource_parent, field, bad):
    attempt, parent = _prepare(case), resource_parent.parent
    original = getattr(parent, field)
    try:
        setattr(parent, field, bad)
        with pytest.raises((ValueError, RuntimeError)):
            reserve(resource_parent, attempt)
    finally:
        setattr(parent, field, original)


@pytest.mark.parametrize("change", [dict(cpu_slots=True), dict(memory_mb=0), dict(child_process_slots=0),
    dict(threads_per_process=0), dict(threads_per_process=True), dict(threads_per_process=2), dict(memory_mb=1025),
    dict(cpu_slots=2), dict(child_process_slots=3)])
def test_closed_positive_demands_must_fit_the_real_parent(case, resource_parent, change):
    attempt = _prepare(case)
    with pytest.raises((ValueError, RuntimeError)):
        reserve(resource_parent, attempt, **change)


@pytest.mark.parametrize("operation", ["cancel", "release"])
def test_cancelled_or_released_native_parent_cannot_admit_leaf(case, resource_parent, operation):
    attempt = _prepare(case)
    getattr(resource_parent.parent, operation)()
    with pytest.raises((ValueError, RuntimeError)):
        reserve(resource_parent, attempt)


def test_real_leaf_cancellation_retains_reservation_until_explicit_unstarted_abort(case, resource_parent):
    binding = reserve(resource_parent, _prepare(case))
    leaf = binding.native_lease
    try:
        assert leaf.cancel() is True
        assert binding.poll() is True and not leaf.released
        assert lease_record(resource_parent, leaf.lease_id) is not None
        with pytest.raises((ValueError, RuntimeError)):
            binding.revalidate()
    finally:
        binding.abort_unstarted()
    assert lease_record(resource_parent, leaf.lease_id) is None


def test_serialized_resource_observation_cannot_recreate_live_binding(case, resource_parent):
    binding = reserve(resource_parent, _prepare(case))
    try:
        observed = json.loads(binding.metadata_bytes)
        assert observed["worker_attempt_id"] and observed["resource_id"]
        with pytest.raises((ValueError, TypeError)):
            resources.NativeCandidateRoundResources(**observed)
        unissued = object.__new__(resources.NativeCandidateRoundResources)
        with pytest.raises((ValueError, AttributeError, TypeError)):
            resources.NativeCandidateRoundResources.revalidate(unissued)
    finally:
        binding.abort_unstarted()


def _owned_root_charge(owner):
    identifier, seen = owner.parent.lease_id, set()
    for _ in range(64):
        record = lease_record(owner, identifier)
        assert record is not None and identifier not in seen
        seen.add(identifier)
        if record["parent_lease_id"] is None:
            return {name:record[name] for name in ("lease_id", "parent_lease_id", "owner_pid",
                "cpu_slots", "memory_mb", "child_process_slots", "gpu_memory_mb",
                "unified_memory_mb", "requires_gpu")}
        identifier = record["parent_lease_id"]
    pytest.fail("existing qualification root ancestry exceeds the native bound")


def test_failed_factory_final_validation_releases_exact_new_unstarted_leaf(case, resource_parent, monkeypatch):
    before = _owned_root_charge(resource_parent)
    attempt, captured = _prepare(case), []
    failure = OSError("authored final resource factory validation failure")
    validate = resources.NativeCandidateRoundResources.revalidate
    def refuse_once(self):
        if not captured:
            captured.append(self)
            raise failure
        return validate(self)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(resources.NativeCandidateRoundResources, "revalidate", refuse_once)
            with pytest.raises(OSError) as error:
                reserve(resource_parent, attempt)
        assert error.value is failure and len(captured) == 1
        binding = captured[0]
        assert binding.native_lease.released is True
        assert lease_record(resource_parent, binding.native_lease.lease_id) is None
        assert _owned_root_charge(resource_parent) == before
        assert not resource_parent.parent.released
    finally:
        if captured and not captured[0].native_lease.released:
            captured[0].abort_unstarted()


def test_failed_factory_abort_retains_exact_handle_on_original_error_until_owner_cleanup(case, resource_parent, monkeypatch):
    before = _owned_root_charge(resource_parent)
    attempt, captured = _prepare(case), []
    failure = OSError("authored final resource factory validation failure")
    validate = resources.NativeCandidateRoundResources.revalidate
    def refuse_once(self):
        if not captured:
            captured.append(self)
            raise failure
        return validate(self)
    def refuse_abort(self):
        assert captured == [self]
        raise OSError("authored unstarted abort unavailable")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(resources.NativeCandidateRoundResources, "revalidate", refuse_once)
            patch.setattr(resources.NativeCandidateRoundResources, "abort_unstarted", refuse_abort)
            with pytest.raises(OSError) as error:
                reserve(resource_parent, attempt)
        assert error.value is failure and len(captured) == 1
        binding = captured[0]
        assert error.value.candidate_resources is binding
        assert not binding.native_lease.released
        assert lease_record(resource_parent, binding.native_lease.lease_id) is not None
        assert _owned_root_charge(resource_parent) == before
        receipt = binding.abort_unstarted()
        assert receipt["released"] is True
        assert lease_record(resource_parent, binding.native_lease.lease_id) is None
        assert not resource_parent.parent.released
    finally:
        if captured and not captured[0].native_lease.released:
            captured[0].abort_unstarted()
