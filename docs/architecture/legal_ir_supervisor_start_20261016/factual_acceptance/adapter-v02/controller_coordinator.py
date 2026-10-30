"""Supported same-process host publication after independent factual checking.

This is a callable SDK, not a command accepting worker receipts. Publication
requires the actual native one-use handle AND a separate genuine same-process
ORCHESTRATION root -> phase -> canonical_parented_phase disk owner. The caller
obtains that admission after validation has finalized. Every other task stays
blocked; a new phase packet is a separate reviewed operation.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat

import factual_candidate as facts
import native_acceptance as acceptance

PARENTED_MODULE = "ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources"
SCHEDULER_MODULE = "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler"
SCHEMA = "legal-ir-host-factual-completion-request@1"
FIELDS = frozenset({
    "schema", "purpose", "task_id", "task_cid", "task_revision", "native_attempt_id",
    "candidate_commit", "board_path", "expected_board_sha256", "event_path", "source_sha256",
    "scheduler_state_path", "expected_scheduler_policy_sha256", "root_lease_id", "phase_lease_id",
    "disk_reservation_id", "root_request_id", "phase_request_id", "external_board_charge_bytes",
})


def require(condition, reason):
    acceptance.require(condition, reason)


def check_controller_projection(value, root, phase, active, request, *, owner_pid):
    """Pure metadata correspondence only; cannot mint/consume acceptance."""
    require(type(value) is dict and value.get("status") == "active"
            and value.get("scheduler_lane") == "orchestration"
            and value.get("workload") == "canonical_parented_phase"
            and value.get("reservation_id") == request["disk_reservation_id"]
            and value.get("cleanup_error") is None, "controller_disk_owner_not_active_orchestration")
    child, record = value.get("resource_lease"), value.get("record")
    require(type(child) is dict and type(record) is dict, "controller_native_metadata_missing")
    require(type(active) is list and all(type(row) is dict and type(row.get("lease_id")) is str for row in active), "controller_active_lease_metadata_invalid")
    by_id = {row["lease_id"]: row for row in active}
    require(len(by_id) == len(active), "controller_active_lease_duplicate")
    envelopes = (root, phase, child)
    parent_ids = (None, request["root_lease_id"], request["phase_lease_id"])
    ids = (request["root_lease_id"], request["phase_lease_id"], record.get("native_lease_id"))
    require(len(set(ids)) == 3 and all(type(identifier) is str and identifier for identifier in ids), "controller_ancestry_identity_invalid")
    fields = ("lease_id", "parent_lease_id", "owner_pid", "lane", "cpu_slots", "memory_mb",
              "child_process_slots", "requires_gpu", "gpu_memory_mb", "unified_memory_mb")
    for envelope, parent_id, identifier in zip(envelopes, parent_ids, ids):
        require(type(envelope) is dict and envelope.get("lease_id") == identifier
                and envelope.get("parent_lease_id") == parent_id
                and envelope.get("owner_pid") == owner_pid and envelope.get("lane") == "orchestration"
                and envelope.get("released") is False and envelope.get("cancelled") is False
                and envelope.get("requires_gpu") is False
                and envelope.get("gpu_memory_mb") == 0 and envelope.get("unified_memory_mb") == 0,
                "controller_live_orchestration_ancestry_differs")
        observed = by_id.get(identifier)
        require(type(observed) is dict and all(observed.get(key) == envelope.get(key) for key in fields),
                "controller_native_active_envelope_differs")
    require(by_id[root["lease_id"]].get("request_id") == request["root_request_id"]
            and by_id[phase["lease_id"]].get("request_id") == request["phase_request_id"], "controller_native_request_identity_differs")
    require(record.get("parent_lease_id") == phase["lease_id"] and record.get("owner_pid") == owner_pid,
            "controller_disk_phase_parent_differs")
    return {"root_lease_id": root["lease_id"], "phase_lease_id": phase["lease_id"],
            "disk_native_lease_id": child["lease_id"], "disk_reservation_id": value["reservation_id"], "owner_pid": owner_pid}


def _read_request(path, expected_sha256):
    raw = facts.read_source_file(Path(path))
    require(facts.sha(raw) == expected_sha256, "frozen_controller_request_changed")
    request = facts.strict_json(raw)
    require(type(request) is dict and set(request) == FIELDS and request["schema"] == SCHEMA
            and request["purpose"] == "controller_preparation" and request["task_id"] == facts.TASK,
            "controller_request_scope_invalid")
    require(type(request["source_sha256"]) is dict and 0 < len(request["source_sha256"]) <= 256,
            "controller_source_pins_required")
    require(type(request["external_board_charge_bytes"]) is int
            and 0 < request["external_board_charge_bytes"] <= facts.MAX_DOCUMENT_BYTES,
            "controller_external_board_bound_invalid")
    for name in ("board_path", "event_path", "scheduler_state_path"):
        require(type(request[name]) is str and Path(request[name]).is_absolute(), "controller_absolute_path_required:" + name)
    for spelling, digest in request["source_sha256"].items():
        require(type(spelling) is str and Path(spelling).is_absolute()
                and type(digest) is str and len(digest) == 64, "controller_source_pin_shape_invalid")
        require(facts.sha(facts.read_source_file(Path(spelling))) == digest, "controller_source_pin_changed:" + spelling)
    return request


def _verify_admission(owner, root_lease, phase_lease, request):
    require(type(owner).__module__ == PARENTED_MODULE
            and type(owner).__name__ == "ParentedPhaseResourceReservation", "genuine_parented_controller_owner_required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources import ParentedPhaseResourceReservation
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLease
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
    require(type(owner) is ParentedPhaseResourceReservation and type(root_lease) is ResourceLease
            and type(phase_lease) is ResourceLease, "actual_native_controller_capabilities_required")
    require(root_lease.renew() and phase_lease.renew() and owner.heartbeat(), "controller_native_heartbeat_failed")
    owner.check_usage(owner.directory, full=True)
    native = owner.native_lease  # Public property performs genuine full ancestry verification.
    require(type(native) is ResourceLease, "actual_native_controller_disk_lease_required")
    scheduler = open_existing_scheduler(request["scheduler_state_path"], request["expected_scheduler_policy_sha256"])
    return check_controller_projection(owner.to_dict(), root_lease.to_dict(), phase_lease.to_dict(),
                                       scheduler.active_leases(), request, owner_pid=os.getpid())


def complete_accepted_legacy_task(*, handle, current_context, orchestration_owner, root_lease, phase_lease,
                                  request_path, frozen_request_sha256):
    """Publish exactly LIR-9001 completed after genuine independent acceptance.

Run only from the trusted admitted host owner, in the same interpreter that
holds the unconsumed native handle. The controller request must be reviewed and
frozen outside its owned phase attempt. No provider/worker CLI can invoke an
accepting path. The public legacy taskboard lock supplies exact bytes CAS.
An ambiguous/failed publication is retained for inspection; never reset the
attempt counter, undo the audit intent, reconstruct a handle, or blindly retry.
    """
    acceptance._entry(handle)
    acceptance._context(current_context)
    request = _read_request(request_path, frozen_request_sha256)
    for name in ("task_cid", "task_revision", "native_attempt_id", "candidate_commit"):
        require(request[name] == getattr(current_context, name), "controller_candidate_binding_differs:" + name)
    board = Path(request["board_path"])
    event_path = Path(request["event_path"])
    require(board == Path(current_context.source_root) / facts.SOURCE_PATHS["execution_board"], "controller_board_path_differs")
    require(event_path.parent == orchestration_owner.directory and event_path.name == "factual-completion.jsonl",
            "controller_event_must_be_inside_private_phase_attempt")
    require(not event_path.exists(), "controller_completion_event_already_exists")
    request_root = Path(request_path)
    require(request_root != orchestration_owner.directory and orchestration_owner.directory not in request_root.parents,
            "controller_request_must_be_outside_phase_attempt")
    native = _verify_admission(orchestration_owner, root_lease, phase_lease, request)
    roots = [Path(item["path"]) for item in orchestration_owner.to_dict()["roots"]]
    require(any(root in board.parents for root in roots), "controller_board_outside_admitted_storage_roots")
    # The genuine parented API explicitly charges this external maximum. It
    # is a declared bound, not a claim of kernel-enforced filesystem quota.
    orchestration_owner.account_external_bytes("legal_ir_factual_completion_board", request["external_board_charge_bytes"])
    import sys
    modules = (
        sys.modules[__name__], facts, acceptance,
        sys.modules[PARENTED_MODULE], sys.modules[SCHEDULER_MODULE],
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import taskboard_store
    from ipfs_accelerate_py.agent_supervisor.runtime import event_log
    for module in (*modules, taskboard_store, event_log):
        path = Path(module.__file__).absolute()
        require(request["source_sha256"].get(str(path)) == facts.sha(facts.read_source_file(path)), "controller_implementation_source_not_pinned")
    sources, frozen = facts.load_source_bundle(Path(current_context.source_root))
    report = facts.read_source_file(Path(current_context.report_path))
    for path in [Path(current_context.report_path),
                 *(Path(current_context.source_root) / name for name in facts.SOURCE_PATHS.values()),
                 *(Path(current_context.source_root) / name for name in frozen)]:
        require(request["source_sha256"].get(str(path)) == facts.sha(facts.read_source_file(path)), "controller_candidate_source_not_pinned")
    _verify_admission(orchestration_owner, root_lease, phase_lease, request)
    _read_request(request_path, frozen_request_sha256)
    with taskboard_store.locked_taskboard(board) as stream:
        opened = os.fstat(stream.fileno())
        current = board.lstat()
        require(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1
                and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino), "controller_locked_board_identity_changed")
        raw = stream.read(facts.MAX_DOCUMENT_BYTES + 1).encode("utf-8")
        require(len(raw) <= facts.MAX_DOCUMENT_BYTES and facts.sha(raw) == request["expected_board_sha256"], "controller_legacy_board_revision_conflict")
        proposal, summary = acceptance.consume_for_manual_completion(
            handle, current_context=current_context, board_raw=raw, report_raw=report,
            source_documents=sources, frozen_files=frozen,
        )
        require(len(proposal) <= request["external_board_charge_bytes"], "controller_board_exceeds_precharge")
        # Intent is durable before status changes. Event files are artifacts,
        # never acceptance capabilities and never reusable after a restart.
        intent = event_log.append_jsonl_event(event_path, "legal_ir_factual_completion_intent", {
            "scope": "source-only factual engineering acceptance", "native_controller": native,
            "accepted_facts": summary, "task_id": facts.TASK, "next_phase_activated": False,
        }, max_bytes=64 * 1024)
        _verify_admission(orchestration_owner, root_lease, phase_lease, request)
        current = board.lstat()
        require((opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino), "controller_board_replaced_before_publication")
        stream.seek(0)
        require(stream.read().encode("utf-8") == raw, "controller_board_changed_before_publication")
        require(taskboard_store.replace_locked_taskboard(stream, proposal.decode("utf-8")), "controller_completion_publication_no_change")
        stream.seek(0)
        require(stream.read().encode("utf-8") == proposal, "controller_completion_readback_differs")
        completed = event_log.append_jsonl_event(event_path, "legal_ir_factual_completion_published", {
            "task_id": facts.TASK, "intent_event_id": intent["event_id"],
            "accepted_board_sha256": facts.sha(raw), "completed_board_sha256": facts.sha(proposal),
            "all_other_63_tasks_remain_blocked": True, "next_phase_activated": False,
        }, max_bytes=64 * 1024)
    orchestration_owner.check_usage(orchestration_owner.directory, full=True)
    return {"task_id": facts.TASK, "updated": True, "completed_board_sha256": facts.sha(proposal),
            "native_controller": native, "event_id": completed["event_id"],
            "next_phase_activated": False, "task_attempt_counters_modified": False,
            "scope": "Exact frozen engineering-source factual acceptance only"}
