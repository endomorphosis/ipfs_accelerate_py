"""Controller-only bridge from genuine native validation to manual acceptance.

There is deliberately no accepting CLI, JSON receipt loader, environment
switch, injected verifier callback, or serialization of the acceptance handle.
Import this pinned module in the trusted host coordinator, never a provider
worker. NativeCandidateContext contains that coordinator's actual observations;
constructing it from worker JSON is outside this supported API's contract.

The adapter cannot authenticate arbitrary host callers against themselves.
Its boundary is an actual pinned, independent native owner plus a same-process
one-use handle. Native resource accounting alone is not factual acceptance;
the predicate is independently recomputed here before genuine finalization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
import time

import factual_candidate as facts

CHECK = "legal-ir-source-factual-correspondence-v1"
NATIVE_MODULE = "ipfs_datasets_py.optimizers.logic_theorem_optimizer.validation_run_resources"
HANDLE_LIFETIME_SECONDS = 900


@dataclass(frozen=True)
class NativeCandidateContext:
    """Actual host observations, not a model-provided attestation format.

The host obtains task identity/revision from its own legacy task identity and
exact board observation, attempt/provider facts from its actual dispatch and
fenced closeout, candidate commit from its actual candidate, and the receipt
pin from the corresponding native owner outcome. These inputs must not be
reconstructed from candidate fields or unrelated historical receipts.
    """
    task_cid: str
    task_revision: str
    native_attempt_id: str
    candidate_commit: str
    orchestration_namespace: str
    source_root: str
    report_path: str
    report_sha256: str
    provider_id: str
    model_id: str
    reasoning_effort: str
    provider_trigger: str
    provider_execution_id: str
    provider_worker_pid: int
    provider_receipt_path: str
    provider_receipt_sha256: str
    controller_observation_path: str
    controller_observation_sha256: str


class AcceptanceRefusal(facts.FactualRefusal):
    pass


def require(condition, reason):
    if not condition:
        raise AcceptanceRefusal(reason)


def _context(context):
    require(type(context) is NativeCandidateContext, "actual_host_context_object_required")
    for name in ("task_cid", "task_revision", "native_attempt_id", "orchestration_namespace", "provider_execution_id"):
        value = getattr(context, name)
        require(type(value) is str and 0 < len(value) <= 1024
                and not any(char in value for char in "\x00\n\r"), "host_identity_invalid:" + name)
    require(re.fullmatch(r"[0-9a-f]{40}", context.candidate_commit) is not None, "candidate_commit_invalid")
    for name in ("report_sha256", "provider_receipt_sha256", "controller_observation_sha256"):
        require(type(getattr(context, name)) is str and re.fullmatch(r"[0-9a-f]{64}", getattr(context, name)), "host_digest_invalid:" + name)
    require(type(context.provider_worker_pid) is int and context.provider_worker_pid > 1
            and context.provider_worker_pid != os.getpid(), "validation_owner_not_distinct_from_provider_worker")
    route = (context.provider_id, context.model_id, context.reasoning_effort, context.provider_trigger)
    require(route in {
        ("grok_cli", "grok-4.7", "", "primary"),
        ("codex", "gpt-6.1-sol", "high", "primary_quota_exhausted"),
    }, "provider_outcome_route_not_supported")
    for name in ("source_root", "report_path", "provider_receipt_path", "controller_observation_path"):
        require(type(getattr(context, name)) is str and Path(getattr(context, name)).is_absolute(), "absolute_host_path_required:" + name)
    report = Path(context.report_path)
    # Worker-authored output paths cannot double as actual owner observations.
    for name in ("provider_receipt_path", "controller_observation_path"):
        path = Path(getattr(context, name))
        require(path != report and report.parent not in path.parents, "owner_observation_inside_worker_output")


def check_native_projection(value, *, expected_owner_pid, expected_worker_pid, expected_request_sha256, expected_child_pid):
    """Validate public native metadata; this pure helper grants no authority.

Only issue_native_acceptance calls it on a genuine live owner's public API.
Passing an otherwise plausible mapping to this function cannot mint a handle.
    """
    require(type(value) is dict and value.get("scheduler_lane") == "validation"
            and value.get("workload") == "canonical_validation"
            and value.get("status") == "active", "native_validation_identity_not_active")
    lease, record = value.get("resource_lease"), value.get("record")
    require(type(lease) is dict and type(record) is dict, "native_validation_metadata_missing")
    require(lease.get("lane") == "validation" and lease.get("owner_pid") == expected_owner_pid
            and lease.get("owner_pid") != expected_worker_pid
            and lease.get("parent_lease_id") is None and lease.get("requires_gpu") is False
            and lease.get("released") is False and lease.get("cancelled") is False,
            "native_validation_lease_not_live_distinct_root")
    require(type(lease.get("lease_id")) is str and lease["lease_id"]
            and record.get("native_lease_id") == lease["lease_id"]
            and record.get("owner_pid") == expected_owner_pid
            and record.get("validation_request_sha256") == expected_request_sha256,
            "native_validation_request_or_lease_differs")
    child = record.get("child")
    require(type(child) is dict and child.get("pid") == expected_child_pid,
            "native_validation_attached_child_differs")
    require(value.get("cleanup_error") is None, "native_validation_cleanup_error")
    return {"reservation_id": value["reservation_id"], "lease_id": lease["lease_id"],
            "owner_pid": expected_owner_pid, "request_sha256": expected_request_sha256,
            "child_pid": expected_child_pid}


class FactualAcceptanceHandle:
    """Opaque, same-process, one-use capability. Exported summaries are inert."""
    __slots__ = ("__weakref__",)

    def __new__(cls, *args, **kwargs):
        raise AcceptanceRefusal("native_owner_only_handle_constructor")

    def __reduce__(self):
        raise AcceptanceRefusal("acceptance_handle_not_serializable")

    def __copy__(self):
        raise AcceptanceRefusal("acceptance_handle_not_copyable")

    def __deepcopy__(self, memo):
        raise AcceptanceRefusal("acceptance_handle_not_copyable")


# Object identity is deliberately process-local. No persisted field can be
# loaded to populate this registry; restart requires fresh actual validation.
_HANDLES = {}


def _pinned_source(path, request):
    raw = facts.read_source_file(Path(path))
    require(request["source_sha256"].get(str(path)) == facts.sha(raw), "native_request_source_not_bound:" + str(path))
    return raw


def issue_native_acceptance(*, reservation, process, context, evidence_path, result_path):
    """Check, seal, and finalize an actual independent native validation run.

The trusted launcher opens the genuine reservation, starts a hermetic direct
child behind a startup barrier, attaches that actual child, verifies heartbeat,
then permits execution. This function is called after wait/reap with that same
Popen object. It never creates a reservation or launches productive work.
    """
    _context(context)
    require(type(reservation).__module__ == NATIVE_MODULE
            and type(reservation).__name__ == "ValidationResourceReservation", "genuine_native_validation_owner_required")
    # Lazy import ensures merely importing this adapter does not initialize a
    # product package or scheduler. The host imports under actual native scope.
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.validation_run_resources import ValidationResourceReservation
    require(type(reservation) is ValidationResourceReservation, "genuine_native_validation_owner_required")
    require(type(process) is subprocess.Popen and process.poll() == 0 and process.returncode == 0,
            "actual_successful_reaped_validation_child_required")
    reservation.guard()
    require(reservation.heartbeat(), "native_validation_heartbeat_failed")
    limits = reservation.check_limits(full=True)
    require(limits.get("full_ledger_check") is True, "native_validation_full_check_missing")
    request = reservation.request
    import sys
    for module_name in (
        NATIVE_MODULE,
        "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_daemon_resources",
        "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler",
        "ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler",
    ):
        module = sys.modules.get(module_name)
        require(module is not None and getattr(module, "__file__", None), "native_owner_source_module_missing")
        _pinned_source(Path(module.__file__).absolute(), request)
    facts.same(request["selected_checks"], [CHECK], "selected_factual_check_differs")
    facts.same(process.args, request["argv"], "actual_validation_child_argv_differs")
    native = check_native_projection(
        reservation.to_dict(), expected_owner_pid=os.getpid(), expected_worker_pid=context.provider_worker_pid,
        expected_request_sha256=reservation.request_sha256, expected_child_pid=process.pid,
    )
    # Freeze the implementation and every input actually used by the predicate.
    for path in (Path(__file__).absolute(), Path(facts.__file__).absolute()):
        _pinned_source(path, request)
    sources, frozen = facts.load_source_bundle(Path(context.source_root))
    for role, name in facts.SOURCE_PATHS.items():
        require(_pinned_source(Path(context.source_root) / name, request) == sources[role], "source_bundle_changed")
    for name, raw in frozen.items():
        require(_pinned_source(Path(context.source_root) / name, request) == raw, "frozen_source_bundle_changed")
    report = _pinned_source(Path(context.report_path), request)
    require(facts.sha(report) == context.report_sha256, "actual_candidate_report_pin_differs")
    for path_name, digest_name in (("provider_receipt_path", "provider_receipt_sha256"),
                                   ("controller_observation_path", "controller_observation_sha256")):
        raw = _pinned_source(Path(getattr(context, path_name)), request)
        require(facts.sha(raw) == getattr(context, digest_name), "actual_owner_observation_pin_differs")
    correspondence = facts.check_facts(report, sources, frozen)
    require(facts.strict_json(report)["orchestration_namespace"] == context.orchestration_namespace,
            "actual_candidate_namespace_differs")
    evidence_path, result_path = Path(evidence_path), Path(result_path)
    for path in (evidence_path, result_path):
        require(path.is_absolute() and reservation.directory in path.parents,
                "native_result_outside_private_validation_attempt")
    evidence_raw = facts.read_source_file(evidence_path)
    facts.same(facts.strict_json(evidence_raw), correspondence, "actual_child_factual_evidence_differs")
    result_raw = facts.read_source_file(result_path)
    result = facts.strict_json(result_raw)
    require(type(result) is dict and result.get("evidence_paths") == [str(evidence_path)], "native_result_evidence_path_differs")
    # Genuine API checks attached group death, selected checks/returncode,
    # exact source request, regular owned artifacts, and fsync durability.
    sealed = reservation.seal_results(result_path)
    finalized = reservation.finalize(artifacts_durable=True)
    require(finalized.get("status") == "released" and finalized.get("cleanup_error") is None
            and finalized.get("resource_lease", {}).get("released") is True
            and finalized.get("record", {}).get("artifacts_durable_asserted") is True,
            "native_validation_closeout_not_verified")
    require(sealed["result_sha256"] == facts.sha(result_raw), "sealed_native_result_differs")
    # Recheck pins after real finalization; no new result is accepted by file
    # presence, and the owner object remains the authority for this handoff.
    require(facts.read_source_file(evidence_path) == evidence_raw
            and facts.read_source_file(result_path) == result_raw, "native_artifact_changed_after_finalization")
    require(facts.read_source_file(Path(context.report_path)) == report, "candidate_changed_after_validation")
    handle = object.__new__(FactualAcceptanceHandle)
    _HANDLES[handle] = {
        "owner_pid": os.getpid(), "expires": time.monotonic() + HANDLE_LIFETIME_SECONDS,
        "context": context, "report": report, "sources": sources, "frozen": frozen,
        "summary": {
            "schema": "legal-ir-native-factual-acceptance-summary@1", "task_id": facts.TASK,
            "task_cid": context.task_cid, "task_revision": context.task_revision,
            "native_attempt_id": context.native_attempt_id, "candidate_commit": context.candidate_commit,
            "report": facts.pin(report), "source_commitments": correspondence["source_commitments"],
            "native_validation": native, "result": facts.pin(result_raw), "evidence": facts.pin(evidence_raw),
            "factual_criterion_satisfied": True, "scope": "Exact frozen engineering-source correspondence",
            "completion_mutated": False, "next_phase_activated": False,
            "runtime_effects_verified": False, "current_other_owner_state_authenticated": False,
            "legal_meaning_review_authority": False, "gold_admission_authority": False,
            "training_authority": False, "proof_authority": False,
        },
    }
    return handle


def _entry(handle):
    require(type(handle) is FactualAcceptanceHandle and handle in _HANDLES,
            "genuine_unconsumed_native_handle_required")
    entry = _HANDLES[handle]
    require(entry["owner_pid"] == os.getpid() and time.monotonic() <= entry["expires"], "native_handle_owner_or_deadline_changed")
    return entry


def completion_projection(board_raw, *, expected_board_sha256):
    """Pure exact-byte proposal for one legacy status; it grants no authority."""
    require(type(board_raw) is bytes and facts.sha(board_raw) == expected_board_sha256, "legacy_board_revision_conflict")
    require(b"\r" not in board_raw, "legacy_board_line_ending_not_supported")
    text = board_raw.decode("utf-8")
    headings = list(re.finditer(r"(?m)^## (LIR-[0-9]{4}) [^\n]+$", text))
    require(len(headings) == 64 and len({match.group(1) for match in headings}) == 64, "legacy_completion_task_population_differs")
    match = next((match for match in headings if match.group(1) == facts.TASK), None)
    require(match is not None, "legacy_completion_task_missing")
    index = headings.index(match)
    end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
    block = text[match.end():end]
    require(len(re.findall(r"(?m)^- Status: todo$", block)) == 1
            and len(re.findall(r"(?m)^- Status:", block)) == 1
            and len(re.findall(r"(?m)^- Completion: manual$", block)) == 1,
            "legacy_completion_status_or_policy_differs")
    for other in headings:
        if other is match:
            continue
        position = headings.index(other)
        other_end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
        other_block = text[other.end():other_end]
        require(len(re.findall(r"(?m)^- Status: blocked$", other_block)) == 1, "legacy_future_task_unblocked")
    changed = block.replace("- Status: todo\n", "- Status: completed\n", 1)
    require(changed != block, "legacy_completion_no_change")
    return (text[:match.end()] + changed + text[end:]).encode("utf-8")


def consume_for_manual_completion(handle, *, current_context, board_raw, report_raw, source_documents, frozen_files):
    """Consume only in the trusted coordinator while its board lock is held.

Returns a completion proposal and an inert audit summary. The caller publishes
with the existing locked_taskboard/replace_locked_taskboard public API under
its actual control-plane ownership. This function cannot write task status,
alter attempt counters, create a phase projection, or bypass the closed CLI.
    """
    entry = _entry(handle)
    _context(current_context)
    require(current_context == entry["context"], "actual_candidate_context_changed")
    require(report_raw == entry["report"] and source_documents == entry["sources"] and frozen_files == entry["frozen"],
            "accepted_candidate_or_source_changed")
    facts.check_facts(report_raw, source_documents, frozen_files)
    require(board_raw == source_documents["execution_board"], "accepted_legacy_board_changed")
    proposal = completion_projection(board_raw, expected_board_sha256=facts.sha(entry["sources"]["execution_board"]))
    # Burn before the caller's publication. A failed or unknown mutation must
    # be inspected, never retried by replaying this capability or resetting it.
    del _HANDLES[handle]
    summary = dict(entry["summary"], accepted_board_sha256=facts.sha(board_raw), proposed_board_sha256=facts.sha(proposal))
    return proposal, summary
