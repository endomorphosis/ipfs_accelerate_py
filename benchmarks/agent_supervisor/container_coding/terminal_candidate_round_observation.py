"""Observe closed candidate readers separately from baseline source sessions.

The owner audit establishes bounded response accounting, including the mandatory
summary preflight. These counts neither reconstruct model input nor establish
source semantics or token savings. Retained context replay is a separate audit.
"""
from __future__ import annotations

import os
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_dispatch_context as dispatch
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_request_observation as attribution
from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _protected_directory
from .terminal_numerical_retrieval_observation import (
    _need, _sha, _parse, _digest, _integer, _text, _records, _AUDIT_FIELDS, _COUNT_FIELDS,
    MAX_AUDIT_BYTES, MAX_SESSIONS,
)
from .terminal_output_observation import _open_directory, _identity
from .terminal_owner_candidate_refresh_observation import _private_read

SCHEMA = "terminal-candidate-round-observation@1"
AUDIT_SCHEMA = "supervisor-worker-candidate-round-session@1"
COUNT_FIELDS = tuple(key.replace("rows", "source_bytes") for key in _COUNT_FIELDS)
CANDIDATE_COUNT_FIELDS = ("prepared_pages", "delivered_pages", "prepared_bytes", "delivered_bytes")
MARKERS = frozenset({"candidate_round_dispatch", "candidate_round_preflight",
    "candidate_reply_contract", "candidate_retrieval_guidance"})
JOIN_FIELDS = ("task_id", "native_prompt_sha256", "session_id", "prior_session_id",
    "prior_descriptor_sha256", "workspace_generation", "round_context_sha256", "round_number",
    "candidate_scope_cid", "candidate_semantic_root_cid")
OBSERVATION_FIELDS = frozenset({"schema", *JOIN_FIELDS, "descriptor_sha256", "handoff_sha256",
    "provider_input_sha256", "provider_input_bytes", "native_prompt_bytes", "candidate_artifacts",
    "producer_binding_sha256", "source_files", "expires_monotonic_ns",
    "dispatch_deadline_monotonic_ns", "remaining_timeout_seconds", "validation_phase",
    "owner_validation_inherited", "producer_validation_inherited", "neural_inference_replayed",
    "owner_private_state_replayed_here", "canonical_source_current", "workspace_source_current",
    *dispatch.FLAGS})
PREFLIGHT_FIELDS = frozenset({"schema", "requests", "session_id", "workspace_generation", "artifact",
    "artifact_sha256", "response_sha256", "response_bytes", "provider_input_bytes",
    "child_custody_checked", "source_freshness_checked", "producer_freshness_checked",
    "proof_authority", "completion_authority"})


def _base(status="unavailable", reason="evidence_unavailable"):
    return dict(schema=SCHEMA, status=status, reason=reason, sessions=[], counts=None,
        candidate_context_counts=None, preflight_counts=None, observation_only=True,
        proof_authority=False, execution_authority=False, completion_authority=False,
        settlement_authority=False, retry_authority=False, source_obligations_discharged=0,
        provider_calls=0, provider_tokens=None, total_token_savings_measured=False,
        raw_queries_exported=False, raw_sources_exported=False, raw_prompts_exported=False,
        session_scope="candidate_round_sessions_only", model_input_replayed=False,
        source_semantics_verified=False)


def _dispatch_binding(row, descriptor):
    observation = row["candidate_round_dispatch"]
    _need(type(observation) is dict)
    doctor = observation.get("schema") == dispatch.DOCTOR_OBSERVATION_SCHEMA
    fields = OBSERVATION_FIELDS | ({"doctor_candidate_dispatch"} if doctor else set())
    _need(set(observation) == fields
        and observation["schema"] == (dispatch.DOCTOR_OBSERVATION_SCHEMA if doctor else dispatch.OBSERVATION_SCHEMA)
        and all(type(observation[key]) is type(value) and observation[key] == value
            for key, value in dispatch.FLAGS.items())
        and all(observation[key] == descriptor[key] for key in JOIN_FIELDS)
        and observation["validation_phase"] == "before_provider"
        and all(observation[key] is True for key in ("owner_validation_inherited",
            "producer_validation_inherited", "canonical_source_current", "workspace_source_current"))
        and all(observation[key] is False for key in ("neural_inference_replayed",
            "owner_private_state_replayed_here")))
    if doctor:
        from ipfs_accelerate_py.agent_supervisor.runtime.candidate_doctor_dispatch import validate_doctor_dispatch_observation
        validate_doctor_dispatch_observation(observation["doctor_candidate_dispatch"])
    for key in ("descriptor_sha256", "handoff_sha256", "provider_input_sha256", "producer_binding_sha256"):
        _need(_digest(observation[key]))
    _need(type(observation["round_number"]) is int and observation["round_number"] == 2
        and observation["descriptor_sha256"] == _sha(protocol.canonical_json(descriptor))
        and row.get("native_prompt_sha256") == observation["native_prompt_sha256"]
        and row.get("prompt_sha256") == observation["provider_input_sha256"]
        and type(row.get("native_prompt_bytes")) is int
        and row["native_prompt_bytes"] == observation["native_prompt_bytes"]
        and type(row.get("prompt_bytes")) is int and row["prompt_bytes"] == observation["provider_input_bytes"]
        and _integer(observation["provider_input_bytes"], 1, dispatch.MAX_HANDOFF_BYTES)
        and _integer(observation["native_prompt_bytes"], 1, dispatch.MAX_HANDOFF_BYTES)
        and _integer(observation["source_files"], 1, 768)
        and _integer(observation["expires_monotonic_ns"], descriptor["issued_monotonic_ns"] + 1,
            descriptor["expires_monotonic_ns"])
        and _integer(observation["dispatch_deadline_monotonic_ns"], observation["expires_monotonic_ns"],
            descriptor["expires_monotonic_ns"])
        and _integer(observation["remaining_timeout_seconds"], 0, protocol.MAX_SESSION_SECONDS)
        and row.get("semantic_translation") is None
        and not {"source_view_retrieval", "numerical_obligation_retrieval",
            "source_lookup_retrieval", "source_formalization_context"}.intersection(row))
    artifacts = observation["candidate_artifacts"]
    _need(type(artifacts) is dict and set(artifacts) == {"summary", "capsules", "formal"})
    for pin in artifacts.values():
        _need(type(pin) is dict and set(pin) == {"sha256", "bytes"}
            and _digest(pin["sha256"]) and _integer(pin["bytes"], 1, 16 * 1024 * 1024))
    # Guidance is fixed code plus this candidate generation, not an arbitrary
    # worker-provided string or a token measurement.
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import _candidate_round_guidance
    guidance = _candidate_round_guidance(observation).encode()
    expected = dict(sha256=_sha(guidance), bytes=len(guidance), requests_observed=False,
        proof_authority=False, completion_authority=False)
    actual = row.get("candidate_retrieval_guidance")
    _need(type(actual) is dict and set(actual) == set(expected)
        and all(type(actual[key]) is type(value) and actual[key] == value for key, value in expected.items()))
    return observation


def _preflight(row, descriptor, observation, records):
    probe = row["candidate_round_preflight"]
    _need(type(probe) is dict and set(probe) == PREFLIGHT_FIELDS
        and probe["schema"] == "supervisor-candidate-round-preflight@1"
        and type(probe["requests"]) is int and probe["requests"] == 1
        and probe["session_id"] == descriptor["session_id"]
        and probe["workspace_generation"] == descriptor["workspace_generation"]
        and probe["artifact"] == "summary"
        and probe["artifact_sha256"] == observation["candidate_artifacts"]["summary"]["sha256"]
        and _digest(probe["response_sha256"]) and _integer(probe["response_bytes"], 1, 8192)
        and type(probe["provider_input_bytes"]) is int and probe["provider_input_bytes"] == 0
        and all(probe[key] is True for key in ("child_custody_checked", "source_freshness_checked",
            "producer_freshness_checked"))
        and all(probe[key] is False for key in ("proof_authority", "completion_authority")))
    matches = [record for record in records if record["response_sha256"] == probe["response_sha256"]]
    _need(len(matches) == 1 and matches[0]["response_bytes"] == probe["response_bytes"]
        and matches[0]["status"] == "ok" and matches[0]["delivered"] is True
        and matches[0]["candidate_context_page"] is True
        and matches[0]["request_operation"] == "candidate_context")
    return dict(requests=1, response_bytes=probe["response_bytes"],
        candidate_context_bytes=matches[0]["returned_candidate_context_bytes"], provider_input_bytes=0)


def _session(raw, row, scope, *, profile, policy_binding):
    audit = _parse(raw)
    _need(set(audit) == _AUDIT_FIELDS and audit["schema"] == AUDIT_SCHEMA
        and all(audit[key] is False for key in ("proof_authority", "execution_authority", "completion_authority"))
        and all(type(audit[key]) is int and audit[key] == 0
            for key in ("source_obligations_discharged", "retrieval_provider_calls")))
    descriptor = protocol.validate_descriptor(audit["session"])
    observation = _dispatch_binding(row, descriptor)
    _need(descriptor["task_id"] == scope["task_alias"] and descriptor["workspace"] == row["workspace"]
        and descriptor["owner_uid"] == os.geteuid() and descriptor["profile"] == profile
        and descriptor["view_policy_binding_sha256"] == policy_binding
        and audit["descriptor_sha256"] == observation["descriptor_sha256"])
    boundary = row["external_container_boundary"]
    _need(type(boundary) is dict and boundary.get("manifest_sha256") == descriptor["boundary_manifest_sha256"]
        and all(type(boundary.get(key)) is int and boundary[key] == descriptor[key]
            for key in ("owner_uid", "worker_uid")))
    records, candidate_counts = [], dict.fromkeys(CANDIDATE_COUNT_FIELDS, 0)
    _need(type(audit["requests"]) is list and len(audit["requests"]) <= descriptor["max_requests"])
    for item in audit["requests"]:
        _need(type(item) is dict)
        selected = attribution.validate_record(item)
        _need(selected == attribution.UNKNOWN or (selected[0] in {"read", "candidate_context"}
            and selected[1] == "working-candidate"))
        record = dict(item)
        for key in attribution.FIELDS:
            record.pop(key)
        _need(record.pop("discovery_page", None) is False
            and type(record.get("returned_discovery_rows")) is int
            and record.pop("returned_discovery_rows") == 0)
        candidate = record.pop("candidate_context_page", None)
        size = record.pop("returned_candidate_context_bytes", None)
        _need(type(candidate) is bool and _integer(size, 0, protocol.MAX_FRAME_BYTES)
            and (candidate or size == 0)
            and type(record.get("response_bytes")) is int and size <= record["response_bytes"]
            and (not candidate or (record.get("status") == "ok" and size > 0
                and type(record.get("returned_source_bytes")) is int and record["returned_source_bytes"] == 0)))
        candidate_counts["prepared_pages"] += candidate
        candidate_counts["prepared_bytes"] += size
        if record.get("delivered") is True:
            candidate_counts["delivered_pages"] += candidate
            candidate_counts["delivered_bytes"] += size
        records.append(record)
    counts, statuses, requests, responses = _records(dict(audit, requests=records), descriptor,
        detail_field="returned_source_bytes", max_detail=protocol.MAX_FRAME_BYTES, count_noun="source_bytes")
    _need(all(item["returned_source_bytes"] <= item["response_bytes"] for item in records))
    probe = _preflight(row, descriptor, observation, audit["requests"])
    operations = attribution.operation_counts(audit["requests"])
    _need(operations["accepted_requests"] == counts["accepted_requests"])
    return dict(session_id_sha256=_sha(descriptor["session_id"].encode()),
        prior_session_id_sha256=_sha(descriptor["prior_session_id"].encode()),
        invocation_id_sha256=_sha(row["invocation_id"].encode()), audit_sha256=_sha(raw),
        descriptor_sha256=audit["descriptor_sha256"], prior_descriptor_sha256=descriptor["prior_descriptor_sha256"],
        workspace_generation=descriptor["workspace_generation"], round_context_sha256=descriptor["round_context_sha256"],
        native_prompt_sha256=descriptor["native_prompt_sha256"], handoff_sha256=observation["handoff_sha256"],
        candidate_artifacts=observation["candidate_artifacts"], round_number=2, profile=descriptor["profile"],
        counts=counts, candidate_context_counts=candidate_counts, preflight_counts=probe,
        request_operation_counts=operations, status_counts=dict(sorted(statuses.items())),
        unique_request_hash_count=len(requests), unique_response_hash_count=len(responses)), requests, responses


def collect(*, parent: Path, scope: dict | None, report: dict) -> dict:
    """Require every selected candidate receipt's exact, closed owner audit.

    Expired descriptors are historical evidence. Missing receipts, incomplete
    audits or a failed mandatory preflight yield unknown counts, never zero.
    """
    try:
        _need(type(report) is dict)
        rows = report.get("provider_invocations", [])
        _need(type(rows) is list and len(rows) <= 128 and all(type(row) is dict for row in rows))
        selected = [row for row in rows if MARKERS.intersection(row)]
        if not selected:
            if report.get("candidate_rounds") == 2 and not (
                    report.get("unreceipted_provider_attempt") is None
                    or report.get("unreceipted_provider_attempt") is False):
                return _base(reason="selected_session_receipt_unavailable")
            return _base("not_selected", "no_candidate_round_router_observation")
        _need(len(selected) <= MAX_SESSIONS
            and (report.get("unreceipted_provider_attempt") is None
                or report.get("unreceipted_provider_attempt") is False)
            and type(scope) is dict and _text(scope.get("task_alias")))
        if type(report.get("worker_cleanup_returncode")) is not int or report["worker_cleanup_returncode"] != 0:
            return _base(reason="worker_cleanup_unavailable")
        profile = report.get("source_view_profile", "disabled")
        _need(type(profile) is str and profile in protocol.PROFILES)
        from .terminal_source_view_policy import validate_source_view_policy, source_view_observation
        policy = validate_source_view_policy(report.get("source_view_policy"), profile=profile, arm=report.get("arm"))
        policy_binding = None
        policy_fields = {"source_view_policy_sha256", "source_view_policy_binding_sha256"}
        if policy is None:
            _need(not policy_fields.intersection(report))
        else:
            expected = source_view_observation(profile, policy)
            _need(all(report.get(key) == expected[key] for key in policy_fields))
            policy_binding = expected["source_view_policy_binding_sha256"]
        owner = os.geteuid()
        _need(owner > 0 and owner == os.getuid())
        parent = Path(parent)
        _protected_directory(parent, owner=owner, group=owner)
        fd = _open_directory(parent)
        try:
            directory = os.fstat(fd)
            _need(_identity(directory) == _identity(parent.lstat()))
            sessions, ids, invocations, requests, responses = [], set(), set(), set(), set()
            for row in selected:
                _need(row.get("schema") == "router-implementation-invocation@1"
                    and row.get("phase") == row.get("purpose") == "coding"
                    and row.get("status") in {"failed", "provider_returned"}
                    and _text(row.get("invocation_id")) and row["invocation_id"] not in invocations)
                invocations.add(row["invocation_id"])
                observation = row["candidate_round_dispatch"]
                _need(type(observation) is dict and protocol._hex(observation.get("session_id"), 32)
                    and observation["session_id"] not in ids)
                ids.add(observation["session_id"])
                raw = _private_read(fd, "round-view-" + observation["session_id"] + "-audit.json",
                    directory, limit=MAX_AUDIT_BYTES)
                item, request_hashes, response_hashes = _session(raw, row, scope,
                    profile=profile, policy_binding=policy_binding)
                prior = [earlier for earlier in rows if type(earlier.get("source_view_retrieval")) is dict
                    and earlier["source_view_retrieval"].get("session_id") == observation["prior_session_id"]]
                _need(len(prior) <= 1)
                if prior:
                    _need(prior[0].get("schema") == "router-implementation-invocation@1"
                        and prior[0].get("phase") == prior[0].get("purpose") == "coding"
                        and prior[0].get("workspace") == row["workspace"]
                        and prior[0]["source_view_retrieval"].get("descriptor_sha256") == observation["prior_descriptor_sha256"]
                        and all(prior[0]["source_view_retrieval"].get(key) == observation[key]
                            for key in ("task_id", "native_prompt_sha256")))
                item["prior_router_receipt_matched"] = bool(prior)
                sessions.append(item); requests.update(request_hashes); responses.update(response_hashes)
            _protected_directory(parent, owner=owner, group=owner)
            _need(_identity(directory) == _identity(parent.lstat()))
        finally:
            os.close(fd)
        result = _base("observed", "matched_owner_candidate_session_accounting")
        counts = {key: sum(item["counts"][key] for item in sessions) for key in COUNT_FIELDS}
        counts.update(sessions=len(sessions), unique_request_hash_count=len(requests), unique_response_hash_count=len(responses))
        result.update(sessions=sessions, counts=counts,
            candidate_context_counts={key: sum(item["candidate_context_counts"][key] for item in sessions)
                for key in CANDIDATE_COUNT_FIELDS},
            preflight_counts={key: sum(item["preflight_counts"][key] for item in sessions)
                for key in ("requests", "response_bytes", "candidate_context_bytes", "provider_input_bytes")},
            request_operation_counts=attribution.combine_operation_counts([item["request_operation_counts"] for item in sessions]),
            response_byte_scope="canonical JSON responses including preflight; excludes frame prefixes and CLI newlines",
            source_byte_scope="candidate source slices including repeated reads; excludes advisory artifact bytes",
            candidate_context_byte_scope="advisory artifact slices including repeated reads and preflight; not model tokens")
        _need(len(protocol.canonical_json(result)) <= MAX_AUDIT_BYTES)
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError, OverflowError):
        return _base()


__all__ = ["collect"]
