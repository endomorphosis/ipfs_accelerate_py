"""Bounded source-reader accounting retained after isolated worker cleanup.

Counts describe delivered representations, not unique source coverage or model
tokens. Missing/incomplete selected evidence is unavailable, never zero usage.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_request_observation as request_observation
from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _protected_directory
from .terminal_output_observation import _read, _open_directory, _identity
from .terminal_numerical_retrieval_observation import (
    _need, _sha, _parse, _records, _AUDIT_FIELDS, _COUNT_FIELDS, MAX_AUDIT_BYTES, MAX_SESSIONS,
)

SCHEMA = "terminal-source-view-observation@1"
COUNT_FIELDS = tuple(key.replace("rows", "source_bytes") for key in _COUNT_FIELDS)
OBSERVATION_FIELDS = {"schema", "descriptor_sha256", "session_id", *protocol.BINDING_FIELDS,
    "boundary_manifest_sha256", "guidance_sha256", "guidance_bytes", "requests_observed",
    "proof_authority", "execution_authority", "completion_authority"}
SHARED_PIN_FIELDS = ("view_policy_binding_sha256", "source_scope_sha256")
DISCOVERY_COUNT_FIELDS = ("prepared_pages", "delivered_pages", "prepared_rows", "delivered_rows")


def _base(status="unavailable", reason="evidence_unavailable"):
    return dict(schema=SCHEMA, status=status, reason=reason, sessions=[], counts=None, discovery_counts=None,
        observation_only=True, proof_authority=False, execution_authority=False,
        completion_authority=False, settlement_authority=False, retry_authority=False,
        source_obligations_discharged=0, provider_calls=0, provider_tokens=None,
        total_token_savings_measured=False, raw_queries_exported=False, raw_sources_exported=False)


def _candidate_exclusions(result, rows):
    """Never describe baseline-only accounting as all candidate retrieval."""
    from .terminal_candidate_round_observation import MARKERS, SCHEMA as candidate_schema
    excluded = sum(bool(MARKERS.intersection(row)) for row in rows)
    if excluded:
        result.update(session_scope="baseline_source_sessions_only",
            candidate_sessions_excluded=excluded, candidate_session_observation_schema=candidate_schema,
            all_coding_sessions_observed=False)
        coverage = result.get("request_operation_coverage")
        if coverage is not None:
            coverage.update(all_sessions_observed=False, excluded_candidate_sessions=excluded,
                session_scope="baseline_source_sessions_only")
    return result


def provider_free_coding_application_observed(report):
    """Join requested controls to an observed Doctor materialization, not a route label.

    This only distinguishes an inapplicable model-input observation. It grants
    no completion authority and does not turn missing receipts into zero usage.
    """
    if type(report) is not dict or report.get("arm") != "full":
        return False
    application = report.get("coding_control_application")
    route = report.get("implementation_route")
    expected_schema = {"doctor_candidate": "native-doctor-candidate-materialization@1",
        "doctor_contract_candidate": "native-doctor-contract-candidate-materialization@1"}.get(route)
    if (expected_schema is None or type(application) is not dict or set(application) != {
            "schema", "status", "implementation_route", "requested", "model_coding_dispatched",
            "token_savings_established", "completion_authority"}
            or application["schema"] != "terminal-coding-control-application@1"
            or application["status"] != "not_applicable_provider_free"
            or application["implementation_route"] != route
            or any(application[key] is not False for key in (
                "model_coding_dispatched", "token_savings_established", "completion_authority"))):
        return False
    metadata = report.get("semantic_metadata_view", {})
    if type(metadata) is not dict or type(application["requested"]) is not dict:
        return False
    attempt_selection = {}
    if "candidate_rounds" in report or "doctor_coding_policy" in report:
        from .terminal_doctor_coding_policy import (
            validate_doctor_coding_selection, doctor_coding_selection, doctor_coding_runtime_declared_matches,
        )
        try:
            rounds, policy = report.get("candidate_rounds", 1), report.get("doctor_coding_policy")
            validate_doctor_coding_selection(rounds, policy, arm=report["arm"],
                source_view_profile=report.get("source_view_profile", "disabled"),
                source_edit_mode=report.get("source_edit_mode", "disabled"),
                task_profile=report.get("task_profile"))
            attempt_selection = doctor_coding_selection(rounds, policy)
            if rounds == 2 and not doctor_coding_runtime_declared_matches(report, rounds, policy):
                return False
        except (TypeError, ValueError):
            return False
    optional_selection = {}
    try:
        if "semantic_working_set" in report:
            from .terminal_semantic_working_set_policy import (
                validate_semantic_working_set, semantic_working_set_selection, semantic_working_set_observation,
            )
            working_set = validate_semantic_working_set(report["semantic_working_set"], arm=report["arm"],
                source_view_profile=report.get("source_view_profile", "disabled"),
                transport_schema=report.get("semantic_transport_schema", "supervisor-semantic-router-input@1"),
                metadata_view=metadata.get("requested_mode", "legacy"))
            if any(type(report.get(key)) is not type(value) or report.get(key) != value
                    for key, value in semantic_working_set_observation(working_set).items()):
                return False
            optional_selection.update(semantic_working_set_selection(working_set))
        if "grok_coding_max_turns" in report:
            from .benchmark_provider_profile import resolve_provider_profile
            from ipfs_accelerate_py.agent_supervisor.runtime.grok_coding_budget import (
                validate_grok_coding_max_turns, grok_coding_selection,
            )
            provider = report.get("provider_profile")
            if type(provider) is not dict or provider != resolve_provider_profile(provider.get("id")):
                return False
            turns = validate_grok_coding_max_turns(report["grok_coding_max_turns"], provider=provider["provider"])
            optional_selection.update(grok_coding_selection(turns))
        expected_requested = {
            "semantic_transport_schema": report.get("semantic_transport_schema", "supervisor-semantic-router-input@1"),
            "semantic_metadata_view": metadata.get("requested_mode", "legacy"),
            "coding_reply_mode": report.get("coding_reply_mode", "legacy"),
            "source_view_profile": report.get("source_view_profile", "disabled"),
            "source_edit_mode": report.get("source_edit_mode", "disabled"),
            "source_view_policy": report.get("source_view_policy"), **attempt_selection, **optional_selection}
        if protocol.canonical_json(application["requested"]) != protocol.canonical_json(expected_requested):
            return False
    except (TypeError, ValueError):
        return False
    rows = report.get("provider_invocations")
    doctor = report.get("doctor_dispatch")
    materializations = report.get("doctor_invocations")
    if (type(rows) is not list or len(rows) > 128
            or any(type(row) is not dict or row.get("phase") != "planning" for row in rows)
            or (report.get("unreceipted_provider_attempt") is not None
                and report.get("unreceipted_provider_attempt") is not False)
            or type(report.get("worker_cleanup_returncode")) is not int
            or report["worker_cleanup_returncode"] != 0
            or type(doctor) is not dict or doctor.get("route") != route
            or doctor.get("status") != "candidate_ready"
            or type(doctor.get("provider_calls")) is not int or doctor["provider_calls"] != 0
            or type(doctor.get("task_cid")) is not str or not doctor["task_cid"]
            or type(materializations) is not list or len(materializations) != 1):
        return False
    receipt = materializations[0]
    return (type(receipt) is dict and receipt.get("schema") == expected_schema
        and receipt.get("task_cid") == doctor["task_cid"]
        and receipt.get("status") == "candidate_materialized"
        and type(receipt.get("provider_calls")) is int and receipt["provider_calls"] == 0
        and receipt.get("publication_authority") is False and receipt.get("completion_authority") is False
        and (route != "doctor_candidate" or type(doctor.get("candidate_commit")) is str
             and receipt.get("candidate_commit") == doctor["candidate_commit"]))


def _session(raw, row, scope, *, expected_policy_binding=None, expected_edit_mode="disabled",
             expected_semantic_working_set="legacy"):
    audit = _parse(raw)
    _need(set(audit) == _AUDIT_FIELDS and audit["schema"] in {
        "supervisor-worker-source-session@1", "supervisor-worker-source-session@2",
        "supervisor-worker-source-session@3", "supervisor-worker-source-session@4", request_observation.AUDIT_SCHEMA,
        request_observation.EVIDENCE_AUDIT_SCHEMA}
        and all(audit[key] is False for key in ("proof_authority", "execution_authority", "completion_authority")))
    for key in ("source_obligations_discharged", "retrieval_provider_calls"):
        _need(type(audit[key]) is int and audit[key] == 0)
    descriptor = protocol.validate_descriptor(audit["session"])
    evidence_enabled = descriptor["schema"] == "supervisor-worker-source-descriptor@4"
    _need(evidence_enabled is (expected_semantic_working_set == "capsule-references@1"))
    if evidence_enabled:
        from .terminal_context_audit import _receipt
        # Join the captured model input, selected projection and probe to this
        # session before treating any of its retrieval counts as qualified.
        _receipt(row, Path(row["workspace"]).parent)
        working_set = row.get("semantic_working_set")
        _need(type(working_set) is dict and working_set.get("requested_profile") == expected_semantic_working_set)
        if working_set["selected_mode"] != "legacy":
            probe = row["semantic_working_set_preflight"]
            _need(any(record.get("response_sha256") == probe["response_sha256"]
                and record.get("response_bytes") == probe["response_bytes"]
                and record.get("status") == "ok" and record.get("delivered") is True
                and record.get("semantic_evidence_page") is True
                and record.get("request_operation") == "semantic_evidence"
                for record in audit["requests"]))
    else:
        _need("semantic_working_set" not in row and "semantic_working_set_preflight" not in row)
    editing = descriptor.get("source_edit_mode", "disabled") != "disabled"
    shared = descriptor.get("view_policy_binding_sha256") is not None
    attributed = audit["schema"] in {request_observation.AUDIT_SCHEMA, request_observation.EVIDENCE_AUDIT_SCHEMA}
    _need((audit["schema"] == request_observation.EVIDENCE_AUDIT_SCHEMA) is evidence_enabled)
    _need((attributed or ((audit["schema"] == "supervisor-worker-source-session@4") is editing
        and (editing or (audit["schema"] == "supervisor-worker-source-session@3") is shared)))
        and descriptor.get("source_edit_mode", "disabled") == expected_edit_mode)
    observation = row["source_view_retrieval"]
    expected_fields = OBSERVATION_FIELDS | set(SHARED_PIN_FIELDS) if shared or editing or evidence_enabled else OBSERVATION_FIELDS
    if editing or evidence_enabled:
        expected_fields = expected_fields | {"source_edit_mode"}
    if evidence_enabled:
        expected_fields = expected_fields | {"semantic_evidence_mode"}
    _need(type(observation) is dict and set(observation) == expected_fields
        and observation["schema"] == ("worker-source-dispatch-observation@4" if evidence_enabled else "worker-source-dispatch-observation@3" if editing else "worker-source-dispatch-observation@2" if shared
                                      else "worker-source-dispatch-observation@1")
        and all(observation[key] is False for key in ("requests_observed", "proof_authority",
                                                     "execution_authority", "completion_authority")))
    _need(audit["descriptor_sha256"] == observation["descriptor_sha256"]
        == _sha(protocol.canonical_json(descriptor))
        and descriptor["session_id"] == observation["session_id"]
        and all(descriptor[key] == observation[key] for key in (*protocol.BINDING_FIELDS, "boundary_manifest_sha256"))
        and descriptor["task_id"] == scope["task_alias"]
        and descriptor["workspace"] == row["workspace"] and descriptor["owner_uid"] == os.geteuid())
    if shared or editing or evidence_enabled:
        _need(all(observation[key] == descriptor[key] for key in SHARED_PIN_FIELDS))
    if evidence_enabled:
        _need(observation["semantic_evidence_mode"] == descriptor["semantic_evidence_mode"] == "selected-capsules@1")
    if editing or evidence_enabled:
        _need(observation["source_edit_mode"] == expected_edit_mode)
    if expected_policy_binding is not None:
        _need(shared and descriptor["view_policy_binding_sha256"] == expected_policy_binding)
    semantic = row["semantic_translation"]
    _need(type(semantic) is dict and all(semantic.get(key) == descriptor[key] for key in (
        "task_id", "scope_cid", "semantic_root_cid", "semantic_sha256", "native_prompt_sha256")))
    boundary = row["external_container_boundary"]
    _need(type(boundary) is dict and boundary.get("manifest_sha256") == descriptor["boundary_manifest_sha256"]
        and all(type(boundary.get(key)) is int and boundary[key] == descriptor[key] for key in ("owner_uid", "worker_uid")))
    advice = protocol.guidance(descriptor["profile"],
        **({"source_edit_mode": expected_edit_mode} if editing else {})).encode()
    _need(observation["guidance_sha256"] == _sha(advice)
        and type(observation["guidance_bytes"]) is int and observation["guidance_bytes"] == len(advice))
    candidate_counts = None
    discovery_counts = None
    edit_counts = None
    evidence_counts = None
    accounting_audit = audit
    if audit["schema"] in {"supervisor-worker-source-session@2", "supervisor-worker-source-session@3", "supervisor-worker-source-session@4", request_observation.AUDIT_SCHEMA, request_observation.EVIDENCE_AUDIT_SCHEMA}:
        candidate_counts = dict(prepared_pages=0, delivered_pages=0, prepared_bytes=0, delivered_bytes=0)
        discovery_counts = dict.fromkeys(DISCOVERY_COUNT_FIELDS, 0)
        if evidence_enabled:
            evidence_counts = dict(prepared_pages=0, delivered_pages=0, prepared_bytes=0, delivered_bytes=0)
        if editing:
            edit_counts = dict(prepared_mapping_pages=0, delivered_mapping_pages=0,
                prepared_confirmation_pages=0, delivered_confirmation_pages=0)
        _need(type(audit["requests"]) is list)
        records, discovery_typed = [], []
        for item in audit["requests"]:
            _need(type(item) is dict)
            record = dict(item)
            if attributed:
                request_observation.validate_record(record)
                for field in request_observation.FIELDS:
                    record.pop(field)
            if evidence_enabled:
                evidence = record.pop("semantic_evidence_page", None)
                evidence_size = record.pop("returned_semantic_evidence_bytes", None)
                _need(type(evidence) is bool and type(evidence_size) is int
                    and 0 <= evidence_size <= protocol.MAX_FRAME_BYTES
                    and (evidence or evidence_size == 0)
                    and type(record.get("response_bytes")) is int and evidence_size <= record["response_bytes"]
                    and (not evidence or (record.get("status") == "ok" and evidence_size > 0
                        and record.get("returned_source_bytes") == 0
                        and record.get("returned_candidate_context_bytes") == 0
                        and record.get("candidate_context_page") is False
                        and record.get("discovery_page") is False
                        and record.get("returned_discovery_rows") == 0
                        and record.get("source_edit_page", False) is False)))
                evidence_counts["prepared_pages"] += evidence
                evidence_counts["prepared_bytes"] += evidence_size
                if record.get("delivered") is True:
                    evidence_counts["delivered_pages"] += evidence
                    evidence_counts["delivered_bytes"] += evidence_size
            if editing:
                edit = record.pop("source_edit_page", None)
                operation = record.pop("source_edit_operation", None)
                _need(type(edit) is bool and record.pop("owner_writes_performed", None) is False
                    and (operation in {"prepare", "confirm"} if edit else operation is None)
                    and (not edit or record.get("status") == "ok"))
                if edit:
                    _need(record.get("returned_source_bytes") == 0
                        and record.get("returned_candidate_context_bytes") == 0
                        and record.get("candidate_context_page") is False
                        and record.get("discovery_page") is False
                        and record.get("returned_discovery_rows") == 0)
                    kind = "mapping" if operation == "prepare" else "confirmation"
                    edit_counts["prepared_" + kind + "_pages"] += 1
                    if record.get("delivered") is True:
                        edit_counts["delivered_" + kind + "_pages"] += 1
            size = record.pop("returned_candidate_context_bytes", None)
            candidate = record.pop("candidate_context_page", None)
            pair = {"discovery_page", "returned_discovery_rows"} & set(record)
            _need(not pair or pair == {"discovery_page", "returned_discovery_rows"})
            _need((not shared and not editing and not attributed) or bool(pair))
            discovery_typed.append(bool(pair))
            discovery = record.pop("discovery_page", False)
            rows = record.pop("returned_discovery_rows", 0)
            _need(type(discovery) is bool and type(rows) is int and 0 <= rows <= 64
                and (discovery or rows == 0)
                and (not discovery or (record.get("status") == "ok" and rows > 0
                    and record.get("returned_source_bytes") == 0 and size == 0 and candidate is False)))
            _need(type(size) is int and 0 <= size <= protocol.MAX_FRAME_BYTES
                and type(candidate) is bool and (candidate or size == 0)
                and (not candidate or (record.get("status") == "ok" and size > 0
                    and record.get("returned_source_bytes") == 0 and not discovery))
                and type(record.get("response_bytes")) is int and size <= record["response_bytes"])
            candidate_counts["prepared_pages"] += candidate
            candidate_counts["prepared_bytes"] += size
            discovery_counts["prepared_pages"] += discovery
            discovery_counts["prepared_rows"] += rows
            if record.get("delivered") is True:
                candidate_counts["delivered_pages"] += candidate
                candidate_counts["delivered_bytes"] += size
                discovery_counts["delivered_pages"] += discovery
                discovery_counts["delivered_rows"] += rows
            records.append(record)
        if not shared and not attributed and (not discovery_typed or not all(discovery_typed)):
            discovery_counts = None
        accounting_audit = dict(audit, requests=records)
    counts, statuses, requests, responses = _records(accounting_audit, descriptor,
        detail_field="returned_source_bytes", max_detail=protocol.MAX_FRAME_BYTES, count_noun="source_bytes")
    _need(all(item["returned_source_bytes"] <= item["response_bytes"] for item in audit["requests"]))
    result = dict(session_id_sha256=_sha(descriptor["session_id"].encode()),
        invocation_id_sha256=_sha(row["invocation_id"].encode()), audit_sha256=_sha(raw),
        descriptor_sha256=audit["descriptor_sha256"], profile=descriptor["profile"],
        semantic_sha256=descriptor["semantic_sha256"], source_manifest_sha256=descriptor["source_manifest_sha256"],
        counts=counts, candidate_context_counts=candidate_counts, discovery_counts=discovery_counts,
        status_counts=dict(sorted(statuses.items())),
        unique_request_hash_count=len(requests), unique_response_hash_count=len(responses))
    if editing:
        result.update(source_edit_mode=expected_edit_mode, source_edit_counts=edit_counts,
            owner_writes_performed=False, task_validation_observed=False)
    if evidence_enabled:
        result["semantic_evidence_counts"] = evidence_counts
    if attributed:
        result["request_operation_counts"] = request_observation.operation_counts(audit["requests"])
        _need(result["request_operation_counts"]["accepted_requests"] == counts["accepted_requests"])
    return result, requests, responses


def collect(*, parent: Path, scope: dict | None, report: dict) -> dict:
    try:
        _need(type(report) is dict)
        rows = report.get("provider_invocations", [])
        _need(type(rows) is list and len(rows) <= 128 and all(type(row) is dict for row in rows))
        profile = report.get("source_view_profile", "disabled")
        _need(type(profile) is str and profile in ("disabled", *protocol.PROFILES))
        from .terminal_semantic_working_set_policy import validate_semantic_working_set
        working_set = validate_semantic_working_set(report.get("semantic_working_set", "legacy"),
            arm=report.get("arm"), source_view_profile=profile,
            transport_schema=report.get("semantic_transport_schema", "supervisor-semantic-router-input@1"),
            metadata_view=report.get("semantic_metadata_view", "legacy"))
        from .terminal_source_edit_policy import validate_source_edit_mode
        edit_mode = validate_source_edit_mode(report.get("source_edit_mode", "disabled"),
            arm=report.get("arm"), source_view_profile=profile)
        from .terminal_source_view_policy import validate_source_view_policy, source_view_observation
        policy = validate_source_view_policy(report.get("source_view_policy"), profile=profile,
            arm=report.get("arm"))
        policy_fields = {"source_view_policy_sha256", "source_view_policy_binding_sha256"}
        expected_policy_binding = None
        if policy is not None:
            _need(report.get("arm") == "full")
            expected = source_view_observation(profile, policy)
            _need(all(report.get(key) == expected[key] for key in policy_fields))
            expected_policy_binding = expected["source_view_policy_binding_sha256"]
        else:
            _need(not policy_fields.intersection(report))
        selected = [row for row in rows if "source_view_retrieval" in row]
        if not selected:
            if provider_free_coding_application_observed(report):
                return _base("not_applicable", "provider_free_doctor_materialization_observed")
            if profile != "disabled":
                return _candidate_exclusions(_base(reason="selected_session_receipt_unavailable"), rows)
            return _candidate_exclusions(_base("not_selected", "no_source_view_router_observation"), rows)
        _need(profile != "disabled")
        _need(len(selected) <= MAX_SESSIONS and type(scope) is dict
            and type(scope.get("task_alias")) is str and bool(scope["task_alias"])
            and type(report.get("worker_cleanup_returncode")) is int and report["worker_cleanup_returncode"] == 0)
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
                    and type(row.get("invocation_id")) is str and 0 < len(row["invocation_id"]) <= 512
                    and row["invocation_id"] not in invocations)
                invocations.add(row["invocation_id"])
                observation = row["source_view_retrieval"]
                _need(type(observation) is dict and observation.get("profile") == profile
                    and type(observation.get("session_id")) is str
                    and re.fullmatch("[0-9a-f]{32}", observation["session_id"]) is not None
                    and observation["session_id"] not in ids)
                ids.add(observation["session_id"])
                name = "source-view-" + observation["session_id"] + "-audit.json"
                before = os.stat(name, dir_fd=fd, follow_symlinks=False)
                _need(stat.S_IMODE(before.st_mode) == 0o400 and before.st_gid == directory.st_gid)
                raw = _read(fd, name, limit=MAX_AUDIT_BYTES, private=True)
                after = os.stat(name, dir_fd=fd, follow_symlinks=False)
                _need(_identity(before) == _identity(after) and before.st_gid == after.st_gid)
                item, request_hashes, response_hashes = _session(raw, row, scope,
                    expected_policy_binding=expected_policy_binding, expected_edit_mode=edit_mode,
                    expected_semantic_working_set=working_set)
                sessions.append(item); requests.update(request_hashes); responses.update(response_hashes)
            _protected_directory(parent, owner=owner, group=owner)
            _need(_identity(directory) == _identity(parent.lstat()))
        finally:
            os.close(fd)
        counts = {key: sum(item["counts"][key] for item in sessions) for key in COUNT_FIELDS}
        counts.update(sessions=len(sessions), unique_request_hash_count=len(requests), unique_response_hash_count=len(responses))
        result = _base("observed", "matched_owner_session_accounting")
        result.update(sessions=sessions, counts=counts,
            response_byte_scope="canonical JSON response bodies; excludes frame prefixes and CLI newlines",
            source_byte_scope="source slices including repeated reads; excludes candidate context artifacts and discovery metadata")
        result["candidate_context_counts"] = ({key: sum(item["candidate_context_counts"][key] for item in sessions)
            for key in ("prepared_pages", "delivered_pages", "prepared_bytes", "delivered_bytes")}
            if all(item["candidate_context_counts"] is not None for item in sessions) else None)
        if any("semantic_evidence_counts" in item for item in sessions):
            result["semantic_evidence_counts"] = ({key: sum(item["semantic_evidence_counts"][key] for item in sessions)
                for key in ("prepared_pages", "delivered_pages", "prepared_bytes", "delivered_bytes")}
                if all("semantic_evidence_counts" in item for item in sessions) else None)
        result["discovery_counts"] = ({key: sum(item["discovery_counts"][key] for item in sessions)
            for key in DISCOVERY_COUNT_FIELDS}
            if all(item["discovery_counts"] is not None for item in sessions) else None)
        if policy is not None:
            result.update({key: report[key] for key in policy_fields})
        if any("request_operation_counts" in item for item in sessions):
            observed = [item for item in sessions if "request_operation_counts" in item]
            attributed_requests = sum(item["counts"]["accepted_requests"] for item in observed)
            result["request_operation_counts"] = request_observation.combine_operation_counts(
                [item["request_operation_counts"] for item in observed])
            result["request_operation_coverage"] = dict(
                observed_sessions=len(observed), historical_sessions=len(sessions) - len(observed),
                observed_requests=attributed_requests,
                historical_unattributed_requests=counts["accepted_requests"] - attributed_requests,
                all_sessions_observed=len(observed) == len(sessions))
        if edit_mode != "disabled":
            fields = ("prepared_mapping_pages", "delivered_mapping_pages",
                "prepared_confirmation_pages", "delivered_confirmation_pages")
            result.update(source_edit_mode=edit_mode,
                source_edit_counts={key: sum(item["source_edit_counts"][key] for item in sessions) for key in fields},
                owner_writes_performed=False, task_validation_observed=False)
        _need(len(protocol.canonical_json(result)) <= MAX_AUDIT_BYTES)
        return _candidate_exclusions(result, rows)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError):
        return _base()
