"""Read retained, sealed owner audits after worker cleanup, without state access.

Only explicitly bound sessions are read. A complete population is required;
partial or ambiguous evidence never becomes a zero count. Versioned operation
witnesses can associate results with completed owner socket sends; they cannot
establish worker receipt or model ingestion. Optional invocation-row captures
add separately labelled worker reports without changing those trust limits.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as binding_api
from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _protected_directory
from .terminal_output_observation import _open_directory, _read, _identity, _scope
from .terminal_repository_owner_audit import MAX_AUDIT_BYTES, summarize_owner_session
from .terminal_repository_owner_audit import OPERATION_COUNTERS, operation_correlation_unavailable
from . import terminal_repository_owner_lifecycle as lifecycle
from . import terminal_repository_worker_forwarding as forwarding

SCHEMA = "terminal-repository-owner-observation@1"
MAX_SESSIONS = 16
MAX_INVOCATIONS = 128
MAX_OPERATIONS_BYTES = 512 * 1024
_COUNTERS = ("initializations", "tool_calls", "successful_reads", "compact_reads",
    "canonical_edits", "uncertain_edits", "refreshes", "errors", "returned_bytes")
_REASONS = {"evidence_unavailable", "collector_failed", "not_selected", "population_unavailable",
    "scope_unavailable", "cleanup_unavailable", "session_join_unavailable", "retained_evidence_unavailable"}


def unavailable(reason="evidence_unavailable"):
    """Closed, content-free fallback, also used by the durable report writer."""
    return dict(schema=SCHEMA, status="unavailable",
        reason=reason if type(reason) is str and reason in _REASONS else "evidence_unavailable",
        sessions=[], session_count=None, transport=None, prepared_mcp_counts=None,
        delivered_tool_calls=None, delivered_reads=None, model_consumed_tokens=None,
        unknown_reason="missing_per_operation_correlation_and_model_ingestion_evidence",
        observation_only=True, proof_authority=False, execution_authority=False,
        completion_authority=False, task_correctness_established=False,
        source_semantics_verified=False, provider_ingestion_verified=False,
        token_savings_measured=False, raw_content_exported=False, observer_provider_calls=0,
        native_parent_release_verified=False, worker_cleanup_independently_verified=False)


def _digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _operation_population(sessions):
    """Sum only a complete population of explicit, byte-joined witnesses."""
    observations = [item["operation_correlation"] for item in sessions]
    known = sum(item["status"] == "observed" for item in observations)
    complete = bool(sessions) and known == len(sessions)
    result = operation_correlation_unavailable()
    result.update(schema="terminal-repository-owner-operation-correlation-population@1",
        status="observed" if complete else "unavailable",
        reason=None if complete else "incomplete_operation_correlation_population",
        session_count=len(sessions), known_session_count=known,
        unknown_session_count=len(sessions) - known,
        counts={key: sum(item["counts"][key] for item in observations)
                for key in OPERATION_COUNTERS} if complete else None)
    return result


def _invocation_join(row, scope, owner, prepared):
    binding = row.get("repository_source_policy")
    binding_api.validate_binding(binding, workspace=row.get("workspace"), materialized=False)
    descriptor = binding["descriptor"]
    boundary = row.get("external_container_boundary")
    if (descriptor["owner_uid"] != owner or type(boundary) is not dict
            or boundary.get("schema") != "supervisor-container-worker-boundary@1"
            or any(type(boundary.get(key)) is not int or boundary[key] != descriptor[key]
                   for key in ("owner_uid", "worker_uid"))
            or boundary.get("manifest_sha256") != descriptor["boundary_manifest_sha256"]
            or boundary.get("purpose") != "coding"
            or boundary.get("completion_authority") is not False):
        raise ValueError("boundary join unavailable")
    inclusion = row.get("public_instruction")
    if (type(inclusion) is not dict
            or inclusion.get("schema") not in {"supervisor-public-instruction-inclusion@" + str(i) for i in range(1, 6)}
            or inclusion.get("task_cid") != scope["task_cid"]
            or inclusion.get("task_id") != scope["task_alias"]
            or any(not _digest(inclusion.get(key)) for key in ("artifact_sha256", "source_sha256", "block_sha256"))
            or inclusion.get("manifest_signature_verified") is not True
            or inclusion.get("verbatim_utf8") is not True
            or inclusion.get("source_freshness_verified") is not True
            or inclusion.get("historical_replay") is not False
            or inclusion.get("completion_authority") is not False
            or inclusion.get("scope_expansion_authority") is not False):
        raise ValueError("public instruction join unavailable")
    identities = ("artifact", "context_cid", "manifest_cid", "task_cid", "source_path", "source_sha256")
    if (type(prepared) is not dict or any(type(prepared.get(key)) is not str
            or not 0 < len(prepared[key].encode("utf-8")) <= 4096
            or inclusion.get(key) != prepared[key] for key in identities)
            or inclusion.get("artifact_sha256") != prepared.get("sha256")
            or type(prepared.get("source_bytes")) is not int or prepared["source_bytes"] < 0
            or type(inclusion.get("source_bytes")) is not int
            or inclusion["source_bytes"] != prepared["source_bytes"]
            or prepared.get("completion_authority") is not False
            or prepared.get("scope_expansion_authority") is not False
            or inclusion.get("semantic_minification_applied") is not False):
        raise ValueError("owner prepared instruction differs")
    # The receipt reports previous signature verification. This projection does
    # not reopen the instruction artifact or independently repeat admission.
    return binding


def _sealed_read(fd, name, *, owner, group, limit):
    before = os.stat(name, dir_fd=fd, follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_uid != owner or before.st_gid != group or before.st_nlink != 1):
        raise ValueError("sealed owner evidence required")
    raw = _read(fd, name, limit=limit, private=True)
    after = os.stat(name, dir_fd=fd, follow_symlinks=False)
    if _identity(before) != _identity(after) or before.st_gid != after.st_gid:
        raise ValueError("sealed evidence changed")
    return raw


def collect(*, parent, scope, report):
    """Return a bounded observation without changing binding/usage/adoption."""
    if type(report) is not dict:
        return unavailable("population_unavailable")
    rows = report.get("provider_invocations", [])
    if type(rows) is not list or len(rows) > MAX_INVOCATIONS or any(type(row) is not dict for row in rows):
        return unavailable("population_unavailable")
    selected = [row for row in rows if type(row.get("repository_source_policy")) is dict
        and row["repository_source_policy"].get("schema") == binding_api.SCHEMA]
    policy = report.get("repository_source_policy", "legacy")
    if not selected:
        result = unavailable("not_selected" if policy == "legacy" else "population_unavailable")
        if policy == "legacy":
            result["status"] = "not_selected"
        return result
    if (len(selected) > MAX_SESSIONS or report.get("unreceipted_provider_attempt") is not None
            or any(row.get("phase") == "coding" and row not in selected for row in rows)):
        return unavailable("population_unavailable")
    if type(report.get("worker_cleanup_returncode")) is not int or report["worker_cleanup_returncode"] != 0:
        return unavailable("cleanup_unavailable")
    try:
        _scope(scope)
    except Exception:
        return unavailable("scope_unavailable")
    try:
        owner = os.geteuid()
        if owner <= 0 or os.getuid() != owner:
            raise ValueError("owner identity required")
        sessions, invocations, configurations = set(), set(), set()
        bindings = []
        for row in selected:
            binding = _invocation_join(row, scope, owner, report.get("public_instruction"))
            identity, session = row.get("invocation_id"), binding["session_id"]
            if (type(identity) is not str or re.fullmatch("[0-9a-f]{32}", identity) is None
                    or identity in invocations or session in sessions or binding["config_sha256"] in configurations
                    or sum(other.get("invocation_id") == identity for other in rows) != 1
                    or binding["requested_policy"] != policy):
                raise ValueError("ambiguous or mismatched population")
            invocations.add(identity); sessions.add(session); configurations.add(binding["config_sha256"])
            bindings.append(binding)
    except Exception:
        return unavailable("session_join_unavailable")
    try:
        parent = Path(parent).absolute()
        group = parent.lstat().st_gid
        _protected_directory(parent, owner=owner, group=group)
        fd = _open_directory(parent)
        try:
            before = os.fstat(fd)
            if _identity(before) != _identity(parent.lstat()) or before.st_gid != group:
                raise ValueError("parent changed")
            observed, lifecycle_observed, forwarding_observed = [], [], []
            for row, binding in zip(selected, bindings):
                stem = "repository-mcp-" + binding["session_id"] + "-audit"
                audit_raw = _sealed_read(fd, stem + ".json", owner=owner, group=group, limit=MAX_AUDIT_BYTES)
                operations_raw = _sealed_read(fd, stem + "-operations.jsonl", owner=owner,
                    group=group, limit=MAX_OPERATIONS_BYTES)
                session = summarize_owner_session(audit_raw=audit_raw, operations_raw=operations_raw, row=row)
                observed.append(session)
                # Missing or malformed optional worker reports never invalidate
                # the independently retained prepared-operation recount.
                try:
                    worker = forwarding.summarize_captured_worker_forwarding(
                        audit_raw=audit_raw, operations_raw=operations_raw, row=row)
                except Exception:
                    worker = forwarding.unavailable("invalid_worker_capture")
                worker["invocation_sha256"] = session["invocation_sha256"]
                forwarding_observed.append(worker)
                # A separately retained lifecycle receipt is optional. Historical
                # sessions keep their valid recount when this evidence is absent
                # or malformed; neither state implies zero or failed release.
                name = "repository-mcp-" + binding["session_id"] + "-lifecycle.json"
                try:
                    retirement_raw = _sealed_read(fd, name, owner=owner, group=group,
                        limit=lifecycle.MAX_RETIREMENT_BYTES)
                except FileNotFoundError:
                    retirement = lifecycle.unavailable()
                except Exception:
                    retirement = lifecycle.unavailable("invalid_retained_lifecycle_receipt")
                else:
                    try:
                        retirement = lifecycle.summarize_owner_lifecycle(
                            retirement_raw=retirement_raw, audit_raw=audit_raw)
                    except Exception:
                        retirement = lifecycle.unavailable("invalid_retained_lifecycle_receipt")
                retirement["invocation_sha256"] = session["invocation_sha256"]
                lifecycle_observed.append(retirement)
            _protected_directory(parent, owner=owner, group=group)
            after, named = os.fstat(fd), parent.lstat()
            if (_identity(before) != _identity(after) or _identity(after) != _identity(named)
                    or after.st_gid != group or named.st_gid != group):
                raise ValueError("parent changed during collection")
        finally:
            os.close(fd)
        # Only sum explicitly named counters. Nested event chains stay per
        # session; concatenating them would invent an absent shared ordering.
        totals = {key: sum(item["prepared_mcp_usage"][key] for item in observed) for key in _COUNTERS}
        transports = [item["transport"] for item in observed]
        transport = {key: sum(item[key] for item in transports) for key in (
            "accepted_requests", "prepared_responses", "prepared_response_bytes", "reserved_responses",
            "reserved_response_bytes", "completed_frame_sends", "completed_frame_send_bytes", "prepared_final_changes")}
        transport["status_counts"] = {key: sum(item["status_counts"].get(key, 0) for item in transports)
            for key in sorted({key for item in transports for key in item["status_counts"]})}
        transport["scope"] = "owner_outer_envelopes; frame_prefix_excluded; send_completion_is_not_model_ingestion"
        result = unavailable()
        result.update(status="observed", reason=None, sessions=observed, session_count=len(observed),
            transport=transport, prepared_mcp_counts=totals,
            operation_correlation=_operation_population(observed),
            lifecycle_observation=lifecycle.aggregate(lifecycle_observed),
            worker_forwarding_observation=forwarding.aggregate(forwarding_observed))
        if result["operation_correlation"]["status"] == "observed":
            result["unknown_reason"] = "missing_worker_receipt_and_model_ingestion_evidence"
        return result
    except Exception:
        return unavailable("retained_evidence_unavailable")
