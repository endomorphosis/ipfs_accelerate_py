"""Retain bounded owner-session accounting before the container is removed.

A matching task/router/session audit establishes read-only request observations,
not source correctness, native task settlement, proof or provider token usage.
No private database, source body or model query is opened or exported here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as protocol
from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _protected_directory
from .terminal_output_observation import _read, _open_directory, _identity

SCHEMA = "terminal-numerical-retrieval-observation@1"
MAX_SESSIONS = 16
MAX_AUDIT_BYTES = 65536
_AUDIT_FIELDS = frozenset({"schema", "session", "descriptor_sha256", "requests", "accepted_requests",
    "inflight_request", "response_bytes", "service_stopped", "accounting_complete",
    "retrieval_provider_calls", "source_obligations_discharged", "proof_authority",
    "execution_authority", "completion_authority"})
_RECORD_FIELDS = frozenset({"ordinal", "status", "request_sha256", "response_sha256",
    "response_bytes", "returned_rows", "delivered", "elapsed_ns"})
_OBSERVATION_FIELDS = frozenset({"schema", "descriptor_sha256", "session_id", "task_id", "task_cid",
    "context_bundle_sha256", "context_receipt_sha256", "source_tree_id", "boundary_manifest_sha256",
    "requests_observed", "proof_authority", "execution_authority", "completion_authority"})
_JOIN_FIELDS = ("task_id", "task_cid", "context_bundle_sha256", "context_receipt_sha256",
                "source_tree_id", "boundary_manifest_sha256")
_COUNT_FIELDS = ("accepted_requests", "delivered_responses", "prepared_pages", "delivered_pages",
                 "prepared_rows", "delivered_rows", "response_bytes", "delivered_response_bytes",
                 "request_hash_count", "response_hash_count")


def _need(condition):
    if not condition:
        raise ValueError("unavailable numerical retrieval evidence")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _text(value):
    return type(value) is str and 0 < len(value.encode("utf-8")) <= 512


def _base(status="unavailable", reason="evidence_unavailable"):
    return dict(schema=SCHEMA, status=status, reason=reason, sessions=[], counts=None,
        observation_only=True, proof_authority=False, execution_authority=False,
        completion_authority=False, settlement_authority=False, retry_authority=False,
        source_correctness_established=False, task_correctness_established=False,
        source_obligations_discharged=0, provider_calls=0, provider_tokens=None,
        token_estimate=False, raw_queries_exported=False, raw_sources_exported=False,
        scope="owner-session accounting matched to admitted task and router receipt after worker cleanup")


def _parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result)
            result[key] = value
        return result
    def refuse(_):
        raise ValueError("nonfinite audit JSON")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=refuse)
    _need(type(value) is dict and protocol.canonical_json(value) == raw)
    return value


def _records(audit, descriptor, *, detail_field="returned_rows", max_detail=8, count_noun="rows"):
    records = audit["requests"]
    _need(type(records) is list and len(records) <= descriptor["max_requests"]
          and _integer(audit["accepted_requests"], 0, 64)
          and audit["accepted_requests"] == len(records)
          and audit["inflight_request"] is None and audit["service_stopped"] is True
          and audit["accounting_complete"] is True)
    counts = dict.fromkeys((key.replace("rows", count_noun) for key in _COUNT_FIELDS), 0)
    record_fields = (_RECORD_FIELDS - {"returned_rows"}) | {detail_field}
    statuses, request_hashes, response_hashes = {}, set(), set()
    for ordinal, row in enumerate(records, 1):
        _need(type(row) is dict and set(row) == record_fields
              and type(row["ordinal"]) is int and row["ordinal"] == ordinal
              and type(row["status"]) is str and row["status"] in {"ok", *protocol.REASON_CODES}
              and type(row["delivered"]) is bool
              and _integer(row["response_bytes"], 0, protocol.MAX_FRAME_BYTES)
              and _integer(row[detail_field], 0, max_detail)
              and _integer(row["elapsed_ns"], 0, 3600 * 1_000_000_000))
        for key in ("request_sha256", "response_sha256"):
            _need(row[key] is None or _digest(row[key]))
        has_response = row["response_sha256"] is not None
        _need(has_response == (row["response_bytes"] > 0)
              and (not has_response or row["request_sha256"] is not None)
              and (not row["delivered"] or has_response)
              and (row["status"] == "ok" or row[detail_field] == 0)
              and (row["status"] != "ok" or has_response)
              and (has_response or row["status"] == "invalid_request"))
        counts["accepted_requests"] += 1
        counts["response_bytes"] += row["response_bytes"]
        counts["prepared_pages"] += row["status"] == "ok"
        counts["prepared_" + count_noun] += row[detail_field]
        if row["delivered"]:
            counts["delivered_responses"] += 1
            counts["delivered_response_bytes"] += row["response_bytes"]
            counts["delivered_pages"] += row["status"] == "ok"
            counts["delivered_" + count_noun] += row[detail_field]
        if row["request_sha256"] is not None:
            counts["request_hash_count"] += 1
            request_hashes.add(row["request_sha256"])
        if has_response:
            counts["response_hash_count"] += 1
            response_hashes.add(row["response_sha256"])
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    _need(_integer(audit["response_bytes"], 0, descriptor["max_total_response_bytes"])
          and audit["response_bytes"] == counts["response_bytes"])
    return counts, statuses, request_hashes, response_hashes


def _session(raw, row, scope):
    audit = _parse(raw)
    _need(set(audit) == _AUDIT_FIELDS and audit["schema"] == "supervisor-numerical-obligation-session@1")
    _need(all(audit[key] is False for key in ("proof_authority", "execution_authority", "completion_authority"))
          and type(audit["source_obligations_discharged"]) is int and audit["source_obligations_discharged"] == 0
          and type(audit["retrieval_provider_calls"]) is int and audit["retrieval_provider_calls"] == 0)
    descriptor = protocol.validate_descriptor(audit["session"])
    observation = row["numerical_obligation_retrieval"]
    _need(type(observation) is dict and set(observation) == _OBSERVATION_FIELDS
          and observation["schema"] == "numerical-obligation-dispatch-observation@1"
          and observation["requests_observed"] is False
          and all(observation[key] is False for key in ("proof_authority", "execution_authority", "completion_authority")))
    _need(_digest(audit["descriptor_sha256"])
          and audit["descriptor_sha256"] == observation["descriptor_sha256"]
          == _sha(protocol.canonical_json(descriptor))
          and observation["session_id"] == descriptor["session_id"]
          and all(descriptor[key] == observation[key] for key in _JOIN_FIELDS)
          and descriptor["task_id"] == scope["task_alias"] and descriptor["task_cid"] == scope["task_cid"]
          and descriptor["workspace"] == row["workspace"]
          and descriptor["owner_uid"] == os.geteuid())
    boundary = row["external_container_boundary"]
    formalization = row["source_formalization_context"]
    _need(type(boundary) is dict and type(formalization) is dict
          and boundary.get("manifest_sha256") == descriptor["boundary_manifest_sha256"]
          and type(boundary.get("owner_uid")) is int and boundary["owner_uid"] == descriptor["owner_uid"]
          and type(boundary.get("worker_uid")) is int and boundary["worker_uid"] == descriptor["worker_uid"]
          and all(formalization.get(key) == descriptor[key] for key in
                  ("task_id", "task_cid", "context_bundle_sha256")))
    counts, statuses, requests, responses = _records(audit, descriptor)
    return dict(session_id_sha256=_sha(descriptor["session_id"].encode()),
        invocation_id_sha256=_sha(row["invocation_id"].encode()), audit_sha256=_sha(raw),
        descriptor_sha256=audit["descriptor_sha256"],
        task_cid_sha256=_sha(descriptor["task_cid"].encode()),
        context_bundle_sha256=descriptor["context_bundle_sha256"],
        context_receipt_sha256=descriptor["context_receipt_sha256"],
        source_tree_id_sha256=_sha(descriptor["source_tree_id"].encode()),
        workspace_sha256=_sha(descriptor["workspace"].encode()),
        boundary_manifest_sha256=descriptor["boundary_manifest_sha256"],
        counts=counts, status_counts=dict(sorted(statuses.items())),
        unique_request_hash_count=len(requests), unique_response_hash_count=len(responses)), requests, responses


def collect(*, parent: Path, scope: dict | None, report: dict) -> dict:
    """Read exact session filenames selected by coding receipts after cleanup.

    Missing, malformed, mismatched or incomplete selected evidence yields
    unavailable accounting, never a zero measurement or a partial aggregate.
    Expired descriptors are expected: only their closed shape and identities
    are checked here, without requiring the historical owner process to live.
    """
    try:
        _need(type(report) is dict)
        invocations = report.get("provider_invocations", [])
        _need(type(invocations) is list and len(invocations) <= 128
              and all(type(row) is dict for row in invocations))
        selected = [row for row in invocations if "numerical_obligation_retrieval" in row]
        if not selected:
            selection = report.get("numerical_formalization_selection")
            if (type(selection) is dict and selection.get("enabled") is True
                    and report.get("coding_dispatch_possible") is True):
                return _base(reason="selected_session_receipt_unavailable")
            return _base("not_selected", "no_numerical_router_observation")
        _need(len(selected) <= MAX_SESSIONS)
        if type(report.get("worker_cleanup_returncode")) is not int or report["worker_cleanup_returncode"] != 0:
            return _base(reason="worker_cleanup_unavailable")
        _need(type(scope) is dict and _text(scope.get("task_cid")) and _text(scope.get("task_alias")))
        owner = os.geteuid()
        _need(owner > 0 and owner == os.getuid())
        parent = Path(parent)
        _protected_directory(parent, owner=owner, group=owner)
        root_fd = _open_directory(parent)
        try:
            directory = os.fstat(root_fd)
            _need(_identity(directory) == _identity(parent.lstat()))
            sessions, ids, invocations_seen, request_hashes, response_hashes = [], set(), set(), set(), set()
            for row in selected:
                _need(row.get("schema") == "router-implementation-invocation@1"
                      and row.get("phase") == row.get("purpose") == "coding"
                      and _text(row.get("invocation_id")) and row["invocation_id"] not in invocations_seen)
                invocations_seen.add(row["invocation_id"])
                observation = row["numerical_obligation_retrieval"]
                _need(type(observation) is dict and type(observation.get("session_id")) is str
                      and re.fullmatch("[0-9a-f]{32}", observation["session_id"]) is not None
                      and observation["session_id"] not in ids)
                ids.add(observation["session_id"])
                name = "numerical-" + observation["session_id"] + "-audit.json"
                before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                _need(stat.S_IMODE(before.st_mode) == 0o400 and before.st_gid == directory.st_gid)
                raw = _read(root_fd, name, limit=MAX_AUDIT_BYTES, private=True)
                after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                _need(_identity(before) == _identity(after) and before.st_gid == after.st_gid)
                item, requests, responses = _session(raw, row, scope)
                sessions.append(item)
                request_hashes.update(requests); response_hashes.update(responses)
            _protected_directory(parent, owner=owner, group=owner)
            _need(_identity(directory) == _identity(parent.lstat()))
        finally:
            os.close(root_fd)
        counts = {key: sum(item["counts"][key] for item in sessions) for key in _COUNT_FIELDS}
        counts.update(sessions=len(sessions), unique_request_hash_count=len(request_hashes),
                      unique_response_hash_count=len(response_hashes))
        result = _base("observed", "matched_owner_session_accounting")
        result.update(sessions=sessions, counts=counts,
            response_byte_scope="canonical JSON response bodies prepared by owner; excludes four-byte frame prefixes")
        _need(len(protocol.canonical_json(result)) <= MAX_AUDIT_BYTES)
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError):
        return _base()
