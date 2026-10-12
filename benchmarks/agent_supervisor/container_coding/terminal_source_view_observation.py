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


def _base(status="unavailable", reason="evidence_unavailable"):
    return dict(schema=SCHEMA, status=status, reason=reason, sessions=[], counts=None,
        observation_only=True, proof_authority=False, execution_authority=False,
        completion_authority=False, settlement_authority=False, retry_authority=False,
        source_obligations_discharged=0, provider_calls=0, provider_tokens=None,
        total_token_savings_measured=False, raw_queries_exported=False, raw_sources_exported=False)


def _session(raw, row, scope):
    audit = _parse(raw)
    _need(set(audit) == _AUDIT_FIELDS and audit["schema"] == "supervisor-worker-source-session@1"
        and all(audit[key] is False for key in ("proof_authority", "execution_authority", "completion_authority")))
    for key in ("source_obligations_discharged", "retrieval_provider_calls"):
        _need(type(audit[key]) is int and audit[key] == 0)
    descriptor = protocol.validate_descriptor(audit["session"])
    observation = row["source_view_retrieval"]
    _need(type(observation) is dict and set(observation) == OBSERVATION_FIELDS
        and observation["schema"] == "worker-source-dispatch-observation@1"
        and all(observation[key] is False for key in ("requests_observed", "proof_authority",
                                                     "execution_authority", "completion_authority")))
    _need(audit["descriptor_sha256"] == observation["descriptor_sha256"]
        == _sha(protocol.canonical_json(descriptor))
        and descriptor["session_id"] == observation["session_id"]
        and all(descriptor[key] == observation[key] for key in (*protocol.BINDING_FIELDS, "boundary_manifest_sha256"))
        and descriptor["task_id"] == scope["task_alias"]
        and descriptor["workspace"] == row["workspace"] and descriptor["owner_uid"] == os.geteuid())
    semantic = row["semantic_translation"]
    _need(type(semantic) is dict and all(semantic.get(key) == descriptor[key] for key in (
        "task_id", "scope_cid", "semantic_root_cid", "semantic_sha256", "native_prompt_sha256")))
    boundary = row["external_container_boundary"]
    _need(type(boundary) is dict and boundary.get("manifest_sha256") == descriptor["boundary_manifest_sha256"]
        and all(type(boundary.get(key)) is int and boundary[key] == descriptor[key] for key in ("owner_uid", "worker_uid")))
    advice = protocol.guidance(descriptor["profile"]).encode()
    _need(observation["guidance_sha256"] == _sha(advice)
        and type(observation["guidance_bytes"]) is int and observation["guidance_bytes"] == len(advice))
    counts, statuses, requests, responses = _records(audit, descriptor,
        detail_field="returned_source_bytes", max_detail=protocol.MAX_FRAME_BYTES, count_noun="source_bytes")
    _need(all(item["returned_source_bytes"] <= item["response_bytes"] for item in audit["requests"]))
    return dict(session_id_sha256=_sha(descriptor["session_id"].encode()),
        invocation_id_sha256=_sha(row["invocation_id"].encode()), audit_sha256=_sha(raw),
        descriptor_sha256=audit["descriptor_sha256"], profile=descriptor["profile"],
        semantic_sha256=descriptor["semantic_sha256"], source_manifest_sha256=descriptor["source_manifest_sha256"],
        counts=counts, status_counts=dict(sorted(statuses.items())),
        unique_request_hash_count=len(requests), unique_response_hash_count=len(responses)), requests, responses


def collect(*, parent: Path, scope: dict | None, report: dict) -> dict:
    try:
        _need(type(report) is dict)
        rows = report.get("provider_invocations", [])
        _need(type(rows) is list and len(rows) <= 128 and all(type(row) is dict for row in rows))
        profile = report.get("source_view_profile", "disabled")
        _need(type(profile) is str and profile in ("disabled", *protocol.PROFILES))
        selected = [row for row in rows if "source_view_retrieval" in row]
        if not selected:
            if profile != "disabled":
                return _base(reason="selected_session_receipt_unavailable")
            return _base("not_selected", "no_source_view_router_observation")
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
                item, request_hashes, response_hashes = _session(raw, row, scope)
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
            source_byte_scope="all returned slices including repeated reads and representations")
        _need(len(protocol.canonical_json(result)) <= MAX_AUDIT_BYTES)
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError):
        return _base()
