"""Pure joins of optional retained lifecycle and session evidence.

A matching owner receipt reports release of its own service reservation. It
does not independently inspect scheduler state, establish reclaimed capacity,
release a borrowed caller reservation, or establish worker/model consumption.
Historical sessions without this separate artifact keep unknown disposition.
"""
from __future__ import annotations

import hashlib
import json
import re

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from . import terminal_repository_owner_audit as session_audit

SCHEMA = "terminal-repository-owner-lifecycle-session-observation@1"
AGGREGATE_SCHEMA = "terminal-repository-owner-lifecycle-observation@1"
RETIREMENT_SCHEMA = "repository-source-owner-retirement@1"
MAX_RETIREMENT_BYTES = protocol.MAX_FRAME_BYTES + 65536
_ENVELOPE = {"schema", "session_id", "descriptor_sha256", "session_audit_sha256",
             "session_audit_bytes", "lifecycle"}
_LIFECYCLE = {"schema", "state", "ready", "started", "worker_spawn_registered",
    "cleanup_complete", "cleanup_pending", "native_parent_acquired", "native_parent_released",
    "native_parent", "admission_uncertain", "observation_pending", "admission_request_id",
    "broker_cleanup_complete", "broker_audit", "renewal_unproven", "retained_custody", "error",
    "cleanup_error", "borrowed_parent_present", "resource_scope", "worker_reaping_performed",
    "worker_cleanup_verified", "automatic_terminal_wiring", "cross_uid_deployment_qualified",
    "provider_calls", "token_savings_measured", "proof_authority", "execution_authority",
    "completion_authority"}
_DEMAND = {"cpu_slots": 1, "memory_mb": 512, "child_process_slots": 1}
_PARENT = {"lease_id", "owner_pid", "parent_lease_id", *_DEMAND}
_FALSE = ("ready", "cleanup_pending", "admission_uncertain", "observation_pending",
    "retained_custody", "worker_reaping_performed", "worker_cleanup_verified",
    "automatic_terminal_wiring", "cross_uid_deployment_qualified", "token_savings_measured",
    "proof_authority", "execution_authority", "completion_authority")


def _need(value):
    if not value:
        raise ValueError("invalid retained owner lifecycle evidence")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        _need(key not in result)
        result[key] = value
    return result


def _flags():
    return dict(independent_scheduler_capacity_verified=False, borrowed_caller_release=None,
        worker_reaping_verified=False, worker_cleanup_independently_verified=False,
        provider_ingestion_verified=False, token_savings_measured=False,
        proof_authority=False, execution_authority=False, completion_authority=False)


def unavailable(reason="no_retained_lifecycle_receipt"):
    """Unknown optional evidence; absence never implies an unreleased lease."""
    if reason not in {"no_retained_lifecycle_receipt", "invalid_retained_lifecycle_receipt"}:
        reason = "invalid_retained_lifecycle_receipt"
    return dict(schema=SCHEMA, status="unavailable", reason=reason,
        retirement_sha256=None, retirement_bytes=None, session_audit_sha256=None,
        descriptor_sha256=None, service_reservation_sha256=None, session_child_sha256=None,
        own_service_reservation_release_reported=None, borrowed_parent_present=None,
        renewal_unproven=None, lifecycle_error_reported=None,
        scope="retained_owner_report_of_own_repository_source_service_reservation_only",
        **_flags())


def aggregate(sessions):
    """Require the whole already-joined invocation population for a total claim."""
    known = sum(item["status"] == "joined_complete" for item in sessions)
    complete = bool(sessions) and known == len(sessions)
    return dict(schema=AGGREGATE_SCHEMA, status="observed" if complete else "unavailable",
        reason=None if complete else "incomplete_lifecycle_evidence", sessions=sessions,
        session_count=len(sessions), known_session_count=known,
        unknown_session_count=len(sessions) - known,
        own_service_reservation_release_reported=True if complete else None, **_flags())


def _summarize(*, retirement_raw, audit_raw):
    _need(type(retirement_raw) is bytes and 0 < len(retirement_raw) <= MAX_RETIREMENT_BYTES
          and type(audit_raw) is bytes and 0 < len(audit_raw) <= protocol.MAX_FRAME_BYTES)
    # The envelope may be larger than the protocol frame limit because it wraps
    # one bounded session audit. Reuse canonical tree validation after parsing.
    value = json.loads(retirement_raw.decode("utf-8"), object_pairs_hook=_pairs)
    _need(protocol.canonical_json(value) == retirement_raw)
    audit = protocol.strict_json_loads(audit_raw)
    _need(type(value) is dict and set(value) == _ENVELOPE
          and value["schema"] == RETIREMENT_SCHEMA
          and type(audit) is dict and set(audit) == session_audit._AUDIT_FIELDS
          and audit["schema"] in {"repository-source-owner-session@1", "repository-source-owner-session@2"})
    descriptor = protocol.validate_descriptor(audit["session"])
    _need(value["session_id"] == descriptor["session_id"]
          and value["descriptor_sha256"] == audit["descriptor_sha256"]
              == _sha(protocol.canonical_json(descriptor))
          and value["session_audit_sha256"] == _sha(audit_raw)
          and type(value["session_audit_bytes"]) is int
          and value["session_audit_bytes"] == len(audit_raw))
    receipt = value["lifecycle"]
    _need(type(receipt) is dict and set(receipt) == _LIFECYCLE
          and receipt["schema"] == "repository-source-owner-lifecycle@1"
          and receipt["state"] == "closed"
          and all(receipt[key] is True for key in ("started", "cleanup_complete",
              "native_parent_acquired", "native_parent_released", "broker_cleanup_complete"))
          and all(receipt[key] is False for key in _FALSE)
          and all(type(receipt[key]) is bool for key in ("worker_spawn_registered",
              "borrowed_parent_present", "renewal_unproven"))
          and receipt["cleanup_error"] is None
          and (receipt["error"] is None or type(receipt["error"]) is str
              and re.fullmatch("[a-z][a-z0-9_]{0,127}", receipt["error"]) is not None)
          and type(receipt["admission_request_id"]) is str
          and re.fullmatch("repository-source-lifecycle:[0-9a-f]{32}", receipt["admission_request_id"]) is not None
          and receipt["resource_scope"] == "repository_source_service_only"
          and type(receipt["provider_calls"]) is int and receipt["provider_calls"] == 0
          and protocol.canonical_json(receipt["broker_audit"]) == audit_raw)
    _need(all(audit[key] is True for key in ("service_stopped", "cleanup_complete",
              "accounting_complete", "native_child_acquired", "native_child_released",
              "thread_launch_attempted", "canonical_source_read_only"))
          and audit["cleanup_pending"] is False
          and audit["native_child_acquisition_uncertain"] is False
          and audit["cleanup_error"] is None and audit["join_error"] is None
          and all(audit[key] is False for key in session_audit._FALSE_FIELDS)
          and type(audit["provider_calls"]) is int and audit["provider_calls"] == 0)
    session_audit._child(audit["native_child"], descriptor)
    session_audit._transport(audit, descriptor)
    parent, child = receipt["native_parent"], audit["native_child"]
    _need(type(parent) is dict and set(parent) == _PARENT
          and session_audit._text(parent["lease_id"])
          and parent["lease_id"] == child["parent_lease_id"] != child["lease_id"]
          and type(parent["owner_pid"]) is int and parent["owner_pid"] == descriptor["owner_pid"]
          and all(type(parent[key]) is int and parent[key] == amount for key, amount in _DEMAND.items())
          and (parent["parent_lease_id"] is None or session_audit._text(parent["parent_lease_id"])
              and parent["parent_lease_id"] not in {parent["lease_id"], child["lease_id"]})
          and receipt["borrowed_parent_present"] is (parent["parent_lease_id"] is not None))
    result = unavailable()
    result.update(status="joined_complete", reason=None, retirement_sha256=_sha(retirement_raw),
        retirement_bytes=len(retirement_raw), session_audit_sha256=_sha(audit_raw),
        descriptor_sha256=value["descriptor_sha256"],
        service_reservation_sha256=_sha(parent["lease_id"].encode("utf-8")),
        session_child_sha256=_sha(child["lease_id"].encode("utf-8")),
        own_service_reservation_release_reported=True,
        borrowed_parent_present=receipt["borrowed_parent_present"],
        renewal_unproven=receipt["renewal_unproven"], lifecycle_error_reported=receipt["error"] is not None)
    return result


def summarize_owner_lifecycle(*, retirement_raw: bytes, audit_raw: bytes) -> dict:
    """Validate copied lifecycle/transport bytes without opening live state.

    Call after the phase12 session summarizer has validated the complete
    session and operation evidence. This helper adds release evidence and
    basic transport/authority consistency, but cannot recount prepared
    operations without their independently retained JSONL bytes.
    """
    try:
        return _summarize(retirement_raw=retirement_raw, audit_raw=audit_raw)
    except (KeyError, TypeError, UnicodeError, OverflowError, RecursionError) as error:
        raise ValueError("invalid retained owner lifecycle evidence") from error
