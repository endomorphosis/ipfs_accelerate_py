"""Durable, deduplicating merge queue for implementation lanes.

The queue is deliberately process safe. Producers may be independent daemon
processes, but only one consumer can atomically claim a request. DuckDB is the
authoritative index and small JSON files are retained as human-readable stage
receipts.  A request is idempotent when both its canonical task identity and
source commit match an existing request, including a completed or quarantined
request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from ..proof.formal_verification_contracts import content_identity
from ..task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBRow,
    initialize_duckdb_database,
    open_duckdb_connection,
)

_PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
_ACTIVE_STATES = ("pending", "processing")
_COMMIT_METADATA_KEYS = (
    "commit_sha",
    "source_commit",
    "implementation_commit",
    "candidate_commit",
    "head_sha",
    "commit",
)
_CANONICAL_METADATA_KEYS = (
    "canonical_task_key",
    "canonical_task_id",
    "canonical_task_cid",
    "task_cid",
)
MERGE_QUEUE_THROUGHPUT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/merge-queue-throughput@1"
)
MERGE_TARGET_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
)
SUBMODULE_INTEGRATION_RECOVERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/submodule-integration-recovery@1"
)
LEGACY_POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-review-denial-tombstone@2"
)
POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-review-denial-tombstone@3"
)
POST_MERGE_REVIEW_DENIAL_CONSUMPTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-review-denial-consumption@1"
)
POST_MERGE_CORRECTION_CHAIN_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-correction-chain-record@1"
)
POST_MERGE_CORRECTION_CHAIN_HEAD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-correction-chain-head@1"
)
POST_MERGE_CORRECTION_AUTHORITY_STATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-correction-authority-state@1"
)
POST_MERGE_CORRECTION_CONSUMPTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-correction-authority-consumption@1"
)
POST_MERGE_CORRECTION_FAILURE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-correction-failure@1"
)
POST_MERGE_CORRECTION_REPAIR_GRANT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-correction-repair-grant@1"
)
POST_MERGE_CORRECTION_LEGACY_FAILURE_ANCHOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-correction-legacy-failure-anchor@1"
)
POST_MERGE_CORRECTION_LEGACY_HIGH_WATER_ANCHOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-correction-legacy-high-water-anchor@2"
)
_POST_MERGE_CORRECTION_REGISTRY_MIGRATION_KEY = (
    "post_merge_correction_registry:migrated"
)
_POST_MERGE_CORRECTION_REGISTRY_MIGRATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-correction-registry-migration@1"
)
_FULL_GIT_OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_GIT_TREE_ID = re.compile(r"^git-tree:[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256_EVENT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_DENIAL_RECORD_BYTES = 16 * 1024
_MAX_CORRECTION_CHAIN_RECORD_BYTES = 16 * 1024
_POST_MERGE_CORRECTION_COMMON_FIELDS = frozenset(
    {
        "schema",
        "denial_id",
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "attempt",
        "origin_stream_id",
    }
)
_POST_MERGE_CORRECTION_TRANSITION_FIELDS = {
    "denial_consumed": frozenset(
        {
            "authority_kind",
            "authority_id",
            "started_event_id",
            "started_event_sequence",
        }
    ),
    "correction_failed": frozenset(
        {
            "authority_kind",
            "authority_id",
            "terminal_event_id",
            "terminal_event_sequence",
            "failure_kind",
        }
    ),
    "repair_granted": frozenset(
        {
            "grant_id",
            "grant_event_id",
            "grant_event_sequence",
            "failure_record_id",
            "failure_event_id",
            "failure_event_sequence",
            "failure_kind",
            "repair_task_id",
            "repair_task_binding_id",
            "repair_binding_id",
            "recovery_seed_ref",
            "recovery_seed_tree_id",
            "recovery_seed_submodule_path",
            "recovery_seed_submodule_commit",
        }
    ),
    "grant_consumed": frozenset(
        {
            "authority_kind",
            "authority_id",
            "started_event_id",
            "started_event_sequence",
        }
    ),
    "legacy_failure_anchored": frozenset(
        {
            "authority_kind",
            "authority_id",
            "correction_attempt",
            "correction_started_event_id",
            "correction_started_event_sequence",
            "correction_terminal_event_id",
            "correction_terminal_event_sequence",
            "superseding_started_event_id",
            "superseding_started_event_sequence",
            "terminal_event_id",
            "terminal_event_sequence",
            "failure_kind",
            "migration_reason",
            "recovery_seed_ref",
            "recovery_seed_tree_id",
            "recovery_seed_submodule_path",
            "recovery_seed_submodule_commit",
        }
    ),
    "legacy_high_water_anchored": frozenset(
        {
            "authority_kind",
            "authority_id",
            "first_correction_attempt",
            "attempt_events",
            "terminal_event_id",
            "terminal_event_sequence",
            "failure_kind",
            "migration_reason",
            "recovery_seed_ref",
            "recovery_seed_tree_id",
            "recovery_seed_submodule_path",
            "recovery_seed_submodule_commit",
        }
    ),
}
_POST_MERGE_CORRECTION_SCHEMA_KIND = {
    POST_MERGE_CORRECTION_CONSUMPTION_SCHEMA: "consumption",
    POST_MERGE_CORRECTION_FAILURE_SCHEMA: "correction_failed",
    POST_MERGE_CORRECTION_REPAIR_GRANT_SCHEMA: "repair_granted",
    POST_MERGE_CORRECTION_LEGACY_FAILURE_ANCHOR_SCHEMA: (
        "legacy_failure_anchored"
    ),
    POST_MERGE_CORRECTION_LEGACY_HIGH_WATER_ANCHOR_SCHEMA: (
        "legacy_high_water_anchored"
    ),
}


class MergeQueueFullError(RuntimeError):
    """Raised when accepting another active request would exceed queue capacity."""


class MergeQueueFenceError(RuntimeError):
    """Raised when stale or non-owning work tries to mutate a claimed request."""


class MergeQueueIntegrityError(RuntimeError):
    """Raised when permanent queue authority is malformed or conflicting."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise MergeQueueIntegrityError(
            "post-merge denial record is not canonical JSON"
        ) from exc
    if len(encoded.encode("utf-8")) > _MAX_DENIAL_RECORD_BYTES:
        raise MergeQueueIntegrityError(
            "post-merge denial record exceeds its persistence bound"
        )
    return encoded


def _post_merge_review_terminal_key_material(
    record: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "target_repository_id": str(
            record.get("target_repository_id") or ""
        ),
        "target_branch": str(record.get("target_branch") or ""),
        "task_id": str(record.get("task_id") or ""),
        "canonical_task_key": str(
            record.get("canonical_task_key") or ""
        ),
        "canonical_task_cid": str(
            record.get("canonical_task_cid") or ""
        ),
        "task_binding_id": str(record.get("task_binding_id") or ""),
        "implementation_commit": str(
            record.get("implementation_commit") or ""
        ),
    }


def _validated_post_merge_review_denial(
    value: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
) -> tuple[dict[str, Any], str]:
    """Return one exact, content-addressed terminal denial record."""

    if not isinstance(value, Mapping):
        raise MergeQueueIntegrityError(
            "post-merge denial record must be an object"
        )
    record = dict(value)
    expected_fields = {
        "schema",
        "terminal_key_id",
        "denial_id",
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "review_attempt",
        "implementation_attempt",
        "target_implementation_attempt",
        "implementation_commit",
        "merge_commit",
        "repository_tree_id",
        "review_receipt_id",
        "review_request_id",
        "review_response_id",
        "diff_binding_id",
        "implementer_provenance_id",
        "correction_origin_stream_id",
        "source_event_id",
        "source_event_sequence",
        "correction_authorized",
        "decision",
        "source_finding_count",
        "included_finding_count",
        "truncated",
        "findings",
        "repository_write_authorized",
        "proof_authoritative",
        "completion_authoritative",
    }
    legacy = (
        record.get("schema")
        == LEGACY_POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA
    )
    if legacy:
        expected_fields -= {
            "source_event_id",
            "source_event_sequence",
        }
    if set(record) != expected_fields:
        raise MergeQueueIntegrityError(
            "post-merge denial record schema fields changed"
        )
    required_text = (
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "review_receipt_id",
        "review_request_id",
        "review_response_id",
        "diff_binding_id",
        "implementer_provenance_id",
        "correction_origin_stream_id",
    )
    if (
        (
            record.get("schema")
            != POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA
            and not (allow_legacy and legacy)
        )
        or record.get("decision") != "changes_required"
        or any(not isinstance(record.get(name), str) or not record[name]
               for name in required_text)
        or not _FULL_GIT_OBJECT_ID.fullmatch(
            str(record.get("implementation_commit") or "")
        )
        or not _FULL_GIT_OBJECT_ID.fullmatch(
            str(record.get("merge_commit") or "")
        )
        or not _GIT_TREE_ID.fullmatch(
            str(record.get("repository_tree_id") or "")
        )
        or record.get("repository_write_authorized") is not False
        or record.get("proof_authoritative") is not False
        or record.get("completion_authoritative") is not False
        or not isinstance(record.get("correction_authorized"), bool)
        or not isinstance(record.get("truncated"), bool)
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial record binding is invalid"
        )
    if not legacy:
        source_event_id = record.get("source_event_id")
        source_event_sequence = record.get("source_event_sequence")
        has_source_event = bool(source_event_id)
        if (
            not isinstance(source_event_id, str)
            or isinstance(source_event_sequence, bool)
            or not isinstance(source_event_sequence, int)
            or source_event_sequence < 0
            or has_source_event
            != bool(source_event_sequence)
            or (
                has_source_event
                and _SHA256_EVENT_ID.fullmatch(source_event_id) is None
            )
            or (
                record.get("correction_authorized") is True
                and not has_source_event
            )
        ):
            raise MergeQueueIntegrityError(
                "post-merge denial source event binding is invalid"
            )
    integer_fields = (
        "review_attempt",
        "implementation_attempt",
        "target_implementation_attempt",
        "source_finding_count",
        "included_finding_count",
    )
    for name in integer_fields:
        item = record.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise MergeQueueIntegrityError(
                f"post-merge denial {name} must be a positive integer"
            )
    if (
        record["target_implementation_attempt"]
        != record["implementation_attempt"] + 1
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial target attempt is not exact-next"
        )
    findings = record.get("findings")
    if (
        not isinstance(findings, list)
        or not 1 <= len(findings) <= 4
        or record["included_finding_count"] != len(findings)
        or record["source_finding_count"] < len(findings)
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial finding projection is invalid"
        )
    for finding in findings:
        if (
            not isinstance(finding, Mapping)
            or set(finding)
            != {
                "finding_id",
                "source_ordinal",
                "code",
                "severity",
                "summary",
            }
        ):
            raise MergeQueueIntegrityError(
                "post-merge denial finding schema is invalid"
            )
        material = dict(finding)
        finding_id = str(material.pop("finding_id", "") or "")
        if (
            finding_id != content_identity(material)
            or isinstance(finding.get("source_ordinal"), bool)
            or not isinstance(finding.get("source_ordinal"), int)
            or int(finding["source_ordinal"]) < 1
            or finding.get("severity")
            not in {"blocker", "high", "medium", "low", "info"}
            or not isinstance(finding.get("code"), str)
            or not finding["code"]
            or not isinstance(finding.get("summary"), str)
            or not finding["summary"]
        ):
            raise MergeQueueIntegrityError(
                "post-merge denial finding identity is invalid"
            )
    terminal_key_id = str(record.get("terminal_key_id") or "")
    if terminal_key_id != content_identity(
        _post_merge_review_terminal_key_material(record)
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial terminal key identity is invalid"
        )
    denial_material = dict(record)
    denial_id = str(denial_material.pop("denial_id", "") or "")
    if denial_id != content_identity(denial_material):
        raise MergeQueueIntegrityError(
            "post-merge denial content identity is invalid"
        )
    return record, _canonical_json(record)


def _validated_post_merge_review_denial_consumption(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return one exact, content-addressed correction-consumption marker."""

    if not isinstance(value, Mapping):
        raise MergeQueueIntegrityError(
            "post-merge denial consumption must be an object"
        )
    record = dict(value)
    expected_fields = {
        "schema",
        "consumption_id",
        "terminal_key_id",
        "denial_id",
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "implementation_commit",
        "implementation_attempt",
        "target_implementation_attempt",
        "correction_origin_stream_id",
        "consuming_event_type",
        "consuming_event_id",
        "consuming_event_sequence",
        "consuming_implementation_attempt",
        "attempt_consumed",
        "repository_write_authorized",
        "proof_authoritative",
        "completion_authoritative",
    }
    if set(record) != expected_fields:
        raise MergeQueueIntegrityError(
            "post-merge denial consumption schema fields changed"
        )
    required_text = (
        "terminal_key_id",
        "denial_id",
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "correction_origin_stream_id",
        "consuming_event_type",
    )
    if (
        record.get("schema")
        != POST_MERGE_REVIEW_DENIAL_CONSUMPTION_SCHEMA
        or any(
            not isinstance(record.get(name), str) or not record[name]
            for name in required_text
        )
        or not _FULL_GIT_OBJECT_ID.fullmatch(
            str(record.get("implementation_commit") or "")
        )
        or not _SHA256_EVENT_ID.fullmatch(
            str(record.get("consuming_event_id") or "")
        )
        or record.get("consuming_event_type")
        not in {
            "implementation_finished",
            "implementation_state_recovered",
        }
        or record.get("attempt_consumed") is not True
        or record.get("repository_write_authorized") is not False
        or record.get("proof_authoritative") is not False
        or record.get("completion_authoritative") is not False
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial consumption binding is invalid"
        )
    integer_fields = (
        "implementation_attempt",
        "target_implementation_attempt",
        "consuming_event_sequence",
        "consuming_implementation_attempt",
    )
    for name in integer_fields:
        item = record.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise MergeQueueIntegrityError(
                f"post-merge denial consumption {name} must be positive"
            )
    if (
        record["target_implementation_attempt"]
        != record["implementation_attempt"] + 1
        or record["consuming_implementation_attempt"]
        != record["target_implementation_attempt"]
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial consumption is not causally later"
        )
    if str(record["terminal_key_id"]) != content_identity(
        _post_merge_review_terminal_key_material(record)
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial consumption terminal identity is invalid"
        )
    material = dict(record)
    consumption_id = str(material.pop("consumption_id", "") or "")
    if consumption_id != content_identity(material):
        raise MergeQueueIntegrityError(
            "post-merge denial consumption identity is invalid"
        )
    return record, _canonical_json(record)


def _canonical_correction_chain_json(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise MergeQueueIntegrityError(
            "post-merge correction chain record is not canonical JSON"
        ) from exc
    if (
        len(encoded.encode("utf-8"))
        > _MAX_CORRECTION_CHAIN_RECORD_BYTES
    ):
        raise MergeQueueIntegrityError(
            "post-merge correction chain record exceeds its persistence bound"
        )
    return encoded


def _validated_post_merge_recovery_seed(
    value: Mapping[str, Any],
    *,
    required: bool,
    allow_unresolved_root: bool = False,
) -> dict[str, str]:
    seed = {
        "recovery_seed_ref": str(
            value.get("recovery_seed_ref") or ""
        ),
        "recovery_seed_tree_id": str(
            value.get("recovery_seed_tree_id") or ""
        ),
        "recovery_seed_submodule_path": str(
            value.get("recovery_seed_submodule_path") or ""
        ),
        "recovery_seed_submodule_commit": str(
            value.get("recovery_seed_submodule_commit") or ""
        ),
    }
    if any(not isinstance(value.get(name), str) for name in seed):
        raise MergeQueueIntegrityError(
            "post-merge recovery seed binding is invalid"
        )
    if not any(seed.values()):
        if required:
            raise MergeQueueIntegrityError(
                "post-merge legacy recovery seed is required"
            )
        return seed
    submodule_path = seed["recovery_seed_submodule_path"]
    root_seed_is_absent = (
        not seed["recovery_seed_ref"]
        and not seed["recovery_seed_tree_id"]
    )
    if (
        not _FULL_GIT_OBJECT_ID.fullmatch(
            seed["recovery_seed_submodule_commit"]
        )
        or not submodule_path
        or submodule_path != submodule_path.strip()
        or submodule_path.startswith("/")
        or "\\" in submodule_path
        or any(
            component in {"", ".", ".."}
            for component in submodule_path.split("/")
        )
        or (
            root_seed_is_absent
            and not allow_unresolved_root
        )
        or (
            not root_seed_is_absent
            and (
                not _FULL_GIT_OBJECT_ID.fullmatch(
                    seed["recovery_seed_ref"]
                )
                or not _GIT_TREE_ID.fullmatch(
                    seed["recovery_seed_tree_id"]
                )
            )
        )
    ):
        raise MergeQueueIntegrityError(
            "post-merge recovery seed binding is invalid"
        )
    return seed


def _validated_legacy_high_water_attempt_events(
    value: Any,
    *,
    first_attempt: int,
    high_water_attempt: int,
) -> tuple[dict[str, Any], ...]:
    """Validate one bounded, contiguous legacy attempt history."""

    if (
        not isinstance(value, list)
        or not value
        or len(value) > 32
        or first_attempt < 1
        or high_water_attempt < first_attempt
        or len(value) != high_water_attempt - first_attempt + 1
    ):
        raise MergeQueueIntegrityError(
            "post-merge legacy high-water attempt history is invalid"
        )
    expected_fields = {
        "attempt",
        "started_event_id",
        "started_event_sequence",
        "terminal_event_id",
        "terminal_event_sequence",
        "terminal_event_type",
    }
    allowed_terminal_types = {
        "implementation_finished",
        "implementation_state_recovered",
        "post_merge_correction_queue_reconciled",
    }
    verified: list[dict[str, Any]] = []
    previous_sequence = 0
    seen_event_ids: set[str] = set()
    for offset, raw_entry in enumerate(value):
        if not isinstance(raw_entry, Mapping):
            raise MergeQueueIntegrityError(
                "post-merge legacy high-water attempt entry is invalid"
            )
        entry = dict(raw_entry)
        attempt = entry.get("attempt")
        started_sequence = entry.get("started_event_sequence")
        terminal_sequence = entry.get("terminal_event_sequence")
        started_event_id = entry.get("started_event_id")
        terminal_event_id = entry.get("terminal_event_id")
        terminal_event_type = entry.get("terminal_event_type")
        if (
            set(entry) != expected_fields
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt != first_attempt + offset
            or isinstance(started_sequence, bool)
            or not isinstance(started_sequence, int)
            or isinstance(terminal_sequence, bool)
            or not isinstance(terminal_sequence, int)
            or started_sequence <= previous_sequence
            or terminal_sequence <= started_sequence
            or not isinstance(started_event_id, str)
            or not started_event_id
            or not isinstance(terminal_event_id, str)
            or not terminal_event_id
            or started_event_id == terminal_event_id
            or started_event_id in seen_event_ids
            or terminal_event_id in seen_event_ids
            or terminal_event_type not in allowed_terminal_types
        ):
            raise MergeQueueIntegrityError(
                "post-merge legacy high-water attempt entry is invalid"
            )
        seen_event_ids.update((started_event_id, terminal_event_id))
        previous_sequence = terminal_sequence
        verified.append(entry)
    return tuple(verified)


def _validated_post_merge_correction_transition(
    value: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Validate one semantic transition before assigning chain position."""

    if not isinstance(value, Mapping):
        raise MergeQueueIntegrityError(
            "post-merge correction transition must be an object"
        )
    material = dict(value)
    schema = str(material.get("schema") or "")
    schema_kind = _POST_MERGE_CORRECTION_SCHEMA_KIND.get(schema)
    if schema_kind is None:
        raise MergeQueueIntegrityError(
            "post-merge correction transition schema is invalid"
        )
    authority_kind = str(material.get("authority_kind") or "")
    if schema_kind == "consumption":
        if authority_kind == "review_denial":
            record_kind = "denial_consumed"
        elif authority_kind == "repair_grant":
            record_kind = "grant_consumed"
        else:
            raise MergeQueueIntegrityError(
                "post-merge correction consumption authority kind is invalid"
            )
    else:
        record_kind = schema_kind
    expected_fields = (
        _POST_MERGE_CORRECTION_COMMON_FIELDS
        | _POST_MERGE_CORRECTION_TRANSITION_FIELDS[record_kind]
    )
    if set(material) != expected_fields:
        raise MergeQueueIntegrityError(
            "post-merge correction transition schema fields changed"
        )
    required_text = (
        "denial_id",
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "origin_stream_id",
    )
    if any(
        not isinstance(material.get(name), str) or not material[name]
        for name in required_text
    ):
        raise MergeQueueIntegrityError(
            "post-merge correction transition identity is incomplete"
        )
    attempt = material.get("attempt")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
    ):
        raise MergeQueueIntegrityError(
            "post-merge correction transition attempt must be positive"
        )
    if record_kind in {"denial_consumed", "grant_consumed"}:
        sequence = material.get("started_event_sequence")
        text_fields = ("authority_id", "started_event_id")
        expected_authority_kind = (
            "review_denial"
            if record_kind == "denial_consumed"
            else "repair_grant"
        )
        if authority_kind != expected_authority_kind:
            raise MergeQueueIntegrityError(
                "post-merge correction consumption authority changed"
            )
    elif record_kind == "correction_failed":
        sequence = material.get("terminal_event_sequence")
        text_fields = (
            "authority_kind",
            "authority_id",
            "terminal_event_id",
            "failure_kind",
        )
        if authority_kind not in {"review_denial", "repair_grant"}:
            raise MergeQueueIntegrityError(
                "post-merge correction failure authority kind is invalid"
            )
        if material.get("failure_kind") not in {
            "implementation",
            "validation",
            "merge",
        }:
            raise MergeQueueIntegrityError(
                "post-merge correction failure kind is invalid"
            )
    elif record_kind == "repair_granted":
        sequence = material.get("grant_event_sequence")
        text_fields = (
            "grant_id",
            "grant_event_id",
            "failure_record_id",
            "failure_event_id",
            "failure_kind",
            "repair_task_id",
            "repair_task_binding_id",
            "repair_binding_id",
        )
        failure_sequence = material.get("failure_event_sequence")
        if (
            isinstance(failure_sequence, bool)
            or not isinstance(failure_sequence, int)
            or failure_sequence < 1
            or material.get("failure_kind")
            not in {"implementation", "validation", "merge"}
        ):
            raise MergeQueueIntegrityError(
                "post-merge correction repair grant failure binding is invalid"
            )
        _validated_post_merge_recovery_seed(
            material,
            required=False,
        )
    elif record_kind == "legacy_failure_anchored":
        sequence = material.get("terminal_event_sequence")
        text_fields = (
            "authority_kind",
            "authority_id",
            "correction_started_event_id",
            "correction_terminal_event_id",
            "superseding_started_event_id",
            "terminal_event_id",
            "failure_kind",
            "migration_reason",
        )
        correction_attempt = material.get("correction_attempt")
        ordered_sequences = (
            material.get("correction_started_event_sequence"),
            material.get("correction_terminal_event_sequence"),
            material.get("superseding_started_event_sequence"),
            material.get("terminal_event_sequence"),
        )
        if (
            authority_kind != "review_denial"
            or isinstance(correction_attempt, bool)
            or not isinstance(correction_attempt, int)
            or correction_attempt < 1
            or material["attempt"] != correction_attempt + 1
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 1
                for item in ordered_sequences
            )
            or tuple(sorted(ordered_sequences)) != ordered_sequences
            or len(set(ordered_sequences)) != len(ordered_sequences)
            or material.get("failure_kind")
            not in {"implementation", "validation", "merge"}
            or material.get("migration_reason")
            != "legacy_untyped_retry_lineage"
        ):
            raise MergeQueueIntegrityError(
                "post-merge correction legacy failure anchor is invalid"
            )
        _validated_post_merge_recovery_seed(
            material,
            required=True,
            allow_unresolved_root=True,
        )
    elif record_kind == "legacy_high_water_anchored":
        sequence = material.get("terminal_event_sequence")
        text_fields = (
            "authority_kind",
            "authority_id",
            "terminal_event_id",
            "failure_kind",
            "migration_reason",
        )
        first_attempt = material.get("first_correction_attempt")
        if (
            authority_kind != "review_denial"
            or isinstance(first_attempt, bool)
            or not isinstance(first_attempt, int)
            or first_attempt < 1
            or first_attempt > attempt
            or material.get("failure_kind")
            not in {"implementation", "validation", "merge"}
            or material.get("migration_reason")
            != "legacy_untyped_retry_high_water"
        ):
            raise MergeQueueIntegrityError(
                "post-merge correction legacy high-water anchor is invalid"
            )
        attempt_events = _validated_legacy_high_water_attempt_events(
            material.get("attempt_events"),
            first_attempt=first_attempt,
            high_water_attempt=attempt,
        )
        last_attempt = attempt_events[-1]
        if (
            material.get("terminal_event_id")
            != last_attempt["terminal_event_id"]
            or material.get("terminal_event_sequence")
            != last_attempt["terminal_event_sequence"]
        ):
            raise MergeQueueIntegrityError(
                "post-merge correction legacy high-water terminal changed"
            )
        _validated_post_merge_recovery_seed(
            material,
            required=True,
            allow_unresolved_root=True,
        )
    else:  # pragma: no cover - guarded by the schema map above.
        raise MergeQueueIntegrityError(
            "post-merge correction transition kind is invalid"
        )
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
        or any(
            not isinstance(material.get(name), str)
            or not material[name]
            for name in text_fields
        )
    ):
        raise MergeQueueIntegrityError(
            "post-merge correction transition event binding is invalid"
        )
    _canonical_correction_chain_json(material)
    return record_kind, material


def _post_merge_correction_chain_record(
    transition: Mapping[str, Any],
    *,
    terminal_key_id: str,
    ordinal: int,
    parent_record_id: str,
) -> dict[str, Any]:
    """Content-address one already validated semantic transition."""

    record_kind, material = (
        _validated_post_merge_correction_transition(transition)
    )
    common = {
        name: material[name]
        for name in _POST_MERGE_CORRECTION_COMMON_FIELDS
        if name != "schema"
    }
    detail_fields = _POST_MERGE_CORRECTION_TRANSITION_FIELDS[
        record_kind
    ]
    detail = {
        "schema": material["schema"],
        **{name: material[name] for name in detail_fields},
    }
    record_material: dict[str, Any] = {
        "schema": POST_MERGE_CORRECTION_CHAIN_RECORD_SCHEMA,
        "terminal_key_id": str(terminal_key_id or ""),
        "denial_id": str(material["denial_id"]),
        "ordinal": int(ordinal),
        "parent_record_id": str(parent_record_id or ""),
        "record_kind": record_kind,
        **common,
        "detail": detail,
    }
    return {
        **record_material,
        "record_id": content_identity(record_material),
    }


def _validated_post_merge_correction_chain_record(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return one exact content-addressed correction-chain record."""

    if not isinstance(value, Mapping):
        raise MergeQueueIntegrityError(
            "post-merge correction chain record must be an object"
        )
    record = dict(value)
    expected_fields = {
        "schema",
        "record_id",
        "terminal_key_id",
        "denial_id",
        "ordinal",
        "parent_record_id",
        "record_kind",
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "attempt",
        "origin_stream_id",
        "detail",
    }
    required_text = (
        "terminal_key_id",
        "denial_id",
        "parent_record_id",
        "record_kind",
        "target_repository_id",
        "target_branch",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "board_namespace",
        "task_binding_id",
        "origin_stream_id",
    )
    if (
        set(record) != expected_fields
        or record.get("schema")
        != POST_MERGE_CORRECTION_CHAIN_RECORD_SCHEMA
        or any(
            not isinstance(record.get(name), str) or not record[name]
            for name in required_text
        )
    ):
        raise MergeQueueIntegrityError(
            "post-merge correction chain record schema is invalid"
        )
    for name in ("ordinal", "attempt"):
        item = record.get(name)
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 1
        ):
            raise MergeQueueIntegrityError(
                f"post-merge correction chain {name} must be positive"
            )
    detail = record.get("detail")
    if not isinstance(detail, Mapping):
        raise MergeQueueIntegrityError(
            "post-merge correction chain detail must be an object"
        )
    transition = {
        "schema": detail.get("schema"),
        "denial_id": record["denial_id"],
        "target_repository_id": record["target_repository_id"],
        "target_branch": record["target_branch"],
        "task_id": record["task_id"],
        "canonical_task_key": record["canonical_task_key"],
        "canonical_task_cid": record["canonical_task_cid"],
        "board_namespace": record["board_namespace"],
        "task_binding_id": record["task_binding_id"],
        "attempt": record["attempt"],
        "origin_stream_id": record["origin_stream_id"],
        **{
            name: detail.get(name)
            for name in detail
            if name != "schema"
        },
    }
    record_kind, _material = (
        _validated_post_merge_correction_transition(transition)
    )
    if record_kind != record["record_kind"]:
        raise MergeQueueIntegrityError(
            "post-merge correction chain kind and detail disagree"
        )
    record_material = dict(record)
    record_id = str(record_material.pop("record_id", "") or "")
    if record_id != content_identity(record_material):
        raise MergeQueueIntegrityError(
            "post-merge correction chain content identity is invalid"
        )
    return record, _canonical_correction_chain_json(record)


def _post_merge_correction_chain_head(
    *,
    terminal_key_id: str,
    denial_id: str,
    head_record_id: str,
    head_ordinal: int,
) -> dict[str, Any]:
    material = {
        "schema": POST_MERGE_CORRECTION_CHAIN_HEAD_SCHEMA,
        "terminal_key_id": str(terminal_key_id or ""),
        "denial_id": str(denial_id or ""),
        "head_record_id": str(head_record_id or ""),
        "head_ordinal": int(head_ordinal),
    }
    return {
        **material,
        "head_state_id": content_identity(material),
    }


def _validated_post_merge_correction_chain_head(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MergeQueueIntegrityError(
            "post-merge correction chain head must be an object"
        )
    head = dict(value)
    if set(head) != {
        "schema",
        "terminal_key_id",
        "denial_id",
        "head_record_id",
        "head_ordinal",
        "head_state_id",
    }:
        raise MergeQueueIntegrityError(
            "post-merge correction chain head schema fields changed"
        )
    if (
        head.get("schema") != POST_MERGE_CORRECTION_CHAIN_HEAD_SCHEMA
        or any(
            not isinstance(head.get(name), str) or not head[name]
            for name in (
                "terminal_key_id",
                "denial_id",
                "head_record_id",
            )
        )
        or isinstance(head.get("head_ordinal"), bool)
        or not isinstance(head.get("head_ordinal"), int)
        or int(head["head_ordinal"]) < 0
    ):
        raise MergeQueueIntegrityError(
            "post-merge correction chain head binding is invalid"
        )
    material = dict(head)
    head_state_id = str(material.pop("head_state_id", "") or "")
    if head_state_id != content_identity(material):
        raise MergeQueueIntegrityError(
            "post-merge correction chain head identity is invalid"
        )
    return head


def _decoded_post_merge_review_denial_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify both the denial JSON and its independently indexed bindings."""

    try:
        decoded = json.loads(str(row["record_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MergeQueueIntegrityError(
            "post-merge denial registry contains malformed JSON"
        ) from exc
    record, canonical = _validated_post_merge_review_denial(
        decoded,
        allow_legacy=True,
    )
    row_bindings = {
        "terminal_key_id": str(row["terminal_key_id"]),
        "denial_id": str(row["denial_id"]),
        "target_repository_id": str(row["target_repository_id"]),
        "target_branch": str(row["target_branch"]),
        "task_id": str(row["task_id"]),
        "canonical_task_key": str(row["canonical_task_key"]),
        "canonical_task_cid": str(row["canonical_task_cid"]),
        "task_binding_id": str(row["task_binding_id"]),
        "implementation_commit": str(row["implementation_commit"]),
    }
    if (
        canonical != str(row["record_json"])
        or any(record[name] != item for name, item in row_bindings.items())
    ):
        raise MergeQueueIntegrityError(
            "post-merge denial registry row binding changed"
        )
    return record


def _decoded_post_merge_correction_chain_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one append-only chain row against its indexed columns."""

    try:
        decoded = json.loads(str(row["record_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MergeQueueIntegrityError(
            "post-merge correction registry contains malformed JSON"
        ) from exc
    record, canonical = _validated_post_merge_correction_chain_record(
        decoded
    )
    row_bindings: dict[str, Any] = {
        "record_id": str(row["record_id"]),
        "terminal_key_id": str(row["terminal_key_id"]),
        "denial_id": str(row["denial_id"]),
        "ordinal": int(row["ordinal"]),
        "parent_record_id": str(row["parent_record_id"]),
        "record_kind": str(row["record_kind"]),
        "target_repository_id": str(row["target_repository_id"]),
        "target_branch": str(row["target_branch"]),
        "task_id": str(row["task_id"]),
        "canonical_task_key": str(row["canonical_task_key"]),
        "canonical_task_cid": str(row["canonical_task_cid"]),
        "board_namespace": str(row["board_namespace"]),
        "task_binding_id": str(row["task_binding_id"]),
        "attempt": int(row["attempt"]),
        "origin_stream_id": str(row["origin_stream_id"]),
    }
    if (
        canonical != str(row["record_json"])
        or any(record[name] != item for name, item in row_bindings.items())
    ):
        raise MergeQueueIntegrityError(
            "post-merge correction registry row binding changed"
        )
    return record


def _decoded_post_merge_correction_head_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return _validated_post_merge_correction_chain_head(
        {
            "schema": POST_MERGE_CORRECTION_CHAIN_HEAD_SCHEMA,
            "terminal_key_id": str(row["terminal_key_id"]),
            "denial_id": str(row["denial_id"]),
            "head_record_id": str(row["head_record_id"]),
            "head_ordinal": int(row["head_ordinal"]),
            "head_state_id": str(row["head_state_id"]),
        }
    )


def _post_merge_correction_identity_from_denial(
    denial: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "target_repository_id": str(denial["target_repository_id"]),
        "target_branch": str(denial["target_branch"]),
        "task_id": str(denial["task_id"]),
        "canonical_task_key": str(denial["canonical_task_key"]),
        "canonical_task_cid": str(denial["canonical_task_cid"]),
        "board_namespace": str(denial["board_namespace"]),
        "task_binding_id": str(denial["task_binding_id"]),
        "origin_stream_id": str(
            denial["correction_origin_stream_id"]
        ),
    }


def _post_merge_correction_transition_from_chain_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    detail = dict(record["detail"])
    return {
        "schema": detail.pop("schema"),
        "denial_id": str(record["denial_id"]),
        "target_repository_id": str(record["target_repository_id"]),
        "target_branch": str(record["target_branch"]),
        "task_id": str(record["task_id"]),
        "canonical_task_key": str(record["canonical_task_key"]),
        "canonical_task_cid": str(record["canonical_task_cid"]),
        "board_namespace": str(record["board_namespace"]),
        "task_binding_id": str(record["task_binding_id"]),
        "attempt": int(record["attempt"]),
        "origin_stream_id": str(record["origin_stream_id"]),
        **detail,
    }


def _post_merge_correction_primary_events(
    record: Mapping[str, Any],
) -> tuple[tuple[str, int], ...]:
    detail = record["detail"]
    kind = str(record["record_kind"])
    if kind in {"denial_consumed", "grant_consumed"}:
        return (
            (
                str(detail["started_event_id"]),
                int(detail["started_event_sequence"]),
            ),
        )
    if kind == "correction_failed":
        return (
            (
                str(detail["terminal_event_id"]),
                int(detail["terminal_event_sequence"]),
            ),
        )
    if kind == "repair_granted":
        return (
            (
                str(detail["grant_event_id"]),
                int(detail["grant_event_sequence"]),
            ),
        )
    if kind == "legacy_high_water_anchored":
        attempt_events = _validated_legacy_high_water_attempt_events(
            detail["attempt_events"],
            first_attempt=int(detail["first_correction_attempt"]),
            high_water_attempt=int(record["attempt"]),
        )
        return tuple(
            event
            for attempt_event in attempt_events
            for event in (
                (
                    str(attempt_event["started_event_id"]),
                    int(attempt_event["started_event_sequence"]),
                ),
                (
                    str(attempt_event["terminal_event_id"]),
                    int(attempt_event["terminal_event_sequence"]),
                ),
            )
        )
    return (
        (
            str(detail["correction_started_event_id"]),
            int(detail["correction_started_event_sequence"]),
        ),
        (
            str(detail["correction_terminal_event_id"]),
            int(detail["correction_terminal_event_sequence"]),
        ),
        (
            str(detail["superseding_started_event_id"]),
            int(detail["superseding_started_event_sequence"]),
        ),
        (
            str(detail["terminal_event_id"]),
            int(detail["terminal_event_sequence"]),
        ),
    )


def _validate_post_merge_correction_chain(
    denial: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    head: Mapping[str, Any],
    *,
    seen_primary_events: set[tuple[str, str]],
    seen_primary_positions: set[tuple[str, int]],
    seen_grant_ids: set[str],
    seen_repair_bindings: set[str],
    seen_repair_task_bindings: set[str],
) -> None:
    """Verify a complete one-shot authority state machine and its high-water."""

    terminal_key_id = str(denial["terminal_key_id"])
    denial_id = str(denial["denial_id"])
    expected_identity = _post_merge_correction_identity_from_denial(
        denial
    )
    if (
        head["terminal_key_id"] != terminal_key_id
        or head["denial_id"] != denial_id
    ):
        raise MergeQueueIntegrityError(
            "post-merge correction head crosses denial identity"
        )
    if not records:
        if (
            int(head["head_ordinal"]) != 0
            or head["head_record_id"] != denial_id
        ):
            raise MergeQueueIntegrityError(
                "post-merge correction chain high-water was rolled back"
            )
        return

    parent_record_id = denial_id
    previous: Mapping[str, Any] | None = None
    previous_sequence = 0
    target_attempt = int(denial["target_implementation_attempt"])
    for ordinal, record in enumerate(records, start=1):
        if (
            int(record["ordinal"]) != ordinal
            or record["terminal_key_id"] != terminal_key_id
            or record["denial_id"] != denial_id
            or record["parent_record_id"] != parent_record_id
            or any(
                record[name] != expected
                for name, expected in expected_identity.items()
            )
        ):
            raise MergeQueueIntegrityError(
                "post-merge correction chain identity or order changed"
            )
        primary_events = _post_merge_correction_primary_events(record)
        if (
            primary_events[0][1] <= previous_sequence
            or any(
                sequence <= previous_sequence
                for _event_id, sequence in primary_events
            )
        ):
            raise MergeQueueIntegrityError(
                "post-merge correction event order is not monotonic"
            )
        origin_stream_id = str(record["origin_stream_id"])
        for event_id, sequence in primary_events:
            event_key = (origin_stream_id, event_id)
            event_position = (origin_stream_id, sequence)
            if (
                event_key in seen_primary_events
                or event_position in seen_primary_positions
            ):
                raise MergeQueueIntegrityError(
                    "post-merge correction event is reused"
                )
            seen_primary_events.add(event_key)
            seen_primary_positions.add(event_position)
            if sequence < 1:
                raise MergeQueueIntegrityError(
                    "post-merge correction event sequence is invalid"
                )
        detail = record["detail"]
        kind = str(record["record_kind"])
        attempt = int(record["attempt"])
        if previous is None:
            if (
                denial.get("correction_authorized") is not True
                or kind
                not in {
                    "denial_consumed",
                    "legacy_failure_anchored",
                    "legacy_high_water_anchored",
                }
            ):
                raise MergeQueueIntegrityError(
                    "post-merge correction root lacks one-shot authority"
                )
            if kind == "denial_consumed":
                if (
                    attempt != target_attempt
                    or detail["authority_kind"] != "review_denial"
                    or detail["authority_id"] != denial_id
                ):
                    raise MergeQueueIntegrityError(
                        "post-merge denial consumption binding changed"
                    )
            elif kind == "legacy_failure_anchored" and (
                int(detail["correction_attempt"]) != target_attempt
                or attempt != target_attempt + 1
                or detail["authority_kind"] != "review_denial"
                or detail["authority_id"] != denial_id
            ):
                raise MergeQueueIntegrityError(
                    "post-merge legacy anchor does not bind the denial"
                )
            elif kind == "legacy_high_water_anchored" and (
                int(detail["first_correction_attempt"])
                != target_attempt
                or attempt < target_attempt
                or detail["authority_kind"] != "review_denial"
                or detail["authority_id"] != denial_id
            ):
                raise MergeQueueIntegrityError(
                    "post-merge legacy high-water anchor does not bind the denial"
                )
        else:
            previous_kind = str(previous["record_kind"])
            previous_detail = previous["detail"]
            previous_attempt = int(previous["attempt"])
            allowed = {
                "denial_consumed": "correction_failed",
                "grant_consumed": "correction_failed",
                "correction_failed": "repair_granted",
                "legacy_failure_anchored": "repair_granted",
                "legacy_high_water_anchored": "repair_granted",
                "repair_granted": "grant_consumed",
            }
            if allowed.get(previous_kind) != kind:
                raise MergeQueueIntegrityError(
                    "post-merge correction state transition is invalid"
                )
            if kind == "correction_failed":
                if (
                    attempt != previous_attempt
                    or detail["authority_kind"]
                    != previous_detail["authority_kind"]
                    or detail["authority_id"]
                    != previous_detail["authority_id"]
                    or int(detail["terminal_event_sequence"])
                    <= int(previous_detail["started_event_sequence"])
                ):
                    raise MergeQueueIntegrityError(
                        "post-merge correction failure crosses consumption"
                    )
            elif kind == "repair_granted":
                expected_failure_event_id = str(
                    previous_detail["terminal_event_id"]
                )
                expected_failure_sequence = int(
                    previous_detail["terminal_event_sequence"]
                )
                recovery_seed = (
                    _validated_post_merge_recovery_seed(
                        detail,
                        required=(
                            previous_kind
                            in {
                                "legacy_failure_anchored",
                                "legacy_high_water_anchored",
                            }
                        ),
                    )
                )
                if previous_kind in {
                    "legacy_failure_anchored",
                    "legacy_high_water_anchored",
                }:
                    recovery_seed_matches = (
                        recovery_seed[
                            "recovery_seed_submodule_path"
                        ]
                        == previous_detail[
                            "recovery_seed_submodule_path"
                        ]
                        and recovery_seed[
                            "recovery_seed_submodule_commit"
                        ]
                        == previous_detail[
                            "recovery_seed_submodule_commit"
                        ]
                        and (
                            not previous_detail[
                                "recovery_seed_ref"
                            ]
                            or (
                                recovery_seed[
                                    "recovery_seed_ref"
                                ]
                                == previous_detail[
                                    "recovery_seed_ref"
                                ]
                                and recovery_seed[
                                    "recovery_seed_tree_id"
                                ]
                                == previous_detail[
                                    "recovery_seed_tree_id"
                                ]
                            )
                        )
                    )
                else:
                    recovery_seed_matches = not any(
                        recovery_seed.values()
                    )
                if (
                    attempt != previous_attempt + 1
                    or detail["failure_record_id"]
                    != previous["record_id"]
                    or detail["failure_event_id"]
                    != expected_failure_event_id
                    or int(detail["failure_event_sequence"])
                    != expected_failure_sequence
                    or detail["failure_kind"]
                    != previous_detail["failure_kind"]
                    or int(detail["grant_event_sequence"])
                    <= expected_failure_sequence
                    or not recovery_seed_matches
                ):
                    raise MergeQueueIntegrityError(
                        "post-merge repair grant crosses failure identity"
                    )
                grant_id = str(detail["grant_id"])
                repair_binding_id = str(detail["repair_binding_id"])
                repair_task_binding_id = str(
                    detail["repair_task_binding_id"]
                )
                if (
                    grant_id in seen_grant_ids
                    or repair_binding_id in seen_repair_bindings
                    or repair_task_binding_id
                    in seen_repair_task_bindings
                ):
                    raise MergeQueueIntegrityError(
                        "post-merge correction repair identity is reused"
                    )
                seen_grant_ids.add(grant_id)
                seen_repair_bindings.add(repair_binding_id)
                seen_repair_task_bindings.add(
                    repair_task_binding_id
                )
            elif (
                attempt != previous_attempt
                or detail["authority_kind"] != "repair_grant"
                or detail["authority_id"]
                != previous_detail["grant_id"]
                or int(detail["started_event_sequence"])
                <= int(previous_detail["grant_event_sequence"])
            ):
                raise MergeQueueIntegrityError(
                    "post-merge repair grant consumption binding changed"
                )
        previous = record
        previous_sequence = primary_events[-1][1]
        parent_record_id = str(record["record_id"])

    if (
        int(head["head_ordinal"]) != len(records)
        or head["head_record_id"] != records[-1]["record_id"]
    ):
        raise MergeQueueIntegrityError(
            "post-merge correction chain high-water changed"
        )


@dataclass(frozen=True)
class MergeRequest:
    """One immutable merge candidate and its durable queue state."""

    request_id: str
    branch_name: str
    task_id: str
    priority: str
    lane_id: str
    enqueued_at: float
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: Optional[Path] = None
    commit_sha: str = ""
    canonical_task_id: str = ""
    canonical_task_key: str = ""
    status: str = "pending"
    claimed_at: float = 0.0
    consumer_id: str = ""
    failure_count: int = 0
    failure_reason: str = ""
    claim_token: str = ""
    claim_generation: int = 0

    @property
    def canonical_identity(self) -> str:
        """Return the strongest task identity supplied by the producer."""

        return self.canonical_task_key or self.canonical_task_id or self.task_id

    @property
    def target_repository_id(self) -> str:
        """Return the physical repository this request may mutate."""

        return str(self.metadata.get("target_repository_id") or "").strip()

    @property
    def target_branch(self) -> str:
        """Return the exact local branch this request may mutate."""

        return str(self.metadata.get("target_branch") or "").strip()

    @property
    def has_target_binding(self) -> bool:
        """Return whether the request carries a complete versioned binding."""

        return bool(
            self.metadata.get("target_binding_schema")
            == MERGE_TARGET_BINDING_SCHEMA
            and self.target_repository_id
            and self.target_branch
        )

    @property
    def dedupe_key(self) -> str:
        """Return the stable task-and-commit idempotency key, when available."""

        if not self.commit_sha:
            return ""
        identity = self.canonical_identity.strip().casefold()
        commit = self.commit_sha.strip().casefold()
        parts = [identity, commit]
        if self.has_target_binding:
            parts.extend(
                (
                    self.target_repository_id,
                    self.target_branch,
                )
            )
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "branch_name": self.branch_name,
            "task_id": self.task_id,
            "priority": self.priority,
            "lane_id": self.lane_id,
            "enqueued_at": self.enqueued_at,
            "attempt": self.attempt,
            "metadata": dict(self.metadata),
            "commit_sha": self.commit_sha,
            "canonical_task_id": self.canonical_task_id,
            "canonical_task_key": self.canonical_task_key,
            "status": self.status,
            "claimed_at": self.claimed_at,
            "consumer_id": self.consumer_id,
            "failure_count": self.failure_count,
            "failure_reason": self.failure_reason,
            "claim_token": self.claim_token,
            "claim_generation": self.claim_generation,
            "dedupe_key": self.dedupe_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, file_path: Optional[Path] = None) -> "MergeRequest":
        metadata_value = data.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        commit_sha = str(data.get("commit_sha") or "")
        if not commit_sha:
            commit_sha = _first_metadata_value(metadata, _COMMIT_METADATA_KEYS)
        canonical_task_key = str(data.get("canonical_task_key") or "")
        canonical_task_id = str(data.get("canonical_task_id") or "")
        if not canonical_task_key:
            canonical_task_key = _first_metadata_value(metadata, ("canonical_task_key",))
        if not canonical_task_id:
            canonical_task_id = _first_metadata_value(
                metadata, ("canonical_task_id", "canonical_task_cid", "task_cid")
            )
        return cls(
            request_id=str(data.get("request_id") or ""),
            branch_name=str(data.get("branch_name") or data.get("branch") or ""),
            task_id=str(data.get("task_id") or ""),
            priority=_normalise_priority(str(data.get("priority") or "P2")),
            lane_id=str(data.get("lane_id") or ""),
            enqueued_at=_safe_float(data.get("enqueued_at"), 0.0),
            attempt=max(1, _safe_int(data.get("attempt"), 1)),
            metadata=metadata,
            file_path=file_path,
            commit_sha=commit_sha,
            canonical_task_id=canonical_task_id,
            canonical_task_key=canonical_task_key,
            status=str(data.get("status") or "pending"),
            claimed_at=_safe_float(data.get("claimed_at"), 0.0),
            consumer_id=str(data.get("consumer_id") or ""),
            failure_count=max(0, _safe_int(data.get("failure_count"), 0)),
            failure_reason=str(data.get("failure_reason") or ""),
            claim_token=str(data.get("claim_token") or ""),
            claim_generation=max(0, _safe_int(data.get("claim_generation"), 0)),
        )


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalise_priority(value: str) -> str:
    priority = value.strip().upper()
    return priority if priority in _PRIORITY_ORDER else "P2"


def _first_metadata_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON receipt without exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


class MergeQueue:
    """DuckDB-backed priority queue with atomic claims and bounded retries.

    ``priority_aging_seconds`` promotes an old request by one priority tier for
    every elapsed interval.  This keeps P0 ahead under ordinary load while
    guaranteeing that a continuously busy high-priority tier cannot starve an
    older request forever.
    """

    def __init__(
        self,
        queue_dir: Path | str,
        *,
        max_age_seconds: float = 3600,
        max_queue_size: int = 100,
        max_processing: int | None = None,
        max_worktree_bytes: int | None = None,
        worktree_usage: Callable[[], int] | None = None,
        priority_aging_seconds: float = 300,
        max_attempts: int = 3,
        clock: Callable[[], float] | None = None,
        target_repository_id: str = "",
        target_branch: str = "",
        require_target_binding: bool = False,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.pending_dir = self.queue_dir / "pending"
        self.processing_dir = self.queue_dir / "processing"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"  # compatibility projection
        self.quarantine_dir = self.queue_dir / "quarantine"
        self.cancelled_dir = self.queue_dir / "cancelled"
        self.database_path = self.queue_dir / "merge_queue.duckdb"
        self._legacy_database_path = self.queue_dir / "merge_queue.sqlite3"
        self.max_age_seconds = max(0.0, float(max_age_seconds))
        self.max_queue_size = max(1, int(max_queue_size))
        self.max_processing = max(
            1,
            int(
                max_processing
                if max_processing is not None
                else self.max_queue_size
            ),
        )
        self.max_worktree_bytes = (
            None
            if max_worktree_bytes is None
            else max(0, int(max_worktree_bytes))
        )
        self._worktree_usage = worktree_usage
        self.priority_aging_seconds = max(0.0, float(priority_aging_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self._clock = clock or time.time
        self.target_repository_id = ""
        self.target_branch = ""
        self.require_target_binding = False
        self.bind_target(
            target_repository_id,
            target_branch,
            required=require_target_binding,
        )
        for directory in (
            self.pending_dir,
            self.processing_dir,
            self.completed_dir,
            self.failed_dir,
            self.quarantine_dir,
            self.cancelled_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._import_legacy_files()

    def bind_target(
        self,
        target_repository_id: str,
        target_branch: str,
        *,
        required: bool = True,
    ) -> None:
        """Bind this producer/consumer view to one repository and target ref.

        Binding is process-local while every enqueued request persists the
        versioned values. Existing unbound legacy rows remain in the database
        but are invisible to a required bound consumer.
        """

        repository_id = str(target_repository_id or "").strip()
        branch = str(target_branch or "").strip()
        if bool(repository_id) != bool(branch):
            raise ValueError(
                "target_repository_id and target_branch must be supplied together"
            )
        if required and not repository_id:
            raise ValueError("a required merge target binding must not be empty")
        if (
            self.target_repository_id
            and repository_id
            and self.target_repository_id != repository_id
        ):
            raise ValueError("merge queue target repository binding changed")
        if self.target_branch and branch and self.target_branch != branch:
            raise ValueError("merge queue target branch binding changed")
        if repository_id:
            self.target_repository_id = repository_id
            self.target_branch = branch
        self.require_target_binding = bool(
            self.require_target_binding or required
        )

    def _connect(self) -> DuckDBConnection:
        return open_duckdb_connection(self.database_path)

    def _init_database(self) -> None:
        initialize_duckdb_database(
            self.database_path,
            legacy_sqlite_path=self._legacy_database_path,
            table_names=(
                "merge_requests",
                "post_merge_review_denials",
                "post_merge_correction_chain_records",
                "post_merge_correction_chain_heads",
            ),
            value_transform=lambda table, column, value: (
                None
                if table == "merge_requests"
                and column == "dedupe_key"
                and not str(value or "")
                else value
            ),
            schema_sql="""
                CREATE TABLE IF NOT EXISTS merge_requests (
                    request_id TEXT PRIMARY KEY,
                    branch_name TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    lane_id TEXT NOT NULL,
                    enqueued_at DOUBLE NOT NULL,
                    attempt INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    commit_sha TEXT NOT NULL,
                    canonical_task_id TEXT NOT NULL,
                    canonical_task_key TEXT NOT NULL,
                    dedupe_key TEXT,
                    status TEXT NOT NULL,
                    claimed_at DOUBLE NOT NULL DEFAULT 0,
                    consumer_id TEXT NOT NULL DEFAULT '',
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    failure_reason TEXT NOT NULL DEFAULT '',
                    claim_token TEXT NOT NULL DEFAULT '',
                    claim_generation BIGINT NOT NULL DEFAULT 0,
                    finished_at DOUBLE NOT NULL DEFAULT 0,
                    updated_at DOUBLE NOT NULL
                );
                ALTER TABLE merge_requests
                  ADD COLUMN IF NOT EXISTS claim_token TEXT DEFAULT '';
                ALTER TABLE merge_requests
                  ADD COLUMN IF NOT EXISTS claim_generation BIGINT DEFAULT 0;
                UPDATE merge_requests
                  SET claim_token=COALESCE(claim_token, ''),
                      claim_generation=COALESCE(claim_generation, 0)
                  WHERE claim_token IS NULL OR claim_generation IS NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS merge_requests_dedupe
                  ON merge_requests(dedupe_key);
                CREATE INDEX IF NOT EXISTS merge_requests_stage_order
                  ON merge_requests(status, enqueued_at);
                CREATE TABLE IF NOT EXISTS post_merge_review_denials (
                    terminal_key_id TEXT PRIMARY KEY,
                    denial_id TEXT NOT NULL UNIQUE,
                    target_repository_id TEXT NOT NULL,
                    target_branch TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    canonical_task_key TEXT NOT NULL,
                    canonical_task_cid TEXT NOT NULL,
                    task_binding_id TEXT NOT NULL,
                    implementation_commit TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at DOUBLE NOT NULL
                );
                CREATE INDEX IF NOT EXISTS post_merge_review_denials_target
                  ON post_merge_review_denials(
                    target_repository_id,
                    target_branch,
                    task_id
                  );
                CREATE TABLE IF NOT EXISTS
                  post_merge_review_denial_consumptions (
                    terminal_key_id TEXT PRIMARY KEY,
                    consumption_id TEXT NOT NULL UNIQUE,
                    denial_id TEXT NOT NULL,
                    target_repository_id TEXT NOT NULL,
                    target_branch TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    canonical_task_key TEXT NOT NULL,
                    canonical_task_cid TEXT NOT NULL,
                    task_binding_id TEXT NOT NULL,
                    implementation_commit TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at DOUBLE NOT NULL
                );
                CREATE INDEX IF NOT EXISTS
                  post_merge_review_consumptions_target
                  ON post_merge_review_denial_consumptions(
                    target_repository_id,
                    target_branch,
                    task_id
                  );
                CREATE TABLE IF NOT EXISTS
                  post_merge_correction_chain_records (
                    record_id TEXT PRIMARY KEY,
                    terminal_key_id TEXT NOT NULL,
                    denial_id TEXT NOT NULL,
                    ordinal BIGINT NOT NULL,
                    parent_record_id TEXT NOT NULL UNIQUE,
                    record_kind TEXT NOT NULL,
                    target_repository_id TEXT NOT NULL,
                    target_branch TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    canonical_task_key TEXT NOT NULL,
                    canonical_task_cid TEXT NOT NULL,
                    board_namespace TEXT NOT NULL,
                    task_binding_id TEXT NOT NULL,
                    attempt BIGINT NOT NULL,
                    origin_stream_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at DOUBLE NOT NULL,
                    UNIQUE(terminal_key_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS
                  post_merge_correction_chain_records_target
                  ON post_merge_correction_chain_records(
                    target_repository_id,
                    target_branch,
                    task_id
                  );
                CREATE INDEX IF NOT EXISTS
                  post_merge_correction_chain_records_denial
                  ON post_merge_correction_chain_records(denial_id);
                CREATE TABLE IF NOT EXISTS
                  post_merge_correction_chain_heads (
                    terminal_key_id TEXT PRIMARY KEY,
                    denial_id TEXT NOT NULL UNIQUE,
                    head_record_id TEXT NOT NULL,
                    head_ordinal BIGINT NOT NULL,
                    head_state_id TEXT NOT NULL,
                    updated_at DOUBLE NOT NULL
                );
                """,
        )
        self._initialize_post_merge_correction_registry()

    @staticmethod
    def _post_merge_review_denial_rows(
        connection: DuckDBConnection,
    ) -> list[DuckDBRow]:
        return connection.execute(
            """SELECT terminal_key_id, denial_id,
                      target_repository_id, target_branch, task_id,
                      canonical_task_key, canonical_task_cid,
                      task_binding_id, implementation_commit,
                      record_json
               FROM post_merge_review_denials
               ORDER BY created_at, terminal_key_id"""
        ).fetchall()

    @staticmethod
    def _post_merge_correction_record_rows(
        connection: DuckDBConnection,
    ) -> list[DuckDBRow]:
        return connection.execute(
            """SELECT record_id, terminal_key_id, denial_id, ordinal,
                      parent_record_id, record_kind,
                      target_repository_id, target_branch, task_id,
                      canonical_task_key, canonical_task_cid,
                      board_namespace, task_binding_id, attempt,
                      origin_stream_id, record_json
               FROM post_merge_correction_chain_records
               ORDER BY terminal_key_id, ordinal, record_id"""
        ).fetchall()

    @staticmethod
    def _post_merge_correction_head_rows(
        connection: DuckDBConnection,
    ) -> list[DuckDBRow]:
        return connection.execute(
            """SELECT terminal_key_id, denial_id, head_record_id,
                      head_ordinal, head_state_id
               FROM post_merge_correction_chain_heads
               ORDER BY terminal_key_id"""
        ).fetchall()

    @staticmethod
    def _validate_post_merge_correction_registry_components(
        denials_by_id: Mapping[str, Mapping[str, Any]],
        chains_by_denial_id: Mapping[
            str, Sequence[Mapping[str, Any]]
        ],
        heads_by_denial_id: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if set(heads_by_denial_id) != set(denials_by_id):
            raise MergeQueueIntegrityError(
                "post-merge correction head coverage changed"
            )
        if not set(chains_by_denial_id).issubset(denials_by_id):
            raise MergeQueueIntegrityError(
                "post-merge correction chain is orphaned"
            )
        seen_primary_events: set[tuple[str, str]] = set()
        seen_primary_positions: set[tuple[str, int]] = set()
        seen_grant_ids: set[str] = set()
        seen_repair_bindings: set[str] = set()
        seen_repair_task_bindings: set[str] = set()
        for denial_id, denial in denials_by_id.items():
            _validate_post_merge_correction_chain(
                denial,
                chains_by_denial_id.get(denial_id, ()),
                heads_by_denial_id[denial_id],
                seen_primary_events=seen_primary_events,
                seen_primary_positions=seen_primary_positions,
                seen_grant_ids=seen_grant_ids,
                seen_repair_bindings=seen_repair_bindings,
                seen_repair_task_bindings=(
                    seen_repair_task_bindings
                ),
            )

    def _verified_post_merge_correction_registry(
        self,
        connection: DuckDBConnection,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, tuple[dict[str, Any], ...]],
        dict[str, dict[str, Any]],
    ]:
        denials_by_id: dict[str, dict[str, Any]] = {}
        terminal_keys: set[str] = set()
        for row in self._post_merge_review_denial_rows(connection):
            denial = _decoded_post_merge_review_denial_row(row)
            denial_id = str(denial["denial_id"])
            terminal_key_id = str(denial["terminal_key_id"])
            if (
                denial_id in denials_by_id
                or terminal_key_id in terminal_keys
            ):
                raise MergeQueueIntegrityError(
                    "post-merge denial registry identity is duplicated"
                )
            denials_by_id[denial_id] = denial
            terminal_keys.add(terminal_key_id)

        mutable_chains: dict[str, list[dict[str, Any]]] = {}
        for row in self._post_merge_correction_record_rows(connection):
            record = _decoded_post_merge_correction_chain_row(row)
            mutable_chains.setdefault(
                str(record["denial_id"]), []
            ).append(record)
        chains_by_denial_id = {
            denial_id: tuple(records)
            for denial_id, records in mutable_chains.items()
        }

        heads_by_denial_id: dict[str, dict[str, Any]] = {}
        seen_head_terminals: set[str] = set()
        for row in self._post_merge_correction_head_rows(connection):
            head = _decoded_post_merge_correction_head_row(row)
            denial_id = str(head["denial_id"])
            terminal_key_id = str(head["terminal_key_id"])
            if (
                denial_id in heads_by_denial_id
                or terminal_key_id in seen_head_terminals
            ):
                raise MergeQueueIntegrityError(
                    "post-merge correction head identity is duplicated"
                )
            heads_by_denial_id[denial_id] = head
            seen_head_terminals.add(terminal_key_id)

        self._validate_post_merge_correction_registry_components(
            denials_by_id,
            chains_by_denial_id,
            heads_by_denial_id,
        )
        return (
            denials_by_id,
            chains_by_denial_id,
            heads_by_denial_id,
        )

    def _ensure_post_merge_correction_chain_head(
        self,
        connection: DuckDBConnection,
        denial: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create a migration base head or update an unconsumed one."""

        terminal_key_id = str(denial["terminal_key_id"])
        denial_id = str(denial["denial_id"])
        rows = connection.execute(
            """SELECT terminal_key_id, denial_id, head_record_id,
                      head_ordinal, head_state_id
               FROM post_merge_correction_chain_heads
               WHERE terminal_key_id=? OR denial_id=?""",
            (terminal_key_id, denial_id),
        ).fetchall()
        record_count_row = connection.execute(
            """SELECT COUNT(*) AS count
               FROM post_merge_correction_chain_records
               WHERE terminal_key_id=? OR denial_id=?""",
            (terminal_key_id, denial_id),
        ).fetchone()
        record_count = (
            int(record_count_row["count"])
            if record_count_row is not None
            else 0
        )
        if not rows:
            if record_count:
                raise MergeQueueIntegrityError(
                    "post-merge correction records lack a durable head"
                )
            head = _post_merge_correction_chain_head(
                terminal_key_id=terminal_key_id,
                denial_id=denial_id,
                head_record_id=denial_id,
                head_ordinal=0,
            )
            connection.execute(
                """INSERT INTO post_merge_correction_chain_heads (
                     terminal_key_id, denial_id, head_record_id,
                     head_ordinal, head_state_id, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    head["terminal_key_id"],
                    head["denial_id"],
                    head["head_record_id"],
                    head["head_ordinal"],
                    head["head_state_id"],
                    self._clock(),
                ),
            )
            return head
        if len(rows) != 1:
            raise MergeQueueIntegrityError(
                "post-merge correction head conflicts across denials"
            )
        existing = _decoded_post_merge_correction_head_row(rows[0])
        if existing["terminal_key_id"] != terminal_key_id:
            raise MergeQueueIntegrityError(
                "post-merge correction head crosses terminal keys"
            )
        if existing["denial_id"] == denial_id:
            if (
                int(existing["head_ordinal"]) == 0
                and (
                    existing["head_record_id"] != denial_id
                    or record_count
                )
            ):
                raise MergeQueueIntegrityError(
                    "post-merge correction base head is inconsistent"
                )
            return existing
        if int(existing["head_ordinal"]) != 0 or record_count:
            raise MergeQueueFenceError(
                "post-merge denial cannot evolve after authority consumption"
            )
        replacement = _post_merge_correction_chain_head(
            terminal_key_id=terminal_key_id,
            denial_id=denial_id,
            head_record_id=denial_id,
            head_ordinal=0,
        )
        connection.execute(
            """UPDATE post_merge_correction_chain_heads
               SET denial_id=?, head_record_id=?, head_ordinal=?,
                   head_state_id=?, updated_at=?
               WHERE terminal_key_id=? AND head_state_id=?""",
            (
                replacement["denial_id"],
                replacement["head_record_id"],
                replacement["head_ordinal"],
                replacement["head_state_id"],
                self._clock(),
                terminal_key_id,
                existing["head_state_id"],
            ),
        )
        stored = connection.execute(
            """SELECT terminal_key_id, denial_id, head_record_id,
                      head_ordinal, head_state_id
               FROM post_merge_correction_chain_heads
               WHERE terminal_key_id=?""",
            (terminal_key_id,),
        ).fetchone()
        if (
            stored is None
            or _decoded_post_merge_correction_head_row(stored)
            != replacement
        ):
            raise MergeQueueFenceError(
                "post-merge correction base-head CAS failed"
            )
        return replacement

    def _initialize_post_merge_correction_registry(self) -> None:
        """Backfill legacy heads once; never repair a deployed registry."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                migration = connection.execute(
                    """SELECT value
                       FROM agent_supervisor_store_metadata
                       WHERE key=?""",
                    (_POST_MERGE_CORRECTION_REGISTRY_MIGRATION_KEY,),
                ).fetchone()
                if migration is not None and str(migration["value"]) != (
                    _POST_MERGE_CORRECTION_REGISTRY_MIGRATION_SCHEMA
                ):
                    raise MergeQueueIntegrityError(
                        "post-merge correction registry migration marker changed"
                    )
                if migration is None:
                    for row in self._post_merge_review_denial_rows(
                        connection
                    ):
                        denial = _decoded_post_merge_review_denial_row(row)
                        self._ensure_post_merge_correction_chain_head(
                            connection,
                            denial,
                        )
                self._verified_post_merge_correction_registry(connection)
                if migration is None:
                    connection.execute(
                        """INSERT INTO agent_supervisor_store_metadata (
                             key, value
                           ) VALUES (?, ?)""",
                        (
                            _POST_MERGE_CORRECTION_REGISTRY_MIGRATION_KEY,
                            _POST_MERGE_CORRECTION_REGISTRY_MIGRATION_SCHEMA,
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _assert_post_merge_correction_target(
        self,
        transition: Mapping[str, Any],
    ) -> None:
        if (
            self.target_repository_id
            and (
                transition["target_repository_id"]
                != self.target_repository_id
                or transition["target_branch"] != self.target_branch
            )
        ):
            raise MergeQueueFenceError(
                "post-merge correction target differs from queue binding"
            )
        if self.require_target_binding and not self.target_repository_id:
            raise MergeQueueFenceError(
                "bound merge queue lacks a correction target"
            )

    @staticmethod
    def _insert_post_merge_correction_chain_record(
        connection: DuckDBConnection,
        record: Mapping[str, Any],
        canonical: str,
        created_at: float,
    ) -> None:
        connection.execute(
            """INSERT INTO post_merge_correction_chain_records (
                 record_id, terminal_key_id, denial_id, ordinal,
                 parent_record_id, record_kind,
                 target_repository_id, target_branch, task_id,
                 canonical_task_key, canonical_task_cid,
                 board_namespace, task_binding_id, attempt,
                 origin_stream_id, record_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["record_id"],
                record["terminal_key_id"],
                record["denial_id"],
                record["ordinal"],
                record["parent_record_id"],
                record["record_kind"],
                record["target_repository_id"],
                record["target_branch"],
                record["task_id"],
                record["canonical_task_key"],
                record["canonical_task_cid"],
                record["board_namespace"],
                record["task_binding_id"],
                record["attempt"],
                record["origin_stream_id"],
                canonical,
                float(created_at),
            ),
        )

    def _advance_post_merge_correction_head(
        self,
        connection: DuckDBConnection,
        *,
        current_head: Mapping[str, Any],
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        next_head = _post_merge_correction_chain_head(
            terminal_key_id=str(record["terminal_key_id"]),
            denial_id=str(record["denial_id"]),
            head_record_id=str(record["record_id"]),
            head_ordinal=int(record["ordinal"]),
        )
        connection.execute(
            """UPDATE post_merge_correction_chain_heads
               SET head_record_id=?, head_ordinal=?, head_state_id=?,
                   updated_at=?
               WHERE terminal_key_id=? AND denial_id=?
                 AND head_record_id=? AND head_ordinal=?
                 AND head_state_id=?""",
            (
                next_head["head_record_id"],
                next_head["head_ordinal"],
                next_head["head_state_id"],
                self._clock(),
                current_head["terminal_key_id"],
                current_head["denial_id"],
                current_head["head_record_id"],
                current_head["head_ordinal"],
                current_head["head_state_id"],
            ),
        )
        stored = connection.execute(
            """SELECT terminal_key_id, denial_id, head_record_id,
                      head_ordinal, head_state_id
               FROM post_merge_correction_chain_heads
               WHERE terminal_key_id=?""",
            (record["terminal_key_id"],),
        ).fetchone()
        if (
            stored is None
            or _decoded_post_merge_correction_head_row(stored)
            != next_head
        ):
            raise MergeQueueFenceError(
                "post-merge correction head CAS failed"
            )
        return next_head

    def record_post_merge_correction_transition(
        self,
        value: Mapping[str, Any],
        *,
        expected_parent_record_id: str,
    ) -> dict[str, Any]:
        """CAS-append one exact one-shot correction transition."""

        _record_kind, transition = (
            _validated_post_merge_correction_transition(value)
        )
        self._assert_post_merge_correction_target(transition)
        expected_parent = str(expected_parent_record_id or "")
        if not expected_parent:
            raise MergeQueueFenceError(
                "post-merge correction CAS parent is required"
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                (
                    denials,
                    chains,
                    heads,
                ) = self._verified_post_merge_correction_registry(
                    connection
                )
                denial_id = str(transition["denial_id"])
                denial = denials.get(denial_id)
                if denial is None:
                    raise MergeQueueFenceError(
                        "post-merge correction denial is unavailable"
                    )
                if any(
                    transition[name] != expected
                    for name, expected in (
                        _post_merge_correction_identity_from_denial(
                            denial
                        ).items()
                    )
                ):
                    raise MergeQueueFenceError(
                        "post-merge correction identity differs from denial"
                    )
                chain = list(chains.get(denial_id, ()))
                current_head = heads[denial_id]
                parent_ordinal = (
                    0 if expected_parent == denial_id else None
                )
                if parent_ordinal is None:
                    parent_record = next(
                        (
                            record
                            for record in chain
                            if record["record_id"] == expected_parent
                        ),
                        None,
                    )
                    if parent_record is not None:
                        parent_ordinal = int(parent_record["ordinal"])
                if parent_ordinal is None:
                    raise MergeQueueFenceError(
                        "post-merge correction CAS parent is foreign"
                    )
                candidate = _post_merge_correction_chain_record(
                    transition,
                    terminal_key_id=str(denial["terminal_key_id"]),
                    ordinal=parent_ordinal + 1,
                    parent_record_id=expected_parent,
                )
                candidate, canonical = (
                    _validated_post_merge_correction_chain_record(
                        candidate
                    )
                )
                existing_child = next(
                    (
                        record
                        for record in chain
                        if record["parent_record_id"] == expected_parent
                    ),
                    None,
                )
                if existing_child is not None:
                    if existing_child != candidate:
                        raise MergeQueueFenceError(
                            "post-merge correction CAS parent was consumed"
                        )
                    connection.commit()
                    return existing_child
                if current_head["head_record_id"] != expected_parent:
                    raise MergeQueueFenceError(
                        "post-merge correction CAS parent is stale"
                    )
                next_head = _post_merge_correction_chain_head(
                    terminal_key_id=str(candidate["terminal_key_id"]),
                    denial_id=denial_id,
                    head_record_id=str(candidate["record_id"]),
                    head_ordinal=int(candidate["ordinal"]),
                )
                prospective_chains = dict(chains)
                prospective_chains[denial_id] = (*chain, candidate)
                prospective_heads = dict(heads)
                prospective_heads[denial_id] = next_head
                self._validate_post_merge_correction_registry_components(
                    denials,
                    prospective_chains,
                    prospective_heads,
                )
                self._insert_post_merge_correction_chain_record(
                    connection,
                    candidate,
                    canonical,
                    self._clock(),
                )
                self._advance_post_merge_correction_head(
                    connection,
                    current_head=current_head,
                    record=candidate,
                )
                connection.commit()
                return candidate
            except Exception:
                connection.rollback()
                raise

    def record_post_merge_correction_consumption(
        self,
        value: Mapping[str, Any],
        *,
        expected_parent_record_id: str,
    ) -> dict[str, Any]:
        kind, _transition = _validated_post_merge_correction_transition(
            value
        )
        if kind not in {"denial_consumed", "grant_consumed"}:
            raise MergeQueueIntegrityError(
                "post-merge correction consumption schema is required"
            )
        return self.record_post_merge_correction_transition(
            value,
            expected_parent_record_id=expected_parent_record_id,
        )

    def record_post_merge_correction_failure(
        self,
        value: Mapping[str, Any],
        *,
        expected_parent_record_id: str,
    ) -> dict[str, Any]:
        kind, _transition = _validated_post_merge_correction_transition(
            value
        )
        if kind != "correction_failed":
            raise MergeQueueIntegrityError(
                "post-merge correction failure schema is required"
            )
        return self.record_post_merge_correction_transition(
            value,
            expected_parent_record_id=expected_parent_record_id,
        )

    def record_post_merge_correction_repair_grant(
        self,
        value: Mapping[str, Any],
        *,
        expected_parent_record_id: str,
    ) -> dict[str, Any]:
        kind, _transition = _validated_post_merge_correction_transition(
            value
        )
        if kind != "repair_granted":
            raise MergeQueueIntegrityError(
                "post-merge correction repair-grant schema is required"
            )
        return self.record_post_merge_correction_transition(
            value,
            expected_parent_record_id=expected_parent_record_id,
        )

    def record_post_merge_correction_legacy_failure_anchor(
        self,
        value: Mapping[str, Any],
        *,
        expected_parent_record_id: str,
    ) -> dict[str, Any]:
        """Apply one explicit exact operator migration; never infer lineage."""

        kind, _transition = _validated_post_merge_correction_transition(
            value
        )
        if kind != "legacy_failure_anchored":
            raise MergeQueueIntegrityError(
                "post-merge correction legacy anchor schema is required"
            )
        return self.record_post_merge_correction_transition(
            value,
            expected_parent_record_id=expected_parent_record_id,
        )

    def record_post_merge_correction_legacy_high_water_anchor(
        self,
        value: Mapping[str, Any],
        *,
        expected_parent_record_id: str,
    ) -> dict[str, Any]:
        """Apply one explicit bounded legacy high-water migration."""

        kind, _transition = _validated_post_merge_correction_transition(
            value
        )
        if kind != "legacy_high_water_anchored":
            raise MergeQueueIntegrityError(
                "post-merge correction legacy high-water anchor schema is required"
            )
        return self.record_post_merge_correction_transition(
            value,
            expected_parent_record_id=expected_parent_record_id,
        )

    def mirror_post_merge_correction_history(
        self,
        transitions: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        """Atomically import a verified full prefix from a rotating ledger."""

        verified_transitions = [
            _validated_post_merge_correction_transition(value)[1]
            for value in transitions
        ]
        if not verified_transitions:
            return ()
        denial_ids = {
            str(transition["denial_id"])
            for transition in verified_transitions
        }
        if len(denial_ids) != 1:
            raise MergeQueueIntegrityError(
                "one mirrored correction history must bind one denial"
            )
        for transition in verified_transitions:
            self._assert_post_merge_correction_target(transition)
        denial_id = denial_ids.pop()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                (
                    denials,
                    chains,
                    heads,
                ) = self._verified_post_merge_correction_registry(
                    connection
                )
                denial = denials.get(denial_id)
                if denial is None:
                    raise MergeQueueFenceError(
                        "mirrored post-merge correction denial is unavailable"
                    )
                expected_identity = (
                    _post_merge_correction_identity_from_denial(denial)
                )
                expected_records: list[dict[str, Any]] = []
                parent_record_id = denial_id
                for ordinal, transition in enumerate(
                    verified_transitions,
                    start=1,
                ):
                    if any(
                        transition[name] != expected
                        for name, expected in expected_identity.items()
                    ):
                        raise MergeQueueFenceError(
                            "mirrored correction identity differs from denial"
                        )
                    record = _post_merge_correction_chain_record(
                        transition,
                        terminal_key_id=str(denial["terminal_key_id"]),
                        ordinal=ordinal,
                        parent_record_id=parent_record_id,
                    )
                    record, _canonical = (
                        _validated_post_merge_correction_chain_record(
                            record
                        )
                    )
                    expected_records.append(record)
                    parent_record_id = str(record["record_id"])
                existing = list(chains.get(denial_id, ()))
                shared_length = min(
                    len(existing), len(expected_records)
                )
                if (
                    existing[:shared_length]
                    != expected_records[:shared_length]
                ):
                    raise MergeQueueFenceError(
                        "mirrored correction history conflicts with registry"
                    )
                if len(existing) >= len(expected_records):
                    connection.commit()
                    return tuple(existing)

                prospective_chains = dict(chains)
                prospective_chains[denial_id] = tuple(expected_records)
                prospective_heads = dict(heads)
                prospective_heads[denial_id] = (
                    _post_merge_correction_chain_head(
                        terminal_key_id=str(denial["terminal_key_id"]),
                        denial_id=denial_id,
                        head_record_id=str(
                            expected_records[-1]["record_id"]
                        ),
                        head_ordinal=len(expected_records),
                    )
                )
                self._validate_post_merge_correction_registry_components(
                    denials,
                    prospective_chains,
                    prospective_heads,
                )
                current_head = heads[denial_id]
                for record in expected_records[len(existing) :]:
                    _verified, canonical = (
                        _validated_post_merge_correction_chain_record(
                            record
                        )
                    )
                    self._insert_post_merge_correction_chain_record(
                        connection,
                        record,
                        canonical,
                        self._clock(),
                    )
                    current_head = (
                        self._advance_post_merge_correction_head(
                            connection,
                            current_head=current_head,
                            record=record,
                        )
                    )
                connection.commit()
                return tuple(expected_records)
            except Exception:
                connection.rollback()
                raise

    def verified_post_merge_correction_chain(
        self,
        denial_id: str = "",
    ) -> tuple[dict[str, Any], ...]:
        """Return a fully verified durable chain, optionally for one denial."""

        selected_denial_id = str(denial_id or "")
        with self._connect() as connection:
            denials, chains, _heads = (
                self._verified_post_merge_correction_registry(connection)
            )
        selected: list[dict[str, Any]] = []
        for current_denial_id, denial in denials.items():
            if (
                selected_denial_id
                and current_denial_id != selected_denial_id
            ):
                continue
            if (
                self.target_repository_id
                and (
                    denial["target_repository_id"]
                    != self.target_repository_id
                    or denial["target_branch"] != self.target_branch
                )
            ):
                continue
            selected.extend(chains.get(current_denial_id, ()))
        return tuple(selected)

    def verified_post_merge_correction_authority(
        self,
        denial_id: str,
    ) -> dict[str, Any]:
        """Project the sole live authority without consulting retained events."""

        selected_denial_id = str(denial_id or "")
        if not selected_denial_id:
            return {}
        with self._connect() as connection:
            denials, chains, heads = (
                self._verified_post_merge_correction_registry(connection)
            )
        denial = denials.get(selected_denial_id)
        if denial is None:
            return {}
        if (
            self.target_repository_id
            and (
                denial["target_repository_id"]
                != self.target_repository_id
                or denial["target_branch"] != self.target_branch
            )
        ):
            return {}
        chain = chains.get(selected_denial_id, ())
        head = heads[selected_denial_id]
        authority_available = False
        authority_kind = ""
        authority_id = ""
        authority_event_sequence = 0
        authorized_attempt = 0
        state = "unavailable"
        grant_binding = {
            "failure_record_id": "",
            "failure_event_id": "",
            "failure_event_sequence": 0,
            "failure_kind": "",
            "repair_task_id": "",
            "repair_task_binding_id": "",
            "repair_binding_id": "",
        }
        recovery_seed = {
            "recovery_seed_ref": "",
            "recovery_seed_tree_id": "",
            "recovery_seed_submodule_path": "",
            "recovery_seed_submodule_commit": "",
        }
        if not chain:
            if denial["correction_authorized"] is True:
                authority_available = True
                authority_kind = "review_denial"
                authority_id = selected_denial_id
                authority_event_sequence = int(
                    denial["review_attempt"]
                )
                authorized_attempt = int(
                    denial["target_implementation_attempt"]
                )
                state = "available"
            else:
                state = "not_authorized"
        else:
            terminal = chain[-1]
            detail = terminal["detail"]
            kind = str(terminal["record_kind"])
            if kind == "repair_granted":
                authority_available = True
                authority_kind = "repair_grant"
                authority_id = str(detail["grant_id"])
                authority_event_sequence = int(
                    detail["grant_event_sequence"]
                )
                authorized_attempt = int(terminal["attempt"])
                state = "available"
                grant_binding = {
                    name: detail[name]
                    for name in grant_binding
                }
                recovery_seed = (
                    _validated_post_merge_recovery_seed(
                        detail,
                        required=False,
                    )
                )
            elif kind in {"denial_consumed", "grant_consumed"}:
                authority_kind = str(detail["authority_kind"])
                authority_id = str(detail["authority_id"])
                authorized_attempt = int(terminal["attempt"])
                state = "consumed"
            elif kind == "correction_failed":
                authority_kind = str(detail["authority_kind"])
                authority_id = str(detail["authority_id"])
                authorized_attempt = int(terminal["attempt"])
                state = "failed"
            elif kind == "legacy_failure_anchored":
                authority_kind = "review_denial"
                authority_id = selected_denial_id
                authorized_attempt = int(terminal["attempt"])
                state = "legacy_failure_anchored"
            else:
                authority_kind = "review_denial"
                authority_id = selected_denial_id
                authorized_attempt = int(terminal["attempt"])
                state = "legacy_high_water_anchored"
        material: dict[str, Any] = {
            "schema": POST_MERGE_CORRECTION_AUTHORITY_STATE_SCHEMA,
            "terminal_key_id": str(denial["terminal_key_id"]),
            "denial_id": selected_denial_id,
            "implementation_commit": str(
                denial["implementation_commit"]
            ),
            **_post_merge_correction_identity_from_denial(denial),
            "head_record_id": str(head["head_record_id"]),
            "head_ordinal": int(head["head_ordinal"]),
            "state": state,
            "authority_available": authority_available,
            "authority_kind": authority_kind,
            "authority_id": authority_id,
            "authority_event_sequence": authority_event_sequence,
            "authorized_attempt": authorized_attempt,
            **grant_binding,
            **recovery_seed,
        }
        return {
            **material,
            "authority_state_id": content_identity(material),
        }

    def record_post_merge_review_denial(
        self,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Commit one permanent, exact-candidate review-denial tombstone."""

        record, canonical = _validated_post_merge_review_denial(value)
        if (
            self.target_repository_id
            and (
                record["target_repository_id"]
                != self.target_repository_id
                or record["target_branch"] != self.target_branch
            )
        ):
            raise MergeQueueFenceError(
                "post-merge denial target differs from queue binding"
            )
        if self.require_target_binding and not self.target_repository_id:
            raise MergeQueueFenceError(
                "bound merge queue lacks a target for denial authority"
            )
        terminal_key_id = str(record["terminal_key_id"])
        denial_id = str(record["denial_id"])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    """SELECT terminal_key_id, denial_id, record_json
                       FROM post_merge_review_denials
                       WHERE terminal_key_id=? OR denial_id=?""",
                    (terminal_key_id, denial_id),
                ).fetchall()
                if existing:
                    if len(existing) != 1:
                        raise MergeQueueIntegrityError(
                            "conflicting post-merge denial authority exists"
                        )
                    try:
                        existing_decoded = json.loads(
                            str(existing[0]["record_json"])
                        )
                    except (
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise MergeQueueIntegrityError(
                            "existing post-merge denial authority is malformed"
                        ) from exc
                    existing_record, existing_canonical = (
                        _validated_post_merge_review_denial(
                            existing_decoded,
                            allow_legacy=True,
                        )
                    )
                    if (
                        str(existing[0]["record_json"])
                        != existing_canonical
                        or str(existing[0]["terminal_key_id"])
                        != str(existing_record["terminal_key_id"])
                        or str(existing[0]["denial_id"])
                        != str(existing_record["denial_id"])
                    ):
                        raise MergeQueueIntegrityError(
                            "existing post-merge denial authority changed"
                        )
                    if (
                        str(existing_record["terminal_key_id"])
                        != terminal_key_id
                    ):
                        raise MergeQueueIntegrityError(
                            "post-merge denial identity crosses terminal keys"
                        )
                    if (
                        str(existing[0]["denial_id"]) == denial_id
                        and existing_canonical == canonical
                    ):
                        self._ensure_post_merge_correction_chain_head(
                            connection,
                            existing_record,
                        )
                        self._verified_post_merge_correction_registry(
                            connection
                        )
                        connection.commit()
                        return existing_record
                    consumption_rows = connection.execute(
                        """SELECT terminal_key_id, consumption_id,
                                  denial_id, record_json
                           FROM post_merge_review_denial_consumptions
                           WHERE terminal_key_id=?""",
                        (terminal_key_id,),
                    ).fetchall()
                    if consumption_rows:
                        if len(consumption_rows) != 1:
                            raise MergeQueueIntegrityError(
                                "conflicting denial consumption authority exists"
                            )
                        try:
                            consumption_decoded = json.loads(
                                str(consumption_rows[0]["record_json"])
                            )
                        except (
                            TypeError,
                            ValueError,
                            json.JSONDecodeError,
                        ) as exc:
                            raise MergeQueueIntegrityError(
                                "existing denial consumption is malformed"
                            ) from exc
                        (
                            consumption_record,
                            consumption_canonical,
                        ) = _validated_post_merge_review_denial_consumption(
                            consumption_decoded
                        )
                        consumption_shared_fields = (
                            "terminal_key_id",
                            "denial_id",
                            "target_repository_id",
                            "target_branch",
                            "task_id",
                            "canonical_task_key",
                            "canonical_task_cid",
                            "board_namespace",
                            "task_binding_id",
                            "implementation_commit",
                            "implementation_attempt",
                            "target_implementation_attempt",
                            "correction_origin_stream_id",
                        )
                        if (
                            existing_record.get("schema")
                            != POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA
                            or not str(
                                existing_record.get("source_event_id") or ""
                            )
                            or int(
                                existing_record.get(
                                    "source_event_sequence"
                                )
                                or 0
                            )
                            < 1
                            or str(
                                consumption_rows[0]["record_json"]
                            )
                            != consumption_canonical
                            or str(
                                consumption_rows[0]["terminal_key_id"]
                            )
                            != str(
                                consumption_record["terminal_key_id"]
                            )
                            or str(
                                consumption_rows[0]["consumption_id"]
                            )
                            != str(consumption_record["consumption_id"])
                            or str(consumption_rows[0]["denial_id"])
                            != str(consumption_record["denial_id"])
                            or existing_record.get(
                                "correction_authorized"
                            )
                            is not True
                            or any(
                                consumption_record[name]
                                != existing_record[name]
                                for name in consumption_shared_fields
                            )
                        ):
                            raise MergeQueueIntegrityError(
                                "consumed denial representative changed"
                            )
                        # Consumption is global for the immutable terminal
                        # key, while only its owning origin may mint the
                        # marker. Once consumed, pin that verified origin
                        # representative so later same-terminal migrations
                        # cannot rewrite the authority the marker records.
                        connection.commit()
                        return existing_record
                    if (
                        existing_record.get("schema")
                        == POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA
                        and
                        existing_record.get(
                            "correction_authorized"
                        )
                        is True
                        and record.get("correction_authorized") is True
                        and existing_record[
                            "correction_origin_stream_id"
                        ]
                        != record["correction_origin_stream_id"]
                    ):
                        # One immutable terminal key has one global retry
                        # budget and therefore one origin owner. Competing
                        # authorized streams cannot be deterministically
                        # coalesced before consumption: a terminal event from
                        # the losing representative could otherwise spend the
                        # budget without being able to seal its marker.
                        raise MergeQueueIntegrityError(
                            "post-merge denial has multiple authorized "
                            "origin streams"
                        )
                    candidates: list[
                        tuple[tuple[int, int, str], dict[str, Any]]
                    ] = []
                    for candidate in (existing_record, record):
                        representative = dict(candidate)
                        representative.pop("denial_id", None)
                        representative.pop(
                            "correction_authorized",
                            None,
                        )
                        # Same-terminal records can legitimately differ after
                        # another lane reviewed the same immutable
                        # implementation against a later target HEAD. Prefer
                        # the strictly authorized origin record, then converge
                        # ties by canonical content regardless of migration
                        # order. Authorization remains attached to its own
                        # verified origin payload.
                        candidates.append(
                            (
                                (
                                    0
                                    if candidate.get("schema")
                                    == (
                                        POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA
                                    )
                                    else 1,
                                    0
                                    if candidate[
                                        "correction_authorized"
                                    ]
                                    else 1,
                                    _canonical_json(representative),
                                ),
                                dict(candidate),
                            )
                        )
                    selected = dict(min(candidates, key=lambda item: item[0])[1])
                    selected.pop("denial_id", None)
                    selected["denial_id"] = content_identity(selected)
                    selected_record, selected_canonical = (
                        _validated_post_merge_review_denial(selected)
                    )
                    if selected_canonical == existing_canonical:
                        self._ensure_post_merge_correction_chain_head(
                            connection,
                            existing_record,
                        )
                        self._verified_post_merge_correction_registry(
                            connection
                        )
                        connection.commit()
                        return existing_record
                    self._ensure_post_merge_correction_chain_head(
                        connection,
                        selected_record,
                    )
                    connection.execute(
                        """UPDATE post_merge_review_denials
                           SET denial_id=?, record_json=?
                           WHERE terminal_key_id=?""",
                        (
                            selected_record["denial_id"],
                            selected_canonical,
                            terminal_key_id,
                        ),
                    )
                    self._verified_post_merge_correction_registry(
                        connection
                    )
                    connection.commit()
                    return selected_record
                connection.execute(
                    """INSERT INTO post_merge_review_denials (
                         terminal_key_id, denial_id,
                         target_repository_id, target_branch, task_id,
                         canonical_task_key, canonical_task_cid,
                         task_binding_id, implementation_commit,
                         record_json, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        terminal_key_id,
                        denial_id,
                        record["target_repository_id"],
                        record["target_branch"],
                        record["task_id"],
                        record["canonical_task_key"],
                        record["canonical_task_cid"],
                        record["task_binding_id"],
                        record["implementation_commit"],
                        canonical,
                        self._clock(),
                    ),
                )
                self._ensure_post_merge_correction_chain_head(
                    connection,
                    record,
                )
                self._verified_post_merge_correction_registry(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return record

    def verified_post_merge_review_denials(
        self,
    ) -> tuple[dict[str, Any], ...]:
        """Return all permanent denial tombstones or fail on any corruption."""

        with self._connect() as connection:
            denials, _chains, _heads = (
                self._verified_post_merge_correction_registry(connection)
            )
        verified: list[dict[str, Any]] = []
        for record in denials.values():
            if (
                self.target_repository_id
                and (
                    record["target_repository_id"]
                    != self.target_repository_id
                    or record["target_branch"] != self.target_branch
                )
            ):
                continue
            verified.append(record)
        return tuple(verified)

    def record_post_merge_review_denial_consumption(
        self,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist one exact terminal event that consumed a correction."""

        record, canonical = (
            _validated_post_merge_review_denial_consumption(value)
        )
        if (
            self.target_repository_id
            and (
                record["target_repository_id"]
                != self.target_repository_id
                or record["target_branch"] != self.target_branch
            )
        ):
            raise MergeQueueFenceError(
                "post-merge denial consumption target differs from queue"
            )
        if self.require_target_binding and not self.target_repository_id:
            raise MergeQueueFenceError(
                "bound merge queue lacks a target for consumption authority"
            )
        terminal_key_id = str(record["terminal_key_id"])
        consumption_id = str(record["consumption_id"])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                denial_rows = connection.execute(
                    """SELECT denial_id, record_json
                       FROM post_merge_review_denials
                       WHERE terminal_key_id=?""",
                    (terminal_key_id,),
                ).fetchall()
                if len(denial_rows) != 1:
                    raise MergeQueueIntegrityError(
                        "consumption lacks exactly one permanent denial"
                    )
                try:
                    denial_decoded = json.loads(
                        str(denial_rows[0]["record_json"])
                    )
                except (
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    raise MergeQueueIntegrityError(
                        "consumption denial authority is malformed"
                    ) from exc
                denial, denial_canonical = (
                    _validated_post_merge_review_denial(
                        denial_decoded,
                        allow_legacy=True,
                    )
                )
                shared_fields = (
                    "terminal_key_id",
                    "denial_id",
                    "target_repository_id",
                    "target_branch",
                    "task_id",
                    "canonical_task_key",
                    "canonical_task_cid",
                    "board_namespace",
                    "task_binding_id",
                    "implementation_commit",
                    "implementation_attempt",
                    "target_implementation_attempt",
                    "correction_origin_stream_id",
                )
                if (
                    denial_canonical
                    != str(denial_rows[0]["record_json"])
                    or str(denial_rows[0]["denial_id"])
                    != str(denial["denial_id"])
                    or denial.get("schema")
                    != POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA
                    or not str(denial.get("source_event_id") or "")
                    or int(denial.get("source_event_sequence") or 0) < 1
                    or denial.get("correction_authorized") is not True
                    or any(
                        record[name] != denial[name]
                        for name in shared_fields
                    )
                ):
                    raise MergeQueueIntegrityError(
                        "consumption does not match its permanent denial"
                    )
                existing = connection.execute(
                    """SELECT terminal_key_id, consumption_id, record_json
                       FROM post_merge_review_denial_consumptions
                       WHERE terminal_key_id=? OR consumption_id=?""",
                    (terminal_key_id, consumption_id),
                ).fetchall()
                if existing:
                    if len(existing) != 1:
                        raise MergeQueueIntegrityError(
                            "conflicting denial consumption authority exists"
                        )
                    try:
                        existing_decoded = json.loads(
                            str(existing[0]["record_json"])
                        )
                    except (
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise MergeQueueIntegrityError(
                            "existing denial consumption is malformed"
                        ) from exc
                    existing_record, existing_canonical = (
                        _validated_post_merge_review_denial_consumption(
                            existing_decoded
                        )
                    )
                    if (
                        str(existing[0]["record_json"])
                        != existing_canonical
                        or str(existing[0]["terminal_key_id"])
                        != str(existing_record["terminal_key_id"])
                        or str(existing[0]["consumption_id"])
                        != str(existing_record["consumption_id"])
                        or existing_canonical != canonical
                    ):
                        raise MergeQueueIntegrityError(
                            "existing denial consumption authority changed"
                        )
                    connection.commit()
                    return existing_record
                connection.execute(
                    """INSERT INTO post_merge_review_denial_consumptions (
                         terminal_key_id, consumption_id, denial_id,
                         target_repository_id, target_branch, task_id,
                         canonical_task_key, canonical_task_cid,
                         task_binding_id, implementation_commit,
                         record_json, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        terminal_key_id,
                        consumption_id,
                        record["denial_id"],
                        record["target_repository_id"],
                        record["target_branch"],
                        record["task_id"],
                        record["canonical_task_key"],
                        record["canonical_task_cid"],
                        record["task_binding_id"],
                        record["implementation_commit"],
                        canonical,
                        self._clock(),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return record

    def verified_post_merge_review_denial_consumptions(
        self,
    ) -> tuple[dict[str, Any], ...]:
        """Return durable correction consumptions or fail on any corruption."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT terminal_key_id, consumption_id, denial_id,
                          target_repository_id, target_branch, task_id,
                          canonical_task_key, canonical_task_cid,
                          task_binding_id, implementation_commit,
                          record_json
                   FROM post_merge_review_denial_consumptions
                   ORDER BY created_at, terminal_key_id"""
            ).fetchall()
            denial_rows = connection.execute(
                """SELECT terminal_key_id, denial_id, record_json
                   FROM post_merge_review_denials"""
            ).fetchall()
        denials: dict[str, dict[str, Any]] = {}
        for row in denial_rows:
            try:
                decoded = json.loads(str(row["record_json"]))
            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise MergeQueueIntegrityError(
                    "consumption registry denial is malformed"
                ) from exc
            denial, canonical = _validated_post_merge_review_denial(
                decoded,
                allow_legacy=True,
            )
            terminal_key_id = str(row["terminal_key_id"])
            if (
                canonical != str(row["record_json"])
                or terminal_key_id
                != str(denial["terminal_key_id"])
                or str(row["denial_id"]) != str(denial["denial_id"])
                or denial.get("schema")
                != POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA
                or not str(denial.get("source_event_id") or "")
                or int(denial.get("source_event_sequence") or 0) < 1
            ):
                raise MergeQueueIntegrityError(
                    "consumption registry denial binding changed"
                )
            denials[terminal_key_id] = denial
        verified: list[dict[str, Any]] = []
        for row in rows:
            try:
                decoded = json.loads(str(row["record_json"]))
            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise MergeQueueIntegrityError(
                    "denial consumption registry contains malformed JSON"
                ) from exc
            record, canonical = (
                _validated_post_merge_review_denial_consumption(
                    decoded
                )
            )
            row_bindings = {
                "terminal_key_id": str(row["terminal_key_id"]),
                "consumption_id": str(row["consumption_id"]),
                "denial_id": str(row["denial_id"]),
                "target_repository_id": str(
                    row["target_repository_id"]
                ),
                "target_branch": str(row["target_branch"]),
                "task_id": str(row["task_id"]),
                "canonical_task_key": str(
                    row["canonical_task_key"]
                ),
                "canonical_task_cid": str(
                    row["canonical_task_cid"]
                ),
                "task_binding_id": str(row["task_binding_id"]),
                "implementation_commit": str(
                    row["implementation_commit"]
                ),
            }
            denial = denials.get(str(record["terminal_key_id"]))
            shared_fields = (
                "denial_id",
                "target_repository_id",
                "target_branch",
                "task_id",
                "canonical_task_key",
                "canonical_task_cid",
                "board_namespace",
                "task_binding_id",
                "implementation_commit",
                "implementation_attempt",
                "target_implementation_attempt",
                "correction_origin_stream_id",
            )
            if (
                canonical != str(row["record_json"])
                or any(
                    record[name] != item
                    for name, item in row_bindings.items()
                )
                or denial is None
                or denial.get("correction_authorized") is not True
                or any(
                    record[name] != denial[name]
                    for name in shared_fields
                )
            ):
                raise MergeQueueIntegrityError(
                    "denial consumption registry row binding changed"
                )
            if (
                self.target_repository_id
                and (
                    record["target_repository_id"]
                    != self.target_repository_id
                    or record["target_branch"] != self.target_branch
                )
            ):
                continue
            verified.append(record)
        return tuple(verified)

    def _import_legacy_files(self) -> None:
        """Import legacy JSON queue files once, preserving their original stage."""

        stage_dirs = (
            ("pending", self.pending_dir),
            ("processing", self.processing_dir),
            ("completed", self.completed_dir),
            ("quarantined", self.failed_dir),
            ("quarantined", self.quarantine_dir),
            ("cancelled", self.cancelled_dir),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for status, directory in stage_dirs:
                    for path in directory.glob("*.json"):
                        try:
                            payload = json.loads(path.read_text(encoding="utf-8"))
                            request = MergeRequest.from_dict(payload, file_path=path)
                        except (OSError, json.JSONDecodeError, TypeError, ValueError):
                            continue
                        if not request.request_id:
                            continue
                        request = replace(request, status=status)
                        self._insert(connection, request, ignore=True)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _insert(
        self,
        connection: DuckDBConnection,
        request: MergeRequest,
        *,
        ignore: bool,
    ) -> None:
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        connection.execute(
            f"""{verb} INTO merge_requests (
                request_id, branch_name, task_id, priority, lane_id, enqueued_at,
                attempt, metadata_json, commit_sha, canonical_task_id,
                canonical_task_key, dedupe_key, status, claimed_at, consumer_id,
                failure_count, failure_reason, claim_token, claim_generation,
                finished_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request.request_id,
                request.branch_name,
                request.task_id,
                request.priority,
                request.lane_id,
                request.enqueued_at,
                request.attempt,
                json.dumps(request.metadata, sort_keys=True, separators=(",", ":"), default=str),
                request.commit_sha,
                request.canonical_task_id,
                request.canonical_task_key,
                request.dedupe_key or None,
                request.status,
                request.claimed_at,
                request.consumer_id,
                request.failure_count,
                request.failure_reason,
                request.claim_token,
                request.claim_generation,
                0.0,
                self._clock(),
            ),
        )

    def enqueue(
        self,
        *,
        branch_name: str,
        task_id: str,
        priority: str = "P2",
        lane_id: str = "",
        attempt: int = 1,
        metadata: dict[str, Any] | None = None,
        commit_sha: str = "",
        canonical_task_id: str = "",
        canonical_task_key: str = "",
        canonical_task_cid: str = "",
        target_repository_id: str = "",
        target_branch: str = "",
    ) -> MergeRequest:
        """Atomically enqueue or return the existing task-and-commit request."""

        if not str(branch_name).strip():
            raise ValueError("branch_name must not be empty")
        if not str(task_id).strip():
            raise ValueError("task_id must not be empty")
        metadata_dict = dict(metadata or {})
        declared_repository_id = str(
            target_repository_id
            or metadata_dict.get("target_repository_id")
            or ""
        ).strip()
        declared_branch = str(
            target_branch or metadata_dict.get("target_branch") or ""
        ).strip()
        if self.target_repository_id:
            if (
                declared_repository_id
                and declared_repository_id != self.target_repository_id
            ):
                raise ValueError(
                    "request target repository differs from the queue binding"
                )
            if declared_branch and declared_branch != self.target_branch:
                raise ValueError(
                    "request target branch differs from the queue binding"
                )
            declared_repository_id = self.target_repository_id
            declared_branch = self.target_branch
        if bool(declared_repository_id) != bool(declared_branch):
            raise ValueError(
                "request target_repository_id and target_branch must be "
                "supplied together"
            )
        if self.require_target_binding and not declared_repository_id:
            raise ValueError("bound merge queue refuses an unbound request")
        if declared_repository_id:
            supplied_schema = str(
                metadata_dict.get("target_binding_schema") or ""
            ).strip()
            if supplied_schema and supplied_schema != MERGE_TARGET_BINDING_SCHEMA:
                raise ValueError("request merge target binding schema changed")
            metadata_dict.update(
                {
                    "target_binding_schema": MERGE_TARGET_BINDING_SCHEMA,
                    "target_repository_id": declared_repository_id,
                    "target_branch": declared_branch,
                }
            )
        commit_sha = str(commit_sha or _first_metadata_value(metadata_dict, _COMMIT_METADATA_KEYS)).strip()
        canonical_task_key = str(
            canonical_task_key
            or _first_metadata_value(metadata_dict, ("canonical_task_key",))
        ).strip()
        canonical_task_id = str(
            canonical_task_id
            or canonical_task_cid
            or _first_metadata_value(metadata_dict, ("canonical_task_id", "canonical_task_cid", "task_cid"))
        ).strip()
        now = self._clock()
        request = MergeRequest(
            request_id=f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:12]}",
            branch_name=str(branch_name).strip(),
            task_id=str(task_id).strip(),
            priority=_normalise_priority(priority),
            lane_id=str(lane_id or os.getpid()),
            enqueued_at=now,
            attempt=max(1, int(attempt)),
            metadata=metadata_dict,
            commit_sha=commit_sha,
            canonical_task_id=canonical_task_id,
            canonical_task_key=canonical_task_key,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if request.dedupe_key:
                    row = connection.execute(
                        "SELECT * FROM merge_requests WHERE dedupe_key = ?",
                        (request.dedupe_key,),
                    ).fetchone()
                    if row is not None:
                        connection.commit()
                        return self._request_from_row(row)
                active_rows = connection.execute(
                    """SELECT metadata_json FROM merge_requests
                       WHERE status IN ('pending','processing')"""
                ).fetchall()
                active_count = sum(
                    self._metadata_matches_target(row["metadata_json"])
                    for row in active_rows
                )
                if active_count >= self.max_queue_size:
                    connection.rollback()
                    raise MergeQueueFullError(
                        f"merge queue capacity {self.max_queue_size} has been reached"
                    )
                self._insert(connection, request, ignore=False)
                connection.commit()
            except Exception:
                connection.rollback()
                if not request.dedupe_key:
                    raise
                row = connection.execute(
                    "SELECT * FROM merge_requests WHERE dedupe_key = ?", (request.dedupe_key,)
                ).fetchone()
                if row is None:
                    raise
                return self._request_from_row(row)
        receipt_path = self._write_stage_receipt(request)
        return replace(request, file_path=receipt_path)

    def _metadata_matches_target(self, value: Any) -> bool:
        """Return whether one durable row belongs to this consumer view."""

        if not self.target_repository_id:
            return not self.require_target_binding
        try:
            metadata = (
                json.loads(value or "{}")
                if not isinstance(value, Mapping)
                else value
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(metadata, Mapping):
            return False
        return bool(
            metadata.get("target_binding_schema")
            == MERGE_TARGET_BINDING_SCHEMA
            and str(metadata.get("target_repository_id") or "").strip()
            == self.target_repository_id
            and str(metadata.get("target_branch") or "").strip()
            == self.target_branch
        )

    def _require_row_target(
        self,
        row: DuckDBRow,
        *,
        operation: str,
        request_id: str,
    ) -> None:
        """Fence mutations attempted through a foreign bound queue view."""

        if not self._metadata_matches_target(row["metadata_json"]):
            raise MergeQueueFenceError(
                f"{operation} rejected for request {request_id}: "
                "request target differs from the queue binding"
            )

    def dequeue(self, consumer_id: str = "") -> Optional[MergeRequest]:
        """Atomically claim the fairest pending request for one consumer."""

        claimed = self.dequeue_many(1, consumer_id=consumer_id)
        return claimed[0] if claimed else None

    def dequeue_many(
        self,
        limit: int,
        consumer_id: str = "",
    ) -> tuple[MergeRequest, ...]:
        """Atomically claim a bounded, deterministically ordered preflight batch.

        ``max_processing`` is the merge-debt/backpressure fence.  Batch
        producers cannot reserve more worktrees or validation capacity than
        the configured number of in-flight requests, even when multiple
        processes race to claim work.
        """

        requested = int(limit)
        if requested <= 0:
            return ()
        self._purge_stale()
        consumer = str(consumer_id or os.getpid())
        now = self._clock()
        claimed_rows: list[DuckDBRow] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                processing_rows = connection.execute(
                    "SELECT metadata_json FROM merge_requests WHERE status='processing'"
                ).fetchall()
                if self.target_repository_id or self.require_target_binding:
                    processing_rows = [
                        row
                        for row in processing_rows
                        if self._metadata_matches_target(row["metadata_json"])
                    ]
                processing = len(processing_rows)
                capacity = max(0, self.max_processing - processing)
                claim_count = min(requested, capacity)
                if claim_count <= 0:
                    connection.commit()
                    return ()
                reserved_bytes = sum(
                    self._worktree_bytes_from_metadata_json(row["metadata_json"])
                    for row in processing_rows
                )
                observed_bytes = self._observed_worktree_bytes()
                worktree_bytes = max(reserved_bytes, observed_bytes)
                rows = connection.execute(
                    "SELECT * FROM merge_requests WHERE status = 'pending'"
                ).fetchall()
                if self.target_repository_id or self.require_target_binding:
                    rows = [
                        row
                        for row in rows
                        if self._metadata_matches_target(row["metadata_json"])
                    ]
                if not rows:
                    connection.commit()
                    return ()
                selected: list[DuckDBRow] = []
                for row in sorted(rows, key=lambda item: self._fairness_key(item, now)):
                    if len(selected) >= claim_count:
                        break
                    estimate = self._worktree_bytes_from_metadata_json(
                        row["metadata_json"]
                    )
                    if (
                        self.max_worktree_bytes is not None
                        and (
                            self.max_worktree_bytes <= 0
                            or worktree_bytes + estimate > self.max_worktree_bytes
                        )
                    ):
                        continue
                    selected.append(row)
                    worktree_bytes += estimate
                for row in selected:
                    claim_token = uuid.uuid4().hex
                    updated = connection.execute(
                        """UPDATE merge_requests
                           SET status='processing', claimed_at=?, consumer_id=?,
                               claim_token=?, claim_generation=claim_generation + 1,
                               updated_at=?
                           WHERE request_id=? AND status='pending'""",
                        (
                            now,
                            consumer,
                            claim_token,
                            now,
                            row["request_id"],
                        ),
                    )
                    if updated.rowcount != 1:
                        continue
                    claimed_row = connection.execute(
                        "SELECT * FROM merge_requests WHERE request_id=?",
                        (row["request_id"],),
                    ).fetchone()
                    if claimed_row is not None:
                        claimed_rows.append(claimed_row)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        claimed: list[MergeRequest] = []
        for row in claimed_rows:
            request = self._request_from_row(row)
            receipt_path = self._write_stage_receipt(request)
            claimed.append(replace(request, file_path=receipt_path))
        return tuple(claimed)

    def _worktree_bytes_from_metadata_json(self, value: Any) -> int:
        """Read a reservation estimate, conservatively bounding unknown work."""

        try:
            metadata = json.loads(value or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return self.max_worktree_bytes or 0
        if not isinstance(metadata, Mapping):
            return self.max_worktree_bytes or 0
        for key in (
            "worktree_bytes",
            "estimated_worktree_bytes",
            "worktree_disk_bytes",
        ):
            if key not in metadata:
                continue
            return max(0, _safe_int(metadata.get(key), 0))
        # Once a disk limit is requested, an unestimated worktree reserves the
        # whole budget.  This admits it serially without allowing missing
        # producer metadata to defeat the bound.
        return self.max_worktree_bytes or 0

    def _observed_worktree_bytes(self) -> int:
        """Return observed worktree use, failing closed when a configured probe fails."""

        if self._worktree_usage is None:
            return 0
        try:
            return max(0, int(self._worktree_usage()))
        except Exception:
            return self.max_worktree_bytes or 0

    def _fairness_key(self, row: DuckDBRow, now: float) -> tuple[int, float, str]:
        base = _PRIORITY_ORDER.get(str(row["priority"]), _PRIORITY_ORDER["P2"])
        if self.priority_aging_seconds > 0:
            promotions = int(max(0.0, now - float(row["enqueued_at"])) / self.priority_aging_seconds)
            effective = max(0, base - promotions)
        else:
            effective = base
        return effective, float(row["enqueued_at"]), str(row["request_id"])

    def _claim_matches(
        self,
        row: DuckDBRow,
        request: MergeRequest,
        *,
        consumer_id: str = "",
    ) -> bool:
        """Compare all durable claim coordinates, including ownership."""

        expected_consumer = str(consumer_id or request.consumer_id)
        claimed_at = _safe_float(
            row["claimed_at"] or row["enqueued_at"],
            0.0,
        )
        expired = (
            self.max_age_seconds > 0
            and self._clock() - claimed_at > self.max_age_seconds
        )
        return (
            str(row["status"]) == "processing"
            and not expired
            and bool(request.claim_token)
            and str(row["claim_token"] or "") == request.claim_token
            and int(row["claim_generation"] or 0) == request.claim_generation
            and str(row["consumer_id"] or "") == request.consumer_id
            and (not consumer_id or str(row["consumer_id"] or "") == expected_consumer)
        )

    def owns_claim(
        self,
        request: MergeRequest,
        *,
        consumer_id: str = "",
    ) -> bool:
        """Return whether ``request`` still owns the current processing fence.

        Merge workers should call this immediately before any target mutation.
        The subsequent terminal queue transition performs the same comparison
        atomically, so an expired, cancelled, or recovered claim fails closed.
        """

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request.request_id,),
            ).fetchone()
        return (
            row is not None
            and self._metadata_matches_target(row["metadata_json"])
            and self._claim_matches(row, request, consumer_id=consumer_id)
        )

    def _require_claim(
        self,
        row: DuckDBRow,
        request: MergeRequest,
        *,
        operation: str,
        allow_pending: bool = False,
    ) -> None:
        self._require_row_target(
            row,
            operation=operation,
            request_id=request.request_id,
        )
        status = str(row["status"])
        if allow_pending and status == "pending" and not request.claim_token:
            return
        if not self._claim_matches(row, request):
            raise MergeQueueFenceError(
                f"{operation} rejected for request {request.request_id}: "
                "claim token, generation, owner, or state is stale"
            )

    def complete(self, request: MergeRequest, metadata: Mapping[str, Any] | None = None) -> None:
        """Mark a claimed request complete; duplicate completion is harmless."""

        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return
            self._require_row_target(
                row,
                operation="complete",
                request_id=request.request_id,
            )
            if str(row["status"]) == "completed":
                connection.commit()
                return
            self._require_claim(row, request, operation="complete")
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if metadata:
                request_metadata["completion"] = dict(metadata)
            connection.execute(
                """UPDATE merge_requests SET status='completed', metadata_json=?,
                   finished_at=?, updated_at=?, consumer_id='', claimed_at=0,
                   claim_token='', claim_generation=claim_generation + 1
                   WHERE request_id=? AND status='processing'
                     AND claim_token=? AND claim_generation=? AND consumer_id=?""",
                (
                    json.dumps(request_metadata, sort_keys=True, separators=(",", ":")),
                    now,
                    now,
                    request.request_id,
                    request.claim_token,
                    request.claim_generation,
                    request.consumer_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        self._write_stage_receipt(self._request_from_row(row))
        self._prune_receipts(self.completed_dir, keep=50)

    def fail(
        self,
        request: MergeRequest,
        reason: str = "",
        *,
        retryable: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Record a failure, optionally retrying within the configured bound.

        Terminal failures and exhausted retries are placed in quarantine and
        return the durable receipt path.  A scheduled retry returns ``None``.
        """

        if retryable:
            result = self.requeue(request, reason=reason, metadata=metadata)
            return result if isinstance(result, Path) else None
        return self.quarantine(request, reason=reason, metadata=metadata)

    def requeue(
        self,
        request: MergeRequest,
        reason: str = "",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MergeRequest | Path | None:
        """Retry one request once, or quarantine it after ``max_attempts``."""

        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="requeue",
                request_id=request.request_id,
            )
            if str(row["status"]) in {"completed", "quarantined"}:
                connection.commit()
                resolved = self._request_from_row(row)
                if resolved.status == "quarantined":
                    return self._stage_path(resolved)
                return resolved
            self._require_claim(row, request, operation="requeue")
            next_attempt = max(int(row["attempt"]), int(row["failure_count"]) + 1) + 1
            failure_count = int(row["failure_count"]) + 1
            terminal = next_attempt > self.max_attempts
            status = "quarantined" if terminal else "pending"
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if metadata:
                request_metadata.setdefault("failure_metadata", []).append(dict(metadata))
            connection.execute(
                """UPDATE merge_requests SET status=?, attempt=?, failure_count=?,
                   failure_reason=?, metadata_json=?, claimed_at=0, consumer_id='',
                   claim_token='', claim_generation=claim_generation + 1,
                   finished_at=?, updated_at=? WHERE request_id=?
                     AND status='processing' AND claim_token=?
                     AND claim_generation=? AND consumer_id=?""",
                (
                    status,
                    next_attempt,
                    failure_count,
                    str(reason),
                    json.dumps(request_metadata, sort_keys=True, separators=(",", ":")),
                    now if terminal else 0.0,
                    now,
                    request.request_id,
                    request.claim_token,
                    request.claim_generation,
                    request.consumer_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        updated = self._request_from_row(row)
        path = self._write_stage_receipt(updated)
        return path if terminal else updated

    def quarantine(
        self,
        request: MergeRequest,
        reason: str = "",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Terminally quarantine one request and materialize its receipt."""

        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="quarantine",
                request_id=request.request_id,
            )
            if str(row["status"]) == "quarantined":
                connection.commit()
                return self._stage_path(self._request_from_row(row))
            self._require_claim(
                row,
                request,
                operation="quarantine",
                allow_pending=True,
            )
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if metadata:
                request_metadata["quarantine"] = dict(metadata)
            connection.execute(
                """UPDATE merge_requests SET status='quarantined', failure_count=?,
                   failure_reason=?, metadata_json=?, claimed_at=0, consumer_id='',
                   claim_token='', claim_generation=claim_generation + 1,
                   finished_at=?, updated_at=? WHERE request_id=?""",
                (
                    max(1, int(row["failure_count"])),
                    str(reason or row["failure_reason"]),
                    json.dumps(request_metadata, sort_keys=True, separators=(",", ":")),
                    now,
                    now,
                    request.request_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._write_stage_receipt(self._request_from_row(row))

    def cancel(
        self,
        request: MergeRequest | str,
        reason: str = "cancelled",
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MergeRequest | None:
        """Durably cancel pending work or an exactly fenced processing claim.

        A request id is sufficient for work which has not been claimed.  Once
        processing begins, callers must pass the exact :class:`MergeRequest`
        returned by ``dequeue``; this prevents an operator or stale worker from
        cancelling a newer owner's claim accidentally.
        """

        supplied = request if isinstance(request, MergeRequest) else None
        request_id = supplied.request_id if supplied is not None else str(request)
        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="cancel",
                request_id=request_id,
            )
            status = str(row["status"])
            if status == "cancelled":
                connection.commit()
                return self._request_from_row(row)
            if status in {"completed", "quarantined"}:
                connection.commit()
                return self._request_from_row(row)
            if status == "processing":
                if supplied is None:
                    connection.rollback()
                    raise MergeQueueFenceError(
                        f"cancel rejected for request {request_id}: "
                        "a processing request requires its current claim"
                    )
                self._require_claim(row, supplied, operation="cancel")
            request_metadata = json.loads(row["metadata_json"] or "{}")
            cancellation = {"at": now, "reason": str(reason or "cancelled")}
            if metadata:
                cancellation["metadata"] = dict(metadata)
            request_metadata["cancellation"] = cancellation
            connection.execute(
                """UPDATE merge_requests SET status='cancelled', failure_reason=?,
                   metadata_json=?, claimed_at=0, consumer_id='', claim_token='',
                   claim_generation=claim_generation + 1, finished_at=?, updated_at=?
                   WHERE request_id=? AND status IN ('pending','processing')""",
                (
                    str(reason or "cancelled"),
                    json.dumps(
                        request_metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    now,
                    now,
                    request_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            connection.commit()
        assert row is not None
        cancelled = self._request_from_row(row)
        receipt_path = self._write_stage_receipt(cancelled)
        return replace(cancelled, file_path=receipt_path)

    @staticmethod
    def _normalized_submodule_recovery_targets(
        request_metadata: Mapping[str, Any],
        approved_submodule_integrations: Mapping[str, str],
    ) -> list[dict[str, str]]:
        """Return exact task-bound child postimages or reject the whole grant."""

        raw_task_binding = request_metadata.get(
            "task_owned_submodule_integration_binding"
        )
        raw_targets = (
            raw_task_binding.get("targets")
            if isinstance(raw_task_binding, Mapping)
            else None
        )
        bound_paths = {
            str(target.get("path") or "").strip("/")
            for target in raw_targets or ()
            if isinstance(target, Mapping)
        }
        if (
            not isinstance(approved_submodule_integrations, Mapping)
            or not approved_submodule_integrations
            or len(approved_submodule_integrations) > 256
        ):
            raise ValueError(
                "approved submodule integrations must be a non-empty mapping"
            )
        normalized_targets: list[dict[str, str]] = []
        for raw_path, raw_commit in sorted(
            approved_submodule_integrations.items(),
            key=lambda item: str(item[0]),
        ):
            path = str(raw_path or "").strip("/")
            commit = str(raw_commit or "").strip()
            path_parts = Path(path).parts
            if (
                not path
                or path != str(raw_path or "")
                or path not in bound_paths
                or Path(path).is_absolute()
                or any(part in {"", ".", ".."} for part in path_parts)
            ):
                raise ValueError(
                    f"submodule recovery path is not task-bound: {raw_path!r}"
                )
            if (
                len(commit) not in {40, 64}
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in commit
                )
            ):
                raise ValueError(
                    f"submodule recovery commit is invalid for {path}"
                )
            normalized_targets.append(
                {
                    "path": path,
                    "integrated_target": commit.lower(),
                }
            )
        return normalized_targets

    def revive_quarantined(
        self,
        request: MergeRequest | str,
        reason: str = "",
        *,
        reset_failures: bool = False,
        approved_submodule_integrations: Mapping[str, str] | None = None,
    ) -> MergeRequest | None:
        """Return a quarantined request to pending after operator review.

        The operation is atomic and idempotent.  A revival record is retained
        in request metadata so administrative recovery does not erase why the
        candidate was quarantined.  ``reset_failures`` is intended for false
        positives such as a host suspension while a request was still pending.
        ``approved_submodule_integrations`` binds recovery to exact reviewed
        child postimages; it never authorizes an arbitrary descendant.
        """

        request_id = request.request_id if isinstance(request, MergeRequest) else str(request)
        now = self._clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            self._require_row_target(
                row,
                operation="revive",
                request_id=request_id,
            )
            status = str(row["status"])
            request_metadata = json.loads(row["metadata_json"] or "{}")
            if status != "quarantined":
                if (
                    status in _ACTIVE_STATES
                    and approved_submodule_integrations is not None
                ):
                    if not str(reason or "").strip():
                        connection.rollback()
                        raise ValueError(
                            "submodule integration recovery requires an operator reason"
                        )
                    normalized_targets = (
                        self._normalized_submodule_recovery_targets(
                            request_metadata,
                            approved_submodule_integrations,
                        )
                    )
                    existing = request_metadata.get(
                        "operator_submodule_integration_recovery"
                    )
                    row_generation = int(row["claim_generation"] or 0)
                    existing_generation = (
                        existing.get(
                            "revival_generation"
                            if status == "pending"
                            else "claim_generation"
                        )
                        if isinstance(existing, Mapping)
                        else None
                    )
                    same_active_grant = bool(
                        isinstance(existing, Mapping)
                        and existing.get("schema")
                        == SUBMODULE_INTEGRATION_RECOVERY_SCHEMA
                        and existing.get("request_id") == request_id
                        and existing.get("implementation_commit")
                        == str(row["commit_sha"] or "")
                        and existing.get("target_repository_id")
                        == str(
                            request_metadata.get("target_repository_id")
                            or ""
                        ).strip()
                        and existing.get("target_branch")
                        == str(
                            request_metadata.get("target_branch") or ""
                        ).strip()
                        and existing.get("targets") == normalized_targets
                        and not isinstance(existing_generation, bool)
                        and isinstance(existing_generation, int)
                        and existing_generation == row_generation
                    )
                    if not same_active_grant:
                        connection.rollback()
                        raise MergeQueueFenceError(
                            f"revive rejected for active request {request_id}: "
                            "submodule recovery approval differs from the "
                            "current generation-bound grant"
                        )
                connection.commit()
                return self._request_from_row(row)

            # Recovery is a single-revival capability. Preserve its audit copy
            # in ``revivals`` but never carry the live top-level grant into a
            # later quarantine/revival cycle.
            request_metadata.pop(
                "operator_submodule_integration_recovery",
                None,
            )
            recovery_binding: dict[str, Any] | None = None
            if approved_submodule_integrations is not None:
                if not str(reason or "").strip():
                    connection.rollback()
                    raise ValueError(
                        "submodule integration recovery requires an operator reason"
                    )
                try:
                    normalized_targets = (
                        self._normalized_submodule_recovery_targets(
                            request_metadata,
                            approved_submodule_integrations,
                        )
                    )
                except ValueError:
                    connection.rollback()
                    raise
                quarantine_generation = int(row["claim_generation"] or 0)
                revival_generation = quarantine_generation + 1
                recovery_binding = {
                    "schema": SUBMODULE_INTEGRATION_RECOVERY_SCHEMA,
                    "approved_at": now,
                    "reason": str(reason).strip(),
                    "request_id": request_id,
                    "implementation_commit": str(row["commit_sha"] or ""),
                    "target_repository_id": str(
                        request_metadata.get("target_repository_id") or ""
                    ).strip(),
                    "target_branch": str(
                        request_metadata.get("target_branch") or ""
                    ).strip(),
                    "quarantine_generation": quarantine_generation,
                    "revival_generation": revival_generation,
                    "claim_generation": revival_generation + 1,
                    "targets": normalized_targets,
                }
                request_metadata[
                    "operator_submodule_integration_recovery"
                ] = recovery_binding
            revival = {
                "at": now,
                "reason": str(reason),
                "previous_enqueued_at": float(row["enqueued_at"]),
                "previous_failure_count": int(row["failure_count"]),
                "previous_failure_reason": str(row["failure_reason"]),
            }
            if recovery_binding is not None:
                revival["submodule_integration_recovery"] = recovery_binding
            request_metadata.setdefault("revivals", []).append(revival)
            failure_count = 0 if reset_failures else int(row["failure_count"])
            attempt = 1 if reset_failures else int(row["attempt"])
            connection.execute(
                """UPDATE merge_requests SET status='pending', enqueued_at=?, attempt=?,
                   failure_count=?, failure_reason='', metadata_json=?, claimed_at=0,
                   consumer_id='', claim_token='',
                   claim_generation=claim_generation + 1,
                   finished_at=0, updated_at=? WHERE request_id=?""",
                (
                    now,
                    attempt,
                    failure_count,
                    json.dumps(request_metadata, sort_keys=True, separators=(",", ":")),
                    now,
                    request_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        revived = self._request_from_row(row)
        receipt_path = self._write_stage_receipt(revived)
        return replace(revived, file_path=receipt_path)

    def get(self, request_id: str) -> MergeRequest | None:
        """Return the current durable request by id."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM merge_requests WHERE request_id=?", (request_id,)
            ).fetchone()
        return self._request_from_row(row) if row is not None else None

    def active_canonical_task_ids(self) -> set[str]:
        """Return content identities currently waiting for merge or being merged."""

        return self._canonical_task_ids_for_statuses(_ACTIVE_STATES)

    def completed_canonical_task_ids(
        self,
        *,
        candidate_is_denied: Callable[[MergeRequest], bool] | None = None,
        candidate_is_eligible: Callable[[MergeRequest], bool] | None = None,
    ) -> set[str]:
        """Return identities having at least one eligible merge receipt.

        A completed queue row is candidate-specific: a later independent
        review can permanently deny that immutable implementation commit
        without denying every future implementation of the same task.  The
        optional predicate lets the caller apply its verified, lane-bound
        denial authority before this method projects candidate rows to task
        identities.  ``candidate_is_eligible`` may additionally require
        positive causal evidence (for example, a strictly later
        implementation attempt with the same task binding).  Existential
        projection is intentional: one eligible completed candidate restores
        the task identity.

        Predicate failures propagate.  A caller must never interpret
        unavailable denial or eligibility decisions as an empty denial set.
        """

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM merge_requests WHERE status='completed'"
            ).fetchall()
        completed: set[str] = set()
        for row in rows:
            if not self._metadata_matches_target(row["metadata_json"]):
                continue
            request = self._request_from_row(row)
            if candidate_is_denied is not None:
                denied = candidate_is_denied(request)
                if type(denied) is not bool:
                    raise TypeError(
                        "candidate_is_denied must return an exact bool"
                    )
                if denied:
                    continue
            if candidate_is_eligible is not None:
                eligible = candidate_is_eligible(request)
                if type(eligible) is not bool:
                    raise TypeError(
                        "candidate_is_eligible must return an exact bool"
                    )
                if not eligible:
                    continue
            canonical_task_id = str(request.canonical_task_id)
            if canonical_task_id:
                completed.add(canonical_task_id)
        return completed

    def _canonical_task_ids_for_statuses(self, statuses: tuple[str, ...]) -> set[str]:
        normalized = tuple(
            dict.fromkeys(
                str(status).strip() for status in statuses if str(status).strip()
            )
        )
        if not normalized:
            return set()
        placeholders = ",".join("?" for _ in normalized)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT canonical_task_id, metadata_json
                    FROM merge_requests
                    WHERE status IN ({placeholders}) AND canonical_task_id != ''""",
                normalized,
            ).fetchall()
        return {
            str(row["canonical_task_id"])
            for row in rows
            if self._metadata_matches_target(row["metadata_json"])
        }

    def pending_count(self) -> int:
        return self._count("pending")

    def processing_count(self) -> int:
        return self._count("processing")

    def _count(self, status: str) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT metadata_json FROM merge_requests WHERE status=?",
                (status,),
            ).fetchall()
        return sum(
            self._metadata_matches_target(row["metadata_json"])
            for row in rows
        )

    def has_pending_for_task(
        self,
        task_id: str,
        *,
        commit_sha: str | None = None,
    ) -> bool:
        """Return whether a task (and optionally commit) is active."""

        identity = str(task_id).strip().casefold()
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT task_id, canonical_task_id, canonical_task_key,
                          commit_sha, metadata_json
                   FROM merge_requests WHERE status IN ('pending','processing')"""
            ).fetchall()
        for row in rows:
            if not self._metadata_matches_target(row["metadata_json"]):
                continue
            identities = {
                str(row["task_id"]).casefold(),
                str(row["canonical_task_id"]).casefold(),
                str(row["canonical_task_key"]).casefold(),
            }
            if identity not in identities:
                continue
            if commit_sha is None or str(row["commit_sha"]).casefold() == str(commit_sha).casefold():
                return True
        return False

    def _purge_stale(self) -> int:
        """Recover abandoned consumer claims that exceeded their lease bound.

        Pending requests have no consumer lease and therefore do not expire.
        Queue capacity and explicit cancellation bound their lifetime.  This
        distinction also keeps a suspended host from quarantining valid work.
        """

        if self.max_age_seconds <= 0:
            return 0
        now = self._clock()
        changed: list[MergeRequest] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM merge_requests WHERE status='processing'"
            ).fetchall()
            for row in rows:
                if not self._metadata_matches_target(row["metadata_json"]):
                    continue
                reference_time = float(row["claimed_at"] or row["enqueued_at"])
                if now - reference_time <= self.max_age_seconds:
                    continue
                attempt = int(row["attempt"])
                failure_count = int(row["failure_count"])
                if attempt < self.max_attempts:
                    new_status = "pending"
                    new_attempt = attempt + 1
                    failure_count += 1
                    reason = "consumer claim expired; request recovered"
                    finished_at = 0.0
                else:
                    new_status = "quarantined"
                    new_attempt = attempt
                    failure_count = max(1, failure_count)
                    reason = "processing request exceeded max age"
                    finished_at = now
                connection.execute(
                    """UPDATE merge_requests SET status=?, attempt=?, failure_count=?,
                       failure_reason=?, claimed_at=0, consumer_id='', claim_token='',
                       claim_generation=claim_generation + 1, finished_at=?,
                       updated_at=? WHERE request_id=?""",
                    (
                        new_status,
                        new_attempt,
                        failure_count,
                        reason,
                        finished_at,
                        now,
                        row["request_id"],
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM merge_requests WHERE request_id=?", (row["request_id"],)
                ).fetchone()
                if updated is not None:
                    changed.append(self._request_from_row(updated))
            connection.commit()
        for request in changed:
            self._write_stage_receipt(request)
        return len(changed)

    def recover_abandoned_train_claims(self) -> int:
        """Recover claims left by a crashed process-safe merge train.

        Callers must hold the merge train's repo-wide consumer lock. Once that
        lock is acquired, no live ``merge-train:*`` consumer can still own a
        processing row, so waiting for the general queue age timeout only
        wastes throughput. Claims from other queue consumers are untouched.
        """

        now = self._clock()
        changed: list[MergeRequest] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM merge_requests WHERE status='processing' AND consumer_id LIKE 'merge-train:%'"
            ).fetchall()
            for row in rows:
                if not self._metadata_matches_target(row["metadata_json"]):
                    continue
                attempt = int(row["attempt"])
                failure_count = int(row["failure_count"]) + 1
                if attempt < self.max_attempts:
                    status = "pending"
                    next_attempt = attempt + 1
                    finished_at = 0.0
                    reason = "merge train consumer exited; claim recovered"
                else:
                    status = "quarantined"
                    next_attempt = attempt
                    finished_at = now
                    reason = "merge train consumer exited on final attempt"
                connection.execute(
                    """UPDATE merge_requests SET status=?, attempt=?, failure_count=?,
                       failure_reason=?, claimed_at=0, consumer_id='', claim_token='',
                       claim_generation=claim_generation + 1, finished_at=?,
                       updated_at=? WHERE request_id=? AND status='processing'""",
                    (
                        status,
                        next_attempt,
                        failure_count,
                        reason,
                        finished_at,
                        now,
                        row["request_id"],
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM merge_requests WHERE request_id=?", (row["request_id"],)
                ).fetchone()
                if updated is not None:
                    changed.append(self._request_from_row(updated))
            connection.commit()
        for request in changed:
            self._write_stage_receipt(request)
        return len(changed)

    def status(self) -> dict[str, Any]:
        """Return an authoritative stage summary suitable for daemon status."""

        with self._connect() as connection:
            stage_rows = connection.execute(
                """SELECT status, enqueued_at, finished_at, metadata_json
                   FROM merge_requests"""
            ).fetchall()
            stage_rows = [
                row
                for row in stage_rows
                if self._metadata_matches_target(row["metadata_json"])
            ]
            counts: dict[str, int] = {}
            for row in stage_rows:
                status = str(row["status"])
                counts[status] = counts.get(status, 0) + 1
            timing_rows = connection.execute(
                """SELECT enqueued_at, finished_at, metadata_json
                   FROM merge_requests
                   WHERE status='completed' AND finished_at > 0
                   ORDER BY finished_at"""
            ).fetchall()
            timing_rows = [
                row
                for row in timing_rows
                if self._metadata_matches_target(row["metadata_json"])
            ]
            processing_rows = connection.execute(
                "SELECT metadata_json FROM merge_requests WHERE status='processing'"
            ).fetchall()
            processing_rows = [
                row
                for row in processing_rows
                if self._metadata_matches_target(row["metadata_json"])
            ]
            pending_rows = connection.execute(
                "SELECT metadata_json FROM merge_requests WHERE status='pending'"
            ).fetchall()
            pending_rows = [
                row
                for row in pending_rows
                if self._metadata_matches_target(row["metadata_json"])
            ]
        completed_span = (
            max(
                0.0,
                float(timing_rows[-1]["finished_at"])
                - float(timing_rows[0]["enqueued_at"]),
            )
            if timing_rows
            else 0.0
        )
        active = counts.get("pending", 0) + counts.get("processing", 0)
        merge_debt = counts.get("processing", 0)
        reserved_worktree_bytes = sum(
            self._worktree_bytes_from_metadata_json(row["metadata_json"])
            for row in processing_rows
        )
        observed_worktree_bytes = self._observed_worktree_bytes()
        worktree_bytes_in_use = max(
            reserved_worktree_bytes,
            observed_worktree_bytes,
        )
        disk_backpressure = (
            self.max_worktree_bytes is not None
            and (
                worktree_bytes_in_use >= self.max_worktree_bytes
                or any(
                    worktree_bytes_in_use
                    + self._worktree_bytes_from_metadata_json(row["metadata_json"])
                    > self.max_worktree_bytes
                    for row in pending_rows
                )
            )
        )
        return {
            "pending": counts.get("pending", 0),
            "processing": merge_debt,
            "completed": counts.get("completed", 0),
            "failed": counts.get("quarantined", 0),
            "quarantined": counts.get("quarantined", 0),
            "cancelled": counts.get("cancelled", 0),
            "total": sum(counts.values()),
            "queue_dir": str(self.queue_dir),
            "database_path": str(self.database_path),
            "target_repository_id": self.target_repository_id,
            "target_branch": self.target_branch,
            "target_binding_required": self.require_target_binding,
            "max_attempts": self.max_attempts,
            "max_queue_size": self.max_queue_size,
            "max_processing": self.max_processing,
            "merge_debt": merge_debt,
            "max_worktree_bytes": self.max_worktree_bytes,
            "reserved_worktree_bytes": reserved_worktree_bytes,
            "observed_worktree_bytes": observed_worktree_bytes,
            "worktree_bytes_in_use": worktree_bytes_in_use,
            "disk_backpressure": disk_backpressure,
            "backpressure": (
                active >= self.max_queue_size
                or merge_debt >= self.max_processing
                or disk_backpressure
            ),
            "throughput": {
                "schema": MERGE_QUEUE_THROUGHPUT_SCHEMA,
                "lane": "merge-queue-persistence",
                "accepted_count": len(timing_rows),
                "elapsed_seconds": completed_span,
                "accepted_per_second": (
                    len(timing_rows) / completed_span
                    if completed_span > 0
                    else 0.0
                ),
            },
        }

    def _request_from_row(self, row: DuckDBRow) -> MergeRequest:
        status = str(row["status"])
        payload = {
            "request_id": row["request_id"],
            "branch_name": row["branch_name"],
            "task_id": row["task_id"],
            "priority": row["priority"],
            "lane_id": row["lane_id"],
            "enqueued_at": row["enqueued_at"],
            "attempt": row["attempt"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "commit_sha": row["commit_sha"],
            "canonical_task_id": row["canonical_task_id"],
            "canonical_task_key": row["canonical_task_key"],
            "status": status,
            "claimed_at": row["claimed_at"],
            "consumer_id": row["consumer_id"],
            "failure_count": row["failure_count"],
            "failure_reason": row["failure_reason"],
            "claim_token": row["claim_token"],
            "claim_generation": row["claim_generation"],
        }
        request = MergeRequest.from_dict(payload)
        return replace(request, file_path=self._stage_path(request))

    def _stage_path(self, request: MergeRequest) -> Path:
        stage_dir = {
            "pending": self.pending_dir,
            "processing": self.processing_dir,
            "completed": self.completed_dir,
            "quarantined": self.quarantine_dir,
            "cancelled": self.cancelled_dir,
        }.get(request.status, self.failed_dir)
        return stage_dir / f"{request.request_id}.json"

    def _write_stage_receipt(self, request: MergeRequest) -> Path:
        destination = self._stage_path(request)
        payload = request.to_dict()
        if request.status == "quarantined":
            payload.update(
                {
                    "receipt_type": "merge_quarantine",
                    "quarantined_at": self._clock(),
                    "receipt_id": hashlib.sha256(
                        f"{request.request_id}\0{request.failure_reason}".encode("utf-8")
                    ).hexdigest(),
                }
            )
        elif request.status == "cancelled":
            payload.update(
                {
                    "receipt_type": "merge_cancellation",
                    "cancelled_at": self._clock(),
                    "receipt_id": hashlib.sha256(
                        (
                            f"{request.request_id}\0{request.failure_reason}"
                            f"\0{request.claim_generation}"
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
        _atomic_write_json(destination, payload)
        for directory in (
            self.pending_dir,
            self.processing_dir,
            self.completed_dir,
            self.failed_dir,
            self.quarantine_dir,
            self.cancelled_dir,
        ):
            candidate = directory / destination.name
            if candidate == destination:
                continue
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        return destination

    @staticmethod
    def _prune_receipts(directory: Path, *, keep: int) -> None:
        paths = sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime)
        for path in paths[:-keep]:
            try:
                path.unlink()
            except OSError:
                pass


__all__ = [
    "MergeQueue",
    "MergeQueueFullError",
    "MergeQueueFenceError",
    "MergeQueueIntegrityError",
    "MergeRequest",
    "POST_MERGE_CORRECTION_AUTHORITY_STATE_SCHEMA",
    "POST_MERGE_CORRECTION_CHAIN_HEAD_SCHEMA",
    "POST_MERGE_CORRECTION_CHAIN_RECORD_SCHEMA",
    "POST_MERGE_CORRECTION_CONSUMPTION_SCHEMA",
    "POST_MERGE_CORRECTION_FAILURE_SCHEMA",
    "POST_MERGE_CORRECTION_LEGACY_FAILURE_ANCHOR_SCHEMA",
    "POST_MERGE_CORRECTION_LEGACY_HIGH_WATER_ANCHOR_SCHEMA",
    "POST_MERGE_CORRECTION_REPAIR_GRANT_SCHEMA",
    "POST_MERGE_REVIEW_DENIAL_TOMBSTONE_SCHEMA",
    "_PRIORITY_ORDER",
]
