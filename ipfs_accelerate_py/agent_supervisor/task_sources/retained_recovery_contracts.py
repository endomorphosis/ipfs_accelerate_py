"""Closed evidence contracts for lane-local retained occurrence recovery.

The historical attempt, claim, and lease records used by the retained PCTDD
recovery live in the lane execution/coordinator stores.  They are not rows in
the normalized control-plane store.  These records keep that distinction
explicit: the lane receipt commits the historical terminal row, while a
controller-issued receipt proves that the managed daemon/provider process
tree was quiesced under the owner and launch fences.

The self hashes below are integrity commitments.  Authentication is supplied
only when the controller binds the quiescence receipt into its authenticated
Quack transaction; neither record is standalone recovery authority.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, Final

from .control_plane_contracts import canonical_json_bytes


DATABASE_PORTAL_CONTROLLER_QUIESCENCE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-controller-quiescence-receipt@1"
)
DATABASE_FENCED_PROVIDER_HISTORICAL_OCCURRENCE_AUTHORITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-fenced-provider-historical-occurrence-authority@1"
)

_QUIESCENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "board_namespace",
        "state_prefix",
        "owner_store_id",
        "control_store_generation",
        "trigger",
        "controller_process_birth",
        "managed_daemon_pid",
        "managed_daemon_identity_record_id",
        "managed_daemon_process_birth",
        "daemon_fence_reason",
        "daemon_fence_digest",
        "daemon_safe_to_restart",
        "provider_runner_fence_reason",
        "provider_runner_fence_digest",
        "provider_runner_safe_to_restart",
        "remaining_pid",
        "markers_removed",
        "quiesced",
        "owner_mutation_fence_held",
        "managed_daemon_launch_lock_held",
        "receipt_id",
    }
)
_HISTORICAL_AUTHORITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "board_namespace",
        "owner_store_id",
        "control_store_generation",
        "receipt_nonce",
        "receipt_epoch",
        "subject",
        "inner_receipt_cid",
        "terminal_reconciliation",
        "controller_quiescence_receipt_id",
        "authority_id",
    }
)
_HISTORICAL_SUBJECT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "task_cid",
        "task_alias",
        "task_revision",
        "attempt_id",
        "claim_id",
        "lease_id",
        "attempt_number",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "recovery_manifest_id",
        "recovery_credit_id",
    }
)
_TERMINAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "attempt_id",
        "task_cid",
        "claim_id",
        "attempt_number",
        "owner_session_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
        "intended_database_disposition",
        "evidence_id",
        "prepared_reconciliation_receipt_id",
        "commit_barrier_receipt_id",
        "stage",
        "receipt_id",
        "record_json",
    }
)
_PROCESS_BIRTH_FIELDS: Final[frozenset[str]] = frozenset(
    {"pid", "start_time_ticks", "boot_id", "parent_pid"}
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_CONTENT_CID = re.compile(r"baguqeera[a-z2-7]{52}")


def retained_recovery_sha256(value: Any) -> str:
    """Return the closed canonical SHA-256 identity used by these records."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: Any, *, maximum: int = 4096) -> bool:
    return bool(
        type(value) is str
        and value
        and len(value.encode("utf-8")) <= maximum
        and not any(character in value for character in "\0\n\r")
    )


def _process_birth_valid(value: Any, *, required: bool) -> bool:
    if value is None:
        return not required
    return bool(
        isinstance(value, Mapping)
        and set(value) == _PROCESS_BIRTH_FIELDS
        and type(value.get("pid")) is int
        and value["pid"] >= 1
        and type(value.get("start_time_ticks")) is int
        and value["start_time_ticks"] >= 1
        and _text(value.get("boot_id"), maximum=256)
        and type(value.get("parent_pid")) is int
        and value["parent_pid"] >= 0
    )


def database_portal_controller_quiescence_receipt_valid(value: Any) -> bool:
    """Validate a compact controller observation without trusting its hash."""

    if not isinstance(value, Mapping) or set(value) != _QUIESCENCE_FIELDS:
        return False
    record = dict(value)
    unsigned = dict(record)
    receipt_id = unsigned.pop("receipt_id", "")
    daemon_pid = record.get("managed_daemon_pid")
    identity_id = record.get("managed_daemon_identity_record_id")
    daemon_birth = record.get("managed_daemon_process_birth")
    exact_recorded_identity = bool(
        type(daemon_pid) is int
        and daemon_pid >= 1
        and _text(identity_id)
        and _process_birth_valid(daemon_birth, required=True)
        and daemon_birth["pid"] == daemon_pid
    )
    exact_absence = bool(
        daemon_pid is None
        and identity_id == ""
        and daemon_birth is None
        and record.get("daemon_fence_reason")
        == "managed_daemon_not_recorded"
    )
    return bool(
        record.get("schema")
        == DATABASE_PORTAL_CONTROLLER_QUIESCENCE_RECEIPT_SCHEMA
        and all(
            _text(record.get(name))
            for name in (
                "board_namespace",
                "state_prefix",
                "owner_store_id",
                "control_store_generation",
                "trigger",
                "daemon_fence_reason",
                "provider_runner_fence_reason",
            )
        )
        and _process_birth_valid(
            record.get("controller_process_birth"), required=True
        )
        and (exact_recorded_identity or exact_absence)
        and _SHA256.fullmatch(str(record.get("daemon_fence_digest") or ""))
        is not None
        and _SHA256.fullmatch(
            str(record.get("provider_runner_fence_digest") or "")
        )
        is not None
        and record.get("daemon_safe_to_restart") is True
        and record.get("provider_runner_safe_to_restart") is True
        and record.get("remaining_pid") is None
        and record.get("markers_removed") is True
        and record.get("quiesced") is True
        and record.get("owner_mutation_fence_held") is True
        and record.get("managed_daemon_launch_lock_held") is True
        and _SHA256.fullmatch(str(receipt_id or "")) is not None
        and retained_recovery_sha256(unsigned) == receipt_id
    )


def database_portal_controller_quiescence_receipt(
    *,
    cleanup: Mapping[str, Any],
    board_namespace: str,
    state_prefix: str,
    owner_store_id: str,
    control_store_generation: str,
    trigger: str,
    controller_process_birth: Mapping[str, Any],
    owner_mutation_fence_held: bool,
    managed_daemon_launch_lock_held: bool,
) -> Mapping[str, Any]:
    """Normalize one live controller cleanup result into a closed receipt."""

    if not isinstance(cleanup, Mapping):
        raise ValueError("controller quiescence requires a cleanup mapping")
    daemon_fence = cleanup.get("daemon_fence")
    provider_fence = cleanup.get("provider_runner_fence")
    if not isinstance(daemon_fence, Mapping) or not isinstance(
        provider_fence, Mapping
    ):
        raise ValueError("controller quiescence lacks its process fences")
    identity_id = str(cleanup.get("managed_daemon_identity_record_id") or "")
    daemon_birth = cleanup.get("managed_daemon_process_birth")
    daemon_pid = cleanup.get("pid")
    record: dict[str, Any] = {
        "schema": DATABASE_PORTAL_CONTROLLER_QUIESCENCE_RECEIPT_SCHEMA,
        "board_namespace": str(board_namespace),
        "state_prefix": str(state_prefix),
        "owner_store_id": str(owner_store_id),
        "control_store_generation": str(control_store_generation),
        "trigger": str(trigger),
        "controller_process_birth": dict(controller_process_birth),
        "managed_daemon_pid": daemon_pid,
        "managed_daemon_identity_record_id": identity_id,
        "managed_daemon_process_birth": (
            dict(daemon_birth) if isinstance(daemon_birth, Mapping) else None
        ),
        "daemon_fence_reason": str(daemon_fence.get("reason") or ""),
        "daemon_fence_digest": retained_recovery_sha256(dict(daemon_fence)),
        "daemon_safe_to_restart": daemon_fence.get("safe_to_restart"),
        "provider_runner_fence_reason": str(
            provider_fence.get("reason") or ""
        ),
        "provider_runner_fence_digest": retained_recovery_sha256(
            dict(provider_fence)
        ),
        "provider_runner_safe_to_restart": provider_fence.get(
            "safe_to_restart"
        ),
        "remaining_pid": cleanup.get("remaining_pid"),
        "markers_removed": cleanup.get("markers_removed"),
        "quiesced": cleanup.get("quiesced"),
        "owner_mutation_fence_held": owner_mutation_fence_held,
        "managed_daemon_launch_lock_held": (
            managed_daemon_launch_lock_held
        ),
    }
    record["receipt_id"] = retained_recovery_sha256(record)
    if not database_portal_controller_quiescence_receipt_valid(record):
        raise ValueError("controller quiescence receipt failed closed validation")
    return record


def database_fenced_provider_historical_occurrence_authority_valid(
    value: Any,
) -> bool:
    """Validate the compact lane-local/terminal authority projection."""

    if not isinstance(value, Mapping) or set(value) != _HISTORICAL_AUTHORITY_FIELDS:
        return False
    record = dict(value)
    subject = record.get("subject")
    terminal = record.get("terminal_reconciliation")
    if (
        not isinstance(subject, Mapping)
        or set(subject) != _HISTORICAL_SUBJECT_FIELDS
        or not isinstance(terminal, Mapping)
        or set(terminal) != _TERMINAL_FIELDS
    ):
        return False
    text_subject = (
        "task_cid",
        "task_alias",
        "attempt_id",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "recovery_manifest_id",
        "recovery_credit_id",
    )
    integer_subject = (
        "task_revision",
        "attempt_number",
        "fencing_token",
        "fence_epoch",
    )
    terminal_identity = {
        "attempt_id": subject.get("attempt_id"),
        "task_cid": subject.get("task_cid"),
        "claim_id": subject.get("claim_id"),
        "attempt_number": subject.get("attempt_number"),
        "owner_session_id": subject.get("owner_session_id"),
        "lease_id": subject.get("lease_id"),
        "fencing_token": subject.get("fencing_token"),
        "fence_epoch": subject.get("fence_epoch"),
    }
    record_json = terminal.get("record_json")
    unsigned = dict(record)
    authority_id = unsigned.pop("authority_id", "")
    return bool(
        record.get("schema")
        == DATABASE_FENCED_PROVIDER_HISTORICAL_OCCURRENCE_AUTHORITY_SCHEMA
        and all(
            _text(record.get(name))
            for name in (
                "board_namespace",
                "owner_store_id",
                "control_store_generation",
                "receipt_nonce",
            )
        )
        and type(record.get("receipt_epoch")) is int
        and record["receipt_epoch"] >= 1
        and all(_text(subject.get(name)) for name in text_subject)
        and all(
            type(subject.get(name)) is int
            and subject[name] >= (1 if name == "attempt_number" else 0)
            for name in integer_subject
        )
        and _SHA256.fullmatch(str(record.get("inner_receipt_cid") or ""))
        is not None
        and _SHA256.fullmatch(
            str(record.get("controller_quiescence_receipt_id") or "")
        )
        is not None
        and all(terminal.get(name) == expected for name, expected in terminal_identity.items())
        and _text(terminal.get("intended_database_disposition"))
        and _CONTENT_CID.fullmatch(str(terminal.get("evidence_id") or ""))
        is not None
        and all(
            _SHA256.fullmatch(str(terminal.get(name) or "")) is not None
            for name in (
                "prepared_reconciliation_receipt_id",
                "commit_barrier_receipt_id",
                "receipt_id",
            )
        )
        and terminal.get("stage") == "terminal"
        and isinstance(record_json, Mapping)
        and set(record_json) == {"canonical_sha256", "canonical_byte_length"}
        and _SHA256.fullmatch(str(record_json.get("canonical_sha256") or ""))
        is not None
        and type(record_json.get("canonical_byte_length")) is int
        and 0 < record_json["canonical_byte_length"] <= 1 * 1024 * 1024
        and _SHA256.fullmatch(str(authority_id or "")) is not None
        and retained_recovery_sha256(unsigned) == authority_id
    )


def database_fenced_provider_historical_occurrence_authority(
    *,
    inner_receipt: Mapping[str, Any],
    controller_quiescence_receipt: Mapping[str, Any],
    board_namespace: str,
    owner_store_id: str,
    control_store_generation: str,
) -> Mapping[str, Any]:
    """Bind one exact terminal lane row to one controller quiescence receipt."""

    if not database_portal_controller_quiescence_receipt_valid(
        controller_quiescence_receipt
    ):
        raise ValueError("historical authority lacks controller quiescence")
    if (
        controller_quiescence_receipt.get("board_namespace")
        != board_namespace
        or controller_quiescence_receipt.get("owner_store_id")
        != owner_store_id
        or controller_quiescence_receipt.get("control_store_generation")
        != control_store_generation
    ):
        raise ValueError(
            "historical authority does not match controller store identity"
        )
    subject = inner_receipt.get("subject")
    groups = inner_receipt.get("groups")
    terminal_group = (
        groups.get("database_portal_terminal_reconciliations")
        if isinstance(groups, Mapping)
        else None
    )
    rows = terminal_group.get("rows") if isinstance(terminal_group, Mapping) else None
    if (
        not isinstance(subject, Mapping)
        or set(subject) != _HISTORICAL_SUBJECT_FIELDS
        or not isinstance(rows, list)
        or len(rows) != 1
        or terminal_group.get("count") != 1
        or not isinstance(rows[0], Mapping)
    ):
        raise ValueError("historical authority lacks one exact terminal lane row")
    record = {
        "schema": DATABASE_FENCED_PROVIDER_HISTORICAL_OCCURRENCE_AUTHORITY_SCHEMA,
        "board_namespace": str(board_namespace),
        "owner_store_id": str(owner_store_id),
        "control_store_generation": str(control_store_generation),
        "receipt_nonce": inner_receipt.get("receipt_nonce"),
        "receipt_epoch": inner_receipt.get("receipt_epoch"),
        "subject": dict(subject),
        "inner_receipt_cid": inner_receipt.get("receipt_cid"),
        "terminal_reconciliation": dict(rows[0]),
        "controller_quiescence_receipt_id": controller_quiescence_receipt.get(
            "receipt_id"
        ),
    }
    record["authority_id"] = retained_recovery_sha256(record)
    if not database_fenced_provider_historical_occurrence_authority_valid(record):
        raise ValueError("historical occurrence authority failed closed validation")
    return record


__all__ = [
    "DATABASE_FENCED_PROVIDER_HISTORICAL_OCCURRENCE_AUTHORITY_SCHEMA",
    "DATABASE_PORTAL_CONTROLLER_QUIESCENCE_RECEIPT_SCHEMA",
    "database_fenced_provider_historical_occurrence_authority",
    "database_fenced_provider_historical_occurrence_authority_valid",
    "database_portal_controller_quiescence_receipt",
    "database_portal_controller_quiescence_receipt_valid",
    "retained_recovery_sha256",
]
