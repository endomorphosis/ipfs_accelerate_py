"""DuckDB/Quack-coordinated identity recovery for concurrent supervisors.

Live coordination is the exclusive Quack owner on the operational DuckDB
file. DuckLake stays post-commit history and is never current authority.
Planning is pure. Claiming and overlay CAS are owner-inbox effects using
closed SQL templates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

IDENTITY_RECOVERY_PLAN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-identity-recovery@1"
)
IDENTITY_RECOVERY_SUBJECT_KIND: Final = "identity_recovery"
IDENTITY_RECOVERY_SUBJECT_ID: Final = "eaaef-control-plane-identity"
IDENTITY_RECOVERY_ACTION_KIND: Final = "control_plane_identity"
CAS_TASK_STATUS_SQL: Final = (
    "UPDATE tasks SET status = ?, revision = ?, updated_at = ? "
    "WHERE task_cid = ? AND revision = ?"
)
CLAIM_IDENTITY_RECOVERY_SQL: Final = (
    "INSERT INTO recovery_actions ("
    "action_id, subject_kind, subject_id, task_cid, action_kind, "
    "decided_at, status, body_json"
    ") SELECT ?, 'identity_recovery', ?, '', 'control_plane_identity', ?, "
    "'accepted', ? WHERE NOT EXISTS ("
    "SELECT 1 FROM recovery_actions WHERE subject_kind = 'identity_recovery' "
    "AND subject_id = ? AND status = 'accepted'"
    ") RETURNING action_id"
)
RELEASE_IDENTITY_RECOVERY_SQL: Final = (
    "UPDATE recovery_actions SET status = 'released', body_json = ? "
    "WHERE action_id = ? AND status = 'accepted' RETURNING action_id"
)
OWNER_IDENTITY_RECOVERY_SQL: Final = frozenset(
    {
        CAS_TASK_STATUS_SQL,
        CLAIM_IDENTITY_RECOVERY_SQL,
        RELEASE_IDENTITY_RECOVERY_SQL,
    }
)
_MUTABLE_RUNTIME_KEYS: Final = frozenset(
    {"status", "revision", "updated_at", "updated_at_ms"}
)
_DONE_STATUSES: Final = frozenset(
    {"completed", "accepted", "complete", "done"}
)


class IdentityRecoveryAction(str, Enum):
    NONE = "none"
    WAIT_FOR_HOLDER = "wait_for_holder"
    PRESERVE_AND_REBIND = "preserve_and_rebind"
    ADVANCE_EMPTY = "advance_empty"
    DIAGNOSE_ADMISSION = "diagnose_admission"


@dataclass(frozen=True)
class IdentityRecoveryPlan:
    action: IdentityRecoveryAction
    reason: str
    source_head: str = ""
    materialization_source_head: str = ""
    overlay_completed: int = 0
    ducklake_current_authority: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": IDENTITY_RECOVERY_PLAN_SCHEMA,
            "action": self.action.value,
            "reason": self.reason,
            "source_head": self.source_head,
            "materialization_source_head": self.materialization_source_head,
            "overlay_completed": int(self.overlay_completed),
            "ducklake_current_authority": False,
            "process_started": False,
            "configured_board_launch": False,
        }


def identity_rows(rows: Any) -> list[dict[str, Any]]:
    """Drop mutable overlay fields so identity comparison survives CAS."""

    if not isinstance(rows, list):
        return []
    identity: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        identity.append(
            {
                key: value
                for key, value in item.items()
                if key not in _MUTABLE_RUNTIME_KEYS
            }
        )
    return identity


def identity_control_projection(control: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compare board identities, not live task status."""

    if not isinstance(control, Mapping):
        return {}
    projected = dict(control)
    for key in (
        "tasks",
        "goals",
        "objectives",
        "task_revisions",
        "objective_revisions",
    ):
        if key in projected:
            projected[key] = identity_rows(projected.get(key))
    projected.pop("projection_root", None)
    snapshot = projected.get("intent_snapshot")
    if isinstance(snapshot, Mapping):
        snapshot = dict(snapshot)
        snapshot.pop("projection_cid", None)
        projected["intent_snapshot"] = snapshot
    # Relation fingerprints hash live overlay rows, including status.
    projected.pop("exact_relations", None)
    return projected


def snapshot_overlay_alias_status(projection: Mapping[str, Any] | Path) -> dict[str, str]:
    """Read alias→status from the host projection, never from a live SQL string."""

    payload: Mapping[str, Any]
    if isinstance(projection, Path):
        if not projection.is_file():
            return {}
        try:
            loaded = json.loads(projection.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(loaded, Mapping):
            return {}
        payload = loaded
    else:
        payload = projection
    statuses = payload.get("statuses")
    if not isinstance(statuses, Mapping):
        return {}
    return {
        str(alias): str(status)
        for alias, status in statuses.items()
        if str(alias).strip() and str(status).strip()
    }


def overlay_completed_count(statuses: Mapping[str, str]) -> int:
    return sum(1 for status in statuses.values() if status in _DONE_STATUSES)


def plan_control_plane_identity_recovery(
    *,
    source_head: str,
    materialization_source_head: str,
    overlay_statuses: Mapping[str, str] | None = None,
    recovery_held: bool = False,
    ducklake_current_authority: bool = False,
) -> IdentityRecoveryPlan:
    """Plan one exclusive identity recovery. Never starts a supervisor."""

    if ducklake_current_authority:
        return IdentityRecoveryPlan(
            action=IdentityRecoveryAction.NONE,
            reason="ducklake_is_not_current_authority",
            source_head=source_head,
            materialization_source_head=materialization_source_head,
        )
    statuses = dict(overlay_statuses or {})
    completed = overlay_completed_count(statuses)
    if recovery_held:
        return IdentityRecoveryPlan(
            action=IdentityRecoveryAction.WAIT_FOR_HOLDER,
            reason="another_supervisor_holds_identity_recovery",
            source_head=source_head,
            materialization_source_head=materialization_source_head,
            overlay_completed=completed,
        )
    if not source_head or not materialization_source_head:
        return IdentityRecoveryPlan(
            action=IdentityRecoveryAction.DIAGNOSE_ADMISSION,
            reason="materialization_or_source_identity_missing",
            source_head=source_head,
            materialization_source_head=materialization_source_head,
            overlay_completed=completed,
        )
    if source_head == materialization_source_head:
        return IdentityRecoveryPlan(
            action=IdentityRecoveryAction.DIAGNOSE_ADMISSION,
            reason="source_matches_materialization",
            source_head=source_head,
            materialization_source_head=materialization_source_head,
            overlay_completed=completed,
        )
    if completed:
        return IdentityRecoveryPlan(
            action=IdentityRecoveryAction.PRESERVE_AND_REBIND,
            reason="stale_materialization_has_overlay_work",
            source_head=source_head,
            materialization_source_head=materialization_source_head,
            overlay_completed=completed,
        )
    return IdentityRecoveryPlan(
        action=IdentityRecoveryAction.ADVANCE_EMPTY,
        reason="stale_materialization_without_overlay_work",
        source_head=source_head,
        materialization_source_head=materialization_source_head,
        overlay_completed=0,
    )


def restore_overlay_cas_parameters(
    *,
    live_rows: Sequence[Mapping[str, Any]],
    overlay_statuses: Mapping[str, str],
    updated_at: str,
) -> list[tuple[Any, ...]]:
    """Build allowlisted CAS parameter tuples for alias-preserving restore."""

    parameters: list[tuple[Any, ...]] = []
    for row in live_rows:
        alias = str(row.get("task_alias") or "")
        wanted = overlay_statuses.get(alias)
        if not wanted or wanted not in _DONE_STATUSES:
            continue
        current = str(row.get("status") or "")
        if current == wanted:
            continue
        task_cid = str(row.get("task_cid") or "")
        revision = int(row.get("revision") or 0)
        if not task_cid or revision < 0:
            continue
        parameters.append((wanted, revision + 1, updated_at, task_cid, revision))
    return parameters
