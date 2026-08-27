"""Pure, non-authoritative campaign progress projection.

This module renders an already captured typed-owner completion snapshot.  It
has no database, Quack, board, environment, or discovery fallback: callers
must provide the exact owner envelope, a sealed alias-to-task population, and
an explicit program qualification disposition.

The renderer deliberately keeps operational state separate from normalized,
current-revision completion evidence.  Its JSON and Markdown products are
diagnostic exports and must never be consumed as control-plane authority.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .control_plane_contracts import StoreGeneration, canonical_json_bytes, content_identity

CAMPAIGN_PROGRESS_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/campaign-progress-report@2"
)
CAMPAIGN_PROGRESS_CURRENT_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/campaign-progress-current-manifest@1"
)
PROGRAM_QUALIFICATION_DISPOSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-qualification-disposition@1"
)
INTENT_COMPLETION_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-completion-projection@1"
)
COMPLETION_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1"
)
TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/typed-completion-progress-snapshot@1"
)
TYPED_COMPLETION_PROGRESS_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/typed-completion-progress-request@1"
)

NON_AUTHORITATIVE_BANNER: Final[str] = (
    "NON-AUTHORITATIVE PROGRESS EXPORT — runtime decisions and canonical "
    "task boards must not read this artifact; the typed control-plane "
    "snapshot remains the sole operational authority."
)

PROTECTED_BOARD_PATHS: Final[tuple[str, ...]] = (
    "docs/architecture/agent_supervisor_causal_event_federation.todo.md",
    "docs/architecture/external_agent_autonomous_execution_fabric/TASK_BOARD.md",
    "docs/architecture/external_agent_autonomous_execution_fabric/task_board.json",
)

_SUCCESSFUL_COMPLETION_STATUSES: Final[frozenset[str]] = frozenset(
    {"complete", "completed", "done"}
)
_OPERATIONAL_COMPLETION_STATUSES: Final[frozenset[str]] = frozenset(
    {*_SUCCESSFUL_COMPLETION_STATUSES, "skipped"}
)
_QUALIFICATION_STATUSES: Final[frozenset[str]] = frozenset(
    {"blocked", "not_qualified", "not_run", "qualified"}
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_MAX_PROJECTION_BYTES: Final[int] = 16_777_216
_MAX_TASKS: Final[int] = 50_000
_MAX_RECEIPTS: Final[int] = 50_000
_MAX_JSON_DEPTH: Final[int] = 24
_MAX_TEXT_BYTES: Final[int] = 8_192


class CampaignProgressError(ValueError):
    """Base class for closed campaign-progress contract failures."""


class CampaignProgressValidationError(CampaignProgressError):
    """An input snapshot, population, or disposition failed validation."""


class CampaignProgressDestinationError(CampaignProgressError):
    """An output destination is implicit, protected, or outside its root."""


class CampaignProgressWriteError(RuntimeError):
    """A validated progress export could not be atomically replaced."""


@dataclass(frozen=True)
class CampaignProgressRendering:
    """Deterministic machine and human renderings of one progress report."""

    report: Mapping[str, Any]
    json_text: str
    markdown_text: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of the report."""

        # The report was already checked by canonical_json_bytes.  A canonical
        # JSON round trip is the smallest way to detach all nested containers.
        import json

        return json.loads(self.json_text)


def _json_value(value: Any, *, noun: str, depth: int = 0) -> Any:
    if depth > _MAX_JSON_DEPTH:
        raise CampaignProgressValidationError(f"{noun} exceeds depth bound")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        raise CampaignProgressValidationError(f"{noun} may not contain floats")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CampaignProgressValidationError(f"{noun} mapping keys must be strings")
        return {key: _json_value(value[key], noun=noun, depth=depth + 1) for key in value}
    if isinstance(value, list):
        return [_json_value(item, noun=noun, depth=depth + 1) for item in value]
    raise CampaignProgressValidationError(f"{noun} contains unsupported {type(value).__name__}")


def _closed_mapping(
    value: Any,
    *,
    noun: str,
    fields: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignProgressValidationError(f"{noun} must be a mapping")
    normalized = _json_value(value, noun=noun)
    assert isinstance(normalized, dict)
    observed = frozenset(normalized)
    if observed != fields:
        missing = sorted(fields - observed)
        extra = sorted(observed - fields)
        raise CampaignProgressValidationError(
            f"{noun} fields are not closed; missing={missing}, extra={extra}"
        )
    return normalized


def _text(
    value: Any,
    *,
    noun: str,
    required: bool = True,
    line_only: bool = False,
) -> str:
    if not isinstance(value, str):
        raise CampaignProgressValidationError(f"{noun} must be a string")
    if value != value.strip():
        raise CampaignProgressValidationError(f"{noun} has leading or trailing whitespace")
    if required and not value:
        raise CampaignProgressValidationError(f"{noun} must not be empty")
    if "\x00" in value or (line_only and ("\n" in value or "\r" in value)):
        raise CampaignProgressValidationError(f"{noun} contains control text")
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise CampaignProgressValidationError(f"{noun} exceeds byte bound")
    return value


def _identifier(value: Any, *, noun: str, required: bool = True) -> str:
    text = _text(value, noun=noun, required=required, line_only=True)
    if text and _SAFE_ID.fullmatch(text) is None:
        raise CampaignProgressValidationError(f"{noun} is not a compact ID")
    return text


def _nonnegative_integer(value: Any, *, noun: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CampaignProgressValidationError(f"{noun} must be a non-negative integer")
    return value


def _positive_integer(value: Any, *, noun: str) -> int:
    normalized = _nonnegative_integer(value, noun=noun)
    if normalized < 1:
        raise CampaignProgressValidationError(f"{noun} must be a positive integer")
    return normalized


def _string_list(
    value: Any,
    *,
    noun: str,
    identifiers: bool,
    require_sorted: bool,
) -> list[str]:
    if not isinstance(value, list):
        raise CampaignProgressValidationError(f"{noun} must be a list")
    normalized = [
        (
            _identifier(item, noun=f"{noun} item")
            if identifiers
            else _text(item, noun=f"{noun} item", line_only=True)
        )
        for item in value
    ]
    if len(set(normalized)) != len(normalized):
        raise CampaignProgressValidationError(f"{noun} contains duplicates")
    if require_sorted and normalized != sorted(normalized):
        raise CampaignProgressValidationError(f"{noun} must be sorted")
    return normalized


def _sealed_population(sealed_tasks: Mapping[str, str]) -> tuple[dict[str, str], str]:
    if not isinstance(sealed_tasks, Mapping):
        raise CampaignProgressValidationError("sealed task population must be a mapping")
    if not sealed_tasks:
        raise CampaignProgressValidationError("sealed task population must not be empty")
    if len(sealed_tasks) > _MAX_TASKS:
        raise CampaignProgressValidationError("sealed task population exceeds task bound")
    normalized: dict[str, str] = {}
    for raw_alias, raw_cid in sealed_tasks.items():
        alias = _identifier(raw_alias, noun="task alias")
        task_cid = _identifier(raw_cid, noun=f"task CID for {alias}")
        normalized[alias] = task_cid
    if len(set(normalized.values())) != len(normalized):
        raise CampaignProgressValidationError("sealed task population contains duplicate task CIDs")
    normalized = dict(sorted(normalized.items()))
    population_body = {
        "schema": ("ipfs_accelerate_py/agent-supervisor/campaign-sealed-task-population@1"),
        "tasks": [
            {"task_alias": alias, "task_cid": task_cid} for alias, task_cid in normalized.items()
        ],
    }
    return normalized, content_identity(population_body)


def build_program_qualification_disposition(
    *,
    program_id: str,
    status: str,
    blockers: Sequence[str],
    evidence_refs: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Build a strict, content-addressed qualification disposition."""

    program = _identifier(program_id, noun="program_id")
    disposition_status = _identifier(status, noun="qualification status")
    if disposition_status not in _QUALIFICATION_STATUSES:
        raise CampaignProgressValidationError("qualification status is not supported")
    if isinstance(blockers, (str, bytes)) or not isinstance(blockers, Sequence):
        raise CampaignProgressValidationError("blockers must be a sequence")
    if isinstance(evidence_refs, (str, bytes)) or not isinstance(evidence_refs, Sequence):
        raise CampaignProgressValidationError("qualification evidence_refs must be a sequence")
    blocker_values = [_text(item, noun="blocker", line_only=True) for item in blockers]
    evidence_values = [
        _identifier(item, noun="qualification evidence reference") for item in evidence_refs
    ]
    if len(set(blocker_values)) != len(blocker_values):
        raise CampaignProgressValidationError("blockers contains duplicates")
    if len(set(evidence_values)) != len(evidence_values):
        raise CampaignProgressValidationError("qualification evidence_refs contains duplicates")
    blocker_values.sort()
    evidence_values.sort()
    if disposition_status == "qualified" and blocker_values:
        raise CampaignProgressValidationError("qualified disposition may not declare blockers")
    if disposition_status != "qualified" and not blocker_values:
        raise CampaignProgressValidationError(
            "non-qualified disposition must declare at least one blocker"
        )
    body = {
        "schema": PROGRAM_QUALIFICATION_DISPOSITION_SCHEMA,
        "program_id": program,
        "status": disposition_status,
        "blockers": blocker_values,
        "evidence_refs": evidence_values,
    }
    return MappingProxyType({**body, "disposition_cid": content_identity(body)})


def _qualification_disposition(value: Mapping[str, Any]) -> dict[str, Any]:
    disposition = _closed_mapping(
        value,
        noun="program qualification disposition",
        fields=frozenset(
            {
                "schema",
                "program_id",
                "status",
                "blockers",
                "evidence_refs",
                "disposition_cid",
            }
        ),
    )
    if disposition["schema"] != PROGRAM_QUALIFICATION_DISPOSITION_SCHEMA:
        raise CampaignProgressValidationError(
            "unsupported program qualification disposition schema"
        )
    program_id = _identifier(disposition["program_id"], noun="program_id")
    status = _identifier(disposition["status"], noun="qualification status")
    if status not in _QUALIFICATION_STATUSES:
        raise CampaignProgressValidationError("qualification status is not supported")
    blockers = _string_list(
        disposition["blockers"],
        noun="blockers",
        identifiers=False,
        require_sorted=True,
    )
    evidence_refs = _string_list(
        disposition["evidence_refs"],
        noun="qualification evidence_refs",
        identifiers=True,
        require_sorted=True,
    )
    if status == "qualified" and blockers:
        raise CampaignProgressValidationError("qualified disposition may not declare blockers")
    if status != "qualified" and not blockers:
        raise CampaignProgressValidationError(
            "non-qualified disposition must declare at least one blocker"
        )
    body = {
        "schema": PROGRAM_QUALIFICATION_DISPOSITION_SCHEMA,
        "program_id": program_id,
        "status": status,
        "blockers": blockers,
        "evidence_refs": evidence_refs,
    }
    observed_cid = _identifier(disposition["disposition_cid"], noun="qualification disposition CID")
    if observed_cid != content_identity(body):
        raise CampaignProgressValidationError(
            "qualification disposition CID does not match its body"
        )
    return {**body, "disposition_cid": observed_cid}


def _validate_completion_receipt(
    raw_receipt: Any,
    *,
    task_revisions: Mapping[str, int],
) -> dict[str, Any]:
    receipt = _closed_mapping(
        raw_receipt,
        noun="completion receipt",
        fields=frozenset(
            {
                "receipt_cid",
                "task_cid",
                "goal_cid",
                "attempt_id",
                "claim_cid",
                "fencing_token",
                "completed_at",
                "validation_run_id",
                "evidence_digest",
                "body",
            }
        ),
    )
    task_cid = _identifier(receipt["task_cid"], noun="receipt task CID")
    if task_cid not in task_revisions:
        raise CampaignProgressValidationError(
            "completion receipt task is outside sealed snapshot population"
        )
    body = _closed_mapping(
        receipt["body"],
        noun=f"completion receipt body for {task_cid}",
        fields=frozenset({"schema", "receipt", "evidence_digests", "revision"}),
    )
    if body["schema"] != COMPLETION_EVIDENCE_SCHEMA:
        raise CampaignProgressValidationError("unsupported completion evidence schema")
    revision = _nonnegative_integer(
        body["revision"], noun=f"completion receipt revision for {task_cid}"
    )
    if revision != task_revisions[task_cid]:
        raise CampaignProgressValidationError(
            f"completion receipt for {task_cid} is stale for current revision"
        )
    if not isinstance(body["receipt"], Mapping):
        raise CampaignProgressValidationError(
            f"control receipt for {task_cid} must be a mapping"
        )
    control_receipt = _json_value(body["receipt"], noun=f"control receipt for {task_cid}")
    assert isinstance(control_receipt, dict)
    evidence_digests = _string_list(
        body["evidence_digests"],
        noun=f"completion evidence digests for {task_cid}",
        identifiers=True,
        require_sorted=False,
    )
    expected_evidence_digest = content_identity(
        {
            "task_cid": task_cid,
            "revision": revision,
            "receipt": control_receipt,
            "evidence_digests": evidence_digests,
        }
    )
    observed_evidence_digest = _identifier(
        receipt["evidence_digest"], noun="completion evidence digest"
    )
    if observed_evidence_digest != expected_evidence_digest:
        raise CampaignProgressValidationError(
            f"completion evidence identity is invalid for {task_cid}"
        )
    expected_receipt_cid = content_identity(
        {
            "namespace": "completion-receipt",
            "task_cid": task_cid,
            "revision": revision,
            "evidence_digest": expected_evidence_digest,
        }
    )
    observed_receipt_cid = _identifier(receipt["receipt_cid"], noun="completion receipt CID")
    if observed_receipt_cid != expected_receipt_cid:
        raise CampaignProgressValidationError(
            f"completion receipt identity is invalid for {task_cid}"
        )
    return {
        "receipt_cid": observed_receipt_cid,
        "task_cid": task_cid,
        "goal_cid": _identifier(receipt["goal_cid"], noun="receipt goal CID"),
        "attempt_id": _identifier(receipt["attempt_id"], noun="receipt attempt ID", required=False),
        "claim_cid": _identifier(receipt["claim_cid"], noun="receipt claim CID", required=False),
        "fencing_token": _nonnegative_integer(
            receipt["fencing_token"], noun="receipt fencing token"
        ),
        "completed_at": _text(receipt["completed_at"], noun="receipt completed_at", line_only=True),
        "validation_run_id": _identifier(
            receipt["validation_run_id"],
            noun="receipt validation run ID",
            required=False,
        ),
        "evidence_digest": observed_evidence_digest,
        "revision": revision,
        "control_receipt_cid": content_identity(control_receipt),
    }


def _completion_snapshot(
    value: Mapping[str, Any],
    *,
    sealed_task_cids: frozenset[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    snapshot = _closed_mapping(
        value,
        noun="typed completion snapshot",
        fields=frozenset(
            {
                "schema",
                "event_watermark",
                "task_states",
                "completion_receipts",
                "projection_cid",
            }
        ),
    )
    if snapshot["schema"] != INTENT_COMPLETION_PROJECTION_SCHEMA:
        raise CampaignProgressValidationError("unsupported typed completion snapshot schema")
    material = {key: snapshot[key] for key in snapshot if key != "projection_cid"}
    encoded = canonical_json_bytes(material)
    if len(encoded) > _MAX_PROJECTION_BYTES:
        raise CampaignProgressValidationError("typed completion snapshot exceeds byte bound")
    observed_projection_cid = _identifier(
        snapshot["projection_cid"], noun="completion snapshot projection CID"
    )
    if observed_projection_cid != content_identity(material):
        raise CampaignProgressValidationError(
            "completion snapshot projection CID does not match its body"
        )
    event_watermark = _nonnegative_integer(
        snapshot["event_watermark"], noun="snapshot event watermark"
    )
    if not isinstance(snapshot["task_states"], list):
        raise CampaignProgressValidationError("task_states must be a list")
    if len(snapshot["task_states"]) > _MAX_TASKS:
        raise CampaignProgressValidationError("task_states exceeds task bound")
    task_states: list[dict[str, Any]] = []
    for raw_state in snapshot["task_states"]:
        state = _closed_mapping(
            raw_state,
            noun="task state",
            fields=frozenset({"task_cid", "status", "revision"}),
        )
        task_cid = _identifier(state["task_cid"], noun="task state CID")
        status = _identifier(state["status"], noun=f"task status for {task_cid}")
        if status != status.lower():
            raise CampaignProgressValidationError(f"task status for {task_cid} must be lowercase")
        task_states.append(
            {
                "task_cid": task_cid,
                "status": status,
                "revision": _nonnegative_integer(
                    state["revision"], noun=f"task revision for {task_cid}"
                ),
            }
        )
    task_cids = [state["task_cid"] for state in task_states]
    if len(set(task_cids)) != len(task_cids):
        raise CampaignProgressValidationError(
            "typed completion snapshot contains duplicate task states"
        )
    if task_cids != sorted(task_cids):
        raise CampaignProgressValidationError(
            "typed completion snapshot task states must be sorted by task CID"
        )
    observed_task_cids = frozenset(task_cids)
    if observed_task_cids != sealed_task_cids:
        missing = sorted(sealed_task_cids - observed_task_cids)
        extra = sorted(observed_task_cids - sealed_task_cids)
        raise CampaignProgressValidationError(
            "typed completion snapshot task population does not match seal; "
            f"missing={missing}, extra={extra}"
        )
    task_revisions = {state["task_cid"]: state["revision"] for state in task_states}
    if not isinstance(snapshot["completion_receipts"], list):
        raise CampaignProgressValidationError("completion_receipts must be a list")
    if len(snapshot["completion_receipts"]) > _MAX_RECEIPTS:
        raise CampaignProgressValidationError("completion_receipts exceeds receipt bound")
    receipts: list[dict[str, Any]] = [
        _validate_completion_receipt(raw, task_revisions=task_revisions)
        for raw in snapshot["completion_receipts"]
    ]
    receipt_sort_keys = [
        (item["task_cid"], item["completed_at"], item["receipt_cid"]) for item in receipts
    ]
    if receipt_sort_keys != sorted(receipt_sort_keys):
        raise CampaignProgressValidationError(
            "completion receipts are not in canonical projection order"
        )
    receipt_task_cids = [item["task_cid"] for item in receipts]
    if len(set(receipt_task_cids)) != len(receipt_task_cids):
        raise CampaignProgressValidationError(
            "snapshot has duplicate current-revision completion receipts"
        )
    receipt_by_task = {item["task_cid"]: item for item in receipts}
    states_by_task = {state["task_cid"]: state for state in task_states}
    for task_cid in receipt_by_task:
        if states_by_task[task_cid]["status"] not in _OPERATIONAL_COMPLETION_STATUSES:
            raise CampaignProgressValidationError(
                f"current completion receipt exists for non-complete task {task_cid}"
            )
    return (
        {
            "schema": INTENT_COMPLETION_PROJECTION_SCHEMA,
            "event_watermark": event_watermark,
            "projection_cid": observed_projection_cid,
            "task_states": task_states,
        },
        receipt_by_task,
    )


def _validated_owner_snapshot(
    value: Mapping[str, Any],
    *,
    sealed_task_cids: frozenset[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate and bind the complete typed-owner snapshot envelope."""

    outer = _closed_mapping(
        value,
        noun="typed completion owner snapshot",
        fields=frozenset(
            {
                "schema",
                "request_cid",
                "owner_identity",
                "store_generation",
                "completion_projection",
                "snapshot_cid",
            }
        ),
    )
    if outer["schema"] != TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA:
        raise CampaignProgressValidationError(
            "unsupported typed completion owner snapshot schema"
        )
    request_cid = _identifier(outer["request_cid"], noun="owner snapshot request CID")
    owner = _closed_mapping(
        outer["owner_identity"],
        noun="completion snapshot owner identity",
        fields=frozenset(
            {
                "server_id",
                "process_birth_id",
                "store_id",
                "database_uuid",
                "generation",
                "fence_epoch",
            }
        ),
    )
    normalized_owner = {
        "server_id": _identifier(owner["server_id"], noun="owner server ID"),
        "process_birth_id": _identifier(
            owner["process_birth_id"], noun="owner process birth ID"
        ),
        "store_id": _identifier(owner["store_id"], noun="owner store ID"),
        "database_uuid": _identifier(
            owner["database_uuid"], noun="owner database UUID"
        ),
        "generation": _positive_integer(owner["generation"], noun="owner generation"),
        "fence_epoch": _positive_integer(owner["fence_epoch"], noun="owner fence epoch"),
    }
    request_material = {
        "schema": TYPED_COMPLETION_PROGRESS_REQUEST_SCHEMA,
        "task_cids": sorted(sealed_task_cids),
        "expected_server_id": normalized_owner["server_id"],
        "expected_process_birth_id": normalized_owner["process_birth_id"],
        "expected_store_id": normalized_owner["store_id"],
        "expected_database_uuid": normalized_owner["database_uuid"],
        "expected_generation": normalized_owner["generation"],
        "expected_fence_epoch": normalized_owner["fence_epoch"],
    }
    if request_cid != content_identity(request_material):
        raise CampaignProgressValidationError(
            "typed completion owner snapshot request CID does not match owner and task population"
        )
    generation_payload = _closed_mapping(
        outer["store_generation"],
        noun="completion snapshot store generation",
        fields=frozenset(
            {
                "schema",
                "contract_version",
                "store_id",
                "generation",
                "schema_revision",
                "fence_epoch",
                "revision",
                "database_uuid",
                "birth_id",
            }
        ),
    )
    try:
        generation = StoreGeneration.from_dict(generation_payload)
    except Exception as exc:
        raise CampaignProgressValidationError(
            "completion snapshot store generation is invalid"
        ) from exc
    if (
        generation.store_id != normalized_owner["store_id"]
        or generation.database_uuid != normalized_owner["database_uuid"]
        or generation.generation != normalized_owner["generation"]
        or generation.fence_epoch != normalized_owner["fence_epoch"]
        or generation.birth_id != normalized_owner["process_birth_id"]
    ):
        raise CampaignProgressValidationError(
            "completion snapshot owner and store generation identities differ"
        )
    outer_material = {
        "schema": TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA,
        "request_cid": request_cid,
        "owner_identity": normalized_owner,
        "store_generation": generation.to_dict(),
        "completion_projection": outer["completion_projection"],
    }
    if len(canonical_json_bytes(outer_material)) > _MAX_PROJECTION_BYTES:
        raise CampaignProgressValidationError("typed completion owner snapshot exceeds byte bound")
    snapshot_cid = _identifier(outer["snapshot_cid"], noun="owner snapshot CID")
    if snapshot_cid != content_identity(outer_material):
        raise CampaignProgressValidationError(
            "typed completion owner snapshot CID does not match its body"
        )
    completion, receipts = _completion_snapshot(
        outer["completion_projection"],
        sealed_task_cids=sealed_task_cids,
    )
    source = {
        "schema": TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA,
        "snapshot_cid": snapshot_cid,
        "request_cid": request_cid,
        "owner_identity": normalized_owner,
        "store_generation": generation.to_dict(),
        "completion_projection_schema": completion["schema"],
        "completion_projection_cid": completion["projection_cid"],
        "event_watermark": completion["event_watermark"],
    }
    return source, completion, receipts


def _markdown_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|")


def _markdown(report: Mapping[str, Any]) -> str:
    source = report["source_snapshot"]
    operational = report["operational_state"]
    backed = report["current_revision_normalized_receipt_backed_completions"]
    lacking = report["operational_completions_without_normalized_evidence"]
    qualification = report["program_qualification"]
    lines = [
        "# Campaign progress report",
        "",
        f"> **{NON_AUTHORITATIVE_BANNER}**",
        "",
        f"- Report CID: `{report['report_cid']}`",
        f"- Program: `{report['program_id']}`",
        f"- Typed owner snapshot CID: `{source['snapshot_cid']}`",
        f"- Source request CID: `{source['request_cid']}`",
        f"- Owner server ID: `{source['owner_identity']['server_id']}`",
        f"- Owner process birth ID: `{source['owner_identity']['process_birth_id']}`",
        f"- Store ID: `{source['store_generation']['store_id']}`",
        f"- Database UUID: `{source['store_generation']['database_uuid']}`",
        f"- Store generation: {source['store_generation']['generation']}",
        f"- Store fence epoch: {source['store_generation']['fence_epoch']}",
        f"- Store schema revision: {source['store_generation']['schema_revision']}",
        f"- Store revision: {source['store_generation']['revision']}",
        f"- Source projection CID: `{source['completion_projection_cid']}`",
        f"- Source event watermark: {source['event_watermark']}",
        f"- Sealed task population CID: `{source['sealed_task_population_cid']}`",
        "",
        "## Operational state",
        "",
        f"Operational task rows: {operational['task_count']}",
        "",
        "| Alias | Task CID | Status | Revision |",
        "| --- | --- | --- | ---: |",
    ]
    for task in operational["tasks"]:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(task[key])
                for key in ("task_alias", "task_cid", "status", "revision")
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Current-revision normalized receipt-backed completions",
            "",
            f"Receipt-backed completion rows: {backed['count']}",
            "",
            "| Alias | Task CID | Revision | Completion receipt CID |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for task in backed["tasks"]:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(task[key])
                for key in (
                    "task_alias",
                    "task_cid",
                    "revision",
                    "receipt_cid",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Operational completions lacking normalized evidence",
            "",
            f"Operational completion rows lacking evidence: {lacking['count']}",
            "",
            "| Alias | Task CID | Status | Revision | Reason |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for task in lacking["tasks"]:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(task[key])
                for key in (
                    "task_alias",
                    "task_cid",
                    "status",
                    "revision",
                    "reason",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Program qualification and blockers",
            "",
            f"- Disposition: `{qualification['status']}`",
            f"- Qualified: `{str(qualification['qualified']).lower()}`",
            f"- Disposition CID: `{qualification['disposition_cid']}`",
            "- Blockers:",
        ]
    )
    blockers = qualification["blockers"]
    if blockers:
        lines.extend(f"  - {_markdown_cell(blocker)}" for blocker in blockers)
    else:
        lines.append("  - None")
    lines.extend(
        [
            "",
            "This export makes no live benchmark, DuckLake, fixed-point, or "
            "program-qualification claim beyond the explicit disposition above.",
            "",
        ]
    )
    return "\n".join(lines)


def render_campaign_progress(
    owner_snapshot: Mapping[str, Any],
    *,
    sealed_tasks: Mapping[str, str],
    qualification: Mapping[str, Any],
) -> CampaignProgressRendering:
    """Validate and render one owner-bound, non-authoritative projection."""

    population, population_cid = _sealed_population(sealed_tasks)
    disposition = _qualification_disposition(qualification)
    source_snapshot, snapshot, receipts = _validated_owner_snapshot(
        owner_snapshot,
        sealed_task_cids=frozenset(population.values()),
    )
    alias_by_cid = {task_cid: alias for alias, task_cid in population.items()}
    operational_tasks: list[dict[str, Any]] = []
    receipt_backed: list[dict[str, Any]] = []
    lacking_evidence: list[dict[str, Any]] = []
    for state in sorted(snapshot["task_states"], key=lambda item: alias_by_cid[item["task_cid"]]):
        task_cid = state["task_cid"]
        alias = alias_by_cid[task_cid]
        status = state["status"]
        revision = state["revision"]
        operational_tasks.append(
            {
                "task_alias": alias,
                "task_cid": task_cid,
                "status": status,
                "revision": revision,
            }
        )
        receipt = receipts.get(task_cid)
        if status in _SUCCESSFUL_COMPLETION_STATUSES and receipt is not None:
            receipt_backed.append(
                {
                    "task_alias": alias,
                    "task_cid": task_cid,
                    "status": status,
                    "revision": revision,
                    "receipt_cid": receipt["receipt_cid"],
                    "evidence_digest": receipt["evidence_digest"],
                    "control_receipt_cid": receipt["control_receipt_cid"],
                    "goal_cid": receipt["goal_cid"],
                    "completed_at": receipt["completed_at"],
                    "validation_run_id": receipt["validation_run_id"],
                }
            )
        elif status in _OPERATIONAL_COMPLETION_STATUSES:
            reason = (
                "operational_skip_is_not_a_successful_normalized_completion"
                if status == "skipped"
                else "current_revision_normalized_completion_receipt_absent"
            )
            missing_row: dict[str, Any] = {
                "task_alias": alias,
                "task_cid": task_cid,
                "status": status,
                "revision": revision,
                "reason": reason,
            }
            if receipt is not None:
                missing_row["observed_receipt_cid"] = receipt["receipt_cid"]
            lacking_evidence.append(missing_row)

    status_counts = Counter(task["status"] for task in operational_tasks)
    body: dict[str, Any] = {
        "schema": CAMPAIGN_PROGRESS_REPORT_SCHEMA,
        "authoritative": False,
        "banner": NON_AUTHORITATIVE_BANNER,
        "program_id": disposition["program_id"],
        "source_snapshot": {
            **source_snapshot,
            "sealed_task_population_cid": population_cid,
            "sealed_task_count": len(population),
        },
        "operational_state": {
            "task_count": len(operational_tasks),
            "counts_by_status": {status: status_counts[status] for status in sorted(status_counts)},
            "operational_completion_count": sum(
                task["status"] in _OPERATIONAL_COMPLETION_STATUSES for task in operational_tasks
            ),
            "tasks": operational_tasks,
        },
        "current_revision_normalized_receipt_backed_completions": {
            "count": len(receipt_backed),
            "tasks": receipt_backed,
        },
        "operational_completions_without_normalized_evidence": {
            "count": len(lacking_evidence),
            "tasks": lacking_evidence,
        },
        "program_qualification": {
            **disposition,
            "qualified": disposition["status"] == "qualified",
        },
    }
    report = {**body, "report_cid": content_identity(body)}
    json_text = canonical_json_bytes(report).decode("utf-8") + "\n"
    return CampaignProgressRendering(
        report=MappingProxyType(report),
        json_text=json_text,
        markdown_text=_markdown(report),
    )


def _validated_manifest_destination(
    destination: Path | str,
    *,
    repository_root: Path,
) -> Path:
    """Return a lexical root-relative manifest path without following symlinks."""

    if not isinstance(destination, (Path, str)) or not str(destination):
        raise CampaignProgressDestinationError(
            "current progress manifest destination must be explicitly provided"
        )
    candidate = Path(destination)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    lexical = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = lexical.relative_to(repository_root)
    except ValueError as exc:
        raise CampaignProgressDestinationError(
            "current progress manifest must stay inside repository_root"
        ) from exc
    if not relative.parts:
        raise CampaignProgressDestinationError(
            "current progress manifest must name a file"
        )
    if relative.as_posix() in PROTECTED_BOARD_PATHS:
        raise CampaignProgressDestinationError(
            f"refusing protected canonical board path: {relative.as_posix()}"
        )
    if relative.suffix.lower() != ".json":
        raise CampaignProgressDestinationError(
            "current progress manifest destination must end in .json"
        )
    return relative


_DIRECTORY_OPEN_FLAGS: Final[int] = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_NOFOLLOW_FLAG: Final[int] = getattr(os, "O_NOFOLLOW", 0)


def _open_output_parent(repository_root: Path, relative_parent: Path) -> int:
    """Open/create a descendant directory one no-follow component at a time."""

    try:
        descriptor = os.open(repository_root, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise CampaignProgressDestinationError(
            "repository_root could not be opened without following symlinks"
        ) from exc
    try:
        for component in relative_parent.parts:
            if component in {"", ".", ".."}:
                raise CampaignProgressDestinationError(
                    "current progress manifest parent is not lexical"
                )
            try:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except CampaignProgressDestinationError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise CampaignProgressDestinationError(
            "current progress manifest parent contains a symlink or non-directory"
        ) from exc


def _entry_stat(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _require_regular_or_absent(
    parent_descriptor: int,
    name: str,
    *,
    noun: str,
) -> None:
    observed = _entry_stat(parent_descriptor, name)
    if observed is not None and not stat.S_ISREG(observed.st_mode):
        raise CampaignProgressDestinationError(f"{noun} may not be a symlink or special file")


def _read_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    expected_size: int,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | _FILE_NOFOLLOW_FLAG | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_descriptor,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CampaignProgressWriteError("immutable progress artifact is not a regular file")
        if metadata.st_size != expected_size:
            raise CampaignProgressWriteError(
                "immutable progress artifact size differs for report CID"
            )
        chunks: list[bytes] = []
        remaining = expected_size + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                payload = b"".join(chunks)
                if len(payload) != expected_size:
                    raise CampaignProgressWriteError(
                        "immutable progress artifact size changed while reading"
                    )
                return payload
            chunks.append(chunk)
            remaining -= len(chunk)
        raise CampaignProgressWriteError(
            "immutable progress artifact grew while reading"
        )
    finally:
        os.close(descriptor)


def _stage_at(parent_descriptor: int, final_name: str, payload: bytes) -> str:
    temporary_name = ""
    descriptor = -1
    for _ in range(32):
        temporary_name = (
            f".{final_name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        )
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | _FILE_NOFOLLOW_FLAG
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            break
        except FileExistsError:
            continue
    if descriptor < 0:
        raise CampaignProgressWriteError("could not allocate a progress staging file")
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written < 1:
                raise CampaignProgressWriteError("progress staging write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        raise
    os.close(descriptor)
    return temporary_name


def _publish_immutable(
    parent_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    _require_regular_or_absent(
        parent_descriptor,
        name,
        noun="immutable progress artifact",
    )
    if _entry_stat(parent_descriptor, name) is not None:
        if (
            _read_regular_at(
                parent_descriptor,
                name,
                expected_size=len(payload),
            )
            != payload
        ):
            raise CampaignProgressWriteError(
                "immutable progress artifact content differs for report CID"
            )
        return
    temporary_name = _stage_at(parent_descriptor, name, payload)
    try:
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            _require_regular_or_absent(
                parent_descriptor,
                name,
                noun="immutable progress artifact",
            )
            if (
                _read_regular_at(
                    parent_descriptor,
                    name,
                    expected_size=len(payload),
                )
                != payload
            ):
                raise CampaignProgressWriteError(
                    "concurrent immutable progress artifact content differs"
                ) from None
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_name = ""
        os.fsync(parent_descriptor)
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass


def _replace_current_manifest(
    parent_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    _require_regular_or_absent(
        parent_descriptor,
        name,
        noun="current progress manifest",
    )
    temporary_name = _stage_at(parent_descriptor, name, payload)
    try:
        # Re-check the lexical entry after staging.  Even if another process
        # races after this check, replace(2) replaces the entry itself and does
        # not dereference a symlink target.
        _require_regular_or_absent(
            parent_descriptor,
            name,
            noun="current progress manifest",
        )
        os.replace(
            temporary_name,
            name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = ""
        os.fsync(parent_descriptor)
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass


def write_campaign_progress_outputs(
    rendering: CampaignProgressRendering,
    *,
    repository_root: Path | str,
    current_manifest_destination: Path | str,
) -> Mapping[str, Any]:
    """Publish immutable report artifacts, then one atomic current manifest."""

    if not isinstance(rendering, CampaignProgressRendering):
        raise CampaignProgressDestinationError("rendering must be a CampaignProgressRendering")
    try:
        root = Path(repository_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CampaignProgressDestinationError(
            "repository_root must be an existing directory"
        ) from exc
    if not root.is_dir():
        raise CampaignProgressDestinationError("repository_root must be a directory")
    relative_manifest = _validated_manifest_destination(
        current_manifest_destination,
        repository_root=root,
    )
    report = _json_value(rendering.report, noun="campaign progress rendering")
    if not isinstance(report, dict) or "report_cid" not in report:
        raise CampaignProgressValidationError("campaign progress rendering has no report CID")
    report_cid = _identifier(report["report_cid"], noun="campaign progress report CID")
    report_body = {key: value for key, value in report.items() if key != "report_cid"}
    if report_cid != content_identity(report_body):
        raise CampaignProgressValidationError("campaign progress report CID differs from its body")
    expected_json = canonical_json_bytes(report).decode("utf-8") + "\n"
    if rendering.json_text != expected_json or rendering.markdown_text != _markdown(report):
        raise CampaignProgressValidationError(
            "campaign progress rendering bytes differ from its report"
        )

    json_payload = rendering.json_text.encode("utf-8")
    markdown_payload = rendering.markdown_text.encode("utf-8")
    json_name = f"progress-{report_cid}.json"
    markdown_name = f"progress-{report_cid}.md"
    if relative_manifest.name in {json_name, markdown_name}:
        raise CampaignProgressDestinationError(
            "current manifest destination collides with an immutable artifact"
        )
    json_relative = relative_manifest.parent / json_name
    markdown_relative = relative_manifest.parent / markdown_name
    source = report.get("source_snapshot")
    if not isinstance(source, Mapping):
        raise CampaignProgressValidationError("campaign progress source snapshot is missing")
    manifest_body: dict[str, Any] = {
        "schema": CAMPAIGN_PROGRESS_CURRENT_MANIFEST_SCHEMA,
        "authoritative": False,
        "report_cid": report_cid,
        "owner_snapshot_cid": _identifier(
            source.get("snapshot_cid"), noun="campaign progress owner snapshot CID"
        ),
        "completion_projection_cid": _identifier(
            source.get("completion_projection_cid"),
            noun="campaign progress completion projection CID",
        ),
        "owner_identity": _json_value(
            source.get("owner_identity"), noun="campaign progress owner identity"
        ),
        "store_generation": _json_value(
            source.get("store_generation"), noun="campaign progress store generation"
        ),
        "artifacts": {
            "json": {
                "path": json_relative.as_posix(),
                "media_type": "application/json",
                "sha256": "sha256:" + hashlib.sha256(json_payload).hexdigest(),
                "size_bytes": len(json_payload),
            },
            "markdown": {
                "path": markdown_relative.as_posix(),
                "media_type": "text/markdown; charset=utf-8",
                "sha256": "sha256:" + hashlib.sha256(markdown_payload).hexdigest(),
                "size_bytes": len(markdown_payload),
            },
        },
    }
    manifest = {**manifest_body, "manifest_cid": content_identity(manifest_body)}
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    parent_descriptor = _open_output_parent(root, relative_manifest.parent)
    try:
        _require_regular_or_absent(
            parent_descriptor,
            relative_manifest.name,
            noun="current progress manifest",
        )
        _publish_immutable(parent_descriptor, json_name, json_payload)
        _publish_immutable(parent_descriptor, markdown_name, markdown_payload)
        _replace_current_manifest(
            parent_descriptor,
            relative_manifest.name,
            manifest_payload,
        )
    except (CampaignProgressDestinationError, CampaignProgressWriteError):
        raise
    except Exception as exc:
        raise CampaignProgressWriteError(
            f"atomic campaign progress publication failed: {exc}"
        ) from exc
    finally:
        os.close(parent_descriptor)
    return MappingProxyType(manifest)


__all__ = [
    "CAMPAIGN_PROGRESS_CURRENT_MANIFEST_SCHEMA",
    "CAMPAIGN_PROGRESS_REPORT_SCHEMA",
    "COMPLETION_EVIDENCE_SCHEMA",
    "CampaignProgressDestinationError",
    "CampaignProgressError",
    "CampaignProgressRendering",
    "CampaignProgressValidationError",
    "CampaignProgressWriteError",
    "INTENT_COMPLETION_PROJECTION_SCHEMA",
    "NON_AUTHORITATIVE_BANNER",
    "PROGRAM_QUALIFICATION_DISPOSITION_SCHEMA",
    "PROTECTED_BOARD_PATHS",
    "TYPED_COMPLETION_PROGRESS_REQUEST_SCHEMA",
    "TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA",
    "build_program_qualification_disposition",
    "render_campaign_progress",
    "write_campaign_progress_outputs",
]
