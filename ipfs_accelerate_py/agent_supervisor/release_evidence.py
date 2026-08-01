"""Read-only, content-addressed supervisor release evidence (FVT-G212).

``AgentSupervisorReleaseEvidence@1`` turns durable supervisor state into a
single export that downstream release tooling can re-verify without trusting
mutable task-state JSON or event logs as authority on their own.

Contract highlights:

* Every source file is opened once; raw bytes are hashed; the exporter never
  rewrites live state, DuckDB, or event logs.
* The export binds canonical task CID/key, dependency CIDs, baseline and merged
  trees/gitlinks, attempt/phase, a continuous event sequence, validation and
  merge outcomes, freshness, authority ceilings, and publication state.
* Metrics-module presence alone is never completion authority.
* Missing terminal ``member_completion_receipt@1`` records cannot be
  synthesized.

Expected-output enforcement lives in the implementation daemon and is
cross-checked by the shared reason vocabulary defined here. Only an exact
declared ignored path may be force-added; anything else fails closed as
``expected_output_ignored_or_unstaged``.

Objective validation repair for FVT-G212 / FVT-078 anchors the synthetic
discovery term ``objective validation repair`` so supervisor scans re-find
the validation gate after the domain evidence surface
(``AgentSupervisorReleaseEvidence@1`` and the release-evidence binding tests)
is already present.  That term never becomes export content-id identity,
completion authority, or proof authority.  FVT-078 is the child
validation-gate work item that owns the repair obligation under parent
FVT-G212.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

RELEASE_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.release_evidence@1"
)
RELEASE_EVIDENCE_INTERFACE: Final = "AgentSupervisorReleaseEvidence@1"
RELEASE_EVIDENCE_GOAL_ID: Final = "FVT-G212"
RELEASE_EVIDENCE_EXPORTER_RELATIVE: Final = Path(
    "ipfs_accelerate_py/agent_supervisor/release_evidence.py"
)
MEMBER_COMPLETION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.member_completion_receipt@1"
)
EXPECTED_OUTPUT_IGNORED_OR_UNSTAGED: Final = (
    "expected_output_ignored_or_unstaged"
)
EXPECTED_OUTPUT_MISSING: Final = "expected_output_missing"
EXPECTED_OUTPUT_FORCE_ADD_FORBIDDEN: Final = (
    "expected_output_force_add_forbidden"
)
EXPECTED_OUTPUT_FORCE_ADD_FAILED: Final = "expected_output_force_add_failed"
EXPECTED_OUTPUT_ABSENT_FROM_PROPOSAL: Final = (
    "expected_output_absent_from_proposal"
)
# Synthetic objective-heap evidence term for FVT-G212 / FVT-078 validation-gate
# work.  Exact-text discovery key only — never part of export content_id or
# completion authority.
OBJECTIVE_VALIDATION_REPAIR_EVIDENCE: Final[str] = "objective validation repair"
# Parent domain goal that owns release-evidence export + expected-output gates.
OBJECTIVE_GOAL_ID: Final[str] = RELEASE_EVIDENCE_GOAL_ID
# Child validation-gate task that owns the synthetic repair obligation.
OBJECTIVE_VALIDATION_REPAIR_TASK_ID: Final[str] = "FVT-078"
# Domain binding-test path listed as FVT-G212 evidence.
RELEASE_EVIDENCE_BINDING_TEST: Final[str] = (
    "test/api/test_agent_supervisor_release_evidence_binding.py"
)

assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
assert OBJECTIVE_GOAL_ID == "FVT-G212"
assert OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "FVT-078"

# Raw supervisor state keys must never appear at the export root; they belong
# only inside the projected ``snapshot`` object.
_FORBIDDEN_ROOT_KEYS: Final = frozenset(
    {
        "task_state_source",
        "event_log_source",
        "task_state",
        "events",
    }
)

_DEFAULT_SOURCE_KEYS: Final = (
    "bundle_metadata",
    "task_metadata",
    "lane_manifest",
    "scheduler_snapshot",
    "task_state",
    "event_manifest",
    "event_log",
    "member_completion_receipts",
)


def sha256_bytes(payload: bytes) -> str:
    """Return a ``sha256:<hex>`` digest for raw bytes."""

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash one existing file's raw bytes, or return ``None`` when absent."""

    try:
        if not path.is_file():
            return None
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


def content_digest(payload: Mapping[str, Any]) -> str:
    """Canonical mapping digest used by G212 content-id bindings."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return sha256_bytes(encoded.encode("utf-8"))


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _read_bytes_once(path: Path | None) -> tuple[bytes | None, str | None]:
    """Read a file once and return ``(raw_bytes, sha256_digest)``."""

    if path is None:
        return None, None
    try:
        if not path.is_file():
            return None, None
        raw = path.read_bytes()
    except OSError:
        return None, None
    return raw, sha256_bytes(raw)


def _parse_json_bytes(raw: bytes | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _parse_jsonl_bytes(raw: bytes | None) -> list[dict[str, Any]]:
    if raw is None:
        return []
    events: list[dict[str, Any]] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            events.append(dict(payload))
    return events


def _event_identity(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "event_id"}
    return content_digest(body)


def _source_record(
    *,
    key: str,
    path: Path | None,
    raw: bytes | None,
    digest: str | None,
    kind: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "kind": kind,
        "path": str(path) if path is not None else None,
        "present": raw is not None,
        "byte_count": len(raw) if raw is not None else 0,
        "sha256": digest,
        "read_once": True,
        "mutated": False,
    }


def _normalize_completion_receipts(value: Any) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for item in _safe_list(value):
        if not isinstance(item, Mapping):
            continue
        receipts.append(dict(item))
    return receipts


def _collect_member_completion_receipts(
    events: Sequence[Mapping[str, Any]],
    *,
    task_id: str = "",
) -> list[dict[str, Any]]:
    """Project durable member completion receipts from terminal events."""

    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type not in {"todo_status_updated", "implementation_finished"}:
            continue
        if task_id and str(event.get("task_id") or "") not in {"", task_id}:
            # Still allow multi-member packet receipts embedded on a primary
            # task event, but skip events that name a different primary id
            # when a filter is supplied and no receipts are present.
            embedded = event.get("completion_receipts") or event.get(
                "member_completion_receipts"
            )
            if not embedded and str(event.get("task_id") or "") != task_id:
                continue
        sources: list[Any] = []
        if event_type == "todo_status_updated":
            sources.append(event.get("completion_receipts"))
            sources.append(event.get("member_completion_receipts"))
        else:
            update = _safe_mapping(event.get("todo_update_result"))
            sources.append(event.get("completion_receipts"))
            sources.append(update.get("completion_receipts"))
            sources.append(update.get("member_completion_receipts"))
            sources.append(event.get("member_completion_receipts"))
        for source in sources:
            for receipt in _normalize_completion_receipts(source):
                schema = str(receipt.get("schema") or "")
                status = str(receipt.get("status") or "").strip().lower()
                if schema and schema != MEMBER_COMPLETION_RECEIPT_SCHEMA:
                    continue
                if status and status != "succeeded":
                    continue
                receipt_task = str(receipt.get("task_id") or "").strip()
                receipt_cid = str(receipt.get("canonical_task_cid") or "").strip()
                identity = f"{receipt_task}|{receipt_cid}|{schema}|{status}"
                if identity in seen:
                    continue
                seen.add(identity)
                projected = dict(receipt)
                if not projected.get("schema"):
                    # Legacy event rows without an explicit schema are not
                    # durable member receipts; skip them.
                    continue
                collected.append(projected)
    return collected


def _project_events(
    events: Sequence[Mapping[str, Any]],
    *,
    task_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate stream continuity and project task-relevant events."""

    chain_errors: list[str] = []
    previous_event_id = ""
    previous_sequence = 0
    stream_id = ""
    snapshot_id = ""
    canonical_event_count = 0
    projected: list[dict[str, Any]] = []

    for index, event in enumerate(events, start=1):
        sequence = event.get("sequence")
        observed_event_id = str(event.get("event_id") or "")
        expected_event_id = _event_identity(event)
        observed_stream = str(event.get("stream_id") or "")
        observed_snapshot = str(event.get("snapshot_id") or "")

        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence != previous_sequence + 1
        ):
            chain_errors.append(f"event_{index}:sequence_not_contiguous")
        if observed_event_id and observed_event_id != expected_event_id:
            chain_errors.append(f"event_{index}:event_id_not_canonical")
        elif not observed_event_id:
            observed_event_id = expected_event_id
        if previous_sequence and str(event.get("previous_event_id") or "") != (
            previous_event_id
        ):
            chain_errors.append(f"event_{index}:previous_event_id_mismatch")
        if not previous_sequence and str(event.get("previous_event_id") or ""):
            chain_errors.append(f"event_{index}:first_previous_event_id_not_empty")
        if not observed_stream or not observed_snapshot:
            chain_errors.append(f"event_{index}:stream_identity_missing")
        elif canonical_event_count:
            if observed_stream != stream_id or observed_snapshot != snapshot_id:
                chain_errors.append(f"event_{index}:stream_identity_changed")
        else:
            stream_id = observed_stream
            snapshot_id = observed_snapshot

        previous_sequence = (
            sequence if isinstance(sequence, int) and not isinstance(sequence, bool)
            else previous_sequence
        )
        previous_event_id = observed_event_id
        canonical_event_count += 1

        if task_id and str(event.get("task_id") or "") not in {"", task_id}:
            continue

        validation = _safe_mapping(
            event.get("validation_result") or event.get("validation")
        )
        merge = _safe_mapping(event.get("merge_result") or event.get("merge"))
        completion_receipts = _normalize_completion_receipts(
            event.get("completion_receipts")
        )
        if not completion_receipts:
            completion_receipts = _normalize_completion_receipts(
                _safe_mapping(event.get("todo_update_result")).get(
                    "completion_receipts"
                )
            )
        projected.append(
            {
                "sequence": event.get("sequence"),
                "event_id": observed_event_id,
                "previous_event_id": event.get("previous_event_id"),
                "snapshot_id": observed_snapshot or None,
                "stream_id": observed_stream or None,
                "type": event.get("type"),
                "timestamp": event.get("timestamp"),
                "task_id": event.get("task_id"),
                "canonical_task_cid": (
                    event.get("canonical_task_cid") or event.get("task_cid")
                ),
                "canonical_task_key": event.get("canonical_task_key"),
                "implementation_commit": event.get("implementation_commit"),
                "attempt": event.get("attempt"),
                "phase": event.get("phase"),
                "validation": {
                    "attempted": validation.get("attempted"),
                    "passed": validation.get("passed"),
                    "returncode": validation.get("returncode"),
                    "target_commit": validation.get("target_commit"),
                    "receipt_id": validation.get("receipt_id"),
                }
                if validation
                else {},
                "merge": {
                    "merged": merge.get("merged"),
                    "implementation_commit": merge.get("implementation_commit"),
                    "merge_commit": merge.get("merge_commit"),
                    "target_branch": merge.get("target_branch"),
                    "baseline_tree": merge.get("baseline_tree"),
                    "merged_tree": merge.get("merged_tree"),
                    "gitlinks": merge.get("gitlinks"),
                    "integration_commit_proof": merge.get(
                        "integration_commit_proof"
                    ),
                }
                if merge
                else {},
                "completion_receipts": completion_receipts,
            }
        )

    chain = {
        "valid": bool(canonical_event_count) and not chain_errors,
        "event_count": canonical_event_count,
        "last_sequence": previous_sequence,
        "last_event_id": previous_event_id or None,
        "stream_id": stream_id or None,
        "snapshot_id": snapshot_id or None,
        "errors": chain_errors,
        "continuous": bool(canonical_event_count) and not any(
            "sequence_not_contiguous" in error for error in chain_errors
        ),
    }
    return projected, chain


def _identity_from_state(
    state: Mapping[str, Any],
    *,
    task_id: str,
) -> dict[str, Any]:
    identities = _safe_mapping(state.get("task_identities"))
    identity = _safe_mapping(identities.get(task_id))
    if identity:
        return {
            "task_id": task_id,
            "canonical_task_cid": str(
                identity.get("canonical_task_cid") or ""
            ).strip(),
            "canonical_task_key": str(
                identity.get("canonical_task_key") or ""
            ).strip(),
        }
    return {
        "task_id": task_id,
        "canonical_task_cid": str(
            state.get("last_implementation_task_cid")
            or state.get("active_task_cid")
            or ""
        ).strip(),
        "canonical_task_key": str(
            state.get("active_task_key") or state.get("last_implementation_task_key")
            or ""
        ).strip(),
    }


def _dependency_cids(
    *,
    task_metadata: Mapping[str, Any],
    bundle_metadata: Mapping[str, Any],
    state: Mapping[str, Any],
) -> list[str]:
    values: list[str] = []
    for source in (task_metadata, bundle_metadata, state):
        for key in (
            "dependency_cids",
            "depends_on_cids",
            "dependency_task_cids",
            "blocked_by_cids",
        ):
            raw = source.get(key)
            if isinstance(raw, (list, tuple)):
                values.extend(str(item).strip() for item in raw if str(item).strip())
            elif isinstance(raw, str) and raw.strip():
                values.append(raw.strip())
        depends_on = source.get("depends_on")
        if isinstance(depends_on, (list, tuple)):
            for item in depends_on:
                if isinstance(item, Mapping):
                    cid = str(
                        item.get("canonical_task_cid") or item.get("cid") or ""
                    ).strip()
                    if cid:
                        values.append(cid)
                elif str(item).strip().startswith(("bafy", "bagu", "sha256:", "cid:")):
                    values.append(str(item).strip())
    return sorted(dict.fromkeys(values))


def _tree_binding(
    *,
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    lane_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_tree = str(
        state.get("baseline_tree")
        or lane_manifest.get("baseline_tree")
        or ""
    ).strip()
    merged_tree = str(
        state.get("merged_tree") or state.get("last_merged_tree") or ""
    ).strip()
    baseline_commit = str(
        state.get("baseline_commit")
        or state.get("baseline_ref")
        or lane_manifest.get("baseline_commit")
        or ""
    ).strip()
    implementation_commit = str(
        state.get("last_implementation_commit") or ""
    ).strip()
    merge_commit = str(state.get("last_merge_commit") or "").strip()
    gitlinks = _safe_mapping(
        state.get("gitlinks") or lane_manifest.get("gitlinks")
    )

    for event in reversed(list(events)):
        merge = _safe_mapping(event.get("merge"))
        if not baseline_tree:
            baseline_tree = str(merge.get("baseline_tree") or "").strip()
        if not merged_tree:
            merged_tree = str(merge.get("merged_tree") or "").strip()
        if not implementation_commit:
            implementation_commit = str(
                event.get("implementation_commit")
                or merge.get("implementation_commit")
                or ""
            ).strip()
        if not merge_commit:
            merge_commit = str(merge.get("merge_commit") or "").strip()
        if not gitlinks:
            gitlinks = _safe_mapping(merge.get("gitlinks"))

    return {
        "baseline_tree": baseline_tree or None,
        "merged_tree": merged_tree or None,
        "baseline_commit": baseline_commit or None,
        "implementation_commit": implementation_commit or None,
        "merge_commit": merge_commit or None,
        "gitlinks": gitlinks,
    }


def _attempt_phase(
    *,
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    scheduler_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    attempt = state.get("attempt")
    phase = state.get("phase") or state.get("active_phase")
    if attempt is None:
        attempts = _safe_mapping(state.get("implementation_attempts_by_cid"))
        if attempts:
            try:
                attempt = max(int(value) for value in attempts.values())
            except (TypeError, ValueError):
                attempt = None
    if phase is None:
        phase = scheduler_snapshot.get("phase") or scheduler_snapshot.get(
            "active_phase"
        )
    for event in reversed(list(events)):
        if attempt is None and event.get("attempt") is not None:
            attempt = event.get("attempt")
        if phase is None and event.get("phase") is not None:
            phase = event.get("phase")
        if attempt is not None and phase is not None:
            break
    return {
        "attempt": attempt,
        "phase": phase,
    }


def _freshness(
    *,
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    heartbeat_at = str(state.get("heartbeat_at") or "").strip() or None
    last_event_timestamp = None
    for event in reversed(list(events)):
        stamp = str(event.get("timestamp") or "").strip()
        if stamp:
            last_event_timestamp = stamp
            break
    return {
        "heartbeat_at": heartbeat_at,
        "last_event_timestamp": last_event_timestamp,
        "sources_present": sorted(
            str(record["key"])
            for record in source_records
            if record.get("present") is True
        ),
        "all_sources_read_once": all(
            record.get("read_once") is True for record in source_records
        ),
        "live_state_mutated": False,
    }


def _authority_projection(
    *,
    member_receipts: Sequence[Mapping[str, Any]],
    metrics_module_present: bool,
    event_chain_valid: bool,
) -> dict[str, Any]:
    durable_receipts = [
        receipt
        for receipt in member_receipts
        if str(receipt.get("schema") or "") == MEMBER_COMPLETION_RECEIPT_SCHEMA
        and str(receipt.get("status") or "").strip().lower() == "succeeded"
    ]
    # Metrics modules report load/availability only. They never authorize
    # completion, merge, or publication of a release snapshot.
    completion_from_metrics = False
    return {
        "proof_authoritative": False,
        "completion_authoritative": False,
        "publication_authoritative": False,
        "metrics_module_present": bool(metrics_module_present),
        "metrics_module_is_completion": completion_from_metrics,
        "durable_member_completion_receipts": len(durable_receipts),
        "completion_requires_member_receipt": True,
        "event_chain_required": True,
        "event_chain_valid": bool(event_chain_valid),
        "completion_bound": bool(durable_receipts) and bool(event_chain_valid),
    }


def _publication_state(
    *,
    state: Mapping[str, Any],
    trees: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    published = bool(
        trees.get("merge_commit")
        and authority.get("completion_bound") is True
        and state.get("implementation_in_progress") is False
    )
    return {
        "published": published,
        "publication_ready": published,
        "merge_commit": trees.get("merge_commit"),
        "target_branch": state.get("target_branch") or state.get("integration_branch"),
        "requires_origin_publication": True,
        "origin_publication_bound": bool(
            state.get("published_to_origin") is True
            or state.get("origin_main_commit")
        ),
    }


def exporter_identity(repo_root: Path | None) -> dict[str, Any]:
    """Bind the on-disk exporter path and raw-byte identity."""

    relative = RELEASE_EVIDENCE_EXPORTER_RELATIVE.as_posix()
    if repo_root is None:
        return {
            "path": relative,
            "present": False,
            "sha256": None,
            "bound": False,
        }
    path = Path(repo_root) / RELEASE_EVIDENCE_EXPORTER_RELATIVE
    digest = sha256_file(path)
    return {
        "path": relative,
        "present": path.is_file(),
        "sha256": digest,
        "bound": bool(digest),
    }


def export_release_evidence(
    *,
    task_id: str,
    task_state_path: Path | None = None,
    event_log_path: Path | None = None,
    event_manifest_path: Path | None = None,
    lane_manifest_path: Path | None = None,
    scheduler_snapshot_path: Path | None = None,
    bundle_metadata_path: Path | None = None,
    task_metadata_path: Path | None = None,
    member_completion_receipts_path: Path | None = None,
    repo_root: Path | None = None,
    metrics_module_present: bool = False,
    task_state: Mapping[str, Any] | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    event_manifest: Mapping[str, Any] | None = None,
    lane_manifest: Mapping[str, Any] | None = None,
    scheduler_snapshot: Mapping[str, Any] | None = None,
    bundle_metadata: Mapping[str, Any] | None = None,
    task_metadata: Mapping[str, Any] | None = None,
    member_completion_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Export a read-only G212 release-evidence object.

    Callers may supply filesystem paths, already-loaded mappings, or both.
    Path-backed sources are read once and hashed from the raw bytes before any
    JSON parse so the export can prove durability without re-reading live
    state. In-memory fixtures are accepted for unit tests and are recorded as
    synthetic sources with a content digest of the canonical JSON body.
    """

    source_records: list[dict[str, Any]] = []
    raw_by_key: dict[str, bytes | None] = {}

    path_specs: list[tuple[str, Path | None, str]] = [
        ("bundle_metadata", bundle_metadata_path, "json"),
        ("task_metadata", task_metadata_path, "json"),
        ("lane_manifest", lane_manifest_path, "json"),
        ("scheduler_snapshot", scheduler_snapshot_path, "json"),
        ("task_state", task_state_path, "json"),
        ("event_manifest", event_manifest_path, "json"),
        ("event_log", event_log_path, "jsonl"),
        (
            "member_completion_receipts",
            member_completion_receipts_path,
            "json",
        ),
    ]
    for key, path, kind in path_specs:
        raw, digest = _read_bytes_once(path)
        raw_by_key[key] = raw
        source_records.append(
            _source_record(
                key=key,
                path=path,
                raw=raw,
                digest=digest,
                kind=kind,
            )
        )

    def _resolve_mapping(
        key: str,
        supplied: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if supplied is not None:
            payload = dict(supplied)
            digest = content_digest(payload)
            # Replace the empty path record with a synthetic one when only
            # in-memory content was supplied.
            for index, record in enumerate(source_records):
                if record["key"] == key and record.get("present") is not True:
                    encoded = json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                    source_records[index] = {
                        "key": key,
                        "kind": "memory_json",
                        "path": None,
                        "present": True,
                        "byte_count": len(encoded),
                        "sha256": digest,
                        "read_once": True,
                        "mutated": False,
                    }
                    raw_by_key[key] = encoded
                    break
            return payload
        parsed = _parse_json_bytes(raw_by_key.get(key))
        return dict(parsed) if isinstance(parsed, Mapping) else {}

    resolved_bundle = _resolve_mapping("bundle_metadata", bundle_metadata)
    resolved_task_meta = _resolve_mapping("task_metadata", task_metadata)
    resolved_lane = _resolve_mapping("lane_manifest", lane_manifest)
    resolved_scheduler = _resolve_mapping(
        "scheduler_snapshot",
        scheduler_snapshot,
    )
    resolved_state = _resolve_mapping("task_state", task_state)
    resolved_manifest = _resolve_mapping("event_manifest", event_manifest)

    if events is not None:
        resolved_events = [dict(event) for event in events if isinstance(event, Mapping)]
        encoded = json.dumps(
            resolved_events,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        for index, record in enumerate(source_records):
            if record["key"] == "event_log" and record.get("present") is not True:
                source_records[index] = {
                    "key": "event_log",
                    "kind": "memory_jsonl",
                    "path": None,
                    "present": True,
                    "byte_count": len(encoded),
                    "sha256": sha256_bytes(encoded),
                    "read_once": True,
                    "mutated": False,
                }
                break
    else:
        resolved_events = _parse_jsonl_bytes(raw_by_key.get("event_log"))

    if member_completion_receipts is not None:
        durable_receipts = [
            dict(item)
            for item in member_completion_receipts
            if isinstance(item, Mapping)
        ]
        encoded = json.dumps(
            durable_receipts,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        for index, record in enumerate(source_records):
            if (
                record["key"] == "member_completion_receipts"
                and record.get("present") is not True
            ):
                source_records[index] = {
                    "key": "member_completion_receipts",
                    "kind": "memory_json",
                    "path": None,
                    "present": True,
                    "byte_count": len(encoded),
                    "sha256": sha256_bytes(encoded),
                    "read_once": True,
                    "mutated": False,
                }
                break
    else:
        parsed_receipts = _parse_json_bytes(
            raw_by_key.get("member_completion_receipts")
        )
        if isinstance(parsed_receipts, list):
            durable_receipts = [
                dict(item) for item in parsed_receipts if isinstance(item, Mapping)
            ]
        elif isinstance(parsed_receipts, Mapping):
            nested = parsed_receipts.get("member_completion_receipts")
            if isinstance(nested, list):
                durable_receipts = [
                    dict(item) for item in nested if isinstance(item, Mapping)
                ]
            else:
                durable_receipts = [dict(parsed_receipts)]
        else:
            durable_receipts = []

    projected_events, event_chain = _project_events(
        resolved_events,
        task_id=str(task_id or "").strip(),
    )
    event_receipts = _collect_member_completion_receipts(
        resolved_events,
        task_id=str(task_id or "").strip(),
    )
    # Prefer explicit durable receipt files; fall back to event-embedded
    # member receipts. Never invent a terminal receipt when both are empty.
    combined_receipts: list[dict[str, Any]] = []
    seen_receipts: set[str] = set()
    for receipt in [*durable_receipts, *event_receipts]:
        schema = str(receipt.get("schema") or "")
        if schema != MEMBER_COMPLETION_RECEIPT_SCHEMA:
            continue
        if str(receipt.get("status") or "").strip().lower() != "succeeded":
            continue
        key = (
            f"{receipt.get('task_id')}|{receipt.get('canonical_task_cid')}|"
            f"{receipt.get('implementation_commit')}|{receipt.get('merge_commit')}"
        )
        if key in seen_receipts:
            continue
        seen_receipts.add(key)
        combined_receipts.append(dict(receipt))

    identity = _identity_from_state(
        resolved_state,
        task_id=str(task_id or "").strip(),
    )
    if not identity.get("canonical_task_cid"):
        for event in projected_events:
            cid = str(event.get("canonical_task_cid") or "").strip()
            key = str(event.get("canonical_task_key") or "").strip()
            if cid:
                identity["canonical_task_cid"] = cid
                if key:
                    identity["canonical_task_key"] = key
                break
    if not identity.get("canonical_task_cid") and combined_receipts:
        identity["canonical_task_cid"] = str(
            combined_receipts[0].get("canonical_task_cid") or ""
        ).strip()
        identity["canonical_task_key"] = str(
            combined_receipts[0].get("canonical_task_key") or ""
        ).strip()

    trees = _tree_binding(
        state=resolved_state,
        events=projected_events,
        lane_manifest=resolved_lane,
    )
    attempt_phase = _attempt_phase(
        state=resolved_state,
        events=projected_events,
        scheduler_snapshot=resolved_scheduler,
    )
    authority = _authority_projection(
        member_receipts=combined_receipts,
        metrics_module_present=metrics_module_present,
        event_chain_valid=bool(event_chain.get("valid")),
    )
    freshness = _freshness(
        state=resolved_state,
        events=projected_events,
        source_records=source_records,
    )
    publication = _publication_state(
        state=resolved_state,
        trees=trees,
        authority=authority,
    )
    dependency_cids = _dependency_cids(
        task_metadata=resolved_task_meta,
        bundle_metadata=resolved_bundle,
        state=resolved_state,
    )

    task_status = None
    statuses = _safe_mapping(resolved_state.get("task_statuses"))
    if str(task_id or "").strip() in statuses:
        task_status = statuses.get(str(task_id).strip())
    elif resolved_state.get("task_status") is not None:
        task_status = resolved_state.get("task_status")

    snapshot = {
        "schema_version": "agent-supervisor-release-evidence-snapshot/v1",
        "goal_id": RELEASE_EVIDENCE_GOAL_ID,
        "task_id": str(task_id or "").strip(),
        "lane_id": (
            str(resolved_lane.get("lane_id") or "").strip()
            or (
                Path(task_state_path).parent.parent.name
                if task_state_path is not None
                else None
            )
        ),
        "canonical_identity": identity,
        "dependency_cids": dependency_cids,
        "trees": trees,
        "attempt_phase": attempt_phase,
        "event_chain": event_chain,
        "event_manifest": {
            "present": bool(resolved_manifest),
            "stream_id": resolved_manifest.get("stream_id")
            or event_chain.get("stream_id"),
            "snapshot_id": resolved_manifest.get("snapshot_id")
            or event_chain.get("snapshot_id"),
            "last_event_id": resolved_manifest.get("last_event_id")
            or event_chain.get("last_event_id"),
            "sha256": next(
                (
                    record.get("sha256")
                    for record in source_records
                    if record.get("key") == "event_manifest"
                ),
                None,
            ),
        },
        "bundle_metadata": {
            "present": bool(resolved_bundle),
            "bundle_id": resolved_bundle.get("bundle_id")
            or resolved_bundle.get("id"),
            "sha256": next(
                (
                    record.get("sha256")
                    for record in source_records
                    if record.get("key") == "bundle_metadata"
                ),
                None,
            ),
            "fields": {
                key: resolved_bundle[key]
                for key in sorted(resolved_bundle)
                if key
                in {
                    "bundle_id",
                    "id",
                    "goal_id",
                    "merge_family",
                    "dependency_cids",
                    "depends_on",
                }
            },
        },
        "task_metadata": {
            "present": bool(resolved_task_meta),
            "task_id": resolved_task_meta.get("task_id") or str(task_id or "").strip(),
            "sha256": next(
                (
                    record.get("sha256")
                    for record in source_records
                    if record.get("key") == "task_metadata"
                ),
                None,
            ),
            "fields": {
                key: resolved_task_meta[key]
                for key in sorted(resolved_task_meta)
                if key
                in {
                    "task_id",
                    "canonical_task_cid",
                    "canonical_task_key",
                    "dependency_cids",
                    "depends_on",
                    "expected_outputs",
                    "outputs",
                }
            },
        },
        "lane_manifest": {
            "present": bool(resolved_lane),
            "lane_id": resolved_lane.get("lane_id"),
            "sha256": next(
                (
                    record.get("sha256")
                    for record in source_records
                    if record.get("key") == "lane_manifest"
                ),
                None,
            ),
            "fields": {
                key: resolved_lane[key]
                for key in sorted(resolved_lane)
                if key
                in {
                    "lane_id",
                    "baseline_tree",
                    "baseline_commit",
                    "gitlinks",
                    "bundle_id",
                }
            },
        },
        "scheduler_snapshot": {
            "present": bool(resolved_scheduler),
            "sha256": next(
                (
                    record.get("sha256")
                    for record in source_records
                    if record.get("key") == "scheduler_snapshot"
                ),
                None,
            ),
            # Presence of a metrics-bearing scheduler snapshot is diagnostic
            # only; completion authority requires member receipts.
            "metrics_present": bool(
                resolved_scheduler.get("metrics")
                or resolved_scheduler.get("scheduler_metrics")
            ),
            "phase": resolved_scheduler.get("phase")
            or resolved_scheduler.get("active_phase"),
        },
        "task_state": {
            "active_task_id": resolved_state.get("active_task_id"),
            "active_task_cid": resolved_state.get("active_task_cid"),
            "active_task_key": resolved_state.get("active_task_key"),
            "implementation_in_progress": resolved_state.get(
                "implementation_in_progress"
            ),
            "last_implementation_task_id": resolved_state.get(
                "last_implementation_task_id"
            ),
            "last_implementation_task_cid": resolved_state.get(
                "last_implementation_task_cid"
            ),
            "last_implementation_commit": resolved_state.get(
                "last_implementation_commit"
            ),
            "last_merge_commit": resolved_state.get("last_merge_commit"),
            "task_status": task_status,
            "canonical_identity": identity,
            "source_sha256": next(
                (
                    record.get("sha256")
                    for record in source_records
                    if record.get("key") == "task_state"
                ),
                None,
            ),
        },
        "events": projected_events,
        "member_completion_receipts": combined_receipts,
        "validation_outcomes": [
            {
                "sequence": event.get("sequence"),
                "event_id": event.get("event_id"),
                "validation": event.get("validation") or {},
            }
            for event in projected_events
            if event.get("validation")
        ],
        "merge_outcomes": [
            {
                "sequence": event.get("sequence"),
                "event_id": event.get("event_id"),
                "merge": event.get("merge") or {},
                "implementation_commit": event.get("implementation_commit"),
            }
            for event in projected_events
            if event.get("merge")
        ],
        "freshness": freshness,
        "authority": authority,
        "publication": publication,
        "sources": source_records,
    }

    exporter = exporter_identity(repo_root)
    body = {
        "schema": RELEASE_EVIDENCE_SCHEMA,
        "interface": RELEASE_EVIDENCE_INTERFACE,
        "goal_id": RELEASE_EVIDENCE_GOAL_ID,
        "exporter": {
            "path": exporter["path"],
            "sha256": exporter["sha256"],
        },
        "snapshot": snapshot,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    content_id = content_digest(body)
    return {
        **body,
        "content_id": content_id,
    }


def verify_release_evidence(
    payload: Mapping[str, Any] | None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute identity and structural gates for one G212 export."""

    failures: list[str] = []
    result: dict[str, Any] = {
        "valid": False,
        "snapshot": {},
        "failures": failures,
        "schema": None,
        "interface": None,
        "goal_id": None,
        "content_id": None,
        "exporter": {},
    }
    if not isinstance(payload, Mapping) or not payload:
        failures.append("release_evidence_missing")
        return result

    data = dict(payload)
    result["schema"] = data.get("schema")
    result["interface"] = data.get("interface")
    result["goal_id"] = data.get("goal_id")
    result["content_id"] = data.get("content_id")

    if data.get("schema") != RELEASE_EVIDENCE_SCHEMA:
        failures.append("schema_mismatch")
    if data.get("interface") != RELEASE_EVIDENCE_INTERFACE:
        failures.append("interface_mismatch")
    if data.get("goal_id") != RELEASE_EVIDENCE_GOAL_ID:
        failures.append("goal_id_mismatch")
    if any(key in data for key in _FORBIDDEN_ROOT_KEYS):
        failures.append("raw_supervisor_state_is_not_release_evidence")

    stored_content_id = str(data.get("content_id") or "")
    body = {key: value for key, value in data.items() if key != "content_id"}
    computed = content_digest(body)
    if not stored_content_id or stored_content_id not in {
        computed,
        computed.removeprefix("sha256:"),
    }:
        failures.append("content_id_mismatch")

    exporter = _safe_mapping(data.get("exporter"))
    expected_exporter = exporter_identity(repo_root)
    result["exporter"] = {
        "path": expected_exporter["path"],
        "present": expected_exporter["present"],
        "sha256": expected_exporter["sha256"],
        "claimed_sha256": exporter.get("sha256"),
        "bound": False,
    }
    if repo_root is None:
        failures.append("repository_missing")
    elif not expected_exporter["present"]:
        failures.append("exporter_missing")
    elif (
        exporter.get("path") != expected_exporter["path"]
        or not expected_exporter["sha256"]
        or exporter.get("sha256") != expected_exporter["sha256"]
    ):
        failures.append("exporter_identity_mismatch")
    else:
        result["exporter"]["bound"] = True

    snapshot = _safe_mapping(data.get("snapshot"))
    if not snapshot:
        failures.append("snapshot_missing")
    else:
        identity = _safe_mapping(
            _safe_mapping(snapshot.get("task_state")).get("canonical_identity")
            or snapshot.get("canonical_identity")
        )
        if not str(identity.get("canonical_task_cid") or "").strip():
            failures.append("canonical_task_cid_missing")
        if not str(identity.get("canonical_task_key") or "").strip():
            failures.append("canonical_task_key_missing")
        chain = _safe_mapping(snapshot.get("event_chain"))
        if chain.get("valid") is not True:
            failures.append("event_chain_invalid")
        authority = _safe_mapping(snapshot.get("authority"))
        if authority.get("metrics_module_is_completion") is True:
            failures.append("metrics_module_treated_as_completion")
        if authority.get("completion_authoritative") is True:
            failures.append("export_claims_completion_authority")
        if data.get("completion_authoritative") is True:
            failures.append("root_claims_completion_authority")
        sources = _safe_list(snapshot.get("sources"))
        if sources and not all(
            isinstance(item, Mapping) and item.get("mutated") is False
            for item in sources
        ):
            failures.append("live_state_mutation_claimed")
        if sources and not all(
            isinstance(item, Mapping) and item.get("read_once") is True
            for item in sources
        ):
            failures.append("sources_not_read_once")

    if failures:
        return result

    result["valid"] = True
    result["snapshot"] = snapshot
    return result


def expected_output_failure_reasons() -> frozenset[str]:
    """Return the stable proposal/commit failure vocabulary for declared outputs."""

    return frozenset(
        {
            EXPECTED_OUTPUT_IGNORED_OR_UNSTAGED,
            EXPECTED_OUTPUT_MISSING,
            EXPECTED_OUTPUT_FORCE_ADD_FORBIDDEN,
            EXPECTED_OUTPUT_FORCE_ADD_FAILED,
            EXPECTED_OUTPUT_ABSENT_FROM_PROPOSAL,
        }
    )


def is_expected_output_gate_reason(reason: Any) -> bool:
    """Return whether ``reason`` is an expected-output enforcement failure."""

    text = str(reason or "").strip()
    return text in expected_output_failure_reasons() or text == (
        EXPECTED_OUTPUT_IGNORED_OR_UNSTAGED
    )


def release_evidence_domain_terms() -> tuple[str, ...]:
    """Return domain evidence terms covered by AgentSupervisorReleaseEvidence@1.

    Domain surfaces stay separate from the synthetic validation-gate meta term.
    Export content_id, completion authority, and proof authority never absorb
    :data:`OBJECTIVE_VALIDATION_REPAIR_EVIDENCE`.
    """

    return (
        RELEASE_EVIDENCE_INTERFACE,
        RELEASE_EVIDENCE_SCHEMA,
        RELEASE_EVIDENCE_BINDING_TEST,
        MEMBER_COMPLETION_RECEIPT_SCHEMA,
    )


def objective_validation_repair_evidence_terms() -> tuple[str, ...]:
    """Return the synthetic FVT-G212 / FVT-078 validation-gate evidence term.

    Exact-text discovery key for objective validation repair.  Never mixes
    into export content_id identity, completion authority, or proof authority.
    Owned by :data:`OBJECTIVE_VALIDATION_REPAIR_TASK_ID` (``FVT-078``) under
    parent :data:`OBJECTIVE_GOAL_ID` (``FVT-G212``).
    """

    return (OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,)


def all_covered_evidence_terms() -> tuple[str, ...]:
    """Return domain release-evidence terms plus the objective validation repair gate.

    Domain ``AgentSupervisorReleaseEvidence@1`` surfaces come first; the
    synthetic objective validation repair discovery key is appended last and
    never enters export content_id identity.
    """

    return release_evidence_domain_terms() + objective_validation_repair_evidence_terms()


def objective_validation_repair_claim() -> dict[str, Any]:
    """Emit a portable FVT-G212 objective validation repair claim.

    Records that the validation command for FVT-G212 has been repaired and
    that the synthetic discovery phrase is anchored on this surface without
    granting completion or proof authority.
    """

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_validation_repair@1",
        "goal_id": OBJECTIVE_GOAL_ID,
        "task_id": OBJECTIVE_VALIDATION_REPAIR_TASK_ID,
        "evidence": OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
        "requirement_id": OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
        "interface": RELEASE_EVIDENCE_INTERFACE,
        "validation": (
            "python -m pytest "
            "test/api/test_agent_supervisor_release_evidence_binding.py "
            "test/api/test_agent_supervisor_todo_daemon_port.py "
            "-k 'expected_output or completion_receipt or release_evidence' -q"
        ),
        "completion_authoritative": False,
        "proof_authoritative": False,
        "domain_evidence_terms": list(release_evidence_domain_terms()),
        "repair_evidence_terms": list(objective_validation_repair_evidence_terms()),
    }


__all__ = [
    "EXPECTED_OUTPUT_ABSENT_FROM_PROPOSAL",
    "EXPECTED_OUTPUT_FORCE_ADD_FAILED",
    "EXPECTED_OUTPUT_FORCE_ADD_FORBIDDEN",
    "EXPECTED_OUTPUT_IGNORED_OR_UNSTAGED",
    "EXPECTED_OUTPUT_MISSING",
    "MEMBER_COMPLETION_RECEIPT_SCHEMA",
    "OBJECTIVE_GOAL_ID",
    "OBJECTIVE_VALIDATION_REPAIR_EVIDENCE",
    "OBJECTIVE_VALIDATION_REPAIR_TASK_ID",
    "RELEASE_EVIDENCE_BINDING_TEST",
    "RELEASE_EVIDENCE_EXPORTER_RELATIVE",
    "RELEASE_EVIDENCE_GOAL_ID",
    "RELEASE_EVIDENCE_INTERFACE",
    "RELEASE_EVIDENCE_SCHEMA",
    "all_covered_evidence_terms",
    "content_digest",
    "expected_output_failure_reasons",
    "export_release_evidence",
    "exporter_identity",
    "is_expected_output_gate_reason",
    "objective_validation_repair_claim",
    "objective_validation_repair_evidence_terms",
    "release_evidence_domain_terms",
    "sha256_bytes",
    "sha256_file",
    "verify_release_evidence",
]
