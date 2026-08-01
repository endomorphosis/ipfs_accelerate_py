"""Independent, read-only Codex review for exact post-merge acceptance.

This module is deliberately narrower than the implementation provider router.
It reviews an implementation that is already merged, has already passed fresh
post-merge validation, and is bound to one exact Git commit/tree and diff.
Provider output remains evidence only: the returned gate envelope still has to
pass :mod:`authoritative_completion`, and neither the provider response nor its
receipt carries write, proof, or completion authority.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ..proof.formal_verification_contracts import content_identity
from ..runtime import event_log as _event_log_runtime
from ..validation.scope_adjudication import (
    verified_scope_adjudication_receipt,
)
from .authoritative_completion import (
    POST_MERGE_VALIDATION_EVIDENCE_SCHEMA,
    bound_gate_evidence,
)
from .contract_packet_provider_router import ReviewPresence
from .llm import (
    LLM_CHILD_ENVELOPE_VERSION,
    LLM_CHILD_RESULT_SCHEMA,
    LlmRouterInvocation,
    call_llm_router_with_receipt,
)
from .llm_defaults import DEFAULT_CODEX_MODEL

POST_MERGE_INDEPENDENT_REVIEW_EVENT = "post_merge_independent_review_admitted"
POST_MERGE_INDEPENDENT_REVIEW_DENIED_EVENT = "post_merge_independent_review_denied"
POST_MERGE_INDEPENDENT_REVIEW_FAILED_EVENT = "post_merge_independent_review_failed"
POST_MERGE_INDEPENDENT_REVIEW_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-independent-review-request@2"
)
POST_MERGE_INDEPENDENT_REVIEW_RESPONSE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-independent-review-response@1"
)
POST_MERGE_INDEPENDENT_REVIEW_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-independent-review-receipt@2"
)
POST_MERGE_REVIEWER_EXECUTION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "post-merge-reviewer-execution-receipt@1"
)
VERIFIED_IMPLEMENTER_PROVENANCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "verified-implementation-provider-provenance@1"
)

CODEX_REVIEWER_PROVIDER = "codex_cli"
ALLOWED_IMPLEMENTER_PROVIDERS = frozenset(
    {
        "grok_cli",
        "grok-implement",
        "xai",
    }
)
MAX_REVIEW_DIFF_BYTES = 512 * 1024
MAX_REVIEW_PROMPT_BYTES = 768 * 1024
MAX_REVIEW_RESPONSE_BYTES = 64 * 1024
MAX_REVIEW_FINDINGS = 64
MAX_REVIEW_FINDING_TEXT_BYTES = 2 * 1024
MAX_IMPLEMENTER_LOG_BYTES = 16 * 1024 * 1024
IMPLEMENTER_LOG_BINDING_SCOPE = "review_time_live_artifact"
_FULL_OBJECT_ID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SAFE_TASK_ID = re.compile(r"[^a-z0-9._-]+")
_LIVE_PRODUCTION_REVIEW_SEAL = object()
_CANONICAL_EVENT_ENVELOPE_FIELDS = frozenset(
    {
        "event_id",
        "previous_event_id",
        "sequence",
        "snapshot_id",
        "stream_id",
        "timestamp",
    }
)


class PostMergeReviewError(ValueError):
    """A typed, fail-closed post-merge review rejection."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(detail or reason_code)
        self.reason_code = str(reason_code)
        self.detail = str(detail or reason_code)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ReviewerInvocation:
    """Result from the independent reviewer transport.

    Production constructs this from the isolated ``llm_router`` Codex child.
    Tests may dependency-inject the callable, but must still supply the same
    typed child receipt so the normal verifier exercises the production trust
    boundary.
    """

    provider_id: str
    response_text: str
    transport_receipt: Mapping[str, Any]
    sandbox: str = "read-only"


@dataclass(frozen=True)
class VerifiedImplementerProvenance:
    """Content-bound projection of durable events and their current log.

    The start/finish event identities durably bind ``log_path``. Legacy
    events do not bind the file's bytes, so ``log_sha256`` and ``log_bytes``
    are deliberately labelled as a review-time live-artifact observation,
    never as event-anchored evidence.
    """

    task_id: str
    implementation_attempt: int
    provider_id: str
    runner: str
    grok_binary: str
    model: str
    implementation_commit: str
    branch: str
    log_path: str
    log_bytes: int
    log_sha256: str
    log_binding_scope: str
    log_event_anchored: bool
    started_event_id: str
    started_event_sequence: int
    finished_event_id: str
    finished_event_sequence: int
    source_stream_id: str
    source_snapshot_id: str
    provenance_id: str
    schema: str = VERIFIED_IMPLEMENTER_PROVENANCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "implementation_attempt": self.implementation_attempt,
            "provider_id": self.provider_id,
            "runner": self.runner,
            "grok_binary": self.grok_binary,
            "model": self.model,
            "implementation_commit": self.implementation_commit,
            "branch": self.branch,
            "log_path": self.log_path,
            "log_bytes": self.log_bytes,
            "log_sha256": self.log_sha256,
            "log_binding_scope": self.log_binding_scope,
            "log_event_anchored": self.log_event_anchored,
            "started_event_id": self.started_event_id,
            "started_event_sequence": self.started_event_sequence,
            "finished_event_id": self.finished_event_id,
            "finished_event_sequence": self.finished_event_sequence,
            "source_stream_id": self.source_stream_id,
            "source_snapshot_id": self.source_snapshot_id,
            "provenance_id": self.provenance_id,
        }


ReviewerCallable = Callable[
    [str, Mapping[str, Any]],
    ReviewerInvocation,
]


@dataclass(frozen=True)
class ReceiptVerification:
    valid: bool
    reason_code: str
    detail: str = ""
    admitted: bool = False
    receipt_id: str = ""


@dataclass(frozen=True)
class PostMergeReviewOutcome:
    admitted: bool
    reason_code: str
    detail: str = ""
    receipt: Mapping[str, Any] = field(default_factory=dict)
    receipt_path: str = ""
    event: Mapping[str, Any] = field(default_factory=dict)
    retryable: bool = False
    acceptance_pending: bool = True
    _gate_evidence: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _producer_seal: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _bound_task_id: str = field(default="", repr=False, compare=False)
    _bound_task_binding_id: str = field(
        default="",
        repr=False,
        compare=False,
    )
    _bound_canonical_task_key: str = field(
        default="",
        repr=False,
        compare=False,
    )
    _bound_canonical_task_cid: str = field(
        default="",
        repr=False,
        compare=False,
    )
    _bound_board_namespace: str = field(
        default="",
        repr=False,
        compare=False,
    )
    _bound_implementation_commit: str = field(
        default="",
        repr=False,
        compare=False,
    )
    _bound_merge_commit: str = field(default="", repr=False, compare=False)
    _bound_repository_tree_id: str = field(
        default="",
        repr=False,
        compare=False,
    )
    _bound_review_receipt_id: str = field(
        default="",
        repr=False,
        compare=False,
    )
    _receipt_canonical: bytes = field(
        default=b"",
        repr=False,
        compare=False,
    )
    _event_payload_canonical: bytes = field(
        default=b"",
        repr=False,
        compare=False,
    )


def verified_implementer_provenance_from_events(
    started_event: Mapping[str, Any],
    finished_event: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_task_id: str,
    expected_implementation_attempt: int,
    expected_implementation_commit: str,
) -> VerifiedImplementerProvenance:
    """Verify and project a hash-bound implementation start/finish pair."""

    if not isinstance(started_event, Mapping) or not isinstance(
        finished_event, Mapping
    ):
        raise PostMergeReviewError(
            "implementer_provenance_event_missing",
            "implementer provenance requires durable start and finish events",
        )
    event_ids: list[str] = []
    for event in (started_event, finished_event):
        body = dict(event)
        event_id = str(body.pop("event_id", "") or "")
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if event_id != "sha256:" + hashlib.sha256(encoded).hexdigest():
            raise PostMergeReviewError(
                "implementer_provenance_event_invalid",
                "implementation provider event identity is invalid",
            )
        event_ids.append(event_id)
    command = started_event.get("command")
    branch = str(started_event.get("branch") or "")
    log_path = str(started_event.get("log_path") or "")
    if (
        str(started_event.get("type") or "") != "implementation_started"
        or str(started_event.get("execution_mode") or "") != "model-assisted"
        or str(started_event.get("task_id") or "") != expected_task_id
        or int(started_event.get("attempt") or 0)
        != int(expected_implementation_attempt)
        or str(finished_event.get("type") or "") != "implementation_finished"
        or str(finished_event.get("task_id") or "") != expected_task_id
        or int(finished_event.get("attempt") or 0)
        != int(expected_implementation_attempt)
        or isinstance(finished_event.get("returncode"), bool)
        or finished_event.get("returncode") != 0
        or str(finished_event.get("implementation_commit") or "")
        != expected_implementation_commit
        or str(finished_event.get("branch") or "") != branch
        or str(finished_event.get("log_path") or "") != log_path
        or str(finished_event.get("stream_id") or "")
        != str(started_event.get("stream_id") or "")
        or str(finished_event.get("snapshot_id") or "")
        != str(started_event.get("snapshot_id") or "")
        or int(finished_event.get("sequence") or 0)
        <= int(started_event.get("sequence") or 0)
        or not branch
        or not log_path
        or not isinstance(command, Sequence)
        or isinstance(command, (str, bytes, bytearray))
        or not all(isinstance(item, str) for item in command)
    ):
        raise PostMergeReviewError(
            "implementer_provenance_event_invalid",
            "implementation provider event is not hash/task/attempt bound",
        )
    command_items = list(command)
    runner = next(
        (
            item
            for item in command_items
            if Path(item).name == "grok_cli_runner.py"
        ),
        "",
    )
    try:
        model = command_items[command_items.index("--model") + 1]
        grok_binary = command_items[command_items.index("--grok-bin") + 1]
    except (ValueError, IndexError) as exc:
        raise PostMergeReviewError(
            "implementer_provenance_command_invalid",
            "implementation event does not identify the Grok runner/model",
        ) from exc
    if (
        not Path(runner).is_absolute()
        or Path(runner).name != "grok_cli_runner.py"
        or not model.strip()
        or not Path(grok_binary).is_absolute()
        or Path(grok_binary).name != "grok"
    ):
        raise PostMergeReviewError(
            "implementer_provenance_command_invalid",
            "implementation event does not bind an explicit grok_cli provider",
        )
    root = Path(repo_root).resolve()
    log_file = (root / log_path).resolve()
    try:
        log_file.relative_to(root)
    except ValueError as exc:
        raise PostMergeReviewError(
            "implementer_log_path_invalid",
            "implementation log path escapes the repository",
        ) from exc
    if not log_file.is_file():
        raise PostMergeReviewError(
            "implementer_log_unavailable",
            "implementation log required by provider provenance is unavailable",
        )
    try:
        declared_log_bytes = log_file.stat().st_size
        if (
            declared_log_bytes < 0
            or declared_log_bytes > MAX_IMPLEMENTER_LOG_BYTES
        ):
            raise PostMergeReviewError(
                "implementer_log_size_invalid",
                "implementation log exceeds the review-time artifact bound",
            )
        log_payload = log_file.read_bytes()
    except OSError as exc:
        raise PostMergeReviewError(
            "implementer_log_unavailable",
            "implementation log could not be read for review-time binding",
        ) from exc
    if (
        len(log_payload) != declared_log_bytes
        or len(log_payload) > MAX_IMPLEMENTER_LOG_BYTES
    ):
        raise PostMergeReviewError(
            "implementer_log_changed_during_binding",
            "implementation log changed while its live artifact was bound",
        )
    material = {
        "schema": VERIFIED_IMPLEMENTER_PROVENANCE_SCHEMA,
        "task_id": expected_task_id,
        "implementation_attempt": int(expected_implementation_attempt),
        "provider_id": "grok_cli",
        "runner": runner,
        "grok_binary": grok_binary,
        "model": model,
        "implementation_commit": expected_implementation_commit,
        "branch": branch,
        "log_path": log_path,
        "log_bytes": len(log_payload),
        "log_sha256": hashlib.sha256(log_payload).hexdigest(),
        "log_binding_scope": IMPLEMENTER_LOG_BINDING_SCOPE,
        "log_event_anchored": False,
        "started_event_id": event_ids[0],
        "started_event_sequence": int(started_event.get("sequence") or 0),
        "finished_event_id": event_ids[1],
        "finished_event_sequence": int(finished_event.get("sequence") or 0),
        "source_stream_id": str(started_event.get("stream_id") or ""),
        "source_snapshot_id": str(started_event.get("snapshot_id") or ""),
    }
    if (
        material["started_event_sequence"] < 1
        or material["finished_event_sequence"]
        <= material["started_event_sequence"]
        or not material["source_stream_id"]
        or not material["source_snapshot_id"]
    ):
        raise PostMergeReviewError(
            "implementer_provenance_stream_binding_missing",
            "implementation event lacks durable stream/sequence binding",
        )
    return VerifiedImplementerProvenance(
        **{
            **material,
            "provenance_id": content_identity(material),
        }
    )


def _strict_event_ledger(events_path: Path) -> list[dict[str, Any]]:
    """Read a v2 ledger only when its manifest and every segment are exact."""

    path = Path(events_path)
    manifest = _event_log_runtime._load_event_manifest(path)
    if (
        manifest is None
        or not _event_log_runtime._manifest_matches_metadata(path, manifest)
    ):
        raise PostMergeReviewError(
            "event_ledger_manifest_invalid",
            "event ledger v2 manifest is missing, stale, or invalid",
        )
    records = {
        str(item.get("path") or ""): dict(item)
        for item in manifest.get("files", ())
        if isinstance(item, Mapping)
    }
    sources = _event_log_runtime._source_paths(path)
    if not sources or set(records) != {source.name for source in sources}:
        raise PostMergeReviewError(
            "event_ledger_segments_invalid",
            "event ledger segments do not exactly match the manifest",
        )
    stream_id = str(manifest.get("stream_id") or "")
    snapshot_id = str(manifest.get("snapshot_id") or "")
    ordered_sources = sorted(
        sources,
        key=lambda source: int(
            records[source.name].get("first_sequence") or 0
        ),
    )
    events: list[dict[str, Any]] = []
    prior_event_id = ""
    expected_sequence = int(manifest.get("earliest_sequence") or 0)
    for source_index, source in enumerate(ordered_sources):
        record = records[source.name]
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise PostMergeReviewError(
                "event_ledger_segment_unreadable",
                f"event ledger segment {source.name!r} is unreadable",
            ) from exc
        segment_digest = str(record.get("sha256") or "")
        if segment_digest and segment_digest != hashlib.sha256(raw).hexdigest():
            raise PostMergeReviewError(
                "event_ledger_segment_digest_invalid",
                f"event ledger segment {source.name!r} digest changed",
            )
        segment_events: list[dict[str, Any]] = []
        for raw_line in raw.splitlines():
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PostMergeReviewError(
                    "event_ledger_segment_malformed",
                    f"event ledger segment {source.name!r} is malformed",
                ) from exc
            if not isinstance(event, dict):
                raise PostMergeReviewError(
                    "event_ledger_segment_malformed",
                    f"event ledger segment {source.name!r} has non-object events",
                )
            event_id = str(event.get("event_id") or "")
            body = dict(event)
            body.pop("event_id", None)
            canonical = json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            if (
                event_id
                != "sha256:" + hashlib.sha256(canonical).hexdigest()
                or str(event.get("stream_id") or "") != stream_id
                or str(event.get("snapshot_id") or "") != snapshot_id
                or int(event.get("sequence") or 0) != expected_sequence
            ):
                raise PostMergeReviewError(
                    "event_ledger_chain_invalid",
                    "event ledger identity, stream, snapshot, or sequence changed",
                )
            if events and str(event.get("previous_event_id") or "") != prior_event_id:
                raise PostMergeReviewError(
                    "event_ledger_chain_invalid",
                    "event ledger previous-event chain is discontinuous",
                )
            if not events and (
                str(event.get("previous_event_id") or "")
                != str(record.get("start_previous_event_id") or "")
            ):
                raise PostMergeReviewError(
                    "event_ledger_chain_invalid",
                    "first segment start_previous_event_id does not match",
                )
            prior_event_id = event_id
            expected_sequence += 1
            events.append(event)
            segment_events.append(event)
        if (
            len(segment_events) != int(record.get("event_count") or 0)
            or (
                segment_events
                and (
                    int(segment_events[0]["sequence"])
                    != int(record.get("first_sequence") or 0)
                    or int(segment_events[-1]["sequence"])
                    != int(record.get("last_sequence") or 0)
                    or str(segment_events[0].get("previous_event_id") or "")
                    != str(record.get("start_previous_event_id") or "")
                )
            )
        ):
            raise PostMergeReviewError(
                "event_ledger_segment_count_invalid",
                f"event ledger segment {source.name!r} range/count changed",
            )
        if source_index and segment_events:
            previous_record = records[ordered_sources[source_index - 1].name]
            if int(record.get("first_sequence") or 0) != int(
                previous_record.get("last_sequence") or 0
            ) + 1:
                raise PostMergeReviewError(
                    "event_ledger_segment_range_invalid",
                    "event ledger segment ranges are not contiguous",
                )
    if (
        len(events)
        != int(manifest.get("latest_sequence") or 0)
        - int(manifest.get("earliest_sequence") or 0)
        + 1
        or (events and int(events[-1]["sequence"]) != int(
            manifest.get("latest_sequence") or 0
        ))
        or (events and str(events[-1]["event_id"]) != str(
            manifest.get("last_event_id") or ""
        ))
    ):
        raise PostMergeReviewError(
            "event_ledger_head_invalid",
            "event ledger population or head does not match its manifest",
        )
    return events


def verified_implementer_provenance_from_ledger(
    events_path: Path,
    *,
    repo_root: Path,
    expected_task_id: str,
    expected_implementation_attempt: int,
    expected_implementation_commit: str,
    expected_branch: str | None = None,
    expected_log_path: str | None = None,
) -> VerifiedImplementerProvenance:
    """Select exactly one verified implementation pair from the strict ledger.

    Callers identify the task, implementation attempt, and resulting commit;
    this function owns event selection. Optional branch/log expectations can
    narrow the identity further, but a missing or ambiguous valid pair always
    fails closed.
    """

    if (
        not str(expected_task_id or "")
        or isinstance(expected_implementation_attempt, bool)
        or int(expected_implementation_attempt) < 1
        or not _FULL_OBJECT_ID.fullmatch(
            str(expected_implementation_commit or "")
        )
        or expected_branch is not None
        and not str(expected_branch)
        or expected_log_path is not None
        and not str(expected_log_path)
    ):
        raise PostMergeReviewError(
            "implementer_ledger_query_invalid",
            "ledger provenance query requires exact task/attempt/commit identity",
        )
    ledger = _strict_event_ledger(Path(events_path))

    def event_attempt(event: Mapping[str, Any]) -> int | None:
        value = event.get("attempt")
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    started = [
        event
        for event in ledger
        if event.get("type") == "implementation_started"
        and event.get("task_id") == expected_task_id
        and event_attempt(event) == int(expected_implementation_attempt)
        and (
            expected_branch is None
            or event.get("branch") == expected_branch
        )
        and (
            expected_log_path is None
            or event.get("log_path") == expected_log_path
        )
    ]
    finished = [
        event
        for event in ledger
        if event.get("type") == "implementation_finished"
        and event.get("task_id") == expected_task_id
        and event_attempt(event) == int(expected_implementation_attempt)
        and event.get("implementation_commit")
        == expected_implementation_commit
        and (
            expected_branch is None
            or event.get("branch") == expected_branch
        )
        and (
            expected_log_path is None
            or event.get("log_path") == expected_log_path
        )
    ]
    matches: list[VerifiedImplementerProvenance] = []
    for started_event in started:
        for finished_event in finished:
            try:
                matches.append(
                    verified_implementer_provenance_from_events(
                        started_event,
                        finished_event,
                        repo_root=repo_root,
                        expected_task_id=expected_task_id,
                        expected_implementation_attempt=(
                            expected_implementation_attempt
                        ),
                        expected_implementation_commit=(
                            expected_implementation_commit
                        ),
                    )
                )
            except PostMergeReviewError:
                continue
    if not matches:
        raise PostMergeReviewError(
            "implementer_event_pair_missing",
            "strict event ledger has no valid implementation start/finish pair",
        )
    if len(matches) != 1:
        raise PostMergeReviewError(
            "implementer_event_pair_ambiguous",
            "strict event ledger has multiple valid implementation event pairs",
        )
    return matches[0]


def _verify_implementer_event_membership(
    events_path: Path,
    provenance: VerifiedImplementerProvenance,
    *,
    repo_root: Path,
) -> Mapping[str, Any]:
    """Require provenance membership in the daemon-owned strict v2 ledger."""

    events = _strict_event_ledger(events_path)
    started_match = next(
        (
            event
            for event in events
            if event.get("event_id") == provenance.started_event_id
            and int(event.get("sequence") or 0)
            == provenance.started_event_sequence
        ),
        None,
    )
    finished_match = next(
        (
            event
            for event in events
            if event.get("event_id") == provenance.finished_event_id
            and int(event.get("sequence") or 0)
            == provenance.finished_event_sequence
        ),
        None,
    )
    if started_match is None or finished_match is None:
        raise PostMergeReviewError(
            "implementer_event_not_in_ledger",
            "implementation provider start/finish pair is not in the daemon ledger",
        )
    rebuilt = verified_implementer_provenance_from_events(
        started_match,
        finished_match,
        repo_root=repo_root,
        expected_task_id=provenance.task_id,
        expected_implementation_attempt=provenance.implementation_attempt,
        expected_implementation_commit=provenance.implementation_commit,
    )
    if rebuilt != provenance:
        raise PostMergeReviewError(
            "implementer_event_provenance_mismatch",
            "ledger events do not match supplied implementer provenance",
        )
    return finished_match


def verified_implementation_finished_event_from_ledger(
    events_path: Path,
    provenance: VerifiedImplementerProvenance,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Return the exact hash-chain member bound by verified provenance."""

    return dict(
        _verify_implementer_event_membership(
            events_path,
            provenance,
            repo_root=Path(repo_root).resolve(),
        )
    )


def verified_implementation_finished_event_from_strict_ledger(
    events_path: Path,
    *,
    task_id: str,
    implementation_attempt: int,
    branch: str,
    implementation_commit: str,
    baseline_ref: str,
    canonical_task_key: str,
    canonical_task_cid: str,
) -> dict[str, Any]:
    """Select one exact finish event from a manifest-verified hash chain."""

    if (
        not str(task_id or "")
        or isinstance(implementation_attempt, bool)
        or int(implementation_attempt) < 1
        or not str(branch or "")
        or not _FULL_OBJECT_ID.fullmatch(
            str(implementation_commit or "")
        )
        or not str(baseline_ref or "")
        or not str(canonical_task_key or "")
        or not str(canonical_task_cid or "")
    ):
        raise PostMergeReviewError(
            "implementation_finish_query_invalid",
            "strict finish-event lookup requires every immutable binding",
        )
    matches = [
        event
        for event in _strict_event_ledger(events_path)
        if (
            event.get("type") == "implementation_finished"
            and event.get("task_id") == task_id
            and not isinstance(event.get("attempt"), bool)
            and int(event.get("attempt") or 0)
            == int(implementation_attempt)
            and event.get("branch") == branch
            and event.get("implementation_commit")
            == implementation_commit
            and event.get("baseline_ref") == baseline_ref
            and event.get("returncode") == 0
            and event.get("canonical_task_key") == canonical_task_key
            and (
                event.get("canonical_task_cid")
                or event.get("canonical_task_id")
            )
            == canonical_task_cid
        )
    ]
    if len(matches) != 1:
        raise PostMergeReviewError(
            "implementation_finish_event_unavailable",
            "strict event ledger does not contain one exact finish event",
        )
    return dict(matches[0])


def task_proposal_scope_paths(task: Any) -> tuple[str, ...]:
    """Reconstruct the proposal issuer's original mutable path envelope."""

    projection = _task_projection(task)
    raw_paths = list(projection["outputs"])
    metadata = getattr(task, "metadata", {})
    if isinstance(metadata, Mapping):
        for name in ("predicted files", "allowed paths"):
            raw_paths.extend(
                item.strip()
                for item in str(metadata.get(name) or "").split(",")
                if item.strip()
                and item.strip().casefold() not in {"none", "n/a"}
            )
    normalized: set[str] = set()
    for raw_path in raw_paths:
        path = str(raw_path).strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        if (
            not path
            or path.startswith("/")
            or "\0" in path
            or ".." in PurePosixPath(path).parts
        ):
            continue
        normalized.add(path)
    return tuple(sorted(normalized))


def _scope_authorization_from_implementation_event(
    finished_event: Mapping[str, Any],
    *,
    repo_root: Path,
    task: Any,
    baseline_commit: str,
    implementation_commit: str,
    expected_changed_paths: Sequence[str] | None,
) -> tuple[tuple[str, ...], str]:
    """Recover scope authority from a full or exactly reconstructed receipt."""

    validation_result = finished_event.get("validation_result")
    if not isinstance(validation_result, Mapping):
        return (), ""
    scope = validation_result.get("scope_adjudication")
    if scope is None:
        return (), ""
    proposal = validation_result.get("proposal_gate")
    if not isinstance(scope, Mapping) or not isinstance(proposal, Mapping):
        raise PostMergeReviewError(
            "scope_adjudication_event_invalid",
            "implementation finish event has malformed scope evidence",
        )
    raw_changed_paths = proposal.get("changed_paths")
    changed_paths = (
        tuple(str(path) for path in raw_changed_paths)
        if isinstance(raw_changed_paths, list)
        else ()
    )
    actual_changed_paths = exact_implementation_changed_paths(
        repo_root=repo_root,
        baseline_commit=baseline_commit,
        implementation_commit=implementation_commit,
    )
    expected = (
        tuple(sorted(_normalize_path(path) for path in expected_changed_paths))
        if expected_changed_paths is not None
        else ()
    )
    task_projection = _task_projection(task)
    event_baseline = str(finished_event.get("baseline_ref") or "")
    canonical_task_key = str(
        finished_event.get("canonical_task_key") or ""
    )
    canonical_task_cid = str(
        finished_event.get("canonical_task_cid")
        or finished_event.get("canonical_task_id")
        or ""
    )
    if (
        validation_result.get("passed") is not True
        or proposal.get("accepted") is not True
        or event_baseline != baseline_commit
        or str(proposal.get("repository_tree_id") or "")
        != baseline_commit
        or canonical_task_key
        != task_projection["canonical_task_key"]
        or canonical_task_cid
        != task_projection["canonical_task_cid"]
        or changed_paths != expected
        or changed_paths != actual_changed_paths
    ):
        raise PostMergeReviewError(
            "scope_adjudication_event_invalid",
            "implementation finish event scope evidence is not exact and "
            "task/proposal/baseline/diff bound",
        )
    try:
        receipt = verified_scope_adjudication_receipt(
            scope,
            task_id=task_projection["task_id"],
            proposal_id=str(proposal.get("proposal_id") or ""),
            authorized_policy_id=str(proposal.get("policy_id") or ""),
            repository_id=_scope_receipt_repository_id(repo_root),
            repository_tree_id=str(
                proposal.get("repository_tree_id") or ""
            ),
            baseline_id=baseline_commit,
            original_scope_paths=task_proposal_scope_paths(task),
            candidate_paths=actual_changed_paths,
            allow_legacy_compact=True,
        )
    except (TypeError, ValueError) as exc:
        raise PostMergeReviewError(
            "scope_adjudication_receipt_invalid",
            "implementation finish event scope receipt failed exact "
            "content-addressed verification",
        ) from exc
    authorized_paths = receipt.authorized_paths
    material = {
        "task_binding_id": post_merge_task_binding_id(task),
        "proposal_id": receipt.proposal_id,
        "authorized_policy_id": receipt.authorized_policy_id,
        "receipt_id": receipt.receipt_id,
        "repository_tree_id": receipt.repository_tree_id,
        "changed_paths": list(changed_paths),
        "authorized_paths": list(authorized_paths),
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    return authorized_paths, content_identity(material)


def _git(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=text,
        capture_output=True,
        check=False,
    )


def _scope_receipt_repository_id(repo_root: Path) -> str:
    """Recompute the repository identity used when the receipt was issued."""

    result = _git(repo_root, ["rev-parse", "--git-common-dir"])
    common_dir = str(result.stdout or "").strip()
    if result.returncode != 0 or not common_dir:
        raise PostMergeReviewError(
            "scope_adjudication_repository_unavailable",
            "cannot recompute the scope receipt repository identity",
        )
    common_path = Path(common_dir)
    if not common_path.is_absolute():
        common_path = repo_root / common_path
    try:
        identity_source = str(common_path.resolve(strict=True))
    except (OSError, RuntimeError) as exc:
        raise PostMergeReviewError(
            "scope_adjudication_repository_unavailable",
            "cannot resolve the scope receipt repository identity",
        ) from exc
    digest = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()
    return f"repository:sha256:{digest}"


def _exact_commit(repo_root: Path, value: str, *, field_name: str) -> str:
    candidate = str(value or "").strip()
    if not _FULL_OBJECT_ID.fullmatch(candidate):
        raise PostMergeReviewError(
            f"{field_name}_invalid",
            f"{field_name} must be a full lowercase Git object ID",
        )
    resolved = _git(
        repo_root,
        ["rev-parse", "--verify", "--end-of-options", f"{candidate}^{{commit}}"],
    )
    actual = str(resolved.stdout or "").strip()
    if resolved.returncode != 0 or actual != candidate:
        raise PostMergeReviewError(
            f"{field_name}_unavailable",
            f"{field_name} does not resolve to the exact requested commit",
        )
    return actual


def _tree_id(repo_root: Path, commit: str) -> tuple[str, str]:
    resolved = _git(
        repo_root,
        ["rev-parse", "--verify", "--end-of-options", f"{commit}^{{tree}}"],
    )
    tree = str(resolved.stdout or "").strip()
    if resolved.returncode != 0 or not _FULL_OBJECT_ID.fullmatch(tree):
        raise PostMergeReviewError(
            "repository_tree_unavailable",
            "the merged commit tree could not be resolved",
        )
    return tree, f"git-tree:{tree}"


def _normalize_path(value: Any) -> str:
    path = str(value or "")
    pure = PurePosixPath(path)
    if (
        not path
        or path != path.strip()
        or path.startswith(("/", "\\"))
        or path.endswith("/")
        or "\\" in path
        or "\x00" in path
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise PostMergeReviewError(
            "changed_path_invalid",
            f"unsafe or non-canonical changed path: {path!r}",
        )
    return pure.as_posix()


def _task_projection(task: Any) -> dict[str, Any]:
    task_id = str(getattr(task, "task_id", "") or "").strip()
    title = str(getattr(task, "title", "") or "").strip()
    if not task_id or not title:
        raise PostMergeReviewError(
            "task_identity_missing",
            "post-merge review requires a real task with task_id and title",
        )
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise PostMergeReviewError(
            "task_metadata_invalid",
            "post-merge review requires task metadata to be a mapping",
        )

    def values(name: str) -> list[str]:
        raw = getattr(task, name, None)
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(
            raw, Sequence
        ):
            raise PostMergeReviewError(
                f"task_{name}_invalid",
                f"task {name} must be a sequence",
            )
        return [str(item) for item in raw]

    return {
        "task_id": task_id,
        "title": title,
        "acceptance": str(getattr(task, "acceptance", "") or ""),
        "completion": str(getattr(task, "completion", "") or ""),
        "priority": str(getattr(task, "priority", "") or ""),
        "track": str(getattr(task, "track", "") or ""),
        "depends_on": values("depends_on"),
        "outputs": values("outputs"),
        "validation": values("validation"),
        "metadata": {
            str(key): str(value)
            for key, value in sorted(
                metadata.items(),
                key=lambda item: str(item[0]),
            )
        },
        "canonical_task_key": str(
            getattr(task, "canonical_task_key", "") or ""
        ),
        "canonical_task_cid": str(
            getattr(task, "canonical_task_cid", "") or ""
        ),
        "board_namespace": str(
            getattr(task, "board_namespace", "") or ""
        ),
    }


def post_merge_task_binding_id(task: Any) -> str:
    """Return the exact task-spec identity used by post-merge review."""

    return content_identity(_task_projection(task))


def _path_authorized_by_task(path: str, outputs: Sequence[str]) -> bool:
    for raw in outputs:
        output = str(raw or "").strip()
        if not output:
            continue
        if any(character in output for character in "*?["):
            if fnmatch.fnmatchcase(path, output.replace("\\", "/")):
                return True
            continue
        if output.endswith("/"):
            prefix = output.rstrip("/")
            if path.startswith(f"{prefix}/"):
                return True
            continue
        try:
            normalized = _normalize_path(output)
            if path == normalized or path.startswith(f"{normalized}/"):
                return True
        except PostMergeReviewError:
            continue
    return False


def _tree_entry(repo_root: Path, commit: str, path: str) -> dict[str, str] | None:
    result = _git(
        repo_root,
        ["ls-tree", "-z", commit, "--", path],
        text=False,
    )
    if result.returncode != 0:
        raise PostMergeReviewError(
            "tree_entry_unavailable",
            f"could not inspect {path!r} at {commit}",
        )
    raw = bytes(result.stdout or b"")
    if not raw:
        return None
    rows = [row for row in raw.split(b"\x00") if row]
    if len(rows) != 1 or b"\t" not in rows[0]:
        raise PostMergeReviewError(
            "tree_entry_ambiguous",
            f"tree lookup for {path!r} was not exact",
        )
    header, raw_path = rows[0].split(b"\t", 1)
    try:
        actual_path = raw_path.decode("utf-8", errors="strict")
        mode, object_type, object_id = header.decode(
            "ascii", errors="strict"
        ).split(" ", 2)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PostMergeReviewError(
            "tree_entry_invalid",
            f"tree entry for {path!r} is not canonical UTF-8 Git metadata",
        ) from exc
    if actual_path != path or not _FULL_OBJECT_ID.fullmatch(object_id):
        raise PostMergeReviewError(
            "tree_entry_binding_mismatch",
            f"tree entry for {path!r} did not bind to the requested path",
        )
    return {
        "mode": mode,
        "object_type": object_type,
        "git_object_id": object_id,
    }


def _diff_statuses(
    repo_root: Path,
    base_commit: str,
    implementation_commit: str,
) -> tuple[tuple[str, str], ...]:
    result = _git(
        repo_root,
        [
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            base_commit,
            implementation_commit,
        ],
        text=False,
    )
    if result.returncode != 0:
        raise PostMergeReviewError(
            "implementation_diff_unavailable",
            "could not compute the implementation commit diff",
        )
    parts = bytes(result.stdout or b"").split(b"\x00")
    if parts and parts[-1] == b"":
        parts.pop()
    if len(parts) % 2:
        raise PostMergeReviewError(
            "implementation_diff_malformed",
            "Git returned malformed changed-path metadata",
        )
    rows: list[tuple[str, str]] = []
    for offset in range(0, len(parts), 2):
        try:
            status = parts[offset].decode("ascii", errors="strict")
            path = parts[offset + 1].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PostMergeReviewError(
                "implementation_diff_non_utf8",
                "changed paths must be canonical UTF-8",
            ) from exc
        if status not in {"A", "D", "M", "T"}:
            raise PostMergeReviewError(
                "implementation_diff_status_unsupported",
                f"unsupported Git diff status {status!r}",
            )
        rows.append((status, _normalize_path(path)))
    return tuple(sorted(rows, key=lambda item: item[1]))


def _repository_patch(
    repo_root: Path,
    base_commit: str,
    implementation_commit: str,
    paths: Sequence[str],
) -> bytes:
    patch = _git(
        repo_root,
        [
            "diff",
            "--no-ext-diff",
            "--binary",
            "--full-index",
            "--no-renames",
            base_commit,
            implementation_commit,
            "--",
            *paths,
        ],
        text=False,
    )
    if patch.returncode != 0:
        raise PostMergeReviewError(
            "implementation_patch_unavailable",
            f"could not materialize the exact patch in {repo_root}",
        )
    return bytes(patch.stdout or b"")


def _expand_repository_diff(
    *,
    checkout_root: Path,
    repository_root: Path,
    base_commit: str,
    implementation_commit: str,
    landed_commit: str,
    repository_prefix: str = "",
    depth: int = 0,
) -> dict[str, Any]:
    """Expand a Git diff through exact initialized submodule gitlinks."""

    if depth > 8:
        raise PostMergeReviewError(
            "submodule_diff_depth_exceeded",
            "post-merge review submodule recursion exceeds its depth bound",
        )
    base = _exact_commit(
        repository_root,
        base_commit,
        field_name="diff_base_commit",
    )
    implementation = _exact_commit(
        repository_root,
        implementation_commit,
        field_name="diff_implementation_commit",
    )
    landed = _exact_commit(
        repository_root,
        landed_commit,
        field_name="diff_landed_commit",
    )
    statuses = _diff_statuses(repository_root, base, implementation)
    local_paths = tuple(path for _status, path in statuses)
    if not local_paths:
        return {
            "leaf_statuses": (),
            "content_bindings": (),
            "gitlink_bindings": (),
            "patch_parts": (),
        }
    patch = _repository_patch(
        repository_root,
        base,
        implementation,
        local_paths,
    )
    repository_label = repository_prefix.rstrip("/") or "."
    patch_part = (
        f"\n--- repository:{repository_label} "
        f"base:{base} implementation:{implementation} ---\n"
    ).encode() + patch

    leaf_statuses: list[tuple[str, str]] = []
    content_bindings: list[dict[str, Any]] = []
    gitlink_bindings: list[dict[str, Any]] = []
    patch_parts: list[bytes] = [patch_part]
    status_by_path = {path: status for status, path in statuses}
    for local_path in local_paths:
        full_path = (
            f"{repository_prefix.rstrip('/')}/{local_path}"
            if repository_prefix
            else local_path
        )
        base_entry = _tree_entry(repository_root, base, local_path)
        implementation_entry = _tree_entry(
            repository_root,
            implementation,
            local_path,
        )
        landed_entry = _tree_entry(repository_root, landed, local_path)
        if implementation_entry != landed_entry:
            raise PostMergeReviewError(
                "merged_content_binding_mismatch",
                f"landed content for {full_path!r} differs from "
                "implementation_commit",
            )
        entries = (base_entry, implementation_entry, landed_entry)
        is_gitlink = any(
            entry is not None
            and (
                entry.get("mode") == "160000"
                or entry.get("object_type") == "commit"
            )
            for entry in entries
        )
        if not is_gitlink:
            leaf_statuses.append((status_by_path[local_path], full_path))
            content_bindings.append(
                {
                    "path": full_path,
                    "repository_path": repository_label,
                    "repository_relative_path": local_path,
                    "status": status_by_path[local_path],
                    "base": base_entry,
                    "implementation": implementation_entry,
                    "merged": landed_entry,
                }
            )
            continue
        if (
            base_entry is None
            or implementation_entry is None
            or landed_entry is None
            or any(
                entry.get("mode") != "160000"
                or entry.get("object_type") != "commit"
                for entry in entries
            )
        ):
            raise PostMergeReviewError(
                "submodule_gitlink_transition_unsupported",
                f"added, removed, or malformed submodule gitlink {full_path!r}",
            )
        gitlink_bindings.append(
            {
                "path": full_path,
                "parent_repository_path": repository_label,
                "status": status_by_path[local_path],
                "base": base_entry,
                "implementation": implementation_entry,
                "merged": landed_entry,
            }
        )
        child_root = (repository_root / local_path).resolve()
        try:
            child_root.relative_to(checkout_root)
        except ValueError as exc:
            raise PostMergeReviewError(
                "submodule_checkout_outside_repository",
                f"submodule checkout {full_path!r} escapes the repository",
            ) from exc
        if not child_root.is_dir():
            raise PostMergeReviewError(
                "submodule_checkout_unavailable",
                f"exact initialized submodule checkout is required for {full_path!r}",
            )
        probe = _git(child_root, ["rev-parse", "--show-toplevel"])
        if (
            probe.returncode != 0
            or Path(str(probe.stdout or "").strip()).resolve() != child_root
        ):
            raise PostMergeReviewError(
                "submodule_checkout_unavailable",
                f"exact initialized submodule checkout is required for {full_path!r}",
            )
        nested = _expand_repository_diff(
            checkout_root=checkout_root,
            repository_root=child_root,
            base_commit=str(base_entry["git_object_id"]),
            implementation_commit=str(
                implementation_entry["git_object_id"]
            ),
            landed_commit=str(landed_entry["git_object_id"]),
            repository_prefix=full_path,
            depth=depth + 1,
        )
        if not nested["leaf_statuses"]:
            raise PostMergeReviewError(
                "submodule_diff_empty",
                f"changed gitlink {full_path!r} has no inspectable nested diff",
            )
        leaf_statuses.extend(nested["leaf_statuses"])
        content_bindings.extend(nested["content_bindings"])
        gitlink_bindings.extend(nested["gitlink_bindings"])
        patch_parts.extend(nested["patch_parts"])
    return {
        "leaf_statuses": tuple(leaf_statuses),
        "content_bindings": tuple(content_bindings),
        "gitlink_bindings": tuple(gitlink_bindings),
        "patch_parts": tuple(patch_parts),
    }


def exact_implementation_changed_paths(
    *,
    repo_root: Path,
    baseline_commit: str,
    implementation_commit: str,
) -> tuple[str, ...]:
    """Return the exact leaf paths changed by one implementation candidate."""

    root = Path(repo_root).resolve()
    baseline = _exact_commit(
        root,
        baseline_commit,
        field_name="baseline_commit",
    )
    implementation = _exact_commit(
        root,
        implementation_commit,
        field_name="implementation_commit",
    )
    expanded = _expand_repository_diff(
        checkout_root=root,
        repository_root=root,
        base_commit=baseline,
        implementation_commit=implementation,
        landed_commit=implementation,
    )
    return tuple(
        path
        for _status, path in sorted(
            expanded["leaf_statuses"],
            key=lambda item: item[1],
        )
    )


def _collect_repository_binding(
    *,
    repo_root: Path,
    task: Any,
    baseline_commit: str,
    implementation_commit: str,
    merge_commit: str,
    repository_tree_id: str,
    expected_changed_paths: Sequence[str] | None,
    scope_authorized_paths: Sequence[str] = (),
    scope_adjudication_id: str = "",
) -> dict[str, Any]:
    implementation = _exact_commit(
        repo_root,
        implementation_commit,
        field_name="implementation_commit",
    )
    merged = _exact_commit(
        repo_root,
        merge_commit,
        field_name="merge_commit",
    )
    ancestor = _git(
        repo_root,
        ["merge-base", "--is-ancestor", implementation, merged],
    )
    if ancestor.returncode != 0:
        raise PostMergeReviewError(
            "implementation_not_merged",
            "implementation_commit is not an ancestor of merge_commit",
        )
    baseline = _exact_commit(
        repo_root,
        baseline_commit,
        field_name="baseline_commit",
    )
    baseline_ancestor = _git(
        repo_root,
        ["merge-base", "--is-ancestor", baseline, implementation],
    )
    if baseline_ancestor.returncode != 0:
        raise PostMergeReviewError(
            "baseline_not_ancestor",
            "baseline_commit is not an ancestor of implementation_commit",
        )
    merge_tree, actual_repository_tree_id = _tree_id(repo_root, merged)
    if actual_repository_tree_id != str(repository_tree_id or ""):
        raise PostMergeReviewError(
            "repository_tree_binding_mismatch",
            "repository_tree_id does not identify merge_commit^{tree}",
        )

    base_commit = baseline
    expanded = _expand_repository_diff(
        checkout_root=repo_root,
        repository_root=repo_root,
        base_commit=base_commit,
        implementation_commit=implementation,
        landed_commit=merged,
    )
    statuses = tuple(
        sorted(expanded["leaf_statuses"], key=lambda item: item[1])
    )
    actual_paths = tuple(path for _status, path in statuses)
    if not actual_paths:
        raise PostMergeReviewError(
            "implementation_diff_empty",
            "independent review requires at least one changed path",
        )

    if expected_changed_paths is not None:
        normalized_expected = tuple(
            sorted(_normalize_path(item) for item in expected_changed_paths)
        )
        if len(set(normalized_expected)) != len(normalized_expected):
            raise PostMergeReviewError(
                "expected_changed_paths_duplicated",
                "expected_changed_paths must not contain duplicates",
            )
        if normalized_expected != actual_paths:
            raise PostMergeReviewError(
                "changed_path_binding_mismatch",
                "expected_changed_paths do not exactly equal the implementation diff",
            )

    task_projection = _task_projection(task)
    if isinstance(scope_authorized_paths, (str, bytes, bytearray)):
        raise PostMergeReviewError(
            "scope_authorized_paths_invalid",
            "scope-authorized changed paths must be a sequence",
        )
    normalized_scope_authorized_paths = tuple(
        _normalize_path(item) for item in scope_authorized_paths
    )
    if (
        normalized_scope_authorized_paths
        != tuple(sorted(set(normalized_scope_authorized_paths)))
        or not set(normalized_scope_authorized_paths).issubset(actual_paths)
    ):
        raise PostMergeReviewError(
            "scope_authorized_paths_invalid",
            "scope-authorized changed paths must be canonical, unique, sorted, "
            "and present in the exact implementation diff",
        )
    normalized_scope_adjudication_id = str(
        scope_adjudication_id or ""
    ).strip()
    if bool(normalized_scope_authorized_paths) != bool(
        normalized_scope_adjudication_id
    ):
        raise PostMergeReviewError(
            "scope_adjudication_binding_missing",
            "scope-authorized paths require one verified adjudication identity",
        )
    scope_authorized_path_set = set(normalized_scope_authorized_paths)
    unauthorized = [
        path
        for path in actual_paths
        if (
            not _path_authorized_by_task(
                path,
                task_proposal_scope_paths(task),
            )
            and path not in scope_authorized_path_set
        )
    ]
    if unauthorized:
        raise PostMergeReviewError(
            "changed_path_outside_task_outputs",
            "implementation changed paths outside task.outputs: "
            + ", ".join(unauthorized),
        )

    content_bindings = list(expanded["content_bindings"])
    gitlink_bindings = list(expanded["gitlink_bindings"])
    patch_bytes = b"".join(expanded["patch_parts"])
    if not patch_bytes or len(patch_bytes) > MAX_REVIEW_DIFF_BYTES:
        raise PostMergeReviewError(
            "implementation_patch_size_invalid",
            "implementation patch is empty or exceeds the post-merge review bound",
        )
    try:
        patch_text = patch_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PostMergeReviewError(
            "implementation_patch_non_utf8",
            "post-merge Codex review accepts only UTF-8 textual patches",
        ) from exc

    diff_material = {
        "base_commit": base_commit,
        "implementation_commit": implementation,
        "merge_commit": merged,
        "merge_tree": merge_tree,
        "repository_tree_id": actual_repository_tree_id,
        "changed_paths": list(actual_paths),
        "scope_authorized_paths": list(normalized_scope_authorized_paths),
        "scope_adjudication_id": normalized_scope_adjudication_id,
        "content_bindings": content_bindings,
        "gitlink_bindings": gitlink_bindings,
        "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
        "patch_bytes": len(patch_bytes),
    }
    scope_authorization_material = {
        "task_binding_id": post_merge_task_binding_id(task),
        "base_commit": base_commit,
        "implementation_commit": implementation,
        "changed_paths": list(actual_paths),
        "scope_authorized_paths": list(normalized_scope_authorized_paths),
        "scope_adjudication_id": normalized_scope_adjudication_id,
    }
    return {
        **diff_material,
        "diff_binding_id": content_identity(diff_material),
        "scope_authorization_id": content_identity(
            scope_authorization_material
        ),
        "patch_text": patch_text,
        "task_projection": task_projection,
        "task_binding_id": scope_authorization_material["task_binding_id"],
    }


def _verify_validation_evidence(
    validation_result: Mapping[str, Any],
    *,
    task_validation: Sequence[str],
    task_id: str,
    merge_commit: str,
    repository_tree_id: str,
) -> str:
    if not isinstance(validation_result, Mapping):
        raise PostMergeReviewError(
            "post_merge_validation_missing",
            "post-merge validation evidence is required",
        )
    evidence = dict(validation_result)
    receipt_id = str(evidence.pop("validation_receipt_id", "") or "")
    expected_tree = repository_tree_id.removeprefix("git-tree:")
    declared_commands = [str(item) for item in task_validation]
    validation_plan_material = {
        "task_id": task_id,
        "target_commit": merge_commit,
        "repository_tree_id": repository_tree_id,
        "validation_scope": "post_merge",
        "declared_commands": declared_commands,
    }
    expected_plan_id = content_identity(validation_plan_material)
    if (
        evidence.get("schema") != POST_MERGE_VALIDATION_EVIDENCE_SCHEMA
        or str(evidence.get("task_id") or "") != task_id
        or str(evidence.get("target_commit") or "") != merge_commit
        or str(evidence.get("repository_tree_id") or "") != repository_tree_id
        or str(evidence.get("target_tree") or "") != expected_tree
        or str(evidence.get("validated_commit") or "") != merge_commit
        or str(evidence.get("validated_tree") or "") != expected_tree
        or str(evidence.get("validation_scope") or "") != "post_merge"
        or evidence.get("attempted") is not True
        or evidence.get("passed") is not True
        or evidence.get("stale") is True
        or evidence.get("freshness_authoritative") is not True
        or list(evidence.get("declared_commands") or ()) != declared_commands
        or str(evidence.get("validation_plan_id") or "")
        != expected_plan_id
        or evidence.get("workspace_clean") is not True
        or str(evidence.get("workspace_status_porcelain") or "") != ""
        or list(evidence.get("validation_dirty_paths") or ()) != []
        or int(evidence.get("validation_status_returncode") or 0) != 0
        or str(evidence.get("validation_status_stderr") or "") != ""
    ):
        raise PostMergeReviewError(
            "post_merge_validation_unbound",
            "validation evidence is not fresh and exactly merge/tree bound",
        )
    if not receipt_id or content_identity(evidence) != receipt_id:
        raise PostMergeReviewError(
            "post_merge_validation_receipt_invalid",
            "validation_receipt_id is missing or not content-addressed",
        )
    return receipt_id


def _normalize_implementer_provider(value: str) -> str:
    provider = str(value or "").strip().casefold()
    if provider not in ALLOWED_IMPLEMENTER_PROVIDERS:
        raise PostMergeReviewError(
            "implementer_provider_untrusted",
            "implementer provider must be an explicit admitted implementation "
            f"provider ({', '.join(sorted(ALLOWED_IMPLEMENTER_PROVIDERS))})",
        )
    if provider == CODEX_REVIEWER_PROVIDER or provider.startswith("codex"):
        raise PostMergeReviewError(
            "reviewer_implementer_not_independent",
            "Codex cannot independently review its own implementation",
        )
    return provider


def _verify_implementer_provenance(
    provenance: VerifiedImplementerProvenance,
    *,
    task_id: str,
    implementation_attempt: int,
    implementation_commit: str,
    provider_id: str,
) -> dict[str, Any]:
    if not isinstance(provenance, VerifiedImplementerProvenance):
        raise PostMergeReviewError(
            "implementer_provenance_missing",
            "caller must supply VerifiedImplementerProvenance from the durable "
            "implementation-start event",
        )
    payload = provenance.to_dict()
    material = dict(payload)
    provenance_id = str(material.pop("provenance_id", "") or "")
    if (
        material.get("schema") != VERIFIED_IMPLEMENTER_PROVENANCE_SCHEMA
        or material.get("task_id") != task_id
        or int(material.get("implementation_attempt") or 0)
        != int(implementation_attempt)
        or material.get("implementation_commit") != implementation_commit
        or material.get("provider_id") != provider_id
        or not provenance_id
        or content_identity(material) != provenance_id
    ):
        raise PostMergeReviewError(
            "implementer_provenance_binding_invalid",
            "implementer provenance is not content/task/attempt/commit/provider "
            "bound",
        )
    return payload


def _review_request(
    *,
    task: Any,
    attempt: int,
    implementation_attempt: int,
    implementer_provider: str,
    implementer_provenance: VerifiedImplementerProvenance,
    binding: Mapping[str, Any],
    validation_receipt_id: str,
) -> dict[str, Any]:
    if isinstance(attempt, bool) or int(attempt) < 1:
        raise PostMergeReviewError(
            "queue_attempt_invalid",
            "queue attempt must be a positive integer",
        )
    if isinstance(implementation_attempt, bool) or int(implementation_attempt) < 1:
        raise PostMergeReviewError(
            "implementation_attempt_invalid",
            "implementation attempt must be a positive integer",
        )
    provenance = _verify_implementer_provenance(
        implementer_provenance,
        task_id=str(getattr(task, "task_id", "") or ""),
        implementation_attempt=int(implementation_attempt),
        implementation_commit=str(binding["implementation_commit"]),
        provider_id=implementer_provider,
    )
    request = {
        "schema": POST_MERGE_INDEPENDENT_REVIEW_REQUEST_SCHEMA,
        "task_id": str(getattr(task, "task_id", "") or ""),
        "task_binding_id": str(binding["task_binding_id"]),
        "attempt": int(attempt),
        "implementation_attempt": int(implementation_attempt),
        "implementation_commit": str(binding["implementation_commit"]),
        "merge_commit": str(binding["merge_commit"]),
        "repository_tree_id": str(binding["repository_tree_id"]),
        "base_commit": str(binding["base_commit"]),
        "changed_paths": list(binding["changed_paths"]),
        "scope_authorized_paths": list(binding["scope_authorized_paths"]),
        "scope_adjudication_id": str(binding["scope_adjudication_id"]),
        "scope_authorization_id": str(binding["scope_authorization_id"]),
        "content_bindings": list(binding["content_bindings"]),
        "gitlink_bindings": list(binding["gitlink_bindings"]),
        "patch_sha256": str(binding["patch_sha256"]),
        "patch_bytes": int(binding["patch_bytes"]),
        "diff_binding_id": str(binding["diff_binding_id"]),
        "validation_receipt_id": validation_receipt_id,
        "implementer_provider": implementer_provider,
        "implementer_provenance": provenance,
        "implementer_provenance_id": provenance["provenance_id"],
        "reviewer_provider": CODEX_REVIEWER_PROVIDER,
        "reviewer_role": "independent_post_merge_review",
        "repository_write_allowed": False,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    request["request_id"] = content_identity(request)
    return request


def _review_prompt(
    request: Mapping[str, Any],
    task_projection: Mapping[str, Any],
    patch_text: str,
) -> str:
    response_shape = {
        "schema": POST_MERGE_INDEPENDENT_REVIEW_RESPONSE_SCHEMA,
        "decision": "approve | changes_required",
        "task_id": request["task_id"],
        "implementation_commit": request["implementation_commit"],
        "merge_commit": request["merge_commit"],
        "repository_tree_id": request["repository_tree_id"],
        "diff_binding_id": request["diff_binding_id"],
        "review_request_id": request["request_id"],
        "reviewer_provider": CODEX_REVIEWER_PROVIDER,
        "implementer_provider": request["implementer_provider"],
        "findings": [
            {
                "code": "short-stable-code",
                "severity": "blocker | high | medium | low | info",
                "summary": "bounded finding text",
            }
        ],
        "repository_write_authorized": False,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    request_without_content = {
        key: value
        for key, value in request.items()
        if key not in {"content_bindings", "gitlink_bindings"}
    }
    return "\n".join(
        (
            "You are the independent Codex post-merge reviewer. Review only the "
            "exact task and patch below.",
            "This is a read-only evidence review. Do not edit files, run commands, "
            "invoke tools, or claim write/proof/completion authority.",
            "Approve only when the exact landed patch satisfies the task acceptance "
            "criteria without correctness, security, compatibility, or test gaps.",
            "Return exactly one JSON object with exactly the specified fields. "
            "No Markdown, prose, comments, or extra fields.",
            "Use decision=changes_required and at least one finding for any defect "
            "or uncertainty.",
            "",
            "Bound review request:",
            json.dumps(request_without_content, sort_keys=True, indent=2),
            "",
            "Exact changed-content Git bindings:",
            json.dumps(request["content_bindings"], sort_keys=True, indent=2),
            "",
            "Exact parent-repository submodule gitlink bindings:",
            json.dumps(request["gitlink_bindings"], sort_keys=True, indent=2),
            "",
            "Bound task:",
            json.dumps(task_projection, sort_keys=True, indent=2),
            "",
            "Required response shape:",
            json.dumps(response_shape, sort_keys=True, indent=2),
            "",
            "Exact UTF-8 Git patch:",
            patch_text,
        )
    )


def _production_codex_reviewer(
    prompt: str,
    request: Mapping[str, Any],
    *,
    repo_root: Path,
) -> ReviewerInvocation:
    invocation = LlmRouterInvocation(
        repo_root=repo_root,
        model_name=DEFAULT_CODEX_MODEL,
        provider=CODEX_REVIEWER_PROVIDER,
        allow_local_fallback=False,
        timeout_seconds=600,
        max_new_tokens=4096,
        max_prompt_chars=MAX_REVIEW_PROMPT_BYTES,
        temperature=0.0,
        reject_effective_provider_name="local_hf",
        required_effective_providers=(CODEX_REVIEWER_PROVIDER,),
        request_id=str(request["request_id"]),
        attempt=int(request["attempt"]),
        idempotency_key=str(request["request_id"]),
        codex_read_only=True,
    )
    response_text, child_receipt = call_llm_router_with_receipt(
        prompt,
        invocation,
    )
    if child_receipt is None:
        raise PostMergeReviewError(
            "reviewer_execution_receipt_missing",
            "Codex llm_router call returned no typed child receipt",
        )
    return ReviewerInvocation(
        provider_id=str(child_receipt.effective_provider or ""),
        response_text=response_text,
        transport_receipt=child_receipt.to_dict(),
        sandbox="read-only",
    )


def _canonical_production_codex_reviewer(
    prompt: str,
    request: Mapping[str, Any],
    *,
    repo_root: Path,
    _reviewer: Callable[..., ReviewerInvocation] = _production_codex_reviewer,
) -> ReviewerInvocation:
    """Call the production reviewer captured when this module was loaded.

    The supported ``reviewer=`` dependency-injection seam is evidence-only.
    Capturing this callable also prevents a test or integration shim that
    merely replaces the module's ``_production_codex_reviewer`` symbol from
    being mistaken for the live production route. Python process mutation is
    outside the security boundary; tests fake only the isolated child
    transport below this canonical adapter.
    """

    return _reviewer(prompt, request, repo_root=repo_root)


def _parse_response(
    text: str,
    *,
    request: Mapping[str, Any],
    actual_provider: str,
) -> dict[str, Any]:
    encoded = str(text or "").encode("utf-8")
    if not encoded or len(encoded) > MAX_REVIEW_RESPONSE_BYTES:
        raise PostMergeReviewError(
            "review_response_size_invalid",
            "review response is empty or exceeds its byte bound",
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PostMergeReviewError(
            "review_response_malformed",
            "review response must be one strict JSON object",
        ) from exc
    required_fields = {
        "schema",
        "decision",
        "task_id",
        "implementation_commit",
        "merge_commit",
        "repository_tree_id",
        "diff_binding_id",
        "review_request_id",
        "reviewer_provider",
        "implementer_provider",
        "findings",
        "repository_write_authorized",
        "proof_authoritative",
        "completion_authoritative",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise PostMergeReviewError(
            "review_response_schema_invalid",
            "review response fields do not exactly match the versioned schema",
        )
    expected = {
        "schema": POST_MERGE_INDEPENDENT_REVIEW_RESPONSE_SCHEMA,
        "task_id": request["task_id"],
        "implementation_commit": request["implementation_commit"],
        "merge_commit": request["merge_commit"],
        "repository_tree_id": request["repository_tree_id"],
        "diff_binding_id": request["diff_binding_id"],
        "review_request_id": request["request_id"],
        "reviewer_provider": actual_provider,
        "implementer_provider": request["implementer_provider"],
        "repository_write_authorized": False,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PostMergeReviewError(
                f"review_response_binding_mismatch:{key}",
                f"review response field {key!r} is not exactly request-bound",
            )
    decision = str(payload.get("decision") or "")
    if decision not in {"approve", "changes_required"}:
        raise PostMergeReviewError(
            "review_decision_invalid",
            "review decision must be approve or changes_required",
        )
    findings = payload.get("findings")
    if (
        not isinstance(findings, list)
        or len(findings) > MAX_REVIEW_FINDINGS
        or (decision == "changes_required" and not findings)
        or (decision == "approve" and findings)
    ):
        raise PostMergeReviewError(
            "review_findings_invalid",
            "review findings are missing or exceed their bound",
        )
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {
            "code",
            "severity",
            "summary",
        }:
            raise PostMergeReviewError(
                "review_finding_schema_invalid",
                "every review finding must match the strict finding schema",
            )
        code = str(finding.get("code") or "")
        severity = str(finding.get("severity") or "")
        summary = str(finding.get("summary") or "")
        if (
            not code
            or severity not in {"blocker", "high", "medium", "low", "info"}
            or not summary
            or len(code.encode("utf-8")) > 128
            or len(summary.encode("utf-8")) > MAX_REVIEW_FINDING_TEXT_BYTES
        ):
            raise PostMergeReviewError(
                "review_finding_value_invalid",
                "review finding values are empty, invalid, or oversized",
            )
    return payload


def _verify_transport_receipt(
    receipt: Mapping[str, Any],
    *,
    request_id: str,
    provider_id: str,
    attempt: int,
    response_text: str,
) -> None:
    text_bytes = response_text.encode("utf-8")
    text_sha256 = hashlib.sha256(text_bytes).hexdigest()
    execution_material = {
        "request_id": request_id,
        "attempt": int(attempt),
        "idempotency_key": request_id,
        "effective_provider": provider_id,
        "text_chars": len(response_text),
        "text_bytes": len(text_bytes),
        "text_sha256": text_sha256,
    }
    expected_execution_id = "sha256:" + hashlib.sha256(
        json.dumps(
            execution_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != LLM_CHILD_RESULT_SCHEMA
        or int(receipt.get("contract_version") or 0)
        != LLM_CHILD_ENVELOPE_VERSION
        or str(receipt.get("usage_mode") or "") != "off"
        or str(receipt.get("request_id") or "") != request_id
        or int(receipt.get("attempt") or 0) != int(attempt)
        or str(receipt.get("idempotency_key") or "") != request_id
        or str(receipt.get("status") or "") != "ok"
        or int(receipt.get("exit_code") or 0) != 0
        or str(receipt.get("effective_provider") or "") != provider_id
        or int(receipt.get("text_chars") or 0) != len(response_text)
        or int(receipt.get("text_bytes") or 0) != len(text_bytes)
        or str(receipt.get("text_sha256") or "") != text_sha256
        or str(receipt.get("execution_result_id") or "")
        != expected_execution_id
    ):
        raise PostMergeReviewError(
            "reviewer_execution_receipt_invalid",
            "reviewer transport receipt is not an exact successful Codex route",
        )


def _atomic_persist_receipt(
    receipt_dir: Path,
    receipt: Mapping[str, Any],
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = (
        _SAFE_TASK_ID.sub("-", str(receipt["task_id"]).casefold()).strip("-")
        or "task"
    )
    path = receipt_dir / (
        f"{safe_task_id}-queue-{int(receipt['attempt'])}-"
        f"post-merge-review-{receipt['receipt_id']}.json"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{safe_task_id}-post-merge-review-",
        suffix=".tmp",
        dir=receipt_dir,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return path


def verify_post_merge_review_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    implementation_events_path: Path,
    task: Any,
    validation_result: Mapping[str, Any],
    attempt: int,
    implementation_attempt: int,
    baseline_commit: str,
    implementation_commit: str,
    merge_commit: str,
    repository_tree_id: str,
    expected_changed_paths: Sequence[str] | None = None,
    scope_authorized_paths: Sequence[str] = (),
    scope_adjudication_id: str = "",
    implementer_provenance: VerifiedImplementerProvenance | None = None,
) -> ReceiptVerification:
    """Recompute every provider-review binding from Git and typed evidence."""

    try:
        if not isinstance(receipt, Mapping):
            raise PostMergeReviewError(
                "review_receipt_missing",
                "post-merge review receipt must be a mapping",
            )
        material = dict(receipt)
        receipt_id = str(material.pop("receipt_id", "") or "")
        if (
            material.get("schema")
            != POST_MERGE_INDEPENDENT_REVIEW_RECEIPT_SCHEMA
            or not receipt_id
            or content_identity(material) != receipt_id
        ):
            raise PostMergeReviewError(
                "review_receipt_content_identity_invalid",
                "review receipt schema or content identity is invalid",
            )
        task_projection = _task_projection(task)
        implementer = _normalize_implementer_provider(
            str(material.get("implementer_provider") or "")
        )
        expected_provenance = _verify_implementer_provenance(
            implementer_provenance,  # type: ignore[arg-type]
            task_id=task_projection["task_id"],
            implementation_attempt=implementation_attempt,
            implementation_commit=implementation_commit,
            provider_id=implementer,
        )
        finished_event = _verify_implementer_event_membership(
            implementation_events_path,
            implementer_provenance,  # type: ignore[arg-type]
            repo_root=Path(repo_root).resolve(),
        )
        (
            event_scope_authorized_paths,
            event_scope_adjudication_id,
        ) = _scope_authorization_from_implementation_event(
            finished_event,
            repo_root=Path(repo_root).resolve(),
            task=task,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
            expected_changed_paths=expected_changed_paths,
        )
        if (
            tuple(scope_authorized_paths)
            != event_scope_authorized_paths
            or str(scope_adjudication_id or "")
            != event_scope_adjudication_id
        ):
            raise PostMergeReviewError(
                "scope_adjudication_binding_mismatch",
                "caller scope authorization does not match the exact "
                "implementation finish event",
            )
        reviewer_provider = str(
            material.get("reviewer_provider") or ""
        ).strip().casefold()
        if (
            reviewer_provider != CODEX_REVIEWER_PROVIDER
            or reviewer_provider == implementer
            or material.get("providers_independent") is not True
        ):
            raise PostMergeReviewError(
                "reviewer_implementer_not_independent",
                "reviewer and implementer identities are absent, self, or untrusted",
            )
        if (
            material.get("task_id") != task_projection["task_id"]
            or int(material.get("attempt") or 0) != int(attempt)
            or int(material.get("implementation_attempt") or 0)
            != int(implementation_attempt)
            or material.get("baseline_commit") != baseline_commit
            or material.get("implementation_commit") != implementation_commit
            or material.get("merge_commit") != merge_commit
            or material.get("repository_tree_id") != repository_tree_id
            or material.get("repository_write_allowed") is not False
            or material.get("proof_authoritative") is not False
            or material.get("completion_authoritative") is not False
            or material.get("implementer_provenance")
            != expected_provenance
            or material.get("implementer_provenance_id")
            != expected_provenance["provenance_id"]
        ):
            raise PostMergeReviewError(
                "review_receipt_binding_mismatch",
                "review receipt is not bound to the expected task/attempt/commit/tree",
            )

        binding = _collect_repository_binding(
            repo_root=Path(repo_root).resolve(),
            task=task,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
            merge_commit=merge_commit,
            repository_tree_id=repository_tree_id,
            expected_changed_paths=expected_changed_paths,
            scope_authorized_paths=scope_authorized_paths,
            scope_adjudication_id=scope_adjudication_id,
        )
        validation_receipt_id = _verify_validation_evidence(
            validation_result,
            task_validation=task_projection["validation"],
            task_id=task_projection["task_id"],
            merge_commit=merge_commit,
            repository_tree_id=repository_tree_id,
        )
        if (
            material.get("task_binding_id") != binding["task_binding_id"]
            or material.get("changed_paths") != list(binding["changed_paths"])
            or material.get("scope_authorized_paths")
            != list(binding["scope_authorized_paths"])
            or material.get("scope_adjudication_id")
            != binding["scope_adjudication_id"]
            or material.get("scope_authorization_id")
            != binding["scope_authorization_id"]
            or material.get("content_binding_id")
            != content_identity(binding["content_bindings"])
            or material.get("gitlink_binding_id")
            != content_identity(binding["gitlink_bindings"])
            or material.get("diff_binding_id") != binding["diff_binding_id"]
            or material.get("validation_receipt_id")
            != validation_receipt_id
        ):
            raise PostMergeReviewError(
                "review_receipt_binding_mismatch",
                "review receipt top-level Git/task/validation bindings do not "
                "match recomputed evidence",
            )
        request = material.get("review_request")
        if not isinstance(request, Mapping):
            raise PostMergeReviewError(
                "review_request_missing",
                "review receipt does not embed its typed request",
            )
        expected_request = _review_request(
            task=task,
            attempt=attempt,
            implementation_attempt=implementation_attempt,
            implementer_provider=implementer,
            implementer_provenance=implementer_provenance,  # type: ignore[arg-type]
            binding=binding,
            validation_receipt_id=validation_receipt_id,
        )
        if dict(request) != expected_request:
            raise PostMergeReviewError(
                "review_request_binding_mismatch",
                "embedded review request does not match recomputed Git evidence",
            )
        if material.get("review_request_id") != expected_request["request_id"]:
            raise PostMergeReviewError(
                "review_receipt_binding_mismatch",
                "review receipt request identity does not match its bound request",
            )

        response = material.get("review_response")
        response_text = material.get("review_response_text")
        if not isinstance(response, Mapping):
            raise PostMergeReviewError(
                "review_response_missing",
                "review receipt does not embed a structured response",
            )
        if not isinstance(response_text, str):
            raise PostMergeReviewError(
                "review_response_text_missing",
                "review receipt does not embed the exact provider response text",
            )
        response_id = str(material.get("review_response_id") or "")
        if not response_id or content_identity(dict(response)) != response_id:
            raise PostMergeReviewError(
                "review_response_content_identity_invalid",
                "structured review response is not content-addressed",
            )
        normalized_response = _parse_response(
            response_text,
            request=expected_request,
            actual_provider=reviewer_provider,
        )
        if normalized_response != dict(response):
            raise PostMergeReviewError(
                "review_response_text_binding_mismatch",
                "provider response text does not match the structured response",
            )
        decision = str(normalized_response["decision"])
        production_review_route = (
            material.get("production_review_route") is True
        )
        admitted = decision == "approve" and production_review_route
        expected_presence = (
            ReviewPresence.INDEPENDENT.value
            if admitted
            else ReviewPresence.DECLINED.value
            if decision == "changes_required"
            else ReviewPresence.NOT_APPLICABLE.value
        )
        if (
            material.get("decision") != decision
            or material.get("review_presence") != expected_presence
            or material.get("provider_result_admitted") is not admitted
        ):
            raise PostMergeReviewError(
                "review_decision_binding_mismatch",
                "review receipt disposition does not match its response",
            )

        execution = material.get("reviewer_execution_receipt")
        if not isinstance(execution, Mapping):
            raise PostMergeReviewError(
                "reviewer_execution_receipt_missing",
                "review receipt does not embed reviewer execution evidence",
            )
        execution_material = dict(execution)
        execution_id = str(execution_material.pop("receipt_id", "") or "")
        if (
            execution_material.get("schema")
            != POST_MERGE_REVIEWER_EXECUTION_RECEIPT_SCHEMA
            or not execution_id
            or content_identity(execution_material) != execution_id
            or execution_id
            != str(material.get("reviewer_execution_receipt_id") or "")
            or execution_material.get("request_id")
            != expected_request["request_id"]
            or execution_material.get("response_id") != response_id
            or execution_material.get("provider_id") != reviewer_provider
            or execution_material.get("sandbox") != "read-only"
            or execution_material.get("repository_write_allowed") is not False
            or execution_material.get("proof_authoritative") is not False
            or execution_material.get("completion_authoritative") is not False
        ):
            raise PostMergeReviewError(
                "reviewer_execution_binding_invalid",
                "reviewer execution receipt is not content/provider/policy bound",
            )
        _verify_transport_receipt(
            execution_material.get("transport_receipt") or {},
            request_id=str(expected_request["request_id"]),
            provider_id=reviewer_provider,
            attempt=attempt,
            response_text=response_text,
        )
        return ReceiptVerification(
            valid=True,
            reason_code=(
                "independent_review_approved"
                if admitted
                else "independent_review_changes_required"
            ),
            admitted=admitted,
            receipt_id=receipt_id,
        )
    except (PostMergeReviewError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason_code", "review_receipt_verification_failed")
        return ReceiptVerification(
            valid=False,
            reason_code=str(reason),
            detail=str(exc),
            admitted=False,
        )


def perform_post_merge_independent_review(
    *,
    repo_root: Path,
    receipt_dir: Path,
    implementation_events_path: Path,
    task: Any,
    attempt: int,
    implementation_attempt: int,
    baseline_commit: str,
    implementation_commit: str,
    merge_commit: str,
    repository_tree_id: str,
    validation_result: Mapping[str, Any],
    expected_changed_paths: Sequence[str] | None,
    scope_authorized_paths: Sequence[str] = (),
    scope_adjudication_id: str = "",
    implementer_provider: str,
    implementer_provenance: VerifiedImplementerProvenance,
    reviewer: ReviewerCallable | None = None,
) -> PostMergeReviewOutcome:
    """Run one exact independent review and return evidence, never authority."""

    try:
        root = Path(repo_root).resolve()
        production_review_route = reviewer is None
        implementer = _normalize_implementer_provider(implementer_provider)
        finished_event = _verify_implementer_event_membership(
            implementation_events_path,
            implementer_provenance,
            repo_root=root,
        )
        (
            event_scope_authorized_paths,
            event_scope_adjudication_id,
        ) = _scope_authorization_from_implementation_event(
            finished_event,
            repo_root=root,
            task=task,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
            expected_changed_paths=expected_changed_paths,
        )
        if (
            tuple(scope_authorized_paths)
            != event_scope_authorized_paths
            or str(scope_adjudication_id or "")
            != event_scope_adjudication_id
        ):
            raise PostMergeReviewError(
                "scope_adjudication_binding_mismatch",
                "caller scope authorization does not match the exact "
                "implementation finish event",
            )
        binding = _collect_repository_binding(
            repo_root=root,
            task=task,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
            merge_commit=merge_commit,
            repository_tree_id=repository_tree_id,
            expected_changed_paths=expected_changed_paths,
            scope_authorized_paths=scope_authorized_paths,
            scope_adjudication_id=scope_adjudication_id,
        )
        task_projection = dict(binding["task_projection"])
        validation_receipt_id = _verify_validation_evidence(
            validation_result,
            task_validation=task_projection["validation"],
            task_id=task_projection["task_id"],
            merge_commit=merge_commit,
            repository_tree_id=repository_tree_id,
        )
        request = _review_request(
            task=task,
            attempt=attempt,
            implementation_attempt=implementation_attempt,
            implementer_provider=implementer,
            implementer_provenance=implementer_provenance,
            binding=binding,
            validation_receipt_id=validation_receipt_id,
        )
        prompt = _review_prompt(
            request,
            task_projection,
            str(binding["patch_text"]),
        )
        if len(prompt.encode("utf-8")) > MAX_REVIEW_PROMPT_BYTES:
            raise PostMergeReviewError(
                "review_prompt_too_large",
                "task, binding, and patch exceed the total review prompt bound",
            )
        if reviewer is None:
            invocation = _canonical_production_codex_reviewer(
                prompt,
                request,
                repo_root=root,
            )
        else:
            invocation = reviewer(prompt, request)
        if not isinstance(invocation, ReviewerInvocation):
            raise PostMergeReviewError(
                "reviewer_result_invalid",
                "reviewer callable must return ReviewerInvocation",
            )
        reviewer_provider = str(invocation.provider_id or "").strip().casefold()
        if (
            reviewer_provider != CODEX_REVIEWER_PROVIDER
            or reviewer_provider == implementer
            or invocation.sandbox != "read-only"
        ):
            raise PostMergeReviewError(
                "reviewer_implementer_not_independent",
                "reviewer must be the read-only codex_cli provider and differ "
                "from the explicit implementation provider",
            )
        _verify_transport_receipt(
            invocation.transport_receipt,
            request_id=str(request["request_id"]),
            provider_id=reviewer_provider,
            attempt=attempt,
            response_text=invocation.response_text,
        )
        response = _parse_response(
            invocation.response_text,
            request=request,
            actual_provider=reviewer_provider,
        )
        response_id = content_identity(response)
        execution_material = {
            "schema": POST_MERGE_REVIEWER_EXECUTION_RECEIPT_SCHEMA,
            "request_id": request["request_id"],
            "response_id": response_id,
            "provider_id": reviewer_provider,
            "provider_role": "independent_post_merge_review",
            "sandbox": "read-only",
            "transport_receipt": dict(invocation.transport_receipt),
            "repository_write_allowed": False,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }
        execution = {
            **execution_material,
            "receipt_id": content_identity(execution_material),
        }
        approved = response["decision"] == "approve"
        provider_result_admitted = bool(
            approved and production_review_route
        )
        receipt_material = {
            "schema": POST_MERGE_INDEPENDENT_REVIEW_RECEIPT_SCHEMA,
            "task_id": task_projection["task_id"],
            "task_binding_id": binding["task_binding_id"],
            "attempt": int(attempt),
            "implementation_attempt": int(implementation_attempt),
            "baseline_commit": binding["base_commit"],
            "implementation_commit": binding["implementation_commit"],
            "merge_commit": binding["merge_commit"],
            "repository_tree_id": binding["repository_tree_id"],
            "changed_paths": list(binding["changed_paths"]),
            "scope_authorized_paths": list(
                binding["scope_authorized_paths"]
            ),
            "scope_adjudication_id": binding["scope_adjudication_id"],
            "scope_authorization_id": binding["scope_authorization_id"],
            "content_binding_id": content_identity(binding["content_bindings"]),
            "gitlink_binding_id": content_identity(
                binding["gitlink_bindings"]
            ),
            "diff_binding_id": binding["diff_binding_id"],
            "validation_receipt_id": validation_receipt_id,
            "review_request": request,
            "review_request_id": request["request_id"],
            "review_response": response,
            "review_response_text": invocation.response_text,
            "review_response_id": response_id,
            "reviewer_execution_receipt": execution,
            "reviewer_execution_receipt_id": execution["receipt_id"],
            "implementer_provider": implementer,
            "implementer_provenance": implementer_provenance.to_dict(),
            "implementer_provenance_id": (
                implementer_provenance.provenance_id
            ),
            "reviewer_provider": reviewer_provider,
            "providers_independent": True,
            "production_review_route": production_review_route,
            "decision": response["decision"],
            "review_presence": (
                ReviewPresence.INDEPENDENT.value
                if provider_result_admitted
                else ReviewPresence.DECLINED.value
                if response["decision"] == "changes_required"
                else ReviewPresence.NOT_APPLICABLE.value
            ),
            "provider_result_admitted": provider_result_admitted,
            "repository_write_allowed": False,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }
        receipt = {
            **receipt_material,
            "receipt_id": content_identity(receipt_material),
        }
        path = _atomic_persist_receipt(Path(receipt_dir), receipt)
        verification = verify_post_merge_review_receipt(
            receipt,
            repo_root=root,
            implementation_events_path=implementation_events_path,
            task=task,
            validation_result=validation_result,
            attempt=attempt,
            implementation_attempt=implementation_attempt,
            baseline_commit=baseline_commit,
            implementation_commit=implementation_commit,
            merge_commit=merge_commit,
            repository_tree_id=repository_tree_id,
            expected_changed_paths=expected_changed_paths,
            scope_authorized_paths=scope_authorized_paths,
            scope_adjudication_id=scope_adjudication_id,
            implementer_provenance=implementer_provenance,
        )
        if not verification.valid:
            raise PostMergeReviewError(
                verification.reason_code,
                verification.detail,
            )
        event_type = (
            POST_MERGE_INDEPENDENT_REVIEW_EVENT
            if verification.admitted
            else POST_MERGE_INDEPENDENT_REVIEW_DENIED_EVENT
        )
        event = {
            "type": event_type,
            "task_id": task_projection["task_id"],
            "task_binding_id": binding["task_binding_id"],
            "canonical_task_key": task_projection["canonical_task_key"],
            "canonical_task_cid": task_projection["canonical_task_cid"],
            "board_namespace": task_projection["board_namespace"],
            "attempt": int(attempt),
            "implementation_attempt": int(implementation_attempt),
            "implementation_commit": implementation_commit,
            "merge_commit": merge_commit,
            "repository_tree_id": repository_tree_id,
            "review_receipt": receipt,
            "review_receipt_path": str(path),
            "provider_result_admitted": verification.admitted,
            "repository_write_allowed": False,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }
        gate: dict[str, Any] = {}
        if verification.admitted:
            gate = bound_gate_evidence(
                "provider_review",
                task_id=task_projection["task_id"],
                implementation_commit=implementation_commit,
                merge_commit=merge_commit,
                repository_tree_id=repository_tree_id,
                satisfied=True,
                review_presence="independent",
                provider_result_admitted=True,
                review_receipt_id=verification.receipt_id,
                task_binding_id=binding["task_binding_id"],
                canonical_task_key=task_projection["canonical_task_key"],
                canonical_task_cid=task_projection["canonical_task_cid"],
                board_namespace=task_projection["board_namespace"],
            )
        return PostMergeReviewOutcome(
            admitted=verification.admitted,
            reason_code=verification.reason_code,
            receipt=receipt,
            receipt_path=str(path),
            event=event,
            retryable=False,
            acceptance_pending=True,
            _gate_evidence=gate,
            _producer_seal=(
                _LIVE_PRODUCTION_REVIEW_SEAL
                if verification.admitted and production_review_route
                else None
            ),
            _bound_task_id=task_projection["task_id"],
            _bound_task_binding_id=binding["task_binding_id"],
            _bound_canonical_task_key=task_projection[
                "canonical_task_key"
            ],
            _bound_canonical_task_cid=task_projection[
                "canonical_task_cid"
            ],
            _bound_board_namespace=task_projection["board_namespace"],
            _bound_implementation_commit=implementation_commit,
            _bound_merge_commit=merge_commit,
            _bound_repository_tree_id=repository_tree_id,
            _bound_review_receipt_id=verification.receipt_id,
            _receipt_canonical=_canonical_json_bytes(receipt),
            _event_payload_canonical=_canonical_json_bytes(event),
        )
    except (PostMergeReviewError, OSError, ValueError, RuntimeError) as exc:
        reason_code = str(
            getattr(exc, "reason_code", "independent_review_failed")
        )
        return PostMergeReviewOutcome(
            admitted=False,
            reason_code=reason_code,
            detail=str(exc),
            retryable=reason_code
            in {
                "independent_review_failed",
                "reviewer_execution_receipt_missing",
                "reviewer_execution_receipt_invalid",
                "reviewer_result_invalid",
                "review_response_malformed",
                "review_response_size_invalid",
            },
            acceptance_pending=True,
        )


def mint_gate_from_live_outcome(
    outcome: PostMergeReviewOutcome,
    appended_event: Mapping[str, Any],
    *,
    events_path: Path,
) -> dict[str, Any]:
    """Mint a gate only from a live producer seal after durable event append.

    Persisted receipts and dependency-injected reviewer results intentionally
    cannot cross this boundary. The exact appended event must be a member of
    the current strict manifest-bound ledger. After process restart, callers
    must rerun the production Codex review; structural ``verify_*`` results
    alone must never be wired to completion authority.
    """

    if (
        not isinstance(outcome, PostMergeReviewOutcome)
        or outcome._producer_seal is not _LIVE_PRODUCTION_REVIEW_SEAL
        or not outcome.admitted
        or not isinstance(appended_event, Mapping)
    ):
        return {}
    try:
        ledger_events = _strict_event_ledger(events_path)
    except (PostMergeReviewError, OSError, TypeError, ValueError):
        return {}
    try:
        receipt_snapshot = json.loads(outcome._receipt_canonical)
        event_snapshot = json.loads(outcome._event_payload_canonical)
        if (
            not isinstance(receipt_snapshot, dict)
            or not isinstance(event_snapshot, dict)
            or _canonical_json_bytes(receipt_snapshot)
            != outcome._receipt_canonical
            or _canonical_json_bytes(event_snapshot)
            != outcome._event_payload_canonical
            or _canonical_json_bytes(outcome.receipt)
            != outcome._receipt_canonical
            or _canonical_json_bytes(outcome.event)
            != outcome._event_payload_canonical
        ):
            return {}
        receipt_material = dict(receipt_snapshot)
        receipt_id = str(receipt_material.pop("receipt_id", "") or "")
        if (
            receipt_id != outcome._bound_review_receipt_id
            or content_identity(receipt_material) != receipt_id
            or event_snapshot.get("review_receipt") != receipt_snapshot
            or event_snapshot.get("task_id") != outcome._bound_task_id
            or event_snapshot.get("task_binding_id")
            != outcome._bound_task_binding_id
            or event_snapshot.get("canonical_task_key")
            != outcome._bound_canonical_task_key
            or event_snapshot.get("canonical_task_cid")
            != outcome._bound_canonical_task_cid
            or event_snapshot.get("board_namespace")
            != outcome._bound_board_namespace
            or event_snapshot.get("implementation_commit")
            != outcome._bound_implementation_commit
            or event_snapshot.get("merge_commit")
            != outcome._bound_merge_commit
            or event_snapshot.get("repository_tree_id")
            != outcome._bound_repository_tree_id
            or event_snapshot.get("provider_result_admitted") is not True
            or event_snapshot.get("repository_write_allowed") is not False
            or event_snapshot.get("proof_authoritative") is not False
            or event_snapshot.get("completion_authoritative") is not False
        ):
            return {}
        durable_event = dict(appended_event)
        if durable_event not in ledger_events:
            return {}
        durable_payload = {
            key: value
            for key, value in durable_event.items()
            if key not in _CANONICAL_EVENT_ENVELOPE_FIELDS
        }
        if durable_payload != event_snapshot:
            return {}
        event = dict(durable_event)
        event_id = str(event.pop("event_id", "") or "")
        encoded = json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        sequence = int(appended_event.get("sequence"))
        attempt = int(appended_event.get("attempt"))
        expected_attempt = int(event_snapshot.get("attempt"))
    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ):
        return {}
    if (
        not event_id
        or event_id != "sha256:" + hashlib.sha256(encoded).hexdigest()
        or appended_event.get("type")
        != POST_MERGE_INDEPENDENT_REVIEW_EVENT
        or sequence < 1
        or not str(appended_event.get("stream_id") or "")
        or not str(appended_event.get("snapshot_id") or "")
        or appended_event.get("task_id") != outcome._bound_task_id
        or attempt != expected_attempt
        or appended_event.get("implementation_commit")
        != outcome._bound_implementation_commit
        or appended_event.get("merge_commit")
        != outcome._bound_merge_commit
        or appended_event.get("repository_tree_id")
        != outcome._bound_repository_tree_id
        or appended_event.get("review_receipt") != receipt_snapshot
    ):
        return {}
    return bound_gate_evidence(
        "provider_review",
        task_id=outcome._bound_task_id,
        implementation_commit=outcome._bound_implementation_commit,
        merge_commit=outcome._bound_merge_commit,
        repository_tree_id=outcome._bound_repository_tree_id,
        satisfied=True,
        review_presence="independent",
        provider_result_admitted=True,
        review_receipt_id=outcome._bound_review_receipt_id,
        task_binding_id=outcome._bound_task_binding_id,
        canonical_task_key=outcome._bound_canonical_task_key,
        canonical_task_cid=outcome._bound_canonical_task_cid,
        board_namespace=outcome._bound_board_namespace,
    )


__all__ = [
    "ALLOWED_IMPLEMENTER_PROVIDERS",
    "CODEX_REVIEWER_PROVIDER",
    "IMPLEMENTER_LOG_BINDING_SCOPE",
    "POST_MERGE_INDEPENDENT_REVIEW_DENIED_EVENT",
    "POST_MERGE_INDEPENDENT_REVIEW_EVENT",
    "POST_MERGE_INDEPENDENT_REVIEW_FAILED_EVENT",
    "POST_MERGE_INDEPENDENT_REVIEW_RECEIPT_SCHEMA",
    "POST_MERGE_INDEPENDENT_REVIEW_REQUEST_SCHEMA",
    "POST_MERGE_INDEPENDENT_REVIEW_RESPONSE_SCHEMA",
    "POST_MERGE_REVIEWER_EXECUTION_RECEIPT_SCHEMA",
    "VERIFIED_IMPLEMENTER_PROVENANCE_SCHEMA",
    "PostMergeReviewError",
    "PostMergeReviewOutcome",
    "ReceiptVerification",
    "ReviewerCallable",
    "ReviewerInvocation",
    "VerifiedImplementerProvenance",
    "perform_post_merge_independent_review",
    "mint_gate_from_live_outcome",
    "post_merge_task_binding_id",
    "verified_implementer_provenance_from_events",
    "verified_implementer_provenance_from_ledger",
    "verify_post_merge_review_receipt",
]
