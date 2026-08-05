"""Authoritative, current-tree evidence for proof-test-reuse tasks.

This module is deliberately a boundary adapter.  Board labels and historical
queue state are useful discovery inputs, but neither is completion authority.
The collector emits authority only after it can join a task's canonical board
identity, reviewed completion provenance, and fresh validation to one exact
repository observation.  Every other outcome is represented by a typed gap.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from .proof_cached_test_validation import (
    ProofCachedTestValidationError,
    ProofCachedTestValidationReceipt,
    validation_command_identity,
)
from .proof_test_reuse_current_tree_gate import TaskCompletionProvenanceKind

PROOF_TEST_REUSE_TASK_EVIDENCE_VERSION: Final = 1
PROOF_TEST_REUSE_TASK_EVIDENCE_INTERFACE: Final = "ProofTestReuseTaskEvidence@1"
TASK_VALIDATION_PROVENANCE_INTERFACE: Final = "TaskValidationProvenance@1"
TASK_EVIDENCE_GAP_INTERFACE: Final = "TaskEvidenceGap@1"
PROOF_TEST_REUSE_TASK_EVIDENCE_COLLECTION_INTERFACE: Final = (
    "ProofTestReuseTaskEvidenceCollection@1"
)

PROOF_TEST_REUSE_TASK_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-task-evidence@1"
)
TASK_VALIDATION_PROVENANCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-validation-provenance@1"
)
TASK_EVIDENCE_GAP_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/task-evidence-gap@1"
)
PROOF_TEST_REUSE_TASK_EVIDENCE_COLLECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-test-reuse-task-evidence-collection@1"
)

DEFAULT_EVIDENCE_FRESHNESS_SECONDS: Final = 300.0
REVIEW_REQUIRED_WITHOUT_QUEUE: Final = frozenset(
    {"PTR-000", "PTR-001", "PTR-011", "PTR-041"}
)
_PROOF_REUSE_OFF_PREFIX: Final = "IPFS_TEST_PROOF_REUSE_MODE=off "
_AUTHORITATIVE: Final = "authoritative"
_NO_AUTHORITY: Final = "none"


class TaskEvidenceGapKind(str, Enum):
    """Closed reasons for withholding task-completion authority."""

    BOARD_UNVALIDATED = "board_unvalidated"
    BOARD_MALFORMED = "board_malformed"
    BOARD_POPULATION_MISMATCH = "board_population_mismatch"
    TASK_MALFORMED = "task_malformed"
    TASK_DUPLICATE = "task_duplicate"
    TASK_CID_MISSING = "task_cid_missing"
    TASK_CID_MISMATCH = "task_cid_mismatch"
    COMPLETION_PROVENANCE_MISSING = "completion_provenance_missing"
    COMPLETION_PROVENANCE_MALFORMED = "completion_provenance_malformed"
    COMPLETION_PROVENANCE_CONTRADICTORY = "completion_provenance_contradictory"
    QUEUE_RECORD_UNSUCCESSFUL = "queue_record_unsuccessful"
    ANCESTRY_UNAVAILABLE = "ancestry_unavailable"
    ANCESTRY_UNVERIFIED = "ancestry_unverified"
    APPROVAL_MISSING = "approval_missing"
    APPROVAL_MALFORMED = "approval_malformed"
    APPROVAL_UNVERIFIED = "approval_unverified"
    VALIDATION_MISSING = "validation_missing"
    VALIDATION_MALFORMED = "validation_malformed"
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_STALE = "validation_stale"
    VALIDATION_COMMAND_MISMATCH = "validation_command_mismatch"
    VALIDATION_BINDING_MISMATCH = "validation_binding_mismatch"
    PROOF_REUSE_NOT_OFF = "proof_reuse_not_off"
    ORDINARY_SKIP = "ordinary_skip"
    PROOF_SKIP_VERIFIER_UNAVAILABLE = "proof_skip_verifier_unavailable"
    PROOF_SKIP_UNVERIFIED = "proof_skip_unverified"
    UNEXPECTED_INPUT = "unexpected_input"


class ProofTestReuseTaskEvidenceError(ValueError):
    """Raised only for invalid collector construction or contract creation."""


def _text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _record(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        candidate = to_dict()
        if isinstance(candidate, Mapping):
            return candidate
    fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        return {name: getattr(value, name) for name in fields}
    return {}


def _value(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _boolean(record: Mapping[str, Any], *names: str) -> bool | None:
    value = _value(record, *names)
    return value if isinstance(value, bool) else None


def _integer(record: Mapping[str, Any], *names: str) -> int | None:
    value = _value(record, *names)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _clock_milliseconds(clock: Callable[[], float]) -> int:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProofTestReuseTaskEvidenceError(
            "clock must return seconds since the Unix epoch"
        )
    return int(float(value) * 1_000)


def _immutable_record_cid(
    record: Mapping[str, Any],
    *claim_names: str,
    claim_required: bool,
) -> str:
    """Validate an identity claim over the record with identity fields removed."""

    claims = [_text(record.get(name)) for name in claim_names if record.get(name)]
    if len(set(claims)) > 1:
        return ""
    payload = {
        str(key): value
        for key, value in record.items()
        if str(key) not in set(claim_names)
    }
    try:
        derived = content_identity(payload)
    except Exception:
        # Raw daemon merge rows often contain floats/nested non-canonical
        # metadata.  Callers should project through
        # ``project_managed_merge_queue_record``; never raise into collect().
        return ""
    if claims and claims[0] != derived:
        return ""
    if claim_required and not claims:
        return ""
    return derived


def project_managed_merge_queue_record(raw: Any) -> dict[str, Any] | None:
    """Project a daemon merge-queue row into a collector-safe sealed receipt.

    The live merge queue persists floats, nested metadata, and baguqeera CIDs
    under several key spellings.  The task-evidence collector seals merge
    authority with ``content_identity``, which rejects floats.  This projector
    keeps only the completion claim (task id, task CID, status, commit) and
    derives a stable ``merge_receipt_cid`` over that body.

    Returns ``None`` when the row is not a successful completion claim.  This
    never invents a task completion: absent, unsuccessful, or incomplete rows
    are dropped rather than upgraded.
    """

    record = _record(raw)
    if not record:
        return None
    task_id = _text(_value(record, "task_id", "id"))
    if not task_id:
        return None
    status = _text(_value(record, "status", "state")).lower()
    if status not in {"completed", "merged"}:
        return None
    task_cid = _text(
        _value(record, "task_cid", "canonical_task_cid", "canonical_task_id")
    )
    commit = _text(
        _value(record, "merged_commit_id", "commit_sha", "commit_id", "merge_commit")
    )
    if not task_cid or not commit:
        return None
    body = {
        "task_id": task_id,
        "canonical_task_cid": task_cid,
        "status": "completed",
        "commit_sha": commit,
    }
    try:
        receipt_cid = content_identity(body)
    except Exception:
        return None
    return {
        **body,
        "merge_receipt_cid": receipt_cid,
    }


def project_managed_merge_queue_records(
    values: Iterable[Any],
) -> tuple[dict[str, Any], ...]:
    """Project many merge-queue rows; keep the latest successful row per task."""

    latest: dict[str, dict[str, Any]] = {}
    for value in values:
        projected = project_managed_merge_queue_record(value)
        if projected is None:
            continue
        task_id = projected["task_id"]
        # Prefer later records when raw enqueued_at is available.
        raw = _record(value)
        order = 0
        enqueued = raw.get("enqueued_at")
        if isinstance(enqueued, (int, float)) and not isinstance(enqueued, bool):
            order = int(float(enqueued) * 1000)
        prior = latest.get(task_id)
        if prior is None or order >= int(prior.get("_order", 0)):
            projected = dict(projected)
            projected["_order"] = order
            latest[task_id] = projected
    cleaned: list[dict[str, Any]] = []
    for task_id in sorted(latest):
        body = dict(latest[task_id])
        body.pop("_order", None)
        cleaned.append(body)
    return tuple(cleaned)


def _contract_payload(
    payload: Mapping[str, Any],
    *,
    schema: str,
    interface: str,
    allowed: frozenset[str],
    artifact: str,
) -> dict[str, Any]:
    """Strictly authenticate a persisted canonical contract record."""

    if not isinstance(payload, Mapping):
        raise ProofTestReuseTaskEvidenceError(f"{artifact} must be a mapping")
    if set(payload).difference(allowed | {"content_id"}):
        raise ProofTestReuseTaskEvidenceError(
            f"{artifact} contains unsupported fields"
        )
    body = {str(key): value for key, value in payload.items() if key != "content_id"}
    if (
        body.get("schema") != schema
        or body.get("interface") != interface
        or body.get("contract_version") != PROOF_TEST_REUSE_TASK_EVIDENCE_VERSION
    ):
        raise ProofTestReuseTaskEvidenceError(
            f"{artifact} has an unsupported contract discriminator"
        )
    claimed = _text(payload.get("content_id"))
    if claimed and claimed != content_identity(body):
        raise ProofTestReuseTaskEvidenceError(
            f"{artifact} content identity does not match its payload"
        )
    return body


@dataclass(frozen=True, slots=True)
class TaskEvidenceGap(CanonicalContract):
    """Non-authoritative, content-addressed explanation of a missing premise."""

    SCHEMA: ClassVar[str] = TASK_EVIDENCE_GAP_SCHEMA

    task_id: str
    kind: TaskEvidenceGapKind | str
    detail: str
    input_cid: str = ""

    def __post_init__(self) -> None:
        task_id = _text(self.task_id)
        detail = _text(self.detail)
        if not task_id or not detail:
            raise ProofTestReuseTaskEvidenceError("gap task_id and detail are required")
        try:
            kind = (
                self.kind
                if isinstance(self.kind, TaskEvidenceGapKind)
                else TaskEvidenceGapKind(_text(self.kind))
            )
        except ValueError as exc:
            raise ProofTestReuseTaskEvidenceError("unsupported task evidence gap") from exc
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "detail", detail[:512])
        object.__setattr__(self, "input_cid", _text(self.input_cid))

    @property
    def interface(self) -> str:
        return TASK_EVIDENCE_GAP_INTERFACE

    @property
    def authority(self) -> str:
        return _NO_AUTHORITY

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def reason_code(self) -> str:
        return self.kind.value

    @property
    def gap_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_TEST_REUSE_TASK_EVIDENCE_VERSION,
            "interface": self.interface,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "input_cid": self.input_cid,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskEvidenceGap:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=TASK_EVIDENCE_GAP_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "task_id",
                    "kind",
                    "reason_code",
                    "detail",
                    "input_cid",
                    "authority",
                }
            ),
            artifact="task evidence gap",
        )
        result = cls(
            task_id=body.get("task_id", ""),
            kind=body.get("kind", ""),
            detail=body.get("detail", ""),
            input_cid=body.get("input_cid", ""),
        )
        if (
            body.get("reason_code") != result.reason_code
            or body.get("authority") != result.authority
        ):
            raise ProofTestReuseTaskEvidenceError(
                "task evidence gap carries contradictory derived fields"
            )
        return result


@dataclass(frozen=True, slots=True)
class TaskValidationProvenance(CanonicalContract):
    """A fresh validation receipt bound to one task and repository state."""

    SCHEMA: ClassVar[str] = TASK_VALIDATION_PROVENANCE_SCHEMA

    task_id: str
    goal_id: str
    task_cid: str
    validation_command: str
    validation_receipt_cid: str
    disposition: str
    repository_id: str
    repository_state_cid: str
    git_commit_id: str
    git_tree_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    dirty: bool
    dirty_overlay_cid: str
    observed_at_ms: int
    fresh_until_ms: int
    locally_verified: bool
    receipt: Mapping[str, Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "goal_id",
            "task_cid",
            "validation_command",
            "validation_receipt_cid",
            "repository_id",
            "repository_state_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "repository_forest_cid",
            "dirty_overlay_cid",
        ):
            value = _text(getattr(self, name))
            if not value:
                raise ProofTestReuseTaskEvidenceError(f"{name} is required")
            object.__setattr__(self, name, value)
        if self.disposition not in {"executed", "proof_backed_skip"}:
            raise ProofTestReuseTaskEvidenceError("unsupported validation disposition")
        if not isinstance(self.dirty, bool) or self.locally_verified is not True:
            raise ProofTestReuseTaskEvidenceError(
                "validation must be locally verified and carry a boolean dirty flag"
            )
        if (
            isinstance(self.observed_at_ms, bool)
            or isinstance(self.fresh_until_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or not isinstance(self.fresh_until_ms, int)
            or self.observed_at_ms < 0
            or self.fresh_until_ms <= self.observed_at_ms
        ):
            raise ProofTestReuseTaskEvidenceError("invalid validation freshness window")
        if validation_command_identity(self.validation_command) != _text(
            _value(self.receipt, "validation_command_cid", "command_cid")
        ):
            raise ProofTestReuseTaskEvidenceError(
                "validation receipt does not bind the declared command"
            )
        receipt_cid = _immutable_record_cid(
            self.receipt,
            "validation_receipt_cid",
            "receipt_id",
            "content_id",
            claim_required=True,
        )
        if receipt_cid != self.validation_receipt_cid:
            raise ProofTestReuseTaskEvidenceError(
                "validation receipt content identity does not match its binding"
            )

    @property
    def interface(self) -> str:
        return TASK_VALIDATION_PROVENANCE_INTERFACE

    @property
    def provenance_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_TEST_REUSE_TASK_EVIDENCE_VERSION,
            "interface": self.interface,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "task_cid": self.task_cid,
            "validation_command": self.validation_command,
            "validation_command_cid": validation_command_identity(
                self.validation_command
            ),
            "validation_receipt_cid": self.validation_receipt_cid,
            "disposition": self.disposition,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "git_commit_id": self.git_commit_id,
            "git_tree_id": self.git_tree_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "dirty": self.dirty,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "observed_at_ms": self.observed_at_ms,
            "fresh_until_ms": self.fresh_until_ms,
            "locally_verified": self.locally_verified,
            "receipt": dict(self.receipt),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskValidationProvenance:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=TASK_VALIDATION_PROVENANCE_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "task_id",
                    "goal_id",
                    "task_cid",
                    "validation_command",
                    "validation_command_cid",
                    "validation_receipt_cid",
                    "disposition",
                    "repository_id",
                    "repository_state_cid",
                    "git_commit_id",
                    "git_tree_id",
                    "gitlink_state_cid",
                    "repository_forest_cid",
                    "dirty",
                    "dirty_overlay_cid",
                    "observed_at_ms",
                    "fresh_until_ms",
                    "locally_verified",
                    "receipt",
                }
            ),
            artifact="task validation provenance",
        )
        receipt = body.get("receipt")
        if not isinstance(receipt, Mapping):
            raise ProofTestReuseTaskEvidenceError(
                "task validation provenance receipt must be a mapping"
            )
        result = cls(
            task_id=body.get("task_id", ""),
            goal_id=body.get("goal_id", ""),
            task_cid=body.get("task_cid", ""),
            validation_command=body.get("validation_command", ""),
            validation_receipt_cid=body.get("validation_receipt_cid", ""),
            disposition=body.get("disposition", ""),
            repository_id=body.get("repository_id", ""),
            repository_state_cid=body.get("repository_state_cid", ""),
            git_commit_id=body.get("git_commit_id", ""),
            git_tree_id=body.get("git_tree_id", ""),
            gitlink_state_cid=body.get("gitlink_state_cid", ""),
            repository_forest_cid=body.get("repository_forest_cid", ""),
            dirty=body.get("dirty"),
            dirty_overlay_cid=body.get("dirty_overlay_cid", ""),
            observed_at_ms=body.get("observed_at_ms"),
            fresh_until_ms=body.get("fresh_until_ms"),
            locally_verified=body.get("locally_verified"),
            receipt=dict(receipt),
        )
        if body.get("validation_command_cid") != validation_command_identity(
            result.validation_command
        ):
            raise ProofTestReuseTaskEvidenceError(
                "task validation provenance command CID is contradictory"
            )
        return result


@dataclass(frozen=True, slots=True)
class ProofTestReuseTaskEvidence(CanonicalContract):
    """Replayable authority for one task on one exact current tree."""

    SCHEMA: ClassVar[str] = PROOF_TEST_REUSE_TASK_EVIDENCE_SCHEMA

    task_id: str
    goal_id: str
    task_cid: str
    board_cid: str
    repository_id: str
    repository_state_cid: str
    git_commit_id: str
    git_tree_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    dirty: bool
    dirty_overlay_cid: str
    policy_cid: str
    capability_cid: str
    verifying_key_cid: str
    circuit_cid: str
    task_provenance: Mapping[str, Any]
    validation: TaskValidationProvenance

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "goal_id",
            "task_cid",
            "board_cid",
            "repository_id",
            "repository_state_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "repository_forest_cid",
            "dirty_overlay_cid",
        ):
            value = _text(getattr(self, name))
            if not value:
                raise ProofTestReuseTaskEvidenceError(f"{name} is required")
            object.__setattr__(self, name, value)
        if not isinstance(self.dirty, bool):
            raise ProofTestReuseTaskEvidenceError("dirty must be boolean")
        for name in (
            "policy_cid",
            "capability_cid",
            "verifying_key_cid",
            "circuit_cid",
        ):
            object.__setattr__(self, name, _text(getattr(self, name)))
        if not isinstance(self.task_provenance, Mapping) or not self.task_provenance:
            raise ProofTestReuseTaskEvidenceError("task provenance is required")
        object.__setattr__(self, "task_provenance", dict(self.task_provenance))
        if not isinstance(self.validation, TaskValidationProvenance):
            raise ProofTestReuseTaskEvidenceError("validation provenance is required")
        validation_bindings = {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "task_cid": self.task_cid,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "git_commit_id": self.git_commit_id,
            "git_tree_id": self.git_tree_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "dirty": self.dirty,
            "dirty_overlay_cid": self.dirty_overlay_cid,
        }
        if any(
            getattr(self.validation, name) != expected
            for name, expected in validation_bindings.items()
        ):
            raise ProofTestReuseTaskEvidenceError(
                "validation does not bind the exact task and repository observation"
            )

    @property
    def interface(self) -> str:
        return PROOF_TEST_REUSE_TASK_EVIDENCE_INTERFACE

    @property
    def authority(self) -> str:
        return _AUTHORITATIVE

    @property
    def authoritative(self) -> bool:
        return True

    @property
    def evidence_cid(self) -> str:
        return self.content_id

    @property
    def provenance_cid(self) -> str:
        return self.content_id

    @property
    def validation_command(self) -> str:
        return self.validation.validation_command

    @property
    def validation_receipt_cid(self) -> str:
        return self.validation.validation_receipt_cid

    def _payload(self) -> dict[str, Any]:
        validation = self.validation
        return {
            "contract_version": PROOF_TEST_REUSE_TASK_EVIDENCE_VERSION,
            "interface": self.interface,
            "authority": self.authority,
            "state": "verified_complete",
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "task_cid": self.task_cid,
            "board_cid": self.board_cid,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "commit_id": self.git_commit_id,
            "git_commit_id": self.git_commit_id,
            "tree_id": self.git_tree_id,
            "git_tree_id": self.git_tree_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "gitlink_closure_complete": True,
            "repository_forest_cid": self.repository_forest_cid,
            "dirty": self.dirty,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "policy_cid": self.policy_cid,
            "capability_cid": self.capability_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "circuit_cid": self.circuit_cid,
            "validation_command": validation.validation_command,
            "validation_command_cid": validation_command_identity(
                validation.validation_command
            ),
            "validation_receipt_cid": validation.validation_receipt_cid,
            "validation_disposition": validation.disposition,
            "validation_provenance_cid": validation.provenance_cid,
            "validation": validation.to_record(),
            "task_provenance": dict(self.task_provenance),
            "observed_at_ms": validation.observed_at_ms,
            "fresh_until_ms": validation.fresh_until_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofTestReuseTaskEvidence:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=PROOF_TEST_REUSE_TASK_EVIDENCE_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "authority",
                    "state",
                    "task_id",
                    "goal_id",
                    "task_cid",
                    "board_cid",
                    "repository_id",
                    "repository_state_cid",
                    "commit_id",
                    "git_commit_id",
                    "tree_id",
                    "git_tree_id",
                    "gitlink_state_cid",
                    "gitlink_closure_complete",
                    "repository_forest_cid",
                    "dirty",
                    "dirty_overlay_cid",
                    "policy_cid",
                    "capability_cid",
                    "verifying_key_cid",
                    "circuit_cid",
                    "validation_command",
                    "validation_command_cid",
                    "validation_receipt_cid",
                    "validation_disposition",
                    "validation_provenance_cid",
                    "validation",
                    "task_provenance",
                    "observed_at_ms",
                    "fresh_until_ms",
                }
            ),
            artifact="proof-test-reuse task evidence",
        )
        validation_record = body.get("validation")
        provenance = body.get("task_provenance")
        if not isinstance(validation_record, Mapping) or not isinstance(
            provenance, Mapping
        ):
            raise ProofTestReuseTaskEvidenceError(
                "task evidence requires embedded validation and task provenance"
            )
        validation = TaskValidationProvenance.from_dict(validation_record)
        result = cls(
            task_id=body.get("task_id", ""),
            goal_id=body.get("goal_id", ""),
            task_cid=body.get("task_cid", ""),
            board_cid=body.get("board_cid", ""),
            repository_id=body.get("repository_id", ""),
            repository_state_cid=body.get("repository_state_cid", ""),
            git_commit_id=body.get("git_commit_id", ""),
            git_tree_id=body.get("git_tree_id", ""),
            gitlink_state_cid=body.get("gitlink_state_cid", ""),
            repository_forest_cid=body.get("repository_forest_cid", ""),
            dirty=body.get("dirty"),
            dirty_overlay_cid=body.get("dirty_overlay_cid", ""),
            policy_cid=body.get("policy_cid", ""),
            capability_cid=body.get("capability_cid", ""),
            verifying_key_cid=body.get("verifying_key_cid", ""),
            circuit_cid=body.get("circuit_cid", ""),
            task_provenance=dict(provenance),
            validation=validation,
        )
        derived = {
            "authority": result.authority,
            "state": "verified_complete",
            "commit_id": result.git_commit_id,
            "tree_id": result.git_tree_id,
            "gitlink_closure_complete": True,
            "validation_command": validation.validation_command,
            "validation_command_cid": validation_command_identity(
                validation.validation_command
            ),
            "validation_receipt_cid": validation.validation_receipt_cid,
            "validation_disposition": validation.disposition,
            "validation_provenance_cid": validation.provenance_cid,
            "observed_at_ms": validation.observed_at_ms,
            "fresh_until_ms": validation.fresh_until_ms,
        }
        if any(body.get(name) != value for name, value in derived.items()):
            raise ProofTestReuseTaskEvidenceError(
                "task evidence carries contradictory derived fields"
            )
        return result


@dataclass(frozen=True, slots=True)
class ProofTestReuseTaskEvidenceCollection(CanonicalContract):
    """Atomic collector outcome for the population in one validated board."""

    SCHEMA: ClassVar[str] = PROOF_TEST_REUSE_TASK_EVIDENCE_COLLECTION_SCHEMA

    board_cid: str
    required_task_ids: tuple[str, ...]
    evidence: tuple[ProofTestReuseTaskEvidence, ...]
    gaps: tuple[TaskEvidenceGap, ...]
    evaluated_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "board_cid", _text(self.board_cid))
        required = tuple(_text(task_id) for task_id in self.required_task_ids)
        if (
            any(not task_id for task_id in required)
            or len(required) != len(set(required))
            or required != tuple(sorted(required))
        ):
            raise ProofTestReuseTaskEvidenceError(
                "required task IDs must be nonempty, unique, and sorted"
            )
        if not all(
            isinstance(item, ProofTestReuseTaskEvidence) for item in self.evidence
        ) or not all(isinstance(item, TaskEvidenceGap) for item in self.gaps):
            raise ProofTestReuseTaskEvidenceError(
                "collection children must be typed task evidence or gaps"
            )
        evidence_ids = tuple(item.task_id for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ProofTestReuseTaskEvidenceError(
                "collection contains duplicate task evidence"
            )
        required_set = set(required)
        if any(task_id not in required_set for task_id in evidence_ids):
            raise ProofTestReuseTaskEvidenceError(
                "collection evidence names a task outside the required population"
            )
        if any(
            gap.task_id != "*" and gap.task_id not in required_set
            for gap in self.gaps
        ):
            raise ProofTestReuseTaskEvidenceError(
                "collection gap names a task outside the required population"
            )
        if (
            isinstance(self.evaluated_at_ms, bool)
            or not isinstance(self.evaluated_at_ms, int)
            or self.evaluated_at_ms < 0
        ):
            raise ProofTestReuseTaskEvidenceError(
                "collection evaluated_at_ms must be a nonnegative integer"
            )
        object.__setattr__(self, "required_task_ids", required)
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "gaps", tuple(self.gaps))

    @property
    def interface(self) -> str:
        return PROOF_TEST_REUSE_TASK_EVIDENCE_COLLECTION_INTERFACE

    @property
    def evidence_by_task(self) -> dict[str, ProofTestReuseTaskEvidence]:
        return {item.task_id: item for item in self.evidence}

    @property
    def gaps_by_task(self) -> dict[str, tuple[TaskEvidenceGap, ...]]:
        grouped: dict[str, list[TaskEvidenceGap]] = defaultdict(list)
        for gap in self.gaps:
            grouped[gap.task_id].append(gap)
        return {key: tuple(value) for key, value in grouped.items()}

    @property
    def authoritative(self) -> bool:
        return (
            bool(self.required_task_ids)
            and not self.gaps
            and set(self.evidence_by_task) == set(self.required_task_ids)
        )

    @property
    def authority(self) -> str:
        return _AUTHORITATIVE if self.authoritative else _NO_AUTHORITY

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROOF_TEST_REUSE_TASK_EVIDENCE_VERSION,
            "interface": self.interface,
            "board_cid": self.board_cid,
            "required_task_ids": list(self.required_task_ids),
            "evidence": [item.to_record() for item in self.evidence],
            "gaps": [item.to_record() for item in self.gaps],
            "evaluated_at_ms": self.evaluated_at_ms,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> ProofTestReuseTaskEvidenceCollection:
        body = _contract_payload(
            payload,
            schema=cls.SCHEMA,
            interface=PROOF_TEST_REUSE_TASK_EVIDENCE_COLLECTION_INTERFACE,
            allowed=frozenset(
                {
                    "schema",
                    "contract_version",
                    "interface",
                    "board_cid",
                    "required_task_ids",
                    "evidence",
                    "gaps",
                    "evaluated_at_ms",
                    "authority",
                }
            ),
            artifact="proof-test-reuse task evidence collection",
        )
        required = body.get("required_task_ids")
        evidence = body.get("evidence")
        gaps = body.get("gaps")
        if (
            not isinstance(required, list)
            or not isinstance(evidence, list)
            or not isinstance(gaps, list)
        ):
            raise ProofTestReuseTaskEvidenceError(
                "task evidence collection populations must be lists"
            )
        if not all(isinstance(item, Mapping) for item in (*evidence, *gaps)):
            raise ProofTestReuseTaskEvidenceError(
                "task evidence collection children must be mappings"
            )
        result = cls(
            board_cid=body.get("board_cid", ""),
            required_task_ids=tuple(required),
            evidence=tuple(
                ProofTestReuseTaskEvidence.from_dict(item) for item in evidence
            ),
            gaps=tuple(TaskEvidenceGap.from_dict(item) for item in gaps),
            evaluated_at_ms=body.get("evaluated_at_ms"),
        )
        if body.get("authority") != result.authority:
            raise ProofTestReuseTaskEvidenceError(
                "task evidence collection authority is contradictory"
            )
        return result


@dataclass(frozen=True, slots=True)
class _Task:
    task_id: str
    goal_id: str
    task_cid: str
    validation_command: str


@dataclass(frozen=True, slots=True)
class ProofTestReuseTaskEvidenceCollector:
    """Collect task authority from independently verifiable retained inputs."""

    repository_id: str
    repository_state_cid: str
    git_commit_id: str
    git_tree_id: str
    gitlink_state_cid: str
    repository_forest_cid: str
    dirty: bool
    dirty_overlay_cid: str
    board_namespace: str = "proof-backed-test-reuse-v1"
    objective_revision: str = ""
    policy_cid: str = ""
    capability_cid: str = ""
    verifying_key_cid: str = ""
    circuit_cid: str = ""
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS
    ancestry_verifier: Callable[[str, str], bool] | None = field(
        default=None, repr=False, compare=False
    )
    proof_skip_verifier: Callable[[ProofCachedTestValidationReceipt], bool] | None = (
        field(default=None, repr=False, compare=False)
    )
    approval_verifier: Callable[[Mapping[str, Any]], bool] | None = field(
        default=None, repr=False, compare=False
    )
    clock: Callable[[], float] = field(default=time.time, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "repository_state_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "repository_forest_cid",
            "dirty_overlay_cid",
            "board_namespace",
        ):
            value = _text(getattr(self, name))
            if not value:
                raise ProofTestReuseTaskEvidenceError(f"{name} is required")
            object.__setattr__(self, name, value)
        if not isinstance(self.dirty, bool):
            raise ProofTestReuseTaskEvidenceError("dirty must be boolean")
        if (
            isinstance(self.freshness_seconds, bool)
            or not isinstance(self.freshness_seconds, (int, float))
            or not 0 < float(self.freshness_seconds) <= 3_600
        ):
            raise ProofTestReuseTaskEvidenceError("freshness_seconds is invalid")

    def collect(
        self,
        validated_board: Any = None,
        *,
        board_validation: Any = None,
        task_records: Iterable[Any] = (),
        tasks: Iterable[Any] = (),
        merge_queue_records: Iterable[Any] = (),
        merge_records: Iterable[Any] = (),
        validation_receipts: Iterable[Any] = (),
        approval_records: Iterable[Any] = (),
        retrospective_records: Iterable[Any] = (),
        history_records: Iterable[Any] = (),
    ) -> ProofTestReuseTaskEvidenceCollection:
        """Collect the complete population named by a validated board.

        ``validated_board`` is the board-validator result.  Parsed task records
        may be embedded under ``tasks``/``task_records`` or supplied separately.
        Alternate argument spellings are accepted only to ease integration with
        the existing board and merge-queue adapters.
        """

        now_ms = _clock_milliseconds(self.clock)
        if validated_board is not None and board_validation is not None:
            return self._global_gap(
                TaskEvidenceGapKind.BOARD_MALFORMED,
                "conflicting board validation inputs",
                now_ms,
            )
        board = _record(
            validated_board if validated_board is not None else board_validation
        )
        if not board:
            return self._global_gap(
                TaskEvidenceGapKind.BOARD_MALFORMED,
                "validated board receipt is missing or malformed",
                now_ms,
            )
        board_cid = content_identity(dict(board))
        if board.get("valid") is not True:
            return self._global_gap(
                TaskEvidenceGapKind.BOARD_UNVALIDATED,
                "board validator did not return valid=true",
                now_ms,
                board_cid,
            )
        declared_namespace = _text(
            _value(board, "board_namespace", "namespace")
        )
        if declared_namespace and declared_namespace != self.board_namespace:
            return self._global_gap(
                TaskEvidenceGapKind.BOARD_MALFORMED,
                "validated board namespace does not match collector namespace",
                now_ms,
                board_cid,
            )

        supplied_tasks = tuple(task_records) or tuple(tasks)
        if not supplied_tasks:
            embedded = _value(board, "task_records", "tasks")
            if isinstance(embedded, Iterable) and not isinstance(
                embedded, (str, bytes, Mapping)
            ):
                supplied_tasks = tuple(embedded)
        parsed, task_gaps = self._parse_tasks(supplied_tasks)
        expected_count = _integer(board, "task_count")
        declared_task_ids = _value(board, "task_ids", "required_task_ids")
        if declared_task_ids is not None:
            if (
                not isinstance(declared_task_ids, Iterable)
                or isinstance(declared_task_ids, (str, bytes, Mapping))
                or {_text(item) for item in declared_task_ids} != set(parsed)
            ):
                task_gaps.append(
                    TaskEvidenceGap(
                        task_id="*",
                        kind=TaskEvidenceGapKind.BOARD_POPULATION_MISMATCH,
                        detail="validated board task IDs do not match canonical task records",
                        input_cid=board_cid,
                    )
                )
        declared_task_cids = _value(board, "task_cids", "canonical_task_cids")
        if declared_task_cids is not None:
            if not isinstance(declared_task_cids, Mapping) or any(
                _text(declared_task_cids.get(task_id)) != task.task_cid
                for task_id, task in parsed.items()
            ) or set(map(_text, declared_task_cids)) != set(parsed):
                task_gaps.append(
                    TaskEvidenceGap(
                        task_id="*",
                        kind=TaskEvidenceGapKind.TASK_CID_MISMATCH,
                        detail="validated board task CIDs do not match canonical task records",
                        input_cid=board_cid,
                    )
                )
        if (
            expected_count is None
            or expected_count <= 0
            or expected_count != len(supplied_tasks)
            or len(parsed) != len(supplied_tasks)
        ):
            task_gaps.append(
                TaskEvidenceGap(
                    task_id="*",
                    kind=TaskEvidenceGapKind.BOARD_POPULATION_MISMATCH,
                    detail="validated board task_count does not match canonical task records",
                    input_cid=board_cid,
                )
            )
        if task_gaps:
            return ProofTestReuseTaskEvidenceCollection(
                board_cid=board_cid,
                required_task_ids=tuple(sorted(parsed)),
                evidence=(),
                gaps=tuple(task_gaps),
                evaluated_at_ms=now_ms,
            )

        # Project daemon merge rows into collector-safe sealed receipts before
        # indexing.  Do not collapse duplicates here — ``_index`` still detects
        # contradictory multi-row claims per task.
        projected_merges: list[dict[str, Any]] = []
        for raw in (*tuple(merge_queue_records), *tuple(merge_records)):
            projected = project_managed_merge_queue_record(raw)
            if projected is not None:
                projected_merges.append(projected)
            else:
                # Keep unprojectable successful-looking rows only as empty
                # mappings so they cannot crash content_identity later; the
                # queue path re-projects and gaps malformed claims.
                record = _record(raw)
                if record:
                    projected_merges.append(dict(record))
        queues = self._index(tuple(projected_merges))
        validations = self._index(validation_receipts)
        approvals = self._index(approval_records)
        retrospectives = self._index((*tuple(retrospective_records), *tuple(history_records)))
        required = set(parsed)
        gaps: list[TaskEvidenceGap] = []
        evidence: list[ProofTestReuseTaskEvidence] = []

        for label, index in (
            ("merge", queues),
            ("validation", validations),
            ("approval", approvals),
            ("retrospective", retrospectives),
        ):
            for task_id in sorted(set(index).difference(required)):
                gaps.append(
                    TaskEvidenceGap(
                        task_id=task_id or "*",
                        kind=TaskEvidenceGapKind.UNEXPECTED_INPUT,
                        detail=f"{label} input names a task outside the validated board",
                    )
                )

        for task_id in sorted(required):
            task = parsed[task_id]
            duplicate_sources = [
                label
                for label, index in (
                    ("merge", queues),
                    ("validation", validations),
                    ("approval", approvals),
                    ("retrospective", retrospectives),
                )
                if len(index.get(task_id, ())) > 1
            ]
            if duplicate_sources:
                gaps.append(
                    TaskEvidenceGap(
                        task_id=task_id,
                        kind=TaskEvidenceGapKind.COMPLETION_PROVENANCE_CONTRADICTORY,
                        detail="duplicate " + ", ".join(duplicate_sources) + " inputs",
                    )
                )
                continue

            validation, validation_gap = self._validation(
                task, self._one(validations, task_id), now_ms
            )
            if validation_gap is not None:
                gaps.append(validation_gap)
                continue
            assert validation is not None

            provenance, provenance_gap = self._completion_provenance(
                task=task,
                queue=self._one(queues, task_id),
                approval=self._one(approvals, task_id),
                retrospective=self._one(retrospectives, task_id),
                validation=validation,
            )
            if provenance_gap is not None:
                gaps.append(provenance_gap)
                continue
            assert provenance is not None
            evidence.append(
                ProofTestReuseTaskEvidence(
                    task_id=task.task_id,
                    goal_id=task.goal_id,
                    task_cid=task.task_cid,
                    board_cid=board_cid,
                    repository_id=self.repository_id,
                    repository_state_cid=self.repository_state_cid,
                    git_commit_id=self.git_commit_id,
                    git_tree_id=self.git_tree_id,
                    gitlink_state_cid=self.gitlink_state_cid,
                    repository_forest_cid=self.repository_forest_cid,
                    dirty=self.dirty,
                    dirty_overlay_cid=self.dirty_overlay_cid,
                    policy_cid=self.policy_cid,
                    capability_cid=self.capability_cid,
                    verifying_key_cid=self.verifying_key_cid,
                    circuit_cid=self.circuit_cid,
                    task_provenance=provenance,
                    validation=validation,
                )
            )

        return ProofTestReuseTaskEvidenceCollection(
            board_cid=board_cid,
            required_task_ids=tuple(sorted(required)),
            evidence=tuple(evidence),
            gaps=tuple(gaps),
            evaluated_at_ms=now_ms,
        )

    def _global_gap(
        self,
        kind: TaskEvidenceGapKind,
        detail: str,
        now_ms: int,
        board_cid: str = "",
    ) -> ProofTestReuseTaskEvidenceCollection:
        return ProofTestReuseTaskEvidenceCollection(
            board_cid=board_cid,
            required_task_ids=(),
            evidence=(),
            gaps=(TaskEvidenceGap("*", kind, detail, board_cid),),
            evaluated_at_ms=now_ms,
        )

    def _parse_tasks(
        self, values: tuple[Any, ...]
    ) -> tuple[dict[str, _Task], list[TaskEvidenceGap]]:
        tasks: dict[str, _Task] = {}
        gaps: list[TaskEvidenceGap] = []
        for value in values:
            record = _record(value)
            task_id = _text(_value(record, "task_id", "id"))
            if not task_id:
                gaps.append(
                    TaskEvidenceGap(
                        "*", TaskEvidenceGapKind.TASK_MALFORMED, "task_id is missing"
                    )
                )
                continue
            if task_id in tasks:
                gaps.append(
                    TaskEvidenceGap(
                        task_id,
                        TaskEvidenceGapKind.TASK_DUPLICATE,
                        "validated board contains duplicate task records",
                    )
                )
                continue
            namespace = _text(_value(record, "board_namespace", "namespace"))
            task_cid = _text(
                _value(record, "canonical_task_cid", "task_cid", "canonical_task_id")
            )
            metadata = _record(record.get("metadata"))
            goal_id = _text(
                _value(record, "goal_id")
                or _value(metadata, "goal id", "goal_id")
                or task_id
            )
            commands = _value(record, "validation", "validation_commands")
            if isinstance(commands, str):
                commands = (commands,)
            if not isinstance(commands, Iterable) or isinstance(commands, Mapping):
                commands = ()
            normalized_commands = tuple(_text(item) for item in commands if _text(item))
            if namespace != self.board_namespace:
                gaps.append(
                    TaskEvidenceGap(
                        task_id,
                        TaskEvidenceGapKind.TASK_MALFORMED,
                        "task is not bound to the validated board namespace",
                    )
                )
            elif not task_cid:
                gaps.append(
                    TaskEvidenceGap(
                        task_id,
                        TaskEvidenceGapKind.TASK_CID_MISSING,
                        "canonical task CID is missing",
                    )
                )
            elif len(normalized_commands) != 1:
                gaps.append(
                    TaskEvidenceGap(
                        task_id,
                        TaskEvidenceGapKind.TASK_MALFORMED,
                        "task must declare exactly one validation command",
                    )
                )
            elif not normalized_commands[0].startswith(_PROOF_REUSE_OFF_PREFIX):
                gaps.append(
                    TaskEvidenceGap(
                        task_id,
                        TaskEvidenceGapKind.PROOF_REUSE_NOT_OFF,
                        "declared validation command does not force proof reuse off",
                    )
                )
            else:
                tasks[task_id] = _Task(
                    task_id=task_id,
                    goal_id=goal_id,
                    task_cid=task_cid,
                    validation_command=normalized_commands[0],
                )
        return tasks, gaps

    @staticmethod
    def _index(values: Iterable[Any]) -> dict[str, tuple[Mapping[str, Any], ...]]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for value in values:
            record = _record(value)
            grouped[_text(_value(record, "task_id", "canonical_task_id"))].append(
                record
            )
        return {key: tuple(items) for key, items in grouped.items()}

    @staticmethod
    def _one(
        index: Mapping[str, tuple[Mapping[str, Any], ...]], task_id: str
    ) -> Mapping[str, Any]:
        values = index.get(task_id, ())
        return values[0] if len(values) == 1 else {}

    def _validation(
        self, task: _Task, raw: Mapping[str, Any], now_ms: int
    ) -> tuple[TaskValidationProvenance | None, TaskEvidenceGap | None]:
        if not raw:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_MISSING,
                "no validation receipt was retained",
            )
        if raw.get("schema") == ProofCachedTestValidationReceipt.SCHEMA:
            return self._proof_skip_validation(task, raw, now_ms)
        return self._executed_validation(task, raw, now_ms)

    def _is_fresh(self, observed_at_ms: int, fresh_until_ms: int, now_ms: int) -> bool:
        max_age_ms = int(float(self.freshness_seconds) * 1_000)
        return (
            observed_at_ms <= now_ms <= fresh_until_ms
            and now_ms - observed_at_ms <= max_age_ms
        )

    def _binding_gap(
        self, task: _Task, raw: Mapping[str, Any]
    ) -> TaskEvidenceGap | None:
        expected = {
            "task_id": task.task_id,
            "goal_id": task.goal_id,
            "task_cid": task.task_cid,
            "repository_id": self.repository_id,
            "repository_state_cid": self.repository_state_cid,
            "git_commit_id": self.git_commit_id,
            "git_tree_id": self.git_tree_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "dirty_overlay_cid": self.dirty_overlay_cid,
        }
        for name, wanted in expected.items():
            actual = _text(_value(raw, name, "commit_id" if name == "git_commit_id" else ""))
            if actual != wanted:
                return TaskEvidenceGap(
                    task.task_id,
                    TaskEvidenceGapKind.VALIDATION_BINDING_MISMATCH,
                    f"validation {name} does not match the current task/tree binding",
                )
        if _boolean(raw, "dirty") is not self.dirty:
            return TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_BINDING_MISMATCH,
                "validation dirty flag does not match the current overlay",
            )
        return None

    def _proof_skip_validation(
        self, task: _Task, raw: Mapping[str, Any], now_ms: int
    ) -> tuple[TaskValidationProvenance | None, TaskEvidenceGap | None]:
        try:
            receipt = ProofCachedTestValidationReceipt.from_dict(raw)
        except (ProofCachedTestValidationError, TypeError, ValueError):
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_MALFORMED,
                "proof-backed skip receipt is malformed or has an invalid CID",
            )
        if receipt.task_id != task.task_id or receipt.goal_id != task.goal_id:
            return None, self._gap_binding(task, "proof receipt task/goal")
        if receipt.validation_command != task.validation_command:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_COMMAND_MISMATCH,
                "proof receipt command differs from the board command",
            )
        binding_gap = self._binding_gap(task, {**raw, "task_cid": task.task_cid})
        if binding_gap:
            return None, binding_gap
        if not receipt.is_completion_evidence(
            now_ms=now_ms,
            task_id=task.task_id,
            goal_id=task.goal_id,
            goal_revision=self.objective_revision,
            validation_command=task.validation_command,
            repository_state_cid=self.repository_state_cid,
        ):
            kind = (
                TaskEvidenceGapKind.VALIDATION_STALE
                if not receipt.is_fresh(now_ms=now_ms)
                else TaskEvidenceGapKind.PROOF_SKIP_UNVERIFIED
            )
            return None, TaskEvidenceGap(
                task.task_id, kind, "proof-backed skip receipt is not current authority"
            )
        if not self._is_fresh(
            receipt.verified_at_ms, receipt.fresh_until_ms, now_ms
        ):
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_STALE,
                "proof-backed skip exceeds the collector freshness policy",
            )
        if self.proof_skip_verifier is None:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.PROOF_SKIP_VERIFIER_UNAVAILABLE,
                "no local proof verifier was supplied",
            )
        try:
            verified = self.proof_skip_verifier(receipt)
        except Exception:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.PROOF_SKIP_VERIFIER_UNAVAILABLE,
                "local proof verifier was unavailable",
            )
        if verified is not True:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.PROOF_SKIP_UNVERIFIED,
                "local proof verification rejected the skip receipt",
            )
        return (
            TaskValidationProvenance(
                task_id=task.task_id,
                goal_id=task.goal_id,
                task_cid=task.task_cid,
                validation_command=task.validation_command,
                validation_receipt_cid=receipt.validation_receipt_cid,
                disposition="proof_backed_skip",
                repository_id=self.repository_id,
                repository_state_cid=self.repository_state_cid,
                git_commit_id=self.git_commit_id,
                git_tree_id=self.git_tree_id,
                gitlink_state_cid=self.gitlink_state_cid,
                repository_forest_cid=self.repository_forest_cid,
                dirty=self.dirty,
                dirty_overlay_cid=self.dirty_overlay_cid,
                observed_at_ms=receipt.verified_at_ms,
                fresh_until_ms=receipt.fresh_until_ms,
                locally_verified=True,
                receipt=receipt.to_record(),
            ),
            None,
        )

    def _executed_validation(
        self, task: _Task, raw: Mapping[str, Any], now_ms: int
    ) -> tuple[TaskValidationProvenance | None, TaskEvidenceGap | None]:
        skipped_count = _value(raw, "skipped_count")
        if skipped_count is not None and _integer(raw, "skipped_count") is None:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_MALFORMED,
                "validation skipped_count must be an integer",
            )
        if _text(_value(raw, "disposition", "outcome_kind")) in {
            "skip",
            "skipped",
            "ordinary_skip",
        } or (_integer(raw, "skipped_count") or 0) > 0 or _boolean(
            raw, "skipped"
        ) is True:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.ORDINARY_SKIP,
                "ordinary pytest skips are not validation authority",
            )
        if (
            _text(_value(raw, "status", "result")) not in {"passed", "succeeded"}
            or _boolean(raw, "passed") is not True
            or _integer(raw, "exit_code", "returncode") != 0
        ):
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_FAILED,
                "validation receipt does not prove a successful execution",
            )
        mode = _text(_value(raw, "proof_reuse_mode", "test_proof_reuse_mode"))
        command = _text(_value(raw, "validation_command", "command"))
        if mode != "off" or not command.startswith(_PROOF_REUSE_OFF_PREFIX):
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.PROOF_REUSE_NOT_OFF,
                "validation rerun did not force proof reuse off",
            )
        if command != task.validation_command:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_COMMAND_MISMATCH,
                "validation command differs from the board command",
            )
        command_cid = _text(
            _value(raw, "validation_command_cid", "command_cid")
        )
        if command_cid != validation_command_identity(command):
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_MALFORMED,
                "validation command CID is missing or invalid",
            )
        binding_gap = self._binding_gap(task, raw)
        if binding_gap:
            return None, binding_gap
        observed = _integer(raw, "observed_at_ms", "finished_at_ms")
        fresh_until = _integer(raw, "fresh_until_ms")
        if (
            observed is None
            or fresh_until is None
            or fresh_until <= observed
            or not self._is_fresh(observed, fresh_until, now_ms)
        ):
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_STALE,
                "validation receipt is missing a current freshness window",
            )
        receipt_cid = _immutable_record_cid(
            raw,
            "validation_receipt_cid",
            "receipt_id",
            "content_id",
            claim_required=True,
        )
        if not receipt_cid:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.VALIDATION_MALFORMED,
                "validation receipt has no valid immutable identity",
            )
        return (
            TaskValidationProvenance(
                task_id=task.task_id,
                goal_id=task.goal_id,
                task_cid=task.task_cid,
                validation_command=command,
                validation_receipt_cid=receipt_cid,
                disposition="executed",
                repository_id=self.repository_id,
                repository_state_cid=self.repository_state_cid,
                git_commit_id=self.git_commit_id,
                git_tree_id=self.git_tree_id,
                gitlink_state_cid=self.gitlink_state_cid,
                repository_forest_cid=self.repository_forest_cid,
                dirty=self.dirty,
                dirty_overlay_cid=self.dirty_overlay_cid,
                observed_at_ms=observed,
                fresh_until_ms=fresh_until,
                locally_verified=True,
                receipt=dict(raw),
            ),
            None,
        )

    def _completion_provenance(
        self,
        *,
        task: _Task,
        queue: Mapping[str, Any],
        approval: Mapping[str, Any],
        retrospective: Mapping[str, Any],
        validation: TaskValidationProvenance,
    ) -> tuple[dict[str, Any] | None, TaskEvidenceGap | None]:
        if queue and (approval or retrospective):
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.COMPLETION_PROVENANCE_CONTRADICTORY,
                "queue and non-queue completion claims conflict",
            )
        if queue:
            return self._queue_provenance(task, queue)
        if task.task_id in REVIEW_REQUIRED_WITHOUT_QUEUE and not approval:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.APPROVAL_MISSING,
                "this historic task requires genuine retained approval evidence",
            )
        if retrospective:
            if not approval:
                return None, TaskEvidenceGap(
                    task.task_id,
                    TaskEvidenceGapKind.APPROVAL_MISSING,
                    "retrospective provenance requires immutable reviewed approval",
                )
            if validation.disposition != "executed":
                return None, TaskEvidenceGap(
                    task.task_id,
                    TaskEvidenceGapKind.ORDINARY_SKIP,
                    "retrospective provenance requires a current proof-reuse-off rerun",
                )
            return self._retrospective_provenance(
                task, retrospective, approval, validation
            )
        if approval:
            return self._approval_provenance(task, approval)
        return None, TaskEvidenceGap(
            task.task_id,
            TaskEvidenceGapKind.COMPLETION_PROVENANCE_MISSING,
            "no successful merge or reviewed completion provenance was retained",
        )

    def _verified_ancestry(
        self, task_id: str, ancestor: str
    ) -> TaskEvidenceGap | None:
        if self.ancestry_verifier is None:
            return TaskEvidenceGap(
                task_id,
                TaskEvidenceGapKind.ANCESTRY_UNAVAILABLE,
                "no Git ancestry verifier was supplied",
            )
        try:
            verified = self.ancestry_verifier(ancestor, self.git_commit_id)
        except Exception:
            return TaskEvidenceGap(
                task_id,
                TaskEvidenceGapKind.ANCESTRY_UNAVAILABLE,
                "Git ancestry verification was unavailable",
            )
        if verified is not True:
            return TaskEvidenceGap(
                task_id,
                TaskEvidenceGapKind.ANCESTRY_UNVERIFIED,
                "completion commit is not a verified ancestor of the current commit",
            )
        return None

    def _queue_provenance(
        self, task: _Task, queue: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, TaskEvidenceGap | None]:
        # Re-project so callers that bypass collect()'s adapter still get a
        # sealed integer/string body instead of a content_identity exception.
        projected = project_managed_merge_queue_record(queue) or dict(queue)
        if _text(_value(projected, "status", "state")) not in {"completed", "merged"}:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.QUEUE_RECORD_UNSUCCESSFUL,
                "merge queue record is not successfully completed",
            )
        claimed_task_cid = _text(
            _value(projected, "task_cid", "canonical_task_cid", "canonical_task_id")
        )
        if claimed_task_cid != task.task_cid:
            return None, self._gap_task_cid(task, "merge queue")
        commit = _text(
            _value(projected, "merged_commit_id", "commit_sha", "commit_id")
        )
        if not commit:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.COMPLETION_PROVENANCE_MALFORMED,
                "merge queue record has no merged commit",
            )
        ancestry_gap = self._verified_ancestry(task.task_id, commit)
        if ancestry_gap:
            return None, ancestry_gap
        receipt_cid = _immutable_record_cid(
            projected,
            "merge_receipt_cid",
            "receipt_id",
            "content_id",
            claim_required=False,
        )
        if not receipt_cid:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.COMPLETION_PROVENANCE_MALFORMED,
                "merge queue record carries a contradictory immutable identity",
            )
        return {
            "kind": TaskCompletionProvenanceKind.MANAGED_MERGE.value,
            "merge_receipt_cid": receipt_cid,
            "merged_commit_id": commit,
            "merge_succeeded": True,
        }, None

    def _approval_cid(
        self, task: _Task, approval: Mapping[str, Any]
    ) -> tuple[str, TaskEvidenceGap | None]:
        if (
            _text(_value(approval, "task_id")) != task.task_id
            or _text(_value(approval, "task_cid", "canonical_task_cid"))
            != task.task_cid
            or _boolean(approval, "approved", "reviewed") is not True
            or _text(_value(approval, "reviewer_id", "operator_id")) == ""
        ):
            return "", TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.APPROVAL_MALFORMED,
                "approval is not an explicit reviewed decision for the canonical task",
            )
        cid = _immutable_record_cid(
            approval,
            "approval_cid",
            "operator_approval_cid",
            "operator_review_cid",
            "policy_approval_cid",
            "content_id",
            claim_required=True,
        )
        if not cid:
            return "", TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.APPROVAL_MALFORMED,
                "approval has no valid immutable identity",
            )
        if self.approval_verifier is None:
            return "", TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.APPROVAL_UNVERIFIED,
                "no trusted approval verifier was supplied",
            )
        try:
            approved = self.approval_verifier(approval)
        except Exception:
            approved = False
        if approved is not True:
            return "", TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.APPROVAL_UNVERIFIED,
                "reviewed approval failed local verification",
            )
        return cid, None

    def _approval_provenance(
        self, task: _Task, approval: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, TaskEvidenceGap | None]:
        cid, gap = self._approval_cid(task, approval)
        if gap:
            return None, gap
        kind = _text(_value(approval, "kind", "approval_kind"))
        if task.task_id == "PTR-000":
            if kind not in {"planning_seal", "operator_planning_seal"}:
                return None, TaskEvidenceGap(
                    task.task_id,
                    TaskEvidenceGapKind.APPROVAL_MALFORMED,
                    "PTR-000 requires an operator planning seal",
                )
            revision = _text(
                _value(approval, "sealed_objective_revision", "objective_revision")
            )
            seal_cid = _text(_value(approval, "planning_seal_cid"))
            if not revision or not seal_cid:
                return None, TaskEvidenceGap(
                    task.task_id,
                    TaskEvidenceGapKind.APPROVAL_MALFORMED,
                    "planning seal is missing its revision or seal CID",
                )
            return {
                "kind": TaskCompletionProvenanceKind.OPERATOR_PLANNING_SEAL.value,
                "planning_seal_cid": seal_cid,
                "operator_approval_cid": cid,
                "sealed_objective_revision": revision,
                "planning_seal_accepted": True,
            }, None
        if kind not in {"reviewed_integration", "operator_reviewed_integration"}:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.APPROVAL_MALFORMED,
                "task requires reviewed integration approval",
            )
        commit = _text(
            _value(approval, "integrated_commit_id", "commit_id", "commit_sha")
        )
        target = _text(
            _value(approval, "integration_target_commit_id", "target_commit_id")
        )
        receipt_cid = _text(_value(approval, "integration_receipt_cid"))
        if not commit or not target or target != self.git_commit_id or not receipt_cid:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.APPROVAL_MALFORMED,
                "reviewed integration is not bound to the current commit",
            )
        ancestry_gap = self._verified_ancestry(task.task_id, commit)
        if ancestry_gap:
            return None, ancestry_gap
        return {
            "kind": TaskCompletionProvenanceKind.OPERATOR_REVIEWED_INTEGRATION.value,
            "integration_receipt_cid": receipt_cid,
            "integrated_commit_id": commit,
            "integration_target_commit_id": target,
            "operator_review_cid": cid,
            "integration_verified": True,
        }, None

    def _retrospective_provenance(
        self,
        task: _Task,
        retrospective: Mapping[str, Any],
        approval: Mapping[str, Any],
        validation: TaskValidationProvenance,
    ) -> tuple[dict[str, Any] | None, TaskEvidenceGap | None]:
        if (
            _text(_value(retrospective, "task_id")) != task.task_id
            or _text(
                _value(retrospective, "task_cid", "canonical_task_cid")
            )
            != task.task_cid
        ):
            return None, self._gap_task_cid(task, "retrospective history")
        commit = _text(
            _value(retrospective, "integrated_commit_id", "commit_id", "commit_sha")
        )
        if not commit:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.COMPLETION_PROVENANCE_MALFORMED,
                "retrospective history has no integrated commit",
            )
        ancestry_gap = self._verified_ancestry(task.task_id, commit)
        if ancestry_gap:
            return None, ancestry_gap
        ancestry_cid = _immutable_record_cid(
            retrospective,
            "ancestry_receipt_cid",
            "receipt_id",
            "content_id",
            claim_required=False,
        )
        if not ancestry_cid:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.COMPLETION_PROVENANCE_MALFORMED,
                "retrospective history carries a contradictory identity",
            )
        approval_cid, approval_gap = self._approval_cid(task, approval)
        if approval_gap:
            return None, approval_gap
        if _text(_value(approval, "kind", "approval_kind")) not in {
            "retrospective_review",
            "reviewed_retrospective",
        }:
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.APPROVAL_MALFORMED,
                "retrospective history requires retrospective review approval",
            )
        if (
            _text(_value(approval, "integrated_commit_id")) != commit
            or _text(_value(approval, "approved_policy_cid", "policy_cid"))
            != self.policy_cid
        ):
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.APPROVAL_MALFORMED,
                "retrospective approval is not bound to its commit and current policy",
            )
        if not all(
            (
                self.policy_cid,
                self.capability_cid,
                self.verifying_key_cid,
                self.circuit_cid,
            )
        ):
            return None, TaskEvidenceGap(
                task.task_id,
                TaskEvidenceGapKind.COMPLETION_PROVENANCE_MALFORMED,
                "retrospective authority requires current policy and proof bindings",
            )
        return {
            "kind": (
                TaskCompletionProvenanceKind.RETROSPECTIVE_INTEGRATION_VERIFICATION.value
            ),
            "integrated_commit_id": commit,
            "ancestry_target_commit_id": self.git_commit_id,
            "ancestry_receipt_cid": ancestry_cid,
            "ancestry_verified": True,
            "current_tree_rerun_receipt_cid": validation.validation_receipt_cid,
            "current_tree_rerun_repository_id": self.repository_id,
            "current_tree_rerun_tree_id": self.git_tree_id,
            "current_tree_rerun_commit_id": self.git_commit_id,
            "current_tree_rerun_gitlink_state_cid": self.gitlink_state_cid,
            "current_tree_rerun_repository_forest_cid": self.repository_forest_cid,
            "current_tree_rerun_policy_cid": self.policy_cid,
            "current_tree_rerun_capability_cid": self.capability_cid,
            "current_tree_rerun_verifying_key_cid": self.verifying_key_cid,
            "current_tree_rerun_circuit_cid": self.circuit_cid,
            "current_tree_rerun_passed": True,
            "policy_approval_cid": approval_cid,
            "approved_policy_cid": self.policy_cid,
            "policy_approved": True,
        }, None

    @staticmethod
    def _gap_task_cid(task: _Task, source: str) -> TaskEvidenceGap:
        return TaskEvidenceGap(
            task.task_id,
            TaskEvidenceGapKind.TASK_CID_MISMATCH,
            f"{source} does not name the canonical task CID",
        )

    @staticmethod
    def _gap_binding(task: _Task, source: str) -> TaskEvidenceGap:
        return TaskEvidenceGap(
            task.task_id,
            TaskEvidenceGapKind.VALIDATION_BINDING_MISMATCH,
            f"{source} binding does not match the validated task",
        )


__all__ = [
    "PROOF_TEST_REUSE_TASK_EVIDENCE_INTERFACE",
    "TASK_VALIDATION_PROVENANCE_INTERFACE",
    "TASK_EVIDENCE_GAP_INTERFACE",
    "REVIEW_REQUIRED_WITHOUT_QUEUE",
    "DEFAULT_EVIDENCE_FRESHNESS_SECONDS",
    "TaskCompletionProvenanceKind",
    "TaskEvidenceGapKind",
    "TaskEvidenceGap",
    "TaskValidationProvenance",
    "ProofTestReuseTaskEvidence",
    "ProofTestReuseTaskEvidenceCollection",
    "ProofTestReuseTaskEvidenceCollector",
    "ProofTestReuseTaskEvidenceError",
    "project_managed_merge_queue_record",
    "project_managed_merge_queue_records",
]
