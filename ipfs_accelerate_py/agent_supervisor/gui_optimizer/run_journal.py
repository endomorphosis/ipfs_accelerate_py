"""Interruption-safe improvement journals for VerifiedGuiOptimizer (VGO-054).

Interfaces owned by this module:

* ``GuiRunJournal@1`` — append-only, content-addressed phase journal
* ``GuiRunCheckpoint@1`` — immutable heartbeat/progress/worktree snapshot
* ``GuiResumeDecision@1`` — resume, restart, reject, or return a completed
  terminal receipt identity

The journal never infers completion from process exit and never reuses
stale or foreign worktree state.  Interrupted runs resume only after Git
identities revalidate.  Corrupt, truncated, or mismatched state fails
closed.  Identical completed runs return the same terminal receipt
identity without mutating the canonical branch.

Fail-closed invariants:

* phase records and artifact manifests append atomically;
* the same effect_id cannot apply twice;
* a missing process is interruption, never completion;
* revision, worktree, and canonical-snapshot mismatches reject;
* browser-selected paths cannot locate journal or artifact state.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_datasets_py.logic.gui_optimizer.identity import (
    GuiCanonicalIdentity,
    canonical_identity,
)
from ipfs_datasets_py.logic.gui_optimizer.receipts import (
    GUI_VERIFICATION_RECEIPT_ENVELOPE_INTERFACE,
    envelope_identity,
    improvement_receipt_identity,
)
from ipfs_datasets_py.logic.gui_optimizer.schema import (
    GUI_IMPROVEMENT_RECEIPT_INTERFACE,
)

from .artifact_store import (
    ArtifactKind,
    ArtifactReuseGate,
    GuiArtifactStoreError,
    GuiEvidenceArtifactStore,
    StoredArtifact,
    atomic_write_bytes,
    canonical_json_bytes,
    default_evidence_artifact_store,
    resolve_host_root,
)
from .authority import AuthorityReasonCode, GuiAuthorityError

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

GUI_RUN_JOURNAL_INTERFACE: Final[str] = "GuiRunJournal@1"
GUI_RUN_CHECKPOINT_INTERFACE: Final[str] = "GuiRunCheckpoint@1"
GUI_RESUME_DECISION_INTERFACE: Final[str] = "GuiResumeDecision@1"

GUI_RUN_JOURNAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/run-journal@1"
)
GUI_RUN_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/run-checkpoint@1"
)
GUI_RESUME_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/resume-decision@1"
)
GUI_PHASE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/phase-record@1"
)
GUI_TERMINAL_RECEIPT_REF_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/terminal-receipt-ref@1"
)

DOMAIN_RUN_IDENTITY: Final[str] = "gui.run-identity"
DOMAIN_RUN_CHECKPOINT: Final[str] = "gui.run-checkpoint"
DOMAIN_PHASE_RECORD: Final[str] = "gui.run-phase-record"
DOMAIN_TERMINAL_RECEIPT: Final[str] = "gui.journal-terminal-receipt"

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CID_RE = re.compile(r"^b[a-z2-7]{50,80}$")

_OPEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "application_id",
        "attempt",
        "canonical_branch",
        "canonical_porcelain",
        "canonical_revision",
        "objective_id",
        "proposal_id",
        "run_id",
        "screen_id",
        "source_revision",
        "worktree_lease_id",
        "worktree_path",
        "worktree_revision",
    }
)
_APPEND_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact_cids",
        "effect_id",
        "payload",
        "phase",
        "receipt_cid",
        "run_id",
        "status",
    }
)
_RESUME_KEYS: Final[frozenset[str]] = frozenset(
    {
        "canonical_branch",
        "canonical_porcelain",
        "canonical_revision",
        "process_alive",
        "run_id",
        "source_revision",
        "worktree_lease_id",
        "worktree_path",
        "worktree_revision",
    }
)
_FORBIDDEN_PATH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "browser_input",
        "command",
        "commands",
        "cwd",
        "file_path",
        "filesystem_path",
        "host_path",
        "path",
        "selected_host_paths",
        "working_directory",
    }
)
class GuiRunJournalError(GuiAuthorityError):
    """Malformed or unsafe journal state.  Never grants resume."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_run_journal_input",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


class JournalPhase(str, Enum):
    """Closed improvement-loop phases persisted by ``GuiRunJournal@1``."""

    BASELINE = "baseline"
    SELECT_OBJECTIVE = "select_objective"
    IMPACT = "impact"
    CONTEXT_PACK = "context_pack"
    PROPOSAL = "proposal"
    ISOLATED_WORKTREE = "isolated_worktree"
    RESCAN = "rescan"
    INVALIDATION = "invalidation"
    AFFECTED_CHECKS = "affected_checks"
    FALLBACK = "fallback"
    COMPARE = "compare"
    DECISION = "decision"
    RECEIPT = "receipt"


class PhaseRecordStatus(str, Enum):
    """Closed per-phase outcomes."""

    STARTED = "started"
    COMPLETED = "completed"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"


class RunStatus(str, Enum):
    """Closed journal-level status.  Process exit is never completion."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class ResumeAction(str, Enum):
    """Closed ``GuiResumeDecision@1`` actions."""

    RESUME = "resume"
    RESTART = "restart"
    REJECT = "reject"
    RETURN_COMPLETED = "return_completed"


class JournalReasonCode(str, Enum):
    """Stable reason codes for journal and resume decisions."""

    OPENED = "opened"
    APPENDED = "appended"
    IDEMPOTENT_EFFECT = "idempotent_effect"
    HEARTBEAT = "heartbeat"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    RESUME = "resume"
    RESTART = "restart"
    RETURN_COMPLETED = "return_completed"
    CORRUPT_JOURNAL = "corrupt_journal"
    TRUNCATED_JOURNAL = "truncated_journal"
    REVISION_MISMATCH = "revision_mismatch"
    STALE_WORKTREE = "stale_worktree"
    FOREIGN_WORKTREE = "foreign_worktree"
    CANONICAL_MUTATION_DETECTED = "canonical_mutation_detected"
    PROCESS_EXIT_NOT_COMPLETION = "process_exit_not_completion"
    DUPLICATE_EFFECT = "duplicate_effect"
    COMPLETED_RECEIPT_MISMATCH = "completed_receipt_mismatch"
    MISSING_RUN = "missing_run"
    UNKNOWN_FIELD = AuthorityReasonCode.UNKNOWN_FIELD.value
    INVALID_COLLECTION_TYPE = (
        AuthorityReasonCode.INVALID_COLLECTION_TYPE.value
    )
    INVALID_RUN_JOURNAL_INPUT = "invalid_run_journal_input"
    BROWSER_PATH_FORBIDDEN = (
        AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value
    )
    PATH_ABSOLUTE_OR_TRAVERSAL = (
        AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    )


# ---------------------------------------------------------------------------
# Closed input helpers
# ---------------------------------------------------------------------------


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise GuiRunJournalError(
            f"{name} must be a string",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiRunJournalError(
            f"{name} must not contain NUL",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name},
        )
    text = text_value.strip()
    if required and not text:
        raise GuiRunJournalError(
            f"{name} must not be empty",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name},
        )
    return text


def _identifier(value: Any, name: str) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiRunJournalError(
            f"{name} must not contain NUL",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name},
        )
    if text_value == "" or text_value != text_value.strip():
        raise GuiRunJournalError(
            f"{name} must be a canonical nonempty string identifier",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name},
        )
    return text_value


def _run_id(value: Any) -> str:
    text = _identifier(value, "run_id")
    if not _SAFE_RUN_ID_RE.fullmatch(text) or "/" in text or "\\" in text:
        raise GuiRunJournalError(
            "run_id must be a host-safe identifier",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": "run_id"},
        )
    return text


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise GuiRunJournalError(
            f"{name} must be an integer",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    if value < 1:
        raise GuiRunJournalError(
            f"{name} must be a positive integer",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name, "value": value},
        )
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise GuiRunJournalError(
            f"{name} must be an integer",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    if value < 0:
        raise GuiRunJournalError(
            f"{name} must be a non-negative integer",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name, "value": value},
        )
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GuiRunJournalError(
            f"{name} must be a JSON object",
            reason_code=JournalReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    for key in value:
        if type(key) is not str:
            raise GuiRunJournalError(
                f"{name} keys must be strings",
                reason_code=JournalReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name, "key_type": type(key).__name__},
            )
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise GuiRunJournalError(
            f"{name} must be a JSON array",
            reason_code=JournalReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], noun: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise GuiRunJournalError(
            f"{noun} contains unknown fields: {unknown}",
            reason_code=JournalReasonCode.UNKNOWN_FIELD.value,
            details={"noun": noun, "unknown_fields": unknown},
        )


def _reject_forbidden_path_keys(payload: Mapping[str, Any], noun: str) -> None:
    forbidden = sorted(set(payload) & set(_FORBIDDEN_PATH_KEYS))
    if forbidden:
        raise GuiRunJournalError(
            f"{noun} contains forbidden host-path fields: {forbidden}",
            reason_code=JournalReasonCode.BROWSER_PATH_FORBIDDEN.value,
            details={"noun": noun, "forbidden_fields": forbidden},
        )


def _reject_present_null(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is None:
        raise GuiRunJournalError(
            f"{key} must not be null when present",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": key, "value_type": "NoneType"},
        )


def _optional_identifier(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    text = _exact_str(payload[key], key)
    if text == "":
        return ""
    return _identifier(text, key)


def _optional_text(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _text(payload[key], key, required=False)


def _as_phase(value: Any) -> JournalPhase:
    if type(value) is JournalPhase:
        return value
    text = _text(value, "phase")
    try:
        return JournalPhase(text)
    except ValueError as exc:
        raise GuiRunJournalError(
            f"unknown journal phase: {text}",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"phase": text},
        ) from exc


def _as_phase_status(value: Any) -> PhaseRecordStatus:
    if type(value) is PhaseRecordStatus:
        return value
    text = _text(value, "status")
    try:
        return PhaseRecordStatus(text)
    except ValueError as exc:
        raise GuiRunJournalError(
            f"unknown phase status: {text}",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"status": text},
        ) from exc


def _as_run_status(value: Any) -> RunStatus:
    if type(value) is RunStatus:
        return value
    text = _text(value, "run_status")
    try:
        return RunStatus(text)
    except ValueError as exc:
        raise GuiRunJournalError(
            f"unknown run status: {text}",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"run_status": text},
        ) from exc


def _require_revision(value: Any, name: str) -> str:
    text = _identifier(value, name)
    if not (_FULL_SHA_RE.fullmatch(text) or _DIGEST_RE.fullmatch(text)):
        raise GuiRunJournalError(
            f"{name} must be a 40-character SHA-1 or sha256 digest",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name},
        )
    return text


def _optional_revision(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    text = _exact_str(payload[key], key)
    if text == "":
        return ""
    return _require_revision(text, key)


def _require_cid(value: Any, name: str) -> str:
    text = _identifier(value, name)
    if not _CID_RE.fullmatch(text):
        raise GuiRunJournalError(
            f"{name} must be a CIDv1 raw/sha2-256 base32 string",
            reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            details={"field": name},
        )
    return text


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(_require_mapping(value, "details")))


def journal_binding(*, run_id: str, source_revision: str) -> ArtifactReuseGate:
    """Reuse gate for journal-owned CAS objects."""

    revision = source_revision or ("0" * 40)
    return ArtifactReuseGate(
        repository_id="repository:verified-gui-optimizer",
        repository_revision=revision,
        component_id="comp:gui-run-journal",
        scenario_id=run_id,
        extractor_id=GUI_RUN_JOURNAL_INTERFACE,
        extractor_version=GUI_RUN_JOURNAL_SCHEMA,
        checker_id=GUI_RUN_CHECKPOINT_INTERFACE,
        checker_version=GUI_RUN_CHECKPOINT_SCHEMA,
    )


def run_identity(
    *,
    run_id: str,
    application_id: str,
    screen_id: str,
    objective_id: str,
    source_revision: str,
    proposal_id: str = "",
    attempt: int = 1,
) -> GuiCanonicalIdentity:
    """Stable identity for one improvement run."""

    return canonical_identity(
        {
            "application_id": application_id,
            "attempt": attempt,
            "objective_id": objective_id,
            "proposal_id": proposal_id,
            "run_id": run_id,
            "screen_id": screen_id,
            "source_revision": source_revision,
        },
        domain=DOMAIN_RUN_IDENTITY,
        schema_version=GUI_RUN_JOURNAL_SCHEMA,
    )


def terminal_receipt_identity(
    payload: Mapping[str, Any],
) -> GuiCanonicalIdentity:
    """Identity of a committed terminal receipt.  Identical inputs match."""

    mapping = _require_mapping(payload, "terminal_receipt")
    interface = mapping.get("interface")
    if interface == GUI_IMPROVEMENT_RECEIPT_INTERFACE:
        return improvement_receipt_identity(mapping)
    if interface == GUI_VERIFICATION_RECEIPT_ENVELOPE_INTERFACE:
        return envelope_identity(mapping)
    return canonical_identity(
        mapping,
        domain=DOMAIN_TERMINAL_RECEIPT,
        schema_version=GUI_TERMINAL_RECEIPT_REF_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiPhaseRecord:
    """One immutable phase append.  ``effect_id`` is the idempotency key."""

    phase: JournalPhase
    effect_id: str
    status: PhaseRecordStatus
    payload_cid: str
    artifact_cids: tuple[str, ...] = ()
    receipt_cid: str = ""
    interface: str = "GuiPhaseRecord@1"
    schema_version: str = GUI_PHASE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", _as_phase(self.phase))
        object.__setattr__(
            self, "effect_id", _identifier(self.effect_id, "effect_id")
        )
        object.__setattr__(self, "status", _as_phase_status(self.status))
        object.__setattr__(
            self, "payload_cid", _require_cid(self.payload_cid, "payload_cid")
        )
        cids = tuple(
            _require_cid(item, "artifact_cids[]") for item in self.artifact_cids
        )
        object.__setattr__(self, "artifact_cids", cids)
        if self.receipt_cid:
            object.__setattr__(
                self, "receipt_cid", _require_cid(self.receipt_cid, "receipt_cid")
            )
        else:
            object.__setattr__(self, "receipt_cid", "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_cids": list(self.artifact_cids),
            "effect_id": self.effect_id,
            "interface": self.interface,
            "payload_cid": self.payload_cid,
            "phase": self.phase.value,
            "receipt_cid": self.receipt_cid,
            "schema_version": self.schema_version,
            "status": self.status.value,
        }

    def identity(self) -> GuiCanonicalIdentity:
        return canonical_identity(
            self.to_dict(),
            domain=DOMAIN_PHASE_RECORD,
            schema_version=self.schema_version,
        )


@dataclass(frozen=True)
class GuiRunCheckpoint:
    """Content-addressed journal head.  Interface: ``GuiRunCheckpoint@1``."""

    run_id: str
    run_identity_cid: str
    attempt: int
    phase: JournalPhase
    status: RunStatus
    application_id: str
    screen_id: str
    objective_id: str
    source_revision: str
    canonical_branch: str
    canonical_revision: str
    canonical_porcelain: str
    heartbeat_seq: int
    phase_records: tuple[GuiPhaseRecord, ...]
    cid: str
    digest: str
    proposal_id: str = ""
    worktree_path: str = ""
    worktree_revision: str = ""
    worktree_lease_id: str = ""
    artifact_manifest_cid: str = ""
    terminal_receipt_cid: str = ""
    terminal_receipt_digest: str = ""
    prev_checkpoint_cid: str = ""
    interface: str = GUI_RUN_CHECKPOINT_INTERFACE
    schema_version: str = GUI_RUN_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _run_id(self.run_id))
        object.__setattr__(
            self,
            "run_identity_cid",
            _require_cid(self.run_identity_cid, "run_identity_cid"),
        )
        object.__setattr__(self, "attempt", _positive_int(self.attempt, "attempt"))
        object.__setattr__(self, "phase", _as_phase(self.phase))
        object.__setattr__(self, "status", _as_run_status(self.status))
        object.__setattr__(
            self,
            "application_id",
            _identifier(self.application_id, "application_id"),
        )
        object.__setattr__(
            self, "screen_id", _identifier(self.screen_id, "screen_id")
        )
        object.__setattr__(
            self, "objective_id", _identifier(self.objective_id, "objective_id")
        )
        object.__setattr__(
            self,
            "source_revision",
            _require_revision(self.source_revision, "source_revision"),
        )
        object.__setattr__(
            self,
            "canonical_branch",
            _identifier(self.canonical_branch, "canonical_branch"),
        )
        object.__setattr__(
            self,
            "canonical_revision",
            _require_revision(self.canonical_revision, "canonical_revision"),
        )
        object.__setattr__(
            self,
            "canonical_porcelain",
            _exact_str(self.canonical_porcelain, "canonical_porcelain"),
        )
        object.__setattr__(
            self,
            "heartbeat_seq",
            _nonneg_int(self.heartbeat_seq, "heartbeat_seq"),
        )
        records = tuple(self.phase_records)
        for record in records:
            if type(record) is not GuiPhaseRecord:
                raise GuiRunJournalError(
                    "phase_records must contain GuiPhaseRecord values",
                    reason_code=JournalReasonCode.INVALID_COLLECTION_TYPE.value,
                )
        object.__setattr__(self, "phase_records", records)
        object.__setattr__(self, "cid", _require_cid(self.cid, "cid"))
        digest = _identifier(self.digest, "digest")
        if not _DIGEST_RE.fullmatch(digest):
            raise GuiRunJournalError(
                "checkpoint digest must be sha256:<hex>",
                reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
            )
        object.__setattr__(self, "digest", digest)
        if self.proposal_id:
            object.__setattr__(
                self, "proposal_id", _identifier(self.proposal_id, "proposal_id")
            )
        if self.worktree_revision:
            object.__setattr__(
                self,
                "worktree_revision",
                _require_revision(self.worktree_revision, "worktree_revision"),
            )
        if self.worktree_lease_id:
            object.__setattr__(
                self,
                "worktree_lease_id",
                _identifier(self.worktree_lease_id, "worktree_lease_id"),
            )
        if self.artifact_manifest_cid:
            object.__setattr__(
                self,
                "artifact_manifest_cid",
                _require_cid(self.artifact_manifest_cid, "artifact_manifest_cid"),
            )
        if self.terminal_receipt_cid:
            object.__setattr__(
                self,
                "terminal_receipt_cid",
                _require_cid(self.terminal_receipt_cid, "terminal_receipt_cid"),
            )
        if self.terminal_receipt_digest:
            digest = _identifier(
                self.terminal_receipt_digest, "terminal_receipt_digest"
            )
            if not _DIGEST_RE.fullmatch(digest):
                raise GuiRunJournalError(
                    "terminal_receipt_digest must be sha256:<hex>",
                    reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
                )
            object.__setattr__(self, "terminal_receipt_digest", digest)
        if self.prev_checkpoint_cid:
            object.__setattr__(
                self,
                "prev_checkpoint_cid",
                _require_cid(self.prev_checkpoint_cid, "prev_checkpoint_cid"),
            )
        if (
            self.status is RunStatus.COMPLETED
            and not self.terminal_receipt_cid
        ):
            raise GuiRunJournalError(
                "completed checkpoints require a terminal receipt identity",
                reason_code=JournalReasonCode.CORRUPT_JOURNAL.value,
            )

    @property
    def effect_ids(self) -> tuple[str, ...]:
        return tuple(record.effect_id for record in self.phase_records)

    def record_for_effect(self, effect_id: str) -> GuiPhaseRecord | None:
        for record in self.phase_records:
            if record.effect_id == effect_id:
                return record
        return None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "artifact_manifest_cid": self.artifact_manifest_cid,
            "attempt": self.attempt,
            "canonical_branch": self.canonical_branch,
            "canonical_porcelain": self.canonical_porcelain,
            "canonical_revision": self.canonical_revision,
            "heartbeat_seq": self.heartbeat_seq,
            "interface": self.interface,
            "objective_id": self.objective_id,
            "phase": self.phase.value,
            "phase_records": [record.to_dict() for record in self.phase_records],
            "prev_checkpoint_cid": self.prev_checkpoint_cid,
            "proposal_id": self.proposal_id,
            "run_id": self.run_id,
            "run_identity_cid": self.run_identity_cid,
            "schema_version": self.schema_version,
            "screen_id": self.screen_id,
            "source_revision": self.source_revision,
            "status": self.status.value,
            "terminal_receipt_cid": self.terminal_receipt_cid,
            "terminal_receipt_digest": self.terminal_receipt_digest,
            "worktree_lease_id": self.worktree_lease_id,
            "worktree_path": self.worktree_path,
            "worktree_revision": self.worktree_revision,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["cid"] = self.cid
        payload["digest"] = self.digest
        return payload


@dataclass(frozen=True)
class GuiResumeDecision:
    """Typed resume/restart/reject outcome.  Interface: ``GuiResumeDecision@1``."""

    action: ResumeAction
    reason_codes: tuple[str, ...]
    run_id: str
    checkpoint: GuiRunCheckpoint | None = None
    terminal_receipt_cid: str = ""
    terminal_receipt_digest: str = ""
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    interface: str = GUI_RESUME_DECISION_INTERFACE
    schema_version: str = GUI_RESUME_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.action) is not ResumeAction:
            object.__setattr__(self, "action", ResumeAction(str(self.action)))
        codes = tuple(
            sorted({_text(code, "reason_code") for code in self.reason_codes})
        )
        if not codes:
            codes = (JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,)
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "run_id", _run_id(self.run_id))
        if self.checkpoint is not None and type(self.checkpoint) is not GuiRunCheckpoint:
            raise GuiRunJournalError(
                "checkpoint must be a GuiRunCheckpoint when present",
                reason_code=JournalReasonCode.INVALID_COLLECTION_TYPE.value,
            )
        if self.terminal_receipt_cid:
            object.__setattr__(
                self,
                "terminal_receipt_cid",
                _require_cid(self.terminal_receipt_cid, "terminal_receipt_cid"),
            )
        object.__setattr__(
            self, "details", _freeze_mapping(dict(self.details) if self.details else {})
        )
        object.__setattr__(self, "message", str(self.message or ""))

    @property
    def resume(self) -> bool:
        return self.action is ResumeAction.RESUME

    @property
    def reject(self) -> bool:
        return self.action is ResumeAction.REJECT

    @property
    def completed(self) -> bool:
        return self.action is ResumeAction.RETURN_COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "checkpoint_cid": (
                "" if self.checkpoint is None else self.checkpoint.cid
            ),
            "completed": self.completed,
            "details": dict(self.details),
            "interface": self.interface,
            "message": self.message,
            "reason_codes": list(self.reason_codes),
            "reject": self.reject,
            "resume": self.resume,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "terminal_receipt_cid": self.terminal_receipt_cid,
            "terminal_receipt_digest": self.terminal_receipt_digest,
        }


# ---------------------------------------------------------------------------
# GuiRunJournal@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiRunJournal:
    """Append-only content-addressed journal.  Interface: ``GuiRunJournal@1``."""

    store: GuiEvidenceArtifactStore
    interface: str = GUI_RUN_JOURNAL_INTERFACE
    schema: str = GUI_RUN_JOURNAL_SCHEMA

    def __post_init__(self) -> None:
        if type(self.store) is not GuiEvidenceArtifactStore:
            raise GuiRunJournalError(
                "store must be a GuiEvidenceArtifactStore",
                reason_code=JournalReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"value_type": type(self.store).__name__},
            )
        journals = self.store.host_root / "journals"
        journals.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))

    @property
    def host_root(self) -> Path:
        return self.store.host_root

    def open_run(
        self,
        *,
        run_id: str,
        application_id: str,
        screen_id: str,
        objective_id: str,
        source_revision: str,
        canonical_branch: str,
        canonical_revision: str,
        canonical_porcelain: str = "",
        proposal_id: str = "",
        attempt: int = 1,
        worktree_path: str = "",
        worktree_revision: str = "",
        worktree_lease_id: str = "",
    ) -> GuiRunCheckpoint:
        """Create or return the existing open checkpoint for ``run_id``."""

        typed_run = _run_id(run_id)
        identity = run_identity(
            run_id=typed_run,
            application_id=_identifier(application_id, "application_id"),
            screen_id=_identifier(screen_id, "screen_id"),
            objective_id=_identifier(objective_id, "objective_id"),
            source_revision=_require_revision(source_revision, "source_revision"),
            proposal_id=_identifier(proposal_id, "proposal_id")
            if proposal_id
            else "",
            attempt=_positive_int(attempt, "attempt"),
        )
        existing = self._try_load_head(typed_run)
        if existing is not None:
            if existing.run_identity_cid != identity.cid:
                raise GuiRunJournalError(
                    "open_run identity does not match the persisted journal",
                    reason_code=JournalReasonCode.REVISION_MISMATCH.value,
                    details={
                        "run_id": typed_run,
                        "persisted": existing.run_identity_cid,
                        "requested": identity.cid,
                    },
                )
            return existing
        return self._commit_checkpoint(
            run_id=typed_run,
            run_identity_cid=identity.cid,
            attempt=_positive_int(attempt, "attempt"),
            phase=JournalPhase.BASELINE,
            status=RunStatus.OPEN,
            application_id=_identifier(application_id, "application_id"),
            screen_id=_identifier(screen_id, "screen_id"),
            objective_id=_identifier(objective_id, "objective_id"),
            source_revision=_require_revision(source_revision, "source_revision"),
            canonical_branch=_identifier(canonical_branch, "canonical_branch"),
            canonical_revision=_require_revision(
                canonical_revision, "canonical_revision"
            ),
            canonical_porcelain=_exact_str(
                canonical_porcelain, "canonical_porcelain"
            ),
            heartbeat_seq=0,
            phase_records=(),
            proposal_id=_identifier(proposal_id, "proposal_id")
            if proposal_id
            else "",
            worktree_path=worktree_path,
            worktree_revision=worktree_revision,
            worktree_lease_id=worktree_lease_id,
        )

    def open_from_mapping(self, raw: Mapping[str, Any]) -> GuiRunCheckpoint:
        payload = _require_mapping(raw, "open")
        _reject_forbidden_path_keys(payload, "open")
        _reject_unknown(payload, _OPEN_KEYS, "open")
        required = (
            "run_id",
            "application_id",
            "screen_id",
            "objective_id",
            "source_revision",
            "canonical_branch",
            "canonical_revision",
        )
        for key in required:
            if key not in payload:
                raise GuiRunJournalError(
                    f"open.{key} is required",
                    reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
                    details={"field": key},
                )
            _reject_present_null(payload, key)
        attempt = payload["attempt"] if "attempt" in payload else 1
        if "attempt" in payload:
            _reject_present_null(payload, "attempt")
        return self.open_run(
            run_id=payload["run_id"],
            application_id=payload["application_id"],
            screen_id=payload["screen_id"],
            objective_id=payload["objective_id"],
            source_revision=payload["source_revision"],
            canonical_branch=payload["canonical_branch"],
            canonical_revision=payload["canonical_revision"],
            canonical_porcelain=_optional_text(payload, "canonical_porcelain"),
            proposal_id=_optional_identifier(payload, "proposal_id"),
            attempt=attempt,
            worktree_path=_optional_text(payload, "worktree_path"),
            worktree_revision=_optional_revision(payload, "worktree_revision"),
            worktree_lease_id=_optional_identifier(payload, "worktree_lease_id"),
        )

    def append_phase(
        self,
        *,
        run_id: str,
        phase: JournalPhase | str,
        effect_id: str,
        payload: Mapping[str, Any],
        status: PhaseRecordStatus | str = PhaseRecordStatus.COMPLETED,
        artifact_cids: Sequence[str] = (),
        receipt_cid: str = "",
    ) -> GuiRunCheckpoint:
        """Append one immutable phase record.  Duplicate effects are no-ops."""

        checkpoint = self.require_checkpoint(run_id)
        if checkpoint.status is RunStatus.COMPLETED:
            raise GuiRunJournalError(
                "cannot append to a completed journal",
                reason_code=JournalReasonCode.COMPLETED_RECEIPT_MISMATCH.value,
                details={"run_id": checkpoint.run_id},
            )
        typed_phase = _as_phase(phase)
        typed_effect = _identifier(effect_id, "effect_id")
        typed_status = _as_phase_status(status)
        mapping = _require_mapping(payload, "payload")
        _reject_forbidden_path_keys(mapping, "payload")
        stored_payload = self.store.put(
            mapping,
            kind=ArtifactKind.JOURNAL_RECORD,
            binding=journal_binding(
                run_id=checkpoint.run_id,
                source_revision=checkpoint.source_revision,
            ),
        )
        verified_artifacts = tuple(
            self.store.rehash(item).cid for item in artifact_cids
        )
        typed_receipt = (
            _require_cid(receipt_cid, "receipt_cid") if receipt_cid else ""
        )
        if typed_receipt:
            self.store.rehash(typed_receipt)
        record = GuiPhaseRecord(
            phase=typed_phase,
            effect_id=typed_effect,
            status=typed_status,
            payload_cid=stored_payload.cid,
            artifact_cids=verified_artifacts,
            receipt_cid=typed_receipt,
        )
        existing = checkpoint.record_for_effect(typed_effect)
        if existing is not None:
            if existing.identity().cid != record.identity().cid:
                raise GuiRunJournalError(
                    "effect_id already recorded with a different payload",
                    reason_code=JournalReasonCode.DUPLICATE_EFFECT.value,
                    details={
                        "effect_id": typed_effect,
                        "existing_payload_cid": existing.payload_cid,
                        "requested_payload_cid": record.payload_cid,
                    },
                )
            return checkpoint
        run_status = RunStatus.IN_PROGRESS
        if typed_status is PhaseRecordStatus.INTERRUPTED:
            run_status = RunStatus.INTERRUPTED
        elif typed_status is PhaseRecordStatus.REJECTED:
            run_status = RunStatus.REJECTED
        return self._commit_checkpoint(
            run_id=checkpoint.run_id,
            run_identity_cid=checkpoint.run_identity_cid,
            attempt=checkpoint.attempt,
            phase=typed_phase,
            status=run_status,
            application_id=checkpoint.application_id,
            screen_id=checkpoint.screen_id,
            objective_id=checkpoint.objective_id,
            source_revision=checkpoint.source_revision,
            canonical_branch=checkpoint.canonical_branch,
            canonical_revision=checkpoint.canonical_revision,
            canonical_porcelain=checkpoint.canonical_porcelain,
            heartbeat_seq=checkpoint.heartbeat_seq,
            phase_records=checkpoint.phase_records + (record,),
            proposal_id=checkpoint.proposal_id,
            worktree_path=checkpoint.worktree_path,
            worktree_revision=checkpoint.worktree_revision,
            worktree_lease_id=checkpoint.worktree_lease_id,
            artifact_manifest_cid=checkpoint.artifact_manifest_cid,
            terminal_receipt_cid=checkpoint.terminal_receipt_cid,
            terminal_receipt_digest=checkpoint.terminal_receipt_digest,
            prev_checkpoint_cid=checkpoint.cid,
        )

    def append_from_mapping(self, raw: Mapping[str, Any]) -> GuiRunCheckpoint:
        payload = _require_mapping(raw, "append")
        _reject_forbidden_path_keys(payload, "append")
        _reject_unknown(payload, _APPEND_KEYS, "append")
        for key in ("run_id", "phase", "effect_id", "payload"):
            if key not in payload:
                raise GuiRunJournalError(
                    f"append.{key} is required",
                    reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
                    details={"field": key},
                )
            _reject_present_null(payload, key)
        artifacts = payload["artifact_cids"] if "artifact_cids" in payload else []
        if "artifact_cids" in payload:
            _reject_present_null(payload, "artifact_cids")
            artifacts = _require_list(artifacts, "artifact_cids")
        status = (
            payload["status"]
            if "status" in payload
            else PhaseRecordStatus.COMPLETED
        )
        if "status" in payload:
            _reject_present_null(payload, "status")
        receipt = payload["receipt_cid"] if "receipt_cid" in payload else ""
        if "receipt_cid" in payload:
            _reject_present_null(payload, "receipt_cid")
        return self.append_phase(
            run_id=payload["run_id"],
            phase=payload["phase"],
            effect_id=payload["effect_id"],
            payload=payload["payload"],
            status=status,
            artifact_cids=artifacts,
            receipt_cid=receipt,
        )

    def heartbeat(
        self,
        run_id: str,
        *,
        worktree_path: str = "",
        worktree_revision: str = "",
        worktree_lease_id: str = "",
    ) -> GuiRunCheckpoint:
        """Record progress without treating liveness as completion."""

        checkpoint = self.require_checkpoint(run_id)
        if checkpoint.status is RunStatus.COMPLETED:
            return checkpoint
        return self._commit_checkpoint(
            run_id=checkpoint.run_id,
            run_identity_cid=checkpoint.run_identity_cid,
            attempt=checkpoint.attempt,
            phase=checkpoint.phase,
            status=(
                RunStatus.IN_PROGRESS
                if checkpoint.status is RunStatus.OPEN
                else checkpoint.status
            ),
            application_id=checkpoint.application_id,
            screen_id=checkpoint.screen_id,
            objective_id=checkpoint.objective_id,
            source_revision=checkpoint.source_revision,
            canonical_branch=checkpoint.canonical_branch,
            canonical_revision=checkpoint.canonical_revision,
            canonical_porcelain=checkpoint.canonical_porcelain,
            heartbeat_seq=checkpoint.heartbeat_seq + 1,
            phase_records=checkpoint.phase_records,
            proposal_id=checkpoint.proposal_id,
            worktree_path=worktree_path or checkpoint.worktree_path,
            worktree_revision=worktree_revision or checkpoint.worktree_revision,
            worktree_lease_id=worktree_lease_id or checkpoint.worktree_lease_id,
            artifact_manifest_cid=checkpoint.artifact_manifest_cid,
            terminal_receipt_cid=checkpoint.terminal_receipt_cid,
            terminal_receipt_digest=checkpoint.terminal_receipt_digest,
            prev_checkpoint_cid=checkpoint.cid,
        )

    def mark_interrupted(self, run_id: str) -> GuiRunCheckpoint:
        """Persist interruption.  Process death is never completion."""

        checkpoint = self.require_checkpoint(run_id)
        if checkpoint.status is RunStatus.COMPLETED:
            return checkpoint
        return self._commit_checkpoint(
            run_id=checkpoint.run_id,
            run_identity_cid=checkpoint.run_identity_cid,
            attempt=checkpoint.attempt,
            phase=checkpoint.phase,
            status=RunStatus.INTERRUPTED,
            application_id=checkpoint.application_id,
            screen_id=checkpoint.screen_id,
            objective_id=checkpoint.objective_id,
            source_revision=checkpoint.source_revision,
            canonical_branch=checkpoint.canonical_branch,
            canonical_revision=checkpoint.canonical_revision,
            canonical_porcelain=checkpoint.canonical_porcelain,
            heartbeat_seq=checkpoint.heartbeat_seq,
            phase_records=checkpoint.phase_records,
            proposal_id=checkpoint.proposal_id,
            worktree_path=checkpoint.worktree_path,
            worktree_revision=checkpoint.worktree_revision,
            worktree_lease_id=checkpoint.worktree_lease_id,
            artifact_manifest_cid=checkpoint.artifact_manifest_cid,
            terminal_receipt_cid="",
            terminal_receipt_digest="",
            prev_checkpoint_cid=checkpoint.cid,
        )

    def commit_terminal_receipt(
        self,
        run_id: str,
        receipt: Mapping[str, Any],
    ) -> GuiRunCheckpoint:
        """Seal the journal with a content-addressed terminal receipt."""

        checkpoint = self.require_checkpoint(run_id)
        mapping = _require_mapping(receipt, "terminal_receipt")
        _reject_forbidden_path_keys(mapping, "terminal_receipt")
        identity = terminal_receipt_identity(mapping)
        stored = self.store.put(
            mapping,
            kind=ArtifactKind.RECEIPT,
            binding=journal_binding(
                run_id=checkpoint.run_id,
                source_revision=checkpoint.source_revision,
            ),
        )
        if checkpoint.status is RunStatus.COMPLETED:
            if (
                checkpoint.terminal_receipt_cid != stored.cid
                or checkpoint.terminal_receipt_digest != identity.digest
            ):
                raise GuiRunJournalError(
                    "completed run already sealed with a different receipt identity",
                    reason_code=JournalReasonCode.COMPLETED_RECEIPT_MISMATCH.value,
                    details={
                        "existing_cid": checkpoint.terminal_receipt_cid,
                        "requested_cid": stored.cid,
                    },
                )
            return checkpoint
        return self._commit_checkpoint(
            run_id=checkpoint.run_id,
            run_identity_cid=checkpoint.run_identity_cid,
            attempt=checkpoint.attempt,
            phase=JournalPhase.RECEIPT,
            status=RunStatus.COMPLETED,
            application_id=checkpoint.application_id,
            screen_id=checkpoint.screen_id,
            objective_id=checkpoint.objective_id,
            source_revision=checkpoint.source_revision,
            canonical_branch=checkpoint.canonical_branch,
            canonical_revision=checkpoint.canonical_revision,
            canonical_porcelain=checkpoint.canonical_porcelain,
            heartbeat_seq=checkpoint.heartbeat_seq,
            phase_records=checkpoint.phase_records,
            proposal_id=checkpoint.proposal_id,
            worktree_path=checkpoint.worktree_path,
            worktree_revision=checkpoint.worktree_revision,
            worktree_lease_id=checkpoint.worktree_lease_id,
            artifact_manifest_cid=checkpoint.artifact_manifest_cid,
            terminal_receipt_cid=stored.cid,
            terminal_receipt_digest=identity.digest,
            prev_checkpoint_cid=checkpoint.cid,
        )

    def bind_manifest(
        self,
        run_id: str,
        artifacts: Sequence[StoredArtifact | str],
    ) -> GuiRunCheckpoint:
        """Attach a closed artifact manifest CID to the current checkpoint."""

        checkpoint = self.require_checkpoint(run_id)
        manifest = self.store.put_manifest(
            run_id=checkpoint.run_id,
            artifacts=artifacts,
            binding=journal_binding(
                run_id=checkpoint.run_id,
                source_revision=checkpoint.source_revision,
            ),
        )
        if checkpoint.artifact_manifest_cid == manifest.cid:
            return checkpoint
        if checkpoint.status is RunStatus.COMPLETED:
            raise GuiRunJournalError(
                "cannot mutate a completed journal manifest",
                reason_code=JournalReasonCode.COMPLETED_RECEIPT_MISMATCH.value,
            )
        return self._commit_checkpoint(
            run_id=checkpoint.run_id,
            run_identity_cid=checkpoint.run_identity_cid,
            attempt=checkpoint.attempt,
            phase=checkpoint.phase,
            status=checkpoint.status,
            application_id=checkpoint.application_id,
            screen_id=checkpoint.screen_id,
            objective_id=checkpoint.objective_id,
            source_revision=checkpoint.source_revision,
            canonical_branch=checkpoint.canonical_branch,
            canonical_revision=checkpoint.canonical_revision,
            canonical_porcelain=checkpoint.canonical_porcelain,
            heartbeat_seq=checkpoint.heartbeat_seq,
            phase_records=checkpoint.phase_records,
            proposal_id=checkpoint.proposal_id,
            worktree_path=checkpoint.worktree_path,
            worktree_revision=checkpoint.worktree_revision,
            worktree_lease_id=checkpoint.worktree_lease_id,
            artifact_manifest_cid=manifest.cid,
            terminal_receipt_cid=checkpoint.terminal_receipt_cid,
            terminal_receipt_digest=checkpoint.terminal_receipt_digest,
            prev_checkpoint_cid=checkpoint.cid,
        )

    def decide_resume(
        self,
        *,
        run_id: str,
        source_revision: str,
        canonical_branch: str,
        canonical_revision: str,
        canonical_porcelain: str = "",
        worktree_path: str = "",
        worktree_revision: str = "",
        worktree_lease_id: str = "",
        process_alive: bool | None = None,
    ) -> GuiResumeDecision:
        """Choose resume, restart, reject, or return a sealed receipt."""

        typed_run = _run_id(run_id)
        if process_alive is not None and type(process_alive) is not bool:
            raise GuiRunJournalError(
                "process_alive must be a boolean when present",
                reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
                details={"value_type": type(process_alive).__name__},
            )
        try:
            checkpoint = self._load_head(typed_run)
        except GuiRunJournalError as exc:
            if exc.reason_code == JournalReasonCode.MISSING_RUN.value:
                return GuiResumeDecision(
                    action=ResumeAction.RESTART,
                    reason_codes=(JournalReasonCode.RESTART.value,),
                    run_id=typed_run,
                    message="no durable journal exists; start a fresh run",
                )
            raise
        requested_revision = _require_revision(source_revision, "source_revision")
        requested_branch = _identifier(canonical_branch, "canonical_branch")
        requested_canonical = _require_revision(
            canonical_revision, "canonical_revision"
        )
        requested_porcelain = _exact_str(
            canonical_porcelain, "canonical_porcelain"
        )
        if checkpoint.status is RunStatus.COMPLETED:
            if (
                checkpoint.source_revision != requested_revision
                or checkpoint.canonical_branch != requested_branch
                or checkpoint.canonical_revision != requested_canonical
            ):
                return GuiResumeDecision(
                    action=ResumeAction.REJECT,
                    reason_codes=(JournalReasonCode.REVISION_MISMATCH.value,),
                    run_id=typed_run,
                    checkpoint=checkpoint,
                    message="completed journal identities no longer match",
                    details={
                        "persisted_revision": checkpoint.source_revision,
                        "requested_revision": requested_revision,
                    },
                )
            if (
                requested_porcelain
                and requested_porcelain != checkpoint.canonical_porcelain
            ):
                return GuiResumeDecision(
                    action=ResumeAction.REJECT,
                    reason_codes=(
                        JournalReasonCode.CANONICAL_MUTATION_DETECTED.value,
                    ),
                    run_id=typed_run,
                    checkpoint=checkpoint,
                    message="canonical checkout mutated after completion",
                )
            return GuiResumeDecision(
                action=ResumeAction.RETURN_COMPLETED,
                reason_codes=(JournalReasonCode.RETURN_COMPLETED.value,),
                run_id=typed_run,
                checkpoint=checkpoint,
                terminal_receipt_cid=checkpoint.terminal_receipt_cid,
                terminal_receipt_digest=checkpoint.terminal_receipt_digest,
                message="identical completed run returns the sealed receipt",
            )
        reasons: list[str] = []
        if process_alive is False:
            reasons.append(JournalReasonCode.PROCESS_EXIT_NOT_COMPLETION.value)
        if checkpoint.source_revision != requested_revision:
            return GuiResumeDecision(
                action=ResumeAction.REJECT,
                reason_codes=(JournalReasonCode.REVISION_MISMATCH.value,),
                run_id=typed_run,
                checkpoint=checkpoint,
                message="source revision does not match the journaled identity",
                details={
                    "persisted_revision": checkpoint.source_revision,
                    "requested_revision": requested_revision,
                },
            )
        if (
            checkpoint.canonical_branch != requested_branch
            or checkpoint.canonical_revision != requested_canonical
        ):
            return GuiResumeDecision(
                action=ResumeAction.REJECT,
                reason_codes=(
                    JournalReasonCode.CANONICAL_MUTATION_DETECTED.value,
                ),
                run_id=typed_run,
                checkpoint=checkpoint,
                message="canonical branch identity changed; refusing resume",
                details={
                    "persisted_branch": checkpoint.canonical_branch,
                    "persisted_revision": checkpoint.canonical_revision,
                    "requested_branch": requested_branch,
                    "requested_revision": requested_canonical,
                },
            )
        if (
            requested_porcelain
            and requested_porcelain != checkpoint.canonical_porcelain
        ):
            return GuiResumeDecision(
                action=ResumeAction.REJECT,
                reason_codes=(
                    JournalReasonCode.CANONICAL_MUTATION_DETECTED.value,
                ),
                run_id=typed_run,
                checkpoint=checkpoint,
                message="canonical working tree mutated; refusing resume",
            )
        worktree_reason = self._worktree_mismatch(
            checkpoint,
            worktree_path=worktree_path,
            worktree_revision=worktree_revision,
            worktree_lease_id=worktree_lease_id,
        )
        if worktree_reason:
            return GuiResumeDecision(
                action=ResumeAction.REJECT,
                reason_codes=(worktree_reason,),
                run_id=typed_run,
                checkpoint=checkpoint,
                message="worktree state is stale or foreign",
                details={
                    "persisted_path": checkpoint.worktree_path,
                    "persisted_revision": checkpoint.worktree_revision,
                    "persisted_lease_id": checkpoint.worktree_lease_id,
                    "requested_path": worktree_path,
                    "requested_revision": worktree_revision,
                    "requested_lease_id": worktree_lease_id,
                },
            )
        reasons.append(JournalReasonCode.RESUME.value)
        if checkpoint.status is RunStatus.INTERRUPTED:
            reasons.append(JournalReasonCode.INTERRUPTED.value)
        return GuiResumeDecision(
            action=ResumeAction.RESUME,
            reason_codes=tuple(reasons),
            run_id=typed_run,
            checkpoint=checkpoint,
            message="identities revalidated; resume without duplicate effects",
        )

    def decide_from_mapping(self, raw: Mapping[str, Any]) -> GuiResumeDecision:
        payload = _require_mapping(raw, "resume")
        _reject_forbidden_path_keys(payload, "resume")
        _reject_unknown(payload, _RESUME_KEYS, "resume")
        for key in (
            "run_id",
            "source_revision",
            "canonical_branch",
            "canonical_revision",
        ):
            if key not in payload:
                raise GuiRunJournalError(
                    f"resume.{key} is required",
                    reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
                    details={"field": key},
                )
            _reject_present_null(payload, key)
        alive: bool | None = None
        if "process_alive" in payload:
            _reject_present_null(payload, "process_alive")
            if type(payload["process_alive"]) is not bool:
                raise GuiRunJournalError(
                    "process_alive must be a boolean",
                    reason_code=JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value,
                )
            alive = payload["process_alive"]
        return self.decide_resume(
            run_id=payload["run_id"],
            source_revision=payload["source_revision"],
            canonical_branch=payload["canonical_branch"],
            canonical_revision=payload["canonical_revision"],
            canonical_porcelain=_optional_text(payload, "canonical_porcelain"),
            worktree_path=_optional_text(payload, "worktree_path"),
            worktree_revision=_optional_revision(payload, "worktree_revision"),
            worktree_lease_id=_optional_identifier(payload, "worktree_lease_id"),
            process_alive=alive,
        )

    def require_checkpoint(self, run_id: str) -> GuiRunCheckpoint:
        return self._load_head(_run_id(run_id))

    def load_checkpoint(self, run_id: str) -> GuiRunCheckpoint | None:
        return self._try_load_head(_run_id(run_id))

    def _worktree_mismatch(
        self,
        checkpoint: GuiRunCheckpoint,
        *,
        worktree_path: str,
        worktree_revision: str,
        worktree_lease_id: str,
    ) -> str:
        has_worktree_effect = any(
            record.phase is JournalPhase.ISOLATED_WORKTREE
            and record.status is PhaseRecordStatus.COMPLETED
            for record in checkpoint.phase_records
        )
        persisted_path = checkpoint.worktree_path
        persisted_revision = checkpoint.worktree_revision
        persisted_lease = checkpoint.worktree_lease_id
        if not (persisted_path or persisted_revision or persisted_lease):
            if has_worktree_effect:
                return JournalReasonCode.STALE_WORKTREE.value
            return ""
        if worktree_path and persisted_path and worktree_path != persisted_path:
            return JournalReasonCode.FOREIGN_WORKTREE.value
        if (
            worktree_lease_id
            and persisted_lease
            and worktree_lease_id != persisted_lease
        ):
            return JournalReasonCode.FOREIGN_WORKTREE.value
        if (
            worktree_revision
            and persisted_revision
            and worktree_revision != persisted_revision
        ):
            return JournalReasonCode.STALE_WORKTREE.value
        if has_worktree_effect and not (
            worktree_path or worktree_revision or worktree_lease_id
        ):
            return JournalReasonCode.STALE_WORKTREE.value
        return ""

    def _journal_dir(self, run_id: str) -> Path:
        path = self.store.host_root / "journals" / run_id
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self.store.host_root.resolve(strict=False))
        except ValueError as exc:
            raise GuiRunJournalError(
                "journal path escapes the fixed host root",
                reason_code=JournalReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
                details={"run_id": run_id},
            ) from exc
        return path

    def _head_path(self, run_id: str) -> Path:
        return self._journal_dir(run_id) / "head.json"

    def _try_load_head(self, run_id: str) -> GuiRunCheckpoint | None:
        path = self._head_path(run_id)
        if not path.exists():
            return None
        return self._load_head(run_id)

    def _load_head(self, run_id: str) -> GuiRunCheckpoint:
        path = self._head_path(run_id)
        if not path.is_file():
            raise GuiRunJournalError(
                "journal head is missing",
                reason_code=JournalReasonCode.MISSING_RUN.value,
                details={"run_id": run_id},
            )
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise GuiRunJournalError(
                "journal head could not be read",
                reason_code=JournalReasonCode.CORRUPT_JOURNAL.value,
                details={"run_id": run_id, "error": str(exc)},
            ) from exc
        if not raw or not raw.strip():
            raise GuiRunJournalError(
                "journal head is truncated",
                reason_code=JournalReasonCode.TRUNCATED_JOURNAL.value,
                details={"run_id": run_id},
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuiRunJournalError(
                "journal head is corrupt",
                reason_code=JournalReasonCode.CORRUPT_JOURNAL.value,
                details={"run_id": run_id},
            ) from exc
        mapping = _require_mapping(payload, "journal head")
        if "checkpoint_cid" not in mapping:
            raise GuiRunJournalError(
                "journal head is missing checkpoint_cid",
                reason_code=JournalReasonCode.CORRUPT_JOURNAL.value,
                details={"run_id": run_id},
            )
        try:
            cid = _require_cid(mapping["checkpoint_cid"], "checkpoint_cid")
            body, _record = self.store.get(cid, kind=ArtifactKind.CHECKPOINT)
            decoded = json.loads(body.decode("utf-8"))
        except (GuiArtifactStoreError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuiRunJournalError(
                "journal checkpoint failed closed verification",
                reason_code=JournalReasonCode.CORRUPT_JOURNAL.value,
                details={"run_id": run_id},
            ) from exc
        return self._checkpoint_from_payload(decoded, stored_cid=cid)

    def _checkpoint_from_payload(
        self, raw: Mapping[str, Any], *, stored_cid: str
    ) -> GuiRunCheckpoint:
        payload = _require_mapping(raw, "checkpoint")
        try:
            records = tuple(
                GuiPhaseRecord(
                    phase=item["phase"],
                    effect_id=item["effect_id"],
                    status=item["status"],
                    payload_cid=item["payload_cid"],
                    artifact_cids=tuple(item.get("artifact_cids") or ()),
                    receipt_cid=item.get("receipt_cid") or "",
                )
                for item in _require_list(
                    payload["phase_records"], "phase_records"
                )
            )
            checkpoint = GuiRunCheckpoint(
                run_id=payload["run_id"],
                run_identity_cid=payload["run_identity_cid"],
                attempt=payload["attempt"],
                phase=payload["phase"],
                status=payload["status"],
                application_id=payload["application_id"],
                screen_id=payload["screen_id"],
                objective_id=payload["objective_id"],
                source_revision=payload["source_revision"],
                canonical_branch=payload["canonical_branch"],
                canonical_revision=payload["canonical_revision"],
                canonical_porcelain=payload.get("canonical_porcelain") or "",
                heartbeat_seq=payload["heartbeat_seq"],
                phase_records=records,
                cid=stored_cid,
                digest="sha256:" + ("0" * 64),
                proposal_id=payload.get("proposal_id") or "",
                worktree_path=payload.get("worktree_path") or "",
                worktree_revision=payload.get("worktree_revision") or "",
                worktree_lease_id=payload.get("worktree_lease_id") or "",
                artifact_manifest_cid=payload.get("artifact_manifest_cid") or "",
                terminal_receipt_cid=payload.get("terminal_receipt_cid") or "",
                terminal_receipt_digest=payload.get("terminal_receipt_digest")
                or "",
                prev_checkpoint_cid=payload.get("prev_checkpoint_cid") or "",
            )
        except (KeyError, GuiRunJournalError, TypeError) as exc:
            raise GuiRunJournalError(
                "journal checkpoint payload is corrupt",
                reason_code=JournalReasonCode.CORRUPT_JOURNAL.value,
            ) from exc
        recomputed = canonical_identity(
            checkpoint.identity_payload(),
            domain=DOMAIN_RUN_CHECKPOINT,
            schema_version=GUI_RUN_CHECKPOINT_SCHEMA,
        )
        object.__setattr__(checkpoint, "digest", recomputed.digest)
        stored_identity = canonical_identity(
            checkpoint.identity_payload(),
            domain=DOMAIN_RUN_CHECKPOINT,
            schema_version=GUI_RUN_CHECKPOINT_SCHEMA,
        )
        if stored_identity.digest != recomputed.digest:
            raise GuiRunJournalError(
                "journal checkpoint does not rehash",
                reason_code=JournalReasonCode.CORRUPT_JOURNAL.value,
                details={
                    "stored_digest": stored_identity.digest,
                    "recomputed_digest": recomputed.digest,
                },
            )
        return checkpoint

    def _commit_checkpoint(
        self,
        *,
        run_id: str,
        run_identity_cid: str,
        attempt: int,
        phase: JournalPhase,
        status: RunStatus,
        application_id: str,
        screen_id: str,
        objective_id: str,
        source_revision: str,
        canonical_branch: str,
        canonical_revision: str,
        canonical_porcelain: str,
        heartbeat_seq: int,
        phase_records: tuple[GuiPhaseRecord, ...],
        proposal_id: str = "",
        worktree_path: str = "",
        worktree_revision: str = "",
        worktree_lease_id: str = "",
        artifact_manifest_cid: str = "",
        terminal_receipt_cid: str = "",
        terminal_receipt_digest: str = "",
        prev_checkpoint_cid: str = "",
    ) -> GuiRunCheckpoint:
        draft_payload = {
            "application_id": application_id,
            "artifact_manifest_cid": artifact_manifest_cid,
            "attempt": attempt,
            "canonical_branch": canonical_branch,
            "canonical_porcelain": canonical_porcelain,
            "canonical_revision": canonical_revision,
            "heartbeat_seq": heartbeat_seq,
            "interface": GUI_RUN_CHECKPOINT_INTERFACE,
            "objective_id": objective_id,
            "phase": phase.value if type(phase) is JournalPhase else phase,
            "phase_records": [record.to_dict() for record in phase_records],
            "prev_checkpoint_cid": prev_checkpoint_cid,
            "proposal_id": proposal_id,
            "run_id": run_id,
            "run_identity_cid": run_identity_cid,
            "schema_version": GUI_RUN_CHECKPOINT_SCHEMA,
            "screen_id": screen_id,
            "source_revision": source_revision,
            "status": status.value if type(status) is RunStatus else status,
            "terminal_receipt_cid": terminal_receipt_cid,
            "terminal_receipt_digest": terminal_receipt_digest,
            "worktree_lease_id": worktree_lease_id,
            "worktree_path": worktree_path,
            "worktree_revision": worktree_revision,
        }
        identity = canonical_identity(
            draft_payload,
            domain=DOMAIN_RUN_CHECKPOINT,
            schema_version=GUI_RUN_CHECKPOINT_SCHEMA,
        )
        stored = self.store.put(
            draft_payload,
            kind=ArtifactKind.CHECKPOINT,
            binding=journal_binding(
                run_id=run_id, source_revision=source_revision
            ),
        )
        checkpoint = GuiRunCheckpoint(
            run_id=run_id,
            run_identity_cid=run_identity_cid,
            attempt=attempt,
            phase=phase,
            status=status,
            application_id=application_id,
            screen_id=screen_id,
            objective_id=objective_id,
            source_revision=source_revision,
            canonical_branch=canonical_branch,
            canonical_revision=canonical_revision,
            canonical_porcelain=canonical_porcelain,
            heartbeat_seq=heartbeat_seq,
            phase_records=phase_records,
            cid=stored.cid,
            digest=identity.digest,
            proposal_id=proposal_id,
            worktree_path=worktree_path,
            worktree_revision=worktree_revision,
            worktree_lease_id=worktree_lease_id,
            artifact_manifest_cid=artifact_manifest_cid,
            terminal_receipt_cid=terminal_receipt_cid,
            terminal_receipt_digest=terminal_receipt_digest,
            prev_checkpoint_cid=prev_checkpoint_cid,
        )
        head = {
            "checkpoint_cid": checkpoint.cid,
            "interface": GUI_RUN_JOURNAL_INTERFACE,
            "run_id": run_id,
            "schema_version": GUI_RUN_JOURNAL_SCHEMA,
        }
        directory = self._journal_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self._head_path(run_id), canonical_json_bytes(head))
        return checkpoint


def default_run_journal(host_root: Path | str) -> GuiRunJournal:
    """Construct a journal bound to a host-owned evidence CAS."""

    root = resolve_host_root(host_root, create=True)
    return GuiRunJournal(store=default_evidence_artifact_store(root))


__all__ = (
    "DOMAIN_PHASE_RECORD",
    "DOMAIN_RUN_CHECKPOINT",
    "DOMAIN_RUN_IDENTITY",
    "DOMAIN_TERMINAL_RECEIPT",
    "GUI_PHASE_RECORD_SCHEMA",
    "GUI_RESUME_DECISION_INTERFACE",
    "GUI_RESUME_DECISION_SCHEMA",
    "GUI_RUN_CHECKPOINT_INTERFACE",
    "GUI_RUN_CHECKPOINT_SCHEMA",
    "GUI_RUN_JOURNAL_INTERFACE",
    "GUI_RUN_JOURNAL_SCHEMA",
    "GUI_TERMINAL_RECEIPT_REF_SCHEMA",
    "GuiPhaseRecord",
    "GuiResumeDecision",
    "GuiRunCheckpoint",
    "GuiRunJournal",
    "GuiRunJournalError",
    "JournalPhase",
    "JournalReasonCode",
    "PhaseRecordStatus",
    "ResumeAction",
    "RunStatus",
    "default_run_journal",
    "journal_binding",
    "run_identity",
    "terminal_receipt_identity",
)
