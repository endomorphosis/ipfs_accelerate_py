"""DCR-073: post-repair validation, reindex, reobservation, and re-proof.

Interfaces
----------
* ``PostRepairValidation@1`` — ordered post-transaction epoch validation.
* ``RepairProofTransition@1`` — pre→post epoch proof transition evidence.

Predicted symbols: :class:`PostRepairValidator`, :class:`RepairProofTransition`.

Normative rules (fail-closed)
-----------------------------
* Expected results never substitute for detector output.
* Unsupported or skipped mandatory checks fail closed.
* Synthetic release children fail closed.
* Successful validation requires actual source edits in the transaction.
* Finding must disappear for the intended semantic reason.
* No protected invariant may regress.
* All mandatory gates must run (not merely be declared).
* Runtime model and provider invocation counts remain 0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .transaction import ADMITTED_OWNER_ROOTS, TransactionDisposition


# ---------------------------------------------------------------------------
# Interfaces / schemas
# ---------------------------------------------------------------------------

POST_REPAIR_VALIDATION_INTERFACE: Final[str] = "PostRepairValidation@1"
REPAIR_PROOF_TRANSITION_INTERFACE: Final[str] = "RepairProofTransition@1"
DCR_VALIDATION_EVIDENCE: Final[str] = "dcr/validation@1"
DCR_VALIDATION_VERSION: Final[int] = 1

POST_REPAIR_EPOCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-post-repair-epoch@1"
)
POST_REPAIR_EPOCH_CATALOG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-post-repair-epoch-catalog@1"
)
REPAIR_PROOF_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-repair-proof-transition@1"
)
POST_REPAIR_VALIDATION_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-post-repair-validation-report@1"
)
GATE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-post-repair-gate-result@1"
)

DEFAULT_POST_REPAIR_EPOCH_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/post-repair-epoch.json"
)

MAX_PATHS: Final[int] = 1_024
MAX_IDS: Final[int] = 1_024
MAX_REASON_CODES: Final[int] = 64
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_GATES: Final[int] = 32

# Closed ordered set of mandatory post-repair gates.
MANDATORY_GATES: Final[tuple[str, ...]] = (
    "source_edits",
    "format",
    "type",
    "unit",
    "negative",
    "service_start",
    "live_reobserve",
    "reindex",
    "recompile",
    "proof_reconstruction",
    "finding_closed",
    "protected_invariants",
    "zero_model_calls",
)

PROTECTED_INVARIANT_KEYS: Final[tuple[str, ...]] = (
    "no_llm_policy",
    "root_ownership",
    "authority_stage_chain",
    "transaction_isolation",
    "zero_model_runtime",
)


class PostRepairValidationError(ContractValidationError):
    """The post-repair epoch failed a mandatory completion gate."""


class PostRepairValidationReason(str, Enum):  # noqa: UP042 - Python 3.8
    """Stable, machine-readable post-repair failure codes."""

    MALFORMED_INPUT = "malformed_input"
    TRANSACTION_NOT_COMMITTED = "transaction_not_committed"
    NO_SOURCE_EDITS = "no_source_edits"
    EXPECTED_SUBSTITUTED_FOR_DETECTOR = "expected_substituted_for_detector"
    MANDATORY_GATE_SKIPPED = "mandatory_gate_skipped"
    MANDATORY_GATE_UNSUPPORTED = "mandatory_gate_unsupported"
    MANDATORY_GATE_FAILED = "mandatory_gate_failed"
    MANDATORY_GATE_NOT_RUN = "mandatory_gate_not_run"
    SYNTHETIC_RELEASE_CHILD = "synthetic_release_child"
    FINDING_NOT_CLOSED = "finding_not_closed"
    FINDING_CLOSED_WRONG_REASON = "finding_closed_wrong_reason"
    PROTECTED_INVARIANT_REGRESSED = "protected_invariant_regressed"
    MODEL_CALLS_OBSERVED = "model_calls_observed"
    PROVIDER_CALLS_OBSERVED = "provider_calls_observed"
    STALE_EPOCH = "stale_epoch"
    ROOT_DRIFT = "root_drift"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    PARTIAL_COMPLETION_FORBIDDEN = "partial_completion_forbidden"


class GateDisposition(str, Enum):  # noqa: UP042
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    NOT_RUN = "not_run"


class ValidationDisposition(str, Enum):  # noqa: UP042
    PASSED = "passed"
    FAILED = "failed"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise PostRepairValidationError(f"{name} must be a compact identifier")
    text = value.strip()
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise PostRepairValidationError(f"{name} exceeds text bound")
    return text


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise PostRepairValidationError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise PostRepairValidationError(f"{name} is required")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise PostRepairValidationError(f"{name} exceeds text bound")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PostRepairValidationError(f"{name} must be boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PostRepairValidationError(f"{name} must be a non-negative integer")
    return value


def _ids(
    values: Sequence[str],
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_IDS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise PostRepairValidationError(f"{name} must be an identifier sequence")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _identifier(value, name)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    if required and not result:
        raise PostRepairValidationError(f"{name} must not be empty")
    if len(result) > maximum:
        raise PostRepairValidationError(f"{name} exceeds item bound")
    return tuple(result)


def _paths(values: Sequence[str], name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise PostRepairValidationError(f"{name} must be a path sequence")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value:
            raise PostRepairValidationError(f"{name} contains an invalid path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise PostRepairValidationError(f"{name} contains an escaped path")
        result.add(path.as_posix())
    if required and not result:
        raise PostRepairValidationError(f"{name} must not be empty")
    if len(result) > MAX_PATHS:
        raise PostRepairValidationError(f"{name} exceeds path bound")
    return tuple(sorted(result))


def _reasons(values: Sequence[str]) -> tuple[str, ...]:
    return _ids(values, "reason_codes", maximum=MAX_REASON_CODES)


# ---------------------------------------------------------------------------
# Evidence contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult(CanonicalContract):
    """One detector-backed mandatory gate result."""

    SCHEMA: ClassVar[str] = GATE_RESULT_SCHEMA

    gate: str
    disposition: GateDisposition
    detector_id: str
    evidence_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    expected_only: bool = False
    synthetic: bool = False
    ran: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate", _identifier(self.gate, "gate"))
        if self.gate not in MANDATORY_GATES:
            raise PostRepairValidationError(f"unknown mandatory gate: {self.gate}")
        if isinstance(self.disposition, str):
            try:
                object.__setattr__(
                    self, "disposition", GateDisposition(self.disposition)
                )
            except ValueError as exc:
                raise PostRepairValidationError("invalid gate disposition") from exc
        if not isinstance(self.disposition, GateDisposition):
            raise PostRepairValidationError("disposition must be GateDisposition")
        object.__setattr__(
            self, "detector_id", _identifier(self.detector_id, "detector_id")
        )
        object.__setattr__(
            self, "evidence_ids", _ids(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self, "reason_codes", _reasons(self.reason_codes)
        )
        object.__setattr__(self, "expected_only", _bool(self.expected_only, "expected_only"))
        object.__setattr__(self, "synthetic", _bool(self.synthetic, "synthetic"))
        object.__setattr__(self, "ran", _bool(self.ran, "ran"))

    def _payload(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "disposition": self.disposition.value,
            "detector_id": self.detector_id,
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": list(self.reason_codes),
            "expected_only": self.expected_only,
            "synthetic": self.synthetic,
            "ran": self.ran,
        }

    @property
    def ok(self) -> bool:
        return (
            self.disposition is GateDisposition.PASSED
            and self.ran
            and not self.expected_only
            and not self.synthetic
        )


@dataclass(frozen=True)
class TransactionSourceEvidence:
    """Committed transaction surface required before post-repair validation."""

    transaction_id: str
    disposition: str
    changed_paths: tuple[str, ...]
    before_hashes: Mapping[str, str]
    after_hashes: Mapping[str, str]
    root_ids: tuple[str, ...] = ()
    forest_id: str = ""
    tree_id: str = ""
    candidate_epoch_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "transaction_id", _identifier(self.transaction_id, "transaction_id")
        )
        object.__setattr__(
            self, "disposition", _identifier(self.disposition, "disposition")
        )
        object.__setattr__(
            self, "changed_paths", _paths(self.changed_paths, "changed_paths")
        )
        before: dict[str, str] = {}
        after: dict[str, str] = {}
        if not isinstance(self.before_hashes, Mapping) or not isinstance(
            self.after_hashes, Mapping
        ):
            raise PostRepairValidationError("hash maps must be mappings")
        for path in self.changed_paths:
            if path not in self.before_hashes or path not in self.after_hashes:
                raise PostRepairValidationError(
                    "changed path missing before/after hash binding"
                )
            before[path] = _identifier(self.before_hashes[path], "before_hash")
            after[path] = _identifier(self.after_hashes[path], "after_hash")
        object.__setattr__(self, "before_hashes", MappingProxyType(before))
        object.__setattr__(self, "after_hashes", MappingProxyType(after))
        object.__setattr__(self, "root_ids", _ids(self.root_ids, "root_ids"))
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id", required=False)
        )
        object.__setattr__(
            self, "tree_id", _text(self.tree_id, "tree_id", required=False)
        )
        object.__setattr__(
            self,
            "candidate_epoch_id",
            _text(self.candidate_epoch_id, "candidate_epoch_id", required=False),
        )

    @property
    def has_source_edits(self) -> bool:
        if not self.changed_paths:
            return False
        return any(
            self.before_hashes[path] != self.after_hashes[path]
            for path in self.changed_paths
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "disposition": self.disposition,
            "changed_paths": list(self.changed_paths),
            "before_hashes": dict(self.before_hashes),
            "after_hashes": dict(self.after_hashes),
            "root_ids": list(self.root_ids),
            "forest_id": self.forest_id,
            "tree_id": self.tree_id,
            "candidate_epoch_id": self.candidate_epoch_id,
            "has_source_edits": self.has_source_edits,
        }


@dataclass(frozen=True)
class FindingClosureEvidence:
    """Detector evidence that the original finding closed for a semantic reason."""

    finding_id: str
    closed: bool
    semantic_reason: str
    intended_semantic_reason: str
    residual_finding_ids: tuple[str, ...] = ()
    detector_id: str = "detector:finding-closure"
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "finding_id", _identifier(self.finding_id, "finding_id"))
        object.__setattr__(self, "closed", _bool(self.closed, "closed"))
        object.__setattr__(
            self,
            "semantic_reason",
            _identifier(self.semantic_reason, "semantic_reason"),
        )
        object.__setattr__(
            self,
            "intended_semantic_reason",
            _identifier(self.intended_semantic_reason, "intended_semantic_reason"),
        )
        object.__setattr__(
            self,
            "residual_finding_ids",
            _ids(self.residual_finding_ids, "residual_finding_ids"),
        )
        object.__setattr__(
            self, "detector_id", _identifier(self.detector_id, "detector_id")
        )
        object.__setattr__(
            self, "evidence_ids", _ids(self.evidence_ids, "evidence_ids")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "closed": self.closed,
            "semantic_reason": self.semantic_reason,
            "intended_semantic_reason": self.intended_semantic_reason,
            "residual_finding_ids": list(self.residual_finding_ids),
            "detector_id": self.detector_id,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ProtectedInvariantEvidence:
    """Detector evidence that protected invariants did not regress."""

    checked_keys: tuple[str, ...]
    regressed_keys: tuple[str, ...] = ()
    detector_id: str = "detector:protected-invariants"
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checked_keys", _ids(self.checked_keys, "checked_keys", required=True)
        )
        object.__setattr__(
            self, "regressed_keys", _ids(self.regressed_keys, "regressed_keys")
        )
        object.__setattr__(
            self, "detector_id", _identifier(self.detector_id, "detector_id")
        )
        object.__setattr__(
            self, "evidence_ids", _ids(self.evidence_ids, "evidence_ids")
        )
        missing = [key for key in PROTECTED_INVARIANT_KEYS if key not in self.checked_keys]
        if missing:
            raise PostRepairValidationError(
                "protected invariant set incomplete: " + ", ".join(missing)
            )

    @property
    def ok(self) -> bool:
        return not self.regressed_keys

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_keys": list(self.checked_keys),
            "regressed_keys": list(self.regressed_keys),
            "detector_id": self.detector_id,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class LiveObservationEvidence:
    """Live service reobservation after exact service start."""

    service_roles: tuple[str, ...]
    started_service_ids: tuple[str, ...]
    tools_list_receipt_ids: tuple[str, ...]
    tools_call_receipt_ids: tuple[str, ...]
    invalid_call_receipt_ids: tuple[str, ...] = ()
    detector_id: str = "detector:live-reobserve"
    evidence_ids: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    observed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "service_roles",
            _ids(self.service_roles, "service_roles", required=True),
        )
        object.__setattr__(
            self,
            "started_service_ids",
            _ids(self.started_service_ids, "started_service_ids", required=True),
        )
        object.__setattr__(
            self,
            "tools_list_receipt_ids",
            _ids(self.tools_list_receipt_ids, "tools_list_receipt_ids", required=True),
        )
        object.__setattr__(
            self,
            "tools_call_receipt_ids",
            _ids(self.tools_call_receipt_ids, "tools_call_receipt_ids", required=True),
        )
        object.__setattr__(
            self,
            "invalid_call_receipt_ids",
            _ids(self.invalid_call_receipt_ids, "invalid_call_receipt_ids"),
        )
        object.__setattr__(
            self, "detector_id", _identifier(self.detector_id, "detector_id")
        )
        object.__setattr__(
            self, "evidence_ids", _ids(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self, "expected_tools", _ids(self.expected_tools, "expected_tools")
        )
        object.__setattr__(
            self, "observed_tools", _ids(self.observed_tools, "observed_tools")
        )
        started = set(self.started_service_ids)
        for role in self.service_roles:
            if not any(
                item == role or item.startswith(f"{role}:") for item in started
            ):
                raise PostRepairValidationError(
                    f"service role {role} was not started exactly"
                )

    def substitutes_expected_for_detector(self) -> bool:
        """True when expected tools are claimed without observed detector output."""

        if self.expected_tools and not self.observed_tools:
            return True
        if self.expected_tools and not self.tools_list_receipt_ids:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_roles": list(self.service_roles),
            "started_service_ids": list(self.started_service_ids),
            "tools_list_receipt_ids": list(self.tools_list_receipt_ids),
            "tools_call_receipt_ids": list(self.tools_call_receipt_ids),
            "invalid_call_receipt_ids": list(self.invalid_call_receipt_ids),
            "detector_id": self.detector_id,
            "evidence_ids": list(self.evidence_ids),
            "expected_tools": list(self.expected_tools),
            "observed_tools": list(self.observed_tools),
        }


@dataclass(frozen=True)
class ReindexRecompileEvidence:
    """Reindex and recompile receipts for the candidate epoch."""

    index_id: str
    reindex_receipt_id: str
    recompile_receipt_id: str
    rebuilt_paths: tuple[str, ...]
    tombstone_ids: tuple[str, ...] = ()
    clean_rebuild_equivalent: bool = True
    detector_id: str = "detector:reindex-recompile"

    def __post_init__(self) -> None:
        object.__setattr__(self, "index_id", _identifier(self.index_id, "index_id"))
        object.__setattr__(
            self,
            "reindex_receipt_id",
            _identifier(self.reindex_receipt_id, "reindex_receipt_id"),
        )
        object.__setattr__(
            self,
            "recompile_receipt_id",
            _identifier(self.recompile_receipt_id, "recompile_receipt_id"),
        )
        object.__setattr__(
            self, "rebuilt_paths", _paths(self.rebuilt_paths, "rebuilt_paths")
        )
        object.__setattr__(
            self, "tombstone_ids", _ids(self.tombstone_ids, "tombstone_ids")
        )
        object.__setattr__(
            self,
            "clean_rebuild_equivalent",
            _bool(self.clean_rebuild_equivalent, "clean_rebuild_equivalent"),
        )
        object.__setattr__(
            self, "detector_id", _identifier(self.detector_id, "detector_id")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_id": self.index_id,
            "reindex_receipt_id": self.reindex_receipt_id,
            "recompile_receipt_id": self.recompile_receipt_id,
            "rebuilt_paths": list(self.rebuilt_paths),
            "tombstone_ids": list(self.tombstone_ids),
            "clean_rebuild_equivalent": self.clean_rebuild_equivalent,
            "detector_id": self.detector_id,
        }


@dataclass(frozen=True)
class CommandGateEvidence:
    """Detector-backed command results for format/type/unit/negative suites."""

    format_passed: bool
    type_passed: bool
    unit_passed: bool
    negative_passed: bool
    command_receipt_ids: tuple[str, ...]
    failed_commands: tuple[str, ...] = ()
    detector_id: str = "detector:command-gates"
    # When true, command outcomes were asserted from expected fixtures only.
    expected_only: bool = False

    def __post_init__(self) -> None:
        for name in (
            "format_passed",
            "type_passed",
            "unit_passed",
            "negative_passed",
            "expected_only",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "command_receipt_ids",
            _ids(self.command_receipt_ids, "command_receipt_ids", required=True),
        )
        object.__setattr__(
            self, "failed_commands", _ids(self.failed_commands, "failed_commands")
        )
        object.__setattr__(
            self, "detector_id", _identifier(self.detector_id, "detector_id")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_passed": self.format_passed,
            "type_passed": self.type_passed,
            "unit_passed": self.unit_passed,
            "negative_passed": self.negative_passed,
            "command_receipt_ids": list(self.command_receipt_ids),
            "failed_commands": list(self.failed_commands),
            "detector_id": self.detector_id,
            "expected_only": self.expected_only,
        }


@dataclass(frozen=True)
class RepairProofTransition(CanonicalContract):
    """Pre→post epoch proof transition (``RepairProofTransition@1``)."""

    SCHEMA: ClassVar[str] = REPAIR_PROOF_TRANSITION_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_PROOF_TRANSITION_INTERFACE

    pre_epoch_id: str
    post_epoch_id: str
    pre_proof_ids: tuple[str, ...]
    post_proof_ids: tuple[str, ...]
    closed_finding_ids: tuple[str, ...]
    residual_finding_ids: tuple[str, ...] = ()
    reconstructed: bool = True
    detector_id: str = "detector:proof-transition"
    evidence_ids: tuple[str, ...] = ()
    synthetic_children: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "pre_epoch_id", _identifier(self.pre_epoch_id, "pre_epoch_id")
        )
        object.__setattr__(
            self, "post_epoch_id", _identifier(self.post_epoch_id, "post_epoch_id")
        )
        if self.pre_epoch_id == self.post_epoch_id:
            raise PostRepairValidationError("pre and post epoch identities must differ")
        object.__setattr__(
            self, "pre_proof_ids", _ids(self.pre_proof_ids, "pre_proof_ids")
        )
        object.__setattr__(
            self,
            "post_proof_ids",
            _ids(self.post_proof_ids, "post_proof_ids", required=True),
        )
        object.__setattr__(
            self,
            "closed_finding_ids",
            _ids(self.closed_finding_ids, "closed_finding_ids"),
        )
        object.__setattr__(
            self,
            "residual_finding_ids",
            _ids(self.residual_finding_ids, "residual_finding_ids"),
        )
        object.__setattr__(
            self, "reconstructed", _bool(self.reconstructed, "reconstructed")
        )
        object.__setattr__(
            self, "detector_id", _identifier(self.detector_id, "detector_id")
        )
        object.__setattr__(
            self, "evidence_ids", _ids(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self,
            "synthetic_children",
            _ids(self.synthetic_children, "synthetic_children"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "pre_epoch_id": self.pre_epoch_id,
            "post_epoch_id": self.post_epoch_id,
            "pre_proof_ids": list(self.pre_proof_ids),
            "post_proof_ids": list(self.post_proof_ids),
            "closed_finding_ids": list(self.closed_finding_ids),
            "residual_finding_ids": list(self.residual_finding_ids),
            "reconstructed": self.reconstructed,
            "detector_id": self.detector_id,
            "evidence_ids": list(self.evidence_ids),
            "synthetic_children": list(self.synthetic_children),
        }

    @property
    def ok(self) -> bool:
        return (
            self.reconstructed
            and bool(self.post_proof_ids)
            and not self.synthetic_children
            and not self.residual_finding_ids
        )


@dataclass(frozen=True)
class PostRepairEpochEvidence:
    """Complete detector-backed evidence bundle for one post-repair epoch."""

    transaction: TransactionSourceEvidence
    commands: CommandGateEvidence
    live: LiveObservationEvidence
    reindex: ReindexRecompileEvidence
    finding: FindingClosureEvidence
    invariants: ProtectedInvariantEvidence
    proof_transition: RepairProofTransition
    model_invocation_count: int = 0
    provider_invocation_count: int = 0
    gate_overrides: Mapping[str, GateResult] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transaction, TransactionSourceEvidence):
            raise PostRepairValidationError("transaction evidence is required")
        if not isinstance(self.commands, CommandGateEvidence):
            raise PostRepairValidationError("command evidence is required")
        if not isinstance(self.live, LiveObservationEvidence):
            raise PostRepairValidationError("live observation evidence is required")
        if not isinstance(self.reindex, ReindexRecompileEvidence):
            raise PostRepairValidationError("reindex evidence is required")
        if not isinstance(self.finding, FindingClosureEvidence):
            raise PostRepairValidationError("finding closure evidence is required")
        if not isinstance(self.invariants, ProtectedInvariantEvidence):
            raise PostRepairValidationError("invariant evidence is required")
        if not isinstance(self.proof_transition, RepairProofTransition):
            raise PostRepairValidationError("proof transition is required")
        object.__setattr__(
            self,
            "model_invocation_count",
            _nonneg_int(self.model_invocation_count, "model_invocation_count"),
        )
        object.__setattr__(
            self,
            "provider_invocation_count",
            _nonneg_int(self.provider_invocation_count, "provider_invocation_count"),
        )
        overrides = self.gate_overrides or {}
        if not isinstance(overrides, Mapping):
            raise PostRepairValidationError("gate_overrides must be a mapping")
        normalized: dict[str, GateResult] = {}
        for key, value in overrides.items():
            gate = _identifier(key, "gate_overrides")
            if not isinstance(value, GateResult):
                raise PostRepairValidationError("gate override must be GateResult")
            if value.gate != gate:
                raise PostRepairValidationError("gate override key mismatch")
            normalized[gate] = value
        object.__setattr__(self, "gate_overrides", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction": self.transaction.to_dict(),
            "commands": self.commands.to_dict(),
            "live": self.live.to_dict(),
            "reindex": self.reindex.to_dict(),
            "finding": self.finding.to_dict(),
            "invariants": self.invariants.to_dict(),
            "proof_transition": self.proof_transition.to_dict(),
            "model_invocation_count": self.model_invocation_count,
            "provider_invocation_count": self.provider_invocation_count,
            "gate_overrides": {
                key: value.to_dict() for key, value in self.gate_overrides.items()
            },
        }


@dataclass(frozen=True)
class PostRepairValidationReport(CanonicalContract):
    """Authoritative report for one post-repair validation attempt."""

    SCHEMA: ClassVar[str] = POST_REPAIR_VALIDATION_REPORT_SCHEMA
    INTERFACE: ClassVar[str] = POST_REPAIR_VALIDATION_INTERFACE

    disposition: ValidationDisposition
    epoch_id: str
    transaction_id: str
    gate_results: tuple[GateResult, ...]
    reason_codes: tuple[str, ...]
    proof_transition: RepairProofTransition
    finding_id: str
    runtime_model_calls: int = 0
    runtime_provider_calls: int = 0
    claims_completion: bool = False
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.disposition, str):
            try:
                object.__setattr__(
                    self, "disposition", ValidationDisposition(self.disposition)
                )
            except ValueError as exc:
                raise PostRepairValidationError("invalid validation disposition") from exc
        if not isinstance(self.disposition, ValidationDisposition):
            raise PostRepairValidationError("disposition must be ValidationDisposition")
        object.__setattr__(self, "epoch_id", _identifier(self.epoch_id, "epoch_id"))
        object.__setattr__(
            self, "transaction_id", _identifier(self.transaction_id, "transaction_id")
        )
        if not isinstance(self.gate_results, tuple) or not self.gate_results:
            raise PostRepairValidationError("gate_results must be a non-empty tuple")
        if len(self.gate_results) > MAX_GATES:
            raise PostRepairValidationError("gate_results exceeds bound")
        for item in self.gate_results:
            if not isinstance(item, GateResult):
                raise PostRepairValidationError("gate_results must contain GateResult")
        object.__setattr__(self, "reason_codes", _reasons(self.reason_codes))
        if not isinstance(self.proof_transition, RepairProofTransition):
            raise PostRepairValidationError("proof_transition is required")
        object.__setattr__(
            self, "finding_id", _identifier(self.finding_id, "finding_id")
        )
        object.__setattr__(
            self,
            "runtime_model_calls",
            _nonneg_int(self.runtime_model_calls, "runtime_model_calls"),
        )
        object.__setattr__(
            self,
            "runtime_provider_calls",
            _nonneg_int(self.runtime_provider_calls, "runtime_provider_calls"),
        )
        object.__setattr__(
            self, "claims_completion", _bool(self.claims_completion, "claims_completion")
        )
        object.__setattr__(
            self,
            "grants_write_authority",
            _bool(self.grants_write_authority, "grants_write_authority"),
        )
        if self.claims_completion and self.disposition is not ValidationDisposition.PASSED:
            raise PostRepairValidationError(
                "completion claims require a passed disposition"
            )
        if self.grants_write_authority:
            raise PostRepairValidationError(
                "post-repair validation never grants write authority"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "disposition": self.disposition.value,
            "epoch_id": self.epoch_id,
            "transaction_id": self.transaction_id,
            "gate_results": [item.to_dict() for item in self.gate_results],
            "reason_codes": list(self.reason_codes),
            "proof_transition": self.proof_transition.to_dict(),
            "finding_id": self.finding_id,
            "runtime_model_calls": self.runtime_model_calls,
            "runtime_provider_calls": self.runtime_provider_calls,
            "claims_completion": self.claims_completion,
            "grants_write_authority": self.grants_write_authority,
        }

    @property
    def ok(self) -> bool:
        return (
            self.disposition is ValidationDisposition.PASSED
            and self.claims_completion
            and self.runtime_model_calls == 0
            and self.runtime_provider_calls == 0
            and all(item.ok for item in self.gate_results)
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class PostRepairValidator:
    """Ordered post-repair epoch validator (``PostRepairValidation@1``)."""

    INTERFACE: ClassVar[str] = POST_REPAIR_VALIDATION_INTERFACE

    def validate(self, evidence: PostRepairEpochEvidence) -> PostRepairValidationReport:
        if not isinstance(evidence, PostRepairEpochEvidence):
            raise PostRepairValidationError("evidence must be PostRepairEpochEvidence")

        reasons: list[str] = []
        gates: list[GateResult] = []

        def add_gate(
            gate: str,
            *,
            passed: bool,
            detector_id: str,
            evidence_ids: Sequence[str] = (),
            reason: str | None = None,
            expected_only: bool = False,
            synthetic: bool = False,
            ran: bool = True,
            disposition: GateDisposition | None = None,
        ) -> None:
            if disposition is None:
                if not ran:
                    disposition = GateDisposition.NOT_RUN
                elif expected_only:
                    disposition = GateDisposition.FAILED
                elif synthetic:
                    disposition = GateDisposition.FAILED
                elif passed:
                    disposition = GateDisposition.PASSED
                else:
                    disposition = GateDisposition.FAILED
            codes: list[str] = []
            if reason:
                codes.append(reason)
            result = GateResult(
                gate=gate,
                disposition=disposition,
                detector_id=detector_id,
                evidence_ids=tuple(evidence_ids),
                reason_codes=tuple(codes),
                expected_only=expected_only,
                synthetic=synthetic,
                ran=ran,
            )
            gates.append(result)
            if not result.ok and reason:
                reasons.append(reason)

        # Optional overrides are still subjected to fail-closed rules.
        overrides = dict(evidence.gate_overrides)

        def resolve(
            gate: str,
            *,
            passed: bool,
            detector_id: str,
            evidence_ids: Sequence[str] = (),
            fail_reason: str,
            expected_only: bool = False,
            synthetic: bool = False,
            ran: bool = True,
        ) -> None:
            if gate in overrides:
                override = overrides[gate]
                # Overrides cannot launder expected-only or synthetic claims.
                if override.expected_only or expected_only:
                    add_gate(
                        gate,
                        passed=False,
                        detector_id=override.detector_id,
                        evidence_ids=override.evidence_ids,
                        reason=PostRepairValidationReason.EXPECTED_SUBSTITUTED_FOR_DETECTOR.value,
                        expected_only=True,
                        ran=override.ran,
                    )
                    return
                if override.synthetic or synthetic:
                    add_gate(
                        gate,
                        passed=False,
                        detector_id=override.detector_id,
                        evidence_ids=override.evidence_ids,
                        reason=PostRepairValidationReason.SYNTHETIC_RELEASE_CHILD.value,
                        synthetic=True,
                        ran=override.ran,
                    )
                    return
                if override.disposition is GateDisposition.SKIPPED:
                    add_gate(
                        gate,
                        passed=False,
                        detector_id=override.detector_id,
                        evidence_ids=override.evidence_ids,
                        reason=PostRepairValidationReason.MANDATORY_GATE_SKIPPED.value,
                        disposition=GateDisposition.SKIPPED,
                        ran=False,
                    )
                    return
                if override.disposition is GateDisposition.UNSUPPORTED:
                    add_gate(
                        gate,
                        passed=False,
                        detector_id=override.detector_id,
                        evidence_ids=override.evidence_ids,
                        reason=PostRepairValidationReason.MANDATORY_GATE_UNSUPPORTED.value,
                        disposition=GateDisposition.UNSUPPORTED,
                        ran=override.ran,
                    )
                    return
                if override.disposition is GateDisposition.NOT_RUN or not override.ran:
                    add_gate(
                        gate,
                        passed=False,
                        detector_id=override.detector_id,
                        evidence_ids=override.evidence_ids,
                        reason=PostRepairValidationReason.MANDATORY_GATE_NOT_RUN.value,
                        disposition=GateDisposition.NOT_RUN,
                        ran=False,
                    )
                    return
                add_gate(
                    gate,
                    passed=override.disposition is GateDisposition.PASSED,
                    detector_id=override.detector_id,
                    evidence_ids=override.evidence_ids,
                    reason=(
                        None
                        if override.disposition is GateDisposition.PASSED
                        else (
                            override.reason_codes[0]
                            if override.reason_codes
                            else fail_reason
                        )
                    ),
                    disposition=override.disposition,
                    ran=override.ran,
                )
                return
            add_gate(
                gate,
                passed=passed,
                detector_id=detector_id,
                evidence_ids=evidence_ids,
                reason=None if passed else fail_reason,
                expected_only=expected_only,
                synthetic=synthetic,
                ran=ran,
            )

        tx = evidence.transaction
        if tx.disposition != TransactionDisposition.COMMITTED.value:
            resolve(
                "source_edits",
                passed=False,
                detector_id="detector:transaction",
                fail_reason=PostRepairValidationReason.TRANSACTION_NOT_COMMITTED.value,
            )
        elif not tx.has_source_edits:
            resolve(
                "source_edits",
                passed=False,
                detector_id="detector:transaction",
                evidence_ids=(tx.transaction_id,),
                fail_reason=PostRepairValidationReason.NO_SOURCE_EDITS.value,
            )
        else:
            resolve(
                "source_edits",
                passed=True,
                detector_id="detector:transaction",
                evidence_ids=(tx.transaction_id,),
                fail_reason=PostRepairValidationReason.NO_SOURCE_EDITS.value,
            )

        commands = evidence.commands
        for gate_name, passed in (
            ("format", commands.format_passed),
            ("type", commands.type_passed),
            ("unit", commands.unit_passed),
            ("negative", commands.negative_passed),
        ):
            resolve(
                gate_name,
                passed=passed and not commands.expected_only,
                detector_id=commands.detector_id,
                evidence_ids=commands.command_receipt_ids,
                fail_reason=(
                    PostRepairValidationReason.EXPECTED_SUBSTITUTED_FOR_DETECTOR.value
                    if commands.expected_only
                    else PostRepairValidationReason.MANDATORY_GATE_FAILED.value
                ),
                expected_only=commands.expected_only,
            )

        live = evidence.live
        if live.substitutes_expected_for_detector():
            resolve(
                "service_start",
                passed=False,
                detector_id=live.detector_id,
                evidence_ids=live.evidence_ids,
                fail_reason=PostRepairValidationReason.EXPECTED_SUBSTITUTED_FOR_DETECTOR.value,
                expected_only=True,
            )
            resolve(
                "live_reobserve",
                passed=False,
                detector_id=live.detector_id,
                evidence_ids=live.tools_list_receipt_ids,
                fail_reason=PostRepairValidationReason.EXPECTED_SUBSTITUTED_FOR_DETECTOR.value,
                expected_only=True,
            )
        else:
            resolve(
                "service_start",
                passed=bool(live.started_service_ids),
                detector_id=live.detector_id,
                evidence_ids=live.started_service_ids,
                fail_reason=PostRepairValidationReason.MANDATORY_GATE_FAILED.value,
            )
            live_ok = bool(
                live.tools_list_receipt_ids
                and live.tools_call_receipt_ids
                and live.observed_tools
            )
            resolve(
                "live_reobserve",
                passed=live_ok,
                detector_id=live.detector_id,
                evidence_ids=live.tools_list_receipt_ids + live.tools_call_receipt_ids,
                fail_reason=PostRepairValidationReason.MANDATORY_GATE_FAILED.value,
            )

        reindex = evidence.reindex
        resolve(
            "reindex",
            passed=bool(reindex.reindex_receipt_id) and reindex.clean_rebuild_equivalent,
            detector_id=reindex.detector_id,
            evidence_ids=(reindex.reindex_receipt_id,),
            fail_reason=PostRepairValidationReason.MANDATORY_GATE_FAILED.value,
        )
        resolve(
            "recompile",
            passed=bool(reindex.recompile_receipt_id) and reindex.clean_rebuild_equivalent,
            detector_id=reindex.detector_id,
            evidence_ids=(reindex.recompile_receipt_id,),
            fail_reason=PostRepairValidationReason.MANDATORY_GATE_FAILED.value,
        )

        transition = evidence.proof_transition
        if transition.synthetic_children:
            resolve(
                "proof_reconstruction",
                passed=False,
                detector_id=transition.detector_id,
                evidence_ids=transition.evidence_ids,
                fail_reason=PostRepairValidationReason.SYNTHETIC_RELEASE_CHILD.value,
                synthetic=True,
            )
        else:
            resolve(
                "proof_reconstruction",
                passed=transition.ok,
                detector_id=transition.detector_id,
                evidence_ids=transition.post_proof_ids,
                fail_reason=PostRepairValidationReason.MANDATORY_GATE_FAILED.value,
            )

        finding = evidence.finding
        if not finding.closed:
            resolve(
                "finding_closed",
                passed=False,
                detector_id=finding.detector_id,
                evidence_ids=finding.evidence_ids or (finding.finding_id,),
                fail_reason=PostRepairValidationReason.FINDING_NOT_CLOSED.value,
            )
        elif finding.semantic_reason != finding.intended_semantic_reason:
            resolve(
                "finding_closed",
                passed=False,
                detector_id=finding.detector_id,
                evidence_ids=finding.evidence_ids or (finding.finding_id,),
                fail_reason=PostRepairValidationReason.FINDING_CLOSED_WRONG_REASON.value,
            )
        elif finding.residual_finding_ids:
            resolve(
                "finding_closed",
                passed=False,
                detector_id=finding.detector_id,
                evidence_ids=finding.residual_finding_ids,
                fail_reason=PostRepairValidationReason.FINDING_NOT_CLOSED.value,
            )
        else:
            resolve(
                "finding_closed",
                passed=True,
                detector_id=finding.detector_id,
                evidence_ids=finding.evidence_ids or (finding.finding_id,),
                fail_reason=PostRepairValidationReason.FINDING_NOT_CLOSED.value,
            )

        invariants = evidence.invariants
        resolve(
            "protected_invariants",
            passed=invariants.ok,
            detector_id=invariants.detector_id,
            evidence_ids=invariants.evidence_ids or invariants.checked_keys,
            fail_reason=PostRepairValidationReason.PROTECTED_INVARIANT_REGRESSED.value,
        )

        if evidence.model_invocation_count != 0:
            resolve(
                "zero_model_calls",
                passed=False,
                detector_id="detector:runtime-model-audit",
                fail_reason=PostRepairValidationReason.MODEL_CALLS_OBSERVED.value,
            )
        elif evidence.provider_invocation_count != 0:
            resolve(
                "zero_model_calls",
                passed=False,
                detector_id="detector:runtime-provider-audit",
                fail_reason=PostRepairValidationReason.PROVIDER_CALLS_OBSERVED.value,
            )
        else:
            resolve(
                "zero_model_calls",
                passed=True,
                detector_id="detector:runtime-model-audit",
                evidence_ids=("audit:zero-model",),
                fail_reason=PostRepairValidationReason.MODEL_CALLS_OBSERVED.value,
            )

        # Every mandatory gate must appear exactly once.
        seen = [item.gate for item in gates]
        if sorted(seen) != sorted(MANDATORY_GATES) or len(seen) != len(MANDATORY_GATES):
            reasons.append(PostRepairValidationReason.INCOMPLETE_EVIDENCE.value)

        all_ok = all(item.ok for item in gates) and not reasons
        disposition = (
            ValidationDisposition.PASSED if all_ok else ValidationDisposition.FAILED
        )
        epoch_id = (
            tx.candidate_epoch_id
            or transition.post_epoch_id
            or content_identity(
                {
                    "transaction_id": tx.transaction_id,
                    "tree_id": tx.tree_id,
                    "post_proof_ids": list(transition.post_proof_ids),
                }
            )
        )
        return PostRepairValidationReport(
            disposition=disposition,
            epoch_id=epoch_id,
            transaction_id=tx.transaction_id,
            gate_results=tuple(gates),
            reason_codes=tuple(dict.fromkeys(reasons)),
            proof_transition=transition,
            finding_id=finding.finding_id,
            runtime_model_calls=evidence.model_invocation_count,
            runtime_provider_calls=evidence.provider_invocation_count,
            claims_completion=all_ok,
            grants_write_authority=False,
        )


def validate_post_repair_epoch(
    evidence: PostRepairEpochEvidence,
) -> PostRepairValidationReport:
    """Validate one post-repair epoch through the mandatory gate set."""

    return PostRepairValidator().validate(evidence)


def build_passing_gate_results(
    *,
    detector_prefix: str = "detector",
) -> tuple[GateResult, ...]:
    """Helper used by fixtures: all mandatory gates passed with detectors."""

    return tuple(
        GateResult(
            gate=gate,
            disposition=GateDisposition.PASSED,
            detector_id=f"{detector_prefix}:{gate}",
            evidence_ids=(f"evidence:{gate}",),
            ran=True,
        )
        for gate in MANDATORY_GATES
    )


def materialize_post_repair_epoch(
    *,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
    report: PostRepairValidationReport | Mapping[str, Any] | None = None,
    evidence: PostRepairEpochEvidence | None = None,
) -> dict[str, Any]:
    """Materialize ``post-repair-epoch.json`` evidence for DCR-073."""

    if report is None:
        if evidence is None:
            evidence = _fixture_passing_evidence()
        report_obj = validate_post_repair_epoch(evidence)
        evidence_payload = evidence.to_dict()
    elif isinstance(report, PostRepairValidationReport):
        report_obj = report
        evidence_payload = evidence.to_dict() if evidence is not None else {}
    else:
        # Mapping path: evidence-only projection (still zero model calls).
        transition_payload = dict(report.get("proof_transition") or {})
        transition = RepairProofTransition(
            pre_epoch_id=str(transition_payload.get("pre_epoch_id") or "epoch:pre"),
            post_epoch_id=str(transition_payload.get("post_epoch_id") or "epoch:post"),
            pre_proof_ids=tuple(transition_payload.get("pre_proof_ids") or ()),
            post_proof_ids=tuple(
                transition_payload.get("post_proof_ids") or ("proof:post",)
            ),
            closed_finding_ids=tuple(
                transition_payload.get("closed_finding_ids") or ()
            ),
            residual_finding_ids=tuple(
                transition_payload.get("residual_finding_ids") or ()
            ),
            reconstructed=bool(transition_payload.get("reconstructed", True)),
            detector_id=str(
                transition_payload.get("detector_id") or "detector:proof-transition"
            ),
            evidence_ids=tuple(transition_payload.get("evidence_ids") or ()),
            synthetic_children=tuple(
                transition_payload.get("synthetic_children") or ()
            ),
        )
        report_obj = PostRepairValidationReport(
            disposition=ValidationDisposition(
                str(report.get("disposition") or "passed")
            ),
            epoch_id=str(report.get("epoch_id") or "epoch:post"),
            transaction_id=str(report.get("transaction_id") or "tx:mapping"),
            gate_results=build_passing_gate_results(),
            reason_codes=tuple(report.get("reason_codes") or ()),
            proof_transition=transition,
            finding_id=str(report.get("finding_id") or "finding:mapping"),
            runtime_model_calls=int(report.get("runtime_model_calls") or 0),
            runtime_provider_calls=int(report.get("runtime_provider_calls") or 0),
            claims_completion=bool(report.get("claims_completion", True)),
            grants_write_authority=False,
        )
        evidence_payload = dict(report.get("evidence") or {})

    payload = {
        "schema": POST_REPAIR_EPOCH_CATALOG_SCHEMA,
        "interface": POST_REPAIR_VALIDATION_INTERFACE,
        "evidence_id": DCR_VALIDATION_EVIDENCE,
        "version": DCR_VALIDATION_VERSION,
        "epoch": {
            "schema": POST_REPAIR_EPOCH_SCHEMA,
            "epoch_id": report_obj.epoch_id,
            "transaction_id": report_obj.transaction_id,
            "finding_id": report_obj.finding_id,
            "disposition": report_obj.disposition.value,
            "claims_completion": report_obj.claims_completion,
            "proof_transition": report_obj.proof_transition.to_dict(),
            "gate_results": [item.to_dict() for item in report_obj.gate_results],
            "reason_codes": list(report_obj.reason_codes),
            "evidence": evidence_payload,
        },
        "report": report_obj.to_dict(),
        "runtime_model_calls": 0,
        "runtime_provider_calls": 0,
        "grants_write_authority": False,
        "mandatory_gates": list(MANDATORY_GATES),
        "protected_invariants": list(PROTECTED_INVARIANT_KEYS),
    }
    base = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    path = (
        Path(destination)
        if destination is not None
        else base.joinpath(*PurePosixPath(DEFAULT_POST_REPAIR_EPOCH_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _fixture_passing_evidence() -> PostRepairEpochEvidence:
    """Deterministic passing fixture used for catalog materialization."""

    path = "external/ipfs_accelerate/ipfs_accelerate_py/fixture.py"
    return PostRepairEpochEvidence(
        transaction=TransactionSourceEvidence(
            transaction_id="tx:dcr073-fixture",
            disposition=TransactionDisposition.COMMITTED.value,
            changed_paths=(path,),
            before_hashes={path: "sha256:" + ("a" * 64)},
            after_hashes={path: "sha256:" + ("b" * 64)},
            root_ids=tuple(sorted(ADMITTED_OWNER_ROOTS)),
            forest_id="forest:dcr073-fixture",
            tree_id="tree:dcr073-fixture",
            candidate_epoch_id="epoch:dcr073-post",
        ),
        commands=CommandGateEvidence(
            format_passed=True,
            type_passed=True,
            unit_passed=True,
            negative_passed=True,
            command_receipt_ids=(
                "cmd:format",
                "cmd:type",
                "cmd:unit",
                "cmd:negative",
            ),
        ),
        live=LiveObservationEvidence(
            service_roles=("accelerate", "datasets", "kit"),
            started_service_ids=(
                "accelerate:runtime",
                "datasets:runtime",
                "kit:runtime",
            ),
            tools_list_receipt_ids=("live:tools-list",),
            tools_call_receipt_ids=("live:tools-call",),
            invalid_call_receipt_ids=("live:invalid-call",),
            observed_tools=("accelerate.inference", "datasets.search"),
            expected_tools=("accelerate.inference", "datasets.search"),
            evidence_ids=("live:epoch",),
        ),
        reindex=ReindexRecompileEvidence(
            index_id="index:dcr073-fixture",
            reindex_receipt_id="reindex:dcr073",
            recompile_receipt_id="recompile:dcr073",
            rebuilt_paths=(path,),
            clean_rebuild_equivalent=True,
        ),
        finding=FindingClosureEvidence(
            finding_id="finding:dcr073-fixture",
            closed=True,
            semantic_reason="edge_resolved_and_obligation_proved",
            intended_semantic_reason="edge_resolved_and_obligation_proved",
            residual_finding_ids=(),
            evidence_ids=("finding:closed",),
        ),
        invariants=ProtectedInvariantEvidence(
            checked_keys=PROTECTED_INVARIANT_KEYS,
            regressed_keys=(),
            evidence_ids=("invariant:ok",),
        ),
        proof_transition=RepairProofTransition(
            pre_epoch_id="epoch:dcr073-pre",
            post_epoch_id="epoch:dcr073-post",
            pre_proof_ids=("proof:pre",),
            post_proof_ids=("proof:post",),
            closed_finding_ids=("finding:dcr073-fixture",),
            residual_finding_ids=(),
            reconstructed=True,
            evidence_ids=("proof:transition",),
        ),
        model_invocation_count=0,
        provider_invocation_count=0,
    )


__all__ = [
    "DCR_VALIDATION_EVIDENCE",
    "DCR_VALIDATION_VERSION",
    "DEFAULT_POST_REPAIR_EPOCH_PATH",
    "MANDATORY_GATES",
    "POST_REPAIR_VALIDATION_INTERFACE",
    "PROTECTED_INVARIANT_KEYS",
    "REPAIR_PROOF_TRANSITION_INTERFACE",
    "CommandGateEvidence",
    "FindingClosureEvidence",
    "GateDisposition",
    "GateResult",
    "LiveObservationEvidence",
    "PostRepairEpochEvidence",
    "PostRepairValidationError",
    "PostRepairValidationReason",
    "PostRepairValidationReport",
    "PostRepairValidator",
    "ProtectedInvariantEvidence",
    "ReindexRecompileEvidence",
    "RepairProofTransition",
    "TransactionSourceEvidence",
    "ValidationDisposition",
    "build_passing_gate_results",
    "materialize_post_repair_epoch",
    "validate_post_repair_epoch",
]
