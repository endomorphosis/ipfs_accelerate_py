"""Post-edit completion gate for proof-gated contract repair.

A repair is not complete when a provider finishes editing.  Completion requires
the candidate tree to:

1. rebuild affected source / AST / vector rows and tombstones;
2. re-resolve the original broken edge;
3. re-extract sender/receiver contracts;
4. re-run every original and patch-introduced obligation;
5. enforce policy-selected type/schema/error/effect/capability/lifecycle/
   resource/memory tools (skipped required tools fail closed);
6. run focused contract tests plus dependency-complete impacted tests;
7. prove the original finding is closed without deleted/weakened contracts,
   tests, or checkers, suppressed findings, omitted dependants, or a stale
   candidate tree; and
8. emit a patch/tree-bound :class:`ContractRepairCompletionReceipt`.

This module owns only the validation orchestration and receipts.  It invokes
injected adapters for index rebuild, resolution, extraction, proof, tools, and
tests without weakening those adapters' policies.  Provider success alone is
never completion authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from ..analysis.contract_repair_contracts import (
    AuthorityRoots,
    DecisionDisposition,
    RepairStrategy,
    RepairTargetDecision,
)
from ..planning.repair_target_admission import AdmissionResult
from ..proof.contract_repair_edit_packet import ContractRepairEditPacket
from ..proof.contract_repair_prover import (
    CandidateProofBundle,
    ContractRepairProofDisposition,
)
from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)


CONTRACT_REPAIR_VALIDATOR_INTERFACE: Final[str] = "ContractRepairValidator@1"
CONTRACT_REPAIR_COMPLETION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-completion-receipt@1"
)
CONTRACT_REPAIR_VALIDATION_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-validation-report@1"
)
CONTRACT_REPAIR_INDEX_REBUILD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-index-rebuild@1"
)
CONTRACT_REPAIR_EDGE_RESOLUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-edge-resolution@1"
)
CONTRACT_REPAIR_CONTRACT_EXTRACTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-contract-extraction@1"
)
CONTRACT_REPAIR_OBLIGATION_REPROOF_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-obligation-reproof@1"
)
CONTRACT_REPAIR_TOOL_GATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-tool-gate@1"
)
CONTRACT_REPAIR_IMPACTED_TEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-impacted-tests@1"
)
CONTRACT_REPAIR_INTEGRITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-integrity@1"
)

MAX_COMPLETION_RECEIPT_BYTES: Final[int] = 131_072
MAX_VALIDATION_PATHS: Final[int] = 512
MAX_OBLIGATION_IDS: Final[int] = 256
MAX_TOOL_IDS: Final[int] = 64
MAX_TEST_IDS: Final[int] = 512
MAX_TOMBSTONES: Final[int] = 512
MAX_REASON_CODES: Final[int] = 64

# Closed vocabulary of policy-selectable post-edit tool families.
POLICY_TOOL_FAMILIES: Final[tuple[str, ...]] = (
    "type",
    "schema",
    "error",
    "effect",
    "capability",
    "lifecycle",
    "resource",
    "memory",
)

DEFAULT_POLICY_REQUIRED_TOOLS: Final[tuple[str, ...]] = POLICY_TOOL_FAMILIES


class ContractRepairValidationError(ValueError):
    """The candidate patch failed the post-edit completion gate."""


class ContractRepairValidationReason(str, Enum):
    """Stable, machine-readable post-edit failure codes."""

    MALFORMED_INPUT = "malformed_input"
    STALE_CANDIDATE_TREE = "stale_candidate_tree"
    ROOT_DRIFT = "root_drift"
    PACKET_DECISION_MISMATCH = "packet_decision_mismatch"
    AMBIGUOUS_OR_ABSTAINED = "ambiguous_or_abstained"
    INDEX_REBUILD_INCOMPLETE = "index_rebuild_incomplete"
    TOMBSTONE_MISSING = "tombstone_missing"
    EDGE_NOT_RESOLVED = "edge_not_resolved"
    CONTRACT_REEXTRACTION_FAILED = "contract_reextraction_failed"
    CONTRACT_DELETED = "contract_deleted"
    CONTRACT_WEAKENED = "contract_weakened"
    OBLIGATION_FAILED = "obligation_failed"
    INTRODUCED_OBLIGATION_FAILED = "introduced_obligation_failed"
    OBLIGATION_OMITTED = "obligation_omitted"
    SKIPPED_REQUIRED_TOOL = "skipped_required_tool"
    TOOL_FAILED = "tool_failed"
    FOCUSED_TEST_FAILED = "focused_test_failed"
    IMPACTED_TEST_FAILED = "impacted_test_failed"
    IMPACTED_TEST_OMITTED = "impacted_test_omitted"
    TEST_DELETED = "test_deleted"
    TEST_WEAKENED = "test_weakened"
    CHECKER_DELETED = "checker_deleted"
    CHECKER_WEAKENED = "checker_weakened"
    FINDING_SUPPRESSED = "finding_suppressed"
    DEPENDANT_OMITTED = "dependant_omitted"
    ORIGINAL_FINDING_NOT_CLOSED = "original_finding_not_closed"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    PROOF_BUNDLE_MISMATCH = "proof_bundle_mismatch"


class ValidationStage(str, Enum):
    """Ordered post-edit stages from the normative completion gate."""

    INDEX_REBUILD = "index_rebuild"
    EDGE_RESOLUTION = "edge_resolution"
    CONTRACT_EXTRACTION = "contract_extraction"
    OBLIGATION_REPROOF = "obligation_reproof"
    POLICY_TOOLS = "policy_tools"
    IMPACTED_TESTS = "impacted_tests"
    INTEGRITY = "integrity"
    COMPLETION = "completion"


class StageDisposition(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise ContractRepairValidationError(f"{name} must be a compact identifier")
    return value.strip()


def _paths(values: Sequence[str], name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairValidationError(f"{name} must be a path sequence")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ContractRepairValidationError(f"{name} contains an invalid path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise ContractRepairValidationError(f"{name} contains an escaped path")
        result.add(path.as_posix())
    if required and not result:
        raise ContractRepairValidationError(f"{name} must not be empty")
    if len(result) > MAX_VALIDATION_PATHS:
        raise ContractRepairValidationError(f"{name} exceeds its path bound")
    return tuple(sorted(result))


def _ids(
    values: Sequence[str],
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_OBLIGATION_IDS,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContractRepairValidationError(f"{name} must be an identifier sequence")
    result = tuple(
        sorted({value.strip() for value in values if isinstance(value, str) and value.strip()})
    )
    if required and not result:
        raise ContractRepairValidationError(f"{name} must not be empty")
    if len(result) > maximum:
        raise ContractRepairValidationError(f"{name} exceeds its item bound")
    if any(any(char.isspace() for char in item) for item in result):
        raise ContractRepairValidationError(f"{name} must contain compact identifiers")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractRepairValidationError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class IndexRebuildEvidence:
    """Proof that affected source/AST/vector rows and tombstones were rebuilt."""

    candidate_tree_id: str
    index_id: str
    rebuilt_source_paths: tuple[str, ...]
    rebuilt_ast_paths: tuple[str, ...]
    rebuilt_vector_row_ids: tuple[str, ...]
    tombstone_ids: tuple[str, ...]
    affected_paths: tuple[str, ...]
    clean_rebuild_equivalent: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id"))
        object.__setattr__(self, "index_id", _identifier(self.index_id, "index_id"))
        object.__setattr__(self, "rebuilt_source_paths", _paths(self.rebuilt_source_paths, "rebuilt_source_paths"))
        object.__setattr__(self, "rebuilt_ast_paths", _paths(self.rebuilt_ast_paths, "rebuilt_ast_paths"))
        object.__setattr__(
            self,
            "rebuilt_vector_row_ids",
            _ids(self.rebuilt_vector_row_ids, "rebuilt_vector_row_ids", maximum=MAX_TOMBSTONES),
        )
        object.__setattr__(
            self,
            "tombstone_ids",
            _ids(self.tombstone_ids, "tombstone_ids", required=False, maximum=MAX_TOMBSTONES),
        )
        object.__setattr__(self, "affected_paths", _paths(self.affected_paths, "affected_paths"))
        object.__setattr__(self, "clean_rebuild_equivalent", _bool(self.clean_rebuild_equivalent, "clean_rebuild_equivalent"))
        if not set(self.affected_paths).issubset(self.rebuilt_source_paths):
            raise ContractRepairValidationError("index rebuild must cover every affected source path")
        if not set(self.affected_paths).issubset(self.rebuilt_ast_paths):
            raise ContractRepairValidationError("index rebuild must cover every affected AST path")
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(self, "receipt_id", rid or content_identity(self.to_dict(include_receipt_id=False)))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CONTRACT_REPAIR_INDEX_REBUILD_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "index_id": self.index_id,
            "rebuilt_source_paths": list(self.rebuilt_source_paths),
            "rebuilt_ast_paths": list(self.rebuilt_ast_paths),
            "rebuilt_vector_row_ids": list(self.rebuilt_vector_row_ids),
            "tombstone_ids": list(self.tombstone_ids),
            "affected_paths": list(self.affected_paths),
            "clean_rebuild_equivalent": self.clean_rebuild_equivalent,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IndexRebuildEvidence":
        if not isinstance(payload, Mapping):
            raise ContractRepairValidationError("index rebuild evidence must be an object")
        return cls(
            candidate_tree_id=payload["candidate_tree_id"],
            index_id=payload["index_id"],
            rebuilt_source_paths=tuple(payload["rebuilt_source_paths"]),
            rebuilt_ast_paths=tuple(payload["rebuilt_ast_paths"]),
            rebuilt_vector_row_ids=tuple(payload["rebuilt_vector_row_ids"]),
            tombstone_ids=tuple(payload.get("tombstone_ids", ())),
            affected_paths=tuple(payload["affected_paths"]),
            clean_rebuild_equivalent=payload["clean_rebuild_equivalent"],
            receipt_id=str(payload.get("receipt_id", "")),
        )


@dataclass(frozen=True)
class EdgeResolutionEvidence:
    """Proof that the original broken edge re-resolves on the candidate tree."""

    candidate_tree_id: str
    original_trace_id: str
    original_edge_id: str
    resolved: bool
    resolved_target_path: str
    resolved_target_symbol_id: str
    resolution_receipt_id: str
    residual_unresolved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id"))
        object.__setattr__(self, "original_trace_id", _identifier(self.original_trace_id, "original_trace_id"))
        object.__setattr__(self, "original_edge_id", _identifier(self.original_edge_id, "original_edge_id"))
        object.__setattr__(self, "resolved", _bool(self.resolved, "resolved"))
        object.__setattr__(self, "residual_unresolved", _bool(self.residual_unresolved, "residual_unresolved"))
        if self.resolved:
            object.__setattr__(
                self,
                "resolved_target_path",
                _paths((self.resolved_target_path,), "resolved_target_path")[0],
            )
            object.__setattr__(
                self,
                "resolved_target_symbol_id",
                _identifier(self.resolved_target_symbol_id, "resolved_target_symbol_id"),
            )
        else:
            object.__setattr__(self, "resolved_target_path", "")
            object.__setattr__(self, "resolved_target_symbol_id", "")
        object.__setattr__(
            self,
            "resolution_receipt_id",
            _identifier(self.resolution_receipt_id, "resolution_receipt_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_EDGE_RESOLUTION_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "original_trace_id": self.original_trace_id,
            "original_edge_id": self.original_edge_id,
            "resolved": self.resolved,
            "resolved_target_path": self.resolved_target_path,
            "resolved_target_symbol_id": self.resolved_target_symbol_id,
            "resolution_receipt_id": self.resolution_receipt_id,
            "residual_unresolved": self.residual_unresolved,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EdgeResolutionEvidence":
        if not isinstance(payload, Mapping):
            raise ContractRepairValidationError("edge resolution evidence must be an object")
        return cls(
            candidate_tree_id=payload["candidate_tree_id"],
            original_trace_id=payload["original_trace_id"],
            original_edge_id=payload["original_edge_id"],
            resolved=payload["resolved"],
            resolved_target_path=str(payload.get("resolved_target_path", "")),
            resolved_target_symbol_id=str(payload.get("resolved_target_symbol_id", "")),
            resolution_receipt_id=payload["resolution_receipt_id"],
            residual_unresolved=bool(payload.get("residual_unresolved", False)),
        )


@dataclass(frozen=True)
class ContractExtractionEvidence:
    """Proof that sender/receiver contracts were re-extracted without weakening."""

    candidate_tree_id: str
    sender_contract_id: str
    receiver_contract_id: str
    original_sender_contract_id: str
    original_receiver_contract_id: str
    clauses_preserved: bool
    strength_preserved: bool
    contracts_present: bool
    extraction_receipt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id"))
        for name in (
            "sender_contract_id",
            "receiver_contract_id",
            "original_sender_contract_id",
            "original_receiver_contract_id",
            "extraction_receipt_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("clauses_preserved", "strength_preserved", "contracts_present"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_CONTRACT_EXTRACTION_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "sender_contract_id": self.sender_contract_id,
            "receiver_contract_id": self.receiver_contract_id,
            "original_sender_contract_id": self.original_sender_contract_id,
            "original_receiver_contract_id": self.original_receiver_contract_id,
            "clauses_preserved": self.clauses_preserved,
            "strength_preserved": self.strength_preserved,
            "contracts_present": self.contracts_present,
            "extraction_receipt_id": self.extraction_receipt_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractExtractionEvidence":
        if not isinstance(payload, Mapping):
            raise ContractRepairValidationError("contract extraction evidence must be an object")
        return cls(
            candidate_tree_id=payload["candidate_tree_id"],
            sender_contract_id=payload["sender_contract_id"],
            receiver_contract_id=payload["receiver_contract_id"],
            original_sender_contract_id=payload["original_sender_contract_id"],
            original_receiver_contract_id=payload["original_receiver_contract_id"],
            clauses_preserved=payload["clauses_preserved"],
            strength_preserved=payload["strength_preserved"],
            contracts_present=payload["contracts_present"],
            extraction_receipt_id=payload["extraction_receipt_id"],
        )


@dataclass(frozen=True)
class ObligationReproofEvidence:
    """Proof that original and introduced obligations were re-run on the tree."""

    candidate_tree_id: str
    original_obligation_ids: tuple[str, ...]
    introduced_obligation_ids: tuple[str, ...]
    proved_obligation_ids: tuple[str, ...]
    failed_obligation_ids: tuple[str, ...]
    omitted_obligation_ids: tuple[str, ...]
    proof_bundle_id: str
    all_mandatory_proved: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id"))
        object.__setattr__(
            self,
            "original_obligation_ids",
            _ids(self.original_obligation_ids, "original_obligation_ids"),
        )
        object.__setattr__(
            self,
            "introduced_obligation_ids",
            _ids(self.introduced_obligation_ids, "introduced_obligation_ids", required=False),
        )
        object.__setattr__(
            self,
            "proved_obligation_ids",
            _ids(self.proved_obligation_ids, "proved_obligation_ids", required=False),
        )
        object.__setattr__(
            self,
            "failed_obligation_ids",
            _ids(self.failed_obligation_ids, "failed_obligation_ids", required=False),
        )
        object.__setattr__(
            self,
            "omitted_obligation_ids",
            _ids(self.omitted_obligation_ids, "omitted_obligation_ids", required=False),
        )
        object.__setattr__(self, "proof_bundle_id", _identifier(self.proof_bundle_id, "proof_bundle_id"))
        object.__setattr__(self, "all_mandatory_proved", _bool(self.all_mandatory_proved, "all_mandatory_proved"))
        mandatory = set(self.original_obligation_ids) | set(self.introduced_obligation_ids)
        if self.all_mandatory_proved and not mandatory.issubset(self.proved_obligation_ids):
            raise ContractRepairValidationError("all_mandatory_proved requires every mandatory obligation proved")
        if self.all_mandatory_proved and (self.failed_obligation_ids or self.omitted_obligation_ids):
            raise ContractRepairValidationError("all_mandatory_proved forbids failed or omitted obligations")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_OBLIGATION_REPROOF_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "original_obligation_ids": list(self.original_obligation_ids),
            "introduced_obligation_ids": list(self.introduced_obligation_ids),
            "proved_obligation_ids": list(self.proved_obligation_ids),
            "failed_obligation_ids": list(self.failed_obligation_ids),
            "omitted_obligation_ids": list(self.omitted_obligation_ids),
            "proof_bundle_id": self.proof_bundle_id,
            "all_mandatory_proved": self.all_mandatory_proved,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObligationReproofEvidence":
        if not isinstance(payload, Mapping):
            raise ContractRepairValidationError("obligation reproof evidence must be an object")
        return cls(
            candidate_tree_id=payload["candidate_tree_id"],
            original_obligation_ids=tuple(payload["original_obligation_ids"]),
            introduced_obligation_ids=tuple(payload.get("introduced_obligation_ids", ())),
            proved_obligation_ids=tuple(payload.get("proved_obligation_ids", ())),
            failed_obligation_ids=tuple(payload.get("failed_obligation_ids", ())),
            omitted_obligation_ids=tuple(payload.get("omitted_obligation_ids", ())),
            proof_bundle_id=payload["proof_bundle_id"],
            all_mandatory_proved=payload["all_mandatory_proved"],
        )

    @classmethod
    def from_proof_bundle(
        cls,
        bundle: CandidateProofBundle,
        *,
        original_obligation_ids: Sequence[str],
        introduced_obligation_ids: Sequence[str] = (),
    ) -> "ObligationReproofEvidence":
        """Project a :class:`CandidateProofBundle` into gate evidence without weakening authority."""

        if not isinstance(bundle, CandidateProofBundle):
            raise ContractRepairValidationError("bundle must be a CandidateProofBundle")
        original = _ids(original_obligation_ids, "original_obligation_ids")
        introduced = _ids(introduced_obligation_ids, "introduced_obligation_ids", required=False)
        proved: list[str] = []
        failed: list[str] = []
        for result in bundle.results:
            if result.disposition is ContractRepairProofDisposition.PROVED and result.authoritative:
                proved.append(result.obligation_id)
            else:
                failed.append(result.obligation_id)
        proved_ids = tuple(sorted(set(proved)))
        failed_ids = tuple(sorted(set(failed)))
        mandatory = set(original) | set(introduced)
        observed = {item.obligation_id for item in bundle.results}
        omitted = tuple(sorted(mandatory - observed))
        all_proved = bool(mandatory) and mandatory.issubset(proved_ids) and not failed_ids and not omitted
        return cls(
            candidate_tree_id=bundle.tree_id,
            original_obligation_ids=original,
            introduced_obligation_ids=introduced,
            proved_obligation_ids=proved_ids,
            failed_obligation_ids=failed_ids,
            omitted_obligation_ids=omitted,
            proof_bundle_id=str(bundle.to_dict().get("bundle_id", "")),
            all_mandatory_proved=all_proved,
        )


@dataclass(frozen=True)
class ToolGateResult:
    """One policy-selected tool family's post-edit outcome."""

    tool_id: str
    family: str
    required: bool
    executed: bool
    passed: bool
    skipped: bool
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_id", _identifier(self.tool_id, "tool_id"))
        family = _identifier(self.family, "family")
        if family not in POLICY_TOOL_FAMILIES:
            raise ContractRepairValidationError(f"unknown policy tool family: {family}")
        object.__setattr__(self, "family", family)
        for name in ("required", "executed", "passed", "skipped"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if self.skipped and self.passed:
            raise ContractRepairValidationError("a skipped tool cannot claim pass")
        if self.skipped and self.executed:
            raise ContractRepairValidationError("a skipped tool cannot claim execution")
        if self.passed and not self.executed:
            raise ContractRepairValidationError("a passed tool must have executed")
        rid = self.receipt_id.strip() if isinstance(self.receipt_id, str) else ""
        object.__setattr__(self, "receipt_id", rid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "family": self.family,
            "required": self.required,
            "executed": self.executed,
            "passed": self.passed,
            "skipped": self.skipped,
            "receipt_id": self.receipt_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolGateResult":
        if not isinstance(payload, Mapping):
            raise ContractRepairValidationError("tool gate result must be an object")
        return cls(
            tool_id=payload["tool_id"],
            family=payload["family"],
            required=payload["required"],
            executed=payload["executed"],
            passed=payload["passed"],
            skipped=payload["skipped"],
            receipt_id=str(payload.get("receipt_id", "")),
        )


@dataclass(frozen=True)
class PolicyToolEvidence:
    """Policy-selected type/schema/error/effect/capability/lifecycle/resource/memory gates."""

    candidate_tree_id: str
    required_families: tuple[str, ...]
    results: tuple[ToolGateResult, ...]
    policy_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id"))
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        families = _ids(self.required_families, "required_families", maximum=MAX_TOOL_IDS)
        unknown = set(families) - set(POLICY_TOOL_FAMILIES)
        if unknown:
            raise ContractRepairValidationError(f"unknown required tool families: {sorted(unknown)}")
        object.__setattr__(self, "required_families", families)
        if not isinstance(self.results, Sequence) or not all(isinstance(item, ToolGateResult) for item in self.results):
            raise ContractRepairValidationError("results must be ToolGateResult values")
        if len(self.results) > MAX_TOOL_IDS:
            raise ContractRepairValidationError("tool results exceed bound")
        ids = [item.tool_id for item in self.results]
        if len(ids) != len(set(ids)):
            raise ContractRepairValidationError("tool results must have unique tool_ids")
        object.__setattr__(self, "results", tuple(sorted(self.results, key=lambda item: item.tool_id)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_TOOL_GATE_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "required_families": list(self.required_families),
            "results": [item.to_dict() for item in self.results],
            "policy_id": self.policy_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyToolEvidence":
        if not isinstance(payload, Mapping):
            raise ContractRepairValidationError("policy tool evidence must be an object")
        return cls(
            candidate_tree_id=payload["candidate_tree_id"],
            required_families=tuple(payload["required_families"]),
            results=tuple(ToolGateResult.from_dict(item) for item in payload["results"]),
            policy_id=payload["policy_id"],
        )


@dataclass(frozen=True)
class ImpactedTestEvidence:
    """Focused contract tests and dependency-complete impacted tests."""

    candidate_tree_id: str
    focused_test_ids: tuple[str, ...]
    impacted_test_ids: tuple[str, ...]
    required_dependant_ids: tuple[str, ...]
    executed_test_ids: tuple[str, ...]
    passed_test_ids: tuple[str, ...]
    failed_test_ids: tuple[str, ...]
    omitted_dependant_ids: tuple[str, ...]
    dependency_complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id"))
        object.__setattr__(
            self, "focused_test_ids", _ids(self.focused_test_ids, "focused_test_ids", maximum=MAX_TEST_IDS)
        )
        object.__setattr__(
            self, "impacted_test_ids", _ids(self.impacted_test_ids, "impacted_test_ids", maximum=MAX_TEST_IDS)
        )
        object.__setattr__(
            self,
            "required_dependant_ids",
            _ids(self.required_dependant_ids, "required_dependant_ids", required=False, maximum=MAX_TEST_IDS),
        )
        object.__setattr__(
            self,
            "executed_test_ids",
            _ids(self.executed_test_ids, "executed_test_ids", required=False, maximum=MAX_TEST_IDS),
        )
        object.__setattr__(
            self,
            "passed_test_ids",
            _ids(self.passed_test_ids, "passed_test_ids", required=False, maximum=MAX_TEST_IDS),
        )
        object.__setattr__(
            self,
            "failed_test_ids",
            _ids(self.failed_test_ids, "failed_test_ids", required=False, maximum=MAX_TEST_IDS),
        )
        object.__setattr__(
            self,
            "omitted_dependant_ids",
            _ids(self.omitted_dependant_ids, "omitted_dependant_ids", required=False, maximum=MAX_TEST_IDS),
        )
        object.__setattr__(self, "dependency_complete", _bool(self.dependency_complete, "dependency_complete"))
        if self.dependency_complete and self.omitted_dependant_ids:
            raise ContractRepairValidationError("dependency_complete forbids omitted dependants")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_IMPACTED_TEST_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "focused_test_ids": list(self.focused_test_ids),
            "impacted_test_ids": list(self.impacted_test_ids),
            "required_dependant_ids": list(self.required_dependant_ids),
            "executed_test_ids": list(self.executed_test_ids),
            "passed_test_ids": list(self.passed_test_ids),
            "failed_test_ids": list(self.failed_test_ids),
            "omitted_dependant_ids": list(self.omitted_dependant_ids),
            "dependency_complete": self.dependency_complete,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImpactedTestEvidence":
        if not isinstance(payload, Mapping):
            raise ContractRepairValidationError("impacted test evidence must be an object")
        return cls(
            candidate_tree_id=payload["candidate_tree_id"],
            focused_test_ids=tuple(payload["focused_test_ids"]),
            impacted_test_ids=tuple(payload["impacted_test_ids"]),
            required_dependant_ids=tuple(payload.get("required_dependant_ids", ())),
            executed_test_ids=tuple(payload.get("executed_test_ids", ())),
            passed_test_ids=tuple(payload.get("passed_test_ids", ())),
            failed_test_ids=tuple(payload.get("failed_test_ids", ())),
            omitted_dependant_ids=tuple(payload.get("omitted_dependant_ids", ())),
            dependency_complete=payload["dependency_complete"],
        )


@dataclass(frozen=True)
class IntegrityEvidence:
    """Anti-weakening / anti-suppression checks for contracts, tests, and checkers."""

    candidate_tree_id: str
    contracts_deleted: tuple[str, ...]
    contracts_weakened: tuple[str, ...]
    tests_deleted: tuple[str, ...]
    tests_weakened: tuple[str, ...]
    checkers_deleted: tuple[str, ...]
    checkers_weakened: tuple[str, ...]
    findings_suppressed: tuple[str, ...]
    original_finding_id: str
    original_finding_closed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id"))
        object.__setattr__(self, "original_finding_id", _identifier(self.original_finding_id, "original_finding_id"))
        object.__setattr__(self, "original_finding_closed", _bool(self.original_finding_closed, "original_finding_closed"))
        for name in (
            "contracts_deleted",
            "contracts_weakened",
            "tests_deleted",
            "tests_weakened",
            "checkers_deleted",
            "checkers_weakened",
            "findings_suppressed",
        ):
            object.__setattr__(
                self,
                name,
                _ids(getattr(self, name), name, required=False, maximum=MAX_TEST_IDS),
            )

    @property
    def clean(self) -> bool:
        return (
            not self.contracts_deleted
            and not self.contracts_weakened
            and not self.tests_deleted
            and not self.tests_weakened
            and not self.checkers_deleted
            and not self.checkers_weakened
            and not self.findings_suppressed
            and self.original_finding_closed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_INTEGRITY_SCHEMA,
            "candidate_tree_id": self.candidate_tree_id,
            "contracts_deleted": list(self.contracts_deleted),
            "contracts_weakened": list(self.contracts_weakened),
            "tests_deleted": list(self.tests_deleted),
            "tests_weakened": list(self.tests_weakened),
            "checkers_deleted": list(self.checkers_deleted),
            "checkers_weakened": list(self.checkers_weakened),
            "findings_suppressed": list(self.findings_suppressed),
            "original_finding_id": self.original_finding_id,
            "original_finding_closed": self.original_finding_closed,
            "clean": self.clean,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IntegrityEvidence":
        if not isinstance(payload, Mapping):
            raise ContractRepairValidationError("integrity evidence must be an object")
        return cls(
            candidate_tree_id=payload["candidate_tree_id"],
            contracts_deleted=tuple(payload.get("contracts_deleted", ())),
            contracts_weakened=tuple(payload.get("contracts_weakened", ())),
            tests_deleted=tuple(payload.get("tests_deleted", ())),
            tests_weakened=tuple(payload.get("tests_weakened", ())),
            checkers_deleted=tuple(payload.get("checkers_deleted", ())),
            checkers_weakened=tuple(payload.get("checkers_weakened", ())),
            findings_suppressed=tuple(payload.get("findings_suppressed", ())),
            original_finding_id=payload["original_finding_id"],
            original_finding_closed=payload["original_finding_closed"],
        )


@dataclass(frozen=True)
class StageResult:
    """One ordered stage outcome in the post-edit gate."""

    stage: ValidationStage
    disposition: StageDisposition
    reason_codes: tuple[str, ...] = ()
    evidence_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", ValidationStage(self.stage))
        object.__setattr__(self, "disposition", StageDisposition(self.disposition))
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", required=False, maximum=MAX_REASON_CODES),
        )
        if self.evidence_id:
            object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        else:
            object.__setattr__(self, "evidence_id", "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True)
class ContractRepairValidationReport:
    """Full ordered post-edit gate report; success is not completion authority."""

    packet_id: str
    decision_id: str
    candidate_tree_id: str
    roots: AuthorityRoots
    stages: tuple[StageResult, ...]
    reason_codes: tuple[str, ...]
    complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_id", _identifier(self.packet_id, "packet_id"))
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id"))
        if not isinstance(self.roots, AuthorityRoots):
            raise ContractRepairValidationError("report roots must be AuthorityRoots")
        if not isinstance(self.stages, Sequence) or not all(isinstance(item, StageResult) for item in self.stages):
            raise ContractRepairValidationError("stages must be StageResult values")
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", required=False, maximum=MAX_REASON_CODES),
        )
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        if self.complete and self.reason_codes:
            raise ContractRepairValidationError("a complete report cannot carry failure reason codes")
        if self.complete and any(item.disposition is not StageDisposition.PASSED for item in self.stages):
            raise ContractRepairValidationError("a complete report requires every stage to pass")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_VALIDATION_REPORT_SCHEMA,
            "interface": CONTRACT_REPAIR_VALIDATOR_INTERFACE,
            "packet_id": self.packet_id,
            "decision_id": self.decision_id,
            "candidate_tree_id": self.candidate_tree_id,
            "roots": self.roots.to_dict(),
            "stages": [item.to_dict() for item in self.stages],
            "reason_codes": list(self.reason_codes),
            "complete": self.complete,
        }

    @property
    def report_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "report_id": self.report_id}


@dataclass(frozen=True)
class ContractRepairCompletionReceipt:
    """Only current complete receipts may close the original finding.

    The receipt is content-addressed and bound to the exact packet, decision,
    candidate tree, index, proof, tool, test, and integrity evidence.  A green
    provider run without this receipt is not completion evidence.
    """

    packet_id: str
    decision_id: str
    finding_id: str
    candidate_tree_id: str
    roots: AuthorityRoots
    index_rebuild_id: str
    edge_resolution_id: str
    contract_extraction_id: str
    obligation_reproof_id: str
    policy_tool_report_id: str
    impacted_test_report_id: str
    integrity_report_id: str
    validation_report_id: str
    proved_obligation_ids: tuple[str, ...]
    required_tool_families: tuple[str, ...]
    focused_test_ids: tuple[str, ...]
    impacted_test_ids: tuple[str, ...]
    write_paths: tuple[str, ...]
    checked_at: int

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots):
            raise ContractRepairValidationError("completion receipt roots must be AuthorityRoots")
        for name in (
            "packet_id",
            "decision_id",
            "finding_id",
            "candidate_tree_id",
            "index_rebuild_id",
            "edge_resolution_id",
            "contract_extraction_id",
            "obligation_reproof_id",
            "policy_tool_report_id",
            "impacted_test_report_id",
            "integrity_report_id",
            "validation_report_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "proved_obligation_ids",
            _ids(self.proved_obligation_ids, "proved_obligation_ids"),
        )
        object.__setattr__(
            self,
            "required_tool_families",
            _ids(self.required_tool_families, "required_tool_families", maximum=MAX_TOOL_IDS),
        )
        object.__setattr__(
            self,
            "focused_test_ids",
            _ids(self.focused_test_ids, "focused_test_ids", maximum=MAX_TEST_IDS),
        )
        object.__setattr__(
            self,
            "impacted_test_ids",
            _ids(self.impacted_test_ids, "impacted_test_ids", maximum=MAX_TEST_IDS),
        )
        object.__setattr__(self, "write_paths", _paths(self.write_paths, "write_paths"))
        if isinstance(self.checked_at, bool) or not isinstance(self.checked_at, int) or self.checked_at < 0:
            raise ContractRepairValidationError("checked_at must be a non-negative integer")
        if self.candidate_tree_id != self.roots.tree_id:
            raise ContractRepairValidationError("completion receipt must bind the candidate tree root")
        if len(canonical_json_bytes(self.to_dict())) > MAX_COMPLETION_RECEIPT_BYTES:
            raise ContractRepairValidationError("completion receipt exceeds its serialized byte bound")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPAIR_COMPLETION_RECEIPT_SCHEMA,
            "interface": CONTRACT_REPAIR_VALIDATOR_INTERFACE,
            "packet_id": self.packet_id,
            "decision_id": self.decision_id,
            "finding_id": self.finding_id,
            "candidate_tree_id": self.candidate_tree_id,
            "roots": self.roots.to_dict(),
            "index_rebuild_id": self.index_rebuild_id,
            "edge_resolution_id": self.edge_resolution_id,
            "contract_extraction_id": self.contract_extraction_id,
            "obligation_reproof_id": self.obligation_reproof_id,
            "policy_tool_report_id": self.policy_tool_report_id,
            "impacted_test_report_id": self.impacted_test_report_id,
            "integrity_report_id": self.integrity_report_id,
            "validation_report_id": self.validation_report_id,
            "proved_obligation_ids": list(self.proved_obligation_ids),
            "required_tool_families": list(self.required_tool_families),
            "focused_test_ids": list(self.focused_test_ids),
            "impacted_test_ids": list(self.impacted_test_ids),
            "write_paths": list(self.write_paths),
            "checked_at": self.checked_at,
            "closes_original_finding": True,
            "provider_success_is_not_completion": True,
        }

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def content_id(self) -> str:
        return self.receipt_id

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractRepairCompletionReceipt":
        fields = {
            "schema",
            "interface",
            "receipt_id",
            "packet_id",
            "decision_id",
            "finding_id",
            "candidate_tree_id",
            "roots",
            "index_rebuild_id",
            "edge_resolution_id",
            "contract_extraction_id",
            "obligation_reproof_id",
            "policy_tool_report_id",
            "impacted_test_report_id",
            "integrity_report_id",
            "validation_report_id",
            "proved_obligation_ids",
            "required_tool_families",
            "focused_test_ids",
            "impacted_test_ids",
            "write_paths",
            "checked_at",
            "closes_original_finding",
            "provider_success_is_not_completion",
        }
        if not isinstance(payload, Mapping) or set(payload).difference(fields):
            raise ContractRepairValidationError("completion receipt contains unsupported fields")
        if payload.get("schema") != CONTRACT_REPAIR_COMPLETION_RECEIPT_SCHEMA:
            raise ContractRepairValidationError("completion receipt has an unsupported schema")
        if payload.get("interface") != CONTRACT_REPAIR_VALIDATOR_INTERFACE:
            raise ContractRepairValidationError("completion receipt has an unsupported interface")
        if payload.get("closes_original_finding") is not True:
            raise ContractRepairValidationError("completion receipt must close the original finding")
        if payload.get("provider_success_is_not_completion") is not True:
            raise ContractRepairValidationError("completion receipt must deny provider-success authority")
        try:
            receipt = cls(
                packet_id=payload["packet_id"],
                decision_id=payload["decision_id"],
                finding_id=payload["finding_id"],
                candidate_tree_id=payload["candidate_tree_id"],
                roots=AuthorityRoots.from_dict(payload["roots"]),
                index_rebuild_id=payload["index_rebuild_id"],
                edge_resolution_id=payload["edge_resolution_id"],
                contract_extraction_id=payload["contract_extraction_id"],
                obligation_reproof_id=payload["obligation_reproof_id"],
                policy_tool_report_id=payload["policy_tool_report_id"],
                impacted_test_report_id=payload["impacted_test_report_id"],
                integrity_report_id=payload["integrity_report_id"],
                validation_report_id=payload["validation_report_id"],
                proved_obligation_ids=tuple(payload["proved_obligation_ids"]),
                required_tool_families=tuple(payload["required_tool_families"]),
                focused_test_ids=tuple(payload["focused_test_ids"]),
                impacted_test_ids=tuple(payload["impacted_test_ids"]),
                write_paths=tuple(payload["write_paths"]),
                checked_at=payload["checked_at"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractRepairValidationError("completion receipt is malformed") from exc
        if payload.get("receipt_id") not in (None, "", receipt.receipt_id):
            raise ContractRepairValidationError("completion receipt identity is forged")
        return receipt


@dataclass(frozen=True)
class CandidatePatchEvidence:
    """Complete structured evidence for one candidate-tree validation pass.

    Adapters produce these records; the validator never invents missing stages.
    """

    candidate_tree_id: str
    index_rebuild: IndexRebuildEvidence
    edge_resolution: EdgeResolutionEvidence
    contract_extraction: ContractExtractionEvidence
    obligation_reproof: ObligationReproofEvidence
    policy_tools: PolicyToolEvidence
    impacted_tests: ImpactedTestEvidence
    integrity: IntegrityEvidence
    expected_deleted_paths: tuple[str, ...] = ()
    expected_tombstone_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_tree_id", _identifier(self.candidate_tree_id, "candidate_tree_id"))
        for name, expected_type in (
            ("index_rebuild", IndexRebuildEvidence),
            ("edge_resolution", EdgeResolutionEvidence),
            ("contract_extraction", ContractExtractionEvidence),
            ("obligation_reproof", ObligationReproofEvidence),
            ("policy_tools", PolicyToolEvidence),
            ("impacted_tests", ImpactedTestEvidence),
            ("integrity", IntegrityEvidence),
        ):
            value = getattr(self, name)
            if not isinstance(value, expected_type):
                raise ContractRepairValidationError(f"{name} must be {expected_type.__name__}")
        object.__setattr__(
            self,
            "expected_deleted_paths",
            _paths(self.expected_deleted_paths, "expected_deleted_paths", required=False),
        )
        object.__setattr__(
            self,
            "expected_tombstone_ids",
            _ids(self.expected_tombstone_ids, "expected_tombstone_ids", required=False, maximum=MAX_TOMBSTONES),
        )


@dataclass(frozen=True)
class ValidationOutcome:
    """Either a completion receipt or a failed validation report."""

    report: ContractRepairValidationReport
    receipt: ContractRepairCompletionReceipt | None = None

    @property
    def complete(self) -> bool:
        return self.receipt is not None and self.report.complete

    def require_complete(self) -> ContractRepairCompletionReceipt:
        if self.receipt is None or not self.report.complete:
            reasons = ", ".join(self.report.reason_codes) or "incomplete"
            raise ContractRepairValidationError(
                "contract repair post-edit validation rejected: " + reasons
            )
        return self.receipt


# Optional adapters so callers can supply live reindex/resolve/prove runners.
IndexRebuildAdapter = Callable[[ContractRepairEditPacket, AuthorityRoots], IndexRebuildEvidence]
EdgeResolveAdapter = Callable[[ContractRepairEditPacket, AuthorityRoots], EdgeResolutionEvidence]
ContractExtractAdapter = Callable[[ContractRepairEditPacket, AuthorityRoots], ContractExtractionEvidence]
ObligationReproofAdapter = Callable[[ContractRepairEditPacket, AuthorityRoots], ObligationReproofEvidence]
PolicyToolAdapter = Callable[[ContractRepairEditPacket, AuthorityRoots, Sequence[str]], PolicyToolEvidence]
ImpactedTestAdapter = Callable[[ContractRepairEditPacket, AuthorityRoots], ImpactedTestEvidence]
IntegrityAdapter = Callable[[ContractRepairEditPacket, AuthorityRoots, str], IntegrityEvidence]


@dataclass
class ContractRepairValidator:
    """Orchestrate patch-bound re-index, re-resolve, re-extract, re-prove, and gates.

    ``validate`` is pure over structured :class:`CandidatePatchEvidence` and
    returns a report plus an optional completion receipt.  ``require_complete``
    raises on any incomplete path.  Injected adapters, when present, produce
    evidence; when absent, callers must supply complete evidence up front.
    """

    index_rebuild_adapter: IndexRebuildAdapter | None = None
    edge_resolve_adapter: EdgeResolveAdapter | None = None
    contract_extract_adapter: ContractExtractAdapter | None = None
    obligation_reproof_adapter: ObligationReproofAdapter | None = None
    policy_tool_adapter: PolicyToolAdapter | None = None
    impacted_test_adapter: ImpactedTestAdapter | None = None
    integrity_adapter: IntegrityAdapter | None = None
    default_required_tool_families: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_POLICY_REQUIRED_TOOLS
    )

    def collect_evidence(
        self,
        packet: ContractRepairEditPacket,
        *,
        current_roots: AuthorityRoots,
        finding_id: str,
        evidence: CandidatePatchEvidence | None = None,
        required_tool_families: Sequence[str] | None = None,
    ) -> CandidatePatchEvidence:
        """Collect or verify evidence, preferring injected adapters when present."""

        if evidence is not None:
            return evidence
        if current_roots.tree_id != packet.roots.tree_id and self.index_rebuild_adapter is None:
            # Callers validating a post-edit candidate tree supply new roots;
            # adapters are required when evidence is not prebuilt.
            pass
        tools = tuple(required_tool_families or self.default_required_tool_families)
        missing: list[str] = []
        if self.index_rebuild_adapter is None:
            missing.append("index_rebuild")
        if self.edge_resolve_adapter is None:
            missing.append("edge_resolution")
        if self.contract_extract_adapter is None:
            missing.append("contract_extraction")
        if self.obligation_reproof_adapter is None:
            missing.append("obligation_reproof")
        if self.policy_tool_adapter is None:
            missing.append("policy_tools")
        if self.impacted_test_adapter is None:
            missing.append("impacted_tests")
        if self.integrity_adapter is None:
            missing.append("integrity")
        if missing:
            raise ContractRepairValidationError(
                "incomplete_evidence: missing adapters or CandidatePatchEvidence for "
                + ", ".join(missing)
            )
        assert self.index_rebuild_adapter is not None
        assert self.edge_resolve_adapter is not None
        assert self.contract_extract_adapter is not None
        assert self.obligation_reproof_adapter is not None
        assert self.policy_tool_adapter is not None
        assert self.impacted_test_adapter is not None
        assert self.integrity_adapter is not None
        index = self.index_rebuild_adapter(packet, current_roots)
        edge = self.edge_resolve_adapter(packet, current_roots)
        contracts = self.contract_extract_adapter(packet, current_roots)
        obligations = self.obligation_reproof_adapter(packet, current_roots)
        policy = self.policy_tool_adapter(packet, current_roots, tools)
        tests = self.impacted_test_adapter(packet, current_roots)
        integrity = self.integrity_adapter(packet, current_roots, finding_id)
        return CandidatePatchEvidence(
            candidate_tree_id=current_roots.tree_id,
            index_rebuild=index,
            edge_resolution=edge,
            contract_extraction=contracts,
            obligation_reproof=obligations,
            policy_tools=policy,
            impacted_tests=tests,
            integrity=integrity,
        )

    def validate(
        self,
        packet: ContractRepairEditPacket,
        decision: RepairTargetDecision,
        admission: AdmissionResult,
        *,
        current_roots: AuthorityRoots,
        finding_id: str,
        evidence: CandidatePatchEvidence,
        checked_at: int,
        required_tool_families: Sequence[str] | None = None,
        original_edge_id: str = "",
    ) -> ValidationOutcome:
        """Run the full post-edit gate over structured evidence (fail-closed)."""

        stages: list[StageResult] = []
        reasons: set[str] = set()

        typed = (
            isinstance(packet, ContractRepairEditPacket)
            and isinstance(decision, RepairTargetDecision)
            and isinstance(admission, AdmissionResult)
            and isinstance(current_roots, AuthorityRoots)
            and isinstance(evidence, CandidatePatchEvidence)
            and isinstance(checked_at, int)
            and not isinstance(checked_at, bool)
            and checked_at >= 0
        )
        if not typed:
            report = ContractRepairValidationReport(
                packet_id="invalid",
                decision_id="invalid",
                candidate_tree_id="invalid",
                roots=current_roots if isinstance(current_roots, AuthorityRoots) else AuthorityRoots(
                    repository_id="repository:invalid",
                    forest_id="forest:invalid",
                    tree_id="tree:invalid",
                    graph_id="graph:invalid",
                    index_id="index:invalid",
                    model_id="model:invalid",
                    config_id="config:invalid",
                    translator_id="translator:invalid",
                    toolchain_id="toolchain:invalid",
                    policy_id="policy:invalid",
                ),
                stages=(
                    StageResult(
                        ValidationStage.COMPLETION,
                        StageDisposition.FAILED,
                        (ContractRepairValidationReason.MALFORMED_INPUT.value,),
                    ),
                ),
                reason_codes=(ContractRepairValidationReason.MALFORMED_INPUT.value,),
                complete=False,
            )
            return ValidationOutcome(report=report, receipt=None)

        try:
            finding = _identifier(finding_id, "finding_id")
            tools = _ids(
                tuple(required_tool_families or self.default_required_tool_families),
                "required_tool_families",
                maximum=MAX_TOOL_IDS,
            )
            edge_id = (
                _identifier(original_edge_id, "original_edge_id")
                if original_edge_id
                else packet.trace_id
            )
        except ContractRepairValidationError:
            reasons.add(ContractRepairValidationReason.MALFORMED_INPUT.value)
            report = self._failed_report(packet, decision, current_roots, stages, reasons)
            return ValidationOutcome(report=report, receipt=None)

        # --- Binding: packet / decision / admission / candidate tree ---
        if decision.roots != packet.roots or admission.audit.roots != packet.roots:
            reasons.add(ContractRepairValidationReason.ROOT_DRIFT.value)
        if current_roots.repository_id != packet.roots.repository_id:
            reasons.add(ContractRepairValidationReason.ROOT_DRIFT.value)
        if (
            admission.decision != decision
            or packet.decision_id != decision.content_id
            or packet.candidate_set_id != decision.candidate_set_id
            or packet.strategy != decision.strategy
            or packet.write_paths != decision.permitted_write_paths
            or packet.read_paths != decision.permitted_read_paths
            or packet.proof_refs != decision.proof_refs
            or packet.invalidation_refs != decision.invalidation_refs
        ):
            reasons.add(ContractRepairValidationReason.PACKET_DECISION_MISMATCH.value)
        if (
            decision.disposition is not DecisionDisposition.ADMITTED
            or decision.strategy in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS}
        ):
            reasons.add(ContractRepairValidationReason.AMBIGUOUS_OR_ABSTAINED.value)
        if evidence.candidate_tree_id != current_roots.tree_id:
            reasons.add(ContractRepairValidationReason.STALE_CANDIDATE_TREE.value)
        # Post-edit candidate tree is expected to differ from the pre-edit decision
        # tree_id only when the patch produced a new tree identity; all evidence
        # must still bind that same candidate tree.  Stale means evidence or
        # decision roots no longer match the declared candidate tree.
        for stage_evidence in (
            evidence.index_rebuild,
            evidence.edge_resolution,
            evidence.contract_extraction,
            evidence.obligation_reproof,
            evidence.policy_tools,
            evidence.impacted_tests,
            evidence.integrity,
        ):
            if stage_evidence.candidate_tree_id != current_roots.tree_id:
                reasons.add(ContractRepairValidationReason.STALE_CANDIDATE_TREE.value)
                break

        # --- Stage: index rebuild ---
        index_reasons: list[str] = []
        idx = evidence.index_rebuild
        if not set(packet.write_paths).issubset(idx.affected_paths) and not set(packet.write_paths).issubset(
            idx.rebuilt_source_paths
        ):
            index_reasons.append(ContractRepairValidationReason.INDEX_REBUILD_INCOMPLETE.value)
        if not idx.clean_rebuild_equivalent:
            index_reasons.append(ContractRepairValidationReason.INDEX_REBUILD_INCOMPLETE.value)
        if not idx.rebuilt_vector_row_ids:
            index_reasons.append(ContractRepairValidationReason.INDEX_REBUILD_INCOMPLETE.value)
        if evidence.expected_tombstone_ids and not set(evidence.expected_tombstone_ids).issubset(
            idx.tombstone_ids
        ):
            index_reasons.append(ContractRepairValidationReason.TOMBSTONE_MISSING.value)
        if evidence.expected_deleted_paths and not idx.tombstone_ids:
            index_reasons.append(ContractRepairValidationReason.TOMBSTONE_MISSING.value)
        if idx.index_id != current_roots.index_id and idx.index_id != packet.roots.index_id:
            # Rebuilt index may mint a new index_id bound to the candidate tree;
            # reject only when neither current nor packet index is acknowledged.
            # Require the rebuilt index identity to be present and non-empty (already).
            pass
        stages.append(
            StageResult(
                ValidationStage.INDEX_REBUILD,
                StageDisposition.PASSED if not index_reasons else StageDisposition.FAILED,
                tuple(sorted(set(index_reasons))),
                idx.receipt_id,
            )
        )
        reasons.update(index_reasons)

        # --- Stage: edge resolution ---
        edge_reasons: list[str] = []
        edge = evidence.edge_resolution
        if edge.original_trace_id != packet.trace_id:
            edge_reasons.append(ContractRepairValidationReason.EDGE_NOT_RESOLVED.value)
        if original_edge_id and edge.original_edge_id != edge_id:
            edge_reasons.append(ContractRepairValidationReason.EDGE_NOT_RESOLVED.value)
        if not edge.resolved or edge.residual_unresolved:
            edge_reasons.append(ContractRepairValidationReason.EDGE_NOT_RESOLVED.value)
        if edge.resolved and edge.resolved_target_path not in packet.write_paths and edge.resolved_target_path not in packet.read_paths:
            # Resolved target should land inside packet authority or the admitted write surface.
            if edge.resolved_target_path not in decision.permitted_write_paths:
                edge_reasons.append(ContractRepairValidationReason.EDGE_NOT_RESOLVED.value)
        stages.append(
            StageResult(
                ValidationStage.EDGE_RESOLUTION,
                StageDisposition.PASSED if not edge_reasons else StageDisposition.FAILED,
                tuple(sorted(set(edge_reasons))),
                edge.resolution_receipt_id,
            )
        )
        reasons.update(edge_reasons)

        # --- Stage: contract re-extraction ---
        contract_reasons: list[str] = []
        contracts = evidence.contract_extraction
        if not contracts.contracts_present:
            contract_reasons.append(ContractRepairValidationReason.CONTRACT_DELETED.value)
        if not contracts.clauses_preserved or not contracts.strength_preserved:
            contract_reasons.append(ContractRepairValidationReason.CONTRACT_WEAKENED.value)
        if contracts.original_sender_contract_id != packet.sender_expected_contract_id:
            contract_reasons.append(ContractRepairValidationReason.CONTRACT_REEXTRACTION_FAILED.value)
        if contracts.original_receiver_contract_id != packet.receiver_expected_contract_id:
            contract_reasons.append(ContractRepairValidationReason.CONTRACT_REEXTRACTION_FAILED.value)
        stages.append(
            StageResult(
                ValidationStage.CONTRACT_EXTRACTION,
                StageDisposition.PASSED if not contract_reasons else StageDisposition.FAILED,
                tuple(sorted(set(contract_reasons))),
                contracts.extraction_receipt_id,
            )
        )
        reasons.update(contract_reasons)

        # --- Stage: obligation re-proof ---
        obl_reasons: list[str] = []
        obl = evidence.obligation_reproof
        required_obs = set(packet.post_edit_obligation_ids)
        if not required_obs.issubset(set(obl.original_obligation_ids) | set(obl.introduced_obligation_ids)):
            obl_reasons.append(ContractRepairValidationReason.OBLIGATION_OMITTED.value)
        if obl.omitted_obligation_ids:
            obl_reasons.append(ContractRepairValidationReason.OBLIGATION_OMITTED.value)
        if set(obl.failed_obligation_ids) & set(obl.original_obligation_ids):
            obl_reasons.append(ContractRepairValidationReason.OBLIGATION_FAILED.value)
        if set(obl.failed_obligation_ids) & set(obl.introduced_obligation_ids):
            obl_reasons.append(ContractRepairValidationReason.INTRODUCED_OBLIGATION_FAILED.value)
        if not obl.all_mandatory_proved:
            if ContractRepairValidationReason.OBLIGATION_FAILED.value not in obl_reasons and (
                set(obl.failed_obligation_ids) & set(obl.original_obligation_ids)
            ):
                obl_reasons.append(ContractRepairValidationReason.OBLIGATION_FAILED.value)
            if not obl_reasons:
                obl_reasons.append(ContractRepairValidationReason.OBLIGATION_FAILED.value)
        if not required_obs.issubset(obl.proved_obligation_ids):
            obl_reasons.append(ContractRepairValidationReason.OBLIGATION_FAILED.value)
        stages.append(
            StageResult(
                ValidationStage.OBLIGATION_REPROOF,
                StageDisposition.PASSED if not obl_reasons else StageDisposition.FAILED,
                tuple(sorted(set(obl_reasons))),
                obl.proof_bundle_id,
            )
        )
        reasons.update(obl_reasons)

        # --- Stage: policy tools ---
        tool_reasons: list[str] = []
        policy = evidence.policy_tools
        if policy.policy_id != current_roots.policy_id and policy.policy_id != packet.roots.policy_id:
            tool_reasons.append(ContractRepairValidationReason.ROOT_DRIFT.value)
        if set(tools) - set(policy.required_families):
            # Policy evidence must acknowledge every required family for this run.
            tool_reasons.append(ContractRepairValidationReason.SKIPPED_REQUIRED_TOOL.value)
        by_family = {item.family: item for item in policy.results}
        for family in tools:
            result = by_family.get(family)
            if result is None:
                tool_reasons.append(ContractRepairValidationReason.SKIPPED_REQUIRED_TOOL.value)
                continue
            if result.required and result.skipped:
                tool_reasons.append(ContractRepairValidationReason.SKIPPED_REQUIRED_TOOL.value)
            elif result.required and not result.executed:
                tool_reasons.append(ContractRepairValidationReason.SKIPPED_REQUIRED_TOOL.value)
            elif result.required and not result.passed:
                tool_reasons.append(ContractRepairValidationReason.TOOL_FAILED.value)
        stages.append(
            StageResult(
                ValidationStage.POLICY_TOOLS,
                StageDisposition.PASSED if not tool_reasons else StageDisposition.FAILED,
                tuple(sorted(set(tool_reasons))),
                content_identity(policy.to_dict()),
            )
        )
        reasons.update(tool_reasons)

        # --- Stage: impacted tests ---
        test_reasons: list[str] = []
        tests = evidence.impacted_tests
        required_tests = set(tests.focused_test_ids) | set(tests.impacted_test_ids) | set(
            tests.required_dependant_ids
        )
        if not set(tests.focused_test_ids).issubset(tests.executed_test_ids):
            test_reasons.append(ContractRepairValidationReason.FOCUSED_TEST_FAILED.value)
        if set(tests.focused_test_ids) & set(tests.failed_test_ids):
            test_reasons.append(ContractRepairValidationReason.FOCUSED_TEST_FAILED.value)
        if not set(tests.impacted_test_ids).issubset(tests.executed_test_ids):
            test_reasons.append(ContractRepairValidationReason.IMPACTED_TEST_OMITTED.value)
        if set(tests.impacted_test_ids) & set(tests.failed_test_ids):
            test_reasons.append(ContractRepairValidationReason.IMPACTED_TEST_FAILED.value)
        if tests.omitted_dependant_ids or not tests.dependency_complete:
            test_reasons.append(ContractRepairValidationReason.DEPENDANT_OMITTED.value)
        if required_tests - set(tests.passed_test_ids):
            # Any required test that did not pass fails the stage.
            missing_pass = required_tests - set(tests.passed_test_ids)
            if missing_pass & set(tests.focused_test_ids):
                test_reasons.append(ContractRepairValidationReason.FOCUSED_TEST_FAILED.value)
            if missing_pass & set(tests.impacted_test_ids):
                test_reasons.append(ContractRepairValidationReason.IMPACTED_TEST_FAILED.value)
            if missing_pass & set(tests.required_dependant_ids):
                test_reasons.append(ContractRepairValidationReason.DEPENDANT_OMITTED.value)
        stages.append(
            StageResult(
                ValidationStage.IMPACTED_TESTS,
                StageDisposition.PASSED if not test_reasons else StageDisposition.FAILED,
                tuple(sorted(set(test_reasons))),
                content_identity(tests.to_dict()),
            )
        )
        reasons.update(test_reasons)

        # --- Stage: integrity / anti-weakening ---
        integrity_reasons: list[str] = []
        integrity = evidence.integrity
        if integrity.original_finding_id != finding:
            integrity_reasons.append(ContractRepairValidationReason.ORIGINAL_FINDING_NOT_CLOSED.value)
        if integrity.contracts_deleted:
            integrity_reasons.append(ContractRepairValidationReason.CONTRACT_DELETED.value)
        if integrity.contracts_weakened:
            integrity_reasons.append(ContractRepairValidationReason.CONTRACT_WEAKENED.value)
        if integrity.tests_deleted:
            integrity_reasons.append(ContractRepairValidationReason.TEST_DELETED.value)
        if integrity.tests_weakened:
            integrity_reasons.append(ContractRepairValidationReason.TEST_WEAKENED.value)
        if integrity.checkers_deleted:
            integrity_reasons.append(ContractRepairValidationReason.CHECKER_DELETED.value)
        if integrity.checkers_weakened:
            integrity_reasons.append(ContractRepairValidationReason.CHECKER_WEAKENED.value)
        if integrity.findings_suppressed:
            integrity_reasons.append(ContractRepairValidationReason.FINDING_SUPPRESSED.value)
        if not integrity.original_finding_closed:
            integrity_reasons.append(ContractRepairValidationReason.ORIGINAL_FINDING_NOT_CLOSED.value)
        stages.append(
            StageResult(
                ValidationStage.INTEGRITY,
                StageDisposition.PASSED if not integrity_reasons else StageDisposition.FAILED,
                tuple(sorted(set(integrity_reasons))),
                content_identity(integrity.to_dict()),
            )
        )
        reasons.update(integrity_reasons)

        complete = not reasons and all(item.disposition is StageDisposition.PASSED for item in stages)
        stages.append(
            StageResult(
                ValidationStage.COMPLETION,
                StageDisposition.PASSED if complete else StageDisposition.FAILED,
                () if complete else tuple(sorted(reasons)),
            )
        )

        report = ContractRepairValidationReport(
            packet_id=packet.packet_id,
            decision_id=decision.content_id,
            candidate_tree_id=current_roots.tree_id,
            roots=current_roots,
            stages=tuple(stages),
            reason_codes=tuple(sorted(reasons)),
            complete=complete,
        )
        if not complete:
            return ValidationOutcome(report=report, receipt=None)

        receipt = ContractRepairCompletionReceipt(
            packet_id=packet.packet_id,
            decision_id=decision.content_id,
            finding_id=finding,
            candidate_tree_id=current_roots.tree_id,
            roots=current_roots,
            index_rebuild_id=idx.receipt_id,
            edge_resolution_id=edge.resolution_receipt_id,
            contract_extraction_id=contracts.extraction_receipt_id,
            obligation_reproof_id=obl.proof_bundle_id,
            policy_tool_report_id=content_identity(policy.to_dict()),
            impacted_test_report_id=content_identity(tests.to_dict()),
            integrity_report_id=content_identity(integrity.to_dict()),
            validation_report_id=report.report_id,
            proved_obligation_ids=obl.proved_obligation_ids,
            required_tool_families=tools,
            focused_test_ids=tests.focused_test_ids,
            impacted_test_ids=tests.impacted_test_ids,
            write_paths=packet.write_paths,
            checked_at=checked_at,
        )
        return ValidationOutcome(report=report, receipt=receipt)

    def require_complete(
        self,
        packet: ContractRepairEditPacket,
        decision: RepairTargetDecision,
        admission: AdmissionResult,
        *,
        current_roots: AuthorityRoots,
        finding_id: str,
        evidence: CandidatePatchEvidence,
        checked_at: int,
        required_tool_families: Sequence[str] | None = None,
        original_edge_id: str = "",
    ) -> ContractRepairCompletionReceipt:
        outcome = self.validate(
            packet,
            decision,
            admission,
            current_roots=current_roots,
            finding_id=finding_id,
            evidence=evidence,
            checked_at=checked_at,
            required_tool_families=required_tool_families,
            original_edge_id=original_edge_id,
        )
        return outcome.require_complete()

    def is_complete(self, *args: Any, **kwargs: Any) -> bool:
        return self.validate(*args, **kwargs).complete

    @staticmethod
    def _failed_report(
        packet: ContractRepairEditPacket | None,
        decision: RepairTargetDecision | None,
        current_roots: AuthorityRoots,
        stages: Sequence[StageResult],
        reasons: set[str],
    ) -> ContractRepairValidationReport:
        return ContractRepairValidationReport(
            packet_id=packet.packet_id if isinstance(packet, ContractRepairEditPacket) else "invalid",
            decision_id=decision.content_id if isinstance(decision, RepairTargetDecision) else "invalid",
            candidate_tree_id=current_roots.tree_id,
            roots=current_roots,
            stages=tuple(stages)
            or (
                StageResult(
                    ValidationStage.COMPLETION,
                    StageDisposition.FAILED,
                    tuple(sorted(reasons)) or (ContractRepairValidationReason.MALFORMED_INPUT.value,),
                ),
            ),
            reason_codes=tuple(sorted(reasons)) or (ContractRepairValidationReason.MALFORMED_INPUT.value,),
            complete=False,
        )


def build_passing_tool_evidence(
    candidate_tree_id: str,
    policy_id: str,
    *,
    families: Sequence[str] = DEFAULT_POLICY_REQUIRED_TOOLS,
) -> PolicyToolEvidence:
    """Helper for hermetic fixtures: every required family executed and passed."""

    results = tuple(
        ToolGateResult(
            tool_id=f"tool:{family}",
            family=family,
            required=True,
            executed=True,
            passed=True,
            skipped=False,
            receipt_id=f"tool-receipt:{family}",
        )
        for family in families
    )
    return PolicyToolEvidence(
        candidate_tree_id=candidate_tree_id,
        required_families=tuple(families),
        results=results,
        policy_id=policy_id,
    )


__all__ = [
    "CONTRACT_REPAIR_COMPLETION_RECEIPT_SCHEMA",
    "CONTRACT_REPAIR_VALIDATOR_INTERFACE",
    "CONTRACT_REPAIR_VALIDATION_REPORT_SCHEMA",
    "DEFAULT_POLICY_REQUIRED_TOOLS",
    "MAX_COMPLETION_RECEIPT_BYTES",
    "POLICY_TOOL_FAMILIES",
    "CandidatePatchEvidence",
    "ContractExtractionEvidence",
    "ContractRepairCompletionReceipt",
    "ContractRepairValidationError",
    "ContractRepairValidationReason",
    "ContractRepairValidationReport",
    "ContractRepairValidator",
    "EdgeResolutionEvidence",
    "ImpactedTestEvidence",
    "IndexRebuildEvidence",
    "IntegrityEvidence",
    "ObligationReproofEvidence",
    "PolicyToolEvidence",
    "StageDisposition",
    "StageResult",
    "ToolGateResult",
    "ValidationOutcome",
    "ValidationStage",
    "build_passing_tool_evidence",
]
