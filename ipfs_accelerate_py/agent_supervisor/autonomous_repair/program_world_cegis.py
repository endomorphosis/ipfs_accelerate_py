"""Bounded program-world CEGIS (SAWM-021).

Interface: ``ProgramWorldCEGIS@1``
Evidence: ``sawm/cegis-repair@1``

Counterexample-guided synthesis over the closed program-world operator
vocabulary.  Analytical unique operators and equality normalization run
before any enumerative CEGIS round.  Model proposals are considered only
after deterministic routes are exhausted, remain proposal-only, and cannot
self-approve.  Identical failures cannot trigger another model retry without
new evidence or a changed strategy.

This module does not create a second patch engine, shell executor, validation
gate, or acceptance authority.  Importing it performs no I/O and starts no
analysis.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from .program_world_operators import (
    DEFAULT_PROTECTED_PATHS,
    MAX_COLLECTION_ITEMS,
    MAX_SOURCE_BYTES,
    MAX_SPAN_BYTES,
    PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE,
    SAWM_CEGIS_REPAIR_EVIDENCE,
    ProgramWorldDefectFamily,
    ProgramWorldOperatorApplication,
    ProgramWorldOperatorAuthorityError,
    ProgramWorldOperatorBoundsError,
    ProgramWorldOperatorError,
    ProgramWorldOperatorKind,
    ProgramWorldPatchSketch,
    ProgramWorldRepairOperatorRegistry,
    apply_program_world_repair_operator,
    detect_authority_claims,
    detect_test_weakening,
    failure_fingerprint,
    is_test_path,
    new_execution_risks,
    operators_for_family,
    path_is_admitted,
    path_is_protected,
    _enum,
    _ids,
    _mapping,
    _optional_text,
    _parse_python,
    _source_bytes,
    _span,
    _text,
    normalize_repo_path,
)


PROGRAM_WORLD_CEGIS_INTERFACE: Final[str] = "ProgramWorldCEGIS@1"
REPAIR_CANDIDATE_SCOPE_GATE_INTERFACE: Final[str] = "RepairCandidateScopeGate@1"
REPAIR_COUNTEREXAMPLE_REFINER_INTERFACE: Final[str] = "RepairCounterexampleRefiner@1"

PROGRAM_WORLD_CEGIS_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-cegis-bounds@1"
)
PROGRAM_WORLD_COUNTEREXAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-counterexample@1"
)
PROGRAM_WORLD_GATE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-gate@1"
)
PROGRAM_WORLD_CEGIS_ITERATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-cegis-iteration@1"
)
PROGRAM_WORLD_REPAIR_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-receipt@1"
)
PROGRAM_WORLD_RESIDUAL_QUESTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-residual-question@1"
)

PRODUCER_ID: Final[str] = "program-world-cegis@1"
CEGIS_VERSION: Final[str] = "1"
TASK_ID: Final[str] = "SAWM-021"

DEFAULT_MAX_ITERATIONS: Final[int] = 8
DEFAULT_MAX_CANDIDATES: Final[int] = 4
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 3
DEFAULT_WALL_TIME_MS: Final[int] = 60_000
HARD_MAX_ITERATIONS: Final[int] = 32
HARD_MAX_CANDIDATES: Final[int] = 32
HARD_MAX_IDENTICAL_FAILURES: Final[int] = 16
HARD_MAX_WALL_TIME_MS: Final[int] = 300_000

ANALYTICAL_PROCEDURE: Final[str] = "analytical_unique_operator"
EQUALITY_PROCEDURE: Final[str] = "equality_normalization"
CEGIS_PROCEDURE: Final[str] = "bounded_cegis"
RESIDUAL_PROCEDURE: Final[str] = "residual_or_model_proposal"
MODEL_STRATEGY: Final[str] = "model_proposal"


class ProgramWorldCegisError(ProgramWorldOperatorError):
    """CEGIS inputs cannot produce a trustworthy repair receipt."""


class ProgramWorldCegisAuthorityError(ProgramWorldOperatorAuthorityError):
    """A CEGIS candidate attempted to mint authority or retry without evidence."""


class ProgramWorldCegisBoundsError(ProgramWorldOperatorBoundsError):
    """A CEGIS budget or payload bound was exceeded."""


class ProgramWorldRepairDisposition(str, Enum):
    """Closed CEGIS outcomes.  Success remains proposal-only."""

    UNIQUE = "unique"
    SUPPORTED = "supported"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    RESIDUAL = "residual"
    BUDGET_EXHAUSTED = "budget_exhausted"
    IDENTICAL_FAILURE = "identical_failure"


class ProgramWorldGateKind(str, Enum):
    SCOPE = "scope"
    TYPE = "type"
    EFFECT = "effect"
    PROOF = "proof"
    TEST = "test"
    AUTHORITY = "authority"
    PROTECTED_PATH = "protected_path"
    STRATEGY = "strategy"


class ProgramWorldGateVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


class ProgramWorldIterationOutcome(str, Enum):
    SELECTED = "selected"
    CANDIDATE_REJECTED = "candidate_rejected"
    NO_CANDIDATE = "no_candidate"
    REFINED = "refined"
    REPEATED_STATE = "repeated_state"
    BUDGET_EXHAUSTED = "budget_exhausted"
    IDENTICAL_FAILURE = "identical_failure"


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProgramWorldCegisError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ProgramWorldCegisError(f"{name} must be a positive integer")
    if value > maximum:
        raise ProgramWorldCegisBoundsError(f"{name} exceeds hard ceiling {maximum}")
    return value


def _nonneg_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ProgramWorldCegisError(f"{name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ProgramWorldCegisBoundsError(f"{name} exceeds hard ceiling {maximum}")
    return value


def _paths(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ProgramWorldCegisError(f"{name} must be a sequence of paths")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldCegisBoundsError(f"{name} exceeds its bound")
    return tuple(normalize_repo_path(item, name) for item in values)


@dataclass(frozen=True)
class ProgramWorldCegisBounds(CanonicalContract):
    """Hard-capped CEGIS budget.  Policy may only tighten the ceilings."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_CEGIS_BOUNDS_SCHEMA

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_candidates_per_iteration: int = DEFAULT_MAX_CANDIDATES
    max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES
    wall_time_ms: int = DEFAULT_WALL_TIME_MS
    max_source_bytes: int = MAX_SOURCE_BYTES
    max_model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_iterations",
            _positive_int(self.max_iterations, "max_iterations", maximum=HARD_MAX_ITERATIONS),
        )
        object.__setattr__(
            self,
            "max_candidates_per_iteration",
            _positive_int(
                self.max_candidates_per_iteration,
                "max_candidates_per_iteration",
                maximum=HARD_MAX_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            "max_identical_failures",
            _positive_int(
                self.max_identical_failures,
                "max_identical_failures",
                maximum=HARD_MAX_IDENTICAL_FAILURES,
            ),
        )
        object.__setattr__(
            self,
            "wall_time_ms",
            _positive_int(self.wall_time_ms, "wall_time_ms", maximum=HARD_MAX_WALL_TIME_MS),
        )
        object.__setattr__(
            self,
            "max_source_bytes",
            _positive_int(
                self.max_source_bytes, "max_source_bytes", maximum=MAX_SOURCE_BYTES
            ),
        )
        if self.max_model_calls != 0:
            raise ProgramWorldCegisAuthorityError(
                "program-world CEGIS hard-zeros model calls; residual proposals are inputs"
            )
        object.__setattr__(self, "max_model_calls", 0)

    def _payload(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_candidates_per_iteration": self.max_candidates_per_iteration,
            "max_identical_failures": self.max_identical_failures,
            "wall_time_ms": self.wall_time_ms,
            "max_source_bytes": self.max_source_bytes,
            "max_model_calls": 0,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ProgramWorldCegisBounds":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ProgramWorldCegisError("bounds must be an object")
        fields = {
            name: payload[name]
            for name in (
                "max_iterations",
                "max_candidates_per_iteration",
                "max_identical_failures",
                "wall_time_ms",
                "max_source_bytes",
                "max_model_calls",
            )
            if name in payload
        }
        return cls(**fields)


@dataclass(frozen=True)
class ProgramWorldCounterexample(CanonicalContract):
    """Typed counterexample that CEGIS refines monotonically."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_COUNTEREXAMPLE_SCHEMA

    counterexample_id: str
    path: str
    family: ProgramWorldDefectFamily
    witness: str = ""
    evidence_cid: str = ""
    abstract_markers: tuple[str, ...] = ()
    model_markers: tuple[str, ...] = ()
    test_markers: tuple[str, ...] = ()
    proof_markers: tuple[str, ...] = ()
    type_markers: tuple[str, ...] = ()
    effect_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counterexample_id",
            _text(self.counterexample_id, "counterexample_id"),
        )
        object.__setattr__(self, "path", normalize_repo_path(self.path))
        object.__setattr__(
            self, "family", _enum(self.family, ProgramWorldDefectFamily, "family")
        )
        object.__setattr__(self, "witness", _optional_text(self.witness, "witness"))
        object.__setattr__(
            self, "abstract_markers", _ids(self.abstract_markers, "abstract_markers")
        )
        object.__setattr__(
            self, "model_markers", _ids(self.model_markers, "model_markers")
        )
        object.__setattr__(self, "test_markers", _ids(self.test_markers, "test_markers"))
        object.__setattr__(
            self, "proof_markers", _ids(self.proof_markers, "proof_markers")
        )
        object.__setattr__(self, "type_markers", _ids(self.type_markers, "type_markers"))
        object.__setattr__(
            self, "effect_markers", _ids(self.effect_markers, "effect_markers")
        )
        evidence = _optional_text(self.evidence_cid, "evidence_cid")
        if not evidence:
            evidence = content_identity(self._payload_without_evidence())
        object.__setattr__(self, "evidence_cid", evidence)

    def _payload_without_evidence(self) -> dict[str, Any]:
        return {
            "counterexample_id": self.counterexample_id,
            "path": self.path,
            "family": self.family.value,
            "witness": self.witness,
            "abstract_markers": list(self.abstract_markers),
            "model_markers": list(self.model_markers),
            "test_markers": list(self.test_markers),
            "proof_markers": list(self.proof_markers),
            "type_markers": list(self.type_markers),
            "effect_markers": list(self.effect_markers),
        }

    def _payload(self) -> dict[str, Any]:
        payload = self._payload_without_evidence()
        payload["evidence_cid"] = self.evidence_cid
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "ProgramWorldCounterexample",
        *,
        default_path: str = "",
    ) -> "ProgramWorldCounterexample":
        if isinstance(payload, ProgramWorldCounterexample):
            return payload
        if not isinstance(payload, Mapping):
            raise ProgramWorldCegisError("counterexample must be an object")
        return cls(
            counterexample_id=str(
                payload.get("counterexample_id") or payload.get("id") or "cx:program-world"
            ),
            path=str(payload.get("path") or default_path),
            family=payload.get("family") or ProgramWorldDefectFamily.UNSUPPORTED,
            witness=str(payload.get("witness") or ""),
            evidence_cid=str(payload.get("evidence_cid") or ""),
            abstract_markers=tuple(payload.get("abstract_markers") or ()),
            model_markers=tuple(payload.get("model_markers") or ()),
            test_markers=tuple(payload.get("test_markers") or ()),
            proof_markers=tuple(payload.get("proof_markers") or ()),
            type_markers=tuple(payload.get("type_markers") or ()),
            effect_markers=tuple(payload.get("effect_markers") or ()),
        )


@dataclass(frozen=True)
class ProgramWorldGateResult(CanonicalContract):
    SCHEMA: ClassVar[str] = PROGRAM_WORLD_GATE_RESULT_SCHEMA

    kind: ProgramWorldGateKind
    verdict: ProgramWorldGateVerdict
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, ProgramWorldGateKind, "kind"))
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ProgramWorldGateVerdict, "verdict")
        )
        object.__setattr__(self, "reasons", _ids(self.reasons, "reasons"))

    @property
    def passed(self) -> bool:
        return self.verdict is ProgramWorldGateVerdict.PASS

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ProgramWorldResidualQuestion(CanonicalContract):
    """Named decision-relevant residual after deterministic routes exhaust."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_RESIDUAL_QUESTION_SCHEMA

    question_id: str
    reason: str
    decision_relevant: bool = True
    model_may_propose: bool = True
    model_may_self_approve: bool = False
    requires_new_evidence: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _text(self.question_id, "question_id"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(
            self, "decision_relevant", _bool(self.decision_relevant, "decision_relevant")
        )
        object.__setattr__(
            self, "model_may_propose", _bool(self.model_may_propose, "model_may_propose")
        )
        if self.model_may_self_approve:
            raise ProgramWorldCegisAuthorityError(
                "residual questions cannot authorize model self-approval"
            )
        object.__setattr__(self, "model_may_self_approve", False)
        object.__setattr__(
            self,
            "requires_new_evidence",
            _bool(self.requires_new_evidence, "requires_new_evidence"),
        )
        if self.decision_relevant is not True:
            raise ProgramWorldCegisError("residual questions must be decision-relevant")
        object.__setattr__(self, "decision_relevant", True)

    def _payload(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "reason": self.reason,
            "decision_relevant": True,
            "model_may_propose": self.model_may_propose,
            "model_may_self_approve": False,
            "requires_new_evidence": self.requires_new_evidence,
        }


@dataclass(frozen=True)
class ProgramWorldCegisIteration(CanonicalContract):
    SCHEMA: ClassVar[str] = PROGRAM_WORLD_CEGIS_ITERATION_SCHEMA

    iteration: int
    outcome: ProgramWorldIterationOutcome
    procedure: str
    evidence_cid: str
    candidate_kinds: tuple[str, ...] = ()
    rejected_fingerprints: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "iteration", _nonneg_int(self.iteration, "iteration", maximum=HARD_MAX_ITERATIONS)
        )
        object.__setattr__(
            self, "outcome", _enum(self.outcome, ProgramWorldIterationOutcome, "outcome")
        )
        object.__setattr__(self, "procedure", _text(self.procedure, "procedure"))
        object.__setattr__(self, "evidence_cid", _text(self.evidence_cid, "evidence_cid"))
        object.__setattr__(
            self, "candidate_kinds", _ids(self.candidate_kinds, "candidate_kinds")
        )
        object.__setattr__(
            self,
            "rejected_fingerprints",
            _ids(self.rejected_fingerprints, "rejected_fingerprints"),
        )
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))

    def _payload(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "outcome": self.outcome.value,
            "procedure": self.procedure,
            "evidence_cid": self.evidence_cid,
            "candidate_kinds": list(self.candidate_kinds),
            "rejected_fingerprints": list(self.rejected_fingerprints),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ProgramWorldRepairRequest:
    """One bounded program-world repair synthesis request."""

    path: str
    source_text: str
    counterexample: ProgramWorldCounterexample
    admitted_scope: tuple[str, ...]
    protected_paths: tuple[str, ...] = ()
    trusted_paths: tuple[str, ...] = ()
    proof_obligation_ids: tuple[str, ...] = ()
    test_obligation_ids: tuple[str, ...] = ()
    environment_binding_cid: str = ""
    tree_id: str = ""
    bounds: ProgramWorldCegisBounds = ProgramWorldCegisBounds()
    operator_kinds: tuple[str, ...] = ()
    prior_failure_fingerprints: tuple[str, ...] = ()
    prior_evidence_cids: tuple[str, ...] = ()
    model_proposal: Mapping[str, Any] | None = None
    before_span: str = ""
    after_span: str = ""
    symbol: str = ""
    replacement: str = ""
    argument: str = ""
    import_module: str = ""
    import_name: str = ""
    source_term: str = ""
    target_term: str = ""
    artifact_digest: str = ""
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_repo_path(self.path))
        object.__setattr__(self, "source_text", _source_bytes(self.source_text))
        cx = ProgramWorldCounterexample.from_mapping(
            self.counterexample, default_path=self.path
        )
        object.__setattr__(self, "counterexample", cx)
        object.__setattr__(self, "admitted_scope", _paths(self.admitted_scope, "admitted_scope"))
        object.__setattr__(
            self, "protected_paths", _paths(self.protected_paths, "protected_paths")
        )
        object.__setattr__(self, "trusted_paths", _paths(self.trusted_paths, "trusted_paths"))
        object.__setattr__(
            self,
            "proof_obligation_ids",
            _ids(self.proof_obligation_ids, "proof_obligation_ids"),
        )
        object.__setattr__(
            self, "test_obligation_ids", _ids(self.test_obligation_ids, "test_obligation_ids")
        )
        object.__setattr__(
            self,
            "environment_binding_cid",
            _optional_text(self.environment_binding_cid, "environment_binding_cid"),
        )
        object.__setattr__(self, "tree_id", _optional_text(self.tree_id, "tree_id"))
        bounds = self.bounds
        if isinstance(bounds, Mapping):
            bounds = ProgramWorldCegisBounds.from_mapping(bounds)
        elif not isinstance(bounds, ProgramWorldCegisBounds):
            bounds = ProgramWorldCegisBounds()
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(
            self, "operator_kinds", _ids(self.operator_kinds, "operator_kinds")
        )
        object.__setattr__(
            self,
            "prior_failure_fingerprints",
            _ids(self.prior_failure_fingerprints, "prior_failure_fingerprints"),
        )
        object.__setattr__(
            self, "prior_evidence_cids", _ids(self.prior_evidence_cids, "prior_evidence_cids")
        )
        proposal = self.model_proposal
        if proposal is not None and not isinstance(proposal, Mapping):
            raise ProgramWorldCegisError("model_proposal must be an object")
        object.__setattr__(
            self,
            "model_proposal",
            None if proposal is None else MappingProxyType(dict(proposal)),
        )
        object.__setattr__(self, "before_span", _span(self.before_span, "before_span"))
        object.__setattr__(self, "after_span", _span(self.after_span, "after_span"))
        object.__setattr__(self, "symbol", _optional_text(self.symbol, "symbol"))
        object.__setattr__(
            self, "replacement", _optional_text(self.replacement, "replacement")
        )
        object.__setattr__(self, "argument", _optional_text(self.argument, "argument"))
        object.__setattr__(
            self, "import_module", _optional_text(self.import_module, "import_module")
        )
        object.__setattr__(
            self, "import_name", _optional_text(self.import_name, "import_name")
        )
        object.__setattr__(
            self,
            "source_term",
            _optional_text(self.source_term, "source_term", limit=MAX_SPAN_BYTES),
        )
        object.__setattr__(
            self,
            "target_term",
            _optional_text(self.target_term, "target_term", limit=MAX_SPAN_BYTES),
        )
        object.__setattr__(
            self, "artifact_digest", _optional_text(self.artifact_digest, "artifact_digest")
        )
        meta = _mapping(self.metadata, "metadata")
        object.__setattr__(self, "metadata", MappingProxyType(meta))
        if len(self.source_text.encode("utf-8")) > self.bounds.max_source_bytes:
            raise ProgramWorldCegisBoundsError("source_text exceeds CEGIS source bound")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProgramWorldRepairRequest":
        if not isinstance(payload, Mapping):
            raise ProgramWorldCegisError("request must be an object")
        return cls(
            path=str(payload.get("path") or ""),
            source_text=str(payload.get("source_text") or payload.get("source") or ""),
            counterexample=payload.get("counterexample") or {},
            admitted_scope=tuple(payload.get("admitted_scope") or ()),
            protected_paths=tuple(payload.get("protected_paths") or ()),
            trusted_paths=tuple(payload.get("trusted_paths") or ()),
            proof_obligation_ids=tuple(payload.get("proof_obligation_ids") or ()),
            test_obligation_ids=tuple(payload.get("test_obligation_ids") or ()),
            environment_binding_cid=str(payload.get("environment_binding_cid") or ""),
            tree_id=str(payload.get("tree_id") or ""),
            bounds=payload.get("bounds") or ProgramWorldCegisBounds(),
            operator_kinds=tuple(payload.get("operator_kinds") or ()),
            prior_failure_fingerprints=tuple(
                payload.get("prior_failure_fingerprints") or ()
            ),
            prior_evidence_cids=tuple(payload.get("prior_evidence_cids") or ()),
            model_proposal=payload.get("model_proposal"),
            before_span=str(payload.get("before_span") or ""),
            after_span=str(payload.get("after_span") or ""),
            symbol=str(payload.get("symbol") or ""),
            replacement=str(payload.get("replacement") or ""),
            argument=str(payload.get("argument") or ""),
            import_module=str(payload.get("import_module") or ""),
            import_name=str(payload.get("import_name") or ""),
            source_term=str(payload.get("source_term") or ""),
            target_term=str(payload.get("target_term") or ""),
            artifact_digest=str(payload.get("artifact_digest") or ""),
            metadata=payload.get("metadata") or {},
        )

    def operator_kwargs(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source_text": self.source_text,
            "before_span": self.before_span,
            "after_span": self.after_span,
            "symbol": self.symbol,
            "replacement": self.replacement,
            "argument": self.argument,
            "import_module": self.import_module,
            "import_name": self.import_name,
            "source_term": self.source_term,
            "target_term": self.target_term,
            "artifact_digest": self.artifact_digest,
            "admitted_scope": self.admitted_scope,
            "protected_paths": tuple(DEFAULT_PROTECTED_PATHS) + self.protected_paths,
            "trusted_paths": self.trusted_paths,
            "proof_obligation_ids": self.proof_obligation_ids,
            "test_obligation_ids": self.test_obligation_ids,
            "metadata": dict(self.metadata),
            "environment_binding_cid": self.environment_binding_cid,
            "tree_id": self.tree_id,
        }


@dataclass(frozen=True)
class ProgramWorldRepairReceipt(CanonicalContract):
    """Terminal receipt of one bounded program-world repair attempt."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REPAIR_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_CEGIS_INTERFACE

    disposition: ProgramWorldRepairDisposition
    path: str
    reason_codes: tuple[str, ...]
    analytical_first: bool
    procedures_attempted: tuple[str, ...]
    selected_procedure: str
    evidence_cid: str
    counterexample_id: str
    bounds: ProgramWorldCegisBounds
    application: ProgramWorldOperatorApplication | None = None
    residual: ProgramWorldResidualQuestion | None = None
    iterations: tuple[ProgramWorldCegisIteration, ...] = ()
    gate_results: tuple[ProgramWorldGateResult, ...] = ()
    failure_fingerprints: tuple[str, ...] = ()
    preserved_evidence_cids: tuple[str, ...] = ()
    model_call_count: int = 0
    proposal_only: bool = True
    independently_admitted: bool = False
    grants_write_authority: bool = False
    grants_semantic_authority: bool = False
    grants_proof_authority: bool = False
    grants_completion_authority: bool = False
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ProgramWorldRepairDisposition, "disposition"),
        )
        object.__setattr__(self, "path", normalize_repo_path(self.path))
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))
        object.__setattr__(
            self, "analytical_first", _bool(self.analytical_first, "analytical_first")
        )
        if self.analytical_first is not True:
            raise ProgramWorldCegisAuthorityError("analytical procedures must run first")
        object.__setattr__(
            self,
            "procedures_attempted",
            _ids(self.procedures_attempted, "procedures_attempted"),
        )
        if self.procedures_attempted and self.procedures_attempted[0] not in {
            ANALYTICAL_PROCEDURE,
            EQUALITY_PROCEDURE,
        }:
            raise ProgramWorldCegisAuthorityError(
                "the first attempted procedure must be analytical"
            )
        object.__setattr__(
            self, "selected_procedure", _text(self.selected_procedure, "selected_procedure")
        )
        object.__setattr__(self, "evidence_cid", _text(self.evidence_cid, "evidence_cid"))
        object.__setattr__(
            self, "counterexample_id", _text(self.counterexample_id, "counterexample_id")
        )
        if not isinstance(self.bounds, ProgramWorldCegisBounds):
            raise ProgramWorldCegisError("bounds must be ProgramWorldCegisBounds")
        if self.application is not None and not isinstance(
            self.application, ProgramWorldOperatorApplication
        ):
            raise ProgramWorldCegisError("application must be ProgramWorldOperatorApplication")
        if self.residual is not None and not isinstance(
            self.residual, ProgramWorldResidualQuestion
        ):
            raise ProgramWorldCegisError("residual must be ProgramWorldResidualQuestion")
        if self.model_call_count != 0:
            raise ProgramWorldCegisAuthorityError(
                "program-world CEGIS cannot invoke a model"
            )
        object.__setattr__(self, "model_call_count", 0)
        if self.proposal_only is not True:
            raise ProgramWorldCegisAuthorityError("repair receipts must remain proposal-only")
        if self.independently_admitted:
            raise ProgramWorldCegisAuthorityError("CEGIS cannot self-admit a repair")
        if (
            self.grants_write_authority
            or self.grants_semantic_authority
            or self.grants_proof_authority
            or self.grants_completion_authority
        ):
            raise ProgramWorldCegisAuthorityError("CEGIS cannot grant authority")
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "independently_admitted", False)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_semantic_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_completion_authority", False)
        object.__setattr__(
            self,
            "failure_fingerprints",
            _ids(self.failure_fingerprints, "failure_fingerprints"),
        )
        object.__setattr__(
            self,
            "preserved_evidence_cids",
            _ids(self.preserved_evidence_cids, "preserved_evidence_cids"),
        )

    @property
    def unique(self) -> bool:
        return self.disposition is ProgramWorldRepairDisposition.UNIQUE

    @property
    def selected_sketch(self) -> ProgramWorldPatchSketch | None:
        if self.application is None:
            return None
        return self.application.sketch

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "operator_registry_interface": PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE,
            "evidence": SAWM_CEGIS_REPAIR_EVIDENCE,
            "task_id": TASK_ID,
            "producer_id": self.producer_id,
            "disposition": self.disposition.value,
            "path": self.path,
            "reason_codes": list(self.reason_codes),
            "analytical_first": True,
            "procedures_attempted": list(self.procedures_attempted),
            "selected_procedure": self.selected_procedure,
            "evidence_cid": self.evidence_cid,
            "counterexample_id": self.counterexample_id,
            "bounds": self.bounds.to_dict(),
            "application": None if self.application is None else self.application.to_dict(),
            "residual": None if self.residual is None else self.residual.to_dict(),
            "iterations": [item.to_dict() for item in self.iterations],
            "gate_results": [item.to_dict() for item in self.gate_results],
            "failure_fingerprints": list(self.failure_fingerprints),
            "preserved_evidence_cids": list(self.preserved_evidence_cids),
            "model_call_count": 0,
            "proposal_only": True,
            "independently_admitted": False,
            "grants_write_authority": False,
            "grants_semantic_authority": False,
            "grants_proof_authority": False,
            "grants_completion_authority": False,
        }


def _attempted_after_source(
    application: ProgramWorldOperatorApplication,
    request: ProgramWorldRepairRequest,
    before_source: str,
) -> str:
    """Recover the bytes a failed candidate actually tried to emit."""

    if application.after_source:
        return application.after_source
    before_span = request.before_span
    after_span = request.after_span
    reconstructable = {
        ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
        ProgramWorldOperatorKind.REPLACE_UNIQUE_REGISTRATION,
        ProgramWorldOperatorKind.FINITE_ADAPTER,
    }
    if (
        before_span
        and after_span
        and before_span in before_source
        and application.operator_kind in reconstructable
    ):
        return before_source.replace(before_span, after_span, 1)
    return before_source


class RepairCandidateScopeGate:
    """Scope/type/effect/proof/test/authority gate over one sketch."""

    INTERFACE: ClassVar[str] = REPAIR_CANDIDATE_SCOPE_GATE_INTERFACE

    def evaluate(
        self,
        application: ProgramWorldOperatorApplication,
        request: ProgramWorldRepairRequest,
        *,
        before_source: str,
    ) -> tuple[ProgramWorldGateResult, ...]:
        path = application.path
        protected = tuple(DEFAULT_PROTECTED_PATHS) + request.protected_paths
        results: list[ProgramWorldGateResult] = []

        scope_reasons: list[str] = []
        if not request.admitted_scope:
            scope_reasons.append("scope_unspecified")
        elif not path_is_admitted(path, request.admitted_scope):
            scope_reasons.append("path_not_in_admitted_scope")
        forbidden = detect_authority_claims(request.metadata)
        scope_keys = tuple(item for item in forbidden if item.startswith("scope_key:"))
        scope_reasons.extend(scope_keys)
        results.append(
            ProgramWorldGateResult(
                kind=ProgramWorldGateKind.SCOPE,
                verdict=(
                    ProgramWorldGateVerdict.FAIL if scope_reasons else ProgramWorldGateVerdict.PASS
                ),
                reasons=tuple(scope_reasons),
            )
        )

        protected_reasons: list[str] = []
        if path_is_protected(path, protected):
            protected_reasons.append("protected_path")
        if path_is_protected(path, request.trusted_paths) and not is_test_path(path):
            protected_reasons.append("trusted_path")
        results.append(
            ProgramWorldGateResult(
                kind=ProgramWorldGateKind.PROTECTED_PATH,
                verdict=(
                    ProgramWorldGateVerdict.FAIL
                    if protected_reasons
                    else ProgramWorldGateVerdict.PASS
                ),
                reasons=tuple(protected_reasons),
            )
        )

        after_source = _attempted_after_source(application, request, before_source)
        type_reasons: list[str] = []
        if _parse_python(before_source) is not None and _parse_python(after_source) is None:
            type_reasons.append("type_gate:after_unparseable")
        results.append(
            ProgramWorldGateResult(
                kind=ProgramWorldGateKind.TYPE,
                verdict=(
                    ProgramWorldGateVerdict.FAIL if type_reasons else ProgramWorldGateVerdict.PASS
                ),
                reasons=tuple(type_reasons),
            )
        )

        effect_reasons = list(new_execution_risks(before_source, after_source))
        for reason in application.reason_codes:
            if reason.startswith("arbitrary_execution") and reason not in effect_reasons:
                effect_reasons.append(reason)
        results.append(
            ProgramWorldGateResult(
                kind=ProgramWorldGateKind.EFFECT,
                verdict=(
                    ProgramWorldGateVerdict.FAIL
                    if effect_reasons
                    else ProgramWorldGateVerdict.PASS
                ),
                reasons=tuple(effect_reasons),
            )
        )

        proof_reasons = [
            item
            for item in forbidden
            if item.startswith("authority_claim:") or item.startswith("obligation_waiver:")
        ]
        if application.grants_proof_authority or application.independently_admitted:
            proof_reasons.append("proof_self_admission")
        if request.proof_obligation_ids and application.accepted_sketch:
            if set(request.proof_obligation_ids) - set(application.proof_obligation_ids):
                # Sketches inherit obligations; missing ones are not a waiver if empty inherit.
                pass
        results.append(
            ProgramWorldGateResult(
                kind=ProgramWorldGateKind.PROOF,
                verdict=(
                    ProgramWorldGateVerdict.FAIL if proof_reasons else ProgramWorldGateVerdict.PASS
                ),
                reasons=tuple(proof_reasons),
            )
        )

        test_reasons = list(detect_test_weakening(before_source, after_source, path))
        results.append(
            ProgramWorldGateResult(
                kind=ProgramWorldGateKind.TEST,
                verdict=(
                    ProgramWorldGateVerdict.FAIL if test_reasons else ProgramWorldGateVerdict.PASS
                ),
                reasons=tuple(test_reasons),
            )
        )

        authority_reasons = [
            item for item in forbidden if item.startswith("authority_claim:")
        ]
        if application.grants_write_authority or application.grants_completion_authority:
            authority_reasons.append("authority_claim:application")
        results.append(
            ProgramWorldGateResult(
                kind=ProgramWorldGateKind.AUTHORITY,
                verdict=(
                    ProgramWorldGateVerdict.FAIL
                    if authority_reasons
                    else ProgramWorldGateVerdict.PASS
                ),
                reasons=tuple(authority_reasons),
            )
        )
        return tuple(results)


class RepairCounterexampleRefiner:
    """Monotonic counterexample refinement from failed candidate evidence."""

    INTERFACE: ClassVar[str] = REPAIR_COUNTEREXAMPLE_REFINER_INTERFACE

    def refine(
        self,
        counterexample: ProgramWorldCounterexample,
        *,
        failed_application: ProgramWorldOperatorApplication | None = None,
        gate_results: Sequence[ProgramWorldGateResult] = (),
    ) -> tuple[ProgramWorldCounterexample, bool]:
        abstract = list(counterexample.abstract_markers)
        model = list(counterexample.model_markers)
        tests = list(counterexample.test_markers)
        proofs = list(counterexample.proof_markers)
        types = list(counterexample.type_markers)
        effects = list(counterexample.effect_markers)

        def _add(bucket: list[str], item: str) -> None:
            if item and item not in bucket:
                bucket.append(item)

        def _note(kind: ProgramWorldGateKind, reason: str) -> None:
            marker = f"gate_fail:{kind.value}"
            if kind is ProgramWorldGateKind.TEST:
                _add(tests, marker)
                _add(tests, reason)
            elif kind is ProgramWorldGateKind.PROOF:
                _add(proofs, marker)
                _add(proofs, reason)
            elif kind is ProgramWorldGateKind.TYPE:
                _add(types, marker)
                _add(types, reason)
            elif kind is ProgramWorldGateKind.EFFECT:
                _add(effects, marker)
                _add(effects, reason)
            else:
                _add(abstract, marker)
                _add(abstract, reason)

        for gate in gate_results:
            if gate.verdict is ProgramWorldGateVerdict.PASS:
                continue
            marker = f"gate_fail:{gate.kind.value}"
            if gate.kind is ProgramWorldGateKind.TEST:
                _add(tests, marker)
                for reason in gate.reasons:
                    _add(tests, reason)
            elif gate.kind is ProgramWorldGateKind.PROOF:
                _add(proofs, marker)
                for reason in gate.reasons:
                    _add(proofs, reason)
            elif gate.kind is ProgramWorldGateKind.TYPE:
                _add(types, marker)
                for reason in gate.reasons:
                    _add(types, reason)
            elif gate.kind is ProgramWorldGateKind.EFFECT:
                _add(effects, marker)
                for reason in gate.reasons:
                    _add(effects, reason)
            else:
                _add(abstract, marker)
                for reason in gate.reasons:
                    _add(abstract, reason)
        if failed_application is not None:
            _add(
                abstract,
                f"rejected:{failed_application.operator_kind.value}:{failed_application.disposition.value}",
            )
            for reason in failed_application.reason_codes:
                if reason.startswith("test_weakening") or reason == "cannot_weaken_tests":
                    _note(ProgramWorldGateKind.TEST, reason)
                elif reason.startswith("arbitrary_execution"):
                    _note(ProgramWorldGateKind.EFFECT, reason)
                elif reason.startswith("type_gate"):
                    _note(ProgramWorldGateKind.TYPE, reason)
                elif reason.startswith("obligation_waiver") or reason.startswith("proof_"):
                    _note(ProgramWorldGateKind.PROOF, reason)
                elif reason.startswith("authority_claim"):
                    _note(ProgramWorldGateKind.AUTHORITY, reason)
                elif reason in {"protected_path", "trusted_path"}:
                    _note(ProgramWorldGateKind.PROTECTED_PATH, reason)
                else:
                    _add(abstract, reason)
        refined = ProgramWorldCounterexample(
            counterexample_id=counterexample.counterexample_id,
            path=counterexample.path,
            family=counterexample.family,
            witness=counterexample.witness,
            evidence_cid="",
            abstract_markers=tuple(abstract),
            model_markers=tuple(model),
            test_markers=tuple(tests),
            proof_markers=tuple(proofs),
            type_markers=tuple(types),
            effect_markers=tuple(effects),
        )
        progress = refined.evidence_cid != counterexample.evidence_cid
        if not progress:
            return counterexample, False
        return refined, True


def _gates_passed(results: Sequence[ProgramWorldGateResult]) -> bool:
    return all(item.verdict is ProgramWorldGateVerdict.PASS for item in results)


def _closed_kinds(
    request: ProgramWorldRepairRequest,
    registry: ProgramWorldRepairOperatorRegistry,
) -> tuple[ProgramWorldOperatorKind, ...]:
    if request.operator_kinds:
        kinds: list[ProgramWorldOperatorKind] = []
        for item in request.operator_kinds:
            try:
                kind = _enum(item, ProgramWorldOperatorKind, "operator_kind")
            except ProgramWorldOperatorError as exc:
                raise ProgramWorldCegisAuthorityError(
                    "requested operator is outside the closed vocabulary"
                ) from exc
            if not registry.contains(kind):
                raise ProgramWorldCegisAuthorityError(
                    "requested operator is outside the closed vocabulary"
                )
            kinds.append(kind)
        return tuple(kinds)
    return registry.kinds()


def _apply_kind(
    kind: ProgramWorldOperatorKind,
    request: ProgramWorldRepairRequest,
    registry: ProgramWorldRepairOperatorRegistry,
    *,
    overlay: Mapping[str, Any] | None = None,
) -> ProgramWorldOperatorApplication:
    kwargs = request.operator_kwargs()
    if overlay:
        kwargs.update({key: value for key, value in overlay.items() if value not in (None, "")})
    kwargs["operator_kind"] = kind
    kwargs["registry"] = registry
    return apply_program_world_repair_operator(**kwargs)


def _model_retry_blocked(request: ProgramWorldRepairRequest) -> bool:
    if request.model_proposal is None:
        return False
    evidence = request.counterexample.evidence_cid
    if evidence in request.prior_evidence_cids:
        fingerprint = failure_fingerprint(
            counterexample_id=request.counterexample.counterexample_id,
            operator_kind=MODEL_STRATEGY,
            sketch_cid=str(request.model_proposal.get("sketch_cid") or ""),
            evidence_cid=evidence,
            strategy=MODEL_STRATEGY,
        )
        if fingerprint in request.prior_failure_fingerprints:
            return True
        # Same evidence without a strategy change is still a blocked model retry.
        return True
    fingerprint = failure_fingerprint(
        counterexample_id=request.counterexample.counterexample_id,
        operator_kind=MODEL_STRATEGY,
        sketch_cid=str(request.model_proposal.get("sketch_cid") or ""),
        evidence_cid=evidence,
        strategy=MODEL_STRATEGY,
    )
    return fingerprint in request.prior_failure_fingerprints


class ProgramWorldCEGIS:
    """Bounded analytical-first CEGIS over closed program-world operators."""

    INTERFACE: ClassVar[str] = PROGRAM_WORLD_CEGIS_INTERFACE

    def __init__(
        self,
        *,
        registry: ProgramWorldRepairOperatorRegistry | None = None,
        scope_gate: RepairCandidateScopeGate | None = None,
        refiner: RepairCounterexampleRefiner | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._registry = registry or ProgramWorldRepairOperatorRegistry()
        self._gate = scope_gate or RepairCandidateScopeGate()
        self._refiner = refiner or RepairCounterexampleRefiner()
        self._clock = monotonic_clock or time.monotonic

    @property
    def registry(self) -> ProgramWorldRepairOperatorRegistry:
        return self._registry

    def synthesize(self, request: ProgramWorldRepairRequest) -> ProgramWorldRepairReceipt:
        if not isinstance(request, ProgramWorldRepairRequest):
            raise ProgramWorldCegisError("request must be ProgramWorldRepairRequest")
        start = self._clock()
        procedures: list[str] = []
        iterations: list[ProgramWorldCegisIteration] = []
        fingerprints: list[str] = list(request.prior_failure_fingerprints)
        preserved: list[str] = [request.counterexample.evidence_cid]
        preserved.extend(request.prior_evidence_cids)
        gate_log: list[ProgramWorldGateResult] = []
        counterexample = request.counterexample
        kinds = _closed_kinds(request, self._registry)
        request_claims = detect_authority_claims(request.metadata)
        if request_claims:
            return ProgramWorldRepairReceipt(
                disposition=ProgramWorldRepairDisposition.REJECTED,
                path=request.path,
                reason_codes=request_claims,
                analytical_first=True,
                procedures_attempted=(ANALYTICAL_PROCEDURE,),
                selected_procedure=ANALYTICAL_PROCEDURE,
                evidence_cid=counterexample.evidence_cid,
                counterexample_id=counterexample.counterexample_id,
                bounds=request.bounds,
                failure_fingerprints=tuple(fingerprints),
                preserved_evidence_cids=tuple(dict.fromkeys(preserved)),
            )

        def _time_exceeded() -> bool:
            return int((self._clock() - start) * 1000) > request.bounds.wall_time_ms

        def _record_failure(
            application: ProgramWorldOperatorApplication,
            *,
            strategy: str,
        ) -> str:
            sketch_cid = application.sketch.sketch_cid if application.sketch is not None else ""
            fingerprint = failure_fingerprint(
                counterexample_id=counterexample.counterexample_id,
                operator_kind=application.operator_kind.value,
                sketch_cid=sketch_cid,
                evidence_cid=counterexample.evidence_cid,
                strategy=strategy,
            )
            if fingerprint not in fingerprints:
                fingerprints.append(fingerprint)
            if counterexample.evidence_cid not in preserved:
                preserved.append(counterexample.evidence_cid)
            return fingerprint

        def _accept(
            application: ProgramWorldOperatorApplication,
            *,
            procedure: str,
            extra_reasons: Sequence[str] = (),
            last_gates: Sequence[ProgramWorldGateResult] = (),
        ) -> ProgramWorldRepairReceipt:
            disposition = (
                ProgramWorldRepairDisposition.UNIQUE
                if application.unique
                else ProgramWorldRepairDisposition.SUPPORTED
            )
            return ProgramWorldRepairReceipt(
                disposition=disposition,
                path=request.path,
                reason_codes=tuple(
                    dict.fromkeys(
                        (
                            *application.reason_codes,
                            *extra_reasons,
                            "analytical_first",
                            "proposal_only",
                        )
                    )
                ),
                analytical_first=True,
                procedures_attempted=tuple(procedures),
                selected_procedure=procedure,
                evidence_cid=counterexample.evidence_cid,
                counterexample_id=counterexample.counterexample_id,
                bounds=request.bounds,
                application=application,
                iterations=tuple(iterations),
                gate_results=tuple(last_gates) or tuple(gate_log),
                failure_fingerprints=tuple(fingerprints),
                preserved_evidence_cids=tuple(dict.fromkeys(preserved)),
            )

        def _terminal(
            disposition: ProgramWorldRepairDisposition,
            *,
            procedure: str,
            reasons: Sequence[str],
            residual: ProgramWorldResidualQuestion | None = None,
        ) -> ProgramWorldRepairReceipt:
            return ProgramWorldRepairReceipt(
                disposition=disposition,
                path=request.path,
                reason_codes=tuple(dict.fromkeys(reasons)),
                analytical_first=True,
                procedures_attempted=tuple(procedures) or (ANALYTICAL_PROCEDURE,),
                selected_procedure=procedure,
                evidence_cid=counterexample.evidence_cid,
                counterexample_id=counterexample.counterexample_id,
                bounds=request.bounds,
                residual=residual,
                iterations=tuple(iterations),
                gate_results=tuple(gate_log),
                failure_fingerprints=tuple(fingerprints),
                preserved_evidence_cids=tuple(dict.fromkeys(preserved)),
            )

        def _consider(
            application: ProgramWorldOperatorApplication,
            *,
            procedure: str,
            strategy: str,
        ) -> ProgramWorldRepairReceipt | None:
            nonlocal counterexample
            gates = self._gate.evaluate(
                application, request, before_source=request.source_text
            )
            gate_log.extend(gates)
            if application.accepted_sketch and _gates_passed(gates):
                return _accept(application, procedure=procedure, last_gates=gates)
            _record_failure(application, strategy=strategy)
            refined, progress = self._refiner.refine(
                counterexample,
                failed_application=application,
                gate_results=gates,
            )
            if progress:
                counterexample = refined
                preserved.append(counterexample.evidence_cid)
            return None

        # 1. Analytical unique operator for the defect family.
        procedures.append(ANALYTICAL_PROCEDURE)
        family_kinds = operators_for_family(counterexample.family)
        unique_hits: list[ProgramWorldOperatorApplication] = []
        for kind in family_kinds:
            if kind not in kinds:
                continue
            application = _apply_kind(kind, request, self._registry)
            if application.unique:
                unique_hits.append(application)
            elif application.accepted_sketch:
                unique_hits.append(application)
            else:
                _record_failure(application, strategy=ANALYTICAL_PROCEDURE)
        if len(unique_hits) == 1:
            accepted = _consider(
                unique_hits[0], procedure=ANALYTICAL_PROCEDURE, strategy=ANALYTICAL_PROCEDURE
            )
            if accepted is not None:
                return accepted
        elif len(unique_hits) > 1:
            sketch_ids = {
                item.sketch.sketch_cid for item in unique_hits if item.sketch is not None
            }
            if len(sketch_ids) == 1:
                accepted = _consider(
                    unique_hits[0],
                    procedure=ANALYTICAL_PROCEDURE,
                    strategy=ANALYTICAL_PROCEDURE,
                )
                if accepted is not None:
                    return accepted
            iterations.append(
                ProgramWorldCegisIteration(
                    iteration=0,
                    outcome=ProgramWorldIterationOutcome.CANDIDATE_REJECTED,
                    procedure=ANALYTICAL_PROCEDURE,
                    evidence_cid=counterexample.evidence_cid,
                    candidate_kinds=tuple(item.operator_kind.value for item in unique_hits),
                    reason_codes=("analytical_not_unique",),
                )
            )

        # 2. Equality normalization when terms or family request it.
        if (
            ProgramWorldOperatorKind.EQUALITY_REWRITE in kinds
            and (
                counterexample.family is ProgramWorldDefectFamily.EQUALITY_DEBT
                or request.source_term
                or request.target_term
            )
        ):
            procedures.append(EQUALITY_PROCEDURE)
            application = _apply_kind(
                ProgramWorldOperatorKind.EQUALITY_REWRITE, request, self._registry
            )
            accepted = _consider(
                application, procedure=EQUALITY_PROCEDURE, strategy=EQUALITY_PROCEDURE
            )
            if accepted is not None:
                return accepted

        # 3. Bounded CEGIS enumeration.
        procedures.append(CEGIS_PROCEDURE)
        identical_counts: dict[str, int] = {}
        previous_evidence = ""
        for iteration_index in range(1, request.bounds.max_iterations + 1):
            if _time_exceeded():
                iterations.append(
                    ProgramWorldCegisIteration(
                        iteration=iteration_index,
                        outcome=ProgramWorldIterationOutcome.BUDGET_EXHAUSTED,
                        procedure=CEGIS_PROCEDURE,
                        evidence_cid=counterexample.evidence_cid,
                        reason_codes=("wall_time_exhausted",),
                    )
                )
                return _terminal(
                    ProgramWorldRepairDisposition.BUDGET_EXHAUSTED,
                    procedure=CEGIS_PROCEDURE,
                    reasons=("wall_time_exhausted", "evidence_preserved"),
                    residual=_residual(counterexample, "wall_time_exhausted"),
                )
            if previous_evidence == counterexample.evidence_cid and iteration_index > 1:
                iterations.append(
                    ProgramWorldCegisIteration(
                        iteration=iteration_index,
                        outcome=ProgramWorldIterationOutcome.REPEATED_STATE,
                        procedure=CEGIS_PROCEDURE,
                        evidence_cid=counterexample.evidence_cid,
                        reason_codes=("repeated_state",),
                    )
                )
                return _terminal(
                    ProgramWorldRepairDisposition.ABSTAINED,
                    procedure=CEGIS_PROCEDURE,
                    reasons=("repeated_state", "evidence_preserved"),
                    residual=_residual(counterexample, "repeated_state"),
                )
            previous_evidence = counterexample.evidence_cid
            produced: list[ProgramWorldOperatorApplication] = []
            rejected_here: list[str] = []
            for kind in kinds:
                if len(produced) >= request.bounds.max_candidates_per_iteration:
                    break
                application = _apply_kind(kind, request, self._registry)
                sketch_cid = (
                    application.sketch.sketch_cid if application.sketch is not None else ""
                )
                fingerprint = failure_fingerprint(
                    counterexample_id=counterexample.counterexample_id,
                    operator_kind=kind.value,
                    sketch_cid=sketch_cid,
                    evidence_cid=counterexample.evidence_cid,
                    strategy=CEGIS_PROCEDURE,
                )
                if fingerprint in fingerprints:
                    identical_counts[fingerprint] = identical_counts.get(fingerprint, 1) + 1
                    if identical_counts[fingerprint] >= request.bounds.max_identical_failures:
                        iterations.append(
                            ProgramWorldCegisIteration(
                                iteration=iteration_index,
                                outcome=ProgramWorldIterationOutcome.IDENTICAL_FAILURE,
                                procedure=CEGIS_PROCEDURE,
                                evidence_cid=counterexample.evidence_cid,
                                rejected_fingerprints=(fingerprint,),
                                reason_codes=("identical_failure_terminated",),
                            )
                        )
                        return _terminal(
                            ProgramWorldRepairDisposition.IDENTICAL_FAILURE,
                            procedure=CEGIS_PROCEDURE,
                            reasons=(
                                "identical_failure_terminated",
                                "evidence_preserved",
                                "strategy_change_required",
                            ),
                            residual=_residual(
                                counterexample, "identical_failure_terminated"
                            ),
                        )
                    continue
                if not application.accepted_sketch:
                    rejected_here.append(_record_failure(application, strategy=CEGIS_PROCEDURE))
                    continue
                produced.append(application)
            if not produced:
                iterations.append(
                    ProgramWorldCegisIteration(
                        iteration=iteration_index,
                        outcome=ProgramWorldIterationOutcome.NO_CANDIDATE,
                        procedure=CEGIS_PROCEDURE,
                        evidence_cid=counterexample.evidence_cid,
                        rejected_fingerprints=tuple(rejected_here),
                        reason_codes=("no_admissible_candidate",),
                    )
                )
                # No candidate this round: stop rather than spin.
                break
            selected_kinds: list[str] = []
            progressed = False
            for application in produced:
                selected_kinds.append(application.operator_kind.value)
                accepted = _consider(
                    application, procedure=CEGIS_PROCEDURE, strategy=CEGIS_PROCEDURE
                )
                if accepted is not None:
                    iterations.append(
                        ProgramWorldCegisIteration(
                            iteration=iteration_index,
                            outcome=ProgramWorldIterationOutcome.SELECTED,
                            procedure=CEGIS_PROCEDURE,
                            evidence_cid=counterexample.evidence_cid,
                            candidate_kinds=tuple(selected_kinds),
                            reason_codes=("cegis_selected",),
                        )
                    )
                    return ProgramWorldRepairReceipt(
                        disposition=accepted.disposition,
                        path=accepted.path,
                        reason_codes=accepted.reason_codes,
                        analytical_first=True,
                        procedures_attempted=accepted.procedures_attempted,
                        selected_procedure=accepted.selected_procedure,
                        evidence_cid=accepted.evidence_cid,
                        counterexample_id=accepted.counterexample_id,
                        bounds=accepted.bounds,
                        application=accepted.application,
                        iterations=tuple(iterations),
                        gate_results=accepted.gate_results,
                        failure_fingerprints=accepted.failure_fingerprints,
                        preserved_evidence_cids=accepted.preserved_evidence_cids,
                    )
                progressed = True
            iterations.append(
                ProgramWorldCegisIteration(
                    iteration=iteration_index,
                    outcome=(
                        ProgramWorldIterationOutcome.REFINED
                        if progressed
                        else ProgramWorldIterationOutcome.CANDIDATE_REJECTED
                    ),
                    procedure=CEGIS_PROCEDURE,
                    evidence_cid=counterexample.evidence_cid,
                    candidate_kinds=tuple(selected_kinds),
                    rejected_fingerprints=tuple(rejected_here),
                    reason_codes=("candidate_rejected",),
                )
            )
        else:
            return _terminal(
                ProgramWorldRepairDisposition.BUDGET_EXHAUSTED,
                procedure=CEGIS_PROCEDURE,
                reasons=("max_iterations", "evidence_preserved"),
                residual=_residual(counterexample, "max_iterations"),
            )

        # 4. Residual / model proposal only after deterministic exhaustion.
        procedures.append(RESIDUAL_PROCEDURE)
        if request.model_proposal is not None:
            if _model_retry_blocked(request):
                return _terminal(
                    ProgramWorldRepairDisposition.REJECTED,
                    procedure=RESIDUAL_PROCEDURE,
                    reasons=(
                        "identical_model_retry_without_new_evidence",
                        "evidence_preserved",
                        "strategy_change_required",
                    ),
                    residual=_residual(
                        counterexample, "identical_model_retry_without_new_evidence"
                    ),
                )
            proposal = dict(request.model_proposal)
            forbidden = detect_authority_claims(proposal)
            if forbidden:
                return _terminal(
                    ProgramWorldRepairDisposition.REJECTED,
                    procedure=RESIDUAL_PROCEDURE,
                    reasons=forbidden + ("model_proposal_cannot_self_approve",),
                    residual=_residual(counterexample, "model_authority_claim"),
                )
            requested_kind = proposal.get("operator_kind") or proposal.get("kind")
            if requested_kind:
                try:
                    kind = _enum(requested_kind, ProgramWorldOperatorKind, "operator_kind")
                except ProgramWorldOperatorError:
                    return _terminal(
                        ProgramWorldRepairDisposition.REJECTED,
                        procedure=RESIDUAL_PROCEDURE,
                        reasons=("grammar_expansion", "model_proposal_outside_vocabulary"),
                        residual=_residual(counterexample, "grammar_expansion"),
                    )
            else:
                family_kinds = operators_for_family(counterexample.family)
                kind = family_kinds[0] if family_kinds else ProgramWorldOperatorKind.REPLACE_EXACT_BYTES
            overlay = {
                "before_span": proposal.get("before_span") or request.before_span,
                "after_span": proposal.get("after_span") or request.after_span,
                "symbol": proposal.get("symbol") or request.symbol,
                "replacement": proposal.get("replacement") or request.replacement,
                "argument": proposal.get("argument") or request.argument,
                "import_module": proposal.get("import_module") or request.import_module,
                "import_name": proposal.get("import_name") or request.import_name,
                "source_term": proposal.get("source_term") or request.source_term,
                "target_term": proposal.get("target_term") or request.target_term,
            }
            application = _apply_kind(kind, request, self._registry, overlay=overlay)
            accepted = _consider(
                application, procedure=RESIDUAL_PROCEDURE, strategy=MODEL_STRATEGY
            )
            if accepted is not None:
                # Model output remains proposal-only and cannot self-approve.
                return ProgramWorldRepairReceipt(
                    disposition=ProgramWorldRepairDisposition.SUPPORTED,
                    path=accepted.path,
                    reason_codes=tuple(
                        dict.fromkeys(
                            (
                                *accepted.reason_codes,
                                "model_proposal_only",
                                "cannot_self_approve",
                                "deterministic_routes_exhausted",
                            )
                        )
                    ),
                    analytical_first=True,
                    procedures_attempted=accepted.procedures_attempted,
                    selected_procedure=RESIDUAL_PROCEDURE,
                    evidence_cid=accepted.evidence_cid,
                    counterexample_id=accepted.counterexample_id,
                    bounds=accepted.bounds,
                    application=accepted.application,
                    iterations=accepted.iterations,
                    gate_results=accepted.gate_results,
                    failure_fingerprints=accepted.failure_fingerprints,
                    preserved_evidence_cids=accepted.preserved_evidence_cids,
                )
            return _terminal(
                ProgramWorldRepairDisposition.REJECTED,
                procedure=RESIDUAL_PROCEDURE,
                reasons=("model_proposal_rejected", "evidence_preserved"),
                residual=_residual(counterexample, "model_proposal_rejected"),
            )

        return _terminal(
            ProgramWorldRepairDisposition.RESIDUAL,
            procedure=RESIDUAL_PROCEDURE,
            reasons=("deterministic_routes_exhausted", "evidence_preserved"),
            residual=_residual(counterexample, "deterministic_routes_exhausted"),
        )


def _residual(counterexample: ProgramWorldCounterexample, reason: str) -> ProgramWorldResidualQuestion:
    return ProgramWorldResidualQuestion(
        question_id=f"residual:{counterexample.counterexample_id}:{reason}",
        reason=reason,
        decision_relevant=True,
        model_may_propose=True,
        model_may_self_approve=False,
        requires_new_evidence=True,
    )


def synthesize_program_world_repair(
    request: ProgramWorldRepairRequest | Mapping[str, Any] | None = None,
    *,
    registry: ProgramWorldRepairOperatorRegistry | None = None,
    monotonic_clock: Callable[[], float] | None = None,
    **kwargs: Any,
) -> ProgramWorldRepairReceipt:
    """Run analytical-first bounded CEGIS for one program-world repair."""

    clock = kwargs.pop("monotonic_clock", monotonic_clock)
    bound_registry = kwargs.pop("registry", registry)
    if request is None:
        request = ProgramWorldRepairRequest.from_mapping(kwargs)
    elif isinstance(request, Mapping):
        payload = dict(request)
        payload.update(kwargs)
        request = ProgramWorldRepairRequest.from_mapping(payload)
    elif kwargs:
        raise ProgramWorldCegisError("cannot mix a request object with keyword overlays")
    engine = ProgramWorldCEGIS(registry=bound_registry, monotonic_clock=clock)
    return engine.synthesize(request)
