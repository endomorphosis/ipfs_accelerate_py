"""Bounded program-world CEGIS over closed repair operators (SAWM-021).

Interface: ``ProgramWorldCEGIS@1``
Evidence: ``sawm/cegis-repair@1``

Counterexample-guided synthesis of typed patch sketches.  Analytical unique
operators run first; bounded CEGIS enumerates the remaining closed grammar;
unsupported residuals become explicit questions.  Model sketches are
proposal-only, cannot self-approve, and cannot retry an identical failure
without new evidence.

This module does not create a second patch engine, shell executor, validation
gate, or acceptance authority.  Importing it performs no I/O.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

from ..proof.formal_verification_contracts import CanonicalContract
from .program_world_operators import (
    CLOSED_OPERATOR_VOCABULARY,
    MAX_COLLECTION_ITEMS,
    MAX_SOURCE_BYTES,
    MAX_SPAN_BYTES,
    PROGRAM_WORLD_CEGIS_INTERFACE,
    SAWM_CEGIS_REPAIR_EVIDENCE,
    OperatorApplicationDisposition,
    OperatorApplicationReceipt,
    OperatorReason,
    ProgramWorldRepairAuthorityError,
    ProgramWorldRepairBoundsError,
    ProgramWorldRepairError,
    ProgramWorldRepairOperatorKind,
    ProgramWorldRepairOperatorRegistry,
    ProgramWorldRepairStaleError,
    RepairCandidateScopeGate,
    TypedPatchSketch,
    _assert_current_binding,
    _cid,
    _enum,
    _identity_cid,
    _ids,
    _normalize_path,
    _optional_cid,
    _plain,
    _reason_codes,
    _source_cid,
    _text,
    apply_program_world_repair_operator,
    build_default_program_world_operator_registry,
    identifier_occurs,
)


PROGRAM_WORLD_CEGIS_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-cegis-bounds@1"
)
PROGRAM_WORLD_COUNTEREXAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-counterexample@1"
)
PROGRAM_WORLD_CEGIS_ROUND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-cegis-round@1"
)
PROGRAM_WORLD_REPAIR_SYNTHESIS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-synthesis@1"
)
FAILED_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-failed-attempt@1"
)

PRODUCER_ID: Final[str] = "program-world-cegis@1"
CEGIS_VERSION: Final[str] = "1"

HARD_MAX_ITERATIONS: Final[int] = 32
HARD_MAX_CANDIDATES: Final[int] = 32
HARD_MAX_IDENTICAL_FAILURES: Final[int] = 8
HARD_MAX_WALL_MS: Final[int] = 60_000
DEFAULT_MAX_ITERATIONS: Final[int] = 8
DEFAULT_MAX_CANDIDATES: Final[int] = 8
DEFAULT_MAX_IDENTICAL_FAILURES: Final[int] = 2
DEFAULT_MAX_WALL_MS: Final[int] = 5_000

ANALYTICAL_OPERATOR_ORDER: Final[tuple[str, ...]] = (
    ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN.value,
    ProgramWorldRepairOperatorKind.EQUALITY_REWRITE.value,
    ProgramWorldRepairOperatorKind.RENAME_EXACT_SYMBOL.value,
    ProgramWorldRepairOperatorKind.ADD_UNIQUE_IMPORT.value,
    ProgramWorldRepairOperatorKind.ADD_UNIQUE_ARGUMENT.value,
    ProgramWorldRepairOperatorKind.ADD_UNIQUE_REGISTRATION.value,
    ProgramWorldRepairOperatorKind.RESTORE_TRACKED_ARTIFACT.value,
)


class ProgramWorldCegisError(ProgramWorldRepairError):
    """Bounded CEGIS inputs cannot produce a trustworthy synthesis."""


class ProgramWorldCegisAuthorityError(ProgramWorldRepairAuthorityError):
    """A CEGIS candidate attempted to mint authority or retry without evidence."""


class ProgramWorldCegisBoundsError(ProgramWorldRepairBoundsError):
    """A CEGIS bound would be exceeded."""


class CegisDisposition(str, Enum):
    """Closed outcomes for one program-world repair synthesis."""

    UNIQUE_REPAIR = "unique_repair"
    SYNTHESIZED = "synthesized"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RESIDUAL = "residual"
    IDENTICAL_RETRY_BLOCKED = "identical_retry_blocked"


class CegisRoute(str, Enum):
    """Closed cascade routes.  Analytical always runs first."""

    ANALYTICAL_UNIQUE = "analytical_unique"
    BOUNDED_CEGIS = "bounded_cegis"
    RESIDUAL_QUESTION = "residual_question"
    IDENTICAL_RETRY_BLOCKED = "identical_retry_blocked"
    REJECTED = "rejected"


class CegisRoundDisposition(str, Enum):
    """Closed per-round outcomes."""

    APPLIED = "applied"
    REJECTED = "rejected"
    ABSTAINED = "abstained"
    COUNTEREXAMPLE_REMAINS = "counterexample_remains"
    NEW_COUNTEREXAMPLE = "new_counterexample"
    IDENTICAL_FAILURE = "identical_failure"
    BOUND_HIT = "bound_hit"
    SKIPPED_DUPLICATE = "skipped_duplicate"


class CounterexampleKind(str, Enum):
    """Closed counterexample families that may refine a sketch."""

    SPAN = "span"
    TEST = "test"
    PROOF = "proof"
    TYPE = "type"
    EFFECT = "effect"
    SYMBOL = "symbol"
    IMPORT = "import"
    REGISTRATION = "registration"


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise ProgramWorldCegisError(f"{name} must be a positive integer")
    if value > maximum:
        raise ProgramWorldCegisBoundsError(f"{name} exceeds bound {maximum}")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ProgramWorldCegisError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class CegisBounds(CanonicalContract):
    """Finite iteration, candidate, repetition, and wall-time budgets."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_CEGIS_BOUNDS_SCHEMA

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    max_identical_failures: int = DEFAULT_MAX_IDENTICAL_FAILURES
    max_wall_time_ms: int = DEFAULT_MAX_WALL_MS
    max_model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_iterations",
            _positive_int(self.max_iterations, "max_iterations", maximum=HARD_MAX_ITERATIONS),
        )
        object.__setattr__(
            self,
            "max_candidates",
            _positive_int(self.max_candidates, "max_candidates", maximum=HARD_MAX_CANDIDATES),
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
            "max_wall_time_ms",
            _positive_int(self.max_wall_time_ms, "max_wall_time_ms", maximum=HARD_MAX_WALL_MS),
        )
        if self.max_model_calls != 0:
            raise ProgramWorldCegisAuthorityError(
                "program-world CEGIS hard-zeros model calls; residual model sketches "
                "are ingested only after deterministic routes and new evidence"
            )
        object.__setattr__(self, "max_model_calls", 0)

    def _payload(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_candidates": self.max_candidates,
            "max_identical_failures": self.max_identical_failures,
            "max_wall_time_ms": self.max_wall_time_ms,
            "max_model_calls": 0,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CegisBounds":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ProgramWorldCegisError("bounds must be a mapping")
        fields = {
            name: payload[name]
            for name in ("max_iterations", "max_candidates", "max_identical_failures", "max_wall_time_ms")
            if name in payload
        }
        return cls(**fields)


@dataclass(frozen=True)
class ProgramWorldCounterexample(CanonicalContract):
    """One finite positive/negative repair obligation or witness."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_COUNTEREXAMPLE_SCHEMA

    counterexample_id: str
    kind: CounterexampleKind | str
    path: str
    evidence_cid: str
    before: str = ""
    expected_after: str = ""
    symbol_from: str = ""
    symbol_to: str = ""
    import_module: str = ""
    import_name: str = ""
    argument_name: str = ""
    argument_value: str = ""
    registration_anchor: str = ""
    registration_payload: str = ""
    obligation_id: str = ""
    witness: str = ""
    parent_id: str = ""
    resolved: bool = False
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "counterexample_id", _text(self.counterexample_id, "counterexample_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, CounterexampleKind, "kind"))
        object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(self, "evidence_cid", _cid(self.evidence_cid, "evidence_cid"))
        object.__setattr__(
            self, "before", _text(self.before, "before", empty=True, limit=MAX_SPAN_BYTES)
        )
        object.__setattr__(
            self,
            "expected_after",
            _text(self.expected_after, "expected_after", empty=True, limit=MAX_SPAN_BYTES),
        )
        object.__setattr__(
            self, "symbol_from", _text(self.symbol_from, "symbol_from", empty=True)
        )
        object.__setattr__(self, "symbol_to", _text(self.symbol_to, "symbol_to", empty=True))
        object.__setattr__(
            self, "import_module", _text(self.import_module, "import_module", empty=True)
        )
        object.__setattr__(
            self, "import_name", _text(self.import_name, "import_name", empty=True)
        )
        object.__setattr__(
            self, "argument_name", _text(self.argument_name, "argument_name", empty=True)
        )
        object.__setattr__(
            self, "argument_value", _text(self.argument_value, "argument_value", empty=True)
        )
        object.__setattr__(
            self,
            "registration_anchor",
            _text(self.registration_anchor, "registration_anchor", empty=True, limit=MAX_SPAN_BYTES),
        )
        object.__setattr__(
            self,
            "registration_payload",
            _text(
                self.registration_payload,
                "registration_payload",
                empty=True,
                limit=MAX_SPAN_BYTES,
            ),
        )
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id", empty=True)
        )
        object.__setattr__(self, "witness", _text(self.witness, "witness", empty=True))
        object.__setattr__(self, "parent_id", _text(self.parent_id, "parent_id", empty=True))
        object.__setattr__(self, "resolved", False)
        meta = _plain(dict(self.metadata or {}))
        if not isinstance(meta, dict):
            raise ProgramWorldCegisError("counterexample metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(meta))

    def _payload(self) -> dict[str, Any]:
        return {
            "counterexample_id": self.counterexample_id,
            "kind": self.kind if isinstance(self.kind, str) else self.kind.value,
            "path": self.path,
            "evidence_cid": self.evidence_cid,
            "before": self.before,
            "expected_after": self.expected_after,
            "symbol_from": self.symbol_from,
            "symbol_to": self.symbol_to,
            "import_module": self.import_module,
            "import_name": self.import_name,
            "argument_name": self.argument_name,
            "argument_value": self.argument_value,
            "registration_anchor": self.registration_anchor,
            "registration_payload": self.registration_payload,
            "obligation_id": self.obligation_id,
            "witness": self.witness,
            "parent_id": self.parent_id,
            "resolved": False,
            "metadata": dict(self.metadata),
        }

    @property
    def counterexample_cid(self) -> str:
        return self.content_id

    def with_parent(self, parent_id: str) -> "ProgramWorldCounterexample":
        payload = dict(self._payload())
        payload["parent_id"] = parent_id
        payload["counterexample_id"] = f"{self.counterexample_id}:refined"
        return ProgramWorldCounterexample.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramWorldCounterexample":
        if not isinstance(payload, Mapping):
            raise ProgramWorldCegisError("counterexample must be a mapping")
        evidence = payload.get("evidence_cid")
        if not evidence:
            evidence = cid_for_structured(
                _plain(
                    {
                        "id": str(payload.get("counterexample_id") or ""),
                        "kind": str(payload.get("kind") or ""),
                        "path": str(payload.get("path") or ""),
                        "before": str(payload.get("before") or ""),
                        "expected_after": str(payload.get("expected_after") or ""),
                        "witness": str(payload.get("witness") or ""),
                    }
                )
            )
        return cls(
            counterexample_id=str(payload.get("counterexample_id") or ""),
            kind=str(payload.get("kind") or CounterexampleKind.SPAN.value),
            path=str(payload.get("path") or ""),
            evidence_cid=str(evidence),
            before=str(payload.get("before") or ""),
            expected_after=str(payload.get("expected_after") or ""),
            symbol_from=str(payload.get("symbol_from") or ""),
            symbol_to=str(payload.get("symbol_to") or ""),
            import_module=str(payload.get("import_module") or ""),
            import_name=str(payload.get("import_name") or ""),
            argument_name=str(payload.get("argument_name") or ""),
            argument_value=str(payload.get("argument_value") or ""),
            registration_anchor=str(payload.get("registration_anchor") or ""),
            registration_payload=str(payload.get("registration_payload") or ""),
            obligation_id=str(payload.get("obligation_id") or ""),
            witness=str(payload.get("witness") or ""),
            parent_id=str(payload.get("parent_id") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )


def _coerce_counterexample(
    value: ProgramWorldCounterexample | Mapping[str, Any], *, default_path: str
) -> ProgramWorldCounterexample:
    if isinstance(value, ProgramWorldCounterexample):
        return value
    if not isinstance(value, Mapping):
        raise ProgramWorldCegisError("counterexample must be a mapping")
    payload = dict(value)
    payload.setdefault("path", default_path)
    payload.setdefault("kind", CounterexampleKind.SPAN.value)
    if not payload.get("counterexample_id"):
        payload["counterexample_id"] = "cx:" + _identity_cid(_plain(payload))[:24]
    return ProgramWorldCounterexample.from_dict(payload)


@dataclass(frozen=True)
class FailedRepairAttempt(CanonicalContract):
    """Immutable record of a rejected or abstaining candidate.  Never deleted."""

    SCHEMA: ClassVar[str] = FAILED_ATTEMPT_SCHEMA

    fingerprint: str
    operator_id: str
    path: str
    reason_codes: tuple[str, ...]
    evidence_cids: tuple[str, ...]
    sketch_cid: str | None = None
    origin: str = "analytical"
    residual_question: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fingerprint", _text(self.fingerprint, "fingerprint"))
        object.__setattr__(self, "operator_id", _text(self.operator_id, "operator_id"))
        object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(self, "evidence_cids", _ids(self.evidence_cids, "evidence_cids"))
        object.__setattr__(self, "sketch_cid", _optional_cid(self.sketch_cid, "sketch_cid"))
        object.__setattr__(self, "origin", _text(self.origin, "origin"))
        object.__setattr__(
            self,
            "residual_question",
            _text(self.residual_question, "residual_question", empty=True),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "operator_id": self.operator_id,
            "path": self.path,
            "reason_codes": list(self.reason_codes),
            "evidence_cids": list(self.evidence_cids),
            "sketch_cid": self.sketch_cid,
            "origin": self.origin,
            "residual_question": self.residual_question,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailedRepairAttempt":
        if not isinstance(payload, Mapping):
            raise ProgramWorldCegisError("failed attempt must be a mapping")
        return cls(
            fingerprint=str(payload.get("fingerprint") or ""),
            operator_id=str(payload.get("operator_id") or ""),
            path=str(payload.get("path") or "rejected.py"),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            evidence_cids=tuple(payload.get("evidence_cids") or ()),
            sketch_cid=payload.get("sketch_cid"),
            origin=str(payload.get("origin") or "analytical"),
            residual_question=str(payload.get("residual_question") or ""),
        )


@dataclass(frozen=True)
class CegisRound(CanonicalContract):
    """One bounded CEGIS iteration over a single candidate."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_CEGIS_ROUND_SCHEMA

    round_index: int
    disposition: CegisRoundDisposition | str
    origin: str
    operator_id: str
    fingerprint: str
    reason_codes: tuple[str, ...] = ()
    counterexample_cids: tuple[str, ...] = ()
    sketch_cid: str | None = None
    refined_counterexample_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_index", _nonneg_int(self.round_index, "round_index"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, CegisRoundDisposition, "disposition"),
        )
        object.__setattr__(self, "origin", _text(self.origin, "origin"))
        object.__setattr__(self, "operator_id", _text(self.operator_id, "operator_id"))
        object.__setattr__(self, "fingerprint", _text(self.fingerprint, "fingerprint"))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(
            self,
            "counterexample_cids",
            _ids(self.counterexample_cids, "counterexample_cids"),
        )
        object.__setattr__(self, "sketch_cid", _optional_cid(self.sketch_cid, "sketch_cid"))
        object.__setattr__(
            self,
            "refined_counterexample_cids",
            _ids(self.refined_counterexample_cids, "refined_counterexample_cids"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "disposition": self.disposition
            if isinstance(self.disposition, str)
            else self.disposition.value,
            "origin": self.origin,
            "operator_id": self.operator_id,
            "fingerprint": self.fingerprint,
            "reason_codes": list(self.reason_codes),
            "counterexample_cids": list(self.counterexample_cids),
            "sketch_cid": self.sketch_cid,
            "refined_counterexample_cids": list(self.refined_counterexample_cids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CegisRound":
        if not isinstance(payload, Mapping):
            raise ProgramWorldCegisError("cegis round must be a mapping")
        return cls(
            round_index=int(payload.get("round_index") or 0),
            disposition=str(payload.get("disposition") or ""),
            origin=str(payload.get("origin") or ""),
            operator_id=str(payload.get("operator_id") or ""),
            fingerprint=str(payload.get("fingerprint") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            counterexample_cids=tuple(payload.get("counterexample_cids") or ()),
            sketch_cid=payload.get("sketch_cid"),
            refined_counterexample_cids=tuple(payload.get("refined_counterexample_cids") or ()),
        )


@dataclass(frozen=True)
class ProgramWorldRepairSynthesis(CanonicalContract):
    """Proposal-only synthesis receipt.  Failures preserve every attempt."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REPAIR_SYNTHESIS_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_CEGIS_INTERFACE

    disposition: CegisDisposition | str
    route: CegisRoute | str
    path: str
    environment_binding_cid: str
    bounds: CegisBounds
    analytical_ran_first: bool = True
    model_calls: int = 0
    iterations: int = 0
    candidates_considered: int = 0
    wall_time_ms: int = 0
    sketch: TypedPatchSketch | None = None
    rounds: tuple[CegisRound, ...] = ()
    failed_attempts: tuple[FailedRepairAttempt, ...] = ()
    counterexample_cids: tuple[str, ...] = ()
    residual_question: str = ""
    reason_codes: tuple[str, ...] = ()
    strategy_change_required: bool = False
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    grants_semantic_authority: bool = False
    independently_admitted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", _enum(self.disposition, CegisDisposition, "disposition")
        )
        object.__setattr__(self, "route", _enum(self.route, CegisRoute, "route"))
        object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(
            self,
            "environment_binding_cid",
            _cid(self.environment_binding_cid, "environment_binding_cid"),
        )
        if not isinstance(self.bounds, CegisBounds):
            raise ProgramWorldCegisError("bounds must be CegisBounds")
        object.__setattr__(self, "analytical_ran_first", True)
        object.__setattr__(self, "model_calls", 0)
        object.__setattr__(self, "iterations", _nonneg_int(self.iterations, "iterations"))
        object.__setattr__(
            self,
            "candidates_considered",
            _nonneg_int(self.candidates_considered, "candidates_considered"),
        )
        object.__setattr__(self, "wall_time_ms", _nonneg_int(self.wall_time_ms, "wall_time_ms"))
        if self.sketch is not None and not isinstance(self.sketch, TypedPatchSketch):
            raise ProgramWorldCegisError("sketch must be a TypedPatchSketch")
        rounds = tuple(self.rounds or ())
        if len(rounds) > MAX_COLLECTION_ITEMS:
            raise ProgramWorldCegisBoundsError("rounds exceed collection bound")
        object.__setattr__(self, "rounds", rounds)
        attempts = tuple(self.failed_attempts or ())
        if len(attempts) > MAX_COLLECTION_ITEMS:
            raise ProgramWorldCegisBoundsError("failed attempts exceed collection bound")
        object.__setattr__(self, "failed_attempts", attempts)
        object.__setattr__(
            self, "counterexample_cids", _ids(self.counterexample_cids, "counterexample_cids")
        )
        object.__setattr__(
            self,
            "residual_question",
            _text(self.residual_question, "residual_question", empty=True),
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(
            self,
            "strategy_change_required",
            bool(self.strategy_change_required),
        )
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_semantic_authority", False)
        object.__setattr__(self, "independently_admitted", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": PROGRAM_WORLD_CEGIS_INTERFACE,
            "version": CEGIS_VERSION,
            "disposition": self.disposition
            if isinstance(self.disposition, str)
            else self.disposition.value,
            "route": self.route if isinstance(self.route, str) else self.route.value,
            "path": self.path,
            "environment_binding_cid": self.environment_binding_cid,
            "bounds": self.bounds.to_dict(),
            "analytical_ran_first": True,
            "model_calls": 0,
            "iterations": self.iterations,
            "candidates_considered": self.candidates_considered,
            "wall_time_ms": self.wall_time_ms,
            "sketch": None if self.sketch is None else self.sketch.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
            "failed_attempts": [item.to_dict() for item in self.failed_attempts],
            "counterexample_cids": list(self.counterexample_cids),
            "residual_question": self.residual_question,
            "reason_codes": list(self.reason_codes),
            "strategy_change_required": self.strategy_change_required,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "grants_semantic_authority": False,
            "independently_admitted": False,
            "evidence": SAWM_CEGIS_REPAIR_EVIDENCE,
            "producer_id": PRODUCER_ID,
        }

    @property
    def synthesis_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramWorldRepairSynthesis":
        if not isinstance(payload, Mapping):
            raise ProgramWorldCegisError("synthesis receipt must be a mapping")
        sketch_payload = payload.get("sketch")
        return cls(
            disposition=str(payload.get("disposition") or ""),
            route=str(payload.get("route") or ""),
            path=str(payload.get("path") or "rejected.py"),
            environment_binding_cid=str(payload.get("environment_binding_cid") or ""),
            bounds=CegisBounds.from_dict(payload.get("bounds")),
            iterations=int(payload.get("iterations") or 0),
            candidates_considered=int(payload.get("candidates_considered") or 0),
            wall_time_ms=int(payload.get("wall_time_ms") or 0),
            sketch=TypedPatchSketch.from_dict(sketch_payload)
            if isinstance(sketch_payload, Mapping)
            else None,
            rounds=tuple(
                CegisRound.from_dict(item) if isinstance(item, Mapping) else item
                for item in (payload.get("rounds") or ())
            ),
            failed_attempts=tuple(
                FailedRepairAttempt.from_dict(item) if isinstance(item, Mapping) else item
                for item in (payload.get("failed_attempts") or ())
            ),
            counterexample_cids=tuple(payload.get("counterexample_cids") or ()),
            residual_question=str(payload.get("residual_question") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            strategy_change_required=bool(payload.get("strategy_change_required", False)),
        )


class RepairCounterexampleRefiner:
    """Monotonic counterexample refinement.  Original evidence is retained."""

    def refine(
        self,
        counterexample: ProgramWorldCounterexample,
        *,
        after_source: str,
        failed_sketch: TypedPatchSketch | None = None,
    ) -> ProgramWorldCounterexample | None:
        kind = (
            counterexample.kind
            if isinstance(counterexample.kind, str)
            else counterexample.kind.value
        )
        remains = False
        if counterexample.before and counterexample.before in after_source:
            if not counterexample.expected_after or counterexample.expected_after not in after_source:
                remains = True
        if kind == CounterexampleKind.SYMBOL.value and counterexample.symbol_from:
            if identifier_occurs(after_source, counterexample.symbol_from) and (
                not counterexample.symbol_to
                or not identifier_occurs(after_source, counterexample.symbol_to)
            ):
                remains = True
        if kind == CounterexampleKind.IMPORT.value and counterexample.import_module:
            token = (
                f"from {counterexample.import_module} import"
                if counterexample.import_name
                else f"import {counterexample.import_module}"
            )
            if token not in after_source:
                remains = True
        if not remains:
            if failed_sketch is not None and counterexample.expected_after:
                if counterexample.expected_after not in after_source:
                    remains = True
        if not remains:
            return None
        witness = counterexample.witness or counterexample.before
        if failed_sketch is not None:
            witness = f"{witness}|sketch:{failed_sketch.fingerprint}"
        payload = dict(counterexample._payload())
        payload["parent_id"] = counterexample.counterexample_id
        payload["counterexample_id"] = f"{counterexample.counterexample_id}:refined"
        payload["witness"] = witness
        payload["evidence_cid"] = cid_for_structured(
            _plain(
                {
                    "parent": counterexample.counterexample_cid,
                    "witness": witness,
                    "after_source_cid": _source_cid(after_source),
                }
            )
        )
        return ProgramWorldCounterexample.from_dict(payload)


@dataclass(frozen=True)
class _CandidateSpec:
    operator: str
    origin: str
    before: str = ""
    after: str = ""
    symbol_from: str = ""
    symbol_to: str = ""
    import_module: str = ""
    import_name: str = ""
    argument_name: str = ""
    argument_value: str = ""
    registration_anchor: str = ""
    registration_payload: str = ""
    evidence_cid: str = ""

    def fingerprint(self, path: str) -> str:
        return _identity_cid(
            {
                "operator": self.operator,
                "path": path,
                "before": self.before,
                "after": self.after,
                "symbol_from": self.symbol_from,
                "symbol_to": self.symbol_to,
                "import_module": self.import_module,
                "import_name": self.import_name,
                "argument_name": self.argument_name,
                "argument_value": self.argument_value,
                "registration_anchor": self.registration_anchor,
                "registration_payload": self.registration_payload,
                "origin": self.origin,
            }
        )


def _candidate_from_counterexample(
    item: ProgramWorldCounterexample, *, operators: Sequence[str]
) -> list[_CandidateSpec]:
    kind = item.kind if isinstance(item.kind, str) else item.kind.value
    specs: list[_CandidateSpec] = []
    admitted = set(operators)
    if (
        item.before
        and item.expected_after
        and ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN.value in admitted
    ):
        specs.append(
            _CandidateSpec(
                operator=ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN.value,
                origin="analytical_counterexample",
                before=item.before,
                after=item.expected_after,
                evidence_cid=item.evidence_cid,
            )
        )
    if (
        item.before
        and ProgramWorldRepairOperatorKind.EQUALITY_REWRITE.value in admitted
        and kind in {CounterexampleKind.SPAN.value, CounterexampleKind.TYPE.value}
    ):
        specs.append(
            _CandidateSpec(
                operator=ProgramWorldRepairOperatorKind.EQUALITY_REWRITE.value,
                origin="analytical_equality",
                before=item.before,
                evidence_cid=item.evidence_cid,
            )
        )
    if (
        item.symbol_from
        and item.symbol_to
        and ProgramWorldRepairOperatorKind.RENAME_EXACT_SYMBOL.value in admitted
    ):
        specs.append(
            _CandidateSpec(
                operator=ProgramWorldRepairOperatorKind.RENAME_EXACT_SYMBOL.value,
                origin="analytical_symbol",
                symbol_from=item.symbol_from,
                symbol_to=item.symbol_to,
                evidence_cid=item.evidence_cid,
            )
        )
    if item.import_module and ProgramWorldRepairOperatorKind.ADD_UNIQUE_IMPORT.value in admitted:
        specs.append(
            _CandidateSpec(
                operator=ProgramWorldRepairOperatorKind.ADD_UNIQUE_IMPORT.value,
                origin="analytical_import",
                import_module=item.import_module,
                import_name=item.import_name,
                evidence_cid=item.evidence_cid,
            )
        )
    if (
        item.argument_name
        and item.before
        and ProgramWorldRepairOperatorKind.ADD_UNIQUE_ARGUMENT.value in admitted
    ):
        specs.append(
            _CandidateSpec(
                operator=ProgramWorldRepairOperatorKind.ADD_UNIQUE_ARGUMENT.value,
                origin="analytical_argument",
                before=item.before,
                argument_name=item.argument_name,
                argument_value=item.argument_value or item.argument_name,
                evidence_cid=item.evidence_cid,
            )
        )
    if (
        item.registration_anchor
        and item.registration_payload
        and ProgramWorldRepairOperatorKind.ADD_UNIQUE_REGISTRATION.value in admitted
    ):
        specs.append(
            _CandidateSpec(
                operator=ProgramWorldRepairOperatorKind.ADD_UNIQUE_REGISTRATION.value,
                origin="analytical_registration",
                registration_anchor=item.registration_anchor,
                registration_payload=item.registration_payload,
                evidence_cid=item.evidence_cid,
            )
        )
    if (
        item.expected_after
        and not item.before
        and ProgramWorldRepairOperatorKind.RESTORE_TRACKED_ARTIFACT.value in admitted
    ):
        specs.append(
            _CandidateSpec(
                operator=ProgramWorldRepairOperatorKind.RESTORE_TRACKED_ARTIFACT.value,
                origin="analytical_restore",
                after=item.expected_after,
                evidence_cid=item.evidence_cid,
            )
        )
    return specs


def _closes_counterexamples(
    after_source: str, counterexamples: Sequence[ProgramWorldCounterexample]
) -> tuple[bool, tuple[ProgramWorldCounterexample, ...]]:
    remaining: list[ProgramWorldCounterexample] = []
    for item in counterexamples:
        kind = item.kind if isinstance(item.kind, str) else item.kind.value
        closed = True
        if item.expected_after:
            closed = item.expected_after in after_source
            if item.before and item.before in after_source and item.before != item.expected_after:
                closed = False
        elif kind == CounterexampleKind.SYMBOL.value and item.symbol_from and item.symbol_to:
            closed = not identifier_occurs(after_source, item.symbol_from) and identifier_occurs(
                after_source, item.symbol_to
            )
        elif kind == CounterexampleKind.IMPORT.value and item.import_module:
            token = (
                f"from {item.import_module} import {item.import_name}"
                if item.import_name
                else f"import {item.import_module}"
            )
            closed = token in after_source
        elif item.before:
            closed = item.before not in after_source
        if not closed:
            remaining.append(item)
    return not remaining, tuple(remaining)


def _apply_spec(
    spec: _CandidateSpec,
    *,
    source: str,
    path: str,
    write_scope: Sequence[str],
    protected_paths: Sequence[str],
    environment_binding_cid: str,
    expected_environment_binding_cid: str | None,
    obligations: Sequence[str],
    tests: Sequence[str],
    effects: Sequence[str],
    language: str,
    registry: ProgramWorldRepairOperatorRegistry,
    analytical: bool,
    metadata: Mapping[str, Any] | None = None,
) -> OperatorApplicationReceipt:
    return apply_program_world_repair_operator(
        operator=spec.operator,
        source=source,
        path=path,
        write_scope=write_scope,
        environment_binding_cid=environment_binding_cid,
        expected_environment_binding_cid=expected_environment_binding_cid,
        protected_paths=protected_paths,
        before=spec.before,
        after=spec.after,
        symbol_from=spec.symbol_from,
        symbol_to=spec.symbol_to,
        import_module=spec.import_module,
        import_name=spec.import_name,
        argument_name=spec.argument_name,
        argument_value=spec.argument_value,
        registration_anchor=spec.registration_anchor,
        registration_payload=spec.registration_payload,
        obligations=obligations,
        tests=tests,
        effects=effects,
        language=language,
        metadata=metadata,
        registry=registry,
        analytical=analytical,
    )


def _record_failure(
    spec: _CandidateSpec,
    receipt: OperatorApplicationReceipt,
    *,
    path: str,
    evidence_cids: Sequence[str],
    origin: str,
) -> FailedRepairAttempt:
    fingerprint = spec.fingerprint(path)
    if receipt.sketch is not None:
        fingerprint = receipt.sketch.fingerprint
    return FailedRepairAttempt(
        fingerprint=fingerprint,
        operator_id=receipt.operator_id,
        path=path,
        reason_codes=receipt.reason_codes,
        evidence_cids=tuple(evidence_cids),
        sketch_cid=None if receipt.sketch is None else receipt.sketch.sketch_cid,
        origin=origin,
        residual_question=receipt.residual_question,
    )


def synthesize_program_world_repair(
    *,
    source: str,
    path: str,
    write_scope: Sequence[str],
    environment_binding_cid: str,
    expected_environment_binding_cid: str | None = None,
    counterexamples: Sequence[ProgramWorldCounterexample | Mapping[str, Any]] = (),
    obligations: Sequence[str] = (),
    tests: Sequence[str] = (),
    effects: Sequence[str] = ("pure_local",),
    protected_paths: Sequence[str] = (),
    operators: Sequence[str] | None = None,
    bounds: CegisBounds | Mapping[str, Any] | None = None,
    prior_failures: Sequence[FailedRepairAttempt | Mapping[str, Any]] = (),
    model_proposal: Mapping[str, Any] | None = None,
    language: str = "python",
    registry: ProgramWorldRepairOperatorRegistry | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProgramWorldRepairSynthesis:
    """Synthesize a bounded typed repair sketch.  Analytical routes run first."""

    started = time.monotonic()
    _assert_current_binding(environment_binding_cid, expected_environment_binding_cid)
    path_text = _normalize_path(path)
    source_text = _text(source, "source", empty=True, limit=MAX_SOURCE_BYTES)
    budget = bounds if isinstance(bounds, CegisBounds) else CegisBounds.from_dict(bounds)
    catalogue = registry or build_default_program_world_operator_registry()
    if operators is None:
        admitted_ops = tuple(ANALYTICAL_OPERATOR_ORDER)
    else:
        admitted_ops = _ids(operators, "operators")
        unknown = [item for item in admitted_ops if item not in CLOSED_OPERATOR_VOCABULARY]
        if unknown:
            raise ProgramWorldCegisAuthorityError(
                "operators must stay inside the closed SAWM vocabulary"
            )
    ordered_ops = tuple(item for item in ANALYTICAL_OPERATOR_ORDER if item in set(admitted_ops))
    cx_list = tuple(
        _coerce_counterexample(item, default_path=path_text) for item in (counterexamples or ())
    )
    if len(cx_list) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldCegisBoundsError("counterexamples exceed collection bound")
    prior = tuple(
        item
        if isinstance(item, FailedRepairAttempt)
        else FailedRepairAttempt.from_dict(item)
        for item in (prior_failures or ())
    )
    failed: list[FailedRepairAttempt] = list(prior)
    rounds: list[CegisRound] = []
    refined: list[ProgramWorldCounterexample] = list(cx_list)
    evidence_cids = tuple(item.evidence_cid for item in refined)
    seen_fingerprints = {item.fingerprint for item in failed}
    identical_counts: dict[str, int] = {}
    for item in failed:
        identical_counts[item.fingerprint] = identical_counts.get(item.fingerprint, 0) + 1
    prior_evidence_cids = set()
    for item in prior:
        prior_evidence_cids.update(item.evidence_cids)

    specs: list[_CandidateSpec] = []
    seen_specs: set[str] = set()
    for item in refined:
        for spec in _candidate_from_counterexample(item, operators=ordered_ops):
            key = spec.fingerprint(path_text)
            if key not in seen_specs:
                seen_specs.add(key)
                specs.append(spec)
    specs.sort(key=lambda item: (ANALYTICAL_OPERATOR_ORDER.index(item.operator), item.origin, item.before, item.after))

    def elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    def finish(
        *,
        disposition: CegisDisposition,
        route: CegisRoute,
        sketch: TypedPatchSketch | None = None,
        residual_question: str = "",
        reason_codes: Sequence[str] = (),
        strategy_change_required: bool = False,
        iterations: int = 0,
        candidates_considered: int = 0,
    ) -> ProgramWorldRepairSynthesis:
        return ProgramWorldRepairSynthesis(
            disposition=disposition,
            route=route,
            path=path_text,
            environment_binding_cid=environment_binding_cid,
            bounds=budget,
            iterations=iterations,
            candidates_considered=candidates_considered,
            wall_time_ms=elapsed_ms(),
            sketch=sketch,
            rounds=tuple(rounds),
            failed_attempts=tuple(failed),
            counterexample_cids=tuple(item.counterexample_cid for item in refined),
            residual_question=residual_question,
            reason_codes=reason_codes,
            strategy_change_required=strategy_change_required,
        )

    RepairCandidateScopeGate(
        write_scope=write_scope,
        protected_paths=protected_paths,
        required_effects=effects,
        required_obligations=obligations,
        required_tests=tests,
    )
    refiner = RepairCounterexampleRefiner()
    candidates_considered = 0
    iterations = 0

    # 1. Analytical unique repairs run before bounded CEGIS and residual
    # model sketches.  The first unique closed-operator hit that discharges
    # the counterexamples is returned; later model proposals cannot override it.
    for spec in specs:
        if spec.origin.startswith("analytical") is False:
            continue
        if elapsed_ms() >= budget.max_wall_time_ms:
            break
        candidates_considered += 1
        try:
            receipt = _apply_spec(
                spec,
                source=source_text,
                path=path_text,
                write_scope=write_scope,
                protected_paths=protected_paths,
                environment_binding_cid=environment_binding_cid,
                expected_environment_binding_cid=expected_environment_binding_cid,
                obligations=obligations,
                tests=tests,
                effects=effects,
                language=language,
                registry=catalogue,
                analytical=True,
                metadata=metadata,
            )
        except ProgramWorldRepairStaleError:
            raise
        except ProgramWorldRepairError as exc:
            attempt = FailedRepairAttempt(
                fingerprint=spec.fingerprint(path_text),
                operator_id=spec.operator,
                path=path_text,
                reason_codes=(OperatorReason.RESIDUAL_QUESTION.value,),
                evidence_cids=evidence_cids,
                origin=spec.origin,
                residual_question=str(exc),
            )
            failed.append(attempt)
            seen_fingerprints.add(attempt.fingerprint)
            continue
        if receipt.unique and receipt.sketch is not None:
            closed, remaining = _closes_counterexamples(receipt.sketch.after_source, refined)
            if closed or not refined:
                return finish(
                    disposition=CegisDisposition.UNIQUE_REPAIR,
                    route=CegisRoute.ANALYTICAL_UNIQUE,
                    sketch=receipt.sketch,
                    reason_codes=(
                        OperatorReason.UNIQUE_MATCH.value,
                        OperatorReason.PROPOSAL_ONLY.value,
                    ),
                    candidates_considered=candidates_considered,
                )
            attempt = _record_failure(
                spec, receipt, path=path_text, evidence_cids=evidence_cids, origin=spec.origin
            )
            failed.append(attempt)
            seen_fingerprints.add(attempt.fingerprint)
            for item in remaining:
                child = refiner.refine(
                    item,
                    after_source=receipt.sketch.after_source,
                    failed_sketch=receipt.sketch,
                )
                if child is not None:
                    refined.append(child)
        else:
            attempt = _record_failure(
                spec, receipt, path=path_text, evidence_cids=evidence_cids, origin=spec.origin
            )
            failed.append(attempt)
            seen_fingerprints.add(attempt.fingerprint)

    # 2. Bounded CEGIS over remaining closed operators / refined counterexamples.
    cegis_specs: list[_CandidateSpec] = []
    seen_cegis: set[str] = set()
    for item in refined:
        for spec in _candidate_from_counterexample(item, operators=ordered_ops):
            key = spec.fingerprint(path_text)
            if key not in seen_cegis:
                seen_cegis.add(key)
                cegis_specs.append(spec)
    cegis_specs.sort(
        key=lambda item: (ANALYTICAL_OPERATOR_ORDER.index(item.operator), item.origin, item.before)
    )

    for spec in cegis_specs:
        if iterations >= budget.max_iterations or candidates_considered >= budget.max_candidates:
            return finish(
                disposition=CegisDisposition.BUDGET_EXHAUSTED,
                route=CegisRoute.BOUNDED_CEGIS,
                residual_question="bounded CEGIS exhausted iterations or candidates",
                reason_codes=("budget_exhausted", OperatorReason.RESIDUAL_QUESTION.value),
                strategy_change_required=True,
                iterations=iterations,
                candidates_considered=candidates_considered,
            )
        if elapsed_ms() >= budget.max_wall_time_ms:
            return finish(
                disposition=CegisDisposition.BUDGET_EXHAUSTED,
                route=CegisRoute.BOUNDED_CEGIS,
                residual_question="bounded CEGIS exhausted wall time",
                reason_codes=("wall_time_exhausted", OperatorReason.RESIDUAL_QUESTION.value),
                strategy_change_required=True,
                iterations=iterations,
                candidates_considered=candidates_considered,
            )
        fingerprint = spec.fingerprint(path_text)
        if fingerprint in seen_fingerprints:
            identical_counts[fingerprint] = identical_counts.get(fingerprint, 0) + 1
            rounds.append(
                CegisRound(
                    round_index=iterations,
                    disposition=CegisRoundDisposition.SKIPPED_DUPLICATE,
                    origin=spec.origin,
                    operator_id=spec.operator,
                    fingerprint=fingerprint,
                    reason_codes=("identical_failure_without_new_evidence",),
                    counterexample_cids=tuple(item.counterexample_cid for item in refined),
                )
            )
            # Already-tried candidates never re-execute.  Identical *model*
            # retries are gated after deterministic routes so the residual
            # sketch can still be blocked with a typed evidence-preserving
            # reason rather than aborting the cascade early.
            continue
        iterations += 1
        candidates_considered += 1
        try:
            receipt = _apply_spec(
                spec,
                source=source_text,
                path=path_text,
                write_scope=write_scope,
                protected_paths=protected_paths,
                environment_binding_cid=environment_binding_cid,
                expected_environment_binding_cid=expected_environment_binding_cid,
                obligations=obligations,
                tests=tests,
                effects=effects,
                language=language,
                registry=catalogue,
                analytical=False,
                metadata=metadata,
            )
        except ProgramWorldRepairStaleError:
            raise
        except ProgramWorldRepairError as exc:
            attempt = FailedRepairAttempt(
                fingerprint=spec.fingerprint(path_text),
                operator_id=spec.operator,
                path=path_text,
                reason_codes=(OperatorReason.RESIDUAL_QUESTION.value,),
                evidence_cids=evidence_cids,
                origin="bounded_cegis",
                residual_question=str(exc),
            )
            failed.append(attempt)
            seen_fingerprints.add(attempt.fingerprint)
            rounds.append(
                CegisRound(
                    round_index=iterations,
                    disposition=CegisRoundDisposition.ABSTAINED,
                    origin="bounded_cegis",
                    operator_id=spec.operator,
                    fingerprint=attempt.fingerprint,
                    reason_codes=attempt.reason_codes,
                    counterexample_cids=tuple(item.counterexample_cid for item in refined),
                )
            )
            continue
        if not receipt.unique or receipt.sketch is None:
            attempt = _record_failure(
                spec, receipt, path=path_text, evidence_cids=evidence_cids, origin="bounded_cegis"
            )
            failed.append(attempt)
            seen_fingerprints.add(attempt.fingerprint)
            rounds.append(
                CegisRound(
                    round_index=iterations,
                    disposition=CegisRoundDisposition.REJECTED
                    if receipt.disposition == OperatorApplicationDisposition.REJECTED.value
                    else CegisRoundDisposition.ABSTAINED,
                    origin="bounded_cegis",
                    operator_id=spec.operator,
                    fingerprint=attempt.fingerprint,
                    reason_codes=receipt.reason_codes,
                    counterexample_cids=tuple(item.counterexample_cid for item in refined),
                    sketch_cid=attempt.sketch_cid,
                )
            )
            continue
        closed, remaining = _closes_counterexamples(receipt.sketch.after_source, refined)
        if closed:
            rounds.append(
                CegisRound(
                    round_index=iterations,
                    disposition=CegisRoundDisposition.APPLIED,
                    origin="bounded_cegis",
                    operator_id=spec.operator,
                    fingerprint=receipt.sketch.fingerprint,
                    reason_codes=(OperatorReason.PROPOSAL_ONLY.value,),
                    counterexample_cids=tuple(item.counterexample_cid for item in refined),
                    sketch_cid=receipt.sketch.sketch_cid,
                )
            )
            return finish(
                disposition=CegisDisposition.SYNTHESIZED,
                route=CegisRoute.BOUNDED_CEGIS,
                sketch=receipt.sketch,
                reason_codes=(OperatorReason.PROPOSAL_ONLY.value,),
                iterations=iterations,
                candidates_considered=candidates_considered,
            )
        child_cids: list[str] = []
        for item in remaining:
            child = refiner.refine(
                item,
                after_source=receipt.sketch.after_source,
                failed_sketch=receipt.sketch,
            )
            if child is not None:
                refined.append(child)
                child_cids.append(child.counterexample_cid)
        attempt = _record_failure(
            spec, receipt, path=path_text, evidence_cids=evidence_cids, origin="bounded_cegis"
        )
        failed.append(attempt)
        seen_fingerprints.add(attempt.fingerprint)
        rounds.append(
            CegisRound(
                round_index=iterations,
                disposition=CegisRoundDisposition.COUNTEREXAMPLE_REMAINS
                if remaining
                else CegisRoundDisposition.NEW_COUNTEREXAMPLE,
                origin="bounded_cegis",
                operator_id=spec.operator,
                fingerprint=attempt.fingerprint,
                reason_codes=("original_counterexample_remains",),
                counterexample_cids=tuple(item.counterexample_cid for item in remaining),
                sketch_cid=receipt.sketch.sketch_cid,
                refined_counterexample_cids=tuple(child_cids),
            )
        )

    # 3. Residual model sketch only after deterministic routes, and only with new evidence.
    if model_proposal is not None:
        if not isinstance(model_proposal, Mapping):
            raise ProgramWorldCegisError("model_proposal must be a mapping")
        proposal_evidence = str(model_proposal.get("evidence_cid") or "")
        spec = _CandidateSpec(
            operator=str(
                model_proposal.get("operator")
                or ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN.value
            ),
            origin="residual_model",
            before=str(model_proposal.get("before") or ""),
            after=str(model_proposal.get("after") or ""),
            symbol_from=str(model_proposal.get("symbol_from") or ""),
            symbol_to=str(model_proposal.get("symbol_to") or ""),
            import_module=str(model_proposal.get("import_module") or ""),
            import_name=str(model_proposal.get("import_name") or ""),
            argument_name=str(model_proposal.get("argument_name") or ""),
            argument_value=str(model_proposal.get("argument_value") or ""),
            registration_anchor=str(model_proposal.get("registration_anchor") or ""),
            registration_payload=str(model_proposal.get("registration_payload") or ""),
            evidence_cid=proposal_evidence,
        )
        proposal_fingerprint = spec.fingerprint(path_text)
        new_evidence = bool(
            (proposal_evidence and proposal_evidence not in prior_evidence_cids)
            or (set(evidence_cids) - prior_evidence_cids)
        )
        identical = proposal_fingerprint in seen_fingerprints
        if identical and not new_evidence:
            failed.append(
                FailedRepairAttempt(
                    fingerprint=proposal_fingerprint,
                    operator_id=spec.operator,
                    path=path_text,
                    reason_codes=("identical_model_retry_without_new_evidence",),
                    evidence_cids=evidence_cids,
                    origin="model_proposal",
                    residual_question="identical model retry blocked without new evidence",
                )
            )
            return finish(
                disposition=CegisDisposition.IDENTICAL_RETRY_BLOCKED,
                route=CegisRoute.IDENTICAL_RETRY_BLOCKED,
                residual_question="identical model retry blocked without new evidence",
                reason_codes=("identical_model_retry_without_new_evidence", "strategy_change_required"),
                strategy_change_required=True,
                iterations=iterations,
                candidates_considered=candidates_considered,
            )
        candidates_considered += 1
        try:
            receipt = _apply_spec(
                spec,
                source=source_text,
                path=path_text,
                write_scope=write_scope,
                protected_paths=protected_paths,
                environment_binding_cid=environment_binding_cid,
                expected_environment_binding_cid=expected_environment_binding_cid,
                obligations=obligations,
                tests=tests,
                effects=effects,
                language=language,
                registry=catalogue,
                analytical=False,
                metadata=metadata,
            )
        except ProgramWorldRepairStaleError:
            raise
        except ProgramWorldRepairError as exc:
            failed.append(
                FailedRepairAttempt(
                    fingerprint=proposal_fingerprint,
                    operator_id=spec.operator,
                    path=path_text,
                    reason_codes=(
                        OperatorReason.RESIDUAL_QUESTION.value,
                        "residual_model_sketch",
                    ),
                    evidence_cids=evidence_cids,
                    origin="residual_model",
                    residual_question=str(exc),
                )
            )
            receipt = None
        if receipt is not None and receipt.unique and receipt.sketch is not None:
            closed, remaining = _closes_counterexamples(receipt.sketch.after_source, refined)
            if closed or not remaining:
                return finish(
                    disposition=CegisDisposition.SYNTHESIZED,
                    route=CegisRoute.BOUNDED_CEGIS,
                    sketch=receipt.sketch,
                    reason_codes=(OperatorReason.PROPOSAL_ONLY.value, "residual_model_sketch"),
                    iterations=iterations,
                    candidates_considered=candidates_considered,
                )
        if receipt is not None:
            failed.append(
                _record_failure(
                    spec, receipt, path=path_text, evidence_cids=evidence_cids, origin="residual_model"
                )
            )

    residual = "unsupported repair abstains pending a named residual question"
    if not cx_list:
        residual = "no counterexample uniquely determines a closed operator"
    return finish(
        disposition=CegisDisposition.RESIDUAL,
        route=CegisRoute.RESIDUAL_QUESTION,
        residual_question=residual,
        reason_codes=(OperatorReason.RESIDUAL_QUESTION.value,),
        iterations=iterations,
        candidates_considered=candidates_considered,
    )


class ProgramWorldCEGIS:
    """Bounded CEGIS controller over the closed SAWM operator registry."""

    INTERFACE: ClassVar[str] = PROGRAM_WORLD_CEGIS_INTERFACE

    def __init__(
        self,
        *,
        registry: ProgramWorldRepairOperatorRegistry | None = None,
        bounds: CegisBounds | Mapping[str, Any] | None = None,
        expected_environment_binding_cid: str | None = None,
    ) -> None:
        self.registry = registry or build_default_program_world_operator_registry()
        self.bounds = bounds if isinstance(bounds, CegisBounds) else CegisBounds.from_dict(bounds)
        self.expected_environment_binding_cid = expected_environment_binding_cid
        self.refiner = RepairCounterexampleRefiner()

    def synthesize(self, **kwargs: Any) -> ProgramWorldRepairSynthesis:
        kwargs.setdefault("registry", self.registry)
        kwargs.setdefault("bounds", self.bounds)
        if (
            self.expected_environment_binding_cid is not None
            and "expected_environment_binding_cid" not in kwargs
        ):
            kwargs["expected_environment_binding_cid"] = self.expected_environment_binding_cid
        return synthesize_program_world_repair(**kwargs)


__all__ = [
    "ANALYTICAL_OPERATOR_ORDER",
    "PROGRAM_WORLD_CEGIS_INTERFACE",
    "SAWM_CEGIS_REPAIR_EVIDENCE",
    "CegisBounds",
    "CegisDisposition",
    "CegisRound",
    "CegisRoundDisposition",
    "CegisRoute",
    "CounterexampleKind",
    "FailedRepairAttempt",
    "ProgramWorldCEGIS",
    "ProgramWorldCegisAuthorityError",
    "ProgramWorldCegisBoundsError",
    "ProgramWorldCegisError",
    "ProgramWorldCounterexample",
    "ProgramWorldRepairSynthesis",
    "RepairCounterexampleRefiner",
    "synthesize_program_world_repair",
]
