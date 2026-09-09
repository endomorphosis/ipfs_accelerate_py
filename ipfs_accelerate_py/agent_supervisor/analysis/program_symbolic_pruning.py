"""Authoritative symbolic pruning of static program successors.

Interface: ``SymbolicSuccessorPruner@1``.  Evidence: ``sawm/symbolic-pruning@1``.

Pruning may remove a static possibility or an explicit unknown only when an
authoritative type, effect, path-condition, contract, capability, abstract-
interpretation, or solver reason applies.  Unknown widens.  Solver timeout
and unknown are incomplete, never impossible.  Neural nominations cannot
prune.  Every stage is recorded as selected, skipped, unavailable, rejected,
or escalated.

This module does not introduce a second scanner, SMT solver, or router.  It
evaluates solver-neutral obligations against already-admitted graph facts and
caller-supplied authoritative evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.software_contracts.semantic_state.program_graph import (
    ContractDischargeStatus,
    DynamicFrontierReason,
    ProgramGraphEdgeKind,
    ResolutionStatus,
)

from .program_successors import (
    COMPLETENESS_VALUES,
    CompletenessVerdict,
    FreshnessState,
    SYMBOLIC_PRUNING_EVIDENCE,
    StageStatus,
    StaticSuccessorCandidate,
    StaticSuccessorError,
    StaticSuccessorPlan,
    SuccessorDecisionReceipt,
    SuccessorStageDecision,
    UnresolvedDynamicFrontier,
    _bounded_text,
    _bool,
    _cid,
    _enum_value,
    _fill_remaining_stages,
    _optional_cid,
    _receipt_cid,
    _stage,
    _unique_sorted,
)


SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE: Final[str] = "SymbolicSuccessorPruner@1"
SYMBOLIC_PRUNE_RESULT_INTERFACE: Final[str] = "SymbolicPruneResult@1"
SYMBOLIC_SUCCESSOR_PRUNER_VERSION: Final[str] = "1"
IMPORT_SCAN_PERFORMED: Final[bool] = False
PRODUCER_ID: Final[str] = "ipfs-accelerate.analysis.program-symbolic-pruning"

SYMBOLIC_PRUNE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-prune-result@1"
)
TYPE_CONSTRAINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-type-constraint@1"
)
EFFECT_CONSTRAINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-effect-constraint@1"
)
PATH_CONSTRAINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-path-constraint@1"
)
CONTRACT_CONSTRAINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-contract-constraint@1"
)
CAPABILITY_CONSTRAINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-capability-constraint@1"
)
ABSTRACT_CONSTRAINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-abstract-constraint@1"
)
SOLVER_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-solver-obligation@1"
)
NEURAL_NOMINATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-neural-nomination@1"
)
PRUNE_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/symbolic-prune-record@1"
)

PRUNE_STAGE_ORDER: Final[tuple[str, ...]] = (
    "snapshot_freshness",
    "type",
    "effect",
    "path_condition",
    "contract",
    "capability",
    "abstract_interpretation",
    "solver",
    "neural_nomination",
    "completeness",
)

SOLVER_VERDICTS: Final[frozenset[str]] = frozenset(
    {"sat", "unsat", "unknown", "timeout"}
)
PRUNE_REASON_VALUES: Final[frozenset[str]] = frozenset(
    {
        "type_contradiction",
        "forbidden_effect",
        "unsatisfiable_path",
        "violated_contract",
        "capability_denied",
        "abstract_contradiction",
        "solver_unsat",
        "stale_graph",
    }
)
AUTHORITATIVE_PRUNE_REASONS: Final[frozenset[str]] = PRUNE_REASON_VALUES
INCOMPATIBLE_TYPE_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("int", "str"),
        ("str", "int"),
        ("int", "bytes"),
        ("bytes", "int"),
        ("bool", "str"),
        ("str", "bool"),
        ("float", "str"),
        ("str", "float"),
        ("None", "int"),
        ("int", "None"),
        ("NoneType", "int"),
        ("int", "NoneType"),
    }
)
DYNAMIC_TYPE_MARKERS: Final[frozenset[str]] = frozenset(
    {"", "any", "dynamic", "unknown", "object"}
)


class SymbolicPruneError(StaticSuccessorError):
    """Closed symbolic-pruning contract violation."""


class SolverVerdict(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


class PruneReason(str, Enum):
    TYPE_CONTRADICTION = "type_contradiction"
    FORBIDDEN_EFFECT = "forbidden_effect"
    UNSATISFIABLE_PATH = "unsatisfiable_path"
    VIOLATED_CONTRACT = "violated_contract"
    CAPABILITY_DENIED = "capability_denied"
    ABSTRACT_CONTRADICTION = "abstract_contradiction"
    SOLVER_UNSAT = "solver_unsat"
    STALE_GRAPH = "stale_graph"


# ---------------------------------------------------------------------------
# Evidence records (authoritative only when flagged)
# ---------------------------------------------------------------------------


def _norm_annotation(value: str) -> str:
    text = value.strip()
    if text.startswith("typing."):
        text = text[len("typing.") :]
    return text


@dataclass(frozen=True, slots=True)
class TypeConstraint:
    """Authoritative type fact bound to a successor or subject node."""

    successor_node_cid: str
    observed_annotation: str
    required_annotation: str = ""
    contradiction: bool = False
    authoritative: bool = False
    evidence_cid: str | None = None

    SCHEMA: ClassVar[str] = TYPE_CONSTRAINT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "successor_node_cid",
            _cid(self.successor_node_cid, "successor_node_cid"),
        )
        object.__setattr__(
            self,
            "observed_annotation",
            _bounded_text(self.observed_annotation, "observed_annotation", empty=True),
        )
        object.__setattr__(
            self,
            "required_annotation",
            _bounded_text(self.required_annotation, "required_annotation", empty=True),
        )
        object.__setattr__(self, "contradiction", _bool(self.contradiction, "contradiction"))
        object.__setattr__(self, "authoritative", _bool(self.authoritative, "authoritative"))
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )
        if self.authoritative and self.contradiction and self.evidence_cid is None:
            raise SymbolicPruneError(
                "authoritative type contradictions require evidence_cid"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "successor_node_cid": self.successor_node_cid,
            "observed_annotation": self.observed_annotation,
            "required_annotation": self.required_annotation,
            "contradiction": self.contradiction,
            "authoritative": self.authoritative,
            "evidence_cid": self.evidence_cid,
        }

    @property
    def constraint_cid(self) -> str:
        return _receipt_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class EffectConstraint:
    """Authoritative effect fact; forbidden definite effects may prune."""

    successor_node_cid: str
    effect: str
    forbidden: bool = False
    definite: bool = False
    authoritative: bool = False
    evidence_cid: str | None = None

    SCHEMA: ClassVar[str] = EFFECT_CONSTRAINT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "successor_node_cid",
            _cid(self.successor_node_cid, "successor_node_cid"),
        )
        object.__setattr__(self, "effect", _bounded_text(self.effect, "effect"))
        object.__setattr__(self, "forbidden", _bool(self.forbidden, "forbidden"))
        object.__setattr__(self, "definite", _bool(self.definite, "definite"))
        object.__setattr__(self, "authoritative", _bool(self.authoritative, "authoritative"))
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )
        if self.authoritative and self.forbidden and self.definite and self.evidence_cid is None:
            raise SymbolicPruneError(
                "authoritative forbidden effects require evidence_cid"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "successor_node_cid": self.successor_node_cid,
            "effect": self.effect,
            "forbidden": self.forbidden,
            "definite": self.definite,
            "authoritative": self.authoritative,
            "evidence_cid": self.evidence_cid,
        }

    @property
    def constraint_cid(self) -> str:
        return _receipt_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class PathConstraint:
    """Path-condition fact for a successor edge or CFG branch."""

    successor_edge_cid: str
    satisfiable: bool | None = None
    authoritative: bool = False
    evidence_cid: str | None = None
    condition: str = ""

    SCHEMA: ClassVar[str] = PATH_CONSTRAINT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "successor_edge_cid",
            _cid(self.successor_edge_cid, "successor_edge_cid"),
        )
        if self.satisfiable is not None and not isinstance(self.satisfiable, bool):
            raise SymbolicPruneError("satisfiable must be a boolean or null")
        object.__setattr__(self, "authoritative", _bool(self.authoritative, "authoritative"))
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(
            self, "condition", _bounded_text(self.condition, "condition", empty=True)
        )
        if self.authoritative and self.satisfiable is False and self.evidence_cid is None:
            raise SymbolicPruneError(
                "authoritative unsatisfiable paths require evidence_cid"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "successor_edge_cid": self.successor_edge_cid,
            "satisfiable": self.satisfiable,
            "authoritative": self.authoritative,
            "evidence_cid": self.evidence_cid,
            "condition": self.condition,
        }

    @property
    def constraint_cid(self) -> str:
        return _receipt_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class ContractConstraint:
    """Contract discharge fact bound to a successor or subject."""

    successor_node_cid: str
    discharge_status: ContractDischargeStatus | str
    contract_kind: str = ""
    authoritative: bool = False
    evidence_cid: str | None = None
    specification_cid: str | None = None

    SCHEMA: ClassVar[str] = CONTRACT_CONSTRAINT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "successor_node_cid",
            _cid(self.successor_node_cid, "successor_node_cid"),
        )
        object.__setattr__(
            self,
            "discharge_status",
            _enum_value(
                self.discharge_status,
                frozenset(item.value for item in ContractDischargeStatus),
                "discharge_status",
            ),
        )
        object.__setattr__(
            self,
            "contract_kind",
            _bounded_text(self.contract_kind, "contract_kind", empty=True),
        )
        object.__setattr__(self, "authoritative", _bool(self.authoritative, "authoritative"))
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(
            self,
            "specification_cid",
            _optional_cid(self.specification_cid, "specification_cid"),
        )
        if (
            self.authoritative
            and self.discharge_status == ContractDischargeStatus.VIOLATED.value
            and self.evidence_cid is None
        ):
            raise SymbolicPruneError("authoritative violated contracts require evidence_cid")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "successor_node_cid": self.successor_node_cid,
            "discharge_status": self.discharge_status,
            "contract_kind": self.contract_kind,
            "authoritative": self.authoritative,
            "evidence_cid": self.evidence_cid,
            "specification_cid": self.specification_cid,
        }

    @property
    def constraint_cid(self) -> str:
        return _receipt_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class CapabilityConstraint:
    """Operational capability gate over an effect class."""

    effect: str
    allowed: bool = True
    available: bool = True
    authoritative: bool = False
    evidence_cid: str | None = None

    SCHEMA: ClassVar[str] = CAPABILITY_CONSTRAINT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect", _bounded_text(self.effect, "effect"))
        object.__setattr__(self, "allowed", _bool(self.allowed, "allowed"))
        object.__setattr__(self, "available", _bool(self.available, "available"))
        object.__setattr__(self, "authoritative", _bool(self.authoritative, "authoritative"))
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "effect": self.effect,
            "allowed": self.allowed,
            "available": self.available,
            "authoritative": self.authoritative,
            "evidence_cid": self.evidence_cid,
        }

    @property
    def constraint_cid(self) -> str:
        return _receipt_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class AbstractConstraint:
    """Abstract-interpretation fact (interval / nullness / definite value)."""

    successor_node_cid: str
    domain: str
    contradiction: bool = False
    unknown: bool = False
    authoritative: bool = False
    evidence_cid: str | None = None
    value: str = ""

    SCHEMA: ClassVar[str] = ABSTRACT_CONSTRAINT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "successor_node_cid",
            _cid(self.successor_node_cid, "successor_node_cid"),
        )
        object.__setattr__(self, "domain", _bounded_text(self.domain, "domain"))
        object.__setattr__(self, "contradiction", _bool(self.contradiction, "contradiction"))
        object.__setattr__(self, "unknown", _bool(self.unknown, "unknown"))
        object.__setattr__(self, "authoritative", _bool(self.authoritative, "authoritative"))
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(self, "value", _bounded_text(self.value, "value", empty=True))
        if self.contradiction and self.unknown:
            raise SymbolicPruneError("abstract facts cannot be both contradiction and unknown")
        if self.authoritative and self.contradiction and self.evidence_cid is None:
            raise SymbolicPruneError(
                "authoritative abstract contradictions require evidence_cid"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "successor_node_cid": self.successor_node_cid,
            "domain": self.domain,
            "contradiction": self.contradiction,
            "unknown": self.unknown,
            "authoritative": self.authoritative,
            "evidence_cid": self.evidence_cid,
            "value": self.value,
        }

    @property
    def constraint_cid(self) -> str:
        return _receipt_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class SolverObligation:
    """Solver-neutral obligation evaluation; unknown/timeout never prune."""

    target_cid: str
    verdict: SolverVerdict | str
    authoritative: bool = False
    timeout: bool = False
    obligation_cid: str | None = None
    covers_frontier: bool = False

    SCHEMA: ClassVar[str] = SOLVER_OBLIGATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_cid", _cid(self.target_cid, "target_cid"))
        object.__setattr__(
            self, "verdict", _enum_value(self.verdict, SOLVER_VERDICTS, "verdict")
        )
        object.__setattr__(self, "authoritative", _bool(self.authoritative, "authoritative"))
        object.__setattr__(self, "timeout", _bool(self.timeout, "timeout"))
        object.__setattr__(
            self, "obligation_cid", _optional_cid(self.obligation_cid, "obligation_cid")
        )
        object.__setattr__(
            self, "covers_frontier", _bool(self.covers_frontier, "covers_frontier")
        )
        if self.timeout and self.verdict not in {
            SolverVerdict.TIMEOUT.value,
            SolverVerdict.UNKNOWN.value,
        }:
            raise SymbolicPruneError("timeout obligations must use timeout or unknown verdicts")
        if (
            self.authoritative
            and self.verdict == SolverVerdict.UNSAT.value
            and self.obligation_cid is None
        ):
            raise SymbolicPruneError("authoritative unsat obligations require obligation_cid")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "target_cid": self.target_cid,
            "verdict": self.verdict,
            "authoritative": self.authoritative,
            "timeout": self.timeout,
            "obligation_cid": self.obligation_cid,
            "covers_frontier": self.covers_frontier,
        }

    @property
    def constraint_cid(self) -> str:
        return _receipt_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class NeuralNomination:
    """Proposal-only nomination.  Never authoritative and never prunes."""

    candidate_cid: str
    nominator: str = "neural"
    millipercent: int = 0

    SCHEMA: ClassVar[str] = NEURAL_NOMINATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid"))
        object.__setattr__(self, "nominator", _bounded_text(self.nominator, "nominator"))
        if isinstance(self.millipercent, bool) or not isinstance(self.millipercent, int):
            raise SymbolicPruneError("millipercent must be an integer")
        if self.millipercent < 0 or self.millipercent > 100_000:
            raise SymbolicPruneError("millipercent must be in [0, 100000]")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "candidate_cid": self.candidate_cid,
            "nominator": self.nominator,
            "millipercent": self.millipercent,
            "authoritative": False,
        }

    @property
    def constraint_cid(self) -> str:
        return _receipt_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class SymbolicPruningEvidence:
    """Closed bundle of pruning evidence.  Non-authoritative items cannot prune."""

    type_facts: Sequence[TypeConstraint] = ()
    effect_facts: Sequence[EffectConstraint] = ()
    path_facts: Sequence[PathConstraint] = ()
    contract_facts: Sequence[ContractConstraint] = ()
    capability_facts: Sequence[CapabilityConstraint] = ()
    abstract_facts: Sequence[AbstractConstraint] = ()
    solver_facts: Sequence[SolverObligation] = ()
    neural_nominations: Sequence[NeuralNomination] = ()
    widen_reasons: Sequence[str] = ()
    widen_dimensions: Sequence[str] = ()
    expected_snapshot_cid: str | None = None
    expected_environment_binding_cid: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "type_facts", tuple(self.type_facts))
        object.__setattr__(self, "effect_facts", tuple(self.effect_facts))
        object.__setattr__(self, "path_facts", tuple(self.path_facts))
        object.__setattr__(self, "contract_facts", tuple(self.contract_facts))
        object.__setattr__(self, "capability_facts", tuple(self.capability_facts))
        object.__setattr__(self, "abstract_facts", tuple(self.abstract_facts))
        object.__setattr__(self, "solver_facts", tuple(self.solver_facts))
        object.__setattr__(self, "neural_nominations", tuple(self.neural_nominations))
        object.__setattr__(
            self,
            "widen_reasons",
            _unique_sorted(self.widen_reasons, "widen_reason", normalize=_bounded_text),
        )
        illegal = [item for item in self.widen_reasons if item not in {r.value for r in DynamicFrontierReason}]
        if illegal:
            raise SymbolicPruneError(f"widen_reasons must be DynamicFrontierReason values, not {illegal}")
        object.__setattr__(
            self,
            "widen_dimensions",
            _unique_sorted(
                self.widen_dimensions, "widen_dimension", normalize=_bounded_text
            ),
        )
        object.__setattr__(
            self,
            "expected_snapshot_cid",
            _optional_cid(self.expected_snapshot_cid, "expected_snapshot_cid"),
        )
        object.__setattr__(
            self,
            "expected_environment_binding_cid",
            _optional_cid(
                self.expected_environment_binding_cid,
                "expected_environment_binding_cid",
            ),
        )
        for name, bag in (
            ("type_facts", self.type_facts),
            ("effect_facts", self.effect_facts),
            ("path_facts", self.path_facts),
            ("contract_facts", self.contract_facts),
            ("capability_facts", self.capability_facts),
            ("abstract_facts", self.abstract_facts),
            ("solver_facts", self.solver_facts),
            ("neural_nominations", self.neural_nominations),
        ):
            expected = {
                "type_facts": TypeConstraint,
                "effect_facts": EffectConstraint,
                "path_facts": PathConstraint,
                "contract_facts": ContractConstraint,
                "capability_facts": CapabilityConstraint,
                "abstract_facts": AbstractConstraint,
                "solver_facts": SolverObligation,
                "neural_nominations": NeuralNomination,
            }[name]
            for item in bag:
                if not isinstance(item, expected):
                    raise SymbolicPruneError(f"{name} must contain {expected.__name__} values")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "type_facts": [item.identity_payload() for item in self.type_facts],
            "effect_facts": [item.identity_payload() for item in self.effect_facts],
            "path_facts": [item.identity_payload() for item in self.path_facts],
            "contract_facts": [item.identity_payload() for item in self.contract_facts],
            "capability_facts": [item.identity_payload() for item in self.capability_facts],
            "abstract_facts": [item.identity_payload() for item in self.abstract_facts],
            "solver_facts": [item.identity_payload() for item in self.solver_facts],
            "neural_nominations": [item.identity_payload() for item in self.neural_nominations],
            "widen_reasons": list(self.widen_reasons),
            "widen_dimensions": list(self.widen_dimensions),
            "expected_snapshot_cid": self.expected_snapshot_cid,
            "expected_environment_binding_cid": self.expected_environment_binding_cid,
        }

    @property
    def evidence_cid(self) -> str:
        return _receipt_cid(self.identity_payload())


@dataclass(frozen=True, slots=True)
class PruneRecord:
    """Why one candidate left the retained set."""

    candidate_cid: str
    reason: PruneReason | str
    stage: str
    evidence_cids: Sequence[str] = ()
    detail: str = ""

    SCHEMA: ClassVar[str] = PRUNE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid"))
        object.__setattr__(
            self, "reason", _enum_value(self.reason, PRUNE_REASON_VALUES, "reason")
        )
        object.__setattr__(self, "stage", _bounded_text(self.stage, "stage"))
        object.__setattr__(
            self,
            "evidence_cids",
            _unique_sorted(self.evidence_cids, "evidence_cid", normalize=_cid),
        )
        object.__setattr__(
            self, "detail", _bounded_text(self.detail, "detail", empty=True, limit=256)
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "candidate_cid": self.candidate_cid,
            "reason": self.reason,
            "stage": self.stage,
            "evidence_cids": list(self.evidence_cids),
            "detail": self.detail,
        }

    @property
    def record_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["record_cid"] = self.record_cid
        return payload


@dataclass(frozen=True, slots=True)
class SymbolicPruneResult:
    """Retained successors, pruned records, possibly widened frontier, receipt."""

    plan: StaticSuccessorPlan
    retained: tuple[StaticSuccessorCandidate, ...]
    pruned: tuple[PruneRecord, ...]
    frontier: UnresolvedDynamicFrontier
    receipt: SuccessorDecisionReceipt
    completeness: str

    SCHEMA: ClassVar[str] = SYMBOLIC_PRUNE_RESULT_SCHEMA
    INTERFACE: ClassVar[str] = SYMBOLIC_PRUNE_RESULT_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.plan, StaticSuccessorPlan):
            raise SymbolicPruneError("prune result requires a StaticSuccessorPlan")
        if not isinstance(self.frontier, UnresolvedDynamicFrontier):
            raise SymbolicPruneError("prune result requires UnresolvedDynamicFrontier")
        if not isinstance(self.receipt, SuccessorDecisionReceipt):
            raise SymbolicPruneError("prune result requires SuccessorDecisionReceipt")
        object.__setattr__(self, "retained", tuple(self.retained))
        object.__setattr__(self, "pruned", tuple(self.pruned))
        object.__setattr__(
            self,
            "completeness",
            _enum_value(self.completeness, COMPLETENESS_VALUES, "completeness"),
        )
        retained_ids = {item.candidate_cid for item in self.retained}
        pruned_ids = {item.candidate_cid for item in self.pruned}
        if retained_ids & pruned_ids:
            raise SymbolicPruneError("retained and pruned candidate sets must be disjoint")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "plan_cid": self.plan.plan_cid,
            "retained_cids": [item.candidate_cid for item in self.retained],
            "pruned_cids": [item.candidate_cid for item in self.pruned],
            "frontier_cid": self.frontier.frontier_cid,
            "receipt_cid": self.receipt.receipt_cid,
            "completeness": self.completeness,
        }

    @property
    def result_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_cid"] = self.result_cid
        payload["retained"] = [item.to_dict() for item in self.retained]
        payload["pruned"] = [item.to_dict() for item in self.pruned]
        payload["frontier"] = self.frontier.to_dict()
        payload["receipt"] = self.receipt.to_dict()
        return payload


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------


class SymbolicSuccessorPruner:
    """Prune static successors only with authoritative symbolic reasons."""

    INTERFACE: ClassVar[str] = SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE
    VERSION: ClassVar[str] = SYMBOLIC_SUCCESSOR_PRUNER_VERSION

    def prune(
        self,
        plan: StaticSuccessorPlan,
        *,
        evidence: SymbolicPruningEvidence | None = None,
        expected_snapshot_cid: str | None = None,
        expected_environment_binding_cid: str | None = None,
    ) -> SymbolicPruneResult:
        if not isinstance(plan, StaticSuccessorPlan):
            raise SymbolicPruneError("prune_program_successors requires a StaticSuccessorPlan")
        bundle = evidence or SymbolicPruningEvidence()
        if not isinstance(bundle, SymbolicPruningEvidence):
            raise SymbolicPruneError("evidence must be SymbolicPruningEvidence")

        stages: list[SuccessorStageDecision] = []
        expected_snapshot = expected_snapshot_cid or bundle.expected_snapshot_cid
        expected_env = (
            expected_environment_binding_cid or bundle.expected_environment_binding_cid
        )
        freshness = plan.freshness
        if expected_snapshot and expected_snapshot != plan.snapshot_cid:
            stages.append(
                _stage(
                    "snapshot_freshness",
                    StageStatus.REJECTED,
                    "stale_snapshot",
                    detail="prune refused a stale program graph",
                )
            )
            _fill_remaining_stages(stages, PRUNE_STAGE_ORDER, StageStatus.SKIPPED, "stale_graph")
            frontier = plan.frontier.union(
                reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
                unavailable_dimensions=("stale_graph",),
            )
            receipt = _prune_receipt(
                plan=plan,
                stages=stages,
                retained=plan.candidates,
                pruned=(),
                frontier=frontier,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                freshness=FreshnessState.STALE.value,
            )
            return SymbolicPruneResult(
                plan=plan,
                retained=plan.candidates,
                pruned=(),
                frontier=frontier,
                receipt=receipt,
                completeness=CompletenessVerdict.INCOMPLETE.value,
            )
        env_cid = plan.catalog.environment_binding_cid
        if expected_env and expected_env != env_cid:
            stages.append(
                _stage(
                    "snapshot_freshness",
                    StageStatus.REJECTED,
                    "stale_environment",
                )
            )
            _fill_remaining_stages(stages, PRUNE_STAGE_ORDER, StageStatus.SKIPPED, "stale_graph")
            frontier = plan.frontier.union(
                reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
                unavailable_dimensions=("stale_graph",),
            )
            receipt = _prune_receipt(
                plan=plan,
                stages=stages,
                retained=plan.candidates,
                pruned=(),
                frontier=frontier,
                completeness=CompletenessVerdict.INCOMPLETE.value,
                freshness=FreshnessState.STALE.value,
            )
            return SymbolicPruneResult(
                plan=plan,
                retained=plan.candidates,
                pruned=(),
                frontier=frontier,
                receipt=receipt,
                completeness=CompletenessVerdict.INCOMPLETE.value,
            )
        if expected_snapshot or expected_env:
            stages.append(
                _stage(
                    "snapshot_freshness",
                    StageStatus.SELECTED,
                    "bindings_fresh",
                    evidence_cids=(plan.snapshot_cid,),
                )
            )
            freshness = FreshnessState.FRESH.value
        else:
            stages.append(
                _stage("snapshot_freshness", StageStatus.SKIPPED, "no_expected_bindings")
            )

        retained = {item.candidate_cid: item for item in plan.candidates}
        pruned: dict[str, PruneRecord] = {}
        frontier = plan.frontier
        widened_unknown = False

        type_stage, type_pruned = _apply_type_stage(plan, retained, bundle)
        stages.append(type_stage)
        _absorb_pruned(retained, pruned, type_pruned)

        effect_stage, effect_pruned = _apply_effect_stage(plan, retained, bundle)
        stages.append(effect_stage)
        _absorb_pruned(retained, pruned, effect_pruned)

        path_stage, path_pruned, path_widen = _apply_path_stage(plan, retained, bundle)
        stages.append(path_stage)
        _absorb_pruned(retained, pruned, path_pruned)
        if path_widen:
            frontier = frontier.union(
                reasons=(DynamicFrontierReason.SOLVER_UNKNOWN.value,),
                unavailable_dimensions=("path_condition",),
            )
            widened_unknown = True

        contract_stage, contract_pruned = _apply_contract_stage(plan, retained, bundle)
        stages.append(contract_stage)
        _absorb_pruned(retained, pruned, contract_pruned)

        cap_stage, cap_pruned, cap_widen = _apply_capability_stage(plan, retained, bundle)
        stages.append(cap_stage)
        _absorb_pruned(retained, pruned, cap_pruned)
        if cap_widen:
            frontier = frontier.union(
                reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
                unavailable_dimensions=("capability",),
            )
            widened_unknown = True

        abs_stage, abs_pruned, abs_widen = _apply_abstract_stage(plan, retained, bundle)
        stages.append(abs_stage)
        _absorb_pruned(retained, pruned, abs_pruned)
        if abs_widen:
            frontier = frontier.union(
                reasons=(DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,),
                unavailable_dimensions=("abstract_interpretation",),
            )
            widened_unknown = True

        solver_stage, solver_pruned, solver_widen, frontier_unsat = _apply_solver_stage(
            plan, retained, bundle, frontier
        )
        stages.append(solver_stage)
        _absorb_pruned(retained, pruned, solver_pruned)
        if solver_widen:
            frontier = frontier.union(
                reasons=(DynamicFrontierReason.SOLVER_UNKNOWN.value,),
                unavailable_dimensions=("solver",),
            )
            widened_unknown = True
        if frontier_unsat:
            frontier = UnresolvedDynamicFrontier(
                datasets_frontier_cids=frontier.datasets_frontier_cids,
                widened=False,
            )
            retained = {
                cid: item for cid, item in retained.items() if not item.unknown
            }

        if bundle.neural_nominations:
            stages.append(
                _stage(
                    "neural_nomination",
                    StageStatus.REJECTED,
                    "neural_cannot_prune",
                    detail="neural/ANN/similarity nominations are proposal-only",
                )
            )
        else:
            stages.append(
                _stage("neural_nomination", StageStatus.SKIPPED, "no_neural_nominations")
            )

        if bundle.widen_reasons or bundle.widen_dimensions:
            frontier = frontier.union(
                reasons=bundle.widen_reasons,
                unavailable_dimensions=bundle.widen_dimensions,
            )
            widened_unknown = True

        remaining = tuple(sorted(retained.values(), key=lambda item: item.candidate_cid))
        prune_records = tuple(sorted(pruned.values(), key=lambda item: item.candidate_cid))
        remaining_unknown = any(item.unknown for item in remaining) or frontier.open
        authoritative_pruned = bool(prune_records)
        if not remaining and not remaining_unknown and authoritative_pruned:
            completeness = CompletenessVerdict.IMPOSSIBLE.value
        elif remaining_unknown or plan.completeness == CompletenessVerdict.INCOMPLETE.value or widened_unknown:
            completeness = CompletenessVerdict.INCOMPLETE.value
        else:
            completeness = CompletenessVerdict.COMPLETE.value
        stages.append(
            _stage(
                "completeness",
                StageStatus.SELECTED,
                completeness,
                detail="unknown widens; impossible requires authoritative removal of every possibility",
            )
        )
        receipt = _prune_receipt(
            plan=plan,
            stages=stages,
            retained=remaining,
            pruned=prune_records,
            frontier=frontier,
            completeness=completeness,
            freshness=freshness,
        )
        return SymbolicPruneResult(
            plan=plan,
            retained=remaining,
            pruned=prune_records,
            frontier=frontier,
            receipt=receipt,
            completeness=completeness,
        )


def prune_program_successors(
    plan: StaticSuccessorPlan,
    *,
    evidence: SymbolicPruningEvidence | None = None,
    expected_snapshot_cid: str | None = None,
    expected_environment_binding_cid: str | None = None,
    pruner: SymbolicSuccessorPruner | None = None,
) -> SymbolicPruneResult:
    """Public interface: authoritative symbolic successor pruning."""

    active = pruner or SymbolicSuccessorPruner()
    return active.prune(
        plan,
        evidence=evidence,
        expected_snapshot_cid=expected_snapshot_cid,
        expected_environment_binding_cid=expected_environment_binding_cid,
    )


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _absorb_pruned(
    retained: dict[str, StaticSuccessorCandidate],
    pruned: dict[str, PruneRecord],
    records: Sequence[PruneRecord],
) -> None:
    for record in records:
        if record.candidate_cid in retained:
            pruned[record.candidate_cid] = record
            del retained[record.candidate_cid]


def _graph_type_annotation(plan: StaticSuccessorPlan, node_cid: str) -> str | None:
    try:
        node = plan.catalog.node(node_cid)
    except StaticSuccessorError:
        return None
    meta = node.metadata if isinstance(node.metadata, Mapping) else {}
    annotation = meta.get("annotation")
    if isinstance(annotation, str) and annotation:
        return annotation
    for edge in plan.catalog.outgoing(node_cid):
        if str(edge.edge_kind) != ProgramGraphEdgeKind.TYPE_OF.value:
            continue
        target = plan.catalog.node(edge.target_node_cid)
        target_meta = target.metadata if isinstance(target.metadata, Mapping) else {}
        text = target_meta.get("annotation")
        if isinstance(text, str) and text:
            return text
    return None


def _graph_effects(plan: StaticSuccessorPlan, node_cid: str) -> tuple[tuple[str, bool], ...]:
    effects: list[tuple[str, bool]] = []
    try:
        plan.catalog.node(node_cid)
    except StaticSuccessorError:
        return ()
    for edge in plan.catalog.outgoing(node_cid):
        if str(edge.edge_kind) != ProgramGraphEdgeKind.EFFECT_OF.value:
            continue
        target = plan.catalog.node(edge.target_node_cid)
        name = target.logical_name.rsplit(".effect.", 1)[-1]
        definite = (
            str(edge.resolution_status) == ResolutionStatus.DEFINITE.value
            and not edge.unavailable_dimensions
            and not target.unavailable_dimensions
        )
        effects.append((name, definite))
    return tuple(effects)


def _types_contradict(observed: str, required: str) -> bool:
    left = _norm_annotation(observed)
    right = _norm_annotation(required)
    if not left or not right:
        return False
    if left.lower() in DYNAMIC_TYPE_MARKERS or right.lower() in DYNAMIC_TYPE_MARKERS:
        return False
    if left == right:
        return False
    return (left, right) in INCOMPATIBLE_TYPE_PAIRS or (
        left.split("[", 1)[0],
        right.split("[", 1)[0],
    ) in INCOMPATIBLE_TYPE_PAIRS


def _apply_type_stage(
    plan: StaticSuccessorPlan,
    retained: Mapping[str, StaticSuccessorCandidate],
    evidence: SymbolicPruningEvidence,
) -> tuple[SuccessorStageDecision, tuple[PruneRecord, ...]]:
    facts = evidence.type_facts
    if not facts:
        graph_has_types = any(
            str(edge.edge_kind) == ProgramGraphEdgeKind.TYPE_OF.value
            for edge in plan.catalog.edges
        )
        if not graph_has_types:
            return _stage("type", StageStatus.UNAVAILABLE, "type_evidence_unavailable"), ()
        return _stage("type", StageStatus.SKIPPED, "no_authoritative_type_constraint"), ()
    pruned: list[PruneRecord] = []
    used_authoritative = False
    rejected_non_auth = False
    for fact in facts:
        if fact.contradiction and not fact.authoritative:
            rejected_non_auth = True
            continue
        if not fact.authoritative:
            continue
        contradiction = fact.contradiction or _types_contradict(
            fact.observed_annotation, fact.required_annotation
        )
        if not contradiction:
            continue
        used_authoritative = True
        for candidate in retained.values():
            if candidate.successor_node_cid != fact.successor_node_cid:
                continue
            if candidate.unknown:
                continue
            pruned.append(
                PruneRecord(
                    candidate_cid=candidate.candidate_cid,
                    reason=PruneReason.TYPE_CONTRADICTION.value,
                    stage="type",
                    evidence_cids=tuple(
                        cid for cid in (fact.evidence_cid, fact.constraint_cid) if cid
                    ),
                    detail="authoritative type contradiction",
                )
            )
    if used_authoritative:
        status, reason = StageStatus.SELECTED, "type_constraints_applied"
    elif rejected_non_auth:
        status, reason = StageStatus.REJECTED, "non_authoritative_type_ignored"
    else:
        status, reason = StageStatus.SKIPPED, "type_facts_not_contradictory"
    return _stage("type", status, reason), tuple(pruned)


def _apply_effect_stage(
    plan: StaticSuccessorPlan,
    retained: Mapping[str, StaticSuccessorCandidate],
    evidence: SymbolicPruningEvidence,
) -> tuple[SuccessorStageDecision, tuple[PruneRecord, ...]]:
    facts = evidence.effect_facts
    if not facts:
        return _stage("effect", StageStatus.SKIPPED, "no_effect_constraints"), ()
    pruned: list[PruneRecord] = []
    used = False
    rejected = False
    for fact in facts:
        if fact.forbidden and fact.definite and not fact.authoritative:
            rejected = True
            continue
        if not (fact.authoritative and fact.forbidden and fact.definite):
            continue
        used = True
        for candidate in retained.values():
            if candidate.unknown:
                continue
            match = candidate.successor_node_cid == fact.successor_node_cid
            if not match:
                graph_effects = _graph_effects(plan, candidate.successor_node_cid)
                match = any(name == fact.effect and definite for name, definite in graph_effects)
                if not match:
                    subject_effects = _graph_effects(plan, candidate.subject_node_cid)
                    match = any(name == fact.effect and definite for name, definite in subject_effects)
            if not match:
                continue
            pruned.append(
                PruneRecord(
                    candidate_cid=candidate.candidate_cid,
                    reason=PruneReason.FORBIDDEN_EFFECT.value,
                    stage="effect",
                    evidence_cids=tuple(
                        cid for cid in (fact.evidence_cid, fact.constraint_cid) if cid
                    ),
                    detail=fact.effect,
                )
            )
    if used:
        status, reason = StageStatus.SELECTED, "effect_constraints_applied"
    elif rejected:
        status, reason = StageStatus.REJECTED, "non_authoritative_effect_ignored"
    else:
        status, reason = StageStatus.SKIPPED, "effect_facts_not_forbidden"
    return _stage("effect", status, reason), tuple(pruned)


def _apply_path_stage(
    plan: StaticSuccessorPlan,
    retained: Mapping[str, StaticSuccessorCandidate],
    evidence: SymbolicPruningEvidence,
) -> tuple[SuccessorStageDecision, tuple[PruneRecord, ...], bool]:
    del plan
    facts = evidence.path_facts
    if not facts:
        return _stage("path_condition", StageStatus.SKIPPED, "no_path_constraints"), (), False
    pruned: list[PruneRecord] = []
    used = False
    rejected = False
    widen = False
    for fact in facts:
        if fact.satisfiable is None:
            widen = True
            continue
        if fact.satisfiable is False and not fact.authoritative:
            rejected = True
            continue
        if not (fact.authoritative and fact.satisfiable is False):
            continue
        used = True
        for candidate in retained.values():
            if candidate.successor_edge_cid != fact.successor_edge_cid:
                continue
            if candidate.unknown:
                continue
            pruned.append(
                PruneRecord(
                    candidate_cid=candidate.candidate_cid,
                    reason=PruneReason.UNSATISFIABLE_PATH.value,
                    stage="path_condition",
                    evidence_cids=tuple(
                        cid for cid in (fact.evidence_cid, fact.constraint_cid) if cid
                    ),
                    detail=fact.condition,
                )
            )
    if used:
        status, reason = StageStatus.SELECTED, "path_constraints_applied"
    elif widen:
        status, reason = StageStatus.ESCALATED, "path_condition_unknown"
    elif rejected:
        status, reason = StageStatus.REJECTED, "non_authoritative_path_ignored"
    else:
        status, reason = StageStatus.SKIPPED, "path_facts_satisfiable"
    return _stage("path_condition", status, reason), tuple(pruned), widen


def _apply_contract_stage(
    plan: StaticSuccessorPlan,
    retained: Mapping[str, StaticSuccessorCandidate],
    evidence: SymbolicPruningEvidence,
) -> tuple[SuccessorStageDecision, tuple[PruneRecord, ...]]:
    facts = list(evidence.contract_facts)
    if not facts:
        graph_unknown = any(
            str(getattr(record, "discharge_status", ""))
            in {
                ContractDischargeStatus.UNKNOWN.value,
                ContractDischargeStatus.UNAVAILABLE.value,
            }
            for record in plan.catalog.contract_states
        )
        if graph_unknown:
            return _stage("contract", StageStatus.UNAVAILABLE, "contract_discharge_unknown"), ()
        return _stage("contract", StageStatus.SKIPPED, "no_contract_constraints"), ()
    pruned: list[PruneRecord] = []
    used = False
    rejected = False
    unavailable = False
    for fact in facts:
        status = fact.discharge_status
        if status in {
            ContractDischargeStatus.UNKNOWN.value,
            ContractDischargeStatus.ASSUMED.value,
            ContractDischargeStatus.UNAVAILABLE.value,
        }:
            unavailable = True
            continue
        if status == ContractDischargeStatus.VIOLATED.value and not fact.authoritative:
            rejected = True
            continue
        if not (fact.authoritative and status == ContractDischargeStatus.VIOLATED.value):
            continue
        used = True
        for candidate in retained.values():
            if candidate.unknown:
                continue
            if candidate.successor_node_cid != fact.successor_node_cid and candidate.subject_node_cid != fact.successor_node_cid:
                continue
            pruned.append(
                PruneRecord(
                    candidate_cid=candidate.candidate_cid,
                    reason=PruneReason.VIOLATED_CONTRACT.value,
                    stage="contract",
                    evidence_cids=tuple(
                        cid
                        for cid in (
                            fact.evidence_cid,
                            fact.specification_cid,
                            fact.constraint_cid,
                        )
                        if cid
                    ),
                    detail=fact.contract_kind,
                )
            )
    if used:
        stage_status, reason = StageStatus.SELECTED, "contract_constraints_applied"
    elif unavailable and not facts:
        stage_status, reason = StageStatus.UNAVAILABLE, "contract_discharge_unknown"
    elif rejected:
        stage_status, reason = StageStatus.REJECTED, "non_authoritative_contract_ignored"
    elif unavailable:
        stage_status, reason = StageStatus.UNAVAILABLE, "contract_not_discharged"
    else:
        stage_status, reason = StageStatus.SKIPPED, "contracts_not_violated"
    return _stage("contract", stage_status, reason), tuple(pruned)


def _apply_capability_stage(
    plan: StaticSuccessorPlan,
    retained: Mapping[str, StaticSuccessorCandidate],
    evidence: SymbolicPruningEvidence,
) -> tuple[SuccessorStageDecision, tuple[PruneRecord, ...], bool]:
    facts = evidence.capability_facts
    if not facts:
        return _stage("capability", StageStatus.SKIPPED, "no_capability_constraints"), (), False
    pruned: list[PruneRecord] = []
    used = False
    widen = False
    rejected = False
    for fact in facts:
        if not fact.available:
            widen = True
            continue
        if not fact.allowed and not fact.authoritative:
            rejected = True
            continue
        if not (fact.authoritative and not fact.allowed and fact.available):
            continue
        used = True
        for candidate in retained.values():
            if candidate.unknown:
                continue
            effects = _graph_effects(plan, candidate.successor_node_cid)
            subject_effects = _graph_effects(plan, candidate.subject_node_cid)
            match = any(name == fact.effect and definite for name, definite in (*effects, *subject_effects))
            if not match and fact.effect in candidate.unavailable_dimensions:
                continue
            if not match:
                try:
                    node = plan.catalog.node(candidate.successor_node_cid)
                except StaticSuccessorError:
                    continue
                match = fact.effect in node.logical_name
            if not match:
                continue
            pruned.append(
                PruneRecord(
                    candidate_cid=candidate.candidate_cid,
                    reason=PruneReason.CAPABILITY_DENIED.value,
                    stage="capability",
                    evidence_cids=tuple(
                        cid for cid in (fact.evidence_cid, fact.constraint_cid) if cid
                    ),
                    detail=fact.effect,
                )
            )
    if used:
        status, reason = StageStatus.SELECTED, "capability_constraints_applied"
    elif widen:
        status, reason = StageStatus.UNAVAILABLE, "capability_unavailable"
    elif rejected:
        status, reason = StageStatus.REJECTED, "non_authoritative_capability_ignored"
    else:
        status, reason = StageStatus.SKIPPED, "capabilities_allow_effects"
    return _stage("capability", status, reason), tuple(pruned), widen


def _apply_abstract_stage(
    plan: StaticSuccessorPlan,
    retained: Mapping[str, StaticSuccessorCandidate],
    evidence: SymbolicPruningEvidence,
) -> tuple[SuccessorStageDecision, tuple[PruneRecord, ...], bool]:
    del plan
    facts = evidence.abstract_facts
    if not facts:
        return (
            _stage("abstract_interpretation", StageStatus.UNAVAILABLE, "abstract_interp_unavailable"),
            (),
            False,
        )
    pruned: list[PruneRecord] = []
    used = False
    widen = False
    rejected = False
    for fact in facts:
        if fact.unknown:
            widen = True
            continue
        if fact.contradiction and not fact.authoritative:
            rejected = True
            continue
        if not (fact.authoritative and fact.contradiction):
            continue
        used = True
        for candidate in retained.values():
            if candidate.unknown:
                continue
            if candidate.successor_node_cid != fact.successor_node_cid:
                continue
            pruned.append(
                PruneRecord(
                    candidate_cid=candidate.candidate_cid,
                    reason=PruneReason.ABSTRACT_CONTRADICTION.value,
                    stage="abstract_interpretation",
                    evidence_cids=tuple(
                        cid for cid in (fact.evidence_cid, fact.constraint_cid) if cid
                    ),
                    detail=fact.domain,
                )
            )
    if used:
        status, reason = StageStatus.SELECTED, "abstract_constraints_applied"
    elif widen:
        status, reason = StageStatus.ESCALATED, "abstract_unknown"
    elif rejected:
        status, reason = StageStatus.REJECTED, "non_authoritative_abstract_ignored"
    else:
        status, reason = StageStatus.SKIPPED, "abstract_facts_consistent"
    return _stage("abstract_interpretation", status, reason), tuple(pruned), widen


def _apply_solver_stage(
    plan: StaticSuccessorPlan,
    retained: Mapping[str, StaticSuccessorCandidate],
    evidence: SymbolicPruningEvidence,
    frontier: UnresolvedDynamicFrontier,
) -> tuple[SuccessorStageDecision, tuple[PruneRecord, ...], bool, bool]:
    facts = evidence.solver_facts
    if not facts:
        return _stage("solver", StageStatus.UNAVAILABLE, "solver_evidence_unavailable"), (), False, False
    pruned: list[PruneRecord] = []
    used = False
    widen = False
    rejected = False
    frontier_unsat = False
    for fact in facts:
        verdict = fact.verdict
        if verdict in {SolverVerdict.UNKNOWN.value, SolverVerdict.TIMEOUT.value} or fact.timeout:
            widen = True
            continue
        if verdict == SolverVerdict.UNSAT.value and not fact.authoritative:
            rejected = True
            continue
        if verdict != SolverVerdict.UNSAT.value or not fact.authoritative:
            continue
        used = True
        if fact.covers_frontier and fact.target_cid in {
            plan.subject_node_cid,
            frontier.frontier_cid,
            plan.snapshot_cid,
        }:
            frontier_unsat = True
        for candidate in retained.values():
            targeted = fact.target_cid in {
                candidate.candidate_cid,
                candidate.successor_node_cid,
                candidate.successor_edge_cid,
                candidate.subject_node_cid,
            }
            if fact.covers_frontier:
                targeted = targeted or candidate.unknown
            if not targeted:
                continue
            if candidate.unknown and not fact.covers_frontier:
                continue
            pruned.append(
                PruneRecord(
                    candidate_cid=candidate.candidate_cid,
                    reason=PruneReason.SOLVER_UNSAT.value,
                    stage="solver",
                    evidence_cids=tuple(
                        cid for cid in (fact.obligation_cid, fact.constraint_cid) if cid
                    ),
                    detail=verdict,
                )
            )
    if used:
        status, reason = StageStatus.SELECTED, "solver_unsat_applied"
    elif widen:
        status, reason = StageStatus.ESCALATED, "solver_unknown_or_timeout"
    elif rejected:
        status, reason = StageStatus.REJECTED, "non_authoritative_solver_ignored"
    else:
        status, reason = StageStatus.SKIPPED, "solver_sat_or_unrelated"
    return _stage("solver", status, reason), tuple(pruned), widen, frontier_unsat


def _prune_receipt(
    *,
    plan: StaticSuccessorPlan,
    stages: Sequence[SuccessorStageDecision],
    retained: Sequence[StaticSuccessorCandidate],
    pruned: Sequence[PruneRecord],
    frontier: UnresolvedDynamicFrontier,
    completeness: str,
    freshness: str,
) -> SuccessorDecisionReceipt:
    present = {item.stage for item in stages}
    missing = [name for name in PRUNE_STAGE_ORDER if name not in present]
    if missing:
        raise SymbolicPruneError(f"prune receipt missing stages {missing}")
    return SuccessorDecisionReceipt(
        snapshot_cid=plan.snapshot_cid,
        query_family=plan.query_family,
        completeness=completeness,
        stages=tuple(stages),
        candidate_cids=tuple(item.candidate_cid for item in retained),
        pruned_cids=tuple(item.candidate_cid for item in pruned),
        frontier_cid=frontier.frontier_cid,
        subject_node_cid=plan.subject_node_cid,
        parent_receipt_cid=plan.receipt.receipt_cid,
        freshness=freshness,
        producer_id=PRODUCER_ID,
        evidence_term=SYMBOLIC_PRUNING_EVIDENCE,
    )


__all__ = [
    "IMPORT_SCAN_PERFORMED",
    "AbstractConstraint",
    "CapabilityConstraint",
    "ContractConstraint",
    "EffectConstraint",
    "NeuralNomination",
    "PRUNE_STAGE_ORDER",
    "PathConstraint",
    "PruneReason",
    "PruneRecord",
    "SYMBOLIC_PRUNING_EVIDENCE",
    "SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE",
    "SolverObligation",
    "SolverVerdict",
    "SymbolicPruneError",
    "SymbolicPruneResult",
    "SymbolicPruningEvidence",
    "SymbolicSuccessorPruner",
    "TypeConstraint",
    "prune_program_successors",
]
