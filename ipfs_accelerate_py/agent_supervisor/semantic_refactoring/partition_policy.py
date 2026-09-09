"""SPAR-014 multi-objective partition comparison and policy.

This module extends current supervisor partition orchestration with
``PartitionObjectiveProfile@1`` and ``PartitionComparisonReceipt@1``.  It
evaluates complete reproducible objective breakdowns for SPAR-013 candidates
and fail-closes on SCC, state, consumer, compatibility, ordering, resource,
frontier, proof, or transaction constraints.

Soft scores never override a hard constraint.  Rejected comparisons remain
negative evidence.  Ranking is nomination-only and cannot authorize a
transition, completion, or competing authority.  Observational metadata is
excluded from identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    ANALYZER_ID as SPAR013_ANALYZER_ID,
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
    ConstraintClass,
    PartitionGenerationReceipt,
    ProgramPartitionCandidate,
    compile_partition_evidence,
)


TASK_ID: Final[str] = "SPAR-014"
GOAL_ID: Final[str] = "SPAR-G032"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_policy@1"
)

PARTITION_OBJECTIVE_PROFILE_INTERFACE: Final[str] = "PartitionObjectiveProfile@1"
PARTITION_OBJECTIVE_BREAKDOWN_INTERFACE: Final[str] = (
    "PartitionObjectiveBreakdown@1"
)
PARTITION_COMPARISON_RECEIPT_INTERFACE: Final[str] = (
    "PartitionComparisonReceipt@1"
)

PARTITION_OBJECTIVE_PROFILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/partition-objective-profile@1"
)
PARTITION_OBJECTIVE_BREAKDOWN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/partition-objective-breakdown@1"
)
PARTITION_COMPARISON_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/partition-comparison-receipt@1"
)

PARTITION_CONTRACT_VERSION: Final[str] = "1"

PARTITION_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
PARTITION_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
PARTITION_CAN_CREATE_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_CANDIDATES: Final[int] = 16_384
MAX_VIOLATIONS: Final[int] = 4_096
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_WEIGHT: Final[int] = 1_024
MAX_SCORE: Final[int] = 1_000_000

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

_FORBIDDEN_CAPSULE_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "FunctionSemanticCapsule",
        "MethodSemanticCapsule",
        "ClassSemanticCapsule",
        "TopLevelBlockCapsule",
        "ModuleSemanticCapsule",
        "PackageSemanticCapsule",
        "CallsiteSemanticCapsule",
        "StateOwnerCapsule",
        "RegistrationCapsule",
        "ResourceLifecycleCapsule",
    }
)

HARD_CONSTRAINT_FAMILIES: Final[tuple[str, ...]] = (
    "scc",
    "state",
    "consumer",
    "compatibility",
    "ordering",
    "resource",
    "frontier",
    "proof",
    "transaction",
)

SOFT_OBJECTIVES: Final[tuple[str, ...]] = (
    "call",
    "data",
    "contract",
    "test",
    "proof",
    "cochange",
    "lexical",
    "vector",
    "trace",
)

ORDERING_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "decorator_order",
        "happens_before",
        "initialization_order",
        "registration_order",
    }
)
RESOURCE_EDGE_KINDS: Final[frozenset[str]] = frozenset({"uses_resource"})
TRANSACTION_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {"atomic_with", "lock", "transaction"}
)
PROOF_EDGE_KINDS: Final[frozenset[str]] = frozenset({"proves"})
CALL_EDGE_KINDS: Final[frozenset[str]] = frozenset({"calls"})
DATA_EDGE_KINDS: Final[frozenset[str]] = frozenset({"derived_from", "uses"})
TEST_EDGE_KINDS: Final[frozenset[str]] = frozenset({"tests"})
TRACE_EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    {"runtime_observation", "replayed_counterexample"}
)
_NON_RANKING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {"vector_candidate", "model_hypothesis", "heuristic"}
)

_SCC_VIOLATIONS: Final[frozenset[str]] = frozenset(
    {
        ConstraintClass.SCC_SPLIT.value,
        ConstraintClass.OVERSIZED_CYCLE.value,
        ConstraintClass.UNKNOWN_MEMBER.value,
    }
)
_STATE_VIOLATIONS: Final[frozenset[str]] = frozenset(
    {
        ConstraintClass.UNIQUE_OWNER_SPLIT.value,
        ConstraintClass.UNIQUE_OWNER_OVERLAP.value,
        ConstraintClass.UNRESOLVED_STATE.value,
        ConstraintClass.UNRESOLVED_ALIAS.value,
    }
)


class PartitionPolicyError(ValueError):
    """Fail-closed violation of a SPAR-014 partition-policy contract."""


class HardConstraintFamily(str, Enum):
    SCC = "scc"
    STATE = "state"
    CONSUMER = "consumer"
    COMPATIBILITY = "compatibility"
    ORDERING = "ordering"
    RESOURCE = "resource"
    FRONTIER = "frontier"
    PROOF = "proof"
    TRANSACTION = "transaction"


class SoftObjective(str, Enum):
    CALL = "call"
    DATA = "data"
    CONTRACT = "contract"
    TEST = "test"
    PROOF = "proof"
    COCHANGE = "cochange"
    LEXICAL = "lexical"
    VECTOR = "vector"
    TRACE = "trace"


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise PartitionPolicyError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise PartitionPolicyError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise PartitionPolicyError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise PartitionPolicyError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise PartitionPolicyError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise PartitionPolicyError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PartitionPolicyError(f"{name} must be a boolean")
    return value


def _nat(value: Any, name: str, *, limit: int) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise PartitionPolicyError(f"{name} must be an integer")
    if value < 0:
        raise PartitionPolicyError(f"{name} must be non-negative")
    if value > limit:
        raise PartitionPolicyError(f"{name} exceeds maximum")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise PartitionPolicyError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise PartitionPolicyError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise PartitionPolicyError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise PartitionPolicyError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise PartitionPolicyError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise PartitionPolicyError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise PartitionPolicyError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise PartitionPolicyError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise PartitionPolicyError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise PartitionPolicyError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise PartitionPolicyError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise PartitionPolicyError(f"unknown {name}: {text}") from exc


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise PartitionPolicyError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def _project(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _project(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_project(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _project(to_dict())
    raise PartitionPolicyError(
        f"unsupported projected type {type(value).__name__}"
    )


def _sequence_maps(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        projected = _project(value)
        if isinstance(projected, list):
            value = projected
        else:
            raise PartitionPolicyError(f"{name} must be a list")
    items = []
    for item in value:
        projected = _project(item)
        if not isinstance(projected, dict):
            raise PartitionPolicyError(f"{name} items must be objects")
        items.append(projected)
    if len(items) > MAX_MEMBERS:
        raise PartitionPolicyError(f"{name} exceeds maximum length")
    return tuple(items)


def partition_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def _family_list(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise PartitionPolicyError(f"{name} must be a list")
    ordered = tuple(_enum(item, HardConstraintFamily, name) for item in values)
    if tuple(sorted(ordered)) != tuple(sorted(HARD_CONSTRAINT_FAMILIES)):
        raise PartitionPolicyError(
            "objective profile must include every hard constraint family"
        )
    if len(ordered) != len(set(ordered)):
        raise PartitionPolicyError(f"{name} must not contain duplicates")
    if ordered != HARD_CONSTRAINT_FAMILIES:
        raise PartitionPolicyError(
            "hard constraint families must remain in canonical order"
        )
    return ordered


def _objective_list(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise PartitionPolicyError(f"{name} must be a list")
    ordered = tuple(_enum(item, SoftObjective, name) for item in values)
    if tuple(sorted(ordered)) != tuple(sorted(SOFT_OBJECTIVES)):
        raise PartitionPolicyError(
            "objective profile must include every soft objective"
        )
    if len(ordered) != len(set(ordered)):
        raise PartitionPolicyError(f"{name} must not contain duplicates")
    if ordered != SOFT_OBJECTIVES:
        raise PartitionPolicyError("soft objectives must remain in canonical order")
    return ordered


def _weight_table(values: Any) -> tuple[int, ...]:
    if isinstance(values, Mapping):
        missing = [item for item in SOFT_OBJECTIVES if item not in values]
        extra = [str(key) for key in values if str(key) not in SOFT_OBJECTIVES]
        if missing:
            raise PartitionPolicyError(
                f"soft weights missing objectives: {missing}"
            )
        if extra:
            raise PartitionPolicyError(f"unknown soft weight: {sorted(extra)}")
        return tuple(
            _nat(values[item], f"soft_weights.{item}", limit=MAX_WEIGHT)
            for item in SOFT_OBJECTIVES
        )
    if not isinstance(values, (list, tuple)):
        raise PartitionPolicyError("soft_weights must be a list or object")
    if len(values) != len(SOFT_OBJECTIVES):
        raise PartitionPolicyError("soft_weights must cover every soft objective")
    weights: list[int] = []
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            objective = _enum(item.get("objective"), SoftObjective, "objective")
            if objective != SOFT_OBJECTIVES[index]:
                raise PartitionPolicyError(
                    "soft weight objectives must remain in canonical order"
                )
            weights.append(
                _nat(item.get("weight"), f"soft_weights.{objective}", limit=MAX_WEIGHT)
            )
        else:
            weights.append(
                _nat(item, f"soft_weights[{index}]", limit=MAX_WEIGHT)
            )
    return tuple(weights)


@dataclass(frozen=True, slots=True)
class PartitionObjectiveProfile:
    """Complete versioned objective profile. Not ranking authority."""

    hard_constraint_families: Sequence[str] = HARD_CONSTRAINT_FAMILIES
    soft_objectives: Sequence[str] = SOFT_OBJECTIVES
    soft_weights: Mapping[str, int] | Sequence[int] | Sequence[Mapping[str, Any]] = ()
    vector_may_reorder_admitted: bool = True
    profile_id: str = ANALYZER_ID

    interface: ClassVar[str] = PARTITION_OBJECTIVE_PROFILE_INTERFACE
    schema: ClassVar[str] = PARTITION_OBJECTIVE_PROFILE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "profile_id",
            "hard_constraint_families",
            "soft_objectives",
            "soft_weights",
            "vector_may_reorder_admitted",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "soft_signals_cannot_override_hard_constraints",
            "profile_cid",
        }
    )

    def __post_init__(self) -> None:
        families = _family_list(
            list(self.hard_constraint_families), "hard_constraint_families"
        )
        objectives = _objective_list(list(self.soft_objectives), "soft_objectives")
        weights_in = self.soft_weights
        if weights_in in (None, (), {}):
            weights_in = {item: 1 for item in SOFT_OBJECTIVES}
            weights_in[SoftObjective.VECTOR.value] = 0
        weights = _weight_table(weights_in)
        object.__setattr__(self, "hard_constraint_families", families)
        object.__setattr__(self, "soft_objectives", objectives)
        object.__setattr__(
            self,
            "soft_weights",
            tuple(
                {"objective": objective, "weight": weight}
                for objective, weight in zip(objectives, weights)
            ),
        )
        object.__setattr__(
            self,
            "vector_may_reorder_admitted",
            _bool(
                self.vector_may_reorder_admitted, "vector_may_reorder_admitted"
            ),
        )
        object.__setattr__(self, "profile_id", _text(self.profile_id, "profile_id"))

    def weight_for(self, objective: str) -> int:
        for item in self.soft_weights:
            if item["objective"] == objective:
                return int(item["weight"])
        raise PartitionPolicyError(f"missing soft weight: {objective}")

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def soft_signals_cannot_override_hard_constraints(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PARTITION_OBJECTIVE_PROFILE_SCHEMA,
            "interface": PARTITION_OBJECTIVE_PROFILE_INTERFACE,
            "profile_id": self.profile_id,
            "hard_constraint_families": list(self.hard_constraint_families),
            "soft_objectives": list(self.soft_objectives),
            "soft_weights": [dict(item) for item in self.soft_weights],
            "vector_may_reorder_admitted": self.vector_may_reorder_admitted,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "soft_signals_cannot_override_hard_constraints": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def profile_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["profile_cid"] = self.profile_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionObjectiveProfile":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("profile_cid")
        if payload.pop("schema") != PARTITION_OBJECTIVE_PROFILE_SCHEMA:
            raise PartitionPolicyError("unsupported PartitionObjectiveProfile schema")
        if payload.pop("interface") != PARTITION_OBJECTIVE_PROFILE_INTERFACE:
            raise PartitionPolicyError(
                "unsupported PartitionObjectiveProfile interface"
            )
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "soft_signals_cannot_override_hard_constraints",
        ):
            expected = flag != "soft_signals_cannot_override_hard_constraints"
            claimed_flag = payload.pop(flag)
            if expected and claimed_flag is not False:
                raise PartitionPolicyError(f"profile cannot claim {flag}")
            if not expected and claimed_flag is not True:
                raise PartitionPolicyError(
                    "profile must retain soft-cannot-override-hard"
                )
        result = cls(**payload)
        _verify_cid(claimed, result.profile_cid, "PartitionObjectiveProfile profile_cid")
        return result


def default_partition_objective_profile() -> PartitionObjectiveProfile:
    return PartitionObjectiveProfile()


@dataclass(frozen=True, slots=True)
class PartitionHardResult:
    """One complete hard-constraint family evaluation."""

    family: HardConstraintFamily | str
    passed: bool
    violations: Sequence[str] = ()
    witness_ids: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "family", _enum(self.family, HardConstraintFamily, "family")
        )
        object.__setattr__(self, "passed", _bool(self.passed, "passed"))
        violations = _unique_sorted_text(
            list(self.violations), "violations", limit=MAX_VIOLATIONS
        )
        object.__setattr__(self, "violations", violations)
        object.__setattr__(
            self,
            "witness_ids",
            _unique_sorted_text(
                list(self.witness_ids), "witness_ids", limit=MAX_MEMBERS
            ),
        )
        if self.passed and violations:
            raise PartitionPolicyError(
                "passed hard constraint cannot retain violations"
            )
        if not self.passed and not violations:
            raise PartitionPolicyError(
                "failed hard constraint must retain a violation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "passed": self.passed,
            "violations": list(self.violations),
            "witness_ids": list(self.witness_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionHardResult":
        payload = _mapping(data, "PartitionHardResult")
        return cls(
            family=payload.get("family"),
            passed=payload.get("passed"),
            violations=payload.get("violations") or (),
            witness_ids=payload.get("witness_ids") or (),
        )


@dataclass(frozen=True, slots=True)
class PartitionSoftScore:
    """One integer soft-objective score. Never a hard-constraint override."""

    objective: SoftObjective | str
    observed: int = 0
    applied: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "objective",
            _enum(self.objective, SoftObjective, "objective"),
        )
        object.__setattr__(
            self, "observed", _nat(self.observed, "observed", limit=MAX_SCORE)
        )
        object.__setattr__(
            self, "applied", _nat(self.applied, "applied", limit=MAX_SCORE)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "observed": self.observed,
            "applied": self.applied,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionSoftScore":
        payload = _mapping(data, "PartitionSoftScore")
        return cls(
            objective=payload.get("objective"),
            observed=payload.get("observed", 0),
            applied=payload.get("applied", 0),
        )


def _coerce_hard_results(values: Any) -> tuple[PartitionHardResult, ...]:
    if isinstance(values, (list, tuple)):
        items = tuple(
            item
            if isinstance(item, PartitionHardResult)
            else PartitionHardResult.from_dict(item)
            for item in values
        )
    else:
        raise PartitionPolicyError("hard_results must be a list")
    by_family = {item.family: item for item in items}
    if set(by_family) != set(HARD_CONSTRAINT_FAMILIES):
        raise PartitionPolicyError(
            "objective breakdown must include every hard constraint family"
        )
    if len(items) != len(HARD_CONSTRAINT_FAMILIES):
        raise PartitionPolicyError("hard_results must not contain duplicates")
    return tuple(by_family[family] for family in HARD_CONSTRAINT_FAMILIES)


def _coerce_soft_scores(values: Any) -> tuple[PartitionSoftScore, ...]:
    if isinstance(values, (list, tuple)):
        items = tuple(
            item
            if isinstance(item, PartitionSoftScore)
            else PartitionSoftScore.from_dict(item)
            for item in values
        )
    else:
        raise PartitionPolicyError("soft_scores must be a list")
    by_objective = {item.objective: item for item in items}
    if set(by_objective) != set(SOFT_OBJECTIVES):
        raise PartitionPolicyError(
            "objective breakdown must include every soft objective"
        )
    if len(items) != len(SOFT_OBJECTIVES):
        raise PartitionPolicyError("soft_scores must not contain duplicates")
    return tuple(by_objective[item] for item in SOFT_OBJECTIVES)


@dataclass(frozen=True, slots=True)
class PartitionObjectiveBreakdown:
    """Complete reproducible objective breakdown for one candidate."""

    tree_id: str
    candidate_cid: str
    hard_results: Sequence[PartitionHardResult | Mapping[str, Any]]
    soft_scores: Sequence[PartitionSoftScore | Mapping[str, Any]]
    hard_cut_count: int = 0
    member_count: int = 0
    ranked: bool = False
    advisory: bool = False
    evidence_class: str = "exact_static_fact"

    interface: ClassVar[str] = PARTITION_OBJECTIVE_BREAKDOWN_INTERFACE
    schema: ClassVar[str] = PARTITION_OBJECTIVE_BREAKDOWN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "candidate_cid",
            "hard_results",
            "soft_scores",
            "hard_cut_count",
            "member_count",
            "hard_passed",
            "ranked",
            "advisory",
            "applied_soft_total",
            "evidence_class",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "breakdown_cid",
        }
    )

    def __post_init__(self) -> None:
        hard_results = _coerce_hard_results(self.hard_results)
        soft_scores = _coerce_soft_scores(self.soft_scores)
        ranked = _bool(self.ranked, "ranked")
        advisory = _bool(self.advisory, "advisory")
        hard_passed = all(item.passed for item in hard_results)
        evidence = _text(self.evidence_class, "evidence_class")
        if ranked and not hard_passed:
            raise PartitionPolicyError(
                "soft scores never override a hard constraint"
            )
        if ranked and advisory:
            raise PartitionPolicyError("advisory candidate cannot be ranked")
        if ranked and evidence in _NON_RANKING_EVIDENCE:
            raise PartitionPolicyError(
                "vector or model evidence cannot rank a partition candidate"
            )
        if not ranked:
            zeroed = []
            for item in soft_scores:
                if item.applied != 0:
                    raise PartitionPolicyError(
                        "soft scores never override a hard constraint"
                    )
                zeroed.append(item)
            soft_scores = tuple(zeroed)
        else:
            for item in soft_scores:
                if item.applied != item.observed:
                    raise PartitionPolicyError(
                        "ranked breakdown must apply observed soft scores"
                    )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid"))
        object.__setattr__(self, "hard_results", hard_results)
        object.__setattr__(self, "soft_scores", soft_scores)
        object.__setattr__(
            self,
            "hard_cut_count",
            _nat(self.hard_cut_count, "hard_cut_count", limit=MAX_SCORE),
        )
        object.__setattr__(
            self,
            "member_count",
            _nat(self.member_count, "member_count", limit=MAX_MEMBERS),
        )
        object.__setattr__(self, "ranked", ranked)
        object.__setattr__(self, "advisory", advisory)
        object.__setattr__(self, "evidence_class", evidence)

    @property
    def hard_passed(self) -> bool:
        return all(item.passed for item in self.hard_results)

    @property
    def applied_soft_total(self) -> int:
        return sum(item.applied for item in self.soft_scores)

    @property
    def rejected(self) -> bool:
        return not self.ranked and not self.advisory

    @property
    def ranking_key(self) -> tuple[int, int, int, str]:
        return (
            0 if self.ranked else 1 if self.rejected else 2,
            self.hard_cut_count,
            MAX_SCORE - self.applied_soft_total,
            self.candidate_cid,
        )

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PARTITION_OBJECTIVE_BREAKDOWN_SCHEMA,
            "interface": PARTITION_OBJECTIVE_BREAKDOWN_INTERFACE,
            "tree_id": self.tree_id,
            "candidate_cid": self.candidate_cid,
            "hard_results": [item.to_dict() for item in self.hard_results],
            "soft_scores": [item.to_dict() for item in self.soft_scores],
            "hard_cut_count": self.hard_cut_count,
            "member_count": self.member_count,
            "hard_passed": self.hard_passed,
            "ranked": self.ranked,
            "advisory": self.advisory,
            "applied_soft_total": self.applied_soft_total,
            "evidence_class": self.evidence_class,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def breakdown_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["breakdown_cid"] = self.breakdown_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionObjectiveBreakdown":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("breakdown_cid")
        if payload.pop("schema") != PARTITION_OBJECTIVE_BREAKDOWN_SCHEMA:
            raise PartitionPolicyError(
                "unsupported PartitionObjectiveBreakdown schema"
            )
        if payload.pop("interface") != PARTITION_OBJECTIVE_BREAKDOWN_INTERFACE:
            raise PartitionPolicyError(
                "unsupported PartitionObjectiveBreakdown interface"
            )
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
        ):
            if payload.pop(flag) is not False:
                raise PartitionPolicyError(f"breakdown cannot claim {flag}")
        payload.pop("hard_passed")
        payload.pop("applied_soft_total")
        result = cls(**payload)
        _verify_cid(
            claimed,
            result.breakdown_cid,
            "PartitionObjectiveBreakdown breakdown_cid",
        )
        return result


def _coerce_candidate(
    value: ProgramPartitionCandidate | Mapping[str, Any],
) -> ProgramPartitionCandidate:
    if isinstance(value, ProgramPartitionCandidate):
        return value
    if isinstance(value, Mapping):
        if "candidate_cid" in value:
            return ProgramPartitionCandidate.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "candidate_cid",
                "can_authorize_transition",
                "can_authorize_completion",
                "can_create_authority",
                "projection_is_authority",
            }
        }
        return ProgramPartitionCandidate(**payload)
    raise PartitionPolicyError("candidate must be a ProgramPartitionCandidate")


def _coerce_candidates(
    value: PartitionGenerationReceipt
    | Sequence[ProgramPartitionCandidate | Mapping[str, Any]]
    | Mapping[str, Any],
) -> tuple[ProgramPartitionCandidate, ...]:
    if isinstance(value, PartitionGenerationReceipt):
        return tuple(value.candidates)
    if isinstance(value, Mapping):
        if "receipt_cid" in value and "candidates" in value:
            return tuple(PartitionGenerationReceipt.from_dict(value).candidates)
        if "candidates" in value:
            value = value["candidates"]
        else:
            raise PartitionPolicyError("comparison requires candidates")
    if not isinstance(value, (list, tuple)):
        raise PartitionPolicyError("candidates must be a list")
    candidates = tuple(_coerce_candidate(item) for item in value)
    if not candidates:
        raise PartitionPolicyError("comparison requires candidates")
    if len(candidates) > MAX_CANDIDATES:
        raise PartitionPolicyError("candidates exceed maximum length")
    return candidates


def _coerce_breakdown(
    value: PartitionObjectiveBreakdown | Mapping[str, Any],
) -> PartitionObjectiveBreakdown:
    if isinstance(value, PartitionObjectiveBreakdown):
        return value
    if isinstance(value, Mapping):
        return PartitionObjectiveBreakdown.from_dict(value)
    raise PartitionPolicyError("breakdown must be a PartitionObjectiveBreakdown")


def _coerce_profile(
    value: PartitionObjectiveProfile | Mapping[str, Any] | None,
) -> PartitionObjectiveProfile:
    if value is None:
        return default_partition_objective_profile()
    if isinstance(value, PartitionObjectiveProfile):
        return value
    if isinstance(value, Mapping):
        if "profile_cid" in value:
            return PartitionObjectiveProfile.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "profile_cid",
                "can_authorize_transition",
                "can_authorize_completion",
                "can_create_authority",
                "soft_signals_cannot_override_hard_constraints",
            }
        }
        return PartitionObjectiveProfile(**payload)
    raise PartitionPolicyError("profile must be a PartitionObjectiveProfile")


@dataclass(frozen=True, slots=True)
class _OwnerGroup:
    owner_id: str
    member_ids: tuple[str, ...]
    unresolved: bool = False


@dataclass(frozen=True, slots=True)
class _PolicyFacts:
    view: Any = None
    resources: tuple[_OwnerGroup, ...] = ()
    transactions: tuple[_OwnerGroup, ...] = ()
    frontier_subjects: frozenset[str] = frozenset()
    required_frontier_edges: tuple[tuple[str, str, str], ...] = ()
    proof_obligations: tuple[tuple[str, str], ...] = ()
    lexical_scores: Mapping[str, int] = None  # type: ignore[assignment]
    vector_scores: Mapping[str, int] = None  # type: ignore[assignment]
    evidence_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.lexical_scores is None:
            object.__setattr__(self, "lexical_scores", {})
        if self.vector_scores is None:
            object.__setattr__(self, "vector_scores", {})


def _owner_groups(values: Any, name: str) -> tuple[_OwnerGroup, ...]:
    groups = []
    for item in _sequence_maps(values, name):
        uniqueness = _text(item.get("uniqueness", "unique"), "uniqueness")
        if uniqueness != "unique":
            continue
        groups.append(
            _OwnerGroup(
                owner_id=_text(item.get("owner_id"), "owner_id"),
                member_ids=_unique_sorted_text(
                    list(item.get("member_ids") or ()),
                    "member_ids",
                    limit=MAX_MEMBERS,
                ),
                unresolved=_bool(item.get("unresolved", False), "unresolved"),
            )
        )
    return tuple(groups)


def _int_map(values: Any, name: str) -> dict[str, int]:
    if values in (None, (), {}):
        return {}
    payload = _mapping(values, name)
    result: dict[str, int] = {}
    for key, item in payload.items():
        result[_text(key, name)] = _nat(item, f"{name}.{key}", limit=MAX_SCORE)
    return result


def compile_policy_facts(
    evidence: Mapping[str, Any] | None,
    *,
    tree_id: str,
) -> _PolicyFacts:
    """Normalize SPAR-013 evidence plus SPAR-014 constraint slices."""

    if evidence is None:
        return _PolicyFacts()
    payload = _mapping(evidence, "policy evidence")
    declared_tree = payload.get("tree_id")
    if declared_tree is not None and _tree_id(declared_tree) != tree_id:
        raise PartitionPolicyError("policy evidence tree_id does not match candidates")
    extra_cids: list[str] = []
    resources_payload = payload.get("resources") or payload.get("resource_ownership")
    resources_map = (
        _mapping(resources_payload, "resources") if resources_payload else {}
    )
    resource_cid = _optional_cid(
        resources_map.get("graph_cid"), "resources.graph_cid"
    )
    if resource_cid:
        extra_cids.append(resource_cid)
    resources = _owner_groups(
        resources_map.get("owners"), "resources.owners"
    )
    transaction_payload = payload.get("transactions") or payload.get(
        "transaction_ownership"
    )
    transaction_map = (
        _mapping(transaction_payload, "transactions") if transaction_payload else {}
    )
    transaction_cid = _optional_cid(
        transaction_map.get("graph_cid"), "transactions.graph_cid"
    )
    if transaction_cid:
        extra_cids.append(transaction_cid)
    transactions = _owner_groups(
        transaction_map.get("owners"), "transactions.owners"
    )
    frontier_payload = payload.get("frontier") or payload.get("dynamic_frontier")
    frontier_map = (
        _mapping(frontier_payload, "frontier") if frontier_payload else {}
    )
    frontier_cid = _optional_cid(
        frontier_map.get("frontier_cid") or frontier_map.get("graph_cid"),
        "frontier.frontier_cid",
    )
    if frontier_cid:
        extra_cids.append(frontier_cid)
    frontier_subjects = frozenset(
        _unique_sorted_text(
            list(
                frontier_map.get("unresolved_subject_ids")
                or frontier_map.get("unresolved")
                or ()
            ),
            "frontier.unresolved_subject_ids",
            limit=MAX_MEMBERS,
        )
    )
    required_edges = tuple(
        (
            _text(item.get("source_id"), "source_id"),
            _text(item.get("target_id"), "target_id"),
            _text(item.get("kind", "depends_on"), "kind"),
        )
        for item in _sequence_maps(
            frontier_map.get("required_edges"), "frontier.required_edges"
        )
    )
    proofs_payload = payload.get("proofs") or payload.get("proof_obligations")
    proofs_map = _mapping(proofs_payload, "proofs") if proofs_payload else {}
    proof_cid = _optional_cid(
        proofs_map.get("proof_cid") or proofs_map.get("graph_cid"),
        "proofs.proof_cid",
    )
    if proof_cid:
        extra_cids.append(proof_cid)
    proof_obligations = tuple(
        sorted(
            {
                (
                    _text(item.get("obligation_id"), "obligation_id"),
                    _text(item.get("subject_id"), "subject_id"),
                )
                for item in _sequence_maps(
                    proofs_map.get("obligations"), "proofs.obligations"
                )
            }
        )
    )
    view = None
    if payload.get("scc_snapshot") or payload.get("scc"):
        spar013 = {
            key: item
            for key, item in payload.items()
            if key
            not in {
                "resources",
                "resource_ownership",
                "transactions",
                "transaction_ownership",
                "frontier",
                "dynamic_frontier",
                "proofs",
                "proof_obligations",
                "lexical_scores",
                "vector_scores",
                "candidates",
                "generation_receipt",
            }
        }
        spar013["tree_id"] = tree_id
        spar013["analyzer_id"] = SPAR013_ANALYZER_ID
        view = compile_partition_evidence(spar013)
        extra_cids.append(view.scc.snapshot_cid)
        if view.state is not None:
            extra_cids.append(view.state.graph_cid)
        if view.graph_cid:
            extra_cids.append(view.graph_cid)
        if view.compatibility_cid:
            extra_cids.append(view.compatibility_cid)
        if view.initialization_cid:
            extra_cids.append(view.initialization_cid)
    evidence_cids = tuple(sorted(set(extra_cids)))
    if len(evidence_cids) > MAX_EVIDENCE_CIDS:
        raise PartitionPolicyError("evidence_cids exceed maximum length")
    return _PolicyFacts(
        view=view,
        resources=resources,
        transactions=transactions,
        frontier_subjects=frontier_subjects,
        required_frontier_edges=required_edges,
        proof_obligations=proof_obligations,
        lexical_scores=_int_map(payload.get("lexical_scores"), "lexical_scores"),
        vector_scores=_int_map(payload.get("vector_scores"), "vector_scores"),
        evidence_cids=evidence_cids,
    )


def _result(
    family: str,
    violations: Sequence[str],
    witness_ids: Sequence[str] = (),
) -> PartitionHardResult:
    unique = tuple(sorted(set(violations)))
    return PartitionHardResult(
        family=family,
        passed=not unique,
        violations=unique,
        witness_ids=tuple(sorted(set(witness_ids))),
    )


def _owner_split(
    members: Sequence[str],
    groups: Sequence[_OwnerGroup],
    *,
    split_code: str,
    unresolved_code: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inside = set(members)
    violations: list[str] = []
    witnesses: list[str] = []
    for group in groups:
        required = set(group.member_ids)
        present = required & inside
        if not present:
            continue
        if present != required:
            violations.append(split_code)
            witnesses.append(group.owner_id)
        if group.unresolved:
            violations.append(unresolved_code)
            witnesses.append(group.owner_id)
    return tuple(sorted(set(violations))), tuple(sorted(set(witnesses)))


def _evaluate_hard(
    candidate: ProgramPartitionCandidate,
    facts: _PolicyFacts,
) -> tuple[PartitionHardResult, ...]:
    existing = set(candidate.hard_constraint_violations)
    members = candidate.member_ids
    inside = set(members)
    view = facts.view

    scc_violations = sorted(existing & _SCC_VIOLATIONS)
    scc_witnesses: list[str] = []
    if view is not None:
        unknown = [item for item in members if item not in view.scc.node_to_scc]
        if unknown:
            scc_violations.append(ConstraintClass.UNKNOWN_MEMBER.value)
            scc_witnesses.extend(unknown)
        for scc_id, scc_members in view.scc.members.items():
            present = set(scc_members) & inside
            if present and present != set(scc_members) and view.scc.cyclic.get(scc_id):
                scc_violations.append(ConstraintClass.SCC_SPLIT.value)
                scc_witnesses.append(scc_id)
            if present and view.scc.oversized.get(scc_id):
                scc_violations.append(ConstraintClass.OVERSIZED_CYCLE.value)
                scc_witnesses.append(scc_id)
    scc = _result("scc", scc_violations, scc_witnesses)

    state_violations = sorted(existing & _STATE_VIOLATIONS)
    state_witnesses = list(candidate.state_owner_ids)
    if view is not None and view.state is not None:
        owners = {
            view.state.owner_of[item]
            for item in members
            if item in view.state.owner_of
        }
        for owner_id in owners:
            required = set(view.state.unique_owners[owner_id])
            if required & inside != required:
                state_violations.append(ConstraintClass.UNIQUE_OWNER_SPLIT.value)
                state_witnesses.append(owner_id)
            if owner_id in view.state.unresolved_aliases:
                state_violations.append(ConstraintClass.UNRESOLVED_ALIAS.value)
                state_witnesses.append(owner_id)
        unresolved = [item for item in members if item in view.state.unresolved_subjects]
        if unresolved:
            state_violations.append(ConstraintClass.UNRESOLVED_STATE.value)
            state_witnesses.extend(unresolved)
    state = _result("state", state_violations, state_witnesses)

    consumer_violations: list[str] = []
    consumer_witnesses: list[str] = []
    compatibility_violations: list[str] = []
    compatibility_witnesses: list[str] = []
    if view is not None:
        recorded_consumers = set(candidate.consumer_ids)
        recorded_obligations = set(candidate.obligation_ids)
        for item in view.consumers:
            consumer_id = _text(item.get("consumer_id"), "consumer_id")
            subject = item.get("subject_id") or item.get("module_name")
            if subject in inside and consumer_id not in recorded_consumers:
                consumer_violations.append("undispositioned_consumer")
                consumer_witnesses.append(consumer_id)
        for item in view.obligations:
            subject = _text(item.get("subject_id"), "subject_id")
            if subject not in inside:
                continue
            obligation_id = _text(item.get("obligation_id"), "obligation_id")
            consumer_id = item.get("consumer_id")
            if consumer_id:
                consumer_text = _text(consumer_id, "consumer_id")
                if consumer_text not in recorded_consumers:
                    consumer_violations.append("undispositioned_consumer")
                    consumer_witnesses.append(consumer_text)
            if obligation_id not in recorded_obligations:
                compatibility_violations.append("missing_compatibility_obligation")
                compatibility_witnesses.append(obligation_id)
    consumer = _result("consumer", consumer_violations, consumer_witnesses)
    compatibility = _result(
        "compatibility", compatibility_violations, compatibility_witnesses
    )

    ordering_violations: list[str] = []
    ordering_witnesses: list[str] = []
    for edge in candidate.cut_edges:
        if edge.kind in ORDERING_EDGE_KINDS:
            ordering_violations.append("ordering_cut")
            ordering_witnesses.append(edge.edge_cid)
    if view is not None:
        for source, target, kind in view.init_edges:
            if kind in ORDERING_EDGE_KINDS and (source in inside) ^ (target in inside):
                ordering_violations.append("ordering_cut")
                ordering_witnesses.extend((source, target))
    ordering = _result("ordering", ordering_violations, ordering_witnesses)

    resource_codes, resource_witnesses = _owner_split(
        members,
        facts.resources,
        split_code="resource_owner_split",
        unresolved_code="unresolved_resource",
    )
    for edge in candidate.cut_edges:
        if edge.kind in RESOURCE_EDGE_KINDS and facts.resources:
            resource_codes = tuple(
                sorted(set(resource_codes) | {"resource_cut"})
            )
            resource_witnesses = tuple(
                sorted(set(resource_witnesses) | {edge.edge_cid})
            )
    resource = _result("resource", resource_codes, resource_witnesses)

    frontier_violations: list[str] = []
    frontier_witnesses: list[str] = []
    unresolved_frontier = sorted(inside & facts.frontier_subjects)
    if unresolved_frontier:
        frontier_violations.append("unresolved_frontier")
        frontier_witnesses.extend(unresolved_frontier)
    for source, target, kind in facts.required_frontier_edges:
        if (source in inside) ^ (target in inside):
            frontier_violations.append("required_frontier_edge_cut")
            frontier_witnesses.extend((source, target, kind))
    frontier = _result("frontier", frontier_violations, frontier_witnesses)

    proof_violations: list[str] = []
    proof_witnesses: list[str] = []
    recorded_obligations = set(candidate.obligation_ids)
    for obligation_id, subject in facts.proof_obligations:
        if subject in inside and obligation_id not in recorded_obligations:
            proof_violations.append("missing_proof_obligation")
            proof_witnesses.append(obligation_id)
    for edge in candidate.cut_edges:
        if edge.kind in PROOF_EDGE_KINDS:
            proof_violations.append("proof_edge_cut")
            proof_witnesses.append(edge.edge_cid)
    proof = _result("proof", proof_violations, proof_witnesses)

    transaction_codes, transaction_witnesses = _owner_split(
        members,
        facts.transactions,
        split_code="transaction_owner_split",
        unresolved_code="unresolved_transaction",
    )
    for edge in candidate.cut_edges:
        if edge.kind in TRANSACTION_EDGE_KINDS:
            transaction_codes = tuple(
                sorted(set(transaction_codes) | {"transaction_cut"})
            )
            transaction_witnesses = tuple(
                sorted(set(transaction_witnesses) | {edge.edge_cid})
            )
    transaction = _result(
        "transaction", transaction_codes, transaction_witnesses
    )
    return (
        scc,
        state,
        consumer,
        compatibility,
        ordering,
        resource,
        frontier,
        proof,
        transaction,
    )


def _count_internal(
    candidate: ProgramPartitionCandidate,
    facts: _PolicyFacts,
    *,
    kinds: frozenset[str],
    evidence_classes: frozenset[str] | None = None,
) -> int:
    view = facts.view
    if view is None:
        return 0
    inside = set(candidate.member_ids)
    total = 0
    for source, target, kind, evidence, _confidence in view.graph_edges:
        if kind not in kinds:
            continue
        if evidence_classes is not None and evidence not in evidence_classes:
            continue
        if source in inside and target in inside:
            total += 1
    return min(total, MAX_SCORE)


def _evaluate_soft(
    candidate: ProgramPartitionCandidate,
    facts: _PolicyFacts,
    *,
    apply: bool,
) -> tuple[PartitionSoftScore, ...]:
    view = facts.view
    inside = set(candidate.member_ids)
    call = _count_internal(candidate, facts, kinds=CALL_EDGE_KINDS)
    data = _count_internal(candidate, facts, kinds=DATA_EDGE_KINDS)
    contract = min(len(candidate.obligation_ids) + len(candidate.consumer_ids), MAX_SCORE)
    test = _count_internal(candidate, facts, kinds=TEST_EDGE_KINDS)
    proof = _count_internal(candidate, facts, kinds=PROOF_EDGE_KINDS)
    cochange = 0
    if view is not None:
        cochange = min(
            sum(
                1
                for source, target, _evidence in view.cochange_edges
                if source in inside and target in inside
            ),
            MAX_SCORE,
        )
    lexical = 0
    for member in candidate.member_ids:
        lexical = min(lexical + facts.lexical_scores.get(member, 0), MAX_SCORE)
    vector = min(facts.vector_scores.get(candidate.candidate_cid, 0), MAX_SCORE)
    trace = _count_internal(
        candidate,
        facts,
        kinds=CALL_EDGE_KINDS | DATA_EDGE_KINDS | TEST_EDGE_KINDS,
        evidence_classes=TRACE_EVIDENCE_CLASSES,
    )
    observed = {
        "call": call,
        "data": data,
        "contract": contract,
        "test": test,
        "proof": proof,
        "cochange": cochange,
        "lexical": lexical,
        "vector": vector,
        "trace": trace,
    }
    return tuple(
        PartitionSoftScore(
            objective=objective,
            observed=observed[objective],
            applied=observed[objective] if apply else 0,
        )
        for objective in SOFT_OBJECTIVES
    )


def evaluate_partition_candidate(
    candidate: ProgramPartitionCandidate | Mapping[str, Any],
    *,
    profile: PartitionObjectiveProfile | Mapping[str, Any] | None = None,
    facts: _PolicyFacts | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> PartitionObjectiveBreakdown:
    """Evaluate one candidate against the complete objective profile."""

    resolved = _coerce_candidate(candidate)
    resolved_profile = _coerce_profile(profile)
    resolved_facts = facts if facts is not None else compile_policy_facts(
        evidence, tree_id=resolved.tree_id
    )
    hard_results = _evaluate_hard(resolved, resolved_facts)
    hard_passed = all(item.passed for item in hard_results)
    advisory = bool(resolved.advisory)
    non_ranking = resolved.evidence_class in _NON_RANKING_EVIDENCE
    ranked = hard_passed and not advisory and not non_ranking
    if ranked:
        vector_weight = resolved_profile.weight_for(SoftObjective.VECTOR.value)
        if (
            vector_weight
            and not resolved_profile.vector_may_reorder_admitted
            and resolved_facts.vector_scores.get(resolved.candidate_cid, 0)
        ):
            raise PartitionPolicyError(
                "vector scores cannot reorder when the profile forbids it"
            )
    soft_scores = _evaluate_soft(resolved, resolved_facts, apply=ranked)
    hard_cut_count = sum(
        1 for edge in resolved.cut_edges if edge.constraint_class == "hard"
    )
    return PartitionObjectiveBreakdown(
        tree_id=resolved.tree_id,
        candidate_cid=resolved.candidate_cid,
        hard_results=hard_results,
        soft_scores=soft_scores,
        hard_cut_count=hard_cut_count,
        member_count=len(resolved.member_ids),
        ranked=ranked,
        advisory=advisory,
        evidence_class=resolved.evidence_class,
    )


@dataclass(frozen=True, slots=True)
class PartitionComparisonReceipt:
    """Deterministic multi-objective comparison receipt. Nomination only."""

    tree_id: str
    profile: PartitionObjectiveProfile | Mapping[str, Any]
    breakdowns: Sequence[PartitionObjectiveBreakdown | Mapping[str, Any]]
    evidence_cids: Sequence[str] = ()
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = PARTITION_COMPARISON_RECEIPT_INTERFACE
    schema: ClassVar[str] = PARTITION_COMPARISON_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "profile",
            "breakdowns",
            "evidence_cids",
            "analyzer_id",
            "ranked_candidate_cids",
            "rejected_candidate_cids",
            "advisory_candidate_cids",
            "negative_evidence_cids",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "soft_signals_cannot_override_hard_constraints",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise PartitionPolicyError(
                "analyzer_id must remain the SPAR-014 analyzer"
            )
        profile = _coerce_profile(self.profile)
        breakdowns = tuple(_coerce_breakdown(item) for item in self.breakdowns)
        if not breakdowns:
            raise PartitionPolicyError("comparison requires candidates")
        if len(breakdowns) > MAX_CANDIDATES:
            raise PartitionPolicyError("breakdowns exceed maximum length")
        tree_id = _tree_id(self.tree_id)
        mismatched = [item.candidate_cid for item in breakdowns if item.tree_id != tree_id]
        if mismatched:
            raise PartitionPolicyError("breakdown tree_id does not match receipt")
        seen = [item.candidate_cid for item in breakdowns]
        if len(seen) != len(set(seen)):
            raise PartitionPolicyError("duplicate comparison candidate identity")
        for item in breakdowns:
            if item.ranked and not item.hard_passed:
                raise PartitionPolicyError(
                    "soft scores never override a hard constraint"
                )
        breakdowns = tuple(
            sorted(breakdowns, key=lambda item: item.candidate_cid)
        )
        evidence = tuple(
            sorted(_cid(item, "evidence_cids") for item in self.evidence_cids)
        )
        if len(evidence) != len(set(evidence)):
            raise PartitionPolicyError("evidence_cids must not contain duplicates")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "breakdowns", breakdowns)
        object.__setattr__(self, "evidence_cids", evidence)
        object.__setattr__(self, "analyzer_id", analyzer)

    def _weighted_soft(self, item: PartitionObjectiveBreakdown) -> int:
        return sum(
            score.applied * self.profile.weight_for(score.objective)
            for score in item.soft_scores
        )

    def _ordered(self) -> tuple[PartitionObjectiveBreakdown, ...]:
        ranked = [item for item in self.breakdowns if item.ranked]
        ranked.sort(
            key=lambda item: (
                item.hard_cut_count,
                -self._weighted_soft(item),
                item.member_count,
                item.candidate_cid,
            )
        )
        return tuple(ranked)

    @property
    def ranked_candidate_cids(self) -> tuple[str, ...]:
        return tuple(item.candidate_cid for item in self._ordered())

    @property
    def ranked_breakdowns(self) -> tuple[PartitionObjectiveBreakdown, ...]:
        return self._ordered()

    @property
    def rejected_candidate_cids(self) -> tuple[str, ...]:
        return tuple(
            item.candidate_cid for item in self.breakdowns if item.rejected
        )

    @property
    def advisory_candidate_cids(self) -> tuple[str, ...]:
        return tuple(
            item.candidate_cid for item in self.breakdowns if item.advisory
        )

    @property
    def negative_evidence_cids(self) -> tuple[str, ...]:
        return self.rejected_candidate_cids

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def soft_signals_cannot_override_hard_constraints(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PARTITION_COMPARISON_RECEIPT_SCHEMA,
            "interface": PARTITION_COMPARISON_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "profile": self.profile.to_dict(),
            "breakdowns": [item.to_dict() for item in self.breakdowns],
            "evidence_cids": list(self.evidence_cids),
            "analyzer_id": self.analyzer_id,
            "ranked_candidate_cids": list(self.ranked_candidate_cids),
            "rejected_candidate_cids": list(self.rejected_candidate_cids),
            "advisory_candidate_cids": list(self.advisory_candidate_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "soft_signals_cannot_override_hard_constraints": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartitionComparisonReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != PARTITION_COMPARISON_RECEIPT_SCHEMA:
            raise PartitionPolicyError(
                "unsupported PartitionComparisonReceipt schema"
            )
        if payload.pop("interface") != PARTITION_COMPARISON_RECEIPT_INTERFACE:
            raise PartitionPolicyError(
                "unsupported PartitionComparisonReceipt interface"
            )
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
        ):
            if payload.pop(flag) is not False:
                raise PartitionPolicyError(f"receipt cannot claim {flag}")
        if payload.pop("soft_signals_cannot_override_hard_constraints") is not True:
            raise PartitionPolicyError(
                "receipt must retain soft-cannot-override-hard"
            )
        payload.pop("ranked_candidate_cids")
        payload.pop("rejected_candidate_cids")
        payload.pop("advisory_candidate_cids")
        payload.pop("negative_evidence_cids")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "PartitionComparisonReceipt receipt_cid"
        )
        return result


def compare_partition_candidates(
    candidates: PartitionGenerationReceipt
    | Sequence[ProgramPartitionCandidate | Mapping[str, Any]]
    | Mapping[str, Any],
    *,
    profile: PartitionObjectiveProfile | Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> PartitionComparisonReceipt:
    """Compare SPAR-013 candidates under a complete objective profile.

    Soft scores only apply to candidates that pass every hard family.
    Rejected comparisons are retained as negative evidence and cannot be
    hidden by ranking, vectors, or model output.
    """

    resolved_candidates = _coerce_candidates(candidates)
    tree_ids = {item.tree_id for item in resolved_candidates}
    if len(tree_ids) != 1:
        raise PartitionPolicyError("candidates must share one tree_id")
    tree_id = next(iter(tree_ids))
    resolved_profile = _coerce_profile(profile)
    facts = compile_policy_facts(evidence, tree_id=tree_id)
    breakdowns = tuple(
        evaluate_partition_candidate(
            item, profile=resolved_profile, facts=facts
        )
        for item in resolved_candidates
    )
    return PartitionComparisonReceipt(
        tree_id=tree_id,
        profile=resolved_profile,
        breakdowns=breakdowns,
        evidence_cids=facts.evidence_cids,
        analyzer_id=ANALYZER_ID,
    )


def encode_canonical_receipt(receipt: PartitionComparisonReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(
    payload: Mapping[str, Any],
) -> PartitionComparisonReceipt:
    return PartitionComparisonReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise PartitionPolicyError(
            f"partition policy must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "DUCKLAKE_IS_AUTHORITY",
    "GOAL_ID",
    "HARD_CONSTRAINT_FAMILIES",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "ORDERING_EDGE_KINDS",
    "PARTITION_CAN_AUTHORIZE_COMPLETION",
    "PARTITION_CAN_AUTHORIZE_TRANSITION",
    "PARTITION_CAN_CREATE_AUTHORITY",
    "PARTITION_COMPARISON_RECEIPT_INTERFACE",
    "PARTITION_COMPARISON_RECEIPT_SCHEMA",
    "PARTITION_CONTRACT_VERSION",
    "PARTITION_OBJECTIVE_BREAKDOWN_INTERFACE",
    "PARTITION_OBJECTIVE_PROFILE_INTERFACE",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "SOFT_OBJECTIVES",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "HardConstraintFamily",
    "PartitionComparisonReceipt",
    "PartitionHardResult",
    "PartitionObjectiveBreakdown",
    "PartitionObjectiveProfile",
    "PartitionPolicyError",
    "PartitionSoftScore",
    "SoftObjective",
    "assert_not_competing_capsule_family",
    "compare_partition_candidates",
    "compile_policy_facts",
    "decode_canonical_receipt",
    "default_partition_objective_profile",
    "encode_canonical_receipt",
    "evaluate_partition_candidate",
    "partition_cid_profile",
    "provider_free_exports",
]
