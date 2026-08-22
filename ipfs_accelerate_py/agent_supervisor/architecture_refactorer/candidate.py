"""Canonical refactor-candidate identity (PCAR-012).

A `RefactorCandidate` binds one closed operator declaration to an exact
repository tree, candidate contract identity, expected effects, targets,
scope, migration, rollback, validation, and proofs. Identity is a dag-json
CID of that sealed payload. Candidates cannot authorize execution, promote
themselves, raise the operator autonomy ceiling, expand maximum scope, or
admit script/shell payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .contracts import (
    NodeKind,
    _require_int,
    _require_mapping,
    _require_text,
)
from .refactor_operators import (
    DEFAULT_FRESHNESS,
    EFFECT_CLASS,
    OPERATOR_CAN_ADMIT_SCRIPT_PAYLOAD,
    OPERATOR_CAN_AUTHORIZE_EXECUTION,
    OPERATOR_CAN_RAISE_AUTONOMY_CEILING,
    OPERATOR_CAN_REDUCE_GATES,
    OPERATOR_CAN_SELF_PROMOTE,
    REFACTOR_OPERATOR_EVIDENCE,
    ApiImpact,
    AuthorityImpact,
    AutonomyCeiling,
    ExpectedEffect,
    MaximumScope,
    OperatorKind,
    OperatorMigration,
    OperatorPrecondition,
    OperatorRiskClass,
    OperatorRollback,
    OperatorValidation,
    ProofObligation,
    RefactorOperator,
    RefactorOperatorAuthorityError,
    RefactorOperatorError,
    StateImpact,
    _AUTONOMY_RANK,
    _require_bool,
    _require_closed,
    _require_closed_tuple,
    _require_node_kinds,
    _require_positive_int,
    assert_repository_relative_scope_path,
    assert_scope_within_maximum,
    effects_identity,
    get_operator,
    refuse_autonomy_ceiling_raise,
    refuse_script_payload,
    refuse_self_authorization,
    refuse_self_promotion,
)

REFACTOR_CANDIDATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refactor-candidate@1"
)
REFACTOR_CANDIDATE_VERSION = 1
REFACTOR_CANDIDATE_EVIDENCE = "pcar/refactor-candidate@1"
CANDIDATE_SCOPE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/candidate-scope@1"
)
CANDIDATE_SCOPE_VERSION = 1
CANDIDATE_EXTRACTOR_IDENTITY = "pcar-012-refactor-candidate"
TASK_ID = "PCAR-012"
CANDIDATE_EFFECT_CLASS = EFFECT_CLASS
CANDIDATE_CAN_AUTHORIZE_EXECUTION = OPERATOR_CAN_AUTHORIZE_EXECUTION
CANDIDATE_CAN_SELF_PROMOTE = OPERATOR_CAN_SELF_PROMOTE
CANDIDATE_CAN_RAISE_AUTONOMY_CEILING = OPERATOR_CAN_RAISE_AUTONOMY_CEILING
CANDIDATE_CAN_REDUCE_GATES = OPERATOR_CAN_REDUCE_GATES
CANDIDATE_CAN_ADMIT_SCRIPT_PAYLOAD = OPERATOR_CAN_ADMIT_SCRIPT_PAYLOAD

_UNKNOWN_FIELD_MESSAGE = "unknown refactor-candidate field"
_MISSING_FIELD_MESSAGE = "missing refactor-candidate field"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")

_SCOPE_FIELDS = frozenset(
    {
        "content_identity",
        "file_count",
        "node_ids",
        "package_count",
        "paths",
        "schema",
        "symbol_count",
        "target_kinds",
        "version",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "api_impact",
        "authority_impact",
        "autonomy_ceiling",
        "can_admit_script_payload",
        "can_authorize_execution",
        "can_raise_autonomy_ceiling",
        "can_self_promote",
        "content_identity",
        "contract_identity",
        "effects_identity",
        "expected_effects",
        "freshness",
        "migration",
        "operator_identity",
        "operator_kind",
        "preconditions",
        "proofs",
        "repository_tree",
        "risk_class",
        "rollback",
        "schema",
        "scope",
        "state_impact",
        "targets",
        "validation",
        "version",
    }
)


class RefactorCandidateError(RefactorOperatorError):
    """Fail-closed refactor-candidate contract violation."""


class RefactorCandidateAuthorityError(
    RefactorCandidateError, RefactorOperatorAuthorityError
):
    """Raised when a candidate is asked to authorize, promote, or raise itself."""


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str) -> str:
    try:
        return validate_cid(value, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise RefactorCandidateError(
            "content identity must be a dag-json CIDv1"
        ) from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise RefactorCandidateError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise RefactorCandidateError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_cid(value: Any, name: str) -> str:
    text = _require_text(value, name, error_type=RefactorCandidateError)
    if not text.startswith(_CID_PREFIXES):
        raise RefactorCandidateError(f"{name} must be a dag-json CIDv1")
    return _validate_dag_json_cid(text)


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RefactorCandidateError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=RefactorCandidateError)
        for item in value
    )
    if len(items) != len(set(items)):
        raise RefactorCandidateError(f"{name} must be unique")
    return tuple(sorted(items))


def _bind_identity(record: Any, claimed: str, name: str) -> str:
    identity = _content_identity(record._identity_payload())
    if claimed:
        observed = _validate_dag_json_cid(
            _require_text(claimed, "content_identity", error_type=RefactorCandidateError)
        )
        if observed != identity:
            raise RefactorCandidateError(f"{name} content identity mismatch")
    return identity


def _record_or_mapping(value: Any, record_type: type[Any], name: str) -> Any:
    if isinstance(value, record_type):
        return value
    if isinstance(value, Mapping):
        return record_type.from_mapping(value)
    raise RefactorCandidateError(f"{name} must be an object")


@dataclass(frozen=True)
class CandidateScope:
    """Exact candidate paths and counts, bounded by the operator maximum."""

    paths: tuple[str, ...]
    node_ids: tuple[str, ...]
    file_count: int
    symbol_count: int
    package_count: int
    target_kinds: tuple[NodeKind, ...]
    schema: str = CANDIDATE_SCOPE_SCHEMA
    version: int = CANDIDATE_SCOPE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=RefactorCandidateError)
        if schema != CANDIDATE_SCOPE_SCHEMA:
            raise RefactorCandidateError("unexpected candidate-scope schema")
        version = _require_int(self.version, "version", error_type=RefactorCandidateError)
        if version != CANDIDATE_SCOPE_VERSION:
            raise RefactorCandidateError("unexpected candidate-scope version")
        if isinstance(self.paths, (str, bytes, bytearray)) or not isinstance(
            self.paths, Sequence
        ):
            raise RefactorCandidateError("paths must be a list of strings")
        normalized = tuple(assert_repository_relative_scope_path(path) for path in self.paths)
        unique_paths = tuple(sorted(set(normalized)))
        if len(unique_paths) != len(normalized):
            raise RefactorCandidateError("scope paths must be unique")
        if not unique_paths:
            raise RefactorCandidateError("incomplete refactor-candidate declaration: paths")
        node_ids = _require_text_tuple(self.node_ids, "node_ids")
        if not node_ids:
            raise RefactorCandidateError("incomplete refactor-candidate declaration: node_ids")
        target_kinds = _require_node_kinds(self.target_kinds, "target_kinds")
        symbol_count = _require_positive_int(self.symbol_count, "symbol_count")
        package_count = _require_positive_int(self.package_count, "package_count")
        file_count = _require_int(self.file_count, "file_count", error_type=RefactorCandidateError)
        if file_count != len(unique_paths):
            raise RefactorCandidateError("file_count must equal the unique path count")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "target_kinds", target_kinds)
        object.__setattr__(self, "symbol_count", symbol_count)
        object.__setattr__(self, "package_count", package_count)
        object.__setattr__(self, "file_count", file_count)
        object.__setattr__(self, "paths", unique_paths)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "candidate-scope"),
        )

    def bind_to_operator(self, maximum: MaximumScope, allowed_kinds: Sequence[NodeKind]) -> "CandidateScope":
        """Normalize paths and reject expansion beyond the operator maximum."""

        allowed = frozenset(allowed_kinds)
        extra_kinds = [item.value for item in self.target_kinds if item not in allowed]
        if extra_kinds:
            raise RefactorCandidateError(
                f"candidate target kinds exceed operator targets: {extra_kinds}"
            )
        paths = assert_scope_within_maximum(
            paths=self.paths,
            symbol_count=self.symbol_count,
            package_count=self.package_count,
            maximum=maximum,
        )
        return CandidateScope(
            paths=paths,
            node_ids=self.node_ids,
            file_count=len(paths),
            symbol_count=self.symbol_count,
            package_count=self.package_count,
            target_kinds=self.target_kinds,
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "file_count": self.file_count,
            "node_ids": list(self.node_ids),
            "package_count": self.package_count,
            "paths": list(self.paths),
            "schema": self.schema,
            "symbol_count": self.symbol_count,
            "target_kinds": [item.value for item in self.target_kinds],
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise RefactorCandidateError("candidate-scope content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CandidateScope":
        mapping = _require_mapping(payload, error_type=RefactorCandidateError)
        _require_fields(mapping, _SCOPE_FIELDS)
        record = cls(
            paths=mapping["paths"],
            node_ids=mapping["node_ids"],
            file_count=mapping["file_count"],
            symbol_count=mapping["symbol_count"],
            package_count=mapping["package_count"],
            target_kinds=mapping["target_kinds"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorCandidateError("candidate-scope content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class RefactorCandidate:
    """Bounded declarative candidate bound to exact tree, contract, and effects."""

    operator_kind: OperatorKind
    operator_identity: str
    repository_tree: str
    contract_identity: str
    expected_effects: tuple[ExpectedEffect, ...]
    targets: tuple[str, ...]
    scope: CandidateScope
    preconditions: tuple[OperatorPrecondition, ...]
    authority_impact: AuthorityImpact
    api_impact: ApiImpact
    state_impact: StateImpact
    migration: OperatorMigration
    rollback: OperatorRollback
    validation: OperatorValidation
    proofs: tuple[ProofObligation, ...]
    risk_class: OperatorRiskClass
    autonomy_ceiling: AutonomyCeiling
    effects_identity: str = ""
    freshness: str = DEFAULT_FRESHNESS
    can_authorize_execution: bool = False
    can_self_promote: bool = False
    can_raise_autonomy_ceiling: bool = False
    can_admit_script_payload: bool = False
    schema: str = REFACTOR_CANDIDATE_SCHEMA
    version: int = REFACTOR_CANDIDATE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=RefactorCandidateError)
        if schema != REFACTOR_CANDIDATE_SCHEMA:
            raise RefactorCandidateError("unexpected refactor-candidate schema")
        version = _require_int(self.version, "version", error_type=RefactorCandidateError)
        if version != REFACTOR_CANDIDATE_VERSION:
            raise RefactorCandidateError("unexpected refactor-candidate version")
        kind = _require_closed(self.operator_kind, OperatorKind, "operator kind")
        operator = get_operator(kind)
        operator_identity = _require_cid(self.operator_identity, "operator_identity")
        if operator_identity != operator.content_identity:
            raise RefactorCandidateError("operator identity does not match the sealed catalog")
        if _require_bool(self.can_authorize_execution, "can_authorize_execution"):
            refuse_self_authorization("authorize")
        if _require_bool(self.can_self_promote, "can_self_promote"):
            refuse_self_promotion("promote")
        if _require_bool(self.can_raise_autonomy_ceiling, "can_raise_autonomy_ceiling"):
            refuse_autonomy_ceiling_raise("raise")
        if _require_bool(self.can_admit_script_payload, "can_admit_script_payload"):
            refuse_script_payload("admit")
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=RefactorCandidateError
        )
        contract_identity = _require_cid(self.contract_identity, "contract_identity")
        expected_effects = _require_closed_tuple(
            self.expected_effects, ExpectedEffect, "expected effect"
        )
        if expected_effects != operator.expected_effects:
            raise RefactorCandidateError(
                "candidate effects must match the operator declaration"
            )
        derived_effects = effects_identity(expected_effects)
        claimed_effects = self.effects_identity or derived_effects
        claimed_effects = _require_cid(claimed_effects, "effects_identity")
        if claimed_effects != derived_effects:
            raise RefactorCandidateError("effects identity mismatch")
        targets = _require_text_tuple(self.targets, "targets")
        if not targets:
            raise RefactorCandidateError("incomplete refactor-candidate declaration: targets")
        scope = _record_or_mapping(self.scope, CandidateScope, "scope")
        bound_scope = scope.bind_to_operator(operator.maximum_scope, operator.target_kinds)
        preconditions = _require_closed_tuple(
            self.preconditions, OperatorPrecondition, "precondition"
        )
        if preconditions != operator.preconditions:
            raise RefactorCandidateError(
                "candidate preconditions must match the operator declaration"
            )
        authority_impact = _require_closed(
            self.authority_impact, AuthorityImpact, "authority impact"
        )
        api_impact = _require_closed(self.api_impact, ApiImpact, "api impact")
        state_impact = _require_closed(self.state_impact, StateImpact, "state impact")
        if (
            authority_impact is not operator.authority_impact
            or api_impact is not operator.api_impact
            or state_impact is not operator.state_impact
        ):
            raise RefactorCandidateError(
                "candidate authority/API/state impact must match the operator"
            )
        migration = _record_or_mapping(self.migration, OperatorMigration, "migration")
        if migration.content_identity != operator.migration.content_identity:
            raise RefactorCandidateError("candidate migration must match the operator")
        rollback = _record_or_mapping(self.rollback, OperatorRollback, "rollback")
        if rollback.content_identity != operator.rollback.content_identity:
            raise RefactorCandidateError("candidate rollback must match the operator")
        validation = _record_or_mapping(self.validation, OperatorValidation, "validation")
        if validation.content_identity != operator.validation.content_identity:
            raise RefactorCandidateError("candidate validation must match the operator")
        proofs = _require_closed_tuple(self.proofs, ProofObligation, "proof obligation")
        if proofs != operator.proofs:
            raise RefactorCandidateError("candidate proofs must match the operator declaration")
        risk_class = _require_closed(self.risk_class, OperatorRiskClass, "risk class")
        if risk_class is not operator.risk_class:
            raise RefactorCandidateError("candidate risk class must match the operator")
        ceiling = _require_closed(
            self.autonomy_ceiling, AutonomyCeiling, "autonomy ceiling"
        )
        if _AUTONOMY_RANK[ceiling] > _AUTONOMY_RANK[operator.autonomy_ceiling]:
            refuse_autonomy_ceiling_raise("raise")
        if ceiling != operator.autonomy_ceiling:
            raise RefactorCandidateError(
                "candidate autonomy ceiling must match the operator declaration"
            )
        freshness = _require_text(self.freshness, "freshness", error_type=RefactorCandidateError)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "operator_kind", kind)
        object.__setattr__(self, "operator_identity", operator_identity)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "contract_identity", contract_identity)
        object.__setattr__(self, "expected_effects", expected_effects)
        object.__setattr__(self, "effects_identity", derived_effects)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "scope", bound_scope)
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "authority_impact", authority_impact)
        object.__setattr__(self, "api_impact", api_impact)
        object.__setattr__(self, "state_impact", state_impact)
        object.__setattr__(self, "migration", migration)
        object.__setattr__(self, "rollback", rollback)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "proofs", proofs)
        object.__setattr__(self, "risk_class", risk_class)
        object.__setattr__(self, "autonomy_ceiling", ceiling)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "can_authorize_execution", False)
        object.__setattr__(self, "can_self_promote", False)
        object.__setattr__(self, "can_raise_autonomy_ceiling", False)
        object.__setattr__(self, "can_admit_script_payload", False)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "refactor-candidate"),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "api_impact": self.api_impact.value,
            "authority_impact": self.authority_impact.value,
            "autonomy_ceiling": self.autonomy_ceiling.value,
            "can_admit_script_payload": False,
            "can_authorize_execution": False,
            "can_raise_autonomy_ceiling": False,
            "can_self_promote": False,
            "contract_identity": self.contract_identity,
            "effects_identity": self.effects_identity,
            "expected_effects": [item.value for item in self.expected_effects],
            "freshness": self.freshness,
            "migration": self.migration.to_dict(),
            "operator_identity": self.operator_identity,
            "operator_kind": self.operator_kind.value,
            "preconditions": [item.value for item in self.preconditions],
            "proofs": [item.value for item in self.proofs],
            "repository_tree": self.repository_tree,
            "risk_class": self.risk_class.value,
            "rollback": self.rollback.to_dict(),
            "schema": self.schema,
            "scope": self.scope.to_dict(),
            "state_impact": self.state_impact.value,
            "targets": list(self.targets),
            "validation": self.validation.to_dict(),
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise RefactorCandidateError("refactor-candidate content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RefactorCandidate":
        mapping = _require_mapping(payload, error_type=RefactorCandidateError)
        _require_fields(mapping, _CANDIDATE_FIELDS)
        record = cls(
            operator_kind=mapping["operator_kind"],
            operator_identity=mapping["operator_identity"],
            repository_tree=mapping["repository_tree"],
            contract_identity=mapping["contract_identity"],
            expected_effects=mapping["expected_effects"],
            targets=mapping["targets"],
            scope=mapping["scope"],
            preconditions=mapping["preconditions"],
            authority_impact=mapping["authority_impact"],
            api_impact=mapping["api_impact"],
            state_impact=mapping["state_impact"],
            migration=mapping["migration"],
            rollback=mapping["rollback"],
            validation=mapping["validation"],
            proofs=mapping["proofs"],
            risk_class=mapping["risk_class"],
            autonomy_ceiling=mapping["autonomy_ceiling"],
            effects_identity=mapping["effects_identity"],
            freshness=mapping["freshness"],
            can_authorize_execution=mapping["can_authorize_execution"],
            can_self_promote=mapping["can_self_promote"],
            can_raise_autonomy_ceiling=mapping["can_raise_autonomy_ceiling"],
            can_admit_script_payload=mapping["can_admit_script_payload"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorCandidateError("refactor-candidate content identity mismatch")
        return record

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "RefactorCandidate":
        if type(payload) is not str or not payload:
            raise RefactorCandidateError(
                "refactor-candidate JSON must be a nonempty string"
            )
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RefactorCandidateError("refactor-candidate JSON is invalid") from exc
        return cls.from_mapping(decoded)

    def apply(self) -> None:
        refuse_self_authorization("apply")

    def authorize(self) -> None:
        refuse_self_authorization("authorize")

    def promote(self) -> None:
        refuse_self_promotion("promote")

    def raise_autonomy_ceiling(self) -> None:
        refuse_autonomy_ceiling_raise("raise")

    def admit_script(self) -> None:
        refuse_script_payload("admit")


def declare_refactor_candidate(
    *,
    operator_kind: OperatorKind | str,
    repository_tree: str,
    contract_identity: str,
    targets: Sequence[str],
    paths: Sequence[str],
    node_ids: Sequence[str],
    symbol_count: int,
    package_count: int,
    target_kinds: Sequence[NodeKind | str],
    freshness: str = DEFAULT_FRESHNESS,
    operator: RefactorOperator | None = None,
) -> RefactorCandidate:
    """Build a complete candidate from the sealed operator catalog."""

    catalog_operator = operator if operator is not None else get_operator(operator_kind)
    closed_kind = _require_closed(operator_kind, OperatorKind, "operator kind")
    if catalog_operator.kind is not closed_kind:
        raise RefactorCandidateError("operator kind does not match the provided declaration")
    scope = CandidateScope(
        paths=tuple(paths),
        node_ids=tuple(node_ids),
        file_count=len(tuple(paths)),
        symbol_count=symbol_count,
        package_count=package_count,
        target_kinds=tuple(target_kinds),
    )
    return RefactorCandidate(
        operator_kind=catalog_operator.kind,
        operator_identity=catalog_operator.content_identity,
        repository_tree=repository_tree,
        contract_identity=contract_identity,
        expected_effects=catalog_operator.expected_effects,
        targets=tuple(targets),
        scope=scope,
        preconditions=catalog_operator.preconditions,
        authority_impact=catalog_operator.authority_impact,
        api_impact=catalog_operator.api_impact,
        state_impact=catalog_operator.state_impact,
        migration=catalog_operator.migration,
        rollback=catalog_operator.rollback,
        validation=catalog_operator.validation,
        proofs=catalog_operator.proofs,
        risk_class=catalog_operator.risk_class,
        autonomy_ceiling=catalog_operator.autonomy_ceiling,
        freshness=freshness,
    )


def canonical_candidate_identity(candidate: RefactorCandidate | Mapping[str, Any]) -> str:
    """Return the sealed dag-json identity of one candidate."""

    record = (
        candidate
        if isinstance(candidate, RefactorCandidate)
        else RefactorCandidate.from_mapping(candidate)
    )
    return record.content_identity


__all__ = [
    "CANDIDATE_CAN_ADMIT_SCRIPT_PAYLOAD",
    "CANDIDATE_CAN_AUTHORIZE_EXECUTION",
    "CANDIDATE_CAN_RAISE_AUTONOMY_CEILING",
    "CANDIDATE_CAN_REDUCE_GATES",
    "CANDIDATE_CAN_SELF_PROMOTE",
    "CANDIDATE_EFFECT_CLASS",
    "CANDIDATE_EXTRACTOR_IDENTITY",
    "CANDIDATE_SCOPE_SCHEMA",
    "CANDIDATE_SCOPE_VERSION",
    "REFACTOR_CANDIDATE_EVIDENCE",
    "REFACTOR_CANDIDATE_SCHEMA",
    "REFACTOR_CANDIDATE_VERSION",
    "REFACTOR_OPERATOR_EVIDENCE",
    "TASK_ID",
    "CandidateScope",
    "RefactorCandidate",
    "RefactorCandidateAuthorityError",
    "RefactorCandidateError",
    "canonical_candidate_identity",
    "declare_refactor_candidate",
]
