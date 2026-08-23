"""Canonical refactor-candidate identity (PCAR-012).

`RefactorCandidate` binds one closed operator declaration to an exact
repository tree, contract identity, expected effects, targets, and a
maximum scope that cannot expand the operator bound. Candidates remain
non-authoritative: they cannot execute, authorize, reduce gates, raise
the autonomy ceiling, or promote themselves.
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
    ArchitectureContractError,
    NodeKind,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
    _repository_relative_path,
)
from .refactor_operators import (
    REFACTOR_OPERATOR_SCHEMA,
    AutonomyDisposition,
    AutonomyRiskClass,
    AuthorityImpact,
    ExpectedEffectKind,
    MaximumScope,
    OperatorKind,
    OperatorMigration,
    OperatorRollback,
    PreconditionKind,
    ProofObligationKind,
    PublicApiImpact,
    RefactorOperator,
    RefactorOperatorAuthorityError,
    RefactorOperatorError,
    StateImpact,
    ValidationObligationKind,
    assert_within_maximum_scope,
    autonomy_rank,
    operator_for,
    refuse_ceiling_raise,
    refuse_execution,
    refuse_gate_reduction,
    refuse_scope_expansion,
    refuse_self_authorization,
    refuse_self_promotion,
    refuse_unknown_operator,
)

REFACTOR_CANDIDATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refactor-candidate@1"
)
REFACTOR_CANDIDATE_VERSION = 1
REFACTOR_CANDIDATE_EVIDENCE = "pcar/refactor-candidate@1"
CANDIDATE_CAN_AUTHORIZE_EXECUTION = False
CANDIDATE_CAN_REDUCE_GATES = False
CANDIDATE_CAN_SELF_PROMOTE = False
CANDIDATE_CAN_RAISE_CEILING = False
CANDIDATE_CAN_EXPAND_SCOPE = False
CANDIDATE_IDENTITY_BINDS_TREE = True
CANDIDATE_IDENTITY_BINDS_CONTRACT = True
CANDIDATE_IDENTITY_BINDS_EFFECTS = True

_UNKNOWN_FIELD_MESSAGE = "unknown refactor-candidate field"
_MISSING_FIELD_MESSAGE = "missing refactor-candidate field"
_FORBIDDEN_PAYLOAD_FIELDS = frozenset(
    {
        "script",
        "shell",
        "payload",
        "command",
        "executable",
        "eval",
        "exec",
        "subprocess",
        "arbitrary",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "authority_impact",
        "autonomy_disposition",
        "can_authorize_execution",
        "can_raise_ceiling",
        "can_reduce_gates",
        "can_self_promote",
        "content_identity",
        "contract_identity",
        "expected_effects",
        "freshness",
        "maximum_scope",
        "migration",
        "operator_identity",
        "operator_kind",
        "preconditions",
        "proofs",
        "public_api_impact",
        "repository_tree",
        "risk_class",
        "rollback",
        "schema",
        "state_impact",
        "target_kinds",
        "target_node_ids",
        "target_paths",
        "validation",
        "version",
    }
)


class RefactorCandidateError(RefactorOperatorError):
    """Fail-closed refactor-candidate identity error."""


class RefactorCandidateAuthorityError(RefactorOperatorAuthorityError, RefactorCandidateError):
    """Raised when a candidate is asked to authorize, promote, or expand."""


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str, name: str) -> str:
    text = _require_text(value, name, error_type=RefactorCandidateError)
    try:
        return validate_cid(text, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise RefactorCandidateError(f"{name} must be a dag-json CIDv1") from exc


def _reject_unknown_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = set(payload) - set(allowed)
    forbidden = sorted(extra & _FORBIDDEN_PAYLOAD_FIELDS)
    if forbidden:
        raise RefactorCandidateAuthorityError(
            f"refactor candidate cannot admit arbitrary {forbidden[0]} payloads"
        )
    leftover = sorted(extra)
    if leftover:
        raise RefactorCandidateError(f"{_UNKNOWN_FIELD_MESSAGE}: {leftover}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown_fields(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise RefactorCandidateError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RefactorCandidateError(f"{name} must be a boolean")
    return value


def _require_enum_tuple(value: Any, enum_type: type[Any], name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RefactorCandidateError(f"{name} must be a list of strings")
    items = tuple(
        _closed_enum(item, enum_type, name, error_type=RefactorCandidateError)
        for item in value
    )
    if len(items) != len(set(items)):
        raise RefactorCandidateError(f"{name} must be unique")
    if not items:
        raise RefactorCandidateError(f"{name} must be nonempty")
    return tuple(sorted(items, key=lambda item: item.value))


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RefactorCandidateError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=RefactorCandidateError)
        for item in value
    )
    if len(items) != len(set(items)):
        raise RefactorCandidateError(f"{name} must be unique")
    if not items:
        raise RefactorCandidateError(f"{name} must be nonempty")
    return tuple(sorted(set(items)))


def _require_false_authority_flag(value: Any, name: str, refuse) -> bool:
    flag = _require_bool(value, name)
    if flag:
        refuse()
    return False


def _subset_enum(child: tuple[Any, ...], parent: tuple[Any, ...], name: str) -> None:
    extra = [item.value for item in child if item not in parent]
    if extra:
        raise RefactorCandidateError(f"candidate {name} expands operator {name}: {extra}")


def _require_matching_or_narrower_impact(
    candidate_value: Any,
    operator_value: Any,
    *,
    allowed_narrowing: set[tuple[Any, Any]],
    name: str,
) -> None:
    if candidate_value is operator_value:
        return
    if (operator_value, candidate_value) in allowed_narrowing:
        return
    raise RefactorCandidateError(
        f"candidate {name} cannot expand operator {name}"
    )


_AUTHORITY_NARROWING = {
    (AuthorityImpact.CONSOLIDATE, AuthorityImpact.PRESERVE),
    (AuthorityImpact.CONSOLIDATE, AuthorityImpact.ADAPTER),
    (AuthorityImpact.CONSOLIDATE, AuthorityImpact.QUARANTINE),
    (AuthorityImpact.CONSOLIDATE, AuthorityImpact.NONE),
    (AuthorityImpact.ADAPTER, AuthorityImpact.PRESERVE),
    (AuthorityImpact.ADAPTER, AuthorityImpact.NONE),
    (AuthorityImpact.QUARANTINE, AuthorityImpact.PRESERVE),
    (AuthorityImpact.QUARANTINE, AuthorityImpact.NONE),
    (AuthorityImpact.PRESERVE, AuthorityImpact.NONE),
}
_PUBLIC_NARROWING = {
    (PublicApiImpact.REMOVE_AFTER_GATE, PublicApiImpact.DEPRECATE),
    (PublicApiImpact.REMOVE_AFTER_GATE, PublicApiImpact.COMPATIBILITY),
    (PublicApiImpact.REMOVE_AFTER_GATE, PublicApiImpact.INTERNAL),
    (PublicApiImpact.DEPRECATE, PublicApiImpact.COMPATIBILITY),
    (PublicApiImpact.DEPRECATE, PublicApiImpact.INTERNAL),
    (PublicApiImpact.VERSIONED_MIGRATION, PublicApiImpact.INTERNAL),
    (PublicApiImpact.COMPATIBILITY, PublicApiImpact.INTERNAL),
}
_STATE_NARROWING = {
    (StateImpact.MOVE_TO_OWNER, StateImpact.NO_INDEFINITE_DUAL_AUTHORITY),
    (StateImpact.MOVE_TO_OWNER, StateImpact.READ_ONLY),
    (StateImpact.MOVE_TO_OWNER, StateImpact.NONE),
    (StateImpact.NO_INDEFINITE_DUAL_AUTHORITY, StateImpact.READ_ONLY),
    (StateImpact.NO_INDEFINITE_DUAL_AUTHORITY, StateImpact.NONE),
    (StateImpact.READ_ONLY, StateImpact.NONE),
}


@dataclass(frozen=True)
class RefactorCandidate:
    """Identity-bound candidate for one closed operator on one sealed tree."""

    operator_kind: OperatorKind
    operator_identity: str
    repository_tree: str
    contract_identity: str
    expected_effects: tuple[ExpectedEffectKind, ...]
    target_kinds: tuple[NodeKind, ...]
    target_node_ids: tuple[str, ...]
    target_paths: tuple[str, ...]
    authority_impact: AuthorityImpact
    public_api_impact: PublicApiImpact
    state_impact: StateImpact
    autonomy_disposition: AutonomyDisposition
    risk_class: AutonomyRiskClass
    preconditions: tuple[PreconditionKind, ...]
    validation: tuple[ValidationObligationKind, ...]
    proofs: tuple[ProofObligationKind, ...]
    migration: OperatorMigration
    rollback: OperatorRollback
    maximum_scope: MaximumScope
    freshness: str
    can_authorize_execution: bool = False
    can_reduce_gates: bool = False
    can_self_promote: bool = False
    can_raise_ceiling: bool = False
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
        try:
            kind = _closed_enum(
                self.operator_kind,
                OperatorKind,
                "operator kind",
                error_type=RefactorCandidateError,
            )
        except ArchitectureContractError as exc:
            refuse_unknown_operator(self.operator_kind)
            raise RefactorCandidateError(str(exc)) from exc
        operator = operator_for(kind)
        operator_identity = _validate_dag_json_cid(
            self.operator_identity, "operator_identity"
        )
        if operator_identity != operator.content_identity:
            raise RefactorCandidateError("candidate operator identity mismatch")
        tree = _require_text(
            self.repository_tree, "repository_tree", error_type=RefactorCandidateError
        )
        contract_identity = _validate_dag_json_cid(
            self.contract_identity, "contract_identity"
        )
        effects = _require_enum_tuple(
            self.expected_effects, ExpectedEffectKind, "expected_effects"
        )
        _subset_enum(effects, operator.expected_effects, "expected_effects")
        target_kinds = _require_enum_tuple(self.target_kinds, NodeKind, "target_kinds")
        _subset_enum(target_kinds, operator.target_kinds, "target_kinds")
        node_ids = _require_text_tuple(self.target_node_ids, "target_node_ids")
        paths = tuple(
            _repository_relative_path(
                item, "target_paths item", error_type=RefactorCandidateError
            )
            for item in _require_text_tuple(self.target_paths, "target_paths")
        )
        authority = _closed_enum(
            self.authority_impact,
            AuthorityImpact,
            "authority impact",
            error_type=RefactorCandidateError,
        )
        public_api = _closed_enum(
            self.public_api_impact,
            PublicApiImpact,
            "public API impact",
            error_type=RefactorCandidateError,
        )
        state = _closed_enum(
            self.state_impact,
            StateImpact,
            "state impact",
            error_type=RefactorCandidateError,
        )
        _require_matching_or_narrower_impact(
            authority,
            operator.authority_impact,
            allowed_narrowing=_AUTHORITY_NARROWING,
            name="authority_impact",
        )
        _require_matching_or_narrower_impact(
            public_api,
            operator.public_api_impact,
            allowed_narrowing=_PUBLIC_NARROWING,
            name="public_api_impact",
        )
        _require_matching_or_narrower_impact(
            state,
            operator.state_impact,
            allowed_narrowing=_STATE_NARROWING,
            name="state_impact",
        )
        disposition = _closed_enum(
            self.autonomy_disposition,
            AutonomyDisposition,
            "autonomy disposition",
            error_type=RefactorCandidateError,
        )
        risk = _closed_enum(
            self.risk_class,
            AutonomyRiskClass,
            "autonomy risk class",
            error_type=RefactorCandidateError,
        )
        if autonomy_rank(disposition) < autonomy_rank(operator.autonomy_disposition):
            refuse_ceiling_raise("raise")
        if (
            disposition is operator.autonomy_disposition
            and risk is not operator.risk_class
        ):
            raise RefactorCandidateError(
                "candidate risk class must match the operator when disposition is unchanged"
            )
        preconditions = _require_enum_tuple(
            self.preconditions, PreconditionKind, "preconditions"
        )
        if set(operator.preconditions) - set(preconditions):
            raise RefactorCandidateError(
                "candidate preconditions must include every operator precondition"
            )
        validation = _require_enum_tuple(
            self.validation, ValidationObligationKind, "validation"
        )
        if set(operator.validation) - set(validation):
            refuse_gate_reduction("drop")
        proofs = _require_enum_tuple(self.proofs, ProofObligationKind, "proofs")
        if set(operator.proofs) - set(proofs):
            refuse_gate_reduction("drop")
        migration = (
            self.migration
            if isinstance(self.migration, OperatorMigration)
            else OperatorMigration.from_mapping(self.migration)
        )
        rollback = (
            self.rollback
            if isinstance(self.rollback, OperatorRollback)
            else OperatorRollback.from_mapping(self.rollback)
        )
        scope = (
            self.maximum_scope
            if isinstance(self.maximum_scope, MaximumScope)
            else MaximumScope.from_mapping(self.maximum_scope)
        )
        try:
            assert_within_maximum_scope(operator.maximum_scope, scope)
        except RefactorOperatorAuthorityError as exc:
            raise RefactorCandidateAuthorityError(str(exc)) from exc
        if set(scope.target_kinds) != set(target_kinds):
            raise RefactorCandidateError(
                "candidate maximum-scope target kinds must match candidate target kinds"
            )
        if len(paths) > scope.max_paths or len(node_ids) > scope.max_symbols:
            raise RefactorCandidateAuthorityError(
                "refactor operator cannot expand maximum scope"
            )
        for path in paths:
            try:
                covered = scope.covers_path(path) and operator.maximum_scope.covers_path(
                    path
                )
            except RefactorOperatorError as exc:
                raise RefactorCandidateAuthorityError(str(exc)) from exc
            if not covered:
                raise RefactorCandidateAuthorityError(
                    "refactor operator cannot expand maximum scope"
                )
        freshness = _require_text(self.freshness, "freshness", error_type=RefactorCandidateError)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "operator_kind", kind)
        object.__setattr__(self, "operator_identity", operator_identity)
        object.__setattr__(self, "repository_tree", tree)
        object.__setattr__(self, "contract_identity", contract_identity)
        object.__setattr__(self, "expected_effects", effects)
        object.__setattr__(self, "target_kinds", target_kinds)
        object.__setattr__(self, "target_node_ids", node_ids)
        object.__setattr__(self, "target_paths", paths)
        object.__setattr__(self, "authority_impact", authority)
        object.__setattr__(self, "public_api_impact", public_api)
        object.__setattr__(self, "state_impact", state)
        object.__setattr__(self, "autonomy_disposition", disposition)
        object.__setattr__(self, "risk_class", risk)
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "proofs", proofs)
        object.__setattr__(self, "migration", migration)
        object.__setattr__(self, "rollback", rollback)
        object.__setattr__(self, "maximum_scope", scope)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(
            self,
            "can_authorize_execution",
            _require_false_authority_flag(
                self.can_authorize_execution,
                "can_authorize_execution",
                refuse_self_authorization,
            ),
        )
        object.__setattr__(
            self,
            "can_reduce_gates",
            _require_false_authority_flag(
                self.can_reduce_gates, "can_reduce_gates", refuse_gate_reduction
            ),
        )
        object.__setattr__(
            self,
            "can_self_promote",
            _require_false_authority_flag(
                self.can_self_promote, "can_self_promote", refuse_self_promotion
            ),
        )
        object.__setattr__(
            self,
            "can_raise_ceiling",
            _require_false_authority_flag(
                self.can_raise_ceiling, "can_raise_ceiling", refuse_ceiling_raise
            ),
        )
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(self.content_identity, "content_identity")
            if claimed != identity:
                raise RefactorCandidateError("refactor-candidate content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def operator(self) -> RefactorOperator:
        return operator_for(self.operator_kind)

    def apply(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_execution("apply")

    def execute(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_execution("execute")

    def authorize(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_self_authorization("authorize")

    def promote(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_self_promotion("promote")

    def raise_ceiling(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_ceiling_raise("raise")

    def reduce_gates(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_gate_reduction("reduce")

    def expand_scope(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_scope_expansion("expand")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "authority_impact": self.authority_impact.value,
            "autonomy_disposition": self.autonomy_disposition.value,
            "can_authorize_execution": False,
            "can_raise_ceiling": False,
            "can_reduce_gates": False,
            "can_self_promote": False,
            "contract_identity": self.contract_identity,
            "expected_effects": [item.value for item in self.expected_effects],
            "freshness": self.freshness,
            "maximum_scope": self.maximum_scope.to_dict(),
            "migration": self.migration.to_dict(),
            "operator_identity": self.operator_identity,
            "operator_kind": self.operator_kind.value,
            "preconditions": [item.value for item in self.preconditions],
            "proofs": [item.value for item in self.proofs],
            "public_api_impact": self.public_api_impact.value,
            "repository_tree": self.repository_tree,
            "risk_class": self.risk_class.value,
            "rollback": self.rollback.to_dict(),
            "schema": self.schema,
            "state_impact": self.state_impact.value,
            "target_kinds": [item.value for item in self.target_kinds],
            "target_node_ids": list(self.target_node_ids),
            "target_paths": list(self.target_paths),
            "validation": [item.value for item in self.validation],
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
            target_kinds=mapping["target_kinds"],
            target_node_ids=mapping["target_node_ids"],
            target_paths=mapping["target_paths"],
            authority_impact=mapping["authority_impact"],
            public_api_impact=mapping["public_api_impact"],
            state_impact=mapping["state_impact"],
            autonomy_disposition=mapping["autonomy_disposition"],
            risk_class=mapping["risk_class"],
            preconditions=mapping["preconditions"],
            validation=mapping["validation"],
            proofs=mapping["proofs"],
            migration=mapping["migration"],
            rollback=mapping["rollback"],
            maximum_scope=mapping["maximum_scope"],
            freshness=mapping["freshness"],
            can_authorize_execution=mapping["can_authorize_execution"],
            can_reduce_gates=mapping["can_reduce_gates"],
            can_self_promote=mapping["can_self_promote"],
            can_raise_ceiling=mapping["can_raise_ceiling"],
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
            raise RefactorCandidateError("refactor-candidate JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise RefactorCandidateError("refactor-candidate JSON must contain an object")
        return cls.from_mapping(decoded)


def declare_refactor_candidate(
    operator: RefactorOperator | OperatorKind | str,
    *,
    repository_tree: str,
    contract_identity: str,
    target_node_ids: Sequence[str],
    target_paths: Sequence[str],
    freshness: str,
    expected_effects: Sequence[ExpectedEffectKind | str] | None = None,
    target_kinds: Sequence[NodeKind | str] | None = None,
    authority_impact: AuthorityImpact | str | None = None,
    public_api_impact: PublicApiImpact | str | None = None,
    state_impact: StateImpact | str | None = None,
    autonomy_disposition: AutonomyDisposition | str | None = None,
    risk_class: AutonomyRiskClass | str | None = None,
    preconditions: Sequence[PreconditionKind | str] | None = None,
    validation: Sequence[ValidationObligationKind | str] | None = None,
    proofs: Sequence[ProofObligationKind | str] | None = None,
    migration: OperatorMigration | Mapping[str, Any] | None = None,
    rollback: OperatorRollback | Mapping[str, Any] | None = None,
    maximum_scope: MaximumScope | Mapping[str, Any] | None = None,
) -> RefactorCandidate:
    """Admit a candidate bound to one catalog operator and exact tree/contract/effects."""

    resolved = operator if isinstance(operator, RefactorOperator) else operator_for(operator)
    if resolved.schema != REFACTOR_OPERATOR_SCHEMA:
        raise RefactorCandidateError("candidate operator must use the closed operator schema")
    paths = tuple(target_paths)
    kinds = tuple(target_kinds) if target_kinds is not None else resolved.target_kinds
    scope = maximum_scope
    if scope is None:
        scope = MaximumScope(
            path_prefixes=resolved.maximum_scope.path_prefixes,
            target_kinds=kinds,
            max_paths=min(resolved.maximum_scope.max_paths, max(1, len(paths))),
            max_symbols=min(
                resolved.maximum_scope.max_symbols, max(1, len(tuple(target_node_ids)))
            ),
            allows_public_surface=resolved.maximum_scope.allows_public_surface,
            allows_state_stores=resolved.maximum_scope.allows_state_stores,
            allows_cross_package=resolved.maximum_scope.allows_cross_package,
        )
    return RefactorCandidate(
        operator_kind=resolved.kind,
        operator_identity=resolved.content_identity,
        repository_tree=repository_tree,
        contract_identity=contract_identity,
        expected_effects=tuple(expected_effects)
        if expected_effects is not None
        else resolved.expected_effects,
        target_kinds=kinds,
        target_node_ids=tuple(target_node_ids),
        target_paths=paths,
        authority_impact=authority_impact
        if authority_impact is not None
        else resolved.authority_impact,
        public_api_impact=public_api_impact
        if public_api_impact is not None
        else resolved.public_api_impact,
        state_impact=state_impact if state_impact is not None else resolved.state_impact,
        autonomy_disposition=autonomy_disposition
        if autonomy_disposition is not None
        else resolved.autonomy_disposition,
        risk_class=risk_class if risk_class is not None else resolved.risk_class,
        preconditions=tuple(preconditions)
        if preconditions is not None
        else resolved.preconditions,
        validation=tuple(validation) if validation is not None else resolved.validation,
        proofs=tuple(proofs) if proofs is not None else resolved.proofs,
        migration=migration if migration is not None else resolved.migration,
        rollback=rollback if rollback is not None else resolved.rollback,
        maximum_scope=scope,
        freshness=freshness,
    )


__all__ = [
    "CANDIDATE_CAN_AUTHORIZE_EXECUTION",
    "CANDIDATE_CAN_EXPAND_SCOPE",
    "CANDIDATE_CAN_RAISE_CEILING",
    "CANDIDATE_CAN_REDUCE_GATES",
    "CANDIDATE_CAN_SELF_PROMOTE",
    "CANDIDATE_IDENTITY_BINDS_CONTRACT",
    "CANDIDATE_IDENTITY_BINDS_EFFECTS",
    "CANDIDATE_IDENTITY_BINDS_TREE",
    "REFACTOR_CANDIDATE_EVIDENCE",
    "REFACTOR_CANDIDATE_SCHEMA",
    "REFACTOR_CANDIDATE_VERSION",
    "RefactorCandidate",
    "RefactorCandidateAuthorityError",
    "RefactorCandidateError",
    "declare_refactor_candidate",
]
