"""Closed declarative refactor-operator grammar (PCAR-012).

`RefactorOperator` is the only admitted change vocabulary. Every initial
operator is a complete immutable declaration of preconditions, target kinds,
expected effects, authority/API/state impact, migration, rollback,
validation, proof obligations, maximum scope, and autonomy risk. Unknown
operators and fields, missing declarations, scope expansion, script/shell
payloads, gate reduction, and self-authorization fail closed.

Operators describe bounded candidate changes only. They cannot authorize
execution, reduce gates, raise the autonomy ceiling, or promote themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .contracts import (
    ArchitectureContractError,
    NodeKind,
    _require_int,
    _require_mapping,
    _require_text,
    _repository_relative_path,
)

REFACTOR_OPERATOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refactor-operator@1"
)
REFACTOR_OPERATOR_VERSION = 1
REFACTOR_OPERATOR_EVIDENCE = "pcar/refactor-operator@1"
OPERATOR_SCOPE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/operator-maximum-scope@1"
)
OPERATOR_SCOPE_VERSION = 1
OPERATOR_MIGRATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/operator-migration@1"
)
OPERATOR_MIGRATION_VERSION = 1
OPERATOR_ROLLBACK_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/operator-rollback@1"
)
OPERATOR_ROLLBACK_VERSION = 1
OPERATOR_VALIDATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/operator-validation@1"
)
OPERATOR_VALIDATION_VERSION = 1
OPERATOR_CATALOG_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/operator-catalog@1"
)
OPERATOR_CATALOG_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-012-refactor-operator-grammar"
TASK_ID = "PCAR-012"
DEFAULT_FRESHNESS = "pcar-012-refactor-operator"
EFFECT_CLASS = "internal_pure_contract_addition"
OPERATOR_CAN_AUTHORIZE_EXECUTION = False
OPERATOR_CAN_REDUCE_GATES = False
OPERATOR_CAN_SELF_PROMOTE = False
OPERATOR_CAN_RAISE_AUTONOMY_CEILING = False
OPERATOR_CAN_ADMIT_SCRIPT_PAYLOAD = False
OPERATOR_CAN_EXPAND_SCOPE = False
ARBITRARY_EXECUTABLE_POLICY_PROHIBITED = True
UNKNOWN_OPERATOR_REJECTED = True
INCOMPLETE_DECLARATION_REJECTED = True
CANDIDATE_MAY_RAISE_CEILING = False

_UNKNOWN_FIELD_MESSAGE = "unknown refactor-operator field"
_MISSING_FIELD_MESSAGE = "missing refactor-operator field"
_INCOMPLETE_DECLARATION_MESSAGE = "incomplete refactor-operator declaration"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")
_SCRIPT_MARKERS = (
    "#!/",
    "/bin/sh",
    "/bin/bash",
    "os.system",
    "subprocess.",
    "bash -c",
    "powershell",
    "cmd.exe",
    "eval(",
    "exec(",
)
_SIBLING_PREFIXES = (
    "ipfs_datasets_py/",
    "ipfs_kit_py/",
    "ipfs_accelerate_py/mcplusplus/",
)
PROTECTED_PATHS: frozenset[str] = frozenset(
    {
        "docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_ARCHITECTURE_REFACTORER_PLAN.md",
        "docs/architecture/agent_supervisor_architecture_refactorer.objectives.md",
        "docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
        "config/agent_supervisor_architecture_refactorer_scheduler.json",
        "scripts/validate_agent_supervisor_architecture_refactorer_board.py",
        "scripts/run_agent_supervisor_architecture_refactorer.py",
        "test/api/architecture_refactorer/test_board.py",
    }
)

_EnumT = TypeVar("_EnumT", bound=Enum)


class RefactorOperatorError(ArchitectureContractError):
    """Fail-closed refactor-operator contract violation."""


class RefactorOperatorAuthorityError(RefactorOperatorError):
    """Raised when an operator is asked to authorize, promote, or expand itself."""


class OperatorKind(str, Enum):
    """Closed initial refactor-operator vocabulary (PCAR-PLAN-R1)."""

    EXTRACT_MODULE = "EXTRACT_MODULE"
    EXTRACT_INTERFACE = "EXTRACT_INTERFACE"
    EXTRACT_PURE_FUNCTION = "EXTRACT_PURE_FUNCTION"
    MOVE_STATE_TO_OWNER = "MOVE_STATE_TO_OWNER"
    INTRODUCE_DEPENDENCY_INVERSION = "INTRODUCE_DEPENDENCY_INVERSION"
    REPLACE_DIRECT_CALL_WITH_TYPED_SERVICE = "REPLACE_DIRECT_CALL_WITH_TYPED_SERVICE"
    GENERATE_ADAPTER = "GENERATE_ADAPTER"
    GENERATE_COMPATIBILITY_SHIM = "GENERATE_COMPATIBILITY_SHIM"
    QUARANTINE_LEGACY_PATH = "QUARANTINE_LEGACY_PATH"
    QUARANTINE_SIMULATION_PATH = "QUARANTINE_SIMULATION_PATH"
    REPLACE_BOOLEAN_WITH_CLOSED_OUTCOME = "REPLACE_BOOLEAN_WITH_CLOSED_OUTCOME"
    REPLACE_DYNAMIC_REGISTRY_WITH_TYPED_CATALOG = (
        "REPLACE_DYNAMIC_REGISTRY_WITH_TYPED_CATALOG"
    )
    REPLACE_EAGER_IMPORT_WITH_LAZY_CAPABILITY = (
        "REPLACE_EAGER_IMPORT_WITH_LAZY_CAPABILITY"
    )
    CONSOLIDATE_ERROR_VOCABULARY = "CONSOLIDATE_ERROR_VOCABULARY"
    CONSOLIDATE_RECEIPT_PRODUCER = "CONSOLIDATE_RECEIPT_PRODUCER"
    CONSOLIDATE_CAPABILITY_AUTHORITY = "CONSOLIDATE_CAPABILITY_AUTHORITY"
    REMOVE_CONFIRMED_DEAD_CODE = "REMOVE_CONFIRMED_DEAD_CODE"
    SPLIT_MONOLITH_BY_AUTHORITY = "SPLIT_MONOLITH_BY_AUTHORITY"
    MOVE_GENERATED_PROJECTION_OUT_OF_SOURCE_AUTHORITY = (
        "MOVE_GENERATED_PROJECTION_OUT_OF_SOURCE_AUTHORITY"
    )
    DEPRECATE_PUBLIC_SYMBOL = "DEPRECATE_PUBLIC_SYMBOL"
    REMOVE_DEPRECATED_SYMBOL_AFTER_GATE = "REMOVE_DEPRECATED_SYMBOL_AFTER_GATE"


INITIAL_OPERATORS: tuple[OperatorKind, ...] = tuple(OperatorKind)
REQUIRED_OPERATORS: tuple[OperatorKind, ...] = INITIAL_OPERATORS
CLOSED_OPERATORS: frozenset[str] = frozenset(item.value for item in OperatorKind)


class OperatorRiskClass(str, Enum):
    """Closed autonomy-risk vocabulary bound to each initial operator."""

    PURE_MODULE_EXTRACTION = "pure_module_extraction"
    GENERATED_PROJECTION_REGENERATION = "generated_projection_regeneration"
    LAZY_IMPORT_CONVERSION = "lazy_import_conversion"
    INTERNAL_ADAPTER_GENERATION = "internal_adapter_generation"
    CLOSED_RESULT_TYPE_MIGRATION = "closed_result_type_migration"
    CONFIRMED_DEAD_INTERNAL_CODE_REMOVAL = "confirmed_dead_internal_code_removal"
    TEST_FIXTURE_RELOCATION = "test_fixture_relocation"
    SIMULATION_NAMESPACE_RELOCATION = "simulation_namespace_relocation"
    PUBLIC_API_MIGRATION = "public_api_migration"
    STATE_MIGRATION = "state_migration"
    PROVIDER_MIGRATION = "provider_migration"
    RECEIPT_MIGRATION = "receipt_migration"
    LEGACY_MIGRATION = "legacy_migration"
    AUTHORITY_CONSOLIDATION = "authority_consolidation"
    AUTHORIZATION_POLICY_SECURITY = "authorization_policy_security"
    HIGH_RISK_HUMAN = "high_risk_human"


CLOSED_RISK_CLASSES: frozenset[str] = frozenset(
    item.value for item in OperatorRiskClass
)
AUTOMATIC_RISK_CLASSES: frozenset[OperatorRiskClass] = frozenset(
    {
        OperatorRiskClass.PURE_MODULE_EXTRACTION,
        OperatorRiskClass.GENERATED_PROJECTION_REGENERATION,
        OperatorRiskClass.LAZY_IMPORT_CONVERSION,
        OperatorRiskClass.INTERNAL_ADAPTER_GENERATION,
        OperatorRiskClass.CLOSED_RESULT_TYPE_MIGRATION,
        OperatorRiskClass.CONFIRMED_DEAD_INTERNAL_CODE_REMOVAL,
        OperatorRiskClass.TEST_FIXTURE_RELOCATION,
        OperatorRiskClass.SIMULATION_NAMESPACE_RELOCATION,
    }
)
PROPOSAL_ONLY_RISK_CLASSES: frozenset[OperatorRiskClass] = frozenset(
    {
        OperatorRiskClass.PUBLIC_API_MIGRATION,
        OperatorRiskClass.STATE_MIGRATION,
        OperatorRiskClass.PROVIDER_MIGRATION,
        OperatorRiskClass.RECEIPT_MIGRATION,
        OperatorRiskClass.LEGACY_MIGRATION,
        OperatorRiskClass.AUTHORITY_CONSOLIDATION,
    }
)
HUMAN_APPROVAL_RISK_CLASSES: frozenset[OperatorRiskClass] = frozenset(
    {
        OperatorRiskClass.AUTHORIZATION_POLICY_SECURITY,
        OperatorRiskClass.HIGH_RISK_HUMAN,
    }
)


class AutonomyCeiling(str, Enum):
    """Closed autonomy ceiling. Candidates cannot raise this value."""

    HUMAN_APPROVAL = "human_approval"
    PROPOSAL_ONLY = "proposal_only"
    AUTOMATIC = "automatic"


CLOSED_AUTONOMY_CEILINGS: frozenset[str] = frozenset(
    item.value for item in AutonomyCeiling
)
_AUTONOMY_RANK: dict[AutonomyCeiling, int] = {
    AutonomyCeiling.HUMAN_APPROVAL: 0,
    AutonomyCeiling.PROPOSAL_ONLY: 1,
    AutonomyCeiling.AUTOMATIC: 2,
}


class AuthorityImpact(str, Enum):
    """Closed authority-impact vocabulary. Transfer and promotion are absent."""

    NONE = "none"
    PRESERVE = "preserve"
    ADAPT = "adapt"
    QUARANTINE = "quarantine"
    CONSOLIDATE = "consolidate"
    DEPRECATE = "deprecate"


CLOSED_AUTHORITY_IMPACTS: frozenset[str] = frozenset(
    item.value for item in AuthorityImpact
)


class ApiImpact(str, Enum):
    """Closed public-API impact vocabulary."""

    NONE = "none"
    INTERNAL = "internal"
    COMPATIBILITY = "compatibility"
    DEPRECATE = "deprecate"
    REMOVE_AFTER_GATE = "remove_after_gate"
    PUBLIC_CONTRACT = "public_contract"


CLOSED_API_IMPACTS: frozenset[str] = frozenset(item.value for item in ApiImpact)


class StateImpact(str, Enum):
    """Closed state-impact vocabulary. Indefinite dual authority is absent."""

    NONE = "none"
    READ_ONLY = "read_only"
    MIGRATE_TO_OWNER = "migrate_to_owner"
    QUARANTINE = "quarantine"
    RETIRE = "retire"


CLOSED_STATE_IMPACTS: frozenset[str] = frozenset(item.value for item in StateImpact)


class ExpectedEffect(str, Enum):
    """Closed expected-effect vocabulary for admitted operators."""

    EXTRACT_SYMBOLS = "extract_symbols"
    EXTRACT_INTERFACE = "extract_interface"
    EXTRACT_PURE_FUNCTION = "extract_pure_function"
    MOVE_STATE = "move_state"
    INVERT_DEPENDENCY = "invert_dependency"
    REPLACE_DIRECT_CALL = "replace_direct_call"
    INTRODUCE_ADAPTER = "introduce_adapter"
    INTRODUCE_SHIM = "introduce_shim"
    QUARANTINE_LEGACY = "quarantine_legacy"
    QUARANTINE_SIMULATION = "quarantine_simulation"
    REPLACE_BOOLEAN = "replace_boolean"
    REPLACE_REGISTRY = "replace_registry"
    LAZY_IMPORT = "lazy_import"
    CONSOLIDATE_ERRORS = "consolidate_errors"
    CONSOLIDATE_RECEIPTS = "consolidate_receipts"
    CONSOLIDATE_AUTHORITY = "consolidate_authority"
    REMOVE_DEAD_CODE = "remove_dead_code"
    SPLIT_MODULE = "split_module"
    MOVE_GENERATED = "move_generated"
    DEPRECATE_SYMBOL = "deprecate_symbol"
    REMOVE_DEPRECATED_SYMBOL = "remove_deprecated_symbol"


CLOSED_EXPECTED_EFFECTS: frozenset[str] = frozenset(
    item.value for item in ExpectedEffect
)


class ProhibitedEffect(str, Enum):
    """Closed prohibited-effect vocabulary enforced on every operator."""

    SCRIPT_OR_SHELL_PAYLOAD = "script_or_shell_payload"
    SCOPE_EXPANSION = "scope_expansion"
    SELF_AUTHORIZATION = "self_authorization"
    SELF_PROMOTION = "self_promotion"
    GATE_REDUCTION = "gate_reduction"
    AUTONOMY_CEILING_RAISE = "autonomy_ceiling_raise"
    SIBLING_WRITE = "sibling_write"
    NETWORK_EFFECT = "network_effect"
    AUTHORITY_TRANSFER = "authority_transfer"
    AUTHORITY_WEAKENING = "authority_weakening"
    EFFECT_EXPANSION = "effect_expansion"
    INDEFINITE_DUAL_AUTHORITY = "indefinite_dual_authority"
    PROTECTED_PATH_WRITE = "protected_path_write"
    ARBITRARY_EXECUTABLE_POLICY = "arbitrary_executable_policy"


REQUIRED_PROHIBITED_EFFECTS: tuple[ProhibitedEffect, ...] = tuple(ProhibitedEffect)
CLOSED_PROHIBITED_EFFECTS: frozenset[str] = frozenset(
    item.value for item in ProhibitedEffect
)


class OperatorPrecondition(str, Enum):
    """Closed operator-precondition vocabulary."""

    ACCEPTED_ARCHITECTURE_IR = "accepted_architecture_ir"
    ACCEPTED_OWNERSHIP = "accepted_ownership"
    ACCEPTED_CONTRACTS = "accepted_contracts"
    ACCEPTED_BOUNDARIES = "accepted_boundaries"
    EXACT_REPOSITORY_TREE = "exact_repository_tree"
    DECLARED_SCOPE = "declared_scope"
    DECLARED_ROLLBACK = "declared_rollback"
    DECLARED_VALIDATION = "declared_validation"
    DECLARED_PROOFS = "declared_proofs"
    NO_SCRIPT_PAYLOAD = "no_script_payload"
    NO_SELF_AUTHORIZATION = "no_self_authorization"
    CONFIRMED_DEAD_INTERNAL = "confirmed_dead_internal"
    PRIOR_DEPRECATION = "prior_deprecation"
    CONSUMER_MIGRATION = "consumer_migration"
    COMPATIBILITY_SATISFACTION = "compatibility_satisfaction"
    UNIQUE_STATE_OWNER = "unique_state_owner"
    NO_DUAL_AUTHORITY = "no_dual_authority"
    CANONICAL_OWNER_RESOLVED = "canonical_owner_resolved"
    NO_CONTRACT_AMBIGUITY = "no_contract_ambiguity"
    CURRENT_POLICY_VALIDATION = "current_policy_validation"


CLOSED_PRECONDITIONS: frozenset[str] = frozenset(
    item.value for item in OperatorPrecondition
)
REQUIRED_PRECONDITIONS: tuple[OperatorPrecondition, ...] = (
    OperatorPrecondition.ACCEPTED_ARCHITECTURE_IR,
    OperatorPrecondition.ACCEPTED_OWNERSHIP,
    OperatorPrecondition.ACCEPTED_CONTRACTS,
    OperatorPrecondition.ACCEPTED_BOUNDARIES,
    OperatorPrecondition.EXACT_REPOSITORY_TREE,
    OperatorPrecondition.DECLARED_SCOPE,
    OperatorPrecondition.DECLARED_ROLLBACK,
    OperatorPrecondition.DECLARED_VALIDATION,
    OperatorPrecondition.DECLARED_PROOFS,
    OperatorPrecondition.NO_SCRIPT_PAYLOAD,
    OperatorPrecondition.NO_SELF_AUTHORIZATION,
)


class ProofObligation(str, Enum):
    """Closed proof-obligation vocabulary."""

    BEHAVIOR_PRESERVATION = "behavior_preservation"
    NO_EFFECT_EXPANSION = "no_effect_expansion"
    NO_AUTHORITY_WEAKENING = "no_authority_weakening"
    SCOPE_CONTAINMENT = "scope_containment"
    ROLLBACK_RESTORES_TREE = "rollback_restores_tree"
    VALIDATION_COVERAGE = "validation_coverage"
    PUBLIC_CONTRACT_STABILITY = "public_contract_stability"
    QUARANTINE_NON_PRODUCTION = "quarantine_non_production"
    UNIQUE_STATE_OWNER = "unique_state_owner"
    TRANSLATION_EQUIVALENCE = "translation_equivalence"
    NO_SELF_AUTHORIZATION = "no_self_authorization"


CLOSED_PROOF_OBLIGATIONS: frozenset[str] = frozenset(
    item.value for item in ProofObligation
)
REQUIRED_PROOF_OBLIGATIONS: tuple[ProofObligation, ...] = (
    ProofObligation.BEHAVIOR_PRESERVATION,
    ProofObligation.NO_EFFECT_EXPANSION,
    ProofObligation.NO_AUTHORITY_WEAKENING,
    ProofObligation.SCOPE_CONTAINMENT,
    ProofObligation.ROLLBACK_RESTORES_TREE,
    ProofObligation.VALIDATION_COVERAGE,
    ProofObligation.NO_SELF_AUTHORIZATION,
)


class ValidationObligation(str, Enum):
    """Closed validation-obligation vocabulary."""

    STATIC_TYPE_CHECK = "static_type_check"
    DECLARED_UNIT_TESTS = "declared_unit_tests"
    DIFFERENTIAL_BEHAVIOR = "differential_behavior"
    EFFECT_COMPARISON = "effect_comparison"
    AUTHORITY_COMPARISON = "authority_comparison"
    TRANSLATION_VALIDATION = "translation_validation"
    SCOPE_FENCE = "scope_fence"
    ROLLBACK_DRILL = "rollback_drill"
    CURRENT_TREE_PROOF = "current_tree_proof"


CLOSED_VALIDATION_OBLIGATIONS: frozenset[str] = frozenset(
    item.value for item in ValidationObligation
)
REQUIRED_VALIDATION_OBLIGATIONS: tuple[ValidationObligation, ...] = (
    ValidationObligation.STATIC_TYPE_CHECK,
    ValidationObligation.DECLARED_UNIT_TESTS,
    ValidationObligation.DIFFERENTIAL_BEHAVIOR,
    ValidationObligation.EFFECT_COMPARISON,
    ValidationObligation.AUTHORITY_COMPARISON,
    ValidationObligation.SCOPE_FENCE,
    ValidationObligation.ROLLBACK_DRILL,
    ValidationObligation.CURRENT_TREE_PROOF,
)


class OperatorMigrationPhase(str, Enum):
    """Closed migration-phase vocabulary for operator declarations."""

    DECLARE = "declare"
    ADAPT = "adapt"
    DEPRECATE = "deprecate"
    VALIDATE = "validate"
    SEAL = "seal"
    SNAPSHOT = "snapshot"
    DUAL_READ_SHADOW = "dual_read_shadow"
    CONTROLLED_DUAL_WRITE = "controlled_dual_write"
    CUTOVER = "cutover"
    VALIDATION = "validation"
    READ_ONLY_LEGACY = "read_only_legacy"
    RETIREMENT = "retirement"


CLOSED_MIGRATION_PHASES: frozenset[str] = frozenset(
    item.value for item in OperatorMigrationPhase
)
STRUCTURAL_MIGRATION_PHASES: tuple[OperatorMigrationPhase, ...] = (
    OperatorMigrationPhase.DECLARE,
    OperatorMigrationPhase.ADAPT,
    OperatorMigrationPhase.VALIDATE,
    OperatorMigrationPhase.SEAL,
)
DEPRECATION_MIGRATION_PHASES: tuple[OperatorMigrationPhase, ...] = (
    OperatorMigrationPhase.DECLARE,
    OperatorMigrationPhase.ADAPT,
    OperatorMigrationPhase.DEPRECATE,
    OperatorMigrationPhase.VALIDATE,
    OperatorMigrationPhase.SEAL,
)
STATE_MIGRATION_PHASES: tuple[OperatorMigrationPhase, ...] = (
    OperatorMigrationPhase.SNAPSHOT,
    OperatorMigrationPhase.DUAL_READ_SHADOW,
    OperatorMigrationPhase.CONTROLLED_DUAL_WRITE,
    OperatorMigrationPhase.CUTOVER,
    OperatorMigrationPhase.VALIDATION,
    OperatorMigrationPhase.READ_ONLY_LEGACY,
    OperatorMigrationPhase.RETIREMENT,
)


class RollbackAction(str, Enum):
    """Closed rollback action. Operators never execute rollback themselves."""

    RESTORE_ISOLATED_WORKTREE = "restore_isolated_worktree"


CLOSED_ROLLBACK_ACTIONS: frozenset[str] = frozenset(
    item.value for item in RollbackAction
)
_ROLLBACK_MESSAGE = (
    "restore the isolated worktree to the exact pre-change tree; no candidate was executed"
)

CLOSED_ADAPTERS: frozenset[str] = frozenset(
    {
        "typed_service_adapter",
        "compatibility_shim",
        "state_owner_adapter",
        "dependency_inversion_adapter",
        "lazy_capability_adapter",
        "generated_projection_adapter",
        "quarantine_adapter",
        "deprecation_adapter",
        "error_vocabulary_adapter",
        "receipt_producer_adapter",
        "authority_adapter",
        "closed_outcome_adapter",
        "typed_catalog_adapter",
    }
)

_SCOPE_FIELDS = frozenset(
    {
        "allows_network",
        "allows_protected_path",
        "allows_scope_expansion",
        "allows_script_payload",
        "allows_sibling_write",
        "content_identity",
        "max_files",
        "max_packages",
        "max_paths",
        "max_symbols",
        "repository_relative_only",
        "schema",
        "version",
    }
)
_MIGRATION_FIELDS = frozenset(
    {
        "adapters",
        "content_identity",
        "indefinite_dual_authority",
        "mutates_state",
        "phases",
        "schema",
        "transfers_authority",
        "version",
    }
)
_ROLLBACK_FIELDS = frozenset(
    {
        "action",
        "content_identity",
        "message",
        "required",
        "restores_tree",
        "schema",
        "version",
    }
)
_VALIDATION_FIELDS = frozenset(
    {
        "content_identity",
        "gates_reducible",
        "obligations",
        "required",
        "schema",
        "version",
    }
)
_OPERATOR_FIELDS = frozenset(
    {
        "api_impact",
        "authority_impact",
        "autonomy_ceiling",
        "can_admit_script_payload",
        "can_authorize_execution",
        "can_raise_autonomy_ceiling",
        "can_reduce_gates",
        "can_self_promote",
        "content_identity",
        "expected_effects",
        "kind",
        "maximum_scope",
        "migration",
        "preconditions",
        "prohibited_effects",
        "proofs",
        "risk_class",
        "rollback",
        "schema",
        "state_impact",
        "target_kinds",
        "validation",
        "version",
    }
)
_CATALOG_FIELDS = frozenset(
    {
        "can_admit_script_payload",
        "can_authorize_execution",
        "can_raise_autonomy_ceiling",
        "can_reduce_gates",
        "can_self_promote",
        "content_identity",
        "covers_initial_operators",
        "effect_class",
        "freshness",
        "operators",
        "schema",
        "version",
    }
)
REQUIRED_OPERATOR_DECLARATION_FIELDS: tuple[str, ...] = (
    "kind",
    "preconditions",
    "target_kinds",
    "expected_effects",
    "authority_impact",
    "api_impact",
    "state_impact",
    "migration",
    "rollback",
    "validation",
    "proofs",
    "maximum_scope",
    "risk_class",
    "autonomy_ceiling",
)


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str) -> str:
    try:
        return validate_cid(value, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise RefactorOperatorError(
            "content identity must be a dag-json CIDv1"
        ) from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise RefactorOperatorError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise RefactorOperatorError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RefactorOperatorError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    number = _require_int(value, name, error_type=RefactorOperatorError)
    if number < 1:
        raise RefactorOperatorError(f"{name} must be a positive integer")
    return number


def _require_closed(value: Any, enum_type: type[_EnumT], name: str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise RefactorOperatorError(f"unsupported refactor {name}: {value!r}") from exc


def _require_closed_tuple(
    value: Any,
    enum_type: type[_EnumT],
    name: str,
    *,
    ordered: bool = False,
) -> tuple[_EnumT, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RefactorOperatorError(f"{name} must be a list of {name}")
    items = tuple(_require_closed(item, enum_type, name) for item in value)
    seen: set[str] = set()
    unique: list[_EnumT] = []
    for item in items:
        if item.value in seen:
            if ordered:
                raise RefactorOperatorError(f"{name} must be unique")
            continue
        seen.add(item.value)
        unique.append(item)
    if ordered:
        return tuple(unique)
    return tuple(sorted(unique, key=lambda item: item.value))


def _require_node_kinds(value: Any, name: str) -> tuple[NodeKind, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RefactorOperatorError(f"{name} must be a list of node kinds")
    items: list[NodeKind] = []
    seen: set[str] = set()
    for item in value:
        kind = item if isinstance(item, NodeKind) else None
        if kind is None:
            try:
                kind = NodeKind(item)
            except (TypeError, ValueError) as exc:
                raise RefactorOperatorError(
                    f"unsupported refactor target kind: {item!r}"
                ) from exc
        if kind.value in seen:
            continue
        seen.add(kind.value)
        items.append(kind)
    if not items:
        raise RefactorOperatorError(f"{_INCOMPLETE_DECLARATION_MESSAGE}: {name}")
    return tuple(sorted(items, key=lambda item: item.value))


def _require_adapters(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RefactorOperatorError("adapters must be a list of strings")
    items = tuple(
        _require_text(item, "adapter", error_type=RefactorOperatorError) for item in value
    )
    unknown = sorted(set(items) - CLOSED_ADAPTERS)
    if unknown:
        raise RefactorOperatorError(f"unsupported refactor adapter: {unknown}")
    return tuple(sorted(set(items)))


def _reject_script_text(value: str, name: str) -> str:
    lowered = value.lower()
    if any(marker.lower() in lowered for marker in _SCRIPT_MARKERS):
        refuse_script_payload(name)
    return value


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def autonomy_ceiling_for_risk(risk_class: OperatorRiskClass) -> AutonomyCeiling:
    """Return the declared ceiling for one closed risk class."""

    if risk_class in AUTOMATIC_RISK_CLASSES:
        return AutonomyCeiling.AUTOMATIC
    if risk_class in PROPOSAL_ONLY_RISK_CLASSES:
        return AutonomyCeiling.PROPOSAL_ONLY
    if risk_class in HUMAN_APPROVAL_RISK_CLASSES:
        return AutonomyCeiling.HUMAN_APPROVAL
    raise RefactorOperatorError(f"unsupported refactor risk class: {risk_class!r}")


def refuse_self_authorization(action: str = "authorize") -> None:
    """Reject attempts to treat an operator as execution authority."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} execution"
    )


def refuse_self_promotion(action: str = "promote") -> None:
    """Reject attempts to promote an operator or candidate into authority."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} itself"
    )


def refuse_gate_reduction(action: str = "reduce") -> None:
    """Reject attempts to drop validation or proof gates."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} validation or proof gates"
    )


def refuse_autonomy_ceiling_raise(action: str = "raise") -> None:
    """Reject attempts to raise a declared autonomy ceiling."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} its autonomy ceiling"
    )


def refuse_script_payload(action: str = "admit") -> None:
    """Reject arbitrary script or shell payloads."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} a script or shell payload"
    )


def refuse_scope_expansion(action: str = "expand") -> None:
    """Reject undeclared scope expansion."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} maximum scope"
    )


def _bind_identity(record: Any, claimed: str, name: str) -> str:
    identity = _content_identity(record._identity_payload())
    if claimed:
        observed = _validate_dag_json_cid(
            _require_text(claimed, "content_identity", error_type=RefactorOperatorError)
        )
        if observed != identity:
            raise RefactorOperatorError(f"{name} content identity mismatch")
    return identity


@dataclass(frozen=True)
class MaximumScope:
    """Hard numeric and path bounds. Expansion is never admitted."""

    max_paths: int
    max_files: int
    max_symbols: int
    max_packages: int
    allows_sibling_write: bool = False
    allows_network: bool = False
    allows_script_payload: bool = False
    allows_scope_expansion: bool = False
    allows_protected_path: bool = False
    repository_relative_only: bool = True
    schema: str = OPERATOR_SCOPE_SCHEMA
    version: int = OPERATOR_SCOPE_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=RefactorOperatorError)
        if schema != OPERATOR_SCOPE_SCHEMA:
            raise RefactorOperatorError("unexpected operator-scope schema")
        version = _require_int(self.version, "version", error_type=RefactorOperatorError)
        if version != OPERATOR_SCOPE_VERSION:
            raise RefactorOperatorError("unexpected operator-scope version")
        if _require_bool(self.allows_scope_expansion, "allows_scope_expansion"):
            refuse_scope_expansion("expand")
        if _require_bool(self.allows_script_payload, "allows_script_payload"):
            refuse_script_payload("admit")
        if _require_bool(self.allows_sibling_write, "allows_sibling_write"):
            raise RefactorOperatorError("operator scope cannot write sibling repositories")
        if _require_bool(self.allows_network, "allows_network"):
            raise RefactorOperatorError("operator scope cannot admit network effects")
        if _require_bool(self.allows_protected_path, "allows_protected_path"):
            raise RefactorOperatorError("operator scope cannot write protected paths")
        if _require_bool(self.repository_relative_only, "repository_relative_only") is False:
            raise RefactorOperatorError("operator scope must be repository-relative")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "max_paths", _require_positive_int(self.max_paths, "max_paths"))
        object.__setattr__(self, "max_files", _require_positive_int(self.max_files, "max_files"))
        object.__setattr__(
            self, "max_symbols", _require_positive_int(self.max_symbols, "max_symbols")
        )
        object.__setattr__(
            self, "max_packages", _require_positive_int(self.max_packages, "max_packages")
        )
        if self.max_files > self.max_paths:
            raise RefactorOperatorError("max_files cannot exceed max_paths")
        object.__setattr__(self, "allows_sibling_write", False)
        object.__setattr__(self, "allows_network", False)
        object.__setattr__(self, "allows_script_payload", False)
        object.__setattr__(self, "allows_scope_expansion", False)
        object.__setattr__(self, "allows_protected_path", False)
        object.__setattr__(self, "repository_relative_only", True)
        object.__setattr__(
            self, "content_identity", _bind_identity(self, self.content_identity, "operator-scope")
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "allows_network": False,
            "allows_protected_path": False,
            "allows_scope_expansion": False,
            "allows_script_payload": False,
            "allows_sibling_write": False,
            "max_files": self.max_files,
            "max_packages": self.max_packages,
            "max_paths": self.max_paths,
            "max_symbols": self.max_symbols,
            "repository_relative_only": True,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise RefactorOperatorError("operator-scope content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MaximumScope":
        mapping = _require_mapping(payload, error_type=RefactorOperatorError)
        _require_fields(mapping, _SCOPE_FIELDS)
        record = cls(
            max_paths=mapping["max_paths"],
            max_files=mapping["max_files"],
            max_symbols=mapping["max_symbols"],
            max_packages=mapping["max_packages"],
            allows_sibling_write=mapping["allows_sibling_write"],
            allows_network=mapping["allows_network"],
            allows_script_payload=mapping["allows_script_payload"],
            allows_scope_expansion=mapping["allows_scope_expansion"],
            allows_protected_path=mapping["allows_protected_path"],
            repository_relative_only=mapping["repository_relative_only"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorOperatorError("operator-scope content identity mismatch")
        return record

    from_dict = from_mapping

    def contains(
        self,
        *,
        path_count: int,
        file_count: int,
        symbol_count: int,
        package_count: int,
    ) -> bool:
        return (
            path_count <= self.max_paths
            and file_count <= self.max_files
            and symbol_count <= self.max_symbols
            and package_count <= self.max_packages
        )


@dataclass(frozen=True)
class OperatorMigration:
    """Named migration plan. Operators do not execute it."""

    phases: tuple[OperatorMigrationPhase, ...]
    adapters: tuple[str, ...] = ()
    mutates_state: bool = False
    transfers_authority: bool = False
    indefinite_dual_authority: bool = False
    schema: str = OPERATOR_MIGRATION_SCHEMA
    version: int = OPERATOR_MIGRATION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=RefactorOperatorError)
        if schema != OPERATOR_MIGRATION_SCHEMA:
            raise RefactorOperatorError("unexpected operator-migration schema")
        version = _require_int(self.version, "version", error_type=RefactorOperatorError)
        if version != OPERATOR_MIGRATION_VERSION:
            raise RefactorOperatorError("unexpected operator-migration version")
        if _require_bool(self.transfers_authority, "transfers_authority"):
            raise RefactorOperatorAuthorityError(
                "refactor operator cannot transfer authority"
            )
        if _require_bool(self.indefinite_dual_authority, "indefinite_dual_authority"):
            raise RefactorOperatorError("indefinite dual authority is prohibited")
        phases = _require_closed_tuple(
            self.phases, OperatorMigrationPhase, "migration phase", ordered=True
        )
        if not phases:
            raise RefactorOperatorError(f"{_INCOMPLETE_DECLARATION_MESSAGE}: phases")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "adapters", _require_adapters(self.adapters))
        object.__setattr__(self, "mutates_state", _require_bool(self.mutates_state, "mutates_state"))
        object.__setattr__(self, "transfers_authority", False)
        object.__setattr__(self, "indefinite_dual_authority", False)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "operator-migration"),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "adapters": list(self.adapters),
            "indefinite_dual_authority": False,
            "mutates_state": self.mutates_state,
            "phases": [item.value for item in self.phases],
            "schema": self.schema,
            "transfers_authority": False,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise RefactorOperatorError("operator-migration content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatorMigration":
        mapping = _require_mapping(payload, error_type=RefactorOperatorError)
        _require_fields(mapping, _MIGRATION_FIELDS)
        record = cls(
            phases=mapping["phases"],
            adapters=mapping["adapters"],
            mutates_state=mapping["mutates_state"],
            transfers_authority=mapping["transfers_authority"],
            indefinite_dual_authority=mapping["indefinite_dual_authority"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorOperatorError("operator-migration content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class OperatorRollback:
    """Exact rollback obligation. Missing rollback fails closed."""

    action: RollbackAction = RollbackAction.RESTORE_ISOLATED_WORKTREE
    message: str = _ROLLBACK_MESSAGE
    restores_tree: bool = True
    required: bool = True
    schema: str = OPERATOR_ROLLBACK_SCHEMA
    version: int = OPERATOR_ROLLBACK_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=RefactorOperatorError)
        if schema != OPERATOR_ROLLBACK_SCHEMA:
            raise RefactorOperatorError("unexpected operator-rollback schema")
        version = _require_int(self.version, "version", error_type=RefactorOperatorError)
        if version != OPERATOR_ROLLBACK_VERSION:
            raise RefactorOperatorError("unexpected operator-rollback version")
        action = _require_closed(self.action, RollbackAction, "rollback action")
        message = _reject_script_text(
            _require_text(self.message, "message", error_type=RefactorOperatorError),
            "rollback message",
        )
        if _require_bool(self.required, "required") is False:
            raise RefactorOperatorError(f"{_INCOMPLETE_DECLARATION_MESSAGE}: rollback")
        if _require_bool(self.restores_tree, "restores_tree") is False:
            raise RefactorOperatorError("operator rollback must restore the sealed tree")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "restores_tree", True)
        object.__setattr__(self, "required", True)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "operator-rollback"),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "message": self.message,
            "required": True,
            "restores_tree": True,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise RefactorOperatorError("operator-rollback content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatorRollback":
        mapping = _require_mapping(payload, error_type=RefactorOperatorError)
        _require_fields(mapping, _ROLLBACK_FIELDS)
        record = cls(
            action=mapping["action"],
            message=mapping["message"],
            restores_tree=mapping["restores_tree"],
            required=mapping["required"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorOperatorError("operator-rollback content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class OperatorValidation:
    """Required validation gates. Gates cannot be reduced."""

    obligations: tuple[ValidationObligation, ...]
    required: bool = True
    gates_reducible: bool = False
    schema: str = OPERATOR_VALIDATION_SCHEMA
    version: int = OPERATOR_VALIDATION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=RefactorOperatorError)
        if schema != OPERATOR_VALIDATION_SCHEMA:
            raise RefactorOperatorError("unexpected operator-validation schema")
        version = _require_int(self.version, "version", error_type=RefactorOperatorError)
        if version != OPERATOR_VALIDATION_VERSION:
            raise RefactorOperatorError("unexpected operator-validation version")
        if _require_bool(self.gates_reducible, "gates_reducible"):
            refuse_gate_reduction("reduce")
        if _require_bool(self.required, "required") is False:
            raise RefactorOperatorError(f"{_INCOMPLETE_DECLARATION_MESSAGE}: validation")
        obligations = _require_closed_tuple(
            self.obligations, ValidationObligation, "validation obligation"
        )
        missing = [
            item.value for item in REQUIRED_VALIDATION_OBLIGATIONS if item not in obligations
        ]
        if missing:
            raise RefactorOperatorError(
                f"{_INCOMPLETE_DECLARATION_MESSAGE}: validation {missing}"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "obligations", obligations)
        object.__setattr__(self, "required", True)
        object.__setattr__(self, "gates_reducible", False)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "operator-validation"),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "gates_reducible": False,
            "obligations": [item.value for item in self.obligations],
            "required": True,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise RefactorOperatorError("operator-validation content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatorValidation":
        mapping = _require_mapping(payload, error_type=RefactorOperatorError)
        _require_fields(mapping, _VALIDATION_FIELDS)
        record = cls(
            obligations=mapping["obligations"],
            required=mapping["required"],
            gates_reducible=mapping["gates_reducible"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorOperatorError("operator-validation content identity mismatch")
        return record

    from_dict = from_mapping


def _record_or_mapping(value: Any, record_type: type[Any], name: str) -> Any:
    if isinstance(value, record_type):
        return value
    if isinstance(value, Mapping):
        return record_type.from_mapping(value)
    raise RefactorOperatorError(f"{name} must be an object")


@dataclass(frozen=True)
class RefactorOperator:
    """One complete closed operator declaration."""

    kind: OperatorKind
    preconditions: tuple[OperatorPrecondition, ...]
    target_kinds: tuple[NodeKind, ...]
    expected_effects: tuple[ExpectedEffect, ...]
    authority_impact: AuthorityImpact
    api_impact: ApiImpact
    state_impact: StateImpact
    migration: OperatorMigration
    rollback: OperatorRollback
    validation: OperatorValidation
    proofs: tuple[ProofObligation, ...]
    maximum_scope: MaximumScope
    risk_class: OperatorRiskClass
    prohibited_effects: tuple[ProhibitedEffect, ...] = REQUIRED_PROHIBITED_EFFECTS
    autonomy_ceiling: AutonomyCeiling | None = None
    can_authorize_execution: bool = False
    can_reduce_gates: bool = False
    can_self_promote: bool = False
    can_raise_autonomy_ceiling: bool = False
    can_admit_script_payload: bool = False
    schema: str = REFACTOR_OPERATOR_SCHEMA
    version: int = REFACTOR_OPERATOR_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=RefactorOperatorError)
        if schema != REFACTOR_OPERATOR_SCHEMA:
            raise RefactorOperatorError("unexpected refactor-operator schema")
        version = _require_int(self.version, "version", error_type=RefactorOperatorError)
        if version != REFACTOR_OPERATOR_VERSION:
            raise RefactorOperatorError("unexpected refactor-operator version")
        kind = _require_closed(self.kind, OperatorKind, "operator kind")
        if _require_bool(self.can_authorize_execution, "can_authorize_execution"):
            refuse_self_authorization("authorize")
        if _require_bool(self.can_self_promote, "can_self_promote"):
            refuse_self_promotion("promote")
        if _require_bool(self.can_reduce_gates, "can_reduce_gates"):
            refuse_gate_reduction("reduce")
        if _require_bool(self.can_raise_autonomy_ceiling, "can_raise_autonomy_ceiling"):
            refuse_autonomy_ceiling_raise("raise")
        if _require_bool(self.can_admit_script_payload, "can_admit_script_payload"):
            refuse_script_payload("admit")
        preconditions = _require_closed_tuple(
            self.preconditions, OperatorPrecondition, "precondition"
        )
        missing_pre = [
            item.value for item in REQUIRED_PRECONDITIONS if item not in preconditions
        ]
        if missing_pre:
            raise RefactorOperatorError(
                f"{_INCOMPLETE_DECLARATION_MESSAGE}: preconditions {missing_pre}"
            )
        target_kinds = _require_node_kinds(self.target_kinds, "target_kinds")
        expected_effects = _require_closed_tuple(
            self.expected_effects, ExpectedEffect, "expected effect"
        )
        if not expected_effects:
            raise RefactorOperatorError(
                f"{_INCOMPLETE_DECLARATION_MESSAGE}: expected_effects"
            )
        prohibited = _require_closed_tuple(
            self.prohibited_effects, ProhibitedEffect, "prohibited effect"
        )
        if frozenset(prohibited) != frozenset(REQUIRED_PROHIBITED_EFFECTS):
            raise RefactorOperatorError("prohibited effects must be the closed set")
        proofs = _require_closed_tuple(self.proofs, ProofObligation, "proof obligation")
        missing_proofs = [
            item.value for item in REQUIRED_PROOF_OBLIGATIONS if item not in proofs
        ]
        if missing_proofs:
            raise RefactorOperatorError(
                f"{_INCOMPLETE_DECLARATION_MESSAGE}: proofs {missing_proofs}"
            )
        risk_class = _require_closed(self.risk_class, OperatorRiskClass, "risk class")
        declared_ceiling = autonomy_ceiling_for_risk(risk_class)
        if self.autonomy_ceiling is None:
            ceiling = declared_ceiling
        else:
            ceiling = _require_closed(
                self.autonomy_ceiling, AutonomyCeiling, "autonomy ceiling"
            )
            if _AUTONOMY_RANK[ceiling] > _AUTONOMY_RANK[declared_ceiling]:
                refuse_autonomy_ceiling_raise("raise")
            if ceiling != declared_ceiling:
                raise RefactorOperatorError(
                    "operator autonomy ceiling must match its risk class"
                )
        authority_impact = _require_closed(
            self.authority_impact, AuthorityImpact, "authority impact"
        )
        api_impact = _require_closed(self.api_impact, ApiImpact, "api impact")
        state_impact = _require_closed(self.state_impact, StateImpact, "state impact")
        migration = _record_or_mapping(self.migration, OperatorMigration, "migration")
        rollback = _record_or_mapping(self.rollback, OperatorRollback, "rollback")
        validation = _record_or_mapping(self.validation, OperatorValidation, "validation")
        maximum_scope = _record_or_mapping(
            self.maximum_scope, MaximumScope, "maximum_scope"
        )
        if migration.mutates_state and state_impact is StateImpact.NONE:
            raise RefactorOperatorError(
                "state-mutating operators must declare a state impact"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "target_kinds", target_kinds)
        object.__setattr__(self, "expected_effects", expected_effects)
        object.__setattr__(self, "authority_impact", authority_impact)
        object.__setattr__(self, "api_impact", api_impact)
        object.__setattr__(self, "state_impact", state_impact)
        object.__setattr__(self, "migration", migration)
        object.__setattr__(self, "rollback", rollback)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "proofs", proofs)
        object.__setattr__(self, "maximum_scope", maximum_scope)
        object.__setattr__(self, "risk_class", risk_class)
        object.__setattr__(self, "prohibited_effects", REQUIRED_PROHIBITED_EFFECTS)
        object.__setattr__(self, "autonomy_ceiling", ceiling)
        object.__setattr__(self, "can_authorize_execution", False)
        object.__setattr__(self, "can_reduce_gates", False)
        object.__setattr__(self, "can_self_promote", False)
        object.__setattr__(self, "can_raise_autonomy_ceiling", False)
        object.__setattr__(self, "can_admit_script_payload", False)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "refactor-operator"),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "api_impact": self.api_impact.value,
            "authority_impact": self.authority_impact.value,
            "autonomy_ceiling": self.autonomy_ceiling.value,
            "can_admit_script_payload": False,
            "can_authorize_execution": False,
            "can_raise_autonomy_ceiling": False,
            "can_reduce_gates": False,
            "can_self_promote": False,
            "expected_effects": [item.value for item in self.expected_effects],
            "kind": self.kind.value,
            "maximum_scope": self.maximum_scope.to_dict(),
            "migration": self.migration.to_dict(),
            "preconditions": [item.value for item in self.preconditions],
            "prohibited_effects": [item.value for item in self.prohibited_effects],
            "proofs": [item.value for item in self.proofs],
            "risk_class": self.risk_class.value,
            "rollback": self.rollback.to_dict(),
            "schema": self.schema,
            "state_impact": self.state_impact.value,
            "target_kinds": [item.value for item in self.target_kinds],
            "validation": self.validation.to_dict(),
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise RefactorOperatorError("refactor-operator content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RefactorOperator":
        mapping = _require_mapping(payload, error_type=RefactorOperatorError)
        _require_fields(mapping, _OPERATOR_FIELDS)
        record = cls(
            kind=mapping["kind"],
            preconditions=mapping["preconditions"],
            target_kinds=mapping["target_kinds"],
            expected_effects=mapping["expected_effects"],
            authority_impact=mapping["authority_impact"],
            api_impact=mapping["api_impact"],
            state_impact=mapping["state_impact"],
            migration=mapping["migration"],
            rollback=mapping["rollback"],
            validation=mapping["validation"],
            proofs=mapping["proofs"],
            maximum_scope=mapping["maximum_scope"],
            risk_class=mapping["risk_class"],
            prohibited_effects=mapping["prohibited_effects"],
            autonomy_ceiling=mapping["autonomy_ceiling"],
            can_authorize_execution=mapping["can_authorize_execution"],
            can_reduce_gates=mapping["can_reduce_gates"],
            can_self_promote=mapping["can_self_promote"],
            can_raise_autonomy_ceiling=mapping["can_raise_autonomy_ceiling"],
            can_admit_script_payload=mapping["can_admit_script_payload"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorOperatorError("refactor-operator content identity mismatch")
        return record

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "RefactorOperator":
        if type(payload) is not str or not payload:
            raise RefactorOperatorError("refactor-operator JSON must be a nonempty string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RefactorOperatorError("refactor-operator JSON is invalid") from exc
        return cls.from_mapping(decoded)

    def apply(self) -> None:
        refuse_self_authorization("apply")

    def authorize(self) -> None:
        refuse_self_authorization("authorize")

    def promote(self) -> None:
        refuse_self_promotion("promote")

    def reduce_gates(self) -> None:
        refuse_gate_reduction("reduce")

    def raise_autonomy_ceiling(self) -> None:
        refuse_autonomy_ceiling_raise("raise")

    def admit_script(self) -> None:
        refuse_script_payload("admit")

    def expand_scope(self) -> None:
        refuse_scope_expansion("expand")


@dataclass(frozen=True)
class OperatorCatalog:
    """Complete closed catalog of initial operators."""

    operators: tuple[RefactorOperator, ...]
    covers_initial_operators: bool = True
    can_authorize_execution: bool = False
    can_reduce_gates: bool = False
    can_self_promote: bool = False
    can_raise_autonomy_ceiling: bool = False
    can_admit_script_payload: bool = False
    effect_class: str = EFFECT_CLASS
    freshness: str = DEFAULT_FRESHNESS
    schema: str = OPERATOR_CATALOG_SCHEMA
    version: int = OPERATOR_CATALOG_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=RefactorOperatorError)
        if schema != OPERATOR_CATALOG_SCHEMA:
            raise RefactorOperatorError("unexpected operator-catalog schema")
        version = _require_int(self.version, "version", error_type=RefactorOperatorError)
        if version != OPERATOR_CATALOG_VERSION:
            raise RefactorOperatorError("unexpected operator-catalog version")
        effect_class = _require_text(
            self.effect_class, "effect_class", error_type=RefactorOperatorError
        )
        if effect_class != EFFECT_CLASS:
            raise RefactorOperatorError("unexpected operator-catalog effect class")
        freshness = _reject_script_text(
            _require_text(self.freshness, "freshness", error_type=RefactorOperatorError),
            "freshness",
        )
        if _require_bool(self.can_authorize_execution, "can_authorize_execution"):
            refuse_self_authorization("authorize")
        if _require_bool(self.can_self_promote, "can_self_promote"):
            refuse_self_promotion("promote")
        if _require_bool(self.can_reduce_gates, "can_reduce_gates"):
            refuse_gate_reduction("reduce")
        if _require_bool(self.can_raise_autonomy_ceiling, "can_raise_autonomy_ceiling"):
            refuse_autonomy_ceiling_raise("raise")
        if _require_bool(self.can_admit_script_payload, "can_admit_script_payload"):
            refuse_script_payload("admit")
        if isinstance(self.operators, (str, bytes, bytearray)) or not isinstance(
            self.operators, Sequence
        ):
            raise RefactorOperatorError("operators must be a list of objects")
        records = tuple(
            item if isinstance(item, RefactorOperator) else RefactorOperator.from_mapping(item)
            for item in self.operators
        )
        by_kind = {item.kind: item for item in records}
        if len(by_kind) != len(records):
            raise RefactorOperatorError("operator catalog kinds must be unique")
        missing = [item.value for item in INITIAL_OPERATORS if item not in by_kind]
        if missing:
            raise RefactorOperatorError(
                f"{_INCOMPLETE_DECLARATION_MESSAGE}: operators {missing}"
            )
        extra = sorted(kind.value for kind in by_kind if kind not in set(INITIAL_OPERATORS))
        if extra:
            raise RefactorOperatorError(f"unsupported refactor operator kind: {extra}")
        if _require_bool(self.covers_initial_operators, "covers_initial_operators") is False:
            raise RefactorOperatorError("operator catalog must cover the initial vocabulary")
        ordered = tuple(by_kind[kind] for kind in INITIAL_OPERATORS)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "effect_class", EFFECT_CLASS)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "operators", ordered)
        object.__setattr__(self, "covers_initial_operators", True)
        object.__setattr__(self, "can_authorize_execution", False)
        object.__setattr__(self, "can_reduce_gates", False)
        object.__setattr__(self, "can_self_promote", False)
        object.__setattr__(self, "can_raise_autonomy_ceiling", False)
        object.__setattr__(self, "can_admit_script_payload", False)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "operator-catalog"),
        )

    def operator(self, kind: OperatorKind | str) -> RefactorOperator:
        closed = _require_closed(kind, OperatorKind, "operator kind")
        for item in self.operators:
            if item.kind is closed:
                return item
        raise RefactorOperatorError(f"unsupported refactor operator: {closed.value!r}")

    def autonomy_classification_map(self) -> dict[str, str]:
        return {item.kind.value: item.autonomy_ceiling.value for item in self.operators}

    def risk_classification_map(self) -> dict[str, str]:
        return {item.kind.value: item.risk_class.value for item in self.operators}

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "can_admit_script_payload": False,
            "can_authorize_execution": False,
            "can_raise_autonomy_ceiling": False,
            "can_reduce_gates": False,
            "can_self_promote": False,
            "covers_initial_operators": True,
            "effect_class": EFFECT_CLASS,
            "freshness": self.freshness,
            "operators": [item.to_dict() for item in self.operators],
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise RefactorOperatorError("operator-catalog content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OperatorCatalog":
        mapping = _require_mapping(payload, error_type=RefactorOperatorError)
        _require_fields(mapping, _CATALOG_FIELDS)
        record = cls(
            operators=mapping["operators"],
            covers_initial_operators=mapping["covers_initial_operators"],
            can_authorize_execution=mapping["can_authorize_execution"],
            can_reduce_gates=mapping["can_reduce_gates"],
            can_self_promote=mapping["can_self_promote"],
            can_raise_autonomy_ceiling=mapping["can_raise_autonomy_ceiling"],
            can_admit_script_payload=mapping["can_admit_script_payload"],
            effect_class=mapping["effect_class"],
            freshness=mapping["freshness"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorOperatorError("operator-catalog content identity mismatch")
        return record

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "OperatorCatalog":
        if type(payload) is not str or not payload:
            raise RefactorOperatorError("operator-catalog JSON must be a nonempty string")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RefactorOperatorError("operator-catalog JSON is invalid") from exc
        return cls.from_mapping(decoded)

    def apply(self) -> None:
        refuse_self_authorization("apply")

    def authorize(self) -> None:
        refuse_self_authorization("authorize")

    def promote(self) -> None:
        refuse_self_promotion("promote")


def _scope(max_paths: int, max_files: int, max_symbols: int, max_packages: int) -> MaximumScope:
    return MaximumScope(
        max_paths=max_paths,
        max_files=max_files,
        max_symbols=max_symbols,
        max_packages=max_packages,
    )


def _migration(
    phases: tuple[OperatorMigrationPhase, ...],
    *,
    adapters: tuple[str, ...] = (),
    mutates_state: bool = False,
) -> OperatorMigration:
    return OperatorMigration(phases=phases, adapters=adapters, mutates_state=mutates_state)


def _preconditions(*extra: OperatorPrecondition) -> tuple[OperatorPrecondition, ...]:
    return REQUIRED_PRECONDITIONS + extra


def _proofs(*extra: ProofObligation) -> tuple[ProofObligation, ...]:
    return REQUIRED_PROOF_OBLIGATIONS + extra


def _validation(*extra: ValidationObligation) -> OperatorValidation:
    return OperatorValidation(obligations=REQUIRED_VALIDATION_OBLIGATIONS + extra)


def _declare(
    kind: OperatorKind,
    *,
    extra_preconditions: tuple[OperatorPrecondition, ...] = (),
    target_kinds: tuple[NodeKind, ...],
    expected_effects: tuple[ExpectedEffect, ...],
    authority_impact: AuthorityImpact,
    api_impact: ApiImpact,
    state_impact: StateImpact,
    phases: tuple[OperatorMigrationPhase, ...],
    adapters: tuple[str, ...] = (),
    mutates_state: bool = False,
    extra_proofs: tuple[ProofObligation, ...] = (),
    extra_validations: tuple[ValidationObligation, ...] = (),
    max_paths: int,
    max_files: int,
    max_symbols: int,
    max_packages: int,
    risk_class: OperatorRiskClass,
) -> RefactorOperator:
    preconditions = _preconditions(*extra_preconditions)
    if risk_class in AUTOMATIC_RISK_CLASSES:
        preconditions = _preconditions(
            *extra_preconditions, OperatorPrecondition.CURRENT_POLICY_VALIDATION
        )
    return RefactorOperator(
        kind=kind,
        preconditions=preconditions,
        target_kinds=target_kinds,
        expected_effects=expected_effects,
        authority_impact=authority_impact,
        api_impact=api_impact,
        state_impact=state_impact,
        migration=_migration(phases, adapters=adapters, mutates_state=mutates_state),
        rollback=OperatorRollback(),
        validation=_validation(*extra_validations),
        proofs=_proofs(*extra_proofs),
        maximum_scope=_scope(max_paths, max_files, max_symbols, max_packages),
        risk_class=risk_class,
    )


def build_initial_operator_catalog() -> OperatorCatalog:
    """Return the sealed catalog of every required initial operator."""

    operators = (
        _declare(
            OperatorKind.EXTRACT_MODULE,
            target_kinds=(NodeKind.MODULE, NodeKind.FILE, NodeKind.SYMBOL),
            expected_effects=(ExpectedEffect.EXTRACT_SYMBOLS,),
            authority_impact=AuthorityImpact.PRESERVE,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            max_paths=8,
            max_files=8,
            max_symbols=48,
            max_packages=2,
            risk_class=OperatorRiskClass.PURE_MODULE_EXTRACTION,
        ),
        _declare(
            OperatorKind.EXTRACT_INTERFACE,
            extra_preconditions=(
                OperatorPrecondition.CANONICAL_OWNER_RESOLVED,
                OperatorPrecondition.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=(NodeKind.INTERFACE, NodeKind.SYMBOL, NodeKind.SCHEMA),
            expected_effects=(ExpectedEffect.EXTRACT_INTERFACE,),
            authority_impact=AuthorityImpact.ADAPT,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("typed_service_adapter",),
            extra_proofs=(ProofObligation.TRANSLATION_EQUIVALENCE,),
            extra_validations=(ValidationObligation.TRANSLATION_VALIDATION,),
            max_paths=6,
            max_files=6,
            max_symbols=24,
            max_packages=2,
            risk_class=OperatorRiskClass.INTERNAL_ADAPTER_GENERATION,
        ),
        _declare(
            OperatorKind.EXTRACT_PURE_FUNCTION,
            target_kinds=(NodeKind.SYMBOL, NodeKind.MODULE),
            expected_effects=(ExpectedEffect.EXTRACT_PURE_FUNCTION,),
            authority_impact=AuthorityImpact.PRESERVE,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            max_paths=3,
            max_files=3,
            max_symbols=8,
            max_packages=1,
            risk_class=OperatorRiskClass.PURE_MODULE_EXTRACTION,
        ),
        _declare(
            OperatorKind.MOVE_STATE_TO_OWNER,
            extra_preconditions=(
                OperatorPrecondition.UNIQUE_STATE_OWNER,
                OperatorPrecondition.NO_DUAL_AUTHORITY,
                OperatorPrecondition.CANONICAL_OWNER_RESOLVED,
            ),
            target_kinds=(NodeKind.STATE, NodeKind.MODULE, NodeKind.SYMBOL),
            expected_effects=(ExpectedEffect.MOVE_STATE,),
            authority_impact=AuthorityImpact.PRESERVE,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.MIGRATE_TO_OWNER,
            phases=STATE_MIGRATION_PHASES,
            adapters=("state_owner_adapter",),
            mutates_state=True,
            extra_proofs=(ProofObligation.UNIQUE_STATE_OWNER,),
            max_paths=10,
            max_files=10,
            max_symbols=32,
            max_packages=2,
            risk_class=OperatorRiskClass.STATE_MIGRATION,
        ),
        _declare(
            OperatorKind.INTRODUCE_DEPENDENCY_INVERSION,
            extra_preconditions=(OperatorPrecondition.CANONICAL_OWNER_RESOLVED,),
            target_kinds=(NodeKind.INTERFACE, NodeKind.SYMBOL, NodeKind.MODULE),
            expected_effects=(ExpectedEffect.INVERT_DEPENDENCY,),
            authority_impact=AuthorityImpact.ADAPT,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("dependency_inversion_adapter",),
            max_paths=8,
            max_files=8,
            max_symbols=24,
            max_packages=2,
            risk_class=OperatorRiskClass.INTERNAL_ADAPTER_GENERATION,
        ),
        _declare(
            OperatorKind.REPLACE_DIRECT_CALL_WITH_TYPED_SERVICE,
            extra_preconditions=(OperatorPrecondition.NO_CONTRACT_AMBIGUITY,),
            target_kinds=(NodeKind.SYMBOL, NodeKind.OPERATION, NodeKind.INTERFACE),
            expected_effects=(ExpectedEffect.REPLACE_DIRECT_CALL,),
            authority_impact=AuthorityImpact.ADAPT,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("typed_service_adapter",),
            extra_proofs=(ProofObligation.TRANSLATION_EQUIVALENCE,),
            extra_validations=(ValidationObligation.TRANSLATION_VALIDATION,),
            max_paths=8,
            max_files=8,
            max_symbols=24,
            max_packages=2,
            risk_class=OperatorRiskClass.INTERNAL_ADAPTER_GENERATION,
        ),
        _declare(
            OperatorKind.GENERATE_ADAPTER,
            extra_preconditions=(OperatorPrecondition.NO_CONTRACT_AMBIGUITY,),
            target_kinds=(NodeKind.INTERFACE, NodeKind.SYMBOL, NodeKind.COMPATIBILITY),
            expected_effects=(ExpectedEffect.INTRODUCE_ADAPTER,),
            authority_impact=AuthorityImpact.ADAPT,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("typed_service_adapter",),
            extra_proofs=(ProofObligation.TRANSLATION_EQUIVALENCE,),
            extra_validations=(ValidationObligation.TRANSLATION_VALIDATION,),
            max_paths=4,
            max_files=4,
            max_symbols=12,
            max_packages=1,
            risk_class=OperatorRiskClass.INTERNAL_ADAPTER_GENERATION,
        ),
        _declare(
            OperatorKind.GENERATE_COMPATIBILITY_SHIM,
            extra_preconditions=(
                OperatorPrecondition.CANONICAL_OWNER_RESOLVED,
                OperatorPrecondition.COMPATIBILITY_SATISFACTION,
            ),
            target_kinds=(NodeKind.COMPATIBILITY, NodeKind.SYMBOL, NodeKind.INTERFACE),
            expected_effects=(ExpectedEffect.INTRODUCE_SHIM,),
            authority_impact=AuthorityImpact.ADAPT,
            api_impact=ApiImpact.COMPATIBILITY,
            state_impact=StateImpact.NONE,
            phases=DEPRECATION_MIGRATION_PHASES,
            adapters=("compatibility_shim",),
            extra_proofs=(
                ProofObligation.PUBLIC_CONTRACT_STABILITY,
                ProofObligation.TRANSLATION_EQUIVALENCE,
            ),
            extra_validations=(ValidationObligation.TRANSLATION_VALIDATION,),
            max_paths=4,
            max_files=4,
            max_symbols=12,
            max_packages=1,
            risk_class=OperatorRiskClass.PUBLIC_API_MIGRATION,
        ),
        _declare(
            OperatorKind.QUARANTINE_LEGACY_PATH,
            extra_preconditions=(OperatorPrecondition.CANONICAL_OWNER_RESOLVED,),
            target_kinds=(NodeKind.COMPATIBILITY, NodeKind.FILE, NodeKind.MODULE),
            expected_effects=(ExpectedEffect.QUARANTINE_LEGACY,),
            authority_impact=AuthorityImpact.QUARANTINE,
            api_impact=ApiImpact.COMPATIBILITY,
            state_impact=StateImpact.QUARANTINE,
            phases=DEPRECATION_MIGRATION_PHASES,
            adapters=("quarantine_adapter",),
            extra_proofs=(ProofObligation.QUARANTINE_NON_PRODUCTION,),
            max_paths=12,
            max_files=12,
            max_symbols=40,
            max_packages=2,
            risk_class=OperatorRiskClass.LEGACY_MIGRATION,
        ),
        _declare(
            OperatorKind.QUARANTINE_SIMULATION_PATH,
            extra_preconditions=(OperatorPrecondition.CANONICAL_OWNER_RESOLVED,),
            target_kinds=(NodeKind.SIMULATION, NodeKind.FILE, NodeKind.MODULE),
            expected_effects=(ExpectedEffect.QUARANTINE_SIMULATION,),
            authority_impact=AuthorityImpact.QUARANTINE,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.QUARANTINE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("quarantine_adapter",),
            extra_proofs=(ProofObligation.QUARANTINE_NON_PRODUCTION,),
            max_paths=12,
            max_files=12,
            max_symbols=40,
            max_packages=2,
            risk_class=OperatorRiskClass.SIMULATION_NAMESPACE_RELOCATION,
        ),
        _declare(
            OperatorKind.REPLACE_BOOLEAN_WITH_CLOSED_OUTCOME,
            extra_preconditions=(OperatorPrecondition.NO_CONTRACT_AMBIGUITY,),
            target_kinds=(NodeKind.SYMBOL, NodeKind.SCHEMA, NodeKind.INTERFACE),
            expected_effects=(ExpectedEffect.REPLACE_BOOLEAN,),
            authority_impact=AuthorityImpact.PRESERVE,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("closed_outcome_adapter",),
            max_paths=6,
            max_files=6,
            max_symbols=20,
            max_packages=1,
            risk_class=OperatorRiskClass.CLOSED_RESULT_TYPE_MIGRATION,
        ),
        _declare(
            OperatorKind.REPLACE_DYNAMIC_REGISTRY_WITH_TYPED_CATALOG,
            extra_preconditions=(
                OperatorPrecondition.CANONICAL_OWNER_RESOLVED,
                OperatorPrecondition.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=(NodeKind.PROVIDER, NodeKind.SCHEMA, NodeKind.OPERATION),
            expected_effects=(ExpectedEffect.REPLACE_REGISTRY,),
            authority_impact=AuthorityImpact.ADAPT,
            api_impact=ApiImpact.PUBLIC_CONTRACT,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("typed_catalog_adapter",),
            extra_proofs=(ProofObligation.PUBLIC_CONTRACT_STABILITY,),
            max_paths=8,
            max_files=8,
            max_symbols=24,
            max_packages=2,
            risk_class=OperatorRiskClass.PROVIDER_MIGRATION,
        ),
        _declare(
            OperatorKind.REPLACE_EAGER_IMPORT_WITH_LAZY_CAPABILITY,
            target_kinds=(NodeKind.MODULE, NodeKind.FILE, NodeKind.ENTRYPOINT),
            expected_effects=(ExpectedEffect.LAZY_IMPORT,),
            authority_impact=AuthorityImpact.PRESERVE,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("lazy_capability_adapter",),
            max_paths=6,
            max_files=6,
            max_symbols=16,
            max_packages=2,
            risk_class=OperatorRiskClass.LAZY_IMPORT_CONVERSION,
        ),
        _declare(
            OperatorKind.CONSOLIDATE_ERROR_VOCABULARY,
            extra_preconditions=(OperatorPrecondition.NO_CONTRACT_AMBIGUITY,),
            target_kinds=(NodeKind.SCHEMA, NodeKind.SYMBOL, NodeKind.INTERFACE),
            expected_effects=(ExpectedEffect.CONSOLIDATE_ERRORS,),
            authority_impact=AuthorityImpact.PRESERVE,
            api_impact=ApiImpact.PUBLIC_CONTRACT,
            state_impact=StateImpact.NONE,
            phases=DEPRECATION_MIGRATION_PHASES,
            adapters=("error_vocabulary_adapter",),
            extra_proofs=(ProofObligation.PUBLIC_CONTRACT_STABILITY,),
            max_paths=10,
            max_files=10,
            max_symbols=32,
            max_packages=2,
            risk_class=OperatorRiskClass.PUBLIC_API_MIGRATION,
        ),
        _declare(
            OperatorKind.CONSOLIDATE_RECEIPT_PRODUCER,
            extra_preconditions=(
                OperatorPrecondition.CANONICAL_OWNER_RESOLVED,
                OperatorPrecondition.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=(NodeKind.RECEIPT, NodeKind.SYMBOL, NodeKind.AUTHORITY),
            expected_effects=(ExpectedEffect.CONSOLIDATE_RECEIPTS,),
            authority_impact=AuthorityImpact.CONSOLIDATE,
            api_impact=ApiImpact.PUBLIC_CONTRACT,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("receipt_producer_adapter",),
            extra_proofs=(ProofObligation.PUBLIC_CONTRACT_STABILITY,),
            max_paths=8,
            max_files=8,
            max_symbols=24,
            max_packages=2,
            risk_class=OperatorRiskClass.RECEIPT_MIGRATION,
        ),
        _declare(
            OperatorKind.CONSOLIDATE_CAPABILITY_AUTHORITY,
            extra_preconditions=(
                OperatorPrecondition.CANONICAL_OWNER_RESOLVED,
                OperatorPrecondition.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=(NodeKind.AUTHORITY, NodeKind.PROVIDER, NodeKind.OPERATION),
            expected_effects=(ExpectedEffect.CONSOLIDATE_AUTHORITY,),
            authority_impact=AuthorityImpact.CONSOLIDATE,
            api_impact=ApiImpact.PUBLIC_CONTRACT,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("authority_adapter",),
            max_paths=10,
            max_files=10,
            max_symbols=32,
            max_packages=2,
            risk_class=OperatorRiskClass.AUTHORITY_CONSOLIDATION,
        ),
        _declare(
            OperatorKind.REMOVE_CONFIRMED_DEAD_CODE,
            extra_preconditions=(OperatorPrecondition.CONFIRMED_DEAD_INTERNAL,),
            target_kinds=(NodeKind.SYMBOL, NodeKind.FILE, NodeKind.MODULE),
            expected_effects=(ExpectedEffect.REMOVE_DEAD_CODE,),
            authority_impact=AuthorityImpact.PRESERVE,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            max_paths=6,
            max_files=6,
            max_symbols=24,
            max_packages=1,
            risk_class=OperatorRiskClass.CONFIRMED_DEAD_INTERNAL_CODE_REMOVAL,
        ),
        _declare(
            OperatorKind.SPLIT_MONOLITH_BY_AUTHORITY,
            extra_preconditions=(
                OperatorPrecondition.CANONICAL_OWNER_RESOLVED,
                OperatorPrecondition.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=(NodeKind.AUTHORITY, NodeKind.MODULE, NodeKind.PACKAGE),
            expected_effects=(ExpectedEffect.SPLIT_MODULE,),
            authority_impact=AuthorityImpact.CONSOLIDATE,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("authority_adapter",),
            max_paths=16,
            max_files=16,
            max_symbols=64,
            max_packages=3,
            risk_class=OperatorRiskClass.AUTHORITY_CONSOLIDATION,
        ),
        _declare(
            OperatorKind.MOVE_GENERATED_PROJECTION_OUT_OF_SOURCE_AUTHORITY,
            extra_preconditions=(OperatorPrecondition.CANONICAL_OWNER_RESOLVED,),
            target_kinds=(NodeKind.GENERATED, NodeKind.ARTIFACT, NodeKind.FILE),
            expected_effects=(ExpectedEffect.MOVE_GENERATED,),
            authority_impact=AuthorityImpact.PRESERVE,
            api_impact=ApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            phases=STRUCTURAL_MIGRATION_PHASES,
            adapters=("generated_projection_adapter",),
            extra_proofs=(ProofObligation.TRANSLATION_EQUIVALENCE,),
            extra_validations=(ValidationObligation.TRANSLATION_VALIDATION,),
            max_paths=8,
            max_files=8,
            max_symbols=24,
            max_packages=2,
            risk_class=OperatorRiskClass.GENERATED_PROJECTION_REGENERATION,
        ),
        _declare(
            OperatorKind.DEPRECATE_PUBLIC_SYMBOL,
            extra_preconditions=(
                OperatorPrecondition.CANONICAL_OWNER_RESOLVED,
                OperatorPrecondition.CONSUMER_MIGRATION,
            ),
            target_kinds=(NodeKind.SYMBOL, NodeKind.ENTRYPOINT, NodeKind.INTERFACE),
            expected_effects=(ExpectedEffect.DEPRECATE_SYMBOL,),
            authority_impact=AuthorityImpact.DEPRECATE,
            api_impact=ApiImpact.DEPRECATE,
            state_impact=StateImpact.NONE,
            phases=DEPRECATION_MIGRATION_PHASES,
            adapters=("deprecation_adapter",),
            extra_proofs=(ProofObligation.PUBLIC_CONTRACT_STABILITY,),
            max_paths=4,
            max_files=4,
            max_symbols=8,
            max_packages=1,
            risk_class=OperatorRiskClass.PUBLIC_API_MIGRATION,
        ),
        _declare(
            OperatorKind.REMOVE_DEPRECATED_SYMBOL_AFTER_GATE,
            extra_preconditions=(
                OperatorPrecondition.PRIOR_DEPRECATION,
                OperatorPrecondition.CONSUMER_MIGRATION,
                OperatorPrecondition.COMPATIBILITY_SATISFACTION,
            ),
            target_kinds=(NodeKind.SYMBOL, NodeKind.ENTRYPOINT, NodeKind.INTERFACE),
            expected_effects=(ExpectedEffect.REMOVE_DEPRECATED_SYMBOL,),
            authority_impact=AuthorityImpact.DEPRECATE,
            api_impact=ApiImpact.REMOVE_AFTER_GATE,
            state_impact=StateImpact.RETIRE,
            phases=DEPRECATION_MIGRATION_PHASES,
            adapters=("deprecation_adapter",),
            extra_proofs=(ProofObligation.PUBLIC_CONTRACT_STABILITY,),
            max_paths=4,
            max_files=4,
            max_symbols=8,
            max_packages=1,
            risk_class=OperatorRiskClass.PUBLIC_API_MIGRATION,
        ),
    )
    return OperatorCatalog(operators=operators)


OPERATOR_CATALOG = build_initial_operator_catalog()


def get_operator(kind: OperatorKind | str) -> RefactorOperator:
    """Return one catalog operator or reject an unknown kind."""

    return OPERATOR_CATALOG.operator(kind)


def autonomy_classification_map() -> dict[str, str]:
    """Return operator-kind to autonomy-ceiling mapping."""

    return OPERATOR_CATALOG.autonomy_classification_map()


def risk_classification_map() -> dict[str, str]:
    """Return operator-kind to risk-class mapping."""

    return OPERATOR_CATALOG.risk_classification_map()


def effects_identity(effects: Sequence[ExpectedEffect | str]) -> str:
    """Canonical identity of a closed expected-effect tuple."""

    closed = _require_closed_tuple(effects, ExpectedEffect, "expected effect")
    return _content_identity({"effects": [item.value for item in closed]})


def assert_repository_relative_scope_path(path: str) -> str:
    """Reject absolute, escaping, sibling, protected, and script-bearing paths."""

    relative = _repository_relative_path(
        path, "scope path", error_type=RefactorOperatorError
    )
    _reject_script_text(relative, "scope path")
    if relative in PROTECTED_PATHS:
        raise RefactorOperatorError("operator scope cannot write protected paths")
    if any(relative == prefix[:-1] or relative.startswith(prefix) for prefix in _SIBLING_PREFIXES):
        raise RefactorOperatorError("operator scope cannot write sibling repositories")
    return relative


def assert_scope_within_maximum(
    *,
    paths: Sequence[str],
    symbol_count: int,
    package_count: int,
    maximum: MaximumScope,
) -> tuple[str, ...]:
    """Fail closed when a candidate exceeds the operator's maximum scope."""

    normalized = tuple(assert_repository_relative_scope_path(path) for path in paths)
    unique = tuple(sorted(set(normalized)))
    if len(unique) != len(normalized):
        raise RefactorOperatorError("scope paths must be unique")
    file_count = len(unique)
    if not maximum.contains(
        path_count=file_count,
        file_count=file_count,
        symbol_count=symbol_count,
        package_count=package_count,
    ):
        refuse_scope_expansion("expand")
    return unique


__all__ = [
    "ARBITRARY_EXECUTABLE_POLICY_PROHIBITED",
    "AUTOMATIC_RISK_CLASSES",
    "CANDIDATE_MAY_RAISE_CEILING",
    "CLOSED_ADAPTERS",
    "CLOSED_API_IMPACTS",
    "CLOSED_AUTHORITY_IMPACTS",
    "CLOSED_AUTONOMY_CEILINGS",
    "CLOSED_EXPECTED_EFFECTS",
    "CLOSED_MIGRATION_PHASES",
    "CLOSED_OPERATORS",
    "CLOSED_PRECONDITIONS",
    "CLOSED_PROHIBITED_EFFECTS",
    "CLOSED_PROOF_OBLIGATIONS",
    "CLOSED_RISK_CLASSES",
    "CLOSED_ROLLBACK_ACTIONS",
    "CLOSED_STATE_IMPACTS",
    "CLOSED_VALIDATION_OBLIGATIONS",
    "DEFAULT_FRESHNESS",
    "DEPRECATION_MIGRATION_PHASES",
    "EFFECT_CLASS",
    "EXTRACTOR_IDENTITY",
    "HUMAN_APPROVAL_RISK_CLASSES",
    "INCOMPLETE_DECLARATION_REJECTED",
    "INITIAL_OPERATORS",
    "OPERATOR_CAN_ADMIT_SCRIPT_PAYLOAD",
    "OPERATOR_CAN_AUTHORIZE_EXECUTION",
    "OPERATOR_CAN_EXPAND_SCOPE",
    "OPERATOR_CAN_RAISE_AUTONOMY_CEILING",
    "OPERATOR_CAN_REDUCE_GATES",
    "OPERATOR_CAN_SELF_PROMOTE",
    "OPERATOR_CATALOG",
    "OPERATOR_CATALOG_SCHEMA",
    "OPERATOR_CATALOG_VERSION",
    "OPERATOR_MIGRATION_SCHEMA",
    "OPERATOR_MIGRATION_VERSION",
    "OPERATOR_ROLLBACK_SCHEMA",
    "OPERATOR_ROLLBACK_VERSION",
    "OPERATOR_SCOPE_SCHEMA",
    "OPERATOR_SCOPE_VERSION",
    "OPERATOR_VALIDATION_SCHEMA",
    "OPERATOR_VALIDATION_VERSION",
    "PROPOSAL_ONLY_RISK_CLASSES",
    "PROTECTED_PATHS",
    "REFACTOR_OPERATOR_EVIDENCE",
    "REFACTOR_OPERATOR_SCHEMA",
    "REFACTOR_OPERATOR_VERSION",
    "REQUIRED_OPERATORS",
    "REQUIRED_OPERATOR_DECLARATION_FIELDS",
    "REQUIRED_PRECONDITIONS",
    "REQUIRED_PROHIBITED_EFFECTS",
    "REQUIRED_PROOF_OBLIGATIONS",
    "REQUIRED_VALIDATION_OBLIGATIONS",
    "STATE_MIGRATION_PHASES",
    "STRUCTURAL_MIGRATION_PHASES",
    "TASK_ID",
    "UNKNOWN_OPERATOR_REJECTED",
    "ApiImpact",
    "AuthorityImpact",
    "AutonomyCeiling",
    "ExpectedEffect",
    "MaximumScope",
    "OperatorCatalog",
    "OperatorKind",
    "OperatorMigration",
    "OperatorMigrationPhase",
    "OperatorPrecondition",
    "OperatorRiskClass",
    "OperatorRollback",
    "OperatorValidation",
    "ProhibitedEffect",
    "ProofObligation",
    "RefactorOperator",
    "RefactorOperatorAuthorityError",
    "RefactorOperatorError",
    "RollbackAction",
    "StateImpact",
    "ValidationObligation",
    "assert_repository_relative_scope_path",
    "assert_scope_within_maximum",
    "autonomy_ceiling_for_risk",
    "autonomy_classification_map",
    "build_initial_operator_catalog",
    "effects_identity",
    "get_operator",
    "refuse_autonomy_ceiling_raise",
    "refuse_gate_reduction",
    "refuse_scope_expansion",
    "refuse_script_payload",
    "refuse_self_authorization",
    "refuse_self_promotion",
    "risk_classification_map",
]
