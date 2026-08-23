"""Closed declarative refactor-operator grammar (PCAR-012).

`RefactorOperator` is the only admitted change vocabulary. Every initial
operator declares preconditions, target kinds, expected effects,
authority/API/state impact, migration, rollback, validation, proof
obligations, maximum scope, and a risk/autonomy disposition. Unknown
operators and fields fail closed. Declarations cannot authorize execution,
reduce gates, expand scope, or promote themselves. Arbitrary scripts and
open operator kinds are not admitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
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
from .entropy import NON_COMPENSABLE_INVARIANTS

REFACTOR_OPERATOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refactor-operator@1"
)
REFACTOR_OPERATOR_VERSION = 1
REFACTOR_OPERATOR_EVIDENCE = "pcar/refactor-operator@1"
OPERATOR_CATALOG_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refactor-operator-catalog@1"
)
OPERATOR_CATALOG_VERSION = 1
OPERATOR_MIGRATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refactor-operator-migration@1"
)
OPERATOR_MIGRATION_VERSION = 1
OPERATOR_ROLLBACK_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refactor-operator-rollback@1"
)
OPERATOR_ROLLBACK_VERSION = 1
OPERATOR_SCOPE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/refactor-operator-maximum-scope@1"
)
OPERATOR_SCOPE_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-012-refactor-operator-grammar"
TASK_ID = "PCAR-012"
DEFAULT_FRESHNESS = "pcar-012-operator-grammar"
EFFECT_CLASS = "internal_pure_contract_addition"
OPERATOR_CAN_AUTHORIZE_EXECUTION = False
OPERATOR_CAN_REDUCE_GATES = False
OPERATOR_CAN_SELF_PROMOTE = False
OPERATOR_CAN_RAISE_CEILING = False
OPERATOR_CAN_EXPAND_SCOPE = False
OPERATOR_CAN_ADMIT_ARBITRARY_PAYLOADS = False
DECLARATIONS_ARE_IMMUTABLE = True
UNKNOWN_OPERATORS_FAIL_CLOSED = True
INCOMPLETE_DECLARATIONS_FAIL_CLOSED = True
SCOPE_EXPANSION_FAILS_CLOSED = True

_UNKNOWN_FIELD_MESSAGE = "unknown refactor-operator field"
_MISSING_FIELD_MESSAGE = "missing refactor-operator field"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")
_SIBLING_PREFIXES = (
    "ipfs_datasets_py/",
    "ipfs_kit_py/",
    "ipfs_accelerate_py/mcplusplus/",
)
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
_DEFAULT_PATH_PREFIXES = ("ipfs_accelerate_py/",)


class RefactorOperatorError(ArchitectureContractError):
    """Fail-closed refactor-operator grammar error."""


class RefactorOperatorAuthorityError(RefactorOperatorError):
    """Raised when an operator is asked to authorize, promote, or expand."""


class OperatorKind(str, Enum):
    """Closed initial refactor-operator vocabulary (PCAR-PLAN-R1)."""

    EXTRACT_MODULE = "extract_module"
    EXTRACT_INTERFACE = "extract_interface"
    EXTRACT_PURE_FUNCTION = "extract_pure_function"
    MOVE_STATE_TO_OWNER = "move_state_to_owner"
    INTRODUCE_DEPENDENCY_INVERSION = "introduce_dependency_inversion"
    REPLACE_DIRECT_CALL_WITH_TYPED_SERVICE = "replace_direct_call_with_typed_service"
    GENERATE_ADAPTER = "generate_adapter"
    GENERATE_COMPATIBILITY_SHIM = "generate_compatibility_shim"
    QUARANTINE_LEGACY_PATH = "quarantine_legacy_path"
    QUARANTINE_SIMULATION_PATH = "quarantine_simulation_path"
    REPLACE_BOOLEAN_WITH_CLOSED_OUTCOME = "replace_boolean_with_closed_outcome"
    REPLACE_DYNAMIC_REGISTRY_WITH_TYPED_CATALOG = (
        "replace_dynamic_registry_with_typed_catalog"
    )
    REPLACE_EAGER_IMPORT_WITH_LAZY_CAPABILITY = (
        "replace_eager_import_with_lazy_capability"
    )
    CONSOLIDATE_ERROR_VOCABULARY = "consolidate_error_vocabulary"
    CONSOLIDATE_RECEIPT_PRODUCER = "consolidate_receipt_producer"
    CONSOLIDATE_CAPABILITY_AUTHORITY = "consolidate_capability_authority"
    REMOVE_CONFIRMED_DEAD_CODE = "remove_confirmed_dead_code"
    SPLIT_MONOLITH_BY_AUTHORITY = "split_monolith_by_authority"
    MOVE_GENERATED_PROJECTION_OUT_OF_SOURCE_AUTHORITY = (
        "move_generated_projection_out_of_source_authority"
    )
    DEPRECATE_PUBLIC_SYMBOL = "deprecate_public_symbol"
    REMOVE_DEPRECATED_SYMBOL_AFTER_GATE = "remove_deprecated_symbol_after_gate"


INITIAL_OPERATORS: tuple[OperatorKind, ...] = tuple(OperatorKind)
REQUIRED_OPERATORS: tuple[OperatorKind, ...] = INITIAL_OPERATORS
CLOSED_OPERATOR_KINDS: frozenset[str] = frozenset(item.value for item in OperatorKind)


class AutonomyDisposition(str, Enum):
    """Closed autonomy ceiling vocabulary. Candidates cannot raise this."""

    AUTOMATIC = "automatic"
    PROPOSAL_ONLY = "proposal_only"
    ALWAYS_HUMAN = "always_human"


CLOSED_AUTONOMY_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in AutonomyDisposition
)
_DISPOSITION_RANK: dict[AutonomyDisposition, int] = {
    AutonomyDisposition.AUTOMATIC: 0,
    AutonomyDisposition.PROPOSAL_ONLY: 1,
    AutonomyDisposition.ALWAYS_HUMAN: 2,
}


class AutonomyRiskClass(str, Enum):
    """Closed autonomy risk-class vocabulary, including scheduler allowlists."""

    PURE_MODULE_EXTRACTION = "pure_module_extraction"
    GENERATED_PROJECTION_REGENERATION = "generated_projection_regeneration"
    LAZY_IMPORT_CONVERSION = "lazy_import_conversion"
    INTERNAL_ADAPTER_GENERATION = "internal_adapter_generation"
    CLOSED_RESULT_TYPE_MIGRATION = "closed_result_type_migration"
    CONFIRMED_DEAD_INTERNAL_CODE_REMOVAL = "confirmed_dead_internal_code_removal"
    TEST_FIXTURE_RELOCATION = "test_fixture_relocation"
    SIMULATION_NAMESPACE_RELOCATION = "simulation_namespace_relocation"
    PUBLIC_API = "public_api"
    STATE = "state"
    PROVIDER = "provider"
    RECEIPT = "receipt"
    LEGACY = "legacy"
    MUTABLE_STATE = "mutable_state"
    CROSS_PACKAGE_MIGRATION = "cross_package_migration"
    AUTHORIZATION = "authorization"
    POLICY = "policy"
    SECURITY = "security"
    PAYMENT = "payment"
    WIRE_PROTOCOL = "wire_protocol"
    KEY = "key"
    RELEASE_AUTHORITY = "release_authority"
    LEGAL = "legal"
    FINANCIAL = "financial"


CLOSED_AUTONOMY_RISK_CLASSES: frozenset[str] = frozenset(
    item.value for item in AutonomyRiskClass
)
AUTOMATIC_OPERATOR_CLASSES: tuple[AutonomyRiskClass, ...] = (
    AutonomyRiskClass.PURE_MODULE_EXTRACTION,
    AutonomyRiskClass.GENERATED_PROJECTION_REGENERATION,
    AutonomyRiskClass.LAZY_IMPORT_CONVERSION,
    AutonomyRiskClass.INTERNAL_ADAPTER_GENERATION,
    AutonomyRiskClass.CLOSED_RESULT_TYPE_MIGRATION,
    AutonomyRiskClass.CONFIRMED_DEAD_INTERNAL_CODE_REMOVAL,
    AutonomyRiskClass.TEST_FIXTURE_RELOCATION,
    AutonomyRiskClass.SIMULATION_NAMESPACE_RELOCATION,
)
PROPOSAL_ONLY_RISK_CLASSES: tuple[AutonomyRiskClass, ...] = (
    AutonomyRiskClass.PUBLIC_API,
    AutonomyRiskClass.STATE,
    AutonomyRiskClass.PROVIDER,
    AutonomyRiskClass.RECEIPT,
    AutonomyRiskClass.LEGACY,
    AutonomyRiskClass.MUTABLE_STATE,
    AutonomyRiskClass.CROSS_PACKAGE_MIGRATION,
)
ALWAYS_HUMAN_RISK_CLASSES: tuple[AutonomyRiskClass, ...] = (
    AutonomyRiskClass.AUTHORIZATION,
    AutonomyRiskClass.POLICY,
    AutonomyRiskClass.SECURITY,
    AutonomyRiskClass.PAYMENT,
    AutonomyRiskClass.WIRE_PROTOCOL,
    AutonomyRiskClass.KEY,
    AutonomyRiskClass.RELEASE_AUTHORITY,
    AutonomyRiskClass.LEGAL,
    AutonomyRiskClass.FINANCIAL,
)
CLOSED_AUTOMATIC_OPERATOR_CLASSES: frozenset[str] = frozenset(
    item.value for item in AUTOMATIC_OPERATOR_CLASSES
)
CLOSED_PROPOSAL_ONLY_RISK_CLASSES: frozenset[str] = frozenset(
    item.value for item in PROPOSAL_ONLY_RISK_CLASSES
)
CLOSED_ALWAYS_HUMAN_RISK_CLASSES: frozenset[str] = frozenset(
    item.value for item in ALWAYS_HUMAN_RISK_CLASSES
)


class AuthorityImpact(str, Enum):
    """Closed authority-impact vocabulary. Transfer is not an admitted impact."""

    NONE = "none"
    PRESERVE = "preserve"
    ADAPTER = "adapter"
    QUARANTINE = "quarantine"
    CONSOLIDATE = "consolidate"


CLOSED_AUTHORITY_IMPACTS: frozenset[str] = frozenset(
    item.value for item in AuthorityImpact
)


class PublicApiImpact(str, Enum):
    """Closed public-API impact vocabulary."""

    INTERNAL = "internal"
    COMPATIBILITY = "compatibility"
    VERSIONED_MIGRATION = "versioned_migration"
    DEPRECATE = "deprecate"
    REMOVE_AFTER_GATE = "remove_after_gate"


CLOSED_PUBLIC_API_IMPACTS: frozenset[str] = frozenset(
    item.value for item in PublicApiImpact
)


class StateImpact(str, Enum):
    """Closed state-impact vocabulary. Indefinite dual authority is rejected."""

    NONE = "none"
    READ_ONLY = "read_only"
    MOVE_TO_OWNER = "move_to_owner"
    NO_INDEFINITE_DUAL_AUTHORITY = "no_indefinite_dual_authority"


CLOSED_STATE_IMPACTS: frozenset[str] = frozenset(item.value for item in StateImpact)


class ExpectedEffectKind(str, Enum):
    """Closed expected-effect vocabulary for declarative operators."""

    EXTRACT = "extract"
    MOVE = "move"
    ADAPT = "adapt"
    GENERATE = "generate"
    QUARANTINE = "quarantine"
    REPLACE = "replace"
    CONSOLIDATE = "consolidate"
    REMOVE = "remove"
    SPLIT = "split"
    DEPRECATE = "deprecate"


CLOSED_EXPECTED_EFFECTS: frozenset[str] = frozenset(
    item.value for item in ExpectedEffectKind
)


class PreconditionKind(str, Enum):
    """Closed operator-precondition vocabulary."""

    ACCEPTED_OWNERSHIP = "accepted_ownership"
    ACCEPTED_CONTRACTS = "accepted_contracts"
    ACCEPTED_BOUNDARIES = "accepted_boundaries"
    ROLLBACK_DECLARED = "rollback_declared"
    VALIDATION_DECLARED = "validation_declared"
    PROOF_OBLIGATIONS_DECLARED = "proof_obligations_declared"
    MAXIMUM_SCOPE_DECLARED = "maximum_scope_declared"
    AUTONOMY_DISPOSITION_DECLARED = "autonomy_disposition_declared"
    CANONICAL_OWNER = "canonical_owner"
    UNIQUE_STATE_OWNER = "unique_state_owner"
    NO_CONTRACT_AMBIGUITY = "no_contract_ambiguity"
    PURE_INTERNAL = "pure_internal"
    CONFIRMED_DEAD = "confirmed_dead"
    NO_DYNAMIC_LOADING_UNCERTAINTY = "no_dynamic_loading_uncertainty"
    PRIOR_DEPRECATION = "prior_deprecation"
    CONSUMER_MIGRATION = "consumer_migration"
    COMPATIBILITY_SATISFACTION = "compatibility_satisfaction"
    ISOLATED_WORKTREE = "isolated_worktree"
    NO_INDEFINITE_DUAL_AUTHORITY = "no_indefinite_dual_authority"


CLOSED_PRECONDITIONS: frozenset[str] = frozenset(item.value for item in PreconditionKind)
REQUIRED_PRECONDITIONS: tuple[PreconditionKind, ...] = (
    PreconditionKind.ACCEPTED_OWNERSHIP,
    PreconditionKind.ACCEPTED_CONTRACTS,
    PreconditionKind.ACCEPTED_BOUNDARIES,
    PreconditionKind.ROLLBACK_DECLARED,
    PreconditionKind.VALIDATION_DECLARED,
    PreconditionKind.PROOF_OBLIGATIONS_DECLARED,
    PreconditionKind.MAXIMUM_SCOPE_DECLARED,
    PreconditionKind.AUTONOMY_DISPOSITION_DECLARED,
)


class ValidationObligationKind(str, Enum):
    """Closed validation-obligation vocabulary required of every operator."""

    STATIC_TYPE_CHECKS = "static_type_checks"
    DIFFERENTIAL_BEHAVIOR = "differential_behavior"
    EFFECT_COMPARISON = "effect_comparison"
    AUTHORITY_COMPARISON = "authority_comparison"
    SELECTED_TESTS = "selected_tests"
    SELECTED_PROOFS = "selected_proofs"
    NO_EFFECT_EXPANSION = "no_effect_expansion"
    NO_AUTHORITY_WEAKENING = "no_authority_weakening"
    ROLLBACK_REHEARSAL = "rollback_rehearsal"
    SCOPE_BOUND = "scope_bound"
    PUBLIC_CONTRACT = "public_contract"
    STATE_OWNERSHIP = "state_ownership"
    QUARANTINE_REACHABILITY = "quarantine_reachability"


CLOSED_VALIDATION_OBLIGATIONS: frozenset[str] = frozenset(
    item.value for item in ValidationObligationKind
)
REQUIRED_VALIDATION_OBLIGATIONS: tuple[ValidationObligationKind, ...] = (
    ValidationObligationKind.STATIC_TYPE_CHECKS,
    ValidationObligationKind.DIFFERENTIAL_BEHAVIOR,
    ValidationObligationKind.EFFECT_COMPARISON,
    ValidationObligationKind.AUTHORITY_COMPARISON,
    ValidationObligationKind.SELECTED_TESTS,
    ValidationObligationKind.SELECTED_PROOFS,
    ValidationObligationKind.NO_EFFECT_EXPANSION,
    ValidationObligationKind.NO_AUTHORITY_WEAKENING,
    ValidationObligationKind.ROLLBACK_REHEARSAL,
    ValidationObligationKind.SCOPE_BOUND,
)


class ProofObligationKind(str, Enum):
    """Closed proof-obligation vocabulary retained with every operator."""

    FRAME_CONDITIONS = "frame_conditions"
    NO_AUTHORITY_WEAKENING = "no_authority_weakening"
    NO_EFFECT_EXPANSION = "no_effect_expansion"
    BEHAVIOR_EQUIVALENCE = "behavior_equivalence"
    REFINEMENT = "refinement"
    QUARANTINE_FLOW = "quarantine_flow"
    DEAD_CODE_CONFIRMATION = "dead_code_confirmation"
    CONSUMER_ABSENCE = "consumer_absence"
    UNIQUE_STATE_OWNER = "unique_state_owner"
    NO_INDEFINITE_DUAL_AUTHORITY = "no_indefinite_dual_authority"
    COMPATIBILITY_SATISFACTION = "compatibility_satisfaction"
    CANONICAL_OWNER_PRESERVED = "canonical_owner_preserved"
    ROLLBACK_RESTORATION = "rollback_restoration"
    NO_SELF_AUTHORIZATION = "no_self_authorization"
    NO_SELF_PROMOTION = "no_self_promotion"
    SCOPE_ADHERENCE = "scope_adherence"


CLOSED_PROOF_OBLIGATIONS: frozenset[str] = frozenset(
    item.value for item in ProofObligationKind
)
REQUIRED_PROOF_OBLIGATIONS: tuple[ProofObligationKind, ...] = (
    ProofObligationKind.FRAME_CONDITIONS,
    ProofObligationKind.NO_AUTHORITY_WEAKENING,
    ProofObligationKind.NO_EFFECT_EXPANSION,
    ProofObligationKind.CANONICAL_OWNER_PRESERVED,
    ProofObligationKind.ROLLBACK_RESTORATION,
    ProofObligationKind.NO_SELF_AUTHORIZATION,
    ProofObligationKind.NO_SELF_PROMOTION,
    ProofObligationKind.SCOPE_ADHERENCE,
)


class MigrationPhase(str, Enum):
    """Closed migration-phase vocabulary. Dual authority cannot remain open."""

    DECLARE = "declare"
    SNAPSHOT = "snapshot"
    EXTRACT = "extract"
    ADAPT_CALLERS = "adapt_callers"
    GENERATE_ADAPTER = "generate_adapter"
    SHADOW_COMPARE = "shadow_compare"
    BOUNDED_DUAL_WRITE = "bounded_dual_write"
    CUTOVER = "cutover"
    DEPRECATE = "deprecate"
    QUARANTINE = "quarantine"
    VALIDATE_AND_SEAL = "validate_and_seal"
    READ_ONLY_LEGACY = "read_only_legacy"
    RETIRE = "retire"


CLOSED_MIGRATION_PHASES: frozenset[str] = frozenset(
    item.value for item in MigrationPhase
)


class RollbackAction(str, Enum):
    """Closed rollback-action vocabulary. Declarations have no applied effects."""

    REVERT_UNAPPLIED_CANDIDATE = "revert_unapplied_candidate"
    RESTORE_SEALED_TREE = "restore_sealed_tree"
    CANCEL_AND_RESTORE = "cancel_and_restore"


CLOSED_ROLLBACK_ACTIONS: frozenset[str] = frozenset(
    item.value for item in RollbackAction
)

_MIGRATION_FIELDS = frozenset(
    {
        "adapters",
        "content_identity",
        "deprecated_paths",
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
        "applied_effects",
        "content_identity",
        "message",
        "restores_tree",
        "schema",
        "version",
    }
)
_SCOPE_FIELDS = frozenset(
    {
        "allows_arbitrary_payloads",
        "allows_cross_package",
        "allows_network",
        "allows_public_surface",
        "allows_scope_expansion",
        "allows_sibling_repositories",
        "allows_state_stores",
        "content_identity",
        "max_paths",
        "max_symbols",
        "path_prefixes",
        "schema",
        "target_kinds",
        "version",
    }
)
_OPERATOR_FIELDS = frozenset(
    {
        "authority_impact",
        "autonomy_disposition",
        "can_authorize_execution",
        "can_raise_ceiling",
        "can_reduce_gates",
        "can_self_promote",
        "content_identity",
        "expected_effects",
        "kind",
        "maximum_scope",
        "migration",
        "preconditions",
        "preserved_invariants",
        "proofs",
        "public_api_impact",
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
        "autonomy_classification",
        "can_authorize_execution",
        "can_raise_ceiling",
        "can_reduce_gates",
        "can_self_promote",
        "content_identity",
        "covers_initial_operators",
        "declarations_are_immutable",
        "effect_class",
        "operators",
        "schema",
        "version",
    }
)


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str) -> str:
    try:
        return validate_cid(value, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise RefactorOperatorError("content identity must be a dag-json CIDv1") from exc


def _reject_unknown_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = set(payload) - set(allowed)
    forbidden = sorted(extra & _FORBIDDEN_PAYLOAD_FIELDS)
    if forbidden:
        refuse_arbitrary_payload(",".join(forbidden))
    leftover = sorted(extra)
    if leftover:
        raise RefactorOperatorError(f"{_UNKNOWN_FIELD_MESSAGE}: {leftover}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown_fields(payload, allowed_fields)
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


def _require_enum_tuple(
    value: Any,
    enum_type: type[Enum],
    name: str,
    *,
    ordered: bool = False,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RefactorOperatorError(f"{name} must be a list of strings")
    items = tuple(
        _closed_enum(item, enum_type, name, error_type=RefactorOperatorError)
        for item in value
    )
    if len(items) != len(set(items)):
        raise RefactorOperatorError(f"{name} must be unique")
    if not items:
        raise RefactorOperatorError(f"{name} must be nonempty")
    if ordered:
        return items
    return tuple(sorted(items, key=lambda item: item.value))


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RefactorOperatorError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=RefactorOperatorError)
        for item in value
    )
    if len(items) != len(set(items)):
        raise RefactorOperatorError(f"{name} must be unique")
    return tuple(sorted(set(items)))


def _is_sibling_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in _SIBLING_PREFIXES
    )


def _require_path_prefix(value: Any, name: str) -> str:
    text = _repository_relative_path(value, name, error_type=RefactorOperatorError)
    if _is_sibling_path(text):
        raise RefactorOperatorError(f"{name} cannot target a sibling repository")
    return text


def path_in_scope_prefix(path: str, prefix: str) -> bool:
    """Return whether a repository-relative path is inside one declared prefix."""

    if prefix.endswith("/"):
        return path == prefix[:-1] or path.startswith(prefix)
    return path == prefix or path.startswith(prefix + "/")


def disposition_for_risk(risk: AutonomyRiskClass) -> AutonomyDisposition:
    """Map a closed risk class onto its autonomy disposition."""

    if risk in AUTOMATIC_OPERATOR_CLASSES:
        return AutonomyDisposition.AUTOMATIC
    if risk in PROPOSAL_ONLY_RISK_CLASSES:
        return AutonomyDisposition.PROPOSAL_ONLY
    if risk in ALWAYS_HUMAN_RISK_CLASSES:
        return AutonomyDisposition.ALWAYS_HUMAN
    raise RefactorOperatorError(f"unsupported autonomy risk class: {risk!r}")


def autonomy_rank(disposition: AutonomyDisposition) -> int:
    """Return the closed restriction rank. Higher ranks are more restrictive."""

    return _DISPOSITION_RANK[disposition]


def refuse_self_authorization(action: str = "authorize") -> None:
    """Reject attempts to treat an operator declaration as execution authority."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} execution"
    )


def refuse_gate_reduction(action: str = "reduce") -> None:
    """Reject attempts to drop validation, proof, or safety gates."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} gates"
    )


def refuse_self_promotion(action: str = "promote") -> None:
    """Reject attempts for an operator or candidate to promote itself."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} itself"
    )


def refuse_ceiling_raise(action: str = "raise") -> None:
    """Reject attempts to raise the autonomy ceiling."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} its autonomy ceiling"
    )


def refuse_scope_expansion(action: str = "expand") -> None:
    """Reject attempts to exceed a declared maximum scope."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot {name} maximum scope"
    )


def refuse_arbitrary_payload(action: str = "script") -> None:
    """Reject arbitrary executable refactor payloads."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator cannot admit arbitrary {name} payloads"
    )


def refuse_unknown_operator(kind: Any) -> None:
    """Reject operator kinds outside the closed initial vocabulary."""

    raise RefactorOperatorError(f"unsupported refactor-operator kind: {kind!r}")


def refuse_execution(action: str = "execute") -> None:
    """Reject attempts to apply a declaration."""

    name = _require_text(action, "action", error_type=RefactorOperatorError)
    raise RefactorOperatorAuthorityError(
        f"refactor operator declarations cannot {name}"
    )


def _bind_identity(record: Any, claimed: str, name: str) -> str:
    identity = _content_identity(record._identity_payload())
    if claimed:
        resolved = _validate_dag_json_cid(
            _require_text(claimed, "content_identity", error_type=RefactorOperatorError)
        )
        if resolved != identity:
            raise RefactorOperatorError(f"{name} content identity mismatch")
    return identity


def _require_false_authority_flag(value: Any, name: str, refuse) -> bool:
    flag = _require_bool(value, name)
    if flag:
        refuse()
    return False


@dataclass(frozen=True)
class OperatorMigration:
    """Named migration plan. The grammar itself performs no mutation."""

    phases: tuple[MigrationPhase, ...]
    adapters: tuple[str, ...] = ()
    deprecated_paths: tuple[str, ...] = ()
    mutates_state: bool = False
    transfers_authority: bool = False
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
        phases = _require_enum_tuple(
            self.phases, MigrationPhase, "phases", ordered=True
        )
        if MigrationPhase.VALIDATE_AND_SEAL not in phases:
            raise RefactorOperatorError("migration must include validate_and_seal")
        if phases[-1] is not MigrationPhase.VALIDATE_AND_SEAL:
            raise RefactorOperatorError("validate_and_seal must be the final migration phase")
        transfers = _require_bool(self.transfers_authority, "transfers_authority")
        if transfers:
            raise RefactorOperatorAuthorityError(
                "refactor operators cannot transfer authority"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "adapters", _require_text_tuple(self.adapters, "adapters"))
        object.__setattr__(
            self,
            "deprecated_paths",
            tuple(
                _require_path_prefix(item, "deprecated_paths item")
                for item in _require_text_tuple(self.deprecated_paths, "deprecated_paths")
            ),
        )
        object.__setattr__(
            self, "mutates_state", _require_bool(self.mutates_state, "mutates_state")
        )
        object.__setattr__(self, "transfers_authority", False)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "operator-migration"),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "adapters": list(self.adapters),
            "deprecated_paths": list(self.deprecated_paths),
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
            deprecated_paths=mapping["deprecated_paths"],
            mutates_state=mapping["mutates_state"],
            transfers_authority=mapping["transfers_authority"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorOperatorError("operator-migration content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class OperatorRollback:
    """Exact rollback declared before any candidate is executed."""

    action: RollbackAction = RollbackAction.RESTORE_SEALED_TREE
    message: str = "revert the unapplied candidate; restore the sealed tree"
    applied_effects: bool = False
    restores_tree: bool = True
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
        action = _closed_enum(
            self.action, RollbackAction, "rollback action", error_type=RefactorOperatorError
        )
        message = _require_text(self.message, "message", error_type=RefactorOperatorError)
        applied = _require_bool(self.applied_effects, "applied_effects")
        restores = _require_bool(self.restores_tree, "restores_tree")
        if applied:
            raise RefactorOperatorError("operator declarations perform no applied effects")
        if restores is not True:
            raise RefactorOperatorError("operator rollback must restore the sealed tree")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "applied_effects", False)
        object.__setattr__(self, "restores_tree", True)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "operator-rollback"),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "applied_effects": False,
            "message": self.message,
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
            applied_effects=mapping["applied_effects"],
            restores_tree=mapping["restores_tree"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorOperatorError("operator-rollback content identity mismatch")
        return record

    from_dict = from_mapping


@dataclass(frozen=True)
class MaximumScope:
    """Hard path, kind, and surface bound. Expansion fails closed."""

    path_prefixes: tuple[str, ...]
    target_kinds: tuple[NodeKind, ...]
    max_paths: int
    max_symbols: int
    allows_public_surface: bool = False
    allows_state_stores: bool = False
    allows_cross_package: bool = False
    allows_sibling_repositories: bool = False
    allows_arbitrary_payloads: bool = False
    allows_scope_expansion: bool = False
    allows_network: bool = False
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
        prefixes = tuple(
            _require_path_prefix(item, "path_prefixes item")
            for item in _require_text_tuple(self.path_prefixes, "path_prefixes")
        )
        if not prefixes:
            raise RefactorOperatorError("path_prefixes must be nonempty")
        kinds = _require_enum_tuple(self.target_kinds, NodeKind, "target_kinds")
        if _require_bool(self.allows_sibling_repositories, "allows_sibling_repositories"):
            raise RefactorOperatorAuthorityError(
                "refactor operators cannot write sibling repositories"
            )
        if _require_bool(self.allows_arbitrary_payloads, "allows_arbitrary_payloads"):
            refuse_arbitrary_payload("payload")
        if _require_bool(self.allows_scope_expansion, "allows_scope_expansion"):
            refuse_scope_expansion("expand")
        if _require_bool(self.allows_network, "allows_network"):
            raise RefactorOperatorAuthorityError(
                "refactor operators cannot admit network effects"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "path_prefixes", prefixes)
        object.__setattr__(self, "target_kinds", kinds)
        object.__setattr__(self, "max_paths", _require_positive_int(self.max_paths, "max_paths"))
        object.__setattr__(
            self, "max_symbols", _require_positive_int(self.max_symbols, "max_symbols")
        )
        object.__setattr__(
            self,
            "allows_public_surface",
            _require_bool(self.allows_public_surface, "allows_public_surface"),
        )
        object.__setattr__(
            self,
            "allows_state_stores",
            _require_bool(self.allows_state_stores, "allows_state_stores"),
        )
        object.__setattr__(
            self,
            "allows_cross_package",
            _require_bool(self.allows_cross_package, "allows_cross_package"),
        )
        object.__setattr__(self, "allows_sibling_repositories", False)
        object.__setattr__(self, "allows_arbitrary_payloads", False)
        object.__setattr__(self, "allows_scope_expansion", False)
        object.__setattr__(self, "allows_network", False)
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "operator-scope"),
        )

    def covers_path(self, path: str) -> bool:
        normalized = _require_path_prefix(path, "path")
        return any(path_in_scope_prefix(normalized, prefix) for prefix in self.path_prefixes)

    def contains(self, other: "MaximumScope") -> bool:
        """Return whether ``other`` is a non-expanding subset of this scope."""

        if other.max_paths > self.max_paths or other.max_symbols > self.max_symbols:
            return False
        if other.allows_public_surface and not self.allows_public_surface:
            return False
        if other.allows_state_stores and not self.allows_state_stores:
            return False
        if other.allows_cross_package and not self.allows_cross_package:
            return False
        if set(other.target_kinds) - set(self.target_kinds):
            return False
        return all(
            any(path_in_scope_prefix(prefix, parent) for parent in self.path_prefixes)
            for prefix in other.path_prefixes
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "allows_arbitrary_payloads": False,
            "allows_cross_package": self.allows_cross_package,
            "allows_network": False,
            "allows_public_surface": self.allows_public_surface,
            "allows_scope_expansion": False,
            "allows_sibling_repositories": False,
            "allows_state_stores": self.allows_state_stores,
            "max_paths": self.max_paths,
            "max_symbols": self.max_symbols,
            "path_prefixes": list(self.path_prefixes),
            "schema": self.schema,
            "target_kinds": [item.value for item in self.target_kinds],
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
            path_prefixes=mapping["path_prefixes"],
            target_kinds=mapping["target_kinds"],
            max_paths=mapping["max_paths"],
            max_symbols=mapping["max_symbols"],
            allows_public_surface=mapping["allows_public_surface"],
            allows_state_stores=mapping["allows_state_stores"],
            allows_cross_package=mapping["allows_cross_package"],
            allows_sibling_repositories=mapping["allows_sibling_repositories"],
            allows_arbitrary_payloads=mapping["allows_arbitrary_payloads"],
            allows_scope_expansion=mapping["allows_scope_expansion"],
            allows_network=mapping["allows_network"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != record.content_identity:
            raise RefactorOperatorError("operator-scope content identity mismatch")
        return record

    from_dict = from_mapping


def assert_within_maximum_scope(parent: MaximumScope, child: MaximumScope) -> None:
    """Fail closed when a candidate or payload would expand declared scope."""

    if not parent.contains(child):
        refuse_scope_expansion("expand")


@dataclass(frozen=True)
class RefactorOperator:
    """One closed operator with a complete, identity-bound declaration."""

    kind: OperatorKind
    preconditions: tuple[PreconditionKind, ...]
    target_kinds: tuple[NodeKind, ...]
    expected_effects: tuple[ExpectedEffectKind, ...]
    authority_impact: AuthorityImpact
    public_api_impact: PublicApiImpact
    state_impact: StateImpact
    autonomy_disposition: AutonomyDisposition
    risk_class: AutonomyRiskClass
    migration: OperatorMigration
    rollback: OperatorRollback
    validation: tuple[ValidationObligationKind, ...]
    proofs: tuple[ProofObligationKind, ...]
    maximum_scope: MaximumScope
    preserved_invariants: tuple[str, ...] = NON_COMPENSABLE_INVARIANTS
    can_authorize_execution: bool = False
    can_reduce_gates: bool = False
    can_self_promote: bool = False
    can_raise_ceiling: bool = False
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
        kind = _closed_enum(
            self.kind, OperatorKind, "operator kind", error_type=RefactorOperatorError
        )
        preconditions = _require_enum_tuple(
            self.preconditions, PreconditionKind, "preconditions"
        )
        missing_pre = [
            item.value for item in REQUIRED_PRECONDITIONS if item not in preconditions
        ]
        if missing_pre:
            raise RefactorOperatorError(
                f"{_MISSING_FIELD_MESSAGE}: preconditions {missing_pre}"
            )
        target_kinds = _require_enum_tuple(self.target_kinds, NodeKind, "target_kinds")
        expected_effects = _require_enum_tuple(
            self.expected_effects, ExpectedEffectKind, "expected_effects"
        )
        authority = _closed_enum(
            self.authority_impact,
            AuthorityImpact,
            "authority impact",
            error_type=RefactorOperatorError,
        )
        public_api = _closed_enum(
            self.public_api_impact,
            PublicApiImpact,
            "public API impact",
            error_type=RefactorOperatorError,
        )
        state = _closed_enum(
            self.state_impact,
            StateImpact,
            "state impact",
            error_type=RefactorOperatorError,
        )
        disposition = _closed_enum(
            self.autonomy_disposition,
            AutonomyDisposition,
            "autonomy disposition",
            error_type=RefactorOperatorError,
        )
        risk = _closed_enum(
            self.risk_class,
            AutonomyRiskClass,
            "autonomy risk class",
            error_type=RefactorOperatorError,
        )
        if disposition_for_risk(risk) is not disposition:
            raise RefactorOperatorError(
                "autonomy disposition must match the declared risk class"
            )
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
        validation = _require_enum_tuple(
            self.validation, ValidationObligationKind, "validation"
        )
        missing_validation = [
            item.value
            for item in REQUIRED_VALIDATION_OBLIGATIONS
            if item not in validation
        ]
        if missing_validation:
            raise RefactorOperatorError(
                f"{_MISSING_FIELD_MESSAGE}: validation {missing_validation}"
            )
        proofs = _require_enum_tuple(self.proofs, ProofObligationKind, "proofs")
        missing_proofs = [
            item.value for item in REQUIRED_PROOF_OBLIGATIONS if item not in proofs
        ]
        if missing_proofs:
            raise RefactorOperatorError(
                f"{_MISSING_FIELD_MESSAGE}: proofs {missing_proofs}"
            )
        scope = (
            self.maximum_scope
            if isinstance(self.maximum_scope, MaximumScope)
            else MaximumScope.from_mapping(self.maximum_scope)
        )
        if set(scope.target_kinds) != set(target_kinds):
            raise RefactorOperatorError(
                "maximum-scope target kinds must match operator target kinds"
            )
        invariants = _require_text_tuple(self.preserved_invariants, "preserved_invariants")
        if set(invariants) != set(NON_COMPENSABLE_INVARIANTS):
            refuse_gate_reduction("drop")
        if state is StateImpact.MOVE_TO_OWNER and not scope.allows_state_stores:
            raise RefactorOperatorError(
                "move_state_to_owner requires allows_state_stores"
            )
        if state is StateImpact.MOVE_TO_OWNER and not migration.mutates_state:
            raise RefactorOperatorError(
                "state-moving operators must declare migration.mutates_state"
            )
        if state is not StateImpact.MOVE_TO_OWNER and migration.mutates_state:
            raise RefactorOperatorError(
                "non-state-moving operators cannot declare migration.mutates_state"
            )
        if public_api is not PublicApiImpact.INTERNAL and not scope.allows_public_surface:
            raise RefactorOperatorError(
                "non-internal public API impact requires allows_public_surface"
            )
        if disposition is AutonomyDisposition.AUTOMATIC:
            if public_api is not PublicApiImpact.INTERNAL:
                raise RefactorOperatorAuthorityError(
                    "automatic operators cannot change the public API"
                )
            if state not in {StateImpact.NONE, StateImpact.READ_ONLY}:
                raise RefactorOperatorAuthorityError(
                    "automatic operators cannot mutate authoritative state"
                )
            if authority is AuthorityImpact.CONSOLIDATE:
                raise RefactorOperatorAuthorityError(
                    "automatic operators cannot consolidate authority"
                )
        if (
            disposition is not AutonomyDisposition.ALWAYS_HUMAN
            and authority is AuthorityImpact.CONSOLIDATE
        ):
            raise RefactorOperatorAuthorityError(
                "authority consolidation always requires human approval"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "target_kinds", target_kinds)
        object.__setattr__(self, "expected_effects", expected_effects)
        object.__setattr__(self, "authority_impact", authority)
        object.__setattr__(self, "public_api_impact", public_api)
        object.__setattr__(self, "state_impact", state)
        object.__setattr__(self, "autonomy_disposition", disposition)
        object.__setattr__(self, "risk_class", risk)
        object.__setattr__(self, "migration", migration)
        object.__setattr__(self, "rollback", rollback)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "proofs", proofs)
        object.__setattr__(self, "maximum_scope", scope)
        object.__setattr__(self, "preserved_invariants", invariants)
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
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "refactor-operator"),
        )

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
            "expected_effects": [item.value for item in self.expected_effects],
            "kind": self.kind.value,
            "maximum_scope": self.maximum_scope.to_dict(),
            "migration": self.migration.to_dict(),
            "preconditions": [item.value for item in self.preconditions],
            "preserved_invariants": list(self.preserved_invariants),
            "proofs": [item.value for item in self.proofs],
            "public_api_impact": self.public_api_impact.value,
            "risk_class": self.risk_class.value,
            "rollback": self.rollback.to_dict(),
            "schema": self.schema,
            "state_impact": self.state_impact.value,
            "target_kinds": [item.value for item in self.target_kinds],
            "validation": [item.value for item in self.validation],
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
            public_api_impact=mapping["public_api_impact"],
            state_impact=mapping["state_impact"],
            autonomy_disposition=mapping["autonomy_disposition"],
            risk_class=mapping["risk_class"],
            migration=mapping["migration"],
            rollback=mapping["rollback"],
            validation=mapping["validation"],
            proofs=mapping["proofs"],
            maximum_scope=mapping["maximum_scope"],
            preserved_invariants=mapping["preserved_invariants"],
            can_authorize_execution=mapping["can_authorize_execution"],
            can_reduce_gates=mapping["can_reduce_gates"],
            can_self_promote=mapping["can_self_promote"],
            can_raise_ceiling=mapping["can_raise_ceiling"],
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
            raise RefactorOperatorError("refactor-operator JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise RefactorOperatorError("refactor-operator JSON must contain an object")
        return cls.from_mapping(decoded)


def _merge_required(
    extra: tuple[Enum, ...],
    required: tuple[Enum, ...],
) -> tuple[Enum, ...]:
    return tuple(sorted(set(required) | set(extra), key=lambda item: item.value))


def _scope(
    target_kinds: tuple[NodeKind, ...],
    *,
    path_prefixes: tuple[str, ...] = _DEFAULT_PATH_PREFIXES,
    max_paths: int = 32,
    max_symbols: int = 64,
    allows_public_surface: bool = False,
    allows_state_stores: bool = False,
    allows_cross_package: bool = False,
) -> MaximumScope:
    return MaximumScope(
        path_prefixes=path_prefixes,
        target_kinds=target_kinds,
        max_paths=max_paths,
        max_symbols=max_symbols,
        allows_public_surface=allows_public_surface,
        allows_state_stores=allows_state_stores,
        allows_cross_package=allows_cross_package,
    )


def _migration(
    *phases: MigrationPhase,
    adapters: tuple[str, ...] = (),
    deprecated_paths: tuple[str, ...] = (),
    mutates_state: bool = False,
) -> OperatorMigration:
    ordered = phases + (
        () if phases and phases[-1] is MigrationPhase.VALIDATE_AND_SEAL else (MigrationPhase.VALIDATE_AND_SEAL,)
    )
    return OperatorMigration(
        phases=ordered,
        adapters=adapters,
        deprecated_paths=deprecated_paths,
        mutates_state=mutates_state,
    )


def _rollback(action: RollbackAction = RollbackAction.RESTORE_SEALED_TREE) -> OperatorRollback:
    return OperatorRollback(action=action)


def _operator(
    kind: OperatorKind,
    *,
    extra_preconditions: tuple[PreconditionKind, ...] = (),
    target_kinds: tuple[NodeKind, ...],
    expected_effects: tuple[ExpectedEffectKind, ...],
    authority_impact: AuthorityImpact,
    public_api_impact: PublicApiImpact,
    state_impact: StateImpact,
    risk_class: AutonomyRiskClass,
    migration: OperatorMigration,
    extra_validation: tuple[ValidationObligationKind, ...] = (),
    extra_proofs: tuple[ProofObligationKind, ...] = (),
    maximum_scope: MaximumScope,
) -> RefactorOperator:
    return RefactorOperator(
        kind=kind,
        preconditions=_merge_required(extra_preconditions, REQUIRED_PRECONDITIONS),
        target_kinds=target_kinds,
        expected_effects=expected_effects,
        authority_impact=authority_impact,
        public_api_impact=public_api_impact,
        state_impact=state_impact,
        autonomy_disposition=disposition_for_risk(risk_class),
        risk_class=risk_class,
        migration=migration,
        rollback=_rollback(),
        validation=_merge_required(extra_validation, REQUIRED_VALIDATION_OBLIGATIONS),
        proofs=_merge_required(extra_proofs, REQUIRED_PROOF_OBLIGATIONS),
        maximum_scope=maximum_scope,
        preserved_invariants=NON_COMPENSABLE_INVARIANTS,
    )


def _build_initial_operators() -> tuple[RefactorOperator, ...]:
    extract_module_targets = (NodeKind.FILE, NodeKind.MODULE, NodeKind.SYMBOL)
    extract_interface_targets = (
        NodeKind.INTERFACE,
        NodeKind.MODULE,
        NodeKind.SYMBOL,
    )
    extract_function_targets = (NodeKind.SYMBOL, NodeKind.MODULE)
    state_targets = (NodeKind.STATE, NodeKind.SCHEMA, NodeKind.AUTHORITY)
    inversion_targets = (
        NodeKind.INTERFACE,
        NodeKind.MODULE,
        NodeKind.SYMBOL,
        NodeKind.OPERATION,
    )
    typed_service_targets = (NodeKind.OPERATION, NodeKind.SYMBOL, NodeKind.INTERFACE)
    adapter_targets = (
        NodeKind.SYMBOL,
        NodeKind.INTERFACE,
        NodeKind.COMPATIBILITY,
    )
    shim_targets = (
        NodeKind.COMPATIBILITY,
        NodeKind.SYMBOL,
        NodeKind.ENTRYPOINT,
    )
    legacy_targets = (NodeKind.COMPATIBILITY, NodeKind.FILE, NodeKind.MODULE)
    simulation_targets = (NodeKind.SIMULATION, NodeKind.FILE, NodeKind.MODULE)
    outcome_targets = (NodeKind.SYMBOL, NodeKind.SCHEMA, NodeKind.INTERFACE)
    catalog_targets = (
        NodeKind.SCHEMA,
        NodeKind.OPERATION,
        NodeKind.PROVIDER,
        NodeKind.SYMBOL,
    )
    lazy_targets = (NodeKind.MODULE, NodeKind.SYMBOL, NodeKind.FILE)
    error_targets = (NodeKind.SCHEMA, NodeKind.SYMBOL, NodeKind.INTERFACE)
    receipt_targets = (NodeKind.RECEIPT, NodeKind.SYMBOL, NodeKind.SCHEMA)
    authority_targets = (
        NodeKind.AUTHORITY,
        NodeKind.PROVIDER,
        NodeKind.POLICY,
        NodeKind.OPERATION,
    )
    dead_targets = (NodeKind.SYMBOL, NodeKind.MODULE, NodeKind.FILE)
    split_targets = (
        NodeKind.MODULE,
        NodeKind.PACKAGE,
        NodeKind.AUTHORITY,
        NodeKind.FILE,
    )
    generated_targets = (NodeKind.GENERATED, NodeKind.ARTIFACT, NodeKind.FILE)
    public_targets = (NodeKind.SYMBOL, NodeKind.ENTRYPOINT, NodeKind.INTERFACE)
    return (
        _operator(
            OperatorKind.EXTRACT_MODULE,
            extra_preconditions=(
                PreconditionKind.PURE_INTERNAL,
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=extract_module_targets,
            expected_effects=(ExpectedEffectKind.EXTRACT,),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.PURE_MODULE_EXTRACTION,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.EXTRACT, MigrationPhase.ADAPT_CALLERS),
            extra_proofs=(ProofObligationKind.BEHAVIOR_EQUIVALENCE,),
            maximum_scope=_scope(extract_module_targets, max_paths=16, max_symbols=64),
        ),
        _operator(
            OperatorKind.EXTRACT_INTERFACE,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=extract_interface_targets,
            expected_effects=(ExpectedEffectKind.EXTRACT, ExpectedEffectKind.ADAPT),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.VERSIONED_MIGRATION,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.PUBLIC_API,
            migration=_migration(
                MigrationPhase.DECLARE,
                MigrationPhase.EXTRACT,
                MigrationPhase.ADAPT_CALLERS,
                MigrationPhase.GENERATE_ADAPTER,
            ),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(ProofObligationKind.BEHAVIOR_EQUIVALENCE, ProofObligationKind.REFINEMENT),
            maximum_scope=_scope(
                extract_interface_targets,
                max_paths=24,
                max_symbols=48,
                allows_public_surface=True,
            ),
        ),
        _operator(
            OperatorKind.EXTRACT_PURE_FUNCTION,
            extra_preconditions=(
                PreconditionKind.PURE_INTERNAL,
                PreconditionKind.CANONICAL_OWNER,
            ),
            target_kinds=extract_function_targets,
            expected_effects=(ExpectedEffectKind.EXTRACT,),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.PURE_MODULE_EXTRACTION,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.EXTRACT, MigrationPhase.ADAPT_CALLERS),
            extra_proofs=(ProofObligationKind.BEHAVIOR_EQUIVALENCE,),
            maximum_scope=_scope(extract_function_targets, max_paths=8, max_symbols=16),
        ),
        _operator(
            OperatorKind.MOVE_STATE_TO_OWNER,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.UNIQUE_STATE_OWNER,
                PreconditionKind.NO_INDEFINITE_DUAL_AUTHORITY,
            ),
            target_kinds=state_targets,
            expected_effects=(ExpectedEffectKind.MOVE,),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.MOVE_TO_OWNER,
            risk_class=AutonomyRiskClass.STATE,
            migration=_migration(
                MigrationPhase.DECLARE,
                MigrationPhase.SNAPSHOT,
                MigrationPhase.SHADOW_COMPARE,
                MigrationPhase.BOUNDED_DUAL_WRITE,
                MigrationPhase.CUTOVER,
                MigrationPhase.READ_ONLY_LEGACY,
                MigrationPhase.RETIRE,
                mutates_state=True,
            ),
            extra_validation=(ValidationObligationKind.STATE_OWNERSHIP,),
            extra_proofs=(
                ProofObligationKind.UNIQUE_STATE_OWNER,
                ProofObligationKind.NO_INDEFINITE_DUAL_AUTHORITY,
            ),
            maximum_scope=_scope(
                state_targets,
                max_paths=16,
                max_symbols=32,
                allows_state_stores=True,
            ),
        ),
        _operator(
            OperatorKind.INTRODUCE_DEPENDENCY_INVERSION,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=inversion_targets,
            expected_effects=(ExpectedEffectKind.ADAPT, ExpectedEffectKind.REPLACE),
            authority_impact=AuthorityImpact.ADAPTER,
            public_api_impact=PublicApiImpact.VERSIONED_MIGRATION,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.CROSS_PACKAGE_MIGRATION,
            migration=_migration(
                MigrationPhase.DECLARE,
                MigrationPhase.GENERATE_ADAPTER,
                MigrationPhase.ADAPT_CALLERS,
            ),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(ProofObligationKind.REFINEMENT,),
            maximum_scope=_scope(
                inversion_targets,
                max_paths=32,
                max_symbols=64,
                allows_public_surface=True,
                allows_cross_package=True,
            ),
        ),
        _operator(
            OperatorKind.REPLACE_DIRECT_CALL_WITH_TYPED_SERVICE,
            extra_preconditions=(
                PreconditionKind.PURE_INTERNAL,
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=typed_service_targets,
            expected_effects=(ExpectedEffectKind.REPLACE, ExpectedEffectKind.ADAPT),
            authority_impact=AuthorityImpact.ADAPTER,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.INTERNAL_ADAPTER_GENERATION,
            migration=_migration(
                MigrationPhase.DECLARE,
                MigrationPhase.GENERATE_ADAPTER,
                MigrationPhase.ADAPT_CALLERS,
            ),
            extra_proofs=(ProofObligationKind.BEHAVIOR_EQUIVALENCE, ProofObligationKind.REFINEMENT),
            maximum_scope=_scope(typed_service_targets, max_paths=16, max_symbols=32),
        ),
        _operator(
            OperatorKind.GENERATE_ADAPTER,
            extra_preconditions=(
                PreconditionKind.PURE_INTERNAL,
                PreconditionKind.CANONICAL_OWNER,
            ),
            target_kinds=adapter_targets,
            expected_effects=(ExpectedEffectKind.GENERATE, ExpectedEffectKind.ADAPT),
            authority_impact=AuthorityImpact.ADAPTER,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.INTERNAL_ADAPTER_GENERATION,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.GENERATE_ADAPTER, MigrationPhase.ADAPT_CALLERS),
            extra_proofs=(ProofObligationKind.REFINEMENT,),
            maximum_scope=_scope(adapter_targets, max_paths=12, max_symbols=24),
        ),
        _operator(
            OperatorKind.GENERATE_COMPATIBILITY_SHIM,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.COMPATIBILITY_SATISFACTION,
            ),
            target_kinds=shim_targets,
            expected_effects=(ExpectedEffectKind.GENERATE, ExpectedEffectKind.ADAPT),
            authority_impact=AuthorityImpact.ADAPTER,
            public_api_impact=PublicApiImpact.COMPATIBILITY,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.LEGACY,
            migration=_migration(
                MigrationPhase.DECLARE,
                MigrationPhase.GENERATE_ADAPTER,
                MigrationPhase.DEPRECATE,
            ),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(ProofObligationKind.COMPATIBILITY_SATISFACTION,),
            maximum_scope=_scope(
                shim_targets,
                max_paths=16,
                max_symbols=32,
                allows_public_surface=True,
            ),
        ),
        _operator(
            OperatorKind.QUARANTINE_LEGACY_PATH,
            extra_preconditions=(PreconditionKind.CANONICAL_OWNER,),
            target_kinds=legacy_targets,
            expected_effects=(ExpectedEffectKind.QUARANTINE,),
            authority_impact=AuthorityImpact.QUARANTINE,
            public_api_impact=PublicApiImpact.COMPATIBILITY,
            state_impact=StateImpact.READ_ONLY,
            risk_class=AutonomyRiskClass.LEGACY,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.QUARANTINE, MigrationPhase.READ_ONLY_LEGACY),
            extra_validation=(ValidationObligationKind.QUARANTINE_REACHABILITY,),
            extra_proofs=(ProofObligationKind.QUARANTINE_FLOW,),
            maximum_scope=_scope(
                legacy_targets,
                max_paths=24,
                max_symbols=48,
                allows_public_surface=True,
            ),
        ),
        _operator(
            OperatorKind.QUARANTINE_SIMULATION_PATH,
            extra_preconditions=(PreconditionKind.CANONICAL_OWNER, PreconditionKind.PURE_INTERNAL),
            target_kinds=simulation_targets,
            expected_effects=(ExpectedEffectKind.QUARANTINE, ExpectedEffectKind.MOVE),
            authority_impact=AuthorityImpact.QUARANTINE,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.SIMULATION_NAMESPACE_RELOCATION,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.QUARANTINE),
            extra_validation=(ValidationObligationKind.QUARANTINE_REACHABILITY,),
            extra_proofs=(ProofObligationKind.QUARANTINE_FLOW,),
            maximum_scope=_scope(simulation_targets, max_paths=16, max_symbols=32),
        ),
        _operator(
            OperatorKind.REPLACE_BOOLEAN_WITH_CLOSED_OUTCOME,
            extra_preconditions=(
                PreconditionKind.PURE_INTERNAL,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=outcome_targets,
            expected_effects=(ExpectedEffectKind.REPLACE,),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.CLOSED_RESULT_TYPE_MIGRATION,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.ADAPT_CALLERS),
            extra_proofs=(ProofObligationKind.REFINEMENT,),
            maximum_scope=_scope(outcome_targets, max_paths=16, max_symbols=48),
        ),
        _operator(
            OperatorKind.REPLACE_DYNAMIC_REGISTRY_WITH_TYPED_CATALOG,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=catalog_targets,
            expected_effects=(ExpectedEffectKind.REPLACE,),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.VERSIONED_MIGRATION,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.PROVIDER,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.ADAPT_CALLERS),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(ProofObligationKind.REFINEMENT,),
            maximum_scope=_scope(
                catalog_targets,
                max_paths=24,
                max_symbols=64,
                allows_public_surface=True,
            ),
        ),
        _operator(
            OperatorKind.REPLACE_EAGER_IMPORT_WITH_LAZY_CAPABILITY,
            extra_preconditions=(PreconditionKind.PURE_INTERNAL,),
            target_kinds=lazy_targets,
            expected_effects=(ExpectedEffectKind.REPLACE,),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.LAZY_IMPORT_CONVERSION,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.ADAPT_CALLERS),
            extra_proofs=(ProofObligationKind.BEHAVIOR_EQUIVALENCE,),
            maximum_scope=_scope(lazy_targets, max_paths=24, max_symbols=48),
        ),
        _operator(
            OperatorKind.CONSOLIDATE_ERROR_VOCABULARY,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=error_targets,
            expected_effects=(ExpectedEffectKind.CONSOLIDATE, ExpectedEffectKind.REPLACE),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.VERSIONED_MIGRATION,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.PUBLIC_API,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.ADAPT_CALLERS, MigrationPhase.DEPRECATE),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(ProofObligationKind.REFINEMENT,),
            maximum_scope=_scope(
                error_targets,
                max_paths=32,
                max_symbols=64,
                allows_public_surface=True,
            ),
        ),
        _operator(
            OperatorKind.CONSOLIDATE_RECEIPT_PRODUCER,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=receipt_targets,
            expected_effects=(ExpectedEffectKind.CONSOLIDATE,),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.VERSIONED_MIGRATION,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.RECEIPT,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.ADAPT_CALLERS, MigrationPhase.DEPRECATE),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(ProofObligationKind.REFINEMENT,),
            maximum_scope=_scope(
                receipt_targets,
                max_paths=16,
                max_symbols=32,
                allows_public_surface=True,
            ),
        ),
        _operator(
            OperatorKind.CONSOLIDATE_CAPABILITY_AUTHORITY,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=authority_targets,
            expected_effects=(ExpectedEffectKind.CONSOLIDATE, ExpectedEffectKind.ADAPT),
            authority_impact=AuthorityImpact.CONSOLIDATE,
            public_api_impact=PublicApiImpact.VERSIONED_MIGRATION,
            state_impact=StateImpact.NO_INDEFINITE_DUAL_AUTHORITY,
            risk_class=AutonomyRiskClass.AUTHORIZATION,
            migration=_migration(
                MigrationPhase.DECLARE,
                MigrationPhase.GENERATE_ADAPTER,
                MigrationPhase.ADAPT_CALLERS,
                MigrationPhase.DEPRECATE,
            ),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(ProofObligationKind.CANONICAL_OWNER_PRESERVED,),
            maximum_scope=_scope(
                authority_targets,
                max_paths=32,
                max_symbols=64,
                allows_public_surface=True,
            ),
        ),
        _operator(
            OperatorKind.REMOVE_CONFIRMED_DEAD_CODE,
            extra_preconditions=(
                PreconditionKind.PURE_INTERNAL,
                PreconditionKind.CONFIRMED_DEAD,
                PreconditionKind.NO_DYNAMIC_LOADING_UNCERTAINTY,
            ),
            target_kinds=dead_targets,
            expected_effects=(ExpectedEffectKind.REMOVE,),
            authority_impact=AuthorityImpact.NONE,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.CONFIRMED_DEAD_INTERNAL_CODE_REMOVAL,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.RETIRE),
            extra_proofs=(ProofObligationKind.DEAD_CODE_CONFIRMATION, ProofObligationKind.CONSUMER_ABSENCE),
            maximum_scope=_scope(dead_targets, max_paths=16, max_symbols=32),
        ),
        _operator(
            OperatorKind.SPLIT_MONOLITH_BY_AUTHORITY,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.NO_CONTRACT_AMBIGUITY,
            ),
            target_kinds=split_targets,
            expected_effects=(ExpectedEffectKind.SPLIT, ExpectedEffectKind.EXTRACT),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.VERSIONED_MIGRATION,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.CROSS_PACKAGE_MIGRATION,
            migration=_migration(
                MigrationPhase.DECLARE,
                MigrationPhase.EXTRACT,
                MigrationPhase.ADAPT_CALLERS,
                MigrationPhase.GENERATE_ADAPTER,
            ),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(ProofObligationKind.BEHAVIOR_EQUIVALENCE, ProofObligationKind.REFINEMENT),
            maximum_scope=_scope(
                split_targets,
                max_paths=64,
                max_symbols=128,
                allows_public_surface=True,
                allows_cross_package=True,
            ),
        ),
        _operator(
            OperatorKind.MOVE_GENERATED_PROJECTION_OUT_OF_SOURCE_AUTHORITY,
            extra_preconditions=(PreconditionKind.PURE_INTERNAL, PreconditionKind.CANONICAL_OWNER),
            target_kinds=generated_targets,
            expected_effects=(ExpectedEffectKind.MOVE, ExpectedEffectKind.GENERATE),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.INTERNAL,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.GENERATED_PROJECTION_REGENERATION,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.EXTRACT, MigrationPhase.ADAPT_CALLERS),
            extra_proofs=(ProofObligationKind.REFINEMENT,),
            maximum_scope=_scope(generated_targets, max_paths=24, max_symbols=48),
        ),
        _operator(
            OperatorKind.DEPRECATE_PUBLIC_SYMBOL,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.CONSUMER_MIGRATION,
            ),
            target_kinds=public_targets,
            expected_effects=(ExpectedEffectKind.DEPRECATE,),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.DEPRECATE,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.PUBLIC_API,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.DEPRECATE, MigrationPhase.ADAPT_CALLERS),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(ProofObligationKind.COMPATIBILITY_SATISFACTION,),
            maximum_scope=_scope(
                public_targets,
                max_paths=16,
                max_symbols=32,
                allows_public_surface=True,
            ),
        ),
        _operator(
            OperatorKind.REMOVE_DEPRECATED_SYMBOL_AFTER_GATE,
            extra_preconditions=(
                PreconditionKind.CANONICAL_OWNER,
                PreconditionKind.PRIOR_DEPRECATION,
                PreconditionKind.CONSUMER_MIGRATION,
                PreconditionKind.COMPATIBILITY_SATISFACTION,
            ),
            target_kinds=public_targets,
            expected_effects=(ExpectedEffectKind.REMOVE, ExpectedEffectKind.DEPRECATE),
            authority_impact=AuthorityImpact.PRESERVE,
            public_api_impact=PublicApiImpact.REMOVE_AFTER_GATE,
            state_impact=StateImpact.NONE,
            risk_class=AutonomyRiskClass.PUBLIC_API,
            migration=_migration(MigrationPhase.DECLARE, MigrationPhase.RETIRE),
            extra_validation=(ValidationObligationKind.PUBLIC_CONTRACT,),
            extra_proofs=(
                ProofObligationKind.CONSUMER_ABSENCE,
                ProofObligationKind.COMPATIBILITY_SATISFACTION,
            ),
            maximum_scope=_scope(
                public_targets,
                max_paths=16,
                max_symbols=32,
                allows_public_surface=True,
            ),
        ),
    )


INITIAL_OPERATOR_DECLARATIONS: tuple[RefactorOperator, ...] = _build_initial_operators()
_OPERATORS_BY_KIND: dict[OperatorKind, RefactorOperator] = {
    item.kind: item for item in INITIAL_OPERATOR_DECLARATIONS
}
assert tuple(item.kind for item in INITIAL_OPERATOR_DECLARATIONS) == INITIAL_OPERATORS
assert frozenset(_OPERATORS_BY_KIND) == frozenset(OperatorKind)


def operator_for(kind: OperatorKind | str) -> RefactorOperator:
    """Return the canonical declaration for one closed operator kind."""

    try:
        resolved = kind if isinstance(kind, OperatorKind) else OperatorKind(kind)
    except ValueError:
        refuse_unknown_operator(kind)
    return _OPERATORS_BY_KIND[resolved]


def autonomy_classification_map() -> dict[str, str]:
    """Return kind -> risk-class for every required initial operator."""

    return {item.kind.value: item.risk_class.value for item in INITIAL_OPERATOR_DECLARATIONS}


@dataclass(frozen=True)
class OperatorCatalog:
    """Identity-bound catalog of the closed operator grammar."""

    operators: tuple[RefactorOperator, ...] = INITIAL_OPERATOR_DECLARATIONS
    autonomy_classification: tuple[tuple[str, str], ...] = ()
    covers_initial_operators: bool = True
    can_authorize_execution: bool = False
    can_reduce_gates: bool = False
    can_self_promote: bool = False
    can_raise_ceiling: bool = False
    declarations_are_immutable: bool = True
    effect_class: str = EFFECT_CLASS
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
        if isinstance(self.operators, (str, bytes, bytearray)) or not isinstance(
            self.operators, Sequence
        ):
            raise RefactorOperatorError("operators must be a list of objects")
        operators = tuple(
            item if isinstance(item, RefactorOperator) else RefactorOperator.from_mapping(item)
            for item in self.operators
        )
        kinds = tuple(item.kind for item in operators)
        if kinds != INITIAL_OPERATORS:
            raise RefactorOperatorError("operator catalog must cover the initial vocabulary in order")
        classification_payload = self.autonomy_classification
        expected_pairs = tuple(
            (item.kind.value, item.risk_class.value) for item in operators
        )
        if not classification_payload:
            classification = expected_pairs
        else:
            if isinstance(classification_payload, Mapping):
                pairs = tuple(
                    (
                        _require_text(key, "autonomy classification key", error_type=RefactorOperatorError),
                        _require_text(value, "autonomy classification value", error_type=RefactorOperatorError),
                    )
                    for key, value in classification_payload.items()
                )
            elif isinstance(classification_payload, Sequence) and not isinstance(
                classification_payload, (str, bytes, bytearray)
            ):
                pairs = tuple(
                    (
                        _require_text(item[0], "autonomy classification key", error_type=RefactorOperatorError),
                        _require_text(item[1], "autonomy classification value", error_type=RefactorOperatorError),
                    )
                    for item in classification_payload
                )
            else:
                raise RefactorOperatorError("autonomy_classification must be an object or pair list")
            if set(pairs) != set(expected_pairs):
                raise RefactorOperatorError("autonomy classification must match operator risk classes")
            classification = expected_pairs
        effect_class = _require_text(
            self.effect_class, "effect_class", error_type=RefactorOperatorError
        )
        if effect_class != EFFECT_CLASS:
            raise RefactorOperatorError("unexpected operator-catalog effect class")
        if _require_bool(self.covers_initial_operators, "covers_initial_operators") is not True:
            raise RefactorOperatorError("operator catalog must cover the initial operators")
        if _require_bool(self.declarations_are_immutable, "declarations_are_immutable") is not True:
            raise RefactorOperatorError("operator declarations must remain immutable")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "autonomy_classification", classification)
        object.__setattr__(self, "covers_initial_operators", True)
        object.__setattr__(self, "effect_class", EFFECT_CLASS)
        object.__setattr__(self, "declarations_are_immutable", True)
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
        object.__setattr__(
            self,
            "content_identity",
            _bind_identity(self, self.content_identity, "operator-catalog"),
        )

    def operator(self, kind: OperatorKind | str) -> RefactorOperator:
        return operator_for(kind)

    def apply(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_execution("apply")

    def authorize(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_self_authorization("authorize")

    def promote(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_self_promotion("promote")

    def raise_ceiling(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_ceiling_raise("raise")

    def reduce_gates(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_gate_reduction("reduce")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "autonomy_classification": [
                [kind, risk] for kind, risk in self.autonomy_classification
            ],
            "can_authorize_execution": False,
            "can_raise_ceiling": False,
            "can_reduce_gates": False,
            "can_self_promote": False,
            "covers_initial_operators": True,
            "declarations_are_immutable": True,
            "effect_class": EFFECT_CLASS,
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
            autonomy_classification=mapping["autonomy_classification"],
            covers_initial_operators=mapping["covers_initial_operators"],
            can_authorize_execution=mapping["can_authorize_execution"],
            can_reduce_gates=mapping["can_reduce_gates"],
            can_self_promote=mapping["can_self_promote"],
            can_raise_ceiling=mapping["can_raise_ceiling"],
            declarations_are_immutable=mapping["declarations_are_immutable"],
            effect_class=mapping["effect_class"],
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
            raise RefactorOperatorError("operator-catalog JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise RefactorOperatorError("operator-catalog JSON must contain an object")
        return cls.from_mapping(decoded)


def operator_catalog() -> OperatorCatalog:
    """Return the canonical closed operator catalog."""

    return OperatorCatalog()


def operator_grammar() -> OperatorCatalog:
    """Alias for the closed operator grammar catalog."""

    return operator_catalog()


__all__ = [
    "ALWAYS_HUMAN_RISK_CLASSES",
    "AUTOMATIC_OPERATOR_CLASSES",
    "AutonomyDisposition",
    "AutonomyRiskClass",
    "AuthorityImpact",
    "CLOSED_ALWAYS_HUMAN_RISK_CLASSES",
    "CLOSED_AUTHORITY_IMPACTS",
    "CLOSED_AUTOMATIC_OPERATOR_CLASSES",
    "CLOSED_AUTONOMY_DISPOSITIONS",
    "CLOSED_AUTONOMY_RISK_CLASSES",
    "CLOSED_EXPECTED_EFFECTS",
    "CLOSED_MIGRATION_PHASES",
    "CLOSED_OPERATOR_KINDS",
    "CLOSED_PRECONDITIONS",
    "CLOSED_PROOF_OBLIGATIONS",
    "CLOSED_PROPOSAL_ONLY_RISK_CLASSES",
    "CLOSED_PUBLIC_API_IMPACTS",
    "CLOSED_ROLLBACK_ACTIONS",
    "CLOSED_STATE_IMPACTS",
    "CLOSED_VALIDATION_OBLIGATIONS",
    "DECLARATIONS_ARE_IMMUTABLE",
    "DEFAULT_FRESHNESS",
    "EFFECT_CLASS",
    "EXTRACTOR_IDENTITY",
    "ExpectedEffectKind",
    "INCOMPLETE_DECLARATIONS_FAIL_CLOSED",
    "INITIAL_OPERATORS",
    "INITIAL_OPERATOR_DECLARATIONS",
    "MaximumScope",
    "MigrationPhase",
    "OPERATOR_CAN_ADMIT_ARBITRARY_PAYLOADS",
    "OPERATOR_CAN_AUTHORIZE_EXECUTION",
    "OPERATOR_CAN_EXPAND_SCOPE",
    "OPERATOR_CAN_RAISE_CEILING",
    "OPERATOR_CAN_REDUCE_GATES",
    "OPERATOR_CAN_SELF_PROMOTE",
    "OPERATOR_CATALOG_SCHEMA",
    "OPERATOR_CATALOG_VERSION",
    "OPERATOR_MIGRATION_SCHEMA",
    "OPERATOR_MIGRATION_VERSION",
    "OPERATOR_ROLLBACK_SCHEMA",
    "OPERATOR_ROLLBACK_VERSION",
    "OPERATOR_SCOPE_SCHEMA",
    "OPERATOR_SCOPE_VERSION",
    "OperatorCatalog",
    "OperatorKind",
    "OperatorMigration",
    "OperatorRollback",
    "PROPOSAL_ONLY_RISK_CLASSES",
    "PreconditionKind",
    "ProofObligationKind",
    "PublicApiImpact",
    "REFACTOR_OPERATOR_EVIDENCE",
    "REFACTOR_OPERATOR_SCHEMA",
    "REFACTOR_OPERATOR_VERSION",
    "REQUIRED_OPERATORS",
    "REQUIRED_PRECONDITIONS",
    "REQUIRED_PROOF_OBLIGATIONS",
    "REQUIRED_VALIDATION_OBLIGATIONS",
    "RefactorOperator",
    "RefactorOperatorAuthorityError",
    "RefactorOperatorError",
    "RollbackAction",
    "SCOPE_EXPANSION_FAILS_CLOSED",
    "StateImpact",
    "TASK_ID",
    "UNKNOWN_OPERATORS_FAIL_CLOSED",
    "ValidationObligationKind",
    "assert_within_maximum_scope",
    "autonomy_classification_map",
    "autonomy_rank",
    "disposition_for_risk",
    "operator_catalog",
    "operator_for",
    "operator_grammar",
    "path_in_scope_prefix",
    "refuse_arbitrary_payload",
    "refuse_ceiling_raise",
    "refuse_execution",
    "refuse_gate_reduction",
    "refuse_scope_expansion",
    "refuse_self_authorization",
    "refuse_self_promotion",
    "refuse_unknown_operator",
]
