"""Hermetic PCAR-012 closed refactor-operator grammar tests."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.candidate import (
    CANDIDATE_CAN_AUTHORIZE_EXECUTION,
    CANDIDATE_CAN_EXPAND_SCOPE,
    CANDIDATE_CAN_RAISE_CEILING,
    CANDIDATE_CAN_REDUCE_GATES,
    CANDIDATE_CAN_SELF_PROMOTE,
    CANDIDATE_IDENTITY_BINDS_CONTRACT,
    CANDIDATE_IDENTITY_BINDS_EFFECTS,
    CANDIDATE_IDENTITY_BINDS_TREE,
    REFACTOR_CANDIDATE_EVIDENCE,
    REFACTOR_CANDIDATE_SCHEMA,
    REFACTOR_CANDIDATE_VERSION,
    RefactorCandidate,
    RefactorCandidateAuthorityError,
    RefactorCandidateError,
    declare_refactor_candidate,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import NodeKind
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.entropy import (
    NON_COMPENSABLE_INVARIANTS,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.refactor_operators import (
    ALWAYS_HUMAN_RISK_CLASSES,
    AUTOMATIC_OPERATOR_CLASSES,
    CLOSED_ALWAYS_HUMAN_RISK_CLASSES,
    CLOSED_AUTHORITY_IMPACTS,
    CLOSED_AUTOMATIC_OPERATOR_CLASSES,
    CLOSED_AUTONOMY_DISPOSITIONS,
    CLOSED_AUTONOMY_RISK_CLASSES,
    CLOSED_EXPECTED_EFFECTS,
    CLOSED_MIGRATION_PHASES,
    CLOSED_OPERATOR_KINDS,
    CLOSED_PRECONDITIONS,
    CLOSED_PROOF_OBLIGATIONS,
    CLOSED_PROPOSAL_ONLY_RISK_CLASSES,
    CLOSED_PUBLIC_API_IMPACTS,
    CLOSED_ROLLBACK_ACTIONS,
    CLOSED_STATE_IMPACTS,
    CLOSED_VALIDATION_OBLIGATIONS,
    DECLARATIONS_ARE_IMMUTABLE,
    DEFAULT_FRESHNESS,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    INCOMPLETE_DECLARATIONS_FAIL_CLOSED,
    INITIAL_OPERATORS,
    INITIAL_OPERATOR_DECLARATIONS,
    OPERATOR_CAN_ADMIT_ARBITRARY_PAYLOADS,
    OPERATOR_CAN_AUTHORIZE_EXECUTION,
    OPERATOR_CAN_EXPAND_SCOPE,
    OPERATOR_CAN_RAISE_CEILING,
    OPERATOR_CAN_REDUCE_GATES,
    OPERATOR_CAN_SELF_PROMOTE,
    OPERATOR_CATALOG_SCHEMA,
    PROPOSAL_ONLY_RISK_CLASSES,
    REFACTOR_OPERATOR_EVIDENCE,
    REFACTOR_OPERATOR_SCHEMA,
    REFACTOR_OPERATOR_VERSION,
    REQUIRED_OPERATORS,
    REQUIRED_PRECONDITIONS,
    REQUIRED_PROOF_OBLIGATIONS,
    REQUIRED_VALIDATION_OBLIGATIONS,
    SCOPE_EXPANSION_FAILS_CLOSED,
    TASK_ID,
    UNKNOWN_OPERATORS_FAIL_CLOSED,
    AutonomyDisposition,
    AutonomyRiskClass,
    AuthorityImpact,
    ExpectedEffectKind,
    MaximumScope,
    MigrationPhase,
    OperatorCatalog,
    OperatorKind,
    OperatorMigration,
    OperatorRollback,
    PreconditionKind,
    ProofObligationKind,
    PublicApiImpact,
    RefactorOperator,
    RefactorOperatorAuthorityError,
    RefactorOperatorError,
    RollbackAction,
    StateImpact,
    ValidationObligationKind,
    assert_within_maximum_scope,
    autonomy_classification_map,
    autonomy_rank,
    disposition_for_risk,
    operator_catalog,
    operator_for,
    operator_grammar,
    refuse_arbitrary_payload,
    refuse_ceiling_raise,
    refuse_execution,
    refuse_gate_reduction,
    refuse_scope_expansion,
    refuse_self_authorization,
    refuse_self_promotion,
    refuse_unknown_operator,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-012-fixture"
_INTERNAL_PATH = (
    "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/refactor_operators.py"
)
_CONTRACT = cid_for_dag_json(
    {
        "schema": "ipfs_accelerate_py/agent-supervisor/contract-candidate@1",
        "subject": "n-symbol",
        "tree": _TREE,
    }
)


def _candidate_for(
    kind: OperatorKind = OperatorKind.EXTRACT_PURE_FUNCTION,
    **kwargs,
) -> RefactorCandidate:
    operator = operator_for(kind)
    defaults = {
        "repository_tree": _TREE,
        "contract_identity": _CONTRACT,
        "target_node_ids": ("n-symbol",),
        "target_paths": (_INTERNAL_PATH,),
        "freshness": _FRESHNESS,
    }
    defaults.update(kwargs)
    return declare_refactor_candidate(operator, **defaults)


def test_closed_vocabulary_and_authority_invariants() -> None:
    assert REFACTOR_OPERATOR_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/refactor-operator@1"
    )
    assert REFACTOR_OPERATOR_SCHEMA.endswith("refactor-operator@1")
    assert REFACTOR_OPERATOR_VERSION == 1
    assert REFACTOR_OPERATOR_EVIDENCE == "pcar/refactor-operator@1"
    assert OPERATOR_CATALOG_SCHEMA.endswith("refactor-operator-catalog@1")
    assert REFACTOR_CANDIDATE_SCHEMA.endswith("refactor-candidate@1")
    assert REFACTOR_CANDIDATE_VERSION == 1
    assert REFACTOR_CANDIDATE_EVIDENCE == "pcar/refactor-candidate@1"
    assert EXTRACTOR_IDENTITY == "pcar-012-refactor-operator-grammar"
    assert TASK_ID == "PCAR-012"
    assert DEFAULT_FRESHNESS == "pcar-012-operator-grammar"
    assert EFFECT_CLASS == "internal_pure_contract_addition"
    assert OPERATOR_CAN_AUTHORIZE_EXECUTION is False
    assert OPERATOR_CAN_REDUCE_GATES is False
    assert OPERATOR_CAN_SELF_PROMOTE is False
    assert OPERATOR_CAN_RAISE_CEILING is False
    assert OPERATOR_CAN_EXPAND_SCOPE is False
    assert OPERATOR_CAN_ADMIT_ARBITRARY_PAYLOADS is False
    assert DECLARATIONS_ARE_IMMUTABLE is True
    assert UNKNOWN_OPERATORS_FAIL_CLOSED is True
    assert INCOMPLETE_DECLARATIONS_FAIL_CLOSED is True
    assert SCOPE_EXPANSION_FAILS_CLOSED is True
    assert CANDIDATE_CAN_AUTHORIZE_EXECUTION is False
    assert CANDIDATE_CAN_REDUCE_GATES is False
    assert CANDIDATE_CAN_SELF_PROMOTE is False
    assert CANDIDATE_CAN_RAISE_CEILING is False
    assert CANDIDATE_CAN_EXPAND_SCOPE is False
    assert CANDIDATE_IDENTITY_BINDS_TREE is True
    assert CANDIDATE_IDENTITY_BINDS_CONTRACT is True
    assert CANDIDATE_IDENTITY_BINDS_EFFECTS is True
    assert tuple(item.value for item in INITIAL_OPERATORS) == (
        "extract_module",
        "extract_interface",
        "extract_pure_function",
        "move_state_to_owner",
        "introduce_dependency_inversion",
        "replace_direct_call_with_typed_service",
        "generate_adapter",
        "generate_compatibility_shim",
        "quarantine_legacy_path",
        "quarantine_simulation_path",
        "replace_boolean_with_closed_outcome",
        "replace_dynamic_registry_with_typed_catalog",
        "replace_eager_import_with_lazy_capability",
        "consolidate_error_vocabulary",
        "consolidate_receipt_producer",
        "consolidate_capability_authority",
        "remove_confirmed_dead_code",
        "split_monolith_by_authority",
        "move_generated_projection_out_of_source_authority",
        "deprecate_public_symbol",
        "remove_deprecated_symbol_after_gate",
    )
    assert REQUIRED_OPERATORS == INITIAL_OPERATORS
    assert CLOSED_OPERATOR_KINDS == {item.value for item in OperatorKind}
    assert CLOSED_AUTONOMY_DISPOSITIONS == {"automatic", "proposal_only", "always_human"}
    assert tuple(item.value for item in AUTOMATIC_OPERATOR_CLASSES) == (
        "pure_module_extraction",
        "generated_projection_regeneration",
        "lazy_import_conversion",
        "internal_adapter_generation",
        "closed_result_type_migration",
        "confirmed_dead_internal_code_removal",
        "test_fixture_relocation",
        "simulation_namespace_relocation",
    )
    assert CLOSED_AUTOMATIC_OPERATOR_CLASSES == {
        item.value for item in AUTOMATIC_OPERATOR_CLASSES
    }
    assert CLOSED_PROPOSAL_ONLY_RISK_CLASSES == {
        item.value for item in PROPOSAL_ONLY_RISK_CLASSES
    }
    assert CLOSED_ALWAYS_HUMAN_RISK_CLASSES == {
        item.value for item in ALWAYS_HUMAN_RISK_CLASSES
    }
    assert CLOSED_AUTONOMY_RISK_CLASSES == {item.value for item in AutonomyRiskClass}
    assert (
        CLOSED_AUTOMATIC_OPERATOR_CLASSES
        | CLOSED_PROPOSAL_ONLY_RISK_CLASSES
        | CLOSED_ALWAYS_HUMAN_RISK_CLASSES
        == CLOSED_AUTONOMY_RISK_CLASSES
    )
    assert CLOSED_AUTHORITY_IMPACTS == {item.value for item in AuthorityImpact}
    assert CLOSED_PUBLIC_API_IMPACTS == {item.value for item in PublicApiImpact}
    assert CLOSED_STATE_IMPACTS == {item.value for item in StateImpact}
    assert CLOSED_EXPECTED_EFFECTS == {item.value for item in ExpectedEffectKind}
    assert CLOSED_PRECONDITIONS == {item.value for item in PreconditionKind}
    assert CLOSED_VALIDATION_OBLIGATIONS == {
        item.value for item in ValidationObligationKind
    }
    assert CLOSED_PROOF_OBLIGATIONS == {item.value for item in ProofObligationKind}
    assert CLOSED_MIGRATION_PHASES == {item.value for item in MigrationPhase}
    assert CLOSED_ROLLBACK_ACTIONS == {item.value for item in RollbackAction}
    with pytest.raises(ValueError):
        OperatorKind("SHELL_SCRIPT")
    with pytest.raises(ValueError):
        AutonomyDisposition("self_approved")
    with pytest.raises(ValueError):
        AutonomyRiskClass("unbounded_rewrite")
    with pytest.raises(ValueError):
        AuthorityImpact("transfer")
    with pytest.raises(ValueError):
        PublicApiImpact("break")
    with pytest.raises(ValueError):
        StateImpact("indefinite_dual_write")
    with pytest.raises(ValueError):
        ExpectedEffectKind("run_shell")
    with pytest.raises(RefactorOperatorError, match="unsupported refactor-operator kind"):
        refuse_unknown_operator("SHELL_SCRIPT")
    with pytest.raises(RefactorOperatorError, match="unsupported refactor-operator kind"):
        operator_for("SHELL_SCRIPT")


def test_complete_declarations_cover_every_required_operator() -> None:
    catalog = operator_catalog()
    grammar = operator_grammar()
    assert catalog.content_identity == grammar.content_identity
    assert catalog.covers_initial_operators is True
    assert catalog.declarations_are_immutable is True
    assert catalog.effect_class == EFFECT_CLASS
    assert tuple(item.kind for item in catalog.operators) == INITIAL_OPERATORS
    assert tuple(item.kind for item in INITIAL_OPERATOR_DECLARATIONS) == INITIAL_OPERATORS
    required = {
        "preconditions",
        "target_kinds",
        "expected_effects",
        "authority_impact",
        "public_api_impact",
        "state_impact",
        "autonomy_disposition",
        "risk_class",
        "migration",
        "rollback",
        "validation",
        "proofs",
        "maximum_scope",
        "preserved_invariants",
    }
    for operator in catalog.operators:
        payload = operator.to_dict()
        assert required <= set(payload)
        assert set(REQUIRED_PRECONDITIONS) <= set(operator.preconditions)
        assert set(REQUIRED_VALIDATION_OBLIGATIONS) <= set(operator.validation)
        assert set(REQUIRED_PROOF_OBLIGATIONS) <= set(operator.proofs)
        assert set(operator.preserved_invariants) == set(NON_COMPENSABLE_INVARIANTS)
        assert operator.target_kinds
        assert operator.expected_effects
        assert operator.maximum_scope.path_prefixes
        assert operator.maximum_scope.max_paths >= 1
        assert operator.maximum_scope.max_symbols >= 1
        assert operator.maximum_scope.allows_sibling_repositories is False
        assert operator.maximum_scope.allows_arbitrary_payloads is False
        assert operator.maximum_scope.allows_scope_expansion is False
        assert operator.maximum_scope.allows_network is False
        assert operator.migration.phases[-1] is MigrationPhase.VALIDATE_AND_SEAL
        assert operator.migration.transfers_authority is False
        assert operator.rollback.applied_effects is False
        assert operator.rollback.restores_tree is True
        assert operator.can_authorize_execution is False
        assert operator.can_reduce_gates is False
        assert operator.can_self_promote is False
        assert operator.can_raise_ceiling is False
        assert operator.autonomy_disposition is disposition_for_risk(operator.risk_class)
        assert set(operator.maximum_scope.target_kinds) == set(operator.target_kinds)
        round_trip = RefactorOperator.from_mapping(payload)
        assert round_trip.content_identity == operator.content_identity
        assert json.loads(operator.to_json())["content_identity"] == operator.content_identity
        validate_cid(operator.content_identity, codecs=("dag-json",))
        assert cid_for_dag_json(operator._identity_payload()) == operator.content_identity
        reconstructed = operator_for(operator.kind.value)
        assert reconstructed.content_identity == operator.content_identity


def test_autonomy_classification_map_is_closed_and_complete() -> None:
    catalog = operator_catalog()
    mapping = autonomy_classification_map()
    assert set(mapping) == CLOSED_OPERATOR_KINDS
    assert dict(catalog.autonomy_classification) == mapping
    assert mapping[OperatorKind.EXTRACT_MODULE.value] == "pure_module_extraction"
    assert mapping[OperatorKind.EXTRACT_PURE_FUNCTION.value] == "pure_module_extraction"
    assert mapping[OperatorKind.GENERATE_ADAPTER.value] == "internal_adapter_generation"
    assert (
        mapping[OperatorKind.REPLACE_DIRECT_CALL_WITH_TYPED_SERVICE.value]
        == "internal_adapter_generation"
    )
    assert (
        mapping[OperatorKind.REPLACE_EAGER_IMPORT_WITH_LAZY_CAPABILITY.value]
        == "lazy_import_conversion"
    )
    assert (
        mapping[OperatorKind.REPLACE_BOOLEAN_WITH_CLOSED_OUTCOME.value]
        == "closed_result_type_migration"
    )
    assert (
        mapping[OperatorKind.REMOVE_CONFIRMED_DEAD_CODE.value]
        == "confirmed_dead_internal_code_removal"
    )
    assert (
        mapping[OperatorKind.QUARANTINE_SIMULATION_PATH.value]
        == "simulation_namespace_relocation"
    )
    assert (
        mapping[OperatorKind.MOVE_GENERATED_PROJECTION_OUT_OF_SOURCE_AUTHORITY.value]
        == "generated_projection_regeneration"
    )
    assert mapping[OperatorKind.MOVE_STATE_TO_OWNER.value] == "state"
    assert mapping[OperatorKind.QUARANTINE_LEGACY_PATH.value] == "legacy"
    assert mapping[OperatorKind.CONSOLIDATE_RECEIPT_PRODUCER.value] == "receipt"
    assert mapping[OperatorKind.CONSOLIDATE_CAPABILITY_AUTHORITY.value] == "authorization"
    assert mapping[OperatorKind.DEPRECATE_PUBLIC_SYMBOL.value] == "public_api"
    automatic = [
        item for item in catalog.operators if item.autonomy_disposition is AutonomyDisposition.AUTOMATIC
    ]
    proposal = [
        item
        for item in catalog.operators
        if item.autonomy_disposition is AutonomyDisposition.PROPOSAL_ONLY
    ]
    human = [
        item
        for item in catalog.operators
        if item.autonomy_disposition is AutonomyDisposition.ALWAYS_HUMAN
    ]
    assert automatic
    assert proposal
    assert human
    for item in automatic:
        assert item.risk_class in AUTOMATIC_OPERATOR_CLASSES
        assert item.public_api_impact is PublicApiImpact.INTERNAL
        assert item.state_impact in {StateImpact.NONE, StateImpact.READ_ONLY}
        assert item.authority_impact is not AuthorityImpact.CONSOLIDATE
    for item in human:
        assert item.risk_class in ALWAYS_HUMAN_RISK_CLASSES
        assert autonomy_rank(item.autonomy_disposition) > autonomy_rank(
            AutonomyDisposition.PROPOSAL_ONLY
        )
    authority = operator_for(OperatorKind.CONSOLIDATE_CAPABILITY_AUTHORITY)
    assert authority.authority_impact is AuthorityImpact.CONSOLIDATE
    assert authority.autonomy_disposition is AutonomyDisposition.ALWAYS_HUMAN
    state = operator_for(OperatorKind.MOVE_STATE_TO_OWNER)
    assert state.state_impact is StateImpact.MOVE_TO_OWNER
    assert state.migration.mutates_state is True
    assert state.maximum_scope.allows_state_stores is True
    assert MigrationPhase.BOUNDED_DUAL_WRITE in state.migration.phases
    assert MigrationPhase.RETIRE in state.migration.phases


def test_unknown_fields_and_incomplete_declarations_fail_closed() -> None:
    operator = operator_for(OperatorKind.EXTRACT_MODULE)
    payload = operator.to_dict()
    payload["script"] = "rm -rf /"
    with pytest.raises(RefactorOperatorAuthorityError, match="arbitrary"):
        RefactorOperator.from_mapping(payload)
    payload = operator.to_dict()
    payload["aesthetic"] = 1
    with pytest.raises(RefactorOperatorError, match="unknown"):
        RefactorOperator.from_mapping(payload)
    incomplete = operator.to_dict()
    del incomplete["rollback"]
    with pytest.raises(RefactorOperatorError, match="missing"):
        RefactorOperator.from_mapping(incomplete)
    missing_pre = [
        item
        for item in operator.preconditions
        if item is not PreconditionKind.ACCEPTED_OWNERSHIP
    ]
    with pytest.raises(RefactorOperatorError, match="preconditions"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=tuple(missing_pre),
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            public_api_impact=operator.public_api_impact,
            state_impact=operator.state_impact,
            autonomy_disposition=operator.autonomy_disposition,
            risk_class=operator.risk_class,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
        )
    missing_validation = [
        item
        for item in operator.validation
        if item is not ValidationObligationKind.ROLLBACK_REHEARSAL
    ]
    with pytest.raises(RefactorOperatorError, match="validation"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            public_api_impact=operator.public_api_impact,
            state_impact=operator.state_impact,
            autonomy_disposition=operator.autonomy_disposition,
            risk_class=operator.risk_class,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=tuple(missing_validation),
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
        )
    with pytest.raises(RefactorOperatorError, match="applied effects"):
        OperatorRollback(applied_effects=True)
    with pytest.raises(RefactorOperatorError, match="restore"):
        OperatorRollback(restores_tree=False)
    with pytest.raises(RefactorOperatorError, match="validate_and_seal"):
        OperatorMigration(phases=(MigrationPhase.DECLARE,))
    with pytest.raises(RefactorOperatorError, match="final migration phase"):
        OperatorMigration(
            phases=(MigrationPhase.VALIDATE_AND_SEAL, MigrationPhase.DECLARE)
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="transfer"):
        OperatorMigration(
            phases=(MigrationPhase.DECLARE, MigrationPhase.VALIDATE_AND_SEAL),
            transfers_authority=True,
        )
    with pytest.raises(ValueError):
        OperatorKind("open_rewrite")
    with pytest.raises(RefactorOperatorError, match="unsupported"):
        RefactorOperator.from_mapping({**operator.to_dict(), "kind": "open_rewrite"})


def test_maximum_scope_rejects_expansion_and_siblings() -> None:
    operator = operator_for(OperatorKind.EXTRACT_PURE_FUNCTION)
    sibling = MaximumScope(
        path_prefixes=("ipfs_accelerate_py/",),
        target_kinds=operator.target_kinds,
        max_paths=operator.maximum_scope.max_paths,
        max_symbols=operator.maximum_scope.max_symbols,
    )
    with pytest.raises(RefactorOperatorError, match="sibling"):
        MaximumScope(
            path_prefixes=("ipfs_kit_py/",),
            target_kinds=operator.target_kinds,
            max_paths=1,
            max_symbols=1,
        )
    with pytest.raises(RefactorOperatorError, match="sibling"):
        MaximumScope(
            path_prefixes=("ipfs_datasets_py/logic.py",),
            target_kinds=operator.target_kinds,
            max_paths=1,
            max_symbols=1,
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="sibling"):
        MaximumScope(
            path_prefixes=("ipfs_accelerate_py/",),
            target_kinds=operator.target_kinds,
            max_paths=1,
            max_symbols=1,
            allows_sibling_repositories=True,
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="payload"):
        MaximumScope(
            path_prefixes=("ipfs_accelerate_py/",),
            target_kinds=operator.target_kinds,
            max_paths=1,
            max_symbols=1,
            allows_arbitrary_payloads=True,
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="maximum scope"):
        MaximumScope(
            path_prefixes=("ipfs_accelerate_py/",),
            target_kinds=operator.target_kinds,
            max_paths=1,
            max_symbols=1,
            allows_scope_expansion=True,
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="network"):
        MaximumScope(
            path_prefixes=("ipfs_accelerate_py/",),
            target_kinds=operator.target_kinds,
            max_paths=1,
            max_symbols=1,
            allows_network=True,
        )
    expanded = MaximumScope(
        path_prefixes=("ipfs_accelerate_py/", "docs/"),
        target_kinds=operator.target_kinds,
        max_paths=operator.maximum_scope.max_paths,
        max_symbols=operator.maximum_scope.max_symbols,
    )
    with pytest.raises(RefactorOperatorAuthorityError, match="maximum scope"):
        assert_within_maximum_scope(operator.maximum_scope, expanded)
    wider = MaximumScope(
        path_prefixes=operator.maximum_scope.path_prefixes,
        target_kinds=operator.target_kinds,
        max_paths=operator.maximum_scope.max_paths + 1,
        max_symbols=operator.maximum_scope.max_symbols,
    )
    with pytest.raises(RefactorOperatorAuthorityError, match="maximum scope"):
        assert_within_maximum_scope(operator.maximum_scope, wider)
    public = MaximumScope(
        path_prefixes=operator.maximum_scope.path_prefixes,
        target_kinds=operator.target_kinds,
        max_paths=operator.maximum_scope.max_paths,
        max_symbols=operator.maximum_scope.max_symbols,
        allows_public_surface=True,
    )
    with pytest.raises(RefactorOperatorAuthorityError, match="maximum scope"):
        assert_within_maximum_scope(operator.maximum_scope, public)
    with pytest.raises(RefactorCandidateAuthorityError, match="maximum scope"):
        _candidate_for(target_paths=("docs/architecture/README.md",))
    with pytest.raises(RefactorCandidateAuthorityError, match="sibling"):
        _candidate_for(target_paths=("ipfs_kit_py/ipfs_kit.py",))
    with pytest.raises(RefactorCandidateAuthorityError, match="maximum scope"):
        _candidate_for(
            maximum_scope=MaximumScope(
                path_prefixes=operator.maximum_scope.path_prefixes,
                target_kinds=operator.target_kinds,
                max_paths=operator.maximum_scope.max_paths,
                max_symbols=operator.maximum_scope.max_symbols + 8,
            )
        )
    assert operator.maximum_scope.covers_path(_INTERNAL_PATH) is True
    assert operator.maximum_scope.contains(sibling) is True
    with pytest.raises(RefactorOperatorAuthorityError, match="maximum scope"):
        operator.expand_scope()


def test_self_authorization_and_promotion_are_rejected() -> None:
    catalog = operator_catalog()
    operator = operator_for(OperatorKind.GENERATE_ADAPTER)
    candidate = _candidate_for(OperatorKind.GENERATE_ADAPTER)
    with pytest.raises(RefactorOperatorAuthorityError, match="authorize"):
        operator.authorize()
    with pytest.raises(RefactorOperatorAuthorityError, match="authorize"):
        catalog.authorize()
    with pytest.raises(RefactorOperatorAuthorityError, match="authorize"):
        candidate.authorize()
    with pytest.raises(RefactorOperatorAuthorityError, match="promote"):
        operator.promote()
    with pytest.raises(RefactorOperatorAuthorityError, match="promote"):
        catalog.promote()
    with pytest.raises(RefactorOperatorAuthorityError, match="promote"):
        candidate.promote()
    with pytest.raises(RefactorOperatorAuthorityError, match="ceiling"):
        operator.raise_ceiling()
    with pytest.raises(RefactorOperatorAuthorityError, match="ceiling"):
        catalog.raise_ceiling()
    with pytest.raises(RefactorOperatorAuthorityError, match="ceiling"):
        candidate.raise_ceiling()
    with pytest.raises(RefactorOperatorAuthorityError, match="gates"):
        operator.reduce_gates()
    with pytest.raises(RefactorOperatorAuthorityError, match="execute"):
        operator.execute()
    with pytest.raises(RefactorOperatorAuthorityError, match="apply"):
        operator.apply()
    with pytest.raises(RefactorOperatorAuthorityError, match="apply"):
        catalog.apply()
    with pytest.raises(RefactorOperatorAuthorityError, match="apply"):
        candidate.apply()
    with pytest.raises(RefactorOperatorAuthorityError, match="execute"):
        candidate.execute()
    with pytest.raises(RefactorOperatorAuthorityError):
        refuse_self_authorization("authorize")
    with pytest.raises(RefactorOperatorAuthorityError):
        refuse_self_promotion("promote")
    with pytest.raises(RefactorOperatorAuthorityError):
        refuse_ceiling_raise("raise")
    with pytest.raises(RefactorOperatorAuthorityError):
        refuse_gate_reduction("reduce")
    with pytest.raises(RefactorOperatorAuthorityError):
        refuse_execution("execute")
    with pytest.raises(RefactorOperatorAuthorityError):
        refuse_arbitrary_payload("script")
    with pytest.raises(RefactorOperatorAuthorityError):
        refuse_scope_expansion("expand")
    with pytest.raises(RefactorOperatorAuthorityError, match="authorize"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            public_api_impact=operator.public_api_impact,
            state_impact=operator.state_impact,
            autonomy_disposition=operator.autonomy_disposition,
            risk_class=operator.risk_class,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
            can_authorize_execution=True,
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="ceiling"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=operator.target_kinds and operator.expected_effects,
            authority_impact=operator.authority_impact,
            public_api_impact=operator.public_api_impact,
            state_impact=operator.state_impact,
            autonomy_disposition=operator.autonomy_disposition,
            risk_class=operator.risk_class,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
            can_raise_ceiling=True,
        )
    promoted = dict(catalog.to_dict())
    promoted["can_self_promote"] = True
    with pytest.raises(RefactorOperatorAuthorityError, match="promote"):
        OperatorCatalog.from_mapping(promoted)
    with pytest.raises(RefactorOperatorAuthorityError, match="ceiling"):
        _candidate_for(
            OperatorKind.DEPRECATE_PUBLIC_SYMBOL,
            autonomy_disposition=AutonomyDisposition.AUTOMATIC,
            risk_class=AutonomyRiskClass.PURE_MODULE_EXTRACTION,
        )
    dropped = [
        item
        for item in operator.preserved_invariants
        if item != "NoArchitectureCandidateSelfPromotion"
    ]
    with pytest.raises(RefactorOperatorAuthorityError, match="gates"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            public_api_impact=operator.public_api_impact,
            state_impact=operator.state_impact,
            autonomy_disposition=operator.autonomy_disposition,
            risk_class=operator.risk_class,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
            preserved_invariants=tuple(dropped),
        )


def test_canonical_candidate_identity_binds_tree_contract_and_effects() -> None:
    first = _candidate_for()
    second = _candidate_for()
    operator = operator_for(OperatorKind.EXTRACT_PURE_FUNCTION)
    assert first.content_identity == second.content_identity
    assert first.repository_tree == _TREE
    assert first.contract_identity == _CONTRACT
    assert first.operator_identity == operator.content_identity
    assert first.operator_kind is OperatorKind.EXTRACT_PURE_FUNCTION
    assert first.expected_effects == operator.expected_effects
    payload = first._identity_payload()
    assert payload["repository_tree"] == _TREE
    assert payload["contract_identity"] == _CONTRACT
    assert payload["expected_effects"] == [item.value for item in operator.expected_effects]
    assert cid_for_dag_json(payload) == first.content_identity
    validate_cid(first.content_identity, codecs=("dag-json",))
    round_trip = RefactorCandidate.from_mapping(first.to_dict())
    assert round_trip.content_identity == first.content_identity
    assert RefactorCandidate.from_json(first.to_json()).content_identity == first.content_identity
    other_tree = declare_refactor_candidate(
        operator,
        repository_tree="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        contract_identity=_CONTRACT,
        target_node_ids=("n-symbol",),
        target_paths=(_INTERNAL_PATH,),
        freshness=_FRESHNESS,
    )
    assert other_tree.content_identity != first.content_identity
    other_contract = declare_refactor_candidate(
        operator,
        repository_tree=_TREE,
        contract_identity=cid_for_dag_json({"schema": "other", "tree": _TREE}),
        target_node_ids=("n-symbol",),
        target_paths=(_INTERNAL_PATH,),
        freshness=_FRESHNESS,
    )
    assert other_contract.content_identity != first.content_identity
    other_effects = declare_refactor_candidate(
        OperatorKind.EXTRACT_MODULE,
        repository_tree=_TREE,
        contract_identity=_CONTRACT,
        target_node_ids=("n-module",),
        target_paths=(_INTERNAL_PATH,),
        freshness=_FRESHNESS,
        expected_effects=(ExpectedEffectKind.EXTRACT,),
        target_kinds=(NodeKind.MODULE, NodeKind.FILE, NodeKind.SYMBOL),
    )
    assert other_effects.operator_kind is OperatorKind.EXTRACT_MODULE
    assert other_effects.content_identity != first.content_identity
    catalog = operator_catalog()
    catalog_round_trip = OperatorCatalog.from_json(catalog.to_json())
    assert catalog_round_trip.content_identity == catalog.content_identity
    assert operator_catalog().content_identity == catalog.content_identity
    mismatched = dict(first.to_dict())
    mismatched["content_identity"] = catalog.content_identity
    with pytest.raises(RefactorCandidateError, match="content identity"):
        RefactorCandidate.from_mapping(mismatched)


def test_candidate_cannot_expand_effects_or_drop_operator_obligations() -> None:
    operator = operator_for(OperatorKind.EXTRACT_PURE_FUNCTION)
    with pytest.raises(RefactorCandidateError, match="expected_effects"):
        _candidate_for(expected_effects=(ExpectedEffectKind.EXTRACT, ExpectedEffectKind.REMOVE))
    with pytest.raises(RefactorCandidateError, match="target_kinds"):
        _candidate_for(target_kinds=(NodeKind.MODULE, NodeKind.STATE))
    with pytest.raises(RefactorCandidateError, match="authority_impact"):
        _candidate_for(authority_impact=AuthorityImpact.CONSOLIDATE)
    with pytest.raises(RefactorCandidateError, match="public_api_impact"):
        _candidate_for(public_api_impact=PublicApiImpact.DEPRECATE)
    with pytest.raises(RefactorCandidateError, match="state_impact"):
        _candidate_for(state_impact=StateImpact.MOVE_TO_OWNER)
    dropped_proofs = tuple(
        item for item in operator.proofs if item is not ProofObligationKind.NO_SELF_PROMOTION
    )
    with pytest.raises(RefactorOperatorAuthorityError, match="gates"):
        _candidate_for(proofs=dropped_proofs)
    unknown = dict(_candidate_for().to_dict())
    unknown["payload"] = "echo hi"
    with pytest.raises(RefactorCandidateAuthorityError, match="payload"):
        RefactorCandidate.from_mapping(unknown)
    missing = dict(_candidate_for().to_dict())
    del missing["contract_identity"]
    with pytest.raises(RefactorCandidateError, match="missing"):
        RefactorCandidate.from_mapping(missing)
    with pytest.raises(RefactorCandidateError, match="dag-json"):
        _candidate_for(contract_identity="not-a-cid")
    with pytest.raises(RefactorOperatorError, match="unsupported refactor-operator kind"):
        declare_refactor_candidate(
            "SHELL_SCRIPT",
            repository_tree=_TREE,
            contract_identity=_CONTRACT,
            target_node_ids=("n-symbol",),
            target_paths=(_INTERNAL_PATH,),
            freshness=_FRESHNESS,
        )


def test_automatic_operators_cannot_claim_public_or_state_authority() -> None:
    operator = operator_for(OperatorKind.EXTRACT_MODULE)
    with pytest.raises(RefactorOperatorAuthorityError, match="public API"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            public_api_impact=PublicApiImpact.DEPRECATE,
            state_impact=operator.state_impact,
            autonomy_disposition=operator.autonomy_disposition,
            risk_class=operator.risk_class,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=MaximumScope(
                path_prefixes=operator.maximum_scope.path_prefixes,
                target_kinds=operator.target_kinds,
                max_paths=operator.maximum_scope.max_paths,
                max_symbols=operator.maximum_scope.max_symbols,
                allows_public_surface=True,
            ),
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="authoritative state"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            public_api_impact=operator.public_api_impact,
            state_impact=StateImpact.MOVE_TO_OWNER,
            autonomy_disposition=operator.autonomy_disposition,
            risk_class=operator.risk_class,
            migration=OperatorMigration(
                phases=operator.migration.phases,
                mutates_state=True,
            ),
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=MaximumScope(
                path_prefixes=operator.maximum_scope.path_prefixes,
                target_kinds=operator.target_kinds,
                max_paths=operator.maximum_scope.max_paths,
                max_symbols=operator.maximum_scope.max_symbols,
                allows_state_stores=True,
            ),
        )
    with pytest.raises(RefactorOperatorError, match="autonomy disposition"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            public_api_impact=operator.public_api_impact,
            state_impact=operator.state_impact,
            autonomy_disposition=AutonomyDisposition.ALWAYS_HUMAN,
            risk_class=operator.risk_class,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
        )
    human = operator_for(OperatorKind.CONSOLIDATE_CAPABILITY_AUTHORITY)
    with pytest.raises(RefactorOperatorAuthorityError, match="human approval"):
        RefactorOperator(
            kind=human.kind,
            preconditions=human.preconditions,
            target_kinds=human.target_kinds,
            expected_effects=human.expected_effects,
            authority_impact=AuthorityImpact.CONSOLIDATE,
            public_api_impact=human.public_api_impact,
            state_impact=human.state_impact,
            autonomy_disposition=AutonomyDisposition.PROPOSAL_ONLY,
            risk_class=AutonomyRiskClass.PUBLIC_API,
            migration=human.migration,
            rollback=human.rollback,
            validation=human.validation,
            proofs=human.proofs,
            maximum_scope=human.maximum_scope,
        )


def test_catalog_rejects_open_vocabulary_and_identity_tampering() -> None:
    catalog = operator_catalog()
    payload = catalog.to_dict()
    payload["operators"] = list(payload["operators"])[:-1]
    with pytest.raises(RefactorOperatorError, match="initial vocabulary"):
        OperatorCatalog.from_mapping(payload)
    payload = catalog.to_dict()
    payload["effect_class"] = "arbitrary_script"
    with pytest.raises(RefactorOperatorError, match="effect class"):
        OperatorCatalog.from_mapping(payload)
    payload = catalog.to_dict()
    payload["content_identity"] = cid_for_dag_json({"tampered": True})
    with pytest.raises(RefactorOperatorError, match="content identity"):
        OperatorCatalog.from_mapping(payload)
    payload = catalog.to_dict()
    payload["shell"] = "true"
    with pytest.raises(RefactorOperatorAuthorityError, match="arbitrary"):
        OperatorCatalog.from_mapping(payload)
    assert catalog.operator(OperatorKind.EXTRACT_INTERFACE).kind is OperatorKind.EXTRACT_INTERFACE
    candidate = _candidate_for(OperatorKind.QUARANTINE_SIMULATION_PATH)
    assert candidate.operator().kind is OperatorKind.QUARANTINE_SIMULATION_PATH
    assert candidate.rollback.applied_effects is False
    assert candidate.maximum_scope.allows_scope_expansion is False
