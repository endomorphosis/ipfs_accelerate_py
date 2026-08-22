"""Hermetic PCAR-012 closed refactor-operator grammar tests."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.candidate import (
    CANDIDATE_CAN_ADMIT_SCRIPT_PAYLOAD,
    CANDIDATE_CAN_AUTHORIZE_EXECUTION,
    CANDIDATE_CAN_RAISE_AUTONOMY_CEILING,
    CANDIDATE_CAN_SELF_PROMOTE,
    CANDIDATE_SCOPE_SCHEMA,
    REFACTOR_CANDIDATE_EVIDENCE,
    REFACTOR_CANDIDATE_SCHEMA,
    CandidateScope,
    RefactorCandidate,
    RefactorCandidateError,
    canonical_candidate_identity,
    declare_refactor_candidate,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import NodeKind
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.refactor_operators import (
    AUTOMATIC_RISK_CLASSES,
    CLOSED_API_IMPACTS,
    CLOSED_AUTHORITY_IMPACTS,
    CLOSED_AUTONOMY_CEILINGS,
    CLOSED_EXPECTED_EFFECTS,
    CLOSED_OPERATORS,
    CLOSED_PRECONDITIONS,
    CLOSED_PROHIBITED_EFFECTS,
    CLOSED_PROOF_OBLIGATIONS,
    CLOSED_RISK_CLASSES,
    CLOSED_STATE_IMPACTS,
    CLOSED_VALIDATION_OBLIGATIONS,
    DEFAULT_FRESHNESS,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    HUMAN_APPROVAL_RISK_CLASSES,
    INITIAL_OPERATORS,
    OPERATOR_CAN_ADMIT_SCRIPT_PAYLOAD,
    OPERATOR_CAN_AUTHORIZE_EXECUTION,
    OPERATOR_CAN_EXPAND_SCOPE,
    OPERATOR_CAN_RAISE_AUTONOMY_CEILING,
    OPERATOR_CAN_REDUCE_GATES,
    OPERATOR_CAN_SELF_PROMOTE,
    OPERATOR_CATALOG,
    PROPOSAL_ONLY_RISK_CLASSES,
    PROTECTED_PATHS,
    REFACTOR_OPERATOR_EVIDENCE,
    REFACTOR_OPERATOR_SCHEMA,
    REQUIRED_OPERATOR_DECLARATION_FIELDS,
    REQUIRED_OPERATORS,
    REQUIRED_PRECONDITIONS,
    REQUIRED_PROHIBITED_EFFECTS,
    REQUIRED_PROOF_OBLIGATIONS,
    REQUIRED_VALIDATION_OBLIGATIONS,
    STATE_MIGRATION_PHASES,
    TASK_ID,
    ApiImpact,
    AuthorityImpact,
    AutonomyCeiling,
    ExpectedEffect,
    MaximumScope,
    OperatorCatalog,
    OperatorKind,
    OperatorMigration,
    OperatorPrecondition,
    OperatorRiskClass,
    OperatorRollback,
    OperatorValidation,
    ProhibitedEffect,
    ProofObligation,
    RefactorOperator,
    RefactorOperatorAuthorityError,
    RefactorOperatorError,
    RollbackAction,
    StateImpact,
    ValidationObligation,
    autonomy_ceiling_for_risk,
    autonomy_classification_map,
    effects_identity,
    get_operator,
    refuse_autonomy_ceiling_raise,
    refuse_gate_reduction,
    refuse_scope_expansion,
    refuse_script_payload,
    refuse_self_authorization,
    refuse_self_promotion,
    risk_classification_map,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_CONTRACT = cid_for_dag_json(
    {
        "schema": "ipfs_accelerate_py/agent-supervisor/contract-candidate@1",
        "version": 1,
        "subject": "n-module",
    }
)
_PATH = "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/refactor_operators.py"


def _candidate(
    kind: OperatorKind = OperatorKind.EXTRACT_MODULE,
    *,
    repository_tree: str = _TREE,
    contract_identity: str = _CONTRACT,
    paths: tuple[str, ...] | None = None,
    symbol_count: int | None = None,
    package_count: int = 1,
    target_kinds: tuple[NodeKind, ...] | None = None,
) -> RefactorCandidate:
    operator = get_operator(kind)
    declared_paths = paths if paths is not None else (_PATH,)
    return declare_refactor_candidate(
        operator_kind=kind,
        repository_tree=repository_tree,
        contract_identity=contract_identity,
        targets=("n-module",),
        paths=declared_paths,
        node_ids=("n-module",),
        symbol_count=symbol_count if symbol_count is not None else min(3, operator.maximum_scope.max_symbols),
        package_count=package_count,
        target_kinds=target_kinds if target_kinds is not None else (operator.target_kinds[0],),
    )


def test_closed_vocabulary_and_catalog_invariants() -> None:
    assert TASK_ID == "PCAR-012"
    assert EXTRACTOR_IDENTITY == "pcar-012-refactor-operator-grammar"
    assert REFACTOR_OPERATOR_SCHEMA.endswith("refactor-operator@1")
    assert REFACTOR_OPERATOR_EVIDENCE == "pcar/refactor-operator@1"
    assert REFACTOR_CANDIDATE_SCHEMA.endswith("refactor-candidate@1")
    assert REFACTOR_CANDIDATE_EVIDENCE == "pcar/refactor-candidate@1"
    assert EFFECT_CLASS == "internal_pure_contract_addition"
    assert OPERATOR_CAN_AUTHORIZE_EXECUTION is False
    assert OPERATOR_CAN_REDUCE_GATES is False
    assert OPERATOR_CAN_SELF_PROMOTE is False
    assert OPERATOR_CAN_RAISE_AUTONOMY_CEILING is False
    assert OPERATOR_CAN_ADMIT_SCRIPT_PAYLOAD is False
    assert OPERATOR_CAN_EXPAND_SCOPE is False
    assert CANDIDATE_CAN_AUTHORIZE_EXECUTION is False
    assert CANDIDATE_CAN_SELF_PROMOTE is False
    assert CANDIDATE_CAN_RAISE_AUTONOMY_CEILING is False
    assert CANDIDATE_CAN_ADMIT_SCRIPT_PAYLOAD is False
    assert tuple(item.value for item in INITIAL_OPERATORS) == (
        "EXTRACT_MODULE",
        "EXTRACT_INTERFACE",
        "EXTRACT_PURE_FUNCTION",
        "MOVE_STATE_TO_OWNER",
        "INTRODUCE_DEPENDENCY_INVERSION",
        "REPLACE_DIRECT_CALL_WITH_TYPED_SERVICE",
        "GENERATE_ADAPTER",
        "GENERATE_COMPATIBILITY_SHIM",
        "QUARANTINE_LEGACY_PATH",
        "QUARANTINE_SIMULATION_PATH",
        "REPLACE_BOOLEAN_WITH_CLOSED_OUTCOME",
        "REPLACE_DYNAMIC_REGISTRY_WITH_TYPED_CATALOG",
        "REPLACE_EAGER_IMPORT_WITH_LAZY_CAPABILITY",
        "CONSOLIDATE_ERROR_VOCABULARY",
        "CONSOLIDATE_RECEIPT_PRODUCER",
        "CONSOLIDATE_CAPABILITY_AUTHORITY",
        "REMOVE_CONFIRMED_DEAD_CODE",
        "SPLIT_MONOLITH_BY_AUTHORITY",
        "MOVE_GENERATED_PROJECTION_OUT_OF_SOURCE_AUTHORITY",
        "DEPRECATE_PUBLIC_SYMBOL",
        "REMOVE_DEPRECATED_SYMBOL_AFTER_GATE",
    )
    assert REQUIRED_OPERATORS == INITIAL_OPERATORS
    assert CLOSED_OPERATORS == {item.value for item in OperatorKind}
    assert CLOSED_RISK_CLASSES == {item.value for item in OperatorRiskClass}
    assert CLOSED_AUTONOMY_CEILINGS == {item.value for item in AutonomyCeiling}
    assert CLOSED_AUTHORITY_IMPACTS == {item.value for item in AuthorityImpact}
    assert CLOSED_API_IMPACTS == {item.value for item in ApiImpact}
    assert CLOSED_STATE_IMPACTS == {item.value for item in StateImpact}
    assert CLOSED_EXPECTED_EFFECTS == {item.value for item in ExpectedEffect}
    assert CLOSED_PROHIBITED_EFFECTS == {item.value for item in ProhibitedEffect}
    assert CLOSED_PRECONDITIONS == {item.value for item in OperatorPrecondition}
    assert CLOSED_PROOF_OBLIGATIONS == {item.value for item in ProofObligation}
    assert CLOSED_VALIDATION_OBLIGATIONS == {item.value for item in ValidationObligation}
    assert len(INITIAL_OPERATORS) == 21
    assert OPERATOR_CATALOG.covers_initial_operators is True
    assert tuple(item.kind for item in OPERATOR_CATALOG.operators) == INITIAL_OPERATORS
    with pytest.raises(RefactorOperatorError, match="unsupported refactor operator"):
        get_operator("RUN_ARBITRARY_SCRIPT")
    with pytest.raises(ValueError):
        OperatorKind("RUN_ARBITRARY_SCRIPT")
    with pytest.raises(ValueError):
        OperatorRiskClass("unbounded_rewrite")
    with pytest.raises(ValueError):
        AutonomyCeiling("self_approved")
    with pytest.raises(ValueError):
        AuthorityImpact("transfer")
    with pytest.raises(ValueError):
        ExpectedEffect("shell_out")
    with pytest.raises(ValueError):
        ValidationObligation("skip_if_inconvenient")


def test_every_initial_operator_has_a_complete_declaration() -> None:
    required = set(REQUIRED_OPERATOR_DECLARATION_FIELDS)
    for operator in OPERATOR_CATALOG.operators:
        payload = operator.to_dict()
        assert required <= set(payload)
        assert operator.schema == REFACTOR_OPERATOR_SCHEMA
        assert operator.preconditions
        assert set(REQUIRED_PRECONDITIONS) <= set(operator.preconditions)
        assert operator.target_kinds
        assert all(isinstance(item, NodeKind) for item in operator.target_kinds)
        assert operator.expected_effects
        assert frozenset(operator.prohibited_effects) == frozenset(REQUIRED_PROHIBITED_EFFECTS)
        assert operator.authority_impact in AuthorityImpact
        assert operator.api_impact in ApiImpact
        assert operator.state_impact in StateImpact
        assert operator.migration.phases
        assert operator.migration.transfers_authority is False
        assert operator.migration.indefinite_dual_authority is False
        assert operator.rollback.required is True
        assert operator.rollback.restores_tree is True
        assert operator.rollback.action is RollbackAction.RESTORE_ISOLATED_WORKTREE
        assert operator.validation.required is True
        assert operator.validation.gates_reducible is False
        assert set(REQUIRED_VALIDATION_OBLIGATIONS) <= set(operator.validation.obligations)
        assert set(REQUIRED_PROOF_OBLIGATIONS) <= set(operator.proofs)
        assert operator.maximum_scope.allows_scope_expansion is False
        assert operator.maximum_scope.allows_script_payload is False
        assert operator.maximum_scope.allows_sibling_write is False
        assert operator.maximum_scope.allows_network is False
        assert operator.maximum_scope.allows_protected_path is False
        assert operator.maximum_scope.repository_relative_only is True
        assert operator.autonomy_ceiling is autonomy_ceiling_for_risk(operator.risk_class)
        assert operator.can_authorize_execution is False
        assert operator.can_self_promote is False
        round_trip = RefactorOperator.from_mapping(payload)
        assert round_trip.content_identity == operator.content_identity
        assert json.loads(operator.to_json())["content_identity"] == operator.content_identity
        validate_cid(operator.content_identity, codecs=("dag-json",))
        assert cid_for_dag_json(operator._identity_payload()) == operator.content_identity


def test_autonomy_classification_map_covers_every_operator() -> None:
    mapping = autonomy_classification_map()
    risks = risk_classification_map()
    assert set(mapping) == CLOSED_OPERATORS
    assert set(risks) == CLOSED_OPERATORS
    automatic = {
        OperatorRiskClass.PURE_MODULE_EXTRACTION,
        OperatorRiskClass.GENERATED_PROJECTION_REGENERATION,
        OperatorRiskClass.LAZY_IMPORT_CONVERSION,
        OperatorRiskClass.INTERNAL_ADAPTER_GENERATION,
        OperatorRiskClass.CLOSED_RESULT_TYPE_MIGRATION,
        OperatorRiskClass.CONFIRMED_DEAD_INTERNAL_CODE_REMOVAL,
        OperatorRiskClass.TEST_FIXTURE_RELOCATION,
        OperatorRiskClass.SIMULATION_NAMESPACE_RELOCATION,
    }
    assert AUTOMATIC_RISK_CLASSES == automatic
    assert get_operator(OperatorKind.EXTRACT_MODULE).autonomy_ceiling is AutonomyCeiling.AUTOMATIC
    assert get_operator(OperatorKind.MOVE_STATE_TO_OWNER).autonomy_ceiling is (
        AutonomyCeiling.PROPOSAL_ONLY
    )
    assert get_operator(OperatorKind.DEPRECATE_PUBLIC_SYMBOL).autonomy_ceiling is (
        AutonomyCeiling.PROPOSAL_ONLY
    )
    assert get_operator(OperatorKind.QUARANTINE_SIMULATION_PATH).risk_class in AUTOMATIC_RISK_CLASSES
    assert get_operator(OperatorKind.CONSOLIDATE_RECEIPT_PRODUCER).risk_class in (
        PROPOSAL_ONLY_RISK_CLASSES
    )
    assert OperatorRiskClass.AUTHORIZATION_POLICY_SECURITY in HUMAN_APPROVAL_RISK_CLASSES
    assert get_operator(OperatorKind.MOVE_STATE_TO_OWNER).migration.phases == STATE_MIGRATION_PHASES
    assert get_operator(OperatorKind.MOVE_STATE_TO_OWNER).migration.mutates_state is True
    assert get_operator(OperatorKind.MOVE_STATE_TO_OWNER).state_impact is (
        StateImpact.MIGRATE_TO_OWNER
    )
    for kind, ceiling in mapping.items():
        operator = get_operator(kind)
        assert ceiling == operator.autonomy_ceiling.value
        assert risks[kind] == operator.risk_class.value


def test_complete_declaration_rejects_missing_preconditions_rollback_and_validation() -> None:
    operator = get_operator(OperatorKind.EXTRACT_PURE_FUNCTION)
    with pytest.raises(RefactorOperatorError, match="incomplete"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=(),
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            api_impact=operator.api_impact,
            state_impact=operator.state_impact,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
            risk_class=operator.risk_class,
        )
    with pytest.raises(RefactorOperatorError, match="incomplete"):
        OperatorRollback(required=False)
    with pytest.raises(RefactorOperatorError, match="restore"):
        OperatorRollback(restores_tree=False)
    with pytest.raises(RefactorOperatorError, match="incomplete"):
        OperatorValidation(obligations=REQUIRED_VALIDATION_OBLIGATIONS, required=False)
    with pytest.raises(RefactorOperatorError, match="incomplete"):
        OperatorValidation(obligations=(ValidationObligation.STATIC_TYPE_CHECK,))
    with pytest.raises(RefactorOperatorError, match="incomplete"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=(),
            authority_impact=operator.authority_impact,
            api_impact=operator.api_impact,
            state_impact=operator.state_impact,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
            risk_class=operator.risk_class,
        )
    with pytest.raises(RefactorOperatorError, match="incomplete"):
        OperatorMigration(phases=())


def test_unknown_fields_and_unknown_operators_fail_closed() -> None:
    operator = get_operator(OperatorKind.GENERATE_ADAPTER)
    payload = operator.to_dict()
    payload["script"] = "rm -rf /"
    with pytest.raises(RefactorOperatorError, match="unknown"):
        RefactorOperator.from_mapping(payload)
    payload = operator.to_dict()
    del payload["rollback"]
    with pytest.raises(RefactorOperatorError, match="missing"):
        RefactorOperator.from_mapping(payload)
    catalog_payload = OPERATOR_CATALOG.to_dict()
    catalog_payload["shell"] = "bash -c true"
    with pytest.raises(RefactorOperatorError, match="unknown"):
        type(OPERATOR_CATALOG).from_mapping(catalog_payload)
    candidate = _candidate()
    candidate_payload = candidate.to_dict()
    candidate_payload["payload"] = "os.system('true')"
    with pytest.raises(RefactorCandidateError, match="unknown"):
        RefactorCandidate.from_mapping(candidate_payload)
    with pytest.raises(RefactorOperatorError, match="unsupported"):
        declare_refactor_candidate(
            operator_kind="EXEC_SHELL",
            repository_tree=_TREE,
            contract_identity=_CONTRACT,
            targets=("n-module",),
            paths=(_PATH,),
            node_ids=("n-module",),
            symbol_count=1,
            package_count=1,
            target_kinds=(NodeKind.MODULE,),
        )


def test_maximum_scope_rejects_expansion_siblings_and_protected_paths() -> None:
    operator = get_operator(OperatorKind.EXTRACT_PURE_FUNCTION)
    too_many = tuple(
        f"ipfs_accelerate_py/agent_supervisor/architecture_refactorer/mod_{index}.py"
        for index in range(operator.maximum_scope.max_files + 1)
    )
    with pytest.raises(RefactorOperatorAuthorityError, match="maximum scope"):
        _candidate(OperatorKind.EXTRACT_PURE_FUNCTION, paths=too_many)
    with pytest.raises(RefactorOperatorError, match="sibling"):
        _candidate(
            paths=("ipfs_datasets_py/ipfs_datasets_py/__init__.py",),
        )
    with pytest.raises(RefactorOperatorError, match="protected"):
        _candidate(paths=(next(iter(PROTECTED_PATHS)),))
    with pytest.raises(RefactorOperatorError, match="repository-relative"):
        _candidate(paths=("../escape.py",))
    with pytest.raises(RefactorOperatorError, match="repository-relative"):
        _candidate(paths=("/tmp/outside.py",))
    with pytest.raises(RefactorOperatorAuthorityError, match="script"):
        MaximumScope(
            max_paths=1,
            max_files=1,
            max_symbols=1,
            max_packages=1,
            allows_script_payload=True,
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="maximum scope"):
        MaximumScope(
            max_paths=1,
            max_files=1,
            max_symbols=1,
            max_packages=1,
            allows_scope_expansion=True,
        )
    with pytest.raises(RefactorOperatorError, match="sibling"):
        MaximumScope(
            max_paths=1,
            max_files=1,
            max_symbols=1,
            max_packages=1,
            allows_sibling_write=True,
        )
    with pytest.raises(RefactorCandidateError, match="target kinds exceed"):
        _candidate(target_kinds=(NodeKind.PROVIDER,))
    with pytest.raises(RefactorOperatorAuthorityError, match="maximum scope"):
        refuse_scope_expansion("expand")


def test_self_authorization_and_self_promotion_are_rejected() -> None:
    operator = get_operator(OperatorKind.EXTRACT_MODULE)
    candidate = _candidate()
    with pytest.raises(RefactorOperatorAuthorityError, match="authorize"):
        operator.authorize()
    with pytest.raises(RefactorOperatorAuthorityError, match="apply"):
        operator.apply()
    with pytest.raises(RefactorOperatorAuthorityError, match="promote"):
        operator.promote()
    with pytest.raises(RefactorOperatorAuthorityError, match="raise"):
        operator.raise_autonomy_ceiling()
    with pytest.raises(RefactorOperatorAuthorityError, match="reduce"):
        operator.reduce_gates()
    with pytest.raises(RefactorOperatorAuthorityError, match="script"):
        operator.admit_script()
    with pytest.raises(RefactorOperatorAuthorityError, match="maximum scope"):
        operator.expand_scope()
    with pytest.raises(RefactorOperatorAuthorityError, match="authorize"):
        candidate.authorize()
    with pytest.raises(RefactorOperatorAuthorityError, match="apply"):
        candidate.apply()
    with pytest.raises(RefactorOperatorAuthorityError, match="promote"):
        candidate.promote()
    with pytest.raises(RefactorOperatorAuthorityError, match="raise"):
        candidate.raise_autonomy_ceiling()
    with pytest.raises(RefactorOperatorAuthorityError, match="script"):
        candidate.admit_script()
    with pytest.raises(RefactorOperatorAuthorityError, match="authorize"):
        OPERATOR_CATALOG.authorize()
    with pytest.raises(RefactorOperatorAuthorityError, match="promote"):
        OPERATOR_CATALOG.promote()
    with pytest.raises(RefactorOperatorAuthorityError, match="authorize"):
        refuse_self_authorization("authorize")
    with pytest.raises(RefactorOperatorAuthorityError, match="promote"):
        refuse_self_promotion("promote")
    with pytest.raises(RefactorOperatorAuthorityError, match="reduce"):
        refuse_gate_reduction("reduce")
    with pytest.raises(RefactorOperatorAuthorityError, match="raise"):
        refuse_autonomy_ceiling_raise("raise")
    with pytest.raises(RefactorOperatorAuthorityError, match="script"):
        refuse_script_payload("admit")
    with pytest.raises(RefactorOperatorAuthorityError, match="authorize"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            api_impact=operator.api_impact,
            state_impact=operator.state_impact,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
            risk_class=operator.risk_class,
            can_authorize_execution=True,
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="promote"):
        RefactorOperator(
            kind=operator.kind,
            preconditions=operator.preconditions,
            target_kinds=operator.target_kinds,
            expected_effects=operator.expected_effects,
            authority_impact=operator.authority_impact,
            api_impact=operator.api_impact,
            state_impact=operator.state_impact,
            migration=operator.migration,
            rollback=operator.rollback,
            validation=operator.validation,
            proofs=operator.proofs,
            maximum_scope=operator.maximum_scope,
            risk_class=operator.risk_class,
            can_self_promote=True,
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="reduce"):
        OperatorValidation(
            obligations=REQUIRED_VALIDATION_OBLIGATIONS,
            gates_reducible=True,
        )
    with pytest.raises(RefactorOperatorAuthorityError, match="raise"):
        RefactorOperator(
            kind=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).kind,
            preconditions=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).preconditions,
            target_kinds=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).target_kinds,
            expected_effects=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).expected_effects,
            authority_impact=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).authority_impact,
            api_impact=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).api_impact,
            state_impact=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).state_impact,
            migration=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).migration,
            rollback=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).rollback,
            validation=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).validation,
            proofs=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).proofs,
            maximum_scope=get_operator(OperatorKind.MOVE_STATE_TO_OWNER).maximum_scope,
            risk_class=OperatorRiskClass.STATE_MIGRATION,
            autonomy_ceiling=AutonomyCeiling.AUTOMATIC,
        )


def test_canonical_candidate_identity_binds_tree_contract_and_effects() -> None:
    candidate = _candidate()
    assert candidate.repository_tree == _TREE
    assert candidate.contract_identity == _CONTRACT
    assert candidate.effects_identity == effects_identity(candidate.expected_effects)
    assert candidate.operator_identity == get_operator(OperatorKind.EXTRACT_MODULE).content_identity
    validate_cid(candidate.content_identity, codecs=("dag-json",))
    assert canonical_candidate_identity(candidate) == candidate.content_identity
    assert cid_for_dag_json(candidate._identity_payload()) == candidate.content_identity
    round_trip = RefactorCandidate.from_mapping(candidate.to_dict())
    assert round_trip == candidate
    assert json.loads(candidate.to_json())["content_identity"] == candidate.content_identity
    other_tree = _candidate(repository_tree="b" * 40)
    other_contract = _candidate(
        contract_identity=cid_for_dag_json({"schema": "other-contract", "version": 1})
    )
    other_effects = _candidate(OperatorKind.EXTRACT_PURE_FUNCTION)
    assert other_tree.content_identity != candidate.content_identity
    assert other_contract.content_identity != candidate.content_identity
    assert other_effects.content_identity != candidate.content_identity
    assert other_effects.effects_identity != candidate.effects_identity
    again = _candidate()
    assert again.content_identity == candidate.content_identity
    payload = candidate.to_dict()
    payload["content_identity"] = cid_for_dag_json({"tampered": True})
    with pytest.raises(RefactorCandidateError, match="content identity"):
        RefactorCandidate.from_mapping(payload)
    with pytest.raises(RefactorCandidateError, match="effects must match"):
        RefactorCandidate(
            operator_kind=candidate.operator_kind,
            operator_identity=candidate.operator_identity,
            repository_tree=candidate.repository_tree,
            contract_identity=candidate.contract_identity,
            expected_effects=(ExpectedEffect.INTRODUCE_ADAPTER,),
            targets=candidate.targets,
            scope=candidate.scope,
            preconditions=candidate.preconditions,
            authority_impact=candidate.authority_impact,
            api_impact=candidate.api_impact,
            state_impact=candidate.state_impact,
            migration=candidate.migration,
            rollback=candidate.rollback,
            validation=candidate.validation,
            proofs=candidate.proofs,
            risk_class=candidate.risk_class,
            autonomy_ceiling=candidate.autonomy_ceiling,
        )


def test_candidate_rejects_script_payloads_and_authority_transfer() -> None:
    with pytest.raises(RefactorOperatorAuthorityError, match="script"):
        OperatorRollback(message="restore via /bin/sh -c reset")
    with pytest.raises(RefactorOperatorAuthorityError, match="transfer"):
        OperatorMigration(
            phases=STATE_MIGRATION_PHASES,
            transfers_authority=True,
            mutates_state=True,
        )
    with pytest.raises(RefactorOperatorError, match="dual authority"):
        OperatorMigration(
            phases=STATE_MIGRATION_PHASES,
            indefinite_dual_authority=True,
            mutates_state=True,
        )
    with pytest.raises(RefactorOperatorError, match="adapter"):
        OperatorMigration(phases=STATE_MIGRATION_PHASES, adapters=("bash -c true",))
    with pytest.raises(RefactorOperatorAuthorityError, match="script"):
        CandidateScope(
            paths=("ipfs_accelerate_py/agent_supervisor/os.system.py",),
            node_ids=("n-module",),
            file_count=1,
            symbol_count=1,
            package_count=1,
            target_kinds=(NodeKind.MODULE,),
        )
    assert CANDIDATE_SCOPE_SCHEMA.endswith("candidate-scope@1")
    assert DEFAULT_FRESHNESS == "pcar-012-refactor-operator"


def test_catalog_round_trip_is_deterministic() -> None:
    payload = OPERATOR_CATALOG.to_dict()
    restored = OperatorCatalog.from_mapping(payload)
    assert restored.content_identity == OPERATOR_CATALOG.content_identity
    assert restored.autonomy_classification_map() == autonomy_classification_map()
    assert (
        OperatorCatalog.from_json(OPERATOR_CATALOG.to_json()).content_identity
        == OPERATOR_CATALOG.content_identity
    )
    again = json.loads(OPERATOR_CATALOG.to_json())
    assert again["content_identity"] == OPERATOR_CATALOG.content_identity
    validate_cid(OPERATOR_CATALOG.content_identity, codecs=("dag-json",))
    missing = payload["operators"][:-1]
    broken = dict(payload)
    broken["operators"] = missing
    with pytest.raises(RefactorOperatorError, match="incomplete"):
        OperatorCatalog.from_mapping(broken)
