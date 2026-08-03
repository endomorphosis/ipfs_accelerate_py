"""Tests for the deterministic analysis and formal-method strategy registry."""

from __future__ import annotations

import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_operation_registry import (
    AnalysisOperation,
    attach_strategy_routing,
    create_default_analysis_operation_registry,
    operation_property_classes,
    route_operation_strategies,
)
from ipfs_accelerate_py.agent_supervisor.analysis.analysis_strategy_registry import (
    ANALYSIS_CAPABILITY_RECEIPT_SCHEMA,
    ANALYSIS_STRATEGY_REGISTRY_INTERFACE,
    ANALYSIS_STRATEGY_REGISTRY_VERSION,
    AnalysisCapabilityReceipt,
    AnalysisStrategyRegistry,
    AnalysisStrategyRegistryError,
    AnalysisStrategySpec,
    AuthorityUse,
    CapabilityAdmission,
    FallbackBehavior,
    LazyProviderAdapter,
    MethodRole,
    PropertyQuestionClass,
    SelectionOutcome,
    StrategyAssurance,
    StrategyBudget,
    StrategyCacheRules,
    StrategyMethod,
    StrategyMethodBinding,
    StrategySelection,
    create_default_analysis_strategy_registry,
    default_lazy_provider_adapters,
    default_strategy_specs,
    normalize_property_class,
    normalize_strategy_method,
    property_class_for_operation,
)
from ipfs_accelerate_py.agent_supervisor.analysis.analysis_transport import (
    AnalysisProviderHealth,
)


EXPECTED_PROPERTY_CLASSES = {
    "syntax_structure",
    "control_data_flow",
    "aliasing_state",
    "values_security_flow",
    "contracts",
    "heap_native_safety",
    "recursive_invariants",
    "constraint_solving",
    "state_concurrency",
    "protocol_security",
    "behavioral_tests",
    "runtime_contracts",
    "rewrite_synthesis",
    "supply_chain",
    "retrieval",
    "formal_kernels",
    "cryptographic_lineage",
}

# Evidence subset called out by PDR-012.
EXPECTED_METHOD_FAMILIES = {
    StrategyMethod.PYTHON_AST,
    StrategyMethod.TREE_SITTER,
    StrategyMethod.CFG,
    StrategyMethod.SSA,
    StrategyMethod.PDG,
    StrategyMethod.CALL_GRAPH,
    StrategyMethod.POINTS_TO,
    StrategyMethod.TAINT,
    StrategyMethod.ABSTRACT_INTERPRETATION,
    StrategyMethod.TYPESTATE,
    StrategyMethod.WEAKEST_PRECONDITION,
    StrategyMethod.SEPARATION_LOGIC,
    StrategyMethod.SAT,
    StrategyMethod.SMT,
    StrategyMethod.CHC,
    StrategyMethod.DATALOG,
    StrategyMethod.CEGAR,
    StrategyMethod.CEGIS,
    StrategyMethod.TEMPORAL_LOGIC,
    StrategyMethod.HYPERPROPERTY,
    StrategyMethod.FUZZ,
    StrategyMethod.CONCOLIC,
    StrategyMethod.MUTATION_TEST,
    StrategyMethod.DIFFERENTIAL_TEST,
    StrategyMethod.METAMORPHIC_TEST,
    StrategyMethod.TEMPORAL_MONITOR,
    StrategyMethod.SBOM,
    StrategyMethod.LEAN_KERNEL,
    StrategyMethod.BM25,
    StrategyMethod.GRAPH_RAG,
    StrategyMethod.LEARNED_RANKING,
}


def test_default_portfolio_covers_closed_property_classes() -> None:
    specs = default_strategy_specs()
    classes = {item.property_class.value for item in specs}
    assert classes == EXPECTED_PROPERTY_CLASSES
    methods = {binding.method for spec in specs for binding in spec.methods}
    assert EXPECTED_METHOD_FAMILIES.issubset(methods)

    for spec in specs:
        record = spec.to_dict()
        assert record["schema"].endswith("analysis-strategy@1")
        assert record["required_assurance"]
        assert record["input_schema"]
        assert record["output_schema"]
        assert record["cache"]["content_addressed"] is True
        assert record["cache"]["allow_stale"] is False
        assert record["cache"]["rederive_assurance_on_hit"] is True
        assert record["budget"]["timeout_ms"] > 0
        assert record["fallback"]
        assert record["authority"]["completion_authority"] is False
        assert record["authority"]["proof_authority"] is False
        assert AnalysisStrategySpec.from_dict(record) == spec
        # Methods are ordered by cost rank.
        ranks = [item.cost_rank for item in spec.methods]
        assert ranks == sorted(ranks)


def test_retrieval_and_learned_ranking_remain_nomination_only() -> None:
    registry = create_default_analysis_strategy_registry()
    retrieval = registry.strategy(PropertyQuestionClass.RETRIEVAL)

    assert all(item.nomination_only for item in retrieval.methods)
    assert all(
        item.authority_use is AuthorityUse.NOMINATION_ONLY
        for item in retrieval.methods
    )
    learned = next(
        item
        for item in retrieval.methods
        if item.method is StrategyMethod.LEARNED_RANKING
    )
    assert learned.learned_ranking is True
    assert learned.max_assurance.rank <= StrategyAssurance.CANDIDATE.rank

    # Cannot satisfy kernel assurance via retrieval.
    selection = registry.select(
        "retrieval",
        required_assurance=StrategyAssurance.KERNEL_VERIFIED,
    )
    assert selection.outcome is SelectionOutcome.ABSTAIN
    assert selection.nomination_only is True
    assert "nomination-only" in selection.abstention_reason

    # Usable retrieval routes stay nomination-only when selected.
    cold = registry.select("retrieval")
    assert cold.outcome in {
        SelectionOutcome.SELECTED,
        SelectionOutcome.PARTIAL,
        SelectionOutcome.DEBT_ONLY,
    }
    assert cold.nomination_only is True
    assert cold.achieved_assurance.rank <= StrategyAssurance.CANDIDATE.rank


def test_required_unavailable_methods_abstain() -> None:
    registry = create_default_analysis_strategy_registry()
    # Explicit map: deny SMT capabilities required by constraint_solving.
    selection = registry.select(
        PropertyQuestionClass.CONSTRAINT_SOLVING,
        available_capabilities={
            "smt": False,
            "logic_family_routing": False,
            "z3": False,
            "cvc5": False,
            "sat": False,
            "maxsat": False,
            "tactician_hammer": False,
            "cegis": False,
        },
    )
    assert selection.outcome is SelectionOutcome.ABSTAIN
    assert "required method" in selection.abstention_reason
    assert selection.selected_methods == ()

    # Protocol security requires security_ir.
    protocol = registry.select(
        "protocol_security",
        available_capabilities={"security_ir": False},
    )
    assert protocol.outcome is SelectionOutcome.ABSTAIN
    assert "security_ir" in protocol.abstention_reason or "required" in (
        protocol.abstention_reason
    )


def test_optional_unavailable_methods_add_debt() -> None:
    registry = create_default_analysis_strategy_registry()
    # Syntax: keep required AST/symbol, deny optional tree-sitter/call-graph.
    selection = registry.select(
        PropertyQuestionClass.SYNTAX_STRUCTURE,
        available_capabilities={
            "ast_index_read": True,
            "python_ast": True,
            "symbol_impact": True,
            "tree_sitter": False,
            "polyglot_ast": False,
            "call_graph": False,
        },
    )
    assert selection.outcome in {
        SelectionOutcome.SELECTED,
        SelectionOutcome.PARTIAL,
    }
    assert selection.has_debt
    debt_methods = {item.method for item in selection.debt if item.method}
    assert StrategyMethod.TREE_SITTER in debt_methods or (
        StrategyMethod.CALL_GRAPH in debt_methods
    )
    assert any(
        item.method
        in {StrategyMethod.PYTHON_AST, StrategyMethod.SYMBOL_INDEX}
        for item in selection.selected_methods
    )


def test_least_cost_sufficient_prefers_cheaper_methods() -> None:
    registry = create_default_analysis_strategy_registry()
    selection = registry.select(
        PropertyQuestionClass.SYNTAX_STRUCTURE,
        required_assurance=StrategyAssurance.OBSERVED,
        available_capabilities={
            "ast_index_read": True,
            "python_ast": True,
            "symbol_impact": True,
            "tree_sitter": True,
            "polyglot_ast": True,
            "call_graph": True,
        },
    )
    assert selection.outcome in {
        SelectionOutcome.SELECTED,
        SelectionOutcome.PARTIAL,
    }
    costs = [item.cost_rank for item in selection.selected_methods]
    assert costs == sorted(costs)
    # Cheapest sufficient primary should be among the lowest cost ranks.
    assert selection.selected_methods[0].cost_rank == min(
        item.cost_rank
        for item in registry.strategy("syntax_structure").methods
        if item.max_assurance.satisfies(StrategyAssurance.OBSERVED)
        and item.role is MethodRole.REQUIRED
        or item.method is selection.selected_methods[0].method
    )


def test_capability_receipts_bind_health_version_config() -> None:
    registry = create_default_analysis_strategy_registry()
    receipt = registry.bind_capability_receipt(
        method=StrategyMethod.SMT,
        property_class=PropertyQuestionClass.CONSTRAINT_SOLVING,
        capability_id="smt",
        provider_id="ipfs-datasets-analysis",
        admission=CapabilityAdmission.AVAILABLE,
        health=AnalysisProviderHealth.HEALTHY,
        provider_version="1.2.3",
        config_digest="config:smt@pinned",
        capability_revision="cap@smt@1",
        reason_code="probe_ok",
    )
    assert receipt.receipt_id.startswith("analysis-capability-receipt:sha256:")
    record = receipt.to_dict()
    assert record["schema"] == ANALYSIS_CAPABILITY_RECEIPT_SCHEMA
    assert record["interface"] == "AnalysisCapabilityReceipt@1"
    assert record["provider_version"] == "1.2.3"
    assert record["config_digest"] == "config:smt@pinned"
    assert record["health"] == "healthy"
    assert record["authority"]["completion_authority"] is False
    assert AnalysisCapabilityReceipt.from_dict(record) == receipt

    with pytest.raises(AnalysisStrategyRegistryError):
        AnalysisCapabilityReceipt(
            capability_id="smt",
            provider_id="x",
            method=StrategyMethod.SMT,
            property_class=PropertyQuestionClass.CONSTRAINT_SOLVING,
            admission=CapabilityAdmission.AVAILABLE,
            health=AnalysisProviderHealth.UNAVAILABLE,
        )


def test_discovery_is_cold_lazy_and_does_not_infer_imports() -> None:
    # Importing the strategy module must not pull optional datasets providers.
    polluted = [
        name
        for name in sys.modules
        if name.startswith("ipfs_datasets_py")
        and "program_analysis" in name
    ]
    # The presence of unrelated previously-imported modules is allowed; the
    # factory itself must not import them.
    before = set(sys.modules)
    registry = create_default_analysis_strategy_registry()
    after = set(sys.modules)
    newly = after - before
    assert not any("ipfs_datasets_py" in name for name in newly)

    declarations = registry.discover_provider_declarations()
    assert declarations
    assert all(item["probed_on_import"] is False for item in declarations)
    assert all(item["health"] == "lazy" for item in declarations)
    # Support is not inferred from importability of optional packages.
    assert any(
        item["provider_id"] == "ipfs-datasets-analysis" for item in declarations
    )
    assert all(
        item.to_dict()["has_factory"] is False
        for item in default_lazy_provider_adapters()
    )

    # Cold selection uses LAZY declarations without probing.
    selection = registry.select(PropertyQuestionClass.SYNTAX_STRUCTURE)
    assert selection.outcome in {
        SelectionOutcome.SELECTED,
        SelectionOutcome.PARTIAL,
    }
    assert any(
        receipt.admission is CapabilityAdmission.LAZY
        for receipt in selection.receipts
    )
    # Avoid unused variable lint if polluted empty.
    assert isinstance(polluted, list)


def test_capability_declarations_do_not_inherit_method_max_assurance() -> None:
    registry = create_default_analysis_strategy_registry()

    # Cold LAZY declarations select a possible route, not a kernel replay.
    cold = registry.select(PropertyQuestionClass.FORMAL_KERNELS)
    assert cold.outcome in {
        SelectionOutcome.PARTIAL,
        SelectionOutcome.DEBT_ONLY,
    }
    assert cold.achieved_assurance is StrategyAssurance.UNVERIFIED
    assert not cold.achieved_assurance.satisfies(
        StrategyAssurance.KERNEL_VERIFIED
    )
    assert any(
        item.reason_code == "execution_receipt_required" for item in cold.debt
    )

    # Boolean availability placeholders likewise cannot manufacture solver
    # evidence from the method declaration's theoretical upper bound.
    boolean = registry.select(
        PropertyQuestionClass.CONSTRAINT_SOLVING,
        available_capabilities={
            "smt": True,
            "logic_family_routing": True,
            "sat": False,
            "z3": False,
            "cvc5": False,
            "maxsat": False,
            "tactician_hammer": False,
            "cegis": False,
        },
    )
    assert boolean.outcome is SelectionOutcome.PARTIAL
    assert boolean.achieved_assurance is StrategyAssurance.UNVERIFIED
    assert any(
        receipt.details.get("explicit_bool") is True
        for receipt in boolean.receipts
    )
    assert any(
        item.reason_code == "execution_receipt_required"
        for item in boolean.debt
    )


def test_selection_rejects_unmet_assurance_claim() -> None:
    with pytest.raises(
        AnalysisStrategyRegistryError,
        match="unmet assurance obligation",
    ):
        StrategySelection(
            property_class=PropertyQuestionClass.CONSTRAINT_SOLVING,
            strategy_id="strategy:adversarial",
            outcome=SelectionOutcome.SELECTED,
            selected_methods=(
                StrategyMethodBinding(
                    method=StrategyMethod.SMT,
                    role=MethodRole.REQUIRED,
                    cost_rank=1,
                    max_assurance=StrategyAssurance.SOLVER_CHECKED,
                ),
            ),
            required_assurance=StrategyAssurance.SOLVER_CHECKED,
            achieved_assurance=StrategyAssurance.UNVERIFIED,
        )


def test_formal_kernels_and_crypto_lineage_assurance_contracts() -> None:
    registry = create_default_analysis_strategy_registry()
    kernels = registry.strategy(PropertyQuestionClass.FORMAL_KERNELS)
    assert kernels.required_assurance is StrategyAssurance.KERNEL_VERIFIED
    assert all(
        item.authority_use is AuthorityUse.KERNEL_ASSURANCE
        for item in kernels.methods
    )

    # Explicit unavailable kernels abstain when no method meets assurance.
    denied = registry.select(
        "formal_kernels",
        available_capabilities={
            "lean": False,
            "rocq": False,
            "isabelle": False,
            "kernel_replay": False,
        },
    )
    assert denied.outcome in {
        SelectionOutcome.ABSTAIN,
        SelectionOutcome.DEBT_ONLY,
    }
    assert denied.achieved_assurance is StrategyAssurance.UNVERIFIED

    lineage = registry.strategy("cryptographic_lineage")
    assert lineage.required_assurance is StrategyAssurance.ATTESTED
    assert any(
        item.role is MethodRole.REQUIRED and item.method is StrategyMethod.CID_MERKLE
        for item in lineage.methods
    )


def test_cache_rules_and_budgets_are_fail_closed() -> None:
    with pytest.raises(AnalysisStrategyRegistryError):
        StrategyCacheRules(allow_stale=True)
    with pytest.raises(AnalysisStrategyRegistryError):
        StrategyCacheRules(rederive_assurance_on_hit=False)
    with pytest.raises(AnalysisStrategyRegistryError):
        StrategyBudget(timeout_ms=0)

    budget = StrategyBudget(timeout_ms=5_000, max_solver_fuel=100)
    assert budget.to_dict()["max_solver_fuel"] == 100


def test_aliases_and_unknowns() -> None:
    assert normalize_property_class("syntax") is PropertyQuestionClass.SYNTAX_STRUCTURE
    assert normalize_property_class("smt") is PropertyQuestionClass.CONSTRAINT_SOLVING
    assert normalize_strategy_method("ast") is StrategyMethod.PYTHON_AST
    assert normalize_strategy_method("graphrag") is StrategyMethod.GRAPH_RAG
    assert normalize_strategy_method("lean") is StrategyMethod.LEAN_KERNEL
    with pytest.raises(AnalysisStrategyRegistryError):
        normalize_property_class("not_a_property")
    with pytest.raises(AnalysisStrategyRegistryError):
        normalize_strategy_method("telepathy")


def test_nomination_only_methods_cannot_claim_kernel_assurance() -> None:
    with pytest.raises(AnalysisStrategyRegistryError):
        StrategyMethodBinding(
            method=StrategyMethod.BM25,
            nomination_only=True,
            max_assurance=StrategyAssurance.KERNEL_VERIFIED,
            authority_use=AuthorityUse.NOMINATION_ONLY,
            cost_rank=1,
        )
    with pytest.raises(AnalysisStrategyRegistryError):
        AnalysisStrategySpec(
            property_class=PropertyQuestionClass.RETRIEVAL,
            required_assurance=StrategyAssurance.KERNEL_VERIFIED,
            methods=(
                StrategyMethodBinding(
                    method=StrategyMethod.BM25,
                    nomination_only=True,
                    max_assurance=StrategyAssurance.CANDIDATE,
                    cost_rank=1,
                ),
            ),
        )


def test_registry_round_trip_and_authority_surface() -> None:
    registry = create_default_analysis_strategy_registry()
    record = registry.to_dict()
    assert record["interface"] == ANALYSIS_STRATEGY_REGISTRY_INTERFACE
    assert record["registry_version"] == ANALYSIS_STRATEGY_REGISTRY_VERSION
    assert record["authority"]["retrieval_is_nomination_only"] is True
    assert record["authority"]["import_does_not_imply_support"] is True
    assert record["authority"]["completion_authority"] is False
    assert {item["property_class"] for item in record["strategies"]} == (
        EXPECTED_PROPERTY_CLASSES
    )
    assert registry.registry_id.startswith("analysis-strategy-registry:sha256:")

    # Duplicate registration fails closed.
    with pytest.raises(AnalysisStrategyRegistryError):
        registry.register_strategy(default_strategy_specs()[0])


def test_operation_registry_strategy_bridge() -> None:
    classes = operation_property_classes(AnalysisOperation.SYMBOL_IMPACT)
    assert "syntax_structure" in classes
    assert property_class_for_operation("graphrag_retrieval") == (
        PropertyQuestionClass.RETRIEVAL,
    )

    selections = route_operation_strategies(
        "premise_selection",
        available_capabilities={
            "bm25": True,
            "graph_read": True,
            "graphrag_retrieval": True,
            "vector_index": False,
            "embedding_retrieval": False,
            "kg_neighborhood": False,
            "learned_ranking": False,
            "smt": True,
            "logic_family_routing": True,
            "sat": False,
            "z3": False,
            "cvc5": False,
            "maxsat": False,
            "tactician_hammer": False,
            "cegis": False,
        },
    )
    assert selections
    assert all(isinstance(item, StrategySelection) for item in selections)

    op_registry = create_default_analysis_operation_registry()
    strategy_registry = attach_strategy_routing(op_registry)
    assert isinstance(strategy_registry, AnalysisStrategyRegistry)
    assert strategy_registry.strategy("retrieval").property_class is (
        PropertyQuestionClass.RETRIEVAL
    )


def test_lazy_provider_adapter_declaration_receipt() -> None:
    adapter = LazyProviderAdapter(
        provider_id="fixture-provider",
        capability_ids=("smt", "z3"),
        methods=(StrategyMethod.SMT, StrategyMethod.Z3),
        provider_version="9.9.9",
        config_digest="cfg:fixture",
    )
    receipt = adapter.declaration_receipt(
        method=StrategyMethod.SMT,
        property_class=PropertyQuestionClass.CONSTRAINT_SOLVING,
        strategy_id="strategy:test",
    )
    assert receipt.admission is CapabilityAdmission.LAZY
    assert receipt.provider_version == "9.9.9"
    assert receipt.config_digest == "cfg:fixture"
    assert receipt.details["probed"] is False
    assert receipt.details["import_inferred"] is False
    # Factory must not run on declaration.
    fired = {"value": False}

    def _factory() -> object:
        fired["value"] = True
        return object()

    with_factory = LazyProviderAdapter(
        provider_id="with-factory",
        capability_ids=("lean",),
        methods=(StrategyMethod.LEAN_KERNEL,),
        factory=_factory,
    )
    with_factory.declaration_receipt(
        method=StrategyMethod.LEAN_KERNEL,
        property_class=PropertyQuestionClass.FORMAL_KERNELS,
    )
    assert fired["value"] is False


def test_selection_payload_is_body_free_and_identifiable() -> None:
    registry = create_default_analysis_strategy_registry()
    selection = registry.select(
        PropertyQuestionClass.CONTRACTS,
        available_capabilities={
            "interface_diff": True,
            "contract_analysis": True,
            "pre_postconditions": False,
            "invariants": False,
            "weakest_precondition": False,
            "hoare": False,
        },
    )
    payload = selection.to_dict()
    assert payload["selection_id"].startswith("analysis-strategy-selection:sha256:")
    assert "source_code" not in str(payload)
    assert "body" not in payload
    assert payload["authority"]["completion_authority"] is False
