from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_operation_registry import (
    IPFS_DATASETS_ANALYSIS_PRODUCER_ID,
    LOCAL_ANALYSIS_PRODUCER_ID,
    AnalysisAuthoritySemantics,
    AnalysisBatchingSemantics,
    AnalysisCacheSemantics,
    AnalysisOperation,
    AnalysisOperationBounds,
    AnalysisOperationRegistry,
    AnalysisOperationRegistryError,
    AnalysisOperationSpec,
    AnalysisProducer,
    AnalysisProvenanceSemantics,
    LogicFamily,
    ProvenanceRequirement,
    create_default_analysis_operation_registry,
    default_operation_specs,
    normalize_analysis_operation,
    normalize_analysis_reference,
    normalize_logic_family,
)
from ipfs_accelerate_py.agent_supervisor.analysis.analysis_transport import (
    AnalysisProviderKind,
    AnalysisTransportStatus,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
    create_local_registry_logic_producer,
    create_optional_registry_logic_producer,
    registry_logic_producer_declarations,
)


EXPECTED_OPERATIONS = {
    "symbol_impact",
    "graphrag_retrieval",
    "premise_selection",
    "contradiction_search",
    "logic_translation",
    "proof_candidate_analysis",
    "counterexample_candidate_analysis",
}
EXPECTED_FAMILIES = {
    "tdfol",
    "dcec",
    "flogic",
    "modal",
    "deontic",
    "frame",
    "kg",
    "event_calculus",
}


def test_default_portfolio_declares_every_policy_dimension() -> None:
    specs = default_operation_specs()

    assert {item.operation.value for item in specs} == EXPECTED_OPERATIONS
    for spec in specs:
        record = spec.to_dict()
        assert record["cache"]["content_addressed"] is True
        assert record["cache"]["allow_stale"] is False
        assert record["bounds"]["max_batch_size"] > 0
        assert record["provenance"]["content_ids_required"] is True
        assert record["fallback"] == {
            "strategy": "deterministic_local",
            "provider_id": LOCAL_ANALYSIS_PRODUCER_ID,
            "explicit_receipt": True,
            "fail_closed": True,
        }
        assert record["batching"]["same_tree_required"] is True
        assert record["capability_requirements"]
        assert record["authority"] == {
            "verdict_tier": "diagnostic_candidate",
            "repository_mutation": False,
            "validation_omission_selection": False,
            "candidate_promotion": False,
            "proof_authority": False,
            "completion_authority": False,
        }
        assert AnalysisOperationSpec.from_dict(record) == spec


def test_logic_families_are_explicit_and_never_conflated() -> None:
    logical = {
        AnalysisOperation.PREMISE_SELECTION,
        AnalysisOperation.CONTRADICTION_SEARCH,
        AnalysisOperation.LOGIC_TRANSLATION,
        AnalysisOperation.PROOF_CANDIDATE_ANALYSIS,
        AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS,
    }
    for spec in default_operation_specs():
        if spec.operation in logical:
            assert {item.value for item in spec.logic_families} == EXPECTED_FAMILIES
            assert (
                ProvenanceRequirement.LOGIC_FAMILY in spec.provenance.required
            )
        else:
            assert spec.logic_families == ()

    assert normalize_logic_family("frame_logic") is LogicFamily.FRAME
    assert normalize_logic_family("knowledge-graph") is LogicFamily.KNOWLEDGE_GRAPH
    assert normalize_logic_family("cec") is LogicFamily.DCEC
    assert LogicFamily.FLOGIC is not LogicFamily.FRAME
    assert LogicFamily.DCEC is not LogicFamily.DEONTIC
    assert LogicFamily.TDFOL is not LogicFamily.EVENT_CALCULUS


def test_operation_compatibility_aliases_preserve_canonical_ids() -> None:
    assert normalize_analysis_operation("ast_symbol_impact").value == "symbol_impact"
    assert normalize_analysis_operation("graph_retrieval").value == (
        "graphrag_retrieval"
    )
    assert normalize_analysis_operation("legal_logic_analysis").value == (
        "logic_translation"
    )
    assert normalize_analysis_operation("proof_candidate_selection").value == (
        "proof_candidate_analysis"
    )
    with pytest.raises(AnalysisOperationRegistryError):
        normalize_analysis_operation("modify_repository")


def test_reference_normalization_unifies_local_and_remote_shapes() -> None:
    local = normalize_analysis_reference(
        {
            "id": "node:one",
            "source_id": "artifact:one",
            "path": "src/state.py",
            "symbol": "advance",
            "score": 0.75,
            "kind": "ast",
            "provider_decoration": "discarded",
        }
    )
    remote = normalize_analysis_reference(
        {
            "reference_id": "node:one",
            "artifact_id": "artifact:one",
            "path": "src/state.py",
            "symbol": "advance",
            "score_millionths": 750_000,
            "kind": "ast",
        }
    )

    assert dict(local) == dict(remote)
    assert local["score_millionths"] == 750_000
    assert normalize_analysis_reference(
        {"artifact_id": "artifact:one", "score_millionths": 1}
    )["score_millionths"] == 1
    assert "provider_decoration" not in local
    with pytest.raises(AnalysisOperationRegistryError):
        normalize_analysis_reference(
            {"artifact_id": "artifact:one", "source_code": "secret"}
        )


def test_authority_and_safety_contracts_cannot_be_enabled() -> None:
    with pytest.raises(AnalysisOperationRegistryError):
        AnalysisAuthoritySemantics(verdict_tier="validator")
    with pytest.raises(TypeError):
        AnalysisAuthoritySemantics(repository_mutation=True)  # type: ignore[call-arg]
    with pytest.raises(AnalysisOperationRegistryError):
        AnalysisCacheSemantics(allow_stale=True)
    with pytest.raises(AnalysisOperationRegistryError):
        AnalysisBatchingSemantics(same_tree_required=False)
    forged = default_operation_specs()[0].to_dict()
    forged["authority"]["repository_mutation"] = True
    with pytest.raises(AnalysisOperationRegistryError):
        AnalysisOperationSpec.from_dict(forged)


def test_operation_requires_capabilities_and_logic_provenance() -> None:
    with pytest.raises(AnalysisOperationRegistryError):
        AnalysisOperationSpec(operation=AnalysisOperation.SYMBOL_IMPACT)
    with pytest.raises(AnalysisOperationRegistryError):
        AnalysisOperationSpec(
            operation=AnalysisOperation.PREMISE_SELECTION,
            capability_requirements=("premise_selection",),
            logic_families=(LogicFamily.TDFOL,),
        )
    with pytest.raises(AnalysisOperationRegistryError):
        AnalysisOperationSpec(
            operation=AnalysisOperation.SYMBOL_IMPACT,
            capability_requirements=("symbol_impact",),
            bounds=AnalysisOperationBounds(max_batch_size=2),
            batching=AnalysisBatchingSemantics(max_batch_size=3),
        )


def test_default_registry_discovery_is_lazy_and_complete() -> None:
    registry = create_default_analysis_operation_registry()

    assert registry.frozen is False
    assert {item.operation.value for item in registry.operations()} == (
        EXPECTED_OPERATIONS
    )
    assert [item.producer_id for item in registry.producers()] == [
        LOCAL_ANALYSIS_PRODUCER_ID,
        IPFS_DATASETS_ANALYSIS_PRODUCER_ID,
    ]
    assert all(item.non_authoritative for item in registry.discover_capabilities())
    assert all(
        AnalysisProducer.from_dict(item.to_dict()) == item
        for item in registry.producers()
    )
    assert registry.frozen is False
    record = registry.to_dict()
    assert record["authority"]["repository_mutation"] is False
    assert record["authority"]["validation_omission_selection"] is False
    assert record["authority"]["candidate_promotion"] is False


def test_build_request_is_registry_tree_policy_and_family_bound() -> None:
    registry = create_default_analysis_operation_registry()
    request = registry.build_request(
        "premise_selection",
        "Which premises support the transition?",
        artifact_references=(
            {
                "artifact_id": "premise:one",
                "summary": "ready implies running",
            },
        ),
        repository_id="repo:one",
        tree_id="tree:one",
        objective_revision="objective@1",
        policy_id="policy:one",
        logic_family="tdfol",
    )

    assert request.operation == "premise_selection"
    assert request.metadata["registry_id"] == registry.registry_id
    assert request.metadata["logic_family"] == "tdfol"
    assert request.metadata["tree_id"] == "tree:one"
    with pytest.raises(AnalysisOperationRegistryError):
        registry.build_request(
            "premise_selection",
            "question",
            repository_id="repo",
            tree_id="tree",
            objective_revision="objective",
        )
    with pytest.raises(AnalysisOperationRegistryError):
        registry.build_request(
            "symbol_impact",
            "question",
            repository_id="repo",
            tree_id="tree",
            objective_revision="objective",
            logic_family="tdfol",
        )
    stale_artifact = registry.build_request(
        "symbol_impact",
        "question",
        artifact_references=(
            {"artifact_id": "artifact:stale", "tree_id": "tree:stale"},
        ),
        repository_id="repo",
        tree_id="tree:current",
        objective_revision="objective",
    )
    with pytest.raises(AnalysisOperationRegistryError):
        asyncio.run(registry.dispatch(stale_artifact))


class _FixtureProducer:
    def __init__(self, capability, *, fail: bool = False) -> None:
        self._capability = capability
        self.fail = fail
        self.calls = []

    def capabilities(self):
        return self._capability

    def analyze(self, request, *, negotiated_capability=None, **_kwargs):
        self.calls.append(request)
        if self.fail:
            raise ConnectionError("fixture provider disappeared")
        return {
            "schema": negotiated_capability.result_schema,
            "protocol_version": negotiated_capability.protocol_version,
            "request_id": request.request_id,
            "operation": request.operation,
            "capability_id": negotiated_capability.capability_id,
            "capability_revision": negotiated_capability.capability_revision,
            "evidence_references": [
                {
                    "artifact_id": "artifact:one",
                    "path": "src/state.py",
                    "score_millionths": 750_000,
                    "kind": "ast",
                },
                {
                    "kind": "ast",
                    "score_millionths": 750_000,
                    "path": "src/state.py",
                    "artifact_id": "artifact:one",
                },
            ],
            "provenance_references": [
                {
                    "record_id": "index:one",
                    "kind": "ast_index",
                }
            ],
            "cost": {"records_scanned": 1},
            "verdict": "candidate",
            "truncated": False,
            "non_authoritative": True,
            "completion_authority": False,
            "safe_for_completion_reasoning": False,
        }


def _single_operation_registry(*, optional_fail: bool = False):
    registry = AnalysisOperationRegistry()
    spec = AnalysisOperationSpec(
        operation=AnalysisOperation.SYMBOL_IMPACT,
        capability_requirements=("ast_index_read", "symbol_impact"),
        provenance=AnalysisProvenanceSemantics(),
    )
    registry.register_operation(spec)
    local_declaration = AnalysisProducer(
        producer_id="local",
        provider_kind=AnalysisProviderKind.LOCAL,
        operations=(spec.operation,),
        capability_revision="local@1",
        capabilities=spec.capability_requirements,
    )
    optional_declaration = AnalysisProducer(
        producer_id="optional",
        provider_kind=AnalysisProviderKind.IPFS_DATASETS,
        operations=(spec.operation,),
        capability_revision="optional@1",
        capabilities=spec.capability_requirements,
    )
    local = _FixtureProducer(local_declaration.capability)
    optional = _FixtureProducer(
        optional_declaration.capability, fail=optional_fail
    )
    registry.register_producer(local_declaration, provider=local)
    registry.register_producer(optional_declaration, provider=optional)
    return registry, local, optional


def _logic_registry(backend):
    registry = AnalysisOperationRegistry()
    for spec in default_operation_specs():
        if spec.logic_families:
            registry.register_operation(spec)
    local_declaration, optional_declaration = (
        registry_logic_producer_declarations()
    )
    registry.register_producer(
        local_declaration,
        provider=create_local_registry_logic_producer(),
    )
    registry.register_producer(
        optional_declaration,
        provider=create_optional_registry_logic_producer(backend=backend),
    )
    return registry


def test_dispatch_uses_optional_then_typed_local_fallback_and_dedupes() -> None:
    registry, local, optional = _single_operation_registry(optional_fail=True)
    request = registry.build_request(
        "symbol_impact",
        "What changes?",
        repository_id="repo",
        tree_id="tree",
        objective_revision="objective",
    )
    result = asyncio.run(registry.dispatch(request))

    assert result.status is AnalysisTransportStatus.FALLBACK
    assert result.provider_id == "local"
    assert result.fallback_from_provider_id == "optional"
    assert result.non_authoritative
    assert result.completion_authority is False
    assert result.safe_for_completion_reasoning is False
    assert len(result.evidence_references) == 1
    assert result.evidence_references[0]["producer_id"] == "local"
    assert len(optional.calls) == len(local.calls) == 1
    assert registry.frozen is True
    with pytest.raises(AnalysisOperationRegistryError):
        registry.register_operation(replace(default_operation_specs()[0]))


def test_explicit_local_and_remote_success_share_reference_shape() -> None:
    registry, _local, _optional = _single_operation_registry()
    local_request = registry.build_request(
        "symbol_impact",
        "What changes?",
        repository_id="repo",
        tree_id="tree",
        objective_revision="objective",
        preferred_provider_id="local",
    )
    optional_request = registry.build_request(
        "symbol_impact",
        "What changes?",
        repository_id="repo",
        tree_id="tree",
        objective_revision="objective",
        preferred_provider_id="optional",
    )
    local = asyncio.run(registry.dispatch(local_request))
    remote = asyncio.run(registry.dispatch(optional_request))

    local_shape = set(local.evidence_references[0])
    remote_shape = set(remote.evidence_references[0])
    assert local_shape == remote_shape
    assert local.evidence_references[0]["reference_id"] == (
        remote.evidence_references[0]["reference_id"]
    )
    assert local.evidence_references[0]["producer_id"] == "local"
    assert remote.evidence_references[0]["producer_id"] == "optional"


def test_stale_optional_logic_reference_uses_typed_local_fallback() -> None:
    class StaleBackend:
        def select_premises(self, _payload):
            return {
                "status": "candidate",
                "evidence_references": [
                    {
                        "artifact_id": "artifact:stale",
                        "tree_id": "tree:stale",
                    }
                ],
            }

    registry = _logic_registry(StaleBackend())
    request = registry.build_request(
        "premise_selection",
        "Which premise applies?",
        artifact_references=({"artifact_id": "artifact:current"},),
        repository_id="repo",
        tree_id="tree:current",
        objective_revision="objective",
        logic_family="tdfol",
    )

    result = asyncio.run(registry.dispatch(request))

    assert result.status is AnalysisTransportStatus.FALLBACK
    assert result.provider_id == LOCAL_ANALYSIS_PRODUCER_ID
    assert result.fallback_from_provider_id == (
        IPFS_DATASETS_ANALYSIS_PRODUCER_ID
    )
    assert result.fallback_attempted is True


def test_logic_family_binding_stays_within_provenance_bound() -> None:
    class FullProvenanceBackend:
        def select_premises(self, _payload):
            return {
                "status": "candidate",
                "evidence_references": [{"artifact_id": "artifact:one"}],
                "provenance_references": [
                    {"record_id": f"record:{index}", "kind": "provider"}
                    for index in range(64)
                ],
            }

    registry = _logic_registry(FullProvenanceBackend())
    request = registry.build_request(
        "premise_selection",
        "Which premise applies?",
        artifact_references=({"artifact_id": "artifact:one"},),
        repository_id="repo",
        tree_id="tree",
        objective_revision="objective",
        logic_family="dcec",
    )

    result = asyncio.run(registry.dispatch(request))

    assert result.status is AnalysisTransportStatus.COMPLETED
    assert len(result.provenance_references) == 64
    assert result.truncated is True
    assert any(
        item.get("kind") == "logic_family"
        and item.get("record_id") == "dcec"
        for item in result.provenance_references
    )


def test_hammer_import_uses_managed_writable_environment(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor import ipfs_datasets_logic_provider as logic_provider

    observed = {}
    original_prefix = sys.prefix

    def fake_import(name):
        observed["name"] = name
        observed["home"] = os.environ["HOME"]
        observed["prefix"] = sys.prefix
        return object()

    monkeypatch.setenv("HOME", "/read-only-home")
    monkeypatch.delenv("IPFS_DATASETS_PY_SYMAI_PREFIX", raising=False)
    monkeypatch.setattr(logic_provider.importlib, "import_module", fake_import)

    assert logic_provider._load_hammer() is not None
    assert observed["name"] == "ipfs_datasets_py.logic.hammers"
    # SymbolicAI is preloaded inside the loader's locked critical section with
    # a managed config prefix. The Hammer import itself observes restored
    # process globals rather than a temporary HOME/sys.prefix swap.
    assert observed["home"] == "/read-only-home"
    assert observed["prefix"] == original_prefix
    managed_prefix = Path(os.environ["IPFS_DATASETS_PY_SYMAI_PREFIX"])
    assert managed_prefix != Path("/usr")
    assert managed_prefix.is_dir()
    assert (managed_prefix / ".symai" / "symai.config.json").is_file()
    assert os.environ["HOME"] == "/read-only-home"
    assert sys.prefix == original_prefix


def test_batch_rejects_cross_tree_and_preserves_family_provenance() -> None:
    registry = create_default_analysis_operation_registry()
    first = registry.build_request(
        "premise_selection",
        "premises",
        repository_id="repo",
        tree_id="tree:one",
        objective_revision="objective",
        logic_family="dcec",
    )
    second = registry.build_request(
        "premise_selection",
        "premises",
        repository_id="repo",
        tree_id="tree:two",
        objective_revision="objective",
        logic_family="dcec",
    )
    with pytest.raises(AnalysisOperationRegistryError):
        asyncio.run(registry.dispatch_batch((first, second)))

    forged = replace(
        first,
        metadata={**dict(first.metadata), "registry_id": "forged"},
    )
    with pytest.raises(AnalysisOperationRegistryError):
        asyncio.run(registry.dispatch_batch((forged,)))


@pytest.mark.parametrize(
    ("operation", "logic_family"),
    [
        ("symbol_impact", None),
        ("graphrag_retrieval", None),
        ("premise_selection", "tdfol"),
        ("contradiction_search", "dcec"),
        ("logic_translation", "flogic"),
        ("proof_candidate_analysis", "modal"),
        ("counterexample_candidate_analysis", "event_calculus"),
    ],
)
def test_default_local_producer_executes_complete_portfolio(
    operation, logic_family
) -> None:
    registry = create_default_analysis_operation_registry()
    request = registry.build_request(
        operation,
        "Find impacts and contradictions in the ready transition",
        artifact_references=(
            {
                "artifact_id": "artifact:state",
                "path": "src/state.py",
                "symbol": "advance",
                "summary": "ready implies running",
            },
        ),
        repository_id="repo",
        tree_id="tree",
        objective_revision="objective",
        logic_family=logic_family,
        preferred_provider_id=LOCAL_ANALYSIS_PRODUCER_ID,
    )
    result = asyncio.run(registry.dispatch(request))

    assert result.status is AnalysisTransportStatus.COMPLETED
    assert result.provider_id == LOCAL_ANALYSIS_PRODUCER_ID
    assert result.evidence_references
    assert all(
        item["producer_id"] == LOCAL_ANALYSIS_PRODUCER_ID
        for item in result.evidence_references
    )
    assert result.non_authoritative
    assert result.completion_authority is False
    assert result.safe_for_completion_reasoning is False
    if logic_family:
        assert any(
            item.get("kind") == "logic_family"
            and item.get("record_id") == logic_family
            for item in result.provenance_references
        )


def test_default_optional_activation_failure_falls_back_without_import_probe() -> None:
    calls = []

    def unavailable():
        calls.append("activate")
        raise ModuleNotFoundError("ipfs_datasets_py")

    registry = create_default_analysis_operation_registry(
        optional_provider_factory=unavailable
    )
    request = registry.build_request(
        "logic_translation",
        "Translate the obligation",
        artifact_references=(
            {"artifact_id": "norm:one", "summary": "must validate"},
        ),
        repository_id="repo",
        tree_id="tree",
        objective_revision="objective",
        logic_family="deontic",
    )
    assert calls == []
    result = asyncio.run(registry.dispatch(request))

    assert calls == ["activate"]
    assert result.status is AnalysisTransportStatus.FALLBACK
    assert result.provider_id == LOCAL_ANALYSIS_PRODUCER_ID
    assert result.fallback_from_provider_id == (
        IPFS_DATASETS_ANALYSIS_PRODUCER_ID
    )
    assert result.fallback_attempted is True
    assert result.evidence_references


def test_producer_cannot_omit_required_capabilities_or_families() -> None:
    registry = AnalysisOperationRegistry()
    premise = next(
        item
        for item in default_operation_specs()
        if item.operation is AnalysisOperation.PREMISE_SELECTION
    )
    registry.register_operation(premise)
    with pytest.raises(AnalysisOperationRegistryError):
        registry.register_producer(
            AnalysisProducer(
                producer_id="incomplete",
                provider_kind=AnalysisProviderKind.LOCAL,
                operations=(premise.operation,),
                capability_revision="1",
                capabilities=("premise_selection",),
                logic_families=(LogicFamily.TDFOL,),
            ),
            provider=object(),
        )
