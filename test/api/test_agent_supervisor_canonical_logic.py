"""Contract tests for SupervisorCanonicalLogicAdapter@1.

Covers lossless vocabulary projections for analysis families, property kinds,
translation forms, matrix entries, capability probes, providers, routes,
resources, caches, and receipts; verifies supervisor facade compatibility,
lazy datasets imports, and cross-repo current-revision checks.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_operation_registry import (
    CacheScope,
    LogicFamily,
    to_canonical_logic_family_id,
)
from ipfs_accelerate_py.agent_supervisor.canonical_logic_adapter import (
    ADAPTER_SCHEMA_VERSION,
    SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE,
    CanonicalLogicAdapterError,
    CrossRepoRevisionReport,
    SupervisorCanonicalLogicAdapter,
    VocabularyProjection,
    _clear_import_cache_for_tests,
    check_cross_repo_revision,
    get_canonical_logic_adapter,
    map_analysis_family_to_canonical,
    map_property_kind_to_canonical,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
    to_canonical_registry_logic_family,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
    project_provider_capability_to_canonical,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    ProviderRequest,
    project_provider_request_to_canonical,
    project_resource_budget_to_canonical,
)
from ipfs_accelerate_py.agent_supervisor.proof.logic_translation_validation import (
    ApproximationDirection,
    LogicForm,
    TranslationClass,
    TranslationContract,
    project_translation_contract_to_canonical,
)
from ipfs_accelerate_py.agent_supervisor.proof.multi_prover_router import (
    PropertyKind,
    PropertyObligation,
    ProverLane,
    ProverRole,
    to_canonical_property_kind,
)
from ipfs_accelerate_py.agent_supervisor.proof.prover_matrix_registry import (
    ProverMatrixEntry,
    project_matrix_entry_to_canonical,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SOURCE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "canonical_logic_adapter.py"
)


def _adapter(**kwargs) -> SupervisorCanonicalLogicAdapter:
    return SupervisorCanonicalLogicAdapter(**kwargs)


def test_adapter_interface_and_schema_are_stable() -> None:
    adapter = _adapter()
    payload = adapter.to_dict()
    assert adapter.interface == SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE
    assert payload["schema_version"] == ADAPTER_SCHEMA_VERSION
    assert payload["interface"] == SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE
    inventory = adapter.vocabulary_inventory()
    for domain in (
        "analysis_family",
        "property_kind",
        "logic_form",
        "translation_class",
        "matrix_entry",
        "capability_probe",
        "provider",
        "route",
        "resource",
        "cache",
        "receipt",
    ):
        assert domain in inventory["domains"]


def test_importing_adapter_module_does_not_import_datasets_package() -> None:
    """Static check: the adapter source must not top-level import datasets."""

    source = ADAPTER_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("ipfs_datasets_py"), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("ipfs_datasets_py"), module

    # Runtime: adapter may use datasets only through its lazy importer cache.
    _clear_import_cache_for_tests()
    # Re-importing is fine; the cache is empty after the clear.
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.canonical_logic_adapter"
    )
    adapter = module.SupervisorCanonicalLogicAdapter()
    assert adapter.datasets_import_is_lazy() is True


def test_analysis_families_round_trip_losslessly() -> None:
    adapter = _adapter()
    for family in adapter.vocabulary_inventory()["analysis_families"]:
        projection = adapter.project_analysis_family(family)
        restored = adapter.restore_analysis_family(projection)
        assert restored.value == family
        assert projection.domain == "analysis_family"
        assert projection.to_dict()["canonical_id"] == projection.canonical_id
        # Facade helper stays compatible.
        assert to_canonical_logic_family_id(family) == projection.canonical_id
        assert map_analysis_family_to_canonical(family) == projection.canonical_id


def test_flogic_and_frame_remain_distinct_under_shared_canonical_id() -> None:
    adapter = _adapter()
    flogic = adapter.project_analysis_family(LogicFamily.FLOGIC)
    frame = adapter.project_analysis_family(LogicFamily.FRAME)
    assert flogic.canonical_id == frame.canonical_id == "frame_logic"
    assert adapter.restore_analysis_family(flogic) is LogicFamily.FLOGIC
    assert adapter.restore_analysis_family(frame) is LogicFamily.FRAME


def test_property_kinds_round_trip_losslessly() -> None:
    adapter = _adapter()
    for kind in adapter.vocabulary_inventory()["property_kinds"]:
        projection = adapter.project_property_kind(kind)
        restored = adapter.restore_property_kind(projection)
        assert restored.value == kind
        assert to_canonical_property_kind(kind) == projection.canonical_id
        assert map_property_kind_to_canonical(kind) == projection.canonical_id


def test_protocol_and_runtime_trace_share_canonical_kind_but_restore_exactly() -> None:
    adapter = _adapter()
    protocol = adapter.project_property_kind(PropertyKind.PROTOCOL)
    runtime = adapter.project_property_kind(PropertyKind.RUNTIME_TRACE)
    assert protocol.canonical_id == runtime.canonical_id == "trace_conformance"
    assert adapter.restore_property_kind(protocol) is PropertyKind.PROTOCOL
    assert adapter.restore_property_kind(runtime) is PropertyKind.RUNTIME_TRACE


def test_logic_forms_and_translation_classes_round_trip() -> None:
    adapter = _adapter()
    for form in adapter.vocabulary_inventory()["logic_forms"]:
        projection = adapter.project_logic_form(form)
        assert adapter.restore_logic_form(projection).value == form
    for translation in adapter.vocabulary_inventory()["translation_classes"]:
        projection = adapter.project_translation_class(translation)
        assert adapter.restore_translation_class(projection).value == translation
        assert "taxonomy_translation_kind" in projection.residual


def test_translation_contract_projects_forms_and_class() -> None:
    contract = TranslationContract(
        contract_id="contract:smt",
        source_identity="source:sha256:aa",
        source_form=LogicForm.AST,
        target_form=LogicForm.SMT_LIB,
        translator_id="fixture-translator",
        translator_version="1.0.0",
        translator_identity="translator:sha256:bb",
        semantic_profile_id="profile:default",
        semantic_profile_version="1",
        translation_class=TranslationClass.EQUISATISFIABLE,
        fixture_set_id="fixture:set-1",
        approximation_direction=ApproximationDirection.NONE,
    )
    adapter = _adapter()
    projected = adapter.project_translation_contract(contract)
    assert projected["interface"] == SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE
    source, target, translation = adapter.restore_translation_contract_forms(projected)
    assert source is LogicForm.AST
    assert target is LogicForm.SMT_LIB
    assert translation is TranslationClass.EQUISATISFIABLE
    # Module facade remains compatible.
    via_facade = project_translation_contract_to_canonical(contract)
    assert via_facade["content_id"] == projected["content_id"]


def test_cache_scopes_round_trip() -> None:
    adapter = _adapter()
    for scope in adapter.vocabulary_inventory()["cache_scopes"]:
        projection = adapter.project_cache_scope(scope)
        assert adapter.restore_cache_scope(projection).value == scope


def test_resource_budget_round_trip_and_provider_shape() -> None:
    budget = ResourceBudget(
        wall_time_ms=2_000,
        cpu_time_ms=1_000,
        memory_bytes=64 * 1024 * 1024,
        disk_bytes=1_024,
        max_processes=2,
        max_premises=8,
        max_output_bytes=4_096,
        model_token_limit=128,
        provider_quota=1,
        network_allowed=True,
    )
    adapter = _adapter()
    projected = adapter.project_resource_budget(budget)
    assert projected["schema_version"] == (
        "ipfs_datasets_py/logic-provider-resource-budget@1"
    )
    assert projected["wall_time_ms"] == 2_000
    assert projected["network_allowed"] is True
    restored = adapter.restore_resource_budget(projected)
    assert restored.wall_time_ms == budget.wall_time_ms
    assert restored.memory_bytes == budget.memory_bytes
    assert restored.network_allowed is True
    assert project_resource_budget_to_canonical(budget)["cpu_time_ms"] == 1_000


def test_provider_request_projects_to_canonical_datasets_type() -> None:
    request = ProviderRequest(
        operation=ProofProviderOperation.PROVE,
        payload={"obligation_id": "obligation:1"},
        request_id="request-canonical-logic-1",
        resource_budget=ResourceBudget(wall_time_ms=500, memory_bytes=1024),
        network_allowed=False,
    )
    adapter = _adapter()
    canonical = adapter.project_provider_request(request)
    assert canonical.request_id == request.request_id
    assert canonical.operation.value == "prove"
    assert canonical.resource_budget.wall_time_ms == 500
    # Facade helper.
    again = project_provider_request_to_canonical(request)
    assert again.request_id == request.request_id


def test_capability_probe_projects_without_claiming_proof_success() -> None:
    capability = ProofProviderCapability(
        provider_id="fixture.z3",
        provider_version="4.12.0",
        operations=(
            ProofProviderOperation.CAPABILITY,
            ProofProviderOperation.PROVE,
        ),
        isolation=(ProofProviderIsolation.SUBPROCESS,),
        network_access_required=False,
        resource_limits_supported=True,
        metadata={"lane": "smt"},
    )
    adapter = _adapter()
    projected = adapter.project_provider_capability(capability)
    assert projected["proof_attempted"] is False
    assert projected["proof_success"] is False
    assert projected["runtimes"] == ["native_process"]
    restored = adapter.restore_provider_capability(projected)
    assert restored.provider_id == "fixture.z3"
    assert ProofProviderOperation.PROVE in restored.operations
    assert project_provider_capability_to_canonical(capability)["provider_id"] == (
        "fixture.z3"
    )


def test_route_lane_and_property_obligation_project_losslessly() -> None:
    adapter = _adapter()
    lane = ProverLane(
        prover_id="z3",
        role=ProverRole.MODEL_CHECKER,
        stage=0,
        authority_capability="finite_constraint",
    )
    projected_lane = adapter.project_prover_lane(lane)
    restored_lane = adapter.restore_prover_lane(projected_lane)
    assert restored_lane.prover_id == "z3"
    assert restored_lane.role is ProverRole.MODEL_CHECKER

    obligation = PropertyObligation(
        obligation_id="obligation:auth",
        property_kind=PropertyKind.AUTHORIZATION,
        statement="(assert (not unauthorized))",
        premise_ids=("premise:policy",),
    )
    projected = adapter.project_property_obligation(obligation)
    assert projected["property_kind"]["canonical_id"] == "authorization"
    assert projected["supervisor_obligation"]["obligation_id"] == "obligation:auth"


def test_matrix_entry_projects_and_restores_supervisor_payload() -> None:
    # Minimal discovered-but-not-smoke-tested entry (no receipt required).
    entry = ProverMatrixEntry(
        prover_id="z3",
        display_name="Z3",
        family="smt",
        absent=False,
        discovered=True,
        versioned=True,
        smoke_tested=False,
        translation_conformant=False,
        reconstruction_capable=False,
        authoritative_for=(),
        executable_path="/usr/bin/z3",
        executable_version="4.12.0",
        package_module=None,
        package_version=None,
        reason="discovered executable without self-test",
    )
    adapter = _adapter()
    projected = adapter.project_matrix_entry(entry)
    assert projected["domain"] == "matrix_entry"
    assert projected["canonical_provider_id"] == "z3"
    restored = adapter.restore_matrix_entry_payload(projected)
    assert restored["prover_id"] == "z3"
    assert restored["family"] == "smt"
    assert project_matrix_entry_to_canonical(entry)["prover_id"] == "z3"


@pytest.mark.parametrize(
    ("supervisor_id", "canonical_id", "family"),
    (
        ("coq", "rocq", "kernel"),
        ("e", "eprover", "atp"),
    ),
)
def test_prover_projection_uses_canonical_datasets_dispatch_id(
    supervisor_id: str,
    canonical_id: str,
    family: str,
) -> None:
    adapter = _adapter()
    entry = ProverMatrixEntry(
        prover_id=supervisor_id,
        display_name=supervisor_id,
        family=family,
        absent=False,
        discovered=True,
        versioned=False,
        smoke_tested=False,
        translation_conformant=False,
        reconstruction_capable=False,
        authoritative_for=(),
        executable_path=None,
        executable_version=None,
        package_module="fixture.provider",
        package_version=None,
        reason="fixture",
    )
    lane = ProverLane(
        prover_id=supervisor_id,
        role=ProverRole.KERNEL if family == "kernel" else ProverRole.MODEL_CHECKER,
        stage=1,
        authority_capability="fixture",
    )

    matrix_projection = adapter.project_matrix_entry(entry)
    lane_projection = adapter.project_prover_lane(lane)

    assert matrix_projection["prover_id"] == supervisor_id
    assert matrix_projection["canonical_provider_id"] == canonical_id
    assert matrix_projection["supervisor_entry"]["prover_id"] == supervisor_id
    assert lane_projection["prover_id"] == supervisor_id
    assert lane_projection["canonical_provider_id"] == canonical_id
    assert lane_projection["supervisor_lane"]["prover_id"] == supervisor_id


def test_proof_cache_key_projects_to_verification_cache_shape() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import (
        ProofCacheKey,
    )

    key = ProofCacheKey(
        obligation="obligation:sha256:aa",
        premises=("premise:sha256:bb",),
        translator="translator:v1",
        solver="z3",
        kernel="none",
        toolchain="toolchain:sha256:cc",
        theorem_registry="registry:sha256:dd",
        policy="policy:sha256:ee",
        resource_budget="budget:sha256:ff",
        candidate_tree="tree:sha256:11",
    )
    adapter = _adapter()
    projected = adapter.project_proof_cache_key(key)
    assert projected["schema_version"] == "verification-cache-key/v1"
    assert projected["backend_id"] == "z3"
    assert projected["tree_digest"]
    restored = adapter.restore_proof_cache_key_payload(projected)
    assert restored["solver"] == "z3"
    assert restored["obligation"] == "obligation:sha256:aa"


def test_translation_validation_receipt_never_upgrades_authority() -> None:
    adapter = _adapter()
    projected = adapter.project_translation_validation_receipt(
        {
            "valid": True,
            "issues": [],
            "contract_identity": "contract:1",
            "source_identity": "source:1",
            "target_identity": "target:1",
            "content_id": "receipt:1",
        }
    )
    assert projected["authority"] == "none"
    assert projected["proof_attempted"] is False
    assert projected["proof_success"] is False
    assert projected["valid"] is True
    restored = adapter.restore_translation_validation_payload(projected)
    assert restored["contract_identity"] == "contract:1"


def test_vocabulary_projection_serialization_round_trip() -> None:
    projection = VocabularyProjection(
        domain="analysis_family",
        supervisor_id="tdfol",
        canonical_id="tdfol",
        residual={"extra": "kept"},
    )
    restored = VocabularyProjection.from_dict(projection.to_dict())
    assert restored.supervisor_id == "tdfol"
    assert restored.residual["extra"] == "kept"
    assert restored.residual["supervisor_id"] == "tdfol"


def test_unknown_tokens_fail_closed() -> None:
    adapter = _adapter()
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_logic_form("not-a-form")
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_translation_class("not-a-class")
    with pytest.raises(CanonicalLogicAdapterError):
        adapter.project_cache_scope("not-a-scope")


def test_lazy_datasets_family_and_property_discovery() -> None:
    _clear_import_cache_for_tests()
    adapter = _adapter()
    assert adapter.datasets_import_is_lazy() is True
    families = adapter.list_canonical_family_ids()
    assert "tdfol" in families
    assert "dcec" in families
    assert "first_order" in families
    properties = adapter.list_canonical_property_ids()
    assert "authorization" in properties
    assert "hyperproperty" in properties
    assert adapter.datasets_import_is_lazy() is False
    receipt_type = adapter.load_canonical_translation_receipt_type()
    assert receipt_type.__name__ == "LogicTranslationReceipt"


def test_cross_repo_revision_check_passes_for_current_checkout() -> None:
    report = check_cross_repo_revision(
        repo_root=REPO_ROOT,
        require_git_alignment=True,
    )
    assert isinstance(report, CrossRepoRevisionReport)
    assert report.aligned is True
    assert report.interface == SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE
    assert report.required_modules["ipfs_datasets_py.logic"] is True
    assert report.required_modules["ipfs_datasets_py.logic.backends.provider"] is True
    assert report.required_modules["ipfs_datasets_py.logic.verification_api"] is True
    payload = report.to_dict()
    assert payload["aligned"] is True


def test_hammer_registry_family_facade_uses_adapter() -> None:
    assert to_canonical_registry_logic_family(LogicFamily.DCEC) == "dcec"
    assert to_canonical_registry_logic_family("f_logic") == "frame_logic"


def test_get_canonical_logic_adapter_singleton_and_overrides() -> None:
    first = get_canonical_logic_adapter()
    second = get_canonical_logic_adapter()
    assert first is second
    override = get_canonical_logic_adapter(repo_root=REPO_ROOT)
    assert override is not first
    assert override._repo_root == REPO_ROOT


def test_supervisor_facades_remain_import_compatible() -> None:
    """Existing local facades still export their symbols after adapter wiring."""

    from ipfs_accelerate_py.agent_supervisor.analysis import analysis_operation_registry
    from ipfs_accelerate_py.agent_supervisor.proof import (
        formal_verification_capabilities,
        formal_verification_provider,
        logic_translation_validation,
        multi_prover_router,
        prover_matrix_registry,
    )
    from ipfs_accelerate_py.agent_supervisor.integrations import (
        ipfs_datasets_logic_provider,
    )

    assert hasattr(analysis_operation_registry, "to_canonical_logic_family_id")
    assert hasattr(multi_prover_router, "to_canonical_property_kind")
    assert hasattr(logic_translation_validation, "project_translation_contract_to_canonical")
    assert hasattr(prover_matrix_registry, "project_matrix_entry_to_canonical")
    assert hasattr(
        formal_verification_capabilities, "project_provider_capability_to_canonical"
    )
    assert hasattr(formal_verification_provider, "project_provider_request_to_canonical")
    assert hasattr(ipfs_datasets_logic_provider, "to_canonical_registry_logic_family")
    # Core local APIs remain.
    assert analysis_operation_registry.LogicFamily.TDFOL.value == "tdfol"
    assert multi_prover_router.PropertyKind.AUTHORIZATION.value == "authorization"
    assert formal_verification_provider.PROOF_PROVIDER_PROTOCOL_VERSION == 1


def test_adapter_source_does_not_move_orchestration_into_datasets() -> None:
    source = ADAPTER_SOURCE.read_text(encoding="utf-8")
    # Thin adapter: no process launchers, no scheduler loops, no network servers.
    forbidden = (
        "subprocess.Popen",
        "socket.socket",
        "http.server",
        "ResourceScheduler",
        "MultiProverRouter().execute",
    )
    for token in forbidden:
        assert token not in source, token
