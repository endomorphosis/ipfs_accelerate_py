"""SAWM-015 accelerator semantic-world operational adapter tests."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_index.models import RepositoryState
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_domain_adapters import (
    DomainCapabilityUnavailable,
    ProgramWorldDomainAdapter,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_identity import (
    SemanticObjectEnvelope,
    SemanticObjectKind,
    SemanticWorldRootIdentity,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_relations import (
    ProgramRelationClaim,
    RelationAuthorityStatus,
    RelationKind,
    RelationScope,
    RelationScopeKind,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_transition import (
    ProgramTransitionQuery,
    QueryFamily,
)
from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import DurableCoordinationStore
from ipfs_kit_py.semantic_world_store.verified_store import VerifiedSemanticBlockStore

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_adapters import (
    ADAPTER_ID,
    ANN_SURFACE,
    DATASETS_SURFACE,
    KIT_SURFACE,
    DatasetsProgramWorldAdapter,
    ExecutionTransitionCompiler,
    KitProgramWorldAdapter,
    OperationalWorldRootPublisher,
    ProgramWorldAdapterError,
    ProgramWorldCapabilityUnavailable,
    SemanticWorldOperationalAdapters,
    assert_no_duplicate_semantic_world_authority,
    inspect_ann_capability,
    inspect_datasets_capability,
    inspect_kit_capability,
    load_semantic_world_operational_adapters,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    EVENT_AUTHORITY,
    MERGE_AUTHORITY,
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    PROGRAM_WORLD_CONTEXT_RECEIPT_INTERFACE,
    PROGRAM_WORLD_REUSE_DECISION_INTERFACE,
    SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE,
    VALIDATION_AUTHORITY,
    ExecutionTransitionCompilation,
    OperationalWorldRootPublicationRequest,
    ProgramWorldAdmissionError,
    ProgramWorldCapabilityReceipt,
    ProgramWorldContextItem,
    ProgramWorldContextReceipt,
    ProgramWorldReceiptError,
    ProgramWorldReuseDecision,
    available_capability,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_adapters.py"
)
RECEIPTS_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_receipts.py"
)
_OPT_OUTS = {
    "IPFS_DATASETS_AUTO_INSTALL": "0",
    "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
    "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
    "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _envelope(**overrides: Any) -> SemanticObjectEnvelope:
    fields: dict[str, Any] = {
        "kind": SemanticObjectKind.SYMBOL,
        "language": "python",
        "repository_id": "repo:sawm-015",
        "logical_name": "pkg.mod.answer",
        "declaration_cid": _cid("declaration"),
        "source_cid": _cid("source"),
        "environment_binding_cid": None,
        "metadata": {},
    }
    fields.update(overrides)
    return SemanticObjectEnvelope(**fields)


def _world_root(**overrides: Any) -> SemanticWorldRootIdentity:
    fields: dict[str, Any] = {
        "domain_state_cid": _cid("domain-state"),
        "canonical_program_graph_cid": _cid("graph"),
        "program_graph_snapshot_cid": _cid("snapshot"),
        "semantic_object_index_cid": _cid("objects"),
        "environment_binding_set_cid": _cid("env-set"),
        "policy_cid": _cid("policy"),
        "analysis_limitation_index_cid": _cid("limitations"),
    }
    fields.update(overrides)
    return SemanticWorldRootIdentity(**fields)


def _scope(**overrides: Any) -> RelationScope:
    fields: dict[str, Any] = {
        "scope_kind": RelationScopeKind.MODULE,
        "language": "python",
        "subject_cids": [_cid("mod.a"), _cid("mod.b")],
        "theory_or_policy_cid": _cid("theory-v1"),
        "environment_binding_cid": _cid("env-v1"),
        "assumption_cids": [_cid("asm-a")],
        "unavailable_dimensions": ("native_stack",),
    }
    fields.update(overrides)
    return RelationScope(**fields)


def _claim(scope: RelationScope | None = None, **overrides: Any) -> ProgramRelationClaim:
    bound = scope or _scope()
    fields: dict[str, Any] = {
        "relation_kind": RelationKind.EQUALITY,
        "left_cid": _cid("left"),
        "right_cid": _cid("right"),
        "scope_cid": bound.scope_cid,
        "theory_or_policy_cid": bound.theory_or_policy_cid,
        "environment_binding_cid": bound.environment_binding_cid,
        "authority_status": RelationAuthorityStatus.CANDIDATE,
        "assumption_cids": bound.assumption_cids,
        "evidence_cids": (),
        "invalidator_cids": (),
        "superseded_by_cid": None,
    }
    fields.update(overrides)
    return ProgramRelationClaim(**fields)


def _query(**overrides: Any) -> ProgramTransitionQuery:
    fields: dict[str, Any] = {
        "query_family": QueryFamily.NEXT_CALL,
        "language": "python",
        "subject_cid": _cid("state-current"),
        "current_source_cid": _cid("source-current"),
        "current_state_cid": _cid("state-current"),
        "environment_binding_cid": _cid("env-v1"),
        "policy_cid": _cid("policy-v1"),
        "allowed_symbol_cids": [_cid("sym-a")],
        "evidence_cids": [_cid("ev-1")],
        "unavailable_dimensions": ("native_stack",),
    }
    fields.update(overrides)
    return ProgramTransitionQuery(**fields)


def _context_item(
    label: str,
    *,
    disposition: str = "included",
    required: bool = False,
    reason: str = "exact_identity",
    kind: str = "semantic_object",
) -> ProgramWorldContextItem:
    return ProgramWorldContextItem(
        identity_cid=_cid(label),
        kind=kind,
        reason=reason,
        authority="ipfs_datasets_py.logic.software_contracts.semantic_state.program_identity",
        freshness="fresh",
        disposition=disposition,
        required=required,
        source_cid=_cid(f"{label}-source"),
    )


def _unavailable_loader(surface: str = DATASETS_SURFACE):
    def _load() -> Any:
        raise ProgramWorldCapabilityUnavailable(
            "load",
            "import_failed",
            f"{surface} unavailable in this probe",
            retryable=True,
            surface=surface,
        )

    return _load


# ---------------------------------------------------------------------------
# Interfaces, symbols, AST boundary
# ---------------------------------------------------------------------------


def test_public_interfaces_and_predicted_symbols() -> None:
    assert SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE == (
        "SemanticWorldOperationalAdapters@1"
    )
    assert PROGRAM_WORLD_REUSE_DECISION_INTERFACE == "ProgramWorldReuseDecision@1"
    assert PROGRAM_WORLD_CONTEXT_RECEIPT_INTERFACE == "ProgramWorldContextReceipt@1"
    assert ADAPTER_ID.startswith("ipfs-accelerate.semantic-state.program-world")
    assert OPERATIONAL_ACCEPTANCE_AUTHORITIES == (
        VALIDATION_AUTHORITY,
        MERGE_AUTHORITY,
        EVENT_AUTHORITY,
    )
    predicted = {
        "DatasetsProgramWorldAdapter",
        "KitProgramWorldAdapter",
        "ProgramWorldReuseDecision",
        "ProgramWorldContextReceipt",
        "ExecutionTransitionCompiler",
        "OperationalWorldRootPublisher",
    }
    import ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_adapters as adapters
    import ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts as receipts

    names = set(adapters.__all__) | set(receipts.__all__)
    assert predicted <= names
    assert_no_duplicate_semantic_world_authority()


def test_adapters_do_not_duplicate_datasets_kit_or_context_compiler() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    banned = {
        "SemanticObjectEnvelope",
        "ProgramRelationClaim",
        "ProgramTransitionQuery",
        "ContextCompiler",
        "DurableCoordinationStore",
        "VerifiedSemanticBlockStore",
        "ProgramWorldReuseGate",
    }
    assert not (defined & banned)
    cas_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compare_and_swap_root"
    ]
    assert cas_calls == []
    receipts_source = RECEIPTS_PATH.read_text(encoding="utf-8")
    for forbidden in ("compare_and_swap_root", "class ContextCompiler"):
        assert forbidden not in receipts_source


def test_cold_import_loads_neither_datasets_program_world_nor_kit_store() -> None:
    script = r"""
import json
import os
import sys
import threading

effects = []

def forbidden(name):
    def call(*args, **kwargs):
        effects.append(name)
        raise AssertionError(f"forbidden import side effect: {name}")
    return call

os.system = forbidden("os.system")
_orig_start = threading.Thread.start

def _thread_start(self, *args, **kwargs):
    effects.append("threading.Thread.start")
    raise AssertionError("forbidden import side effect: threading.Thread.start")

threading.Thread.start = _thread_start

import ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_adapters as adapters
import ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts as receipts

identity_loaded = any(
    name.startswith(
        "ipfs_datasets_py.logic.software_contracts.semantic_state.program_"
    )
    for name in sys.modules
)
kit_loaded = any("semantic_world_store" in name for name in sys.modules)
print(json.dumps({
    "effects": effects,
    "identity_loaded": identity_loaded,
    "kit_loaded": kit_loaded,
    "interface": adapters.SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE,
    "reuse_interface": receipts.PROGRAM_WORLD_REUSE_DECISION_INTERFACE,
}))
"""
    env = dict(os.environ)
    env.update(_OPT_OUTS)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT),
            str(REPO_ROOT / "ipfs_datasets_py"),
            str(REPO_ROOT / "ipfs_kit_py"),
            env.get("PYTHONPATH", ""),
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["effects"] == []
    assert payload["identity_loaded"] is False
    assert payload["kit_loaded"] is False
    assert payload["interface"] == "SemanticWorldOperationalAdapters@1"
    assert payload["reuse_interface"] == "ProgramWorldReuseDecision@1"


# ---------------------------------------------------------------------------
# Datasets identity preservation
# ---------------------------------------------------------------------------


def test_datasets_adapter_preserves_semantic_object_and_world_root_identity() -> None:
    adapter = DatasetsProgramWorldAdapter()
    envelope = _envelope()
    before = envelope.to_dict()
    cited = adapter.cite_semantic_object(envelope)
    assert cited.identity_cid == envelope.semantic_object_cid
    assert cited.identity_cid != ADAPTER_ID
    assert cited.adapter_may_replace_identity is False
    assert cited.retains_source_authority is True
    assert envelope.to_dict() == before

    root = _world_root()
    cited_root = adapter.cite_semantic_world_root(root)
    assert cited_root.identity_cid == root.semantic_world_root_cid
    assert cited_root.kind == "semantic_world_root"


def test_datasets_adapter_preserves_relation_and_transition_identity() -> None:
    adapter = DatasetsProgramWorldAdapter()
    claim = _claim()
    cited = adapter.cite_relation(claim)
    assert cited.identity_cid == claim.relation_claim_cid
    query = _query()
    cited_query = adapter.cite_transition(query)
    assert cited_query.identity_cid == query.query_cid


def test_datasets_adapter_preserves_domain_identity() -> None:
    adapter = DatasetsProgramWorldAdapter()
    state = RepositoryState("repo:sawm-015")
    before = state.state_cid
    adapted = adapter.cite_domain("repository_semantic_state", state)
    assert isinstance(adapted, ProgramWorldDomainAdapter)
    assert adapted.domain_identity == before
    assert adapted.adapter_may_replace_domain_identity is False
    assert adapted.retains_domain_authority is True
    assert state.state_cid == before


def test_datasets_adapter_returns_typed_unavailability_for_missing_domains() -> None:
    adapter = DatasetsProgramWorldAdapter()
    result = adapter.cite_domain("ann_index", None)
    assert isinstance(result, DomainCapabilityUnavailable)
    assert result.availability == "unavailable"
    assert result.domain_kind == "ann_index"


# ---------------------------------------------------------------------------
# Fake authority rejection
# ---------------------------------------------------------------------------


def test_fake_datasets_authority_is_rejected() -> None:
    surface = SimpleNamespace(
        capability=available_capability(DATASETS_SURFACE, operations=("cite_semantic_object",))
    )
    adapter = DatasetsProgramWorldAdapter(surface=surface)

    class SemanticObjectEnvelope:  # noqa: N801 - intentional fake authority
        semantic_object_cid = _cid("forged")
        SCHEMA = "ipfs-datasets.software-contracts.semantic-object-envelope@1"

    with pytest.raises(ProgramWorldAdapterError, match="fake datasets authority"):
        adapter.cite_semantic_object(SemanticObjectEnvelope())


def test_fake_kit_authority_is_rejected() -> None:
    surface = SimpleNamespace(
        capability=available_capability(KIT_SURFACE, operations=("consume_verified_semantic_object",))
    )
    adapter = KitProgramWorldAdapter(surface=surface)

    class VerifiedSemanticBlockStore:  # noqa: N801 - intentional fake authority
        INTERFACE = "VerifiedSemanticStore@1"

        def get_verified_semantic_object(self, cid: str) -> Any:
            return SimpleNamespace(semantic_object_cid=cid)

    with pytest.raises(ProgramWorldAdapterError, match="fake kit authority"):
        adapter.consume_verified_semantic_object(_cid("obj"), store=VerifiedSemanticBlockStore())


def test_ann_score_cannot_become_datasets_identity() -> None:
    adapter = DatasetsProgramWorldAdapter()
    with pytest.raises(ProgramWorldAdapterError, match="ANN/score"):
        adapter.cite_semantic_object({"semantic_object_cid": _cid("obj"), "ann_score": 0.9})


# ---------------------------------------------------------------------------
# Kit verified consumption
# ---------------------------------------------------------------------------


def test_kit_adapter_consumes_reverified_semantic_object(tmp_path: Path) -> None:
    coordination = DurableCoordinationStore(tmp_path / "kit-store")
    try:
        store = VerifiedSemanticBlockStore(coordination)
        envelope = _envelope()
        store.put_semantic_object(envelope, operation_id="sawm-015-subject")
        adapter = KitProgramWorldAdapter()
        cited = adapter.consume_verified_semantic_object(
            envelope.semantic_object_cid, store=store
        )
        assert cited.identity_cid == envelope.semantic_object_cid
        assert cited.advisory is False
        assert cited.authoritative is False
        assert cited.kind == "semantic_object"
    finally:
        coordination.close()


def test_kit_projection_consumption_remains_advisory(tmp_path: Path) -> None:
    from ipfs_datasets_py.logic.software_contracts.semantic_state.program_identity import (
        ProjectionIdentity,
    )
    from ipfs_kit_py.semantic_world_store.verified_store import ProjectionRecord

    coordination = DurableCoordinationStore(tmp_path / "kit-proj")
    try:
        store = VerifiedSemanticBlockStore(coordination)
        envelope = _envelope()
        store.put_semantic_object(envelope, operation_id="sawm-015-proj-subject")
        packed = store.put_vector_bytes(
            [1.0, 2.0, 3.0, 4.0],
            dtype="float32",
            byte_order="little",
            dimension=4,
            operation_id="sawm-015-vector",
        )
        vector = store.get_verified_vector(packed.cid)
        record = ProjectionRecord.from_payload(
            ProjectionIdentity(
                subject_cid=envelope.semantic_object_cid,
                subject_kind="semantic_object",
                projection_kind="embedding",
                model_cid=_cid("model-a"),
                tokenizer_cid=_cid("tokenizer-a"),
                preprocessing_profile_cid=_cid("preproc-a"),
                normalization_profile_cid=_cid("norm-a"),
                dimension=vector.dimension,
                metric="cosine",
                dtype=vector.dtype,
                byte_order=vector.byte_order,
                quantization_profile_cid=_cid("quant-a"),
                vector_cid=vector.bytes_cid,
                privacy_class="internal",
                availability_policy="available",
                authoritative=False,
            ).to_dict()
        )
        store.put_projection_record(record, operation_id="sawm-015-proj")
        adapter = KitProgramWorldAdapter()
        cited = adapter.consume_verified_projection(record.projection_cid, store=store)
        assert cited.advisory is True
        assert cited.authoritative is False
        assert cited.kind == "projection"
    finally:
        coordination.close()


# ---------------------------------------------------------------------------
# Capability / unavailability / ANN advisory
# ---------------------------------------------------------------------------


def test_unavailable_datasets_capability_is_visible_fallback() -> None:
    adapter = DatasetsProgramWorldAdapter(loader=_unavailable_loader(DATASETS_SURFACE))
    cap = adapter.capability
    assert cap.available is False
    assert cap.fallback is True
    assert cap.status == "unavailable"
    assert cap.reason_code == "import_failed"
    result = cap.to_unavailable_result("cite_semantic_object")
    assert result.adapter_id == ADAPTER_ID
    with pytest.raises(ProgramWorldCapabilityUnavailable):
        adapter.cite_semantic_object(_envelope())


def test_unavailable_kit_capability_is_visible_fallback() -> None:
    adapter = KitProgramWorldAdapter(loader=_unavailable_loader(KIT_SURFACE))
    cap = adapter.capability
    assert cap.fallback is True
    assert cap.surface == KIT_SURFACE
    facade = SemanticWorldOperationalAdapters(
        datasets=DatasetsProgramWorldAdapter(),
        kit=adapter,
    )
    probes = facade.probe_capabilities()
    assert probes["kit"].fallback is True
    assert probes["ann"].fallback is True
    assert probes["ann"].reason_code in {"import_failed", "ann_advisory_only"}


def test_ann_capability_is_always_advisory() -> None:
    receipt = inspect_ann_capability()
    assert receipt.fallback is True
    assert receipt.reason_code == "ann_advisory_only"
    assert receipt.surface == ANN_SURFACE
    assert receipt.available is False


def test_inspect_live_datasets_and_kit_capabilities() -> None:
    datasets = inspect_datasets_capability()
    kit = inspect_kit_capability()
    assert datasets.available is True
    assert kit.available is True
    assert "cite_semantic_object" in datasets.operations
    assert "consume_verified_semantic_object" in kit.operations


# ---------------------------------------------------------------------------
# Reuse / context / compilation / publication
# ---------------------------------------------------------------------------


def test_exact_reuse_is_proposal_until_supervisor_admission() -> None:
    facade = load_semantic_world_operational_adapters()
    state = _cid("state")
    decision = facade.evaluate_reuse(
        state_cid=state,
        goal_cid=_cid("goal"),
        policy_cid=_cid("policy"),
        environment_cid=_cid("env"),
        toolchain_cid=_cid("toolchain"),
        prior_state_cid=state,
    )
    assert isinstance(decision, ProgramWorldReuseDecision)
    assert decision.verdict == "reuse"
    assert decision.exact_match is True
    assert decision.proposal_only is True
    assert decision.admitted is False
    assert decision.ann_authoritative is False
    assert decision.operational_acceptance_authorities == OPERATIONAL_ACCEPTANCE_AUTHORITIES
    round_trip = ProgramWorldReuseDecision.from_dict(decision.to_dict())
    assert round_trip.decision_cid == decision.decision_cid


def test_ann_candidates_cannot_admit_reuse() -> None:
    facade = load_semantic_world_operational_adapters()
    decision = facade.evaluate_reuse(
        state_cid=_cid("state"),
        goal_cid=_cid("goal"),
        policy_cid=_cid("policy"),
        environment_cid=_cid("env"),
        toolchain_cid=_cid("toolchain"),
        prior_state_cid=_cid("state"),
        ann_candidates=({"score": 0.99, "nearest": True},),
    )
    assert decision.verdict == "reject"
    assert decision.reason_code == "ann_not_authoritative"
    assert decision.exact_match is False
    assert decision.admitted is False


def test_reuse_self_admission_is_rejected() -> None:
    facade = load_semantic_world_operational_adapters()
    with pytest.raises(ProgramWorldAdmissionError, match="cannot admit"):
        facade.evaluate_reuse(
            state_cid=_cid("state"),
            goal_cid=_cid("goal"),
            policy_cid=_cid("policy"),
            environment_cid=_cid("env"),
            toolchain_cid=_cid("toolchain"),
            prior_state_cid=_cid("state"),
            admission_authority=ADAPTER_ID,
            admission_evidence_cid=_cid("forged-admission"),
        )


def test_reuse_unavailable_when_datasets_surface_missing() -> None:
    facade = SemanticWorldOperationalAdapters(
        datasets=DatasetsProgramWorldAdapter(loader=_unavailable_loader()),
        kit=KitProgramWorldAdapter(loader=_unavailable_loader(KIT_SURFACE)),
    )
    decision = facade.evaluate_reuse(
        state_cid=_cid("state"),
        goal_cid=_cid("goal"),
        policy_cid=_cid("policy"),
        environment_cid=_cid("env"),
        toolchain_cid=_cid("toolchain"),
        prior_state_cid=_cid("state"),
    )
    assert decision.verdict == "unavailable"
    assert decision.fallback is True
    assert decision.reason_code == "import_failed"


def test_context_receipt_records_inclusions_and_raw_fallback() -> None:
    facade = load_semantic_world_operational_adapters()
    receipt = facade.record_context(
        included=[_context_item("keep", required=True)],
        omitted=[_context_item("skip", disposition="omitted", reason="out_of_scope")],
        raw_fallbacks=[
            _context_item(
                "raw",
                disposition="raw_fallback",
                required=True,
                reason="opaque_dimension",
                kind="raw_source",
            )
        ],
        unresolved_questions=("what is the current obligation?",),
        token_budget=4096,
    )
    assert isinstance(receipt, ProgramWorldContextReceipt)
    assert receipt.proposal_only is True
    assert receipt.embeddings_suppressed_required is False
    assert receipt.included[0].required is True
    assert receipt.raw_fallbacks[0].disposition == "raw_fallback"
    assert ProgramWorldContextReceipt.from_dict(receipt.to_dict()).receipt_cid == receipt.receipt_cid


def test_embeddings_cannot_omit_required_context() -> None:
    with pytest.raises(ProgramWorldReceiptError, match="required context material"):
        ProgramWorldContextItem(
            identity_cid=_cid("must-keep"),
            kind="raw_source",
            reason="embedding_neighbor",
            authority="ann",
            freshness="fresh",
            disposition="omitted",
            required=True,
        )


def test_execution_transition_compiler_is_proposal_only() -> None:
    compiler = ExecutionTransitionCompiler()
    query = _query()
    compiled = compiler.compile(query)
    assert isinstance(compiled, ExecutionTransitionCompilation)
    assert compiled.status == "compiled"
    assert compiled.query_cid == query.query_cid
    assert compiled.proposal_only is True
    assert compiled.admitted is False
    assert compiled.prediction_authoritative is False
    assert VALIDATION_AUTHORITY in compiled.operational_acceptance_authorities
    assert "compiler_does_not_admit" in compiled.limitations
    assert (
        ExecutionTransitionCompilation.from_dict(compiled.to_dict()).compilation_cid
        == compiled.compilation_cid
    )


def test_prediction_cannot_self_admit_compilation() -> None:
    compiler = ExecutionTransitionCompiler()
    query = _query()
    prediction = SimpleNamespace(
        prediction_cid=query.query_cid,
        proposal_only=True,
        admitted=True,
        SCHEMA="ipfs-datasets.software-contracts.program-transition-prediction@1",
    )
    # Fake prediction type name is not ProgramTransitionPrediction, but admitted=True
    # is still rejected after datasets authority check. Use a real datasets query as
    # the prediction stand-in with an adapter-owned admission authority.
    with pytest.raises(ProgramWorldAdmissionError):
        compiler.compile(
            query,
            prediction=query,
            admission_authority="ExecutionTransitionCompiler",
            admission_evidence_cid=_cid("self"),
        )
    assert prediction.admitted is True


def test_root_publisher_requests_without_changing_current_root() -> None:
    calls: list[str] = []

    class _Port:
        def compare_and_swap_root(self, *args: Any, **kwargs: Any) -> None:
            calls.append("cas")
            raise AssertionError("publisher must not CAS")

    publisher = OperationalWorldRootPublisher()
    root = _world_root()
    request = publisher.request_publication(
        root,
        expected_generation=3,
        expected_root_cid=_cid("expected-root"),
        durable_port=_Port(),
    )
    assert isinstance(request, OperationalWorldRootPublicationRequest)
    assert request.status == "requested"
    assert request.changes_current_root is False
    assert request.proposal_only is True
    assert request.semantic_world_root_cid == root.semantic_world_root_cid
    assert "cannot_change_current_root" in request.limitations
    assert "publisher_does_not_cas" in request.limitations
    assert calls == []
    assert MERGE_AUTHORITY in request.operational_acceptance_authorities
    assert EVENT_AUTHORITY in request.operational_acceptance_authorities
    assert (
        OperationalWorldRootPublicationRequest.from_dict(request.to_dict()).request_cid
        == request.request_cid
    )


def test_root_publisher_falls_back_when_kit_unavailable() -> None:
    publisher = OperationalWorldRootPublisher(
        kit=KitProgramWorldAdapter(loader=_unavailable_loader(KIT_SURFACE))
    )
    request = publisher.request_publication(_world_root(), expected_generation=0)
    assert request.status == "requested"
    assert request.fallback is True
    assert request.kit_verified is False
    assert "kit_verified_store_unavailable" in request.limitations
    assert request.changes_current_root is False


def test_capability_receipt_round_trip() -> None:
    receipt = ProgramWorldCapabilityReceipt.from_dict(
        available_capability(DATASETS_SURFACE, operations=("cite_semantic_object",)).to_dict()
    )
    assert receipt.available is True
    assert receipt.capability_cid.startswith("b")


def test_unknown_operational_acceptance_authority_is_rejected() -> None:
    with pytest.raises(ProgramWorldReceiptError, match="unknown operational acceptance"):
        ProgramWorldReuseDecision(
            verdict="abstain",
            state_cid=_cid("state"),
            goal_cid=_cid("goal"),
            policy_cid=_cid("policy"),
            environment_cid=_cid("env"),
            toolchain_cid=_cid("toolchain"),
            reason_code="no_exact_identity_match",
            operational_acceptance_authorities=(
                VALIDATION_AUTHORITY,
                MERGE_AUTHORITY,
                EVENT_AUTHORITY,
                "fake.authority",
            ),
        )
