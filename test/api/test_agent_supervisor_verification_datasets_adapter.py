"""Tests for the lazy fail-closed datasets semantic input adapter (IVP-002)."""

from __future__ import annotations

import importlib
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.verification import datasets_adapter as da
from ipfs_accelerate_py.agent_supervisor.verification.datasets_adapter import (
    BOUNDED_TOOL_RUNNER_LEAF_MODULE,
    BOUNDED_TOOL_RUNNER_LEAF_SYMBOL,
    CODE_EVIDENCE_LEAF_MODULE,
    CODE_EVIDENCE_LEAF_SYMBOLS,
    CODE_IMPACT_INDEX_SCHEMA,
    CODE_IMPACT_RESULT_SCHEMA,
    DATASETS_CANONICAL_TYPES_GAP,
    DATASETS_CONTEXT_PACK_SCHEMA,
    DATASETS_INVALIDATION_PLAN_SCHEMA,
    DATASETS_REPOSITORY_STATE_SCHEMA,
    DATASETS_SEMANTIC_CAPSULE_SCHEMA,
    DATASETS_VERIFICATION_INPUT_ADAPTER_INTERFACE,
    DatasetsVerificationInputAdapter,
    InputKind,
    ObservationKind,
    create_datasets_verification_input_adapter,
    probe_leaf_symbol,
    probe_top_level_namespace_alone,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return content_identity({"artifact": label, "schema": "fixture-artifact@1"})


SEMANTIC = _cid("semantic-root")
ENVIRONMENT = _cid("environment-root")
LOCK = _cid("dependency-lock-root")
RECEIPT_TREE = _cid("receipt-repository-tree")
OPAQUE_TREE = "datasets-tree:fixture-opaque-selector-001"


def _repository_state_mapping(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": DATASETS_REPOSITORY_STATE_SCHEMA,
        "repository_tree_id": OPAQUE_TREE,
        "semantic_state_root_cid": SEMANTIC,
        "environment_root_cid": ENVIRONMENT,
        "dependency_lock_root_cid": LOCK,
    }
    payload.update(overrides)
    return payload


def _invalidation_plan_mapping(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": DATASETS_INVALIDATION_PLAN_SCHEMA,
        "repository_tree_id": OPAQUE_TREE,
        "semantic_state_root_cid": SEMANTIC,
        "changed_symbols": ["pkg.mod.fn", "pkg.mod.Helper"],
        "changed_paths": ["pkg/mod.py", "pkg/other.py"],
        "edges": [
            {
                "source": "pkg.mod.fn",
                "target": "test/api/test_mod.py::test_fn",
                "kind": "tested_by",
            },
            {
                "source": "pkg.mod.Helper",
                "target": "pkg.mod.fn",
                "kind": "depends_on",
            },
        ],
        "spans": [
            {"path": "pkg/mod.py", "start_line": 10, "end_line": 20, "symbol": "fn"},
        ],
        "contracts": [{"name": "api-surface", "version": "1"}],
        "uncertainty": {"frontier": "exact"},
        "uncovered_symbols": [],
        "uncovered_paths": [],
        "truncated": False,
        "uncovered_impact": False,
    }
    payload.update(overrides)
    return payload


def _semantic_capsule_mapping(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": DATASETS_SEMANTIC_CAPSULE_SCHEMA,
        "semantic_state_root_cid": SEMANTIC,
        "repository_tree_id": OPAQUE_TREE,
        "edges": [
            {
                "source": "pkg.mod.fn",
                "target": "test/api/test_mod.py::test_fn",
                "kind": "tested_by",
            }
        ],
        "spans": [
            {"path": "pkg/mod.py", "start_line": 1, "end_line": 5, "symbol": "fn"},
        ],
        "contracts": [{"contract_id": "c1"}],
        "fixture_references": ["fixture:semantic-capsule-a"],
        "truncated": False,
    }
    payload.update(overrides)
    return payload


def _context_pack_mapping(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema": DATASETS_CONTEXT_PACK_SCHEMA,
        "repository_tree_id": OPAQUE_TREE,
        "semantic_state_root_cid": SEMANTIC,
        "environment_root_cid": ENVIRONMENT,
        "dependency_lock_root_cid": LOCK,
        "token_estimate": 1200,
        "fixture_task_references": ["fixture-task:ivp-semantic-1"],
        "contracts": [{"name": "context-contract"}],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def adapter() -> DatasetsVerificationInputAdapter:
    return create_datasets_verification_input_adapter()


# ---------------------------------------------------------------------------
# Dependency gap + impact schema record (precondition)
# ---------------------------------------------------------------------------


def test_dependency_gap_and_code_evidence_impact_schema_are_recorded(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    gap = adapter.canonical_types_gap()
    assert gap == {
        "RepositoryState": "absent",
        "InvalidationPlan": "absent",
        "SemanticCapsule": "absent",
        "ContextPack": "absent",
    }
    assert gap == dict(DATASETS_CANONICAL_TYPES_GAP)

    schemas = adapter.code_evidence_impact_schemas()
    assert schemas["code_impact_index"] == CODE_IMPACT_INDEX_SCHEMA
    assert schemas["code_impact_result"] == CODE_IMPACT_RESULT_SCHEMA
    assert schemas["code_impact_index"].endswith("@1")
    assert "code-impact-index" in schemas["code_impact_index"]

    capability = adapter.capability_declaration()
    assert capability["interface"] == DATASETS_VERIFICATION_INPUT_ADAPTER_INTERFACE
    assert capability["lazy"] is True
    assert capability["authoritative"] is False
    assert capability["completion_authority"] is False
    assert capability["invokes_to_dict"] is False
    assert capability["arbitrary_attribute_traversal"] is False
    assert capability["network_side_effects"] is False
    assert capability["install_side_effects"] is False
    assert capability["repository_tree_id_is_opaque"] is True
    assert capability["repository_tree_id_not_receipt_tree_cid"] is True
    assert capability["top_level_namespace_insufficient"] is True


# ---------------------------------------------------------------------------
# Cold import without ipfs_datasets
# ---------------------------------------------------------------------------


def test_module_import_does_not_load_ipfs_datasets() -> None:
    """Importing the adapter module must not pull ipfs_datasets leaf modules."""

    prefixes = (
        "ipfs_datasets_py.knowledge_graphs",
        "ipfs_datasets_py.logic.backends",
    )
    # Capability declaration / construction must stay cold.
    for name in list(sys.modules):
        if name.startswith(prefixes):
            # Do not delete if other tests need them; just assert construction
            # path does not *require* them.
            pass

    calls: list[str] = []

    def explosive_importer(name: str) -> Any:
        calls.append(name)
        raise AssertionError(f"cold path imported {name}")

    cold = DatasetsVerificationInputAdapter(importer=explosive_importer)
    decl = cold.capability_declaration()
    gap = cold.canonical_types_gap()

    assert calls == []
    assert decl["lazy"] is True
    assert gap["RepositoryState"] == "absent"

    # Mapping normalization must also stay cold.
    result = cold.normalize_repository_state(_repository_state_mapping())
    assert result.ok is True
    assert calls == []


def test_accelerator_cold_import_subprocess_without_datasets() -> None:
    """Fresh process imports the adapter with ipfs_datasets blocked."""

    repo = Path(__file__).resolve().parents[2]
    probe = r"""
import sys
import builtins

real_import = builtins.__import__

def blocked(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py."):
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked

# Ensure datasets is not already present.
for key in list(sys.modules):
    if key == "ipfs_datasets_py" or key.startswith("ipfs_datasets_py."):
        del sys.modules[key]

from ipfs_accelerate_py.agent_supervisor.verification.datasets_adapter import (
    DatasetsVerificationInputAdapter,
    create_datasets_verification_input_adapter,
)
adapter = create_datasets_verification_input_adapter()
assert adapter.capability_declaration()["lazy"] is True
assert "ipfs_datasets_py" not in sys.modules
print("cold-ok")
"""
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(repo),
    }
    completed = subprocess.run(
        [sys.executable, "-P", "-c", probe],
        cwd=str(repo),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "cold-ok" in completed.stdout


# ---------------------------------------------------------------------------
# Strict canonical mapping normalization (deterministic)
# ---------------------------------------------------------------------------


def test_strict_canonical_mappings_normalize_deterministically(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    first = adapter.normalize_repository_state(_repository_state_mapping())
    second = adapter.normalize_repository_state(_repository_state_mapping())
    assert first.ok and second.ok
    assert first.view is not None and second.view is not None
    assert first.view.to_dict() == second.view.to_dict()
    assert first.view.repository_tree_id == OPAQUE_TREE
    assert first.view.semantic_state_root_cid == SEMANTIC
    assert first.view.authoritative is False

    plan_a = adapter.normalize_invalidation_plan(
        _invalidation_plan_mapping(
            edges=[
                {"source": "b", "target": "t2", "kind": "tested_by"},
                {"source": "a", "target": "t1", "kind": "tested_by"},
            ]
        )
    )
    plan_b = adapter.normalize_invalidation_plan(
        _invalidation_plan_mapping(
            edges=[
                {"source": "a", "target": "t1", "kind": "tested_by"},
                {"source": "b", "target": "t2", "kind": "tested_by"},
            ]
        )
    )
    assert plan_a.ok and plan_b.ok
    assert plan_a.view is not None and plan_b.view is not None
    assert [e.to_dict() for e in plan_a.view.edges] == [
        e.to_dict() for e in plan_b.view.edges
    ]
    assert plan_a.view.changed_symbols == ("pkg.mod.Helper", "pkg.mod.fn")
    assert plan_a.view.changed_paths == ("pkg/mod.py", "pkg/other.py")

    capsule = adapter.normalize_semantic_capsule(_semantic_capsule_mapping())
    assert capsule.ok and capsule.view is not None
    assert capsule.view.fixture_references == ("fixture:semantic-capsule-a",)
    assert capsule.view.semantic_state_root_cid == SEMANTIC

    context = adapter.normalize_context_pack(_context_pack_mapping())
    assert context.ok and context.view is not None
    assert context.view.token_estimate == 1200
    assert context.view.fixture_task_references == ("fixture-task:ivp-semantic-1",)


def test_unknown_schema_and_malformed_cid_are_typed_observations(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    unknown = adapter.normalize_repository_state(
        _repository_state_mapping(schema="totally-unknown-schema@1")
    )
    assert unknown.ok is False
    assert unknown.observation.kind is ObservationKind.UNKNOWN_SCHEMA
    assert unknown.observation.authoritative is False
    assert unknown.observation.completion_authority is False

    malformed = adapter.normalize_repository_state(
        _repository_state_mapping(semantic_state_root_cid="not-a-cid")
    )
    assert malformed.ok is False
    assert malformed.observation.kind is ObservationKind.MALFORMED
    assert "cid" in malformed.observation.reason_code or "CID" in malformed.observation.message or "cid" in malformed.observation.message.lower()

    missing = adapter.normalize_repository_state(
        {
            "schema": DATASETS_REPOSITORY_STATE_SCHEMA,
            "repository_tree_id": OPAQUE_TREE,
            # semantic root missing
        }
    )
    assert missing.ok is False
    assert missing.observation.kind is ObservationKind.MISSING_IDENTITY


# ---------------------------------------------------------------------------
# Registered upstream types (authority unchanged)
# ---------------------------------------------------------------------------


def test_registered_upstream_types_normalize_without_changing_authority(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    class UpstreamRepositoryState:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def to_dict(self) -> dict[str, Any]:
            raise AssertionError("adapter must not call to_dict")

    payload = _repository_state_mapping()

    def convert(instance: UpstreamRepositoryState) -> dict[str, Any]:
        return dict(instance._payload)

    adapter.register_upstream_type(
        UpstreamRepositoryState,
        input_kind=InputKind.REPOSITORY_STATE,
        converter=convert,
        module_name="ipfs_datasets_py.future.repository_state",
        symbol_name="RepositoryState",
    )
    registered = adapter.registered_upstream_types()
    assert registered[0]["symbol_name"] == "RepositoryState"

    result = adapter.normalize_repository_state(UpstreamRepositoryState(payload))
    assert result.ok is True
    assert result.view is not None
    assert result.view.authoritative is False
    assert result.observation.authoritative is False
    assert result.view.repository_tree_id == OPAQUE_TREE


def test_unregistered_object_is_unsupported_without_to_dict(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    class Hostile:
        def to_dict(self) -> dict[str, Any]:
            raise AssertionError("to_dict must not run")

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"attribute traversal of {name!r} is forbidden")

    result = adapter.normalize_repository_state(Hostile())
    assert result.ok is False
    assert result.observation.kind is ObservationKind.UNSUPPORTED
    assert result.observation.reason_code == "unregistered_input_type"


# ---------------------------------------------------------------------------
# Leaf-symbol probes vs top-level namespace
# ---------------------------------------------------------------------------


def test_exact_leaf_symbol_probes_and_top_level_namespace_insufficient(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    top = adapter.probe_top_level_namespace()
    assert top.available is False
    assert top.reason_code == "top_level_namespace_insufficient"
    assert top.top_level_namespace_insufficient is True
    assert top.authoritative is False

    # Injected importer: leaf available, namespace alone still insufficient.
    code_evidence = ModuleType(CODE_EVIDENCE_LEAF_MODULE)
    code_evidence.CodeEvidenceCorpusAdapter = object
    code_evidence.impact_from_index = lambda *a, **k: {}
    code_evidence.normalize_impact_index = lambda *a, **k: {}

    process = ModuleType(BOUNDED_TOOL_RUNNER_LEAF_MODULE)
    process.BoundedToolRunner = type("BoundedToolRunner", (), {})

    namespace = ModuleType("ipfs_datasets_py")

    def importer(name: str) -> Any:
        if name == CODE_EVIDENCE_LEAF_MODULE:
            return code_evidence
        if name == BOUNDED_TOOL_RUNNER_LEAF_MODULE:
            return process
        if name == "ipfs_datasets_py":
            return namespace
        raise ModuleNotFoundError(name)

    probing = DatasetsVerificationInputAdapter(importer=importer)
    evidence = probing.probe_code_evidence()
    for symbol in CODE_EVIDENCE_LEAF_SYMBOLS:
        assert evidence[symbol].available is True, symbol
        assert evidence[symbol].module == CODE_EVIDENCE_LEAF_MODULE
        assert evidence[symbol].symbol == symbol

    runner = probing.probe_bounded_tool_runner()
    assert runner.available is True
    assert runner.module == BOUNDED_TOOL_RUNNER_LEAF_MODULE
    assert runner.symbol == BOUNDED_TOOL_RUNNER_LEAF_SYMBOL

    alone = probing.probe_top_level_namespace()
    assert alone.available is False
    assert alone.reason_code == "top_level_namespace_insufficient"


def test_absent_leaf_modules_produce_typed_observations() -> None:
    def missing_importer(name: str) -> Any:
        raise ModuleNotFoundError(name)

    adapter = DatasetsVerificationInputAdapter(importer=missing_importer)
    evidence = adapter.probe_code_evidence()
    for symbol, capability in evidence.items():
        assert capability.available is False
        assert capability.reason_code == "leaf_module_absent"
        assert capability.authoritative is False

    runner = adapter.probe_bounded_tool_runner()
    assert runner.available is False
    assert runner.reason_code == "leaf_module_absent"

    top = probe_top_level_namespace_alone(importer=missing_importer)
    assert top.available is False

    absent_symbol = ModuleType("mod")
    cap = probe_leaf_symbol(
        "mod",
        "Missing",
        importer=lambda name: absent_symbol if name == "mod" else (_ for _ in ()).throw(
            ModuleNotFoundError(name)
        ),
    )
    assert cap.available is False
    assert cap.reason_code == "leaf_symbol_absent"


def test_live_leaf_probes_when_pythonpath_includes_datasets() -> None:
    """Under validation PYTHONPATH, exact leaves should resolve."""

    try:
        importlib.import_module(CODE_EVIDENCE_LEAF_MODULE)
        importlib.import_module(BOUNDED_TOOL_RUNNER_LEAF_MODULE)
    except ModuleNotFoundError:
        pytest.skip("ipfs_datasets_py leaf modules not on PYTHONPATH")

    adapter = DatasetsVerificationInputAdapter()
    evidence = adapter.probe_code_evidence()
    for symbol in CODE_EVIDENCE_LEAF_SYMBOLS:
        assert evidence[symbol].available is True, symbol
    assert adapter.probe_bounded_tool_runner().available is True
    # Namespace alone remains insufficient by policy.
    assert adapter.probe_top_level_namespace().available is False


# ---------------------------------------------------------------------------
# Opaque repository_tree_id vs receipt repository_tree_cid
# ---------------------------------------------------------------------------


def test_opaque_repository_tree_id_stays_separate_from_receipt_tree_cid(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    # Even if a caller aliases the fields, the view refuses the collision.
    result = adapter.normalize_repository_state(
        _repository_state_mapping(repository_tree_cid=OPAQUE_TREE)
    )
    assert result.ok and result.view is not None
    assert result.view.repository_tree_id == OPAQUE_TREE
    assert result.view.repository_tree_cid == ""
    assert result.view.to_dict()["opaque_tree_id_is_not_receipt_tree_cid"] is True

    with_receipt = adapter.normalize_repository_state(
        _repository_state_mapping(repository_tree_cid=RECEIPT_TREE)
    )
    assert with_receipt.ok and with_receipt.view is not None
    assert with_receipt.view.repository_tree_id == OPAQUE_TREE
    assert with_receipt.view.repository_tree_cid == RECEIPT_TREE
    assert with_receipt.view.repository_tree_id != with_receipt.view.repository_tree_cid

    plan = adapter.normalize_invalidation_plan(_invalidation_plan_mapping())
    context = adapter.normalize_context_pack(_context_pack_mapping())
    capsule = adapter.normalize_semantic_capsule(_semantic_capsule_mapping())
    assert plan.view is not None and context.view is not None and capsule.view is not None

    mismatches = adapter.cross_check_identity_roots(
        repository_state=with_receipt.view,
        invalidation_plan=plan.view,
        context_pack=context.view,
        semantic_capsule=capsule.view,
    )
    assert mismatches == ()

    # Disagreeing opaque tree ids fail closed.
    bad_plan = adapter.normalize_invalidation_plan(
        _invalidation_plan_mapping(repository_tree_id="other-opaque-tree")
    )
    assert bad_plan.view is not None
    cross = adapter.cross_check_identity_roots(
        repository_state=with_receipt.view,
        invalidation_plan=bad_plan.view,
        context_pack=context.view,
    )
    assert any(o.reason_code == "repository_tree_id_mismatch" for o in cross)
    assert all(o.authoritative is False for o in cross)


# ---------------------------------------------------------------------------
# Validation IDs → exact pytest nodes (or broader selection)
# ---------------------------------------------------------------------------


def test_validation_ids_require_exact_pytest_node_mapping_or_broader(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    exact = adapter.map_validation_ids_to_pytest_nodes(
        ["val-a", "val-b"],
        {
            "val-a": "test/api/test_mod.py::test_fn",
            "val-b": "test/api/test_mod.py::TestCls::test_method",
        },
    )
    assert exact.requires_broader_selection is False
    assert exact.mapped_pytest_node_ids == (
        "test/api/test_mod.py::TestCls::test_method",
        "test/api/test_mod.py::test_fn",
    )
    assert exact.unmapped_validation_ids == ()
    assert exact.authoritative is False

    missing = adapter.map_validation_ids_to_pytest_nodes(
        ["val-a", "val-missing"],
        {"val-a": "test/api/test_mod.py::test_fn"},
    )
    assert missing.requires_broader_selection is True
    assert missing.unmapped_validation_ids == ("val-missing",)
    assert any(
        o.kind is ObservationKind.BROADER_SELECTION_REQUIRED
        for o in missing.observations
    )

    bad_shape = adapter.map_validation_ids_to_pytest_nodes(
        ["val-x"],
        {"val-x": "not-a-pytest-node"},
    )
    assert bad_shape.requires_broader_selection is True
    assert bad_shape.unmapped_validation_ids == ("val-x",)

    empty = adapter.map_validation_ids_to_pytest_nodes([], {})
    assert empty.requires_broader_selection is False


# ---------------------------------------------------------------------------
# Opaque / uncovered / truncated edges
# ---------------------------------------------------------------------------


def test_opaque_uncovered_truncated_edges_force_broader_selection(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    opaque = adapter.normalize_invalidation_plan(
        _invalidation_plan_mapping(
            edges=[
                {
                    "source": "pkg.mod.fn",
                    "target": "mystery",
                    "kind": "opaque",
                }
            ]
        )
    )
    assert opaque.ok and opaque.view is not None
    assert opaque.view.requires_broader_selection is True
    assert opaque.requires_broader_selection is True
    assert any(e.disposition.value == "opaque" for e in opaque.view.edges)
    assert any(o.kind is ObservationKind.OPAQUE for o in opaque.view.observations)

    uncovered = adapter.normalize_invalidation_plan(
        _invalidation_plan_mapping(
            uncovered_symbols=["pkg.missing.symbol"],
            uncovered_impact=True,
            edges=[],
        )
    )
    assert uncovered.view is not None
    assert uncovered.view.requires_broader_selection is True
    assert any(o.kind is ObservationKind.UNCOVERED for o in uncovered.view.observations)

    truncated = adapter.normalize_semantic_capsule(
        _semantic_capsule_mapping(
            truncated=True,
            edges=[
                {
                    "source": "a",
                    "target": "b",
                    "kind": "depends_on",
                    "truncated": True,
                }
            ],
        )
    )
    assert truncated.view is not None
    assert truncated.view.requires_broader_selection is True
    assert any(o.kind is ObservationKind.TRUNCATED for o in truncated.view.observations)


def test_impact_index_normalization_keeps_tree_id_opaque(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    payload = {
        "schema": CODE_IMPACT_INDEX_SCHEMA,
        "repository_tree_id": OPAQUE_TREE,
        "semantic_state_root_cid": SEMANTIC,
        "symbol_paths": {"pkg.mod.fn": "pkg/mod.py"},
        "validation_targets": {
            "val-1": ["pkg.mod.fn"],
        },
    }
    result = adapter.normalize_impact_index(payload)
    assert result.ok and result.view is not None
    assert result.view.repository_tree_id == OPAQUE_TREE
    assert result.view.repository_tree_cid == ""
    assert any(e.kind == "tested_by" for e in result.view.edges)

    unknown = adapter.normalize_impact_index(
        {"schema": "other-schema@1", "repository_tree_id": OPAQUE_TREE}
    )
    assert unknown.ok is False
    assert unknown.observation.kind is ObservationKind.UNKNOWN_SCHEMA


# ---------------------------------------------------------------------------
# No network / install side effects
# ---------------------------------------------------------------------------


def test_no_network_or_install_side_effects(
    adapter: DatasetsVerificationInputAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_socket(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("network side effect is forbidden")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    monkeypatch.setattr(socket, "create_connection", forbid_socket)

    def forbid_run(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("install/subprocess side effect is forbidden")

    monkeypatch.setattr(subprocess, "run", forbid_run)
    monkeypatch.setattr(subprocess, "Popen", forbid_run)

    # Mapping path
    assert adapter.normalize_repository_state(_repository_state_mapping()).ok
    assert adapter.normalize_invalidation_plan(_invalidation_plan_mapping()).ok
    assert adapter.normalize_semantic_capsule(_semantic_capsule_mapping()).ok
    assert adapter.normalize_context_pack(_context_pack_mapping()).ok
    adapter.map_validation_ids_to_pytest_nodes(
        ["v1"], {"v1": "test/api/test_x.py::test_y"}
    )
    adapter.capability_declaration()

    # Leaf probe with local fake modules only.
    mod = ModuleType(CODE_EVIDENCE_LEAF_MODULE)
    mod.CodeEvidenceCorpusAdapter = object
    mod.impact_from_index = object
    mod.normalize_impact_index = object

    def importer(name: str) -> Any:
        if name == CODE_EVIDENCE_LEAF_MODULE:
            return mod
        raise ModuleNotFoundError(name)

    local = DatasetsVerificationInputAdapter(importer=importer)
    assert local.probe_code_evidence()["CodeEvidenceCorpusAdapter"].available is True


# ---------------------------------------------------------------------------
# Cross-check identity roots + non-authoritative envelope
# ---------------------------------------------------------------------------


def test_all_outputs_are_non_authoritative(
    adapter: DatasetsVerificationInputAdapter,
) -> None:
    results = [
        adapter.normalize_repository_state(_repository_state_mapping()),
        adapter.normalize_invalidation_plan(_invalidation_plan_mapping()),
        adapter.normalize_semantic_capsule(_semantic_capsule_mapping()),
        adapter.normalize_context_pack(_context_pack_mapping()),
    ]
    for result in results:
        assert result.observation.authoritative is False
        assert result.observation.completion_authority is False
        assert result.to_dict()["authoritative"] is False
        if result.view is not None:
            assert result.view.authoritative is False
            assert result.view.to_dict()["authoritative"] is False


def test_protocols_are_typing_only() -> None:
    # Runtime checkable but empty — not used for acceptance.
    assert issubclass(type("X", (), {}), object)
    assert da.RepositoryStateProtocol is not None
    assert da.InvalidationPlanProtocol is not None
    assert da.SemanticCapsuleProtocol is not None
    assert da.ContextPackProtocol is not None


def test_factory_and_interface_constants() -> None:
    adapter = create_datasets_verification_input_adapter()
    assert adapter.interface == DATASETS_VERIFICATION_INPUT_ADAPTER_INTERFACE
    assert adapter.version == 1
    assert "DatasetsVerificationInputAdapter" in da.__all__
