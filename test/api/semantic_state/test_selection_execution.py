"""SCH-008 sealed selection projection and execution adapter tests."""

from __future__ import annotations

import ast
import inspect
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    TestSelectionRef,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.selection_execution import (
    ADAPTER_ID,
    FALLBACK_BOTH,
    FALLBACK_FULL_PROOFS,
    FALLBACK_FULL_PYTEST,
    FALLBACK_NONE,
    CommandBinding,
    CommandKind,
    FallbackWeakeningError,
    HarnessAssurancePolicy,
    MaterializedSelectionPlan,
    SELECTION_EXECUTION_INTERFACE,
    SelectionBindingError,
    SelectionCancelled,
    SelectionExecutionAdapter,
    SelectionExecutionError,
    TypedTimeout,
    assert_fallback_not_weakened,
    combine_fallbacks,
    materialize_selection_commands,
    selection_execution_descriptor,
    selection_ref_from_selection,
    verify_selection_binding,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    ValidationStage,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_state/selection_execution.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _binding(**overrides: object) -> CommandBinding:
    payload = {
        "tree_cid": _cid("tree"),
        "config_cid": _cid("config"),
        "dependency_lock_cid": _cid("lock"),
        "toolchain_cid": _cid("toolchain"),
        "policy_cid": _cid("policy"),
        "interface_cid": _cid("interface"),
    }
    payload.update(overrides)
    return CommandBinding.from_dict(payload)


def _reason_path(path_cid: str | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        path_cid=path_cid or _cid("reason-path-a"),
        seed_subject_id="sym:a",
        target_node_id="tests/test_mod.py::test_a",
        edge_ids=("e1",),
        link_cids=(_cid("link-a"),),
        relation_steps=("calls",),
    )


def _selection(
    *,
    selection_cid: str | None = None,
    previous: str | None = None,
    current: str | None = None,
    pytest_nodes: tuple[str, ...] = ("tests/test_mod.py::test_a",),
    proof_ids: tuple[str, ...] = (),
    fallback: str = FALLBACK_NONE,
    fallback_reasons: tuple[str, ...] = (),
    reason_paths: tuple[Any, ...] | None = None,
) -> types.SimpleNamespace:
    prev = previous if previous is not None else _cid("prev-root")
    curr = current or _cid("curr-root")
    paths = reason_paths if reason_paths is not None else (_reason_path(),)
    # selection_cid is opaque identity; tests pin a stable label digest.
    sel_cid = selection_cid or _cid(
        f"sel|{curr}|{','.join(sorted(pytest_nodes))}|{fallback}"
    )
    return types.SimpleNamespace(
        selection_cid=sel_cid,
        previous_root_cid=prev,
        current_root_cid=curr,
        selected_pytest_node_ids=pytest_nodes,
        selected_proof_ids=proof_ids,
        reason_paths=paths,
        covered_seed_obligation_ids=("obl:1",),
        unresolved_obligation_ids=(),
        known_test_universe_cid=_cid("universe"),
        known_test_universe_count=3,
        fallback=fallback,
        fallback_reasons=fallback_reasons,
        policy_cid=_cid("selection-policy"),
    )


def _ref_for(selection: Any) -> TestSelectionRef:
    return selection_ref_from_selection(selection)


# ---------------------------------------------------------------------------
# AST / authority guards
# ---------------------------------------------------------------------------


def test_module_does_not_implement_graph_reselection() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    forbidden = {
        "run_impact_selected",
        "select_tests_and_proofs",
        "traverse_edges",
        "guess_node_id",
        "collect_tests",
    }
    assert not (names & forbidden)
    # Mentions in docs/forbid lists are fine; calls are not.
    assert "run_impact_selected(" not in source
    assert "select_tests_and_proofs(" not in source


def test_descriptor_forbids_reselection() -> None:
    descriptor = selection_execution_descriptor()
    assert descriptor["interface"] == SELECTION_EXECUTION_INTERFACE
    assert descriptor["adapter_id"] == ADAPTER_ID
    forbids = set(descriptor["forbids"])
    assert "run_impact_selected" in forbids
    assert "graph_traversal" in forbids
    assert "weaken_producer_fallback" in forbids


# ---------------------------------------------------------------------------
# Binding and provenance
# ---------------------------------------------------------------------------


def test_verify_selection_binding_accepts_matching_ref() -> None:
    selection = _selection()
    ref = _ref_for(selection)
    verify_selection_binding(ref, selection)


def test_verify_selection_binding_rejects_root_mismatch() -> None:
    selection = _selection()
    ref = TestSelectionRef.from_dict(
        {
            "selection_cid": selection.selection_cid,
            "previous_semantic_state_root_cid": selection.previous_root_cid,
            "current_semantic_state_root_cid": _cid("other-root"),
        }
    )
    with pytest.raises(SelectionBindingError, match="current_semantic_state_root_cid"):
        verify_selection_binding(ref, selection)


def test_command_provenance_retains_reason_paths_and_roots() -> None:
    selection = _selection(
        fallback=FALLBACK_FULL_PYTEST,
        fallback_reasons=("native_or_opaque_reachability",),
    )
    plan = materialize_selection_commands(selection, binding=_binding())
    assert plan.reason_path_cids
    assert plan.producer_fallback == FALLBACK_FULL_PYTEST
    assert plan.effective_fallback == FALLBACK_FULL_PYTEST
    assert "native_or_opaque_reachability" in plan.fallback_reasons
    for command in plan.commands:
        prov = command.provenance
        assert prov.selection_cid == selection.selection_cid
        assert prov.current_semantic_state_root_cid == selection.current_root_cid
        assert prov.previous_semantic_state_root_cid == selection.previous_root_cid
        assert prov.reason_path_cids == plan.reason_path_cids
        assert prov.binding.tree_cid == plan.binding.tree_cid
        assert prov.binding.toolchain_cid == plan.binding.toolchain_cid


def test_commands_bind_exact_tree_config_toolchain() -> None:
    binding = _binding(
        tree_cid=_cid("exact-tree"),
        config_cid=_cid("exact-config"),
        toolchain_cid=_cid("exact-toolchain"),
    )
    plan = materialize_selection_commands(_selection(), binding=binding)
    assert plan.binding.tree_cid == binding.tree_cid
    assert plan.binding.config_cid == binding.config_cid
    assert plan.binding.toolchain_cid == binding.toolchain_cid
    assert plan.binding.dependency_lock_cid == binding.dependency_lock_cid
    for command in plan.commands:
        assert command.provenance.binding.to_dict() == binding.to_dict()


# ---------------------------------------------------------------------------
# Fallback: cannot weaken; may escalate
# ---------------------------------------------------------------------------


def test_combine_fallbacks_is_least_upper_bound() -> None:
    assert combine_fallbacks(FALLBACK_NONE, FALLBACK_NONE) == FALLBACK_NONE
    assert combine_fallbacks(FALLBACK_FULL_PYTEST, FALLBACK_NONE) == FALLBACK_FULL_PYTEST
    assert combine_fallbacks(FALLBACK_FULL_PROOFS, FALLBACK_FULL_PYTEST) == FALLBACK_BOTH
    assert combine_fallbacks(FALLBACK_BOTH, FALLBACK_NONE) == FALLBACK_BOTH


def test_assert_fallback_not_weakened() -> None:
    assert_fallback_not_weakened(
        producer=FALLBACK_FULL_PYTEST, effective=FALLBACK_FULL_PYTEST
    )
    assert_fallback_not_weakened(producer=FALLBACK_FULL_PYTEST, effective=FALLBACK_BOTH)
    with pytest.raises(FallbackWeakeningError):
        assert_fallback_not_weakened(
            producer=FALLBACK_FULL_PYTEST, effective=FALLBACK_NONE
        )
    with pytest.raises(FallbackWeakeningError):
        assert_fallback_not_weakened(
            producer=FALLBACK_BOTH, effective=FALLBACK_FULL_PYTEST
        )


def test_opaque_producer_fallback_cannot_be_weakened_by_assurance() -> None:
    selection = _selection(
        fallback=FALLBACK_FULL_PYTEST,
        fallback_reasons=("native_or_opaque_reachability", "unknown_test_universe"),
        pytest_nodes=(),
    )
    # Assurance that does not force full pytest must still retain producer force.
    policy = HarnessAssurancePolicy(force_full_pytest=False, force_full_proofs=False)
    plan = materialize_selection_commands(
        selection, binding=_binding(), assurance=policy
    )
    assert plan.effective_fallback == FALLBACK_FULL_PYTEST
    kinds = {command.kind for command in plan.commands}
    assert CommandKind.FULL_PYTEST.value in kinds
    assert CommandKind.PYTEST_NODE.value not in kinds


def test_assurance_may_only_escalate_producer_fallback() -> None:
    selection = _selection(fallback=FALLBACK_NONE, pytest_nodes=("t::a",))
    policy = HarnessAssurancePolicy(force_full_pytest=True, force_full_proofs=True)
    plan = materialize_selection_commands(
        selection, binding=_binding(), assurance=policy
    )
    assert plan.producer_fallback == FALLBACK_NONE
    assert plan.effective_fallback == FALLBACK_BOTH
    kinds = {command.kind for command in plan.commands}
    assert CommandKind.FULL_PYTEST.value in kinds
    assert CommandKind.FULL_PROOFS.value in kinds


def test_config_dependency_fallback_reasons_are_retained() -> None:
    selection = _selection(
        fallback=FALLBACK_BOTH,
        fallback_reasons=(
            "insufficient_graph_evidence",
            "dynamic_pytest_plugin",
        ),
        pytest_nodes=(),
        proof_ids=(),
    )
    plan = materialize_selection_commands(selection, binding=_binding())
    assert plan.fallback_reasons == (
        "dynamic_pytest_plugin",
        "insufficient_graph_evidence",
    )
    assert plan.effective_fallback == FALLBACK_BOTH


# ---------------------------------------------------------------------------
# Materialization: explicit selected IDs only
# ---------------------------------------------------------------------------


def test_materialize_selected_pytest_nodes_to_targeted_commands() -> None:
    nodes = ("tests/test_a.py::test_one", "tests/test_b.py::test_two")
    selection = _selection(pytest_nodes=nodes, proof_ids=("proof:1",))
    plan = materialize_selection_commands(selection, binding=_binding())
    pytest_cmds = [
        c for c in plan.commands if c.kind == CommandKind.PYTEST_NODE.value
    ]
    assert len(pytest_cmds) == 2
    for command, node in zip(pytest_cmds, sorted(nodes), strict=True):
        assert node in (command.shell_command or "")
        assert command.target_ids == (node,)
        assert command.validation_command is not None
        assert command.validation_command.stage is ValidationStage.TARGETED
        assert isinstance(command.timeout, TypedTimeout)
        assert command.timeout.seconds > 0
    proof_cmds = [c for c in plan.commands if c.kind == CommandKind.PROOF.value]
    assert len(proof_cmds) == 1
    assert proof_cmds[0].proof_id == "proof:1"


def test_materialize_does_not_invent_node_ids() -> None:
    selection = _selection(pytest_nodes=())
    plan = materialize_selection_commands(selection, binding=_binding())
    pytest_cmds = [
        c
        for c in plan.commands
        if c.kind in {CommandKind.PYTEST_NODE.value, CommandKind.FULL_PYTEST.value}
    ]
    assert pytest_cmds == []


def test_static_checks_are_cheap_stage() -> None:
    selection = _selection()
    policy = HarnessAssurancePolicy(
        require_static_checks=True,
        static_check_commands=("python3.12 -m compileall pkg",),
    )
    plan = materialize_selection_commands(
        selection, binding=_binding(), assurance=policy
    )
    static = [c for c in plan.commands if c.kind == CommandKind.STATIC_CHECK.value]
    assert len(static) == 1
    assert static[0].validation_command is not None
    assert static[0].validation_command.stage is ValidationStage.CHEAP


# ---------------------------------------------------------------------------
# Adapter execution: no reselection; typed cancel
# ---------------------------------------------------------------------------


class _FakeScheduler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_staged(self, commands, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"commands": list(commands), "kwargs": dict(kwargs)})
        # Contract: empty changed_files must be passed so impact reselection
        # cannot run a second affected set.
        assert kwargs.get("changed_files") == ()
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [{"command": c.command, "returncode": 0} for c in commands],
        }

    def run_impact_selected(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("run_impact_selected must never be called")


def test_adapter_uses_run_staged_never_impact_selected(tmp_path: Path) -> None:
    scheduler = _FakeScheduler()
    adapter = SelectionExecutionAdapter(validation_scheduler=scheduler)
    selection = _selection()
    report = adapter.execute(
        selection,
        binding=_binding(),
        workspace_path=tmp_path,
        proof_executor=None,
        prover_available=False,
    )
    assert scheduler.calls
    assert report["producer_fallback"] == FALLBACK_NONE
    assert report["selection_cid"] == selection.selection_cid
    assert "reason_path_cids" in report
    assert report["binding"]["tree_cid"] == _binding().tree_cid


def test_unavailable_prover_never_reported_as_passed() -> None:
    adapter = SelectionExecutionAdapter()
    selection = _selection(proof_ids=("proof:x",))
    plan = adapter.materialize(selection, binding=_binding())
    results = adapter.run_proofs(plan, prover_available=False)
    assert len(results) == 1
    assert results[0]["status"] == "unavailable"
    assert results[0]["passed"] is False


def test_typed_cancellation_raises(tmp_path: Path) -> None:
    adapter = SelectionExecutionAdapter(validation_scheduler=_FakeScheduler())
    token = CancellationToken("cancel-1")
    token.cancel(cancellation_id="cancel-1", reason="operator-stop")
    selection = _selection()
    with pytest.raises(SelectionCancelled) as excinfo:
        adapter.execute(
            selection,
            binding=_binding(),
            workspace_path=tmp_path,
            cancellation=token,
        )
    assert excinfo.value.cancellation_id == "cancel-1"
    assert excinfo.value.reason_code == "execution_cancelled"


def test_typed_timeout_record_is_positive() -> None:
    timeout = TypedTimeout(seconds=12.5, stage="pytest_node")
    assert timeout.to_dict() == {"seconds": 12.5, "stage": "pytest_node"}
    with pytest.raises(SelectionExecutionError):
        TypedTimeout(seconds=0, stage="x")
    with pytest.raises(SelectionExecutionError):
        TypedTimeout(seconds=-1, stage="x")


def test_plan_to_dict_is_closed_and_sortable() -> None:
    plan = materialize_selection_commands(_selection(), binding=_binding())
    assert isinstance(plan, MaterializedSelectionPlan)
    payload = plan.to_dict()
    assert payload["interface"] == SELECTION_EXECUTION_INTERFACE
    assert payload["selection_ref"]["selection_cid"] == plan.selection_ref.selection_cid
    assert isinstance(payload["commands"], list)


def test_cold_import_is_side_effect_free() -> None:
    # Import path already exercised; ensure module source documents cold import.
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "Cold import is side-effect free" in source
    # Public adapter constructor does not touch schedulers until execute.
    adapter = SelectionExecutionAdapter()
    assert adapter.validation_scheduler is None


def test_source_exports_predicted_symbols() -> None:
    import ipfs_accelerate_py.agent_supervisor.semantic_state.selection_execution as mod

    assert hasattr(mod, "SelectionExecutionAdapter")
    assert hasattr(mod, "materialize_selection_commands")
    assert callable(mod.materialize_selection_commands)
    # Signature retains binding and assurance knobs.
    sig = inspect.signature(mod.materialize_selection_commands)
    assert "binding" in sig.parameters
    assert "assurance" in sig.parameters
