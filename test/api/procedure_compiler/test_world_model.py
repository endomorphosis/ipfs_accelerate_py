from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ProcedureBoundsError,
    ProcedureContractError,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.world_model import (
    AbstractRepositoryState,
    ArtifactClass,
    RepositoryWorldModel,
    RepositoryWorldState,
    TransitionClass,
    WorldDimension,
    WorldModelError,
    WorldProjectionError,
    WorldProjectionRole,
    WorldProjectionStatus,
    WorldStateDelta,
    abstract_repository_state,
    extract_world_state_delta,
)


def _bindings(*, repository: str = "repo", tree: str = "tree") -> ArtifactBindings:
    return ArtifactBindings(
        repository_id=repository,
        repository_commit="commit-1",
        tree_id=tree,
        objective_id="PCPC-G000",
        task_id="PCPC-006",
        contract_revision="contract-1",
        policy_revision="policy-1",
        environment_id="environment-1",
    )


def _state(**changes: object) -> RepositoryWorldState:
    values = {
        "bindings": _bindings(),
        "world_snapshot_cid": "sha256:" + "1" * 64,
        "repository_reference": "sha256:" + "2" * 64,
        "repository_snapshot_id": "sca-repository-snapshot:sha256:" + "3" * 64,
        "analysis_head_tree_id": "git-tree-1",
        "analysis_index_tree_id": "git-index-1",
        "changed_files": ("src/a.py",),
        "changed_symbols": ("src.a:run",),
        "package_graph_id": "package-graph-1",
        "import_graph_id": "import-graph-1",
        "dependency_graph_id": "dependency-graph-1",
        "interface_graph_id": "interface-graph-1",
        "effect_graph_id": "effect-graph-1",
        "acceptance_state_id": "acceptance-1",
        "active_task_ids": ("PCPC-006",),
        "task_dependency_ids": ("PCPC-000",),
        "task_dependency_state_id": "task-dependencies-1",
        "proof_status_id": "proof-status-1",
        "test_status_id": "test-status-1",
        "capability_state_id": "capability-state-1",
        "provider_capacity_id": "provider-capacity-1",
        "worktree_ids": ("worktree-1",),
        "lease_ids": ("lease-1",),
        "merge_queue_id": "merge-queue-1",
        "cache_state_id": "cache-state-1",
        "artifact_pressure_id": "artifact-pressure-1",
        "token_budget_remaining": 10_000,
        "resource_budget_id": "resource-budget-1",
        "known_failure_signature_ids": ("failure-1",),
        "procedure_registry_revision": 4,
        "procedure_registry_id": "registry-4",
        "source_evidence_ids": ("receipt-1",),
    }
    values.update(changes)
    return RepositoryWorldState(**values)


def test_import_is_cold_with_respect_to_source_authority_modules() -> None:
    root = Path(__file__).resolve().parents[3]
    script = """
import json, sys
import ipfs_accelerate_py.agent_supervisor.procedure_compiler.world_model
world_module = (
  'ipfs_accelerate_py.agent_supervisor.semantic_state.'
  'world_snapshot_contracts'
)
repository_module = (
  'ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot'
)
print(json.dumps({
  'world': world_module in sys.modules,
  'repository': repository_module in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"world": False, "repository": False}


def test_world_state_covers_bounded_planning_dimensions_without_authority() -> None:
    state = _state()

    assert state.repository_id == "repo"
    assert state.tree_id == "tree"
    assert state.changed_files == ("src/a.py",)
    assert state.package_graph_id == "package-graph-1"
    assert state.import_graph_id == "import-graph-1"
    assert state.dependency_graph_id == "dependency-graph-1"
    assert state.interface_graph_id == "interface-graph-1"
    assert state.effect_graph_id == "effect-graph-1"
    assert state.acceptance_state_id == "acceptance-1"
    assert state.active_task_ids == ("PCPC-006",)
    assert state.task_dependency_ids == ("PCPC-000",)
    assert state.proof_status_id == "proof-status-1"
    assert state.test_status_id == "test-status-1"
    assert state.capability_state_id == "capability-state-1"
    assert state.provider_capacity_id == "provider-capacity-1"
    assert state.worktree_ids == ("worktree-1",)
    assert state.lease_ids == ("lease-1",)
    assert state.merge_queue_id == "merge-queue-1"
    assert state.cache_state_id == "cache-state-1"
    assert state.artifact_pressure_id == "artifact-pressure-1"
    assert state.token_budget_remaining == 10_000
    assert state.resource_budget_id == "resource-budget-1"
    assert state.known_failure_signature_ids == ("failure-1",)
    assert state.procedure_registry_revision == 4
    assert state.projection_role is WorldProjectionRole.PLANNING_PROJECTION_ONLY
    assert state.is_authoritative is False
    assert state.can_grant_authority is False


def test_world_state_round_trip_identity_and_unknown_field_rejection() -> None:
    state = _state(changed_files=("src/z.py", "src/a.py", "src/z.py"))
    assert state.changed_files == ("src/a.py", "src/z.py")
    assert RepositoryWorldState.from_dict(state.to_dict()) == state
    assert RepositoryWorldState.from_dict(state.to_record()) == state

    forged = {**state.to_dict(), "unknown_normative_field": True}
    with pytest.raises(ProcedureContractError):
        RepositoryWorldState.from_dict(forged)

    wrong_identity = {**state.to_dict(), "content_id": "forged"}
    with pytest.raises(ProcedureContractError):
        RepositoryWorldState.from_dict(wrong_identity)


@pytest.mark.parametrize("path", ("../escape.py", "/tmp/escape.py", "src//a.py", "."))
def test_world_state_rejects_path_escape_and_noncanonical_paths(path: str) -> None:
    with pytest.raises(WorldModelError):
        _state(changed_files=(path,))


def test_world_state_rejects_unbounded_references_and_non_integer_budget() -> None:
    with pytest.raises(ProcedureBoundsError):
        _state(active_task_ids=tuple(f"task-{index}" for index in range(257)))
    with pytest.raises(ProcedureContractError):
        _state(token_budget_remaining=1.5)


def test_abstract_state_is_deterministic_and_keeps_exact_bindings() -> None:
    state = _state(changed_files=("docs/readme.md", "test/api/test_x.py", "src/x.py"))
    abstract = abstract_repository_state(
        state,
        known_effect_classes=("validation", "repository_write"),
        known_failure_family_ids=("known-flake",),
    )

    assert abstract.bindings == state.bindings
    assert abstract.source_world_state_id == state.content_id
    assert abstract.changed_file_classes == (
        ArtifactClass.DOCUMENTATION,
        ArtifactClass.PYTHON_SOURCE,
        ArtifactClass.TEST,
    )
    assert abstract.changed_file_count == 3
    assert abstract.is_authoritative is False
    assert AbstractRepositoryState.from_dict(abstract.to_dict()) == abstract


def test_delta_extraction_is_deterministic_and_complete_for_bounded_sets() -> None:
    before = _state()
    after = _state(
        changed_files=("src/b.py", "src/a.py"),
        changed_symbols=("src.a:run", "src.b:run"),
        active_task_ids=("PCPC-007",),
        task_dependency_ids=("PCPC-006",),
        worktree_ids=("worktree-1", "worktree-2"),
        lease_ids=(),
        test_status_id="test-status-2",
        token_budget_remaining=9_000,
        known_failure_signature_ids=("failure-2",),
    )

    first = extract_world_state_delta(before, after, evidence_ids=("delta-receipt",))
    second = RepositoryWorldModel.delta(before, after, evidence_ids=("delta-receipt",))

    assert first == second
    assert first.content_id == second.content_id
    assert first.transition_class is TransitionClass.LEASE_EXPIRY
    assert first.added_changed_files == ("src/b.py",)
    assert first.added_changed_symbols == ("src.b:run",)
    assert first.added_active_task_ids == ("PCPC-007",)
    assert first.removed_active_task_ids == ("PCPC-006",)
    assert first.removed_lease_ids == ("lease-1",)
    assert first.token_budget_delta == -1_000
    assert {item.dimension for item in first.reference_changes} == {WorldDimension.TEST_STATUS}
    assert first.is_authoritative is False
    assert first.has_changes is True
    assert WorldStateDelta.from_dict(first.to_dict()) == first


def test_delta_cannot_cross_repository_objective_or_task() -> None:
    before = _state()
    other_repository = _state(bindings=_bindings(repository="other"))
    with pytest.raises(WorldModelError):
        extract_world_state_delta(before, other_repository)

    other_task_binding = replace(before.bindings, task_id="PCPC-007")
    with pytest.raises(WorldModelError):
        extract_world_state_delta(before, _state(bindings=other_task_binding))


def test_projection_reuses_world_and_repository_snapshot_identities() -> None:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_contracts import (
        example_current_snapshot,
    )

    snapshot = example_current_snapshot(digest_seed="pcpc-world")
    repository_snapshot = {
        "snapshot_id": "sca-repository-snapshot:sha256:" + "4" * 64,
        "head_commit_id": "a" * 40,
        "head_tree_id": "b" * 40,
        "index_tree_id": "c" * 40,
        "dispositions": (
            {"path": "src/clean.py", "git_status": "clean"},
            {"path": "src/changed.py", "git_status": "modified"},
        ),
    }
    state = RepositoryWorldModel.project(
        snapshot,
        repository_snapshot,
        objective_id="PCPC-G000",
        task_id="PCPC-006",
        changed_symbols=("src.changed:run",),
        package_graph_id="package-graph",
        import_graph_id="import-graph",
        dependency_graph_id="dependency-graph",
        interface_graph_id="interface-graph",
        effect_graph_id="effect-graph",
        proof_status_id="proof-status",
        test_status_id="test-status",
        provider_capacity_id="provider-capacity",
        cache_state_id="cache-state",
        artifact_pressure_id="artifact-pressure",
        procedure_registry_id="procedure-registry",
    )

    assert state.world_snapshot_cid == snapshot["snapshot_cid"]
    assert state.tree_id == snapshot["components"]["repository_tree"]["cid"]
    assert state.repository_snapshot_id == repository_snapshot["snapshot_id"]
    assert state.repository_commit == repository_snapshot["head_commit_id"]
    assert state.analysis_head_tree_id == repository_snapshot["head_tree_id"]
    assert state.changed_files == ("src/changed.py",)
    assert state.projection_status is WorldProjectionStatus.CURRENT
    assert state.is_authoritative is False


def test_projection_rejects_receipt_shaped_or_incomplete_sources() -> None:
    with pytest.raises(WorldProjectionError):
        RepositoryWorldModel.project(
            {"snapshot_cid": "sha256:" + "0" * 64},
            {},
            objective_id="PCPC-G000",
            task_id="PCPC-006",
        )
