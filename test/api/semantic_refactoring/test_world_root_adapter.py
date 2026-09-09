"""Independent contract tests for SPAR-039 SemanticRefactorWorldRootAdapter."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.world_root_adapter import (
    ACCELERATOR_TASK_OWNER,
    ADAPTER_CAN_AUTHORIZE_COMPLETION,
    ADAPTER_CAN_AUTHORIZE_TRANSITION,
    ADAPTER_CAN_CREATE_AUTHORITY,
    ADAPTER_IS_NOMINATION_ONLY,
    ADAPTER_OWNS_CAS,
    ADAPTER_OWNS_VFS,
    ADAPTER_WRITES_REPOSITORY,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    AdapterStatus,
    CAS_AUTHORITY_OWNER,
    CRASH_RECOVERY_WITHOUT_OVERWRITE,
    CROSS_REPOSITORY_OWNERSHIP_INTERFACE,
    CrossRepositoryOwnership,
    DECLARED_ADAPTER_STATUSES,
    DECLARED_PERSIST_KINDS,
    DECLARED_REPOSITORY_OWNERS,
    DECLARED_TERMINAL_KINDS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXISTING_ADAPTER_AUTHORITIES,
    FORBIDDEN_WORLD_ROOT_NAMES,
    GENERATION_CAS_BINDING_INTERFACE,
    GENERATION_CAS_REQUIRED,
    GITLINK_BINDING_INTERFACE,
    GOAL_ID,
    GenerationCasBinding,
    GitlinkBinding,
    IDENTITY_EXCLUDED_FIELDS,
    KIT_OWNS_CAS,
    KIT_OWNS_OUTBOX,
    KIT_OWNS_VFS,
    KIT_OWNS_WAL_RECOVERY,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    NEGATIVE_EVIDENCE_RETAINED,
    NESTED_WRITES_REQUIRE_ACCELERATOR_TASK_OWNER,
    NESTED_WRITES_REQUIRE_EXPLICIT_GITLINK,
    NETWORK_DENIED,
    NETWORK_DENY,
    PERSIST_KINDS,
    PREDECESSOR_TASK_IDS,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    ROOT_CONFLICT_OVERWRITE_FORBIDDEN,
    SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_INTERFACE,
    SEMANTIC_WORLD_ROOT_INTERFACE,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    STALE_WRITER_REJECTED,
    STALE_WRITER_REJECTION_INTERFACE,
    SemanticRefactorWorldRootAdapter,
    SemanticWorldRoot,
    StaleWriterRejection,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TYPED_TERMINAL_INTERFACE,
    TerminalKind,
    TypedTerminal,
    USES_CURRENT_LEASE_FENCE,
    USES_CURRENT_WORKTREE,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    VFS_AUTHORITY_OWNER,
    VFS_OUTBOX_NOMINATION_INTERFACE,
    VfsOutboxNomination,
    WAL_RECOVERY_PLAN_INTERFACE,
    WORKER_SELF_APPROVAL,
    WORLD_ROOT_CONTRACT_VERSION,
    WORLD_ROOT_RECEIPT_INTERFACE,
    WalRecoveryPlan,
    WorldRootAdapterError,
    WorldRootReceipt,
    assert_not_competing_capsule_family,
    decode_canonical_receipt,
    dry_run_world_root,
    encode_canonical_receipt,
    integrate_world_root,
    persist_through_kit_authorities,
    provider_free_exports,
    recover_world_root,
    world_root_adapter_cid_profile,
    world_root_adapter_descriptor,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "world_root_adapter.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/world_root_adapter.py",
    "test/api/semantic_refactoring/test_world_root_adapter.py",
)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
)
CAPSULE_TYPES = (
    "FunctionSemanticCapsule",
    "MethodSemanticCapsule",
    "ClassSemanticCapsule",
    "TopLevelBlockCapsule",
    "ModuleSemanticCapsule",
    "PackageSemanticCapsule",
    "CallsiteSemanticCapsule",
    "StateOwnerCapsule",
    "RegistrationCapsule",
    "ResourceLifecycleCapsule",
    "WorldRootStore",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
OTHER_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
NESTED_TREE = "80bbdc3443e560b9bf40339c864a32689ccad8ef"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _gitlink(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "gitlink_path": "ipfs_kit_py",
        "nested_repository_id": "ipfs_kit_py",
        "nested_tree_id": NESTED_TREE,
        "owner_repository": "ipfs_kit_py",
        "explicit": True,
    }
    fields.update(overrides)
    return fields


def _evidence(**overrides: Any) -> dict[str, Any]:
    root = _cid("world-root:pre")
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "pre_world_root_cid": root,
        "current_world_root_cid": root,
        "expected_root_generation": 3,
        "current_root_generation": 3,
        "packet_cids": [_cid("packet")],
        "projection_cids": [_cid("projection")],
        "receipt_cids": [_cid("receipt")],
        "transition_cids": [_cid("transition")],
        "task_owner": ACCELERATOR_TASK_OWNER,
        "gitlinks": [],
        "nested_write_paths": [],
        "kit_vfs_available": True,
        "crash": False,
        "network": NETWORK_DENY,
        "worktree_id": _cid("worktree"),
        "lease_id": "lease-1",
        "fence_id": "fence-1",
    }
    fields.update(overrides)
    return fields


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-039"
    assert GOAL_ID == "SPAR-G072"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-025", "SPAR-033")
    assert (
        SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_INTERFACE
        == "SemanticRefactorWorldRootAdapter@1"
    )
    assert SEMANTIC_WORLD_ROOT_INTERFACE == "SemanticWorldRoot@1"
    assert GENERATION_CAS_BINDING_INTERFACE == "GenerationCasBinding@1"
    assert VFS_OUTBOX_NOMINATION_INTERFACE == "VfsOutboxNomination@1"
    assert WAL_RECOVERY_PLAN_INTERFACE == "WalRecoveryPlan@1"
    assert STALE_WRITER_REJECTION_INTERFACE == "StaleWriterRejection@1"
    assert GITLINK_BINDING_INTERFACE == "GitlinkBinding@1"
    assert CROSS_REPOSITORY_OWNERSHIP_INTERFACE == "CrossRepositoryOwnership@1"
    assert WORLD_ROOT_RECEIPT_INTERFACE == "WorldRootReceipt@1"
    assert TYPED_TERMINAL_INTERFACE == "TypedTerminal@1"
    assert WORLD_ROOT_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("world_root_adapter@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()
    assert PERSIST_KINDS == ("packet", "projection", "receipt", "transition")
    assert DECLARED_PERSIST_KINDS == set(PERSIST_KINDS)
    assert DECLARED_ADAPTER_STATUSES == {
        "nominated_persist",
        "rejected_stale_writer",
        "rejected_root_conflict",
        "recovered",
        "typed_terminal",
    }
    assert DECLARED_TERMINAL_KINDS == {
        "unsupported",
        "human_review",
        "capability_unavailable",
        "undeclared_repository",
    }
    assert dict(DECLARED_REPOSITORY_OWNERS) == {
        "ipfs_accelerate_py": "operational",
        "ipfs_datasets_py": "semantic",
        "ipfs_kit_py": "storage",
    }


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "storage and retrieval authority"
    assert AUTHORITY_OWNER == "ipfs_kit_py"
    assert VFS_AUTHORITY_OWNER == "ipfs_kit_py"
    assert CAS_AUTHORITY_OWNER == "ipfs_kit_py"
    assert ADAPTER_CAN_AUTHORIZE_COMPLETION is False
    assert ADAPTER_CAN_AUTHORIZE_TRANSITION is False
    assert ADAPTER_CAN_CREATE_AUTHORITY is False
    assert ADAPTER_OWNS_VFS is False
    assert ADAPTER_OWNS_CAS is False
    assert ADAPTER_WRITES_REPOSITORY is False
    assert KIT_OWNS_VFS is True
    assert KIT_OWNS_CAS is True
    assert KIT_OWNS_OUTBOX is True
    assert KIT_OWNS_WAL_RECOVERY is True
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert ADAPTER_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert NEGATIVE_EVIDENCE_RETAINED is True
    assert STALE_WRITER_REJECTED is True
    assert ROOT_CONFLICT_OVERWRITE_FORBIDDEN is True
    assert GENERATION_CAS_REQUIRED is True
    assert CRASH_RECOVERY_WITHOUT_OVERWRITE is True
    assert NESTED_WRITES_REQUIRE_EXPLICIT_GITLINK is True
    assert NESTED_WRITES_REQUIRE_ACCELERATOR_TASK_OWNER is True
    assert USES_CURRENT_LEASE_FENCE is True
    assert USES_CURRENT_WORKTREE is True
    profile = world_root_adapter_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert "kit_vfs" in EXISTING_ADAPTER_AUTHORITIES
    assert "kit_generation_cas" in EXISTING_ADAPTER_AUTHORITIES
    assert "WorldRootStore" in FORBIDDEN_WORLD_ROOT_NAMES
    descriptor = world_root_adapter_descriptor()
    assert descriptor["nomination_only"] is True
    assert descriptor["writes_repository"] is False
    assert descriptor["predecessor_task_ids"] == ["SPAR-025", "SPAR-033"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "SemanticRefactorWorldRootAdapter" in names
    assert "WorldRootReceipt" in names
    assert "SemanticWorldRoot" in names
    assert "WorldRootStore" not in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "SemanticRefactorWorldRootAdapter" in exports
    assert "integrate_world_root" in exports
    assert "persist_through_kit_authorities" in exports
    assert "recover_world_root" in exports
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "from ipfs_kit_py" not in source
    assert "WorldRootStore" in FORBIDDEN_WORLD_ROOT_NAMES


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_persist_nominates_kit_artifacts_without_writing() -> None:
    receipt = persist_through_kit_authorities(_evidence())
    assert receipt.status == AdapterStatus.NOMINATED_PERSIST.value
    assert receipt.nominated is True
    assert receipt.accepted is False
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_authority is False
    assert receipt.world_root.resulting_root_generation == 4
    assert receipt.world_root.expected_root_generation == 3
    assert receipt.world_root.pre_world_root_cid != receipt.world_root.post_world_root_cid
    assert receipt.cas_binding.cas_matches is True
    kinds = {item.artifact_kind for item in receipt.outbox}
    assert kinds == {"packet", "projection", "receipt", "transition"}
    assert all(item.mutated is False for item in receipt.outbox)
    assert all(item.writes_repository is False for item in receipt.outbox)
    assert all(item.kit_owns_vfs is True for item in receipt.outbox)
    assert receipt.analyzer_id == ANALYZER_ID
    encoded = encode_canonical_receipt(receipt)
    restored = decode_canonical_receipt(encoded)
    assert restored == receipt
    assert restored.receipt_cid == receipt.receipt_cid


def test_stale_writer_is_rejected_without_overwrite() -> None:
    current = _cid("world-root:current")
    receipt = integrate_world_root(
        _evidence(
            expected_root_generation=2,
            current_root_generation=4,
            pre_world_root_cid=_cid("world-root:stale"),
            current_world_root_cid=current,
        )
    )
    assert receipt.status == AdapterStatus.REJECTED_STALE_WRITER.value
    assert receipt.nominated is False
    assert receipt.accepted is False
    assert receipt.overwrite_prevented is True
    assert receipt.world_root.post_world_root_cid == current
    assert receipt.world_root.resulting_root_generation == 4
    assert receipt.stale_writer is not None
    assert receipt.stale_writer.overwrite_prevented is True
    assert receipt.stale_writer.rejection_cid in receipt.negative_evidence_cids


def test_root_conflict_is_rejected_without_overwrite() -> None:
    current = _cid("world-root:current")
    receipt = integrate_world_root(
        _evidence(
            pre_world_root_cid=_cid("world-root:other"),
            current_world_root_cid=current,
            expected_root_generation=3,
            current_root_generation=3,
        )
    )
    assert receipt.status == AdapterStatus.REJECTED_ROOT_CONFLICT.value
    assert receipt.accepted is False
    assert receipt.overwrite_prevented is True
    assert receipt.world_root.post_world_root_cid == current
    assert receipt.world_root.resulting_root_generation == 3
    assert receipt.cas_binding.root_conflict is True


def test_crash_recovers_last_committed_root_without_applying_outbox() -> None:
    committed = _cid("world-root:committed")
    pending = _cid("pending-outbox")
    receipt = recover_world_root(
        _evidence(
            crash=True,
            current_world_root_cid=_cid("world-root:dirty"),
            last_committed_world_root_cid=committed,
            last_committed_root_generation=3,
            pending_outbox_cids=[pending],
        )
    )
    assert receipt.status == AdapterStatus.RECOVERED.value
    assert receipt.accepted is False
    assert receipt.overwrite_prevented is True
    assert receipt.recovery is not None
    assert receipt.recovery.recovered_world_root_cid == committed
    assert receipt.recovery.recovered_root_generation == 3
    assert pending in receipt.recovery.pending_outbox_cids
    assert receipt.world_root.post_world_root_cid == committed
    assert receipt.world_root.resulting_root_generation == 3
    assert pending in receipt.negative_evidence_cids


def test_stale_writer_after_recovery_cannot_overwrite() -> None:
    committed = _cid("world-root:committed")
    recovered = recover_world_root(
        _evidence(
            crash=True,
            current_world_root_cid=committed,
            last_committed_world_root_cid=committed,
            last_committed_root_generation=5,
            expected_root_generation=5,
            current_root_generation=5,
        )
    )
    stale = integrate_world_root(
        _evidence(
            expected_root_generation=4,
            current_root_generation=recovered.world_root.resulting_root_generation,
            pre_world_root_cid=_cid("world-root:stale"),
            current_world_root_cid=recovered.world_root.post_world_root_cid,
        )
    )
    assert recovered.status == AdapterStatus.RECOVERED.value
    assert stale.status == AdapterStatus.REJECTED_STALE_WRITER.value
    assert stale.world_root.post_world_root_cid == recovered.world_root.post_world_root_cid
    assert stale.world_root.resulting_root_generation == 5


def test_explicit_gitlink_nested_write_nominates_when_accelerate_owns_task() -> None:
    receipt = integrate_world_root(
        _evidence(
            gitlinks=[_gitlink()],
            nested_write_paths=["ipfs_kit_py/ipfs_kit_py/semantic_refactoring/projection_store.py"],
        )
    )
    assert receipt.status == AdapterStatus.NOMINATED_PERSIST.value
    assert receipt.ownership is not None
    assert receipt.ownership.task_owner == ACCELERATOR_TASK_OWNER
    assert receipt.ownership.gitlinks[0].explicit is True


def test_nested_write_without_explicit_gitlink_is_typed_terminal() -> None:
    receipt = integrate_world_root(
        _evidence(
            gitlinks=[_gitlink(explicit=False)],
            nested_write_paths=["ipfs_kit_py/block.py"],
        )
    )
    assert receipt.status == AdapterStatus.TYPED_TERMINAL.value
    assert receipt.accepted is False
    assert receipt.nominated is False
    assert receipt.terminal is not None
    assert receipt.terminal.kind == TerminalKind.UNSUPPORTED.value
    assert receipt.world_root.post_world_root_cid == receipt.cas_binding.current_world_root_cid


def test_nested_write_without_covering_gitlink_is_typed_terminal() -> None:
    receipt = integrate_world_root(
        _evidence(
            gitlinks=[_gitlink()],
            nested_write_paths=["ipfs_datasets_py/secret.py"],
        )
    )
    assert receipt.status == AdapterStatus.TYPED_TERMINAL.value
    assert receipt.terminal.kind == TerminalKind.UNSUPPORTED.value


def test_nested_write_by_non_accelerator_owner_is_typed_terminal() -> None:
    receipt = integrate_world_root(
        _evidence(
            task_owner="ipfs_kit_py",
            gitlinks=[_gitlink()],
            nested_write_paths=["ipfs_kit_py/block.py"],
        )
    )
    assert receipt.status == AdapterStatus.TYPED_TERMINAL.value
    assert receipt.terminal.kind == TerminalKind.UNSUPPORTED.value


def test_undeclared_repository_is_typed_terminal_never_success() -> None:
    receipt = integrate_world_root(
        _evidence(
            gitlinks=[
                _gitlink(
                    gitlink_path="foreign_repo",
                    nested_repository_id="foreign_repo",
                    owner_repository="foreign_repo",
                )
            ],
            nested_write_paths=["foreign_repo/mod.py"],
        )
    )
    assert receipt.status == AdapterStatus.TYPED_TERMINAL.value
    assert receipt.accepted is False
    assert receipt.terminal.kind == TerminalKind.UNDECLARED_REPOSITORY.value


def test_kit_unavailable_is_typed_terminal() -> None:
    receipt = integrate_world_root(_evidence(kit_vfs_available=False))
    assert receipt.status == AdapterStatus.TYPED_TERMINAL.value
    assert receipt.terminal.kind == TerminalKind.CAPABILITY_UNAVAILABLE.value
    assert receipt.accepted is False


def test_explicit_human_review_terminal_stops_without_completion() -> None:
    receipt = integrate_world_root(
        _evidence(
            terminal={
                "kind": TerminalKind.HUMAN_REVIEW.value,
                "reason": "cross-repository ownership requires review",
            }
        )
    )
    assert receipt.status == AdapterStatus.TYPED_TERMINAL.value
    assert receipt.terminal.kind == TerminalKind.HUMAN_REVIEW.value
    assert receipt.accepted is False


def test_empty_persist_without_crash_fails_closed() -> None:
    with pytest.raises(WorldRootAdapterError, match="requires packet"):
        integrate_world_root(
            _evidence(
                packet_cids=[],
                projection_cids=[],
                receipt_cids=[],
                transition_cids=[],
            )
        )


def test_wrong_gitlink_owner_fails_closed() -> None:
    with pytest.raises(WorldRootAdapterError, match="declared nested ownership"):
        integrate_world_root(
            _evidence(
                gitlinks=[_gitlink(owner_repository="ipfs_accelerate_py")],
            )
        )


def test_dry_run_is_deterministic_and_never_mutates() -> None:
    payload = _evidence()
    first = dry_run_world_root(payload)
    second = SemanticRefactorWorldRootAdapter().dry_run(payload)
    assert first.receipt_cid == second.receipt_cid
    assert DRY_RUN_MUTATES is False
    with pytest.raises(WorldRootAdapterError, match="cannot mutate"):
        integrate_world_root(payload, mutate=True)
    with pytest.raises(WorldRootAdapterError, match="cannot mutate"):
        integrate_world_root(_evidence(mutate=True))


def test_identity_excludes_observational_fields() -> None:
    receipt = integrate_world_root(_evidence())
    encoded = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(encoded))
    encoded["timestamp"] = "now"
    with pytest.raises(WorldRootAdapterError, match="observational"):
        WorldRootReceipt.from_dict(encoded)
    with pytest.raises(WorldRootAdapterError, match="observational"):
        integrate_world_root(_evidence(model_output="guess"))


def test_vector_and_model_evidence_cannot_admit_a_root() -> None:
    with pytest.raises(WorldRootAdapterError, match="cannot admit"):
        integrate_world_root(_evidence(vector_candidate={"score": 1}))
    with pytest.raises(WorldRootAdapterError, match="cannot admit"):
        integrate_world_root(_evidence(model_hypothesis={"ok": True}))
    with pytest.raises(WorldRootAdapterError, match="cannot admit"):
        integrate_world_root(_evidence(heuristic=True))


def test_network_is_denied() -> None:
    with pytest.raises(WorldRootAdapterError, match="network is denied"):
        integrate_world_root(_evidence(network="allow"))


def test_tree_mismatch_fails_closed() -> None:
    receipt = integrate_world_root(_evidence())
    payload = receipt.to_dict()
    payload["tree_id"] = OTHER_TREE
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(WorldRootAdapterError, match="tree_id"):
        WorldRootReceipt.from_dict(payload)
    with pytest.raises(WorldRootAdapterError, match="tree_id"):
        integrate_world_root(_evidence(tree_id="not-a-tree"))


def test_worker_cannot_self_approve_receipt() -> None:
    receipt = integrate_world_root(_evidence())
    payload = receipt.to_dict()
    payload["accepted"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(WorldRootAdapterError, match="self-approve"):
        WorldRootReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(WorldRootAdapterError, match="can_authorize_completion"):
        WorldRootReceipt.from_dict(payload)


def test_vfs_outbox_cannot_claim_mutation_or_authority() -> None:
    with pytest.raises(WorldRootAdapterError, match="cannot mutate VFS"):
        VfsOutboxNomination(
            artifact_kind="packet",
            artifact_cid=_cid("packet"),
            tree_id=TREE_ID,
            mutated=True,
        )
    with pytest.raises(WorldRootAdapterError, match="cannot write"):
        VfsOutboxNomination(
            artifact_kind="packet",
            artifact_cid=_cid("packet"),
            tree_id=TREE_ID,
            writes_repository=True,
        )
    nomination = VfsOutboxNomination(
        artifact_kind="transition",
        artifact_cid=_cid("transition"),
        tree_id=TREE_ID,
    )
    restored = VfsOutboxNomination.from_dict(nomination.to_dict())
    assert restored == nomination
    assert restored.can_authorize_transition is False


def test_recovery_plan_cannot_overwrite_committed_generation() -> None:
    committed = _cid("committed")
    with pytest.raises(WorldRootAdapterError, match="last committed"):
        WalRecoveryPlan(
            last_committed_world_root_cid=committed,
            last_committed_root_generation=3,
            recovered_world_root_cid=_cid("other"),
            recovered_root_generation=3,
            pending_outbox_cids=[],
        )
    with pytest.raises(WorldRootAdapterError, match="cannot advance"):
        WalRecoveryPlan(
            last_committed_world_root_cid=committed,
            last_committed_root_generation=3,
            recovered_world_root_cid=committed,
            recovered_root_generation=4,
            pending_outbox_cids=[],
        )


def test_generation_cas_binding_round_trip() -> None:
    binding = GenerationCasBinding(
        expected_root_generation=3,
        current_root_generation=3,
        pre_world_root_cid=_cid("pre"),
        current_world_root_cid=_cid("pre"),
    )
    restored = GenerationCasBinding.from_dict(binding.to_dict())
    assert restored == binding
    assert restored.cas_matches is True
    conflict = GenerationCasBinding(
        expected_root_generation=3,
        current_root_generation=3,
        pre_world_root_cid=_cid("pre"),
        current_world_root_cid=_cid("other"),
    )
    assert conflict.root_conflict is True
    assert conflict.cas_matches is False


def test_gitlink_and_world_root_round_trip() -> None:
    binding = GitlinkBinding.from_mapping(_gitlink())
    restored = GitlinkBinding.from_dict(binding.to_dict())
    assert restored == binding
    assert binding.covers("ipfs_kit_py/core/vfs.py")
    assert not binding.covers("ipfs_datasets_py/mod.py")
    ownership = CrossRepositoryOwnership(
        task_owner=ACCELERATOR_TASK_OWNER,
        gitlinks=[binding],
        nested_write_paths=["ipfs_kit_py/core/vfs.py"],
    )
    assert CrossRepositoryOwnership.from_dict(ownership.to_dict()) == ownership
    root = SemanticWorldRoot(
        tree_id=TREE_ID,
        pre_world_root_cid=_cid("pre"),
        post_world_root_cid=_cid("post"),
        expected_root_generation=1,
        resulting_root_generation=2,
        packet_cids=[_cid("packet")],
    )
    assert SemanticWorldRoot.from_dict(root.to_dict()) == root


def test_adapter_methods_match_module_functions() -> None:
    adapter = SemanticRefactorWorldRootAdapter()
    payload = _evidence()
    assert adapter.interface == SEMANTIC_REFACTOR_WORLD_ROOT_ADAPTER_INTERFACE
    assert adapter.integrate(payload).receipt_cid == integrate_world_root(payload).receipt_cid
    assert adapter.persist(payload).receipt_cid == persist_through_kit_authorities(
        payload
    ).receipt_cid
    recovered = adapter.recover(_evidence(crash=True, last_committed_root_generation=3))
    assert recovered.status == AdapterStatus.RECOVERED.value
    terminal = TypedTerminal(
        kind=TerminalKind.CAPABILITY_UNAVAILABLE.value,
        reason="required kit VFS is unavailable",
    )
    assert TypedTerminal.from_dict(terminal.to_dict()) == terminal
    rejection = StaleWriterRejection(
        attempted_expected_root_generation=1,
        current_root_generation=2,
        attempted_pre_world_root_cid=_cid("old"),
        current_world_root_cid=_cid("now"),
    )
    assert StaleWriterRejection.from_dict(rejection.to_dict()) == rejection
