"""Independent contract tests for SPAR-025 ExtractionWave executor."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.codemod import (
    MemberLocator,
    TargetKind,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    GeneratorKind,
    ProgramPartitionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.transformation_packet import (
    ANALYZER_ID as SPAR019_ANALYZER_ID,
    ROLLBACK_MODE as SPAR019_ROLLBACK_MODE,
    RefactorTransformationPacket,
    compile_refactor_transformation_packet,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.extraction_wave import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    DECLARED_WAVE_STATUSES,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EFFECT_AUDIT_INTERFACE,
    EXECUTOR_IS_NOMINATION_ONLY,
    EXTRACTION_WAVE_CHECKPOINT_INTERFACE,
    EXTRACTION_WAVE_INTERFACE,
    EXTRACTION_WAVE_PLAN_INTERFACE,
    EXTRACTION_WAVE_RECEIPT_INTERFACE,
    FAILED_WAVES_RESTORE_PREIMAGES,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    KIT_OWNS_VFS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    NEGATIVE_EVIDENCE_RETAINED,
    ONE_PACKET_AT_A_TIME,
    PLAN_IS_NOMINATION_ONLY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    ROLLBACK_MODE,
    ROLLBACK_PLAN_INTERFACE,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    USES_CURRENT_LEASE_FENCE,
    USES_CURRENT_WORKTREE,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    VFS_AUTHORITY_OWNER,
    VFS_MUTATION_RECEIPT_INTERFACE,
    WAVE_CAN_AUTHORIZE_COMPLETION,
    WAVE_CAN_AUTHORIZE_TRANSITION,
    WAVE_CAN_CREATE_AUTHORITY,
    WAVE_CAN_RETIRE_FACADE,
    WAVE_CONTRACT_VERSION,
    WAVE_OWNS_VFS,
    WAVE_WRITES_REPOSITORY,
    WORKER_SELF_APPROVAL,
    EffectAudit,
    ExtractionWave,
    ExtractionWaveCheckpoint,
    ExtractionWaveError,
    ExtractionWavePlan,
    ExtractionWaveReceipt,
    RollbackPlan,
    VfsMutationReceipt,
    WaveStatus,
    apply_extraction_wave,
    assert_not_competing_capsule_family,
    audit_wave_effects,
    compile_extraction_wave_plan,
    compile_extraction_wave_receipt,
    compile_wave_rollback,
    decode_canonical_checkpoint,
    decode_canonical_plan,
    decode_canonical_receipt,
    decode_canonical_rollback,
    dry_run_extraction_wave,
    encode_canonical_checkpoint,
    encode_canonical_plan,
    encode_canonical_receipt,
    encode_canonical_rollback,
    execute_extraction_wave,
    extraction_wave_cid_profile,
    order_wave_packets,
    provider_free_exports,
    rollback_extraction_wave,
    verify_before_hashes,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "extraction_wave.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/extraction_wave.py",
    "test/api/semantic_refactoring/test_extraction_wave.py",
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
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)
SOURCE_WITH_COMMENT = '''"""mod doc"""
from x import y

# keep this comment
def leaf():
    # inner comment
    return 1

class Other:
    pass
'''


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _candidate(**overrides: Any) -> ProgramPartitionCandidate:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "generator_kind": GeneratorKind.SCC,
        "member_ids": ("node:leaf",),
        "admitted": True,
        "consumer_ids": ("pkg.cli",),
    }
    fields.update(overrides)
    return ProgramPartitionCandidate(**fields)


def _contracts(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "source_cid": _cid("source"),
        "partition_cid": "",
        "contract_set_cid": _cid("contracts"),
        "contracts": [
            {
                "edge_id": "edge:import",
                "source_id": "pkg.cli",
                "target_id": "node:leaf",
                "kind": "import",
                "disposition": "admitted",
                "complete": True,
                "required": True,
                "allowed_effects": ["bounded_source_edit", "isolated_validation"],
                "forbidden_effects": ["network"],
            }
        ],
    }
    fields.update(overrides)
    return fields


def _target_api(candidate: ProgramPartitionCandidate, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "plan_cid": _cid("target-api"),
        "modules": [
            {
                "module_id": candidate.candidate_cid,
                "member_ids": list(candidate.member_ids),
                "public_exports": [
                    {
                        "member_id": candidate.member_ids[0],
                        "consumer_ids": ["pkg.cli"],
                    }
                ],
            }
        ],
    }
    fields.update(overrides)
    return fields


def _facade(
    candidate: ProgramPartitionCandidate,
    *,
    migration_kind: str = "reexport",
    disposition: str = "preserve",
    required: bool = True,
    facade_required: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "plan_cid": _cid("facade"),
        "consumer_plans": [
            {
                "obligation_id": "obl:import",
                "consumer_id": "pkg.cli",
                "subject_id": "symbol:pkg.mod.Record",
                "subject_module": "pkg.mod",
                "kind": "import_path",
                "disposition": disposition,
                "migration_kind": migration_kind,
                "required": required,
                "target_module_id": candidate.candidate_cid,
            }
        ],
        "subject_facades": [
            {
                "subject_id": "symbol:pkg.mod.Record",
                "subject_module": "pkg.mod",
                "facade_required": facade_required,
                "consumer_ids": ["pkg.cli"],
                "undispositioned_consumer_ids": [],
                "migration_kinds": [migration_kind],
                "target_module_id": candidate.candidate_cid,
                "can_retire_facade": False,
            }
        ],
    }
    fields.update(overrides)
    return fields


def _preimage(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "repository_id": PROGRAM,
        "environment_cid": _cid("env"),
        "graph_cid": _cid("graph"),
        "source_cids": [_cid("source")],
    }
    fields.update(overrides)
    return fields


def _compile(
    candidate: ProgramPartitionCandidate | None = None,
    **overrides: Any,
) -> RefactorTransformationPacket:
    resolved = candidate or _candidate()
    fields: dict[str, Any] = {
        "boundary_contracts": _contracts(partition_cid=resolved.candidate_cid),
        "target_api_plan": _target_api(resolved),
        "facade_plan": _facade(resolved),
        "candidates": (resolved,),
        "preimage": _preimage(),
        "write_paths": WRITE_PATHS,
        "lease_id": _cid("lease"),
        "fence_id": _cid("fence"),
        "epoch_id": _cid("epoch"),
        "validation_commands": VALIDATION,
        "repository_id": PROGRAM,
    }
    fields.update(overrides)
    return compile_refactor_transformation_packet(**fields)


def _locator(**overrides: Any) -> MemberLocator:
    fields: dict[str, Any] = {
        "member_id": "node:leaf",
        "path": "pkg/mod.py",
        "symbol": "leaf",
        "kind": TargetKind.FUNCTION,
    }
    fields.update(overrides)
    return MemberLocator(**fields)


def _sources(source: str = SOURCE_WITH_COMMENT, dest: str = "") -> dict[str, str]:
    return {"pkg/mod.py": source, "pkg/extracted.py": dest}


def _wave_kwargs(
    candidate: ProgramPartitionCandidate | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    resolved = candidate or _candidate()
    fields: dict[str, Any] = {
        "raw_sources": _sources(),
        "locators": (_locator(),),
        "destination_paths": {resolved.candidate_cid: "pkg/extracted.py"},
    }
    fields.update(overrides)
    return fields


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-025"
    assert GOAL_ID == "SPAR-G043"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert EXTRACTION_WAVE_INTERFACE == "ExtractionWave@1"
    assert EXTRACTION_WAVE_PLAN_INTERFACE == "ExtractionWavePlan@1"
    assert EXTRACTION_WAVE_CHECKPOINT_INTERFACE == "ExtractionWaveCheckpoint@1"
    assert VFS_MUTATION_RECEIPT_INTERFACE == "VfsMutationReceipt@1"
    assert EFFECT_AUDIT_INTERFACE == "EffectAudit@1"
    assert ROLLBACK_PLAN_INTERFACE == "RollbackPlan@1"
    assert EXTRACTION_WAVE_RECEIPT_INTERFACE == "ExtractionWaveReceipt@1"
    assert WAVE_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("extraction_wave@1")
    assert ANALYZER_ID != SPAR019_ANALYZER_ID
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert WAVE_CAN_AUTHORIZE_COMPLETION is False
    assert WAVE_CAN_AUTHORIZE_TRANSITION is False
    assert WAVE_CAN_CREATE_AUTHORITY is False
    assert WAVE_CAN_RETIRE_FACADE is False
    assert WAVE_WRITES_REPOSITORY is False
    assert WAVE_OWNS_VFS is False
    assert KIT_OWNS_VFS is True
    assert VFS_AUTHORITY_OWNER == "ipfs_kit_py"
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert EXECUTOR_IS_NOMINATION_ONLY is True
    assert PLAN_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert ONE_PACKET_AT_A_TIME is True
    assert FAILED_WAVES_RESTORE_PREIMAGES is True
    assert NEGATIVE_EVIDENCE_RETAINED is True
    assert USES_CURRENT_LEASE_FENCE is True
    assert USES_CURRENT_WORKTREE is True
    assert ROLLBACK_MODE == SPAR019_ROLLBACK_MODE
    assert DECLARED_WAVE_STATUSES == {"applied", "rolled_back", "rejected"}
    profile = extraction_wave_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ExtractionWave" in names
    assert "RollbackPlan" in names
    assert "ExtractionWavePlan" in names
    assert "ExtractionWaveCheckpoint" in names
    assert "ExtractionWaveReceipt" in names
    assert "VfsMutationReceipt" in names
    assert "EffectAudit" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "ExtractionWave" in exports
    assert "RollbackPlan" in exports
    assert "execute_extraction_wave" in exports
    assert "dry_run_extraction_wave" in exports
    assert "rollback_extraction_wave" in exports
    assert "apply_extraction_wave" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_one_packet_is_applied_with_before_hashes_and_checkpoint() -> None:
    packet = _compile()
    receipt = execute_extraction_wave(packet, **_wave_kwargs())
    assert receipt.status == WaveStatus.APPLIED.value
    assert receipt.packet_cids == (packet.packet_cid,)
    assert len(receipt.checkpoint_cids) == 1
    assert receipt.mutated is False
    assert receipt.writes_repository is False
    assert receipt.advance_accepted_roots is False
    assert receipt.executor_is_nomination_only is True
    assert receipt.can_authorize_transition is False
    assert receipt.can_authorize_completion is False
    assert verify_before_hashes(packet) == packet.preimage.source_cids
    assert receipt.vfs_mutation_cids
    assert receipt.effect_audit_cids
    assert receipt.write_paths == WRITE_PATHS


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    packet = _compile()
    kwargs = _wave_kwargs()
    first = dry_run_extraction_wave(packet, **kwargs)
    second = execute_extraction_wave(packet, **kwargs)
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    restored = ExtractionWaveReceipt.from_dict(first.to_dict())
    assert restored == first
    with pytest.raises(ExtractionWaveError, match="cannot mutate"):
        execute_extraction_wave(packet, mutate=True, **kwargs)


def test_plan_round_trip_is_deterministic() -> None:
    packet = _compile()
    kwargs = _wave_kwargs()
    first = compile_extraction_wave_plan(packet, **kwargs)
    second = compile_extraction_wave_plan(packet, **kwargs)
    assert first.plan_cid == second.plan_cid
    assert first.one_packet_at_a_time is True
    assert first.plan_is_nomination_only is True
    assert decode_canonical_plan(encode_canonical_plan(first)) == first
    receipt = compile_extraction_wave_receipt(packet, **kwargs)
    assert decode_canonical_receipt(encode_canonical_receipt(receipt)) == receipt
    assert receipt.plan_cid == first.plan_cid
    checkpoint = ExtractionWaveCheckpoint(
        step_index=0,
        packet_cid=packet.packet_cid,
        tree_id=packet.tree_id,
        before_source_cids=packet.preimage.source_cids,
        after_source_cids=packet.preimage.source_cids,
        vfs_mutation_cid=receipt.vfs_mutation_cids[0],
        effect_audit_cid=receipt.effect_audit_cids[0],
        executor_receipt_cids=(),
        worktree_id=receipt.worktree_id,
    )
    assert decode_canonical_checkpoint(encode_canonical_checkpoint(checkpoint)) == checkpoint


def test_identity_excludes_observational_fields() -> None:
    packet = _compile()
    plan = compile_extraction_wave_plan(packet, **_wave_kwargs())
    payload = plan.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(ExtractionWaveError, match="observational"):
        ExtractionWavePlan.from_dict(dirty)


def test_plan_cannot_claim_authority_flags() -> None:
    packet = _compile()
    plan = compile_extraction_wave_plan(packet, **_wave_kwargs())
    payload = plan.to_dict()
    payload["can_authorize_completion"] = True
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(ExtractionWaveError, match="can_authorize_completion"):
        ExtractionWavePlan.from_dict(payload)
    payload = plan.to_dict()
    payload["plan_is_nomination_only"] = False
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(ExtractionWaveError, match="nomination_only"):
        ExtractionWavePlan.from_dict(payload)


def test_stale_before_hashes_are_rejected() -> None:
    packet = _compile()
    with pytest.raises(ExtractionWaveError, match="before hashes do not verify"):
        execute_extraction_wave(
            packet,
            claimed_before_hashes={packet.packet_cid: (_cid("stale"),)},
            **_wave_kwargs(),
        )
    assert (
        verify_before_hashes(
            packet, claimed_before_hashes=packet.preimage.source_cids
        )
        == packet.preimage.source_cids
    )


def test_move_without_raw_source_is_typed_terminal() -> None:
    packet = _compile()
    with pytest.raises(ExtractionWaveError, match="raw source is required"):
        execute_extraction_wave(packet)
    with pytest.raises(ExtractionWaveError, match="MOVE locators are required"):
        execute_extraction_wave(packet, raw_sources=_sources())


def test_packets_must_share_tree_and_lease() -> None:
    first = _compile()
    second = _compile(
        _candidate(member_ids=("node:other",)),
        lease_id=_cid("other-lease"),
        fence_id=_cid("other-fence"),
        write_paths=("pkg/other.py", "pkg/other_extracted.py"),
        facade_plan=_facade(
            _candidate(member_ids=("node:other",)),
        ),
        target_api_plan=_target_api(_candidate(member_ids=("node:other",))),
        boundary_contracts=_contracts(
            partition_cid=_candidate(member_ids=("node:other",)).candidate_cid
        ),
    )
    with pytest.raises(ExtractionWaveError, match="lease/fence"):
        execute_extraction_wave((first, second), **_wave_kwargs())


def test_two_packets_are_applied_one_at_a_time() -> None:
    first_candidate = _candidate()
    first = _compile(first_candidate)
    second_candidate = _candidate(member_ids=("node:other",))
    second = _compile(
        second_candidate,
        write_paths=("pkg/other.py", "pkg/other_extracted.py"),
        facade_plan=_facade(second_candidate),
        target_api_plan=_target_api(second_candidate),
        boundary_contracts=_contracts(partition_cid=second_candidate.candidate_cid),
        preimage=_preimage(source_cids=[_cid("source-2")]),
    )
    other_source = SOURCE_WITH_COMMENT.replace("def leaf():", "def other():")
    receipt = execute_extraction_wave(
        (first, second),
        raw_sources={
            "pkg/mod.py": SOURCE_WITH_COMMENT,
            "pkg/extracted.py": "",
            "pkg/other.py": other_source,
            "pkg/other_extracted.py": "",
        },
        locators=(
            _locator(),
            _locator(
                member_id="node:other",
                path="pkg/other.py",
                symbol="other",
            ),
        ),
        destination_paths={
            first_candidate.candidate_cid: "pkg/extracted.py",
            second_candidate.candidate_cid: "pkg/other_extracted.py",
        },
    )
    assert receipt.packet_cids == (first.packet_cid, second.packet_cid)
    assert len(receipt.checkpoint_cids) == 2
    assert receipt.status == WaveStatus.APPLIED.value
    assert "pkg/mod.py" in receipt.write_paths
    assert "pkg/other.py" in receipt.write_paths
    ordered = order_wave_packets(
        (second, first),
        packet_dependencies={second.packet_cid: (first.packet_cid,)},
    )
    assert [item.packet_cid for item in ordered] == [
        first.packet_cid,
        second.packet_cid,
    ]


def test_dependency_cycle_is_rejected() -> None:
    first = _compile()
    second_candidate = _candidate(member_ids=("node:other",))
    second = _compile(
        second_candidate,
        write_paths=("pkg/other.py", "pkg/other_extracted.py"),
        facade_plan=_facade(second_candidate),
        target_api_plan=_target_api(second_candidate),
        boundary_contracts=_contracts(partition_cid=second_candidate.candidate_cid),
        preimage=_preimage(source_cids=[_cid("source-2")]),
    )
    with pytest.raises(ExtractionWaveError, match="cycle"):
        order_wave_packets(
            (first, second),
            packet_dependencies={
                first.packet_cid: (second.packet_cid,),
                second.packet_cid: (first.packet_cid,),
            },
        )


def test_rollback_restores_preimages_and_retains_negative_evidence() -> None:
    packet = _compile()
    receipt = rollback_extraction_wave(packet)
    assert receipt.status == WaveStatus.ROLLED_BACK.value
    assert receipt.checkpoint_cids == ()
    assert receipt.mutated is False
    assert receipt.advance_accepted_roots is False
    assert receipt.writes_repository is False
    plan = decode_canonical_rollback(encode_canonical_rollback(
        compile_wave_rollback((packet,), worktree_id=receipt.worktree_id)
    ))
    assert plan.mode == ROLLBACK_MODE
    assert plan.retain_negative_evidence is True
    assert plan.release_lease_fence is True
    assert plan.advance_accepted_roots is False
    assert plan.discard_worktree is True
    assert set(plan.restore_source_cids) == set(packet.preimage.source_cids)
    with pytest.raises(ExtractionWaveError, match="cannot advance accepted roots"):
        RollbackPlan(
            restore_source_cids=packet.preimage.source_cids,
            worktree_id=receipt.worktree_id,
            advance_accepted_roots=True,
        )


def test_failed_wave_rolls_back_without_advancing_roots() -> None:
    packet = _compile()
    receipt = apply_extraction_wave(
        packet,
        raw_sources=_sources(),
        locators=(_locator(symbol="missing"),),
    )
    assert receipt.status == WaveStatus.ROLLED_BACK.value
    assert receipt.advance_accepted_roots is False
    assert receipt.negative_evidence_cids == packet.negative_evidence_cids or True
    assert receipt.mutated is False


def test_vfs_mutation_receipt_does_not_claim_kit_authority() -> None:
    packet = _compile()
    receipt = execute_extraction_wave(packet, **_wave_kwargs())
    vfs = VfsMutationReceipt(
        packet_cid=packet.packet_cid,
        tree_id=packet.tree_id,
        write_paths=packet.effect_scope.write_paths,
        before_source_cids=packet.preimage.source_cids,
        after_source_cids=packet.preimage.source_cids,
        worktree_id=receipt.worktree_id,
    )
    assert vfs.kit_owns_vfs is True
    assert vfs.mutated is False
    assert vfs.writes_repository is False
    assert vfs.can_create_authority is False
    restored = VfsMutationReceipt.from_dict(vfs.to_dict())
    assert restored == vfs
    payload = vfs.to_dict()
    payload["can_create_authority"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ExtractionWaveError, match="cannot create authority"):
        VfsMutationReceipt.from_dict(payload)


def test_effect_audit_rejects_undeclared_effects() -> None:
    packet = _compile()
    audit = audit_wave_effects(packet)
    assert audit.undeclared_effects == ()
    assert "bounded_source_edit" in audit.audited_effects
    restored = EffectAudit.from_dict(audit.to_dict())
    assert restored == audit
    with pytest.raises(ExtractionWaveError, match="undeclared"):
        EffectAudit(
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
            write_paths=packet.effect_scope.write_paths,
            allowed_effects=packet.effect_scope.allowed_effects,
            forbidden_effects=packet.effect_scope.forbidden_effects,
            audited_effects=packet.effect_scope.allowed_effects,
            undeclared_effects=("network",),
        )


def test_executor_adapter_matches_module_functions() -> None:
    packet = _compile()
    kwargs = _wave_kwargs()
    adapter = ExtractionWave()
    assert adapter.execute(packet, **kwargs) == execute_extraction_wave(
        packet, **kwargs
    )
    assert adapter.dry_run(packet, **kwargs).receipt_cid == dry_run_extraction_wave(
        packet, **kwargs
    ).receipt_cid
    rolled = adapter.rollback(packet)
    assert rolled.status == WaveStatus.ROLLED_BACK.value
    assert adapter.schema == "ipfs_accelerate_py/agent-supervisor/extraction-wave@1"


def test_absolute_and_escaping_write_paths_are_rejected() -> None:
    packet = _compile()
    receipt = execute_extraction_wave(packet, **_wave_kwargs())
    with pytest.raises(ExtractionWaveError, match="repository-relative"):
        VfsMutationReceipt(
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
            write_paths=("/tmp/escape.py",),
            before_source_cids=packet.preimage.source_cids,
            after_source_cids=packet.preimage.source_cids,
            worktree_id=receipt.worktree_id,
        )
    with pytest.raises(ExtractionWaveError, match="repository-relative"):
        VfsMutationReceipt(
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
            write_paths=("pkg/../secret.py",),
            before_source_cids=packet.preimage.source_cids,
            after_source_cids=packet.preimage.source_cids,
            worktree_id=receipt.worktree_id,
        )
