"""Independent contract tests for SPAR-038 fixed-point controller."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.fixed_point import (
    ACCEPTED_WAVE_FIXED_POINT_INTERFACE,
    ALLOWED_VALIDATION_STATUSES,
    ALLOWED_WAVE_STATUSES,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    BOOLEAN_FIXED_POINT_IS_AUTHORITY,
    COMPOSED_ROOT_INTERFACE,
    CONTROLLER_CAN_ACCEPT_FIXED_POINT,
    CONTROLLER_CAN_AUTHORIZE_COMPLETION,
    CONTROLLER_CAN_AUTHORIZE_TRANSITION,
    CONTROLLER_CAN_CREATE_AUTHORITY,
    CONTROLLER_IS_NOMINATION_ONLY,
    CONTROLLER_WRITES_REPOSITORY,
    ControllerStatus,
    ComposedRoot,
    DECLARED_CONTROLLER_STATUSES,
    DECLARED_REBUILD_SLICES,
    DECLARED_TERMINAL_KINDS,
    DECLARED_WAVE_OUTCOMES,
    DEFAULT_MAX_ITERATIONS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXISTING_ADAPTER_AUTHORITIES,
    FEDERATION_FIXED_POINT_STORE_IS_AUTHORITY,
    FIXED_POINT_CONTRACT_VERSION,
    FIXED_POINT_ITERATION_INTERFACE,
    FIXED_POINT_RECEIPT_INTERFACE,
    FIXED_POINT_REMODULARIZATION_CONTROLLER_INTERFACE,
    FORBIDDEN_FIXED_POINT_NAMES,
    FixedPointError,
    FixedPointIteration,
    FixedPointReceipt,
    FixedPointRemodularizationController,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    LIVE_REBUILD_REQUIRED,
    MARKDOWN_IS_NOT_COMPLETION,
    MAX_ITERATIONS,
    MERGE_RECEIPT_INTERFACE,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    MergeReceipt,
    NEGATIVE_EVIDENCE_RETAINED,
    PREBUILT_FIXED_POINT_IS_AUTHORITY,
    PREDECESSOR_TASK_IDS,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    REBUILD_AFTER_ACCEPTED_WAVE,
    REBUILD_AFTER_REJECTED_WAVE,
    REBUILD_SLICE_ORDER,
    REBUILD_SLICE_SET_INTERFACE,
    RebuildSliceSet,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TWO_EPOCH_UNCHANGED_REQUIRED,
    TYPED_TERMINAL_INTERFACE,
    TerminalKind,
    TypedTerminal,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WAVE_DISPOSITION_INTERFACE,
    WORKER_SELF_APPROVAL,
    WaveDisposition,
    WaveOutcome,
    assert_not_competing_capsule_family,
    compose_current_roots,
    decode_canonical_receipt,
    dry_run_fixed_point,
    encode_canonical_receipt,
    fixed_point_cid_profile,
    provider_free_exports,
    rebuild_after_wave,
    run_fixed_point_controller,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "fixed_point.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/fixed_point.py",
    "test/api/semantic_refactoring/test_fixed_point.py",
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
    "FixedPointStore",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
OTHER_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _slices(label: str = "epoch-1") -> dict[str, str]:
    return {name: _cid(f"{label}:{name}") for name in REBUILD_SLICE_ORDER}


def _merge(
    *,
    label: str = "merge",
    wave_cid: str | None = None,
    disposition: str = "accepted",
    validation_cid: str | None = None,
) -> dict[str, Any]:
    return {
        "merge_cid": _cid(label),
        "wave_cid": wave_cid or _cid("wave"),
        "tree_id": TREE_ID,
        "disposition": disposition,
        "validation_cid": validation_cid or _cid("validation"),
    }


def _wave(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "wave_cid": _cid("wave"),
        "status": "applied",
        "packet_cids": [_cid("packet")],
    }
    fields.update(overrides)
    return fields


def _validation(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "validation_cid": _cid("validation"),
        "status": "validated",
    }
    fields.update(overrides)
    return fields


def _evidence(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "current_roots": _slices("prior"),
        "rebuild": _slices("epoch-1"),
        "wave": _wave(),
        "validation": _validation(),
        "merge_receipts": [_merge()],
        "remaining_mandatory_task_cids": [_cid("task:next")],
        "pending_merge_cids": [],
        "iteration": 1,
    }
    fields.update(overrides)
    return fields


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-038"
    assert GOAL_ID == "SPAR-G071"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-027", "SPAR-037")
    assert (
        FIXED_POINT_REMODULARIZATION_CONTROLLER_INTERFACE
        == "FixedPointRemodularizationController@1"
    )
    assert ACCEPTED_WAVE_FIXED_POINT_INTERFACE == "AcceptedWaveFixedPoint@1"
    assert COMPOSED_ROOT_INTERFACE == "ComposedRoot@1"
    assert REBUILD_SLICE_SET_INTERFACE == "RebuildSliceSet@1"
    assert MERGE_RECEIPT_INTERFACE == "SparMergeReceipt@1"
    assert WAVE_DISPOSITION_INTERFACE == "WaveDisposition@1"
    assert TYPED_TERMINAL_INTERFACE == "TypedTerminal@1"
    assert FIXED_POINT_ITERATION_INTERFACE == "FixedPointIteration@1"
    assert FIXED_POINT_RECEIPT_INTERFACE == "FixedPointReceipt@1"
    assert FIXED_POINT_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("fixed_point@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()
    assert REBUILD_SLICE_ORDER == (
        "semantic_state",
        "graph",
        "sccs",
        "contracts",
        "frontier",
        "partitions",
        "selections",
    )
    assert DECLARED_REBUILD_SLICES == set(REBUILD_SLICE_ORDER)
    assert DECLARED_WAVE_OUTCOMES == {"accepted", "rejected"}
    assert DECLARED_CONTROLLER_STATUSES == {
        "continue",
        "nominated_fixed_point",
        "typed_terminal",
    }
    assert DECLARED_TERMINAL_KINDS == {
        "unsupported",
        "human_review",
        "capability_unavailable",
        "max_iterations",
    }
    assert ALLOWED_WAVE_STATUSES == {"applied", "rolled_back", "rejected"}
    assert ALLOWED_VALIDATION_STATUSES == {
        "validated",
        "rejected",
        "unsupported",
        "incomplete",
    }
    assert DEFAULT_MAX_ITERATIONS == 8
    assert MAX_ITERATIONS == 8


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "operational refactoring authority"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert CONTROLLER_CAN_AUTHORIZE_COMPLETION is False
    assert CONTROLLER_CAN_AUTHORIZE_TRANSITION is False
    assert CONTROLLER_CAN_CREATE_AUTHORITY is False
    assert CONTROLLER_CAN_ACCEPT_FIXED_POINT is False
    assert CONTROLLER_WRITES_REPOSITORY is False
    assert FEDERATION_FIXED_POINT_STORE_IS_AUTHORITY is False
    assert BOOLEAN_FIXED_POINT_IS_AUTHORITY is False
    assert PREBUILT_FIXED_POINT_IS_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert CONTROLLER_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert NEGATIVE_EVIDENCE_RETAINED is True
    assert REBUILD_AFTER_ACCEPTED_WAVE is True
    assert REBUILD_AFTER_REJECTED_WAVE is True
    assert LIVE_REBUILD_REQUIRED is True
    assert TWO_EPOCH_UNCHANGED_REQUIRED is True
    profile = fixed_point_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert "spar_narrow" in EXISTING_ADAPTER_AUTHORITIES
    assert "FixedPointStore" in FORBIDDEN_FIXED_POINT_NAMES


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "FixedPointRemodularizationController" in names
    assert "FixedPointReceipt" in names
    assert "ComposedRoot" in names
    assert "FixedPointStore" not in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "FixedPointRemodularizationController" in exports
    assert "compose_current_roots" in exports
    assert "rebuild_after_wave" in exports
    assert "run_fixed_point_controller" in exports
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "federation.fixed_point" not in source
    assert "from .federation" not in source


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_accepted_wave_rebuilds_all_slices_and_continues_first_epoch() -> None:
    receipt = run_fixed_point_controller(_evidence())
    assert receipt.status == ControllerStatus.CONTINUE.value
    assert receipt.nominated is False
    assert receipt.accepted is False
    assert receipt.can_authorize_completion is False
    assert receipt.can_accept_fixed_point is False
    iteration = receipt.iteration
    assert iteration.wave_disposition.outcome == WaveOutcome.ACCEPTED.value
    assert set(iteration.composed_root.slices.as_mapping()) == set(REBUILD_SLICE_ORDER)
    assert iteration.composed_root.merge_receipt_cids == (_cid("merge"),)
    assert receipt.analyzer_id == ANALYZER_ID
    encoded = encode_canonical_receipt(receipt)
    restored = decode_canonical_receipt(encoded)
    assert restored == receipt
    assert restored.receipt_cid == receipt.receipt_cid


def test_rejected_wave_still_rebuilds_and_retains_negative_evidence() -> None:
    receipt = run_fixed_point_controller(
        _evidence(
            wave=_wave(status="rejected"),
            validation=_validation(status="rejected"),
            merge_receipts=[],
        )
    )
    assert receipt.status == ControllerStatus.CONTINUE.value
    assert receipt.iteration.wave_disposition.outcome == WaveOutcome.REJECTED.value
    assert _cid("wave") in receipt.iteration.negative_evidence_cids
    assert _cid("validation") in receipt.iteration.negative_evidence_cids
    assert receipt.accepted is False


def test_two_unchanged_live_epochs_nominate_but_do_not_accept() -> None:
    first = run_fixed_point_controller(
        _evidence(remaining_mandatory_task_cids=[], rebuild=_slices("stable"))
    )
    second = run_fixed_point_controller(
        _evidence(
            remaining_mandatory_task_cids=[],
            rebuild=_slices("stable"),
            prior_composed_root_cid=first.iteration.composed_root.composed_root_cid,
            iteration=2,
        )
    )
    assert first.status == ControllerStatus.CONTINUE.value
    assert second.status == ControllerStatus.NOMINATED_FIXED_POINT.value
    assert second.nominated is True
    assert second.accepted is False
    assert second.iteration.prior_composed_root_cid == (
        second.iteration.composed_root.composed_root_cid
    )
    controller = FixedPointRemodularizationController()
    again = controller.run(
        _evidence(
            remaining_mandatory_task_cids=[],
            rebuild=_slices("stable"),
            prior_composed_root_cid=first.iteration.composed_root.composed_root_cid,
            iteration=2,
        )
    )
    assert again.receipt_cid == second.receipt_cid


def test_remaining_work_or_pending_merge_blocks_nominated_fixed_point() -> None:
    first = run_fixed_point_controller(
        _evidence(remaining_mandatory_task_cids=[], rebuild=_slices("stable"))
    )
    remaining = run_fixed_point_controller(
        _evidence(
            rebuild=_slices("stable"),
            remaining_mandatory_task_cids=[_cid("task:left")],
            prior_composed_root_cid=first.iteration.composed_root.composed_root_cid,
            iteration=2,
        )
    )
    assert remaining.status == ControllerStatus.CONTINUE.value
    pending = run_fixed_point_controller(
        _evidence(
            remaining_mandatory_task_cids=[],
            pending_merge_cids=[_cid("merge:pending")],
            rebuild=_slices("stable"),
            prior_composed_root_cid=first.iteration.composed_root.composed_root_cid,
            iteration=2,
        )
    )
    assert pending.status == ControllerStatus.CONTINUE.value


def test_changed_composed_root_continues() -> None:
    first = run_fixed_point_controller(
        _evidence(remaining_mandatory_task_cids=[], rebuild=_slices("epoch-1"))
    )
    second = run_fixed_point_controller(
        _evidence(
            remaining_mandatory_task_cids=[],
            rebuild=_slices("epoch-2"),
            prior_composed_root_cid=first.iteration.composed_root.composed_root_cid,
            iteration=2,
        )
    )
    assert second.status == ControllerStatus.CONTINUE.value
    assert (
        second.iteration.composed_root.composed_root_cid
        != first.iteration.composed_root.composed_root_cid
    )


def test_accepted_wave_without_merge_receipt_fails_closed() -> None:
    with pytest.raises(FixedPointError, match="merge receipt"):
        run_fixed_point_controller(_evidence(merge_receipts=[]))


def test_missing_rebuild_slice_fails_closed() -> None:
    rebuild = _slices()
    rebuild.pop("frontier")
    with pytest.raises(FixedPointError, match="every slice"):
        run_fixed_point_controller(_evidence(rebuild=rebuild))
    with pytest.raises(FixedPointError, match="live rebuild"):
        run_fixed_point_controller(_evidence(rebuild=None))


def test_boolean_prebuilt_and_federation_store_cannot_admit_fixed_point() -> None:
    with pytest.raises(FixedPointError, match="boolean or prebuilt"):
        run_fixed_point_controller(_evidence(fixed_point=True))
    with pytest.raises(FixedPointError, match="boolean or prebuilt"):
        run_fixed_point_controller(_evidence(prebuilt_fixed_point=True))
    with pytest.raises(FixedPointError, match="FixedPointStore"):
        run_fixed_point_controller(_evidence(federation_store={"ok": True}))
    with pytest.raises(FixedPointError, match="cannot admit"):
        run_fixed_point_controller(_evidence(vector_candidate={"score": 1}))


def test_worker_cannot_self_approve_receipt() -> None:
    receipt = run_fixed_point_controller(_evidence())
    payload = receipt.to_dict()
    payload["accepted"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(FixedPointError, match="self-approve"):
        FixedPointReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(FixedPointError, match="can_authorize_completion"):
        FixedPointReceipt.from_dict(payload)


def test_unsupported_validation_is_typed_terminal_never_success() -> None:
    receipt = run_fixed_point_controller(
        _evidence(
            wave=_wave(status="applied"),
            validation=_validation(status="unsupported"),
            merge_receipts=[],
        )
    )
    assert receipt.status == ControllerStatus.TYPED_TERMINAL.value
    assert receipt.iteration.terminal is not None
    assert receipt.iteration.terminal.kind == TerminalKind.UNSUPPORTED.value
    assert receipt.accepted is False
    assert receipt.nominated is False


def test_explicit_human_review_terminal_stops_without_completion() -> None:
    receipt = run_fixed_point_controller(
        _evidence(
            terminal={
                "kind": TerminalKind.HUMAN_REVIEW.value,
                "reason": "public compatibility requires review",
            }
        )
    )
    assert receipt.status == ControllerStatus.TYPED_TERMINAL.value
    assert receipt.iteration.terminal.kind == TerminalKind.HUMAN_REVIEW.value
    assert receipt.accepted is False


def test_max_iterations_is_typed_terminal() -> None:
    first = run_fixed_point_controller(
        _evidence(remaining_mandatory_task_cids=[], rebuild=_slices("moving-1"))
    )
    receipt = run_fixed_point_controller(
        _evidence(
            remaining_mandatory_task_cids=[],
            rebuild=_slices("moving-2"),
            prior_composed_root_cid=first.iteration.composed_root.composed_root_cid,
            iteration=9,
            max_iterations=8,
        )
    )
    assert receipt.status == ControllerStatus.TYPED_TERMINAL.value
    assert receipt.iteration.terminal.kind == TerminalKind.MAX_ITERATIONS.value


def test_dry_run_is_deterministic_and_never_mutates() -> None:
    payload = _evidence()
    first = dry_run_fixed_point(payload)
    second = FixedPointRemodularizationController().dry_run(payload)
    assert first.receipt_cid == second.receipt_cid
    assert DRY_RUN_MUTATES is False
    with pytest.raises(FixedPointError, match="cannot mutate"):
        run_fixed_point_controller(payload, mutate=True)
    with pytest.raises(FixedPointError, match="cannot mutate"):
        run_fixed_point_controller(_evidence(mutate=True))


def test_identity_excludes_observational_fields() -> None:
    receipt = run_fixed_point_controller(_evidence())
    encoded = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(encoded))
    encoded["timestamp"] = "now"
    with pytest.raises(FixedPointError, match="observational"):
        FixedPointReceipt.from_dict(encoded)
    dirty = receipt.iteration.composed_root.slices.to_dict()
    dirty["model_output"] = "guess"
    with pytest.raises(FixedPointError, match="observational"):
        RebuildSliceSet.from_dict(dirty)


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(FixedPointError, match="tree_id"):
        run_fixed_point_controller(_evidence(tree_id=OTHER_TREE))
    receipt = run_fixed_point_controller(_evidence())
    payload = receipt.to_dict()
    payload["tree_id"] = OTHER_TREE
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(FixedPointError, match="tree_id"):
        FixedPointReceipt.from_dict(payload)


def test_compose_current_roots_binds_merge_receipts() -> None:
    slices = RebuildSliceSet.from_mapping(_slices("stable"))
    merge = MergeReceipt(
        merge_cid=_cid("merge"),
        wave_cid=_cid("wave"),
        tree_id=TREE_ID,
        disposition="accepted",
        validation_cid=_cid("validation"),
    )
    composed = compose_current_roots(
        tree_id=TREE_ID,
        rebuild=slices,
        merge_receipts=[merge],
        wave_cid=_cid("wave"),
        wave_outcome=WaveOutcome.ACCEPTED.value,
    )
    assert composed.merge_receipt_cids == (_cid("merge"),)
    assert composed.slices.slice_set_cid == slices.slice_set_cid
    restored = ComposedRoot.from_dict(composed.to_dict())
    assert restored == composed
    disposition = WaveDisposition(
        wave_cid=_cid("wave"),
        outcome=WaveOutcome.ACCEPTED.value,
        wave_status="applied",
        validation_status="validated",
        validation_cid=_cid("validation"),
        packet_cids=(_cid("packet"),),
    )
    restored_disp = WaveDisposition.from_dict(disposition.to_dict())
    assert restored_disp == disposition


def test_mixed_merge_dispositions_fail_closed() -> None:
    with pytest.raises(FixedPointError, match="mix accepted and rejected"):
        run_fixed_point_controller(
            _evidence(
                merge_receipts=[
                    _merge(label="a", disposition="accepted"),
                    _merge(label="b", disposition="rejected"),
                ]
            )
        )


def test_iteration_round_trip_and_controller_methods() -> None:
    controller = FixedPointRemodularizationController()
    rebuilt = controller.rebuild(_evidence())
    restored = FixedPointIteration.from_dict(rebuilt.to_dict())
    assert restored == rebuilt
    terminal = TypedTerminal(
        kind=TerminalKind.CAPABILITY_UNAVAILABLE.value,
        reason="required prover is unavailable",
    )
    restored_terminal = TypedTerminal.from_dict(terminal.to_dict())
    assert restored_terminal == terminal
    assert controller.interface == FIXED_POINT_REMODULARIZATION_CONTROLLER_INTERFACE
