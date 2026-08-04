"""PDR-092: independently replayed terminal Planner/Doctor release receipt.

The release validator deliberately requires runtime evidence that lives outside
the candidate repository.  These tests therefore build measured, sealed
receipts under pytest's external temporary directory instead of letting the
release module attest to itself.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.validation import (
    planner_doctor_release as release,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = (
    _REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "planner_doctor_release.py"
)
_DOC_PATH = (
    _REPO_ROOT / "docs" / "architecture" / "PROOF_DIRECTED_PLANNER_DOCTOR_RELEASE.md"
)

REQUIRED_AST_SYMBOLS = {
    "PlannerDoctorReleasePolicy",
    "PlannerDoctorReleaseReceipt",
    "PlannerDoctorReleaseValidator",
    "validate_planner_doctor_release",
    "replay_release_receipt",
    "classify_evidence_disposition",
    "check_child_goal_coverage",
    "check_reject_bad_evidence",
    "check_zero_safety_floors",
    "check_exact_rollback",
    "check_six_lane_supervisor_drain",
}

REQUIRED_CHECKS = {
    "declared_artifacts",
    "protected_anchors",
    "canonical_board",
    "source_artifact_reload",
    "child_goal_coverage",
    "task_vs_objective_completion",
    "reject_bad_evidence",
    "zero_safety_floors",
    "exact_rollback",
    "optional_capabilities",
    "automatic_promotion_gated",
    "six_lane_supervisor_drain",
    "cold_imports",
    "report_only_no_write",
}


@dataclass(frozen=True)
class ExternalReleaseEvidence:
    """Paths to independently produced, exact-current-commit evidence."""

    root: Path
    repository_commit: str
    floor_path: Path
    rollback_path: Path
    lane_paths: tuple[Path, ...]
    drain_path: Path

    def validation_kwargs(self) -> dict[str, Any]:
        return {
            "floor_projection_path": self.floor_path,
            "rollback_receipt_path": self.rollback_path,
            "lane_state_paths": self.lane_paths,
            "drain_receipt_path": self.drain_path,
        }


def _write_json(path: Path, payload: Mapping[str, Any], *, seal: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = release.seal_payload(payload) if seal else dict(payload)
    path.write_text(
        json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _build_external_evidence(
    root: Path,
    *,
    repository_commit: str | None = None,
) -> ExternalReleaseEvidence:
    """Create honest runtime evidence outside the candidate source tree."""

    root = root.resolve()
    assert root != _REPO_ROOT and _REPO_ROOT not in root.parents
    commit = repository_commit or release._current_commit(_REPO_ROOT)
    assert commit

    floor_path = _write_json(
        root / "pdr-090" / "terminal-safety-floors.json",
        {
            "interface": "PlannerDoctorTerminalSafetyFloorReceipt@1",
            "producer_task_id": "PDR-090",
            "evidence_class": "measured",
            "current": True,
            "repository_commit": commit,
            "safety_floors": release._zero_floors(),
            "source_receipt_ids": [
                release.content_identity(
                    {"producer_task_id": "PDR-090", "kind": "measured-safety"}
                )
            ],
        },
        seal=True,
    )

    tree_root = release.content_identity(
        {"repository_commit": commit, "surface": "doctor-tree"}
    )
    forest_root = release.content_identity(
        {"repository_commit": commit, "surface": "doctor-forest"}
    )
    rollback_path = _write_json(
        root / "pdr-052" / "exact-doctor-restore.json",
        {
            "interface": "DoctorExactRestoreProofReceipt@1",
            "producer_task_id": "PDR-052",
            "evidence_class": "measured",
            "current": True,
            "repository_commit": commit,
            "transaction_receipt_id": release.content_identity(
                {"producer_task_id": "PDR-052", "kind": "restore-transaction"}
            ),
            "restore_proof": {
                "expected_tree_cid": tree_root,
                "observed_tree_cid": tree_root,
                "expected_forest_cid": forest_root,
                "observed_forest_cid": forest_root,
                "restored": True,
                "quarantined": False,
                "ref_restored": True,
                "gitlinks_equal": True,
            },
        },
        seal=True,
    )

    terminal_statuses = {task_id: "completed" for task_id in release.EXPECTED_TASK_IDS}
    lane_paths: list[Path] = []
    lane_content_ids: dict[str, str] = {}
    for lane_id in range(release.LANE_COUNT):
        lane_path = _write_json(
            root / f"lane-{lane_id}" / "terminal-state.json",
            {
                "interface": "SupervisorLaneTerminalState@1",
                "lane_id": lane_id,
                "task_count": release.CANONICAL_TASK_COUNT,
                "completed_count": release.CANONICAL_TASK_COUNT,
                "task_statuses": terminal_statuses,
                "ready_count": 0,
                "selectable_ready_count": 0,
                "eligible_ready_count": 0,
                "waiting_count": 0,
                "blocked_count": 0,
                "external_reserved_count": 0,
                "resource_reserved_count": 0,
                "active_task_id": "",
                "implementation_in_progress": False,
                "heartbeat_at": f"2026-08-03T00:00:0{lane_id}Z",
            },
            seal=False,
        )
        lane_paths.append(lane_path)
        lane_content_ids[str(lane_id)] = release.file_content_identity(lane_path)

    drain_path = _write_json(
        root / "multi-supervisor" / "terminal-drain.json",
        {
            "interface": "MultiSupervisorDrainReceipt@1",
            "repository_commit": commit,
            "track_count": release.LANE_COUNT,
            "terminal_quiescent": True,
            "all_trees_fenced": True,
            "interrupted": False,
            "lane_state_content_ids": lane_content_ids,
        },
        seal=True,
    )
    return ExternalReleaseEvidence(
        root=root,
        repository_commit=commit,
        floor_path=floor_path,
        rollback_path=rollback_path,
        lane_paths=tuple(lane_paths),
        drain_path=drain_path,
    )


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> ExternalReleaseEvidence:
    return _build_external_evidence(tmp_path_factory.mktemp("pdr092-external"))


@pytest.fixture(scope="module")
def honest_receipt(
    evidence: ExternalReleaseEvidence,
) -> release.PlannerDoctorReleaseReceipt:
    return release.validate_planner_doctor_release(
        _REPO_ROOT,
        **evidence.validation_kwargs(),
    )


def _patch_current_release_gaps_for_isolated_unit_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate pending authority/clean-tree gaps for unrelated unit contracts."""

    original = release.check_canonical_board

    def canonical_with_test_seals(repo_root: Path | None = None) -> release.CheckResult:
        observed = original(repo_root)
        evidence = dict(observed.evidence)
        # Accept either incomplete boards (historical mid-stream runs) or the
        # terminal completed board.  Isolation still forces a clean PASS body
        # so other unit contracts do not depend on live task-status churn.
        if observed.status is release.CheckStatus.FAIL:
            assert evidence.get("nonterminal_statuses")
        else:
            assert observed.status is release.CheckStatus.PASS
            assert evidence.get("all_tasks_completed") is True
        evidence["errors"] = []
        evidence["all_tasks_completed"] = True
        evidence["nonterminal_statuses"] = {}
        evidence["task_statuses"] = {
            task_id: "completed" for task_id in release.EXPECTED_TASK_IDS
        }
        evidence["verified_manual_completion_seals"] = {
            "test_isolation_only": "not-release-authority"
        }
        return release.CheckResult(
            "canonical_board",
            release.CheckStatus.PASS,
            "canonical structure accepted with isolated test-only scheduler seam",
            evidence,
        )

    monkeypatch.setattr(release, "check_canonical_board", canonical_with_test_seals)
    monkeypatch.setattr(
        release,
        "_verify_external_evidence_authority",
        lambda _payload, *, repo_root, evidence_kind: (
            True,
            "test_only_authenticated_evaluator",
        ),
    )

    def clean_report_only_for_test(
        repo_root: Path | None = None,
        *,
        before_identity: Mapping[str, Any] | None = None,
    ) -> release.CheckResult:
        return release.CheckResult(
            "report_only_no_write",
            release.CheckStatus.PASS,
            "dirty shared development tree isolated by the unit-test seam",
            {
                "mode": "report_only",
                "mutation_authorized": False,
                "completion_authoritative": False,
                "tree_unchanged": True,
                "tree_clean_before": True,
                "tree_clean_after": True,
                "test_isolation_only": True,
            },
        )

    monkeypatch.setattr(
        release,
        "check_report_only_no_write",
        clean_report_only_for_test,
    )

    # The production cold-import receipt binds raw subprocess stdout/stderr,
    # which may contain timestamped host logging.  Keep replay deterministic
    # while this unit probe varies only source artifacts/evidence objects.
    monkeypatch.setattr(
        release,
        "check_cold_imports",
        lambda _root=None: release.CheckResult(
            "cold_imports",
            release.CheckStatus.PASS,
            "deterministic isolated cold-import result",
            {
                "failed": [],
                "fresh_process_per_module": True,
                "optional_providers_not_required": True,
                "test_isolation_only": True,
            },
        ),
    )


def _patch_external_authority_for_structural_unit_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticate fixtures only inside tests of receipt-body validation."""

    monkeypatch.setattr(
        release,
        "_verify_external_evidence_authority",
        lambda _payload, *, repo_root, evidence_kind: (
            True,
            "test_only_authenticated_evaluator",
        ),
    )

@pytest.fixture
def isolated_valid_receipt(
    monkeypatch: pytest.MonkeyPatch,
    evidence: ExternalReleaseEvidence,
) -> release.PlannerDoctorReleaseReceipt:
    _patch_current_release_gaps_for_isolated_unit_receipt(monkeypatch)
    receipt = release.validate_planner_doctor_release(
        _REPO_ROOT,
        **evidence.validation_kwargs(),
    )
    assert receipt.valid, receipt.checks
    return receipt


# ---------------------------------------------------------------------------
# Declared outputs / policy
# ---------------------------------------------------------------------------


def test_declared_outputs_and_ast_interfaces_exist() -> None:
    assert _MODULE_PATH.is_file()
    assert _DOC_PATH.is_file()
    assert Path(__file__).is_file()
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not (REQUIRED_AST_SYMBOLS - names)


def test_interfaces_and_policy_defaults_are_fail_closed() -> None:
    policy = release.default_release_policy()
    assert policy.task_id == "PDR-092"
    assert policy.goal_id == "PDR-G100"
    assert policy.board_namespace == "agent-supervisor-proof-directed-planner-doctor-v1"
    assert policy.default_mode == "report_only"
    assert policy.mutation_authorized is False
    assert policy.completion_authoritative is False
    assert policy.automatic_promotion_enabled is False
    assert policy.doctor_mutation_authorized is False
    assert policy.refill_enabled is False
    assert policy.llm_invocations_allowed is False
    assert policy.remote_model_provider_calls_allowed is False
    assert set(policy.safety_floors) == set(release.SAFETY_FLOOR_KEYS)
    assert all(value == 0 for value in policy.safety_floors.values())
    assert release.RELEASE_POLICY_INTERFACE == "PlannerDoctorReleasePolicy@1"
    assert release.RELEASE_RECEIPT_INTERFACE == "PlannerDoctorReleaseReceipt@1"
    assert release.RELEASE_VALIDATOR_INTERFACE == "PlannerDoctorReleaseValidator@1"
    assert release.CANONICAL_TASK_COUNT == 43
    assert release.CANONICAL_GOAL_COUNT == 11
    assert release.TERMINAL_TASK_ID == "PDR-092"
    assert release.LANE_COUNT == 6
    assert len(release.SAFETY_FLOOR_KEYS) == 20


def test_policy_rejects_unsafe_authority_and_floor_values() -> None:
    for kwargs in (
        {"default_mode": "automatic"},
        {"mutation_authorized": True},
        {"completion_authoritative": True},
        {"automatic_promotion_enabled": True},
        {"doctor_mutation_authorized": True},
        {"refill_enabled": True},
        {"llm_invocations_allowed": True},
        {"remote_model_provider_calls_allowed": True},
        {
            "safety_floors": {
                **release._zero_floors(),
                "false_completion_count": 1,
            }
        },
    ):
        with pytest.raises(release.PlannerDoctorReleaseError):
            release.PlannerDoctorReleasePolicy(**kwargs)


@pytest.mark.parametrize("gate", release.MANDATORY_POLICY_GATES)
def test_policy_rejects_every_disabled_mandatory_gate(gate: str) -> None:
    with pytest.raises(release.PlannerDoctorReleaseError, match="mandatory gates"):
        release.PlannerDoctorReleasePolicy(**{gate: False})


# ---------------------------------------------------------------------------
# Current tree / task-output-derived goal evidence
# ---------------------------------------------------------------------------


def test_declared_artifacts_and_protected_anchors_pass() -> None:
    artifacts = release.check_declared_artifacts(_REPO_ROOT)
    protected = release.check_protected_anchors(_REPO_ROOT)
    assert artifacts.status is release.CheckStatus.PASS, artifacts.detail
    assert protected.status is release.CheckStatus.PASS, protected.detail
    assert all(artifacts.evidence["artifacts"].values())
    assert all(protected.evidence["protected_present"].values())
    assert protected.evidence["release_may_rewrite_protected"] is False


def test_canonical_board_structure_reopens_unverified_dependency_closure() -> None:
    """Structural board invariants for the terminal release gate.

    Mid-board snapshots previously froze a nonterminal task set.  As sealed
    manuals land, that set shrinks.  This test keeps the permanent invariants
    (43/11, unique terminal sink, terminal deps, preimages, seal verification)
    and only requires FAIL while any canonical task is still nonterminal.
    """

    result = release.check_canonical_board(_REPO_ROOT)
    assert result.evidence["canonical_task_count"] == 43
    assert result.evidence["goal_count"] == 11
    assert result.evidence["sinks"] == ["PDR-092"]
    assert result.evidence["terminal_goal"] == "PDR-G100"
    assert result.evidence["terminal_depends_on"] == list(
        release.TERMINAL_DEPENDENCIES
    )
    assert result.evidence["max_lanes"] == 6
    assert len(result.evidence["task_preimages"]) == 43
    assert result.evidence["task_preimage_root"].startswith("sha256:")
    assert "scheduler_validation_error" not in result.evidence

    verified = set(result.evidence["verified_manual_completion_seals"])
    # Baseline policy seals always verify once the authority package is sealed.
    assert {"PDR-002", "PDR-003"} <= verified
    # Later sealed manuals must only appear when their receipts verify; never
    # invent unverified IDs.
    assert verified <= {
        "PDR-002",
        "PDR-003",
        "PDR-060",
        "PDR-072",
        "PDR-082",
        "PDR-091",
        "PDR-092",
    }

    nonterminal = set(result.evidence["nonterminal_statuses"])
    assert nonterminal <= set(release.EXPECTED_TASK_IDS)
    if nonterminal:
        assert result.status is release.CheckStatus.FAIL
        assert result.evidence["all_tasks_completed"] is False
        assert "nonterminal canonical tasks" in result.detail
    else:
        assert result.status is release.CheckStatus.PASS
        assert result.evidence["all_tasks_completed"] is True
        assert "PDR-092" in verified


def test_source_reload_is_derived_from_every_declared_task_output() -> None:
    result = release.check_source_artifact_reload(_REPO_ROOT)
    assert result.status is release.CheckStatus.PASS, result.detail
    tasks = release._parse_task_file(_REPO_ROOT)
    declared_outputs = [
        release._safe_repo_relative_path(path)
        for task in tasks
        for path in task.outputs
    ]
    assert result.evidence["declared_task_output_count"] == len(declared_outputs)
    assert result.evidence["unique_task_output_count"] == len(set(declared_outputs))
    assert set(declared_outputs) <= set(result.evidence["artifacts"])
    assert result.evidence["declaration_errors"] == []
    assert result.evidence["missing"] == []
    assert result.evidence["forged"] == []
    assert result.evidence["reloaded_from_current_tree"] is True
    assert result.evidence["release_module_is_validator_not_source_receipt"] is True
    assert result.evidence["forest_root"].startswith("sha256:")


def test_child_goal_coverage_is_all_producer_output_derived() -> None:
    result = release.check_child_goal_coverage(_REPO_ROOT)
    assert result.status is release.CheckStatus.PASS, result.detail
    assert result.evidence["coverage_is_task_output_derived"] is True
    assert result.evidence["objective_completion_from_task_counts"] is False
    assert result.evidence["covered_count"] == result.evidence["required_count"]
    for goal_id in release.CHILD_GOAL_IDS:
        body = result.evidence["coverage"][goal_id]
        assert body["covered"] is True, goal_id
        assert body["uses_task_count_as_authority"] is False
        assert body["assigned_tasks"] == sorted(body["producing_tasks"])
        assert sorted(body["evidence_tasks"]) == sorted(body["producing_tasks"])
        assert body["artifact_count"] == len(body["artifact_records"])
        assert body["artifact_count"] > 0
        assert body["missing_artifacts"] == []
        assert body["unsafe_artifacts"] == []
        assert body["unbacked_objective_outputs"] == []
        assert body["evidence_root"].startswith("sha256:")
    root = result.evidence["coverage"][release.ROOT_GOAL_ID]
    assert root["covered"] is True
    assert root["children_covered"] is True
    assert set(root["child_evidence_roots"]) == set(release.CHILD_GOAL_IDS)


def test_shared_outputs_preserve_each_producer_provenance() -> None:
    tasks = release._parse_task_file(_REPO_ROOT)
    declared_by_path: dict[str, set[str]] = {}
    for task in tasks:
        for path in task.outputs:
            declared_by_path.setdefault(path, set()).add(task.task_id)
    shared = {
        path: task_ids
        for path, task_ids in declared_by_path.items()
        if len(task_ids) > 1
    }
    assert shared, "the canonical board must exercise shared-output provenance"

    coverage = release.check_child_goal_coverage(_REPO_ROOT)
    assert coverage.status is release.CheckStatus.PASS, coverage.detail
    observed_pairs = {
        (record["path"], record["task_id"])
        for goal_id in release.CHILD_GOAL_IDS
        for record in coverage.evidence["coverage"][goal_id]["artifact_records"]
    }
    for path, task_ids in shared.items():
        assert {(path, task_id) for task_id in task_ids} <= observed_pairs


def test_root_producing_and_output_mutations_break_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_goals = list(release._parse_goals(_REPO_ROOT))
    root_index = next(
        index
        for index, goal in enumerate(original_goals)
        if goal.goal_id == release.ROOT_GOAL_ID
    )
    cases = (
        (
            "producing_tasks",
            "PDR-000",
            "root_producing_declaration_mismatch",
        ),
        (
            "outputs",
            "",
            "root_objective_output_mismatch",
        ),
    )
    for field_name, field_value, expected_error in cases:
        mutated = list(original_goals)
        fields = dict(mutated[root_index].fields)
        fields[field_name] = field_value
        mutated[root_index] = replace(mutated[root_index], fields=fields)
        monkeypatch.setattr(
            release,
            "_parse_goals",
            lambda _root, rows=tuple(mutated): list(rows),
        )
        result = release.check_child_goal_coverage(_REPO_ROOT)
        assert result.status is release.CheckStatus.FAIL, field_name
        assert any(
            expected_error in error for error in result.evidence["errors"]
        ), field_name


def test_extra_task_and_duplicate_goal_break_exact_canonical_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_tasks = tuple(release._parse_task_file(_REPO_ROOT))
    original_goals = tuple(release._parse_goals(_REPO_ROOT))

    extra_task = replace(
        original_tasks[0],
        task_id="PDR-999",
        canonical_task_key="test-extra-task",
        canonical_task_cid="sha256:" + ("9" * 64),
    )
    monkeypatch.setattr(
        release,
        "_parse_task_file",
        lambda _root: (*original_tasks, extra_task),
    )
    extra_result = release.check_child_goal_coverage(_REPO_ROOT)
    assert extra_result.status is release.CheckStatus.FAIL
    assert "canonical_task_set_mismatch" in extra_result.evidence["errors"]

    monkeypatch.setattr(release, "_parse_task_file", lambda _root: original_tasks)
    monkeypatch.setattr(
        release,
        "_parse_goals",
        lambda _root: [*original_goals, original_goals[0]],
    )
    duplicate_goal_result = release.check_child_goal_coverage(_REPO_ROOT)
    assert duplicate_goal_result.status is release.CheckStatus.FAIL
    assert "canonical_goal_set_mismatch" in duplicate_goal_result.evidence["errors"]


def test_duplicate_local_task_output_breaks_goal_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = list(release._parse_task_file(_REPO_ROOT))
    task_index = next(
        index
        for index, task in enumerate(tasks)
        if task.task_id != "PDR-000" and task.outputs
    )
    target = tasks[task_index]
    duplicated_path = target.outputs[0]
    tasks[task_index] = replace(
        target,
        outputs=[duplicated_path, duplicated_path, *target.outputs[1:]],
    )
    monkeypatch.setattr(release, "_parse_task_file", lambda _root: tuple(tasks))
    result = release.check_child_goal_coverage(_REPO_ROOT)
    assert result.status is release.CheckStatus.FAIL
    assert any(
        f"{target.task_id}:duplicate_output:{duplicated_path}" in error
        for error in result.evidence["errors"]
    )


def test_pending_task_status_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = list(release._parse_task_file(_REPO_ROOT))
    baseline_pending = {task.task_id for task in tasks if task.status != "completed"}
    task_index = next(
        index for index, task in enumerate(tasks) if task.status == "completed"
    )
    pending_id = tasks[task_index].task_id
    tasks[task_index] = replace(tasks[task_index], status="pending")
    monkeypatch.setattr(release, "_parse_task_file", lambda _root: tuple(tasks))
    result = release.check_canonical_board(_REPO_ROOT)
    assert result.status is release.CheckStatus.FAIL
    assert result.evidence["all_tasks_completed"] is False
    assert set(result.evidence["nonterminal_statuses"]) == baseline_pending | {
        pending_id
    }
    assert "nonterminal canonical tasks" in result.detail


# ---------------------------------------------------------------------------
# External safety-floor and exact-rollback authority
# ---------------------------------------------------------------------------


def test_measured_zero_safety_floor_body_passes_with_authenticated_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    evidence: ExternalReleaseEvidence,
) -> None:
    _patch_external_authority_for_structural_unit_test(monkeypatch)
    result = release.check_zero_safety_floors(
        repo_root=_REPO_ROOT,
        floor_projection_path=evidence.floor_path,
    )
    assert result.status is release.CheckStatus.PASS, result.detail
    assert result.evidence["metrics_authoritative"] is True
    assert result.evidence["producer_authenticated"] is True
    assert result.evidence["producer_task_id"] == "PDR-090"
    assert result.evidence["repository_commit"] == evidence.repository_commit
    assert result.evidence["source_path"] == str(evidence.floor_path.resolve())
    assert result.evidence["source_content_id"] == release.file_content_identity(
        evidence.floor_path
    )
    assert result.evidence["source_receipt_id"] == _read_json(evidence.floor_path)[
        "receipt_id"
    ]
    assert result.evidence["source_receipt_ids"]
    assert set(result.evidence["floors"]) == set(release.SAFETY_FLOOR_KEYS)
    assert result.evidence["nonzero"] == {}
    assert result.evidence["errors"] == []


def test_default_mapping_only_and_missing_floor_evidence_fail_closed(
    tmp_path: Path,
) -> None:
    default = release.check_zero_safety_floors(repo_root=_REPO_ROOT)
    mapping_only = release.check_zero_safety_floors(
        repo_root=_REPO_ROOT,
        floor_projection=release._zero_floors(),
    )
    missing = release.check_zero_safety_floors(
        repo_root=_REPO_ROOT,
        floor_projection_path=tmp_path / "does-not-exist.json",
    )
    for result in (default, mapping_only, missing):
        assert result.status is release.CheckStatus.FAIL
        assert result.evidence["metrics_authoritative"] is False
    assert mapping_only.evidence["caller_projection_ignored"] == release._zero_floors()
    assert missing.evidence["source_result"] == "missing"


def test_unkeyed_self_hashes_never_authenticate_external_evidence(
    monkeypatch: pytest.MonkeyPatch,
    evidence: ExternalReleaseEvidence,
) -> None:
    floor = release.check_zero_safety_floors(
        repo_root=_REPO_ROOT,
        floor_projection_path=evidence.floor_path,
    )
    rollback = release.check_exact_rollback(
        _REPO_ROOT,
        rollback_receipt_path=evidence.rollback_path,
    )
    for result in (floor, rollback):
        assert result.status is release.CheckStatus.FAIL
        assert result.evidence["integrity_seal_ok"] is True
        assert result.evidence["producer_authenticated"] is False
        assert (
            "authenticated_external_evidence_authority_not_configured"
            in result.evidence["errors"]
        )

    observed_board = release.check_canonical_board(_REPO_ROOT)
    monkeypatch.setattr(
        release,
        "check_canonical_board",
        lambda _root=None: release.CheckResult(
            "canonical_board",
            release.CheckStatus.PASS,
            "board-status authority isolated for drain authentication probe",
            {**dict(observed_board.evidence), "errors": []},
        ),
    )
    drain = release.check_six_lane_supervisor_drain(
        _REPO_ROOT,
        lane_state_paths=evidence.lane_paths,
        drain_receipt_path=evidence.drain_path,
    )
    assert drain.status is release.CheckStatus.FAIL
    assert drain.evidence["integrity_seal_ok"] is True
    assert drain.evidence["producer_authenticated"] is False
    assert (
        "authenticated_external_evidence_authority_not_configured"
        in drain.evidence["errors"]
    )


def test_partial_extra_and_nonzero_floor_evidence_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evidence: ExternalReleaseEvidence,
) -> None:
    _patch_external_authority_for_structural_unit_test(monkeypatch)
    base = _read_json(evidence.floor_path)
    base.pop("receipt_id")
    variants: dict[str, tuple[dict[str, Any], str]] = {}

    partial = json.loads(json.dumps(base))
    partial["safety_floors"].pop(release.SAFETY_FLOOR_KEYS[0])
    variants["partial"] = (partial, "safety_floor_key_set_mismatch")

    extra = json.loads(json.dumps(base))
    extra["safety_floors"]["undeclared_floor"] = 0
    variants["extra"] = (extra, "safety_floor_key_set_mismatch")

    nonzero = json.loads(json.dumps(base))
    nonzero["safety_floors"]["false_completion_count"] = 1
    variants["nonzero"] = (nonzero, "nonzero_safety_floor")

    for name, (payload, expected_error) in variants.items():
        path = _write_json(tmp_path / name / "floors.json", payload, seal=True)
        result = release.check_zero_safety_floors(
            repo_root=_REPO_ROOT,
            floor_projection_path=path,
        )
        assert result.status is release.CheckStatus.FAIL, name
        assert result.evidence["metrics_authoritative"] is False, name
        assert expected_error in result.evidence["errors"], name


def test_measured_exact_doctor_restore_body_passes_with_authenticated_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    evidence: ExternalReleaseEvidence,
) -> None:
    _patch_external_authority_for_structural_unit_test(monkeypatch)
    result = release.check_exact_rollback(
        _REPO_ROOT,
        rollback_receipt_path=evidence.rollback_path,
    )
    assert result.status is release.CheckStatus.PASS, result.detail
    proof = result.evidence["restore_proof"]
    assert result.evidence["rollback_authoritative"] is True
    assert result.evidence["producer_authenticated"] is True
    assert result.evidence["rollback_failure_count"] == 0
    assert result.evidence["producer_task_id"] == "PDR-052"
    assert result.evidence["repository_commit"] == evidence.repository_commit
    assert result.evidence["source_path"] == str(evidence.rollback_path.resolve())
    assert result.evidence["source_content_id"] == release.file_content_identity(
        evidence.rollback_path
    )
    assert result.evidence["transaction_receipt_id"].startswith("sha256:")
    assert result.evidence["roots_match"] is True
    assert result.evidence["before_root"] == result.evidence["restored_root"]
    assert result.evidence["before_forest_root"] == result.evidence[
        "restored_forest_root"
    ]
    assert proof["restored"] is True
    assert proof["quarantined"] is False
    assert proof["ref_restored"] is True
    assert proof["gitlinks_equal"] is True
    assert result.evidence["errors"] == []


def test_fake_and_quarantined_rollback_receipts_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evidence: ExternalReleaseEvidence,
) -> None:
    _patch_external_authority_for_structural_unit_test(monkeypatch)
    base = _read_json(evidence.rollback_path)
    base.pop("receipt_id")

    fake = json.loads(json.dumps(base))
    fake["producer_task_id"] = "PDR-999"
    fake_path = _write_json(tmp_path / "fake" / "rollback.json", fake, seal=True)
    fake_result = release.check_exact_rollback(
        _REPO_ROOT,
        rollback_receipt_path=fake_path,
    )
    assert fake_result.status is release.CheckStatus.FAIL
    assert "untrusted_producer" in fake_result.evidence["errors"]
    assert fake_result.evidence["rollback_authoritative"] is False

    quarantined = json.loads(json.dumps(base))
    quarantined["restore_proof"]["quarantined"] = True
    quarantine_path = _write_json(
        tmp_path / "quarantined" / "rollback.json",
        quarantined,
        seal=True,
    )
    quarantine_result = release.check_exact_rollback(
        _REPO_ROOT,
        rollback_receipt_path=quarantine_path,
    )
    assert quarantine_result.status is release.CheckStatus.FAIL
    assert "restore_quarantined" in quarantine_result.evidence["errors"]
    assert quarantine_result.evidence["rollback_authoritative"] is False


# ---------------------------------------------------------------------------
# Terminal six-lane drain
# ---------------------------------------------------------------------------


def test_six_external_terminal_lanes_are_bound_by_sealed_drain_receipt(
    monkeypatch: pytest.MonkeyPatch,
    evidence: ExternalReleaseEvidence,
) -> None:
    _patch_current_release_gaps_for_isolated_unit_receipt(monkeypatch)
    result = release.check_six_lane_supervisor_drain(
        _REPO_ROOT,
        lane_state_paths=evidence.lane_paths,
        drain_receipt_path=evidence.drain_path,
    )
    assert result.status is release.CheckStatus.PASS, result.detail
    assert result.evidence["producer_authenticated"] is True
    assert result.evidence["lanes"] == 6
    assert result.evidence["terminal_quiescent"] is True
    assert result.evidence["all_trees_fenced"] is True
    assert result.evidence["interrupted"] is False
    assert result.evidence["terminal_task_id"] == "PDR-092"
    assert result.evidence["errors"] == []
    assert set(result.evidence["lane_states"]) == {str(i) for i in range(6)}
    expected_ids = {
        str(lane_id): release.file_content_identity(path)
        for lane_id, path in enumerate(evidence.lane_paths)
    }
    assert result.evidence["lane_state_content_ids"] == expected_ids
    assert result.evidence["drain_receipt_id"] == _read_json(evidence.drain_path)[
        "receipt_id"
    ]
    for row in result.evidence["lane_states"].values():
        assert row["completed_count"] == 43
        assert row["active_task_id"] == ""
        assert row["implementation_in_progress"] is False
        assert row["errors"] == []


def test_missing_and_duplicate_lane_projections_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    evidence: ExternalReleaseEvidence,
) -> None:
    _patch_current_release_gaps_for_isolated_unit_receipt(monkeypatch)
    missing = release.check_six_lane_supervisor_drain(
        _REPO_ROOT,
        lane_state_paths=evidence.lane_paths[:-1],
        drain_receipt_path=evidence.drain_path,
    )
    assert missing.status is release.CheckStatus.FAIL
    assert missing.evidence["lanes"] == 5

    duplicate_paths = (
        evidence.lane_paths[0],
        evidence.lane_paths[0],
        *evidence.lane_paths[1:5],
    )
    duplicate = release.check_six_lane_supervisor_drain(
        _REPO_ROOT,
        lane_state_paths=duplicate_paths,
        drain_receipt_path=evidence.drain_path,
    )
    assert duplicate.status is release.CheckStatus.FAIL
    assert any(
        error.startswith("duplicate_lane:0") for error in duplicate.evidence["errors"]
    )
    assert "lane_set_mismatch" in duplicate.evidence["errors"]


# ---------------------------------------------------------------------------
# Whole receipt / replay / exact-PASS aggregation
# ---------------------------------------------------------------------------


def test_current_tree_release_fails_closed_on_pending_and_untrusted_evidence(
    honest_receipt: release.PlannerDoctorReleaseReceipt,
    evidence: ExternalReleaseEvidence,
) -> None:
    """Honest current-tree validation stays fail-closed without authority.

    After the board reaches 43/43 and seals land, ``canonical_board`` and
    ``task_vs_objective_completion`` may PASS.  External metrics/rollback/drain
    evidence still fails closed until authenticated, and a dirty shared tree
    keeps ``report_only_no_write`` FAIL.  The aggregate receipt remains invalid.
    """

    report = honest_receipt.to_dict()
    assert honest_receipt.valid is False
    assert report["valid"] is False
    assert set(report["checks"]) == REQUIRED_CHECKS

    always_fail = {
        "zero_safety_floors",
        "exact_rollback",
        "six_lane_supervisor_drain",
        "report_only_no_write",
    }
    for name in always_fail:
        assert report["checks"][name]["status"] == "fail", name

    # Board/objective gates pass only once every canonical task is completed
    # with verified seals; otherwise they must remain FAIL.
    board_status = report["checks"]["canonical_board"]["status"]
    objective_status = report["checks"]["task_vs_objective_completion"]["status"]
    if board_status == "pass":
        assert report["checks"]["canonical_board"]["evidence"][
            "all_tasks_completed"
        ] is True
        assert objective_status == "pass"
    else:
        assert board_status == "fail"
        assert objective_status == "fail"

    conditional_fail = {
        "canonical_board",
        "task_vs_objective_completion",
    }
    for name in REQUIRED_CHECKS - always_fail - conditional_fail:
        assert report["checks"][name]["status"] == "pass", name
    assert report["checks"]["zero_safety_floors"]["evidence"][
        "metrics_authoritative"
    ] is False
    assert report["checks"]["exact_rollback"]["evidence"][
        "rollback_authoritative"
    ] is False
    assert report["repository_commit"] == evidence.repository_commit
    assert report["repository_state_root"].startswith("sha256:")
    assert report["forest_root"].startswith("sha256:")
    assert report["task_preimage_root"].startswith("sha256:")
    assert report["child_goal_coverage_root"].startswith("sha256:")
    assert report["mutation_authorized"] is False
    assert report["completion_authoritative"] is False
    assert report["automatic_promotion_enabled"] is False
    assert release.verify_sealed(report)


def test_isolated_all_pass_receipt_and_current_source_replay_are_identity_equivalent(
    isolated_valid_receipt: release.PlannerDoctorReleaseReceipt,
    evidence: ExternalReleaseEvidence,
) -> None:
    assert isolated_valid_receipt.valid is True
    assert all(
        item["status"] == "pass" for item in isolated_valid_receipt.checks.values()
    )
    replay = release.replay_release_receipt(
        isolated_valid_receipt,
        repo_root=_REPO_ROOT,
    )
    assert replay["valid"] is True
    assert replay["identity_ok"] is True
    assert replay["seal_ok"] is True
    assert replay["current_validation_valid"] is True
    assert replay["source_replay_performed"] is True
    assert replay["current_repository_commit"] == evidence.repository_commit
    assert replay["claimed_receipt_id"] == isolated_valid_receipt.receipt_id
    assert replay["recomputed_receipt_id"] == isolated_valid_receipt.receipt_id


def test_stale_source_replay_after_external_evidence_mutation_fails_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evidence: ExternalReleaseEvidence,
) -> None:
    _patch_current_release_gaps_for_isolated_unit_receipt(monkeypatch)
    mutable = _build_external_evidence(
        tmp_path / "mutable-external",
        repository_commit=evidence.repository_commit,
    )
    receipt = release.validate_planner_doctor_release(
        _REPO_ROOT,
        **mutable.validation_kwargs(),
    )
    assert receipt.valid is True
    mutable.floor_path.write_bytes(mutable.floor_path.read_bytes() + b"\n")
    replay = release.replay_release_receipt(receipt, repo_root=_REPO_ROOT)
    assert replay["source_replay_performed"] is True
    assert replay["seal_ok"] is True
    assert replay["current_validation_valid"] is True
    assert replay["identity_ok"] is False
    assert replay["valid"] is False
    assert replay["claimed_receipt_id"] != replay["recomputed_receipt_id"]


def test_forged_release_receipt_fails_closed(
    isolated_valid_receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    forged = isolated_valid_receipt.to_dict()
    forged["forest_root"] = "sha256:" + ("0" * 64)
    assert release.verify_sealed(forged) is False
    replay = release.replay_release_receipt(forged, repo_root=_REPO_ROOT)
    assert replay["seal_ok"] is False
    assert replay["identity_ok"] is False
    assert replay["valid"] is False


def _patch_validation_checks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    exceptional_status: release.CheckStatus,
) -> None:
    check_functions = {
        "check_declared_artifacts": "declared_artifacts",
        "check_protected_anchors": "protected_anchors",
        "check_canonical_board": "canonical_board",
        "check_source_artifact_reload": "source_artifact_reload",
        "check_child_goal_coverage": "child_goal_coverage",
        "check_task_vs_objective_completion": "task_vs_objective_completion",
        "check_reject_bad_evidence": "reject_bad_evidence",
        "check_zero_safety_floors": "zero_safety_floors",
        "check_exact_rollback": "exact_rollback",
        "check_optional_capabilities": "optional_capabilities",
        "check_automatic_promotion_gated": "automatic_promotion_gated",
        "check_six_lane_supervisor_drain": "six_lane_supervisor_drain",
        "check_cold_imports": "cold_imports",
        "check_report_only_no_write": "report_only_no_write",
    }
    for function_name, check_name in check_functions.items():
        status = (
            exceptional_status
            if check_name == "optional_capabilities"
            else release.CheckStatus.PASS
        )

        def result_factory(
            *args: Any,
            _name: str = check_name,
            _status: release.CheckStatus = status,
            **kwargs: Any,
        ) -> release.CheckResult:
            return release.CheckResult(_name, _status, "isolated aggregation probe", {})

        monkeypatch.setattr(release, function_name, result_factory)
    identity = "sha256:" + ("a" * 64)
    monkeypatch.setattr(
        release,
        "_worktree_identity",
        lambda _root: {"identity": identity},
    )


@pytest.mark.parametrize(
    "status",
    (release.CheckStatus.WARN, release.CheckStatus.SKIP),
)
def test_required_warn_or_skip_never_counts_as_release_pass(
    monkeypatch: pytest.MonkeyPatch,
    status: release.CheckStatus,
) -> None:
    _patch_validation_checks(monkeypatch, exceptional_status=status)
    receipt = release.validate_planner_doctor_release(_REPO_ROOT)
    assert receipt.checks["optional_capabilities"]["status"] == status.value
    assert receipt.valid is False


def test_evidence_classifier_rejects_every_required_bad_class() -> None:
    result = release.check_reject_bad_evidence()
    assert result.status is release.CheckStatus.PASS, result.detail
    assert set(release.REJECTED_EVIDENCE_CLASSES) <= set(
        result.evidence["rejected_classes_seen"]
    )
    assert result.evidence["wrongly_admitted"] == []
    assert release.classify_evidence_disposition(
        present=False,
        required=True,
    ) is release.EvidenceDisposition.UNAVAILABLE_REQUIRED
    assert release.classify_evidence_disposition(
        present=True,
        required=True,
        skipped=True,
    ) is release.EvidenceDisposition.SKIPPED
    assert not release.evidence_is_admissible(
        release.EvidenceDisposition.SKIPPED,
        required=True,
    )


def test_optional_capabilities_and_cold_imports_pass_while_dirty_tree_fails_closed(
    honest_receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    optional = honest_receipt.checks["optional_capabilities"]
    assert optional["status"] == "pass"
    assert optional["evidence"]["absence_converted_to_pass"] is False
    assert optional["evidence"]["required_gates_independent_of_optional_presence"] is True
    assert all(
        body["required"] is False and body["counts_as_release_pass"] is False
        for body in optional["evidence"]["capabilities"].values()
    )
    cold = honest_receipt.checks["cold_imports"]
    assert cold["status"] == "pass"
    assert cold["evidence"]["failed"] == []
    assert cold["evidence"]["optional_providers_not_required"] is True
    report_only = honest_receipt.checks["report_only_no_write"]
    assert report_only["status"] == "fail"
    assert report_only["evidence"]["mode"] == "report_only"
    assert report_only["evidence"]["mutation_authorized"] is False
    assert report_only["evidence"]["tree_unchanged"] is True
    assert report_only["evidence"]["tree_clean_before"] is False
    assert report_only["evidence"]["tree_clean_after"] is False


def test_automatic_promotion_remains_gated(
    honest_receipt: release.PlannerDoctorReleaseReceipt,
) -> None:
    result = honest_receipt.checks["automatic_promotion_gated"]
    assert result["status"] == "pass"
    assert result["evidence"]["automatic_enabled"] is False
    assert result["evidence"]["release_grants_automatic"] is False
    assert result["evidence"]["holdout_operator_decision_required"] is True
    assert result["evidence"]["holdout_manifest_present"] is True
    assert result["evidence"]["authority_automatic_promotion_enabled"] is not True
    assert result["evidence"]["rollout_automatic_in_allowed_modes"] is False


def test_seals_policy_binding_and_validator_facade_are_stable(
    evidence: ExternalReleaseEvidence,
) -> None:
    body = {"nested": {"b": 2, "a": 1}, "list": [3, 1, 2]}
    first = release.seal_payload(body)
    second = release.seal_payload(body)
    assert first["receipt_id"] == second["receipt_id"]
    assert release.verify_sealed(first)
    assert (
        release.default_release_policy().policy_binding_id
        == release.default_release_policy().policy_binding_id
    )
    validator = release.PlannerDoctorReleaseValidator(_REPO_ROOT)
    facade = validator.to_dict()
    assert facade["interface"] == "PlannerDoctorReleaseValidator@1"
    assert facade["task_id"] == "PDR-092"
    report = validator.run_all(**evidence.validation_kwargs())
    assert report["validator_interface"] == "PlannerDoctorReleaseValidator@1"
    assert report["valid"] is False
    assert report["repository_commit"] == evidence.repository_commit


def test_release_doc_documents_terminal_gate() -> None:
    text = _DOC_PATH.read_text(encoding="utf-8").casefold()
    for topic in (
        "pdr-092",
        "report-only",
        "child-goal",
        "rollback",
        "safety floor",
        "automatic",
        "holdout",
        "optional",
        "task completion",
        "objective completion",
        "identity",
        "no automatic",
    ):
        assert topic in text, topic
