from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.control.manual_completion_seal import (
    verify_manual_completion_seal,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    dependency_satisfied_references,
    parse_task_file,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    load_supervisor_scheduler_config,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_PATH = (
    ROOT / "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json"
)
OBJECTIVES_PATH = (
    ROOT
    / "docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md"
)
AUTHORITY_PATH = (
    ROOT / "config/agent_supervisor_planner_doctor_authority_policy.json"
)
BENCHMARK_PATH = ROOT / "config/agent_supervisor_planner_doctor_benchmark.json"
MANIFEST_PATH = (
    ROOT / "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_pdr_g010_foundation_is_sealed_and_unblocks_exactly_the_next_wave() -> None:
    scheduler = _load(SCHEDULER_PATH)
    tasks = parse_task_file(
        ROOT / scheduler["taskboard_path"],
        scheduler["task_prefix"],
    )
    task_by_id = {task.task_id: task for task in tasks}
    completed = {task.task_id for task in tasks if task.status == "completed"}
    foundation_task_ids = {"PDR-000", "PDR-001", "PDR-002", "PDR-003"}

    assert len(tasks) == 43
    assert len(task_by_id) == len(tasks)
    assert foundation_task_ids <= completed
    assert {
        task_by_id[task_id].completion for task_id in ("PDR-002", "PDR-003")
    } == {"manual"}

    foundation_outputs = {
        task_id: set(task_by_id[task_id].outputs)
        for task_id in ("PDR-001", "PDR-002", "PDR-003")
    }
    assert all(
        (ROOT / output).is_file()
        for outputs in foundation_outputs.values()
        for output in outputs
    )

    verified_receipts: dict[str, dict[str, Any]] = {}
    for task_id in ("PDR-002", "PDR-003"):
        seal = scheduler["manual_completion_seals"][task_id]
        receipt = verify_manual_completion_seal(
            seal["receipt_path"],
            repo_root=ROOT,
            task_id=task_id,
            board_namespace=scheduler["board_namespace"],
            schema=seal["schema"],
            interface=seal["interface"],
            policy_revision=seal["policy_revision"],
            expected_receipt_id=seal["expected_receipt_id"],
            artifact_paths=seal["artifact_paths"],
            grant_type=seal["grant_type"],
            grant_action=seal["grant_action"],
            reviewed_base_claims=seal["reviewed_base_claims"],
            grant_claims=seal["grant_claims"],
        )
        assert receipt["decision"] == "sealed"
        assert receipt["receipt_id"] == seal["expected_receipt_id"]
        assert set(scheduler["protected_after_manual_completion"][task_id]) == (
            foundation_outputs[task_id]
        )
        assert set(seal["artifact_paths"].values()) == (
            foundation_outputs[task_id] - {seal["receipt_path"]}
        )
        verified_receipts[task_id] = receipt

    profile = load_supervisor_scheduler_config(SCHEDULER_PATH, repo_root=ROOT)
    assert {"PDR-002", "PDR-003"} <= set(
        profile["activated_protected_task_ids"]
    )
    assert all(
        profile["verified_manual_completion_seals"][task_id]
        == receipt["receipt_id"]
        for task_id, receipt in verified_receipts.items()
    )
    protected = set(profile["protected_paths"])
    assert all(
        output in protected
        for task_id in ("PDR-002", "PDR-003")
        for output in foundation_outputs[task_id]
    )

    authority = _load(AUTHORITY_PATH)
    benchmark = _load(BENCHMARK_PATH)
    manifest = _load(MANIFEST_PATH)
    source = manifest["source_repository"]
    population = benchmark["population"]
    partitions = manifest["partitions"]
    invariants = manifest["partition_invariants"]
    benchmark_claims = verified_receipts["PDR-003"]["reviewed_base"]

    actual_tree = subprocess.check_output(
        ["git", "rev-parse", f"{source['audited_commit']}^{{tree}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert actual_tree == source["audited_tree"]
    assert (population["repository_commit"], population["repository_tree"]) == (
        source["audited_commit"],
        source["audited_tree"],
    )
    assert manifest["partition_policy"]["source_tree"] == source["audited_tree"]
    assert (
        population["development_root_cid"],
        population["heldout_root_cid"],
    ) == (
        partitions["development"]["index_cid"],
        partitions["heldout"]["index_cid"],
    )
    assert population["development_root_cid"] != population["heldout_root_cid"]
    assert invariants["disjoint"] is True
    assert all(
        invariants[name] == 0
        for name in (
            "exact_object_overlap_count",
            "normalized_ast_overlap_count",
            "normalized_contract_overlap_count",
            "provenance_family_overlap_count",
        )
    )
    assert benchmark_claims["development_index_cid"] == (
        population["development_root_cid"]
    )
    assert benchmark_claims["heldout_index_cid"] == population["heldout_root_cid"]

    assert population["corpus_authority"] == "public-conformance-only"
    assert population["promotion_authority"] is False
    assert manifest["exposure_contract"]["promotion_authority"] is False
    assert all(case["promotion_eligible"] is False for case in manifest["cases"])
    assert benchmark["preregistration"]["automatic_promotion_enabled"] is False

    policy_oracle = benchmark["quality_oracle"]
    manifest_oracle = manifest["oracle_contract"]
    assert policy_oracle["producer_task_id"] == manifest_oracle["producer_task_id"] == (
        "PDR-072"
    )
    assert policy_oracle["oracle_handle"] == manifest_oracle["oracle_handle"]
    assert population["external_pdr_072_holdout_and_oracle_required_for_promotion"]
    assert manifest["exposure_contract"][
        "future_external_promotion_evidence_required"
    ]
    assert manifest_oracle["public_conformance_partition_may_promote"] is False
    assert policy_oracle["missing_unsealed_or_incomplete_disposition"] == (
        "reject-promotion"
    )

    assert authority["safety_floors"]
    assert set(authority["safety_floors"].values()) == {0}
    benchmark_floors = benchmark["non_compensable_safety_floors"]
    assert benchmark_floors["comparison"] == "exact-raw-count-equals-zero"
    assert set(benchmark_floors["metrics"].values()) == {0}
    assert scheduler["benchmark"]["safety_floor_maximum"] == 0

    objectives = OBJECTIVES_PATH.read_text(encoding="utf-8")
    root_goal = objectives.split("## PDR-G000 ", 1)[1].split("\n## ", 1)[0]
    foundation_goal = objectives.split("## PDR-G010 ", 1)[1].split(
        "\n## ",
        1,
    )[0]
    assert "- Status: active" in root_goal
    assert "- Status: completed" in foundation_goal

    initial_satisfied = dependency_satisfied_references(
        tasks,
        completed_task_ids=foundation_task_ids,
    )
    initial_frontier = {
        task.task_id
        for task in tasks
        if task.task_id not in foundation_task_ids
        and all(
            dependency in initial_satisfied
            for dependency in task.depends_on
        )
    }
    assert initial_frontier == {"PDR-010", "PDR-012", "PDR-015", "PDR-020"}
