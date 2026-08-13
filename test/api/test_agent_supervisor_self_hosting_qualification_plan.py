"""Fail-closed invariants for the operator-authored qualification plan."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import goal_graph, parse_goal_heap
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    external_authority_goal_fence,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "docs/architecture/SELF_HOSTING_QUALIFICATION_PLAN.md"
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.objectives.md"
)
ACTIVE_TODO_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.todo.md"
)
V1_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v1_history.todo.md"
)
V2_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v2_history.todo.md"
)
V3_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v3_history.todo.md"
)
V4_HISTORY_PATH = (
    REPO_ROOT / "docs/architecture/self_hosting_qualification.v4_history.todo.md"
)


def _goals():
    return parse_goal_heap(OBJECTIVE_PATH.read_text(encoding="utf-8"))


def _descendant_closure(goals, seed_goal_ids: set[str]) -> set[str]:
    closure = set(seed_goal_ids)
    while True:
        expanded = closure | {
            goal.goal_id
            for goal in goals
            if closure.intersection(goal.parent_goal_ids)
        }
        if expanded == closure:
            return closure
        closure = expanded


def test_bootstrap_context_envelope_preserves_the_input_allowance() -> None:
    plan = PLAN_PATH.read_text(encoding="utf-8")
    normalized_plan = " ".join(plan.split())

    assert "model_context_window=49152" in plan
    assert "IPFS_ACCELERATE_AGENT_CODEX_CONTEXT_WINDOW=49152" in plan
    assert "--context-budget-tokens 24576" in plan
    assert "model_context_window=24576" not in plan
    assert "IPFS_ACCELERATE_AGENT_CODEX_CONTEXT_WINDOW=24576" not in plan
    assert 49_152 - 16_384 - 8_192 == 24_576
    assert "operator preflight/detective controls" in normalized_plan


def test_plan_has_one_closed_combined_goal_dag() -> None:
    goals = _goals()
    goal_ids = [goal.goal_id for goal in goals]
    goal_id_set = set(goal_ids)
    hierarchy = goal_graph(goals)

    assert len(goals) == 48
    assert len(goal_id_set) == len(goal_ids)
    assert hierarchy["roots"] == ["SHQ-G000"]
    assert all(
        edge["from"] in goal_id_set and edge["to"] in goal_id_set
        for edge in hierarchy["edges"]
    )

    # Parent and explicit dependency relations are both prerequisites.  Check
    # the union so a cycle split across the two native fields cannot hide from
    # the hierarchy-only projection.
    prerequisites = {
        goal.goal_id: set(goal.parent_goal_ids) | set(goal.dependencies)
        for goal in goals
    }
    assert all(
        prerequisite in goal_id_set
        for values in prerequisites.values()
        for prerequisite in values
    )
    assert all(goal_id not in values for goal_id, values in prerequisites.items())

    dependents: dict[str, set[str]] = {goal_id: set() for goal_id in goal_ids}
    remaining = {
        goal_id: set(values) for goal_id, values in prerequisites.items()
    }
    for goal_id, values in remaining.items():
        for prerequisite in values:
            dependents[prerequisite].add(goal_id)
    ready = sorted(goal_id for goal_id, values in remaining.items() if not values)
    visited: list[str] = []
    while ready:
        goal_id = ready.pop(0)
        visited.append(goal_id)
        for dependent in sorted(dependents[goal_id]):
            remaining[dependent].discard(goal_id)
            if not remaining[dependent] and dependent not in visited and dependent not in ready:
                ready.append(dependent)
        ready.sort()

    assert set(visited) == goal_id_set, {
        goal_id: sorted(values)
        for goal_id, values in remaining.items()
        if values
    }


def test_external_admission_and_preregistration_gates_fail_closed() -> None:
    goals = _goals()

    declared_external, blocked = external_authority_goal_fence(
        goals,
        trust_recorded_completion=False,
    )
    expected_external = {"SHQ-G010", "SHQ-G072"}
    expected_blocked = _descendant_closure(goals, expected_external)

    assert declared_external == expected_external
    assert blocked == expected_blocked
    assert external_authority_goal_fence(goals)[1] == expected_blocked
    assert "SHQ-G006" not in blocked
    assert "SHQ-G007" not in blocked
    assert _descendant_closure(goals, {"SHQ-G072"}) == {
        "SHQ-G072",
        "SHQ-G073",
        "SHQ-G074",
        "SHQ-G075",
        "SHQ-G076",
    }


def test_work_units_target_and_repository_ownership_are_explicit() -> None:
    source = OBJECTIVE_PATH.read_text(encoding="utf-8")
    normalized_source = " ".join(source.split())
    goals = _goals()
    by_id = {goal.goal_id: goal for goal in goals}
    work_goals = [goal for goal in goals if goal.required_evidence]
    external_work_goals = [
        goal for goal in work_goals if goal.requires_external_completion
    ]
    local_work_goals = [
        goal for goal in work_goals if not goal.requires_external_completion
    ]

    assert "endomorphosis/ipfs_kit_py:ipfs_kit_py/core/wal" in source
    assert "`core.operation_contracts` is a read-only dependency" in normalized_source
    assert len(work_goals) == 40
    assert len(local_work_goals) == 38
    assert {goal.goal_id for goal in external_work_goals} == {
        "SHQ-G010",
        "SHQ-G072",
    }
    assert all(goal.fields.get("gap_task") for goal in work_goals)
    assert all(goal.predicted_files for goal in work_goals)
    assert all(goal.validation_commands for goal in work_goals)

    # These interfaces are the ownership seams: datasets owns task meaning and
    # evaluation, kit owns durable evidence/release state, MCP++ owns only the
    # wire schema, and accelerate owns orchestration/runtime behavior.
    assert "SelfHostingTaskCorpus" in by_id["SHQ-G032"].fields["interfaces"]
    assert "compare_task_outcomes" in by_id["SHQ-G036"].fields["interfaces"]
    assert "QualificationArtifactStore" in by_id["SHQ-G041"].fields["interfaces"]
    assert "create_qualification_manifest" in by_id["SHQ-G044"].fields["interfaces"]
    assert "QualificationRuntimePort@1" in by_id["SHQ-G031"].fields["interfaces"]
    assert "GovernedCodingAgentRuntime" in by_id["SHQ-G052"].fields["interfaces"]
    assert "SelfHostingQualificationHarness" in by_id["SHQ-G053"].fields["interfaces"]

    assert by_id["SHQ-G006"].predicted_files == [
        ".gitignore",
        "scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py",
        "test/api/test_agent_supervisor_self_hosting_qualification_prerequisites.py",
    ]
    assert by_id["SHQ-G007"].predicted_files == [
        "artifacts/agent_supervisor/self_hosting_qualification/prerequisite_observation.json"
    ]
    assert by_id["SHQ-G007"].dependencies == ["SHQ-G006"]
    assert by_id["SHQ-G010"].dependencies == ["SHQ-G007"]
    assert by_id["SHQ-G006"].fields["bundle"] != by_id["SHQ-G007"].fields["bundle"]
    assert by_id["SHQ-G006"].fields["bundle"].endswith("bounded-v5")
    assert by_id["SHQ-G007"].fields["bundle"].endswith("bounded-v5")
    assert by_id["SHQ-G006"].fields["parallel_lane"].endswith("bounded-v5")
    assert by_id["SHQ-G007"].fields["parallel_lane"].endswith("bounded-v5")
    assert all(
        command.startswith("/usr/bin/python3.12 ")
        for goal_id in ("SHQ-G006", "SHQ-G007")
        for command in by_id[goal_id].validation_commands
    )

    active_todo = ACTIVE_TODO_PATH.read_text(encoding="utf-8")
    history = V1_HISTORY_PATH.read_text(encoding="utf-8")
    v2_history = V2_HISTORY_PATH.read_text(encoding="utf-8")
    v3_history = V3_HISTORY_PATH.read_text(encoding="utf-8")
    v4_history = V4_HISTORY_PATH.read_text(encoding="utf-8")
    assert "## SHQ-001 " not in active_todo
    assert "## SHQ-001 " in history
    assert "- Status: blocked" in history
    assert "- Completion: superseded:SHQ-002" in history
    assert "- Historical task: true" in history
    assert "- Is schedulable: false" in history
    assert "## SHQ-002 " in v2_history
    assert "## SHQ-003 " in v2_history
    assert "SHQ-002: cancelled/retryable" in v2_history
    assert "superseded by SHQ-004" in v2_history
    assert "SHQ-003: never launched" in v2_history
    assert "superseded by SHQ-005" in v2_history
    assert v2_history.count("- Status: todo") == 2
    assert v2_history.count("- Completion: manual") == 2
    assert v2_history.count("- Is schedulable: true") == 2
    assert "## SHQ-004 " in v3_history
    assert "## SHQ-005 " in v3_history
    assert "SHQ-004: never launched" in v3_history
    assert "SHQ-005: never launched" in v3_history
    assert "Neither task was submitted to coordination" in v3_history
    assert "SHQ-006: rejected/cancelled retryable" in v4_history
    assert "SHQ-007: never launched" in v4_history
    assert "superseded by the bounded-v5 G006 projection" in v4_history
    assert "superseded by the bounded-v5 G007 projection" in v4_history
    v4_blocks = v4_history[v4_history.index("## SHQ-006 ") :]
    assert hashlib.sha256(v4_blocks.encode("utf-8")).hexdigest() == (
        "7c4027e329873364a3742276d5e4582d3a997826c9b1f12a3cffd04ddb783f50"
    )
    # This test is intentionally valid on both sides of the reviewed tracked
    # migration. Before migration the legacy active blocks must be exact; after
    # migration the board is title-only until v5 allocates SHQ-008/009.
    if "## SHQ-006 " in active_todo:
        assert active_todo[active_todo.index("## SHQ-006 ") :] == v4_blocks
    else:
        assert active_todo.strip() == "# Objective Todo" or (
            "## SHQ-008 " in active_todo
            and "## SHQ-009 " in active_todo
            and "## SHQ-006 " not in active_todo
            and "## SHQ-007 " not in active_todo
        )

    datasets_goal_ids = {
        "SHQ-G032",
        "SHQ-G033",
        "SHQ-G034",
        "SHQ-G035",
        "SHQ-G036",
        "SHQ-G037",
        "SHQ-G038",
    }
    assert {
        by_id[goal_id].fields["bundle"] for goal_id in datasets_goal_ids
    } == {"datasets/self-hosting/corpus"}
    assert all(
        by_id[goal_id].fields.get("submodules") == "ipfs_datasets_py"
        for goal_id in datasets_goal_ids
    )
    assert all(
        output.startswith(
            (
                "ipfs_datasets_py/",
                "artifacts/agent_supervisor/self_hosting_qualification/",
            )
        )
        for goal_id in datasets_goal_ids
        for output in by_id[goal_id].predicted_files
    )

    kit_goal_ids = {"SHQ-G041", "SHQ-G042", "SHQ-G043", "SHQ-G044"}
    assert all(
        output.startswith("ipfs_kit_py/")
        for goal_id in kit_goal_ids
        for output in by_id[goal_id].predicted_files
    )


def test_v5_observer_contract_reuses_authorities_and_fails_closed() -> None:
    source = OBJECTIVE_PATH.read_text(encoding="utf-8")
    plan = PLAN_PATH.read_text(encoding="utf-8")
    normalized = " ".join(source.split())

    assert "PrerequisiteTestReceipt@1" not in source
    for authority in (
        "verification.contracts.VerificationIdentityCompiler",
        "verification.process_runner.VerificationProcessRunner",
        "verification.process_runner.VerificationCommand",
        "verification.contracts.TestReceipt@1",
        "verification.contracts.DirectExecutionObservation@1",
        "verification.receipt_cache.VerificationReceiptCache",
    ):
        assert authority in source
    for invariant in (
        "exact non-empty ordered list of ten unique requested systems",
        "absolute root or `..` component",
        "existing parent and symlink",
        "exact module-level definition or assignment",
        "exact package export",
        "trusted in-process construction",
        "injected phase report",
        "present real run result",
        "stdout/stderr CID",
        "outer/tree/gitlink/submodule/tracked-content source identity",
        "canonical repository-relative values",
        "final whole-snapshot two-phase revalidation counterexamples",
        "same-directory exclusive temporary file",
        "No incomplete, stale, partially validated, or source-raced artifact",
    ):
        assert invariant in normalized

    assert 'SHQ_PROJECTION="$SHQ_DATA/projections/v5"' in plan
    assert "self_hosting_qualification.v4_history.todo.md" in plan
    assert "must allocate SHQ-008 and SHQ-009" in plan
    assert "leave SHQ_ACTIVE_TODO title-only" in plan
    assert "SHQ_PYTHON=/usr/bin/python3.12" in plan
    assert "self-hosting-qualification-v5" in plan
    assert 'test ! -e "$SHQ_RUN"' in plan
    assert '--provider-capacity-path "$SHQ_CAPACITY_PATH"' in plan
    assert '--state-root "$SHQ_RUN/state"' in plan
    assert '--coordination-path "$SHQ_RUN/state/coordination.duckdb"' in plan
    assert "bounded-v4/state" not in plan
