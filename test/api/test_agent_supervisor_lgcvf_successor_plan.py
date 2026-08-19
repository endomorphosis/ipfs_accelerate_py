"""Conformance tests for the additive LGCVF successor plan."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    FormalWorkPlan,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    load_supervisor_scheduler_config,
    parse_args,
    supervisor_config_from_args,
)
from scripts.emit_logic_governed_compositional_verification_fabric_plan import (
    IMMEDIATE_PREDECESSOR_PLAN_CID,
    PLAN_REVISION,
    PREDECESSOR_ARCHIVE,
    PROGRAM,
    PROGRAM_ANCESTOR_PLAN_CID,
    TASKS,
    build_plan,
)
from scripts.validate_logic_governed_compositional_verification_fabric_plan import (
    FORMAL_PATH,
    TODO_PATH,
    validate,
)


def test_lgcvf_projections_are_consistent_and_ancestry_bound() -> None:
    report = validate()

    assert report["valid"] is True, report["errors"]
    assert report["immediate_predecessor_plan_cid"] == IMMEDIATE_PREDECESSOR_PLAN_CID
    assert report["program_ancestor_plan_cid"] == PROGRAM_ANCESTOR_PLAN_CID
    assert report["plan_revision"] == PLAN_REVISION == 2
    assert report["board_namespace"] == PROGRAM
    assert report["tasks"] == len(TASKS) == 28


def test_lgcvf_formal_plan_round_trips_with_canonical_identity() -> None:
    payload = json.loads(FORMAL_PATH.read_text(encoding="utf-8"))
    persisted = FormalWorkPlan.from_dict(payload)
    generated = build_plan()

    assert persisted == FormalWorkPlan.from_json(persisted.to_json())
    assert persisted.content_id == generated.content_id
    assert persisted.metadata["predecessor_plan_cid"] == IMMEDIATE_PREDECESSOR_PLAN_CID
    assert persisted.metadata["program_ancestor_plan_cid"] == PROGRAM_ANCESTOR_PLAN_CID
    assert persisted.metadata["plan_revision"] == PLAN_REVISION
    assert persisted.metadata["release_qualified"] is False
    assert persisted.metadata["production_authorized"] is False

    tasks = {task.task_id: task for task in persisted.tasks}
    assert tasks["LGCVF-113"].depends_on == ("LGCVF-111", "LGCVF-112")
    assert tasks["LGCVF-120"].depends_on == (
        "LGCVF-111",
        "LGCVF-112",
        "LGCVF-113",
    )


def test_lgcvf_operator_and_external_gates_remain_unschedulable() -> None:
    text = TODO_PATH.read_text(encoding="utf-8")

    for task_id, blocker in (
        ("LGCVF-121", "blocked_external_authority"),
        ("LGCVF-123", "blocked_manual"),
    ):
        start = text.index(f"## {task_id} ")
        next_heading = text.find("\n## LGCVF-", start + 4)
        block = text[start : next_heading if next_heading >= 0 else len(text)]
        assert "- Status: blocked" in block
        assert "- Is schedulable: false" in block
        assert "- Review only: true" in block
        assert blocker in block


def test_lgcvf_automatic_tasks_do_not_own_protected_control_files() -> None:
    protected = {
        "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md",
        "docs/architecture/logic_governed_compositional_verification_fabric.objectives.md",
        "docs/architecture/logic_governed_compositional_verification_fabric.todo.md",
        "scripts/validate_logic_governed_compositional_verification_fabric_plan.py",
    }
    text = TODO_PATH.read_text(encoding="utf-8")
    headings = list(
        __import__("re").finditer(r"^## (LGCVF-\d{3}) ", text, __import__("re").MULTILINE)
    )
    for index, heading in enumerate(headings):
        task_id = heading.group(1)
        if task_id == "LGCVF-002":
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[heading.start() : end]
        outputs_line = next(line for line in block.splitlines() if line.startswith("- Outputs:"))
        outputs = {item.strip() for item in outputs_line.split(":", 1)[1].split(",")}
        assert outputs.isdisjoint(protected), (task_id, outputs & protected)


def test_lgcvf_plan_files_are_inside_checkout() -> None:
    # Guards against accidentally writing the requested deliverables into one
    # of the unrelated dirty checkouts under ~/lift_coding.
    checkout = Path(__file__).resolve().parents[2]
    assert FORMAL_PATH.is_relative_to(checkout)
    assert TODO_PATH.is_relative_to(checkout)
    assert PREDECESSOR_ARCHIVE.is_relative_to(checkout)
    archived = FormalWorkPlan.from_dict(json.loads(PREDECESSOR_ARCHIVE.read_text()))
    assert archived.content_id == IMMEDIATE_PREDECESSOR_PLAN_CID


def test_lgcvf_scheduler_profile_is_directly_launchable_with_exact_policy() -> None:
    checkout = Path(__file__).resolve().parents[2]
    scheduler_path = (
        checkout
        / "config"
        / "agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json"
    )
    profile = load_supervisor_scheduler_config(
        scheduler_path,
        repo_root=checkout,
    )
    parsed = parse_args(
        ["--scheduler-config", str(scheduler_path), "--once"]
    )
    config = supervisor_config_from_args(parsed, repo_root=checkout)

    assert profile["task_prefix"] == config.task_prefix == "## LGCVF-"
    formal = FormalWorkPlan.from_dict(json.loads(FORMAL_PATH.read_text(encoding="utf-8")))
    assert profile["task_prefix"] == "## " + formal.metadata["task_prefix"]
    assert profile["max_task_attempts"] == config.max_task_attempts == 3
    assert profile["validation_max_workers"] == config.validation_max_workers == 4
    assert set(profile["protected_paths"]) == set(
        config.implementation_protected_paths
    )
    assert len(profile["protected_paths"]) == len(
        config.implementation_protected_paths
    )
    supervisor = PortalImplementationSupervisor(config)
    command = supervisor._build_daemon_command()
    assert command.count("--max-task-attempts") == 1
    assert command[command.index("--max-task-attempts") + 1] == "3"
    assert command.count("--validation-max-workers") == 1
    assert command[command.index("--validation-max-workers") + 1] == "4"
    assert command.count("--implementation-protected-path") == len(
        profile["protected_paths"]
    )
    assert supervisor._managed_daemon_matches_command_line(" ".join(command))
