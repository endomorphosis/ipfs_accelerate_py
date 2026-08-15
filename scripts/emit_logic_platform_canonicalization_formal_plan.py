#!/usr/bin/env python3
"""Emit FormalWorkPlan@1 for the logic-platform canonicalization campaign.

This script is a construction tool.  It does not execute work, install
providers, or claim that a valid plan proves any source change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    Actor,
    ActorKind,
    EvidenceRequirement,
    EvidenceRequirementKind,
    FormalWorkPlan,
    Goal,
    PlanTask,
    RefinementMode,
    Subgoal,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_logic_vocabulary import (
    LOGIC_VOCABULARY_VERSION,
    ReviewedPredicate,
    TDFOLVocabulary,
    TermSort,
    atom,
    constant,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
)


ROOT_GOAL = "LPC-G000"
TRACE_BOUND = 64
OUTPUT = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_platform_canonicalization"
    / "formal_work_plan.json"
)


# Subgoal refinements of the root.  Order is the user's required dependency
# spine; later phases depend on earlier ones.
PHASES: list[dict[str, object]] = [
    {
        "id": "LPC-G010",
        "title": "Current-state inventory",
        "depends_on": (),
        "tasks": (
            "LPC-001",
            "LPC-002",
            "LPC-003",
            "LPC-004",
            "LPC-005",
            "LPC-006",
            "LPC-007",
            "LPC-008",
        ),
    },
    {
        "id": "LPC-G020",
        "title": "Canonical logic catalog snapshot",
        "depends_on": ("LPC-G010",),
        "tasks": ("LPC-020", "LPC-021", "LPC-022", "LPC-023"),
    },
    {
        "id": "LPC-G030",
        "title": "Orthogonal status, evidence, authority, boundedness axes",
        "depends_on": ("LPC-G010",),
        "tasks": ("LPC-030", "LPC-031", "LPC-032"),
    },
    {
        "id": "LPC-G040",
        "title": "Typed syntax and formalization new-write path",
        "depends_on": ("LPC-G020", "LPC-G030"),
        "tasks": ("LPC-040", "LPC-041", "LPC-042", "LPC-043", "LPC-044"),
    },
    {
        "id": "LPC-G050",
        "title": "Operation-specific typed provider protocol",
        "depends_on": ("LPC-G020", "LPC-G030"),
        "tasks": ("LPC-050", "LPC-051", "LPC-052"),
    },
    {
        "id": "LPC-G060",
        "title": "verification_api decomposition with compatibility facades",
        "depends_on": ("LPC-G050",),
        "tasks": ("LPC-060", "LPC-061", "LPC-062"),
    },
    {
        "id": "LPC-G070",
        "title": "Canonical proof-plan / tactician model",
        "depends_on": ("LPC-G040", "LPC-G050"),
        "tasks": ("LPC-070", "LPC-071"),
    },
    {
        "id": "LPC-G080",
        "title": "Canonical semantic cache-key and proof repository",
        "depends_on": ("LPC-G030", "LPC-G050"),
        "tasks": ("LPC-080", "LPC-081"),
    },
    {
        "id": "LPC-G090",
        "title": "Replace supervisor hand-maintained semantic maps",
        "depends_on": ("LPC-G020", "LPC-G030"),
        "tasks": ("LPC-090", "LPC-091"),
    },
    {
        "id": "LPC-G100",
        "title": "Package-neutral LogicPlatformManifest handshake",
        "depends_on": ("LPC-G020",),
        "tasks": ("LPC-100",),
    },
    {
        "id": "LPC-G110",
        "title": "SupervisorLogicPlatformClient cutover",
        "depends_on": ("LPC-G050", "LPC-G090", "LPC-G100"),
        "tasks": ("LPC-110", "LPC-111"),
    },
    {
        "id": "LPC-G120",
        "title": "Hammer and datasets-logic adapter cleanup",
        "depends_on": ("LPC-G020", "LPC-G050", "LPC-G090"),
        "tasks": ("LPC-120",),
    },
    {
        "id": "LPC-G130",
        "title": "Python / CLI / MCP operation parity",
        "depends_on": ("LPC-G060", "LPC-G110"),
        "tasks": ("LPC-130",),
    },
    {
        "id": "LPC-G140",
        "title": "Mandatory conformance and parity tests",
        "depends_on": ("LPC-G040", "LPC-G050", "LPC-G080", "LPC-G110"),
        "tasks": ("LPC-140", "LPC-141", "LPC-142"),
    },
    {
        "id": "LPC-G150",
        "title": "Independent packaging and blocking CI",
        "depends_on": ("LPC-G140",),
        "tasks": ("LPC-150", "LPC-151"),
    },
    {
        "id": "LPC-G160",
        "title": "Generated documentation and migration guide",
        "depends_on": ("LPC-G150",),
        "tasks": ("LPC-160",),
    },
    {
        "id": "LPC-G170",
        "title": "Evidence-based final report",
        "depends_on": ("LPC-G160",),
        "tasks": ("LPC-170",),
    },
]


def _goal_formula(goal_id: str):
    return atom(
        ReviewedPredicate.GOAL_SATISFIED,
        constant(TermSort.GOAL, goal_id),
    )


def build_plan() -> FormalWorkPlan:
    formulas = []
    evidence = []
    subgoals = []
    tasks = []

    root_formula = _goal_formula(ROOT_GOAL)
    formulas.append(root_formula)
    evidence.append(
        EvidenceRequirement(
            requirement_id="evidence:lpc-final-report",
            kind=EvidenceRequirementKind.ARTIFACT,
            subject_ids=(ROOT_GOAL, "LPC-170"),
            source_scope_ids=(
                "docs/architecture/LOGIC_PLATFORM_CANONICALIZATION_PLAN.md",
            ),
            minimum_code_assurance=AssuranceLevel.UNVERIFIED,
        )
    )

    for phase in PHASES:
        subgoal_id = str(phase["id"])
        formula = TDFOLVocabulary.subgoal_satisfaction(subgoal_id, TRACE_BOUND)
        formulas.append(formula)
        requirement_id = f"evidence:{subgoal_id.lower()}"
        task_ids = tuple(str(item) for item in phase["tasks"])  # type: ignore[arg-type]
        evidence.append(
            EvidenceRequirement(
                requirement_id=requirement_id,
                kind=EvidenceRequirementKind.PLAN_CONFORMANCE,
                subject_ids=(subgoal_id, *task_ids),
                source_scope_ids=(
                    "docs/architecture/logic_platform_canonicalization.todo.md",
                ),
                minimum_code_assurance=AssuranceLevel.UNVERIFIED,
            )
        )
        subgoals.append(
            Subgoal(
                subgoal_id=subgoal_id,
                goal_id=ROOT_GOAL,
                parent_id=ROOT_GOAL,
                refinement_mode=RefinementMode.SUFFICIENT,
                satisfaction_formula_id=formula.formula_id,
                depends_on=tuple(str(item) for item in phase["depends_on"]),  # type: ignore[arg-type]
                evidence_requirement_ids=(requirement_id,),
                metadata={
                    "title": phase["title"],
                    "schema": "ipfs_accelerate_py/agent-supervisor/formal-plan-subgoal@1",
                },
            )
        )
        previous = ""
        for task_id in task_ids:
            depends_on = (previous,) if previous else ()
            # Inventory tasks and a few explicitly independent pairs stay
            # sibling-parallel; later waves serialize only within a phase
            # when they share a contract surface.
            if subgoal_id == "LPC-G010":
                depends_on = ()
            tasks.append(
                PlanTask(
                    task_id=task_id,
                    goal_id=ROOT_GOAL,
                    subgoal_id=subgoal_id,
                    actor_ids=("actor:implementer",),
                    depends_on=depends_on,
                    evidence_requirement_ids=(requirement_id,),
                    metadata={"phase": subgoal_id},
                )
            )
            previous = task_id

    return FormalWorkPlan(
        vocabulary_profile_id="supervisor-reviewed-dcec-tdfol",
        vocabulary_version=LOGIC_VOCABULARY_VERSION,
        source_ids=(
            "docs/architecture/logic_platform_canonicalization.objectives.md",
            "docs/architecture/logic_platform_canonicalization.todo.md",
            "docs/architecture/LOGIC_PLATFORM_CANONICALIZATION_PLAN.md",
        ),
        repository_tree_id="tree:lift-coding-current-heads",
        trace_bound=TRACE_BOUND,
        actors=(
            Actor(
                actor_id="actor:supervisor",
                kind=ActorKind.SUPERVISOR,
                capabilities=("schedule", "lease", "cancel", "merge"),
            ),
            Actor(
                actor_id="actor:implementer",
                kind=ActorKind.AGENT,
                capabilities=("implement", "inventory", "test"),
            ),
        ),
        goals=(
            Goal(
                goal_id=ROOT_GOAL,
                owner_actor_id="actor:supervisor",
                satisfaction_formula_id=root_formula.formula_id,
                evidence_requirement_ids=("evidence:lpc-final-report",),
                source_ids=(
                    "docs/architecture/LOGIC_PLATFORM_CANONICALIZATION_PLAN.md",
                ),
                metadata={
                    "title": (
                        "ipfs_datasets_py.logic is the semantic authority; "
                        "agent_supervisor owns only execution"
                    )
                },
            ),
        ),
        subgoals=tuple(subgoals),
        tasks=tuple(tasks),
        events=(),
        fluents=(),
        preconditions=(),
        effects=(),
        norms=(),
        temporal_constraints=(),
        evidence_requirements=tuple(evidence),
        formulas=tuple(formulas),
        metadata={
            "program": "logic-platform-canonicalization",
            "task_prefix": "LPC-",
            "reviewed_baseline_datasets": (
                "ac82107e246b30e35a2bbdcf75e01370d22350c6"
            ),
            "reviewed_baseline_accelerate": (
                "485edc0871c55b0e2ef21d83bece9fa12c2c8d84"
            ),
        },
    )


def main() -> int:
    plan = build_plan()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.to_dict()
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": FormalWorkPlan.SCHEMA,
                "valid": True,
                "path": str(OUTPUT),
                "content_id": plan.content_id,
                "goals": len(plan.goals),
                "subgoals": len(plan.subgoals),
                "tasks": len(plan.tasks),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
