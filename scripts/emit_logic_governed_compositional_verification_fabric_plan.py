#!/usr/bin/env python3
"""Emit the ancestry-bound LGCVF ``FormalWorkPlan@1`` projection.

This is a construction tool only.  A valid plan proves neither source
correctness nor release qualification, and it never changes task status.
"""

# ruff: noqa: E402

from __future__ import annotations

import json
import subprocess
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

PROGRAM = "logic-governed-compositional-verification-fabric-v1"
ROOT_GOAL = "LGCVF-G000"
PLAN_REVISION = 2
IMMEDIATE_PREDECESSOR_PLAN_CID = (
    "baguqeeraqe65yknsg7gy5vkze76exc3qhe4kn2owecnwa65zg6kaepl7id3q"
)
PROGRAM_ANCESTOR_PLAN_CID = (
    "sha256:651702def0aaa564830ec2fda46531a6dcb07fd834484682e0da18837a09589e"
)
ACCELERATOR_BASE_HEAD = "3f95a908d8220517d8255421ad993609f64fca60"
ACCELERATOR_BASE_TREE = "dbbddee9eaa755bcb69384f73620aebcbf93e561"
DATASETS_BASE_HEAD = "af1d2d76d2cd6332baf8cea50df6b2eb4e988203"
DATASETS_BASE_TREE = "8f44e00a49d6f67c67ee6810e04e4aae4d869af5"
TRACE_BOUND = 128
OUTPUT = (
    REPO_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_governed_compositional_verification_fabric"
    / "formal_work_plan.json"
)
PREDECESSOR_ARCHIVE = (
    OUTPUT.parent
    / "plan_revisions"
    / f"{IMMEDIATE_PREDECESSOR_PLAN_CID}.json"
)


# Statuses describe the evidence snapshot at plan construction time.  The
# operational store owns later transitions and must never infer them from this
# metadata.  ``blocked_*`` is represented operationally by Status: blocked in
# the Markdown board with the precise reason kept separately.
TASKS: tuple[dict[str, object], ...] = (
    {"id": "LGCVF-001", "phase": "LGCVF-G010", "status": "completed", "deps": ()},
    {"id": "LGCVF-002", "phase": "LGCVF-G010", "status": "completed", "deps": ("LGCVF-001",)},
    {"id": "LGCVF-010", "phase": "LGCVF-G020", "status": "completed", "deps": ("LGCVF-002",)},
    {"id": "LGCVF-020", "phase": "LGCVF-G030", "status": "completed", "deps": ("LGCVF-010",)},
    {
        "id": "LGCVF-030",
        "phase": "LGCVF-G040",
        "status": "completed",
        "deps": ("LGCVF-010", "LGCVF-020"),
    },
    {
        "id": "LGCVF-040",
        "phase": "LGCVF-G050",
        "status": "completed",
        "deps": ("LGCVF-020", "LGCVF-030"),
    },
    {
        "id": "LGCVF-050",
        "phase": "LGCVF-G060",
        "status": "completed",
        "deps": ("LGCVF-030", "LGCVF-040"),
    },
    {"id": "LGCVF-051", "phase": "LGCVF-G060", "status": "todo", "deps": ("LGCVF-050",)},
    {"id": "LGCVF-060", "phase": "LGCVF-G070", "status": "todo", "deps": ("LGCVF-050",)},
    {"id": "LGCVF-061", "phase": "LGCVF-G070", "status": "todo", "deps": ("LGCVF-060",)},
    {
        "id": "LGCVF-070",
        "phase": "LGCVF-G080",
        "status": "todo",
        "deps": ("LGCVF-040", "LGCVF-050"),
    },
    {"id": "LGCVF-071", "phase": "LGCVF-G080", "status": "todo", "deps": ("LGCVF-070",)},
    {
        "id": "LGCVF-080",
        "phase": "LGCVF-G090",
        "status": "todo",
        "deps": ("LGCVF-030", "LGCVF-050"),
    },
    {
        "id": "LGCVF-081",
        "phase": "LGCVF-G090",
        "status": "todo",
        "deps": ("LGCVF-060", "LGCVF-080"),
    },
    {
        "id": "LGCVF-090",
        "phase": "LGCVF-G100",
        "status": "todo",
        "deps": ("LGCVF-061", "LGCVF-071", "LGCVF-081"),
    },
    {"id": "LGCVF-091", "phase": "LGCVF-G100", "status": "todo", "deps": ("LGCVF-090",)},
    {
        "id": "LGCVF-100",
        "phase": "LGCVF-G110",
        "status": "todo",
        "deps": ("LGCVF-090", "LGCVF-091"),
    },
    {"id": "LGCVF-101", "phase": "LGCVF-G110", "status": "todo", "deps": ("LGCVF-100",)},
    {"id": "LGCVF-102", "phase": "LGCVF-G110", "status": "todo", "deps": ("LGCVF-100",)},
    {
        "id": "LGCVF-110",
        "phase": "LGCVF-G120",
        "status": "todo",
        "deps": ("LGCVF-051", "LGCVF-061", "LGCVF-071", "LGCVF-081", "LGCVF-101", "LGCVF-102"),
    },
    {"id": "LGCVF-111", "phase": "LGCVF-G120", "status": "todo", "deps": ("LGCVF-110",)},
    {"id": "LGCVF-112", "phase": "LGCVF-G120", "status": "todo", "deps": ("LGCVF-110",)},
    {
        "id": "LGCVF-113",
        "phase": "LGCVF-G120",
        "status": "todo",
        "deps": ("LGCVF-111", "LGCVF-112"),
    },
    {
        "id": "LGCVF-120",
        "phase": "LGCVF-G130",
        "status": "todo",
        "deps": ("LGCVF-111", "LGCVF-112", "LGCVF-113"),
    },
    {
        "id": "LGCVF-121",
        "phase": "LGCVF-G130",
        "status": "blocked_external_authority",
        "deps": ("LGCVF-120",),
    },
    {"id": "LGCVF-122", "phase": "LGCVF-G130", "status": "todo", "deps": ("LGCVF-120",)},
    {
        "id": "LGCVF-123",
        "phase": "LGCVF-G130",
        "status": "blocked_manual",
        "deps": ("LGCVF-121", "LGCVF-122"),
    },
    {
        "id": "LGCVF-124",
        "phase": "LGCVF-G130",
        "status": "todo",
        "deps": ("LGCVF-120", "LGCVF-122"),
    },
)

PHASES: tuple[dict[str, object], ...] = (
    {"id": "LGCVF-G010", "title": "P0 current-tree truth and immutable reconciliation", "deps": ()},
    {
        "id": "LGCVF-G020",
        "title": "P1 canonical compositional contract kernel",
        "deps": ("LGCVF-G010",),
    },
    {
        "id": "LGCVF-G030",
        "title": "P2 conservative abstract interpretation",
        "deps": ("LGCVF-G020",),
    },
    {
        "id": "LGCVF-G040",
        "title": "P3 assume-guarantee composition",
        "deps": ("LGCVF-G020", "LGCVF-G030"),
    },
    {
        "id": "LGCVF-G050",
        "title": "P4 incremental semantic and verification state",
        "deps": ("LGCVF-G030", "LGCVF-G040"),
    },
    {
        "id": "LGCVF-G060",
        "title": "P5 reusable incremental SMT",
        "deps": ("LGCVF-G040", "LGCVF-G050"),
    },
    {"id": "LGCVF-G070", "title": "P6 validated interpolation and CEGAR", "deps": ("LGCVF-G060",)},
    {
        "id": "LGCVF-G080",
        "title": "P7 translation receipts and obligation slicing",
        "deps": ("LGCVF-G050", "LGCVF-G060"),
    },
    {
        "id": "LGCVF-G090",
        "title": "P8 reviewed equality saturation and synthesis",
        "deps": ("LGCVF-G040", "LGCVF-G060"),
    },
    {
        "id": "LGCVF-G100",
        "title": "P9 proof-carrying artifact and context",
        "deps": ("LGCVF-G070", "LGCVF-G080", "LGCVF-G090"),
    },
    {
        "id": "LGCVF-G110",
        "title": "P10 Planner/Doctor integration and deterministic routing",
        "deps": ("LGCVF-G050", "LGCVF-G100"),
    },
    {
        "id": "LGCVF-G120",
        "title": "P11 complete vertical slice and adversarial qualification",
        "deps": (
            "LGCVF-G060",
            "LGCVF-G070",
            "LGCVF-G080",
            "LGCVF-G090",
            "LGCVF-G100",
            "LGCVF-G110",
        ),
    },
    {
        "id": "LGCVF-G130",
        "title": "P12 paired benchmark, release evidence, and successors",
        "deps": ("LGCVF-G120",),
    },
)


def _goal_formula(goal_id: str):
    return atom(ReviewedPredicate.GOAL_SATISFIED, constant(TermSort.GOAL, goal_id))


def build_plan() -> FormalWorkPlan:
    root_formula = _goal_formula(ROOT_GOAL)
    formulas = [root_formula]
    requirements: list[EvidenceRequirement] = [
        EvidenceRequirement(
            requirement_id="evidence:lgcvf-final-report",
            kind=EvidenceRequirementKind.ARTIFACT,
            subject_ids=(ROOT_GOAL, "LGCVF-124"),
            source_scope_ids=(
                "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md",
            ),
            minimum_code_assurance=AssuranceLevel.UNVERIFIED,
        )
    ]
    subgoals: list[Subgoal] = []
    plan_tasks: list[PlanTask] = []

    for phase in PHASES:
        phase_id = str(phase["id"])
        phase_tasks = tuple(str(task["id"]) for task in TASKS if task["phase"] == phase_id)
        formula = TDFOLVocabulary.subgoal_satisfaction(phase_id, TRACE_BOUND)
        formulas.append(formula)
        requirement_id = f"evidence:{phase_id.lower()}"
        requirements.append(
            EvidenceRequirement(
                requirement_id=requirement_id,
                kind=EvidenceRequirementKind.PLAN_CONFORMANCE,
                subject_ids=(phase_id, *phase_tasks),
                source_scope_ids=(
                    "docs/architecture/logic_governed_compositional_verification_fabric.todo.md",
                ),
                minimum_code_assurance=AssuranceLevel.UNVERIFIED,
            )
        )
        subgoals.append(
            Subgoal(
                subgoal_id=phase_id,
                goal_id=ROOT_GOAL,
                parent_id=ROOT_GOAL,
                refinement_mode=RefinementMode.SUFFICIENT,
                satisfaction_formula_id=formula.formula_id,
                depends_on=tuple(str(item) for item in phase["deps"]),
                evidence_requirement_ids=(requirement_id,),
                metadata={"title": str(phase["title"]), "program": PROGRAM},
            )
        )

    for task in TASKS:
        phase_id = str(task["phase"])
        plan_tasks.append(
            PlanTask(
                task_id=str(task["id"]),
                goal_id=ROOT_GOAL,
                subgoal_id=phase_id,
                actor_ids=("actor:lgcvf-implementer",),
                depends_on=tuple(str(item) for item in task["deps"]),
                evidence_requirement_ids=(f"evidence:{phase_id.lower()}",),
                metadata={
                    "board_namespace": PROGRAM,
                    "construction_status": str(task["status"]),
                    "phase": phase_id,
                },
            )
        )

    return FormalWorkPlan(
        vocabulary_profile_id="supervisor-reviewed-dcec-tdfol",
        vocabulary_version=LOGIC_VOCABULARY_VERSION,
        source_ids=(
            "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md",
            "docs/architecture/logic_governed_compositional_verification_fabric.objectives.md",
            "docs/architecture/logic_governed_compositional_verification_fabric.todo.md",
        ),
        repository_tree_id=f"git-tree:{ACCELERATOR_BASE_TREE}",
        trace_bound=TRACE_BOUND,
        actors=(
            Actor(
                actor_id="actor:lgcvf-supervisor",
                kind=ActorKind.SUPERVISOR,
                capabilities=("schedule", "lease", "fence", "cancel", "validate", "rollback"),
            ),
            Actor(
                actor_id="actor:lgcvf-implementer",
                kind=ActorKind.AGENT,
                capabilities=("analyze", "implement", "test", "emit-candidate-evidence"),
            ),
            Actor(
                actor_id="actor:lgcvf-operator",
                kind=ActorKind.HUMAN,
                capabilities=("authorize-release", "authorize-production"),
            ),
        ),
        goals=(
            Goal(
                goal_id=ROOT_GOAL,
                owner_actor_id="actor:lgcvf-supervisor",
                satisfaction_formula_id=root_formula.formula_id,
                evidence_requirement_ids=("evidence:lgcvf-final-report",),
                source_ids=(
                    "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_PLAN.md",
                ),
                metadata={
                    "title": (
                        "Content-addressed deterministic-first compositional "
                        "verification loop"
                    ),
                    "status": "active",
                },
            ),
        ),
        subgoals=tuple(subgoals),
        tasks=tuple(plan_tasks),
        events=(),
        fluents=(),
        preconditions=(),
        effects=(),
        norms=(),
        temporal_constraints=(),
        evidence_requirements=tuple(requirements),
        formulas=tuple(formulas),
        metadata={
            "accelerator_construction_head": ACCELERATOR_BASE_HEAD,
            "accelerator_construction_tree": ACCELERATOR_BASE_TREE,
            "authority_boundary": "datasets-semantic_accelerator-operational",
            "board_namespace": PROGRAM,
            "datasets_construction_head": DATASETS_BASE_HEAD,
            "datasets_construction_tree": DATASETS_BASE_TREE,
            "immediate_predecessor_plan_cid": IMMEDIATE_PREDECESSOR_PLAN_CID,
            "plan_revision": PLAN_REVISION,
            "predecessor_board_namespace": PROGRAM,
            "predecessor_plan_cid": IMMEDIATE_PREDECESSOR_PLAN_CID,
            "program_ancestor_board_namespace": (
                "logic-governed-semantic-work-fabric-actual-v1"
            ),
            "program_ancestor_plan_cid": PROGRAM_ANCESTOR_PLAN_CID,
            "production_authorized": False,
            "program": "logic-governed-compositional-verification-fabric",
            "release_qualified": False,
            "task_prefix": "LGCVF-",
        },
    )


def _archive_immediate_predecessor(*, replacement: FormalWorkPlan) -> None:
    """Preserve revision 1 exactly before replacing the active projection."""

    predecessor_bytes: bytes | None = None
    if OUTPUT.is_file():
        try:
            current = FormalWorkPlan.from_dict(
                json.loads(OUTPUT.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("existing formal plan cannot be reconstructed") from exc
        if current.content_id == IMMEDIATE_PREDECESSOR_PLAN_CID:
            predecessor_bytes = OUTPUT.read_bytes()
        elif current.content_id != replacement.content_id:
            # Revision 2 may be regenerated while it is still an uncommitted
            # construction overlay, but never after revision 2 itself enters
            # Git history.  Bind that exception to the exact predecessor blob
            # at HEAD; this keeps later reruns append-only.
            completed = subprocess.run(
                [
                    "/usr/bin/git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "show",
                    f"HEAD:{OUTPUT.relative_to(REPO_ROOT).as_posix()}",
                ],
                cwd=REPO_ROOT,
                env={
                    "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_OPTIONAL_LOCKS": "0",
                    "LANG": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                },
                capture_output=True,
                check=False,
                timeout=30,
            )
            try:
                head = FormalWorkPlan.from_dict(json.loads(completed.stdout))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "existing formal plan is neither the immediate predecessor nor this revision"
                ) from exc
            if (
                completed.returncode != 0
                or head.content_id != IMMEDIATE_PREDECESSOR_PLAN_CID
                or current.metadata.get("plan_revision") != PLAN_REVISION
                or current.metadata.get("immediate_predecessor_plan_cid")
                != IMMEDIATE_PREDECESSOR_PLAN_CID
            ):
                raise RuntimeError(
                    "existing formal plan is neither the immediate predecessor nor this revision"
                )

    if PREDECESSOR_ARCHIVE.is_file():
        try:
            archived = FormalWorkPlan.from_dict(
                json.loads(PREDECESSOR_ARCHIVE.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("archived predecessor cannot be reconstructed") from exc
        if archived.content_id != IMMEDIATE_PREDECESSOR_PLAN_CID:
            raise RuntimeError("archived predecessor identity differs")
        if predecessor_bytes is not None and PREDECESSOR_ARCHIVE.read_bytes() != predecessor_bytes:
            raise RuntimeError("archived predecessor bytes differ from revision 1")
        return

    if predecessor_bytes is None:
        raise RuntimeError("immediate predecessor is unavailable for archival")
    PREDECESSOR_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    PREDECESSOR_ARCHIVE.write_bytes(predecessor_bytes)


def main() -> int:
    plan = build_plan()
    _archive_immediate_predecessor(replacement=plan)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "content_id": plan.content_id,
                "goals": len(plan.goals),
                "plan_revision": PLAN_REVISION,
                "predecessor_archive": str(PREDECESSOR_ARCHIVE),
                "path": str(OUTPUT),
                "schema": FormalWorkPlan.SCHEMA,
                "subgoals": len(plan.subgoals),
                "tasks": len(plan.tasks),
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
