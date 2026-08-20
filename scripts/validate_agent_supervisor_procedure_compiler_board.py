#!/usr/bin/env python3
"""Fail-closed validator for the proof-carrying procedure compiler board."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (  # noqa: E402
    ConfiguredBoardError,
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_text,
)

PLAN_PATH = REPO_ROOT / "docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_PROCEDURE_COMPILER_PLAN.md"
OBJECTIVES_PATH = REPO_ROOT / "docs/architecture/agent_supervisor_procedure_compiler.objectives.md"
TODO_PATH = REPO_ROOT / "docs/architecture/agent_supervisor_procedure_compiler.todo.md"
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
INVENTORY_ROOT = REPO_ROOT / "docs/architecture/procedure_compiler_inventory"
BENCHMARK_MANIFEST = REPO_ROOT / "benchmarks/agent_supervisor/procedure_compiler/manifest.json"
RUNTIME_IMAGE_MANIFEST = (
    REPO_ROOT
    / "scripts/ops/agent_supervisor/pcpc_external_runtime_image_v3.manifest.json"
)

PROGRAM = "agent-supervisor-proof-carrying-procedure-compiler-v1"
BRANCH = "codex/proof-carrying-procedure-compiler-v1"
BASE_COMMIT = "bbf7f68799072c2b81f7d96eac91f2df3c4b3952"
BASE_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
TASK_IDS = tuple(f"PCPC-{index:03d}" for index in range(32))
GOAL_IDS = ("PCPC-G000", "PCPC-G010", "PCPC-G020", "PCPC-G030", "PCPC-G040")
P0_COMPLETED = tuple(f"PCPC-{index:03d}" for index in range(9))
INITIAL_READY = ("PCPC-009", "PCPC-011", "PCPC-013")
TERMINAL_TASK = "PCPC-031"
LANE_COUNT = 4

TASK_GROUPS = {
    "PCPC-G010": tuple(f"PCPC-{index:03d}" for index in range(0, 9)),
    "PCPC-G020": tuple(f"PCPC-{index:03d}" for index in range(9, 18)),
    "PCPC-G030": tuple(f"PCPC-{index:03d}" for index in range(18, 28)),
    "PCPC-G040": tuple(f"PCPC-{index:03d}" for index in range(28, 32)),
}

REQUIRED_TASK_FIELDS = (
    "status",
    "completion",
    "is schedulable",
    "review only",
    "priority",
    "track",
    "depends on",
    "goal id",
    "goal",
    "outputs",
    "validation",
    "board namespace",
    "bundle",
    "parallel lane",
    "resource class",
    "resource stage",
    "estimated tokens",
    "token budget",
    "implementation timeout seconds",
    "predicted files",
    "predicted symbols",
    "interfaces",
    "effect class",
    "allow concurrent with",
    "conflict policy",
    "preconditions",
    "effects",
    "evidence subset",
    "symbolic first",
    "llm context budget bytes",
    "proof requirements",
    "acceptance criteria",
    "acceptance",
    "embedding query",
)

REQUIRED_GOAL_FIELDS = (
    "status",
    "review_only",
    "parent",
    "depends_on",
    "fib_priority",
    "track",
    "priority",
    "bundle",
    "parallel_lane",
    "resource_class",
    "goal",
    "evidence",
    "evidence_criteria",
    "evidence_source_policy",
    "outputs",
    "predicted_files",
    "interfaces",
    "validation",
    "acceptance",
    "gap_task",
    "refinement",
    "embedding_query",
    "ast_query",
    "conflict_policy",
)

P0_FILES = (
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/contracts.py",
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/procedure_ir.py",
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/interpreter.py",
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/runtime.py",
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/world_model.py",
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/transition_model.py",
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/trajectory.py",
    "ipfs_accelerate_py/agent_supervisor/procedure_compiler/task_family.py",
    "test/api/procedure_compiler/test_contracts.py",
    "test/api/procedure_compiler/test_procedure_ir.py",
    "test/api/procedure_compiler/test_interpreter.py",
    "test/api/procedure_compiler/test_world_model.py",
    "test/api/procedure_compiler/test_transition_model.py",
    "test/api/procedure_compiler/test_task_family_contracts.py",
    "test/api/procedure_compiler/test_inventory.py",
)

REQUIRED_PLAN_TERMS = (
    "proofcarryingprocedurecompiler",
    "procedureir",
    "deterministic interpreter",
    "repository world model",
    "anti-unification",
    "cegis",
    "typed holes",
    "duckdb",
    "quack",
    "ducklake",
    "non-authoritative",
    "zero unauthorized",
    "cross-repository transfer",
    "amortization",
    "current-tree",
    "rollback",
)

REQUIRED_ARTIFACT_VOCABULARY = (
    "RepositoryWorldState",
    "AbstractRepositoryState",
    "WorldStateDelta",
    "TransitionObservation",
    "TransitionModel",
    "TransitionPrediction",
    "PredictionCalibration",
    "ExecutionTrajectory",
    "TrajectoryStep",
    "TrajectoryOutcome",
    "TrajectoryNormalizationReceipt",
    "TaskFamily",
    "TaskFamilyMembership",
    "TaskFamilyBoundary",
    "TaskFamilyCounterexample",
    "ProcedureSpec",
    "ProcedureVersion",
    "ProcedureParameter",
    "ProcedureLocal",
    "ProcedureStep",
    "ProcedureBranch",
    "ProcedureLoop",
    "ProcedureHole",
    "ProcedureEffect",
    "ProcedureObservation",
    "ProcedurePrecondition",
    "ProcedureInvariant",
    "ProcedurePostcondition",
    "ProcedureRollback",
    "ProcedureFallback",
    "ProcedureResourceEnvelope",
    "ProcedureAuthorityEnvelope",
    "ProcedureValidationPlan",
    "ProcedureCandidate",
    "ProcedureSynthesisPlan",
    "ProcedureSynthesisCounterexample",
    "ProcedureVerificationResult",
    "ProcedureCertificate",
    "ProcedureInvocation",
    "ProcedureInvocationReceipt",
    "ProcedureExecutionTrace",
    "ProcedureOutcome",
    "ProcedureFailure",
    "ProcedureRecoveryPlan",
    "SpecificationCandidate",
    "SpecificationEvidence",
    "SpecificationCounterexample",
    "SpecificationMiningReceipt",
    "InvariantCandidate",
    "InvariantValidationReceipt",
    "NonVacuityReceipt",
    "AntiUnificationPattern",
    "GeneralizationBoundary",
    "GeneralizationCounterexample",
    "ProcedureRegistry",
    "ProcedureRegistryRevision",
    "ProcedurePromotionReceipt",
    "ProcedureRollbackReceipt",
    "ProcedureDeprecationReceipt",
    "ProcedureDriftReport",
    "HoleRequest",
    "HoleCandidate",
    "HoleResolution",
    "HoleValidationReceipt",
    "DistillationCorpus",
    "DistillationExample",
    "DistillationEvaluation",
    "LocalDecisionModelArtifact",
    "GeneratedToolSpec",
    "GeneratedToolCandidate",
    "GeneratedToolCertificate",
    "GeneratedToolInvocationReceipt",
    "ExperimentPlan",
    "ExperimentObservation",
    "ExperimentEvaluation",
    "ProcedureCompilerRunReceipt",
    "ProcedureCompilerReleaseReceipt",
)

ALLOWED_PROCEDURE_OPERATIONS = (
    "READ_STATE",
    "QUERY_AST_INDEX",
    "QUERY_DEPENDENCY_GRAPH",
    "QUERY_SEMANTIC_INDEX",
    "QUERY_RECEIPT_CACHE",
    "SELECT_EVIDENCE",
    "EXPAND_CONTEXT_REFERENCE",
    "CHECK_CAPABILITY",
    "CHECK_POLICY",
    "CHECK_AUTHORITY",
    "CREATE_ISOLATED_WORKTREE",
    "APPLY_APPROVED_PATCH_TEMPLATE",
    "REQUEST_TYPED_MODEL_HOLE",
    "RUN_STATIC_ANALYSIS",
    "RUN_TYPE_CHECK",
    "RUN_SELECTED_TESTS",
    "RUN_FULL_TEST_FALLBACK",
    "RUN_PROOF",
    "RUN_ADVERSARIAL_ASSURANCE",
    "CHECK_DIFF",
    "CHECK_SCOPE",
    "CHECK_POSTCONDITION",
    "PREPARE_MERGE",
    "MERGE_IN_ISOLATED_TRAIN",
    "VERIFY_MERGED_TREE",
    "PERSIST_ARTIFACT",
    "EMIT_RECEIPT",
    "ROLLBACK",
    "ESCALATE",
)

FORBIDDEN_PROCEDURE_OPERATIONS = (
    "ARBITRARY_SHELL",
    "ARBITRARY_PYTHON",
    "ARBITRARY_NETWORK_REQUEST",
    "ARBITRARY_FILESYSTEM_PATH",
    "DISABLE_VALIDATION",
    "MODIFY_AUTHORITY_POLICY",
    "MODIFY_TRUSTED_KEYS",
    "CLAIM_COMPLETION",
)

PROCEDURE_READ_OPERATIONS = (
    "procedures.capabilities",
    "procedures.list",
    "procedures.get",
    "procedures.explain",
    "procedures.match",
    "procedures.registry_status",
    "procedures.task_families",
    "procedures.counterexamples",
    "procedures.drift",
    "procedures.metrics",
    "procedures.shadow_results",
    "procedures.synthesis_status",
    "procedures.world_model_status",
)

PROCEDURE_MUTATION_OPERATIONS = (
    "procedures.synthesize",
    "procedures.evaluate",
    "procedures.promote",
    "procedures.rollback",
    "procedures.revoke",
    "procedures.quarantine",
    "procedures.run_shadow",
    "procedures.cancel",
    "procedures.request_review",
)

REQUIRED_CERTIFICATE_FIELDS = (
    "procedure CID",
    "procedure version",
    "task-family CID",
    "source episode CIDs",
    "specification CIDs",
    "counterexample-set CID",
    "operation-catalog revision",
    "effect-policy revision",
    "authority-policy revision",
    "verification-policy revision",
    "repository families",
    "supported language and framework classes",
    "risk ceiling",
    "proof and test receipts",
    "adversarial-assurance results",
    "held-out evaluation",
    "shadow evaluation",
    "known limitations",
    "issuer",
    "signature",
    "expiry or review horizon",
)

REQUIRED_PLAN_TEST_TERMS = (
    "confirmation",
    "concurrent invocation",
    "idle stability",
    "large-artifact rejection",
)

REQUIRED_NUMERIC_GATE_TERMS = (
    "50% lower median planning tokens",
    "40% lower total model input tokens",
    "60% fewer remote-model calls",
    "70% lower retry tokens",
    "60% of eligible recurring tasks",
    "80% of deterministic repair-family tasks",
    "30% of accepted benchmark work",
    "25% fewer human interventions",
    "100% required-postcondition coverage",
    "100% validation retention",
    "100% correct rejection",
    "zero unsafe cross-repository transfer",
)

REQUIRED_AUTHORITIES = {
    "SemanticCompressionHarness",
    "SemanticCompressionGovernor",
    "AdversarialAssuranceEngine",
    "IncrementalVerificationPlanner",
    "IncrementalProofSealer",
    "AdaptivePlanner",
    "SupervisorControlService",
    "ContextCompiler",
    "ValueOfInformation evidence selection",
    "Delta retry contexts",
    "Provider capacity and route policy",
    "Worktree lease fencing and merge controls",
    "AutonomousMetaController",
    "autonomy package",
    "cognitive scheduler",
    "experience ledger",
    "policy-distillation subsystem",
}
DISPOSITIONS = {"available", "available_with_caveats", "incompatible", "stale", "missing"}


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _safe_relative(value: str) -> bool:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    return bool(text and not path.is_absolute() and ".." not in path.parts and "\x00" not in text)


def _csv(value: object) -> tuple[str, ...]:
    return tuple(part.strip() for part in re.split(r"[,;]", str(value or "")) if part.strip())


def _positive_int(value: object) -> bool:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return False
    return parsed > 0


def _acyclic(graph: Mapping[str, Iterable[str]]) -> tuple[bool, tuple[str, ...]]:
    visiting: set[str] = set()
    visited: set[str] = set()
    trail: list[str] = []

    def visit(node: str) -> bool:
        if node in visiting:
            trail.append(node)
            return False
        if node in visited:
            return True
        visiting.add(node)
        trail.append(node)
        for dependency in graph.get(node, ()):
            if not visit(dependency):
                return False
        trail.pop()
        visiting.remove(node)
        visited.add(node)
        return True

    for node in graph:
        if node not in visited and not visit(node):
            return False, tuple(trail)
    return True, ()


def _transitive_dependencies(
    graph: Mapping[str, Iterable[str]], task_id: str
) -> frozenset[str]:
    result: set[str] = set()
    pending = list(graph.get(task_id, ()))
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(graph.get(dependency, ()))
    return frozenset(result)


def _append(checks: list[dict[str, Any]], errors: list[str], *, name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def _task_shard(task_id: str) -> int:
    return int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16) % LANE_COUNT


def validate_program() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    control_files = (
        PLAN_PATH,
        OBJECTIVES_PATH,
        TODO_PATH,
        CONFIG_PATH,
        INVENTORY_ROOT / "README.md",
        INVENTORY_ROOT / "baseline.json",
        INVENTORY_ROOT / "prerequisites.json",
        INVENTORY_ROOT / "authority_reuse.md",
        BENCHMARK_MANIFEST,
        RUNTIME_IMAGE_MANIFEST,
        Path(__file__).resolve(),
        REPO_ROOT / "scripts/materialize_agent_supervisor_procedure_compiler_program.py",
        REPO_ROOT / "scripts/ops/agent_supervisor/procedure_compiler_program.py",
        *(REPO_ROOT / relative for relative in P0_FILES),
    )
    missing = [path.relative_to(REPO_ROOT).as_posix() for path in control_files if not path.is_file()]
    _append(checks, errors, name="required_files", passed=not missing, detail=missing)
    if any(path in missing for path in (
        PLAN_PATH.relative_to(REPO_ROOT).as_posix(),
        OBJECTIVES_PATH.relative_to(REPO_ROOT).as_posix(),
        TODO_PATH.relative_to(REPO_ROOT).as_posix(),
        CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
    )):
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler-board-validation@1",
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    objective_text = OBJECTIVES_PATH.read_text(encoding="utf-8")
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    normalized_plan = re.sub(r"[^a-z0-9-]+", " ", plan_text.lower())
    whitespace_normalized_plan = " ".join(plan_text.lower().split())
    missing_terms = [term for term in REQUIRED_PLAN_TERMS if term not in normalized_plan]
    _append(checks, errors, name="plan_vocabulary", passed=not missing_terms, detail=missing_terms)
    required_normative_terms = (
        *REQUIRED_ARTIFACT_VOCABULARY,
        *ALLOWED_PROCEDURE_OPERATIONS,
        *FORBIDDEN_PROCEDURE_OPERATIONS,
        *PROCEDURE_READ_OPERATIONS,
        *PROCEDURE_MUTATION_OPERATIONS,
        *REQUIRED_CERTIFICATE_FIELDS,
        *REQUIRED_PLAN_TEST_TERMS,
        *REQUIRED_NUMERIC_GATE_TERMS,
    )
    missing_normative_terms = [
        term for term in required_normative_terms if term.lower() not in whitespace_normalized_plan
    ]
    _append(
        checks,
        errors,
        name="self_contained_normative_vocabulary",
        passed=not missing_normative_terms,
        detail=missing_normative_terms,
    )

    task_headings = re.findall(r"^## (PCPC-\d{3})\b", todo_text, flags=re.MULTILINE)
    goal_headings = re.findall(r"^## (PCPC-G\d{3})\b", objective_text, flags=re.MULTILINE)
    tasks = parse_task_text(todo_text, path=TODO_PATH, task_header_prefix="## PCPC-")
    goals = parse_goal_heap(objective_text)
    _append(
        checks,
        errors,
        name="task_population",
        passed=tuple(task_headings) == TASK_IDS and tuple(task.task_id for task in tasks) == TASK_IDS,
        detail={"expected": list(TASK_IDS), "headings": task_headings, "parsed": [task.task_id for task in tasks]},
    )
    _append(
        checks,
        errors,
        name="goal_population",
        passed=tuple(goal_headings) == GOAL_IDS and tuple(goal.goal_id for goal in goals) == GOAL_IDS,
        detail={"expected": list(GOAL_IDS), "headings": goal_headings, "parsed": [goal.goal_id for goal in goals]},
    )
    _append(checks, errors, name="unique_ids", passed=len(set(task_headings)) == 32 and len(set(goal_headings)) == 5, detail={"task_duplicates": [item for item, count in Counter(task_headings).items() if count > 1], "goal_duplicates": [item for item, count in Counter(goal_headings).items() if count > 1]})

    task_by_id = {task.task_id: task for task in tasks}
    task_field_errors: dict[str, list[str]] = {}
    graph: dict[str, tuple[str, ...]] = {}
    declared_concurrency: dict[str, tuple[str, ...]] = {}
    lane_errors: dict[str, dict[str, object]] = {}
    for task in tasks:
        problems = [f"missing:{field}" for field in REQUIRED_TASK_FIELDS if field not in task.metadata]
        if task.status not in {"todo", "completed"}:
            problems.append(f"status:{task.status}")
        if task.metadata.get("board namespace") != PROGRAM:
            problems.append("board-namespace")
        if task.metadata.get("is schedulable", "").lower() != "true":
            problems.append("schedulable")
        if task.metadata.get("symbolic first", "").lower() != "true":
            problems.append("symbolic-first")
        if task.metadata.get("goal id") not in GOAL_IDS:
            problems.append("goal-id")
        for field in ("estimated tokens", "token budget", "implementation timeout seconds", "llm context budget bytes"):
            if not _positive_int(task.metadata.get(field)):
                problems.append(field)
        if not task.outputs or not all(_safe_relative(path) for path in task.outputs):
            problems.append("outputs")
        if not task.validation or not task.acceptance.strip() or not task.metadata.get("proof requirements", "").strip():
            problems.append("evidence-gate")
        unknown = [dependency for dependency in task.depends_on if dependency not in task_by_id]
        duplicate_dependencies = [item for item, count in Counter(task.depends_on).items() if count > 1]
        if unknown:
            problems.append(f"unknown-dependencies:{unknown}")
        if task.task_id in task.depends_on:
            problems.append("self-dependency")
        if duplicate_dependencies:
            problems.append(f"duplicate-dependencies:{duplicate_dependencies}")
        graph[task.task_id] = tuple(task.depends_on)
        expected_lane = f"pcpc-lane-{_task_shard(task.task_id)}"
        observed_lane = str(task.metadata.get("parallel lane") or "")
        if observed_lane != expected_lane:
            lane_errors[task.task_id] = {
                "expected": expected_lane,
                "observed": observed_lane,
            }
        concurrent = _csv(task.metadata.get("allow concurrent with"))
        declared_concurrency[task.task_id] = concurrent
        unknown_concurrent = [item for item in concurrent if item not in task_by_id]
        duplicate_concurrent = [item for item, count in Counter(concurrent).items() if count > 1]
        if unknown_concurrent:
            problems.append(f"unknown-concurrency:{unknown_concurrent}")
        if task.task_id in concurrent:
            problems.append("self-concurrency")
        if duplicate_concurrent:
            problems.append(f"duplicate-concurrency:{duplicate_concurrent}")
        if problems:
            task_field_errors[task.task_id] = problems
    _append(checks, errors, name="task_contracts", passed=not task_field_errors, detail=task_field_errors)
    _append(
        checks,
        errors,
        name="task_parallel_lanes",
        passed=not lane_errors,
        detail=lane_errors,
    )
    acyclic, cycle = _acyclic(graph)
    dependency_count = sum(len(values) for values in graph.values())
    _append(checks, errors, name="dependency_dag", passed=acyclic and dependency_count == 75, detail={"acyclic": acyclic, "cycle": list(cycle), "dependency_count": dependency_count})
    unsafe_concurrency: list[dict[str, str]] = []
    if acyclic:
        transitive = {task_id: _transitive_dependencies(graph, task_id) for task_id in graph}
        for task_id, peers in declared_concurrency.items():
            for peer in peers:
                if peer not in graph:
                    continue
                if peer in transitive[task_id]:
                    unsafe_concurrency.append(
                        {"task_id": task_id, "peer": peer, "relation": "task_depends_on_peer"}
                    )
                elif task_id in transitive[peer]:
                    unsafe_concurrency.append(
                        {"task_id": task_id, "peer": peer, "relation": "peer_depends_on_task"}
                    )
    _append(
        checks,
        errors,
        name="concurrency_dependency_safety",
        passed=acyclic and not unsafe_concurrency,
        detail=unsafe_concurrency,
    )

    completed = tuple(task.task_id for task in tasks if task.status == "completed")
    completed_set = set(completed)
    ready = tuple(task.task_id for task in tasks if task.status == "todo" and set(task.depends_on) <= completed_set)
    _append(checks, errors, name="bootstrap_status", passed=completed == P0_COMPLETED and ready == INITIAL_READY and all(task.status != "blocked" for task in tasks), detail={"completed": list(completed), "ready": list(ready)})

    goal_errors: dict[str, list[str]] = {}
    for goal in goals:
        problems = [f"missing:{field}" for field in REQUIRED_GOAL_FIELDS if field not in goal.fields]
        if goal.fields.get("status") != "active":
            problems.append("status")
        parent = goal.fields.get("parent", "")
        if goal.goal_id == "PCPC-G000" and parent:
            problems.append("root-parent")
        if goal.goal_id != "PCPC-G000" and parent != "PCPC-G000":
            problems.append("subgoal-parent")
        if problems:
            goal_errors[goal.goal_id] = problems
    _append(checks, errors, name="goal_contracts", passed=not goal_errors, detail=goal_errors)

    config: dict[str, Any] = {}
    try:
        config = _load_json(CONFIG_PATH)
        board = load_configured_board(CONFIG_PATH, repo_root=REPO_ROOT)
        config_error = ""
    except (OSError, ValueError, ConfiguredBoardError) as exc:
        board = None
        config_error = f"{type(exc).__name__}: {exc}"
    _append(checks, errors, name="scheduler_schema", passed=board is not None, detail=config_error)
    if board is not None:
        initial = config.get("initial_projection") if isinstance(config.get("initial_projection"), Mapping) else {}
        _append(checks, errors, name="scheduler_identity", passed=board.board_namespace == PROGRAM and board.merge_target_branch == BRANCH and board.task_prefix == "PCPC-" and board.max_lanes == LANE_COUNT, detail={"namespace": board.board_namespace, "branch": board.merge_target_branch, "task_prefix": board.task_prefix, "max_lanes": board.max_lanes})
        _append(checks, errors, name="scheduler_projection", passed=tuple(initial.get("completed_task_ids") or ()) == P0_COMPLETED and tuple(initial.get("ready_task_ids") or ()) == INITIAL_READY and not (initial.get("blocked_task_ids") or ()) and initial.get("terminal_task_id") == TERMINAL_TASK and initial.get("task_dependency_count") == dependency_count, detail=initial)
        groups = config.get("task_groups") if isinstance(config.get("task_groups"), Mapping) else {}
        group_match = all(tuple(groups.get(goal_id) or ()) == task_ids for goal_id, task_ids in TASK_GROUPS.items()) and set(groups) == set(TASK_GROUPS)
        task_goal_match = all(task.metadata.get("goal id") == goal_id for goal_id, task_ids in TASK_GROUPS.items() for task in (task_by_id[item] for item in task_ids))
        _append(checks, errors, name="goal_task_groups", passed=group_match and task_goal_match, detail={"groups_match": group_match, "task_goal_match": task_goal_match})
        lanes = config.get("lanes") if isinstance(config.get("lanes"), list) else []
        initial_lane_map = {task_id: lane.get("index") for lane in lanes if isinstance(lane, Mapping) for task_id in (lane.get("initial_task_ids") or ())}
        expected_lane_map = {task_id: _task_shard(task_id) for task_id in INITIAL_READY}
        _append(
            checks,
            errors,
            name="parallel_shards",
            passed=(
                initial_lane_map == expected_lane_map
                and config.get("strict_task_sharding") is True
                and config.get("lane_assignment_algorithm")
                == "sha256_task_id_first_8_hex_mod_4"
                # Database-mode virgin transfer is deliberately disabled until
                # it can be fenced through a schema-compatible Quack
                # coordination adapter. Deterministic home shards still run
                # concurrently without cross-lane duplicate claims.
                and config.get("idle_lane_work_stealing") == ""
            ),
            detail={"expected": expected_lane_map, "actual": initial_lane_map},
        )
        source = config.get("source_binding") if isinstance(config.get("source_binding"), Mapping) else {}
        _append(checks, errors, name="source_binding", passed=source.get("accelerator_required_ancestor") == BASE_COMMIT and source.get("accelerator_required_branch") == BRANCH and source.get("bootstrap_task_source") == "duckdb" and source.get("planning_revision_is_runtime_completion_evidence") is False, detail=source)
        program = config.get("database_program") if isinstance(config.get("database_program"), Mapping) else {}
        owner = config.get("quack_owner_isolation") if isinstance(config.get("quack_owner_isolation"), Mapping) else {}
        ducklake = config.get("ducklake_projection_program") if isinstance(config.get("ducklake_projection_program"), Mapping) else {}
        _append(checks, errors, name="database_authority", passed=program.get("authority_mode") == "quack" and program.get("task_source_kind") == "duckdb" and str(program.get("quack_endpoint") or "").startswith("quack:127.0.0.1:") and program.get("failover_policy") == "fail_closed" and program.get("schema_revision") == "schema-v1", detail={"authority_mode": program.get("authority_mode"), "task_source_kind": program.get("task_source_kind"), "quack_endpoint": program.get("quack_endpoint"), "failover_policy": program.get("failover_policy"), "schema_revision": program.get("schema_revision")})
        owner_fields = {
            "schema",
            "required",
            "backend",
            "runtime_executable",
            "runtime_endpoint",
            "image_id",
            "image_os",
            "image_architecture",
            "image_label",
            "python_executable",
            "network",
            "host",
            "port",
            "container_bind_host",
            "container_port",
            "owner_write_root",
            "state_dir",
            "extension_directory",
            "extension_files_sha256",
            "pids_limit",
            "memory_bytes",
            "cpus",
            "tmpfs_size_bytes",
        }
        provider = config.get("provider") if isinstance(config.get("provider"), Mapping) else {}
        provider_isolation = provider.get("external_isolation") if isinstance(provider.get("external_isolation"), Mapping) else {}
        extension_hashes = owner.get("extension_files_sha256") if isinstance(owner.get("extension_files_sha256"), Mapping) else {}
        owner_root = PurePosixPath(str(owner.get("owner_write_root") or ""))
        store_path = PurePosixPath(str(program.get("store_id") or ""))
        owner_valid = (
            set(owner) == owner_fields
            and owner.get("schema")
            == "ipfs_accelerate_py.agent_supervisor.pcpc-quack-owner-isolation@1"
            and owner.get("required") is True
            and owner.get("backend") == "docker"
            and owner.get("runtime_executable") == "/usr/bin/docker"
            and str(owner.get("runtime_endpoint") or "").startswith("unix:///run/user/")
            and owner.get("image_id") == provider_isolation.get("image_id")
            and owner.get("image_os") == provider_isolation.get("image_os")
            and owner.get("image_architecture")
            == provider_isolation.get("image_architecture")
            and owner.get("image_label") == provider_isolation.get("image_label")
            and owner.get("network") == "bridge"
            and owner.get("host") == "127.0.0.1"
            and owner.get("port") == 45671
            and owner.get("container_bind_host") == "0.0.0.0"
            and owner.get("container_port") == owner.get("port")
            and _safe_relative(str(owner.get("owner_write_root") or ""))
            and _safe_relative(str(owner.get("state_dir") or ""))
            and store_path.parent == owner_root
            and PurePosixPath(str(owner.get("state_dir") or ""))
            == owner_root / "quack-owner"
            and set(extension_hashes)
            == {
                "httpfs.duckdb_extension",
                "httpfs.duckdb_extension.info",
                "quack.duckdb_extension",
                "quack.duckdb_extension.info",
            }
            and all(
                isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                for digest in extension_hashes.values()
            )
            and all(
                isinstance(owner.get(field), int)
                and not isinstance(owner.get(field), bool)
                and int(owner[field]) > 0
                for field in (
                    "port",
                    "container_port",
                    "pids_limit",
                    "memory_bytes",
                    "cpus",
                    "tmpfs_size_bytes",
                )
            )
        )
        _append(
            checks,
            errors,
            name="quack_owner_isolation",
            passed=owner_valid,
            detail={
                "owner_write_root": owner.get("owner_write_root"),
                "state_dir": owner.get("state_dir"),
                "store_id": program.get("store_id"),
                "image_id": owner.get("image_id"),
                "extension_files": sorted(extension_hashes),
            },
        )
        _append(checks, errors, name="ducklake_non_authority", passed=ducklake.get("mode") == "enabled_non_authoritative" and ducklake.get("authority") is False and ducklake.get("scheduling_prerequisite") is False and _safe_relative(str(ducklake.get("catalog_path") or "")) and _safe_relative(str(ducklake.get("data_path") or "")), detail=ducklake)

    try:
        baseline = _load_json(INVENTORY_ROOT / "baseline.json")
        prerequisites = _load_json(INVENTORY_ROOT / "prerequisites.json")
        inventory_error = ""
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        baseline = {}
        prerequisites = {}
        inventory_error = f"{type(exc).__name__}: {exc}"
    repository = baseline.get("repository") if isinstance(baseline.get("repository"), Mapping) else {}
    dispositions = prerequisites.get("dispositions") if isinstance(prerequisites.get("dispositions"), list) else []
    disposition_map = {str(item.get("authority")): str(item.get("status")) for item in dispositions if isinstance(item, Mapping)}
    _append(checks, errors, name="baseline_binding", passed=not inventory_error and repository.get("commit") == BASE_COMMIT and repository.get("tree") == BASE_TREE and baseline.get("package", {}).get("version") == "0.0.45", detail=inventory_error or repository)
    _append(checks, errors, name="prerequisite_gate", passed=set(disposition_map) == REQUIRED_AUTHORITIES and set(disposition_map.values()) <= DISPOSITIONS and disposition_map.get("AdaptivePlanner") == "incompatible" and disposition_map.get("AutonomousMetaController") == "missing", detail=disposition_map)

    try:
        benchmark = _load_json(BENCHMARK_MANIFEST)
        benchmark_error = ""
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        benchmark = {}
        benchmark_error = f"{type(exc).__name__}: {exc}"
    partitions = benchmark.get("partitions") if isinstance(benchmark.get("partitions"), list) else []
    _append(
        checks,
        errors,
        name="benchmark_scaffold",
        passed=(
            not benchmark_error
            and benchmark.get("frozen") is True
            and benchmark.get("status") == "scaffold_only"
            and benchmark.get("case_corpus_qualified") is False
            and benchmark.get("partition_coverage_established") is False
            and not (benchmark.get("case_manifest_refs") or ())
            and not any((benchmark.get("partition_case_counts") or {}).values())
            and set(partitions)
            == {"synthesis", "development", "held_out", "negative", "boundary", "adversarial"}
            and benchmark.get("held_out_disjoint") is True
        ),
        detail=benchmark_error
        or {
            "status": benchmark.get("status"),
            "families": len(benchmark.get("task_families") or ()),
            "partitions": partitions,
        },
    )

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler-board-validation@1",
        "valid": not errors,
        "program": PROGRAM,
        "task_count": len(tasks),
        "goal_count": len(goals),
        "dependency_count": dependency_count,
        "completed_task_ids": list(completed),
        "ready_task_ids": list(ready),
        "blocked_task_ids": [],
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true", help="Validate the complete sealed program")
    parser.parse_args(argv)
    try:
        report = validate_program()
    except Exception as exc:  # pragma: no cover - final fail-closed guard
        report = {
            "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler-board-validation@1",
            "valid": False,
            "errors": [f"validator_internal_error: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "checks": [],
        }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
