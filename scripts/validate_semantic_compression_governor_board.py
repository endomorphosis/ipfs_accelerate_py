#!/usr/bin/env python3
"""Fail-closed, dependency-free validator for the SCG supervisor board.

The planning controls are authority inputs to the implementation supervisor.
This validator intentionally uses only the Python standard library: importing
the product while validating the plan would let a broken implementation alter
the meaning of its own launch controls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_REL = "docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md"
OBJECTIVES_REL = "docs/architecture/semantic_compression_governor.objectives.md"
TODO_REL = "docs/architecture/semantic_compression_governor.todo.md"
CONFIG_REL = "config/semantic_compression_governor_scheduler.json"

BOARD_NAMESPACE = "semantic-compression-governor-v1"
SCHEDULER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "semantic_compression_governor.scheduler_config@1"
)
REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "semantic-compression-governor-board-validation@1"
)
BRANCH = "agent/semantic-compression-governor-v1"
TASK_IDS = tuple(f"SCG-{index:03d}" for index in range(49))
GOAL_IDS = ("SCG-G000",) + tuple(f"SCG-G{index:03d}" for index in range(10, 100, 10))
INITIAL_COMPLETED = ("SCG-000",)
INITIAL_READY = ("SCG-001", "SCG-002", "SCG-003", "SCG-004")
TERMINAL_TASK = "SCG-048"
MAX_CONTROL_BYTES = 4 * 1024 * 1024
MAX_METADATA_LINE_BYTES = 32 * 1024

ACCELERATE_PIN = "dfd92b554e662d4312411f2e8e63a52368806f2a"
DATASETS_PIN = "1330038f626ef92993f03d46f21e1a57719e9c25"
KIT_PIN = "df2f9cc092456329de9724c45a50c54b410875d1"
MCP_PLUS_PLUS_PIN = "dc3164653a48d059ae9812078359daeafb451c07"
IVP_PIN = "8c7800cedc5e1b848367db9952f912428466f8cc"
SEALER_DEVELOPMENT_PIN = "7dc8f1422cb7e80757077948dc0785c1aaa4fd25"

EXPECTED_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "SCG-000": (),
    "SCG-001": ("SCG-000",),
    "SCG-002": ("SCG-000",),
    "SCG-003": ("SCG-000",),
    "SCG-004": ("SCG-000",),
    "SCG-005": ("SCG-001", "SCG-002", "SCG-003", "SCG-004"),
    "SCG-006": ("SCG-005",),
    "SCG-007": ("SCG-006",),
    "SCG-008": ("SCG-006",),
    "SCG-009": ("SCG-006",),
    "SCG-010": ("SCG-006",),
    "SCG-011": ("SCG-007",),
    "SCG-012": ("SCG-009", "SCG-011"),
    "SCG-013": ("SCG-007",),
    "SCG-014": ("SCG-011", "SCG-013"),
    "SCG-015": ("SCG-012", "SCG-014"),
    "SCG-016": ("SCG-007", "SCG-009"),
    "SCG-017": ("SCG-015", "SCG-016"),
    "SCG-018": ("SCG-012", "SCG-015", "SCG-017"),
    "SCG-019": ("SCG-007", "SCG-008", "SCG-009", "SCG-010"),
    "SCG-020": ("SCG-019",),
    "SCG-021": ("SCG-019",),
    "SCG-022": ("SCG-020", "SCG-021"),
    "SCG-023": ("SCG-018", "SCG-022"),
    "SCG-024": ("SCG-013", "SCG-023"),
    "SCG-025": ("SCG-024",),
    "SCG-026": ("SCG-025",),
    "SCG-027": ("SCG-026",),
    "SCG-028": ("SCG-023",),
    "SCG-029": ("SCG-018", "SCG-027", "SCG-028"),
    "SCG-030": ("SCG-016", "SCG-029"),
    "SCG-031": ("SCG-022", "SCG-030"),
    "SCG-032": ("SCG-031",),
    "SCG-033": ("SCG-017", "SCG-022", "SCG-032"),
    "SCG-034": ("SCG-021", "SCG-033", "SCG-035"),
    "SCG-035": ("SCG-023", "SCG-033"),
    "SCG-036": ("SCG-032", "SCG-034", "SCG-035"),
    "SCG-037": ("SCG-036", "SCG-039"),
    "SCG-038": ("SCG-016", "SCG-032"),
    "SCG-039": ("SCG-036", "SCG-038"),
    "SCG-040": ("SCG-005",),
    "SCG-041": ("SCG-018", "SCG-040"),
    "SCG-042": ("SCG-024", "SCG-028", "SCG-029", "SCG-032", "SCG-041"),
    "SCG-043": ("SCG-022", "SCG-026", "SCG-029", "SCG-032", "SCG-041"),
    "SCG-044": ("SCG-037", "SCG-039", "SCG-042", "SCG-043"),
    "SCG-045": ("SCG-038", "SCG-044"),
    "SCG-046": ("SCG-045",),
    "SCG-047": ("SCG-035", "SCG-045"),
    "SCG-048": ("SCG-046", "SCG-047"),
}

EXPECTED_GROUPS: dict[str, tuple[str, ...]] = {
    "SCG-G000": ("SCG-000",),
    "SCG-G010": tuple(f"SCG-{index:03d}" for index in range(1, 6)),
    "SCG-G020": tuple(f"SCG-{index:03d}" for index in range(6, 11)),
    "SCG-G030": tuple(f"SCG-{index:03d}" for index in range(11, 19)),
    "SCG-G040": tuple(f"SCG-{index:03d}" for index in range(19, 23)),
    "SCG-G050": tuple(f"SCG-{index:03d}" for index in range(23, 33)),
    "SCG-G060": tuple(f"SCG-{index:03d}" for index in range(33, 36)),
    "SCG-G070": tuple(f"SCG-{index:03d}" for index in range(36, 40)),
    "SCG-G080": tuple(f"SCG-{index:03d}" for index in range(40, 46)),
    "SCG-G090": tuple(f"SCG-{index:03d}" for index in range(46, 49)),
}

EXPECTED_GOAL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "SCG-G000": (),
    "SCG-G010": (),
    "SCG-G020": ("SCG-G010",),
    "SCG-G030": ("SCG-G020",),
    "SCG-G040": ("SCG-G020",),
    "SCG-G050": ("SCG-G030", "SCG-G040"),
    "SCG-G060": ("SCG-G050",),
    "SCG-G070": ("SCG-G050", "SCG-G060"),
    "SCG-G080": ("SCG-G030", "SCG-G050", "SCG-G070"),
    "SCG-G090": ("SCG-G060", "SCG-G070", "SCG-G080"),
}

TASK_GOALS = {
    task_id: goal_id
    for goal_id, task_ids in EXPECTED_GROUPS.items()
    for task_id in task_ids
}

REQUIRED_TASK_FIELDS = frozenset(
    {
        "status",
        "completion",
        "is schedulable",
        "review only",
        "priority",
        "track",
        "depends on",
        "goal id",
        "outputs",
        "validation",
        "board namespace",
        "bundle",
        "parallel lane",
        "resource class",
        "implementation timeout seconds",
        "predicted files",
        "interfaces",
        "conflict policy",
        "preconditions",
        "effects",
        "evidence subset",
        "symbolic first",
        "llm context budget bytes",
        "acceptance",
    }
)
OPTIONAL_TASK_FIELDS = frozenset(
    {"completion evidence", "provider role", "context budget tokens"}
)
REQUIRED_GOAL_FIELDS = frozenset(
    {
        "status",
        "parent",
        "parent goal ids json",
        "depends on",
        "dependencies json",
        "fib priority",
        "track",
        "priority",
        "bundle",
        "parallel lane",
        "resource class",
        "goal",
        "producing tasks",
        "evidence",
        "evidence requirements json",
        "evidence criteria",
        "outputs",
        "predicted files",
        "predicted files json",
        "interfaces",
        "validation",
        "acceptance",
        "gap task",
        "refinement",
        "embedding query",
        "ast query",
        "conflict policy",
    }
)

PROTECTED_PATHS = (
    ".gitignore",
    PLAN_REL,
    OBJECTIVES_REL,
    TODO_REL,
    CONFIG_REL,
    "scripts/validate_semantic_compression_governor_board.py",
    "scripts/ops/agent_supervisor/semantic_compression_governor_scheduler.py",
    "scripts/ops/agent_supervisor/incremental_verification_planner_scheduler.py",
    "test/api/test_semantic_compression_governor_board.py",
)

EXPECTED_SOURCE_BINDING: dict[str, Any] = {
    "accelerator_required_ancestor": ACCELERATE_PIN,
    "accelerator_required_branch": BRANCH,
    "bootstrap_task_source": "legacy-markdown",
    "ipfs_datasets_submodule_path": "ipfs_datasets_py",
    "ipfs_datasets_planning_revision": DATASETS_PIN,
    "ipfs_kit_submodule_path": "ipfs_kit_py",
    "ipfs_kit_planning_revision": KIT_PIN,
    "mcp_plus_plus_submodule_path": "ipfs_accelerate_py/mcplusplus",
    "mcp_plus_plus_planning_revision": MCP_PLUS_PLUS_PIN,
    "require_initialized_gitlinks": True,
    "require_superproject_gitlink_equals_nested_head": True,
    "require_clean_nested_worktree_at_task_start": True,
    "record_recursive_repository_forest_at_launch": True,
    "changed_revision_requires_fresh_inventory_and_baseline": True,
    "planning_revision_is_runtime_completion_evidence": False,
}

EXPECTED_LANES: tuple[dict[str, Any], ...] = (
    {
        "index": 0,
        "name": "scg-lane-0",
        "strict_shard_remainder": 0,
        "initial_task_ids": ["SCG-003"],
        "initial_focus": "kit-storage-policy-cas-and-promotion",
    },
    {
        "index": 1,
        "name": "scg-lane-1",
        "strict_shard_remainder": 1,
        "initial_task_ids": ["SCG-001", "SCG-004"],
        "initial_focus": "accelerate-runtime-mcplusplus-and-sealing",
    },
    {
        "index": 2,
        "name": "scg-lane-2",
        "strict_shard_remainder": 2,
        "initial_task_ids": ["SCG-002"],
        "initial_focus": "datasets-analysis-expansion-and-calibration",
    },
)

EXPECTED_AUTHORITY_POLICY: dict[str, Any] = {
    "canonical_identity_authority": "ipfs_datasets_py.logic.software_contracts.content",
    "canonical_verification_authority": "ipfs_accelerate_py.agent_supervisor.verification",
    "canonical_storage_authority": (
        "ipfs_kit_py DurableCoordinationStore and DurableStateRootAdapter"
    ),
    "exact_receipt_identity_required": True,
    "provider_claim_is_verification_authority": False,
    "test_or_proof_presence_is_verification_authority": False,
    "verification_pass_alone_proves_sufficiency": False,
    "semantic_uncertainty_requires_broader_execution": True,
    "model_route_is_provider_selection": False,
    "model_agreement_is_proof": False,
    "production_acceptance_requires_current_admitted_receipts": True,
    "timeout_unavailable_unknown_stale_invalid_cancelled_or_simulated_can_pass": False,
    "nonreproducible_environment_requires_human_review": True,
}

EXPECTED_CAPABILITY_POLICY: dict[str, Any] = {
    "automatic_dependency_installation_allowed": False,
    "mock_hardware_or_inference_allowed": False,
    "new_content_identity_allowed": False,
    "new_receipt_format_allowed": False,
    "new_mcplusplus_profile_allowed": False,
    "new_proof_system_allowed": False,
    "missing_incremental_proof_sealer_disposition": "typed_unavailable",
    "missing_optional_dependency_disposition": "typed_unavailable",
    "shell_string_execution_allowed": False,
    "process_tree_termination_required": True,
    "bounded_stdout_and_stderr_required": True,
    "isolated_evaluation_worktree_required": True,
    "unapproved_external_expanded_shadow_allowed": False,
}

EXPECTED_COMPLETION_POLICY: dict[str, Any] = {
    "terminal_task_id": TERMINAL_TASK,
    "all_task_dependencies_terminal_required": True,
    "goal_heap_is_planning_lineage_only": True,
    "current_tree_evidence_required": True,
    "held_out_evaluation_required_for_promotion": True,
    "separate_authorization_required_for_promotion": True,
    "policy_compare_and_swap_required": True,
    "critical_assurance_reduction_without_authorization_allowed": False,
    "stale_partial_simulated_skipped_or_print_only_evidence_satisfies_release": False,
    "intentional_critical_omission_acceptance_floor": 0,
    "stale_or_simulated_production_acceptance_floor": 0,
    "honest_unmet_target_reporting_required": True,
    "proof_scope_must_be_bounded": True,
    "final_report_required": True,
}

ARTIFACT_MODELS = (
    "CompressionAuditCase",
    "ContextSufficiencyClaim",
    "ContextCoverageManifest",
    "ExcludedArtifactRecord",
    "OmissionHypothesis",
    "OmissionEvidence",
    "ContextExpansionPlan",
    "ContextExpansionStep",
    "ShadowExecutionPlan",
    "ShadowExecutionResult",
    "DifferentialPatchReport",
    "SemanticOutcomeComparison",
    "CapsuleCalibrationRecord",
    "AnalyzerCalibrationProfile",
    "TaskClassCalibrationProfile",
    "ModelRouteCalibrationProfile",
    "RuleProposal",
    "RuleEvaluationReport",
    "CompressionPolicy",
    "CompressionPolicyCandidate",
    "CompressionPolicyPromotionReceipt",
    "GovernorDecision",
    "GovernorRunReceipt",
)
SUFFICIENCY_STATES = (
    "sufficient",
    "sufficient_with_caveats",
    "expansion_required",
    "frontier_escalation_required",
    "human_review_required",
    "inconclusive",
    "invalid",
    "stale",
    "evaluation_failed",
)
OUTCOME_CLASSES = (
    "equivalent_success",
    "compressed_better",
    "expanded_better",
    "both_valid_different",
    "compressed_failed_expanded_succeeded",
    "compressed_succeeded_expanded_failed",
    "both_failed_same_reason",
    "both_failed_different_reason",
    "verification_inconclusive",
    "human_review_required",
)
REQUIRED_APIS = (
    "evaluate_context_sufficiency",
    "create_shadow_plan",
    "compare_shadow_results",
    "diagnose_omission",
    "plan_context_expansion",
    "execute_expansion_loop",
    "update_calibration",
    "propose_rule_change",
    "evaluate_rule_candidate",
    "promote_compression_policy",
)
REQUIRED_COMMANDS = (
    "audit",
    "shadow",
    "diagnose",
    "expand",
    "calibrate",
    "propose-rules",
    "evaluate-policy",
    "promote-policy",
    "report",
    "dashboard-data",
)

REQUIRED_PLAN_PHRASES = (
    "auditor and proposal system, not an autonomous production self-modifier",
    "may not rewrite production safety rules",
    "lower assurance",
    "directly change a production route",
    "mark a heuristic capsule exact",
    "disable a full-suite fallback",
    "suppress a verification failure",
    "treat model agreement as proof",
    "isolated evaluation worktree",
    "held-out partition disjoint",
    "expected-version cas",
    "expanded private source is local-only",
    "no public server is added",
    "ivp merkle commitment remains explicitly non-zk",
    "never claims to prove that every compressed context is semantically complete",
)


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def _csv(value: object) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def _read_text(root: Path, relative: str, errors: list[str]) -> str:
    path = root / relative
    try:
        if path.is_symlink():
            errors.append(f"{relative} must not be a symlink")
            return ""
        raw = path.read_bytes()
    except OSError as exc:
        errors.append(f"cannot read {relative}: {type(exc).__name__}")
        return ""
    if len(raw) > MAX_CONTROL_BYTES:
        errors.append(f"{relative} exceeds the control size limit")
        return ""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{relative} is not UTF-8")
        return ""
    if "\x00" in text:
        errors.append(f"{relative} contains a NUL byte")
    return text


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_config(root: Path, errors: list[str]) -> dict[str, Any]:
    text = _read_text(root, CONFIG_REL, errors)
    if not text:
        return {}
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{CONFIG_REL} is not closed valid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{CONFIG_REL} root must be an object")
        return {}
    return value


def _safe_relative_paths(
    values: Iterable[str], *, noun: str, errors: list[str]
) -> None:
    for raw in values:
        value = str(raw).strip().replace("\\", "/")
        path = PurePosixPath(value)
        if (
            not value
            or "\x00" in value
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() in {".", ".."}
            or (path.parts and path.parts[0].endswith(":"))
        ):
            errors.append(f"{noun} contains unsafe path {raw!r}")


def _positive_int(
    value: object, *, noun: str, errors: list[str], allow_zero: bool = False
) -> int:
    if isinstance(value, bool):
        errors.append(f"{noun} is not an integer")
        return -1
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        errors.append(f"{noun} is not an integer")
        return -1
    if parsed < (0 if allow_zero else 1):
        errors.append(f"{noun} is outside its admitted range")
    return parsed


def _parse_records(
    text: str,
    *,
    heading_pattern: re.Pattern[str],
    noun: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    matches = list(heading_pattern.finditer(text))
    records: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        block = text[match.end() : end]
        for line_number, line in enumerate(block.splitlines(), start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_METADATA_LINE_BYTES:
                errors.append(f"{match.group(1)} metadata row exceeds 32 KiB")
                continue
            field_match = re.fullmatch(r"- ([A-Za-z][A-Za-z ]*):(?: (.*))?", line)
            if field_match is None:
                errors.append(
                    f"{match.group(1)} contains non-one-line metadata at block line "
                    f"{line_number}"
                )
                continue
            field = field_match.group(1).strip().lower()
            if field in fields:
                errors.append(f"{match.group(1)} repeats metadata field {field!r}")
                continue
            fields[field] = field_match.group(2) or ""
        records.append(
            {"id": match.group(1), "title": match.group(2).strip(), "fields": fields}
        )
    if not matches:
        errors.append(f"{noun} contains no recognized records")
    return records


def _cycles(edges: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: set[str] = set()

    def visit(node: str, lineage: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle.add(node)
            if node in lineage:
                cycle.update(lineage[lineage.index(node) :])
            return
        visiting.add(node)
        for dependency in edges.get(node, ()):
            visit(dependency, (*lineage, node))
        visiting.remove(node)
        visited.add(node)

    for node in sorted(edges):
        visit(node, ())
    return tuple(sorted(cycle))


def _dependency_waves(
    edges: Mapping[str, Sequence[str]], population: Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    """Return the canonical earliest-start waves for an acyclic task graph."""

    remaining = set(population)
    completed: set[str] = set()
    waves: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(
            task_id
            for task_id in population
            if task_id in remaining
            and set(edges.get(task_id, ())).issubset(completed)
        )
        if not ready:
            # Unknown dependencies and cycles are reported separately. Returning
            # the maximal sound prefix prevents a forged wave summary from
            # concealing either condition.
            break
        waves.append(ready)
        completed.update(ready)
        remaining.difference_update(ready)
    return tuple(waves)


def _parse_waves(
    text: str,
    *,
    noun: str,
    expected_count: int,
    errors: list[str],
) -> tuple[tuple[str, ...], ...]:
    rows: dict[int, tuple[str, ...]] = {}
    for match in re.finditer(r"(?m)^W(\d+)\s+(.+?)\s*$", text):
        index = int(match.group(1))
        task_ids = tuple(re.findall(r"SCG-\d{3}", match.group(2)))
        if index in rows:
            errors.append(f"{noun} repeats wave W{index}")
        rows[index] = task_ids
    expected_indexes = tuple(range(expected_count))
    if tuple(sorted(rows)) != expected_indexes:
        errors.append(f"{noun} wave indexes differ from W0..W{expected_count - 1}")
    return tuple(rows.get(index, ()) for index in expected_indexes)


def _validate_plan(
    plan: str,
    todo: str,
    expected_waves: tuple[tuple[str, ...], ...],
    errors: list[str],
) -> None:
    normalized = _normalize(plan)
    for value in (
        BOARD_NAMESPACE,
        ACCELERATE_PIN,
        DATASETS_PIN,
        KIT_PIN,
        MCP_PLUS_PLUS_PIN,
        IVP_PIN,
        SEALER_DEVELOPMENT_PIN,
        *ARTIFACT_MODELS,
        *SUFFICIENCY_STATES,
        *OUTCOME_CLASSES,
        *REQUIRED_APIS,
        *REQUIRED_COMMANDS,
    ):
        if value.lower() not in normalized:
            errors.append(f"plan is missing required term {value!r}")
    for model in ARTIFACT_MODELS:
        if re.search(rf"(?m)^{re.escape(model)}$", plan) is None:
            errors.append(f"plan artifact list is missing {model!r}")
    for phrase in REQUIRED_PLAN_PHRASES:
        if phrase not in normalized:
            errors.append(f"plan is missing safety claim {phrase!r}")
    plan_waves = _parse_waves(
        plan, noun="plan", expected_count=len(expected_waves), errors=errors
    )
    todo_waves = _parse_waves(
        todo, noun="taskboard", expected_count=len(expected_waves), errors=errors
    )
    if plan_waves != expected_waves:
        errors.append("plan parallel waves differ from the closed task DAG")
    if todo_waves != expected_waves:
        errors.append("taskboard parallel waves differ from the closed task DAG")


def _expand_producing_tasks(value: str) -> tuple[str, ...]:
    range_match = re.fullmatch(r"SCG-(\d{3}) through SCG-(\d{3})", value.strip())
    if range_match:
        first, last = (int(range_match.group(1)), int(range_match.group(2)))
        if last < first:
            return ()
        return tuple(f"SCG-{index:03d}" for index in range(first, last + 1))
    return _csv(value)


def _validate_goals(text: str, errors: list[str]) -> dict[str, Any]:
    records = _parse_records(
        text,
        heading_pattern=re.compile(r"(?m)^## (SCG-G\d{3})\s+(.+)$"),
        noun="objective heap",
        errors=errors,
    )
    observed = tuple(record["id"] for record in records)
    if observed != GOAL_IDS:
        errors.append(f"goal IDs/order differ: expected {GOAL_IDS}, got {observed}")
    if len(observed) != len(set(observed)):
        errors.append("objective heap contains duplicate goal IDs")
    goal_set = set(observed)
    dependency_edges: dict[str, tuple[str, ...]] = {}
    parent_edges: dict[str, tuple[str, ...]] = {}
    for record in records:
        goal_id = record["id"]
        fields: dict[str, str] = record["fields"]
        missing = sorted(REQUIRED_GOAL_FIELDS.difference(fields))
        unknown = sorted(set(fields).difference(REQUIRED_GOAL_FIELDS))
        if missing:
            errors.append(f"{goal_id} is missing goal fields: {missing}")
        if unknown:
            errors.append(f"{goal_id} has unknown goal fields: {unknown}")
        for field in REQUIRED_GOAL_FIELDS.difference({"depends on"}):
            if field in fields and not fields[field].strip():
                errors.append(f"{goal_id} has empty {field}")
        if fields.get("status") != "active":
            errors.append(f"{goal_id} must begin active")
        if fields.get("priority") not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{goal_id} has invalid priority")
        _positive_int(fields.get("fib priority"), noun=f"{goal_id} Fib priority", errors=errors)

        parent = fields.get("parent", "").strip()
        parent_edges[goal_id] = () if parent == "none" else (parent,)
        expected_parent = "none" if goal_id == "SCG-G000" else "SCG-G000"
        if parent != expected_parent:
            errors.append(f"{goal_id} parent must be {expected_parent}")
        if parent != "none" and parent not in goal_set:
            errors.append(f"{goal_id} has unknown parent {parent!r}")
        try:
            parent_ids = json.loads(fields.get("parent goal ids json", ""))
        except json.JSONDecodeError:
            errors.append(f"{goal_id} parent goal IDs JSON is invalid")
        else:
            expected_parent_ids = [] if parent == "none" else [parent]
            if parent_ids != expected_parent_ids:
                errors.append(f"{goal_id} parent goal IDs JSON differs from Parent")

        dependencies = _csv(fields.get("depends on"))
        dependency_edges[goal_id] = dependencies
        expected_dependencies = EXPECTED_GOAL_DEPENDENCIES.get(goal_id, ())
        if dependencies != expected_dependencies:
            errors.append(
                f"{goal_id} dependencies differ: expected {expected_dependencies}, "
                f"got {dependencies}"
            )
        for dependency in dependencies:
            if dependency not in goal_set:
                errors.append(f"{goal_id} has unknown dependency {dependency!r}")
        try:
            dependency_ids = json.loads(fields.get("dependencies json", ""))
        except json.JSONDecodeError:
            errors.append(f"{goal_id} dependencies JSON is invalid")
        else:
            if dependency_ids != list(dependencies):
                errors.append(f"{goal_id} dependencies JSON differs from Depends on")

        expected_producers = (
            TASK_IDS[1:] if goal_id == "SCG-G000" else EXPECTED_GROUPS.get(goal_id, ())
        )
        producers = _expand_producing_tasks(fields.get("producing tasks", ""))
        if producers != expected_producers:
            errors.append(
                f"{goal_id} producing tasks differ: expected {expected_producers}, "
                f"got {producers}"
            )

        evidence = _csv(fields.get("evidence"))
        try:
            requirements = json.loads(fields.get("evidence requirements json", ""))
        except json.JSONDecodeError:
            errors.append(f"{goal_id} evidence requirements JSON is invalid")
        else:
            if not isinstance(requirements, list) or tuple(requirements) != evidence:
                errors.append(f"{goal_id} evidence requirements do not match Evidence")
        try:
            criteria = json.loads(fields.get("evidence criteria", ""))
        except json.JSONDecodeError:
            errors.append(f"{goal_id} evidence criteria JSON is invalid")
        else:
            if not isinstance(criteria, dict) or not criteria:
                errors.append(f"{goal_id} evidence criteria must be a nonempty object")
            elif goal_id == "SCG-G020" and criteria.get("required_models") != len(
                ARTIFACT_MODELS
            ):
                errors.append(
                    f"{goal_id} required_models must bind all {len(ARTIFACT_MODELS)} "
                    "named artifacts"
                )

        _safe_relative_paths(
            _csv(fields.get("predicted files")),
            noun=f"{goal_id} predicted files",
            errors=errors,
        )
        predicted_files = _csv(fields.get("predicted files"))
        try:
            predicted_json = json.loads(fields.get("predicted files json", ""))
        except json.JSONDecodeError:
            errors.append(f"{goal_id} predicted files JSON is invalid")
        else:
            if predicted_json != list(predicted_files):
                errors.append(
                    f"{goal_id} predicted files JSON differs from Predicted files"
                )
        gap_tasks = _csv(fields.get("gap task"))
        for task_id in gap_tasks:
            if task_id not in TASK_IDS:
                errors.append(f"{goal_id} has unknown gap task {task_id!r}")

    for noun, edges in (("goal parent", parent_edges), ("goal dependency", dependency_edges)):
        cycle = _cycles(edges)
        if cycle:
            errors.append(f"{noun} graph has a cycle: {cycle}")
    return {
        "goal_count": len(records),
        "dependency_edges": dependency_edges,
    }


def _transitive_dependencies(
    task_id: str, edges: Mapping[str, Sequence[str]]
) -> frozenset[str]:
    pending = list(edges.get(task_id, ()))
    reached: set[str] = set()
    while pending:
        dependency = pending.pop()
        if dependency in reached:
            continue
        reached.add(dependency)
        pending.extend(edges.get(dependency, ()))
    return frozenset(reached)


def _validate_tasks(
    text: str,
    config: Mapping[str, Any],
    goal_dependencies: Mapping[str, Sequence[str]],
    errors: list[str],
) -> dict[str, Any]:
    records = _parse_records(
        text,
        heading_pattern=re.compile(r"(?m)^## (SCG-\d{3})\s+(.+)$"),
        noun="taskboard",
        errors=errors,
    )
    observed = tuple(record["id"] for record in records)
    if observed != TASK_IDS:
        errors.append(f"task IDs/order differ: expected {TASK_IDS}, got {observed}")
    if len(observed) != len(set(observed)):
        errors.append("taskboard contains duplicate task IDs")
    task_set = set(observed)
    edges: dict[str, tuple[str, ...]] = {}
    predicted_by_task: dict[str, set[str]] = {}
    completed: list[str] = []
    for record in records:
        task_id = record["id"]
        fields: dict[str, str] = record["fields"]
        allowed = REQUIRED_TASK_FIELDS | OPTIONAL_TASK_FIELDS
        missing = sorted(REQUIRED_TASK_FIELDS.difference(fields))
        unknown = sorted(set(fields).difference(allowed))
        if missing:
            errors.append(f"{task_id} is missing task fields: {missing}")
        if unknown:
            errors.append(f"{task_id} has unknown task fields: {unknown}")
        for field in REQUIRED_TASK_FIELDS.difference({"depends on"}):
            if field in fields and not fields[field].strip():
                errors.append(f"{task_id} has empty {field}")

        status = fields.get("status")
        if status not in {"todo", "completed"}:
            errors.append(f"{task_id} has inadmissible status {status!r}")
        if task_id == "SCG-000" and status != "completed":
            errors.append("SCG-000 must remain completed")
        if status == "completed":
            completed.append(task_id)
        expected_completion = "manual" if task_id == "SCG-000" else "auto"
        if fields.get("completion") != expected_completion:
            errors.append(f"{task_id} completion must be {expected_completion}")
        expected_schedulable = "false" if task_id == "SCG-000" else "true"
        if fields.get("is schedulable") != expected_schedulable:
            errors.append(f"{task_id} is schedulable must be {expected_schedulable}")
        if fields.get("review only") != "false":
            errors.append(f"{task_id} must not be review-only")
        if fields.get("symbolic first") != "true":
            errors.append(f"{task_id} must remain symbolic-first")
        if fields.get("priority") not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{task_id} has invalid priority")
        if fields.get("board namespace") != BOARD_NAMESPACE:
            errors.append(f"{task_id} has wrong board namespace")
        task_timeout = _positive_int(
            fields.get("implementation timeout seconds"),
            noun=f"{task_id} implementation timeout",
            errors=errors,
        )
        scheduler_max = config.get("implementation_max_timeout_seconds")
        if type(scheduler_max) is int and task_timeout > scheduler_max:
            errors.append(f"{task_id} timeout exceeds scheduler hard maximum")
        _positive_int(
            fields.get("llm context budget bytes"),
            noun=f"{task_id} LLM context budget",
            errors=errors,
            allow_zero=task_id == "SCG-000",
        )

        dependencies = _csv(fields.get("depends on"))
        edges[task_id] = dependencies
        expected_dependencies = EXPECTED_DEPENDENCIES.get(task_id, ())
        if dependencies != expected_dependencies:
            errors.append(
                f"{task_id} dependencies differ: expected {expected_dependencies}, "
                f"got {dependencies}"
            )
        for dependency in dependencies:
            if dependency not in task_set:
                errors.append(f"{task_id} has unknown dependency {dependency!r}")

        expected_goal = TASK_GOALS.get(task_id)
        if fields.get("goal id") != expected_goal:
            errors.append(
                f"{task_id} goal differs: expected {expected_goal!r}, "
                f"got {fields.get('goal id')!r}"
            )

        outputs = _csv(fields.get("outputs"))
        predicted = _csv(fields.get("predicted files"))
        if not outputs or not predicted:
            errors.append(f"{task_id} must own nonempty outputs and predicted files")
        if outputs != predicted:
            errors.append(f"{task_id} outputs and predicted files differ")
        _safe_relative_paths(outputs, noun=f"{task_id} outputs", errors=errors)
        _safe_relative_paths(predicted, noun=f"{task_id} predicted files", errors=errors)
        predicted_by_task[task_id] = set(predicted)
        if task_id == "SCG-000":
            if predicted != PROTECTED_PATHS[1:]:
                errors.append("SCG-000 must own every non-gitignore protected control")
        else:
            forbidden = sorted(set(predicted).intersection(PROTECTED_PATHS))
            if forbidden:
                errors.append(f"{task_id} owns protected controls: {forbidden}")

    cycle = _cycles(edges)
    if cycle:
        errors.append(f"task dependency graph has a cycle: {cycle}")
    completed_set = set(completed)
    if "SCG-000" not in completed_set:
        errors.append("completed task population must include SCG-000")
    for task_id in completed:
        missing_dependencies = tuple(
            dependency
            for dependency in edges.get(task_id, ())
            if dependency not in completed_set
        )
        if missing_dependencies:
            errors.append(
                f"{task_id} is completed before dependencies {missing_dependencies}"
            )
    ready = tuple(
        task_id
        for task_id in TASK_IDS
        if task_id not in completed_set
        and all(dependency in completed_set for dependency in edges.get(task_id, ()))
    )
    if completed_set == set(INITIAL_COMPLETED) and ready != INITIAL_READY:
        errors.append(f"initial ready frontier differs: expected {INITIAL_READY}, got {ready}")
    if edges.get(TERMINAL_TASK) != ("SCG-046", "SCG-047"):
        errors.append("SCG-048 terminal fan-in must remain SCG-046 and SCG-047")
    if TERMINAL_TASK in completed_set:
        terminal_prerequisites = _transitive_dependencies(TERMINAL_TASK, edges)
        missing_terminal = tuple(
            task_id for task_id in TASK_IDS if task_id in terminal_prerequisites - completed_set
        )
        if missing_terminal:
            errors.append(
                f"SCG-048 is completed before transitive dependencies {missing_terminal}"
            )

    for task_id, dependencies in edges.items():
        task_goal = TASK_GOALS.get(task_id, "")
        admitted_goals = {
            task_goal,
            *_transitive_dependencies(task_goal, goal_dependencies),
        }
        for dependency in dependencies:
            if dependency == "SCG-000":
                continue
            dependency_goal = TASK_GOALS.get(dependency, "")
            if dependency_goal not in admitted_goals:
                errors.append(
                    f"{task_id} dependency {dependency} is absent from "
                    f"{task_goal} goal lineage"
                )

    for left_index, left in enumerate(TASK_IDS):
        left_closure = _transitive_dependencies(left, edges)
        for right in TASK_IDS[left_index + 1 :]:
            overlap = predicted_by_task.get(left, set()).intersection(
                predicted_by_task.get(right, set())
            )
            if not overlap:
                continue
            right_closure = _transitive_dependencies(right, edges)
            if left not in right_closure and right not in left_closure:
                errors.append(
                    f"unordered tasks {left}/{right} overlap predicted files: "
                    f"{sorted(overlap)}"
                )
    return {
        "task_count": len(records),
        "completed_task_ids": tuple(completed),
        "ready_task_ids": ready,
        "dependency_waves": _dependency_waves(edges, TASK_IDS),
    }


def _validate_scheduler(config: Mapping[str, Any], errors: list[str]) -> None:
    exact_scalars: dict[str, Any] = {
        "schema": SCHEDULER_SCHEMA,
        "taskboard_path": TODO_REL,
        "objectives_path": OBJECTIVES_REL,
        "plan_path": PLAN_REL,
        "validator_path": "scripts/validate_semantic_compression_governor_board.py",
        "task_prefix": "SCG-",
        "goal_prefix": "SCG-G",
        "board_namespace": BOARD_NAMESPACE,
        "merge_target_branch": BRANCH,
        "max_lanes": 3,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "poll_interval_seconds": 5,
        "daemon_interval_seconds": 45,
        "check_interval_seconds": 20,
        "stale_seconds": 1200,
        "watchdog_startup_grace_seconds": 300,
        "max_restarts": 8,
        "max_task_attempts": 4,
        "implementation_retry_budget": 4,
        "validation_retry_budget": 4,
        "merge_retry_budget": 4,
        "implementation_timeout_seconds": 14400,
        "implementation_max_timeout_seconds": 14400,
        "implementation_log_stall_seconds": 900,
    }
    expected_keys = set(exact_scalars).union(
        {
            "source_binding",
            "initial_projection",
            "worktree_submodule_paths",
            "protected_paths",
            "runtime_paths",
            "lanes",
            "task_groups",
            "provider",
            "authority_policy",
            "capability_policy",
            "completion_policy",
        }
    )
    missing_keys = sorted(expected_keys.difference(config))
    unknown_keys = sorted(set(config).difference(expected_keys))
    if missing_keys:
        errors.append(f"scheduler is missing fields: {missing_keys}")
    if unknown_keys:
        errors.append(f"scheduler has unknown fields: {unknown_keys}")
    for field, expected in exact_scalars.items():
        if config.get(field) != expected:
            errors.append(
                f"scheduler {field} differs: expected {expected!r}, got {config.get(field)!r}"
            )

    if config.get("source_binding") != EXPECTED_SOURCE_BINDING:
        errors.append("scheduler source binding differs from the reviewed exact pins")
    expected_projection = {
        "task_count": len(TASK_IDS),
        "completed_task_ids": list(INITIAL_COMPLETED),
        "ready_task_ids": list(INITIAL_READY),
        "blocked_task_ids": [],
        "terminal_task_id": TERMINAL_TASK,
        "goal_count": len(GOAL_IDS),
        "root_goal_id": "SCG-G000",
    }
    if config.get("initial_projection") != expected_projection:
        errors.append("scheduler initial projection differs from the board frontier")
    if config.get("task_groups") != {
        goal_id: list(task_ids) for goal_id, task_ids in EXPECTED_GROUPS.items()
    }:
        errors.append("scheduler task groups differ from exact goal ownership")

    lanes = config.get("lanes")
    if lanes != list(EXPECTED_LANES):
        errors.append("scheduler lane mapping differs from the reviewed mapping")
    if isinstance(lanes, list):
        lane_tasks: list[str] = []
        max_lanes = config.get("max_lanes")
        for lane in lanes:
            if not isinstance(lane, dict):
                errors.append("scheduler lanes must contain objects")
                continue
            remainder = lane.get("strict_shard_remainder")
            for task_id in lane.get("initial_task_ids", []):
                lane_tasks.append(str(task_id))
                match = re.fullmatch(r"SCG-(\d{3})", str(task_id))
                if (
                    match is None
                    or type(max_lanes) is not int
                    or int(match.group(1)) % max_lanes != remainder
                ):
                    errors.append(f"scheduler lane parity mismatch for {task_id}")
        if tuple(sorted(lane_tasks)) != tuple(sorted(INITIAL_READY)):
            errors.append("scheduler lanes do not cover the initial ready frontier exactly")

    if config.get("protected_paths") != list(PROTECTED_PATHS):
        errors.append("scheduler protected controls differ from the closed control set")
    expected_submodules = [
        "ipfs_datasets_py",
        "ipfs_kit_py",
        "ipfs_accelerate_py/mcplusplus",
    ]
    if config.get("worktree_submodule_paths") != expected_submodules:
        errors.append("scheduler worktree submodule paths differ from source binding")

    runtime_root = "data/agent_supervisor/semantic_compression_governor/run-scg-v1"
    expected_runtime = {
        "root": runtime_root,
        "state": f"{runtime_root}/state",
        "worktrees": f"{runtime_root}/worktrees",
        "merge_queue": f"{runtime_root}/merge-queue",
        "logs": f"{runtime_root}/logs",
        "evidence": f"{runtime_root}/evidence",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    if config.get("runtime_paths") != expected_runtime:
        errors.append("scheduler runtime paths differ from the isolated SCG namespace")
    _safe_relative_paths(
        [value for value in expected_runtime.values() if isinstance(value, str)],
        noun="scheduler runtime paths",
        errors=errors,
    )

    expected_provider = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 3,
        "secrets_from_environment_only": True,
        "secrets_in_argv_prompts_logs_or_receipts": False,
    }
    if config.get("provider") != expected_provider:
        errors.append("scheduler provider route or secret policy differs")
    if config.get("authority_policy") != EXPECTED_AUTHORITY_POLICY:
        errors.append("scheduler authority safety claims differ")
    if config.get("capability_policy") != EXPECTED_CAPABILITY_POLICY:
        errors.append("scheduler capability safety claims differ")
    if config.get("completion_policy") != EXPECTED_COMPLETION_POLICY:
        errors.append("scheduler completion and promotion safety claims differ")


def validate(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Validate the four SCG controls below *repo_root* and return JSON data."""

    root = Path(repo_root)
    errors: list[str] = []
    plan = _read_text(root, PLAN_REL, errors)
    objectives = _read_text(root, OBJECTIVES_REL, errors)
    todo = _read_text(root, TODO_REL, errors)
    config = _load_config(root, errors)

    goal_summary = _validate_goals(objectives, errors)
    goal_dependencies = goal_summary.get("dependency_edges", {})
    task_summary = _validate_tasks(todo, config, goal_dependencies, errors)
    expected_waves = task_summary.get(
        "dependency_waves", _dependency_waves(EXPECTED_DEPENDENCIES, TASK_IDS)
    )
    _validate_plan(plan, todo, expected_waves, errors)
    _validate_scheduler(config, errors)

    # Keep output deterministic and avoid duplicated diagnostics from joined checks.
    errors = sorted(dict.fromkeys(errors))
    return {
        "schema": REPORT_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "valid": not errors,
        "errors": errors,
        "task_count": task_summary.get("task_count", 0),
        "goal_count": goal_summary.get("goal_count", 0),
        "completed_task_ids": list(task_summary.get("completed_task_ids", ())),
        "ready_task_ids": list(task_summary.get("ready_task_ids", ())),
        "terminal_task_id": TERMINAL_TASK,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the sealed Semantic Compression Governor board."
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="validate the plan, goal heap, taskboard, and scheduler profile",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.check_all:
        payload = {
            "schema": REPORT_SCHEMA,
            "board_namespace": BOARD_NAMESPACE,
            "valid": False,
            "errors": ["explicit --check-all is required"],
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 2
    payload = validate()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
