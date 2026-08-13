#!/usr/bin/env python3
"""Fail-closed, stdlib-only validator for the AAE supervisor controls."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_REL = "docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_PLAN.md"
OBJECTIVES_REL = "docs/architecture/adversarial_assurance_engine.objectives.md"
TODO_REL = "docs/architecture/adversarial_assurance_engine.todo.md"
SCHEDULER_REL = "config/adversarial_assurance_engine_scheduler.json"
PREREQUISITES_REL = "config/adversarial_assurance_prerequisites.json"
LAUNCHER_REL = "scripts/ops/agent_supervisor/adversarial_assurance_engine_scheduler.py"

BOARD_NAMESPACE = "adversarial-assurance-engine-v1"
BRANCH = "agent/adversarial-assurance-engine-v1"
BASE_REVISION = "7c9f3fa3d2ac14c7b5bfa5036e2fe6fb59f0afda"
DATASETS_REVISION = "fbd1ba9f70803de157622bb20e22595ef09d606f"
KIT_REVISION = "c7e5feeb24582ab68c1f5ca626366b665a82ad61"
MCP_REVISION = "dc3164653a48d059ae9812078359daeafb451c07"

TASK_IDS = tuple(f"AAE-{index:03d}" for index in range(64))
GOAL_IDS = ("AAE-G000",) + tuple(f"AAE-G{index:03d}" for index in range(10, 100, 10))
TERMINAL_TASK = "AAE-063"
OPERATOR_GATE = "AAE-006"
INITIAL_COMPLETED = ("AAE-000",)
INITIAL_READY = ("AAE-001", "AAE-002", "AAE-003", "AAE-004")
INITIAL_BLOCKED = (OPERATOR_GATE,)

EXPECTED_GROUPS: dict[str, tuple[str, ...]] = {
    "AAE-G000": ("AAE-000",),
    "AAE-G010": tuple(f"AAE-{value:03d}" for value in range(1, 6)),
    "AAE-G020": tuple(f"AAE-{value:03d}" for value in range(7, 14)),
    "AAE-G030": tuple(f"AAE-{value:03d}" for value in range(14, 25)),
    "AAE-G040": tuple(f"AAE-{value:03d}" for value in range(25, 34)),
    "AAE-G050": tuple(f"AAE-{value:03d}" for value in range(34, 39)),
    "AAE-G060": ("AAE-006",) + tuple(f"AAE-{value:03d}" for value in range(39, 49)),
    "AAE-G070": tuple(f"AAE-{value:03d}" for value in range(49, 56)),
    "AAE-G080": tuple(f"AAE-{value:03d}" for value in range(56, 63)),
    "AAE-G090": ("AAE-063",),
}
TASK_GOALS = {
    task_id: goal_id
    for goal_id, task_ids in EXPECTED_GROUPS.items()
    for task_id in task_ids
}
EXPECTED_GOAL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "AAE-G000": (),
    "AAE-G010": (),
    "AAE-G020": ("AAE-G010",),
    "AAE-G030": ("AAE-G020",),
    "AAE-G040": ("AAE-G020", "AAE-G030"),
    "AAE-G050": ("AAE-G020",),
    "AAE-G060": ("AAE-G010", "AAE-G030", "AAE-G040", "AAE-G050"),
    "AAE-G070": ("AAE-G030", "AAE-G040", "AAE-G050", "AAE-G060"),
    "AAE-G080": ("AAE-G050", "AAE-G060", "AAE-G070"),
    "AAE-G090": ("AAE-G080",),
}

EXPECTED_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "AAE-000": (),
    "AAE-001": ("AAE-000",),
    "AAE-002": ("AAE-000",),
    "AAE-003": ("AAE-000",),
    "AAE-004": ("AAE-000",),
    "AAE-005": ("AAE-001", "AAE-002", "AAE-003", "AAE-004"),
    "AAE-006": ("AAE-005",),
    "AAE-007": ("AAE-005",),
    "AAE-008": ("AAE-007",),
    "AAE-009": ("AAE-007",),
    "AAE-010": ("AAE-007",),
    "AAE-011": ("AAE-007",),
    "AAE-012": ("AAE-007", "AAE-008", "AAE-009", "AAE-010", "AAE-011"),
    "AAE-013": ("AAE-008", "AAE-009", "AAE-010", "AAE-011", "AAE-012"),
    "AAE-014": ("AAE-008", "AAE-012"),
    "AAE-015": ("AAE-014",),
    "AAE-016": ("AAE-014",),
    "AAE-017": ("AAE-014",),
    "AAE-018": ("AAE-014",),
    "AAE-019": ("AAE-014",),
    "AAE-020": ("AAE-014",),
    "AAE-021": ("AAE-008", "AAE-010", "AAE-012"),
    "AAE-022": (
        "AAE-014", "AAE-015", "AAE-016", "AAE-017", "AAE-018", "AAE-019",
        "AAE-020", "AAE-021",
    ),
    "AAE-023": ("AAE-009", "AAE-021"),
    "AAE-024": ("AAE-008", "AAE-009", "AAE-022", "AAE-023"),
    "AAE-025": ("AAE-009", "AAE-024"),
    "AAE-026": ("AAE-010", "AAE-012"),
    "AAE-027": ("AAE-010", "AAE-012"),
    "AAE-028": ("AAE-009", "AAE-010", "AAE-023"),
    "AAE-029": ("AAE-010", "AAE-028"),
    "AAE-030": ("AAE-025", "AAE-026", "AAE-027", "AAE-028", "AAE-029"),
    "AAE-031": ("AAE-030",),
    "AAE-032": ("AAE-011", "AAE-030"),
    "AAE-033": ("AAE-011", "AAE-012", "AAE-032"),
    "AAE-034": ("AAE-007", "AAE-008", "AAE-009", "AAE-010", "AAE-011", "AAE-012"),
    "AAE-035": ("AAE-034",),
    "AAE-036": ("AAE-034", "AAE-035"),
    "AAE-037": ("AAE-012", "AAE-034", "AAE-035"),
    "AAE-038": ("AAE-034", "AAE-035", "AAE-036", "AAE-037"),
    "AAE-039": ("AAE-006", "AAE-013", "AAE-038"),
    "AAE-040": ("AAE-024", "AAE-039"),
    "AAE-041": ("AAE-024", "AAE-039"),
    "AAE-042": ("AAE-041",),
    "AAE-043": ("AAE-006", "AAE-039", "AAE-040"),
    "AAE-044": ("AAE-040", "AAE-041", "AAE-042", "AAE-043"),
    "AAE-045": ("AAE-031", "AAE-044"),
    "AAE-046": ("AAE-032", "AAE-033", "AAE-045"),
    "AAE-047": ("AAE-037", "AAE-046"),
    "AAE-048": ("AAE-040", "AAE-044", "AAE-045", "AAE-046", "AAE-047"),
    "AAE-049": ("AAE-013", "AAE-024", "AAE-033", "AAE-038"),
    "AAE-050": ("AAE-049", "AAE-044", "AAE-045"),
    "AAE-051": ("AAE-049", "AAE-044", "AAE-045"),
    "AAE-052": ("AAE-049", "AAE-044", "AAE-045"),
    "AAE-053": ("AAE-006", "AAE-049", "AAE-044", "AAE-045"),
    "AAE-054": ("AAE-049", "AAE-044", "AAE-045"),
    "AAE-055": ("AAE-026", "AAE-027", "AAE-049", "AAE-044", "AAE-045"),
    "AAE-056": ("AAE-048",),
    "AAE-057": ("AAE-047", "AAE-048", "AAE-056"),
    "AAE-058": ("AAE-035", "AAE-044", "AAE-046", "AAE-048"),
    "AAE-059": ("AAE-041", "AAE-042", "AAE-049"),
    "AAE-060": ("AAE-038", "AAE-042", "AAE-044", "AAE-047", "AAE-049"),
    "AAE-061": (
        "AAE-050", "AAE-051", "AAE-052", "AAE-053", "AAE-054", "AAE-055",
        "AAE-046", "AAE-047", "AAE-059", "AAE-060",
    ),
    "AAE-062": ("AAE-006", "AAE-036", "AAE-043", "AAE-052", "AAE-053", "AAE-058", "AAE-061"),
    "AAE-063": ("AAE-056", "AAE-057", "AAE-058", "AAE-061", "AAE-062"),
}

REQUIRED_TASK_FIELDS = (
    "status", "completion", "is schedulable", "review only", "priority", "track",
    "depends on", "goal id", "outputs", "validation", "board namespace", "bundle",
    "parallel lane", "resource class", "implementation timeout seconds",
    "llm context budget bytes", "predicted files", "interfaces", "conflict policy",
    "preconditions", "effects", "evidence subset", "symbolic first", "acceptance",
)
REQUIRED_GOAL_FIELDS = (
    "status", "parent", "parent goal ids json", "depends on", "dependencies json",
    "fib priority", "track", "priority", "bundle", "parallel lane", "resource class",
    "goal", "producing tasks", "evidence", "evidence requirements json",
    "evidence criteria", "outputs", "predicted files", "predicted files json",
    "interfaces", "validation", "acceptance", "gap task", "refinement",
    "embedding query", "ast query", "conflict policy",
)

PROTECTED_PATHS = (
    ".gitignore",
    PLAN_REL,
    OBJECTIVES_REL,
    TODO_REL,
    SCHEDULER_REL,
    PREREQUISITES_REL,
    "scripts/validate_adversarial_assurance_engine_board.py",
    LAUNCHER_REL,
    "scripts/ops/agent_supervisor/incremental_verification_planner_scheduler.py",
    "test/api/test_adversarial_assurance_engine_board.py",
    "scripts/ops/agent_supervisor/implementation_supervisor_entry.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
    "ipfs_accelerate_py/agent_supervisor/control/lifecycle_orchestrator.py",
    "ipfs_accelerate_py/agent_supervisor/provider_fallback_runner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
)

AUTHORITY_POLICY = {
    "canonical_identity_authority": "ipfs_datasets_py.logic.software_contracts.content",
    "canonical_semantic_authority": "IncrementalSemanticIndex and SemanticCapsuleCompiler@1",
    "canonical_context_authority": "ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack.ContextPacker",
    "canonical_verification_authority": "ipfs_accelerate_py.agent_supervisor.verification",
    "canonical_storage_authority": "ipfs_kit_py DurableCoordinationStore and DurableStateRootAdapter",
    "canonical_sealing_authority": "released IncrementalProofSealer only",
    "exact_receipt_identity_required": True,
    "provider_claim_is_verification_authority": False,
    "test_or_proof_presence_is_verification_authority": False,
    "verification_pass_alone_proves_sufficiency": False,
    "mutation_score_proves_correctness": False,
    "receipt_inclusion_proves_execution": False,
    "semantic_uncertainty_requires_broader_execution": True,
    "timeout_unavailable_unknown_stale_invalid_cancelled_or_simulated_can_pass": False,
    "nonreproducible_environment_requires_human_review": True,
}
CAPABILITY_POLICY = {
    "automatic_dependency_installation_allowed": False,
    "production_credentials_available_to_mutants": False,
    "mutant_network_access_default": False,
    "mock_external_effects_required": True,
    "new_content_identity_allowed": False,
    "new_receipt_format_allowed": False,
    "new_scheduler_allowed": False,
    "new_context_packer_allowed": False,
    "new_mcplusplus_profile_allowed": False,
    "new_proof_or_zk_system_allowed": False,
    "missing_incremental_proof_sealer_disposition": "typed_unavailable",
    "shell_string_execution_allowed": False,
    "process_tree_termination_required": True,
    "bounded_stdout_and_stderr_required": True,
    "isolated_disposable_worktree_required": True,
    "arbitrary_external_repository_mutation_allowed": False,
    "production_policy_change_during_fixture_campaign_allowed": False,
    "production_deployment_allowed": False,
    "general_autonomous_repair_allowed": False,
    "gui_implementation_allowed": False,
    "legal_advice_allowed": False,
    "payment_processing_allowed": False,
    "automatic_assurance_lowering_allowed": False,
}
COMPLETION_POLICY = {
    "terminal_task_id": TERMINAL_TASK,
    "all_task_dependencies_terminal_required": True,
    "prerequisite_gate_must_be_completed_by_operator": True,
    "current_tree_evidence_required": True,
    "held_out_evaluation_required_for_promotion": True,
    "separate_authorization_required_for_promotion": True,
    "policy_compare_and_swap_required": True,
    "new_incremental_seal_required_for_promotion": True,
    "critical_assurance_reduction_without_authorization_allowed": False,
    "stale_partial_simulated_skipped_or_print_only_evidence_satisfies_release": False,
    "zero_controlled_critical_security_survivors_is_target_not_assumed": True,
    "high_risk_detection_target_percent": 90,
    "incremental_savings_target_percent": 50,
    "honest_unmet_target_reporting_required": True,
    "proof_scope_must_be_bounded": True,
    "final_report_required": True,
}

REQUIRED_MODELS = (
    "MutationOperatorDefinition", "MutationTarget", "MutationCandidate",
    "MutationCampaignPolicy", "MutationCampaignPlan", "ExpectedDetectionSet",
    "MutationExecutionPlan", "MutationExecutionReceipt", "MutationOutcome",
    "MutationEquivalenceAssessment", "SurvivingMutantReport", "AssuranceGap",
    "VacuityFinding", "DetectionFailure", "TestAdequacyProfile",
    "ProofAdequacyProfile", "PolicyAdequacyProfile", "CapsuleAdequacyProfile",
    "CandidateTestSpecification", "CandidateProofObligation",
    "CandidatePolicyConstraint", "CandidateAnalyzerRule", "GapRemediationPlan",
    "RemediationEvaluationReport", "AssuranceCampaignReceipt",
    "AssurancePolicyPromotionReceipt", "AssuranceManifest",
)
OUTCOME_STATUSES = (
    "killed_by_static_analysis", "killed_by_type_check", "killed_by_test",
    "killed_by_formal_proof", "killed_by_policy", "killed_by_runtime_invariant",
    "killed_by_full_suite", "survived_selected_verification",
    "survived_full_verification", "equivalent", "probably_equivalent",
    "invalid_mutant", "uncompilable", "infrastructure_failure", "timeout",
    "inconclusive", "human_review_required",
)
GAP_STATUSES = (
    "missing_test", "weak_assertion", "missing_proof_obligation", "vacuous_proof",
    "missing_policy_constraint", "stale_or_incomplete_dependency_edge",
    "capsule_completeness_failure", "test_selection_failure", "unmodeled_side_effect",
    "missing_state_transition_constraint", "missing_environment_binding",
    "receipt_authenticity_gap", "specification_ambiguity", "intentionally_unconstrained",
)
REQUIRED_APIS = (
    "create_assurance_manifest", "generate_mutation_candidates", "predict_detection_set",
    "execute_mutation", "classify_mutation_outcome", "diagnose_surviving_mutant",
    "analyze_vacuity", "propose_gap_remediation", "evaluate_remediation",
    "promote_assurance_policy", "plan_mutation_campaign", "execute_mutation_campaign",
)
REQUIRED_PLAN_TERMS = (
    "IncrementalSemanticIndex", "SemanticCapsuleCompiler", "ContextPackBuilder",
    "ContextPacker", "VerificationReceiptCache", "IncrementalVerificationPlanner",
    "ModelRoutePlanner", "IncrementalProofSealer", "SemanticCompressionGovernor",
    "control flow", "data/schema", "interface contract", "side effect", "error/retry",
    "authorization/policy", "state/distributed", "storage/durability", "test/proof",
    "semantic compression", "GUI/action binding", "authentication bypass",
    "caller-selected tenant", "payment-as-authority", "simulated production evidence",
    "receipt leaf", "proof-forest order", "after mutant creation",
    "during worktree setup/test/proof", "before policy CAS", "after CAS before cleanup",
    "risk-weighted score", "proof-cache reuse", "cost per critical gap",
    "no public network service is required", "new MCP++ profile is forbidden",
    "difficulty to kill never implies equivalence", "Every vacuity finding states exactly what remains proven",
    "held-out", "expected-old policy revision CAS", "temporary proof forest",
    "signed assurance evidence", "signer/key identity", "signature verification",
    "no production deployment", "no general autonomous code repair", "no GUI implementation",
    "no legal advice", "no payment processing", "no automatic lowering of assurance requirements",
)
FINAL_CLAIM = """The system used semantically targeted counterfactual mutations to test whether
declared tests, proofs, policies, semantic summaries, and incremental seals
reject important incorrect behavior. Surviving mutants were classified as
assurance gaps, candidate remediations were evaluated against held-out
mutations, and accepted assurance-policy changes were promoted through a
reproducible, content-addressed qualification process."""


@dataclass(frozen=True)
class MarkdownRecord:
    record_id: str
    title: str
    fields: Mapping[str, str]


def _csv(value: object) -> tuple[str, ...]:
    return tuple(part.strip() for part in re.split(r"[,;]", str(value or "")) if part.strip())


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _load_json(path: Path, errors: list[str]) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name} is not valid JSON: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} root must be an object")
        return {}
    return value


def _parse_markdown_records(
    text: str,
    *,
    header: re.Pattern[str],
    noun: str,
    errors: list[str],
) -> tuple[MarkdownRecord, ...]:
    matches = list(header.finditer(text))
    records: list[MarkdownRecord] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        for line_number, line in enumerate(text[match.end():end].splitlines(), start=1):
            if not line.strip():
                continue
            metadata = re.fullmatch(r"- ([A-Za-z][A-Za-z0-9 /+()-]*):(?: ?(.*))?", line)
            if metadata is None:
                errors.append(
                    f"{match.group(1)} contains non-one-line {noun} metadata at block line {line_number}"
                )
                continue
            key = metadata.group(1).strip().casefold()
            if key in fields:
                errors.append(f"{match.group(1)} repeats metadata field {key!r}")
            fields[key] = metadata.group(2) or ""
            if len(line.encode("utf-8")) > 16_384:
                errors.append(f"{match.group(1)} metadata row exceeds 16 KiB")
        records.append(MarkdownRecord(match.group(1), match.group(2).strip(), fields))
    return tuple(records)


def _safe_paths(values: Iterable[str], *, noun: str, errors: list[str]) -> None:
    for raw in values:
        value = raw.strip().replace("\\", "/")
        path = PurePosixPath(value)
        if (
            not value
            or "\x00" in value
            or path.is_absolute()
            or ".." in path.parts
            or value in {".", ".."}
            or (path.parts and (path.parts[0].endswith(":") or path.parts[0] == ".git"))
            or any(character in value for character in "*?[]{}")
        ):
            errors.append(f"{noun} contains unsafe path {raw!r}")


def _path_in_prefixes(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def _is_path_overlap(left: str, right: str) -> bool:
    left_path = PurePosixPath(left)
    right_path = PurePosixPath(right)
    return left_path == right_path or left_path in right_path.parents or right_path in left_path.parents


def _cycle_nodes(edges: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: set[str] = set()

    def visit(node: str, lineage: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle.add(node)
            if node in lineage:
                cycle.update(lineage[lineage.index(node):])
            return
        visiting.add(node)
        for dependency in edges.get(node, ()):
            visit(dependency, (*lineage, node))
        visiting.remove(node)
        visited.add(node)

    for node in sorted(edges):
        visit(node, ())
    return tuple(sorted(cycle))


def _transitive_dependencies(task_id: str, edges: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
    pending = list(edges.get(task_id, ()))
    reached: set[str] = set()
    while pending:
        dependency = pending.pop()
        if dependency in reached:
            continue
        reached.add(dependency)
        pending.extend(edges.get(dependency, ()))
    return frozenset(reached)


def _positive_int(value: object, *, noun: str, errors: list[str]) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError):
        errors.append(f"{noun} is not an integer")
        return -1
    if result <= 0:
        errors.append(f"{noun} must be positive")
    return result


def _json_string_list(value: str, *, noun: str, errors: list[str]) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        errors.append(f"{noun} is not valid JSON")
        return ()
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        errors.append(f"{noun} must be a JSON string list")
        return ()
    return tuple(parsed)


def _task_owner_prefixes(task_id: str) -> tuple[str, ...]:
    number = int(task_id[-3:])
    if number == 0:
        return PROTECTED_PATHS
    if 1 <= number <= 5:
        return ("docs/architecture/adversarial_assurance_inventory",)
    if number == 6:
        return (PREREQUISITES_REL, "docs/architecture/adversarial_assurance_inventory")
    if 7 <= number <= 11 or 14 <= number <= 23 or 25 <= number <= 33:
        return (
            "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance",
            "ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance",
        )
    if number == 12:
        return (
            "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/adversarial_assurance",
            "ipfs_datasets_py/tests/unit/logic/software_contracts/adversarial_assurance",
        )
    if number == 13:
        return (
            "ipfs_accelerate_py/mcplusplus/docs/architecture",
            "ipfs_accelerate_py/mcplusplus/schemas",
            "ipfs_accelerate_py/mcplusplus/conformance/vectors",
            "ipfs_accelerate_py/mcplusplus/tests-py/integration",
            "ipfs_accelerate_py/mcplusplus/tests-go",
            "ipfs_accelerate_py/mcplusplus/tests-rs/tests",
            "ipfs_accelerate_py/mcplusplus/tests-ts/src/__tests__",
        )
    if number == 24:
        return ("ipfs_accelerate_py/agent_supervisor/adversarial_assurance", "test/api/adversarial_assurance")
    if 34 <= number <= 38:
        return ("ipfs_kit_py/ipfs_kit_py/adversarial_assurance_store", "ipfs_kit_py/tests/adversarial_assurance_store")
    if number == 56:
        return (
            "ipfs_accelerate_py/agent_supervisor/adversarial_assurance",
            "ipfs_accelerate_py/cli.py",
            "test/api/adversarial_assurance",
        )
    if 39 <= number <= 48 or 57 <= number <= 58:
        return ("ipfs_accelerate_py/agent_supervisor/adversarial_assurance", "test/api/adversarial_assurance")
    if 49 <= number <= 55:
        return ("test/fixtures/adversarial_assurance", "test/api/adversarial_assurance")
    if 59 <= number <= 61:
        return ("test/api/adversarial_assurance", "test/fixtures/adversarial_assurance")
    if number == 62:
        return ("benchmarks/agent_supervisor", "artifacts/agent_supervisor/adversarial_assurance", "test/api/adversarial_assurance")
    if number == 63:
        return ("docs/guides", "docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md", "test/api/adversarial_assurance")
    return ()


def _parse_producing_tasks(value: str) -> tuple[str, ...]:
    range_match = re.fullmatch(r"(AAE-\d{3})\s+through\s+(AAE-\d{3})", value.strip())
    if range_match:
        start = int(range_match.group(1)[-3:])
        end = int(range_match.group(2)[-3:])
        if start <= end:
            return tuple(f"AAE-{index:03d}" for index in range(start, end + 1))
    return tuple(re.findall(r"AAE-\d{3}", value))


def _validate_plan(plan_text: str, todo_text: str, errors: list[str]) -> None:
    plan_normalized = _normalized(plan_text)
    for term in (*REQUIRED_MODELS, *OUTCOME_STATUSES, *GAP_STATUSES, *REQUIRED_APIS, *REQUIRED_PLAN_TERMS):
        if _normalized(term) not in plan_normalized:
            errors.append(f"plan is missing required coverage term {term!r}")
    for term in ("equivalent`, `probably_equivalent`,\n`not_equivalent`, and `unknown",):
        if _normalized(term) not in plan_normalized:
            errors.append("plan is missing the closed equivalence status set")
    for command in (
        "assurance mutate plan|run|target|explain", "assurance gaps", "assurance vacuity",
        "assurance remediate", "assurance evaluate-remediation", "assurance promote",
        "assurance report", "assurance benchmark",
    ):
        if _normalized(command) not in plan_normalized:
            errors.append(f"plan is missing required CLI coverage {command!r}")
    if _normalized(FINAL_CLAIM) not in plan_normalized:
        errors.append("plan is missing or altered the prescribed bounded final claim")
    wave_rows = re.findall(r"^W(\d{2})\s+(.+)$", todo_text, re.MULTILINE)
    if tuple(index for index, _ in wave_rows) != tuple(f"{index:02d}" for index in range(21)):
        errors.append("taskboard wave numbering must be exactly W00 through W20")
        return
    task_wave: dict[str, int] = {}
    for wave_index, (_, body) in enumerate(wave_rows):
        for task_id in re.findall(r"AAE-\d{3}", body):
            if task_id in task_wave:
                errors.append(f"parallel waves repeat {task_id}")
            task_wave[task_id] = wave_index
    if set(task_wave) != set(TASK_IDS):
        errors.append("parallel waves do not cover the exact task population")
    for task_id, dependencies in EXPECTED_DEPENDENCIES.items():
        for dependency in dependencies:
            if task_wave.get(dependency, -1) >= task_wave.get(task_id, 10_000):
                errors.append(f"parallel wave order violates {task_id} dependency {dependency}")


def _validate_goals(objective_text: str, errors: list[str]) -> tuple[MarkdownRecord, ...]:
    records = _parse_markdown_records(
        objective_text,
        header=re.compile(r"^## (AAE-G\d{3})\s+(.+)$", re.MULTILINE),
        noun="goal",
        errors=errors,
    )
    observed = tuple(record.record_id for record in records)
    if observed != GOAL_IDS:
        errors.append(f"goal IDs/order differ: expected {GOAL_IDS}, got {observed}")
    if len(observed) != len(set(observed)):
        errors.append("objective heap contains duplicate goal IDs")
    parent_edges: dict[str, tuple[str, ...]] = {}
    dependency_edges: dict[str, tuple[str, ...]] = {}
    for record in records:
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in record.fields]
        if missing:
            errors.append(f"{record.record_id} is missing goal fields: {missing}")
        if record.fields.get("status") != "active":
            errors.append(f"{record.record_id} must remain active in the sealed goal heap")
        expected_parent = "none" if record.record_id == "AAE-G000" else "AAE-G000"
        parent = record.fields.get("parent", "")
        if parent != expected_parent:
            errors.append(f"{record.record_id} parent differs: expected {expected_parent!r}, got {parent!r}")
        parent_edges[record.record_id] = () if parent == "none" else (parent,)
        parent_json = _json_string_list(
            record.fields.get("parent goal ids json", ""),
            noun=f"{record.record_id} parent goal IDs JSON",
            errors=errors,
        )
        expected_parent_json = () if record.record_id == "AAE-G000" else ("AAE-G000",)
        if parent_json != expected_parent_json:
            errors.append(f"{record.record_id} parent JSON differs")
        dependencies = _csv(record.fields.get("depends on"))
        dependency_edges[record.record_id] = dependencies
        expected_dependencies = EXPECTED_GOAL_DEPENDENCIES.get(record.record_id, ())
        if dependencies != expected_dependencies:
            errors.append(
                f"{record.record_id} dependencies differ: expected {expected_dependencies}, got {dependencies}"
            )
        dependencies_json = _json_string_list(
            record.fields.get("dependencies json", ""),
            noun=f"{record.record_id} dependencies JSON",
            errors=errors,
        )
        if dependencies_json != dependencies:
            errors.append(f"{record.record_id} dependency JSON differs from Depends on")
        _positive_int(record.fields.get("fib priority"), noun=f"{record.record_id} Fib priority", errors=errors)
        if record.fields.get("priority") not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{record.record_id} has invalid priority")
        if record.fields.get("validation") != "python3 scripts/validate_adversarial_assurance_engine_board.py --check-all":
            errors.append(f"{record.record_id} validation does not use the canonical board validator")
        produced = _parse_producing_tasks(record.fields.get("producing tasks", ""))
        expected_produced = TASK_IDS[1:] if record.record_id == "AAE-G000" else EXPECTED_GROUPS.get(record.record_id, ())
        if produced != expected_produced:
            errors.append(f"{record.record_id} producing tasks differ from the sealed group")
        predicted = _csv(record.fields.get("predicted files"))
        predicted_json = _json_string_list(
            record.fields.get("predicted files json", ""),
            noun=f"{record.record_id} predicted files JSON",
            errors=errors,
        )
        if predicted_json != predicted:
            errors.append(f"{record.record_id} predicted files JSON differs")
        _safe_paths(predicted, noun=f"{record.record_id} predicted files", errors=errors)
        evidence = _csv(record.fields.get("evidence"))
        evidence_json = _json_string_list(
            record.fields.get("evidence requirements json", ""),
            noun=f"{record.record_id} evidence requirements JSON",
            errors=errors,
        )
        if evidence_json != evidence:
            errors.append(f"{record.record_id} evidence requirements differ from Evidence")
        try:
            criteria = json.loads(record.fields.get("evidence criteria", ""))
        except json.JSONDecodeError:
            errors.append(f"{record.record_id} evidence criteria is invalid JSON")
        else:
            expected_criteria = {
                "results_honest": True,
                "canonical_identity_required": True,
                "held_out_required": record.record_id in {
                    "AAE-G000", "AAE-G040", "AAE-G060", "AAE-G070", "AAE-G080", "AAE-G090"
                },
                "unauthorized_policy_changes": 0,
                "proof_scope_bounded": True,
            }
            if criteria != expected_criteria:
                errors.append(f"{record.record_id} evidence criteria differs")
        for field in REQUIRED_GOAL_FIELDS:
            if field not in {"depends on"} and field in record.fields and not record.fields[field].strip():
                errors.append(f"{record.record_id} has empty {field}")
    if _cycle_nodes(parent_edges):
        errors.append("goal parent graph contains a cycle")
    cycle = _cycle_nodes(dependency_edges)
    if cycle:
        errors.append(f"goal dependency graph contains a cycle: {cycle}")
    return records


def _validate_tasks(
    todo_text: str,
    scheduler: Mapping[str, object],
    prerequisite: Mapping[str, object],
    errors: list[str],
) -> tuple[MarkdownRecord, ...]:
    records = _parse_markdown_records(
        todo_text,
        header=re.compile(r"^## (AAE-\d{3})\s+(.+)$", re.MULTILINE),
        noun="task",
        errors=errors,
    )
    observed = tuple(record.record_id for record in records)
    if observed != TASK_IDS:
        errors.append(f"task IDs/order differ: expected {TASK_IDS}, got {observed}")
    if len(observed) != len(set(observed)):
        errors.append("taskboard contains duplicate task IDs")
    task_by_id = {record.record_id: record for record in records}
    edges: dict[str, tuple[str, ...]] = {}
    predicted_by_task: dict[str, tuple[str, ...]] = {}
    max_timeout = int(scheduler.get("implementation_max_timeout_seconds") or 0)
    protected = tuple(str(value) for value in scheduler.get("protected_paths", ()) if isinstance(value, str))
    for record in records:
        fields = record.fields
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in fields]
        if missing:
            errors.append(f"{record.record_id} is missing task fields: {missing}")
        allowed_statuses = {"completed"} if record.record_id == "AAE-000" else {"blocked", "completed"} if record.record_id == OPERATOR_GATE else {"todo", "completed"}
        receipt_status = str(prerequisite.get("status") or "")
        if record.record_id == OPERATOR_GATE and receipt_status == "completed":
            allowed_statuses = {"completed"}
        if fields.get("status") not in allowed_statuses:
            errors.append(
                f"{record.record_id} status must be one of {sorted(allowed_statuses)!r}"
            )
        expected_completion = "manual" if record.record_id in {"AAE-000", OPERATOR_GATE} else "auto"
        if fields.get("completion") != expected_completion:
            errors.append(f"{record.record_id} completion must be {expected_completion}")
        expected_schedulable = "false" if record.record_id in {"AAE-000", OPERATOR_GATE} else "true"
        if fields.get("is schedulable") != expected_schedulable:
            errors.append(f"{record.record_id} schedulability differs")
        if fields.get("review only") != "false":
            errors.append(f"{record.record_id} must not be review-only")
        if fields.get("priority") not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{record.record_id} has invalid priority")
        if fields.get("board namespace") != BOARD_NAMESPACE:
            errors.append(f"{record.record_id} board namespace differs")
        if fields.get("symbolic first") != "true":
            errors.append(f"{record.record_id} must remain symbolic-first")
        timeout = _positive_int(fields.get("implementation timeout seconds"), noun=f"{record.record_id} timeout", errors=errors)
        if max_timeout > 0 and timeout > max_timeout:
            errors.append(f"{record.record_id} timeout exceeds scheduler maximum")
        _positive_int(fields.get("llm context budget bytes"), noun=f"{record.record_id} context budget", errors=errors)
        dependencies = _csv(fields.get("depends on"))
        edges[record.record_id] = dependencies
        expected_dependencies = EXPECTED_DEPENDENCIES.get(record.record_id, ())
        if dependencies != expected_dependencies:
            errors.append(
                f"{record.record_id} dependencies differ: expected {expected_dependencies}, got {dependencies}"
            )
        for dependency in dependencies:
            if dependency not in TASK_IDS:
                errors.append(f"{record.record_id} has unknown dependency {dependency}")
            if dependency not in fields.get("preconditions", ""):
                errors.append(f"{record.record_id} preconditions omit dependency {dependency}")
        if fields.get("goal id") != TASK_GOALS.get(record.record_id):
            errors.append(f"{record.record_id} goal assignment differs")
        outputs = _csv(fields.get("outputs"))
        predicted = _csv(fields.get("predicted files"))
        predicted_by_task[record.record_id] = predicted
        if outputs != predicted:
            errors.append(f"{record.record_id} Outputs and Predicted files differ")
        _safe_paths(outputs, noun=f"{record.record_id} outputs", errors=errors)
        allowed = _task_owner_prefixes(record.record_id)
        for path in predicted:
            if not _path_in_prefixes(path, allowed):
                errors.append(f"{record.record_id} predicted path violates repository ownership: {path}")
            if record.record_id not in {"AAE-000", OPERATOR_GATE}:
                for protected_path in protected:
                    if _is_path_overlap(path, protected_path):
                        errors.append(f"{record.record_id} predicted path overlaps protected control: {path}")
        if record.record_id in {"AAE-000", OPERATOR_GATE}:
            if fields.get("provider role") != "operator-only":
                errors.append(f"{record.record_id} must remain operator-only")
        elif "provider role" in fields:
            errors.append(f"{record.record_id} must not claim an operator provider role")
        if record.record_id == OPERATOR_GATE:
            if "blocked reason" not in fields or not fields["blocked reason"].strip():
                errors.append("AAE-006 requires a nonempty blocked reason")
            if fields.get("validation") != "python3 scripts/validate_adversarial_assurance_engine_board.py --check-prerequisites":
                errors.append("AAE-006 must use the prerequisite validator")
            doctrine = _normalized(fields.get("acceptance", "") + " " + fields.get("conflict policy", ""))
            for term in ("only an operator", "workers cannot", "released checkpoint and delta sealer apis"):
                if _normalized(term) not in doctrine:
                    errors.append(f"AAE-006 operator doctrine omits {term!r}")
        for field in REQUIRED_TASK_FIELDS:
            if field != "depends on" and field in fields and not fields[field].strip():
                errors.append(f"{record.record_id} has empty {field}")
    cycle = _cycle_nodes(edges)
    if cycle:
        errors.append(f"task dependency graph contains a cycle: {cycle}")
    if records:
        completed = tuple(record.record_id for record in records if record.fields.get("status") == "completed")
        blocked = tuple(record.record_id for record in records if record.fields.get("status") == "blocked")
        completed_set = set(completed)
        for task_id in completed:
            missing_completed_dependencies = sorted(
                dependency
                for dependency in edges.get(task_id, ())
                if dependency not in completed_set
            )
            if missing_completed_dependencies:
                errors.append(
                    f"{task_id} is completed before dependencies: "
                    f"{missing_completed_dependencies}"
                )
        bootstrap_state = (
            prerequisite.get("status") != "completed"
            and completed == INITIAL_COMPLETED
            and blocked == INITIAL_BLOCKED
        )
        if bootstrap_state:
            if completed != INITIAL_COMPLETED:
                errors.append(f"initial completed projection differs: {completed}")
            if blocked != INITIAL_BLOCKED:
                errors.append(f"initial blocked projection differs: {blocked}")
            ready = tuple(
                task_id for task_id in TASK_IDS
                if task_by_id.get(task_id)
                and task_by_id[task_id].fields.get("status") == "todo"
                and task_by_id[task_id].fields.get("is schedulable") == "true"
                and all(dependency in completed_set for dependency in edges.get(task_id, ()))
            )
            if ready != INITIAL_READY:
                errors.append(f"initial ready projection differs: expected {INITIAL_READY}, got {ready}")
    for left_index, left in enumerate(TASK_IDS):
        left_dependencies = _transitive_dependencies(left, edges)
        for right in TASK_IDS[left_index + 1:]:
            right_dependencies = _transitive_dependencies(right, edges)
            if left in right_dependencies or right in left_dependencies:
                continue
            for left_path in predicted_by_task.get(left, ()):
                for right_path in predicted_by_task.get(right, ()):
                    if _is_path_overlap(left_path, right_path):
                        errors.append(f"unordered tasks {left}/{right} overlap predicted paths: {left_path} / {right_path}")
    if not records:
        return records
    ancestors = _transitive_dependencies(TERMINAL_TASK, edges)
    missing = sorted(set(TASK_IDS[:-1]) - ancestors)
    if missing:
        errors.append(f"terminal task does not fan in tasks: {missing}")
    sinks = sorted(task_id for task_id in TASK_IDS if not any(task_id in dependencies for dependencies in edges.values()))
    if sinks != [TERMINAL_TASK]:
        errors.append(f"terminal sink differs: expected {[TERMINAL_TASK]}, got {sinks}")
    terminal = task_by_id.get(TERMINAL_TASK)
    if terminal is not None:
        if terminal.fields.get("goal id") != "AAE-G090":
            errors.append("AAE-063 must belong to AAE-G090")
        for required in (
            "docs/architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md",
            "test/api/adversarial_assurance/test_current_tree_conformance.py",
        ):
            if required not in predicted_by_task.get(TERMINAL_TASK, ()):
                errors.append(f"AAE-063 terminal outputs omit {required}")
        if "--check-all" not in terminal.fields.get("validation", ""):
            errors.append("AAE-063 does not rerun board validation")
    return records


def _validate_scheduler(
    scheduler: Mapping[str, object],
    *,
    repo_root: Path,
    check_repository: bool,
    errors: list[str],
) -> None:
    exact = {
        "schema": "ipfs_accelerate_py.agent_supervisor.adversarial_assurance_engine.scheduler_config@1",
        "taskboard_path": TODO_REL,
        "objectives_path": OBJECTIVES_REL,
        "plan_path": PLAN_REL,
        "validator_path": "scripts/validate_adversarial_assurance_engine_board.py",
        "task_prefix": "AAE-",
        "goal_prefix": "AAE-G",
        "board_namespace": BOARD_NAMESPACE,
        "merge_target_branch": BRANCH,
        "max_lanes": 2,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
    }
    for field, expected in exact.items():
        if scheduler.get(field) != expected:
            errors.append(f"scheduler {field} differs: expected {expected!r}, got {scheduler.get(field)!r}")
    source = scheduler.get("source_binding")
    expected_source = {
        "accelerator_required_ancestor": BASE_REVISION,
        "accelerator_required_branch": BRANCH,
        "bootstrap_task_source": "legacy-markdown",
        "ipfs_datasets_submodule_path": "ipfs_datasets_py",
        "ipfs_datasets_planning_revision": DATASETS_REVISION,
        "ipfs_kit_submodule_path": "ipfs_kit_py",
        "ipfs_kit_planning_revision": KIT_REVISION,
        "mcp_plus_plus_submodule_path": "ipfs_accelerate_py/mcplusplus",
        "mcp_plus_plus_planning_revision": MCP_REVISION,
        "require_initialized_gitlinks": True,
        "require_superproject_gitlink_equals_nested_head": True,
        "require_clean_nested_worktree_at_task_start": True,
        "record_recursive_repository_forest_at_launch": True,
        "changed_revision_requires_fresh_inventory_and_baseline": True,
        "planning_revision_is_runtime_completion_evidence": False,
    }
    if source != expected_source:
        errors.append("scheduler source_binding differs from the reviewed source pins")
    if scheduler.get("worktree_submodule_paths") != ["ipfs_datasets_py", "ipfs_kit_py", "ipfs_accelerate_py/mcplusplus"]:
        errors.append("scheduler worktree submodule population/order differs")
    protected = scheduler.get("protected_paths")
    if protected != list(PROTECTED_PATHS):
        errors.append("scheduler protected_paths differ from the exact reviewed controls")
    if isinstance(protected, list) and len(protected) != len(set(map(str, protected))):
        errors.append("scheduler protected_paths contain duplicates")
    expected_projection = {
        "task_count": 64,
        "completed_task_ids": list(INITIAL_COMPLETED),
        "ready_task_ids": list(INITIAL_READY),
        "blocked_task_ids": list(INITIAL_BLOCKED),
        "terminal_task_id": TERMINAL_TASK,
        "goal_count": 10,
        "root_goal_id": "AAE-G000",
    }
    if scheduler.get("initial_projection") != expected_projection:
        errors.append("scheduler initial_projection differs")
    groups = scheduler.get("task_groups")
    expected_groups_json = {goal_id: list(task_ids) for goal_id, task_ids in EXPECTED_GROUPS.items()}
    if groups != expected_groups_json:
        errors.append("scheduler task_groups differ from the reviewed goal projection")
    lanes = scheduler.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 2:
        errors.append("scheduler must define exactly two strict lanes")
    else:
        all_initial: set[str] = set()
        for index, lane in enumerate(lanes):
            if not isinstance(lane, Mapping):
                errors.append(f"scheduler lane {index} is not an object")
                continue
            if lane.get("index") != index or lane.get("strict_shard_remainder") != index:
                errors.append(f"scheduler lane {index} sharding metadata differs")
            if lane.get("name") != f"aae-lane-{index}":
                errors.append(f"scheduler lane {index} name differs")
            initial = lane.get("initial_task_ids")
            if not isinstance(initial, list):
                errors.append(f"scheduler lane {index} initial tasks are not a list")
                continue
            for task_id in initial:
                match = re.fullmatch(r"AAE-(\d{3})", str(task_id))
                if match is None or int(match.group(1)) % 2 != index:
                    errors.append(f"scheduler task {task_id!r} is assigned to the wrong strict shard")
                all_initial.add(str(task_id))
        if all_initial != set(INITIAL_READY):
            errors.append("scheduler lane initial tasks differ from the initial ready frontier")
    provider = scheduler.get("provider")
    expected_provider = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 2,
        "secrets_from_environment_only": True,
        "secrets_in_argv_prompts_logs_or_receipts": False,
    }
    if provider != expected_provider:
        errors.append("scheduler provider route or concurrency differs")
    for field in (
        "poll_interval_seconds", "daemon_interval_seconds", "check_interval_seconds",
        "stale_seconds", "watchdog_startup_grace_seconds", "max_restarts",
        "max_task_attempts", "implementation_retry_budget", "validation_retry_budget",
        "merge_retry_budget", "implementation_timeout_seconds",
        "implementation_max_timeout_seconds", "implementation_log_stall_seconds",
    ):
        _positive_int(scheduler.get(field), noun=f"scheduler {field}", errors=errors)
    if scheduler.get("authority_policy") != AUTHORITY_POLICY:
        errors.append("scheduler assurance authority doctrine differs")
    if scheduler.get("capability_policy") != CAPABILITY_POLICY:
        errors.append("scheduler capability/security doctrine differs")
    if scheduler.get("completion_policy") != COMPLETION_POLICY:
        errors.append("scheduler completion/promotion doctrine differs")
    gate = scheduler.get("prerequisite_gate")
    expected_gate = {
        "task_id": OPERATOR_GATE,
        "initial_status": "blocked",
        "operator_only": True,
        "runtime_tasks": ["AAE-039", "AAE-043", "AAE-053", "AAE-062"],
        "required_evidence": [
            "genuine terminal SemanticCompressionGovernor receipt and final commit",
            "released full-checkpoint and delta IncrementalProofSealer APIs",
            "clean exact recursive repository forest",
            "fresh focused baselines at the released pins",
        ],
        "missing_capability_disposition": "typed_unavailable",
        "worker_may_complete": False,
    }
    if gate != expected_gate:
        errors.append("scheduler prerequisite gate differs")
    for runtime_task in expected_gate["runtime_tasks"]:
        if OPERATOR_GATE not in EXPECTED_DEPENDENCIES[runtime_task]:
            errors.append(f"runtime task {runtime_task} lacks direct operator-gate dependency")
    runtime = scheduler.get("runtime_paths")
    runtime_root = "data/agent_supervisor/adversarial_assurance_engine/run-aae-v1"
    if not isinstance(runtime, Mapping) or runtime.get("root") != runtime_root:
        errors.append("scheduler runtime root differs")
    else:
        for field in ("state", "worktrees", "merge_queue", "logs", "evidence"):
            value = str(runtime.get(field) or "")
            if not value.startswith(runtime_root + "/"):
                errors.append(f"scheduler runtime {field} escapes the AAE runtime root")
        if runtime.get("generated_runtime_artifacts_are_completion_authority") is not False:
            errors.append("generated runtime artifacts must not be completion authority")
    if check_repository:
        for protected_path in PROTECTED_PATHS:
            if not (repo_root / protected_path).exists():
                errors.append(f"protected path is absent: {protected_path}")
        try:
            ignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            errors.append(".gitignore is unavailable for runtime-root validation")
        else:
            if "/data/agent_supervisor/adversarial_assurance_engine/" not in ignore_text:
                errors.append("AAE generated runtime root is not ignored")


def _run_git(repo_root: Path, argv: Sequence[str], errors: list[str]) -> str:
    try:
        result = subprocess.run(
            ("git", *argv), cwd=repo_root, text=True, capture_output=True,
            check=False, timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"git {' '.join(argv)} failed: {type(exc).__name__}")
        return ""
    if result.returncode != 0:
        errors.append(f"git {' '.join(argv)} rejected: {result.stderr.strip()[-500:] or result.returncode}")
        return ""
    return result.stdout.strip()


def _validate_repository_pins(repo_root: Path, scheduler: Mapping[str, object], errors: list[str]) -> None:
    branch = _run_git(repo_root, ("branch", "--show-current"), errors)
    if branch and branch != BRANCH:
        errors.append(f"controller branch differs: expected {BRANCH}, got {branch}")
    _run_git(repo_root, ("merge-base", "--is-ancestor", BASE_REVISION, "HEAD"), errors)
    source = scheduler.get("source_binding")
    if not isinstance(source, Mapping):
        return
    bindings = (
        ("ipfs_datasets_py", "ipfs_datasets_planning_revision"),
        ("ipfs_kit_py", "ipfs_kit_planning_revision"),
        ("ipfs_accelerate_py/mcplusplus", "mcp_plus_plus_planning_revision"),
    )
    for path, revision_field in bindings:
        expected = str(source.get(revision_field) or "")
        gitlink = _run_git(repo_root, ("rev-parse", f"HEAD:{path}"), errors)
        nested_head = _run_git(repo_root / path, ("rev-parse", "HEAD"), errors)
        if gitlink and nested_head and gitlink != nested_head:
            errors.append(f"{path} gitlink differs from nested HEAD")
        if expected and nested_head:
            _run_git(repo_root / path, ("merge-base", "--is-ancestor", expected, nested_head), errors)
        status = _run_git(repo_root / path, ("status", "--porcelain"), errors)
        if status:
            errors.append(f"{path} nested worktree is not clean")


def _is_full_hex(value: object) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", str(value or "")) is not None


def _prerequisite_release_report(
    repo_root: Path,
    *,
    check_repository: bool,
) -> dict[str, object]:
    errors: list[str] = []
    scheduler = _load_json(repo_root / SCHEDULER_REL, errors)
    receipt = _load_json(repo_root / PREREQUISITES_REL, errors)
    try:
        todo_text = (repo_root / TODO_REL).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"taskboard is unreadable: {type(exc).__name__}: {exc}")
        todo_text = ""
    parse_errors: list[str] = []
    records = _parse_markdown_records(
        todo_text,
        header=re.compile(r"^## (AAE-\d{3})\s+(.+)$", re.MULTILINE),
        noun="task",
        errors=parse_errors,
    )
    errors.extend(parse_errors)
    task_by_id = {record.record_id: record for record in records}
    if receipt.get("schema") != "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-prerequisite-receipt@1":
        errors.append("prerequisite receipt schema differs")
    if receipt.get("task_id") != OPERATOR_GATE:
        errors.append("prerequisite receipt task identity differs")
    if receipt.get("status") != "completed":
        errors.append("prerequisite receipt is not completed")
    gate_task = task_by_id.get(OPERATOR_GATE)
    if gate_task is None or gate_task.fields.get("status") != "completed":
        errors.append("AAE-006 is not operator-marked completed")
    dependency_task = task_by_id.get("AAE-005")
    if dependency_task is None or dependency_task.fields.get("status") != "completed":
        errors.append("AAE-006 dependency AAE-005 is not completed")
    observed_at = str(receipt.get("observed_at") or "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", observed_at) is None:
        errors.append("prerequisite receipt observed_at is not a canonical UTC timestamp")
    source = scheduler.get("source_binding")
    controller = receipt.get("controller")
    if not isinstance(source, Mapping) or not isinstance(controller, Mapping):
        errors.append("prerequisite controller/source binding is absent")
    else:
        expected_controller = {
            "repository": "endomorphosis/ipfs_accelerate_py",
            "branch": source.get("accelerator_required_branch"),
            "required_ancestor": source.get("accelerator_required_ancestor"),
            "planning_gitlinks": {
                "ipfs_datasets_py": source.get("ipfs_datasets_planning_revision"),
                "ipfs_kit_py": source.get("ipfs_kit_planning_revision"),
                "ipfs_accelerate_py/mcplusplus": source.get("mcp_plus_plus_planning_revision"),
            },
        }
        if controller != expected_controller:
            errors.append("prerequisite controller binding differs from scheduler source pins")
    governor = receipt.get("semantic_compression_governor")
    if not isinstance(governor, Mapping):
        errors.append("SCG release evidence is absent")
    else:
        if not _is_full_hex(governor.get("observed_commit")) or not _is_full_hex(governor.get("observed_datasets_commit")):
            errors.append("SCG release commits are not exact 40-hex identities")
        if governor.get("terminal_receipt_valid") is not True:
            errors.append("SCG terminal receipt is not valid")
        if governor.get("disposition") not in {"released", "verified_complete", "terminal_completed"}:
            errors.append("SCG disposition is not terminal/released")
    sealer = receipt.get("incremental_proof_sealer")
    if not isinstance(sealer, Mapping):
        errors.append("IncrementalProofSealer release evidence is absent")
    else:
        if not _is_full_hex(sealer.get("observed_commit")):
            errors.append("IncrementalProofSealer commit is not an exact 40-hex identity")
        for field in ("public_full_checkpoint_api_available", "public_delta_seal_api_available", "terminal_receipt_valid"):
            if sealer.get(field) is not True:
                errors.append(f"IncrementalProofSealer {field} is not true")
        if sealer.get("disposition") not in {"released", "verified_complete", "terminal_completed"}:
            errors.append("IncrementalProofSealer disposition is not terminal/released")
    baseline = receipt.get("baseline")
    if not isinstance(baseline, Mapping) or set(baseline) != {"datasets", "accelerate", "ipfs_kit_py", "mcp_plus_plus"}:
        errors.append("prerequisite focused baseline population differs")
    else:
        for name in ("datasets", "accelerate", "ipfs_kit_py", "mcp_plus_plus"):
            entry = baseline.get(name)
            if not isinstance(entry, Mapping):
                errors.append(f"{name} baseline is absent")
                continue
            if not isinstance(entry.get("passed"), int) or int(entry.get("passed") or 0) <= 0:
                errors.append(f"{name} baseline has no positive pass count")
            if entry.get("failed") != 0:
                errors.append(f"{name} focused baseline is not green")
            if entry.get("known_failure"):
                errors.append(f"{name} baseline still carries a known failure")
    requirements = receipt.get("completion_requirements")
    if not isinstance(requirements, list) or len(requirements) < 5:
        errors.append("prerequisite completion requirements are incomplete")
    else:
        joined = _normalized(" ".join(str(value) for value in requirements))
        for term in ("scg lifecycle", "full-checkpoint and delta", "clean recursive repository forest", "focused baselines", "operator review"):
            if _normalized(term) not in joined:
                errors.append(f"prerequisite completion requirements omit {term!r}")
    if receipt.get("worker_may_complete") is not False:
        errors.append("workers may not complete the prerequisite receipt")
    if receipt.get("runtime_and_sealing_authorized") is not True:
        errors.append("runtime and sealing are not authorized")
    identity = receipt.get("canonical_identity")
    if not isinstance(identity, str) or re.fullmatch(r"b[a-z2-7]{20,}", identity) is None:
        errors.append("prerequisite receipt lacks a strict non-pseudo canonical identity")
    provenance = _normalized(str(receipt.get("provenance") or ""))
    if "operator" not in provenance or "not release evidence" in provenance:
        errors.append("prerequisite provenance is not affirmative operator release evidence")
    if check_repository and isinstance(source, Mapping):
        _validate_repository_pins(repo_root, scheduler, errors)
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-prerequisite-validation@1",
        "valid": not errors,
        "errors": errors,
        "task_id": OPERATOR_GATE,
        "receipt_status": str(receipt.get("status") or "missing"),
        "runtime_and_sealing_authorized": receipt.get("runtime_and_sealing_authorized") is True,
    }


def _validate_blocked_prerequisite_state(
    prerequisite: Mapping[str, object],
    *,
    release_report: Mapping[str, object],
    errors: list[str],
) -> None:
    status = prerequisite.get("status")
    if status == "completed":
        if release_report.get("valid") is not True:
            errors.append("completed prerequisite receipt does not satisfy release validation")
        return
    if status != "blocked":
        errors.append("prerequisite receipt status must be blocked or completed")
    if prerequisite.get("worker_may_complete") is not False:
        errors.append("blocked prerequisite receipt permits worker completion")
    if prerequisite.get("runtime_and_sealing_authorized") is not False:
        errors.append("blocked prerequisite receipt authorizes runtime or sealing")
    if prerequisite.get("canonical_identity") is not None:
        errors.append("blocked prerequisite receipt must not claim canonical release identity")
    if prerequisite.get("observation_identity_scope") != "entire blocked observation excluding observation_identity":
        errors.append("blocked prerequisite observation identity scope differs")
    observation_identity = prerequisite.get("observation_identity")
    if not isinstance(observation_identity, str) or re.fullmatch(r"b[a-z2-7]{20,}", observation_identity) is None:
        errors.append("blocked prerequisite observation lacks a strict non-pseudo content identity")
    governor = prerequisite.get("semantic_compression_governor")
    sealer = prerequisite.get("incremental_proof_sealer")
    if not isinstance(governor, Mapping) or governor.get("terminal_receipt_valid") is not False:
        errors.append("blocked prerequisite SCG evidence is inconsistent")
    if not isinstance(sealer, Mapping):
        errors.append("blocked prerequisite sealer evidence is absent")
    else:
        for field in ("public_full_checkpoint_api_available", "public_delta_seal_api_available", "terminal_receipt_valid"):
            if sealer.get(field) is not False:
                errors.append(f"blocked prerequisite sealer {field} must be false")


def validate(
    repo_root: Path = REPO_ROOT,
    *,
    check_repository: bool = True,
) -> dict[str, object]:
    """Validate the immutable board controls; a truthful blocked gate is valid."""

    root = Path(repo_root).resolve()
    errors: list[str] = []
    primary_paths = (PLAN_REL, OBJECTIVES_REL, TODO_REL, SCHEDULER_REL, PREREQUISITES_REL, LAUNCHER_REL)
    for relative in primary_paths:
        if not (root / relative).is_file():
            errors.append(f"required control is missing: {relative}")
    if errors:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-board-validation@1",
            "valid": False,
            "errors": errors,
            "warnings": [],
        }
    try:
        plan_text = (root / PLAN_REL).read_text(encoding="utf-8")
        objectives_text = (root / OBJECTIVES_REL).read_text(encoding="utf-8")
        todo_text = (root / TODO_REL).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"Markdown controls are unreadable: {type(exc).__name__}: {exc}")
        plan_text = objectives_text = todo_text = ""
    scheduler = _load_json(root / SCHEDULER_REL, errors)
    prerequisite = _load_json(root / PREREQUISITES_REL, errors)
    _validate_plan(plan_text, todo_text, errors)
    _validate_goals(objectives_text, errors)
    _validate_scheduler(scheduler, repo_root=root, check_repository=check_repository, errors=errors)
    _validate_tasks(todo_text, scheduler, prerequisite, errors)
    release_report = _prerequisite_release_report(root, check_repository=check_repository)
    _validate_blocked_prerequisite_state(prerequisite, release_report=release_report, errors=errors)
    if check_repository:
        _validate_repository_pins(root, scheduler, errors)
    errors = list(dict.fromkeys(errors))
    warnings = []
    if prerequisite.get("status") == "blocked":
        warnings.append("AAE-006 remains operator-blocked; runtime and sealing tasks are not authorized")
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-board-validation@1",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "task_count": len(TASK_IDS),
        "goal_count": len(GOAL_IDS),
        "initial_completed_task_ids": list(INITIAL_COMPLETED),
        "initial_ready_task_ids": list(INITIAL_READY),
        "initial_blocked_task_ids": list(INITIAL_BLOCKED),
        "terminal_task_id": TERMINAL_TASK,
        "operator_gate": {
            "task_id": OPERATOR_GATE,
            "receipt_status": prerequisite.get("status"),
            "release_valid": release_report.get("valid") is True,
        },
    }


def validate_prerequisites(
    repo_root: Path = REPO_ROOT,
    *,
    check_repository: bool = True,
) -> dict[str, object]:
    """Validate genuine operator completion of the upstream release gate."""

    return _prerequisite_release_report(Path(repo_root).resolve(), check_repository=check_repository)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-all", action="store_true")
    group.add_argument("--check-prerequisites", action="store_true")
    arguments = parser.parse_args(argv)
    report = validate_prerequisites() if arguments.check_prerequisites else validate()
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
