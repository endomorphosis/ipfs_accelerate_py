#!/usr/bin/env python3
"""Fail-closed validator for AAE controls and canonical release evidence.

Board parsing is stdlib-only.  Release completion deliberately calls the
existing datasets CID and accelerate did:key verification authorities rather
than implementing either identity mechanism here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

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
DATASETS_REVISION = "3af23d9d20e671d2e5260c3509623dba6dc29486"
KIT_REVISION = "523fc9b3d6f1014751428c4a90be4cfc3c871adf"
MCP_REVISION = "96238cc9a86e69d224ab7b52d211a79ecf27b382"
OPERATOR_AUTHORITY_DID = "did:key:z6Mku1TT7TcoD2VksFwNmYGNpE1zprQMmXsT3tz39BzhVdsy"
PREREQUISITE_EVIDENCE_PREFIX = (
    "docs/architecture/adversarial_assurance_inventory/prerequisite_evidence"
)
SEALER_RELEASE_RECEIPT_REL = (
    "artifacts/agent_supervisor/incremental_proof_sealer/release_validation.json"
)
FOCUSED_BASELINE_RUNNER = (
    "python3",
    "docs/architecture/adversarial_assurance_inventory/run_focused_baselines.py",
    "--verify-bundle",
    "--output-dir",
    PREREQUISITE_EVIDENCE_PREFIX,
)

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
    "monotonic pin generation", "configured `did:key` signature", "explicit no-change outcome",
    "strictly increasing, single-use launch generation", "exact controller HEAD",
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
            for term in (
                "only the configured operator did:key",
                "workers cannot",
                "released checkpoint/delta sealer api bindings",
                "every copied evidence cid",
                "verifies the signature",
            ):
                if _normalized(term) not in doctrine:
                    errors.append(f"AAE-006 operator doctrine omits {term!r}")
        if record.record_id == "AAE-013":
            doctrine = _normalized(
                fields.get("acceptance", "")
                + " "
                + fields.get("conflict policy", "")
                + " "
                + fields.get("output policy", "")
            )
            for term in (
                "conditional",
                "no mcp++ change",
                "existing python, go, rust, and typescript harnesses",
            ):
                if _normalized(term) not in doctrine:
                    errors.append(f"AAE-013 scoped MCP++ doctrine omits {term!r}")
        if record.record_id in {"AAE-034", "AAE-035", "AAE-036", "AAE-062"}:
            doctrine = _normalized(fields.get("acceptance", ""))
            if _normalized("before persistence") not in doctrine:
                errors.append(f"{record.record_id} signature doctrine does not reject before persistence")
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
    prerequisite: Mapping[str, object],
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
        "pin_state": "bootstrap_blocked",
        "pin_generation": 0,
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
    if scheduler.get("bootstrap_source_binding") != expected_source:
        errors.append("scheduler immutable bootstrap_source_binding differs")
    if not isinstance(source, Mapping):
        errors.append("scheduler source_binding is absent")
    elif prerequisite.get("status") == "blocked":
        if source != expected_source:
            errors.append("blocked scheduler source_binding differs from bootstrap pins")
    else:
        immutable_source = dict(expected_source)
        for mutable in (
            "pin_state", "pin_generation", "ipfs_datasets_planning_revision",
            "ipfs_kit_planning_revision", "mcp_plus_plus_planning_revision",
        ):
            immutable_source.pop(mutable)
        for field, expected in immutable_source.items():
            if source.get(field) != expected:
                errors.append(f"released scheduler source_binding {field} differs")
        if source.get("pin_state") != "operator_released":
            errors.append("released scheduler source_binding is not operator_released")
        _positive_int(
            source.get("pin_generation"),
            noun="released scheduler pin_generation",
            errors=errors,
        )
        for field in (
            "ipfs_datasets_planning_revision", "ipfs_kit_planning_revision",
            "mcp_plus_plus_planning_revision",
        ):
            if not _is_full_hex(source.get(field)):
                errors.append(f"released scheduler {field} is not exact 40-hex")
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
        "operator_authority_did": OPERATOR_AUTHORITY_DID,
        "runtime_tasks": ["AAE-039", "AAE-043", "AAE-053", "AAE-062"],
        "required_evidence": [
            "genuine terminal SemanticCompressionGovernor receipt and final commit",
            "released full-checkpoint and delta IncrementalProofSealer APIs",
            "clean exact recursive repository forest",
            "fresh focused baselines at the released pins",
            "recomputed canonical identities for copied upstream and baseline evidence",
            "verified operator did:key signature over the release receipt identity",
            "single-use signed exact-HEAD launch admission after gate completion",
        ],
        "missing_capability_disposition": "typed_unavailable",
        "worker_may_complete": False,
    }
    if gate != expected_gate:
        errors.append("scheduler prerequisite gate differs")
    expected_launch_admission = {
        "required_after_prerequisite_completion": True,
        "admission_schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "adversarial-assurance-launch-admission@1"
        ),
        "external_path_environment": "IPFS_ACCELERATE_AAE_LAUNCH_ADMISSION_PATH",
        "must_be_outside_repository": True,
        "bind_exact_controller_head": True,
        "bind_prerequisite_receipt_and_gitlinks": True,
        "strictly_increasing_single_use_generation": True,
        "chained_ledger_scope": (
            "git-common-dir/agent-supervisor/adversarial-assurance-engine-v1"
        ),
        "consume_under_lifecycle_lock_before_spawn": True,
        "dry_run_consumes": False,
    }
    if scheduler.get("launch_admission_policy") != expected_launch_admission:
        errors.append("scheduler launch admission policy differs")
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
    bootstrap = scheduler.get("bootstrap_source_binding")
    if not isinstance(source, Mapping) or not isinstance(bootstrap, Mapping):
        return
    bindings = (
        ("ipfs_datasets_py", "ipfs_datasets_planning_revision"),
        ("ipfs_kit_py", "ipfs_kit_planning_revision"),
        ("ipfs_accelerate_py/mcplusplus", "mcp_plus_plus_planning_revision"),
    )
    for path, revision_field in bindings:
        expected = str(source.get(revision_field) or "")
        bootstrap_revision = str(bootstrap.get(revision_field) or "")
        gitlink = _run_git(repo_root, ("rev-parse", f"HEAD:{path}"), errors)
        nested_head = _run_git(repo_root / path, ("rev-parse", "HEAD"), errors)
        if gitlink and nested_head and gitlink != nested_head:
            errors.append(f"{path} gitlink differs from nested HEAD")
        if expected and nested_head:
            _run_git(repo_root / path, ("merge-base", "--is-ancestor", expected, nested_head), errors)
        if bootstrap_revision and expected:
            _run_git(
                repo_root / path,
                ("merge-base", "--is-ancestor", bootstrap_revision, expected),
                errors,
            )
        status = _run_git(repo_root / path, ("status", "--porcelain"), errors)
        if status:
            errors.append(f"{path} nested worktree is not clean")


def _is_full_hex(value: object) -> bool:
    return re.fullmatch(r"[0-9a-f]{40}", str(value or "")) is not None


def _canonical_cid(
    repo_root: Path,
    value: object,
    *,
    noun: str,
    errors: list[str],
) -> str:
    """Use the pinned datasets CID authority without copying its profile."""

    module_root = repo_root / "ipfs_datasets_py"
    inserted = False
    try:
        if not module_root.is_dir():
            raise FileNotFoundError(module_root)
        module_text = str(module_root)
        if module_text not in sys.path:
            sys.path.insert(0, module_text)
            inserted = True
        module = importlib.import_module(
            "ipfs_datasets_py.logic.software_contracts.content"
        )
        authority_path = Path(str(module.__file__ or "")).resolve()
        if module_root.resolve() not in authority_path.parents:
            raise RuntimeError("content authority was imported from another tree")
        return str(module.cid_for_obj(value))
    except Exception as exc:  # fail closed across optional dependency failures
        errors.append(f"{noun} canonical identity could not be recomputed: {type(exc).__name__}")
        return ""
    finally:
        if inserted:
            try:
                sys.path.remove(str(module_root))
            except ValueError:
                pass


def _verify_operator_signature(
    repo_root: Path,
    *,
    identity_did: str,
    payload: Mapping[str, object],
    signature: str,
    errors: list[str],
) -> None:
    """Use the existing supervisor did:key verifier and configured authority."""

    inserted = False
    root_text = str(repo_root)
    try:
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
            inserted = True
        module = importlib.import_module(
            "ipfs_accelerate_py.agent_supervisor.control.profile_authority"
        )
        authority_path = Path(str(module.__file__ or "")).resolve()
        if repo_root.resolve() not in authority_path.parents:
            raise RuntimeError("signature authority was imported from another tree")
        module.verify_did_key_signature(
            identity_did=identity_did,
            payload=payload,
            signature=signature,
        )
    except Exception as exc:  # includes malformed/invalid signatures
        errors.append(f"operator release signature verification failed: {type(exc).__name__}")
    finally:
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass


def _bound_json_artifact(
    repo_root: Path,
    specification: object,
    *,
    noun: str,
    allowed_paths: Sequence[str] = (PREREQUISITE_EVIDENCE_PREFIX,),
    errors: list[str],
) -> dict[str, object]:
    if not isinstance(specification, Mapping):
        errors.append(f"{noun} evidence binding is absent")
        return {}
    if set(specification) != {"path", "canonical_identity"}:
        errors.append(f"{noun} evidence binding fields differ")
    relative = str(specification.get("path") or "")
    identity = str(specification.get("canonical_identity") or "")
    path_errors: list[str] = []
    _safe_paths((relative,), noun=f"{noun} evidence path", errors=path_errors)
    if path_errors or not _path_in_prefixes(relative, allowed_paths):
        errors.extend(path_errors or [f"{noun} evidence path leaves the prerequisite evidence root"])
        return {}
    candidate = repo_root / PurePosixPath(relative)
    cursor = repo_root
    try:
        for component in PurePosixPath(relative).parts:
            cursor /= component
            if cursor.is_symlink():
                raise ValueError("symlink component")
        if not candidate.is_file() or candidate.stat().st_size > 1024 * 1024:
            raise ValueError("missing, non-file, or oversized artifact")
    except OSError as exc:
        errors.append(f"{noun} evidence artifact is unavailable: {type(exc).__name__}")
        return {}
    except ValueError as exc:
        errors.append(f"{noun} evidence artifact is invalid: {exc}")
        return {}
    payload = _load_json(candidate, errors)
    recomputed = _canonical_cid(repo_root, payload, noun=noun, errors=errors)
    if not identity or recomputed != identity:
        errors.append(f"{noun} evidence canonical identity differs")
    return payload


def _validate_scg_terminal_evidence(
    lifecycle: Mapping[str, object],
    terminal: Mapping[str, object],
    governor: Mapping[str, object],
    errors: list[str],
) -> None:
    if lifecycle.get("schema") != "ipfs_accelerate_py/agent-supervisor/semantic-compression-governor-lifecycle@1":
        errors.append("SCG lifecycle evidence schema differs")
    if terminal.get("schema") != "ipfs_accelerate_py/agent-supervisor/semantic-compression-governor-terminal@1":
        errors.append("SCG terminal evidence schema differs")
    plan = lifecycle.get("plan")
    profile = lifecycle.get("profile")
    if not isinstance(plan, Mapping) or not isinstance(profile, Mapping):
        errors.append("SCG lifecycle plan/profile binding is absent")
        return
    if plan.get("source_head") != governor.get("terminal_launch_commit"):
        errors.append("SCG terminal source head differs from its launch commit")
    if plan.get("expected_task_count") != 49 or terminal.get("expected_task_count") != 49:
        errors.append("SCG terminal task population differs")
    for field in ("run_id", "profile_id", "configuration_root"):
        if terminal.get(field) != profile.get(field):
            errors.append(f"SCG terminal {field} differs from lifecycle profile")
    if plan.get("configuration_root") != profile.get("configuration_root"):
        errors.append("SCG lifecycle plan/profile configuration roots differ")
    if terminal.get("drained") is not True:
        errors.append("SCG terminal receipt is not drained")
    lanes = terminal.get("lane_evidence")
    if not isinstance(lanes, list) or len(lanes) != 3:
        errors.append("SCG terminal lane population differs")
        return
    for lane in lanes:
        if not isinstance(lane, Mapping):
            errors.append("SCG terminal lane evidence is malformed")
            continue
        if (
            lane.get("terminal") is not True
            or lane.get("completed_count") != 49
            or lane.get("blocked_count") != 0
            or lane.get("ready_count") != 0
            or lane.get("waiting_count") != 0
            or lane.get("active_task_id") not in {None, ""}
        ):
            errors.append("SCG terminal lane does not prove a drained 49/49 board")


def _validate_baseline_receipt(
    name: str,
    payload: Mapping[str, object],
    expected_state: str,
    errors: list[str],
) -> None:
    expected_keys = {
        "schema", "runner_id", "repository", "repository_state_root",
        "started_at", "finished_at", "duration_ns", "command_argv",
        "returncode", "terminal_status", "passed", "failed", "skipped",
        "environment_identity", "dependency_lock_identity",
        "bounded_log_digest", "network_access", "production_credentials_available",
    }
    if set(payload) != expected_keys:
        errors.append(f"{name} baseline fields differ")
    if payload.get("schema") != "ipfs_accelerate_py/adversarial-assurance/focused-baseline-receipt@1":
        errors.append(f"{name} baseline schema differs")
    if payload.get("runner_id") != "protected-aae-focused-baseline-runner@1":
        errors.append(f"{name} baseline runner differs")
    if payload.get("repository") != name:
        errors.append(f"{name} baseline repository identity differs")
    if payload.get("repository_state_root") != expected_state:
        errors.append(f"{name} baseline repository-state root differs")
    timestamps: list[datetime] = []
    for field in ("started_at", "finished_at"):
        raw = str(payload.get(field) or "")
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            errors.append(f"{name} baseline {field} is not canonical UTC")
        else:
            timestamps.append(parsed)
    if len(timestamps) == 2 and timestamps[1] < timestamps[0]:
        errors.append(f"{name} baseline time interval is reversed")
    if not isinstance(payload.get("duration_ns"), int) or int(payload.get("duration_ns") or 0) <= 0:
        errors.append(f"{name} baseline duration is not positive")
    argv = payload.get("command_argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        errors.append(f"{name} baseline argv is absent or malformed")
    if payload.get("returncode") != 0 or payload.get("terminal_status") != "passed":
        errors.append(f"{name} baseline is not terminal passed")
    if not isinstance(payload.get("passed"), int) or int(payload.get("passed") or 0) <= 0:
        errors.append(f"{name} baseline has no positive pass count")
    if payload.get("failed") != 0:
        errors.append(f"{name} baseline is not green")
    if not isinstance(payload.get("skipped"), int) or int(payload.get("skipped") or 0) < 0:
        errors.append(f"{name} baseline skipped count is invalid")
    for field in ("environment_identity", "dependency_lock_identity"):
        if re.fullmatch(r"b[a-z2-7]{20,}", str(payload.get(field) or "")) is None:
            errors.append(f"{name} baseline {field} is not canonical")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", str(payload.get("bounded_log_digest") or "")) is None:
        errors.append(f"{name} baseline log digest is invalid")
    if payload.get("network_access") != "disabled":
        errors.append(f"{name} baseline network policy is not disabled")
    if payload.get("production_credentials_available") is not False:
        errors.append(f"{name} baseline exposed production credentials")


def _probe_sealer_api_bindings(
    repo_root: Path,
    bindings: object,
    errors: list[str],
) -> None:
    required = {
        "IncrementalProofSealer", "FullCheckpointSeal", "DeltaSeal",
        "create_full_checkpoint", "publish_full_checkpoint",
        "build_delta_seal", "publish_delta_seal",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != required:
        errors.append("IncrementalProofSealer public API binding population differs")
        return
    inserted = False
    root_text = str(repo_root)
    try:
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
            inserted = True
        for symbol in sorted(required):
            module_name = str(bindings.get(symbol) or "")
            if not module_name.startswith("ipfs_accelerate_py.agent_supervisor."):
                errors.append(f"IncrementalProofSealer {symbol} module leaves canonical package")
                continue
            try:
                module = importlib.import_module(module_name)
                value = getattr(module, symbol)
            except Exception as exc:
                errors.append(f"IncrementalProofSealer {symbol} import failed: {type(exc).__name__}")
                continue
            module_path = Path(str(module.__file__ or "")).resolve()
            if repo_root.resolve() not in module_path.parents or not callable(value):
                errors.append(f"IncrementalProofSealer {symbol} is not a current-tree callable")
    finally:
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass


def _run_release_probe(
    repo_root: Path,
    argv: object,
    *,
    expected: Sequence[str],
    noun: str,
    errors: list[str],
) -> dict[str, object]:
    if argv != list(expected):
        errors.append(f"{noun} release probe argv differs")
        return {}
    try:
        result = subprocess.run(
            list(expected), cwd=repo_root, text=True, capture_output=True,
            check=False, timeout=3600.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"{noun} release probe failed: {type(exc).__name__}")
        return {}
    if result.returncode != 0:
        errors.append(f"{noun} release probe rejected: {result.returncode}")
        return {}
    if len(result.stdout.encode("utf-8")) > 1024 * 1024:
        errors.append(f"{noun} release probe output exceeds one MiB")
        return {}
    try:
        report = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        errors.append(f"{noun} release probe did not emit one JSON report")
        return {}
    if not isinstance(report, dict):
        errors.append(f"{noun} release probe report is not an object")
        return {}
    if report.get("valid") is not True or report.get("errors") != []:
        errors.append(f"{noun} release probe report is not valid")
    return report


def _prerequisite_release_report(
    repo_root: Path,
    *,
    check_repository: bool,
    execute_release_probes: bool = False,
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
    expected_receipt_fields = {
        "schema", "task_id", "status", "observed_at", "controller",
        "semantic_compression_governor", "incremental_proof_sealer",
        "evidence_artifacts", "completion_requirements", "worker_may_complete",
        "runtime_and_sealing_authorized", "canonical_identity_scope",
        "canonical_identity", "authorization", "provenance",
        "baseline_qualification_argv",
    }
    if receipt.get("status") == "completed" and set(receipt) != expected_receipt_fields:
        errors.append("completed prerequisite receipt fields differ")
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
    gate = scheduler.get("prerequisite_gate")
    controller = receipt.get("controller")
    release_commit = ""
    release_gitlinks: dict[str, object] = {}
    pin_generation = -1
    if not isinstance(source, Mapping) or not isinstance(controller, Mapping):
        errors.append("prerequisite controller/source binding is absent")
    else:
        release_commit = str(controller.get("release_commit") or "")
        release_gitlinks = {
            "ipfs_datasets_py": source.get("ipfs_datasets_planning_revision"),
            "ipfs_kit_py": source.get("ipfs_kit_planning_revision"),
            "ipfs_accelerate_py/mcplusplus": source.get("mcp_plus_plus_planning_revision"),
        }
        try:
            pin_generation = int(source.get("pin_generation"))
        except (TypeError, ValueError):
            pin_generation = -1
        expected_controller = {
            "repository": "endomorphosis/ipfs_accelerate_py",
            "branch": source.get("accelerator_required_branch"),
            "required_ancestor": source.get("accelerator_required_ancestor"),
            "pin_generation": pin_generation,
            "release_commit": release_commit,
            "release_gitlinks": release_gitlinks,
        }
        if controller != expected_controller:
            errors.append("prerequisite controller binding differs from active release pins")
        if source.get("pin_state") != "operator_released" or pin_generation <= 0:
            errors.append("prerequisite source pins were not operator-released")
        if not _is_full_hex(release_commit):
            errors.append("prerequisite release commit is not exact 40-hex")

    sealer_probe_report: dict[str, object] = {}
    baseline_probe_report: dict[str, object] = {}
    if execute_release_probes and check_repository and receipt.get("status") == "completed":
        raw_sealer = receipt.get("incremental_proof_sealer")
        sealer_argv = raw_sealer.get("qualification_argv") if isinstance(raw_sealer, Mapping) else None
        sealer_probe_report = _run_release_probe(
            repo_root,
            sealer_argv,
            expected=("python3", "scripts/validate_incremental_proof_sealer_board.py", "--run-release-validation"),
            noun="IncrementalProofSealer",
            errors=errors,
        )
        baseline_probe_report = _run_release_probe(
            repo_root,
            receipt.get("baseline_qualification_argv"),
            expected=FOCUSED_BASELINE_RUNNER,
            noun="focused baseline",
            errors=errors,
        )

    evidence_specs = receipt.get("evidence_artifacts")
    evidence_keys = {
        "scg_lifecycle", "scg_terminal", "incremental_proof_sealer_release",
        "datasets_baseline", "accelerate_baseline", "ipfs_kit_py_baseline",
        "mcp_plus_plus_baseline",
    }
    evidence: dict[str, dict[str, object]] = {}
    if not isinstance(evidence_specs, Mapping) or set(evidence_specs) != evidence_keys:
        errors.append("prerequisite evidence artifact population differs")
    else:
        for name in sorted(evidence_keys):
            allowed_paths = (
                (SEALER_RELEASE_RECEIPT_REL,)
                if name == "incremental_proof_sealer_release"
                else (PREREQUISITE_EVIDENCE_PREFIX,)
            )
            evidence[name] = _bound_json_artifact(
                repo_root,
                evidence_specs.get(name),
                noun=name,
                allowed_paths=allowed_paths,
                errors=errors,
            )
            if (
                name == "incremental_proof_sealer_release"
                and isinstance(evidence_specs.get(name), Mapping)
                and evidence_specs[name].get("path") != SEALER_RELEASE_RECEIPT_REL
            ):
                errors.append("IncrementalProofSealer evidence does not use its canonical release receipt")

    if sealer_probe_report and sealer_probe_report.get("runner") != "release":
        errors.append("IncrementalProofSealer release probe runner differs")
    if baseline_probe_report:
        expected_probe_fields = {
            "schema", "valid", "runner", "source_head", "receipt_bindings", "errors",
        }
        if set(baseline_probe_report) != expected_probe_fields:
            errors.append("focused baseline verification report fields differ")
        if (
            baseline_probe_report.get("schema")
            != "ipfs_accelerate_py/adversarial-assurance/focused-baseline-verification@1"
            or baseline_probe_report.get("runner") != "protected-aae-focused-baseline-runner@1"
            or baseline_probe_report.get("source_head") != release_commit
        ):
            errors.append("focused baseline verification report binding differs")
        expected_bindings = {
            name: evidence_specs.get(name)
            for name in (
                "datasets_baseline", "accelerate_baseline",
                "ipfs_kit_py_baseline", "mcp_plus_plus_baseline",
            )
        } if isinstance(evidence_specs, Mapping) else {}
        if baseline_probe_report.get("receipt_bindings") != expected_bindings:
            errors.append("focused baseline probe output differs from signed evidence bindings")

    governor = receipt.get("semantic_compression_governor")
    if not isinstance(governor, Mapping):
        errors.append("SCG release evidence is absent")
        governor = {}
    else:
        if set(governor) != {
            "observed_commit", "terminal_launch_commit",
            "observed_datasets_commit", "disposition",
        }:
            errors.append("SCG release evidence fields differ")
        for field in ("observed_commit", "terminal_launch_commit", "observed_datasets_commit"):
            if not _is_full_hex(governor.get(field)):
                errors.append(f"SCG {field} is not exact 40-hex")
        if governor.get("disposition") not in {"released", "verified_complete", "terminal_completed"}:
            errors.append("SCG disposition is not terminal/released")
    _validate_scg_terminal_evidence(
        evidence.get("scg_lifecycle", {}),
        evidence.get("scg_terminal", {}),
        governor,
        errors,
    )

    sealer = receipt.get("incremental_proof_sealer")
    if not isinstance(sealer, Mapping):
        errors.append("IncrementalProofSealer release evidence is absent")
        sealer = {}
    else:
        if set(sealer) != {"observed_commit", "disposition", "api_bindings", "qualification_argv"}:
            errors.append("IncrementalProofSealer release evidence fields differ")
        if not _is_full_hex(sealer.get("observed_commit")):
            errors.append("IncrementalProofSealer commit is not exact 40-hex")
        if sealer.get("disposition") not in {"released", "verified_complete", "terminal_completed"}:
            errors.append("IncrementalProofSealer disposition is not terminal/released")
        if sealer.get("qualification_argv") != [
            "python3", "scripts/validate_incremental_proof_sealer_board.py",
            "--run-release-validation",
        ]:
            errors.append("IncrementalProofSealer qualification argv differs")
    if receipt.get("baseline_qualification_argv") != list(FOCUSED_BASELINE_RUNNER):
        errors.append("focused baseline qualification argv differs")
    sealer_release = evidence.get("incremental_proof_sealer_release", {})
    if sealer_release.get("schema_version") != "incremental-proof-sealer-release-validation@2":
        errors.append("IncrementalProofSealer release evidence schema differs")
    if sealer_release.get("runner_id") != "protected-board-release-validation-runner@1":
        errors.append("IncrementalProofSealer release runner differs")
    terminal_gate = sealer_release.get("terminal_gate")
    if not isinstance(terminal_gate, Mapping) or (
        terminal_gate.get("id") != "terminal-board-gate"
        or terminal_gate.get("capture_status") != "completed"
        or terminal_gate.get("exit_code") != 0
    ):
        errors.append("IncrementalProofSealer canonical terminal gate is not passed")
    if not _is_full_hex(sealer_release.get("validation_worktree_parent_revision")):
        errors.append("IncrementalProofSealer release parent revision is invalid")
    source_revisions = sealer_release.get("source_revisions")
    if not isinstance(source_revisions, Mapping) or set(source_revisions) != {"accelerate", "datasets", "kit"}:
        errors.append("IncrementalProofSealer release source revision population differs")
    elif any(not _is_full_hex(value) for value in source_revisions.values()):
        errors.append("IncrementalProofSealer release source revisions are invalid")
    declared_sealer_digest = str(sealer_release.get("receipt_digest") or "")
    sealer_body = dict(sealer_release)
    sealer_body.pop("receipt_digest", None)
    try:
        expected_sealer_digest = "sha256:" + hashlib.sha256(
            json.dumps(
                sealer_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        expected_sealer_digest = ""
    if declared_sealer_digest != expected_sealer_digest:
        errors.append("IncrementalProofSealer release digest is invalid")
    _probe_sealer_api_bindings(repo_root, sealer.get("api_bindings"), errors)

    baseline_states = {
        "datasets": str(release_gitlinks.get("ipfs_datasets_py") or ""),
        "accelerate": release_commit,
        "ipfs_kit_py": str(release_gitlinks.get("ipfs_kit_py") or ""),
        "mcp_plus_plus": str(release_gitlinks.get("ipfs_accelerate_py/mcplusplus") or ""),
    }
    for name, state_root in baseline_states.items():
        _validate_baseline_receipt(
            name,
            evidence.get(f"{name}_baseline", {}),
            state_root,
            errors,
        )
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
    identity = str(receipt.get("canonical_identity") or "")
    if receipt.get("canonical_identity_scope") != "entire completed receipt excluding canonical_identity and authorization":
        errors.append("prerequisite receipt canonical identity scope differs")
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("canonical_identity", None)
    unsigned_receipt.pop("authorization", None)
    recomputed_identity = _canonical_cid(
        repo_root, unsigned_receipt, noun="prerequisite receipt", errors=errors
    )
    if not identity or identity != recomputed_identity:
        errors.append("prerequisite receipt canonical identity differs")

    authorization = receipt.get("authorization")
    expected_authority = gate.get("operator_authority_did") if isinstance(gate, Mapping) else None
    if not isinstance(authorization, Mapping):
        errors.append("prerequisite operator authorization is absent")
    else:
        expected_authorization_fields = {
            "schema", "identity_did", "audience", "action", "receipt_cid",
            "pin_generation", "release_commit", "release_gitlinks", "signature",
        }
        if set(authorization) != expected_authorization_fields:
            errors.append("prerequisite operator authorization fields differ")
        if (
            expected_authority != OPERATOR_AUTHORITY_DID
            or authorization.get("identity_did") != expected_authority
            or authorization.get("schema") != "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-prerequisite-authorization@1"
            or authorization.get("audience") != BOARD_NAMESPACE
            or authorization.get("action") != "complete:AAE-006"
            or authorization.get("receipt_cid") != identity
            or authorization.get("pin_generation") != pin_generation
            or authorization.get("release_commit") != release_commit
            or authorization.get("release_gitlinks") != release_gitlinks
        ):
            errors.append("prerequisite operator authorization bindings differ")
        signature_payload = dict(authorization)
        signature = str(signature_payload.pop("signature", ""))
        _verify_operator_signature(
            repo_root,
            identity_did=str(authorization.get("identity_did") or ""),
            payload=signature_payload,
            signature=signature,
            errors=errors,
        )
    provenance = _normalized(str(receipt.get("provenance") or ""))
    if "operator" not in provenance or "not release evidence" in provenance:
        errors.append("prerequisite provenance is not affirmative operator release evidence")
    if check_repository and isinstance(source, Mapping):
        _validate_repository_pins(repo_root, scheduler, errors)
        for noun, revision in (
            ("controller release", release_commit),
            ("SCG release", str(governor.get("observed_commit") or "")),
            ("IncrementalProofSealer release", str(sealer.get("observed_commit") or "")),
        ):
            if _is_full_hex(revision):
                before = len(errors)
                _run_git(repo_root, ("merge-base", "--is-ancestor", revision, "HEAD"), errors)
                if len(errors) > before:
                    errors.append(f"{noun} commit is not integrated into the AAE controller")
        launch_commit = str(governor.get("terminal_launch_commit") or "")
        final_scg_commit = str(governor.get("observed_commit") or "")
        if _is_full_hex(launch_commit) and _is_full_hex(final_scg_commit):
            _run_git(repo_root, ("merge-base", "--is-ancestor", launch_commit, final_scg_commit), errors)
        scg_datasets_commit = str(governor.get("observed_datasets_commit") or "")
        active_datasets_commit = str(release_gitlinks.get("ipfs_datasets_py") or "")
        if _is_full_hex(scg_datasets_commit) and _is_full_hex(active_datasets_commit):
            _run_git(
                repo_root / "ipfs_datasets_py",
                ("merge-base", "--is-ancestor", scg_datasets_commit, active_datasets_commit),
                errors,
            )
        if isinstance(source_revisions, Mapping):
            sealer_targets = {
                "accelerate": str(sealer.get("observed_commit") or ""),
                "datasets": str(release_gitlinks.get("ipfs_datasets_py") or ""),
                "kit": str(release_gitlinks.get("ipfs_kit_py") or ""),
            }
            for repository, target in sealer_targets.items():
                revision = str(source_revisions.get(repository) or "")
                git_root = repo_root if repository == "accelerate" else repo_root / (
                    "ipfs_datasets_py" if repository == "datasets" else "ipfs_kit_py"
                )
                if _is_full_hex(revision) and _is_full_hex(target):
                    _run_git(git_root, ("merge-base", "--is-ancestor", revision, target), errors)
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
    repo_root: Path,
    check_repository: bool,
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
    if check_repository:
        observation = dict(prerequisite)
        observation.pop("observation_identity", None)
        recomputed = _canonical_cid(
            repo_root,
            observation,
            noun="blocked prerequisite observation",
            errors=errors,
        )
        if observation_identity != recomputed:
            errors.append("blocked prerequisite observation identity differs")
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
    _validate_scheduler(
        scheduler,
        prerequisite=prerequisite,
        repo_root=root,
        check_repository=check_repository,
        errors=errors,
    )
    _validate_tasks(todo_text, scheduler, prerequisite, errors)
    release_report = _prerequisite_release_report(root, check_repository=check_repository)
    _validate_blocked_prerequisite_state(
        prerequisite,
        repo_root=root,
        check_repository=check_repository,
        release_report=release_report,
        errors=errors,
    )
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

    return _prerequisite_release_report(
        Path(repo_root).resolve(),
        check_repository=check_repository,
        execute_release_probes=True,
    )


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
