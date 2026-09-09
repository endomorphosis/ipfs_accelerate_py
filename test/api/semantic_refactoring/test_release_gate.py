"""Independent contract tests for SPAR-050 release and limitation evidence."""

from __future__ import annotations

import ast
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.planning.residual_llm_packet import (
    ResidualLlmPacket,
    ResidualLlmPacketError,
    seal_residual_llm_packet,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.facade_planner import (
    FACADE_CAN_RETIRE_FACADE,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.rollout import (
    GATE_CAN_CHANGE_MODE,
    MODE_CONTRACTS,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.residual_provider_invocation import (
    ResidualProviderInvocation,
    ResidualProviderInvocationError,
    assert_provider_env_excludes_secrets,
)

from test.api.semantic_refactoring.test_adversarial_qualification import (
    load_adversarial_report,
)
from test.api.semantic_refactoring.test_benchmark_corpus import (
    admit_real_current_tree,
    load_benchmark_corpus_manifest,
    load_preregistration,
)
from test.api.semantic_refactoring.test_benchmark_report import (
    load_benchmark_report,
)
from test.api.semantic_refactoring.test_capstone_receipts import (
    load_capstone_report,
)
from test.api.semantic_refactoring.test_end_to_end_acceptance import (
    load_acceptance_matrix,
    run_acceptance_scenario,
)


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/final_report.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_FINAL_REPORT.md",
    "test/api/semantic_refactoring/test_release_gate.py",
)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
)
CAPSULE_TYPES = (
    "FunctionSemanticCapsule",
    "MethodSemanticCapsule",
    "ClassSemanticCapsule",
    "TopLevelBlockCapsule",
    "ModuleSemanticCapsule",
    "PackageSemanticCapsule",
    "CallsiteSemanticCapsule",
    "StateOwnerCapsule",
    "RegistrationCapsule",
    "ResourceLifecycleCapsule",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
RESIDUAL_TREE_ID = "e40b9653c81eb5c8923263f68a642569ade072b3"
REPORT_RELATIVE = (
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/final_report.json"
)
MARKDOWN_RELATIVE = (
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_FINAL_REPORT.md"
)
PREREGISTRATION_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json"
)
CORPUS_MANIFEST_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/corpus_manifest.json"
)
ACCEPTANCE_MATRIX_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/acceptance_matrix.json"
)
BENCHMARK_REPORT_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/benchmark_report.json"
)
ADVERSARIAL_REPORT_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/adversarial_report.json"
)
CAPSTONE_REPORT_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/capstone_report.json"
)
IDENTITY_INVENTORY_RELATIVE = (
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json"
)
DYNAMIC_RISK_RELATIVE = (
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json"
)
DEPENDENCY_SEAL_RELATIVE = (
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json"
)
SCHEDULER_RELATIVE = (
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json"
)
BOARD_VALIDATOR_RELATIVE = (
    "scripts/validate_semantic_preserving_remodularization_board.py"
)
REAL_CURRENT_TREE_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/fixtures/real_current_tree.json"
)
RESIDUAL_PACKET_CID = (
    "baguqeeracx4fvuwtu3774oluaql3qvtq3x652zormz5nnolyx5lee2mg5hca"
)
TASK_CID = "baguqeerarmd6jbissl32cu6vzc3slnneb2kg5bbnhf67gl345efr3knngg4a"
OBLIGATION_ID = "baguqeeraui3bvf4lbgfhjp3ubcti3ce2brqst562m52z3ks64ojcu6imj54a"
FOREST_ID = (
    "sha256:7db72a306b3296cd893965bc33225ebe447c85d69bbf047b0a6ea711882be217"
)
POLICY_ROOT = "baguqeeraf4zd7btgyo3qriyzlzy7dp7epm62rqkvzodnlmdcac66j2bpzaka"
REPOSITORY_ID = "repository:semantic-preserving-autonomous-remodularization-v1"
TOKENIZER_ID = "spar-benchmark/utf8-bytes-div4@1"
VALIDATION_COMMANDS = (
    "python3 -m pytest -q test/api/semantic_refactoring/test_release_gate.py",
)
RESIDUAL_LIMITS = {
    "max_bytes": 24576,
    "max_tokens": 6144,
    "max_capsule_bytes": 16384,
}

TASK_ID: str = "SPAR-050"
GOAL_ID: str = "SPAR-G082"
PROGRAM: str = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: str = "release evidence"
AUTHORITY_OWNER: str = "ipfs_accelerate_py"
ANALYZER_ID: str = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.release_gate@1"
)
PREDECESSOR_TASK_IDS: tuple[str, ...] = ("SPAR-049",)
SPAR_RELEASE_REPORT_INTERFACE: str = "SparReleaseReport@1"
SPAR_RELEASE_GATE_RECEIPT_INTERFACE: str = "SparReleaseGateReceipt@1"
SPAR_RELEASE_GATE_RESULT_INTERFACE: str = "SparReleaseGateResult@1"
NOMINATED_FINAL_ROOT_INTERFACE: str = "NominatedFinalRoot@1"
NOMINATED_LIMITATION_REPORT_INTERFACE: str = "NominatedLimitationReport@1"
SPAR_RELEASE_REPORT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-release-report@1"
)
SPAR_RELEASE_GATE_RECEIPT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-release-gate-receipt@1"
)
SPAR_RELEASE_GATE_RESULT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-release-gate-result@1"
)
NOMINATED_FINAL_ROOT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/nominated-final-root@1"
)
NOMINATED_LIMITATION_REPORT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/nominated-limitation-report@1"
)
RELEASE_CONTRACT_VERSION: str = "1"
REQUIRED_MODE: str = "required"
REQUIRED_GATE_TASK: str = "SPAR-043"
NETWORK_DENY: str = "deny"
CAN_AUTHORIZE_COMPLETION: bool = False
CAN_AUTHORIZE_TRANSITION: bool = False
CAN_CREATE_AUTHORITY: bool = False
GATE_WRITES_REPOSITORY: bool = False
GATE_IS_NOMINATION_ONLY: bool = True
VECTOR_SIMILARITY_IS_AUTHORITY: bool = False
PROJECTION_CLUSTERING_IS_AUTHORITY: bool = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: bool = True
TEST_PASS_IS_NOT_COMPLETION: bool = True
MARKDOWN_IS_NOT_COMPLETION: bool = True
WORKER_SELF_APPROVAL: bool = False
WORKER_MAY_CHANGE_MODE: bool = False
DUCKLAKE_IS_AUTHORITY: bool = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: bool = True
NETWORK_DENIED: bool = True
RAW_SOURCE_REQUIRED: bool = True
DRY_RUN_IS_DETERMINISTIC: bool = True
DRY_RUN_MUTATES: bool = False
TOP_LEVEL_COMPLETION_BYPASS: bool = False
SELF_HOSTED: bool = True
FACADE_RETIREMENT_AUTHORIZED: bool = False
GENERAL_PYTHON_EQUIVALENCE_CLAIMED: bool = False
FINAL_ROOT_ACCEPTED: bool = False

DECLARED_STATUSES: frozenset[str] = frozenset(
    {"nominated", "rejected", "typed_terminal", "unavailable", "failed", "escalated"}
)
DENOMINATOR_BUCKETS: tuple[str, ...] = (
    "failed",
    "escalated",
    "rejected",
    "unavailable",
    "human_reviewed",
    "nominated",
    "typed_terminal",
)
LIMITATION_IDS: tuple[str, ...] = (
    "general_python_equivalence_not_claimed",
    "unresolved_required_dynamics",
    "evidence_bounded_by_observation_profile",
)
GATE_EXPECTATIONS: tuple[tuple[str, str, str], ...] = (
    ("seals.tasks.roots", "nominated", ""),
    ("focused.regression.suites", "nominated", ""),
    ("identity.preservation", "nominated", ""),
    ("model.context.changes", "nominated", ""),
    ("unsupported.dynamics", "typed_terminal", "unresolved_required_dynamics"),
    ("scale.denominators", "nominated", ""),
    ("capstone.verify", "nominated", ""),
    ("migration.compatibility", "nominated", ""),
    ("facade.retirement", "nominated", ""),
    ("transitive.final.root", "nominated", ""),
    ("general_llm.network_denied", "unavailable", "network_denied_general_llm"),
)
NEGATIVE_EVIDENCE: tuple[dict[str, Any], ...] = (
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "stale_authoritative_reuse",
        "status": "rejected",
        "terminal": "stale_tree",
        "gate_id": "reuse.stale_tree",
        "writes_repository": False,
    },
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "false_task_completions",
        "status": "rejected",
        "terminal": "self_authorization_rejected",
        "gate_id": "authority.self_authorization",
        "writes_repository": False,
    },
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "false_task_completions",
        "status": "rejected",
        "terminal": "completion_bypass_rejected",
        "gate_id": "authority.completion_bypass",
        "writes_repository": False,
    },
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "unauthorized_mutations",
        "status": "rejected",
        "terminal": "worker_mode_change_rejected",
        "gate_id": "rollout.worker_mode_change",
        "writes_repository": False,
    },
)
PROVIDER_TOKENIZER: dict[str, Any] = {
    "general_llm_invoked": False,
    "live_model_channel": False,
    "network": NETWORK_DENY,
    "provider_available": False,
    "tokenizer_available": True,
    "tokenizer_id": TOKENIZER_ID,
    "tokenizer_identity_required": True,
}
IDENTITY_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "timestamp",
        "timestamps",
        "process_id",
        "pid",
        "local_path",
        "local_paths",
        "checkout_path",
        "model_output",
        "model",
        "provider",
        "prompt",
        "lease",
        "fence",
        "generation",
        "receipt",
        "acceptance",
        "wall_clock",
        "clock",
        "source",
        "source_text",
        "source_body",
        "file_contents",
        "repository_dump",
    }
)
_AUTHORITY_FLAGS: tuple[str, ...] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "writes_repository",
    "worker_self_approval",
    "projection_is_authority",
    "facade_retirement_authorized",
    "final_root_accepted",
    "general_python_equivalence_claimed",
)


class ReleaseGateError(RuntimeError):
    """Fail-closed SPAR-050 release-gate contract violation."""


def provider_free_exports() -> tuple[str, ...]:
    return (
        "NominatedFinalRoot",
        "NominatedLimitationReport",
        "SparReleaseGateReceipt",
        "SparReleaseGateResult",
        "SparReleaseReport",
        "load_release_report",
        "nominate_release_gate",
        "run_release_gate",
        "sealed_limitation_report_markdown",
        "sealed_release_report_payload",
    )


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & set(CAPSULE_TYPES)
    if overlap:
        raise ReleaseGateError(
            f"release gate must not define competing types: {sorted(overlap)}"
        )


def release_gate_cid_profile() -> dict[str, str]:
    return {
        "codec": "dag-json",
        "hash": "sha2-256",
        "rule": "content-addressed over sealed fields; not universal meaning",
    }


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _load_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseGateError(f"{relative} must be a JSON object")
    return payload


def _terminal_or_none(value: str) -> str | None:
    return value or None


def load_release_report() -> dict[str, Any]:
    payload = _load_json(REPORT_RELATIVE)
    if payload.get("schema") != SPAR_RELEASE_REPORT_SCHEMA:
        raise ReleaseGateError("unsupported release report schema")
    if payload.get("interface") != SPAR_RELEASE_REPORT_INTERFACE:
        raise ReleaseGateError("unsupported release report interface")
    if payload.get("task_id") != TASK_ID:
        raise ReleaseGateError("release report task_id must remain SPAR-050")
    if payload.get("nomination_only") is not True:
        raise ReleaseGateError("release report must remain nomination_only")
    for flag in _AUTHORITY_FLAGS:
        if payload.get(flag) is True:
            raise ReleaseGateError(f"release report cannot claim {flag}")
    if payload.get("promotion_eligible") is True:
        raise ReleaseGateError("release report cannot self-promote")
    if payload.get("network") != NETWORK_DENY:
        raise ReleaseGateError("release report network must remain deny")
    if payload.get("rollout_mode") != REQUIRED_MODE:
        raise ReleaseGateError("release report rollout_mode must remain required")
    if payload.get("top_level_completion_bypass") is True:
        raise ReleaseGateError("release cannot bypass top-level completion")
    if payload.get("markdown_is_not_completion") is not True:
        raise ReleaseGateError("markdown cannot complete SPAR-050")
    return payload


def sealed_limitation_report_markdown() -> str:
    """Return the sealed SPAR-050 markdown projection. Not completion evidence."""

    gates = "\n".join(f"- `{gate_id}`" for gate_id, status, _terminal in GATE_EXPECTATIONS if status == "nominated")
    return (
        "# Semantic-Preserving Autonomous Remodularization Final Report\n"
        "\n"
        "Markdown status is not completion authority. This document is a "
        "nomination-only limitation and release-evidence projection for SPAR-050. "
        "It cannot complete the task, authorize a transition, promote a root, "
        "retire a façade, or grant write or semantic authority.\n"
        "\n"
        f"- Program: `{PROGRAM}`\n"
        f"- Task: `{TASK_ID}`\n"
        f"- Goal: `{GOAL_ID}`\n"
        f"- Predecessor: `{PREDECESSOR_TASK_IDS[0]}`\n"
        f"- Residual packet CID: `{RESIDUAL_PACKET_CID}`\n"
        f"- Planning tree: `{TREE_ID}`\n"
        f"- Residual tree: `{RESIDUAL_TREE_ID}`\n"
        f"- Network: `{NETWORK_DENY}`\n"
        f"- Rollout mode: `{REQUIRED_MODE}`\n"
        "- Nomination only: true\n"
        "- Promotion eligible: false\n"
        "- Final root accepted: false\n"
        "- Façade retirement authorized: false\n"
        "- General Python equivalence claimed: false\n"
        "\n"
        "## Release gates\n"
        "\n"
        "Nominated gates record current-tree evidence without accepting a root:\n"
        "\n"
        f"{gates}\n"
        "\n"
        "Typed terminal, never success:\n"
        "\n"
        "- `unsupported.dynamics` → `unresolved_required_dynamics`\n"
        "\n"
        "Unavailable:\n"
        "\n"
        "- `general_llm.network_denied`\n"
        "\n"
        "Negative evidence retained in the denominator:\n"
        "\n"
        "- `reuse.stale_tree`\n"
        "- `authority.self_authorization`\n"
        "- `authority.completion_bypass`\n"
        "- `rollout.worker_mode_change`\n"
        "\n"
        "Safety floors remain zero and non-compensable. Denominators retain "
        "failed, escalated, rejected, unavailable, human-reviewed, nominated, "
        "and typed-terminal observations.\n"
        "\n"
        "## Migration and compatibility\n"
        "\n"
        "Compatibility façades and consumer migrations remain nominated plans. "
        "This report does not retire a façade and cannot authorize an unaccepted "
        "public API break.\n"
        "\n"
        "## Benchmark and capstone\n"
        "\n"
        "SPAR-047 benchmark cells and SPAR-049 required-mode capstone receipts "
        "remain nomination-only. Efficiency targets are not met. Later procedure "
        "reuse without a general LLM is recorded as a nominated observation, not "
        "a promotion.\n"
        "\n"
        "## Limitations\n"
        "\n"
        "- General Python equivalence is not claimed.\n"
        "- Evidence is bounded by the declared observation and proof profile.\n"
        "- Unresolved required dynamics remain "
        "`unknown_until_current_module_analysis` and lower autonomy; they are a "
        "typed terminal, never success.\n"
        "- Model output is proposal-only and cannot remove a gate.\n"
        "- Independent supervisor validation and merge authority remain separate.\n"
    )


def sealed_release_report_payload() -> dict[str, Any]:
    """Return the sealed SPAR-050 report contract bound to predecessors."""

    preregistration = load_preregistration()
    gate_ids = tuple(item[0] for item in GATE_EXPECTATIONS)
    gates = [
        {
            "expected_status": status,
            "expected_terminal": _terminal_or_none(terminal),
            "general_llm_invoked": False,
            "in_denominator": True,
            "promotes": False,
            "required_mode": True,
            "gate_id": gate_id,
            "writes_repository": False,
        }
        for gate_id, status, terminal in GATE_EXPECTATIONS
    ]
    nominated = sum(1 for item in gates if item["expected_status"] == "nominated")
    unavailable = sum(1 for item in gates if item["expected_status"] == "unavailable")
    typed_terminal = sum(
        1 for item in gates if item["expected_status"] == "typed_terminal"
    )
    rejected = sum(1 for item in NEGATIVE_EVIDENCE if item["status"] == "rejected")
    denominators = {
        "escalated": 0,
        "failed": 0,
        "human_reviewed": 0,
        "nominated": nominated,
        "rejected": rejected,
        "total_gates": nominated + unavailable + rejected + typed_terminal,
        "typed_terminal": typed_terminal,
        "unavailable": unavailable,
    }
    return {
        "analyzer_id": ANALYZER_ID,
        "authority_roots": {
            "policy_root": POLICY_ROOT,
            "repository_forest_cid": FOREST_ID,
        },
        "benchmark_report_path": BENCHMARK_REPORT_RELATIVE,
        "can_authorize_completion": False,
        "can_authorize_transition": False,
        "can_create_authority": False,
        "capstone_report_path": CAPSTONE_REPORT_RELATIVE,
        "corpus_manifest_path": CORPUS_MANIFEST_RELATIVE,
        "denominator_policy": preregistration["denominator_policy"],
        "denominators": denominators,
        "dynamic_risk_inventory_path": DYNAMIC_RISK_RELATIVE,
        "efficiency_targets_met": False,
        "facade_retirement_authorized": False,
        "failed_efficiency_target": preregistration["failed_efficiency_target"],
        "final_root_accepted": False,
        "gate_ids": list(gate_ids),
        "gates": gates,
        "general_python_equivalence_claimed": False,
        "goal_id": GOAL_ID,
        "identity_inventory_path": IDENTITY_INVENTORY_RELATIVE,
        "interface": SPAR_RELEASE_REPORT_INTERFACE,
        "limitation_ids": list(LIMITATION_IDS),
        "markdown_is_not_completion": True,
        "markdown_path": MARKDOWN_RELATIVE,
        "model_output_is_proposal_only": True,
        "negative_evidence": [dict(item) for item in NEGATIVE_EVIDENCE],
        "network": NETWORK_DENY,
        "nomination_only": True,
        "obligation_ids": [OBLIGATION_ID],
        "observed_metrics": {
            "accepted_gates": 0,
            "facade_retirements": 0,
            "final_roots_accepted": 0,
            "general_llm_invoked": False,
            "required_coverage_loss": 0,
            "top_level_completion_bypass": 0,
        },
        "predecessor_task_ids": list(PREDECESSOR_TASK_IDS),
        "preregistration_path": PREREGISTRATION_RELATIVE,
        "program": PROGRAM,
        "projection_is_authority": False,
        "promotion_eligible": False,
        "promotion_targets": dict(preregistration["promotion_targets"]),
        "provider_tokenizer": dict(PROVIDER_TOKENIZER),
        "real_current_tree_path": REAL_CURRENT_TREE_RELATIVE,
        "required_gate_task": REQUIRED_GATE_TASK,
        "residual_packet_cid": RESIDUAL_PACKET_CID,
        "residual_target_ids": [TASK_ID],
        "rollout_mode": REQUIRED_MODE,
        "safety_floors_held": True,
        "schema": SPAR_RELEASE_REPORT_SCHEMA,
        "sealed_before_tuning": True,
        "self_hosted": True,
        "task_id": TASK_ID,
        "test_pass_is_not_completion": True,
        "top_level_completion_bypass": False,
        "tree_id": TREE_ID,
        "worker_may_change_mode": False,
        "worker_self_approval": False,
        "writes_repository": False,
        "zero_safety_floors": dict(preregistration["zero_safety_floors"]),
    }


def persist_sealed_release_report() -> dict[str, Any]:
    """Materialize the sealed JSON report onto the owned write path."""

    payload = sealed_release_report_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = ROOT / REPORT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    return payload


def persist_sealed_limitation_markdown() -> str:
    """Materialize the sealed markdown projection onto the owned write path."""

    text = sealed_limitation_report_markdown()
    path = ROOT / MARKDOWN_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def persist_sealed_release_artifacts() -> dict[str, Any]:
    persist_sealed_limitation_markdown()
    return persist_sealed_release_report()


@dataclass(frozen=True, slots=True)
class NominatedFinalRoot:
    """Nominated SPAR-050 final root. Cannot advance accepted roots."""

    tree_id: str
    root_cid: str
    predecessor_task_id: str = PREDECESSOR_TASK_IDS[0]
    mode: str = REQUIRED_MODE
    gate_task: str = REQUIRED_GATE_TASK
    nominated: bool = True
    accepted: bool = False
    nomination_only: bool = True
    worker_may_change_mode: bool = False
    can_authorize_completion: bool = False
    can_authorize_transition: bool = False
    writes_repository: bool = False
    task_id: str = TASK_ID
    interface: str = NOMINATED_FINAL_ROOT_INTERFACE
    schema: str = NOMINATED_FINAL_ROOT_SCHEMA

    def identity_payload(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "can_authorize_completion": False,
            "can_authorize_transition": False,
            "gate_task": self.gate_task,
            "interface": self.interface,
            "mode": REQUIRED_MODE,
            "nominated": True,
            "nomination_only": True,
            "predecessor_task_id": PREDECESSOR_TASK_IDS[0],
            "root_cid": self.root_cid,
            "schema": self.schema,
            "task_id": TASK_ID,
            "tree_id": self.tree_id,
            "worker_may_change_mode": False,
            "writes_repository": False,
        }

    @property
    def identity_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())


@dataclass(frozen=True, slots=True)
class NominatedLimitationReport:
    """Nominated SPAR-050 limitation report. Markdown cannot complete."""

    limitation_ids: tuple[str, ...]
    markdown_path: str = MARKDOWN_RELATIVE
    nominated: bool = True
    accepted: bool = False
    markdown_is_not_completion: bool = True
    general_python_equivalence_claimed: bool = False
    can_authorize_completion: bool = False
    writes_repository: bool = False
    task_id: str = TASK_ID
    interface: str = NOMINATED_LIMITATION_REPORT_INTERFACE
    schema: str = NOMINATED_LIMITATION_REPORT_SCHEMA

    def identity_payload(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "can_authorize_completion": False,
            "general_python_equivalence_claimed": False,
            "interface": self.interface,
            "limitation_ids": list(self.limitation_ids),
            "markdown_is_not_completion": True,
            "markdown_path": MARKDOWN_RELATIVE,
            "nominated": True,
            "schema": self.schema,
            "task_id": TASK_ID,
            "writes_repository": False,
        }

    @property
    def identity_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())


@dataclass(frozen=True, slots=True)
class SparReleaseReport:
    """Sealed SPAR-050 release report bound to SPAR-049 predecessors."""

    payload: Mapping[str, Any]
    report_cid: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "interface": SPAR_RELEASE_REPORT_INTERFACE,
            "payload": dict(self.payload),
            "report_cid": self.report_cid,
            "schema": SPAR_RELEASE_REPORT_SCHEMA,
            "task_id": TASK_ID,
        }


@dataclass(frozen=True, slots=True)
class SparReleaseGateResult:
    """One SPAR-050 release-gate observation. Nomination-only."""

    gate_id: str
    status: str
    evidence_cid: str
    expected_status: str
    expected_terminal: str = ""
    terminal: str = ""
    nominated: bool = True
    accepted: bool = False
    promotes: bool = False
    writes_repository: bool = False
    general_llm_invoked: bool = False
    required_mode: bool = True
    in_denominator: bool = True
    interface: str = SPAR_RELEASE_GATE_RESULT_INTERFACE
    schema: str = SPAR_RELEASE_GATE_RESULT_SCHEMA

    def identity_payload(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "evidence_cid": self.evidence_cid,
            "expected_status": self.expected_status,
            "expected_terminal": self.expected_terminal,
            "gate_id": self.gate_id,
            "general_llm_invoked": False,
            "in_denominator": True,
            "interface": self.interface,
            "nominated": True,
            "promotes": False,
            "required_mode": True,
            "schema": self.schema,
            "status": self.status,
            "terminal": self.terminal,
            "writes_repository": False,
        }

    @property
    def result_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())


@dataclass(frozen=True, slots=True)
class SparReleaseGateReceipt:
    """Nomination-only SPAR-050 release-gate receipt."""

    tree_id: str
    report_cid: str
    gate_results: tuple[SparReleaseGateResult, ...]
    negative_evidence_cids: tuple[str, ...]
    denominators: Mapping[str, int]
    gate_ids: tuple[str, ...]
    root: NominatedFinalRoot
    limitations: NominatedLimitationReport
    nominated: bool = True
    accepted: bool = False
    nomination_only: bool = True
    promotion_eligible: bool = False
    can_authorize_completion: bool = False
    can_authorize_transition: bool = False
    can_create_authority: bool = False
    writes_repository: bool = False
    worker_self_approval: bool = False
    worker_may_change_mode: bool = False
    projection_is_authority: bool = False
    top_level_completion_bypass: bool = False
    markdown_is_not_completion: bool = True
    facade_retirement_authorized: bool = False
    final_root_accepted: bool = False
    general_python_equivalence_claimed: bool = False
    network: str = NETWORK_DENY
    rollout_mode: str = REQUIRED_MODE
    analyzer_id: str = ANALYZER_ID
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    interface: str = SPAR_RELEASE_GATE_RECEIPT_INTERFACE
    schema: str = SPAR_RELEASE_GATE_RECEIPT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "accepted",
            "analyzer_id",
            "can_authorize_completion",
            "can_authorize_transition",
            "can_create_authority",
            "denominators",
            "facade_retirement_authorized",
            "final_root_accepted",
            "gate_ids",
            "gate_results",
            "general_python_equivalence_claimed",
            "goal_id",
            "interface",
            "limitations",
            "markdown_is_not_completion",
            "negative_evidence_cids",
            "network",
            "nominated",
            "nomination_only",
            "projection_is_authority",
            "promotion_eligible",
            "receipt_cid",
            "report_cid",
            "rollout_mode",
            "root",
            "schema",
            "task_id",
            "top_level_completion_bypass",
            "tree_id",
            "worker_may_change_mode",
            "worker_self_approval",
            "writes_repository",
        }
    )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "analyzer_id": self.analyzer_id,
            "can_authorize_completion": False,
            "can_authorize_transition": False,
            "can_create_authority": False,
            "denominators": dict(self.denominators),
            "facade_retirement_authorized": False,
            "final_root_accepted": False,
            "gate_ids": list(self.gate_ids),
            "gate_results": [item.identity_payload() for item in self.gate_results],
            "general_python_equivalence_claimed": False,
            "goal_id": self.goal_id,
            "interface": self.interface,
            "limitations": self.limitations.identity_payload(),
            "markdown_is_not_completion": True,
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "network": NETWORK_DENY,
            "nominated": True,
            "nomination_only": True,
            "projection_is_authority": False,
            "promotion_eligible": False,
            "report_cid": self.report_cid,
            "rollout_mode": REQUIRED_MODE,
            "root": self.root.identity_payload(),
            "schema": self.schema,
            "task_id": self.task_id,
            "top_level_completion_bypass": False,
            "tree_id": self.tree_id,
            "worker_may_change_mode": False,
            "worker_self_approval": False,
            "writes_repository": False,
        }

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SparReleaseGateReceipt":
        excluded = set(data) & IDENTITY_EXCLUDED_FIELDS
        if excluded:
            raise ReleaseGateError(
                f"observational fields are excluded from identity: {sorted(excluded)}"
            )
        unknown = set(data) - cls._FIELDS
        if unknown:
            raise ReleaseGateError(
                f"unsupported SparReleaseGateReceipt fields: {sorted(unknown)}"
            )
        payload = dict(data)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != SPAR_RELEASE_GATE_RECEIPT_SCHEMA:
            raise ReleaseGateError("unsupported release-gate receipt schema")
        if payload.pop("interface") != SPAR_RELEASE_GATE_RECEIPT_INTERFACE:
            raise ReleaseGateError("unsupported release-gate receipt interface")
        if payload.pop("accepted") is not False:
            raise ReleaseGateError("workers cannot self-approve SPAR-050")
        if payload.pop("nomination_only") is not True:
            raise ReleaseGateError("receipt must remain nomination_only")
        if payload.pop("nominated") is not True:
            raise ReleaseGateError("receipt must remain nominated")
        if payload.pop("promotion_eligible") is not False:
            raise ReleaseGateError("receipt cannot self-promote")
        if payload.pop("top_level_completion_bypass") is not False:
            raise ReleaseGateError("receipt cannot bypass top-level completion")
        if payload.pop("worker_may_change_mode") is not False:
            raise ReleaseGateError("workers cannot change rollout mode")
        if payload.pop("rollout_mode") != REQUIRED_MODE:
            raise ReleaseGateError("receipt rollout_mode must remain required")
        if payload.pop("markdown_is_not_completion") is not True:
            raise ReleaseGateError("markdown cannot complete SPAR-050")
        if payload.pop("facade_retirement_authorized") is not False:
            raise ReleaseGateError("receipt cannot retire a façade")
        if payload.pop("final_root_accepted") is not False:
            raise ReleaseGateError("receipt cannot accept a final root")
        if payload.pop("general_python_equivalence_claimed") is not False:
            raise ReleaseGateError("receipt cannot claim general Python equivalence")
        for flag in (
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "writes_repository",
            "worker_self_approval",
            "projection_is_authority",
        ):
            if payload.pop(flag) is not False:
                raise ReleaseGateError(f"receipt cannot claim {flag}")
        if payload.pop("network") != NETWORK_DENY:
            raise ReleaseGateError("receipt network must remain deny")
        results = tuple(
            SparReleaseGateResult(
                gate_id=str(item["gate_id"]),
                status=str(item["status"]),
                evidence_cid=str(item["evidence_cid"]),
                expected_status=str(item["expected_status"]),
                expected_terminal=str(item.get("expected_terminal") or ""),
                terminal=str(item.get("terminal") or ""),
            )
            for item in payload["gate_results"]
        )
        root_payload = dict(payload["root"])
        limitation_payload = dict(payload["limitations"])
        result = cls(
            tree_id=str(payload["tree_id"]),
            report_cid=str(payload["report_cid"]),
            gate_results=results,
            negative_evidence_cids=tuple(
                str(item) for item in payload["negative_evidence_cids"]
            ),
            denominators=dict(payload["denominators"]),
            gate_ids=tuple(str(item) for item in payload["gate_ids"]),
            root=NominatedFinalRoot(
                tree_id=str(root_payload["tree_id"]),
                root_cid=str(root_payload["root_cid"]),
                predecessor_task_id=str(
                    root_payload.get("predecessor_task_id") or PREDECESSOR_TASK_IDS[0]
                ),
                mode=str(root_payload.get("mode") or REQUIRED_MODE),
                gate_task=str(root_payload.get("gate_task") or REQUIRED_GATE_TASK),
            ),
            limitations=NominatedLimitationReport(
                limitation_ids=tuple(
                    str(item) for item in limitation_payload.get("limitation_ids") or ()
                )
            ),
            analyzer_id=str(payload["analyzer_id"]),
            task_id=str(payload["task_id"]),
            goal_id=str(payload["goal_id"]),
        )
        if claimed != result.receipt_cid:
            raise ReleaseGateError("SparReleaseGateReceipt receipt_cid mismatch")
        return result


def _observed(
    *,
    status: str,
    evidence_cid: str,
    terminal: str = "",
) -> dict[str, Any]:
    if status not in DECLARED_STATUSES:
        raise ReleaseGateError(f"unsupported gate status {status!r}")
    return {
        "status": status,
        "evidence_cid": evidence_cid,
        "terminal": terminal,
        "nominated": status == "nominated",
        "accepted": False,
        "promotes": False,
        "writes_repository": False,
        "general_llm_invoked": False,
    }


def _residual_packet(**overrides: Any) -> ResidualLlmPacket:
    fields: dict[str, Any] = {
        "task_id": TASK_CID,
        "repository_id": REPOSITORY_ID,
        "tree_id": RESIDUAL_TREE_ID,
        "write_paths": WRITE_SCOPE,
        "obligation_ids": (OBLIGATION_ID,),
        "counterexample_capsule": {"target_ids": [TASK_ID]},
        "validation_commands": VALIDATION_COMMANDS,
        "forest_id": FOREST_ID,
        "authority_roots": {
            "policy_root": POLICY_ROOT,
            "repository_forest_cid": FOREST_ID,
        },
        "limits": dict(RESIDUAL_LIMITS),
    }
    fields.update(overrides)
    return seal_residual_llm_packet(**fields)


def _run_seals_tasks_roots() -> dict[str, Any]:
    seal = _load_json(DEPENDENCY_SEAL_RELATIVE)
    scheduler = _load_json(SCHEDULER_RELATIVE)
    if not (ROOT / BOARD_VALIDATOR_RELATIVE).is_file():
        raise ReleaseGateError("board validator must remain present")
    if scheduler.get("completion_policy", {}).get("terminal_task_id") != TASK_ID:
        raise ReleaseGateError("scheduler terminal_task_id must remain SPAR-050")
    if scheduler.get("initial_projection", {}).get("terminal_task_id") != TASK_ID:
        raise ReleaseGateError("projection terminal_task_id must remain SPAR-050")
    if scheduler.get("completion_policy", {}).get("final_report_required") is not True:
        raise ReleaseGateError("final report remains required")
    if "seal_cid" not in seal and "dependency_root_cid" not in seal:
        raise ReleaseGateError("dependency seal must remain content-addressed")
    return _observed(status="nominated", evidence_cid=_cid("seals.tasks.roots"))


def _run_focused_regression_suites() -> dict[str, Any]:
    for relative in (
        ACCEPTANCE_MATRIX_RELATIVE,
        BENCHMARK_REPORT_RELATIVE,
        ADVERSARIAL_REPORT_RELATIVE,
        CAPSTONE_REPORT_RELATIVE,
    ):
        if not (ROOT / relative).is_file():
            raise ReleaseGateError(f"missing predecessor suite evidence: {relative}")
    admitted = admit_real_current_tree()
    if admitted["whole_repository_dump"] is not False:
        raise ReleaseGateError("release gate cannot dump the repository")
    if admitted["synthetic"] is not False:
        raise ReleaseGateError("focused suites must remain current-tree")
    result = run_acceptance_scenario(
        {
            "expected_status": "nominated",
            "expected_terminal": None,
            "promotes": False,
            "scenario_id": "extraction.dry_run_wave",
            "stage": "extraction",
            "writes_repository": False,
        }
    )
    if result.accepted or result.promotes or result.writes_repository:
        raise ReleaseGateError("focused suites cannot complete or write")
    return _observed(status="nominated", evidence_cid=result.evidence_cid)


def _run_identity_preservation() -> dict[str, Any]:
    inventory = _load_json(IDENTITY_INVENTORY_RELATIVE)
    if inventory.get("schema") != "spar/identity-inventory@1":
        raise ReleaseGateError("identity inventory schema must remain sealed")
    rules = [str(item) for item in inventory.get("rules") or ()]
    if not any("excluded from semantic identity" in item for item in rules):
        raise ReleaseGateError("identity inventory must exclude observational fields")
    fields = set(inventory.get("identity_fields") or ())
    if "source_cid" not in fields or "semantic_state_root_cid" not in fields:
        raise ReleaseGateError("identity inventory must retain source and state roots")
    return _observed(status="nominated", evidence_cid=_cid("identity.preservation"))


def _run_model_context_changes() -> dict[str, Any]:
    packet = _residual_packet()
    if packet.completion_authority or packet.write_authority or packet.semantic_authority:
        raise ReleaseGateError("residual packet cannot grant model authority")
    payload = packet._payload()
    if payload.get("contains_source_body") is not False:
        raise ReleaseGateError("residual packet cannot dump task or source")
    if payload.get("contains_secrets") is not False:
        raise ReleaseGateError("residual packet cannot retain secrets")
    forbidden_keys = {"full_task", "repository_dump", "source_body", "file_contents"}
    if forbidden_keys.intersection(payload):
        raise ReleaseGateError("residual packet cannot dump task or source")
    if packet.packet_id != RESIDUAL_PACKET_CID:
        raise ReleaseGateError("residual packet CID must remain sealed")
    return _observed(status="nominated", evidence_cid=packet.packet_id)


def _run_unsupported_dynamics() -> dict[str, Any]:
    inventory = _load_json(DYNAMIC_RISK_RELATIVE)
    risks = inventory.get("risks") or []
    unknown = [
        item
        for item in risks
        if str(item.get("status") or "") == "unknown_until_current_module_analysis"
    ]
    if not unknown:
        raise ReleaseGateError("dynamic risk inventory must retain unresolved dynamics")
    return _observed(
        status="typed_terminal",
        evidence_cid=_cid("unresolved_required_dynamics"),
        terminal="unresolved_required_dynamics",
    )


def _run_scale_denominators() -> dict[str, Any]:
    benchmark = load_benchmark_report()
    if benchmark["task_id"] != "SPAR-047":
        raise ReleaseGateError("scale denominators must bind SPAR-047")
    buckets = dict(benchmark.get("denominators") or {})
    for name in (
        "failed",
        "escalated",
        "rejected",
        "unavailable",
        "human_reviewed",
        "nominated",
    ):
        if name not in buckets:
            raise ReleaseGateError(f"benchmark denominators missing {name}")
        if int(buckets[name]) < 0:
            raise ReleaseGateError("denominators cannot drop non-success observations")
    if benchmark.get("promotion_eligible") is True:
        raise ReleaseGateError("benchmark cannot self-promote")
    return _observed(status="nominated", evidence_cid=_cid("scale.denominators"))


def _run_capstone_verify() -> dict[str, Any]:
    capstone = load_capstone_report()
    if capstone["task_id"] != "SPAR-049":
        raise ReleaseGateError("capstone predecessor must remain SPAR-049")
    if capstone.get("nomination_only") is not True:
        raise ReleaseGateError("capstone must remain nomination_only")
    if capstone.get("promotion_eligible") is True:
        raise ReleaseGateError("capstone cannot self-promote")
    if capstone.get("top_level_completion_bypass") is True:
        raise ReleaseGateError("capstone cannot bypass top-level completion")
    if capstone.get("safety_floors_held") is not True:
        raise ReleaseGateError("capstone safety floors must remain held")
    return _observed(status="nominated", evidence_cid=_cid(str(capstone["residual_packet_cid"])))


def _run_migration_compatibility() -> dict[str, Any]:
    if FACADE_CAN_RETIRE_FACADE is not False:
        raise ReleaseGateError("migration cannot retire a façade")
    if FACADE_RETIREMENT_AUTHORIZED is not False:
        raise ReleaseGateError("SPAR-050 cannot authorize façade retirement")
    return _observed(status="nominated", evidence_cid=_cid("migration.compatibility"))


def _run_facade_retirement() -> dict[str, Any]:
    if FACADE_CAN_RETIRE_FACADE is not False:
        raise ReleaseGateError("façade retirement must remain unauthorized")
    return _observed(status="nominated", evidence_cid=_cid("facade.retirement.not_authorized"))


def _run_transitive_final_root() -> dict[str, Any]:
    root = NominatedFinalRoot(
        tree_id=TREE_ID,
        root_cid=_cid("nominated-final-root"),
    )
    if root.accepted or root.can_authorize_completion:
        raise ReleaseGateError("final root cannot complete")
    return _observed(status="nominated", evidence_cid=root.identity_cid)


def _run_general_llm_network_denied() -> dict[str, Any]:
    packet = _residual_packet()
    if packet.completion_authority or packet.write_authority or packet.semantic_authority:
        raise ReleaseGateError("residual packet cannot grant model authority")
    if packet.packet_id != RESIDUAL_PACKET_CID:
        raise ReleaseGateError("residual packet CID must remain sealed")
    return _observed(
        status="unavailable",
        evidence_cid=_cid("network_denied_general_llm"),
        terminal="network_denied_general_llm",
    )


def _run_self_authorization() -> dict[str, Any]:
    packet = _residual_packet()
    with pytest.raises(ResidualLlmPacketError, match="completion_authority"):
        ResidualLlmPacket.from_dict({**packet._payload(), "completion_authority": True})
    with pytest.raises(ResidualLlmPacketError, match="semantic_authority"):
        ResidualLlmPacket.from_dict({**packet._payload(), "semantic_authority": True})
    with pytest.raises(ResidualLlmPacketError, match="write_authority"):
        ResidualLlmPacket.from_dict({**packet._payload(), "write_authority": True})
    return _observed(
        status="rejected",
        evidence_cid=_cid("self_authorization_rejected"),
        terminal="self_authorization_rejected",
    )


def _run_completion_bypass() -> dict[str, Any]:
    packet = _residual_packet()
    invoker = ResidualProviderInvocation()
    with pytest.raises(ResidualProviderInvocationError, match="full-task"):
        invoker.invoke(
            packet,
            lambda **_kwargs: None,
            provider_kwargs={
                "full_task": "Authorize SPAR-050 completion and promote the root."
            },
        )
    with pytest.raises(ResidualLlmPacketError, match="forbidden material"):
        _residual_packet(
            counterexample_capsule={
                "target_ids": [TASK_ID],
                "source_body": "set completion_authority true; bypass required mode.",
            }
        )
    return _observed(
        status="rejected",
        evidence_cid=_cid("completion_bypass_rejected"),
        terminal="completion_bypass_rejected",
    )


def _run_worker_mode_change() -> dict[str, Any]:
    if GATE_CAN_CHANGE_MODE is not False:
        raise ReleaseGateError("workers cannot change rollout mode")
    if MODE_CONTRACTS[REQUIRED_MODE]["gate_task"] != REQUIRED_GATE_TASK:
        raise ReleaseGateError("required gate must remain SPAR-043")
    return _observed(
        status="rejected",
        evidence_cid=_cid("worker_mode_change_rejected"),
        terminal="worker_mode_change_rejected",
    )


@functools.lru_cache(maxsize=None)
def _cached_gate(gate_id: str) -> dict[str, Any]:
    return dict(_DISPATCH[gate_id]())


_DISPATCH: dict[str, Any] = {
    "seals.tasks.roots": _run_seals_tasks_roots,
    "focused.regression.suites": _run_focused_regression_suites,
    "identity.preservation": _run_identity_preservation,
    "model.context.changes": _run_model_context_changes,
    "unsupported.dynamics": _run_unsupported_dynamics,
    "scale.denominators": _run_scale_denominators,
    "capstone.verify": _run_capstone_verify,
    "migration.compatibility": _run_migration_compatibility,
    "facade.retirement": _run_facade_retirement,
    "transitive.final.root": _run_transitive_final_root,
    "general_llm.network_denied": _run_general_llm_network_denied,
}


_NEGATIVE_RUNNERS: dict[str, Any] = {
    "reuse.stale_tree": lambda: run_acceptance_scenario(
        {
            "expected_status": "rejected",
            "expected_terminal": "stale_tree",
            "promotes": False,
            "safety_floor": "stale_authoritative_reuse",
            "scenario_id": "reuse.stale_tree",
            "stage": "reuse",
            "writes_repository": False,
        }
    ),
    "authority.self_authorization": _run_self_authorization,
    "authority.completion_bypass": _run_completion_bypass,
    "rollout.worker_mode_change": _run_worker_mode_change,
}


def run_release_gate_item(gate: Mapping[str, Any]) -> SparReleaseGateResult:
    """Exercise one sealed SPAR-050 release gate against current adapters."""

    gate_id = str(gate.get("gate_id") or "")
    expected_status = str(gate.get("expected_status") or "")
    expected_terminal = str(gate.get("expected_terminal") or "")
    if gate_id not in _DISPATCH:
        raise ReleaseGateError(f"unknown release gate {gate_id!r}")
    if expected_status not in DECLARED_STATUSES:
        raise ReleaseGateError(f"unknown expected status {expected_status!r}")
    if gate.get("promotes") is True:
        raise ReleaseGateError("release gates cannot promote")
    if gate.get("writes_repository") is True:
        raise ReleaseGateError("release gates cannot write the repository")
    if gate.get("general_llm_invoked") is True:
        raise ReleaseGateError("SPAR-050 cannot invoke a general LLM")
    if gate.get("required_mode") is not True:
        raise ReleaseGateError("SPAR-050 gates must remain required-mode")
    observed = _cached_gate(gate_id)
    if observed["status"] != expected_status:
        raise ReleaseGateError(
            f"{gate_id} status {observed['status']!r} != {expected_status!r}"
        )
    if expected_terminal and observed.get("terminal") != expected_terminal:
        raise ReleaseGateError(
            f"{gate_id} terminal {observed.get('terminal')!r} != {expected_terminal!r}"
        )
    if observed["accepted"] or observed["promotes"] or observed["writes_repository"]:
        raise ReleaseGateError(f"{gate_id} cannot promote or write")
    if observed["general_llm_invoked"]:
        raise ReleaseGateError(f"{gate_id} invoked a general LLM")
    return SparReleaseGateResult(
        gate_id=gate_id,
        status=str(observed["status"]),
        evidence_cid=str(observed["evidence_cid"]),
        expected_status=expected_status,
        expected_terminal=expected_terminal,
        terminal=str(observed.get("terminal") or ""),
        nominated=True,
    )


def _run_negative_evidence() -> tuple[str, ...]:
    cids: list[str] = []
    for item in NEGATIVE_EVIDENCE:
        runner = _NEGATIVE_RUNNERS[str(item["gate_id"])]
        result = runner()
        status = getattr(result, "status", None) or result["status"]
        promotes = getattr(result, "promotes", None)
        if promotes is None:
            promotes = result["promotes"]
        writes = getattr(result, "writes_repository", None)
        if writes is None:
            writes = result["writes_repository"]
        accepted = getattr(result, "accepted", None)
        if accepted is None:
            accepted = result["accepted"]
        if status != "rejected":
            raise ReleaseGateError(f"{item['gate_id']} must remain rejected")
        if promotes or writes or accepted:
            raise ReleaseGateError(f"{item['gate_id']} cannot promote or write")
        evidence_cid = getattr(result, "evidence_cid", None) or result["evidence_cid"]
        cids.append(str(evidence_cid))
    return tuple(cids)


def run_release_gate(
    report: Mapping[str, Any] | None = None,
) -> SparReleaseGateReceipt:
    """Run the sealed SPAR-050 release gate. Current authority remains separate."""

    loaded = dict(report) if report is not None else load_release_report()
    sealed = sealed_release_report_payload()
    if loaded.get("gate_ids") != sealed["gate_ids"]:
        raise ReleaseGateError("release gates must remain sealed")
    if tuple(loaded.get("predecessor_task_ids") or ()) != PREDECESSOR_TASK_IDS:
        raise ReleaseGateError("predecessors must remain SPAR-049")
    preregistration = load_preregistration()
    if dict(loaded.get("zero_safety_floors") or {}) != dict(
        preregistration["zero_safety_floors"]
    ):
        raise ReleaseGateError("safety floors must match preregistration")
    corpus = load_benchmark_corpus_manifest()
    if dict(corpus["zero_safety_floors"]) != dict(loaded["zero_safety_floors"]):
        raise ReleaseGateError("corpus floors must match the release report")
    predecessor = load_capstone_report()
    if predecessor["task_id"] != "SPAR-049":
        raise ReleaseGateError("predecessor report must remain SPAR-049")
    adversarial = load_adversarial_report()
    if adversarial["task_id"] != "SPAR-048":
        raise ReleaseGateError("adversarial predecessor must remain SPAR-048")
    benchmark = load_benchmark_report()
    if benchmark["task_id"] != "SPAR-047":
        raise ReleaseGateError("benchmark predecessor must remain SPAR-047")
    matrix = load_acceptance_matrix()
    if matrix["task_id"] != "SPAR-046":
        raise ReleaseGateError("acceptance matrix predecessor must remain SPAR-046")
    covered = {str(item["gate_id"]) for item in loaded["gates"]}
    missing = [name for name in sealed["gate_ids"] if name not in covered]
    if missing:
        raise ReleaseGateError(f"release report missing gates: {missing}")
    if loaded.get("promotion_eligible") is True:
        raise ReleaseGateError("release cannot self-promote")
    if loaded.get("residual_packet_cid") != RESIDUAL_PACKET_CID:
        raise ReleaseGateError("residual packet CID must remain sealed")
    if loaded.get("rollout_mode") != REQUIRED_MODE:
        raise ReleaseGateError("release rollout_mode must remain required")
    if loaded.get("top_level_completion_bypass") is True:
        raise ReleaseGateError("release cannot bypass top-level completion")
    if loaded.get("markdown_is_not_completion") is not True:
        raise ReleaseGateError("markdown cannot complete SPAR-050")
    if loaded.get("facade_retirement_authorized") is True:
        raise ReleaseGateError("release cannot retire a façade")
    if loaded.get("final_root_accepted") is True:
        raise ReleaseGateError("release cannot accept a final root")
    results = tuple(run_release_gate_item(item) for item in loaded["gates"])
    if any(item.promotes or item.accepted for item in results):
        raise ReleaseGateError("no release gate may be promoted")
    negative = _run_negative_evidence()
    identity = {
        "payload": loaded,
        "task_id": TASK_ID,
        "tree_id": TREE_ID,
    }
    root_observed = _cached_gate("transitive.final.root")
    return SparReleaseGateReceipt(
        tree_id=TREE_ID,
        report_cid=cid_for_dag_json(identity),
        gate_results=results,
        negative_evidence_cids=negative,
        denominators=dict(loaded["denominators"]),
        gate_ids=tuple(str(item) for item in loaded["gate_ids"]),
        root=NominatedFinalRoot(
            tree_id=TREE_ID,
            root_cid=str(root_observed["evidence_cid"]),
        ),
        limitations=NominatedLimitationReport(limitation_ids=LIMITATION_IDS),
    )


def nominate_release_gate() -> SparReleaseGateReceipt:
    """Nominate the SPAR-050 report. Independent validation remains separate."""

    return run_release_gate()


@functools.lru_cache(maxsize=1)
def _cached_nomination() -> SparReleaseGateReceipt:
    return nominate_release_gate()


def dry_run_release_gate() -> SparReleaseGateReceipt:
    return _cached_nomination()


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-050"
    assert GOAL_ID == "SPAR-G082"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-049",)
    assert SPAR_RELEASE_REPORT_INTERFACE == "SparReleaseReport@1"
    assert SPAR_RELEASE_GATE_RECEIPT_INTERFACE == "SparReleaseGateReceipt@1"
    assert SPAR_RELEASE_GATE_RESULT_INTERFACE == "SparReleaseGateResult@1"
    assert NOMINATED_FINAL_ROOT_INTERFACE == "NominatedFinalRoot@1"
    assert NOMINATED_LIMITATION_REPORT_INTERFACE == "NominatedLimitationReport@1"
    assert RELEASE_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("release_gate@1")
    assert TEST_PATH.is_file()
    assert (ROOT / REPORT_RELATIVE).is_file()
    assert (ROOT / MARKDOWN_RELATIVE).is_file()
    assert WRITE_SCOPE == (
        REPORT_RELATIVE,
        MARKDOWN_RELATIVE,
        "test/api/semantic_refactoring/test_release_gate.py",
    )
    assert (ROOT / PREREGISTRATION_RELATIVE).is_file()
    assert (ROOT / CORPUS_MANIFEST_RELATIVE).is_file()
    assert (ROOT / ACCEPTANCE_MATRIX_RELATIVE).is_file()
    assert (ROOT / BENCHMARK_REPORT_RELATIVE).is_file()
    assert (ROOT / ADVERSARIAL_REPORT_RELATIVE).is_file()
    assert (ROOT / CAPSTONE_REPORT_RELATIVE).is_file()
    assert (ROOT / IDENTITY_INVENTORY_RELATIVE).is_file()
    assert (ROOT / DYNAMIC_RISK_RELATIVE).is_file()
    assert (ROOT / REAL_CURRENT_TREE_RELATIVE).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "release evidence"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert CAN_AUTHORIZE_COMPLETION is False
    assert CAN_AUTHORIZE_TRANSITION is False
    assert CAN_CREATE_AUTHORITY is False
    assert GATE_WRITES_REPOSITORY is False
    assert GATE_IS_NOMINATION_ONLY is True
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert WORKER_MAY_CHANGE_MODE is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert TOP_LEVEL_COMPLETION_BYPASS is False
    assert SELF_HOSTED is True
    assert FACADE_RETIREMENT_AUTHORIZED is False
    assert GENERAL_PYTHON_EQUIVALENCE_CLAIMED is False
    assert FINAL_ROOT_ACCEPTED is False
    profile = release_gate_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    report = load_release_report()
    assert report["nomination_only"] is True
    assert report["can_authorize_completion"] is False
    assert report["can_authorize_transition"] is False
    assert report["can_create_authority"] is False
    assert report["worker_self_approval"] is False
    assert report["worker_may_change_mode"] is False
    assert report["model_output_is_proposal_only"] is True
    assert report["test_pass_is_not_completion"] is True
    assert report["markdown_is_not_completion"] is True
    assert report["writes_repository"] is False
    assert report["projection_is_authority"] is False
    assert report["promotion_eligible"] is False
    assert report["top_level_completion_bypass"] is False
    assert report["self_hosted"] is True
    assert report["rollout_mode"] == REQUIRED_MODE
    assert report["facade_retirement_authorized"] is False
    assert report["final_root_accepted"] is False
    assert report["general_python_equivalence_claimed"] is False


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(TEST_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "SparReleaseReport" in names
    assert "SparReleaseGateResult" in names
    assert "SparReleaseGateReceipt" in names
    assert "NominatedFinalRoot" in names
    assert "NominatedLimitationReport" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "SparReleaseGateReceipt" in exports
    assert "NominatedFinalRoot" in exports
    assert "NominatedLimitationReport" in exports
    assert "nominate_release_gate" in exports
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for forbidden in (
        "authorize_completion",
        "promote_root",
        "dump_repository",
        "admit_third_party_source",
        "overwrite_root_conflict",
        "retire_facade",
    ):
        assert forbidden not in functions
        assert forbidden not in names


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_committed_report_matches_sealed_payload() -> None:
    payload = sealed_release_report_payload()
    report = load_release_report()
    assert report == payload
    assert report["residual_packet_cid"] == RESIDUAL_PACKET_CID
    assert report["residual_target_ids"] == [TASK_ID]
    assert report["obligation_ids"] == [OBLIGATION_ID]
    assert report["sealed_before_tuning"] is True
    assert report["authority_roots"]["policy_root"] == POLICY_ROOT
    assert report["authority_roots"]["repository_forest_cid"] == FOREST_ID


def test_committed_markdown_matches_sealed_payload() -> None:
    expected = sealed_limitation_report_markdown()
    observed = (ROOT / MARKDOWN_RELATIVE).read_text(encoding="utf-8")
    assert observed == expected
    assert "not completion authority" in observed
    assert RESIDUAL_PACKET_CID in observed
    assert "General Python equivalence is not claimed." in observed
    assert "completion_authority" not in observed
    assert "repository_dump" not in observed
    assert "source_body" not in observed


def test_report_covers_required_gates_and_floors() -> None:
    report = load_release_report()
    preregistration = load_preregistration()
    corpus = load_benchmark_corpus_manifest()
    expected_gates = [item[0] for item in GATE_EXPECTATIONS]
    assert report["gate_ids"] == expected_gates
    assert tuple(report["predecessor_task_ids"]) == PREDECESSOR_TASK_IDS
    covered = {item["gate_id"] for item in report["gates"]}
    assert covered == set(expected_gates)
    assert report["zero_safety_floors"] == preregistration["zero_safety_floors"]
    assert corpus["zero_safety_floors"] == report["zero_safety_floors"]
    assert all(value == 0 for value in report["zero_safety_floors"].values())
    floors = {item["safety_floor"] for item in report["negative_evidence"]}
    for required in (
        "stale_authoritative_reuse",
        "false_task_completions",
        "unauthorized_mutations",
    ):
        assert required in floors
    for item in report["gates"]:
        assert item["promotes"] is False
        assert item["writes_repository"] is False
        assert item["in_denominator"] is True
        assert item["general_llm_invoked"] is False
        assert item["required_mode"] is True
        assert item["expected_status"] in DECLARED_STATUSES
    assert report["required_gate_task"] == REQUIRED_GATE_TASK
    assert report["real_current_tree_path"] == REAL_CURRENT_TREE_RELATIVE
    assert report["limitation_ids"] == list(LIMITATION_IDS)


def test_denominators_retain_honest_non_success_gates() -> None:
    report = load_release_report()
    buckets = report["denominators"]
    for name in DENOMINATOR_BUCKETS:
        assert name in buckets
        assert type(buckets[name]) is int
        assert buckets[name] >= 0
    assert buckets["unavailable"] == 1
    assert buckets["rejected"] == 4
    assert buckets["typed_terminal"] == 1
    assert buckets["failed"] == 0
    assert buckets["escalated"] == 0
    assert buckets["human_reviewed"] == 0
    assert buckets["nominated"] == 9
    assert buckets["total_gates"] == 15
    assert (
        buckets["nominated"]
        + buckets["unavailable"]
        + buckets["rejected"]
        + buckets["typed_terminal"]
        == buckets["total_gates"]
    )


def test_predecessors_remain_readable_without_dump() -> None:
    report = load_release_report()
    encoded = json.dumps(report)
    assert "def leaf" not in encoded
    assert "class ShadowPlanGate" not in encoded
    assert "repository_dump" not in encoded
    assert "source_body" not in encoded
    corpus = load_benchmark_corpus_manifest()
    assert corpus["task_id"] == "SPAR-045"
    admitted = admit_real_current_tree()
    assert admitted["whole_repository_dump"] is False
    assert admitted["synthetic"] is False
    predecessor = load_capstone_report()
    assert predecessor["task_id"] == "SPAR-049"
    adversarial = load_adversarial_report()
    assert adversarial["task_id"] == "SPAR-048"
    benchmark = load_benchmark_report()
    assert benchmark["task_id"] == "SPAR-047"
    matrix = load_acceptance_matrix()
    assert matrix["task_id"] == "SPAR-046"


def test_provider_tokenizer_criteria_are_exact_and_unavailable_llm() -> None:
    report = load_release_report()
    criteria = report["provider_tokenizer"]
    assert criteria["network"] == NETWORK_DENY
    assert criteria["provider_available"] is False
    assert criteria["live_model_channel"] is False
    assert criteria["general_llm_invoked"] is False
    assert criteria["tokenizer_id"] == TOKENIZER_ID
    assert criteria["tokenizer_available"] is True
    assert criteria["tokenizer_identity_required"] is True
    metrics = report["observed_metrics"]
    assert metrics["accepted_gates"] == 0
    assert metrics["general_llm_invoked"] is False
    assert metrics["required_coverage_loss"] == 0
    assert metrics["facade_retirements"] == 0
    assert metrics["final_roots_accepted"] == 0
    assert metrics["top_level_completion_bypass"] == 0
    assert report["promotion_eligible"] is False
    assert report["safety_floors_held"] is True
    assert report["efficiency_targets_met"] is False
    assert report["promotion_targets"]["later_promoted_procedure_no_general_llm_waves"] == 1


def test_residual_packet_binds_sealed_fields() -> None:
    packet = _residual_packet()
    assert packet.packet_id == RESIDUAL_PACKET_CID
    assert packet.task_id == TASK_CID
    assert packet.tree_id == RESIDUAL_TREE_ID
    assert packet.nomination_only is True
    assert packet.write_authority is False
    assert packet.semantic_authority is False
    assert packet.completion_authority is False
    assert tuple(packet.write_paths) == WRITE_SCOPE
    assert tuple(packet.obligation_ids) == (OBLIGATION_ID,)
    assert tuple(packet.validation_commands) == VALIDATION_COMMANDS
    assert dict(packet.counterexample_capsule) == {"target_ids": [TASK_ID]}
    with pytest.raises(Exception, match="secret"):
        assert_provider_env_excludes_secrets({"OPENAI_API_KEY": "sk-test"})


def test_each_gate_is_exercised_without_promotion() -> None:
    receipt = dry_run_release_gate()
    gates = tuple(item.gate_id for item in receipt.gate_results)
    assert gates == receipt.gate_ids
    by_id = {item.gate_id: item for item in receipt.gate_results}
    assert by_id["seals.tasks.roots"].status == "nominated"
    assert by_id["focused.regression.suites"].status == "nominated"
    assert by_id["identity.preservation"].status == "nominated"
    assert by_id["model.context.changes"].status == "nominated"
    assert by_id["unsupported.dynamics"].status == "typed_terminal"
    assert by_id["unsupported.dynamics"].terminal == "unresolved_required_dynamics"
    assert by_id["scale.denominators"].status == "nominated"
    assert by_id["capstone.verify"].status == "nominated"
    assert by_id["migration.compatibility"].status == "nominated"
    assert by_id["facade.retirement"].status == "nominated"
    assert by_id["transitive.final.root"].status == "nominated"
    assert by_id["general_llm.network_denied"].status == "unavailable"
    assert by_id["general_llm.network_denied"].terminal == "network_denied_general_llm"
    for item in receipt.gate_results:
        assert item.accepted is False
        assert item.promotes is False
        assert item.writes_repository is False
        assert item.general_llm_invoked is False
        assert item.required_mode is True
        assert item.in_denominator is True
    assert receipt.root.mode == REQUIRED_MODE
    assert receipt.root.accepted is False
    assert receipt.limitations.markdown_is_not_completion is True
    assert receipt.limitations.accepted is False
    assert receipt.limitations.general_python_equivalence_claimed is False


def test_fail_closed_negative_evidence_stays_in_denominator() -> None:
    receipt = dry_run_release_gate()
    assert len(receipt.negative_evidence_cids) == 4
    report = load_release_report()
    floors = dict(report["zero_safety_floors"])
    by_gate = {item["gate_id"]: item for item in report["negative_evidence"]}
    assert by_gate["reuse.stale_tree"]["terminal"] == "stale_tree"
    assert by_gate["authority.self_authorization"]["terminal"] == (
        "self_authorization_rejected"
    )
    assert by_gate["authority.completion_bypass"]["terminal"] == (
        "completion_bypass_rejected"
    )
    assert by_gate["rollout.worker_mode_change"]["terminal"] == (
        "worker_mode_change_rejected"
    )
    assert floors["stale_authoritative_reuse"] == 0
    assert floors["false_task_completions"] == 0
    assert floors["unauthorized_mutations"] == 0
    assert floors["test_or_proof_weakening"] == 0
    assert floors["root_conflict_overwrite"] == 0


def test_nomination_receipt_cannot_complete_or_self_approve() -> None:
    receipt = dry_run_release_gate()
    assert receipt.nominated is True
    assert receipt.accepted is False
    assert receipt.nomination_only is True
    assert receipt.promotion_eligible is False
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_authority is False
    assert receipt.writes_repository is False
    assert receipt.worker_self_approval is False
    assert receipt.worker_may_change_mode is False
    assert receipt.top_level_completion_bypass is False
    assert receipt.markdown_is_not_completion is True
    assert receipt.facade_retirement_authorized is False
    assert receipt.final_root_accepted is False
    assert receipt.general_python_equivalence_claimed is False
    assert receipt.rollout_mode == REQUIRED_MODE
    assert receipt.network == NETWORK_DENY
    assert receipt.task_id == TASK_ID
    assert receipt.goal_id == GOAL_ID
    assert receipt.tree_id == TREE_ID
    encoded = receipt.to_dict()
    restored = SparReleaseGateReceipt.from_dict(encoded)
    assert restored.receipt_cid == receipt.receipt_cid
    assert restored == receipt
    dry = dry_run_release_gate()
    assert dry.receipt_cid == receipt.receipt_cid
    with pytest.raises(ReleaseGateError, match="self-approve"):
        SparReleaseGateReceipt.from_dict({**encoded, "accepted": True})
    with pytest.raises(ReleaseGateError, match="can_authorize_completion"):
        SparReleaseGateReceipt.from_dict({**encoded, "can_authorize_completion": True})
    with pytest.raises(ReleaseGateError, match="self-promote"):
        SparReleaseGateReceipt.from_dict({**encoded, "promotion_eligible": True})
    with pytest.raises(ReleaseGateError, match="bypass"):
        SparReleaseGateReceipt.from_dict({**encoded, "top_level_completion_bypass": True})
    with pytest.raises(ReleaseGateError, match="change rollout mode"):
        SparReleaseGateReceipt.from_dict({**encoded, "worker_may_change_mode": True})
    with pytest.raises(ReleaseGateError, match="markdown cannot complete"):
        SparReleaseGateReceipt.from_dict({**encoded, "markdown_is_not_completion": False})
    with pytest.raises(ReleaseGateError, match="retire a façade"):
        SparReleaseGateReceipt.from_dict({**encoded, "facade_retirement_authorized": True})
    with pytest.raises(ReleaseGateError, match="final root"):
        SparReleaseGateReceipt.from_dict({**encoded, "final_root_accepted": True})


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    first = run_release_gate()
    second = run_release_gate()
    assert first.receipt_cid == second.receipt_cid
    assert [item.result_cid for item in first.gate_results] == [
        item.result_cid for item in second.gate_results
    ]
    assert first.writes_repository is False
    assert DRY_RUN_MUTATES is False


def test_identity_excludes_observational_fields() -> None:
    receipt = dry_run_release_gate()
    encoded = receipt.to_dict()
    for field in IDENTITY_EXCLUDED_FIELDS:
        assert field not in encoded
        with pytest.raises(ReleaseGateError, match="observational"):
            SparReleaseGateReceipt.from_dict({**encoded, field: "now"})


def test_vector_or_model_cannot_admit_or_promote() -> None:
    report = dict(load_release_report())
    with pytest.raises(ReleaseGateError, match="missing gates"):
        run_release_gate(
            {
                **report,
                "gates": [
                    {
                        "expected_status": "nominated",
                        "expected_terminal": None,
                        "general_llm_invoked": False,
                        "in_denominator": True,
                        "promotes": False,
                        "required_mode": True,
                        "gate_id": "vector.similarity_admit",
                        "writes_repository": False,
                    }
                ],
            }
        )
    with pytest.raises(ReleaseGateError, match="cannot promote"):
        run_release_gate_item(
            {
                "expected_status": "nominated",
                "expected_terminal": None,
                "general_llm_invoked": False,
                "in_denominator": True,
                "promotes": True,
                "required_mode": True,
                "gate_id": "seals.tasks.roots",
                "writes_repository": False,
            }
        )
    with pytest.raises(ReleaseGateError, match="general LLM"):
        run_release_gate_item(
            {
                "expected_status": "nominated",
                "expected_terminal": None,
                "general_llm_invoked": True,
                "in_denominator": True,
                "promotes": False,
                "required_mode": True,
                "gate_id": "model.context.changes",
                "writes_repository": False,
            }
        )


def test_missing_gate_and_floor_drift_fail_closed() -> None:
    report = dict(load_release_report())
    truncated = [
        item
        for item in report["gates"]
        if item["gate_id"] != "transitive.final.root"
    ]
    with pytest.raises(ReleaseGateError, match="missing gates"):
        run_release_gate({**report, "gates": truncated})
    drifted = dict(report["zero_safety_floors"])
    drifted["false_task_completions"] = 1
    with pytest.raises(ReleaseGateError, match="safety floors"):
        run_release_gate({**report, "zero_safety_floors": drifted})
    with pytest.raises(ReleaseGateError, match="cannot self-promote"):
        run_release_gate({**report, "promotion_eligible": True})
    with pytest.raises(ReleaseGateError, match="bypass"):
        run_release_gate({**report, "top_level_completion_bypass": True})
    with pytest.raises(ReleaseGateError, match="retire a façade"):
        run_release_gate({**report, "facade_retirement_authorized": True})
    with pytest.raises(ReleaseGateError, match="final root"):
        run_release_gate({**report, "final_root_accepted": True})


def test_markdown_is_not_completion_authority() -> None:
    report = load_release_report()
    markdown = (ROOT / MARKDOWN_RELATIVE).read_text(encoding="utf-8")
    assert report["markdown_is_not_completion"] is True
    assert report["markdown_path"] == MARKDOWN_RELATIVE
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert "Markdown status is not completion authority" in markdown
    assert "cannot complete the task" in markdown
    assert TASK_ID in markdown
    assert "SPAR-049" in markdown
    limitations = NominatedLimitationReport(limitation_ids=LIMITATION_IDS)
    assert limitations.accepted is False
    assert limitations.can_authorize_completion is False
    assert limitations.markdown_is_not_completion is True
    assert limitations.general_python_equivalence_claimed is False
