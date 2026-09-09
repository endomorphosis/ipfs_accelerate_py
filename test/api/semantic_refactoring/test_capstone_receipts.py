"""Independent contract tests for SPAR-049 required-mode self-hosted capstone."""

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
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.procedure_adapter import (
    AdversarialPolarity,
    CompilationStatus,
    TrajectoryOutcome,
    compile_refactor_procedure,
    dry_run_refactor_procedure,
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
from test.api.semantic_refactoring.test_end_to_end_acceptance import (
    load_acceptance_matrix,
    run_acceptance_scenario,
)


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "benchmarks/agent_supervisor/semantic_refactoring/capstone_report.json",
    "test/api/semantic_refactoring/test_capstone_receipts.py",
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
RESIDUAL_TREE_ID = "6c8203a6e27a887d229ce21ffd35f9c2e094abcb"
REPORT_RELATIVE = "benchmarks/agent_supervisor/semantic_refactoring/capstone_report.json"
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
REAL_CURRENT_TREE_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/fixtures/real_current_tree.json"
)
RESIDUAL_PACKET_CID = (
    "baguqeerafe2wlut3zh2bebeyp5d6m2xu6xuvkfpqpuujgrlidfq7gvil6kca"
)
TASK_CID = "baguqeerazs7yclxi7roljovkm2k3cq3w2iz7uxxadpp7p4ezben5765m3b3a"
OBLIGATION_ID = "baguqeerasqnzvok3us4dh23fxj4jciiygp7opqkgrofeh6mbiihjkkuxiaga"
FOREST_ID = (
    "sha256:622ba35978e1aa8cc211032a43da88a1bf11dea6d091ae6dd667ba5bd444459f"
)
POLICY_ROOT = "baguqeeravnmu4tddkf5ozkn7opsfedilg6dqef6yafbenlm7hrfejszpluqq"
REPOSITORY_ID = "repository:semantic-preserving-autonomous-remodularization-v1"
TOKENIZER_ID = "spar-benchmark/utf8-bytes-div4@1"
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)
VALIDATION_COMMANDS = (
    "python3 -m pytest -q test/api/semantic_refactoring/test_capstone_receipts.py",
)
RESIDUAL_LIMITS = {
    "max_bytes": 24576,
    "max_tokens": 6144,
    "max_capsule_bytes": 16384,
}

TASK_ID: str = "SPAR-049"
GOAL_ID: str = "SPAR-G082"
PROGRAM: str = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: str = "required-mode self-hosted capstone"
AUTHORITY_OWNER: str = "ipfs_accelerate_py"
ANALYZER_ID: str = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.capstone_receipts@1"
)
PREDECESSOR_TASK_IDS: tuple[str, ...] = ("SPAR-048",)
SPAR_CAPSTONE_REPORT_INTERFACE: str = "SparCapstoneReport@1"
SPAR_CAPSTONE_RECEIPT_INTERFACE: str = "SparCapstoneReceipt@1"
SPAR_CAPSTONE_WAVE_INTERFACE: str = "SparCapstoneWaveResult@1"
REQUIRED_MODE_CAPSTONE_ROOT_INTERFACE: str = "RequiredModeCapstoneRoot@1"
NOMINATED_CAPSTONE_TRANSITION_INTERFACE: str = "NominatedCapstoneTransition@1"
SPAR_CAPSTONE_REPORT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-capstone-report@1"
)
SPAR_CAPSTONE_RECEIPT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-capstone-receipt@1"
)
SPAR_CAPSTONE_WAVE_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-capstone-wave-result@1"
)
REQUIRED_MODE_CAPSTONE_ROOT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/required-mode-capstone-root@1"
)
NOMINATED_CAPSTONE_TRANSITION_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/nominated-capstone-transition@1"
)
CAPSTONE_CONTRACT_VERSION: str = "1"
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
)
WAVE_EXPECTATIONS: tuple[tuple[str, str, str], ...] = (
    ("required_mode.bind", "nominated", ""),
    ("real_module.decompose_verified_waves", "nominated", ""),
    ("procedure.compile_reusable", "nominated", ""),
    ("later_procedure.no_general_llm", "nominated", ""),
    ("general_llm.network_denied", "unavailable", "network_denied_general_llm"),
)
NEGATIVE_EVIDENCE: tuple[dict[str, Any], ...] = (
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "stale_authoritative_reuse",
        "status": "rejected",
        "terminal": "stale_tree",
        "wave_id": "reuse.stale_tree",
        "writes_repository": False,
    },
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "false_task_completions",
        "status": "rejected",
        "terminal": "self_authorization_rejected",
        "wave_id": "authority.self_authorization",
        "writes_repository": False,
    },
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "false_task_completions",
        "status": "rejected",
        "terminal": "completion_bypass_rejected",
        "wave_id": "authority.completion_bypass",
        "writes_repository": False,
    },
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "unauthorized_mutations",
        "status": "rejected",
        "terminal": "worker_mode_change_rejected",
        "wave_id": "rollout.worker_mode_change",
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
)


class CapstoneReceiptError(RuntimeError):
    """Fail-closed SPAR-049 capstone-receipt contract violation."""


def provider_free_exports() -> tuple[str, ...]:
    return (
        "NominatedCapstoneTransition",
        "RequiredModeCapstoneRoot",
        "SparCapstoneReceipt",
        "SparCapstoneReport",
        "SparCapstoneWaveResult",
        "load_capstone_report",
        "nominate_capstone_receipts",
        "run_capstone_receipts",
        "sealed_capstone_report_payload",
    )


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & set(CAPSULE_TYPES)
    if overlap:
        raise CapstoneReceiptError(
            f"capstone receipts must not define competing types: {sorted(overlap)}"
        )


def capstone_cid_profile() -> dict[str, str]:
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
        raise CapstoneReceiptError(f"{relative} must be a JSON object")
    return payload


def _terminal_or_none(value: str) -> str | None:
    return value or None


def load_capstone_report() -> dict[str, Any]:
    payload = _load_json(REPORT_RELATIVE)
    if payload.get("schema") != SPAR_CAPSTONE_REPORT_SCHEMA:
        raise CapstoneReceiptError("unsupported capstone report schema")
    if payload.get("interface") != SPAR_CAPSTONE_REPORT_INTERFACE:
        raise CapstoneReceiptError("unsupported capstone report interface")
    if payload.get("task_id") != TASK_ID:
        raise CapstoneReceiptError("capstone report task_id must remain SPAR-049")
    if payload.get("nomination_only") is not True:
        raise CapstoneReceiptError("capstone report must remain nomination_only")
    for flag in _AUTHORITY_FLAGS:
        if payload.get(flag) is True:
            raise CapstoneReceiptError(f"capstone report cannot claim {flag}")
    if payload.get("promotion_eligible") is True:
        raise CapstoneReceiptError("capstone report cannot self-promote")
    if payload.get("network") != NETWORK_DENY:
        raise CapstoneReceiptError("capstone report network must remain deny")
    if payload.get("rollout_mode") != REQUIRED_MODE:
        raise CapstoneReceiptError("capstone report rollout_mode must remain required")
    if payload.get("top_level_completion_bypass") is True:
        raise CapstoneReceiptError("capstone cannot bypass top-level completion")
    return payload


def sealed_capstone_report_payload() -> dict[str, Any]:
    """Return the sealed SPAR-049 report contract bound to predecessors."""

    preregistration = load_preregistration()
    wave_ids = tuple(item[0] for item in WAVE_EXPECTATIONS)
    waves = [
        {
            "expected_status": status,
            "expected_terminal": _terminal_or_none(terminal),
            "general_llm_invoked": False,
            "in_denominator": True,
            "promotes": False,
            "required_mode": True,
            "wave_id": wave_id,
            "writes_repository": False,
        }
        for wave_id, status, terminal in WAVE_EXPECTATIONS
    ]
    nominated = sum(1 for item in waves if item["expected_status"] == "nominated")
    unavailable = sum(1 for item in waves if item["expected_status"] == "unavailable")
    rejected = sum(1 for item in NEGATIVE_EVIDENCE if item["status"] == "rejected")
    denominators = {
        "escalated": 0,
        "failed": 0,
        "human_reviewed": 0,
        "nominated": nominated,
        "rejected": rejected,
        "total_waves": nominated + unavailable + rejected,
        "unavailable": unavailable,
    }
    return {
        "analyzer_id": ANALYZER_ID,
        "authority_roots": {
            "policy_root": POLICY_ROOT,
            "repository_forest_cid": FOREST_ID,
        },
        "can_authorize_completion": False,
        "can_authorize_transition": False,
        "can_create_authority": False,
        "corpus_manifest_path": CORPUS_MANIFEST_RELATIVE,
        "denominator_policy": preregistration["denominator_policy"],
        "denominators": denominators,
        "efficiency_targets_met": False,
        "failed_efficiency_target": preregistration["failed_efficiency_target"],
        "goal_id": GOAL_ID,
        "interface": SPAR_CAPSTONE_REPORT_INTERFACE,
        "model_output_is_proposal_only": True,
        "negative_evidence": [dict(item) for item in NEGATIVE_EVIDENCE],
        "network": NETWORK_DENY,
        "nomination_only": True,
        "obligation_ids": [OBLIGATION_ID],
        "observed_metrics": {
            "accepted_waves": 0,
            "general_llm_invoked": False,
            "later_promoted_procedure_no_general_llm_waves": 1,
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
        "schema": SPAR_CAPSTONE_REPORT_SCHEMA,
        "sealed_before_tuning": True,
        "self_hosted": True,
        "task_id": TASK_ID,
        "test_pass_is_not_completion": True,
        "top_level_completion_bypass": False,
        "tree_id": TREE_ID,
        "wave_ids": list(wave_ids),
        "waves": waves,
        "worker_may_change_mode": False,
        "worker_self_approval": False,
        "writes_repository": False,
        "zero_safety_floors": dict(preregistration["zero_safety_floors"]),
    }


def persist_sealed_capstone_report() -> dict[str, Any]:
    """Materialize the sealed report contract onto the owned write path."""

    payload = sealed_capstone_report_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = ROOT / REPORT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    return payload


@dataclass(frozen=True, slots=True)
class RequiredModeCapstoneRoot:
    """Nominated required-mode capstone root. Cannot advance accepted roots."""

    tree_id: str
    root_cid: str
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
    interface: str = REQUIRED_MODE_CAPSTONE_ROOT_INTERFACE
    schema: str = REQUIRED_MODE_CAPSTONE_ROOT_SCHEMA

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
class NominatedCapstoneTransition:
    """Nominated required-mode capstone transition. Cannot self-accept."""

    tree_id: str
    wave_id: str
    procedure_cid: str
    evidence_cid: str
    nominated: bool = True
    accepted: bool = False
    general_llm_invoked: bool = False
    writes_repository: bool = False
    task_id: str = TASK_ID
    interface: str = NOMINATED_CAPSTONE_TRANSITION_INTERFACE
    schema: str = NOMINATED_CAPSTONE_TRANSITION_SCHEMA

    def identity_payload(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "evidence_cid": self.evidence_cid,
            "general_llm_invoked": False,
            "interface": self.interface,
            "nominated": True,
            "procedure_cid": self.procedure_cid,
            "schema": self.schema,
            "task_id": TASK_ID,
            "tree_id": self.tree_id,
            "wave_id": self.wave_id,
            "writes_repository": False,
        }

    @property
    def transition_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())


@dataclass(frozen=True, slots=True)
class SparCapstoneReport:
    """Sealed SPAR-049 capstone report bound to SPAR-048 predecessors."""

    payload: Mapping[str, Any]
    report_cid: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "interface": SPAR_CAPSTONE_REPORT_INTERFACE,
            "payload": dict(self.payload),
            "report_cid": self.report_cid,
            "schema": SPAR_CAPSTONE_REPORT_SCHEMA,
            "task_id": TASK_ID,
        }


@dataclass(frozen=True, slots=True)
class SparCapstoneWaveResult:
    """One SPAR-049 capstone wave observation. Nomination-only."""

    wave_id: str
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
    interface: str = SPAR_CAPSTONE_WAVE_INTERFACE
    schema: str = SPAR_CAPSTONE_WAVE_SCHEMA

    def identity_payload(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "evidence_cid": self.evidence_cid,
            "expected_status": self.expected_status,
            "expected_terminal": self.expected_terminal,
            "general_llm_invoked": False,
            "in_denominator": True,
            "interface": self.interface,
            "nominated": True,
            "promotes": False,
            "required_mode": True,
            "schema": self.schema,
            "status": self.status,
            "terminal": self.terminal,
            "wave_id": self.wave_id,
            "writes_repository": False,
        }

    @property
    def result_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())


@dataclass(frozen=True, slots=True)
class SparCapstoneReceipt:
    """Nomination-only SPAR-049 required-mode capstone receipt."""

    tree_id: str
    report_cid: str
    wave_results: tuple[SparCapstoneWaveResult, ...]
    negative_evidence_cids: tuple[str, ...]
    denominators: Mapping[str, int]
    wave_ids: tuple[str, ...]
    root: RequiredModeCapstoneRoot
    later_transition: NominatedCapstoneTransition
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
    network: str = NETWORK_DENY
    rollout_mode: str = REQUIRED_MODE
    analyzer_id: str = ANALYZER_ID
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    interface: str = SPAR_CAPSTONE_RECEIPT_INTERFACE
    schema: str = SPAR_CAPSTONE_RECEIPT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "accepted",
            "analyzer_id",
            "can_authorize_completion",
            "can_authorize_transition",
            "can_create_authority",
            "denominators",
            "goal_id",
            "interface",
            "later_transition",
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
            "wave_ids",
            "wave_results",
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
            "goal_id": self.goal_id,
            "interface": self.interface,
            "later_transition": self.later_transition.identity_payload(),
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
            "wave_ids": list(self.wave_ids),
            "wave_results": [item.identity_payload() for item in self.wave_results],
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
    def from_dict(cls, data: Mapping[str, Any]) -> "SparCapstoneReceipt":
        excluded = set(data) & IDENTITY_EXCLUDED_FIELDS
        if excluded:
            raise CapstoneReceiptError(
                f"observational fields are excluded from identity: {sorted(excluded)}"
            )
        unknown = set(data) - cls._FIELDS
        if unknown:
            raise CapstoneReceiptError(
                f"unsupported SparCapstoneReceipt fields: {sorted(unknown)}"
            )
        payload = dict(data)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != SPAR_CAPSTONE_RECEIPT_SCHEMA:
            raise CapstoneReceiptError("unsupported capstone receipt schema")
        if payload.pop("interface") != SPAR_CAPSTONE_RECEIPT_INTERFACE:
            raise CapstoneReceiptError("unsupported capstone receipt interface")
        if payload.pop("accepted") is not False:
            raise CapstoneReceiptError("workers cannot self-approve SPAR-049")
        if payload.pop("nomination_only") is not True:
            raise CapstoneReceiptError("receipt must remain nomination_only")
        if payload.pop("nominated") is not True:
            raise CapstoneReceiptError("receipt must remain nominated")
        if payload.pop("promotion_eligible") is not False:
            raise CapstoneReceiptError("receipt cannot self-promote")
        if payload.pop("top_level_completion_bypass") is not False:
            raise CapstoneReceiptError("receipt cannot bypass top-level completion")
        if payload.pop("worker_may_change_mode") is not False:
            raise CapstoneReceiptError("workers cannot change rollout mode")
        if payload.pop("rollout_mode") != REQUIRED_MODE:
            raise CapstoneReceiptError("receipt rollout_mode must remain required")
        for flag in _AUTHORITY_FLAGS:
            if payload.pop(flag) is not False:
                raise CapstoneReceiptError(f"receipt cannot claim {flag}")
        if payload.pop("network") != NETWORK_DENY:
            raise CapstoneReceiptError("receipt network must remain deny")
        results = tuple(
            SparCapstoneWaveResult(
                wave_id=str(item["wave_id"]),
                status=str(item["status"]),
                evidence_cid=str(item["evidence_cid"]),
                expected_status=str(item["expected_status"]),
                expected_terminal=str(item.get("expected_terminal") or ""),
                terminal=str(item.get("terminal") or ""),
            )
            for item in payload["wave_results"]
        )
        root_payload = dict(payload["root"])
        later_payload = dict(payload["later_transition"])
        result = cls(
            tree_id=str(payload["tree_id"]),
            report_cid=str(payload["report_cid"]),
            wave_results=results,
            negative_evidence_cids=tuple(
                str(item) for item in payload["negative_evidence_cids"]
            ),
            denominators=dict(payload["denominators"]),
            wave_ids=tuple(str(item) for item in payload["wave_ids"]),
            root=RequiredModeCapstoneRoot(
                tree_id=str(root_payload["tree_id"]),
                root_cid=str(root_payload["root_cid"]),
                mode=str(root_payload.get("mode") or REQUIRED_MODE),
                gate_task=str(root_payload.get("gate_task") or REQUIRED_GATE_TASK),
            ),
            later_transition=NominatedCapstoneTransition(
                tree_id=str(later_payload["tree_id"]),
                wave_id=str(later_payload["wave_id"]),
                procedure_cid=str(later_payload["procedure_cid"]),
                evidence_cid=str(later_payload["evidence_cid"]),
            ),
            analyzer_id=str(payload["analyzer_id"]),
            task_id=str(payload["task_id"]),
            goal_id=str(payload["goal_id"]),
        )
        if claimed != result.receipt_cid:
            raise CapstoneReceiptError("SparCapstoneReceipt receipt_cid mismatch")
        return result


def _observed(
    *,
    status: str,
    evidence_cid: str,
    terminal: str = "",
) -> dict[str, Any]:
    if status not in DECLARED_STATUSES:
        raise CapstoneReceiptError(f"unsupported wave status {status!r}")
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


def _procedure_wave(label: str, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "receipt_cid": _cid(label),
        "packet_cids": [_cid(f"packet-{label}")],
        "write_paths": list(WRITE_PATHS),
        "status": "applied",
        "writes_repository": False,
        "executor_is_nomination_only": True,
    }
    fields.update(overrides)
    return fields


def _procedure_transition(label: str, *, wave_label: str, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "outcome": TrajectoryOutcome.ACCEPTED.value,
        "wave_receipt_cid": _cid(wave_label),
        "packet_cid": _cid(f"packet-{wave_label}"),
        "raw_source_cids": [_cid("source")],
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "evidence_class": "transition",
        "transition_cid": _cid(label),
    }
    fields.update(overrides)
    return fields


def _compile_reusable_procedure() -> Any:
    return compile_refactor_procedure(
        waves=[
            _procedure_wave("wave-a"),
            _procedure_wave("wave-b"),
            _procedure_wave("wave-held"),
        ],
        transitions=[
            _procedure_transition("t1", wave_label="wave-a"),
            _procedure_transition("t2", wave_label="wave-b"),
        ],
        held_out=[
            _procedure_transition(
                "held-out",
                wave_label="wave-held",
                transition_cid=_cid("held-out"),
            )
        ],
        adversarial=[
            {
                "case_id": "adv:boundary",
                "kind": "boundary",
                "polarity": AdversarialPolarity.MUST_SURVIVE.value,
                "tree_id": TREE_ID,
                "source_authority": "specification",
            }
        ],
    )


def _run_required_mode_bind() -> dict[str, Any]:
    contract = MODE_CONTRACTS[REQUIRED_MODE]
    if contract["gate_task"] != REQUIRED_GATE_TASK:
        raise CapstoneReceiptError("required gate must remain SPAR-043")
    if GATE_CAN_CHANGE_MODE is not False:
        raise CapstoneReceiptError("capstone cannot change rollout mode")
    if WORKER_MAY_CHANGE_MODE is not False:
        raise CapstoneReceiptError("workers cannot change rollout mode")
    root = RequiredModeCapstoneRoot(
        tree_id=TREE_ID,
        root_cid=_cid("required-mode-root"),
        mode=REQUIRED_MODE,
        gate_task=str(contract["gate_task"]),
    )
    if root.accepted or root.can_authorize_completion:
        raise CapstoneReceiptError("required-mode root cannot complete")
    return _observed(status="nominated", evidence_cid=root.identity_cid)


def _run_decompose_verified_waves() -> dict[str, Any]:
    admitted = admit_real_current_tree()
    if admitted["synthetic"] is not False:
        raise CapstoneReceiptError("capstone module must remain current-tree")
    if admitted["whole_repository_dump"] is not False:
        raise CapstoneReceiptError("capstone cannot dump the repository")
    if int(admitted["loc"]) < 1:
        raise CapstoneReceiptError("admitted current-tree module is empty")
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
        raise CapstoneReceiptError("decompose wave cannot complete or write")
    return _observed(status="nominated", evidence_cid=result.evidence_cid)


def _run_compile_reusable_procedure() -> dict[str, Any]:
    receipt = _compile_reusable_procedure()
    if receipt.status != CompilationStatus.NOMINATED.value:
        raise CapstoneReceiptError("reusable procedure must remain nominated")
    if receipt.can_authorize_completion is not False:
        raise CapstoneReceiptError("procedure compilation cannot complete")
    return _observed(status="nominated", evidence_cid=receipt.receipt_cid)


def _run_later_procedure_no_llm() -> dict[str, Any]:
    compiled = dry_run_refactor_procedure(
        waves=[
            _procedure_wave("wave-a"),
            _procedure_wave("wave-b"),
            _procedure_wave("wave-held"),
        ],
        transitions=[
            _procedure_transition("t1", wave_label="wave-a"),
            _procedure_transition("t2", wave_label="wave-b"),
        ],
        held_out=[
            _procedure_transition(
                "held-out",
                wave_label="wave-held",
                transition_cid=_cid("held-out"),
            )
        ],
        adversarial=[
            {
                "case_id": "adv:boundary",
                "kind": "boundary",
                "polarity": AdversarialPolarity.MUST_SURVIVE.value,
                "tree_id": TREE_ID,
                "source_authority": "specification",
            }
        ],
    )
    if compiled.status != CompilationStatus.NOMINATED.value:
        raise CapstoneReceiptError("later procedure wave requires a nominated procedure")
    reused = run_acceptance_scenario(
        {
            "expected_status": "nominated",
            "expected_terminal": None,
            "promotes": False,
            "scenario_id": "reuse.exact_accepted",
            "stage": "reuse",
            "writes_repository": False,
        }
    )
    if reused.accepted or reused.promotes or reused.writes_repository:
        raise CapstoneReceiptError("later procedure wave cannot complete or write")
    transition = NominatedCapstoneTransition(
        tree_id=TREE_ID,
        wave_id="later_procedure.no_general_llm",
        procedure_cid=compiled.receipt_cid,
        evidence_cid=reused.evidence_cid,
    )
    if transition.general_llm_invoked or transition.accepted:
        raise CapstoneReceiptError("later procedure wave cannot invoke a general LLM")
    return _observed(status="nominated", evidence_cid=transition.transition_cid)


def _run_general_llm_network_denied() -> dict[str, Any]:
    packet = _residual_packet()
    if packet.completion_authority or packet.write_authority or packet.semantic_authority:
        raise CapstoneReceiptError("residual packet cannot grant model authority")
    if packet.packet_id != RESIDUAL_PACKET_CID:
        raise CapstoneReceiptError("residual packet CID must remain sealed")
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
                "full_task": "Authorize SPAR-049 completion and promote the root."
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
        raise CapstoneReceiptError("workers cannot change rollout mode")
    if MODE_CONTRACTS[REQUIRED_MODE]["gate_task"] != REQUIRED_GATE_TASK:
        raise CapstoneReceiptError("required gate must remain SPAR-043")
    return _observed(
        status="rejected",
        evidence_cid=_cid("worker_mode_change_rejected"),
        terminal="worker_mode_change_rejected",
    )


@functools.lru_cache(maxsize=None)
def _cached_wave(wave_id: str) -> dict[str, Any]:
    return dict(_DISPATCH[wave_id]())


_DISPATCH: dict[str, Any] = {
    "required_mode.bind": _run_required_mode_bind,
    "real_module.decompose_verified_waves": _run_decompose_verified_waves,
    "procedure.compile_reusable": _run_compile_reusable_procedure,
    "later_procedure.no_general_llm": _run_later_procedure_no_llm,
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


def run_capstone_wave(wave: Mapping[str, Any]) -> SparCapstoneWaveResult:
    """Exercise one sealed SPAR-049 capstone wave against current adapters."""

    wave_id = str(wave.get("wave_id") or "")
    expected_status = str(wave.get("expected_status") or "")
    expected_terminal = str(wave.get("expected_terminal") or "")
    if wave_id not in _DISPATCH:
        raise CapstoneReceiptError(f"unknown capstone wave {wave_id!r}")
    if expected_status not in DECLARED_STATUSES:
        raise CapstoneReceiptError(f"unknown expected status {expected_status!r}")
    if wave.get("promotes") is True:
        raise CapstoneReceiptError("capstone waves cannot promote")
    if wave.get("writes_repository") is True:
        raise CapstoneReceiptError("capstone waves cannot write the repository")
    if wave.get("general_llm_invoked") is True:
        raise CapstoneReceiptError("SPAR-049 cannot invoke a general LLM")
    if wave.get("required_mode") is not True:
        raise CapstoneReceiptError("SPAR-049 waves must remain required-mode")
    observed = _cached_wave(wave_id)
    if observed["status"] != expected_status:
        raise CapstoneReceiptError(
            f"{wave_id} status {observed['status']!r} != {expected_status!r}"
        )
    if expected_terminal and observed.get("terminal") != expected_terminal:
        raise CapstoneReceiptError(
            f"{wave_id} terminal {observed.get('terminal')!r} != {expected_terminal!r}"
        )
    if observed["accepted"] or observed["promotes"] or observed["writes_repository"]:
        raise CapstoneReceiptError(f"{wave_id} cannot promote or write")
    if observed["general_llm_invoked"]:
        raise CapstoneReceiptError(f"{wave_id} invoked a general LLM")
    return SparCapstoneWaveResult(
        wave_id=wave_id,
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
        runner = _NEGATIVE_RUNNERS[str(item["wave_id"])]
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
            raise CapstoneReceiptError(f"{item['wave_id']} must remain rejected")
        if promotes or writes or accepted:
            raise CapstoneReceiptError(f"{item['wave_id']} cannot promote or write")
        evidence_cid = getattr(result, "evidence_cid", None) or result["evidence_cid"]
        cids.append(str(evidence_cid))
    return tuple(cids)


def run_capstone_receipts(
    report: Mapping[str, Any] | None = None,
) -> SparCapstoneReceipt:
    """Run the sealed SPAR-049 capstone. Current authority remains separate."""

    loaded = dict(report) if report is not None else load_capstone_report()
    sealed = sealed_capstone_report_payload()
    if loaded.get("wave_ids") != sealed["wave_ids"]:
        raise CapstoneReceiptError("capstone waves must remain sealed")
    if tuple(loaded.get("predecessor_task_ids") or ()) != PREDECESSOR_TASK_IDS:
        raise CapstoneReceiptError("predecessors must remain SPAR-048")
    preregistration = load_preregistration()
    if dict(loaded.get("zero_safety_floors") or {}) != dict(
        preregistration["zero_safety_floors"]
    ):
        raise CapstoneReceiptError("safety floors must match preregistration")
    corpus = load_benchmark_corpus_manifest()
    if dict(corpus["zero_safety_floors"]) != dict(loaded["zero_safety_floors"]):
        raise CapstoneReceiptError("corpus floors must match the capstone report")
    predecessor = load_adversarial_report()
    if predecessor["task_id"] != "SPAR-048":
        raise CapstoneReceiptError("predecessor report must remain SPAR-048")
    benchmark = load_benchmark_report()
    if benchmark["task_id"] != "SPAR-047":
        raise CapstoneReceiptError("benchmark predecessor must remain SPAR-047")
    matrix = load_acceptance_matrix()
    if matrix["task_id"] != "SPAR-046":
        raise CapstoneReceiptError("acceptance matrix predecessor must remain SPAR-046")
    covered = {str(item["wave_id"]) for item in loaded["waves"]}
    missing = [name for name in sealed["wave_ids"] if name not in covered]
    if missing:
        raise CapstoneReceiptError(f"capstone report missing waves: {missing}")
    if loaded.get("promotion_eligible") is True:
        raise CapstoneReceiptError("capstone cannot self-promote")
    if loaded.get("residual_packet_cid") != RESIDUAL_PACKET_CID:
        raise CapstoneReceiptError("residual packet CID must remain sealed")
    if loaded.get("rollout_mode") != REQUIRED_MODE:
        raise CapstoneReceiptError("capstone rollout_mode must remain required")
    if loaded.get("top_level_completion_bypass") is True:
        raise CapstoneReceiptError("capstone cannot bypass top-level completion")
    results = tuple(run_capstone_wave(item) for item in loaded["waves"])
    if any(item.promotes or item.accepted for item in results):
        raise CapstoneReceiptError("no capstone wave may be promoted")
    negative = _run_negative_evidence()
    identity = {
        "payload": loaded,
        "task_id": TASK_ID,
        "tree_id": TREE_ID,
    }
    later = _cached_wave("later_procedure.no_general_llm")
    compiled = _cached_wave("procedure.compile_reusable")
    root_observed = _cached_wave("required_mode.bind")
    return SparCapstoneReceipt(
        tree_id=TREE_ID,
        report_cid=cid_for_dag_json(identity),
        wave_results=results,
        negative_evidence_cids=negative,
        denominators=dict(loaded["denominators"]),
        wave_ids=tuple(str(item) for item in loaded["wave_ids"]),
        root=RequiredModeCapstoneRoot(
            tree_id=TREE_ID,
            root_cid=str(root_observed["evidence_cid"]),
        ),
        later_transition=NominatedCapstoneTransition(
            tree_id=TREE_ID,
            wave_id="later_procedure.no_general_llm",
            procedure_cid=str(compiled["evidence_cid"]),
            evidence_cid=str(later["evidence_cid"]),
        ),
    )


def nominate_capstone_receipts() -> SparCapstoneReceipt:
    """Nominate the SPAR-049 report. Independent validation remains separate."""

    return run_capstone_receipts()


@functools.lru_cache(maxsize=1)
def _cached_nomination() -> SparCapstoneReceipt:
    return nominate_capstone_receipts()


def dry_run_capstone_receipts() -> SparCapstoneReceipt:
    return _cached_nomination()


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-049"
    assert GOAL_ID == "SPAR-G082"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-048",)
    assert SPAR_CAPSTONE_REPORT_INTERFACE == "SparCapstoneReport@1"
    assert SPAR_CAPSTONE_RECEIPT_INTERFACE == "SparCapstoneReceipt@1"
    assert SPAR_CAPSTONE_WAVE_INTERFACE == "SparCapstoneWaveResult@1"
    assert REQUIRED_MODE_CAPSTONE_ROOT_INTERFACE == "RequiredModeCapstoneRoot@1"
    assert NOMINATED_CAPSTONE_TRANSITION_INTERFACE == "NominatedCapstoneTransition@1"
    assert CAPSTONE_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("capstone_receipts@1")
    assert TEST_PATH.is_file()
    assert (ROOT / REPORT_RELATIVE).is_file()
    assert WRITE_SCOPE == (
        REPORT_RELATIVE,
        "test/api/semantic_refactoring/test_capstone_receipts.py",
    )
    assert (ROOT / PREREGISTRATION_RELATIVE).is_file()
    assert (ROOT / CORPUS_MANIFEST_RELATIVE).is_file()
    assert (ROOT / ACCEPTANCE_MATRIX_RELATIVE).is_file()
    assert (ROOT / BENCHMARK_REPORT_RELATIVE).is_file()
    assert (ROOT / ADVERSARIAL_REPORT_RELATIVE).is_file()
    assert (ROOT / REAL_CURRENT_TREE_RELATIVE).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "required-mode self-hosted capstone"
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
    profile = capstone_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    report = load_capstone_report()
    assert report["nomination_only"] is True
    assert report["can_authorize_completion"] is False
    assert report["can_authorize_transition"] is False
    assert report["can_create_authority"] is False
    assert report["worker_self_approval"] is False
    assert report["worker_may_change_mode"] is False
    assert report["model_output_is_proposal_only"] is True
    assert report["test_pass_is_not_completion"] is True
    assert report["writes_repository"] is False
    assert report["projection_is_authority"] is False
    assert report["promotion_eligible"] is False
    assert report["top_level_completion_bypass"] is False
    assert report["self_hosted"] is True
    assert report["rollout_mode"] == REQUIRED_MODE


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(TEST_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "SparCapstoneReport" in names
    assert "SparCapstoneWaveResult" in names
    assert "SparCapstoneReceipt" in names
    assert "RequiredModeCapstoneRoot" in names
    assert "NominatedCapstoneTransition" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "SparCapstoneReceipt" in exports
    assert "RequiredModeCapstoneRoot" in exports
    assert "NominatedCapstoneTransition" in exports
    assert "nominate_capstone_receipts" in exports
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for forbidden in (
        "authorize_completion",
        "promote_root",
        "dump_repository",
        "admit_third_party_source",
        "overwrite_root_conflict",
    ):
        assert forbidden not in functions
        assert forbidden not in names


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_committed_report_matches_sealed_payload() -> None:
    payload = sealed_capstone_report_payload()
    report = load_capstone_report()
    assert report == payload
    assert report["residual_packet_cid"] == RESIDUAL_PACKET_CID
    assert report["residual_target_ids"] == [TASK_ID]
    assert report["obligation_ids"] == [OBLIGATION_ID]
    assert report["sealed_before_tuning"] is True
    assert report["authority_roots"]["policy_root"] == POLICY_ROOT
    assert report["authority_roots"]["repository_forest_cid"] == FOREST_ID


def test_report_covers_required_waves_and_floors() -> None:
    report = load_capstone_report()
    preregistration = load_preregistration()
    corpus = load_benchmark_corpus_manifest()
    expected_waves = [item[0] for item in WAVE_EXPECTATIONS]
    assert report["wave_ids"] == expected_waves
    assert tuple(report["predecessor_task_ids"]) == PREDECESSOR_TASK_IDS
    covered = {item["wave_id"] for item in report["waves"]}
    assert covered == set(expected_waves)
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
    for item in report["waves"]:
        assert item["promotes"] is False
        assert item["writes_repository"] is False
        assert item["in_denominator"] is True
        assert item["general_llm_invoked"] is False
        assert item["required_mode"] is True
        assert item["expected_status"] in DECLARED_STATUSES
    assert report["required_gate_task"] == REQUIRED_GATE_TASK
    assert report["real_current_tree_path"] == REAL_CURRENT_TREE_RELATIVE


def test_denominators_retain_honest_non_success_waves() -> None:
    report = load_capstone_report()
    buckets = report["denominators"]
    for name in DENOMINATOR_BUCKETS:
        assert name in buckets
        assert type(buckets[name]) is int
        assert buckets[name] >= 0
    assert buckets["unavailable"] == 1
    assert buckets["rejected"] == 4
    assert buckets["failed"] == 0
    assert buckets["escalated"] == 0
    assert buckets["human_reviewed"] == 0
    assert buckets["nominated"] == 4
    assert buckets["total_waves"] == 9
    assert buckets["nominated"] + buckets["unavailable"] + buckets["rejected"] == (
        buckets["total_waves"]
    )


def test_predecessors_remain_readable_without_dump() -> None:
    report = load_capstone_report()
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
    predecessor = load_adversarial_report()
    assert predecessor["task_id"] == "SPAR-048"
    benchmark = load_benchmark_report()
    assert benchmark["task_id"] == "SPAR-047"
    matrix = load_acceptance_matrix()
    assert matrix["task_id"] == "SPAR-046"


def test_provider_tokenizer_criteria_are_exact_and_unavailable_llm() -> None:
    report = load_capstone_report()
    criteria = report["provider_tokenizer"]
    assert criteria["network"] == NETWORK_DENY
    assert criteria["provider_available"] is False
    assert criteria["live_model_channel"] is False
    assert criteria["general_llm_invoked"] is False
    assert criteria["tokenizer_id"] == TOKENIZER_ID
    assert criteria["tokenizer_available"] is True
    assert criteria["tokenizer_identity_required"] is True
    metrics = report["observed_metrics"]
    assert metrics["accepted_waves"] == 0
    assert metrics["general_llm_invoked"] is False
    assert metrics["required_coverage_loss"] == 0
    assert metrics["later_promoted_procedure_no_general_llm_waves"] == 1
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


def test_each_wave_is_exercised_without_promotion() -> None:
    receipt = dry_run_capstone_receipts()
    waves = tuple(item.wave_id for item in receipt.wave_results)
    assert waves == receipt.wave_ids
    by_id = {item.wave_id: item for item in receipt.wave_results}
    assert by_id["required_mode.bind"].status == "nominated"
    assert by_id["real_module.decompose_verified_waves"].status == "nominated"
    assert by_id["procedure.compile_reusable"].status == "nominated"
    assert by_id["later_procedure.no_general_llm"].status == "nominated"
    assert by_id["general_llm.network_denied"].status == "unavailable"
    assert by_id["general_llm.network_denied"].terminal == "network_denied_general_llm"
    for item in receipt.wave_results:
        assert item.accepted is False
        assert item.promotes is False
        assert item.writes_repository is False
        assert item.general_llm_invoked is False
        assert item.required_mode is True
        assert item.in_denominator is True
    assert receipt.root.mode == REQUIRED_MODE
    assert receipt.root.accepted is False
    assert receipt.later_transition.wave_id == "later_procedure.no_general_llm"
    assert receipt.later_transition.accepted is False
    assert receipt.later_transition.general_llm_invoked is False


def test_fail_closed_negative_evidence_stays_in_denominator() -> None:
    receipt = dry_run_capstone_receipts()
    assert len(receipt.negative_evidence_cids) == 4
    report = load_capstone_report()
    floors = dict(report["zero_safety_floors"])
    by_wave = {item["wave_id"]: item for item in report["negative_evidence"]}
    assert by_wave["reuse.stale_tree"]["terminal"] == "stale_tree"
    assert by_wave["authority.self_authorization"]["terminal"] == (
        "self_authorization_rejected"
    )
    assert by_wave["authority.completion_bypass"]["terminal"] == (
        "completion_bypass_rejected"
    )
    assert by_wave["rollout.worker_mode_change"]["terminal"] == (
        "worker_mode_change_rejected"
    )
    assert floors["stale_authoritative_reuse"] == 0
    assert floors["false_task_completions"] == 0
    assert floors["unauthorized_mutations"] == 0
    assert floors["test_or_proof_weakening"] == 0
    assert floors["root_conflict_overwrite"] == 0


def test_nomination_receipt_cannot_complete_or_self_approve() -> None:
    receipt = dry_run_capstone_receipts()
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
    assert receipt.rollout_mode == REQUIRED_MODE
    assert receipt.network == NETWORK_DENY
    assert receipt.task_id == TASK_ID
    assert receipt.goal_id == GOAL_ID
    assert receipt.tree_id == TREE_ID
    encoded = receipt.to_dict()
    restored = SparCapstoneReceipt.from_dict(encoded)
    assert restored.receipt_cid == receipt.receipt_cid
    assert restored == receipt
    dry = dry_run_capstone_receipts()
    assert dry.receipt_cid == receipt.receipt_cid
    with pytest.raises(CapstoneReceiptError, match="self-approve"):
        SparCapstoneReceipt.from_dict({**encoded, "accepted": True})
    with pytest.raises(CapstoneReceiptError, match="can_authorize_completion"):
        SparCapstoneReceipt.from_dict({**encoded, "can_authorize_completion": True})
    with pytest.raises(CapstoneReceiptError, match="self-promote"):
        SparCapstoneReceipt.from_dict({**encoded, "promotion_eligible": True})
    with pytest.raises(CapstoneReceiptError, match="bypass"):
        SparCapstoneReceipt.from_dict({**encoded, "top_level_completion_bypass": True})
    with pytest.raises(CapstoneReceiptError, match="change rollout mode"):
        SparCapstoneReceipt.from_dict({**encoded, "worker_may_change_mode": True})


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    first = run_capstone_receipts()
    second = run_capstone_receipts()
    assert first.receipt_cid == second.receipt_cid
    assert [item.result_cid for item in first.wave_results] == [
        item.result_cid for item in second.wave_results
    ]
    assert first.writes_repository is False
    assert DRY_RUN_MUTATES is False


def test_identity_excludes_observational_fields() -> None:
    receipt = dry_run_capstone_receipts()
    encoded = receipt.to_dict()
    for field in IDENTITY_EXCLUDED_FIELDS:
        assert field not in encoded
        with pytest.raises(CapstoneReceiptError, match="observational"):
            SparCapstoneReceipt.from_dict({**encoded, field: "now"})


def test_vector_or_model_cannot_admit_or_promote() -> None:
    report = dict(load_capstone_report())
    with pytest.raises(CapstoneReceiptError, match="missing waves"):
        run_capstone_receipts(
            {
                **report,
                "waves": [
                    {
                        "expected_status": "nominated",
                        "expected_terminal": None,
                        "general_llm_invoked": False,
                        "in_denominator": True,
                        "promotes": False,
                        "required_mode": True,
                        "wave_id": "vector.similarity_admit",
                        "writes_repository": False,
                    }
                ],
            }
        )
    with pytest.raises(CapstoneReceiptError, match="cannot promote"):
        run_capstone_wave(
            {
                "expected_status": "nominated",
                "expected_terminal": None,
                "general_llm_invoked": False,
                "in_denominator": True,
                "promotes": True,
                "required_mode": True,
                "wave_id": "required_mode.bind",
                "writes_repository": False,
            }
        )
    with pytest.raises(CapstoneReceiptError, match="general LLM"):
        run_capstone_wave(
            {
                "expected_status": "nominated",
                "expected_terminal": None,
                "general_llm_invoked": True,
                "in_denominator": True,
                "promotes": False,
                "required_mode": True,
                "wave_id": "later_procedure.no_general_llm",
                "writes_repository": False,
            }
        )


def test_missing_wave_and_floor_drift_fail_closed() -> None:
    report = dict(load_capstone_report())
    truncated = [
        item
        for item in report["waves"]
        if item["wave_id"] != "later_procedure.no_general_llm"
    ]
    with pytest.raises(CapstoneReceiptError, match="missing waves"):
        run_capstone_receipts({**report, "waves": truncated})
    drifted = dict(report["zero_safety_floors"])
    drifted["false_task_completions"] = 1
    with pytest.raises(CapstoneReceiptError, match="safety floors"):
        run_capstone_receipts({**report, "zero_safety_floors": drifted})
    with pytest.raises(CapstoneReceiptError, match="cannot self-promote"):
        run_capstone_receipts({**report, "promotion_eligible": True})
    with pytest.raises(CapstoneReceiptError, match="bypass"):
        run_capstone_receipts({**report, "top_level_completion_bypass": True})
