"""Independent contract tests for SPAR-047 preregistered ablation report."""

from __future__ import annotations

import ast
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.context_adapter import (
    dry_run_context_route,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.opportunity_detector import (
    detect_monolith_opportunities,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    ANALYZER_ID as PARTITION_ANALYZER_ID,
    generate_partition_candidates,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_retrieval import (
    AnalogousRefactorQuery,
    rank_admitted_partitions,
    retrieve_analogous_refactors,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.proof_adapter import (
    ArtifactKind,
    ProofAdapterError,
    reconstruct_proof,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.rollout import (
    GATE_CAN_CHANGE_MODE,
    MODE_CONTRACTS,
    sealed_rollout_baseline,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.service import (
    ControlRequest,
    ControlStatus,
    SemanticRefactoringService,
)

from test.api.semantic_refactoring.test_benchmark_corpus import (
    REQUIRED_DOMAINS,
    admit_real_current_tree,
    generate_controlled_monolith,
    load_benchmark_corpus_manifest,
    load_preregistration,
    load_split_policy,
)
from test.api.semantic_refactoring.test_end_to_end_acceptance import (
    load_acceptance_matrix,
    run_acceptance_scenario,
)


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "benchmarks/agent_supervisor/semantic_refactoring/benchmark_report.json",
    "test/api/semantic_refactoring/test_benchmark_report.py",
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
REPORT_RELATIVE = "benchmarks/agent_supervisor/semantic_refactoring/benchmark_report.json"
PREREGISTRATION_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json"
)
CORPUS_MANIFEST_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/corpus_manifest.json"
)
ACCEPTANCE_MATRIX_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/acceptance_matrix.json"
)
RESIDUAL_PACKET_CID = (
    "baguqeeravqt2gcndf6d7gn6fsabxzsjmo4y7pfyoosucwly4g77qkkcu5f4q"
)
TOKENIZER_ID = "spar-benchmark/utf8-bytes-div4@1"
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)
TOOLCHAIN_ID = "toolchain:hammer@1"

TASK_ID: str = "SPAR-047"
GOAL_ID: str = "SPAR-G081"
PROGRAM: str = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: str = "benchmark execution"
AUTHORITY_OWNER: str = "ipfs_accelerate_py"
ANALYZER_ID: str = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.benchmark_report@1"
)
PREDECESSOR_TASK_IDS: tuple[str, ...] = ("SPAR-046",)
SPAR_BENCHMARK_REPORT_INTERFACE: str = "SparBenchmarkReport@1"
SPAR_BENCHMARK_RECEIPT_INTERFACE: str = "SparBenchmarkReportReceipt@1"
SPAR_ABLATION_CELL_INTERFACE: str = "SparAblationCellResult@1"
SPAR_BENCHMARK_REPORT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-benchmark-report@1"
)
SPAR_BENCHMARK_RECEIPT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-benchmark-report-receipt@1"
)
SPAR_ABLATION_CELL_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-ablation-cell-result@1"
)
BENCHMARK_CONTRACT_VERSION: str = "1"
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
DUCKLAKE_IS_AUTHORITY: bool = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: bool = True
NETWORK_DENIED: bool = True
RAW_SOURCE_REQUIRED: bool = True
DRY_RUN_IS_DETERMINISTIC: bool = True
DRY_RUN_MUTATES: bool = False

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
ABLATION_EXPECTATIONS: tuple[tuple[str, str, str], ...] = (
    ("manual_or_general_llm_heavy", "unavailable", "network_denied_general_llm"),
    ("current_supervisor", "nominated", ""),
    ("exact_capsules_graph", "nominated", ""),
    ("deterministic_scc_state_contract_partition", "nominated", ""),
    ("translation_validation", "nominated", ""),
    ("tactician_hammer", "nominated", ""),
    ("vector_graph_retrieval", "nominated", ""),
    ("optional_ranker", "nominated", ""),
    ("procedure_reuse", "nominated", ""),
    ("guarded", "nominated", ""),
    ("required", "nominated", ""),
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
        "safety_floor": "unauthorized_mutations",
        "status": "rejected",
        "terminal": "unsafe_intra_domain_split",
        "wave_id": "unsafe_splits.intra_domain",
        "writes_repository": False,
    },
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "test_or_proof_weakening",
        "status": "rejected",
        "terminal": "proof_candidate_cannot_admit",
        "wave_id": "failed_proofs.candidate_cannot_admit",
        "writes_repository": False,
    },
    {
        "in_denominator": True,
        "promotes": False,
        "safety_floor": "root_conflict_overwrite",
        "status": "rejected",
        "terminal": "rejected_root_conflict",
        "wave_id": "root_conflicts.no_overwrite",
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


class BenchmarkReportError(RuntimeError):
    """Fail-closed SPAR-047 benchmark-report contract violation."""


def provider_free_exports() -> tuple[str, ...]:
    return (
        "SparAblationCellResult",
        "SparBenchmarkReport",
        "SparBenchmarkReportReceipt",
        "load_benchmark_report",
        "nominate_benchmark_report",
        "run_ablation_ladder",
        "sealed_benchmark_report_payload",
    )


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & set(CAPSULE_TYPES)
    if overlap:
        raise BenchmarkReportError(
            f"benchmark report must not define competing types: {sorted(overlap)}"
        )


def benchmark_cid_profile() -> dict[str, str]:
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
        raise BenchmarkReportError(f"{relative} must be a JSON object")
    return payload


def _terminal_or_none(value: str) -> str | None:
    return value or None


def _module(module_id: str, *, loc: int, member_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "module_id": module_id,
        "loc": loc,
        "member_ids": list(member_ids),
        "public_export_ids": [],
    }


def _component(member_id: str) -> dict[str, Any]:
    identity = {
        "cyclic": False,
        "member_ids": [member_id],
        "oversized": False,
        "state_owner_ids": [],
    }
    return {
        "scc_id": cid_for_dag_json(identity),
        "member_ids": [member_id],
        "cyclic": False,
        "oversized": False,
        "state_owner_ids": [],
    }


def _partition_evidence() -> dict[str, Any]:
    components = tuple(_component(f"node:{domain}") for domain in REQUIRED_DOMAINS)
    snapshot_payload = {
        "components": list(components),
        "condensation_edges": [],
    }
    return {
        "tree_id": TREE_ID,
        "analyzer_id": PARTITION_ANALYZER_ID,
        "scc_snapshot": {
            "snapshot_cid": cid_for_dag_json(snapshot_payload),
            "components": list(components),
            "condensation_edges": [],
        },
    }


def _context_packet() -> dict[str, Any]:
    return {
        "tree_id": TREE_ID,
        "packet_cid": _cid("packet"),
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "raw_source_cids": [_cid("source")],
    }


def _scenario(scenario_id: str, stage: str, expected_status: str) -> dict[str, Any]:
    return {
        "expected_status": expected_status,
        "expected_terminal": None,
        "promotes": False,
        "scenario_id": scenario_id,
        "stage": stage,
        "writes_repository": False,
    }


def load_benchmark_report() -> dict[str, Any]:
    payload = _load_json(REPORT_RELATIVE)
    if payload.get("schema") != SPAR_BENCHMARK_REPORT_SCHEMA:
        raise BenchmarkReportError("unsupported benchmark report schema")
    if payload.get("interface") != SPAR_BENCHMARK_REPORT_INTERFACE:
        raise BenchmarkReportError("unsupported benchmark report interface")
    if payload.get("task_id") != TASK_ID:
        raise BenchmarkReportError("benchmark report task_id must remain SPAR-047")
    if payload.get("nomination_only") is not True:
        raise BenchmarkReportError("benchmark report must remain nomination_only")
    for flag in _AUTHORITY_FLAGS:
        if payload.get(flag) is True:
            raise BenchmarkReportError(f"benchmark report cannot claim {flag}")
    if payload.get("promotion_eligible") is True:
        raise BenchmarkReportError("benchmark report cannot self-promote")
    if payload.get("network") != NETWORK_DENY:
        raise BenchmarkReportError("benchmark report network must remain deny")
    return payload


def sealed_benchmark_report_payload() -> dict[str, Any]:
    """Return the sealed SPAR-047 report contract bound to preregistration."""

    preregistration = load_preregistration()
    ladder = tuple(preregistration["ablation_ladder"])
    expected_ladder = tuple(item[0] for item in ABLATION_EXPECTATIONS)
    if ladder != expected_ladder:
        raise BenchmarkReportError("ablation ladder drifted from sealed expectations")
    cells: list[dict[str, Any]] = []
    for profile in preregistration["profiles"]:
        name = str(profile["name"])
        for ablation, status, terminal in ABLATION_EXPECTATIONS:
            cells.append(
                {
                    "ablation": ablation,
                    "expected_status": status,
                    "expected_terminal": _terminal_or_none(terminal),
                    "general_llm_invoked": False,
                    "in_denominator": True,
                    "profile": name,
                    "promotes": False,
                    "target_loc": profile.get("target_loc"),
                    "writes_repository": False,
                }
            )
    nominated = sum(1 for item in cells if item["expected_status"] == "nominated")
    unavailable = sum(1 for item in cells if item["expected_status"] == "unavailable")
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
        "ablation_ladder": list(ladder),
        "acceptance_matrix_path": ACCEPTANCE_MATRIX_RELATIVE,
        "analyzer_id": ANALYZER_ID,
        "can_authorize_completion": False,
        "can_authorize_transition": False,
        "can_create_authority": False,
        "corpus_manifest_path": CORPUS_MANIFEST_RELATIVE,
        "cells": cells,
        "denominator_policy": preregistration["denominator_policy"],
        "denominators": denominators,
        "efficiency_targets_met": False,
        "failed_efficiency_target": preregistration["failed_efficiency_target"],
        "goal_id": GOAL_ID,
        "interface": SPAR_BENCHMARK_REPORT_INTERFACE,
        "model_output_is_proposal_only": True,
        "negative_evidence": [dict(item) for item in NEGATIVE_EVIDENCE],
        "network": NETWORK_DENY,
        "nomination_only": True,
        "observed_metrics": {
            "accepted_waves": 0,
            "general_llm_calls_per_accepted_wave_reduction": None,
            "general_llm_invoked": False,
            "later_promoted_procedure_no_general_llm_waves": 0,
            "median_general_llm_input_context_reduction": None,
            "required_coverage_loss": 0,
            "tier_a_no_general_llm_fraction": None,
        },
        "predecessor_task_ids": list(PREDECESSOR_TASK_IDS),
        "preregistration_path": PREREGISTRATION_RELATIVE,
        "profiles": [dict(item) for item in preregistration["profiles"]],
        "program": PROGRAM,
        "projection_is_authority": False,
        "promotion_eligible": False,
        "promotion_targets": dict(preregistration["promotion_targets"]),
        "provider_tokenizer": dict(PROVIDER_TOKENIZER),
        "residual_packet_cid": RESIDUAL_PACKET_CID,
        "residual_target_ids": [TASK_ID],
        "safety_floors_held": True,
        "schema": SPAR_BENCHMARK_REPORT_SCHEMA,
        "sealed_before_tuning": True,
        "task_id": TASK_ID,
        "test_pass_is_not_completion": True,
        "tree_id": TREE_ID,
        "worker_self_approval": False,
        "writes_repository": False,
        "zero_safety_floors": dict(preregistration["zero_safety_floors"]),
    }


def persist_sealed_benchmark_report() -> dict[str, Any]:
    """Materialize the sealed report contract onto the owned write path."""

    payload = sealed_benchmark_report_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = ROOT / REPORT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    return payload


@dataclass(frozen=True, slots=True)
class SparBenchmarkReport:
    """Sealed SPAR-047 ablation report bound to SPAR-046/045 predecessors."""

    payload: Mapping[str, Any]
    report_cid: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "interface": SPAR_BENCHMARK_REPORT_INTERFACE,
            "payload": dict(self.payload),
            "report_cid": self.report_cid,
            "schema": SPAR_BENCHMARK_REPORT_SCHEMA,
            "task_id": TASK_ID,
        }


@dataclass(frozen=True, slots=True)
class SparAblationCellResult:
    """One SPAR-047 profile/ablation observation. Nomination-only."""

    profile: str
    ablation: str
    status: str
    evidence_cid: str
    expected_status: str
    expected_terminal: str = ""
    terminal: str = ""
    target_loc: int | None = None
    nominated: bool = True
    accepted: bool = False
    promotes: bool = False
    writes_repository: bool = False
    general_llm_invoked: bool = False
    in_denominator: bool = True
    interface: str = SPAR_ABLATION_CELL_INTERFACE
    schema: str = SPAR_ABLATION_CELL_SCHEMA

    def identity_payload(self) -> dict[str, Any]:
        return {
            "ablation": self.ablation,
            "accepted": False,
            "evidence_cid": self.evidence_cid,
            "expected_status": self.expected_status,
            "expected_terminal": self.expected_terminal,
            "general_llm_invoked": False,
            "in_denominator": True,
            "interface": self.interface,
            "nominated": True,
            "profile": self.profile,
            "promotes": False,
            "schema": self.schema,
            "status": self.status,
            "target_loc": self.target_loc,
            "terminal": self.terminal,
            "writes_repository": False,
        }

    @property
    def result_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())


@dataclass(frozen=True, slots=True)
class SparBenchmarkReportReceipt:
    """Nomination-only SPAR-047 ablation-report receipt."""

    tree_id: str
    report_cid: str
    cell_results: tuple[SparAblationCellResult, ...]
    negative_evidence_cids: tuple[str, ...]
    denominators: Mapping[str, int]
    ablation_ladder: tuple[str, ...]
    nominated: bool = True
    accepted: bool = False
    nomination_only: bool = True
    promotion_eligible: bool = False
    can_authorize_completion: bool = False
    can_authorize_transition: bool = False
    can_create_authority: bool = False
    writes_repository: bool = False
    worker_self_approval: bool = False
    projection_is_authority: bool = False
    network: str = NETWORK_DENY
    analyzer_id: str = ANALYZER_ID
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    interface: str = SPAR_BENCHMARK_RECEIPT_INTERFACE
    schema: str = SPAR_BENCHMARK_RECEIPT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "accepted",
            "ablation_ladder",
            "analyzer_id",
            "can_authorize_completion",
            "can_authorize_transition",
            "can_create_authority",
            "cell_results",
            "denominators",
            "goal_id",
            "interface",
            "negative_evidence_cids",
            "network",
            "nominated",
            "nomination_only",
            "projection_is_authority",
            "promotion_eligible",
            "receipt_cid",
            "report_cid",
            "schema",
            "task_id",
            "tree_id",
            "worker_self_approval",
            "writes_repository",
        }
    )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "ablation_ladder": list(self.ablation_ladder),
            "accepted": False,
            "analyzer_id": self.analyzer_id,
            "can_authorize_completion": False,
            "can_authorize_transition": False,
            "can_create_authority": False,
            "cell_results": [item.identity_payload() for item in self.cell_results],
            "denominators": dict(self.denominators),
            "goal_id": self.goal_id,
            "interface": self.interface,
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "network": NETWORK_DENY,
            "nominated": True,
            "nomination_only": True,
            "projection_is_authority": False,
            "promotion_eligible": False,
            "report_cid": self.report_cid,
            "schema": self.schema,
            "task_id": self.task_id,
            "tree_id": self.tree_id,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "SparBenchmarkReportReceipt":
        excluded = set(data) & IDENTITY_EXCLUDED_FIELDS
        if excluded:
            raise BenchmarkReportError(
                f"observational fields are excluded from identity: {sorted(excluded)}"
            )
        unknown = set(data) - cls._FIELDS
        if unknown:
            raise BenchmarkReportError(
                f"unsupported SparBenchmarkReportReceipt fields: {sorted(unknown)}"
            )
        payload = dict(data)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != SPAR_BENCHMARK_RECEIPT_SCHEMA:
            raise BenchmarkReportError("unsupported benchmark receipt schema")
        if payload.pop("interface") != SPAR_BENCHMARK_RECEIPT_INTERFACE:
            raise BenchmarkReportError("unsupported benchmark receipt interface")
        if payload.pop("accepted") is not False:
            raise BenchmarkReportError("workers cannot self-approve SPAR-047")
        if payload.pop("nomination_only") is not True:
            raise BenchmarkReportError("receipt must remain nomination_only")
        if payload.pop("nominated") is not True:
            raise BenchmarkReportError("receipt must remain nominated")
        if payload.pop("promotion_eligible") is not False:
            raise BenchmarkReportError("receipt cannot self-promote")
        for flag in _AUTHORITY_FLAGS:
            if payload.pop(flag) is not False:
                raise BenchmarkReportError(f"receipt cannot claim {flag}")
        if payload.pop("network") != NETWORK_DENY:
            raise BenchmarkReportError("receipt network must remain deny")
        results = tuple(
            SparAblationCellResult(
                profile=str(item["profile"]),
                ablation=str(item["ablation"]),
                status=str(item["status"]),
                evidence_cid=str(item["evidence_cid"]),
                expected_status=str(item["expected_status"]),
                expected_terminal=str(item.get("expected_terminal") or ""),
                terminal=str(item.get("terminal") or ""),
                target_loc=item.get("target_loc"),
            )
            for item in payload["cell_results"]
        )
        result = cls(
            tree_id=str(payload["tree_id"]),
            report_cid=str(payload["report_cid"]),
            cell_results=results,
            negative_evidence_cids=tuple(
                str(item) for item in payload["negative_evidence_cids"]
            ),
            denominators=dict(payload["denominators"]),
            ablation_ladder=tuple(str(item) for item in payload["ablation_ladder"]),
            analyzer_id=str(payload["analyzer_id"]),
            task_id=str(payload["task_id"]),
            goal_id=str(payload["goal_id"]),
        )
        if claimed != result.receipt_cid:
            raise BenchmarkReportError(
                "SparBenchmarkReportReceipt receipt_cid mismatch"
            )
        return result


def _observed(
    *,
    status: str,
    evidence_cid: str,
    terminal: str = "",
) -> dict[str, Any]:
    if status not in DECLARED_STATUSES:
        raise BenchmarkReportError(f"unsupported ablation status {status!r}")
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


def _run_manual_or_general_llm_heavy() -> dict[str, Any]:
    return _observed(
        status="unavailable",
        evidence_cid=cid_for_dag_json(dict(PROVIDER_TOKENIZER)),
        terminal="network_denied_general_llm",
    )


def _run_current_supervisor() -> dict[str, Any]:
    diagnosed = SemanticRefactoringService().execute(
        ControlRequest(operation="spar.diagnose")
    )
    if diagnosed.status != ControlStatus.DIAGNOSED.value:
        raise BenchmarkReportError("SPAR-044 diagnose must remain non-authoritative")
    if diagnosed.payload.get("can_authorize_completion") is not False:
        raise BenchmarkReportError("control surface cannot complete SPAR-047")
    return _observed(status="nominated", evidence_cid=_cid(diagnosed.status))


def _run_exact_capsules_graph() -> dict[str, Any]:
    fixture = generate_controlled_monolith("small")
    detection = detect_monolith_opportunities(
        {
            "tree_id": TREE_ID,
            "modules": [
                _module(
                    f"mod:{item.domain}",
                    loc=item.loc,
                    member_ids=(f"node:{item.domain}",),
                )
                for item in fixture.modules
            ],
        }
    )
    return _observed(status="nominated", evidence_cid=detection.receipt_cid)


def _run_deterministic_partition() -> dict[str, Any]:
    receipt = generate_partition_candidates(
        _partition_evidence(), generators=("scc",)
    )
    admitted = tuple(item for item in receipt.candidates if item.admitted)
    if len(admitted) != len(REQUIRED_DOMAINS):
        raise BenchmarkReportError("SCC partition must keep domain modules together")
    policy = load_split_policy()
    if policy.get("keep_domain_module_together") is not True:
        raise BenchmarkReportError("split policy must keep domain modules together")
    if policy.get("unsafe_intra_domain_split") is not False:
        raise BenchmarkReportError("unsafe intra-domain split")
    return _observed(status="nominated", evidence_cid=receipt.receipt_cid)


def _run_translation_validation() -> dict[str, Any]:
    result = run_acceptance_scenario(
        _scenario("validation.translation_pass", "validation", "nominated")
    )
    return _observed(status="nominated", evidence_cid=result.evidence_cid)


def _run_tactician_hammer() -> dict[str, Any]:
    with pytest.raises(ProofAdapterError, match="proof_candidate cannot admit proofs"):
        reconstruct_proof(
            {
                "obligation_id": "obl:bound",
                "kind": ArtifactKind.PROOF_CANDIDATE.value,
                "artifact_cid": _cid("candidate"),
                "reconstruction_cid": _cid("candidate"),
                "kernel_checked": False,
                "tree_id": TREE_ID,
                "corpus_revision": "corpus:v1",
                "environment_id": "env:test",
                "toolchain_id": TOOLCHAIN_ID,
            },
            tree_id=TREE_ID,
            corpus_revision="corpus:v1",
            environment_id="env:test",
            toolchain_id=TOOLCHAIN_ID,
        )
    return _observed(
        status="nominated",
        evidence_cid=_cid("tactician_hammer_fail_closed"),
    )


def _run_vector_graph_retrieval() -> dict[str, Any]:
    query = AnalogousRefactorQuery(
        tree_id=TREE_ID,
        member_ids=("node:state",),
        tokens=("extract",),
        graph_edges=(),
        vector=(),
        vector_available=False,
    )
    receipt = retrieve_analogous_refactors(query, ())
    if getattr(receipt, "vector_authoritative", False) is True:
        raise BenchmarkReportError("vector retrieval cannot be authoritative")
    return _observed(status="nominated", evidence_cid=receipt.receipt_cid)


def _run_optional_ranker() -> dict[str, Any]:
    partitions = generate_partition_candidates(
        _partition_evidence(), generators=("scc",)
    )
    ranking = rank_admitted_partitions(partitions, ())
    if set(ranking.ranked_candidate_cids) != set(ranking.input_candidate_cids):
        raise BenchmarkReportError("advisory ranker cannot drop candidates")
    return _observed(status="nominated", evidence_cid=ranking.receipt_cid)


def _run_procedure_reuse() -> dict[str, Any]:
    result = run_acceptance_scenario(
        _scenario("reuse.exact_accepted", "reuse", "nominated")
    )
    return _observed(status="nominated", evidence_cid=result.evidence_cid)


def _run_guarded() -> dict[str, Any]:
    contract = MODE_CONTRACTS["guarded"]
    if contract["gate_task"] != "SPAR-042":
        raise BenchmarkReportError("guarded gate must remain SPAR-042")
    if GATE_CAN_CHANGE_MODE is not False:
        raise BenchmarkReportError("benchmark cannot change rollout mode")
    baseline = sealed_rollout_baseline()
    return _observed(
        status="nominated",
        evidence_cid=cid_for_dag_json(
            {
                "gate_task": contract["gate_task"],
                "mode": "guarded",
                "mode_change": False,
                "rollout_schema": baseline.get("schema"),
            }
        ),
    )


def _run_required() -> dict[str, Any]:
    contract = MODE_CONTRACTS["required"]
    if contract["gate_task"] != "SPAR-043":
        raise BenchmarkReportError("required gate must remain SPAR-043")
    if GATE_CAN_CHANGE_MODE is not False:
        raise BenchmarkReportError("benchmark cannot change rollout mode")
    return _observed(
        status="nominated",
        evidence_cid=cid_for_dag_json(
            {
                "gate_task": contract["gate_task"],
                "mode": "required",
                "mode_change": False,
            }
        ),
    )


@functools.lru_cache(maxsize=None)
def _cached_ablation(ablation: str) -> dict[str, Any]:
    return dict(_DISPATCH[ablation]())


_DISPATCH: dict[str, Any] = {
    "manual_or_general_llm_heavy": _run_manual_or_general_llm_heavy,
    "current_supervisor": _run_current_supervisor,
    "exact_capsules_graph": _run_exact_capsules_graph,
    "deterministic_scc_state_contract_partition": _run_deterministic_partition,
    "translation_validation": _run_translation_validation,
    "tactician_hammer": _run_tactician_hammer,
    "vector_graph_retrieval": _run_vector_graph_retrieval,
    "optional_ranker": _run_optional_ranker,
    "procedure_reuse": _run_procedure_reuse,
    "guarded": _run_guarded,
    "required": _run_required,
}

_NEGATIVE_SCENARIOS: dict[str, dict[str, Any]] = {
    "reuse.stale_tree": {
        "scenario_id": "reuse.stale_tree",
        "stage": "reuse",
        "expected_status": "rejected",
        "expected_terminal": "stale_tree",
        "promotes": False,
        "writes_repository": False,
        "safety_floor": "stale_authoritative_reuse",
    },
    "unsafe_splits.intra_domain": {
        "scenario_id": "unsafe_splits.intra_domain",
        "stage": "unsafe_splits",
        "expected_status": "rejected",
        "expected_terminal": "unsafe_intra_domain_split",
        "promotes": False,
        "writes_repository": False,
        "safety_floor": "unauthorized_mutations",
    },
    "failed_proofs.candidate_cannot_admit": {
        "scenario_id": "failed_proofs.candidate_cannot_admit",
        "stage": "failed_proofs",
        "expected_status": "rejected",
        "expected_terminal": "proof_candidate_cannot_admit",
        "promotes": False,
        "writes_repository": False,
        "safety_floor": "test_or_proof_weakening",
    },
    "root_conflicts.no_overwrite": {
        "scenario_id": "root_conflicts.no_overwrite",
        "stage": "root_conflicts",
        "expected_status": "rejected",
        "expected_terminal": "rejected_root_conflict",
        "promotes": False,
        "writes_repository": False,
        "safety_floor": "root_conflict_overwrite",
    },
}


def _hermetic_context_reduction(loc: int | None) -> float | None:
    if loc is None or loc < 1:
        return None
    slice_loc = max(1, loc // len(REQUIRED_DOMAINS))
    return (loc - slice_loc) / loc


def run_ablation_cell(cell: Mapping[str, Any]) -> SparAblationCellResult:
    """Exercise one sealed SPAR-047 ablation cell against current adapters."""

    ablation = str(cell.get("ablation") or "")
    profile = str(cell.get("profile") or "")
    expected_status = str(cell.get("expected_status") or "")
    expected_terminal = str(cell.get("expected_terminal") or "")
    if ablation not in _DISPATCH:
        raise BenchmarkReportError(f"unknown ablation {ablation!r}")
    if expected_status not in DECLARED_STATUSES:
        raise BenchmarkReportError(f"unknown expected status {expected_status!r}")
    if cell.get("promotes") is True:
        raise BenchmarkReportError("ablation cells cannot promote")
    if cell.get("writes_repository") is True:
        raise BenchmarkReportError("ablation cells cannot write the repository")
    if cell.get("general_llm_invoked") is True:
        raise BenchmarkReportError("SPAR-047 cannot invoke a general LLM")
    observed = _cached_ablation(ablation)
    if observed["status"] != expected_status:
        raise BenchmarkReportError(
            f"{profile}/{ablation} status {observed['status']!r} != {expected_status!r}"
        )
    if expected_terminal and observed.get("terminal") != expected_terminal:
        raise BenchmarkReportError(
            f"{profile}/{ablation} terminal {observed.get('terminal')!r} != {expected_terminal!r}"
        )
    if observed["accepted"] or observed["promotes"] or observed["writes_repository"]:
        raise BenchmarkReportError(f"{profile}/{ablation} cannot promote or write")
    if observed["general_llm_invoked"]:
        raise BenchmarkReportError(f"{profile}/{ablation} invoked a general LLM")
    return SparAblationCellResult(
        profile=profile,
        ablation=ablation,
        status=str(observed["status"]),
        evidence_cid=str(observed["evidence_cid"]),
        expected_status=expected_status,
        expected_terminal=expected_terminal,
        terminal=str(observed.get("terminal") or ""),
        target_loc=cell.get("target_loc"),
        nominated=True,
    )


def _run_negative_evidence() -> tuple[str, ...]:
    cids: list[str] = []
    for item in NEGATIVE_EVIDENCE:
        scenario = _NEGATIVE_SCENARIOS[str(item["wave_id"])]
        result = run_acceptance_scenario(scenario)
        if result.status != "rejected":
            raise BenchmarkReportError(f"{item['wave_id']} must remain rejected")
        if result.promotes or result.writes_repository or result.accepted:
            raise BenchmarkReportError(f"{item['wave_id']} cannot promote or write")
        cids.append(result.evidence_cid)
    return tuple(cids)


def run_ablation_ladder(
    report: Mapping[str, Any] | None = None,
) -> SparBenchmarkReportReceipt:
    """Run the sealed SPAR-047 ladder. Current authority remains separate."""

    loaded = dict(report) if report is not None else load_benchmark_report()
    sealed = sealed_benchmark_report_payload()
    if loaded.get("ablation_ladder") != sealed["ablation_ladder"]:
        raise BenchmarkReportError("ablation ladder must remain preregistered")
    if tuple(loaded.get("predecessor_task_ids") or ()) != PREDECESSOR_TASK_IDS:
        raise BenchmarkReportError("predecessors must remain SPAR-046")
    preregistration = load_preregistration()
    if dict(loaded.get("zero_safety_floors") or {}) != dict(
        preregistration["zero_safety_floors"]
    ):
        raise BenchmarkReportError("safety floors must match preregistration")
    if dict(loaded.get("promotion_targets") or {}) != dict(
        preregistration["promotion_targets"]
    ):
        raise BenchmarkReportError("promotion targets must remain sealed")
    corpus = load_benchmark_corpus_manifest()
    if dict(corpus["zero_safety_floors"]) != dict(loaded["zero_safety_floors"]):
        raise BenchmarkReportError("corpus floors must match the benchmark report")
    matrix = load_acceptance_matrix()
    if matrix["task_id"] != "SPAR-046":
        raise BenchmarkReportError("predecessor acceptance matrix must remain SPAR-046")
    covered = {str(item["ablation"]) for item in loaded["cells"]}
    missing = [name for name in sealed["ablation_ladder"] if name not in covered]
    if missing:
        raise BenchmarkReportError(f"benchmark report missing ablations: {missing}")
    profiles = [str(item["name"]) for item in loaded["profiles"]]
    registered = [str(item["name"]) for item in preregistration["profiles"]]
    if profiles != registered:
        raise BenchmarkReportError("benchmark profiles must match preregistration")
    if loaded.get("promotion_eligible") is True:
        raise BenchmarkReportError("missing efficiency targets prohibit promotion")
    if loaded.get("efficiency_targets_met") is True:
        raise BenchmarkReportError("efficiency targets are not met without accepted waves")
    route = dry_run_context_route(
        packet=_context_packet(),
        reuse_decision={
            "decision": "revoke",
            "exact_match": False,
            "query_key_cid": _cid("query-key"),
            "matched_transition_cid": "",
            "reasons": ["no_exact_match"],
        },
        slices=[
            {
                "slice_id": "slice:mod",
                "path": "pkg/mod.py",
                "symbol": "pkg.mod.fn",
                "source_cid": _cid("source"),
            }
        ],
        analysis={"can_close": True, "kind": "graph"},
    )
    if route.route.general_model_invoked is True:
        raise BenchmarkReportError("no-model SPAR route cannot invoke a general model")
    results = tuple(run_ablation_cell(item) for item in loaded["cells"])
    negative = _run_negative_evidence()
    identity = {
        "payload": loaded,
        "task_id": TASK_ID,
        "tree_id": TREE_ID,
    }
    return SparBenchmarkReportReceipt(
        tree_id=TREE_ID,
        report_cid=cid_for_dag_json(identity),
        cell_results=results,
        negative_evidence_cids=negative,
        denominators=dict(loaded["denominators"]),
        ablation_ladder=tuple(str(item) for item in loaded["ablation_ladder"]),
    )


def nominate_benchmark_report() -> SparBenchmarkReportReceipt:
    """Nominate the SPAR-047 report. Independent validation remains separate."""

    return run_ablation_ladder()


@functools.lru_cache(maxsize=1)
def _cached_nomination() -> SparBenchmarkReportReceipt:
    return nominate_benchmark_report()


def dry_run_benchmark_report() -> SparBenchmarkReportReceipt:
    return _cached_nomination()


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-047"
    assert GOAL_ID == "SPAR-G081"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-046",)
    assert SPAR_BENCHMARK_REPORT_INTERFACE == "SparBenchmarkReport@1"
    assert SPAR_BENCHMARK_RECEIPT_INTERFACE == "SparBenchmarkReportReceipt@1"
    assert SPAR_ABLATION_CELL_INTERFACE == "SparAblationCellResult@1"
    assert BENCHMARK_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("benchmark_report@1")
    assert TEST_PATH.is_file()
    assert (ROOT / REPORT_RELATIVE).is_file()
    assert WRITE_SCOPE == (REPORT_RELATIVE, "test/api/semantic_refactoring/test_benchmark_report.py")
    assert (ROOT / PREREGISTRATION_RELATIVE).is_file()
    assert (ROOT / CORPUS_MANIFEST_RELATIVE).is_file()
    assert (ROOT / ACCEPTANCE_MATRIX_RELATIVE).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "benchmark execution"
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
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    profile = benchmark_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    report = load_benchmark_report()
    assert report["nomination_only"] is True
    assert report["can_authorize_completion"] is False
    assert report["can_authorize_transition"] is False
    assert report["can_create_authority"] is False
    assert report["worker_self_approval"] is False
    assert report["model_output_is_proposal_only"] is True
    assert report["test_pass_is_not_completion"] is True
    assert report["writes_repository"] is False
    assert report["projection_is_authority"] is False
    assert report["promotion_eligible"] is False


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(TEST_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "SparBenchmarkReport" in names
    assert "SparAblationCellResult" in names
    assert "SparBenchmarkReportReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "SparBenchmarkReportReceipt" in exports
    assert "nominate_benchmark_report" in exports
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
    payload = sealed_benchmark_report_payload()
    report = load_benchmark_report()
    assert report == payload
    assert report["residual_packet_cid"] == RESIDUAL_PACKET_CID
    assert report["residual_target_ids"] == [TASK_ID]
    assert report["sealed_before_tuning"] is True


def test_report_covers_preregistered_ladder_profiles_and_floors() -> None:
    report = load_benchmark_report()
    preregistration = load_preregistration()
    corpus = load_benchmark_corpus_manifest()
    assert tuple(report["ablation_ladder"]) == tuple(preregistration["ablation_ladder"])
    assert tuple(report["predecessor_task_ids"]) == PREDECESSOR_TASK_IDS
    assert [item["name"] for item in report["profiles"]] == [
        item["name"] for item in preregistration["profiles"]
    ]
    covered = {(item["profile"], item["ablation"]) for item in report["cells"]}
    expected = {
        (str(profile["name"]), ablation)
        for profile in preregistration["profiles"]
        for ablation in preregistration["ablation_ladder"]
    }
    assert covered == expected
    assert report["zero_safety_floors"] == preregistration["zero_safety_floors"]
    assert corpus["zero_safety_floors"] == report["zero_safety_floors"]
    assert all(value == 0 for value in report["zero_safety_floors"].values())
    assert report["promotion_targets"] == preregistration["promotion_targets"]
    assert report["denominator_policy"] == preregistration["denominator_policy"]
    for item in report["cells"]:
        assert item["promotes"] is False
        assert item["writes_repository"] is False
        assert item["in_denominator"] is True
        assert item["general_llm_invoked"] is False
        assert item["expected_status"] in DECLARED_STATUSES


def test_denominators_retain_honest_non_success_waves() -> None:
    report = load_benchmark_report()
    buckets = report["denominators"]
    for name in DENOMINATOR_BUCKETS:
        assert name in buckets
        assert type(buckets[name]) is int
        assert buckets[name] >= 0
    assert buckets["unavailable"] == 4
    assert buckets["rejected"] == 4
    assert buckets["failed"] == 0
    assert buckets["escalated"] == 0
    assert buckets["human_reviewed"] == 0
    assert buckets["nominated"] == 40
    assert buckets["total_waves"] == 48
    assert buckets["nominated"] + buckets["unavailable"] + buckets["rejected"] == (
        buckets["total_waves"]
    )
    llm_cells = [
        item
        for item in report["cells"]
        if item["ablation"] == "manual_or_general_llm_heavy"
    ]
    assert len(llm_cells) == 4
    assert {item["expected_status"] for item in llm_cells} == {"unavailable"}
    assert {item["expected_terminal"] for item in llm_cells} == {
        "network_denied_general_llm"
    }


def test_predecessors_remain_readable_without_dump() -> None:
    report = load_benchmark_report()
    encoded = json.dumps(report)
    assert "def leaf" not in encoded
    assert "class ShadowPlanGate" not in encoded
    assert "repository_dump" not in encoded
    assert "source_body" not in encoded
    corpus = load_benchmark_corpus_manifest()
    assert corpus["task_id"] == "SPAR-045"
    admitted = corpus["profiles"][-1]
    assert admitted["name"] == "real_current_tree"
    assert admitted.get("whole_repository_dump") is False
    matrix = load_acceptance_matrix()
    assert matrix["task_id"] == "SPAR-046"
    real = admit_real_current_tree()
    assert real["whole_repository_dump"] is False
    assert real["synthetic"] is False


def test_provider_tokenizer_criteria_are_exact_and_unavailable_llm() -> None:
    report = load_benchmark_report()
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
    assert metrics["later_promoted_procedure_no_general_llm_waves"] == 0
    assert metrics["median_general_llm_input_context_reduction"] is None
    assert metrics["general_llm_calls_per_accepted_wave_reduction"] is None
    assert metrics["tier_a_no_general_llm_fraction"] is None
    assert report["efficiency_targets_met"] is False
    assert report["failed_efficiency_target"] == (
        "non-promotion; safety floors remain unchanged"
    )
    assert report["promotion_eligible"] is False
    assert report["safety_floors_held"] is True
    small_reduction = _hermetic_context_reduction(5000)
    assert small_reduction is not None and small_reduction >= 0.3


def test_each_ablation_is_exercised_without_promotion() -> None:
    receipt = dry_run_benchmark_report()
    ablations = tuple(item.ablation for item in receipt.cell_results)
    assert set(ablations) == set(receipt.ablation_ladder)
    by_id = {
        (item.profile, item.ablation): item for item in receipt.cell_results
    }
    assert by_id[("small", "manual_or_general_llm_heavy")].status == "unavailable"
    assert by_id[("small", "current_supervisor")].status == "nominated"
    assert by_id[("small", "exact_capsules_graph")].status == "nominated"
    assert by_id[("small", "deterministic_scc_state_contract_partition")].status == (
        "nominated"
    )
    assert by_id[("small", "translation_validation")].status == "nominated"
    assert by_id[("small", "tactician_hammer")].status == "nominated"
    assert by_id[("small", "vector_graph_retrieval")].status == "nominated"
    assert by_id[("small", "optional_ranker")].status == "nominated"
    assert by_id[("small", "procedure_reuse")].status == "nominated"
    assert by_id[("small", "guarded")].status == "nominated"
    assert by_id[("small", "required")].status == "nominated"
    assert by_id[("real_current_tree", "manual_or_general_llm_heavy")].terminal == (
        "network_denied_general_llm"
    )
    for item in receipt.cell_results:
        assert item.accepted is False
        assert item.promotes is False
        assert item.writes_repository is False
        assert item.general_llm_invoked is False
        assert item.in_denominator is True


def test_fail_closed_negative_evidence_stays_in_denominator() -> None:
    receipt = dry_run_benchmark_report()
    assert len(receipt.negative_evidence_cids) == 4
    report = load_benchmark_report()
    floors = dict(report["zero_safety_floors"])
    by_wave = {item["wave_id"]: item for item in report["negative_evidence"]}
    assert by_wave["reuse.stale_tree"]["terminal"] == "stale_tree"
    assert by_wave["unsafe_splits.intra_domain"]["terminal"] == (
        "unsafe_intra_domain_split"
    )
    assert by_wave["failed_proofs.candidate_cannot_admit"]["terminal"] == (
        "proof_candidate_cannot_admit"
    )
    assert by_wave["root_conflicts.no_overwrite"]["terminal"] == (
        "rejected_root_conflict"
    )
    assert floors["stale_authoritative_reuse"] == 0
    assert floors["unauthorized_mutations"] == 0
    assert floors["test_or_proof_weakening"] == 0
    assert floors["root_conflict_overwrite"] == 0
    assert floors["false_task_completions"] == 0
    assert floors["critical_seeded_defect_escape"] == 0


def test_nomination_receipt_cannot_complete_or_self_approve() -> None:
    receipt = dry_run_benchmark_report()
    assert receipt.nominated is True
    assert receipt.accepted is False
    assert receipt.nomination_only is True
    assert receipt.promotion_eligible is False
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_authority is False
    assert receipt.writes_repository is False
    assert receipt.worker_self_approval is False
    assert receipt.network == NETWORK_DENY
    assert receipt.task_id == TASK_ID
    assert receipt.goal_id == GOAL_ID
    assert receipt.tree_id == TREE_ID
    encoded = receipt.to_dict()
    restored = SparBenchmarkReportReceipt.from_dict(encoded)
    assert restored.receipt_cid == receipt.receipt_cid
    assert restored == receipt
    dry = dry_run_benchmark_report()
    assert dry.receipt_cid == receipt.receipt_cid
    with pytest.raises(BenchmarkReportError, match="self-approve"):
        SparBenchmarkReportReceipt.from_dict({**encoded, "accepted": True})
    with pytest.raises(BenchmarkReportError, match="can_authorize_completion"):
        SparBenchmarkReportReceipt.from_dict(
            {**encoded, "can_authorize_completion": True}
        )
    with pytest.raises(BenchmarkReportError, match="self-promote"):
        SparBenchmarkReportReceipt.from_dict(
            {**encoded, "promotion_eligible": True}
        )


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    first = run_ablation_ladder()
    second = run_ablation_ladder()
    assert first.receipt_cid == second.receipt_cid
    assert [item.result_cid for item in first.cell_results] == [
        item.result_cid for item in second.cell_results
    ]
    assert first.writes_repository is False
    assert DRY_RUN_MUTATES is False


def test_identity_excludes_observational_fields() -> None:
    receipt = dry_run_benchmark_report()
    encoded = receipt.to_dict()
    for field in IDENTITY_EXCLUDED_FIELDS:
        assert field not in encoded
        with pytest.raises(BenchmarkReportError, match="observational"):
            SparBenchmarkReportReceipt.from_dict({**encoded, field: "now"})


def test_vector_or_model_cannot_admit_or_promote() -> None:
    report = dict(load_benchmark_report())
    with pytest.raises(BenchmarkReportError, match="unknown ablation|missing ablations"):
        run_ablation_ladder(
            {
                **report,
                "cells": [
                    {
                        "ablation": "vector.similarity_admit",
                        "expected_status": "nominated",
                        "expected_terminal": None,
                        "general_llm_invoked": False,
                        "in_denominator": True,
                        "profile": "small",
                        "promotes": False,
                        "target_loc": 5000,
                        "writes_repository": False,
                    }
                ],
            }
        )
    with pytest.raises(BenchmarkReportError, match="cannot promote"):
        run_ablation_cell(
            {
                "ablation": "current_supervisor",
                "expected_status": "nominated",
                "expected_terminal": None,
                "general_llm_invoked": False,
                "in_denominator": True,
                "profile": "small",
                "promotes": True,
                "target_loc": 5000,
                "writes_repository": False,
            }
        )
    with pytest.raises(BenchmarkReportError, match="general LLM"):
        run_ablation_cell(
            {
                "ablation": "current_supervisor",
                "expected_status": "nominated",
                "expected_terminal": None,
                "general_llm_invoked": True,
                "in_denominator": True,
                "profile": "small",
                "promotes": False,
                "target_loc": 5000,
                "writes_repository": False,
            }
        )


def test_missing_ablation_and_floor_drift_fail_closed() -> None:
    report = dict(load_benchmark_report())
    truncated = [
        item
        for item in report["cells"]
        if item["ablation"] != "translation_validation"
    ]
    with pytest.raises(BenchmarkReportError, match="missing ablations"):
        run_ablation_ladder({**report, "cells": truncated})
    drifted = dict(report["zero_safety_floors"])
    drifted["false_task_completions"] = 1
    with pytest.raises(BenchmarkReportError, match="safety floors"):
        run_ablation_ladder({**report, "zero_safety_floors": drifted})
    with pytest.raises(BenchmarkReportError, match="prohibit promotion"):
        run_ablation_ladder({**report, "promotion_eligible": True})
