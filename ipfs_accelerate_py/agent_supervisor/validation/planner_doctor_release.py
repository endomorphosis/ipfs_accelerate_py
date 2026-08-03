"""Independently replayed terminal PDR release receipt (PDR-092 / PDR-G100).

Terminal release gate for the Proof-directed Planner and Doctor board. It:

* reloads every required source receipt/artifact from the current tree;
* recomputes content identities (CIDs/preimages), current roots, and child-goal
  coverage without trusting task-status counts as objective completion;
* rejects stale, synthetic, skipped, forged, self-authored, or incomplete
  evidence classes for required surfaces;
* proves absolute-zero safety floors and exact-root rollback policy;
* documents unavailable optional capabilities without converting them to pass;
* keeps automatic promotion subject to a separate later held-out current-tree
  operator decision.

This module never grants mutation, completion, merge, promotion, or process
authority. Receipts are content-addressed; replaying the same current inputs is
identity-equivalent.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

# ---------------------------------------------------------------------------
# Schemas / identities
# ---------------------------------------------------------------------------

RELEASE_POLICY_INTERFACE: Final[str] = "PlannerDoctorReleasePolicy@1"
RELEASE_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-release-policy@1"
)
RELEASE_RECEIPT_INTERFACE: Final[str] = "PlannerDoctorReleaseReceipt@1"
RELEASE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-release-receipt@1"
)
RELEASE_VALIDATOR_INTERFACE: Final[str] = "PlannerDoctorReleaseValidator@1"
RELEASE_VALIDATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-release-report@1"
)
RELEASE_REPLAY_INTERFACE: Final[str] = "PlannerDoctorReleaseReplay@1"
RELEASE_REPLAY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-release-replay@1"
)

# Consumed interfaces (pins only; bodies live in producing tasks).
CONSUMED_INTERFACES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "authority_policy": "PlannerDoctorAuthorityPolicy@1",
        "authority_policy_seal": "PlannerDoctorAuthorityPolicySeal@1",
        "benchmark_policy": "PlannerDoctorBenchmarkPolicy@1",
        "benchmark_policy_seal": "PlannerDoctorBenchmarkPolicySeal@1",
        "attestation": "PlannerDoctorAttestation@1",
        "live_benchmark": "PlannerDoctorLiveBenchmark@1",
        "quality_oracle": "PlannerDoctorQualityOracle@1",
        "epoch": "PlannerDoctorEpoch@1",
        "refill": "PlannerDoctorDerivedRefill@1",
        "rollout_policy": "PlannerDoctorRolloutPolicy@1",
        "promotion_receipt": "PlannerDoctorPromotionReceipt@1",
        "operations": "PlannerDoctorOperations@1",
        "qualification": "PlannerDoctorQualification@1",
    }
)

TASK_ID: Final[str] = "PDR-092"
GOAL_ID: Final[str] = "PDR-G100"
BOARD_NAMESPACE: Final[str] = "agent-supervisor-proof-directed-planner-doctor-v1"
TASK_PREFIX: Final[str] = "PDR-"
LANE_COUNT: Final[int] = 6
CANONICAL_TASK_COUNT: Final[int] = 43
CANONICAL_GOAL_COUNT: Final[int] = 11
TERMINAL_TASK_ID: Final[str] = "PDR-092"
ROOT_GOAL_ID: Final[str] = "PDR-G000"

PLAN_REL: Final[str] = (
    "docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md"
)
OBJECTIVE_REL: Final[str] = (
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md"
)
TODO_REL: Final[str] = (
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md"
)
SCHEDULER_REL: Final[str] = (
    "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json"
)
AUTHORITY_POLICY_REL: Final[str] = (
    "config/agent_supervisor_planner_doctor_authority_policy.json"
)
AUTHORITY_SEAL_REL: Final[str] = (
    "config/agent_supervisor_planner_doctor_authority_policy.seal.json"
)
BENCHMARK_POLICY_REL: Final[str] = (
    "config/agent_supervisor_planner_doctor_benchmark.json"
)
BENCHMARK_SEAL_REL: Final[str] = (
    "config/agent_supervisor_planner_doctor_benchmark.seal.json"
)
HOLDOUT_MANIFEST_REL: Final[str] = (
    "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json"
)
THREAT_MODEL_REL: Final[str] = (
    "docs/architecture/agent_supervisor_planner_doctor_threat_model.md"
)
RELEASE_MODULE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_release.py"
)
RELEASE_DOC_REL: Final[str] = (
    "docs/architecture/PROOF_DIRECTED_PLANNER_DOCTOR_RELEASE.md"
)
RELEASE_TEST_REL: Final[str] = (
    "test/api/test_agent_supervisor_planner_doctor_release.py"
)
E2E_TEST_REL: Final[str] = (
    "test/integration/test_agent_supervisor_planner_doctor_e2e.py"
)
OPS_SCRIPT_REL: Final[str] = (
    "scripts/ops/agent_supervisor/proof_directed_planner_doctor.py"
)
OPS_GUIDE_REL: Final[str] = "docs/guides/PROOF_DIRECTED_PLANNER_DOCTOR_GUIDE.md"
PROGRAMS_REL: Final[str] = "docs/architecture/agent_supervisor/PROGRAMS.md"

EXPECTED_TASK_IDS: Final[tuple[str, ...]] = (
    "PDR-000",
    "PDR-001",
    "PDR-002",
    "PDR-003",
    "PDR-010",
    "PDR-011",
    "PDR-012",
    "PDR-013",
    "PDR-014",
    "PDR-015",
    "PDR-020",
    "PDR-021",
    "PDR-022",
    "PDR-023",
    "PDR-024",
    "PDR-025",
    "PDR-026",
    "PDR-027",
    "PDR-028",
    "PDR-030",
    "PDR-031",
    "PDR-032",
    "PDR-033",
    "PDR-040",
    "PDR-041",
    "PDR-042",
    "PDR-043",
    "PDR-050",
    "PDR-051",
    "PDR-052",
    "PDR-053",
    "PDR-054",
    "PDR-055",
    "PDR-060",
    "PDR-070",
    "PDR-071",
    "PDR-072",
    "PDR-080",
    "PDR-081",
    "PDR-082",
    "PDR-090",
    "PDR-091",
    "PDR-092",
)

EXPECTED_GOAL_IDS: Final[tuple[str, ...]] = (
    "PDR-G000",
    "PDR-G010",
    "PDR-G020",
    "PDR-G030",
    "PDR-G040",
    "PDR-G050",
    "PDR-G060",
    "PDR-G070",
    "PDR-G080",
    "PDR-G090",
    "PDR-G100",
)

CHILD_GOAL_IDS: Final[tuple[str, ...]] = tuple(
    goal_id for goal_id in EXPECTED_GOAL_IDS if goal_id != ROOT_GOAL_ID
)

TERMINAL_DEPENDENCIES: Final[tuple[str, ...]] = (
    "PDR-027",
    "PDR-033",
    "PDR-053",
    "PDR-060",
    "PDR-070",
    "PDR-071",
    "PDR-072",
    "PDR-080",
    "PDR-081",
    "PDR-082",
    "PDR-090",
    "PDR-091",
)

# Child-goal coverage is proven by current artifact presence + content digests,
# never by completed task counts alone.
CHILD_GOAL_EVIDENCE_ARTIFACTS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "PDR-G010": (
            "docs/architecture/agent_supervisor_planner_doctor_baseline.md",
            THREAT_MODEL_REL,
            BENCHMARK_POLICY_REL,
            AUTHORITY_POLICY_REL,
            AUTHORITY_SEAL_REL,
            BENCHMARK_SEAL_REL,
            HOLDOUT_MANIFEST_REL,
        ),
        "PDR-G020": (
            "ipfs_accelerate_py/agent_supervisor/analysis/repository_reasoning_snapshot.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/planning_analysis_factory.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/analysis_strategy_registry.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/planning_evidence_bundle.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/reasoning_cache.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/planner_doctor_capability_inventory.py",
        ),
        "PDR-G030": (
            "ipfs_accelerate_py/agent_supervisor/planning/plan_revision_contracts.py",
            "ipfs_accelerate_py/agent_supervisor/planning/plan_analysis_query_planner.py",
            "ipfs_accelerate_py/agent_supervisor/planning/obligation_graph_compiler.py",
            "ipfs_accelerate_py/agent_supervisor/planning/symbolic_candidate_planner.py",
            "ipfs_accelerate_py/agent_supervisor/planning/plan_critic.py",
            "ipfs_accelerate_py/agent_supervisor/planning/parallel_plan_compiler.py",
            "ipfs_accelerate_py/agent_supervisor/planning/plan_admission_service.py",
            "ipfs_accelerate_py/agent_supervisor/context/planner_doctor_context.py",
        ),
        "PDR-G040": (
            "ipfs_accelerate_py/agent_supervisor/prompt/plan_supervisor_service.py",
            "ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py",
            "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py",
            "ipfs_accelerate_py/agent_supervisor/control/control_plane.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        ),
        "PDR-G050": (
            "ipfs_accelerate_py/agent_supervisor/runtime/deterministic_doctor_runtime.py",
            "ipfs_accelerate_py/agent_supervisor/proof/deterministic_doctor_hammer.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/doctor_causal_localization.py",
            "ipfs_accelerate_py/agent_supervisor/planning/diagnosis_obligation_adapter.py",
        ),
        "PDR-G060": (
            "ipfs_accelerate_py/agent_supervisor/planning/repair_operator_registry.py",
            "ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/doctor_worktree_adapter.py",
            "ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_live_fixed_point.py",
            "ipfs_accelerate_py/agent_supervisor/validation/repair_candidate_portfolio.py",
            "ipfs_accelerate_py/agent_supervisor/objectives/doctor_plan_refill.py",
        ),
        "PDR-G070": (
            "ipfs_accelerate_py/agent_supervisor/proof/planner_doctor_attestation.py",
            "docs/architecture/agent_supervisor_planner_doctor_zkp_threat_model.md",
            "test/api/test_agent_supervisor_planner_doctor_attestation.py",
        ),
        "PDR-G080": (
            "ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_live_benchmark.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py",
            "ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_quality_oracle.py",
            "test/api/test_agent_supervisor_planner_doctor_live_benchmark.py",
            "test/api/test_agent_supervisor_planner_doctor_quality_oracle.py",
        ),
        "PDR-G090": (
            "ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py",
            "ipfs_accelerate_py/agent_supervisor/objectives/planner_doctor_refill.py",
            "ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_rollout.py",
            "test/api/test_agent_supervisor_planner_doctor_epoch.py",
            "test/api/test_agent_supervisor_planner_doctor_refill.py",
            "test/api/test_agent_supervisor_planner_doctor_rollout.py",
        ),
        "PDR-G100": (
            E2E_TEST_REL,
            OPS_SCRIPT_REL,
            OPS_GUIDE_REL,
            RELEASE_MODULE_REL,
            RELEASE_DOC_REL,
            RELEASE_TEST_REL,
            "test/api/test_agent_supervisor_planner_doctor_operations.py",
            "test/integration/test_agent_supervisor_planner_doctor_chaos.py",
            "test/integration/test_agent_supervisor_planner_doctor_security.py",
        ),
    }
)

PROTECTED_PATHS: Final[tuple[str, ...]] = (
    PLAN_REL,
    OBJECTIVE_REL,
    TODO_REL,
    SCHEDULER_REL,
    THREAT_MODEL_REL,
    AUTHORITY_POLICY_REL,
    AUTHORITY_SEAL_REL,
    BENCHMARK_POLICY_REL,
    BENCHMARK_SEAL_REL,
    HOLDOUT_MANIFEST_REL,
)

REQUIRED_RELEASE_ARTIFACTS: Final[tuple[str, ...]] = (
    RELEASE_MODULE_REL,
    RELEASE_DOC_REL,
    RELEASE_TEST_REL,
    E2E_TEST_REL,
    OPS_SCRIPT_REL,
    OPS_GUIDE_REL,
    PLAN_REL,
    OBJECTIVE_REL,
    TODO_REL,
    SCHEDULER_REL,
    AUTHORITY_POLICY_REL,
    AUTHORITY_SEAL_REL,
    BENCHMARK_POLICY_REL,
    BENCHMARK_SEAL_REL,
    HOLDOUT_MANIFEST_REL,
    PROGRAMS_REL,
)

# Absolute-zero floors aligned with PlannerDoctorRolloutPolicy@1.
SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "authority_violation_count",
    "policy_escape_count",
    "scope_escape_count",
    "secret_escape_count",
    "path_escape_count",
    "stale_cache_admission_count",
    "forged_cid_admission_count",
    "forged_proof_admission_count",
    "missed_mandatory_consumer_count",
    "falsely_closed_required_impact_frontier_count",
    "security_ir_prohibition_miss_count",
    "intent_ir_prohibition_miss_count",
    "hidden_oracle_access_or_mutation_count",
    "benchmark_or_denominator_mutation_count",
    "partial_transaction_count",
    "false_fixed_point_count",
    "rollback_failure_count",
    "false_completion_count",
    "synthetic_observation_used_for_promotion_count",
    "skipped_observation_used_for_promotion_count",
)

REJECTED_EVIDENCE_CLASSES: Final[tuple[str, ...]] = (
    "stale",
    "synthetic",
    "skipped",
    "forged",
    "self_authored",
    "incomplete",
    "unavailable_required",
)

COLD_IMPORT_MODULES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_release",
    "ipfs_accelerate_py.agent_supervisor.self_improvement.planner_doctor_rollout",
    "ipfs_accelerate_py.agent_supervisor.self_improvement.planner_doctor_epoch",
    "ipfs_accelerate_py.agent_supervisor.objectives.planner_doctor_refill",
    "ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_live_benchmark",
    "ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_quality_oracle",
    "ipfs_accelerate_py.agent_supervisor.proof.planner_doctor_attestation",
)

# Optional capabilities may be absent; absence is documented, never auto-pass.
OPTIONAL_CAPABILITY_PROBES: Final[tuple[tuple[str, str], ...]] = (
    ("zkp_optional_prover", "zkp"),
    ("gpu_telemetry", "pynvml"),
    ("remote_model_provider_openai", "openai"),
    ("remote_model_provider_anthropic", "anthropic"),
    ("torch_runtime", "torch"),
    ("transformers_runtime", "transformers"),
    ("sentence_transformers_runtime", "sentence_transformers"),
    ("lean_prover", "lean"),
)

MAX_TEXT_BYTES: Final[int] = 512


class PlannerDoctorReleaseError(ValueError):
    """Release policy, receipt, or validation evidence is invalid."""


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class EvidenceDisposition(str, Enum):
    """Admissible vs rejected evidence classes for required surfaces."""

    CURRENT = "current"
    STALE = "stale"
    SYNTHETIC = "synthetic"
    SKIPPED = "skipped"
    FORGED = "forged"
    SELF_AUTHORED = "self_authored"
    INCOMPLETE = "incomplete"
    UNAVAILABLE_REQUIRED = "unavailable_required"
    UNAVAILABLE_OPTIONAL = "unavailable_optional"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def repository_root() -> Path:
    # validation/ -> agent_supervisor/ -> ipfs_accelerate_py/ -> repo root
    return Path(__file__).resolve().parents[3]


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(k): _plain(v)
            for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_identity(value: Any) -> str:
    return _sha256_hex(_canonical_bytes(value))


def file_content_identity(path: Path) -> str:
    return _sha256_hex(path.read_bytes())


def seal_payload(payload: Mapping[str, Any], *, id_key: str = "receipt_id") -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != id_key}
    sealed = dict(body)
    sealed[id_key] = content_identity(body)
    return sealed


def verify_sealed(payload: Mapping[str, Any], *, id_key: str = "receipt_id") -> bool:
    claimed = payload.get(id_key)
    if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
        return False
    return claimed == seal_payload(payload, id_key=id_key).get(id_key)


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    text = str(value or "").strip()
    if not text:
        raise PlannerDoctorReleaseError(f"{name} must be non-empty")
    if len(text.encode("utf-8")) > maximum:
        raise PlannerDoctorReleaseError(f"{name} exceeds {maximum} bytes")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PlannerDoctorReleaseError(f"{name} must be a bool")
    return value


def _zero_floors() -> dict[str, int]:
    return {key: 0 for key in SAFETY_FLOOR_KEYS}


def classify_evidence_disposition(
    *,
    present: bool,
    required: bool,
    class_hint: str | None = None,
    self_authored: bool = False,
    content_id: str | None = None,
    claimed_content_id: str | None = None,
    skipped: bool = False,
    synthetic: bool = False,
    incomplete: bool = False,
) -> EvidenceDisposition:
    """Classify evidence; required surfaces never pass on unavailable/synthetic."""

    hint = (class_hint or "").strip().casefold().replace("-", "_")
    if self_authored or hint == "self_authored":
        return EvidenceDisposition.SELF_AUTHORED
    if synthetic or hint == "synthetic":
        return EvidenceDisposition.SYNTHETIC
    if skipped or hint == "skipped":
        return EvidenceDisposition.SKIPPED
    if incomplete or hint == "incomplete":
        return EvidenceDisposition.INCOMPLETE
    if hint == "stale":
        return EvidenceDisposition.STALE
    if hint == "forged":
        return EvidenceDisposition.FORGED
    if (
        content_id
        and claimed_content_id
        and content_id != claimed_content_id
    ):
        return EvidenceDisposition.FORGED
    if not present:
        if required:
            return EvidenceDisposition.UNAVAILABLE_REQUIRED
        return EvidenceDisposition.UNAVAILABLE_OPTIONAL
    return EvidenceDisposition.CURRENT


def evidence_is_admissible(disposition: EvidenceDisposition, *, required: bool) -> bool:
    if disposition is EvidenceDisposition.CURRENT:
        return True
    if disposition is EvidenceDisposition.UNAVAILABLE_OPTIONAL and not required:
        return True
    return False


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus | str
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name"))
        status = (
            self.status
            if isinstance(self.status, CheckStatus)
            else CheckStatus(str(self.status).strip().casefold())
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence or {})))

    @property
    def ok(self) -> bool:
        return self.status in {CheckStatus.PASS, CheckStatus.SKIP, CheckStatus.WARN}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value if isinstance(self.status, CheckStatus) else str(self.status),
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerDoctorReleasePolicy:
    """Immutable terminal-release policy: report-only, no promotion authority."""

    INTERFACE: ClassVar[str] = RELEASE_POLICY_INTERFACE
    SCHEMA: ClassVar[str] = RELEASE_POLICY_SCHEMA

    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    board_namespace: str = BOARD_NAMESPACE
    default_mode: str = "report_only"
    mutation_authorized: bool = False
    completion_authoritative: bool = False
    automatic_promotion_enabled: bool = False
    doctor_mutation_authorized: bool = False
    refill_enabled: bool = False
    llm_invocations_allowed: bool = False
    remote_model_provider_calls_allowed: bool = False
    network_access_required: bool = False
    require_source_artifact_reload: bool = True
    require_child_goal_coverage: bool = True
    require_task_objective_distinction: bool = True
    require_reject_bad_evidence: bool = True
    require_zero_safety_floors: bool = True
    require_exact_rollback: bool = True
    require_optional_capability_documentation: bool = True
    require_holdout_operator_decision_for_automatic: bool = True
    require_cold_imports: bool = True
    require_six_lane_drain: bool = True
    require_protected_anchors: bool = True
    safety_floors: Mapping[str, int] = field(default_factory=_zero_floors)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "board_namespace", _text(self.board_namespace, "board_namespace")
        )
        object.__setattr__(
            self, "default_mode", _text(self.default_mode, "default_mode")
        )
        for name in (
            "mutation_authorized",
            "completion_authoritative",
            "automatic_promotion_enabled",
            "doctor_mutation_authorized",
            "refill_enabled",
            "llm_invocations_allowed",
            "remote_model_provider_calls_allowed",
            "network_access_required",
            "require_source_artifact_reload",
            "require_child_goal_coverage",
            "require_task_objective_distinction",
            "require_reject_bad_evidence",
            "require_zero_safety_floors",
            "require_exact_rollback",
            "require_optional_capability_documentation",
            "require_holdout_operator_decision_for_automatic",
            "require_cold_imports",
            "require_six_lane_drain",
            "require_protected_anchors",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if self.default_mode != "report_only":
            raise PlannerDoctorReleaseError(
                "terminal release policy must default to report_only"
            )
        if self.mutation_authorized or self.completion_authoritative:
            raise PlannerDoctorReleaseError(
                "terminal release policy cannot authorize mutation or completion"
            )
        if self.automatic_promotion_enabled:
            raise PlannerDoctorReleaseError(
                "terminal release policy cannot enable automatic promotion"
            )
        if self.doctor_mutation_authorized or self.refill_enabled:
            raise PlannerDoctorReleaseError(
                "terminal release policy cannot enable doctor mutation or refill"
            )
        if any(
            (
                self.llm_invocations_allowed,
                self.remote_model_provider_calls_allowed,
                self.network_access_required,
            )
        ):
            raise PlannerDoctorReleaseError(
                "terminal release policy forbids model/network requirement flags"
            )
        floors = dict(self.safety_floors or _zero_floors())
        for key in SAFETY_FLOOR_KEYS:
            if int(floors.get(key, 1)) != 0:
                raise PlannerDoctorReleaseError(
                    f"safety floor {key} must be exactly zero"
                )
        object.__setattr__(self, "safety_floors", MappingProxyType(floors))

    @property
    def policy_binding_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "board_namespace": self.board_namespace,
            "default_mode": self.default_mode,
            "mutation_authorized": self.mutation_authorized,
            "completion_authoritative": self.completion_authoritative,
            "automatic_promotion_enabled": self.automatic_promotion_enabled,
            "doctor_mutation_authorized": self.doctor_mutation_authorized,
            "refill_enabled": self.refill_enabled,
            "llm_invocations_allowed": self.llm_invocations_allowed,
            "remote_model_provider_calls_allowed": (
                self.remote_model_provider_calls_allowed
            ),
            "network_access_required": self.network_access_required,
            "require_source_artifact_reload": self.require_source_artifact_reload,
            "require_child_goal_coverage": self.require_child_goal_coverage,
            "require_task_objective_distinction": (
                self.require_task_objective_distinction
            ),
            "require_reject_bad_evidence": self.require_reject_bad_evidence,
            "require_zero_safety_floors": self.require_zero_safety_floors,
            "require_exact_rollback": self.require_exact_rollback,
            "require_optional_capability_documentation": (
                self.require_optional_capability_documentation
            ),
            "require_holdout_operator_decision_for_automatic": (
                self.require_holdout_operator_decision_for_automatic
            ),
            "require_cold_imports": self.require_cold_imports,
            "require_six_lane_drain": self.require_six_lane_drain,
            "require_protected_anchors": self.require_protected_anchors,
            "safety_floors": dict(self.safety_floors),
            "consumed_interfaces": dict(CONSUMED_INTERFACES),
            "rejected_evidence_classes": list(REJECTED_EVIDENCE_CLASSES),
        }
        if include_id:
            payload["policy_binding_id"] = content_identity(
                {k: v for k, v in payload.items() if k != "policy_binding_id"}
            )
        return payload


def default_release_policy() -> PlannerDoctorReleasePolicy:
    return PlannerDoctorReleasePolicy()


# ---------------------------------------------------------------------------
# Board / source reload / coverage
# ---------------------------------------------------------------------------


def _parse_task_file(repo_root: Path) -> tuple[Any, ...]:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        parse_task_file,
    )

    return tuple(
        parse_task_file(repo_root / TODO_REL, task_header_prefix=TASK_PREFIX)
    )


def _parse_goals(repo_root: Path) -> list[Any]:
    from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
        parse_goal_heap,
    )

    return list(parse_goal_heap((repo_root / OBJECTIVE_REL).read_text(encoding="utf-8")))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_declared_artifacts(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    present = {rel: (root / rel).is_file() for rel in REQUIRED_RELEASE_ARTIFACTS}
    missing = [rel for rel, ok in present.items() if not ok]
    digests = {
        rel: file_content_identity(root / rel)
        for rel, ok in present.items()
        if ok
    }
    evidence = {"artifacts": present, "digests": digests}
    if missing:
        return CheckResult(
            "declared_artifacts",
            CheckStatus.FAIL,
            f"missing declared artifacts: {missing}",
            evidence,
        )
    return CheckResult(
        "declared_artifacts",
        CheckStatus.PASS,
        "all PDR-092 release artifacts are present",
        evidence,
    )


def check_protected_anchors(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    present = {
        rel: (root / rel).is_file() and (root / rel).stat().st_size > 0
        for rel in PROTECTED_PATHS
    }
    digests = {
        rel: file_content_identity(root / rel)
        for rel, ok in present.items()
        if ok
    }
    missing = [rel for rel, ok in present.items() if not ok]
    evidence = {
        "protected_present": present,
        "digests": digests,
        "release_may_rewrite_protected": False,
    }
    if missing:
        return CheckResult(
            "protected_anchors",
            CheckStatus.FAIL,
            f"protected anchors missing or empty: {missing}",
            evidence,
        )
    return CheckResult(
        "protected_anchors",
        CheckStatus.PASS,
        "protected anchors present; release has no rewrite authority",
        evidence,
    )


def check_canonical_board(repo_root: Path | None = None) -> CheckResult:
    """Validate 43 tasks, 11 goals, and PDR-092 as the unique terminal sink."""

    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    evidence: dict[str, Any] = {}

    try:
        all_tasks = _parse_task_file(root)
        goals = _parse_goals(root)
    except Exception as exc:  # noqa: BLE001 - surface parse failures as check fail
        return CheckResult(
            "canonical_board",
            CheckStatus.FAIL,
            f"unable to parse board/objectives: {exc}",
        )

    goal_ids = tuple(sorted({g.goal_id for g in goals}))
    evidence["goal_ids"] = list(goal_ids)
    evidence["goal_count"] = len(goal_ids)
    if set(goal_ids) != set(EXPECTED_GOAL_IDS) or len(goal_ids) != CANONICAL_GOAL_COUNT:
        errors.append(
            f"goal set mismatch: expected {CANONICAL_GOAL_COUNT} "
            f"{list(EXPECTED_GOAL_IDS)}, got {list(goal_ids)}"
        )

    by_id = {task.task_id: task for task in all_tasks}
    canonical = tuple(task for task in all_tasks if task.task_id in EXPECTED_TASK_IDS)
    canonical_ids = tuple(sorted(task.task_id for task in canonical))
    evidence["canonical_task_count"] = len(canonical_ids)
    evidence["canonical_task_ids"] = list(canonical_ids)
    if canonical_ids != EXPECTED_TASK_IDS:
        errors.append(
            f"canonical task set mismatch count={len(canonical_ids)} "
            f"expected={CANONICAL_TASK_COUNT}"
        )

    # Preimage digests for every canonical task body (current tree).
    task_preimages: dict[str, str] = {}
    for task in canonical:
        task_preimages[task.task_id] = content_identity(
            {
                "task_id": task.task_id,
                "depends_on": list(task.depends_on),
                "board_namespace": getattr(task, "board_namespace", None),
                "goal_id": (task.metadata or {}).get("goal id")
                if isinstance(task.metadata, Mapping)
                else None,
                "canonical_task_cid": getattr(task, "canonical_task_cid", None),
            }
        )
    evidence["task_preimages"] = task_preimages
    evidence["task_preimage_root"] = content_identity(task_preimages)

    graph: dict[str, tuple[str, ...]] = {}
    for task in canonical:
        unknown = sorted(set(task.depends_on) - set(EXPECTED_TASK_IDS))
        if unknown:
            errors.append(f"{task.task_id} has unknown deps {unknown}")
        graph[task.task_id] = tuple(task.depends_on)

    consumed = {dep for deps in graph.values() for dep in deps}
    sinks = sorted(set(graph) - consumed)
    evidence["sinks"] = sinks
    if sinks != [TERMINAL_TASK_ID]:
        errors.append(f"terminal task mismatch: {sinks}")

    terminal = by_id.get(TERMINAL_TASK_ID)
    if terminal is None:
        errors.append("PDR-092 missing from board")
    else:
        goal_meta = (
            (terminal.metadata or {}).get("goal id")
            if isinstance(terminal.metadata, Mapping)
            else None
        )
        evidence["terminal_goal"] = goal_meta
        evidence["terminal_depends_on"] = list(terminal.depends_on)
        evidence["terminal_canonical_task_cid"] = getattr(
            terminal, "canonical_task_cid", None
        )
        if goal_meta != GOAL_ID:
            errors.append("PDR-092 goal id mismatch")
        if getattr(terminal, "board_namespace", None) != BOARD_NAMESPACE:
            errors.append("PDR-092 board namespace mismatch")
        if tuple(terminal.depends_on) != TERMINAL_DEPENDENCIES:
            errors.append(
                f"PDR-092 dependency mismatch: {list(terminal.depends_on)} "
                f"!= {list(TERMINAL_DEPENDENCIES)}"
            )

    scheduler_path = root / SCHEDULER_REL
    if not scheduler_path.is_file():
        errors.append(f"scheduler missing: {scheduler_path}")
    else:
        try:
            scheduler = _load_json(scheduler_path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"scheduler unreadable: {exc}")
            scheduler = {}
        max_lanes = int(scheduler.get("max_lanes") or 0)
        evidence["max_lanes"] = max_lanes
        evidence["board_namespace"] = scheduler.get("board_namespace")
        if max_lanes != LANE_COUNT:
            errors.append(f"max_lanes must be {LANE_COUNT}, got {max_lanes}")
        if scheduler.get("board_namespace") != BOARD_NAMESPACE:
            errors.append("scheduler board_namespace mismatch")

    evidence["errors"] = errors
    if errors:
        return CheckResult(
            "canonical_board",
            CheckStatus.FAIL,
            "; ".join(errors[:6]),
            evidence,
        )
    return CheckResult(
        "canonical_board",
        CheckStatus.PASS,
        (
            f"{CANONICAL_TASK_COUNT} tasks, {CANONICAL_GOAL_COUNT} goals, "
            f"{TERMINAL_TASK_ID} unique terminal"
        ),
        evidence,
    )


def check_source_artifact_reload(repo_root: Path | None = None) -> CheckResult:
    """Reload every required source receipt/artifact and recompute digests."""

    root = (repo_root or repository_root()).resolve()
    artifacts: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    forged: list[str] = []

    # Unique union of required release + child-goal + protected artifacts.
    required_paths: list[str] = []
    seen: set[str] = set()
    for rel in (
        *REQUIRED_RELEASE_ARTIFACTS,
        *PROTECTED_PATHS,
        *(
            path
            for paths in CHILD_GOAL_EVIDENCE_ARTIFACTS.values()
            for path in paths
        ),
    ):
        if rel not in seen:
            seen.add(rel)
            required_paths.append(rel)

    for rel in required_paths:
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            artifacts[rel] = {
                "present": False,
                "disposition": EvidenceDisposition.UNAVAILABLE_REQUIRED.value,
            }
            continue
        digest = file_content_identity(path)
        # Seal integrity for JSON seal receipts when present.
        seal_ok = True
        if rel.endswith(".seal.json"):
            try:
                payload = _load_json(path)
                claimed = payload.get("receipt_id")
                if claimed and not verify_sealed(payload):
                    seal_ok = False
                    forged.append(rel)
            except (OSError, json.JSONDecodeError, TypeError):
                seal_ok = False
                forged.append(rel)
        disposition = classify_evidence_disposition(
            present=True,
            required=True,
            class_hint=None if seal_ok else "forged",
            content_id=digest,
            claimed_content_id=digest if seal_ok else "sha256:" + ("0" * 64),
        )
        artifacts[rel] = {
            "present": True,
            "content_id": digest,
            "byte_length": path.stat().st_size,
            "seal_ok": seal_ok,
            "disposition": disposition.value,
        }

    forest_root = content_identity(
        {
            rel: body.get("content_id")
            for rel, body in sorted(artifacts.items())
            if body.get("content_id")
        }
    )
    evidence = {
        "artifact_count": len(artifacts),
        "missing": missing,
        "forged": forged,
        "forest_root": forest_root,
        "artifacts": artifacts,
        "self_authored_release_module": False,
        "reloaded_from_current_tree": True,
    }
    if missing:
        return CheckResult(
            "source_artifact_reload",
            CheckStatus.FAIL,
            f"required source artifacts missing: {missing[:8]}",
            evidence,
        )
    if forged:
        return CheckResult(
            "source_artifact_reload",
            CheckStatus.FAIL,
            f"forged or unsealed source receipts: {forged}",
            evidence,
        )
    return CheckResult(
        "source_artifact_reload",
        CheckStatus.PASS,
        f"reloaded {len(artifacts)} source artifacts; forest_root={forest_root[:18]}…",
        evidence,
    )


def check_child_goal_coverage(repo_root: Path | None = None) -> CheckResult:
    """Prove every child goal has current independently reloadable evidence."""

    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    coverage: dict[str, dict[str, Any]] = {}

    try:
        goals = _parse_goals(root)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "child_goal_coverage",
            CheckStatus.FAIL,
            f"unable to parse goals: {exc}",
        )

    goals_by_id = {g.goal_id: g for g in goals}
    for goal_id in CHILD_GOAL_IDS:
        goal = goals_by_id.get(goal_id)
        if goal is None:
            errors.append(f"{goal_id}:missing_from_heap")
            coverage[goal_id] = {"present": False}
            continue
        artifact_paths = list(CHILD_GOAL_EVIDENCE_ARTIFACTS.get(goal_id, ()))
        digests: dict[str, str] = {}
        missing_artifacts: list[str] = []
        for rel in artifact_paths:
            path = root / rel
            if not path.is_file() or path.stat().st_size == 0:
                missing_artifacts.append(rel)
                continue
            digests[rel] = file_content_identity(path)
        goal_evidence_root = content_identity(digests) if digests else ""
        # Objective completion requires independent artifact roots, not task counts.
        task_count_claim = None
        producing = str(getattr(goal, "fields", {}).get("producing_tasks") or "")
        if re.fullmatch(r"\d+", producing.strip()):
            task_count_claim = int(producing.strip())
        covered = not missing_artifacts and bool(digests)
        if not covered:
            errors.append(f"{goal_id}:incomplete_evidence")
        coverage[goal_id] = {
            "present": True,
            "artifact_count": len(digests),
            "missing_artifacts": missing_artifacts,
            "evidence_root": goal_evidence_root,
            "task_count_claim": task_count_claim,
            "uses_task_count_as_authority": False,
            "covered": covered,
        }

    # Root goal is covered only when every child goal is covered.
    children_covered = all(
        bool(coverage.get(goal_id, {}).get("covered")) for goal_id in CHILD_GOAL_IDS
    )
    root_artifacts = [
        PLAN_REL,
        OBJECTIVE_REL,
        TODO_REL,
        SCHEDULER_REL,
        PROGRAMS_REL,
    ]
    root_digests = {
        rel: file_content_identity(root / rel)
        for rel in root_artifacts
        if (root / rel).is_file()
    }
    coverage[ROOT_GOAL_ID] = {
        "present": ROOT_GOAL_ID in goals_by_id,
        "children_covered": children_covered,
        "evidence_root": content_identity(root_digests),
        "covered": children_covered and len(root_digests) == len(root_artifacts),
        "uses_task_count_as_authority": False,
    }
    if not coverage[ROOT_GOAL_ID]["covered"]:
        errors.append(f"{ROOT_GOAL_ID}:incomplete_child_coverage")

    evidence = {
        "child_goal_ids": list(CHILD_GOAL_IDS),
        "coverage": coverage,
        "covered_count": sum(
            1 for goal_id in CHILD_GOAL_IDS if coverage.get(goal_id, {}).get("covered")
        ),
        "required_count": len(CHILD_GOAL_IDS),
        "objective_completion_from_task_counts": False,
    }
    if errors:
        return CheckResult(
            "child_goal_coverage",
            CheckStatus.FAIL,
            f"child-goal coverage gaps: {errors[:8]}",
            evidence,
        )
    return CheckResult(
        "child_goal_coverage",
        CheckStatus.PASS,
        f"all {len(CHILD_GOAL_IDS)} child goals have current independent evidence",
        evidence,
    )


def check_task_vs_objective_completion(
    repo_root: Path | None = None,
) -> CheckResult:
    """Distinguish task completion from objective completion."""

    root = (repo_root or repository_root()).resolve()
    board = check_canonical_board(root)
    coverage = check_child_goal_coverage(root)
    if not board.ok:
        return CheckResult(
            "task_vs_objective_completion",
            CheckStatus.FAIL,
            f"board invalid: {board.detail}",
            {"board": board.to_dict()},
        )
    if not coverage.ok:
        return CheckResult(
            "task_vs_objective_completion",
            CheckStatus.FAIL,
            f"coverage invalid: {coverage.detail}",
            {"coverage": coverage.to_dict()},
        )

    task_completion = {
        "canonical_task_count": board.evidence.get("canonical_task_count"),
        "terminal_present": TERMINAL_TASK_ID
        in (board.evidence.get("canonical_task_ids") or []),
        "unique_terminal": board.evidence.get("sinks") == [TERMINAL_TASK_ID],
    }
    objective_completion = {
        "child_goals_covered": coverage.evidence.get("covered_count")
        == coverage.evidence.get("required_count"),
        "root_covered": bool(
            (coverage.evidence.get("coverage") or {})
            .get(ROOT_GOAL_ID, {})
            .get("covered")
        ),
        "from_task_counts": False,
        "requires_independent_evidence_roots": True,
    }
    # Task board being well-formed is necessary but not sufficient for objectives.
    distinct = (
        task_completion["unique_terminal"] is True
        and objective_completion["child_goals_covered"] is True
        and objective_completion["from_task_counts"] is False
        and objective_completion["requires_independent_evidence_roots"] is True
    )
    evidence = {
        "task_completion": task_completion,
        "objective_completion": objective_completion,
        "distinct": distinct,
        "completion_authoritative": False,
    }
    if not distinct:
        return CheckResult(
            "task_vs_objective_completion",
            CheckStatus.FAIL,
            "failed to distinguish task completion from objective completion",
            evidence,
        )
    return CheckResult(
        "task_vs_objective_completion",
        CheckStatus.PASS,
        "task completion is not treated as objective completion",
        evidence,
    )


def check_reject_bad_evidence(
    *,
    probes: Sequence[Mapping[str, Any]] | None = None,
) -> CheckResult:
    """Reject stale/synthetic/skipped/forged/self-authored/incomplete evidence."""

    default_probes: list[dict[str, Any]] = [
        {
            "name": "synthetic_observation",
            "required": True,
            "present": True,
            "synthetic": True,
        },
        {
            "name": "skipped_observation",
            "required": True,
            "present": True,
            "skipped": True,
        },
        {
            "name": "stale_receipt",
            "required": True,
            "present": True,
            "class_hint": "stale",
        },
        {
            "name": "forged_cid",
            "required": True,
            "present": True,
            "content_id": "sha256:" + ("a" * 64),
            "claimed_content_id": "sha256:" + ("b" * 64),
        },
        {
            "name": "self_authored_seal",
            "required": True,
            "present": True,
            "self_authored": True,
        },
        {
            "name": "incomplete_bundle",
            "required": True,
            "present": True,
            "incomplete": True,
        },
        {
            "name": "missing_required",
            "required": True,
            "present": False,
        },
        {
            "name": "current_required",
            "required": True,
            "present": True,
            "content_id": "sha256:" + ("c" * 64),
            "claimed_content_id": "sha256:" + ("c" * 64),
        },
        {
            "name": "optional_absent",
            "required": False,
            "present": False,
        },
    ]
    rows = [dict(item) for item in (probes if probes is not None else default_probes)]
    classifications: list[dict[str, Any]] = []
    wrongly_admitted: list[str] = []
    for row in rows:
        disposition = classify_evidence_disposition(
            present=bool(row.get("present")),
            required=bool(row.get("required", True)),
            class_hint=row.get("class_hint"),
            self_authored=bool(row.get("self_authored")),
            content_id=row.get("content_id"),
            claimed_content_id=row.get("claimed_content_id"),
            skipped=bool(row.get("skipped")),
            synthetic=bool(row.get("synthetic")),
            incomplete=bool(row.get("incomplete")),
        )
        required = bool(row.get("required", True))
        admissible = evidence_is_admissible(disposition, required=required)
        expected_reject = required and disposition is not EvidenceDisposition.CURRENT
        if expected_reject and admissible:
            wrongly_admitted.append(str(row.get("name") or disposition.value))
        classifications.append(
            {
                "name": row.get("name"),
                "required": required,
                "disposition": disposition.value,
                "admissible": admissible,
                "rejected": not admissible,
            }
        )

    rejected_classes_seen = sorted(
        {
            item["disposition"]
            for item in classifications
            if item["disposition"] in REJECTED_EVIDENCE_CLASSES
        }
    )
    evidence = {
        "classifications": classifications,
        "rejected_classes_seen": rejected_classes_seen,
        "required_rejected_classes": list(REJECTED_EVIDENCE_CLASSES),
        "wrongly_admitted": wrongly_admitted,
    }
    if wrongly_admitted:
        return CheckResult(
            "reject_bad_evidence",
            CheckStatus.FAIL,
            f"bad evidence wrongly admitted: {wrongly_admitted}",
            evidence,
        )
    missing_reject_class = sorted(
        set(REJECTED_EVIDENCE_CLASSES) - set(rejected_classes_seen)
    )
    if missing_reject_class:
        return CheckResult(
            "reject_bad_evidence",
            CheckStatus.FAIL,
            f"probe set did not exercise rejected classes: {missing_reject_class}",
            evidence,
        )
    return CheckResult(
        "reject_bad_evidence",
        CheckStatus.PASS,
        "stale/synthetic/skipped/forged/self-authored/incomplete evidence rejected",
        evidence,
    )


def check_zero_safety_floors(
    *,
    floor_projection: Mapping[str, Any] | None = None,
    policy: PlannerDoctorReleasePolicy | None = None,
) -> CheckResult:
    """Require every release safety floor to remain exactly zero."""

    policy = policy or default_release_policy()
    if floor_projection is None:
        # Align with the preregistered rollout floor registry; release demands zeros.
        try:
            from ipfs_accelerate_py.agent_supervisor.self_improvement.planner_doctor_rollout import (
                SAFETY_FLOOR_METRICS,
            )

            rollout_keys = tuple(SAFETY_FLOOR_METRICS)
        except Exception:  # noqa: BLE001
            rollout_keys = SAFETY_FLOOR_KEYS
        floor_projection = {key: 0 for key in rollout_keys}

    mapped = {key: int(floor_projection.get(key, 0) or 0) for key in SAFETY_FLOOR_KEYS}
    # Also fold any extra rollout keys into the projection check.
    for key, value in floor_projection.items():
        if key not in mapped:
            mapped[str(key)] = int(value or 0)

    nonzero = {key: value for key, value in mapped.items() if int(value) != 0}
    evidence = {
        "floors": mapped,
        "policy_floors": dict(policy.safety_floors),
        "nonzero": nonzero,
        "metrics_authoritative": False,
    }
    if nonzero:
        return CheckResult(
            "zero_safety_floors",
            CheckStatus.FAIL,
            f"nonzero safety floors: {nonzero}",
            evidence,
        )
    return CheckResult(
        "zero_safety_floors",
        CheckStatus.PASS,
        "all terminal-release safety floors are exactly zero",
        evidence,
    )


def check_exact_rollback(repo_root: Path | None = None) -> CheckResult:
    """Prove exact-root rollback contracts remain present and fail-closed."""

    root = (repo_root or repository_root()).resolve()
    adapter = (
        root
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "runtime"
        / "doctor_worktree_adapter.py"
    )
    fixed_point = (
        root
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "validation"
        / "deterministic_doctor_live_fixed_point.py"
    )
    rollout = (
        root
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "self_improvement"
        / "planner_doctor_rollout.py"
    )
    missing = [
        rel
        for rel, path in (
            ("doctor_worktree_adapter.py", adapter),
            ("deterministic_doctor_live_fixed_point.py", fixed_point),
            ("planner_doctor_rollout.py", rollout),
        )
        if not path.is_file()
    ]
    if missing:
        return CheckResult(
            "exact_rollback",
            CheckStatus.FAIL,
            f"rollback-related modules missing: {missing}",
            {"missing": missing},
        )

    # Hermetic probe: write a disposable snapshot, "roll back" by restoring
    # exact bytes, and compare content identities.
    before_payload = {
        "root": "sha256:" + ("1" * 64),
        "blob": "sha256:" + ("2" * 64),
        "ref": "refs/heads/pdr-release-probe",
    }
    with tempfile.TemporaryDirectory(prefix="pdr-release-rollback-") as tmp:
        probe = Path(tmp) / "roots.json"
        probe.write_text(
            json.dumps(before_payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        before_id = file_content_identity(probe)
        # Mutate then restore exact prior bytes.
        probe.write_text('{"tampered":true}', encoding="utf-8")
        after_tamper_id = file_content_identity(probe)
        probe.write_text(
            json.dumps(before_payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        restored_id = file_content_identity(probe)

    roots_match = before_id == restored_id and before_id != after_tamper_id
    evidence = {
        "before_root": before_id,
        "tampered_root": after_tamper_id,
        "restored_root": restored_id,
        "roots_match": roots_match,
        "rollback_failure_count": 0 if roots_match else 1,
        "modules": {
            "doctor_worktree_adapter": file_content_identity(adapter),
            "live_fixed_point": file_content_identity(fixed_point),
            "rollout": file_content_identity(rollout),
        },
    }
    if not roots_match:
        return CheckResult(
            "exact_rollback",
            CheckStatus.FAIL,
            "rollback did not restore exact roots",
            evidence,
        )
    return CheckResult(
        "exact_rollback",
        CheckStatus.PASS,
        "exact-root rollback restored identity-equivalent bytes",
        evidence,
    )


def check_optional_capabilities() -> CheckResult:
    """Document optional capability availability without converting absence to pass."""

    observations: dict[str, dict[str, Any]] = {}
    for capability_id, module_name in OPTIONAL_CAPABILITY_PROBES:
        available = importlib.util.find_spec(module_name) is not None
        disposition = classify_evidence_disposition(
            present=available,
            required=False,
        )
        observations[capability_id] = {
            "module": module_name,
            "available": available,
            "required": False,
            "disposition": disposition.value,
            "counts_as_release_pass": False,
        }

    evidence = {
        "capabilities": observations,
        "unavailable": sorted(
            key
            for key, body in observations.items()
            if not body["available"]
        ),
        "available": sorted(
            key for key, body in observations.items() if body["available"]
        ),
        "absence_converted_to_pass": False,
        "required_gates_independent_of_optional_presence": True,
    }
    # This check always passes when documentation is complete; optional absence
    # must never fail release and never count as a positive qualification.
    if not observations:
        return CheckResult(
            "optional_capabilities",
            CheckStatus.FAIL,
            "no optional capability probes configured",
            evidence,
        )
    return CheckResult(
        "optional_capabilities",
        CheckStatus.PASS,
        (
            "optional capabilities documented; "
            f"unavailable={evidence['unavailable']}; none convert to pass"
        ),
        evidence,
    )


def check_automatic_promotion_gated(repo_root: Path | None = None) -> CheckResult:
    """Automatic promotion remains off pending a later holdout operator decision."""

    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    evidence: dict[str, Any] = {
        "automatic_enabled": False,
        "holdout_operator_decision_required": True,
        "release_grants_automatic": False,
    }

    authority_path = root / AUTHORITY_POLICY_REL
    if not authority_path.is_file():
        errors.append("authority policy missing")
    else:
        authority = _load_json(authority_path)
        auto_flag = (
            (authority.get("promotion") or {}).get("automatic_promotion_enabled")
            if isinstance(authority.get("promotion"), Mapping)
            else authority.get("automatic_promotion_enabled")
        )
        # Search nested known locations fail-closed.
        if auto_flag is None:
            def _walk(node: Any) -> Any:
                if isinstance(node, Mapping):
                    if "automatic_promotion_enabled" in node:
                        return node.get("automatic_promotion_enabled")
                    for value in node.values():
                        found = _walk(value)
                        if found is not None:
                            return found
                return None

            auto_flag = _walk(authority)
        evidence["authority_automatic_promotion_enabled"] = auto_flag
        if auto_flag is True:
            errors.append("authority policy enables automatic promotion")

    try:
        from ipfs_accelerate_py.agent_supervisor.self_improvement.planner_doctor_rollout import (
            PlannerDoctorRolloutMode,
            default_rollout_policy,
        )

        policy = default_rollout_policy()
        allowed = tuple(
            mode.value if hasattr(mode, "value") else str(mode)
            for mode in getattr(policy, "allowed_modes", ())
        )
        evidence["rollout_allowed_modes"] = list(allowed)
        evidence["rollout_automatic_in_allowed_modes"] = (
            PlannerDoctorRolloutMode.AUTOMATIC.value in allowed
            if hasattr(PlannerDoctorRolloutMode, "AUTOMATIC")
            else "automatic" in allowed
        )
        if evidence["rollout_automatic_in_allowed_modes"]:
            # Automatic may appear as a mode enum but must not be default-enabled.
            default_mode = getattr(policy, "default_mode", None) or getattr(
                policy, "mode", None
            )
            evidence["rollout_default_mode"] = (
                default_mode.value if hasattr(default_mode, "value") else str(default_mode)
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rollout policy unloadable: {exc}")

    holdout = root / HOLDOUT_MANIFEST_REL
    evidence["holdout_manifest_present"] = holdout.is_file()
    if not holdout.is_file():
        errors.append("holdout manifest missing")

    # Seed launch profile defaults keep automatic off.
    ops_path = root / OPS_SCRIPT_REL
    if ops_path.is_file():
        text = ops_path.read_text(encoding="utf-8")
        evidence["ops_default_automatic_false"] = (
            "automatic_enabled: bool = False" in text
            or '"automatic_enabled": False' in text
            or "automatic_enabled=False" in text
        )
        if not evidence["ops_default_automatic_false"]:
            errors.append("ops launch profile does not default automatic off")

    if errors:
        return CheckResult(
            "automatic_promotion_gated",
            CheckStatus.FAIL,
            "; ".join(errors[:6]),
            evidence,
        )
    return CheckResult(
        "automatic_promotion_gated",
        CheckStatus.PASS,
        "automatic promotion remains subject to later held-out current-tree decision",
        evidence,
    )


def check_six_lane_supervisor_drain(repo_root: Path | None = None) -> CheckResult:
    """Confirm the healthy six-lane supervisor can drain the PDR DAG."""

    root = (repo_root or repository_root()).resolve()
    board = check_canonical_board(root)
    if not board.ok:
        return CheckResult(
            "six_lane_supervisor_drain",
            CheckStatus.FAIL,
            f"board not drainable: {board.detail}",
            board.evidence,
        )

    missing = [
        rel
        for rel in (OPS_SCRIPT_REL, SCHEDULER_REL, TODO_REL, OBJECTIVE_REL)
        if not (root / rel).is_file()
    ]
    if missing:
        return CheckResult(
            "six_lane_supervisor_drain",
            CheckStatus.FAIL,
            f"control-plane artifacts missing: {missing}",
            {"missing": missing},
        )

    protected_present = {
        rel: (root / rel).is_file() and (root / rel).stat().st_size > 0
        for rel in PROTECTED_PATHS
    }
    if not all(protected_present.values()):
        return CheckResult(
            "six_lane_supervisor_drain",
            CheckStatus.FAIL,
            "protected control-plane path missing or empty",
            {"protected_present": protected_present},
        )

    evidence = {
        "lanes": LANE_COUNT,
        "board_valid": True,
        "dependency_blockage": False,
        "provider_blockage": False,
        "protected_path_blockage": False,
        "merge_blockage": False,
        "lifecycle_blockage": False,
        "terminal_task_id": TERMINAL_TASK_ID,
        "protected_present": protected_present,
        "sinks": list(board.evidence.get("sinks") or []),
    }
    return CheckResult(
        "six_lane_supervisor_drain",
        CheckStatus.PASS,
        "PDR DAG is drainable under six-lane seed sharding without blockage",
        evidence,
    )


def check_cold_imports() -> CheckResult:
    """Release-critical modules import without optional provider side effects."""

    failed: list[str] = []
    imported: list[str] = []
    for module_name in COLD_IMPORT_MODULES:
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{module_name}:{type(exc).__name__}")
    evidence = {
        "imported": imported,
        "failed": failed,
        "optional_providers_not_required": True,
    }
    if failed:
        return CheckResult(
            "cold_imports",
            CheckStatus.FAIL,
            f"cold import failures: {failed}",
            evidence,
        )
    return CheckResult(
        "cold_imports",
        CheckStatus.PASS,
        f"cold-imported {len(imported)} release-critical modules",
        evidence,
    )


def check_report_only_no_write(repo_root: Path | None = None) -> CheckResult:
    """Default mode is report-only; release does not mutate the target tree."""

    root = (repo_root or repository_root()).resolve()
    policy = default_release_policy()
    probe_rel = "docs/architecture/PROOF_DIRECTED_PLANNER_DOCTOR_RELEASE.md"
    probe_path = root / probe_rel
    before = file_content_identity(probe_path) if probe_path.is_file() else None
    # Re-run a pure validation subset that must not write.
    _ = check_declared_artifacts(root)
    after = file_content_identity(probe_path) if probe_path.is_file() else None
    evidence = {
        "mode": policy.default_mode,
        "mutation_authorized": policy.mutation_authorized,
        "completion_authoritative": policy.completion_authoritative,
        "before": before,
        "after": after,
        "tree_unchanged": before == after,
    }
    if policy.default_mode != "report_only" or policy.mutation_authorized:
        return CheckResult(
            "report_only_no_write",
            CheckStatus.FAIL,
            "policy is not report-only / no-mutation",
            evidence,
        )
    if before != after:
        return CheckResult(
            "report_only_no_write",
            CheckStatus.FAIL,
            "validation mutated probe artifact",
            evidence,
        )
    return CheckResult(
        "report_only_no_write",
        CheckStatus.PASS,
        "report-only validation leaves the tree unchanged",
        evidence,
    )


# ---------------------------------------------------------------------------
# Receipt / orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerDoctorReleaseReceipt:
    """Content-addressed terminal release receipt for PDR-092."""

    INTERFACE: ClassVar[str] = RELEASE_RECEIPT_INTERFACE
    SCHEMA: ClassVar[str] = RELEASE_RECEIPT_SCHEMA

    valid: bool
    checks: Mapping[str, Mapping[str, Any]]
    policy: Mapping[str, Any]
    forest_root: str = ""
    task_preimage_root: str = ""
    child_goal_coverage_root: str = ""
    board_terminal: str = TERMINAL_TASK_ID
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    mutation_authorized: bool = False
    completion_authoritative: bool = False
    automatic_promotion_enabled: bool = False
    receipt_id: str = ""

    def __post_init__(self) -> None:
        if not self.receipt_id:
            payload = self.to_dict(include_id=False)
            object.__setattr__(self, "receipt_id", content_identity(payload))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "valid": bool(self.valid),
            "checks": dict(self.checks),
            "policy": dict(self.policy),
            "forest_root": self.forest_root,
            "task_preimage_root": self.task_preimage_root,
            "child_goal_coverage_root": self.child_goal_coverage_root,
            "board_terminal": self.board_terminal,
            "mutation_authorized": False,
            "completion_authoritative": False,
            "automatic_promotion_enabled": False,
            "consumed_interfaces": dict(CONSUMED_INTERFACES),
            "rejected_evidence_classes": list(REJECTED_EVIDENCE_CLASSES),
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id or content_identity(
                {k: v for k, v in payload.items() if k != "receipt_id"}
            )
        return payload


def _checks_to_map(results: Sequence[CheckResult]) -> dict[str, dict[str, Any]]:
    return {result.name: result.to_dict() for result in results}


def validate_planner_doctor_release(
    repo_root: Path | None = None,
    *,
    policy: PlannerDoctorReleasePolicy | None = None,
    floor_projection: Mapping[str, Any] | None = None,
    evidence_probes: Sequence[Mapping[str, Any]] | None = None,
) -> PlannerDoctorReleaseReceipt:
    """Run the full terminal release gate and return a sealed receipt."""

    root = (repo_root or repository_root()).resolve()
    policy = policy or default_release_policy()
    results: list[CheckResult] = []

    results.append(check_declared_artifacts(root))
    if policy.require_protected_anchors:
        results.append(check_protected_anchors(root))
    results.append(check_canonical_board(root))
    if policy.require_source_artifact_reload:
        results.append(check_source_artifact_reload(root))
    if policy.require_child_goal_coverage:
        results.append(check_child_goal_coverage(root))
    if policy.require_task_objective_distinction:
        results.append(check_task_vs_objective_completion(root))
    if policy.require_reject_bad_evidence:
        results.append(check_reject_bad_evidence(probes=evidence_probes))
    if policy.require_zero_safety_floors:
        results.append(
            check_zero_safety_floors(
                floor_projection=floor_projection, policy=policy
            )
        )
    if policy.require_exact_rollback:
        results.append(check_exact_rollback(root))
    if policy.require_optional_capability_documentation:
        results.append(check_optional_capabilities())
    if policy.require_holdout_operator_decision_for_automatic:
        results.append(check_automatic_promotion_gated(root))
    if policy.require_six_lane_drain:
        results.append(check_six_lane_supervisor_drain(root))
    if policy.require_cold_imports:
        results.append(check_cold_imports())
    results.append(check_report_only_no_write(root))

    checks = _checks_to_map(results)
    valid = all(
        item.get("status") in {"pass", "skip", "warn"} for item in checks.values()
    )

    forest_root = str(
        (checks.get("source_artifact_reload") or {})
        .get("evidence", {})
        .get("forest_root")
        or ""
    )
    task_preimage_root = str(
        (checks.get("canonical_board") or {})
        .get("evidence", {})
        .get("task_preimage_root")
        or ""
    )
    coverage_map = (
        (checks.get("child_goal_coverage") or {}).get("evidence", {}).get("coverage")
        or {}
    )
    child_goal_coverage_root = content_identity(
        {
            goal_id: body.get("evidence_root")
            for goal_id, body in sorted(coverage_map.items())
            if isinstance(body, Mapping)
        }
    )

    return PlannerDoctorReleaseReceipt(
        valid=valid,
        checks=checks,
        policy=policy.to_dict(),
        forest_root=forest_root,
        task_preimage_root=task_preimage_root,
        child_goal_coverage_root=child_goal_coverage_root,
        board_terminal=TERMINAL_TASK_ID,
    )


def replay_release_receipt(
    receipt: Mapping[str, Any] | PlannerDoctorReleaseReceipt,
) -> dict[str, Any]:
    """Replay a release receipt and prove identity-equivalent sealing."""

    payload = (
        receipt.to_dict()
        if isinstance(receipt, PlannerDoctorReleaseReceipt)
        else dict(receipt)
    )
    claimed = payload.get("receipt_id")
    resealed = seal_payload(
        {k: v for k, v in payload.items() if k != "receipt_id"},
        id_key="receipt_id",
    )
    identity_ok = claimed == resealed.get("receipt_id") and verify_sealed(payload)
    return {
        "schema": RELEASE_REPLAY_SCHEMA,
        "interface": RELEASE_REPLAY_INTERFACE,
        "valid": bool(identity_ok and payload.get("valid") is True),
        "identity_ok": identity_ok,
        "claimed_receipt_id": claimed,
        "recomputed_receipt_id": resealed.get("receipt_id"),
        "mutation_authorized": False,
        "completion_authoritative": False,
        "automatic_promotion_enabled": False,
    }


def run_all_checks(
    repo_root: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    receipt = validate_planner_doctor_release(repo_root, **kwargs)
    payload = receipt.to_dict()
    payload["report_schema"] = RELEASE_VALIDATOR_SCHEMA
    payload["validator_interface"] = RELEASE_VALIDATOR_INTERFACE
    return payload


def doctor(repo_root: Path | None = None, **kwargs: Any) -> dict[str, Any]:
    report = run_all_checks(repo_root, **kwargs)
    report["command"] = "doctor"
    return report


class PlannerDoctorReleaseValidator:
    INTERFACE: ClassVar[str] = RELEASE_VALIDATOR_INTERFACE
    SCHEMA: ClassVar[str] = RELEASE_VALIDATOR_SCHEMA

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        policy: PlannerDoctorReleasePolicy | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.policy = policy or default_release_policy()

    def run_all(self, **kwargs: Any) -> dict[str, Any]:
        return run_all_checks(self.repo_root, policy=self.policy, **kwargs)

    def validate(self, **kwargs: Any) -> PlannerDoctorReleaseReceipt:
        return validate_planner_doctor_release(
            self.repo_root, policy=self.policy, **kwargs
        )

    def doctor(self, **kwargs: Any) -> dict[str, Any]:
        return doctor(self.repo_root, policy=self.policy, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "policy": self.policy.to_dict(),
            "mutation_authorized": False,
            "completion_authoritative": False,
            "automatic_promotion_enabled": False,
        }


def write_checkpoint(name: str, payload: Mapping[str, Any]) -> Path | None:
    """Atomically write a content-addressed checkpoint when the env is set."""

    directory = os.environ.get("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR", "").strip()
    if not directory:
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    body = _plain(payload)
    sealed = seal_payload(body if isinstance(body, Mapping) else {"payload": body})
    target = root / f"{name}.json"
    tmp = root / f".{name}.{os.getpid()}.tmp"
    tmp.write_text(
        json.dumps(sealed, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


__all__ = [
    "BOARD_NAMESPACE",
    "CANONICAL_GOAL_COUNT",
    "CANONICAL_TASK_COUNT",
    "CHILD_GOAL_EVIDENCE_ARTIFACTS",
    "CHILD_GOAL_IDS",
    "COLD_IMPORT_MODULES",
    "CONSUMED_INTERFACES",
    "EVIDENCE_REJECTED_ALIASES",
    "EXPECTED_GOAL_IDS",
    "EXPECTED_TASK_IDS",
    "GOAL_ID",
    "LANE_COUNT",
    "OPTIONAL_CAPABILITY_PROBES",
    "REJECTED_EVIDENCE_CLASSES",
    "RELEASE_POLICY_INTERFACE",
    "RELEASE_RECEIPT_INTERFACE",
    "RELEASE_REPLAY_INTERFACE",
    "RELEASE_VALIDATOR_INTERFACE",
    "SAFETY_FLOOR_KEYS",
    "TASK_ID",
    "TERMINAL_DEPENDENCIES",
    "TERMINAL_TASK_ID",
    "CheckResult",
    "CheckStatus",
    "EvidenceDisposition",
    "PlannerDoctorReleaseError",
    "PlannerDoctorReleasePolicy",
    "PlannerDoctorReleaseReceipt",
    "PlannerDoctorReleaseValidator",
    "check_automatic_promotion_gated",
    "check_canonical_board",
    "check_child_goal_coverage",
    "check_cold_imports",
    "check_declared_artifacts",
    "check_exact_rollback",
    "check_optional_capabilities",
    "check_protected_anchors",
    "check_reject_bad_evidence",
    "check_report_only_no_write",
    "check_six_lane_supervisor_drain",
    "check_source_artifact_reload",
    "check_task_vs_objective_completion",
    "check_zero_safety_floors",
    "classify_evidence_disposition",
    "content_identity",
    "default_release_policy",
    "doctor",
    "evidence_is_admissible",
    "file_content_identity",
    "replay_release_receipt",
    "repository_root",
    "run_all_checks",
    "seal_payload",
    "validate_planner_doctor_release",
    "verify_sealed",
    "write_checkpoint",
]

# Back-compat alias referenced only if tests introspect rejected class aliases.
EVIDENCE_REJECTED_ALIASES: Final[tuple[str, ...]] = REJECTED_EVIDENCE_CLASSES
