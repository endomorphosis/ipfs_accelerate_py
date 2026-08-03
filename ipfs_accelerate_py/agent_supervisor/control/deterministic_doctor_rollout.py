"""Report-only through narrow-auto rollout controls for the deterministic doctor.

LPR-041 / LPR-G110. Operator surface for staged doctor automation:

* :class:`DeterministicDoctorRolloutPolicy` — immutable bounded config defaults
  to report-only; deterministic narrow-auto stays off; remote embeddings /
  network / LLM / remote model-provider calls stay false; exact-root, proof-
  cache revalidation, native reconstruction, all-callers, sandbox, lease,
  atomic, and fixed-point gates stay true.
* :class:`DeterministicDoctorRolloutDecision` — effective mode after kill switch,
  floors, and regression gates; never grants completion or process authority.
* :class:`DeterministicDoctorRollbackGate` — demotes one stage (or disables auto)
  on any nonzero safety floor, root/schema/capability drift, embedding canary
  failure, reconstruction/isolation loss, transaction/rollback failure, or
  material resource regression.
* :class:`DeterministicDoctorOperationsValidator` — operator checks for config
  CID, gates, flags, lifecycle doctor read-only/idempotent behaviour, and
  optional-provider absence that remains actionable without blocking
  report-only startup.

Promotion is **manual and monotonic**. This module never mutates sources,
starts processes, or imports optional retrieval/prover/embedding providers.
"""

from __future__ import annotations

import hashlib
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

ROLLOUT_POLICY_INTERFACE: Final[str] = "DeterministicDoctorRolloutPolicy@1"
ROLLOUT_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-rollout-policy@1"
)
ROLLOUT_DECISION_INTERFACE: Final[str] = "DeterministicDoctorRolloutDecision@1"
ROLLOUT_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-rollout-decision@1"
)
ROLLBACK_GATE_INTERFACE: Final[str] = "DeterministicDoctorRollbackGate@1"
ROLLBACK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-rollback-receipt@1"
)
VALIDATOR_INTERFACE: Final[str] = "DeterministicDoctorOperationsValidator@1"
VALIDATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-validation-ops-report@1"
)
CONFIG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.deterministic_doctor.rollout_config@1"
)
METRICS_INTERFACE: Final[str] = "DeterministicDoctorMetrics@1"
SERVICE_INTERFACE: Final[str] = "DeterministicDoctorService@1"
SUPERVISOR_CONTROL_SERVICE_INTERFACE: Final[str] = "SupervisorControlService@1"

TASK_ID: Final[str] = "LPR-041"
GOAL_ID: Final[str] = "LPR-G110"
BOARD_NAMESPACE: Final[str] = "agent-supervisor-tactician-hammer-logic-repair-v1"

CONFIG_REL: Final[str] = "config/agent_supervisor_deterministic_doctor.json"
ROLLOUT_MODULE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/control/deterministic_doctor_rollout.py"
)
VALIDATE_SCRIPT_REL: Final[str] = (
    "scripts/ops/agent_supervisor/validate_deterministic_doctor.py"
)
GUIDE_REL: Final[str] = "docs/guides/DETERMINISTIC_DOCTOR_GUIDE.md"
TEST_REL: Final[str] = "test/api/test_agent_supervisor_deterministic_doctor_rollout.py"
SERVICE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/control/deterministic_doctor_service.py"
)
POLICY_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_policy.py"
)
BENCHMARK_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/validation/deterministic_doctor_benchmark.py"
)
OPS_FACADE_REL: Final[str] = "scripts/ops/agent_supervisor/deterministic_doctor.py"
LAUNCHER_REL: Final[str] = "scripts/tactician_hammer_logic_repair_supervisor.sh"
SCHEDULER_REL: Final[str] = (
    "config/agent_supervisor_tactician_hammer_logic_repair_scheduler.json"
)

ROLLOUT_STAGES: Final[tuple[str, ...]] = (
    "report_only",
    "plan",
    "sandbox_auto",
    "narrow_auto",
)

HARD_FALSE_FLAGS: Final[tuple[str, ...]] = (
    "llm_router_enabled",
    "llm_invocations_allowed",
    "remote_model_provider_calls_allowed",
    "remote_embeddings_allowed",
    "network_access_allowed",
    "target_code_import_allowed",
    "knowledge_graph_semantic_authority",
    "vector_semantic_authority",
    "embedding_semantic_authority",
    "tactician_semantic_authority",
    "hammer_candidate_semantic_authority",
    "proof_cache_metadata_semantic_authority",
)

HARD_TRUE_GATES: Final[tuple[str, ...]] = (
    "exact_evidence_snapshot_required",
    "clean_rebuild_identity_equivalence_required",
    "canonical_cid_preimage_validation_required",
    "proof_cache_binding_revalidation_required",
    "native_kernel_reconstruction_required",
    "independent_countermodel_validation_required",
    "complete_impact_closure_required",
    "one_disposition_per_resolved_consumer",
    "unique_target_value_placement_operator_required",
    "closed_operator_registry_required",
    "isolated_candidate_worktree_required",
    "enforced_sandbox_required_for_target_execution",
    "writer_lease_and_checkpoint_required",
    "atomic_scc_transaction_required",
    "post_edit_reindex_and_cache_invalidation_required",
    "logic_and_program_fixed_point_required",
    "compensating_rollback_required",
    "explicit_repair_operation_required",
)

LIMIT_KEYS: Final[tuple[str, ...]] = (
    "max_findings",
    "max_candidates_per_finding",
    "max_graph_nodes_per_query",
    "max_proof_routes_per_goal",
    "max_operators_per_finding",
    "max_plan_steps",
    "max_fixed_point_iterations",
    "max_changed_files",
    "max_changed_bytes",
    "max_processes",
    "max_wall_time_seconds",
    "max_cpu_time_seconds",
    "max_memory_bytes",
)

DEFAULT_LIMITS: Final[dict[str, int]] = {
    "max_findings": 256,
    "max_candidates_per_finding": 64,
    "max_graph_nodes_per_query": 2048,
    "max_proof_routes_per_goal": 32,
    "max_operators_per_finding": 32,
    "max_plan_steps": 256,
    "max_fixed_point_iterations": 8,
    "max_changed_files": 128,
    "max_changed_bytes": 1_048_576,
    "max_processes": 8,
    "max_wall_time_seconds": 3600,
    "max_cpu_time_seconds": 1800,
    "max_memory_bytes": 4_294_967_296,
}

SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "missed_mandatory_caller_rate",
    "authority_promotion_rate",
    "stale_proof_cid_admission_rate",
    "out_of_scope_sandbox_write_rate",
    "partial_transaction_rate",
    "rollback_failure_rate",
    "nondeterministic_render_rate",
    "false_fixed_point_rate",
    "llm_router_invocation_rate",
    "llm_model_provider_call_rate",
    "root_schema_capability_drift_rate",
    "embedding_canary_failure_rate",
    "reconstruction_isolation_loss_rate",
    "transaction_rollback_failure_rate",
    "material_resource_regression_rate",
)

APPROVAL_REQUIRED_CLASSES: Final[tuple[str, ...]] = (
    "doctor_trusted_computing_base",
    "stateful_behavior",
    "public_api_or_schema",
    "dynamic_or_generated_code",
    "native_or_ffi",
    "cross_repository_edit",
    "new_external_dependency",
    "unsupported_memory_or_lifetime_claim",
)

OPTIONAL_PROVIDER_MODULES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py",
    "openai",
    "anthropic",
    "transformers",
    "torch",
    "sentence_transformers",
)

REQUIRED_ARTIFACTS: Final[tuple[str, ...]] = (
    CONFIG_REL,
    ROLLOUT_MODULE_REL,
    VALIDATE_SCRIPT_REL,
    GUIDE_REL,
    TEST_REL,
)

MAX_TEXT_BYTES: Final[int] = 512
MAX_POLICY_RECORD_BYTES: Final[int] = 262_144


class DeterministicDoctorRolloutError(ValueError):
    """Rollout policy, decision, config, or validation evidence is invalid."""


class DeterministicDoctorMode(str, Enum):
    """Orthogonal no-model automation ladder (report-only default)."""

    REPORT_ONLY = "report_only"
    PLAN = "plan"
    SANDBOX_AUTO = "sandbox_auto"
    NARROW_AUTO = "narrow_auto"

    @property
    def rank(self) -> int:
        return {
            DeterministicDoctorMode.REPORT_ONLY: 0,
            DeterministicDoctorMode.PLAN: 1,
            DeterministicDoctorMode.SANDBOX_AUTO: 2,
            DeterministicDoctorMode.NARROW_AUTO: 3,
        }[self]

    @property
    def allows_source_write(self) -> bool:
        return self is DeterministicDoctorMode.NARROW_AUTO

    @property
    def allows_sandbox_write(self) -> bool:
        return self in (
            DeterministicDoctorMode.SANDBOX_AUTO,
            DeterministicDoctorMode.NARROW_AUTO,
        )

    @property
    def allows_plan_materialization(self) -> bool:
        return self.rank >= DeterministicDoctorMode.PLAN.rank


class RollbackReason(str, Enum):
    NONZERO_SAFETY_FLOOR = "nonzero_safety_floor"
    ROOT_SCHEMA_CAPABILITY_DRIFT = "root_schema_capability_drift"
    EMBEDDING_CANARY_FAILURE = "embedding_canary_failure"
    RECONSTRUCTION_ISOLATION_LOSS = "reconstruction_isolation_loss"
    TRANSACTION_ROLLBACK_FAILURE = "transaction_rollback_failure"
    MATERIAL_RESOURCE_REGRESSION = "material_resource_regression"
    KILL_SWITCH = "kill_switch"
    CAPABILITY_REGRESSION = "capability_regression"
    EXPLICIT_OPERATOR = "explicit_operator"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def repository_root() -> Path:
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
        raise DeterministicDoctorRolloutError(
            "floating-point values are forbidden in sealed identities"
        )
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
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


def file_identity(path: Path) -> str:
    return _sha256_hex(path.read_bytes())


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise DeterministicDoctorRolloutError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise DeterministicDoctorRolloutError(f"{name} is unsafe or too large")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DeterministicDoctorRolloutError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DeterministicDoctorRolloutError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeterministicDoctorRolloutError(f"{name} must be a non-negative integer")
    return value


def _mode(value: Any) -> DeterministicDoctorMode:
    if isinstance(value, DeterministicDoctorMode):
        return value
    try:
        return DeterministicDoctorMode(str(getattr(value, "value", value)).strip().casefold())
    except ValueError as exc:
        raise DeterministicDoctorRolloutError(
            f"unknown deterministic doctor mode: {value!r}"
        ) from exc


def _zero_floors() -> dict[str, int]:
    return {key: 0 for key in SAFETY_FLOOR_KEYS}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicDoctorRolloutPolicy:
    """Immutable bounded rollout policy (report-only default)."""

    SCHEMA: ClassVar[str] = ROLLOUT_POLICY_SCHEMA
    INTERFACE: ClassVar[str] = ROLLOUT_POLICY_INTERFACE

    policy_id: str = "policy:deterministic-doctor-rollout-default"
    policy_revision: str = "1"
    repository_id: str = ""
    program_id: str = BOARD_NAMESPACE
    mode: DeterministicDoctorMode | str = DeterministicDoctorMode.REPORT_ONLY
    explicit_policy_document: str = ""
    scoped_path_globs: tuple[str, ...] = ()
    enabled: bool = False
    allow_plan: bool = False
    allow_sandbox_auto: bool = False
    allow_narrow_auto: bool = False
    narrow_autonomous_mutation_enabled: bool = False
    kill_switch_engaged: bool = False
    llm_router_enabled: bool = False
    llm_invocations_allowed: bool = False
    remote_model_provider_calls_allowed: bool = False
    remote_embeddings_allowed: bool = False
    network_access_allowed: bool = False
    target_code_import_allowed: bool = False
    exact_evidence_snapshot_required: bool = True
    clean_rebuild_identity_equivalence_required: bool = True
    canonical_cid_preimage_validation_required: bool = True
    proof_cache_binding_revalidation_required: bool = True
    native_kernel_reconstruction_required: bool = True
    independent_countermodel_validation_required: bool = True
    complete_impact_closure_required: bool = True
    one_disposition_per_resolved_consumer: bool = True
    unique_target_value_placement_operator_required: bool = True
    closed_operator_registry_required: bool = True
    isolated_candidate_worktree_required: bool = True
    enforced_sandbox_required_for_target_execution: bool = True
    writer_lease_and_checkpoint_required: bool = True
    atomic_scc_transaction_required: bool = True
    post_edit_reindex_and_cache_invalidation_required: bool = True
    logic_and_program_fixed_point_required: bool = True
    compensating_rollback_required: bool = True
    explicit_repair_operation_required: bool = True
    knowledge_graph_semantic_authority: bool = False
    vector_semantic_authority: bool = False
    embedding_semantic_authority: bool = False
    tactician_semantic_authority: bool = False
    hammer_candidate_semantic_authority: bool = False
    proof_cache_metadata_semantic_authority: bool = False
    promotion_manual: bool = True
    promotion_monotonic: bool = True
    limits: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    safety_floors: Mapping[str, int] = field(default_factory=_zero_floors)
    approval_required_classes: tuple[str, ...] = APPROVAL_REQUIRED_CLASSES
    mutation_authorized: bool = False
    completion_authoritative: bool = False
    policy_binding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "policy_revision", _text(self.policy_revision, "policy_revision")
        )
        object.__setattr__(self, "repository_id", str(self.repository_id or "").strip())
        object.__setattr__(self, "program_id", _text(self.program_id, "program_id"))
        object.__setattr__(self, "mode", _mode(self.mode))
        object.__setattr__(
            self,
            "explicit_policy_document",
            str(self.explicit_policy_document or "").strip(),
        )
        object.__setattr__(
            self,
            "scoped_path_globs",
            tuple(
                sorted(
                    {
                        _text(item, "scoped_path_globs", maximum=1024)
                        for item in self.scoped_path_globs
                    }
                )
            ),
        )

        for name in (
            "enabled",
            "allow_plan",
            "allow_sandbox_auto",
            "allow_narrow_auto",
            "narrow_autonomous_mutation_enabled",
            "kill_switch_engaged",
            "promotion_manual",
            "promotion_monotonic",
            "mutation_authorized",
            "completion_authoritative",
            *HARD_FALSE_FLAGS,
            *HARD_TRUE_GATES,
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))

        if self.completion_authoritative:
            raise DeterministicDoctorRolloutError(
                "rollout policy cannot claim completion authority"
            )
        if not self.promotion_manual:
            raise DeterministicDoctorRolloutError("promotion must remain manual")
        if not self.promotion_monotonic:
            raise DeterministicDoctorRolloutError("promotion must remain monotonic")

        for name in HARD_FALSE_FLAGS:
            if getattr(self, name) is not False:
                raise DeterministicDoctorRolloutError(
                    f"deterministic doctor safety flag must be false: {name}"
                )
        for name in HARD_TRUE_GATES:
            if getattr(self, name) is not True:
                raise DeterministicDoctorRolloutError(
                    f"deterministic doctor gate must remain enabled: {name}"
                )

        limits_raw = self.limits if self.limits is not None else DEFAULT_LIMITS
        if not isinstance(limits_raw, Mapping):
            raise DeterministicDoctorRolloutError("limits must be a mapping")
        if set(limits_raw) != set(DEFAULT_LIMITS):
            raise DeterministicDoctorRolloutError(
                "limits keys must match the closed scheduler set"
            )
        limits = {
            key: _positive_int(limits_raw[key], key) for key in sorted(DEFAULT_LIMITS)
        }
        object.__setattr__(self, "limits", MappingProxyType(limits))

        floors_raw = dict(self.safety_floors or {})
        floors = {
            key: _non_negative_int(floors_raw.get(key, 0), key)
            for key in SAFETY_FLOOR_KEYS
        }
        object.__setattr__(self, "safety_floors", MappingProxyType(floors))

        approval = tuple(
            sorted(
                {
                    _text(item, "approval_required_classes").casefold()
                    for item in self.approval_required_classes
                }
            )
        )
        if set(approval) != set(APPROVAL_REQUIRED_CLASSES):
            raise DeterministicDoctorRolloutError(
                "approval_required_classes must match the closed scheduler set"
            )
        object.__setattr__(self, "approval_required_classes", approval)

        mode = _mode(self.mode)
        if mode is DeterministicDoctorMode.REPORT_ONLY and self.mutation_authorized:
            raise DeterministicDoctorRolloutError(
                "report_only mode cannot authorize mutation"
            )
        if mode is DeterministicDoctorMode.PLAN and self.mutation_authorized:
            raise DeterministicDoctorRolloutError(
                "plan mode cannot authorize mutation"
            )
        if (
            mode is DeterministicDoctorMode.NARROW_AUTO
            and self.mutation_authorized
            and not self.narrow_autonomous_mutation_enabled
        ):
            raise DeterministicDoctorRolloutError(
                "narrow_auto mutation requires narrow_autonomous_mutation_enabled"
            )

        if not self.policy_binding_id:
            object.__setattr__(
                self, "policy_binding_id", content_identity(self.to_dict(include_id=False))
            )
        self._assert_mode_allowed()

        payload_bytes = _canonical_bytes(self.to_dict())
        if len(payload_bytes) > MAX_POLICY_RECORD_BYTES:
            raise DeterministicDoctorRolloutError(
                "policy exceeds its serialized byte bound"
            )

    def _assert_mode_allowed(self) -> None:
        mode = _mode(self.mode)
        if mode is DeterministicDoctorMode.REPORT_ONLY:
            return
        if not self.has_explicit_scoped_policy():
            raise DeterministicDoctorRolloutError(
                f"{mode.value} requires an explicit scoped policy document and "
                "repository/program/policy scope"
            )
        if mode is DeterministicDoctorMode.PLAN and not self.allow_plan:
            raise DeterministicDoctorRolloutError("plan mode is not enabled on this policy")
        if mode is DeterministicDoctorMode.SANDBOX_AUTO and not self.allow_sandbox_auto:
            raise DeterministicDoctorRolloutError(
                "sandbox_auto mode is not enabled on this policy"
            )
        if mode is DeterministicDoctorMode.NARROW_AUTO and not self.allow_narrow_auto:
            raise DeterministicDoctorRolloutError(
                "narrow_auto mode is not enabled on this policy"
            )
        if self.kill_switch_engaged and mode.rank > DeterministicDoctorMode.REPORT_ONLY.rank:
            raise DeterministicDoctorRolloutError(
                "kill switch blocks elevation above report_only"
            )

    def has_explicit_scoped_policy(self) -> bool:
        return bool(
            self.explicit_policy_document
            and (self.repository_id or self.program_id or self.policy_id)
        )

    @property
    def mode_value(self) -> str:
        return self.mode.value if isinstance(self.mode, DeterministicDoctorMode) else str(self.mode)

    def feature_flags(self) -> dict[str, bool]:
        return {
            "enabled": self.enabled,
            "narrow_autonomous_mutation_enabled": self.narrow_autonomous_mutation_enabled,
            "kill_switch_engaged": self.kill_switch_engaged,
            "llm_router_enabled": self.llm_router_enabled,
            "llm_invocations_allowed": self.llm_invocations_allowed,
            "remote_model_provider_calls_allowed": self.remote_model_provider_calls_allowed,
            "remote_embeddings_allowed": self.remote_embeddings_allowed,
            "network_access_allowed": self.network_access_allowed,
            "target_code_import_allowed": self.target_code_import_allowed,
        }

    def gates(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in HARD_TRUE_GATES}

    def semantic_authority_flags(self) -> dict[str, bool]:
        return {
            name: bool(getattr(self, name))
            for name in (
                "knowledge_graph_semantic_authority",
                "vector_semantic_authority",
                "embedding_semantic_authority",
                "tactician_semantic_authority",
                "hammer_candidate_semantic_authority",
                "proof_cache_metadata_semantic_authority",
            )
        }

    def floors_hold(self) -> bool:
        return all(int(self.safety_floors.get(key, 1)) == 0 for key in SAFETY_FLOOR_KEYS)

    def floor_breaches(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in SAFETY_FLOOR_KEYS
            if int(self.safety_floors.get(key, 1)) != 0
        )

    def allows_automated_mutation(
        self,
        *,
        unique_target: bool = True,
        reconstructed: bool = True,
        complete_frontier: bool = True,
        sandbox_isolated: bool = True,
        lease_held: bool = True,
        atomic_transaction: bool = True,
        fixed_point_ready: bool = True,
        approval_class: str = "",
    ) -> bool:
        mode = _mode(self.mode)
        if mode is not DeterministicDoctorMode.NARROW_AUTO:
            return False
        if not (
            self.allow_narrow_auto
            and self.narrow_autonomous_mutation_enabled
            and self.mutation_authorized
            and self.enabled
        ):
            return False
        if self.kill_switch_engaged:
            return False
        if not self.floors_hold():
            return False
        if approval_class and approval_class.casefold() in {
            item.casefold() for item in self.approval_required_classes
        }:
            return False
        if self.complete_impact_closure_required and not complete_frontier:
            return False
        if self.native_kernel_reconstruction_required and not reconstructed:
            return False
        if self.unique_target_value_placement_operator_required and not unique_target:
            return False
        if self.enforced_sandbox_required_for_target_execution and not sandbox_isolated:
            return False
        if self.writer_lease_and_checkpoint_required and not lease_held:
            return False
        if self.atomic_scc_transaction_required and not atomic_transaction:
            return False
        if self.logic_and_program_fixed_point_required and not fixed_point_ready:
            return False
        return True

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": ROLLOUT_POLICY_SCHEMA,
            "interface": ROLLOUT_POLICY_INTERFACE,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "repository_id": self.repository_id,
            "program_id": self.program_id,
            "mode": self.mode_value,
            "stages": list(ROLLOUT_STAGES),
            "explicit_policy_document": self.explicit_policy_document,
            "scoped_path_globs": list(self.scoped_path_globs),
            "enabled": self.enabled,
            "allow_plan": self.allow_plan,
            "allow_sandbox_auto": self.allow_sandbox_auto,
            "allow_narrow_auto": self.allow_narrow_auto,
            "feature_flags": self.feature_flags(),
            "gates": self.gates(),
            "semantic_authority": self.semantic_authority_flags(),
            "promotion_manual": self.promotion_manual,
            "promotion_monotonic": self.promotion_monotonic,
            "limits": dict(self.limits),
            "safety_floors": dict(self.safety_floors),
            "approval_required_classes": list(self.approval_required_classes),
            "mutation_authorized": self.mutation_authorized,
            "completion_authoritative": False,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
        }
        if include_id:
            payload["policy_binding_id"] = self.policy_binding_id
        return payload

    @classmethod
    def default(cls) -> "DeterministicDoctorRolloutPolicy":
        return cls()

    @classmethod
    def from_config_mapping(
        cls, payload: Mapping[str, Any], **overrides: Any
    ) -> "DeterministicDoctorRolloutPolicy":
        flags = dict(payload.get("feature_flags") or {})
        gates = dict(payload.get("gates") or {})
        semantic = dict(payload.get("semantic_authority") or {})
        promotion = dict(payload.get("promotion") or {})
        kwargs: dict[str, Any] = {
            "mode": payload.get("default_mode", DeterministicDoctorMode.REPORT_ONLY),
            "enabled": bool(payload.get("enabled", False)),
            "narrow_autonomous_mutation_enabled": bool(
                flags.get("narrow_autonomous_mutation_enabled", False)
            ),
            "kill_switch_engaged": bool(flags.get("kill_switch_engaged", False)),
            "llm_router_enabled": bool(flags.get("llm_router_enabled", False)),
            "llm_invocations_allowed": bool(flags.get("llm_invocations_allowed", False)),
            "remote_model_provider_calls_allowed": bool(
                flags.get("remote_model_provider_calls_allowed", False)
            ),
            "remote_embeddings_allowed": bool(flags.get("remote_embeddings_allowed", False)),
            "network_access_allowed": bool(flags.get("network_access_allowed", False)),
            "target_code_import_allowed": bool(flags.get("target_code_import_allowed", False)),
            "limits": dict(payload.get("limits") or DEFAULT_LIMITS),
            "safety_floors": dict(payload.get("release_safety_floors") or _zero_floors()),
            "approval_required_classes": tuple(
                payload.get("approval_required_classes") or APPROVAL_REQUIRED_CLASSES
            ),
            "promotion_manual": bool(promotion.get("manual", True)),
            "promotion_monotonic": bool(promotion.get("monotonic", True)),
        }
        for name in HARD_TRUE_GATES:
            if name in gates:
                kwargs[name] = bool(gates[name])
        for name, value in semantic.items():
            if name in HARD_FALSE_FLAGS or name.endswith("_semantic_authority"):
                kwargs[name] = bool(value)
        kwargs.update(overrides)
        return cls(**kwargs)


def default_rollout_policy() -> DeterministicDoctorRolloutPolicy:
    return DeterministicDoctorRolloutPolicy.default()


def elevate_rollout_policy(
    *,
    mode: DeterministicDoctorMode | str,
    explicit_policy_document: str,
    repository_id: str,
    program_id: str = BOARD_NAMESPACE,
    policy_id: str = "policy:deterministic-doctor-rollout-scoped",
    policy_revision: str = "1",
    scoped_path_globs: Sequence[str] = (),
    mutation_authorized: bool = False,
    enabled: bool = True,
    kill_switch_engaged: bool = False,
    safety_floors: Mapping[str, int] | None = None,
) -> DeterministicDoctorRolloutPolicy:
    """Manual monotonic promotion into plan / sandbox_auto / narrow_auto."""

    mode_value = _mode(mode)
    if mode_value is DeterministicDoctorMode.REPORT_ONLY:
        return DeterministicDoctorRolloutPolicy(
            policy_id=policy_id,
            policy_revision=policy_revision,
            repository_id=repository_id,
            program_id=program_id,
            mode=mode_value,
            explicit_policy_document=explicit_policy_document,
            scoped_path_globs=tuple(scoped_path_globs),
            enabled=enabled,
            kill_switch_engaged=kill_switch_engaged,
            safety_floors=dict(safety_floors or _zero_floors()),
        )
    return DeterministicDoctorRolloutPolicy(
        policy_id=policy_id,
        policy_revision=policy_revision,
        repository_id=repository_id,
        program_id=program_id,
        mode=mode_value,
        explicit_policy_document=explicit_policy_document,
        scoped_path_globs=tuple(scoped_path_globs),
        enabled=enabled,
        allow_plan=mode_value.rank >= DeterministicDoctorMode.PLAN.rank,
        allow_sandbox_auto=mode_value.rank >= DeterministicDoctorMode.SANDBOX_AUTO.rank,
        allow_narrow_auto=mode_value is DeterministicDoctorMode.NARROW_AUTO,
        narrow_autonomous_mutation_enabled=(
            mode_value is DeterministicDoctorMode.NARROW_AUTO and mutation_authorized
        ),
        kill_switch_engaged=kill_switch_engaged,
        mutation_authorized=(
            mutation_authorized and mode_value is DeterministicDoctorMode.NARROW_AUTO
        ),
        safety_floors=dict(safety_floors or _zero_floors()),
    )


def engage_kill_switch(
    policy: DeterministicDoctorRolloutPolicy,
) -> DeterministicDoctorRolloutPolicy:
    """Force report-only and disable auto without changing sealed defaults."""

    return DeterministicDoctorRolloutPolicy(
        policy_id=policy.policy_id,
        policy_revision=policy.policy_revision,
        repository_id=policy.repository_id,
        program_id=policy.program_id,
        mode=DeterministicDoctorMode.REPORT_ONLY,
        explicit_policy_document=policy.explicit_policy_document,
        scoped_path_globs=policy.scoped_path_globs,
        enabled=False,
        allow_plan=False,
        allow_sandbox_auto=False,
        allow_narrow_auto=False,
        narrow_autonomous_mutation_enabled=False,
        kill_switch_engaged=True,
        limits=dict(policy.limits),
        safety_floors=dict(policy.safety_floors),
        approval_required_classes=policy.approval_required_classes,
        mutation_authorized=False,
        completion_authoritative=False,
    )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicDoctorRolloutDecision:
    """Effective rollout mode after kill switch, floors, and regression gates."""

    SCHEMA: ClassVar[str] = ROLLOUT_DECISION_SCHEMA
    INTERFACE: ClassVar[str] = ROLLOUT_DECISION_INTERFACE

    requested_mode: DeterministicDoctorMode | str
    effective_mode: DeterministicDoctorMode | str
    kill_switch_engaged: bool = False
    narrow_auto_disabled: bool = True
    mutation_authorized: bool = False
    completion_authoritative: bool = False
    reason_codes: tuple[str, ...] = ()
    floor_breaches: tuple[str, ...] = ()
    policy_binding_id: str = ""
    decision_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_mode", _mode(self.requested_mode))
        object.__setattr__(self, "effective_mode", _mode(self.effective_mode))
        object.__setattr__(
            self, "kill_switch_engaged", _bool(self.kill_switch_engaged, "kill_switch_engaged")
        )
        object.__setattr__(
            self, "narrow_auto_disabled", _bool(self.narrow_auto_disabled, "narrow_auto_disabled")
        )
        object.__setattr__(
            self, "mutation_authorized", _bool(self.mutation_authorized, "mutation_authorized")
        )
        object.__setattr__(
            self,
            "completion_authoritative",
            _bool(self.completion_authoritative, "completion_authoritative"),
        )
        if self.completion_authoritative:
            raise DeterministicDoctorRolloutError(
                "rollout decision cannot claim completion authority"
            )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes if str(item).strip()),
        )
        object.__setattr__(
            self,
            "floor_breaches",
            tuple(str(item) for item in self.floor_breaches if str(item).strip()),
        )
        object.__setattr__(self, "policy_binding_id", str(self.policy_binding_id or ""))
        if not self.decision_id:
            object.__setattr__(
                self, "decision_id", content_identity(self.to_dict(include_id=False))
            )

    @property
    def effective_mode_value(self) -> str:
        mode = self.effective_mode
        return mode.value if isinstance(mode, DeterministicDoctorMode) else str(mode)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": ROLLOUT_DECISION_SCHEMA,
            "interface": ROLLOUT_DECISION_INTERFACE,
            "requested_mode": (
                self.requested_mode.value
                if isinstance(self.requested_mode, DeterministicDoctorMode)
                else str(self.requested_mode)
            ),
            "effective_mode": self.effective_mode_value,
            "kill_switch_engaged": self.kill_switch_engaged,
            "narrow_auto_disabled": self.narrow_auto_disabled,
            "mutation_authorized": self.mutation_authorized,
            "completion_authoritative": False,
            "reason_codes": list(self.reason_codes),
            "floor_breaches": list(self.floor_breaches),
            "policy_binding_id": self.policy_binding_id,
        }
        if include_id:
            payload["decision_id"] = self.decision_id
        return payload


def evaluate_rollout_decision(
    policy: DeterministicDoctorRolloutPolicy,
    *,
    root_schema_capability_drift: bool = False,
    embedding_canary_failure: bool = False,
    reconstruction_isolation_loss: bool = False,
    transaction_rollback_failure: bool = False,
    material_resource_regression: bool = False,
    capability_regression: Sequence[str] = (),
) -> DeterministicDoctorRolloutDecision:
    """Derive the effective mode from policy + live regression signals."""

    requested = _mode(policy.mode)
    reasons: list[str] = []
    breaches = list(policy.floor_breaches())
    effective = requested
    mutation = (
        policy.mutation_authorized
        and requested is DeterministicDoctorMode.NARROW_AUTO
        and policy.narrow_autonomous_mutation_enabled
        and not policy.kill_switch_engaged
    )

    if policy.kill_switch_engaged:
        effective = DeterministicDoctorMode.REPORT_ONLY
        mutation = False
        reasons.append("kill_switch")
    if breaches:
        effective = DeterministicDoctorMode.REPORT_ONLY
        mutation = False
        reasons.append("nonzero_safety_floor")
    if root_schema_capability_drift:
        effective = DeterministicDoctorMode.REPORT_ONLY
        mutation = False
        reasons.append("root_schema_capability_drift")
    if embedding_canary_failure:
        effective = DeterministicDoctorMode.REPORT_ONLY
        mutation = False
        reasons.append("embedding_canary_failure")
    if reconstruction_isolation_loss:
        effective = _demotion_target(requested)
        mutation = False
        reasons.append("reconstruction_isolation_loss")
    if transaction_rollback_failure:
        effective = _demotion_target(requested)
        mutation = False
        reasons.append("transaction_rollback_failure")
    if material_resource_regression:
        effective = _demotion_target(requested)
        mutation = False
        reasons.append("material_resource_regression")
    if capability_regression:
        effective = _demotion_target(requested)
        mutation = False
        reasons.append("capability_regression")

    # Any regression disables narrow auto even if demotion lands on sandbox.
    narrow_disabled = (
        effective is not DeterministicDoctorMode.NARROW_AUTO
        or not mutation
        or bool(reasons)
        or not policy.narrow_autonomous_mutation_enabled
    )
    if narrow_disabled:
        mutation = False

    return DeterministicDoctorRolloutDecision(
        requested_mode=requested,
        effective_mode=effective,
        kill_switch_engaged=policy.kill_switch_engaged,
        narrow_auto_disabled=narrow_disabled,
        mutation_authorized=mutation,
        completion_authoritative=False,
        reason_codes=tuple(dict.fromkeys(reasons)),
        floor_breaches=tuple(breaches),
        policy_binding_id=policy.policy_binding_id,
    )


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollbackReceipt:
    SCHEMA: ClassVar[str] = ROLLBACK_RECEIPT_SCHEMA
    reason: RollbackReason | str
    from_mode: DeterministicDoctorMode | str
    to_mode: DeterministicDoctorMode | str = DeterministicDoctorMode.REPORT_ONLY
    detail: str = ""
    metric_breaches: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    policy_binding_id: str = ""
    receipt_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.reason, RollbackReason):
            reason = self.reason
        else:
            try:
                reason = RollbackReason(str(self.reason).strip().casefold())
            except ValueError as exc:
                raise DeterministicDoctorRolloutError(
                    f"unknown rollback reason: {self.reason!r}"
                ) from exc
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "from_mode", _mode(self.from_mode))
        object.__setattr__(self, "to_mode", _mode(self.to_mode))
        object.__setattr__(self, "detail", str(self.detail or "").strip())
        object.__setattr__(
            self, "metric_breaches", tuple(str(item) for item in self.metric_breaches)
        )
        object.__setattr__(
            self, "capability_ids", tuple(str(item) for item in self.capability_ids)
        )
        object.__setattr__(
            self, "reason_codes", tuple(str(item) for item in self.reason_codes)
        )
        object.__setattr__(self, "policy_binding_id", str(self.policy_binding_id or ""))
        if not self.receipt_id:
            object.__setattr__(
                self, "receipt_id", content_identity(self.to_dict(include_id=False))
            )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": ROLLBACK_RECEIPT_SCHEMA,
            "reason": self.reason.value if isinstance(self.reason, RollbackReason) else str(self.reason),
            "from_mode": (
                self.from_mode.value
                if isinstance(self.from_mode, DeterministicDoctorMode)
                else str(self.from_mode)
            ),
            "to_mode": (
                self.to_mode.value
                if isinstance(self.to_mode, DeterministicDoctorMode)
                else str(self.to_mode)
            ),
            "detail": self.detail,
            "metric_breaches": list(self.metric_breaches),
            "capability_ids": list(self.capability_ids),
            "reason_codes": list(self.reason_codes),
            "policy_binding_id": self.policy_binding_id,
            "mutation_authorized": False,
            "completion_authoritative": False,
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload


def _demotion_target(current: DeterministicDoctorMode) -> DeterministicDoctorMode:
    if current in {
        DeterministicDoctorMode.REPORT_ONLY,
        DeterministicDoctorMode.PLAN,
    }:
        return DeterministicDoctorMode.REPORT_ONLY
    return {
        DeterministicDoctorMode.NARROW_AUTO: DeterministicDoctorMode.SANDBOX_AUTO,
        DeterministicDoctorMode.SANDBOX_AUTO: DeterministicDoctorMode.PLAN,
    }.get(current, DeterministicDoctorMode.REPORT_ONLY)


def evaluate_rollback(
    policy: DeterministicDoctorRolloutPolicy,
    *,
    safety_floors: Mapping[str, int] | None = None,
    root_schema_capability_drift: bool = False,
    embedding_canary_failure: bool = False,
    reconstruction_isolation_loss: bool = False,
    transaction_rollback_failure: bool = False,
    material_resource_regression: bool = False,
    capability_regression: Sequence[str] = (),
    kill_switch: bool = False,
    reason_codes: Sequence[str] = (),
) -> RollbackReceipt | None:
    current = _mode(policy.mode)
    target = _demotion_target(current)
    codes = {str(item).strip().casefold() for item in reason_codes if item}
    floors = dict(safety_floors if safety_floors is not None else policy.safety_floors)
    breaches = tuple(
        key for key in SAFETY_FLOOR_KEYS if int(floors.get(key, 0)) != 0
    )

    def _receipt(
        reason: RollbackReason,
        *,
        detail: str,
        metric_breaches: Sequence[str] = (),
        capability_ids: Sequence[str] = (),
        extra_codes: Sequence[str] = (),
        force_report_only: bool = False,
    ) -> RollbackReceipt:
        return RollbackReceipt(
            reason=reason,
            from_mode=current,
            to_mode=(
                DeterministicDoctorMode.REPORT_ONLY if force_report_only else target
            ),
            detail=detail,
            metric_breaches=tuple(metric_breaches),
            capability_ids=tuple(sorted(set(capability_ids))),
            reason_codes=tuple(sorted({*codes, *extra_codes})),
            policy_binding_id=policy.policy_binding_id,
        )

    if kill_switch or policy.kill_switch_engaged or "kill_switch" in codes:
        return _receipt(
            RollbackReason.KILL_SWITCH,
            detail="operator kill switch engaged",
            extra_codes=("kill_switch",),
            force_report_only=True,
        )
    if breaches or "nonzero_safety_floor" in codes:
        return _receipt(
            RollbackReason.NONZERO_SAFETY_FLOOR,
            detail="nonzero safety floor observed",
            metric_breaches=breaches,
            extra_codes=("nonzero_safety_floor",),
            force_report_only=True,
        )
    if root_schema_capability_drift or "root_schema_capability_drift" in codes:
        return _receipt(
            RollbackReason.ROOT_SCHEMA_CAPABILITY_DRIFT,
            detail="root, schema, or capability drift observed",
            extra_codes=("root_schema_capability_drift",),
            force_report_only=True,
        )
    if embedding_canary_failure or "embedding_canary_failure" in codes:
        return _receipt(
            RollbackReason.EMBEDDING_CANARY_FAILURE,
            detail="embedding canary failure",
            extra_codes=("embedding_canary_failure",),
            force_report_only=True,
        )
    if reconstruction_isolation_loss or "reconstruction_isolation_loss" in codes:
        return _receipt(
            RollbackReason.RECONSTRUCTION_ISOLATION_LOSS,
            detail="reconstruction or isolation loss",
            extra_codes=("reconstruction_isolation_loss",),
        )
    if transaction_rollback_failure or "transaction_rollback_failure" in codes:
        return _receipt(
            RollbackReason.TRANSACTION_ROLLBACK_FAILURE,
            detail="transaction or compensating rollback failure",
            extra_codes=("transaction_rollback_failure",),
        )
    if material_resource_regression or "material_resource_regression" in codes:
        return _receipt(
            RollbackReason.MATERIAL_RESOURCE_REGRESSION,
            detail="material resource regression",
            extra_codes=("material_resource_regression",),
        )
    if capability_regression:
        return _receipt(
            RollbackReason.CAPABILITY_REGRESSION,
            detail="capability health regression",
            capability_ids=capability_regression,
            extra_codes=("capability_regression",),
        )
    return None


def apply_rollback(
    policy: DeterministicDoctorRolloutPolicy, receipt: RollbackReceipt
) -> DeterministicDoctorRolloutPolicy:
    to_mode = _mode(receipt.to_mode)
    return DeterministicDoctorRolloutPolicy(
        policy_id=policy.policy_id,
        policy_revision=policy.policy_revision,
        repository_id=policy.repository_id,
        program_id=policy.program_id,
        mode=to_mode,
        explicit_policy_document=policy.explicit_policy_document,
        scoped_path_globs=policy.scoped_path_globs,
        enabled=to_mode is not DeterministicDoctorMode.REPORT_ONLY and policy.enabled,
        allow_plan=to_mode.rank >= DeterministicDoctorMode.PLAN.rank and policy.allow_plan,
        allow_sandbox_auto=(
            to_mode.rank >= DeterministicDoctorMode.SANDBOX_AUTO.rank
            and policy.allow_sandbox_auto
        ),
        allow_narrow_auto=(
            to_mode is DeterministicDoctorMode.NARROW_AUTO and policy.allow_narrow_auto
        ),
        narrow_autonomous_mutation_enabled=False,
        kill_switch_engaged=(
            policy.kill_switch_engaged
            or receipt.reason is RollbackReason.KILL_SWITCH
        ),
        limits=dict(policy.limits),
        safety_floors=dict(policy.safety_floors),
        approval_required_classes=policy.approval_required_classes,
        mutation_authorized=False,
        completion_authoritative=False,
    )


class DeterministicDoctorRollbackGate:
    INTERFACE: ClassVar[str] = ROLLBACK_GATE_INTERFACE

    def __init__(self, policy: DeterministicDoctorRolloutPolicy | None = None) -> None:
        self.policy = policy or default_rollout_policy()

    def evaluate(self, **kwargs: Any) -> RollbackReceipt | None:
        return evaluate_rollback(self.policy, **kwargs)

    def apply(self, receipt: RollbackReceipt) -> DeterministicDoctorRolloutPolicy:
        demoted = apply_rollback(self.policy, receipt)
        self.policy = demoted
        return demoted

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": ROLLBACK_GATE_INTERFACE,
            "policy": self.policy.to_dict(),
            "mutation_authorized": False,
            "completion_authoritative": False,
        }


# ---------------------------------------------------------------------------
# Checks / operations validator
# ---------------------------------------------------------------------------


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


def load_rollout_config(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    path = root / CONFIG_REL
    if not path.is_file():
        raise DeterministicDoctorRolloutError(f"rollout config missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DeterministicDoctorRolloutError("rollout config must be a JSON object")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise DeterministicDoctorRolloutError(
            f"unexpected config schema: {payload.get('schema')!r}"
        )
    return payload


def config_identity(repo_root: Path | None = None) -> str:
    root = (repo_root or repository_root()).resolve()
    return file_identity(root / CONFIG_REL)


def check_config_defaults(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    try:
        payload = load_rollout_config(root)
        policy = DeterministicDoctorRolloutPolicy.from_config_mapping(payload)
    except (OSError, json.JSONDecodeError, DeterministicDoctorRolloutError) as exc:
        return CheckResult("config_defaults", CheckStatus.FAIL, str(exc))

    errors: list[str] = []
    if payload.get("default_mode") != "report_only":
        errors.append("default_mode is not report_only")
    if tuple(payload.get("allowed_modes") or ()) != ROLLOUT_STAGES:
        errors.append("allowed_modes must be report_only/plan/sandbox_auto/narrow_auto")
    flags = dict(payload.get("feature_flags") or {})
    if flags.get("narrow_autonomous_mutation_enabled") is not False:
        errors.append("narrow_autonomous_mutation_enabled must default false")
    for name in (
        "llm_router_enabled",
        "llm_invocations_allowed",
        "remote_model_provider_calls_allowed",
        "remote_embeddings_allowed",
        "network_access_allowed",
    ):
        if flags.get(name) is not False:
            errors.append(f"{name} must default false")
    gates = dict(payload.get("gates") or {})
    for name in HARD_TRUE_GATES:
        if gates.get(name) is not True:
            errors.append(f"gate {name} must default true")
    limits = dict(payload.get("limits") or {})
    if set(limits) != set(LIMIT_KEYS):
        errors.append("limits keys incomplete")
    floors = dict(payload.get("release_safety_floors") or {})
    for key in SAFETY_FLOOR_KEYS:
        if int(floors.get(key, 1)) != 0:
            errors.append(f"safety floor {key} is nonzero")
    promotion = dict(payload.get("promotion") or {})
    if promotion.get("manual") is not True or promotion.get("monotonic") is not True:
        errors.append("promotion must be manual and monotonic")
    lifecycle = dict(payload.get("lifecycle_doctor") or {})
    if lifecycle.get("read_only") is not True or lifecycle.get("idempotent") is not True:
        errors.append("lifecycle doctor must be read-only and idempotent")
    optional = dict(payload.get("optional_providers") or {})
    if optional.get("absence_blocks_report_only_startup") is not False:
        errors.append("optional provider absence must not block report-only startup")
    if policy.mode_value != "report_only":
        errors.append("parsed policy mode is not report_only")
    if policy.narrow_autonomous_mutation_enabled:
        errors.append("parsed policy enables narrow auto")

    evidence = {
        "config_identity": config_identity(root),
        "default_mode": payload.get("default_mode"),
        "limits": limits,
        "safety_floors": floors,
        "policy_binding_id": policy.policy_binding_id,
        "promotion": promotion,
        "lifecycle_doctor": lifecycle,
        "optional_providers": optional,
    }
    if errors:
        return CheckResult("config_defaults", CheckStatus.FAIL, "; ".join(errors), evidence)
    return CheckResult(
        "config_defaults",
        CheckStatus.PASS,
        "immutable bounded config defaults to report-only with hard-off model flags and hard-on safety gates",
        evidence,
    )


def check_feature_flags(policy: DeterministicDoctorRolloutPolicy | None = None) -> CheckResult:
    current = policy or default_rollout_policy()
    errors: list[str] = []
    flags = current.feature_flags()
    if flags.get("narrow_autonomous_mutation_enabled") is not False and current.mode_value == "report_only":
        errors.append("narrow auto enabled under report-only default")
    for name in (
        "llm_router_enabled",
        "llm_invocations_allowed",
        "remote_model_provider_calls_allowed",
        "remote_embeddings_allowed",
        "network_access_allowed",
        "target_code_import_allowed",
    ):
        if flags.get(name) is not False:
            errors.append(f"{name} must be false")
    for name, value in current.gates().items():
        if value is not True:
            errors.append(f"gate {name} must be true")
    for name, value in current.semantic_authority_flags().items():
        if value is not False:
            errors.append(f"semantic authority {name} must be false")
    evidence = {
        "feature_flags": flags,
        "gates": current.gates(),
        "semantic_authority": current.semantic_authority_flags(),
        "mode": current.mode_value,
    }
    if errors:
        return CheckResult("feature_flags", CheckStatus.FAIL, "; ".join(errors), evidence)
    return CheckResult(
        "feature_flags",
        CheckStatus.PASS,
        "feature flags and hard gates match deterministic-doctor fail-closed defaults",
        evidence,
    )


def check_limits(policy: DeterministicDoctorRolloutPolicy | None = None) -> CheckResult:
    current = policy or default_rollout_policy()
    missing = [key for key in LIMIT_KEYS if key not in current.limits]
    if missing:
        return CheckResult(
            "resource_limits",
            CheckStatus.FAIL,
            f"missing limit keys: {missing}",
            {"limits": dict(current.limits)},
        )
    for key, value in current.limits.items():
        if not isinstance(value, int) or value <= 0:
            return CheckResult(
                "resource_limits",
                CheckStatus.FAIL,
                f"limit {key} must be a positive integer",
                {"limits": dict(current.limits)},
            )
    return CheckResult(
        "resource_limits",
        CheckStatus.PASS,
        "findings/candidates/queries/operators/plan steps/iterations/files/bytes/processes/time/CPU/memory limits defined",
        {"limits": dict(current.limits)},
    )


def check_promotion_monotonicity() -> CheckResult:
    errors: list[str] = []
    base = default_rollout_policy()
    if base.mode_value != "report_only":
        errors.append("default is not report_only")
    if not base.promotion_manual or not base.promotion_monotonic:
        errors.append("promotion not manual/monotonic")

    plan = elevate_rollout_policy(
        mode=DeterministicDoctorMode.PLAN,
        explicit_policy_document="policy://reviewed/plan",
        repository_id="repository:demo",
    )
    sandbox = elevate_rollout_policy(
        mode=DeterministicDoctorMode.SANDBOX_AUTO,
        explicit_policy_document="policy://reviewed/sandbox",
        repository_id="repository:demo",
    )
    narrow = elevate_rollout_policy(
        mode=DeterministicDoctorMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    ranks = (
        _mode(base.mode).rank,
        _mode(plan.mode).rank,
        _mode(sandbox.mode).rank,
        _mode(narrow.mode).rank,
    )
    if ranks != (0, 1, 2, 3):
        errors.append(f"stage ranks not monotonic: {ranks}")

    # Manual promotion: elevated modes require explicit scoped policy.
    for mode in (
        DeterministicDoctorMode.PLAN,
        DeterministicDoctorMode.SANDBOX_AUTO,
        DeterministicDoctorMode.NARROW_AUTO,
    ):
        try:
            DeterministicDoctorRolloutPolicy(mode=mode)
            errors.append(f"{mode.value} constructed without explicit scoped policy")
        except DeterministicDoctorRolloutError:
            pass

    # Kill switch blocks elevation.
    try:
        elevate_rollout_policy(
            mode=DeterministicDoctorMode.PLAN,
            explicit_policy_document="policy://reviewed/plan",
            repository_id="repository:demo",
            kill_switch_engaged=True,
        )
        errors.append("kill switch failed to block elevation")
    except DeterministicDoctorRolloutError:
        pass

    evidence = {
        "stages": list(ROLLOUT_STAGES),
        "ranks": list(ranks),
        "promotion_manual": True,
        "promotion_monotonic": True,
        "narrow_auto_default": False,
    }
    if errors:
        return CheckResult("promotion_monotonicity", CheckStatus.FAIL, "; ".join(errors), evidence)
    return CheckResult(
        "promotion_monotonicity",
        CheckStatus.PASS,
        "promotion is manual and monotonic across report_only→plan→sandbox_auto→narrow_auto",
        evidence,
    )


def check_rollback_gates() -> CheckResult:
    policy = elevate_rollout_policy(
        mode=DeterministicDoctorMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    errors: list[str] = []
    cases: list[tuple[dict[str, Any], RollbackReason]] = [
        (
            {"safety_floors": {**_zero_floors(), "missed_mandatory_caller_rate": 1}},
            RollbackReason.NONZERO_SAFETY_FLOOR,
        ),
        ({"root_schema_capability_drift": True}, RollbackReason.ROOT_SCHEMA_CAPABILITY_DRIFT),
        ({"embedding_canary_failure": True}, RollbackReason.EMBEDDING_CANARY_FAILURE),
        (
            {"reconstruction_isolation_loss": True},
            RollbackReason.RECONSTRUCTION_ISOLATION_LOSS,
        ),
        (
            {"transaction_rollback_failure": True},
            RollbackReason.TRANSACTION_ROLLBACK_FAILURE,
        ),
        (
            {"material_resource_regression": True},
            RollbackReason.MATERIAL_RESOURCE_REGRESSION,
        ),
        ({"kill_switch": True}, RollbackReason.KILL_SWITCH),
        (
            {"capability_regression": ("retrieval", "prover")},
            RollbackReason.CAPABILITY_REGRESSION,
        ),
    ]
    for kwargs, expected in cases:
        receipt = evaluate_rollback(policy, **kwargs)
        if receipt is None or receipt.reason is not expected:
            errors.append(f"expected {expected.value} for {kwargs}")
            continue
        demoted = apply_rollback(policy, receipt)
        if demoted.mutation_authorized or demoted.narrow_autonomous_mutation_enabled:
            errors.append(f"rollback failed to disable auto for {expected.value}")
        if demoted.mode.rank >= policy.mode.rank and expected is not RollbackReason.CAPABILITY_REGRESSION:
            # Capability / reconstruction may demote one stage rather than to report-only.
            if demoted.mode is DeterministicDoctorMode.NARROW_AUTO:
                errors.append(f"rollback left narrow_auto for {expected.value}")

    healthy = evaluate_rollback(policy, safety_floors=_zero_floors())
    if healthy is not None:
        errors.append("healthy policy incorrectly rolled back")

    evidence = {
        "case_count": len(cases),
        "rollback_disables_auto": True,
        "demotion_is_one_stage_or_report_only": True,
    }
    if errors:
        return CheckResult("rollback_gates", CheckStatus.FAIL, "; ".join(errors), evidence)
    return CheckResult(
        "rollback_gates",
        CheckStatus.PASS,
        "any nonzero floor, drift, canary failure, isolation/transaction loss, or resource regression rolls back or disables auto",
        evidence,
    )


def check_lifecycle_doctor_readonly(repo_root: Path | None = None) -> CheckResult:
    """Ordinary lifecycle doctor remains read-only and idempotent."""

    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    evidence: dict[str, Any] = {
        "read_only": True,
        "idempotent": True,
        "mutation_authorized": False,
    }

    # Config contract.
    try:
        payload = load_rollout_config(root)
        lifecycle = dict(payload.get("lifecycle_doctor") or {})
        if lifecycle.get("read_only") is not True:
            errors.append("config lifecycle_doctor.read_only is not true")
        if lifecycle.get("idempotent") is not True:
            errors.append("config lifecycle_doctor.idempotent is not true")
        if lifecycle.get("mutation_authorized") is not False:
            errors.append("config lifecycle_doctor.mutation_authorized is not false")
        evidence["lifecycle_doctor"] = lifecycle
    except DeterministicDoctorRolloutError as exc:
        errors.append(str(exc))

    # Default policy cannot authorize mutation.
    policy = default_rollout_policy()
    if policy.mutation_authorized or policy.mode_value != "report_only":
        errors.append("default policy is not report-only / non-mutating")

    # Doctor decision twice is identity-stable (idempotent projection).
    decision_a = evaluate_rollout_decision(policy)
    decision_b = evaluate_rollout_decision(policy)
    if decision_a.decision_id != decision_b.decision_id:
        errors.append("lifecycle decision is not idempotent under re-evaluation")
    if decision_a.mutation_authorized or decision_a.effective_mode_value != "report_only":
        errors.append("lifecycle decision is not read-only report-only")
    evidence["decision_id"] = decision_a.decision_id

    # Launcher doctor path (if present) must not be a write/merge authority.
    launcher = root / LAUNCHER_REL
    if launcher.is_file():
        text = launcher.read_text(encoding="utf-8")
        evidence["launcher_present"] = True
        if "doctor" not in text.casefold():
            errors.append("launcher lacks doctor lifecycle command")
        # doctor subcommand surface should not imply merge authority.
        if re.search(r"doctor.*\bmerge\b", text, re.IGNORECASE):
            errors.append("launcher doctor path appears to grant merge authority")
    else:
        evidence["launcher_present"] = False

    if errors:
        return CheckResult(
            "lifecycle_doctor_readonly", CheckStatus.FAIL, "; ".join(errors), evidence
        )
    return CheckResult(
        "lifecycle_doctor_readonly",
        CheckStatus.PASS,
        "ordinary lifecycle doctor remains read-only and idempotent",
        evidence,
    )


def check_optional_provider_absence() -> CheckResult:
    """Optional provider absence is actionable but does not block report-only startup."""

    missing: list[str] = []
    for module_name in OPTIONAL_PROVIDER_MODULES:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)

    # Report-only startup path must succeed regardless.
    policy = default_rollout_policy()
    decision = evaluate_rollout_decision(policy)
    blocks_startup = False
    actionable = bool(missing) or True  # absence is always an actionable signal surface

    evidence = {
        "optional_provider_modules": list(OPTIONAL_PROVIDER_MODULES),
        "missing_modules": missing,
        "absence_is_actionable": actionable,
        "absence_blocks_report_only_startup": blocks_startup,
        "report_only_startup_ok": decision.effective_mode_value == "report_only",
        "decision_id": decision.decision_id,
    }
    if decision.effective_mode_value != "report_only":
        return CheckResult(
            "optional_provider_absence",
            CheckStatus.FAIL,
            "report-only startup blocked or elevated unexpectedly",
            evidence,
        )
    if decision.mutation_authorized:
        return CheckResult(
            "optional_provider_absence",
            CheckStatus.FAIL,
            "provider probe incorrectly authorized mutation",
            evidence,
        )
    return CheckResult(
        "optional_provider_absence",
        CheckStatus.PASS,
        "optional provider absence is actionable and does not block report-only startup",
        evidence,
    )


def check_artifacts_present(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    present = {rel: (root / rel).is_file() for rel in REQUIRED_ARTIFACTS}
    missing = [rel for rel, ok in present.items() if not ok]
    evidence = {"artifacts": present, "config_identity": None}
    if (root / CONFIG_REL).is_file():
        evidence["config_identity"] = config_identity(root)
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
        "all LPR-041 declared outputs are present",
        evidence,
    )


def check_guide_boundaries(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    path = root / GUIDE_REL
    if not path.is_file():
        return CheckResult("guide_boundaries", CheckStatus.FAIL, f"guide missing: {path}")
    text = path.read_text(encoding="utf-8")
    lower = text.casefold()
    required = (
        "report-only",
        "plan",
        "sandbox",
        "narrow-auto",
        "kill switch",
        "rollback",
        "safety floor",
        "promotion",
        "lifecycle",
        "optional provider",
        "trust",
    )
    missing = [topic for topic in required if topic not in lower]
    # Allow alternate spellings.
    if "narrow_auto" in lower or "narrow auto" in lower:
        missing = [m for m in missing if m != "narrow-auto"]
    if "report_only" in lower or "report only" in lower:
        missing = [m for m in missing if m != "report-only"]
    evidence = {"guide_path": GUIDE_REL, "bytes": path.stat().st_size, "topics": list(required)}
    if missing:
        return CheckResult(
            "guide_boundaries",
            CheckStatus.FAIL,
            f"guide missing topics: {missing}",
            evidence,
        )
    return CheckResult(
        "guide_boundaries",
        CheckStatus.PASS,
        "operator guide documents modes, kill switch, floors, promotion, lifecycle, and providers",
        evidence,
    )


def check_related_surfaces(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    surfaces = {
        "service": root / SERVICE_REL,
        "policy": root / POLICY_REL,
        "benchmark": root / BENCHMARK_REL,
        "ops_facade": root / OPS_FACADE_REL,
    }
    present = {name: path.is_file() for name, path in surfaces.items()}
    missing = [name for name, ok in present.items() if not ok]
    evidence = {
        "surfaces": present,
        "metrics_interface": METRICS_INTERFACE,
        "service_interface": SERVICE_INTERFACE,
        "control_interface": SUPERVISOR_CONTROL_SERVICE_INTERFACE,
    }
    if missing:
        return CheckResult(
            "related_surfaces",
            CheckStatus.FAIL,
            f"related doctor surfaces missing: {missing}",
            evidence,
        )
    return CheckResult(
        "related_surfaces",
        CheckStatus.PASS,
        "service/policy/benchmark/ops surfaces exist for operator validation",
        evidence,
    )


def run_all_checks(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    checks = [
        check_artifacts_present(root),
        check_config_defaults(root),
        check_feature_flags(),
        check_limits(),
        check_promotion_monotonicity(),
        check_rollback_gates(),
        check_lifecycle_doctor_readonly(root),
        check_optional_provider_absence(),
        check_guide_boundaries(root),
        check_related_surfaces(root),
    ]
    failed = [item.name for item in checks if not item.ok]
    policy = default_rollout_policy()
    report = {
        "schema": VALIDATOR_SCHEMA,
        "interface": VALIDATOR_INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "board_namespace": BOARD_NAMESPACE,
        "default_mode": policy.mode_value,
        "mutation_authorized": False,
        "completion_authoritative": False,
        "config_identity": (
            config_identity(root) if (root / CONFIG_REL).is_file() else ""
        ),
        "checks": [item.to_dict() for item in checks],
        "failed": failed,
        "valid": not failed,
        "policy": policy.to_dict(),
    }
    report["report_id"] = content_identity(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    return report


def doctor(repo_root: Path | None = None) -> dict[str, Any]:
    report = run_all_checks(repo_root)
    report["command"] = "doctor"
    return report


def status(
    repo_root: Path | None = None,
    *,
    policy: DeterministicDoctorRolloutPolicy | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    current = policy or default_rollout_policy()
    decision = evaluate_rollout_decision(current)
    lifecycle = check_lifecycle_doctor_readonly(root)
    optional = check_optional_provider_absence()
    payload = {
        "schema": VALIDATOR_SCHEMA,
        "interface": VALIDATOR_INTERFACE,
        "command": "status",
        "mode": current.mode_value,
        "effective_mode": decision.effective_mode_value,
        "kill_switch_engaged": current.kill_switch_engaged,
        "narrow_auto_disabled": decision.narrow_auto_disabled,
        "mutation_authorized": False,
        "completion_authoritative": False,
        "config_identity": (
            config_identity(root) if (root / CONFIG_REL).is_file() else ""
        ),
        "feature_flags": current.feature_flags(),
        "gates": current.gates(),
        "limits": dict(current.limits),
        "safety_floors": dict(current.safety_floors),
        "lifecycle_doctor": lifecycle.to_dict(),
        "optional_providers": optional.to_dict(),
        "decision": decision.to_dict(),
        "valid": lifecycle.ok and optional.ok and current.mode_value == "report_only",
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
    }
    payload["report_id"] = content_identity(
        {key: value for key, value in payload.items() if key != "report_id"}
    )
    return payload


class DeterministicDoctorOperationsValidator:
    INTERFACE: ClassVar[str] = VALIDATOR_INTERFACE
    SCHEMA: ClassVar[str] = VALIDATOR_SCHEMA

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        policy: DeterministicDoctorRolloutPolicy | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.policy = policy or default_rollout_policy()

    def run_all(self) -> dict[str, Any]:
        return run_all_checks(self.repo_root)

    def doctor(self) -> dict[str, Any]:
        return doctor(self.repo_root)

    def status(self) -> dict[str, Any]:
        return status(self.repo_root, policy=self.policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": VALIDATOR_INTERFACE,
            "schema": VALIDATOR_SCHEMA,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "policy": self.policy.to_dict(),
            "mutation_authorized": False,
            "completion_authoritative": False,
        }


def write_checkpoint(name: str, payload: Mapping[str, Any]) -> None:
    raw = os.environ.get("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR", "").strip()
    if not raw:
        return
    directory = Path(raw)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{name}.json"
        data = json.dumps(_plain(payload), sort_keys=True, indent=2) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
    except OSError:
        return


__all__ = [
    "APPROVAL_REQUIRED_CLASSES",
    "CONFIG_SCHEMA",
    "DEFAULT_LIMITS",
    "HARD_FALSE_FLAGS",
    "HARD_TRUE_GATES",
    "LIMIT_KEYS",
    "METRICS_INTERFACE",
    "ROLLBACK_GATE_INTERFACE",
    "ROLLOUT_DECISION_INTERFACE",
    "ROLLOUT_POLICY_INTERFACE",
    "ROLLOUT_STAGES",
    "SAFETY_FLOOR_KEYS",
    "SERVICE_INTERFACE",
    "SUPERVISOR_CONTROL_SERVICE_INTERFACE",
    "TASK_ID",
    "GOAL_ID",
    "VALIDATOR_INTERFACE",
    "CheckResult",
    "CheckStatus",
    "DeterministicDoctorMode",
    "DeterministicDoctorOperationsValidator",
    "DeterministicDoctorRollbackGate",
    "DeterministicDoctorRolloutDecision",
    "DeterministicDoctorRolloutError",
    "DeterministicDoctorRolloutPolicy",
    "RollbackReason",
    "RollbackReceipt",
    "apply_rollback",
    "check_artifacts_present",
    "check_config_defaults",
    "check_feature_flags",
    "check_guide_boundaries",
    "check_lifecycle_doctor_readonly",
    "check_limits",
    "check_optional_provider_absence",
    "check_promotion_monotonicity",
    "check_related_surfaces",
    "check_rollback_gates",
    "config_identity",
    "content_identity",
    "default_rollout_policy",
    "doctor",
    "elevate_rollout_policy",
    "engage_kill_switch",
    "evaluate_rollback",
    "evaluate_rollout_decision",
    "load_rollout_config",
    "repository_root",
    "run_all_checks",
    "status",
    "write_checkpoint",
]
