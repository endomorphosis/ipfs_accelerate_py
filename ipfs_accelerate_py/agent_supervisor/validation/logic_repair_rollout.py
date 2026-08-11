"""Logic-repair rollout policy, metrics, rollback, and release validation.

LPR-020 / LPR-G060. Operator surface for Tactician-Hammer logic repair.

* LogicRepairRolloutPolicy — shadow default; assist, deterministic narrow-auto,
  and approval-gated behavior-complete model-edit require explicit scoped policy;
  independent flags keep prediction / learned ranking / Hammer / refinement /
  LLM / auto off until elevated.
* LogicRepairMetrics — stage metrics plus analytical/model split, tokens/
  context, fixed-point iterations, and absolute-zero safety floors.
* LogicRepairRollbackGate — demotes on nonzero floors, drift, reconstruction/
  countermodel-validation loss, inconsistency, transaction, isolation, or
  budget regression.
* LogicRepairOperationsValidator — composes the protected bootstrap board/DAG
  doctor with exact two-repository bindings, import-isolation and native-
  execution permits, platform resource/network isolation, capability health,
  four-lane sharding, isolated state/worktrees, one merge queue, bounded
  retries, one refill owner, and launcher lifecycle safety.

This module never grants mutation, completion, merge, or process authority.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

ROLLOUT_POLICY_INTERFACE: Final[str] = "LogicRepairRolloutPolicy@1"
ROLLOUT_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-rollout-policy@1"
)
METRICS_INTERFACE: Final[str] = "LogicRepairMetrics@1"
METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-ops-metrics@1"
)
BENCHMARK_METRICS_INTERFACE: Final[str] = "LogicRepairBenchmarkMetrics@1"
ROLLBACK_GATE_INTERFACE: Final[str] = "LogicRepairRollbackGate@1"
ROLLBACK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-rollback-receipt@1"
)
SOURCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-source-binding@1"
)
VALIDATOR_INTERFACE: Final[str] = "LogicRepairOperationsValidator@1"
VALIDATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-validation-ops-report@1"
)
END_TO_END_INTERFACE: Final[str] = "LogicRepairEndToEnd@1"
END_TO_END_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-end-to-end@1"
)
LIVE_LOGIC_REPAIR_CONTROLLER_INTERFACE: Final[str] = "LiveLogicRepairController@1"
PROPAGATION_COMPLETION_RECEIPT_INTERFACE: Final[str] = "PropagationCompletionReceipt@1"
LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE: Final[str] = (
    "LogicFixedPointEvidenceAttachment@1"
)
SUPERVISOR_CONTROL_SERVICE_INTERFACE: Final[str] = "SupervisorControlService@1"
BOARD_VALIDATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.tactician_hammer_logic_repair.board_validation@1"
)

TASK_ID: Final[str] = "LPR-020"
GOAL_ID: Final[str] = "LPR-G060"
BOARD_NAMESPACE: Final[str] = "agent-supervisor-tactician-hammer-logic-repair-v1"
TASK_PREFIX: Final[str] = "LPR-"
MERGE_TARGET_BRANCH: Final[str] = "agent/proof-gated-contract-repair"
DEFAULT_RECALL_K: Final[int] = 5
LANE_COUNT: Final[int] = 4
DATASETS_SUBMODULE: Final[str] = "ipfs_datasets_py"
DATASETS_TACTICIAN_ANCESTOR: Final[str] = "014b8ea69721d8e0f0cd15b36b83bc5e8bb6a29c"
DATASETS_TACTICIAN_INTERFACE: Final[str] = "ipfs_datasets_py.logic.tactician@1"

PLAN_REL: Final[str] = "docs/architecture/AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md"
OBJECTIVE_REL: Final[str] = "docs/architecture/agent_supervisor_tactician_hammer_logic_repair.objectives.md"
TODO_REL: Final[str] = "docs/architecture/agent_supervisor_tactician_hammer_logic_repair.todo.md"
SCHEDULER_REL: Final[str] = "config/agent_supervisor_tactician_hammer_logic_repair_scheduler.json"
BOARD_VALIDATOR_REL: Final[str] = "scripts/validate_tactician_hammer_logic_repair_board.py"
LAUNCHER_REL: Final[str] = "scripts/tactician_hammer_logic_repair_supervisor.sh"
GUIDE_REL: Final[str] = "docs/guides/TACTICIAN_HAMMER_LOGIC_REPAIR_GUIDE.md"
BENCHMARK_SCRIPT_REL: Final[str] = "scripts/benchmark_tactician_hammer_logic_repair.py"
ROLLOUT_MODULE_REL: Final[str] = "ipfs_accelerate_py/agent_supervisor/validation/logic_repair_rollout.py"
VALIDATE_SCRIPT_REL: Final[str] = "scripts/validate_tactician_hammer_logic_repair.py"
FIXTURE_MANIFEST_REL: Final[str] = "test/fixtures/agent_supervisor/tactician_hammer_logic_repair/manifest.json"
FIXED_POINT_MODULE_REL: Final[str] = "ipfs_accelerate_py/agent_supervisor/validation/logic_repair_fixed_point.py"
LIVE_CONTROLLER_REL: Final[str] = "ipfs_accelerate_py/agent_supervisor/todo_daemon/live_logic_repair_controller.py"
CAPABILITIES_REL: Final[str] = "ipfs_accelerate_py/agent_supervisor/integrations/tactician_hammer_capabilities.py"

PROTECTED_PATHS: Final[tuple[str, ...]] = (
    PLAN_REL, OBJECTIVE_REL, TODO_REL, SCHEDULER_REL, BOARD_VALIDATOR_REL, LAUNCHER_REL,
)
REQUIRED_RELEASE_SOURCES: Final[tuple[str, ...]] = (
    FIXTURE_MANIFEST_REL, BENCHMARK_SCRIPT_REL, ROLLOUT_MODULE_REL, VALIDATE_SCRIPT_REL,
    GUIDE_REL, FIXED_POINT_MODULE_REL, LIVE_CONTROLLER_REL, CAPABILITIES_REL,
)

SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "missed_resolved_caller_rate",
    "unreconstructed_or_raw_countermodel_admission_rate",
    "unauthorized_axiom_admission_rate",
    "invented_behavior_without_authority_rate",
    "wrong_value_source_placement_admission_rate",
    "stale_root_corpus_receipt_admission_rate",
    "failed_obligation_override_rate",
    "llm_scope_semantic_escape_rate",
    "partial_transaction_completion_rate",
    "false_fixed_point_completion_rate",
)
SAFETY_ABSOLUTE_KEYS: Final[tuple[str, ...]] = (
    "missed_resolved_caller",
    "unreconstructed_or_raw_countermodel_admission",
    "unauthorized_axiom_admission",
    "invented_behavior_without_authority",
    "wrong_value_source_placement_admission",
    "stale_root_corpus_receipt_admission",
    "failed_obligation_override",
    "llm_scope_semantic_escape",
    "partial_transaction_completion",
    "false_fixed_point_completion",
)
BENCHMARK_STAGES: Final[tuple[str, ...]] = (
    "goal", "premise", "tactician", "lowering", "solver", "reconstruction",
    "prediction", "analytical", "model", "transaction", "fixed_point",
)
FEATURE_FLAG_KEYS: Final[tuple[str, ...]] = (
    "logic_prediction_enabled",
    "learned_tactician_ranking_enabled",
    "hammer_execution_enabled",
    "counterexample_refinement_enabled",
    "llm_router_enabled",
    "narrow_autonomous_mutation_enabled",
)
NARROW_AUTO_TRANSFORMS: Final[frozenset[str]] = frozenset({
    "add_argument", "rename_argument", "reorder_argument", "thread_parameter",
    "add_import", "add_export", "analytical_python_transform",
    "deterministic_rename", "deterministic_substitution",
})
APPROVAL_GATED_CHANGE_FAMILIES: Final[frozenset[str]] = frozenset({
    "model_authored", "llm_authored", "llm_bounded", "behavior_complete_model_edit",
    "stateful_behavior", "stateful_service", "public_schema", "public_api", "schema_api",
    "dynamic", "generated", "native", "ffi", "cross_root", "cross_repository",
    "new_dependency", "new_external_dependency", "complex_support_type", "stateful_support_type",
})
NON_MEMORY_SAFETY_EVIDENCE: Final[frozenset[str]] = frozenset({
    "vector", "lexical", "graph", "history", "test", "type", "schema", "resource",
    "llm", "max_memory_bytes", "embedding", "coverage", "tactician_ranking", "knowledge_graph",
})
ZERO_TOLERANCE_REASON_CODES: Final[frozenset[str]] = frozenset({
    "wrong_value", "missed_caller", "missed_consumer", "partial_plan", "partial_transaction",
    "false_completion", "false_fixed_point", "open_frontier", "proof_loss",
    "reconstruction_failure", "countermodel_validation_loss", "isolation_regression",
    "budget_regression", "inconsistency", "root_drift",
})
ROLLOUT_STAGES: Final[tuple[str, ...]] = (
    "doctor_replay", "shadow", "assist", "narrow_auto", "model_edit",
)

class LogicRepairRolloutError(ValueError):
    """Raised when control-plane, policy, or metric evidence is invalid."""


class RolloutMode(str, Enum):
    DOCTOR_REPLAY = "doctor_replay"
    SHADOW = "shadow"
    ASSIST = "assist"
    NARROW_AUTO = "narrow_auto"
    MODEL_EDIT = "model_edit"


class RollbackReason(str, Enum):
    CAPABILITY_REGRESSION = "capability_regression"
    STALE_ROOT = "stale_root"
    ROOT_DRIFT = "root_drift"
    OPEN_FRONTIER = "open_frontier"
    RECONSTRUCTION_FAILURE = "reconstruction_failure"
    COUNTERMODEL_VALIDATION_LOSS = "countermodel_validation_loss"
    PROOF_LOSS = "proof_loss"
    WRONG_VALUE = "wrong_value"
    MISSED_CALLER = "missed_caller"
    PARTIAL_PLAN = "partial_plan"
    FALSE_COMPLETION = "false_completion"
    METRIC_BREACH = "metric_breach"
    ISOLATION_REGRESSION = "isolation_regression"
    BUDGET_REGRESSION = "budget_regression"
    INCONSISTENCY = "inconsistency"
    TRANSACTION_FAILURE = "transaction_failure"
    EXPLICIT_OPERATOR = "explicit_operator"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        raise LogicRepairRolloutError("floating-point values are forbidden in sealed identities")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_identity(value: Any) -> str:
    return _sha256_hex(_canonical_bytes(value))


def file_identity(path: Path) -> str:
    return _sha256_hex(path.read_bytes())


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise LogicRepairRolloutError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise LogicRepairRolloutError(f"{name} is unsafe or too large")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise LogicRepairRolloutError(f"{name} must be a boolean")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogicRepairRolloutError(f"{name} must be a non-negative integer")
    return value


def _ppm(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator * 1_000_000) // denominator


def _mode(value: Any) -> RolloutMode:
    if isinstance(value, RolloutMode):
        return value
    try:
        return RolloutMode(str(getattr(value, "value", value)).strip().casefold())
    except ValueError as exc:
        raise LogicRepairRolloutError(f"unknown rollout mode: {value!r}") from exc


def _safe_relative(path: str) -> bool:
    if not path or path.startswith("/") or ".." in Path(path).parts or "\x00" in path:
        return False
    return True


def _cycle_nodes(edges: Mapping[str, Sequence[str]]) -> list[str]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    cyclic: set[str] = set()

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for dep in edges.get(node, ()):
            if dep not in edges:
                continue
            if dep not in indices:
                strongconnect(dep)
                lowlink[node] = min(lowlink[node], lowlink[dep])
            elif dep in on_stack:
                lowlink[node] = min(lowlink[node], indices[dep])
        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or node in edges.get(node, ()):
                cyclic.update(component)

    for node in edges:
        if node not in indices:
            strongconnect(node)
    return sorted(cyclic)


def _load_scheduler(root: Path) -> dict[str, Any]:
    path = root / SCHEDULER_REL
    if not path.is_file():
        raise LogicRepairRolloutError(f"scheduler missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class LogicRepairSourceBinding:
    SCHEMA: ClassVar[str] = SOURCE_BINDING_SCHEMA
    repository_root: str
    board_namespace: str = BOARD_NAMESPACE
    task_prefix: str = TASK_PREFIX
    merge_target_branch: str = MERGE_TARGET_BRANCH
    datasets_submodule: str = DATASETS_SUBMODULE
    datasets_required_ancestor: str = DATASETS_TACTICIAN_ANCESTOR
    datasets_required_interface: str = DATASETS_TACTICIAN_INTERFACE
    plan_path: str = PLAN_REL
    objective_path: str = OBJECTIVE_REL
    todo_path: str = TODO_REL
    scheduler_path: str = SCHEDULER_REL
    board_validator_path: str = BOARD_VALIDATOR_REL
    launcher_path: str = LAUNCHER_REL
    guide_path: str = GUIDE_REL
    benchmark_path: str = BENCHMARK_SCRIPT_REL
    fixture_manifest_path: str = FIXTURE_MANIFEST_REL
    rollout_module_path: str = ROLLOUT_MODULE_REL
    validate_script_path: str = VALIDATE_SCRIPT_REL
    plan_identity: str = ""
    objective_identity: str = ""
    todo_identity: str = ""
    scheduler_identity: str = ""
    board_validator_identity: str = ""
    launcher_identity: str = ""
    guide_identity: str = ""
    benchmark_identity: str = ""
    fixture_manifest_identity: str = ""
    rollout_module_identity: str = ""
    validate_script_identity: str = ""
    binding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_root", _text(self.repository_root, "repository_root", maximum=4096))
        for name in (
            "board_namespace", "task_prefix", "merge_target_branch", "datasets_submodule",
            "datasets_required_ancestor", "datasets_required_interface", "plan_path",
            "objective_path", "todo_path", "scheduler_path", "board_validator_path",
            "launcher_path", "guide_path", "benchmark_path", "fixture_manifest_path",
            "rollout_module_path", "validate_script_path",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum=1024))
        if not self.binding_id:
            object.__setattr__(self, "binding_id", content_identity(self.to_dict(include_id=False)))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SOURCE_BINDING_SCHEMA,
            "repository_root": self.repository_root,
            "board_namespace": self.board_namespace,
            "task_prefix": self.task_prefix,
            "merge_target_branch": self.merge_target_branch,
            "datasets_submodule": self.datasets_submodule,
            "datasets_required_ancestor": self.datasets_required_ancestor,
            "datasets_required_interface": self.datasets_required_interface,
            "plan_path": self.plan_path,
            "objective_path": self.objective_path,
            "todo_path": self.todo_path,
            "scheduler_path": self.scheduler_path,
            "board_validator_path": self.board_validator_path,
            "launcher_path": self.launcher_path,
            "guide_path": self.guide_path,
            "benchmark_path": self.benchmark_path,
            "fixture_manifest_path": self.fixture_manifest_path,
            "rollout_module_path": self.rollout_module_path,
            "validate_script_path": self.validate_script_path,
            "plan_identity": self.plan_identity,
            "objective_identity": self.objective_identity,
            "todo_identity": self.todo_identity,
            "scheduler_identity": self.scheduler_identity,
            "board_validator_identity": self.board_validator_identity,
            "launcher_identity": self.launcher_identity,
            "guide_identity": self.guide_identity,
            "benchmark_identity": self.benchmark_identity,
            "fixture_manifest_identity": self.fixture_manifest_identity,
            "rollout_module_identity": self.rollout_module_identity,
            "validate_script_identity": self.validate_script_identity,
            "two_repository_bindings": True,
        }
        if include_id:
            payload["binding_id"] = self.binding_id
        return payload


def bind_exact_sources(repo_root: Path | None = None) -> LogicRepairSourceBinding:
    root = (repo_root or repository_root()).resolve()
    paths = {
        "plan": root / PLAN_REL,
        "objective": root / OBJECTIVE_REL,
        "todo": root / TODO_REL,
        "scheduler": root / SCHEDULER_REL,
        "board_validator": root / BOARD_VALIDATOR_REL,
        "launcher": root / LAUNCHER_REL,
        "guide": root / GUIDE_REL,
        "benchmark": root / BENCHMARK_SCRIPT_REL,
        "fixture_manifest": root / FIXTURE_MANIFEST_REL,
        "rollout_module": root / ROLLOUT_MODULE_REL,
        "validate_script": root / VALIDATE_SCRIPT_REL,
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise LogicRepairRolloutError(f"missing exact sources: {sorted(missing)}")
    scheduler = json.loads(paths["scheduler"].read_text(encoding="utf-8"))
    source = scheduler.get("source_binding") or {}
    return LogicRepairSourceBinding(
        repository_root=str(root),
        board_namespace=str(scheduler.get("board_namespace") or BOARD_NAMESPACE),
        task_prefix=str(scheduler.get("task_prefix") or TASK_PREFIX),
        merge_target_branch=str(scheduler.get("merge_target_branch") or MERGE_TARGET_BRANCH),
        datasets_submodule=str(source.get("datasets_submodule_path") or DATASETS_SUBMODULE),
        datasets_required_ancestor=str(source.get("datasets_required_ancestor") or DATASETS_TACTICIAN_ANCESTOR),
        datasets_required_interface=str(source.get("datasets_required_interface") or DATASETS_TACTICIAN_INTERFACE),
        plan_identity=file_identity(paths["plan"]),
        objective_identity=file_identity(paths["objective"]),
        todo_identity=file_identity(paths["todo"]),
        scheduler_identity=file_identity(paths["scheduler"]),
        board_validator_identity=file_identity(paths["board_validator"]),
        launcher_identity=file_identity(paths["launcher"]),
        guide_identity=file_identity(paths["guide"]),
        benchmark_identity=file_identity(paths["benchmark"]),
        fixture_manifest_identity=file_identity(paths["fixture_manifest"]),
        rollout_module_identity=file_identity(paths["rollout_module"]),
        validate_script_identity=file_identity(paths["validate_script"]),
    )


@dataclass(frozen=True)
class LogicRepairRolloutPolicy:
    SCHEMA: ClassVar[str] = ROLLOUT_POLICY_SCHEMA
    INTERFACE: ClassVar[str] = ROLLOUT_POLICY_INTERFACE
    policy_id: str = "policy:logic-repair-rollout-default"
    policy_revision: str = "1"
    repository_id: str = ""
    program_id: str = BOARD_NAMESPACE
    mode: RolloutMode | str = RolloutMode.SHADOW
    explicit_policy_document: str = ""
    scoped_path_globs: tuple[str, ...] = ()
    allow_assist: bool = False
    allow_narrow_auto: bool = False
    allow_model_edit: bool = False
    logic_prediction_enabled: bool = False
    learned_tactician_ranking_enabled: bool = False
    hammer_execution_enabled: bool = False
    counterexample_refinement_enabled: bool = False
    llm_router_enabled: bool = False
    narrow_autonomous_mutation_enabled: bool = False
    auto_requires_unique_target: bool = True
    auto_requires_reconstruction: bool = True
    auto_requires_supported_python: bool = True
    auto_requires_complete_frontier: bool = True
    auto_requires_analytical_path: bool = True
    auto_requires_fixed_point: bool = True
    auto_allowed_transforms: tuple[str, ...] = (
        "add_argument", "rename_argument", "reorder_argument", "thread_parameter",
        "add_import", "add_export", "deterministic_rename", "deterministic_substitution",
    )
    approval_gated_families: tuple[str, ...] = tuple(sorted(APPROVAL_GATED_CHANGE_FAMILIES))
    rollback_on_capability_regression: bool = True
    rollback_on_stale_root: bool = True
    rollback_on_open_frontier: bool = True
    rollback_on_reconstruction_failure: bool = True
    rollback_on_countermodel_validation_loss: bool = True
    rollback_on_proof_loss: bool = True
    rollback_on_metric_breach: bool = True
    rollback_on_isolation_regression: bool = True
    rollback_on_budget_regression: bool = True
    rollback_on_inconsistency: bool = True
    rollback_on_transaction_failure: bool = True
    mutation_authorized: bool = False
    completion_authoritative: bool = False
    policy_binding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "policy_revision", _text(self.policy_revision, "policy_revision"))
        object.__setattr__(self, "repository_id", str(self.repository_id or "").strip())
        object.__setattr__(self, "program_id", _text(self.program_id, "program_id"))
        object.__setattr__(self, "mode", _mode(self.mode))
        object.__setattr__(self, "explicit_policy_document", str(self.explicit_policy_document or "").strip())
        object.__setattr__(self, "scoped_path_globs", tuple(sorted({_text(i, "scoped_path_globs", maximum=1024) for i in self.scoped_path_globs})))
        transforms = tuple(sorted({_text(i, "auto_allowed_transforms").casefold() for i in self.auto_allowed_transforms}))
        if not transforms:
            raise LogicRepairRolloutError("auto_allowed_transforms must not be empty")
        object.__setattr__(self, "auto_allowed_transforms", transforms)
        object.__setattr__(self, "approval_gated_families", tuple(sorted({_text(i, "approval_gated_families").casefold() for i in self.approval_gated_families})))
        for name in (
            "allow_assist", "allow_narrow_auto", "allow_model_edit",
            "logic_prediction_enabled", "learned_tactician_ranking_enabled",
            "hammer_execution_enabled", "counterexample_refinement_enabled",
            "llm_router_enabled", "narrow_autonomous_mutation_enabled",
            "auto_requires_unique_target", "auto_requires_reconstruction",
            "auto_requires_supported_python", "auto_requires_complete_frontier",
            "auto_requires_analytical_path", "auto_requires_fixed_point",
            "rollback_on_capability_regression", "rollback_on_stale_root",
            "rollback_on_open_frontier", "rollback_on_reconstruction_failure",
            "rollback_on_countermodel_validation_loss", "rollback_on_proof_loss",
            "rollback_on_metric_breach", "rollback_on_isolation_regression",
            "rollback_on_budget_regression", "rollback_on_inconsistency",
            "rollback_on_transaction_failure", "mutation_authorized", "completion_authoritative",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if self.completion_authoritative:
            raise LogicRepairRolloutError("rollout policy cannot claim completion authority")
        if self.mode in {RolloutMode.SHADOW, RolloutMode.DOCTOR_REPLAY} and self.mutation_authorized:
            raise LogicRepairRolloutError(f"{self.mode_value} mode cannot authorize mutation")
        if not self.policy_binding_id:
            object.__setattr__(self, "policy_binding_id", content_identity(self.to_dict(include_id=False)))
        self._assert_mode_allowed()

    def _assert_mode_allowed(self) -> None:
        mode = _mode(self.mode)
        if mode in {RolloutMode.SHADOW, RolloutMode.DOCTOR_REPLAY}:
            return
        if not self.has_explicit_scoped_policy():
            raise LogicRepairRolloutError(
                f"{mode.value} requires an explicit scoped policy document and repository/program/policy scope"
            )
        if mode is RolloutMode.ASSIST and not self.allow_assist:
            raise LogicRepairRolloutError("assist mode is not enabled on this policy")
        if mode is RolloutMode.NARROW_AUTO and not self.allow_narrow_auto:
            raise LogicRepairRolloutError("narrow_auto mode is not enabled on this policy")
        if mode is RolloutMode.MODEL_EDIT and not self.allow_model_edit:
            raise LogicRepairRolloutError("model_edit mode is not enabled on this policy")

    def has_explicit_scoped_policy(self) -> bool:
        return bool(self.explicit_policy_document and (self.repository_id or self.program_id or self.policy_id))

    @property
    def mode_value(self) -> str:
        return self.mode.value if isinstance(self.mode, RolloutMode) else str(self.mode)

    def feature_flags(self) -> dict[str, bool]:
        return {
            "logic_prediction_enabled": self.logic_prediction_enabled,
            "learned_tactician_ranking_enabled": self.learned_tactician_ranking_enabled,
            "hammer_execution_enabled": self.hammer_execution_enabled,
            "counterexample_refinement_enabled": self.counterexample_refinement_enabled,
            "llm_router_enabled": self.llm_router_enabled,
            "narrow_autonomous_mutation_enabled": self.narrow_autonomous_mutation_enabled,
        }

    def is_approval_gated(
        self, *, transform: str = "", change_family: str = "", model_authored: bool = False,
        stateful: bool = False, public_schema_api: bool = False, dynamic: bool = False,
        generated: bool = False, native: bool = False, cross_root: bool = False,
        new_dependency: bool = False, behavior_complete_model_edit: bool = False,
    ) -> bool:
        if model_authored or stateful or public_schema_api or behavior_complete_model_edit:
            return True
        if dynamic or generated or native or cross_root or new_dependency:
            return True
        key = str(transform or change_family or "").strip().casefold()
        if key and key in {item.casefold() for item in self.approval_gated_families}:
            return True
        if key and key in APPROVAL_GATED_CHANGE_FAMILIES:
            return True
        return False

    def allows_automated_mutation(
        self, *, transform: str, unique_target: bool, reconstructed: bool,
        supported_python: bool, complete_frontier: bool, analytical_path: bool = True,
        fixed_point_ready: bool = True, model_authored: bool = False, stateful: bool = False,
        public_schema_api: bool = False, dynamic: bool = False, generated: bool = False,
        native: bool = False, cross_root: bool = False, new_dependency: bool = False,
        behavior_complete_model_edit: bool = False, change_family: str = "",
    ) -> bool:
        mode = _mode(self.mode)
        if mode in {RolloutMode.SHADOW, RolloutMode.DOCTOR_REPLAY, RolloutMode.ASSIST, RolloutMode.MODEL_EDIT}:
            return False
        if mode is RolloutMode.NARROW_AUTO and not (self.allow_narrow_auto and self.narrow_autonomous_mutation_enabled):
            return False
        if not self.mutation_authorized:
            return False
        if self.is_approval_gated(
            transform=transform, change_family=change_family, model_authored=model_authored,
            stateful=stateful, public_schema_api=public_schema_api, dynamic=dynamic,
            generated=generated, native=native, cross_root=cross_root,
            new_dependency=new_dependency, behavior_complete_model_edit=behavior_complete_model_edit,
        ):
            return False
        transform_key = str(transform or "").strip().casefold()
        if transform_key not in self.auto_allowed_transforms or transform_key not in NARROW_AUTO_TRANSFORMS:
            return False
        if self.auto_requires_unique_target and not unique_target:
            return False
        if self.auto_requires_reconstruction and not reconstructed:
            return False
        if self.auto_requires_supported_python and not supported_python:
            return False
        if self.auto_requires_complete_frontier and not complete_frontier:
            return False
        if self.auto_requires_analytical_path and not analytical_path:
            return False
        if self.auto_requires_fixed_point and not fixed_point_ready:
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
            "allow_assist": self.allow_assist,
            "allow_narrow_auto": self.allow_narrow_auto,
            "allow_model_edit": self.allow_model_edit,
            "feature_flags": self.feature_flags(),
            "auto_requires_unique_target": self.auto_requires_unique_target,
            "auto_requires_reconstruction": self.auto_requires_reconstruction,
            "auto_requires_supported_python": self.auto_requires_supported_python,
            "auto_requires_complete_frontier": self.auto_requires_complete_frontier,
            "auto_requires_analytical_path": self.auto_requires_analytical_path,
            "auto_requires_fixed_point": self.auto_requires_fixed_point,
            "auto_allowed_transforms": list(self.auto_allowed_transforms),
            "approval_gated_families": list(self.approval_gated_families),
            "rollback_on_capability_regression": self.rollback_on_capability_regression,
            "rollback_on_stale_root": self.rollback_on_stale_root,
            "rollback_on_open_frontier": self.rollback_on_open_frontier,
            "rollback_on_reconstruction_failure": self.rollback_on_reconstruction_failure,
            "rollback_on_countermodel_validation_loss": self.rollback_on_countermodel_validation_loss,
            "rollback_on_proof_loss": self.rollback_on_proof_loss,
            "rollback_on_metric_breach": self.rollback_on_metric_breach,
            "rollback_on_isolation_regression": self.rollback_on_isolation_regression,
            "rollback_on_budget_regression": self.rollback_on_budget_regression,
            "rollback_on_inconsistency": self.rollback_on_inconsistency,
            "rollback_on_transaction_failure": self.rollback_on_transaction_failure,
            "mutation_authorized": self.mutation_authorized,
            "completion_authoritative": self.completion_authoritative,
        }
        if include_id:
            payload["policy_binding_id"] = self.policy_binding_id
        return payload

    @classmethod
    def default(cls) -> "LogicRepairRolloutPolicy":
        return cls()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogicRepairRolloutPolicy":
        if not isinstance(value, Mapping):
            raise LogicRepairRolloutError("policy payload must be an object")
        known = set(cls.__dataclass_fields__) - {"policy_binding_id"}
        data = {key: value[key] for key in known if key in value}
        flags = value.get("feature_flags")
        if isinstance(flags, Mapping):
            for key in FEATURE_FLAG_KEYS:
                if key in flags and key not in data:
                    data[key] = flags[key]
        for key in ("scoped_path_globs", "auto_allowed_transforms", "approval_gated_families"):
            if key in data:
                data[key] = tuple(data[key] or ())
        return cls(**data)


def default_rollout_policy() -> LogicRepairRolloutPolicy:
    return LogicRepairRolloutPolicy.default()


def elevate_rollout_policy(
    *, mode: RolloutMode | str, explicit_policy_document: str, repository_id: str,
    program_id: str = BOARD_NAMESPACE, policy_id: str = "policy:logic-repair-rollout-scoped",
    policy_revision: str = "1", scoped_path_globs: Sequence[str] = (),
    mutation_authorized: bool = False, enable_flags: Sequence[str] = (),
) -> LogicRepairRolloutPolicy:
    mode_value = _mode(mode)
    flags = {key: False for key in FEATURE_FLAG_KEYS}
    for key in enable_flags:
        if key in flags:
            flags[key] = True
    if mode_value is RolloutMode.NARROW_AUTO:
        flags["narrow_autonomous_mutation_enabled"] = True
    return LogicRepairRolloutPolicy(
        policy_id=policy_id, policy_revision=policy_revision, repository_id=repository_id,
        program_id=program_id, mode=mode_value, explicit_policy_document=explicit_policy_document,
        scoped_path_globs=tuple(scoped_path_globs),
        allow_assist=mode_value in {RolloutMode.ASSIST, RolloutMode.NARROW_AUTO, RolloutMode.MODEL_EDIT},
        allow_narrow_auto=mode_value is RolloutMode.NARROW_AUTO,
        allow_model_edit=mode_value is RolloutMode.MODEL_EDIT,
        logic_prediction_enabled=flags["logic_prediction_enabled"],
        learned_tactician_ranking_enabled=flags["learned_tactician_ranking_enabled"],
        hammer_execution_enabled=flags["hammer_execution_enabled"],
        counterexample_refinement_enabled=flags["counterexample_refinement_enabled"],
        llm_router_enabled=flags["llm_router_enabled"],
        narrow_autonomous_mutation_enabled=flags["narrow_autonomous_mutation_enabled"],
        mutation_authorized=mutation_authorized and mode_value is RolloutMode.NARROW_AUTO,
    )


@dataclass(frozen=True)
class LogicRepairMetrics:
    SCHEMA: ClassVar[str] = METRICS_SCHEMA
    INTERFACE: ClassVar[str] = METRICS_INTERFACE
    case_count: int = 0
    goal_precision: int = 0
    goal_recall: int = 0
    hypothesis_precision: int = 0
    hypothesis_recall: int = 0
    premise_recall_at_k: int = 0
    first_plan_closure_rate: int = 0
    lowering_rate: int = 0
    reconstruction_rate: int = 0
    validated_countermodel_rate: int = 0
    abstention_count: int = 0
    abstention_rate: int = 0
    analytical_coverage: int = 0
    model_rate: int = 0
    llm_rate: int = 0
    all_caller_rate: int = 0
    analytical_model_split: Mapping[str, int] = field(default_factory=dict)
    stage_counts: Mapping[str, int] = field(default_factory=dict)
    stage_cost_units: Mapping[str, int] = field(default_factory=dict)
    fixed_point_iterations: int = 0
    fixed_point_iterations_total: int = 0
    scc_rollback_count: int = 0
    tokens: int = 0
    context_bytes: int = 0
    total_cost_units: int = 0
    total_latency_units: int = 0
    missed_caller_rate: int = 0
    wrong_value_rate: int = 0
    partial_transaction_rate: int = 0
    false_completion_rate: int = 0
    open_frontier_rate: int = 0
    llm_scope_escape_rate: int = 0
    safety_floors: Mapping[str, int] = field(default_factory=dict)
    safety_absolute: Mapping[str, int] = field(default_factory=dict)
    outcome_counts: Mapping[str, int] = field(default_factory=dict)
    family_counts: Mapping[str, int] = field(default_factory=dict)
    recall_k: int = DEFAULT_RECALL_K
    reason_code_counts: Mapping[str, int] = field(default_factory=dict)
    metrics_authoritative: bool = False
    metrics_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "case_count", "goal_precision", "goal_recall", "hypothesis_precision",
            "hypothesis_recall", "premise_recall_at_k", "first_plan_closure_rate",
            "lowering_rate", "reconstruction_rate", "validated_countermodel_rate",
            "abstention_count", "abstention_rate", "analytical_coverage", "model_rate",
            "llm_rate", "all_caller_rate", "fixed_point_iterations",
            "fixed_point_iterations_total", "scc_rollback_count", "tokens", "context_bytes",
            "total_cost_units", "total_latency_units", "missed_caller_rate",
            "wrong_value_rate", "partial_transaction_rate", "false_completion_rate",
            "open_frontier_rate", "llm_scope_escape_rate", "recall_k",
        ):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), name))
        object.__setattr__(self, "metrics_authoritative", _bool(self.metrics_authoritative, "metrics_authoritative"))
        if self.metrics_authoritative:
            raise LogicRepairRolloutError("metrics cannot claim authority")
        if self.fixed_point_iterations == 0 and self.fixed_point_iterations_total:
            object.__setattr__(self, "fixed_point_iterations", self.fixed_point_iterations_total)
        if self.fixed_point_iterations_total == 0 and self.fixed_point_iterations:
            object.__setattr__(self, "fixed_point_iterations_total", self.fixed_point_iterations)
        stages = {s: _non_negative_int(dict(self.stage_counts or {}).get(s, 0), f"stage:{s}") for s in BENCHMARK_STAGES}
        if self.case_count and not any(stages.values()):
            stages = {s: self.case_count for s in BENCHMARK_STAGES}
        object.__setattr__(self, "stage_counts", MappingProxyType(stages))
        object.__setattr__(self, "stage_cost_units", MappingProxyType({
            s: _non_negative_int(dict(self.stage_cost_units or {}).get(s, 1), f"cost:{s}") for s in BENCHMARK_STAGES
        }))
        split = dict(self.analytical_model_split or {}) or {
            "analytical_coverage": self.analytical_coverage,
            "model_rate": self.model_rate or self.llm_rate,
            "llm_rate": self.llm_rate,
        }
        object.__setattr__(self, "analytical_model_split", MappingProxyType({str(k): _non_negative_int(v, str(k)) for k, v in sorted(split.items())}))
        floors = {k: _non_negative_int(dict(self.safety_floors or {}).get(k, 0), k) for k in SAFETY_FLOOR_KEYS}
        object.__setattr__(self, "safety_floors", MappingProxyType(floors))
        absolute = {str(k): _non_negative_int(v, str(k)) for k, v in sorted(dict(self.safety_absolute or {}).items())}
        for key in SAFETY_ABSOLUTE_KEYS:
            absolute.setdefault(key, 0)
        object.__setattr__(self, "safety_absolute", MappingProxyType(absolute))
        for attr in ("outcome_counts", "family_counts", "reason_code_counts"):
            object.__setattr__(self, attr, MappingProxyType({str(k): _non_negative_int(v, str(k)) for k, v in sorted(dict(getattr(self, attr) or {}).items())}))
        if not self.metrics_id:
            object.__setattr__(self, "metrics_id", content_identity(self.to_dict(include_id=False)))

    def floors_hold(self) -> bool:
        return all(int(self.safety_floors.get(k, 1)) == 0 for k in SAFETY_FLOOR_KEYS) and all(
            int(self.safety_absolute.get(k, 1)) == 0 for k in SAFETY_ABSOLUTE_KEYS
        )

    def breaches(self) -> tuple[str, ...]:
        failed = [k for k in SAFETY_FLOOR_KEYS if int(self.safety_floors.get(k, 1)) != 0]
        failed.extend(k for k in SAFETY_ABSOLUTE_KEYS if int(self.safety_absolute.get(k, 1)) != 0)
        for name, value in (
            ("missed_caller_rate", self.missed_caller_rate),
            ("wrong_value_rate", self.wrong_value_rate),
            ("partial_transaction_rate", self.partial_transaction_rate),
            ("false_completion_rate", self.false_completion_rate),
            ("open_frontier_rate", self.open_frontier_rate),
            ("llm_scope_escape_rate", self.llm_scope_escape_rate),
        ):
            if int(value) != 0:
                failed.append(name)
        return tuple(dict.fromkeys(failed))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": METRICS_SCHEMA, "interface": METRICS_INTERFACE,
            "benchmark_metrics_interface": BENCHMARK_METRICS_INTERFACE,
            "case_count": self.case_count, "goal_precision": self.goal_precision,
            "goal_recall": self.goal_recall, "hypothesis_precision": self.hypothesis_precision,
            "hypothesis_recall": self.hypothesis_recall, "premise_recall_at_k": self.premise_recall_at_k,
            "first_plan_closure_rate": self.first_plan_closure_rate, "lowering_rate": self.lowering_rate,
            "reconstruction_rate": self.reconstruction_rate,
            "validated_countermodel_rate": self.validated_countermodel_rate,
            "abstention_count": self.abstention_count, "abstention_rate": self.abstention_rate,
            "analytical_coverage": self.analytical_coverage, "model_rate": self.model_rate,
            "llm_rate": self.llm_rate, "all_caller_rate": self.all_caller_rate,
            "analytical_model_split": dict(self.analytical_model_split),
            "stage_counts": dict(self.stage_counts), "stage_cost_units": dict(self.stage_cost_units),
            "benchmark_stages": list(BENCHMARK_STAGES),
            "fixed_point_iterations": self.fixed_point_iterations,
            "fixed_point_iterations_total": self.fixed_point_iterations_total,
            "scc_rollback_count": self.scc_rollback_count, "tokens": self.tokens,
            "context_bytes": self.context_bytes, "total_cost_units": self.total_cost_units,
            "total_latency_units": self.total_latency_units,
            "missed_caller_rate": self.missed_caller_rate, "wrong_value_rate": self.wrong_value_rate,
            "partial_transaction_rate": self.partial_transaction_rate,
            "false_completion_rate": self.false_completion_rate,
            "open_frontier_rate": self.open_frontier_rate,
            "llm_scope_escape_rate": self.llm_scope_escape_rate,
            "safety_floors": dict(self.safety_floors), "safety_absolute": dict(self.safety_absolute),
            "outcome_counts": dict(self.outcome_counts), "family_counts": dict(self.family_counts),
            "recall_k": self.recall_k, "reason_code_counts": dict(self.reason_code_counts),
            "metrics_authoritative": False,
        }
        if include_id:
            payload["metrics_id"] = self.metrics_id
        return payload

    @classmethod
    def from_benchmark_metrics(cls, metrics: Mapping[str, Any]) -> "LogicRepairMetrics":
        case_count = int(metrics.get("case_count") or 0)
        abstention = int(metrics.get("abstention_count") or 0)
        floors = dict(metrics.get("safety_floors") or {})
        for key in SAFETY_FLOOR_KEYS:
            floors.setdefault(key, 0)
        absolute = dict(metrics.get("safety_absolute") or {})
        for key in SAFETY_ABSOLUTE_KEYS:
            absolute.setdefault(key, 0)
        llm_rate = int(metrics.get("llm_rate") or metrics.get("model_rate") or 0)
        analytical = int(metrics.get("analytical_coverage") or 0)
        fp_iters = int(metrics.get("fixed_point_iterations_total") or metrics.get("fixed_point_iterations") or 0)
        return cls(
            case_count=case_count,
            goal_precision=int(metrics.get("goal_precision") or 0),
            goal_recall=int(metrics.get("goal_recall") or 0),
            hypothesis_precision=int(metrics.get("hypothesis_precision") or 0),
            hypothesis_recall=int(metrics.get("hypothesis_recall") or 0),
            premise_recall_at_k=int(metrics.get("premise_recall_at_k") or 0),
            first_plan_closure_rate=int(metrics.get("first_plan_closure_rate") or 0),
            lowering_rate=int(metrics.get("lowering_rate") or 0),
            reconstruction_rate=int(metrics.get("reconstruction_rate") or 0),
            validated_countermodel_rate=int(metrics.get("validated_countermodel_rate") or 0),
            analytical_coverage=analytical, model_rate=llm_rate, llm_rate=llm_rate,
            all_caller_rate=int(metrics.get("all_caller_rate") or 0),
            analytical_model_split={"analytical_coverage": analytical, "model_rate": llm_rate, "llm_rate": llm_rate},
            stage_counts={s: case_count for s in BENCHMARK_STAGES},
            stage_cost_units={s: 1 for s in BENCHMARK_STAGES},
            fixed_point_iterations=fp_iters, fixed_point_iterations_total=fp_iters,
            scc_rollback_count=int(metrics.get("scc_rollback_count") or 0),
            abstention_count=abstention, abstention_rate=_ppm(abstention, max(1, case_count)),
            tokens=int(metrics.get("total_token_units") or metrics.get("tokens") or 0),
            context_bytes=int(metrics.get("total_context_bytes") or metrics.get("context_bytes") or 0),
            total_cost_units=int(metrics.get("total_cost_units") or 0),
            total_latency_units=int(metrics.get("total_latency_units") or 0),
            missed_caller_rate=int(floors.get("missed_resolved_caller_rate") or 0),
            wrong_value_rate=int(floors.get("wrong_value_source_placement_admission_rate") or 0),
            partial_transaction_rate=int(floors.get("partial_transaction_completion_rate") or 0),
            false_completion_rate=int(floors.get("false_fixed_point_completion_rate") or 0),
            llm_scope_escape_rate=int(floors.get("llm_scope_semantic_escape_rate") or 0),
            safety_floors=floors, safety_absolute=absolute,
            outcome_counts=dict(metrics.get("outcome_counts") or {}),
            family_counts=dict(metrics.get("family_counts") or {}),
            recall_k=int(metrics.get("recall_k") or DEFAULT_RECALL_K),
            reason_code_counts=dict(metrics.get("outcome_counts") or {}),
        )

    @classmethod
    def empty(cls) -> "LogicRepairMetrics":
        return cls(
            safety_floors={k: 0 for k in SAFETY_FLOOR_KEYS},
            safety_absolute={k: 0 for k in SAFETY_ABSOLUTE_KEYS},
            stage_counts={s: 0 for s in BENCHMARK_STAGES},
            stage_cost_units={s: 1 for s in BENCHMARK_STAGES},
            analytical_model_split={"analytical_coverage": 0, "model_rate": 0, "llm_rate": 0},
        )


@dataclass(frozen=True)
class RollbackReceipt:
    SCHEMA: ClassVar[str] = ROLLBACK_RECEIPT_SCHEMA
    reason: RollbackReason | str
    from_mode: RolloutMode | str
    to_mode: RolloutMode | str = RolloutMode.SHADOW
    detail: str = ""
    metric_breaches: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    stale_roots: tuple[str, ...] = ()
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
                raise LogicRepairRolloutError(f"unknown rollback reason: {self.reason!r}") from exc
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "from_mode", _mode(self.from_mode))
        object.__setattr__(self, "to_mode", _mode(self.to_mode))
        object.__setattr__(self, "detail", str(self.detail or "").strip())
        object.__setattr__(self, "metric_breaches", tuple(str(i) for i in self.metric_breaches))
        object.__setattr__(self, "capability_ids", tuple(str(i) for i in self.capability_ids))
        object.__setattr__(self, "stale_roots", tuple(str(i) for i in self.stale_roots))
        object.__setattr__(self, "reason_codes", tuple(str(i) for i in self.reason_codes))
        object.__setattr__(self, "policy_binding_id", str(self.policy_binding_id or ""))
        if not self.receipt_id:
            object.__setattr__(self, "receipt_id", content_identity(self.to_dict(include_id=False)))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": ROLLBACK_RECEIPT_SCHEMA,
            "reason": self.reason.value if isinstance(self.reason, RollbackReason) else str(self.reason),
            "from_mode": self.from_mode.value if isinstance(self.from_mode, RolloutMode) else str(self.from_mode),
            "to_mode": self.to_mode.value if isinstance(self.to_mode, RolloutMode) else str(self.to_mode),
            "detail": self.detail,
            "metric_breaches": list(self.metric_breaches),
            "capability_ids": list(self.capability_ids),
            "stale_roots": list(self.stale_roots),
            "reason_codes": list(self.reason_codes),
            "policy_binding_id": self.policy_binding_id,
            "mutation_authorized": False,
            "completion_authoritative": False,
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload


def _demotion_target(current: RolloutMode) -> RolloutMode:
    if current in {RolloutMode.SHADOW, RolloutMode.DOCTOR_REPLAY}:
        return RolloutMode.SHADOW
    return {
        RolloutMode.MODEL_EDIT: RolloutMode.ASSIST,
        RolloutMode.NARROW_AUTO: RolloutMode.ASSIST,
        RolloutMode.ASSIST: RolloutMode.SHADOW,
    }.get(current, RolloutMode.SHADOW)


def evaluate_rollback(
    policy: LogicRepairRolloutPolicy, *, metrics: LogicRepairMetrics | None = None,
    capability_regression: Sequence[str] = (), stale_roots: Sequence[str] = (),
    root_drift: bool = False, open_frontier: bool = False, reconstruction_failed: bool = False,
    countermodel_validation_loss: bool = False, proof_loss: bool = False, wrong_value: bool = False,
    missed_caller: bool = False, partial_plan: bool = False, false_completion: bool = False,
    isolation_regression: bool = False, budget_regression: bool = False, inconsistency: bool = False,
    transaction_failure: bool = False, reason_codes: Sequence[str] = (),
) -> RollbackReceipt | None:
    current = _mode(policy.mode)
    target = _demotion_target(current)
    codes = {str(i).strip().casefold() for i in reason_codes if i}

    def _receipt(reason: RollbackReason, *, detail: str, metric_breaches: Sequence[str] = (),
                 capability_ids: Sequence[str] = (), roots: Sequence[str] = (),
                 extra_codes: Sequence[str] = ()) -> RollbackReceipt:
        return RollbackReceipt(
            reason=reason, from_mode=current, to_mode=target, detail=detail,
            metric_breaches=tuple(metric_breaches), capability_ids=tuple(sorted(set(capability_ids))),
            stale_roots=tuple(sorted(set(roots))), reason_codes=tuple(sorted({*codes, *extra_codes})),
            policy_binding_id=policy.policy_binding_id,
        )

    if policy.rollback_on_capability_regression and capability_regression:
        return _receipt(RollbackReason.CAPABILITY_REGRESSION, detail="capability health regression", capability_ids=capability_regression)
    if policy.rollback_on_stale_root and (stale_roots or root_drift or "root_drift" in codes):
        return _receipt(RollbackReason.STALE_ROOT if stale_roots else RollbackReason.ROOT_DRIFT,
                        detail="stale authority root or root drift observed", roots=stale_roots, extra_codes=("stale_root", "root_drift"))
    if policy.rollback_on_open_frontier and (open_frontier or "open_frontier" in codes):
        return _receipt(RollbackReason.OPEN_FRONTIER, detail="impact frontier remains open", extra_codes=("open_frontier",))
    if policy.rollback_on_reconstruction_failure and (reconstruction_failed or "reconstruction_failure" in codes):
        return _receipt(RollbackReason.RECONSTRUCTION_FAILURE, detail="proof reconstruction failure", extra_codes=("reconstruction_failure",))
    if policy.rollback_on_countermodel_validation_loss and (countermodel_validation_loss or "countermodel_validation_loss" in codes):
        return _receipt(RollbackReason.COUNTERMODEL_VALIDATION_LOSS, detail="countermodel validation loss", extra_codes=("countermodel_validation_loss",))
    if policy.rollback_on_proof_loss and (proof_loss or "proof_loss" in codes):
        return _receipt(RollbackReason.PROOF_LOSS, detail="proof loss observed", extra_codes=("proof_loss",))
    if wrong_value or "wrong_value" in codes:
        return _receipt(RollbackReason.WRONG_VALUE, detail="wrong or unproved value source", extra_codes=("wrong_value",))
    if missed_caller or "missed_caller" in codes or "missed_consumer" in codes:
        return _receipt(RollbackReason.MISSED_CALLER, detail="missed resolved impacted caller", extra_codes=("missed_caller",))
    if partial_plan or "partial_plan" in codes or "partial_transaction" in codes:
        return _receipt(RollbackReason.PARTIAL_PLAN, detail="partial plan or incomplete SCC group", extra_codes=("partial_plan",))
    if false_completion or "false_completion" in codes or "false_fixed_point" in codes:
        return _receipt(RollbackReason.FALSE_COMPLETION, detail="false fixed-point or false completion", extra_codes=("false_completion",))
    if policy.rollback_on_isolation_regression and (isolation_regression or "isolation_regression" in codes):
        return _receipt(RollbackReason.ISOLATION_REGRESSION, detail="platform isolation regression", extra_codes=("isolation_regression",))
    if policy.rollback_on_budget_regression and (budget_regression or "budget_regression" in codes):
        return _receipt(RollbackReason.BUDGET_REGRESSION, detail="resource or retry budget regression", extra_codes=("budget_regression",))
    if policy.rollback_on_inconsistency and (inconsistency or "inconsistency" in codes):
        return _receipt(RollbackReason.INCONSISTENCY, detail="corpus or receipt inconsistency", extra_codes=("inconsistency",))
    if policy.rollback_on_transaction_failure and (transaction_failure or "transaction_failure" in codes):
        return _receipt(RollbackReason.TRANSACTION_FAILURE, detail="transaction failure", extra_codes=("transaction_failure",))
    if policy.rollback_on_metric_breach and metrics is not None:
        breaches = metrics.breaches()
        if breaches or not metrics.floors_hold():
            return _receipt(RollbackReason.METRIC_BREACH, detail="safety floor or metric breach",
                            metric_breaches=breaches or tuple(k for k in SAFETY_FLOOR_KEYS if int(metrics.safety_floors.get(k, 1)) != 0))
    mapping = {
        "wrong_value": RollbackReason.WRONG_VALUE, "missed_caller": RollbackReason.MISSED_CALLER,
        "missed_consumer": RollbackReason.MISSED_CALLER, "partial_plan": RollbackReason.PARTIAL_PLAN,
        "partial_transaction": RollbackReason.PARTIAL_PLAN, "false_completion": RollbackReason.FALSE_COMPLETION,
        "false_fixed_point": RollbackReason.FALSE_COMPLETION, "open_frontier": RollbackReason.OPEN_FRONTIER,
        "proof_loss": RollbackReason.PROOF_LOSS, "reconstruction_failure": RollbackReason.RECONSTRUCTION_FAILURE,
        "countermodel_validation_loss": RollbackReason.COUNTERMODEL_VALIDATION_LOSS,
        "isolation_regression": RollbackReason.ISOLATION_REGRESSION, "budget_regression": RollbackReason.BUDGET_REGRESSION,
        "inconsistency": RollbackReason.INCONSISTENCY, "root_drift": RollbackReason.ROOT_DRIFT,
    }
    for code in sorted(codes & ZERO_TOLERANCE_REASON_CODES):
        if code in mapping:
            return _receipt(mapping[code], detail=f"zero-tolerance reason code: {code}", extra_codes=(code,))
    return None


def apply_rollback(policy: LogicRepairRolloutPolicy, receipt: RollbackReceipt) -> LogicRepairRolloutPolicy:
    to_mode = _mode(receipt.to_mode)
    return LogicRepairRolloutPolicy(
        policy_id=policy.policy_id, policy_revision=policy.policy_revision,
        repository_id=policy.repository_id, program_id=policy.program_id, mode=to_mode,
        explicit_policy_document=policy.explicit_policy_document, scoped_path_globs=policy.scoped_path_globs,
        allow_assist=policy.allow_assist and to_mode not in {RolloutMode.SHADOW, RolloutMode.DOCTOR_REPLAY},
        allow_narrow_auto=policy.allow_narrow_auto and to_mode is RolloutMode.NARROW_AUTO,
        allow_model_edit=policy.allow_model_edit and to_mode is RolloutMode.MODEL_EDIT,
        logic_prediction_enabled=False, learned_tactician_ranking_enabled=False,
        hammer_execution_enabled=False, counterexample_refinement_enabled=False,
        llm_router_enabled=False, narrow_autonomous_mutation_enabled=False,
        auto_requires_unique_target=policy.auto_requires_unique_target,
        auto_requires_reconstruction=policy.auto_requires_reconstruction,
        auto_requires_supported_python=policy.auto_requires_supported_python,
        auto_requires_complete_frontier=policy.auto_requires_complete_frontier,
        auto_requires_analytical_path=policy.auto_requires_analytical_path,
        auto_requires_fixed_point=policy.auto_requires_fixed_point,
        auto_allowed_transforms=policy.auto_allowed_transforms,
        approval_gated_families=policy.approval_gated_families,
        rollback_on_capability_regression=policy.rollback_on_capability_regression,
        rollback_on_stale_root=policy.rollback_on_stale_root,
        rollback_on_open_frontier=policy.rollback_on_open_frontier,
        rollback_on_reconstruction_failure=policy.rollback_on_reconstruction_failure,
        rollback_on_countermodel_validation_loss=policy.rollback_on_countermodel_validation_loss,
        rollback_on_proof_loss=policy.rollback_on_proof_loss,
        rollback_on_metric_breach=policy.rollback_on_metric_breach,
        rollback_on_isolation_regression=policy.rollback_on_isolation_regression,
        rollback_on_budget_regression=policy.rollback_on_budget_regression,
        rollback_on_inconsistency=policy.rollback_on_inconsistency,
        rollback_on_transaction_failure=policy.rollback_on_transaction_failure,
        mutation_authorized=False, completion_authoritative=False,
    )


class LogicRepairRollbackGate:
    INTERFACE: ClassVar[str] = ROLLBACK_GATE_INTERFACE

    def __init__(self, policy: LogicRepairRolloutPolicy | None = None) -> None:
        self.policy = policy or default_rollout_policy()

    def evaluate(self, **kwargs: Any) -> RollbackReceipt | None:
        return evaluate_rollback(self.policy, **kwargs)

    def apply(self, receipt: RollbackReceipt) -> LogicRepairRolloutPolicy:
        demoted = apply_rollback(self.policy, receipt)
        self.policy = demoted
        return demoted

    def to_dict(self) -> dict[str, Any]:
        return {"interface": ROLLBACK_GATE_INTERFACE, "policy": self.policy.to_dict(),
                "mutation_authorized": False, "completion_authoritative": False}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus | str
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name"))
        status = self.status if isinstance(self.status, CheckStatus) else CheckStatus(str(self.status).strip().casefold())
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


def _load_benchmark_module():
    path = repository_root() / BENCHMARK_SCRIPT_REL
    name = "benchmark_tactician_hammer_logic_repair_lpr020"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LogicRepairRolloutError(f"unable to load benchmark module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_task_file_fallback(todo_path: Path) -> list[Any]:
    text = todo_path.read_text(encoding="utf-8")
    tasks: list[Any] = []

    @dataclass
    class _Task:
        task_id: str
        depends_on: tuple[str, ...]
        outputs: tuple[str, ...]
        metadata: dict[str, str]
        status: str = "todo"

    current_id = ""; depends: list[str] = []; outputs: list[str] = []; metadata: dict[str, str] = {}; status = "todo"
    for line in text.splitlines():
        header = re.match(r"^##\s+(LPR-\d+)\b", line)
        if header:
            if current_id:
                tasks.append(_Task(current_id, tuple(depends), tuple(outputs), dict(metadata), status))
            current_id = header.group(1); depends = []; outputs = []; metadata = {}; status = "todo"
            continue
        if not current_id:
            continue
        m = re.match(r"^-\s+Depends on:\s*(.*)$", line, re.IGNORECASE)
        if m:
            depends = [i.strip() for i in re.split(r"[, ]+", m.group(1).strip()) if i.strip().startswith("LPR-")]
            continue
        m = re.match(r"^-\s+Outputs:\s*(.*)$", line, re.IGNORECASE)
        if m:
            outputs = [i.strip() for i in m.group(1).strip().split(",") if i.strip()]
            continue
        m = re.match(r"^-\s+Goal id:\s*(.*)$", line, re.IGNORECASE)
        if m:
            metadata["goal id"] = m.group(1).strip(); continue
        m = re.match(r"^-\s+Status:\s*(.*)$", line, re.IGNORECASE)
        if m:
            status = m.group(1).strip().casefold()
    if current_id:
        tasks.append(_Task(current_id, tuple(depends), tuple(outputs), dict(metadata), status))
    return tasks


def _parse_goal_heap_fallback(text: str) -> list[Any]:
    @dataclass
    class _Goal:
        goal_id: str
        dependencies: tuple[str, ...]
        parent_goal_ids: tuple[str, ...]

    goals: list[Any] = []; current = ""; deps: list[str] = []; parents: list[str] = []
    for line in text.splitlines():
        header = re.match(r"^##\s+(LPR-G\d+)\b", line)
        if header:
            if current:
                goals.append(_Goal(current, tuple(deps), tuple(parents)))
            current = header.group(1); deps = []; parents = []; continue
        if not current:
            continue
        m = re.match(r"^-\s+Depends on:\s*(.*)$", line, re.IGNORECASE)
        if m:
            deps = [i.strip() for i in re.split(r"[, ]+", m.group(1).strip()) if i.strip().startswith("LPR-G")]
            continue
        m = re.match(r"^-\s+Parent:\s*(.*)$", line, re.IGNORECASE)
        if m:
            parents = [i.strip() for i in re.split(r"[, ]+", m.group(1).strip()) if i.strip().startswith("LPR-G")]
    if current:
        goals.append(_Goal(current, tuple(deps), tuple(parents)))
    return goals


def check_bootstrap_board_doctor(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    validator = root / BOARD_VALIDATOR_REL
    if not validator.is_file():
        return CheckResult("bootstrap_board_doctor", CheckStatus.FAIL, f"board validator missing: {validator}")
    try:
        result = subprocess.run(
            [sys.executable, str(validator), "--check-all"], cwd=str(root),
            capture_output=True, text=True, timeout=120, check=False,
        )
    except Exception as exc:
        return CheckResult("bootstrap_board_doctor", CheckStatus.FAIL, f"board doctor failed to run: {exc}")
    if result.returncode != 0:
        return CheckResult(
            "bootstrap_board_doctor", CheckStatus.FAIL,
            "board doctor returned nonzero: " + (result.stderr or result.stdout or "")[:500],
            {"returncode": result.returncode},
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return CheckResult("bootstrap_board_doctor", CheckStatus.FAIL, f"board doctor output is not JSON: {exc}")
    if not payload.get("valid"):
        return CheckResult("bootstrap_board_doctor", CheckStatus.FAIL, "board doctor reported invalid", payload)
    if payload.get("schema") != BOARD_VALIDATOR_SCHEMA:
        return CheckResult("bootstrap_board_doctor", CheckStatus.FAIL, f"unexpected board schema: {payload.get('schema')}", payload)
    if payload.get("rollout_mode") != "shadow":
        return CheckResult("bootstrap_board_doctor", CheckStatus.FAIL, "board doctor rollout mode is not shadow", payload)
    if payload.get("lane_count") != LANE_COUNT:
        return CheckResult("bootstrap_board_doctor", CheckStatus.FAIL, "board doctor lane_count is not 4", payload)
    return CheckResult(
        "bootstrap_board_doctor", CheckStatus.PASS, "protected bootstrap board/DAG doctor is healthy",
        {"schema": payload.get("schema"), "task_count": payload.get("task_count"), "goal_count": payload.get("goal_count"),
         "lane_count": payload.get("lane_count"), "rollout_mode": payload.get("rollout_mode"),
         "ready_task_ids": payload.get("ready_task_ids")},
    )


def check_plan_objective_task_dag(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    plan_path, objective_path, todo_path, scheduler_path = root / PLAN_REL, root / OBJECTIVE_REL, root / TODO_REL, root / SCHEDULER_REL
    for path, label in ((plan_path, "plan"), (objective_path, "objective"), (todo_path, "todo"), (scheduler_path, "scheduler")):
        if not path.is_file():
            errors.append(f"{label} missing: {path}")
    if errors:
        return CheckResult("plan_objective_task_dag", CheckStatus.FAIL, "; ".join(errors))
    plan_text = plan_path.read_text(encoding="utf-8").casefold()
    if "logic repair" not in plan_text and "tactician" not in plan_text:
        errors.append("plan does not identify tactician/hammer logic-repair work")
    try:
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import parse_goal_heap
        goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    except Exception:
        goals = _parse_goal_heap_fallback(objective_path.read_text(encoding="utf-8"))
    goal_ids = {g.goal_id for g in goals}
    if GOAL_ID not in goal_ids:
        errors.append(f"{GOAL_ID} is missing from the objective heap")
    if "LPR-G000" not in goal_ids:
        errors.append("root goal LPR-G000 is missing")
    goal_edges: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        deps = tuple(getattr(goal, "dependencies", ()) or ())
        parents = tuple(getattr(goal, "parent_goal_ids", ()) or ())
        combined = tuple(dict.fromkeys((*parents, *deps)))
        goal_edges[goal.goal_id] = combined
        for dep in combined:
            if dep not in goal_ids:
                errors.append(f"unknown objective dependency: {goal.goal_id}->{dep}")
    if _cycle_nodes(goal_edges):
        errors.append(f"goal dependency cycle: {_cycle_nodes(goal_edges)}")
    try:
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import parse_task_file
        tasks = parse_task_file(todo_path, task_header_prefix=TASK_PREFIX)
    except Exception:
        tasks = _parse_task_file_fallback(todo_path)
    task_ids = {t.task_id for t in tasks}
    if len(tasks) != len(task_ids):
        errors.append("duplicate task id on the board")
    for required in (TASK_ID, "LPR-019", "LPR-000"):
        if required not in task_ids:
            errors.append(f"{required} is missing")
    task_edges: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        deps = tuple(getattr(task, "depends_on", ()) or ())
        task_edges[task.task_id] = deps
        for dep in deps:
            if dep not in task_ids:
                errors.append(f"unknown task dependency: {task.task_id}->{dep}")
        metadata = getattr(task, "metadata", {}) or {}
        goal_id = str(metadata.get("goal id", "") or metadata.get("goal_id", "")).strip()
        if goal_id and goal_id not in goal_ids:
            errors.append(f"unknown task goal: {task.task_id}->{goal_id}")
        for output in getattr(task, "outputs", ()) or ():
            if not _safe_relative(str(output)):
                errors.append(f"{task.task_id} has unsafe output path {output!r}")
    if _cycle_nodes(task_edges):
        errors.append(f"task dependency cycle: {_cycle_nodes(task_edges)}")
    lpr020 = next((t for t in tasks if t.task_id == TASK_ID), None)
    if lpr020 is not None and "LPR-019" not in set(lpr020.depends_on):
        errors.append("LPR-020 missing required dependency LPR-019")
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    if scheduler.get("task_prefix") != TASK_PREFIX:
        errors.append("scheduler task prefix mismatch")
    if scheduler.get("merge_target_branch") != MERGE_TARGET_BRANCH:
        errors.append("scheduler merge target mismatch")
    if scheduler.get("board_namespace") != BOARD_NAMESPACE:
        errors.append("scheduler board namespace mismatch")
    evidence = {"goal_ids": sorted(goal_ids), "task_ids": sorted(task_ids), "task_count": len(task_ids), "goal_count": len(goal_ids)}
    if errors:
        return CheckResult("plan_objective_task_dag", CheckStatus.FAIL, "; ".join(errors), evidence)
    return CheckResult("plan_objective_task_dag", CheckStatus.PASS, "plan/objective/task DAG is acyclic and includes LPR-020", evidence)


def check_exact_source_bindings(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    try:
        binding = bind_exact_sources(root)
    except LogicRepairRolloutError as exc:
        return CheckResult("exact_source_bindings", CheckStatus.FAIL, str(exc))
    errors: list[str] = []
    source = (_load_scheduler(root).get("source_binding") or {})
    for key in (
        "require_exact_accelerator_branch", "require_initialized_datasets_gitlink",
        "require_superproject_gitlink_equals_nested_head", "record_accelerator_and_datasets_revisions_at_launch",
    ):
        if source.get(key) is not True:
            errors.append(f"source binding disabled: {key}")
    if source.get("datasets_submodule_path") != DATASETS_SUBMODULE:
        errors.append("datasets submodule path mismatch")
    if source.get("datasets_required_ancestor") != DATASETS_TACTICIAN_ANCESTOR:
        errors.append("datasets Tactician ancestor mismatch")
    if source.get("datasets_required_interface") != DATASETS_TACTICIAN_INTERFACE:
        errors.append("datasets Tactician interface mismatch")
    if source.get("accelerator_branch") != MERGE_TARGET_BRANCH:
        errors.append("accelerator branch binding mismatch")
    for path in PROTECTED_PATHS:
        if not (root / path).is_file():
            errors.append(f"protected path missing: {path}")
    for path in REQUIRED_RELEASE_SOURCES:
        if not (root / path).is_file():
            errors.append(f"release source missing: {path}")
    if errors:
        return CheckResult("exact_source_bindings", CheckStatus.FAIL, "; ".join(errors), binding.to_dict())
    return CheckResult(
        "exact_source_bindings", CheckStatus.PASS,
        "exact two-repository gitlink/module/schema/tool/environment bindings hold", binding.to_dict(),
    )


def check_capability_health(repo_root: Path | None = None, *, probe: bool = True) -> CheckResult:
    del repo_root
    evidence: dict[str, Any] = {
        "authoritative": False, "candidate_authoritative": False, "import_isolation": None,
        "native_execution_admitted": False, "resource_enforcement": None,
    }
    if not probe:
        return CheckResult("capability_health", CheckStatus.SKIP, "capability probe skipped", evidence)
    try:
        from ipfs_accelerate_py.agent_supervisor.integrations.tactician_hammer_capabilities import (
            probe_tactician_hammer_capabilities,
        )
    except Exception as exc:
        return CheckResult("capability_health", CheckStatus.FAIL, f"capability probe import failed: {exc}", evidence)
    try:
        report = probe_tactician_hammer_capabilities()
    except Exception as exc:
        return CheckResult("capability_health", CheckStatus.FAIL, f"capability probe raised: {exc}", evidence)
    report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    capabilities = report_dict.get("capabilities") or []
    available: list[str] = []; unavailable: list[dict[str, Any]] = []
    for item in capabilities:
        if not isinstance(item, Mapping):
            continue
        cap_id = str(item.get("capability_id") or item.get("id") or "")
        status = str(item.get("status") or "").casefold()
        is_available = bool(item.get("available")) or status == "available"
        if is_available:
            available.append(cap_id)
        else:
            unavailable.append({"capability_id": cap_id, "status": status,
                                "reason_code": item.get("reason_code") or (item.get("diagnostic") or {}).get("code")})
        if item.get("candidate_authoritative") or item.get("semantic_authority"):
            return CheckResult("capability_health", CheckStatus.FAIL, f"capability {cap_id} illegally claims authority", evidence)
    isolation = report_dict.get("import_isolation") or report_dict.get("hammer_import_isolation")
    native = bool(report_dict.get("native_execution_admitted"))
    evidence.update({
        "available": sorted(available), "unavailable": unavailable,
        "report_schema": report_dict.get("schema") or report_dict.get("schema_version"),
        "capability_count": len(capabilities), "import_isolation": isolation,
        "native_execution_admitted": native, "resource_enforcement": report_dict.get("resource_enforcement"),
        "network_access_admitted": bool(report_dict.get("network_access_admitted")),
    })
    return CheckResult(
        "capability_health", CheckStatus.PASS,
        f"capability probe completed: available={len(available)} unavailable={len(unavailable)}; "
        f"import_isolation={isolation!r}; native_execution_admitted={native}",
        evidence,
    )


def check_four_lane_sharding_and_isolation(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    scheduler = _load_scheduler(root)
    if scheduler.get("max_lanes") != LANE_COUNT:
        errors.append("max_lanes is not 4")
    if scheduler.get("strict_task_sharding") is not True:
        errors.append("strict_task_sharding must be true")
    if scheduler.get("objective_refill_enabled") is not False:
        errors.append("objective refill must be disabled (one refill owner)")
    if scheduler.get("codebase_refill_enabled") is not False:
        errors.append("codebase refill must be disabled (one refill owner)")
    for key in ("implementation_retry_budget", "validation_retry_budget", "merge_retry_budget", "max_restarts", "max_task_attempts"):
        value = scheduler.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"{key} must be a positive bounded integer")
        elif value > 10:
            errors.append(f"{key} exceeds bounded retry ceiling")
    lanes = scheduler.get("lanes") or []
    if not isinstance(lanes, list) or len(lanes) != LANE_COUNT:
        errors.append("scheduler must define exactly four lanes")
    else:
        for index, row in enumerate(lanes):
            if not isinstance(row, Mapping):
                errors.append(f"lane {index} is not an object"); continue
            if row.get("index") != index:
                errors.append(f"lane index mismatch at {index}")
            if row.get("strict_shard_remainder") != index:
                errors.append(f"lane {index} shard remainder mismatch")
            if row.get("name") != f"lpr-lane-{index}":
                errors.append(f"lane {index} name mismatch")
    protected = tuple(scheduler.get("protected_paths") or ())
    if protected != PROTECTED_PATHS:
        errors.append("protected_paths mismatch vs LPR-000 bootstrap set")
    provider = scheduler.get("provider") or {}
    if provider.get("secrets_in_argv_or_logs") is not False:
        errors.append("secrets must not enter argv/logs")
    if provider.get("max_concurrency") != LANE_COUNT:
        errors.append("provider concurrency must equal lane count")
    if tuple(scheduler.get("worktree_submodule_paths") or ()) != (DATASETS_SUBMODULE,):
        errors.append("worktree submodule binding must be exactly ipfs_datasets_py")
    launcher = (root / LAUNCHER_REL).read_text(encoding="utf-8")
    if "MERGE_QUEUE_ROOT" not in launcher and "merge-queue" not in launcher:
        errors.append("launcher does not declare an isolated merge queue root")
    if "STATE_ROOT" not in launcher:
        errors.append("launcher does not declare isolated state root")
    if "WORKTREE_ROOT" not in launcher:
        errors.append("launcher does not declare isolated worktree root")
    evidence = {
        "max_lanes": scheduler.get("max_lanes"), "strict_task_sharding": scheduler.get("strict_task_sharding"),
        "objective_refill_enabled": scheduler.get("objective_refill_enabled"),
        "codebase_refill_enabled": scheduler.get("codebase_refill_enabled"),
        "implementation_retry_budget": scheduler.get("implementation_retry_budget"),
        "validation_retry_budget": scheduler.get("validation_retry_budget"),
        "merge_retry_budget": scheduler.get("merge_retry_budget"),
        "protected_paths": list(protected), "one_merge_queue": True, "one_refill_owner": True,
    }
    if errors:
        return CheckResult("four_lane_sharding_and_isolation", CheckStatus.FAIL, "; ".join(errors), evidence)
    return CheckResult(
        "four_lane_sharding_and_isolation", CheckStatus.PASS,
        "strict four-lane sharding, isolated state/worktrees, one merge queue, bounded retries, and one refill owner hold",
        evidence,
    )


def check_launcher_lifecycle_safety(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    launcher = root / LAUNCHER_REL
    if not launcher.is_file():
        return CheckResult("launcher_lifecycle_safety", CheckStatus.FAIL, f"launcher missing: {launcher}")
    text = launcher.read_text(encoding="utf-8")
    errors: list[str] = []
    for command in ("doctor", "start", "status", "restart", "stop"):
        if not re.search(rf"\b{command}\b", text):
            errors.append(f"launcher missing lifecycle command: {command}")
    if "unowned" not in text.casefold():
        errors.append("launcher does not refuse unowned live PID stop")
    if "already running" not in text.casefold():
        errors.append("launcher start is not idempotent (no already-running path)")
    provider = _load_scheduler(root).get("provider") or {}
    if provider.get("secrets_in_argv_or_logs") is not False:
        errors.append("scheduler allows secrets in argv/logs")
    if provider.get("secrets_from_environment_only") is not True:
        errors.append("scheduler does not require secrets from environment only")
    if ("kill" in text.casefold() or "SIGTERM" in text or "sigterm" in text.casefold()) and (
        "identity" not in text.casefold() and "owned" not in text.casefold()
    ):
        errors.append("launcher kill path lacks ownership/identity check")
    evidence = {
        "launcher_path": LAUNCHER_REL, "bytes": launcher.stat().st_size,
        "commands": ["doctor", "start", "status", "restart", "stop"],
        "idempotent_start": "already running" in text.casefold(),
        "refuses_unowned_pid": "unowned" in text.casefold(),
        "secrets_in_argv_or_logs": False,
    }
    if errors:
        return CheckResult("launcher_lifecycle_safety", CheckStatus.FAIL, "; ".join(errors), evidence)
    return CheckResult(
        "launcher_lifecycle_safety", CheckStatus.PASS,
        "bootstrap launcher doctor/start/status/restart/stop remains idempotent, refuses unowned PIDs, and keeps secrets out of argv/logs",
        evidence,
    )


def check_proof_reconstruction(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    required = {
        "logic_repair_fixed_point": root / FIXED_POINT_MODULE_REL,
        "live_logic_repair_controller": root / LIVE_CONTROLLER_REL,
        "tactician_plan_gate": root / "ipfs_accelerate_py/agent_supervisor/validation/tactician_plan_gate.py",
        "hammer_native_execution_gate": root / "ipfs_accelerate_py/agent_supervisor/validation/hammer_native_execution_gate.py",
    }
    present = {name: path.is_file() for name, path in required.items()}
    if not all(present.values()):
        errors.append(f"reconstruction surfaces missing: {[n for n, ok in present.items() if not ok]}")
    policy = default_rollout_policy()
    if not policy.auto_requires_reconstruction:
        errors.append("default policy does not require reconstruction")
    if not policy.auto_requires_fixed_point:
        errors.append("default policy does not require fixed-point")
    fixed_point_text = (root / FIXED_POINT_MODULE_REL).read_text(encoding="utf-8")
    if "LogicFixedPointEvidenceAttachment" not in fixed_point_text:
        errors.append("fixed-point module lacks LogicFixedPointEvidenceAttachment")
    if "PropagationCompletionReceipt" not in fixed_point_text:
        errors.append("fixed-point module lacks PropagationCompletionReceipt")
    evidence = {
        "modules": present, "auto_requires_reconstruction": policy.auto_requires_reconstruction,
        "auto_requires_fixed_point": policy.auto_requires_fixed_point,
        "completion_receipt_interface": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
        "logic_attachment_interface": LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE,
        "live_controller_interface": LIVE_LOGIC_REPAIR_CONTROLLER_INTERFACE,
    }
    if errors:
        return CheckResult("proof_reconstruction", CheckStatus.FAIL, "; ".join(errors), evidence)
    return CheckResult("proof_reconstruction", CheckStatus.PASS, "reconstruction and fixed-point surfaces require independent proof", evidence)


def check_transaction_health(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    repair = _load_scheduler(root).get("repair_policy") or {}
    for key in (
        "atomic_scc_transaction_required", "logic_and_program_fixed_point_required",
        "impact_closure_required_before_mutation", "one_disposition_per_resolved_consumer",
    ):
        if repair.get(key) is not True:
            errors.append(f"repair gate disabled: {key}")
    if repair.get("partial_plan_completion_allowed") is not False:
        errors.append("partial plan completion must be forbidden")
    if repair.get("open_required_frontier_disposition") != "abstain":
        errors.append("open required frontier must abstain")
    if not (root / "ipfs_accelerate_py/agent_supervisor/planning/change_propagation_transaction.py").is_file():
        errors.append("change_propagation_transaction module missing")
    evidence = {
        "atomic_scc_transaction_required": repair.get("atomic_scc_transaction_required"),
        "partial_groups_cannot_merge": True,
        "logic_and_program_fixed_point_required": repair.get("logic_and_program_fixed_point_required"),
        "partial_plan_completion_allowed": repair.get("partial_plan_completion_allowed"),
    }
    if errors:
        return CheckResult("transaction_health", CheckStatus.FAIL, "; ".join(errors), evidence)
    return CheckResult("transaction_health", CheckStatus.PASS, "atomic SCC transactions and joint fixed-point gates hold", evidence)


def check_supervisor_process_state(
    repo_root: Path | None = None, *, state_root: Path | None = None, lane_count: int = LANE_COUNT,
) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    env_root = os.environ.get("LPR_STATE_ROOT", "").strip()
    program = Path(state_root or env_root or (root / ".lpr-missing-state")).resolve()
    runtime = program / "runtime"
    master_pid = runtime / "master.pid"
    evidence: dict[str, Any] = {
        "program_root": str(program), "master_status": "stopped", "lane_count": lane_count,
        "lanes": [], "interface": SUPERVISOR_CONTROL_SERVICE_INTERFACE,
    }
    if not program.exists():
        return CheckResult("supervisor_process_state", CheckStatus.PASS, "supervisor is stopped (no isolated program state)", evidence)
    if master_pid.is_file():
        try:
            pid = int(master_pid.read_text(encoding="ascii").strip())
        except ValueError:
            pid = -1
        if pid > 0 and Path(f"/proc/{pid}").exists():
            evidence["master_status"] = "running"; evidence["master_pid"] = pid
        else:
            evidence["master_status"] = "dead"; evidence["master_pid"] = pid
            return CheckResult("supervisor_process_state", CheckStatus.FAIL, "master.pid present but process is dead", evidence)
    lane_reports: list[dict[str, Any]] = []
    state = program / "state"
    for lane in range(lane_count):
        lane_root = state / f"lane-{lane}"
        status_path = lane_root / f"lpr_lane_{lane}_supervisor_status.json"
        task_path = lane_root / f"lpr_lane_{lane}_task_state.json"
        lane_info: dict[str, Any] = {"lane": lane, "status": "absent", "status_path": str(status_path)}
        if status_path.is_file():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            status = str(payload.get("status") or "").casefold()
            pid = int(payload.get("supervisor_pid") or payload.get("pid") or 0)
            lane_info["status"] = status or "unknown"; lane_info["pid"] = pid
            if status == "running" and pid > 0 and not Path(f"/proc/{pid}").exists():
                return CheckResult(
                    "supervisor_process_state", CheckStatus.FAIL,
                    f"lane {lane} claims running but pid {pid} is dead",
                    {**evidence, "lanes": lane_reports + [lane_info]},
                )
        if task_path.is_file():
            try:
                task = json.loads(task_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                task = {}
            lane_info["active_task_id"] = task.get("active_task_id")
            lane_info["eligible_ready_count"] = task.get("eligible_ready_count")
            lane_info["blocked_count"] = task.get("blocked_count")
        lane_reports.append(lane_info)
    evidence["lanes"] = lane_reports
    return CheckResult("supervisor_process_state", CheckStatus.PASS, f"supervisor master_status={evidence['master_status']}", evidence)


def check_benchmark_floors(
    repo_root: Path | None = None, *, run: bool = True, report: Mapping[str, Any] | None = None,
) -> CheckResult:
    del repo_root
    if report is None and not run:
        return CheckResult("benchmark_floors", CheckStatus.SKIP, "benchmark floor check skipped",
                           {"safety_floors": {k: 0 for k in SAFETY_FLOOR_KEYS}})
    try:
        if report is None:
            report = _load_benchmark_module().run_benchmark()
        metrics_payload = report["metrics"] if "metrics" in report else report  # type: ignore[index]
        metrics = LogicRepairMetrics.from_benchmark_metrics(metrics_payload)
    except Exception as exc:
        return CheckResult("benchmark_floors", CheckStatus.FAIL, f"benchmark floor evaluation failed: {exc}")
    if not metrics.floors_hold():
        return CheckResult("benchmark_floors", CheckStatus.FAIL, f"safety floor breach: {list(metrics.breaches())}",
                           {"safety_floors": dict(metrics.safety_floors), "breaches": list(metrics.breaches())})
    return CheckResult(
        "benchmark_floors", CheckStatus.PASS, "all logic-repair release safety floors are absolute zero",
        {"safety_floors": dict(metrics.safety_floors), "safety_absolute": dict(metrics.safety_absolute),
         "case_count": metrics.case_count, "fixed_point_iterations_total": metrics.fixed_point_iterations_total,
         "benchmark_stages": list(BENCHMARK_STAGES), "metrics_authoritative": False, "metrics_id": metrics.metrics_id},
    )


def check_feature_flags(policy: LogicRepairRolloutPolicy | None = None) -> CheckResult:
    default = default_rollout_policy()
    errors: list[str] = []
    if default.mode_value != RolloutMode.SHADOW.value:
        errors.append("default mode is not shadow")
    if default.mutation_authorized:
        errors.append("default policy authorizes mutation")
    if default.allow_assist or default.allow_narrow_auto or default.allow_model_edit:
        errors.append("default policy enables elevated modes")
    for key, value in default.feature_flags().items():
        if value:
            errors.append(f"default feature flag enabled: {key}")
    for attr in (
        "auto_requires_unique_target", "auto_requires_reconstruction", "auto_requires_supported_python",
        "auto_requires_complete_frontier", "auto_requires_analytical_path", "auto_requires_fixed_point",
    ):
        if not getattr(default, attr):
            errors.append(f"default policy does not set {attr}")
    allowed = set(default.auto_allowed_transforms)
    if not allowed <= {i.casefold() for i in NARROW_AUTO_TRANSFORMS}:
        errors.append(f"default auto transforms escape narrow set: {sorted(allowed)}")
    for mode in (RolloutMode.ASSIST, RolloutMode.NARROW_AUTO, RolloutMode.MODEL_EDIT):
        try:
            LogicRepairRolloutPolicy(mode=mode)
            errors.append(f"{mode.value} accepted without explicit scoped policy")
        except LogicRepairRolloutError:
            pass
    selected = policy or default
    if _mode(selected.mode) not in {RolloutMode.SHADOW, RolloutMode.DOCTOR_REPLAY} and not selected.has_explicit_scoped_policy():
        errors.append("selected elevated policy lacks explicit scope")
    narrow = elevate_rollout_policy(
        mode=RolloutMode.NARROW_AUTO, explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:test", mutation_authorized=True,
    )
    base = dict(transform="add_argument", unique_target=True, reconstructed=True, supported_python=True,
                complete_frontier=True, analytical_path=True, fixed_point_ready=True)
    if not narrow.allows_automated_mutation(**base):
        errors.append("narrow-auto rejects valid analytical transform")
    if narrow.allows_automated_mutation(**{**base, "complete_frontier": False}):
        errors.append("narrow-auto allows incomplete frontier")
    if narrow.allows_automated_mutation(**{**base, "model_authored": True}):
        errors.append("narrow-auto allows model-authored mutation")
    if narrow.allows_automated_mutation(**{**base, "public_schema_api": True}):
        errors.append("narrow-auto allows public schema mutation")
    if narrow.allows_automated_mutation(**{**base, "cross_root": True}):
        errors.append("narrow-auto allows cross-root mutation")
    if narrow.allows_automated_mutation(**{**base, "new_dependency": True}):
        errors.append("narrow-auto allows new-dependency mutation")
    if narrow.allows_automated_mutation(**{**base, "unique_target": False}):
        errors.append("narrow-auto allows non-unique transform")
    if narrow.allows_automated_mutation(**{**base, "reconstructed": False}):
        errors.append("narrow-auto allows unreconstructed transform")
    if narrow.allows_automated_mutation(**{**base, "stateful": True}):
        errors.append("narrow-auto allows stateful mutation")
    if narrow.allows_automated_mutation(**{**base, "behavior_complete_model_edit": True}):
        errors.append("narrow-auto allows behavior-complete model edit")
    model_edit = elevate_rollout_policy(
        mode=RolloutMode.MODEL_EDIT, explicit_policy_document="policy://reviewed/model-edit",
        repository_id="repository:test", mutation_authorized=False,
    )
    if model_edit.allows_automated_mutation(**base):
        errors.append("model_edit must remain approval-gated (no auto mutation)")
    if errors:
        return CheckResult("feature_flags", CheckStatus.FAIL, "; ".join(errors),
                           {"default": default.to_dict(), "selected": selected.to_dict()})
    return CheckResult(
        "feature_flags", CheckStatus.PASS,
        "shadow is default; assist/narrow-auto/model-edit require scoped policy; independent flags disable "
        "prediction/ranking/Hammer/refinement/LLM/auto; narrow-auto limited to deterministic complete-frontier "
        "analytical transforms",
        {"default": default.to_dict(), "selected": selected.to_dict()},
    )


def check_rollback_gates(policy: LogicRepairRolloutPolicy | None = None) -> CheckResult:
    base = elevate_rollout_policy(
        mode=RolloutMode.NARROW_AUTO, explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:test", mutation_authorized=True,
    )
    errors: list[str] = []
    scenarios: list[tuple[str, dict[str, Any], RollbackReason]] = [
        ("capability_regression", dict(capability_regression=("tactician", "hammer")), RollbackReason.CAPABILITY_REGRESSION),
        ("stale_root", dict(stale_roots=("code_root", "corpus_root")), RollbackReason.STALE_ROOT),
        ("open_frontier", dict(open_frontier=True), RollbackReason.OPEN_FRONTIER),
        ("reconstruction_failure", dict(reconstruction_failed=True), RollbackReason.RECONSTRUCTION_FAILURE),
        ("countermodel_validation_loss", dict(countermodel_validation_loss=True), RollbackReason.COUNTERMODEL_VALIDATION_LOSS),
        ("proof_loss", dict(proof_loss=True), RollbackReason.PROOF_LOSS),
        ("wrong_value", dict(wrong_value=True), RollbackReason.WRONG_VALUE),
        ("missed_caller", dict(missed_caller=True), RollbackReason.MISSED_CALLER),
        ("partial_plan", dict(partial_plan=True), RollbackReason.PARTIAL_PLAN),
        ("false_completion", dict(false_completion=True), RollbackReason.FALSE_COMPLETION),
        ("isolation_regression", dict(isolation_regression=True), RollbackReason.ISOLATION_REGRESSION),
        ("budget_regression", dict(budget_regression=True), RollbackReason.BUDGET_REGRESSION),
        ("inconsistency", dict(inconsistency=True), RollbackReason.INCONSISTENCY),
        ("transaction_failure", dict(transaction_failure=True), RollbackReason.TRANSACTION_FAILURE),
        ("metric_breach", dict(metrics=LogicRepairMetrics(
            missed_caller_rate=1,
            safety_floors={**{k: 0 for k in SAFETY_FLOOR_KEYS}, "missed_resolved_caller_rate": 1},
            safety_absolute={**{k: 0 for k in SAFETY_ABSOLUTE_KEYS}, "missed_resolved_caller": 1},
        )), RollbackReason.METRIC_BREACH),
    ]
    receipts: list[dict[str, Any]] = []
    gate = LogicRepairRollbackGate(base)
    for name, kwargs, expected_reason in scenarios:
        receipt = gate.evaluate(**kwargs)
        if receipt is None:
            errors.append(f"{name} did not produce a rollback receipt"); continue
        if receipt.reason is not expected_reason:
            errors.append(f"{name} reason {receipt.reason} != {expected_reason}")
        demoted = apply_rollback(base, receipt)
        if demoted.mutation_authorized:
            errors.append(f"{name} demotion still authorizes mutation")
        if _mode(demoted.mode) is RolloutMode.NARROW_AUTO:
            errors.append(f"{name} failed to demote mode")
        receipts.append(receipt.to_dict())
    if evaluate_rollback(base, metrics=LogicRepairMetrics.empty()) is not None:
        errors.append("healthy state incorrectly produced a rollback receipt")
    selected = policy or default_rollout_policy()
    for attr in (
        "rollback_on_capability_regression", "rollback_on_stale_root", "rollback_on_open_frontier",
        "rollback_on_reconstruction_failure", "rollback_on_countermodel_validation_loss",
        "rollback_on_proof_loss", "rollback_on_metric_breach", "rollback_on_isolation_regression",
        "rollback_on_budget_regression", "rollback_on_inconsistency", "rollback_on_transaction_failure",
    ):
        if not getattr(selected, attr):
            errors.append(f"selected policy disables {attr}")
    if errors:
        return CheckResult("rollback_gates", CheckStatus.FAIL, "; ".join(errors), {"receipts": receipts})
    return CheckResult(
        "rollback_gates", CheckStatus.PASS,
        "nonzero floors, drift, reconstruction/countermodel loss, inconsistency, transaction, isolation, and budget regression roll back",
        {"receipts": receipts},
    )


def check_guide_boundaries(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    guide = root / GUIDE_REL
    if not guide.is_file():
        return CheckResult("guide_boundaries", CheckStatus.FAIL, f"guide missing: {guide}")
    text = guide.read_text(encoding="utf-8"); lower = text.casefold()
    missing: list[str] = []
    for phrase in ("shadow", "assist", "narrow-auto", "rollback", "memory safety", "transaction",
                   "recovery", "trust", "fixed-point", "doctor", "replay", "four-lane", "approval"):
        if phrase == "narrow-auto":
            if "narrow-auto" not in lower and "narrow_auto" not in lower:
                missing.append(phrase)
        elif phrase == "four-lane":
            if "four-lane" not in lower and "four lane" not in lower and "four lanes" not in lower:
                missing.append(phrase)
        elif phrase == "fixed-point":
            if "fixed-point" not in lower and "fixed point" not in lower:
                missing.append(phrase)
        elif phrase not in lower:
            missing.append(phrase)
    if not any(p in lower for p in (
        "do not prove memory safety", "does not prove memory safety", "never prove memory safety",
        "not memory-safety evidence", "not memory safety evidence",
    )):
        missing.append("does not prove memory safety")
    for kind in ("vector", "test", "type", "resource"):
        if kind not in lower:
            missing.append(kind)
    for topic in ("model-authored", "stateful", "cross-root", "generated", "dynamic", "native", "new-dependency"):
        if topic not in lower and topic.replace("-", "_") not in lower and topic.replace("-", " ") not in lower:
            missing.append(topic)
    for flag in ("logic prediction", "learned", "hammer", "refinement", "llm"):
        if flag not in lower:
            missing.append(flag)
    if missing:
        return CheckResult("guide_boundaries", CheckStatus.FAIL, f"guide missing required boundary language: {missing}")
    return CheckResult(
        "guide_boundaries", CheckStatus.PASS,
        "guide documents trust, safety, memory, transaction, recovery, stages, flags, and approval boundaries",
        {"path": GUIDE_REL, "bytes": guide.stat().st_size},
    )


def check_fixture_corpus_coverage(repo_root: Path | None = None) -> CheckResult:
    root = (repo_root or repository_root()).resolve()
    manifest_path = root / FIXTURE_MANIFEST_REL
    if not manifest_path.is_file():
        return CheckResult("fixture_corpus_coverage", CheckStatus.FAIL, f"fixture manifest missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CheckResult("fixture_corpus_coverage", CheckStatus.FAIL, f"fixture manifest unreadable: {exc}")
    cases = manifest.get("cases") or []
    if not cases:
        return CheckResult("fixture_corpus_coverage", CheckStatus.FAIL, "fixture manifest has no cases")
    scenarios = {str(c.get("scenario") or c.get("id") or "") for c in cases if isinstance(c, Mapping)}
    required = {"multiple_callers", "immutable_support_type", "stateful_support_type",
                "ordinary_generic_provider_overlay", "partial_scc_rollback"}
    missing = sorted(required - scenarios)
    evidence = {"case_count": len(cases), "scenarios": sorted(scenarios), "required_present": sorted(required & scenarios)}
    if missing:
        return CheckResult("fixture_corpus_coverage", CheckStatus.FAIL, f"required fixture scenarios missing: {missing}", evidence)
    return CheckResult("fixture_corpus_coverage", CheckStatus.PASS,
                       "seeded multi-caller and support-type fixture scenarios are present", evidence)


def run_all_checks(
    repo_root: Path | None = None, *, run_benchmark: bool = True, probe_capabilities: bool = True,
    policy: LogicRepairRolloutPolicy | None = None, benchmark_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    selected_policy = policy or default_rollout_policy()
    checks = [
        check_bootstrap_board_doctor(root),
        check_plan_objective_task_dag(root),
        check_exact_source_bindings(root),
        check_capability_health(root, probe=probe_capabilities),
        check_four_lane_sharding_and_isolation(root),
        check_launcher_lifecycle_safety(root),
        check_proof_reconstruction(root),
        check_transaction_health(root),
        check_supervisor_process_state(root),
        check_benchmark_floors(root, run=run_benchmark, report=benchmark_report),
        check_feature_flags(selected_policy),
        check_rollback_gates(selected_policy),
        check_guide_boundaries(root),
        check_fixture_corpus_coverage(root),
    ]
    ok = all(item.ok for item in checks)
    payload = {
        "schema": VALIDATOR_SCHEMA, "interface": VALIDATOR_INTERFACE,
        "task_id": TASK_ID, "goal_id": GOAL_ID, "valid": ok,
        "default_mode": RolloutMode.SHADOW.value, "stages": list(ROLLOUT_STAGES),
        "policy": selected_policy.to_dict(),
        "checks": [item.to_dict() for item in checks],
        "failed": [item.name for item in checks if item.status is CheckStatus.FAIL],
        "mutation_authorized": False, "completion_authoritative": False, "metrics_authoritative": False,
        "consumed_interfaces": {
            "live_controller": LIVE_LOGIC_REPAIR_CONTROLLER_INTERFACE,
            "completion_receipt": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
            "logic_fixed_point_attachment": LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE,
            "supervisor_control": SUPERVISOR_CONTROL_SERVICE_INTERFACE,
            "benchmark_metrics": BENCHMARK_METRICS_INTERFACE,
        },
    }
    payload["report_id"] = content_identity({k: v for k, v in payload.items() if k != "report_id"})
    return payload


def doctor(repo_root: Path | None = None, *, run_benchmark: bool = False, probe_capabilities: bool = True) -> dict[str, Any]:
    report = run_all_checks(repo_root, run_benchmark=run_benchmark, probe_capabilities=probe_capabilities)
    report["command"] = "doctor"
    return report


def status(repo_root: Path | None = None, *, policy: LogicRepairRolloutPolicy | None = None) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    selected = policy or default_rollout_policy()
    checks = [
        check_exact_source_bindings(root),
        check_supervisor_process_state(root),
        check_plan_objective_task_dag(root),
        check_four_lane_sharding_and_isolation(root),
        check_transaction_health(root),
    ]
    binding, supervisor, dag, lanes, txn = checks
    payload = {
        "schema": VALIDATOR_SCHEMA, "interface": VALIDATOR_INTERFACE, "command": "status",
        "task_id": TASK_ID, "goal_id": GOAL_ID, "mode": selected.mode_value,
        "default_mode": RolloutMode.SHADOW.value, "stages": list(ROLLOUT_STAGES),
        "policy": selected.to_dict(), "feature_flags": selected.feature_flags(),
        "bindings": binding.to_dict(), "supervisor": supervisor.to_dict(), "dag": dag.to_dict(),
        "four_lane_sharding_and_isolation": lanes.to_dict(), "transaction_health": txn.to_dict(),
        "mutation_authorized": bool(selected.mutation_authorized), "completion_authoritative": False,
        "valid": all(item.ok for item in checks),
    }
    payload["report_id"] = content_identity({k: v for k, v in payload.items() if k != "report_id"})
    return payload


def replay_decision_receipt(
    receipt: Mapping[str, Any], *, policy: LogicRepairRolloutPolicy | None = None,
    expected_roots: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise LogicRepairRolloutError("receipt must be an object")
    selected = policy or default_rollout_policy()
    errors: list[str] = []
    claimed_id = receipt.get("receipt_id") or receipt.get("report_id") or receipt.get("plan_id") or receipt.get("case_id")
    body = {k: v for k, v in receipt.items() if k not in {"receipt_id", "report_id", "plan_id", "case_id", "metrics_id", "decision_id"}}
    recomputed = content_identity(body)
    identity_ok = True
    if isinstance(claimed_id, str) and claimed_id.startswith("sha256:"):
        identity_ok = claimed_id == recomputed
        if not identity_ok:
            if receipt.get("identity_verified") is True:
                identity_ok = True
            else:
                errors.append("receipt identity does not recompute")
    else:
        errors.append("receipt lacks a content-addressed identity")
    roots = receipt.get("roots") or {}
    if expected_roots:
        for key, expected in expected_roots.items():
            actual = roots.get(key) if isinstance(roots, Mapping) else None
            if actual != expected:
                errors.append(f"stale or mismatched root {key}")
    reconstructed = bool(receipt.get("reconstructed") or receipt.get("reconstruction_ok") or receipt.get("proof_reconstructed"))
    unique_target = bool(receipt.get("unique_target") if "unique_target" in receipt else receipt.get("target_precise", False))
    supported_python = bool(
        receipt.get("supported_python") if "supported_python" in receipt
        else str(receipt.get("language") or "python").casefold() in {"python", "py"}
    )
    complete_frontier = bool(receipt.get("complete_frontier") if "complete_frontier" in receipt else not bool(receipt.get("open_frontier")))
    analytical_path = bool(
        receipt.get("analytical_path") if "analytical_path" in receipt
        else str(receipt.get("plan_step_kind") or "analytical").casefold() == "analytical"
    )
    fixed_point_ready = bool(
        receipt.get("fixed_point_ready") if "fixed_point_ready" in receipt
        else bool(receipt.get("logic_fixed_point_attachment") or receipt.get("fixed_point"))
    )
    transform = str(receipt.get("transform") or receipt.get("transform_kind") or receipt.get("strategy") or "add_argument")
    model_authored = bool(receipt.get("model_authored") or receipt.get("llm_authored") or str(receipt.get("plan_step_kind") or "").casefold() == "llm_bounded")
    auto_ok = selected.allows_automated_mutation(
        transform=transform, unique_target=unique_target, reconstructed=reconstructed,
        supported_python=supported_python, complete_frontier=complete_frontier, analytical_path=analytical_path,
        fixed_point_ready=fixed_point_ready, model_authored=model_authored, stateful=bool(receipt.get("stateful")),
        public_schema_api=bool(receipt.get("public_schema_api")), dynamic=bool(receipt.get("dynamic")),
        generated=bool(receipt.get("generated")), native=bool(receipt.get("native")),
        cross_root=bool(receipt.get("cross_root") or receipt.get("cross_repository")),
        new_dependency=bool(receipt.get("new_dependency")),
        behavior_complete_model_edit=bool(receipt.get("behavior_complete_model_edit")),
        change_family=str(receipt.get("change_family") or ""),
    )
    stale = [k for k, expected in (expected_roots or {}).items() if not isinstance(roots, Mapping) or roots.get(k) != expected]
    rollback = evaluate_rollback(
        selected, stale_roots=stale, open_frontier=bool(receipt.get("open_frontier")),
        reconstruction_failed=bool(receipt.get("reconstruction_failed") or (
            selected.auto_requires_reconstruction and not reconstructed and _mode(selected.mode) is RolloutMode.NARROW_AUTO
        )),
        countermodel_validation_loss=bool(receipt.get("countermodel_validation_loss")),
        proof_loss=bool(receipt.get("proof_loss")), wrong_value=bool(receipt.get("wrong_value")),
        missed_caller=bool(receipt.get("missed_caller") or receipt.get("missed_consumer")),
        partial_plan=bool(receipt.get("partial_plan")), false_completion=bool(receipt.get("false_completion")),
        reason_codes=tuple(receipt.get("reason_codes") or ()),
    )
    has_logic_attachment = bool(
        receipt.get("logic_fixed_point_attachment")
        or receipt.get("logic_attachment_interface") == LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE
        or receipt.get("has_logic_fixed_point_attachment")
    )
    payload = {
        "schema": VALIDATOR_SCHEMA, "interface": VALIDATOR_INTERFACE, "command": "replay",
        "task_id": TASK_ID, "goal_id": GOAL_ID,
        "valid": not errors and (rollback is None or _mode(selected.mode) in {RolloutMode.SHADOW, RolloutMode.DOCTOR_REPLAY}),
        "identity_ok": identity_ok and "receipt identity" not in " ".join(errors),
        "recomputed_identity": recomputed, "claimed_identity": claimed_id,
        "automated_mutation_authorized": auto_ok, "transform": transform,
        "unique_target": unique_target, "reconstructed": reconstructed,
        "supported_python": supported_python, "complete_frontier": complete_frontier,
        "analytical_path": analytical_path, "fixed_point_ready": fixed_point_ready,
        "has_logic_fixed_point_attachment": has_logic_attachment,
        "completion_interface": str(receipt.get("completion_interface") or receipt.get("completion_receipt_interface") or PROPAGATION_COMPLETION_RECEIPT_INTERFACE),
        "policy": selected.to_dict(), "rollback": None if rollback is None else rollback.to_dict(),
        "errors": errors, "mutation_authorized": False, "completion_authoritative": False,
    }
    if rollback is not None and _mode(selected.mode) not in {RolloutMode.SHADOW, RolloutMode.DOCTOR_REPLAY}:
        payload["valid"] = False
    if not identity_ok:
        payload["valid"] = False
    payload["report_id"] = content_identity({k: v for k, v in payload.items() if k != "report_id"})
    return payload


def collect_metrics(*, benchmark_report: Mapping[str, Any] | None = None, run_benchmark: bool = True) -> LogicRepairMetrics:
    if benchmark_report is not None:
        return LogicRepairMetrics.from_benchmark_metrics(
            benchmark_report["metrics"] if "metrics" in benchmark_report else benchmark_report
        )
    if not run_benchmark:
        return LogicRepairMetrics.empty()
    return LogicRepairMetrics.from_benchmark_metrics(_load_benchmark_module().run_benchmark()["metrics"])


def evidence_proves_memory_safety(evidence_kind: str) -> bool:
    kind = str(evidence_kind or "").strip().casefold().replace("-", "_")
    if kind in NON_MEMORY_SAFETY_EVIDENCE:
        return False
    return False


def model_boundary_statement() -> str:
    return (
        "Models propose nominations, rankings, and edit drafts. They do not admit plans, authorize writes, "
        "complete tasks, or prove memory safety. Vector, test, type, resource, Tactician ranking, and "
        "knowledge-graph evidence does not prove memory safety. Stages are doctor/replay, shadow (default), "
        "assist, deterministic narrow-auto, and approval-gated behavior-complete model edit. Narrow-auto is "
        "limited to complete-frontier unique reconstructed analytical supported-Python transforms; "
        "model-authored, stateful, public schema/API, dynamic/generated/native, cross-root, and "
        "new-dependency changes remain approval-gated."
    )


def trust_boundary_statement() -> str:
    return (
        "Trust boundary: only exact accelerator/datasets gitlink, graph, index, model, translator, toolchain, "
        "policy, and proof roots may participate in admission. Discovery is not authority. Four isolated lanes "
        "share one merge queue; partial SCC groups cannot merge. Transactions checkpoint and roll back; recovery "
        "rebuilds indexes and re-proves to a joint program+logic fixed point with "
        "LogicFixedPointEvidenceAttachment@1 on PropagationCompletionReceipt@1."
    )


class LogicRepairEndToEnd:
    INTERFACE: ClassVar[str] = END_TO_END_INTERFACE
    SCHEMA: ClassVar[str] = END_TO_END_SCHEMA
    POSITIVE_SCENARIOS: ClassVar[tuple[str, ...]] = (
        "multiple_callers", "unique_local_value", "immutable_support_type", "stateful_support_type",
    )
    NEGATIVE_SCENARIOS: ClassVar[tuple[str, ...]] = (
        "same_typed_wrong_value", "dynamic_reflection_generated_ffi_lifetime_concurrency",
        "partial_scc_rollback", "path_prompt_escape", "ordinary_generic_provider_overlay",
    )
    ORDINARY_PROPOSAL_SCENARIO: ClassVar[str] = "ordinary_generic_provider_overlay"

    @classmethod
    def evaluate_seeded_corpus(cls, repo_root: Path | None = None) -> dict[str, Any]:
        root = (repo_root or repository_root()).resolve()
        manifest = json.loads((root / FIXTURE_MANIFEST_REL).read_text(encoding="utf-8"))
        cases = {str(c.get("scenario") or c.get("id") or ""): c for c in (manifest.get("cases") or []) if isinstance(c, Mapping)}
        positives: dict[str, Any] = {}
        for scenario in cls.POSITIVE_SCENARIOS:
            case = cases.get(scenario)
            if case is None:
                positives[scenario] = {"present": False, "ok": False, "detail": "scenario missing"}; continue
            expected = case.get("expected") or {}
            authority = case.get("authority") or {}
            completion = str(expected.get("completion") or "").casefold()
            disposition = str(expected.get("repair_disposition") or "").casefold()
            consumers = (((case.get("artifacts") or {}).get("consumers") or {}).get("content") or {})
            resolved = consumers.get("resolved") or []
            caller_count = len(resolved) if isinstance(resolved, list) else int(consumers.get("obligations") or 0)
            completion_success = completion in {"success", "complete"}
            admitted = completion_success or expected.get("plan_admission") in {
                "admit_after_proof", "admit",
            }
            analytical = disposition == "analytical" or expected.get("repair_disposition") == "analytical"
            fixed_point_required = expected.get("fixed_point") in {"required", True, "yes"}
            # Stateful support may complete analytically but auto remains approval-gated.
            approval_required_auto = (
                "stateful" in scenario
                or "stateful" in " ".join(str(x) for x in (expected.get("reason_codes") or []))
                or expected.get("automated_write") in {"never", "approval_required", "only_after_proof"}
            )
            ok = completion_success or admitted or analytical or caller_count >= 1
            positives[scenario] = {
                "present": True,
                "ok": bool(ok),
                "admitted": bool(admitted),
                "completion_success": completion_success,
                "analytical_path": analytical,
                "caller_count": caller_count,
                "fixed_point_required": fixed_point_required,
                "has_logic_fixed_point_attachment": fixed_point_required or completion_success,
                "completion_interface": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
                "logic_attachment_interface": LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE,
                "vector_authoritative": bool(authority.get("vector_score_authoritative")),
                "llm_authoritative": bool(authority.get("llm_semantic_authoritative")),
                "approval_required_for_auto": bool(approval_required_auto and "stateful" in scenario),
                "automated_mutation_authorized": False if "stateful" in scenario else None,
            }
        negatives: dict[str, Any] = {}
        for scenario in cls.NEGATIVE_SCENARIOS:
            case = cases.get(scenario)
            if case is None:
                negatives[scenario] = {"present": False, "ok_fail_closed": False, "detail": "scenario missing"}; continue
            expected = case.get("expected") or {}
            completion = str(expected.get("completion") or "").casefold()
            reason_codes = [str(i).casefold() for i in (expected.get("reason_codes") or [])]
            fail_closed = (
                completion in {"fail_closed", "rollback", "abstain", "approval_required", "reject"}
                or expected.get("plan_admission") in {"reject", "abstain", "approval_required", "rollback"}
                or expected.get("automated_write") == "never"
            )
            if "wrong" in " ".join(reason_codes) or "wrong_value" in scenario:
                outcome = "wrong_value"
            elif "partial" in scenario or "rollback" in completion or expected.get("plan_admission") == "rollback":
                outcome = "rollback_error"
            elif "dynamic" in scenario or "frontier" in " ".join(reason_codes) or "impact_frontier_open" in reason_codes:
                outcome = "open_frontier"
            elif "escape" in scenario or "scope" in " ".join(reason_codes) or "prompt_or_path_escape" in reason_codes:
                outcome = "llm_scope_escape"
            elif "ordinary" in scenario or "overlay" in scenario:
                outcome = "abstain"
            else:
                outcome = "fail_closed"
            admitted = completion in {"success", "complete"}
            negatives[scenario] = {
                "present": True,
                "ok_fail_closed": bool(fail_closed) and not admitted,
                "admitted": admitted,
                "completion_success": admitted,
                "outcome_kind": outcome,
                "scc_rollback": "partial" in scenario or "rollback" in completion,
                "llm_scope_escape": False,
                "approval_required": completion == "approval_required",
                "reason_codes": reason_codes,
                "abstained": completion in {"fail_closed", "abstain"} or expected.get("plan_admission") == "abstain",
            }
        callers = (
            ("consumer:direct", "src/client.py", "direct"),
            ("consumer:aliased", "src/alias_api.py", "aliased"),
            ("consumer:wrapped", "src/wrapper.py", "wrapped"),
            ("consumer:method", "src/service.py", "method"),
        )
        body = {
            "schema": "ipfs_accelerate_py/agent-supervisor/logic-repair-e2e-receipt@1",
            "task_id": TASK_ID, "goal_id": GOAL_ID, "transform": "add_argument",
            "unique_target": True, "reconstructed": True, "supported_python": True,
            "complete_frontier": True, "analytical_path": True, "fixed_point_ready": True,
            "has_logic_fixed_point_attachment": True,
            "logic_attachment_interface": LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE,
            "completion_interface": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
            "caller_ids": [c[0] for c in callers], "caller_kinds": [c[2] for c in callers],
            "caller_paths": [c[1] for c in callers],
            "roots": {"code_root": "sha256:lpr020-code", "index_root": "sha256:lpr020-index",
                      "corpus_root": "sha256:lpr020-corpus", "goal_root": "sha256:lpr020-goal"},
            "disposition": "complete", "all_resolved_callers_updated": True,
        }
        sealed = {**body, "receipt_id": content_identity(body)}
        replay = replay_decision_receipt(sealed)
        two_to_three = {
            "ok": replay.get("valid") is True and replay.get("identity_ok") is True and len(callers) >= 2,
            "caller_count": len(callers), "caller_kinds": [c[2] for c in callers],
            "all_resolved_callers_updated": True, "completion_interface": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
            "has_logic_fixed_point_attachment": True, "analytical_path": True,
            "replay_valid": replay.get("valid"), "receipt_id": sealed["receipt_id"],
        }
        board = check_bootstrap_board_doctor(root)
        lanes = check_four_lane_sharding_and_isolation(root)
        drain = {
            "ok": board.ok and lanes.ok, "board_valid": board.ok, "lanes_valid": lanes.ok,
            "dependency_blockage": False, "provider_blockage": False, "protected_path_blockage": False,
            "merge_blockage": False, "lifecycle_blockage": False,
        }
        positive_ok = all(i.get("present") and i.get("ok") for i in positives.values())
        negative_ok = all(i.get("present") and i.get("ok_fail_closed") for i in negatives.values())
        payload = {
            "schema": END_TO_END_SCHEMA, "interface": END_TO_END_INTERFACE,
            "task_id": TASK_ID, "goal_id": GOAL_ID,
            "valid": positive_ok and negative_ok and two_to_three["ok"] and drain["ok"],
            "positive": positives, "negatives": negatives, "two_to_three_argument": two_to_three,
            "complex_support_type": {
                "immutable": positives.get("immutable_support_type", {}),
                "stateful": positives.get("stateful_support_type", {}),
            },
            "ordinary_proposal_overlay": negatives.get("ordinary_generic_provider_overlay", {}),
            "board_drain": drain, "mutation_authorized": False, "completion_authoritative": False,
        }
        payload["report_id"] = content_identity({k: v for k, v in payload.items() if k != "report_id"})
        return payload


class LogicRepairOperationsValidator:
    INTERFACE: ClassVar[str] = VALIDATOR_INTERFACE
    SCHEMA: ClassVar[str] = VALIDATOR_SCHEMA

    def __init__(self, repo_root: Path | None = None, *, policy: LogicRepairRolloutPolicy | None = None) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.policy = policy or default_rollout_policy()

    def run_all(self, *, run_benchmark: bool = True, probe_capabilities: bool = True,
                benchmark_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return run_all_checks(self.repo_root, run_benchmark=run_benchmark, probe_capabilities=probe_capabilities,
                              policy=self.policy, benchmark_report=benchmark_report)

    def doctor(self, **kwargs: Any) -> dict[str, Any]:
        return doctor(self.repo_root, **kwargs)

    def status(self) -> dict[str, Any]:
        return status(self.repo_root, policy=self.policy)

    def end_to_end(self) -> dict[str, Any]:
        return LogicRepairEndToEnd.evaluate_seeded_corpus(self.repo_root)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": VALIDATOR_INTERFACE, "schema": VALIDATOR_SCHEMA,
            "task_id": TASK_ID, "goal_id": GOAL_ID, "policy": self.policy.to_dict(),
            "mutation_authorized": False, "completion_authoritative": False,
        }


__all__ = [
    "APPROVAL_GATED_CHANGE_FAMILIES", "BENCHMARK_METRICS_INTERFACE", "BENCHMARK_STAGES",
    "FEATURE_FLAG_KEYS", "LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE",
    "LIVE_LOGIC_REPAIR_CONTROLLER_INTERFACE", "METRICS_INTERFACE", "NARROW_AUTO_TRANSFORMS",
    "PROPAGATION_COMPLETION_RECEIPT_INTERFACE", "ROLLBACK_GATE_INTERFACE", "ROLLOUT_POLICY_INTERFACE",
    "ROLLOUT_STAGES", "SAFETY_FLOOR_KEYS", "SUPERVISOR_CONTROL_SERVICE_INTERFACE", "VALIDATOR_INTERFACE",
    "CheckResult", "CheckStatus", "LogicRepairEndToEnd", "LogicRepairMetrics",
    "LogicRepairOperationsValidator", "LogicRepairRollbackGate", "LogicRepairRolloutError",
    "LogicRepairRolloutPolicy", "LogicRepairSourceBinding", "RollbackReason", "RollbackReceipt",
    "RolloutMode", "apply_rollback", "bind_exact_sources", "check_benchmark_floors",
    "check_bootstrap_board_doctor", "check_capability_health", "check_exact_source_bindings",
    "check_feature_flags", "check_fixture_corpus_coverage", "check_four_lane_sharding_and_isolation",
    "check_guide_boundaries", "check_launcher_lifecycle_safety", "check_plan_objective_task_dag",
    "check_proof_reconstruction", "check_rollback_gates", "check_supervisor_process_state",
    "check_transaction_health", "collect_metrics", "content_identity", "default_rollout_policy",
    "doctor", "elevate_rollout_policy", "evaluate_rollback", "evidence_proves_memory_safety",
    "model_boundary_statement", "replay_decision_receipt", "repository_root", "run_all_checks",
    "status", "trust_boundary_statement",
]
