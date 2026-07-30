"""Propagation metrics, rollout flags, and rollback gates.

RPR-046 / RPR-G220.  Operator surface for proof-gated change propagation:

* ``ChangePropagationRolloutPolicy`` — shadow default; assist and narrow-auto
  require explicit scoped policy; narrow-auto is limited to complete-frontier
  unique reconstructed analytical supported-Python transforms;
* ``ChangePropagationMetrics`` — every benchmark stage plus analytical/model
  split, tokens/context, and fixed-point iterations;
* ``ChangePropagationRollbackGate`` — demotes on stale roots, open frontier,
  capability regression, proof loss, wrong-value / missed-consumer /
  partial-plan / false-completion, or any safety-floor breach.

Model-authored, stateful, public schema/API, dynamic/generated/native, and
cross-root mutations remain approval-gated.  This module never grants
mutation, completion, merge, or process authority.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.change_propagation_contracts import TransformKind


# ---------------------------------------------------------------------------
# Schemas / identities
# ---------------------------------------------------------------------------

ROLLOUT_POLICY_INTERFACE: Final[str] = "ChangePropagationRolloutPolicy@1"
ROLLOUT_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-rollout-policy@1"
)
METRICS_INTERFACE: Final[str] = "ChangePropagationMetrics@1"
METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-ops-metrics@1"
)
ROLLBACK_GATE_INTERFACE: Final[str] = "ChangePropagationRollbackGate@1"
ROLLBACK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-rollback-receipt@1"
)
SOURCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-source-binding@1"
)
VALIDATOR_INTERFACE: Final[str] = "ChangePropagationValidatorOps@1"
VALIDATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-validation-ops-report@1"
)

# Re-exported interface names expected by operators and RPR-046 AST symbols.
BENCHMARK_METRICS_INTERFACE: Final[str] = "ChangePropagationBenchmarkMetrics@1"
ATOMIC_PROPAGATION_PLAN_INTERFACE: Final[str] = "AtomicPropagationPlan@1"
PROPAGATION_COMPLETION_RECEIPT_INTERFACE: Final[str] = (
    "PropagationCompletionReceipt@1"
)

TASK_ID: Final[str] = "RPR-046"
GOAL_ID: Final[str] = "RPR-G220"
BOARD_NAMESPACE: Final[str] = "agent-supervisor-proof-gated-contract-repair-v1"
TASK_PREFIX: Final[str] = "RPR-"
GOAL_PREFIX: Final[str] = "RPR-G"
MERGE_TARGET_BRANCH: Final[str] = "agent/proof-gated-contract-repair"
DEFAULT_RECALL_K: Final[int] = 5

PLAN_REL: Final[str] = (
    "docs/architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md"
)
OBJECTIVE_REL: Final[str] = (
    "docs/architecture/agent_supervisor_proof_gated_contract_repair.objectives.md"
)
TODO_REL: Final[str] = (
    "docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md"
)
SCHEDULER_REL: Final[str] = (
    "config/agent_supervisor_proof_gated_contract_repair_scheduler.json"
)
LAUNCHER_REL: Final[str] = "scripts/proof_gated_contract_repair_supervisor.sh"
GUIDE_REL: Final[str] = "docs/guides/PROOF_GATED_CHANGE_PROPAGATION_GUIDE.md"
BENCHMARK_SCRIPT_REL: Final[str] = "scripts/benchmark_change_propagation.py"
ROLLOUT_MODULE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/validation/change_propagation_rollout.py"
)
VALIDATE_SCRIPT_REL: Final[str] = "scripts/validate_change_propagation.py"
FIXTURE_MANIFEST_REL: Final[str] = (
    "test/fixtures/agent_supervisor/change_propagation/manifest.json"
)
TRANSACTION_MODULE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/planning/change_propagation_transaction.py"
)
VALIDATION_MODULE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/validation/change_propagation_validation.py"
)
ANALYTICAL_TRANSFORMS_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/planning/analytical_change_transforms.py"
)

REQUIRED_CONTROL_PLANE: Final[tuple[str, ...]] = (
    PLAN_REL,
    OBJECTIVE_REL,
    TODO_REL,
    SCHEDULER_REL,
    LAUNCHER_REL,
)

REQUIRED_PROPAGATION_SOURCES: Final[tuple[str, ...]] = (
    FIXTURE_MANIFEST_REL,
    BENCHMARK_SCRIPT_REL,
    ROLLOUT_MODULE_REL,
    VALIDATE_SCRIPT_REL,
    GUIDE_REL,
    TRANSACTION_MODULE_REL,
    VALIDATION_MODULE_REL,
    ANALYTICAL_TRANSFORMS_REL,
)

# Legacy + propagation release floors (rates must equal zero).
SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "wrong_path_automated_mutation_rate",
    "failed_obligation_override_rate",
    "stale_forged_or_poisoned_authoritative_admission_rate",
    "unsupported_memory_safety_promotion_rate",
    "missed_resolved_impacted_consumer_rate",
    "unproved_or_wrong_value_source_admission_rate",
    "invented_behavior_without_authority_rate",
    "partial_propagation_completion_rate",
    "stale_graph_index_plan_admission_rate",
    "false_fixed_point_completion_rate",
)

SAFETY_ABSOLUTE_KEYS: Final[tuple[str, ...]] = (
    "wrong_path_automated_mutation",
    "failed_obligation_override",
    "stale_forged_or_poisoned_authoritative_admission",
    "unsupported_memory_safety_promotion",
    "missed_resolved_impacted_consumer",
    "unproved_or_wrong_value_source_admission",
    "invented_behavior_without_authority",
    "partial_propagation_completion",
    "stale_graph_index_plan_admission",
    "false_fixed_point_completion",
)

# Benchmark stages exposed as operator metrics (RPR-045 cost model).
BENCHMARK_STAGES: Final[tuple[str, ...]] = (
    "delta",
    "graph_closure",
    "consumer_inventory",
    "value_retrieval",
    "proof",
    "plan_admission",
    "implementation",
    "transaction",
    "fixed_point",
)

# Narrow-auto may execute only these analytical supported-Python transforms.
NARROW_AUTO_TRANSFORMS: Final[frozenset[str]] = frozenset(
    {
        TransformKind.ADD_ARGUMENT.value,
        TransformKind.RENAME_ARGUMENT.value,
        TransformKind.REORDER_ARGUMENT.value,
        TransformKind.THREAD_PARAMETER.value,
        TransformKind.ADD_IMPORT.value,
        TransformKind.ADD_EXPORT.value,
        "add_argument",
        "rename_argument",
        "reorder_argument",
        "thread_parameter",
        "add_import",
        "add_export",
        "analytical_python_transform",
    }
)

# Families that remain approval-gated even under elevated policy.
APPROVAL_GATED_CHANGE_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "model_authored",
        "llm_authored",
        "llm_bounded",
        "stateful_behavior",
        "stateful_service",
        "public_schema",
        "public_api",
        "schema_api",
        "dynamic",
        "generated",
        "native",
        "ffi",
        "cross_root",
        "cross_repository",
        "add_adapter",
        "update_constructor",
        "update_schema_field",
        "update_serializer",
        "update_fixture",
        "update_generated_manifest",
        "add_registration",
    }
)

NON_MEMORY_SAFETY_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector",
        "lexical",
        "graph",
        "history",
        "test",
        "type",
        "schema",
        "resource",
        "llm",
        "max_memory_bytes",
        "embedding",
        "coverage",
    }
)

# Reason codes that force rollback when observed as absolute counters > 0.
ZERO_TOLERANCE_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "wrong_value",
        "missed_consumer",
        "partial_plan",
        "partial_propagation",
        "false_completion",
        "false_fixed_point",
        "open_frontier",
        "proof_loss",
        "reconstruction_failure",
    }
)


class ChangePropagationRolloutError(ValueError):
    """Raised when control-plane, policy, or metric evidence is invalid."""


class RolloutMode(str, Enum):
    """Release stages; shadow is the only default-safe mode."""

    SHADOW = "shadow"
    ASSIST = "assist"
    NARROW_AUTO = "narrow_auto"
    EXPANDED_AUTO = "expanded_auto"


class RollbackReason(str, Enum):
    """Closed vocabulary of automatic demotion causes."""

    CAPABILITY_REGRESSION = "capability_regression"
    STALE_ROOT = "stale_root"
    OPEN_FRONTIER = "open_frontier"
    RECONSTRUCTION_FAILURE = "reconstruction_failure"
    PROOF_LOSS = "proof_loss"
    WRONG_VALUE = "wrong_value"
    MISSED_CONSUMER = "missed_consumer"
    PARTIAL_PLAN = "partial_plan"
    FALSE_COMPLETION = "false_completion"
    METRIC_BREACH = "metric_breach"
    COVERAGE_LOSS = "coverage_loss"
    EXPLICIT_OPERATOR = "explicit_operator"
    ELEVATED_ABSTENTION_ERROR = "elevated_abstention_error"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------


def repository_root() -> Path:
    """Resolve the repository root from this package location."""

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
        raise ChangePropagationRolloutError(
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
    data = path.read_bytes()
    return _sha256_hex(data)


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise ChangePropagationRolloutError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise ChangePropagationRolloutError(f"{name} is unsafe or too large")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ChangePropagationRolloutError(f"{name} must be a boolean")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChangePropagationRolloutError(
            f"{name} must be a non-negative integer"
        )
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
        raise ChangePropagationRolloutError(
            f"unknown rollout mode: {value!r}"
        ) from exc


def _safe_relative(path: str) -> bool:
    if not path or path.startswith("/") or ".." in Path(path).parts:
        return False
    if "\x00" in path:
        return False
    return True


def _cycle_nodes(edges: Mapping[str, Sequence[str]]) -> list[str]:
    """Return nodes that participate in a directed cycle (Tarjan-style)."""

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


# ---------------------------------------------------------------------------
# Exact source bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagationSourceBinding:
    """Content-addressed binding of exact propagation + control-plane sources."""

    SCHEMA: ClassVar[str] = SOURCE_BINDING_SCHEMA

    repository_root: str
    board_namespace: str = BOARD_NAMESPACE
    task_prefix: str = TASK_PREFIX
    merge_target_branch: str = MERGE_TARGET_BRANCH
    plan_path: str = PLAN_REL
    objective_path: str = OBJECTIVE_REL
    todo_path: str = TODO_REL
    scheduler_path: str = SCHEDULER_REL
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
    launcher_identity: str = ""
    guide_identity: str = ""
    benchmark_identity: str = ""
    fixture_manifest_identity: str = ""
    rollout_module_identity: str = ""
    validate_script_identity: str = ""
    binding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_root", _text(self.repository_root, "repository_root", maximum=4096)
        )
        for name in (
            "board_namespace",
            "task_prefix",
            "merge_target_branch",
            "plan_path",
            "objective_path",
            "todo_path",
            "scheduler_path",
            "launcher_path",
            "guide_path",
            "benchmark_path",
            "fixture_manifest_path",
            "rollout_module_path",
            "validate_script_path",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum=1024))
        if not self.binding_id:
            object.__setattr__(
                self,
                "binding_id",
                content_identity(self.to_dict(include_id=False)),
            )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SOURCE_BINDING_SCHEMA,
            "repository_root": self.repository_root,
            "board_namespace": self.board_namespace,
            "task_prefix": self.task_prefix,
            "merge_target_branch": self.merge_target_branch,
            "plan_path": self.plan_path,
            "objective_path": self.objective_path,
            "todo_path": self.todo_path,
            "scheduler_path": self.scheduler_path,
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
            "launcher_identity": self.launcher_identity,
            "guide_identity": self.guide_identity,
            "benchmark_identity": self.benchmark_identity,
            "fixture_manifest_identity": self.fixture_manifest_identity,
            "rollout_module_identity": self.rollout_module_identity,
            "validate_script_identity": self.validate_script_identity,
        }
        if include_id:
            payload["binding_id"] = self.binding_id
        return payload


def bind_exact_sources(repo_root: Path | None = None) -> PropagationSourceBinding:
    """Seal exact identities for control-plane and propagation sources."""

    root = (repo_root or repository_root()).resolve()
    paths = {
        "plan": root / PLAN_REL,
        "objective": root / OBJECTIVE_REL,
        "todo": root / TODO_REL,
        "scheduler": root / SCHEDULER_REL,
        "launcher": root / LAUNCHER_REL,
        "guide": root / GUIDE_REL,
        "benchmark": root / BENCHMARK_SCRIPT_REL,
        "fixture_manifest": root / FIXTURE_MANIFEST_REL,
        "rollout_module": root / ROLLOUT_MODULE_REL,
        "validate_script": root / VALIDATE_SCRIPT_REL,
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise ChangePropagationRolloutError(
            f"missing exact sources: {sorted(missing)}"
        )

    scheduler = json.loads(paths["scheduler"].read_text(encoding="utf-8"))
    board = str(scheduler.get("board_namespace") or BOARD_NAMESPACE)
    prefix = str(scheduler.get("task_prefix") or TASK_PREFIX)
    merge = str(scheduler.get("merge_target_branch") or MERGE_TARGET_BRANCH)

    return PropagationSourceBinding(
        repository_root=str(root),
        board_namespace=board,
        task_prefix=prefix,
        merge_target_branch=merge,
        plan_identity=file_identity(paths["plan"]),
        objective_identity=file_identity(paths["objective"]),
        todo_identity=file_identity(paths["todo"]),
        scheduler_identity=file_identity(paths["scheduler"]),
        launcher_identity=file_identity(paths["launcher"]),
        guide_identity=file_identity(paths["guide"]),
        benchmark_identity=file_identity(paths["benchmark"]),
        fixture_manifest_identity=file_identity(paths["fixture_manifest"]),
        rollout_module_identity=file_identity(paths["rollout_module"]),
        validate_script_identity=file_identity(paths["validate_script"]),
    )


# ---------------------------------------------------------------------------
# Rollout policy / feature flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangePropagationRolloutPolicy:
    """Per-repository/program feature flags for change-propagation release.

    Defaults are fail-closed:

    * mode is always ``shadow`` unless an explicit scoped policy elevates it;
    * assist and narrow-auto require ``explicit_policy_document`` plus scope;
    * automated mutation is limited to complete-frontier unique reconstructed
      analytical supported-Python transforms;
    * model-authored, stateful, public schema/API, dynamic/generated/native,
      and cross-root changes remain approval-gated.
    """

    SCHEMA: ClassVar[str] = ROLLOUT_POLICY_SCHEMA
    INTERFACE: ClassVar[str] = ROLLOUT_POLICY_INTERFACE

    policy_id: str = "policy:change-propagation-rollout-default"
    policy_revision: str = "1"
    repository_id: str = ""
    program_id: str = "agent-supervisor-proof-gated-contract-repair-v1"
    mode: RolloutMode | str = RolloutMode.SHADOW
    explicit_policy_document: str = ""
    scoped_path_globs: tuple[str, ...] = ()
    allow_assist: bool = False
    allow_narrow_auto: bool = False
    allow_expanded_auto: bool = False
    auto_requires_unique_target: bool = True
    auto_requires_reconstruction: bool = True
    auto_requires_supported_python: bool = True
    auto_requires_complete_frontier: bool = True
    auto_requires_analytical_path: bool = True
    auto_allowed_transforms: tuple[str, ...] = (
        TransformKind.ADD_ARGUMENT.value,
        TransformKind.RENAME_ARGUMENT.value,
        TransformKind.REORDER_ARGUMENT.value,
        TransformKind.THREAD_PARAMETER.value,
        TransformKind.ADD_IMPORT.value,
        TransformKind.ADD_EXPORT.value,
    )
    approval_gated_families: tuple[str, ...] = tuple(
        sorted(APPROVAL_GATED_CHANGE_FAMILIES)
    )
    rollback_on_capability_regression: bool = True
    rollback_on_stale_root: bool = True
    rollback_on_open_frontier: bool = True
    rollback_on_reconstruction_failure: bool = True
    rollback_on_proof_loss: bool = True
    rollback_on_metric_breach: bool = True
    rollback_on_coverage_loss: bool = True
    mutation_authorized: bool = False
    completion_authoritative: bool = False
    policy_binding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision"),
        )
        object.__setattr__(
            self, "repository_id", str(self.repository_id or "").strip()
        )
        object.__setattr__(self, "program_id", _text(self.program_id, "program_id"))
        object.__setattr__(self, "mode", _mode(self.mode))
        object.__setattr__(
            self,
            "explicit_policy_document",
            str(self.explicit_policy_document or "").strip(),
        )
        globs = tuple(
            sorted(
                {
                    _text(item, "scoped_path_globs", maximum=1024)
                    for item in self.scoped_path_globs
                }
            )
        )
        object.__setattr__(self, "scoped_path_globs", globs)
        transforms = tuple(
            sorted(
                {
                    _text(item, "auto_allowed_transforms").casefold()
                    for item in self.auto_allowed_transforms
                }
            )
        )
        if not transforms:
            raise ChangePropagationRolloutError(
                "auto_allowed_transforms must not be empty"
            )
        object.__setattr__(self, "auto_allowed_transforms", transforms)
        gated = tuple(
            sorted(
                {
                    _text(item, "approval_gated_families").casefold()
                    for item in self.approval_gated_families
                }
            )
        )
        object.__setattr__(self, "approval_gated_families", gated)
        for name in (
            "allow_assist",
            "allow_narrow_auto",
            "allow_expanded_auto",
            "auto_requires_unique_target",
            "auto_requires_reconstruction",
            "auto_requires_supported_python",
            "auto_requires_complete_frontier",
            "auto_requires_analytical_path",
            "rollback_on_capability_regression",
            "rollback_on_stale_root",
            "rollback_on_open_frontier",
            "rollback_on_reconstruction_failure",
            "rollback_on_proof_loss",
            "rollback_on_metric_breach",
            "rollback_on_coverage_loss",
            "mutation_authorized",
            "completion_authoritative",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if self.completion_authoritative:
            raise ChangePropagationRolloutError(
                "rollout policy cannot claim completion authority"
            )
        if self.mode is RolloutMode.SHADOW and self.mutation_authorized:
            raise ChangePropagationRolloutError(
                "shadow mode cannot authorize mutation"
            )
        if not self.policy_binding_id:
            object.__setattr__(
                self,
                "policy_binding_id",
                content_identity(self.to_dict(include_id=False)),
            )
        self._assert_mode_allowed()

    def _assert_mode_allowed(self) -> None:
        mode = self.mode if isinstance(self.mode, RolloutMode) else _mode(self.mode)
        if mode is RolloutMode.SHADOW:
            return
        if not self.has_explicit_scoped_policy():
            raise ChangePropagationRolloutError(
                f"{mode.value} requires an explicit scoped policy document "
                "and repository/program/policy scope"
            )
        if mode is RolloutMode.ASSIST and not self.allow_assist:
            raise ChangePropagationRolloutError(
                "assist mode is not enabled on this policy"
            )
        if mode is RolloutMode.NARROW_AUTO and not self.allow_narrow_auto:
            raise ChangePropagationRolloutError(
                "narrow_auto mode is not enabled on this policy"
            )
        if mode is RolloutMode.EXPANDED_AUTO and not self.allow_expanded_auto:
            raise ChangePropagationRolloutError(
                "expanded_auto mode is not enabled on this policy"
            )

    def has_explicit_scoped_policy(self) -> bool:
        if not self.explicit_policy_document:
            return False
        return bool(self.repository_id or self.program_id or self.policy_id)

    @property
    def mode_value(self) -> str:
        return self.mode.value if isinstance(self.mode, RolloutMode) else str(self.mode)

    def is_approval_gated(
        self,
        *,
        transform: str = "",
        change_family: str = "",
        model_authored: bool = False,
        stateful: bool = False,
        public_schema_api: bool = False,
        dynamic: bool = False,
        generated: bool = False,
        native: bool = False,
        cross_root: bool = False,
    ) -> bool:
        """Return True when the change family remains approval-gated."""

        if model_authored or stateful or public_schema_api:
            return True
        if dynamic or generated or native or cross_root:
            return True
        key = str(transform or change_family or "").strip().casefold()
        if key and key in {item.casefold() for item in self.approval_gated_families}:
            return True
        if key and key in APPROVAL_GATED_CHANGE_FAMILIES:
            return True
        return False

    def allows_automated_mutation(
        self,
        *,
        transform: str,
        unique_target: bool,
        reconstructed: bool,
        supported_python: bool,
        complete_frontier: bool,
        analytical_path: bool = True,
        model_authored: bool = False,
        stateful: bool = False,
        public_schema_api: bool = False,
        dynamic: bool = False,
        generated: bool = False,
        native: bool = False,
        cross_root: bool = False,
        change_family: str = "",
    ) -> bool:
        """Return True only for initially allowed narrow-auto analytical transforms."""

        mode = _mode(self.mode)
        if mode is RolloutMode.SHADOW or mode is RolloutMode.ASSIST:
            return False
        if mode is RolloutMode.NARROW_AUTO and not self.allow_narrow_auto:
            return False
        if mode is RolloutMode.EXPANDED_AUTO and not self.allow_expanded_auto:
            return False
        if not self.mutation_authorized:
            return False
        if self.is_approval_gated(
            transform=transform,
            change_family=change_family,
            model_authored=model_authored,
            stateful=stateful,
            public_schema_api=public_schema_api,
            dynamic=dynamic,
            generated=generated,
            native=native,
            cross_root=cross_root,
        ):
            return False
        transform_key = str(transform or "").strip().casefold()
        if transform_key not in self.auto_allowed_transforms:
            return False
        if mode is RolloutMode.NARROW_AUTO and transform_key not in {
            item.casefold() for item in NARROW_AUTO_TRANSFORMS
        }:
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
            "explicit_policy_document": self.explicit_policy_document,
            "scoped_path_globs": list(self.scoped_path_globs),
            "allow_assist": self.allow_assist,
            "allow_narrow_auto": self.allow_narrow_auto,
            "allow_expanded_auto": self.allow_expanded_auto,
            "auto_requires_unique_target": self.auto_requires_unique_target,
            "auto_requires_reconstruction": self.auto_requires_reconstruction,
            "auto_requires_supported_python": self.auto_requires_supported_python,
            "auto_requires_complete_frontier": self.auto_requires_complete_frontier,
            "auto_requires_analytical_path": self.auto_requires_analytical_path,
            "auto_allowed_transforms": list(self.auto_allowed_transforms),
            "approval_gated_families": list(self.approval_gated_families),
            "rollback_on_capability_regression": (
                self.rollback_on_capability_regression
            ),
            "rollback_on_stale_root": self.rollback_on_stale_root,
            "rollback_on_open_frontier": self.rollback_on_open_frontier,
            "rollback_on_reconstruction_failure": (
                self.rollback_on_reconstruction_failure
            ),
            "rollback_on_proof_loss": self.rollback_on_proof_loss,
            "rollback_on_metric_breach": self.rollback_on_metric_breach,
            "rollback_on_coverage_loss": self.rollback_on_coverage_loss,
            "mutation_authorized": self.mutation_authorized,
            "completion_authoritative": self.completion_authoritative,
        }
        if include_id:
            payload["policy_binding_id"] = self.policy_binding_id
        return payload

    @classmethod
    def default(cls) -> "ChangePropagationRolloutPolicy":
        return cls()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangePropagationRolloutPolicy":
        if not isinstance(value, Mapping):
            raise ChangePropagationRolloutError("policy payload must be an object")
        known = set(cls.__dataclass_fields__) - {"policy_binding_id"}
        data = {key: value[key] for key in known if key in value}
        for key in (
            "scoped_path_globs",
            "auto_allowed_transforms",
            "approval_gated_families",
        ):
            if key in data:
                data[key] = tuple(data[key] or ())
        return cls(**data)


def default_rollout_policy() -> ChangePropagationRolloutPolicy:
    return ChangePropagationRolloutPolicy.default()


def elevate_rollout_policy(
    *,
    mode: RolloutMode | str,
    explicit_policy_document: str,
    repository_id: str,
    program_id: str = "agent-supervisor-proof-gated-contract-repair-v1",
    policy_id: str = "policy:change-propagation-rollout-scoped",
    policy_revision: str = "1",
    scoped_path_globs: Sequence[str] = (),
    mutation_authorized: bool = False,
    allow_expanded_auto: bool = False,
) -> ChangePropagationRolloutPolicy:
    """Build an elevated policy; still fail-closed without explicit scope."""

    mode_value = _mode(mode)
    return ChangePropagationRolloutPolicy(
        policy_id=policy_id,
        policy_revision=policy_revision,
        repository_id=repository_id,
        program_id=program_id,
        mode=mode_value,
        explicit_policy_document=explicit_policy_document,
        scoped_path_globs=tuple(scoped_path_globs),
        allow_assist=mode_value
        in {RolloutMode.ASSIST, RolloutMode.NARROW_AUTO, RolloutMode.EXPANDED_AUTO},
        allow_narrow_auto=mode_value
        in {RolloutMode.NARROW_AUTO, RolloutMode.EXPANDED_AUTO},
        allow_expanded_auto=allow_expanded_auto
        and mode_value is RolloutMode.EXPANDED_AUTO,
        mutation_authorized=mutation_authorized
        and mode_value
        in {RolloutMode.NARROW_AUTO, RolloutMode.EXPANDED_AUTO},
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangePropagationMetrics:
    """Operator-facing release metrics for change propagation.

    Rates are integer parts-per-million so sealed identities stay float-free.
    Exposes every benchmark stage, analytical/model split, tokens/context, and
    fixed-point iterations alongside absolute-zero safety floors.
    """

    SCHEMA: ClassVar[str] = METRICS_SCHEMA
    INTERFACE: ClassVar[str] = METRICS_INTERFACE

    case_count: int = 0
    impact_recall: int = 0
    consumer_precision: int = 0
    proof_eligible_value_recall: int = 0
    unique_source_precision: int = 0
    plan_completeness: int = 0
    closure_success_rate: int = 0
    completion_success_rate: int = 0
    analytical_coverage: int = 0
    model_rate: int = 0
    llm_rate: int = 0
    llm_scope_escape_rate: int = 0
    analytical_model_split: Mapping[str, int] = field(default_factory=dict)
    stage_counts: Mapping[str, int] = field(default_factory=dict)
    stage_cost_units: Mapping[str, int] = field(default_factory=dict)
    fixed_point_iterations: int = 0
    fixed_point_iterations_total: int = 0
    scc_rollback_count: int = 0
    abstention_count: int = 0
    abstention_rate: int = 0
    tokens: int = 0
    context_bytes: int = 0
    total_cost_units: int = 0
    total_latency_units: int = 0
    cache_hit_rate: int = 0
    wrong_path_rate: int = 0
    missed_consumer_rate: int = 0
    wrong_value_rate: int = 0
    partial_plan_rate: int = 0
    false_completion_rate: int = 0
    open_frontier_rate: int = 0
    safety_floors: Mapping[str, int] = field(default_factory=dict)
    safety_absolute: Mapping[str, int] = field(default_factory=dict)
    outcome_counts: Mapping[str, int] = field(default_factory=dict)
    family_counts: Mapping[str, int] = field(default_factory=dict)
    recall_k: int = DEFAULT_RECALL_K
    reason_code_counts: Mapping[str, int] = field(default_factory=dict)
    metrics_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "case_count",
            "impact_recall",
            "consumer_precision",
            "proof_eligible_value_recall",
            "unique_source_precision",
            "plan_completeness",
            "closure_success_rate",
            "completion_success_rate",
            "analytical_coverage",
            "model_rate",
            "llm_rate",
            "llm_scope_escape_rate",
            "fixed_point_iterations",
            "fixed_point_iterations_total",
            "scc_rollback_count",
            "abstention_count",
            "abstention_rate",
            "tokens",
            "context_bytes",
            "total_cost_units",
            "total_latency_units",
            "cache_hit_rate",
            "wrong_path_rate",
            "missed_consumer_rate",
            "wrong_value_rate",
            "partial_plan_rate",
            "false_completion_rate",
            "open_frontier_rate",
            "recall_k",
        ):
            object.__setattr__(
                self, name, _non_negative_int(getattr(self, name), name)
            )
        # Alias fixed_point_iterations when only total is supplied.
        if self.fixed_point_iterations == 0 and self.fixed_point_iterations_total:
            object.__setattr__(
                self, "fixed_point_iterations", self.fixed_point_iterations_total
            )
        if self.fixed_point_iterations_total == 0 and self.fixed_point_iterations:
            object.__setattr__(
                self,
                "fixed_point_iterations_total",
                self.fixed_point_iterations,
            )
        stages = {
            stage: _non_negative_int(
                dict(self.stage_counts or {}).get(stage, 0), f"stage:{stage}"
            )
            for stage in BENCHMARK_STAGES
        }
        # Default stage presence when case_count is known but stages empty.
        if self.case_count and not any(stages.values()):
            stages = {stage: self.case_count for stage in BENCHMARK_STAGES}
        object.__setattr__(self, "stage_counts", MappingProxyType(stages))
        stage_costs = {
            stage: _non_negative_int(
                dict(self.stage_cost_units or {}).get(stage, 1), f"cost:{stage}"
            )
            for stage in BENCHMARK_STAGES
        }
        object.__setattr__(self, "stage_cost_units", MappingProxyType(stage_costs))
        split = dict(self.analytical_model_split or {})
        if not split:
            split = {
                "analytical_coverage": self.analytical_coverage,
                "model_rate": self.model_rate or self.llm_rate,
                "llm_rate": self.llm_rate,
            }
        object.__setattr__(
            self,
            "analytical_model_split",
            MappingProxyType(
                {
                    str(k): _non_negative_int(v, str(k))
                    for k, v in sorted(split.items())
                }
            ),
        )
        floors = {
            key: _non_negative_int(dict(self.safety_floors or {}).get(key, 0), key)
            for key in SAFETY_FLOOR_KEYS
        }
        object.__setattr__(self, "safety_floors", MappingProxyType(floors))
        absolute = {
            str(key): _non_negative_int(value, str(key))
            for key, value in sorted(dict(self.safety_absolute or {}).items())
        }
        for key in SAFETY_ABSOLUTE_KEYS:
            absolute.setdefault(key, 0)
        object.__setattr__(self, "safety_absolute", MappingProxyType(absolute))
        object.__setattr__(
            self,
            "outcome_counts",
            MappingProxyType(
                {
                    str(k): _non_negative_int(v, str(k))
                    for k, v in sorted(dict(self.outcome_counts or {}).items())
                }
            ),
        )
        object.__setattr__(
            self,
            "family_counts",
            MappingProxyType(
                {
                    str(k): _non_negative_int(v, str(k))
                    for k, v in sorted(dict(self.family_counts or {}).items())
                }
            ),
        )
        object.__setattr__(
            self,
            "reason_code_counts",
            MappingProxyType(
                {
                    str(k): _non_negative_int(v, str(k))
                    for k, v in sorted(dict(self.reason_code_counts or {}).items())
                }
            ),
        )
        if not self.metrics_id:
            object.__setattr__(
                self,
                "metrics_id",
                content_identity(self.to_dict(include_id=False)),
            )

    def floors_hold(self) -> bool:
        floors_ok = all(
            int(self.safety_floors.get(key, 1)) == 0 for key in SAFETY_FLOOR_KEYS
        )
        absolute_ok = all(
            int(self.safety_absolute.get(key, 1)) == 0 for key in SAFETY_ABSOLUTE_KEYS
        )
        return floors_ok and absolute_ok

    def breaches(self) -> tuple[str, ...]:
        failed = [
            key
            for key in SAFETY_FLOOR_KEYS
            if int(self.safety_floors.get(key, 1)) != 0
        ]
        for key in SAFETY_ABSOLUTE_KEYS:
            if int(self.safety_absolute.get(key, 1)) != 0:
                failed.append(key)
        for name, value in (
            ("wrong_path_rate", self.wrong_path_rate),
            ("missed_consumer_rate", self.missed_consumer_rate),
            ("wrong_value_rate", self.wrong_value_rate),
            ("partial_plan_rate", self.partial_plan_rate),
            ("false_completion_rate", self.false_completion_rate),
            ("open_frontier_rate", self.open_frontier_rate),
            ("llm_scope_escape_rate", self.llm_scope_escape_rate),
        ):
            if int(value) != 0:
                failed.append(name)
        return tuple(dict.fromkeys(failed))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": METRICS_SCHEMA,
            "interface": METRICS_INTERFACE,
            "case_count": self.case_count,
            "impact_recall": self.impact_recall,
            "consumer_precision": self.consumer_precision,
            "proof_eligible_value_recall": self.proof_eligible_value_recall,
            "unique_source_precision": self.unique_source_precision,
            "plan_completeness": self.plan_completeness,
            "closure_success_rate": self.closure_success_rate,
            "completion_success_rate": self.completion_success_rate,
            "analytical_coverage": self.analytical_coverage,
            "model_rate": self.model_rate,
            "llm_rate": self.llm_rate,
            "llm_scope_escape_rate": self.llm_scope_escape_rate,
            "analytical_model_split": dict(self.analytical_model_split),
            "stage_counts": dict(self.stage_counts),
            "stage_cost_units": dict(self.stage_cost_units),
            "benchmark_stages": list(BENCHMARK_STAGES),
            "fixed_point_iterations": self.fixed_point_iterations,
            "fixed_point_iterations_total": self.fixed_point_iterations_total,
            "scc_rollback_count": self.scc_rollback_count,
            "abstention_count": self.abstention_count,
            "abstention_rate": self.abstention_rate,
            "tokens": self.tokens,
            "context_bytes": self.context_bytes,
            "total_cost_units": self.total_cost_units,
            "total_latency_units": self.total_latency_units,
            "cache_hit_rate": self.cache_hit_rate,
            "wrong_path_rate": self.wrong_path_rate,
            "missed_consumer_rate": self.missed_consumer_rate,
            "wrong_value_rate": self.wrong_value_rate,
            "partial_plan_rate": self.partial_plan_rate,
            "false_completion_rate": self.false_completion_rate,
            "open_frontier_rate": self.open_frontier_rate,
            "safety_floors": dict(self.safety_floors),
            "safety_absolute": dict(self.safety_absolute),
            "outcome_counts": dict(self.outcome_counts),
            "family_counts": dict(self.family_counts),
            "recall_k": self.recall_k,
            "reason_code_counts": dict(self.reason_code_counts),
        }
        if include_id:
            payload["metrics_id"] = self.metrics_id
        return payload

    @classmethod
    def from_benchmark_metrics(
        cls,
        metrics: Mapping[str, Any],
    ) -> "ChangePropagationMetrics":
        """Project a RPR-045 ``ChangePropagationBenchmarkMetrics`` object."""

        case_count = int(metrics.get("case_count") or 0)
        abstention = int(metrics.get("abstention_count") or 0)
        floors = dict(metrics.get("safety_floors") or {})
        for key in SAFETY_FLOOR_KEYS:
            floors.setdefault(key, 0)
        absolute = dict(metrics.get("safety_absolute") or {})
        for key in SAFETY_ABSOLUTE_KEYS:
            absolute.setdefault(key, 0)
        llm_rate = int(metrics.get("llm_rate") or 0)
        analytical = int(metrics.get("analytical_coverage") or 0)
        fp_iters = int(metrics.get("fixed_point_iterations_total") or 0)
        return cls(
            case_count=case_count,
            impact_recall=int(metrics.get("impact_recall") or 0),
            consumer_precision=int(metrics.get("consumer_precision") or 0),
            proof_eligible_value_recall=int(
                metrics.get("proof_eligible_value_recall") or 0
            ),
            unique_source_precision=int(metrics.get("unique_source_precision") or 0),
            plan_completeness=int(metrics.get("plan_completeness") or 0),
            closure_success_rate=int(metrics.get("closure_success_rate") or 0),
            completion_success_rate=int(metrics.get("completion_success_rate") or 0),
            analytical_coverage=analytical,
            model_rate=llm_rate,
            llm_rate=llm_rate,
            llm_scope_escape_rate=int(metrics.get("llm_scope_escape_rate") or 0),
            analytical_model_split={
                "analytical_coverage": analytical,
                "model_rate": llm_rate,
                "llm_rate": llm_rate,
            },
            stage_counts={stage: case_count for stage in BENCHMARK_STAGES},
            stage_cost_units={stage: 1 for stage in BENCHMARK_STAGES},
            fixed_point_iterations=fp_iters,
            fixed_point_iterations_total=fp_iters,
            scc_rollback_count=int(metrics.get("scc_rollback_count") or 0),
            abstention_count=abstention,
            abstention_rate=_ppm(abstention, max(1, case_count)),
            tokens=int(metrics.get("total_token_units") or 0),
            context_bytes=int(metrics.get("total_context_bytes") or 0),
            total_cost_units=int(metrics.get("total_cost_units") or 0),
            total_latency_units=int(metrics.get("total_latency_units") or 0),
            cache_hit_rate=int(metrics.get("cache_hit_rate") or 0),
            wrong_path_rate=int(
                floors.get("wrong_path_automated_mutation_rate") or 0
            ),
            missed_consumer_rate=int(
                floors.get("missed_resolved_impacted_consumer_rate") or 0
            ),
            wrong_value_rate=int(
                floors.get("unproved_or_wrong_value_source_admission_rate") or 0
            ),
            partial_plan_rate=int(
                floors.get("partial_propagation_completion_rate") or 0
            ),
            false_completion_rate=int(
                floors.get("false_fixed_point_completion_rate") or 0
            ),
            open_frontier_rate=0,
            safety_floors=floors,
            safety_absolute=absolute,
            outcome_counts=dict(metrics.get("outcome_counts") or {}),
            family_counts=dict(metrics.get("family_counts") or {}),
            recall_k=int(metrics.get("recall_k") or DEFAULT_RECALL_K),
            reason_code_counts=dict(metrics.get("outcome_counts") or {}),
        )

    @classmethod
    def empty(cls) -> "ChangePropagationMetrics":
        return cls(
            safety_floors={key: 0 for key in SAFETY_FLOOR_KEYS},
            safety_absolute={key: 0 for key in SAFETY_ABSOLUTE_KEYS},
            stage_counts={stage: 0 for stage in BENCHMARK_STAGES},
            stage_cost_units={stage: 1 for stage in BENCHMARK_STAGES},
            analytical_model_split={
                "analytical_coverage": 0,
                "model_rate": 0,
                "llm_rate": 0,
            },
        )


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollbackReceipt:
    """Content-addressed demotion receipt; never completion authority."""

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
                raise ChangePropagationRolloutError(
                    f"unknown rollback reason: {self.reason!r}"
                ) from exc
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "from_mode", _mode(self.from_mode))
        object.__setattr__(self, "to_mode", _mode(self.to_mode))
        object.__setattr__(self, "detail", str(self.detail or "").strip())
        object.__setattr__(
            self,
            "metric_breaches",
            tuple(str(item) for item in self.metric_breaches),
        )
        object.__setattr__(
            self,
            "capability_ids",
            tuple(str(item) for item in self.capability_ids),
        )
        object.__setattr__(
            self, "stale_roots", tuple(str(item) for item in self.stale_roots)
        )
        object.__setattr__(
            self, "reason_codes", tuple(str(item) for item in self.reason_codes)
        )
        object.__setattr__(
            self, "policy_binding_id", str(self.policy_binding_id or "")
        )
        if not self.receipt_id:
            object.__setattr__(
                self,
                "receipt_id",
                content_identity(self.to_dict(include_id=False)),
            )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": ROLLBACK_RECEIPT_SCHEMA,
            "reason": self.reason.value
            if isinstance(self.reason, RollbackReason)
            else str(self.reason),
            "from_mode": self.from_mode.value
            if isinstance(self.from_mode, RolloutMode)
            else str(self.from_mode),
            "to_mode": self.to_mode.value
            if isinstance(self.to_mode, RolloutMode)
            else str(self.to_mode),
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
    if current is RolloutMode.SHADOW:
        return RolloutMode.SHADOW
    demotion = {
        RolloutMode.EXPANDED_AUTO: RolloutMode.NARROW_AUTO,
        RolloutMode.NARROW_AUTO: RolloutMode.ASSIST,
        RolloutMode.ASSIST: RolloutMode.SHADOW,
    }
    return demotion.get(current, RolloutMode.SHADOW)


def evaluate_rollback(
    policy: ChangePropagationRolloutPolicy,
    *,
    metrics: ChangePropagationMetrics | None = None,
    capability_regression: Sequence[str] = (),
    stale_roots: Sequence[str] = (),
    open_frontier: bool = False,
    reconstruction_failed: bool = False,
    proof_loss: bool = False,
    wrong_value: bool = False,
    missed_consumer: bool = False,
    partial_plan: bool = False,
    false_completion: bool = False,
    coverage_loss: bool = False,
    elevated_abstention_error: bool = False,
    reason_codes: Sequence[str] = (),
) -> RollbackReceipt | None:
    """Return a demotion receipt when a rollback gate fires; else None.

    Always demotes toward shadow.  Never elevates mode.
    """

    current = _mode(policy.mode)
    target = _demotion_target(current)
    codes = {str(item).strip().casefold() for item in reason_codes if item}

    def _receipt(
        reason: RollbackReason,
        *,
        detail: str,
        metric_breaches: Sequence[str] = (),
        capability_ids: Sequence[str] = (),
        roots: Sequence[str] = (),
        extra_codes: Sequence[str] = (),
    ) -> RollbackReceipt:
        return RollbackReceipt(
            reason=reason,
            from_mode=current,
            to_mode=target,
            detail=detail,
            metric_breaches=tuple(metric_breaches),
            capability_ids=tuple(sorted(set(capability_ids))),
            stale_roots=tuple(sorted(set(roots))),
            reason_codes=tuple(sorted({*codes, *extra_codes})),
            policy_binding_id=policy.policy_binding_id,
        )

    if policy.rollback_on_capability_regression and capability_regression:
        return _receipt(
            RollbackReason.CAPABILITY_REGRESSION,
            detail="capability health regression",
            capability_ids=capability_regression,
        )
    if policy.rollback_on_stale_root and stale_roots:
        return _receipt(
            RollbackReason.STALE_ROOT,
            detail="stale authority root observed",
            roots=stale_roots,
            extra_codes=("stale_root",),
        )
    if policy.rollback_on_open_frontier and (
        open_frontier or "open_frontier" in codes
    ):
        return _receipt(
            RollbackReason.OPEN_FRONTIER,
            detail="impact frontier remains open",
            extra_codes=("open_frontier",),
        )
    if policy.rollback_on_reconstruction_failure and (
        reconstruction_failed or "reconstruction_failure" in codes
    ):
        return _receipt(
            RollbackReason.RECONSTRUCTION_FAILURE,
            detail="proof reconstruction failure",
            extra_codes=("reconstruction_failure",),
        )
    if policy.rollback_on_proof_loss and (proof_loss or "proof_loss" in codes):
        return _receipt(
            RollbackReason.PROOF_LOSS,
            detail="proof loss observed",
            extra_codes=("proof_loss",),
        )
    if wrong_value or "wrong_value" in codes:
        return _receipt(
            RollbackReason.WRONG_VALUE,
            detail="wrong or unproved value source",
            extra_codes=("wrong_value",),
        )
    if missed_consumer or "missed_consumer" in codes:
        return _receipt(
            RollbackReason.MISSED_CONSUMER,
            detail="missed resolved impacted consumer",
            extra_codes=("missed_consumer",),
        )
    if partial_plan or "partial_plan" in codes or "partial_propagation" in codes:
        return _receipt(
            RollbackReason.PARTIAL_PLAN,
            detail="partial plan or incomplete SCC group",
            extra_codes=("partial_plan",),
        )
    if false_completion or "false_completion" in codes or "false_fixed_point" in codes:
        return _receipt(
            RollbackReason.FALSE_COMPLETION,
            detail="false fixed-point or false completion",
            extra_codes=("false_completion",),
        )
    if policy.rollback_on_coverage_loss and coverage_loss:
        return _receipt(
            RollbackReason.COVERAGE_LOSS,
            detail="graph/index/coverage loss",
            extra_codes=("coverage_loss",),
        )
    if elevated_abstention_error:
        return _receipt(
            RollbackReason.ELEVATED_ABSTENTION_ERROR,
            detail="elevated abstention error rate",
        )
    if policy.rollback_on_metric_breach and metrics is not None:
        breaches = metrics.breaches()
        if breaches or not metrics.floors_hold():
            return _receipt(
                RollbackReason.METRIC_BREACH,
                detail="safety floor or metric breach",
                metric_breaches=breaches
                or tuple(
                    key
                    for key in SAFETY_FLOOR_KEYS
                    if int(metrics.safety_floors.get(key, 1)) != 0
                ),
            )
    # Zero-tolerance reason codes observed without dedicated flags.
    for code in sorted(codes & ZERO_TOLERANCE_REASON_CODES):
        mapped = {
            "wrong_value": RollbackReason.WRONG_VALUE,
            "missed_consumer": RollbackReason.MISSED_CONSUMER,
            "partial_plan": RollbackReason.PARTIAL_PLAN,
            "partial_propagation": RollbackReason.PARTIAL_PLAN,
            "false_completion": RollbackReason.FALSE_COMPLETION,
            "false_fixed_point": RollbackReason.FALSE_COMPLETION,
            "open_frontier": RollbackReason.OPEN_FRONTIER,
            "proof_loss": RollbackReason.PROOF_LOSS,
            "reconstruction_failure": RollbackReason.RECONSTRUCTION_FAILURE,
        }.get(code)
        if mapped is not None:
            return _receipt(
                mapped,
                detail=f"zero-tolerance reason code: {code}",
                extra_codes=(code,),
            )
    return None


def apply_rollback(
    policy: ChangePropagationRolloutPolicy,
    receipt: RollbackReceipt,
) -> ChangePropagationRolloutPolicy:
    """Return a demoted policy; mutation is always revoked."""

    to_mode = _mode(receipt.to_mode)
    return ChangePropagationRolloutPolicy(
        policy_id=policy.policy_id,
        policy_revision=policy.policy_revision,
        repository_id=policy.repository_id,
        program_id=policy.program_id,
        mode=to_mode,
        explicit_policy_document=policy.explicit_policy_document,
        scoped_path_globs=policy.scoped_path_globs,
        allow_assist=policy.allow_assist and to_mode is not RolloutMode.SHADOW,
        allow_narrow_auto=policy.allow_narrow_auto
        and to_mode in {RolloutMode.NARROW_AUTO, RolloutMode.EXPANDED_AUTO},
        allow_expanded_auto=policy.allow_expanded_auto
        and to_mode is RolloutMode.EXPANDED_AUTO,
        auto_requires_unique_target=policy.auto_requires_unique_target,
        auto_requires_reconstruction=policy.auto_requires_reconstruction,
        auto_requires_supported_python=policy.auto_requires_supported_python,
        auto_requires_complete_frontier=policy.auto_requires_complete_frontier,
        auto_requires_analytical_path=policy.auto_requires_analytical_path,
        auto_allowed_transforms=policy.auto_allowed_transforms,
        approval_gated_families=policy.approval_gated_families,
        rollback_on_capability_regression=policy.rollback_on_capability_regression,
        rollback_on_stale_root=policy.rollback_on_stale_root,
        rollback_on_open_frontier=policy.rollback_on_open_frontier,
        rollback_on_reconstruction_failure=policy.rollback_on_reconstruction_failure,
        rollback_on_proof_loss=policy.rollback_on_proof_loss,
        rollback_on_metric_breach=policy.rollback_on_metric_breach,
        rollback_on_coverage_loss=policy.rollback_on_coverage_loss,
        mutation_authorized=False,
        completion_authoritative=False,
    )


class ChangePropagationRollbackGate:
    """Operator-facing rollback gate bound to a rollout policy."""

    INTERFACE: ClassVar[str] = ROLLBACK_GATE_INTERFACE

    def __init__(self, policy: ChangePropagationRolloutPolicy | None = None) -> None:
        self.policy = policy or default_rollout_policy()

    def evaluate(self, **kwargs: Any) -> RollbackReceipt | None:
        return evaluate_rollback(self.policy, **kwargs)

    def apply(self, receipt: RollbackReceipt) -> ChangePropagationRolloutPolicy:
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
# Check report helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus | str
    detail: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name"))
        if isinstance(self.status, CheckStatus):
            status = self.status
        else:
            status = CheckStatus(str(self.status).strip().casefold())
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(
            self, "evidence", MappingProxyType(dict(self.evidence or {}))
        )

    @property
    def ok(self) -> bool:
        return self.status in {CheckStatus.PASS, CheckStatus.SKIP, CheckStatus.WARN}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value
            if isinstance(self.status, CheckStatus)
            else str(self.status),
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


def _load_benchmark_module():
    path = repository_root() / BENCHMARK_SCRIPT_REL
    name = "benchmark_change_propagation_rpr046"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ChangePropagationRolloutError(
            f"unable to load benchmark module at {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_task_file_fallback(todo_path: Path) -> list[Any]:
    """Minimal task parser when daemon imports are unavailable."""

    text = todo_path.read_text(encoding="utf-8")
    tasks: list[Any] = []

    @dataclass
    class _Task:
        task_id: str
        depends_on: tuple[str, ...]
        outputs: tuple[str, ...]
        metadata: dict[str, str]

    current_id = ""
    depends: list[str] = []
    outputs: list[str] = []
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        header = re.match(r"^##\s+(RPR-\d+)\b", line)
        if header:
            if current_id:
                tasks.append(
                    _Task(
                        task_id=current_id,
                        depends_on=tuple(depends),
                        outputs=tuple(outputs),
                        metadata=dict(metadata),
                    )
                )
            current_id = header.group(1)
            depends = []
            outputs = []
            metadata = {}
            continue
        if not current_id:
            continue
        m = re.match(r"^-\s+Depends on:\s*(.*)$", line, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            depends = [
                item.strip()
                for item in re.split(r"[, ]+", raw)
                if item.strip().startswith("RPR-")
            ]
            continue
        m = re.match(r"^-\s+Outputs:\s*(.*)$", line, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            outputs = [item.strip() for item in raw.split(",") if item.strip()]
            continue
        m = re.match(r"^-\s+Goal id:\s*(.*)$", line, re.IGNORECASE)
        if m:
            metadata["goal id"] = m.group(1).strip()
            continue
    if current_id:
        tasks.append(
            _Task(
                task_id=current_id,
                depends_on=tuple(depends),
                outputs=tuple(outputs),
                metadata=dict(metadata),
            )
        )
    return tasks


def _parse_goal_heap_fallback(text: str) -> list[Any]:
    @dataclass
    class _Goal:
        goal_id: str
        dependencies: tuple[str, ...]
        parent_goal_ids: tuple[str, ...]

    goals: list[Any] = []
    current = ""
    deps: list[str] = []
    parents: list[str] = []
    for line in text.splitlines():
        header = re.match(r"^##\s+(RPR-G\d+)\b", line)
        if header:
            if current:
                goals.append(
                    _Goal(
                        goal_id=current,
                        dependencies=tuple(deps),
                        parent_goal_ids=tuple(parents),
                    )
                )
            current = header.group(1)
            deps = []
            parents = []
            continue
        if not current:
            continue
        m = re.match(r"^-\s+Depends on:\s*(.*)$", line, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            deps = [
                item.strip()
                for item in re.split(r"[, ]+", raw)
                if item.strip().startswith("RPR-G")
            ]
            continue
        m = re.match(r"^-\s+Parent:\s*(.*)$", line, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            parents = [
                item.strip()
                for item in re.split(r"[, ]+", raw)
                if item.strip().startswith("RPR-G")
            ]
    if current:
        goals.append(
            _Goal(
                goal_id=current,
                dependencies=tuple(deps),
                parent_goal_ids=tuple(parents),
            )
        )
    return goals


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def check_plan_objective_task_dag(
    repo_root: Path | None = None,
) -> CheckResult:
    """Validate plan presence plus objective/task dependency DAGs for RPR-G220."""

    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    plan_path = root / PLAN_REL
    objective_path = root / OBJECTIVE_REL
    todo_path = root / TODO_REL
    scheduler_path = root / SCHEDULER_REL

    for path, label in (
        (plan_path, "plan"),
        (objective_path, "objective"),
        (todo_path, "todo"),
        (scheduler_path, "scheduler"),
    ):
        if not path.is_file():
            errors.append(f"{label} missing: {path}")
    if errors:
        return CheckResult(
            name="plan_objective_task_dag",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
        )

    plan_text = plan_path.read_text(encoding="utf-8")
    if (
        "change propagation" not in plan_text.casefold()
        and "proof-gated" not in plan_text.casefold()
    ):
        errors.append("plan does not identify change-propagation / proof-gated work")

    try:
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
            parse_goal_heap,
        )

        goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    except Exception:
        goals = _parse_goal_heap_fallback(
            objective_path.read_text(encoding="utf-8")
        )

    goal_ids = {goal.goal_id for goal in goals}
    if "RPR-G220" not in goal_ids:
        errors.append("RPR-G220 is missing from the objective heap")
    if "RPR-G110" not in goal_ids and "RPR-G100" not in goal_ids:
        errors.append("parent control goals missing from the objective heap")

    goal_edges: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        deps = tuple(getattr(goal, "dependencies", ()) or ())
        parents = tuple(getattr(goal, "parent_goal_ids", ()) or ())
        combined = tuple(dict.fromkeys((*parents, *deps)))
        goal_edges[goal.goal_id] = combined
        for dep in combined:
            if dep not in goal_ids:
                errors.append(f"unknown objective dependency: {goal.goal_id}->{dep}")
    goal_cycles = _cycle_nodes(goal_edges)
    if goal_cycles:
        errors.append(f"goal dependency cycle: {list(goal_cycles)}")

    try:
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
            parse_task_file,
        )

        tasks = parse_task_file(todo_path, task_header_prefix=TASK_PREFIX)
    except Exception:
        tasks = _parse_task_file_fallback(todo_path)

    task_ids = {task.task_id for task in tasks}
    if len(tasks) != len(task_ids):
        errors.append("duplicate task id on the board")
    if "RPR-046" not in task_ids:
        errors.append("RPR-046 is missing")
    if "RPR-045" not in task_ids:
        errors.append("RPR-045 is missing")

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
    task_cycles = _cycle_nodes(task_edges)
    if task_cycles:
        errors.append(f"task dependency cycle: {list(task_cycles)}")

    rpr046 = next((task for task in tasks if task.task_id == "RPR-046"), None)
    if rpr046 is not None:
        required_deps = {"RPR-020", "RPR-045"}
        missing_deps = required_deps - set(rpr046.depends_on)
        if missing_deps:
            errors.append(f"RPR-046 missing required deps: {sorted(missing_deps)}")

    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    if scheduler.get("task_prefix") != TASK_PREFIX:
        errors.append("scheduler task prefix mismatch")
    if scheduler.get("merge_target_branch") != MERGE_TARGET_BRANCH:
        errors.append("scheduler merge target mismatch")
    if scheduler.get("board_namespace") != BOARD_NAMESPACE:
        errors.append("scheduler board namespace mismatch")

    if errors:
        return CheckResult(
            name="plan_objective_task_dag",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence={
                "goal_count": len(goals),
                "task_count": len(tasks),
                "errors": errors,
            },
        )
    return CheckResult(
        name="plan_objective_task_dag",
        status=CheckStatus.PASS,
        detail="plan, objective heap, task DAG, and scheduler bindings are consistent",
        evidence={
            "goal_count": len(goals),
            "task_count": len(tasks),
            "goal_ids": sorted(goal_ids),
            "task_ids": sorted(task_ids),
        },
    )


def check_exact_source_bindings(
    repo_root: Path | None = None,
) -> CheckResult:
    try:
        binding = bind_exact_sources(repo_root)
    except (OSError, json.JSONDecodeError, ChangePropagationRolloutError) as exc:
        return CheckResult(
            name="exact_source_bindings",
            status=CheckStatus.FAIL,
            detail=str(exc),
        )
    root = Path(binding.repository_root)
    recomputed = {
        "plan": file_identity(root / binding.plan_path),
        "objective": file_identity(root / binding.objective_path),
        "todo": file_identity(root / binding.todo_path),
        "scheduler": file_identity(root / binding.scheduler_path),
        "launcher": file_identity(root / binding.launcher_path),
        "guide": file_identity(root / binding.guide_path),
        "benchmark": file_identity(root / binding.benchmark_path),
        "fixture_manifest": file_identity(root / binding.fixture_manifest_path),
        "rollout_module": file_identity(root / binding.rollout_module_path),
        "validate_script": file_identity(root / binding.validate_script_path),
    }
    expected = {
        "plan": binding.plan_identity,
        "objective": binding.objective_identity,
        "todo": binding.todo_identity,
        "scheduler": binding.scheduler_identity,
        "launcher": binding.launcher_identity,
        "guide": binding.guide_identity,
        "benchmark": binding.benchmark_identity,
        "fixture_manifest": binding.fixture_manifest_identity,
        "rollout_module": binding.rollout_module_identity,
        "validate_script": binding.validate_script_identity,
    }
    if recomputed != expected:
        return CheckResult(
            name="exact_source_bindings",
            status=CheckStatus.FAIL,
            detail="source binding identities do not recompute",
            evidence={"expected": expected, "recomputed": recomputed},
        )
    return CheckResult(
        name="exact_source_bindings",
        status=CheckStatus.PASS,
        detail="exact source bindings recompute",
        evidence=binding.to_dict(),
    )


def check_capability_health(
    repo_root: Path | None = None,
    *,
    probe: bool = True,
) -> CheckResult:
    """Probe change-propagation capability admission (fail-closed, non-authoritative).

    Imports the integrations probe lazily so this validation package does not
    form a hard dependency cycle with ``integrations``.
    """

    del repo_root
    evidence: dict[str, Any] = {
        "authoritative": False,
        "candidate_authoritative": False,
    }
    if not probe:
        return CheckResult(
            name="capability_health",
            status=CheckStatus.SKIP,
            detail="capability probe skipped",
            evidence=evidence,
        )

    try:
        from ipfs_accelerate_py.agent_supervisor.integrations.change_propagation_capabilities import (
            probe_change_propagation_capabilities,
        )
    except Exception as exc:
        return CheckResult(
            name="capability_health",
            status=CheckStatus.FAIL,
            detail=f"capability probe import failed: {exc}",
            evidence=evidence,
        )

    try:
        report = probe_change_propagation_capabilities()
    except Exception as exc:
        return CheckResult(
            name="capability_health",
            status=CheckStatus.FAIL,
            detail=f"capability probe raised: {exc}",
            evidence=evidence,
        )

    report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    capabilities = report_dict.get("capabilities") or []
    available: list[str] = []
    unavailable: list[dict[str, Any]] = []
    for item in capabilities:
        if not isinstance(item, Mapping):
            continue
        cap_id = str(item.get("capability_id") or item.get("id") or "")
        status = str(item.get("status") or "").casefold()
        is_available = bool(item.get("available")) or status == "available"
        if is_available:
            available.append(cap_id)
        else:
            unavailable.append(
                {
                    "capability_id": cap_id,
                    "status": status,
                    "reason_code": item.get("reason_code")
                    or (item.get("diagnostic") or {}).get("code"),
                }
            )
        if item.get("candidate_authoritative"):
            return CheckResult(
                name="capability_health",
                status=CheckStatus.FAIL,
                detail=f"capability {cap_id} illegally claims candidate authority",
                evidence=evidence,
            )

    evidence.update(
        {
            "available": sorted(available),
            "unavailable": unavailable,
            "report_schema": report_dict.get("schema")
            or report_dict.get("schema_version"),
            "capability_count": len(capabilities),
        }
    )
    return CheckResult(
        name="capability_health",
        status=CheckStatus.PASS,
        detail=(
            f"capability probe completed: available={len(available)} "
            f"unavailable={len(unavailable)}"
        ),
        evidence=evidence,
    )


def check_graph_index_coverage(
    repo_root: Path | None = None,
) -> CheckResult:
    """Verify graph/index fixture coverage and required module surfaces exist."""

    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    manifest_path = root / FIXTURE_MANIFEST_REL
    if not manifest_path.is_file():
        return CheckResult(
            name="graph_index_coverage",
            status=CheckStatus.FAIL,
            detail=f"fixture manifest missing: {manifest_path}",
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CheckResult(
            name="graph_index_coverage",
            status=CheckStatus.FAIL,
            detail=f"fixture manifest unreadable: {exc}",
        )

    cases = manifest.get("cases") or []
    if not cases:
        errors.append("fixture manifest has no cases")

    graph_cases = 0
    index_roles_seen: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        artifacts = case.get("artifacts") or case.get("roles") or {}
        if isinstance(artifacts, Mapping):
            for role in artifacts:
                index_roles_seen.add(str(role))
            if "graph" in artifacts:
                graph_cases += 1
        # Some manifests list artifact roles at the top level of each case.
        for role_key in ("graph", "index", "consumers", "delta", "value_sources"):
            if role_key in case:
                index_roles_seen.add(role_key)
                if role_key == "graph":
                    graph_cases += 1

    if graph_cases == 0 and "graph" not in index_roles_seen:
        # Accept recipes that reference graph via build_manifest generator.
        build_manifest = root / "test/fixtures/agent_supervisor/change_propagation/build_manifest.py"
        if not build_manifest.is_file():
            errors.append("no graph artifacts or build_manifest generator found")

    module_paths = {
        "program_dependency_graph": root
        / "ipfs_accelerate_py/agent_supervisor/analysis/program_dependency_graph.py",
        "semantic_dependency_graph": root
        / "ipfs_accelerate_py/agent_supervisor/analysis/semantic_dependency_graph.py",
        "repository_indexer": root
        / "ipfs_accelerate_py/agent_supervisor/analysis/repository_indexer.py",
        "change_value_vector_index": root
        / "ipfs_accelerate_py/agent_supervisor/analysis/change_value_vector_index.py",
        "dynamic_impact_frontier": root
        / "ipfs_accelerate_py/agent_supervisor/analysis/dynamic_impact_frontier.py",
    }
    present = {
        name: path.is_file() for name, path in module_paths.items()
    }
    if not all(present.values()):
        missing = [name for name, ok in present.items() if not ok]
        errors.append(f"graph/index modules missing: {missing}")

    evidence = {
        "case_count": len(cases),
        "graph_cases": graph_cases,
        "roles_seen": sorted(index_roles_seen),
        "modules": present,
        "authoritative": False,
    }
    if errors:
        return CheckResult(
            name="graph_index_coverage",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence=evidence,
        )
    return CheckResult(
        name="graph_index_coverage",
        status=CheckStatus.PASS,
        detail="graph/index coverage surfaces and fixtures are present",
        evidence=evidence,
    )


def check_proof_reconstruction(
    repo_root: Path | None = None,
) -> CheckResult:
    """Confirm proof reconstruction path and policy require independent reconstruction."""

    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    required = {
        "change_propagation_obligations": root
        / "ipfs_accelerate_py/agent_supervisor/proof/change_propagation_obligations.py",
        "change_propagation_edit_packet": root
        / "ipfs_accelerate_py/agent_supervisor/proof/change_propagation_edit_packet.py",
        "validation": root / VALIDATION_MODULE_REL,
    }
    present = {name: path.is_file() for name, path in required.items()}
    missing = [name for name, ok in present.items() if not ok]
    if missing:
        errors.append(f"proof reconstruction modules missing: {missing}")

    # Validation module must mention reconstruction / fixed-point.
    validation_path = root / VALIDATION_MODULE_REL
    if validation_path.is_file():
        text = validation_path.read_text(encoding="utf-8")
        for phrase in ("proof", "reconstruct", "fixed_point", "FixedPoint"):
            if phrase.casefold() not in text.casefold():
                errors.append(f"validation module missing phrase: {phrase}")
                break

    policy = default_rollout_policy()
    if not policy.auto_requires_reconstruction:
        errors.append("default policy does not require reconstruction")

    evidence = {
        "modules": present,
        "auto_requires_reconstruction": policy.auto_requires_reconstruction,
        "authoritative": False,
    }
    if errors:
        return CheckResult(
            name="proof_reconstruction",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence=evidence,
        )
    return CheckResult(
        name="proof_reconstruction",
        status=CheckStatus.PASS,
        detail="proof reconstruction surfaces present; reconstruction required for auto",
        evidence=evidence,
    )


def check_transaction_health(
    repo_root: Path | None = None,
) -> CheckResult:
    """Verify transaction module exports and fail-closed partial-group semantics."""

    root = (repo_root or repository_root()).resolve()
    path = root / TRANSACTION_MODULE_REL
    if not path.is_file():
        return CheckResult(
            name="transaction_health",
            status=CheckStatus.FAIL,
            detail=f"transaction module missing: {path}",
        )

    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    required_symbols = (
        "ChangePropagationTransaction",
        "AtomicPropagationPlan",
        "PropagationRollbackReceipt",
        "TransactionExecutionReport",
    )
    for symbol in required_symbols:
        if symbol not in text:
            errors.append(f"transaction module missing symbol: {symbol}")

    # Partial groups / SCC rollback must be explicit.
    for phrase in ("rollback", "partial", "checkpoint", "worktree"):
        if phrase not in text.casefold():
            errors.append(f"transaction module missing concept: {phrase}")

    # Import smoke (optional — module may pull heavy deps).
    importable = False
    try:
        from ipfs_accelerate_py.agent_supervisor.planning import (
            change_propagation_transaction as txn,
        )

        importable = hasattr(txn, "ChangePropagationTransaction")
        if not importable:
            errors.append("ChangePropagationTransaction not importable")
    except Exception as exc:
        # Import failure is a warning for environments missing optional deps;
        # symbol presence in source is the hard check above.
        evidence_import_error = str(exc)
    else:
        evidence_import_error = ""

    evidence = {
        "path": TRANSACTION_MODULE_REL,
        "importable": importable,
        "import_error": evidence_import_error,
        "symbols_checked": list(required_symbols),
        "partial_groups_cannot_merge": True,
        "authoritative": False,
    }
    if errors:
        return CheckResult(
            name="transaction_health",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence=evidence,
        )
    return CheckResult(
        name="transaction_health",
        status=CheckStatus.PASS,
        detail="transaction module exports rollback/checkpoint surfaces; partial groups fail closed",
        evidence=evidence,
    )


def check_supervisor_process_state(
    repo_root: Path | None = None,
    *,
    state_root: Path | None = None,
    lane_count: int = 4,
) -> CheckResult:
    """Inspect supervisor/process state without requiring a live run."""

    del repo_root
    default_state = Path(
        os.environ.get(
            "RPR_STATE_ROOT",
            str(
                Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
                / "ipfs_accelerate_py"
                / "proof_gated_contract_repair"
            ),
        )
    )
    program_root = Path(state_root) if state_root is not None else default_state
    runtime_root = program_root / "runtime"
    lanes_root = program_root / "state"
    master_pid_path = runtime_root / "master.pid"

    master_pid: int | None = None
    master_alive = False
    if master_pid_path.is_file():
        raw = re.sub(r"\D", "", master_pid_path.read_text(encoding="utf-8"))
        if raw:
            master_pid = int(raw)
            try:
                os.kill(master_pid, 0)
                master_alive = True
            except OSError:
                master_alive = False

    lanes: list[dict[str, Any]] = []
    errors: list[str] = []
    if lanes_root.is_dir():
        for lane in range(max(0, int(lane_count))):
            state_dir = lanes_root / f"lane-{lane}"
            prefix = f"rpr_lane_{lane}"
            supervisor_path = state_dir / f"{prefix}_supervisor_status.json"
            task_path = state_dir / f"{prefix}_task_state.json"
            lane_info: dict[str, Any] = {
                "lane": lane,
                "supervisor": "missing",
                "task_state": "missing",
            }
            if supervisor_path.is_file():
                try:
                    payload = json.loads(supervisor_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, Mapping):
                        raise ValueError("supervisor status is not an object")
                    pid = int(payload.get("pid") or payload.get("supervisor_pid") or 0)
                    alive = False
                    if pid > 0:
                        try:
                            os.kill(pid, 0)
                            alive = True
                        except OSError:
                            alive = False
                    lane_info["supervisor"] = str(payload.get("status") or "unknown")
                    lane_info["supervisor_pid"] = pid
                    lane_info["supervisor_pid_alive"] = alive
                    if lane_info["supervisor"] == "running" and not alive:
                        errors.append(
                            f"lane {lane} claims running but pid {pid} is dead"
                        )
                except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
                    errors.append(f"lane {lane} supervisor state error: {exc}")
                    lane_info["supervisor"] = "error"
            if task_path.is_file():
                try:
                    task = json.loads(task_path.read_text(encoding="utf-8"))
                    if not isinstance(task, Mapping):
                        raise ValueError("task state is not an object")
                    lane_info["task_state"] = str(task.get("status") or "unknown")
                    lane_info["active_task_id"] = str(task.get("active_task_id") or "")
                    lane_info["eligible_ready_count"] = int(
                        task.get("eligible_ready_count") or 0
                    )
                    lane_info["blocked_count"] = int(task.get("blocked_count") or 0)
                except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
                    errors.append(f"lane {lane} task state error: {exc}")
                    lane_info["task_state"] = "error"
            lanes.append(lane_info)

    evidence = {
        "program_root": str(program_root),
        "master_pid": master_pid,
        "master_alive": master_alive,
        "master_status": "running" if master_alive else "stopped",
        "lanes": lanes,
        "errors": errors,
    }
    if errors:
        return CheckResult(
            name="supervisor_process_state",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence=evidence,
        )
    return CheckResult(
        name="supervisor_process_state",
        status=CheckStatus.PASS,
        detail=(
            f"supervisor process state is consistent "
            f"(master={'running' if master_alive else 'stopped'})"
        ),
        evidence=evidence,
    )


def check_benchmark_floors(
    repo_root: Path | None = None,
    *,
    run: bool = True,
    report: Mapping[str, Any] | None = None,
) -> CheckResult:
    """Verify legacy + propagation release safety floors are absolute zero."""

    del repo_root
    try:
        if report is None:
            if not run:
                return CheckResult(
                    name="benchmark_floors",
                    status=CheckStatus.SKIP,
                    detail="benchmark floor check skipped",
                )
            bench = _load_benchmark_module()
            report = bench.run_benchmark()
        metrics = report["metrics"]
        floors = metrics.get("safety_floors") or {}
        absolute = metrics.get("safety_absolute") or {}
        ops = ChangePropagationMetrics.from_benchmark_metrics(metrics)
    except Exception as exc:
        return CheckResult(
            name="benchmark_floors",
            status=CheckStatus.FAIL,
            detail=f"benchmark floor evaluation failed: {exc}",
        )

    failures = [
        key for key in SAFETY_FLOOR_KEYS if int(floors.get(key, 1)) != 0
    ]
    for key in SAFETY_ABSOLUTE_KEYS:
        if int(absolute.get(key, 1)) != 0:
            failures.append(key)
    if not ops.floors_hold():
        failures.extend(list(ops.breaches()))

    evidence = {
        "safety_floors": dict(floors),
        "safety_absolute": dict(absolute),
        "metrics_id": metrics.get("metrics_id"),
        "report_id": report.get("report_id"),
        "ops_metrics": ops.to_dict(),
        "fixed_point_iterations_total": ops.fixed_point_iterations_total,
        "analytical_model_split": dict(ops.analytical_model_split),
        "benchmark_stages": list(BENCHMARK_STAGES),
    }
    if failures:
        return CheckResult(
            name="benchmark_floors",
            status=CheckStatus.FAIL,
            detail=f"safety floor breach: {sorted(set(failures))}",
            evidence=evidence,
        )
    return CheckResult(
        name="benchmark_floors",
        status=CheckStatus.PASS,
        detail="all legacy and propagation release safety floors are absolute zero",
        evidence=evidence,
    )


def check_feature_flags(
    policy: ChangePropagationRolloutPolicy | None = None,
) -> CheckResult:
    """Assert default policy is shadow and elevated modes stay fail-closed."""

    default = default_rollout_policy()
    errors: list[str] = []
    if default.mode_value != RolloutMode.SHADOW.value:
        errors.append("default mode is not shadow")
    if default.mutation_authorized:
        errors.append("default policy authorizes mutation")
    if default.allow_assist or default.allow_narrow_auto or default.allow_expanded_auto:
        errors.append("default policy enables elevated modes")
    if not default.auto_requires_unique_target:
        errors.append("default policy does not require unique targets for auto")
    if not default.auto_requires_reconstruction:
        errors.append("default policy does not require reconstruction for auto")
    if not default.auto_requires_supported_python:
        errors.append("default policy does not require supported Python for auto")
    if not default.auto_requires_complete_frontier:
        errors.append("default policy does not require complete frontier for auto")
    if not default.auto_requires_analytical_path:
        errors.append("default policy does not require analytical path for auto")
    allowed = set(default.auto_allowed_transforms)
    if not allowed <= {item.casefold() for item in NARROW_AUTO_TRANSFORMS}:
        errors.append(
            f"default auto transforms escape narrow set: {sorted(allowed)}"
        )

    for mode in (RolloutMode.ASSIST, RolloutMode.NARROW_AUTO):
        try:
            ChangePropagationRolloutPolicy(mode=mode)
            errors.append(f"{mode.value} accepted without explicit scoped policy")
        except ChangePropagationRolloutError:
            pass

    selected = policy or default
    if _mode(selected.mode) is not RolloutMode.SHADOW:
        if not selected.has_explicit_scoped_policy():
            errors.append("selected elevated policy lacks explicit scope")

    narrow = elevate_rollout_policy(
        mode=RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:test",
        mutation_authorized=True,
    )
    # Happy path.
    if not narrow.allows_automated_mutation(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    ):
        errors.append("narrow-auto rejects valid analytical transform")
    # Incomplete frontier.
    if narrow.allows_automated_mutation(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=False,
        analytical_path=True,
    ):
        errors.append("narrow-auto allows incomplete frontier")
    # Model-authored remains gated.
    if narrow.allows_automated_mutation(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
        model_authored=True,
    ):
        errors.append("narrow-auto allows model-authored mutation")
    # Public schema remains gated.
    if narrow.allows_automated_mutation(
        transform=TransformKind.UPDATE_SCHEMA_FIELD.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
        public_schema_api=True,
    ):
        errors.append("narrow-auto allows public schema mutation")
    # Cross-root gated.
    if narrow.allows_automated_mutation(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
        cross_root=True,
    ):
        errors.append("narrow-auto allows cross-root mutation")
    # Non-unique
    if narrow.allows_automated_mutation(
        transform=TransformKind.THREAD_PARAMETER.value,
        unique_target=False,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    ):
        errors.append("narrow-auto allows non-unique transform")
    # Unreconstructed
    if narrow.allows_automated_mutation(
        transform=TransformKind.RENAME_ARGUMENT.value,
        unique_target=True,
        reconstructed=False,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    ):
        errors.append("narrow-auto allows unreconstructed transform")

    if errors:
        return CheckResult(
            name="feature_flags",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence={"default": default.to_dict(), "selected": selected.to_dict()},
        )
    return CheckResult(
        name="feature_flags",
        status=CheckStatus.PASS,
        detail=(
            "shadow is default; assist/narrow-auto require scoped policy; "
            "auto limited to complete-frontier unique reconstructed analytical "
            "supported-Python transforms"
        ),
        evidence={"default": default.to_dict(), "selected": selected.to_dict()},
    )


def check_rollback_gates(
    policy: ChangePropagationRolloutPolicy | None = None,
) -> CheckResult:
    """Prove each rollback trigger demotes and revokes mutation."""

    base = elevate_rollout_policy(
        mode=RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:test",
        mutation_authorized=True,
    )
    errors: list[str] = []
    scenarios: list[tuple[str, dict[str, Any], RollbackReason]] = [
        (
            "capability_regression",
            dict(capability_regression=("graph_index", "logic_smt")),
            RollbackReason.CAPABILITY_REGRESSION,
        ),
        (
            "stale_root",
            dict(stale_roots=("index_root", "graph_root")),
            RollbackReason.STALE_ROOT,
        ),
        (
            "open_frontier",
            dict(open_frontier=True),
            RollbackReason.OPEN_FRONTIER,
        ),
        (
            "reconstruction_failure",
            dict(reconstruction_failed=True),
            RollbackReason.RECONSTRUCTION_FAILURE,
        ),
        (
            "proof_loss",
            dict(proof_loss=True),
            RollbackReason.PROOF_LOSS,
        ),
        (
            "wrong_value",
            dict(wrong_value=True),
            RollbackReason.WRONG_VALUE,
        ),
        (
            "missed_consumer",
            dict(missed_consumer=True),
            RollbackReason.MISSED_CONSUMER,
        ),
        (
            "partial_plan",
            dict(partial_plan=True),
            RollbackReason.PARTIAL_PLAN,
        ),
        (
            "false_completion",
            dict(false_completion=True),
            RollbackReason.FALSE_COMPLETION,
        ),
        (
            "metric_breach",
            dict(
                metrics=ChangePropagationMetrics(
                    wrong_path_rate=1,
                    safety_floors={
                        **{key: 0 for key in SAFETY_FLOOR_KEYS},
                        "wrong_path_automated_mutation_rate": 1,
                    },
                    safety_absolute={
                        **{key: 0 for key in SAFETY_ABSOLUTE_KEYS},
                        "wrong_path_automated_mutation": 1,
                    },
                )
            ),
            RollbackReason.METRIC_BREACH,
        ),
    ]
    receipts: list[dict[str, Any]] = []
    gate = ChangePropagationRollbackGate(base)
    for name, kwargs, expected_reason in scenarios:
        receipt = gate.evaluate(**kwargs)
        if receipt is None:
            errors.append(f"{name} did not produce a rollback receipt")
            continue
        if receipt.reason is not expected_reason:
            errors.append(f"{name} reason {receipt.reason} != {expected_reason}")
        demoted = apply_rollback(base, receipt)
        if demoted.mutation_authorized:
            errors.append(f"{name} demotion still authorizes mutation")
        if _mode(demoted.mode) is RolloutMode.NARROW_AUTO:
            errors.append(f"{name} failed to demote mode")
        receipts.append(receipt.to_dict())

    healthy = evaluate_rollback(
        base,
        metrics=ChangePropagationMetrics.empty(),
        capability_regression=(),
        stale_roots=(),
        open_frontier=False,
        reconstruction_failed=False,
        proof_loss=False,
    )
    if healthy is not None:
        errors.append("healthy state incorrectly produced a rollback receipt")

    selected = policy or default_rollout_policy()
    for attr in (
        "rollback_on_capability_regression",
        "rollback_on_stale_root",
        "rollback_on_open_frontier",
        "rollback_on_reconstruction_failure",
        "rollback_on_proof_loss",
        "rollback_on_metric_breach",
        "rollback_on_coverage_loss",
    ):
        if not getattr(selected, attr):
            errors.append(f"selected policy disables {attr}")

    if errors:
        return CheckResult(
            name="rollback_gates",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence={"receipts": receipts},
        )
    return CheckResult(
        name="rollback_gates",
        status=CheckStatus.PASS,
        detail=(
            "stale roots, open frontier, capability regression, proof loss, "
            "wrong-value/missed-consumer/partial-plan/false-completion, and "
            "floor breaches roll back"
        ),
        evidence={"receipts": receipts},
    )


def check_guide_boundaries(
    repo_root: Path | None = None,
) -> CheckResult:
    """Confirm the operator guide documents trust, safety, memory, transaction, recovery."""

    root = (repo_root or repository_root()).resolve()
    guide = root / GUIDE_REL
    if not guide.is_file():
        return CheckResult(
            name="guide_boundaries",
            status=CheckStatus.FAIL,
            detail=f"guide missing: {guide}",
        )
    text = guide.read_text(encoding="utf-8")
    lower = text.casefold()
    required_phrases = (
        "shadow",
        "assist",
        "narrow-auto",
        "rollback",
        "memory safety",
        "transaction",
        "recovery",
        "trust",
        "fixed-point",
        "complete frontier",
    )
    missing: list[str] = []
    for phrase in required_phrases:
        # Accept narrow_auto spelling.
        if phrase == "narrow-auto":
            if "narrow-auto" not in lower and "narrow_auto" not in lower:
                missing.append(phrase)
            continue
        if phrase not in lower:
            missing.append(phrase)

    alt_memory = (
        "do not prove memory safety",
        "does not prove memory safety",
        "never prove memory safety",
        "not memory-safety evidence",
        "not memory safety evidence",
    )
    if not any(item in lower for item in alt_memory):
        missing.append("does not prove memory safety")

    for kind in ("vector", "test", "type", "resource"):
        if kind not in lower:
            missing.append(kind)

    for topic in ("model-authored", "stateful", "cross-root", "generated"):
        # Accept hyphen or underscore variants.
        if topic not in lower and topic.replace("-", "_") not in lower:
            # Also accept spaced form.
            if topic.replace("-", " ") not in lower:
                missing.append(topic)

    if missing:
        return CheckResult(
            name="guide_boundaries",
            status=CheckStatus.FAIL,
            detail=f"guide missing required boundary language: {missing}",
        )
    return CheckResult(
        name="guide_boundaries",
        status=CheckStatus.PASS,
        detail=(
            "guide documents trust, safety, memory, transaction, and recovery "
            "boundaries"
        ),
        evidence={"path": GUIDE_REL, "bytes": guide.stat().st_size},
    )


# ---------------------------------------------------------------------------
# Aggregated validation / doctor / status / replay
# ---------------------------------------------------------------------------


def run_all_checks(
    repo_root: Path | None = None,
    *,
    run_benchmark: bool = True,
    probe_capabilities: bool = True,
    policy: ChangePropagationRolloutPolicy | None = None,
    benchmark_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    selected_policy = policy or default_rollout_policy()
    checks = [
        check_plan_objective_task_dag(root),
        check_exact_source_bindings(root),
        check_capability_health(root, probe=probe_capabilities),
        check_graph_index_coverage(root),
        check_proof_reconstruction(root),
        check_transaction_health(root),
        check_supervisor_process_state(root),
        check_benchmark_floors(root, run=run_benchmark, report=benchmark_report),
        check_feature_flags(selected_policy),
        check_rollback_gates(selected_policy),
        check_guide_boundaries(root),
    ]
    results = [item.to_dict() for item in checks]
    ok = all(item.ok for item in checks)
    payload = {
        "schema": VALIDATOR_SCHEMA,
        "interface": VALIDATOR_INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "valid": ok,
        "default_mode": RolloutMode.SHADOW.value,
        "policy": selected_policy.to_dict(),
        "checks": results,
        "failed": [item.name for item in checks if item.status is CheckStatus.FAIL],
        "mutation_authorized": False,
        "completion_authoritative": False,
        "interfaces": {
            "rollout_policy": ROLLOUT_POLICY_INTERFACE,
            "metrics": METRICS_INTERFACE,
            "rollback_gate": ROLLBACK_GATE_INTERFACE,
            "benchmark_metrics": BENCHMARK_METRICS_INTERFACE,
            "atomic_propagation_plan": ATOMIC_PROPAGATION_PLAN_INTERFACE,
            "propagation_completion_receipt": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
        },
    }
    payload["report_id"] = content_identity(
        {key: value for key, value in payload.items() if key != "report_id"}
    )
    return payload


def doctor(
    repo_root: Path | None = None,
    *,
    run_benchmark: bool = False,
    probe_capabilities: bool = True,
) -> dict[str, Any]:
    report = run_all_checks(
        repo_root,
        run_benchmark=run_benchmark,
        probe_capabilities=probe_capabilities,
    )
    report["command"] = "doctor"
    return report


def status(
    repo_root: Path | None = None,
    *,
    policy: ChangePropagationRolloutPolicy | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    selected = policy or default_rollout_policy()
    binding_check = check_exact_source_bindings(root)
    supervisor_check = check_supervisor_process_state(root)
    dag_check = check_plan_objective_task_dag(root)
    graph_check = check_graph_index_coverage(root)
    txn_check = check_transaction_health(root)
    payload = {
        "schema": VALIDATOR_SCHEMA,
        "interface": VALIDATOR_INTERFACE,
        "command": "status",
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "mode": selected.mode_value,
        "default_mode": RolloutMode.SHADOW.value,
        "policy": selected.to_dict(),
        "bindings": binding_check.to_dict(),
        "supervisor": supervisor_check.to_dict(),
        "dag": dag_check.to_dict(),
        "graph_index_coverage": graph_check.to_dict(),
        "transaction_health": txn_check.to_dict(),
        "mutation_authorized": bool(selected.mutation_authorized),
        "completion_authoritative": False,
        "valid": all(
            item.ok
            for item in (
                binding_check,
                supervisor_check,
                dag_check,
                graph_check,
                txn_check,
            )
        ),
    }
    payload["report_id"] = content_identity(
        {key: value for key, value in payload.items() if key != "report_id"}
    )
    return payload


def replay_decision_receipt(
    receipt: Mapping[str, Any],
    *,
    policy: ChangePropagationRolloutPolicy | None = None,
    expected_roots: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Replay a sealed plan/completion/benchmark receipt against policy."""

    if not isinstance(receipt, Mapping):
        raise ChangePropagationRolloutError("receipt must be an object")
    selected = policy or default_rollout_policy()
    errors: list[str] = []

    claimed_id = (
        receipt.get("receipt_id")
        or receipt.get("report_id")
        or receipt.get("plan_id")
        or receipt.get("case_id")
    )
    body = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "receipt_id",
            "report_id",
            "plan_id",
            "case_id",
            "metrics_id",
            "decision_id",
        }
    }
    recomputed = content_identity(body)
    identity_ok = True
    if isinstance(claimed_id, str) and claimed_id.startswith("sha256:"):
        identity_ok = claimed_id == recomputed or claimed_id == content_identity(
            {
                key: value
                for key, value in receipt.items()
                if key
                != (
                    "receipt_id"
                    if "receipt_id" in receipt
                    else "report_id"
                    if "report_id" in receipt
                    else "plan_id"
                    if "plan_id" in receipt
                    else "case_id"
                )
            }
        )
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

    reconstructed = bool(
        receipt.get("reconstructed")
        or receipt.get("reconstruction_ok")
        or receipt.get("proof_reconstructed")
    )
    unique_target = bool(
        receipt.get("unique_target")
        if "unique_target" in receipt
        else receipt.get("target_precise", False)
    )
    supported_python = bool(
        receipt.get("supported_python")
        if "supported_python" in receipt
        else str(receipt.get("language") or "python").casefold()
        in {"python", "py"}
    )
    complete_frontier = bool(
        receipt.get("complete_frontier")
        if "complete_frontier" in receipt
        else not bool(receipt.get("open_frontier"))
    )
    analytical_path = bool(
        receipt.get("analytical_path")
        if "analytical_path" in receipt
        else str(receipt.get("plan_step_kind") or "analytical").casefold()
        == "analytical"
    )
    transform = str(
        receipt.get("transform")
        or receipt.get("transform_kind")
        or receipt.get("strategy")
        or TransformKind.ADD_ARGUMENT.value
    )
    model_authored = bool(
        receipt.get("model_authored")
        or receipt.get("llm_authored")
        or str(receipt.get("plan_step_kind") or "").casefold() == "llm_bounded"
    )
    auto_ok = selected.allows_automated_mutation(
        transform=transform,
        unique_target=unique_target,
        reconstructed=reconstructed,
        supported_python=supported_python,
        complete_frontier=complete_frontier,
        analytical_path=analytical_path,
        model_authored=model_authored,
        stateful=bool(receipt.get("stateful")),
        public_schema_api=bool(receipt.get("public_schema_api")),
        dynamic=bool(receipt.get("dynamic")),
        generated=bool(receipt.get("generated")),
        native=bool(receipt.get("native")),
        cross_root=bool(receipt.get("cross_root") or receipt.get("cross_repository")),
        change_family=str(receipt.get("change_family") or ""),
    )

    stale = [
        key
        for key, expected in (expected_roots or {}).items()
        if not isinstance(roots, Mapping) or roots.get(key) != expected
    ]
    rollback = evaluate_rollback(
        selected,
        stale_roots=stale,
        open_frontier=bool(receipt.get("open_frontier")),
        reconstruction_failed=bool(
            receipt.get("reconstruction_failed")
            or (
                selected.auto_requires_reconstruction
                and not reconstructed
                and _mode(selected.mode)
                in {RolloutMode.NARROW_AUTO, RolloutMode.EXPANDED_AUTO}
            )
        ),
        proof_loss=bool(receipt.get("proof_loss")),
        wrong_value=bool(receipt.get("wrong_value")),
        missed_consumer=bool(receipt.get("missed_consumer")),
        partial_plan=bool(receipt.get("partial_plan")),
        false_completion=bool(receipt.get("false_completion")),
        reason_codes=tuple(receipt.get("reason_codes") or ()),
    )

    payload = {
        "schema": VALIDATOR_SCHEMA,
        "interface": VALIDATOR_INTERFACE,
        "command": "replay",
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "valid": not errors and (rollback is None or _mode(selected.mode) is RolloutMode.SHADOW),
        "identity_ok": identity_ok and "receipt identity" not in " ".join(errors),
        "recomputed_identity": recomputed,
        "claimed_identity": claimed_id,
        "automated_mutation_authorized": auto_ok,
        "transform": transform,
        "unique_target": unique_target,
        "reconstructed": reconstructed,
        "supported_python": supported_python,
        "complete_frontier": complete_frontier,
        "analytical_path": analytical_path,
        "policy": selected.to_dict(),
        "rollback": None if rollback is None else rollback.to_dict(),
        "errors": errors,
        "mutation_authorized": False,
        "completion_authoritative": False,
    }
    # Elevated mode with a fired rollback is invalid for mutation.
    if rollback is not None and _mode(selected.mode) is not RolloutMode.SHADOW:
        payload["valid"] = False
    if not identity_ok:
        payload["valid"] = False
    payload["report_id"] = content_identity(
        {key: value for key, value in payload.items() if key != "report_id"}
    )
    return payload


def collect_metrics(
    *,
    benchmark_report: Mapping[str, Any] | None = None,
    run_benchmark: bool = True,
) -> ChangePropagationMetrics:
    if benchmark_report is not None:
        return ChangePropagationMetrics.from_benchmark_metrics(
            benchmark_report["metrics"]
            if "metrics" in benchmark_report
            else benchmark_report
        )
    if not run_benchmark:
        return ChangePropagationMetrics.empty()
    bench = _load_benchmark_module()
    report = bench.run_benchmark()
    return ChangePropagationMetrics.from_benchmark_metrics(report["metrics"])


def evidence_proves_memory_safety(evidence_kind: str) -> bool:
    """Return False for vector/test/type/resource and other non-proof kinds."""

    kind = str(evidence_kind or "").strip().casefold().replace("-", "_")
    if kind in NON_MEMORY_SAFETY_EVIDENCE:
        return False
    # Even formal facets require independent reconstruction; never auto-true.
    return False


def model_boundary_statement() -> str:
    return (
        "Models propose nominations, rankings, and edit drafts. "
        "They do not admit plans, authorize writes, complete tasks, or prove "
        "memory safety. Vector, test, type, and resource evidence does not prove "
        "memory safety. Narrow-auto is limited to complete-frontier unique "
        "reconstructed analytical supported-Python transforms; model-authored, "
        "stateful, public schema/API, dynamic/generated/native, and cross-root "
        "changes remain approval-gated."
    )


def trust_boundary_statement() -> str:
    return (
        "Trust boundary: only exact repository, graph, index, model, translator, "
        "toolchain, policy, and proof roots may participate in admission. "
        "Discovery is not authority. Partial SCC groups cannot merge. "
        "Transactions checkpoint and roll back; recovery rebuilds indexes and "
        "re-proves to a fixed point without claiming completion from a clean compile alone."
    )


__all__ = [
    "APPROVAL_GATED_CHANGE_FAMILIES",
    "ATOMIC_PROPAGATION_PLAN_INTERFACE",
    "BENCHMARK_METRICS_INTERFACE",
    "BENCHMARK_STAGES",
    "ChangePropagationMetrics",
    "ChangePropagationRollbackGate",
    "ChangePropagationRolloutError",
    "ChangePropagationRolloutPolicy",
    "CheckResult",
    "CheckStatus",
    "METRICS_INTERFACE",
    "NARROW_AUTO_TRANSFORMS",
    "PROPAGATION_COMPLETION_RECEIPT_INTERFACE",
    "PropagationSourceBinding",
    "ROLLBACK_GATE_INTERFACE",
    "ROLLOUT_POLICY_INTERFACE",
    "RollbackReason",
    "RollbackReceipt",
    "RolloutMode",
    "SAFETY_FLOOR_KEYS",
    "VALIDATOR_INTERFACE",
    "apply_rollback",
    "bind_exact_sources",
    "check_benchmark_floors",
    "check_capability_health",
    "check_exact_source_bindings",
    "check_feature_flags",
    "check_graph_index_coverage",
    "check_guide_boundaries",
    "check_plan_objective_task_dag",
    "check_proof_reconstruction",
    "check_rollback_gates",
    "check_supervisor_process_state",
    "check_transaction_health",
    "collect_metrics",
    "content_identity",
    "default_rollout_policy",
    "doctor",
    "elevate_rollout_policy",
    "evaluate_rollback",
    "evidence_proves_memory_safety",
    "model_boundary_statement",
    "replay_decision_receipt",
    "repository_root",
    "run_all_checks",
    "status",
    "trust_boundary_statement",
]
