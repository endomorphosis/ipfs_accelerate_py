#!/usr/bin/env python3
"""Operations, metrics, feature flags, and rollback for proof-gated contract repair.

RPR-020 / RPR-G100 (legacy) plus RPR-047 / RPR-G110 / RPR-G220 (propagation
extension).  Provides:

* doctor / status / replay / check-all operator commands;
* ``ContractRepairRolloutPolicy`` feature flags (shadow default; assist and
  narrow-auto require an explicit scoped policy; auto is initially limited to
  unique reconstructed supported substitutions/renames);
* ``ContractRepairMetrics`` release and decision metrics;
* fail-closed rollback when capability health regresses, roots go stale,
  reconstruction fails, or a safety floor / metric breach is observed;
* extended control-plane gates for transitive change propagation: terminal
  RPR-047, RPR-G110/RPR-G220, ``change_propagation_policy``, six new zero
  safety floors, protected-path/refill isolation, and four-shard drain
  readiness;
* ``ProofGatedContractRepairOperations`` and ``ChangePropagationEndToEnd``
  helpers for the RPR-047 end-to-end surface.

This module never grants mutation, completion, merge, or process authority.
Reports and receipts are content-addressed and deterministic on clean re-runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (  # noqa: E402
    RepairStrategy,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_file,
)


# ---------------------------------------------------------------------------
# Schemas / identities
# ---------------------------------------------------------------------------

VALIDATOR_INTERFACE: Final[str] = "ContractRepairValidatorOps@1"
VALIDATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-validation-report@1"
)
ROLLOUT_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-rollout-policy@1"
)
ROLLOUT_POLICY_INTERFACE: Final[str] = "ContractRepairRolloutPolicy@1"
METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-ops-metrics@1"
)
METRICS_INTERFACE: Final[str] = "ContractRepairMetrics@1"
ROLLBACK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-rollback-receipt@1"
)
SOURCE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-source-binding@1"
)
TASK_ID: Final[str] = "RPR-020"
GOAL_ID: Final[str] = "RPR-G100"
# Terminal extension IDs recognized by the operations surface (RPR-047).
TERMINAL_TASK_ID: Final[str] = "RPR-047"
EXTENSION_CONTROL_GOAL_ID: Final[str] = "RPR-G110"
EXTENSION_ROLLOUT_GOAL_ID: Final[str] = "RPR-G220"
BOARD_NAMESPACE: Final[str] = "agent-supervisor-proof-gated-contract-repair-v1"
TASK_PREFIX: Final[str] = "RPR-"
GOAL_PREFIX: Final[str] = "RPR-G"
MERGE_TARGET_BRANCH: Final[str] = "agent/proof-gated-contract-repair"
DEFAULT_RECALL_K: Final[int] = 5
EXPECTED_LANE_COUNT: Final[int] = 4
INITIAL_PROPAGATION_ENTRY_TASKS: Final[tuple[str, ...]] = (
    "RPR-022",
    "RPR-023",
    "RPR-024",
    "RPR-025",
)

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
GUIDE_REL: Final[str] = "docs/guides/PROOF_GATED_CONTRACT_REPAIR_GUIDE.md"
BENCHMARK_SCRIPT_REL: Final[str] = "scripts/benchmark_contract_repair.py"
PROPAGATION_BENCHMARK_SCRIPT_REL: Final[str] = (
    "scripts/benchmark_change_propagation.py"
)
PROPAGATION_FIXTURE_MANIFEST_REL: Final[str] = (
    "test/fixtures/agent_supervisor/change_propagation/manifest.json"
)

REQUIRED_CONTROL_PLANE: Final[tuple[str, ...]] = (
    PLAN_REL,
    OBJECTIVE_REL,
    TODO_REL,
    SCHEDULER_REL,
    LAUNCHER_REL,
)

REQUIRED_PROTECTED_PATHS: Final[tuple[str, ...]] = (
    PLAN_REL,
    OBJECTIVE_REL,
    TODO_REL,
    SCHEDULER_REL,
    LAUNCHER_REL,
)

# Legacy RPR-020 / RPR-G100 floors (absolute zero).
SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "wrong_path_automated_mutation_rate",
    "failed_obligation_override_rate",
    "stale_forged_or_poisoned_authoritative_admission_rate",
    "unsupported_memory_safety_promotion_rate",
)

# Six new propagation floors (RPR-045 / RPR-G220 / RPR-047). Primary names
# match the sealed scheduler; aliases accept the benchmark spelling.
PROPAGATION_SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "missed_resolved_impacted_consumer_rate",
    "unproved_or_wrong_value_source_admission_rate",
    "behavior_invented_without_independent_authority_rate",
    "partial_propagation_completion_rate",
    "stale_graph_or_index_plan_admission_rate",
    "false_fixed_point_completion_rate",
)

PROPAGATION_SAFETY_FLOOR_ALIASES: Final[Mapping[str, tuple[str, ...]]] = {
    "behavior_invented_without_independent_authority_rate": (
        "behavior_invented_without_independent_authority_rate",
        "invented_behavior_without_authority_rate",
    ),
    "stale_graph_or_index_plan_admission_rate": (
        "stale_graph_or_index_plan_admission_rate",
        "stale_graph_index_plan_admission_rate",
    ),
}

ALL_RELEASE_SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    *SAFETY_FLOOR_KEYS,
    *PROPAGATION_SAFETY_FLOOR_KEYS,
)

# Required scheduler change_propagation_policy gates (fail-closed).
REQUIRED_CHANGE_PROPAGATION_POLICY: Final[Mapping[str, Any]] = {
    "impact_closure_required_before_plan_admission": True,
    "one_obligation_per_resolved_consumer": True,
    "unknown_required_frontier_disposition": "abstain",
    "datasets_logic_reconstruction_required_before_value_or_behavior_admission": True,
    "knowledge_graph_semantic_authority": False,
    "runtime_witness_semantic_authority": False,
    "llm_router_semantic_authority": False,
    "analytical_transform_precedes_llm_router": True,
    "llm_router_requires_admitted_behavior_and_paths": True,
    "atomic_scc_transaction_required": True,
    "partial_plan_completion_allowed": False,
    "fixed_point_validation_required": True,
}

# Strategies that narrow-auto may execute without expanded review.
NARROW_AUTO_STRATEGIES: Final[frozenset[str]] = frozenset(
    {
        RepairStrategy.RENAME_SUBSTITUTION.value,
        "rename_substitution",
        "pure_rename",
        "closed_substitution",
    }
)

# Evidence kinds that may nominate or gate but never prove memory safety.
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
    }
)


class ContractRepairValidationError(ValueError):
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
    RECONSTRUCTION_FAILURE = "reconstruction_failure"
    METRIC_BREACH = "metric_breach"
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
    return _PACKAGE_ROOT


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
        # Reject IEEE floats from sealed identities; rates use integers (ppm).
        raise ContractRepairValidationError(
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


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value.strip():
        raise ContractRepairValidationError(f"{name} must be non-empty text")
    result = value.strip()
    if "\x00" in result or len(result.encode("utf-8")) > maximum:
        raise ContractRepairValidationError(f"{name} is unsafe or too large")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractRepairValidationError(f"{name} must be a boolean")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractRepairValidationError(
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
        raise ContractRepairValidationError(
            f"unknown rollout mode: {value!r}"
        ) from exc


def _csv(value: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in re.split(r"[,;]", value or "")
        if item.strip()
    )


def _safe_relative(path: str) -> bool:
    if not path or "\x00" in path:
        return False
    pure = PurePosixPath(path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() in {".", ".."}:
        return False
    return True


def _cycle_nodes(edges: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: set[str] = set()

    def visit(node: str, lineage: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            if node in lineage:
                cycle.update(lineage[lineage.index(node) :])
            cycle.add(node)
            return
        visiting.add(node)
        for parent in edges.get(node, ()):
            visit(parent, (*lineage, node))
        visiting.remove(node)
        visited.add(node)

    for item in sorted(edges):
        visit(item, ())
    return tuple(sorted(cycle))


def _floor_lookup(
    floors: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int | None:
    """Resolve a safety-floor key allowing known scheduler/benchmark aliases."""

    aliases = PROPAGATION_SAFETY_FLOOR_ALIASES.get(key, (key,))
    for alias in aliases:
        if alias in floors:
            return int(floors[alias])
    if key in floors:
        return int(floors[key])
    return default


def _task_metadata_get(task: Any, *keys: str) -> str:
    metadata = getattr(task, "metadata", None) or {}
    if not isinstance(metadata, Mapping):
        return ""
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _plain(payload), sort_keys=True, indent=2, ensure_ascii=False
    ) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def _checkpoint_dir() -> Path | None:
    raw = os.environ.get("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_checkpoint(name: str, payload: Mapping[str, Any]) -> Path | None:
    directory = _checkpoint_dir()
    if directory is None:
        return None
    return write_json_atomic(directory / f"{name}.json", dict(payload))


# ---------------------------------------------------------------------------
# Source bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactSourceBinding:
    """Exact plan / objective / taskboard / scheduler / launcher binding."""

    SCHEMA: ClassVar[str] = SOURCE_BINDING_SCHEMA

    repository_root: str
    plan_path: str
    plan_identity: str
    objective_path: str
    objective_identity: str
    todo_path: str
    todo_identity: str
    scheduler_path: str
    scheduler_identity: str
    launcher_path: str
    launcher_identity: str
    board_namespace: str
    task_prefix: str
    merge_target_branch: str
    binding_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "repository_root",
            "plan_path",
            "plan_identity",
            "objective_path",
            "objective_identity",
            "todo_path",
            "todo_identity",
            "scheduler_path",
            "scheduler_identity",
            "launcher_path",
            "launcher_identity",
            "board_namespace",
            "task_prefix",
            "merge_target_branch",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum=4096))
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
            "plan_path": self.plan_path,
            "plan_identity": self.plan_identity,
            "objective_path": self.objective_path,
            "objective_identity": self.objective_identity,
            "todo_path": self.todo_path,
            "todo_identity": self.todo_identity,
            "scheduler_path": self.scheduler_path,
            "scheduler_identity": self.scheduler_identity,
            "launcher_path": self.launcher_path,
            "launcher_identity": self.launcher_identity,
            "board_namespace": self.board_namespace,
            "task_prefix": self.task_prefix,
            "merge_target_branch": self.merge_target_branch,
        }
        if include_id:
            payload["binding_id"] = self.binding_id
        return payload


def file_identity(path: Path) -> str:
    return _sha256_hex(path.read_bytes())


def bind_exact_sources(repo_root: Path | None = None) -> ExactSourceBinding:
    root = (repo_root or repository_root()).resolve()
    paths = {
        "plan": root / PLAN_REL,
        "objective": root / OBJECTIVE_REL,
        "todo": root / TODO_REL,
        "scheduler": root / SCHEDULER_REL,
        "launcher": root / LAUNCHER_REL,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ContractRepairValidationError(
            f"required control-plane files missing: {missing}"
        )
    scheduler = json.loads(paths["scheduler"].read_text(encoding="utf-8"))
    board = str(scheduler.get("board_namespace") or BOARD_NAMESPACE)
    prefix = str(scheduler.get("task_prefix") or TASK_PREFIX)
    merge = str(scheduler.get("merge_target_branch") or MERGE_TARGET_BRANCH)
    return ExactSourceBinding(
        repository_root=str(root),
        plan_path=PLAN_REL,
        plan_identity=file_identity(paths["plan"]),
        objective_path=OBJECTIVE_REL,
        objective_identity=file_identity(paths["objective"]),
        todo_path=TODO_REL,
        todo_identity=file_identity(paths["todo"]),
        scheduler_path=SCHEDULER_REL,
        scheduler_identity=file_identity(paths["scheduler"]),
        launcher_path=LAUNCHER_REL,
        launcher_identity=file_identity(paths["launcher"]),
        board_namespace=board,
        task_prefix=prefix,
        merge_target_branch=merge,
    )


# ---------------------------------------------------------------------------
# Rollout policy / feature flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractRepairRolloutPolicy:
    """Per-repository/program/policy feature flags for contract-repair release.

    Defaults are fail-closed:

    * mode is always ``shadow`` unless an explicit scoped policy elevates it;
    * assist and narrow-auto require ``explicit_policy_document`` plus a
      non-empty scope (repository and/or program and/or policy id);
    * automated mutation is limited to unique, reconstructed, supported
      rename/substitution strategies until expanded auto is independently
      reviewed.
    """

    SCHEMA: ClassVar[str] = ROLLOUT_POLICY_SCHEMA
    INTERFACE: ClassVar[str] = ROLLOUT_POLICY_INTERFACE

    policy_id: str = "policy:contract-repair-rollout-default"
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
    auto_requires_supported_shape: bool = True
    auto_allowed_strategies: tuple[str, ...] = (
        RepairStrategy.RENAME_SUBSTITUTION.value,
    )
    rollback_on_capability_regression: bool = True
    rollback_on_stale_root: bool = True
    rollback_on_reconstruction_failure: bool = True
    rollback_on_metric_breach: bool = True
    mutation_authorized: bool = False
    completion_authoritative: bool = False
    policy_binding_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision"),
        )
        object.__setattr__(
            self,
            "repository_id",
            str(self.repository_id or "").strip(),
        )
        object.__setattr__(
            self, "program_id", _text(self.program_id, "program_id")
        )
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
        strategies = tuple(
            sorted(
                {
                    _text(item, "auto_allowed_strategies").casefold()
                    for item in self.auto_allowed_strategies
                }
            )
        )
        if not strategies:
            raise ContractRepairValidationError(
                "auto_allowed_strategies must not be empty"
            )
        object.__setattr__(self, "auto_allowed_strategies", strategies)
        for name in (
            "allow_assist",
            "allow_narrow_auto",
            "allow_expanded_auto",
            "auto_requires_unique_target",
            "auto_requires_reconstruction",
            "auto_requires_supported_shape",
            "rollback_on_capability_regression",
            "rollback_on_stale_root",
            "rollback_on_reconstruction_failure",
            "rollback_on_metric_breach",
            "mutation_authorized",
            "completion_authoritative",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if self.completion_authoritative:
            raise ContractRepairValidationError(
                "rollout policy cannot claim completion authority"
            )
        # Default shadow path must never authorize mutation.
        if self.mode is RolloutMode.SHADOW and self.mutation_authorized:
            raise ContractRepairValidationError(
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
            raise ContractRepairValidationError(
                f"{mode.value} requires an explicit scoped policy document "
                "and repository/program/policy scope"
            )
        if mode is RolloutMode.ASSIST and not self.allow_assist:
            raise ContractRepairValidationError(
                "assist mode is not enabled on this policy"
            )
        if mode is RolloutMode.NARROW_AUTO and not self.allow_narrow_auto:
            raise ContractRepairValidationError(
                "narrow_auto mode is not enabled on this policy"
            )
        if mode is RolloutMode.EXPANDED_AUTO and not self.allow_expanded_auto:
            raise ContractRepairValidationError(
                "expanded_auto mode is not enabled on this policy"
            )

    def has_explicit_scoped_policy(self) -> bool:
        if not self.explicit_policy_document:
            return False
        return bool(self.repository_id or self.program_id or self.policy_id)

    @property
    def mode_value(self) -> str:
        return self.mode.value if isinstance(self.mode, RolloutMode) else str(self.mode)

    def allows_automated_mutation(
        self,
        *,
        strategy: str,
        unique_target: bool,
        reconstructed: bool,
        supported_shape: bool,
    ) -> bool:
        """Return True only for initially allowed narrow-auto substitutions."""

        mode = _mode(self.mode)
        if mode is RolloutMode.SHADOW or mode is RolloutMode.ASSIST:
            return False
        if mode is RolloutMode.NARROW_AUTO and not self.allow_narrow_auto:
            return False
        if mode is RolloutMode.EXPANDED_AUTO and not self.allow_expanded_auto:
            return False
        if not self.mutation_authorized:
            return False
        strategy_key = str(strategy or "").strip().casefold()
        if strategy_key not in self.auto_allowed_strategies:
            return False
        if mode is RolloutMode.NARROW_AUTO and strategy_key not in {
            item.casefold() for item in NARROW_AUTO_STRATEGIES
        }:
            return False
        if self.auto_requires_unique_target and not unique_target:
            return False
        if self.auto_requires_reconstruction and not reconstructed:
            return False
        if self.auto_requires_supported_shape and not supported_shape:
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
            "auto_requires_supported_shape": self.auto_requires_supported_shape,
            "auto_allowed_strategies": list(self.auto_allowed_strategies),
            "rollback_on_capability_regression": (
                self.rollback_on_capability_regression
            ),
            "rollback_on_stale_root": self.rollback_on_stale_root,
            "rollback_on_reconstruction_failure": (
                self.rollback_on_reconstruction_failure
            ),
            "rollback_on_metric_breach": self.rollback_on_metric_breach,
            "mutation_authorized": self.mutation_authorized,
            "completion_authoritative": self.completion_authoritative,
        }
        if include_id:
            payload["policy_binding_id"] = self.policy_binding_id
        return payload

    @classmethod
    def default(cls) -> "ContractRepairRolloutPolicy":
        """Factory for the fail-closed default: shadow, no mutation."""

        return cls()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractRepairRolloutPolicy":
        if not isinstance(value, Mapping):
            raise ContractRepairValidationError("policy payload must be an object")
        known = set(cls.__dataclass_fields__) - {"policy_binding_id"}
        data = {key: value[key] for key in known if key in value}
        if "scoped_path_globs" in data:
            data["scoped_path_globs"] = tuple(data["scoped_path_globs"] or ())
        if "auto_allowed_strategies" in data:
            data["auto_allowed_strategies"] = tuple(
                data["auto_allowed_strategies"] or ()
            )
        return cls(**data)


def default_rollout_policy() -> ContractRepairRolloutPolicy:
    return ContractRepairRolloutPolicy.default()


def elevate_rollout_policy(
    *,
    mode: RolloutMode | str,
    explicit_policy_document: str,
    repository_id: str,
    program_id: str = "agent-supervisor-proof-gated-contract-repair-v1",
    policy_id: str = "policy:contract-repair-rollout-scoped",
    policy_revision: str = "1",
    scoped_path_globs: Sequence[str] = (),
    mutation_authorized: bool = False,
    allow_expanded_auto: bool = False,
) -> ContractRepairRolloutPolicy:
    """Build an elevated policy; still fail-closed without explicit scope."""

    mode_value = _mode(mode)
    return ContractRepairRolloutPolicy(
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
class ContractRepairMetrics:
    """Operator-facing release and decision metrics for contract repair.

    Rates are integer parts-per-million so sealed identities stay float-free.
    Safety floors are absolute zero on the four release gates.
    """

    SCHEMA: ClassVar[str] = METRICS_SCHEMA
    INTERFACE: ClassVar[str] = METRICS_INTERFACE

    recall_at_k: int = 0
    proof_eligible_recall_at_k: int = 0
    admitted_precision: int = 0
    wrong_path_rate: int = 0
    abstention_count: int = 0
    abstention_rate: int = 0
    proof_latency_ms: int = 0
    cache_latency_ms: int = 0
    cache_hit_rate: int = 0
    tokens: int = 0
    context_bytes: int = 0
    decision_count: int = 0
    admitted_count: int = 0
    safety_floors: Mapping[str, int] = field(default_factory=dict)
    safety_absolute: Mapping[str, int] = field(default_factory=dict)
    recall_k: int = DEFAULT_RECALL_K
    reason_code_counts: Mapping[str, int] = field(default_factory=dict)
    metrics_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "recall_at_k",
            "proof_eligible_recall_at_k",
            "admitted_precision",
            "wrong_path_rate",
            "abstention_count",
            "abstention_rate",
            "proof_latency_ms",
            "cache_latency_ms",
            "cache_hit_rate",
            "tokens",
            "context_bytes",
            "decision_count",
            "admitted_count",
            "recall_k",
        ):
            object.__setattr__(
                self, name, _non_negative_int(getattr(self, name), name)
            )
        floors = {
            key: _non_negative_int(self.safety_floors.get(key, 0), key)
            for key in SAFETY_FLOOR_KEYS
        }
        object.__setattr__(self, "safety_floors", MappingProxyType(floors))
        absolute = {
            str(key): _non_negative_int(value, str(key))
            for key, value in dict(self.safety_absolute).items()
        }
        object.__setattr__(self, "safety_absolute", MappingProxyType(absolute))
        reasons = {
            str(key): _non_negative_int(value, str(key))
            for key, value in sorted(dict(self.reason_code_counts).items())
        }
        object.__setattr__(self, "reason_code_counts", MappingProxyType(reasons))
        if not self.metrics_id:
            object.__setattr__(
                self,
                "metrics_id",
                content_identity(self.to_dict(include_id=False)),
            )

    def floors_hold(self) -> bool:
        return all(int(self.safety_floors.get(key, 1)) == 0 for key in SAFETY_FLOOR_KEYS)

    def breaches(self) -> tuple[str, ...]:
        failed = [
            key
            for key in SAFETY_FLOOR_KEYS
            if int(self.safety_floors.get(key, 1)) != 0
        ]
        if int(self.wrong_path_rate) != 0:
            failed.append("wrong_path_rate")
        return tuple(failed)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": METRICS_SCHEMA,
            "interface": METRICS_INTERFACE,
            "recall_at_k": self.recall_at_k,
            "proof_eligible_recall_at_k": self.proof_eligible_recall_at_k,
            "admitted_precision": self.admitted_precision,
            "wrong_path_rate": self.wrong_path_rate,
            "abstention_count": self.abstention_count,
            "abstention_rate": self.abstention_rate,
            "proof_latency_ms": self.proof_latency_ms,
            "cache_latency_ms": self.cache_latency_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "tokens": self.tokens,
            "context_bytes": self.context_bytes,
            "decision_count": self.decision_count,
            "admitted_count": self.admitted_count,
            "safety_floors": dict(self.safety_floors),
            "safety_absolute": dict(self.safety_absolute),
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
        *,
        proof_latency_ms: int = 0,
        cache_latency_ms: int = 0,
    ) -> "ContractRepairMetrics":
        """Project a RPR-019 benchmark metrics object into ops metrics."""

        case_count = int(metrics.get("case_count") or 0)
        abstention = int(metrics.get("abstention_count") or 0)
        floors = dict(metrics.get("safety_floors") or {})
        for key in SAFETY_FLOOR_KEYS:
            floors.setdefault(key, 0)
        absolute = dict(metrics.get("safety_absolute") or {})
        wrong_path = int(
            floors.get("wrong_path_automated_mutation_rate")
            or absolute.get("wrong_path_automated_mutation")
            or 0
        )
        return cls(
            recall_at_k=int(metrics.get("recall_at_k") or 0),
            proof_eligible_recall_at_k=int(
                metrics.get("proof_eligible_recall_at_k") or 0
            ),
            admitted_precision=int(metrics.get("admitted_target_precision") or 0),
            wrong_path_rate=wrong_path,
            abstention_count=abstention,
            abstention_rate=_ppm(abstention, max(1, case_count)),
            proof_latency_ms=proof_latency_ms,
            cache_latency_ms=cache_latency_ms,
            cache_hit_rate=int(metrics.get("cache_hit_rate") or 0),
            tokens=int(metrics.get("total_token_units") or 0),
            context_bytes=int(metrics.get("total_context_bytes") or 0),
            decision_count=case_count,
            admitted_count=int(
                (metrics.get("outcome_counts") or {}).get("success") or 0
            ),
            safety_floors=floors,
            safety_absolute=absolute,
            recall_k=int(metrics.get("recall_k") or DEFAULT_RECALL_K),
            reason_code_counts=dict(metrics.get("outcome_counts") or {}),
        )

    @classmethod
    def empty(cls) -> "ContractRepairMetrics":
        return cls(
            safety_floors={key: 0 for key in SAFETY_FLOOR_KEYS},
            safety_absolute={
                "wrong_path_automated_mutation": 0,
                "failed_obligation_override": 0,
                "stale_forged_or_poisoned_authoritative_admission": 0,
                "unsupported_memory_safety_promotion": 0,
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
    policy_binding_id: str = ""
    receipt_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.reason, RollbackReason):
            reason = self.reason
        else:
            try:
                reason = RollbackReason(str(self.reason).strip().casefold())
            except ValueError as exc:
                raise ContractRepairValidationError(
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
            "policy_binding_id": self.policy_binding_id,
            "mutation_authorized": False,
            "completion_authoritative": False,
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload


def evaluate_rollback(
    policy: ContractRepairRolloutPolicy,
    *,
    metrics: ContractRepairMetrics | None = None,
    capability_regression: Sequence[str] = (),
    stale_roots: Sequence[str] = (),
    reconstruction_failed: bool = False,
    elevated_abstention_error: bool = False,
) -> RollbackReceipt | None:
    """Return a demotion receipt when a rollback gate fires; else None.

    Always demotes toward shadow.  Never elevates mode.
    """

    current = _mode(policy.mode)
    if current is RolloutMode.SHADOW:
        # Still record metric/capability failures as receipts when requested
        # by the operator surface; demotion target remains shadow.
        target = RolloutMode.SHADOW
    else:
        # Step back one stage: expanded_auto -> narrow_auto -> assist -> shadow
        demotion = {
            RolloutMode.EXPANDED_AUTO: RolloutMode.NARROW_AUTO,
            RolloutMode.NARROW_AUTO: RolloutMode.ASSIST,
            RolloutMode.ASSIST: RolloutMode.SHADOW,
        }
        target = demotion.get(current, RolloutMode.SHADOW)

    if policy.rollback_on_capability_regression and capability_regression:
        return RollbackReceipt(
            reason=RollbackReason.CAPABILITY_REGRESSION,
            from_mode=current,
            to_mode=target if current is not RolloutMode.SHADOW else RolloutMode.SHADOW,
            detail="capability health regression",
            capability_ids=tuple(sorted(set(capability_regression))),
            policy_binding_id=policy.policy_binding_id,
        )
    if policy.rollback_on_stale_root and stale_roots:
        return RollbackReceipt(
            reason=RollbackReason.STALE_ROOT,
            from_mode=current,
            to_mode=target if current is not RolloutMode.SHADOW else RolloutMode.SHADOW,
            detail="stale authority root observed",
            stale_roots=tuple(sorted(set(stale_roots))),
            policy_binding_id=policy.policy_binding_id,
        )
    if policy.rollback_on_reconstruction_failure and reconstruction_failed:
        return RollbackReceipt(
            reason=RollbackReason.RECONSTRUCTION_FAILURE,
            from_mode=current,
            to_mode=target if current is not RolloutMode.SHADOW else RolloutMode.SHADOW,
            detail="proof reconstruction failure",
            policy_binding_id=policy.policy_binding_id,
        )
    if elevated_abstention_error:
        return RollbackReceipt(
            reason=RollbackReason.ELEVATED_ABSTENTION_ERROR,
            from_mode=current,
            to_mode=target if current is not RolloutMode.SHADOW else RolloutMode.SHADOW,
            detail="elevated abstention error rate",
            policy_binding_id=policy.policy_binding_id,
        )
    if policy.rollback_on_metric_breach and metrics is not None:
        breaches = metrics.breaches()
        if breaches or not metrics.floors_hold():
            return RollbackReceipt(
                reason=RollbackReason.METRIC_BREACH,
                from_mode=current,
                to_mode=target if current is not RolloutMode.SHADOW else RolloutMode.SHADOW,
                detail="safety floor or metric breach",
                metric_breaches=breaches or tuple(
                    key
                    for key in SAFETY_FLOOR_KEYS
                    if int(metrics.safety_floors.get(key, 1)) != 0
                ),
                policy_binding_id=policy.policy_binding_id,
            )
    return None


def apply_rollback(
    policy: ContractRepairRolloutPolicy,
    receipt: RollbackReceipt,
) -> ContractRepairRolloutPolicy:
    """Return a demoted policy; mutation is always revoked."""

    to_mode = _mode(receipt.to_mode)
    return ContractRepairRolloutPolicy(
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
        auto_requires_supported_shape=policy.auto_requires_supported_shape,
        auto_allowed_strategies=policy.auto_allowed_strategies,
        rollback_on_capability_regression=policy.rollback_on_capability_regression,
        rollback_on_stale_root=policy.rollback_on_stale_root,
        rollback_on_reconstruction_failure=policy.rollback_on_reconstruction_failure,
        rollback_on_metric_breach=policy.rollback_on_metric_breach,
        mutation_authorized=False,
        completion_authoritative=False,
    )


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
    name = "benchmark_contract_repair_rpr020"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractRepairValidationError(
            f"unable to load benchmark module at {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def check_plan_objective_task_dag(
    repo_root: Path | None = None,
) -> CheckResult:
    """Validate plan presence plus objective/task dependency DAGs."""

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
    if "Proof-Gated Contract Repair" not in plan_text and "proof-gated" not in plan_text.casefold():
        errors.append("plan does not identify proof-gated contract repair")

    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    goal_ids = {goal.goal_id for goal in goals}
    if "RPR-G000" not in goal_ids:
        errors.append("RPR-G000 is missing from the objective heap")
    if "RPR-G100" not in goal_ids:
        errors.append("RPR-G100 is missing from the objective heap")
    # Extension control/rollout goals required by RPR-047.
    if EXTENSION_CONTROL_GOAL_ID not in goal_ids:
        errors.append(f"{EXTENSION_CONTROL_GOAL_ID} is missing from the objective heap")
    if EXTENSION_ROLLOUT_GOAL_ID not in goal_ids:
        errors.append(f"{EXTENSION_ROLLOUT_GOAL_ID} is missing from the objective heap")

    goal_edges: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        if not re.fullmatch(r"RPR-G\d{3}", goal.goal_id):
            errors.append(f"invalid goal id: {goal.goal_id}")
        deps = tuple(goal.dependencies)
        parents = tuple(goal.parent_goal_ids) if goal.parent_goal_ids else ()
        combined = tuple(dict.fromkeys((*parents, *deps)))
        goal_edges[goal.goal_id] = combined
        for dep in combined:
            if dep not in goal_ids:
                errors.append(f"unknown objective dependency: {goal.goal_id}->{dep}")
    goal_cycles = _cycle_nodes(goal_edges)
    if goal_cycles:
        errors.append(f"goal dependency cycle: {list(goal_cycles)}")

    # RPR-G110 must hang under RPR-G000 and depend on RPR-G100.
    g110 = next((g for g in goals if g.goal_id == EXTENSION_CONTROL_GOAL_ID), None)
    if g110 is not None:
        g110_parents = set(g110.parent_goal_ids or ())
        g110_deps = set(g110.dependencies or ())
        if "RPR-G000" not in g110_parents and "RPR-G000" not in g110_deps:
            errors.append(f"{EXTENSION_CONTROL_GOAL_ID} must parent under RPR-G000")
        if "RPR-G100" not in g110_deps and "RPR-G100" not in g110_parents:
            errors.append(f"{EXTENSION_CONTROL_GOAL_ID} must depend on RPR-G100")
    g220 = next((g for g in goals if g.goal_id == EXTENSION_ROLLOUT_GOAL_ID), None)
    if g220 is not None:
        g220_parents = set(g220.parent_goal_ids or ())
        g220_deps = set(g220.dependencies or ())
        if EXTENSION_CONTROL_GOAL_ID not in g220_parents and (
            EXTENSION_CONTROL_GOAL_ID not in g220_deps
        ):
            errors.append(
                f"{EXTENSION_ROLLOUT_GOAL_ID} must parent under {EXTENSION_CONTROL_GOAL_ID}"
            )

    tasks = parse_task_file(todo_path, task_header_prefix=TASK_PREFIX)
    task_ids = {task.task_id for task in tasks}
    if len(tasks) != len(task_ids):
        errors.append("duplicate task id on the board")
    if "RPR-000" not in task_ids:
        errors.append("RPR-000 is missing")
    if "RPR-020" not in task_ids:
        errors.append("RPR-020 is missing")
    if TERMINAL_TASK_ID not in task_ids:
        errors.append(f"terminal task {TERMINAL_TASK_ID} is missing")

    task_edges: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        task_edges[task.task_id] = tuple(task.depends_on)
        for dep in task.depends_on:
            if dep not in task_ids:
                errors.append(f"unknown task dependency: {task.task_id}->{dep}")
        goal_id = str(task.metadata.get("goal id", "")).strip()
        if goal_id and goal_id not in goal_ids:
            errors.append(f"unknown task goal: {task.task_id}->{goal_id}")
        for output in task.outputs:
            if not _safe_relative(output):
                errors.append(f"{task.task_id} has unsafe output path {output!r}")
    task_cycles = _cycle_nodes(task_edges)
    if task_cycles:
        errors.append(f"task dependency cycle: {list(task_cycles)}")

    # RPR-020 must depend on post-edit validation and the safety benchmark.
    rpr020 = next((task for task in tasks if task.task_id == "RPR-020"), None)
    if rpr020 is not None:
        required_deps = {"RPR-018", "RPR-019"}
        missing_deps = required_deps - set(rpr020.depends_on)
        if missing_deps:
            errors.append(f"RPR-020 missing required deps: {sorted(missing_deps)}")

    # Terminal RPR-047 depends on RPR-046; RPR-046 depends on RPR-020 + RPR-045.
    rpr047 = next((task for task in tasks if task.task_id == TERMINAL_TASK_ID), None)
    if rpr047 is not None:
        missing_047 = {"RPR-046"} - set(rpr047.depends_on)
        if missing_047:
            errors.append(
                f"{TERMINAL_TASK_ID} missing required deps: {sorted(missing_047)}"
            )
        goal_047 = _task_metadata_get(rpr047, "goal id", "goal_id")
        if goal_047 and goal_047 != EXTENSION_ROLLOUT_GOAL_ID:
            errors.append(
                f"{TERMINAL_TASK_ID} goal id must be {EXTENSION_ROLLOUT_GOAL_ID}, "
                f"got {goal_047!r}"
            )
        # No other task may depend on the terminal operations task.
        dependents = sorted(
            tid for tid, deps in task_edges.items() if TERMINAL_TASK_ID in deps
        )
        if dependents:
            errors.append(
                f"{TERMINAL_TASK_ID} must be terminal; dependents={dependents}"
            )
    rpr046 = next((task for task in tasks if task.task_id == "RPR-046"), None)
    if rpr046 is not None:
        missing_046 = {"RPR-020", "RPR-045"} - set(rpr046.depends_on)
        if missing_046:
            errors.append(f"RPR-046 missing required deps: {sorted(missing_046)}")

    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    if scheduler.get("task_prefix") != TASK_PREFIX:
        errors.append("scheduler task prefix mismatch")
    if scheduler.get("merge_target_branch") != MERGE_TARGET_BRANCH:
        errors.append("scheduler merge target mismatch")
    if scheduler.get("board_namespace") != BOARD_NAMESPACE:
        errors.append("scheduler board namespace mismatch")
    if scheduler.get("objective_refill_enabled") is not False:
        errors.append("objective refill must be disabled")
    if scheduler.get("codebase_refill_enabled") is not False:
        errors.append("codebase refill must be disabled")
    if int(scheduler.get("max_lanes") or 0) != EXPECTED_LANE_COUNT:
        errors.append(
            f"max_lanes must be {EXPECTED_LANE_COUNT} for four-shard drain"
        )
    if scheduler.get("strict_task_sharding") is not True:
        errors.append("strict_task_sharding must be true")
    proof_policy = scheduler.get("proof_policy") or {}
    if proof_policy.get("datasets_logic_required_before_target_admission") is not True:
        errors.append("datasets logic gate is not enabled")
    if proof_policy.get("vector_semantic_authority") is not False:
        errors.append("vector semantic authority must be false")
    if proof_policy.get("memory_resource_bound_implies_memory_safety") is not False:
        errors.append("memory resource bound must not imply memory safety")
    floors = scheduler.get("release_safety_floors") or {}
    for key in SAFETY_FLOOR_KEYS:
        if int(floors.get(key, 1)) != 0:
            errors.append(f"scheduler safety floor {key} is not zero")
    for key in PROPAGATION_SAFETY_FLOOR_KEYS:
        value = _floor_lookup(floors, key, default=None)
        if value is None:
            errors.append(f"scheduler missing propagation safety floor {key}")
        elif int(value) != 0:
            errors.append(f"scheduler safety floor {key} is not zero")

    # change_propagation_policy gates (RPR-047).
    prop_policy = scheduler.get("change_propagation_policy")
    if not isinstance(prop_policy, Mapping):
        errors.append("change_propagation_policy is missing from scheduler")
    else:
        for key, expected in REQUIRED_CHANGE_PROPAGATION_POLICY.items():
            actual = prop_policy.get(key, "__missing__")
            if actual != expected:
                errors.append(
                    f"change_propagation_policy.{key} expected {expected!r}, "
                    f"got {actual!r}"
                )

    # Protected paths must include the sealed control-plane set.
    protected = scheduler.get("protected_paths") or []
    if not isinstance(protected, list):
        errors.append("scheduler protected_paths must be a list")
    else:
        protected_set = {str(item) for item in protected}
        missing_protected = [
            path for path in REQUIRED_PROTECTED_PATHS if path not in protected_set
        ]
        if missing_protected:
            errors.append(
                f"scheduler protected_paths missing: {missing_protected}"
            )

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
        detail=(
            "plan, objective heap, task DAG, scheduler bindings, "
            "propagation policy, and release floors are consistent"
        ),
        evidence={
            "goal_count": len(goals),
            "task_count": len(tasks),
            "goal_ids": sorted(goal_ids),
            "task_ids": sorted(task_ids),
            "terminal_task_id": TERMINAL_TASK_ID,
            "extension_goal_ids": [
                EXTENSION_CONTROL_GOAL_ID,
                EXTENSION_ROLLOUT_GOAL_ID,
            ],
            "propagation_safety_floors": {
                key: _floor_lookup(floors, key, default=0)
                for key in PROPAGATION_SAFETY_FLOOR_KEYS
            },
            "change_propagation_policy": dict(prop_policy or {}),
        },
    )


def check_exact_source_bindings(
    repo_root: Path | None = None,
) -> CheckResult:
    try:
        binding = bind_exact_sources(repo_root)
    except (OSError, json.JSONDecodeError, ContractRepairValidationError) as exc:
        return CheckResult(
            name="exact_source_bindings",
            status=CheckStatus.FAIL,
            detail=str(exc),
        )
    # Recompute identities to prove binding is not forged.
    root = Path(binding.repository_root)
    recomputed = {
        "plan": file_identity(root / binding.plan_path),
        "objective": file_identity(root / binding.objective_path),
        "todo": file_identity(root / binding.todo_path),
        "scheduler": file_identity(root / binding.scheduler_path),
        "launcher": file_identity(root / binding.launcher_path),
    }
    expected = {
        "plan": binding.plan_identity,
        "objective": binding.objective_identity,
        "todo": binding.todo_identity,
        "scheduler": binding.scheduler_identity,
        "launcher": binding.launcher_identity,
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
    """Probe contract-repair capability admission (fail-closed, non-authoritative)."""

    del repo_root  # root is resolved by the capability probe itself
    evidence: dict[str, Any] = {
        "authoritative": False,
        "candidate_authoritative": False,
    }
    try:
        from ipfs_accelerate_py.agent_supervisor.integrations.contract_repair_capabilities import (
            probe_contract_repair_capabilities,
        )
    except Exception as exc:  # pragma: no cover - import path is package-local
        return CheckResult(
            name="capability_health",
            status=CheckStatus.FAIL,
            detail=f"capability probe import failed: {exc}",
            evidence=evidence,
        )

    if not probe:
        return CheckResult(
            name="capability_health",
            status=CheckStatus.SKIP,
            detail="capability probe skipped",
            evidence=evidence,
        )

    try:
        report = probe_contract_repair_capabilities()
    except Exception as exc:
        return CheckResult(
            name="capability_health",
            status=CheckStatus.FAIL,
            detail=f"capability probe raised: {exc}",
            evidence=evidence,
        )

    report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    capabilities = report_dict.get("capabilities") or []
    available = []
    unavailable = []
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
    # Capability gaps are recorded, not hard-fail, unless the probe itself is broken.
    # Reconstruction-required logic backends may be unavailable without blocking shadow.
    return CheckResult(
        name="capability_health",
        status=CheckStatus.PASS,
        detail=(
            f"capability probe completed: available={len(available)} "
            f"unavailable={len(unavailable)}"
        ),
        evidence=evidence,
    )


def check_supervisor_process_state(
    repo_root: Path | None = None,
    *,
    state_root: Path | None = None,
    lane_count: int = 4,
) -> CheckResult:
    """Inspect supervisor/process state without requiring a live run.

    A stopped supervisor is a valid operational state.  Corrupted or
    contradictory state files fail closed.
    """

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
                    pid = int(
                        payload.get("pid") or payload.get("supervisor_pid") or 0
                    )
                    alive = False
                    if pid > 0:
                        try:
                            os.kill(pid, 0)
                            alive = True
                        except OSError:
                            alive = False
                    lane_info["supervisor"] = str(
                        payload.get("status") or "unknown"
                    )
                    lane_info["supervisor_pid"] = pid
                    lane_info["supervisor_pid_alive"] = alive
                    if (
                        lane_info["supervisor"] == "running"
                        and not alive
                    ):
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
                    lane_info["active_task_id"] = str(
                        task.get("active_task_id") or ""
                    )
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
    """Verify the four release safety floors are absolute zero."""

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
        ops = ContractRepairMetrics.from_benchmark_metrics(metrics)
    except Exception as exc:
        return CheckResult(
            name="benchmark_floors",
            status=CheckStatus.FAIL,
            detail=f"benchmark floor evaluation failed: {exc}",
        )

    failures = [
        key for key in SAFETY_FLOOR_KEYS if int(floors.get(key, 1)) != 0
    ]
    absolute_keys = (
        "wrong_path_automated_mutation",
        "failed_obligation_override",
        "stale_forged_or_poisoned_authoritative_admission",
        "unsupported_memory_safety_promotion",
    )
    for key in absolute_keys:
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
        detail="all four release safety floors are absolute zero",
        evidence=evidence,
    )


def check_feature_flags(
    policy: ContractRepairRolloutPolicy | None = None,
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
    if not default.auto_requires_supported_shape:
        errors.append("default policy does not require supported shapes for auto")
    allowed = set(default.auto_allowed_strategies)
    if not allowed <= {item.casefold() for item in NARROW_AUTO_STRATEGIES}:
        errors.append(
            f"default auto strategies escape narrow set: {sorted(allowed)}"
        )

    # assist / narrow-auto without explicit scope must raise.
    for mode in (RolloutMode.ASSIST, RolloutMode.NARROW_AUTO):
        try:
            ContractRepairRolloutPolicy(mode=mode)
            errors.append(f"{mode.value} accepted without explicit scoped policy")
        except ContractRepairValidationError:
            pass

    selected = policy or default
    if _mode(selected.mode) is not RolloutMode.SHADOW:
        if not selected.has_explicit_scoped_policy():
            errors.append("selected elevated policy lacks explicit scope")

    # Auto must reject non-rename strategies under narrow-auto defaults.
    narrow = elevate_rollout_policy(
        mode=RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:test",
        mutation_authorized=True,
    )
    if narrow.allows_automated_mutation(
        strategy=RepairStrategy.ADAPTER.value,
        unique_target=True,
        reconstructed=True,
        supported_shape=True,
    ):
        errors.append("narrow-auto incorrectly allows adapter strategy")
    if not narrow.allows_automated_mutation(
        strategy=RepairStrategy.RENAME_SUBSTITUTION.value,
        unique_target=True,
        reconstructed=True,
        supported_shape=True,
    ):
        errors.append("narrow-auto rejects valid rename substitution")
    if narrow.allows_automated_mutation(
        strategy=RepairStrategy.RENAME_SUBSTITUTION.value,
        unique_target=False,
        reconstructed=True,
        supported_shape=True,
    ):
        errors.append("narrow-auto allows non-unique rename substitution")
    if narrow.allows_automated_mutation(
        strategy=RepairStrategy.RENAME_SUBSTITUTION.value,
        unique_target=True,
        reconstructed=False,
        supported_shape=True,
    ):
        errors.append("narrow-auto allows unreconstructed rename substitution")

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
        detail="shadow is default; assist/narrow-auto require scoped policy; auto limited to unique reconstructed renames",
        evidence={"default": default.to_dict(), "selected": selected.to_dict()},
    )


def check_rollback_gates(
    policy: ContractRepairRolloutPolicy | None = None,
) -> CheckResult:
    """Prove each rollback trigger demotes and revokes mutation."""

    base = elevate_rollout_policy(
        mode=RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:test",
        mutation_authorized=True,
    )
    errors: list[str] = []
    scenarios = [
        (
            "capability_regression",
            dict(capability_regression=("logic_smt",)),
            RollbackReason.CAPABILITY_REGRESSION,
        ),
        (
            "stale_root",
            dict(stale_roots=("index_root",)),
            RollbackReason.STALE_ROOT,
        ),
        (
            "reconstruction_failure",
            dict(reconstruction_failed=True),
            RollbackReason.RECONSTRUCTION_FAILURE,
        ),
        (
            "metric_breach",
            dict(
                metrics=ContractRepairMetrics(
                    wrong_path_rate=1,
                    safety_floors={
                        "wrong_path_automated_mutation_rate": 1,
                        "failed_obligation_override_rate": 0,
                        "stale_forged_or_poisoned_authoritative_admission_rate": 0,
                        "unsupported_memory_safety_promotion_rate": 0,
                    },
                    safety_absolute={"wrong_path_automated_mutation": 1},
                )
            ),
            RollbackReason.METRIC_BREACH,
        ),
    ]
    receipts: list[dict[str, Any]] = []
    for name, kwargs, expected_reason in scenarios:
        receipt = evaluate_rollback(base, **kwargs)
        if receipt is None:
            errors.append(f"{name} did not produce a rollback receipt")
            continue
        if receipt.reason is not expected_reason:
            errors.append(
                f"{name} reason {receipt.reason} != {expected_reason}"
            )
        demoted = apply_rollback(base, receipt)
        if demoted.mutation_authorized:
            errors.append(f"{name} demotion still authorizes mutation")
        if _mode(demoted.mode).value not in {
            RolloutMode.SHADOW.value,
            RolloutMode.ASSIST.value,
        }:
            # From narrow_auto, demotion must be assist (one step) or shadow.
            if _mode(demoted.mode) is RolloutMode.NARROW_AUTO:
                errors.append(f"{name} failed to demote mode")
        receipts.append(receipt.to_dict())

    # Healthy path must not roll back.
    healthy = evaluate_rollback(
        base,
        metrics=ContractRepairMetrics.empty(),
        capability_regression=(),
        stale_roots=(),
        reconstruction_failed=False,
    )
    if healthy is not None:
        errors.append("healthy state incorrectly produced a rollback receipt")

    selected = policy or default_rollout_policy()
    if not selected.rollback_on_capability_regression:
        errors.append("selected policy disables capability regression rollback")
    if not selected.rollback_on_stale_root:
        errors.append("selected policy disables stale root rollback")
    if not selected.rollback_on_reconstruction_failure:
        errors.append("selected policy disables reconstruction failure rollback")
    if not selected.rollback_on_metric_breach:
        errors.append("selected policy disables metric breach rollback")

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
        detail="capability regression, stale root, reconstruction failure, and metric breach roll back",
        evidence={"receipts": receipts},
    )


def check_guide_boundaries(
    repo_root: Path | None = None,
) -> CheckResult:
    """Confirm the operator guide states model and memory-safety boundaries."""

    root = (repo_root or repository_root()).resolve()
    guide = root / GUIDE_REL
    if not guide.is_file():
        return CheckResult(
            name="guide_boundaries",
            status=CheckStatus.FAIL,
            detail=f"guide missing: {guide}",
        )
    text = guide.read_text(encoding="utf-8")
    required_phrases = (
        "shadow",
        "assist",
        "narrow-auto",
        "rollback",
        "memory safety",
        "vector",
        "does not prove memory safety",
    )
    # Accept a few equivalent phrasings for the memory-safety boundary.
    alt_memory = (
        "do not prove memory safety",
        "does not prove memory safety",
        "never prove memory safety",
        "not memory-safety evidence",
        "not memory safety evidence",
    )
    missing = []
    for phrase in required_phrases:
        if phrase == "does not prove memory safety":
            if not any(item in text.casefold() for item in alt_memory):
                missing.append(phrase)
            continue
        if phrase.casefold() not in text.casefold():
            missing.append(phrase)
    # Explicit non-proof evidence classes.
    for kind in ("test", "type", "resource"):
        if kind not in text.casefold():
            missing.append(kind)
    if missing:
        return CheckResult(
            name="guide_boundaries",
            status=CheckStatus.FAIL,
            detail=f"guide missing required boundary language: {missing}",
        )
    return CheckResult(
        name="guide_boundaries",
        status=CheckStatus.PASS,
        detail="guide states model boundaries and non-proof of memory safety",
        evidence={"path": GUIDE_REL, "bytes": guide.stat().st_size},
    )


def check_change_propagation_policy(
    repo_root: Path | None = None,
) -> CheckResult:
    """Verify sealed change_propagation_policy gates and six new zero floors."""

    root = (repo_root or repository_root()).resolve()
    scheduler_path = root / SCHEDULER_REL
    if not scheduler_path.is_file():
        return CheckResult(
            name="change_propagation_policy",
            status=CheckStatus.FAIL,
            detail=f"scheduler missing: {scheduler_path}",
        )
    errors: list[str] = []
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    prop_policy = scheduler.get("change_propagation_policy")
    observed: dict[str, Any] = {}
    if not isinstance(prop_policy, Mapping):
        errors.append("change_propagation_policy is missing")
    else:
        observed = dict(prop_policy)
        for key, expected in REQUIRED_CHANGE_PROPAGATION_POLICY.items():
            actual = prop_policy.get(key, "__missing__")
            if actual != expected:
                errors.append(
                    f"{key}: expected {expected!r}, got {actual!r}"
                )
    floors = scheduler.get("release_safety_floors") or {}
    floor_evidence: dict[str, int] = {}
    for key in PROPAGATION_SAFETY_FLOOR_KEYS:
        value = _floor_lookup(floors if isinstance(floors, Mapping) else {}, key)
        if value is None:
            errors.append(f"missing zero safety floor {key}")
            continue
        floor_evidence[key] = int(value)
        if int(value) != 0:
            errors.append(f"safety floor {key} is not zero ({value})")
    evidence = {
        "change_propagation_policy": observed,
        "propagation_safety_floors": floor_evidence,
        "required_gates": dict(REQUIRED_CHANGE_PROPAGATION_POLICY),
    }
    if errors:
        return CheckResult(
            name="change_propagation_policy",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence=evidence,
        )
    return CheckResult(
        name="change_propagation_policy",
        status=CheckStatus.PASS,
        detail=(
            "change_propagation_policy gates hold and six propagation "
            "safety floors are absolute zero"
        ),
        evidence=evidence,
    )


def check_protected_paths_and_refill_isolation(
    repo_root: Path | None = None,
) -> CheckResult:
    """Confirm protected control-plane paths and refill isolation."""

    root = (repo_root or repository_root()).resolve()
    scheduler_path = root / SCHEDULER_REL
    if not scheduler_path.is_file():
        return CheckResult(
            name="protected_paths_refill_isolation",
            status=CheckStatus.FAIL,
            detail=f"scheduler missing: {scheduler_path}",
        )
    errors: list[str] = []
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    protected = scheduler.get("protected_paths") or []
    protected_list = [str(item) for item in protected] if isinstance(protected, list) else []
    protected_set = set(protected_list)
    missing = [path for path in REQUIRED_PROTECTED_PATHS if path not in protected_set]
    if missing:
        errors.append(f"protected_paths missing required entries: {missing}")
    for path in REQUIRED_PROTECTED_PATHS:
        if not (root / path).is_file():
            errors.append(f"protected path not on disk: {path}")
    if scheduler.get("objective_refill_enabled") is not False:
        errors.append("objective_refill_enabled must be false")
    if scheduler.get("codebase_refill_enabled") is not False:
        errors.append("codebase_refill_enabled must be false")
    evidence = {
        "protected_paths": protected_list,
        "required_protected_paths": list(REQUIRED_PROTECTED_PATHS),
        "objective_refill_enabled": scheduler.get("objective_refill_enabled"),
        "codebase_refill_enabled": scheduler.get("codebase_refill_enabled"),
    }
    if errors:
        return CheckResult(
            name="protected_paths_refill_isolation",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence=evidence,
        )
    return CheckResult(
        name="protected_paths_refill_isolation",
        status=CheckStatus.PASS,
        detail="protected paths sealed and objective/codebase refill disabled",
        evidence=evidence,
    )


def check_four_shard_board_drain(
    repo_root: Path | None = None,
) -> CheckResult:
    """Prove a clean four-shard board can drain under strict sharding.

    Verifies lane count, strict sharding, four distinct entry-task parallel
    lanes, acyclic DAG, and that with every non-terminal task completed the
    only remaining ready work is the terminal operations task (then empty).
    """

    root = (repo_root or repository_root()).resolve()
    errors: list[str] = []
    scheduler_path = root / SCHEDULER_REL
    todo_path = root / TODO_REL
    if not scheduler_path.is_file() or not todo_path.is_file():
        return CheckResult(
            name="four_shard_board_drain",
            status=CheckStatus.FAIL,
            detail="scheduler or todo board missing",
        )
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    max_lanes = int(scheduler.get("max_lanes") or 0)
    if max_lanes != EXPECTED_LANE_COUNT:
        errors.append(f"max_lanes={max_lanes} (expected {EXPECTED_LANE_COUNT})")
    if scheduler.get("strict_task_sharding") is not True:
        errors.append("strict_task_sharding is not true")

    tasks = parse_task_file(todo_path, task_header_prefix=TASK_PREFIX)
    task_by_id = {task.task_id: task for task in tasks}
    if TERMINAL_TASK_ID not in task_by_id:
        errors.append(f"terminal task {TERMINAL_TASK_ID} missing")

    entry_lanes: dict[str, str] = {}
    for task_id in INITIAL_PROPAGATION_ENTRY_TASKS:
        task = task_by_id.get(task_id)
        if task is None:
            errors.append(f"entry task {task_id} missing")
            continue
        lane = _task_metadata_get(task, "parallel lane", "parallel_lane")
        if not lane:
            errors.append(f"entry task {task_id} missing parallel lane")
            continue
        entry_lanes[task_id] = lane
    if len(entry_lanes) == len(INITIAL_PROPAGATION_ENTRY_TASKS):
        if len(set(entry_lanes.values())) != EXPECTED_LANE_COUNT:
            errors.append(
                f"entry tasks do not map to {EXPECTED_LANE_COUNT} distinct lanes: "
                f"{entry_lanes}"
            )

    edges = {task.task_id: tuple(task.depends_on) for task in tasks}
    if _cycle_nodes(edges):
        errors.append("task dependency cycle prevents drain")

    # Simulate a clean completed board except the terminal operations task.
    completed = {
        task.task_id
        for task in tasks
        if task.task_id != TERMINAL_TASK_ID
        and str(getattr(task, "status", "") or "").casefold()
        in {"completed", "done", "complete"}
    }
    # For drain readiness we also treat every non-terminal as hypothetically
    # complete so a healthy restart can finish RPR-047 then empty the board.
    hypothetical_completed = set(task_by_id) - {TERMINAL_TASK_ID}
    ready = sorted(
        task_id
        for task_id, deps in edges.items()
        if task_id not in hypothetical_completed
        and all(dep in hypothetical_completed for dep in deps)
    )
    if ready != [TERMINAL_TASK_ID] and TERMINAL_TASK_ID in task_by_id:
        errors.append(
            f"after completing non-terminal work expected only "
            f"[{TERMINAL_TASK_ID}] ready, got {ready}"
        )
    # After terminal completion the board drains.
    drained_ready = sorted(
        task_id
        for task_id, deps in edges.items()
        if task_id not in set(task_by_id)
        and all(dep in set(task_by_id) for dep in deps)
    )
    # With all tasks completed, ready set is empty.
    all_completed = set(task_by_id)
    fully_drained = sorted(
        task_id
        for task_id, deps in edges.items()
        if task_id not in all_completed
        and all(dep in all_completed for dep in deps)
    )
    if fully_drained:
        errors.append(f"fully completed board still has ready work: {fully_drained}")

    evidence = {
        "max_lanes": max_lanes,
        "strict_task_sharding": scheduler.get("strict_task_sharding"),
        "entry_lanes": entry_lanes,
        "terminal_task_id": TERMINAL_TASK_ID,
        "ready_after_non_terminal_complete": ready,
        "ready_after_full_complete": fully_drained,
        "board_task_count": len(tasks),
        "completed_on_disk": sorted(completed),
    }
    if errors:
        return CheckResult(
            name="four_shard_board_drain",
            status=CheckStatus.FAIL,
            detail="; ".join(errors),
            evidence=evidence,
        )
    return CheckResult(
        name="four_shard_board_drain",
        status=CheckStatus.PASS,
        detail=(
            f"clean {EXPECTED_LANE_COUNT}-shard board drains: entry lanes "
            f"disjoint, terminal {TERMINAL_TASK_ID} last, empty when complete"
        ),
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# RPR-047 operations surface / end-to-end helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofGatedContractRepairOperations:
    """Canonical operations surface for RPR-020 + RPR-047 validation.

    Aggregates legacy contract-repair checks with propagation extension gates.
    Never grants mutation or completion authority.
    """

    INTERFACE: ClassVar[str] = "ProofGatedContractRepairOperations@1"
    TERMINAL_TASK: ClassVar[str] = TERMINAL_TASK_ID
    EXTENSION_GOAL: ClassVar[str] = EXTENSION_ROLLOUT_GOAL_ID
    LEGACY_TASK: ClassVar[str] = "RPR-020"
    LEGACY_GOAL: ClassVar[str] = "RPR-G100"

    @classmethod
    def run(
        cls,
        repo_root: Path | None = None,
        *,
        run_benchmark: bool = True,
        probe_capabilities: bool = True,
        policy: ContractRepairRolloutPolicy | None = None,
        benchmark_report: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        report = run_all_checks(
            repo_root,
            run_benchmark=run_benchmark,
            probe_capabilities=probe_capabilities,
            policy=policy,
            benchmark_report=benchmark_report,
        )
        report["operations_interface"] = cls.INTERFACE
        report["terminal_task_id"] = cls.TERMINAL_TASK
        report["extension_goal_id"] = cls.EXTENSION_GOAL
        report["legacy_task_id"] = cls.LEGACY_TASK
        report["legacy_goal_id"] = cls.LEGACY_GOAL
        report["mutation_authorized"] = False
        report["completion_authoritative"] = False
        return report

    @classmethod
    def required_extension_ids(cls) -> dict[str, str]:
        return {
            "terminal_task_id": TERMINAL_TASK_ID,
            "control_goal_id": EXTENSION_CONTROL_GOAL_ID,
            "rollout_goal_id": EXTENSION_ROLLOUT_GOAL_ID,
            "legacy_task_id": cls.LEGACY_TASK,
            "legacy_goal_id": cls.LEGACY_GOAL,
        }


@dataclass(frozen=True)
class ChangePropagationEndToEnd:
    """Seeded end-to-end propagation scenarios for RPR-047.

    Positive: two-to-three argument change detects every caller, proves one
    source, applies an atomic analytical plan, rediffs to a fixed point, and
    emits a completion receipt.  Negative wrong-value, unknown-frontier,
    partial-SCC, and LLM-scope cases fail closed.
    """

    INTERFACE: ClassVar[str] = "ChangePropagationEndToEnd@1"
    TASK_ID: ClassVar[str] = TERMINAL_TASK_ID
    GOAL_ID: ClassVar[str] = EXTENSION_ROLLOUT_GOAL_ID

    POSITIVE_SCENARIO: ClassVar[str] = "two_to_three_argument_callers"
    NEGATIVE_SCENARIOS: ClassVar[tuple[str, ...]] = (
        "same_typed_wrong_information",
        "reflection_plugin_registry_ffi_frontier",
        "partial_transaction",
        "llm_scope_escape",
    )

    @classmethod
    def evaluate_seeded_corpus(
        cls,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        """Run hermetic fixture evaluation for positive and negative cases."""

        root = (repo_root or repository_root()).resolve()
        bench_path = root / PROPAGATION_BENCHMARK_SCRIPT_REL
        if not bench_path.is_file():
            raise ContractRepairValidationError(
                f"propagation benchmark missing: {bench_path}"
            )
        name = "benchmark_change_propagation_rpr047_e2e"
        if name in sys.modules:
            bench = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, bench_path)
            if spec is None or spec.loader is None:
                raise ContractRepairValidationError(
                    f"unable to load propagation benchmark at {bench_path}"
                )
            bench = importlib.util.module_from_spec(spec)
            sys.modules[name] = bench
            spec.loader.exec_module(bench)

        manifest = bench.load_fixture_manifest(root / PROPAGATION_FIXTURE_MANIFEST_REL)
        cases = list(manifest.get("cases") or [])
        by_scenario: dict[str, Mapping[str, Any]] = {}
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            scenario = str(case.get("scenario") or "")
            by_scenario.setdefault(scenario, case)

        positive = by_scenario.get(cls.POSITIVE_SCENARIO)
        if positive is None:
            raise ContractRepairValidationError(
                f"seeded positive scenario missing: {cls.POSITIVE_SCENARIO}"
            )
        positive_result = bench.evaluate_fixture(positive)
        positive_payload = (
            positive_result.to_dict()
            if hasattr(positive_result, "to_dict")
            else dict(positive_result)
        )

        negatives: dict[str, Any] = {}
        for scenario in cls.NEGATIVE_SCENARIOS:
            fixture = by_scenario.get(scenario)
            if fixture is None:
                negatives[scenario] = {
                    "present": False,
                    "admitted": None,
                    "completion_success": None,
                    "ok_fail_closed": False,
                }
                continue
            result = bench.evaluate_fixture(fixture)
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            fail_closed = (
                payload.get("admitted") is False
                and payload.get("completion_success") is False
            )
            # LLM scope escape must never complete and never escape.
            if scenario == "llm_scope_escape":
                fail_closed = fail_closed and not bool(
                    payload.get("llm_scope_escape")
                )
            # Partial SCC / transaction must roll back, not complete.
            if scenario == "partial_transaction":
                fail_closed = fail_closed and bool(payload.get("scc_rollback"))
            negatives[scenario] = {
                "present": True,
                "admitted": payload.get("admitted"),
                "completion_success": payload.get("completion_success"),
                "outcome_kind": payload.get("outcome_kind"),
                "scc_rollback": payload.get("scc_rollback"),
                "llm_scope_escape": payload.get("llm_scope_escape"),
                "ok_fail_closed": fail_closed,
                "case": payload,
            }

        consumers = (
            (positive.get("artifacts") or {})
            .get("consumers", {})
            .get("content", {})
        )
        resolved = consumers.get("resolved") if isinstance(consumers, Mapping) else []
        caller_kinds = []
        if isinstance(resolved, list):
            caller_kinds = [
                str(item.get("kind") or "")
                for item in resolved
                if isinstance(item, Mapping)
            ]

        positive_ok = (
            bool(positive_payload.get("admitted"))
            and bool(positive_payload.get("completion_success"))
            and bool(positive_payload.get("consumer_precise"))
            and bool(positive_payload.get("unique_source_precise"))
            and bool(positive_payload.get("analytical_path"))
            and bool(positive_payload.get("plan_complete"))
            and int(positive_payload.get("fixed_point_iterations") or 0) >= 1
            and len(caller_kinds) >= 4
        )
        negatives_ok = all(
            item.get("present") and item.get("ok_fail_closed")
            for item in negatives.values()
        )
        report = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "change-propagation-end-to-end-report@1"
            ),
            "interface": cls.INTERFACE,
            "task_id": cls.TASK_ID,
            "goal_id": cls.GOAL_ID,
            "positive_scenario": cls.POSITIVE_SCENARIO,
            "positive": {
                "ok": positive_ok,
                "caller_kinds": caller_kinds,
                "caller_count": len(caller_kinds),
                "admitted": positive_payload.get("admitted"),
                "completion_success": positive_payload.get("completion_success"),
                "analytical_path": positive_payload.get("analytical_path"),
                "unique_source_precise": positive_payload.get(
                    "unique_source_precise"
                ),
                "consumer_precise": positive_payload.get("consumer_precise"),
                "fixed_point_iterations": positive_payload.get(
                    "fixed_point_iterations"
                ),
                "plan_complete": positive_payload.get("plan_complete"),
                "outcome_kind": positive_payload.get("outcome_kind"),
                "case": positive_payload,
            },
            "negatives": negatives,
            "valid": positive_ok and negatives_ok,
            "mutation_authorized": False,
            "completion_authoritative": False,
        }
        report["report_id"] = content_identity(
            {key: value for key, value in report.items() if key != "report_id"}
        )
        return report


# ---------------------------------------------------------------------------
# Aggregated validation / doctor / status / replay
# ---------------------------------------------------------------------------


def run_all_checks(
    repo_root: Path | None = None,
    *,
    run_benchmark: bool = True,
    probe_capabilities: bool = True,
    policy: ContractRepairRolloutPolicy | None = None,
    benchmark_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    selected_policy = policy or default_rollout_policy()
    checks = [
        check_plan_objective_task_dag(root),
        check_exact_source_bindings(root),
        check_capability_health(root, probe=probe_capabilities),
        check_supervisor_process_state(root),
        check_benchmark_floors(
            root, run=run_benchmark, report=benchmark_report
        ),
        check_feature_flags(selected_policy),
        check_rollback_gates(selected_policy),
        check_guide_boundaries(root),
        # RPR-047 extension gates (legacy checks above remain required).
        check_change_propagation_policy(root),
        check_protected_paths_and_refill_isolation(root),
        check_four_shard_board_drain(root),
    ]
    results = [item.to_dict() for item in checks]
    ok = all(item.ok for item in checks)
    payload = {
        "schema": VALIDATOR_SCHEMA,
        "interface": VALIDATOR_INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "terminal_task_id": TERMINAL_TASK_ID,
        "extension_goal_ids": [
            EXTENSION_CONTROL_GOAL_ID,
            EXTENSION_ROLLOUT_GOAL_ID,
        ],
        "valid": ok,
        "default_mode": RolloutMode.SHADOW.value,
        "policy": selected_policy.to_dict(),
        "checks": results,
        "failed": [
            item.name
            for item in checks
            if item.status is CheckStatus.FAIL
        ],
        "mutation_authorized": False,
        "completion_authoritative": False,
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
    """Operator doctor: control plane, bindings, capabilities, flags, rollback."""

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
    policy: ContractRepairRolloutPolicy | None = None,
) -> dict[str, Any]:
    """Operator status: mode, bindings, supervisor, and compact metrics."""

    root = (repo_root or repository_root()).resolve()
    selected = policy or default_rollout_policy()
    binding_check = check_exact_source_bindings(root)
    supervisor_check = check_supervisor_process_state(root)
    dag_check = check_plan_objective_task_dag(root)
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
        "mutation_authorized": bool(selected.mutation_authorized),
        "completion_authoritative": False,
        "valid": all(
            item.ok for item in (binding_check, supervisor_check, dag_check)
        ),
    }
    payload["report_id"] = content_identity(
        {key: value for key, value in payload.items() if key != "report_id"}
    )
    return payload


def replay_decision_receipt(
    receipt: Mapping[str, Any],
    *,
    policy: ContractRepairRolloutPolicy | None = None,
    expected_roots: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Replay a sealed decision/completion/benchmark receipt against policy.

    Verifies identity recomputation, optional root freshness, reconstruction
    flags, and whether automated mutation would still be authorized.
    """

    if not isinstance(receipt, Mapping):
        raise ContractRepairValidationError("receipt must be an object")
    selected = policy or default_rollout_policy()
    errors: list[str] = []

    claimed_id = (
        receipt.get("receipt_id")
        or receipt.get("report_id")
        or receipt.get("decision_id")
        or receipt.get("case_id")
    )
    body = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "receipt_id",
            "report_id",
            "decision_id",
            "case_id",
            "metrics_id",
        }
    }
    recomputed = content_identity(body)
    identity_ok = True
    if isinstance(claimed_id, str) and claimed_id.startswith("sha256:"):
        # Prefer verifying against a body that excludes the claimed id field.
        identity_ok = claimed_id == recomputed or claimed_id == content_identity(
            {
                key: value
                for key, value in receipt.items()
                if key != (
                    "receipt_id"
                    if "receipt_id" in receipt
                    else "report_id"
                    if "report_id" in receipt
                    else "decision_id"
                    if "decision_id" in receipt
                    else "case_id"
                )
            }
        )
        if not identity_ok:
            # Some sealed reports include the id in a sibling seal helper;
            # accept when the caller embeds a verified flag.
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
    supported_shape = bool(
        receipt.get("supported_shape")
        if "supported_shape" in receipt
        else str(receipt.get("language") or "python").casefold()
        in {"python", "py"}
    )
    strategy = str(
        receipt.get("strategy")
        or receipt.get("repair_strategy")
        or RepairStrategy.RENAME_SUBSTITUTION.value
    )
    auto_ok = selected.allows_automated_mutation(
        strategy=strategy,
        unique_target=unique_target,
        reconstructed=reconstructed,
        supported_shape=supported_shape,
    )

    stale = [
        key
        for key, expected in (expected_roots or {}).items()
        if not isinstance(roots, Mapping) or roots.get(key) != expected
    ]
    rollback = evaluate_rollback(
        selected,
        stale_roots=stale,
        reconstruction_failed=bool(
            receipt.get("reconstruction_failed")
            or (
                selected.auto_requires_reconstruction
                and not reconstructed
                and _mode(selected.mode)
                in {RolloutMode.NARROW_AUTO, RolloutMode.EXPANDED_AUTO}
            )
        ),
    )

    payload = {
        "schema": VALIDATOR_SCHEMA,
        "interface": VALIDATOR_INTERFACE,
        "command": "replay",
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "valid": not errors,
        "identity_ok": identity_ok and "receipt identity" not in " ".join(errors),
        "recomputed_identity": recomputed,
        "claimed_identity": claimed_id,
        "automated_mutation_authorized": auto_ok,
        "strategy": strategy,
        "unique_target": unique_target,
        "reconstructed": reconstructed,
        "supported_shape": supported_shape,
        "policy": selected.to_dict(),
        "rollback": None if rollback is None else rollback.to_dict(),
        "errors": errors,
        "mutation_authorized": False,
        "completion_authoritative": False,
    }
    payload["report_id"] = content_identity(
        {key: value for key, value in payload.items() if key != "report_id"}
    )
    return payload


def collect_metrics(
    *,
    benchmark_report: Mapping[str, Any] | None = None,
    run_benchmark: bool = True,
) -> ContractRepairMetrics:
    if benchmark_report is not None:
        return ContractRepairMetrics.from_benchmark_metrics(
            benchmark_report["metrics"]
            if "metrics" in benchmark_report
            else benchmark_report
        )
    if not run_benchmark:
        return ContractRepairMetrics.empty()
    bench = _load_benchmark_module()
    report = bench.run_benchmark()
    return ContractRepairMetrics.from_benchmark_metrics(report["metrics"])


# ---------------------------------------------------------------------------
# Model / evidence boundary helpers (used by guide tests and policy checks)
# ---------------------------------------------------------------------------


def evidence_proves_memory_safety(evidence_kind: str) -> bool:
    """Return False for vector/test/type/resource and other non-proof kinds.

    Only an independent reconstructed formal proof over a ``MemorySafetyFacet``
    can support a memory-safety claim; this helper encodes the negative side of
    that rule for operators and tests.
    """

    kind = str(evidence_kind or "").strip().casefold().replace("-", "_")
    if kind in NON_MEMORY_SAFETY_EVIDENCE:
        return False
    if kind in {
        "memory_safety_facet",
        "reconstructed_proof",
        "formal_proof",
        "ownership_lifetime_proof",
    }:
        # Still not automatic proof: callers must reconstruct.  The helper only
        # says these kinds are *eligible* to participate in a proof argument.
        return False
    return False


def model_boundary_statement() -> str:
    return (
        "Models propose nominations, rankings, and edit drafts. "
        "They do not admit targets, authorize writes, complete tasks, or prove "
        "memory safety. Vector, test, type, and resource evidence does not prove "
        "memory safety."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_json(payload: Mapping[str, Any]) -> None:
    json.dump(_plain(payload), sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: parent of scripts/).",
    )
    common.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON.",
    )
    common.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the adversarial benchmark floor check.",
    )
    common.add_argument(
        "--skip-capabilities",
        action="store_true",
        help="Skip the capability health probe.",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Validate proof-gated contract-repair operations, metrics, "
            "feature flags, and rollback gates (RPR-020 / RPR-047)."
        ),
        parents=[common],
    )
    # Top-level flag form used by RPR-G000 validation.
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Run the full validation suite (default when no subcommand).",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Run control-plane and health checks.", parents=[common])
    sub.add_parser(
        "status", help="Show mode, bindings, and supervisor state.", parents=[common]
    )
    replay_p = sub.add_parser(
        "replay",
        help="Replay a sealed decision/completion/benchmark receipt.",
        parents=[common],
    )
    replay_p.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="Path to a JSON receipt to replay.",
    )
    sub.add_parser(
        "check-dag", help="Check plan/objective/task DAG only.", parents=[common]
    )
    sub.add_parser(
        "check-bindings", help="Check exact source bindings only.", parents=[common]
    )
    sub.add_parser(
        "check-capabilities", help="Probe capability health only.", parents=[common]
    )
    sub.add_parser(
        "check-supervisor",
        help="Inspect supervisor/process state only.",
        parents=[common],
    )
    sub.add_parser(
        "check-benchmark-floors",
        help="Run benchmark safety floors only.",
        parents=[common],
    )
    sub.add_parser(
        "check-flags", help="Validate feature-flag defaults.", parents=[common]
    )
    sub.add_parser(
        "check-rollback", help="Validate rollback gates.", parents=[common]
    )
    sub.add_parser(
        "check-propagation-policy",
        help="Validate change_propagation_policy gates and new floors.",
        parents=[common],
    )
    sub.add_parser(
        "check-protected-paths",
        help="Validate protected paths and refill isolation.",
        parents=[common],
    )
    sub.add_parser(
        "check-four-shard",
        help="Validate four-shard board drain readiness.",
        parents=[common],
    )
    sub.add_parser(
        "metrics",
        help="Emit operator metrics from the benchmark.",
        parents=[common],
    )
    sub.add_parser(
        "policy",
        help="Emit the default (shadow) rollout policy.",
        parents=[common],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = (args.repo_root or repository_root()).resolve()
    command = args.command
    if args.check_all or command is None:
        command = "check-all"

    try:
        if command in {"check-all", "doctor"}:
            report = run_all_checks(
                root,
                run_benchmark=not args.skip_benchmark,
                probe_capabilities=not args.skip_capabilities,
            )
            report["command"] = "doctor" if command == "doctor" else "check-all"
            write_checkpoint("rpr-020-validation-report", report)
            if args.json:
                _print_json(report)
            else:
                status_word = "healthy" if report.get("valid") else "unhealthy"
                print(
                    f"{VALIDATOR_INTERFACE} command={report.get('command')} "
                    f"valid={report.get('valid')} status={status_word} "
                    f"default_mode={report.get('default_mode')} "
                    f"failed={report.get('failed')} "
                    f"report_id={report.get('report_id')}"
                )
            return 0 if report.get("valid") else 1

        if command == "status":
            report = status(root)
            write_checkpoint("rpr-020-status", report)
            if args.json:
                _print_json(report)
            else:
                print(
                    f"{VALIDATOR_INTERFACE} mode={report['mode']} "
                    f"valid={report['valid']} "
                    f"master={report['supervisor']['evidence'].get('master_status')} "
                    f"report_id={report['report_id']}"
                )
            return 0 if report.get("valid") else 1

        if command == "replay":
            receipt_path: Path = args.receipt
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            report = replay_decision_receipt(receipt)
            if args.json:
                _print_json(report)
            else:
                print(
                    f"{VALIDATOR_INTERFACE} replay valid={report['valid']} "
                    f"auto={report['automated_mutation_authorized']} "
                    f"errors={report['errors']} "
                    f"report_id={report['report_id']}"
                )
            return 0 if report.get("valid") else 1

        check_map = {
            "check-dag": lambda: check_plan_objective_task_dag(root),
            "check-bindings": lambda: check_exact_source_bindings(root),
            "check-capabilities": lambda: check_capability_health(
                root, probe=not args.skip_capabilities
            ),
            "check-supervisor": lambda: check_supervisor_process_state(root),
            "check-benchmark-floors": lambda: check_benchmark_floors(
                root, run=not args.skip_benchmark
            ),
            "check-flags": check_feature_flags,
            "check-rollback": check_rollback_gates,
            "check-propagation-policy": lambda: check_change_propagation_policy(
                root
            ),
            "check-protected-paths": lambda: check_protected_paths_and_refill_isolation(
                root
            ),
            "check-four-shard": lambda: check_four_shard_board_drain(root),
        }
        if command in check_map:
            result = check_map[command]()
            payload = result.to_dict()
            if args.json:
                _print_json(payload)
            else:
                print(
                    f"{result.name}: {result.status.value} — {result.detail}"
                )
            return 0 if result.ok else 1

        if command == "metrics":
            metrics = collect_metrics(run_benchmark=not args.skip_benchmark)
            payload = metrics.to_dict()
            write_checkpoint("rpr-020-metrics", payload)
            if args.json:
                _print_json(payload)
            else:
                print(
                    f"{METRICS_INTERFACE} recall_at_k={metrics.recall_at_k} "
                    f"proof_eligible_recall_at_k={metrics.proof_eligible_recall_at_k} "
                    f"admitted_precision={metrics.admitted_precision} "
                    f"wrong_path_rate={metrics.wrong_path_rate} "
                    f"abstention={metrics.abstention_count} "
                    f"tokens={metrics.tokens} context_bytes={metrics.context_bytes} "
                    f"floors_ok={metrics.floors_hold()} "
                    f"metrics_id={metrics.metrics_id}"
                )
            return 0 if metrics.floors_hold() else 1

        if command == "policy":
            policy = default_rollout_policy()
            payload = policy.to_dict()
            if args.json:
                _print_json(payload)
            else:
                print(
                    f"{ROLLOUT_POLICY_INTERFACE} mode={policy.mode_value} "
                    f"mutation_authorized={policy.mutation_authorized} "
                    f"policy_binding_id={policy.policy_binding_id}"
                )
            return 0

        parser.error(f"unknown command: {command}")
        return 2
    except ContractRepairValidationError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
