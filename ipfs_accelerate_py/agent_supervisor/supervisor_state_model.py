"""Finite TLA+ models and bounded model-check receipts for supervisor workflows.

This module deliberately separates three trust boundaries:

* :class:`SupervisorTransitionSchema` is the finite, typed source of truth.
* :class:`SupervisorStateModelGenerator` deterministically translates that
  schema; it does not contain a fixed workflow in its rendering logic.
* :class:`SupervisorStateModelChecker` invokes an optional TLC or Apalache
  executable and records the exact bounded experiment.

A successful receipt is *bounded model-check evidence*.  It is never exposed
as an unbounded proof.  Apalache checks the combined safety invariant over a
finite trace length; TLC additionally checks the generated temporal liveness
properties.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .formal_planning_contracts import FormalWorkPlan
from .formal_verification_contracts import canonical_json, content_identity
from .prover_matrix_registry import CommandRequest, CommandResult


SUPERVISOR_STATE_MODEL_VERSION: Final = 1
SUPERVISOR_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/transition-schema@1"
)
SUPERVISOR_TLA_MODEL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/tla-state-model@1"
)
MODEL_CHECK_BOUNDS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-check-bounds@1"
)
MODEL_CHECK_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-check-receipt@1"
)
TLA_TRANSLATOR_ID: Final = "supervisor-transition-schema-to-tla"
TLA_TRANSLATOR_VERSION: Final = 1

DEFAULT_MODEL_CHECK_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_VERSION_TIMEOUT_SECONDS: Final = 3.0
DEFAULT_MAX_MODEL_CHECK_OUTPUT_BYTES: Final = 2 * 1024 * 1024

SAFETY_PROPERTIES: Final = (
    "UniqueAcceptance",
    "FencingSafety",
    "DependencyOrder",
    "IdempotentMerge",
    "CapacitySafety",
    "EvidenceGates",
)
LIVENESS_PROPERTIES: Final = ("BoundedProgress", "TerminalOutcomes")

_NO_AGENT = "__NO_AGENT__"
_RESERVED_IDENTIFIERS = {_NO_AGENT}
_TLC_SUCCESS_MARKERS = (
    "model checking completed. no error has been found",
    "model checking completed",
    "no error has been found",
)
_APALACHE_SUCCESS_MARKERS = (
    "checker reports no error",
    "no error up to computation length",
    "verification result: pass",
    "result: pass",
)
_COUNTEREXAMPLE_MARKERS = (
    "counterexample",
    "is violated",
    "temporal properties were violated",
    "checker reports an error",
    "checker has found an error",
    "found an invariant violation",
    "error trace",
)


class ModelValidationError(ValueError):
    """The finite transition schema or its requested bounds are invalid."""


class ModelCheckerTool(str, Enum):
    TLC = "tlc"
    APALACHE = "apalache"


class ModelCheckStatus(str, Enum):
    PASSED = "passed"
    COUNTEREXAMPLE = "counterexample"
    UNKNOWN = "unknown"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


def _strict_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelValidationError(f"{field_name} must be an object")
    try:
        decoded = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            f"{field_name} must contain strict JSON values"
        ) from exc
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise ModelValidationError(f"{field_name} must be an object")
    return decoded


def _strings(
    values: Iterable[Any] | Any,
    field_name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        items: Iterable[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Iterable) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise ModelValidationError(f"{field_name} must be a sequence of strings")
    result = tuple(
        sorted({str(item).strip() for item in items if str(item).strip()})
    )
    if required and not result:
        raise ModelValidationError(f"{field_name} must not be empty")
    if any(item in _RESERVED_IDENTIFIERS for item in result):
        raise ModelValidationError(f"{field_name} contains a reserved identifier")
    return result


def _positive_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelValidationError(f"{field_name} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ModelValidationError(f"{field_name} must be {qualifier}")
    return value


def _boolean(value: Any, field_name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ModelValidationError(f"{field_name} must be boolean")
    return value


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_timestamp(clock: Callable[[], float]) -> str:
    return (
        datetime.fromtimestamp(clock(), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _enum_value(value: Any, enum_type: type[Enum], field_name: str) -> Any:
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except ValueError as exc:
        raise ModelValidationError(f"unsupported {field_name}: {raw!r}") from exc


@dataclass(frozen=True)
class TransitionRule:
    """One schema-driven supervisor transition.

    The booleans are a small reviewed effect/guard vocabulary, rather than raw
    TLA+ snippets.  Consequently untrusted schema values cannot inject model
    text and equivalent schemas always render identically.
    """

    name: str
    source_states: tuple[str, ...]
    target_state: str
    requires_dependencies: bool = False
    requires_evidence: bool = False
    requires_owner: bool = False
    requires_current_fence: bool = True
    accepts_claim: bool = False
    replaces_claim: bool = False
    increments_fence: bool = False
    clears_claim: bool = False
    capacity_delta: int = 0
    records_merge: bool = False
    produces_evidence: bool = False
    marks_progress: bool = False
    increments_retry: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ModelValidationError(
                "transition name must be a TLA-compatible identifier"
            )
        sources = _strings(self.source_states, "source_states", required=True)
        target = str(self.target_state).strip()
        if not target or target in _RESERVED_IDENTIFIERS:
            raise ModelValidationError("target_state must not be empty or reserved")
        if self.accepts_claim and self.replaces_claim:
            raise ModelValidationError(
                "a transition cannot both accept and replace a claim"
            )
        if self.clears_claim and (self.accepts_claim or self.replaces_claim):
            raise ModelValidationError(
                "a transition cannot set and clear a claim simultaneously"
            )
        if (
            isinstance(self.capacity_delta, bool)
            or self.capacity_delta not in (-1, 0, 1)
        ):
            raise ModelValidationError("capacity_delta must be -1, 0, or 1")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_states", sources)
        object.__setattr__(self, "target_state", target)
        object.__setattr__(
            self, "metadata", _strict_mapping(self.metadata, "transition metadata")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransitionRule":
        if not isinstance(value, Mapping):
            raise ModelValidationError("transition must be an object")
        return cls(
            name=value.get("name", ""),
            source_states=tuple(value.get("source_states") or ()),
            target_state=value.get("target_state", ""),
            requires_dependencies=_boolean(
                value.get("requires_dependencies"),
                "requires_dependencies",
                default=False,
            ),
            requires_evidence=_boolean(
                value.get("requires_evidence"), "requires_evidence", default=False
            ),
            requires_owner=_boolean(
                value.get("requires_owner"), "requires_owner", default=False
            ),
            requires_current_fence=_boolean(
                value.get("requires_current_fence"),
                "requires_current_fence",
                default=True,
            ),
            accepts_claim=_boolean(
                value.get("accepts_claim"), "accepts_claim", default=False
            ),
            replaces_claim=_boolean(
                value.get("replaces_claim"), "replaces_claim", default=False
            ),
            increments_fence=_boolean(
                value.get("increments_fence"), "increments_fence", default=False
            ),
            clears_claim=_boolean(
                value.get("clears_claim"), "clears_claim", default=False
            ),
            capacity_delta=value.get("capacity_delta", 0),
            records_merge=_boolean(
                value.get("records_merge"), "records_merge", default=False
            ),
            produces_evidence=_boolean(
                value.get("produces_evidence"), "produces_evidence", default=False
            ),
            marks_progress=_boolean(
                value.get("marks_progress"), "marks_progress", default=False
            ),
            increments_retry=_boolean(
                value.get("increments_retry"), "increments_retry", default=False
            ),
            metadata=value.get("metadata") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_states": list(self.source_states),
            "target_state": self.target_state,
            "requires_dependencies": self.requires_dependencies,
            "requires_evidence": self.requires_evidence,
            "requires_owner": self.requires_owner,
            "requires_current_fence": self.requires_current_fence,
            "accepts_claim": self.accepts_claim,
            "replaces_claim": self.replaces_claim,
            "increments_fence": self.increments_fence,
            "clears_claim": self.clears_claim,
            "capacity_delta": self.capacity_delta,
            "records_merge": self.records_merge,
            "produces_evidence": self.produces_evidence,
            "marks_progress": self.marks_progress,
            "increments_retry": self.increments_retry,
            "metadata": dict(self.metadata),
        }


DEFAULT_SUPERVISOR_TRANSITIONS: Final = (
    TransitionRule(
        "AcceptClaim",
        ("pending",),
        "accepted",
        accepts_claim=True,
        increments_fence=True,
    ),
    TransitionRule(
        "Start",
        ("accepted", "retrying"),
        "running",
        requires_dependencies=True,
        requires_owner=True,
        capacity_delta=1,
        marks_progress=True,
    ),
    TransitionRule(
        "ProduceEvidence",
        ("running",),
        "evidence_ready",
        requires_owner=True,
        produces_evidence=True,
    ),
    TransitionRule(
        "Merge",
        ("evidence_ready",),
        "completed",
        requires_dependencies=True,
        requires_evidence=True,
        requires_owner=True,
        capacity_delta=-1,
        records_merge=True,
        clears_claim=True,
    ),
    TransitionRule(
        "Retry",
        ("running",),
        "retrying",
        requires_owner=True,
        capacity_delta=-1,
        increments_retry=True,
    ),
    TransitionRule(
        "Fail",
        ("running", "evidence_ready"),
        "failed",
        requires_owner=True,
        capacity_delta=-1,
        clears_claim=True,
    ),
    TransitionRule(
        "Cancel",
        ("pending", "accepted", "running", "retrying", "evidence_ready"),
        "cancelled",
        capacity_delta=-1,
        clears_claim=True,
    ),
)


@dataclass(frozen=True)
class SupervisorTransitionSchema:
    """Canonical finite input consumed by the TLA+ generator."""

    tasks: tuple[str, ...]
    agents: tuple[str, ...]
    states: tuple[str, ...]
    initial_state: str
    terminal_states: tuple[str, ...]
    dependency_satisfied_states: tuple[str, ...]
    transitions: tuple[TransitionRule, ...]
    dependencies: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    required_evidence: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    capacity: int = 1
    source_identity: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SUPERVISOR_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SUPERVISOR_TRANSITION_SCHEMA:
            raise ModelValidationError("unsupported supervisor transition schema")
        tasks = _strings(self.tasks, "tasks", required=True)
        agents = _strings(self.agents, "agents", required=True)
        states = _strings(self.states, "states", required=True)
        terminal = _strings(self.terminal_states, "terminal_states", required=True)
        dependency_states = _strings(
            self.dependency_satisfied_states,
            "dependency_satisfied_states",
            required=True,
        )
        initial = str(self.initial_state).strip()
        if initial not in states:
            raise ModelValidationError("initial_state must be one of states")
        if initial in terminal:
            raise ModelValidationError("initial_state cannot be terminal")
        if not set(terminal) <= set(states):
            raise ModelValidationError("terminal_states must be a subset of states")
        if not set(dependency_states) <= set(terminal):
            raise ModelValidationError(
                "dependency_satisfied_states must be terminal states"
            )
        capacity = _positive_int(self.capacity, "capacity")

        rules: list[TransitionRule] = []
        for item in self.transitions:
            rules.append(
                item if isinstance(item, TransitionRule) else TransitionRule.from_dict(item)
            )
        rules.sort(key=lambda item: item.name)
        if not rules:
            raise ModelValidationError("transitions must not be empty")
        names = [item.name for item in rules]
        if len(names) != len(set(names)):
            raise ModelValidationError("transition names must be unique")
        for rule in rules:
            if not set(rule.source_states) <= set(states):
                raise ModelValidationError(
                    f"transition {rule.name} has an unknown source state"
                )
            if rule.target_state not in states:
                raise ModelValidationError(
                    f"transition {rule.name} has an unknown target state"
                )
            if set(rule.source_states) & set(terminal):
                raise ModelValidationError(
                    f"transition {rule.name} leaves a terminal state"
                )

        dependencies = self._normalize_task_map(
            self.dependencies, tasks, "dependencies", known_values=set(tasks)
        )
        self._validate_acyclic(tasks, dependencies)
        evidence = self._normalize_task_map(
            self.required_evidence,
            tasks,
            "required_evidence",
            known_values=None,
        )
        source_identity = str(self.source_identity).strip()
        metadata = _strict_mapping(self.metadata, "schema metadata")

        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "terminal_states", terminal)
        object.__setattr__(
            self, "dependency_satisfied_states", dependency_states
        )
        object.__setattr__(self, "transitions", tuple(rules))
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "required_evidence", evidence)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(self, "metadata", metadata)

    @staticmethod
    def _normalize_task_map(
        value: Mapping[str, Iterable[str]],
        tasks: tuple[str, ...],
        field_name: str,
        *,
        known_values: set[str] | None,
    ) -> dict[str, tuple[str, ...]]:
        if not isinstance(value, Mapping):
            raise ModelValidationError(f"{field_name} must be an object")
        unknown_keys = set(map(str, value)) - set(tasks)
        if unknown_keys:
            raise ModelValidationError(
                f"{field_name} contains unknown tasks: {sorted(unknown_keys)!r}"
            )
        result: dict[str, tuple[str, ...]] = {}
        for task in tasks:
            items = _strings(value.get(task, ()), f"{field_name}[{task}]")
            if known_values is not None and not set(items) <= known_values:
                raise ModelValidationError(
                    f"{field_name}[{task}] contains unknown identifiers"
                )
            if task in items:
                raise ModelValidationError(f"{task} cannot depend on itself")
            result[task] = items
        return result

    @staticmethod
    def _validate_acyclic(
        tasks: tuple[str, ...], dependencies: Mapping[str, tuple[str, ...]]
    ) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task: str) -> None:
            if task in visited:
                return
            if task in visiting:
                raise ModelValidationError("dependencies must be acyclic")
            visiting.add(task)
            for dependency in dependencies[task]:
                visit(dependency)
            visiting.remove(task)
            visited.add(task)

        for task in tasks:
            visit(task)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    evidence_id
                    for values in self.required_evidence.values()
                    for evidence_id in values
                }
            )
        )

    @property
    def schema_identity(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "version": SUPERVISOR_STATE_MODEL_VERSION,
            "tasks": list(self.tasks),
            "agents": list(self.agents),
            "states": list(self.states),
            "initial_state": self.initial_state,
            "terminal_states": list(self.terminal_states),
            "dependency_satisfied_states": list(
                self.dependency_satisfied_states
            ),
            "transitions": [item.to_dict() for item in self.transitions],
            "dependencies": {
                task: list(self.dependencies[task]) for task in self.tasks
            },
            "required_evidence": {
                task: list(self.required_evidence[task]) for task in self.tasks
            },
            "capacity": self.capacity,
            "source_identity": self.source_identity,
            "metadata": dict(self.metadata),
        }
        if include_identity:
            payload["schema_identity"] = self.schema_identity
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SupervisorTransitionSchema":
        if not isinstance(value, Mapping):
            raise ModelValidationError("transition schema must be an object")
        result = cls(
            tasks=tuple(value.get("tasks") or ()),
            agents=tuple(value.get("agents") or ()),
            states=tuple(value.get("states") or ()),
            initial_state=value.get("initial_state", ""),
            terminal_states=tuple(value.get("terminal_states") or ()),
            dependency_satisfied_states=tuple(
                value.get("dependency_satisfied_states") or ("completed",)
            ),
            transitions=tuple(value.get("transitions") or ()),
            dependencies=value.get("dependencies") or {},
            required_evidence=value.get("required_evidence") or {},
            capacity=value.get("capacity", 1),
            source_identity=value.get("source_identity", ""),
            metadata=value.get("metadata") or {},
            schema=value.get("schema", SUPERVISOR_TRANSITION_SCHEMA),
        )
        claimed = value.get("schema_identity")
        if claimed and claimed != result.schema_identity:
            raise ModelValidationError("transition schema identity does not match")
        return result

    @classmethod
    def from_formal_work_plan(
        cls,
        plan: FormalWorkPlan | Mapping[str, Any],
        *,
        transitions: Iterable[TransitionRule | Mapping[str, Any]] = (
            DEFAULT_SUPERVISOR_TRANSITIONS
        ),
        states: Iterable[str] = (
            "pending",
            "accepted",
            "running",
            "evidence_ready",
            "retrying",
            "completed",
            "failed",
            "cancelled",
        ),
        capacity: int | None = None,
    ) -> "SupervisorTransitionSchema":
        work_plan = (
            plan if isinstance(plan, FormalWorkPlan) else FormalWorkPlan.from_dict(plan)
        )
        task_ids = tuple(item.task_id for item in work_plan.tasks)
        actor_ids = tuple(item.actor_id for item in work_plan.actors)
        requirements = {
            task.task_id: task.evidence_requirement_ids for task in work_plan.tasks
        }
        terminal = tuple(
            sorted(
                {
                    state
                    for task in work_plan.tasks
                    for state in task.terminal_states
                }
            )
        )
        configured_capacity = (
            capacity
            if capacity is not None
            else work_plan.metadata.get("model_capacity", len(actor_ids))
        )
        return cls(
            tasks=task_ids,
            agents=actor_ids,
            states=tuple(states),
            initial_state="pending",
            terminal_states=terminal,
            dependency_satisfied_states=("completed",),
            transitions=tuple(transitions),
            dependencies={
                task.task_id: task.depends_on for task in work_plan.tasks
            },
            required_evidence=requirements,
            capacity=configured_capacity,
            source_identity=work_plan.content_id,
            metadata={
                "source_schema": work_plan.SCHEMA,
                "repository_tree_id": work_plan.repository_tree_id,
            },
        )


@dataclass(frozen=True)
class ModelCheckBounds:
    """Finite semantic and execution bounds bound into a generated model."""

    max_steps: int = 64
    max_retries: int = 1
    max_fence: int = 8
    max_tasks: int = 32
    max_agents: int = 16
    max_states: int = 32
    max_transitions: int = 64
    max_evidence_ids: int = 128
    schema: str = MODEL_CHECK_BOUNDS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MODEL_CHECK_BOUNDS_SCHEMA:
            raise ModelValidationError("unsupported model-check bounds schema")
        for name in (
            "max_steps",
            "max_fence",
            "max_tasks",
            "max_agents",
            "max_states",
            "max_transitions",
            "max_evidence_ids",
        ):
            _positive_int(getattr(self, name), name)
        _positive_int(self.max_retries, "max_retries", allow_zero=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelCheckBounds":
        if not isinstance(value, Mapping):
            raise ModelValidationError("bounds must be an object")
        return cls(
            max_steps=value.get("max_steps", 64),
            max_retries=value.get("max_retries", 1),
            max_fence=value.get("max_fence", 8),
            max_tasks=value.get("max_tasks", 32),
            max_agents=value.get("max_agents", 16),
            max_states=value.get("max_states", 32),
            max_transitions=value.get("max_transitions", 64),
            max_evidence_ids=value.get("max_evidence_ids", 128),
            schema=value.get("schema", MODEL_CHECK_BOUNDS_SCHEMA),
        )

    def validate_schema(self, value: SupervisorTransitionSchema) -> None:
        dimensions = (
            ("tasks", len(value.tasks), self.max_tasks),
            ("agents", len(value.agents), self.max_agents),
            ("states", len(value.states), self.max_states),
            ("transitions", len(value.transitions), self.max_transitions),
            ("evidence ids", len(value.evidence_ids), self.max_evidence_ids),
        )
        exceeded = [
            f"{name}={actual}>{maximum}"
            for name, actual, maximum in dimensions
            if actual > maximum
        ]
        if exceeded:
            raise ModelValidationError(
                "transition schema exceeds finite bounds: " + ", ".join(exceeded)
            )
        if value.capacity > self.max_tasks:
            raise ModelValidationError("capacity exceeds max_tasks")
        if sum(rule.increments_fence for rule in value.transitions) and (
            self.max_fence < 1
        ):  # pragma: no cover - max_fence already positive
            raise ModelValidationError("max_fence cannot represent accepted claims")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "max_steps": self.max_steps,
            "max_retries": self.max_retries,
            "max_fence": self.max_fence,
            "max_tasks": self.max_tasks,
            "max_agents": self.max_agents,
            "max_states": self.max_states,
            "max_transitions": self.max_transitions,
            "max_evidence_ids": self.max_evidence_ids,
        }

    @property
    def identity(self) -> str:
        return content_identity(self.to_dict())

    @property
    def label(self) -> str:
        return (
            f"tasks<={self.max_tasks},agents<={self.max_agents},"
            f"states<={self.max_states},transitions<={self.max_transitions},"
            f"steps<={self.max_steps},retries<={self.max_retries},"
            f"fence<={self.max_fence},evidence<={self.max_evidence_ids}"
        )


def _tla_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _tla_set(values: Iterable[str]) -> str:
    items = tuple(sorted(set(values)))
    return "{" + ", ".join(_tla_string(item) for item in items) + "}"


def _tla_function(
    name: str, tasks: tuple[str, ...], values: Mapping[str, tuple[str, ...]]
) -> str:
    rendered_cases = [
        f"t = {_tla_string(task)} -> {_tla_set(values[task])}"
        for task in tasks
    ]
    cases = "\n".join(
        ("        " if index == 0 else "        [] ") + rendered
        for index, rendered in enumerate(rendered_cases)
    )
    return (
        f"{name} == [t \\in Tasks |-> CASE\n"
        f"{cases}\n"
        "        [] OTHER -> {}]"
    )


def _model_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(value).strip())
    if not cleaned:
        cleaned = "SupervisorState"
    if not cleaned[0].isalpha():
        cleaned = "M_" + cleaned
    return cleaned


@dataclass(frozen=True)
class GeneratedSupervisorStateModel:
    """Exact generated model and both checker configurations."""

    module_name: str
    model_text: str
    tlc_config_text: str
    apalache_config_text: str
    transition_schema: SupervisorTransitionSchema
    bounds: ModelCheckBounds
    schema: str = SUPERVISOR_TLA_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SUPERVISOR_TLA_MODEL_SCHEMA:
            raise ModelValidationError("unsupported generated model schema")
        if not self.model_text.endswith("\n"):
            raise ModelValidationError("model_text must end with a newline")

    @property
    def model_identity(self) -> str:
        return _sha256_text(self.model_text)

    @property
    def tlc_config_identity(self) -> str:
        return _sha256_text(self.tlc_config_text)

    @property
    def apalache_config_identity(self) -> str:
        return _sha256_text(self.apalache_config_text)

    @property
    def artifact_identity(self) -> str:
        return content_identity(self.to_dict(include_text=False))

    def configuration_for(self, tool: ModelCheckerTool | str) -> str:
        selected = _enum_value(tool, ModelCheckerTool, "model checker")
        return (
            self.tlc_config_text
            if selected is ModelCheckerTool.TLC
            else self.apalache_config_text
        )

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "version": SUPERVISOR_STATE_MODEL_VERSION,
            "module_name": self.module_name,
            "translator": {
                "id": TLA_TRANSLATOR_ID,
                "version": TLA_TRANSLATOR_VERSION,
            },
            "transition_schema_identity": self.transition_schema.schema_identity,
            "bounds": self.bounds.to_dict(),
            "bounds_identity": self.bounds.identity,
            "model_identity": self.model_identity,
            "tlc_config_identity": self.tlc_config_identity,
            "apalache_config_identity": self.apalache_config_identity,
            "safety_properties": list(SAFETY_PROPERTIES),
            "liveness_properties": list(LIVENESS_PROPERTIES),
            "bounded": True,
            "unbounded_proof": False,
        }
        if include_text:
            payload.update(
                {
                    "model_text": self.model_text,
                    "tlc_config_text": self.tlc_config_text,
                    "apalache_config_text": self.apalache_config_text,
                }
            )
        return payload


class SupervisorStateModelGenerator:
    """Deterministically render a finite schema as TLA+."""

    def generate(
        self,
        schema: SupervisorTransitionSchema | Mapping[str, Any],
        *,
        bounds: ModelCheckBounds | Mapping[str, Any] | None = None,
        module_name: str = "SupervisorState",
    ) -> GeneratedSupervisorStateModel:
        transition_schema = (
            schema
            if isinstance(schema, SupervisorTransitionSchema)
            else SupervisorTransitionSchema.from_dict(schema)
        )
        finite_bounds = (
            ModelCheckBounds()
            if bounds is None
            else bounds
            if isinstance(bounds, ModelCheckBounds)
            else ModelCheckBounds.from_dict(bounds)
        )
        finite_bounds.validate_schema(transition_schema)
        name = _model_name(module_name)
        model_text = self._render_model(name, transition_schema, finite_bounds)
        tlc_config = self._render_tlc_config()
        # Apalache's ``--config`` option consumes TLC configuration syntax.
        # The finite trace length is part of both the artifact bounds and the
        # recorded command line because TLC config has no length directive.
        apalache_config = "INIT Init\nNEXT Next\nINVARIANT Safety\n"
        return GeneratedSupervisorStateModel(
            module_name=name,
            model_text=model_text,
            tlc_config_text=tlc_config,
            apalache_config_text=apalache_config,
            transition_schema=transition_schema,
            bounds=finite_bounds,
        )

    @staticmethod
    def _render_tlc_config() -> str:
        lines = ["SPECIFICATION Spec"]
        lines.extend(f"INVARIANT {name}" for name in ("TypeOK", *SAFETY_PROPERTIES))
        lines.extend(f"PROPERTY {name}" for name in LIVENESS_PROPERTIES)
        return "\n".join(lines) + "\n"

    def _render_model(
        self,
        module_name: str,
        schema: SupervisorTransitionSchema,
        bounds: ModelCheckBounds,
    ) -> str:
        chunks = [
            f"---- MODULE {module_name} ----",
            "EXTENDS Naturals, FiniteSets, TLC",
            "",
            "\\* Generated deterministically from transition schema:",
            f"\\* {schema.schema_identity}",
            f"\\* Finite explored bounds: {bounds.label}",
            "\\* @type: Set(Str);",
            f"Tasks == {_tla_set(schema.tasks)}",
            "\\* @type: Set(Str);",
            f"Agents == {_tla_set(schema.agents)}",
            "\\* @type: Set(Str);",
            f"States == {_tla_set(schema.states)}",
            "\\* @type: Set(Str);",
            f"TerminalStates == {_tla_set(schema.terminal_states)}",
            "\\* @type: Set(Str);",
            (
                "DependencySatisfiedStates == "
                f"{_tla_set(schema.dependency_satisfied_states)}"
            ),
            "\\* @type: Set(Str);",
            f"EvidenceIds == {_tla_set(schema.evidence_ids)}",
            f"InitialState == {_tla_string(schema.initial_state)}",
            f"MaxCapacity == {schema.capacity}",
            f"MaxSteps == {bounds.max_steps}",
            f"MaxRetries == {bounds.max_retries}",
            f"MaxFence == {bounds.max_fence}",
            "FenceTokens == 0..MaxFence",
            "\\* @type: Str -> Set(Str);",
            _tla_function("Dependencies", schema.tasks, schema.dependencies),
            "\\* @type: Str -> Set(Str);",
            _tla_function(
                "RequiredEvidence", schema.tasks, schema.required_evidence
            ),
            "",
            "VARIABLES",
            "    \\* @type: Str -> Str;",
            "    state,",
            "    \\* @type: Str -> Set(Str);",
            "    claims,",
            "    \\* @type: Str -> Int;",
            "    fence,",
            "    \\* @type: Str -> Int;",
            "    lastMutationFence,",
            "    \\* @type: Str -> Set(Str);",
            "    evidence,",
            "    \\* @type: Str -> Bool;",
            "    merged,",
            "    \\* @type: Str -> Int;",
            "    mergeCount,",
            "    \\* @type: Set(Str);",
            "    active,",
            "    \\* @type: Str -> Bool;",
            "    progressed,",
            "    \\* @type: Str -> Int;",
            "    retryCount,",
            "    \\* @type: Int;",
            "    step",
            (
                "vars == <<state, claims, fence, lastMutationFence, evidence, "
                "merged, mergeCount, active, progressed, retryCount, step>>"
            ),
            "",
            "Init ==",
            "    /\\ state = [t \\in Tasks |-> InitialState]",
            "    /\\ claims = [t \\in Tasks |-> {}]",
            "    /\\ fence = [t \\in Tasks |-> 0]",
            "    /\\ lastMutationFence = [t \\in Tasks |-> 0]",
            "    /\\ evidence = [t \\in Tasks |-> {}]",
            "    /\\ merged = [t \\in Tasks |-> FALSE]",
            "    /\\ mergeCount = [t \\in Tasks |-> 0]",
            "    /\\ active = {}",
            "    /\\ progressed = [t \\in Tasks |-> FALSE]",
            "    /\\ retryCount = [t \\in Tasks |-> 0]",
            "    /\\ step = 0",
            "",
        ]
        for rule in schema.transitions:
            chunks.extend(self._render_transition(rule, schema))
            chunks.append("")
        disjunction = "\n".join(
            (
                f"    \\/ \\E t \\in Tasks, a \\in Agents, f \\in FenceTokens: "
                f"{rule.name}(t, a, f)"
            )
            for rule in schema.transitions
        )
        chunks.extend(
            [
                "Next ==",
                disjunction,
                "",
                "TypeOK ==",
                "    /\\ state \\in [Tasks -> States]",
                "    /\\ claims \\in [Tasks -> SUBSET Agents]",
                "    /\\ fence \\in [Tasks -> FenceTokens]",
                "    /\\ lastMutationFence \\in [Tasks -> FenceTokens]",
                "    /\\ evidence \\in [Tasks -> SUBSET EvidenceIds]",
                "    /\\ merged \\in [Tasks -> BOOLEAN]",
                "    /\\ mergeCount \\in [Tasks -> 0..1]",
                "    /\\ active \\in SUBSET Tasks",
                "    /\\ progressed \\in [Tasks -> BOOLEAN]",
                "    /\\ retryCount \\in [Tasks -> 0..MaxRetries]",
                "    /\\ step \\in 0..MaxSteps",
                "",
                "UniqueAcceptance ==",
                "    \\A t \\in Tasks: Cardinality(claims[t]) <= 1",
                "",
                "FencingSafety ==",
                "    \\A t \\in Tasks: lastMutationFence[t] = fence[t]",
                "",
                "DependencyOrder ==",
                (
                    "    \\A t \\in Tasks: progressed[t] => "
                    "(\\A d \\in Dependencies[t]: "
                    "state[d] \\in DependencySatisfiedStates)"
                ),
                "",
                "IdempotentMerge ==",
                (
                    "    \\A t \\in Tasks: "
                    "(mergeCount[t] <= 1 /\\ (merged[t] <=> mergeCount[t] = 1))"
                ),
                "",
                "CapacitySafety ==",
                "    Cardinality(active) <= MaxCapacity",
                "",
                "EvidenceGates ==",
                (
                    "    \\A t \\in Tasks: "
                    "((merged[t] \\/ state[t] \\in DependencySatisfiedStates) "
                    "=> RequiredEvidence[t] \\subseteq evidence[t])"
                ),
                "",
                "Safety ==",
                "    /\\ TypeOK",
                *[f"    /\\ {item}" for item in SAFETY_PROPERTIES],
                "",
                "AllTerminal ==",
                "    \\A t \\in Tasks: state[t] \\in TerminalStates",
                "",
                "\\* Liveness is checked only inside MaxSteps.",
                "BoundedProgress ==",
                "    <>(AllTerminal \\/ step = MaxSteps)",
                "",
                "TerminalOutcomes ==",
                (
                    "    \\A t \\in Tasks: "
                    "<>(state[t] \\in TerminalStates \\/ step = MaxSteps)"
                ),
                "",
                "Spec == Init /\\ [][Next]_vars /\\ WF_vars(Next)",
                "",
                "====",
                "",
            ]
        )
        return "\n".join(chunks)

    @staticmethod
    def _render_transition(
        rule: TransitionRule, schema: SupervisorTransitionSchema
    ) -> list[str]:
        guards = [
            "    /\\ step < MaxSteps",
            f"    /\\ state[t] \\in {_tla_set(rule.source_states)}",
        ]
        if rule.requires_current_fence:
            guards.append("    /\\ f = fence[t]")
        if rule.requires_dependencies:
            guards.append(
                "    /\\ \\A d \\in Dependencies[t]: "
                "state[d] \\in DependencySatisfiedStates"
            )
        if rule.requires_evidence:
            guards.append(
                "    /\\ RequiredEvidence[t] \\subseteq evidence[t]"
            )
        if rule.requires_owner:
            guards.append("    /\\ claims[t] = {a}")
        if rule.accepts_claim:
            guards.append("    /\\ claims[t] = {}")
        if rule.increments_fence:
            guards.append("    /\\ fence[t] < MaxFence")
        if rule.capacity_delta > 0:
            guards.extend(
                (
                    "    /\\ t \\notin active",
                    "    /\\ Cardinality(active) < MaxCapacity",
                )
            )
        elif rule.capacity_delta < 0:
            # Cancellation of queued work remains legal and idempotently leaves
            # capacity untouched, while running work releases its slot.
            pass
        if rule.records_merge:
            guards.append("    /\\ ~merged[t]")
        if rule.increments_retry:
            guards.append("    /\\ retryCount[t] < MaxRetries")

        if rule.accepts_claim or rule.replaces_claim:
            claim_value = "{a}"
        elif rule.clears_claim:
            claim_value = "{}"
        else:
            claim_value = "@"
        fence_value = "@ + 1" if rule.increments_fence else "@"
        evidence_value = (
            "@ \\union RequiredEvidence[t]" if rule.produces_evidence else "@"
        )
        merged_value = "TRUE" if rule.records_merge else "@"
        merge_count_value = "@ + 1" if rule.records_merge else "@"
        progressed_value = "TRUE" if rule.marks_progress else "@"
        retry_value = "@ + 1" if rule.increments_retry else "@"
        if rule.capacity_delta > 0:
            active_value = "active \\union {t}"
        elif rule.capacity_delta < 0:
            active_value = "active \\ {t}"
        else:
            active_value = "active"
        # Recording the supplied fence (or its newly accepted successor) makes
        # a transition which omits the current-fence guard falsify
        # FencingSafety instead of silently masking the stale mutation.
        new_fence = "f + 1" if rule.increments_fence else "f"
        updates = [
            f"    /\\ state' = [state EXCEPT ![t] = {_tla_string(rule.target_state)}]",
            f"    /\\ claims' = [claims EXCEPT ![t] = {claim_value}]",
            f"    /\\ fence' = [fence EXCEPT ![t] = {fence_value}]",
            (
                "    /\\ lastMutationFence' = "
                f"[lastMutationFence EXCEPT ![t] = {new_fence}]"
            ),
            f"    /\\ evidence' = [evidence EXCEPT ![t] = {evidence_value}]",
            f"    /\\ merged' = [merged EXCEPT ![t] = {merged_value}]",
            (
                "    /\\ mergeCount' = "
                f"[mergeCount EXCEPT ![t] = {merge_count_value}]"
            ),
            f"    /\\ active' = {active_value}",
            (
                "    /\\ progressed' = "
                f"[progressed EXCEPT ![t] = {progressed_value}]"
            ),
            (
                "    /\\ retryCount' = "
                f"[retryCount EXCEPT ![t] = {retry_value}]"
            ),
            "    /\\ step' = step + 1",
        ]
        return [f"{rule.name}(t, a, f) =="] + guards + updates


@dataclass(frozen=True)
class CounterexampleState:
    index: int
    label: str
    assignments: Mapping[str, str]
    raw: str

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ModelValidationError("counterexample state index must be positive")
        object.__setattr__(
            self, "assignments", _strict_mapping(self.assignments, "assignments")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "assignments": dict(self.assignments),
            "raw": self.raw,
        }


@dataclass(frozen=True)
class CounterexampleTrace:
    states: tuple[CounterexampleState, ...] = ()
    raw: str = ""
    source: str = "stdout_stderr"

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": [item.to_dict() for item in self.states],
            "raw": self.raw,
            "source": self.source,
        }


@dataclass(frozen=True)
class ModelCheckerExecutionConfig:
    timeout_seconds: float = DEFAULT_MODEL_CHECK_TIMEOUT_SECONDS
    version_timeout_seconds: float = DEFAULT_VERSION_TIMEOUT_SECONDS
    max_output_bytes: int = DEFAULT_MAX_MODEL_CHECK_OUTPUT_BYTES

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.version_timeout_seconds <= 0:
            raise ModelValidationError("execution timeouts must be positive")
        _positive_int(self.max_output_bytes, "max_output_bytes")


@dataclass(frozen=True)
class ModelCheckReceipt:
    """Self-contained receipt for one exact bounded checker execution."""

    tool: ModelCheckerTool
    status: ModelCheckStatus
    model: GeneratedSupervisorStateModel
    configuration_text: str
    executable: str
    tool_version: str
    version_returncode: int | None
    version_stdout: str
    version_stderr: str
    command: tuple[str, ...]
    version_command: tuple[str, ...]
    started_at: str
    duration_ms: int
    timeout_seconds: float
    max_output_bytes: int
    returncode: int | None
    stdout: str
    stderr: str
    output_truncated: bool
    reason: str
    checked_safety_properties: tuple[str, ...]
    checked_liveness_properties: tuple[str, ...]
    counterexample: CounterexampleTrace | None = None
    schema: str = MODEL_CHECK_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool", _enum_value(self.tool, ModelCheckerTool, "model checker")
        )
        object.__setattr__(
            self, "status", _enum_value(self.status, ModelCheckStatus, "status")
        )
        if self.schema != MODEL_CHECK_RECEIPT_SCHEMA:
            raise ModelValidationError("unsupported model-check receipt schema")
        if self.duration_ms < 0 or self.timeout_seconds <= 0:
            raise ModelValidationError("invalid receipt timing")
        _positive_int(self.max_output_bytes, "max_output_bytes")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "version_command", tuple(self.version_command))
        object.__setattr__(
            self,
            "checked_safety_properties",
            tuple(sorted(set(self.checked_safety_properties))),
        )
        object.__setattr__(
            self,
            "checked_liveness_properties",
            tuple(sorted(set(self.checked_liveness_properties))),
        )
        if not set(self.checked_safety_properties) <= set(SAFETY_PROPERTIES):
            raise ModelValidationError(
                "receipt contains an unknown safety property"
            )
        if not set(self.checked_liveness_properties) <= set(LIVENESS_PROPERTIES):
            raise ModelValidationError(
                "receipt contains an unknown liveness property"
            )
        if (
            self.tool is ModelCheckerTool.APALACHE
            and self.checked_liveness_properties
        ):
            raise ModelValidationError(
                "Apalache receipt cannot claim temporal liveness checking"
            )
        if self.configuration_text != self.model.configuration_for(self.tool):
            raise ModelValidationError(
                "receipt configuration does not match the exact generated model"
            )
        if (
            self.status is ModelCheckStatus.UNAVAILABLE
            and (self.checked_safety_properties or self.checked_liveness_properties)
        ):
            raise ModelValidationError(
                "unavailable checker cannot claim properties were checked"
            )

    @property
    def bounded(self) -> bool:
        return True

    @property
    def unbounded_proof(self) -> bool:
        return False

    @property
    def passed(self) -> bool:
        return self.status is ModelCheckStatus.PASSED

    @property
    def assurance_label(self) -> str:
        if self.passed:
            prefix = "bounded_model_check_passed"
        else:
            prefix = f"bounded_model_check_{self.status.value}"
        return f"{prefix}[tool={self.tool.value};{self.model.bounds.label}]"

    @property
    def output_identity(self) -> str:
        return content_identity({"stdout": self.stdout, "stderr": self.stderr})

    @property
    def receipt_id(self) -> str:
        # Execution timeouts are intentionally represented as fractional
        # seconds.  Planning contracts reject floats, so execution-receipt
        # identity uses strict sorted JSON directly (with NaN disallowed).
        encoded = json.dumps(
            self.to_dict(include_id=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return _sha256_text(encoded)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "version": SUPERVISOR_STATE_MODEL_VERSION,
            "tool": self.tool.value,
            "status": self.status.value,
            "bounded": True,
            "unbounded_proof": False,
            "assurance_label": self.assurance_label,
            "model": self.model.to_dict(include_text=True),
            "configuration_text": self.configuration_text,
            "bounds": self.model.bounds.to_dict(),
            "executable": self.executable,
            "tool_version": self.tool_version,
            "version_returncode": self.version_returncode,
            "version_stdout": self.version_stdout,
            "version_stderr": self.version_stderr,
            "command": list(self.command),
            "version_command": list(self.version_command),
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_sha256": _sha256_text(self.stdout),
            "stderr_sha256": _sha256_text(self.stderr),
            "output_identity": self.output_identity,
            "output_truncated": self.output_truncated,
            "reason": self.reason,
            "checked_safety_properties": list(self.checked_safety_properties),
            "checked_liveness_properties": list(
                self.checked_liveness_properties
            ),
            "counterexample": (
                self.counterexample.to_dict() if self.counterexample else None
            ),
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload


CommandRunner = Callable[[CommandRequest], CommandResult | Mapping[str, Any]]
ExecutableFinder = Callable[[str], str | None]


def _normalize_result(value: CommandResult | Mapping[str, Any]) -> CommandResult:
    if isinstance(value, CommandResult):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("command runner must return CommandResult or a mapping")
    return CommandResult(
        returncode=value.get("returncode"),
        stdout=str(value.get("stdout") or ""),
        stderr=str(value.get("stderr") or ""),
        timed_out=bool(value.get("timed_out", False)),
        error=str(value["error"]) if value.get("error") else None,
        output_truncated=bool(value.get("output_truncated", False)),
    )


def _bounded_text(value: str, maximum_bytes: int) -> tuple[str, bool]:
    encoded = str(value).encode("utf-8", errors="replace")
    if len(encoded) <= maximum_bytes:
        return str(value), False
    return (
        encoded[:maximum_bytes].decode("utf-8", errors="replace"),
        True,
    )


def _default_command_runner(request: CommandRequest) -> CommandResult:
    """Run a checker with a process-group deadline and bounded capture."""

    try:
        process = subprocess.Popen(
            list(request.command),
            stdin=subprocess.PIPE if request.stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=request.cwd,
            env={**os.environ, "NO_PROXY": "*", "no_proxy": "*"},
            start_new_session=True,
        )
    except OSError as exc:
        return CommandResult(None, error=f"{type(exc).__name__}: {exc}")

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}

    def drain(name: str, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(16 * 1024)
                if not chunk:
                    return
                remaining = request.max_output_bytes - len(buffers[name])
                if remaining > 0:
                    buffers[name].extend(chunk[:remaining])
                if len(chunk) > max(remaining, 0):
                    truncated[name] = True
        except (OSError, ValueError):
            return

    threads = [
        threading.Thread(
            target=drain,
            args=(name, stream),
            name=f"supervisor-model-{name}",
            daemon=True,
        )
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr))
    ]
    for thread in threads:
        thread.start()
    if process.stdin is not None:
        try:
            if request.stdin_text is not None:
                process.stdin.write(request.stdin_text.encode("utf-8"))
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
    try:
        returncode = process.wait(timeout=request.timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired:
        returncode = None
        timed_out = True
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
        except (OSError, ProcessLookupError):
            process.kill()
        process.wait()
    for thread in threads:
        thread.join(timeout=1)
    return CommandResult(
        returncode,
        stdout=bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
        stderr=bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
        timed_out=timed_out,
        output_truncated=truncated["stdout"] or truncated["stderr"],
    )


def parse_counterexample_trace(output: str) -> CounterexampleTrace:
    """Parse TLC/Apalache state blocks while retaining the exact raw trace."""

    text = str(output)
    pattern = re.compile(
        r"(?ms)^State\s+(\d+):\s*([^\n]*)\n(.*?)(?=^State\s+\d+:|\Z)"
    )
    states: list[CounterexampleState] = []
    for match in pattern.finditer(text):
        body = match.group(3).rstrip()
        assignments: dict[str, str] = {}
        for assignment in re.finditer(
            r"(?m)^\s*/?\\?\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$",
            body,
        ):
            assignments[assignment.group(1)] = assignment.group(2)
        raw = match.group(0).rstrip()
        states.append(
            CounterexampleState(
                index=int(match.group(1)),
                label=match.group(2).strip().removeprefix("<").removesuffix(">"),
                assignments=assignments,
                raw=raw,
            )
        )
    return CounterexampleTrace(states=tuple(states), raw=text)


class SupervisorStateModelChecker:
    """Execute TLC or Apalache without making discovery count as evidence."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner | None = None,
        which: ExecutableFinder = shutil.which,
        wall_clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._run = command_runner or _default_command_runner
        self._which = which
        self._wall_clock = wall_clock
        self._monotonic = monotonic

    def check(
        self,
        model: GeneratedSupervisorStateModel,
        *,
        tool: ModelCheckerTool | str = ModelCheckerTool.TLC,
        executable: str | Path | None = None,
        config: ModelCheckerExecutionConfig | None = None,
    ) -> ModelCheckReceipt:
        selected = _enum_value(tool, ModelCheckerTool, "model checker")
        execution = config or ModelCheckerExecutionConfig()
        candidates = ("tlc", "tlc2") if selected is ModelCheckerTool.TLC else (
            "apalache-mc",
            "apalache",
        )
        resolved = str(executable) if executable else next(
            (path for name in candidates if (path := self._which(name))), ""
        )
        started_at = _utc_timestamp(self._wall_clock)
        started = self._monotonic()
        configuration = model.configuration_for(selected)
        safety = tuple(SAFETY_PROPERTIES)
        liveness = (
            tuple(LIVENESS_PROPERTIES)
            if selected is ModelCheckerTool.TLC
            else ()
        )
        if not resolved:
            return ModelCheckReceipt(
                tool=selected,
                status=ModelCheckStatus.UNAVAILABLE,
                model=model,
                configuration_text=configuration,
                executable="",
                tool_version="",
                version_returncode=None,
                version_stdout="",
                version_stderr="",
                command=(),
                version_command=(),
                started_at=started_at,
                duration_ms=max(0, int((self._monotonic() - started) * 1000)),
                timeout_seconds=execution.timeout_seconds,
                max_output_bytes=execution.max_output_bytes,
                returncode=None,
                stdout="",
                stderr="",
                output_truncated=False,
                reason=(
                    f"{selected.value} executable unavailable; no model check ran"
                ),
                checked_safety_properties=(),
                checked_liveness_properties=(),
            )

        version_command = (
            (resolved, "--version")
            if selected is ModelCheckerTool.TLC
            else (resolved, "version")
        )
        version_result = self._call(
            CommandRequest(
                command=version_command,
                stdin_text=None,
                cwd=None,
                timeout_seconds=execution.version_timeout_seconds,
                max_output_bytes=min(execution.max_output_bytes, 64 * 1024),
            )
        )
        version = (version_result.stdout or version_result.stderr).strip()
        if version_result.error:
            version = f"unavailable: {version_result.error}"
        elif version_result.timed_out:
            version = "unavailable: version command timed out"

        with tempfile.TemporaryDirectory(prefix="supervisor-tla-") as directory:
            root = Path(directory)
            tla_path = root / f"{model.module_name}.tla"
            tla_path.write_text(model.model_text, encoding="utf-8")
            if selected is ModelCheckerTool.TLC:
                config_path = root / f"{model.module_name}.cfg"
                config_path.write_text(model.tlc_config_text, encoding="utf-8")
                command = (
                    resolved,
                    "-config",
                    str(config_path),
                    str(tla_path),
                )
            else:
                config_path = root / "apalache.cfg"
                config_path.write_text(
                    model.apalache_config_text, encoding="utf-8"
                )
                command = (
                    resolved,
                    "check",
                    f"--config={config_path}",
                    f"--length={model.bounds.max_steps}",
                    "--inv=Safety",
                    "--no-deadlock",
                    str(tla_path),
                )
            result = self._call(
                CommandRequest(
                    command=command,
                    stdin_text=None,
                    cwd=str(root),
                    timeout_seconds=execution.timeout_seconds,
                    max_output_bytes=execution.max_output_bytes,
                )
            )
            supplemental_trace = self._counterexample_file(root, config_path)

        combined = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        )
        status, reason = self._classify(selected, result, combined)
        counterexample: CounterexampleTrace | None = None
        if status is ModelCheckStatus.COUNTEREXAMPLE:
            trace_text = supplemental_trace or combined
            counterexample = parse_counterexample_trace(trace_text)
            if supplemental_trace:
                counterexample = CounterexampleTrace(
                    states=counterexample.states,
                    raw=counterexample.raw,
                    source="checker_counterexample_file",
                )
        return ModelCheckReceipt(
            tool=selected,
            status=status,
            model=model,
            configuration_text=configuration,
            executable=resolved,
            tool_version=version,
            version_returncode=version_result.returncode,
            version_stdout=version_result.stdout,
            version_stderr=version_result.stderr,
            command=command,
            version_command=version_command,
            started_at=started_at,
            duration_ms=max(0, int((self._monotonic() - started) * 1000)),
            timeout_seconds=execution.timeout_seconds,
            max_output_bytes=execution.max_output_bytes,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            output_truncated=result.output_truncated,
            reason=reason,
            checked_safety_properties=safety,
            checked_liveness_properties=liveness,
            counterexample=counterexample,
        )

    def _call(self, request: CommandRequest) -> CommandResult:
        try:
            raw = _normalize_result(self._run(request))
        except Exception as exc:  # fail closed across injected/provider runners
            return CommandResult(
                None, error=f"{type(exc).__name__}: {exc}"
            )
        stdout, stdout_cut = _bounded_text(
            raw.stdout, request.max_output_bytes
        )
        stderr, stderr_cut = _bounded_text(
            raw.stderr, request.max_output_bytes
        )
        return CommandResult(
            returncode=raw.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=raw.timed_out,
            error=raw.error,
            output_truncated=(
                raw.output_truncated or stdout_cut or stderr_cut
            ),
        )

    @staticmethod
    def _classify(
        tool: ModelCheckerTool, result: CommandResult, combined: str
    ) -> tuple[ModelCheckStatus, str]:
        lower = combined.lower()
        if result.timed_out:
            return (
                ModelCheckStatus.TIMED_OUT,
                "bounded model check timed out before completing exploration",
            )
        if result.error:
            return (
                ModelCheckStatus.ERROR,
                f"bounded model checker failed: {result.error}",
            )
        if any(marker in lower for marker in _COUNTEREXAMPLE_MARKERS):
            return (
                ModelCheckStatus.COUNTEREXAMPLE,
                "bounded model checker reported a counterexample",
            )
        if result.output_truncated:
            return (
                ModelCheckStatus.UNKNOWN,
                "bounded checker output was truncated, so success cannot be established",
            )
        success_markers = (
            _TLC_SUCCESS_MARKERS
            if tool is ModelCheckerTool.TLC
            else _APALACHE_SUCCESS_MARKERS
        )
        if result.returncode == 0 and any(
            marker in lower for marker in success_markers
        ):
            return (
                ModelCheckStatus.PASSED,
                "bounded model check passed within the explicitly recorded explored bounds",
            )
        if result.returncode not in (0, None):
            return (
                ModelCheckStatus.ERROR,
                f"bounded model checker exited with code {result.returncode}",
            )
        return (
            ModelCheckStatus.UNKNOWN,
            "checker output did not contain a reviewed success or counterexample marker",
        )

    @staticmethod
    def _counterexample_file(root: Path, config_path: Path) -> str:
        candidates: list[Path] = []
        for pattern in ("*counterexample*", "*violation*", "example*.tla"):
            candidates.extend(root.rglob(pattern))
        for path in sorted(set(candidates), key=lambda item: str(item)):
            if path == config_path or not path.is_file():
                continue
            try:
                if path.stat().st_size <= DEFAULT_MAX_MODEL_CHECK_OUTPUT_BYTES:
                    return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        return ""


def generate_supervisor_state_model(
    schema: SupervisorTransitionSchema | Mapping[str, Any] | FormalWorkPlan,
    *,
    bounds: ModelCheckBounds | Mapping[str, Any] | None = None,
    module_name: str = "SupervisorState",
) -> GeneratedSupervisorStateModel:
    """Generate a deterministic finite model from a schema or canonical plan."""

    is_plan_record = isinstance(schema, Mapping) and (
        schema.get("schema") == FormalWorkPlan.SCHEMA
        or (
            "vocabulary_profile_id" in schema
            and "tasks" in schema
            and "goals" in schema
        )
    )
    transition_schema = (
        SupervisorTransitionSchema.from_formal_work_plan(schema)
        if isinstance(schema, FormalWorkPlan) or is_plan_record
        else schema
    )
    return SupervisorStateModelGenerator().generate(
        transition_schema, bounds=bounds, module_name=module_name
    )


def check_supervisor_state_model(
    model: GeneratedSupervisorStateModel,
    *,
    tool: ModelCheckerTool | str = ModelCheckerTool.TLC,
    executable: str | Path | None = None,
    config: ModelCheckerExecutionConfig | None = None,
    command_runner: CommandRunner | None = None,
    which: ExecutableFinder = shutil.which,
) -> ModelCheckReceipt:
    """Run one bounded model check and return a self-contained receipt."""

    return SupervisorStateModelChecker(
        command_runner=command_runner, which=which
    ).check(model, tool=tool, executable=executable, config=config)


# Descriptive compatibility aliases used by callers in the formal-plan layer.
SupervisorStateTransition = TransitionRule
SupervisorStateSchema = SupervisorTransitionSchema
TlaModel = GeneratedSupervisorStateModel
TlaModelGenerator = SupervisorStateModelGenerator
ModelCheckerReceipt = ModelCheckReceipt
generate_tla_model = generate_supervisor_state_model
run_model_checker = check_supervisor_state_model


__all__ = [
    "SUPERVISOR_STATE_MODEL_VERSION",
    "SUPERVISOR_TRANSITION_SCHEMA",
    "SUPERVISOR_TLA_MODEL_SCHEMA",
    "MODEL_CHECK_BOUNDS_SCHEMA",
    "MODEL_CHECK_RECEIPT_SCHEMA",
    "TLA_TRANSLATOR_ID",
    "TLA_TRANSLATOR_VERSION",
    "DEFAULT_MODEL_CHECK_TIMEOUT_SECONDS",
    "DEFAULT_VERSION_TIMEOUT_SECONDS",
    "DEFAULT_MAX_MODEL_CHECK_OUTPUT_BYTES",
    "SAFETY_PROPERTIES",
    "LIVENESS_PROPERTIES",
    "ModelValidationError",
    "ModelCheckerTool",
    "ModelCheckStatus",
    "TransitionRule",
    "SupervisorStateTransition",
    "DEFAULT_SUPERVISOR_TRANSITIONS",
    "SupervisorTransitionSchema",
    "SupervisorStateSchema",
    "ModelCheckBounds",
    "GeneratedSupervisorStateModel",
    "TlaModel",
    "SupervisorStateModelGenerator",
    "TlaModelGenerator",
    "CounterexampleState",
    "CounterexampleTrace",
    "ModelCheckerExecutionConfig",
    "ModelCheckReceipt",
    "ModelCheckerReceipt",
    "SupervisorStateModelChecker",
    "parse_counterexample_trace",
    "generate_supervisor_state_model",
    "generate_tla_model",
    "check_supervisor_state_model",
    "run_model_checker",
]
