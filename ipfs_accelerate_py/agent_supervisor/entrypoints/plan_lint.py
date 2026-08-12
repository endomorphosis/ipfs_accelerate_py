"""Reusable goal/task/profile plan lint for prompt-only entrypoints (ASE-011).

``lint_supervisor_plan`` is a pure, read-only checker.  It never mutates the
plan document, never starts work, and never admits proposals.  Findings are
bound to parsed goal/task/profile identities and are stable under identical
inputs.

Lint coverage:

- **duplicate** goal or task identifiers;
- **unknown** dependency or parent references;
- **cyclic** goal hierarchy or task dependency graphs;
- **missing** required metadata (title, acceptance, outputs, validation, …);
- **unsafe** validation commands, path escapes, or secret-bearing fields;
- **conflicting** predicted-file ownership and profile completeness gaps.

The module deliberately accepts plain mappings so callers can lint backlog
boards, objective packets, and inferred profiles without importing the formal
planning runtime.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)

from . import contracts as _contracts
from .contracts import EntrypointContractError
from .inference_explain import INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
PLAN_LINT_REPORT_SCHEMA: Final = f"{SCHEMA_PREFIX}/plan-lint-report@1"
PLAN_LINT_FINDING_SCHEMA: Final = f"{SCHEMA_PREFIX}/plan-lint-finding@1"
SUPERVISOR_PLAN_DOCUMENT_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/supervisor-plan-document@1"
)

PLAN_LINT_REQUIREMENT_ID: Final = (
    "requirement:agent-supervisor.entrypoints.plan-lint@1"
)

MAX_FINDINGS: Final = 256
MAX_GOALS: Final = 1_024
MAX_TASKS: Final = 4_096
MAX_ID_BYTES: Final = 256
MAX_TEXT_BYTES: Final = 4_096
MAX_PREDICTED_FILES: Final = 256
MAX_VALIDATION_COMMANDS: Final = 64
MAX_DEPENDENCIES: Final = 256

REQUIRED_GOAL_FIELDS: Final[tuple[str, ...]] = (
    "goal_id",
    "title",
    "acceptance",
)
REQUIRED_TASK_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "title",
    "goal_id",
    "acceptance",
    "outputs",
    "predicted_files",
    "validation_commands",
)
REQUIRED_PROFILE_FIELDS: Final[tuple[str, ...]] = (
    "profile_name",
    "mode",
    "repository_root",
    "state_root",
    "run_namespace",
    "policy_cid",
    "principal_ref",
    "effect_ceiling_cid",
    "task_source_kind",
    "task_source_cid",
    "provider_route",
    "validation_profile_cid",
    "worktree_strategy",
    "expected_effects",
)

_SHELL_METACHAR_RE = re.compile(r"[|;&`$<>\n\r]")
_PATH_ESCAPE_MARKERS: Final[tuple[str, ...]] = (
    "..",
    "\x00",
    "\\",
)
_UNSAFE_VALIDATION_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "rm",
        "rmdir",
        "dd",
        "mkfs",
        "shutdown",
        "reboot",
        "curl",
        "wget",
        "nc",
        "ncat",
        "bash",
        "sh",
        "zsh",
        "fish",
        "powershell",
        "pwsh",
        "cmd",
        "eval",
        "exec",
        "python -c",
        "perl -e",
        "ruby -e",
    }
)
# Reuse the closed secret-scanner inventory from contracts so this module does
# not re-introduce PEM header / credential marker literals that the proposal
# gate treats as secret-bearing content when added as new files.
_JWT_RE = _contracts._JWT_RE
_SECRET_ASSIGNMENT_RE = _contracts._SECRET_ASSIGNMENT_RE
_KNOWN_SECRET_TOKEN_RE = _contracts._KNOWN_SECRET_TOKEN_RE
_SECRET_TEXT_MARKERS = _contracts._SECRET_TEXT_MARKERS
_FORBIDDEN_ARG_MARKERS = _contracts._FORBIDDEN_ARG_MARKERS


class PlanLintError(EntrypointContractError):
    """Raised when the plan document itself is not well-formed enough to lint."""


class PlanLintKind(str, Enum):
    DUPLICATE = "duplicate"
    UNKNOWN = "unknown"
    CYCLIC = "cyclic"
    MISSING = "missing"
    UNSAFE = "unsafe"
    CONFLICTING = "conflicting"


class PlanLintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class PlanLintSubjectKind(str, Enum):
    GOAL = "goal"
    TASK = "task"
    PROFILE = "profile"
    PLAN = "plan"
    VALIDATION = "validation"
    PREDICTED_FILE = "predicted_file"


def _safe_error(message: str) -> PlanLintError:
    return PlanLintError(_redact(str(message or "plan lint failed"))[:512])


def _contains_secret_material(value: str) -> bool:
    if not value:
        return False
    lowered = value.casefold()
    if any(marker in lowered for marker in _SECRET_TEXT_MARKERS):
        return True
    if _JWT_RE.search(value):
        return True
    if _SECRET_ASSIGNMENT_RE.search(value):
        return True
    if _KNOWN_SECRET_TOKEN_RE.search(value):
        return True
    return False


def _redact(value: str) -> str:
    text = str(value or "")
    text = _JWT_RE.sub("[redacted-jwt]", text)
    text = _KNOWN_SECRET_TOKEN_RE.sub("[redacted-token]", text)
    text = _SECRET_ASSIGNMENT_RE.sub("[redacted-assignment]", text)
    for marker in _SECRET_TEXT_MARKERS:
        if marker in text.casefold():
            text = re.sub(re.escape(marker), "[redacted-marker]", text, flags=re.I)
    return text


def _raw_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _text(
    value: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_TEXT_BYTES,
    reject_secrets: bool = True,
) -> str:
    text = _raw_text(value)
    if required and not text:
        raise _safe_error(f"{name} is required")
    if len(text.encode("utf-8")) > maximum:
        raise _safe_error(f"{name} exceeds {maximum} UTF-8 bytes")
    if "\x00" in text:
        raise _safe_error(f"{name} contains a NUL byte")
    if reject_secrets and _contains_secret_material(text):
        raise _safe_error(f"{name} contains secret-bearing material")
    return text


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    text = _raw_text(value)
    if len(text.encode("utf-8")) > MAX_ID_BYTES:
        raise _safe_error(f"{name} exceeds {MAX_ID_BYTES} UTF-8 bytes")
    if required and not text:
        return ""
    if text and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", text):
        # Non-canonical identifiers are treated as missing/invalid by callers.
        return ""
    if text and _contains_secret_material(text):
        return ""
    return text


def _string_list(
    value: Any,
    name: str,
    *,
    maximum_items: int,
    paths: bool = False,
    allow_secret_items: bool = False,
) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        raw: Iterable[Any] = re.split(r"[,;\n]+", value)
    elif isinstance(value, Mapping):
        raw = value.keys()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw = value
    else:
        raise _safe_error(f"{name} must be a sequence of strings")
    items: list[str] = []
    for item in raw:
        if item in (None, ""):
            continue
        text = _raw_text(item)
        if not text or "\x00" in text:
            continue
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            continue
        if not allow_secret_items and _contains_secret_material(text):
            # Keep the item so unsafe/path checks can report it without embedding
            # secret bodies into durable finding messages.
            text = f"[redacted-{name}-item]"
        if paths:
            text = text.replace("\\", "/")
            while text.startswith("./"):
                text = text[2:]
            text = re.sub(r"/+", "/", text).rstrip("/")
        if text:
            items.append(text)
    unique = sorted(set(items), key=lambda item: (item.casefold(), item))
    if len(unique) > maximum_items:
        unique = unique[:maximum_items]
    return tuple(unique)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _safe_error(f"{name} must be a mapping")
    return value


def _normalize_key(key: Any) -> str:
    return str(key or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _get(payload: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    normalized = {_normalize_key(key): value for key, value in payload.items()}
    for name in names:
        key = _normalize_key(name)
        if key in normalized:
            return normalized[key]
    return default


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _path_is_unsafe(path: str) -> bool:
    if not path:
        return True
    if path.startswith("/") or path.startswith("~"):
        return True
    lowered = path.casefold()
    for marker in _PATH_ESCAPE_MARKERS:
        if marker == ".." and (
            path == ".."
            or path.startswith("../")
            or "/../" in path
            or path.endswith("/..")
        ):
            return True
        if marker != ".." and marker in path:
            return True
    if _contains_secret_material(path):
        return True
    if any(part == "" for part in path.split("/")):
        return True
    _ = lowered
    return False


def _validation_is_unsafe(command: str) -> tuple[bool, str]:
    text = command.strip()
    if not text:
        return True, "empty_validation_command"
    if _contains_secret_material(text):
        return True, "secret_bearing_validation"
    lowered = text.casefold()
    for marker in _FORBIDDEN_ARG_MARKERS:
        if marker in lowered:
            return True, "forbidden_validation_flag"
    if _SHELL_METACHAR_RE.search(text):
        return True, "shell_metacharacters"
    for token in _UNSAFE_VALIDATION_TOKENS:
        if token in lowered.split() or lowered.startswith(token + " ") or lowered == token:
            return True, "unsafe_validation_token"
        if " " in token and token in lowered:
            return True, "unsafe_validation_token"
    # Require an allowlisted pytest/python -m form for structured validations.
    tokens = text.split()
    if tokens[0] not in {"python", "python3", "py", "pytest", "ruff", "mypy", "git"}:
        return True, "validation_command_not_allowlisted"
    if tokens[0] in {"python", "python3", "py"}:
        if len(tokens) < 3 or tokens[1] != "-m":
            return True, "python_validation_must_use_module_form"
    return False, ""


def _cyclic_nodes(graph: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Tarjan-inspired cycle detection; returns nodes participating in cycles."""

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    cyclic: set[str] = set()

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for dep in graph.get(node, ()):
            if dep not in graph and dep not in indices:
                # Unknown edges are reported separately; skip for cycles.
                continue
            if dep not in indices:
                if dep in graph:
                    strongconnect(dep)
                    lowlinks[node] = min(lowlinks[node], lowlinks[dep])
                continue
            if dep in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[dep])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                cyclic.update(component)
            elif component and component[0] in graph.get(component[0], ()):
                cyclic.add(component[0])

    for node in sorted(graph):
        if node not in indices:
            strongconnect(node)
    return tuple(sorted(cyclic))


@dataclass(frozen=True)
class PlanLintFinding:
    """One immutable, body-free lint finding."""

    SCHEMA: ClassVar[str] = PLAN_LINT_FINDING_SCHEMA

    kind: PlanLintKind
    severity: PlanLintSeverity
    code: str
    message: str
    subject_kind: PlanLintSubjectKind
    subject_id: str
    related_ids: tuple[str, ...] = ()
    field_name: str = ""

    def __post_init__(self) -> None:
        kind = self.kind
        if not isinstance(kind, PlanLintKind):
            kind = PlanLintKind(str(kind))
        object.__setattr__(self, "kind", kind)
        severity = self.severity
        if not isinstance(severity, PlanLintSeverity):
            severity = PlanLintSeverity(str(severity))
        object.__setattr__(self, "severity", severity)
        subject_kind = self.subject_kind
        if not isinstance(subject_kind, PlanLintSubjectKind):
            subject_kind = PlanLintSubjectKind(str(subject_kind))
        object.__setattr__(self, "subject_kind", subject_kind)
        object.__setattr__(
            self,
            "code",
            _text(self.code, "code", required=True, maximum=128),
        )
        object.__setattr__(
            self,
            "message",
            _text(self.message, "message", required=True, maximum=MAX_TEXT_BYTES),
        )
        object.__setattr__(
            self,
            "subject_id",
            _text(self.subject_id, "subject_id", required=True, maximum=MAX_ID_BYTES),
        )
        related = _string_list(
            self.related_ids, "related_ids", maximum_items=MAX_DEPENDENCIES
        )
        object.__setattr__(self, "related_ids", related)
        object.__setattr__(
            self,
            "field_name",
            _text(self.field_name, "field_name", maximum=128),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "subject_kind": self.subject_kind.value,
            "subject_id": self.subject_id,
            "related_ids": list(self.related_ids),
            "field_name": self.field_name,
        }

    @property
    def finding_id(self) -> str:
        return cid_for_dag_json(self.to_dict())


@dataclass(frozen=True)
class PlanLintReport:
    """Deterministic lint report for one supervisor plan document."""

    SCHEMA: ClassVar[str] = PLAN_LINT_REPORT_SCHEMA

    requirement_id: str
    plan_id: str
    findings: tuple[PlanLintFinding, ...]
    goal_count: int
    task_count: int
    profile_present: bool
    total_finding_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            _text(self.requirement_id, "requirement_id", required=True, maximum=256),
        )
        object.__setattr__(
            self,
            "plan_id",
            _text(self.plan_id, "plan_id", required=True, maximum=MAX_ID_BYTES),
        )
        findings = tuple(self.findings)
        if any(not isinstance(item, PlanLintFinding) for item in findings):
            raise _safe_error("findings must contain PlanLintFinding values")
        # Stable ordering: severity, kind, code, subject, related.
        findings = tuple(
            sorted(
                findings,
                key=lambda item: (
                    0 if item.severity is PlanLintSeverity.ERROR else 1,
                    item.kind.value,
                    item.code,
                    item.subject_kind.value,
                    item.subject_id,
                    item.field_name,
                    item.related_ids,
                    item.message,
                ),
            )
        )
        if len(findings) > MAX_FINDINGS:
            findings = findings[:MAX_FINDINGS]
        object.__setattr__(self, "findings", findings)
        for name in ("goal_count", "task_count", "total_finding_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise _safe_error(f"{name} must be a non-negative integer")
        if not isinstance(self.profile_present, bool):
            raise _safe_error("profile_present must be a boolean")
        if self.total_finding_count < len(findings):
            raise _safe_error(
                "total_finding_count cannot be smaller than retained findings"
            )
        _ = self.content_id

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self.to_dict(include_identity=False))

    @property
    def report_cid(self) -> str:
        return self.content_id

    @property
    def accepted(self) -> bool:
        return self.total_finding_count == 0

    @property
    def error_count(self) -> int:
        return sum(
            1 for item in self.findings if item.severity is PlanLintSeverity.ERROR
        )

    @property
    def warning_count(self) -> int:
        return sum(
            1 for item in self.findings if item.severity is PlanLintSeverity.WARNING
        )

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted({item.kind.value for item in self.findings}))

    def findings_of(self, kind: PlanLintKind | str) -> tuple[PlanLintFinding, ...]:
        key = kind if isinstance(kind, PlanLintKind) else PlanLintKind(str(kind))
        return tuple(item for item in self.findings if item.kind is key)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "requirement_id": self.requirement_id,
            "plan_id": self.plan_id,
            "findings": [item.to_dict() for item in self.findings],
            "goal_count": self.goal_count,
            "task_count": self.task_count,
            "profile_present": self.profile_present,
            "total_finding_count": self.total_finding_count,
            "accepted": self.accepted,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "kinds": list(self.kinds),
        }
        if include_identity:
            payload["report_cid"] = self.content_id
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            ensure_ascii=True,
        )


@dataclass(frozen=True)
class SupervisorPlanDocument:
    """Normalized, body-free plan document consumed by the linter."""

    SCHEMA: ClassVar[str] = SUPERVISOR_PLAN_DOCUMENT_SCHEMA

    plan_id: str
    goals: tuple[Mapping[str, Any], ...] = ()
    tasks: tuple[Mapping[str, Any], ...] = ()
    profile: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_id",
            _identifier(self.plan_id, "plan_id", required=True),
        )
        goals = tuple(self.goals or ())
        tasks = tuple(self.tasks or ())
        if len(goals) > MAX_GOALS:
            raise _safe_error(f"goals exceeds {MAX_GOALS} items")
        if len(tasks) > MAX_TASKS:
            raise _safe_error(f"tasks exceeds {MAX_TASKS} items")
        for index, goal in enumerate(goals):
            _mapping(goal, f"goals[{index}]")
        for index, task in enumerate(tasks):
            _mapping(task, f"tasks[{index}]")
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "tasks", tasks)
        if self.profile is not None:
            object.__setattr__(self, "profile", dict(_mapping(self.profile, "profile")))
        metadata = self.metadata or {}
        object.__setattr__(self, "metadata", dict(_mapping(metadata, "metadata")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_id": self.plan_id,
            "goals": [dict(item) for item in self.goals],
            "tasks": [dict(item) for item in self.tasks],
            "profile": dict(self.profile) if self.profile is not None else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SupervisorPlanDocument:
        payload = _mapping(value, "plan")
        plan_id = _get(payload, "plan_id", "id", "plan", default="plan")
        goals = _get(payload, "goals", "objectives", default=()) or ()
        tasks = _get(payload, "tasks", default=()) or ()
        profile = _get(payload, "profile", "resolved_profile", default=None)
        metadata = _get(payload, "metadata", default={}) or {}
        if isinstance(goals, Mapping):
            goals = list(goals.values())
        if isinstance(tasks, Mapping):
            tasks = list(tasks.values())
        if not isinstance(goals, Sequence) or isinstance(goals, (str, bytes)):
            raise _safe_error("goals must be a sequence")
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
            raise _safe_error("tasks must be a sequence")
        return cls(
            plan_id=str(plan_id or "plan"),
            goals=tuple(goals),
            tasks=tuple(tasks),
            profile=profile if isinstance(profile, Mapping) else None,
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )


class _FindingBuilder:
    def __init__(self) -> None:
        self._items: list[PlanLintFinding] = []
        self._total = 0

    def add(
        self,
        *,
        kind: PlanLintKind,
        code: str,
        message: str,
        subject_kind: PlanLintSubjectKind,
        subject_id: str,
        related_ids: Sequence[str] = (),
        field_name: str = "",
        severity: PlanLintSeverity = PlanLintSeverity.ERROR,
    ) -> None:
        self._total += 1
        if len(self._items) >= MAX_FINDINGS:
            return
        self._items.append(
            PlanLintFinding(
                kind=kind,
                severity=severity,
                code=code,
                message=message,
                subject_kind=subject_kind,
                subject_id=subject_id,
                related_ids=tuple(related_ids),
                field_name=field_name,
            )
        )

    @property
    def findings(self) -> tuple[PlanLintFinding, ...]:
        return tuple(self._items)

    @property
    def total(self) -> int:
        return self._total


def _goal_id(goal: Mapping[str, Any]) -> str:
    return _identifier(
        _get(goal, "goal_id", "id", "objective_id", default=""),
        "goal_id",
        required=False,
    )


def _task_id(task: Mapping[str, Any]) -> str:
    return _identifier(
        _get(task, "task_id", "id", default=""),
        "task_id",
        required=False,
    )


def _lint_goal_hierarchy(
    goals: Sequence[Mapping[str, Any]],
    findings: _FindingBuilder,
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    seen: dict[str, int] = {}
    for index, goal in enumerate(goals):
        goal_id = _goal_id(goal)
        if not goal_id:
            findings.add(
                kind=PlanLintKind.MISSING,
                code="goal_id_missing",
                message="goal is missing goal_id",
                subject_kind=PlanLintSubjectKind.GOAL,
                subject_id=f"goal-index-{index}",
                field_name="goal_id",
            )
            continue
        seen[goal_id] = seen.get(goal_id, 0) + 1
        if seen[goal_id] == 2:
            findings.add(
                kind=PlanLintKind.DUPLICATE,
                code="duplicate_goal_id",
                message=f"duplicate goal_id {goal_id}",
                subject_kind=PlanLintSubjectKind.GOAL,
                subject_id=goal_id,
                field_name="goal_id",
            )
        if seen[goal_id] == 1:
            by_id[goal_id] = goal

    parent_graph: dict[str, list[str]] = {goal_id: [] for goal_id in by_id}
    dep_graph: dict[str, list[str]] = {goal_id: [] for goal_id in by_id}

    for goal_id, goal in by_id.items():
        for field_name in REQUIRED_GOAL_FIELDS:
            if field_name == "goal_id":
                continue
            if not _is_present(_get(goal, field_name, field_name.replace("_", ""))):
                # acceptance may also appear as acceptance_criteria
                if field_name == "acceptance" and _is_present(
                    _get(goal, "acceptance_criteria", "acceptance")
                ):
                    continue
                if field_name == "title" and _is_present(_get(goal, "name", "summary")):
                    continue
                findings.add(
                    kind=PlanLintKind.MISSING,
                    code=f"goal_{field_name}_missing",
                    message=f"goal {goal_id} is missing required field {field_name}",
                    subject_kind=PlanLintSubjectKind.GOAL,
                    subject_id=goal_id,
                    field_name=field_name,
                )

        parent = _identifier(
            _get(goal, "parent", "parent_goal_id", "parent_id", default=""),
            "parent",
            required=False,
        )
        if parent:
            if parent not in by_id:
                findings.add(
                    kind=PlanLintKind.UNKNOWN,
                    code="unknown_parent_goal",
                    message=f"goal {goal_id} references unknown parent {parent}",
                    subject_kind=PlanLintSubjectKind.GOAL,
                    subject_id=goal_id,
                    related_ids=(parent,),
                    field_name="parent",
                )
            else:
                parent_graph[goal_id].append(parent)

        deps = _string_list(
            _get(
                goal,
                "depends_on",
                "dependency_goal_ids",
                "dependencies",
                default=(),
            ),
            "depends_on",
            maximum_items=MAX_DEPENDENCIES,
        )
        for dep in deps:
            if dep == goal_id:
                findings.add(
                    kind=PlanLintKind.CYCLIC,
                    code="goal_self_dependency",
                    message=f"goal {goal_id} depends on itself",
                    subject_kind=PlanLintSubjectKind.GOAL,
                    subject_id=goal_id,
                    related_ids=(dep,),
                    field_name="depends_on",
                )
                continue
            if dep not in by_id:
                findings.add(
                    kind=PlanLintKind.UNKNOWN,
                    code="unknown_goal_dependency",
                    message=f"goal {goal_id} depends on unknown goal {dep}",
                    subject_kind=PlanLintSubjectKind.GOAL,
                    subject_id=goal_id,
                    related_ids=(dep,),
                    field_name="depends_on",
                )
            else:
                dep_graph[goal_id].append(dep)

        # Secret scan free-form text fields without retaining bodies.
        for field_name in ("title", "summary", "description", "acceptance"):
            value = _get(goal, field_name, default="")
            if isinstance(value, str) and _contains_secret_material(value):
                findings.add(
                    kind=PlanLintKind.UNSAFE,
                    code="goal_secret_bearing_field",
                    message=f"goal {goal_id} field {field_name} appears secret-bearing",
                    subject_kind=PlanLintSubjectKind.GOAL,
                    subject_id=goal_id,
                    field_name=field_name,
                )

    for node in _cyclic_nodes(parent_graph):
        findings.add(
            kind=PlanLintKind.CYCLIC,
            code="cyclic_goal_parent_hierarchy",
            message=f"goal {node} participates in a cyclic parent hierarchy",
            subject_kind=PlanLintSubjectKind.GOAL,
            subject_id=node,
            field_name="parent",
        )
    for node in _cyclic_nodes(dep_graph):
        findings.add(
            kind=PlanLintKind.CYCLIC,
            code="cyclic_goal_dependency",
            message=f"goal {node} participates in a cyclic goal dependency graph",
            subject_kind=PlanLintSubjectKind.GOAL,
            subject_id=node,
            field_name="depends_on",
        )
    return by_id


def _lint_tasks(
    tasks: Sequence[Mapping[str, Any]],
    goals_by_id: Mapping[str, Mapping[str, Any]],
    findings: _FindingBuilder,
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    seen: dict[str, int] = {}
    predicted_owners: dict[str, list[str]] = {}

    for index, task in enumerate(tasks):
        task_id = _task_id(task)
        if not task_id:
            findings.add(
                kind=PlanLintKind.MISSING,
                code="task_id_missing",
                message="task is missing task_id",
                subject_kind=PlanLintSubjectKind.TASK,
                subject_id=f"task-index-{index}",
                field_name="task_id",
            )
            continue
        seen[task_id] = seen.get(task_id, 0) + 1
        if seen[task_id] == 2:
            findings.add(
                kind=PlanLintKind.DUPLICATE,
                code="duplicate_task_id",
                message=f"duplicate task_id {task_id}",
                subject_kind=PlanLintSubjectKind.TASK,
                subject_id=task_id,
                field_name="task_id",
            )
        if seen[task_id] == 1:
            by_id[task_id] = task

    dep_graph: dict[str, list[str]] = {task_id: [] for task_id in by_id}

    for task_id, task in by_id.items():
        for field_name in REQUIRED_TASK_FIELDS:
            if field_name == "task_id":
                continue
            value = _get(
                task,
                field_name,
                "validation" if field_name == "validation_commands" else field_name,
                "acceptance_criteria" if field_name == "acceptance" else field_name,
                "name" if field_name == "title" else field_name,
                "predicted_files_json" if field_name == "predicted_files" else field_name,
                "outputs_json" if field_name == "outputs" else field_name,
            )
            if field_name == "goal_id":
                goal_id = _identifier(value or "", "goal_id", required=False)
                if not goal_id:
                    findings.add(
                        kind=PlanLintKind.MISSING,
                        code="task_goal_id_missing",
                        message=f"task {task_id} is missing goal_id",
                        subject_kind=PlanLintSubjectKind.TASK,
                        subject_id=task_id,
                        field_name="goal_id",
                    )
                elif goals_by_id and goal_id not in goals_by_id:
                    findings.add(
                        kind=PlanLintKind.UNKNOWN,
                        code="task_unknown_goal",
                        message=f"task {task_id} references unknown goal {goal_id}",
                        subject_kind=PlanLintSubjectKind.TASK,
                        subject_id=task_id,
                        related_ids=(goal_id,),
                        field_name="goal_id",
                    )
                continue
            if not _is_present(value):
                findings.add(
                    kind=PlanLintKind.MISSING,
                    code=f"task_{field_name}_missing",
                    message=f"task {task_id} is missing required field {field_name}",
                    subject_kind=PlanLintSubjectKind.TASK,
                    subject_id=task_id,
                    field_name=field_name,
                )

        deps = _string_list(
            _get(task, "depends_on", "dependencies", "depends_on_tasks", default=()),
            "depends_on",
            maximum_items=MAX_DEPENDENCIES,
        )
        for dep in deps:
            if dep == task_id:
                findings.add(
                    kind=PlanLintKind.CYCLIC,
                    code="task_self_dependency",
                    message=f"task {task_id} depends on itself",
                    subject_kind=PlanLintSubjectKind.TASK,
                    subject_id=task_id,
                    related_ids=(dep,),
                    field_name="depends_on",
                )
                continue
            if dep not in by_id:
                findings.add(
                    kind=PlanLintKind.UNKNOWN,
                    code="unknown_task_dependency",
                    message=f"task {task_id} depends on unknown task {dep}",
                    subject_kind=PlanLintSubjectKind.TASK,
                    subject_id=task_id,
                    related_ids=(dep,),
                    field_name="depends_on",
                )
            else:
                dep_graph[task_id].append(dep)

        predicted = _string_list(
            _get(task, "predicted_files", "predicted_files_json", default=()),
            "predicted_files",
            maximum_items=MAX_PREDICTED_FILES,
            paths=True,
        )
        for path in predicted:
            if _path_is_unsafe(path):
                findings.add(
                    kind=PlanLintKind.UNSAFE,
                    code="unsafe_predicted_file",
                    message=f"task {task_id} predicts an unsafe path",
                    subject_kind=PlanLintSubjectKind.PREDICTED_FILE,
                    subject_id=task_id,
                    related_ids=(path,),
                    field_name="predicted_files",
                )
                continue
            predicted_owners.setdefault(path, []).append(task_id)

        validations = _string_list(
            _get(
                task,
                "validation_commands",
                "validation",
                "validation_commands_json",
                default=(),
            ),
            "validation_commands",
            maximum_items=MAX_VALIDATION_COMMANDS,
        )
        for command in validations:
            unsafe, code = _validation_is_unsafe(command)
            if unsafe:
                findings.add(
                    kind=PlanLintKind.UNSAFE,
                    code=code or "unsafe_validation_command",
                    message=f"task {task_id} has an unsafe validation command",
                    subject_kind=PlanLintSubjectKind.VALIDATION,
                    subject_id=task_id,
                    field_name="validation_commands",
                )

        for field_name in ("title", "summary", "description", "acceptance"):
            value = _get(task, field_name, default="")
            if isinstance(value, str) and _contains_secret_material(value):
                findings.add(
                    kind=PlanLintKind.UNSAFE,
                    code="task_secret_bearing_field",
                    message=f"task {task_id} field {field_name} appears secret-bearing",
                    subject_kind=PlanLintSubjectKind.TASK,
                    subject_id=task_id,
                    field_name=field_name,
                )

    for node in _cyclic_nodes(dep_graph):
        findings.add(
            kind=PlanLintKind.CYCLIC,
            code="cyclic_task_dependency",
            message=f"task {node} participates in a cyclic task dependency graph",
            subject_kind=PlanLintSubjectKind.TASK,
            subject_id=node,
            field_name="depends_on",
        )

    for path, owners in sorted(predicted_owners.items()):
        unique_owners = sorted(set(owners))
        if len(unique_owners) > 1:
            findings.add(
                kind=PlanLintKind.CONFLICTING,
                code="predicted_file_conflict",
                message=(
                    f"predicted file {path} is claimed by multiple tasks: "
                    + ", ".join(unique_owners)
                ),
                subject_kind=PlanLintSubjectKind.PREDICTED_FILE,
                subject_id=path,
                related_ids=tuple(unique_owners),
                field_name="predicted_files",
            )
    return by_id


def _lint_profile(
    profile: Mapping[str, Any] | None,
    findings: _FindingBuilder,
) -> None:
    if profile is None:
        return
    profile_name = _text(
        _get(profile, "profile_name", "name", default="profile"),
        "profile_name",
        maximum=128,
    ) or "profile"
    for field_name in REQUIRED_PROFILE_FIELDS:
        value = _get(profile, field_name)
        if not _is_present(value):
            findings.add(
                kind=PlanLintKind.MISSING,
                code=f"profile_{field_name}_missing",
                message=f"profile is missing required field {field_name}",
                subject_kind=PlanLintSubjectKind.PROFILE,
                subject_id=profile_name,
                field_name=field_name,
            )

    # Credential handles are allowed as opaque references; bodies are not.
    for field_name in ("credential_handles", "environment_names"):
        value = _get(profile, field_name, default=())
        if isinstance(value, str) and _contains_secret_material(value):
            findings.add(
                kind=PlanLintKind.UNSAFE,
                code="profile_secret_bearing_field",
                message=f"profile field {field_name} appears secret-bearing",
                subject_kind=PlanLintSubjectKind.PROFILE,
                subject_id=profile_name,
                field_name=field_name,
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, str) and _contains_secret_material(item):
                    findings.add(
                        kind=PlanLintKind.UNSAFE,
                        code="profile_secret_bearing_field",
                        message=f"profile field {field_name} appears secret-bearing",
                        subject_kind=PlanLintSubjectKind.PROFILE,
                        subject_id=profile_name,
                        field_name=field_name,
                    )
                    break

    argv = _get(profile, "supervisor_argv", "daemon_argv", default=())
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        joined = " ".join(str(item) for item in argv)
        if _contains_secret_material(joined):
            findings.add(
                kind=PlanLintKind.UNSAFE,
                code="profile_argv_secret_bearing",
                message="profile argv appears to embed secret-bearing material",
                subject_kind=PlanLintSubjectKind.PROFILE,
                subject_id=profile_name,
                field_name="supervisor_argv",
            )
        lowered = joined.casefold()
        for marker in _FORBIDDEN_ARG_MARKERS:
            if marker in lowered:
                findings.add(
                    kind=PlanLintKind.UNSAFE,
                    code="profile_forbidden_argv_flag",
                    message="profile argv contains a forbidden secret/prompt flag",
                    subject_kind=PlanLintSubjectKind.PROFILE,
                    subject_id=profile_name,
                    field_name="supervisor_argv",
                )
                break

    # Cross-field profile consistency (effect-bearing worktree needs principal).
    worktree = str(_get(profile, "worktree_strategy", default="") or "").casefold()
    principal = _get(profile, "principal_ref", default="")
    if worktree and worktree not in {"", "none"} and not _is_present(principal):
        findings.add(
            kind=PlanLintKind.CONFLICTING,
            code="profile_worktree_without_principal",
            message="effect-bearing worktree strategy requires principal_ref",
            subject_kind=PlanLintSubjectKind.PROFILE,
            subject_id=profile_name,
            field_name="worktree_strategy",
        )


def lint_supervisor_plan(
    plan: SupervisorPlanDocument | Mapping[str, Any],
    *,
    requirement_id: str = INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID,
    require_profile: bool = False,
) -> PlanLintReport:
    """Lint a goal/task/profile plan document without mutating it.

    Parameters
    ----------
    plan:
        A :class:`SupervisorPlanDocument` or mapping with ``goals``, ``tasks``,
        and optional ``profile`` populations.
    requirement_id:
        Stable evidence identifier shared with inference explanation.
    require_profile:
        When true, a missing profile is reported as a missing finding.
    """

    try:
        if isinstance(plan, SupervisorPlanDocument):
            document = plan
        elif isinstance(plan, Mapping):
            document = SupervisorPlanDocument.from_mapping(plan)
        else:
            raise _safe_error("plan must be a SupervisorPlanDocument or mapping")
    except PlanLintError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed without bodies
        raise _safe_error("plan document could not be loaded") from exc

    # Deep-copy-ish isolation: never mutate caller mappings.
    frozen_goals = tuple(dict(item) for item in document.goals)
    frozen_tasks = tuple(dict(item) for item in document.tasks)
    frozen_profile = dict(document.profile) if document.profile is not None else None

    findings = _FindingBuilder()
    goals_by_id = _lint_goal_hierarchy(frozen_goals, findings)
    _lint_tasks(frozen_tasks, goals_by_id, findings)

    if require_profile and frozen_profile is None:
        findings.add(
            kind=PlanLintKind.MISSING,
            code="profile_missing",
            message="plan is missing an inferred or resolved profile",
            subject_kind=PlanLintSubjectKind.PROFILE,
            subject_id=document.plan_id,
            field_name="profile",
        )
    _lint_profile(frozen_profile, findings)

    if not frozen_goals and not frozen_tasks and frozen_profile is None:
        findings.add(
            kind=PlanLintKind.MISSING,
            code="plan_empty",
            message="plan contains no goals, tasks, or profile",
            subject_kind=PlanLintSubjectKind.PLAN,
            subject_id=document.plan_id,
        )

    report = PlanLintReport(
        requirement_id=requirement_id,
        plan_id=document.plan_id,
        findings=findings.findings,
        goal_count=len(frozen_goals),
        task_count=len(frozen_tasks),
        profile_present=frozen_profile is not None,
        total_finding_count=findings.total,
    )
    # Final body-free seal of the serialized report.
    serialized = report.to_json(indent=None)
    if _contains_secret_material(serialized):
        raise _safe_error("lint report projection became secret-bearing")
    return report


def lint_plan(
    plan: SupervisorPlanDocument | Mapping[str, Any],
    **kwargs: Any,
) -> PlanLintReport:
    """Alias for :func:`lint_supervisor_plan`."""

    return lint_supervisor_plan(plan, **kwargs)


__all__ = (
    "INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID",
    "MAX_FINDINGS",
    "PLAN_LINT_FINDING_SCHEMA",
    "PLAN_LINT_REPORT_SCHEMA",
    "PLAN_LINT_REQUIREMENT_ID",
    "PlanLintError",
    "PlanLintFinding",
    "PlanLintKind",
    "PlanLintReport",
    "PlanLintSeverity",
    "PlanLintSubjectKind",
    "REQUIRED_GOAL_FIELDS",
    "REQUIRED_PROFILE_FIELDS",
    "REQUIRED_TASK_FIELDS",
    "SUPERVISOR_PLAN_DOCUMENT_SCHEMA",
    "SupervisorPlanDocument",
    "lint_plan",
    "lint_supervisor_plan",
)
