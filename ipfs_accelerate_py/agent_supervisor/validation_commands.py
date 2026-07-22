"""Validation command parsing, classification, and impact selection.

Todo boards intentionally store validation as shell text.  This module turns
that text into immutable command specifications without attempting to split
shell pipelines or ``&&`` expressions.  The resulting metadata lets the
validation scheduler enforce stage barriers and make conservative, explainable
impact decisions while retaining the original command for logs.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import PurePosixPath
from typing import Iterable, Sequence

INLINE_CODE_COMMAND_RE = re.compile(r"^`+(?P<command>[^`]+?)`+\s*\.?\s*$", re.DOTALL)


class ValidationStage(IntEnum):
    """Ordered validation barriers.

    Cheap deterministic checks always finish before targeted or broad tests are
    admitted.  Numeric ordering is intentional and is used by the scheduler.
    """

    CHEAP = 0
    TARGETED = 1
    BROAD = 2

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class ValidationCommand:
    """One atomic shell validation command and its scheduling metadata."""

    command: str
    raw_command: str = ""
    stage: ValidationStage = ValidationStage.TARGETED
    resource_cost: int = 1
    impact_paths: tuple[str, ...] = ()
    environment_keys: tuple[str, ...] = ()
    cacheable: bool = True
    timeout_seconds: float | None = None
    ordinal: int = 0

    def with_stage(self, stage: ValidationStage) -> "ValidationCommand":
        return replace(self, stage=stage)


@dataclass(frozen=True)
class ValidationSelectionItem:
    """An explainable selection decision for one command."""

    spec: ValidationCommand
    selected: bool
    reason: str
    matched_paths: tuple[str, ...] = ()
    original_stage: ValidationStage | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.spec.command,
            "selected": self.selected,
            "reason": self.reason,
            "stage": self.spec.stage.label,
            "original_stage": (self.original_stage or self.spec.stage).label,
            "impact_paths": list(self.spec.impact_paths),
            "matched_paths": list(self.matched_paths),
        }


@dataclass(frozen=True)
class ValidationSelection:
    """Impact selection output consumed by :mod:`validation_scheduler`."""

    items: tuple[ValidationSelectionItem, ...]
    changed_files: tuple[str, ...]
    scope: str
    escalated: bool = False
    escalation_reason: str = ""

    @property
    def selected(self) -> tuple[ValidationCommand, ...]:
        return tuple(item.spec for item in self.items if item.selected)

    @property
    def skipped(self) -> tuple[ValidationCommand, ...]:
        return tuple(item.spec for item in self.items if not item.selected)

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "changed_files": list(self.changed_files),
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "selected_count": len(self.selected),
            "skipped_count": len(self.skipped),
            "decisions": [item.to_dict() for item in self.items],
        }


_CHEAP_PATTERNS = (
    re.compile(r"(?:^|[;&|]\s*)git\s+diff\s+--check(?:\s|$)"),
    re.compile(r"\b(?:py_compile|compileall)\b"),
    re.compile(r"\b(?:ruff|flake8|pylint|mypy|pyright)\b"),
    re.compile(r"\bblack\b.*\s--check(?:\s|$)"),
    re.compile(r"\b(?:tsc|eslint)\b"),
    re.compile(r"\b(?:cargo\s+fmt|gofmt|go\s+vet)\b"),
    re.compile(r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?(?:lint|typecheck|format:check)\b"),
    re.compile(r"^\s*test\s+(?:-[defrsx]\s+)?"),
)
_TEST_RUNNER_RE = re.compile(
    r"(?:^|\s)(?:python(?:3)?\s+-m\s+)?(?:pytest|unittest|jest|vitest|mocha|cargo\s+test|go\s+test)(?:\s|$)"
)
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_GLOBAL_IMPACT_NAMES = frozenset(
    {
        ".gitmodules",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "pytest.ini",
        "requirements.txt",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "cargo.toml",
        "cargo.lock",
        "go.mod",
        "go.sum",
    }
)
_DEPENDENCY_SUFFIXES = (".lock", ".lock.json")


def normalize_validation_command_text(value: str) -> str:
    """Return a shell command with markdown-only inline-code wrapping removed."""

    command = str(value or "").strip()
    match = INLINE_CODE_COMMAND_RE.fullmatch(command)
    if match:
        return match.group("command").strip()
    return command


def split_validation_commands(value: str) -> list[str]:
    """Split semicolon-separated shell commands without splitting quoted code."""

    text = normalize_validation_command_text(value)
    commands: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    def flush() -> None:
        command = normalize_validation_command_text("".join(current))
        if command:
            commands.append(command)
        current.clear()

    for char in text:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and not in_single_quote:
            current.append(char)
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current.append(char)
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current.append(char)
            continue
        if char == ";" and not in_single_quote and not in_double_quote:
            flush()
            continue
        current.append(char)

    flush()
    return commands


def _shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return []


def _normalize_path(value: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _looks_like_impact_path(token: str) -> bool:
    value = _normalize_path(token.split("::", 1)[0])
    if not value or value.startswith("-") or value in {".", ".."}:
        return False
    return (
        "/" in value
        or value.startswith(("test_", "tests", "test"))
        or value.endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs"))
    )


def infer_validation_impact_paths(command: str) -> tuple[str, ...]:
    """Extract explicit test/file targets from a shell command.

    An empty result means the command has global or unknown impact and must not
    be omitted by impact selection.
    """

    if not _TEST_RUNNER_RE.search(command):
        return ()
    tokens = _shell_tokens(command)
    impacts: list[str] = []
    after_runner = False
    for token in tokens:
        if _ENV_ASSIGNMENT_RE.match(token) and not after_runner:
            continue
        if token in {"pytest", "unittest", "jest", "vitest", "mocha", "test"}:
            after_runner = True
            continue
        if token == "-m" and not after_runner:
            continue
        if not after_runner:
            continue
        if _looks_like_impact_path(token):
            value = _normalize_path(token.split("::", 1)[0])
            if value and value not in impacts:
                impacts.append(value)
    return tuple(impacts)


def classify_validation_command(
    raw_command: str,
    *,
    ordinal: int = 0,
    environment_keys: Sequence[str] = (),
) -> ValidationCommand:
    """Classify command cost and impact using deliberately conservative rules."""

    command = normalize_validation_command_text(raw_command)
    cheap = any(pattern.search(command) for pattern in _CHEAP_PATTERNS)
    impacts = infer_validation_impact_paths(command)
    if cheap:
        stage = ValidationStage.CHEAP
        resource_cost = 1
    elif impacts:
        stage = ValidationStage.TARGETED
        resource_cost = 1
    else:
        # Unknown, build, integration, and unscoped test commands are broad.
        stage = ValidationStage.BROAD
        resource_cost = 2 if _TEST_RUNNER_RE.search(command) else 1
    return ValidationCommand(
        command=command,
        raw_command=str(raw_command),
        stage=stage,
        resource_cost=resource_cost,
        impact_paths=impacts,
        environment_keys=tuple(sorted({str(key) for key in environment_keys if str(key)})),
        ordinal=int(ordinal),
    )


def build_validation_commands(commands: Iterable[str | ValidationCommand]) -> tuple[ValidationCommand, ...]:
    """Build stable command specs, preserving list order and duplicate commands."""

    result: list[ValidationCommand] = []
    for ordinal, value in enumerate(commands):
        if isinstance(value, ValidationCommand):
            result.append(replace(value, ordinal=ordinal))
        else:
            command = normalize_validation_command_text(str(value))
            if command:
                result.append(classify_validation_command(str(value), ordinal=ordinal))
    return tuple(result)


def is_global_impact_change(path: str) -> bool:
    """Return whether a path can alter validation or dependency behavior globally."""

    normalized = _normalize_path(path).lower()
    name = PurePosixPath(normalized).name
    return (
        not normalized
        or name in _GLOBAL_IMPACT_NAMES
        or name.startswith("requirements") and name.endswith((".txt", ".in"))
        or name.endswith(_DEPENDENCY_SUFFIXES)
        or normalized.startswith((".github/", "ci/", "scripts/ci/"))
    )


def _path_related(changed: str, impact: str) -> bool:
    changed = _normalize_path(changed)
    impact = _normalize_path(impact)
    if not changed or not impact:
        return False
    if changed == impact or changed.startswith(f"{impact.rstrip('/')}/") or impact.startswith(f"{changed.rstrip('/')}/"):
        return True

    changed_path = PurePosixPath(changed)
    impact_path = PurePosixPath(impact)
    changed_stem = changed_path.stem.removeprefix("test_").removesuffix("_test")
    impact_stem = impact_path.stem.removeprefix("test_").removesuffix("_test")
    if changed_stem and changed_stem == impact_stem:
        return True
    if changed_stem and impact_stem and (
        impact_stem.endswith(f"_{changed_stem}")
        or changed_stem.endswith(f"_{impact_stem}")
    ):
        return True

    # Conventional source/test mapping: a test target names the source module.
    if impact_path.name.startswith("test_") and changed_path.stem == impact_stem:
        return True
    return False


def select_validation_commands(
    commands: Iterable[str | ValidationCommand],
    changed_files: Iterable[str] = (),
    *,
    require_full_validation: bool = False,
    scope: str | None = None,
) -> ValidationSelection:
    """Select impacted commands and explain every inclusion or omission.

    Selection is conservative: cheap checks and commands with unknown/global
    impact always run; missing change information and dependency/configuration
    changes select everything.  ``require_full_validation`` promotes otherwise
    unrelated targeted tests into the broad stage.  Supervisors use that mode
    as the final pre-merge gate.
    """

    specs = build_validation_commands(commands)
    changed = tuple(sorted({_normalize_path(path) for path in changed_files if _normalize_path(path)}))
    broad_trigger = not changed or any(is_global_impact_change(path) for path in changed)
    items: list[ValidationSelectionItem] = []
    escalated = False

    for spec in specs:
        original_stage = spec.stage
        matched = tuple(
            path for path in changed if any(_path_related(path, impact) for impact in spec.impact_paths)
        )
        selected = True
        selected_spec = spec
        if spec.stage == ValidationStage.CHEAP:
            reason = "cheap_deterministic_check"
        elif not spec.impact_paths:
            reason = "global_or_unknown_impact"
        elif broad_trigger:
            reason = "conservative_broad_change" if changed else "change_set_unavailable"
        elif matched:
            reason = "changed_path_matches_command_target"
        elif require_full_validation:
            reason = "pre_merge_broad_escalation"
            selected_spec = spec.with_stage(ValidationStage.BROAD)
            escalated = True
        else:
            selected = False
            reason = "no_changed_path_matches_command_target"
        items.append(
            ValidationSelectionItem(
                spec=selected_spec,
                selected=selected,
                reason=reason,
                matched_paths=matched,
                original_stage=original_stage,
            )
        )

    effective_scope = scope or ("pre_merge" if require_full_validation else "impact")
    return ValidationSelection(
        items=tuple(items),
        changed_files=changed,
        scope=effective_scope,
        escalated=escalated,
        escalation_reason="all_declared_validations_required_before_merge" if escalated else "",
    )


# Compatibility-friendly aliases for callers using more explicit nouns.
ValidationCommandSpec = ValidationCommand
select_impacted_validations = select_validation_commands
