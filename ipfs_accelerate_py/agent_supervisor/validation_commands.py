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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from pathlib import PurePosixPath
from typing import Iterable, Sequence

INLINE_CODE_COMMAND_RE = re.compile(r"^`+(?P<command>[^`]+?)`+\s*\.?\s*$", re.DOTALL)


class ValidationStage(IntEnum):
    """Ordered validation barriers.

    Cheap deterministic checks always finish before proof work is admitted.
    Proof phases retain their trust-boundary ordering before focused and broad
    tests, and optional attestation is last.  The historical names remain
    available as aliases for compatibility.
    """

    CHEAP = 0
    DETERMINISTIC = 0
    TRANSLATION = 1
    SOLVER = 2
    KERNEL = 3
    TARGETED = 4
    FOCUSED = 4
    BROAD = 5
    ATTESTATION = 6

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def for_proof_stage(cls, stage: object) -> "ValidationStage":
        """Map a proof-plan stage without importing the proof contracts."""

        value = str(getattr(stage, "value", stage) or "").strip().lower()
        mapping = {
            "translate": cls.TRANSLATION,
            "model_draft": cls.SOLVER,
            "solve": cls.SOLVER,
            "reconstruct": cls.KERNEL,
            "kernel_verify": cls.KERNEL,
            "validate": cls.FOCUSED,
            "attest": cls.ATTESTATION,
        }
        try:
            return mapping[value]
        except KeyError as exc:
            raise ValueError(
                f"unsupported proof validation stage: {value or '<empty>'}"
            ) from exc


class ValidationVerdictKind(str, Enum):
    """Independent verdict channel retained in combined reports."""

    DETERMINISTIC = "deterministic"
    TRANSLATION = "translation"
    SOLVER = "solver"
    KERNEL = "kernel"
    TEST = "test"
    ATTESTATION = "attestation"


class ValidationDecisionKind(str, Enum):
    """Why a selection item appears in an execution plan."""

    INCLUDED = "included"
    OMITTED = "omitted"
    ESCALATED = "escalated"
    FALLBACK = "fallback"


class ValidationRequirementKind(str, Enum):
    """Kind of reviewed fallback validation declared by an obligation."""

    FOCUSED_TEST = "focused_test"
    STATIC_CHECK = "static_check"
    MANUAL_REVIEW = "manual_review"


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
    validation_id: str = ""
    requirement_kind: ValidationRequirementKind | None = None
    verdict_kind: ValidationVerdictKind | None = None
    source: str = "declared"
    fallback: bool = False

    def with_stage(self, stage: ValidationStage) -> "ValidationCommand":
        return replace(self, stage=stage)

    @property
    def effective_verdict_kind(self) -> ValidationVerdictKind:
        """Return the independent report channel for this check."""

        if self.verdict_kind is not None:
            return (
                self.verdict_kind
                if isinstance(self.verdict_kind, ValidationVerdictKind)
                else ValidationVerdictKind(str(self.verdict_kind))
            )
        if self.stage is ValidationStage.CHEAP:
            return ValidationVerdictKind.DETERMINISTIC
        if self.stage is ValidationStage.TRANSLATION:
            return ValidationVerdictKind.TRANSLATION
        if self.stage is ValidationStage.SOLVER:
            return ValidationVerdictKind.SOLVER
        if self.stage is ValidationStage.KERNEL:
            return ValidationVerdictKind.KERNEL
        if self.stage is ValidationStage.ATTESTATION:
            return ValidationVerdictKind.ATTESTATION
        return ValidationVerdictKind.TEST

    @property
    def check_kind(self) -> ValidationVerdictKind:
        """Compatibility spelling for integrations that call checks by kind."""

        return self.effective_verdict_kind


@dataclass(frozen=True)
class DeclaredValidation:
    """A stable fallback declaration and its optional executable command."""

    validation_id: str
    kind: ValidationRequirementKind
    command: ValidationCommand | None = None
    declaration: str = ""
    reason: str = ""

    @property
    def executable(self) -> bool:
        return self.command is not None

    @property
    def manual_review_required(self) -> bool:
        return self.kind is ValidationRequirementKind.MANUAL_REVIEW

    def to_dict(self) -> dict[str, object]:
        return {
            "validation_id": self.validation_id,
            "kind": self.kind.value,
            "declaration": self.declaration or self.validation_id,
            "executable": self.executable,
            "command": self.command.command if self.command is not None else "",
            "stage": self.command.stage.label if self.command is not None else "",
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DeclaredValidation":
        """Restore a declaration without treating its ID as executable text."""

        validation_id = str(payload.get("validation_id") or "").strip()
        if not validation_id:
            raise ValueError("validation_id is required")
        try:
            kind = ValidationRequirementKind(
                str(payload.get("kind") or "").strip().lower()
            )
        except ValueError as exc:
            raise ValueError("unsupported validation requirement kind") from exc
        raw_command = str(payload.get("command") or "").strip()
        command = None
        if raw_command and kind is not ValidationRequirementKind.MANUAL_REVIEW:
            command = classify_validation_command(raw_command)
            command = replace(
                command,
                validation_id=validation_id,
                requirement_kind=kind,
                stage=(
                    ValidationStage.CHEAP
                    if kind is ValidationRequirementKind.STATIC_CHECK
                    else ValidationStage.TARGETED
                ),
            )
        return cls(
            validation_id=validation_id,
            kind=kind,
            command=command,
            declaration=str(payload.get("declaration") or validation_id),
            reason=str(payload.get("reason") or ""),
        )


@dataclass(frozen=True)
class ValidationSelectionItem:
    """An explainable selection decision for one command."""

    spec: ValidationCommand | None
    selected: bool
    reason: str
    matched_paths: tuple[str, ...] = ()
    original_stage: ValidationStage | None = None
    decision_kind: ValidationDecisionKind | None = None
    declaration: DeclaredValidation | None = None

    @property
    def decision(self) -> ValidationDecisionKind:
        if self.decision_kind is not None:
            return (
                self.decision_kind
                if isinstance(self.decision_kind, ValidationDecisionKind)
                else ValidationDecisionKind(str(self.decision_kind))
            )
        if self.declaration is not None or (
            self.spec is not None and self.spec.fallback
        ):
            return ValidationDecisionKind.FALLBACK
        if (
            self.spec is not None
            and self.original_stage is not None
            and self.spec.stage is not self.original_stage
        ):
            return ValidationDecisionKind.ESCALATED
        return (
            ValidationDecisionKind.INCLUDED
            if self.selected
            else ValidationDecisionKind.OMITTED
        )

    def to_dict(self) -> dict[str, object]:
        declaration = self.declaration
        spec = self.spec
        stage = spec.stage.label if spec is not None else ""
        return {
            "command": spec.command if spec is not None else "",
            "validation_id": (
                declaration.validation_id
                if declaration is not None
                else spec.validation_id
                if spec is not None
                else ""
            ),
            "selected": self.selected,
            "decision": self.decision.value,
            "reason": self.reason,
            "stage": stage,
            "original_stage": (
                self.original_stage.label
                if self.original_stage is not None
                else stage
            ),
            "verdict_kind": (
                spec.effective_verdict_kind.value if spec is not None else ""
            ),
            "source": (
                "fallback"
                if declaration is not None
                else spec.source
                if spec is not None
                else ""
            ),
            "fallback": bool(
                declaration is not None or (spec is not None and spec.fallback)
            ),
            "executable": spec is not None,
            "requirement_kind": (
                declaration.kind.value
                if declaration is not None
                else spec.requirement_kind.value
                if spec is not None and spec.requirement_kind is not None
                else ""
            ),
            "impact_paths": list(spec.impact_paths if spec is not None else ()),
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
        result: list[ValidationCommand] = []
        seen: set[tuple[str, ValidationStage]] = set()
        for item in self.items:
            if not item.selected or item.spec is None:
                continue
            identity = (item.spec.command, item.spec.stage)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(item.spec)
        return tuple(result)

    @property
    def skipped(self) -> tuple[ValidationCommand, ...]:
        return tuple(
            item.spec
            for item in self.items
            if not item.selected and item.spec is not None
        )

    @property
    def fallback_items(self) -> tuple[ValidationSelectionItem, ...]:
        return tuple(
            item
            for item in self.items
            if item.decision is ValidationDecisionKind.FALLBACK
        )

    @property
    def unresolved_fallbacks(self) -> tuple[DeclaredValidation, ...]:
        return tuple(
            item.declaration
            for item in self.fallback_items
            if item.declaration is not None and item.spec is None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "changed_files": list(self.changed_files),
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "selected_count": len(self.selected),
            "skipped_count": len(self.skipped),
            "fallback_count": len(self.fallback_items),
            "unresolved_fallback_count": len(self.unresolved_fallbacks),
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
_DECLARATION_PREFIXES = {
    "pytest": ValidationRequirementKind.FOCUSED_TEST,
    "test": ValidationRequirementKind.FOCUSED_TEST,
    "unittest": ValidationRequirementKind.FOCUSED_TEST,
    "focused-test": ValidationRequirementKind.FOCUSED_TEST,
    "focused_test": ValidationRequirementKind.FOCUSED_TEST,
    "static": ValidationRequirementKind.STATIC_CHECK,
    "lint": ValidationRequirementKind.STATIC_CHECK,
    "typecheck": ValidationRequirementKind.STATIC_CHECK,
    "manual": ValidationRequirementKind.MANUAL_REVIEW,
    "review": ValidationRequirementKind.MANUAL_REVIEW,
    "manual-review": ValidationRequirementKind.MANUAL_REVIEW,
    "manual_review": ValidationRequirementKind.MANUAL_REVIEW,
}


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


def parse_validation_declaration(
    value: str | ValidationCommand | DeclaredValidation,
    *,
    command_catalog: Mapping[str, str | ValidationCommand] | None = None,
) -> DeclaredValidation:
    """Parse a reviewed fallback declaration without guessing a shell command.

    Named declarations such as ``pytest:lease-fence`` are resolved only
    through ``command_catalog``.  This keeps provider/template text from
    becoming executable shell input.  Unknown declarations conservatively
    require manual review.
    """

    if isinstance(value, DeclaredValidation):
        return value
    if isinstance(value, ValidationCommand):
        kind = value.requirement_kind or (
            ValidationRequirementKind.STATIC_CHECK
            if value.stage is ValidationStage.CHEAP
            else ValidationRequirementKind.FOCUSED_TEST
        )
        validation_id = value.validation_id or value.command
        return DeclaredValidation(
            validation_id=validation_id,
            kind=kind,
            command=replace(
                value,
                validation_id=validation_id,
                requirement_kind=kind,
            ),
            declaration=validation_id,
            reason="executable_validation_command",
        )

    declaration = normalize_validation_command_text(str(value))
    if not declaration:
        raise ValueError("validation declaration must not be empty")
    prefix, separator, suffix = declaration.partition(":")
    declared_kind = (
        _DECLARATION_PREFIXES.get(prefix.strip().lower()) if separator else None
    )
    if separator and not suffix.strip():
        raise ValueError("validation declaration suffix must not be empty")

    resolved: str | ValidationCommand | None = None
    if command_catalog is not None:
        resolved = command_catalog.get(declaration)
    command: ValidationCommand | None = None
    if resolved is not None:
        command = (
            resolved
            if isinstance(resolved, ValidationCommand)
            else classify_validation_command(str(resolved))
        )
    elif declared_kind is None and (
        _TEST_RUNNER_RE.search(declaration)
        or any(pattern.search(declaration) for pattern in _CHEAP_PATTERNS)
    ):
        # Direct commands remain supported for todo-board compatibility.
        command = classify_validation_command(declaration)
        declared_kind = (
            ValidationRequirementKind.STATIC_CHECK
            if command.stage is ValidationStage.CHEAP
            else ValidationRequirementKind.FOCUSED_TEST
        )

    if declared_kind is None:
        declared_kind = ValidationRequirementKind.MANUAL_REVIEW
        reason = "unknown_validation_declaration_requires_manual_review"
    elif declared_kind is ValidationRequirementKind.MANUAL_REVIEW:
        # A catalog cannot turn an explicit human decision into automation.
        command = None
        reason = "declared_manual_review"
    elif command is None:
        reason = "declared_validation_requires_catalog_resolution"
    else:
        reason = "declared_validation_resolved"

    if command is not None:
        command = replace(
            command,
            validation_id=declaration,
            requirement_kind=declared_kind,
            stage=(
                ValidationStage.CHEAP
                if declared_kind is ValidationRequirementKind.STATIC_CHECK
                else ValidationStage.TARGETED
            ),
        )
    return DeclaredValidation(
        validation_id=declaration,
        kind=declared_kind,
        command=command,
        declaration=declaration,
        reason=reason,
    )


def build_declared_validations(
    declarations: Iterable[str | ValidationCommand | DeclaredValidation],
    *,
    command_catalog: Mapping[str, str | ValidationCommand] | None = None,
) -> tuple[DeclaredValidation, ...]:
    """Return stable declarations, de-duplicated by validation identity."""

    result: list[DeclaredValidation] = []
    seen: set[str] = set()
    for value in declarations:
        declaration = parse_validation_declaration(
            value, command_catalog=command_catalog
        )
        if declaration.validation_id in seen:
            continue
        seen.add(declaration.validation_id)
        result.append(declaration)
    return tuple(result)


def build_focused_validation_commands(
    declarations: Iterable[str | ValidationCommand | DeclaredValidation],
    *,
    command_catalog: Mapping[str, str | ValidationCommand] | None = None,
) -> tuple[ValidationCommand, ...]:
    """Resolve executable focused/static declarations for the scheduler."""

    commands = tuple(
        declaration.command
        for declaration in build_declared_validations(
            declarations, command_catalog=command_catalog
        )
        if declaration.command is not None
    )
    return tuple(
        replace(command, ordinal=index) for index, command in enumerate(commands)
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
    fallback_validations: Iterable[
        str | ValidationCommand | DeclaredValidation
    ] = (),
    command_catalog: Mapping[str, str | ValidationCommand] | None = None,
) -> ValidationSelection:
    """Select impacted commands and explain every inclusion or omission.

    Selection is conservative: cheap checks and commands with unknown/global
    impact always run; missing change information and dependency/configuration
    changes select everything.  ``require_full_validation`` promotes otherwise
    unrelated targeted tests into the broad stage.  Supervisors use that mode
    as the final pre-merge gate.

    Proof fallbacks are explicit reviewed declarations.  Executable fallback
    checks are always selected because the proof outcome, rather than path
    inference, established their relevance.  Manual-review and unresolved
    declarations remain non-executable selection items so reports cannot
    silently lose a required fallback or mistake its identifier for shell text.
    """

    specs = build_validation_commands(commands)
    fallback_declarations = build_declared_validations(
        fallback_validations,
        command_catalog=command_catalog,
    )
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
            # Tests are promoted to the broad barrier.  A proof-phase check
            # retains its trust-boundary stage even when full validation makes
            # an otherwise unrelated check mandatory.
            if spec.stage is ValidationStage.TARGETED:
                selected_spec = spec.with_stage(ValidationStage.BROAD)
            escalated = True
        else:
            selected = False
            reason = "no_changed_path_matches_command_target"
        decision_kind = (
            ValidationDecisionKind.ESCALATED
            if reason == "pre_merge_broad_escalation"
            else ValidationDecisionKind.INCLUDED
            if selected
            else ValidationDecisionKind.OMITTED
        )
        items.append(
            ValidationSelectionItem(
                spec=selected_spec,
                selected=selected,
                reason=reason,
                matched_paths=matched,
                original_stage=original_stage,
                decision_kind=decision_kind,
            )
        )

    existing = {
        (item.spec.validation_id or item.spec.command, item.spec.command)
        for item in items
        if item.spec is not None and item.selected
    }
    next_ordinal = len(specs)
    for declaration in fallback_declarations:
        command = declaration.command
        if command is None:
            items.append(
                ValidationSelectionItem(
                    spec=None,
                    selected=False,
                    reason=declaration.reason
                    or (
                        "declared_manual_review"
                        if declaration.manual_review_required
                        else "declared_validation_requires_catalog_resolution"
                    ),
                    decision_kind=ValidationDecisionKind.FALLBACK,
                    declaration=declaration,
                )
            )
            continue

        fallback_spec = replace(
            command,
            ordinal=next_ordinal,
            validation_id=declaration.validation_id,
            requirement_kind=declaration.kind,
            source="fallback",
            fallback=True,
            verdict_kind=(
                ValidationVerdictKind.DETERMINISTIC
                if declaration.kind is ValidationRequirementKind.STATIC_CHECK
                else ValidationVerdictKind.TEST
            ),
        )
        next_ordinal += 1
        identity = (
            fallback_spec.validation_id or fallback_spec.command,
            fallback_spec.command,
        )
        duplicate = identity in existing or any(
            item.spec is not None
            and item.spec.command == fallback_spec.command
            and item.selected
            for item in items
        )
        existing.add(identity)
        items.append(
            ValidationSelectionItem(
                spec=fallback_spec,
                selected=True,
                reason=(
                    "proof_fallback_required_existing_command"
                    if duplicate
                    else "proof_fallback_required"
                ),
                original_stage=fallback_spec.stage,
                decision_kind=ValidationDecisionKind.FALLBACK,
                declaration=declaration,
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
FallbackValidationKind = ValidationRequirementKind
ValidationDeclaration = DeclaredValidation
ValidationPhase = ValidationStage
ValidationCheckKind = ValidationVerdictKind
ValidationSelectionDecision = ValidationDecisionKind
