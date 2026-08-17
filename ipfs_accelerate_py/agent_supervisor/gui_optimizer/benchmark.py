"""Controlled 15-task GUI optimizer benchmark catalog (VGO-083).

Interfaces owned by this module:

* ``GuiOptimizationBenchmark@1`` — sealed catalog of exactly 15 tasks
* ``GuiBenchmarkTask@1`` — one bounded, uniquely identified improvement task
* ``GuiBenchmarkResult@1`` — terminal result record for one executed task

The catalog benchmarks only the selected Agent Supervisor screen and its
controlled fixture variants.  It never asks a provider to make the app
generally better and never auto-approves subjective redesigns.

Catalog construction is pure and byte-identical across repeated builds.
Fixtures are inert: no production credentials, services, user data, or
effectful host commands.  External provers are not invoked and are not
claimed available.

Fail-closed invariants:

* any count other than 15 is rejected;
* duplicate task IDs and duplicate kinds are rejected;
* every required kind is present exactly once;
* every task has one or two measurable objectives, bounded files,
  controlled fixtures, hard gates, an expected decision, a route, and
  an evidence class;
* bounded files stay under optimizer-allowed roots;
* primary-action hierarchy cannot be automatically accepted.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .authority import (
    AuthorityReasonCode,
    DEFAULT_ALLOWED_ROOTS,
    GuiAuthorityError,
    _normalize_repo_path,
    path_has_forbidden_segment,
    path_under_allowed_roots,
)
from .check_plan import REGISTERED_CHECK_IDS

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

GUI_OPTIMIZATION_BENCHMARK_INTERFACE: Final[str] = "GuiOptimizationBenchmark@1"
GUI_BENCHMARK_TASK_INTERFACE: Final[str] = "GuiBenchmarkTask@1"
GUI_BENCHMARK_RESULT_INTERFACE: Final[str] = "GuiBenchmarkResult@1"

GUI_OPTIMIZATION_BENCHMARK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/optimization-benchmark@1"
)
GUI_BENCHMARK_TASK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/benchmark-task@1"
)
GUI_BENCHMARK_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/benchmark-result@1"
)

GUI_BENCHMARK_CATALOG_VERSION: Final[str] = "gui-optimizer-benchmark-catalog@1.0.0"
CANONICAL_JSON_PROFILE: Final[str] = "gui-optimizer-canonical-json/v1"

EXPECTED_TASK_COUNT: Final[int] = 15
MAX_OBJECTIVES_PER_TASK: Final[int] = 2
BENCHMARK_ID: Final[str] = "benchmark-v1"
CATALOG_ID: Final[str] = "catalog:gui-optimizer-benchmark-v1"
DEFAULT_APPLICATION_ID: Final[str] = "app:agent-supervisor"
DEFAULT_SCREEN_ID: Final[str] = "screen:agent-supervisor"
DEFAULT_ROUTE_ID: Final[str] = "route:agent-supervisor"
DEFAULT_SOURCE_PATH: Final[str] = "swissknife/web/js/apps/agent-supervisor.js"
DEFAULT_CATALOG_RELATIVE_PATH: Final[str] = (
    "external/ipfs_accelerate/test/fixtures/gui_optimizer/benchmark-tasks.json"
)
PACKAGE_CATALOG_RELATIVE_PATH: Final[str] = (
    "test/fixtures/gui_optimizer/benchmark-tasks.json"
)

CONFLICT_POLICY: Final[str] = (
    "Benchmark only the selected screen and controlled variants; do not ask "
    "a provider to make the app generally better or auto-approve subjective "
    "redesigns."
)

_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/#@-]{0,255}$")
_REWRITE_RE: Final = re.compile(
    r"(?i)\b(whole[- ]app|entire application|redesign the (?:app|application)|"
    r"aesthetic rewrite|rewrite everything)\b"
)
_VENDOR_TOKEN_RE: Final = re.compile(
    r"(?i)(?<![a-z0-9])("
    r"openai|anthropic|claude|gpt-?[0-9]|gpt4|chatgpt|grok|xai|"
    r"gemini|mistral|llama|copilot|bedrock|vertexai|vertex-ai|"
    r"togetherai|ollama|cohere"
    r")(?![a-z0-9])"
)

COMPONENT_CONSOLE_ROOT: Final[str] = "comp:console-root"
COMPONENT_GOAL_FORM: Final[str] = "comp:goal-form"
REGISTERED_COMPONENT_IDS: Final[frozenset[str]] = frozenset(
    {COMPONENT_CONSOLE_ROOT, COMPONENT_GOAL_FORM}
)

STABLE_SCENARIO_IDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "initial_load": "scenario:initial-load",
        "loading": "scenario:loading",
        "success": "scenario:success",
        "empty": "scenario:empty",
        "recoverable_failure": "scenario:recoverable-failure",
        "unrecoverable_failure": "scenario:unrecoverable-failure",
        "invalid_submission": "scenario:invalid-submission",
        "valid_submission": "scenario:valid-submission",
        "keyboard_only": "scenario:keyboard-only",
        "viewport_mobile": "scenario:viewport-mobile",
        "viewport_desktop": "scenario:viewport-desktop",
        "viewport_wide": "scenario:viewport-wide",
        "text_scale_200": "scenario:text-scale-200",
        "reduced_motion": "scenario:reduced-motion",
        "dark_mode": "scenario:dark-mode",
        "service_unavailable": "scenario:service-unavailable",
        "confirmation_grant": "scenario:confirmation-grant",
        "confirmation_deny": "scenario:confirmation-deny",
    }
)
REGISTERED_SCENARIO_IDS: Final[frozenset[str]] = frozenset(
    STABLE_SCENARIO_IDS.values()
)

FIXTURE_HOST_ID: Final[str] = "fixture:agent-supervisor-host"
FIXTURE_SERVICES_ID: Final[str] = "fixture:agent-supervisor-services"
FIXTURE_SCENARIOS_ID: Final[str] = "fixture:agent-supervisor-scenarios"
FIXTURE_HOST_PATH: Final[str] = (
    "swissknife/test/fixtures/gui-optimizer/agent-supervisor/fixture-host.html"
)
FIXTURE_SERVICES_PATH: Final[str] = (
    "swissknife/test/fixtures/gui-optimizer/agent-supervisor/fixture-services.js"
)
FIXTURE_SCENARIOS_PATH: Final[str] = (
    "swissknife/test/fixtures/gui-optimizer/agent-supervisor/fixture-scenarios.json"
)

GATE_NO_A11Y_REGRESSION: Final[str] = "gate:no-critical-accessibility-regression"
GATE_NO_AUTHORIZATION_REGRESSION: Final[str] = "gate:no-authorization-regression"
GATE_NO_CONFIRMATION_REGRESSION: Final[str] = "gate:no-confirmation-regression"
GATE_NO_HIDDEN_DISPATCH: Final[str] = "gate:no-hidden-dispatch"
GATE_SCOPE_SELECTED_SCREEN: Final[str] = "gate:scope-selected-screen-only"
GATE_NO_WHOLE_APP_REWRITE: Final[str] = "gate:no-whole-app-rewrite"
GATE_MEASURABLE_IMPROVEMENT: Final[str] = "gate:measurable-objective-improvement"
GATE_ISOLATED_WORKTREE: Final[str] = "gate:isolated-worktree-only"

COMMON_HARD_GATES: Final[tuple[str, ...]] = (
    GATE_NO_A11Y_REGRESSION,
    GATE_NO_AUTHORIZATION_REGRESSION,
    GATE_NO_CONFIRMATION_REGRESSION,
    GATE_SCOPE_SELECTED_SCREEN,
    GATE_NO_WHOLE_APP_REWRITE,
    GATE_MEASURABLE_IMPROVEMENT,
    GATE_ISOLATED_WORKTREE,
)


class BenchmarkTaskKind(str, Enum):
    """Closed VGO-083 task kinds.  Order is the sealed catalog order."""

    FOCUS_RESTORATION = "focus_restoration"
    ACCESSIBLE_LABELS = "accessible_labels"
    ERROR_PRESENTATION = "error_presentation"
    LOADING_STATE = "loading_state"
    FAILURE_STATE = "failure_state"
    INTERACTION_STEP_REDUCTION = "interaction_step_reduction"
    RESPONSIVE_OVERFLOW = "responsive_overflow"
    PRIMARY_ACTION_HIERARCHY = "primary_action_hierarchy"
    DESIGN_TOKEN_CONSISTENCY = "design_token_consistency"
    CONFIRMATION_UX = "confirmation_ux"
    EMPTY_STATE_GUIDANCE = "empty_state_guidance"
    KEYBOARD_REACHABILITY = "keyboard_reachability"
    LOCALIZATION_CLIPPING = "localization_clipping"
    MODAL_FOCUS_LIFECYCLE = "modal_focus_lifecycle"
    ACTION_BINDING_INTEGRITY = "action_binding_integrity"


REQUIRED_TASK_KINDS: Final[tuple[str, ...]] = tuple(
    item.value for item in BenchmarkTaskKind
)


class BenchmarkRouteKind(str, Enum):
    """Caller-declared proposal route.  This module does not choose vendors."""

    DETERMINISTIC_TRANSFORM = "deterministic_transform"
    SMALL_LOCAL_MODEL = "small_local_model"
    MEDIUM_MODEL = "medium_model"
    FRONTIER_MODEL = "frontier_model"
    HUMAN_REVIEW = "human_review"


class BenchmarkDecision(str, Enum):
    """Expected or observed terminal decision for a benchmark task."""

    ACCEPT = "accept"
    REJECT = "reject"
    HUMAN_REVIEW = "human_review"
    PENDING = "pending"


class BenchmarkEvidenceClass(str, Enum):
    """Authority label for the task's expected evidence class."""

    AUTOMATED = "automated"
    STRUCTURAL = "structural"
    INTEGRITY = "integrity"
    HEURISTIC = "heuristic"
    HUMAN_REVIEWED = "human_reviewed"
    SIMULATED = "simulated"


class BenchmarkTier(str, Enum):
    DETERMINISTIC = "deterministic"
    SMALL_LOCAL = "small_local"
    MEDIUM = "medium"
    FRONTIER = "frontier"
    HUMAN = "human"


ROUTE_DEFAULT_TIER: Final[Mapping[str, str]] = MappingProxyType(
    {
        BenchmarkRouteKind.DETERMINISTIC_TRANSFORM.value: (
            BenchmarkTier.DETERMINISTIC.value
        ),
        BenchmarkRouteKind.SMALL_LOCAL_MODEL.value: BenchmarkTier.SMALL_LOCAL.value,
        BenchmarkRouteKind.MEDIUM_MODEL.value: BenchmarkTier.MEDIUM.value,
        BenchmarkRouteKind.FRONTIER_MODEL.value: BenchmarkTier.FRONTIER.value,
        BenchmarkRouteKind.HUMAN_REVIEW.value: BenchmarkTier.HUMAN.value,
    }
)

SUBJECTIVE_KINDS: Final[frozenset[str]] = frozenset(
    {BenchmarkTaskKind.PRIMARY_ACTION_HIERARCHY.value}
)


class BenchmarkReasonCode(str, Enum):
    OK = "ok"
    TASK_COUNT_MISMATCH = "benchmark_task_count_mismatch"
    DUPLICATE_TASK_ID = "duplicate_task_id"
    DUPLICATE_TASK_KIND = "duplicate_task_kind"
    MISSING_REQUIRED_KIND = "missing_required_kind"
    INVALID_TASK_INPUT = "invalid_task_input"
    INVALID_RESULT_INPUT = "invalid_result_input"
    UNKNOWN_FIELD = AuthorityReasonCode.UNKNOWN_FIELD.value
    INVALID_COLLECTION_TYPE = AuthorityReasonCode.INVALID_COLLECTION_TYPE.value
    PATH_OUTSIDE_ALLOWED_ROOTS = (
        AuthorityReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value
    )
    PATH_ABSOLUTE_OR_TRAVERSAL = (
        AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
    )
    PATH_FORBIDDEN_SEGMENT = AuthorityReasonCode.PATH_FORBIDDEN_SEGMENT.value
    TOO_MANY_OBJECTIVES = "too_many_objectives"
    WHOLE_APP_REWRITE = "whole_app_rewrite"
    SUBJECTIVE_AUTO_ACCEPT_FORBIDDEN = "subjective_auto_accept_forbidden"
    VENDOR_FORBIDDEN = "vendor_forbidden"
    CATALOG_UNAVAILABLE = "benchmark_catalog_unavailable"
    UNKNOWN_TASK_ID = "unknown_task_id"
    UNKNOWN_CHECK_ID = "unknown_check_id"
    UNKNOWN_COMPONENT_ID = "unknown_component_id"
    UNKNOWN_SCENARIO_ID = "unknown_scenario_id"
    MISSING_HARD_GATE = "missing_hard_gate"
    PRODUCTION_SURFACE_FORBIDDEN = "production_surface_forbidden"


_TASK_KEYS: Final[frozenset[str]] = frozenset(
    {
        "affected_check_ids",
        "affected_component_ids",
        "affected_scenario_ids",
        "application_id",
        "baseline_id",
        "bounded_file_paths",
        "conflict_policy",
        "controlled_fixture_ids",
        "declared_method",
        "declared_tier",
        "evidence_class",
        "expected_decision",
        "expected_route",
        "hard_gate_ids",
        "interface",
        "kind",
        "objective_ids",
        "objective_metric_ids",
        "raw_retrieval_token_estimate",
        "reference_id",
        "route_id",
        "schema_version",
        "screen_id",
        "task_id",
        "title",
    }
)
_BENCHMARK_KEYS: Final[frozenset[str]] = frozenset(
    {
        "application_id",
        "benchmark_id",
        "catalog_id",
        "conflict_policy",
        "expected_task_count",
        "interface",
        "schema_version",
        "screen_id",
        "tasks",
        "uses_production_credentials",
        "uses_production_services",
    }
)
_RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "artifact_ids",
        "benchmark_id",
        "decision",
        "declared_method",
        "declared_tier",
        "hard_gate_passed",
        "interface",
        "measurable_improvement",
        "metric_values",
        "reason_codes",
        "receipt_id",
        "result_id",
        "route_kind",
        "schema_version",
        "task_id",
    }
)


class GuiBenchmarkError(ValueError):
    """Malformed or out-of-contract benchmark catalog input."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = BenchmarkReasonCode.INVALID_TASK_INPUT.value,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


def _reject_unknown(payload: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise GuiBenchmarkError(
            f"unknown {label} field(s): {', '.join(unknown)}",
            reason_code=BenchmarkReasonCode.UNKNOWN_FIELD.value,
            details={"fields": unknown, "record": label},
        )


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise GuiBenchmarkError(
            f"{name} must be a string",
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = _exact_str(value, name)
    if "\x00" in text:
        raise GuiBenchmarkError(f"{name} must not contain NUL", details={"field": name})
    stripped = text.strip()
    if required and not stripped:
        raise GuiBenchmarkError(f"{name} must not be empty", details={"field": name})
    return stripped


def _identifier(value: Any, name: str) -> str:
    text = _exact_str(value, name)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise GuiBenchmarkError(
            f"{name} is not a stable identifier",
            details={"field": name, "value": text},
        )
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise GuiBenchmarkError(
            f"{name} must be a boolean",
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or type(value) is bool:
        raise GuiBenchmarkError(
            f"{name} must be an integer",
            details={"field": name, "value_type": type(value).__name__},
        )
    if minimum is not None and value < minimum:
        raise GuiBenchmarkError(
            f"{name} must be >= {minimum}",
            details={"field": name, "value": value},
        )
    return value


def _finite_number(value: Any, name: str) -> int | float:
    if type(value) is bool or type(value) not in (int, float):
        raise GuiBenchmarkError(
            f"{name} must be a finite number",
            details={"field": name, "value_type": type(value).__name__},
        )
    if type(value) is float and value != value:  # NaN
        raise GuiBenchmarkError(f"{name} must be finite", details={"field": name})
    if type(value) is float and value in (float("inf"), float("-inf")):
        raise GuiBenchmarkError(f"{name} must be finite", details={"field": name})
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GuiBenchmarkError(
            f"{name} must be a JSON object",
            reason_code=BenchmarkReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _require_json_array(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise GuiBenchmarkError(
            f"{name} must be a JSON array",
            reason_code=BenchmarkReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _identifiers(value: Any, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    items = _require_json_array(value, name)
    parsed = tuple(_identifier(item, f"{name}[{index}]") for index, item in enumerate(items))
    if nonempty and not parsed:
        raise GuiBenchmarkError(f"{name} must not be empty", details={"field": name})
    if len(set(parsed)) != len(parsed):
        raise GuiBenchmarkError(
            f"{name} must not contain duplicates",
            reason_code=BenchmarkReasonCode.DUPLICATE_TASK_ID.value,
            details={"field": name},
        )
    return parsed


def _closed_enum(value: Any, enum_cls: type[Enum], name: str) -> str:
    text = _text(value, name)
    try:
        return enum_cls(text).value
    except ValueError as exc:
        raise GuiBenchmarkError(
            f"unknown {name}: {text}",
            details={"field": name, "value": text},
        ) from exc


def _reject_vendor(value: str, name: str) -> str:
    if _VENDOR_TOKEN_RE.search(value):
        raise GuiBenchmarkError(
            f"{name} must not name a vendor",
            reason_code=BenchmarkReasonCode.VENDOR_FORBIDDEN.value,
            details={"field": name},
        )
    return value


def _reject_rewrite(value: str, name: str) -> str:
    if _REWRITE_RE.search(value):
        raise GuiBenchmarkError(
            f"{name} must not request a whole-app rewrite",
            reason_code=BenchmarkReasonCode.WHOLE_APP_REWRITE.value,
            details={"field": name},
        )
    return value


def _repo_path(value: Any, name: str) -> str:
    try:
        path = _normalize_repo_path(_text(value, name), name)
    except GuiAuthorityError as exc:
        raise GuiBenchmarkError(
            str(exc),
            reason_code=getattr(
                exc,
                "reason_code",
                BenchmarkReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
            ),
            details=getattr(exc, "details", {"field": name}),
        ) from exc
    if path_has_forbidden_segment(path):
        raise GuiBenchmarkError(
            f"{name} contains a forbidden path segment",
            reason_code=BenchmarkReasonCode.PATH_FORBIDDEN_SEGMENT.value,
            details={"field": name, "path": path},
        )
    if not path_under_allowed_roots(path, allowed_roots=DEFAULT_ALLOWED_ROOTS):
        raise GuiBenchmarkError(
            f"{name} is outside allowed optimizer roots",
            reason_code=BenchmarkReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value,
            details={"field": name, "path": path},
        )
    return path


def _bounded_paths(value: Any, name: str) -> tuple[str, ...]:
    items = _require_json_array(value, name)
    parsed = tuple(_repo_path(item, f"{name}[{index}]") for index, item in enumerate(items))
    if not parsed:
        raise GuiBenchmarkError(f"{name} must not be empty", details={"field": name})
    if len(set(parsed)) != len(parsed):
        raise GuiBenchmarkError(
            f"{name} must not contain duplicates",
            details={"field": name},
        )
    return parsed


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _digest_id(prefix: str, payload: Mapping[str, Any] | bytes) -> str:
    digest = hashlib.sha256(
        payload if isinstance(payload, bytes) else _canonical_bytes(payload)
    ).hexdigest()
    return f"{prefix}:{digest[:24]}"


def _require_interface(payload: Mapping[str, Any], expected: str, schema: str) -> None:
    interface = _text(payload.get("interface", ""), "interface")
    if interface != expected:
        raise GuiBenchmarkError(
            f"interface must be {expected}",
            details={"interface": interface},
        )
    schema_version = _text(payload.get("schema_version", ""), "schema_version")
    if schema_version != schema:
        raise GuiBenchmarkError(
            f"schema_version must be {schema}",
            details={"schema_version": schema_version},
        )


# ---------------------------------------------------------------------------
# Wire models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiBenchmarkTask:
    """One bounded ``GuiBenchmarkTask@1`` catalog entry."""

    task_id: str
    kind: str
    title: str
    objective_ids: tuple[str, ...]
    objective_metric_ids: tuple[str, ...]
    application_id: str = DEFAULT_APPLICATION_ID
    screen_id: str = DEFAULT_SCREEN_ID
    route_id: str = DEFAULT_ROUTE_ID
    expected_route: str = BenchmarkRouteKind.DETERMINISTIC_TRANSFORM.value
    declared_method: str = "exact_label_substitution"
    declared_tier: str = BenchmarkTier.DETERMINISTIC.value
    expected_decision: str = BenchmarkDecision.ACCEPT.value
    evidence_class: str = BenchmarkEvidenceClass.STRUCTURAL.value
    baseline_id: str = ""
    reference_id: str = ""
    bounded_file_paths: tuple[str, ...] = (DEFAULT_SOURCE_PATH,)
    controlled_fixture_ids: tuple[str, ...] = (FIXTURE_HOST_ID,)
    hard_gate_ids: tuple[str, ...] = COMMON_HARD_GATES
    affected_component_ids: tuple[str, ...] = (COMPONENT_GOAL_FORM,)
    affected_scenario_ids: tuple[str, ...] = (
        STABLE_SCENARIO_IDS["keyboard_only"],
    )
    affected_check_ids: tuple[str, ...] = ("check:direct-tests",)
    raw_retrieval_token_estimate: int = 1800
    conflict_policy: str = CONFLICT_POLICY
    interface: str = GUI_BENCHMARK_TASK_INTERFACE
    schema_version: str = GUI_BENCHMARK_TASK_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(
            self, "kind", _closed_enum(self.kind, BenchmarkTaskKind, "kind")
        )
        title = _reject_rewrite(_reject_vendor(_text(self.title, "title"), "title"), "title")
        object.__setattr__(self, "title", title)
        objectives = tuple(
            _identifier(item, f"objective_ids[{index}]")
            for index, item in enumerate(self.objective_ids)
        )
        if not objectives:
            raise GuiBenchmarkError(
                "objective_ids must not be empty", details={"field": "objective_ids"}
            )
        if len(objectives) > MAX_OBJECTIVES_PER_TASK:
            raise GuiBenchmarkError(
                "each task may declare at most two objectives",
                reason_code=BenchmarkReasonCode.TOO_MANY_OBJECTIVES.value,
                details={"count": len(objectives)},
            )
        if len(set(objectives)) != len(objectives):
            raise GuiBenchmarkError(
                "objective_ids must be unique",
                details={"field": "objective_ids"},
            )
        object.__setattr__(self, "objective_ids", objectives)
        metrics = tuple(
            _identifier(item, f"objective_metric_ids[{index}]")
            for index, item in enumerate(self.objective_metric_ids)
        )
        if not metrics or len(metrics) > MAX_OBJECTIVES_PER_TASK:
            raise GuiBenchmarkError(
                "objective_metric_ids must contain one or two identifiers",
                details={"field": "objective_metric_ids"},
            )
        object.__setattr__(self, "objective_metric_ids", metrics)
        object.__setattr__(
            self, "application_id", _identifier(self.application_id, "application_id")
        )
        object.__setattr__(self, "screen_id", _identifier(self.screen_id, "screen_id"))
        object.__setattr__(self, "route_id", _identifier(self.route_id, "route_id"))
        if self.application_id != DEFAULT_APPLICATION_ID:
            raise GuiBenchmarkError(
                "benchmark tasks may target only app:agent-supervisor",
                details={"application_id": self.application_id},
            )
        if self.screen_id != DEFAULT_SCREEN_ID:
            raise GuiBenchmarkError(
                "benchmark tasks may target only screen:agent-supervisor",
                details={"screen_id": self.screen_id},
            )
        if self.route_id != DEFAULT_ROUTE_ID:
            raise GuiBenchmarkError(
                "benchmark tasks may target only route:agent-supervisor",
                details={"route_id": self.route_id},
            )
        route = _closed_enum(self.expected_route, BenchmarkRouteKind, "expected_route")
        object.__setattr__(self, "expected_route", route)
        method = _reject_vendor(
            _identifier(self.declared_method, "declared_method"), "declared_method"
        )
        object.__setattr__(self, "declared_method", method)
        tier = _closed_enum(self.declared_tier, BenchmarkTier, "declared_tier")
        expected_tier = ROUTE_DEFAULT_TIER[route]
        if tier != expected_tier:
            raise GuiBenchmarkError(
                "declared_tier must match the expected route",
                details={"declared_tier": tier, "expected_route": route},
            )
        object.__setattr__(self, "declared_tier", tier)
        decision = _closed_enum(
            self.expected_decision, BenchmarkDecision, "expected_decision"
        )
        if decision == BenchmarkDecision.PENDING.value:
            raise GuiBenchmarkError(
                "catalog tasks must declare a terminal expected decision",
                details={"expected_decision": decision},
            )
        if (
            self.kind in SUBJECTIVE_KINDS
            and decision == BenchmarkDecision.ACCEPT.value
        ):
            raise GuiBenchmarkError(
                "subjective tasks cannot be automatically accepted",
                reason_code=BenchmarkReasonCode.SUBJECTIVE_AUTO_ACCEPT_FORBIDDEN.value,
                details={"kind": self.kind},
            )
        if (
            self.kind in SUBJECTIVE_KINDS
            and route != BenchmarkRouteKind.HUMAN_REVIEW.value
        ):
            raise GuiBenchmarkError(
                "subjective tasks must use the human_review route",
                reason_code=BenchmarkReasonCode.SUBJECTIVE_AUTO_ACCEPT_FORBIDDEN.value,
                details={"kind": self.kind, "expected_route": route},
            )
        object.__setattr__(self, "expected_decision", decision)
        object.__setattr__(
            self,
            "evidence_class",
            _closed_enum(self.evidence_class, BenchmarkEvidenceClass, "evidence_class"),
        )
        object.__setattr__(self, "baseline_id", _identifier(self.baseline_id, "baseline_id"))
        object.__setattr__(
            self, "reference_id", _identifier(self.reference_id, "reference_id")
        )
        paths = tuple(
            _repo_path(item, f"bounded_file_paths[{index}]")
            for index, item in enumerate(self.bounded_file_paths)
        )
        if not paths:
            raise GuiBenchmarkError(
                "bounded_file_paths must not be empty",
                details={"field": "bounded_file_paths"},
            )
        object.__setattr__(self, "bounded_file_paths", paths)
        fixtures = tuple(
            _identifier(item, f"controlled_fixture_ids[{index}]")
            for index, item in enumerate(self.controlled_fixture_ids)
        )
        if not fixtures:
            raise GuiBenchmarkError(
                "controlled_fixture_ids must not be empty",
                details={"field": "controlled_fixture_ids"},
            )
        object.__setattr__(self, "controlled_fixture_ids", fixtures)
        gates = tuple(
            _identifier(item, f"hard_gate_ids[{index}]")
            for index, item in enumerate(self.hard_gate_ids)
        )
        required_common = {
            GATE_NO_A11Y_REGRESSION,
            GATE_NO_AUTHORIZATION_REGRESSION,
            GATE_NO_CONFIRMATION_REGRESSION,
            GATE_SCOPE_SELECTED_SCREEN,
        }
        missing_common = sorted(required_common - set(gates))
        if missing_common:
            raise GuiBenchmarkError(
                "hard_gate_ids is missing required gates",
                reason_code=BenchmarkReasonCode.MISSING_HARD_GATE.value,
                details={"missing": missing_common},
            )
        if self.kind == BenchmarkTaskKind.INTERACTION_STEP_REDUCTION.value:
            if GATE_NO_CONFIRMATION_REGRESSION not in gates:
                raise GuiBenchmarkError(
                    "interaction-step reduction must keep the confirmation gate",
                    reason_code=BenchmarkReasonCode.MISSING_HARD_GATE.value,
                    details={"kind": self.kind},
                )
        if self.kind == BenchmarkTaskKind.ACTION_BINDING_INTEGRITY.value:
            required_binding = {
                "check:policy",
                "check:confirmation",
                "check:host-boundary",
            }
            missing_binding = sorted(required_binding - set(self.affected_check_ids))
            if missing_binding:
                raise GuiBenchmarkError(
                    "action-binding tasks must include policy, confirmation, and host checks",
                    reason_code=BenchmarkReasonCode.MISSING_HARD_GATE.value,
                    details={"missing": missing_binding},
                )
        object.__setattr__(self, "hard_gate_ids", gates)
        components = tuple(
            _identifier(item, f"affected_component_ids[{index}]")
            for index, item in enumerate(self.affected_component_ids)
        )
        unknown_components = sorted(set(components) - REGISTERED_COMPONENT_IDS)
        if not components or unknown_components:
            raise GuiBenchmarkError(
                "affected_component_ids must be registered Agent Supervisor components",
                reason_code=BenchmarkReasonCode.UNKNOWN_COMPONENT_ID.value,
                details={"unknown": unknown_components or ["<empty>"]},
            )
        object.__setattr__(self, "affected_component_ids", components)
        scenarios = tuple(
            _identifier(item, f"affected_scenario_ids[{index}]")
            for index, item in enumerate(self.affected_scenario_ids)
        )
        unknown_scenarios = sorted(set(scenarios) - REGISTERED_SCENARIO_IDS)
        if not scenarios or unknown_scenarios:
            raise GuiBenchmarkError(
                "affected_scenario_ids must be registered evaluation scenarios",
                reason_code=BenchmarkReasonCode.UNKNOWN_SCENARIO_ID.value,
                details={"unknown": unknown_scenarios or ["<empty>"]},
            )
        object.__setattr__(self, "affected_scenario_ids", scenarios)
        checks = tuple(
            _identifier(item, f"affected_check_ids[{index}]")
            for index, item in enumerate(self.affected_check_ids)
        )
        unknown_checks = sorted(set(checks) - REGISTERED_CHECK_IDS)
        if not checks or unknown_checks:
            raise GuiBenchmarkError(
                "affected_check_ids must be registered host checks",
                reason_code=BenchmarkReasonCode.UNKNOWN_CHECK_ID.value,
                details={"unknown": unknown_checks or ["<empty>"]},
            )
        object.__setattr__(self, "affected_check_ids", checks)
        object.__setattr__(
            self,
            "raw_retrieval_token_estimate",
            _int(
                self.raw_retrieval_token_estimate,
                "raw_retrieval_token_estimate",
                minimum=1,
            ),
        )
        policy = _reject_rewrite(
            _reject_vendor(_text(self.conflict_policy, "conflict_policy"), "conflict_policy"),
            "conflict_policy",
        )
        object.__setattr__(self, "conflict_policy", policy)
        object.__setattr__(
            self,
            "interface",
            _text(self.interface, "interface") or GUI_BENCHMARK_TASK_INTERFACE,
        )
        if self.interface != GUI_BENCHMARK_TASK_INTERFACE:
            raise GuiBenchmarkError(
                "task interface must be GuiBenchmarkTask@1",
                details={"interface": self.interface},
            )
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, "schema_version") or GUI_BENCHMARK_TASK_SCHEMA,
        )
        if self.schema_version != GUI_BENCHMARK_TASK_SCHEMA:
            raise GuiBenchmarkError(
                "task schema_version is unsupported",
                details={"schema_version": self.schema_version},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "affected_check_ids": list(self.affected_check_ids),
            "affected_component_ids": list(self.affected_component_ids),
            "affected_scenario_ids": list(self.affected_scenario_ids),
            "application_id": self.application_id,
            "baseline_id": self.baseline_id,
            "bounded_file_paths": list(self.bounded_file_paths),
            "conflict_policy": self.conflict_policy,
            "controlled_fixture_ids": list(self.controlled_fixture_ids),
            "declared_method": self.declared_method,
            "declared_tier": self.declared_tier,
            "evidence_class": self.evidence_class,
            "expected_decision": self.expected_decision,
            "expected_route": self.expected_route,
            "hard_gate_ids": list(self.hard_gate_ids),
            "interface": self.interface,
            "kind": self.kind,
            "objective_ids": list(self.objective_ids),
            "objective_metric_ids": list(self.objective_metric_ids),
            "raw_retrieval_token_estimate": self.raw_retrieval_token_estimate,
            "reference_id": self.reference_id,
            "route_id": self.route_id,
            "schema_version": self.schema_version,
            "screen_id": self.screen_id,
            "task_id": self.task_id,
            "title": self.title,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any) -> "GuiBenchmarkTask":
        payload = _require_mapping(value, "GuiBenchmarkTask")
        _reject_unknown(payload, _TASK_KEYS, "GuiBenchmarkTask")
        _require_interface(
            payload, GUI_BENCHMARK_TASK_INTERFACE, GUI_BENCHMARK_TASK_SCHEMA
        )
        return cls(
            task_id=payload["task_id"],
            kind=payload["kind"],
            title=payload["title"],
            objective_ids=tuple(_identifiers(payload["objective_ids"], "objective_ids")),
            objective_metric_ids=tuple(
                _identifiers(payload["objective_metric_ids"], "objective_metric_ids")
            ),
            application_id=payload["application_id"],
            screen_id=payload["screen_id"],
            route_id=payload["route_id"],
            expected_route=payload["expected_route"],
            declared_method=payload["declared_method"],
            declared_tier=payload["declared_tier"],
            expected_decision=payload["expected_decision"],
            evidence_class=payload["evidence_class"],
            baseline_id=payload["baseline_id"],
            reference_id=payload["reference_id"],
            bounded_file_paths=tuple(
                _bounded_paths(payload["bounded_file_paths"], "bounded_file_paths")
            ),
            controlled_fixture_ids=tuple(
                _identifiers(payload["controlled_fixture_ids"], "controlled_fixture_ids")
            ),
            hard_gate_ids=tuple(_identifiers(payload["hard_gate_ids"], "hard_gate_ids")),
            affected_component_ids=tuple(
                _identifiers(payload["affected_component_ids"], "affected_component_ids")
            ),
            affected_scenario_ids=tuple(
                _identifiers(payload["affected_scenario_ids"], "affected_scenario_ids")
            ),
            affected_check_ids=tuple(
                _identifiers(payload["affected_check_ids"], "affected_check_ids")
            ),
            raw_retrieval_token_estimate=payload["raw_retrieval_token_estimate"],
            conflict_policy=payload["conflict_policy"],
            interface=payload["interface"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class GuiOptimizationBenchmark:
    """Sealed ``GuiOptimizationBenchmark@1`` catalog."""

    benchmark_id: str
    catalog_id: str
    tasks: tuple[GuiBenchmarkTask, ...]
    application_id: str = DEFAULT_APPLICATION_ID
    screen_id: str = DEFAULT_SCREEN_ID
    expected_task_count: int = EXPECTED_TASK_COUNT
    conflict_policy: str = CONFLICT_POLICY
    uses_production_services: bool = False
    uses_production_credentials: bool = False
    interface: str = GUI_OPTIMIZATION_BENCHMARK_INTERFACE
    schema_version: str = GUI_OPTIMIZATION_BENCHMARK_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "benchmark_id", _identifier(self.benchmark_id, "benchmark_id")
        )
        object.__setattr__(self, "catalog_id", _identifier(self.catalog_id, "catalog_id"))
        object.__setattr__(
            self, "application_id", _identifier(self.application_id, "application_id")
        )
        object.__setattr__(self, "screen_id", _identifier(self.screen_id, "screen_id"))
        object.__setattr__(
            self,
            "expected_task_count",
            _int(self.expected_task_count, "expected_task_count", minimum=1),
        )
        object.__setattr__(
            self,
            "uses_production_services",
            _bool(self.uses_production_services, "uses_production_services"),
        )
        object.__setattr__(
            self,
            "uses_production_credentials",
            _bool(self.uses_production_credentials, "uses_production_credentials"),
        )
        if self.uses_production_services or self.uses_production_credentials:
            raise GuiBenchmarkError(
                "benchmark fixtures must not use production services or credentials",
                reason_code=BenchmarkReasonCode.PRODUCTION_SURFACE_FORBIDDEN.value,
            )
        object.__setattr__(
            self,
            "conflict_policy",
            _reject_rewrite(
                _reject_vendor(
                    _text(self.conflict_policy, "conflict_policy"), "conflict_policy"
                ),
                "conflict_policy",
            ),
        )
        object.__setattr__(
            self,
            "interface",
            _text(self.interface, "interface") or GUI_OPTIMIZATION_BENCHMARK_INTERFACE,
        )
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, "schema_version")
            or GUI_OPTIMIZATION_BENCHMARK_SCHEMA,
        )
        if self.interface != GUI_OPTIMIZATION_BENCHMARK_INTERFACE:
            raise GuiBenchmarkError(
                "benchmark interface must be GuiOptimizationBenchmark@1",
                details={"interface": self.interface},
            )
        if self.schema_version != GUI_OPTIMIZATION_BENCHMARK_SCHEMA:
            raise GuiBenchmarkError(
                "benchmark schema_version is unsupported",
                details={"schema_version": self.schema_version},
            )
        if type(self.tasks) is not tuple:
            raise GuiBenchmarkError(
                "tasks must be a tuple of GuiBenchmarkTask values",
                reason_code=BenchmarkReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"value_type": type(self.tasks).__name__},
            )
        validated: list[GuiBenchmarkTask] = []
        for index, task in enumerate(self.tasks):
            if type(task) is not GuiBenchmarkTask:
                raise GuiBenchmarkError(
                    f"tasks[{index}] must be a GuiBenchmarkTask",
                    details={"value_type": type(task).__name__},
                )
            validated.append(task)
        object.__setattr__(self, "tasks", tuple(validated))
        _validate_task_inventory(self.tasks, expected=self.expected_task_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "benchmark_id": self.benchmark_id,
            "catalog_id": self.catalog_id,
            "conflict_policy": self.conflict_policy,
            "expected_task_count": self.expected_task_count,
            "interface": self.interface,
            "schema_version": self.schema_version,
            "screen_id": self.screen_id,
            "tasks": [task.to_dict() for task in self.tasks],
            "uses_production_credentials": self.uses_production_credentials,
            "uses_production_services": self.uses_production_services,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def fixture_bytes(self) -> bytes:
        return _pretty_bytes(self.to_dict())

    def catalog_identity(self) -> str:
        return _digest_id("cid", self.canonical_bytes())

    def task_by_id(self, task_id: str) -> GuiBenchmarkTask:
        ident = _identifier(task_id, "task_id")
        for task in self.tasks:
            if task.task_id == ident:
                return task
        raise GuiBenchmarkError(
            f"unknown task_id: {ident}",
            reason_code=BenchmarkReasonCode.UNKNOWN_TASK_ID.value,
            details={"task_id": ident},
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any) -> "GuiOptimizationBenchmark":
        payload = _require_mapping(value, "GuiOptimizationBenchmark")
        _reject_unknown(payload, _BENCHMARK_KEYS, "GuiOptimizationBenchmark")
        _require_interface(
            payload,
            GUI_OPTIMIZATION_BENCHMARK_INTERFACE,
            GUI_OPTIMIZATION_BENCHMARK_SCHEMA,
        )
        raw_tasks = _require_json_array(payload.get("tasks"), "tasks")
        tasks = tuple(
            GuiBenchmarkTask.from_mapping(item) for item in raw_tasks
        )
        return cls(
            benchmark_id=payload["benchmark_id"],
            catalog_id=payload["catalog_id"],
            tasks=tasks,
            application_id=payload["application_id"],
            screen_id=payload["screen_id"],
            expected_task_count=payload["expected_task_count"],
            conflict_policy=payload["conflict_policy"],
            uses_production_services=payload["uses_production_services"],
            uses_production_credentials=payload["uses_production_credentials"],
            interface=payload["interface"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class GuiBenchmarkResult:
    """Terminal ``GuiBenchmarkResult@1`` record for one executed task."""

    result_id: str
    benchmark_id: str
    task_id: str
    decision: str
    reason_codes: tuple[str, ...] = ()
    route_kind: str = BenchmarkRouteKind.DETERMINISTIC_TRANSFORM.value
    declared_method: str = ""
    declared_tier: str = BenchmarkTier.DETERMINISTIC.value
    measurable_improvement: bool = False
    hard_gate_passed: bool = False
    receipt_id: str = ""
    artifact_ids: tuple[str, ...] = ()
    metric_values: Mapping[str, int | float] = MappingProxyType({})
    interface: str = GUI_BENCHMARK_RESULT_INTERFACE
    schema_version: str = GUI_BENCHMARK_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_id", _identifier(self.result_id, "result_id"))
        object.__setattr__(
            self, "benchmark_id", _identifier(self.benchmark_id, "benchmark_id")
        )
        object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))
        object.__setattr__(
            self, "decision", _closed_enum(self.decision, BenchmarkDecision, "decision")
        )
        codes = tuple(
            _identifier(item, f"reason_codes[{index}]")
            for index, item in enumerate(self.reason_codes)
        )
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(
            self,
            "route_kind",
            _closed_enum(self.route_kind, BenchmarkRouteKind, "route_kind"),
        )
        method = self.declared_method
        if method:
            object.__setattr__(
                self,
                "declared_method",
                _reject_vendor(_identifier(method, "declared_method"), "declared_method"),
            )
        else:
            object.__setattr__(self, "declared_method", "")
        object.__setattr__(
            self,
            "declared_tier",
            _closed_enum(self.declared_tier, BenchmarkTier, "declared_tier"),
        )
        object.__setattr__(
            self,
            "measurable_improvement",
            _bool(self.measurable_improvement, "measurable_improvement"),
        )
        object.__setattr__(
            self, "hard_gate_passed", _bool(self.hard_gate_passed, "hard_gate_passed")
        )
        if self.receipt_id:
            object.__setattr__(
                self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
            )
        else:
            object.__setattr__(self, "receipt_id", "")
        artifacts = tuple(
            _identifier(item, f"artifact_ids[{index}]")
            for index, item in enumerate(self.artifact_ids)
        )
        object.__setattr__(self, "artifact_ids", artifacts)
        metrics_raw = self.metric_values
        if type(metrics_raw) is not dict and not isinstance(metrics_raw, Mapping):
            raise GuiBenchmarkError(
                "metric_values must be a JSON object",
                reason_code=BenchmarkReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"value_type": type(metrics_raw).__name__},
            )
        metrics: dict[str, int | float] = {}
        for key, item in dict(metrics_raw).items():
            ident = _identifier(key, "metric_values.key")
            metrics[ident] = _finite_number(item, f"metric_values[{ident}]")
        object.__setattr__(self, "metric_values", MappingProxyType(metrics))
        object.__setattr__(
            self,
            "interface",
            _text(self.interface, "interface") or GUI_BENCHMARK_RESULT_INTERFACE,
        )
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, "schema_version") or GUI_BENCHMARK_RESULT_SCHEMA,
        )
        if self.interface != GUI_BENCHMARK_RESULT_INTERFACE:
            raise GuiBenchmarkError(
                "result interface must be GuiBenchmarkResult@1",
                details={"interface": self.interface},
            )
        if self.schema_version != GUI_BENCHMARK_RESULT_SCHEMA:
            raise GuiBenchmarkError(
                "result schema_version is unsupported",
                details={"schema_version": self.schema_version},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_ids": list(self.artifact_ids),
            "benchmark_id": self.benchmark_id,
            "decision": self.decision,
            "declared_method": self.declared_method,
            "declared_tier": self.declared_tier,
            "hard_gate_passed": self.hard_gate_passed,
            "interface": self.interface,
            "measurable_improvement": self.measurable_improvement,
            "metric_values": {key: self.metric_values[key] for key in sorted(self.metric_values)},
            "reason_codes": list(self.reason_codes),
            "receipt_id": self.receipt_id,
            "result_id": self.result_id,
            "route_kind": self.route_kind,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any) -> "GuiBenchmarkResult":
        payload = _require_mapping(value, "GuiBenchmarkResult")
        _reject_unknown(payload, _RESULT_KEYS, "GuiBenchmarkResult")
        _require_interface(
            payload, GUI_BENCHMARK_RESULT_INTERFACE, GUI_BENCHMARK_RESULT_SCHEMA
        )
        metrics = _require_mapping(payload.get("metric_values"), "metric_values")
        return cls(
            result_id=payload["result_id"],
            benchmark_id=payload["benchmark_id"],
            task_id=payload["task_id"],
            decision=payload["decision"],
            reason_codes=tuple(
                _identifiers(payload["reason_codes"], "reason_codes", nonempty=False)
            ),
            route_kind=payload["route_kind"],
            declared_method=payload["declared_method"],
            declared_tier=payload["declared_tier"],
            measurable_improvement=payload["measurable_improvement"],
            hard_gate_passed=payload["hard_gate_passed"],
            receipt_id=payload["receipt_id"],
            artifact_ids=tuple(
                _identifiers(payload["artifact_ids"], "artifact_ids", nonempty=False)
            ),
            metric_values=metrics,
            interface=payload["interface"],
            schema_version=payload["schema_version"],
        )


def _validate_task_inventory(
    tasks: Sequence[GuiBenchmarkTask], *, expected: int = EXPECTED_TASK_COUNT
) -> None:
    if len(tasks) != expected or expected != EXPECTED_TASK_COUNT:
        raise GuiBenchmarkError(
            f"benchmark catalog must contain exactly {EXPECTED_TASK_COUNT} tasks",
            reason_code=BenchmarkReasonCode.TASK_COUNT_MISMATCH.value,
            details={"count": len(tasks), "expected": EXPECTED_TASK_COUNT},
        )
    ids = [task.task_id for task in tasks]
    if len(set(ids)) != len(ids):
        dupes = sorted({item for item in ids if ids.count(item) > 1})
        raise GuiBenchmarkError(
            "benchmark catalog task IDs must be unique",
            reason_code=BenchmarkReasonCode.DUPLICATE_TASK_ID.value,
            details={"duplicates": dupes},
        )
    kinds = [task.kind for task in tasks]
    if len(set(kinds)) != len(kinds):
        dupes = sorted({item for item in kinds if kinds.count(item) > 1})
        raise GuiBenchmarkError(
            "benchmark catalog task kinds must be unique",
            reason_code=BenchmarkReasonCode.DUPLICATE_TASK_KIND.value,
            details={"duplicates": dupes},
        )
    missing = [kind for kind in REQUIRED_TASK_KINDS if kind not in set(kinds)]
    if missing:
        raise GuiBenchmarkError(
            "benchmark catalog is missing required task kinds",
            reason_code=BenchmarkReasonCode.MISSING_REQUIRED_KIND.value,
            details={"missing": missing},
        )


def _task(
    kind: BenchmarkTaskKind,
    *,
    title: str,
    metric: str,
    method: str,
    route: BenchmarkRouteKind = BenchmarkRouteKind.DETERMINISTIC_TRANSFORM,
    decision: BenchmarkDecision = BenchmarkDecision.ACCEPT,
    evidence: BenchmarkEvidenceClass = BenchmarkEvidenceClass.STRUCTURAL,
    components: tuple[str, ...] = (COMPONENT_GOAL_FORM,),
    scenarios: tuple[str, ...],
    checks: tuple[str, ...],
    files: tuple[str, ...] = (DEFAULT_SOURCE_PATH, FIXTURE_HOST_PATH),
    fixtures: tuple[str, ...] = (FIXTURE_HOST_ID, FIXTURE_SCENARIOS_ID),
    extra_gates: tuple[str, ...] = (),
    tokens: int,
) -> GuiBenchmarkTask:
    slug = kind.value.replace("_", "-")
    gates = COMMON_HARD_GATES + extra_gates
    return GuiBenchmarkTask(
        task_id=f"task:{slug}",
        kind=kind.value,
        title=title,
        objective_ids=(f"objective:{slug}",),
        objective_metric_ids=(metric,),
        expected_route=route.value,
        declared_method=method,
        declared_tier=ROUTE_DEFAULT_TIER[route.value],
        expected_decision=decision.value,
        evidence_class=evidence.value,
        baseline_id=f"baseline:agent-supervisor-{slug}",
        reference_id=f"ref:vgo-083-{slug}",
        bounded_file_paths=files,
        controlled_fixture_ids=fixtures,
        hard_gate_ids=gates,
        affected_component_ids=components,
        affected_scenario_ids=scenarios,
        affected_check_ids=checks,
        raw_retrieval_token_estimate=tokens,
    )


def sealed_benchmark_tasks() -> tuple[GuiBenchmarkTask, ...]:
    """Return the sealed 15-task inventory in declaration order."""

    source_and_host = (DEFAULT_SOURCE_PATH, FIXTURE_HOST_PATH)
    source_host_scenarios = (
        DEFAULT_SOURCE_PATH,
        FIXTURE_HOST_PATH,
        FIXTURE_SCENARIOS_PATH,
    )
    return (
        _task(
            BenchmarkTaskKind.FOCUS_RESTORATION,
            title="Restore focus after a controlled Agent Supervisor rerender.",
            metric="focus_restoration_coverage",
            method="exact_aria_reference_repair",
            scenarios=(
                STABLE_SCENARIO_IDS["keyboard_only"],
                STABLE_SCENARIO_IDS["valid_submission"],
            ),
            checks=(
                "check:accessibility-contracts",
                "check:interaction-scenarios",
                "check:direct-tests",
            ),
            files=source_and_host,
            tokens=1720,
        ),
        _task(
            BenchmarkTaskKind.ACCESSIBLE_LABELS,
            title="Add exact accessible names on the selected goal form.",
            metric="accessible_name_coverage",
            method="exact_label_substitution",
            scenarios=(
                STABLE_SCENARIO_IDS["invalid_submission"],
                STABLE_SCENARIO_IDS["keyboard_only"],
            ),
            checks=(
                "check:accessible-name",
                "check:accessibility-contracts",
                "check:direct-tests",
            ),
            tokens=1680,
        ),
        _task(
            BenchmarkTaskKind.ERROR_PRESENTATION,
            title="Associate validation errors with the fields they describe.",
            metric="error_association_coverage",
            method="aria_reference_repair",
            scenarios=(
                STABLE_SCENARIO_IDS["invalid_submission"],
                STABLE_SCENARIO_IDS["keyboard_only"],
            ),
            checks=(
                "check:accessibility-contracts",
                "check:accessibility-scenarios",
                "check:direct-tests",
            ),
            tokens=1740,
        ),
        _task(
            BenchmarkTaskKind.LOADING_STATE,
            title="Add the missing loading outcome on the selected screen.",
            metric="loading_state_completeness",
            method="exact_route_migration",
            components=(COMPONENT_CONSOLE_ROOT,),
            scenarios=(
                STABLE_SCENARIO_IDS["loading"],
                STABLE_SCENARIO_IDS["initial_load"],
            ),
            checks=(
                "check:interaction-scenarios",
                "check:outcome",
                "check:reachability",
            ),
            files=source_host_scenarios,
            fixtures=(FIXTURE_HOST_ID, FIXTURE_SERVICES_ID, FIXTURE_SCENARIOS_ID),
            tokens=1810,
        ),
        _task(
            BenchmarkTaskKind.FAILURE_STATE,
            title="Add the missing failure and recovery outcome on the selected screen.",
            metric="failure_state_completeness",
            method="exact_route_migration",
            components=(COMPONENT_CONSOLE_ROOT,),
            scenarios=(
                STABLE_SCENARIO_IDS["recoverable_failure"],
                STABLE_SCENARIO_IDS["unrecoverable_failure"],
                STABLE_SCENARIO_IDS["service_unavailable"],
            ),
            checks=(
                "check:interaction-scenarios",
                "check:outcome",
                "check:reachability",
            ),
            files=source_host_scenarios,
            fixtures=(FIXTURE_HOST_ID, FIXTURE_SERVICES_ID, FIXTURE_SCENARIOS_ID),
            tokens=1860,
        ),
        _task(
            BenchmarkTaskKind.INTERACTION_STEP_REDUCTION,
            title="Reduce steps for one non-sensitive, non-destructive task.",
            metric="interaction_step_count",
            method="exact_action_binding_migration",
            scenarios=(
                STABLE_SCENARIO_IDS["valid_submission"],
                STABLE_SCENARIO_IDS["success"],
            ),
            checks=(
                "check:interaction",
                "check:confirmation",
                "check:interaction-scenarios",
            ),
            extra_gates=(GATE_NO_HIDDEN_DISPATCH,),
            tokens=1790,
        ),
        _task(
            BenchmarkTaskKind.RESPONSIVE_OVERFLOW,
            title="Remove narrow-viewport document overflow on the selected screen.",
            metric="overflow_violation_count",
            method="design_token_substitution",
            evidence=BenchmarkEvidenceClass.AUTOMATED,
            components=(COMPONENT_CONSOLE_ROOT,),
            scenarios=(
                STABLE_SCENARIO_IDS["viewport_mobile"],
                STABLE_SCENARIO_IDS["viewport_desktop"],
            ),
            checks=(
                "check:overflow",
                "check:responsive",
                "check:containing-screenshots",
            ),
            tokens=1650,
        ),
        _task(
            BenchmarkTaskKind.PRIMARY_ACTION_HIERARCHY,
            title="Improve primary-action hierarchy under human review only.",
            metric="primary_action_hierarchy_score",
            method="human_hierarchy_review",
            route=BenchmarkRouteKind.HUMAN_REVIEW,
            decision=BenchmarkDecision.HUMAN_REVIEW,
            evidence=BenchmarkEvidenceClass.HUMAN_REVIEWED,
            components=(COMPONENT_CONSOLE_ROOT, COMPONENT_GOAL_FORM),
            scenarios=(
                STABLE_SCENARIO_IDS["initial_load"],
                STABLE_SCENARIO_IDS["viewport_desktop"],
            ),
            checks=(
                "check:containing-screenshots",
                "check:accessibility-contracts",
                "check:direct-tests",
            ),
            tokens=1900,
        ),
        _task(
            BenchmarkTaskKind.DESIGN_TOKEN_CONSISTENCY,
            title="Replace one inconsistent design token on the selected screen.",
            metric="design_token_consistency",
            method="design_token_substitution",
            components=(COMPONENT_CONSOLE_ROOT,),
            scenarios=(
                STABLE_SCENARIO_IDS["dark_mode"],
                STABLE_SCENARIO_IDS["viewport_desktop"],
            ),
            checks=(
                "check:dependent-screenshots",
                "check:contrast",
                "check:direct-tests",
            ),
            tokens=1620,
        ),
        _task(
            BenchmarkTaskKind.CONFIRMATION_UX,
            title="Enforce exact destructive confirmation on the selected screen.",
            metric="confirmation_binding_coverage",
            method="exact_action_binding_migration",
            evidence=BenchmarkEvidenceClass.INTEGRITY,
            scenarios=(
                STABLE_SCENARIO_IDS["confirmation_grant"],
                STABLE_SCENARIO_IDS["confirmation_deny"],
            ),
            checks=(
                "check:confirmation",
                "check:policy",
                "check:interaction",
            ),
            extra_gates=(GATE_NO_HIDDEN_DISPATCH,),
            tokens=1760,
        ),
        _task(
            BenchmarkTaskKind.EMPTY_STATE_GUIDANCE,
            title="Improve empty-state guidance on the selected screen.",
            metric="empty_state_guidance_coverage",
            method="exact_label_substitution",
            components=(COMPONENT_CONSOLE_ROOT,),
            scenarios=(STABLE_SCENARIO_IDS["empty"],),
            checks=(
                "check:accessible-name",
                "check:accessibility-scenarios",
                "check:direct-tests",
            ),
            tokens=1580,
        ),
        _task(
            BenchmarkTaskKind.KEYBOARD_REACHABILITY,
            title="Repair keyboard activation for a custom control on the selected screen.",
            metric="keyboard_access_coverage",
            method="aria_reference_repair",
            scenarios=(
                STABLE_SCENARIO_IDS["keyboard_only"],
                STABLE_SCENARIO_IDS["initial_load"],
            ),
            checks=(
                "check:accessibility-contracts",
                "check:interaction-scenarios",
                "check:direct-tests",
            ),
            tokens=1700,
        ),
        _task(
            BenchmarkTaskKind.LOCALIZATION_CLIPPING,
            title="Prevent localized text clipping on the selected screen.",
            metric="localization_clipping_count",
            method="exact_label_substitution",
            evidence=BenchmarkEvidenceClass.AUTOMATED,
            components=(COMPONENT_CONSOLE_ROOT, COMPONENT_GOAL_FORM),
            scenarios=(
                STABLE_SCENARIO_IDS["text_scale_200"],
                STABLE_SCENARIO_IDS["viewport_mobile"],
            ),
            checks=(
                "check:clipping",
                "check:locale-scenarios",
                "check:text-layout-screenshots",
            ),
            tokens=1840,
        ),
        _task(
            BenchmarkTaskKind.MODAL_FOCUS_LIFECYCLE,
            title="Restore modal focus trap and return focus on the selected screen.",
            metric="modal_focus_lifecycle_coverage",
            method="aria_reference_repair",
            scenarios=(
                STABLE_SCENARIO_IDS["confirmation_grant"],
                STABLE_SCENARIO_IDS["keyboard_only"],
            ),
            checks=(
                "check:accessibility-contracts",
                "check:interaction-scenarios",
                "check:interaction",
            ),
            tokens=1780,
        ),
        _task(
            BenchmarkTaskKind.ACTION_BINDING_INTEGRITY,
            title="Keep action bindings exact and host-authorized on the selected screen.",
            metric="action_binding_integrity",
            method="exact_action_binding_migration",
            evidence=BenchmarkEvidenceClass.INTEGRITY,
            scenarios=(
                STABLE_SCENARIO_IDS["valid_submission"],
                STABLE_SCENARIO_IDS["confirmation_deny"],
            ),
            checks=(
                "check:action-bindings",
                "check:policy",
                "check:confirmation",
                "check:host-boundary",
                "check:invocation-tests",
            ),
            extra_gates=(GATE_NO_HIDDEN_DISPATCH,),
            tokens=1880,
        ),
    )


def build_benchmark_catalog() -> GuiOptimizationBenchmark:
    """Construct the sealed catalog.  Repeated calls are byte-identical."""

    return GuiOptimizationBenchmark(
        benchmark_id=BENCHMARK_ID,
        catalog_id=CATALOG_ID,
        tasks=sealed_benchmark_tasks(),
        application_id=DEFAULT_APPLICATION_ID,
        screen_id=DEFAULT_SCREEN_ID,
        expected_task_count=EXPECTED_TASK_COUNT,
        conflict_policy=CONFLICT_POLICY,
        uses_production_services=False,
        uses_production_credentials=False,
    )


def default_catalog_path(repo_root: Path | None = None) -> Path:
    """Resolve the durable catalog fixture.

    ``repo_root`` is the lift_coding superproject root used by ``gui-opt``.
    When omitted, the path is resolved from this package's accelerator tree.
    """

    if repo_root is not None:
        return Path(repo_root) / DEFAULT_CATALOG_RELATIVE_PATH
    package_root = Path(__file__).resolve().parents[3]
    return package_root / PACKAGE_CATALOG_RELATIVE_PATH


def load_benchmark_catalog(path: str | Path | None = None) -> GuiOptimizationBenchmark:
    """Load and validate a catalog document from disk."""

    catalog_path = Path(path) if path is not None else default_catalog_path()
    try:
        raw = catalog_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GuiBenchmarkError(
            f"benchmark catalog is unavailable: {catalog_path}",
            reason_code=BenchmarkReasonCode.CATALOG_UNAVAILABLE.value,
            details={"path": str(catalog_path)},
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GuiBenchmarkError(
            "benchmark catalog is not valid JSON",
            reason_code=BenchmarkReasonCode.CATALOG_UNAVAILABLE.value,
            details={"path": str(catalog_path)},
        ) from exc
    return GuiOptimizationBenchmark.from_mapping(payload)


def render_catalog_document(catalog: GuiOptimizationBenchmark | None = None) -> bytes:
    """Return the durable fixture encoding of the sealed catalog."""

    built = catalog if catalog is not None else build_benchmark_catalog()
    return built.fixture_bytes()


def write_catalog_fixture(path: str | Path | None = None) -> Path:
    """Write the sealed catalog to the durable fixture path."""

    catalog_path = Path(path) if path is not None else default_catalog_path()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_bytes(render_catalog_document())
    return catalog_path


def materialize_catalog_fixture(path: str | Path | None = None) -> Path:
    """Alias for :func:`write_catalog_fixture` used by module CLI rescue."""

    return write_catalog_fixture(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Module CLI: ``write`` / ``materialize`` the sealed catalog fixture."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        sys.stdout.write(
            "usage: python -m ipfs_accelerate_py.agent_supervisor.gui_optimizer."
            "benchmark write|materialize [path]\n"
        )
        return 0
    command = args[0]
    if command not in {"write", "materialize"}:
        sys.stderr.write(f"unknown command: {command}\n")
        return 2
    if len(args) > 2:
        sys.stderr.write("write|materialize accepts at most one path\n")
        return 2
    target = Path(args[1]) if len(args) > 1 else default_catalog_path()
    written = write_catalog_fixture(target)
    sys.stdout.write(f"{written}\n")
    return 0


def empty_benchmark_result(
    task: GuiBenchmarkTask,
    *,
    result_id: str | None = None,
) -> GuiBenchmarkResult:
    """Return a pending result shell for later execution (VGO-090)."""

    return GuiBenchmarkResult(
        result_id=result_id or f"result:{task.task_id.removeprefix('task:')}",
        benchmark_id=BENCHMARK_ID,
        task_id=task.task_id,
        decision=BenchmarkDecision.PENDING.value,
        reason_codes=(),
        route_kind=task.expected_route,
        declared_method=task.declared_method,
        declared_tier=task.declared_tier,
        measurable_improvement=False,
        hard_gate_passed=False,
        receipt_id="",
        artifact_ids=(),
        metric_values={},
    )


__all__ = (
    "BENCHMARK_ID",
    "BenchmarkDecision",
    "BenchmarkEvidenceClass",
    "BenchmarkReasonCode",
    "BenchmarkRouteKind",
    "BenchmarkTaskKind",
    "BenchmarkTier",
    "CATALOG_ID",
    "CONFLICT_POLICY",
    "DEFAULT_CATALOG_RELATIVE_PATH",
    "EXPECTED_TASK_COUNT",
    "GUI_BENCHMARK_RESULT_INTERFACE",
    "GUI_BENCHMARK_RESULT_SCHEMA",
    "GUI_BENCHMARK_TASK_INTERFACE",
    "GUI_OPTIMIZATION_BENCHMARK_INTERFACE",
    "SUBJECTIVE_KINDS",
    "GuiBenchmarkError",
    "GuiBenchmarkResult",
    "GuiBenchmarkTask",
    "GuiOptimizationBenchmark",
    "REQUIRED_TASK_KINDS",
    "build_benchmark_catalog",
    "default_catalog_path",
    "empty_benchmark_result",
    "load_benchmark_catalog",
    "main",
    "materialize_catalog_fixture",
    "render_catalog_document",
    "sealed_benchmark_tasks",
    "write_catalog_fixture",
)


if __name__ == "__main__":
    raise SystemExit(main())

