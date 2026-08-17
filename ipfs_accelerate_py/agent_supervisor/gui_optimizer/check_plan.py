"""Affected-check selection with uncertainty fallback (VGO-051).

Interfaces owned by this module:

* ``GuiAffectedCheckPlanner@1`` — select and order repository-owned checks
  from an invalidation plan and evaluator risk classifications
* ``GuiCheckPlan@1`` — closed, ordered plan of registry checks
* ``GuiCheckExecutionReceipt@1`` — typed record of executed checks and
  whether a required failure blocked acceptance

Commands come only from a fixed host registry.  Browser input and
proposals cannot inject subprocesses, name an executable, or suppress
mandatory fallback.  Direct unit, component, and scenario checks run
first; policy, host, browser, build, or broader suites are added when
graph confidence, dynamic behavior, shared tokens, or failures require
them.

Fail-closed invariants:

* local style or component changes never pull unrelated screenshots;
* action-binding changes always include policy, interaction, and host
  checks;
* uncertainty (non-exact confidence, missing/stale/opaque edges,
  dynamic behavior, shared tokens, or prior failures) expands to a
  documented broader fallback;
* a failed required check blocks acceptance;
* argv is looked up from the registry — callers cannot supply it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Protocol

from .authority import (
    AuthorityReasonCode,
    FORBIDDEN_BROWSER_COMMAND_FIELDS,
    FORBIDDEN_BROWSER_PATH_FIELDS,
    GuiAuthorityError,
    GuiHostBoundaryPolicy,
)
from .patch_scope import (
    GUI_INVALIDATION_PLAN_INTERFACE,
    GUI_INVALIDATION_PLAN_SCHEMA,
    PatchScopeInvalidationRecord,
)

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

GUI_AFFECTED_CHECK_PLANNER_INTERFACE: Final[str] = "GuiAffectedCheckPlanner@1"
GUI_CHECK_PLAN_INTERFACE: Final[str] = "GuiCheckPlan@1"
GUI_CHECK_EXECUTION_RECEIPT_INTERFACE: Final[str] = "GuiCheckExecutionReceipt@1"

GUI_AFFECTED_CHECK_PLANNER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
    "affected-check-planner@1"
)
GUI_CHECK_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/check-plan@1"
)
GUI_CHECK_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
    "check-execution-receipt@1"
)
GUI_AFFECTED_CHECK_PLANNER_VERSION: Final[str] = (
    "gui-affected-check-planner@1.0.0"
)

HOST_PYTHON_EXECUTABLE: Final[str] = "/usr/bin/python3.12"
HOST_VALIDATION_PATH: Final[str] = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
)
ACCEPTANCE_POLICY: Final[str] = "block_on_required_failure"

_DISPATCH_SCRIPT: Final[str] = (
    "import sys;"
    "raise SystemExit(0 if len(sys.argv)==2 and sys.argv[1].startswith('check:') else 2)"
)
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/#@-]{0,255}$")
_COMMAND_META_RE: Final = re.compile(r"[;&|`$<>\n]|\$\(|\)")


class CheckFamily(str, Enum):
    """Closed execution families.  Direct families run first."""

    UNIT = "unit"
    COMPONENT = "component"
    SCENARIO = "scenario"
    POLICY = "policy"
    INTERACTION = "interaction"
    HOST = "host"
    BROWSER = "browser"
    BUILD = "build"
    FALLBACK = "fallback"


FAMILY_ORDER: Final[tuple[CheckFamily, ...]] = (
    CheckFamily.UNIT,
    CheckFamily.COMPONENT,
    CheckFamily.SCENARIO,
    CheckFamily.POLICY,
    CheckFamily.INTERACTION,
    CheckFamily.HOST,
    CheckFamily.BROWSER,
    CheckFamily.BUILD,
    CheckFamily.FALLBACK,
)
DIRECT_FAMILIES: Final[frozenset[CheckFamily]] = frozenset(
    {CheckFamily.UNIT, CheckFamily.COMPONENT, CheckFamily.SCENARIO}
)
EXPANSION_FAMILIES: Final[frozenset[CheckFamily]] = frozenset(
    {
        CheckFamily.POLICY,
        CheckFamily.INTERACTION,
        CheckFamily.HOST,
        CheckFamily.BROWSER,
        CheckFamily.BUILD,
        CheckFamily.FALLBACK,
    }
)


class CheckRiskClass(str, Enum):
    """Evaluator risk classifications reused from VGO-040."""

    HARD = "hard"
    HEURISTIC = "heuristic"
    NEUTRAL = "neutral"
    REVIEW = "review"


class ExtractionConfidence(str, Enum):
    """Graph / invalidation extraction confidence."""

    EXACT = "exact"
    CONSERVATIVE = "conservative"
    HEURISTIC = "heuristic"
    OPAQUE = "opaque"


class ScreenshotScope(str, Enum):
    """How a check may name screenshots."""

    NONE = "none"
    AFFECTED = "affected"
    NEVER_GLOBAL = "never_global"


class CheckStatus(str, Enum):
    """Closed per-check execution outcomes."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class PlanDisposition(str, Enum):
    """Closed outcomes for ``GuiCheckPlan@1`` / execution receipts."""

    READY = "ready"
    EXECUTED = "executed"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class CheckPlanReasonCode(str, Enum):
    """Stable reason codes for planning and execution."""

    PLANNED = "planned"
    EXECUTED = "executed"
    FALLBACK_EXPANDED = "fallback_expanded"
    UNCERTAIN_GRAPH_CONFIDENCE = "uncertain_graph_confidence"
    DYNAMIC_BEHAVIOR = "dynamic_behavior"
    SHARED_TOKENS = "shared_tokens"
    MISSING_EDGE = "missing_edge"
    STALE_EDGE = "stale_edge"
    OPAQUE_EDGE = "opaque_edge"
    PRIOR_FAILURE = "prior_failure"
    CRITICAL_EVIDENCE_UNKNOWN = "critical_evidence_unknown"
    HARD_GATE_REGRESSION = "hard_gate_regression"
    ACTION_POLICY_REQUIRED = "action_policy_required"
    LOCAL_SCREENSHOT_PRECISION = "local_screenshot_precision"
    UNRELATED_SCREENSHOT_EXCLUDED = "unrelated_screenshot_excluded"
    REQUIRED_CHECK_FAILED = "required_check_failed"
    ACCEPTANCE_BLOCKED = "acceptance_blocked"
    ACCEPTANCE_ALLOWED = "acceptance_allowed"
    COMMAND_STRING_FORBIDDEN = "command_string_forbidden"
    FALLBACK_SUPPRESSION_FORBIDDEN = "fallback_suppression_forbidden"
    BROWSER_COMMAND_FORBIDDEN = "browser_command_forbidden"
    UNKNOWN_CHECK_ID = "unknown_check_id"
    UNKNOWN_FIELD = AuthorityReasonCode.UNKNOWN_FIELD.value
    INVALID_COLLECTION_TYPE = AuthorityReasonCode.INVALID_COLLECTION_TYPE.value
    INVALID_CHECK_PLAN_INPUT = "invalid_check_plan_input"
    MISSING_INVALIDATION_RECORD = "missing_invalidation_record"
    CHECK_UNAVAILABLE = "check_unavailable"


UNCERTAIN_CONFIDENCES: Final[frozenset[str]] = frozenset(
    {
        ExtractionConfidence.CONSERVATIVE.value,
        ExtractionConfidence.HEURISTIC.value,
        ExtractionConfidence.OPAQUE.value,
    }
)
UNCERTAIN_REASONS: Final[frozenset[str]] = frozenset(
    {
        "missing_edge",
        "stale_edge",
        "opaque_edge",
        "fallback_expansion",
    }
)
KNOWN_CHANGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "component_implementation",
        "props_event_contract",
        "state_machine",
        "css_design_token",
        "action_binding",
        "localization",
        "accessibility",
        "test",
        "screenshot",
        "other",
    }
)
ACTION_CHANGE_KINDS: Final[frozenset[str]] = frozenset({"action_binding"})
STYLE_CHANGE_KINDS: Final[frozenset[str]] = frozenset({"css_design_token"})
LOCAL_CHANGE_KINDS: Final[frozenset[str]] = frozenset(
    {"component_implementation", "css_design_token", "localization", "screenshot"}
)
REASON_TO_CHANGE_KIND: Final[dict[str, str]] = {
    "component_changed": "component_implementation",
    "props_changed": "props_event_contract",
    "state_changed": "state_machine",
    "style_changed": "css_design_token",
    "action_changed": "action_binding",
    "localization_changed": "localization",
}
HARD_GATE_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "accessibility",
        "policy",
        "security",
        "functional",
        "confirmation",
        "invariant",
    }
)
HARD_GATE_TO_CHECKS: Final[dict[str, tuple[str, ...]]] = {
    "accessibility": (
        "check:accessibility-scenarios",
        "check:accessibility-contracts",
        "check:accessible-name",
        "check:contrast",
    ),
    "policy": ("check:policy",),
    "security": ("check:host-boundary", "check:policy"),
    "functional": (
        "check:direct-tests",
        "check:invocation-tests",
        "check:outcome",
    ),
    "confirmation": ("check:confirmation", "check:policy"),
    "invariant": ("check:formal", "check:reachability"),
}
ACTION_REQUIRED_CHECKS: Final[tuple[str, ...]] = (
    "check:policy",
    "check:interaction",
    "check:host-boundary",
)
FALLBACK_EXPANSION_CHECKS: Final[tuple[str, ...]] = (
    "check:broader-screen-fallback",
    "check:policy",
    "check:host-boundary",
    "check:interaction",
    "check:formal",
    "check:accessibility-scenarios",
    "check:interaction-scenarios",
)
SHARED_TOKEN_CHECKS: Final[tuple[str, ...]] = (
    "check:dependent-screenshots",
    "check:contrast",
    "check:responsive",
    "check:clipping",
    "check:overflow",
    "check:broader-screen-fallback",
)
DYNAMIC_BEHAVIOR_CHECKS: Final[tuple[str, ...]] = (
    "check:interaction",
    "check:interaction-scenarios",
    "check:host-boundary",
    "check:invocation-tests",
    "check:broader-screen-fallback",
)
KIND_DEFAULT_CHECKS: Final[dict[str, tuple[str, ...]]] = {
    "component_implementation": (
        "check:capsule",
        "check:direct-tests",
        "check:containing-screenshots",
        "check:accessibility-scenarios",
    ),
    "props_event_contract": (
        "check:parents-consumers",
        "check:action-bindings",
        "check:interface-descriptors",
        "check:contract-tests",
    ),
    "state_machine": (
        "check:reachability",
        "check:outcome",
        "check:formal",
        "check:interaction-scenarios",
    ),
    "css_design_token": (
        "check:dependent-screenshots",
        "check:responsive",
        "check:contrast",
        "check:clipping",
        "check:overflow",
    ),
    "action_binding": (
        "check:policy",
        "check:confirmation",
        "check:host-boundary",
        "check:interaction",
        "check:invocation-tests",
    ),
    "localization": (
        "check:text-layout-screenshots",
        "check:accessible-name",
        "check:locale-scenarios",
    ),
    "accessibility": (
        "check:accessibility-contracts",
        "check:accessibility-scenarios",
        "check:accessible-name",
    ),
    "test": ("check:test-artifacts", "check:direct-tests"),
    "screenshot": (
        "check:screenshot-artifacts",
        "check:containing-screenshots",
    ),
    "other": ("check:broader-screen-fallback",),
}

_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "affected_component_ids",
        "affected_screenshot_ids",
        "application_id",
        "browser_input",
        "change_kinds",
        "evaluator_risk",
        "invalidation",
        "known_screenshot_ids",
        "prior_failures",
        "proposal",
        "screen_id",
        "unrelated_screenshot_ids",
    }
)
_EVALUATOR_RISK_KEYS: Final[frozenset[str]] = frozenset(
    {
        "classification",
        "critical_evidence_unknown",
        "dynamic_behavior",
        "failed_check_ids",
        "graph_confidence",
        "hard_gate_families",
        "hard_gate_regression",
        "risk_class",
        "shared_tokens",
    }
)
_FORBIDDEN_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "argv",
        "cmd",
        "command",
        "commands",
        "cwd",
        "disable_fallback",
        "env",
        "executable",
        "file_path",
        "host_path",
        "no_fallback",
        "process_command",
        "python_process",
        "shell",
        "shell_command",
        "skip_fallback",
        "stdio",
        "subprocess",
        "suppress_fallback",
        "working_directory",
    }
    | FORBIDDEN_BROWSER_COMMAND_FIELDS
    | FORBIDDEN_BROWSER_PATH_FIELDS
)
_NESTED_COMMAND_KEYS: Final[frozenset[str]] = frozenset(
    {
        "argv",
        "cmd",
        "command",
        "commands",
        "cwd",
        "disable_fallback",
        "executable",
        "no_fallback",
        "process_command",
        "shell",
        "shell_command",
        "skip_fallback",
        "subprocess",
        "suppress_fallback",
    }
)
_PROPOSAL_ALLOWED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "acceptance_criteria",
        "analysis_classification",
        "application_id",
        "context_pack_id",
        "decision",
        "expected_screenshot_ids",
        "expected_test_ids",
        "intended_component_ids",
        "intended_file_paths",
        "interface",
        "objective",
        "proposal_id",
        "route_kind",
        "schema_version",
        "screen_id",
        "state_effect_ids",
        "verification_status",
        "visual_effect_summary",
    }
)


class GuiCheckPlanError(GuiAuthorityError):
    """Malformed check-plan input.  Never grants execution."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


# ---------------------------------------------------------------------------
# Closed input helpers
# ---------------------------------------------------------------------------


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise GuiCheckPlanError(
            f"{name} must be a string",
            reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiCheckPlanError(
            f"{name} must not contain NUL",
            reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            details={"field": name},
        )
    text = text_value.strip()
    if required and not text:
        raise GuiCheckPlanError(
            f"{name} must not be empty",
            reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            details={"field": name},
        )
    return text


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiCheckPlanError(
            f"{name} must not contain NUL",
            reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            details={"field": name},
        )
    if text_value == "":
        if required:
            raise GuiCheckPlanError(
                f"{name} must be a nonempty string identifier",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
                details={"field": name},
            )
        return ""
    if text_value != text_value.strip() or not _IDENTIFIER_RE.fullmatch(text_value):
        raise GuiCheckPlanError(
            f"{name} must be a canonical nonempty string identifier",
            reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            details={"field": name},
        )
    return text_value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise GuiCheckPlanError(
            f"{name} must be a boolean",
            reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GuiCheckPlanError(
            f"{name} must be a JSON object",
            reason_code=CheckPlanReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    for key in value:
        if type(key) is not str:
            raise GuiCheckPlanError(
                f"{name} keys must be strings",
                reason_code=CheckPlanReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name, "key_type": type(key).__name__},
            )
    return value


def _require_json_array(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise GuiCheckPlanError(
            f"{name} must be a JSON array (list); "
            f"{type(value).__name__} is not a valid collection",
            reason_code=CheckPlanReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _require_python_sequence(value: Any, name: str) -> Sequence[Any]:
    if type(value) is list or type(value) is tuple:
        return value
    raise GuiCheckPlanError(
        f"{name} must be a JSON array/sequence",
        reason_code=CheckPlanReasonCode.INVALID_COLLECTION_TYPE.value,
        details={"field": name, "value_type": type(value).__name__},
    )


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], noun: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise GuiCheckPlanError(
            f"{noun} contains unknown fields: {unknown}",
            reason_code=CheckPlanReasonCode.UNKNOWN_FIELD.value,
            details={"noun": noun, "unknown_fields": unknown},
        )


def _reject_present_null(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is None:
        raise GuiCheckPlanError(
            f"{key} must not be null when present",
            reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            details={"field": key, "value_type": "NoneType"},
        )


def _optional_bool(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    _reject_present_null(payload, key)
    return _bool(payload[key], key)


def _optional_text(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _text(payload[key], key, required=False)


def _optional_identifier(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _identifier(payload[key], key, required=True)


def _unique_strings(
    value: Any,
    name: str,
    *,
    wire: bool,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        raise GuiCheckPlanError(
            f"{name} must be a JSON array; null is not a collection",
            reason_code=CheckPlanReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": "NoneType"},
        )
    sequence = (
        _require_json_array(value, name)
        if wire
        else _require_python_sequence(value, name)
    )
    items: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(sequence):
        text = _identifier(raw, f"{name}[{index}]", required=True)
        if text in seen:
            raise GuiCheckPlanError(
                f"{name} entries must be unique",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
                details={"field": name, "duplicate": text},
            )
        seen.add(text)
        items.append(text)
    if required and not items:
        raise GuiCheckPlanError(
            f"{name} must not be empty",
            reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            details={"field": name},
        )
    return tuple(items)


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    mapping = _require_mapping(value, "details")
    return MappingProxyType(dict(mapping))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _reject_command_injection(payload: Mapping[str, Any], noun: str) -> None:
    forbidden = sorted(set(payload) & set(_FORBIDDEN_REQUEST_KEYS))
    if not forbidden:
        forbidden = sorted(set(payload) & set(_NESTED_COMMAND_KEYS))
    if forbidden:
        if any(
            key in payload
            for key in (
                "suppress_fallback",
                "skip_fallback",
                "disable_fallback",
                "no_fallback",
            )
        ):
            raise GuiCheckPlanError(
                f"{noun} cannot suppress mandatory fallback",
                reason_code=CheckPlanReasonCode.FALLBACK_SUPPRESSION_FORBIDDEN.value,
                details={"noun": noun, "forbidden_fields": forbidden},
            )
        raise GuiCheckPlanError(
            f"{noun} cannot inject host commands",
            reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
            details={"noun": noun, "forbidden_fields": forbidden},
        )


def _looks_like_command_string(value: str) -> bool:
    if _COMMAND_META_RE.search(value):
        return True
    lowered = value.lower()
    return any(
        token in lowered
        for token in (
            "rm -rf",
            "/bin/sh",
            "/bin/bash",
            "python -c",
            "subprocess",
        )
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _registry_argv(check_id: str) -> tuple[str, ...]:
    return (
        HOST_PYTHON_EXECUTABLE,
        "-I",
        "-B",
        "-c",
        _DISPATCH_SCRIPT,
        check_id,
    )


@dataclass(frozen=True)
class RegisteredCheck:
    """One host-owned check.  Argv is fixed and never caller-supplied."""

    check_id: str
    family: CheckFamily
    risk_class: CheckRiskClass
    screenshot_scope: ScreenshotScope
    required_for_kinds: frozenset[str] = frozenset()
    always_required: bool = False

    @property
    def argv(self) -> tuple[str, ...]:
        return _registry_argv(self.check_id)

    @property
    def required(self) -> bool:
        return self.always_required or self.risk_class is CheckRiskClass.HARD


def _entry(
    check_id: str,
    family: CheckFamily,
    *,
    risk: CheckRiskClass = CheckRiskClass.HARD,
    screenshots: ScreenshotScope = ScreenshotScope.NONE,
    kinds: frozenset[str] | None = None,
    always_required: bool = False,
) -> RegisteredCheck:
    return RegisteredCheck(
        check_id=check_id,
        family=family,
        risk_class=risk,
        screenshot_scope=screenshots,
        required_for_kinds=kinds or frozenset(),
        always_required=always_required,
    )


_REGISTRY_ENTRIES: Final[tuple[RegisteredCheck, ...]] = (
    _entry(
        "check:direct-tests",
        CheckFamily.UNIT,
        kinds=frozenset({"component_implementation", "test"}),
        always_required=True,
    ),
    _entry(
        "check:contract-tests",
        CheckFamily.UNIT,
        kinds=frozenset({"props_event_contract"}),
        always_required=True,
    ),
    _entry(
        "check:invocation-tests",
        CheckFamily.UNIT,
        kinds=frozenset({"action_binding"}),
        always_required=True,
    ),
    _entry("check:test-artifacts", CheckFamily.UNIT, kinds=frozenset({"test"})),
    _entry(
        "check:capsule",
        CheckFamily.COMPONENT,
        kinds=frozenset({"component_implementation"}),
        always_required=True,
    ),
    _entry(
        "check:parents-consumers",
        CheckFamily.COMPONENT,
        kinds=frozenset({"props_event_contract"}),
    ),
    _entry(
        "check:interface-descriptors",
        CheckFamily.COMPONENT,
        kinds=frozenset({"props_event_contract"}),
    ),
    _entry(
        "check:action-bindings",
        CheckFamily.COMPONENT,
        kinds=frozenset({"props_event_contract", "action_binding"}),
    ),
    _entry(
        "check:accessibility-scenarios",
        CheckFamily.SCENARIO,
        kinds=frozenset({"component_implementation", "accessibility"}),
        always_required=True,
    ),
    _entry(
        "check:interaction-scenarios",
        CheckFamily.SCENARIO,
        kinds=frozenset({"state_machine", "action_binding"}),
        always_required=True,
    ),
    _entry(
        "check:locale-scenarios",
        CheckFamily.SCENARIO,
        kinds=frozenset({"localization"}),
    ),
    _entry(
        "check:accessible-name",
        CheckFamily.SCENARIO,
        kinds=frozenset({"localization", "accessibility"}),
    ),
    _entry(
        "check:accessibility-contracts",
        CheckFamily.SCENARIO,
        kinds=frozenset({"accessibility"}),
        always_required=True,
    ),
    _entry(
        "check:policy",
        CheckFamily.POLICY,
        kinds=frozenset({"action_binding"}),
        always_required=True,
    ),
    _entry(
        "check:confirmation",
        CheckFamily.POLICY,
        kinds=frozenset({"action_binding"}),
        always_required=True,
    ),
    _entry(
        "check:interaction",
        CheckFamily.INTERACTION,
        kinds=frozenset({"action_binding"}),
        always_required=True,
    ),
    _entry(
        "check:host-boundary",
        CheckFamily.HOST,
        kinds=frozenset({"action_binding"}),
        always_required=True,
    ),
    _entry(
        "check:containing-screenshots",
        CheckFamily.BROWSER,
        risk=CheckRiskClass.NEUTRAL,
        screenshots=ScreenshotScope.AFFECTED,
        kinds=frozenset({"component_implementation", "screenshot"}),
    ),
    _entry(
        "check:dependent-screenshots",
        CheckFamily.BROWSER,
        risk=CheckRiskClass.NEUTRAL,
        screenshots=ScreenshotScope.AFFECTED,
        kinds=frozenset({"css_design_token"}),
    ),
    _entry(
        "check:text-layout-screenshots",
        CheckFamily.BROWSER,
        risk=CheckRiskClass.NEUTRAL,
        screenshots=ScreenshotScope.AFFECTED,
        kinds=frozenset({"localization"}),
    ),
    _entry(
        "check:responsive",
        CheckFamily.BROWSER,
        risk=CheckRiskClass.HEURISTIC,
        screenshots=ScreenshotScope.AFFECTED,
        kinds=frozenset({"css_design_token"}),
    ),
    _entry(
        "check:contrast",
        CheckFamily.BROWSER,
        screenshots=ScreenshotScope.AFFECTED,
        kinds=frozenset({"css_design_token", "accessibility"}),
        always_required=True,
    ),
    _entry(
        "check:clipping",
        CheckFamily.BROWSER,
        risk=CheckRiskClass.HEURISTIC,
        screenshots=ScreenshotScope.AFFECTED,
        kinds=frozenset({"css_design_token"}),
    ),
    _entry(
        "check:overflow",
        CheckFamily.BROWSER,
        risk=CheckRiskClass.HEURISTIC,
        screenshots=ScreenshotScope.AFFECTED,
        kinds=frozenset({"css_design_token"}),
    ),
    _entry(
        "check:screenshot-artifacts",
        CheckFamily.BROWSER,
        risk=CheckRiskClass.NEUTRAL,
        screenshots=ScreenshotScope.AFFECTED,
        kinds=frozenset({"screenshot"}),
    ),
    _entry(
        "check:reachability",
        CheckFamily.BUILD,
        kinds=frozenset({"state_machine"}),
        always_required=True,
    ),
    _entry(
        "check:outcome",
        CheckFamily.BUILD,
        kinds=frozenset({"state_machine"}),
        always_required=True,
    ),
    _entry(
        "check:formal",
        CheckFamily.BUILD,
        kinds=frozenset({"state_machine"}),
        always_required=True,
    ),
    _entry(
        "check:broader-screen-fallback",
        CheckFamily.FALLBACK,
        screenshots=ScreenshotScope.NEVER_GLOBAL,
        always_required=True,
    ),
)

CHECK_REGISTRY: Final[Mapping[str, RegisteredCheck]] = MappingProxyType(
    {entry.check_id: entry for entry in _REGISTRY_ENTRIES}
)
REGISTERED_CHECK_IDS: Final[frozenset[str]] = frozenset(CHECK_REGISTRY)


def registry_argv(check_id: str) -> tuple[str, ...]:
    """Return the host-fixed argv for a registered check."""
    entry = CHECK_REGISTRY.get(check_id)
    if entry is None:
        raise GuiCheckPlanError(
            f"unknown check id: {check_id}",
            reason_code=CheckPlanReasonCode.UNKNOWN_CHECK_ID.value,
            details={"check_id": check_id},
        )
    return entry.argv


def require_registered(check_id: str) -> RegisteredCheck:
    entry = CHECK_REGISTRY.get(check_id)
    if entry is None:
        raise GuiCheckPlanError(
            f"unknown check id: {check_id}",
            reason_code=CheckPlanReasonCode.UNKNOWN_CHECK_ID.value,
            details={"check_id": check_id},
        )
    return entry


# ---------------------------------------------------------------------------
# Evaluator risk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluatorRiskClassification:
    """Closed evaluator risk facts consumed by the planner."""

    graph_confidence: str = ExtractionConfidence.EXACT.value
    risk_class: str = CheckRiskClass.HARD.value
    dynamic_behavior: bool = False
    shared_tokens: bool = False
    critical_evidence_unknown: bool = False
    hard_gate_regression: bool = False
    hard_gate_families: tuple[str, ...] = ()
    failed_check_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        confidence = _text(self.graph_confidence, "graph_confidence")
        if confidence not in {item.value for item in ExtractionConfidence}:
            raise GuiCheckPlanError(
                f"unknown graph_confidence: {confidence}",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
                details={"graph_confidence": confidence},
            )
        risk = _text(self.risk_class, "risk_class")
        if risk not in {item.value for item in CheckRiskClass}:
            raise GuiCheckPlanError(
                f"unknown risk_class: {risk}",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
                details={"risk_class": risk},
            )
        object.__setattr__(self, "graph_confidence", confidence)
        object.__setattr__(self, "risk_class", risk)
        object.__setattr__(
            self, "dynamic_behavior", _bool(self.dynamic_behavior, "dynamic_behavior")
        )
        object.__setattr__(
            self, "shared_tokens", _bool(self.shared_tokens, "shared_tokens")
        )
        object.__setattr__(
            self,
            "critical_evidence_unknown",
            _bool(self.critical_evidence_unknown, "critical_evidence_unknown"),
        )
        object.__setattr__(
            self,
            "hard_gate_regression",
            _bool(self.hard_gate_regression, "hard_gate_regression"),
        )
        families = _unique_strings(
            self.hard_gate_families, "hard_gate_families", wire=False
        )
        unknown_families = sorted(set(families) - HARD_GATE_FAMILIES)
        if unknown_families:
            raise GuiCheckPlanError(
                f"unknown hard_gate_families: {unknown_families}",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
                details={"unknown_families": unknown_families},
            )
        object.__setattr__(self, "hard_gate_families", families)
        failed = _unique_strings(self.failed_check_ids, "failed_check_ids", wire=False)
        unknown_failed = sorted(set(failed) - REGISTERED_CHECK_IDS)
        if unknown_failed:
            raise GuiCheckPlanError(
                f"unknown failed_check_ids: {unknown_failed}",
                reason_code=CheckPlanReasonCode.UNKNOWN_CHECK_ID.value,
                details={"unknown_check_ids": unknown_failed},
            )
        object.__setattr__(self, "failed_check_ids", failed)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvaluatorRiskClassification":
        payload = _require_mapping(raw, "evaluator_risk")
        _reject_command_injection(payload, "evaluator_risk")
        _reject_unknown(payload, _EVALUATOR_RISK_KEYS, "evaluator_risk")
        classification = _optional_text(payload, "classification")
        risk_class = _optional_text(payload, "risk_class") or classification or (
            CheckRiskClass.HARD.value
        )
        return cls(
            graph_confidence=_optional_text(payload, "graph_confidence")
            or ExtractionConfidence.EXACT.value,
            risk_class=risk_class,
            dynamic_behavior=_optional_bool(payload, "dynamic_behavior", False),
            shared_tokens=_optional_bool(payload, "shared_tokens", False),
            critical_evidence_unknown=_optional_bool(
                payload, "critical_evidence_unknown", False
            ),
            hard_gate_regression=_optional_bool(payload, "hard_gate_regression", False),
            hard_gate_families=_unique_strings(
                payload["hard_gate_families"], "hard_gate_families", wire=True
            )
            if "hard_gate_families" in payload
            else (),
            failed_check_ids=_unique_strings(
                payload["failed_check_ids"], "failed_check_ids", wire=True
            )
            if "failed_check_ids" in payload
            else (),
        )

    @classmethod
    def from_any(cls, value: Any) -> "EvaluatorRiskClassification":
        if type(value) is cls:
            return value
        if value is None:
            return cls()
        if type(value) is dict:
            return cls.from_mapping(value)
        raise GuiCheckPlanError(
            "evaluator_risk must be a JSON object",
            reason_code=CheckPlanReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"value_type": type(value).__name__},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_evidence_unknown": self.critical_evidence_unknown,
            "dynamic_behavior": self.dynamic_behavior,
            "failed_check_ids": list(self.failed_check_ids),
            "graph_confidence": self.graph_confidence,
            "hard_gate_families": list(self.hard_gate_families),
            "hard_gate_regression": self.hard_gate_regression,
            "risk_class": self.risk_class,
            "shared_tokens": self.shared_tokens,
        }


# ---------------------------------------------------------------------------
# Plan / receipt records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiCheckPlanEntry:
    """One selected registry check in ``GuiCheckPlan@1``."""

    check_id: str
    family: CheckFamily
    required: bool
    argv: tuple[str, ...]
    risk_class: CheckRiskClass
    screenshot_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        entry = require_registered(self.check_id)
        object.__setattr__(self, "check_id", entry.check_id)
        if type(self.family) is not CheckFamily:
            object.__setattr__(self, "family", CheckFamily(str(self.family)))
        if type(self.risk_class) is not CheckRiskClass:
            object.__setattr__(
                self, "risk_class", CheckRiskClass(str(self.risk_class))
            )
        object.__setattr__(self, "required", self.required is True)
        argv = tuple(_exact_str(item, "argv[]") for item in self.argv)
        if argv != entry.argv:
            raise GuiCheckPlanError(
                "check argv must match the host registry",
                reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"check_id": self.check_id, "argv": list(argv)},
            )
        object.__setattr__(self, "argv", argv)
        object.__setattr__(
            self,
            "screenshot_ids",
            _unique_strings(self.screenshot_ids, "screenshot_ids", wire=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "check_id": self.check_id,
            "family": self.family.value,
            "required": self.required,
            "risk_class": self.risk_class.value,
            "screenshot_ids": list(self.screenshot_ids),
        }


@dataclass(frozen=True)
class GuiCheckPlan:
    """Closed ``GuiCheckPlan@1`` produced by the planner."""

    plan_id: str
    change_set_id: str
    invalidation_plan_id: str
    selected_check_ids: tuple[str, ...]
    required_check_ids: tuple[str, ...]
    fallback_check_ids: tuple[str, ...]
    fallback_triggered: bool
    fallback_explanation: str
    screenshot_ids: tuple[str, ...]
    families: tuple[str, ...]
    entries: tuple[GuiCheckPlanEntry, ...]
    uncertainty_reasons: tuple[str, ...]
    confidence: str
    change_kinds: tuple[str, ...]
    acceptance_policy: str = ACCEPTANCE_POLICY
    interface: str = GUI_CHECK_PLAN_INTERFACE
    schema: str = GUI_CHECK_PLAN_SCHEMA
    disposition: PlanDisposition = PlanDisposition.READY
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "change_set_id", _identifier(self.change_set_id, "change_set_id")
        )
        object.__setattr__(
            self,
            "invalidation_plan_id",
            _identifier(self.invalidation_plan_id, "invalidation_plan_id"),
        )
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self,
            "acceptance_policy",
            _text(self.acceptance_policy, "acceptance_policy"),
        )
        if self.acceptance_policy != ACCEPTANCE_POLICY:
            raise GuiCheckPlanError(
                "acceptance_policy is fixed by the host",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
                details={"acceptance_policy": self.acceptance_policy},
            )
        object.__setattr__(
            self,
            "fallback_triggered",
            _bool(self.fallback_triggered, "fallback_triggered"),
        )
        object.__setattr__(
            self,
            "fallback_explanation",
            _text(self.fallback_explanation, "fallback_explanation", required=False),
        )
        confidence = _text(self.confidence, "confidence")
        if confidence not in {item.value for item in ExtractionConfidence}:
            raise GuiCheckPlanError(
                f"unknown confidence: {confidence}",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            )
        object.__setattr__(self, "confidence", confidence)
        if type(self.disposition) is not PlanDisposition:
            object.__setattr__(
                self, "disposition", PlanDisposition(str(self.disposition))
            )
        codes = tuple(
            _identifier(code, "reason_codes[]") for code in (self.reason_codes or ())
        )
        object.__setattr__(self, "reason_codes", codes)

    @property
    def required_ids(self) -> frozenset[str]:
        return frozenset(self.required_check_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_policy": self.acceptance_policy,
            "change_kinds": list(self.change_kinds),
            "change_set_id": self.change_set_id,
            "confidence": self.confidence,
            "disposition": self.disposition.value,
            "entries": [entry.to_dict() for entry in self.entries],
            "fallback_check_ids": list(self.fallback_check_ids),
            "fallback_explanation": self.fallback_explanation,
            "fallback_triggered": self.fallback_triggered,
            "families": list(self.families),
            "interface": self.interface,
            "invalidation_plan_id": self.invalidation_plan_id,
            "plan_id": self.plan_id,
            "reason_codes": list(self.reason_codes),
            "required_check_ids": list(self.required_check_ids),
            "schema": self.schema,
            "screenshot_ids": list(self.screenshot_ids),
            "selected_check_ids": list(self.selected_check_ids),
            "uncertainty_reasons": list(self.uncertainty_reasons),
        }


@dataclass(frozen=True)
class GuiCheckResult:
    """One executed check inside ``GuiCheckExecutionReceipt@1``."""

    check_id: str
    family: CheckFamily
    required: bool
    status: CheckStatus
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        entry = require_registered(self.check_id)
        object.__setattr__(self, "check_id", entry.check_id)
        if type(self.family) is not CheckFamily:
            object.__setattr__(self, "family", CheckFamily(str(self.family)))
        if type(self.status) is not CheckStatus:
            object.__setattr__(self, "status", CheckStatus(str(self.status)))
        object.__setattr__(self, "required", self.required is True)
        argv = tuple(_exact_str(item, "argv[]") for item in self.argv)
        if argv != entry.argv:
            raise GuiCheckPlanError(
                "executed argv must match the host registry",
                reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"check_id": self.check_id},
            )
        object.__setattr__(self, "argv", argv)
        if type(self.returncode) is not int or type(self.returncode) is bool:
            raise GuiCheckPlanError(
                "returncode must be an integer",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            )
        object.__setattr__(self, "stdout", str(self.stdout or ""))
        object.__setattr__(self, "stderr", str(self.stderr or ""))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_identifier(code, "reason_codes[]") for code in self.reason_codes),
        )

    @property
    def failed_required(self) -> bool:
        return self.required and self.status in {
            CheckStatus.FAILED,
            CheckStatus.ERROR,
            CheckStatus.SKIPPED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "check_id": self.check_id,
            "family": self.family.value,
            "reason_codes": list(self.reason_codes),
            "required": self.required,
            "returncode": self.returncode,
            "status": self.status.value,
            "stderr": self.stderr,
            "stdout": self.stdout,
        }


@dataclass(frozen=True)
class GuiCheckExecutionReceipt:
    """Typed ``GuiCheckExecutionReceipt@1``."""

    plan_id: str
    receipt_id: str
    disposition: PlanDisposition
    acceptance_blocked: bool
    executed_check_ids: tuple[str, ...]
    failed_required_check_ids: tuple[str, ...]
    check_results: tuple[GuiCheckResult, ...]
    fallback_applied: bool
    fallback_explanation: str
    screenshot_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    interface: str = GUI_CHECK_EXECUTION_RECEIPT_INTERFACE
    schema: str = GUI_CHECK_EXECUTION_RECEIPT_SCHEMA
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "receipt_id", _identifier(self.receipt_id, "receipt_id")
        )
        if type(self.disposition) is not PlanDisposition:
            object.__setattr__(
                self, "disposition", PlanDisposition(str(self.disposition))
            )
        object.__setattr__(
            self, "acceptance_blocked", self.acceptance_blocked is True
        )
        if self.acceptance_blocked and self.disposition is PlanDisposition.EXECUTED:
            object.__setattr__(self, "disposition", PlanDisposition.BLOCKED)
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(
            self,
            "fallback_applied",
            _bool(self.fallback_applied, "fallback_applied"),
        )
        object.__setattr__(
            self,
            "fallback_explanation",
            _text(self.fallback_explanation, "fallback_explanation", required=False),
        )
        object.__setattr__(
            self,
            "details",
            _freeze_mapping(dict(self.details) if self.details is not None else {}),
        )

    @property
    def blocked(self) -> bool:
        return self.acceptance_blocked or self.disposition is PlanDisposition.BLOCKED

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_blocked": self.acceptance_blocked,
            "blocked": self.blocked,
            "check_results": [item.to_dict() for item in self.check_results],
            "details": dict(self.details),
            "disposition": self.disposition.value,
            "executed_check_ids": list(self.executed_check_ids),
            "failed_required_check_ids": list(self.failed_required_check_ids),
            "fallback_applied": self.fallback_applied,
            "fallback_explanation": self.fallback_explanation,
            "interface": self.interface,
            "plan_id": self.plan_id,
            "reason_codes": list(self.reason_codes),
            "receipt_id": self.receipt_id,
            "schema": self.schema,
            "screenshot_ids": list(self.screenshot_ids),
        }


# ---------------------------------------------------------------------------
# Host-fixed check runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostCheckResult:
    """Captured result of one host-fixed check argv."""

    check_id: str
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def status(self) -> CheckStatus:
        if self.returncode == 0:
            return CheckStatus.PASSED
        if self.returncode in {127, 78}:
            return CheckStatus.SKIPPED
        if self.returncode < 0:
            return CheckStatus.ERROR
        return CheckStatus.FAILED


def sealed_check_environment() -> dict[str, str]:
    """Return the host-fixed environment used for every check subprocess."""
    env = {
        "PATH": HOST_VALIDATION_PATH,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    home = os.environ.get("HOME")
    if type(home) is str and home:
        env["HOME"] = home
        env["XDG_CACHE_HOME"] = f"{home}/.cache"
        env["XDG_CONFIG_HOME"] = f"{home}/.config"
        env["XDG_DATA_HOME"] = f"{home}/.local/share"
        env["XDG_STATE_HOME"] = f"{home}/.local/state"
    tmpdir = os.environ.get("TMPDIR")
    if type(tmpdir) is str and tmpdir:
        env["TMPDIR"] = tmpdir
    return env


class CheckRunner(Protocol):
    """Execute one registered check.  Implementations must not accept argv."""

    def run(self, check_id: str) -> HostCheckResult:
        """Run the registry argv for ``check_id``."""


@dataclass(frozen=True)
class HostCheckRunner:
    """Execute a closed check-id set with a host-fixed interpreter.

    Callers pass only a registered check id.  The runner looks up argv
    from ``CHECK_REGISTRY`` and never uses a shell.
    """

    executable: str = HOST_PYTHON_EXECUTABLE
    timeout_seconds: float = 30.0
    scripted_results: Mapping[str, HostCheckResult] | None = None

    def __post_init__(self) -> None:
        executable = _text(self.executable, "executable")
        if executable != HOST_PYTHON_EXECUTABLE:
            raise GuiCheckPlanError(
                "check executable is fixed by the host",
                reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"executable": executable},
            )
        object.__setattr__(self, "executable", executable)
        timeout = self.timeout_seconds
        if type(timeout) is not float and type(timeout) is not int:
            raise GuiCheckPlanError(
                "timeout_seconds must be a number",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            )
        if float(timeout) <= 0:
            raise GuiCheckPlanError(
                "timeout_seconds must be positive",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            )
        object.__setattr__(self, "timeout_seconds", float(timeout))
        if self.scripted_results is not None:
            object.__setattr__(
                self, "scripted_results", MappingProxyType(dict(self.scripted_results))
            )

    def validate_argv(self, argv: Sequence[str]) -> None:
        if type(argv) is not list and type(argv) is not tuple:
            raise GuiCheckPlanError(
                "check argv must be a sequence of strings",
                reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"value_type": type(argv).__name__},
            )
        tokens = tuple(_exact_str(item, f"argv[{index}]") for index, item in enumerate(argv))
        if not tokens:
            raise GuiCheckPlanError(
                "check argv must not be empty",
                reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
            )
        if tokens[0] != HOST_PYTHON_EXECUTABLE:
            raise GuiCheckPlanError(
                "check executable is fixed by the host",
                reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"executable": tokens[0]},
            )
        for token in tokens:
            if _looks_like_command_string(token) and token != _DISPATCH_SCRIPT:
                raise GuiCheckPlanError(
                    "check argv contains a forbidden command string",
                    reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
                    details={"token": token},
                )
        allowed = {entry.argv for entry in CHECK_REGISTRY.values()}
        if tokens not in allowed:
            raise GuiCheckPlanError(
                "check argv is not in the host registry",
                reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
                details={"argv": list(tokens)},
            )

    def run(self, check_id: str) -> HostCheckResult:
        entry = require_registered(check_id)
        self.validate_argv(entry.argv)
        if self.scripted_results is not None:
            scripted = self.scripted_results.get(check_id)
            if scripted is None:
                return HostCheckResult(
                    check_id=check_id,
                    argv=entry.argv,
                    returncode=127,
                    stdout="",
                    stderr=f"{check_id} has no scripted result",
                )
            if tuple(scripted.argv) != entry.argv:
                raise GuiCheckPlanError(
                    "scripted argv must match the host registry",
                    reason_code=CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value,
                    details={"check_id": check_id},
                )
            return scripted
        try:
            completed = subprocess.run(
                entry.argv,
                env=sealed_check_environment(),
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except FileNotFoundError as exc:
            return HostCheckResult(
                check_id=check_id,
                argv=entry.argv,
                returncode=127,
                stdout="",
                stderr=str(exc),
            )
        except subprocess.TimeoutExpired:
            return HostCheckResult(
                check_id=check_id,
                argv=entry.argv,
                returncode=124,
                stdout="",
                stderr="check operation timed out",
            )
        except OSError as exc:
            return HostCheckResult(
                check_id=check_id,
                argv=entry.argv,
                returncode=1,
                stdout="",
                stderr=str(exc),
            )
        return HostCheckResult(
            check_id=check_id,
            argv=entry.argv,
            returncode=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


# ---------------------------------------------------------------------------
# Planning request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiCheckPlanRequest:
    """Closed planner input.  Callers cannot name argv or suppress fallback."""

    invalidation: PatchScopeInvalidationRecord
    evaluator_risk: EvaluatorRiskClassification
    change_kinds: tuple[str, ...] = ()
    affected_screenshot_ids: tuple[str, ...] = ()
    known_screenshot_ids: tuple[str, ...] = ()
    unrelated_screenshot_ids: tuple[str, ...] = ()
    affected_component_ids: tuple[str, ...] = ()
    application_id: str = ""
    screen_id: str = ""
    prior_failures: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GuiCheckPlanRequest":
        payload = _require_mapping(raw, "request")
        _reject_command_injection(payload, "request")
        _reject_unknown(payload, _REQUEST_KEYS, "request")
        if "invalidation" not in payload:
            raise GuiCheckPlanError(
                "request.invalidation is required",
                reason_code=CheckPlanReasonCode.MISSING_INVALIDATION_RECORD.value,
                details={"field": "invalidation"},
            )
        _reject_present_null(payload, "invalidation")
        if "evaluator_risk" in payload:
            _reject_present_null(payload, "evaluator_risk")
        if "proposal" in payload:
            _reject_present_null(payload, "proposal")
            proposal = _require_mapping(payload["proposal"], "proposal")
            _reject_command_injection(proposal, "proposal")
            _reject_unknown(proposal, _PROPOSAL_ALLOWED_KEYS, "proposal")
        if "browser_input" in payload:
            _reject_present_null(payload, "browser_input")
            browser = _require_mapping(payload["browser_input"], "browser_input")
            _reject_command_injection(browser, "browser_input")
            decision = GuiHostBoundaryPolicy().evaluate(browser)
            if not decision.allowed:
                raise GuiCheckPlanError(
                    "browser input cannot select host commands or paths",
                    reason_code=CheckPlanReasonCode.BROWSER_COMMAND_FORBIDDEN.value,
                    details={"reason_codes": list(decision.reason_codes)},
                )
        change_kinds = (
            _unique_strings(payload["change_kinds"], "change_kinds", wire=True)
            if "change_kinds" in payload
            else ()
        )
        unknown_kinds = sorted(set(change_kinds) - KNOWN_CHANGE_KINDS)
        if unknown_kinds:
            raise GuiCheckPlanError(
                f"unknown change_kinds: {unknown_kinds}",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
                details={"unknown_kinds": unknown_kinds},
            )
        return cls(
            invalidation=PatchScopeInvalidationRecord.from_any(payload["invalidation"]),
            evaluator_risk=EvaluatorRiskClassification.from_any(
                payload.get("evaluator_risk")
            ),
            change_kinds=change_kinds,
            affected_screenshot_ids=_unique_strings(
                payload["affected_screenshot_ids"],
                "affected_screenshot_ids",
                wire=True,
            )
            if "affected_screenshot_ids" in payload
            else (),
            known_screenshot_ids=_unique_strings(
                payload["known_screenshot_ids"], "known_screenshot_ids", wire=True
            )
            if "known_screenshot_ids" in payload
            else (),
            unrelated_screenshot_ids=_unique_strings(
                payload["unrelated_screenshot_ids"],
                "unrelated_screenshot_ids",
                wire=True,
            )
            if "unrelated_screenshot_ids" in payload
            else (),
            affected_component_ids=_unique_strings(
                payload["affected_component_ids"],
                "affected_component_ids",
                wire=True,
            )
            if "affected_component_ids" in payload
            else (),
            application_id=_optional_identifier(payload, "application_id"),
            screen_id=_optional_identifier(payload, "screen_id"),
            prior_failures=_unique_strings(
                payload["prior_failures"], "prior_failures", wire=True
            )
            if "prior_failures" in payload
            else (),
        )

    @classmethod
    def from_any(cls, value: Any) -> "GuiCheckPlanRequest":
        if type(value) is cls:
            return value
        if type(value) is dict:
            return cls.from_mapping(value)
        raise GuiCheckPlanError(
            "request must be a GuiCheckPlanRequest or JSON object",
            reason_code=CheckPlanReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"value_type": type(value).__name__},
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _infer_change_kinds(
    request: GuiCheckPlanRequest,
) -> tuple[str, ...]:
    kinds: list[str] = []
    seen: set[str] = set()
    for kind in request.change_kinds:
        if kind not in seen:
            seen.add(kind)
            kinds.append(kind)
    for reason in request.invalidation.reasons:
        mapped = REASON_TO_CHANGE_KIND.get(reason)
        if mapped and mapped not in seen:
            seen.add(mapped)
            kinds.append(mapped)
    if not kinds:
        kinds.append("other")
    return tuple(kinds)


def _worst_confidence(*values: str) -> str:
    rank = {
        ExtractionConfidence.EXACT.value: 0,
        ExtractionConfidence.CONSERVATIVE.value: 1,
        ExtractionConfidence.HEURISTIC.value: 2,
        ExtractionConfidence.OPAQUE.value: 3,
    }
    worst = ExtractionConfidence.EXACT.value
    worst_rank = 0
    for value in values:
        text = value or ExtractionConfidence.EXACT.value
        score = rank.get(text, 3)
        if score > worst_rank:
            worst = text if text in rank else ExtractionConfidence.OPAQUE.value
            worst_rank = score
    return worst


def _detect_uncertainty(
    request: GuiCheckPlanRequest,
    kinds: Sequence[str],
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    reasons: list[str] = []
    explanations: list[str] = []
    invalidation = request.invalidation
    risk = request.evaluator_risk

    if invalidation.fallback_triggered:
        reasons.append(CheckPlanReasonCode.FALLBACK_EXPANDED.value)
        explanations.append(
            invalidation.fallback_explanation
            or "invalidation requested broader fallback"
        )
    invalidation_confidence = invalidation.confidence or ExtractionConfidence.EXACT.value
    if invalidation_confidence in UNCERTAIN_CONFIDENCES:
        reasons.append(CheckPlanReasonCode.UNCERTAIN_GRAPH_CONFIDENCE.value)
        explanations.append(
            f"invalidation confidence is {invalidation_confidence}"
        )
    if risk.graph_confidence in UNCERTAIN_CONFIDENCES:
        reasons.append(CheckPlanReasonCode.UNCERTAIN_GRAPH_CONFIDENCE.value)
        explanations.append(f"graph confidence is {risk.graph_confidence}")
    for reason in invalidation.reasons:
        if reason == "missing_edge":
            reasons.append(CheckPlanReasonCode.MISSING_EDGE.value)
            explanations.append("typed dependency edges are missing")
        elif reason == "stale_edge":
            reasons.append(CheckPlanReasonCode.STALE_EDGE.value)
            explanations.append("typed dependency edges are stale")
        elif reason == "opaque_edge":
            reasons.append(CheckPlanReasonCode.OPAQUE_EDGE.value)
            explanations.append("typed dependency edges are opaque")
        elif reason == "fallback_expansion":
            reasons.append(CheckPlanReasonCode.FALLBACK_EXPANDED.value)
            explanations.append("invalidation already expanded fallback")
    if risk.dynamic_behavior:
        reasons.append(CheckPlanReasonCode.DYNAMIC_BEHAVIOR.value)
        explanations.append("dynamic behavior requires broader interaction/host checks")
    if risk.shared_tokens:
        reasons.append(CheckPlanReasonCode.SHARED_TOKENS.value)
        explanations.append("shared design tokens require broader visual fallback")
    if risk.critical_evidence_unknown:
        reasons.append(CheckPlanReasonCode.CRITICAL_EVIDENCE_UNKNOWN.value)
        explanations.append("unknown critical evidence cannot pretend precision")
    if risk.hard_gate_regression:
        reasons.append(CheckPlanReasonCode.HARD_GATE_REGRESSION.value)
        explanations.append("hard-gate regression requires broader verification")
    if risk.failed_check_ids or request.prior_failures:
        reasons.append(CheckPlanReasonCode.PRIOR_FAILURE.value)
        explanations.append("prior required-check failures expand fallback")
    if "other" in kinds:
        reasons.append(CheckPlanReasonCode.FALLBACK_EXPANDED.value)
        explanations.append("untyped change kind requires broader fallback")

    unique_reasons: list[str] = []
    seen: set[str] = set()
    for item in reasons:
        if item not in seen:
            seen.add(item)
            unique_reasons.append(item)
    unique_explanations: list[str] = []
    seen_exp: set[str] = set()
    for item in explanations:
        if item and item not in seen_exp:
            seen_exp.add(item)
            unique_explanations.append(item)
    return bool(unique_reasons), tuple(unique_reasons), tuple(unique_explanations)


def _scoped_screenshots(request: GuiCheckPlanRequest) -> tuple[str, ...]:
    unrelated = set(request.unrelated_screenshot_ids)
    affected = [
        item for item in request.affected_screenshot_ids if item not in unrelated
    ]
    # Local precision: never take the global inventory just because it exists.
    # Ownership unknown → leave screenshots empty rather than invalidate all.
    return tuple(affected)


def _is_required(
    entry: RegisteredCheck,
    kinds: Sequence[str],
    *,
    fallback: bool,
    risk: EvaluatorRiskClassification,
) -> bool:
    if entry.always_required and (
        entry.family in DIRECT_FAMILIES
        or entry.check_id in ACTION_REQUIRED_CHECKS
        or (fallback and entry.family is CheckFamily.FALLBACK)
        or any(kind in entry.required_for_kinds for kind in kinds)
    ):
        return True
    if entry.check_id in ACTION_REQUIRED_CHECKS and any(
        kind in ACTION_CHANGE_KINDS for kind in kinds
    ):
        return True
    if fallback and entry.check_id == "check:broader-screen-fallback":
        return True
    if entry.risk_class is CheckRiskClass.HARD and any(
        kind in entry.required_for_kinds for kind in kinds
    ):
        return True
    for family in risk.hard_gate_families:
        if entry.check_id in HARD_GATE_TO_CHECKS.get(family, ()):
            return True
    return False


def _order_check_ids(check_ids: Sequence[str]) -> tuple[str, ...]:
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for check_id in check_ids:
        if check_id in seen:
            continue
        seen.add(check_id)
        entry = require_registered(check_id)
        ranked.append((FAMILY_ORDER.index(entry.family), check_id))
    ranked.sort()
    return tuple(item[1] for item in ranked)


def _deterministic_plan_id(
    change_set_id: str,
    selected: Sequence[str],
    screenshots: Sequence[str],
    fallback: bool,
    confidence: str,
) -> str:
    body = _canonical_json(
        {
            "change_set_id": change_set_id,
            "confidence": confidence,
            "fallback_triggered": fallback,
            "planner": GUI_AFFECTED_CHECK_PLANNER_VERSION,
            "screenshot_ids": list(screenshots),
            "selected_check_ids": list(selected),
        }
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", change_set_id).strip("-")[:48] or "plan"
    return f"checkplan:{slug}:{digest}"


def _build_entries(
    selected: Sequence[str],
    kinds: Sequence[str],
    *,
    fallback: bool,
    risk: EvaluatorRiskClassification,
    screenshots: Sequence[str],
) -> tuple[GuiCheckPlanEntry, ...]:
    entries: list[GuiCheckPlanEntry] = []
    for check_id in selected:
        spec = require_registered(check_id)
        shot_ids: tuple[str, ...] = ()
        if spec.screenshot_scope is ScreenshotScope.AFFECTED:
            shot_ids = tuple(screenshots)
        entries.append(
            GuiCheckPlanEntry(
                check_id=spec.check_id,
                family=spec.family,
                required=_is_required(spec, kinds, fallback=fallback, risk=risk),
                argv=spec.argv,
                risk_class=spec.risk_class,
                screenshot_ids=shot_ids,
            )
        )
    return tuple(entries)


# ---------------------------------------------------------------------------
# GuiAffectedCheckPlanner@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiAffectedCheckPlanner:
    """Select affected checks and expand fallback when precision is absent.

    Interface: ``GuiAffectedCheckPlanner@1``.
    """

    runner: HostCheckRunner = field(default_factory=HostCheckRunner)
    interface: str = GUI_AFFECTED_CHECK_PLANNER_INTERFACE
    schema: str = GUI_AFFECTED_CHECK_PLANNER_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.runner, HostCheckRunner):
            raise GuiCheckPlanError(
                "runner must be a HostCheckRunner",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
                details={"value_type": type(self.runner).__name__},
            )
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))

    def plan(self, request: GuiCheckPlanRequest | Mapping[str, Any]) -> GuiCheckPlan:
        """Select and order registry checks for one invalidation plan."""
        normalized = GuiCheckPlanRequest.from_any(request)
        kinds = _infer_change_kinds(normalized)
        fallback, uncertainty_reasons, explanations = _detect_uncertainty(
            normalized, kinds
        )
        selected: list[str] = []
        seen: set[str] = set()

        def _add(check_id: str) -> None:
            if check_id in seen:
                return
            require_registered(check_id)
            seen.add(check_id)
            selected.append(check_id)

        for check_id in normalized.invalidation.affected_check_ids:
            _add(check_id)
        for kind in kinds:
            for check_id in KIND_DEFAULT_CHECKS.get(kind, ()):
                _add(check_id)
        if any(kind in ACTION_CHANGE_KINDS for kind in kinds) or any(
            reason == "action_changed" for reason in normalized.invalidation.reasons
        ):
            for check_id in ACTION_REQUIRED_CHECKS:
                _add(check_id)
        for family in normalized.evaluator_risk.hard_gate_families:
            for check_id in HARD_GATE_TO_CHECKS.get(family, ()):
                _add(check_id)
        if normalized.evaluator_risk.shared_tokens:
            for check_id in SHARED_TOKEN_CHECKS:
                _add(check_id)
        if normalized.evaluator_risk.dynamic_behavior:
            for check_id in DYNAMIC_BEHAVIOR_CHECKS:
                _add(check_id)
        if fallback:
            for check_id in FALLBACK_EXPANSION_CHECKS:
                _add(check_id)

        ordered = _order_check_ids(selected)
        screenshots = _scoped_screenshots(normalized)
        if not screenshots and (
            normalized.known_screenshot_ids or normalized.unrelated_screenshot_ids
        ):
            extra_reasons = (
                CheckPlanReasonCode.LOCAL_SCREENSHOT_PRECISION.value,
                CheckPlanReasonCode.UNRELATED_SCREENSHOT_EXCLUDED.value,
            )
        elif normalized.unrelated_screenshot_ids:
            extra_reasons = (CheckPlanReasonCode.UNRELATED_SCREENSHOT_EXCLUDED.value,)
        elif any(kind in LOCAL_CHANGE_KINDS for kind in kinds):
            extra_reasons = (CheckPlanReasonCode.LOCAL_SCREENSHOT_PRECISION.value,)
        else:
            extra_reasons = ()

        reason_codes = [
            CheckPlanReasonCode.PLANNED.value,
            *uncertainty_reasons,
            *extra_reasons,
        ]
        if any(kind in ACTION_CHANGE_KINDS for kind in kinds):
            reason_codes.append(CheckPlanReasonCode.ACTION_POLICY_REQUIRED.value)
        unique_codes: list[str] = []
        seen_codes: set[str] = set()
        for code in reason_codes:
            if code not in seen_codes:
                seen_codes.add(code)
                unique_codes.append(code)

        confidence = _worst_confidence(
            normalized.invalidation.confidence or ExtractionConfidence.EXACT.value,
            normalized.evaluator_risk.graph_confidence,
            ExtractionConfidence.CONSERVATIVE.value if fallback else ExtractionConfidence.EXACT.value,
        )
        entries = _build_entries(
            ordered,
            kinds,
            fallback=fallback,
            risk=normalized.evaluator_risk,
            screenshots=screenshots,
        )
        required = tuple(entry.check_id for entry in entries if entry.required)
        fallback_ids = tuple(
            entry.check_id
            for entry in entries
            if entry.family is CheckFamily.FALLBACK
            or entry.check_id in FALLBACK_EXPANSION_CHECKS
            and fallback
        )
        families = tuple(
            dict.fromkeys(entry.family.value for entry in entries)
        )
        explanation = (
            "; ".join(explanations)
            if explanations
            else (
                "Uncertainty in typed dependency closure requires broader screen-scoped fallback"
                if fallback
                else "No uncertainty requires broad fallback."
            )
        )
        plan_id = _deterministic_plan_id(
            normalized.invalidation.change_set_id,
            ordered,
            screenshots,
            fallback,
            confidence,
        )
        return GuiCheckPlan(
            plan_id=plan_id,
            change_set_id=normalized.invalidation.change_set_id,
            invalidation_plan_id=normalized.invalidation.plan_id,
            selected_check_ids=ordered,
            required_check_ids=required,
            fallback_check_ids=fallback_ids,
            fallback_triggered=fallback,
            fallback_explanation=explanation,
            screenshot_ids=screenshots,
            families=families,
            entries=entries,
            uncertainty_reasons=uncertainty_reasons,
            confidence=confidence,
            change_kinds=kinds,
            reason_codes=tuple(unique_codes),
        )

    def execute(
        self,
        request: GuiCheckPlanRequest | Mapping[str, Any] | GuiCheckPlan,
        *,
        runner: HostCheckRunner | None = None,
    ) -> GuiCheckExecutionReceipt:
        """Run the planned registry checks.  Required failures block acceptance."""
        if type(request) is GuiCheckPlan:
            plan = request
        else:
            plan = self.plan(request)
        active = runner if runner is not None else self.runner
        if not isinstance(active, HostCheckRunner):
            raise GuiCheckPlanError(
                "runner must be a HostCheckRunner",
                reason_code=CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value,
            )

        results: list[GuiCheckResult] = []
        executed: list[str] = []
        failed_required: list[str] = []
        seen: set[str] = set()
        pending = list(plan.entries)
        fallback_applied = plan.fallback_triggered

        while pending:
            entry = pending.pop(0)
            if entry.check_id in seen:
                continue
            seen.add(entry.check_id)
            raw = active.run(entry.check_id)
            status = raw.status
            codes: list[str] = []
            if status is CheckStatus.PASSED:
                codes.append(CheckPlanReasonCode.EXECUTED.value)
            elif status is CheckStatus.SKIPPED:
                codes.append(CheckPlanReasonCode.CHECK_UNAVAILABLE.value)
            else:
                codes.append(CheckPlanReasonCode.REQUIRED_CHECK_FAILED.value)
            result = GuiCheckResult(
                check_id=entry.check_id,
                family=entry.family,
                required=entry.required,
                status=status,
                argv=entry.argv,
                returncode=raw.returncode,
                stdout=raw.stdout,
                stderr=raw.stderr,
                reason_codes=tuple(codes),
            )
            results.append(result)
            executed.append(entry.check_id)
            if result.failed_required:
                failed_required.append(entry.check_id)
                if not fallback_applied:
                    fallback_applied = True
                    extra_ids = [
                        check_id
                        for check_id in FALLBACK_EXPANSION_CHECKS
                        if check_id not in seen
                    ]
                    extra_entries = _build_entries(
                        _order_check_ids(extra_ids),
                        plan.change_kinds,
                        fallback=True,
                        risk=EvaluatorRiskClassification(),
                        screenshots=plan.screenshot_ids,
                    )
                    pending.extend(extra_entries)

        blocked = bool(failed_required)
        reason_codes = [CheckPlanReasonCode.EXECUTED.value]
        if fallback_applied:
            reason_codes.append(CheckPlanReasonCode.FALLBACK_EXPANDED.value)
        if blocked:
            reason_codes.append(CheckPlanReasonCode.REQUIRED_CHECK_FAILED.value)
            reason_codes.append(CheckPlanReasonCode.ACCEPTANCE_BLOCKED.value)
        else:
            reason_codes.append(CheckPlanReasonCode.ACCEPTANCE_ALLOWED.value)
        receipt_body = _canonical_json(
            {
                "executed_check_ids": executed,
                "failed_required_check_ids": failed_required,
                "plan_id": plan.plan_id,
            }
        )
        receipt_id = "checkreceipt:" + hashlib.sha256(
            receipt_body.encode("utf-8")
        ).hexdigest()[:32]
        return GuiCheckExecutionReceipt(
            plan_id=plan.plan_id,
            receipt_id=receipt_id,
            disposition=(
                PlanDisposition.BLOCKED if blocked else PlanDisposition.EXECUTED
            ),
            acceptance_blocked=blocked,
            executed_check_ids=tuple(executed),
            failed_required_check_ids=tuple(failed_required),
            check_results=tuple(results),
            fallback_applied=fallback_applied,
            fallback_explanation=(
                plan.fallback_explanation
                if plan.fallback_triggered
                else (
                    "required check failure expanded broader fallback"
                    if fallback_applied and not plan.fallback_triggered
                    else plan.fallback_explanation
                )
            ),
            screenshot_ids=plan.screenshot_ids,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            details={
                "acceptance_policy": ACCEPTANCE_POLICY,
                "selected_check_ids": list(plan.selected_check_ids),
            },
        )

    def plan_request(self, request: Mapping[str, Any]) -> GuiCheckPlan:
        return self.plan(request)

    def execute_request(
        self, request: Mapping[str, Any], *, runner: HostCheckRunner | None = None
    ) -> GuiCheckExecutionReceipt:
        return self.execute(request, runner=runner)


def default_affected_check_planner() -> GuiAffectedCheckPlanner:
    """Return the host-owned affected-check planner."""
    return GuiAffectedCheckPlanner()


__all__ = (
    "ACCEPTANCE_POLICY",
    "ACTION_REQUIRED_CHECKS",
    "CHECK_REGISTRY",
    "CheckFamily",
    "CheckPlanReasonCode",
    "CheckRiskClass",
    "CheckStatus",
    "DIRECT_FAMILIES",
    "EvaluatorRiskClassification",
    "EXPANSION_FAMILIES",
    "ExtractionConfidence",
    "FAMILY_ORDER",
    "FALLBACK_EXPANSION_CHECKS",
    "GUI_AFFECTED_CHECK_PLANNER_INTERFACE",
    "GUI_AFFECTED_CHECK_PLANNER_SCHEMA",
    "GUI_AFFECTED_CHECK_PLANNER_VERSION",
    "GUI_CHECK_EXECUTION_RECEIPT_INTERFACE",
    "GUI_CHECK_EXECUTION_RECEIPT_SCHEMA",
    "GUI_CHECK_PLAN_INTERFACE",
    "GUI_CHECK_PLAN_SCHEMA",
    "GUI_INVALIDATION_PLAN_INTERFACE",
    "GUI_INVALIDATION_PLAN_SCHEMA",
    "GuiAffectedCheckPlanner",
    "GuiCheckExecutionReceipt",
    "GuiCheckPlan",
    "GuiCheckPlanEntry",
    "GuiCheckPlanError",
    "GuiCheckPlanRequest",
    "GuiCheckResult",
    "HOST_PYTHON_EXECUTABLE",
    "HOST_VALIDATION_PATH",
    "HostCheckResult",
    "HostCheckRunner",
    "PlanDisposition",
    "REGISTERED_CHECK_IDS",
    "RegisteredCheck",
    "ScreenshotScope",
    "default_affected_check_planner",
    "registry_argv",
    "require_registered",
    "sealed_check_environment",
)
