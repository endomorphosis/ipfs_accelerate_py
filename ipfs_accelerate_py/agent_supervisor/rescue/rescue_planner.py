"""Exhaustion-gated, proposal-only LLM rescue planning.

The planner in this module is deliberately a narrow trust boundary.  It never
executes a recovery action and it never accepts source code, shell commands, or
open-ended paths from a model.  A provider can only select operations from a
closed catalog after the deterministic recovery controller has emitted a fresh,
incident-bound exhaustion receipt.

Provider imports are lazy so importing this module does not load ``llm_router``
or any model runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..prompt.prompt_workflow import (
    PROGRAMMATIC_RECOVERY_EXHAUSTION_SCHEMA,
    RESCUE_ACTION_SCHEMA,
    RESCUE_PLAN_SCHEMA,
    ProgrammaticRecoveryExhaustionReceipt,
    PromptWorkflowContractError,
    RecordStatus,
    RescueAction,
    RescueOperation,
    RescuePlan,
    SupervisorIncident,
)


RESCUE_PLAN_RESPONSE_NAME = "RescuePlan/v1"
DEFAULT_RESCUE_MODEL = "gpt-5.3-codex-spark"
DEFAULT_RESCUE_ROUTE = "supervisor-rescue-planner"
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024
DEFAULT_MAX_JSON_DEPTH = 12


class RescuePlannerError(RuntimeError):
    """Base error for invalid planner configuration."""


class RescuePlannerValidationError(RescuePlannerError):
    """A provider response is not a safe, closed rescue plan."""

    def __init__(self, message: str, *, reason_code: str = "malformed_plan") -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "malformed_plan")


class RescuePlanningDisposition(str, Enum):
    """Typed outcome of one proposal attempt."""

    PROPOSED = "proposed"
    REUSED = "reused"
    NO_PLAN = "no_plan"
    QUARANTINE = "quarantine"


class RescueGuidanceStep(str, Enum):
    """Closed, effect-free next steps returned when no plan is available."""

    OPERATOR_REVIEW = "operator_review"
    WAIT_FOR_COOLDOWN = "wait_for_cooldown"
    RETRY_PROVIDER_AFTER_BACKOFF = "retry_provider_after_backoff"
    REFRESH_INCIDENT_EVIDENCE = "refresh_incident_evidence"
    REFRESH_EXHAUSTION_RECEIPT = "refresh_exhaustion_receipt"
    QUARANTINE_INCIDENT = "quarantine_incident"


@dataclass(frozen=True)
class RescueParameterSpec:
    """One closed parameter in an operation schema."""

    kind: str
    required: bool = True
    minimum: int | None = None
    maximum: int | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"boolean", "integer", "string"}:
            raise RescuePlannerError(f"unsupported rescue parameter kind {self.kind!r}")
        if self.kind != "integer" and (
            self.minimum is not None or self.maximum is not None
        ):
            raise RescuePlannerError("only integer parameters may have numeric bounds")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise RescuePlannerError("parameter minimum exceeds maximum")
        if self.choices and self.kind == "integer" and not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in self.choices
        ):
            raise RescuePlannerError("integer parameter choices must be integers")
        if self.choices and self.kind == "boolean" and not all(
            isinstance(item, bool) for item in self.choices
        ):
            raise RescuePlannerError("boolean parameter choices must be booleans")
        if self.choices and self.kind == "string" and not all(
            isinstance(item, str) and item for item in self.choices
        ):
            raise RescuePlannerError("string parameter choices must be nonempty text")

    def to_prompt_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.kind,
            "required": self.required,
        }
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.choices:
            result["enum"] = list(self.choices)
        return result

    def validate(self, value: Any, name: str) -> None:
        if self.kind == "boolean":
            valid = isinstance(value, bool)
        elif self.kind == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise RescuePlannerValidationError(
                f"parameter {name!r} must have type {self.kind}",
                reason_code="invalid_parameters",
            )
        if self.kind == "integer":
            if self.minimum is not None and value < self.minimum:
                raise RescuePlannerValidationError(
                    f"parameter {name!r} is below its closed minimum",
                    reason_code="invalid_parameters",
                )
            if self.maximum is not None and value > self.maximum:
                raise RescuePlannerValidationError(
                    f"parameter {name!r} exceeds its closed maximum",
                    reason_code="invalid_parameters",
                )
        if self.choices and value not in self.choices:
            raise RescuePlannerValidationError(
                f"parameter {name!r} is outside its closed enum",
                reason_code="invalid_parameters",
            )


@dataclass(frozen=True)
class RescueOperationSpec:
    """Closed schema and semantics for one model-selectable operation."""

    operation: RescueOperation
    parameters: Mapping[str, RescueParameterSpec]
    expected_effects: tuple[str, ...]
    success_test: str
    stop_condition: str
    target_prefixes: tuple[str, ...] = ()
    rollback_operations: tuple[RescueOperation | None, ...] = (None,)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, RescueOperation):
            raise RescuePlannerError("operation catalog keys must be RescueOperation")
        normalized: dict[str, RescueParameterSpec] = {}
        for key, value in self.parameters.items():
            name = str(key).strip()
            if not name or not isinstance(value, RescueParameterSpec):
                raise RescuePlannerError("operation parameter schema is malformed")
            normalized[name] = value
        object.__setattr__(self, "parameters", MappingProxyType(normalized))
        if not self.expected_effects or any(not item for item in self.expected_effects):
            raise RescuePlannerError("operation expected effects must be nonempty")
        if not self.success_test or not self.stop_condition:
            raise RescuePlannerError("operation success and stop conditions are required")
        if not self.rollback_operations:
            raise RescuePlannerError("rollback operation catalog must not be empty")

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "parameters": {
                name: spec.to_prompt_dict()
                for name, spec in sorted(self.parameters.items())
            },
            "additional_parameters": False,
            "expected_effects": list(self.expected_effects),
            "success_test": self.success_test,
            "stop_condition": self.stop_condition,
            "target_rule": (
                "one exact incident target"
                if not self.target_prefixes
                else "one exact incident target with prefix "
                + ", ".join(self.target_prefixes)
            ),
            "rollback_operations": [
                None if item is None else item.value
                for item in self.rollback_operations
            ],
        }


def _spec(
    operation: RescueOperation,
    expected_effect: str,
    success_test: str,
    stop_condition: str,
    *,
    parameters: Mapping[str, RescueParameterSpec] | None = None,
    target_prefixes: tuple[str, ...] = (),
) -> RescueOperationSpec:
    return RescueOperationSpec(
        operation=operation,
        parameters=parameters or {},
        expected_effects=(expected_effect,),
        success_test=success_test,
        stop_condition=stop_condition,
        target_prefixes=target_prefixes,
    )


def default_rescue_operation_catalog() -> Mapping[RescueOperation, RescueOperationSpec]:
    """Return the immutable operation catalog exposed to rescue models."""

    specs = (
        _spec(
            RescueOperation.STATUS,
            "status_observed",
            "status projection is current",
            "stop after one bounded status observation",
        ),
        _spec(
            RescueOperation.HEALTH,
            "health_observed",
            "health evidence is current",
            "stop immediately if health is restored or unchanged",
        ),
        _spec(
            RescueOperation.EVENTS,
            "events_observed",
            "bounded incident events are observed",
            "stop after the bounded event page is observed",
            parameters={
                "limit": RescueParameterSpec(
                    "integer", minimum=1, maximum=100
                )
            },
        ),
        _spec(
            RescueOperation.RECONCILE_PROJECTION,
            "projection_reconciled",
            "projection matches authoritative state",
            "stop if projection is current or reconciliation is denied",
        ),
        _spec(
            RescueOperation.REPAIR_LIFECYCLE_STATE,
            "lifecycle_state_repaired",
            "lifecycle state passes its invariant check",
            "stop after one invariant-checked lifecycle repair",
            target_prefixes=("task:", "attempt:", "lane:", "run:"),
        ),
        _spec(
            RescueOperation.REPAIR_EXPIRED_LEASE,
            "expired_lease_repaired",
            "lease ownership is current and unambiguous",
            "stop if lease is current, disputed, or repaired",
            target_prefixes=("lease:", "lane:", "task:"),
        ),
        _spec(
            RescueOperation.REPAIR_ORPHANED_LOCK,
            "orphaned_lock_repaired",
            "lock ownership is current or the orphan is removed",
            "stop if ownership is live, ambiguous, or repaired",
            target_prefixes=("lock:", "lane:", "worktree:"),
        ),
        _spec(
            RescueOperation.RETRY,
            "retry_scheduled",
            "one policy-bounded retry is scheduled",
            "stop after one retry is scheduled or denied",
            parameters={
                "attempt_limit": RescueParameterSpec(
                    "integer", minimum=1, maximum=3
                )
            },
            target_prefixes=(
                "task:",
                "attempt:",
                "lane:",
                "validation:",
                "merge:",
                "provider:",
            ),
        ),
        _spec(
            RescueOperation.RESTART_LANE,
            "lane_restart_requested",
            "the exact lane has a fresh process identity",
            "stop after one restart or if the lane becomes healthy",
            parameters={
                "grace_period_ms": RescueParameterSpec(
                    "integer", minimum=0, maximum=60_000
                )
            },
            target_prefixes=("lane:",),
        ),
        _spec(
            RescueOperation.VALIDATION_REPLAY,
            "validation_replay_requested",
            "the frozen validation is replayed once",
            "stop after one replay result or on root drift",
            target_prefixes=("validation:", "task:", "lane:"),
        ),
        _spec(
            RescueOperation.RESCUE_DIRTY_WORK,
            "dirty_work_preservation_requested",
            "dirty work is preserved under existing worktree policy",
            "stop after preservation or on ownership ambiguity",
            parameters={
                "preserve_untracked": RescueParameterSpec("boolean")
            },
            target_prefixes=("worktree:",),
        ),
        _spec(
            RescueOperation.RECONCILE_WORKTREE,
            "worktree_reconciliation_requested",
            "worktree metadata matches the exact current root",
            "stop after one reconciliation or on root drift",
            target_prefixes=("worktree:",),
        ),
        _spec(
            RescueOperation.QUARANTINE,
            "scope_quarantine_requested",
            "the exact incident scope is quarantined",
            "stop after quarantine is observed",
            parameters={
                "reason_code": RescueParameterSpec(
                    "string",
                    choices=(
                        "unchanged_failure",
                        "unsafe_recovery",
                        "provider_uncertain",
                        "root_drift",
                        "manual_review_required",
                    ),
                )
            },
        ),
        _spec(
            RescueOperation.REASSIGN_INDEPENDENT_WORK,
            "independent_work_reassignment_requested",
            "only dependency-independent work is reassigned",
            "stop after one bounded reassignment or on dependency ambiguity",
            parameters={
                "max_tasks": RescueParameterSpec(
                    "integer", minimum=1, maximum=8
                )
            },
            target_prefixes=("lane:", "task:"),
        ),
        _spec(
            RescueOperation.OBJECTIVE_RECONCILE,
            "objective_reconciliation_requested",
            "objective projection matches existing authoritative evidence",
            "stop after one reconciliation and never claim completion",
            target_prefixes=("objective:",),
        ),
    )
    return MappingProxyType({item.operation: item for item in specs})


DEFAULT_RESCUE_OPERATION_CATALOG = default_rescue_operation_catalog()


def _parameter_response_schema(spec: RescueParameterSpec) -> dict[str, Any]:
    result: dict[str, Any] = {"type": spec.kind}
    if spec.minimum is not None:
        result["minimum"] = spec.minimum
    if spec.maximum is not None:
        result["maximum"] = spec.maximum
    if spec.choices:
        result["enum"] = list(spec.choices)
    return result


def _response_schema(
    *,
    operation_catalog: Mapping[
        RescueOperation, RescueOperationSpec
    ] = DEFAULT_RESCUE_OPERATION_CATALOG,
    allowed_operations: Sequence[RescueOperation] | None = None,
    target_ids: Sequence[str] = (),
    incident_cid: str = "",
    exhaustion_receipt_cid: str = "",
    repository_root_cid: str = "",
    run_cid: str = "",
    policy_root: str = "",
    evidence_reference_cids: Sequence[str] = (),
    max_actions: int | None = None,
) -> dict[str, Any]:
    selected_operations = tuple(
        allowed_operations
        if allowed_operations is not None
        else operation_catalog
    )
    action_fields = [
        "schema",
        "contract_version",
        "operation",
        "target_id",
        "parameters",
        "precondition_cids",
        "expected_effects",
        "success_test",
        "stop_condition",
        "rollback_operation",
    ]
    plan_fields = [
        "schema",
        "contract_version",
        "incident_cid",
        "exhaustion_receipt_cid",
        "repository_root_cid",
        "run_cid",
        "policy_root",
        "actions",
        "rationale_reference_cids",
        "unresolved_risks",
        "max_actions",
        "status",
        "created_at_ms",
        "updated_at_ms",
    ]
    action_variants: list[dict[str, Any]] = []
    for operation in selected_operations:
        if operation not in operation_catalog:
            continue
        spec = operation_catalog[operation]
        compatible_targets = [
            target
            for target in target_ids
            if not spec.target_prefixes or target.startswith(spec.target_prefixes)
        ]
        if target_ids and not compatible_targets:
            continue
        parameter_properties = {
            name: _parameter_response_schema(parameter)
            for name, parameter in sorted(spec.parameters.items())
        }
        required_parameters = [
            name
            for name, parameter in sorted(spec.parameters.items())
            if parameter.required
        ]
        rollback_values = [
            None if item is None else item.value
            for item in spec.rollback_operations
        ]
        properties: dict[str, Any] = {
            "schema": {"const": RESCUE_ACTION_SCHEMA},
            "contract_version": {"const": 1},
            "operation": {"const": operation.value},
            "target_id": (
                {"enum": compatible_targets}
                if compatible_targets
                else {"type": "string", "minLength": 1}
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": required_parameters,
                "properties": parameter_properties,
            },
            "precondition_cids": {
                "type": "array",
                "minItems": 2,
                "uniqueItems": True,
                "items": (
                    {
                        "enum": list(
                            dict.fromkeys(
                                (
                                    incident_cid,
                                    exhaustion_receipt_cid,
                                    *evidence_reference_cids,
                                )
                            )
                        )
                    }
                    if incident_cid and exhaustion_receipt_cid
                    else {"type": "string"}
                ),
            },
            "expected_effects": {"const": list(spec.expected_effects)},
            "success_test": {"const": spec.success_test},
            "stop_condition": {"const": spec.stop_condition},
            "rollback_operation": {"enum": rollback_values},
            "content_id": {"type": "string"},
        }
        action_variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": action_fields,
                "properties": properties,
            }
        )

    def exact_string(value: str) -> dict[str, Any]:
        return {"const": value} if value else {"type": "string"}

    actions_schema: dict[str, Any] = {
        "type": "array",
        "minItems": 1,
        "items": (
            {"oneOf": action_variants}
            if action_variants
            else {"not": {}}
        ),
    }
    if max_actions is not None:
        actions_schema["maxItems"] = max_actions

    max_actions_schema: dict[str, Any] = {"type": "integer", "minimum": 1}
    if max_actions is not None:
        max_actions_schema["maximum"] = max_actions

    return {
        "title": RESCUE_PLAN_RESPONSE_NAME,
        "type": "object",
        "additionalProperties": False,
        "required": plan_fields,
        "properties": {
            "schema": {"const": RESCUE_PLAN_SCHEMA},
            "contract_version": {"const": 1},
            "incident_cid": exact_string(incident_cid),
            "exhaustion_receipt_cid": exact_string(exhaustion_receipt_cid),
            "repository_root_cid": exact_string(repository_root_cid),
            "run_cid": exact_string(run_cid),
            "policy_root": exact_string(policy_root),
            "actions": actions_schema,
            "rationale_reference_cids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": (
                    {"enum": list(evidence_reference_cids)}
                    if evidence_reference_cids
                    else {"type": "string"}
                ),
            },
            "unresolved_risks": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "items": {"type": "string"},
            },
            "max_actions": max_actions_schema,
            "status": {"const": "proposed"},
            "created_at_ms": {"const": 0},
            "updated_at_ms": {"const": 0},
        },
    }


RESCUE_PLAN_V1_JSON_SCHEMA = MappingProxyType(_response_schema())


@dataclass(frozen=True)
class RescuePlannerPolicy:
    """Explicit provider, safety, freshness, and resource policy."""

    enabled: bool = False
    allow_provider_calls: bool = False
    provider: str = "llm_router"
    model: str = DEFAULT_RESCUE_MODEL
    allowed_providers: tuple[str, ...] = ("llm_router",)
    allowed_models: tuple[str, ...] = (DEFAULT_RESCUE_MODEL,)
    allowed_operations: tuple[RescueOperation, ...] = tuple(
        DEFAULT_RESCUE_OPERATION_CATALOG
    )
    max_actions: int = 4
    max_prompt_tokens: int = 8_192
    max_provider_tokens: int = 2_048
    max_latency_ms: int = 60_000
    max_cost_microunits: int = 50_000
    cost_per_1k_tokens_microunits: int = 1_000
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_json_depth: int = DEFAULT_MAX_JSON_DEPTH
    max_diagnostic_bytes: int = 16_384
    max_diagnostic_items: int = 256
    max_diagnostic_depth: int = 8
    max_receipt_age_ms: int = 300_000
    cooldown_ms: int = 60_000
    circuit_breaker_failures: int = 2

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "allow_provider_calls",
        ):
            if not isinstance(getattr(self, name), bool):
                raise RescuePlannerError(f"{name} must be boolean")
        if not self.provider.strip() or not self.model.strip():
            raise RescuePlannerError("provider and model must be explicit")
        if not self.allowed_providers or not self.allowed_models:
            raise RescuePlannerError("provider and model allowlists must be nonempty")
        if self.provider not in self.allowed_providers:
            raise RescuePlannerError("configured rescue provider is not allowlisted")
        if self.model not in self.allowed_models:
            raise RescuePlannerError("configured rescue model is not allowlisted")
        operations: list[RescueOperation] = []
        for item in self.allowed_operations:
            try:
                operation = (
                    item if isinstance(item, RescueOperation) else RescueOperation(item)
                )
            except (TypeError, ValueError) as exc:
                raise RescuePlannerError("policy contains an unknown operation") from exc
            if operation not in DEFAULT_RESCUE_OPERATION_CATALOG:
                raise RescuePlannerError(
                    f"operation {operation.value!r} is outside the rescue catalog"
                )
            operations.append(operation)
        if not operations or len(set(operations)) != len(operations):
            raise RescuePlannerError(
                "allowed_operations must be nonempty and contain no duplicates"
            )
        object.__setattr__(self, "allowed_operations", tuple(operations))
        positive = (
            "max_actions",
            "max_prompt_tokens",
            "max_provider_tokens",
            "max_latency_ms",
            "max_cost_microunits",
            "cost_per_1k_tokens_microunits",
            "max_response_bytes",
            "max_json_depth",
            "max_diagnostic_bytes",
            "max_diagnostic_items",
            "max_diagnostic_depth",
            "max_receipt_age_ms",
            "circuit_breaker_failures",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RescuePlannerError(f"{name} must be a positive integer")
        if (
            isinstance(self.cooldown_ms, bool)
            or not isinstance(self.cooldown_ms, int)
            or self.cooldown_ms < 0
        ):
            raise RescuePlannerError("cooldown_ms must be a non-negative integer")
        if self.max_actions > 32:
            raise RescuePlannerError("max_actions exceeds the contract bound")
        if self.max_response_bytes > 1_048_576:
            raise RescuePlannerError("max_response_bytes exceeds the contract bound")

    @classmethod
    def permit(
        cls,
        *,
        provider: str = "llm_router",
        model: str = DEFAULT_RESCUE_MODEL,
        **changes: Any,
    ) -> "RescuePlannerPolicy":
        """Build an explicitly enabled policy for one provider/model pair."""

        return cls(
            enabled=True,
            allow_provider_calls=True,
            provider=provider,
            model=model,
            allowed_providers=(provider,),
            allowed_models=(model,),
            **changes,
        )


@dataclass(frozen=True)
class RescuePlanningRequest:
    """All caller-owned bindings for one rescue proposal attempt."""

    incident: SupervisorIncident
    exhaustion_receipt: ProgrammaticRecoveryExhaustionReceipt
    diagnostics: Mapping[str, Any]
    evidence_redacted: bool
    current_repository_root_cid: str
    current_run_cid: str
    current_policy_root: str
    evidence_reference_cids: tuple[str, ...] = ()
    max_provider_tokens: int | None = None
    timeout_ms: int | None = None
    max_cost_microunits: int | None = None
    now_ms: int | None = None


@dataclass(frozen=True)
class RescueNoPlanGuidance:
    """Typed, effect-free guidance for a denied or failed proposal."""

    disposition: RescuePlanningDisposition
    reason_code: str
    incident_cid: str
    exhaustion_receipt_cid: str
    next_steps: tuple[RescueGuidanceStep, ...]
    retry_after_ms: int = 0
    effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.disposition not in {
            RescuePlanningDisposition.NO_PLAN,
            RescuePlanningDisposition.QUARANTINE,
        }:
            raise RescuePlannerError("no-plan guidance has an invalid disposition")
        if not self.reason_code:
            raise RescuePlannerError("guidance requires a reason code")
        if self.effects:
            raise RescuePlannerError("no-plan guidance cannot contain effects")
        if self.retry_after_ms < 0:
            raise RescuePlannerError("retry_after_ms cannot be negative")
        if not self.next_steps:
            raise RescuePlannerError("guidance requires a closed next step")
        object.__setattr__(
            self,
            "next_steps",
            tuple(
                item
                if isinstance(item, RescueGuidanceStep)
                else RescueGuidanceStep(item)
                for item in self.next_steps
            ),
        )

    @property
    def quarantine_required(self) -> bool:
        return self.disposition is RescuePlanningDisposition.QUARANTINE

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
            "incident_cid": self.incident_cid,
            "exhaustion_receipt_cid": self.exhaustion_receipt_cid,
            "next_steps": [item.value for item in self.next_steps],
            "retry_after_ms": self.retry_after_ms,
            "effects": [],
        }


@dataclass(frozen=True)
class RescuePlanningResult:
    """Validated proposal or typed no-plan outcome; never an execution receipt."""

    disposition: RescuePlanningDisposition
    incident_cid: str
    exhaustion_receipt_cid: str
    plan: RescuePlan | None = None
    guidance: RescueNoPlanGuidance | None = None
    provider_invoked: bool = False
    reused: bool = False
    reason_code: str = ""
    prompt_sha256: str = ""
    response_sha256: str = ""
    estimated_cost_microunits: int = 0
    elapsed_ms: int = 0
    effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.effects:
            raise RescuePlannerError("rescue planning results cannot contain effects")
        has_plan = self.plan is not None
        if self.disposition in {
            RescuePlanningDisposition.PROPOSED,
            RescuePlanningDisposition.REUSED,
        }:
            if not has_plan or self.guidance is not None:
                raise RescuePlannerError("proposal result must contain only a plan")
        elif has_plan or self.guidance is None:
            raise RescuePlannerError("no-plan result must contain only guidance")
        if self.disposition is RescuePlanningDisposition.REUSED and not self.reused:
            raise RescuePlannerError("reused disposition must be marked reused")
        if self.elapsed_ms < 0 or self.estimated_cost_microunits < 0:
            raise RescuePlannerError("result resource accounting cannot be negative")

    @property
    def proposed(self) -> bool:
        return self.plan is not None

    @property
    def quarantine_required(self) -> bool:
        return bool(self.guidance and self.guidance.quarantine_required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "incident_cid": self.incident_cid,
            "exhaustion_receipt_cid": self.exhaustion_receipt_cid,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "guidance": (
                None if self.guidance is None else self.guidance.to_dict()
            ),
            "provider_invoked": self.provider_invoked,
            "reused": self.reused,
            "reason_code": self.reason_code,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "estimated_cost_microunits": self.estimated_cost_microunits,
            "elapsed_ms": self.elapsed_ms,
            "effects": [],
        }


@dataclass
class RescuePlannerState:
    """Caller-owned state supporting deduplication and circuit breaking."""

    prior_results: dict[str, RescuePlanningResult] = field(default_factory=dict)
    last_provider_call_ms: dict[str, int] = field(default_factory=dict)
    consecutive_failures: dict[str, int] = field(default_factory=dict)
    open_circuits: set[str] = field(default_factory=set)
    in_flight_incidents: set[str] = field(default_factory=set)
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )


RescueProvider = Callable[[str], str]


_FORBIDDEN_DIAGNOSTIC_KEYS = frozenset(
    {
        "api_key",
        "argv",
        "authorization",
        "code",
        "command",
        "cookie",
        "credential",
        "credentials",
        "new_path",
        "output_path",
        "password",
        "patch",
        "path",
        "paths",
        "policy_change",
        "private_key",
        "prompt",
        "raw_log",
        "refresh_token",
        "script",
        "secret",
        "shell",
        "source",
        "source_code",
        "taskboard",
        "token",
    }
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*(?:PRIVATE KEY|SECRET)[A-Z0-9 ]*-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
        r"(?:\.[A-Za-z0-9_-]{8,})?\b"
    ),
)
_SECRET_FIELD_MARKERS = frozenset(
    {
        "apikey",
        "accesstoken",
        "authorization",
        "authtoken",
        "bearer",
        "clientsecret",
        "cookie",
        "credential",
        "password",
        "passwd",
        "privatekey",
        "refreshtoken",
        "secret",
        "secretkey",
        "sessiontoken",
        "token",
    }
)
_FORBIDDEN_PLAN_TEXT = (
    re.compile(r"`"),
    re.compile(r"\bdiff --git\b", re.IGNORECASE),
    re.compile(
        r"(?:^|\s)(?:sudo|bash|sh|powershell|cmd\.exe|rm|mv|cp|git|"
        r"python3?|curl|wget|chmod|chown|kill|echo)\s",
        re.IGNORECASE,
    ),
    re.compile(r"(?:\$\(|&&|\|\|)"),
    re.compile(
        r"\b(?:run|execute|invoke)\s+(?:a\s+|the\s+)?"
        r"(?:shell\s+)?(?:command|script|binary)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:apply|write|edit)\s+(?:a\s+)?(?:code\s+)?patch\b", re.IGNORECASE),
    re.compile(r"\b(?:password|api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(?:^|\s)/(?:tmp|etc|home|root|workspace|var)(?:/|\b)"),
    re.compile(r"(?:^|\s)(?:\.\.?/|[A-Za-z]:\\|file://)"),
    re.compile(
        r"\b(?:self[- ]?authoriz\w*|grant (?:me|itself) authority)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:change|override|weaken|disable)\s+(?:the\s+)?policy\b", re.IGNORECASE),
    re.compile(
        r"\b(?:change|modify|override|expand|elevate|grant|weaken|disable)\s+"
        r"(?:the\s+|model(?:'s)?\s+|operator(?:'s)?\s+)?"
        r"(?:authority|authorization|permissions?|roles?|scope|budgets?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:rewrite|edit|update)\s+(?:the\s+)?task\s*board\b", re.IGNORECASE),
    re.compile(r"\bmark\s+(?:the\s+)?task\s+(?:as\s+)?complete", re.IGNORECASE),
    re.compile(r"\bclaim\s+(?:task\s+)?completion\b", re.IGNORECASE),
    re.compile(
        r"\b(?:set|change)\s+(?:the\s+)?(?:task\s+)?status\s+to\s+completed\b",
        re.IGNORECASE,
    ),
)


def _json_depth(value: Any, depth: int = 0) -> int:
    if isinstance(value, Mapping):
        return max(
            (depth, *(_json_depth(item, depth + 1) for item in value.values()))
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return max((depth, *(_json_depth(item, depth + 1) for item in value)))
    return depth


def _bounded_redacted_diagnostics(
    value: Any,
    *,
    policy: RescuePlannerPolicy,
) -> Any:
    seen = 0

    def visit(item: Any, depth: int, field_name: str = "") -> Any:
        nonlocal seen
        seen += 1
        if seen > policy.max_diagnostic_items:
            raise RescuePlannerValidationError(
                "diagnostics exceed the item bound",
                reason_code="diagnostics_over_budget",
            )
        if depth > policy.max_diagnostic_depth:
            raise RescuePlannerValidationError(
                "diagnostics exceed the depth bound",
                reason_code="diagnostics_over_budget",
            )
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            raise RescuePlannerValidationError(
                "diagnostics cannot contain floating-point values",
                reason_code="unredacted_evidence",
            )
        if isinstance(item, str):
            if any(pattern.search(item) for pattern in _SECRET_VALUE_PATTERNS):
                raise RescuePlannerValidationError(
                    "diagnostics contain credential-like material",
                    reason_code="unredacted_evidence",
                )
            if len(item.encode("utf-8", errors="surrogatepass")) > 2_048:
                raise RescuePlannerValidationError(
                    "a diagnostic string exceeds its bound",
                    reason_code="diagnostics_over_budget",
                )
            return item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            normalized_keys: set[str] = set()
            for raw_key in sorted(item, key=lambda member: str(member)):
                if not isinstance(raw_key, str):
                    raise RescuePlannerValidationError(
                        "diagnostic object keys must be strings",
                        reason_code="unredacted_evidence",
                    )
                key = raw_key.strip()
                normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
                compact = normalized.replace("_", "")
                if (
                    not key
                    or normalized in _FORBIDDEN_DIAGNOSTIC_KEYS
                    or compact in _SECRET_FIELD_MARKERS
                    or any(
                        marker in compact
                        for marker in _SECRET_FIELD_MARKERS - {"token"}
                    )
                ):
                    raise RescuePlannerValidationError(
                        "diagnostics contain a forbidden or unredacted field",
                        reason_code="unredacted_evidence",
                    )
                if normalized in normalized_keys:
                    raise RescuePlannerValidationError(
                        "diagnostics contain ambiguous duplicate fields",
                        reason_code="unredacted_evidence",
                    )
                normalized_keys.add(normalized)
                result[key] = visit(item[raw_key], depth + 1, key)
            return result
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return [visit(member, depth + 1, field_name) for member in item]
        raise RescuePlannerValidationError(
            "diagnostics contain an unsupported value",
            reason_code="unredacted_evidence",
        )

    result = visit(value, 0)
    encoded = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(encoded) > policy.max_diagnostic_bytes:
        raise RescuePlannerValidationError(
            "diagnostics exceed the serialized byte bound",
            reason_code="diagnostics_over_budget",
        )
    return result


def _strict_json_object(text: str, *, policy: RescuePlannerPolicy) -> Mapping[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise RescuePlannerValidationError(
            "provider returned an empty rescue plan",
            reason_code="empty_response",
        )
    if len(text.encode("utf-8", errors="surrogatepass")) > policy.max_response_bytes:
        raise RescuePlannerValidationError(
            "provider response exceeds the byte budget",
            reason_code="response_over_budget",
        )

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise RescuePlannerValidationError(
            "provider response is not strict JSON",
            reason_code="malformed_json",
        ) from exc
    if not isinstance(payload, Mapping):
        raise RescuePlannerValidationError(
            "provider response must be one JSON object",
            reason_code="malformed_plan",
        )
    if _json_depth(payload) > policy.max_json_depth:
        raise RescuePlannerValidationError(
            "provider response exceeds the nesting bound",
            reason_code="response_over_budget",
        )
    return payload


def _exact_fields(
    payload: Mapping[str, Any],
    expected: Sequence[str],
    noun: str,
) -> None:
    actual = set(payload)
    required = set(expected)
    if actual != required:
        detail = "missing fields" if required - actual else "unknown fields"
        raise RescuePlannerValidationError(
            f"{noun} contains {detail}",
            reason_code="invalid_schema",
        )


def _scan_forbidden_plan_text(value: Any) -> None:
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
            raise RescuePlannerValidationError(
                "rescue plan contains credential-like material",
                reason_code="forbidden_content",
            )
        if any(pattern.search(value) for pattern in _FORBIDDEN_PLAN_TEXT):
            raise RescuePlannerValidationError(
                "rescue plan contains commands, patches, authority changes, or completion claims",
                reason_code="forbidden_content",
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _FORBIDDEN_DIAGNOSTIC_KEYS:
                raise RescuePlannerValidationError(
                    "rescue plan contains a forbidden open-ended field",
                    reason_code="forbidden_content",
                )
            _scan_forbidden_plan_text(item)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            _scan_forbidden_plan_text(item)


def _parameter_schema_matches(
    parameters: Any,
    spec: RescueOperationSpec,
) -> None:
    if not isinstance(parameters, Mapping):
        raise RescuePlannerValidationError(
            "action parameters must be an object",
            reason_code="invalid_parameters",
        )
    expected = set(spec.parameters)
    required = {
        name for name, parameter in spec.parameters.items() if parameter.required
    }
    actual = set(parameters)
    if not required.issubset(actual) or not actual.issubset(expected):
        raise RescuePlannerValidationError(
            "action parameters do not match the closed operation schema",
            reason_code="invalid_parameters",
        )
    for name, value in parameters.items():
        spec.parameters[name].validate(value, name)


def _closed_operation_catalog(
    operation_catalog: Mapping[RescueOperation, RescueOperationSpec],
    *,
    error_type: type[RescuePlannerError] = RescuePlannerError,
) -> Mapping[RescueOperation, RescueOperationSpec]:
    """Copy and verify that a supplied catalog cannot widen rescue authority."""

    normalized: dict[RescueOperation, RescueOperationSpec] = {}
    try:
        entries = operation_catalog.items()
    except AttributeError as exc:
        raise error_type("operation catalog must be a mapping") from exc
    for raw_operation, spec in entries:
        try:
            operation = (
                raw_operation
                if isinstance(raw_operation, RescueOperation)
                else RescueOperation(raw_operation)
            )
        except (TypeError, ValueError) as exc:
            raise error_type("operation catalog contains an unknown operation") from exc
        canonical = DEFAULT_RESCUE_OPERATION_CATALOG.get(operation)
        if (
            not isinstance(spec, RescueOperationSpec)
            or spec.operation is not operation
            or canonical is None
            or spec != canonical
        ):
            raise error_type(
                "operation catalog differs from the closed rescue catalog"
            )
        normalized[operation] = spec
    if not normalized:
        raise error_type("operation catalog must not be empty")
    return MappingProxyType(normalized)


def _validate_parse_context(
    *,
    incident: SupervisorIncident,
    exhaustion_receipt: ProgrammaticRecoveryExhaustionReceipt,
    current_repository_root_cid: str,
    current_run_cid: str,
    current_policy_root: str,
    evidence_reference_cids: Sequence[str],
) -> tuple[str, ...]:
    if not isinstance(incident, SupervisorIncident):
        raise RescuePlannerValidationError(
            "rescue parsing requires a typed incident",
            reason_code="invalid_incident",
        )
    if not isinstance(
        exhaustion_receipt, ProgrammaticRecoveryExhaustionReceipt
    ):
        raise RescuePlannerValidationError(
            "rescue parsing requires a typed exhaustion receipt",
            reason_code="missing_exhaustion",
        )
    if (
        exhaustion_receipt.incident_cid != incident.incident_cid
        or exhaustion_receipt.repository_root_cid
        != incident.repository_root_cid
        or exhaustion_receipt.run_cid != incident.run_cid
        or exhaustion_receipt.policy_root != incident.policy_root
    ):
        raise RescuePlannerValidationError(
            "exhaustion receipt is not bound to the exact incident",
            reason_code="exhaustion_mismatch",
        )
    if exhaustion_receipt.status is not RecordStatus.QUARANTINED:
        raise RescuePlannerValidationError(
            "exhaustion receipt is not terminal",
            reason_code="exhaustion_not_terminal",
        )
    if exhaustion_receipt.updated_at_ms < exhaustion_receipt.created_at_ms:
        raise RescuePlannerValidationError(
            "exhaustion receipt has non-monotonic timestamps",
            reason_code="stale_exhaustion",
        )
    if incident.status not in {RecordStatus.FAILED, RecordStatus.QUARANTINED}:
        raise RescuePlannerValidationError(
            "incident is not active",
            reason_code="incident_not_active",
        )
    if incident.updated_at_ms < incident.observed_at_ms:
        raise RescuePlannerValidationError(
            "incident has non-monotonic timestamps",
            reason_code="stale_incident",
        )
    if any(
        attempt.target_id not in incident.target_ids
        for attempt in exhaustion_receipt.attempts
    ):
        raise RescuePlannerValidationError(
            "exhaustion attempts contain an unbound target",
            reason_code="exhaustion_mismatch",
        )
    if (
        current_repository_root_cid != incident.repository_root_cid
        or current_run_cid != incident.run_cid
        or current_policy_root != incident.policy_root
    ):
        raise RescuePlannerValidationError(
            "parser context contains stale authority roots",
            reason_code="stale_roots",
        )
    if isinstance(evidence_reference_cids, (str, bytes)) or not isinstance(
        evidence_reference_cids, Sequence
    ):
        raise RescuePlannerValidationError(
            "evidence references must be a bounded sequence",
            reason_code="invalid_evidence_references",
        )
    references = tuple(evidence_reference_cids)
    allowed_references = {
        *incident.evidence_cids,
        *(
            attempt.receipt_cid
            for attempt in exhaustion_receipt.attempts
            if attempt.receipt_cid
        ),
    }
    if (
        not references
        or any(not isinstance(item, str) for item in references)
        or len(set(references)) != len(references)
        or not set(references).issubset(allowed_references)
    ):
        raise RescuePlannerValidationError(
            "parser context contains unbound evidence references",
            reason_code="invalid_evidence_references",
        )
    return references


def parse_rescue_plan(
    text: str,
    *,
    incident: SupervisorIncident,
    exhaustion_receipt: ProgrammaticRecoveryExhaustionReceipt,
    current_repository_root_cid: str,
    current_run_cid: str,
    current_policy_root: str,
    evidence_reference_cids: Sequence[str],
    policy: RescuePlannerPolicy,
    operation_catalog: Mapping[
        RescueOperation, RescueOperationSpec
    ] = DEFAULT_RESCUE_OPERATION_CATALOG,
) -> RescuePlan:
    """Parse and validate an exact ``RescuePlan/v1`` provider response."""

    evidence_references = _validate_parse_context(
        incident=incident,
        exhaustion_receipt=exhaustion_receipt,
        current_repository_root_cid=current_repository_root_cid,
        current_run_cid=current_run_cid,
        current_policy_root=current_policy_root,
        evidence_reference_cids=evidence_reference_cids,
    )
    try:
        closed_catalog = _closed_operation_catalog(
            operation_catalog,
            error_type=RescuePlannerValidationError,
        )
    except RescuePlannerValidationError as exc:
        exc.reason_code = "invalid_catalog"
        raise
    payload = _strict_json_object(text, policy=policy)
    plan_fields = (
        "schema",
        "contract_version",
        "incident_cid",
        "exhaustion_receipt_cid",
        "repository_root_cid",
        "run_cid",
        "policy_root",
        "actions",
        "rationale_reference_cids",
        "unresolved_risks",
        "max_actions",
        "status",
        "created_at_ms",
        "updated_at_ms",
    )
    _exact_fields(payload, plan_fields, RESCUE_PLAN_RESPONSE_NAME)
    if payload.get("schema") != RESCUE_PLAN_SCHEMA or payload.get(
        "contract_version"
    ) != 1:
        raise RescuePlannerValidationError(
            "provider returned an unsupported rescue plan version",
            reason_code="invalid_schema",
        )
    exact_bindings = (
        ("incident_cid", incident.incident_cid, "incident_mismatch"),
        (
            "exhaustion_receipt_cid",
            exhaustion_receipt.receipt_cid,
            "exhaustion_mismatch",
        ),
        (
            "repository_root_cid",
            current_repository_root_cid,
            "stale_roots",
        ),
        ("run_cid", current_run_cid, "stale_roots"),
        ("policy_root", current_policy_root, "stale_roots"),
    )
    for field_name, expected, reason_code in exact_bindings:
        if payload.get(field_name) != expected:
            raise RescuePlannerValidationError(
                f"rescue plan {field_name} does not match current authority",
                reason_code=reason_code,
            )
    if (
        payload.get("status") != RecordStatus.PROPOSED.value
        or payload.get("created_at_ms") != 0
        or payload.get("updated_at_ms") != 0
    ):
        raise RescuePlannerValidationError(
            "model cannot change lifecycle status or timestamps",
            reason_code="self_authorization",
        )
    max_actions = payload.get("max_actions")
    effective_max_actions = min(
        policy.max_actions,
        exhaustion_receipt.budget.max_rescue_actions,
    )
    if (
        isinstance(max_actions, bool)
        or not isinstance(max_actions, int)
        or max_actions < 1
        or max_actions > effective_max_actions
    ):
        raise RescuePlannerValidationError(
            "rescue plan action budget exceeds current authority",
            reason_code="excess_actions",
        )
    raw_actions = payload.get("actions")
    if (
        not isinstance(raw_actions, list)
        or not raw_actions
        or len(raw_actions) > max_actions
        or len(raw_actions) > effective_max_actions
    ):
        raise RescuePlannerValidationError(
            "rescue plan is missing actions or exceeds its action budget",
            reason_code="excess_actions",
        )

    allowed_operations = set(policy.allowed_operations)
    available_preconditions = {
        incident.incident_cid,
        exhaustion_receipt.receipt_cid,
        *evidence_references,
    }
    required_preconditions = {
        incident.incident_cid,
        exhaustion_receipt.receipt_cid,
    }
    action_fields = (
        "schema",
        "contract_version",
        "operation",
        "target_id",
        "parameters",
        "precondition_cids",
        "expected_effects",
        "success_test",
        "stop_condition",
        "rollback_operation",
    )
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, Mapping):
            raise RescuePlannerValidationError(
                f"actions[{index}] must be an object",
                reason_code="invalid_schema",
            )
        actual_action_fields = set(raw_action)
        required_action_fields = set(action_fields)
        if actual_action_fields not in (
            required_action_fields,
            required_action_fields | {"content_id"},
        ):
            detail = (
                "missing fields"
                if required_action_fields - actual_action_fields
                else "unknown fields"
            )
            raise RescuePlannerValidationError(
                f"actions[{index}] contains {detail}",
                reason_code="invalid_schema",
            )
        if raw_action.get("schema") != RESCUE_ACTION_SCHEMA or raw_action.get(
            "contract_version"
        ) != 1:
            raise RescuePlannerValidationError(
                f"actions[{index}] has an unsupported schema",
                reason_code="invalid_schema",
            )
        try:
            operation = RescueOperation(raw_action.get("operation"))
        except (TypeError, ValueError) as exc:
            raise RescuePlannerValidationError(
                f"actions[{index}] selects an unknown operation",
                reason_code="unknown_operation",
            ) from exc
        if operation not in allowed_operations or operation not in closed_catalog:
            raise RescuePlannerValidationError(
                f"actions[{index}] selects a non-permitted operation",
                reason_code="unknown_operation",
            )
        if operation in exhaustion_receipt.inapplicable_operations:
            raise RescuePlannerValidationError(
                f"actions[{index}] selects an operation proven inapplicable",
                reason_code="inapplicable_operation",
            )
        spec = closed_catalog[operation]
        target_id = raw_action.get("target_id")
        if not isinstance(target_id, str) or target_id not in incident.target_ids:
            raise RescuePlannerValidationError(
                f"actions[{index}] target is not an exact incident target",
                reason_code="unknown_target",
            )
        if spec.target_prefixes and not target_id.startswith(spec.target_prefixes):
            raise RescuePlannerValidationError(
                f"actions[{index}] operation is invalid for its target type",
                reason_code="invalid_target_type",
            )
        _parameter_schema_matches(raw_action.get("parameters"), spec)
        preconditions = raw_action.get("precondition_cids")
        if (
            not isinstance(preconditions, list)
            or any(not isinstance(item, str) for item in preconditions)
            or len(set(preconditions)) != len(preconditions)
            or not required_preconditions.issubset(preconditions)
            or not set(preconditions).issubset(available_preconditions)
        ):
            raise RescuePlannerValidationError(
                f"actions[{index}] preconditions are missing or unbound",
                reason_code="invalid_preconditions",
            )
        effects = raw_action.get("expected_effects")
        if (
            not isinstance(effects, list)
            or tuple(effects) != spec.expected_effects
        ):
            raise RescuePlannerValidationError(
                f"actions[{index}] expected effects are outside the catalog",
                reason_code="invalid_expected_effects",
            )
        if raw_action.get("success_test") != spec.success_test:
            raise RescuePlannerValidationError(
                f"actions[{index}] success test is not the catalog test",
                reason_code="missing_success_condition",
            )
        if raw_action.get("stop_condition") != spec.stop_condition:
            raise RescuePlannerValidationError(
                f"actions[{index}] stop condition is missing or open-ended",
                reason_code="missing_stop_condition",
            )
        rollback = raw_action.get("rollback_operation")
        try:
            typed_rollback = (
                None if rollback is None else RescueOperation(rollback)
            )
        except (TypeError, ValueError) as exc:
            raise RescuePlannerValidationError(
                f"actions[{index}] rollback operation is unknown",
                reason_code="unknown_operation",
            ) from exc
        if typed_rollback not in spec.rollback_operations:
            raise RescuePlannerValidationError(
                f"actions[{index}] rollback is outside the catalog",
                reason_code="unknown_operation",
            )

    references = payload.get("rationale_reference_cids")
    allowed_references = set(evidence_references)
    if (
        not isinstance(references, list)
        or not references
        or any(not isinstance(item, str) for item in references)
        or len(set(references)) != len(references)
        or not set(references).issubset(allowed_references)
    ):
        raise RescuePlannerValidationError(
            "plan evidence references are missing or unbound",
            reason_code="invalid_evidence_references",
        )
    risks = payload.get("unresolved_risks")
    if (
        not isinstance(risks, list)
        or not risks
        or len(risks) > 32
        or any(
            not isinstance(item, str)
            or not item.strip()
            or len(item.encode("utf-8", errors="surrogatepass")) > 1_024
            for item in risks
        )
    ):
        raise RescuePlannerValidationError(
            "plan must contain bounded unresolved risks",
            reason_code="missing_risks",
        )
    _scan_forbidden_plan_text(
        {
            "actions": raw_actions,
            "unresolved_risks": risks,
        }
    )
    try:
        plan = RescuePlan.from_dict(payload)
    except (PromptWorkflowContractError, TypeError, ValueError, UnicodeError) as exc:
        raise RescuePlannerValidationError(
            f"rescue plan violates its canonical contract: {exc}",
            reason_code="invalid_schema",
        ) from exc
    return plan


def build_rescue_prompt(
    request: RescuePlanningRequest,
    *,
    policy: RescuePlannerPolicy,
    diagnostics: Mapping[str, Any] | None = None,
    operation_catalog: Mapping[
        RescueOperation, RescueOperationSpec
    ] = DEFAULT_RESCUE_OPERATION_CATALOG,
) -> str:
    """Build the bounded JSON-only provider context for one rescue request."""

    incident = request.incident
    exhaustion = request.exhaustion_receipt
    bounded = (
        _bounded_redacted_diagnostics(request.diagnostics, policy=policy)
        if diagnostics is None
        else diagnostics
    )
    evidence_references = tuple(
        request.evidence_reference_cids or incident.evidence_cids
    )
    closed_catalog = _closed_operation_catalog(operation_catalog)
    selected_operations = tuple(
        operation
        for operation in policy.allowed_operations
        if (
            operation in closed_catalog
            and operation not in exhaustion.inapplicable_operations
            and any(
                not closed_catalog[operation].target_prefixes
                or target.startswith(closed_catalog[operation].target_prefixes)
                for target in incident.target_ids
            )
        )
    )
    selected_catalog: dict[str, Any] = {}
    for operation in selected_operations:
        spec = closed_catalog[operation]
        entry = spec.to_prompt_dict()
        entry["exact_target_ids"] = [
            target
            for target in incident.target_ids
            if not spec.target_prefixes or target.startswith(spec.target_prefixes)
        ]
        selected_catalog[operation.value] = entry
    max_actions = min(
        policy.max_actions, exhaustion.budget.max_rescue_actions
    )
    provider_output_tokens = min(
        request.max_provider_tokens or policy.max_provider_tokens,
        policy.max_provider_tokens,
        exhaustion.budget.max_provider_tokens,
    )
    timeout_ms = min(
        request.timeout_ms or policy.max_latency_ms,
        policy.max_latency_ms,
        exhaustion.budget.max_latency_ms,
    )
    max_cost_microunits = min(
        request.max_cost_microunits or policy.max_cost_microunits,
        policy.max_cost_microunits,
    )
    payload = {
        "instruction": (
            "Return exactly one strict JSON RescuePlan/v1 proposal. Select only "
            "from the closed operation catalog. Do not emit commands, code, "
            "patches, credentials, paths, policy or authority changes, task-board "
            "changes, completion claims, or executable effects."
        ),
        "incident_reference": {
            "schema": incident.SCHEMA,
            "incident_cid": incident.incident_cid,
            "kind": incident.kind.value,
            "target_ids": list(incident.target_ids),
        },
        "exhaustion_reference": {
            "schema": PROGRAMMATIC_RECOVERY_EXHAUSTION_SCHEMA,
            "exhaustion_receipt_cid": exhaustion.receipt_cid,
            "incident_cid": exhaustion.incident_cid,
        },
        "exact_roots": {
            "repository_root": incident.repository_root,
            "state_root": incident.state_root,
            "repository_root_cid": request.current_repository_root_cid,
            "run_cid": request.current_run_cid,
            "policy_root": request.current_policy_root,
        },
        "bounded_redacted_diagnostics": bounded,
        "evidence_reference_cids": list(evidence_references),
        "closed_operation_catalog": selected_catalog,
        "limits": {
            "max_actions": max_actions,
            "provider_output_tokens": provider_output_tokens,
            "timeout_ms": timeout_ms,
            "max_cost_microunits": max_cost_microunits,
        },
        "response_schema": _response_schema(
            operation_catalog=closed_catalog,
            allowed_operations=selected_operations,
            target_ids=incident.target_ids,
            incident_cid=incident.incident_cid,
            exhaustion_receipt_cid=exhaustion.receipt_cid,
            repository_root_cid=request.current_repository_root_cid,
            run_cid=request.current_run_cid,
            policy_root=request.current_policy_root,
            evidence_reference_cids=evidence_references,
            max_actions=max_actions,
        ),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _guidance_steps(reason_code: str, *, quarantine: bool) -> tuple[RescueGuidanceStep, ...]:
    if quarantine:
        return (
            RescueGuidanceStep.QUARANTINE_INCIDENT,
            RescueGuidanceStep.OPERATOR_REVIEW,
        )
    if "cooldown" in reason_code:
        return (
            RescueGuidanceStep.WAIT_FOR_COOLDOWN,
            RescueGuidanceStep.OPERATOR_REVIEW,
        )
    if reason_code in {"stale_incident", "stale_roots"}:
        return (
            RescueGuidanceStep.REFRESH_INCIDENT_EVIDENCE,
            RescueGuidanceStep.OPERATOR_REVIEW,
        )
    if reason_code in {"stale_exhaustion", "exhaustion_mismatch"}:
        return (
            RescueGuidanceStep.REFRESH_EXHAUSTION_RECEIPT,
            RescueGuidanceStep.OPERATOR_REVIEW,
        )
    if reason_code.startswith("provider_"):
        return (
            RescueGuidanceStep.RETRY_PROVIDER_AFTER_BACKOFF,
            RescueGuidanceStep.OPERATOR_REVIEW,
        )
    return (RescueGuidanceStep.OPERATOR_REVIEW,)


class RescuePlanner:
    """Single-flight, bounded, proposal-only rescue planner."""

    def __init__(
        self,
        policy: RescuePlannerPolicy | None = None,
        *,
        provider: RescueProvider | None = None,
        operation_catalog: Mapping[
            RescueOperation, RescueOperationSpec
        ] = DEFAULT_RESCUE_OPERATION_CATALOG,
        state: RescuePlannerState | None = None,
        clock_ms: Callable[[], int] | None = None,
        provider_batch_scheduler: Any = None,
    ) -> None:
        self.policy = policy or RescuePlannerPolicy()
        self.provider = provider
        self.state = state or RescuePlannerState()
        self.operation_catalog = _closed_operation_catalog(operation_catalog)
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.provider_batch_scheduler = provider_batch_scheduler
        self._lock = self.state._lock
        if set(self.policy.allowed_operations).difference(self.operation_catalog):
            raise RescuePlannerError(
                "policy operations are missing from the provided operation catalog"
            )

    def _no_plan(
        self,
        request: RescuePlanningRequest,
        reason_code: str,
        *,
        quarantine: bool = False,
        provider_invoked: bool = False,
        retry_after_ms: int = 0,
        prompt_sha256: str = "",
        response_sha256: str = "",
        estimated_cost_microunits: int = 0,
        elapsed_ms: int = 0,
    ) -> RescuePlanningResult:
        incident_cid = (
            request.incident.incident_cid
            if isinstance(request.incident, SupervisorIncident)
            else ""
        )
        exhaustion_cid = (
            request.exhaustion_receipt.receipt_cid
            if isinstance(
                request.exhaustion_receipt,
                ProgrammaticRecoveryExhaustionReceipt,
            )
            else ""
        )
        disposition = (
            RescuePlanningDisposition.QUARANTINE
            if quarantine
            else RescuePlanningDisposition.NO_PLAN
        )
        guidance = RescueNoPlanGuidance(
            disposition=disposition,
            reason_code=reason_code,
            incident_cid=incident_cid,
            exhaustion_receipt_cid=exhaustion_cid,
            next_steps=_guidance_steps(reason_code, quarantine=quarantine),
            retry_after_ms=max(0, int(retry_after_ms)),
        )
        return RescuePlanningResult(
            disposition=disposition,
            incident_cid=incident_cid,
            exhaustion_receipt_cid=exhaustion_cid,
            guidance=guidance,
            provider_invoked=provider_invoked,
            reason_code=reason_code,
            prompt_sha256=prompt_sha256,
            response_sha256=response_sha256,
            estimated_cost_microunits=estimated_cost_microunits,
            elapsed_ms=elapsed_ms,
        )

    def _gate(
        self,
        request: RescuePlanningRequest,
        now_ms: int,
    ) -> tuple[RescuePlanningResult | None, Mapping[str, Any] | None]:
        policy = self.policy
        if not policy.enabled or not policy.allow_provider_calls:
            return self._no_plan(request, "policy_denied"), None
        if policy.provider not in policy.allowed_providers or policy.model not in (
            policy.allowed_models
        ):
            return self._no_plan(request, "provider_not_permitted"), None
        if not isinstance(request.incident, SupervisorIncident):
            return self._no_plan(request, "invalid_incident"), None
        if not isinstance(
            request.exhaustion_receipt,
            ProgrammaticRecoveryExhaustionReceipt,
        ):
            return self._no_plan(request, "missing_exhaustion"), None
        incident = request.incident
        exhaustion = request.exhaustion_receipt
        if (
            exhaustion.incident_cid != incident.incident_cid
            or exhaustion.repository_root_cid != incident.repository_root_cid
            or exhaustion.run_cid != incident.run_cid
            or exhaustion.policy_root != incident.policy_root
        ):
            return self._no_plan(request, "exhaustion_mismatch"), None
        if exhaustion.status is not RecordStatus.QUARANTINED:
            return self._no_plan(request, "exhaustion_not_terminal"), None
        if exhaustion.updated_at_ms < exhaustion.created_at_ms:
            return self._no_plan(request, "stale_exhaustion"), None
        if any(
            attempt.target_id not in incident.target_ids
            for attempt in exhaustion.attempts
        ):
            return self._no_plan(request, "exhaustion_mismatch"), None
        if (
            request.current_repository_root_cid != incident.repository_root_cid
            or request.current_run_cid != incident.run_cid
            or request.current_policy_root != incident.policy_root
        ):
            return self._no_plan(request, "stale_roots"), None
        if incident.status not in {
            RecordStatus.FAILED,
            RecordStatus.QUARANTINED,
        }:
            return self._no_plan(request, "incident_not_active"), None
        if incident.updated_at_ms < incident.observed_at_ms:
            return self._no_plan(request, "stale_incident"), None
        receipt_time = max(exhaustion.created_at_ms, exhaustion.updated_at_ms)
        if (
            receipt_time > now_ms
            or now_ms - receipt_time > policy.max_receipt_age_ms
        ):
            return self._no_plan(request, "stale_exhaustion"), None
        incident_time = max(incident.observed_at_ms, incident.updated_at_ms)
        if (
            incident_time > now_ms
            or now_ms - incident_time > policy.max_receipt_age_ms
        ):
            return self._no_plan(request, "stale_incident"), None
        if exhaustion.circuit_open:
            return self._no_plan(
                request, "programmatic_circuit_open", quarantine=True
            ), None
        if "cooldown" in exhaustion.exhaustion_reason.casefold():
            return self._no_plan(
                request,
                "programmatic_cooldown_active",
                retry_after_ms=policy.cooldown_ms,
            ), None
        if not isinstance(request.evidence_redacted, bool) or not (
            request.evidence_redacted
        ):
            return self._no_plan(request, "unredacted_evidence"), None
        try:
            diagnostics = _bounded_redacted_diagnostics(
                request.diagnostics, policy=policy
            )
        except RescuePlannerValidationError as exc:
            return self._no_plan(request, exc.reason_code), None
        if not isinstance(diagnostics, Mapping) or not diagnostics:
            return self._no_plan(request, "missing_diagnostics"), None
        allowed_evidence_references = {
            *incident.evidence_cids,
            *(
                attempt.receipt_cid
                for attempt in exhaustion.attempts
                if attempt.receipt_cid
            ),
        }
        evidence_references = tuple(
            request.evidence_reference_cids or incident.evidence_cids
        )
        if (
            not evidence_references
            or any(not isinstance(item, str) for item in evidence_references)
            or len(set(evidence_references)) != len(evidence_references)
            or not set(evidence_references).issubset(
                allowed_evidence_references
            )
        ):
            return self._no_plan(request, "invalid_evidence_references"), None
        applicable_operations = tuple(
            operation
            for operation in policy.allowed_operations
            if (
                operation in self.operation_catalog
                and operation not in exhaustion.inapplicable_operations
                and any(
                    not self.operation_catalog[operation].target_prefixes
                    or target.startswith(
                        self.operation_catalog[operation].target_prefixes
                    )
                    for target in incident.target_ids
                )
            )
        )
        if not applicable_operations:
            return self._no_plan(request, "no_applicable_operations"), None
        for requested, maximum, reason in (
            (
                request.max_provider_tokens,
                min(
                    policy.max_provider_tokens,
                    exhaustion.budget.max_provider_tokens,
                ),
                "token_budget_denied",
            ),
            (
                request.timeout_ms,
                min(policy.max_latency_ms, exhaustion.budget.max_latency_ms),
                "time_budget_denied",
            ),
            (
                request.max_cost_microunits,
                policy.max_cost_microunits,
                "cost_budget_denied",
            ),
        ):
            if requested is not None and (
                isinstance(requested, bool)
                or not isinstance(requested, int)
                or requested < 1
                or requested > maximum
            ):
                return self._no_plan(request, reason), None
        return None, diagnostics

    def _default_provider(
        self,
        prompt: str,
        *,
        repository_root: str,
        output_tokens: int,
        timeout_ms: int,
        prompt_token_limit: int,
    ) -> str:
        # These imports are intentionally deferred until every safety gate has
        # passed, preserving import-time isolation from providers and models.
        from ..planning.task_proposal_router import _call_text_provider
        from ..todo_daemon.llm import LlmRouterInvocation

        invocation = LlmRouterInvocation(
            repo_root=Path(repository_root),
            model_name=self.policy.model,
            provider=(
                None
                if self.policy.provider == "llm_router"
                else self.policy.provider
            ),
            allow_local_fallback=False,
            timeout_seconds=max(1, math.ceil(timeout_ms / 1_000)),
            max_new_tokens=output_tokens,
            max_prompt_chars=len(prompt),
            reject_effective_provider_name="local_hf",
        )
        response, _ = _call_text_provider(
            prompt,
            invocation,
            scheduler=self.provider_batch_scheduler,
            route=DEFAULT_RESCUE_ROUTE,
            operation="rescue_plan.v1",
            context_limit=prompt_token_limit,
            response_contract=RESCUE_PLAN_RESPONSE_NAME,
            provenance={"proposal_only": True},
        )
        return response

    def _record_failure(
        self,
        request: RescuePlanningRequest,
        result: RescuePlanningResult,
        cooldown_key: str,
    ) -> RescuePlanningResult:
        failures = self.state.consecutive_failures.get(cooldown_key, 0) + 1
        self.state.consecutive_failures[cooldown_key] = failures
        if failures >= self.policy.circuit_breaker_failures:
            self.state.open_circuits.add(cooldown_key)
            result = self._no_plan(
                request,
                "planner_circuit_open",
                quarantine=True,
                provider_invoked=result.provider_invoked,
                prompt_sha256=result.prompt_sha256,
                response_sha256=result.response_sha256,
                estimated_cost_microunits=result.estimated_cost_microunits,
                elapsed_ms=result.elapsed_ms,
            )
        self.state.prior_results[request.incident.incident_cid] = result
        return result

    def plan(self, request: RescuePlanningRequest) -> RescuePlanningResult:
        """Return one validated proposal or effect-free typed guidance."""

        with self._lock:
            current_time = (
                request.now_ms
                if request.now_ms is not None
                else self.clock_ms()
            )
            if (
                isinstance(current_time, bool)
                or not isinstance(current_time, int)
                or current_time < 0
            ):
                return self._no_plan(request, "invalid_current_time")
            now_ms = current_time
            gated, diagnostics = self._gate(request, now_ms)
            if gated is not None:
                return gated
            assert diagnostics is not None
            incident = request.incident
            exhaustion = request.exhaustion_receipt
            cooldown_key = incident.cooldown_key or incident.incident_cid
            if incident.incident_cid in self.state.in_flight_incidents:
                return self._no_plan(request, "identical_incident_in_flight")

            prior = self.state.prior_results.get(incident.incident_cid)
            if prior is not None:
                if not isinstance(prior, RescuePlanningResult):
                    self.state.open_circuits.add(cooldown_key)
                    return self._no_plan(
                        request,
                        "planner_state_invalid",
                        quarantine=True,
                    )
                if prior.incident_cid != incident.incident_cid:
                    self.state.open_circuits.add(cooldown_key)
                    return self._no_plan(
                        request,
                        "prior_result_binding_mismatch",
                        quarantine=True,
                    )
                if prior.exhaustion_receipt_cid != exhaustion.receipt_cid:
                    return self._no_plan(
                        request,
                        "identical_incident_receipt_changed",
                        quarantine=True,
                    )
                if (
                    prior.guidance is not None
                    and (
                        prior.guidance.incident_cid != incident.incident_cid
                        or prior.guidance.exhaustion_receipt_cid
                        != exhaustion.receipt_cid
                    )
                ):
                    self.state.open_circuits.add(cooldown_key)
                    return self._no_plan(
                        request,
                        "prior_result_binding_mismatch",
                        quarantine=True,
                    )
                if prior.plan is not None:
                    try:
                        parse_rescue_plan(
                            prior.plan.to_json(),
                            incident=incident,
                            exhaustion_receipt=exhaustion,
                            current_repository_root_cid=(
                                request.current_repository_root_cid
                            ),
                            current_run_cid=request.current_run_cid,
                            current_policy_root=request.current_policy_root,
                            evidence_reference_cids=tuple(
                                request.evidence_reference_cids
                                or incident.evidence_cids
                            ),
                            policy=self.policy,
                            operation_catalog=self.operation_catalog,
                        )
                    except Exception:
                        self.state.open_circuits.add(cooldown_key)
                        return self._no_plan(
                            request,
                            "prior_proposal_invalid",
                            quarantine=True,
                        )
                    return replace(
                        prior,
                        disposition=RescuePlanningDisposition.REUSED,
                        provider_invoked=False,
                        reused=True,
                        reason_code="identical_incident_reused",
                        elapsed_ms=0,
                    )
                guidance = self._no_plan(
                    request,
                    "identical_incident_circuit_break",
                    quarantine=prior.quarantine_required,
                )
                return replace(guidance, reused=True)
            if cooldown_key in self.state.open_circuits:
                return self._no_plan(
                    request, "planner_circuit_open", quarantine=True
                )
            failure_count = self.state.consecutive_failures.get(cooldown_key, 0)
            if (
                isinstance(failure_count, bool)
                or not isinstance(failure_count, int)
                or failure_count < 0
            ):
                self.state.open_circuits.add(cooldown_key)
                return self._no_plan(
                    request,
                    "planner_state_invalid",
                    quarantine=True,
                )
            if failure_count >= self.policy.circuit_breaker_failures:
                self.state.open_circuits.add(cooldown_key)
                return self._no_plan(
                    request, "planner_circuit_open", quarantine=True
                )
            last_call = self.state.last_provider_call_ms.get(cooldown_key)
            if last_call is not None and (
                isinstance(last_call, bool)
                or not isinstance(last_call, int)
                or last_call < 0
            ):
                self.state.open_circuits.add(cooldown_key)
                return self._no_plan(
                    request,
                    "planner_state_invalid",
                    quarantine=True,
                )
            if (
                last_call is not None
                and now_ms - last_call < self.policy.cooldown_ms
            ):
                return self._no_plan(
                    request,
                    "cooldown_active",
                    retry_after_ms=self.policy.cooldown_ms - (now_ms - last_call),
                )

            prompt = build_rescue_prompt(
                request,
                policy=self.policy,
                diagnostics=diagnostics,
                operation_catalog=self.operation_catalog,
            )
            prompt_bytes = prompt.encode("utf-8", errors="surrogatepass")
            prompt_tokens = (len(prompt_bytes) + 3) // 4
            prompt_limit = min(
                self.policy.max_prompt_tokens,
                exhaustion.budget.max_prompt_tokens,
            )
            if prompt_tokens > prompt_limit:
                return self._no_plan(request, "prompt_over_budget")
            output_tokens = min(
                request.max_provider_tokens or self.policy.max_provider_tokens,
                self.policy.max_provider_tokens,
                exhaustion.budget.max_provider_tokens,
            )
            timeout_ms = min(
                request.timeout_ms or self.policy.max_latency_ms,
                self.policy.max_latency_ms,
                exhaustion.budget.max_latency_ms,
            )
            cost_limit = min(
                request.max_cost_microunits
                or self.policy.max_cost_microunits,
                self.policy.max_cost_microunits,
            )
            estimated_cost = math.ceil(
                (prompt_tokens + output_tokens)
                * self.policy.cost_per_1k_tokens_microunits
                / 1_000
            )
            if estimated_cost > cost_limit:
                return self._no_plan(
                    request,
                    "cost_budget_denied",
                    estimated_cost_microunits=estimated_cost,
                )

            prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
            started = int(self.clock_ms())
            self.state.last_provider_call_ms[cooldown_key] = now_ms
            self.state.in_flight_incidents.add(incident.incident_cid)
            response = ""
            try:
                try:
                    response = (
                        self.provider(prompt)
                        if self.provider is not None
                        else self._default_provider(
                            prompt,
                            repository_root=incident.repository_root,
                            output_tokens=output_tokens,
                            timeout_ms=timeout_ms,
                            prompt_token_limit=prompt_limit,
                        )
                    )
                finally:
                    self.state.in_flight_incidents.discard(
                        incident.incident_cid
                    )
            except Exception:
                elapsed = max(0, int(self.clock_ms()) - started)
                result = self._no_plan(
                    request,
                    "provider_unavailable",
                    provider_invoked=True,
                    prompt_sha256=prompt_sha256,
                    estimated_cost_microunits=estimated_cost,
                    elapsed_ms=elapsed,
                )
                return self._record_failure(request, result, cooldown_key)

            elapsed = max(0, int(self.clock_ms()) - started)
            response_sha256 = ""
            if isinstance(response, str):
                try:
                    response_bytes = response.encode("utf-8")
                except UnicodeEncodeError:
                    result = self._no_plan(
                        request,
                        "provider_malformed_unicode",
                        provider_invoked=True,
                        prompt_sha256=prompt_sha256,
                        estimated_cost_microunits=estimated_cost,
                        elapsed_ms=elapsed,
                    )
                    return self._record_failure(request, result, cooldown_key)
                response_sha256 = hashlib.sha256(response_bytes).hexdigest()
            if elapsed > timeout_ms:
                result = self._no_plan(
                    request,
                    "provider_time_over_budget",
                    provider_invoked=True,
                    prompt_sha256=prompt_sha256,
                    response_sha256=response_sha256,
                    estimated_cost_microunits=estimated_cost,
                    elapsed_ms=elapsed,
                )
                return self._record_failure(request, result, cooldown_key)
            if (
                not isinstance(response, str)
                or (len(response.encode("utf-8", errors="surrogatepass")) + 3)
                // 4
                > output_tokens
            ):
                result = self._no_plan(
                    request,
                    "provider_token_over_budget",
                    provider_invoked=True,
                    prompt_sha256=prompt_sha256,
                    response_sha256=response_sha256,
                    estimated_cost_microunits=estimated_cost,
                    elapsed_ms=elapsed,
                )
                return self._record_failure(request, result, cooldown_key)
            try:
                plan = parse_rescue_plan(
                    response,
                    incident=incident,
                    exhaustion_receipt=exhaustion,
                    current_repository_root_cid=request.current_repository_root_cid,
                    current_run_cid=request.current_run_cid,
                    current_policy_root=request.current_policy_root,
                    evidence_reference_cids=tuple(
                        request.evidence_reference_cids
                        or incident.evidence_cids
                    ),
                    policy=self.policy,
                    operation_catalog=self.operation_catalog,
                )
            except RescuePlannerValidationError as exc:
                quarantine = exc.reason_code in {
                    "forbidden_content",
                    "self_authorization",
                }
                result = self._no_plan(
                    request,
                    f"provider_{exc.reason_code}",
                    quarantine=quarantine,
                    provider_invoked=True,
                    prompt_sha256=prompt_sha256,
                    response_sha256=response_sha256,
                    estimated_cost_microunits=estimated_cost,
                    elapsed_ms=elapsed,
                )
                return self._record_failure(request, result, cooldown_key)
            except Exception:
                result = self._no_plan(
                    request,
                    "provider_malformed_plan",
                    provider_invoked=True,
                    prompt_sha256=prompt_sha256,
                    response_sha256=response_sha256,
                    estimated_cost_microunits=estimated_cost,
                    elapsed_ms=elapsed,
                )
                return self._record_failure(request, result, cooldown_key)

            result = RescuePlanningResult(
                disposition=RescuePlanningDisposition.PROPOSED,
                incident_cid=incident.incident_cid,
                exhaustion_receipt_cid=exhaustion.receipt_cid,
                plan=plan,
                provider_invoked=True,
                reason_code="validated_proposal",
                prompt_sha256=prompt_sha256,
                response_sha256=response_sha256,
                estimated_cost_microunits=estimated_cost,
                elapsed_ms=elapsed,
            )
            self.state.consecutive_failures[cooldown_key] = 0
            self.state.prior_results[incident.incident_cid] = result
            return result

    propose = plan


def plan_rescue(
    request: RescuePlanningRequest,
    *,
    policy: RescuePlannerPolicy,
    provider: RescueProvider | None = None,
    state: RescuePlannerState | None = None,
    clock_ms: Callable[[], int] | None = None,
) -> RescuePlanningResult:
    """Functional entry point for one bounded rescue proposal."""

    return RescuePlanner(
        policy,
        provider=provider,
        state=state,
        clock_ms=clock_ms,
    ).plan(request)


# Names used by adjacent supervisor components and older planning call sites.
ExhaustionGatedRescuePlanner = RescuePlanner
RescuePlanPolicy = RescuePlannerPolicy
RescuePlanRequest = RescuePlanningRequest
RescuePlanResult = RescuePlanningResult
parse_rescue_plan_v1 = parse_rescue_plan
build_rescue_plan_prompt = build_rescue_prompt


__all__ = [
    "DEFAULT_MAX_JSON_DEPTH",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_RESCUE_MODEL",
    "DEFAULT_RESCUE_OPERATION_CATALOG",
    "DEFAULT_RESCUE_ROUTE",
    "ExhaustionGatedRescuePlanner",
    "RESCUE_PLAN_RESPONSE_NAME",
    "RESCUE_PLAN_V1_JSON_SCHEMA",
    "RescueGuidanceStep",
    "RescueNoPlanGuidance",
    "RescueOperationSpec",
    "RescueParameterSpec",
    "RescuePlanPolicy",
    "RescuePlanRequest",
    "RescuePlanResult",
    "RescuePlanner",
    "RescuePlannerError",
    "RescuePlannerPolicy",
    "RescuePlannerState",
    "RescuePlannerValidationError",
    "RescuePlanningDisposition",
    "RescuePlanningRequest",
    "RescuePlanningResult",
    "build_rescue_plan_prompt",
    "build_rescue_prompt",
    "default_rescue_operation_catalog",
    "parse_rescue_plan",
    "parse_rescue_plan_v1",
    "plan_rescue",
]
