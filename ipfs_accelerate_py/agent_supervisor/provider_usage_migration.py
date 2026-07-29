"""Complete supervisor provider-callsite migration and coverage gate (ASI-168).

Every in-scope supervisor consumer routes chargeable provider work through the
reservation-aware gateway (or a typed contract-equivalent adapter) while
preserving:

* deterministic / local fallback behaviour
* proof and completion authority boundaries (model output never authorizes)
* off-mode compatibility with existing focused suites

This module also owns the generated AST/import/runtime inventory that rejects
unregistered direct provider imports, wrapper aliases, subprocess bypasses,
missing attribution, and receipt drops. Provider-free discovery remains
allowlisted.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Optional, Sequence

from ipfs_accelerate_py.endpoint_usage import UsageVector
from ipfs_accelerate_py.endpoint_usage.schema import LimitWindow, WindowKind

from .provider_execution import (
    ProviderExecutionError,
    ProviderExecutionGateway,
    ProviderExecutionPhase,
    ProviderExecutionRequest,
    ProviderExecutionResult,
    SideEffectBoundary,
    build_execution_request,
    new_attempt_idempotency_key,
)
from .provider_usage import (
    SupervisorToEndpointRequest,
    SupervisorUsageBudget,
    SupervisorUsageEnvelope,
    SupervisorUsageFinalStatus,
    SupervisorUsageLevel,
    SupervisorUsageReceipt,
    SupervisorUsageScope,
    build_child_envelope,
)


COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID: Final[str] = (
    "requirement:complete-provider-callsite.v1"
)
COMPLETE_PROVIDER_CALLSITE_GOAL_ID: Final[str] = "ASI-G520"
PROVIDER_USAGE_MIGRATION_CONTRACT_VERSION: Final[str] = (
    "supervisor-provider-usage-migration/v1"
)
PROVIDER_USAGE_MIGRATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-usage-migration@1"
)
CALLSITE_INVENTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/provider-callsite-inventory@1"
)
MIGRATED_CALL_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/migrated-provider-call-receipt@1"
)
NON_METERABLE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/non-meterable-provider-result@1"
)

# Authority bounds — migration never grants completion or authorization power.
MIGRATION_IS_COMPLETION_EVIDENCE: Final[bool] = False
MIGRATION_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
MIGRATION_AUTHORIZES_USAGE: Final[bool] = False
MIGRATION_MAY_RETRY_SIDE_EFFECTING_AGENT_WORK: Final[bool] = False
MIGRATION_MAY_ROUTE_TO_FORBIDDEN_ENDPOINT: Final[bool] = False
MIGRATION_MAY_CHANGE_PROMPT_SOURCE_OUTPUT_CONTRACTS: Final[bool] = False

# Conservative reviewed ceiling for typed non-meterable results under enforce.
# Units are request counts; callers may only lower this reviewed default.
DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS: Final[int] = 1
MAX_NON_METERABLE_ENFORCE_CEILING_REQUESTS: Final[int] = 8

_USAGE_MODE_ENV: Final[str] = "IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE"
_TEXT_SAFE = re.compile(r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+$")


class ProviderUsageMigrationError(RuntimeError):
    """Fail-closed migration / inventory error with stable reason codes."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.reason_codes = tuple(str(code) for code in reason_codes if str(code))


class SupervisorUsageMode(str, Enum):
    """Staged usage-aware migration modes (compatible with endpoint usage)."""

    OFF = "off"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ASSIST = "assist"
    ENFORCE = "enforce"


class ConsumerId(str, Enum):
    """Closed in-scope supervisor provider consumer population (ASI-168)."""

    TASK_PROPOSAL_ROUTER = "task_proposal_router"
    PROMPT_GOAL_PLANNER = "prompt_goal_planner"
    RESCUE_PLANNER = "rescue_planner"
    LEANSTRAL_PROOF_PROVIDER = "leanstral_proof_provider"
    LEANSTRAL_GOAL_DEVELOPMENT = "leanstral_goal_development"


class CallsiteKind(str, Enum):
    GATEWAY = "gateway"
    TYPED_ADAPTER = "typed_adapter"
    CHILD_PROCESS = "child_process"
    CLI_AGENT = "cli_agent"
    BATCH_SCHEDULER = "batch_scheduler"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    PROVIDER_FREE = "provider_free"


class CallsiteViolationKind(str, Enum):
    UNREGISTERED_DIRECT_IMPORT = "unregistered_direct_import"
    UNREGISTERED_DIRECT_CALL = "unregistered_direct_call"
    WRAPPER_ALIAS = "wrapper_alias"
    SUBPROCESS_BYPASS = "subprocess_bypass"
    MISSING_ATTRIBUTION = "missing_attribution"
    RECEIPT_DROP = "receipt_drop"


# Canonical ASREF-landed module paths for the closed consumer population.
IN_SCOPE_CONSUMER_MODULES: Final[Mapping[ConsumerId, str]] = {
    ConsumerId.TASK_PROPOSAL_ROUTER: (
        "ipfs_accelerate_py/agent_supervisor/planning/task_proposal_router.py"
    ),
    ConsumerId.PROMPT_GOAL_PLANNER: (
        "ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py"
    ),
    ConsumerId.RESCUE_PLANNER: (
        "ipfs_accelerate_py/agent_supervisor/rescue/rescue_planner.py"
    ),
    ConsumerId.LEANSTRAL_PROOF_PROVIDER: (
        "ipfs_accelerate_py/agent_supervisor/proof/leanstral_proof_provider.py"
    ),
    ConsumerId.LEANSTRAL_GOAL_DEVELOPMENT: (
        "ipfs_accelerate_py/agent_supervisor/proof/leanstral_goal_development.py"
    ),
}

# Declared historical flat stems (taskboard outputs) map to landed owners.
DECLARED_FLAT_CONSUMER_STEMS: Final[Mapping[str, ConsumerId]] = {
    "task_proposal_router": ConsumerId.TASK_PROPOSAL_ROUTER,
    "prompt_goal_planner": ConsumerId.PROMPT_GOAL_PLANNER,
    "rescue_planner": ConsumerId.RESCUE_PLANNER,
    "leanstral_proof_provider": ConsumerId.LEANSTRAL_PROOF_PROVIDER,
    "leanstral_goal_development": ConsumerId.LEANSTRAL_GOAL_DEVELOPMENT,
}

# Symbols that must not appear as unregistered direct provider entrypoints.
_FORBIDDEN_PROVIDER_IMPORT_MODULES: Final[frozenset[str]] = frozenset(
    {
        "ipfs_accelerate_py.llm_router",
        "ipfs_datasets_py.llm_router",
        "llm_router",
    }
)
_FORBIDDEN_PROVIDER_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "generate_text",
        "generate_text_batch",
        "chat_completion",
        "complete",
    }
)
_FORBIDDEN_SUBPROCESS_NAMES: Final[frozenset[str]] = frozenset(
    {
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_output",
        "os.system",
        "os.popen",
    }
)
# Allowlisted provider-free discovery modules (never fail coverage for these).
PROVIDER_FREE_DISCOVERY_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "ipfs_accelerate_py/agent_supervisor/provider_usage.py",
        "ipfs_accelerate_py/agent_supervisor/provider_usage_migration.py",
        "ipfs_accelerate_py/agent_supervisor/provider_execution.py",
        "ipfs_accelerate_py/agent_supervisor/asref_layout_evidence.py",
    }
)

# Approved gateway / adapter entry symbols that satisfy coverage.
_APPROVED_GATEWAY_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "dispatch_migrated_provider_call",
        "execute_via_provider_gateway",
        "ProviderExecutionGateway",
        "build_execution_request",
        "migrated_text_provider",
        "consume_usage_receipt",
        "call_llm_router",  # typed child-process adapter (ASI-166)
        "_call_text_provider",  # batch-or-adapter dispatcher
        "scheduler.execute",
        "ProviderBatchScheduler",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise ProviderUsageMigrationError(
                f"{name} is required",
                reason_codes=("missing_field", name),
            )
        return ""
    text = str(value).strip()
    if required and not text:
        raise ProviderUsageMigrationError(
            f"{name} must not be empty",
            reason_codes=("empty_field", name),
        )
    if text and not _TEXT_SAFE.fullmatch(text):
        raise ProviderUsageMigrationError(
            f"{name} contains control characters",
            reason_codes=("unsafe_text", name),
        )
    return text


def resolve_usage_mode(
    mode: SupervisorUsageMode | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> SupervisorUsageMode:
    """Resolve the active migration mode from an argument or environment."""

    if isinstance(mode, SupervisorUsageMode):
        return mode
    if mode is not None and str(mode).strip():
        raw = str(mode).strip()
        # Accept enum-like values and plain mode strings.
        if "." in raw and raw.rsplit(".", 1)[-1].casefold() in {
            item.value for item in SupervisorUsageMode
        }:
            raw = raw.rsplit(".", 1)[-1]
        try:
            return SupervisorUsageMode(raw.casefold())
        except ValueError as exc:
            raise ProviderUsageMigrationError(
                f"unknown usage mode: {mode!r}",
                reason_codes=("unknown_usage_mode",),
            ) from exc
    env = environ if environ is not None else os.environ
    raw = str(env.get(_USAGE_MODE_ENV, SupervisorUsageMode.OFF.value)).strip()
    if not raw:
        return SupervisorUsageMode.OFF
    try:
        return SupervisorUsageMode(raw.casefold())
    except ValueError as exc:
        raise ProviderUsageMigrationError(
            f"unknown usage mode from env: {raw!r}",
            reason_codes=("unknown_usage_mode",),
        ) from exc


def mode_requires_envelope(mode: SupervisorUsageMode) -> bool:
    return mode is not SupervisorUsageMode.OFF


def mode_requires_receipt(mode: SupervisorUsageMode) -> bool:
    return mode in {
        SupervisorUsageMode.ASSIST,
        SupervisorUsageMode.ENFORCE,
    }


def mode_admits_non_meterable_under_ceiling(
    mode: SupervisorUsageMode,
) -> bool:
    """Non-meterable child/CLI results need a reviewed ceiling only in enforce."""

    return mode is SupervisorUsageMode.ENFORCE


@dataclass(frozen=True)
class ConsumerCallContext:
    """Run/goal/task/attempt/stage/lane/request lineage for one provider call."""

    consumer_id: ConsumerId
    repository_id: str
    state_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    catalog_revision: str
    usage_revision: str
    supervisor_run_id: str
    goal_id: str
    objective_id: str
    objective_revision: str
    task_id: str
    attempt: int
    stage: str
    lane: str
    request_id: str
    endpoint_scope_id: str
    caller_id: str
    deadline_at: str
    idempotency_key: str
    lease_id: str
    fence_id: str
    provider_id: str
    modality: str = "text"
    side_effect: str = "generate_text"
    expected_output_kind: str = "text"
    estimated_requests: int = 1
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer_id", ConsumerId(self.consumer_id)
        )
        for name in (
            "repository_id",
            "state_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "catalog_revision",
            "usage_revision",
            "supervisor_run_id",
            "goal_id",
            "objective_id",
            "objective_revision",
            "task_id",
            "stage",
            "lane",
            "request_id",
            "endpoint_scope_id",
            "caller_id",
            "deadline_at",
            "idempotency_key",
            "lease_id",
            "fence_id",
            "provider_id",
            "modality",
            "side_effect",
            "expected_output_kind",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        attempt = int(self.attempt)
        if attempt < 1:
            raise ProviderUsageMigrationError(
                "attempt must be >= 1",
                reason_codes=("invalid_attempt",),
            )
        object.__setattr__(self, "attempt", attempt)
        for name in (
            "estimated_requests",
            "estimated_input_tokens",
            "estimated_output_tokens",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ProviderUsageMigrationError(
                    f"{name} must be non-negative",
                    reason_codes=("invalid_estimate", name),
                )
            object.__setattr__(self, name, value)
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", dict(meta))


@dataclass(frozen=True)
class NonMeterableProviderResult:
    """Typed result when child/CLI work cannot expose usage/reset metadata.

    Enforce mode admits these only under a conservative reviewed ceiling.
    """

    schema: str = NON_METERABLE_RESULT_SCHEMA
    reason_code: str = "usage_metadata_unavailable"
    consumer_id: str = ""
    provider_id: str = ""
    ceiling_requests: int = DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS
    output_text: str = ""
    usage_metadata: Mapping[str, Any] = field(default_factory=dict)
    reset_metadata: Mapping[str, Any] = field(default_factory=dict)
    is_completion_evidence: bool = False
    is_correctness_evidence: bool = False

    def __post_init__(self) -> None:
        if self.is_completion_evidence or self.is_correctness_evidence:
            raise ProviderUsageMigrationError(
                "non-meterable results cannot claim proof/completion authority",
                reason_codes=("authority_boundary",),
            )
        ceiling = int(self.ceiling_requests)
        if ceiling < 1 or ceiling > MAX_NON_METERABLE_ENFORCE_CEILING_REQUESTS:
            raise ProviderUsageMigrationError(
                "non-meterable ceiling outside reviewed bound",
                reason_codes=("ceiling_out_of_bounds",),
            )
        object.__setattr__(self, "ceiling_requests", ceiling)
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "usage_metadata",
            dict(self.usage_metadata)
            if isinstance(self.usage_metadata, Mapping)
            else {},
        )
        object.__setattr__(
            self,
            "reset_metadata",
            dict(self.reset_metadata)
            if isinstance(self.reset_metadata, Mapping)
            else {},
        )

    def to_record(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "contract_version": PROVIDER_USAGE_MIGRATION_CONTRACT_VERSION,
            "reason_code": self.reason_code,
            "consumer_id": self.consumer_id,
            "provider_id": self.provider_id,
            "ceiling_requests": self.ceiling_requests,
            "usage_metadata": dict(self.usage_metadata),
            "reset_metadata": dict(self.reset_metadata),
            "is_completion_evidence": False,
            "is_correctness_evidence": False,
            # Never embed model output in the durable record.
            "output_bytes": len(self.output_text.encode("utf-8")),
            "output_sha256": hashlib.sha256(
                self.output_text.encode("utf-8")
            ).hexdigest(),
        }
        payload["content_id"] = _content_id(payload)
        return payload


@dataclass(frozen=True)
class MigratedProviderCallResult:
    """Normalized consumer-facing outcome of one migrated provider call."""

    schema: str = MIGRATED_CALL_RECEIPT_SCHEMA
    mode: SupervisorUsageMode = SupervisorUsageMode.OFF
    consumer_id: ConsumerId = ConsumerId.TASK_PROPOSAL_ROUTER
    text: str = ""
    metered: bool = False
    phase: str = ""
    execution_result: Optional[ProviderExecutionResult] = None
    usage_receipt: Optional[SupervisorUsageReceipt] = None
    non_meterable: Optional[NonMeterableProviderResult] = None
    reason_codes: tuple[str, ...] = ()
    is_completion_evidence: bool = False
    is_correctness_evidence: bool = False

    def __post_init__(self) -> None:
        if self.is_completion_evidence or self.is_correctness_evidence:
            raise ProviderUsageMigrationError(
                "migrated call results cannot claim proof/completion authority",
                reason_codes=("authority_boundary",),
            )
        object.__setattr__(self, "mode", SupervisorUsageMode(self.mode))
        object.__setattr__(
            self, "consumer_id", ConsumerId(self.consumer_id)
        )
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_record(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "contract_version": PROVIDER_USAGE_MIGRATION_CONTRACT_VERSION,
            "requirement_id": COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID,
            "mode": self.mode.value,
            "consumer_id": self.consumer_id.value,
            "metered": self.metered,
            "phase": self.phase,
            "reason_codes": list(self.reason_codes),
            "usage_receipt": (
                self.usage_receipt.to_record()
                if self.usage_receipt is not None
                else None
            ),
            "execution_result": (
                self.execution_result.to_record()
                if self.execution_result is not None
                else None
            ),
            "non_meterable": (
                self.non_meterable.to_record()
                if self.non_meterable is not None
                else None
            ),
            "is_completion_evidence": False,
            "is_correctness_evidence": False,
            "output_bytes": len(self.text.encode("utf-8")),
            "output_sha256": hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest(),
        }
        payload["content_id"] = _content_id(payload)
        return payload


def build_request_lineage_envelope(
    context: ConsumerCallContext,
    *,
    deployment_budget_requests: int = 10_000,
) -> SupervisorUsageEnvelope:
    """Build a nested deployment→request envelope for one consumer call."""

    window = LimitWindow(kind=WindowKind.LIFETIME)

    def budget(*_args: Any, **dimensions: int) -> SupervisorUsageBudget:
        return SupervisorUsageBudget.of(window=window, currency="USD", **dimensions)

    base = {
        "repository_id": context.repository_id,
        "state_id": context.state_id,
        "tree_id": context.tree_id,
        "policy_id": context.policy_id,
        "policy_revision": context.policy_revision,
        "catalog_revision": context.catalog_revision,
        "usage_revision": context.usage_revision,
        "supervisor_run_id": "",
        "goal_id": "",
        "objective_id": "",
        "objective_revision": "",
        "task_id": "",
        "attempt": 0,
        "stage": "",
        "lane": "",
        "request_id": "",
        "endpoint_scope_id": "",
        "caller_id": "",
        "deadline_at": "",
        "idempotency_key": "",
        "lease_id": "",
        "fence_id": "",
        "parent_scope_id": "",
    }
    root = SupervisorUsageEnvelope(
        scope=SupervisorUsageScope(
            level=SupervisorUsageLevel.DEPLOYMENT,
            **base,
        ),
        budget=budget(requests=max(deployment_budget_requests, context.estimated_requests)),
    )
    run = build_child_envelope(
        root,
        level=SupervisorUsageLevel.RUN,
        budget=budget(requests=max(deployment_budget_requests // 2, context.estimated_requests)),
        supervisor_run_id=context.supervisor_run_id,
    )
    goal = build_child_envelope(
        run,
        level=SupervisorUsageLevel.GOAL,
        budget=budget(requests=max(deployment_budget_requests // 4, context.estimated_requests)),
        goal_id=context.goal_id,
        objective_id=context.objective_id,
        objective_revision=context.objective_revision,
    )
    task = build_child_envelope(
        goal,
        level=SupervisorUsageLevel.TASK,
        budget=budget(requests=max(deployment_budget_requests // 8, context.estimated_requests)),
        task_id=context.task_id,
    )
    attempt = build_child_envelope(
        task,
        level=SupervisorUsageLevel.ATTEMPT,
        budget=budget(requests=max(8, context.estimated_requests)),
        attempt=context.attempt,
    )
    stage = build_child_envelope(
        attempt,
        level=SupervisorUsageLevel.STAGE,
        budget=budget(requests=max(4, context.estimated_requests)),
        stage=context.stage,
    )
    lane = build_child_envelope(
        stage,
        level=SupervisorUsageLevel.LANE,
        budget=budget(requests=max(2, context.estimated_requests)),
        lane=context.lane,
    )
    request = build_child_envelope(
        lane,
        level=SupervisorUsageLevel.REQUEST,
        budget=budget(requests=max(1, context.estimated_requests)),
        request_id=context.request_id,
        endpoint_scope_id=context.endpoint_scope_id,
        caller_id=context.caller_id,
        deadline_at=context.deadline_at,
        idempotency_key=context.idempotency_key,
        lease_id=context.lease_id,
        fence_id=context.fence_id,
    )
    # Rebuild nested tree bottom-up so parent validation sees children.
    lane = SupervisorUsageEnvelope(
        scope=lane.scope, budget=lane.budget, children=(request,)
    )
    stage = SupervisorUsageEnvelope(
        scope=stage.scope, budget=stage.budget, children=(lane,)
    )
    attempt = SupervisorUsageEnvelope(
        scope=attempt.scope, budget=attempt.budget, children=(stage,)
    )
    task = SupervisorUsageEnvelope(
        scope=task.scope, budget=task.budget, children=(attempt,)
    )
    goal = SupervisorUsageEnvelope(
        scope=goal.scope, budget=goal.budget, children=(task,)
    )
    run = SupervisorUsageEnvelope(
        scope=run.scope, budget=run.budget, children=(goal,)
    )
    return SupervisorUsageEnvelope(
        scope=root.scope, budget=root.budget, children=(run,)
    )


def request_leaf_envelope(
    lineage: SupervisorUsageEnvelope,
) -> SupervisorUsageEnvelope:
    node = lineage
    while node.children:
        node = node.children[0]
    if node.scope.level is not SupervisorUsageLevel.REQUEST:
        raise ProviderUsageMigrationError(
            "lineage does not terminate at a request envelope",
            reason_codes=("missing_request_envelope",),
        )
    return node


def build_bridge_request(
    context: ConsumerCallContext,
    envelope: SupervisorUsageEnvelope,
) -> SupervisorToEndpointRequest:
    scope = envelope.scope
    estimated_kwargs: dict[str, int] = {
        "requests": max(1, context.estimated_requests),
    }
    if context.estimated_input_tokens:
        estimated_kwargs["input_tokens"] = context.estimated_input_tokens
    if context.estimated_output_tokens:
        estimated_kwargs["output_tokens"] = context.estimated_output_tokens
    return SupervisorToEndpointRequest(
        scope=scope,
        envelope_id=envelope.envelope_id,
        endpoint_scope_id=scope.endpoint_scope_id,
        catalog_revision=scope.catalog_revision,
        usage_revision=scope.usage_revision,
        estimated=UsageVector.of(**estimated_kwargs),
        request_id=scope.request_id,
        attempt=scope.attempt,
        idempotency_key=scope.idempotency_key,
        caller_id=scope.caller_id,
        deadline_at=scope.deadline_at,
        lease_id=scope.lease_id,
        fence_id=scope.fence_id,
    )


def consume_usage_receipt(
    result: ProviderExecutionResult,
    *,
    expected_request_id: str = "",
) -> SupervisorUsageReceipt:
    """Require and return the operational usage receipt from a gateway result."""

    if result.is_completion_evidence or result.is_correctness_evidence:
        raise ProviderUsageMigrationError(
            "gateway result claimed forbidden authority",
            reason_codes=("authority_boundary",),
        )
    receipt = result.receipt
    if receipt is None:
        raise ProviderUsageMigrationError(
            "provider call dropped usage receipt",
            reason_codes=("receipt_drop",),
        )
    if receipt.is_completion_evidence or receipt.is_correctness_evidence:
        raise ProviderUsageMigrationError(
            "usage receipt claimed forbidden authority",
            reason_codes=("authority_boundary",),
        )
    if expected_request_id and receipt.request_id != expected_request_id:
        raise ProviderUsageMigrationError(
            "usage receipt request_id is foreign to the call",
            reason_codes=("missing_attribution",),
        )
    return receipt


def extract_child_usage_metadata(
    payload: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Extract structured usage/reset metadata from a child/CLI payload."""

    if not isinstance(payload, Mapping):
        return {}, {}
    usage = payload.get("usage") or payload.get("usage_metadata") or {}
    reset = payload.get("reset") or payload.get("reset_metadata") or {}
    if not isinstance(usage, Mapping):
        usage = {}
    if not isinstance(reset, Mapping):
        reset = {}
    # Redact secret-shaped keys only (exact / suffix match — not "input_tokens").
    def _is_secret_key(key: str) -> bool:
        lowered = str(key).casefold()
        if lowered in {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
        }:
            return False
        if lowered.endswith(("_input_tokens", "_output_tokens", "_total_tokens")):
            return False
        return (
            lowered in {"secret", "token", "password", "api_key", "prompt", "authorization"}
            or lowered.endswith("_secret")
            or lowered.endswith("_password")
            or lowered.endswith("_api_key")
            or lowered.endswith("_token")
        )

    safe_usage = {
        str(key): value
        for key, value in usage.items()
        if not _is_secret_key(str(key))
    }
    safe_reset = {
        str(key): value
        for key, value in reset.items()
        if not _is_secret_key(str(key))
    }
    return safe_usage, safe_reset


def non_meterable_from_child(
    *,
    consumer_id: ConsumerId | str,
    provider_id: str,
    output_text: str = "",
    payload: Any = None,
    ceiling_requests: int = DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
) -> NonMeterableProviderResult:
    usage, reset = extract_child_usage_metadata(payload)
    reason = (
        "structured_usage_present"
        if usage or reset
        else "usage_metadata_unavailable"
    )
    return NonMeterableProviderResult(
        reason_code=reason,
        consumer_id=ConsumerId(consumer_id).value,
        provider_id=_text(provider_id, "provider_id"),
        ceiling_requests=min(
            max(1, int(ceiling_requests)),
            MAX_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
        ),
        output_text=str(output_text or ""),
        usage_metadata=usage,
        reset_metadata=reset,
    )


def admit_non_meterable(
    result: NonMeterableProviderResult,
    *,
    mode: SupervisorUsageMode,
    reviewed_ceiling: int = DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
) -> NonMeterableProviderResult:
    """Enforce mode admits non-meterable results only under a reviewed ceiling."""

    if not mode_admits_non_meterable_under_ceiling(mode):
        return result
    ceiling = min(
        max(1, int(reviewed_ceiling)),
        MAX_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
    )
    if result.ceiling_requests > ceiling:
        raise ProviderUsageMigrationError(
            "non-meterable result exceeds reviewed enforce ceiling",
            reason_codes=("non_meterable_ceiling_exceeded",),
        )
    return result


def build_consumer_call_context(
    *,
    consumer_id: ConsumerId | str,
    provider_id: str,
    repository_id: str = "repository:supervisor",
    state_id: str = "state:default",
    tree_id: str = "tree:unknown",
    policy_id: str = "policy:supervisor",
    policy_revision: str = "policy:supervisor@1",
    catalog_revision: str = "catalog:rev-1",
    usage_revision: str = "usage:rev-1",
    supervisor_run_id: str = "",
    goal_id: str = "",
    objective_id: str = "",
    objective_revision: str = "",
    task_id: str = "",
    attempt: int = 1,
    stage: str = "provider",
    lane: str = "lane-0",
    request_id: str = "",
    endpoint_scope_id: str = "endpoint:scope:default",
    caller_id: str = "caller:supervisor",
    deadline_at: str = "2099-01-01T00:00:00Z",
    idempotency_key: str = "",
    lease_id: str = "lease:default",
    fence_id: str = "fence:default",
    modality: str = "text",
    side_effect: str = "generate_text",
    expected_output_kind: str = "text",
    estimated_requests: int = 1,
    estimated_input_tokens: int = 0,
    estimated_output_tokens: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> ConsumerCallContext:
    """Build a fully-attributed call context with stable defaults."""

    cid = ConsumerId(consumer_id)
    req = request_id or f"request:{cid.value}:{uuid.uuid4().hex[:12]}"
    run = supervisor_run_id or f"run:{cid.value}"
    goal = goal_id or f"goal:{cid.value}"
    objective = objective_id or cid.value
    task = task_id or cid.value
    idem = idempotency_key or f"idem:{req}"
    return ConsumerCallContext(
        consumer_id=cid,
        repository_id=repository_id,
        state_id=state_id,
        tree_id=tree_id,
        policy_id=policy_id,
        policy_revision=policy_revision,
        catalog_revision=catalog_revision,
        usage_revision=usage_revision,
        supervisor_run_id=run,
        goal_id=goal,
        objective_id=objective,
        objective_revision=objective_revision or f"{objective}@1",
        task_id=task,
        attempt=attempt,
        stage=stage,
        lane=lane,
        request_id=req,
        endpoint_scope_id=endpoint_scope_id,
        caller_id=caller_id,
        deadline_at=deadline_at,
        idempotency_key=idem,
        lease_id=lease_id,
        fence_id=fence_id,
        provider_id=provider_id,
        modality=modality,
        side_effect=side_effect,
        expected_output_kind=expected_output_kind,
        estimated_requests=estimated_requests,
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        metadata=dict(metadata or {}),
    )


def dispatch_migrated_provider_call(
    *,
    context: ConsumerCallContext,
    invoke: Callable[[], str],
    mode: SupervisorUsageMode | str | None = None,
    gateway: Optional[ProviderExecutionGateway] = None,
    cancelled: bool = False,
    child_payload: Any = None,
    non_meterable_ceiling: int = DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS,
) -> MigratedProviderCallResult:
    """Route one provider call through the gateway (or preserve off-mode path).

    ``invoke`` is the existing consumer-side call (subprocess adapter, router,
    or injected test double). The migration never retries side-effecting work:
    a single invoke occurs per dispatch.
    """

    resolved_mode = resolve_usage_mode(mode)
    if not mode_requires_envelope(resolved_mode):
        text = invoke()
        return MigratedProviderCallResult(
            mode=resolved_mode,
            consumer_id=context.consumer_id,
            text=str(text if text is not None else ""),
            metered=False,
            phase="off",
            reason_codes=("usage_mode_off",),
        )

    lineage = build_request_lineage_envelope(context)
    envelope = request_leaf_envelope(lineage)
    bridge = build_bridge_request(context, envelope)
    # Map legacy migration side_effect labels onto ProviderExecutionRequest fields.
    side_effect_raw = str(context.side_effect or "generate_text").strip().lower()
    if side_effect_raw in {"none", "read_only", "readonly"}:
        side_effect_boundary = SideEffectBoundary.READ_ONLY
    elif side_effect_raw in {"idempotent", "cache", "lookup"}:
        side_effect_boundary = SideEffectBoundary.IDEMPOTENT
    else:
        # generate_text / agent / tool / mutate → side-effecting by default
        side_effect_boundary = SideEffectBoundary.SIDE_EFFECTING
    request = build_execution_request(
        bridge=bridge,
        envelope=envelope,
        provider_id=context.provider_id,
        modality=context.modality,
        side_effect_boundary=side_effect_boundary,
        operation=side_effect_raw or "text.generate",
        cancelled=cancelled,
        metadata={
            **dict(context.metadata),
            "consumer_id": context.consumer_id.value,
            "migration_requirement": COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID,
            "expected_output_kind": context.expected_output_kind,
        },
    )

    invoked: dict[str, Any] = {"done": False, "text": ""}

    def _invoker(_req: ProviderExecutionRequest) -> Mapping[str, Any]:
        if invoked["done"]:
            # Exact replay protection is owned by the gateway; consumers never
            # re-enter side-effecting agent work through this adapter.
            raise ProviderUsageMigrationError(
                "migration refuses to retry side-effecting provider work",
                reason_codes=("side_effect_retry_forbidden",),
            )
        invoked["done"] = True
        text = invoke()
        invoked["text"] = str(text if text is not None else "")
        observation: dict[str, Any] = {
            "provider_id": context.provider_id,
            "endpoint": context.endpoint_scope_id,
            "status": "ok",
            "output_kind": context.expected_output_kind,
            "units": UsageVector.of(
                requests=max(1, context.estimated_requests)
            ).to_dict(),
        }
        usage, reset = extract_child_usage_metadata(child_payload)
        if usage:
            observation["usage"] = dict(usage)
        if reset:
            observation["reset"] = dict(reset)
        return observation

    active_gateway = gateway or ProviderExecutionGateway(invoker=_invoker)
    if gateway is not None and gateway is not active_gateway:
        pass  # pragma: no cover - defensive
    # Prefer the caller's gateway when provided; otherwise the simulated one
    # above already has the invoker bound.
    if gateway is not None:
        # Re-bind invoker only when the gateway has no invoker of its own.
        if getattr(gateway, "_invoker", None) is None:
            active_gateway = ProviderExecutionGateway(
                coordinator=getattr(gateway, "_coordinator", None),
                invoker=_invoker,
                owner_id=getattr(gateway, "_owner_id", "supervisor-provider-execution"),
                reservation_ttl_ms=getattr(gateway, "_reservation_ttl_ms", 60_000),
            )
        else:
            # Wrap existing invoker: still a single side-effect path.
            original = gateway._invoker

            def _wrapped(req: ProviderExecutionRequest) -> Mapping[str, Any]:
                if invoked["done"]:
                    raise ProviderUsageMigrationError(
                        "migration refuses to retry side-effecting provider work",
                        reason_codes=("side_effect_retry_forbidden",),
                    )
                invoked["done"] = True
                # Existing gateway invoker owns the remote call; still capture text
                # from the consumer invoke path for return compatibility.
                text = invoke()
                invoked["text"] = str(text if text is not None else "")
                raw = original(req)
                if not isinstance(raw, Mapping):
                    raise ProviderUsageMigrationError(
                        "gateway invoker must return a mapping",
                        reason_codes=("invalid_observation",),
                    )
                return dict(raw)

            active_gateway = ProviderExecutionGateway(
                coordinator=getattr(gateway, "_coordinator", None),
                invoker=_wrapped,
                owner_id=getattr(gateway, "_owner_id", "supervisor-provider-execution"),
                reservation_ttl_ms=getattr(gateway, "_reservation_ttl_ms", 60_000),
            )

    try:
        execution = active_gateway.execute(request)
    except ProviderExecutionError as exc:
        raise ProviderUsageMigrationError(
            str(exc),
            reason_codes=exc.reason_codes or ("gateway_error",),
        ) from exc

    # Observe/shadow may continue without a hard receipt; assist/enforce require it.
    receipt: Optional[SupervisorUsageReceipt] = None
    if execution.phase is ProviderExecutionPhase.SETTLED:
        receipt = consume_usage_receipt(
            execution, expected_request_id=context.request_id
        )
    elif mode_requires_receipt(resolved_mode):
        if execution.phase in {
            ProviderExecutionPhase.DENIED,
            ProviderExecutionPhase.CANCELLED,
            ProviderExecutionPhase.FAILED,
        }:
            # Capacity / cancel paths are typed backpressure, not receipt drops.
            non_meterable = admit_non_meterable(
                non_meterable_from_child(
                    consumer_id=context.consumer_id,
                    provider_id=context.provider_id,
                    output_text=str(invoked.get("text") or ""),
                    payload=child_payload,
                    ceiling_requests=non_meterable_ceiling,
                ),
                mode=resolved_mode,
                reviewed_ceiling=non_meterable_ceiling,
            )
            return MigratedProviderCallResult(
                mode=resolved_mode,
                consumer_id=context.consumer_id,
                text=str(invoked.get("text") or ""),
                metered=False,
                phase=execution.phase.value,
                execution_result=execution,
                non_meterable=non_meterable,
                reason_codes=execution.reason_codes + ("typed_backpressure",),
            )
        raise ProviderUsageMigrationError(
            "enforce/assist mode requires a settled usage receipt",
            reason_codes=("receipt_drop", execution.phase.value),
        )

    text = str(invoked.get("text") or "")
    if not text and isinstance(execution.observation, Mapping):
        # Gateway-only invoker paths may not populate consumer text.
        text = str(execution.observation.get("text") or "")

    # When the child path produced no structured usage under enforce, attach a
    # typed non-meterable admission under the reviewed ceiling.
    non_meterable: Optional[NonMeterableProviderResult] = None
    if (
        resolved_mode is SupervisorUsageMode.ENFORCE
        and receipt is not None
        and not extract_child_usage_metadata(child_payload)[0]
    ):
        non_meterable = admit_non_meterable(
            non_meterable_from_child(
                consumer_id=context.consumer_id,
                provider_id=context.provider_id,
                output_text=text,
                payload=child_payload,
                ceiling_requests=non_meterable_ceiling,
            ),
            mode=resolved_mode,
            reviewed_ceiling=non_meterable_ceiling,
        )

    return MigratedProviderCallResult(
        mode=resolved_mode,
        consumer_id=context.consumer_id,
        text=text,
        metered=receipt is not None,
        phase=execution.phase.value,
        execution_result=execution,
        usage_receipt=receipt,
        non_meterable=non_meterable,
        reason_codes=execution.reason_codes,
    )


def migrated_text_provider(
    *,
    consumer_id: ConsumerId | str,
    provider_id: str,
    invoke: Callable[[], str],
    mode: SupervisorUsageMode | str | None = None,
    gateway: Optional[ProviderExecutionGateway] = None,
    context: Optional[ConsumerCallContext] = None,
    **context_kwargs: Any,
) -> MigratedProviderCallResult:
    """Convenience wrapper used by consumer modules."""

    ctx = context or build_consumer_call_context(
        consumer_id=consumer_id,
        provider_id=provider_id,
        **context_kwargs,
    )
    return dispatch_migrated_provider_call(
        context=ctx,
        invoke=invoke,
        mode=mode,
        gateway=gateway,
    )


# ---------------------------------------------------------------------------
# AST inventory + coverage gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallsiteRecord:
    """One discovered provider-related callsite or import."""

    module_path: str
    kind: str
    symbol: str
    lineno: int
    registered: bool
    allowlisted: bool = False
    consumer_id: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "kind": self.kind,
            "symbol": self.symbol,
            "lineno": self.lineno,
            "registered": self.registered,
            "allowlisted": self.allowlisted,
            "consumer_id": self.consumer_id,
        }


@dataclass(frozen=True)
class CallsiteViolation:
    kind: CallsiteViolationKind
    module_path: str
    symbol: str
    lineno: int
    detail: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "module_path": self.module_path,
            "symbol": self.symbol,
            "lineno": self.lineno,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CallsiteInventory:
    schema: str = CALLSITE_INVENTORY_SCHEMA
    requirement_id: str = COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID
    consumers: tuple[str, ...] = ()
    callsites: tuple[CallsiteRecord, ...] = ()
    violations: tuple[CallsiteViolation, ...] = ()
    coverage_complete: bool = False

    def to_record(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "contract_version": PROVIDER_USAGE_MIGRATION_CONTRACT_VERSION,
            "requirement_id": self.requirement_id,
            "goal_id": COMPLETE_PROVIDER_CALLSITE_GOAL_ID,
            "consumers": list(self.consumers),
            "callsites": [item.to_record() for item in self.callsites],
            "violations": [item.to_record() for item in self.violations],
            "coverage_complete": self.coverage_complete,
            "is_completion_evidence": False,
            "is_correctness_evidence": False,
        }
        payload["content_id"] = _content_id(payload)
        return payload


def _module_consumer_id(module_path: str) -> str:
    normalized = module_path.replace("\\", "/")
    for consumer, path in IN_SCOPE_CONSUMER_MODULES.items():
        if normalized.endswith(path) or normalized == path:
            return consumer.value
        # Also match declared flat stems.
        stem = Path(path).stem
        if normalized.endswith(f"/{stem}.py"):
            return consumer.value
    return ""


def _is_allowlisted_path(module_path: str) -> bool:
    normalized = module_path.replace("\\", "/")
    for allowed in PROVIDER_FREE_DISCOVERY_ALLOWLIST:
        if normalized.endswith(allowed) or normalized == allowed:
            return True
    return False


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def scan_module_callsites(
    source: str,
    *,
    module_path: str,
    registered_symbols: frozenset[str] | set[str] | None = None,
) -> tuple[list[CallsiteRecord], list[CallsiteViolation]]:
    """Scan one module for provider imports/calls and coverage violations."""

    registered = frozenset(registered_symbols or _APPROVED_GATEWAY_SYMBOLS)
    allowlisted = _is_allowlisted_path(module_path)
    consumer_id = _module_consumer_id(module_path)
    records: list[CallsiteRecord] = []
    violations: list[CallsiteViolation] = []

    try:
        tree = ast.parse(source, filename=module_path)
    except SyntaxError as exc:
        violations.append(
            CallsiteViolation(
                kind=CallsiteViolationKind.UNREGISTERED_DIRECT_CALL,
                module_path=module_path,
                symbol="<syntax>",
                lineno=getattr(exc, "lineno", 0) or 0,
                detail=f"syntax error: {exc.msg}",
            )
        )
        return records, violations

    # Track whether the module uses an approved migration/gateway entry.
    uses_approved = False
    imports_migration = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                asname = alias.asname or name
                if name in _FORBIDDEN_PROVIDER_IMPORT_MODULES or name.endswith(
                    ".llm_router"
                ):
                    records.append(
                        CallsiteRecord(
                            module_path=module_path,
                            kind="import",
                            symbol=name,
                            lineno=getattr(node, "lineno", 0) or 0,
                            registered=allowlisted or bool(consumer_id),
                            allowlisted=allowlisted,
                            consumer_id=consumer_id,
                        )
                    )
                    if alias.asname and alias.asname not in {
                        "llm_router",
                        name.rsplit(".", 1)[-1],
                    }:
                        violations.append(
                            CallsiteViolation(
                                kind=CallsiteViolationKind.WRAPPER_ALIAS,
                                module_path=module_path,
                                symbol=f"{name} as {asname}",
                                lineno=getattr(node, "lineno", 0) or 0,
                                detail="provider import bound to wrapper alias",
                            )
                        )
                    if not allowlisted and not consumer_id:
                        violations.append(
                            CallsiteViolation(
                                kind=CallsiteViolationKind.UNREGISTERED_DIRECT_IMPORT,
                                module_path=module_path,
                                symbol=name,
                                lineno=getattr(node, "lineno", 0) or 0,
                            )
                        )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in _FORBIDDEN_PROVIDER_IMPORT_MODULES or module.endswith(
                ".llm_router"
            ):
                for alias in node.names:
                    symbol = f"{module}.{alias.name}"
                    records.append(
                        CallsiteRecord(
                            module_path=module_path,
                            kind="import_from",
                            symbol=symbol,
                            lineno=getattr(node, "lineno", 0) or 0,
                            registered=allowlisted or bool(consumer_id),
                            allowlisted=allowlisted,
                            consumer_id=consumer_id,
                        )
                    )
                    if alias.asname and alias.asname != alias.name:
                        violations.append(
                            CallsiteViolation(
                                kind=CallsiteViolationKind.WRAPPER_ALIAS,
                                module_path=module_path,
                                symbol=f"{symbol} as {alias.asname}",
                                lineno=getattr(node, "lineno", 0) or 0,
                            )
                        )
                    if not allowlisted and not consumer_id:
                        violations.append(
                            CallsiteViolation(
                                kind=CallsiteViolationKind.UNREGISTERED_DIRECT_IMPORT,
                                module_path=module_path,
                                symbol=symbol,
                                lineno=getattr(node, "lineno", 0) or 0,
                            )
                        )
            if module.endswith("provider_usage_migration") or module.endswith(
                "provider_execution"
            ):
                imports_migration = True
                uses_approved = True
            if any(
                alias.name in registered for alias in node.names
            ) and (
                "provider_usage_migration" in module
                or "provider_execution" in module
                or "todo_daemon.llm" in module
                or "provider_batch_scheduler" in module
            ):
                uses_approved = True
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            short = name.rsplit(".", 1)[-1] if name else ""
            if short in _FORBIDDEN_PROVIDER_CALL_NAMES or name in _FORBIDDEN_PROVIDER_CALL_NAMES:
                registered_call = uses_approved or allowlisted or bool(consumer_id)
                records.append(
                    CallsiteRecord(
                        module_path=module_path,
                        kind="call",
                        symbol=name or short,
                        lineno=getattr(node, "lineno", 0) or 0,
                        registered=registered_call,
                        allowlisted=allowlisted,
                        consumer_id=consumer_id,
                    )
                )
                if not registered_call:
                    violations.append(
                        CallsiteViolation(
                            kind=CallsiteViolationKind.UNREGISTERED_DIRECT_CALL,
                            module_path=module_path,
                            symbol=name or short,
                            lineno=getattr(node, "lineno", 0) or 0,
                        )
                    )
            if name in _FORBIDDEN_SUBPROCESS_NAMES or short in {
                "run",
                "Popen",
                "call",
                "check_output",
                "system",
                "popen",
            }:
                # Only flag subprocess when the surrounding module is a consumer
                # and does not import the migration gateway / typed adapter.
                if consumer_id and not (uses_approved or imports_migration):
                    # Heuristic: provider-ish subprocess (python -c / agent CLIs).
                    records.append(
                        CallsiteRecord(
                            module_path=module_path,
                            kind="subprocess",
                            symbol=name or short,
                            lineno=getattr(node, "lineno", 0) or 0,
                            registered=False,
                            allowlisted=False,
                            consumer_id=consumer_id,
                        )
                    )
            if short in registered or name in registered:
                uses_approved = True
                records.append(
                    CallsiteRecord(
                        module_path=module_path,
                        kind="approved",
                        symbol=name or short,
                        lineno=getattr(node, "lineno", 0) or 0,
                        registered=True,
                        allowlisted=allowlisted,
                        consumer_id=consumer_id,
                    )
                )

    # In-scope consumers must show approved gateway/adapter usage.
    if consumer_id and not allowlisted and not uses_approved:
        violations.append(
            CallsiteViolation(
                kind=CallsiteViolationKind.MISSING_ATTRIBUTION,
                module_path=module_path,
                symbol=consumer_id,
                lineno=0,
                detail="in-scope consumer lacks approved gateway/adapter usage",
            )
        )

    return records, violations


def build_callsite_inventory(
    sources: Mapping[str, str],
    *,
    require_consumers: bool = True,
) -> CallsiteInventory:
    """Build an inventory across module path → source text mappings."""

    all_records: list[CallsiteRecord] = []
    all_violations: list[CallsiteViolation] = []
    seen_consumers: set[str] = set()

    for module_path, source in sorted(sources.items()):
        consumer = _module_consumer_id(module_path)
        if consumer:
            seen_consumers.add(consumer)
        records, violations = scan_module_callsites(
            source, module_path=module_path
        )
        all_records.extend(records)
        all_violations.extend(violations)

    if require_consumers:
        for consumer in ConsumerId:
            if consumer.value not in seen_consumers:
                all_violations.append(
                    CallsiteViolation(
                        kind=CallsiteViolationKind.MISSING_ATTRIBUTION,
                        module_path=IN_SCOPE_CONSUMER_MODULES[consumer],
                        symbol=consumer.value,
                        lineno=0,
                        detail="in-scope consumer missing from inventory sources",
                    )
                )

    coverage_complete = not all_violations
    return CallsiteInventory(
        consumers=tuple(sorted(seen_consumers)),
        callsites=tuple(all_records),
        violations=tuple(all_violations),
        coverage_complete=coverage_complete,
    )


def inventory_repository_consumers(
    repo_root: str | os.PathLike[str],
) -> CallsiteInventory:
    """Load in-scope consumer sources from a repository tree and inventory them."""

    root = Path(repo_root)
    sources: dict[str, str] = {}
    for consumer, rel in IN_SCOPE_CONSUMER_MODULES.items():
        path = root / rel
        if path.is_file():
            sources[rel] = path.read_text(encoding="utf-8")
        else:
            # Fall back to declared flat path for dual-layout trees.
            flat = (
                root
                / "ipfs_accelerate_py"
                / "agent_supervisor"
                / f"{consumer.value}.py"
            )
            if flat.is_file():
                sources[str(flat.relative_to(root))] = flat.read_text(
                    encoding="utf-8"
                )
    # Include the migration module itself (provider-free discovery allowlist).
    migration_rel = (
        "ipfs_accelerate_py/agent_supervisor/provider_usage_migration.py"
    )
    migration_path = root / migration_rel
    if migration_path.is_file():
        sources[migration_rel] = migration_path.read_text(encoding="utf-8")
    return build_callsite_inventory(sources, require_consumers=True)


def assert_coverage_complete(inventory: CallsiteInventory) -> CallsiteInventory:
    """Fail closed when the inventory reports any coverage violation."""

    if not inventory.coverage_complete or inventory.violations:
        details = "; ".join(
            f"{item.kind.value}:{item.module_path}:{item.symbol}"
            for item in inventory.violations[:8]
        )
        raise ProviderUsageMigrationError(
            f"provider callsite coverage incomplete: {details}",
            reason_codes=("coverage_incomplete",),
        )
    return inventory


def migration_status_for(consumer_id: ConsumerId | str) -> dict[str, Any]:
    """Return a compact migration status record for one consumer."""

    cid = ConsumerId(consumer_id)
    return {
        "schema": PROVIDER_USAGE_MIGRATION_SCHEMA,
        "requirement_id": COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID,
        "goal_id": COMPLETE_PROVIDER_CALLSITE_GOAL_ID,
        "consumer_id": cid.value,
        "module_path": IN_SCOPE_CONSUMER_MODULES[cid],
        "migrated": True,
        "is_completion_evidence": False,
        "is_correctness_evidence": False,
        "may_retry_side_effecting_agent_work": (
            MIGRATION_MAY_RETRY_SIDE_EFFECTING_AGENT_WORK
        ),
        "may_route_to_forbidden_endpoint": (
            MIGRATION_MAY_ROUTE_TO_FORBIDDEN_ENDPOINT
        ),
        "may_change_prompt_source_output_contracts": (
            MIGRATION_MAY_CHANGE_PROMPT_SOURCE_OUTPUT_CONTRACTS
        ),
    }


def registered_consumer_ids() -> tuple[str, ...]:
    return tuple(item.value for item in ConsumerId)


def discover_migration_schemas() -> tuple[str, ...]:
    """Provider-free schema discovery (cold-import safe)."""

    return (
        PROVIDER_USAGE_MIGRATION_SCHEMA,
        CALLSITE_INVENTORY_SCHEMA,
        MIGRATED_CALL_RECEIPT_SCHEMA,
        NON_METERABLE_RESULT_SCHEMA,
    )


# Thread-local last-call receipt storage for consumers that retain operational
# receipts without changing their public return contracts.
_last_call_lock = threading.RLock()
_last_call_results: dict[str, MigratedProviderCallResult] = {}


def retain_last_call_result(
    consumer_id: ConsumerId | str,
    result: MigratedProviderCallResult,
) -> MigratedProviderCallResult:
    """Retain the last operational call result without treating it as proof."""

    if result.is_completion_evidence or result.is_correctness_evidence:
        raise ProviderUsageMigrationError(
            "cannot retain a call result that claims proof authority",
            reason_codes=("authority_boundary",),
        )
    key = ConsumerId(consumer_id).value
    with _last_call_lock:
        _last_call_results[key] = result
    return result


def last_call_result(
    consumer_id: ConsumerId | str,
) -> Optional[MigratedProviderCallResult]:
    with _last_call_lock:
        return _last_call_results.get(ConsumerId(consumer_id).value)


def clear_last_call_results() -> None:
    with _last_call_lock:
        _last_call_results.clear()


__all__ = [
    "CALLSITE_INVENTORY_SCHEMA",
    "COMPLETE_PROVIDER_CALLSITE_GOAL_ID",
    "COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID",
    "CallsiteInventory",
    "CallsiteKind",
    "CallsiteRecord",
    "CallsiteViolation",
    "CallsiteViolationKind",
    "ConsumerCallContext",
    "ConsumerId",
    "DECLARED_FLAT_CONSUMER_STEMS",
    "DEFAULT_NON_METERABLE_ENFORCE_CEILING_REQUESTS",
    "IN_SCOPE_CONSUMER_MODULES",
    "MAX_NON_METERABLE_ENFORCE_CEILING_REQUESTS",
    "MIGRATED_CALL_RECEIPT_SCHEMA",
    "MIGRATION_AUTHORIZES_USAGE",
    "MIGRATION_IS_COMPLETION_EVIDENCE",
    "MIGRATION_IS_CORRECTNESS_EVIDENCE",
    "MIGRATION_MAY_CHANGE_PROMPT_SOURCE_OUTPUT_CONTRACTS",
    "MIGRATION_MAY_RETRY_SIDE_EFFECTING_AGENT_WORK",
    "MIGRATION_MAY_ROUTE_TO_FORBIDDEN_ENDPOINT",
    "MigratedProviderCallResult",
    "NON_METERABLE_RESULT_SCHEMA",
    "NonMeterableProviderResult",
    "PROVIDER_FREE_DISCOVERY_ALLOWLIST",
    "PROVIDER_USAGE_MIGRATION_CONTRACT_VERSION",
    "PROVIDER_USAGE_MIGRATION_SCHEMA",
    "ProviderUsageMigrationError",
    "SupervisorUsageMode",
    "admit_non_meterable",
    "assert_coverage_complete",
    "build_bridge_request",
    "build_callsite_inventory",
    "build_consumer_call_context",
    "build_request_lineage_envelope",
    "clear_last_call_results",
    "consume_usage_receipt",
    "discover_migration_schemas",
    "dispatch_migrated_provider_call",
    "extract_child_usage_metadata",
    "inventory_repository_consumers",
    "last_call_result",
    "migrated_text_provider",
    "migration_status_for",
    "mode_admits_non_meterable_under_ceiling",
    "mode_requires_envelope",
    "mode_requires_receipt",
    "new_attempt_idempotency_key",
    "non_meterable_from_child",
    "registered_consumer_ids",
    "request_leaf_envelope",
    "resolve_usage_mode",
    "retain_last_call_result",
    "scan_module_callsites",
]
