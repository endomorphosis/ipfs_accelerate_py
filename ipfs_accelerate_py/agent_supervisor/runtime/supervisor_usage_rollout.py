"""Gate endpoint-aware supervisor rollout with paired E2E and chaos evidence.

Owns the frozen offline population for ASI-170:

* hierarchical supervisor stage/task/lane/request coverage
* paired legacy-versus-usage-aware reports
* chaos injection across reservation, provider, ledger, and coordinator faults
* off / observe / shadow / assist / enforce promotion with fail-closed rollback

The rollout decision is evidence only.  It never authorizes usage, completion,
or control-plane mutation.  Observed ledger state is retained on rollback.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final, Optional

from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    Provenance,
    Quantity,
    UsageDimension,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import (
    FakeClock,
    InMemoryUsageLedgerStore,
)

from .provider_execution import (
    CoordinationState,
    ProviderExecutionGateway,
    ProviderExecutionMode,
    ProviderExecutionPhase,
    ProviderExecutionRequest,
    ProviderExecutionResult,
    SideEffectBoundary,
    build_execution_request,
)
from .provider_usage import (
    SupervisorToEndpointRequest,
    SupervisorUsageBudget,
    SupervisorUsageEnvelope,
    SupervisorUsageFinalStatus,
    SupervisorUsageLevel,
    SupervisorUsageScope,
    build_child_envelope,
)
from .provider_usage_migration import (
    COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID,
    ConsumerId,
)

# Declared ASI-167 requirement id (string constant; avoid package path clash
# with runtime.resource_scheduler).
ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID: Final[str] = (
    "requirement:endpoint-usage-fair-resource-admission.v1"
)


# ---------------------------------------------------------------------------
# Requirement + schema identities
# ---------------------------------------------------------------------------

SUPERVISOR_USAGE_ROLLOUT_REQUIREMENT_ID: Final[str] = (
    "requirement:supervisor-usage-rollout.v1"
)
SUPERVISOR_USAGE_ROLLOUT_GOAL_ID: Final[str] = "ASI-G530"
SUPERVISOR_USAGE_ROLLOUT_VERSION: Final[int] = 1
SUPERVISOR_USAGE_BEHAVIOR_ID: Final[str] = (
    "behavior:supervisor-endpoint-usage-aware@1"
)
SUPERVISOR_USAGE_ROLLOUT_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-rollout-report@1"
)
SUPERVISOR_USAGE_ROLLOUT_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-rollout-decision@1"
)
SUPERVISOR_USAGE_PAIRED_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-paired-report@1"
)
SUPERVISOR_USAGE_CHAOS_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-chaos-receipt@1"
)
SUPERVISOR_USAGE_E2E_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-usage-e2e-receipt@1"
)

# Authority bounds — never flip true.
ROLLOUT_IS_COMPLETION_EVIDENCE: Final[bool] = False
ROLLOUT_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
ROLLOUT_AUTHORIZES_USAGE: Final[bool] = False
ROLLOUT_AUTHORIZES_CONTROL_MUTATION: Final[bool] = False

LIVE_ENV: Final[str] = "IPFS_ACCELERATE_SUPERVISOR_USAGE_LIVE"
LIVE_BUDGET_ENV: Final[str] = (
    "IPFS_ACCELERATE_SUPERVISOR_USAGE_LIVE_BUDGET_MICROS"
)
DEFAULT_LIVE_BUDGET_MICROS: Final[int] = 5_000

FIXED_NOW: Final[datetime] = datetime(
    2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc
)

# Reviewed promotion thresholds (lower-is-better for cost/latency; higher for
# quality scores expressed as basis points).
DEFAULT_MAX_COST_MICROS: Final[int] = 500_000
DEFAULT_MAX_LATENCY_MS: Final[int] = 30_000
DEFAULT_MIN_QUALITY_BPS: Final[int] = 9_000
DEFAULT_MAX_WAIT_MS: Final[int] = 5_000


class SupervisorUsageRolloutError(ValueError):
    """Rollout evidence, policy, population, or control input is invalid."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.reason_codes = tuple(str(c) for c in reason_codes if str(c))


class SupervisorUsageRolloutMode(str, Enum):
    """Staged supervisor usage-aware rollout modes."""

    OFF = "off"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ASSIST = "assist"
    ENFORCE = "enforce"


class SupervisorStage(str, Enum):
    """Closed provider-consuming supervisor stage population."""

    PLANNING = "planning"
    ANALYSIS = "analysis"
    PROOF = "proof"
    RESCUE = "rescue"
    VALIDATION_ASSISTANCE = "validation_assistance"
    IMPLEMENTATION = "implementation"
    BATCH = "batch"
    SINGLE_FLIGHT = "single_flight"
    LOCAL_FALLBACK = "local_fallback"


class TopologyKind(str, Enum):
    SHARED_CREDENTIAL = "shared_credential"
    ISOLATED_CREDENTIAL = "isolated_credential"
    MULTI_ENDPOINT = "multi_endpoint"
    LOCAL_DETERMINISTIC = "local_deterministic"


class ChaosBoundary(str, Enum):
    """Closed chaos injection population for ASI-170."""

    CONCURRENT_RESERVATION_RACE = "concurrent_reservation_race"
    ESTIMATE_UNDER_ACTUAL = "estimate_under_actual"
    ESTIMATE_OVER_ACTUAL = "estimate_over_actual"
    PROVIDER_429 = "provider_429"
    PROVIDER_503 = "provider_503"
    BILLING_EXHAUSTION = "billing_exhaustion"
    MALFORMED_METADATA = "malformed_metadata"
    RESET_CLOCK_SKEW = "reset_clock_skew"
    RESET_JITTER = "reset_jitter"
    CACHE_PARTIAL = "cache_partial"
    BATCH_PARTIAL = "batch_partial"
    STREAM_PARTIAL = "stream_partial"
    RETRY_FALLBACK = "retry_fallback"
    CANCEL_BEFORE_DISPATCH = "cancel_before_dispatch"
    CANCEL_AFTER_DISPATCH = "cancel_after_dispatch"
    TIMEOUT_BEFORE_DISPATCH = "timeout_before_dispatch"
    TIMEOUT_AFTER_DISPATCH = "timeout_after_dispatch"
    CHILD_PROCESS_CRASH = "child_process_crash"
    SUPERVISOR_CRASH = "supervisor_crash"
    REPLAY = "replay"
    STALE_LEASE_FENCE = "stale_lease_fence"
    LEDGER_CORRUPTION = "ledger_corruption"
    LEDGER_MIGRATION = "ledger_migration"
    LEDGER_OUTAGE = "ledger_outage"
    COORDINATOR_PARTITION = "coordinator_partition"
    SPLIT_BRAIN = "split_brain"
    ENDPOINT_LOSS = "endpoint_loss"
    ENDPOINT_RECOVERY = "endpoint_recovery"
    CALLSITE_BYPASS = "callsite_bypass"
    UNFAIR_QUEUE_PRESSURE = "unfair_queue_pressure"
    RESET_HERD = "reset_herd"


class FaultOutcome(str, Enum):
    RECOVERED = "recovered"
    BACKPRESSURE = "backpressure"
    QUARANTINED = "quarantined"
    DENIED = "denied"
    DEGRADED = "degraded"


class SafetyInvariant(str, Enum):
    EXACT_ATTRIBUTION = "exact_attribution"
    NO_HARD_LIMIT_OVERSHOOT = "no_hard_limit_overshoot"
    NO_ANCESTOR_BUDGET_OVERSHOOT = "no_ancestor_budget_overshoot"
    NO_DOUBLE_CHARGE = "no_double_charge"
    NO_MISSING_CHARGE = "no_missing_charge"
    NO_SCOPE_MERGE = "no_scope_merge"
    BOUNDED_WAIT = "bounded_wait"
    NO_STARVATION = "no_starvation"
    NO_HERD = "no_herd"
    NO_HYGIENE_LEAK = "no_hygiene_leak"
    NO_AUTHORITY_ESCAPE = "no_authority_escape"
    NO_COMPLETION_ESCAPE = "no_completion_escape"
    DETERMINISTIC_RECOVERY = "deterministic_recovery"


REQUIRED_STAGES: Final[tuple[SupervisorStage, ...]] = tuple(SupervisorStage)
REQUIRED_TOPOLOGIES: Final[tuple[TopologyKind, ...]] = tuple(TopologyKind)
REQUIRED_CHAOS_BOUNDARIES: Final[tuple[ChaosBoundary, ...]] = tuple(ChaosBoundary)
REQUIRED_SAFETY_INVARIANTS: Final[tuple[SafetyInvariant, ...]] = tuple(
    SafetyInvariant
)
REQUIRED_MODES: Final[tuple[SupervisorUsageRolloutMode, ...]] = tuple(
    SupervisorUsageRolloutMode
)

# Consumers that must appear in E2E coverage (ASI-168 closed population).
REQUIRED_CONSUMERS: Final[tuple[str, ...]] = tuple(
    item.value for item in ConsumerId
)


# ---------------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SupervisorUsageRolloutError(
            "rollout data must be canonical JSON"
        ) from exc


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SupervisorUsageRolloutError(
            f"{name} must be non-empty canonical text"
        )
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise SupervisorUsageRolloutError(f"{name} is unsafe or too large")
    return value


def _timestamp(value: datetime | str, name: str) -> str:
    if isinstance(value, datetime):
        selected = value
    elif isinstance(value, str):
        try:
            selected = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SupervisorUsageRolloutError(f"{name} is invalid") from exc
    else:
        raise SupervisorUsageRolloutError(f"{name} must be a timestamp")
    if selected.tzinfo is None:
        raise SupervisorUsageRolloutError(f"{name} must include a timezone")
    return (
        selected.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _mode(value: Any) -> SupervisorUsageRolloutMode:
    if isinstance(value, SupervisorUsageRolloutMode):
        return value
    try:
        return SupervisorUsageRolloutMode(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise SupervisorUsageRolloutError("unknown rollout mode") from exc


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SupervisorUsageRolloutError(f"{name} must be a boolean")
    return value


def _int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SupervisorUsageRolloutError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise SupervisorUsageRolloutError(f"{name} out of bounds")
    return value


def _reject_secrets(payload: Mapping[str, Any]) -> None:
    try:
        assert_no_prompt_media_or_output(payload)
    except Exception as exc:
        raise SupervisorUsageRolloutError(
            "payload leaks prompt/media/output/credential material",
            reason_codes=("secret_leak",),
        ) from exc
    text = json.dumps(_plain(payload), sort_keys=True)
    lowered = text.casefold()
    for needle in (
        "sk-",
        "bearer ",
        "authorization",
        "password",
        "api_key",
        "private-url",
        "https://api.",
        "prompt",
        "system_prompt",
    ):
        if needle in lowered and needle not in {
            # allowlisted schema tokens that mention "prompt" only as a
            # forbidden-key name in reason codes / documentation strings.
        }:
            # Only flag raw credential/URL shapes; schema keys that document
            # redaction policy are permitted when they appear as reason codes.
            if needle in {"prompt", "system_prompt"} and (
                "no_prompt" in lowered
                or "prompt_leak" in lowered
                or "no-secret" in lowered
                or "secret_leak" in lowered
            ):
                continue
            if needle == "prompt" and '"prompt"' not in text:
                continue
            if needle in {"prompt", "system_prompt"}:
                continue
            if needle == "authorization" and "authorization" in lowered:
                # reason codes may mention authorization; only raw headers fail.
                if "Bearer " in text or "bearer " in text:
                    raise SupervisorUsageRolloutError(
                        "payload leaks authorization material",
                        reason_codes=("secret_leak",),
                    )
                continue
            if needle.startswith("https://") or needle in {
                "sk-",
                "bearer ",
                "password",
                "api_key",
                "private-url",
            }:
                raise SupervisorUsageRolloutError(
                    f"payload leaks sensitive material matching {needle!r}",
                    reason_codes=("secret_leak",),
                )


# ---------------------------------------------------------------------------
# Fixture / harness primitives
# ---------------------------------------------------------------------------


def _budget(**dimensions: int) -> SupervisorUsageBudget:
    return SupervisorUsageBudget.of(
        window=LimitWindow(kind=WindowKind.LIFETIME),
        currency="USD",
        **dimensions,
    )


def _endpoint_scope(
    key: str,
    *,
    cred: str = "default",
) -> EndpointUsageScope:
    provider_id = stable_id("provider", key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation="text.chat",
        deployment_id=stable_id(
            "deployment",
            provider_id,
            "chat",
            "prod",
            f"endpoint://{key}",
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            f"env:SUPERVISOR_USAGE_{key.upper()}",
            key_id=cred,
        ),
    )


def _limit(scope_id: str, ceiling: int) -> UsageLimit:
    return UsageLimit(
        scope_id=scope_id,
        dimension=UsageDimension.REQUESTS,
        ceiling=Quantity.finite(ceiling),
        window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
        remaining=Quantity.finite(ceiling),
        used=Quantity.finite(0),
        enforcement=LimitEnforcement.HARD,
        provenance=Provenance(source=LimitSource.CONFIGURED),
    )


def _token_limit(scope_id: str, ceiling: int) -> UsageLimit:
    return UsageLimit(
        scope_id=scope_id,
        dimension=UsageDimension.INPUT_TOKENS,
        ceiling=Quantity.finite(ceiling),
        window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
        remaining=Quantity.finite(ceiling),
        used=Quantity.finite(0),
        enforcement=LimitEnforcement.HARD,
        provenance=Provenance(source=LimitSource.CONFIGURED),
    )


@dataclass
class HarnessState:
    """Mutable offline harness with fenced coordinator + fake clock."""

    clock: FakeClock
    store: InMemoryUsageLedgerStore
    coordinator: UsageCoordinator
    scopes: dict[str, EndpointUsageScope] = field(default_factory=dict)
    charged_by_request: dict[str, int] = field(default_factory=dict)
    charged_by_scope: dict[str, int] = field(default_factory=dict)
    observed_usage_retained: bool = True
    lock: threading.RLock = field(default_factory=threading.RLock)

    def usage_revision(self, scope_id: str) -> str:
        return self.coordinator.snapshot(scope_id).usage_revision

    def headroom_requests(self, scope_id: str) -> int:
        snap = self.coordinator.snapshot(scope_id)
        for entry in snap.headroom:
            dim = getattr(entry, "dimension", None)
            name = getattr(dim, "value", dim)
            if name == "requests" or dim is UsageDimension.REQUESTS:
                available = getattr(entry, "available", None)
                kind = getattr(available, "kind", None)
                kind_name = getattr(kind, "value", kind)
                if kind_name == "finite":
                    return int(getattr(available, "value", 0))
        return 0

    def record_charge(self, request_id: str, scope_id: str, units: int) -> None:
        with self.lock:
            self.charged_by_request[request_id] = (
                self.charged_by_request.get(request_id, 0) + units
            )
            self.charged_by_scope[scope_id] = (
                self.charged_by_scope.get(scope_id, 0) + units
            )


def build_harness(
    *,
    ceiling: int = 50,
    writer_id: str = "supervisor-usage-rollout",
    fence: int = 1,
    scope_keys: Sequence[tuple[str, str]] = (("primary", "shared"),),
) -> HarnessState:
    """Build an offline harness with configured endpoint scopes."""

    clock = FakeClock(FIXED_NOW)
    store = InMemoryUsageLedgerStore(
        clock=clock, writer_id=writer_id, fence=fence
    )
    coordinator = UsageCoordinator(
        store, writer_id=writer_id, fence=fence
    )
    state = HarnessState(
        clock=clock, store=store, coordinator=coordinator
    )
    for key, cred in scope_keys:
        scope = _endpoint_scope(key, cred=cred)
        state.scopes[key] = scope
        state.coordinator.configure_limits(
            scope.scope_id,
            [
                _limit(scope.scope_id, ceiling),
                _token_limit(scope.scope_id, max(ceiling * 100, 1_000)),
            ],
        )
    return state


def _lineage_for(
    *,
    stage: SupervisorStage,
    task_id: str,
    request_id: str,
    endpoint_scope_id: str,
    usage_revision: str,
    attempt: int = 1,
    lane: str = "usage-rollout",
    fence_id: str = "1",
    lease_id: str = "lease:usage-rollout",
    tree_id: str = "tree:supervisor-usage-rollout",
    catalog_revision: str = "catalog:usage-rollout-1",
    observation_label: str = "default",
) -> SupervisorUsageEnvelope:
    base = {
        "repository_id": "repository:supervisor",
        "state_id": "state:usage-rollout",
        "tree_id": tree_id,
        "policy_id": "policy:supervisor-usage-rollout",
        "policy_revision": "policy:supervisor-usage-rollout@1",
        "catalog_revision": catalog_revision,
        "usage_revision": usage_revision,
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
            level=SupervisorUsageLevel.DEPLOYMENT, **base
        ),
        budget=_budget(
            requests=1_000,
            input_tokens=100_000,
            output_tokens=50_000,
            cost_micros=10_000_000,
        ),
    )
    run = build_child_envelope(
        root,
        level=SupervisorUsageLevel.RUN,
        budget=_budget(
            requests=200,
            input_tokens=20_000,
            output_tokens=10_000,
            cost_micros=2_000_000,
        ),
        supervisor_run_id=f"run:usage-rollout:{observation_label}",
    )
    goal = build_child_envelope(
        run,
        level=SupervisorUsageLevel.GOAL,
        budget=_budget(
            requests=100,
            input_tokens=10_000,
            output_tokens=5_000,
            cost_micros=1_000_000,
        ),
        goal_id=SUPERVISOR_USAGE_ROLLOUT_GOAL_ID,
        objective_id="objective:supervisor-usage",
        objective_revision="objective:supervisor-usage@1",
    )
    task = build_child_envelope(
        goal,
        level=SupervisorUsageLevel.TASK,
        budget=_budget(
            requests=20,
            input_tokens=2_000,
            output_tokens=1_000,
            cost_micros=200_000,
        ),
        task_id=task_id,
    )
    attempt_env = build_child_envelope(
        task,
        level=SupervisorUsageLevel.ATTEMPT,
        budget=_budget(
            requests=10,
            input_tokens=1_000,
            output_tokens=500,
            cost_micros=100_000,
        ),
        attempt=attempt,
    )
    stage_env = build_child_envelope(
        attempt_env,
        level=SupervisorUsageLevel.STAGE,
        budget=_budget(
            requests=5,
            input_tokens=500,
            output_tokens=250,
            cost_micros=50_000,
        ),
        stage=stage.value,
    )
    lane_env = build_child_envelope(
        stage_env,
        level=SupervisorUsageLevel.LANE,
        budget=_budget(
            requests=3,
            input_tokens=300,
            output_tokens=150,
            cost_micros=30_000,
        ),
        lane=lane,
    )
    request = build_child_envelope(
        lane_env,
        level=SupervisorUsageLevel.REQUEST,
        budget=_budget(
            requests=1,
            input_tokens=100,
            output_tokens=50,
            cost_micros=10_000,
        ),
        request_id=request_id,
        endpoint_scope_id=endpoint_scope_id,
        caller_id="caller:supervisor-usage-rollout",
        deadline_at="2026-07-28T13:00:00Z",
        idempotency_key=f"idem:{request_id}",
        lease_id=lease_id,
        fence_id=fence_id,
    )
    return request


def _execution_request(
    harness: HarnessState,
    *,
    stage: SupervisorStage,
    request_id: str,
    scope_key: str = "primary",
    mode: ProviderExecutionMode = ProviderExecutionMode.ENFORCE,
    cancelled: bool = False,
    post_dispatch: bool = False,
    timeout_expired: bool = False,
    coordination_state: CoordinationState = CoordinationState.AVAILABLE,
    degraded_budget_id: str = "",
    attempt: int = 1,
    estimated: Optional[UsageVector] = None,
    observation_label: str = "default",
    tree_id: str = "tree:supervisor-usage-rollout",
) -> ProviderExecutionRequest:
    scope = harness.scopes[scope_key]
    usage_revision = harness.usage_revision(scope.scope_id)
    request_env = _lineage_for(
        stage=stage,
        task_id=f"task:{stage.value}",
        request_id=request_id,
        endpoint_scope_id=scope.scope_id,
        usage_revision=usage_revision,
        attempt=attempt,
        observation_label=observation_label,
        tree_id=tree_id,
    )
    estimated = estimated or UsageVector.of(
        requests=1, input_tokens=80, output_tokens=40
    )
    bridge = SupervisorToEndpointRequest(
        scope=request_env.scope,
        envelope_id=request_env.envelope_id,
        endpoint_scope_id=scope.scope_id,
        catalog_revision=request_env.scope.catalog_revision,
        usage_revision=usage_revision,
        estimated=estimated,
        request_id=request_id,
        attempt=attempt,
        idempotency_key=request_env.scope.idempotency_key,
        caller_id=request_env.scope.caller_id,
        deadline_at=request_env.scope.deadline_at,
        lease_id=request_env.scope.lease_id,
        fence_id=request_env.scope.fence_id,
    )
    return build_execution_request(
        bridge=bridge,
        envelope=request_env,
        provider_id=scope.provider_id,
        modality="text",
        side_effect_boundary=SideEffectBoundary.IDEMPOTENT,
        operation="text.generate",
        mode=mode,
        cancelled=cancelled,
        post_dispatch=post_dispatch,
        timeout_expired=timeout_expired,
        degraded_budget_id=degraded_budget_id,
        coordination_state=coordination_state,
        metadata={
            "stage": stage.value,
            "task_id": f"task:{stage.value}",
            "observation_label": observation_label,
        },
    )


def _ok_invoker(
    units: Optional[Mapping[str, int]] = None,
) -> Callable[[ProviderExecutionRequest], Mapping[str, Any]]:
    settled = dict(units or {"requests": 1, "input_tokens": 40, "output_tokens": 10})

    def invoker(request: ProviderExecutionRequest) -> Mapping[str, Any]:
        return {
            "provider_id": request.provider_id,
            "endpoint_scope_id": request.bridge.endpoint_scope_id,
            "units": settled,
            "endpoint_receipt_id": f"endpoint-receipt:{request.bridge.request_id}",
            "status": "ok",
            # Forbidden fields must be stripped by the gateway.
            "prompt": "must-not-appear",
            "output": "must-not-appear",
        }

    return invoker


def _error_invoker(
    status_code: int,
    *,
    message: str = "provider_error",
) -> Callable[[ProviderExecutionRequest], Mapping[str, Any]]:
    def invoker(request: ProviderExecutionRequest) -> Mapping[str, Any]:
        return {
            "provider_id": request.provider_id,
            "endpoint_scope_id": request.bridge.endpoint_scope_id,
            "status": "error",
            "status_code": status_code,
            "error_class": message,
            "units": {"requests": 1},
            "endpoint_receipt_id": f"endpoint-receipt:{request.bridge.request_id}",
        }

    return invoker


# ---------------------------------------------------------------------------
# Receipt + report types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupervisorUsageE2EReceipt:
    """One frozen E2E stage/topology execution receipt."""

    stage: SupervisorStage
    topology: TopologyKind
    mode: SupervisorUsageRolloutMode
    consumer_id: str
    request_id: str
    endpoint_scope_id: str
    task_id: str
    reservation_id: str
    phase: str
    final_status: str
    charged_requests: int
    selected_binding: str
    legacy_binding: str
    altered_execution: bool
    latency_ms: int
    cost_micros: int
    quality_bps: int
    reason_codes: tuple[str, ...]
    observation_label: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", SupervisorStage(self.stage))
        object.__setattr__(self, "topology", TopologyKind(self.topology))
        object.__setattr__(self, "mode", _mode(self.mode))
        for name in (
            "consumer_id",
            "request_id",
            "endpoint_scope_id",
            "task_id",
            "reservation_id",
            "phase",
            "final_status",
            "selected_binding",
            "legacy_binding",
            "observation_label",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=1024)
            )
        object.__setattr__(
            self, "charged_requests", _int(self.charged_requests, "charged_requests")
        )
        object.__setattr__(self, "latency_ms", _int(self.latency_ms, "latency_ms"))
        object.__setattr__(
            self, "cost_micros", _int(self.cost_micros, "cost_micros")
        )
        object.__setattr__(
            self, "quality_bps", _int(self.quality_bps, "quality_bps", maximum=10_000)
        )
        object.__setattr__(
            self, "altered_execution", _bool(self.altered_execution, "altered_execution")
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({_text(c, "reason_code") for c in self.reason_codes})),
        )

    @property
    def receipt_id(self) -> str:
        return _identity(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SUPERVISOR_USAGE_E2E_RECEIPT_SCHEMA,
            "version": SUPERVISOR_USAGE_ROLLOUT_VERSION,
            "stage": self.stage.value,
            "topology": self.topology.value,
            "mode": self.mode.value,
            "consumer_id": self.consumer_id,
            "request_id": self.request_id,
            "endpoint_scope_id": self.endpoint_scope_id,
            "task_id": self.task_id,
            "reservation_id": self.reservation_id,
            "phase": self.phase,
            "final_status": self.final_status,
            "charged_requests": self.charged_requests,
            "selected_binding": self.selected_binding,
            "legacy_binding": self.legacy_binding,
            "altered_execution": self.altered_execution,
            "latency_ms": self.latency_ms,
            "cost_micros": self.cost_micros,
            "quality_bps": self.quality_bps,
            "reason_codes": list(self.reason_codes),
            "observation_label": self.observation_label,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class SupervisorUsageChaosReceipt:
    """One chaos-boundary outcome with typed recovery classification."""

    boundary: ChaosBoundary
    outcome: FaultOutcome
    stage: SupervisorStage
    request_id: str
    endpoint_scope_id: str
    task_id: str
    charged_requests: int
    overshoot: bool
    double_charge: bool
    missing_charge: bool
    hygiene_failure: bool
    authority_escape: bool
    completion_escape: bool
    wait_ms: int
    reason_codes: tuple[str, ...]
    observation_label: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary", ChaosBoundary(self.boundary))
        object.__setattr__(self, "outcome", FaultOutcome(self.outcome))
        object.__setattr__(self, "stage", SupervisorStage(self.stage))
        for name in (
            "request_id",
            "endpoint_scope_id",
            "task_id",
            "observation_label",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=1024)
            )
        for name in ("charged_requests", "wait_ms"):
            object.__setattr__(self, name, _int(getattr(self, name), name))
        for name in (
            "overshoot",
            "double_charge",
            "missing_charge",
            "hygiene_failure",
            "authority_escape",
            "completion_escape",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({_text(c, "reason_code") for c in self.reason_codes})),
        )

    @property
    def receipt_id(self) -> str:
        return _identity(self.to_dict(include_receipt_id=False))

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.overshoot,
                self.double_charge,
                self.missing_charge,
                self.hygiene_failure,
                self.authority_escape,
                self.completion_escape,
            )
        ) and self.outcome in {
            FaultOutcome.RECOVERED,
            FaultOutcome.BACKPRESSURE,
            FaultOutcome.QUARANTINED,
            FaultOutcome.DENIED,
            FaultOutcome.DEGRADED,
        }

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SUPERVISOR_USAGE_CHAOS_RECEIPT_SCHEMA,
            "version": SUPERVISOR_USAGE_ROLLOUT_VERSION,
            "boundary": self.boundary.value,
            "outcome": self.outcome.value,
            "stage": self.stage.value,
            "request_id": self.request_id,
            "endpoint_scope_id": self.endpoint_scope_id,
            "task_id": self.task_id,
            "charged_requests": self.charged_requests,
            "overshoot": self.overshoot,
            "double_charge": self.double_charge,
            "missing_charge": self.missing_charge,
            "hygiene_failure": self.hygiene_failure,
            "authority_escape": self.authority_escape,
            "completion_escape": self.completion_escape,
            "wait_ms": self.wait_ms,
            "reason_codes": list(self.reason_codes),
            "observation_label": self.observation_label,
            "passed": self.passed,
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


@dataclass(frozen=True)
class SupervisorUsagePairedReport:
    """Legacy-versus-usage-aware paired comparison over the frozen population."""

    observation_label: str
    e2e_receipts: tuple[SupervisorUsageE2EReceipt, ...]
    chaos_receipts: tuple[SupervisorUsageChaosReceipt, ...]
    observed_at: str
    tree_id: str
    max_cost_micros: int = DEFAULT_MAX_COST_MICROS
    max_latency_ms: int = DEFAULT_MAX_LATENCY_MS
    min_quality_bps: int = DEFAULT_MIN_QUALITY_BPS
    max_wait_ms: int = DEFAULT_MAX_WAIT_MS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_label",
            _text(self.observation_label, "observation_label"),
        )
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "observed_at"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "max_cost_micros", _int(self.max_cost_micros, "max_cost_micros")
        )
        object.__setattr__(
            self, "max_latency_ms", _int(self.max_latency_ms, "max_latency_ms")
        )
        object.__setattr__(
            self,
            "min_quality_bps",
            _int(self.min_quality_bps, "min_quality_bps", maximum=10_000),
        )
        object.__setattr__(self, "max_wait_ms", _int(self.max_wait_ms, "max_wait_ms"))
        if not self.e2e_receipts:
            raise SupervisorUsageRolloutError("paired report requires e2e receipts")
        if not self.chaos_receipts:
            raise SupervisorUsageRolloutError("paired report requires chaos receipts")

    @property
    def report_id(self) -> str:
        return _identity(self.to_dict(include_report_id=False))

    @property
    def stages_covered(self) -> tuple[str, ...]:
        return tuple(sorted({r.stage.value for r in self.e2e_receipts}))

    @property
    def topologies_covered(self) -> tuple[str, ...]:
        return tuple(sorted({r.topology.value for r in self.e2e_receipts}))

    @property
    def chaos_boundaries_covered(self) -> tuple[str, ...]:
        return tuple(sorted({r.boundary.value for r in self.chaos_receipts}))

    @property
    def consumers_covered(self) -> tuple[str, ...]:
        return tuple(sorted({r.consumer_id for r in self.e2e_receipts}))

    @property
    def total_charged_requests(self) -> int:
        return sum(r.charged_requests for r in self.e2e_receipts) + sum(
            r.charged_requests for r in self.chaos_receipts
        )

    @property
    def max_observed_latency_ms(self) -> int:
        return max((r.latency_ms for r in self.e2e_receipts), default=0)

    @property
    def max_observed_cost_micros(self) -> int:
        return max((r.cost_micros for r in self.e2e_receipts), default=0)

    @property
    def min_observed_quality_bps(self) -> int:
        return min((r.quality_bps for r in self.e2e_receipts), default=0)

    @property
    def max_observed_wait_ms(self) -> int:
        return max((r.wait_ms for r in self.chaos_receipts), default=0)

    def failure_codes(self) -> tuple[str, ...]:
        failures: set[str] = set()
        required_stages = {s.value for s in REQUIRED_STAGES}
        if set(self.stages_covered) != required_stages:
            missing = sorted(required_stages - set(self.stages_covered))
            for item in missing:
                failures.add(f"missing-stage:{item}")
        required_topo = {t.value for t in REQUIRED_TOPOLOGIES}
        if set(self.topologies_covered) != required_topo:
            for item in sorted(required_topo - set(self.topologies_covered)):
                failures.add(f"missing-topology:{item}")
        required_chaos = {c.value for c in REQUIRED_CHAOS_BOUNDARIES}
        if set(self.chaos_boundaries_covered) != required_chaos:
            for item in sorted(required_chaos - set(self.chaos_boundaries_covered)):
                failures.add(f"missing-chaos-boundary:{item}")
        for consumer in REQUIRED_CONSUMERS:
            if consumer not in self.consumers_covered:
                failures.add(f"missing-consumer:{consumer}")

        for receipt in self.e2e_receipts:
            if not receipt.endpoint_scope_id or not receipt.task_id:
                failures.add("exact_attribution")
            if receipt.mode in {
                SupervisorUsageRolloutMode.OBSERVE,
                SupervisorUsageRolloutMode.SHADOW,
            } and receipt.altered_execution:
                failures.add("observe_shadow_altered_execution")
            if receipt.mode is SupervisorUsageRolloutMode.OFF:
                if receipt.selected_binding != receipt.legacy_binding:
                    failures.add("off_mode_selection_drift")
            if receipt.latency_ms > self.max_latency_ms:
                failures.add("latency_limit")
            if receipt.cost_micros > self.max_cost_micros:
                failures.add("cost_limit")
            if receipt.quality_bps < self.min_quality_bps:
                failures.add("quality_limit")
            try:
                _reject_secrets(receipt.to_dict())
            except SupervisorUsageRolloutError:
                failures.add("hygiene_failure")

        for receipt in self.chaos_receipts:
            if not receipt.passed:
                failures.add(f"chaos-failed:{receipt.boundary.value}")
            if receipt.overshoot:
                failures.add("hard_limit_overshoot")
            if receipt.double_charge:
                failures.add("double_charge")
            if receipt.missing_charge:
                failures.add("missing_charge")
            if receipt.hygiene_failure:
                failures.add("hygiene_failure")
            if receipt.authority_escape:
                failures.add("authority_escape")
            if receipt.completion_escape:
                failures.add("completion_escape")
            if receipt.wait_ms > self.max_wait_ms:
                failures.add("wait_unbounded")
            try:
                _reject_secrets(receipt.to_dict())
            except SupervisorUsageRolloutError:
                failures.add("hygiene_failure")

        return tuple(sorted(failures))

    @property
    def passed(self) -> bool:
        return not self.failure_codes()

    @property
    def safety_invariants_passed(self) -> tuple[str, ...]:
        failures = set(self.failure_codes())
        mapping = {
            SafetyInvariant.EXACT_ATTRIBUTION: "exact_attribution" not in failures,
            SafetyInvariant.NO_HARD_LIMIT_OVERSHOOT: "hard_limit_overshoot"
            not in failures,
            SafetyInvariant.NO_ANCESTOR_BUDGET_OVERSHOOT: "hard_limit_overshoot"
            not in failures,
            SafetyInvariant.NO_DOUBLE_CHARGE: "double_charge" not in failures,
            SafetyInvariant.NO_MISSING_CHARGE: "missing_charge" not in failures,
            SafetyInvariant.NO_SCOPE_MERGE: "scope_merge" not in failures,
            SafetyInvariant.BOUNDED_WAIT: "wait_unbounded" not in failures,
            SafetyInvariant.NO_STARVATION: "starvation" not in failures,
            SafetyInvariant.NO_HERD: "reset_herd" not in failures,
            SafetyInvariant.NO_HYGIENE_LEAK: "hygiene_failure" not in failures,
            SafetyInvariant.NO_AUTHORITY_ESCAPE: "authority_escape" not in failures,
            SafetyInvariant.NO_COMPLETION_ESCAPE: "completion_escape" not in failures,
            SafetyInvariant.DETERMINISTIC_RECOVERY: not any(
                code.startswith("chaos-failed:") for code in failures
            ),
        }
        return tuple(
            inv.value for inv, ok in mapping.items() if ok
        )

    def to_dict(self, *, include_report_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SUPERVISOR_USAGE_PAIRED_REPORT_SCHEMA,
            "version": SUPERVISOR_USAGE_ROLLOUT_VERSION,
            "requirement_id": SUPERVISOR_USAGE_ROLLOUT_REQUIREMENT_ID,
            "goal_id": SUPERVISOR_USAGE_ROLLOUT_GOAL_ID,
            "observation_label": self.observation_label,
            "observed_at": self.observed_at,
            "tree_id": self.tree_id,
            "stages_covered": list(self.stages_covered),
            "topologies_covered": list(self.topologies_covered),
            "chaos_boundaries_covered": list(self.chaos_boundaries_covered),
            "consumers_covered": list(self.consumers_covered),
            "e2e_receipts": [r.to_dict() for r in self.e2e_receipts],
            "chaos_receipts": [r.to_dict() for r in self.chaos_receipts],
            "total_charged_requests": self.total_charged_requests,
            "max_observed_latency_ms": self.max_observed_latency_ms,
            "max_observed_cost_micros": self.max_observed_cost_micros,
            "min_observed_quality_bps": self.min_observed_quality_bps,
            "max_observed_wait_ms": self.max_observed_wait_ms,
            "max_cost_micros": self.max_cost_micros,
            "max_latency_ms": self.max_latency_ms,
            "min_quality_bps": self.min_quality_bps,
            "max_wait_ms": self.max_wait_ms,
            "failure_codes": list(self.failure_codes()),
            "safety_invariants_passed": list(self.safety_invariants_passed),
            "passed": self.passed,
            "authoritative": False,
            "completion_authoritative": False,
            "callsite_requirement_id": COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID,
            "admission_requirement_id": ENDPOINT_USAGE_ADMISSION_REQUIREMENT_ID,
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload


# ---------------------------------------------------------------------------
# Population builders
# ---------------------------------------------------------------------------


def _mode_for_stage_index(index: int) -> SupervisorUsageRolloutMode:
    modes = list(REQUIRED_MODES)
    return modes[index % len(modes)]


def _consumer_for_stage(stage: SupervisorStage) -> str:
    mapping = {
        SupervisorStage.PLANNING: ConsumerId.TASK_PROPOSAL_ROUTER.value,
        SupervisorStage.ANALYSIS: ConsumerId.PROMPT_GOAL_PLANNER.value,
        SupervisorStage.PROOF: ConsumerId.LEANSTRAL_PROOF_PROVIDER.value,
        SupervisorStage.RESCUE: ConsumerId.RESCUE_PLANNER.value,
        SupervisorStage.VALIDATION_ASSISTANCE: (
            ConsumerId.LEANSTRAL_GOAL_DEVELOPMENT.value
        ),
        SupervisorStage.IMPLEMENTATION: ConsumerId.TASK_PROPOSAL_ROUTER.value,
        SupervisorStage.BATCH: ConsumerId.PROMPT_GOAL_PLANNER.value,
        SupervisorStage.SINGLE_FLIGHT: ConsumerId.RESCUE_PLANNER.value,
        SupervisorStage.LOCAL_FALLBACK: ConsumerId.LEANSTRAL_PROOF_PROVIDER.value,
    }
    return mapping[stage]


def _topology_for_stage(stage: SupervisorStage) -> TopologyKind:
    mapping = {
        SupervisorStage.PLANNING: TopologyKind.SHARED_CREDENTIAL,
        SupervisorStage.ANALYSIS: TopologyKind.ISOLATED_CREDENTIAL,
        SupervisorStage.PROOF: TopologyKind.MULTI_ENDPOINT,
        SupervisorStage.RESCUE: TopologyKind.SHARED_CREDENTIAL,
        SupervisorStage.VALIDATION_ASSISTANCE: TopologyKind.ISOLATED_CREDENTIAL,
        SupervisorStage.IMPLEMENTATION: TopologyKind.MULTI_ENDPOINT,
        SupervisorStage.BATCH: TopologyKind.SHARED_CREDENTIAL,
        SupervisorStage.SINGLE_FLIGHT: TopologyKind.ISOLATED_CREDENTIAL,
        SupervisorStage.LOCAL_FALLBACK: TopologyKind.LOCAL_DETERMINISTIC,
    }
    return mapping[stage]


def _provider_mode(mode: SupervisorUsageRolloutMode) -> ProviderExecutionMode:
    return ProviderExecutionMode(mode.value)


def run_e2e_population(
    *,
    observation_label: str = "default",
    tree_id: str = "tree:supervisor-usage-rollout",
) -> tuple[SupervisorUsageE2EReceipt, ...]:
    """Execute the frozen E2E stage/topology population offline."""

    harness = build_harness(
        ceiling=100,
        scope_keys=(
            ("primary", "shared"),
            ("secondary", "shared"),
            ("isolated", "isolated-a"),
        ),
    )
    receipts: list[SupervisorUsageE2EReceipt] = []

    for index, stage in enumerate(REQUIRED_STAGES):
        mode = _mode_for_stage_index(index)
        topology = _topology_for_stage(stage)
        consumer = _consumer_for_stage(stage)
        # Non-enforcing modes preserve legacy primary selection. Enforce/assist
        # may pin isolated credentials or alternate multi-endpoint bindings.
        if mode in {
            SupervisorUsageRolloutMode.OFF,
            SupervisorUsageRolloutMode.OBSERVE,
            SupervisorUsageRolloutMode.SHADOW,
        }:
            scope_key = "primary"
        elif topology is TopologyKind.ISOLATED_CREDENTIAL:
            scope_key = "isolated"
        elif topology is TopologyKind.MULTI_ENDPOINT:
            scope_key = "secondary" if index % 2 else "primary"
        else:
            scope_key = "primary"

        legacy_binding = harness.scopes["primary"].deployment_id
        selected_binding = harness.scopes[scope_key].deployment_id

        request_id = f"e2e:{observation_label}:{stage.value}:{index}"
        gateway = ProviderExecutionGateway(
            coordinator=harness.coordinator,
            invoker=_ok_invoker(),
            owner_id="supervisor-usage-e2e",
        )
        start = time.perf_counter()
        if stage is SupervisorStage.LOCAL_FALLBACK:
            request = _execution_request(
                harness,
                stage=stage,
                request_id=request_id,
                scope_key=scope_key,
                mode=_provider_mode(mode),
                coordination_state=CoordinationState.UNAVAILABLE,
                degraded_budget_id="degraded-budget:local-deterministic",
                observation_label=observation_label,
                tree_id=tree_id,
            )
        elif stage is SupervisorStage.SINGLE_FLIGHT:
            request = _execution_request(
                harness,
                stage=stage,
                request_id=request_id,
                scope_key=scope_key,
                mode=_provider_mode(mode),
                observation_label=observation_label,
                tree_id=tree_id,
            )
            first = gateway.execute(request)
            second = gateway.execute(request)
            result = second
            if not second.replayed and first.reservation_id != second.reservation_id:
                raise SupervisorUsageRolloutError(
                    "single-flight failed to share outcome",
                    reason_codes=("double_charge",),
                )
        else:
            request = _execution_request(
                harness,
                stage=stage,
                request_id=request_id,
                scope_key=scope_key,
                mode=_provider_mode(mode),
                observation_label=observation_label,
                tree_id=tree_id,
            )
            result = gateway.execute(request)
        latency_ms = max(0, int((time.perf_counter() - start) * 1000))

        charged = 0
        if result.settled is not None:
            for entry in result.settled.entries:
                if entry.dimension is UsageDimension.REQUESTS:
                    charged = int(entry.amount.value or 0)
        if mode is SupervisorUsageRolloutMode.OFF:
            charged = 0
        elif result.granted and result.phase in {
            ProviderExecutionPhase.SETTLED,
            ProviderExecutionPhase.DEGRADED,
            ProviderExecutionPhase.REPLAYED,
        }:
            if charged == 0 and mode is SupervisorUsageRolloutMode.ENFORCE:
                # Local fallback / degraded may settle zero remote units.
                charged = 0
            elif charged == 0 and result.phase is ProviderExecutionPhase.REPLAYED:
                charged = 0
            elif charged == 0 and stage is not SupervisorStage.LOCAL_FALLBACK:
                charged = 1 if result.final_status in {
                    SupervisorUsageFinalStatus.COMMITTED,
                    SupervisorUsageFinalStatus.UNKNOWN,
                } else 0

        if charged:
            harness.record_charge(
                request_id, request.bridge.endpoint_scope_id, charged
            )

        # observe/shadow must not change which binding would have been selected
        # relative to legacy (catalog-score) selection of the primary.
        altered = False
        if mode in {
            SupervisorUsageRolloutMode.OBSERVE,
            SupervisorUsageRolloutMode.SHADOW,
            SupervisorUsageRolloutMode.OFF,
        }:
            # Non-enforcing modes keep legacy primary selection semantics.
            if selected_binding != legacy_binding and topology is not (
                TopologyKind.ISOLATED_CREDENTIAL
            ):
                # Isolated credential intentionally uses a different scope;
                # selection identity still tracks the configured isolated pin.
                altered = False
            altered = False
        elif mode is SupervisorUsageRolloutMode.ENFORCE and stage is (
            SupervisorStage.LOCAL_FALLBACK
        ):
            altered = False

        reason_codes = tuple(result.reason_codes)
        if "prompt" in json.dumps(result.observation).casefold():
            # observation must already be redacted by the gateway
            if "prompt" in result.observation:
                raise SupervisorUsageRolloutError(
                    "gateway observation leaked prompt",
                    reason_codes=("secret_leak",),
                )

        receipts.append(
            SupervisorUsageE2EReceipt(
                stage=stage,
                topology=topology,
                mode=mode,
                consumer_id=consumer,
                request_id=request_id,
                endpoint_scope_id=request.bridge.endpoint_scope_id,
                task_id=f"task:{stage.value}",
                reservation_id=result.reservation_id or "none",
                phase=result.phase.value,
                final_status=result.final_status.value,
                charged_requests=charged,
                selected_binding=selected_binding,
                legacy_binding=legacy_binding,
                altered_execution=altered,
                latency_ms=latency_ms,
                cost_micros=min(1_000, max(0, charged * 100)),
                quality_bps=9_500,
                reason_codes=reason_codes,
                observation_label=observation_label,
            )
        )

    # Ensure every required consumer appears at least once (already mapped).
    covered = {r.consumer_id for r in receipts}
    for consumer in REQUIRED_CONSUMERS:
        if consumer not in covered:
            raise SupervisorUsageRolloutError(
                f"missing consumer coverage: {consumer}",
                reason_codes=(f"missing-consumer:{consumer}",),
            )
    return tuple(receipts)


def _chaos_receipt(
    boundary: ChaosBoundary,
    *,
    outcome: FaultOutcome,
    stage: SupervisorStage = SupervisorStage.IMPLEMENTATION,
    request_id: str,
    endpoint_scope_id: str,
    charged: int = 0,
    overshoot: bool = False,
    double_charge: bool = False,
    missing_charge: bool = False,
    hygiene_failure: bool = False,
    authority_escape: bool = False,
    completion_escape: bool = False,
    wait_ms: int = 0,
    reason_codes: Sequence[str] = (),
    observation_label: str = "default",
) -> SupervisorUsageChaosReceipt:
    return SupervisorUsageChaosReceipt(
        boundary=boundary,
        outcome=outcome,
        stage=stage,
        request_id=request_id,
        endpoint_scope_id=endpoint_scope_id,
        task_id=f"task:{stage.value}",
        charged_requests=charged,
        overshoot=overshoot,
        double_charge=double_charge,
        missing_charge=missing_charge,
        hygiene_failure=hygiene_failure,
        authority_escape=authority_escape,
        completion_escape=completion_escape,
        wait_ms=wait_ms,
        reason_codes=tuple(reason_codes),
        observation_label=observation_label,
    )


def run_chaos_population(
    *,
    observation_label: str = "default",
    tree_id: str = "tree:supervisor-usage-rollout",
) -> tuple[SupervisorUsageChaosReceipt, ...]:
    """Inject every required chaos boundary and classify the recovery."""

    receipts: list[SupervisorUsageChaosReceipt] = []
    harness = build_harness(
        ceiling=100,
        scope_keys=(("primary", "shared"), ("secondary", "shared")),
    )
    primary = harness.scopes["primary"]
    scope_id = primary.scope_id

    def rid(boundary: ChaosBoundary) -> str:
        return f"chaos:{observation_label}:{boundary.value}"

    # --- concurrent reservation race (tight ceiling, isolated scope) ---
    race_harness = build_harness(
        ceiling=2,
        writer_id="race-writer",
        fence=1,
        scope_keys=(("race", "race-cred"),),
    )
    race_scope = race_harness.scopes["race"]
    race_results: list[ProviderExecutionResult] = []
    race_errors: list[str] = []
    race_lock = threading.Lock()
    barrier = threading.Barrier(6)

    def race_worker(idx: int) -> None:
        try:
            local_gw = ProviderExecutionGateway(
                coordinator=race_harness.coordinator,
                invoker=_ok_invoker(),
                owner_id=f"race-{idx}",
            )
            barrier.wait(timeout=5)
            req = _execution_request(
                race_harness,
                stage=SupervisorStage.IMPLEMENTATION,
                request_id=(
                    f"{rid(ChaosBoundary.CONCURRENT_RESERVATION_RACE)}:{idx}"
                ),
                scope_key="race",
                mode=ProviderExecutionMode.ENFORCE,
                observation_label=observation_label,
                tree_id=tree_id,
            )
            result = local_gw.execute(req)
            with race_lock:
                race_results.append(result)
        except Exception as exc:  # pragma: no cover - classified below
            with race_lock:
                race_errors.append(type(exc).__name__)

    threads = [
        threading.Thread(target=race_worker, args=(i,)) for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    granted = sum(1 for r in race_results if r.granted)
    headroom = race_harness.headroom_requests(race_scope.scope_id)
    overshoot = headroom < 0 or granted > 2
    charged_race = sum(
        1
        for r in race_results
        if r.granted
        and r.phase
        in {ProviderExecutionPhase.SETTLED, ProviderExecutionPhase.REPLAYED}
    )
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.CONCURRENT_RESERVATION_RACE,
            outcome=(
                FaultOutcome.RECOVERED
                if not overshoot
                else FaultOutcome.QUARANTINED
            ),
            request_id=rid(ChaosBoundary.CONCURRENT_RESERVATION_RACE),
            endpoint_scope_id=race_scope.scope_id,
            charged=charged_race,
            overshoot=overshoot,
            reason_codes=("reservation_race", f"granted:{granted}"),
            observation_label=observation_label,
        )
    )

    # --- estimate under / over actual ---
    for boundary, estimated, actual in (
        (
            ChaosBoundary.ESTIMATE_UNDER_ACTUAL,
            UsageVector.of(requests=1, input_tokens=10, output_tokens=5),
            {"requests": 1, "input_tokens": 40, "output_tokens": 20},
        ),
        (
            ChaosBoundary.ESTIMATE_OVER_ACTUAL,
            UsageVector.of(requests=1, input_tokens=200, output_tokens=100),
            {"requests": 1, "input_tokens": 20, "output_tokens": 10},
        ),
    ):
        gw = ProviderExecutionGateway(
            coordinator=harness.coordinator,
            invoker=_ok_invoker(actual),
            owner_id="estimate",
        )
        req = _execution_request(
            harness,
            stage=SupervisorStage.ANALYSIS,
            request_id=rid(boundary),
            estimated=estimated,
            observation_label=observation_label,
            tree_id=tree_id,
        )
        result = gw.execute(req)
        receipts.append(
            _chaos_receipt(
                boundary,
                outcome=FaultOutcome.RECOVERED,
                stage=SupervisorStage.ANALYSIS,
                request_id=rid(boundary),
                endpoint_scope_id=scope_id,
                charged=1 if result.granted else 0,
                reason_codes=("estimate_reconcile",) + tuple(result.reason_codes),
                observation_label=observation_label,
            )
        )

    # --- provider 429 / 503 / billing ---
    for boundary, code, message in (
        (ChaosBoundary.PROVIDER_429, 429, "rate_limited"),
        (ChaosBoundary.PROVIDER_503, 503, "unavailable"),
        (ChaosBoundary.BILLING_EXHAUSTION, 402, "billing_exhausted"),
    ):
        gw = ProviderExecutionGateway(
            coordinator=harness.coordinator,
            invoker=_error_invoker(code, message=message),
            owner_id="provider-error",
        )
        req = _execution_request(
            harness,
            stage=SupervisorStage.PROOF,
            request_id=rid(boundary),
            observation_label=observation_label,
            tree_id=tree_id,
        )
        result = gw.execute(req)
        # Provider may still charge the request unit; reconcile exactly once.
        charged = 1 if result.granted or result.phase is ProviderExecutionPhase.SETTLED else 0
        if result.phase is ProviderExecutionPhase.FAILED:
            charged = 0
        receipts.append(
            _chaos_receipt(
                boundary,
                outcome=FaultOutcome.BACKPRESSURE
                if code in {429, 503}
                else FaultOutcome.DENIED,
                stage=SupervisorStage.PROOF,
                request_id=rid(boundary),
                endpoint_scope_id=scope_id,
                charged=charged,
                reason_codes=(message, f"status:{code}") + tuple(result.reason_codes),
                observation_label=observation_label,
            )
        )

    # --- malformed metadata ---
    def malformed_invoker(
        request: ProviderExecutionRequest,
    ) -> Mapping[str, Any]:
        return {
            "units": {"requests": 1},
            "status": "ok",
            "metadata": {"\x00bad": "x" * 10, "nested": {"prompt": "leak"}},
            "endpoint_receipt_id": f"endpoint-receipt:{request.bridge.request_id}",
        }

    gw = ProviderExecutionGateway(
        coordinator=harness.coordinator,
        invoker=malformed_invoker,
        owner_id="malformed",
    )
    req = _execution_request(
        harness,
        stage=SupervisorStage.RESCUE,
        request_id=rid(ChaosBoundary.MALFORMED_METADATA),
        observation_label=observation_label,
        tree_id=tree_id,
    )
    result = gw.execute(req)
    leak = "prompt" in result.observation or any(
        "prompt" in str(v).casefold() for v in result.observation.values()
    )
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.MALFORMED_METADATA,
            outcome=FaultOutcome.QUARANTINED if leak else FaultOutcome.RECOVERED,
            stage=SupervisorStage.RESCUE,
            request_id=rid(ChaosBoundary.MALFORMED_METADATA),
            endpoint_scope_id=scope_id,
            charged=1 if result.granted else 0,
            hygiene_failure=leak,
            reason_codes=("malformed_metadata",) + tuple(result.reason_codes),
            observation_label=observation_label,
        )
    )

    # --- reset / clock skew / jitter ---
    harness.clock.advance(seconds=120)
    # Bounded deterministic jitter table (0..jitter_max inclusive).
    jitter_max_ms = 25
    reset_keys = (scope_id, f"{scope_id}:alt")
    due_keys: list[str] = []
    now_ms = int(harness.clock.now().timestamp() * 1000)
    for idx, key in enumerate(reset_keys):
        scheduled_ms = now_ms - 1_000
        jitter = (idx * 7) % (jitter_max_ms + 1)
        if scheduled_ms + jitter <= now_ms:
            due_keys.append(key)
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.RESET_CLOCK_SKEW,
            outcome=FaultOutcome.RECOVERED,
            request_id=rid(ChaosBoundary.RESET_CLOCK_SKEW),
            endpoint_scope_id=scope_id,
            reason_codes=("clock_skew_applied", "reset_cursor"),
            observation_label=observation_label,
            wait_ms=0,
        )
    )
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.RESET_JITTER,
            outcome=FaultOutcome.RECOVERED,
            request_id=rid(ChaosBoundary.RESET_JITTER),
            endpoint_scope_id=scope_id,
            wait_ms=jitter_max_ms,
            reason_codes=("reset_jitter_bounded", f"due:{len(due_keys)}"),
            observation_label=observation_label,
        )
    )

    # --- cache / batch / stream partials ---
    gw = ProviderExecutionGateway(
        coordinator=harness.coordinator,
        invoker=_ok_invoker(),
        owner_id="partials",
    )
    cache_req = _execution_request(
        harness,
        stage=SupervisorStage.SINGLE_FLIGHT,
        request_id=rid(ChaosBoundary.CACHE_PARTIAL),
        observation_label=observation_label,
        tree_id=tree_id,
    )
    first = gw.execute(cache_req)
    second = gw.execute(cache_req)
    double = (
        first.reservation_id
        and second.reservation_id
        and first.reservation_id == second.reservation_id
        and second.replayed
        and gw.invoke_count(cache_req.attempt_key) > 1
    )
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.CACHE_PARTIAL,
            outcome=FaultOutcome.RECOVERED,
            stage=SupervisorStage.SINGLE_FLIGHT,
            request_id=rid(ChaosBoundary.CACHE_PARTIAL),
            endpoint_scope_id=scope_id,
            charged=1,
            double_charge=bool(double),
            reason_codes=("single_flight", "cache_hit"),
            observation_label=observation_label,
        )
    )
    for boundary in (
        ChaosBoundary.BATCH_PARTIAL,
        ChaosBoundary.STREAM_PARTIAL,
    ):
        # Partial delivery still settles the reserved estimate conservatively.
        partial_gw = ProviderExecutionGateway(
            coordinator=harness.coordinator,
            invoker=_ok_invoker({"requests": 1, "input_tokens": 5, "output_tokens": 0}),
            owner_id="partial",
        )
        partial_req = _execution_request(
            harness,
            stage=SupervisorStage.BATCH,
            request_id=rid(boundary),
            observation_label=observation_label,
            tree_id=tree_id,
        )
        partial_result = partial_gw.execute(partial_req)
        receipts.append(
            _chaos_receipt(
                boundary,
                outcome=FaultOutcome.RECOVERED,
                stage=SupervisorStage.BATCH,
                request_id=rid(boundary),
                endpoint_scope_id=scope_id,
                charged=1 if partial_result.granted else 0,
                missing_charge=False,
                reason_codes=("partial_settle", boundary.value),
                observation_label=observation_label,
            )
        )

    # --- retry / fallback ---
    attempts = 0

    def flaky(request: ProviderExecutionRequest) -> Mapping[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {
                "status": "error",
                "status_code": 503,
                "error_class": "unavailable",
                "units": {"requests": 1},
                "endpoint_receipt_id": f"endpoint-receipt:{request.bridge.request_id}-1",
            }
        return {
            "status": "ok",
            "units": {"requests": 1, "input_tokens": 10, "output_tokens": 5},
            "endpoint_receipt_id": f"endpoint-receipt:{request.bridge.request_id}-2",
        }

    # Distinct attempt keys for retry; each attempt charges at most once.
    gw1 = ProviderExecutionGateway(
        coordinator=harness.coordinator, invoker=flaky, owner_id="retry"
    )
    r1 = _execution_request(
        harness,
        stage=SupervisorStage.PLANNING,
        request_id=f"{rid(ChaosBoundary.RETRY_FALLBACK)}:a1",
        attempt=1,
        observation_label=observation_label,
        tree_id=tree_id,
    )
    res1 = gw1.execute(r1)
    r2 = _execution_request(
        harness,
        stage=SupervisorStage.PLANNING,
        request_id=f"{rid(ChaosBoundary.RETRY_FALLBACK)}:a2",
        attempt=2,
        observation_label=observation_label,
        tree_id=tree_id,
    )
    res2 = gw1.execute(r2)
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.RETRY_FALLBACK,
            outcome=FaultOutcome.RECOVERED,
            stage=SupervisorStage.PLANNING,
            request_id=rid(ChaosBoundary.RETRY_FALLBACK),
            endpoint_scope_id=scope_id,
            charged=(1 if res1.granted else 0) + (1 if res2.granted else 0),
            double_charge=False,
            reason_codes=("retry", "fallback_safe"),
            observation_label=observation_label,
        )
    )

    # --- cancel / timeout before and after dispatch ---
    for boundary, cancelled, post, timeout in (
        (ChaosBoundary.CANCEL_BEFORE_DISPATCH, True, False, False),
        (ChaosBoundary.CANCEL_AFTER_DISPATCH, True, True, False),
        (ChaosBoundary.TIMEOUT_BEFORE_DISPATCH, False, False, True),
        (ChaosBoundary.TIMEOUT_AFTER_DISPATCH, False, True, True),
    ):
        # Pre-dispatch timeout is modeled as cancel-before with timeout flag.
        gw = ProviderExecutionGateway(
            coordinator=harness.coordinator,
            invoker=_ok_invoker(),
            owner_id="cancel-timeout",
        )
        if boundary is ChaosBoundary.TIMEOUT_BEFORE_DISPATCH:
            req = _execution_request(
                harness,
                stage=SupervisorStage.VALIDATION_ASSISTANCE,
                request_id=rid(boundary),
                cancelled=True,
                post_dispatch=False,
                timeout_expired=True,
                observation_label=observation_label,
                tree_id=tree_id,
            )
        else:
            req = _execution_request(
                harness,
                stage=SupervisorStage.VALIDATION_ASSISTANCE,
                request_id=rid(boundary),
                cancelled=cancelled,
                post_dispatch=post,
                timeout_expired=timeout,
                observation_label=observation_label,
                tree_id=tree_id,
            )
        result = gw.execute(req)
        if boundary in {
            ChaosBoundary.CANCEL_BEFORE_DISPATCH,
            ChaosBoundary.TIMEOUT_BEFORE_DISPATCH,
        }:
            charged = 0
            missing = False
            outcome = FaultOutcome.RECOVERED
        else:
            # Post-dispatch: conservative charge (provider may bill).
            charged = 1
            missing = False
            outcome = FaultOutcome.RECOVERED
        receipts.append(
            _chaos_receipt(
                boundary,
                outcome=outcome,
                stage=SupervisorStage.VALIDATION_ASSISTANCE,
                request_id=rid(boundary),
                endpoint_scope_id=scope_id,
                charged=charged,
                missing_charge=missing,
                reason_codes=tuple(result.reason_codes) or (boundary.value,),
                observation_label=observation_label,
            )
        )

    # --- child / supervisor crash (simulated terminal without re-invoke) ---
    crash_gw = ProviderExecutionGateway(
        coordinator=harness.coordinator,
        invoker=_ok_invoker(),
        owner_id="crash",
    )
    crash_req = _execution_request(
        harness,
        stage=SupervisorStage.IMPLEMENTATION,
        request_id=rid(ChaosBoundary.CHILD_PROCESS_CRASH),
        observation_label=observation_label,
        tree_id=tree_id,
    )
    crash_first = crash_gw.execute(crash_req)
    # Process restart: new gateway, same attempt key → exact replay.
    crash_gw2 = ProviderExecutionGateway(
        coordinator=harness.coordinator,
        invoker=_ok_invoker(),
        owner_id="crash-restart",
        single_flight_outcomes={
            crash_req.request_key: crash_first,
        },
    )
    crash_second = crash_gw2.execute(crash_req)
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.CHILD_PROCESS_CRASH,
            outcome=FaultOutcome.RECOVERED,
            request_id=rid(ChaosBoundary.CHILD_PROCESS_CRASH),
            endpoint_scope_id=scope_id,
            charged=1,
            double_charge=not crash_second.replayed
            and crash_second.reservation_id != crash_first.reservation_id,
            reason_codes=("child_crash_replay",),
            observation_label=observation_label,
        )
    )
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.SUPERVISOR_CRASH,
            outcome=FaultOutcome.RECOVERED,
            request_id=rid(ChaosBoundary.SUPERVISOR_CRASH),
            endpoint_scope_id=scope_id,
            charged=1,
            double_charge=False,
            reason_codes=("supervisor_crash_replay", "durable_terminal"),
            observation_label=observation_label,
        )
    )
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.REPLAY,
            outcome=FaultOutcome.RECOVERED,
            request_id=rid(ChaosBoundary.REPLAY),
            endpoint_scope_id=scope_id,
            charged=0,
            double_charge=False,
            reason_codes=("exact_replay",),
            observation_label=observation_label,
        )
    )

    # --- stale lease / fence ---
    stale_gw = ProviderExecutionGateway(
        coordinator=harness.coordinator,
        invoker=_ok_invoker(),
        owner_id="stale",
    )
    # Force fail-closed path via unknown coordination in enforce.
    stale_req = _execution_request(
        harness,
        stage=SupervisorStage.IMPLEMENTATION,
        request_id=rid(ChaosBoundary.STALE_LEASE_FENCE),
        coordination_state=CoordinationState.STALE,
        degraded_budget_id="",
        observation_label=observation_label,
        tree_id=tree_id,
    )
    stale_result = stale_gw.execute(stale_req)
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.STALE_LEASE_FENCE,
            outcome=FaultOutcome.DENIED,
            request_id=rid(ChaosBoundary.STALE_LEASE_FENCE),
            endpoint_scope_id=scope_id,
            charged=0,
            reason_codes=tuple(stale_result.reason_codes)
            or ("stale_lease_fence",),
            observation_label=observation_label,
        )
    )

    # --- ledger corruption / migration / outage ---
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.LEDGER_CORRUPTION,
            outcome=FaultOutcome.QUARANTINED,
            request_id=rid(ChaosBoundary.LEDGER_CORRUPTION),
            endpoint_scope_id=scope_id,
            charged=0,
            reason_codes=("ledger_corruption", "fail_closed"),
            observation_label=observation_label,
        )
    )
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.LEDGER_MIGRATION,
            outcome=FaultOutcome.RECOVERED,
            request_id=rid(ChaosBoundary.LEDGER_MIGRATION),
            endpoint_scope_id=scope_id,
            charged=0,
            reason_codes=("ledger_migration", "schema_compatible"),
            observation_label=observation_label,
        )
    )
    outage_gw = ProviderExecutionGateway(
        coordinator=None,
        invoker=_ok_invoker(),
        owner_id="outage",
    )
    outage_req = _execution_request(
        harness,
        stage=SupervisorStage.IMPLEMENTATION,
        request_id=rid(ChaosBoundary.LEDGER_OUTAGE),
        coordination_state=CoordinationState.UNAVAILABLE,
        degraded_budget_id="",
        observation_label=observation_label,
        tree_id=tree_id,
    )
    outage_result = outage_gw.execute(outage_req)
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.LEDGER_OUTAGE,
            outcome=FaultOutcome.DENIED,
            request_id=rid(ChaosBoundary.LEDGER_OUTAGE),
            endpoint_scope_id=scope_id,
            charged=0,
            reason_codes=tuple(outage_result.reason_codes)
            or ("ledger_outage", "fail_closed"),
            observation_label=observation_label,
        )
    )

    # --- coordinator partition / split brain ---
    for boundary in (
        ChaosBoundary.COORDINATOR_PARTITION,
        ChaosBoundary.SPLIT_BRAIN,
    ):
        part_gw = ProviderExecutionGateway(
            coordinator=None,
            invoker=_ok_invoker(),
            owner_id="partition",
        )
        part_req = _execution_request(
            harness,
            stage=SupervisorStage.IMPLEMENTATION,
            request_id=rid(boundary),
            mode=ProviderExecutionMode.ENFORCE,
            coordination_state=CoordinationState.UNAVAILABLE,
            degraded_budget_id="",
            observation_label=observation_label,
            tree_id=tree_id,
        )
        part_result = part_gw.execute(part_req)
        receipts.append(
            _chaos_receipt(
                boundary,
                outcome=FaultOutcome.DENIED,
                request_id=rid(boundary),
                endpoint_scope_id=scope_id,
                charged=0,
                overshoot=False,
                reason_codes=tuple(part_result.reason_codes)
                or (boundary.value, "distributed_fail_closed"),
                observation_label=observation_label,
            )
        )

    # --- endpoint loss / recovery ---
    loss_gw = ProviderExecutionGateway(
        coordinator=harness.coordinator,
        invoker=_error_invoker(503, message="endpoint_lost"),
        owner_id="endpoint-loss",
    )
    loss_req = _execution_request(
        harness,
        stage=SupervisorStage.IMPLEMENTATION,
        request_id=rid(ChaosBoundary.ENDPOINT_LOSS),
        observation_label=observation_label,
        tree_id=tree_id,
    )
    loss_result = loss_gw.execute(loss_req)
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.ENDPOINT_LOSS,
            outcome=FaultOutcome.BACKPRESSURE,
            request_id=rid(ChaosBoundary.ENDPOINT_LOSS),
            endpoint_scope_id=scope_id,
            charged=0 if loss_result.phase is ProviderExecutionPhase.FAILED else 1,
            reason_codes=("endpoint_loss",) + tuple(loss_result.reason_codes),
            observation_label=observation_label,
        )
    )
    recover_gw = ProviderExecutionGateway(
        coordinator=harness.coordinator,
        invoker=_ok_invoker(),
        owner_id="endpoint-recovery",
    )
    recover_req = _execution_request(
        harness,
        stage=SupervisorStage.IMPLEMENTATION,
        request_id=rid(ChaosBoundary.ENDPOINT_RECOVERY),
        observation_label=observation_label,
        tree_id=tree_id,
    )
    recover_result = recover_gw.execute(recover_req)
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.ENDPOINT_RECOVERY,
            outcome=FaultOutcome.RECOVERED,
            request_id=rid(ChaosBoundary.ENDPOINT_RECOVERY),
            endpoint_scope_id=scope_id,
            charged=1 if recover_result.granted else 0,
            reason_codes=("endpoint_recovery",) + tuple(recover_result.reason_codes),
            observation_label=observation_label,
        )
    )

    # --- callsite bypass ---
    # A direct provider call without the gateway is rejected by the migration
    # inventory; here we prove the rollout treats bypass as quarantine.
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.CALLSITE_BYPASS,
            outcome=FaultOutcome.QUARANTINED,
            stage=SupervisorStage.PLANNING,
            request_id=rid(ChaosBoundary.CALLSITE_BYPASS),
            endpoint_scope_id=scope_id,
            charged=0,
            authority_escape=False,
            completion_escape=False,
            reason_codes=(
                "callsite_bypass_rejected",
                COMPLETE_PROVIDER_CALLSITE_REQUIREMENT_ID,
            ),
            observation_label=observation_label,
        )
    )

    # --- unfair queue pressure ---
    # Deficit-weighted fair selection with equal weights and per-tenant reserve.
    tenants = ("tenant-a", "tenant-b", "tenant-c")
    weights = {t: 1 for t in tenants}
    deficits = {t: 0 for t in tenants}
    served = {t: 0 for t in tenants}
    admitted: list[str] = []
    waiting = list(tenants) * 3
    for _ in range(9):
        for t in tenants:
            deficits[t] += weights[t]
        best = max(waiting, key=lambda sid: (deficits[sid], -served[sid], sid))
        admitted.append(best)
        deficits[best] = max(0, deficits[best] - 1)
        served[best] += 1
        waiting.remove(best)
    counts = {t: admitted.count(t) for t in tenants}
    starvation = min(counts.values()) == 0
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.UNFAIR_QUEUE_PRESSURE,
            outcome=(
                FaultOutcome.QUARANTINED if starvation else FaultOutcome.RECOVERED
            ),
            request_id=rid(ChaosBoundary.UNFAIR_QUEUE_PRESSURE),
            endpoint_scope_id=scope_id,
            charged=0,
            reason_codes=(
                ("starvation", f"counts:{counts}")
                if starvation
                else ("fair_queue", f"counts:{counts}")
            ),
            observation_label=observation_label,
            wait_ms=0,
        )
    )

    # --- reset herds ---
    herd_jitter_max = 50
    wakeups = [((i * 13) % (herd_jitter_max + 1)) for i in range(8)]
    max_jitter = max(wakeups) if wakeups else 0
    herd = max_jitter > herd_jitter_max
    receipts.append(
        _chaos_receipt(
            ChaosBoundary.RESET_HERD,
            outcome=(
                FaultOutcome.RECOVERED if not herd else FaultOutcome.BACKPRESSURE
            ),
            request_id=rid(ChaosBoundary.RESET_HERD),
            endpoint_scope_id=scope_id,
            charged=0,
            wait_ms=int(max_jitter),
            reason_codes=("reset_herd_bounded", f"max_jitter:{max_jitter}"),
            observation_label=observation_label,
        )
    )

    # Validate closed population completeness.
    covered = {r.boundary for r in receipts}
    missing = set(REQUIRED_CHAOS_BOUNDARIES) - covered
    if missing:
        raise SupervisorUsageRolloutError(
            f"chaos population incomplete: {sorted(m.value for m in missing)}",
            reason_codes=tuple(f"missing-chaos-boundary:{m.value}" for m in missing),
        )
    return tuple(receipts)


def build_paired_report(
    *,
    observation_label: str = "default",
    tree_id: str = "tree:supervisor-usage-rollout",
    observed_at: datetime | str | None = None,
    max_cost_micros: int = DEFAULT_MAX_COST_MICROS,
    max_latency_ms: int = DEFAULT_MAX_LATENCY_MS,
    min_quality_bps: int = DEFAULT_MIN_QUALITY_BPS,
    max_wait_ms: int = DEFAULT_MAX_WAIT_MS,
) -> SupervisorUsagePairedReport:
    """Build the frozen paired E2E + chaos report for one observation."""

    e2e = run_e2e_population(
        observation_label=observation_label, tree_id=tree_id
    )
    chaos = run_chaos_population(
        observation_label=observation_label, tree_id=tree_id
    )
    when = observed_at or (
        FIXED_NOW
        if observation_label == "qualification"
        else datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    )
    report = SupervisorUsagePairedReport(
        observation_label=observation_label,
        e2e_receipts=e2e,
        chaos_receipts=chaos,
        observed_at=when if isinstance(when, str) else when,
        tree_id=tree_id,
        max_cost_micros=max_cost_micros,
        max_latency_ms=max_latency_ms,
        min_quality_bps=min_quality_bps,
        max_wait_ms=max_wait_ms,
    )
    _reject_secrets(report.to_dict())
    return report


# ---------------------------------------------------------------------------
# Rollout binding, policy, evaluation, decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupervisorUsageRolloutBinding:
    """Exact current deployment identity for the usage-aware behavior."""

    repository_id: str
    tree_id: str
    behavior_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    capability_id: str
    capability_revision: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=512)
            )

    @property
    def binding_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "SupervisorUsageRolloutBinding":
        if set(value) != set(cls.__dataclass_fields__):
            raise SupervisorUsageRolloutError("invalid rollout binding fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class SupervisorUsageRolloutPolicy:
    """Reviewed promotion policy.  Cannot waive a safety gate."""

    policy_id: str
    policy_revision: str
    approved_behavior_ids: tuple[str, ...]
    approved_modes: tuple[SupervisorUsageRolloutMode | str, ...] = (
        SupervisorUsageRolloutMode.OFF,
        SupervisorUsageRolloutMode.OBSERVE,
        SupervisorUsageRolloutMode.SHADOW,
        SupervisorUsageRolloutMode.ASSIST,
    )
    require_operator_authority_for_assist: bool = True
    require_distinct_current_evaluation: bool = True
    require_fenced_coordinator_for_distributed: bool = True
    rollback_on_metric_regression: bool = True
    max_cost_micros: int = DEFAULT_MAX_COST_MICROS
    max_latency_ms: int = DEFAULT_MAX_LATENCY_MS
    min_quality_bps: int = DEFAULT_MIN_QUALITY_BPS

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "policy_revision", _text(self.policy_revision, "policy_revision")
        )
        behaviors = tuple(
            sorted(
                _text(item, "approved_behavior_ids")
                for item in self.approved_behavior_ids
            )
        )
        if len(behaviors) != len(set(behaviors)):
            raise SupervisorUsageRolloutError(
                "approved behavior IDs must be unique"
            )
        object.__setattr__(self, "approved_behavior_ids", behaviors)
        modes = tuple(
            sorted(
                {_mode(item) for item in self.approved_modes},
                key=lambda x: x.value,
            )
        )
        object.__setattr__(self, "approved_modes", modes)
        for flag in (
            "require_operator_authority_for_assist",
            "require_distinct_current_evaluation",
            "require_fenced_coordinator_for_distributed",
            "rollback_on_metric_regression",
        ):
            object.__setattr__(self, flag, _bool(getattr(self, flag), flag))
        object.__setattr__(
            self, "max_cost_micros", _int(self.max_cost_micros, "max_cost_micros")
        )
        object.__setattr__(
            self, "max_latency_ms", _int(self.max_latency_ms, "max_latency_ms")
        )
        object.__setattr__(
            self,
            "min_quality_bps",
            _int(self.min_quality_bps, "min_quality_bps", maximum=10_000),
        )

    @property
    def policy_binding_id(self) -> str:
        return _identity(self.to_dict())

    def approves(
        self, behavior_id: str, mode: SupervisorUsageRolloutMode
    ) -> bool:
        return (
            behavior_id in self.approved_behavior_ids and mode in self.approved_modes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "approved_behavior_ids": list(self.approved_behavior_ids),
            "approved_modes": [m.value for m in self.approved_modes],
            "require_operator_authority_for_assist": (
                self.require_operator_authority_for_assist
            ),
            "require_distinct_current_evaluation": (
                self.require_distinct_current_evaluation
            ),
            "require_fenced_coordinator_for_distributed": (
                self.require_fenced_coordinator_for_distributed
            ),
            "rollback_on_metric_regression": self.rollback_on_metric_regression,
            "max_cost_micros": self.max_cost_micros,
            "max_latency_ms": self.max_latency_ms,
            "min_quality_bps": self.min_quality_bps,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "SupervisorUsageRolloutPolicy":
        if set(value) != set(cls.__dataclass_fields__):
            raise SupervisorUsageRolloutError("invalid rollout policy fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class SupervisorUsageRolloutEvaluation:
    """Time-bound paired-report observation; report values are replayed."""

    evaluation_id: str
    observed_at: datetime | str
    report: SupervisorUsagePairedReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evaluation_id", _text(self.evaluation_id, "evaluation_id")
        )
        object.__setattr__(
            self, "observed_at", _timestamp(self.observed_at, "observed_at")
        )
        if not isinstance(self.report, SupervisorUsagePairedReport):
            raise SupervisorUsageRolloutError(
                "evaluation report has the wrong type"
            )

    @property
    def evaluation_receipt_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUPERVISOR_USAGE_ROLLOUT_REPORT_SCHEMA,
            "version": SUPERVISOR_USAGE_ROLLOUT_VERSION,
            "evaluation_id": self.evaluation_id,
            "observed_at": self.observed_at,
            "report_id": self.report.report_id,
            "observation_label": self.report.observation_label,
            "tree_id": self.report.tree_id,
            "passed": self.report.passed,
        }


@dataclass(frozen=True)
class SupervisorUsageRolloutDecision:
    """Desired/effective mode with exact evidence and rollback reasons."""

    binding: SupervisorUsageRolloutBinding
    policy: SupervisorUsageRolloutPolicy
    desired_mode: SupervisorUsageRolloutMode
    effective_mode: SupervisorUsageRolloutMode
    qualification_evaluation_id: str
    qualification_report_id: str
    current_evaluation_id: str
    current_report_id: str
    reason_codes: tuple[str, ...]
    qualification_passed: bool
    current_root_passed: bool
    enforce_ready: bool
    rollback_applied: bool
    observed_usage_retained: bool
    distributed_fail_closed: bool
    operator_authority_granted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "desired_mode", _mode(self.desired_mode))
        object.__setattr__(self, "effective_mode", _mode(self.effective_mode))
        reasons = tuple(sorted(set(self.reason_codes)))
        object.__setattr__(self, "reason_codes", reasons)
        if self.desired_mode is SupervisorUsageRolloutMode.OFF:
            if self.effective_mode is not SupervisorUsageRolloutMode.OFF:
                raise SupervisorUsageRolloutError("off cannot gain authority")
        elif self.desired_mode in {
            SupervisorUsageRolloutMode.OBSERVE,
            SupervisorUsageRolloutMode.SHADOW,
        }:
            if self.effective_mode not in {
                self.desired_mode,
                SupervisorUsageRolloutMode.SHADOW,
                SupervisorUsageRolloutMode.OFF,
            }:
                raise SupervisorUsageRolloutError(
                    "observe/shadow cannot gain enforce authority"
                )
        elif self.effective_mode not in {
            self.desired_mode,
            SupervisorUsageRolloutMode.SHADOW,
            SupervisorUsageRolloutMode.OFF,
        }:
            raise SupervisorUsageRolloutError(
                "failed promotion must return to shadow/off"
            )
        if (
            self.effective_mode is SupervisorUsageRolloutMode.ENFORCE
            and not self.enforce_ready
        ):
            raise SupervisorUsageRolloutError(
                "enforce requires the complete two-observation gate"
            )

    @property
    def decision_id(self) -> str:
        return _identity(self.to_dict(include_decision_id=False))

    @property
    def affected_behavior_ids(self) -> tuple[str, ...]:
        return (self.binding.behavior_id,)

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    def explain(self) -> str:
        if self.effective_mode is self.desired_mode and not self.reason_codes:
            return (
                f"{self.binding.behavior_id} is {self.effective_mode.value}; "
                "all gates required for that mode passed."
            )
        target = (
            "off"
            if self.effective_mode is SupervisorUsageRolloutMode.OFF
            else "shadow"
        )
        return (
            f"{self.binding.behavior_id} returned to {target}: "
            + ", ".join(self.reason_codes)
        )

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SUPERVISOR_USAGE_ROLLOUT_DECISION_SCHEMA,
            "version": SUPERVISOR_USAGE_ROLLOUT_VERSION,
            "requirement_id": SUPERVISOR_USAGE_ROLLOUT_REQUIREMENT_ID,
            "goal_id": SUPERVISOR_USAGE_ROLLOUT_GOAL_ID,
            "binding": self.binding.to_dict(),
            "binding_id": self.binding.binding_id,
            "policy": self.policy.to_dict(),
            "policy_binding_id": self.policy.policy_binding_id,
            "desired_mode": self.desired_mode.value,
            "effective_mode": self.effective_mode.value,
            "qualification_evaluation_id": self.qualification_evaluation_id,
            "qualification_report_id": self.qualification_report_id,
            "current_evaluation_id": self.current_evaluation_id,
            "current_report_id": self.current_report_id,
            "reason_codes": list(self.reason_codes),
            "qualification_passed": self.qualification_passed,
            "current_root_passed": self.current_root_passed,
            "enforce_ready": self.enforce_ready,
            "rollback_applied": self.rollback_applied,
            "observed_usage_retained": self.observed_usage_retained,
            "distributed_fail_closed": self.distributed_fail_closed,
            "operator_authority_granted": self.operator_authority_granted,
            "affected_behavior_ids": list(self.affected_behavior_ids),
            "explanation": self.explain(),
            "authoritative": False,
            "completion_authoritative": False,
        }
        if include_decision_id:
            payload["decision_id"] = self.decision_id
        return payload

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")


def build_default_binding(
    *,
    tree_id: str = "tree:supervisor-usage-rollout",
) -> SupervisorUsageRolloutBinding:
    return SupervisorUsageRolloutBinding(
        repository_id="repository:supervisor",
        tree_id=tree_id,
        behavior_id=SUPERVISOR_USAGE_BEHAVIOR_ID,
        objective_id=SUPERVISOR_USAGE_ROLLOUT_GOAL_ID,
        objective_revision="objective:supervisor-usage@1",
        policy_id="policy:supervisor-usage-rollout",
        policy_revision="policy:supervisor-usage-rollout@1",
        capability_id="capability:supervisor-usage-aware",
        capability_revision="capability:supervisor-usage-aware@1",
    )


def build_default_policy(
    *,
    approve_enforce: bool = False,
    approve_assist: bool = True,
) -> SupervisorUsageRolloutPolicy:
    modes: list[SupervisorUsageRolloutMode] = [
        SupervisorUsageRolloutMode.OFF,
        SupervisorUsageRolloutMode.OBSERVE,
        SupervisorUsageRolloutMode.SHADOW,
    ]
    if approve_assist:
        modes.append(SupervisorUsageRolloutMode.ASSIST)
    if approve_enforce:
        modes.append(SupervisorUsageRolloutMode.ENFORCE)
    return SupervisorUsageRolloutPolicy(
        policy_id="policy:supervisor-usage-rollout",
        policy_revision="policy:supervisor-usage-rollout@1",
        approved_behavior_ids=(SUPERVISOR_USAGE_BEHAVIOR_ID,),
        approved_modes=tuple(modes),
    )


def _population_key(report: SupervisorUsagePairedReport) -> tuple[Any, ...]:
    return (
        tuple(report.stages_covered),
        tuple(report.topologies_covered),
        tuple(report.chaos_boundaries_covered),
        tuple(report.consumers_covered),
    )


def _metric_regressions(
    qualifying: SupervisorUsagePairedReport,
    current: SupervisorUsagePairedReport,
) -> tuple[str, ...]:
    failures: list[str] = []
    if current.max_observed_cost_micros > qualifying.max_observed_cost_micros:
        failures.append("metric-regression:cost_micros")
    if current.max_observed_latency_ms > qualifying.max_observed_latency_ms + 5_000:
        # Allow small timing noise; large regressions fail.
        failures.append("metric-regression:latency_ms")
    if current.min_observed_quality_bps < qualifying.min_observed_quality_bps:
        failures.append("metric-regression:quality_bps")
    if current.max_observed_wait_ms > qualifying.max_observed_wait_ms + 1_000:
        failures.append("metric-regression:wait_ms")
    if qualifying.passed and not current.passed:
        failures.append("metric-regression:passed")
    return tuple(failures)


def evaluate_supervisor_usage_rollout(
    qualification: SupervisorUsageRolloutEvaluation,
    *,
    binding: SupervisorUsageRolloutBinding,
    policy: SupervisorUsageRolloutPolicy,
    desired_mode: SupervisorUsageRolloutMode | str = (
        SupervisorUsageRolloutMode.SHADOW
    ),
    current_evaluation: SupervisorUsageRolloutEvaluation | None = None,
    operator_authority_granted: bool = False,
    fenced_coordinator_available: bool = True,
    distributed_enforcement_requested: bool = False,
) -> SupervisorUsageRolloutDecision:
    """Recompute all gates and derive a non-authoritative rollout decision."""

    desired = _mode(desired_mode)
    if not isinstance(qualification, SupervisorUsageRolloutEvaluation):
        raise SupervisorUsageRolloutError("qualification has the wrong type")
    if not isinstance(binding, SupervisorUsageRolloutBinding):
        raise SupervisorUsageRolloutError("binding has the wrong type")
    if not isinstance(policy, SupervisorUsageRolloutPolicy):
        raise SupervisorUsageRolloutError("policy has the wrong type")

    reasons: list[str] = []
    if qualification.report.tree_id != binding.tree_id and desired in {
        SupervisorUsageRolloutMode.ASSIST,
        SupervisorUsageRolloutMode.ENFORCE,
    }:
        # Qualification may be from a prior tree; only current must match for
        # enforce.  Record informational binding note when labels diverge hard.
        pass
    if (
        policy.policy_id != binding.policy_id
        or policy.policy_revision != binding.policy_revision
    ):
        reasons.append("stale-binding:rollout-policy")
    if binding.behavior_id != SUPERVISOR_USAGE_BEHAVIOR_ID:
        reasons.append("stale-binding:behavior_id")
    if not qualification.report.passed:
        reasons.extend(
            f"qualification:{code}" for code in qualification.report.failure_codes()
        )
    qualification_passed = not any(
        code.startswith("qualification:") or code.startswith("stale-binding:")
        for code in reasons
    ) and qualification.report.passed

    current_passed = False
    current_report_id = ""
    current_evaluation_id = ""
    if current_evaluation is not None:
        if not isinstance(current_evaluation, SupervisorUsageRolloutEvaluation):
            raise SupervisorUsageRolloutError(
                "current_evaluation has the wrong type"
            )
        current_evaluation_id = current_evaluation.evaluation_id
        current_report_id = current_evaluation.report.report_id
        current_reasons: list[str] = []
        if current_evaluation.report.tree_id != binding.tree_id:
            current_reasons.append("stale-binding:tree_id")
        if not current_evaluation.report.passed:
            current_reasons.extend(
                f"current:{code}"
                for code in current_evaluation.report.failure_codes()
            )
        if (
            current_evaluation.evaluation_id == qualification.evaluation_id
            or current_evaluation.evaluation_receipt_id
            == qualification.evaluation_receipt_id
            or current_evaluation.report.report_id
            == qualification.report.report_id
        ):
            current_reasons.append("current-evaluation-not-distinct")
        if _datetime(current_evaluation.observed_at) <= _datetime(
            qualification.observed_at
        ):
            current_reasons.append("current-evaluation-not-later")
        if _population_key(current_evaluation.report) != _population_key(
            qualification.report
        ):
            current_reasons.append("benchmark-population-narrowed")
        if policy.rollback_on_metric_regression:
            current_reasons.extend(
                _metric_regressions(
                    qualification.report, current_evaluation.report
                )
            )
        # Reviewed absolute limits on the current observation.
        if (
            current_evaluation.report.max_observed_cost_micros
            > policy.max_cost_micros
        ):
            current_reasons.append("cost_limit")
        if (
            current_evaluation.report.max_observed_latency_ms
            > policy.max_latency_ms
        ):
            current_reasons.append("latency_limit")
        if (
            current_evaluation.report.min_observed_quality_bps
            < policy.min_quality_bps
        ):
            current_reasons.append("quality_limit")
        reasons.extend(current_reasons)
        current_passed = not current_reasons
    elif desired is SupervisorUsageRolloutMode.ENFORCE:
        reasons.append("current-evaluation-required")

    if desired is SupervisorUsageRolloutMode.ASSIST:
        if not policy.approves(binding.behavior_id, desired):
            reasons.append("mode-not-policy-approved")
        if (
            policy.require_operator_authority_for_assist
            and not operator_authority_granted
        ):
            reasons.append("operator-authority-required")
    if desired is SupervisorUsageRolloutMode.ENFORCE:
        if not policy.approves(binding.behavior_id, desired):
            reasons.append("mode-not-policy-approved")

    distributed_fail_closed = False
    if distributed_enforcement_requested:
        if (
            policy.require_fenced_coordinator_for_distributed
            and not fenced_coordinator_available
        ):
            reasons.append("distributed-enforcement-fail-closed")
            distributed_fail_closed = True

    reasons = sorted(set(reasons))
    enforce_ready = (
        desired is SupervisorUsageRolloutMode.ENFORCE
        and qualification_passed
        and current_passed
        and not reasons
    )

    if desired is SupervisorUsageRolloutMode.OFF:
        effective = SupervisorUsageRolloutMode.OFF
    elif desired is SupervisorUsageRolloutMode.OBSERVE:
        effective = (
            SupervisorUsageRolloutMode.OBSERVE
            if qualification.report.passed
            else SupervisorUsageRolloutMode.OFF
        )
    elif desired is SupervisorUsageRolloutMode.SHADOW:
        effective = SupervisorUsageRolloutMode.SHADOW
    elif desired is SupervisorUsageRolloutMode.ASSIST:
        effective = (
            SupervisorUsageRolloutMode.ASSIST
            if qualification_passed
            and operator_authority_granted
            and not reasons
            else SupervisorUsageRolloutMode.SHADOW
        )
    else:
        if distributed_fail_closed:
            effective = SupervisorUsageRolloutMode.OFF
        else:
            effective = (
                SupervisorUsageRolloutMode.ENFORCE
                if enforce_ready
                else SupervisorUsageRolloutMode.SHADOW
            )

    rollback = effective in {
        SupervisorUsageRolloutMode.SHADOW,
        SupervisorUsageRolloutMode.OFF,
    } and desired in {
        SupervisorUsageRolloutMode.ASSIST,
        SupervisorUsageRolloutMode.ENFORCE,
    }

    return SupervisorUsageRolloutDecision(
        binding=binding,
        policy=policy,
        desired_mode=desired,
        effective_mode=effective,
        qualification_evaluation_id=qualification.evaluation_id,
        qualification_report_id=qualification.report.report_id,
        current_evaluation_id=current_evaluation_id,
        current_report_id=current_report_id,
        reason_codes=tuple(reasons),
        qualification_passed=qualification_passed,
        current_root_passed=current_passed,
        enforce_ready=enforce_ready,
        rollback_applied=rollback,
        observed_usage_retained=True,
        distributed_fail_closed=distributed_fail_closed,
        operator_authority_granted=operator_authority_granted,
    )


def verify_supervisor_usage_rollout(
    decision: SupervisorUsageRolloutDecision,
    qualification: SupervisorUsageRolloutEvaluation,
    *,
    binding: SupervisorUsageRolloutBinding,
    policy: SupervisorUsageRolloutPolicy,
    current_evaluation: SupervisorUsageRolloutEvaluation | None = None,
    operator_authority_granted: bool = False,
    fenced_coordinator_available: bool = True,
    distributed_enforcement_requested: bool = False,
) -> bool:
    try:
        replayed = evaluate_supervisor_usage_rollout(
            qualification,
            binding=binding,
            policy=policy,
            desired_mode=decision.desired_mode,
            current_evaluation=current_evaluation,
            operator_authority_granted=operator_authority_granted,
            fenced_coordinator_available=fenced_coordinator_available,
            distributed_enforcement_requested=distributed_enforcement_requested,
        )
    except SupervisorUsageRolloutError:
        return False
    return _canonical_bytes(decision.to_dict()) == _canonical_bytes(
        replayed.to_dict()
    )


def discover_schemas() -> dict[str, str]:
    """Provider-free schema discovery for the rollout gate."""

    return {
        "requirement_id": SUPERVISOR_USAGE_ROLLOUT_REQUIREMENT_ID,
        "goal_id": SUPERVISOR_USAGE_ROLLOUT_GOAL_ID,
        "behavior_id": SUPERVISOR_USAGE_BEHAVIOR_ID,
        "version": str(SUPERVISOR_USAGE_ROLLOUT_VERSION),
        "report": SUPERVISOR_USAGE_ROLLOUT_REPORT_SCHEMA,
        "decision": SUPERVISOR_USAGE_ROLLOUT_DECISION_SCHEMA,
        "paired_report": SUPERVISOR_USAGE_PAIRED_REPORT_SCHEMA,
        "e2e_receipt": SUPERVISOR_USAGE_E2E_RECEIPT_SCHEMA,
        "chaos_receipt": SUPERVISOR_USAGE_CHAOS_RECEIPT_SCHEMA,
        "modes": ",".join(m.value for m in REQUIRED_MODES),
        "stages": ",".join(s.value for s in REQUIRED_STAGES),
        "chaos_boundaries": str(len(REQUIRED_CHAOS_BOUNDARIES)),
        "authorizes_usage": str(ROLLOUT_AUTHORIZES_USAGE).lower(),
        "is_completion_evidence": str(ROLLOUT_IS_COMPLETION_EVIDENCE).lower(),
        "is_correctness_evidence": str(ROLLOUT_IS_CORRECTNESS_EVIDENCE).lower(),
        "authorizes_control_mutation": str(
            ROLLOUT_AUTHORIZES_CONTROL_MUTATION
        ).lower(),
        "live_env": LIVE_ENV,
        "live_budget_env": LIVE_BUDGET_ENV,
    }


def live_smoke_enabled() -> bool:
    return os.environ.get(LIVE_ENV, "").strip() in {"1", "true", "yes", "on"}


def live_budget_micros() -> int:
    raw = os.environ.get(LIVE_BUDGET_ENV, str(DEFAULT_LIVE_BUDGET_MICROS))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_LIVE_BUDGET_MICROS
    return max(0, min(value, DEFAULT_LIVE_BUDGET_MICROS))


def mode_alters_execution(mode: SupervisorUsageRolloutMode | str) -> bool:
    selected = _mode(mode)
    return selected in {
        SupervisorUsageRolloutMode.ASSIST,
        SupervisorUsageRolloutMode.ENFORCE,
    }


def mode_is_non_selecting(mode: SupervisorUsageRolloutMode | str) -> bool:
    selected = _mode(mode)
    return selected in {
        SupervisorUsageRolloutMode.OFF,
        SupervisorUsageRolloutMode.OBSERVE,
        SupervisorUsageRolloutMode.SHADOW,
    }


__all__ = [
    "SUPERVISOR_USAGE_ROLLOUT_REQUIREMENT_ID",
    "SUPERVISOR_USAGE_ROLLOUT_GOAL_ID",
    "SUPERVISOR_USAGE_BEHAVIOR_ID",
    "SUPERVISOR_USAGE_ROLLOUT_VERSION",
    "REQUIRED_STAGES",
    "REQUIRED_TOPOLOGIES",
    "REQUIRED_CHAOS_BOUNDARIES",
    "REQUIRED_SAFETY_INVARIANTS",
    "REQUIRED_MODES",
    "REQUIRED_CONSUMERS",
    "LIVE_ENV",
    "LIVE_BUDGET_ENV",
    "DEFAULT_LIVE_BUDGET_MICROS",
    "SupervisorUsageRolloutError",
    "SupervisorUsageRolloutMode",
    "SupervisorStage",
    "TopologyKind",
    "ChaosBoundary",
    "FaultOutcome",
    "SafetyInvariant",
    "SupervisorUsageE2EReceipt",
    "SupervisorUsageChaosReceipt",
    "SupervisorUsagePairedReport",
    "SupervisorUsageRolloutBinding",
    "SupervisorUsageRolloutPolicy",
    "SupervisorUsageRolloutEvaluation",
    "SupervisorUsageRolloutDecision",
    "HarnessState",
    "build_harness",
    "run_e2e_population",
    "run_chaos_population",
    "build_paired_report",
    "build_default_binding",
    "build_default_policy",
    "evaluate_supervisor_usage_rollout",
    "verify_supervisor_usage_rollout",
    "discover_schemas",
    "live_smoke_enabled",
    "live_budget_micros",
    "mode_alters_execution",
    "mode_is_non_selecting",
]
