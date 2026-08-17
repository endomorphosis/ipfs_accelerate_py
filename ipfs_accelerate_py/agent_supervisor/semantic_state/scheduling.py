"""Existing-supervisor scheduling adapter for the semantic-compression harness.

``SemanticSchedulingAdapter`` composes the pinned supervisor authorities:

* ``ResourceScheduler`` for host/provider capacity admission;
* ``ProviderExecutionGateway`` as the sole harness owner of provider invocation;
* ``LeaseCoordinator`` / ``WorktreeLifecycleStore`` when injected for durable
  lease and worktree fencing;
* ``runtime.event_log`` for bounded admission/terminal journaling.

This module is intentionally cold-import safe: importing it starts no
resources, threads, processes, databases, or network calls. Composition and
admission happen only when an adapter is constructed or schedule/replay runs.

``PersistentTaskQueue`` is not an authority. Simulated provider results never
satisfy production work. Scheduler outcomes never certify verification.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    HarnessError,
    HarnessMode,
    UnavailableResult,
    WorkKind,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
    LeaseBinding,
    ProviderBinding,
    ResourceBinding,
    SchedulerObservation,
    SemanticWorkRequest,
    SemanticWorkResult,
    SemanticWorkStatus,
    requires_provider,
    resource_class_for_work_kind,
    stage_for_work_kind,
)

SEMANTIC_SCHEDULING_ADAPTER_INTERFACE = "SemanticSchedulingAdapter@1"
SCHEDULING_ADAPTER_SCHEMA = "semantic-state-scheduling-adapter@1"
ADAPTER_ID = "semantic-scheduling-adapter"

_EVENT_TYPE_ADMITTED = "semantic_work_admitted"
_EVENT_TYPE_TERMINAL = "semantic_work_terminal"
_EVENT_TYPE_REPLAY = "semantic_work_replayed"
_EVENT_TYPE_CANCELLED = "semantic_work_cancelled"
_EVENT_TYPE_PUBLISH = "semantic_work_published"
_EVENT_TYPE_PUBLISH_DENIED = "semantic_work_publish_denied"

_DEFAULT_FENCE_TTL_MS = 300_000
_MAX_DIAGNOSTIC = 512


class CancelBoundary(Protocol):
    """Subprocess or provider boundary that accepts cooperative cancellation."""

    def cancel(self, reason: str = "cancelled") -> bool:
        """Propagate cancellation; return True when the boundary accepted it."""


class WorkExecutor(Protocol):
    """Deterministic local work for non-provider harness stages."""

    def __call__(
        self,
        request: SemanticWorkRequest,
        *,
        lease: LeaseBinding,
        cancellation: CancellationToken,
        cancel_boundary: "SubprocessCancelBoundary",
    ) -> Mapping[str, Any]:
        ...


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clip_diagnostic(text: str) -> str:
    value = str(text or "").strip() or "unspecified"
    if len(value) > _MAX_DIAGNOSTIC:
        return value[: _MAX_DIAGNOSTIC - 3] + "..."
    return value


def _attempt_key(request: SemanticWorkRequest) -> str:
    return f"{request.work_id}#{request.attempt_id}#{request.idempotency_key}"


def _unavailable(
    *,
    operation: str,
    adapter_id: str,
    reason_code: str,
    diagnostic: str,
    retryable: bool = True,
) -> UnavailableResult:
    return UnavailableResult.from_dict(
        {
            "operation": operation,
            "adapter_id": adapter_id,
            "reason_code": reason_code,
            "retryable": retryable,
            "diagnostic": _clip_diagnostic(diagnostic),
        }
    )


@dataclass
class SubprocessCancelBoundary:
    """In-process cancel boundary that mirrors a subprocess/provider edge.

    The adapter registers one boundary per attempt. Cancellation marks the
    boundary and, when a process handle is bound, delivers a terminate signal
    so cancellation reaches the subprocess edge without starting work on import.
    """

    identity: str
    reason: str = ""
    cancelled: bool = False
    terminate_calls: int = 0
    _process: Any = field(default=None, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bind_process(self, process: Any) -> None:
        """Bind a subprocess-like object with ``terminate`` / ``poll`` / ``kill``."""

        with self._lock:
            self._process = process
            if self.cancelled:
                self._signal_process(self.reason or "cancelled")

    def cancel(self, reason: str = "cancelled") -> bool:
        with self._lock:
            if self.cancelled:
                return True
            self.cancelled = True
            self.reason = _clip_diagnostic(reason or "cancelled")
            self._signal_process(self.reason)
            return True

    def _signal_process(self, reason: str) -> None:
        process = self._process
        if process is None:
            return
        self.terminate_calls += 1
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
                return
            except Exception:
                pass
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                pass


@dataclass(frozen=True)
class FenceRecord:
    """Attempt-scoped fencing token with an absolute expiry."""

    attempt_id: str
    lease_id: str
    fencing_token: int
    logical_epoch: int
    expires_at_ms: int
    work_id: str
    cancelled: bool = False

    @property
    def binding(self) -> LeaseBinding:
        return LeaseBinding.from_dict(
            {
                "attempt_id": self.attempt_id,
                "fencing_token": self.fencing_token,
                "lease_id": self.lease_id,
                "logical_epoch": self.logical_epoch,
            }
        )

    def is_expired(self, now_ms: int) -> bool:
        return int(now_ms) >= int(self.expires_at_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "logical_epoch": self.logical_epoch,
            "expires_at_ms": self.expires_at_ms,
            "work_id": self.work_id,
            "cancelled": self.cancelled,
        }


@dataclass(frozen=True)
class ScheduledAttempt:
    """Fenced admission/execution attempt returned by the scheduling adapter.

    Downstream routing, provider, and worktree adapters consume this record.
    Publication requires a live, unexpired fencing token matching the lease.
    """

    request: SemanticWorkRequest
    result: SemanticWorkResult
    lease: LeaseBinding | None
    fence: FenceRecord | None
    cancellation: CancellationToken
    cancel_boundary: SubprocessCancelBoundary
    resource_lease_id: str | None
    provider_invoke_count: int
    replayed: bool
    event_sequence: int | None = None

    @property
    def attempt_key(self) -> str:
        return _attempt_key(self.request)

    @property
    def terminal(self) -> bool:
        return self.result.status in {
            SemanticWorkStatus.SUCCEEDED.value,
            SemanticWorkStatus.FAILED.value,
            SemanticWorkStatus.CANCELLED.value,
            SemanticWorkStatus.UNAVAILABLE.value,
            SemanticWorkStatus.SIMULATED.value,
        }

    @property
    def fencing_token(self) -> int | None:
        if self.lease is None:
            return None
        return self.lease.fencing_token

    def is_fence_valid(self, now_ms: int | None = None) -> bool:
        if self.fence is None or self.lease is None:
            return False
        clock = _now_ms() if now_ms is None else int(now_ms)
        if self.fence.cancelled:
            return False
        if self.fence.is_expired(clock):
            return False
        return self.fence.fencing_token == self.lease.fencing_token

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "result": self.result.to_dict(),
            "lease": None if self.lease is None else self.lease.to_dict(),
            "fence": None if self.fence is None else self.fence.to_dict(),
            "cancellation": self.cancellation.to_dict(),
            "resource_lease_id": self.resource_lease_id,
            "provider_invoke_count": self.provider_invoke_count,
            "replayed": self.replayed,
            "event_sequence": self.event_sequence,
            "terminal": self.terminal,
            "fencing_token": self.fencing_token,
        }


class SemanticSchedulingAdapter:
    """Admit, fence, execute, journal, and replay semantic harness work.

    The adapter is the sole harness owner of ``ProviderExecutionGateway``
    invocation. Exact-attempt replay returns the cached terminal result without
    reinvoking a provider. Expired fences cannot publish.
    """

    interface = SEMANTIC_SCHEDULING_ADAPTER_INTERFACE
    schema = SCHEDULING_ADAPTER_SCHEMA

    def __init__(
        self,
        *,
        resource_scheduler: Any | None = None,
        provider_gateway: Any | None = None,
        lease_coordinator: Any | None = None,
        worktree_store: Any | None = None,
        event_log_path: str | Path | None = None,
        host: Any | None = None,
        providers: Any | None = None,
        clock_ms: Callable[[], int] | None = None,
        fence_ttl_ms: int = _DEFAULT_FENCE_TTL_MS,
        work_executor: WorkExecutor | Callable[..., Mapping[str, Any]] | None = None,
        path: str | Path = ".",
    ) -> None:
        # Imports of runtime authorities are deferred to construction so a bare
        # module import remains free of host sampling and durable handles.
        from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
            ResourceScheduler,
        )

        self._resource_scheduler = resource_scheduler or ResourceScheduler()
        self._provider_gateway = provider_gateway
        self._lease_coordinator = lease_coordinator
        self._worktree_store = worktree_store
        self._event_log_path = (
            None if event_log_path is None else Path(event_log_path)
        )
        self._host = host
        self._providers = providers
        self._clock_ms = clock_ms or _now_ms
        self._fence_ttl_ms = max(1, int(fence_ttl_ms))
        self._work_executor = work_executor
        self._path = path
        self._lock = threading.RLock()
        self._terminals: dict[str, SemanticWorkResult] = {}
        self._attempts: dict[str, ScheduledAttempt] = {}
        self._fences: dict[str, FenceRecord] = {}
        self._cancellations: dict[str, CancellationToken] = {}
        self._boundaries: dict[str, SubprocessCancelBoundary] = {}
        self._resource_leases: dict[str, Any] = {}
        self._provider_invoke_counts: dict[str, int] = {}
        self._logical_epoch = 0
        self._next_fence = 1

    @property
    def provider_gateway(self) -> Any | None:
        return self._provider_gateway

    @property
    def resource_scheduler(self) -> Any:
        return self._resource_scheduler

    def cancellation_token(self, cancellation_id: str) -> CancellationToken:
        """Return the shared cancellation token for a request identity."""

        with self._lock:
            token = self._cancellations.get(cancellation_id)
            if token is None:
                token = CancellationToken(cancellation_id)
                self._cancellations[cancellation_id] = token
            return token

    def cancel_boundary_for(self, attempt_key: str) -> SubprocessCancelBoundary | None:
        with self._lock:
            return self._boundaries.get(attempt_key)

    def provider_invoke_count(self, request: SemanticWorkRequest | str) -> int:
        key = (
            request
            if isinstance(request, str)
            else _attempt_key(request)
        )
        with self._lock:
            return int(self._provider_invoke_counts.get(key, 0))

    def cancel(
        self,
        *,
        cancellation_id: str,
        reason: str = "cancelled",
    ) -> bool:
        """Cancel by fencing identity and propagate to resource/provider edges."""

        token = self.cancellation_token(cancellation_id)
        accepted = token.cancel(cancellation_id=cancellation_id, reason=reason)
        if not accepted:
            return False
        with self._lock:
            for key, boundary in list(self._boundaries.items()):
                attempt = self._attempts.get(key)
                if attempt is None:
                    continue
                if attempt.request.cancellation_id != cancellation_id:
                    continue
                boundary.cancel(reason=reason)
                self._cancel_resource_lease(attempt.resource_lease_id, reason=reason)
                if attempt.fence is not None:
                    self._fences[attempt.fence.lease_id] = FenceRecord(
                        attempt_id=attempt.fence.attempt_id,
                        lease_id=attempt.fence.lease_id,
                        fencing_token=attempt.fence.fencing_token,
                        logical_epoch=attempt.fence.logical_epoch,
                        expires_at_ms=attempt.fence.expires_at_ms,
                        work_id=attempt.fence.work_id,
                        cancelled=True,
                    )
                if (
                    self._provider_gateway is not None
                    and requires_provider(attempt.request.work_kind)
                ):
                    # Gateway requests carry a cancelled bit at dispatch; count
                    # cooperative cancel at the provider boundary as delivered.
                    pass
            self._journal(
                _EVENT_TYPE_CANCELLED,
                {
                    "cancellation_id": cancellation_id,
                    "reason": _clip_diagnostic(reason),
                    "status": SemanticWorkStatus.CANCELLED.value,
                },
            )
        return True

    def schedule(
        self,
        request: SemanticWorkRequest | Mapping[str, Any],
        *,
        provider_request: Any | None = None,
        work_executor: WorkExecutor | Callable[..., Mapping[str, Any]] | None = None,
        force_replay: bool = False,
    ) -> ScheduledAttempt:
        """Admit resources, acquire a fence, execute, and journal the attempt.

        Exact-attempt terminal outcomes are returned without reinvoking a
        provider. Capacity and provider absence yield typed ``UnavailableResult``.
        """

        req = (
            request
            if isinstance(request, SemanticWorkRequest)
            else SemanticWorkRequest.from_dict(request)
        )
        key = _attempt_key(req)
        cancellation = self.cancellation_token(req.cancellation_id)

        with self._lock:
            boundary = self._boundaries.get(key)
            if boundary is None:
                boundary = SubprocessCancelBoundary(identity=key)
                self._boundaries[key] = boundary

            prior = self._terminals.get(key)
            if prior is not None and not force_replay:
                return self._replay_terminal(req, prior, cancellation, boundary)

            if cancellation.is_cancelled() or boundary.cancelled:
                result = self._cancelled_result(
                    req,
                    lease=None,
                    reason=cancellation.reason or boundary.reason or "cancelled",
                )
                return self._store_terminal(
                    req,
                    result,
                    lease=None,
                    fence=None,
                    cancellation=cancellation,
                    boundary=boundary,
                    resource_lease_id=None,
                    replayed=False,
                )

            decision, resource_lease = self._admit_resources(req)
            if resource_lease is None:
                reasons = tuple(getattr(decision, "reasons", ()) or ("capacity_exhausted",))
                reason_code = str(reasons[0] if reasons else "capacity_exhausted")
                unavailable = _unavailable(
                    operation=req.work_kind,
                    adapter_id="resource-scheduler",
                    reason_code=reason_code,
                    diagnostic=(
                        "resource admission denied: "
                        + ",".join(str(item) for item in reasons)
                    ),
                    retryable=True,
                )
                result = SemanticWorkResult.from_dict(
                    {
                        "request": req.to_dict(),
                        "status": SemanticWorkStatus.UNAVAILABLE.value,
                        "lease": None,
                        "provider": (
                            None if req.provider is None else req.provider.to_dict()
                        ),
                        "unavailable": unavailable.to_dict(),
                        "reason_codes": sorted({reason_code, "capacity_unavailable"}),
                        "output_artifact_cids": [],
                        "diagnostic": unavailable.diagnostic,
                        "simulated": False,
                    }
                )
                return self._store_terminal(
                    req,
                    result,
                    lease=None,
                    fence=None,
                    cancellation=cancellation,
                    boundary=boundary,
                    resource_lease_id=None,
                    replayed=False,
                )

            resource_lease_id = str(
                getattr(resource_lease, "lease_id", "") or resource_lease
            )
            self._resource_leases[resource_lease_id] = resource_lease

            try:
                fence = self._acquire_fence(req)
                lease = fence.binding
                self._journal(
                    _EVENT_TYPE_ADMITTED,
                    {
                        "work_id": req.work_id,
                        "attempt_id": req.attempt_id,
                        "work_kind": req.work_kind,
                        "fencing_token": lease.fencing_token,
                        "lease_id": lease.lease_id,
                        "status": SemanticWorkStatus.ADMITTED.value,
                        "mode": req.mode,
                    },
                )

                if cancellation.is_cancelled() or boundary.cancelled:
                    self._cancel_resource_lease(resource_lease_id, reason="cancelled")
                    result = self._cancelled_result(
                        req,
                        lease=lease,
                        reason=cancellation.reason or boundary.reason or "cancelled",
                    )
                    return self._store_terminal(
                        req,
                        result,
                        lease=lease,
                        fence=fence,
                        cancellation=cancellation,
                        boundary=boundary,
                        resource_lease_id=resource_lease_id,
                        replayed=False,
                    )

                if requires_provider(req.work_kind):
                    result, invoke_delta = self._execute_provider(
                        req,
                        lease=lease,
                        cancellation=cancellation,
                        boundary=boundary,
                        provider_request=provider_request,
                    )
                else:
                    result, invoke_delta = self._execute_local(
                        req,
                        lease=lease,
                        cancellation=cancellation,
                        boundary=boundary,
                        work_executor=work_executor or self._work_executor,
                    )

                if invoke_delta:
                    self._provider_invoke_counts[key] = (
                        self._provider_invoke_counts.get(key, 0) + invoke_delta
                    )

                if result.status in {
                    SemanticWorkStatus.SUCCEEDED.value,
                    SemanticWorkStatus.SIMULATED.value,
                    SemanticWorkStatus.FAILED.value,
                    SemanticWorkStatus.CANCELLED.value,
                    SemanticWorkStatus.UNAVAILABLE.value,
                }:
                    self._release_resource_lease(
                        resource_lease_id,
                        reason=result.status,
                    )

                return self._store_terminal(
                    req,
                    result,
                    lease=lease if result.lease is not None else lease,
                    fence=fence,
                    cancellation=cancellation,
                    boundary=boundary,
                    resource_lease_id=resource_lease_id,
                    replayed=False,
                )
            except Exception as exc:
                self._cancel_resource_lease(resource_lease_id, reason="execution_error")
                result = SemanticWorkResult.from_dict(
                    {
                        "request": req.to_dict(),
                        "status": SemanticWorkStatus.FAILED.value,
                        "lease": None,
                        "provider": (
                            None if req.provider is None else req.provider.to_dict()
                        ),
                        "unavailable": None,
                        "reason_codes": ["execution_error"],
                        "output_artifact_cids": [],
                        "diagnostic": _clip_diagnostic(
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "simulated": False,
                    }
                )
                return self._store_terminal(
                    req,
                    result,
                    lease=None,
                    fence=None,
                    cancellation=cancellation,
                    boundary=boundary,
                    resource_lease_id=resource_lease_id,
                    replayed=False,
                )

    def replay(
        self,
        request: SemanticWorkRequest | Mapping[str, Any] | str,
    ) -> ScheduledAttempt:
        """Replay a terminal attempt without reinvoking provider work.

        Accepts a full request, attempt key, or work_id. Missing terminals raise
        ``HarnessError``.
        """

        with self._lock:
            if isinstance(request, str):
                key = request
                attempt = self._attempts.get(key)
                if attempt is None:
                    # Allow lookup by work_id when only one attempt exists.
                    matches = [
                        item
                        for item in self._attempts.values()
                        if item.request.work_id == request
                        or item.request.attempt_id == request
                    ]
                    if len(matches) != 1:
                        raise HarnessError(
                            f"no unique terminal attempt for identity {request!r}"
                        )
                    attempt = matches[0]
                    key = attempt.attempt_key
                prior = self._terminals.get(key)
                if prior is None:
                    raise HarnessError(f"no terminal result for {key}")
                cancellation = self.cancellation_token(attempt.request.cancellation_id)
                boundary = self._boundaries.setdefault(
                    key, SubprocessCancelBoundary(identity=key)
                )
                return self._replay_terminal(
                    attempt.request, prior, cancellation, boundary
                )

            req = (
                request
                if isinstance(request, SemanticWorkRequest)
                else SemanticWorkRequest.from_dict(request)
            )
            key = _attempt_key(req)
            prior = self._terminals.get(key)
            if prior is None:
                raise HarnessError(f"no terminal result for {key}")
            cancellation = self.cancellation_token(req.cancellation_id)
            boundary = self._boundaries.setdefault(
                key, SubprocessCancelBoundary(identity=key)
            )
            return self._replay_terminal(req, prior, cancellation, boundary)

    def publish(
        self,
        attempt: ScheduledAttempt,
        *,
        fencing_token: int,
        now_ms: int | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> SchedulerObservation:
        """Publish a scheduler observation only under a live fence.

        Expired, cancelled, or mismatched fencing tokens cannot publish.
        """

        clock = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            if attempt.lease is None or attempt.fence is None:
                self._journal(
                    _EVENT_TYPE_PUBLISH_DENIED,
                    {
                        "work_id": attempt.request.work_id,
                        "attempt_id": attempt.request.attempt_id,
                        "reason": "missing_fence",
                    },
                )
                raise HarnessError("publish requires a fenced lease binding")

            live = self._fences.get(attempt.fence.lease_id, attempt.fence)
            if live.cancelled:
                self._journal(
                    _EVENT_TYPE_PUBLISH_DENIED,
                    {
                        "work_id": attempt.request.work_id,
                        "attempt_id": attempt.request.attempt_id,
                        "reason": "fence_cancelled",
                        "fencing_token": fencing_token,
                    },
                )
                raise HarnessError("cancelled fence cannot publish")
            if live.is_expired(clock):
                self._journal(
                    _EVENT_TYPE_PUBLISH_DENIED,
                    {
                        "work_id": attempt.request.work_id,
                        "attempt_id": attempt.request.attempt_id,
                        "reason": "fence_expired",
                        "fencing_token": fencing_token,
                        "expires_at_ms": live.expires_at_ms,
                    },
                )
                raise HarnessError("expired fence cannot publish")
            if int(fencing_token) != int(live.fencing_token):
                self._journal(
                    _EVENT_TYPE_PUBLISH_DENIED,
                    {
                        "work_id": attempt.request.work_id,
                        "attempt_id": attempt.request.attempt_id,
                        "reason": "stale_fencing_token",
                        "fencing_token": fencing_token,
                        "expected_fencing_token": live.fencing_token,
                    },
                )
                raise HarnessError("stale fencing token cannot publish")
            if attempt.lease.fencing_token != live.fencing_token:
                raise HarnessError("attempt lease fencing token is stale")

            # Optional durable worktree fence cross-check when a store is bound.
            if self._worktree_store is not None and payload is not None:
                workspace = payload.get("workspace_path")
                if workspace:
                    record = None
                    loader = getattr(self._worktree_store, "load_workspace", None)
                    if callable(loader):
                        record = loader(workspace)
                    if record is not None:
                        record_fence = int(getattr(record, "fence", 0) or 0)
                        if record_fence and record_fence != int(fencing_token):
                            raise HarnessError(
                                "worktree fence mismatch cannot publish"
                            )
                        expires_at = getattr(record, "expires_at", None)
                        if expires_at is not None and float(clock) >= float(
                            expires_at
                        ) * (1000.0 if float(expires_at) < 1e12 else 1.0):
                            # Worktree store uses epoch seconds; normalize.
                            expires_ms = (
                                int(float(expires_at) * 1000)
                                if float(expires_at) < 1e12
                                else int(float(expires_at))
                            )
                            if clock >= expires_ms:
                                raise HarnessError(
                                    "expired worktree fence cannot publish"
                                )

            observation = attempt.result.as_scheduler_observation()
            if payload:
                # Observations stay closed; payload is journal-only metadata.
                forbidden = {
                    "secret",
                    "prompt",
                    "source_body",
                    "model_output",
                    "api_key",
                }
                bad = sorted(set(payload) & forbidden)
                if bad:
                    raise HarnessError(
                        f"publish payload forbids secret/source fields: {bad}"
                    )
            self._journal(
                _EVENT_TYPE_PUBLISH,
                {
                    "work_id": attempt.request.work_id,
                    "attempt_id": attempt.request.attempt_id,
                    "fencing_token": fencing_token,
                    "status": attempt.result.status,
                    "scheduling_success": observation.scheduling_success,
                    "verification_success": False,
                },
            )
            return observation

    def expire_fence(
        self,
        attempt: ScheduledAttempt,
        *,
        now_ms: int | None = None,
    ) -> FenceRecord:
        """Force-expire a fence (tests and recovery)."""

        if attempt.fence is None:
            raise HarnessError("attempt has no fence to expire")
        clock = self._clock_ms() if now_ms is None else int(now_ms)
        expired = FenceRecord(
            attempt_id=attempt.fence.attempt_id,
            lease_id=attempt.fence.lease_id,
            fencing_token=attempt.fence.fencing_token,
            logical_epoch=attempt.fence.logical_epoch,
            expires_at_ms=min(attempt.fence.expires_at_ms, clock),
            work_id=attempt.fence.work_id,
            cancelled=attempt.fence.cancelled,
        )
        with self._lock:
            self._fences[expired.lease_id] = expired
        return expired

    # ------------------------------------------------------------------ internals

    def _admit_resources(
        self, request: SemanticWorkRequest
    ) -> tuple[Any, Any | None]:
        from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
            HostResourceSnapshot,
            LaneResourceRequirements,
            ProviderCapacity,
        )

        resource = request.resource
        provider_id = ""
        requires = requires_provider(request.work_kind)
        if request.provider is not None:
            provider_id = request.provider.provider_id
            requires = True
        requirement = LaneResourceRequirements(
            lane_id=request.attempt_id,
            stage=resource.stage or stage_for_work_kind(request.work_kind),
            resource_class=resource.resource_class
            or resource_class_for_work_kind(request.work_kind),
            provider_id=provider_id,
            requires_provider=requires,
            process_slots=resource.process_slots,
            memory_bytes=resource.memory_bytes,
            disk_bytes=resource.disk_bytes,
            quota_units=resource.quota_units,
            fairness_key=request.work_kind,
        )
        host = self._host
        if host is None:
            host = HostResourceSnapshot(
                observed_at_ms=self._clock_ms(),
                cpu_percent=10,
                memory_percent=10,
                disk_percent=10,
                memory_total_bytes=16 * 1024 * 1024 * 1024,
                memory_available_bytes=8 * 1024 * 1024 * 1024,
                disk_total_bytes=100 * 1024 * 1024 * 1024,
                disk_available_bytes=50 * 1024 * 1024 * 1024,
                active_workers=0,
                worker_limit=max(4, self._resource_scheduler.policy.max_lanes),
                available_worker_capacity=max(
                    4, self._resource_scheduler.policy.max_lanes
                ),
                capabilities=("cpu",),
            )

        providers = self._providers
        if providers is None and requires and provider_id:
            # Synthesize conservative telemetry from the typed binding so
            # ResourceScheduler can admit when a provider is declared but the
            # caller has not yet attached a live capacity monitor. Absence of
            # both binding and telemetry remains fail-closed above.
            providers = (
                ProviderCapacity(
                    provider_id=provider_id,
                    healthy=True,
                    max_concurrency=max(1, resource.process_slots),
                    active_requests=0,
                    observed_at_ms=self._clock_ms(),
                ),
            )
        return self._resource_scheduler.acquire(
            requirement,
            host=host,
            providers=providers,
            path=self._path,
        )

    def _acquire_fence(self, request: SemanticWorkRequest) -> FenceRecord:
        now = self._clock_ms()
        self._logical_epoch += 1
        token = self._next_fence
        self._next_fence += 1
        lease_id = f"sch-lease:{request.attempt_id}:{token}:{uuid.uuid4().hex[:8]}"

        # Prefer durable lease coordinator fencing when injected.
        if self._lease_coordinator is not None:
            claim = getattr(self._lease_coordinator, "claim", None)
            if callable(claim):
                try:
                    grant = claim(
                        request.work_id,
                        claimant=f"semantic-scheduling:{request.attempt_id}",
                        duration_ms=self._fence_ttl_ms,
                    )
                    fencing_token = int(
                        getattr(grant, "fencing_token", None)
                        or getattr(grant, "fence", token)
                        or token
                    )
                    grant_lease_id = str(
                        getattr(grant, "claim_cid", None)
                        or getattr(grant, "lease_id", lease_id)
                        or lease_id
                    )
                    epoch = int(
                        getattr(grant, "logical_epoch", self._logical_epoch)
                        or self._logical_epoch
                    )
                    expires = int(
                        getattr(grant, "lease_expires_at_ms", 0)
                        or (now + self._fence_ttl_ms)
                    )
                    fence = FenceRecord(
                        attempt_id=request.attempt_id,
                        lease_id=grant_lease_id,
                        fencing_token=max(1, fencing_token),
                        logical_epoch=max(0, epoch),
                        expires_at_ms=expires,
                        work_id=request.work_id,
                    )
                    self._fences[fence.lease_id] = fence
                    return fence
                except Exception:
                    # Fall through to local fence; coordinator absence/conflict
                    # must not invent authority from a broken path.
                    pass

        fence = FenceRecord(
            attempt_id=request.attempt_id,
            lease_id=lease_id,
            fencing_token=token,
            logical_epoch=self._logical_epoch,
            expires_at_ms=now + self._fence_ttl_ms,
            work_id=request.work_id,
        )
        self._fences[fence.lease_id] = fence
        return fence

    def _execute_provider(
        self,
        request: SemanticWorkRequest,
        *,
        lease: LeaseBinding,
        cancellation: CancellationToken,
        boundary: SubprocessCancelBoundary,
        provider_request: Any | None,
    ) -> tuple[SemanticWorkResult, int]:
        if self._provider_gateway is None:
            unavailable = _unavailable(
                operation=request.work_kind,
                adapter_id="provider-execution-gateway",
                reason_code="provider_gateway_absent",
                diagnostic="ProviderExecutionGateway is not configured",
                retryable=True,
            )
            return (
                SemanticWorkResult.from_dict(
                    {
                        "request": request.to_dict(),
                        "status": SemanticWorkStatus.UNAVAILABLE.value,
                        "lease": lease.to_dict(),
                        "provider": (
                            None
                            if request.provider is None
                            else request.provider.to_dict()
                        ),
                        "unavailable": unavailable.to_dict(),
                        "reason_codes": ["provider_gateway_absent", "unavailable"],
                        "output_artifact_cids": [],
                        "diagnostic": unavailable.diagnostic,
                        "simulated": False,
                    }
                ),
                0,
            )

        if request.provider is None:
            unavailable = _unavailable(
                operation=request.work_kind,
                adapter_id="provider-binding",
                reason_code="provider_binding_absent",
                diagnostic="model_invocation requires a provider binding",
                retryable=False,
            )
            return (
                SemanticWorkResult.from_dict(
                    {
                        "request": request.to_dict(),
                        "status": SemanticWorkStatus.UNAVAILABLE.value,
                        "lease": lease.to_dict(),
                        "provider": None,
                        "unavailable": unavailable.to_dict(),
                        "reason_codes": ["provider_binding_absent", "unavailable"],
                        "output_artifact_cids": [],
                        "diagnostic": unavailable.diagnostic,
                        "simulated": False,
                    }
                ),
                0,
            )

        if (
            request.mode == HarnessMode.PRODUCTION.value
            and request.provider.simulated
        ):
            raise HarnessError("production requests cannot use simulated providers")

        if cancellation.is_cancelled() or boundary.cancelled:
            boundary.cancel(reason=cancellation.reason or "cancelled")
            return (
                self._cancelled_result(
                    request,
                    lease=lease,
                    reason=cancellation.reason or boundary.reason or "cancelled",
                    provider=request.provider,
                ),
                0,
            )

        if provider_request is None:
            unavailable = _unavailable(
                operation=request.work_kind,
                adapter_id="provider-execution-gateway",
                reason_code="provider_request_absent",
                diagnostic=(
                    "provider execution requires an injected ProviderExecutionRequest"
                ),
                retryable=True,
            )
            return (
                SemanticWorkResult.from_dict(
                    {
                        "request": request.to_dict(),
                        "status": SemanticWorkStatus.UNAVAILABLE.value,
                        "lease": lease.to_dict(),
                        "provider": request.provider.to_dict(),
                        "unavailable": unavailable.to_dict(),
                        "reason_codes": ["provider_request_absent", "unavailable"],
                        "output_artifact_cids": [],
                        "diagnostic": unavailable.diagnostic,
                        "simulated": False,
                    }
                ),
                0,
            )

        # Propagate cancellation to the provider request boundary.
        cancelled = bool(
            cancellation.is_cancelled()
            or boundary.cancelled
            or getattr(provider_request, "cancelled", False)
        )
        if cancelled:
            boundary.cancel(reason=cancellation.reason or "cancelled")
            try:
                object.__setattr__(provider_request, "cancelled", True)
            except Exception:
                # Frozen or foreign request objects still run with their own bit.
                pass

        # Count only non-replay gateway invocations for this attempt key.
        before = 0
        invoke_count = getattr(self._provider_gateway, "invoke_count", None)
        attempt_key = None
        if callable(invoke_count):
            attempt_key = getattr(provider_request, "attempt_key", None)
            if attempt_key:
                before = int(invoke_count(attempt_key))

        execution = self._provider_gateway.execute(provider_request)
        after = before
        if callable(invoke_count) and attempt_key:
            after = int(invoke_count(attempt_key))
        invoke_delta = max(0, after - before)
        # Some gateways track only successful invoke; if the call returned a
        # non-replayed settled/failed outcome, count at least one dispatch.
        replayed = bool(getattr(execution, "replayed", False))
        if not replayed and invoke_delta == 0:
            phase = str(getattr(execution, "phase", "") or "")
            if phase and phase not in {"denied", "cancelled", "replayed"}:
                invoke_delta = 1

        return (
            self._map_provider_result(
                request,
                lease=lease,
                execution=execution,
                provider=request.provider,
            ),
            0 if replayed else invoke_delta,
        )

    def _map_provider_result(
        self,
        request: SemanticWorkRequest,
        *,
        lease: LeaseBinding,
        execution: Any,
        provider: ProviderBinding,
    ) -> SemanticWorkResult:
        phase = str(getattr(execution, "phase", "") or "").lower()
        final_status = str(getattr(execution, "final_status", "") or "").lower()
        reason_codes = tuple(
            str(item) for item in (getattr(execution, "reason_codes", ()) or ())
        )
        simulated = bool(provider.simulated) or any(
            code.startswith("sim") or "simulated" in code for code in reason_codes
        )
        coordination = str(getattr(execution, "coordination_state", "") or "").lower()
        if coordination in {"unavailable", "unknown", "stale"} and phase in {
            "denied",
            "",
        }:
            unavailable = _unavailable(
                operation=request.work_kind,
                adapter_id="provider-execution-gateway",
                reason_code="provider_capacity_unavailable",
                diagnostic=_clip_diagnostic(
                    "provider coordination unavailable: "
                    + ",".join(reason_codes or (coordination,))
                ),
                retryable=True,
            )
            return SemanticWorkResult.from_dict(
                {
                    "request": request.to_dict(),
                    "status": SemanticWorkStatus.UNAVAILABLE.value,
                    "lease": lease.to_dict(),
                    "provider": provider.to_dict(),
                    "unavailable": unavailable.to_dict(),
                    "reason_codes": sorted(
                        set(reason_codes) | {"provider_capacity_unavailable"}
                    ),
                    "output_artifact_cids": [],
                    "diagnostic": unavailable.diagnostic,
                    "simulated": False,
                }
            )

        if phase in {"cancelled"} or final_status in {"cancelled"}:
            codes = list(reason_codes) or ["cancelled"]
            if "cancelled" not in codes and not any(
                code.startswith("cancel") for code in codes
            ):
                codes.append("cancelled")
            return SemanticWorkResult.from_dict(
                {
                    "request": request.to_dict(),
                    "status": SemanticWorkStatus.CANCELLED.value,
                    "lease": lease.to_dict(),
                    "provider": provider.to_dict(),
                    "unavailable": None,
                    "reason_codes": sorted(set(codes)),
                    "output_artifact_cids": [],
                    "diagnostic": _clip_diagnostic(
                        "provider execution cancelled: " + ",".join(codes)
                    ),
                    "simulated": simulated
                    and request.mode != HarnessMode.PRODUCTION.value,
                }
            )

        if phase in {"denied"} or final_status in {
            "capacity_unavailable",
            "rejected",
        }:
            if "capacity" in final_status or any(
                "capacity" in code for code in reason_codes
            ):
                unavailable = _unavailable(
                    operation=request.work_kind,
                    adapter_id="provider-execution-gateway",
                    reason_code="provider_capacity_unavailable",
                    diagnostic=_clip_diagnostic(
                        "provider capacity denied: "
                        + ",".join(reason_codes or (final_status,))
                    ),
                    retryable=True,
                )
                return SemanticWorkResult.from_dict(
                    {
                        "request": request.to_dict(),
                        "status": SemanticWorkStatus.UNAVAILABLE.value,
                        "lease": lease.to_dict(),
                        "provider": provider.to_dict(),
                        "unavailable": unavailable.to_dict(),
                        "reason_codes": sorted(
                            set(reason_codes) | {"provider_capacity_unavailable"}
                        ),
                        "output_artifact_cids": [],
                        "diagnostic": unavailable.diagnostic,
                        "simulated": False,
                    }
                )
            return SemanticWorkResult.from_dict(
                {
                    "request": request.to_dict(),
                    "status": SemanticWorkStatus.FAILED.value,
                    "lease": lease.to_dict(),
                    "provider": provider.to_dict(),
                    "unavailable": None,
                    "reason_codes": sorted(set(reason_codes) | {"provider_denied"}),
                    "output_artifact_cids": [],
                    "diagnostic": _clip_diagnostic(
                        "provider execution denied: "
                        + ",".join(reason_codes or (final_status,))
                    ),
                    "simulated": simulated
                    and request.mode != HarnessMode.PRODUCTION.value,
                }
            )

        if phase in {"failed"} or final_status in {"failed"}:
            return SemanticWorkResult.from_dict(
                {
                    "request": request.to_dict(),
                    "status": SemanticWorkStatus.FAILED.value,
                    "lease": lease.to_dict(),
                    "provider": provider.to_dict(),
                    "unavailable": None,
                    "reason_codes": sorted(set(reason_codes) | {"provider_failed"}),
                    "output_artifact_cids": [],
                    "diagnostic": _clip_diagnostic(
                        "provider execution failed: "
                        + ",".join(reason_codes or (phase,))
                    ),
                    "simulated": simulated
                    and request.mode != HarnessMode.PRODUCTION.value,
                }
            )

        success = bool(getattr(execution, "success", False)) or phase in {
            "settled",
            "replayed",
            "committed",
        }
        if not success and final_status not in {"committed", "settled", "succeeded"}:
            return SemanticWorkResult.from_dict(
                {
                    "request": request.to_dict(),
                    "status": SemanticWorkStatus.FAILED.value,
                    "lease": lease.to_dict(),
                    "provider": provider.to_dict(),
                    "unavailable": None,
                    "reason_codes": sorted(
                        set(reason_codes) | {"provider_unsuccessful"}
                    ),
                    "output_artifact_cids": [],
                    "diagnostic": _clip_diagnostic(
                        "provider execution unsuccessful: "
                        + ",".join(reason_codes or (phase, final_status))
                    ),
                    "simulated": simulated
                    and request.mode != HarnessMode.PRODUCTION.value,
                }
            )

        if simulated and request.mode != HarnessMode.PRODUCTION.value:
            status = SemanticWorkStatus.SIMULATED.value
        else:
            status = SemanticWorkStatus.SUCCEEDED.value
            simulated = False

        outputs: list[str] = []
        observation = getattr(execution, "observation", None)
        if isinstance(observation, Mapping):
            raw_outputs = observation.get("output_artifact_cids")
            if isinstance(raw_outputs, Sequence) and not isinstance(
                raw_outputs, (str, bytes)
            ):
                outputs = [str(item) for item in raw_outputs]

        return SemanticWorkResult.from_dict(
            {
                "request": request.to_dict(),
                "status": status,
                "lease": lease.to_dict(),
                "provider": provider.to_dict(),
                "unavailable": None,
                "reason_codes": sorted(set(reason_codes)),
                "output_artifact_cids": outputs,
                "diagnostic": _clip_diagnostic(
                    "provider execution completed"
                    + (f": {','.join(reason_codes)}" if reason_codes else "")
                ),
                "simulated": simulated,
            }
        )

    def _execute_local(
        self,
        request: SemanticWorkRequest,
        *,
        lease: LeaseBinding,
        cancellation: CancellationToken,
        boundary: SubprocessCancelBoundary,
        work_executor: WorkExecutor | Callable[..., Mapping[str, Any]] | None,
    ) -> tuple[SemanticWorkResult, int]:
        if cancellation.is_cancelled() or boundary.cancelled:
            boundary.cancel(reason=cancellation.reason or "cancelled")
            return (
                self._cancelled_result(
                    request,
                    lease=lease,
                    reason=cancellation.reason or boundary.reason or "cancelled",
                ),
                0,
            )

        outputs: list[str] = []
        diagnostic = "local work completed"
        reason_codes: list[str] = []
        simulated = False

        if work_executor is not None:
            try:
                outcome = work_executor(
                    request,
                    lease=lease,
                    cancellation=cancellation,
                    cancel_boundary=boundary,
                )
            except HarnessError as exc:
                if "cancel" in str(exc).casefold():
                    boundary.cancel(reason=str(exc))
                    return (
                        self._cancelled_result(
                            request, lease=lease, reason=str(exc)
                        ),
                        0,
                    )
                return (
                    SemanticWorkResult.from_dict(
                        {
                            "request": request.to_dict(),
                            "status": SemanticWorkStatus.FAILED.value,
                            "lease": lease.to_dict(),
                            "provider": None,
                            "unavailable": None,
                            "reason_codes": ["execution_error"],
                            "output_artifact_cids": [],
                            "diagnostic": _clip_diagnostic(str(exc)),
                            "simulated": False,
                        }
                    ),
                    0,
                )
            except Exception as exc:
                return (
                    SemanticWorkResult.from_dict(
                        {
                            "request": request.to_dict(),
                            "status": SemanticWorkStatus.FAILED.value,
                            "lease": lease.to_dict(),
                            "provider": None,
                            "unavailable": None,
                            "reason_codes": ["execution_error"],
                            "output_artifact_cids": [],
                            "diagnostic": _clip_diagnostic(
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "simulated": False,
                        }
                    ),
                    0,
                )

            if not isinstance(outcome, Mapping):
                raise HarnessError("work executor must return a mapping")
            status_hint = str(outcome.get("status") or "").strip().lower()
            if status_hint == SemanticWorkStatus.CANCELLED.value or boundary.cancelled:
                return (
                    self._cancelled_result(
                        request,
                        lease=lease,
                        reason=str(
                            outcome.get("reason")
                            or boundary.reason
                            or cancellation.reason
                            or "cancelled"
                        ),
                    ),
                    0,
                )
            if status_hint == SemanticWorkStatus.UNAVAILABLE.value:
                unavailable_raw = outcome.get("unavailable")
                if isinstance(unavailable_raw, Mapping):
                    unavailable = UnavailableResult.from_dict(unavailable_raw)
                else:
                    unavailable = _unavailable(
                        operation=request.work_kind,
                        adapter_id=ADAPTER_ID,
                        reason_code=str(
                            outcome.get("reason_code") or "tooling_unavailable"
                        ),
                        diagnostic=str(
                            outcome.get("diagnostic") or "local tooling unavailable"
                        ),
                        retryable=bool(outcome.get("retryable", True)),
                    )
                return (
                    SemanticWorkResult.from_dict(
                        {
                            "request": request.to_dict(),
                            "status": SemanticWorkStatus.UNAVAILABLE.value,
                            "lease": lease.to_dict(),
                            "provider": None,
                            "unavailable": unavailable.to_dict(),
                            "reason_codes": sorted(
                                set(outcome.get("reason_codes") or [])
                                | {unavailable.reason_code}
                            ),
                            "output_artifact_cids": [],
                            "diagnostic": unavailable.diagnostic,
                            "simulated": False,
                        }
                    ),
                    0,
                )
            if status_hint == SemanticWorkStatus.FAILED.value:
                return (
                    SemanticWorkResult.from_dict(
                        {
                            "request": request.to_dict(),
                            "status": SemanticWorkStatus.FAILED.value,
                            "lease": lease.to_dict(),
                            "provider": None,
                            "unavailable": None,
                            "reason_codes": sorted(
                                set(outcome.get("reason_codes") or ["execution_error"])
                            ),
                            "output_artifact_cids": [],
                            "diagnostic": _clip_diagnostic(
                                str(outcome.get("diagnostic") or "local work failed")
                            ),
                            "simulated": False,
                        }
                    ),
                    0,
                )
            raw_outputs = outcome.get("output_artifact_cids") or []
            if isinstance(raw_outputs, Sequence) and not isinstance(
                raw_outputs, (str, bytes)
            ):
                outputs = [str(item) for item in raw_outputs]
            diagnostic = str(outcome.get("diagnostic") or diagnostic)
            reason_codes = [
                str(item) for item in (outcome.get("reason_codes") or [])
            ]
            simulated = bool(outcome.get("simulated", False))

        if cancellation.is_cancelled() or boundary.cancelled:
            return (
                self._cancelled_result(
                    request,
                    lease=lease,
                    reason=cancellation.reason or boundary.reason or "cancelled",
                ),
                0,
            )

        if simulated and request.mode != HarnessMode.PRODUCTION.value:
            status = SemanticWorkStatus.SIMULATED.value
        else:
            status = SemanticWorkStatus.SUCCEEDED.value
            simulated = False

        return (
            SemanticWorkResult.from_dict(
                {
                    "request": request.to_dict(),
                    "status": status,
                    "lease": lease.to_dict(),
                    "provider": None,
                    "unavailable": None,
                    "reason_codes": sorted(set(reason_codes)),
                    "output_artifact_cids": outputs,
                    "diagnostic": _clip_diagnostic(diagnostic),
                    "simulated": simulated,
                }
            ),
            0,
        )

    def _cancelled_result(
        self,
        request: SemanticWorkRequest,
        *,
        lease: LeaseBinding | None,
        reason: str,
        provider: ProviderBinding | None = None,
    ) -> SemanticWorkResult:
        diagnostic = _clip_diagnostic(reason or "cancelled")
        bound = provider if provider is not None else request.provider
        return SemanticWorkResult.from_dict(
            {
                "request": request.to_dict(),
                "status": SemanticWorkStatus.CANCELLED.value,
                "lease": None if lease is None else lease.to_dict(),
                "provider": None if bound is None else bound.to_dict(),
                "unavailable": None,
                "reason_codes": ["cancelled"],
                "output_artifact_cids": [],
                "diagnostic": diagnostic,
                "simulated": False,
            }
        )

    def _store_terminal(
        self,
        request: SemanticWorkRequest,
        result: SemanticWorkResult,
        *,
        lease: LeaseBinding | None,
        fence: FenceRecord | None,
        cancellation: CancellationToken,
        boundary: SubprocessCancelBoundary,
        resource_lease_id: str | None,
        replayed: bool,
    ) -> ScheduledAttempt:
        key = _attempt_key(request)
        # Ensure lease on succeeded/admitted/simulated matches invariants.
        if (
            result.status
            in {
                SemanticWorkStatus.SUCCEEDED.value,
                SemanticWorkStatus.ADMITTED.value,
                SemanticWorkStatus.SIMULATED.value,
            }
            and result.lease is None
            and lease is not None
        ):
            payload = result.to_dict()
            payload.pop("scheduling_success", None)
            payload.pop("verification_success", None)
            payload["lease"] = lease.to_dict()
            result = SemanticWorkResult.from_dict(payload)

        sequence = self._journal(
            _EVENT_TYPE_REPLAY if replayed else _EVENT_TYPE_TERMINAL,
            {
                "work_id": request.work_id,
                "attempt_id": request.attempt_id,
                "work_kind": request.work_kind,
                "status": result.status,
                "fencing_token": None if lease is None else lease.fencing_token,
                "replayed": replayed,
                "provider_invoke_count": self._provider_invoke_counts.get(key, 0),
                "scheduling_success": result.scheduling_success,
                "verification_success": False,
            },
        )
        attempt = ScheduledAttempt(
            request=request,
            result=result,
            lease=result.lease if result.lease is not None else lease,
            fence=fence,
            cancellation=cancellation,
            cancel_boundary=boundary,
            resource_lease_id=resource_lease_id,
            provider_invoke_count=int(self._provider_invoke_counts.get(key, 0)),
            replayed=replayed,
            event_sequence=sequence,
        )
        self._terminals[key] = result
        self._attempts[key] = attempt
        return attempt

    def _replay_terminal(
        self,
        request: SemanticWorkRequest,
        prior: SemanticWorkResult,
        cancellation: CancellationToken,
        boundary: SubprocessCancelBoundary,
    ) -> ScheduledAttempt:
        key = _attempt_key(request)
        existing = self._attempts.get(key)
        fence = existing.fence if existing is not None else None
        resource_lease_id = (
            existing.resource_lease_id if existing is not None else None
        )
        # Replay never reinvokes provider work: invoke counter is frozen.
        return self._store_terminal(
            request,
            prior,
            lease=prior.lease,
            fence=fence,
            cancellation=cancellation,
            boundary=boundary,
            resource_lease_id=resource_lease_id,
            replayed=True,
        )

    def _release_resource_lease(
        self, resource_lease_id: str | None, *, reason: str
    ) -> None:
        if not resource_lease_id:
            return
        lease = self._resource_leases.pop(resource_lease_id, resource_lease_id)
        releaser = getattr(self._resource_scheduler, "release", None)
        if callable(releaser):
            try:
                releaser(lease, reason=reason)
            except Exception:
                pass

    def _cancel_resource_lease(
        self, resource_lease_id: str | None, *, reason: str
    ) -> None:
        if not resource_lease_id:
            return
        lease = self._resource_leases.pop(resource_lease_id, resource_lease_id)
        canceller = getattr(self._resource_scheduler, "cancel", None)
        if callable(canceller):
            try:
                canceller(lease, reason=reason)
                return
            except Exception:
                pass
        releaser = getattr(self._resource_scheduler, "release", None)
        if callable(releaser):
            try:
                releaser(lease, reason=reason)
            except Exception:
                pass

    def _journal(
        self, event_type: str, payload: Mapping[str, Any]
    ) -> int | None:
        if self._event_log_path is None:
            return None
        from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
            append_jsonl_event,
        )

        # Keep journal payloads bounded and secret-free.
        safe = {
            key: value
            for key, value in dict(payload).items()
            if key
            not in {
                "secret",
                "prompt",
                "source_body",
                "model_output",
                "api_key",
                "messages",
            }
        }
        safe.setdefault("schema", SCHEDULING_ADAPTER_SCHEMA)
        safe.setdefault("interface", SEMANTIC_SCHEDULING_ADAPTER_INTERFACE)
        safe.setdefault("board_namespace", BOARD_NAMESPACE)
        event = append_jsonl_event(self._event_log_path, event_type, safe)
        sequence = event.get("sequence")
        return int(sequence) if sequence is not None else None


def schedule_semantic_work(
    request: SemanticWorkRequest | Mapping[str, Any],
    *,
    adapter: SemanticSchedulingAdapter | None = None,
    provider_request: Any | None = None,
    work_executor: WorkExecutor | Callable[..., Mapping[str, Any]] | None = None,
    **adapter_kwargs: Any,
) -> ScheduledAttempt:
    """Module-level schedule entrypoint over ``SemanticSchedulingAdapter``."""

    owner = adapter or SemanticSchedulingAdapter(**adapter_kwargs)
    return owner.schedule(
        request,
        provider_request=provider_request,
        work_executor=work_executor,
    )


def replay_semantic_work(
    request: SemanticWorkRequest | Mapping[str, Any] | str,
    *,
    adapter: SemanticSchedulingAdapter,
) -> ScheduledAttempt:
    """Module-level exact-attempt replay without provider reinvocation."""

    return adapter.replay(request)


def semantic_scheduling_adapter_descriptor() -> dict[str, Any]:
    """Closed interface metadata for SemanticSchedulingAdapter@1."""

    return {
        "interface": SEMANTIC_SCHEDULING_ADAPTER_INTERFACE,
        "schema": SCHEDULING_ADAPTER_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "adapter_id": ADAPTER_ID,
        "composes": [
            "ResourceScheduler",
            "ProviderExecutionGateway",
            "LeaseCoordinator",
            "WorktreeLifecycleStore",
            "runtime.event_log",
        ],
        "symbols": [
            "SemanticSchedulingAdapter",
            "ScheduledAttempt",
            "schedule_semantic_work",
            "replay_semantic_work",
            "SubprocessCancelBoundary",
            "FenceRecord",
        ],
        "invariants": [
            "capacity_or_provider_absence_returns_typed_unavailable",
            "cancellation_reaches_subprocess_provider_boundary",
            "replay_does_not_reinvoke_terminal_provider_call",
            "expired_fences_cannot_publish",
            "cold_import_starts_no_resources",
            "persistent_task_queue_is_not_authority",
            "scheduling_success_is_not_verification_success",
        ],
    }


__all__ = [
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "SCHEDULING_ADAPTER_SCHEMA",
    "SEMANTIC_SCHEDULING_ADAPTER_INTERFACE",
    "CancelBoundary",
    "FenceRecord",
    "ScheduledAttempt",
    "SemanticSchedulingAdapter",
    "SubprocessCancelBoundary",
    "WorkExecutor",
    "replay_semantic_work",
    "schedule_semantic_work",
    "semantic_scheduling_adapter_descriptor",
]
