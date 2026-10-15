"""Logical attribution for a configured physical-batch quota owner.

This bridge owns submission and validates native scheduler evidence followed by
the authoritative ledger record. It never reserves or settles quota itself.
The operator must supply a callback making one physical provider request per
batch; generic scalar fan-out routers do not meet that contract.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import json
import threading
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.endpoint_usage.identity import canonical_json, stable_id
from ipfs_accelerate_py.endpoint_usage.schema import EndpointUsageScope, QuantityKind, UsageVector

from .direct_provider_usage import _selector
from .provider_batch_scheduler import ProviderBatchRequest, ProviderBatchResult, ProviderBatchScheduler
from .provider_batch_usage import PhysicalBatchUsageAdmission
from .provider_usage_migration import ConsumerId


_ATTRIBUTION_SEAL = object()
_SCHEMA = "ipfs_accelerate_py.agent_supervisor.batch_usage_attribution@1"


@dataclass(frozen=True)
class _AttributionIssuer:
    content_id: str
    seal: object = field(repr=False)


class BatchProviderUsageError(RuntimeError):
    def __init__(self, message: str, *, reason_codes: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reason_codes = tuple(reason_codes)


def _fail(message: str, reason: str) -> None:
    raise BatchProviderUsageError(message, reason_codes=(reason,))


def _bounded(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value.encode("utf-8")) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        _fail(f"{name} must be bounded nonempty text", "batch_usage_binding_invalid")
    return value


@dataclass(frozen=True)
class BatchUsageAttribution:
    """An observational reference to a charge, not another usage receipt.

    Serialized projections carry no authorization. The private local marker
    allows the aggregation helper to distinguish this bridge's validated output
    from caller-constructed objects. Original prompt/output text is not retained.
    """

    consumer_id: ConsumerId
    logical_request_id: str
    execution_fingerprint: str
    status: str
    provider_id: str
    batch_id: str
    execution_id: str
    batch_receipt_id: str
    singleflight_shared: bool
    usage_state: str
    physical_charge_id: str = ""
    _physical_json: str = field(default="{}", repr=False)
    _producer: object | None = field(default=None, repr=False, compare=False)

    @property
    def physical_usage(self) -> dict[str, Any]:
        return json.loads(self._physical_json)

    @property
    def additional_physical_charge(self) -> int:
        return 0

    def to_dict(self) -> dict[str, Any]:
        fields = {
            "schema": _SCHEMA, "consumer_id": self.consumer_id.value,
            "logical_request_id": self.logical_request_id,
            "execution_fingerprint": self.execution_fingerprint, "status": self.status,
            "provider_id": self.provider_id, "batch_id": self.batch_id,
            "execution_id": self.execution_id, "batch_receipt_id": self.batch_receipt_id,
            "singleflight_shared": self.singleflight_shared, "usage_state": self.usage_state,
            "physical_charge_id": self.physical_charge_id, "physical_usage": self.physical_usage,
            "additional_physical_charge": 0, "authorizes_execution": False,
        }
        return {**fields, "content_id": stable_id("batchattrib", fields)}


@dataclass(frozen=True, kw_only=True)
class BatchProviderUsageBinding:
    """Explicit ownership for immutable text submitted to one native scheduler.

    Endpoint scope operation is ``text.generate``; ``operation`` and ``route``
    name the consumer's exact logical batch contract. The configured callback
    must preserve per-member output bounds within its one-request batch format.
    ``single_physical_request`` is an explicit trusted deployment declaration;
    it cannot inspect or certify arbitrary callback internals.
    Live owners are deliberately absent from serialized attribution.
    """

    scheduler: ProviderBatchScheduler = field(repr=False, compare=False)
    admission: PhysicalBatchUsageAdmission = field(repr=False, compare=False)
    scope: EndpointUsageScope
    provider_id: str
    model: str
    route: str
    operation: str
    fence: int
    single_physical_request: bool = False
    attribution_limit: int = 128
    _dispatch_owner: Any = field(init=False, repr=False, compare=False)
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False, compare=False)
    _attributions: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler, ProviderBatchScheduler):
            _fail("binding requires a native ProviderBatchScheduler", "batch_usage_owner_required")
        if not isinstance(self.admission, PhysicalBatchUsageAdmission):
            _fail("binding requires PhysicalBatchUsageAdmission", "batch_usage_owner_required")
        if self.single_physical_request is not True:
            _fail("binding requires an explicit single physical request declaration",
                  "batch_usage_physical_contract_required")
        if not isinstance(self.scope, EndpointUsageScope) or self.scope.unknown_scope:
            _fail("binding requires an established endpoint scope", "batch_usage_scope_invalid")
        if self.scope.operation != "text.generate":
            _fail("physical endpoint operation must be text.generate", "batch_usage_scope_invalid")
        _selector(self.provider_id, "provider_id")
        _selector(self.model, "model")
        _bounded(self.route, "route")
        _bounded(self.operation, "operation")
        if type(self.fence) is not int or self.fence <= 0:
            _fail("binding requires an explicit positive fence", "batch_usage_fence_invalid")
        if type(self.attribution_limit) is not int or not 1 <= self.attribution_limit <= 4096:
            _fail("attribution_limit must be between 1 and 4096", "batch_usage_bounds_invalid")
        object.__setattr__(self, "_dispatch_owner", self.scheduler.dispatch_callback_for(self.provider_id))
        self._check_owner()
        # Pin before the first submission so registration cannot swap the
        # declared physical callback while this consumer waits in the queue.
        self.scheduler.pin_dispatch_callback(self.provider_id, self._dispatch_owner)

    def _check_owner(self) -> None:
        if self.scheduler.admission is not self.admission:
            _fail("scheduler has a different physical quota owner", "batch_usage_owner_mismatch")
        if self.admission.require_hard_limits is not True:
            _fail("physical owner must require configured hard request and token limits",
                  "batch_usage_hard_limits_required")
        dispatch = self.scheduler.dispatch_callback_for(self.provider_id)
        if dispatch is not self._dispatch_owner:
            _fail("scheduler dispatch callback differs from the configured owner",
                  "batch_usage_owner_mismatch")
        if getattr(dispatch, "_provider_batch_may_fan_out", False):
            _fail("generic fan-out dispatch cannot own one aggregate physical charge",
                  "batch_usage_fanout_forbidden")
        coordinator = self.admission.coordinator
        if (type(getattr(coordinator, "_fence", None)) is not int
                or coordinator._fence != self.fence
                or not getattr(coordinator.store, "authorizes_admission", False)):
            _fail("physical coordinator fence is not the configured fence", "batch_usage_fence_mismatch")
        try:
            scope_id = self.admission.scope_supplier(self.provider_id)
        except Exception as exc:
            raise BatchProviderUsageError("physical scope is unavailable",
                reason_codes=("batch_usage_scope_unavailable",)) from exc
        if scope_id != self.scope.scope_id:
            _fail("physical owner supplies a different endpoint scope", "batch_usage_scope_mismatch")

    def last_attribution(self, consumer_id: ConsumerId | str) -> BatchUsageAttribution | None:
        consumer = ConsumerId(consumer_id)
        with self._lock:
            return next((item for item in reversed(self._attributions.values())
                         if item.consumer_id is consumer), None)

    def attribution_for(self, request_id: str) -> BatchUsageAttribution | None:
        with self._lock:
            return self._attributions.get(request_id)

    def _remember(self, request, result, consumer, physical, usage_state) -> None:
        charge_id = (stable_id("physicalcharge", physical["scope_digest"], physical["reservation_id"])
                     if usage_state == "confirmed" else "")
        attribution = BatchUsageAttribution(
            consumer_id=consumer, logical_request_id=request.request_id,
            execution_fingerprint=request.execution_fingerprint, status=result.status.value,
            provider_id=result.provider_id, batch_id=result.batch_id, execution_id=result.execution_id,
            batch_receipt_id=result.receipt_id, singleflight_shared=result.singleflight_shared,
            usage_state=usage_state, physical_charge_id=charge_id,
            _physical_json=canonical_json(physical),
        )
        attribution = replace(attribution, _producer=_AttributionIssuer(
            attribution.to_dict()["content_id"], _ATTRIBUTION_SEAL,
        ))
        with self._lock:
            self._attributions[request.request_id] = attribution
            self._attributions.move_to_end(request.request_id)
            while len(self._attributions) > self.attribution_limit:
                self._attributions.popitem(last=False)

    def execute(self, request: ProviderBatchRequest, *, consumer_id: ConsumerId | str) -> ProviderBatchResult:
        self._check_owner()
        consumer = ConsumerId(consumer_id)
        if not isinstance(request, ProviderBatchRequest) or not isinstance(request.payload, str):
            _fail("bound batches require immutable text requests", "batch_usage_request_invalid")
        _bounded(request.request_id, "request_id")
        if (request.provider_id != self.provider_id or request.model != self.model
                or request.route != self.route or request.operation != self.operation):
            _fail("request differs from configured provider/model/route/operation", "batch_usage_binding_mismatch")
        if request.usage_scope_id not in ("", self.scope.scope_id):
            _fail("request carries a different usage scope", "batch_usage_scope_mismatch")
        if (request.context_limit < max(1, len(request.payload.encode("utf-8")))
                or request.token_budget <= 0 or request.timeout_ms <= 0
                or self.admission.reservation_ttl_ms <= request.timeout_ms + self.admission.settlement_margin_ms):
            _fail("request lacks sufficient bounded input/output or reservation lifetime", "batch_usage_bounds_invalid")
        for options in (request.policy, request.generation_settings):
            if options.get("allow_local_fallback") or options.get("allow_cross_provider_fallback"):
                _fail("bound batch dispatch cannot change providers", "batch_usage_fallback_forbidden")
            if options.get("disable_model_retry") is False:
                _fail("bound batch dispatch cannot retry models", "batch_usage_fallback_forbidden")
        # replace invokes the request constructor, detaching mutable mappings.
        # Cancellation is intentionally shared with its owning logical caller.
        captured = replace(request, usage_scope_id=self.scope.scope_id)
        result = self.scheduler.execute(captured)
        physical: Mapping[str, Any] = {}
        if result.receipt_id:
            try:
                self.scheduler.validate_result_receipt(captured, result)
            except Exception as exc:
                raise BatchProviderUsageError("batch accounting evidence requires reconciliation",
                    reason_codes=("batch_usage_receipt_invalid", "reconciliation_required")) from exc
            try:
                self._check_owner()
                if result.physical_usage:
                    physical = self.admission.validate_receipt(result.physical_usage, request=captured)
                self._check_owner()
            except Exception as exc:
                # Keep native owner evidence for reconciliation without
                # claiming confirmed units or returning successful output.
                self._remember(captured, result, consumer, result.physical_usage,
                               "reconciliation_required")
                raise BatchProviderUsageError("physical settlement requires reconciliation",
                    reason_codes=("batch_usage_receipt_invalid", "reconciliation_required")) from exc
        else:
            self._check_owner()
            if result.successful or result.physical_usage:
                _fail("batch result has no native producer receipt", "batch_usage_receipt_missing")
        if result.successful and (
            not physical or physical.get("state") != "committed"
            or physical.get("settlement_basis") not in {"estimated_ceiling", "provider_reported"}
        ):
            _fail("successful output lacks confirmed physical settlement", "batch_usage_unsettled")
        self._remember(captured, result, consumer, physical,
                       "confirmed" if physical else "not_observed")
        return result


def summarize_batch_usage(attributions: Sequence[BatchUsageAttribution]) -> dict[str, Any]:
    """Deduplicate validated local references without inventing a cost split.

    This observational summary neither authorizes work nor turns estimated
    ceilings into measured billing. Missing observations remain explicit.
    """
    logical: dict[tuple[str, str], str] = {}
    physical: dict[str, str] = {}
    settled: dict[str, int] = {}
    missing = 0
    for item in attributions:
        if (not isinstance(item, BatchUsageAttribution)
                or not isinstance(item._producer, _AttributionIssuer)
                or item._producer.seal is not _ATTRIBUTION_SEAL):
            _fail("aggregation requires validated local attributions", "batch_usage_attribution_invalid")
        key = (item.logical_request_id, item.batch_receipt_id)
        identity = item.to_dict()["content_id"]
        if item._producer.content_id != identity:
            _fail("attribution differs from its issued content", "batch_usage_attribution_invalid")
        if (item.usage_state == "confirmed") != bool(item.physical_charge_id):
            _fail("attribution settlement state is inconsistent", "batch_usage_attribution_invalid")
        if key in logical:
            if logical[key] != identity:
                _fail("logical attribution differs on replay", "batch_usage_attribution_conflict")
            continue
        logical[key] = identity
        if not item.physical_charge_id:
            missing += 1
            continue
        receipt = item.physical_usage
        encoded = canonical_json(receipt)
        if item.physical_charge_id in physical:
            if physical[item.physical_charge_id] != encoded:
                _fail("one physical charge has conflicting evidence", "batch_usage_attribution_conflict")
            continue
        physical[item.physical_charge_id] = encoded
        for entry in UsageVector.from_dict(receipt["settled"]).entries:
            if entry.amount.kind is not QuantityKind.FINITE:
                _fail("physical settlement must be finite", "batch_usage_attribution_invalid")
            dimension = entry.dimension.value
            if entry.currency:
                dimension += ":" + entry.currency
            settled[dimension] = settled.get(dimension, 0) + int(entry.amount.value)
    return {"logical_requests": len(logical), "physical_calls": len(physical),
            "unobserved_logical_requests": missing, "settled": settled,
            "additional_physical_charge": 0, "authorizes_execution": False}


__all__ = ["BatchProviderUsageBinding", "BatchProviderUsageError", "BatchUsageAttribution",
           "summarize_batch_usage"]
