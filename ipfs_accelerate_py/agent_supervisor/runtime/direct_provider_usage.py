"""Explicit deployment ownership for one bounded direct text-provider call.

The operator binds an exact router selector/model to an established quota scope.
No provider discovery, credentials, ledger creation, limit installation, global
registration, or provider invocation occurs on import or construction. Callers
must enforce their declared timeout and token ceilings and disable fallback and
inner quota owners. This adapter records conservative declared usage, not billing
evidence; batch and multi-step agent callbacks need their own physical owner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, NoReturn

from ipfs_accelerate_py.endpoint_usage.admission import UsageAdmissionSnapshot
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope,
    LimitEnforcement,
    QuantityKind,
    UsageDimension,
)

from .provider_execution import ProviderExecutionGateway
from .provider_usage_migration import (
    ConsumerCallContext,
    ConsumerId,
    MigratedProviderCallResult,
    ProviderUsageMigrationError,
    SupervisorUsageMode,
    build_consumer_call_context,
    dispatch_migrated_provider_call,
)


_MAX_INTEGER = (1 << 63) - 1
_SNAPSHOT_ID = re.compile(r"usnap_[0-9a-f]{64}\Z")
_REQUIRED_DIMENSIONS = frozenset((
    UsageDimension.REQUESTS, UsageDimension.INPUT_TOKENS, UsageDimension.OUTPUT_TOKENS,
))


def _fail(message: str, reason: str) -> NoReturn:
    raise ProviderUsageMigrationError(message, reason_codes=(reason,))


def _integer(value: Any, name: str, *, zero: bool = False) -> int:
    if type(value) is not int or not (0 if zero else 1) <= value <= _MAX_INTEGER:
        _fail(f"{name} must be a bounded {'nonnegative' if zero else 'positive'} integer",
              "direct_usage_bounds_invalid")
    return value


def _text(value: Any, name: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value.encode("utf-8")) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        _fail(f"{name} must be bounded nonempty text", "direct_usage_binding_invalid")
    return value


def _selector(value: Any, name: str) -> str:
    result = _text(value, name)
    if (result.casefold().rsplit(":", 1)[-1] in {"auto", "efficient", "default", "none", "unknown"}
            or (name == "provider_id" and result.casefold() in {"router", "llm_router", "grok"})
            or "*" in result):
        _fail(f"{name} must name an exact configured selection", "direct_usage_binding_invalid")
    return result


@dataclass(frozen=True, kw_only=True)
class DirectProviderUsageBinding:
    """One explicit provider/model/scope association with a reusable owner.

    ``provider_id`` is the exact router selector; ``scope.provider_id`` is its
    catalog identity. Their association is trusted deployment configuration,
    never inferred from display names. The supplied coordinator must have an
    explicit fence. Its store must already contain finite hard limits for all
    three charged dimensions; request-local budgets cannot substitute for them.
    """

    coordinator: UsageCoordinator = field(repr=False)
    scope: EndpointUsageScope
    provider_id: str
    model: str
    reservation_ttl_ms: int
    fence: int
    catalog_revision: str = "catalog:configured"
    owner_id: str = "supervisor-direct-provider"
    settlement_margin_ms: int = 5_000
    gateway: ProviderExecutionGateway = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinator, UsageCoordinator):
            _fail("direct usage requires a real UsageCoordinator", "direct_usage_owner_required")
        if not isinstance(self.scope, EndpointUsageScope) or self.scope.unknown_scope:
            _fail("direct usage requires an established endpoint scope", "direct_usage_scope_invalid")
        if self.scope.operation != "text.generate":
            _fail("direct usage supports the text.generate operation", "direct_usage_operation_mismatch")
        _selector(self.provider_id, "provider_id")
        _selector(self.model, "model")
        _text(self.catalog_revision, "catalog_revision")
        _text(self.owner_id, "owner_id")
        _integer(self.fence, "fence")
        _integer(self.reservation_ttl_ms, "reservation_ttl_ms")
        _integer(self.settlement_margin_ms, "settlement_margin_ms", zero=True)
        if self.reservation_ttl_ms <= self.settlement_margin_ms:
            _fail("reservation TTL must exceed the settlement margin", "direct_usage_bounds_invalid")
        self._check_owner()
        object.__setattr__(self, "gateway", ProviderExecutionGateway(
            coordinator=self.coordinator, owner_id=self.owner_id,
            reservation_ttl_ms=self.reservation_ttl_ms,
        ))

    def _check_owner(self) -> None:
        # The coordinator currently has no public configured-fence accessor.
        # Checking its configured fence as well as the captured document fence
        # rejects accidental unfenced owners and explicit takeover mismatches.
        configured_fence = getattr(self.coordinator, "_fence", None)
        if type(configured_fence) is not int or configured_fence != self.fence:
            _fail("coordinator fence differs from direct deployment binding",
                  "direct_usage_fence_mismatch")
        if not getattr(self.coordinator.store, "authorizes_admission", False):
            _fail("coordinator store cannot authorize direct usage", "direct_usage_owner_required")

    def _admission(self) -> UsageAdmissionSnapshot:
        self._check_owner()
        if self.gateway.coordinator is not self.coordinator or self.gateway.has_invoker:
            _fail("direct gateway execution owner was changed", "direct_usage_owner_mismatch")
        try:
            view = self.coordinator.admission_snapshot(self.scope.scope_id)
        except Exception as exc:
            raise ProviderUsageMigrationError(
                "direct usage ledger observation is unavailable",
                reason_codes=("direct_usage_admission_unavailable",),
            ) from exc
        if not isinstance(view, UsageAdmissionSnapshot) or view.scope_id != self.scope.scope_id:
            _fail("direct usage observation belongs to another scope", "direct_usage_scope_mismatch")
        if view.fence != self.fence:
            _fail("stored writer fence differs from direct deployment binding",
                  "direct_usage_fence_mismatch")
        configured = {
            limit.dimension for limit in view.snapshot.limits
            if limit.scope_id == self.scope.scope_id
            and limit.enforcement is LimitEnforcement.HARD
            and limit.ceiling.kind is QuantityKind.FINITE
        }
        if not _REQUIRED_DIMENSIONS.issubset(configured):
            _fail("direct usage requires finite configured hard request, input, and output limits",
                  "direct_usage_limits_required")
        return view

    def dispatch(
        self, *, consumer_id: ConsumerId | str, provider_id: str, model: str,
        invoke: Callable[[], str], timeout_ms: int, estimated_input_tokens: int,
        estimated_output_tokens: int, context: ConsumerCallContext | None = None,
        **context_kwargs: Any,
    ) -> MigratedProviderCallResult:
        """Enforce one direct call; no environment downgrade or automatic retry.

        An explicit context must already bind this deployment and carry paired
        observation/admission identities. It is never refreshed or rewritten.
        Without one, a context is built from one current paired ledger view.
        Exact replay requires retaining that original context; a prior ledger
        reservation alone never authorizes another callback after a restart.
        """
        if provider_id != self.provider_id or model != self.model:
            _fail("provider or model differs from direct deployment binding",
                  "direct_usage_binding_mismatch")
        if not callable(invoke):
            _fail("direct usage requires a callback", "direct_usage_invoker_required")
        _integer(timeout_ms, "timeout_ms")
        _integer(estimated_input_tokens, "estimated_input_tokens")
        _integer(estimated_output_tokens, "estimated_output_tokens")
        if self.reservation_ttl_ms <= timeout_ms + self.settlement_margin_ms:
            _fail("reservation TTL must exceed callback timeout plus settlement margin",
                  "direct_usage_bounds_invalid")
        try:
            consumer = ConsumerId(consumer_id)
        except (TypeError, ValueError) as exc:
            raise ProviderUsageMigrationError(
                "direct usage consumer is not registered",
                reason_codes=("direct_usage_consumer_invalid",),
            ) from exc
        if context is not None and (not isinstance(context, ConsumerCallContext) or context_kwargs):
            _fail("supply one complete direct context or context construction fields",
                  "direct_usage_context_conflict")

        view = self._admission()
        required = {
            "consumer_id": consumer,
            "provider_id": self.provider_id,
            "endpoint_scope_id": self.scope.scope_id,
            "fence_id": str(self.fence),
            "catalog_revision": self.catalog_revision,
            "side_effect": self.scope.operation,
            "modality": "text",
            "expected_output_kind": "text",
            "estimated_requests": 1,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
        }
        if context is None:
            fields = {
                **required,
                "usage_revision": view.snapshot.usage_revision,
                "admission_revision": view.admission_revision,
            }
            if any(name in fields and value != fields[name] for name, value in context_kwargs.items()):
                _fail("context fields differ from captured direct deployment binding",
                      "direct_usage_binding_mismatch")
            try:
                context = build_consumer_call_context(**{**context_kwargs, **fields})
            except ProviderUsageMigrationError:
                raise
            except (TypeError, ValueError) as exc:
                raise ProviderUsageMigrationError(
                    "direct usage context is malformed",
                    reason_codes=("direct_usage_context_invalid",),
                ) from exc
        elif any(getattr(context, name) != value for name, value in required.items()):
            _fail("supplied context differs from direct deployment binding",
                  "direct_usage_binding_mismatch")
        if not context.admission_revision or not _SNAPSHOT_ID.fullmatch(context.usage_revision):
            _fail("direct context requires paired ledger observation and admission identities",
                  "direct_usage_pin_required")
        return dispatch_migrated_provider_call(
            context=context, invoke=invoke, mode=SupervisorUsageMode.ENFORCE,
            gateway=self.gateway,
        )


__all__ = ["DirectProviderUsageBinding"]
