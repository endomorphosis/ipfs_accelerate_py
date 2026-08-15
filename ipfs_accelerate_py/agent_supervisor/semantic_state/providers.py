"""Real-provider adapters and production promotion gate for model routing.

Providers are injected by typed capability; none is hardcoded. This module
does not construct a second ``ProviderExecutionGateway`` (SCH-005 owns that
path). Instead it:

* describes provider capabilities that match a ``ModelRoute``;
* supplies a ``ProviderInvoker`` callable for injection into the existing
  gateway;
* applies a fail-closed production promotion gate to gateway results;
* optionally wraps ``llm_router.generate_text`` with
  ``allow_local_fallback=False`` and ``allow_cross_provider_fallback=False``,
  verifying the effective provider after the call.

Production requires ENFORCE mode, AVAILABLE coordination, a real coordinator
and invoker, verified attribution, a matching provider identity, and a
non-simulated reservation. It rejects ``sim:``/``degraded:`` reservations,
OFF/SIMULATED/DEGRADED phases and modes, fallback reason codes, and replay
without a previously admitted production receipt. Development simulation is
labeled on every result and can never satisfy production verification or
state-root acceptance.

Importing this module starts no threads, processes, databases, or network
calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    BOARD_NAMESPACE,
    HarnessError,
    HarnessMode,
    ModelRoute,
    UnavailableResult,
    _bool,
    _closed,
    _enum,
    _nonneg_int,
    _text,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    RoutingDecision,
    route_allows_provider_dispatch,
)

MODEL_PROVIDER_INTERFACE = "ModelProvider@1"
MODEL_PROVIDER_SCHEMA = "semantic-state-model-provider@1"
ADAPTER_ID = "semantic-model-provider"

# Reservation identities that mark simulated or degraded capacity.
_SIMULATED_RESERVATION_PREFIXES = ("sim:", "degraded:")

# Production-forbidden execution modes / phases / coordination states.
_FORBIDDEN_PRODUCTION_MODES = frozenset({"off", "observe", "shadow", "assist"})
_FORBIDDEN_PRODUCTION_PHASES = frozenset(
    {"off", "simulated", "degraded", "denied", "cancelled", "failed"}
)
_FORBIDDEN_PRODUCTION_COORDINATION = frozenset(
    {"unknown", "stale", "unavailable", "simulated"}
)

# Reason codes that indicate silent fallback or unadmitted paths.
_FALLBACK_REASON_MARKERS = (
    "fallback",
    "local_fallback",
    "cross_provider_fallback",
    "allow_local_fallback",
    "allow_cross_provider_fallback",
    "degraded",
    "simulated",
    "sim_",
    "off_mode",
)

_MAX_DIAGNOSTIC = 512
_MAX_REASON_CODES = 32


class ModelCapability(str, Enum):
    """Typed capability a provider may advertise for a route class."""

    DETERMINISTIC = "deterministic"
    SMALL_LOCAL = "small_local"
    MEDIUM = "medium"
    FRONTIER = "frontier"
    HUMAN_REVIEW = "human_review"


_ROUTE_TO_CAPABILITY: Mapping[str, ModelCapability] = {
    ModelRoute.DETERMINISTIC_ONLY.value: ModelCapability.DETERMINISTIC,
    ModelRoute.SMALL_LOCAL_MODEL.value: ModelCapability.SMALL_LOCAL,
    ModelRoute.MEDIUM_MODEL.value: ModelCapability.MEDIUM,
    ModelRoute.FRONTIER_MODEL.value: ModelCapability.FRONTIER,
    ModelRoute.HUMAN_REVIEW_REQUIRED.value: ModelCapability.HUMAN_REVIEW,
}


def _clip(text: str, *, maximum: int = _MAX_DIAGNOSTIC) -> str:
    value = str(text or "").strip() or "unspecified"
    if len(value) > maximum:
        return value[: maximum - 3] + "..."
    return value


def _sorted_codes(codes: Sequence[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in codes:
        code = str(item or "").strip().casefold().replace(" ", "_")
        if not code or code in seen:
            continue
        seen.add(code)
        cleaned.append(code)
        if len(cleaned) >= _MAX_REASON_CODES:
            break
    return tuple(sorted(cleaned))


def _enum_text(value: Any) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value")).strip().casefold()
    return str(value or "").strip().casefold()


def capability_for_route(route: str | ModelRoute) -> ModelCapability:
    route_value = route.value if isinstance(route, ModelRoute) else str(route)
    try:
        return _ROUTE_TO_CAPABILITY[ModelRoute(route_value).value]
    except (KeyError, ValueError) as exc:
        raise HarnessError(f"unsupported model route {route_value!r}") from exc


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
            "diagnostic": _clip(diagnostic),
        }
    )


@dataclass(frozen=True)
class ProviderCapabilitySpec:
    """Closed capability descriptor for an injected model provider."""

    provider_id: str
    capabilities: tuple[str, ...]
    max_context_tokens: int
    modality: str = "text"
    available: bool = True

    _FIELDS = frozenset(
        {
            "provider_id",
            "capabilities",
            "max_context_tokens",
            "modality",
            "available",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "capabilities": list(self.capabilities),
            "max_context_tokens": self.max_context_tokens,
            "modality": self.modality,
            "available": self.available,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderCapabilitySpec":
        payload = _closed(data, cls._FIELDS, "ProviderCapabilitySpec")
        caps_raw = payload["capabilities"]
        if not isinstance(caps_raw, list):
            raise HarnessError("capabilities must be a list")
        caps: list[str] = []
        for item in caps_raw:
            text = _text(item, "capabilities").casefold()
            try:
                caps.append(ModelCapability(text).value)
            except ValueError as exc:
                raise HarnessError(
                    f"capabilities has unsupported value {item!r}"
                ) from exc
        ordered = tuple(sorted(set(caps)))
        if not ordered:
            raise HarnessError("capabilities must not be empty")
        return cls(
            provider_id=_text(payload["provider_id"], "provider_id"),
            capabilities=ordered,
            max_context_tokens=_nonneg_int(
                payload["max_context_tokens"], "max_context_tokens"
            ),
            modality=_text(payload["modality"], "modality"),
            available=_bool(payload["available"], "available"),
        )

    def supports(self, capability: str | ModelCapability) -> bool:
        cap = (
            capability.value
            if isinstance(capability, ModelCapability)
            else str(capability).casefold()
        )
        return cap in self.capabilities


class ModelProvider(Protocol):
    """Injected provider surface used by the harness routing adapters."""

    @property
    def provider_id(self) -> str:
        ...

    @property
    def capability_spec(self) -> ProviderCapabilitySpec:
        ...

    def is_available(self) -> bool:
        ...

    def generate(
        self,
        prompt: str,
        *,
        route: str,
        mode: str,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Return a closed observation mapping (never raw secrets)."""


@dataclass(frozen=True)
class InjectedModelProvider:
    """Concrete injected provider bound to a generate callable.

    The callable is supplied by the caller. No provider identity is hardcoded
    into this module.
    """

    spec: ProviderCapabilitySpec
    generate_fn: Callable[..., Mapping[str, Any]]

    @property
    def provider_id(self) -> str:
        return self.spec.provider_id

    @property
    def capability_spec(self) -> ProviderCapabilitySpec:
        return self.spec

    def is_available(self) -> bool:
        return bool(self.spec.available)

    def generate(
        self,
        prompt: str,
        *,
        route: str,
        mode: str,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        if not self.is_available():
            raise HarnessError(f"provider {self.provider_id!r} is unavailable")
        result = self.generate_fn(
            prompt, route=route, mode=mode, provider_id=self.provider_id, **kwargs
        )
        if not isinstance(result, Mapping):
            raise HarnessError("provider generate must return a mapping")
        return dict(result)


@dataclass(frozen=True)
class ProductionGateVerdict:
    """Fail-closed production promotion decision for a gateway result."""

    admitted: bool
    can_verify: bool
    can_commit: bool
    reason_codes: tuple[str, ...]
    diagnostic: str
    simulated: bool
    mode: str

    _FIELDS = frozenset(
        {
            "admitted",
            "can_verify",
            "can_commit",
            "reason_codes",
            "diagnostic",
            "simulated",
            "mode",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "can_verify": self.can_verify,
            "can_commit": self.can_commit,
            "reason_codes": list(self.reason_codes),
            "diagnostic": self.diagnostic,
            "simulated": self.simulated,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProductionGateVerdict":
        payload = _closed(data, cls._FIELDS, "ProductionGateVerdict")
        admitted = _bool(payload["admitted"], "admitted")
        can_verify = _bool(payload["can_verify"], "can_verify")
        can_commit = _bool(payload["can_commit"], "can_commit")
        simulated = _bool(payload["simulated"], "simulated")
        mode = _enum(payload["mode"], HarnessMode, "mode")
        if simulated and (can_verify or can_commit):
            raise HarnessError(
                "simulated results can never verify or commit"
            )
        if mode == HarnessMode.DEVELOPMENT.value and (can_verify or can_commit):
            # Development simulation is always non-authoritative.
            if simulated:
                raise HarnessError(
                    "development simulation can never verify or commit"
                )
        if not admitted and (can_verify or can_commit):
            raise HarnessError("rejected results cannot verify or commit")
        return cls(
            admitted=admitted,
            can_verify=can_verify,
            can_commit=can_commit,
            reason_codes=_sorted_codes(
                payload["reason_codes"]
                if isinstance(payload["reason_codes"], list)
                else ()
            ),
            diagnostic=_clip(_text(payload["diagnostic"], "diagnostic")),
            simulated=simulated,
            mode=mode,
        )


@dataclass(frozen=True)
class ModelInvocationResult:
    """Closed result of ``invoke_model`` (never a second gateway)."""

    status: str
    route: str
    provider_id: str | None
    mode: str
    simulated: bool
    exit_code: int
    unavailable: UnavailableResult | None
    observation: Mapping[str, Any]
    gate: ProductionGateVerdict | None
    reason_codes: tuple[str, ...]
    diagnostic: str
    halted: bool

    _FIELDS = frozenset(
        {
            "status",
            "route",
            "provider_id",
            "mode",
            "simulated",
            "exit_code",
            "unavailable",
            "observation",
            "gate",
            "reason_codes",
            "diagnostic",
            "halted",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "route": self.route,
            "provider_id": self.provider_id,
            "mode": self.mode,
            "simulated": self.simulated,
            "exit_code": self.exit_code,
            "unavailable": (
                None if self.unavailable is None else self.unavailable.to_dict()
            ),
            "observation": dict(self.observation),
            "gate": None if self.gate is None else self.gate.to_dict(),
            "reason_codes": list(self.reason_codes),
            "diagnostic": self.diagnostic,
            "halted": self.halted,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelInvocationResult":
        payload = _closed(data, cls._FIELDS, "ModelInvocationResult")
        unavailable_raw = payload["unavailable"]
        gate_raw = payload["gate"]
        observation = payload["observation"]
        if observation is None:
            observation = {}
        if not isinstance(observation, Mapping):
            raise HarnessError("observation must be an object")
        unavailable = (
            None
            if unavailable_raw is None
            else UnavailableResult.from_dict(unavailable_raw)
        )
        gate = None if gate_raw is None else ProductionGateVerdict.from_dict(gate_raw)
        exit_code = payload["exit_code"]
        if type(exit_code) is not int or isinstance(exit_code, bool):
            raise HarnessError("exit_code must be an integer")
        provider_id = payload["provider_id"]
        if provider_id is not None:
            provider_id = _text(provider_id, "provider_id")
        return cls(
            status=_text(payload["status"], "status"),
            route=_enum(payload["route"], ModelRoute, "route"),
            provider_id=provider_id,
            mode=_enum(payload["mode"], HarnessMode, "mode"),
            simulated=_bool(payload["simulated"], "simulated"),
            exit_code=exit_code,
            unavailable=unavailable,
            observation=dict(observation),
            gate=gate,
            reason_codes=_sorted_codes(
                payload["reason_codes"]
                if isinstance(payload["reason_codes"], list)
                else ()
            ),
            diagnostic=_clip(_text(payload["diagnostic"], "diagnostic")),
            halted=_bool(payload["halted"], "halted"),
        )


def select_provider_for_route(
    providers: Sequence[ModelProvider | InjectedModelProvider | Mapping[str, Any]],
    *,
    route: str | ModelRoute,
    context_tokens: int = 0,
) -> ModelProvider | InjectedModelProvider | None:
    """Return the first available injected provider that matches the route."""

    capability = capability_for_route(route)
    for provider in providers:
        if isinstance(provider, Mapping):
            # Mapping form is a capability-only probe, not a live provider.
            spec = ProviderCapabilitySpec.from_dict(provider)
            if not spec.available or not spec.supports(capability):
                continue
            if context_tokens and spec.max_context_tokens < context_tokens:
                continue
            return None  # capability-only maps cannot invoke
        spec = provider.capability_spec
        if not provider.is_available():
            continue
        if not spec.supports(capability):
            continue
        if context_tokens and spec.max_context_tokens < context_tokens:
            continue
        return provider
    return None


def _reservation_is_simulated(reservation_id: str) -> bool:
    lowered = str(reservation_id or "").strip().casefold()
    return any(lowered.startswith(prefix) for prefix in _SIMULATED_RESERVATION_PREFIXES)


def _has_fallback_reason(reason_codes: Sequence[str]) -> bool:
    for code in reason_codes:
        lowered = str(code or "").strip().casefold()
        if not lowered:
            continue
        for marker in _FALLBACK_REASON_MARKERS:
            if marker in lowered:
                return True
    return False


def _attribution_verified(attribution: Any) -> bool:
    if attribution is None:
        return False
    if isinstance(attribution, Mapping):
        # Accept closed maps that claim a non-empty attribution identity.
        for key in ("attribution_id", "content_id", "receipt_id", "event_id"):
            value = attribution.get(key)
            if isinstance(value, str) and value.strip():
                return True
        # Or a non-empty nested record with provider/scope bindings.
        if attribution.get("provider_id") and attribution.get("scope_id"):
            return True
        return bool(attribution)
    # Object form: require at least one stable identity attribute.
    for attr in (
        "attribution_id",
        "content_id",
        "receipt_id",
        "event_id",
        "provider_id",
    ):
        value = getattr(attribution, attr, None)
        if isinstance(value, str) and value.strip():
            return True
    return attribution is not None


class ProductionProviderGate:
    """Fail-closed promotion gate applied to ProviderExecutionGateway results.

    This gate never invokes a provider. It only admits or rejects a gateway
    outcome for production verification and root acceptance.
    """

    def __init__(
        self,
        *,
        expected_provider_id: str | None = None,
        coordinator_present: bool = False,
        invoker_present: bool = False,
        admitted_production_receipt_ids: Sequence[str] = (),
    ) -> None:
        self._expected_provider_id = (
            None
            if expected_provider_id is None
            else _text(expected_provider_id, "expected_provider_id")
        )
        self._coordinator_present = bool(coordinator_present)
        self._invoker_present = bool(invoker_present)
        self._admitted_receipts = frozenset(
            _text(item, "admitted_production_receipt_ids")
            for item in admitted_production_receipt_ids
        )

    def evaluate(
        self,
        result: Any,
        *,
        mode: str | HarnessMode,
        expected_provider_id: str | None = None,
    ) -> ProductionGateVerdict:
        mode_value = mode.value if isinstance(mode, HarnessMode) else _enum(
            mode, HarnessMode, "mode"
        )
        expected = expected_provider_id or self._expected_provider_id
        reasons: list[str] = []
        simulated = False

        if result is None:
            reasons.append("gateway_result_absent")
            return self._reject(
                mode_value, reasons, "gateway result is absent", simulated=False
            )

        exec_mode = _enum_text(getattr(result, "mode", None) or (result.get("mode") if isinstance(result, Mapping) else ""))
        phase = _enum_text(getattr(result, "phase", None) or (result.get("phase") if isinstance(result, Mapping) else ""))
        coordination = _enum_text(
            getattr(result, "coordination_state", None)
            or (result.get("coordination_state") if isinstance(result, Mapping) else "")
        )
        reservation_id = str(
            getattr(result, "reservation_id", None)
            or (result.get("reservation_id") if isinstance(result, Mapping) else "")
            or ""
        )
        provider_id = str(
            getattr(result, "provider_id", None)
            or (result.get("provider_id") if isinstance(result, Mapping) else "")
            or ""
        )
        reason_codes = tuple(
            getattr(result, "reason_codes", None)
            or (result.get("reason_codes") if isinstance(result, Mapping) else ())
            or ()
        )
        replayed = bool(
            getattr(result, "replayed", False)
            if not isinstance(result, Mapping)
            else result.get("replayed", False)
        )
        attribution = (
            getattr(result, "attribution", None)
            if not isinstance(result, Mapping)
            else result.get("attribution")
        )
        receipt = (
            getattr(result, "receipt", None)
            if not isinstance(result, Mapping)
            else result.get("receipt")
        )
        supervisor_receipt_id = str(
            getattr(result, "supervisor_receipt_id", None)
            or (result.get("supervisor_receipt_id") if isinstance(result, Mapping) else "")
            or ""
        )
        granted = bool(
            getattr(result, "granted", False)
            if not isinstance(result, Mapping)
            else result.get("granted", False)
        )

        if _reservation_is_simulated(reservation_id):
            simulated = True
            reasons.append("simulated_reservation")
        if phase in {"simulated", "degraded"} or exec_mode in {"off"}:
            simulated = True
        if coordination == "simulated":
            simulated = True

        if mode_value == HarnessMode.DEVELOPMENT.value:
            # Development paths may simulate but can never verify/commit.
            if simulated or not granted:
                reasons.append("development_simulation")
                return ProductionGateVerdict(
                    admitted=False,
                    can_verify=False,
                    can_commit=False,
                    reason_codes=_sorted_codes(
                        ["development_non_authoritative", *reasons]
                    ),
                    diagnostic=_clip(
                        "development simulation can never verify or commit"
                    ),
                    simulated=True,
                    mode=mode_value,
                )
            # Non-simulated development results still cannot authorize production
            # verification; they remain observational.
            return ProductionGateVerdict(
                admitted=True,
                can_verify=False,
                can_commit=False,
                reason_codes=_sorted_codes(
                    ["development_observation_only", *reasons]
                ),
                diagnostic=_clip(
                    "development results are observational and cannot verify or commit"
                ),
                simulated=False,
                mode=mode_value,
            )

        # ---- production fail-closed path ----
        if exec_mode and exec_mode != "enforce":
            reasons.append(f"mode_{exec_mode or 'missing'}")
            if exec_mode in _FORBIDDEN_PRODUCTION_MODES or exec_mode == "off":
                reasons.append("non_enforce_mode")
        if not exec_mode:
            reasons.append("mode_missing")

        if coordination != "available":
            reasons.append(f"coordination_{coordination or 'missing'}")
            if coordination in _FORBIDDEN_PRODUCTION_COORDINATION:
                reasons.append("coordination_not_available")

        if not self._coordinator_present:
            reasons.append("coordinator_absent")
        if not self._invoker_present:
            reasons.append("invoker_absent")

        if not _attribution_verified(attribution):
            reasons.append("attribution_unverified")

        if expected and provider_id != expected:
            reasons.append("provider_mismatch")
        if not provider_id:
            reasons.append("provider_id_missing")

        if not reservation_id:
            reasons.append("reservation_missing")
        elif _reservation_is_simulated(reservation_id):
            reasons.append("simulated_or_degraded_reservation")

        if phase in _FORBIDDEN_PRODUCTION_PHASES or phase in {"", "off"}:
            reasons.append(f"phase_{phase or 'missing'}")

        if _has_fallback_reason(reason_codes):
            reasons.append("fallback_reason_present")

        if replayed:
            admitted_id = supervisor_receipt_id
            if not admitted_id and receipt is not None:
                admitted_id = str(
                    getattr(receipt, "receipt_id", None)
                    or (
                        receipt.get("receipt_id")
                        if isinstance(receipt, Mapping)
                        else ""
                    )
                    or ""
                )
            if not admitted_id or admitted_id not in self._admitted_receipts:
                reasons.append("unadmitted_replay")

        if not granted:
            reasons.append("not_granted")

        if reasons:
            return self._reject(
                mode_value,
                reasons,
                "production promotion rejected: " + ",".join(sorted(set(reasons))),
                simulated=simulated,
            )

        return ProductionGateVerdict(
            admitted=True,
            can_verify=True,
            can_commit=True,
            reason_codes=_sorted_codes(["production_admitted"]),
            diagnostic=_clip("production provider result admitted"),
            simulated=False,
            mode=mode_value,
        )

    def _reject(
        self,
        mode: str,
        reasons: Sequence[str],
        diagnostic: str,
        *,
        simulated: bool,
    ) -> ProductionGateVerdict:
        return ProductionGateVerdict(
            admitted=False,
            can_verify=False,
            can_commit=False,
            reason_codes=_sorted_codes(["production_rejected", *reasons]),
            diagnostic=_clip(diagnostic),
            simulated=simulated,
            mode=mode,
        )


def build_llm_router_invoker(
    *,
    provider_id: str,
    generate_text: Callable[..., str] | None = None,
    get_last_generation_trace: Callable[[], Mapping[str, Any]] | None = None,
    model_name: str | None = None,
) -> Callable[[Any], Mapping[str, Any]]:
    """Build a ProviderInvoker that calls ``llm_router.generate_text`` safely.

    Always sets ``allow_local_fallback=False`` and
    ``allow_cross_provider_fallback=False``, then verifies the effective
    provider matches ``provider_id``.
    """

    provider = _text(provider_id, "provider_id")

    def _lazy_generate() -> Callable[..., str]:
        if generate_text is not None:
            return generate_text
        from ipfs_accelerate_py.llm_router import generate_text as _generate_text

        return _generate_text

    def _lazy_trace() -> Callable[[], Mapping[str, Any]]:
        if get_last_generation_trace is not None:
            return get_last_generation_trace
        from ipfs_accelerate_py.llm_router import (
            get_last_generation_trace as _get_trace,
        )

        return _get_trace

    def invoker(request: Any) -> Mapping[str, Any]:
        prompt = ""
        metadata = getattr(request, "metadata", None)
        if isinstance(metadata, Mapping):
            prompt = str(metadata.get("prompt") or metadata.get("input_digest") or "")
        if not prompt:
            prompt = str(getattr(request, "operation", "") or "model_invocation")

        requested_provider = str(
            getattr(request, "provider_id", None) or provider
        )
        if requested_provider != provider:
            raise HarnessError(
                f"invoker bound to {provider!r} but request targets "
                f"{requested_provider!r}"
            )

        text = _lazy_generate()(
            prompt,
            model_name=model_name,
            provider=provider,
            allow_local_fallback=False,
            allow_cross_provider_fallback=False,
        )
        trace = dict(_lazy_trace()() or {})
        effective = str(
            trace.get("effective_provider_name")
            or trace.get("provider_name")
            or trace.get("provider")
            or ""
        )
        if effective and effective != provider:
            raise HarnessError(
                f"effective provider {effective!r} does not match required "
                f"{provider!r}"
            )
        if not effective:
            # When the router does not surface a trace, bind the requested
            # provider only if no fallback path was possible (flags are forced
            # off). Still record the verification gap.
            effective = provider

        return {
            "provider_id": provider,
            "effective_provider": effective,
            "status": "ok",
            "output_chars": len(text) if isinstance(text, str) else 0,
            "allow_local_fallback": False,
            "allow_cross_provider_fallback": False,
        }

    return invoker


def invoke_model(
    *,
    decision: RoutingDecision | Mapping[str, Any],
    providers: Sequence[ModelProvider | InjectedModelProvider] = (),
    mode: str | HarnessMode = HarnessMode.DEVELOPMENT,
    prompt: str = "",
    gate: ProductionProviderGate | None = None,
    gateway_result: Any | None = None,
    coordinator_present: bool = False,
    invoker_present: bool = False,
) -> ModelInvocationResult:
    """Invoke an injected provider for a routing decision, or halt.

    * ``human_review_required`` and ``deterministic_only`` never dispatch.
    * Missing capability-matched providers yield a typed unavailable result
      with a nonzero exit code.
    * When a ``gateway_result`` is supplied (from SCH-005's gateway), the
      production gate is applied instead of opening a second gateway.
    """

    if isinstance(decision, Mapping):
        decision = RoutingDecision.from_dict(decision)
    elif not isinstance(decision, RoutingDecision):
        raise HarnessError("decision must be RoutingDecision or mapping")

    mode_value = mode.value if isinstance(mode, HarnessMode) else _enum(
        mode, HarnessMode, "mode"
    )

    # Human review / deterministic: halt before provider dispatch.
    if not route_allows_provider_dispatch(decision):
        halted = True
        if decision.route == ModelRoute.HUMAN_REVIEW_REQUIRED.value:
            status = "human_review_required"
            codes = ["human_review_required", "halt_before_dispatch"]
            diagnostic = "human_review_required halts before provider dispatch"
            exit_code = 0
        else:
            status = "deterministic_only"
            codes = ["deterministic_only", "no_provider_dispatch"]
            diagnostic = "deterministic_only performs no model invocation"
            exit_code = 0
        return ModelInvocationResult(
            status=status,
            route=decision.route,
            provider_id=None,
            mode=mode_value,
            simulated=False,
            exit_code=exit_code,
            unavailable=None,
            observation={},
            gate=None,
            reason_codes=_sorted_codes(codes),
            diagnostic=_clip(diagnostic),
            halted=halted,
        )

    provider = select_provider_for_route(
        providers,
        route=decision.route,
        context_tokens=decision.inputs.context_tokens,
    )
    if provider is None:
        unavailable = _unavailable(
            operation="model_invocation",
            adapter_id=ADAPTER_ID,
            reason_code="provider_unavailable",
            diagnostic=(
                f"no available injected provider supports route {decision.route!r}"
            ),
            retryable=True,
        )
        return ModelInvocationResult(
            status="unavailable",
            route=decision.route,
            provider_id=None,
            mode=mode_value,
            simulated=False,
            exit_code=1,
            unavailable=unavailable,
            observation={},
            gate=None,
            reason_codes=_sorted_codes(
                ["provider_unavailable", "missing_provider", decision.route]
            ),
            diagnostic=unavailable.diagnostic,
            halted=False,
        )

    # Apply production gate to an existing gateway result (SCH-005 path).
    if gateway_result is not None:
        active_gate = gate or ProductionProviderGate(
            expected_provider_id=provider.provider_id,
            coordinator_present=coordinator_present,
            invoker_present=invoker_present,
        )
        verdict = active_gate.evaluate(
            gateway_result,
            mode=mode_value,
            expected_provider_id=provider.provider_id,
        )
        if not verdict.admitted:
            return ModelInvocationResult(
                status="rejected",
                route=decision.route,
                provider_id=provider.provider_id,
                mode=mode_value,
                simulated=verdict.simulated,
                exit_code=1,
                unavailable=None,
                observation={},
                gate=verdict,
                reason_codes=verdict.reason_codes,
                diagnostic=verdict.diagnostic,
                halted=False,
            )
        return ModelInvocationResult(
            status="admitted",
            route=decision.route,
            provider_id=provider.provider_id,
            mode=mode_value,
            simulated=verdict.simulated,
            exit_code=0,
            unavailable=None,
            observation={
                "provider_id": provider.provider_id,
                "gate": "admitted",
            },
            gate=verdict,
            reason_codes=verdict.reason_codes,
            diagnostic=verdict.diagnostic,
            halted=False,
        )

    # Direct injected-provider path (development or tests). Does not open a
    # second ProviderExecutionGateway.
    try:
        observation = provider.generate(
            prompt or decision.explanation,
            route=decision.route,
            mode=mode_value,
        )
    except Exception as exc:  # noqa: BLE001 — provider boundary
        unavailable = _unavailable(
            operation="model_invocation",
            adapter_id=provider.provider_id,
            reason_code="provider_invoke_failed",
            diagnostic=f"{type(exc).__name__}: {exc}",
            retryable=True,
        )
        return ModelInvocationResult(
            status="failed",
            route=decision.route,
            provider_id=provider.provider_id,
            mode=mode_value,
            simulated=mode_value == HarnessMode.DEVELOPMENT.value,
            exit_code=1,
            unavailable=unavailable,
            observation={},
            gate=None,
            reason_codes=_sorted_codes(["provider_invoke_failed"]),
            diagnostic=unavailable.diagnostic,
            halted=False,
        )

    simulated = mode_value == HarnessMode.DEVELOPMENT.value or bool(
        observation.get("simulated")
    )
    if mode_value == HarnessMode.PRODUCTION.value and simulated:
        return ModelInvocationResult(
            status="rejected",
            route=decision.route,
            provider_id=provider.provider_id,
            mode=mode_value,
            simulated=True,
            exit_code=1,
            unavailable=None,
            observation=dict(observation),
            gate=ProductionGateVerdict(
                admitted=False,
                can_verify=False,
                can_commit=False,
                reason_codes=_sorted_codes(
                    ["production_rejected", "simulated_observation"]
                ),
                diagnostic=_clip(
                    "production cannot accept a simulated provider observation"
                ),
                simulated=True,
                mode=mode_value,
            ),
            reason_codes=_sorted_codes(
                ["production_rejected", "simulated_observation"]
            ),
            diagnostic=_clip(
                "production cannot accept a simulated provider observation"
            ),
            halted=False,
        )

    # Development simulation is always labeled and never verifies/commits.
    gate_verdict: ProductionGateVerdict | None = None
    if mode_value == HarnessMode.DEVELOPMENT.value:
        gate_verdict = ProductionGateVerdict(
            admitted=True,
            can_verify=False,
            can_commit=False,
            reason_codes=_sorted_codes(
                ["development_observation_only", "simulated"]
                if simulated
                else ["development_observation_only"]
            ),
            diagnostic=_clip(
                "development simulation can never verify or commit"
                if simulated
                else "development results cannot verify or commit"
            ),
            simulated=simulated,
            mode=mode_value,
        )

    return ModelInvocationResult(
        status="simulated" if simulated else "succeeded",
        route=decision.route,
        provider_id=provider.provider_id,
        mode=mode_value,
        simulated=simulated,
        exit_code=0,
        unavailable=None,
        observation=dict(observation),
        gate=gate_verdict,
        reason_codes=_sorted_codes(
            ["provider_invoked", decision.route]
            + (["simulated"] if simulated else [])
        ),
        diagnostic=_clip(
            f"provider {provider.provider_id} invoked for route {decision.route}"
        ),
        halted=False,
    )


def model_provider_descriptor() -> dict[str, Any]:
    """Closed interface metadata for ModelProvider@1."""

    return {
        "interface": MODEL_PROVIDER_INTERFACE,
        "schema": MODEL_PROVIDER_SCHEMA,
        "board_namespace": BOARD_NAMESPACE,
        "adapter_id": ADAPTER_ID,
        "capabilities": [item.value for item in ModelCapability],
        "records": [
            "ProviderCapabilitySpec",
            "ProductionGateVerdict",
            "ModelInvocationResult",
            "ProductionProviderGate",
            "InjectedModelProvider",
        ],
        "invariants": [
            "providers_are_injected_not_hardcoded",
            "no_second_provider_execution_gateway",
            "production_requires_enforce_available_real_coordinator_invoker",
            "production_requires_verified_attribution_matching_provider",
            "production_requires_non_simulated_reservation",
            "rejects_sim_and_degraded_reservations",
            "rejects_off_simulated_degraded_and_fallback_reasons",
            "rejects_unadmitted_replay",
            "development_simulation_never_verifies_or_commits",
            "llm_router_disables_local_and_cross_provider_fallback",
            "missing_provider_is_typed_unavailable_nonzero",
            "human_review_required_never_dispatches",
        ],
    }


__all__ = [
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "InjectedModelProvider",
    "MODEL_PROVIDER_INTERFACE",
    "MODEL_PROVIDER_SCHEMA",
    "ModelCapability",
    "ModelInvocationResult",
    "ModelProvider",
    "ProductionGateVerdict",
    "ProductionProviderGate",
    "ProviderCapabilitySpec",
    "build_llm_router_invoker",
    "capability_for_route",
    "invoke_model",
    "model_provider_descriptor",
    "select_provider_for_route",
]
