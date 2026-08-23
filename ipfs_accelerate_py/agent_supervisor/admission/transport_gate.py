"""FACP-040 — common transport gate for migrated Accelerate transports.

Routes CLI / MCP / MCP++ (and the inventoried Python host seam) through one
Effect Admission Kernel unlock path. Fail-closed invariants:

* Direct handler call without a gate-issued unlock permit fails.
* Every migrated transport makes the same ``kernel.unlock_handler`` call.
* Browser / model / peer / prompt / payment / UI inputs cannot select
  authority (tenant, policy, endpoint, path, actor, or issuer).
* Denied admission yields **zero** handler invocations.

Cold import is hermetic: no network, provider execution, or process mutation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.admission.formal_kernel import (
    FORBIDDEN_TOKEN_ISSUERS,
    FREE_FORM_AUTHORITY_KEYS,
    KERNEL_ISSUER,
    AdmissionDecision,
    AdmissionError,
    AdmissionErrorCode,
    AdmissionToken,
    AdmissionVerdict,
    EffectAdmissionKernel,
    OperationSpecView,
    binding_cid,
    default_kernel,
    derive_token_obligations,
)

# ---------------------------------------------------------------------------
# FACP evidence envelope
# ---------------------------------------------------------------------------

SCHEMA: Final[str] = "facp/common-transport-gate@1"
DECISION_SCHEMA: Final[str] = "facp/transport-gate-decision@1"
TASK_ID: Final[str] = "FACP-040"
GOAL_ID: Final[str] = "FACP-G320"
BUNDLE: Final[str] = "facp/admission/transports"
INTERFACE: Final[str] = "CommonTransportGate@1"
KERNEL_CALL: Final[str] = "effect_admission_kernel.unlock_handler"
UNSAFE_PROMOTION: Final[bool] = False

EVIDENCE_SUBSET: Final[tuple[str, ...]] = (
    "same_token_decision_across_transports",
    "effect_class_match",
    "exact_args",
    "revocation",
    "denial",
    "typed_observation_outcome",
)

# Migrated Accelerate transport seams owned by this gate.
MIGRATED_TRANSPORTS: Final[tuple[str, ...]] = ("cli", "mcp", "mcp++", "python")

CLOSED_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "Unavailable",
        "Rejected",
        "Simulated",
        "Attempted",
        "Unknown",
        "Observed",
        "Verified",
        "Failed",
        "Compensated",
    }
)

# Inputs that must never select host authority bindings.
UNTRUSTED_AUTHORITY_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "browser",
        "browser_consent",
        "prompt",
        "model",
        "peer",
        "payment",
        "ui",
        "caller",
        "consent",
        "dry_run",
        "allow",
    }
)

# Caller-facing fields that would select or widen authority if trusted.
AUTHORITY_SELECTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "tenant",
        "tenant_cid",
        "tenant_id",
        "policy",
        "policy_cid",
        "policy_id",
        "endpoint",
        "endpoint_id",
        "path",
        "paths",
        "actor",
        "actor_cid",
        "issuer",
        "authority",
        "authorization",
        "permission",
        "grant",
        "consent",
        "allowed",
        "dry_run",
        "delegation",
        "delegation_cid",
        "lease",
        "lease_id",
        "confirmation",
        "confirmation_cid",
    }
)

# Inventoried CLI / MCP / MCP++ handler seams (Accelerate migration surface).
INVENTORIED_TRANSPORT_SEAMS: Final[tuple[Mapping[str, Any], ...]] = (
    {
        "transport": "cli",
        "seam_id": "seam:accelerate-cli-inference",
        "operation_id": "accelerate.inference",
        "symbol": "ipfs_accelerate_py.cli.inference",
        "effect_class": "process",
    },
    {
        "transport": "cli",
        "seam_id": "seam:accelerate-cli-capability-probe",
        "operation_id": "accelerate.capability_probe",
        "symbol": "ipfs_accelerate_py.cli.capability_probe",
        "effect_class": "read",
    },
    {
        "transport": "mcp",
        "seam_id": "seam:accelerate-mcp-inference",
        "operation_id": "accelerate.inference",
        "symbol": "ipfs_accelerate_py.mcp.inference_tools",
        "effect_class": "process",
    },
    {
        "transport": "mcp",
        "seam_id": "seam:accelerate-mcp-capability",
        "operation_id": "accelerate.capability_probe",
        "symbol": "ipfs_accelerate_py.mcp.unified_tools",
        "effect_class": "read",
    },
    {
        "transport": "mcp++",
        "seam_id": "seam:accelerate-mcpp-inference",
        "operation_id": "accelerate.inference",
        "symbol": "ipfs_accelerate_py.mcp_server.tools.inference",
        "effect_class": "process",
    },
    {
        "transport": "mcp++",
        "seam_id": "seam:accelerate-mcpp-capability",
        "operation_id": "accelerate.capability_probe",
        "symbol": "ipfs_accelerate_py.mcp_server.mcplusplus",
        "effect_class": "read",
    },
    {
        "transport": "python",
        "seam_id": "seam:accelerate-python-host",
        "operation_id": "accelerate.inference",
        "symbol": "ipfs_accelerate_py.agent_supervisor.admission.transport_gate",
        "effect_class": "process",
    },
)


class TransportKind(str, Enum):
    """Closed set of migrated Accelerate transport adapters."""

    CLI = "cli"
    MCP = "mcp"
    MCP_PLUS_PLUS = "mcp++"
    PYTHON = "python"


class TransportGateError(ValueError):
    """Fail-closed transport gate contract violation."""

    def __init__(self, code: AdmissionErrorCode | str, message: str) -> None:
        if isinstance(code, AdmissionErrorCode):
            self.code = code
        else:
            try:
                self.code = AdmissionErrorCode(code)
            except ValueError:
                self.code = AdmissionErrorCode.INVALID_TYPE
                message = f"{code}: {message}"
        super().__init__(message)


class HandlerNotUnlockedError(TransportGateError):
    """Raised when an effectful handler is invoked without an unlock permit."""

    def __init__(self, message: str = "effectful handler requires gate unlock") -> None:
        super().__init__(AdmissionErrorCode.HANDLER_NOT_UNLOCKED, message)


# Thread-local unlock permits: only the gate may install them.
_UNLOCK: threading.local = threading.local()


def _current_permit() -> "_UnlockPermit | None":
    return getattr(_UNLOCK, "permit", None)


def _install_permit(permit: "_UnlockPermit") -> None:
    _UNLOCK.permit = permit


def _clear_permit() -> None:
    _UNLOCK.permit = None


@dataclass(frozen=True)
class _UnlockPermit:
    """One-shot, argument-bound permit issued only after kernel unlock."""

    operation_id: str
    argument_cid: str
    token_id: str
    transport: str
    permit_id: str


@dataclass(frozen=True)
class KernelCallRecord:
    """Evidence that transports share one kernel unlock call shape."""

    method: str
    operation_id: str
    effect_class: str
    argument_cid: str
    typestate: str
    token_id: str
    transport: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "operation_id": self.operation_id,
            "effect_class": self.effect_class,
            "argument_cid": self.argument_cid,
            "typestate": self.typestate,
            "token_id": self.token_id,
            "transport": self.transport,
        }

    def identity_without_transport(self) -> dict[str, Any]:
        """Kernel unlock call shape shared by every migrated transport.

        Token identifiers differ per one-use mint; the authoritative shared
        call is ``unlock_handler`` bound to operation / effect / args /
        typestate.
        """
        return {
            "method": self.method,
            "operation_id": self.operation_id,
            "effect_class": self.effect_class,
            "argument_cid": self.argument_cid,
            "typestate": self.typestate,
        }


@dataclass(frozen=True)
class TransportRequest:
    """Normalized request shared by all migrated transports."""

    operation_id: str
    arguments: Mapping[str, Any]
    typestate: str = "Reserved"
    token: AdmissionToken | Mapping[str, Any] | None = None
    authority_source: str = "host"
    # Untrusted overlays: may carry browser/model/peer fields; never authority.
    transport_overlay: Mapping[str, Any] = field(default_factory=dict)
    host_bindings: Mapping[str, Any] = field(default_factory=dict)
    # When False, unlock verifies without consuming the nonce (parity probes).
    consume_token: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _require_text(self.operation_id, "operation_id")
        )
        object.__setattr__(
            self, "typestate", _require_text(self.typestate, "typestate")
        )
        object.__setattr__(
            self,
            "authority_source",
            _require_text(self.authority_source, "authority_source", required=False)
            or "host",
        )
        if not isinstance(self.arguments, Mapping):
            raise TransportGateError(
                AdmissionErrorCode.INVALID_TYPE, "arguments must be a mapping"
            )
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        if not isinstance(self.transport_overlay, Mapping):
            raise TransportGateError(
                AdmissionErrorCode.INVALID_TYPE,
                "transport_overlay must be a mapping",
            )
        object.__setattr__(
            self, "transport_overlay", MappingProxyType(dict(self.transport_overlay))
        )
        if not isinstance(self.host_bindings, Mapping):
            raise TransportGateError(
                AdmissionErrorCode.INVALID_TYPE, "host_bindings must be a mapping"
            )
        object.__setattr__(
            self, "host_bindings", MappingProxyType(dict(self.host_bindings))
        )


@dataclass(frozen=True)
class TransportResult:
    """Typed observation outcome for a gated transport dispatch."""

    outcome: str
    admitted: bool
    handler_invoked: bool
    decision: AdmissionDecision
    kernel_call: KernelCallRecord | None
    transport: str
    operation_id: str
    argument_cid: str
    result: Any = None
    code: str | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in CLOSED_OUTCOMES:
            raise TransportGateError(
                AdmissionErrorCode.UNKNOWN_ENUM,
                f"unknown outcome={self.outcome!r}",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "schema_version": 1,
            "outcome": self.outcome,
            "admitted": self.admitted,
            "handler_invoked": self.handler_invoked,
            "decision": self.decision.to_dict(),
            "kernel_call": None if self.kernel_call is None else self.kernel_call.to_dict(),
            "transport": self.transport,
            "operation_id": self.operation_id,
            "argument_cid": self.argument_cid,
            "code": self.code,
            "message": self.message,
            "result": self.result,
            "unsafe_promotion": UNSAFE_PROMOTION,
        }


HandlerFn = Callable[[Mapping[str, Any]], Any]


@dataclass
class _RegisteredHandler:
    spec: OperationSpecView
    handler: HandlerFn
    invocation_count: int = 0
    gated: "GatedHandler | None" = None


class GatedHandler:
    """Handler wrapper that refuses direct calls without a gate unlock permit.

    Migrated transport seams MUST expose this wrapper (or an equivalent
    ``require_unlock`` check) so a direct Python call without admission fails.
    """

    def __init__(
        self,
        operation_id: str,
        handler: HandlerFn,
        *,
        gate: "CommonTransportGate",
    ) -> None:
        self.operation_id = _require_text(operation_id, "operation_id")
        self._handler = handler
        self._gate = gate

    def __call__(self, arguments: Mapping[str, Any] | None = None) -> Any:
        arguments = {} if arguments is None else arguments
        if not isinstance(arguments, Mapping):
            raise TransportGateError(
                AdmissionErrorCode.INVALID_TYPE, "arguments must be a mapping"
            )
        permit = _current_permit()
        if permit is None:
            raise HandlerNotUnlockedError(
                f"direct call to {self.operation_id!r} without admission token"
            )
        if permit.operation_id != self.operation_id:
            raise HandlerNotUnlockedError(
                f"unlock permit operation_id mismatch for {self.operation_id!r}"
            )
        argument_cid = argument_cid_for(arguments)
        if permit.argument_cid != argument_cid:
            raise HandlerNotUnlockedError(
                f"unlock permit argument_cid mismatch for {self.operation_id!r}"
            )
        return self._handler(MappingProxyType(dict(arguments)))

    @property
    def raw_handler(self) -> HandlerFn:
        """Underlying callable — still must not be used as a production seam."""
        return self._handler


def _require_text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise TransportGateError(
            AdmissionErrorCode.INVALID_TYPE, f"{name} must be a string"
        )
    if "\x00" in value or value != value.strip():
        raise TransportGateError(
            AdmissionErrorCode.INVALID_TYPE,
            f"{name} must not contain NUL or surrounding whitespace",
        )
    if required and not value:
        raise TransportGateError(
            AdmissionErrorCode.MISSING_FIELD, f"{name} is required"
        )
    return value


def _normalize_transport(transport: TransportKind | str) -> str:
    if isinstance(transport, TransportKind):
        value = transport.value
    else:
        value = _require_text(transport, "transport")
    if value not in MIGRATED_TRANSPORTS:
        raise TransportGateError(
            AdmissionErrorCode.UNKNOWN_ENUM,
            f"unknown migrated transport={value!r}; expected one of {list(MIGRATED_TRANSPORTS)}",
        )
    return value


def argument_cid_for(arguments: Mapping[str, Any]) -> str:
    """Deterministic argument CID for exact-args binding."""
    if not isinstance(arguments, Mapping):
        raise TransportGateError(
            AdmissionErrorCode.INVALID_TYPE, "arguments must be a mapping"
        )
    return binding_cid("argument", dict(arguments))


def reject_untrusted_authority_selection(
    *,
    authority_source: str,
    overlay: Mapping[str, Any],
) -> None:
    """Fail closed when browser/model/peer/... try to select host authority."""
    source = (authority_source or "host").strip()
    if source not in UNTRUSTED_AUTHORITY_SOURCES and source != "host":
        # Unknown sources are treated as untrusted (fail closed).
        untrusted = True
    else:
        untrusted = source in UNTRUSTED_AUTHORITY_SOURCES

    if not untrusted:
        # Host path still rejects free-form authority keys in overlays.
        free = sorted(set(overlay) & FREE_FORM_AUTHORITY_KEYS)
        if free:
            raise TransportGateError(
                AdmissionErrorCode.FREE_FORM_AUTHORITY,
                f"free-form authority keys forbidden in overlay: {free}",
            )
        return

    # Untrusted sources may never select authority bindings.
    selected = sorted(set(overlay) & AUTHORITY_SELECTION_KEYS)
    if selected:
        raise TransportGateError(
            AdmissionErrorCode.FORBIDDEN_ISSUER
            if source in FORBIDDEN_TOKEN_ISSUERS | UNTRUSTED_AUTHORITY_SOURCES
            else AdmissionErrorCode.FREE_FORM_AUTHORITY,
            f"{source!r} inputs cannot select authority fields: {selected}",
        )
    free = sorted(set(overlay) & FREE_FORM_AUTHORITY_KEYS)
    if free:
        raise TransportGateError(
            AdmissionErrorCode.FREE_FORM_AUTHORITY,
            f"{source!r} free-form authority keys forbidden: {free}",
        )
    # Explicit issuer spoofing via overlay.
    issuer = overlay.get("issuer")
    if isinstance(issuer, str) and issuer and issuer != KERNEL_ISSUER:
        raise TransportGateError(
            AdmissionErrorCode.NON_KERNEL_TOKEN_ISSUER,
            f"{source!r} cannot set admission token issuer",
        )


@dataclass
class CommonTransportGate:
    """Host gate: one kernel unlock path for all migrated Accelerate transports."""

    kernel: EffectAdmissionKernel = field(default_factory=lambda: default_kernel(now_ms=0))
    _handlers: dict[str, _RegisteredHandler] = field(
        default_factory=dict, init=False, repr=False
    )
    _kernel_calls: list[KernelCallRecord] = field(
        default_factory=list, init=False, repr=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _total_handler_invocations: int = field(default=0, init=False, repr=False)

    def register_handler(
        self,
        spec: OperationSpecView | Mapping[str, Any],
        handler: HandlerFn,
    ) -> GatedHandler:
        """Register an effectful handler behind the common gate."""
        view = (
            spec
            if isinstance(spec, OperationSpecView)
            else OperationSpecView.from_mapping(spec)
        )
        if not callable(handler):
            raise TransportGateError(
                AdmissionErrorCode.INVALID_TYPE, "handler must be callable"
            )
        gated = GatedHandler(view.operation_id, handler, gate=self)
        with self._lock:
            self._handlers[view.operation_id] = _RegisteredHandler(
                spec=view, handler=handler, gated=gated
            )
        return gated

    def get_gated_handler(self, operation_id: str) -> GatedHandler:
        reg = self._require_registered(operation_id)
        assert reg.gated is not None
        return reg.gated

    def handler_invocation_count(self, operation_id: str | None = None) -> int:
        with self._lock:
            if operation_id is None:
                return self._total_handler_invocations
            reg = self._handlers.get(operation_id)
            return 0 if reg is None else reg.invocation_count

    def kernel_calls(self) -> tuple[KernelCallRecord, ...]:
        with self._lock:
            return tuple(self._kernel_calls)

    def clear_telemetry(self) -> None:
        with self._lock:
            self._kernel_calls.clear()
            self._total_handler_invocations = 0
            for reg in self._handlers.values():
                reg.invocation_count = 0

    def _require_registered(self, operation_id: str) -> _RegisteredHandler:
        op = _require_text(operation_id, "operation_id")
        with self._lock:
            reg = self._handlers.get(op)
        if reg is None:
            raise TransportGateError(
                AdmissionErrorCode.MISSING_FIELD,
                f"no handler registered for operation_id={op!r}",
            )
        return reg

    def _record_kernel_call(self, record: KernelCallRecord) -> None:
        with self._lock:
            self._kernel_calls.append(record)

    def _invoke_handler(
        self,
        reg: _RegisteredHandler,
        arguments: Mapping[str, Any],
        permit: _UnlockPermit,
    ) -> Any:
        _install_permit(permit)
        try:
            assert reg.gated is not None
            result = reg.gated(arguments)
            with self._lock:
                reg.invocation_count += 1
                self._total_handler_invocations += 1
            return result
        finally:
            _clear_permit()

    def dispatch(
        self,
        transport: TransportKind | str,
        request: TransportRequest | Mapping[str, Any],
    ) -> TransportResult:
        """Admit then (only if unlocked) invoke — shared by every transport."""
        transport_id = _normalize_transport(transport)
        req = (
            request
            if isinstance(request, TransportRequest)
            else TransportRequest(
                operation_id=str(request["operation_id"]),
                arguments=dict(request.get("arguments") or {}),
                typestate=str(request.get("typestate") or "Reserved"),
                token=request.get("token"),
                authority_source=str(request.get("authority_source") or "host"),
                transport_overlay=dict(request.get("transport_overlay") or {}),
                host_bindings=dict(request.get("host_bindings") or {}),
                consume_token=bool(request.get("consume_token", True)),
            )
        )

        # Authority selection from browser/model/peer/... fails before unlock.
        try:
            reject_untrusted_authority_selection(
                authority_source=req.authority_source,
                overlay=req.transport_overlay,
            )
        except TransportGateError as exc:
            decision = AdmissionDecision(
                verdict=AdmissionVerdict.DENY,
                code=exc.code,
                message=str(exc),
            )
            return TransportResult(
                outcome="Rejected",
                admitted=False,
                handler_invoked=False,
                decision=decision,
                kernel_call=None,
                transport=transport_id,
                operation_id=req.operation_id,
                argument_cid=argument_cid_for(req.arguments),
                code=exc.code.value,
                message=str(exc),
            )

        reg = self._require_registered(req.operation_id)
        argument_cid = argument_cid_for(req.arguments)

        # Effect-class match: request must address the registered OperationSpec.
        derived = derive_token_obligations(reg.spec)

        # Same kernel call for every migrated transport.
        decision = self.kernel.unlock_handler(
            spec=reg.spec,
            typestate=req.typestate,
            token=req.token,
            argument_cid=argument_cid,
            consume=req.consume_token,
        )
        token_id = ""
        if isinstance(req.token, AdmissionToken):
            token_id = req.token.token_id
        elif isinstance(req.token, Mapping):
            token_id = str(req.token.get("token_id") or "")

        call = KernelCallRecord(
            method=KERNEL_CALL,
            operation_id=reg.spec.operation_id,
            effect_class=reg.spec.effect_class,
            argument_cid=argument_cid,
            typestate=req.typestate,
            token_id=token_id or decision.token_id,
            transport=transport_id,
        )
        self._record_kernel_call(call)

        if decision.verdict is not AdmissionVerdict.ADMIT or not decision.unlocked:
            # Denial / unlock failure: zero handler invocations.
            return TransportResult(
                outcome="Rejected",
                admitted=False,
                handler_invoked=False,
                decision=decision,
                kernel_call=call,
                transport=transport_id,
                operation_id=reg.spec.operation_id,
                argument_cid=argument_cid,
                code=None if decision.code is None else decision.code.value,
                message=decision.message,
            )

        # Effect-class obligations must be covered (already checked by kernel);
        # retain derived set for evidence.
        _ = derived

        permit = _UnlockPermit(
            operation_id=reg.spec.operation_id,
            argument_cid=argument_cid,
            token_id=decision.token_id,
            transport=transport_id,
            permit_id=binding_cid(
                "permit",
                {
                    "token_id": decision.token_id,
                    "argument_cid": argument_cid,
                    "transport": transport_id,
                },
            ),
        )
        try:
            result = self._invoke_handler(reg, req.arguments, permit)
        except Exception as exc:  # noqa: BLE001 — map to typed Failed outcome
            return TransportResult(
                outcome="Failed",
                admitted=True,
                handler_invoked=True,
                decision=decision,
                kernel_call=call,
                transport=transport_id,
                operation_id=reg.spec.operation_id,
                argument_cid=argument_cid,
                code=AdmissionErrorCode.INVALID_TYPE.value,
                message=str(exc),
            )

        # Typed observation outcome (FACP closed outcome algebra).
        outcome = _coerce_observation_outcome(result)
        return TransportResult(
            outcome=outcome,
            admitted=True,
            handler_invoked=True,
            decision=decision,
            kernel_call=call,
            transport=transport_id,
            operation_id=reg.spec.operation_id,
            argument_cid=argument_cid,
            result=result,
            message="handler observed",
        )

    # -- Transport adapters: thin projections onto the same dispatch path -----

    def invoke_cli(self, request: TransportRequest | Mapping[str, Any]) -> TransportResult:
        return self.dispatch(TransportKind.CLI, request)

    def invoke_mcp(self, request: TransportRequest | Mapping[str, Any]) -> TransportResult:
        return self.dispatch(TransportKind.MCP, request)

    def invoke_mcp_plus_plus(
        self, request: TransportRequest | Mapping[str, Any]
    ) -> TransportResult:
        return self.dispatch(TransportKind.MCP_PLUS_PLUS, request)

    def invoke_python(
        self, request: TransportRequest | Mapping[str, Any]
    ) -> TransportResult:
        return self.dispatch(TransportKind.PYTHON, request)

    def invoke_all_migrated(
        self, request: TransportRequest | Mapping[str, Any]
    ) -> dict[str, TransportResult]:
        """Run the same request through every migrated transport adapter."""
        return {
            kind: self.dispatch(kind, request) for kind in MIGRATED_TRANSPORTS
        }


def _coerce_observation_outcome(result: Any) -> str:
    if isinstance(result, Mapping):
        claimed = result.get("outcome")
        if isinstance(claimed, str) and claimed in CLOSED_OUTCOMES:
            return claimed
        # Forbidden success booleans never promote to Observed.
        if result.get("success") is True or result.get("ok") is True:
            return "Unknown"
    return "Observed"


def default_transport_gate(
    *,
    now_ms: int = 0,
    kernel: EffectAdmissionKernel | None = None,
) -> CommonTransportGate:
    """Construct a hermetic gate bound to a fixed-clock kernel."""
    return CommonTransportGate(kernel=kernel or default_kernel(now_ms=now_ms))


def migrated_transports() -> tuple[str, ...]:
    return MIGRATED_TRANSPORTS


def same_kernel_call(left: KernelCallRecord, right: KernelCallRecord) -> bool:
    """True when two transports exercised the identical kernel unlock identity."""
    return left.identity_without_transport() == right.identity_without_transport()


__all__ = (
    "SCHEMA",
    "DECISION_SCHEMA",
    "TASK_ID",
    "GOAL_ID",
    "BUNDLE",
    "INTERFACE",
    "KERNEL_CALL",
    "UNSAFE_PROMOTION",
    "EVIDENCE_SUBSET",
    "MIGRATED_TRANSPORTS",
    "CLOSED_OUTCOMES",
    "UNTRUSTED_AUTHORITY_SOURCES",
    "AUTHORITY_SELECTION_KEYS",
    "INVENTORIED_TRANSPORT_SEAMS",
    "TransportKind",
    "TransportGateError",
    "HandlerNotUnlockedError",
    "KernelCallRecord",
    "TransportRequest",
    "TransportResult",
    "GatedHandler",
    "CommonTransportGate",
    "argument_cid_for",
    "reject_untrusted_authority_selection",
    "default_transport_gate",
    "migrated_transports",
    "same_kernel_call",
)
