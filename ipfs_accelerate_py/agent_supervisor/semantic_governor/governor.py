"""Compose the required SemanticCompressionGovernor public APIs (SCG-036).

This module is the stable accelerate-side facade over the ten plan-required
entry points:

* datasets analysis: ``evaluate_context_sufficiency``, ``diagnose_omission``,
  ``plan_context_expansion``, ``update_calibration``, ``propose_rule_change``
* accelerate runtime: ``create_shadow_plan``, ``compare_shadow_results``,
  ``execute_expansion_loop``, ``evaluate_rule_candidate``,
  ``promote_compression_policy``

Composition is lazy and dependency-injectable. Importing this module performs
no I/O, starts no processes or network activity, and does not load optional
providers or installers. Module-level API names resolve to the same callable
objects as their owning leaf modules so safety and identity gates cannot be
bypassed by going through the facade.

Unknown commands and unknown mapping/parameter fields fail closed.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping, Sequence

# ---------------------------------------------------------------------------
# Evidence / interface / schema pins
# ---------------------------------------------------------------------------

SCG_PUBLIC_API_EVIDENCE: Final[str] = "scg/public-api@1"
SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE: Final[str] = "SemanticCompressionGovernor@1"
SEMANTIC_COMPRESSION_GOVERNOR_SCHEMA: Final[str] = (
    "ipfs-accelerate.semantic-compression-governor-public-api@1"
)
SEMANTIC_COMPRESSION_GOVERNOR_PACKAGE_INTERFACE: Final[str] = (
    "SemanticCompressionGovernorPublicApi@1"
)

# Ten plan-required module-level APIs (order is stable for evidence dumps).
REQUIRED_PUBLIC_APIS: Final[tuple[str, ...]] = (
    "evaluate_context_sufficiency",
    "create_shadow_plan",
    "compare_shadow_results",
    "diagnose_omission",
    "plan_context_expansion",
    "execute_expansion_loop",
    "update_calibration",
    "propose_rule_change",
    "evaluate_rule_candidate",
    "promote_compression_policy",
)

# Closed command vocabulary for :meth:`SemanticCompressionGovernor.invoke`.
# Identical to the ten required APIs (CLI renames live in SCG-037).
REQUIRED_COMMANDS: Final[tuple[str, ...]] = REQUIRED_PUBLIC_APIS

# Closed top-level fields for mapping-form invoke envelopes.
_INVOKE_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {"command", "args", "kwargs", "arguments"}
)

# name -> (import module path, attribute)
_API_OWNERS: Final[dict[str, tuple[str, str]]] = {
    "evaluate_context_sufficiency": (
        "ipfs_datasets_py.logic.software_contracts.semantic_governor",
        "evaluate_context_sufficiency",
    ),
    "create_shadow_plan": (
        "ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow_plan",
        "create_shadow_plan",
    ),
    "compare_shadow_results": (
        "ipfs_accelerate_py.agent_supervisor.semantic_governor.differential",
        "compare_shadow_results",
    ),
    "diagnose_omission": (
        "ipfs_datasets_py.logic.software_contracts.semantic_governor",
        "diagnose_omission",
    ),
    "plan_context_expansion": (
        "ipfs_datasets_py.logic.software_contracts.semantic_governor",
        "plan_context_expansion",
    ),
    "execute_expansion_loop": (
        "ipfs_accelerate_py.agent_supervisor.semantic_governor.expansion_loop",
        "execute_expansion_loop",
    ),
    "update_calibration": (
        "ipfs_datasets_py.logic.software_contracts.semantic_governor",
        "update_calibration",
    ),
    "propose_rule_change": (
        "ipfs_datasets_py.logic.software_contracts.semantic_governor",
        "propose_rule_change",
    ),
    "evaluate_rule_candidate": (
        "ipfs_accelerate_py.agent_supervisor.semantic_governor.policy_evaluation",
        "evaluate_rule_candidate",
    ),
    "promote_compression_policy": (
        "ipfs_accelerate_py.agent_supervisor.semantic_governor.promotion",
        "promote_compression_policy",
    ),
}

# Per-API interface pins (stable evidence labels).
_API_INTERFACE_IDS: Final[dict[str, str]] = {
    "evaluate_context_sufficiency": "evaluate_context_sufficiency@1",
    "create_shadow_plan": "create_shadow_plan@1",
    "compare_shadow_results": "compare_shadow_results@1",
    "diagnose_omission": "diagnose_omission@1",
    "plan_context_expansion": "plan_context_expansion@1",
    "execute_expansion_loop": "execute_expansion_loop@1",
    "update_calibration": "update_calibration@1",
    "propose_rule_change": "propose_rule_change@1",
    "evaluate_rule_candidate": "evaluate_rule_candidate@1",
    "promote_compression_policy": "promote_compression_policy@1",
}

if frozenset(_API_OWNERS) != frozenset(REQUIRED_PUBLIC_APIS):
    raise RuntimeError("REQUIRED_PUBLIC_APIS and _API_OWNERS must match exactly")
if frozenset(_API_INTERFACE_IDS) != frozenset(REQUIRED_PUBLIC_APIS):
    raise RuntimeError("REQUIRED_PUBLIC_APIS and _API_INTERFACE_IDS must match exactly")


# ---------------------------------------------------------------------------
# Errors / typed unavailable
# ---------------------------------------------------------------------------


class GovernorPublicApiError(ValueError):
    """Base error for the public SemanticCompressionGovernor facade."""

    def __init__(self, message: str, *, reason_code: str = "public_api_error") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class UnknownCommandError(GovernorPublicApiError):
    """Raised when an invoke command is outside the closed vocabulary."""

    def __init__(self, command: str) -> None:
        super().__init__(
            f"unknown command: {command!r}; allowed={list(REQUIRED_COMMANDS)}",
            reason_code="unknown_command",
        )
        self.command = command


class UnknownFieldError(GovernorPublicApiError):
    """Raised when a closed mapping or parameter set contains unknown fields."""

    def __init__(
        self,
        fields: Sequence[str],
        *,
        context: str = "payload",
    ) -> None:
        ordered = tuple(sorted(str(item) for item in fields))
        super().__init__(
            f"{context} has unknown fields: {list(ordered)}",
            reason_code="unknown_field",
        )
        self.fields = ordered
        self.context = context


class GovernorApiUnavailableError(GovernorPublicApiError):
    """Raised when a required API surface is typed-unavailable."""

    def __init__(
        self,
        command: str,
        *,
        reason_code: str = "api_unavailable",
        diagnostic: str | None = None,
        status: str = "unavailable",
    ) -> None:
        message = diagnostic or f"required API {command!r} is unavailable"
        super().__init__(message, reason_code=reason_code)
        self.command = command
        self.status = status
        self.diagnostic = message


class ApiAvailability(str, Enum):
    """Closed availability vocabulary for typed public-API probes."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class GovernorApiUnavailableResult:
    """Typed unavailable result for a required public API.

    Used when a dependency cannot be loaded or an injected surface reports
    unavailability. Never silently upgraded into a success artifact.
    """

    command: str
    status: str = ApiAvailability.UNAVAILABLE.value
    reason_code: str = "api_unavailable"
    diagnostic: str | None = None
    interface_id: str = SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE
    evidence_id: str = SCG_PUBLIC_API_EVIDENCE
    api_interface_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", str(self.command))
        object.__setattr__(
            self,
            "status",
            str(self.status or ApiAvailability.UNAVAILABLE.value),
        )
        object.__setattr__(self, "reason_code", str(self.reason_code))
        if self.diagnostic is not None:
            object.__setattr__(self, "diagnostic", str(self.diagnostic))
        object.__setattr__(self, "interface_id", str(self.interface_id))
        object.__setattr__(self, "evidence_id", str(self.evidence_id))
        if self.api_interface_id is not None:
            object.__setattr__(self, "api_interface_id", str(self.api_interface_id))
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", MappingProxyType(dict(meta)))

    @property
    def available(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "status": self.status,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "interface_id": self.interface_id,
            "evidence_id": self.evidence_id,
            "api_interface_id": self.api_interface_id,
            "available": False,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Lazy API resolution
# ---------------------------------------------------------------------------


def _load_api(name: str) -> Callable[..., Any]:
    """Import and return the owning leaf implementation of a required API."""

    owner = _API_OWNERS.get(name)
    if owner is None:
        raise UnknownCommandError(name)
    module_path, attr = owner
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise GovernorApiUnavailableError(
            name,
            reason_code="import_failed",
            diagnostic=f"failed to import {module_path!r}: {exc}",
            status=ApiAvailability.MISSING.value,
        ) from exc
    try:
        value = getattr(module, attr)
    except AttributeError as exc:
        raise GovernorApiUnavailableError(
            name,
            reason_code="missing_export",
            diagnostic=f"{module_path!r} has no attribute {attr!r}",
            status=ApiAvailability.MISSING.value,
        ) from exc
    if not callable(value):
        raise GovernorApiUnavailableError(
            name,
            reason_code="not_callable",
            diagnostic=f"{module_path}.{attr} is not callable",
            status=ApiAvailability.INCOMPATIBLE.value,
        )
    return value


def _reject_unknown_params(
    fn: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Reject kwargs that are not parameters of ``fn`` (closed field set)."""

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # Callables without introspectable signatures: leave checks to callee.
        return

    # parameters that accept **kwargs mean the callee owns closed-field checks.
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return

    try:
        signature.bind_partial(*args, **dict(kwargs))
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" in message:
            unknown: list[str] = []
            for key in kwargs:
                try:
                    signature.bind_partial(*args, **{key: kwargs[key]})
                except TypeError:
                    unknown.append(key)
            if unknown:
                raise UnknownFieldError(unknown, context=context) from exc
        raise GovernorPublicApiError(
            f"{context} rejected parameters: {message}",
            reason_code="invalid_parameters",
        ) from exc


def _closed_mapping(
    value: Mapping[str, Any] | None,
    allowed: frozenset[str],
    *,
    name: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise GovernorPublicApiError(
            f"{name} must be a mapping",
            reason_code="invalid_mapping",
        )
    data = dict(value)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise UnknownFieldError(unknown, context=name)
    return data


# ---------------------------------------------------------------------------
# Module-level helpers / pins
# ---------------------------------------------------------------------------


def public_api_evidence_id() -> str:
    """Return the public-API evidence pin."""

    return SCG_PUBLIC_API_EVIDENCE


def public_api_interface_id() -> str:
    """Return the versioned package public-API interface pin."""

    return SEMANTIC_COMPRESSION_GOVERNOR_PACKAGE_INTERFACE


def public_api_schema() -> str:
    """Return the public API schema identifier."""

    return SEMANTIC_COMPRESSION_GOVERNOR_SCHEMA


def governor_interface_id() -> str:
    """Return the SemanticCompressionGovernor class interface pin."""

    return SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE


def required_public_apis() -> tuple[str, ...]:
    """Return the closed primary public entry-point names."""

    return REQUIRED_PUBLIC_APIS


def required_commands() -> tuple[str, ...]:
    """Return the closed invoke-command vocabulary."""

    return REQUIRED_COMMANDS


def api_interface_id(name: str) -> str:
    """Return the stable interface id for one required public API."""

    if name not in _API_INTERFACE_IDS:
        raise UnknownCommandError(name)
    return _API_INTERFACE_IDS[name]


def api_interface_ids() -> Mapping[str, str]:
    """Return the closed mapping of required API name → interface id."""

    return MappingProxyType(dict(_API_INTERFACE_IDS))


def resolve_public_api(name: str) -> Callable[..., Any]:
    """Resolve a required public API callable (lazy leaf import).

    Returns the exact leaf callable so signatures, return types, and identity
    gates match the owning implementation.
    """

    if name not in REQUIRED_PUBLIC_APIS:
        raise UnknownCommandError(name)
    cached = globals().get(name)
    if cached is not None and callable(cached):
        owner = _API_OWNERS[name]
        # Cached module-level re-export of the leaf function.
        if getattr(cached, "__name__", None) == owner[1]:
            return cached  # type: ignore[return-value]
    value = _load_api(name)
    globals()[name] = value
    return value


# ---------------------------------------------------------------------------
# Module-level required APIs (exact leaf identities via __getattr__)
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Resolve required public API callables from their owning modules."""

    if name not in _API_OWNERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = _load_api(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# ---------------------------------------------------------------------------
# SemanticCompressionGovernor composition
# ---------------------------------------------------------------------------


@dataclass
class SemanticCompressionGovernor:
    """Lazy composition facade over the ten required public APIs.

    Each required API may be dependency-injected for tests or alternate
    surfaces. When omitted, the owning leaf implementation is loaded on first
    use. Safety, identity, and closed-field gates remain those of the leaf
    implementations; this class never weakens them.
    """

    evaluate_context_sufficiency_fn: Callable[..., Any] | None = None
    create_shadow_plan_fn: Callable[..., Any] | None = None
    compare_shadow_results_fn: Callable[..., Any] | None = None
    diagnose_omission_fn: Callable[..., Any] | None = None
    plan_context_expansion_fn: Callable[..., Any] | None = None
    execute_expansion_loop_fn: Callable[..., Any] | None = None
    update_calibration_fn: Callable[..., Any] | None = None
    propose_rule_change_fn: Callable[..., Any] | None = None
    evaluate_rule_candidate_fn: Callable[..., Any] | None = None
    promote_compression_policy_fn: Callable[..., Any] | None = None
    # Optional runtime collaborators (DI only; never auto-started).
    datasets_adapter: Any | None = None
    seal_adapter: Any | None = None
    runtime: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", MappingProxyType(dict(meta)))

    # -- identity -----------------------------------------------------------

    @property
    def interface_id(self) -> str:
        return SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE

    @property
    def schema(self) -> str:
        return SEMANTIC_COMPRESSION_GOVERNOR_SCHEMA

    @property
    def evidence_id(self) -> str:
        return SCG_PUBLIC_API_EVIDENCE

    def required_public_apis(self) -> tuple[str, ...]:
        return REQUIRED_PUBLIC_APIS

    def required_commands(self) -> tuple[str, ...]:
        return REQUIRED_COMMANDS

    def runtime_view(self) -> Mapping[str, Any]:
        """Bounded, deterministic projection of the facade identity."""

        return MappingProxyType(
            {
                "interface_id": self.interface_id,
                "schema": self.schema,
                "evidence_id": self.evidence_id,
                "required_public_apis": list(REQUIRED_PUBLIC_APIS),
                "required_commands": list(REQUIRED_COMMANDS),
                "api_interface_ids": dict(_API_INTERFACE_IDS),
                "has_datasets_adapter": self.datasets_adapter is not None,
                "has_seal_adapter": self.seal_adapter is not None,
                "has_runtime": self.runtime is not None,
                "metadata": dict(self.metadata),
            }
        )

    # -- resolution ---------------------------------------------------------

    def _override_for(self, name: str) -> Callable[..., Any] | None:
        mapping = {
            "evaluate_context_sufficiency": self.evaluate_context_sufficiency_fn,
            "create_shadow_plan": self.create_shadow_plan_fn,
            "compare_shadow_results": self.compare_shadow_results_fn,
            "diagnose_omission": self.diagnose_omission_fn,
            "plan_context_expansion": self.plan_context_expansion_fn,
            "execute_expansion_loop": self.execute_expansion_loop_fn,
            "update_calibration": self.update_calibration_fn,
            "propose_rule_change": self.propose_rule_change_fn,
            "evaluate_rule_candidate": self.evaluate_rule_candidate_fn,
            "promote_compression_policy": self.promote_compression_policy_fn,
        }
        return mapping.get(name)

    def resolve(self, name: str) -> Callable[..., Any]:
        """Resolve one required API (injected override or lazy leaf)."""

        if name not in REQUIRED_PUBLIC_APIS:
            raise UnknownCommandError(name)
        override = self._override_for(name)
        if override is not None:
            if not callable(override):
                raise GovernorApiUnavailableError(
                    name,
                    reason_code="not_callable",
                    diagnostic=f"injected {name!r} is not callable",
                    status=ApiAvailability.INCOMPATIBLE.value,
                )
            return override
        # Prefer datasets adapter for analysis APIs when injected.
        if self.datasets_adapter is not None and name in {
            "evaluate_context_sufficiency",
            "diagnose_omission",
            "plan_context_expansion",
            "update_calibration",
            "propose_rule_change",
        }:
            api = getattr(self.datasets_adapter, "api", None)
            if callable(api):
                try:
                    return api(name)
                except Exception as exc:  # noqa: BLE001 — surface as typed unavailable
                    raise GovernorApiUnavailableError(
                        name,
                        reason_code="adapter_unavailable",
                        diagnostic=str(exc),
                        status=ApiAvailability.UNAVAILABLE.value,
                    ) from exc
            method = getattr(self.datasets_adapter, name, None)
            if callable(method):
                return method
        return resolve_public_api(name)

    def probe_api(self, name: str) -> Mapping[str, Any]:
        """Probe availability of one required API without invoking it."""

        if name not in REQUIRED_PUBLIC_APIS:
            raise UnknownCommandError(name)
        try:
            fn = self.resolve(name)
        except GovernorApiUnavailableError as exc:
            return MappingProxyType(
                GovernorApiUnavailableResult(
                    command=name,
                    status=exc.status,
                    reason_code=exc.reason_code,
                    diagnostic=exc.diagnostic,
                    api_interface_id=_API_INTERFACE_IDS[name],
                ).to_dict()
            )
        return MappingProxyType(
            {
                "command": name,
                "status": ApiAvailability.AVAILABLE.value,
                "available": True,
                "reason_code": None,
                "diagnostic": None,
                "interface_id": SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE,
                "evidence_id": SCG_PUBLIC_API_EVIDENCE,
                "api_interface_id": _API_INTERFACE_IDS[name],
                "callable": True,
                "module": getattr(fn, "__module__", None),
                "qualname": getattr(fn, "__qualname__", getattr(fn, "__name__", None)),
            }
        )

    # -- required API methods -----------------------------------------------

    def evaluate_context_sufficiency(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("evaluate_context_sufficiency", args, kwargs)

    def create_shadow_plan(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("create_shadow_plan", args, kwargs)

    def compare_shadow_results(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("compare_shadow_results", args, kwargs)

    def diagnose_omission(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("diagnose_omission", args, kwargs)

    def plan_context_expansion(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("plan_context_expansion", args, kwargs)

    def execute_expansion_loop(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("execute_expansion_loop", args, kwargs)

    def update_calibration(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("update_calibration", args, kwargs)

    def propose_rule_change(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("propose_rule_change", args, kwargs)

    def evaluate_rule_candidate(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("evaluate_rule_candidate", args, kwargs)

    def promote_compression_policy(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("promote_compression_policy", args, kwargs)

    def _call(
        self,
        name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        fn = self.resolve(name)
        _reject_unknown_params(fn, args, kwargs, context=name)
        return fn(*args, **dict(kwargs))

    # -- closed command dispatch --------------------------------------------

    def invoke(self, command: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch a closed command to the matching required public API.

        Unknown commands and unexpected keyword fields are rejected.
        """

        if command not in REQUIRED_COMMANDS:
            raise UnknownCommandError(command)
        return self._call(command, args, kwargs)

    def invoke_envelope(self, payload: Mapping[str, Any]) -> Any:
        """Dispatch from a closed mapping envelope.

        Allowed fields: ``command`` (required), ``args``, ``kwargs`` /
        ``arguments`` (optional). Any other top-level field is rejected.
        """

        data = _closed_mapping(payload, _INVOKE_ENVELOPE_FIELDS, name="invoke_envelope")
        if "command" not in data:
            raise GovernorPublicApiError(
                "invoke_envelope requires 'command'",
                reason_code="missing_command",
            )
        command = data["command"]
        if not isinstance(command, str):
            raise GovernorPublicApiError(
                "command must be a string",
                reason_code="invalid_command",
            )
        if command not in REQUIRED_COMMANDS:
            raise UnknownCommandError(command)

        raw_args = data.get("args", ())
        if raw_args is None:
            raw_args = ()
        if not isinstance(raw_args, Sequence) or isinstance(raw_args, (str, bytes)):
            raise GovernorPublicApiError(
                "args must be a sequence",
                reason_code="invalid_args",
            )

        if "kwargs" in data and "arguments" in data:
            raise GovernorPublicApiError(
                "invoke_envelope accepts only one of 'kwargs' or 'arguments'",
                reason_code="conflicting_fields",
            )
        raw_kwargs = data.get("kwargs", data.get("arguments", {}))
        if raw_kwargs is None:
            raw_kwargs = {}
        if not isinstance(raw_kwargs, Mapping):
            raise GovernorPublicApiError(
                "kwargs/arguments must be a mapping",
                reason_code="invalid_kwargs",
            )
        return self.invoke(command, *tuple(raw_args), **dict(raw_kwargs))


def create_semantic_compression_governor(
    **dependencies: Any,
) -> SemanticCompressionGovernor:
    """Construct a :class:`SemanticCompressionGovernor` with optional DI.

    Unknown dependency field names are rejected (closed constructor surface).
    """

    allowed = frozenset(
        {
            "evaluate_context_sufficiency_fn",
            "create_shadow_plan_fn",
            "compare_shadow_results_fn",
            "diagnose_omission_fn",
            "plan_context_expansion_fn",
            "execute_expansion_loop_fn",
            "update_calibration_fn",
            "propose_rule_change_fn",
            "evaluate_rule_candidate_fn",
            "promote_compression_policy_fn",
            "datasets_adapter",
            "seal_adapter",
            "runtime",
            "metadata",
        }
    )
    unknown = sorted(set(dependencies) - allowed)
    if unknown:
        raise UnknownFieldError(unknown, context="create_semantic_compression_governor")
    return SemanticCompressionGovernor(**dependencies)


# ---------------------------------------------------------------------------
# Module-level invoke helpers (stateless default governor)
# ---------------------------------------------------------------------------


def invoke(command: str, *args: Any, **kwargs: Any) -> Any:
    """Dispatch a closed command through a default governor instance."""

    return SemanticCompressionGovernor().invoke(command, *args, **kwargs)


def invoke_envelope(payload: Mapping[str, Any]) -> Any:
    """Dispatch a closed mapping envelope through a default governor instance."""

    return SemanticCompressionGovernor().invoke_envelope(payload)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "SCG_PUBLIC_API_EVIDENCE",
    "SEMANTIC_COMPRESSION_GOVERNOR_INTERFACE",
    "SEMANTIC_COMPRESSION_GOVERNOR_SCHEMA",
    "SEMANTIC_COMPRESSION_GOVERNOR_PACKAGE_INTERFACE",
    "REQUIRED_PUBLIC_APIS",
    "REQUIRED_COMMANDS",
    "ApiAvailability",
    "GovernorApiUnavailableError",
    "GovernorApiUnavailableResult",
    "GovernorPublicApiError",
    "SemanticCompressionGovernor",
    "UnknownCommandError",
    "UnknownFieldError",
    "api_interface_id",
    "api_interface_ids",
    "compare_shadow_results",
    "create_semantic_compression_governor",
    "create_shadow_plan",
    "diagnose_omission",
    "evaluate_context_sufficiency",
    "evaluate_rule_candidate",
    "execute_expansion_loop",
    "governor_interface_id",
    "invoke",
    "invoke_envelope",
    "plan_context_expansion",
    "promote_compression_policy",
    "propose_rule_change",
    "public_api_evidence_id",
    "public_api_interface_id",
    "public_api_schema",
    "required_commands",
    "required_public_apis",
    "resolve_public_api",
    "update_calibration",
]
