"""DCR-001: fail-closed execution authority for deterministic repair.

This module is intentionally independent of provider, retry, rescue, and
self-improvement packages.  A repair runtime can invoke work only through a
``DeterministicRepairAuthorityPolicy``.  Every other route is denied before a
callback is evaluated, and the process-wide guard retains a small audit
counter for the denial.

The policy is an execution boundary, not a classifier: an abstention or a
deferral is a typed result and never a request to try another (possibly model
backed) route.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock, local
from typing import Any, ClassVar, Final, TypeVar
from urllib.parse import urlparse

NO_LLM_EXECUTION_GUARD_INTERFACE: Final[str] = "NoLlmExecutionGuard@1"
DETERMINISTIC_REPAIR_AUTHORITY_POLICY_INTERFACE: Final[str] = "DeterministicRepairAuthorityPolicy@1"
DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-authority-policy@1"
)
TARGET_REPAIR_RUNTIME: Final[str] = "target_repair_runtime"

_T = TypeVar("_T")

_FORBIDDEN_MODEL_PROVIDER_MODULES: Final[frozenset[str]] = frozenset(
    {
        "anthropic",
        "azure.ai.inference",
        "azure.ai.openai",
        "cohere",
        "google.genai",
        "google.generativeai",
        "groq",
        "huggingface_hub",
        "langchain",
        "langchain_openai",
        "litellm",
        "mistralai",
        "ollama",
        "openai",
        "transformers",
        "vertexai",
        "vllm",
    }
)
_SUBPROCESS_AUDIT_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "os.exec",
        "os.fork",
        "os.posix_spawn",
        "os.spawn",
        "os.system",
        "subprocess.Popen",
    }
)
_SOCKET_AUDIT_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostname",
        "socket.getnameinfo",
    }
)
_DYNAMIC_EXEC_AUDIT_EVENT: Final[str] = "exec"


class RepairExecutionRoute(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """The only route identities understood by the deterministic runtime."""

    DETERMINISTIC_LOCAL_LOGIC = "deterministic_local_logic"
    PROVER_SUBPROCESS = "prover_subprocess"
    LOOPBACK_MCP = "loopback_mcp"


class RepairAuthorityDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed, non-fallback outcomes from the execution boundary."""

    ALLOWED = "allowed"
    ABSTAIN = "abstain"
    DEFER = "defer"
    DENIED = "denied"


class NoLlmExecutionDenied(PermissionError):
    """A forbidden route was rejected before it could be invoked."""

    def __init__(self, *, route: str, reason: str) -> None:
        self.route = route
        self.reason = reason
        self.disposition = RepairAuthorityDisposition.DENIED
        super().__init__(f"no-LLM deterministic repair denied route {route!r}: {reason}")


# Compatibility spelling for consumers that use a policy-oriented error name.
DeterministicRepairAuthorityDenied = NoLlmExecutionDenied


@dataclass(frozen=True)
class RepairAuthorityDecision:
    """An auditable decision; only ``ALLOWED`` permits callback execution."""

    disposition: RepairAuthorityDisposition
    route: str
    reason: str
    pin: str = ""
    invoked: bool = False

    @property
    def allowed(self) -> bool:
        return self.disposition is RepairAuthorityDisposition.ALLOWED

    @property
    def terminal(self) -> bool:
        return not self.allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "route": self.route,
            "reason": self.reason,
            "pin": self.pin,
            "invoked": self.invoked,
            "fallback_authorized": False,
        }


def _route_text(value: RepairExecutionRoute | str) -> str:
    if isinstance(value, RepairExecutionRoute):
        return value.value
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _route_key(value: RepairExecutionRoute | str) -> str:
    text = _route_text(value)
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def _pin_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _is_loopback(endpoint: object) -> bool:
    if not isinstance(endpoint, str) or not endpoint.strip():
        return False
    parsed = urlparse(endpoint.strip())
    # URL parsing returns no hostname for bare ``localhost:port``; that is not
    # an explicit network endpoint and must fail closed.
    if parsed.scheme not in {"http", "https"}:
        return False
    # ``http://user@127.0.0.1`` is not an acceptable local authority.  Reject
    # credentials explicitly instead of relying on URL rendering or a client
    # implementation to preserve the parsed hostname.
    if parsed.username is not None or parsed.password is not None:
        return False
    try:
        _port = parsed.port
    except ValueError:
        return False
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _loopback_port(endpoint: str) -> int:
    parsed = urlparse(endpoint)
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def _is_forbidden_model_provider_import(value: object) -> bool:
    if not isinstance(value, str):
        return False
    name = value.strip().lower()
    return any(
        name == module or name.startswith(module + ".")
        for module in _FORBIDDEN_MODEL_PROVIDER_MODULES
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _AuditRouteContext:
    """Thread-local authority bound to one policy ``invoke`` callback."""

    route: str
    pin: str
    endpoint: str
    loopback_port: int | None


class NoLlmExecutionGuard:
    """A nest-safe, process-wide pre-invocation guard.

    The active depth and denial counters live on the class, rather than a
    policy instance, so a nested repair scope cannot create an unguarded gap.
    Unknown routes also deny by default.
    """

    INTERFACE: ClassVar[Final[str]] = NO_LLM_EXECUTION_GUARD_INTERFACE
    _lock: ClassVar[Final[RLock]] = RLock()
    _active_depth: ClassVar[int] = 0
    _denied_by_reason: ClassVar[Counter[str]] = Counter()
    _denied_by_route: ClassVar[Counter[str]] = Counter()
    _allowed_by_route: ClassVar[Counter[str]] = Counter()
    _thread_state: ClassVar[local] = local()
    _audit_hook_installed: ClassVar[bool] = False
    _provider_import_finder_installed: ClassVar[bool] = False

    def __init__(self, *, runtime: str = TARGET_REPAIR_RUNTIME) -> None:
        self.runtime = runtime
        self._local_depth = 0

    def __enter__(self) -> NoLlmExecutionGuard:
        type(self)._ensure_audit_hook()
        with self._lock:
            type(self)._active_depth += 1
            self._local_depth += 1
        type(self)._thread_state.guard_depth = type(self)._thread_guard_depth() + 1
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        with self._lock:
            if self._local_depth <= 0:
                raise RuntimeError("NoLlmExecutionGuard exit without enter")
            self._local_depth -= 1
            type(self)._active_depth -= 1
        depth = type(self)._thread_guard_depth()
        if depth <= 0:
            raise RuntimeError("NoLlmExecutionGuard thread exit without enter")
        type(self)._thread_state.guard_depth = depth - 1
        return False

    @classmethod
    def active(cls) -> bool:
        with cls._lock:
            return cls._active_depth > 0

    @classmethod
    def audit_snapshot(cls) -> dict[str, Any]:
        """Return process-wide counters without resetting evidence."""

        with cls._lock:
            denied_by_reason = dict(sorted(cls._denied_by_reason.items()))
            denied_by_route = dict(sorted(cls._denied_by_route.items()))
            allowed_by_route = dict(sorted(cls._allowed_by_route.items()))
            return {
                "interface": cls.INTERFACE,
                "runtime": TARGET_REPAIR_RUNTIME,
                "active_depth": cls._active_depth,
                "denied_total": sum(denied_by_reason.values()),
                "denied_by_reason": denied_by_reason,
                "denied_by_route": denied_by_route,
                "allowed_by_route": allowed_by_route,
            }

    # ``denied_counters`` is a compact name for callers that export metrics.
    denied_counters = audit_snapshot

    @classmethod
    def reset_audit_for_testing(cls) -> None:
        """Clear counters only for isolated tests; never use in production."""

        with cls._lock:
            if cls._active_depth:
                raise RuntimeError("cannot reset no-LLM audit while guard is active")
            cls._denied_by_reason.clear()
            cls._denied_by_route.clear()
            cls._allowed_by_route.clear()

    @classmethod
    def _record_denial(cls, *, route: str, reason: str) -> None:
        with cls._lock:
            cls._denied_by_route[route or "<invalid>"] += 1
            cls._denied_by_reason[reason] += 1

    @classmethod
    def _record_allowed(cls, *, route: str) -> None:
        with cls._lock:
            cls._allowed_by_route[route] += 1

    @classmethod
    def _ensure_audit_hook(cls) -> None:
        """Install one inert-until-guarded audit hook for this process."""

        with cls._lock:
            if cls._audit_hook_installed:
                return
            sys.addaudithook(cls._audit_hook)
            cls._audit_hook_installed = True
            # The ``import`` audit event covers built-in ``__import__``.  A
            # meta-path observer closes the equivalent ``importlib`` route
            # before module execution, since CPython does not audit that call.
            if not cls._provider_import_finder_installed:
                sys.meta_path.insert(0, _ForbiddenProviderImportFinder())
                cls._provider_import_finder_installed = True

    @classmethod
    def _thread_guard_depth(cls) -> int:
        return int(getattr(cls._thread_state, "guard_depth", 0))

    @classmethod
    def _route_stack(cls) -> list[_AuditRouteContext]:
        stack = getattr(cls._thread_state, "route_stack", None)
        if stack is None:
            stack = []
            cls._thread_state.route_stack = stack
        return stack

    @classmethod
    def _active_route_context(cls) -> _AuditRouteContext | None:
        stack = cls._route_stack()
        return stack[-1] if stack else None

    @classmethod
    @contextmanager
    def authorized_route(
        cls,
        decision: RepairAuthorityDecision,
        *,
        endpoint: str = "",
    ) -> Iterator[None]:
        """Bind an admitted route to this thread for exactly one callback.

        A child thread does not inherit this context.  Since the process-wide
        guard remains active, its audit events fail closed rather than gaining
        the parent callback's authority.
        """

        if cls._thread_guard_depth() <= 0:
            cls.deny(decision.route, "route_context_outside_guard")
        loopback_port = _loopback_port(endpoint) if endpoint else None
        context = _AuditRouteContext(
            route=decision.route,
            pin=decision.pin,
            endpoint=endpoint,
            loopback_port=loopback_port,
        )
        stack = cls._route_stack()
        stack.append(context)
        try:
            yield
        finally:
            if not stack or stack[-1] is not context:
                raise RuntimeError("no-LLM route context stack is corrupt")
            stack.pop()

    @classmethod
    def _audit_hook(cls, event: str, arguments: tuple[Any, ...]) -> None:
        """Deny side effects before they execute while any guard is active."""

        if not cls.active():
            return
        context = cls._active_route_context()
        if context is None or cls._thread_guard_depth() <= 0:
            cls.deny("<unguarded_thread>", "unguarded_thread_audit_event")
        if event == "import":
            cls._audit_provider_import(arguments[0] if arguments else None)
        if event in _SUBPROCESS_AUDIT_EVENTS:
            if context.route == RepairExecutionRoute.PROVER_SUBPROCESS.value:
                # A capability pin alone is not an executable hash/path
                # binding.  Until that receipt exists, the safe typed outcome
                # is defer; executing an ambiguous prover command is denied.
                cls.deny(context.route, "prover_executable_binding_unavailable_defer")
            cls.deny(context.route, "subprocess_forbidden_for_route")
        if (
            event == _DYNAMIC_EXEC_AUDIT_EVENT
            and context.route == RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC.value
        ):
            cls.deny(context.route, "dynamic_exec_forbidden_for_route")
        if event in _SOCKET_AUDIT_EVENTS:
            cls._authorize_socket_audit_event(event, arguments, context)

    @classmethod
    def _audit_provider_import(cls, module_name: object) -> None:
        if _is_forbidden_model_provider_import(module_name):
            context = cls._active_route_context()
            if context is None or cls._thread_guard_depth() <= 0:
                cls.deny("<unguarded_thread>", "unguarded_thread_audit_event")
            cls.deny(context.route, "forbidden_model_provider_import")

    @classmethod
    def _authorize_socket_audit_event(
        cls,
        event: str,
        arguments: tuple[Any, ...],
        context: _AuditRouteContext,
    ) -> None:
        if context.route != RepairExecutionRoute.LOOPBACK_MCP.value:
            cls.deny(context.route, "network_forbidden_for_route")
        if event == "socket.__new__" and cls._is_admitted_loopback_socket(arguments):
            return
        if event == "socket.connect" and cls._is_admitted_loopback_connect(arguments, context):
            return
        if event == "socket.getaddrinfo" and cls._is_admitted_loopback_lookup(arguments, context):
            return
        cls.deny(context.route, "non_admitted_loopback_network")

    @staticmethod
    def _is_admitted_loopback_socket(arguments: tuple[Any, ...]) -> bool:
        if len(arguments) < 3:
            return False
        family, socket_type = arguments[1], arguments[2]
        return family in {socket.AF_INET, socket.AF_INET6} and socket_type == socket.SOCK_STREAM

    @staticmethod
    def _is_admitted_loopback_connect(
        arguments: tuple[Any, ...], context: _AuditRouteContext
    ) -> bool:
        if len(arguments) < 2 or context.loopback_port is None:
            return False
        address = arguments[1]
        if not isinstance(address, tuple) or len(address) < 2:
            return False
        host, port = address[0], address[1]
        if not isinstance(host, str) or port != context.loopback_port:
            return False
        try:
            observed = ipaddress.ip_address(host)
        except ValueError:
            return False
        expected_host = urlparse(context.endpoint).hostname
        if expected_host == "localhost":
            return observed.is_loopback
        try:
            return observed == ipaddress.ip_address(expected_host or "")
        except ValueError:
            return False

    @staticmethod
    def _is_admitted_loopback_lookup(
        arguments: tuple[Any, ...], context: _AuditRouteContext
    ) -> bool:
        if len(arguments) < 2 or context.loopback_port is None:
            return False
        host, port = arguments[0], arguments[1]
        if port != context.loopback_port or not isinstance(host, str):
            return False
        expected_host = urlparse(context.endpoint).hostname
        if expected_host == "localhost":
            return host.strip().lower() in {"localhost", "127.0.0.1", "::1"}
        return host.strip().lower() == expected_host

    @classmethod
    def deny(cls, route: RepairExecutionRoute | str, reason: str) -> None:
        route_text = _route_text(route) or "<invalid>"
        cls._record_denial(route=route_text, reason=reason)
        raise NoLlmExecutionDenied(route=route_text, reason=reason)


class _ForbiddenProviderImportFinder:
    """Observe importlib resolution before a forbidden module can execute."""

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        del path, target
        if NoLlmExecutionGuard.active():
            NoLlmExecutionGuard._audit_provider_import(fullname)
        return None


@dataclass(frozen=True)
class DeterministicRepairAuthorityPolicy:
    """Allowlist execution authority for the TARGET deterministic runtime.

    Pins are capability identities supplied by the embedding runtime.  An empty
    allowlist admits nothing.  This deliberately avoids a convenience default
    that could turn a renamed provider route into an accidental fallback.
    """

    local_logic_pins: frozenset[str] = field(default_factory=frozenset)
    prover_subprocess_pins: frozenset[str] = field(default_factory=frozenset)
    loopback_mcp_pins: frozenset[str] = field(default_factory=frozenset)
    runtime: str = TARGET_REPAIR_RUNTIME

    INTERFACE: ClassVar[Final[str]] = DETERMINISTIC_REPAIR_AUTHORITY_POLICY_INTERFACE

    def __post_init__(self) -> None:
        for name in (
            "local_logic_pins",
            "prover_subprocess_pins",
            "loopback_mcp_pins",
        ):
            pins = getattr(self, name)
            if isinstance(pins, str):
                raise ValueError(f"{name} must be a collection of explicit pins")
            normalized = frozenset(_pin_text(pin) for pin in pins)
            if "" in normalized:
                raise ValueError(f"{name} cannot contain an empty pin")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.runtime, str) or not self.runtime.strip():
            raise ValueError("runtime must be non-empty text")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DeterministicRepairAuthorityPolicy:
        """Load the reviewed authority schema without permissive defaults."""

        if not isinstance(raw, Mapping):
            raise ValueError("deterministic repair authority policy must be an object")
        raw = dict(raw)
        expected_fields = {
            "schema",
            "interface",
            "runtime",
            "pin_strategy",
            "local_logic_pins",
            "prover_subprocess_pins",
            "loopback_mcp_pins",
            "model_call_budget",
            "llm_call_budget",
            "remote_provider_call_budget",
            "network_policy",
            "model_or_remote_fallback_authorized",
        }
        if set(raw) != expected_fields:
            drift = sorted(set(raw).symmetric_difference(expected_fields))
            raise ValueError(
                "deterministic repair authority fields must match exactly: " + ",".join(drift)
            )
        if raw.get("schema") != DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA:
            raise ValueError("unsupported deterministic repair authority schema")
        if raw.get("interface") != cls.INTERFACE:
            raise ValueError("unsupported deterministic repair authority interface")
        if raw.get("runtime") != TARGET_REPAIR_RUNTIME:
            raise ValueError("authority policy targets an unsupported runtime")
        if raw.get("pin_strategy") != "verified_capability_receipt_ids":
            raise ValueError("authority pin strategy must use verified capability receipts")
        for counter in (
            "model_call_budget",
            "llm_call_budget",
            "remote_provider_call_budget",
        ):
            if type(raw.get(counter)) is not int or raw[counter] != 0:
                raise ValueError(f"{counter} must be exactly zero")
        if raw.get("network_policy") != "deny_except_explicit_loopback":
            raise ValueError("authority network policy is not fail closed")
        if raw.get("model_or_remote_fallback_authorized") is not False:
            raise ValueError("model or remote fallback must be explicitly false")
        for name in (
            "local_logic_pins",
            "prover_subprocess_pins",
            "loopback_mcp_pins",
        ):
            if not isinstance(raw.get(name), list):
                raise ValueError(f"{name} must be an explicit list")
        return cls(
            local_logic_pins=frozenset(raw["local_logic_pins"]),
            prover_subprocess_pins=frozenset(raw["prover_subprocess_pins"]),
            loopback_mcp_pins=frozenset(raw["loopback_mcp_pins"]),
            runtime=str(raw["runtime"]),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> DeterministicRepairAuthorityPolicy:
        try:
            payload = json.loads(
                Path(path).read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("deterministic repair authority policy is unreadable") from exc
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA,
            "interface": self.INTERFACE,
            "runtime": self.runtime,
            "pin_strategy": "verified_capability_receipt_ids",
            "local_logic_pins": sorted(self.local_logic_pins),
            "prover_subprocess_pins": sorted(self.prover_subprocess_pins),
            "loopback_mcp_pins": sorted(self.loopback_mcp_pins),
            "model_call_budget": 0,
            "llm_call_budget": 0,
            "remote_provider_call_budget": 0,
            "network_policy": "deny_except_explicit_loopback",
            "model_or_remote_fallback_authorized": False,
        }

    def evaluate(
        self,
        route: RepairExecutionRoute | str,
        *,
        pin: str = "",
        endpoint: str = "",
    ) -> RepairAuthorityDecision:
        """Evaluate a route without invoking it; unknown and aliases deny."""

        text = _route_text(route)
        key = _route_key(route)
        normalized_pin = _pin_text(pin)
        denied_reason = self._denial_reason(
            text=text, key=key, pin=normalized_pin, endpoint=endpoint
        )
        if denied_reason:
            return RepairAuthorityDecision(
                disposition=RepairAuthorityDisposition.DENIED,
                route=text or "<invalid>",
                reason=denied_reason,
                pin=normalized_pin,
            )
        return RepairAuthorityDecision(
            disposition=RepairAuthorityDisposition.ALLOWED,
            route=text,
            reason="explicit_pinned_deterministic_route",
            pin=normalized_pin,
        )

    def authorize(
        self,
        route: RepairExecutionRoute | str,
        *,
        pin: str = "",
        endpoint: str = "",
    ) -> RepairAuthorityDecision:
        """Authorize or record and raise before a target callback is reached."""

        decision = self.evaluate(route, pin=pin, endpoint=endpoint)
        if not decision.allowed:
            NoLlmExecutionGuard.deny(decision.route, decision.reason)
        NoLlmExecutionGuard._record_allowed(route=decision.route)
        return decision

    # Policy vocabulary used by adapters that call policy checks explicitly.
    assert_authorized = authorize

    def invoke(
        self,
        route: RepairExecutionRoute | str,
        callback: Callable[..., _T],
        *args: Any,
        pin: str = "",
        endpoint: str = "",
        **kwargs: Any,
    ) -> _T:
        """Invoke exactly one admitted callback inside the process-wide guard."""

        if not callable(callback):
            raise TypeError("deterministic repair callback must be callable")
        with NoLlmExecutionGuard(runtime=self.runtime):
            decision = self.authorize(route, pin=pin, endpoint=endpoint)
            with NoLlmExecutionGuard.authorized_route(decision, endpoint=endpoint):
                return callback(*args, **kwargs)

    execute = invoke

    def abstain(self, reason: str) -> RepairAuthorityDecision:
        return self._terminal(RepairAuthorityDisposition.ABSTAIN, reason)

    def defer(self, reason: str) -> RepairAuthorityDecision:
        return self._terminal(RepairAuthorityDisposition.DEFER, reason)

    def _terminal(
        self, disposition: RepairAuthorityDisposition, reason: str
    ) -> RepairAuthorityDecision:
        text = _pin_text(reason)
        if not text:
            raise ValueError("terminal deterministic repair reason must be non-empty")
        return RepairAuthorityDecision(
            disposition=disposition,
            route="",
            reason=text,
        )

    def _denial_reason(self, *, text: str, key: str, pin: str, endpoint: str) -> str:
        if not text:
            return "invalid_route"

        # These aliases cover direct calls and indirection/retry/rescue/residual
        # escape hatches.  Match route *components*, not arbitrary substrings:
        # a legitimate pin such as ``logic:model-checker@1`` must not become a
        # provider route merely because of its name.  Unknown spellings still
        # deny below, so relaxing substring matching creates no fail-open path.
        route_terms = frozenset(part for part in key.split("_") if part)
        forbidden_terms = (
            "llm",
            "model",
            "inference",
            "provider",
            "remote",
            "retry",
            "rescue",
            "residual",
            "self_improvement",
            "selfimprovement",
            "fallback",
            "completion",
            "generation",
            "chat",
        )
        if route_terms.intersection(forbidden_terms) or (
            {"self", "improvement"}.issubset(route_terms)
        ):
            return "forbidden_model_or_remote_route"

        if text == RepairExecutionRoute.DETERMINISTIC_LOCAL_LOGIC.value:
            return "" if pin in self.local_logic_pins else "unpinned_local_logic"
        if text == RepairExecutionRoute.PROVER_SUBPROCESS.value:
            return "" if pin in self.prover_subprocess_pins else "unpinned_prover_subprocess"
        if text == RepairExecutionRoute.LOOPBACK_MCP.value:
            if pin not in self.loopback_mcp_pins:
                return "unpinned_loopback_mcp"
            if not _is_loopback(endpoint):
                return "non_loopback_mcp_endpoint"
            return ""
        return "unknown_route"


# Short aliases keep integrations explicit while avoiding a second policy type.
NoLlmPolicy = DeterministicRepairAuthorityPolicy
NoLlmGuard = NoLlmExecutionGuard


__all__ = [
    "DETERMINISTIC_REPAIR_AUTHORITY_POLICY_INTERFACE",
    "DETERMINISTIC_REPAIR_AUTHORITY_POLICY_SCHEMA",
    "TARGET_REPAIR_RUNTIME",
    "NO_LLM_EXECUTION_GUARD_INTERFACE",
    "DeterministicRepairAuthorityDenied",
    "DeterministicRepairAuthorityPolicy",
    "NoLlmExecutionDenied",
    "NoLlmExecutionGuard",
    "NoLlmGuard",
    "NoLlmPolicy",
    "RepairAuthorityDecision",
    "RepairAuthorityDisposition",
    "RepairExecutionRoute",
]
