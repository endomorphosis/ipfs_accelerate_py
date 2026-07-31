"""Supervisor-owned native Hammer execution authorization gate (LPR-012).

Solver, frontend, and kernel execution default to **disabled**.  A request is
admitted only when it carries an exact operation permit, a pinned environment
lock, and an intersecting resource/policy envelope.  This module never runs
solvers or kernels itself.

Authority rules (fail-closed):

* ``network=false`` is policy metadata, not OS isolation, unless an OS
  isolation receipt is supplied.
* Executable path/version locks are not signed supply-chain integrity;
  autonomous lanes that require integrity need a reviewed executable digest
  or an isolated execution receipt.
* CPU/memory enforcement strength is platform-typed (POSIX RLIMIT vs
  partial/unsupported).  Required bounds that cannot be enforced block
  autonomous native lanes.
* Model execution and learned selection remain opt-in fields on the permit
  and never expand the solver allowlist.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final


# ---------------------------------------------------------------------------
# Schemas / constants
# ---------------------------------------------------------------------------

NATIVE_EXECUTION_GATE_INTERFACE: Final = "NativeExecutionAuthorizationGate@1"
NATIVE_EXECUTION_GATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/hammer-native-execution-gate@1"
)
NATIVE_EXECUTION_PERMIT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/hammer-native-execution-permit@1"
)
NATIVE_EXECUTION_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/hammer-native-execution-decision@1"
)
RESOURCE_ENFORCEMENT_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/resource-enforcement-report@1"
)
POLICY_INTERSECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/hammer-policy-intersection@1"
)

PRODUCER_ID: Final = "hammer-native-execution-gate@1"
GATE_VERSION: Final = 1

KNOWN_SOLVERS: Final = frozenset({"cvc5", "e", "vampire", "z3", "eprover"})
_SOLVER_ALIASES: Final = {"eprover": "e"}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NativeExecutionOperation(str, Enum):
    """Exact native operations the gate can authorize."""

    SOLVER = "solver"
    FRONTEND = "frontend"
    KERNEL = "kernel"
    RECONSTRUCTION = "reconstruction"
    PORTFOLIO = "portfolio"
    PREMISE_SELECTION = "premise_selection"
    TRANSLATION = "translation"
    COUNTERMODEL_REPLAY = "countermodel_replay"


class ResourceEnforcementStrength(str, Enum):
    """How strongly the host can enforce CPU/memory process bounds."""

    POSIX_RLIMIT = "posix_rlimit"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class NativeExecutionDisposition(str, Enum):
    """Gate outcome for one authorization attempt."""

    AUTHORIZED = "authorized"
    POLICY_DENIED = "policy_denied"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    RESOURCE_UNENFORCEABLE = "resource_unenforceable"
    SUPPLY_CHAIN_DENIED = "supply_chain_denied"
    PERMIT_MISSING = "permit_missing"
    PERMIT_MISMATCH = "permit_mismatch"
    DISABLED_BY_DEFAULT = "disabled_by_default"
    STALE = "stale"
    MALFORMED = "malformed"


class NativeExecutionLane(str, Enum):
    """Execution autonomy level.  Autonomous lanes are stricter."""

    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"
    DIAGNOSTIC = "diagnostic"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NativeExecutionGateError(ValueError):
    """Raised when a gate contract is malformed."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise NativeExecutionGateError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise NativeExecutionGateError(f"{field_name} must not be empty")
    return result


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeExecutionGateError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NativeExecutionGateError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeExecutionGateError(f"{field_name} must be an object")
    return {str(k): v for k, v in value.items()}


def _solver_names(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise NativeExecutionGateError(f"{field_name} must be an array of strings")
    result: list[str] = []
    for item in value:
        name = _text(item, field_name=field_name).lower()
        name = _SOLVER_ALIASES.get(name, name)
        if name not in KNOWN_SOLVERS and name != "e":
            # eprover already aliased; unknown names are rejected.
            if name not in {"cvc5", "e", "vampire", "z3"}:
                raise NativeExecutionGateError(
                    f"{field_name} contains unknown solver: {name}"
                )
        if name not in result:
            result.append(name)
    return tuple(result)


def _digest(payload: Mapping[str, Any], *, prefix: str) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}:sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _minimum_positive(*values: int) -> int:
    positives = [v for v in values if v > 0]
    if not positives:
        raise NativeExecutionGateError("no positive resource bound available")
    return min(positives)


# ---------------------------------------------------------------------------
# Resource enforcement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceEnforcementReport:
    """Platform-typed resource-enforcement strength for native solver lanes."""

    platform: str
    cpu_enforcement: ResourceEnforcementStrength
    memory_enforcement: ResourceEnforcementStrength
    process_isolation: ResourceEnforcementStrength
    network_policy_denied: bool = True
    network_os_isolation: bool = False
    environment_lock_path_version_only: bool = True
    signed_binary_integrity: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def cpu_memory_enforceable(self) -> bool:
        """True when both CPU and memory bounds can be OS-enforced."""

        strong = {
            ResourceEnforcementStrength.POSIX_RLIMIT,
            ResourceEnforcementStrength.PARTIAL,
        }
        # Autonomous lanes require full POSIX RLIMIT for both.
        return (
            self.cpu_enforcement is ResourceEnforcementStrength.POSIX_RLIMIT
            and self.memory_enforcement is ResourceEnforcementStrength.POSIX_RLIMIT
        )

    @property
    def cpu_memory_partially_enforceable(self) -> bool:
        return (
            self.cpu_enforcement
            not in {
                ResourceEnforcementStrength.UNSUPPORTED,
                ResourceEnforcementStrength.UNKNOWN,
            }
            and self.memory_enforcement
            not in {
                ResourceEnforcementStrength.UNSUPPORTED,
                ResourceEnforcementStrength.UNKNOWN,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESOURCE_ENFORCEMENT_REPORT_SCHEMA,
            "platform": self.platform,
            "cpu_enforcement": self.cpu_enforcement.value,
            "memory_enforcement": self.memory_enforcement.value,
            "process_isolation": self.process_isolation.value,
            "network_policy_denied": self.network_policy_denied,
            "network_os_isolation": self.network_os_isolation,
            "environment_lock_path_version_only": self.environment_lock_path_version_only,
            "signed_binary_integrity": self.signed_binary_integrity,
            "cpu_memory_enforceable": self.cpu_memory_enforceable,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceEnforcementReport":
        data = _mapping(payload, field_name="resource_enforcement")
        return cls(
            platform=_text(data.get("platform", platform.system()), field_name="platform"),
            cpu_enforcement=ResourceEnforcementStrength(
                str(data.get("cpu_enforcement", "unknown"))
            ),
            memory_enforcement=ResourceEnforcementStrength(
                str(data.get("memory_enforcement", "unknown"))
            ),
            process_isolation=ResourceEnforcementStrength(
                str(data.get("process_isolation", "unknown"))
            ),
            network_policy_denied=bool(data.get("network_policy_denied", True)),
            network_os_isolation=bool(data.get("network_os_isolation", False)),
            environment_lock_path_version_only=bool(
                data.get("environment_lock_path_version_only", True)
            ),
            signed_binary_integrity=bool(data.get("signed_binary_integrity", False)),
            details=_mapping(data.get("details") or {}, field_name="details"),
        )


def probe_resource_enforcement(
    *,
    network_policy_denied: bool = True,
    network_os_isolation: bool = False,
    signed_binary_integrity: bool = False,
) -> ResourceEnforcementReport:
    """Probe host CPU/memory/process enforcement without granting authority."""

    system = platform.system().lower() or "unknown"
    cpu = ResourceEnforcementStrength.UNSUPPORTED
    memory = ResourceEnforcementStrength.UNSUPPORTED
    process = ResourceEnforcementStrength.UNSUPPORTED
    details: dict[str, Any] = {"system": system}

    if system in {"linux", "darwin", "freebsd", "openbsd", "netbsd"}:
        try:
            # Presence of RLIMIT_* symbols is the POSIX signal; actual setrlimit
            # may still be restricted by container policy.
            has_cpu = hasattr(resource, "RLIMIT_CPU")
            has_as = hasattr(resource, "RLIMIT_AS") or hasattr(resource, "RLIMIT_DATA")
            has_nproc = hasattr(resource, "RLIMIT_NPROC")
            if has_cpu and has_as:
                cpu = ResourceEnforcementStrength.POSIX_RLIMIT
                memory = ResourceEnforcementStrength.POSIX_RLIMIT
                process = (
                    ResourceEnforcementStrength.POSIX_RLIMIT
                    if has_nproc
                    else ResourceEnforcementStrength.PARTIAL
                )
            elif has_cpu or has_as:
                cpu = (
                    ResourceEnforcementStrength.POSIX_RLIMIT
                    if has_cpu
                    else ResourceEnforcementStrength.UNSUPPORTED
                )
                memory = (
                    ResourceEnforcementStrength.POSIX_RLIMIT
                    if has_as
                    else ResourceEnforcementStrength.UNSUPPORTED
                )
                process = ResourceEnforcementStrength.PARTIAL
            details["rlimit_cpu"] = has_cpu
            details["rlimit_address_space"] = has_as
            details["rlimit_nproc"] = has_nproc
        except Exception as exc:  # pragma: no cover - defensive
            details["probe_error"] = type(exc).__name__
            cpu = ResourceEnforcementStrength.UNKNOWN
            memory = ResourceEnforcementStrength.UNKNOWN
            process = ResourceEnforcementStrength.UNKNOWN
    elif system.startswith("win"):
        cpu = ResourceEnforcementStrength.UNSUPPORTED
        memory = ResourceEnforcementStrength.UNSUPPORTED
        process = ResourceEnforcementStrength.PARTIAL
        details["note"] = "Windows Job Objects not wired; autonomous native lane blocked"
    else:
        details["note"] = "unknown platform resource model"

    return ResourceEnforcementReport(
        platform=system,
        cpu_enforcement=cpu,
        memory_enforcement=memory,
        process_isolation=process,
        network_policy_denied=network_policy_denied,
        network_os_isolation=network_os_isolation,
        environment_lock_path_version_only=True,
        signed_binary_integrity=signed_binary_integrity,
        details=details,
    )


# ---------------------------------------------------------------------------
# Policy intersection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourcePolicySlice:
    """One layer of resource policy (supervisor / request / provider)."""

    allowed_solvers: tuple[str, ...] = ()
    timeout_ms: int = 0
    cpu_time_ms: int = 0
    memory_bytes: int = 0
    max_premises: int = 0
    max_parallel_processes: int = 0
    network_allowed: bool = False
    native_execution_allowed: bool = False
    model_execution_allowed: bool = False
    learned_selector_allowed: bool = False
    require_supply_chain_integrity: bool = False
    reviewed_executable_digests: Mapping[str, str] = field(default_factory=dict)
    isolated_execution_receipt_ids: tuple[str, ...] = ()
    os_network_isolation_receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_solvers",
            _solver_names(self.allowed_solvers, field_name="allowed_solvers")
            if self.allowed_solvers
            else (),
        )
        for name in (
            "timeout_ms",
            "cpu_time_ms",
            "memory_bytes",
            "max_premises",
            "max_parallel_processes",
        ):
            value = getattr(self, name)
            if value != 0:
                object.__setattr__(
                    self, name, _positive_int(value, field_name=name)
                )
        digests = {
            _text(k, field_name="reviewed_executable_digests.key"): _text(
                v, field_name="reviewed_executable_digests.value"
            )
            for k, v in dict(self.reviewed_executable_digests or {}).items()
        }
        object.__setattr__(self, "reviewed_executable_digests", MappingProxyType(digests))
        object.__setattr__(
            self,
            "isolated_execution_receipt_ids",
            tuple(
                _text(item, field_name="isolated_execution_receipt_ids")
                for item in self.isolated_execution_receipt_ids
            ),
        )
        object.__setattr__(
            self,
            "os_network_isolation_receipt_id",
            _text(
                self.os_network_isolation_receipt_id,
                field_name="os_network_isolation_receipt_id",
                required=False,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_solvers": list(self.allowed_solvers),
            "timeout_ms": self.timeout_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_bytes": self.memory_bytes,
            "max_premises": self.max_premises,
            "max_parallel_processes": self.max_parallel_processes,
            "network_allowed": self.network_allowed,
            "native_execution_allowed": self.native_execution_allowed,
            "model_execution_allowed": self.model_execution_allowed,
            "learned_selector_allowed": self.learned_selector_allowed,
            "require_supply_chain_integrity": self.require_supply_chain_integrity,
            "reviewed_executable_digests": dict(self.reviewed_executable_digests),
            "isolated_execution_receipt_ids": list(self.isolated_execution_receipt_ids),
            "os_network_isolation_receipt_id": self.os_network_isolation_receipt_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResourcePolicySlice":
        if payload is None:
            return cls()
        data = _mapping(payload, field_name="resource_policy")
        return cls(
            allowed_solvers=tuple(data.get("allowed_solvers") or ()),
            timeout_ms=int(data.get("timeout_ms") or 0),
            cpu_time_ms=int(data.get("cpu_time_ms") or 0),
            memory_bytes=int(data.get("memory_bytes") or 0),
            max_premises=int(data.get("max_premises") or 0),
            max_parallel_processes=int(data.get("max_parallel_processes") or 0),
            network_allowed=bool(data.get("network_allowed", False)),
            native_execution_allowed=bool(data.get("native_execution_allowed", False)),
            model_execution_allowed=bool(data.get("model_execution_allowed", False)),
            learned_selector_allowed=bool(data.get("learned_selector_allowed", False)),
            require_supply_chain_integrity=bool(
                data.get("require_supply_chain_integrity", False)
            ),
            reviewed_executable_digests=dict(
                data.get("reviewed_executable_digests") or {}
            ),
            isolated_execution_receipt_ids=tuple(
                data.get("isolated_execution_receipt_ids") or ()
            ),
            os_network_isolation_receipt_id=str(
                data.get("os_network_isolation_receipt_id") or ""
            ),
        )


@dataclass(frozen=True)
class PolicyIntersection:
    """Intersection of supervisor, request, and provider resource policies."""

    allowed_solvers: tuple[str, ...]
    timeout_ms: int
    cpu_time_ms: int
    memory_bytes: int
    max_premises: int
    max_parallel_processes: int
    network_allowed: bool
    network_is_os_isolation: bool
    native_execution_allowed: bool
    model_execution_allowed: bool
    learned_selector_allowed: bool
    require_supply_chain_integrity: bool
    reviewed_executable_digests: Mapping[str, str]
    isolated_execution_receipt_ids: tuple[str, ...]
    policy_id: str
    layers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reviewed_executable_digests", MappingProxyType(dict(self.reviewed_executable_digests))
        )
        object.__setattr__(self, "layers", MappingProxyType(dict(self.layers)))

    @property
    def network_false_is_metadata_unless_os_isolation(self) -> bool:
        """``network=false`` is metadata unless an OS isolation receipt exists."""

        return not self.network_allowed and not self.network_is_os_isolation

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_INTERSECTION_SCHEMA,
            "allowed_solvers": list(self.allowed_solvers),
            "timeout_ms": self.timeout_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_bytes": self.memory_bytes,
            "max_premises": self.max_premises,
            "max_parallel_processes": self.max_parallel_processes,
            "network_allowed": self.network_allowed,
            "network_is_os_isolation": self.network_is_os_isolation,
            "network_false_is_metadata_unless_os_isolation": (
                self.network_false_is_metadata_unless_os_isolation
            ),
            "native_execution_allowed": self.native_execution_allowed,
            "model_execution_allowed": self.model_execution_allowed,
            "learned_selector_allowed": self.learned_selector_allowed,
            "require_supply_chain_integrity": self.require_supply_chain_integrity,
            "reviewed_executable_digests": dict(self.reviewed_executable_digests),
            "isolated_execution_receipt_ids": list(self.isolated_execution_receipt_ids),
            "policy_id": self.policy_id,
            "layers": {k: dict(v) for k, v in self.layers.items()},
        }


def intersect_resource_policies(
    *,
    supervisor: ResourcePolicySlice | Mapping[str, Any] | None = None,
    request: ResourcePolicySlice | Mapping[str, Any] | None = None,
    provider: ResourcePolicySlice | Mapping[str, Any] | None = None,
) -> PolicyIntersection:
    """Intersect solver/process/time/CPU/memory/native/model policy layers.

    A tighter (smaller) numeric bound wins.  Booleans are AND-ed.  Solver
    allowlists are set-intersected.  Empty supervisor allowlist means "no
    solvers admitted".
    """

    layers: dict[str, ResourcePolicySlice] = {
        "supervisor": (
            supervisor
            if isinstance(supervisor, ResourcePolicySlice)
            else ResourcePolicySlice.from_dict(supervisor)
        ),
        "request": (
            request
            if isinstance(request, ResourcePolicySlice)
            else ResourcePolicySlice.from_dict(request)
        ),
        "provider": (
            provider
            if isinstance(provider, ResourcePolicySlice)
            else ResourcePolicySlice.from_dict(provider)
        ),
    }

    # Solver intersection: start from supervisor; empty supervisor => deny all.
    sup_solvers = set(layers["supervisor"].allowed_solvers)
    if not sup_solvers:
        allowed: tuple[str, ...] = ()
    else:
        candidates = sup_solvers
        for name in ("request", "provider"):
            layer_solvers = set(layers[name].allowed_solvers)
            if layer_solvers:
                candidates &= layer_solvers
        allowed = tuple(sorted(candidates))

    def _bound(attr: str, default: int) -> int:
        values = []
        for layer in layers.values():
            value = getattr(layer, attr)
            if value > 0:
                values.append(value)
        return min(values) if values else default

    network_allowed = all(layer.network_allowed for layer in layers.values())
    os_receipt = next(
        (
            layer.os_network_isolation_receipt_id
            for layer in layers.values()
            if layer.os_network_isolation_receipt_id
        ),
        "",
    )
    network_is_os = bool(os_receipt) and not network_allowed

    digests: dict[str, str] = {}
    for layer in layers.values():
        digests.update(dict(layer.reviewed_executable_digests))
    isolated_ids: list[str] = []
    for layer in layers.values():
        for item in layer.isolated_execution_receipt_ids:
            if item not in isolated_ids:
                isolated_ids.append(item)

    require_integrity = any(
        layer.require_supply_chain_integrity for layer in layers.values()
    )
    # Native/model/learned are fail-closed AND of all layers (defaults False).
    # An explicit operation permit is checked separately by the gate.
    native_allowed = all(layer.native_execution_allowed for layer in layers.values())
    model_allowed = all(layer.model_execution_allowed for layer in layers.values())
    # Learned selection is opt-in: supervisor must allow; request/provider may
    # only tighten (AND).  Defaults keep it denied.
    learned_allowed = all(layer.learned_selector_allowed for layer in layers.values())

    policy_body = {
        "allowed_solvers": list(allowed),
        "timeout_ms": _bound("timeout_ms", 30_000),
        "cpu_time_ms": _bound("cpu_time_ms", 30_000),
        "memory_bytes": _bound("memory_bytes", 512 * 1024 * 1024),
        "max_premises": _bound("max_premises", 64),
        "max_parallel_processes": _bound("max_parallel_processes", 4),
        "network_allowed": network_allowed,
        "native_execution_allowed": native_allowed,
        "model_execution_allowed": model_allowed,
        "learned_selector_allowed": learned_allowed,
        "require_supply_chain_integrity": require_integrity,
    }
    return PolicyIntersection(
        allowed_solvers=allowed,
        timeout_ms=policy_body["timeout_ms"],
        cpu_time_ms=policy_body["cpu_time_ms"],
        memory_bytes=policy_body["memory_bytes"],
        max_premises=policy_body["max_premises"],
        max_parallel_processes=policy_body["max_parallel_processes"],
        network_allowed=network_allowed,
        network_is_os_isolation=network_is_os,
        native_execution_allowed=native_allowed,
        model_execution_allowed=model_allowed,
        learned_selector_allowed=learned_allowed,
        require_supply_chain_integrity=require_integrity,
        reviewed_executable_digests=digests,
        isolated_execution_receipt_ids=tuple(isolated_ids),
        policy_id=_digest(policy_body, prefix="hammer-policy-intersection"),
        layers={name: layer.to_dict() for name, layer in layers.items()},
    )


# ---------------------------------------------------------------------------
# Permits and decisions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeExecutionPermit:
    """Exact operation permit required before native execution.

    Defaults keep solver/frontend/kernel execution disabled.  Callers must
    construct an explicit permit with matching environment and policy digests.
    """

    permit_id: str
    operations: tuple[NativeExecutionOperation, ...] = ()
    environment_lock_id: str = ""
    environment_lock_digest: str = ""
    policy_id: str = ""
    lane: NativeExecutionLane = NativeExecutionLane.SUPERVISED
    allowed_solvers: tuple[str, ...] = ()
    reviewed_executable_digests: Mapping[str, str] = field(default_factory=dict)
    isolated_execution_receipt_id: str = ""
    os_network_isolation_receipt_id: str = ""
    require_supply_chain_integrity: bool = False
    learned_selector_model_digest: str = ""
    learned_selector_ranking_only: bool = True
    issued_by: str = PRODUCER_ID
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "permit_id", _text(self.permit_id, field_name="permit_id")
        )
        ops: list[NativeExecutionOperation] = []
        for item in self.operations:
            if isinstance(item, NativeExecutionOperation):
                op = item
            else:
                op = NativeExecutionOperation(str(item))
            if op not in ops:
                ops.append(op)
        object.__setattr__(self, "operations", tuple(ops))
        object.__setattr__(
            self,
            "environment_lock_id",
            _text(
                self.environment_lock_id,
                field_name="environment_lock_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "environment_lock_digest",
            _text(
                self.environment_lock_digest,
                field_name="environment_lock_digest",
                required=False,
            ),
        )
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, field_name="policy_id", required=False)
        )
        if not isinstance(self.lane, NativeExecutionLane):
            object.__setattr__(self, "lane", NativeExecutionLane(str(self.lane)))
        object.__setattr__(
            self,
            "allowed_solvers",
            _solver_names(self.allowed_solvers, field_name="allowed_solvers")
            if self.allowed_solvers
            else (),
        )
        digests = {
            _text(k, field_name="reviewed_executable_digests.key"): _text(
                v, field_name="reviewed_executable_digests.value"
            )
            for k, v in dict(self.reviewed_executable_digests or {}).items()
        }
        object.__setattr__(self, "reviewed_executable_digests", MappingProxyType(digests))
        object.__setattr__(
            self,
            "isolated_execution_receipt_id",
            _text(
                self.isolated_execution_receipt_id,
                field_name="isolated_execution_receipt_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "os_network_isolation_receipt_id",
            _text(
                self.os_network_isolation_receipt_id,
                field_name="os_network_isolation_receipt_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "learned_selector_model_digest",
            _text(
                self.learned_selector_model_digest,
                field_name="learned_selector_model_digest",
                required=False,
            ),
        )
        if not isinstance(self.learned_selector_ranking_only, bool):
            raise NativeExecutionGateError(
                "learned_selector_ranking_only must be a boolean"
            )
        if not isinstance(self.require_supply_chain_integrity, bool):
            raise NativeExecutionGateError(
                "require_supply_chain_integrity must be a boolean"
            )

    @property
    def admits_any_execution(self) -> bool:
        return bool(self.operations)

    def authorizes(self, operation: NativeExecutionOperation | str) -> bool:
        op = (
            operation
            if isinstance(operation, NativeExecutionOperation)
            else NativeExecutionOperation(str(operation))
        )
        return op in self.operations

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NATIVE_EXECUTION_PERMIT_SCHEMA,
            "permit_id": self.permit_id,
            "operations": [op.value for op in self.operations],
            "environment_lock_id": self.environment_lock_id,
            "environment_lock_digest": self.environment_lock_digest,
            "policy_id": self.policy_id,
            "lane": self.lane.value,
            "allowed_solvers": list(self.allowed_solvers),
            "reviewed_executable_digests": dict(self.reviewed_executable_digests),
            "isolated_execution_receipt_id": self.isolated_execution_receipt_id,
            "os_network_isolation_receipt_id": self.os_network_isolation_receipt_id,
            "require_supply_chain_integrity": self.require_supply_chain_integrity,
            "learned_selector_model_digest": self.learned_selector_model_digest,
            "learned_selector_ranking_only": self.learned_selector_ranking_only,
            "issued_by": self.issued_by,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NativeExecutionPermit":
        data = _mapping(payload, field_name="permit")
        return cls(
            permit_id=str(data.get("permit_id") or ""),
            operations=tuple(data.get("operations") or ()),
            environment_lock_id=str(data.get("environment_lock_id") or ""),
            environment_lock_digest=str(data.get("environment_lock_digest") or ""),
            policy_id=str(data.get("policy_id") or ""),
            lane=str(data.get("lane") or NativeExecutionLane.SUPERVISED.value),
            allowed_solvers=tuple(data.get("allowed_solvers") or ()),
            reviewed_executable_digests=dict(
                data.get("reviewed_executable_digests") or {}
            ),
            isolated_execution_receipt_id=str(
                data.get("isolated_execution_receipt_id") or ""
            ),
            os_network_isolation_receipt_id=str(
                data.get("os_network_isolation_receipt_id") or ""
            ),
            require_supply_chain_integrity=bool(
                data.get("require_supply_chain_integrity", False)
            ),
            learned_selector_model_digest=str(
                data.get("learned_selector_model_digest") or ""
            ),
            learned_selector_ranking_only=bool(
                data.get("learned_selector_ranking_only", True)
            ),
            issued_by=str(data.get("issued_by") or PRODUCER_ID),
            notes=str(data.get("notes") or ""),
        )

    @classmethod
    def disabled(cls) -> "NativeExecutionPermit":
        """Default permit: no operations authorized."""

        return cls(permit_id="permit:disabled-by-default", operations=())


@dataclass(frozen=True)
class NativeExecutionDecision:
    """Immutable authorization decision with full audit projection."""

    disposition: NativeExecutionDisposition
    operation: NativeExecutionOperation
    authorized: bool
    permit_id: str
    policy_intersection: PolicyIntersection
    resource_enforcement: ResourceEnforcementReport
    environment_lock_id: str = ""
    reason_codes: tuple[str, ...] = ()
    decision_id: str = ""
    lane: NativeExecutionLane = NativeExecutionLane.SUPERVISED
    network_false_is_metadata: bool = True
    supply_chain_satisfied: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decision_id:
            body = {
                "disposition": self.disposition.value,
                "operation": self.operation.value,
                "permit_id": self.permit_id,
                "policy_id": self.policy_intersection.policy_id,
                "environment_lock_id": self.environment_lock_id,
                "reason_codes": list(self.reason_codes),
            }
            object.__setattr__(
                self, "decision_id", _digest(body, prefix="native-exec-decision")
            )
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NATIVE_EXECUTION_DECISION_SCHEMA,
            "decision_id": self.decision_id,
            "disposition": self.disposition.value,
            "operation": self.operation.value,
            "authorized": self.authorized,
            "permit_id": self.permit_id,
            "policy_intersection": self.policy_intersection.to_dict(),
            "resource_enforcement": self.resource_enforcement.to_dict(),
            "environment_lock_id": self.environment_lock_id,
            "reason_codes": list(self.reason_codes),
            "lane": self.lane.value,
            "network_false_is_metadata": self.network_false_is_metadata,
            "supply_chain_satisfied": self.supply_chain_satisfied,
            "details": dict(self.details),
            "producer_id": PRODUCER_ID,
            "gate_interface": NATIVE_EXECUTION_GATE_INTERFACE,
            "gate_version": GATE_VERSION,
        }


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeExecutionAuthorizationGate:
    """Fail-closed gate: native execution is disabled until explicitly permitted.

    The gate is pure authorization.  It does not import Hammer, launch solvers,
    or mutate process environment.
    """

    default_permit: NativeExecutionPermit = field(
        default_factory=NativeExecutionPermit.disabled
    )
    resource_enforcement: ResourceEnforcementReport | None = None
    supervisor_policy: ResourcePolicySlice = field(default_factory=ResourcePolicySlice)
    provider_policy: ResourcePolicySlice = field(default_factory=ResourcePolicySlice)
    require_environment_lock: bool = True

    def __post_init__(self) -> None:
        if self.resource_enforcement is None:
            object.__setattr__(
                self, "resource_enforcement", probe_resource_enforcement()
            )
        if not isinstance(self.default_permit, NativeExecutionPermit):
            raise NativeExecutionGateError(
                "default_permit must be a NativeExecutionPermit"
            )

    def _decision(
        self,
        *,
        disposition: NativeExecutionDisposition,
        operation: NativeExecutionOperation,
        authorized: bool,
        permit: NativeExecutionPermit,
        intersection: PolicyIntersection,
        environment_lock_id: str,
        reason_codes: Sequence[str],
        supply_chain_satisfied: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> NativeExecutionDecision:
        assert self.resource_enforcement is not None
        network_metadata = (
            not intersection.network_allowed
            and not intersection.network_is_os_isolation
        )
        return NativeExecutionDecision(
            disposition=disposition,
            operation=operation,
            authorized=authorized,
            permit_id=permit.permit_id,
            policy_intersection=intersection,
            resource_enforcement=self.resource_enforcement,
            environment_lock_id=environment_lock_id,
            reason_codes=tuple(reason_codes),
            lane=permit.lane,
            network_false_is_metadata=network_metadata,
            supply_chain_satisfied=supply_chain_satisfied,
            details=dict(details or {}),
        )

    def authorize(
        self,
        operation: NativeExecutionOperation | str,
        *,
        permit: NativeExecutionPermit | Mapping[str, Any] | None = None,
        environment_lock: Mapping[str, Any] | None = None,
        request_policy: ResourcePolicySlice | Mapping[str, Any] | None = None,
        required_solvers: Sequence[str] | None = None,
        executable_paths: Mapping[str, str] | None = None,
    ) -> NativeExecutionDecision:
        """Authorize one native operation under the intersecting policy."""

        op = (
            operation
            if isinstance(operation, NativeExecutionOperation)
            else NativeExecutionOperation(str(operation))
        )
        if permit is None:
            active = self.default_permit
        elif isinstance(permit, NativeExecutionPermit):
            active = permit
        else:
            try:
                active = NativeExecutionPermit.from_dict(permit)
            except (TypeError, ValueError, NativeExecutionGateError) as exc:
                empty = ResourcePolicySlice()
                intersection = intersect_resource_policies(
                    supervisor=self.supervisor_policy,
                    request=empty,
                    provider=self.provider_policy,
                )
                return self._decision(
                    disposition=NativeExecutionDisposition.MALFORMED,
                    operation=op,
                    authorized=False,
                    permit=NativeExecutionPermit.disabled(),
                    intersection=intersection,
                    environment_lock_id="",
                    reason_codes=("permit_malformed",),
                    details={"error": str(exc)},
                )

        # Tighten allowlist/digests with the permit without using the permit as
        # a full policy layer (permit is the operation admission, not a bound).
        request_slice = (
            request_policy
            if isinstance(request_policy, ResourcePolicySlice)
            else ResourcePolicySlice.from_dict(request_policy)
        )
        if active.allowed_solvers:
            request_slice = ResourcePolicySlice(
                allowed_solvers=active.allowed_solvers,
                timeout_ms=request_slice.timeout_ms,
                cpu_time_ms=request_slice.cpu_time_ms,
                memory_bytes=request_slice.memory_bytes,
                max_premises=request_slice.max_premises,
                max_parallel_processes=request_slice.max_parallel_processes,
                network_allowed=request_slice.network_allowed,
                native_execution_allowed=True,
                model_execution_allowed=request_slice.model_execution_allowed,
                learned_selector_allowed=bool(active.learned_selector_model_digest)
                and active.learned_selector_ranking_only,
                require_supply_chain_integrity=(
                    active.require_supply_chain_integrity
                    or request_slice.require_supply_chain_integrity
                ),
                reviewed_executable_digests={
                    **dict(request_slice.reviewed_executable_digests),
                    **dict(active.reviewed_executable_digests),
                },
                isolated_execution_receipt_ids=(
                    *request_slice.isolated_execution_receipt_ids,
                    *(
                        (active.isolated_execution_receipt_id,)
                        if active.isolated_execution_receipt_id
                        else ()
                    ),
                ),
                os_network_isolation_receipt_id=(
                    active.os_network_isolation_receipt_id
                    or request_slice.os_network_isolation_receipt_id
                ),
            )

        intersection = intersect_resource_policies(
            supervisor=self.supervisor_policy,
            request=request_slice,
            provider=self.provider_policy,
        )
        # Fold permit digests/isolation receipts into the intersection view.
        if active.reviewed_executable_digests or active.isolated_execution_receipt_id:
            merged_digests = {
                **dict(intersection.reviewed_executable_digests),
                **dict(active.reviewed_executable_digests),
            }
            merged_isolated = list(intersection.isolated_execution_receipt_ids)
            if (
                active.isolated_execution_receipt_id
                and active.isolated_execution_receipt_id not in merged_isolated
            ):
                merged_isolated.append(active.isolated_execution_receipt_id)
            intersection = PolicyIntersection(
                allowed_solvers=intersection.allowed_solvers,
                timeout_ms=intersection.timeout_ms,
                cpu_time_ms=intersection.cpu_time_ms,
                memory_bytes=intersection.memory_bytes,
                max_premises=intersection.max_premises,
                max_parallel_processes=intersection.max_parallel_processes,
                network_allowed=intersection.network_allowed,
                network_is_os_isolation=intersection.network_is_os_isolation
                or bool(active.os_network_isolation_receipt_id),
                native_execution_allowed=True,
                model_execution_allowed=intersection.model_execution_allowed,
                learned_selector_allowed=intersection.learned_selector_allowed
                or (
                    bool(active.learned_selector_model_digest)
                    and active.learned_selector_ranking_only
                ),
                require_supply_chain_integrity=(
                    intersection.require_supply_chain_integrity
                    or active.require_supply_chain_integrity
                ),
                reviewed_executable_digests=merged_digests,
                isolated_execution_receipt_ids=tuple(merged_isolated),
                policy_id=intersection.policy_id,
                layers=dict(intersection.layers),
            )

        lock = _mapping(environment_lock or {}, field_name="environment_lock")
        lock_id = str(lock.get("lock_id") or "")
        lock_digest = ""
        if lock:
            identity = dict(lock)
            identity.pop("lock_id", None)
            lock_digest = _digest(identity, prefix="hammer-environment")

        # 1. Default deny when no operations on permit.
        if not active.operations:
            return self._decision(
                disposition=NativeExecutionDisposition.DISABLED_BY_DEFAULT,
                operation=op,
                authorized=False,
                permit=active,
                intersection=intersection,
                environment_lock_id=lock_id,
                reason_codes=("native_execution_disabled_by_default", "permit_empty"),
            )

        # 2. Exact operation must be listed.
        if not active.authorizes(op):
            return self._decision(
                disposition=NativeExecutionDisposition.PERMIT_MISMATCH,
                operation=op,
                authorized=False,
                permit=active,
                intersection=intersection,
                environment_lock_id=lock_id,
                reason_codes=("operation_not_permitted", op.value),
                details={
                    "permitted_operations": [item.value for item in active.operations]
                },
            )

        # 3. Environment lock binding.
        if self.require_environment_lock:
            if not lock_id:
                return self._decision(
                    disposition=NativeExecutionDisposition.ENVIRONMENT_MISMATCH,
                    operation=op,
                    authorized=False,
                    permit=active,
                    intersection=intersection,
                    environment_lock_id="",
                    reason_codes=("environment_lock_required",),
                )
            if active.environment_lock_id and active.environment_lock_id != lock_id:
                return self._decision(
                    disposition=NativeExecutionDisposition.ENVIRONMENT_MISMATCH,
                    operation=op,
                    authorized=False,
                    permit=active,
                    intersection=intersection,
                    environment_lock_id=lock_id,
                    reason_codes=("environment_lock_id_mismatch",),
                    details={
                        "permit_environment_lock_id": active.environment_lock_id,
                        "request_environment_lock_id": lock_id,
                    },
                )
            if (
                active.environment_lock_digest
                and lock_digest
                and active.environment_lock_digest != lock_digest
                and active.environment_lock_digest != lock_id
            ):
                return self._decision(
                    disposition=NativeExecutionDisposition.ENVIRONMENT_MISMATCH,
                    operation=op,
                    authorized=False,
                    permit=active,
                    intersection=intersection,
                    environment_lock_id=lock_id,
                    reason_codes=("environment_lock_digest_mismatch",),
                )

        # 4. Policy id binding when present on the permit.
        if active.policy_id:
            known = {
                intersection.policy_id,
                str(
                    (intersection.layers.get("supervisor") or {}).get("policy_id") or ""
                ),
            }
            known.discard("")
            if known and active.policy_id not in known:
                # Allow stable caller policy labels that match supervisor digests
                # computed outside this gate (exact string equality only).
                if active.policy_id != intersection.policy_id:
                    return self._decision(
                        disposition=NativeExecutionDisposition.POLICY_DENIED,
                        operation=op,
                        authorized=False,
                        permit=active,
                        intersection=intersection,
                        environment_lock_id=lock_id,
                        reason_codes=("policy_id_mismatch",),
                        details={
                            "permit_policy_id": active.policy_id,
                            "intersection_policy_id": intersection.policy_id,
                        },
                    )

        # 5. Solver allowlist for solver/portfolio operations.
        if op in {
            NativeExecutionOperation.SOLVER,
            NativeExecutionOperation.PORTFOLIO,
        }:
            if not intersection.allowed_solvers:
                return self._decision(
                    disposition=NativeExecutionDisposition.POLICY_DENIED,
                    operation=op,
                    authorized=False,
                    permit=active,
                    intersection=intersection,
                    environment_lock_id=lock_id,
                    reason_codes=("solver_allowlist_empty",),
                )
            if required_solvers:
                required = _solver_names(
                    required_solvers, field_name="required_solvers"
                )
                missing = sorted(set(required) - set(intersection.allowed_solvers))
                if missing:
                    return self._decision(
                        disposition=NativeExecutionDisposition.POLICY_DENIED,
                        operation=op,
                        authorized=False,
                        permit=active,
                        intersection=intersection,
                        environment_lock_id=lock_id,
                        reason_codes=("solver_not_allowlisted",),
                        details={"missing_solvers": missing},
                    )

        # 6. Autonomous lane requires enforceable CPU/memory bounds.
        assert self.resource_enforcement is not None
        if active.lane is NativeExecutionLane.AUTONOMOUS:
            if not self.resource_enforcement.cpu_memory_enforceable:
                return self._decision(
                    disposition=NativeExecutionDisposition.RESOURCE_UNENFORCEABLE,
                    operation=op,
                    authorized=False,
                    permit=active,
                    intersection=intersection,
                    environment_lock_id=lock_id,
                    reason_codes=(
                        "autonomous_lane_requires_enforced_cpu_memory",
                        f"cpu={self.resource_enforcement.cpu_enforcement.value}",
                        f"memory={self.resource_enforcement.memory_enforcement.value}",
                    ),
                    details={
                        "resource_enforcement": self.resource_enforcement.to_dict(),
                        "lane": active.lane.value,
                    },
                )

        # 7. Supply-chain integrity when policy-required.
        require_integrity = (
            active.require_supply_chain_integrity
            or intersection.require_supply_chain_integrity
        )
        supply_ok = False
        if require_integrity:
            digests = dict(intersection.reviewed_executable_digests)
            digests.update(dict(active.reviewed_executable_digests))
            isolated = set(intersection.isolated_execution_receipt_ids)
            if active.isolated_execution_receipt_id:
                isolated.add(active.isolated_execution_receipt_id)
            # Path/version alone is insufficient — digests or isolated receipt.
            if not digests and not isolated:
                return self._decision(
                    disposition=NativeExecutionDisposition.SUPPLY_CHAIN_DENIED,
                    operation=op,
                    authorized=False,
                    permit=active,
                    intersection=intersection,
                    environment_lock_id=lock_id,
                    reason_codes=(
                        "supply_chain_integrity_required",
                        "reviewed_digest_or_isolated_receipt_missing",
                    ),
                    details={
                        "environment_lock_path_version_only": True,
                        "signed_binary_integrity": False,
                        "executable_paths_present": bool(
                            executable_paths
                            or (lock.get("executable_paths") if lock else None)
                        ),
                    },
                )
            supply_ok = True

        # 8. Network=false remains metadata unless OS isolation receipt exists.
        network_metadata = (
            not intersection.network_allowed
            and not intersection.network_is_os_isolation
            and not active.os_network_isolation_receipt_id
        )

        return self._decision(
            disposition=NativeExecutionDisposition.AUTHORIZED,
            operation=op,
            authorized=True,
            permit=active,
            intersection=intersection,
            environment_lock_id=lock_id,
            reason_codes=("authorized",),
            supply_chain_satisfied=supply_ok,
            details={
                "network_false_is_metadata": network_metadata,
                "network_os_isolation": intersection.network_is_os_isolation
                or bool(active.os_network_isolation_receipt_id),
                "learned_selector_ranking_only": active.learned_selector_ranking_only,
                "learned_selector_model_digest": active.learned_selector_model_digest,
                "allowed_solvers": list(intersection.allowed_solvers),
                "timeout_ms": intersection.timeout_ms,
                "cpu_time_ms": intersection.cpu_time_ms,
                "memory_bytes": intersection.memory_bytes,
                "max_parallel_processes": intersection.max_parallel_processes,
            },
        )

    def require(
        self,
        operation: NativeExecutionOperation | str,
        **kwargs: Any,
    ) -> NativeExecutionDecision:
        """Authorize or raise :class:`NativeExecutionGateError`."""

        decision = self.authorize(operation, **kwargs)
        if not decision.authorized:
            raise NativeExecutionGateError(
                f"native execution denied for {decision.operation.value}: "
                + ",".join(decision.reason_codes)
            )
        return decision


__all__ = [
    "NATIVE_EXECUTION_GATE_INTERFACE",
    "NATIVE_EXECUTION_GATE_SCHEMA",
    "NATIVE_EXECUTION_PERMIT_SCHEMA",
    "NATIVE_EXECUTION_DECISION_SCHEMA",
    "RESOURCE_ENFORCEMENT_REPORT_SCHEMA",
    "POLICY_INTERSECTION_SCHEMA",
    "PRODUCER_ID",
    "GATE_VERSION",
    "NativeExecutionOperation",
    "ResourceEnforcementStrength",
    "NativeExecutionDisposition",
    "NativeExecutionLane",
    "NativeExecutionGateError",
    "ResourceEnforcementReport",
    "probe_resource_enforcement",
    "ResourcePolicySlice",
    "PolicyIntersection",
    "intersect_resource_policies",
    "NativeExecutionPermit",
    "NativeExecutionDecision",
    "NativeExecutionAuthorizationGate",
]
