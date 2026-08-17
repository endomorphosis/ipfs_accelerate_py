"""Supervisor-owned adapter for the :mod:`ipfs_datasets_py` Hammer portfolio.

The Hammer package is an optional dependency of ``ipfs_accelerate_py``.  This
module consequently imports it lazily, after the generic proof-provider
boundary has admitted a request.  The adapter does not reimplement premise
selection, translation, solver execution, provenance normalization, or kernel
reconstruction.

The adapter has four deliberately strict properties:

* a code obligation becomes a reproducible Hammer request only when the
  translation family, every declared premise, and a pinned environment lock
  are explicit;
* the Hammer policy is the intersection of the provider's supervisor policy,
  a per-request supervisor policy, and the generic resource envelope;
* solver attempts and candidates are returned with a provenance projection
  that binds them to the obligation, tree, premises, and upstream receipts;
* a missing/unknown lowering is a typed ``unsupported`` provider response
  containing the exact configured fallback checks, never a guessed proof; and
* when a trust-aware cache is supplied, identical portfolio work is guarded by
  its cross-thread and cross-process single-flight lease.

ATP/SMT candidates remain untrusted.  This provider never maps a portfolio
verdict to kernel-verified assurance; independent reconstruction is owned by
the kernel-verification integration.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final

from ..proof.formal_verification_capabilities import (
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
)
from ..proof.formal_verification_contracts import (
    CodeProofObligation,
    ResourceBudget,
    canonical_json,
)
from ..proof.formal_verification_cache import (
    FormalVerificationCache,
    ProofCacheKey,
    build_proof_cache_key,
)
from ..proof.formal_verification_provider import (
    PROOF_PROVIDER_PROTOCOL_VERSION,
    ProofProviderError,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
)
from ..analysis.analysis_operation_registry import (
    IPFS_DATASETS_ANALYSIS_PRODUCER_ID,
    LOCAL_ANALYSIS_PRODUCER_ID,
    AnalysisOperation,
    AnalysisProducer,
    LogicFamily,
    normalize_analysis_operation,
    normalize_analysis_reference,
    normalize_logic_family,
    normalized_reference_payload,
)
from ..analysis.analysis_transport import (
    ANALYSIS_TRANSPORT_PROTOCOL_VERSION,
    ANALYSIS_TRANSPORT_REQUEST_SCHEMA,
    ANALYSIS_TRANSPORT_RESULT_SCHEMA,
    AnalysisProviderKind,
    AnalysisRequest,
)


IPFS_DATASETS_LOGIC_PROVIDER_ID: Final = "hammer"
IPFS_DATASETS_LOGIC_PROVIDER_VERSION: Final = "1.0.0"
HAMMER_ADAPTER_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/hammer-adapter-result@1"
)
HAMMER_PROVENANCE_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/hammer-provenance@1"
)
HAMMER_TRANSLATOR_ID: Final = "ipfs-datasets-py-hammer-adapter@1"

KNOWN_HAMMER_SOLVERS: Final = ("cvc5", "e", "vampire", "z3")
_HAMMER_IMPORT_LOCK: Final = threading.Lock()
_HAMMER_SYMAI_CONFIG_ROOT: tempfile.TemporaryDirectory[str] | None = None
# Semantic reasoning families routed through the analysis registry are
# deliberately separate from Hammer's target-language translation formats
# below.  In particular, FLogic and frame reasoning, and DCEC and deontic
# reasoning, must never be collapsed merely because bridges exist between
# them.
SUPPORTED_LOGIC_FAMILIES: Final = tuple(item.value for item in LogicFamily)
SUPPORTED_TRANSLATION_FAMILIES: Final = (
    "coq",
    "first_order",
    "isabelle",
    "lean",
    "lean4",
    "smtlib",
    "smtlib2",
    "tptp",
)
_FAMILY_ALIASES: Final = {
    "fol": "first_order",
    "first-order": "first_order",
    "lean_4": "lean4",
    "smt-lib": "smtlib",
    "smt-lib2": "smtlib2",
}
_FAMILY_ITP: Final = {
    "lean": "lean",
    "lean4": "lean",
    "coq": "coq",
    "isabelle": "isabelle",
}
_FAMILY_TARGET: Final = {
    "first_order": "tptp",
    "tptp": "tptp",
    "smtlib": "smtlib",
    "smtlib2": "smtlib",
}
_SOLVER_ALIASES: Final = {"eprover": "e"}
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class HammerAdapterStatus(str, Enum):
    """Stable, non-authoritative adapter outcomes."""

    TRANSLATED = "translated"
    CANDIDATE = "candidate"
    COUNTEREXAMPLE = "counterexample"
    UNKNOWN = "unknown"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    POLICY_DENIED = "policy_denied"


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _strings(
    value: Any,
    *,
    field_name: str,
    sort: bool = True,
) -> tuple[str, ...]:
    if value is None:
        raw: Sequence[Any] = ()
    elif isinstance(value, str):
        raw = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        raw = value
    else:
        raise ValueError(f"{field_name} must be a string or array of strings")
    result: list[str] = []
    for item in raw:
        normalized = _text(item, field_name=field_name)
        if normalized not in result:
            result.append(normalized)
    return tuple(sorted(result) if sort else result)


def _strict_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    converter = getattr(value, "to_dict", None)
    if not isinstance(value, Mapping) and callable(converter):
        value = converter()
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    try:
        encoded = canonical_json(dict(value))
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must contain canonical JSON values") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{field_name} must be an object")
    return decoded


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _family(value: Any) -> str:
    normalized = _text(value, field_name="translation_family").lower()
    normalized = _FAMILY_ALIASES.get(normalized, normalized)
    return normalized


def _solver_names(value: Any, *, field_name: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _SOLVER_ALIASES.get(name.lower(), name.lower())
                for name in _strings(value, field_name=field_name)
            }
        )
    )


def _digest(value: Mapping[str, Any], *, prefix: str) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _semantic_binding_projection(
    obligation: CodeProofObligation,
    payload: Mapping[str, Any],
    *,
    policy_id: str,
    environment_lock_id: str,
) -> dict[str, Any]:
    """Project supervisor semantic bindings explicitly across Hammer.

    The obligation identity already covers metadata, but carrying these
    dimensions as first-class provenance prevents consumers from relying on
    opaque content-ID transitivity when deciding freshness.
    """

    metadata = obligation.metadata
    toolchain_id = str(
        payload.get("toolchain_id")
        or metadata.get("code_proof_toolchain_id")
        or metadata.get("toolchain_id")
        or environment_lock_id
    ).strip()
    return {
        "goal_id": str(
            payload.get("goal_id")
            or metadata.get("goal_id")
            or metadata.get("objective_id")
            or ""
        ).strip(),
        "accepted_plan_id": str(
            payload.get("accepted_plan_id")
            or metadata.get("accepted_plan_id")
            or ""
        ).strip(),
        "assumptions_digest": str(
            payload.get("assumptions_digest")
            or metadata.get("assumptions_digest")
            or ""
        ).strip(),
        "toolchain_id": toolchain_id,
        "changed_scope_set_id": str(
            payload.get("changed_scope_set_id")
            or metadata.get("scope_set_id")
            or metadata.get("changed_scope_set_id")
            or ""
        ).strip(),
        "effect_scope_map": _provider_safe(
            payload.get("effect_scope_map")
            or metadata.get("effect_scope_map")
            or {}
        ),
        "policy_id": policy_id,
    }


def _minimum_positive(*values: int) -> int:
    positive = [value for value in values if value > 0]
    return min(positive) if positive else 0


def _seconds_within(milliseconds: int, *, field_name: str) -> int:
    """Return an integral Hammer seconds budget that never exceeds ``ms``."""

    seconds = milliseconds // 1000
    if seconds <= 0:
        raise ProofProviderError(
            ProviderFailureCode.RESOURCE_EXHAUSTED,
            f"{field_name} is below Hammer's one-second execution granularity",
            details={f"{field_name}_ms": milliseconds},
        )
    return seconds


def _memory_mb_within(memory_bytes: int) -> int:
    memory_mb = memory_bytes // (1024 * 1024)
    if memory_mb <= 0:
        raise ProofProviderError(
            ProviderFailureCode.RESOURCE_EXHAUSTED,
            "memory budget is below Hammer's one-MiB execution granularity",
            details={"memory_bytes": memory_bytes},
        )
    return memory_mb


def _provider_safe(value: Any) -> Any:
    """Convert Hammer records to the provider's deterministic JSON subset.

    The proof-provider protocol intentionally excludes binary floating-point
    values.  Hammer uses floats for observed seconds, so non-integral values
    are rendered as exact decimal strings in provider diagnostics.  Policy
    values produced by this adapter are integral before this function runs.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Hammer output contains a non-finite number")
        if value.is_integer():
            return int(value)
        return format(value, ".17g")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _provider_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_provider_safe(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _provider_safe(converter())
    raise ValueError(f"Hammer output contains unsupported {type(value).__name__}")


@dataclass(frozen=True)
class HammerSupervisorPolicy:
    """Supervisor-owned upper bounds for every Hammer invocation.

    A request may tighten these settings but cannot add a solver, enable
    network access, increase a resource budget, or replace the pinned
    environment lock.
    """

    allowed_solvers: tuple[str, ...] = ()
    timeout_ms: int = 30_000
    cpu_time_ms: int = 30_000
    memory_bytes: int = 512 * 1024 * 1024
    max_premises: int = 64
    max_parallel_processes: int = 4
    network_allowed: bool = False
    translation_families: tuple[str, ...] = SUPPORTED_TRANSLATION_FAMILIES
    fallback_checks: tuple[str, ...] = ()
    environment_lock: Mapping[str, Any] = field(default_factory=dict)
    target_itp: str = "lean"
    require_authoritative_reconstruction: bool = False
    # Compatibility input spellings matching Hammer's public policy.  The
    # canonical supervisor projection always uses integer ms/bytes.
    timeout_seconds: int | None = None
    cpu_seconds: int | None = None
    memory_mb: int | None = None

    def __post_init__(self) -> None:
        timeout_ms = self.timeout_ms
        cpu_time_ms = self.cpu_time_ms
        memory_bytes = self.memory_bytes
        if self.timeout_seconds is not None:
            seconds = _positive_int(
                self.timeout_seconds, field_name="timeout_seconds"
            )
            converted = seconds * 1000
            if self.timeout_ms != 30_000 and self.timeout_ms != converted:
                raise ValueError("timeout_ms and timeout_seconds disagree")
            timeout_ms = converted
        if self.cpu_seconds is not None:
            seconds = _positive_int(self.cpu_seconds, field_name="cpu_seconds")
            converted = seconds * 1000
            if self.cpu_time_ms != 30_000 and self.cpu_time_ms != converted:
                raise ValueError("cpu_time_ms and cpu_seconds disagree")
            cpu_time_ms = converted
        if self.memory_mb is not None:
            memory_mb = _positive_int(self.memory_mb, field_name="memory_mb")
            converted = memory_mb * 1024 * 1024
            if (
                self.memory_bytes != 512 * 1024 * 1024
                and self.memory_bytes != converted
            ):
                raise ValueError("memory_bytes and memory_mb disagree")
            memory_bytes = converted
        object.__setattr__(self, "timeout_ms", timeout_ms)
        object.__setattr__(self, "cpu_time_ms", cpu_time_ms)
        object.__setattr__(self, "memory_bytes", memory_bytes)
        solvers = _solver_names(
            self.allowed_solvers, field_name="allowed_solvers"
        )
        unknown = sorted(set(solvers) - set(KNOWN_HAMMER_SOLVERS))
        if unknown:
            raise ValueError(
                "allowed_solvers contains unknown Hammer solver families: "
                + ", ".join(unknown)
            )
        families = tuple(
            sorted({_family(item) for item in self.translation_families})
        )
        unknown_families = sorted(
            set(families) - set(SUPPORTED_TRANSLATION_FAMILIES)
        )
        if unknown_families:
            raise ValueError(
                "translation_families contains unsupported values: "
                + ", ".join(unknown_families)
            )
        for name in (
            "timeout_ms",
            "cpu_time_ms",
            "memory_bytes",
            "max_premises",
            "max_parallel_processes",
        ):
            _positive_int(getattr(self, name), field_name=name)
        if not isinstance(self.network_allowed, bool):
            raise ValueError("network_allowed must be a boolean")
        if not isinstance(self.require_authoritative_reconstruction, bool):
            raise ValueError(
                "require_authoritative_reconstruction must be a boolean"
            )
        target_itp = _text(self.target_itp, field_name="target_itp").lower()
        if target_itp == "lean4":
            target_itp = "lean"
        if target_itp not in {"lean", "coq", "isabelle"}:
            raise ValueError("target_itp must be lean, coq, or isabelle")
        lock = _strict_mapping(
            self.environment_lock, field_name="environment_lock"
        )
        object.__setattr__(self, "allowed_solvers", solvers)
        object.__setattr__(self, "translation_families", families)
        object.__setattr__(
            self,
            "fallback_checks",
            _strings(self.fallback_checks, field_name="fallback_checks"),
        )
        object.__setattr__(self, "environment_lock", lock)
        object.__setattr__(self, "target_itp", target_itp)

    @property
    def policy_id(self) -> str:
        return _digest(self.to_dict(), prefix="hammer-policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_solvers": list(self.allowed_solvers),
            "timeout_ms": self.timeout_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_bytes": self.memory_bytes,
            "max_premises": self.max_premises,
            "max_parallel_processes": self.max_parallel_processes,
            "network_allowed": self.network_allowed,
            "translation_families": list(self.translation_families),
            "fallback_checks": list(self.fallback_checks),
            "environment_lock": dict(self.environment_lock),
            "target_itp": self.target_itp,
            "require_authoritative_reconstruction": (
                self.require_authoritative_reconstruction
            ),
        }


# Compatibility spelling for callers naming the provider rather than Hammer.
IpfsDatasetsLogicProviderConfig = HammerSupervisorPolicy
IPFSDatasetsLogicProviderConfig = HammerSupervisorPolicy
HammerProviderPolicy = HammerSupervisorPolicy
IpfsDatasetsProviderPolicy = HammerSupervisorPolicy


@dataclass(frozen=True)
class EffectiveHammerPolicy:
    allowed_solvers: tuple[str, ...]
    timeout_ms: int
    cpu_time_ms: int
    memory_bytes: int
    max_premises: int
    max_parallel_processes: int
    network_allowed: bool
    fallback_checks: tuple[str, ...]
    environment_lock: Mapping[str, Any]
    target_itp: str
    policy_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_solvers": list(self.allowed_solvers),
            "timeout_ms": self.timeout_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_bytes": self.memory_bytes,
            "max_premises": self.max_premises,
            "max_parallel_processes": self.max_parallel_processes,
            "network_allowed": self.network_allowed,
            "fallback_checks": list(self.fallback_checks),
            "environment_lock": dict(self.environment_lock),
            "target_itp": self.target_itp,
            "policy_id": self.policy_id,
        }


@dataclass(frozen=True)
class HammerRequestBundle:
    """Canonical provider projection of a Hammer request and its bindings."""

    obligation_id: str
    translation_family: str
    hammer_request: Mapping[str, Any]
    premises: tuple[Mapping[str, Any], ...]
    environment_lock: Mapping[str, Any]
    portfolio_policy: Mapping[str, Any]
    fallback_checks: tuple[str, ...]
    upstream_receipt_ids: tuple[str, ...]
    provenance: Mapping[str, Any]
    _runtime: Any = field(default=None, repr=False, compare=False)

    @property
    def request(self) -> Mapping[str, Any]:
        return self.hammer_request

    @property
    def request_id(self) -> str:
        return str(self.hammer_request["request_id"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HAMMER_ADAPTER_SCHEMA_VERSION,
            "status": HammerAdapterStatus.TRANSLATED.value,
            "obligation_id": self.obligation_id,
            "translation_family": self.translation_family,
            "hammer_request": dict(self.hammer_request),
            "premises": [dict(premise) for premise in self.premises],
            "environment_lock": dict(self.environment_lock),
            "portfolio_policy": dict(self.portfolio_policy),
            "fallback_checks": list(self.fallback_checks),
            "upstream_receipt_ids": list(self.upstream_receipt_ids),
            "provenance": dict(self.provenance),
            "authoritative_assurance": "unverified",
            "kernel_checked": False,
            "proof_success": False,
        }


@dataclass(frozen=True)
class HammerPortfolioInvocation:
    """Arguments supplied to an injected Hammer portfolio runner."""

    bundle: HammerRequestBundle
    hammer_request: Any
    premises: tuple[Any, ...]
    environment_lock: Any
    hammer_policy: Any
    portfolio_policy: Any
    translations: tuple[Any, ...]
    attempt_specs: tuple[Any, ...]


PortfolioRunner = Callable[[HammerPortfolioInvocation], Any]


def _load_hammer() -> Any:
    global _HAMMER_SYMAI_CONFIG_ROOT
    try:
        with _HAMMER_IMPORT_LOCK:
            configured_root = os.environ.get("IPFS_DATASETS_PY_SYMAI_PREFIX")
            if not configured_root and _HAMMER_SYMAI_CONFIG_ROOT is None:
                _HAMMER_SYMAI_CONFIG_ROOT = tempfile.TemporaryDirectory(
                    prefix="ipfs-accelerate-symai-"
                )
            import_root = configured_root or _HAMMER_SYMAI_CONFIG_ROOT.name
            original_prefix = sys.prefix
            original_home = os.environ.get("HOME")
            if not configured_root:
                os.environ["IPFS_DATASETS_PY_SYMAI_PREFIX"] = import_root
            os.environ["HOME"] = import_root
            sys.prefix = import_root
            try:
                return importlib.import_module("ipfs_datasets_py.logic.hammers")
            finally:
                sys.prefix = original_prefix
                if original_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = original_home
                if not configured_root:
                    os.environ.pop("IPFS_DATASETS_PY_SYMAI_PREFIX", None)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise ProofProviderError(
            ProviderFailureCode.UNAVAILABLE,
            "ipfs_datasets_py Hammer portfolio is unavailable",
            details={
                "module": "ipfs_datasets_py.logic.hammers",
                "reason_code": "optional_dependency_import_failed",
            },
        ) from exc


def _obligation(value: Any) -> CodeProofObligation:
    if isinstance(value, CodeProofObligation):
        return value
    if isinstance(value, Mapping):
        return CodeProofObligation.from_dict(value)
    raise ValueError("obligation must be a CodeProofObligation or object")


def _effective_policy(
    base: HammerSupervisorPolicy,
    request: ProviderRequest,
    payload: Mapping[str, Any],
) -> EffectiveHammerPolicy:
    override_raw = payload.get("supervisor_policy") or {}
    override = _strict_mapping(override_raw, field_name="supervisor_policy")

    requested_solvers = _solver_names(
        override.get("allowed_solvers", base.allowed_solvers),
        field_name="supervisor_policy.allowed_solvers",
    )
    if not set(requested_solvers).issubset(base.allowed_solvers):
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            "request supervisor policy cannot expand the solver allowlist",
            details={
                "configured_allowed_solvers": list(base.allowed_solvers),
                "requested_allowed_solvers": list(requested_solvers),
            },
        )

    def bounded(name: str, configured: int, envelope: int = 0) -> int:
        requested = override.get(name, configured)
        _positive_int(requested, field_name=f"supervisor_policy.{name}")
        if requested > configured:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                f"request supervisor policy cannot increase {name}",
                details={"configured": configured, "requested": requested},
            )
        return _minimum_positive(requested, envelope)

    budget = request.resource_budget
    timeout_ms = bounded("timeout_ms", base.timeout_ms, budget.wall_time_ms)
    cpu_time_ms = bounded("cpu_time_ms", base.cpu_time_ms, budget.cpu_time_ms)
    memory_bytes = bounded("memory_bytes", base.memory_bytes, budget.memory_bytes)
    max_premises = bounded("max_premises", base.max_premises, budget.max_premises)
    max_processes = bounded(
        "max_parallel_processes",
        base.max_parallel_processes,
        budget.max_processes,
    )

    requested_network = override.get("network_allowed", base.network_allowed)
    if not isinstance(requested_network, bool):
        raise ValueError("supervisor_policy.network_allowed must be a boolean")
    network_allowed = bool(
        base.network_allowed
        and requested_network
        and request.network_allowed
        and budget.network_allowed
    )

    lock_override = override.get("environment_lock")
    if lock_override is not None:
        lock_override = _strict_mapping(
            lock_override, field_name="supervisor_policy.environment_lock"
        )
        if base.environment_lock and lock_override != dict(base.environment_lock):
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                "request supervisor policy cannot replace the environment lock",
            )
        environment_lock = lock_override
    else:
        environment_lock = dict(base.environment_lock)
    if not environment_lock:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            "a pinned Hammer environment_lock is required",
        )

    target_itp = str(override.get("target_itp", base.target_itp)).strip().lower()
    if target_itp == "lean4":
        target_itp = "lean"
    if target_itp != base.target_itp:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            "request supervisor policy cannot replace the target ITP",
        )

    fallback_checks = tuple(
        sorted(
            set(base.fallback_checks)
            | set(
                _strings(
                    override.get("fallback_checks"),
                    field_name="supervisor_policy.fallback_checks",
                )
            )
        )
    )
    identity_payload = {
        "allowed_solvers": list(requested_solvers),
        "timeout_ms": timeout_ms,
        "cpu_time_ms": cpu_time_ms,
        "memory_bytes": memory_bytes,
        "max_premises": max_premises,
        "max_parallel_processes": max_processes,
        "network_allowed": network_allowed,
        "fallback_checks": list(fallback_checks),
        "environment_lock": environment_lock,
        "target_itp": target_itp,
    }
    return EffectiveHammerPolicy(
        allowed_solvers=requested_solvers,
        timeout_ms=timeout_ms,
        cpu_time_ms=cpu_time_ms,
        memory_bytes=memory_bytes,
        max_premises=max_premises,
        max_parallel_processes=max_processes,
        network_allowed=network_allowed,
        fallback_checks=fallback_checks,
        environment_lock=environment_lock,
        target_itp=target_itp,
        policy_id=_digest(identity_payload, prefix="hammer-policy"),
    )


def _resolve_family(
    obligation: CodeProofObligation,
    payload: Mapping[str, Any],
    policy: HammerSupervisorPolicy,
) -> str:
    raw = payload.get("translation_family")
    if raw is None:
        raw = obligation.metadata.get("translation_family")
    if raw is None:
        raw = obligation.metadata.get("backend_id")
    if raw is None:
        raise ProofProviderError(
            ProviderFailureCode.UNSUPPORTED,
            "obligation does not declare a Hammer translation family",
            details={
                "reason_code": "translation_family_missing",
                "supported_translation_families": list(
                    policy.translation_families
                ),
            },
        )
    family = _family(raw)
    if family not in policy.translation_families:
        raise ProofProviderError(
            ProviderFailureCode.UNSUPPORTED,
            f"Hammer translation family {family!r} is not supported by policy",
            details={
                "reason_code": "translation_family_unsupported",
                "translation_family": family,
                "supported_translation_families": list(
                    policy.translation_families
                ),
            },
        )
    return family


def _premise_payloads(
    obligation: CodeProofObligation,
    payload: Mapping[str, Any],
    *,
    corpus_revision: str,
    itp: str,
    max_premises: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], dict[str, Any]]:
    if len(obligation.premise_ids) > max_premises:
        raise ProofProviderError(
            ProviderFailureCode.RESOURCE_EXHAUSTED,
            "obligation premise count exceeds supervisor policy",
            details={
                "premise_count": len(obligation.premise_ids),
                "max_premises": max_premises,
            },
        )
    raw = payload.get("premises")
    if raw is None:
        raw = obligation.metadata.get("premises", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("premises must be an array of explicit premise objects")

    by_id: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("premises must contain objects")
        premise_id = _text(item.get("premise_id"), field_name="premise_id")
        if premise_id in by_id:
            raise ValueError(f"duplicate explicit premise {premise_id!r}")
        by_id[premise_id] = item

    declared = set(obligation.premise_ids)
    supplied = set(by_id)
    missing = sorted(declared - supplied)
    unexpected = sorted(supplied - declared)
    if missing or unexpected:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            "explicit Hammer premises must exactly match obligation premise_ids",
            details={"missing_premise_ids": missing, "unexpected_premise_ids": unexpected},
        )

    premise_records: list[dict[str, Any]] = []
    all_receipts: set[str] = set(
        _strings(
            payload.get("upstream_receipt_ids"),
            field_name="upstream_receipt_ids",
        )
    )
    premise_provenance: dict[str, Any] = {}
    for rank, premise_id in enumerate(obligation.premise_ids):
        item = by_id[premise_id]
        statement = _text(item.get("statement"), field_name="premise.statement")
        item_revision = str(item.get("corpus_revision") or corpus_revision).strip()
        if item_revision != corpus_revision:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                "premise corpus revision does not match the obligation request",
                details={"premise_id": premise_id},
            )
        receipts = set(
            _strings(
                item.get("upstream_receipt_ids"),
                field_name="premise.upstream_receipt_ids",
            )
        )
        receipt_id = str(item.get("receipt_id") or "").strip()
        if receipt_id:
            receipts.add(receipt_id)
        all_receipts.update(receipts)
        record = {
            "schema_version": "1.0.0",
            "premise_id": premise_id,
            "statement": statement,
            "source_itp": itp,
            "corpus_revision": corpus_revision,
            "rank": rank,
            "score": 0,
            "selection_method": "supervisor-explicit-premises@1",
            "content_digest": item.get("content_digest"),
        }
        premise_records.append(record)
        premise_provenance[premise_id] = {
            "rank": rank,
            "upstream_receipt_ids": sorted(receipts),
        }
    all_receipts.update(
        _strings(
            obligation.metadata.get("upstream_receipt_ids"),
            field_name="obligation.metadata.upstream_receipt_ids",
        )
    )
    return (
        tuple(premise_records),
        tuple(sorted(all_receipts)),
        premise_provenance,
    )


def _hammer_selected_premise_payloads(
    hammer: Any,
    obligation: CodeProofObligation,
    payload: Mapping[str, Any],
    *,
    corpus_revision: str,
    itp: str,
    max_premises: int,
    hammer_policy: Any,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[str, ...],
    dict[str, Any],
    dict[str, Any],
] | None:
    """Delegate reviewed deterministic premise selection to Hammer.

    Selection may rank a pinned corpus, but it cannot mutate the theorem:
    selected premise identities must already be frozen into the canonical
    obligation.  This prevents a cache hit or selector update from silently
    changing the theorem being attempted.
    """

    selection_config = payload.get("premise_selection")
    if selection_config is None:
        return None
    config = _strict_mapping(
        selection_config, field_name="premise_selection"
    )
    manifest_payload = payload.get("corpus_manifest")
    if not isinstance(manifest_payload, Mapping):
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            "Hammer premise selection requires a pinned corpus_manifest",
            details={"reason_code": "premise_corpus_manifest_missing"},
        )
    manifest = hammer.CorpusManifest.from_dict(dict(manifest_payload))
    manifest.validate()
    if manifest.revision != corpus_revision:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            "premise corpus manifest revision does not match the request",
            details={
                "expected_corpus_revision": corpus_revision,
                "manifest_revision": manifest.revision,
            },
        )
    top_k = config.get("top_k", len(obligation.premise_ids))
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("premise_selection.top_k must be a positive integer")
    if top_k > max_premises:
        raise ProofProviderError(
            ProviderFailureCode.RESOURCE_EXHAUSTED,
            "premise selection exceeds the supervisor premise bound",
            details={"top_k": top_k, "max_premises": max_premises},
        )
    goal = hammer.GoalFeatures.from_statement(
        obligation.statement,
        theorem_id=str(config.get("theorem_id") or "") or None,
        imports=_strings(
            config.get("imports"), field_name="premise_selection.imports"
        ),
        extra_symbols=_strings(
            config.get("extra_symbols"),
            field_name="premise_selection.extra_symbols",
        ),
        extra_types=_strings(
            config.get("extra_types"),
            field_name="premise_selection.extra_types",
        ),
    )
    selection = hammer.select_premises(
        manifest,
        goal,
        top_k=top_k,
        policy=hammer_policy,
        exclude_theorem_ids=_strings(
            config.get("exclude_theorem_ids"),
            field_name="premise_selection.exclude_theorem_ids",
        ),
    )
    selection.validate()
    records = tuple(item.to_dict() for item in selection.selected)
    selected_ids = {str(item["premise_id"]) for item in records}
    declared_ids = set(obligation.premise_ids)
    if selected_ids != declared_ids:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            "Hammer-selected premises do not match canonical obligation premise_ids",
            details={
                "missing_premise_ids": sorted(declared_ids - selected_ids),
                "unexpected_premise_ids": sorted(selected_ids - declared_ids),
            },
        )
    receipts = set(
        _strings(
            payload.get("upstream_receipt_ids"),
            field_name="upstream_receipt_ids",
        )
    )
    receipts.update(
        _strings(
            obligation.metadata.get("upstream_receipt_ids"),
            field_name="obligation.metadata.upstream_receipt_ids",
        )
    )
    provenance = {
        str(item["premise_id"]): {
            "rank": item["rank"],
            "content_digest": item.get("content_digest"),
            "selection_method": item["selection_method"],
            "corpus_revision": item["corpus_revision"],
            "upstream_receipt_ids": sorted(receipts),
        }
        for item in records
    }
    selection_projection = _provider_safe(selection.to_dict())
    # Hammer records wall-clock creation time for diagnostics.  It is not a
    # semantic selector input and must not perturb the canonical request.
    selection_projection.pop("created_at", None)
    selection_projection["selected_premise_ids"] = sorted(selected_ids)
    return (
        records,
        tuple(sorted(receipts)),
        provenance,
        selection_projection,
    )


def _environment_lock(
    hammer: Any,
    value: Mapping[str, Any],
    *,
    itp: str,
    policy_id: str,
    allowed_solvers: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    lock = dict(value)
    lock_itp = str(lock.get("itp") or itp).strip().lower()
    if lock_itp == "lean4":
        lock_itp = "lean"
    if lock_itp != itp:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            "environment lock ITP does not match the Hammer request",
            details={"expected_itp": itp, "lock_itp": lock_itp},
        )
    lock["itp"] = lock_itp
    # Preserve a digest captured by Hammer itself when one is supplied.  A
    # newly declared supervisor lock binds the effective supervisor policy.
    # In both cases the supervisor policy id is independently carried in the
    # request and provenance projection.
    lock.setdefault("policy_digest", policy_id)
    lock.setdefault("pinned_at", _EPOCH.isoformat())
    if not lock.get("lock_id"):
        identity = dict(lock)
        identity.pop("lock_id", None)
        lock["lock_id"] = _digest(identity, prefix="hammer-environment")

    versions = lock.get("solver_versions") or {}
    if not isinstance(versions, Mapping):
        raise ValueError("environment_lock.solver_versions must be an object")
    missing_versions = sorted(set(allowed_solvers) - set(versions))
    if missing_versions:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            "environment lock must pin every allowed solver version",
            details={"missing_solver_versions": missing_versions},
        )
    try:
        record = hammer.EnvironmentLockRecord.from_dict(lock)
        record.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ProofProviderError(
            ProviderFailureCode.MALFORMED_REQUEST,
            f"invalid Hammer environment lock: {exc}",
        ) from exc
    return record, _provider_safe(record.to_dict())


def translate_obligation_to_hammer_request(
    obligation: CodeProofObligation | Mapping[str, Any],
    *,
    premises: Sequence[Mapping[str, Any]] = (),
    policy: HammerSupervisorPolicy | None = None,
    translation_family: str | None = None,
    upstream_receipt_ids: Sequence[str] = (),
    resource_budget: ResourceBudget | Mapping[str, Any] | None = None,
    request_id: str = "hammer-adapter-translation",
) -> HammerRequestBundle:
    """Translate one obligation without invoking a solver.

    This helper uses the same provider path as production requests, making it
    useful to proof-plan compilers and cache-key builders.
    """

    config = policy or HammerSupervisorPolicy()
    budget = (
        resource_budget
        if isinstance(resource_budget, ResourceBudget)
        else ResourceBudget.from_dict(resource_budget or {})
    )
    payload: dict[str, Any] = {
        "obligation": (
            obligation.to_dict()
            if isinstance(obligation, CodeProofObligation)
            else dict(obligation)
        ),
        "premises": list(premises),
        "upstream_receipt_ids": list(upstream_receipt_ids),
    }
    if translation_family is not None:
        payload["translation_family"] = translation_family
    request = ProviderRequest(
        request_id=request_id,
        operation=ProofProviderOperation.TRANSLATE,
        payload=payload,
        resource_budget=budget,
        network_allowed=False,
    )
    return IpfsDatasetsLogicProvider(config)._build_bundle(request)


def _status_from_hammer(value: Any) -> HammerAdapterStatus:
    raw = str(getattr(value, "value", value)).strip().lower()
    return {
        "candidate": HammerAdapterStatus.CANDIDATE,
        "counterexample": HammerAdapterStatus.COUNTEREXAMPLE,
        "timeout": HammerAdapterStatus.TIMED_OUT,
        "timed_out": HammerAdapterStatus.TIMED_OUT,
        "unavailable": HammerAdapterStatus.UNAVAILABLE,
        "unsupported": HammerAdapterStatus.UNSUPPORTED,
        "unsupported_translation": HammerAdapterStatus.UNSUPPORTED,
        "policy_denied": HammerAdapterStatus.POLICY_DENIED,
    }.get(raw, HammerAdapterStatus.UNKNOWN)


def adapt_hammer_result(
    result: Any,
    bundle: HammerRequestBundle,
    *,
    hammer_receipt_id: str = "",
) -> dict[str, Any]:
    """Project Hammer output without weakening its trust or provenance."""

    if hasattr(result, "result") and hasattr(result, "receipt_id"):
        hammer_receipt_id = str(result.receipt_id)
        result = result.result
    raw = result.to_dict() if hasattr(result, "to_dict") else result
    if not isinstance(raw, Mapping):
        raise ValueError("Hammer result must provide a mapping or to_dict()")

    request_raw = raw.get("request") or {}
    result_request_id = str(
        (
            request_raw.get("request_id")
            if isinstance(request_raw, Mapping)
            else ""
        )
        or raw.get("request_id")
        or ""
    )
    if result_request_id and result_request_id != bundle.request_id:
        raise ValueError("Hammer result request_id does not match adapter request")

    attempts = raw.get("solver_attempts")
    if attempts is None:
        attempts = raw.get("attempts", ())
    candidate = raw.get("proof_candidate")
    status = _status_from_hammer(raw.get("status", "unknown"))
    if candidate is not None and status is not HammerAdapterStatus.CANDIDATE:
        raise ValueError(
            "Hammer proof candidate is inconsistent with the portfolio status"
        )
    if candidate is None and status is HammerAdapterStatus.CANDIDATE:
        raise ValueError(
            "Hammer candidate status requires an exact proof candidate record"
        )
    provenance = dict(bundle.provenance)
    attempt_provenance: dict[str, Any] = {}
    for attempt in attempts or ():
        attempt_dict = (
            attempt.to_dict() if hasattr(attempt, "to_dict") else attempt
        )
        if not isinstance(attempt_dict, Mapping):
            raise ValueError("Hammer solver attempts must be objects")
        attempt_id = _text(
            attempt_dict.get("attempt_id"), field_name="attempt.attempt_id"
        )
        if str(attempt_dict.get("request_id") or bundle.request_id) != bundle.request_id:
            raise ValueError("Hammer solver attempt request_id does not match")
        attempt_provenance[attempt_id] = {
            "request_id": bundle.request_id,
            "obligation_id": bundle.obligation_id,
            "repository_tree_id": provenance["repository_tree_id"],
            "translation_id": str(attempt_dict.get("translation_id") or ""),
            "upstream_receipt_ids": list(bundle.upstream_receipt_ids),
            "hammer_receipt_id": hammer_receipt_id,
        }

    candidate_provenance: dict[str, Any] = {}
    if candidate is not None:
        candidate_dict = (
            candidate.to_dict() if hasattr(candidate, "to_dict") else candidate
        )
        if not isinstance(candidate_dict, Mapping):
            raise ValueError("Hammer proof candidate must be an object")
        candidate_id = _text(
            candidate_dict.get("candidate_id"),
            field_name="proof_candidate.candidate_id",
        )
        if str(candidate_dict.get("request_id") or "") != bundle.request_id:
            raise ValueError("Hammer proof candidate request_id does not match")
        attempt_id = _text(
            candidate_dict.get("solver_attempt_id"),
            field_name="proof_candidate.solver_attempt_id",
        )
        if attempt_id not in attempt_provenance:
            raise ValueError("Hammer candidate references an unknown solver attempt")
        candidate_provenance[candidate_id] = {
            "request_id": bundle.request_id,
            "solver_attempt_id": attempt_id,
            "obligation_id": bundle.obligation_id,
            "repository_tree_id": provenance["repository_tree_id"],
            "upstream_receipt_ids": list(bundle.upstream_receipt_ids),
            "hammer_receipt_id": hammer_receipt_id,
            "trusted": False,
        }

    provenance.update(
        {
            "solver_attempts": attempt_provenance,
            "proof_candidates": candidate_provenance,
            "hammer_receipt_id": hammer_receipt_id,
        }
    )
    return {
        "schema_version": HAMMER_ADAPTER_SCHEMA_VERSION,
        "status": status.value,
        "hammer_result": _provider_safe(raw),
        "fallback_checks": list(bundle.fallback_checks),
        "upstream_receipt_ids": list(bundle.upstream_receipt_ids),
        "provenance": _provider_safe(provenance),
        "authoritative_assurance": "unverified",
        "kernel_checked": False,
        "proof_success": False,
    }


class IpfsDatasetsLogicProvider:
    """Lazy, policy-bounded provider facade over the installed Hammer package."""

    provider_id = IPFS_DATASETS_LOGIC_PROVIDER_ID
    provider_version = IPFS_DATASETS_LOGIC_PROVIDER_VERSION
    protocol_version = PROOF_PROVIDER_PROTOCOL_VERSION

    def __init__(
        self,
        policy: HammerSupervisorPolicy | None = None,
        *,
        portfolio_runner: PortfolioRunner | None = None,
        verification_cache: FormalVerificationCache | None = None,
        proof_cache: FormalVerificationCache | None = None,
        cache: FormalVerificationCache | None = None,
        kernel_verifier: Any = None,
    ) -> None:
        self.policy = policy or HammerSupervisorPolicy()
        if not isinstance(self.policy, HammerSupervisorPolicy):
            raise ValueError("policy must be a HammerSupervisorPolicy")
        if portfolio_runner is not None and not callable(portfolio_runner):
            raise ValueError("portfolio_runner must be callable")
        supplied_caches = [
            item
            for item in (verification_cache, proof_cache, cache)
            if item is not None
        ]
        if len({id(item) for item in supplied_caches}) > 1:
            raise ValueError(
                "verification_cache, proof_cache, and cache must identify "
                "the same cache when more than one is supplied"
            )
        selected_cache = supplied_caches[0] if supplied_caches else None
        if selected_cache is not None and not isinstance(
            selected_cache, FormalVerificationCache
        ):
            raise ValueError(
                "verification_cache must be a FormalVerificationCache"
            )
        self._portfolio_runner = portfolio_runner
        if kernel_verifier is not None and not callable(
            getattr(kernel_verifier, "reconstruct_and_verify", None)
        ):
            raise ValueError(
                "kernel_verifier must expose reconstruct_and_verify"
            )
        self.kernel_verifier = kernel_verifier
        self.verification_cache = selected_cache
        self.proof_cache = selected_cache

    def capabilities(self) -> ProofProviderCapability:
        return ProofProviderCapability(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            protocol_versions=(self.protocol_version,),
            operations=(
                ProofProviderOperation.CAPABILITY,
                ProofProviderOperation.TRANSLATE,
                ProofProviderOperation.PROVE,
                *(
                    (
                        ProofProviderOperation.RECONSTRUCT,
                        ProofProviderOperation.VERIFY,
                    )
                    if self.kernel_verifier is not None
                    else ()
                ),
            ),
            isolation=(
                ProofProviderIsolation.IN_PROCESS,
                ProofProviderIsolation.SUBPROCESS,
            ),
            network_access_required=False,
            resource_limits_supported=True,
            metadata={
                "adapter_schema": HAMMER_ADAPTER_SCHEMA_VERSION,
                "hammer_import": "lazy",
                "translation_families": list(self.policy.translation_families),
                "allowed_solvers": list(self.policy.allowed_solvers),
                "network_allowed": self.policy.network_allowed,
                "max_premises": self.policy.max_premises,
                "candidate_authoritative": False,
                "kernel_reconstruction_required": True,
                "kernel_reconstruction_available": (
                    self.kernel_verifier is not None
                ),
                "trust_aware_cache_enabled": self.verification_cache is not None,
                "cross_process_single_flight": self.verification_cache is not None,
                "proof_attempted": False,
                "proof_success": False,
            },
        )

    def capability(self, request: ProviderRequest) -> Mapping[str, Any]:
        return self.capabilities().to_dict()

    def _unsupported(
        self,
        request: ProviderRequest,
        exc: ProofProviderError,
        *,
        obligation: CodeProofObligation | None = None,
        policy: EffectiveHammerPolicy | None = None,
    ) -> ProviderResponse:
        details = dict(exc.failure.details)
        fallbacks = set(self.policy.fallback_checks)
        if obligation is not None:
            fallbacks.update(obligation.fallback_checks)
        if policy is not None:
            fallbacks.update(policy.fallback_checks)
        details.update(
            {
                "status": HammerAdapterStatus.UNSUPPORTED.value,
                "fallback_checks": sorted(fallbacks),
                "authoritative_assurance": "unverified",
                "proof_success": False,
            }
        )
        return ProviderResponse.failure(
            request,
            ProviderFailureCode.UNSUPPORTED,
            exc.failure.message,
            details=details,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )

    def _build_bundle(self, request: ProviderRequest) -> HammerRequestBundle:
        payload = request.payload
        try:
            obligation = _obligation(payload.get("obligation"))
        except (TypeError, ValueError) as exc:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                f"invalid code proof obligation: {exc}",
            ) from exc
        family = _resolve_family(obligation, payload, self.policy)
        policy = _effective_policy(self.policy, request, payload)
        itp = _FAMILY_ITP.get(family, policy.target_itp)
        corpus_revision = str(
            payload.get("corpus_revision")
            or obligation.metadata.get("corpus_revision")
            or obligation.repository_tree_id
        ).strip()
        hammer = _load_hammer()
        lock_record, lock_dict = _environment_lock(
            hammer,
            policy.environment_lock,
            itp=itp,
            policy_id=policy.policy_id,
            allowed_solvers=policy.allowed_solvers,
        )

        hammer_policy = hammer.HammerPolicy(
            timeout_seconds=_seconds_within(
                policy.timeout_ms, field_name="timeout"
            ),
            cpu_seconds=_seconds_within(
                policy.cpu_time_ms, field_name="cpu_time"
            ),
            memory_mb=_memory_mb_within(policy.memory_bytes),
            network_allowed=policy.network_allowed,
            allowed_solvers=list(policy.allowed_solvers),
            allow_learned_premise_selector=False,
            allow_llm_premise_ranking=False,
            max_premises=policy.max_premises,
            allow_native_automation_fallback=False,
            allow_llm_decomposition_hints=False,
        )
        hammer_policy.validate()
        selected = _hammer_selected_premise_payloads(
            hammer,
            obligation,
            payload,
            corpus_revision=corpus_revision,
            itp=itp,
            max_premises=policy.max_premises,
            hammer_policy=hammer_policy,
        )
        if selected is None:
            premise_dicts, upstream_receipts, premise_provenance = (
                _premise_payloads(
                    obligation,
                    payload,
                    corpus_revision=corpus_revision,
                    itp=itp,
                    max_premises=policy.max_premises,
                )
            )
            premise_selection: dict[str, Any] = {
                "selection_method": "supervisor-explicit-premises@1",
                "selected_premise_ids": list(obligation.premise_ids),
            }
        else:
            (
                premise_dicts,
                upstream_receipts,
                premise_provenance,
                premise_selection,
            ) = selected
        semantic_bindings = _semantic_binding_projection(
            obligation,
            payload,
            policy_id=policy.policy_id,
            environment_lock_id=lock_dict["lock_id"],
        )
        request_identity = {
            "obligation_id": obligation.obligation_id,
            "repository_tree_id": obligation.repository_tree_id,
            "semantic_bindings": semantic_bindings,
            "translation_family": family,
            "premises": _provider_safe(premise_dicts),
            "premise_selection": premise_selection,
            "upstream_receipt_ids": list(upstream_receipts),
            "environment_lock": lock_dict,
            "policy_id": policy.policy_id,
            "translator_id": HAMMER_TRANSLATOR_ID,
        }
        hammer_request_id = _digest(request_identity, prefix="hammer-request")
        hammer_request = hammer.HammerRequest(
            request_id=hammer_request_id,
            itp=hammer.ITPKind(itp),
            theorem_id=obligation.obligation_id,
            goal_statement=obligation.statement,
            corpus_revision=corpus_revision,
            policy=hammer_policy,
            created_at=_EPOCH,
            metadata={
                "obligation_id": obligation.obligation_id,
                "repository_id": obligation.repository_id,
                "repository_tree_id": obligation.repository_tree_id,
                "ast_scope_ids": list(obligation.ast_scope_ids),
                "premise_ids": list(obligation.premise_ids),
                "premise_selection": premise_selection,
                "semantic_bindings": semantic_bindings,
                "translation_family": family,
                "environment_lock_id": lock_dict["lock_id"],
                "policy_id": policy.policy_id,
                "translator_id": HAMMER_TRANSLATOR_ID,
                "upstream_receipt_ids": list(upstream_receipts),
            },
        )
        hammer_request.validate()
        portfolio_policy = hammer.PortfolioPolicy(
            hammer_policy=hammer_policy,
            max_parallel_processes=policy.max_parallel_processes,
            cancel_on_first_conclusive=True,
        )
        portfolio_policy.validate()
        provenance = {
            "schema_version": HAMMER_PROVENANCE_SCHEMA_VERSION,
            "request_id": hammer_request_id,
            "obligation_id": obligation.obligation_id,
            "repository_id": obligation.repository_id,
            "repository_tree_id": obligation.repository_tree_id,
            "ast_scope_ids": list(obligation.ast_scope_ids),
            "premise_ids": list(obligation.premise_ids),
            "semantic_bindings": semantic_bindings,
            "premises": premise_provenance,
            "premise_selection": premise_selection,
            "upstream_receipt_ids": list(upstream_receipts),
            "environment_lock_id": lock_dict["lock_id"],
            "policy_id": policy.policy_id,
            "translator_id": HAMMER_TRANSLATOR_ID,
        }
        fallback_checks = tuple(
            sorted(
                set(self.policy.fallback_checks)
                | set(policy.fallback_checks)
                | set(obligation.fallback_checks)
            )
        )
        # Keep validated objects on this immutable, invocation-local bundle;
        # providers can be called concurrently and must not use shared
        # "last request" state.
        runtime = (
            hammer,
            obligation,
            policy,
            hammer_request,
            tuple(
                hammer.PremiseRecord.from_dict(dict(item))
                for item in premise_dicts
            ),
            lock_record,
            hammer_policy,
            portfolio_policy,
        )
        return HammerRequestBundle(
            obligation_id=obligation.obligation_id,
            translation_family=family,
            hammer_request=_provider_safe(hammer_request.to_dict()),
            premises=tuple(_provider_safe(item) for item in premise_dicts),
            environment_lock=lock_dict,
            portfolio_policy={
                "hammer_policy": _provider_safe(hammer_policy.to_dict()),
                "solver_budgets": {},
                "executable_overrides": {},
                "max_parallel_processes": policy.max_parallel_processes,
                "cancel_on_first_conclusive": True,
                "supervisor_policy_id": policy.policy_id,
            },
            fallback_checks=fallback_checks,
            upstream_receipt_ids=upstream_receipts,
            provenance=provenance,
            _runtime=runtime,
        )

    def translate(
        self, request: ProviderRequest
    ) -> Mapping[str, Any] | ProviderResponse:
        obligation: CodeProofObligation | None = None
        policy: EffectiveHammerPolicy | None = None
        try:
            if isinstance(request.payload.get("obligation"), Mapping):
                obligation = _obligation(request.payload["obligation"])
            return self._build_bundle(request).to_dict()
        except ProofProviderError as exc:
            if exc.code is ProviderFailureCode.UNSUPPORTED:
                return self._unsupported(
                    request, exc, obligation=obligation, policy=policy
                )
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                f"could not adapt obligation to Hammer: {exc}",
            ) from exc

    def build_request(self, request: ProviderRequest) -> HammerRequestBundle:
        """Public typed spelling used by proof planners and cache builders."""

        return self._build_bundle(request)

    def _cache_key(
        self,
        request: ProviderRequest,
        bundle: HammerRequestBundle,
        effective: EffectiveHammerPolicy,
    ) -> ProofCacheKey:
        """Bind the complete Hammer execution identity for caching/flight work."""

        payload = request.payload
        (
            _hammer,
            obligation,
            _runtime_policy,
            _hammer_request,
            _premises,
            _lock_record,
            _hammer_policy,
            _portfolio_policy,
        ) = bundle._runtime
        solver_versions = bundle.environment_lock.get("solver_versions") or {}
        corpus_revision = str(
            bundle.hammer_request.get("corpus_revision")
            or obligation.metadata.get("corpus_revision")
            or obligation.repository_tree_id
        )
        kernel = payload.get("kernel")
        if kernel is None:
            kernel = {
                "kernel_id": str(
                    payload.get("kernel_id")
                    or "independent-kernel-provider-required"
                ),
                "kernel_version": str(payload.get("kernel_version") or ""),
            }
        toolchain = payload.get("toolchain")
        if toolchain is None:
            toolchain = {
                "toolchain_id": str(
                    payload.get("toolchain_id")
                    or bundle.environment_lock.get("lock_id")
                ),
                "environment_lock": dict(bundle.environment_lock),
            }
        theorem_registry = payload.get("theorem_registry")
        if theorem_registry is None:
            theorem_registry = {
                "theorem_registry_id": str(
                    payload.get("theorem_registry_id") or corpus_revision
                ),
                "corpus_revision": corpus_revision,
            }
        return build_proof_cache_key(
            obligation=obligation.to_dict(),
            premises=bundle.premises,
            translator={
                "translator_id": HAMMER_TRANSLATOR_ID,
                "adapter_version": self.provider_version,
                "translation_family": bundle.translation_family,
            },
            solver={
                "solver_ids": list(effective.allowed_solvers),
                "solver_versions": dict(solver_versions),
            },
            kernel=kernel,
            toolchain=toolchain,
            theorem_registry=theorem_registry,
            policy=effective.to_dict(),
            resource_budget=request.resource_budget.to_dict(),
            candidate_tree={
                "candidate_tree_id": obligation.repository_tree_id,
                "repository_id": obligation.repository_id,
            },
        )

    def build_cache_key(self, request: ProviderRequest) -> ProofCacheKey:
        """Return the exact key used for proof-cache and single-flight work."""

        bundle = self._build_bundle(request)
        effective = bundle._runtime[2]
        return self._cache_key(request, bundle, effective)

    def _translation_records(
        self,
        hammer: Any,
        bundle: HammerRequestBundle,
        hammer_request: Any,
        obligation: CodeProofObligation,
        payload: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        raw = payload.get("translations")
        if raw is None:
            # Legal/logic lowerers must still cross the Hammer typed boundary;
            # this is an input alias, not a parallel supervisor proof path.
            raw = payload.get("legal_logic_translations")
        if raw is None:
            raw = obligation.metadata.get("hammer_translations")
        if raw is None:
            raw = obligation.metadata.get("legal_logic_translations")
        if raw is None:
            target = _FAMILY_TARGET.get(bundle.translation_family)
            statement_format = str(
                obligation.metadata.get("statement_format") or ""
            ).lower()
            if target and statement_format in {
                target,
                "smtlib2" if target == "smtlib" else target,
            }:
                translation_id = _digest(
                    {
                        "request_id": bundle.request_id,
                        "target": target,
                        "statement": obligation.statement,
                    },
                    prefix="hammer-translation",
                )
                raw = (
                    {
                        "translation_id": translation_id,
                        "request_id": bundle.request_id,
                        "target": target,
                        "status": "supported",
                        "source_construct": obligation.obligation_id,
                        "translated_text": obligation.statement,
                        "obligations": [],
                        "unsupported_reason": None,
                    },
                )
            else:
                raise ProofProviderError(
                    ProviderFailureCode.UNSUPPORTED,
                    "obligation has no reviewed Hammer lowering artifact",
                    details={
                        "reason_code": "lowering_artifact_missing",
                        "translation_family": bundle.translation_family,
                    },
                )
        if not isinstance(raw, Sequence) or isinstance(
            raw, (str, bytes, bytearray)
        ):
            raise ValueError("translations must be an array")
        records = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("translations must contain objects")
            normalized = dict(item)
            normalized.setdefault("schema_version", "1.0.0")
            normalized.setdefault("request_id", bundle.request_id)
            if normalized["request_id"] != bundle.request_id:
                raise ValueError("translation request_id does not match Hammer request")
            record = hammer.TranslationRecord.from_dict(normalized)
            record.validate()
            records.append(record)
        if not records:
            raise ProofProviderError(
                ProviderFailureCode.UNSUPPORTED,
                "obligation has no Hammer lowering artifacts",
                details={"reason_code": "lowering_artifact_missing"},
            )
        return tuple(sorted(records, key=lambda item: item.translation_id))

    def _default_run(self, invocation: HammerPortfolioInvocation) -> Any:
        hammer = _load_hammer()
        portfolio = hammer.SolverPortfolio(invocation.portfolio_policy)
        run = portfolio.run(
            invocation.hammer_request.request_id,
            invocation.attempt_specs,
        )
        normalized = hammer.normalize_portfolio_run(
            run,
            request_id=invocation.hammer_request.request_id,
            premise_ids=[
                premise.premise_id for premise in invocation.premises
            ],
        )
        status = hammer.aggregate_recommended_status(normalized.values())
        if not run.attempts:
            status = (
                hammer.HammerResultStatus.UNAVAILABLE
                if run.denied
                else hammer.HammerResultStatus.UNKNOWN
            )
        elif all(
            attempt.verdict is hammer.SolverVerdict.TIMEOUT
            for attempt in run.attempts
        ):
            status = hammer.HammerResultStatus.TIMEOUT
        candidate = None
        if status is hammer.HammerResultStatus.CANDIDATE:
            for attempt in run.attempts:
                evidence = normalized.get(attempt.attempt_id)
                if (
                    evidence is not None
                    and evidence.recommended_status
                    is hammer.HammerResultStatus.CANDIDATE
                ):
                    candidate = hammer.build_proof_candidate_record(
                        evidence,
                        candidate_id=_digest(
                            {
                                "request_id": invocation.bundle.request_id,
                                "attempt_id": attempt.attempt_id,
                            },
                            prefix="hammer-candidate",
                        ),
                        request_id=invocation.bundle.request_id,
                        solver_attempt_id=attempt.attempt_id,
                    )
                    break
        return {
            "request_id": invocation.bundle.request_id,
            "status": status.value,
            "attempts": [attempt.to_dict() for attempt in run.attempts],
            "proof_candidate": (
                candidate.to_dict() if candidate is not None else None
            ),
            "portfolio_run": run.to_dict(),
            "normalized_evidence": {
                key: value.to_dict() for key, value in normalized.items()
            },
        }

    def prove(
        self, request: ProviderRequest
    ) -> Mapping[str, Any] | ProviderResponse:
        obligation: CodeProofObligation | None = None
        effective: EffectiveHammerPolicy | None = None
        bundle: HammerRequestBundle | None = None
        try:
            obligation = _obligation(request.payload.get("obligation"))
            bundle = self._build_bundle(request)
            (
                hammer,
                runtime_obligation,
                _runtime_policy,
                hammer_request,
                premises,
                lock_record,
                hammer_policy,
                portfolio_policy,
            ) = bundle._runtime
            effective = _runtime_policy
            translations = self._translation_records(
                hammer,
                bundle,
                hammer_request,
                runtime_obligation,
                request.payload,
            )
            attempts = tuple(
                hammer.PortfolioAttemptSpec(
                    translation=translation,
                    solver_name=solver,
                )
                for translation in translations
                for solver in effective.allowed_solvers
            )
            if not attempts:
                raise ProofProviderError(
                    ProviderFailureCode.UNSUPPORTED,
                    "supervisor policy allows no Hammer solver attempts",
                    details={"reason_code": "solver_allowlist_empty"},
                )
            invocation = HammerPortfolioInvocation(
                bundle=bundle,
                hammer_request=hammer_request,
                premises=premises,
                environment_lock=lock_record,
                hammer_policy=hammer_policy,
                portfolio_policy=portfolio_policy,
                translations=translations,
                attempt_specs=attempts,
            )
            runner = self._portfolio_runner or self._default_run

            def execute_portfolio() -> dict[str, Any]:
                raw_result = runner(invocation)
                projected = adapt_hammer_result(raw_result, bundle)
                projected["environment_lock"] = dict(bundle.environment_lock)
                projected["portfolio_policy"] = dict(bundle.portfolio_policy)
                projected["premises"] = [
                    dict(premise) for premise in bundle.premises
                ]
                return projected

            if self.verification_cache is None:
                projected_result = execute_portfolio()
            else:
                cache_key = self._cache_key(request, bundle, effective)
                shared = self.verification_cache.single_flight(
                    cache_key,
                    execute_portfolio,
                    lease_seconds=max(
                        1, (effective.timeout_ms + 999) // 1000 + 5
                    ),
                    wait_timeout_seconds=max(
                        1, (effective.timeout_ms + 999) // 1000 + 30
                    ),
                )
                if not isinstance(shared, Mapping):
                    raise ValueError("shared Hammer result must be an object")
                projected_result = dict(shared)

            projected_result["authoritative_reconstruction_required"] = (
                self.policy.require_authoritative_reconstruction
            )
            if (
                self.policy.require_authoritative_reconstruction
                and projected_result.get("status")
                == HammerAdapterStatus.CANDIDATE.value
            ):
                candidate = (
                    projected_result.get("hammer_result", {}).get(
                        "proof_candidate"
                    )
                    if isinstance(
                        projected_result.get("hammer_result"), Mapping
                    )
                    else None
                )
                if self.kernel_verifier is None:
                    return ProviderResponse.failure(
                        request,
                        ProviderFailureCode.UNSUPPORTED,
                        "policy requires independent kernel reconstruction",
                        details={
                            "status": HammerAdapterStatus.UNSUPPORTED.value,
                            "reason_code": (
                                "independent_kernel_provider_required"
                            ),
                            "candidate": _provider_safe(candidate),
                            "provenance": dict(bundle.provenance),
                            "proof_success": False,
                            "authoritative_assurance": "unverified",
                        },
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                    )
                if (
                    not isinstance(candidate, Mapping)
                    or not request.payload.get("goal_snapshot")
                    or not request.payload.get("native_source")
                    or not request.payload.get("kernel_id")
                ):
                    return ProviderResponse.failure(
                        request,
                        ProviderFailureCode.UNSUPPORTED,
                        "policy-required reconstruction inputs are missing",
                        details={
                            "status": HammerAdapterStatus.UNSUPPORTED.value,
                            "reason_code": (
                                "authoritative_reconstruction_inputs_required"
                            ),
                            "candidate": _provider_safe(candidate),
                            "provenance": dict(bundle.provenance),
                            "proof_success": False,
                            "authoritative_assurance": "unverified",
                        },
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                    )
                reconstruction_payload = dict(request.payload)
                reconstruction_payload["proof_candidate"] = dict(candidate)
                reconstruction_request = ProviderRequest(
                    request_id=request.request_id,
                    operation=request.operation,
                    payload=reconstruction_payload,
                    resource_budget=request.resource_budget,
                    network_allowed=request.network_allowed,
                    deadline_unix_ms=request.deadline_unix_ms,
                )
                return self.reconstruct(reconstruction_request)
            return projected_result
        except ProofProviderError as exc:
            if exc.code is ProviderFailureCode.UNSUPPORTED:
                return self._unsupported(
                    request,
                    exc,
                    obligation=obligation,
                    policy=effective,
                )
            raise
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            details: dict[str, Any] = {
                "status": HammerAdapterStatus.TIMED_OUT.value,
                "reason_code": "hammer_execution_timed_out",
                "proof_success": False,
                "authoritative_assurance": "unverified",
                "fallback_checks": sorted(
                    set(self.policy.fallback_checks)
                    | set(obligation.fallback_checks if obligation else ())
                    | set(effective.fallback_checks if effective else ())
                ),
            }
            if bundle is not None:
                details.update(
                    {
                        "hammer_request_id": bundle.request_id,
                        "provenance": dict(bundle.provenance),
                    }
                )
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.TIMED_OUT,
                f"Hammer portfolio timed out: {exc}",
                retryable=True,
                details=details,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                f"invalid Hammer portfolio request or result: {exc}",
            ) from exc

    def reconstruct(
        self, request: ProviderRequest
    ) -> Mapping[str, Any] | ProviderResponse:
        """Independently reconstruct one untrusted Hammer candidate."""

        if self.kernel_verifier is None:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.UNSUPPORTED,
                "independent kernel reconstruction is unavailable",
                details={
                    "status": HammerAdapterStatus.UNSUPPORTED.value,
                    "reason_code": "independent_kernel_provider_required",
                    "proof_success": False,
                    "authoritative_assurance": "unverified",
                },
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )

        bundle: HammerRequestBundle | None = None
        try:
            bundle = self._build_bundle(request)
            (
                hammer,
                obligation,
                effective,
                hammer_request,
                _premises,
                environment_lock,
                _hammer_policy,
                _portfolio_policy,
            ) = bundle._runtime
            candidate_raw = request.payload.get("proof_candidate")
            snapshot_raw = request.payload.get("goal_snapshot")
            native_source = _text(
                request.payload.get("native_source"),
                field_name="native_source",
            )
            if not isinstance(candidate_raw, Mapping):
                raise ValueError("proof_candidate must be an object")
            if not isinstance(snapshot_raw, Mapping):
                raise ValueError("goal_snapshot must be an object")
            candidate = hammer.ProofCandidateRecord.from_dict(
                dict(candidate_raw)
            )
            candidate.validate()
            if candidate.request_id != bundle.request_id:
                raise ValueError(
                    "proof_candidate.request_id does not match Hammer request"
                )
            snapshot = hammer.GoalSnapshot.from_dict(dict(snapshot_raw))
            snapshot.validate()
            if snapshot.goal_text.strip() != obligation.statement.strip():
                raise ValueError(
                    "goal_snapshot.goal_text does not match the obligation"
                )

            semantic_bindings = dict(
                bundle.provenance.get("semantic_bindings") or {}
            )
            toolchain_id = _text(
                request.payload.get("toolchain_id")
                or semantic_bindings.get("toolchain_id"),
                field_name="toolchain_id",
            )
            kernel_id = _text(
                request.payload.get("kernel_id"),
                field_name="kernel_id",
            )
            from ..proof.kernel_verification import (
                KernelVerificationBindings,
                KernelVerificationResult,
            )

            bindings = KernelVerificationBindings(
                obligation_id=obligation.obligation_id,
                request_id=bundle.request_id,
                candidate_id=candidate.candidate_id,
                kernel_id=kernel_id,
                toolchain_id=toolchain_id,
                expected_statement=obligation.statement,
                expected_statement_digest=str(
                    request.payload.get("expected_statement_digest") or ""
                ),
                expected_checked_source_digest=str(
                    request.payload.get("expected_checked_source_digest") or ""
                ),
                expected_native_source=native_source,
            )
            result = self.kernel_verifier.reconstruct_and_verify(
                request=hammer_request,
                candidate=candidate,
                goal_snapshot=snapshot,
                native_source=native_source,
                bindings=bindings,
                environment_lock=environment_lock,
                timeout=max(0.001, effective.timeout_ms / 1000.0),
                provider_status=HammerAdapterStatus.CANDIDATE.value,
            )
            if not isinstance(result, KernelVerificationResult):
                raise ValueError(
                    "kernel verifier returned an invalid result record"
                )
            if (
                result.obligation_id != obligation.obligation_id
                or result.request_id != bundle.request_id
                or result.candidate_id != candidate.candidate_id
                or result.kernel_id != kernel_id
                or result.toolchain_id != toolchain_id
            ):
                raise ValueError(
                    "kernel verification result is not bound to the request"
                )
            return {
                "schema_version": HAMMER_ADAPTER_SCHEMA_VERSION,
                "status": result.status.value,
                "kernel_verification": result.to_dict(),
                "provenance": {
                    **dict(bundle.provenance),
                    "candidate_id": candidate.candidate_id,
                    "kernel_id": kernel_id,
                    "toolchain_id": toolchain_id,
                    "kernel_receipt_id": result.kernel_receipt_id,
                },
                "authoritative_verdict": result.verdict.value,
                "authoritative_assurance": result.assurance.value,
                "kernel_checked": (
                    result.assurance.value == "kernel_verified"
                ),
                "proof_success": result.accepted,
            }
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            return ProviderResponse.failure(
                request,
                ProviderFailureCode.TIMED_OUT,
                f"kernel reconstruction timed out: {exc}",
                retryable=True,
                details={
                    "status": HammerAdapterStatus.TIMED_OUT.value,
                    "reason_code": "kernel_reconstruction_timed_out",
                    "proof_success": False,
                    "authoritative_assurance": "unverified",
                    "provenance": (
                        dict(bundle.provenance) if bundle is not None else {}
                    ),
                },
                provider_id=self.provider_id,
                provider_version=self.provider_version,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                f"invalid independent reconstruction request or result: {exc}",
            ) from exc

    def verify(
        self, request: ProviderRequest
    ) -> Mapping[str, Any] | ProviderResponse:
        return self.reconstruct(request)

    def attest(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse.failure(
            request,
            ProviderFailureCode.UNSUPPORTED,
            "the Hammer adapter does not issue attestations",
            details={
                "status": HammerAdapterStatus.UNSUPPORTED.value,
                "reason_code": "attestation_not_supported",
                "proof_success": False,
            },
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )


_REGISTRY_LOGIC_OPERATIONS: Final = (
    AnalysisOperation.PREMISE_SELECTION,
    AnalysisOperation.CONTRADICTION_SEARCH,
    AnalysisOperation.LOGIC_TRANSLATION,
    AnalysisOperation.PROOF_CANDIDATE_ANALYSIS,
    AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS,
)
_REGISTRY_LOGIC_CAPABILITIES: Final = (
    "contradiction_search",
    "counterexample_candidate_analysis",
    "logic_family_routing",
    "logic_translation",
    "premise_selection",
    "proof_candidate_analysis",
)
_REGISTRY_LOGIC_MAX_BATCH_SIZE: Final = 16
_REGISTRY_LOGIC_TOKEN_RE: Final = re.compile(r"[A-Za-z0-9_:.@/+~-]+")
_REGISTRY_NEGATION_TOKENS: Final = frozenset(
    {"contradiction", "contradicts", "false", "not", "never", "no", "counterexample"}
)
_OPTIONAL_LOGIC_METHODS: Final = {
    AnalysisOperation.PREMISE_SELECTION: (
        "select_premises",
        "premise_selection",
    ),
    AnalysisOperation.CONTRADICTION_SEARCH: (
        "search_contradictions",
        "find_contradictions",
        "contradiction_search",
    ),
    AnalysisOperation.LOGIC_TRANSLATION: (
        "translate_logic",
        "translate_legal_logic",
        "logic_translation",
        "analyze_legal_logic",
    ),
    AnalysisOperation.PROOF_CANDIDATE_ANALYSIS: (
        "analyze_proof_candidates",
        "select_proof_candidates",
        "proof_candidate_analysis",
    ),
    AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS: (
        "analyze_counterexample_candidates",
        "find_counterexamples",
        "counterexample_candidate_analysis",
    ),
}
_OPTIONAL_LOGIC_MODULES: Final = {
    AnalysisOperation.PREMISE_SELECTION: (
        "ipfs_datasets_py.logic.hammers",
    ),
    AnalysisOperation.CONTRADICTION_SEARCH: (
        "ipfs_datasets_py.logic.integration.domain.document_consistency_checker",
        "ipfs_datasets_py.logic.TDFOL",
    ),
    AnalysisOperation.LOGIC_TRANSLATION: (
        "ipfs_datasets_py.logic.integration.logic_translation_core",
        "ipfs_datasets_py.logic.deontic",
    ),
    AnalysisOperation.PROOF_CANDIDATE_ANALYSIS: (
        "ipfs_datasets_py.logic.hammers",
    ),
    AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS: (
        "ipfs_datasets_py.logic.hammers",
        "ipfs_datasets_py.logic.TDFOL",
    ),
}


class RegistryLogicProviderUnavailable(RuntimeError):
    """The optional reasoning surface cannot satisfy a registry request."""


def normalize_registry_logic_family(value: Any) -> LogicFamily:
    """Normalize a registry family without conflating semantic formalisms."""

    return normalize_logic_family(value)


def to_canonical_registry_logic_family(value: Any) -> str:
    """Project a Hammer/registry logic family onto the datasets family_id space."""

    from ..canonical_logic_adapter import map_analysis_family_to_canonical

    return map_analysis_family_to_canonical(normalize_registry_logic_family(value))


def _registry_logic_capability_revision(provider_kind: AnalysisProviderKind) -> str:
    return _digest(
        {
            "adapter": "agent-supervisor-registry-logic@1",
            "provider_kind": provider_kind.value,
            "operations": [item.value for item in _REGISTRY_LOGIC_OPERATIONS],
            "capabilities": list(_REGISTRY_LOGIC_CAPABILITIES),
            "logic_families": list(SUPPORTED_LOGIC_FAMILIES),
            "transport_protocol": ANALYSIS_TRANSPORT_PROTOCOL_VERSION,
            "request_schema": ANALYSIS_TRANSPORT_REQUEST_SCHEMA,
            "result_schema": ANALYSIS_TRANSPORT_RESULT_SCHEMA,
        },
        prefix="analysis-logic-capability",
    )


def registry_logic_producer_declarations() -> tuple[AnalysisProducer, AnalysisProducer]:
    """Declare local and optional logic producers without importing datasets code."""

    common = {
        "operations": _REGISTRY_LOGIC_OPERATIONS,
        "provider_version": IPFS_DATASETS_LOGIC_PROVIDER_VERSION,
        "capabilities": _REGISTRY_LOGIC_CAPABILITIES,
        "logic_families": tuple(LogicFamily),
        "max_batch_size": _REGISTRY_LOGIC_MAX_BATCH_SIZE,
        "supports_cancellation": True,
        "supports_progress": False,
        "supports_batching": True,
    }
    local = AnalysisProducer(
        producer_id=LOCAL_ANALYSIS_PRODUCER_ID,
        provider_kind=AnalysisProviderKind.LOCAL,
        capability_revision=_registry_logic_capability_revision(
            AnalysisProviderKind.LOCAL
        ),
        max_concurrency=4,
        **common,
    )
    optional = AnalysisProducer(
        producer_id=IPFS_DATASETS_ANALYSIS_PRODUCER_ID,
        provider_kind=AnalysisProviderKind.IPFS_DATASETS,
        capability_revision=_registry_logic_capability_revision(
            AnalysisProviderKind.IPFS_DATASETS
        ),
        max_concurrency=2,
        **common,
    )
    return local, optional


def _registry_cancelled(token: Any) -> bool:
    if token is None:
        return False
    for name in ("cancelled", "is_cancelled", "is_set"):
        value = getattr(token, name, None)
        if callable(value):
            try:
                return bool(value())
            except TypeError:
                continue
        if value is not None:
            return bool(value)
    return False


def _registry_logic_family(request: AnalysisRequest) -> LogicFamily:
    raw = request.metadata.get("logic_family")
    if not raw:
        raise ValueError("registry logic requests require logic_family metadata")
    return normalize_registry_logic_family(raw)


def _registry_reference_source_id(
    reference: Mapping[str, Any], request: AnalysisRequest
) -> str:
    for name in (
        "reference_id",
        "artifact_content_id",
        "artifact_id",
        "evidence_id",
        "record_id",
        "cid",
        "digest",
        "sha256",
        "uri",
        "path",
    ):
        value = str(reference.get(name) or "").strip()
        if value:
            return value
    return request.request_id


def _registry_logic_tokens(value: Any) -> frozenset[str]:
    return frozenset(
        token.casefold()
        for token in _REGISTRY_LOGIC_TOKEN_RE.findall(str(value))
        if token
    )


def _registry_candidate_score(
    operation: AnalysisOperation,
    question_tokens: frozenset[str],
    reference: Mapping[str, Any],
) -> int:
    reference_tokens: set[str] = set()
    for name in ("summary", "symbol", "path", "kind"):
        reference_tokens.update(_registry_logic_tokens(reference.get(name, "")))
    overlap = len(question_tokens.intersection(reference_tokens))
    denominator = max(1, len(question_tokens))
    overlap_score = min(250_000, (overlap * 250_000) // denominator)
    base = {
        AnalysisOperation.PREMISE_SELECTION: 650_000,
        AnalysisOperation.CONTRADICTION_SEARCH: 500_000,
        AnalysisOperation.LOGIC_TRANSLATION: 600_000,
        AnalysisOperation.PROOF_CANDIDATE_ANALYSIS: 550_000,
        AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS: 500_000,
    }[operation]
    if (
        operation
        in {
            AnalysisOperation.CONTRADICTION_SEARCH,
            AnalysisOperation.COUNTEREXAMPLE_CANDIDATE_ANALYSIS,
        }
        and question_tokens.intersection(_REGISTRY_NEGATION_TOKENS)
    ):
        base += 100_000
    return min(1_000_000, base + overlap_score)


def _registry_logic_evidence(
    request: AnalysisRequest,
    operation: AnalysisOperation,
    family: LogicFamily,
    producer_id: str,
) -> tuple[Mapping[str, Any], ...]:
    question_tokens = _registry_logic_tokens(request.question)
    source_references: tuple[Mapping[str, Any], ...] = request.artifact_references
    if not source_references:
        return ()
    candidates: list[Mapping[str, Any]] = []
    for reference in source_references:
        source_id = _registry_reference_source_id(reference, request)
        semantic_identity = {
            "operation": operation.value,
            "logic_family": family.value,
            "request_id": request.request_id,
            "source_id": source_id,
        }
        raw: dict[str, Any] = {
            "reference_id": _digest(
                semantic_identity, prefix="analysis-logic-candidate"
            ),
            "artifact_id": str(reference.get("artifact_id") or source_id),
            "kind": f"{operation.value}:{family.value}:candidate",
            "score_millionths": _registry_candidate_score(
                operation, question_tokens, reference
            ),
            "summary": (
                f"non-authoritative {family.value} "
                f"{operation.value.replace('_', ' ')} candidate"
            ),
        }
        for name in (
            "artifact_content_id",
            "cid",
            "digest",
            "path",
            "sha256",
            "symbol",
            "tree_id",
            "uri",
        ):
            if reference.get(name):
                raw[name] = reference[name]
        if not raw.get("tree_id") and request.metadata.get("tree_id"):
            raw["tree_id"] = request.metadata["tree_id"]
        candidates.append(
            normalize_analysis_reference(
                raw,
                default_kind=f"{operation.value}:{family.value}:candidate",
                producer_id=producer_id,
            )
        )
    candidates.sort(
        key=lambda item: (
            -int(item.get("score_millionths", 0)),
            str(item.get("reference_id", "")),
        )
    )
    return tuple(candidates[:64])


def _registry_logic_provenance(
    request: AnalysisRequest,
    operation: AnalysisOperation,
    family: LogicFamily,
    declaration: AnalysisProducer,
) -> tuple[Mapping[str, Any], ...]:
    bindings = (
        ("repository_id", request.metadata.get("repository_id")),
        ("tree_id", request.metadata.get("tree_id")),
        ("objective_revision", request.metadata.get("objective_revision")),
        ("policy_id", request.metadata.get("policy_id")),
        ("operation_spec", request.metadata.get("operation_spec_id")),
    )
    result: list[Mapping[str, Any]] = []
    for kind, value in bindings:
        normalized_value = str(value or "").strip()
        if not normalized_value:
            continue
        raw: dict[str, Any] = {
            "reference_id": _digest(
                {
                    "request_id": request.request_id,
                    "kind": kind,
                    "value": normalized_value,
                    "operation": operation.value,
                    "logic_family": family.value,
                },
                prefix="analysis-logic-provenance",
            ),
            "kind": kind,
            "record_id": normalized_value,
            "revision": declaration.capability_revision,
            "summary": f"{kind} binding for {family.value} analysis",
        }
        if request.metadata.get("tree_id"):
            raw["tree_id"] = request.metadata["tree_id"]
        result.append(
            normalize_analysis_reference(
                raw,
                default_kind=kind,
                producer_id=declaration.producer_id,
            )
        )
    return tuple(result)


def _registry_negotiated_fields(
    declaration: AnalysisProducer,
    operation: AnalysisOperation,
    negotiated_capability: Any,
) -> dict[str, Any]:
    capability = declaration.capability
    if negotiated_capability is None:
        return {
            "schema": ANALYSIS_TRANSPORT_RESULT_SCHEMA,
            "protocol_version": ANALYSIS_TRANSPORT_PROTOCOL_VERSION,
            "capability_id": capability.capability_id,
            "capability_revision": capability.capability_revision,
        }
    if str(getattr(negotiated_capability, "operation", "")) != operation.value:
        raise ValueError("negotiated operation does not match request")
    if not str(getattr(negotiated_capability, "request_schema", "")).strip():
        raise ValueError("negotiated request schema is missing")
    if not str(getattr(negotiated_capability, "result_schema", "")).strip():
        raise ValueError("negotiated result schema is missing")
    return {
        "schema": str(negotiated_capability.result_schema),
        "protocol_version": int(negotiated_capability.protocol_version),
        "capability_id": str(negotiated_capability.capability_id),
        "capability_revision": str(
            negotiated_capability.capability_revision
        ),
    }


def _registry_transport_result(
    request: AnalysisRequest,
    operation: AnalysisOperation,
    declaration: AnalysisProducer,
    evidence: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    *,
    negotiated_capability: Any = None,
    verdict: str = "candidate",
    cost: Mapping[str, int] | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    negotiated = _registry_negotiated_fields(
        declaration, operation, negotiated_capability
    )
    return {
        **negotiated,
        "request_id": request.request_id,
        "operation": operation.value,
        "evidence_references": [dict(item) for item in evidence],
        "provenance_references": [dict(item) for item in provenance],
        "cost": dict(cost or {}),
        "verdict": str(verdict or "inconclusive"),
        "truncated": bool(truncated),
        "non_authoritative": True,
        "completion_authority": False,
        "safe_for_completion_reasoning": False,
    }


def _registry_alias_reference_ids(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if not any(
        result.get(name)
        for name in (
            "reference_id",
            "artifact_content_id",
            "artifact_id",
            "evidence_id",
            "record_id",
            "cid",
            "digest",
            "sha256",
            "uri",
            "path",
        )
    ):
        for name in (
            "candidate_id",
            "premise_id",
            "contradiction_id",
            "counterexample_id",
            "translation_id",
            "receipt_id",
            "provenance_id",
        ):
            if result.get(name):
                result["reference_id"] = result[name]
                break
    return result


def _registry_normalize_optional_references(
    values: Any,
    *,
    default_kind: str,
    producer_id: str,
    expected_tree_id: str,
) -> tuple[Mapping[str, Any], ...]:
    if values is None:
        source: Sequence[Any] = ()
    elif isinstance(values, Mapping):
        source = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise ValueError("optional logic references must be an object or sequence")
    unique: dict[str, Mapping[str, Any]] = {}
    for value in source:
        converter = getattr(value, "to_dict", None)
        if not isinstance(value, Mapping) and callable(converter):
            value = converter()
        if not isinstance(value, Mapping):
            raise ValueError("optional logic references must contain objects")
        reference_tree_id = str(value.get("tree_id") or "").strip()
        if reference_tree_id and reference_tree_id != expected_tree_id:
            raise ValueError(
                "optional logic reference tree_id does not match request tree_id"
            )
        aliased = _registry_alias_reference_ids(value)
        # Derive a semantic identity before attaching provider provenance so an
        # equivalent local and remote reference receives the same reference ID.
        semantic = normalized_reference_payload(
            aliased, default_kind=default_kind
        )
        normalized = normalize_analysis_reference(
            semantic,
            default_kind=default_kind,
            producer_id=producer_id,
        )
        unique[str(normalized["reference_id"])] = normalized
    return tuple(unique[key] for key in sorted(unique))


def _registry_optional_result_parts(
    raw: Any,
    *,
    operation: AnalysisOperation,
    producer_id: str,
    expected_tree_id: str,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    str,
    Mapping[str, int],
    bool,
]:
    converter = getattr(raw, "to_dict", None)
    if not isinstance(raw, Mapping) and callable(converter):
        raw = converter()
    if isinstance(raw, Sequence) and not isinstance(
        raw, (str, bytes, bytearray, memoryview)
    ):
        raw = {"results": raw}
    if not isinstance(raw, Mapping):
        raise ValueError("optional logic provider returned a non-object result")
    for name in (
        "authoritative",
        "completion_authority",
        "proof_success",
        "safe_for_completion_reasoning",
        "repository_mutation",
        "validation_omission_selection",
        "candidate_promotion",
    ):
        if raw.get(name) is True:
            raise ValueError(f"optional logic result claims forbidden {name}")
    status = str(getattr(raw.get("status", ""), "value", raw.get("status", ""))).lower()
    if status and status not in {
        "candidate",
        "completed",
        "counterexample",
        "ok",
        "success",
        "translated",
        "unknown",
    }:
        raise RegistryLogicProviderUnavailable(
            f"optional logic provider status is {status}"
        )
    evidence_raw = raw.get("evidence_references")
    if evidence_raw is None:
        for name in (
            "results",
            "candidates",
            "premises",
            "contradictions",
            "translations",
            "proof_candidates",
            "counterexamples",
        ):
            if raw.get(name) is not None:
                evidence_raw = raw[name]
                break
    provenance_raw = raw.get(
        "provenance_references", raw.get("provenance", ())
    )
    evidence = _registry_normalize_optional_references(
        evidence_raw or (),
        default_kind=f"{operation.value}:candidate",
        producer_id=producer_id,
        expected_tree_id=expected_tree_id,
    )
    provenance = _registry_normalize_optional_references(
        provenance_raw or (),
        default_kind="logic_provenance",
        producer_id=producer_id,
        expected_tree_id=expected_tree_id,
    )
    cost_raw = raw.get("cost", raw.get("resource_use", {}))
    if not isinstance(cost_raw, Mapping):
        raise ValueError("optional logic result cost must be an object")
    cost: dict[str, int] = {}
    for name, value in cost_raw.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                "optional logic cost must contain non-negative integer counters"
            )
        cost[name] = value
    return (
        evidence,
        provenance,
        str(raw.get("verdict") or status or "candidate"),
        cost,
        bool(raw.get("truncated", False)),
    )


class _RegistryLogicProducer:
    """Transport adapter for compact local or optional logic analysis."""

    def __init__(
        self,
        declaration: AnalysisProducer,
        *,
        importer: Callable[[str], Any] | None = None,
        backend: Any = None,
    ) -> None:
        self.declaration = declaration
        self._importer = importer or importlib.import_module
        self._backend = backend

    def capabilities(self) -> Any:
        return self.declaration.capability

    capability = capabilities

    def supports(self, operation: Any) -> bool:
        try:
            normalized = normalize_analysis_operation(operation)
        except Exception:
            return False
        return normalized in self.declaration.operations

    def _normalize_request(
        self, request: AnalysisRequest | Mapping[str, Any]
    ) -> tuple[AnalysisRequest, AnalysisOperation, LogicFamily]:
        normalized = AnalysisRequest.from_value(request)
        operation = normalize_analysis_operation(normalized.operation)
        if operation not in self.declaration.operations:
            raise ValueError(
                f"logic producer does not support {operation.value}"
            )
        for name in (
            "repository_id",
            "tree_id",
            "objective_revision",
            "policy_id",
        ):
            if not str(normalized.metadata.get(name) or "").strip():
                raise ValueError(
                    f"registry logic request requires {name} provenance"
                )
        tree_id = normalized.metadata["tree_id"]
        if any(
            reference.get("tree_id")
            and reference.get("tree_id") != tree_id
            for reference in normalized.artifact_references
        ):
            raise ValueError(
                "registry logic artifact tree_id does not match request tree_id"
            )
        return normalized, operation, _registry_logic_family(normalized)

    def _local_analyze(
        self,
        request: AnalysisRequest,
        operation: AnalysisOperation,
        family: LogicFamily,
        *,
        negotiated_capability: Any,
    ) -> dict[str, Any]:
        evidence = _registry_logic_evidence(
            request, operation, family, self.declaration.producer_id
        )
        provenance = _registry_logic_provenance(
            request, operation, family, self.declaration
        )
        return _registry_transport_result(
            request,
            operation,
            self.declaration,
            evidence,
            provenance,
            negotiated_capability=negotiated_capability,
            verdict="candidate" if evidence else "inconclusive",
            cost={
                "items_examined": len(request.artifact_references),
                "provider_calls": 1,
            },
        )

    def _optional_backend(self, operation: AnalysisOperation) -> tuple[Any, Any]:
        if self._backend is not None:
            candidates = (self._backend,)
        else:
            candidates_list: list[Any] = []
            try:
                candidates_list.append(self._importer("ipfs_datasets_py"))
            except (ImportError, ModuleNotFoundError):
                pass
            for module_name in _OPTIONAL_LOGIC_MODULES[operation]:
                try:
                    candidates_list.append(self._importer(module_name))
                except (ImportError, ModuleNotFoundError):
                    continue
            candidates = tuple(candidates_list)
        for candidate in candidates:
            for owner in (
                candidate,
                getattr(candidate, "logic", None),
                getattr(candidate, "analysis", None),
                getattr(candidate, "reasoning", None),
            ):
                if owner is None:
                    continue
                for name in _OPTIONAL_LOGIC_METHODS[operation]:
                    method = getattr(owner, name, None)
                    if callable(method):
                        return owner, method
                generic = getattr(owner, "analyze", None)
                if callable(generic):
                    return owner, generic
            if callable(candidate):
                return candidate, candidate
        raise RegistryLogicProviderUnavailable(
            f"ipfs_datasets_py has no {operation.value} reasoning function"
        )

    def _optional_payload(
        self,
        request: AnalysisRequest,
        operation: AnalysisOperation,
        family: LogicFamily,
    ) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_TRANSPORT_REQUEST_SCHEMA,
            "protocol_version": ANALYSIS_TRANSPORT_PROTOCOL_VERSION,
            "request_id": request.request_id,
            "operation": operation.value,
            "question": request.question,
            "artifact_references": [
                dict(reference) for reference in request.artifact_references
            ],
            "metadata": {
                key: request.metadata[key]
                for key in (
                    "repository_id",
                    "tree_id",
                    "objective_revision",
                    "policy_id",
                    "operation_spec_id",
                )
                if request.metadata.get(key)
            },
            "logic_family": family.value,
            "non_authoritative": True,
        }

    def _finish_optional(
        self,
        raw: Any,
        request: AnalysisRequest,
        operation: AnalysisOperation,
        family: LogicFamily,
        negotiated_capability: Any,
    ) -> dict[str, Any]:
        evidence, optional_provenance, verdict, cost, truncated = (
            _registry_optional_result_parts(
                raw,
                operation=operation,
                producer_id=self.declaration.producer_id,
                expected_tree_id=str(request.metadata["tree_id"]),
            )
        )
        provenance = optional_provenance + _registry_logic_provenance(
            request, operation, family, self.declaration
        )
        return _registry_transport_result(
            request,
            operation,
            self.declaration,
            evidence,
            provenance,
            negotiated_capability=negotiated_capability,
            verdict=verdict,
            cost=cost,
            truncated=truncated,
        )

    def analyze(
        self,
        request: AnalysisRequest | Mapping[str, Any],
        *,
        cancellation_token: Any = None,
        negotiated_capability: Any = None,
        **_kwargs: Any,
    ) -> Any:
        normalized, operation, family = self._normalize_request(request)
        if _registry_cancelled(cancellation_token):
            raise RegistryLogicProviderUnavailable(
                "registry logic analysis was cancelled"
            )
        if self.declaration.provider_kind is AnalysisProviderKind.LOCAL:
            return self._local_analyze(
                normalized,
                operation,
                family,
                negotiated_capability=negotiated_capability,
            )
        _owner, method = self._optional_backend(operation)
        raw = method(self._optional_payload(normalized, operation, family))
        if inspect.isawaitable(raw):
            async def finish() -> dict[str, Any]:
                resolved = await raw
                return self._finish_optional(
                    resolved,
                    normalized,
                    operation,
                    family,
                    negotiated_capability,
                )

            return finish()
        return self._finish_optional(
            raw,
            normalized,
            operation,
            family,
            negotiated_capability,
        )

    def analyze_batch(
        self,
        requests: Sequence[AnalysisRequest | Mapping[str, Any]],
        *,
        cancellation_token: Any = None,
        negotiated_capability: Any = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(requests, (str, bytes, bytearray)) or not isinstance(
            requests, Sequence
        ):
            raise ValueError("registry logic batch must be a sequence")
        if not requests:
            raise ValueError("registry logic batch must not be empty")
        if len(requests) > self.declaration.max_batch_size:
            raise ValueError("registry logic batch exceeds producer bound")
        results = tuple(
            self.analyze(
                request,
                cancellation_token=cancellation_token,
                negotiated_capability=negotiated_capability,
                **kwargs,
            )
            for request in requests
        )
        if any(inspect.isawaitable(item) for item in results):
            async def finish_batch() -> tuple[Any, ...]:
                output: list[Any] = []
                for item in results:
                    output.append(await item if inspect.isawaitable(item) else item)
                return tuple(output)

            return finish_batch()
        return results


def create_local_registry_logic_producer() -> _RegistryLogicProducer:
    """Create the deterministic compact-metadata local logic producer."""

    local, _optional = registry_logic_producer_declarations()
    return _RegistryLogicProducer(local)


def create_optional_registry_logic_producer(
    *,
    importer: Callable[[str], Any] | None = None,
    backend: Any = None,
) -> _RegistryLogicProducer:
    """Create a lazy optional datasets producer with transport fallback on failure."""

    _local, optional = registry_logic_producer_declarations()
    return _RegistryLogicProducer(
        optional,
        importer=importer,
        backend=backend,
    )


# Conventional class aliases used by entry-point declarations.
IPFSDatasetsLogicProvider = IpfsDatasetsLogicProvider
HammerProofProvider = IpfsDatasetsLogicProvider


def create_ipfs_datasets_logic_provider(
    policy: HammerSupervisorPolicy | None = None,
    *,
    portfolio_runner: PortfolioRunner | None = None,
    verification_cache: FormalVerificationCache | None = None,
    proof_cache: FormalVerificationCache | None = None,
    cache: FormalVerificationCache | None = None,
    kernel_verifier: Any = None,
) -> IpfsDatasetsLogicProvider:
    """Entry-point-friendly provider factory without importing Hammer."""

    return IpfsDatasetsLogicProvider(
        policy,
        portfolio_runner=portfolio_runner,
        verification_cache=verification_cache,
        proof_cache=proof_cache,
        cache=cache,
        kernel_verifier=kernel_verifier,
    )


__all__ = [
    "HAMMER_ADAPTER_SCHEMA_VERSION",
    "HAMMER_PROVENANCE_SCHEMA_VERSION",
    "HAMMER_TRANSLATOR_ID",
    "IPFS_DATASETS_LOGIC_PROVIDER_ID",
    "IPFS_DATASETS_LOGIC_PROVIDER_VERSION",
    "KNOWN_HAMMER_SOLVERS",
    "SUPPORTED_LOGIC_FAMILIES",
    "SUPPORTED_TRANSLATION_FAMILIES",
    "LogicFamily",
    "HammerAdapterStatus",
    "HammerSupervisorPolicy",
    "HammerProviderPolicy",
    "IpfsDatasetsProviderPolicy",
    "EffectiveHammerPolicy",
    "HammerRequestBundle",
    "HammerPortfolioInvocation",
    "IpfsDatasetsLogicProviderConfig",
    "IPFSDatasetsLogicProviderConfig",
    "IpfsDatasetsLogicProvider",
    "IPFSDatasetsLogicProvider",
    "HammerProofProvider",
    "translate_obligation_to_hammer_request",
    "adapt_hammer_result",
    "create_ipfs_datasets_logic_provider",
    "RegistryLogicProviderUnavailable",
    "normalize_registry_logic_family",
    "to_canonical_registry_logic_family",
    "registry_logic_producer_declarations",
    "create_local_registry_logic_producer",
    "create_optional_registry_logic_producer",
]
