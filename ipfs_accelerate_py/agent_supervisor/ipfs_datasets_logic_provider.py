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
  containing the exact configured fallback checks, never a guessed proof.

ATP/SMT candidates remain untrusted.  This provider never maps a portfolio
verdict to kernel-verified assurance; independent reconstruction is owned by
the kernel-verification integration.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final

from .formal_verification_capabilities import (
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
)
from .formal_verification_contracts import (
    CodeProofObligation,
    ResourceBudget,
    canonical_json,
)
from .formal_verification_provider import (
    PROOF_PROVIDER_PROTOCOL_VERSION,
    ProofProviderError,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
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
    try:
        return importlib.import_module("ipfs_datasets_py.logic.hammers")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProofProviderError(
            ProviderFailureCode.UNAVAILABLE,
            "ipfs_datasets_py Hammer portfolio is unavailable",
            details={"module": "ipfs_datasets_py.logic.hammers"},
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
            "selection_method": HAMMER_TRANSLATOR_ID,
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
    raw = str(getattr(value, "value", value)).lower()
    return {
        "candidate": HammerAdapterStatus.CANDIDATE,
        "counterexample": HammerAdapterStatus.COUNTEREXAMPLE,
        "timeout": HammerAdapterStatus.TIMED_OUT,
        "unavailable": HammerAdapterStatus.UNAVAILABLE,
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
    ) -> None:
        self.policy = policy or HammerSupervisorPolicy()
        if not isinstance(self.policy, HammerSupervisorPolicy):
            raise ValueError("policy must be a HammerSupervisorPolicy")
        if portfolio_runner is not None and not callable(portfolio_runner):
            raise ValueError("portfolio_runner must be callable")
        self._portfolio_runner = portfolio_runner

    def capabilities(self) -> ProofProviderCapability:
        return ProofProviderCapability(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            protocol_versions=(self.protocol_version,),
            operations=(
                ProofProviderOperation.CAPABILITY,
                ProofProviderOperation.TRANSLATE,
                ProofProviderOperation.PROVE,
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
        premise_dicts, upstream_receipts, premise_provenance = _premise_payloads(
            obligation,
            payload,
            corpus_revision=corpus_revision,
            itp=itp,
            max_premises=policy.max_premises,
        )
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
        request_identity = {
            "obligation_id": obligation.obligation_id,
            "repository_tree_id": obligation.repository_tree_id,
            "translation_family": family,
            "premises": list(premise_dicts),
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
            "premises": premise_provenance,
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
            raw = obligation.metadata.get("hammer_translations")
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
            raw_result = runner(invocation)
            projected = adapt_hammer_result(raw_result, bundle)
            projected["environment_lock"] = dict(bundle.environment_lock)
            projected["portfolio_policy"] = dict(bundle.portfolio_policy)
            projected["premises"] = [
                dict(premise) for premise in bundle.premises
            ]
            return projected
        except ProofProviderError as exc:
            if exc.code is ProviderFailureCode.UNSUPPORTED:
                return self._unsupported(
                    request,
                    exc,
                    obligation=obligation,
                    policy=effective,
                )
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise ProofProviderError(
                ProviderFailureCode.MALFORMED_REQUEST,
                f"invalid Hammer portfolio request or result: {exc}",
            ) from exc

    def reconstruct(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse.failure(
            request,
            ProviderFailureCode.UNSUPPORTED,
            "independent kernel reconstruction is not implemented by this adapter",
            details={
                "reason_code": "independent_kernel_provider_required",
                "proof_success": False,
            },
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )

    verify = reconstruct
    attest = reconstruct


# Conventional class aliases used by entry-point declarations.
IPFSDatasetsLogicProvider = IpfsDatasetsLogicProvider
HammerProofProvider = IpfsDatasetsLogicProvider


def create_ipfs_datasets_logic_provider(
    policy: HammerSupervisorPolicy | None = None,
    *,
    portfolio_runner: PortfolioRunner | None = None,
) -> IpfsDatasetsLogicProvider:
    """Entry-point-friendly provider factory without importing Hammer."""

    return IpfsDatasetsLogicProvider(
        policy,
        portfolio_runner=portfolio_runner,
    )


__all__ = [
    "HAMMER_ADAPTER_SCHEMA_VERSION",
    "HAMMER_PROVENANCE_SCHEMA_VERSION",
    "HAMMER_TRANSLATOR_ID",
    "IPFS_DATASETS_LOGIC_PROVIDER_ID",
    "IPFS_DATASETS_LOGIC_PROVIDER_VERSION",
    "KNOWN_HAMMER_SOLVERS",
    "SUPPORTED_TRANSLATION_FAMILIES",
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
]
