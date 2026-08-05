"""Privacy-safe session reporting for proof-backed pytest reuse.

The reporting contract intentionally contains aggregates only.  Node ids,
paths, parameter values, receipt bodies, exception text, and test output are
never accepted into a metrics snapshot, including xdist worker snapshots.

PTR-149 also publishes :class:`ProofReuseRuntimeActivationReport`, a live
typed composition and capability snapshot.  That report derives availability
only from already-composed services and bounded non-mutating probes; it never
imports or installs packages merely to claim readiness, and it always
separates native Groth16 installation from test-certificate authority.
"""

from __future__ import annotations

import math
import os
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

PROOF_REUSE_METRICS_INTERFACE: Final = "ProofReuseMetrics@1"
PROOF_REUSE_METRICS_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse-metrics@1"
)
PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_INTERFACE: Final = (
    "ProofReuseRuntimeActivationReport@1"
)
PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse-runtime-activation-report@1"
)
MAX_REASON_CODES: Final = 128
MAX_REASON_LENGTH: Final = 96
MAX_COUNTER_VALUE: Final = (1 << 63) - 1


class ProofReuseOutcome(str, Enum):
    """Closed outcome vocabulary used by local and xdist reporters."""

    PREDICTED = "predicted"
    VERIFIED = "verified"
    SKIPPED = "skipped"
    EXECUTED = "executed"
    DEFERRED = "deferred"
    DEGRADED = "degraded"


_OUTCOMES: Final = tuple(outcome.value for outcome in ProofReuseOutcome)
_SAFE_REASON_CHARS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyz0123456789_:-."
)
_SAFE_REASON_CODES: Final = frozenset(
    {
        "absence_fail_open_to_run",
        "cache_unavailable",
        "candidate_integrity_failed",
        "candidate_missing",
        "certificate_deferred",
        "certificate_non_attested",
        "certificate_provider_unavailable",
        "cid_provider_unavailable",
        "circuit_unavailable",
        "coordination_unavailable",
        "deferred_issuer_unavailable",
        "eligibility_denied",
        "exception_fail_open_to_run",
        "execution_key_mismatch",
        "expired_or_revoked",
        "illegal_authority",
        "incomplete_trace",
        "internal_error_fail_open_to_run",
        "invalidation",
        "issuer_revoked",
        "key_unavailable",
        "lookup_decision_invalid",
        "lookup_hit",
        "malformed_artifact",
        "mode_off",
        "mode_shadow",
        "mode_write_only",
        "non_reusable",
        "over_budget",
        "plugin_unavailable",
        "policy_mismatch",
        "positive_v4_publication_pending_ptr155",
        "private_material",
        "proof_cache_hit",
        "proof_verified",
        "publication_failed",
        "publication_intent_disagrees",
        "publication_intent_invalid",
        "publication_over_budget",
        "real_execution",
        "receipt_mismatch",
        "reuse_disabled",
        "runtime_hook_failed",
        "runtime_registration_failed",
        "timeout",
        "trust_policy_rejected",
        "unknown",
        "unsupported",
        "verifier_unavailable",
        "worker_crash",
        "worker_output_missing",
        "worker_output_rejected",
    }
)


def _safe_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("metric counts must be integers")
    if value < 0 or value > MAX_COUNTER_VALUE:
        raise ValueError("metric count is out of bounds")
    return value


def _safe_reason(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return ""
    value = value.strip().lower()
    if not value or len(value) > MAX_REASON_LENGTH:
        return ""
    if any(character not in _SAFE_REASON_CHARS for character in value):
        return ""
    if value not in _SAFE_REASON_CODES:
        return ""
    return value


def _safe_packet_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        return ""
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        return ""
    return normalized


def _safe_latency(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("latency must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or result > 86_400_000:
        raise ValueError("latency is out of bounds")
    return result


@dataclass(frozen=True)
class ProofReuseMetricsSnapshot:
    """Immutable, JSON-safe aggregate suitable for worker transport."""

    counts: Mapping[str, int]
    reasons: Mapping[str, int]
    verify_latency_ms: float = 0.0
    execution_latency_ms: float = 0.0
    bytes_read: int = 0
    bytes_written: int = 0

    @property
    def interface(self) -> str:
        return PROOF_REUSE_METRICS_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_REUSE_METRICS_SCHEMA,
            "interface": PROOF_REUSE_METRICS_INTERFACE,
            "counts": {name: int(self.counts.get(name, 0)) for name in _OUTCOMES},
            "reasons": dict(sorted(self.reasons.items())),
            "verify_latency_ms": self.verify_latency_ms,
            "execution_latency_ms": self.execution_latency_ms,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofReuseMetricsSnapshot":
        if not isinstance(payload, Mapping):
            raise ValueError("metrics snapshot must be a mapping")
        allowed = {
            "schema",
            "interface",
            "counts",
            "reasons",
            "verify_latency_ms",
            "execution_latency_ms",
            "bytes_read",
            "bytes_written",
        }
        if set(payload) - allowed:
            raise ValueError("metrics snapshot contains private or unknown fields")
        if payload.get("schema") != PROOF_REUSE_METRICS_SCHEMA:
            raise ValueError("metrics snapshot schema mismatch")
        if payload.get("interface") != PROOF_REUSE_METRICS_INTERFACE:
            raise ValueError("metrics snapshot interface mismatch")
        raw_counts = payload.get("counts")
        raw_reasons = payload.get("reasons")
        if not isinstance(raw_counts, Mapping) or not isinstance(raw_reasons, Mapping):
            raise ValueError("metrics counters must be mappings")
        if set(raw_counts) - set(_OUTCOMES):
            raise ValueError("metrics snapshot contains an unknown outcome")
        counts = {name: _safe_count(raw_counts.get(name, 0)) for name in _OUTCOMES}
        if len(raw_reasons) > MAX_REASON_CODES:
            raise ValueError("metrics snapshot has too many reason codes")
        reasons: dict[str, int] = {}
        for raw_reason, raw_count in raw_reasons.items():
            reason = _safe_reason(raw_reason)
            if not reason or reason != raw_reason:
                raise ValueError("metrics snapshot has an unsafe reason code")
            reasons[reason] = _safe_count(raw_count)
        return cls(
            counts=counts,
            reasons=reasons,
            verify_latency_ms=_safe_latency(payload.get("verify_latency_ms", 0)),
            execution_latency_ms=_safe_latency(
                payload.get("execution_latency_ms", 0)
            ),
            bytes_read=_safe_count(payload.get("bytes_read", 0)),
            bytes_written=_safe_count(payload.get("bytes_written", 0)),
        )


class ProofReuseSessionMetrics:
    """Thread-safe aggregate with explicit proof-reuse outcome dimensions."""

    interface = PROOF_REUSE_METRICS_INTERFACE

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()
        self._verify_latency_ms = 0.0
        self._execution_latency_ms = 0.0
        self._bytes_read = 0
        self._bytes_written = 0
        self._merged_packets: set[str] = set()
        self._lock = threading.RLock()

    def record(
        self,
        outcome: ProofReuseOutcome | str,
        *,
        count: int = 1,
        reason_code: Any = "",
        latency_ms: int | float = 0,
        bytes_read: int = 0,
        bytes_written: int = 0,
    ) -> None:
        name = outcome.value if isinstance(outcome, ProofReuseOutcome) else str(outcome)
        if name not in _OUTCOMES:
            raise ValueError("unknown proof reuse outcome")
        increment = _safe_count(count)
        latency = _safe_latency(latency_ms)
        read_count = _safe_count(bytes_read)
        written_count = _safe_count(bytes_written)
        reason = _safe_reason(reason_code)
        with self._lock:
            self._counts[name] = min(
                MAX_COUNTER_VALUE, self._counts[name] + increment
            )
            if reason:
                if reason in self._reasons or len(self._reasons) < MAX_REASON_CODES:
                    self._reasons[reason] = min(
                        MAX_COUNTER_VALUE, self._reasons[reason] + increment
                    )
            if name == ProofReuseOutcome.VERIFIED.value:
                self._verify_latency_ms += latency
            elif name == ProofReuseOutcome.EXECUTED.value:
                self._execution_latency_ms += latency
            self._bytes_read = min(
                MAX_COUNTER_VALUE, self._bytes_read + read_count
            )
            self._bytes_written = min(
                MAX_COUNTER_VALUE, self._bytes_written + written_count
            )

    def predicted(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.PREDICTED, **kwargs)

    def verified(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.VERIFIED, **kwargs)

    def skipped(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.SKIPPED, **kwargs)

    def executed(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.EXECUTED, **kwargs)

    def deferred(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.DEFERRED, **kwargs)

    def degraded(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.DEGRADED, **kwargs)

    def count(self, outcome: ProofReuseOutcome | str) -> int:
        name = outcome.value if isinstance(outcome, ProofReuseOutcome) else str(outcome)
        with self._lock:
            return int(self._counts.get(name, 0))

    @property
    def counts(self) -> Mapping[str, int]:
        return self.snapshot().counts

    @property
    def reasons(self) -> Mapping[str, int]:
        return self.snapshot().reasons

    def snapshot(self) -> ProofReuseMetricsSnapshot:
        with self._lock:
            return ProofReuseMetricsSnapshot(
                counts={name: int(self._counts.get(name, 0)) for name in _OUTCOMES},
                reasons=dict(sorted(self._reasons.items())),
                verify_latency_ms=round(self._verify_latency_ms, 3),
                execution_latency_ms=round(self._execution_latency_ms, 3),
                bytes_read=self._bytes_read,
                bytes_written=self._bytes_written,
            )

    def merge(
        self,
        snapshot: ProofReuseMetricsSnapshot | Mapping[str, Any],
        *,
        packet_id: str = "",
    ) -> bool:
        """Merge one worker aggregate once; malformed/private data is rejected."""

        if isinstance(snapshot, Mapping):
            snapshot = ProofReuseMetricsSnapshot.from_dict(snapshot)
        if not isinstance(snapshot, ProofReuseMetricsSnapshot):
            raise TypeError("snapshot must be ProofReuseMetricsSnapshot")
        safe_packet_id = _safe_packet_id(packet_id)
        with self._lock:
            if safe_packet_id and safe_packet_id in self._merged_packets:
                return False
            for name in _OUTCOMES:
                self._counts[name] = min(
                    MAX_COUNTER_VALUE,
                    self._counts[name] + _safe_count(snapshot.counts.get(name, 0)),
                )
            for reason, count in snapshot.reasons.items():
                safe_reason = _safe_reason(reason)
                if not safe_reason or safe_reason != reason:
                    raise ValueError("unsafe reason code")
                if (
                    safe_reason in self._reasons
                    or len(self._reasons) < MAX_REASON_CODES
                ):
                    self._reasons[safe_reason] = min(
                        MAX_COUNTER_VALUE,
                        self._reasons[safe_reason] + _safe_count(count),
                    )
            self._verify_latency_ms += _safe_latency(snapshot.verify_latency_ms)
            self._execution_latency_ms += _safe_latency(
                snapshot.execution_latency_ms
            )
            self._bytes_read = min(
                MAX_COUNTER_VALUE,
                self._bytes_read + _safe_count(snapshot.bytes_read),
            )
            self._bytes_written = min(
                MAX_COUNTER_VALUE,
                self._bytes_written + _safe_count(snapshot.bytes_written),
            )
            if safe_packet_id:
                self._merged_packets.add(safe_packet_id)
        return True

    def summary_line(self) -> str:
        snapshot = self.snapshot()
        values = " ".join(
            f"{name}={snapshot.counts.get(name, 0)}" for name in _OUTCOMES
        )
        return f"proof reuse: {values}"


def _bounded_mapping(payload: Mapping[str, Any] | None, *, depth: int = 0) -> dict[str, Any]:
    """Copy a mapping with bounded depth/size for safe report embedding."""

    if not isinstance(payload, Mapping) or depth > 4:
        return {}
    out: dict[str, Any] = {}
    for index, (raw_key, value) in enumerate(payload.items()):
        if index >= 64:
            break
        key = str(raw_key)[:96]
        if isinstance(value, bool) or value is None or isinstance(value, int):
            out[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                out[key] = value
        elif isinstance(value, str):
            out[key] = value[:256]
        elif isinstance(value, Mapping):
            out[key] = _bounded_mapping(value, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            items: list[Any] = []
            for item in list(value)[:32]:
                if isinstance(item, Mapping):
                    items.append(_bounded_mapping(item, depth=depth + 1))
                elif isinstance(item, (bool, int)) or item is None:
                    items.append(item)
                elif isinstance(item, str):
                    items.append(item[:128])
                else:
                    items.append(type(item).__name__[:64])
            out[key] = items
        else:
            out[key] = type(value).__name__[:64]
    return out


@dataclass(frozen=True, slots=True)
class ProofReuseRuntimeActivationReport:
    """Live typed runtime-activation and capability report (PTR-149).

    Availability is derived only from composed service handles and bounded
    non-mutating probes.  Native Groth16 installation/readiness is always
    reported separately from test-certificate authority; the generic
    pre-PTR-144 knowledge-of-axioms backend can never satisfy the latter.
    """

    interface: str = PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_INTERFACE
    schema: str = PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_SCHEMA
    live: bool = True
    network_attempted: bool = False
    install_attempted: bool = False
    import_for_readiness: bool = False
    process_started: bool = False
    prove_attempted: bool = False
    composition: Mapping[str, Any] = field(default_factory=dict)
    native_groth16: Mapping[str, Any] = field(default_factory=dict)
    test_certificate_authority: Mapping[str, Any] = field(default_factory=dict)
    inventory: Mapping[str, Any] = field(default_factory=dict)
    activation_gap: Mapping[str, Any] = field(default_factory=dict)
    activation_blocker_codes: tuple[str, ...] = ()
    ordinary_default_composition_usable: bool = False
    ordinary_warm_skip_path_complete: bool = False
    native_groth16_installed: bool = False
    native_groth16_ready: bool = False
    test_certificate_authority_ready: bool = False
    knowledge_of_axioms_cannot_satisfy_test_certificate_authority: bool = True
    unmanifested_native_binary_cannot_satisfy_test_certificate_authority: bool = (
        True
    )
    activation_gap_present: bool = False
    source: str = "live"
    reason_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "live": True,
            "network_attempted": False,
            "install_attempted": False,
            "import_for_readiness": False,
            "process_started": bool(self.process_started),
            "prove_attempted": False,
            "composition": _bounded_mapping(self.composition),
            "native_groth16": _bounded_mapping(self.native_groth16),
            "test_certificate_authority": _bounded_mapping(
                self.test_certificate_authority
            ),
            "inventory": _bounded_mapping(self.inventory),
            "activation_gap": _bounded_mapping(self.activation_gap),
            "activation_blocker_codes": list(self.activation_blocker_codes),
            "ordinary_default_composition_usable": (
                self.ordinary_default_composition_usable
            ),
            "ordinary_warm_skip_path_complete": (
                self.ordinary_warm_skip_path_complete
            ),
            "native_groth16_installed": self.native_groth16_installed,
            "native_groth16_ready": self.native_groth16_ready,
            "test_certificate_authority_ready": (
                self.test_certificate_authority_ready
            ),
            "knowledge_of_axioms_cannot_satisfy_test_certificate_authority": True,
            "unmanifested_native_binary_cannot_satisfy_test_certificate_authority": True,
            "activation_gap_present": bool(self.activation_gap_present),
            "source": str(self.source)[:32],
            "reason_code": str(self.reason_code)[:96],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ProofReuseRuntimeActivationReport":
        if not isinstance(payload, Mapping):
            raise ValueError("activation report must be a mapping")
        if payload.get("schema") != PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_SCHEMA:
            raise ValueError("activation report schema mismatch")
        if (
            payload.get("interface")
            != PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_INTERFACE
        ):
            raise ValueError("activation report interface mismatch")
        blockers_raw = payload.get("activation_blocker_codes") or ()
        if not isinstance(blockers_raw, (list, tuple)):
            raise ValueError("activation_blocker_codes must be a sequence")
        blockers = tuple(str(item)[:96] for item in blockers_raw[:64])
        gap = _bounded_mapping(payload.get("activation_gap") or {})
        gap_present = bool(
            payload.get("activation_gap_present")
            if "activation_gap_present" in payload
            else gap.get("present")
        )
        return cls(
            composition=_bounded_mapping(payload.get("composition") or {}),
            native_groth16=_bounded_mapping(payload.get("native_groth16") or {}),
            test_certificate_authority=_bounded_mapping(
                payload.get("test_certificate_authority") or {}
            ),
            inventory=_bounded_mapping(payload.get("inventory") or {}),
            activation_gap=gap,
            activation_blocker_codes=blockers,
            ordinary_default_composition_usable=bool(
                payload.get("ordinary_default_composition_usable")
            ),
            ordinary_warm_skip_path_complete=bool(
                payload.get("ordinary_warm_skip_path_complete")
            ),
            native_groth16_installed=bool(payload.get("native_groth16_installed")),
            native_groth16_ready=bool(payload.get("native_groth16_ready")),
            test_certificate_authority_ready=bool(
                payload.get("test_certificate_authority_ready")
            ),
            activation_gap_present=gap_present,
            process_started=bool(payload.get("process_started")),
            source=str(payload.get("source") or "live")[:32],
            reason_code=str(payload.get("reason_code") or "")[:96],
        )


def proof_reuse_runtime_activation_report(
    *,
    services: Any = None,
    mode: Any = None,
    root_path: str | os.PathLike[str] | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    config: Any = None,
    environ: Mapping[str, str] | None = None,
    installer: Any = None,
    artifacts_root: str | os.PathLike[str] | None = None,
    binary_path: str | os.PathLike[str] | None = None,
    compose_if_missing: bool = True,
) -> ProofReuseRuntimeActivationReport:
    """Build a live runtime-activation report from typed services and probes.

    Never installs packages, never starts a prove/setup/network process, and
    never imports optional stacks merely to flip readiness booleans.  When
    ``services`` is omitted and ``compose_if_missing`` is true, default
    composition is assembled with the supplied installer (which must itself
    refuse installs when consent is denied).
    """

    env = environ if environ is not None else os.environ
    resolved_services = services
    source = "explicit"
    reason = ""

    if resolved_services is None and compose_if_missing:
        source = "composed_defaults"
        try:
            from .services import compose_default_proof_reuse_services

            resolved_services = compose_default_proof_reuse_services(
                mode=mode,
                root_path=root_path,
                cache_root=cache_root,
                config=config,
                installer=installer,
                environ=env,
            )
        except Exception as exc:
            reason = f"compose_failed:{type(exc).__name__}"[:96]
            resolved_services = None
    elif resolved_services is None:
        source = "missing"
        reason = "services_missing"

    try:
        from .services import live_runtime_activation_inventory
    except Exception as exc:
        return ProofReuseRuntimeActivationReport(
            source=source,
            reason_code=f"probe_import_failed:{type(exc).__name__}"[:96],
        )

    inventory = live_runtime_activation_inventory(
        resolved_services,
        installer=installer,
        environ=env,
        artifacts_root=artifacts_root,
        binary_path=binary_path,
    )
    composition = inventory.get("composition") or {}
    native = inventory.get("native_groth16") or {}
    certificate = inventory.get("test_certificate_authority") or {}
    activation_gap_raw = inventory.get("activation_gap") or {}
    blockers = tuple(
        str(item)[:96]
        for item in (inventory.get("activation_blocker_codes") or ())[:64]
    )

    # Hard invariant: knowledge-of-axioms can never satisfy certificate authority.
    cert_ready = bool(inventory.get("test_certificate_authority_ready"))
    if isinstance(certificate, Mapping) and certificate.get(
        "knowledge_of_axioms_circuit"
    ) and cert_ready:
        cert_ready = False
        certificate = dict(certificate)
        certificate["ready"] = False
        certificate["reason_code"] = (
            "knowledge_of_axioms_cannot_satisfy_test_certificate_authority"
        )
    # Unmanifested native binary alone can never satisfy certificate authority.
    if (
        cert_ready
        and isinstance(certificate, Mapping)
        and certificate.get("unmanifested_native_binary_rejected")
    ):
        cert_ready = False
        certificate = dict(certificate)
        certificate["ready"] = False
        certificate["reason_code"] = (
            "unmanifested_native_binary_cannot_satisfy_test_certificate_authority"
        )

    gap = (
        _bounded_mapping(activation_gap_raw)
        if isinstance(activation_gap_raw, Mapping)
        else {}
    )
    gap_present = bool(
        inventory.get("activation_gap_present")
        if "activation_gap_present" in inventory
        else gap.get("present")
    )
    if not cert_ready and not gap_present:
        # Truthful gap when authority is unready even if the inventory omitted
        # the packet (fail-closed default for missing reviewed keys/manifest).
        gap = {
            "present": True,
            "reason_code": str(
                (certificate.get("reason_code") if isinstance(certificate, Mapping) else "")
                or "reviewed_v4_keys_or_manifest_absent"
            )[:96],
            "warm_skip_authorized": False,
            "closeout_authorized": False,
            "tests_continue": True,
            "reviewed_v4_keys_or_manifest_required": True,
            "native_binary_alone_non_authoritative": True,
            "knowledge_of_axioms_cannot_satisfy": True,
        }
        gap_present = True
    if gap_present:
        gap = dict(gap)
        gap["present"] = True
        gap["warm_skip_authorized"] = False
        gap["closeout_authorized"] = False

    warm_complete = (
        bool(inventory.get("ordinary_warm_skip_path_complete"))
        and cert_ready
        and not gap_present
    )

    return ProofReuseRuntimeActivationReport(
        composition=_bounded_mapping(composition if isinstance(composition, Mapping) else {}),
        native_groth16=_bounded_mapping(native if isinstance(native, Mapping) else {}),
        test_certificate_authority=_bounded_mapping(
            certificate if isinstance(certificate, Mapping) else {}
        ),
        inventory=_bounded_mapping(inventory if isinstance(inventory, Mapping) else {}),
        activation_gap=gap,
        activation_blocker_codes=blockers,
        ordinary_default_composition_usable=bool(
            composition.get("ordinary_default_composition_usable")
            if isinstance(composition, Mapping)
            else False
        ),
        ordinary_warm_skip_path_complete=warm_complete,
        native_groth16_installed=bool(inventory.get("native_groth16_installed")),
        native_groth16_ready=bool(inventory.get("native_groth16_ready")),
        test_certificate_authority_ready=cert_ready,
        activation_gap_present=gap_present,
        process_started=bool(
            native.get("process_started") if isinstance(native, Mapping) else False
        ),
        source=source,
        reason_code=reason
        or str(getattr(resolved_services, "reason_code", "") or "")[:96],
    )


def proof_reuse_runtime_activation_report_from_root(
    root_path: str | os.PathLike[str],
    *,
    mode: Any = None,
    cache_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    installer: Any = None,
) -> ProofReuseRuntimeActivationReport:
    """Convenience wrapper that composes defaults under ``root_path``."""

    root = Path(root_path)
    resolved_cache = (
        Path(cache_root) if cache_root is not None else root / ".proof-reuse-cache"
    )
    return proof_reuse_runtime_activation_report(
        mode=mode,
        root_path=root,
        cache_root=resolved_cache,
        environ=environ,
        installer=installer,
        compose_if_missing=True,
    )


__all__ = [
    "MAX_REASON_CODES",
    "PROOF_REUSE_METRICS_INTERFACE",
    "PROOF_REUSE_METRICS_SCHEMA",
    "PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_INTERFACE",
    "PROOF_REUSE_RUNTIME_ACTIVATION_REPORT_SCHEMA",
    "ProofReuseMetricsSnapshot",
    "ProofReuseOutcome",
    "ProofReuseRuntimeActivationReport",
    "ProofReuseSessionMetrics",
    "proof_reuse_runtime_activation_report",
    "proof_reuse_runtime_activation_report_from_root",
]
