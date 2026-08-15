"""Current-policy re-verification of legacy proof/test receipts (IPS-044).

Consumes datasets classification without cloning schema authority.  Legacy
evidence never enters the reusable cache unless current-policy verification
admits it.  Simulated, unsigned, and integrity-only payloads keep that meaning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.bootstrap import (
    IncrementalSealingBootstrap,
    bind_bootstrap,
)

MIGRATION_EVIDENCE: Final[str] = "ips/cross-repository-migration@1"
MIGRATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "legacy-evidence-migration-result@1"
)


class MigrationError(ValueError):
    """Fail-closed accelerate migration contract violation."""


class CacheAdmission(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    REQUIRES_REVERIFY = "requires_reverify"


@dataclass(frozen=True, slots=True)
class LegacyEvidenceMigrationResult:
    """Honest migration outcome.  Assurance is never upgraded."""

    schema: str
    evidence_subset: str
    disposition: str
    assurance: str
    cache_admission: CacheAdmission
    reusable: bool
    simulated: bool
    reason: str
    path_family: str = "unknown"

    def __post_init__(self) -> None:
        if self.reusable and self.cache_admission is not CacheAdmission.ADMITTED:
            raise MigrationError("reusable cache entry requires current-policy admission")
        if self.simulated and self.reusable:
            raise MigrationError("simulated evidence cannot enter the reusable cache")
        if self.reusable and self.disposition == "reject":
            raise MigrationError("rejected evidence cannot be reusable")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "disposition": self.disposition,
            "assurance": self.assurance,
            "cache_admission": self.cache_admission.value,
            "reusable": self.reusable,
            "simulated": self.simulated,
            "reason": self.reason,
            "path_family": self.path_family,
        }


def _as_mapping(payload: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    if hasattr(payload, "to_canonical") and callable(payload.to_canonical):
        canonical = payload.to_canonical()
        if isinstance(canonical, Mapping):
            return canonical
    raise MigrationError("legacy payload must be a mapping or canonical record")


def _classify(
    payload: Mapping[str, Any],
    bootstrap: IncrementalSealingBootstrap,
) -> Any:
    if bootstrap.datasets_classify is not None:
        return bootstrap.datasets_classify(payload)
    return None


def _field(classified: Any, name: str, default: Any = None) -> Any:
    if classified is None:
        return default
    if hasattr(classified, name):
        value = getattr(classified, name)
        return getattr(value, "value", value)
    if isinstance(classified, Mapping):
        return classified.get(name, default)
    return default


def migrate_legacy_evidence(
    payload: Mapping[str, Any] | Any,
    *,
    bootstrap: IncrementalSealingBootstrap | None = None,
    admit: Callable[[Mapping[str, Any]], bool] | None = None,
) -> LegacyEvidenceMigrationResult:
    """Accept, adapt, or reject legacy evidence under the current policy."""

    bound = bootstrap or bind_bootstrap(admit=admit)
    raw = _as_mapping(payload)
    classified = _classify(raw, bound)

    disposition = str(_field(classified, "disposition", "") or "")
    assurance = str(_field(classified, "assurance", "") or "")
    family = str(_field(classified, "path_family", "unknown") or "unknown")
    reason = str(_field(classified, "reason", "") or "unclassified_legacy_payload")
    simulated = bool(_field(classified, "simulated", False)) or assurance == "simulated"

    text = " ".join(str(value) for value in raw.values()).casefold()
    if not simulated:
        simulated = "simulated" in text or raw.get("simulated") is True
    if not disposition:
        if simulated:
            disposition = "reject"
            assurance = assurance or "simulated"
            reason = "simulated_evidence_rejected"
        elif raw.get("signature") or raw.get("signer_id"):
            disposition = "adapt"
            assurance = assurance or "signed_receipt"
            reason = "signed_receipt_requires_current_policy_reverify"
        elif raw.get("digest") or raw.get("integrity"):
            disposition = "adapt"
            assurance = assurance or "integrity_only"
            reason = "integrity_requires_current_policy_reverify"
        else:
            disposition = "reject"
            assurance = assurance or "unknown"
            reason = "unknown_legacy_payload"

    if simulated or disposition == "reject":
        return LegacyEvidenceMigrationResult(
            schema=MIGRATION_SCHEMA,
            evidence_subset=MIGRATION_EVIDENCE,
            disposition="reject",
            assurance="simulated" if simulated else (assurance or "unknown"),
            cache_admission=CacheAdmission.REJECTED,
            reusable=False,
            simulated=simulated,
            reason=reason,
            path_family=family,
        )

    verifier = admit or bound.admit
    if verifier is None:
        return LegacyEvidenceMigrationResult(
            schema=MIGRATION_SCHEMA,
            evidence_subset=MIGRATION_EVIDENCE,
            disposition=disposition,
            assurance=assurance or "integrity_only",
            cache_admission=CacheAdmission.REQUIRES_REVERIFY,
            reusable=False,
            simulated=False,
            reason="current_policy_verification_required_before_cache",
            path_family=family,
        )

    admitted = bool(verifier(raw))
    if not admitted:
        return LegacyEvidenceMigrationResult(
            schema=MIGRATION_SCHEMA,
            evidence_subset=MIGRATION_EVIDENCE,
            disposition="reject",
            assurance=assurance or "unknown",
            cache_admission=CacheAdmission.REJECTED,
            reusable=False,
            simulated=False,
            reason="current_policy_verification_rejected",
            path_family=family,
        )
    return LegacyEvidenceMigrationResult(
        schema=MIGRATION_SCHEMA,
        evidence_subset=MIGRATION_EVIDENCE,
        disposition=disposition,
        assurance=assurance or "integrity_only",
        cache_admission=CacheAdmission.ADMITTED,
        reusable=True,
        simulated=False,
        reason="current_policy_verified",
        path_family=family,
    )


__all__ = (
    "MIGRATION_EVIDENCE",
    "CacheAdmission",
    "LegacyEvidenceMigrationResult",
    "MigrationError",
    "migrate_legacy_evidence",
)
