"""Truthful legacy proof/test receipt migration (IPS-044).

Classifies existing proof, test, and cache evidence into closed dispositions
without upgrading integrity, signed, direct, or simulated meaning.  Legacy
bytes may be staged for integrity transport, but they never enter the reusable
proof-unit cache unless current-policy verification admits them.

Dispositions:

* ``accept`` — payload is already a canonical IPS evidence record;
* ``adapt`` — fields map into a canonical shape without assurance upgrade;
* ``reverify`` — a legacy cache/index candidate requires fresh current-policy
  verification before any cache admission;
* ``reject`` — simulated, unknown, or malformed evidence; never cacheable.

This module consumes datasets classification and kit staging adapters without
cloning their schema or storage authority.  Classification is pure; staging
and admission are explicit opt-in side effects.

Evidence: ``ips/cross-repository-migration@1``.

Interfaces: ``LegacyEvidenceMigrationResult``, ``migrate_legacy_evidence``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

MIGRATION_EVIDENCE: Final[str] = "ips/cross-repository-migration@1"
IMPORT_HERMETICITY_EVIDENCE: Final[str] = "ips/import-hermeticity@1"
MIGRATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "legacy-evidence-migration@1"
)
MIGRATION_NAMESPACE: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/migration"
)
CONTRACT_VERSION: Final[int] = 1

MAX_IDENTIFIER_BYTES: Final[int] = 512
MAX_REASON_BYTES: Final[int] = 1_024
MAX_PAYLOAD_KEYS: Final[int] = 256

# Default proof-system labels used when adapting legacy integrity evidence.
_DEFAULT_INTEGRITY_SYSTEM: Final[str] = "integrity"
_DEFAULT_SIGNED_SYSTEM: Final[str] = "signed_receipt"
_DEFAULT_DIRECT_SYSTEM: Final[str] = "groth16"
_DEFAULT_AGGREGATION_SYSTEM: Final[str] = "receipt_aggregation"


class MigrationError(ValueError):
    """Fail-closed legacy evidence migration contract violation."""


class MigrationDisposition(str, Enum):
    """Closed accept / adapt / reverify / reject outcomes."""

    ACCEPT = "accept"
    ADAPT = "adapt"
    REVERIFY = "reverify"
    REJECT = "reject"


def closed_migration_dispositions() -> frozenset[str]:
    return frozenset(item.value for item in MigrationDisposition)


def _require_text(value: Any, field_name: str, *, maximum: int = MAX_IDENTIFIER_BYTES) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MigrationError(f"{field_name} must be a non-empty string")
    text = value.strip()
    if len(text.encode("utf-8")) > maximum:
        raise MigrationError(f"{field_name} exceeds {maximum} bytes")
    return text


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _looks_legacy_cache_candidate(payload: Mapping[str, Any]) -> bool:
    """Detect legacy cache/index hints that must reverify before reuse."""

    keys = {str(key).strip().casefold() for key in payload}
    markers = {
        "cache_key",
        "cache_index",
        "candidate_cid",
        "proof_cache_key",
        "reusable_cache",
        "index_hint",
        "stale_candidate",
        "legacy_cache_entry",
    }
    if keys & markers:
        return True
    role = str(payload.get("role") or payload.get("artifact_role") or "").casefold()
    if role in {"candidate", "cache_candidate", "index_hint"}:
        return True
    status = str(payload.get("cache_status") or payload.get("index_status") or "").casefold()
    if status in {"candidate", "stale", "unverified", "legacy"}:
        return True
    return False


def _proof_system_for_assurance(assurance: str, payload: Mapping[str, Any]) -> str:
    declared = payload.get("proof_system_id") or payload.get("proof_system")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    mapping = {
        "integrity_only": _DEFAULT_INTEGRITY_SYSTEM,
        "structural": _DEFAULT_INTEGRITY_SYSTEM,
        "predicate_only": _DEFAULT_INTEGRITY_SYSTEM,
        "signed_receipt": _DEFAULT_SIGNED_SYSTEM,
        "receipt_aggregation": _DEFAULT_AGGREGATION_SYSTEM,
        "direct_execution": _DEFAULT_DIRECT_SYSTEM,
        "simulated": "simulated",
        "unknown": _DEFAULT_INTEGRITY_SYSTEM,
    }
    return mapping.get(assurance, _DEFAULT_INTEGRITY_SYSTEM)


def _proof_mode_for_assurance(assurance: str) -> str:
    from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import ProofMode

    mapping = {
        "integrity_only": ProofMode.INTEGRITY_ONLY.value,
        "structural": ProofMode.INTEGRITY_ONLY.value,
        "predicate_only": ProofMode.INTEGRITY_ONLY.value,
        "signed_receipt": ProofMode.SIGNED_RECEIPT.value,
        "receipt_aggregation": ProofMode.RECEIPT_AGGREGATION.value,
        "direct_execution": ProofMode.DIRECT_EXECUTION_PROOF.value,
        "simulated": ProofMode.SIMULATED.value,
        "unknown": ProofMode.INTEGRITY_ONLY.value,
    }
    return mapping.get(assurance, ProofMode.INTEGRITY_ONLY.value)


def _terminal_for_assurance(assurance: str) -> str:
    from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import ProofTerminalStatus

    if assurance == "direct_execution":
        return ProofTerminalStatus.PROVED.value
    if assurance == "signed_receipt":
        return ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED.value
    if assurance == "receipt_aggregation":
        return ProofTerminalStatus.PROVED.value
    if assurance == "simulated":
        return ProofTerminalStatus.UNKNOWN.value
    return ProofTerminalStatus.INTEGRITY_VERIFIED.value


def _classify_with_datasets(
    payload: Mapping[str, Any] | None,
    *,
    declared_path: str,
) -> Any:
    """Delegate pure classification to the datasets migration authority."""

    from ipfs_datasets_py.logic.zkp.incremental_sealing.migration import (
        classify_legacy_receipt,
    )

    return classify_legacy_receipt(payload, declared_path=declared_path)


def _map_disposition(datasets_disposition: str, *, reverify: bool) -> MigrationDisposition:
    if reverify:
        return MigrationDisposition.REVERIFY
    if datasets_disposition == "accept":
        return MigrationDisposition.ACCEPT
    if datasets_disposition == "adapt":
        return MigrationDisposition.ADAPT
    return MigrationDisposition.REJECT


@dataclass(frozen=True, slots=True)
class LegacyEvidenceMigrationResult:
    """Truthful cross-repository migration outcome for one legacy payload.

    ``cache_admitted`` is True only when current-policy verification issued a
    :class:`CacheAdmissionRecord`.  Staging alone never sets it.  Simulated and
    rejected evidence always keep ``cache_admitted`` False.
    """

    disposition: MigrationDisposition
    path_family: str
    assurance: str
    proof_mode: str
    target_evidence_class: str
    establishes: str
    does_not_establish: str
    production_seal_allowed: bool
    reasons: tuple[str, ...]
    adapted_payload: Mapping[str, Any] | None = None
    cache_eligible: bool = False
    cache_admitted: bool = False
    requires_current_policy_verification: bool = True
    staged_cid: str | None = None
    staged_only: bool = False
    admission_reason_code: str | None = None
    verification_digest: str | None = None
    cache_admission_record: Mapping[str, Any] | None = None
    datasets_disposition: str | None = None
    schema: str = MIGRATION_SCHEMA
    evidence: str = MIGRATION_EVIDENCE

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, MigrationDisposition):
            raise MigrationError("disposition must be MigrationDisposition")
        object.__setattr__(
            self, "path_family", _require_text(self.path_family, "path_family")
        )
        object.__setattr__(
            self, "assurance", _require_text(self.assurance, "assurance")
        )
        object.__setattr__(
            self, "proof_mode", _require_text(self.proof_mode, "proof_mode")
        )
        object.__setattr__(
            self,
            "target_evidence_class",
            _require_text(self.target_evidence_class, "target_evidence_class"),
        )
        object.__setattr__(
            self,
            "establishes",
            _require_text(self.establishes, "establishes", maximum=MAX_REASON_BYTES),
        )
        object.__setattr__(
            self,
            "does_not_establish",
            _require_text(
                self.does_not_establish, "does_not_establish", maximum=MAX_REASON_BYTES
            ),
        )
        if type(self.production_seal_allowed) is not bool:
            raise MigrationError("production_seal_allowed must be bool")
        if type(self.cache_eligible) is not bool:
            raise MigrationError("cache_eligible must be bool")
        if type(self.cache_admitted) is not bool:
            raise MigrationError("cache_admitted must be bool")
        if type(self.requires_current_policy_verification) is not bool:
            raise MigrationError("requires_current_policy_verification must be bool")
        if type(self.staged_only) is not bool:
            raise MigrationError("staged_only must be bool")
        reasons = tuple(
            _require_text(item, "reason", maximum=MAX_REASON_BYTES) for item in self.reasons
        )
        if not reasons:
            raise MigrationError("reasons must be non-empty")
        object.__setattr__(self, "reasons", reasons)
        if self.adapted_payload is not None and not isinstance(
            self.adapted_payload, Mapping
        ):
            raise MigrationError("adapted_payload must be a mapping or None")
        if self.cache_admission_record is not None and not isinstance(
            self.cache_admission_record, Mapping
        ):
            raise MigrationError("cache_admission_record must be a mapping or None")
        # Hard non-upgrade invariants.
        if self.assurance in {"simulated", "unknown", "structural", "predicate_only"}:
            if self.cache_admitted:
                raise MigrationError(
                    "simulated/unknown/structural/predicate evidence cannot be "
                    "cache-admitted"
                )
            if self.production_seal_allowed and self.assurance in {
                "simulated",
                "unknown",
                "structural",
                "predicate_only",
            }:
                # production_seal_allowed may only be true for signed/direct/agg.
                if self.assurance in {"simulated", "unknown"}:
                    raise MigrationError(
                        "simulated/unknown assurance cannot allow production seals"
                    )
        if self.cache_admitted and not self.cache_eligible:
            raise MigrationError("cache_admitted requires cache_eligible")
        if self.cache_admitted and self.requires_current_policy_verification is False:
            # Admitted results already satisfied verification; flag stays True as
            # documentation of the gate that was applied.
            pass
        if (
            self.disposition is MigrationDisposition.REJECT
            and self.cache_admitted
        ):
            raise MigrationError("rejected evidence cannot be cache-admitted")
        if self.disposition is MigrationDisposition.REVERIFY and self.cache_admitted:
            raise MigrationError(
                "reverify disposition means cache admission has not completed"
            )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence": self.evidence,
            "contract_version": CONTRACT_VERSION,
            "disposition": self.disposition.value,
            "path_family": self.path_family,
            "assurance": self.assurance,
            "proof_mode": self.proof_mode,
            "target_evidence_class": self.target_evidence_class,
            "establishes": self.establishes,
            "does_not_establish": self.does_not_establish,
            "production_seal_allowed": self.production_seal_allowed,
            "reasons": list(self.reasons),
            "adapted_payload": (
                dict(self.adapted_payload) if self.adapted_payload is not None else None
            ),
            "cache_eligible": self.cache_eligible,
            "cache_admitted": self.cache_admitted,
            "requires_current_policy_verification": (
                self.requires_current_policy_verification
            ),
            "staged_cid": self.staged_cid,
            "staged_only": self.staged_only,
            "admission_reason_code": self.admission_reason_code,
            "verification_digest": self.verification_digest,
            "cache_admission_record": (
                dict(self.cache_admission_record)
                if self.cache_admission_record is not None
                else None
            ),
            "datasets_disposition": self.datasets_disposition,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_canonical())


def _build_candidate_evidence(
    classification: Any,
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Choose the best canonical evidence mapping for admission."""

    adapted = classification.adapted_payload
    if isinstance(adapted, Mapping) and adapted.get("evidence_class"):
        return dict(adapted)
    if payload.get("evidence_class"):
        return dict(payload)
    # Integrity-shaped digests from adapted integrity_only payloads.
    if isinstance(adapted, Mapping):
        digest = adapted.get("digest") or payload.get("digest")
        cid = adapted.get("cid") or payload.get("cid")
        if isinstance(digest, str) and isinstance(cid, str):
            return {
                "evidence_class": "IntegrityCommitment",
                "digest": digest,
                "cid": cid,
                "merkle_inclusion": adapted.get("merkle_inclusion")
                or payload.get("merkle_inclusion")
                or "leaf:0",
                "byte_length": int(
                    adapted.get("byte_length") or payload.get("byte_length") or 32
                ),
            }
    digest = payload.get("digest")
    cid = payload.get("cid")
    if isinstance(digest, str) and isinstance(cid, str):
        return {
            "evidence_class": "IntegrityCommitment",
            "digest": digest,
            "cid": cid,
            "merkle_inclusion": payload.get("merkle_inclusion") or "leaf:0",
            "byte_length": int(payload.get("byte_length") or 32),
        }
    return None


def _attempt_cache_admission(
    *,
    evidence: Mapping[str, Any],
    classification: Any,
    payload: Mapping[str, Any],
    admission_policy: Any | None,
    proof_unit_id: str,
    public_input_cid: str | None,
    proof_object_cid: str | None,
    required_for_seal: bool,
) -> tuple[bool, str | None, str | None, Mapping[str, Any] | None, tuple[str, ...]]:
    """Run current-policy verification; never treat classification as admission."""

    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
        EvidenceCandidate,
        verify_for_admission,
    )

    assurance = classification.assurance.value
    proof_system_id = _proof_system_for_assurance(assurance, payload)
    mode = _proof_mode_for_assurance(assurance)
    terminal = _terminal_for_assurance(assurance)
    pi = public_input_cid or str(
        evidence.get("digest")
        or evidence.get("input_commitment")
        or evidence.get("receipt_digest")
        or payload.get("public_input_cid")
        or "n/a"
    )
    obj = proof_object_cid or str(
        evidence.get("proof_cid")
        or evidence.get("cid")
        or payload.get("proof_object_cid")
        or "n/a"
    )
    expected = evidence.get("digest") if isinstance(evidence.get("digest"), str) else None
    candidate = EvidenceCandidate(
        evidence=evidence,
        proof_system_id=proof_system_id,
        public_input_cid=pi,
        proof_unit_id=proof_unit_id,
        proof_object_cid=obj,
        required_for_seal=required_for_seal,
        proof_mode=mode,
        terminal_status=terminal,
        expected_digest=expected,
        observed_digest=expected,
        logical_epoch=int(payload.get("logical_epoch") or 0),
    )
    decision = verify_for_admission(candidate, policy=admission_policy)
    if decision.admitted and decision.cache_admission_record is not None:
        record = decision.cache_admission_record.to_canonical()
        return (
            True,
            None,
            decision.verification_digest,
            record,
            (
                "current-policy verification admitted evidence",
                "legacy classification alone never grants cache admission",
            ),
        )
    message = (decision.message or "verification failed").strip()
    if len(message.encode("utf-8")) > MAX_REASON_BYTES:
        message = message.encode("utf-8")[: MAX_REASON_BYTES - 3].decode(
            "utf-8", errors="ignore"
        ) + "..."
    return (
        False,
        decision.reason_code,
        decision.verification_digest,
        None,
        (
            "current-policy verification rejected cache admission",
            f"reason:{decision.reason_code or 'unspecified'}",
            message,
        ),
    )


def _stage_legacy_bytes(
    stage_root: Path | str,
    payload: Mapping[str, Any],
    *,
    raw_bytes: bytes | None,
) -> tuple[str | None, tuple[str, ...]]:
    """Integrity-only staging via kit; never admission."""

    from ipfs_kit_py.proof_seal_store import stage_legacy_certificate_blob

    data = raw_bytes
    if data is None:
        data = _canonical_json(payload).encode("utf-8")
    if type(data) is not bytes:
        raise MigrationError("raw_bytes must be exact bytes when provided")
    staged = stage_legacy_certificate_blob(stage_root, data)
    if not staged.staged or staged.admitted or staged.accepted:
        return None, (
            "kit staging did not produce a non-admitted staged blob",
            f"staged={staged.staged}",
            f"admitted={staged.admitted}",
            f"accepted={staged.accepted}",
        )
    return staged.cid, (
        "legacy bytes staged for integrity transport only",
        "staging is not cache admission",
        "accelerate current-policy verification is still required",
    )


def migrate_legacy_evidence(
    payload: Mapping[str, Any] | None,
    *,
    declared_path: str = "",
    admit_to_cache: bool = False,
    admission_policy: Any | None = None,
    stage_root: Path | str | None = None,
    raw_bytes: bytes | None = None,
    proof_unit_id: str = "unit/legacy-migrated",
    public_input_cid: str | None = None,
    proof_object_cid: str | None = None,
    required_for_seal: bool = True,
    force_reverify: bool = False,
) -> LegacyEvidenceMigrationResult:
    """Migrate one legacy proof/test/cache payload without assurance upgrade.

    Classification is always pure.  Optional staging writes exact bytes through
    kit without admission.  Optional cache admission runs accelerate
    current-policy verification and is the only path that may set
    ``cache_admitted``.
    """

    if type(admit_to_cache) is not bool:
        raise MigrationError("admit_to_cache must be bool")
    if type(required_for_seal) is not bool:
        raise MigrationError("required_for_seal must be bool")
    if type(force_reverify) is not bool:
        raise MigrationError("force_reverify must be bool")
    if payload is not None and not isinstance(payload, Mapping):
        raise MigrationError("legacy evidence payload must be a mapping or None")
    if payload is not None and len(payload) > MAX_PAYLOAD_KEYS:
        raise MigrationError(f"legacy evidence payload exceeds {MAX_PAYLOAD_KEYS} keys")

    working = dict(payload) if payload is not None else None
    classification = _classify_with_datasets(working, declared_path=declared_path)
    datasets_disposition = classification.disposition.value
    assurance = classification.assurance.value
    reverify_hint = False
    if working is not None:
        reverify_hint = _looks_legacy_cache_candidate(working) or force_reverify
    if assurance == "simulated" or datasets_disposition == "reject":
        reverify_hint = False

    disposition = _map_disposition(datasets_disposition, reverify=reverify_hint)

    reasons: list[str] = list(classification.reasons)
    reasons.append("datasets classification preserves declared assurance class")
    reasons.append("legacy evidence never enters reusable cache without verification")

    staged_cid: str | None = None
    staged_only = False
    if stage_root is not None and working is not None:
        staged_cid, stage_reasons = _stage_legacy_bytes(
            stage_root, working, raw_bytes=raw_bytes
        )
        reasons.extend(stage_reasons)
        staged_only = staged_cid is not None and not admit_to_cache

    adapted = _as_mapping(classification.adapted_payload)
    cache_eligible = False
    cache_admitted = False
    admission_reason_code: str | None = None
    verification_digest: str | None = None
    cache_admission_record: Mapping[str, Any] | None = None

    if disposition is MigrationDisposition.REJECT or assurance == "simulated":
        reasons.append("rejected or simulated evidence is never cache-admitted")
        if admit_to_cache:
            reasons.append("admit_to_cache ignored for rejected/simulated evidence")
        return LegacyEvidenceMigrationResult(
            disposition=MigrationDisposition.REJECT,
            path_family=classification.path_family,
            assurance=assurance,
            proof_mode=classification.proof_mode.value,
            target_evidence_class=classification.target_evidence_class,
            establishes=classification.establishes,
            does_not_establish=classification.does_not_establish,
            production_seal_allowed=False,
            reasons=tuple(reasons),
            adapted_payload=adapted,
            cache_eligible=False,
            cache_admitted=False,
            requires_current_policy_verification=True,
            staged_cid=staged_cid,
            staged_only=staged_only,
            admission_reason_code="rejected_before_verification",
            datasets_disposition=datasets_disposition,
        )

    if disposition is MigrationDisposition.REVERIFY and not admit_to_cache:
        reasons.append(
            "legacy cache/index candidate requires reverify under current policy"
        )
        reasons.append("cache admission deferred until verify_for_admission succeeds")
        return LegacyEvidenceMigrationResult(
            disposition=MigrationDisposition.REVERIFY,
            path_family=classification.path_family,
            assurance=assurance,
            proof_mode=classification.proof_mode.value,
            target_evidence_class=classification.target_evidence_class,
            establishes=classification.establishes,
            does_not_establish=classification.does_not_establish,
            production_seal_allowed=bool(classification.production_seal_allowed),
            reasons=tuple(reasons),
            adapted_payload=adapted,
            cache_eligible=False,
            cache_admitted=False,
            requires_current_policy_verification=True,
            staged_cid=staged_cid,
            staged_only=staged_only,
            admission_reason_code="reverify_required",
            datasets_disposition=datasets_disposition,
        )

    if admit_to_cache:
        evidence = None
        if working is not None:
            evidence = _build_candidate_evidence(classification, working)
        if evidence is None:
            reasons.append("no canonical evidence mapping available for admission")
            return LegacyEvidenceMigrationResult(
                disposition=MigrationDisposition.REVERIFY
                if reverify_hint
                else (
                    MigrationDisposition.ACCEPT
                    if datasets_disposition == "accept"
                    else MigrationDisposition.ADAPT
                ),
                path_family=classification.path_family,
                assurance=assurance,
                proof_mode=classification.proof_mode.value,
                target_evidence_class=classification.target_evidence_class,
                establishes=classification.establishes,
                does_not_establish=classification.does_not_establish,
                production_seal_allowed=bool(classification.production_seal_allowed),
                reasons=tuple(reasons),
                adapted_payload=adapted,
                cache_eligible=False,
                cache_admitted=False,
                requires_current_policy_verification=True,
                staged_cid=staged_cid,
                staged_only=staged_only,
                admission_reason_code="missing_canonical_evidence",
                datasets_disposition=datasets_disposition,
            )
        (
            cache_admitted,
            admission_reason_code,
            verification_digest,
            cache_admission_record,
            admit_reasons,
        ) = _attempt_cache_admission(
            evidence=evidence,
            classification=classification,
            payload=working or {},
            admission_policy=admission_policy,
            proof_unit_id=proof_unit_id,
            public_input_cid=public_input_cid,
            proof_object_cid=proof_object_cid,
            required_for_seal=required_for_seal,
        )
        reasons.extend(admit_reasons)
        cache_eligible = cache_admitted
        if cache_admitted:
            # Successful current-policy verification clears the reverify hold.
            final_disposition = (
                MigrationDisposition.ACCEPT
                if datasets_disposition == "accept"
                else MigrationDisposition.ADAPT
            )
            staged_only = False
        else:
            final_disposition = MigrationDisposition.REVERIFY
    else:
        final_disposition = (
            MigrationDisposition.REVERIFY
            if reverify_hint
            else (
                MigrationDisposition.ACCEPT
                if datasets_disposition == "accept"
                else MigrationDisposition.ADAPT
            )
        )
        reasons.append(
            "classification complete; cache admission not requested "
            "(admit_to_cache=False)"
        )

    return LegacyEvidenceMigrationResult(
        disposition=final_disposition,
        path_family=classification.path_family,
        assurance=assurance,
        proof_mode=classification.proof_mode.value,
        target_evidence_class=classification.target_evidence_class,
        establishes=classification.establishes,
        does_not_establish=classification.does_not_establish,
        production_seal_allowed=bool(classification.production_seal_allowed),
        reasons=tuple(reasons),
        adapted_payload=adapted,
        cache_eligible=cache_eligible,
        cache_admitted=cache_admitted,
        requires_current_policy_verification=True,
        staged_cid=staged_cid,
        staged_only=staged_only and not cache_admitted,
        admission_reason_code=admission_reason_code,
        verification_digest=verification_digest,
        cache_admission_record=cache_admission_record,
        datasets_disposition=datasets_disposition,
    )


def migrate_legacy_evidence_batch(
    payloads: Sequence[Mapping[str, Any] | None],
    **kwargs: Any,
) -> tuple[LegacyEvidenceMigrationResult, ...]:
    """Migrate a sequence of legacy payloads with the same policy options."""

    return tuple(migrate_legacy_evidence(item, **kwargs) for item in payloads)


def migration_contract() -> dict[str, Any]:
    """Return the closed migration contract surface (pure)."""

    return {
        "schema": MIGRATION_SCHEMA,
        "evidence": MIGRATION_EVIDENCE,
        "import_hermeticity": IMPORT_HERMETICITY_EVIDENCE,
        "contract_version": CONTRACT_VERSION,
        "dispositions": sorted(closed_migration_dispositions()),
        "cache_admission_gate": "current_policy_verification_required",
        "assurance_upgrade_forbidden": True,
        "staging_is_not_admission": True,
        "simulated_never_cache_admitted": True,
        "consumes": {
            "datasets": "classify_legacy_receipt",
            "kit": "stage_legacy_certificate_blob",
            "accelerate": "verify_for_admission",
        },
    }


__all__ = (
    "CONTRACT_VERSION",
    "IMPORT_HERMETICITY_EVIDENCE",
    "MIGRATION_EVIDENCE",
    "MIGRATION_NAMESPACE",
    "MIGRATION_SCHEMA",
    "LegacyEvidenceMigrationResult",
    "MigrationDisposition",
    "MigrationError",
    "closed_migration_dispositions",
    "migrate_legacy_evidence",
    "migrate_legacy_evidence_batch",
    "migration_contract",
)
