"""Closed runtime modes and fail-closed admission/promotion policy (PCCE-022).

Accelerator policy owns runtime admission only. Frozen MCP++ status and error
taxonomies remain wire authority. Importing this module performs no I/O,
network, process, or filesystem mutation and does not read environment
variables to select or promote a mode.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.compatibility import (
    FROZEN_MATRIX,
    CompatibilityError,
    reject_mock,
    reject_pseudo_cid,
)

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1"
POLICY_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/policy"
POLICY_RESULT_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/policy-result"
CONTRACT_VERSION: Final[str] = "0.1"
CONTRACT_SCHEMA_PREFIX: Final[str] = "pcce/proof-context/v0.1/"

PCCE_006_CONTENT_ID: Final[str] = (
    "sha256:b5503d2c2ec22e34091b3f747241fbde0519a9f0b213a03e0456a8f980a43f37"
)
COMPATIBILITY_MATRIX_CONTENT_ID: Final[str] = FROZEN_MATRIX["content_id"]
STATUS_TAXONOMY_CONTENT_ID: Final[str] = (
    "sha256:5f206feebb6213d3a1c113e37373ac8402003170cea609035ec9b871ca9fdd19"
)
ERROR_TAXONOMY_CONTENT_ID: Final[str] = (
    "sha256:570d43769cd47207f7c5f77bb7434252e6cefb1b4cf10791ccb82208db216a38"
)

MODES: Final[tuple[str, ...]] = (
    "production",
    "supervised",
    "evaluation",
    "simulation",
)
LIVE_MODES: Final[frozenset[str]] = frozenset({"production", "supervised"})
PROVENANCES: Final[tuple[str, ...]] = ("live", "replayed", "simulated")
QUALITY_CLASSES: Final[tuple[str, ...]] = PROVENANCES
FORBIDDEN_EVIDENCE: Final[tuple[str, ...]] = (
    "simulated",
    "replayed",
    "stale",
    "invalid",
    "unavailable",
    "pseudo-cid",
    "unsigned-required",
    "unsealed",
)
STATUSES: Final[tuple[str, ...]] = (
    "succeeded",
    "rejected",
    "verification_failed",
    "proof_failed",
    "assurance_failed",
    "context_insufficient",
    "model_escalation_required",
    "human_review_required",
    "unavailable",
    "timeout",
    "cancelled",
    "invalid",
    "stale",
    "simulated",
    "infrastructure_failure",
    "partial_effect",
    "repair_required",
)
ERRORS: Final[tuple[str, ...]] = (
    "unknown_field",
    "malformed",
    "identity_inconsistent",
    "stale_root",
    "simulated_promoted",
    "pseudo_cid",
    "schema_mismatch",
    "boundary_violation",
    "unavailable_capability",
    "timeout",
    "cancelled",
    "verification_failed",
    "proof_failed",
    "assurance_failed",
    "context_insufficient",
    "infrastructure_failure",
    "partial_effect",
    "repair_required",
    "human_review_required",
)

SIMULATION_WATERMARK: Final[str] = "pcce/proof-context/v0.1/simulation-watermark"
CID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{58,}$")
_CID_FIELDS: Final[tuple[str, ...]] = (
    "artifact_cid",
    "evidence_cid",
    "cid",
    "seal_cid",
    "repository_state_cid",
    "receipt_cid",
    "result_cid",
    "unit_cid",
    "parent_seal_cid",
    "parent_unit_cid",
)
_PARENT_FIELDS: Final[tuple[str, ...]] = (
    "parents",
    "parent_evidence",
    "evidence",
    "evidence_cids",
    "dependencies",
)
_PROMOTION_ENV_VARS: Final[tuple[str, ...]] = (
    "PCCE_MODE",
    "PCCE_RUNTIME_MODE",
    "IPFS_ACCELERATE_PCCE_MODE",
    "PROOF_CONTEXT_MODE",
    "PROOF_CONTEXT_RUNTIME_MODE",
)
_DEFECT_PRIORITY: Final[tuple[str, ...]] = (
    "self-approved",
    "pseudo-cid",
    "simulated",
    "replayed",
    "stale",
    "invalid",
    "unavailable",
    "unsigned-required",
    "unsealed",
)
_DEFECT_ERROR: Final[Mapping[str, str]] = MappingProxyType(
    {
        "self-approved": "boundary_violation",
        "pseudo-cid": "pseudo_cid",
        "simulated": "simulated_promoted",
        "replayed": "boundary_violation",
        "stale": "stale_root",
        "invalid": "malformed",
        "unavailable": "unavailable_capability",
        "unsigned-required": "malformed",
        "unsealed": "boundary_violation",
    }
)
_DEFECT_STATUS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "self-approved": "rejected",
        "pseudo-cid": "invalid",
        "simulated": "simulated",
        "replayed": "rejected",
        "stale": "stale",
        "invalid": "invalid",
        "unavailable": "unavailable",
        "unsigned-required": "invalid",
        "unsealed": "rejected",
    }
)


class PolicyError(RuntimeError):
    """Fail-closed policy error. Reason is a closed v0.1 error code."""

    reason = "malformed"

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        if reason is not None:
            if reason not in ERRORS:
                raise ValueError(f"policy error reason {reason!r} is not in the frozen taxonomy")
            self.reason = reason


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze(item) for item in sorted(value, key=repr))
    return value


def _canonicalize(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        parts = []
        for key in sorted(str(item) for item in value):
            parts.append(
                json.dumps(str(key), ensure_ascii=False, separators=(",", ":"))
                + ":"
                + _canonicalize(value[key] if key in value else value[str(key)])
            )
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonicalize(item) for item in value) + "]"
    raise PolicyError(
        f"unsupported policy canonicalization type {type(value).__name__}",
        reason="malformed",
    )


def mint_policy_cid(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonicalize(value).encode("utf-8")).digest()
    raw = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


def _as_mapping(evidence: Any) -> Mapping[str, Any]:
    if evidence is None:
        raise PolicyError("evidence is required", reason="malformed")
    if isinstance(evidence, PolicyResult):
        return evidence.to_mapping()
    if isinstance(evidence, Mapping):
        return evidence
    payload: dict[str, Any] = {}
    for name in (
        "mode",
        "provenance",
        "status",
        "quality_claim",
        "quality_class",
        "watermark",
        "sealed",
        "unsealed",
        "signature",
        "signature_required",
        "signed",
        "self_approved",
        "adapter_approved",
        "adapter_id",
        "approver_id",
        "stale",
        "available",
        "simulated",
        "replayed",
        "kind",
        *_CID_FIELDS,
        *_PARENT_FIELDS,
        "quality",
        "quality_claims",
        "payload",
    ):
        if hasattr(evidence, name):
            payload[name] = getattr(evidence, name)
    if not payload:
        raise PolicyError("evidence must be a mapping or policy record", reason="malformed")
    return payload


def _truthy(value: Any) -> bool:
    return value is True or value == "true" or value == 1


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _iter_parents(evidence: Mapping[str, Any]) -> tuple[Any, ...]:
    collected: list[Any] = []
    for name in _PARENT_FIELDS:
        raw = evidence.get(name)
        if raw is None:
            continue
        if isinstance(raw, Mapping) or not isinstance(raw, (list, tuple)):
            collected.append(raw)
        else:
            collected.extend(raw)
    return tuple(collected)


def _is_cidv1(value: str) -> bool:
    return bool(CID_PATTERN.fullmatch(value)) and len(value) >= 59


def _cid_values(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for name in _CID_FIELDS:
        raw = evidence.get(name)
        if raw is None or raw is False:
            continue
        if isinstance(raw, (list, tuple)):
            values.extend(str(item) for item in raw if item)
        else:
            text = str(raw)
            if text:
                values.append(text)
    for parent in _iter_parents(evidence):
        if isinstance(parent, str) and parent:
            values.append(parent)
    return tuple(values)


def admit_cid(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyError("CID is required", reason="pseudo_cid")
    try:
        reject_pseudo_cid(value)
    except CompatibilityError as exc:
        raise PolicyError("pseudo-CID is not admitted", reason="pseudo_cid") from exc
    if not _is_cidv1(value):
        raise PolicyError("pseudo-CID is not admitted", reason="pseudo_cid")
    return value


def admit_mode(mode: Any) -> str:
    if not isinstance(mode, str) or mode not in MODES:
        raise PolicyError(f"unknown runtime mode {mode!r}", reason="unknown_field")
    return mode


def admit_provenance(provenance: Any) -> str:
    if not isinstance(provenance, str) or provenance not in PROVENANCES:
        raise PolicyError(
            f"unknown provenance {provenance!r}",
            reason="unknown_field",
        )
    return provenance


def admit_status(status: Any) -> str:
    if not isinstance(status, str) or status not in STATUSES:
        raise PolicyError(f"unknown status {status!r}", reason="unknown_field")
    return status


def _quality_class_for(
    provenance: str,
    defects: Sequence[str],
    watermarked: bool,
) -> str:
    if watermarked or "simulated" in defects or provenance == "simulated":
        return "simulated"
    if "replayed" in defects or provenance == "replayed":
        return "replayed"
    return "live"


def _signature_present(evidence: Mapping[str, Any]) -> bool:
    if _truthy(evidence.get("signed")):
        return True
    signature = _optional_str(evidence.get("signature"))
    return bool(signature)


def _signature_required(evidence: Mapping[str, Any], *, status: str, mode: str) -> bool:
    if "signature_required" in evidence:
        return _truthy(evidence.get("signature_required"))
    if mode in LIVE_MODES and status == "succeeded":
        return True
    return False


def _is_unsealed(evidence: Mapping[str, Any], *, status: str, mode: str) -> bool:
    if _truthy(evidence.get("unsealed")):
        return True
    if "sealed" in evidence:
        return not _truthy(evidence.get("sealed"))
    seal_cid = _optional_str(evidence.get("seal_cid"))
    if mode in LIVE_MODES and status == "succeeded":
        return not bool(seal_cid)
    return False


def _self_approved(evidence: Mapping[str, Any]) -> bool:
    if _truthy(evidence.get("self_approved")) or _truthy(evidence.get("adapter_approved")):
        return True
    adapter_id = _optional_str(evidence.get("adapter_id"))
    approver_id = _optional_str(evidence.get("approver_id"))
    return bool(adapter_id and approver_id and adapter_id == approver_id)


def inspect_simulation_watermark(
    evidence: Any,
    *,
    _seen: frozenset[int] | None = None,
) -> bool:
    payload = _as_mapping(evidence)
    marker = id(payload)
    seen = _seen or frozenset()
    if marker in seen:
        return False
    next_seen = seen | {marker}
    provenance = payload.get("provenance")
    status = payload.get("status")
    if provenance == "simulated" or status == "simulated" or _truthy(payload.get("simulated")):
        return True
    watermark = _optional_str(payload.get("watermark"))
    if watermark:
        return True
    mode = payload.get("mode")
    if mode == "simulation":
        return True
    for parent in _iter_parents(payload):
        if isinstance(parent, str):
            continue
        try:
            if inspect_simulation_watermark(parent, _seen=next_seen):
                return True
        except PolicyError:
            continue
    return False


def apply_simulation_watermark(evidence: Any) -> Mapping[str, Any]:
    payload = dict(_as_mapping(evidence))
    payload["provenance"] = "simulated"
    payload["watermark"] = SIMULATION_WATERMARK
    payload["simulated"] = True
    if payload.get("status") == "succeeded":
        payload["status"] = "simulated"
    quality = payload.get("quality_claim") or payload.get("quality_class")
    if quality == "live":
        payload["quality_claim"] = "simulated"
        payload["quality_class"] = "simulated"
    return MappingProxyType(payload)


def _collect_defects(
    mode: str,
    evidence: Mapping[str, Any],
    *,
    provenance: str,
    status: str,
) -> tuple[str, ...]:
    defects: list[str] = []
    watermarked = inspect_simulation_watermark(evidence)
    if _self_approved(evidence):
        defects.append("self-approved")
    for cid in _cid_values(evidence):
        try:
            admit_cid(cid)
        except PolicyError:
            defects.append("pseudo-cid")
            break
    if (
        watermarked
        or provenance == "simulated"
        or status == "simulated"
        or _truthy(evidence.get("simulated"))
        or mode == "simulation"
    ):
        defects.append("simulated")
    replayed_parent = False
    for parent in _iter_parents(evidence):
        if isinstance(parent, str):
            continue
        try:
            parent_map = _as_mapping(parent)
        except PolicyError:
            continue
        if parent_map.get("provenance") == "replayed" or _truthy(parent_map.get("replayed")):
            replayed_parent = True
    if provenance == "replayed" or _truthy(evidence.get("replayed")) or replayed_parent:
        defects.append("replayed")
    if status == "stale" or _truthy(evidence.get("stale")):
        defects.append("stale")
    if status == "invalid":
        defects.append("invalid")
    if status == "unavailable" or evidence.get("available") is False:
        defects.append("unavailable")
    if _signature_required(evidence, status=status, mode=mode) and not _signature_present(
        evidence
    ):
        defects.append("unsigned-required")
    if _is_unsealed(evidence, status=status, mode=mode):
        defects.append("unsealed")
    ordered = tuple(item for item in _DEFECT_PRIORITY if item in set(defects))
    return ordered


def _forbidden_for_mode(mode: str, defects: Sequence[str], provenance: str) -> tuple[str, ...]:
    present = set(defects)
    if provenance == "simulated":
        present.add("simulated")
    elif provenance == "replayed":
        present.add("replayed")
    if mode in LIVE_MODES:
        blocked = {
            item
            for item in present
            if item in FORBIDDEN_EVIDENCE or item == "self-approved"
        }
        return tuple(item for item in _DEFECT_PRIORITY if item in blocked)
    if mode == "evaluation":
        blocked = present & {
            "self-approved",
            "stale",
            "invalid",
            "unavailable",
            "pseudo-cid",
            "unsigned-required",
            "unsealed",
        }
        return tuple(item for item in _DEFECT_PRIORITY if item in blocked)
    blocked = present & {"self-approved", "pseudo-cid"}
    return tuple(item for item in _DEFECT_PRIORITY if item in blocked)


def _admitted_for_mode(
    mode: str,
    *,
    provenance: str,
    defects: Sequence[str],
) -> bool:
    blocked = _forbidden_for_mode(mode, defects, provenance)
    if blocked:
        return False
    if mode in LIVE_MODES:
        return provenance == "live" and "simulated" not in defects and "replayed" not in defects
    if mode == "evaluation":
        return provenance in PROVENANCES
    return True


def _accepted_for_mode(
    mode: str,
    *,
    provenance: str,
    status: str,
    defects: Sequence[str],
    admitted: bool,
) -> bool:
    if not admitted:
        return False
    if mode in LIVE_MODES:
        return provenance == "live" and status == "succeeded" and not defects
    return False


def _result_status(
    *,
    requested_status: str,
    defects: Sequence[str],
    admitted: bool,
    accepted: bool,
    quality_class: str,
) -> str:
    if defects:
        return _DEFECT_STATUS[defects[0]]
    if quality_class == "simulated" or not admitted:
        if requested_status == "succeeded":
            return "simulated" if quality_class == "simulated" else "rejected"
    if accepted:
        return requested_status
    if requested_status == "succeeded" and quality_class == "simulated":
        return "simulated"
    return requested_status


def _quality_claim_conflict(evidence: Mapping[str, Any], quality_class: str) -> bool:
    declared = evidence.get("quality_claim") or evidence.get("quality_class")
    if declared is None:
        return False
    if declared not in QUALITY_CLASSES:
        raise PolicyError(
            f"unknown quality class {declared!r}",
            reason="unknown_field",
        )
    return declared != quality_class


def _reject_merged_quality_claims(evidence: Mapping[str, Any]) -> None:
    if "quality" in evidence and "quality_claims" not in evidence:
        raise PolicyError(
            "evaluation quality claims must be separated by live/replayed/simulated",
            reason="boundary_violation",
        )
    claims = evidence.get("quality_claims")
    if claims is None:
        return
    if not isinstance(claims, Mapping):
        raise PolicyError("quality_claims must be a mapping", reason="malformed")
    if any(key in claims for key in ("merged", "combined", "aggregate")):
        raise PolicyError(
            "evaluation quality claims cannot be merged across provenances",
            reason="boundary_violation",
        )
    unknown = [key for key in claims if key not in QUALITY_CLASSES]
    if unknown:
        raise PolicyError(
            f"unknown quality class {unknown[0]!r}",
            reason="unknown_field",
        )


@dataclass(frozen=True)
class PolicyResult:
    """Admission/promotion decision. Every result names all four closed modes."""

    schema: str
    mode: str
    closed_modes: tuple[str, ...]
    provenance: str
    quality_class: str
    status: str
    admitted: bool
    accepted: bool
    promotion_admitted: bool
    error: str | None
    reasons: tuple[str, ...]
    watermark: str | None
    policy_cid: str
    forbidden_evidence: tuple[str, ...]
    per_mode: Mapping[str, Any] = field(default_factory=dict)
    quality_claim: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "closed_modes", tuple(self.closed_modes))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "forbidden_evidence", tuple(self.forbidden_evidence))
        object.__setattr__(self, "per_mode", _freeze(self.per_mode))
        if self.closed_modes != MODES:
            raise PolicyError(
                "policy results must enumerate the four closed runtime modes",
                reason="schema_mismatch",
            )
        if self.mode not in MODES:
            raise PolicyError(f"unknown runtime mode {self.mode!r}", reason="unknown_field")
        if self.provenance not in PROVENANCES:
            raise PolicyError(
                f"unknown provenance {self.provenance!r}",
                reason="unknown_field",
            )
        if self.quality_class not in QUALITY_CLASSES:
            raise PolicyError(
                f"unknown quality class {self.quality_class!r}",
                reason="unknown_field",
            )
        if self.status not in STATUSES:
            raise PolicyError(f"unknown status {self.status!r}", reason="unknown_field")
        if self.error is not None and self.error not in ERRORS:
            raise PolicyError(f"unknown error {self.error!r}", reason="unknown_field")
        if self.accepted and not self.admitted:
            raise PolicyError(
                "accepted evidence must be admitted",
                reason="identity_inconsistent",
            )
        if self.accepted and self.mode in LIVE_MODES and self.quality_class != "live":
            raise PolicyError(
                "production and supervised cannot accept non-live quality",
                reason="simulated_promoted",
            )
        if self.status == "succeeded" and self.quality_class == "simulated":
            raise PolicyError(
                "simulated results cannot be labeled succeeded",
                reason="simulated_promoted",
            )

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "mode": self.mode,
                "closed_modes": self.closed_modes,
                "provenance": self.provenance,
                "quality_class": self.quality_class,
                "quality_claim": self.quality_claim,
                "status": self.status,
                "admitted": self.admitted,
                "accepted": self.accepted,
                "promotion_admitted": self.promotion_admitted,
                "error": self.error,
                "reasons": self.reasons,
                "watermark": self.watermark,
                "policy_cid": self.policy_cid,
                "forbidden_evidence": self.forbidden_evidence,
                "per_mode": dict(self.per_mode),
                "contract_version": CONTRACT_VERSION,
            }
        )


def _mode_snapshot(
    mode: str,
    *,
    provenance: str,
    status: str,
    defects: Sequence[str],
) -> dict[str, Any]:
    snapshot_defects = tuple(defects)
    snapshot_provenance = provenance
    if mode == "simulation":
        snapshot_defects = tuple(
            item for item in _DEFECT_PRIORITY if item in set(snapshot_defects) | {"simulated"}
        )
        snapshot_provenance = "simulated"
    admitted = _admitted_for_mode(
        mode,
        provenance=snapshot_provenance,
        defects=snapshot_defects,
    )
    accepted = _accepted_for_mode(
        mode,
        provenance=snapshot_provenance,
        status=status,
        defects=snapshot_defects,
        admitted=admitted,
    )
    quality_class = _quality_class_for(
        snapshot_provenance,
        snapshot_defects,
        watermarked="simulated" in snapshot_defects or mode == "simulation",
    )
    if mode == "simulation":
        quality_class = "simulated"
        accepted = False
    return {
        "mode": mode,
        "admitted": admitted,
        "accepted": accepted,
        "quality_class": quality_class,
        "promotion_admitted": False,
    }


def _build_result(
    mode: str,
    evidence: Mapping[str, Any],
    *,
    provenance: str,
    requested_status: str,
    defects: Sequence[str],
    promotion_admitted: bool = False,
    extra_reasons: Sequence[str] = (),
) -> PolicyResult:
    watermarked = inspect_simulation_watermark(evidence) or mode == "simulation"
    quality_class = _quality_class_for(provenance, defects, watermarked)
    if mode == "simulation":
        quality_class = "simulated"
        watermarked = True
        if "simulated" not in defects:
            defects = tuple(item for item in _DEFECT_PRIORITY if item in set(defects) | {"simulated"})
    if _quality_claim_conflict(evidence, quality_class):
        raise PolicyError(
            "quality claim does not match evidence provenance",
            reason="identity_inconsistent",
        )
    blocked = _forbidden_for_mode(mode, defects, provenance)
    result_provenance = "simulated" if watermarked else provenance
    admitted = _admitted_for_mode(mode, provenance=result_provenance, defects=defects)
    accepted = _accepted_for_mode(
        mode,
        provenance=result_provenance,
        status=requested_status,
        defects=defects,
        admitted=admitted,
    )
    if mode == "simulation":
        accepted = False
    if extra_reasons:
        # Closed promotion denials never admit into the target mode.
        admitted = False
        accepted = False
    result_status = _result_status(
        requested_status=requested_status,
        defects=blocked if not admitted else (),
        admitted=admitted,
        accepted=accepted,
        quality_class=quality_class,
    )
    if mode == "simulation" and result_status == "succeeded":
        result_status = "simulated"
    error = None
    if extra_reasons:
        error = extra_reasons[0] if extra_reasons[0] in ERRORS else "boundary_violation"
    elif not admitted:
        error = _DEFECT_ERROR[blocked[0]] if blocked else "boundary_violation"
    reasons = tuple(dict.fromkeys((*blocked, *extra_reasons)))
    snapshot_defects = defects
    if watermarked and "simulated" not in snapshot_defects:
        snapshot_defects = tuple(
            item for item in _DEFECT_PRIORITY if item in set(snapshot_defects) | {"simulated"}
        )
    per_mode = {
        item: _mode_snapshot(
            item,
            provenance=result_provenance,
            status=requested_status,
            defects=snapshot_defects,
        )
        for item in MODES
    }
    per_mode["simulation"]["quality_class"] = "simulated"
    per_mode["simulation"]["accepted"] = False
    per_mode["simulation"]["promotion_admitted"] = False
    watermark = SIMULATION_WATERMARK if watermarked or quality_class == "simulated" else None
    quality_claim = quality_class if mode in {"evaluation", "simulation"} else None
    return PolicyResult(
        schema=POLICY_RESULT_SCHEMA,
        mode=mode,
        closed_modes=MODES,
        provenance="simulated" if watermarked else provenance,
        quality_class=quality_class,
        status=result_status,
        admitted=admitted,
        accepted=accepted,
        promotion_admitted=promotion_admitted and admitted,
        error=error,
        reasons=reasons,
        watermark=watermark,
        policy_cid=POLICY_CID,
        forbidden_evidence=FORBIDDEN_EVIDENCE,
        per_mode=per_mode,
        quality_claim=quality_claim,
    )


def admit_evidence(mode: Any, evidence: Any) -> PolicyResult:
    """Admit evidence under an explicit closed runtime mode."""

    admitted_mode = admit_mode(mode)
    if admitted_mode in LIVE_MODES:
        try:
            reject_mock(evidence)
        except CompatibilityError as exc:
            raise PolicyError(
                "production and supervised reject mock evidence",
                reason="boundary_violation",
            ) from exc
    payload = _as_mapping(evidence)
    declared_mode = payload.get("mode")
    if declared_mode is not None and declared_mode != admitted_mode:
        raise PolicyError(
            "evidence mode does not match the requested runtime mode",
            reason="identity_inconsistent",
        )
    provenance = admit_provenance(payload.get("provenance", "live"))
    status = admit_status(payload.get("status", "rejected"))
    _reject_merged_quality_claims(payload)
    defects = _collect_defects(
        admitted_mode,
        payload,
        provenance=provenance,
        status=status,
    )
    return _build_result(
        admitted_mode,
        payload,
        provenance=provenance,
        requested_status=status,
        defects=defects,
    )


def require_admitted(mode: Any, evidence: Any) -> PolicyResult:
    result = admit_evidence(mode, evidence)
    if not result.admitted:
        raise PolicyError(
            result.error or "evidence was not admitted",
            reason=result.error or "boundary_violation",
        )
    return result


def promote(
    evidence: Any,
    *,
    source_mode: Any,
    target_mode: Any,
) -> PolicyResult:
    """Closed promotion. Simulation has no promotion path."""

    source = admit_mode(source_mode)
    target = admit_mode(target_mode)
    payload = _as_mapping(evidence)
    provenance = admit_provenance(payload.get("provenance", "live"))
    status = admit_status(payload.get("status", "rejected"))
    defects = list(
        _collect_defects(target, payload, provenance=provenance, status=status)
    )
    extra: list[str] = []
    if source == "simulation" or inspect_simulation_watermark(payload):
        extra.append("simulated_promoted")
        payload = dict(apply_simulation_watermark(payload))
        provenance = "simulated"
        if "simulated" not in defects:
            defects.append("simulated")
    if source != target:
        extra.append("boundary_violation")
    if provenance == "replayed" and target in LIVE_MODES:
        extra.append("boundary_violation")
        if "replayed" not in defects:
            defects.append("replayed")
    if extra or source != target:
        return _build_result(
            target,
            payload,
            provenance=provenance,
            requested_status=status,
            defects=tuple(item for item in _DEFECT_PRIORITY if item in set(defects)),
            promotion_admitted=False,
            extra_reasons=tuple(extra or ("boundary_violation",)),
        )
    result = admit_evidence(target, payload)
    return PolicyResult(
        schema=result.schema,
        mode=result.mode,
        closed_modes=result.closed_modes,
        provenance=result.provenance,
        quality_class=result.quality_class,
        status=result.status,
        admitted=result.admitted,
        accepted=result.accepted,
        promotion_admitted=False,
        error=result.error,
        reasons=result.reasons,
        watermark=result.watermark,
        policy_cid=result.policy_cid,
        forbidden_evidence=result.forbidden_evidence,
        per_mode=result.per_mode,
        quality_claim=result.quality_claim,
    )


def evaluation_quality_claims(evidence: Any) -> Mapping[str, Any]:
    """Separate evaluation claims by live/replayed/simulated. Never merge."""

    payload = _as_mapping(evidence)
    _reject_merged_quality_claims(payload)
    declared = payload.get("quality_claims")
    buckets: dict[str, Any] = {name: None for name in QUALITY_CLASSES}
    if isinstance(declared, Mapping):
        for name in QUALITY_CLASSES:
            if name in declared:
                buckets[name] = declared[name]
    else:
        provenance = admit_provenance(payload.get("provenance", "live"))
        result = admit_evidence("evaluation", payload)
        buckets[result.quality_class] = {
            "admitted": result.admitted,
            "status": result.status,
            "provenance": provenance,
            "watermark": result.watermark,
        }
        for name in QUALITY_CLASSES:
            if name != result.quality_class:
                buckets[name] = None
    return MappingProxyType(buckets)


def environment_promotion_keys() -> tuple[str, ...]:
    """Env vars that must never promote a mode. Policy does not read them."""

    return _PROMOTION_ENV_VARS


def _clean_decision_admitted(mode: str, provenance: str, evidence_class: str) -> bool:
    defects = () if evidence_class == "clean" else (evidence_class,)
    if evidence_class == "clean" and provenance == "simulated":
        defects = ("simulated",)
    if evidence_class == "clean" and provenance == "replayed":
        defects = ("replayed",)
    return _admitted_for_mode(mode, provenance=provenance, defects=defects)


def _clean_decision_accepted(mode: str, provenance: str, evidence_class: str) -> bool:
    defects = () if evidence_class == "clean" else (evidence_class,)
    if evidence_class == "clean" and provenance == "simulated":
        defects = ("simulated",)
    if evidence_class == "clean" and provenance == "replayed":
        defects = ("replayed",)
    admitted = _admitted_for_mode(mode, provenance=provenance, defects=defects)
    status = "succeeded" if evidence_class == "clean" and provenance != "simulated" else "rejected"
    if provenance == "simulated" or evidence_class == "simulated":
        status = "simulated"
    return _accepted_for_mode(
        mode,
        provenance=provenance,
        status=status,
        defects=defects,
        admitted=admitted,
    )


def _build_decision_table() -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    classes = ("clean", *FORBIDDEN_EVIDENCE)
    for mode in MODES:
        for provenance in PROVENANCES:
            for evidence_class in classes:
                admitted = _clean_decision_admitted(mode, provenance, evidence_class)
                accepted = _clean_decision_accepted(mode, provenance, evidence_class)
                quality = _quality_class_for(
                    provenance,
                    () if evidence_class == "clean" else (evidence_class,),
                    provenance == "simulated"
                    or evidence_class == "simulated"
                    or mode == "simulation",
                )
                if mode == "simulation":
                    quality = "simulated"
                    accepted = False
                error = None
                if not admitted:
                    defect = evidence_class if evidence_class != "clean" else (
                        "simulated" if provenance == "simulated" else "replayed"
                    )
                    error = _DEFECT_ERROR.get(defect, "boundary_violation")
                rows.append(
                    MappingProxyType(
                        {
                            "mode": mode,
                            "provenance": provenance,
                            "evidence_class": evidence_class,
                            "admitted": admitted,
                            "accepted": accepted,
                            "quality_class": quality,
                            "error": error,
                            "promotion_admitted": False,
                        }
                    )
                )
    return tuple(rows)


DECISION_TABLE: Final[tuple[Mapping[str, Any], ...]] = _build_decision_table()


def decision_table() -> tuple[Mapping[str, Any], ...]:
    return DECISION_TABLE


def decision_for(mode: str, provenance: str, evidence_class: str) -> Mapping[str, Any]:
    admit_mode(mode)
    admit_provenance(provenance)
    if evidence_class != "clean" and evidence_class not in FORBIDDEN_EVIDENCE:
        raise PolicyError(
            f"unknown evidence class {evidence_class!r}",
            reason="unknown_field",
        )
    for row in DECISION_TABLE:
        if (
            row["mode"] == mode
            and row["provenance"] == provenance
            and row["evidence_class"] == evidence_class
        ):
            return row
    raise PolicyError("decision table is incomplete", reason="schema_mismatch")


_DESCRIPTOR_BODY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "schema": POLICY_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_schema_prefix": CONTRACT_SCHEMA_PREFIX,
        "modes": MODES,
        "live_modes": tuple(sorted(LIVE_MODES)),
        "provenances": PROVENANCES,
        "quality_classes": QUALITY_CLASSES,
        "forbidden_evidence": FORBIDDEN_EVIDENCE,
        "statuses": STATUSES,
        "errors": ERRORS,
        "simulation_watermark": SIMULATION_WATERMARK,
        "pcce_006_content_id": PCCE_006_CONTENT_ID,
        "compatibility_matrix_content_id": COMPATIBILITY_MATRIX_CONTENT_ID,
        "status_taxonomy_content_id": STATUS_TAXONOMY_CONTENT_ID,
        "error_taxonomy_content_id": ERROR_TAXONOMY_CONTENT_ID,
        "environment_promotion_keys": _PROMOTION_ENV_VARS,
        "promotion_paths": (),
        "decision_table": tuple(dict(row) for row in DECISION_TABLE),
    }
)
POLICY_CID: Final[str] = mint_policy_cid(_DESCRIPTOR_BODY)
POLICY_DESCRIPTOR: Final[Mapping[str, Any]] = MappingProxyType(
    {**dict(_DESCRIPTOR_BODY), "cid": POLICY_CID}
)
POLICY: Final[Mapping[str, Any]] = POLICY_DESCRIPTOR


def policy_descriptor() -> Mapping[str, Any]:
    return POLICY_DESCRIPTOR


def policy_cid() -> str:
    return POLICY_CID


def frozen_taxonomy() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "pcce_006_content_id": PCCE_006_CONTENT_ID,
            "compatibility_matrix_content_id": COMPATIBILITY_MATRIX_CONTENT_ID,
            "status_taxonomy_content_id": STATUS_TAXONOMY_CONTENT_ID,
            "error_taxonomy_content_id": ERROR_TAXONOMY_CONTENT_ID,
            "statuses": STATUSES,
            "errors": ERRORS,
            "modes": MODES,
            "provenances": PROVENANCES,
        }
    )


__all__ = [
    "COMPATIBILITY_MATRIX_CONTENT_ID",
    "CONTRACT_SCHEMA_PREFIX",
    "CONTRACT_VERSION",
    "DECISION_TABLE",
    "ERRORS",
    "ERROR_TAXONOMY_CONTENT_ID",
    "FORBIDDEN_EVIDENCE",
    "LIVE_MODES",
    "MODES",
    "PCCE_006_CONTENT_ID",
    "POLICY",
    "POLICY_CID",
    "POLICY_DESCRIPTOR",
    "POLICY_RESULT_SCHEMA",
    "POLICY_SCHEMA",
    "PROVENANCES",
    "QUALITY_CLASSES",
    "SCHEMA",
    "SIMULATION_WATERMARK",
    "STATUSES",
    "STATUS_TAXONOMY_CONTENT_ID",
    "PolicyError",
    "PolicyResult",
    "admit_cid",
    "admit_evidence",
    "admit_mode",
    "admit_provenance",
    "admit_status",
    "apply_simulation_watermark",
    "decision_for",
    "decision_table",
    "environment_promotion_keys",
    "evaluation_quality_claims",
    "frozen_taxonomy",
    "inspect_simulation_watermark",
    "mint_policy_cid",
    "policy_cid",
    "policy_descriptor",
    "promote",
    "require_admitted",
]
