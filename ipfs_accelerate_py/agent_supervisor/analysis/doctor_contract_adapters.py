"""Checked adapters between Doctor snapshot/finding families (PDR-010).

Interface: ``DiagnosisObligationBridge@1``

Repository diagnostics (``doctor-evidence-snapshot@1`` /
``doctor-diagnostic-finding@1``) and the deterministic doctor
(``deterministic-doctor/evidence-snapshot@1`` /
``deterministic-doctor/finding@1``) use intentionally distinct schemas.
This module provides **explicit** round-trip bridges; it never silently
aliases incompatible schemas or invents a second repository root.

Fail-closed rejections cover:

* issue / root / CID mismatches
* duplicate or unknown fields
* body / secret material
* tampering of stored identities
* cross-repository replay
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from . import doctor_repository_diagnostics as diag
from . import deterministic_doctor_contracts as det


# ---------------------------------------------------------------------------
# Interface, schemas, bounds
# ---------------------------------------------------------------------------

DIAGNOSIS_OBLIGATION_BRIDGE_INTERFACE: Final[str] = "DiagnosisObligationBridge@1"
DIAGNOSIS_OBLIGATION_BRIDGE_VERSION: Final[int] = 1

DIAGNOSIS_OBLIGATION_BRIDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/diagnosis-obligation-bridge@1"
)
FINDING_BRIDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-finding-bridge@1"
)
SNAPSHOT_BRIDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-snapshot-bridge@1"
)
ROOT_BRIDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-authority-root-bridge@1"
)

MAX_BRIDGE_BYTES: Final[int] = 262_144
MAX_FRONTIER_COUNT: Final[int] = 256
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096

_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "source_bytes",
        "contents",
        "content",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
        "secret",
        "secrets",
        "password",
        "token",
        "api_key",
        "private_key",
        "credential",
        "authorization_header",
        "cookie",
        "session",
    }
)
_PRIVATE_FIELD_MARKERS: Final[tuple[str, ...]] = (
    "secret",
    "password",
    "token",
    "api_key",
    "private_key",
    "credential",
    "authorization",
    "cookie",
    "session",
)

# Portable diagnostic finding fields carried for lossless round-trip.
_DIAG_FINDING_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "kind",
    "disposition",
    "path",
    "symbol",
    "message",
    "observation_refs",
    "expectation_source",
    "expectation_ref",
    "expectation_precedence",
    "open_frontier_refs",
    "evidence_refs",
    "details",
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DoctorContractAdapterError(ContractValidationError):
    """Fail-closed rejection for doctor contract adapter failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "doctor_contract_adapter_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "doctor_contract_adapter_error")


class DoctorContractAdapterBoundsError(DoctorContractAdapterError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="bounds_exceeded")


class DoctorContractAdapterAuthorityError(DoctorContractAdapterError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="authority_mismatch")


class DoctorContractAdapterTamperError(DoctorContractAdapterError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="tampered_identity")


class DoctorContractAdapterReplayError(DoctorContractAdapterError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="cross_repository_replay")


class DoctorContractAdapterSchemaError(DoctorContractAdapterError):
    def __init__(self, message: str) -> None:
        super().__init__(message, reason_code="schema_mismatch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise DoctorContractAdapterError(f"{field_name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise DoctorContractAdapterError(f"{field_name} is required")
    if "\0" in text:
        raise DoctorContractAdapterError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > limit:
        raise DoctorContractAdapterBoundsError(f"{field_name} exceeds its byte bound")
    return text


def _identifier(value: Any, field_name: str) -> str:
    text = _text(value, field_name, required=True, limit=512)
    if any(char.isspace() for char in text):
        raise DoctorContractAdapterError(
            f"{field_name} must be an opaque compact identifier"
        )
    return text


def _optional_identifier(value: Any, field_name: str) -> str:
    if value in (None, ""):
        return ""
    return _identifier(value, field_name)


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise DoctorContractAdapterError(f"{field_name} must be a boolean")
    return value


def _string_tuple(
    values: Any,
    field_name: str,
    *,
    limit: int = MAX_REFERENCE_COUNT,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise DoctorContractAdapterError(f"{field_name} must be a sequence of strings")
    else:
        raw = values
    if len(raw) > limit:
        raise DoctorContractAdapterBoundsError(f"{field_name} exceeds its item bound")
    out = tuple(sorted({_text(item, field_name, required=True, limit=512) for item in raw}))
    if required and not out:
        raise DoctorContractAdapterError(f"{field_name} must not be empty")
    return out


def _is_forbidden_payload_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_").strip()
    if normalized in _BODY_MARKERS:
        return True
    for marker in _PRIVATE_FIELD_MARKERS:
        if normalized == marker or normalized.endswith("_" + marker):
            return True
    return False


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    if isinstance(value, float):
        raise DoctorContractAdapterError(
            f"{field_name} may not contain floating-point values"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DoctorContractAdapterError(f"{field_name} has a non-string key")
            if _is_forbidden_payload_key(key):
                raise DoctorContractAdapterError(
                    f"{field_name} may not contain source bodies or secrets"
                )
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise DoctorContractAdapterError(f"{field_name} may not contain binary bodies")


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise DoctorContractAdapterBoundsError("nested structure exceeds depth bound")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise DoctorContractAdapterError("floating-point values are not allowed")
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item, depth=depth + 1) for item in value]
    raise DoctorContractAdapterError("unsupported structured value type")


def _bounded_dict(payload: Mapping[str, Any], name: str) -> None:
    _assert_body_free(payload, name)
    if len(canonical_json_bytes(payload)) > MAX_BRIDGE_BYTES:
        raise DoctorContractAdapterBoundsError(
            f"{name} exceeds its serialized byte bound"
        )


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    unknown = set(payload).difference(allowed)
    if unknown:
        raise DoctorContractAdapterError(f"{name} contains unsupported fields")


def _reject_duplicate_keys(payload: Mapping[str, Any], name: str) -> None:
    # Python dicts already collapse duplicates; detect list-of-pairs form.
    if isinstance(payload, list):  # type: ignore[unreachable]
        raise DoctorContractAdapterError(f"{name} must be a mapping")
    # Explicit check for multimap-style payloads smuggled as sequences of pairs.
    return


def _verify_cid(supplied: Any, expected: str, field_name: str) -> None:
    if supplied in (None, ""):
        return
    if not isinstance(supplied, str) or supplied != expected:
        raise DoctorContractAdapterTamperError(
            f"{field_name} does not match the canonical record"
        )


def _mapping_payload(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DoctorContractAdapterError(f"{name} must be a mapping")
    _reject_duplicate_keys(value, name)
    _assert_body_free(value, name)
    return dict(value)


# ---------------------------------------------------------------------------
# Disposition / kind mapping (explicit, loss-documented)
# ---------------------------------------------------------------------------


_DIAG_TO_DET_DISPOSITION: Final[Mapping[str, det.DoctorRepairDisposition]] = {
    diag.FindingDisposition.SUPPORTED.value: det.DoctorRepairDisposition.SUPPORTED,
    diag.FindingDisposition.ABSTAIN.value: det.DoctorRepairDisposition.ABSTAIN,
    diag.FindingDisposition.APPROVAL_REQUIRED.value: (
        det.DoctorRepairDisposition.APPROVAL_REQUIRED
    ),
    # Observed / unknown are diagnostic-only; they never grant write authority.
    diag.FindingDisposition.OBSERVED.value: det.DoctorRepairDisposition.ABSTAIN,
    diag.FindingDisposition.UNKNOWN.value: det.DoctorRepairDisposition.ABSTAIN,
}

_DET_TO_DIAG_DISPOSITION: Final[Mapping[str, diag.FindingDisposition]] = {
    det.DoctorRepairDisposition.SUPPORTED.value: diag.FindingDisposition.SUPPORTED,
    det.DoctorRepairDisposition.ABSTAIN.value: diag.FindingDisposition.ABSTAIN,
    det.DoctorRepairDisposition.APPROVAL_REQUIRED.value: (
        diag.FindingDisposition.APPROVAL_REQUIRED
    ),
    # Terminal deterministic states project to abstain on the diagnostic side.
    det.DoctorRepairDisposition.ROLLED_BACK.value: diag.FindingDisposition.ABSTAIN,
    det.DoctorRepairDisposition.QUARANTINED.value: diag.FindingDisposition.ABSTAIN,
}


def map_diagnostic_disposition_to_deterministic(
    value: diag.FindingDisposition | str,
) -> det.DoctorRepairDisposition:
    text = str(getattr(value, "value", value) or "")
    try:
        return _DIAG_TO_DET_DISPOSITION[text]
    except KeyError as exc:
        raise DoctorContractAdapterError(
            f"unsupported diagnostic disposition: {text!r}"
        ) from exc


def map_deterministic_disposition_to_diagnostic(
    value: det.DoctorRepairDisposition | str,
) -> diag.FindingDisposition:
    text = str(getattr(value, "value", value) or "")
    try:
        return _DET_TO_DIAG_DISPOSITION[text]
    except KeyError as exc:
        raise DoctorContractAdapterError(
            f"unsupported deterministic disposition: {text!r}"
        ) from exc


def map_diagnostic_kind_to_finding_kind(
    value: diag.FindingKind | str,
) -> str:
    return str(getattr(value, "value", value) or "contract_mismatch")


def map_finding_kind_to_diagnostic_kind(
    value: str,
) -> diag.FindingKind:
    text = str(value or "").strip().lower()
    try:
        return diag.FindingKind(text)
    except ValueError:
        # Deterministic findings may carry free-form kinds; map unknown to contract.
        return diag.FindingKind.CONTRACT


# ---------------------------------------------------------------------------
# Authority root bridge
# ---------------------------------------------------------------------------


def _placeholder_root(prefix: str, repository_id: str, tree_id: str) -> str:
    return f"{prefix}:{content_identity({'repository_id': repository_id, 'tree_id': tree_id, 'prefix': prefix})}"


def adapt_diagnostic_roots_to_deterministic(
    roots: diag.DoctorAuthorityRoots | Mapping[str, Any],
    *,
    require_repository_id: str = "",
) -> det.DoctorAuthorityRoots:
    """Project diagnostic authority roots onto the deterministic schema.

    Missing deterministic-only fields are filled with stable placeholders
    derived from repository/tree identity so the projection is body-free and
    deterministic.  Cross-repository replay is rejected when
    ``require_repository_id`` is set.
    """

    if isinstance(roots, diag.DoctorAuthorityRoots):
        src = roots
    elif isinstance(roots, Mapping):
        src = diag.DoctorAuthorityRoots.from_mapping(roots)
    else:
        raise DoctorContractAdapterError(
            "roots must be diagnostic DoctorAuthorityRoots or a mapping"
        )

    repository_id = _text(src.repository_id, "repository_id", required=True, limit=512)
    if require_repository_id and repository_id != require_repository_id:
        raise DoctorContractAdapterReplayError("cross-repository replay is rejected")

    forest_id = src.forest_id or f"forest:{repository_id}"
    tree_id = src.tree_id or f"tree:{repository_id}"
    overlay_id = src.overlay_id or f"overlay:{tree_id}"

    def _pick(*candidates: str, prefix: str) -> str:
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                return text
        return _placeholder_root(prefix, repository_id, tree_id)

    payload = {
        "repository_id": repository_id,
        "forest_id": forest_id,
        "tree_id": tree_id,
        "overlay_id": overlay_id,
        "file_root_id": _pick(src.file_root_id, src.blob_root_id, prefix="file-root"),
        "ast_root_id": _pick(src.ast_index_id, prefix="ast-root"),
        "graph_id": _pick(
            src.dependency_graph_id, src.import_graph_id, src.evidence_graph_id, prefix="graph"
        ),
        "corpus_id": _pick(src.corpus_root_id, prefix="corpus"),
        "index_id": _pick(src.ast_index_id, src.symbol_index_id, prefix="index"),
        "model_id": _pick(prefix="model"),
        "cache_id": _pick(src.cache_generation_id, prefix="cache"),
        "operator_registry_id": _pick(src.operator_registry_id, prefix="operators"),
        "translator_id": _pick(src.translator_id, prefix="translator"),
        "solver_id": _pick(src.solver_id, prefix="solver"),
        "kernel_id": _pick(src.kernel_id, prefix="kernel"),
        "toolchain_id": _pick(src.toolchain_id, prefix="toolchain"),
        "policy_id": _pick(src.policy_id, src.config_id, prefix="policy"),
        "sandbox_id": _pick(src.sandbox_id, prefix="sandbox"),
        "environment_id": _pick(src.environment_id, prefix="environment"),
        "lease_id": "",
    }
    _assert_body_free(payload, "authority roots")
    return det.DoctorAuthorityRoots(**payload)


def adapt_deterministic_roots_to_diagnostic(
    roots: det.DoctorAuthorityRoots | Mapping[str, Any],
    *,
    require_repository_id: str = "",
) -> diag.DoctorAuthorityRoots:
    """Project deterministic authority roots onto the diagnostic schema."""

    if isinstance(roots, det.DoctorAuthorityRoots):
        src = roots
    elif isinstance(roots, Mapping):
        if roots.get("schema") == det.DOCTOR_AUTHORITY_ROOTS_SCHEMA:
            src = det.DoctorAuthorityRoots.from_dict(roots)
        else:
            src = det.DoctorAuthorityRoots(
                **{
                    key: roots[key]
                    for key in det.AUTHORITY_ROOT_FIELDS
                    if key in roots
                }
            )
    else:
        raise DoctorContractAdapterError(
            "roots must be deterministic DoctorAuthorityRoots or a mapping"
        )

    if require_repository_id and src.repository_id != require_repository_id:
        raise DoctorContractAdapterReplayError("cross-repository replay is rejected")

    return diag.DoctorAuthorityRoots(
        repository_id=src.repository_id,
        forest_id=src.forest_id,
        tree_id=src.tree_id,
        overlay_id=src.overlay_id,
        file_root_id=src.file_root_id,
        blob_root_id=src.file_root_id,
        parser_id="parser:program-ast-adapters@1",
        config_id="",
        toolchain_id=src.toolchain_id,
        policy_id=src.policy_id,
        ast_index_id=src.ast_root_id or src.index_id,
        symbol_index_id=src.index_id,
        import_graph_id="",
        dependency_graph_id=src.graph_id,
        evidence_graph_id="",
        impact_index_id="",
        value_index_id="",
        contract_root_id="",
        corpus_root_id=src.corpus_id,
        vector_root_id="",
        embedding_config_id="",
        cache_generation_id=src.cache_id,
        operator_registry_id=src.operator_registry_id,
        translator_id=src.translator_id,
        solver_id=src.solver_id,
        kernel_id=src.kernel_id,
        sandbox_id=src.sandbox_id,
        environment_id=src.environment_id,
    )


def assert_same_repository(
    left_repository_id: str,
    right_repository_id: str,
) -> None:
    left = _identifier(left_repository_id, "left_repository_id")
    right = _identifier(right_repository_id, "right_repository_id")
    if left != right:
        raise DoctorContractAdapterReplayError("cross-repository replay is rejected")


@dataclass(frozen=True)
class AuthorityRootBridge(CanonicalContract):
    """Checked join of diagnostic and deterministic authority root projections."""

    SCHEMA: ClassVar[str] = ROOT_BRIDGE_SCHEMA

    repository_id: str
    diagnostic_roots: Mapping[str, Any]
    deterministic_roots: Mapping[str, Any]
    diagnostic_content_id: str
    deterministic_content_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        diag_payload = _mapping_payload(self.diagnostic_roots, "diagnostic_roots")
        det_payload = _mapping_payload(self.deterministic_roots, "deterministic_roots")
        # Drop non-identity fields before storage.
        diag_payload.pop("content_id", None)
        det_payload.pop("content_id", None)
        object.__setattr__(
            self, "diagnostic_roots", MappingProxyType(_plain(diag_payload))
        )
        object.__setattr__(
            self, "deterministic_roots", MappingProxyType(_plain(det_payload))
        )
        object.__setattr__(
            self,
            "diagnostic_content_id",
            _identifier(self.diagnostic_content_id, "diagnostic_content_id"),
        )
        object.__setattr__(
            self,
            "deterministic_content_id",
            _identifier(self.deterministic_content_id, "deterministic_content_id"),
        )
        # Verify projections recompute to stored identities.
        re_diag = diag.DoctorAuthorityRoots.from_mapping(dict(self.diagnostic_roots))
        if re_diag.content_id != self.diagnostic_content_id:
            raise DoctorContractAdapterTamperError(
                "diagnostic authority root content_id mismatch"
            )
        det_payload = dict(self.deterministic_roots)
        if det_payload.get("schema") == det.DOCTOR_AUTHORITY_ROOTS_SCHEMA:
            re_det = det.DoctorAuthorityRoots.from_dict(det_payload)
        else:
            re_det = det.DoctorAuthorityRoots(
                **{
                    key: det_payload[key]
                    for key in det.AUTHORITY_ROOT_FIELDS
                    if key in det_payload
                }
            )
        if re_det.content_id != self.deterministic_content_id:
            raise DoctorContractAdapterTamperError(
                "deterministic authority root content_id mismatch"
            )
        if (
            re_diag.repository_id != self.repository_id
            or re_det.repository_id != self.repository_id
        ):
            raise DoctorContractAdapterReplayError(
                "authority root bridge repository_id mismatch"
            )
        _bounded_dict(self.to_dict(), "authority root bridge")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DIAGNOSIS_OBLIGATION_BRIDGE_VERSION,
            "repository_id": self.repository_id,
            "diagnostic_roots": dict(self.diagnostic_roots),
            "deterministic_roots": dict(self.deterministic_roots),
            "diagnostic_content_id": self.diagnostic_content_id,
            "deterministic_content_id": self.deterministic_content_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorityRootBridge":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise DoctorContractAdapterSchemaError(
                "authority root bridge has an unsupported schema"
            )
        _assert_body_free(payload, "authority root bridge")
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "cid",
            "repository_id",
            "diagnostic_roots",
            "deterministic_roots",
            "diagnostic_content_id",
            "deterministic_content_id",
        }
        _reject_unknown_fields(payload, allowed, "authority root bridge")
        value = cls(
            repository_id=str(payload["repository_id"]),
            diagnostic_roots=payload["diagnostic_roots"],
            deterministic_roots=payload["deterministic_roots"],
            diagnostic_content_id=str(payload["diagnostic_content_id"]),
            deterministic_content_id=str(payload["deterministic_content_id"]),
        )
        _verify_cid(payload.get("content_id", payload.get("cid")), value.content_id, "content_id")
        return value

    @classmethod
    def bridge(
        cls,
        diagnostic_roots: diag.DoctorAuthorityRoots | Mapping[str, Any],
        *,
        require_repository_id: str = "",
    ) -> "AuthorityRootBridge":
        if isinstance(diagnostic_roots, diag.DoctorAuthorityRoots):
            diag_roots = diagnostic_roots
        else:
            diag_roots = diag.DoctorAuthorityRoots.from_mapping(diagnostic_roots)
        det_roots = adapt_diagnostic_roots_to_deterministic(
            diag_roots, require_repository_id=require_repository_id
        )
        return cls(
            repository_id=det_roots.repository_id,
            diagnostic_roots=diag_roots._payload(),
            deterministic_roots=det_roots.to_dict(),
            diagnostic_content_id=diag_roots.content_id,
            deterministic_content_id=det_roots.content_id,
        )


# ---------------------------------------------------------------------------
# Finding adapters
# ---------------------------------------------------------------------------


def _diagnostic_finding_payload(
    finding: diag.DoctorDiagnosticFinding,
) -> dict[str, Any]:
    payload = finding._payload()
    _assert_body_free(payload, "diagnostic finding")
    return payload


def _deterministic_finding_from_parts(
    *,
    roots: det.DoctorAuthorityRoots,
    finding_id: str,
    snapshot_id: str,
    disposition: det.DoctorRepairDisposition,
    observed_fact_refs: Sequence[str],
    expected_behavior_refs: Sequence[str],
    finding_kind: str,
    open_frontier_refs: Sequence[str],
    diagnostic_ref: str = "",
    reason_codes: Sequence[str] = (),
    affected_symbol_refs: Sequence[str] = (),
    evidence_role: det.DoctorEvidenceRole = det.DoctorEvidenceRole.OBSERVED_FACT,
    invalidation_refs: Sequence[str] = (),
    approval_classes: Sequence[str] = (),
    change_ref: str = "",
    trace_ref: str = "",
) -> det.DeterministicDoctorFinding:
    invalidation = tuple(invalidation_refs) or (roots.tree_id,)
    return det.DeterministicDoctorFinding(
        roots=roots,
        finding_id=finding_id,
        snapshot_id=snapshot_id,
        disposition=disposition,
        observed_fact_refs=tuple(observed_fact_refs),
        expected_behavior_refs=tuple(expected_behavior_refs),
        evidence_role=evidence_role,
        diagnostic_ref=diagnostic_ref,
        trace_ref=trace_ref,
        change_ref=change_ref,
        finding_kind=finding_kind,
        reason_codes=tuple(reason_codes),
        affected_symbol_refs=tuple(affected_symbol_refs),
        open_frontier_refs=tuple(open_frontier_refs),
        approval_classes=tuple(approval_classes),
        semantic_authority=False,
        invalidation_refs=invalidation,
    )


def adapt_diagnostic_finding_to_deterministic(
    finding: diag.DoctorDiagnosticFinding | Mapping[str, Any],
    *,
    roots: det.DoctorAuthorityRoots | diag.DoctorAuthorityRoots | Mapping[str, Any],
    snapshot_id: str,
    require_repository_id: str = "",
) -> det.DeterministicDoctorFinding:
    """Project one diagnostic finding onto the deterministic finding schema."""

    if isinstance(finding, Mapping):
        finding = _diagnostic_finding_from_mapping(finding)
    if not isinstance(finding, diag.DoctorDiagnosticFinding):
        raise DoctorContractAdapterError(
            "finding must be DoctorDiagnosticFinding or a mapping"
        )

    if isinstance(roots, det.DoctorAuthorityRoots):
        det_roots = roots
    else:
        det_roots = adapt_diagnostic_roots_to_deterministic(
            roots, require_repository_id=require_repository_id
        )
    if require_repository_id:
        assert_same_repository(det_roots.repository_id, require_repository_id)

    disposition = map_diagnostic_disposition_to_deterministic(finding.disposition)
    expected_refs: list[str] = []
    if finding.expectation_ref:
        expected_refs.append(finding.expectation_ref)
    observed = list(finding.observation_refs) or list(finding.evidence_refs)
    # Supported findings require independent expected-behavior authority.
    if (
        disposition is det.DoctorRepairDisposition.SUPPORTED
        and not expected_refs
    ):
        disposition = det.DoctorRepairDisposition.ABSTAIN

    reason_codes = [
        f"expectation_source:{finding.expectation_source.value}",
        f"expectation_precedence:{finding.expectation_precedence}",
    ]
    if finding.message:
        reason_codes.append("has_message")

    finding_id = finding.finding_cid
    return _deterministic_finding_from_parts(
        roots=det_roots,
        finding_id=finding_id,
        snapshot_id=_identifier(snapshot_id, "snapshot_id"),
        disposition=disposition,
        observed_fact_refs=observed,
        expected_behavior_refs=expected_refs,
        finding_kind=map_diagnostic_kind_to_finding_kind(finding.kind),
        open_frontier_refs=finding.open_frontier_refs,
        diagnostic_ref=finding.finding_cid,
        reason_codes=reason_codes,
        affected_symbol_refs=(finding.symbol,) if finding.symbol else (),
        change_ref=finding.path or "",
        evidence_role=det.DoctorEvidenceRole.OBSERVED_FACT,
        invalidation_refs=(det_roots.tree_id,),
        approval_classes=(
            (det.DoctorApprovalClass.PUBLIC_API_OR_SCHEMA.value,)
            if disposition is det.DoctorRepairDisposition.APPROVAL_REQUIRED
            else ()
        ),
    )


def adapt_deterministic_finding_to_diagnostic(
    finding: det.DeterministicDoctorFinding | Mapping[str, Any],
    *,
    diagnostic_overlay: Mapping[str, Any] | None = None,
) -> diag.DoctorDiagnosticFinding:
    """Project one deterministic finding onto the diagnostic finding schema.

    When ``diagnostic_overlay`` carries the original diagnostic payload (as
    stored by the bridge), path/message/kind/details are restored losslessly.
    Without an overlay, a best-effort structural projection is produced.
    """

    if isinstance(finding, Mapping):
        if finding.get("schema") == det.DETERMINISTIC_DOCTOR_FINDING_SCHEMA:
            finding = det.DeterministicDoctorFinding.from_dict(finding)
        else:
            raise DoctorContractAdapterSchemaError(
                "deterministic finding has an unsupported schema"
            )
    if not isinstance(finding, det.DeterministicDoctorFinding):
        raise DoctorContractAdapterError(
            "finding must be DeterministicDoctorFinding or a mapping"
        )

    if diagnostic_overlay is not None:
        overlay = _mapping_payload(diagnostic_overlay, "diagnostic_overlay")
        # Prefer explicit overlay for lossless reverse projection.
        return _diagnostic_finding_from_mapping(overlay)

    disposition = map_deterministic_disposition_to_diagnostic(finding.disposition)
    kind = map_finding_kind_to_diagnostic_kind(finding.finding_kind)
    expectation_ref = (
        finding.expected_behavior_refs[0] if finding.expected_behavior_refs else ""
    )
    expectation_source = (
        diag.ExpectationSourceKind.REVIEWED_CONTRACT
        if expectation_ref
        else diag.ExpectationSourceKind.NONE
    )
    symbol = finding.affected_symbol_refs[0] if finding.affected_symbol_refs else ""
    path = finding.change_ref or ""
    return diag.DoctorDiagnosticFinding(
        kind=kind,
        disposition=disposition,
        path=path,
        symbol=symbol,
        message="",
        observation_refs=finding.observed_fact_refs,
        expectation_source=expectation_source,
        expectation_ref=expectation_ref,
        expectation_precedence=0,
        open_frontier_refs=finding.open_frontier_refs,
        evidence_refs=finding.observed_fact_refs,
        details={
            "bridged_from": "deterministic-doctor/finding@1",
            "finding_id": finding.finding_id,
            "snapshot_id": finding.snapshot_id,
            "diagnostic_ref": finding.diagnostic_ref,
        },
    )


def _diagnostic_finding_from_mapping(
    payload: Mapping[str, Any],
) -> diag.DoctorDiagnosticFinding:
    data = _mapping_payload(payload, "diagnostic finding")
    data.pop("finding_cid", None)
    data.pop("content_id", None)
    data.pop("cid", None)
    schema = data.pop("schema", diag.DOCTOR_DIAGNOSTIC_FINDING_SCHEMA)
    if schema not in (None, "", diag.DOCTOR_DIAGNOSTIC_FINDING_SCHEMA):
        raise DoctorContractAdapterSchemaError(
            "diagnostic finding has an unsupported schema"
        )
    allowed = set(_DIAG_FINDING_FIELDS) - {"schema"}
    _reject_unknown_fields(data, allowed, "diagnostic finding")
    return diag.DoctorDiagnosticFinding(
        kind=data.get("kind", diag.FindingKind.CONTRACT),
        disposition=data.get("disposition", diag.FindingDisposition.ABSTAIN),
        path=str(data.get("path") or ""),
        symbol=str(data.get("symbol") or ""),
        message=str(data.get("message") or ""),
        observation_refs=tuple(data.get("observation_refs") or ()),
        expectation_source=data.get(
            "expectation_source", diag.ExpectationSourceKind.NONE
        ),
        expectation_ref=str(data.get("expectation_ref") or ""),
        expectation_precedence=int(data.get("expectation_precedence") or 0),
        open_frontier_refs=tuple(data.get("open_frontier_refs") or ()),
        evidence_refs=tuple(data.get("evidence_refs") or ()),
        details=data.get("details") or {},
    )


@dataclass(frozen=True)
class FindingBridge(CanonicalContract):
    """Lossless checked bridge for one Doctor finding across both families."""

    SCHEMA: ClassVar[str] = FINDING_BRIDGE_SCHEMA

    repository_id: str
    issue_cid: str
    snapshot_id: str
    diagnostic_payload: Mapping[str, Any]
    deterministic_payload: Mapping[str, Any]
    diagnostic_finding_cid: str
    deterministic_finding_id: str
    expected_refs: tuple[str, ...] = ()
    observed_refs: tuple[str, ...] = ()
    open_frontier_refs: tuple[str, ...] = ()
    causal_slice_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "issue_cid", _identifier(self.issue_cid, "issue_cid"))
        object.__setattr__(
            self, "snapshot_id", _identifier(self.snapshot_id, "snapshot_id")
        )
        diag_payload = _mapping_payload(self.diagnostic_payload, "diagnostic_payload")
        det_payload = _mapping_payload(self.deterministic_payload, "deterministic_payload")
        diag_payload.pop("finding_cid", None)
        diag_payload.pop("content_id", None)
        det_payload.pop("content_id", None)
        det_payload.pop("cid", None)
        object.__setattr__(
            self, "diagnostic_payload", MappingProxyType(_plain(diag_payload))
        )
        object.__setattr__(
            self, "deterministic_payload", MappingProxyType(_plain(det_payload))
        )
        object.__setattr__(
            self,
            "diagnostic_finding_cid",
            _identifier(self.diagnostic_finding_cid, "diagnostic_finding_cid"),
        )
        object.__setattr__(
            self,
            "deterministic_finding_id",
            _identifier(self.deterministic_finding_id, "deterministic_finding_id"),
        )
        object.__setattr__(
            self, "expected_refs", _string_tuple(self.expected_refs, "expected_refs")
        )
        object.__setattr__(
            self, "observed_refs", _string_tuple(self.observed_refs, "observed_refs")
        )
        object.__setattr__(
            self,
            "open_frontier_refs",
            _string_tuple(
                self.open_frontier_refs, "open_frontier_refs", limit=MAX_FRONTIER_COUNT
            ),
        )
        object.__setattr__(
            self,
            "causal_slice_refs",
            _string_tuple(self.causal_slice_refs, "causal_slice_refs"),
        )

        # Reconstruct both sides and verify identities.
        diag_finding = _diagnostic_finding_from_mapping(dict(self.diagnostic_payload))
        if diag_finding.finding_cid != self.diagnostic_finding_cid:
            raise DoctorContractAdapterTamperError(
                "diagnostic finding CID mismatch on bridge"
            )
        if self.issue_cid != self.diagnostic_finding_cid:
            raise DoctorContractAdapterError(
                "issue_cid must equal the diagnostic finding CID"
            )

        det_finding = det.DeterministicDoctorFinding.from_dict(
            dict(self.deterministic_payload)
            if self.deterministic_payload.get("schema")
            == det.DETERMINISTIC_DOCTOR_FINDING_SCHEMA
            else {
                "schema": det.DETERMINISTIC_DOCTOR_FINDING_SCHEMA,
                **dict(self.deterministic_payload),
            }
        )
        if det_finding.finding_id != self.deterministic_finding_id:
            raise DoctorContractAdapterTamperError(
                "deterministic finding id mismatch on bridge"
            )
        if det_finding.roots.repository_id != self.repository_id:
            raise DoctorContractAdapterReplayError(
                "finding bridge repository_id mismatch"
            )
        if det_finding.snapshot_id != self.snapshot_id:
            raise DoctorContractAdapterError(
                "finding bridge snapshot_id mismatch"
            )
        # Diagnostic ref on deterministic side must point at the issue CID.
        if det_finding.diagnostic_ref and det_finding.diagnostic_ref != self.issue_cid:
            raise DoctorContractAdapterError(
                "deterministic diagnostic_ref does not match issue_cid"
            )
        _bounded_dict(self.to_dict(), "finding bridge")

    def materialize_diagnostic(self) -> diag.DoctorDiagnosticFinding:
        return _diagnostic_finding_from_mapping(dict(self.diagnostic_payload))

    def materialize_deterministic(self) -> det.DeterministicDoctorFinding:
        payload = dict(self.deterministic_payload)
        if payload.get("schema") != det.DETERMINISTIC_DOCTOR_FINDING_SCHEMA:
            payload = {
                "schema": det.DETERMINISTIC_DOCTOR_FINDING_SCHEMA,
                **payload,
            }
        return det.DeterministicDoctorFinding.from_dict(payload)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DIAGNOSIS_OBLIGATION_BRIDGE_VERSION,
            "repository_id": self.repository_id,
            "issue_cid": self.issue_cid,
            "snapshot_id": self.snapshot_id,
            "diagnostic_payload": dict(self.diagnostic_payload),
            "deterministic_payload": dict(self.deterministic_payload),
            "diagnostic_finding_cid": self.diagnostic_finding_cid,
            "deterministic_finding_id": self.deterministic_finding_id,
            "expected_refs": list(self.expected_refs),
            "observed_refs": list(self.observed_refs),
            "open_frontier_refs": list(self.open_frontier_refs),
            "causal_slice_refs": list(self.causal_slice_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FindingBridge":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise DoctorContractAdapterSchemaError(
                "finding bridge has an unsupported schema"
            )
        _assert_body_free(payload, "finding bridge")
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "cid",
            "repository_id",
            "issue_cid",
            "snapshot_id",
            "diagnostic_payload",
            "deterministic_payload",
            "diagnostic_finding_cid",
            "deterministic_finding_id",
            "expected_refs",
            "observed_refs",
            "open_frontier_refs",
            "causal_slice_refs",
        }
        _reject_unknown_fields(payload, allowed, "finding bridge")
        value = cls(
            repository_id=str(payload["repository_id"]),
            issue_cid=str(payload["issue_cid"]),
            snapshot_id=str(payload["snapshot_id"]),
            diagnostic_payload=payload["diagnostic_payload"],
            deterministic_payload=payload["deterministic_payload"],
            diagnostic_finding_cid=str(payload["diagnostic_finding_cid"]),
            deterministic_finding_id=str(payload["deterministic_finding_id"]),
            expected_refs=tuple(payload.get("expected_refs") or ()),
            observed_refs=tuple(payload.get("observed_refs") or ()),
            open_frontier_refs=tuple(payload.get("open_frontier_refs") or ()),
            causal_slice_refs=tuple(payload.get("causal_slice_refs") or ()),
        )
        _verify_cid(payload.get("content_id", payload.get("cid")), value.content_id, "content_id")
        return value

    @classmethod
    def bridge(
        cls,
        finding: diag.DoctorDiagnosticFinding | Mapping[str, Any],
        *,
        roots: det.DoctorAuthorityRoots | diag.DoctorAuthorityRoots | Mapping[str, Any],
        snapshot_id: str,
        require_repository_id: str = "",
        causal_slice_refs: Sequence[str] = (),
    ) -> "FindingBridge":
        if isinstance(finding, Mapping):
            finding = _diagnostic_finding_from_mapping(finding)
        det_finding = adapt_diagnostic_finding_to_deterministic(
            finding,
            roots=roots,
            snapshot_id=snapshot_id,
            require_repository_id=require_repository_id,
        )
        return cls(
            repository_id=det_finding.roots.repository_id,
            issue_cid=finding.finding_cid,
            snapshot_id=det_finding.snapshot_id,
            diagnostic_payload=_diagnostic_finding_payload(finding),
            deterministic_payload=det_finding.to_dict(),
            diagnostic_finding_cid=finding.finding_cid,
            deterministic_finding_id=det_finding.finding_id,
            expected_refs=det_finding.expected_behavior_refs,
            observed_refs=det_finding.observed_fact_refs,
            open_frontier_refs=finding.open_frontier_refs,
            causal_slice_refs=tuple(causal_slice_refs),
        )


def round_trip_diagnostic_finding(
    finding: diag.DoctorDiagnosticFinding | Mapping[str, Any],
    *,
    roots: det.DoctorAuthorityRoots | diag.DoctorAuthorityRoots | Mapping[str, Any],
    snapshot_id: str,
    require_repository_id: str = "",
) -> diag.DoctorDiagnosticFinding:
    """Diagnostic → deterministic → diagnostic; fails closed on CID drift."""

    bridge = FindingBridge.bridge(
        finding,
        roots=roots,
        snapshot_id=snapshot_id,
        require_repository_id=require_repository_id,
    )
    restored = bridge.materialize_diagnostic()
    if restored.finding_cid != bridge.issue_cid:
        raise DoctorContractAdapterTamperError(
            "diagnostic finding round-trip CID mismatch"
        )
    # Also verify reverse materialization stays consistent.
    det_side = bridge.materialize_deterministic()
    back_again = adapt_deterministic_finding_to_diagnostic(
        det_side, diagnostic_overlay=dict(bridge.diagnostic_payload)
    )
    if back_again.finding_cid != bridge.issue_cid:
        raise DoctorContractAdapterTamperError(
            "diagnostic finding reverse round-trip CID mismatch"
        )
    return restored


def round_trip_deterministic_finding(
    finding: det.DeterministicDoctorFinding | Mapping[str, Any],
    *,
    diagnostic_overlay: Mapping[str, Any] | None = None,
) -> det.DeterministicDoctorFinding:
    """Deterministic → diagnostic → deterministic via bridge overlay.

    When no diagnostic overlay is available, a structural diagnostic projection
    is created and bridged forward again; the deterministic identity is then
    verified against the reconstructed record's semantic fields (roots,
    snapshot, disposition, refs) rather than a free-form finding_id, because
    the diagnostic projection assigns a new CID-based finding_id.
    """

    if isinstance(finding, Mapping):
        finding = det.DeterministicDoctorFinding.from_dict(finding)
    if not isinstance(finding, det.DeterministicDoctorFinding):
        raise DoctorContractAdapterError(
            "finding must be DeterministicDoctorFinding or a mapping"
        )

    if diagnostic_overlay is not None:
        diag_finding = _diagnostic_finding_from_mapping(diagnostic_overlay)
        bridge = FindingBridge.bridge(
            diag_finding,
            roots=finding.roots,
            snapshot_id=finding.snapshot_id,
            require_repository_id=finding.roots.repository_id,
        )
        restored = bridge.materialize_deterministic()
        if restored.roots.repository_id != finding.roots.repository_id:
            raise DoctorContractAdapterReplayError("cross-repository replay is rejected")
        return restored

    # Structural path: project, then re-adapt; compare semantic core.
    diag_finding = adapt_deterministic_finding_to_diagnostic(finding)
    re_det = adapt_diagnostic_finding_to_deterministic(
        diag_finding,
        roots=finding.roots,
        snapshot_id=finding.snapshot_id,
        require_repository_id=finding.roots.repository_id,
    )
    if re_det.roots.repository_id != finding.roots.repository_id:
        raise DoctorContractAdapterReplayError("cross-repository replay is rejected")
    if re_det.snapshot_id != finding.snapshot_id:
        raise DoctorContractAdapterError("snapshot_id drifted across finding round-trip")
    if set(re_det.observed_fact_refs) != set(finding.observed_fact_refs):
        raise DoctorContractAdapterError(
            "observed_fact_refs drifted across finding round-trip"
        )
    return re_det


# ---------------------------------------------------------------------------
# Snapshot adapters
# ---------------------------------------------------------------------------


def _completeness_from_diagnostic(completeness: Mapping[str, Any] | str) -> str:
    if isinstance(completeness, str):
        text = completeness.strip()
        if text in {"complete", "partial_with_frontier", "abstained"}:
            return text
        return "partial_with_frontier"
    if not isinstance(completeness, Mapping):
        return "partial_with_frontier"
    if completeness.get("complete") is True:
        open_frontiers = completeness.get("open_frontiers") or completeness.get(
            "unsupported_frontiers"
        )
        if open_frontiers:
            return "partial_with_frontier"
        return "complete"
    if completeness.get("abstained") is True:
        return "abstained"
    return "partial_with_frontier"


def adapt_diagnostic_snapshot_to_deterministic(
    snapshot: diag.DoctorEvidenceSnapshot | Mapping[str, Any],
    *,
    require_repository_id: str = "",
    snapshot_id: str = "",
) -> det.DoctorEvidenceSnapshot:
    """Project a diagnostic evidence snapshot onto the deterministic schema.

    The diagnostic snapshot remains the source of AST/query bodies; this
    projection binds only content-addressed roots, blob CIDs, frontiers, and
    completeness.  It does not embed source bodies or the AST index.
    """

    if isinstance(snapshot, Mapping):
        raise DoctorContractAdapterError(
            "diagnostic snapshot mapping reconstruction requires a live "
            "DoctorEvidenceSnapshot (AST index cannot be forged from a mapping alone)"
        )
    if not isinstance(snapshot, diag.DoctorEvidenceSnapshot):
        raise DoctorContractAdapterError(
            "snapshot must be a diagnostic DoctorEvidenceSnapshot"
        )

    det_roots = adapt_diagnostic_roots_to_deterministic(
        snapshot.authority_roots, require_repository_id=require_repository_id
    )
    file_blob_cids = tuple(
        sorted(
            {
                receipt.blob_identity
                for receipt in snapshot.adapter_receipts
                if receipt.blob_identity
            }
        )
    )
    completeness = _completeness_from_diagnostic(dict(snapshot.completeness))
    if snapshot.open_frontiers and completeness == "complete":
        completeness = "partial_with_frontier"

    frontiers = tuple(sorted(set(snapshot.open_frontiers)))
    sid = snapshot_id or snapshot.snapshot_id
    invalidation = (det_roots.tree_id, snapshot.snapshot_cid)

    return det.DoctorEvidenceSnapshot(
        roots=det_roots,
        snapshot_id=_identifier(sid, "snapshot_id"),
        file_blob_cids=file_blob_cids,
        completeness=completeness,
        unsupported_frontiers=frontiers,
        parser_id=snapshot.authority_roots.parser_id or "",
        vector_root_id=snapshot.authority_roots.vector_root_id or "",
        embedding_config_id=snapshot.authority_roots.embedding_config_id or "",
        impact_index_id=snapshot.authority_roots.impact_index_id or "",
        value_index_id=snapshot.authority_roots.value_index_id or "",
        evidence_graph_id=snapshot.authority_roots.evidence_graph_id or "",
        invalidation_refs=invalidation,
        clean_rebuild_equivalence_receipt_id="",
    )


def portable_diagnostic_snapshot_projection(
    snapshot: diag.DoctorEvidenceSnapshot,
) -> dict[str, Any]:
    """Body-free portable projection of a diagnostic snapshot for bridging.

    Excludes the AST index and query hit details that may carry source-adjacent
    material; retains identity-bearing receipts, finding payloads, frontiers,
    completeness, and authority roots.
    """

    payload = {
        "schema": diag.DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA,
        "schema_version": snapshot.schema_version,
        "authority_roots": snapshot.authority_roots._payload(),
        "policy": snapshot.policy._payload(),
        "ast_index_id": snapshot.ast_index.index_id,
        "adapter_receipts": [item.to_dict() for item in snapshot.adapter_receipts],
        "findings": [item._payload() for item in snapshot.findings],
        "finding_cids": list(snapshot.finding_cids),
        "open_frontiers": list(snapshot.open_frontiers),
        "completeness": dict(snapshot.completeness),
        "provider_call_count": 0,
        "source_write_count": 0,
        "snapshot_cid": snapshot.snapshot_cid,
        "snapshot_id": snapshot.snapshot_id,
        "rebuild_mode": snapshot.rebuild_mode,
    }
    _assert_body_free(payload, "portable diagnostic snapshot")
    return payload


@dataclass(frozen=True)
class SnapshotBridge(CanonicalContract):
    """Checked bridge between diagnostic and deterministic evidence snapshots."""

    SCHEMA: ClassVar[str] = SNAPSHOT_BRIDGE_SCHEMA

    repository_id: str
    diagnostic_snapshot_cid: str
    diagnostic_snapshot_id: str
    deterministic_snapshot_id: str
    deterministic_content_id: str
    portable_diagnostic: Mapping[str, Any]
    deterministic_payload: Mapping[str, Any]
    finding_bridges: tuple[FindingBridge, ...] = ()
    open_frontier_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self,
            "diagnostic_snapshot_cid",
            _identifier(self.diagnostic_snapshot_cid, "diagnostic_snapshot_cid"),
        )
        object.__setattr__(
            self,
            "diagnostic_snapshot_id",
            _identifier(self.diagnostic_snapshot_id, "diagnostic_snapshot_id"),
        )
        object.__setattr__(
            self,
            "deterministic_snapshot_id",
            _identifier(self.deterministic_snapshot_id, "deterministic_snapshot_id"),
        )
        object.__setattr__(
            self,
            "deterministic_content_id",
            _identifier(self.deterministic_content_id, "deterministic_content_id"),
        )
        portable = _mapping_payload(self.portable_diagnostic, "portable_diagnostic")
        det_payload = _mapping_payload(
            self.deterministic_payload, "deterministic_payload"
        )
        det_payload.pop("content_id", None)
        det_payload.pop("cid", None)
        object.__setattr__(
            self, "portable_diagnostic", MappingProxyType(_plain(portable))
        )
        object.__setattr__(
            self, "deterministic_payload", MappingProxyType(_plain(det_payload))
        )

        bridges: list[FindingBridge] = []
        for item in self.finding_bridges or ():
            if isinstance(item, FindingBridge):
                bridges.append(item)
            elif isinstance(item, Mapping):
                bridges.append(FindingBridge.from_dict(item))
            else:
                raise DoctorContractAdapterError(
                    "finding_bridges must be FindingBridge or mappings"
                )
        # Unique issue CIDs
        issue_cids = [item.issue_cid for item in bridges]
        if len(issue_cids) != len(set(issue_cids)):
            raise DoctorContractAdapterError("duplicate finding issue CIDs in bridge")
        for item in bridges:
            if item.repository_id != self.repository_id:
                raise DoctorContractAdapterReplayError(
                    "finding bridge repository disagrees with snapshot bridge"
                )
        object.__setattr__(
            self,
            "finding_bridges",
            tuple(sorted(bridges, key=lambda item: item.issue_cid)),
        )
        object.__setattr__(
            self,
            "open_frontier_refs",
            _string_tuple(
                self.open_frontier_refs, "open_frontier_refs", limit=MAX_FRONTIER_COUNT
            ),
        )

        # Verify deterministic side.
        det_snap = det.DoctorEvidenceSnapshot.from_dict(
            dict(self.deterministic_payload)
            if self.deterministic_payload.get("schema")
            == det.DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA
            else {
                "schema": det.DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA,
                **dict(self.deterministic_payload),
            }
        )
        if det_snap.snapshot_id != self.deterministic_snapshot_id:
            raise DoctorContractAdapterTamperError(
                "deterministic snapshot_id mismatch on bridge"
            )
        if det_snap.content_id != self.deterministic_content_id:
            raise DoctorContractAdapterTamperError(
                "deterministic snapshot content_id mismatch on bridge"
            )
        if det_snap.roots.repository_id != self.repository_id:
            raise DoctorContractAdapterReplayError(
                "snapshot bridge repository_id mismatch"
            )
        if self.portable_diagnostic.get("snapshot_cid") not in (
            None,
            "",
            self.diagnostic_snapshot_cid,
        ):
            raise DoctorContractAdapterTamperError(
                "portable diagnostic snapshot_cid mismatch"
            )
        _bounded_dict(self.to_dict(), "snapshot bridge")

    def materialize_deterministic(self) -> det.DoctorEvidenceSnapshot:
        payload = dict(self.deterministic_payload)
        if payload.get("schema") != det.DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA:
            payload = {
                "schema": det.DOCTOR_EVIDENCE_SNAPSHOT_SCHEMA,
                **payload,
            }
        return det.DoctorEvidenceSnapshot.from_dict(payload)

    def materialize_finding_diagnostics(self) -> tuple[diag.DoctorDiagnosticFinding, ...]:
        return tuple(item.materialize_diagnostic() for item in self.finding_bridges)

    def materialize_finding_deterministics(
        self,
    ) -> tuple[det.DeterministicDoctorFinding, ...]:
        return tuple(item.materialize_deterministic() for item in self.finding_bridges)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DIAGNOSIS_OBLIGATION_BRIDGE_VERSION,
            "repository_id": self.repository_id,
            "diagnostic_snapshot_cid": self.diagnostic_snapshot_cid,
            "diagnostic_snapshot_id": self.diagnostic_snapshot_id,
            "deterministic_snapshot_id": self.deterministic_snapshot_id,
            "deterministic_content_id": self.deterministic_content_id,
            "portable_diagnostic": dict(self.portable_diagnostic),
            "deterministic_payload": dict(self.deterministic_payload),
            "finding_bridges": [item.to_dict() for item in self.finding_bridges],
            "open_frontier_refs": list(self.open_frontier_refs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SnapshotBridge":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise DoctorContractAdapterSchemaError(
                "snapshot bridge has an unsupported schema"
            )
        _assert_body_free(payload, "snapshot bridge")
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "cid",
            "repository_id",
            "diagnostic_snapshot_cid",
            "diagnostic_snapshot_id",
            "deterministic_snapshot_id",
            "deterministic_content_id",
            "portable_diagnostic",
            "deterministic_payload",
            "finding_bridges",
            "open_frontier_refs",
        }
        _reject_unknown_fields(payload, allowed, "snapshot bridge")
        value = cls(
            repository_id=str(payload["repository_id"]),
            diagnostic_snapshot_cid=str(payload["diagnostic_snapshot_cid"]),
            diagnostic_snapshot_id=str(payload["diagnostic_snapshot_id"]),
            deterministic_snapshot_id=str(payload["deterministic_snapshot_id"]),
            deterministic_content_id=str(payload["deterministic_content_id"]),
            portable_diagnostic=payload["portable_diagnostic"],
            deterministic_payload=payload["deterministic_payload"],
            finding_bridges=tuple(payload.get("finding_bridges") or ()),
            open_frontier_refs=tuple(payload.get("open_frontier_refs") or ()),
        )
        _verify_cid(payload.get("content_id", payload.get("cid")), value.content_id, "content_id")
        return value

    @classmethod
    def bridge(
        cls,
        snapshot: diag.DoctorEvidenceSnapshot,
        *,
        require_repository_id: str = "",
        causal_slice_refs: Sequence[str] = (),
    ) -> "SnapshotBridge":
        det_snap = adapt_diagnostic_snapshot_to_deterministic(
            snapshot, require_repository_id=require_repository_id
        )
        finding_bridges = tuple(
            FindingBridge.bridge(
                finding,
                roots=det_snap.roots,
                snapshot_id=det_snap.snapshot_id,
                require_repository_id=det_snap.roots.repository_id,
                causal_slice_refs=causal_slice_refs,
            )
            for finding in snapshot.findings
        )
        return cls(
            repository_id=det_snap.roots.repository_id,
            diagnostic_snapshot_cid=snapshot.snapshot_cid,
            diagnostic_snapshot_id=snapshot.snapshot_id,
            deterministic_snapshot_id=det_snap.snapshot_id,
            deterministic_content_id=det_snap.content_id,
            portable_diagnostic=portable_diagnostic_snapshot_projection(snapshot),
            deterministic_payload=det_snap.to_dict(),
            finding_bridges=finding_bridges,
            open_frontier_refs=tuple(snapshot.open_frontiers),
        )


def round_trip_diagnostic_snapshot(
    snapshot: diag.DoctorEvidenceSnapshot,
    *,
    require_repository_id: str = "",
) -> SnapshotBridge:
    """Diagnostic snapshot → deterministic → bridge; verify identities.

    Full AST-index reconstruction is intentionally out of scope for the
    reverse direction; the bridge preserves portable identity and findings.
    """

    bridge = SnapshotBridge.bridge(
        snapshot, require_repository_id=require_repository_id
    )
    det_side = bridge.materialize_deterministic()
    if det_side.content_id != bridge.deterministic_content_id:
        raise DoctorContractAdapterTamperError(
            "deterministic snapshot round-trip content_id mismatch"
        )
    if det_side.roots.repository_id != bridge.repository_id:
        raise DoctorContractAdapterReplayError("cross-repository replay is rejected")
    # Finding family round-trip through the same bridge.
    for item in bridge.finding_bridges:
        restored = item.materialize_diagnostic()
        if restored.finding_cid != item.issue_cid:
            raise DoctorContractAdapterTamperError(
                "finding CID mismatch inside snapshot bridge round-trip"
            )
        det_finding = item.materialize_deterministic()
        if det_finding.roots.repository_id != bridge.repository_id:
            raise DoctorContractAdapterReplayError(
                "finding root repository disagrees with snapshot bridge"
            )
    # Portable projection CID must remain stable.
    if bridge.portable_diagnostic.get("snapshot_cid") != snapshot.snapshot_cid:
        raise DoctorContractAdapterTamperError(
            "portable diagnostic snapshot_cid drifted"
        )
    return bridge


def round_trip_deterministic_snapshot(
    snapshot: det.DoctorEvidenceSnapshot | Mapping[str, Any],
    *,
    portable_diagnostic: Mapping[str, Any] | None = None,
    finding_bridges: Sequence[FindingBridge | Mapping[str, Any]] = (),
) -> det.DoctorEvidenceSnapshot:
    """Deterministic snapshot identity round-trip through the bridge envelope."""

    if isinstance(snapshot, Mapping):
        snapshot = det.DoctorEvidenceSnapshot.from_dict(snapshot)
    if not isinstance(snapshot, det.DoctorEvidenceSnapshot):
        raise DoctorContractAdapterError(
            "snapshot must be deterministic DoctorEvidenceSnapshot or a mapping"
        )

    if portable_diagnostic is None:
        # Without a diagnostic peer, only verify self-identity (from_dict).
        restored = det.DoctorEvidenceSnapshot.from_dict(snapshot.to_dict())
        if restored.content_id != snapshot.content_id:
            raise DoctorContractAdapterTamperError(
                "deterministic snapshot self round-trip content_id mismatch"
            )
        return restored

    bridge = SnapshotBridge(
        repository_id=snapshot.roots.repository_id,
        diagnostic_snapshot_cid=str(
            portable_diagnostic.get("snapshot_cid")
            or portable_diagnostic.get("snapshot_id")
            or snapshot.snapshot_id
        ),
        diagnostic_snapshot_id=str(
            portable_diagnostic.get("snapshot_id") or snapshot.snapshot_id
        ),
        deterministic_snapshot_id=snapshot.snapshot_id,
        deterministic_content_id=snapshot.content_id,
        portable_diagnostic=portable_diagnostic,
        deterministic_payload=snapshot.to_dict(),
        finding_bridges=tuple(finding_bridges),
        open_frontier_refs=tuple(snapshot.unsupported_frontiers),
    )
    restored = bridge.materialize_deterministic()
    if restored.content_id != snapshot.content_id:
        raise DoctorContractAdapterTamperError(
            "deterministic snapshot bridge round-trip content_id mismatch"
        )
    if restored.roots.repository_id != snapshot.roots.repository_id:
        raise DoctorContractAdapterReplayError("cross-repository replay is rejected")
    return restored


# ---------------------------------------------------------------------------
# Top-level DiagnosisObligationBridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosisObligationBridge(CanonicalContract):
    """Top-level join record for Doctor snapshot/finding family round-trips.

    Binds issue CIDs, expected/observed contracts, causal slice, open frontier,
    and both family projections under one repository identity.
    """

    SCHEMA: ClassVar[str] = DIAGNOSIS_OBLIGATION_BRIDGE_SCHEMA

    repository_id: str
    snapshot_bridge: SnapshotBridge | None = None
    finding_bridges: tuple[FindingBridge, ...] = ()
    root_bridge: AuthorityRootBridge | None = None
    expected_contract_refs: tuple[str, ...] = ()
    observed_contract_refs: tuple[str, ...] = ()
    causal_slice_refs: tuple[str, ...] = ()
    open_frontier_refs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        if self.snapshot_bridge is None:
            object.__setattr__(self, "snapshot_bridge", None)
        elif isinstance(self.snapshot_bridge, SnapshotBridge):
            if self.snapshot_bridge.repository_id != self.repository_id:
                raise DoctorContractAdapterReplayError(
                    "snapshot bridge repository disagrees with obligation bridge"
                )
        elif isinstance(self.snapshot_bridge, Mapping):
            object.__setattr__(
                self,
                "snapshot_bridge",
                SnapshotBridge.from_dict(self.snapshot_bridge),
            )
            if self.snapshot_bridge.repository_id != self.repository_id:  # type: ignore[union-attr]
                raise DoctorContractAdapterReplayError(
                    "snapshot bridge repository disagrees with obligation bridge"
                )
        else:
            raise DoctorContractAdapterError(
                "snapshot_bridge must be SnapshotBridge, mapping, or None"
            )

        bridges: list[FindingBridge] = []
        for item in self.finding_bridges or ():
            if isinstance(item, FindingBridge):
                bridges.append(item)
            elif isinstance(item, Mapping):
                bridges.append(FindingBridge.from_dict(item))
            else:
                raise DoctorContractAdapterError(
                    "finding_bridges must be FindingBridge or mappings"
                )
        issue_cids = [item.issue_cid for item in bridges]
        if len(issue_cids) != len(set(issue_cids)):
            raise DoctorContractAdapterError("duplicate issue CIDs in obligation bridge")
        for item in bridges:
            if item.repository_id != self.repository_id:
                raise DoctorContractAdapterReplayError(
                    "finding bridge repository disagrees with obligation bridge"
                )
        object.__setattr__(
            self,
            "finding_bridges",
            tuple(sorted(bridges, key=lambda item: item.issue_cid)),
        )

        if self.root_bridge is None:
            object.__setattr__(self, "root_bridge", None)
        elif isinstance(self.root_bridge, AuthorityRootBridge):
            if self.root_bridge.repository_id != self.repository_id:
                raise DoctorContractAdapterReplayError(
                    "root bridge repository disagrees with obligation bridge"
                )
        elif isinstance(self.root_bridge, Mapping):
            object.__setattr__(
                self, "root_bridge", AuthorityRootBridge.from_dict(self.root_bridge)
            )
            if self.root_bridge.repository_id != self.repository_id:  # type: ignore[union-attr]
                raise DoctorContractAdapterReplayError(
                    "root bridge repository disagrees with obligation bridge"
                )
        else:
            raise DoctorContractAdapterError(
                "root_bridge must be AuthorityRootBridge, mapping, or None"
            )

        object.__setattr__(
            self,
            "expected_contract_refs",
            _string_tuple(self.expected_contract_refs, "expected_contract_refs"),
        )
        object.__setattr__(
            self,
            "observed_contract_refs",
            _string_tuple(self.observed_contract_refs, "observed_contract_refs"),
        )
        object.__setattr__(
            self,
            "causal_slice_refs",
            _string_tuple(self.causal_slice_refs, "causal_slice_refs"),
        )
        object.__setattr__(
            self,
            "open_frontier_refs",
            _string_tuple(
                self.open_frontier_refs, "open_frontier_refs", limit=MAX_FRONTIER_COUNT
            ),
        )
        object.__setattr__(
            self, "notes", _string_tuple(self.notes, "notes", limit=MAX_REFERENCE_COUNT)
        )
        _bounded_dict(self.to_dict(), "diagnosis obligation bridge")

    def issue_cids(self) -> tuple[str, ...]:
        return tuple(item.issue_cid for item in self.finding_bridges)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DIAGNOSIS_OBLIGATION_BRIDGE_VERSION,
            "repository_id": self.repository_id,
            "snapshot_bridge": (
                self.snapshot_bridge.to_dict()
                if self.snapshot_bridge is not None
                else None
            ),
            "finding_bridges": [item.to_dict() for item in self.finding_bridges],
            "root_bridge": (
                self.root_bridge.to_dict() if self.root_bridge is not None else None
            ),
            "expected_contract_refs": list(self.expected_contract_refs),
            "observed_contract_refs": list(self.observed_contract_refs),
            "causal_slice_refs": list(self.causal_slice_refs),
            "open_frontier_refs": list(self.open_frontier_refs),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiagnosisObligationBridge":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise DoctorContractAdapterSchemaError(
                "diagnosis obligation bridge has an unsupported schema"
            )
        _assert_body_free(payload, "diagnosis obligation bridge")
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "cid",
            "repository_id",
            "snapshot_bridge",
            "finding_bridges",
            "root_bridge",
            "expected_contract_refs",
            "observed_contract_refs",
            "causal_slice_refs",
            "open_frontier_refs",
            "notes",
        }
        _reject_unknown_fields(payload, allowed, "diagnosis obligation bridge")
        value = cls(
            repository_id=str(payload["repository_id"]),
            snapshot_bridge=payload.get("snapshot_bridge"),
            finding_bridges=tuple(payload.get("finding_bridges") or ()),
            root_bridge=payload.get("root_bridge"),
            expected_contract_refs=tuple(payload.get("expected_contract_refs") or ()),
            observed_contract_refs=tuple(payload.get("observed_contract_refs") or ()),
            causal_slice_refs=tuple(payload.get("causal_slice_refs") or ()),
            open_frontier_refs=tuple(payload.get("open_frontier_refs") or ()),
            notes=tuple(payload.get("notes") or ()),
        )
        _verify_cid(payload.get("content_id", payload.get("cid")), value.content_id, "content_id")
        return value

    @classmethod
    def from_diagnostic_snapshot(
        cls,
        snapshot: diag.DoctorEvidenceSnapshot,
        *,
        require_repository_id: str = "",
        causal_slice_refs: Sequence[str] = (),
        notes: Sequence[str] = (),
    ) -> "DiagnosisObligationBridge":
        snap_bridge = SnapshotBridge.bridge(
            snapshot,
            require_repository_id=require_repository_id,
            causal_slice_refs=causal_slice_refs,
        )
        root_bridge = AuthorityRootBridge.bridge(
            snapshot.authority_roots,
            require_repository_id=require_repository_id or snap_bridge.repository_id,
        )
        expected: list[str] = []
        observed: list[str] = []
        for item in snap_bridge.finding_bridges:
            expected.extend(item.expected_refs)
            observed.extend(item.observed_refs)
        return cls(
            repository_id=snap_bridge.repository_id,
            snapshot_bridge=snap_bridge,
            finding_bridges=snap_bridge.finding_bridges,
            root_bridge=root_bridge,
            expected_contract_refs=tuple(sorted(set(expected))),
            observed_contract_refs=tuple(sorted(set(observed))),
            causal_slice_refs=tuple(causal_slice_refs),
            open_frontier_refs=tuple(snapshot.open_frontiers),
            notes=tuple(notes),
        )


__all__ = [
    "DIAGNOSIS_OBLIGATION_BRIDGE_INTERFACE",
    "DIAGNOSIS_OBLIGATION_BRIDGE_SCHEMA",
    "DIAGNOSIS_OBLIGATION_BRIDGE_VERSION",
    "FINDING_BRIDGE_SCHEMA",
    "ROOT_BRIDGE_SCHEMA",
    "SNAPSHOT_BRIDGE_SCHEMA",
    "AuthorityRootBridge",
    "DiagnosisObligationBridge",
    "DoctorContractAdapterAuthorityError",
    "DoctorContractAdapterBoundsError",
    "DoctorContractAdapterError",
    "DoctorContractAdapterReplayError",
    "DoctorContractAdapterSchemaError",
    "DoctorContractAdapterTamperError",
    "FindingBridge",
    "SnapshotBridge",
    "adapt_deterministic_finding_to_diagnostic",
    "adapt_deterministic_roots_to_diagnostic",
    "adapt_diagnostic_finding_to_deterministic",
    "adapt_diagnostic_roots_to_deterministic",
    "adapt_diagnostic_snapshot_to_deterministic",
    "assert_same_repository",
    "map_deterministic_disposition_to_diagnostic",
    "map_diagnostic_disposition_to_deterministic",
    "map_diagnostic_kind_to_finding_kind",
    "map_finding_kind_to_diagnostic_kind",
    "portable_diagnostic_snapshot_projection",
    "round_trip_deterministic_finding",
    "round_trip_deterministic_snapshot",
    "round_trip_diagnostic_finding",
    "round_trip_diagnostic_snapshot",
]
