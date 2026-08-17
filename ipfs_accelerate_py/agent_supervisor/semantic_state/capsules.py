"""Admit datasets-owned semantic capsules without restating their facts.

Accelerate verifies, binds, caches, and projects producer capsules. Authoritative
AST facts, signatures, relations, and confidence remain solely in the datasets
``SemanticCapsule`` identified by its producer CID. This module never recompiles
semantics, never raises confidence, and never treats docstrings or model
summaries as proof or exact facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    SemanticCapsuleRef,
    _bool,
    _text,
    _unique_sorted_cids,
    _unique_sorted_texts,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload

# ---------------------------------------------------------------------------
# Interfaces / closed vocabularies
# ---------------------------------------------------------------------------

CAPSULE_ADMISSION_INTERFACE = "SemanticCapsuleAdmission@1"
CAPSULE_ADMISSION_SCHEMA = "ipfs-accelerate.semantic-capsule-admission@1"
CAPSULE_CACHE_INDEX_SCHEMA = "ipfs-accelerate.semantic-capsule-cache-index@1"

CONFIDENCE_VALUES = frozenset({"exact", "conservative", "heuristic", "opaque"})
SUBSTITUTABLE_CONFIDENCE = frozenset({"exact", "conservative"})
NON_SUBSTITUTABLE_CONFIDENCE = frozenset({"heuristic", "opaque"})

ADMISSION_EXACT = "exact_substitute"
ADMISSION_CONSERVATIVE = "conservative_substitute_with_caveats"
ADMISSION_RAW = "raw_source_required"
ADMISSION_DECISIONS = frozenset(
    {ADMISSION_EXACT, ADMISSION_CONSERVATIVE, ADMISSION_RAW}
)

FRESHNESS_FRESH = "fresh"
FRESHNESS_STALE = "stale"
FRESHNESS_UNKNOWN = "unknown"
FRESHNESS_VALUES = frozenset({FRESHNESS_FRESH, FRESHNESS_STALE, FRESHNESS_UNKNOWN})

# Metadata keys that may hold non-authoritative model/docstring material.
_HEURISTIC_HINT_KEYS = frozenset(
    {
        "llm_summary",
        "llm_summaries",
        "heuristic_summary",
        "model_summary",
        "summary",
        "ai_summary",
        "generated_summary",
        "heuristic_annotations",
        "docstring",
        "docstring_hint",
    }
)


class CapsuleAdmissionError(HarnessError):
    """Closed admission or capsule-cache contract violation."""


@runtime_checkable
class _DurablePort(Protocol):
    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]: ...

    def get(self, cid: str) -> Mapping[str, Any]: ...

    def has(self, cid: str) -> bool: ...


# ---------------------------------------------------------------------------
# Field extraction helpers (identity only; no fact restatement)
# ---------------------------------------------------------------------------


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _confidence_of(capsule: Any) -> str:
    raw = _attr(capsule, "confidence")
    if raw is None:
        raise CapsuleAdmissionError("capsule confidence is required")
    value = str(getattr(raw, "value", raw)).strip()
    if value not in CONFIDENCE_VALUES:
        raise CapsuleAdmissionError(
            f"capsule confidence must be one of {sorted(CONFIDENCE_VALUES)}"
        )
    return value


def _optional_cid_field(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    return validate_opaque_cid(value, name)


def _cid_field(value: Any, name: str) -> str:
    if value is None or value == "":
        raise CapsuleAdmissionError(f"{name} is required on the datasets capsule")
    return validate_opaque_cid(value, name)


def _freshness_value(value: Any) -> str:
    text = str(getattr(value, "value", value)).strip()
    if text not in FRESHNESS_VALUES:
        raise CapsuleAdmissionError(
            f"freshness must be one of {sorted(FRESHNESS_VALUES)}"
        )
    return text


def _admission_value(value: Any) -> str:
    text = str(getattr(value, "value", value)).strip()
    if text not in ADMISSION_DECISIONS:
        raise CapsuleAdmissionError(
            f"admission must be one of {sorted(ADMISSION_DECISIONS)}"
        )
    return text


def _has_heuristic_hints(capsule: Any) -> bool:
    metadata = _attr(capsule, "metadata") or {}
    if isinstance(metadata, Mapping):
        for key in metadata:
            if str(key) in _HEURISTIC_HINT_KEYS:
                return True
        annotations = metadata.get("heuristic_annotations")
        if annotations:
            return True
    for key in (
        "docstring_hint",
        "heuristic_annotations",
        "model_summary",
        "llm_summary",
    ):
        if _attr(capsule, key):
            return True
    return False


def _claimed_raised_confidence(capsule: Any) -> str | None:
    """Detect attempts to promote confidence via non-authoritative hints."""

    for container_name in ("metadata", "heuristic_annotations"):
        container = _attr(capsule, container_name)
        if not isinstance(container, Mapping):
            continue
        for key in (
            "raised_confidence",
            "promoted_confidence",
            "effective_confidence",
            "confidence_override",
        ):
            claimed = container.get(key)
            if claimed is None:
                continue
            claimed_text = str(getattr(claimed, "value", claimed)).strip()
            if claimed_text in CONFIDENCE_VALUES:
                return claimed_text
    return None


# ---------------------------------------------------------------------------
# Admission record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapsuleAdmission:
    """Admission-only projection of one datasets capsule.

    The ``ref`` carries only opaque CIDs and confidence/raw-source flags.
    Authoritative capsule facts remain addressable solely via ``ref.capsule_cid``.
    """

    ref: SemanticCapsuleRef
    admission: str
    freshness: str
    caveats: tuple[str, ...]
    assessment_cid: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, SemanticCapsuleRef):
            raise CapsuleAdmissionError("ref must be a SemanticCapsuleRef")
        object.__setattr__(self, "admission", _admission_value(self.admission))
        object.__setattr__(self, "freshness", _freshness_value(self.freshness))
        caveats = _unique_sorted_texts(list(self.caveats), "caveats")
        object.__setattr__(self, "caveats", caveats)
        if self.assessment_cid is not None:
            object.__setattr__(
                self,
                "assessment_cid",
                validate_opaque_cid(self.assessment_cid, "assessment_cid"),
            )
        # Substitution may never silence raw-source requirements.
        if self.admission == ADMISSION_RAW and not self.ref.raw_source_required:
            raise CapsuleAdmissionError(
                "raw_source_required admission requires ref.raw_source_required"
            )
        if (
            self.admission in {ADMISSION_EXACT, ADMISSION_CONSERVATIVE}
            and self.ref.raw_source_required
        ):
            raise CapsuleAdmissionError(
                "substitutable admission cannot require raw source"
            )
        if self.admission == ADMISSION_CONSERVATIVE and not any(
            "conservative" in item for item in self.caveats
        ):
            raise CapsuleAdmissionError(
                "conservative substitution requires a visible conservative caveat"
            )
        if self.ref.confidence not in CONFIDENCE_VALUES:
            raise CapsuleAdmissionError("ref.confidence is not a closed confidence value")
        if (
            self.admission == ADMISSION_EXACT
            and self.ref.confidence != "exact"
        ):
            raise CapsuleAdmissionError("exact_substitute requires exact confidence")
        if (
            self.admission == ADMISSION_CONSERVATIVE
            and self.ref.confidence != "conservative"
        ):
            raise CapsuleAdmissionError(
                "conservative_substitute_with_caveats requires conservative confidence"
            )

    @property
    def may_substitute(self) -> bool:
        return self.admission in {ADMISSION_EXACT, ADMISSION_CONSERVATIVE}

    @property
    def requires_raw_source(self) -> bool:
        return self.admission == ADMISSION_RAW or self.ref.raw_source_required

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAPSULE_ADMISSION_SCHEMA,
            "interface": CAPSULE_ADMISSION_INTERFACE,
            "ref": self.ref.to_dict(),
            "admission": self.admission,
            "freshness": self.freshness,
            "caveats": list(self.caveats),
            "assessment_cid": self.assessment_cid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapsuleAdmission":
        if not isinstance(data, Mapping):
            raise CapsuleAdmissionError("CapsuleAdmission must be an object")
        allowed = frozenset(
            {
                "schema",
                "interface",
                "ref",
                "admission",
                "freshness",
                "caveats",
                "assessment_cid",
            }
        )
        unknown = set(data) - allowed
        if unknown:
            raise CapsuleAdmissionError(
                f"CapsuleAdmission contains unsupported fields: {sorted(unknown)}"
            )
        schema = data.get("schema")
        if schema is not None and schema != CAPSULE_ADMISSION_SCHEMA:
            raise CapsuleAdmissionError(
                f"unsupported CapsuleAdmission schema {schema!r}"
            )
        interface = data.get("interface")
        if interface is not None and interface != CAPSULE_ADMISSION_INTERFACE:
            raise CapsuleAdmissionError(
                f"unsupported CapsuleAdmission interface {interface!r}"
            )
        ref_raw = data.get("ref")
        if not isinstance(ref_raw, Mapping):
            raise CapsuleAdmissionError("ref must be an object")
        ref = SemanticCapsuleRef.from_dict(ref_raw)
        caveats = data.get("caveats", [])
        if not isinstance(caveats, list):
            raise CapsuleAdmissionError("caveats must be a list")
        return cls(
            ref=ref,
            admission=_text(data.get("admission"), "admission"),
            freshness=_text(data.get("freshness"), "freshness"),
            caveats=tuple(str(item) for item in caveats),
            assessment_cid=data.get("assessment_cid"),
        )

    def content_cid(self) -> str:
        """Return the Kubo CIDv1 of the closed admission artifact."""

        return cid_for_payload(self.to_dict())


# ---------------------------------------------------------------------------
# Admission decision
# ---------------------------------------------------------------------------


def _freshness_from_assessment(assessment: Any) -> tuple[str, str, tuple[str, ...], str | None]:
    """Return (freshness, admission, caveats, assessment_cid) from producer assessment."""

    if assessment is None:
        return FRESHNESS_UNKNOWN, ADMISSION_RAW, ("freshness:unknown",), None

    freshness = _freshness_value(
        _attr(assessment, "freshness", default=FRESHNESS_UNKNOWN)
    )
    admission_raw = _attr(assessment, "admission")
    if admission_raw is not None:
        admission = _admission_value(admission_raw)
    else:
        admission = ADMISSION_RAW

    caveats_raw = _attr(assessment, "caveats", default=()) or ()
    if isinstance(caveats_raw, str):
        caveats_list = [caveats_raw]
    else:
        caveats_list = [str(item) for item in caveats_raw]

    assessment_cid = _optional_cid_field(
        _attr(assessment, "assessment_cid"), "assessment_cid"
    )
    return freshness, admission, tuple(caveats_list), assessment_cid


def admit_capsule(
    capsule: Any,
    *,
    semantic_state_root_cid: str,
    validity_bindings: Any = (),
    freshness: Any = None,
    assessment: Any = None,
    force_raw_source: bool = False,
) -> CapsuleAdmission:
    """Admit one datasets capsule as a harness reference.

    Parameters
    ----------
    capsule:
        Datasets ``SemanticCapsule`` (object or mapping). Only identity fields
        are read; facts are never copied into the admission record.
    semantic_state_root_cid:
        Producing semantic-state root CID that binds the capsule.
    validity_bindings:
        Sorted unique CIDs for dependency/policy/interface/configuration pins
        that must match for reuse.
    freshness / assessment:
        Optional datasets ``CapsuleFreshness`` (or equivalent). When present its
        admission decision is honored and never relaxed by this harness.
    force_raw_source:
        Caller-side obligation that forces raw tree-bound source retrieval.
    """

    if capsule is None:
        raise CapsuleAdmissionError("capsule is required")

    root_cid = validate_opaque_cid(semantic_state_root_cid, "semantic_state_root_cid")
    bindings = _unique_sorted_cids(list(validity_bindings or ()), "validity_bindings")

    capsule_cid = _cid_field(_attr(capsule, "capsule_cid"), "capsule_cid")
    stable_symbol_id = _text(
        _attr(capsule, "stable_symbol_id"), "stable_symbol_id"
    )
    version_cid = _cid_field(_attr(capsule, "version_cid"), "version_cid")
    source_cid_raw = _attr(capsule, "source_cid")
    if source_cid_raw is None or source_cid_raw == "":
        # Missing source identity forces raw retrieval with an opaque placeholder
        # rejected — require an explicit source CID for the admission ref.
        raise CapsuleAdmissionError(
            "capsule source_cid is required for admission and tree-bound retrieval"
        )
    source_cid = validate_opaque_cid(source_cid_raw, "source_cid")
    confidence = _confidence_of(capsule)

    # Model summaries / docstring hints can never raise confidence.
    claimed = _claimed_raised_confidence(capsule)
    if claimed is not None:
        rank = {"exact": 0, "conservative": 1, "heuristic": 2, "opaque": 3}
        if rank.get(claimed, 3) < rank.get(confidence, 3):
            raise CapsuleAdmissionError(
                "heuristic or model summary cannot raise capsule confidence"
            )

    assessment_obj = assessment if assessment is not None else freshness
    caveats: list[str] = []

    if assessment_obj is not None:
        fr, adm, assess_caveats, assessment_cid = _freshness_from_assessment(
            assessment_obj
        )
        caveats.extend(assess_caveats)
        freshness_value = fr
        admission = adm
    else:
        assessment_cid = None
        freshness_value = FRESHNESS_FRESH
        if confidence == "exact":
            admission = ADMISSION_EXACT
        elif confidence == "conservative":
            admission = ADMISSION_CONSERVATIVE
            caveats.append("confidence:conservative")
        else:
            admission = ADMISSION_RAW
            caveats.append(f"unsafe_confidence:{confidence}")

    # Local non-relaxable gates (may only force raw source, never promote).
    if force_raw_source:
        admission = ADMISSION_RAW
        caveats.append("force_raw_source")
    if confidence in NON_SUBSTITUTABLE_CONFIDENCE:
        admission = ADMISSION_RAW
        marker = f"unsafe_confidence:{confidence}"
        if marker not in caveats:
            caveats.append(marker)
    if freshness_value in {FRESHNESS_STALE, FRESHNESS_UNKNOWN}:
        admission = ADMISSION_RAW
        marker = f"freshness:{freshness_value}"
        if marker not in caveats:
            caveats.append(marker)
    if _has_heuristic_hints(capsule) and admission != ADMISSION_RAW:
        # Hints are allowed on substitutable capsules but must remain visible.
        caveats.append("heuristic_hints_non_authoritative")

    # Never promote producer decisions; may only force raw or keep producer level.
    if assessment_obj is not None:
        producer_admission = _admission_value(
            _attr(assessment_obj, "admission", default=ADMISSION_RAW)
        )
        if producer_admission == ADMISSION_RAW:
            admission = ADMISSION_RAW
        elif (
            producer_admission == ADMISSION_CONSERVATIVE
            and admission == ADMISSION_EXACT
            and confidence == "conservative"
        ):
            admission = ADMISSION_CONSERVATIVE
            if not any("conservative" in item for item in caveats):
                caveats.append("confidence:conservative")
        elif (
            producer_admission == ADMISSION_CONSERVATIVE
            and admission == ADMISSION_EXACT
            and confidence == "exact"
        ):
            # Exact confidence stays exact_substitute; visible producer caveats remain.
            admission = ADMISSION_EXACT

    # Align admission with confidence when substitution is claimed.
    if admission == ADMISSION_EXACT and confidence != "exact":
        admission = ADMISSION_RAW
        caveats.append(f"confidence_admission_mismatch:{confidence}")
    if admission == ADMISSION_CONSERVATIVE and confidence != "conservative":
        admission = ADMISSION_RAW
        caveats.append(f"confidence_admission_mismatch:{confidence}")

    if admission == ADMISSION_CONSERVATIVE and not any(
        "conservative" in item for item in caveats
    ):
        caveats.append("confidence:conservative")

    raw_source_required = admission == ADMISSION_RAW
    # Opaque/heuristic always require exact scanned-tree source.
    if confidence in NON_SUBSTITUTABLE_CONFIDENCE:
        raw_source_required = True
        admission = ADMISSION_RAW

    ref = SemanticCapsuleRef(
        capsule_cid=capsule_cid,
        semantic_state_root_cid=root_cid,
        stable_symbol_id=stable_symbol_id,
        version_cid=version_cid,
        source_cid=source_cid,
        confidence=confidence,
        validity_bindings=bindings,
        raw_source_required=raw_source_required,
    )
    return CapsuleAdmission(
        ref=ref,
        admission=admission,
        freshness=freshness_value,
        caveats=tuple(sorted(set(caveats))),
        assessment_cid=assessment_cid,
    )


def retrieve_opaque_source(
    provider: Any,
    semantic_index: Any,
    symbol_id: str,
    *,
    expected_producer_state_cid: str,
    read_source_blob: Any = None,
) -> Any:
    """Retrieve exact tree-bound source for opaque/stale/invalid capsules.

    Delegates to the sealed datasets provider. Ambient post-scan filesystem
    reads are forbidden by the provider adapter.
    """

    if provider is None:
        raise CapsuleAdmissionError("semantic-state provider is required")
    reader = getattr(provider, "read_required_source", None)
    if not callable(reader):
        raise CapsuleAdmissionError(
            "provider must expose read_required_source for tree-bound retrieval"
        )
    validate_opaque_cid(expected_producer_state_cid, "expected_producer_state_cid")
    symbol = _text(symbol_id, "symbol_id")
    if read_source_blob is None:
        return reader(
            semantic_index,
            symbol,
            expected_producer_state_cid=expected_producer_state_cid,
        )
    return reader(
        semantic_index,
        symbol,
        expected_producer_state_cid=expected_producer_state_cid,
        read_source_blob=read_source_blob,
    )


# ---------------------------------------------------------------------------
# Capsule cache (index over DurableSemanticStatePort only)
# ---------------------------------------------------------------------------


@dataclass
class CapsuleCache:
    """Admission/capsule index over an injected durable port.

    This is not a second block, filesystem, or CID authority. Capsule fact
    bytes remain producer-owned; the cache only stores closed admission
    records and optional already-CID'd capsule envelopes the caller verified.
    """

    port: _DurablePort

    def __post_init__(self) -> None:
        if not isinstance(self.port, _DurablePort) and not all(
            callable(getattr(self.port, name, None)) for name in ("put", "get", "has")
        ):
            raise CapsuleAdmissionError(
                "CapsuleCache requires a DurableSemanticStatePort-compatible port"
            )

    def store_admission(self, admission: CapsuleAdmission) -> str:
        if not isinstance(admission, CapsuleAdmission):
            raise CapsuleAdmissionError("admission must be a CapsuleAdmission")
        artifact = admission.to_dict()
        expected = admission.content_cid()
        self.port.put(artifact, expected_cid=expected, codec="dag-json")
        return expected

    def get_admission(self, admission_cid: str) -> CapsuleAdmission:
        cid = validate_opaque_cid(admission_cid, "admission_cid")
        if not self.port.has(cid):
            raise CapsuleAdmissionError(f"admission not found: {cid}")
        payload = self.port.get(cid)
        return CapsuleAdmission.from_dict(payload)

    def store_capsule_envelope(
        self,
        *,
        capsule_cid: str,
        envelope: Mapping[str, Any],
    ) -> str:
        """Store a caller-verified capsule envelope under its producer CID.

        The envelope must already claim ``capsule_cid``; this method never
        recomputes capsule identity from restated facts.
        """

        expected = validate_opaque_cid(capsule_cid, "capsule_cid")
        if not isinstance(envelope, Mapping):
            raise CapsuleAdmissionError("capsule envelope must be an object")
        claimed = envelope.get("capsule_cid")
        if claimed is not None and claimed != expected:
            raise CapsuleAdmissionError(
                "envelope capsule_cid does not match the producer CID"
            )
        # Ensure the stored artifact binds the producer CID without inventing facts.
        body = dict(envelope)
        body["capsule_cid"] = expected
        # Durable port verifies expected_cid against its codec; callers that
        # need byte-identity must supply an envelope whose port-native CID
        # equals the producer capsule_cid. When the port uses a different
        # codec, store under a harness wrapper keyed by producer CID.
        wrapper = {
            "schema": CAPSULE_CACHE_INDEX_SCHEMA,
            "capsule_cid": expected,
            "envelope": body,
        }
        # Use harness CID for the wrapper; producer CID is the authority key.
        wrapper_cid = cid_for_payload(wrapper)
        self.port.put(wrapper, expected_cid=wrapper_cid, codec="dag-json")
        # Also store a pointer artifact keyed for lookup by producer CID when
        # the port can accept the producer CID as expected (best effort).
        pointer = {
            "schema": CAPSULE_CACHE_INDEX_SCHEMA + "/pointer",
            "capsule_cid": expected,
            "wrapper_cid": wrapper_cid,
        }
        pointer_cid = cid_for_payload(pointer)
        self.port.put(pointer, expected_cid=pointer_cid, codec="dag-json")
        return wrapper_cid

    def get_capsule_envelope(self, wrapper_cid: str) -> Mapping[str, Any]:
        cid = validate_opaque_cid(wrapper_cid, "wrapper_cid")
        if not self.port.has(cid):
            raise CapsuleAdmissionError(f"capsule envelope not found: {cid}")
        payload = self.port.get(cid)
        if not isinstance(payload, Mapping):
            raise CapsuleAdmissionError("capsule envelope artifact is malformed")
        if payload.get("schema") != CAPSULE_CACHE_INDEX_SCHEMA:
            raise CapsuleAdmissionError("artifact is not a capsule cache envelope")
        envelope = payload.get("envelope")
        if not isinstance(envelope, Mapping):
            raise CapsuleAdmissionError("capsule envelope body is malformed")
        return dict(envelope)

    def has(self, cid: str) -> bool:
        return bool(self.port.has(validate_opaque_cid(cid, "cid")))


def capsule_may_substitute(admission: CapsuleAdmission) -> bool:
    """Return True only for exact/conservative non-stale substitutions."""

    if not isinstance(admission, CapsuleAdmission):
        raise CapsuleAdmissionError("admission must be a CapsuleAdmission")
    return (
        admission.may_substitute
        and admission.freshness == FRESHNESS_FRESH
        and not admission.requires_raw_source
        and admission.ref.confidence in SUBSTITUTABLE_CONFIDENCE
    )


__all__ = [
    "ADMISSION_CONSERVATIVE",
    "ADMISSION_DECISIONS",
    "ADMISSION_EXACT",
    "ADMISSION_RAW",
    "CAPSULE_ADMISSION_INTERFACE",
    "CAPSULE_ADMISSION_SCHEMA",
    "CONFIDENCE_VALUES",
    "CapsuleAdmission",
    "CapsuleAdmissionError",
    "CapsuleCache",
    "FRESHNESS_FRESH",
    "FRESHNESS_STALE",
    "FRESHNESS_UNKNOWN",
    "FRESHNESS_VALUES",
    "NON_SUBSTITUTABLE_CONFIDENCE",
    "SUBSTITUTABLE_CONFIDENCE",
    "admit_capsule",
    "capsule_may_substitute",
    "retrieve_opaque_source",
]
