"""Exact-first, non-authoritative nomination of doctor repair candidates.

Retrieval follows a hard precedence order:

1. exact symbol / contract / value / lineage / graph routes
2. lexical similarity
3. knowledge-graph neighborhoods
4. vector / embedding similarity (optional lane only)

Approximate sources (KG, lexical, vector) remain
``semantic_authority=false`` and cannot select the target, required
behavior, value, placement, or write path.

This adapter:

* rejects stale, cross-tree, generated, read-only, poisoned, and forged
  candidates **before** scoring;
* carries candidate CIDs, source authority, hard compatibility facts, and
  information-content refs **separately** from scores;
* records deterministic ties: zero candidates and multiple equally eligible
  candidates remain explicit and never authorize semantics, values,
  placements, targets, or writes; and
* disables only the optional vector lane when an embedding canary fails —
  exact routes continue.

It reuses the RPR signal patterns from contract-repair and missing-input
retrieval without granting edit authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from enum import Enum
from typing import Any, ClassVar, Iterable

from ..proof.formal_verification_contracts import CanonicalContract, content_identity

try:
    from ..integrations.ipfs_datasets_embedding_provider import (
        EmbeddingLaneStatus,
        EmbeddingResult,
        IpfsDatasetsEmbeddingProvider,
        PinnedEmbeddingPolicy,
    )
except Exception:  # pragma: no cover - optional at import for partial trees
    EmbeddingLaneStatus = None  # type: ignore[misc, assignment]
    EmbeddingResult = None  # type: ignore[misc, assignment]
    IpfsDatasetsEmbeddingProvider = None  # type: ignore[misc, assignment]
    PinnedEmbeddingPolicy = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

DOCTOR_CANDIDATE_QUERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/doctor-candidate-query@1"
)
DOCTOR_CANDIDATE_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/doctor-candidate-evidence@1"
)
DOCTOR_CANDIDATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/doctor-repair-candidate@1"
)
DOCTOR_CANDIDATE_NOMINATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/doctor-candidate-nomination@1"
)
DOCTOR_CANDIDATE_SET_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/doctor-candidate-set@1"
)
DOCTOR_CANDIDATE_BOUNDS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/doctor-candidate-retrieval-bounds@1"
)
DOCTOR_SIGNAL_REF_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/doctor-candidate-signal-ref@1"
)
DOCTOR_AUTHORITY_ROOTS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/doctor-retrieval-authority-roots@1"
)

PRODUCER_ID = "doctor-repair-candidate-retrieval@1"
MAX_CANDIDATE_COUNT = 256
DEFAULT_MAX_PER_SIGNAL = 64
MAX_REF_BYTES = 512
MAX_TEXT_BYTES = 1_024

# Exact routes always precede approximate ones.  Tuple order is normative.
SIGNAL_PRECEDENCE: tuple[str, ...] = (
    "exact_symbol",
    "exact_contract",
    "exact_value",
    "exact_lineage",
    "exact_graph",
    "lexical",
    "knowledge_graph",
    "vector",
)


class DoctorCandidateRetrievalError(ValueError):
    """A doctor candidate signal cannot safely participate in a nomination."""


class DoctorCandidateRetrievalBindingError(DoctorCandidateRetrievalError):
    """A required root, query, index, or policy binding was mixed."""


class DoctorCandidateRetrievalBoundsError(DoctorCandidateRetrievalError):
    """A producer attempted to exceed the fixed retrieval budget."""


class DoctorCandidateSignal(str, Enum):
    """Closed signal families ordered by exact-first precedence."""

    EXACT_SYMBOL = "exact_symbol"
    EXACT_CONTRACT = "exact_contract"
    EXACT_VALUE = "exact_value"
    EXACT_LINEAGE = "exact_lineage"
    EXACT_GRAPH = "exact_graph"
    LEXICAL = "lexical"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    VECTOR = "vector"


class DoctorCandidateKind(str, Enum):
    """What a nominated candidate is proposing (classification only)."""

    RENAME = "rename"
    MOVE = "move"
    STRUCTURAL_EQUIVALENT = "structural_equivalent"
    CONSTRUCTOR = "constructor"
    FACTORY = "factory"
    ADAPTER = "adapter"
    REACHING_VALUE = "reaching_value"
    ANALOGOUS_REPAIR = "analogous_repair"
    UNKNOWN = "unknown"


class DoctorCandidateDisposition(str, Enum):
    NOMINATED = "nominated"
    REJECTED = "rejected"


class DoctorEligibilityStatus(str, Enum):
    """Explicit set-level eligibility; never an admission decision."""

    NO_CANDIDATE = "no_candidate"
    UNIQUE_ELIGIBLE = "unique_eligible"
    MULTIPLE_EQUALLY_ELIGIBLE = "multiple_equally_eligible"
    NOMINATED_SET = "nominated_set"
    ALL_REJECTED = "all_rejected"


class DoctorSourceAuthority(str, Enum):
    """How strongly a signal may be trusted as *source* material.

    Source authority is not semantic or write authority.  Approximate
    signals are always ``NOMINATED``.
    """

    AUTHORITATIVE = "authoritative"
    REVIEWED = "reviewed"
    NOMINATED = "nominated"
    UNKNOWN = "unknown"


SIGNAL_FAMILIES = tuple(item.value for item in DoctorCandidateSignal)

_SIGNAL_ALIASES = {
    "symbol": DoctorCandidateSignal.EXACT_SYMBOL.value,
    "ast": DoctorCandidateSignal.EXACT_SYMBOL.value,
    "ast_symbol": DoctorCandidateSignal.EXACT_SYMBOL.value,
    "contract": DoctorCandidateSignal.EXACT_CONTRACT.value,
    "schema": DoctorCandidateSignal.EXACT_CONTRACT.value,
    "value": DoctorCandidateSignal.EXACT_VALUE.value,
    "reaching_value": DoctorCandidateSignal.EXACT_VALUE.value,
    "lineage": DoctorCandidateSignal.EXACT_LINEAGE.value,
    "history": DoctorCandidateSignal.EXACT_LINEAGE.value,
    "git_lineage": DoctorCandidateSignal.EXACT_LINEAGE.value,
    "graph": DoctorCandidateSignal.EXACT_GRAPH.value,
    "dependency_graph": DoctorCandidateSignal.EXACT_GRAPH.value,
    "program_graph": DoctorCandidateSignal.EXACT_GRAPH.value,
    "bm25": DoctorCandidateSignal.LEXICAL.value,
    "lexical_bm25": DoctorCandidateSignal.LEXICAL.value,
    "kg": DoctorCandidateSignal.KNOWLEDGE_GRAPH.value,
    "graphrag": DoctorCandidateSignal.KNOWLEDGE_GRAPH.value,
    "embedding": DoctorCandidateSignal.VECTOR.value,
    "vector_hit": DoctorCandidateSignal.VECTOR.value,
}

EXACT_SIGNALS = frozenset(SIGNAL_PRECEDENCE[:5])
APPROXIMATE_SIGNALS = frozenset(SIGNAL_PRECEDENCE[5:])

# Stable public diagnostics.  Do not change without a versioned receipt schema.
REJECTION_STALE_OR_CROSS_TREE = "stale_or_cross_tree"
REJECTION_GENERATED = "generated_or_vendor_archive"
REJECTION_READ_ONLY = "read_only_target"
REJECTION_POISONED = "poisoned_candidate"
REJECTION_FORGED = "forged_candidate"
REJECTION_PARTIAL = "partial_candidate"
REJECTION_BODY_OR_SECRET = "body_or_secret_payload"
REJECTION_SEMANTIC_AUTHORITY_CLAIM = "semantic_authority_claim"
REJECTION_COMPATIBILITY_CLAIM = "hard_compatibility_claim"
REJECTION_WRITE_SCOPE_CLAIM = "write_scope_claim"
REJECTION_PLACEMENT_CLAIM = "placement_claim"
REJECTION_TARGET_CLAIM = "target_selection_claim"
REJECTION_VALUE_AUTHORITY_CLAIM = "value_authority_claim"
REJECTION_VECTOR_LANE_DISABLED = "vector_lane_disabled"
REJECTION_UNPINNED_EMBEDDING = "unpinned_embedding"
REJECTION_INVALID_PAYLOAD = "invalid_candidate_payload"
REJECTION_SCORE_BEFORE_FILTER = "score_used_before_rejection_filter"

_BODY_FIELDS = frozenset(
    {
        "source",
        "source_body",
        "source_text",
        "source_code",
        "body",
        "content",
        "contents",
        "text",
        "code",
        "raw",
        "raw_text",
        "ast",
        "ast_body",
        "embedding",
        "query_vector",
        "model_output",
        "completion",
        "prompt",
        "snippet",
        "file_text",
    }
)
_SECRET_FIELDS = frozenset(
    {
        "secret",
        "password",
        "api_key",
        "access_token",
        "refresh_token",
        "private_key",
        "authorization",
        "credential",
        "session_token",
        "cookie",
        "token",
        "passwd",
    }
)
_GENERATED_PARTS = frozenset(
    {
        "vendor",
        "vendors",
        "node_modules",
        "third_party",
        "archive",
        "archives",
        "generated",
        "build",
        "dist",
        "__pycache__",
    }
)
_ROOT_KEYS = (
    "repository_id",
    "forest_id",
    "tree_id",
    "overlay_id",
    "graph_id",
    "index_id",
    "model_id",
    "config_id",
    "corpus_id",
    "policy_id",
    "toolchain_id",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, CanonicalContract):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        converter = getattr(value, "to_dict", None)
        return _canonical(converter() if callable(converter) else vars(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return "<non-finite>"
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _fingerprint(value: Any, *, prefix: str = "doctor-candidate") -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"{prefix}:sha256:" + hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        result = converter()
        return dict(result) if isinstance(result, Mapping) else {}
    if is_dataclass(value) and not isinstance(value, type):
        return dict(vars(value))
    return {}


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_BYTES) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        text = str(value)
    else:
        text = value
    text = text.strip()
    if required and not text:
        raise DoctorCandidateRetrievalError(f"{name} is required")
    if "\x00" in text or len(text.encode("utf-8")) > limit:
        raise DoctorCandidateRetrievalBoundsError(f"{name} is invalid or exceeds its bound")
    return text


def _signal(name: Any) -> str:
    normalized = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = _SIGNAL_ALIASES.get(normalized, normalized)
    if normalized not in SIGNAL_FAMILIES:
        raise DoctorCandidateRetrievalError(f"unsupported doctor candidate signal: {name}")
    return normalized


def _signal_rank(signal: str) -> int:
    try:
        return SIGNAL_PRECEDENCE.index(signal)
    except ValueError:
        return len(SIGNAL_PRECEDENCE)


def _kind(name: Any) -> DoctorCandidateKind:
    if isinstance(name, DoctorCandidateKind):
        return name
    normalized = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "renamed": DoctorCandidateKind.RENAME.value,
        "rename_substitution": DoctorCandidateKind.RENAME.value,
        "moved": DoctorCandidateKind.MOVE.value,
        "move_module": DoctorCandidateKind.MOVE.value,
        "structural": DoctorCandidateKind.STRUCTURAL_EQUIVALENT.value,
        "equivalent": DoctorCandidateKind.STRUCTURAL_EQUIVALENT.value,
        "ctor": DoctorCandidateKind.CONSTRUCTOR.value,
        "factory_route": DoctorCandidateKind.FACTORY.value,
        "adapter_mapping": DoctorCandidateKind.ADAPTER.value,
        "value": DoctorCandidateKind.REACHING_VALUE.value,
        "missing_value": DoctorCandidateKind.REACHING_VALUE.value,
        "analog": DoctorCandidateKind.ANALOGOUS_REPAIR.value,
        "analogue": DoctorCandidateKind.ANALOGOUS_REPAIR.value,
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return DoctorCandidateKind(normalized)
    except ValueError as exc:
        raise DoctorCandidateRetrievalError(f"unsupported candidate kind: {name}") from exc


def _source_authority(name: Any, signal: str) -> DoctorSourceAuthority:
    if isinstance(name, DoctorSourceAuthority):
        authority = name
    elif name not in (None, ""):
        try:
            authority = DoctorSourceAuthority(
                str(name).strip().casefold().replace("-", "_")
            )
        except ValueError as exc:
            raise DoctorCandidateRetrievalError(
                f"unsupported source authority: {name}"
            ) from exc
    else:
        authority = (
            DoctorSourceAuthority.REVIEWED
            if signal in EXACT_SIGNALS
            else DoctorSourceAuthority.NOMINATED
        )
    # Approximate signals can never be elevated to authoritative.
    if signal in APPROXIMATE_SIGNALS and authority is DoctorSourceAuthority.AUTHORITATIVE:
        return DoctorSourceAuthority.NOMINATED
    return authority


def _verify_record_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    claimed = payload.get("content_id", payload.get("cid", ""))
    if claimed not in (None, "", record.content_id):
        raise DoctorCandidateRetrievalBindingError(
            "stored content identity does not match the canonical record"
        )


def _contains_body_or_secret(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _BODY_FIELDS or normalized in _SECRET_FIELDS:
                return True
            if _contains_body_or_secret(item):
                return True
        return False
    if isinstance(value, (bytes, bytearray)):
        return True
    return isinstance(value, Sequence) and not isinstance(value, str) and any(
        _contains_body_or_secret(item) for item in value
    )


def _redact_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _BODY_FIELDS or normalized in _SECRET_FIELDS:
                result[str(key)] = "<redacted>"
            else:
                result[str(key)] = _redact_payload(item)
        return result
    if isinstance(value, (bytes, bytearray)):
        return "<redacted-bytes>"
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_redact_payload(item) for item in value]
    return value


def candidate_set_identity(candidates: Sequence["DoctorRepairCandidate"]) -> str:
    """Bind the complete, deterministically ordered candidate set."""
    if not candidates:
        raise DoctorCandidateRetrievalBoundsError("candidate set must be nonempty")
    if len(candidates) > MAX_CANDIDATE_COUNT:
        raise DoctorCandidateRetrievalBoundsError("candidate set exceeds hard bound")
    ids = tuple(sorted(item.content_id for item in candidates))
    if len(set(ids)) != len(ids):
        raise DoctorCandidateRetrievalError("candidate set contains duplicate candidates")
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/doctor-candidate-set-id@1",
            "candidate_ids": list(ids),
        }
    )


# ---------------------------------------------------------------------------
# Authority roots (retrieval-local; LPR-029 owns the full doctor roots)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorRetrievalAuthorityRoots(CanonicalContract):
    """Roots whose drift invalidates a doctor candidate retrieval receipt.

    Intentionally a subset of the full doctor authority surface so this module
    does not depend on LPR-029 contracts.  Later stages re-bind the complete
    doctor roots before proof or write admission.
    """

    SCHEMA: ClassVar[str] = DOCTOR_AUTHORITY_ROOTS_SCHEMA

    repository_id: str
    forest_id: str
    tree_id: str
    overlay_id: str
    graph_id: str
    index_id: str
    model_id: str
    config_id: str
    corpus_id: str = ""
    policy_id: str = ""
    toolchain_id: str = ""
    embedding_policy_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "forest_id",
            "tree_id",
            "overlay_id",
            "graph_id",
            "index_id",
            "model_id",
            "config_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("corpus_id", "policy_id", "toolchain_id", "embedding_policy_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "forest_id": self.forest_id,
            "tree_id": self.tree_id,
            "overlay_id": self.overlay_id,
            "graph_id": self.graph_id,
            "index_id": self.index_id,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "corpus_id": self.corpus_id,
            "policy_id": self.policy_id,
            "toolchain_id": self.toolchain_id,
            "embedding_policy_id": self.embedding_policy_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorRetrievalAuthorityRoots":
        allowed = {
            "schema",
            "content_id",
            "cid",
            *cls.__dataclass_fields__,
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise DoctorCandidateRetrievalError("unsupported doctor retrieval roots payload")
        value = cls(
            **{
                name: payload.get(name, "")
                for name in cls.__dataclass_fields__
                if name != "SCHEMA"
            }
        )
        _verify_record_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorCandidateSignalRef(CanonicalContract):
    """Compact per-signal evidence pointer; never holds bodies."""

    SCHEMA: ClassVar[str] = DOCTOR_SIGNAL_REF_SCHEMA

    signal: str
    artifact_id: str
    locator: str = ""
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal", _signal(self.signal))
        object.__setattr__(
            self, "artifact_id", _text(self.artifact_id, "artifact_id", limit=MAX_REF_BYTES)
        )
        object.__setattr__(
            self, "locator", _text(self.locator, "locator", required=False, limit=MAX_REF_BYTES)
        )
        object.__setattr__(
            self,
            "producer_id",
            _text(self.producer_id or PRODUCER_ID, "producer_id", limit=MAX_REF_BYTES),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "artifact_id": self.artifact_id,
            "locator": self.locator,
            "producer_id": self.producer_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorCandidateSignalRef":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "signal",
            "artifact_id",
            "locator",
            "producer_id",
        }
        if not isinstance(payload, Mapping) or set(payload).difference(allowed):
            raise DoctorCandidateRetrievalError("unsupported doctor signal ref payload")
        if payload.get("schema") not in (None, cls.SCHEMA):
            raise DoctorCandidateRetrievalError("unsupported doctor signal ref schema")
        value = cls(
            signal=payload.get("signal", ""),
            artifact_id=payload.get("artifact_id", ""),
            locator=payload.get("locator", ""),
            producer_id=payload.get("producer_id", PRODUCER_ID),
        )
        _verify_record_identity(payload, value)
        return value


def _refs(value: Any, signal: str, raw: Mapping[str, Any]) -> tuple[DoctorCandidateSignalRef, ...]:
    if value is None:
        values: Iterable[Any] = ()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        values = value
    else:
        values = (value,)
    refs: list[DoctorCandidateSignalRef] = []
    for item in values:
        try:
            if isinstance(item, DoctorCandidateSignalRef):
                ref = item
            elif isinstance(item, Mapping):
                ref = DoctorCandidateSignalRef(
                    signal=str(item.get("signal", signal)),
                    artifact_id=str(item.get("artifact_id", item.get("locator", ""))),
                    locator=str(item.get("locator", "")),
                    producer_id=str(item.get("producer_id", PRODUCER_ID)),
                )
            elif isinstance(item, str) and item.strip():
                ref = DoctorCandidateSignalRef(signal=signal, artifact_id=item.strip())
            else:
                continue
        except (KeyError, DoctorCandidateRetrievalError, TypeError):
            continue
        if ref not in refs:
            refs.append(ref)
    if not refs:
        refs.append(
            DoctorCandidateSignalRef(
                signal=signal,
                artifact_id=_fingerprint(raw, prefix="signal-artifact"),
            )
        )
    return tuple(sorted(refs, key=lambda item: item.content_id))


@dataclass(frozen=True)
class DoctorCandidateRetrievalBounds(CanonicalContract):
    """Fixed, replayable caps; over-budget input is rejected, never truncated."""

    SCHEMA: ClassVar[str] = DOCTOR_CANDIDATE_BOUNDS_SCHEMA

    max_candidates: int = MAX_CANDIDATE_COUNT
    max_candidates_per_signal: int = DEFAULT_MAX_PER_SIGNAL

    def __post_init__(self) -> None:
        for name in ("max_candidates", "max_candidates_per_signal"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= MAX_CANDIDATE_COUNT
            ):
                raise DoctorCandidateRetrievalBoundsError(
                    f"{name} must be an integer from 1 through {MAX_CANDIDATE_COUNT}"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "max_candidates_per_signal": self.max_candidates_per_signal,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorCandidateRetrievalBounds":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "max_candidates",
            "max_candidates_per_signal",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise DoctorCandidateRetrievalError("unsupported doctor retrieval bounds payload")
        value = cls(
            max_candidates=payload.get("max_candidates", MAX_CANDIDATE_COUNT),
            max_candidates_per_signal=payload.get(
                "max_candidates_per_signal", DEFAULT_MAX_PER_SIGNAL
            ),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorCandidateQuery(CanonicalContract):
    """Exact binding of a doctor candidate retrieval query to authority roots."""

    SCHEMA: ClassVar[str] = DOCTOR_CANDIDATE_QUERY_SCHEMA

    roots: DoctorRetrievalAuthorityRoots
    finding_id: str
    query_kind: str = "repair"
    subject_path: str = ""
    subject_symbol: str = ""
    subject_span_ref: str = ""
    obligation_refs: tuple[str, ...] = ()
    expected_behavior_refs: tuple[str, ...] = ()
    embedding_policy_id: str = ""
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorRetrievalAuthorityRoots):
            raise DoctorCandidateRetrievalBindingError(
                "query roots must be DoctorRetrievalAuthorityRoots"
            )
        object.__setattr__(self, "finding_id", _text(self.finding_id, "finding_id"))
        object.__setattr__(
            self,
            "query_kind",
            _text(self.query_kind or "repair", "query_kind", limit=64),
        )
        object.__setattr__(
            self, "subject_path", _text(self.subject_path, "subject_path", required=False)
        )
        object.__setattr__(
            self,
            "subject_symbol",
            _text(self.subject_symbol, "subject_symbol", required=False),
        )
        object.__setattr__(
            self,
            "subject_span_ref",
            _text(self.subject_span_ref, "subject_span_ref", required=False),
        )
        object.__setattr__(
            self,
            "obligation_refs",
            tuple(
                sorted(
                    {
                        _text(item, "obligation_refs")
                        for item in (self.obligation_refs or ())
                        if str(item).strip()
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "expected_behavior_refs",
            tuple(
                sorted(
                    {
                        _text(item, "expected_behavior_refs")
                        for item in (self.expected_behavior_refs or ())
                        if str(item).strip()
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "embedding_policy_id",
            _text(self.embedding_policy_id, "embedding_policy_id", required=False),
        )
        if self.semantic_authority is not False:
            raise DoctorCandidateRetrievalBindingError(
                "doctor candidate query cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "roots": self.roots.to_dict(),
            "finding_id": self.finding_id,
            "query_kind": self.query_kind,
            "subject_path": self.subject_path,
            "subject_symbol": self.subject_symbol,
            "subject_span_ref": self.subject_span_ref,
            "obligation_refs": list(self.obligation_refs),
            "expected_behavior_refs": list(self.expected_behavior_refs),
            "embedding_policy_id": self.embedding_policy_id,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorCandidateQuery":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "roots",
            "finding_id",
            "query_kind",
            "subject_path",
            "subject_symbol",
            "subject_span_ref",
            "obligation_refs",
            "expected_behavior_refs",
            "embedding_policy_id",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise DoctorCandidateRetrievalError("unsupported doctor candidate query payload")
        roots = payload.get("roots")
        value = cls(
            roots=roots
            if isinstance(roots, DoctorRetrievalAuthorityRoots)
            else DoctorRetrievalAuthorityRoots.from_dict(roots),
            finding_id=payload.get("finding_id", ""),
            query_kind=payload.get("query_kind", "repair"),
            subject_path=payload.get("subject_path", ""),
            subject_symbol=payload.get("subject_symbol", ""),
            subject_span_ref=payload.get("subject_span_ref", ""),
            obligation_refs=tuple(payload.get("obligation_refs", ())),
            expected_behavior_refs=tuple(payload.get("expected_behavior_refs", ())),
            embedding_policy_id=payload.get("embedding_policy_id", ""),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorCandidateEvidence(CanonicalContract):
    """Facts carried separately from scores: CID, authority, compatibility, IC."""

    SCHEMA: ClassVar[str] = DOCTOR_CANDIDATE_EVIDENCE_SCHEMA

    candidate_cid: str
    source_authority: DoctorSourceAuthority
    hard_compatible: bool | None = None
    information_content_ref: str = ""
    signal_refs: tuple[DoctorCandidateSignalRef, ...] = ()
    primary_signal: str = ""
    score_millionths: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_cid", _text(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self,
            "source_authority",
            _source_authority(self.source_authority, self.primary_signal or "exact_symbol"),
        )
        if self.hard_compatible is not None and not isinstance(self.hard_compatible, bool):
            raise DoctorCandidateRetrievalError("hard_compatible must be bool or None")
        # hard_compatible is an observed *fact* flag (True/False/unknown), not a
        # compatibility claim that admits the candidate.  It never grants write
        # or semantic authority.
        object.__setattr__(
            self,
            "information_content_ref",
            _text(self.information_content_ref, "information_content_ref", required=False),
        )
        refs = tuple(
            sorted(
                (
                    item
                    if isinstance(item, DoctorCandidateSignalRef)
                    else DoctorCandidateSignalRef.from_dict(item)
                    for item in (self.signal_refs or ())
                ),
                key=lambda item: item.content_id,
            )
        )
        object.__setattr__(self, "signal_refs", refs)
        primary = self.primary_signal
        if primary:
            object.__setattr__(self, "primary_signal", _signal(primary))
        else:
            object.__setattr__(self, "primary_signal", "")
        if self.score_millionths is not None:
            if isinstance(self.score_millionths, bool) or not isinstance(
                self.score_millionths, int
            ):
                raise DoctorCandidateRetrievalError("score_millionths must be int or None")
            object.__setattr__(self, "score_millionths", int(self.score_millionths))
        notes = tuple(
            sorted(
                {
                    _text(item, "notes", limit=MAX_TEXT_BYTES)
                    for item in (self.notes or ())
                    if str(item).strip()
                }
            )
        )
        object.__setattr__(self, "notes", notes)

    def _payload(self) -> dict[str, Any]:
        return {
            "candidate_cid": self.candidate_cid,
            "source_authority": self.source_authority.value,
            "hard_compatible": self.hard_compatible,
            "information_content_ref": self.information_content_ref,
            "signal_refs": [ref.to_dict() for ref in self.signal_refs],
            "primary_signal": self.primary_signal,
            "score_millionths": self.score_millionths,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorCandidateEvidence":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "candidate_cid",
            "source_authority",
            "hard_compatible",
            "information_content_ref",
            "signal_refs",
            "primary_signal",
            "score_millionths",
            "notes",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise DoctorCandidateRetrievalError("unsupported doctor candidate evidence payload")
        refs = payload.get("signal_refs", ())
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
            raise DoctorCandidateRetrievalError("signal_refs must be a sequence")
        value = cls(
            candidate_cid=payload.get("candidate_cid", ""),
            source_authority=payload.get("source_authority", DoctorSourceAuthority.UNKNOWN),
            hard_compatible=payload.get("hard_compatible", None),
            information_content_ref=payload.get("information_content_ref", ""),
            signal_refs=tuple(
                item
                if isinstance(item, DoctorCandidateSignalRef)
                else DoctorCandidateSignalRef.from_dict(item)
                for item in refs
            ),
            primary_signal=payload.get("primary_signal", ""),
            score_millionths=payload.get("score_millionths", None),
            notes=tuple(payload.get("notes", ())),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorRepairCandidate(CanonicalContract):
    """One non-authoritative repair/value nomination."""

    SCHEMA: ClassVar[str] = DOCTOR_CANDIDATE_SCHEMA

    roots: DoctorRetrievalAuthorityRoots
    finding_id: str
    candidate_ref: str
    kind: DoctorCandidateKind
    path: str = ""
    symbol_id: str = ""
    evidence: DoctorCandidateEvidence | None = None
    diagnostics: tuple[str, ...] = ()
    semantic_authority: bool = False
    write_paths: tuple[str, ...] = ()
    placement_claim: bool = False
    target_claim: bool = False
    value_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorRetrievalAuthorityRoots):
            raise DoctorCandidateRetrievalBindingError(
                "candidate roots must be DoctorRetrievalAuthorityRoots"
            )
        object.__setattr__(self, "finding_id", _text(self.finding_id, "finding_id"))
        object.__setattr__(
            self, "candidate_ref", _text(self.candidate_ref, "candidate_ref")
        )
        object.__setattr__(self, "kind", _kind(self.kind))
        object.__setattr__(self, "path", _text(self.path, "path", required=False))
        object.__setattr__(
            self, "symbol_id", _text(self.symbol_id, "symbol_id", required=False)
        )
        if self.evidence is not None and not isinstance(
            self.evidence, DoctorCandidateEvidence
        ):
            raise DoctorCandidateRetrievalError(
                "evidence must be DoctorCandidateEvidence when present"
            )
        diagnostics = tuple(
            sorted({str(item).strip() for item in (self.diagnostics or ()) if str(item).strip()})
        )
        object.__setattr__(self, "diagnostics", diagnostics)
        for flag_name in (
            "semantic_authority",
            "placement_claim",
            "target_claim",
            "value_authority",
        ):
            if getattr(self, flag_name) is not False:
                raise DoctorCandidateRetrievalBindingError(
                    f"retrieval cannot assert {flag_name}"
                )
            object.__setattr__(self, flag_name, False)
        if self.write_paths:
            raise DoctorCandidateRetrievalBindingError(
                "retrieval cannot assert write scope"
            )
        object.__setattr__(self, "write_paths", ())

    @property
    def candidate_cid(self) -> str:
        if self.evidence is not None and self.evidence.candidate_cid:
            return self.evidence.candidate_cid
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "roots": self.roots.to_dict(),
            "finding_id": self.finding_id,
            "candidate_ref": self.candidate_ref,
            "kind": self.kind.value,
            "path": self.path,
            "symbol_id": self.symbol_id,
            "evidence": None if self.evidence is None else self.evidence.to_dict(),
            "diagnostics": list(self.diagnostics),
            "semantic_authority": False,
            "write_paths": [],
            "placement_claim": False,
            "target_claim": False,
            "value_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorRepairCandidate":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "roots",
            "finding_id",
            "candidate_ref",
            "kind",
            "path",
            "symbol_id",
            "evidence",
            "diagnostics",
            "semantic_authority",
            "write_paths",
            "placement_claim",
            "target_claim",
            "value_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise DoctorCandidateRetrievalError("unsupported doctor repair candidate payload")
        roots = payload.get("roots")
        evidence = payload.get("evidence")
        value = cls(
            roots=roots
            if isinstance(roots, DoctorRetrievalAuthorityRoots)
            else DoctorRetrievalAuthorityRoots.from_dict(roots),
            finding_id=payload.get("finding_id", ""),
            candidate_ref=payload.get("candidate_ref", ""),
            kind=payload.get("kind", DoctorCandidateKind.UNKNOWN),
            path=payload.get("path", ""),
            symbol_id=payload.get("symbol_id", ""),
            evidence=(
                None
                if evidence in (None, "")
                else (
                    evidence
                    if isinstance(evidence, DoctorCandidateEvidence)
                    else DoctorCandidateEvidence.from_dict(evidence)
                )
            ),
            diagnostics=tuple(payload.get("diagnostics", ())),
            semantic_authority=payload.get("semantic_authority", False),
            write_paths=tuple(payload.get("write_paths", ())),
            placement_claim=payload.get("placement_claim", False),
            target_claim=payload.get("target_claim", False),
            value_authority=payload.get("value_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorCandidateNomination(CanonicalContract):
    """One candidate plus complete per-signal provenance and no authority."""

    SCHEMA: ClassVar[str] = DOCTOR_CANDIDATE_NOMINATION_SCHEMA

    candidate: DoctorRepairCandidate
    disposition: DoctorCandidateDisposition
    signal_evidence: tuple[tuple[str, tuple[DoctorCandidateSignalRef, ...]], ...]
    diagnostics: tuple[str, ...] = ()
    eligibility_rank: int = 0
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DoctorRepairCandidate):
            raise DoctorCandidateRetrievalError(
                "nomination requires DoctorRepairCandidate"
            )
        object.__setattr__(
            self, "disposition", DoctorCandidateDisposition(self.disposition)
        )
        rows: list[tuple[str, tuple[DoctorCandidateSignalRef, ...]]] = []
        raw_evidence = (
            self.signal_evidence.items()
            if isinstance(self.signal_evidence, Mapping)
            else self.signal_evidence
        )
        for item in raw_evidence:
            try:
                signal, refs = item
            except (TypeError, ValueError) as exc:
                raise DoctorCandidateRetrievalError(
                    "signal evidence rows must contain signal and references"
                ) from exc
            normalized = _signal(signal)
            checked = tuple(
                ref
                if isinstance(ref, DoctorCandidateSignalRef)
                else DoctorCandidateSignalRef.from_dict(ref)
                for ref in (
                    refs
                    if isinstance(refs, Sequence)
                    and not isinstance(refs, (str, bytes, bytearray, Mapping))
                    else (refs,)
                )
            )
            checked = tuple(sorted(checked, key=lambda ref: ref.content_id))
            rows.append((normalized, checked))
        # Exact-first ordering of signal evidence rows.
        rows.sort(key=lambda item: (_signal_rank(item[0]), item[0]))
        if len({item[0] for item in rows}) != len(rows):
            raise DoctorCandidateRetrievalError("nomination has duplicate signal evidence")
        object.__setattr__(self, "signal_evidence", tuple(rows))
        diagnostics = tuple(
            sorted({str(item).strip() for item in (self.diagnostics or ()) if str(item).strip()})
        )
        object.__setattr__(self, "diagnostics", diagnostics)
        if isinstance(self.eligibility_rank, bool) or not isinstance(
            self.eligibility_rank, int
        ) or self.eligibility_rank < 0:
            raise DoctorCandidateRetrievalError(
                "eligibility_rank must be a non-negative integer"
            )
        if self.semantic_authority is not False:
            raise DoctorCandidateRetrievalBindingError(
                "nominations cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        if (
            self.disposition is DoctorCandidateDisposition.NOMINATED
            and diagnostics
        ):
            raise DoctorCandidateRetrievalError(
                "nominated candidates cannot carry rejection diagnostics"
            )
        if (
            self.disposition is DoctorCandidateDisposition.REJECTED
            and not diagnostics
        ):
            raise DoctorCandidateRetrievalError(
                "rejected candidates require stable diagnostics"
            )

    @property
    def write_paths(self) -> tuple[str, ...]:
        return ()

    @property
    def primary_signal(self) -> str:
        if not self.signal_evidence:
            return ""
        return min(self.signal_evidence, key=lambda item: _signal_rank(item[0]))[0]

    def _payload(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "disposition": self.disposition.value,
            "signal_evidence": [
                {
                    "signal": signal,
                    "evidence_refs": [ref.to_dict() for ref in refs],
                }
                for signal, refs in self.signal_evidence
            ],
            "diagnostics": list(self.diagnostics),
            "eligibility_rank": self.eligibility_rank,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorCandidateNomination":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "candidate",
            "disposition",
            "signal_evidence",
            "diagnostics",
            "eligibility_rank",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise DoctorCandidateRetrievalError(
                "unsupported doctor candidate nomination payload"
            )
        signal_evidence: list[tuple[str, tuple[DoctorCandidateSignalRef, ...]]] = []
        supplied = payload.get("signal_evidence", ())
        if not isinstance(supplied, Sequence) or isinstance(
            supplied, (str, bytes, bytearray)
        ):
            raise DoctorCandidateRetrievalError("signal_evidence must be a sequence")
        for row in supplied:
            if not isinstance(row, Mapping):
                raise DoctorCandidateRetrievalError("signal evidence row must be an object")
            refs = row.get("evidence_refs", ())
            if not isinstance(refs, Sequence) or isinstance(
                refs, (str, bytes, bytearray)
            ):
                raise DoctorCandidateRetrievalError(
                    "signal evidence references must be a sequence"
                )
            signal_evidence.append(
                (
                    str(row.get("signal", "")),
                    tuple(
                        item
                        if isinstance(item, DoctorCandidateSignalRef)
                        else DoctorCandidateSignalRef.from_dict(item)
                        for item in refs
                    ),
                )
            )
        candidate = payload.get("candidate")
        value = cls(
            candidate=candidate
            if isinstance(candidate, DoctorRepairCandidate)
            else DoctorRepairCandidate.from_dict(candidate),
            disposition=payload.get("disposition", ""),
            signal_evidence=tuple(signal_evidence),
            diagnostics=tuple(payload.get("diagnostics", ())),
            eligibility_rank=payload.get("eligibility_rank", 0),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class DoctorCandidateSet(CanonicalContract):
    """The complete bounded candidate set; this is not a target decision."""

    SCHEMA: ClassVar[str] = DOCTOR_CANDIDATE_SET_SCHEMA

    roots: DoctorRetrievalAuthorityRoots
    query: DoctorCandidateQuery
    finding_id: str
    bounds: DoctorCandidateRetrievalBounds
    candidates: tuple[DoctorCandidateNomination, ...]
    candidate_set_id: str
    eligibility_status: DoctorEligibilityStatus
    signal_roots: tuple[tuple[str, str], ...] = ()
    vector_lane_status: str = "not_probed"
    embedding_policy_id: str = ""
    vector_query_id: str = ""
    equally_eligible_ids: tuple[str, ...] = ()
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, DoctorRetrievalAuthorityRoots):
            raise DoctorCandidateRetrievalError(
                "receipt roots must be DoctorRetrievalAuthorityRoots"
            )
        if not isinstance(self.query, DoctorCandidateQuery):
            raise DoctorCandidateRetrievalError("receipt query must be DoctorCandidateQuery")
        if not isinstance(self.bounds, DoctorCandidateRetrievalBounds):
            raise DoctorCandidateRetrievalError(
                "receipt bounds must be DoctorCandidateRetrievalBounds"
            )
        if self.query.roots != self.roots:
            raise DoctorCandidateRetrievalBindingError(
                "query roots do not match receipt roots"
            )
        object.__setattr__(self, "finding_id", _text(self.finding_id, "finding_id"))
        if self.finding_id != self.query.finding_id:
            raise DoctorCandidateRetrievalBindingError(
                "receipt finding_id does not match query"
            )
        candidates = tuple(sorted(self.candidates, key=lambda item: item.content_id))
        if not candidates or len(candidates) > self.bounds.max_candidates:
            raise DoctorCandidateRetrievalBoundsError(
                "receipt candidate count is outside its declared bound"
            )
        if any(not isinstance(item, DoctorCandidateNomination) for item in candidates):
            raise DoctorCandidateRetrievalError("receipt candidates must be nominations")
        if len({item.content_id for item in candidates}) != len(candidates):
            raise DoctorCandidateRetrievalError("receipt contains duplicate nominations")
        if any(item.candidate.roots != self.roots for item in candidates):
            raise DoctorCandidateRetrievalBindingError(
                "candidate roots do not match receipt roots"
            )
        if any(item.candidate.finding_id != self.finding_id for item in candidates):
            raise DoctorCandidateRetrievalBindingError(
                "candidate finding_id does not match receipt"
            )
        object.__setattr__(self, "candidates", candidates)
        expected = candidate_set_identity(tuple(item.candidate for item in candidates))
        if self.candidate_set_id != expected:
            raise DoctorCandidateRetrievalBindingError(
                "candidate_set_id does not bind the complete candidate set"
            )
        status = self.eligibility_status
        if not isinstance(status, DoctorEligibilityStatus):
            status = DoctorEligibilityStatus(str(status))
        object.__setattr__(self, "eligibility_status", status)
        roots: list[tuple[str, str]] = []
        for signal, root in self.signal_roots:
            normalized = _signal(signal)
            if not isinstance(root, str) or not root:
                raise DoctorCandidateRetrievalBindingError(
                    "signal roots must be nonempty identities"
                )
            roots.append((normalized, root))
        # Exact-first ordering of signal roots.
        roots.sort(key=lambda item: (_signal_rank(item[0]), item[0]))
        if len({item[0] for item in roots}) != len(roots):
            raise DoctorCandidateRetrievalBindingError(
                "receipt contains duplicate signal roots"
            )
        object.__setattr__(self, "signal_roots", tuple(roots))
        object.__setattr__(
            self,
            "vector_lane_status",
            _text(self.vector_lane_status or "not_probed", "vector_lane_status"),
        )
        object.__setattr__(
            self,
            "embedding_policy_id",
            _text(self.embedding_policy_id, "embedding_policy_id", required=False),
        )
        object.__setattr__(
            self,
            "vector_query_id",
            _text(self.vector_query_id, "vector_query_id", required=False),
        )
        ids = tuple(
            sorted(
                {
                    _text(item, "equally_eligible_ids")
                    for item in (self.equally_eligible_ids or ())
                    if str(item).strip()
                }
            )
        )
        object.__setattr__(self, "equally_eligible_ids", ids)
        if self.semantic_authority is not False:
            raise DoctorCandidateRetrievalBindingError(
                "retrieval receipts cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    @property
    def repair_candidates(self) -> tuple[DoctorRepairCandidate, ...]:
        return tuple(item.candidate for item in self.candidates)

    @property
    def write_paths(self) -> tuple[str, ...]:
        """Retrieval never provides mutation authority."""
        return ()

    @property
    def admitted_candidate_id(self) -> str:
        """There is deliberately no winner at retrieval time."""
        return ""

    @property
    def query_id(self) -> str:
        return self.query.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "roots": self.roots.to_dict(),
            "query": self.query.to_dict(),
            "finding_id": self.finding_id,
            "bounds": self.bounds.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "candidate_set_id": self.candidate_set_id,
            "eligibility_status": self.eligibility_status.value,
            "signal_roots": [
                {"signal": signal, "root_id": root} for signal, root in self.signal_roots
            ],
            "vector_lane_status": self.vector_lane_status,
            "embedding_policy_id": self.embedding_policy_id,
            "vector_query_id": self.vector_query_id,
            "equally_eligible_ids": list(self.equally_eligible_ids),
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorCandidateSet":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "roots",
            "query",
            "finding_id",
            "bounds",
            "candidates",
            "candidate_set_id",
            "eligibility_status",
            "signal_roots",
            "vector_lane_status",
            "embedding_policy_id",
            "vector_query_id",
            "equally_eligible_ids",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise DoctorCandidateRetrievalError("unsupported doctor candidate set payload")
        rows = payload.get("signal_roots", ())
        candidates = payload.get("candidates", ())
        if (
            not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes, bytearray))
            or not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes, bytearray))
        ):
            raise DoctorCandidateRetrievalError(
                "receipt signal roots and candidates must be sequences"
            )
        signal_roots: list[tuple[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise DoctorCandidateRetrievalError(
                    "receipt signal root row must be an object"
                )
            signal_roots.append(
                (str(row.get("signal", "")), str(row.get("root_id", "")))
            )
        roots = payload.get("roots")
        query = payload.get("query")
        bounds = payload.get("bounds")
        value = cls(
            roots=roots
            if isinstance(roots, DoctorRetrievalAuthorityRoots)
            else DoctorRetrievalAuthorityRoots.from_dict(roots),
            query=query
            if isinstance(query, DoctorCandidateQuery)
            else DoctorCandidateQuery.from_dict(query),
            finding_id=payload.get("finding_id", ""),
            bounds=bounds
            if isinstance(bounds, DoctorCandidateRetrievalBounds)
            else DoctorCandidateRetrievalBounds.from_dict(bounds),
            candidates=tuple(
                item
                if isinstance(item, DoctorCandidateNomination)
                else DoctorCandidateNomination.from_dict(item)
                for item in candidates
            ),
            candidate_set_id=payload.get("candidate_set_id", ""),
            eligibility_status=payload.get(
                "eligibility_status", DoctorEligibilityStatus.NOMINATED_SET
            ),
            signal_roots=tuple(signal_roots),
            vector_lane_status=payload.get("vector_lane_status", "not_probed"),
            embedding_policy_id=payload.get("embedding_policy_id", ""),
            vector_query_id=payload.get("vector_query_id", ""),
            equally_eligible_ids=tuple(payload.get("equally_eligible_ids", ())),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Diagnostics and inference
# ---------------------------------------------------------------------------


def _infer_kind(raw: Mapping[str, Any], signals: set[str]) -> DoctorCandidateKind:
    supplied = raw.get("kind", raw.get("candidate_kind", raw.get("strategy")))
    if supplied:
        try:
            return _kind(supplied)
        except DoctorCandidateRetrievalError:
            pass
    if raw.get("rename") is True or raw.get("renamed") is True:
        return DoctorCandidateKind.RENAME
    if raw.get("move") is True or raw.get("moved") is True:
        return DoctorCandidateKind.MOVE
    if raw.get("structural") is True or raw.get("structural_equivalent") is True:
        return DoctorCandidateKind.STRUCTURAL_EQUIVALENT
    if raw.get("constructor") is True or raw.get("ctor") is True:
        return DoctorCandidateKind.CONSTRUCTOR
    if raw.get("factory") is True:
        return DoctorCandidateKind.FACTORY
    if raw.get("adapter") is True:
        return DoctorCandidateKind.ADAPTER
    if (
        raw.get("reaching_value") is True
        or raw.get("value") is True
        or DoctorCandidateSignal.EXACT_VALUE.value in signals
    ):
        return DoctorCandidateKind.REACHING_VALUE
    if DoctorCandidateSignal.VECTOR.value in signals or DoctorCandidateSignal.LEXICAL.value in signals:
        return DoctorCandidateKind.ANALOGOUS_REPAIR
    if DoctorCandidateSignal.EXACT_LINEAGE.value in signals:
        return DoctorCandidateKind.MOVE
    if DoctorCandidateSignal.EXACT_SYMBOL.value in signals:
        return DoctorCandidateKind.RENAME
    return DoctorCandidateKind.UNKNOWN


def _candidate_ref(raw: Mapping[str, Any], query: DoctorCandidateQuery) -> str:
    for key in (
        "candidate_ref",
        "expression_ref",
        "symbol_id",
        "symbol",
        "name",
        "qualified_name",
        "target_ref",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    path = raw.get("path")
    if isinstance(path, str) and path.strip():
        return f"path:{path.strip()}"
    return f"candidate:{query.finding_id}:{_fingerprint(raw).split(':')[-1][:16]}"


def _path_of(raw: Mapping[str, Any], query: DoctorCandidateQuery) -> str:
    for key in ("path", "target_path", "file_path"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\\", "/")
    span = raw.get("target_span") or raw.get("span")
    if isinstance(span, Mapping) and span.get("path"):
        return str(span["path"]).replace("\\", "/")
    return query.subject_path


def _symbol_of(raw: Mapping[str, Any], query: DoctorCandidateQuery) -> str:
    for key in ("symbol_id", "symbol", "qualified_name", "name"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return query.subject_symbol


def _score_millionths(raw: Mapping[str, Any]) -> int | None:
    if "score_millionths" in raw:
        try:
            value = int(raw["score_millionths"])
            return value if math.isfinite(float(value)) else None
        except (TypeError, ValueError):
            return None
    if "score" in raw:
        try:
            score = float(raw["score"])
            if not math.isfinite(score):
                return None
            return int(round(score * 1_000_000))
        except (TypeError, ValueError):
            return None
    return None


def _hard_compatible_fact(raw: Mapping[str, Any]) -> bool | None:
    """Extract an observed compatibility *fact*, not a claim of admission.

    Explicit ``hard_compatible`` / ``signature_compatible`` booleans are
    retained as facts.  Bare ``compatible=True`` without the hard-prefix is
    treated as a forbidden compatibility claim (rejected by diagnostics).
    """
    for key in ("hard_compatible", "signature_compatible", "type_compatible_fact"):
        if key in raw:
            value = raw[key]
            if isinstance(value, bool):
                return value
    return None


def _diagnostics(
    signal: str,
    raw: Mapping[str, Any],
    path: str,
    expected_roots: DoctorRetrievalAuthorityRoots,
    *,
    vector_lane_enabled: bool,
    embedding_policy_id: str,
) -> set[str]:
    reasons: set[str] = set()
    if raw.get("partial") is True or raw.get("complete") is False:
        reasons.add(REJECTION_PARTIAL)
    if _contains_body_or_secret(raw):
        reasons.add(REJECTION_BODY_OR_SECRET)
    if raw.get("forged") is True or raw.get("forged_history") is True:
        reasons.add(REJECTION_FORGED)
    if raw.get("history_reviewed") is False and signal in {
        DoctorCandidateSignal.EXACT_LINEAGE.value,
        DoctorCandidateSignal.EXACT_SYMBOL.value,
    }:
        reasons.add(REJECTION_FORGED)
    if raw.get("read_only") is True or raw.get("writable") is False:
        reasons.add(REJECTION_READ_ONLY)
    parts = {part.casefold() for part in path.split("/")}
    if (
        raw.get("generated") is True
        or raw.get("vendor") is True
        or raw.get("archive") is True
        or parts.intersection(_GENERATED_PARTS)
    ):
        reasons.add(REJECTION_GENERATED)
    if raw.get("semantic_authority") is True:
        reasons.add(REJECTION_SEMANTIC_AUTHORITY_CLAIM)
    if raw.get("poisoned") is True:
        reasons.add(REJECTION_POISONED)
    # Forbidden claims that would authorize later stages.
    if raw.get("compatible") is True and "hard_compatible" not in raw:
        reasons.add(REJECTION_COMPATIBILITY_CLAIM)
    if raw.get("compatibility_claim") is True:
        reasons.add(REJECTION_COMPATIBILITY_CLAIM)
    for key in (
        "write_paths",
        "write_scope",
        "candidate_write_paths",
        "permitted_write_paths",
        "mutation_paths",
        "edit_paths",
    ):
        value = raw.get(key)
        if value not in (None, (), [], ""):
            reasons.add(REJECTION_WRITE_SCOPE_CLAIM)
            break
    for key in (
        "placement",
        "placement_path",
        "placement_decision",
        "chosen_placement",
        "admitted_placement",
    ):
        value = raw.get(key)
        if value not in (None, (), [], "", False):
            reasons.add(REJECTION_PLACEMENT_CLAIM)
            break
    if raw.get("selected_target") is True or raw.get("admitted_target") is True:
        reasons.add(REJECTION_TARGET_CLAIM)
    if raw.get("value_authority") is True or raw.get("admits_value") is True:
        reasons.add(REJECTION_VALUE_AUTHORITY_CLAIM)

    # Stale / cross-tree bindings — rejected before scoring.
    for key in _ROOT_KEYS:
        if key in raw and raw[key] not in (None, "", getattr(expected_roots, key, None)):
            reasons.add(REJECTION_STALE_OR_CROSS_TREE)
    candidate_roots = raw.get("roots")
    if isinstance(candidate_roots, DoctorRetrievalAuthorityRoots):
        if candidate_roots != expected_roots:
            reasons.add(REJECTION_STALE_OR_CROSS_TREE)
    elif isinstance(candidate_roots, Mapping):
        if any(
            key in candidate_roots
            and candidate_roots[key] not in (None, "", getattr(expected_roots, key, None))
            for key in _ROOT_KEYS
        ):
            reasons.add(REJECTION_STALE_OR_CROSS_TREE)

    if signal == DoctorCandidateSignal.VECTOR.value:
        if not vector_lane_enabled:
            reasons.add(REJECTION_VECTOR_LANE_DISABLED)
        if raw.get("unpinned") is True or raw.get("remote_unpinned") is True:
            reasons.add(REJECTION_UNPINNED_EMBEDDING)
        if embedding_policy_id and raw.get("embedding_policy_id") not in (
            None,
            "",
            embedding_policy_id,
        ):
            reasons.add(REJECTION_UNPINNED_EMBEDDING)
        try:
            score = raw.get("score", raw.get("score_millionths", 0))
            if score is not None and not math.isfinite(float(score)):
                reasons.add(REJECTION_POISONED)
        except (TypeError, ValueError):
            reasons.add(REJECTION_POISONED)
        if raw.get("semantic_authority", False) is not False:
            reasons.add(REJECTION_POISONED)
        for key, attr in (
            ("tree_id", "tree_id"),
            ("config_id", "config_id"),
            ("model_id", "model_id"),
            ("index_id", "index_id"),
        ):
            if key in raw and raw[key] not in (
                None,
                "",
                getattr(expected_roots, attr, None),
            ):
                reasons.add(REJECTION_STALE_OR_CROSS_TREE)

    # Scores must not influence rejection ordering: if a payload was scored
    # before filters and carries a pre-filter admission flag, reject.
    if raw.get("admitted_by_score") is True or raw.get("score_selected") is True:
        reasons.add(REJECTION_SCORE_BEFORE_FILTER)

    return reasons


def _eligibility_rank(signals: set[str], reasons: set[str]) -> int:
    """Lower is better.  Rejected candidates receive a large rank."""
    if reasons:
        return 10_000 + min((_signal_rank(s) for s in signals), default=100)
    if not signals:
        return 9_999
    return min(_signal_rank(s) for s in signals)


def _compute_eligibility(
    nominations: Sequence[DoctorCandidateNomination],
) -> tuple[DoctorEligibilityStatus, tuple[str, ...]]:
    nominated = [
        item
        for item in nominations
        if item.disposition is DoctorCandidateDisposition.NOMINATED
    ]
    if not nominated:
        if nominations and all(
            item.disposition is DoctorCandidateDisposition.REJECTED
            for item in nominations
        ):
            # Distinguish "no signals" partial from "all rejected".
            if all(
                REJECTION_PARTIAL in item.diagnostics
                and not item.signal_evidence
                for item in nominations
            ):
                return DoctorEligibilityStatus.NO_CANDIDATE, ()
            return DoctorEligibilityStatus.ALL_REJECTED, ()
        return DoctorEligibilityStatus.NO_CANDIDATE, ()
    best = min(item.eligibility_rank for item in nominated)
    tied = [item for item in nominated if item.eligibility_rank == best]
    if len(tied) == 1 and len(nominated) == 1:
        return DoctorEligibilityStatus.UNIQUE_ELIGIBLE, (tied[0].candidate.content_id,)
    if len(tied) > 1:
        return (
            DoctorEligibilityStatus.MULTIPLE_EQUALLY_ELIGIBLE,
            tuple(sorted(item.candidate.content_id for item in tied)),
        )
    return DoctorEligibilityStatus.NOMINATED_SET, ()


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class DoctorRepairCandidateRetriever:
    """Union bounded doctor signal families into a diagnostic-only candidate set.

    Exact signal families are processed and ranked before approximate ones.
    Vector participation requires a passed embedding canary / enabled lane.
    """

    def __init__(
        self,
        roots: DoctorRetrievalAuthorityRoots,
        *,
        bounds: DoctorCandidateRetrievalBounds | None = None,
        embedding_provider: Any | None = None,
        embedding_policy: Any | None = None,
    ) -> None:
        if not isinstance(roots, DoctorRetrievalAuthorityRoots):
            raise DoctorCandidateRetrievalBindingError(
                "roots must be DoctorRetrievalAuthorityRoots"
            )
        self.roots = roots
        self.bounds = bounds or DoctorCandidateRetrievalBounds()
        self.embedding_provider = embedding_provider
        self.embedding_policy = embedding_policy
        if (
            embedding_policy is not None
            and PinnedEmbeddingPolicy is not None
            and not isinstance(embedding_policy, PinnedEmbeddingPolicy)
        ):
            # Accept duck-typed policies with policy_id for forward compatibility.
            if not hasattr(embedding_policy, "policy_id"):
                raise DoctorCandidateRetrievalBindingError(
                    "embedding_policy must expose policy_id"
                )

    def _vector_lane_enabled(self) -> tuple[bool, str, str]:
        """Return (enabled, status_text, embedding_policy_id)."""
        policy_id = self.roots.embedding_policy_id
        if self.embedding_policy is not None:
            policy_id = str(
                getattr(self.embedding_policy, "policy_id", "") or policy_id
            )
        provider = self.embedding_provider
        if provider is None:
            return False, "not_probed" if not policy_id else "disabled", policy_id
        status = getattr(provider, "vector_lane", None)
        if status is not None:
            status_text = getattr(status, "value", str(status))
            enabled = bool(getattr(provider, "vector_lane_enabled", False))
            if EmbeddingLaneStatus is not None and status is EmbeddingLaneStatus.ENABLED:
                enabled = True
            return enabled, str(status_text), policy_id
        enabled = bool(getattr(provider, "vector_lane_enabled", False))
        return enabled, ("enabled" if enabled else "disabled"), policy_id

    def retrieve(
        self,
        finding_id: str,
        *,
        query: DoctorCandidateQuery | None = None,
        subject_path: str = "",
        subject_symbol: str = "",
        subject_span_ref: str = "",
        obligation_refs: Sequence[str] = (),
        expected_behavior_refs: Sequence[str] = (),
        candidates_by_signal: Mapping[str, Any] | None = None,
        **signal_candidates: Any,
    ) -> DoctorCandidateSet:
        finding_id = _text(finding_id, "finding_id")
        if query is None:
            policy_id = self.roots.embedding_policy_id
            if self.embedding_policy is not None:
                policy_id = str(
                    getattr(self.embedding_policy, "policy_id", "") or policy_id
                )
            query = DoctorCandidateQuery(
                roots=self.roots,
                finding_id=finding_id,
                subject_path=subject_path,
                subject_symbol=subject_symbol,
                subject_span_ref=subject_span_ref,
                obligation_refs=tuple(obligation_refs),
                expected_behavior_refs=tuple(expected_behavior_refs),
                embedding_policy_id=policy_id,
            )
        if not isinstance(query, DoctorCandidateQuery):
            raise DoctorCandidateRetrievalBindingError(
                "query must be DoctorCandidateQuery"
            )
        if query.roots != self.roots:
            raise DoctorCandidateRetrievalBindingError(
                "query roots do not match retriever roots"
            )
        if query.finding_id != finding_id:
            raise DoctorCandidateRetrievalBindingError(
                "query finding_id does not match finding_id"
            )

        vector_enabled, vector_status, embedding_policy_id = self._vector_lane_enabled()

        # Default signal roots: exact routes bind graph/tree; approximate bind index.
        signal_roots: dict[str, str] = {
            DoctorCandidateSignal.EXACT_SYMBOL.value: self.roots.index_id,
            DoctorCandidateSignal.EXACT_CONTRACT.value: self.roots.graph_id,
            DoctorCandidateSignal.EXACT_VALUE.value: self.roots.graph_id,
            DoctorCandidateSignal.EXACT_LINEAGE.value: self.roots.tree_id,
            DoctorCandidateSignal.EXACT_GRAPH.value: self.roots.graph_id,
            DoctorCandidateSignal.LEXICAL.value: self.roots.index_id,
            DoctorCandidateSignal.KNOWLEDGE_GRAPH.value: self.roots.graph_id,
            DoctorCandidateSignal.VECTOR.value: self.roots.index_id,
        }

        supplied = dict(candidates_by_signal or {})
        for name, value in signal_candidates.items():
            if value is not None:
                supplied[name] = value

        # Process signals in exact-first precedence so aggregate ranking is stable.
        ordered_signals = sorted(
            (( _signal(name), value) for name, value in supplied.items()),
            key=lambda pair: (_signal_rank(pair[0]), pair[0]),
        )

        grouped: dict[str, list[Any]] = {}
        for signal, value in ordered_signals:
            if value is None:
                entries: tuple[Any, ...] = ()
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray, Mapping)
            ):
                entries = tuple(value)
            else:
                entries = (value,)
            if len(entries) > self.bounds.max_candidates_per_signal:
                raise DoctorCandidateRetrievalBoundsError(
                    f"{signal} exceeds max_candidates_per_signal"
                )
            # Vector lane hard gate: still record candidates for diagnostics
            # but they will be rejected when the lane is disabled.
            grouped.setdefault(signal, []).extend(entries)

        aggregate: dict[tuple[Any, ...], dict[str, Any]] = {}
        for signal in sorted(grouped, key=lambda s: (_signal_rank(s), s)):
            entries = grouped[signal]
            if len(entries) > self.bounds.max_candidates_per_signal:
                raise DoctorCandidateRetrievalBoundsError(
                    f"{signal} exceeds max_candidates_per_signal"
                )
            for item in entries:
                raw = _mapping(item)
                if not isinstance(raw, Mapping):
                    raw = {"value": raw}
                candidate_ref = _candidate_ref(raw, query)
                path = _path_of(raw, query)
                symbol = _symbol_of(raw, query)
                had_body = _contains_body_or_secret(raw)
                safe_raw = _redact_payload(raw) if had_body else dict(raw)
                if not isinstance(safe_raw, Mapping):
                    safe_raw = {"value": safe_raw}
                safe_raw = dict(safe_raw)
                safe_raw["candidate_ref"] = candidate_ref
                safe_raw["path"] = path
                safe_raw["symbol_id"] = symbol
                key = (candidate_ref, path, symbol)
                if safe_raw.get("partial") is True or not candidate_ref:
                    key = key + (_fingerprint(safe_raw),)
                entry = aggregate.setdefault(
                    key,
                    {
                        "candidate_ref": candidate_ref,
                        "path": path,
                        "symbol": symbol,
                        "signals": set(),
                        "refs": {},
                        "reasons": set(),
                        "raw": [],
                        "kinds": set(),
                    },
                )
                entry["signals"].add(signal)
                entry["refs"].setdefault(signal, []).extend(
                    _refs(
                        safe_raw.get("evidence_refs", safe_raw.get("evidence_ref")),
                        signal,
                        safe_raw,
                    )
                )
                reasons = _diagnostics(
                    signal,
                    safe_raw,
                    path,
                    self.roots,
                    vector_lane_enabled=vector_enabled,
                    embedding_policy_id=embedding_policy_id,
                )
                if had_body:
                    reasons.add(REJECTION_BODY_OR_SECRET)
                entry["reasons"].update(reasons)
                entry["raw"].append(safe_raw)
                try:
                    entry["kinds"].add(_infer_kind(safe_raw, entry["signals"]).value)
                except DoctorCandidateRetrievalError:
                    entry["reasons"].add(REJECTION_INVALID_PAYLOAD)

        if not aggregate:
            raw = {
                "partial": True,
                "reason": "no_signal_candidates",
                "candidate_ref": f"missing:{finding_id}",
            }
            aggregate[("empty", finding_id)] = {
                "candidate_ref": raw["candidate_ref"],
                "path": query.subject_path,
                "symbol": query.subject_symbol,
                "signals": set(),
                "refs": {},
                "reasons": {REJECTION_PARTIAL},
                "raw": [raw],
                "kinds": set(),
            }
        if len(aggregate) > self.bounds.max_candidates:
            raise DoctorCandidateRetrievalBoundsError(
                "unioned candidate set exceeds max_candidates; refusing partial union"
            )

        nominations: list[DoctorCandidateNomination] = []
        for entry in aggregate.values():
            signals = set(entry["signals"])
            reasons = set(entry["reasons"])
            # Disabling the optional vector lane must not poison candidates that
            # also have exact/lexical/KG evidence.  Drop the vector contribution
            # and keep the rest of the nomination.
            if (
                REJECTION_VECTOR_LANE_DISABLED in reasons
                and signals - {DoctorCandidateSignal.VECTOR.value}
            ):
                reasons.discard(REJECTION_VECTOR_LANE_DISABLED)
                signals.discard(DoctorCandidateSignal.VECTOR.value)
                entry["refs"].pop(DoctorCandidateSignal.VECTOR.value, None)
                entry["signals"] = signals
            raw = min(entry["raw"], key=_fingerprint)
            kind = _infer_kind(raw, signals)
            if len(entry["kinds"]) > 1:
                # Conflicting kind labels remain nominated with unknown kind
                # rather than inventing a winner.
                kind = DoctorCandidateKind.UNKNOWN

            # Build evidence facts separately from scores.
            all_refs = tuple(
                sorted(
                    {ref for refs in entry["refs"].values() for ref in refs},
                    key=lambda ref: ref.content_id,
                )
            )
            primary = (
                min(signals, key=_signal_rank)
                if signals
                else DoctorCandidateSignal.EXACT_SYMBOL.value
            )
            source_authority = _source_authority(
                raw.get("source_authority"), primary
            )
            # Approximate-only candidates cannot claim reviewed/authoritative.
            if signals and signals.issubset(APPROXIMATE_SIGNALS):
                source_authority = DoctorSourceAuthority.NOMINATED

            candidate_cid = str(
                raw.get("candidate_cid")
                or raw.get("cid")
                or raw.get("content_id")
                or ""
            )
            provisional_ref = entry["candidate_ref"]
            # Evidence CID is bound after candidate construction if missing.

            score = _score_millionths(raw)
            # Do not attach scores for rejected candidates — facts only.
            if reasons:
                score = None

            hard_compat = _hard_compatible_fact(raw)
            info_ref = str(raw.get("information_content_ref", "") or "")

            rank = _eligibility_rank(signals, reasons)

            # Build a provisional candidate to obtain content_id when needed.
            provisional = DoctorRepairCandidate(
                roots=self.roots,
                finding_id=finding_id,
                candidate_ref=provisional_ref,
                kind=kind,
                path=entry["path"],
                symbol_id=entry["symbol"],
                evidence=None,
                diagnostics=(),
                semantic_authority=False,
            )
            if not candidate_cid:
                candidate_cid = provisional.content_id

            evidence = DoctorCandidateEvidence(
                candidate_cid=candidate_cid,
                source_authority=source_authority,
                hard_compatible=hard_compat,
                information_content_ref=info_ref,
                signal_refs=all_refs,
                primary_signal=primary if signals else "",
                score_millionths=score,
                notes=tuple(raw.get("notes", ()) or ()),
            )
            candidate = DoctorRepairCandidate(
                roots=self.roots,
                finding_id=finding_id,
                candidate_ref=provisional_ref,
                kind=kind,
                path=entry["path"],
                symbol_id=entry["symbol"],
                evidence=evidence,
                diagnostics=tuple(sorted(reasons)),
                semantic_authority=False,
            )
            signal_evidence = tuple(
                (
                    signal,
                    tuple(sorted(set(refs), key=lambda ref: ref.content_id)),
                )
                for signal, refs in sorted(
                    entry["refs"].items(),
                    key=lambda item: (_signal_rank(item[0]), item[0]),
                )
            )
            nominations.append(
                DoctorCandidateNomination(
                    candidate=candidate,
                    disposition=(
                        DoctorCandidateDisposition.REJECTED
                        if reasons
                        else DoctorCandidateDisposition.NOMINATED
                    ),
                    signal_evidence=signal_evidence,
                    diagnostics=tuple(sorted(reasons)),
                    eligibility_rank=rank,
                    semantic_authority=False,
                )
            )

        nominations.sort(key=lambda item: item.content_id)
        eligibility_status, equally_eligible = _compute_eligibility(nominations)
        candidates = tuple(nominations)
        return DoctorCandidateSet(
            roots=self.roots,
            query=query,
            finding_id=finding_id,
            bounds=self.bounds,
            candidates=candidates,
            candidate_set_id=candidate_set_identity(
                tuple(item.candidate for item in candidates)
            ),
            eligibility_status=eligibility_status,
            signal_roots=tuple(
                sorted(signal_roots.items(), key=lambda item: (_signal_rank(item[0]), item[0]))
            ),
            vector_lane_status=vector_status,
            embedding_policy_id=embedding_policy_id,
            vector_query_id="",
            equally_eligible_ids=equally_eligible,
            semantic_authority=False,
        )

    nominate = retrieve
    search = retrieve


def retrieve_doctor_repair_candidates(
    roots: DoctorRetrievalAuthorityRoots,
    finding_id: str,
    **kwargs: Any,
) -> DoctorCandidateSet:
    """Stateless convenience entry point for the retrieval-only boundary."""
    bounds = kwargs.pop("bounds", None)
    embedding_provider = kwargs.pop("embedding_provider", None)
    embedding_policy = kwargs.pop("embedding_policy", None)
    return DoctorRepairCandidateRetriever(
        roots,
        bounds=bounds,
        embedding_provider=embedding_provider,
        embedding_policy=embedding_policy,
    ).retrieve(finding_id, **kwargs)


__all__ = (
    "DOCTOR_CANDIDATE_QUERY_SCHEMA",
    "DOCTOR_CANDIDATE_EVIDENCE_SCHEMA",
    "DOCTOR_CANDIDATE_SCHEMA",
    "DOCTOR_CANDIDATE_NOMINATION_SCHEMA",
    "DOCTOR_CANDIDATE_SET_SCHEMA",
    "DOCTOR_CANDIDATE_BOUNDS_SCHEMA",
    "DOCTOR_SIGNAL_REF_SCHEMA",
    "DOCTOR_AUTHORITY_ROOTS_SCHEMA",
    "SIGNAL_FAMILIES",
    "SIGNAL_PRECEDENCE",
    "EXACT_SIGNALS",
    "APPROXIMATE_SIGNALS",
    "PRODUCER_ID",
    "DoctorCandidateRetrievalError",
    "DoctorCandidateRetrievalBindingError",
    "DoctorCandidateRetrievalBoundsError",
    "DoctorCandidateSignal",
    "DoctorCandidateKind",
    "DoctorCandidateDisposition",
    "DoctorEligibilityStatus",
    "DoctorSourceAuthority",
    "DoctorRetrievalAuthorityRoots",
    "DoctorCandidateSignalRef",
    "DoctorCandidateRetrievalBounds",
    "DoctorCandidateQuery",
    "DoctorCandidateEvidence",
    "DoctorRepairCandidate",
    "DoctorCandidateNomination",
    "DoctorCandidateSet",
    "DoctorRepairCandidateRetriever",
    "retrieve_doctor_repair_candidates",
    "candidate_set_identity",
    "REJECTION_STALE_OR_CROSS_TREE",
    "REJECTION_GENERATED",
    "REJECTION_READ_ONLY",
    "REJECTION_POISONED",
    "REJECTION_FORGED",
    "REJECTION_PARTIAL",
    "REJECTION_BODY_OR_SECRET",
    "REJECTION_SEMANTIC_AUTHORITY_CLAIM",
    "REJECTION_COMPATIBILITY_CLAIM",
    "REJECTION_WRITE_SCOPE_CLAIM",
    "REJECTION_PLACEMENT_CLAIM",
    "REJECTION_TARGET_CLAIM",
    "REJECTION_VALUE_AUTHORITY_CLAIM",
    "REJECTION_VECTOR_LANE_DISABLED",
    "REJECTION_UNPINNED_EMBEDDING",
    "REJECTION_INVALID_PAYLOAD",
    "REJECTION_SCORE_BEFORE_FILTER",
)
