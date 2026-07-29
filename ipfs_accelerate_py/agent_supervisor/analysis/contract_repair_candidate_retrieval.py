"""Fail-closed, non-authoritative nomination of contract-repair candidates.

This adapter is deliberately a *recall* boundary.  It joins the independent
history, structural, resolver, ownership, AST, lexical, and vector signals
into one canonical snapshot, but never ranks a winner or grants an edit path.
Later proof/admission code must consume the complete receipt rather than an
individual result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, ClassVar, Iterable

from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from .analysis_retrieval import BoundRetrievalCandidate
from .code_symbol_vector_index import (
    CodeSymbolIndexRow,
    CodeVectorHit,
    CodeVectorIndexSnapshot,
    CodeVectorQuery,
    CodeVectorSearchResult,
)
from .contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    CallRequirementContract,
    ContractRepairError,
    EvidenceReference,
    MAX_CANDIDATE_COUNT,
    MemorySafetyFacet,
    RepairCandidate,
    RepairStrategy,
    SourceSpan,
    TraceDisposition,
    candidate_set_identity,
)


CANDIDATE_NOMINATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-candidate-nomination@1"
)
CANDIDATE_NOMINATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-candidate-nomination-receipt@1"
)
CANDIDATE_RETRIEVAL_BOUNDS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-candidate-retrieval-bounds@1"
)


class CandidateRetrievalError(ContractRepairError):
    """A candidate signal cannot safely participate in a nomination."""


class CandidateRetrievalBindingError(CandidateRetrievalError):
    """A required trace, contract, facet, or vector root was mixed."""


class CandidateRetrievalBoundsError(CandidateRetrievalError):
    """A producer attempted to exceed the fixed retrieval budget."""


class CandidateSignal(str, Enum):
    EXACT_HISTORY = "exact_history"
    STRUCTURAL_FINGERPRINT = "structural_fingerprint"
    RESOLVER_ROUTE = "resolver_route"
    DEPENDENCY_OWNERSHIP = "dependency_ownership"
    AST = "ast"
    LEXICAL = "lexical"
    VECTOR = "vector"


class CandidateDisposition(str, Enum):
    NOMINATED = "nominated"
    REJECTED = "rejected"
    DIAGNOSTIC = "rejected"  # Compatibility spelling; diagnostics are rejected.


SIGNAL_FAMILIES = tuple(item.value for item in CandidateSignal)
_SIGNAL_ALIASES = {
    "history": CandidateSignal.EXACT_HISTORY.value,
    "exact": CandidateSignal.EXACT_HISTORY.value,
    "structural": CandidateSignal.STRUCTURAL_FINGERPRINT.value,
    "fingerprint": CandidateSignal.STRUCTURAL_FINGERPRINT.value,
    "resolver": CandidateSignal.RESOLVER_ROUTE.value,
    "route": CandidateSignal.RESOLVER_ROUTE.value,
    "dependency": CandidateSignal.DEPENDENCY_OWNERSHIP.value,
    "ownership": CandidateSignal.DEPENDENCY_OWNERSHIP.value,
    "ast_symbol": CandidateSignal.AST.value,
    "symbol": CandidateSignal.AST.value,
    "bm25": CandidateSignal.LEXICAL.value,
}

# These values are public diagnostics: do not change them without a versioned
# receipt schema.  Consumers use them to distinguish a retry from an
# inadmissible target.
REJECTION_SAME_NAME_INCOMPATIBLE = "same_name_incompatible"
REJECTION_POISONED_VECTOR = "poisoned_vector"
REJECTION_STALE_OR_CROSS_TREE = "stale_or_cross_tree"
REJECTION_READ_ONLY_TARGET = "read_only_target"
REJECTION_GENERATED_VENDOR_ARCHIVE_TARGET = "generated_vendor_archive_target"
REJECTION_FORBIDDEN_LAYER = "forbidden_layer"
REJECTION_PARTIAL_CANDIDATE = "partial_candidate"
REJECTION_FORGED_HISTORY = "forged_history"
REJECTION_CONFLICTING_STRATEGY_SIGNALS = "conflicting_strategy_signals"
REJECTION_INVALID_CANDIDATE_PAYLOAD = "invalid_candidate_payload"

_BODY_FIELDS = frozenset({
    "source", "source_body", "source_text", "body", "content", "contents",
    "text", "code", "raw", "raw_text", "ast", "ast_body", "embedding",
    "query_vector", "model_output", "completion", "prompt",
})
_GENERATED_PARTS = frozenset({"vendor", "vendors", "node_modules", "third_party", "archive", "archives", "generated", "build", "dist"})


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, CanonicalContract):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        converter = getattr(value, "to_dict", None)
        return _canonical(converter() if callable(converter) else vars(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return "<non-finite>"
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "candidate-input:sha256:" + hashlib.sha256(encoded).hexdigest()


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


def _contains_body(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).casefold().replace("-", "_") in _BODY_FIELDS or _contains_body(item) for key, item in value.items())
    return isinstance(value, (bytes, bytearray)) or (
        isinstance(value, Sequence) and not isinstance(value, str) and any(_contains_body(item) for item in value)
    )


def _refs(value: Any, signal: str, raw: Mapping[str, Any]) -> tuple[EvidenceReference, ...]:
    values: Iterable[Any]
    if value is None:
        values = ()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        values = value
    else:
        values = (value,)
    refs: list[EvidenceReference] = []
    for item in values:
        try:
            if isinstance(item, EvidenceReference):
                ref = item
            elif isinstance(item, Mapping):
                ref = EvidenceReference(**{key: item[key] for key in ("kind", "artifact_id", "locator", "producer_id") if key in item})
            elif isinstance(item, str) and item.strip():
                ref = EvidenceReference(signal, item.strip(), producer_id="contract-repair-candidate-retrieval@1")
            else:
                continue
        except (KeyError, ContractRepairError, TypeError):
            continue
        if ref not in refs:
            refs.append(ref)
    if not refs:
        refs.append(EvidenceReference(signal, _fingerprint(raw), producer_id="contract-repair-candidate-retrieval@1"))
    return tuple(sorted(refs, key=lambda item: item.content_id))


def _signal(name: Any) -> str:
    normalized = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = _SIGNAL_ALIASES.get(normalized, normalized)
    if normalized not in SIGNAL_FAMILIES:
        raise CandidateRetrievalError("unsupported candidate signal: " + str(name))
    return normalized


def _verify_record_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    claimed = payload.get("content_id", payload.get("cid", ""))
    if claimed not in (None, "", record.content_id):
        raise CandidateRetrievalBindingError("stored content identity does not match the canonical record")


@dataclass(frozen=True)
class CandidateRetrievalBounds(CanonicalContract):
    """Fixed, replayable caps; over-budget input is rejected, never truncated."""

    SCHEMA: ClassVar[str] = CANDIDATE_RETRIEVAL_BOUNDS_SCHEMA

    max_candidates: int = MAX_CANDIDATE_COUNT
    max_candidates_per_signal: int = 64

    def __post_init__(self) -> None:
        for name in ("max_candidates", "max_candidates_per_signal"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_CANDIDATE_COUNT:
                raise CandidateRetrievalBoundsError(f"{name} must be an integer from 1 through {MAX_CANDIDATE_COUNT}")

    def _payload(self) -> dict[str, Any]:
        return {"max_candidates": self.max_candidates, "max_candidates_per_signal": self.max_candidates_per_signal}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateRetrievalBounds":
        allowed = {"schema", "content_id", "cid", "max_candidates", "max_candidates_per_signal"}
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA or set(payload).difference(allowed):
            raise CandidateRetrievalError("unsupported candidate retrieval bounds payload")
        value = cls(
            max_candidates=payload.get("max_candidates", MAX_CANDIDATE_COUNT),
            max_candidates_per_signal=payload.get("max_candidates_per_signal", 64),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class CandidateNomination(CanonicalContract):
    """One candidate plus complete per-signal provenance and no authority."""

    SCHEMA: ClassVar[str] = CANDIDATE_NOMINATION_SCHEMA

    candidate: RepairCandidate
    disposition: CandidateDisposition
    signal_evidence: tuple[tuple[str, tuple[EvidenceReference, ...]], ...]
    diagnostics: tuple[str, ...] = ()
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, RepairCandidate):
            raise CandidateRetrievalError("candidate nomination requires RepairCandidate")
        if self.candidate.candidate_write_paths or self.candidate.permitted_read_paths:
            raise CandidateRetrievalBindingError("candidate retrieval cannot emit path authority")
        object.__setattr__(self, "disposition", CandidateDisposition(self.disposition))
        rows: list[tuple[str, tuple[EvidenceReference, ...]]] = []
        raw_evidence = self.signal_evidence.items() if isinstance(self.signal_evidence, Mapping) else self.signal_evidence
        for item in raw_evidence:
            try:
                signal, refs = item
            except (TypeError, ValueError) as exc:
                raise CandidateRetrievalError("signal evidence rows must contain signal and references") from exc
            normalized = _signal(signal)
            checked = _refs(refs, normalized, {"candidate": self.candidate.content_id})
            rows.append((normalized, checked))
        rows.sort(key=lambda item: item[0])
        if len({item[0] for item in rows}) != len(rows):
            raise CandidateRetrievalError("candidate nomination has duplicate signal evidence")
        object.__setattr__(self, "signal_evidence", tuple(rows))
        diagnostics = tuple(sorted({str(item).strip() for item in self.diagnostics if str(item).strip()}))
        object.__setattr__(self, "diagnostics", diagnostics)
        if self.semantic_authority is not False:
            raise CandidateRetrievalBindingError("candidate nomination cannot claim semantic authority")
        object.__setattr__(self, "semantic_authority", False)
        if self.disposition is CandidateDisposition.NOMINATED and diagnostics:
            raise CandidateRetrievalError("nominated candidates cannot carry rejection diagnostics")
        if self.disposition is CandidateDisposition.REJECTED and not diagnostics:
            raise CandidateRetrievalError("rejected candidates require stable diagnostics")

    @property
    def target_span(self) -> SourceSpan:
        return self.candidate.target_span

    @property
    def strategy(self) -> RepairStrategy:
        return self.candidate.strategy

    @property
    def evidence_refs(self) -> tuple[EvidenceReference, ...]:
        return self.candidate.evidence_refs

    @property
    def candidate_id(self) -> str:
        return self.candidate.content_id

    @property
    def write_paths(self) -> tuple[str, ...]:
        return ()

    def _payload(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(), "disposition": self.disposition.value,
            "signal_evidence": [{"signal": signal, "evidence_refs": [ref.to_dict() for ref in refs]} for signal, refs in self.signal_evidence],
            "diagnostics": list(self.diagnostics), "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateNomination":
        allowed = {"schema", "content_id", "cid", "candidate", "disposition", "signal_evidence", "diagnostics", "semantic_authority"}
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA or set(payload).difference(allowed):
            raise CandidateRetrievalError("unsupported candidate nomination payload")
        signal_evidence: list[tuple[str, tuple[EvidenceReference, ...]]] = []
        supplied = payload.get("signal_evidence", ())
        if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes, bytearray)):
            raise CandidateRetrievalError("signal_evidence must be a sequence")
        for row in supplied:
            if not isinstance(row, Mapping):
                raise CandidateRetrievalError("signal evidence row must be an object")
            refs = row.get("evidence_refs", ())
            if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
                raise CandidateRetrievalError("signal evidence references must be a sequence")
            signal_evidence.append((str(row.get("signal", "")), tuple(
                item if isinstance(item, EvidenceReference) else EvidenceReference.from_dict(item) for item in refs
            )))
        candidate = payload.get("candidate")
        value = cls(
            candidate=candidate if isinstance(candidate, RepairCandidate) else RepairCandidate.from_dict(candidate),
            disposition=payload.get("disposition", ""), signal_evidence=tuple(signal_evidence),
            diagnostics=tuple(payload.get("diagnostics", ())), semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class CandidateNominationReceipt(CanonicalContract):
    """The complete bounded candidate set; this is not a target decision."""

    SCHEMA: ClassVar[str] = CANDIDATE_NOMINATION_RECEIPT_SCHEMA

    roots: AuthorityRoots
    trace_id: str
    call_requirement_id: str
    memory_safety_facet_id: str
    bounds: CandidateRetrievalBounds
    candidates: tuple[CandidateNomination, ...]
    candidate_set_id: str
    signal_roots: tuple[tuple[str, str], ...] = ()
    vector_query_id: str = ""
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots) or not isinstance(self.bounds, CandidateRetrievalBounds):
            raise CandidateRetrievalError("receipt roots and bounds must be canonical")
        for name in ("trace_id", "call_requirement_id", "memory_safety_facet_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise CandidateRetrievalError(f"{name} is required")
        candidates = tuple(sorted(self.candidates, key=lambda item: item.content_id))
        if not candidates or len(candidates) > self.bounds.max_candidates:
            raise CandidateRetrievalBoundsError("receipt candidate count is outside its declared bound")
        if any(not isinstance(item, CandidateNomination) for item in candidates):
            raise CandidateRetrievalError("receipt candidates must be nominations")
        if len({item.content_id for item in candidates}) != len(candidates):
            raise CandidateRetrievalError("receipt contains duplicate nominations")
        if any(item.candidate.roots != self.roots for item in candidates):
            raise CandidateRetrievalBindingError("candidate roots do not match receipt roots")
        object.__setattr__(self, "candidates", candidates)
        expected = candidate_set_identity(tuple(item.candidate for item in candidates))
        if self.candidate_set_id != expected:
            raise CandidateRetrievalBindingError("candidate_set_id does not bind the complete candidate set")
        roots: list[tuple[str, str]] = []
        for signal, root in self.signal_roots:
            normalized = _signal(signal)
            if not isinstance(root, str) or not root:
                raise CandidateRetrievalBindingError("signal roots must be nonempty identities")
            roots.append((normalized, root))
        roots.sort()
        if len({item[0] for item in roots}) != len(roots):
            raise CandidateRetrievalBindingError("receipt contains duplicate signal roots")
        object.__setattr__(self, "signal_roots", tuple(roots))
        if self.semantic_authority is not False:
            raise CandidateRetrievalBindingError("retrieval receipts cannot claim semantic authority")
        object.__setattr__(self, "semantic_authority", False)

    @property
    def repair_candidates(self) -> tuple[RepairCandidate, ...]:
        return tuple(item.candidate for item in self.candidates)

    @property
    def write_paths(self) -> tuple[str, ...]:
        """Retrieval never provides mutation authority."""
        return ()

    @property
    def admitted_candidate_id(self) -> str:
        """There is deliberately no winner at retrieval time."""
        return ""

    def _payload(self) -> dict[str, Any]:
        return {
            "roots": self.roots.to_dict(), "trace_id": self.trace_id,
            "call_requirement_id": self.call_requirement_id,
            "memory_safety_facet_id": self.memory_safety_facet_id,
            "bounds": self.bounds.to_dict(), "candidates": [item.to_dict() for item in self.candidates],
            "candidate_set_id": self.candidate_set_id,
            "signal_roots": [{"signal": signal, "root_id": root} for signal, root in self.signal_roots],
            "vector_query_id": self.vector_query_id, "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateNominationReceipt":
        allowed = {
            "schema", "content_id", "cid", "roots", "trace_id", "call_requirement_id",
            "memory_safety_facet_id", "bounds", "candidates", "candidate_set_id",
            "signal_roots", "vector_query_id", "semantic_authority",
        }
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA or set(payload).difference(allowed):
            raise CandidateRetrievalError("unsupported candidate nomination receipt payload")
        rows = payload.get("signal_roots", ())
        candidates = payload.get("candidates", ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)) or not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
            raise CandidateRetrievalError("receipt signal roots and candidates must be sequences")
        signal_roots: list[tuple[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise CandidateRetrievalError("receipt signal root row must be an object")
            signal_roots.append((str(row.get("signal", "")), str(row.get("root_id", ""))))
        roots = payload.get("roots")
        bounds = payload.get("bounds")
        value = cls(
            roots=roots if isinstance(roots, AuthorityRoots) else AuthorityRoots.from_dict(roots),
            trace_id=payload.get("trace_id", ""), call_requirement_id=payload.get("call_requirement_id", ""),
            memory_safety_facet_id=payload.get("memory_safety_facet_id", ""),
            bounds=bounds if isinstance(bounds, CandidateRetrievalBounds) else CandidateRetrievalBounds.from_dict(bounds),
            candidates=tuple(item if isinstance(item, CandidateNomination) else CandidateNomination.from_dict(item) for item in candidates),
            candidate_set_id=payload.get("candidate_set_id", ""), signal_roots=tuple(signal_roots),
            vector_query_id=payload.get("vector_query_id", ""), semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


def _span(raw: Mapping[str, Any], fallback: BrokenContractTrace) -> tuple[SourceSpan, bool]:
    value = raw.get("target_span", raw.get("span"))
    try:
        if isinstance(value, SourceSpan):
            return value, False
        if isinstance(value, Mapping):
            return SourceSpan(**{key: value[key] for key in ("path", "start", "end", "artifact_id")}), False
        row = raw.get("row")
        if isinstance(row, CodeSymbolIndexRow):
            return SourceSpan(row.path, row.line_start, row.line_end, row.sidecar.blob_identity), False
        if isinstance(row, Mapping):
            sidecar = row.get("sidecar") or {}
            return SourceSpan(str(row["path"]), int(row.get("line_start", 0)), int(row.get("line_end", 0)), str(sidecar.get("blob_identity", ""))), False
        if all(name in raw for name in ("path", "start", "end", "artifact_id")):
            return SourceSpan(str(raw["path"]), int(raw["start"]), int(raw["end"]), str(raw["artifact_id"])), False
    except (KeyError, TypeError, ValueError, ContractRepairError):
        pass
    # A rejected diagnostic still needs a typed, body-free anchor.  The unique
    # artifact id prevents unrelated partial candidates from being deduplicated.
    return SourceSpan(fallback.caller_span.path, fallback.caller_span.start, fallback.caller_span.end, "partial:" + _fingerprint(raw).split(":")[-1]), True


def _strategy(trace: BrokenContractTrace, raw: Mapping[str, Any], signals: set[str]) -> RepairStrategy:
    supplied = raw.get("strategy", "")
    if supplied:
        try:
            return RepairStrategy(supplied)
        except ValueError:
            return RepairStrategy.AMBIGUOUS
    if raw.get("existing_declaration") is True:
        return RepairStrategy.IMPLEMENT_EXISTING_DECLARATION
    if raw.get("new_site") is True or raw.get("implementation_anchor") is True:
        return RepairStrategy.NEW_IMPLEMENTATION
    if trace.disposition is TraceDisposition.ADAPTER_REQUIRED or raw.get("adapter_mapping") is True:
        return RepairStrategy.ADAPTER
    if trace.disposition is TraceDisposition.LIKELY_REFACTOR and CandidateSignal.EXACT_HISTORY.value in signals:
        return RepairStrategy.RENAME_SUBSTITUTION
    if trace.disposition is TraceDisposition.RESOLVED_MISMATCH:
        return RepairStrategy.IMPLEMENT_EXISTING_DECLARATION
    return RepairStrategy.AMBIGUOUS


def _diagnostics(
    signal: str, raw: Mapping[str, Any], path: str, span_partial: bool,
    expected_roots: AuthorityRoots, vector_roots: tuple[str, str, str] | None,
) -> set[str]:
    reasons: set[str] = set()
    if span_partial or raw.get("partial") is True or raw.get("complete") is False:
        reasons.add(REJECTION_PARTIAL_CANDIDATE)
    if _contains_body(raw):
        reasons.add(REJECTION_INVALID_CANDIDATE_PAYLOAD)
    if raw.get("same_name") is True and raw.get("signature_compatible") is False:
        reasons.add(REJECTION_SAME_NAME_INCOMPATIBLE)
    if raw.get("compatible") is False:
        reasons.add(REJECTION_SAME_NAME_INCOMPATIBLE)
    if raw.get("read_only") is True or raw.get("writable") is False:
        reasons.add(REJECTION_READ_ONLY_TARGET)
    parts = {part.casefold() for part in path.split("/")}
    if raw.get("generated") is True or raw.get("vendor") is True or raw.get("archive") is True or parts.intersection(_GENERATED_PARTS):
        reasons.add(REJECTION_GENERATED_VENDOR_ARCHIVE_TARGET)
    if raw.get("forbidden_layer") is True or raw.get("layer_allowed") is False:
        reasons.add(REJECTION_FORBIDDEN_LAYER)
    if raw.get("forged_history") is True or raw.get("history_reviewed") is False:
        reasons.add(REJECTION_FORGED_HISTORY)
    for key in ("tree_id", "graph_id", "index_id", "model_id", "config_id"):
        if key in raw and raw[key] != getattr(expected_roots, key):
            reasons.add(REJECTION_STALE_OR_CROSS_TREE)
    candidate_roots = raw.get("roots")
    if isinstance(candidate_roots, Mapping):
        if any(key in candidate_roots and candidate_roots[key] != getattr(expected_roots, key) for key in ("tree_id", "graph_id", "index_id", "model_id", "config_id")):
            reasons.add(REJECTION_STALE_OR_CROSS_TREE)
    binding = raw.get("binding")
    if isinstance(binding, Mapping) and vector_roots is not None:
        tree_id, config_id, model_id = vector_roots
        if binding.get("graph_root_id") not in (None, "", tree_id) or binding.get("configuration_id") not in (None, "", config_id) or binding.get("model_id") not in (None, "", model_id):
            reasons.add(REJECTION_STALE_OR_CROSS_TREE)
    if signal == CandidateSignal.VECTOR.value:
        try:
            score = raw.get("score", raw.get("score_millionths", 0))
            if not math.isfinite(float(score)) or raw.get("semantic_authority", False) is not False:
                reasons.add(REJECTION_POISONED_VECTOR)
        except (TypeError, ValueError):
            reasons.add(REJECTION_POISONED_VECTOR)
        if vector_roots is not None:
            tree_id, config_id, model_id = vector_roots
            if raw.get("tree_id") not in (None, "", tree_id) or raw.get("config_id") not in (None, "", config_id) or raw.get("model_id") not in (None, "", model_id):
                reasons.add(REJECTION_STALE_OR_CROSS_TREE)
    return reasons


class ContractRepairCandidateRetriever:
    """Union bounded signal families into a diagnostic-only candidate receipt."""

    def __init__(self, roots: AuthorityRoots, *, bounds: CandidateRetrievalBounds | None = None) -> None:
        if not isinstance(roots, AuthorityRoots):
            raise CandidateRetrievalBindingError("roots must be AuthorityRoots")
        self.roots = roots
        self.bounds = bounds or CandidateRetrievalBounds()

    def retrieve(
        self,
        trace: BrokenContractTrace,
        call_requirement: CallRequirementContract,
        memory_safety_facet: MemorySafetyFacet,
        *,
        candidates_by_signal: Mapping[str, Any] | None = None,
        code_index: CodeVectorIndexSnapshot | None = None,
        vector_query: CodeVectorQuery | None = None,
        **signal_candidates: Any,
    ) -> CandidateNominationReceipt:
        if not isinstance(trace, BrokenContractTrace) or not isinstance(call_requirement, CallRequirementContract) or not isinstance(memory_safety_facet, MemorySafetyFacet):
            raise CandidateRetrievalBindingError("trace, call requirement, and memory facet must be typed contracts")
        if trace.roots != self.roots or call_requirement.roots != self.roots or memory_safety_facet.roots != self.roots:
            raise CandidateRetrievalBindingError("trace, call requirement, memory facet, and retriever must share exact roots")
        if call_requirement.trace_id != trace.content_id:
            raise CandidateRetrievalBindingError("call requirement does not bind the supplied trace")
        if memory_safety_facet.subject_span.path != trace.caller_span.path and memory_safety_facet.subject_span != trace.target_span:
            raise CandidateRetrievalBindingError("memory facet is not scoped to the trace caller or target")

        vector_roots: tuple[str, str, str] | None = None
        # Each signal points at the immutable root that actually constrains it;
        # the enclosing AuthorityRoots still binds the complete tree/graph/
        # index/model/config tuple for every replay.
        signal_roots: dict[str, str] = {
            CandidateSignal.EXACT_HISTORY.value: self.roots.tree_id,
            CandidateSignal.STRUCTURAL_FINGERPRINT.value: self.roots.tree_id,
            CandidateSignal.RESOLVER_ROUTE.value: self.roots.graph_id,
            CandidateSignal.DEPENDENCY_OWNERSHIP.value: self.roots.graph_id,
            CandidateSignal.AST.value: self.roots.index_id,
            CandidateSignal.LEXICAL.value: self.roots.index_id,
            CandidateSignal.VECTOR.value: self.roots.index_id,
        }
        query_id = ""
        if code_index is not None:
            if not isinstance(code_index, CodeVectorIndexSnapshot):
                raise CandidateRetrievalBindingError("code_index must be a canonical CodeVectorIndexSnapshot")
            if (code_index.forest_id, code_index.tree_id, code_index.index_id, code_index.config.model_id, code_index.config.config_id) != (self.roots.forest_id, self.roots.tree_id, self.roots.index_id, self.roots.model_id, self.roots.config_id):
                raise CandidateRetrievalBindingError("code index does not bind the receipt forest/tree/index/model/config roots")
            vector_roots = (code_index.tree_id, code_index.config.config_id, code_index.config.model_id)
            signal_roots[CandidateSignal.VECTOR.value] = code_index.index_id
        if vector_query is not None:
            if not isinstance(vector_query, CodeVectorQuery) or vector_query.semantic_authority is not False:
                raise CandidateRetrievalBindingError("vector query must be canonical and non-authoritative")
            if code_index is None or (vector_query.forest_id, vector_query.tree_id, vector_query.index_id, vector_query.config_id) != (code_index.forest_id, code_index.tree_id, code_index.index_id, code_index.config.config_id):
                raise CandidateRetrievalBindingError("vector query does not bind the supplied code index")
            query_id = vector_query.query_id

        supplied = dict(candidates_by_signal or {})
        for name, value in signal_candidates.items():
            if value is not None:
                supplied[name] = value
        grouped: dict[str, list[Any]] = {}
        for raw_signal, value in supplied.items():
            signal = _signal(raw_signal)
            if isinstance(value, CodeVectorSearchResult):
                if signal != CandidateSignal.VECTOR.value or value.semantic_authority is not False or value.complete is not True:
                    raise CandidateRetrievalBindingError("vector results must be complete, non-authoritative vector evidence")
                if code_index is not None and value.index_id != code_index.index_id:
                    raise CandidateRetrievalBindingError("vector result index differs from code index")
                grouped.setdefault(signal, []).extend(value.hits)
                query_id = value.query.query_id
                continue
            if value is None:
                entries: tuple[Any, ...] = ()
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
                entries = tuple(value)
            else:
                entries = (value,)
            if len(entries) > self.bounds.max_candidates_per_signal:
                raise CandidateRetrievalBoundsError(f"{signal} exceeds max_candidates_per_signal")
            grouped.setdefault(signal, []).extend(entries)

        aggregate: dict[tuple[Any, ...], dict[str, Any]] = {}
        for signal in sorted(grouped):
            entries = grouped[signal]
            if len(entries) > self.bounds.max_candidates_per_signal:
                raise CandidateRetrievalBoundsError(f"{signal} exceeds max_candidates_per_signal")
            for item in entries:
                raw = _mapping(item)
                if isinstance(item, CodeVectorHit):
                    raw = item.to_dict()
                elif isinstance(item, CodeSymbolIndexRow):
                    raw = {"row": item}
                elif isinstance(item, BoundRetrievalCandidate):
                    raw = item.to_dict()
                elif isinstance(item, RepairCandidate):
                    raw = {"repair_candidate": item, "target_span": item.target_span, "strategy": item.strategy.value, "evidence_refs": item.evidence_refs}
                span, span_partial = _span(raw, trace)
                key = (span.path, span.start, span.end, span.artifact_id)
                if span_partial:
                    key += (_fingerprint(raw),)
                entry = aggregate.setdefault(key, {"span": span, "signals": set(), "refs": {}, "reasons": set(), "raw": []})
                entry["signals"].add(signal)
                entry["refs"].setdefault(signal, []).extend(_refs(raw.get("evidence_refs", raw.get("evidence_ref")), signal, raw))
                entry["reasons"].update(_diagnostics(signal, raw, span.path, span_partial, self.roots, vector_roots))
                entry["raw"].append(raw)

        if not aggregate:
            # Empty retrieval is a valid, explicit diagnostic rather than an
            # implicit winner.  The caller span is only an audit anchor.
            raw = {"partial": True, "reason": "no_signal_candidates"}
            span, _ = _span(raw, trace)
            aggregate[(span.path, span.start, span.end, span.artifact_id, "empty")] = {"span": span, "signals": set(), "refs": {}, "reasons": {REJECTION_PARTIAL_CANDIDATE}, "raw": [raw]}
        if len(aggregate) > self.bounds.max_candidates:
            raise CandidateRetrievalBoundsError("unioned candidate set exceeds max_candidates; refusing partial union")

        nominations: list[CandidateNomination] = []
        for entry in aggregate.values():
            signals = set(entry["signals"])
            reasons = set(entry["reasons"])
            strategy_values = {raw.get("strategy") for raw in entry["raw"] if raw.get("strategy")}
            if len(strategy_values) > 1:
                reasons.add(REJECTION_CONFLICTING_STRATEGY_SIGNALS)
            raw = min(entry["raw"], key=_fingerprint)
            # A rejection records why the proposal cannot advance, but does
            # not erase its placement classification. Only malformed,
            # partial, or poisoned payloads lack enough target facts to name a
            # strategy at all.
            strategy = _strategy(trace, raw, signals)
            if reasons.intersection({REJECTION_PARTIAL_CANDIDATE, REJECTION_INVALID_CANDIDATE_PAYLOAD, REJECTION_POISONED_VECTOR}):
                strategy = RepairStrategy.REJECT
            evidence = tuple(sorted({ref for refs in entry["refs"].values() for ref in refs} | set(trace.evidence_refs) | set(call_requirement.evidence_refs), key=lambda ref: ref.content_id))
            candidate = RepairCandidate(
                roots=self.roots, trace_id=trace.content_id, strategy=strategy,
                target_span=entry["span"], evidence_refs=evidence,
                proof_refs=(), permitted_read_paths=(), candidate_write_paths=(),
                rejection_reasons=tuple(sorted(reasons)),
            )
            nominations.append(CandidateNomination(
                candidate=candidate,
                disposition=CandidateDisposition.REJECTED if reasons else CandidateDisposition.NOMINATED,
                signal_evidence=tuple((signal, tuple(sorted(set(refs), key=lambda ref: ref.content_id))) for signal, refs in entry["refs"].items()),
                diagnostics=tuple(sorted(reasons)), semantic_authority=False,
            ))
        nominations.sort(key=lambda item: item.content_id)
        candidates = tuple(nominations)
        return CandidateNominationReceipt(
            roots=self.roots, trace_id=trace.content_id, call_requirement_id=call_requirement.content_id,
            memory_safety_facet_id=memory_safety_facet.content_id, bounds=self.bounds,
            candidates=candidates, candidate_set_id=candidate_set_identity(tuple(item.candidate for item in candidates)),
            signal_roots=tuple(signal_roots.items()), vector_query_id=query_id, semantic_authority=False,
        )

    nominate = retrieve
    search = retrieve


def retrieve_contract_repair_candidates(
    roots: AuthorityRoots, trace: BrokenContractTrace, call_requirement: CallRequirementContract,
    memory_safety_facet: MemorySafetyFacet, **kwargs: Any,
) -> CandidateNominationReceipt:
    """Stateless convenience entry point for the retrieval-only boundary."""
    bounds = kwargs.pop("bounds", None)
    return ContractRepairCandidateRetriever(roots, bounds=bounds).retrieve(trace, call_requirement, memory_safety_facet, **kwargs)


__all__ = (
    "CANDIDATE_NOMINATION_SCHEMA", "CANDIDATE_NOMINATION_RECEIPT_SCHEMA", "CANDIDATE_RETRIEVAL_BOUNDS_SCHEMA",
    "SIGNAL_FAMILIES", "CandidateSignal", "CandidateDisposition", "CandidateRetrievalError",
    "CandidateRetrievalBindingError", "CandidateRetrievalBoundsError", "CandidateRetrievalBounds",
    "CandidateNomination", "CandidateNominationReceipt", "ContractRepairCandidateRetriever",
    "retrieve_contract_repair_candidates", "REJECTION_SAME_NAME_INCOMPATIBLE", "REJECTION_POISONED_VECTOR",
    "REJECTION_STALE_OR_CROSS_TREE", "REJECTION_READ_ONLY_TARGET", "REJECTION_GENERATED_VENDOR_ARCHIVE_TARGET",
    "REJECTION_FORBIDDEN_LAYER", "REJECTION_PARTIAL_CANDIDATE", "REJECTION_FORGED_HISTORY",
)
