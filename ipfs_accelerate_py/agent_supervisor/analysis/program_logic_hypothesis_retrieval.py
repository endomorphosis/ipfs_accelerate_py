"""Fail-closed, non-authoritative nomination of program-logic hypotheses (LPR-009).

This adapter is deliberately a *recall* boundary.  For each ``LogicGap@1``
bound to a current ``TacticianSearchPlan@1`` it unions bounded signal
families into one canonical candidate set:

* exact analytical constructions / deterministic templates
* existing values, constructors, and adapters
* theorem premises (from the independent premise corpus)
* dataflow facts
* graph neighborhoods
* schema analogues
* Git / lineage history
* test and specification analogues
* lexical / BM25 hits
* vector hits
* Tactician subgoals
* optional learned / model nominations

It never:

* claims ``semantic_authority`` (always false until later proof admission);
* confuses ranking scores with hard compatibility or information-content
  facts;
* treats same name, type, or similarity as information sufficiency;
* selects an edit target, admits a proof, or lowers a goal;
* retains source bodies or secrets.

Poisoned, stale, and cross-root rows are rejected or retained only with
stable rejection reasons.  Ambiguity and the empty (no-candidate) case
remain explicit dispositions.  Later proof/admission stages must consume
the complete candidate set rather than an individual nomination.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from enum import Enum
from typing import Any, ClassVar, Iterable

from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from .program_logic_prediction_contracts import (
    HypothesisDisposition,
    LogicGap,
    LogicHypothesis,
    ProgramLogicAuthorityRoots,
    ProofStatus,
    SourceAuthorityClass,
    SourceRouteKind,
    TacticianSearchPlan,
)


# ---------------------------------------------------------------------------
# Schemas / bounds
# ---------------------------------------------------------------------------

HYPOTHESIS_QUERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-hypothesis-query@1"
)
HYPOTHESIS_SIGNAL_REF_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-hypothesis-signal-ref@1"
)
HYPOTHESIS_HARD_GATE_FACTS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-hypothesis-hard-gate-facts@1"
)
LOGIC_HYPOTHESIS_NOMINATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/logic-hypothesis-nomination@1"
)
HYPOTHESIS_CANDIDATE_SET_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-hypothesis-candidate-set@1"
)
HYPOTHESIS_RETRIEVAL_BOUNDS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/program-logic-hypothesis-retrieval-bounds@1"
)

PRODUCER_ID = "program-logic-hypothesis-retrieval@1"
MAX_CANDIDATE_COUNT = 256
DEFAULT_MAX_PER_SIGNAL = 64
MAX_REF_BYTES = 512
MAX_TEXT_BYTES = 1_024
MAX_EVIDENCE_PER_NOMINATION = 64
MAX_SCORE_MILLIPERCENT = 100_000


class HypothesisRetrievalError(ValueError):
    """A hypothesis signal cannot safely participate in a nomination."""


class HypothesisRetrievalBindingError(HypothesisRetrievalError):
    """A required gap, plan, corpus, or root binding was mixed."""


class HypothesisRetrievalBoundsError(HypothesisRetrievalError):
    """A producer attempted to exceed the fixed retrieval budget."""


class HypothesisSignal(str, Enum):
    """Closed signal families that may contribute a hypothesis nomination hit."""

    ANALYTICAL_CONSTRUCTION = "analytical_construction"
    EXISTING_VALUE = "existing_value"
    CONSTRUCTOR_ADAPTER = "constructor_adapter"
    THEOREM_PREMISE = "theorem_premise"
    DATAFLOW = "dataflow"
    GRAPH_NEIGHBORHOOD = "graph_neighborhood"
    SCHEMA = "schema"
    LINEAGE = "lineage"
    TEST_SPEC_ANALOGUE = "test_spec_analogue"
    LEXICAL = "lexical"
    VECTOR = "vector"
    TACTICIAN_SUBGOAL = "tactician_subgoal"
    LEARNED_MODEL = "learned_model"


class HypothesisNominationDisposition(str, Enum):
    """Closed retrieval-time dispositions; admission happens later."""

    NOMINATED = "nominated"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    NO_CANDIDATE = "no_candidate"


SIGNAL_FAMILIES = tuple(item.value for item in HypothesisSignal)

_SIGNAL_ALIASES = {
    "analytical": HypothesisSignal.ANALYTICAL_CONSTRUCTION.value,
    "template": HypothesisSignal.ANALYTICAL_CONSTRUCTION.value,
    "deterministic_template": HypothesisSignal.ANALYTICAL_CONSTRUCTION.value,
    "exact_construction": HypothesisSignal.ANALYTICAL_CONSTRUCTION.value,
    "construction": HypothesisSignal.ANALYTICAL_CONSTRUCTION.value,
    "value": HypothesisSignal.EXISTING_VALUE.value,
    "existing": HypothesisSignal.EXISTING_VALUE.value,
    "existing_values": HypothesisSignal.EXISTING_VALUE.value,
    "constructor": HypothesisSignal.CONSTRUCTOR_ADAPTER.value,
    "adapter": HypothesisSignal.CONSTRUCTOR_ADAPTER.value,
    "constructors": HypothesisSignal.CONSTRUCTOR_ADAPTER.value,
    "adapters": HypothesisSignal.CONSTRUCTOR_ADAPTER.value,
    "factory": HypothesisSignal.CONSTRUCTOR_ADAPTER.value,
    "theorem": HypothesisSignal.THEOREM_PREMISE.value,
    "premise": HypothesisSignal.THEOREM_PREMISE.value,
    "premises": HypothesisSignal.THEOREM_PREMISE.value,
    "corpus": HypothesisSignal.THEOREM_PREMISE.value,
    "static_dataflow": HypothesisSignal.DATAFLOW.value,
    "graph": HypothesisSignal.GRAPH_NEIGHBORHOOD.value,
    "kg": HypothesisSignal.GRAPH_NEIGHBORHOOD.value,
    "neighborhood": HypothesisSignal.GRAPH_NEIGHBORHOOD.value,
    "knowledge_graph": HypothesisSignal.GRAPH_NEIGHBORHOOD.value,
    "schema_protocol": HypothesisSignal.SCHEMA.value,
    "history": HypothesisSignal.LINEAGE.value,
    "git_lineage": HypothesisSignal.LINEAGE.value,
    "test": HypothesisSignal.TEST_SPEC_ANALOGUE.value,
    "tests": HypothesisSignal.TEST_SPEC_ANALOGUE.value,
    "spec": HypothesisSignal.TEST_SPEC_ANALOGUE.value,
    "specs": HypothesisSignal.TEST_SPEC_ANALOGUE.value,
    "test_spec": HypothesisSignal.TEST_SPEC_ANALOGUE.value,
    "bm25": HypothesisSignal.LEXICAL.value,
    "lexical_bm25": HypothesisSignal.LEXICAL.value,
    "embedding": HypothesisSignal.VECTOR.value,
    "vector_hit": HypothesisSignal.VECTOR.value,
    "tactician": HypothesisSignal.TACTICIAN_SUBGOAL.value,
    "subgoal": HypothesisSignal.TACTICIAN_SUBGOAL.value,
    "subgoals": HypothesisSignal.TACTICIAN_SUBGOAL.value,
    "model": HypothesisSignal.LEARNED_MODEL.value,
    "llm": HypothesisSignal.LEARNED_MODEL.value,
    "learned": HypothesisSignal.LEARNED_MODEL.value,
    "model_hypothesis": HypothesisSignal.LEARNED_MODEL.value,
}

# Map closed retrieval signals onto LogicHypothesis evidence route kinds.
_SIGNAL_TO_ROUTE: dict[str, SourceRouteKind] = {
    HypothesisSignal.ANALYTICAL_CONSTRUCTION.value: SourceRouteKind.LOCAL_STATIC,
    HypothesisSignal.EXISTING_VALUE.value: SourceRouteKind.DATAFLOW,
    HypothesisSignal.CONSTRUCTOR_ADAPTER.value: SourceRouteKind.LOCAL_STATIC,
    HypothesisSignal.THEOREM_PREMISE.value: SourceRouteKind.REVIEWED_CONTRACT,
    HypothesisSignal.DATAFLOW.value: SourceRouteKind.DATAFLOW,
    HypothesisSignal.GRAPH_NEIGHBORHOOD.value: SourceRouteKind.GRAPH,
    HypothesisSignal.SCHEMA.value: SourceRouteKind.LOCAL_STATIC,
    HypothesisSignal.LINEAGE.value: SourceRouteKind.HISTORY,
    HypothesisSignal.TEST_SPEC_ANALOGUE.value: SourceRouteKind.REVIEWED_TEST,
    HypothesisSignal.LEXICAL.value: SourceRouteKind.VECTOR,
    HypothesisSignal.VECTOR.value: SourceRouteKind.VECTOR,
    HypothesisSignal.TACTICIAN_SUBGOAL.value: SourceRouteKind.TACTICIAN,
    HypothesisSignal.LEARNED_MODEL.value: SourceRouteKind.LLM,
}

# Stable public diagnostics.  Do not change without a versioned receipt schema.
REJECTION_STALE_OR_CROSS_ROOT = "stale_or_cross_root"
REJECTION_POISONED = "poisoned_signal"
REJECTION_FORGED = "forged_result"
REJECTION_PARTIAL = "partial_candidate"
REJECTION_BODY_OR_SECRET = "body_or_secret_payload"
REJECTION_SEMANTIC_AUTHORITY_CLAIM = "semantic_authority_claim"
REJECTION_SUFFICIENCY_CLAIM = "sufficiency_claim_from_name_type_or_similarity"
REJECTION_COMPATIBILITY_AS_ADMISSION = "compatibility_fact_used_as_admission"
REJECTION_EXCLUDED_PREMISE = "excluded_premise"
REJECTION_INVALID_PAYLOAD = "invalid_candidate_payload"
REJECTION_CROSS_GAP_OR_GOAL = "cross_gap_or_goal"
REJECTION_SCORE_AS_HARD_GATE = "score_used_as_hard_gate"

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
        "theorem_text",
        "proof_script",
        "prompt_body",
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
        "private_witness",
        "private_premise",
        "client_secret",
    }
)
_SUFFICIENCY_KEYS = frozenset(
    {
        "sufficient",
        "sufficiency",
        "information_sufficient",
        "information_sufficiency",
        "admits_sufficiency",
        "proved_sufficient",
        "complete_information",
        "completeness",
        "is_sufficient",
        "name_establishes_sufficiency",
        "type_establishes_sufficiency",
        "similarity_establishes_sufficiency",
    }
)
_ADMISSION_COMPAT_KEYS = frozenset(
    {
        "admits_compatibility",
        "proved_compatible",
        "compatibility_establishes_admission",
        "hard_gate_passed_by_score",
        "score_establishes_compatibility",
    }
)
_ROOT_KEYS = (
    "repository_id",
    "objective_id",
    "trace_id",
    "change_id",
    "consumer_id",
    "forest_id",
    "tree_id",
    "overlay_id",
    "graph_id",
    "index_id",
    "corpus_id",
    "model_id",
    "translator_id",
    "toolchain_id",
    "policy_id",
    "environment_id",
)
_SECRET_VALUE_RE = re.compile(
    r"(?:^|[^a-z0-9_])(?:api[_-]?key|password|secret|token|passwd)"
    r"(?:[^a-z0-9_]|$)|"
    r"bearer\s+[a-z0-9._\-]{8,}|"
    r"-----begin\s+",
    re.IGNORECASE,
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


def _fingerprint(value: Any, *, prefix: str = "hypothesis") -> str:
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
        raise HypothesisRetrievalError(f"{name} is required")
    if "\x00" in text or len(text.encode("utf-8")) > limit:
        raise HypothesisRetrievalBoundsError(f"{name} is invalid or exceeds its bound")
    return text


def _signal(name: Any) -> str:
    normalized = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = _SIGNAL_ALIASES.get(normalized, normalized)
    if normalized not in SIGNAL_FAMILIES:
        raise HypothesisRetrievalError(f"unsupported hypothesis signal: {name}")
    return normalized


def _score_millipercent(value: Any, name: str = "nomination_score_millipercent") -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        raise HypothesisRetrievalError(f"{name} must be an integer millipercent")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HypothesisRetrievalError(f"{name} must be finite")
        # Accept legacy 0.0-1.0 similarity as millipercent.
        if 0.0 <= value <= 1.0:
            value = int(round(value * MAX_SCORE_MILLIPERCENT))
        else:
            value = int(round(value))
    if not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise HypothesisRetrievalError(f"{name} must be an integer millipercent") from exc
    if value < 0 or value > MAX_SCORE_MILLIPERCENT:
        raise HypothesisRetrievalBoundsError(
            f"{name} must be in 0..{MAX_SCORE_MILLIPERCENT}"
        )
    return value


def _verify_record_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    claimed = payload.get("content_id", payload.get("cid", ""))
    if claimed not in (None, "", record.content_id):
        raise HypothesisRetrievalBindingError(
            "stored content identity does not match the canonical record"
        )


def _contains_body_or_secret(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _BODY_FIELDS or normalized in _SECRET_FIELDS:
                return True
            if isinstance(item, str) and _SECRET_VALUE_RE.search(item):
                return True
            if _contains_body_or_secret(item):
                return True
        return False
    if isinstance(value, (bytes, bytearray)):
        return True
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        return True
    return isinstance(value, Sequence) and not isinstance(value, str) and any(
        _contains_body_or_secret(item) for item in value
    )


def _redact_payload(value: Any) -> Any:
    """Drop body/secret fields while preserving compact structure for diagnostics."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _BODY_FIELDS or normalized in _SECRET_FIELDS:
                result[str(key)] = "<redacted>"
            elif isinstance(item, str) and _SECRET_VALUE_RE.search(item):
                result[str(key)] = "<redacted>"
            else:
                result[str(key)] = _redact_payload(item)
        return result
    if isinstance(value, (bytes, bytearray)):
        return "<redacted-bytes>"
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_redact_payload(item) for item in value]
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        return "<redacted>"
    return value


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _roots_equal(left: ProgramLogicAuthorityRoots, right: ProgramLogicAuthorityRoots) -> bool:
    return left.content_id == right.content_id


def candidate_set_identity(nominations: Sequence["LogicHypothesisNomination"]) -> str:
    """Bind the complete, deterministically ordered nomination set."""
    if len(nominations) > MAX_CANDIDATE_COUNT:
        raise HypothesisRetrievalBoundsError("candidate set exceeds hard bound")
    ids = tuple(sorted(item.content_id for item in nominations))
    if len(set(ids)) != len(ids):
        raise HypothesisRetrievalError("candidate set contains duplicate nominations")
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/program-logic-hypothesis-set@1",
            "nomination_ids": list(ids),
        }
    )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisSignalRef(CanonicalContract):
    """Compact per-signal evidence pointer; never holds bodies."""

    SCHEMA: ClassVar[str] = HYPOTHESIS_SIGNAL_REF_SCHEMA

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
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypothesisSignalRef":
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
            raise HypothesisRetrievalError("unsupported hypothesis signal ref payload")
        if payload.get("schema") not in (None, cls.SCHEMA):
            raise HypothesisRetrievalError("unsupported hypothesis signal ref schema")
        value = cls(
            signal=payload.get("signal", ""),
            artifact_id=payload.get("artifact_id", ""),
            locator=payload.get("locator", ""),
            producer_id=payload.get("producer_id", PRODUCER_ID),
        )
        _verify_record_identity(payload, value)
        return value


def _refs(value: Any, signal: str, raw: Mapping[str, Any]) -> tuple[HypothesisSignalRef, ...]:
    if value is None:
        values: Iterable[Any] = ()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        values = value
    else:
        values = (value,)
    refs: list[HypothesisSignalRef] = []
    for item in values:
        try:
            if isinstance(item, HypothesisSignalRef):
                ref = item
            elif isinstance(item, Mapping):
                ref = HypothesisSignalRef(
                    signal=str(item.get("signal", signal)),
                    artifact_id=str(
                        item.get("artifact_id", item.get("locator", item.get("ref", "")))
                    ),
                    locator=str(item.get("locator", "")),
                    producer_id=str(item.get("producer_id", PRODUCER_ID)),
                )
            elif isinstance(item, str) and item.strip():
                ref = HypothesisSignalRef(signal=signal, artifact_id=item.strip())
            else:
                continue
        except (KeyError, HypothesisRetrievalError, TypeError):
            continue
        if ref not in refs:
            refs.append(ref)
    if not refs:
        refs.append(
            HypothesisSignalRef(
                signal=signal,
                artifact_id=_fingerprint(raw, prefix="signal-artifact"),
            )
        )
    return tuple(sorted(refs, key=lambda item: item.content_id))


@dataclass(frozen=True)
class HypothesisRetrievalBounds(CanonicalContract):
    """Fixed, replayable caps; over-budget input is rejected, never truncated."""

    SCHEMA: ClassVar[str] = HYPOTHESIS_RETRIEVAL_BOUNDS_SCHEMA

    max_candidates: int = MAX_CANDIDATE_COUNT
    max_candidates_per_signal: int = DEFAULT_MAX_PER_SIGNAL
    max_evidence_per_nomination: int = MAX_EVIDENCE_PER_NOMINATION
    max_text_bytes: int = MAX_TEXT_BYTES

    def __post_init__(self) -> None:
        for name, upper in (
            ("max_candidates", MAX_CANDIDATE_COUNT),
            ("max_candidates_per_signal", MAX_CANDIDATE_COUNT),
            ("max_evidence_per_nomination", MAX_EVIDENCE_PER_NOMINATION),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise HypothesisRetrievalBoundsError(
                    f"{name} must be an integer from 1 through {upper}"
                )
        if (
            isinstance(self.max_text_bytes, bool)
            or not isinstance(self.max_text_bytes, int)
            or not 1 <= self.max_text_bytes <= MAX_TEXT_BYTES
        ):
            raise HypothesisRetrievalBoundsError(
                f"max_text_bytes must be an integer from 1 through {MAX_TEXT_BYTES}"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "max_candidates_per_signal": self.max_candidates_per_signal,
            "max_evidence_per_nomination": self.max_evidence_per_nomination,
            "max_text_bytes": self.max_text_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypothesisRetrievalBounds":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "max_candidates",
            "max_candidates_per_signal",
            "max_evidence_per_nomination",
            "max_text_bytes",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise HypothesisRetrievalError("unsupported hypothesis retrieval bounds payload")
        value = cls(
            max_candidates=payload.get("max_candidates", MAX_CANDIDATE_COUNT),
            max_candidates_per_signal=payload.get(
                "max_candidates_per_signal", DEFAULT_MAX_PER_SIGNAL
            ),
            max_evidence_per_nomination=payload.get(
                "max_evidence_per_nomination", MAX_EVIDENCE_PER_NOMINATION
            ),
            max_text_bytes=payload.get("max_text_bytes", MAX_TEXT_BYTES),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class HypothesisQuery(CanonicalContract):
    """Exact binding of a hypothesis retrieval query to gap, plan, and roots."""

    SCHEMA: ClassVar[str] = HYPOTHESIS_QUERY_SCHEMA

    roots: ProgramLogicAuthorityRoots
    gap_id: str
    goal_id: str
    plan_id: str
    corpus_id: str = ""
    missing_class: str = ""
    counterexample_target_ref: str = ""
    selected_premise_ids: tuple[str, ...] = ()
    excluded_premise_ids: tuple[str, ...] = ()
    query_refs: tuple[str, ...] = ()
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, ProgramLogicAuthorityRoots):
            raise HypothesisRetrievalBindingError(
                "query roots must be ProgramLogicAuthorityRoots"
            )
        object.__setattr__(self, "gap_id", _text(self.gap_id, "gap_id"))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "corpus_id", _text(self.corpus_id, "corpus_id", required=False)
        )
        object.__setattr__(
            self, "missing_class", _text(self.missing_class, "missing_class", required=False)
        )
        object.__setattr__(
            self,
            "counterexample_target_ref",
            _text(
                self.counterexample_target_ref,
                "counterexample_target_ref",
                required=False,
            ),
        )
        selected = tuple(
            sorted(
                {
                    _text(item, "selected_premise_ids")
                    for item in (self.selected_premise_ids or ())
                    if str(item).strip()
                }
            )
        )
        excluded = tuple(
            sorted(
                {
                    _text(item, "excluded_premise_ids")
                    for item in (self.excluded_premise_ids or ())
                    if str(item).strip()
                }
            )
        )
        if set(selected) & set(excluded):
            raise HypothesisRetrievalBindingError(
                "selected and excluded premises must be disjoint"
            )
        object.__setattr__(self, "selected_premise_ids", selected)
        object.__setattr__(self, "excluded_premise_ids", excluded)
        refs = tuple(
            sorted(
                {
                    _text(item, "query_refs")
                    for item in (self.query_refs or ())
                    if str(item).strip()
                }
            )
        )
        object.__setattr__(self, "query_refs", refs)
        if self.semantic_authority is not False:
            raise HypothesisRetrievalBindingError(
                "hypothesis queries cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    @classmethod
    def from_gap_and_plan(
        cls,
        gap: LogicGap,
        plan: TacticianSearchPlan,
        *,
        corpus_id: str = "",
        counterexample_target_ref: str = "",
    ) -> "HypothesisQuery":
        if not isinstance(gap, LogicGap):
            raise HypothesisRetrievalBindingError("query requires a typed LogicGap")
        if not isinstance(plan, TacticianSearchPlan):
            raise HypothesisRetrievalBindingError(
                "query requires a typed TacticianSearchPlan"
            )
        if not _roots_equal(gap.roots, plan.roots):
            raise HypothesisRetrievalBindingError(
                "gap and plan must share exact authority roots"
            )
        if gap.goal_id not in plan.goal_ids:
            raise HypothesisRetrievalBindingError(
                "gap goal_id must be listed in plan goal_ids"
            )
        return cls(
            roots=gap.roots,
            gap_id=gap.gap_id,
            goal_id=gap.goal_id,
            plan_id=plan.plan_id,
            corpus_id=corpus_id or gap.roots.corpus_id,
            missing_class=str(
                getattr(gap.missing_class, "value", gap.missing_class) or ""
            ),
            counterexample_target_ref=counterexample_target_ref,
            selected_premise_ids=plan.selected_premise_ids,
            excluded_premise_ids=plan.excluded_premise_ids,
            query_refs=plan.query_refs,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "roots": self.roots.to_dict(),
            "gap_id": self.gap_id,
            "goal_id": self.goal_id,
            "plan_id": self.plan_id,
            "corpus_id": self.corpus_id,
            "missing_class": self.missing_class,
            "counterexample_target_ref": self.counterexample_target_ref,
            "selected_premise_ids": list(self.selected_premise_ids),
            "excluded_premise_ids": list(self.excluded_premise_ids),
            "query_refs": list(self.query_refs),
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypothesisQuery":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "roots",
            "gap_id",
            "goal_id",
            "plan_id",
            "corpus_id",
            "missing_class",
            "counterexample_target_ref",
            "selected_premise_ids",
            "excluded_premise_ids",
            "query_refs",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise HypothesisRetrievalError("unsupported hypothesis query payload")
        roots = payload.get("roots")
        value = cls(
            roots=roots
            if isinstance(roots, ProgramLogicAuthorityRoots)
            else ProgramLogicAuthorityRoots.from_dict(roots),
            gap_id=payload.get("gap_id", ""),
            goal_id=payload.get("goal_id", ""),
            plan_id=payload.get("plan_id", ""),
            corpus_id=payload.get("corpus_id", ""),
            missing_class=payload.get("missing_class", ""),
            counterexample_target_ref=payload.get("counterexample_target_ref", ""),
            selected_premise_ids=tuple(payload.get("selected_premise_ids", ())),
            excluded_premise_ids=tuple(payload.get("excluded_premise_ids", ())),
            query_refs=tuple(payload.get("query_refs", ())),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class HypothesisHardGateFacts(CanonicalContract):
    """Hard compatibility / information-content facts, never ranking scores.

    Retrieval may *observe* type/effect/auth facts and record information-
    content references.  It cannot promote those observations into admission,
    completeness, or information sufficiency.  ``information_sufficiency`` is
    always false at this boundary — same name, type, or similarity never
    establishes sufficiency.
    """

    SCHEMA: ClassVar[str] = HYPOTHESIS_HARD_GATE_FACTS_SCHEMA

    type_compatible: bool | None = None
    effect_compatible: bool | None = None
    auth_compatible: bool | None = None
    resource_compatible: bool | None = None
    lifecycle_compatible: bool | None = None
    information_content_ref: str = ""
    information_sufficiency: bool = False
    same_name: bool = False
    same_type: bool = False
    similarity_millipercent: int = 0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "type_compatible",
            "effect_compatible",
            "auth_compatible",
            "resource_compatible",
            "lifecycle_compatible",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise HypothesisRetrievalError(f"{name} must be a boolean or null")
        object.__setattr__(
            self,
            "information_content_ref",
            _text(
                self.information_content_ref,
                "information_content_ref",
                required=False,
            ),
        )
        # Hard rule: retrieval cannot establish sufficiency.
        if self.information_sufficiency is not False:
            raise HypothesisRetrievalBindingError(
                "retrieval cannot establish information sufficiency; "
                "same name/type/similarity are not sufficient"
            )
        object.__setattr__(self, "information_sufficiency", False)
        for name in ("same_name", "same_type"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise HypothesisRetrievalError(f"{name} must be a boolean")
        object.__setattr__(
            self,
            "similarity_millipercent",
            _score_millipercent(self.similarity_millipercent, "similarity_millipercent"),
        )
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
            "type_compatible": self.type_compatible,
            "effect_compatible": self.effect_compatible,
            "auth_compatible": self.auth_compatible,
            "resource_compatible": self.resource_compatible,
            "lifecycle_compatible": self.lifecycle_compatible,
            "information_content_ref": self.information_content_ref,
            "information_sufficiency": False,
            "same_name": self.same_name,
            "same_type": self.same_type,
            "similarity_millipercent": self.similarity_millipercent,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypothesisHardGateFacts":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "type_compatible",
            "effect_compatible",
            "auth_compatible",
            "resource_compatible",
            "lifecycle_compatible",
            "information_content_ref",
            "information_sufficiency",
            "same_name",
            "same_type",
            "similarity_millipercent",
            "notes",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") not in (None, cls.SCHEMA)
            or set(payload).difference(allowed)
        ):
            raise HypothesisRetrievalError("unsupported hard-gate facts payload")
        value = cls(
            type_compatible=payload.get("type_compatible"),
            effect_compatible=payload.get("effect_compatible"),
            auth_compatible=payload.get("auth_compatible"),
            resource_compatible=payload.get("resource_compatible"),
            lifecycle_compatible=payload.get("lifecycle_compatible"),
            information_content_ref=payload.get("information_content_ref", ""),
            information_sufficiency=payload.get("information_sufficiency", False),
            same_name=bool(payload.get("same_name", False)),
            same_type=bool(payload.get("same_type", False)),
            similarity_millipercent=payload.get("similarity_millipercent", 0),
            notes=tuple(payload.get("notes", ())),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class LogicHypothesisNomination(CanonicalContract):
    """One nominated (or rejected) hypothesis with complete signal provenance.

    Ranking scores live only on ``nomination_score_millipercent`` and never
    override hard-gate facts.  ``semantic_authority`` is always false.
    """

    SCHEMA: ClassVar[str] = LOGIC_HYPOTHESIS_NOMINATION_SCHEMA

    hypothesis: LogicHypothesis
    disposition: HypothesisNominationDisposition
    signal_evidence: tuple[tuple[str, tuple[HypothesisSignalRef, ...]], ...]
    hard_gate_facts: HypothesisHardGateFacts
    nomination_score_millipercent: int = 0
    diagnostics: tuple[str, ...] = ()
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis, LogicHypothesis):
            raise HypothesisRetrievalError("nomination requires LogicHypothesis")
        object.__setattr__(
            self,
            "disposition",
            (
                self.disposition
                if isinstance(self.disposition, HypothesisNominationDisposition)
                else HypothesisNominationDisposition(self.disposition)
            ),
        )
        if not isinstance(self.hard_gate_facts, HypothesisHardGateFacts):
            raise HypothesisRetrievalError(
                "nomination requires HypothesisHardGateFacts"
            )
        rows: list[tuple[str, tuple[HypothesisSignalRef, ...]]] = []
        raw_evidence = (
            self.signal_evidence.items()
            if isinstance(self.signal_evidence, Mapping)
            else self.signal_evidence
        )
        for item in raw_evidence:
            try:
                signal, refs = item
            except (TypeError, ValueError) as exc:
                raise HypothesisRetrievalError(
                    "signal evidence rows must contain signal and references"
                ) from exc
            normalized = _signal(signal)
            checked = tuple(
                ref
                if isinstance(ref, HypothesisSignalRef)
                else HypothesisSignalRef.from_dict(ref)
                for ref in (
                    refs
                    if isinstance(refs, Sequence)
                    and not isinstance(refs, (str, bytes, bytearray, Mapping))
                    else (refs,)
                )
            )
            checked = tuple(sorted(checked, key=lambda ref: ref.content_id))
            if len(checked) > MAX_EVIDENCE_PER_NOMINATION:
                raise HypothesisRetrievalBoundsError(
                    "signal evidence exceeds max_evidence_per_nomination"
                )
            rows.append((normalized, checked))
        rows.sort(key=lambda item: item[0])
        if len({item[0] for item in rows}) != len(rows):
            raise HypothesisRetrievalError("nomination has duplicate signal evidence")
        object.__setattr__(self, "signal_evidence", tuple(rows))
        object.__setattr__(
            self,
            "nomination_score_millipercent",
            _score_millipercent(self.nomination_score_millipercent),
        )
        diagnostics = tuple(
            sorted({str(item).strip() for item in (self.diagnostics or ()) if str(item).strip()})
        )
        object.__setattr__(self, "diagnostics", diagnostics)
        if self.semantic_authority is not False:
            raise HypothesisRetrievalBindingError(
                "nominations cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        if self.hypothesis.semantic_authority is not False:
            raise HypothesisRetrievalBindingError(
                "wrapped hypothesis cannot claim semantic authority"
            )
        # Score must not silently disagree with the hypothesis score field.
        if (
            self.hypothesis.nomination_score_millipercent
            != self.nomination_score_millipercent
        ):
            raise HypothesisRetrievalBindingError(
                "nomination score must match hypothesis.nomination_score_millipercent"
            )
        if (
            self.disposition is HypothesisNominationDisposition.NOMINATED
            and diagnostics
        ):
            raise HypothesisRetrievalError(
                "nominated candidates cannot carry rejection diagnostics"
            )
        if (
            self.disposition
            in {
                HypothesisNominationDisposition.REJECTED,
                HypothesisNominationDisposition.NO_CANDIDATE,
            }
            and not diagnostics
        ):
            raise HypothesisRetrievalError(
                "rejected/no-candidate nominations require stable diagnostics"
            )
        if (
            self.disposition is HypothesisNominationDisposition.AMBIGUOUS
            and not diagnostics
            and not rows
        ):
            # Ambiguity without evidence is still explicit, but require a reason.
            raise HypothesisRetrievalError(
                "ambiguous nominations require diagnostics or signal evidence"
            )

    @property
    def hypothesis_id(self) -> str:
        return self.hypothesis.hypothesis_id

    @property
    def write_paths(self) -> tuple[str, ...]:
        """Retrieval never provides mutation authority."""
        return ()

    def _payload(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "disposition": self.disposition.value,
            "signal_evidence": [
                {
                    "signal": signal,
                    "evidence_refs": [ref.to_dict() for ref in refs],
                }
                for signal, refs in self.signal_evidence
            ],
            "hard_gate_facts": self.hard_gate_facts.to_dict(),
            "nomination_score_millipercent": self.nomination_score_millipercent,
            "diagnostics": list(self.diagnostics),
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicHypothesisNomination":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "hypothesis",
            "disposition",
            "signal_evidence",
            "hard_gate_facts",
            "nomination_score_millipercent",
            "diagnostics",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise HypothesisRetrievalError("unsupported logic hypothesis nomination payload")
        signal_evidence: list[tuple[str, tuple[HypothesisSignalRef, ...]]] = []
        supplied = payload.get("signal_evidence", ())
        if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes, bytearray)):
            raise HypothesisRetrievalError("signal_evidence must be a sequence")
        for row in supplied:
            if not isinstance(row, Mapping):
                raise HypothesisRetrievalError("signal evidence row must be an object")
            refs = row.get("evidence_refs", ())
            if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
                raise HypothesisRetrievalError(
                    "signal evidence references must be a sequence"
                )
            signal_evidence.append(
                (
                    str(row.get("signal", "")),
                    tuple(
                        item
                        if isinstance(item, HypothesisSignalRef)
                        else HypothesisSignalRef.from_dict(item)
                        for item in refs
                    ),
                )
            )
        hypothesis = payload.get("hypothesis")
        facts = payload.get("hard_gate_facts")
        value = cls(
            hypothesis=hypothesis
            if isinstance(hypothesis, LogicHypothesis)
            else LogicHypothesis.from_dict(hypothesis),
            disposition=payload.get("disposition", ""),
            signal_evidence=tuple(signal_evidence),
            hard_gate_facts=facts
            if isinstance(facts, HypothesisHardGateFacts)
            else HypothesisHardGateFacts.from_dict(facts or {}),
            nomination_score_millipercent=payload.get(
                "nomination_score_millipercent", 0
            ),
            diagnostics=tuple(payload.get("diagnostics", ())),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class HypothesisCandidateSet(CanonicalContract):
    """The complete bounded candidate set; this is not a target decision."""

    SCHEMA: ClassVar[str] = HYPOTHESIS_CANDIDATE_SET_SCHEMA

    roots: ProgramLogicAuthorityRoots
    query: HypothesisQuery
    gap_id: str
    plan_id: str
    bounds: HypothesisRetrievalBounds
    nominations: tuple[LogicHypothesisNomination, ...]
    candidate_set_id: str
    signal_roots: tuple[tuple[str, str], ...] = ()
    corpus_id: str = ""
    graph_id: str = ""
    vector_query_id: str = ""
    ambiguous: bool = False
    no_candidate: bool = False
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, ProgramLogicAuthorityRoots):
            raise HypothesisRetrievalError(
                "candidate set roots must be ProgramLogicAuthorityRoots"
            )
        if not isinstance(self.query, HypothesisQuery):
            raise HypothesisRetrievalError("candidate set query must be HypothesisQuery")
        if not isinstance(self.bounds, HypothesisRetrievalBounds):
            raise HypothesisRetrievalError(
                "candidate set bounds must be HypothesisRetrievalBounds"
            )
        if not _roots_equal(self.query.roots, self.roots):
            raise HypothesisRetrievalBindingError(
                "query roots do not match candidate set roots"
            )
        object.__setattr__(self, "gap_id", _text(self.gap_id, "gap_id"))
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        if self.gap_id != self.query.gap_id:
            raise HypothesisRetrievalBindingError(
                "candidate set gap_id does not match query"
            )
        if self.plan_id != self.query.plan_id:
            raise HypothesisRetrievalBindingError(
                "candidate set plan_id does not match query"
            )
        nominations = tuple(sorted(self.nominations, key=lambda item: item.content_id))
        if not nominations or len(nominations) > self.bounds.max_candidates:
            raise HypothesisRetrievalBoundsError(
                "candidate count is outside its declared bound"
            )
        if any(not isinstance(item, LogicHypothesisNomination) for item in nominations):
            raise HypothesisRetrievalError("candidates must be LogicHypothesisNomination")
        if len({item.content_id for item in nominations}) != len(nominations):
            raise HypothesisRetrievalError("candidate set contains duplicate nominations")
        if any(
            not _roots_equal(item.hypothesis.roots, self.roots) for item in nominations
        ):
            raise HypothesisRetrievalBindingError(
                "nomination roots do not match candidate set roots"
            )
        object.__setattr__(self, "nominations", nominations)
        expected = candidate_set_identity(nominations)
        if self.candidate_set_id != expected:
            raise HypothesisRetrievalBindingError(
                "candidate_set_id does not bind the complete candidate set"
            )
        roots: list[tuple[str, str]] = []
        for signal, root in self.signal_roots:
            normalized = _signal(signal)
            if not isinstance(root, str) or not root:
                raise HypothesisRetrievalBindingError(
                    "signal roots must be nonempty identities"
                )
            roots.append((normalized, root))
        roots.sort()
        if len({item[0] for item in roots}) != len(roots):
            raise HypothesisRetrievalBindingError(
                "candidate set contains duplicate signal roots"
            )
        object.__setattr__(self, "signal_roots", tuple(roots))
        object.__setattr__(
            self, "corpus_id", _text(self.corpus_id, "corpus_id", required=False)
        )
        object.__setattr__(
            self, "graph_id", _text(self.graph_id, "graph_id", required=False)
        )
        object.__setattr__(
            self,
            "vector_query_id",
            _text(self.vector_query_id, "vector_query_id", required=False),
        )
        if not isinstance(self.ambiguous, bool):
            raise HypothesisRetrievalError("ambiguous must be a boolean")
        if not isinstance(self.no_candidate, bool):
            raise HypothesisRetrievalError("no_candidate must be a boolean")
        # Derive explicit flags from nominations (fail-closed consistency).
        has_ambiguous = any(
            item.disposition is HypothesisNominationDisposition.AMBIGUOUS
            for item in nominations
        )
        has_no_candidate = any(
            item.disposition is HypothesisNominationDisposition.NO_CANDIDATE
            for item in nominations
        )
        only_no_candidate = all(
            item.disposition is HypothesisNominationDisposition.NO_CANDIDATE
            for item in nominations
        )
        if self.ambiguous != has_ambiguous:
            raise HypothesisRetrievalBindingError(
                "ambiguous flag must match nomination dispositions"
            )
        if self.no_candidate != (has_no_candidate and only_no_candidate):
            raise HypothesisRetrievalBindingError(
                "no_candidate flag must match exclusive no-candidate disposition"
            )
        if self.semantic_authority is not False:
            raise HypothesisRetrievalBindingError(
                "candidate sets cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    @property
    def candidates(self) -> tuple[LogicHypothesisNomination, ...]:
        return self.nominations

    @property
    def write_paths(self) -> tuple[str, ...]:
        return ()

    @property
    def admitted_hypothesis_id(self) -> str:
        """There is deliberately no winner at retrieval time."""
        return ""

    @property
    def query_id(self) -> str:
        return self.query.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "roots": self.roots.to_dict(),
            "query": self.query.to_dict(),
            "gap_id": self.gap_id,
            "plan_id": self.plan_id,
            "bounds": self.bounds.to_dict(),
            "nominations": [item.to_dict() for item in self.nominations],
            "candidate_set_id": self.candidate_set_id,
            "signal_roots": [
                {"signal": signal, "root_id": root} for signal, root in self.signal_roots
            ],
            "corpus_id": self.corpus_id,
            "graph_id": self.graph_id,
            "vector_query_id": self.vector_query_id,
            "ambiguous": self.ambiguous,
            "no_candidate": self.no_candidate,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypothesisCandidateSet":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "roots",
            "query",
            "gap_id",
            "plan_id",
            "bounds",
            "nominations",
            "candidate_set_id",
            "signal_roots",
            "corpus_id",
            "graph_id",
            "vector_query_id",
            "ambiguous",
            "no_candidate",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise HypothesisRetrievalError(
                "unsupported hypothesis candidate set payload"
            )
        rows = payload.get("signal_roots", ())
        nominations = payload.get("nominations", ())
        if (
            not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes, bytearray))
            or not isinstance(nominations, Sequence)
            or isinstance(nominations, (str, bytes, bytearray))
        ):
            raise HypothesisRetrievalError(
                "signal roots and nominations must be sequences"
            )
        signal_roots: list[tuple[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise HypothesisRetrievalError("signal root row must be an object")
            signal_roots.append(
                (str(row.get("signal", "")), str(row.get("root_id", "")))
            )
        roots = payload.get("roots")
        query = payload.get("query")
        bounds = payload.get("bounds")
        value = cls(
            roots=roots
            if isinstance(roots, ProgramLogicAuthorityRoots)
            else ProgramLogicAuthorityRoots.from_dict(roots),
            query=query
            if isinstance(query, HypothesisQuery)
            else HypothesisQuery.from_dict(query),
            gap_id=payload.get("gap_id", ""),
            plan_id=payload.get("plan_id", ""),
            bounds=bounds
            if isinstance(bounds, HypothesisRetrievalBounds)
            else HypothesisRetrievalBounds.from_dict(bounds),
            nominations=tuple(
                item
                if isinstance(item, LogicHypothesisNomination)
                else LogicHypothesisNomination.from_dict(item)
                for item in nominations
            ),
            candidate_set_id=payload.get("candidate_set_id", ""),
            signal_roots=tuple(signal_roots),
            corpus_id=payload.get("corpus_id", ""),
            graph_id=payload.get("graph_id", ""),
            vector_query_id=payload.get("vector_query_id", ""),
            ambiguous=bool(payload.get("ambiguous", False)),
            no_candidate=bool(payload.get("no_candidate", False)),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


# ---------------------------------------------------------------------------
# Signal normalisation / diagnostics
# ---------------------------------------------------------------------------


def _consequence_ref(raw: Mapping[str, Any], query: HypothesisQuery) -> str:
    for key in (
        "claimed_consequence_ref",
        "consequence_ref",
        "consequence",
        "claim_ref",
        "claim",
        "hypothesis_ref",
        "name",
        "symbol",
        "symbol_id",
        "expression_ref",
        "construction_ref",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"consequence:{query.gap_id}:{_fingerprint(raw).split(':')[-1][:16]}"


def _construction_ref(raw: Mapping[str, Any]) -> str:
    for key in (
        "construction_ref",
        "construction",
        "template_ref",
        "factory_ref",
        "adapter_ref",
        "expression_ref",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _value_ref(raw: Mapping[str, Any]) -> str:
    for key in ("value_ref", "value", "existing_value_ref", "expression_ref"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _placement_ref(raw: Mapping[str, Any]) -> str:
    for key in ("placement_ref", "placement"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _premise_ids(raw: Mapping[str, Any], query: HypothesisQuery) -> tuple[str, ...]:
    values = raw.get(
        "selected_premise_ids",
        raw.get("premise_ids", raw.get("premises", ())),
    )
    if values is None or values == "":
        values = ()
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        return ()
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            result.append(text)
    if not result and query.selected_premise_ids:
        # Inherit plan-selected premises only when the hit does not override.
        if raw.get("inherit_plan_premises", True) is not False:
            result.extend(query.selected_premise_ids)
    return tuple(sorted(set(result)))


def _counterexample_target(raw: Mapping[str, Any], query: HypothesisQuery) -> str:
    for key in (
        "counterexample_target_ref",
        "counterexample_target",
        "negative_target_ref",
        "counterexample",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return query.counterexample_target_ref


def _hard_gate_facts(raw: Mapping[str, Any]) -> HypothesisHardGateFacts:
    facts_raw = raw.get("hard_gate_facts", raw.get("hard_facts", {}))
    if isinstance(facts_raw, HypothesisHardGateFacts):
        return facts_raw
    if not isinstance(facts_raw, Mapping):
        facts_raw = {}
    merged = dict(facts_raw)
    for key, dest in (
        ("type_compatible", "type_compatible"),
        ("effect_compatible", "effect_compatible"),
        ("auth_compatible", "auth_compatible"),
        ("resource_compatible", "resource_compatible"),
        ("lifecycle_compatible", "lifecycle_compatible"),
        ("information_content_ref", "information_content_ref"),
        ("same_name", "same_name"),
        ("same_type", "same_type"),
    ):
        if key in raw and dest not in merged:
            merged[dest] = raw[key]
    if "similarity_millipercent" not in merged:
        if "similarity_millipercent" in raw:
            merged["similarity_millipercent"] = raw["similarity_millipercent"]
        elif "similarity" in raw:
            merged["similarity_millipercent"] = raw["similarity"]
        elif "score" in raw and isinstance(raw.get("score"), float) and 0.0 <= float(
            raw["score"]
        ) <= 1.0:
            # Vector similarity may populate the *observation* field only.
            try:
                merged["similarity_millipercent"] = _score_millipercent(raw["score"])
            except HypothesisRetrievalError:
                pass
    # Never accept sufficiency from the raw payload.
    merged["information_sufficiency"] = False
    return HypothesisHardGateFacts(
        type_compatible=_optional_bool(merged.get("type_compatible")),
        effect_compatible=_optional_bool(merged.get("effect_compatible")),
        auth_compatible=_optional_bool(merged.get("auth_compatible")),
        resource_compatible=_optional_bool(merged.get("resource_compatible")),
        lifecycle_compatible=_optional_bool(merged.get("lifecycle_compatible")),
        information_content_ref=str(merged.get("information_content_ref", "") or ""),
        information_sufficiency=False,
        same_name=bool(merged.get("same_name", False)),
        same_type=bool(merged.get("same_type", False)),
        similarity_millipercent=merged.get("similarity_millipercent", 0) or 0,
        notes=tuple(merged.get("notes", ()) or ()),
    )


def _nomination_score(raw: Mapping[str, Any]) -> int:
    for key in (
        "nomination_score_millipercent",
        "score_millipercent",
        "rank_score_millipercent",
    ):
        if key in raw and raw[key] not in (None, ""):
            return _score_millipercent(raw[key], key)
    if "score" in raw and raw["score"] not in (None, ""):
        return _score_millipercent(raw["score"], "score")
    return 0


def _route_kinds_for_signals(signals: set[str]) -> tuple[SourceRouteKind, ...]:
    routes = {_SIGNAL_TO_ROUTE[signal] for signal in signals if signal in _SIGNAL_TO_ROUTE}
    return tuple(sorted(routes, key=lambda item: item.value))


def _diagnostics(
    signal: str,
    raw: Mapping[str, Any],
    expected_roots: ProgramLogicAuthorityRoots,
    query: HypothesisQuery,
    vector_roots: tuple[str, str, str] | None,
) -> set[str]:
    reasons: set[str] = set()
    if raw.get("partial") is True or raw.get("complete") is False:
        reasons.add(REJECTION_PARTIAL)
    if _contains_body_or_secret(raw):
        reasons.add(REJECTION_BODY_OR_SECRET)
    if (
        raw.get("forged") is True
        or raw.get("forged_history") is True
        or raw.get("history_reviewed") is False
    ):
        reasons.add(REJECTION_FORGED)
    if raw.get("semantic_authority") is True:
        reasons.add(REJECTION_SEMANTIC_AUTHORITY_CLAIM)
    # Sufficiency cannot be established by name / type / similarity.
    for key in _SUFFICIENCY_KEYS:
        value = raw.get(key)
        if value is True or (
            isinstance(value, str) and value.strip().casefold() in {"true", "yes", "1"}
        ):
            reasons.add(REJECTION_SUFFICIENCY_CLAIM)
            break
    hard = raw.get("hard_gate_facts")
    if isinstance(hard, Mapping) and hard.get("information_sufficiency") is True:
        reasons.add(REJECTION_SUFFICIENCY_CLAIM)
    if raw.get("name_match_sufficient") is True or raw.get("type_match_sufficient") is True:
        reasons.add(REJECTION_SUFFICIENCY_CLAIM)
    if raw.get("similarity_sufficient") is True or raw.get("score_sufficient") is True:
        reasons.add(REJECTION_SUFFICIENCY_CLAIM)
    for key in _ADMISSION_COMPAT_KEYS:
        if raw.get(key) is True:
            reasons.add(REJECTION_COMPATIBILITY_AS_ADMISSION)
            break
    if raw.get("score_as_hard_gate") is True or raw.get("rank_overrides_gate") is True:
        reasons.add(REJECTION_SCORE_AS_HARD_GATE)
    # Stale / cross-root bindings.
    for key in _ROOT_KEYS:
        if key in raw and raw[key] not in (None, "", getattr(expected_roots, key)):
            reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    candidate_roots = raw.get("roots")
    if isinstance(candidate_roots, ProgramLogicAuthorityRoots):
        if not _roots_equal(candidate_roots, expected_roots):
            reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    elif isinstance(candidate_roots, Mapping):
        if any(
            key in candidate_roots
            and candidate_roots[key] not in (None, "", getattr(expected_roots, key))
            for key in _ROOT_KEYS
        ):
            reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    for alias_key, attr in (("tree_id", "tree_id"), ("forest_id", "forest_id")):
        if alias_key in raw and raw[alias_key] not in (
            None,
            "",
            getattr(expected_roots, attr),
        ):
            reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    # Cross gap / goal.
    for key, expected in (
        ("gap_id", query.gap_id),
        ("goal_id", query.goal_id),
        ("target_goal_id", query.goal_id),
        ("plan_id", query.plan_id),
    ):
        if key in raw and raw[key] not in (None, "", expected):
            reasons.add(REJECTION_CROSS_GAP_OR_GOAL)
    # Excluded premises.
    premises = _premise_ids(raw, query)
    excluded = set(query.excluded_premise_ids)
    if any(item in excluded for item in premises):
        reasons.add(REJECTION_EXCLUDED_PREMISE)
    # Vector / learned poisoning.
    if signal in {
        HypothesisSignal.VECTOR.value,
        HypothesisSignal.LEARNED_MODEL.value,
        HypothesisSignal.LEXICAL.value,
    }:
        try:
            score = raw.get(
                "score",
                raw.get("score_millionths", raw.get("nomination_score_millipercent", 0)),
            )
            if score not in (None, "") and not math.isfinite(float(score)):
                reasons.add(REJECTION_POISONED)
        except (TypeError, ValueError):
            reasons.add(REJECTION_POISONED)
        if raw.get("semantic_authority", False) is not False:
            reasons.add(REJECTION_POISONED)
        if vector_roots is not None:
            tree_id, index_id, model_id = vector_roots
            if raw.get("tree_id") not in (None, "", tree_id):
                reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
            if raw.get("index_id") not in (None, "", index_id):
                reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
            if raw.get("model_id") not in (None, "", model_id):
                reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    return reasons


def _raw_from_item(item: Any) -> dict[str, Any]:
    if isinstance(item, LogicHypothesis):
        return {
            "claimed_consequence_ref": item.claimed_consequence_ref,
            "construction_ref": item.construction_ref,
            "placement_ref": item.placement_ref,
            "value_ref": item.value_ref,
            "selected_premise_ids": item.selected_premise_ids,
            "counterexample_target_ref": item.counterexample_target_ref,
            "evidence_refs": item.evidence_refs,
            "nomination_score_millipercent": item.nomination_score_millipercent,
            "roots": item.roots,
            "target_goal_id": item.target_goal_id,
            "hypothesis_id": item.hypothesis_id,
            "semantic_authority": item.semantic_authority,
        }
    if isinstance(item, LogicHypothesisNomination):
        return {
            "claimed_consequence_ref": item.hypothesis.claimed_consequence_ref,
            "construction_ref": item.hypothesis.construction_ref,
            "placement_ref": item.hypothesis.placement_ref,
            "value_ref": item.hypothesis.value_ref,
            "selected_premise_ids": item.hypothesis.selected_premise_ids,
            "counterexample_target_ref": item.hypothesis.counterexample_target_ref,
            "nomination_score_millipercent": item.nomination_score_millipercent,
            "roots": item.hypothesis.roots,
            "hard_gate_facts": item.hard_gate_facts.to_dict(),
            "diagnostics": item.diagnostics,
            "evidence_refs": [
                ref.artifact_id
                for _, refs in item.signal_evidence
                for ref in refs
            ],
        }
    # Code-symbol / change-value vector hits (duck-typed to avoid hard import).
    row = getattr(item, "row", None)
    if row is not None and hasattr(item, "score"):
        path = getattr(row, "path", "") or ""
        name = (
            getattr(row, "qualified_name", None)
            or getattr(row, "name", None)
            or getattr(row, "row_id", None)
            or "symbol:unknown"
        )
        return {
            "claimed_consequence_ref": f"consequence:{name}",
            "construction_ref": str(name),
            "value_ref": str(name),
            "path": path,
            "score": getattr(item, "score", 0),
            "index_id": getattr(item, "index_id", ""),
            "query_id": getattr(item, "query_id", ""),
            "semantic_authority": getattr(item, "semantic_authority", False),
            "compatibility_claim": getattr(item, "compatibility_claim", False),
            "evidence_refs": (f"hit:{getattr(item, 'hit_id', name)}",),
            "type_ref": getattr(row, "type_ref", "") or "",
        }
    return _mapping(item)


def _hypothesis_id(consequence: str, construction: str, value: str, signals: set[str]) -> str:
    digest = _fingerprint(
        {
            "consequence": consequence,
            "construction": construction,
            "value": value,
            "signals": sorted(signals),
        },
        prefix="hyp-id",
    )
    return f"hypothesis:{digest.split(':')[-1][:24]}"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class ProgramLogicHypothesisRetriever:
    """Union bounded program-logic hypothesis signals into a diagnostic-only set."""

    def __init__(
        self,
        roots: ProgramLogicAuthorityRoots,
        *,
        bounds: HypothesisRetrievalBounds | None = None,
    ) -> None:
        if not isinstance(roots, ProgramLogicAuthorityRoots):
            raise HypothesisRetrievalBindingError(
                "roots must be ProgramLogicAuthorityRoots"
            )
        self.roots = roots
        self.bounds = bounds or HypothesisRetrievalBounds()

    def retrieve(
        self,
        gap: LogicGap,
        plan: TacticianSearchPlan,
        *,
        query: HypothesisQuery | None = None,
        candidates_by_signal: Mapping[str, Any] | None = None,
        corpus: Any = None,
        corpus_id: str = "",
        counterexample_target_ref: str = "",
        graph_id: str = "",
        vector_query_id: str = "",
        vector_tree_id: str = "",
        vector_index_id: str = "",
        vector_model_id: str = "",
        **signal_candidates: Any,
    ) -> HypothesisCandidateSet:
        if not isinstance(gap, LogicGap):
            raise HypothesisRetrievalBindingError("gap must be a typed LogicGap")
        if not isinstance(plan, TacticianSearchPlan):
            raise HypothesisRetrievalBindingError(
                "plan must be a typed TacticianSearchPlan"
            )
        if not _roots_equal(gap.roots, self.roots):
            raise HypothesisRetrievalBindingError(
                "gap and retriever must share exact roots"
            )
        if not _roots_equal(plan.roots, self.roots):
            raise HypothesisRetrievalBindingError(
                "plan and retriever must share exact roots"
            )
        if gap.goal_id not in plan.goal_ids:
            raise HypothesisRetrievalBindingError(
                "gap goal_id must be listed in plan goal_ids"
            )
        if plan.semantic_authority is not False:
            raise HypothesisRetrievalBindingError(
                "tactician plan must be non-authoritative"
            )
        if gap.semantic_authority is not False:
            raise HypothesisRetrievalBindingError(
                "logic gap must be non-authoritative"
            )

        bound_corpus_id = corpus_id or self.roots.corpus_id
        if corpus is not None:
            corpus_roots = getattr(corpus, "roots", None)
            if isinstance(corpus_roots, ProgramLogicAuthorityRoots) and not _roots_equal(
                corpus_roots, self.roots
            ):
                raise HypothesisRetrievalBindingError(
                    "premise corpus roots do not match retriever roots"
                )
            if getattr(corpus, "roots", None) is not None:
                # Prefer the corpus content id when present.
                content = getattr(corpus, "content_id", None) or getattr(
                    corpus, "corpus_id", None
                )
                if isinstance(content, str) and content:
                    # Prefer explicit corpus_id argument when provided.
                    if not corpus_id:
                        bound_corpus_id = content
            if (
                bound_corpus_id
                and self.roots.corpus_id
                and bound_corpus_id
                not in (self.roots.corpus_id, getattr(corpus, "content_id", None))
                and corpus_id
                and corpus_id != self.roots.corpus_id
                and corpus_id != getattr(corpus, "content_id", None)
            ):
                raise HypothesisRetrievalBindingError(
                    "supplied corpus_id does not match authority corpus_id"
                )

        if query is None:
            query = HypothesisQuery.from_gap_and_plan(
                gap,
                plan,
                corpus_id=bound_corpus_id,
                counterexample_target_ref=counterexample_target_ref,
            )
        if not isinstance(query, HypothesisQuery):
            raise HypothesisRetrievalBindingError("query must be HypothesisQuery")
        if not _roots_equal(query.roots, self.roots):
            raise HypothesisRetrievalBindingError(
                "query roots do not match retriever roots"
            )
        if query.gap_id != gap.gap_id or query.goal_id != gap.goal_id:
            raise HypothesisRetrievalBindingError(
                "query gap/goal does not match supplied gap"
            )
        if query.plan_id != plan.plan_id:
            raise HypothesisRetrievalBindingError(
                "query plan_id does not match supplied plan"
            )

        signal_roots: dict[str, str] = {
            HypothesisSignal.ANALYTICAL_CONSTRUCTION.value: self.roots.graph_id,
            HypothesisSignal.EXISTING_VALUE.value: self.roots.graph_id,
            HypothesisSignal.CONSTRUCTOR_ADAPTER.value: self.roots.graph_id,
            HypothesisSignal.THEOREM_PREMISE.value: self.roots.corpus_id,
            HypothesisSignal.DATAFLOW.value: self.roots.graph_id,
            HypothesisSignal.GRAPH_NEIGHBORHOOD.value: self.roots.graph_id,
            HypothesisSignal.SCHEMA.value: self.roots.index_id,
            HypothesisSignal.LINEAGE.value: self.roots.tree_id,
            HypothesisSignal.TEST_SPEC_ANALOGUE.value: self.roots.policy_id,
            HypothesisSignal.LEXICAL.value: self.roots.index_id,
            HypothesisSignal.VECTOR.value: self.roots.index_id,
            HypothesisSignal.TACTICIAN_SUBGOAL.value: plan.plan_id,
            HypothesisSignal.LEARNED_MODEL.value: self.roots.model_id,
        }
        bound_graph_id = graph_id or self.roots.graph_id
        if graph_id and graph_id != self.roots.graph_id:
            raise HypothesisRetrievalBindingError(
                "supplied graph_id does not match authority roots"
            )

        vector_roots: tuple[str, str, str] | None = None
        if vector_tree_id or vector_index_id or vector_model_id:
            tree_id = vector_tree_id or self.roots.tree_id
            index_id = vector_index_id or self.roots.index_id
            model_id = vector_model_id or self.roots.model_id
            if tree_id != self.roots.tree_id:
                raise HypothesisRetrievalBindingError(
                    "vector tree does not match authority tree_id"
                )
            if index_id != self.roots.index_id:
                raise HypothesisRetrievalBindingError(
                    "vector index does not match authority index_id"
                )
            if model_id != self.roots.model_id:
                raise HypothesisRetrievalBindingError(
                    "vector model does not match authority model_id"
                )
            vector_roots = (tree_id, index_id, model_id)
            signal_roots[HypothesisSignal.VECTOR.value] = index_id

        # Auto-project Tactician subgoals when not supplied.
        supplied = dict(candidates_by_signal or {})
        for name, value in signal_candidates.items():
            if value is not None:
                supplied[name] = value
        if (
            HypothesisSignal.TACTICIAN_SUBGOAL.value not in {
                _signal(name) for name in supplied
            }
            and plan.subgoals
        ):
            subgoal_hits = []
            for subgoal in plan.subgoals:
                if subgoal.goal_id != gap.goal_id:
                    continue
                subgoal_hits.append(
                    {
                        "claimed_consequence_ref": subgoal.claim_ref,
                        "construction_ref": f"subgoal:{subgoal.subgoal_id}",
                        "evidence_refs": (f"subgoal:{subgoal.subgoal_id}",),
                        "nomination_score_millipercent": subgoal.score_millipercent,
                        "selected_premise_ids": plan.selected_premise_ids,
                        "counterexample_target_ref": counterexample_target_ref
                        or query.counterexample_target_ref,
                    }
                )
            if subgoal_hits:
                supplied[HypothesisSignal.TACTICIAN_SUBGOAL.value] = tuple(subgoal_hits)

        # Project theorem premises from corpus when not supplied.
        if (
            corpus is not None
            and HypothesisSignal.THEOREM_PREMISE.value
            not in {_signal(name) for name in supplied}
        ):
            premises = getattr(corpus, "premises", ()) or ()
            premise_hits = []
            selected = set(plan.selected_premise_ids)
            for premise in premises:
                premise_id = getattr(premise, "premise_id", None) or getattr(
                    premise, "id", None
                )
                if not premise_id:
                    continue
                if selected and premise_id not in selected:
                    continue
                if premise_id in plan.excluded_premise_ids:
                    continue
                statement = getattr(premise, "statement_ref", None) or getattr(
                    premise, "statement_digest", None
                ) or premise_id
                premise_hits.append(
                    {
                        "claimed_consequence_ref": f"consequence:premise:{premise_id}",
                        "construction_ref": f"premise:{premise_id}",
                        "selected_premise_ids": (premise_id,),
                        "evidence_refs": (f"premise:{premise_id}",),
                        "information_content_ref": str(statement),
                    }
                )
            if premise_hits:
                if len(premise_hits) > self.bounds.max_candidates_per_signal:
                    raise HypothesisRetrievalBoundsError(
                        "theorem_premise exceeds max_candidates_per_signal"
                    )
                supplied[HypothesisSignal.THEOREM_PREMISE.value] = tuple(premise_hits)

        grouped: dict[str, list[Any]] = {}
        bound_vector_query_id = vector_query_id
        for raw_signal, value in supplied.items():
            signal = _signal(raw_signal)
            if value is None:
                entries: tuple[Any, ...] = ()
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray, Mapping)
            ):
                entries = tuple(value)
            else:
                # Duck-typed vector search result containers.
                hits = getattr(value, "hits", None)
                if hits is not None and hasattr(value, "query"):
                    if getattr(value, "semantic_authority", False) is not False:
                        raise HypothesisRetrievalBindingError(
                            "vector results must be non-authoritative"
                        )
                    if getattr(value, "complete", True) is not True:
                        raise HypothesisRetrievalBindingError(
                            "vector results must be complete"
                        )
                    entries = tuple(hits)
                    query_obj = getattr(value, "query", None)
                    if query_obj is not None:
                        bound_vector_query_id = (
                            bound_vector_query_id
                            or getattr(query_obj, "query_id", "")
                            or getattr(query_obj, "content_id", "")
                        )
                else:
                    entries = (value,)
            if len(entries) > self.bounds.max_candidates_per_signal:
                raise HypothesisRetrievalBoundsError(
                    f"{signal} exceeds max_candidates_per_signal"
                )
            grouped.setdefault(signal, []).extend(entries)

        aggregate: dict[tuple[Any, ...], dict[str, Any]] = {}
        for signal in sorted(grouped):
            entries = grouped[signal]
            if len(entries) > self.bounds.max_candidates_per_signal:
                raise HypothesisRetrievalBoundsError(
                    f"{signal} exceeds max_candidates_per_signal"
                )
            for item in entries:
                raw = _raw_from_item(item)
                if not isinstance(raw, Mapping):
                    raw = {"value": raw}
                consequence = _consequence_ref(raw, query)
                construction = _construction_ref(raw)
                value_ref = _value_ref(raw)
                placement = _placement_ref(raw)
                had_body = _contains_body_or_secret(raw)
                safe_raw = _redact_payload(raw) if had_body else dict(raw)
                if not isinstance(safe_raw, Mapping):
                    safe_raw = {"value": safe_raw}
                safe_raw = dict(safe_raw)
                safe_raw["claimed_consequence_ref"] = consequence
                safe_raw["construction_ref"] = construction
                safe_raw["value_ref"] = value_ref
                safe_raw["placement_ref"] = placement
                key = (consequence, construction, value_ref, placement)
                if safe_raw.get("partial") is True or not consequence:
                    key = key + (_fingerprint(safe_raw),)
                entry = aggregate.setdefault(
                    key,
                    {
                        "consequence": consequence,
                        "construction": construction,
                        "value_ref": value_ref,
                        "placement": placement,
                        "signals": set(),
                        "refs": {},
                        "reasons": set(),
                        "raw": [],
                        "scores": [],
                        "hard_facts": [],
                        "premises": set(),
                        "counterexamples": set(),
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
                    signal, safe_raw, self.roots, query, vector_roots
                )
                if had_body:
                    reasons.add(REJECTION_BODY_OR_SECRET)
                entry["reasons"].update(reasons)
                entry["raw"].append(safe_raw)
                try:
                    entry["scores"].append(_nomination_score(safe_raw))
                except (HypothesisRetrievalError, HypothesisRetrievalBoundsError):
                    entry["reasons"].add(REJECTION_POISONED)
                    entry["scores"].append(0)
                try:
                    entry["hard_facts"].append(_hard_gate_facts(safe_raw))
                except (HypothesisRetrievalError, HypothesisRetrievalBindingError):
                    entry["reasons"].add(REJECTION_SUFFICIENCY_CLAIM)
                    entry["hard_facts"].append(HypothesisHardGateFacts())
                for premise_id in _premise_ids(safe_raw, query):
                    entry["premises"].add(premise_id)
                cx = _counterexample_target(safe_raw, query)
                if cx:
                    entry["counterexamples"].add(cx)

        if not aggregate:
            # Empty retrieval is a valid, explicit diagnostic rather than an
            # implicit winner.
            raw = {
                "partial": True,
                "reason": "no_signal_candidates",
                "claimed_consequence_ref": f"consequence:missing:{gap.gap_id}",
            }
            aggregate[("empty", gap.gap_id)] = {
                "consequence": raw["claimed_consequence_ref"],
                "construction": "",
                "value_ref": "",
                "placement": "",
                "signals": set(),
                "refs": {},
                "reasons": {REJECTION_PARTIAL},
                "raw": [raw],
                "scores": [0],
                "hard_facts": [HypothesisHardGateFacts()],
                "premises": set(),
                "counterexamples": set(),
            }

        if len(aggregate) > self.bounds.max_candidates:
            raise HypothesisRetrievalBoundsError(
                "unioned candidate set exceeds max_candidates; refusing partial union"
            )

        # Detect ambiguity: same consequence, conflicting constructions under
        # multiple non-rejected aggregates → mark each as ambiguous.
        by_consequence: dict[str, list[dict[str, Any]]] = {}
        for entry in aggregate.values():
            by_consequence.setdefault(entry["consequence"], []).append(entry)
        ambiguous_consequences: set[str] = set()
        for consequence, entries in by_consequence.items():
            constructions = {
                (item["construction"], item["value_ref"], item["placement"])
                for item in entries
                if not item["reasons"]
            }
            if len(constructions) > 1:
                ambiguous_consequences.add(consequence)

        nominations: list[LogicHypothesisNomination] = []
        for entry in aggregate.values():
            signals = set(entry["signals"])
            reasons = set(entry["reasons"])
            is_empty = not signals and REJECTION_PARTIAL in reasons
            is_ambiguous = entry["consequence"] in ambiguous_consequences
            if is_ambiguous and not is_empty:
                reasons.add("hypothesis_ambiguous")
            score = max(entry["scores"]) if entry["scores"] else 0
            # Merge hard-gate facts conservatively (None if conflict).
            hard = _merge_hard_facts(entry["hard_facts"])
            premises = tuple(sorted(entry["premises"]))
            # Cap evidence per nomination.
            all_refs: list[HypothesisSignalRef] = []
            signal_evidence_rows: list[tuple[str, tuple[HypothesisSignalRef, ...]]] = []
            for signal, refs in sorted(entry["refs"].items()):
                unique = tuple(sorted(set(refs), key=lambda ref: ref.content_id))
                if len(unique) > self.bounds.max_evidence_per_nomination:
                    raise HypothesisRetrievalBoundsError(
                        "evidence exceeds max_evidence_per_nomination"
                    )
                signal_evidence_rows.append((signal, unique))
                all_refs.extend(unique)
            if len(all_refs) > self.bounds.max_evidence_per_nomination * max(
                1, len(signal_evidence_rows)
            ):
                # Soft cap already enforced per signal; total is bounded by
                # families × per-signal bound.
                pass
            evidence_ids = tuple(
                sorted({ref.artifact_id for ref in all_refs})
            )[: self.bounds.max_evidence_per_nomination]
            counterexample = ""
            if entry["counterexamples"]:
                counterexample = sorted(entry["counterexamples"])[0]
            elif query.counterexample_target_ref:
                counterexample = query.counterexample_target_ref

            route_kinds = _route_kinds_for_signals(signals)
            # Nominating routes force nominating authority.
            nominating = any(
                route
                in {
                    SourceRouteKind.VECTOR,
                    SourceRouteKind.KNOWLEDGE_GRAPH,
                    SourceRouteKind.TACTICIAN,
                    SourceRouteKind.LLM,
                    SourceRouteKind.SOLVER,
                    SourceRouteKind.RUNTIME_WITNESS,
                    SourceRouteKind.HISTORY,
                }
                for route in route_kinds
            )
            source_authority = (
                SourceAuthorityClass.NOMINATING
                if nominating or not route_kinds
                else SourceAuthorityClass.NOMINATING
            )
            # Retrieval never claims completeness or proof.
            hyp_disposition = HypothesisDisposition.NOMINATED
            if is_empty:
                hyp_disposition = HypothesisDisposition.ABSTAINED
            elif is_ambiguous:
                hyp_disposition = HypothesisDisposition.AMBIGUOUS
            elif reasons:
                if REJECTION_STALE_OR_CROSS_ROOT in reasons:
                    hyp_disposition = HypothesisDisposition.STALE
                else:
                    hyp_disposition = HypothesisDisposition.ABSTAINED

            hypothesis = LogicHypothesis(
                roots=self.roots,
                hypothesis_id=_hypothesis_id(
                    entry["consequence"],
                    entry["construction"],
                    entry["value_ref"],
                    signals,
                ),
                target_goal_id=query.goal_id,
                disposition=hyp_disposition,
                claimed_consequence_ref=entry["consequence"],
                construction_ref=entry["construction"],
                placement_ref=entry["placement"],
                value_ref=entry["value_ref"],
                evidence_refs=evidence_ids,
                evidence_route_kinds=route_kinds,
                selected_premise_ids=premises,
                counterexample_target_ref=counterexample,
                source_authority=source_authority,
                proof_status=ProofStatus.UNPROVED,
                completeness=False,
                unsupported_flags=tuple(sorted(reasons)) if reasons else (),
                nomination_score_millipercent=score,
                semantic_authority=False,
                invalidation_refs=(
                    self.roots.tree_id,
                    self.roots.corpus_id,
                    self.roots.graph_id,
                ),
            )

            if is_empty:
                disposition = HypothesisNominationDisposition.NO_CANDIDATE
            elif is_ambiguous:
                disposition = HypothesisNominationDisposition.AMBIGUOUS
            elif reasons:
                disposition = HypothesisNominationDisposition.REJECTED
            else:
                disposition = HypothesisNominationDisposition.NOMINATED

            nominations.append(
                LogicHypothesisNomination(
                    hypothesis=hypothesis,
                    disposition=disposition,
                    signal_evidence=tuple(signal_evidence_rows),
                    hard_gate_facts=hard,
                    nomination_score_millipercent=score,
                    diagnostics=tuple(sorted(reasons)),
                    semantic_authority=False,
                )
            )

        nominations.sort(key=lambda item: item.content_id)
        candidate_tuple = tuple(nominations)
        has_ambiguous = any(
            item.disposition is HypothesisNominationDisposition.AMBIGUOUS
            for item in candidate_tuple
        )
        only_no_candidate = all(
            item.disposition is HypothesisNominationDisposition.NO_CANDIDATE
            for item in candidate_tuple
        )
        return HypothesisCandidateSet(
            roots=self.roots,
            query=query,
            gap_id=gap.gap_id,
            plan_id=plan.plan_id,
            bounds=self.bounds,
            nominations=candidate_tuple,
            candidate_set_id=candidate_set_identity(candidate_tuple),
            signal_roots=tuple(signal_roots.items()),
            corpus_id=bound_corpus_id,
            graph_id=bound_graph_id,
            vector_query_id=bound_vector_query_id,
            ambiguous=has_ambiguous,
            no_candidate=only_no_candidate,
            semantic_authority=False,
        )

    nominate = retrieve
    search = retrieve


def _merge_hard_facts(
    facts: Sequence[HypothesisHardGateFacts],
) -> HypothesisHardGateFacts:
    if not facts:
        return HypothesisHardGateFacts()
    if len(facts) == 1:
        return facts[0]

    def _merge_bool(values: Sequence[bool | None]) -> bool | None:
        present = [item for item in values if item is not None]
        if not present:
            return None
        if all(item is True for item in present):
            return True
        if all(item is False for item in present):
            return False
        return None  # conflict → unknown

    info_refs = sorted(
        {item.information_content_ref for item in facts if item.information_content_ref}
    )
    return HypothesisHardGateFacts(
        type_compatible=_merge_bool([item.type_compatible for item in facts]),
        effect_compatible=_merge_bool([item.effect_compatible for item in facts]),
        auth_compatible=_merge_bool([item.auth_compatible for item in facts]),
        resource_compatible=_merge_bool([item.resource_compatible for item in facts]),
        lifecycle_compatible=_merge_bool([item.lifecycle_compatible for item in facts]),
        information_content_ref=info_refs[0] if len(info_refs) == 1 else "",
        information_sufficiency=False,
        same_name=any(item.same_name for item in facts),
        same_type=any(item.same_type for item in facts),
        similarity_millipercent=max(item.similarity_millipercent for item in facts),
        notes=tuple(sorted({note for item in facts for note in item.notes})),
    )


def retrieve_program_logic_hypotheses(
    roots: ProgramLogicAuthorityRoots,
    gap: LogicGap,
    plan: TacticianSearchPlan,
    **kwargs: Any,
) -> HypothesisCandidateSet:
    """Stateless convenience entry point for the retrieval-only boundary."""
    bounds = kwargs.pop("bounds", None)
    return ProgramLogicHypothesisRetriever(roots, bounds=bounds).retrieve(
        gap, plan, **kwargs
    )


__all__ = (
    "HYPOTHESIS_QUERY_SCHEMA",
    "HYPOTHESIS_SIGNAL_REF_SCHEMA",
    "HYPOTHESIS_HARD_GATE_FACTS_SCHEMA",
    "LOGIC_HYPOTHESIS_NOMINATION_SCHEMA",
    "HYPOTHESIS_CANDIDATE_SET_SCHEMA",
    "HYPOTHESIS_RETRIEVAL_BOUNDS_SCHEMA",
    "PRODUCER_ID",
    "MAX_CANDIDATE_COUNT",
    "SIGNAL_FAMILIES",
    "HypothesisSignal",
    "HypothesisNominationDisposition",
    "HypothesisRetrievalError",
    "HypothesisRetrievalBindingError",
    "HypothesisRetrievalBoundsError",
    "HypothesisSignalRef",
    "HypothesisRetrievalBounds",
    "HypothesisQuery",
    "HypothesisHardGateFacts",
    "LogicHypothesisNomination",
    "HypothesisCandidateSet",
    "ProgramLogicHypothesisRetriever",
    "retrieve_program_logic_hypotheses",
    "candidate_set_identity",
    "REJECTION_STALE_OR_CROSS_ROOT",
    "REJECTION_POISONED",
    "REJECTION_FORGED",
    "REJECTION_PARTIAL",
    "REJECTION_BODY_OR_SECRET",
    "REJECTION_SEMANTIC_AUTHORITY_CLAIM",
    "REJECTION_SUFFICIENCY_CLAIM",
    "REJECTION_COMPATIBILITY_AS_ADMISSION",
    "REJECTION_EXCLUDED_PREMISE",
    "REJECTION_INVALID_PAYLOAD",
    "REJECTION_CROSS_GAP_OR_GOAL",
    "REJECTION_SCORE_AS_HARD_GATE",
)
