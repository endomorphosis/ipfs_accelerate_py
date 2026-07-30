"""Fail-closed, non-authoritative nomination of missing-input value routes.

This adapter is deliberately a *recall* boundary.  For each
``MissingInputRequirement@1`` it unions bounded signal families (in-scope
symbols, receiver state, caller parameters, constants/defaults, request/
session context, reaching-definition hints, reviewed config/env providers,
DI/registry providers, factories/builders/constructors, schemas, lineage,
authoritative specs/tests, lexical/BM25, graph, and vector hits) into one
canonical receipt.

It never:

* claims ``semantic_authority``;
* asserts compatibility, placement, or write scope;
* selects a winner or code path;
* retains source bodies or secrets.

Later proof/admission (value provenance, synthesis) must consume the complete
receipt rather than an individual nomination.  Vector, graph, lexical, and
history signals remain nomination-only.
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
from .change_propagation_contracts import (
    GraphNodeRef,
    GraphProvenance,
    MissingInputRequirement,
    PropagationAuthorityRoots,
)
from .change_value_vector_index import (
    ChangeValueHit,
    ChangeValueIndexRow,
    ChangeValueIndexSnapshot,
    ChangeValueQuery,
    ChangeValueSearchResult,
)


MISSING_INPUT_QUERY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/missing-input-query@1"
)
VALUE_PROVENANCE_CANDIDATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/value-provenance-candidate@1"
)
CONSTRUCTION_ROUTE_CANDIDATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/construction-route-candidate@1"
)
MISSING_INPUT_NOMINATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/missing-input-candidate-nomination@1"
)
MISSING_INPUT_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/missing-input-candidate-receipt@1"
)
MISSING_INPUT_BOUNDS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/missing-input-candidate-retrieval-bounds@1"
)
MISSING_INPUT_SIGNAL_REF_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/missing-input-signal-ref@1"
)

PRODUCER_ID = "missing-input-candidate-retrieval@1"
MAX_CANDIDATE_COUNT = 256
DEFAULT_MAX_PER_SIGNAL = 64
MAX_REF_BYTES = 512
MAX_TEXT_BYTES = 1_024


class MissingInputRetrievalError(ValueError):
    """A missing-input signal cannot safely participate in a nomination."""


class MissingInputRetrievalBindingError(MissingInputRetrievalError):
    """A required requirement, root, graph, or vector binding was mixed."""


class MissingInputRetrievalBoundsError(MissingInputRetrievalError):
    """A producer attempted to exceed the fixed retrieval budget."""


class MissingInputSignal(str, Enum):
    """Closed signal families that may contribute a nomination hit."""

    IN_SCOPE_SYMBOL = "in_scope_symbol"
    RECEIVER_STATE = "receiver_state"
    CALLER_PARAMETER = "caller_parameter"
    CONSTANT_DEFAULT = "constant_default"
    REQUEST_SESSION_CONTEXT = "request_session_context"
    REACHING_DEFINITION = "reaching_definition"
    CONFIG_ENV_PROVIDER = "config_env_provider"
    DI_REGISTRY_PROVIDER = "di_registry_provider"
    FACTORY_BUILDER_CONSTRUCTOR = "factory_builder_constructor"
    SCHEMA = "schema"
    LINEAGE = "lineage"
    AUTHORITATIVE_SPEC_TEST = "authoritative_spec_test"
    LEXICAL_BM25 = "lexical_bm25"
    GRAPH = "graph"
    VECTOR = "vector"


class ConstructionRouteKind(str, Enum):
    """How a nominated source might later satisfy a missing input.

    Routes are classification labels only.  Retrieval never proves that a
    route is total, safe, or admitted.
    """

    REUSE = "reuse"
    THREAD = "thread"
    CONVERT = "convert"
    CONSTRUCT = "construct"
    NEW_BEHAVIOR = "new_behavior"


class MissingInputCandidateDisposition(str, Enum):
    NOMINATED = "nominated"
    REJECTED = "rejected"


SIGNAL_FAMILIES = tuple(item.value for item in MissingInputSignal)

_SIGNAL_ALIASES = {
    "scope": MissingInputSignal.IN_SCOPE_SYMBOL.value,
    "in_scope": MissingInputSignal.IN_SCOPE_SYMBOL.value,
    "symbol": MissingInputSignal.IN_SCOPE_SYMBOL.value,
    "local": MissingInputSignal.IN_SCOPE_SYMBOL.value,
    "local_name": MissingInputSignal.IN_SCOPE_SYMBOL.value,
    "receiver": MissingInputSignal.RECEIVER_STATE.value,
    "self": MissingInputSignal.RECEIVER_STATE.value,
    "parameter": MissingInputSignal.CALLER_PARAMETER.value,
    "parameters": MissingInputSignal.CALLER_PARAMETER.value,
    "caller_param": MissingInputSignal.CALLER_PARAMETER.value,
    "constant": MissingInputSignal.CONSTANT_DEFAULT.value,
    "default": MissingInputSignal.CONSTANT_DEFAULT.value,
    "defaults": MissingInputSignal.CONSTANT_DEFAULT.value,
    "request": MissingInputSignal.REQUEST_SESSION_CONTEXT.value,
    "session": MissingInputSignal.REQUEST_SESSION_CONTEXT.value,
    "context": MissingInputSignal.REQUEST_SESSION_CONTEXT.value,
    "request_context": MissingInputSignal.REQUEST_SESSION_CONTEXT.value,
    "session_context": MissingInputSignal.REQUEST_SESSION_CONTEXT.value,
    "reaching": MissingInputSignal.REACHING_DEFINITION.value,
    "reaching_def": MissingInputSignal.REACHING_DEFINITION.value,
    "rd": MissingInputSignal.REACHING_DEFINITION.value,
    "config": MissingInputSignal.CONFIG_ENV_PROVIDER.value,
    "env": MissingInputSignal.CONFIG_ENV_PROVIDER.value,
    "environment": MissingInputSignal.CONFIG_ENV_PROVIDER.value,
    "config_provider": MissingInputSignal.CONFIG_ENV_PROVIDER.value,
    "di": MissingInputSignal.DI_REGISTRY_PROVIDER.value,
    "registry": MissingInputSignal.DI_REGISTRY_PROVIDER.value,
    "injection": MissingInputSignal.DI_REGISTRY_PROVIDER.value,
    "di_container": MissingInputSignal.DI_REGISTRY_PROVIDER.value,
    "factory": MissingInputSignal.FACTORY_BUILDER_CONSTRUCTOR.value,
    "builder": MissingInputSignal.FACTORY_BUILDER_CONSTRUCTOR.value,
    "constructor": MissingInputSignal.FACTORY_BUILDER_CONSTRUCTOR.value,
    "constructors": MissingInputSignal.FACTORY_BUILDER_CONSTRUCTOR.value,
    "schema_default": MissingInputSignal.SCHEMA.value,
    "lineage_history": MissingInputSignal.LINEAGE.value,
    "history": MissingInputSignal.LINEAGE.value,
    "spec": MissingInputSignal.AUTHORITATIVE_SPEC_TEST.value,
    "specs": MissingInputSignal.AUTHORITATIVE_SPEC_TEST.value,
    "test": MissingInputSignal.AUTHORITATIVE_SPEC_TEST.value,
    "tests": MissingInputSignal.AUTHORITATIVE_SPEC_TEST.value,
    "lexical": MissingInputSignal.LEXICAL_BM25.value,
    "bm25": MissingInputSignal.LEXICAL_BM25.value,
    "kg": MissingInputSignal.GRAPH.value,
    "dependency_graph": MissingInputSignal.GRAPH.value,
    "embedding": MissingInputSignal.VECTOR.value,
    "vector_hit": MissingInputSignal.VECTOR.value,
}

# Stable public diagnostics.  Do not change without a versioned receipt schema.
REJECTION_STALE_OR_CROSS_ROOT = "stale_or_cross_root"
REJECTION_POISONED = "poisoned_signal"
REJECTION_FORGED = "forged_result"
REJECTION_PARTIAL = "partial_candidate"
REJECTION_BODY_OR_SECRET = "body_or_secret_payload"
REJECTION_COMPATIBILITY_CLAIM = "compatibility_claim"
REJECTION_WRITE_SCOPE_CLAIM = "write_scope_claim"
REJECTION_PLACEMENT_CLAIM = "placement_claim"
REJECTION_SEMANTIC_AUTHORITY_CLAIM = "semantic_authority_claim"
REJECTION_FORBIDDEN_CONFIG_ENV = "forbidden_config_env"
REJECTION_CONFLICTING_ROUTES = "conflicting_route_signals"
REJECTION_INVALID_PAYLOAD = "invalid_candidate_payload"

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
        "private_witness",
        "private_premise",
    }
)
_COMPATIBILITY_KEYS = frozenset(
    {
        "compatible",
        "compatibility",
        "compatibility_claim",
        "type_compatible",
        "semantically_compatible",
        "is_compatible",
        "admits_compatibility",
        "proved_compatible",
    }
)
_WRITE_SCOPE_KEYS = frozenset(
    {
        "write_paths",
        "write_scope",
        "candidate_write_paths",
        "permitted_write_paths",
        "mutation_paths",
        "edit_paths",
    }
)
_PLACEMENT_KEYS = frozenset(
    {
        "placement",
        "placement_path",
        "placement_decision",
        "chosen_placement",
        "admitted_placement",
        "write_target",
    }
)
_ROOT_KEYS = (
    "repository_id",
    "base_forest_id",
    "base_tree_id",
    "base_overlay_id",
    "candidate_forest_id",
    "candidate_tree_id",
    "candidate_overlay_id",
    "graph_id",
    "index_id",
    "model_id",
    "config_id",
    "translator_id",
    "toolchain_id",
    "policy_id",
)
# Match secret *values*, not identifiers that merely contain the substring
# "secret" (e.g. expression refs like ``expr:secret_body``).
_SECRET_VALUE_RE = re.compile(
    r"(?:^|[^a-z0-9_])(?:api[_-]?key|password|secret|token|passwd)"
    r"(?:[^a-z0-9_]|$)|"
    r"bearer\s+[a-z0-9._\-]{8,}|"
    r"-----begin\s+",
    re.IGNORECASE,
)


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


def _fingerprint(value: Any, *, prefix: str = "missing-input") -> str:
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
        raise MissingInputRetrievalError(f"{name} is required")
    if "\x00" in text or len(text.encode("utf-8")) > limit:
        raise MissingInputRetrievalBoundsError(f"{name} is invalid or exceeds its bound")
    return text


def _signal(name: Any) -> str:
    normalized = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = _SIGNAL_ALIASES.get(normalized, normalized)
    if normalized not in SIGNAL_FAMILIES:
        raise MissingInputRetrievalError(f"unsupported missing-input signal: {name}")
    return normalized


def _route(name: Any) -> ConstructionRouteKind:
    if isinstance(name, ConstructionRouteKind):
        return name
    normalized = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "reuse_existing": ConstructionRouteKind.REUSE.value,
        "existing": ConstructionRouteKind.REUSE.value,
        "thread_upward": ConstructionRouteKind.THREAD.value,
        "thread_parameter": ConstructionRouteKind.THREAD.value,
        "conversion": ConstructionRouteKind.CONVERT.value,
        "adapter": ConstructionRouteKind.CONVERT.value,
        "construction": ConstructionRouteKind.CONSTRUCT.value,
        "factory": ConstructionRouteKind.CONSTRUCT.value,
        "builder": ConstructionRouteKind.CONSTRUCT.value,
        "constructor": ConstructionRouteKind.CONSTRUCT.value,
        "new": ConstructionRouteKind.NEW_BEHAVIOR.value,
        "new_type": ConstructionRouteKind.NEW_BEHAVIOR.value,
        "synthesize": ConstructionRouteKind.NEW_BEHAVIOR.value,
        "support_behavior": ConstructionRouteKind.NEW_BEHAVIOR.value,
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return ConstructionRouteKind(normalized)
    except ValueError as exc:
        raise MissingInputRetrievalError(f"unsupported construction route: {name}") from exc


def _verify_record_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    claimed = payload.get("content_id", payload.get("cid", ""))
    if claimed not in (None, "", record.content_id):
        raise MissingInputRetrievalBindingError(
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


def candidate_set_identity(candidates: Sequence["ValueProvenanceCandidate"]) -> str:
    """Bind the complete, deterministically ordered candidate set."""
    if not candidates:
        raise MissingInputRetrievalBoundsError("candidate set must be nonempty")
    if len(candidates) > MAX_CANDIDATE_COUNT:
        raise MissingInputRetrievalBoundsError("candidate set exceeds hard bound")
    ids = tuple(sorted(item.content_id for item in candidates))
    if len(set(ids)) != len(ids):
        raise MissingInputRetrievalError("candidate set contains duplicate candidates")
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/missing-input-candidate-set@1",
            "candidate_ids": list(ids),
        }
    )


@dataclass(frozen=True)
class MissingInputSignalRef(CanonicalContract):
    """Compact per-signal evidence pointer; never holds bodies."""

    SCHEMA: ClassVar[str] = MISSING_INPUT_SIGNAL_REF_SCHEMA

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
    def from_dict(cls, payload: Mapping[str, Any]) -> "MissingInputSignalRef":
        allowed = {"schema", "content_id", "cid", "signal", "artifact_id", "locator", "producer_id"}
        if not isinstance(payload, Mapping) or set(payload).difference(allowed):
            raise MissingInputRetrievalError("unsupported missing-input signal ref payload")
        if payload.get("schema") not in (None, cls.SCHEMA):
            raise MissingInputRetrievalError("unsupported missing-input signal ref schema")
        value = cls(
            signal=payload.get("signal", ""),
            artifact_id=payload.get("artifact_id", ""),
            locator=payload.get("locator", ""),
            producer_id=payload.get("producer_id", PRODUCER_ID),
        )
        _verify_record_identity(payload, value)
        return value


def _refs(value: Any, signal: str, raw: Mapping[str, Any]) -> tuple[MissingInputSignalRef, ...]:
    if value is None:
        values: Iterable[Any] = ()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        values = value
    else:
        values = (value,)
    refs: list[MissingInputSignalRef] = []
    for item in values:
        try:
            if isinstance(item, MissingInputSignalRef):
                ref = item
            elif isinstance(item, Mapping):
                ref = MissingInputSignalRef(
                    signal=str(item.get("signal", signal)),
                    artifact_id=str(item.get("artifact_id", item.get("locator", ""))),
                    locator=str(item.get("locator", "")),
                    producer_id=str(item.get("producer_id", PRODUCER_ID)),
                )
            elif isinstance(item, str) and item.strip():
                ref = MissingInputSignalRef(signal=signal, artifact_id=item.strip())
            else:
                continue
        except (KeyError, MissingInputRetrievalError, TypeError):
            continue
        if ref not in refs:
            refs.append(ref)
    if not refs:
        refs.append(
            MissingInputSignalRef(
                signal=signal,
                artifact_id=_fingerprint(raw, prefix="signal-artifact"),
            )
        )
    return tuple(sorted(refs, key=lambda item: item.content_id))


@dataclass(frozen=True)
class MissingInputRetrievalBounds(CanonicalContract):
    """Fixed, replayable caps; over-budget input is rejected, never truncated."""

    SCHEMA: ClassVar[str] = MISSING_INPUT_BOUNDS_SCHEMA

    max_candidates: int = MAX_CANDIDATE_COUNT
    max_candidates_per_signal: int = DEFAULT_MAX_PER_SIGNAL

    def __post_init__(self) -> None:
        for name in ("max_candidates", "max_candidates_per_signal"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_CANDIDATE_COUNT:
                raise MissingInputRetrievalBoundsError(
                    f"{name} must be an integer from 1 through {MAX_CANDIDATE_COUNT}"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "max_candidates_per_signal": self.max_candidates_per_signal,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MissingInputRetrievalBounds":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "max_candidates",
            "max_candidates_per_signal",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise MissingInputRetrievalError("unsupported missing-input retrieval bounds payload")
        value = cls(
            max_candidates=payload.get("max_candidates", MAX_CANDIDATE_COUNT),
            max_candidates_per_signal=payload.get(
                "max_candidates_per_signal", DEFAULT_MAX_PER_SIGNAL
            ),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class MissingInputQuery(CanonicalContract):
    """Exact binding of a missing-input retrieval query to authority roots."""

    SCHEMA: ClassVar[str] = MISSING_INPUT_QUERY_SCHEMA

    roots: PropagationAuthorityRoots
    requirement_id: str
    obligation_id: str
    clause_id: str
    parameter_name: str
    type_ref: str
    information_content_ref: str
    consumer_path: str = ""
    consumer_node_id: str = ""
    consumer_context_refs: tuple[str, ...] = ()
    missing_contract_refs: tuple[str, ...] = ()
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise MissingInputRetrievalBindingError("query roots must be PropagationAuthorityRoots")
        for name in (
            "requirement_id",
            "obligation_id",
            "clause_id",
            "parameter_name",
            "type_ref",
            "information_content_ref",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "consumer_path", _text(self.consumer_path, "consumer_path", required=False)
        )
        object.__setattr__(
            self,
            "consumer_node_id",
            _text(self.consumer_node_id, "consumer_node_id", required=False),
        )
        refs = tuple(
            sorted(
                {
                    _text(item, "consumer_context_refs")
                    for item in (self.consumer_context_refs or ())
                    if str(item).strip()
                }
            )
        )
        object.__setattr__(self, "consumer_context_refs", refs)
        contracts = tuple(
            sorted(
                {
                    _text(item, "missing_contract_refs")
                    for item in (self.missing_contract_refs or ())
                    if str(item).strip()
                }
            )
        )
        object.__setattr__(self, "missing_contract_refs", contracts)
        if not self.consumer_path and not self.consumer_node_id and not self.consumer_context_refs:
            raise MissingInputRetrievalBindingError(
                "query requires consumer context (path, node, or refs)"
            )
        if self.semantic_authority is not False:
            raise MissingInputRetrievalBindingError(
                "missing-input queries cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    @classmethod
    def from_requirement(
        cls,
        requirement: MissingInputRequirement,
        *,
        consumer_path: str = "",
        consumer_node_id: str = "",
        consumer_context_refs: Sequence[str] = (),
        missing_contract_refs: Sequence[str] = (),
    ) -> "MissingInputQuery":
        if not isinstance(requirement, MissingInputRequirement):
            raise MissingInputRetrievalBindingError(
                "query requires a typed MissingInputRequirement"
            )
        return cls(
            roots=requirement.roots,
            requirement_id=requirement.requirement_id,
            obligation_id=requirement.obligation_id,
            clause_id=requirement.clause_id,
            parameter_name=requirement.parameter_name,
            type_ref=requirement.type_ref,
            information_content_ref=requirement.information_content_ref,
            consumer_path=consumer_path,
            consumer_node_id=consumer_node_id,
            consumer_context_refs=tuple(consumer_context_refs),
            missing_contract_refs=tuple(missing_contract_refs)
            or (requirement.clause_id,),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "roots": self.roots.to_dict(),
            "requirement_id": self.requirement_id,
            "obligation_id": self.obligation_id,
            "clause_id": self.clause_id,
            "parameter_name": self.parameter_name,
            "type_ref": self.type_ref,
            "information_content_ref": self.information_content_ref,
            "consumer_path": self.consumer_path,
            "consumer_node_id": self.consumer_node_id,
            "consumer_context_refs": list(self.consumer_context_refs),
            "missing_contract_refs": list(self.missing_contract_refs),
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MissingInputQuery":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "roots",
            "requirement_id",
            "obligation_id",
            "clause_id",
            "parameter_name",
            "type_ref",
            "information_content_ref",
            "consumer_path",
            "consumer_node_id",
            "consumer_context_refs",
            "missing_contract_refs",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise MissingInputRetrievalError("unsupported missing-input query payload")
        roots = payload.get("roots")
        value = cls(
            roots=roots
            if isinstance(roots, PropagationAuthorityRoots)
            else PropagationAuthorityRoots.from_dict(roots),
            requirement_id=payload.get("requirement_id", ""),
            obligation_id=payload.get("obligation_id", ""),
            clause_id=payload.get("clause_id", ""),
            parameter_name=payload.get("parameter_name", ""),
            type_ref=payload.get("type_ref", ""),
            information_content_ref=payload.get("information_content_ref", ""),
            consumer_path=payload.get("consumer_path", ""),
            consumer_node_id=payload.get("consumer_node_id", ""),
            consumer_context_refs=tuple(payload.get("consumer_context_refs", ())),
            missing_contract_refs=tuple(payload.get("missing_contract_refs", ())),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class ConstructionRouteCandidate(CanonicalContract):
    """Classification of how a nominated source might satisfy a missing input."""

    SCHEMA: ClassVar[str] = CONSTRUCTION_ROUTE_CANDIDATE_SCHEMA

    route: ConstructionRouteKind
    expression_ref: str
    source_node_id: str = ""
    conversion_ref: str = ""
    factory_ref: str = ""
    dependency_refs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", _route(self.route))
        object.__setattr__(
            self, "expression_ref", _text(self.expression_ref, "expression_ref")
        )
        object.__setattr__(
            self,
            "source_node_id",
            _text(self.source_node_id, "source_node_id", required=False),
        )
        object.__setattr__(
            self, "conversion_ref", _text(self.conversion_ref, "conversion_ref", required=False)
        )
        object.__setattr__(
            self, "factory_ref", _text(self.factory_ref, "factory_ref", required=False)
        )
        deps = tuple(
            sorted(
                {
                    _text(item, "dependency_refs")
                    for item in (self.dependency_refs or ())
                    if str(item).strip()
                }
            )
        )
        object.__setattr__(self, "dependency_refs", deps)
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
        if self.semantic_authority is not False:
            raise MissingInputRetrievalBindingError(
                "construction routes cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "route": self.route.value,
            "expression_ref": self.expression_ref,
            "source_node_id": self.source_node_id,
            "conversion_ref": self.conversion_ref,
            "factory_ref": self.factory_ref,
            "dependency_refs": list(self.dependency_refs),
            "notes": list(self.notes),
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConstructionRouteCandidate":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "route",
            "expression_ref",
            "source_node_id",
            "conversion_ref",
            "factory_ref",
            "dependency_refs",
            "notes",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise MissingInputRetrievalError("unsupported construction route candidate payload")
        value = cls(
            route=payload.get("route", ""),
            expression_ref=payload.get("expression_ref", ""),
            source_node_id=payload.get("source_node_id", ""),
            conversion_ref=payload.get("conversion_ref", ""),
            factory_ref=payload.get("factory_ref", ""),
            dependency_refs=tuple(payload.get("dependency_refs", ())),
            notes=tuple(payload.get("notes", ())),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class ValueProvenanceCandidate(CanonicalContract):
    """One non-authoritative missing-input value nomination.

    Retrieval records exact scope hints, expression refs, type refs, and route
    classification only.  It cannot prove path conditions, information
    sufficiency, compatibility, placement, or write scope.
    """

    SCHEMA: ClassVar[str] = VALUE_PROVENANCE_CANDIDATE_SCHEMA

    roots: PropagationAuthorityRoots
    requirement_id: str
    expression_ref: str
    type_ref: str
    route: ConstructionRouteCandidate
    source_node: GraphNodeRef | None = None
    information_content_ref: str = ""
    signal_refs: tuple[MissingInputSignalRef, ...] = ()
    diagnostics: tuple[str, ...] = ()
    semantic_authority: bool = False
    compatibility_claim: bool = False
    placement_claim: bool = False
    write_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise MissingInputRetrievalBindingError(
                "value candidate roots must be PropagationAuthorityRoots"
            )
        object.__setattr__(
            self, "requirement_id", _text(self.requirement_id, "requirement_id")
        )
        object.__setattr__(
            self, "expression_ref", _text(self.expression_ref, "expression_ref")
        )
        object.__setattr__(self, "type_ref", _text(self.type_ref, "type_ref"))
        if not isinstance(self.route, ConstructionRouteCandidate):
            raise MissingInputRetrievalError(
                "value candidate requires ConstructionRouteCandidate"
            )
        if self.source_node is not None and not isinstance(self.source_node, GraphNodeRef):
            raise MissingInputRetrievalError("source_node must be GraphNodeRef when present")
        object.__setattr__(
            self,
            "information_content_ref",
            _text(self.information_content_ref, "information_content_ref", required=False),
        )
        refs = tuple(
            sorted(
                (
                    item
                    if isinstance(item, MissingInputSignalRef)
                    else MissingInputSignalRef.from_dict(item)
                    for item in (self.signal_refs or ())
                ),
                key=lambda item: item.content_id,
            )
        )
        object.__setattr__(self, "signal_refs", refs)
        diagnostics = tuple(
            sorted({str(item).strip() for item in (self.diagnostics or ()) if str(item).strip()})
        )
        object.__setattr__(self, "diagnostics", diagnostics)
        if self.semantic_authority is not False:
            raise MissingInputRetrievalBindingError(
                "value provenance candidates cannot claim semantic authority"
            )
        if self.compatibility_claim is not False:
            raise MissingInputRetrievalBindingError(
                "retrieval cannot assert compatibility"
            )
        if self.placement_claim is not False:
            raise MissingInputRetrievalBindingError(
                "retrieval cannot assert placement"
            )
        if self.write_paths:
            raise MissingInputRetrievalBindingError(
                "retrieval cannot assert write scope"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "compatibility_claim", False)
        object.__setattr__(self, "placement_claim", False)
        object.__setattr__(self, "write_paths", ())

    @property
    def route_kind(self) -> ConstructionRouteKind:
        return self.route.route

    @property
    def write_scope(self) -> tuple[str, ...]:
        """Retrieval never provides mutation authority."""
        return ()

    def _payload(self) -> dict[str, Any]:
        return {
            "roots": self.roots.to_dict(),
            "requirement_id": self.requirement_id,
            "expression_ref": self.expression_ref,
            "type_ref": self.type_ref,
            "route": self.route.to_dict(),
            "source_node": None if self.source_node is None else self.source_node.to_dict(),
            "information_content_ref": self.information_content_ref,
            "signal_refs": [ref.to_dict() for ref in self.signal_refs],
            "diagnostics": list(self.diagnostics),
            "semantic_authority": False,
            "compatibility_claim": False,
            "placement_claim": False,
            "write_paths": [],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValueProvenanceCandidate":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "roots",
            "requirement_id",
            "expression_ref",
            "type_ref",
            "route",
            "source_node",
            "information_content_ref",
            "signal_refs",
            "diagnostics",
            "semantic_authority",
            "compatibility_claim",
            "placement_claim",
            "write_paths",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise MissingInputRetrievalError("unsupported value provenance candidate payload")
        roots = payload.get("roots")
        route = payload.get("route")
        source = payload.get("source_node")
        refs = payload.get("signal_refs", ())
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
            raise MissingInputRetrievalError("signal_refs must be a sequence")
        value = cls(
            roots=roots
            if isinstance(roots, PropagationAuthorityRoots)
            else PropagationAuthorityRoots.from_dict(roots),
            requirement_id=payload.get("requirement_id", ""),
            expression_ref=payload.get("expression_ref", ""),
            type_ref=payload.get("type_ref", ""),
            route=route
            if isinstance(route, ConstructionRouteCandidate)
            else ConstructionRouteCandidate.from_dict(route),
            source_node=(
                None
                if source in (None, "")
                else (
                    source
                    if isinstance(source, GraphNodeRef)
                    else GraphNodeRef.from_dict(source)
                )
            ),
            information_content_ref=payload.get("information_content_ref", ""),
            signal_refs=tuple(
                item
                if isinstance(item, MissingInputSignalRef)
                else MissingInputSignalRef.from_dict(item)
                for item in refs
            ),
            diagnostics=tuple(payload.get("diagnostics", ())),
            semantic_authority=payload.get("semantic_authority", False),
            compatibility_claim=payload.get("compatibility_claim", False),
            placement_claim=payload.get("placement_claim", False),
            write_paths=tuple(payload.get("write_paths", ())),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class MissingInputCandidateNomination(CanonicalContract):
    """One candidate plus complete per-signal provenance and no authority."""

    SCHEMA: ClassVar[str] = MISSING_INPUT_NOMINATION_SCHEMA

    candidate: ValueProvenanceCandidate
    disposition: MissingInputCandidateDisposition
    signal_evidence: tuple[tuple[str, tuple[MissingInputSignalRef, ...]], ...]
    diagnostics: tuple[str, ...] = ()
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ValueProvenanceCandidate):
            raise MissingInputRetrievalError(
                "nomination requires ValueProvenanceCandidate"
            )
        object.__setattr__(
            self, "disposition", MissingInputCandidateDisposition(self.disposition)
        )
        rows: list[tuple[str, tuple[MissingInputSignalRef, ...]]] = []
        raw_evidence = (
            self.signal_evidence.items()
            if isinstance(self.signal_evidence, Mapping)
            else self.signal_evidence
        )
        for item in raw_evidence:
            try:
                signal, refs = item
            except (TypeError, ValueError) as exc:
                raise MissingInputRetrievalError(
                    "signal evidence rows must contain signal and references"
                ) from exc
            normalized = _signal(signal)
            checked = tuple(
                ref
                if isinstance(ref, MissingInputSignalRef)
                else MissingInputSignalRef.from_dict(ref)
                for ref in (
                    refs
                    if isinstance(refs, Sequence)
                    and not isinstance(refs, (str, bytes, bytearray, Mapping))
                    else (refs,)
                )
            )
            checked = tuple(sorted(checked, key=lambda ref: ref.content_id))
            rows.append((normalized, checked))
        rows.sort(key=lambda item: item[0])
        if len({item[0] for item in rows}) != len(rows):
            raise MissingInputRetrievalError("nomination has duplicate signal evidence")
        object.__setattr__(self, "signal_evidence", tuple(rows))
        diagnostics = tuple(
            sorted({str(item).strip() for item in (self.diagnostics or ()) if str(item).strip()})
        )
        object.__setattr__(self, "diagnostics", diagnostics)
        if self.semantic_authority is not False:
            raise MissingInputRetrievalBindingError(
                "nominations cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        if (
            self.disposition is MissingInputCandidateDisposition.NOMINATED
            and diagnostics
        ):
            raise MissingInputRetrievalError(
                "nominated candidates cannot carry rejection diagnostics"
            )
        if (
            self.disposition is MissingInputCandidateDisposition.REJECTED
            and not diagnostics
        ):
            raise MissingInputRetrievalError(
                "rejected candidates require stable diagnostics"
            )

    @property
    def route_kind(self) -> ConstructionRouteKind:
        return self.candidate.route_kind

    @property
    def write_paths(self) -> tuple[str, ...]:
        return ()

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
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MissingInputCandidateNomination":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "candidate",
            "disposition",
            "signal_evidence",
            "diagnostics",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise MissingInputRetrievalError("unsupported missing-input nomination payload")
        signal_evidence: list[tuple[str, tuple[MissingInputSignalRef, ...]]] = []
        supplied = payload.get("signal_evidence", ())
        if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes, bytearray)):
            raise MissingInputRetrievalError("signal_evidence must be a sequence")
        for row in supplied:
            if not isinstance(row, Mapping):
                raise MissingInputRetrievalError("signal evidence row must be an object")
            refs = row.get("evidence_refs", ())
            if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
                raise MissingInputRetrievalError("signal evidence references must be a sequence")
            signal_evidence.append(
                (
                    str(row.get("signal", "")),
                    tuple(
                        item
                        if isinstance(item, MissingInputSignalRef)
                        else MissingInputSignalRef.from_dict(item)
                        for item in refs
                    ),
                )
            )
        candidate = payload.get("candidate")
        value = cls(
            candidate=candidate
            if isinstance(candidate, ValueProvenanceCandidate)
            else ValueProvenanceCandidate.from_dict(candidate),
            disposition=payload.get("disposition", ""),
            signal_evidence=tuple(signal_evidence),
            diagnostics=tuple(payload.get("diagnostics", ())),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


@dataclass(frozen=True)
class MissingInputCandidateReceipt(CanonicalContract):
    """The complete bounded candidate set; this is not a target decision."""

    SCHEMA: ClassVar[str] = MISSING_INPUT_RECEIPT_SCHEMA

    roots: PropagationAuthorityRoots
    query: MissingInputQuery
    requirement_id: str
    bounds: MissingInputRetrievalBounds
    candidates: tuple[MissingInputCandidateNomination, ...]
    candidate_set_id: str
    signal_roots: tuple[tuple[str, str], ...] = ()
    vector_query_id: str = ""
    graph_id: str = ""
    semantic_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.roots, PropagationAuthorityRoots):
            raise MissingInputRetrievalError("receipt roots must be PropagationAuthorityRoots")
        if not isinstance(self.query, MissingInputQuery):
            raise MissingInputRetrievalError("receipt query must be MissingInputQuery")
        if not isinstance(self.bounds, MissingInputRetrievalBounds):
            raise MissingInputRetrievalError("receipt bounds must be MissingInputRetrievalBounds")
        if self.query.roots != self.roots:
            raise MissingInputRetrievalBindingError("query roots do not match receipt roots")
        object.__setattr__(
            self, "requirement_id", _text(self.requirement_id, "requirement_id")
        )
        if self.requirement_id != self.query.requirement_id:
            raise MissingInputRetrievalBindingError(
                "receipt requirement_id does not match query"
            )
        candidates = tuple(sorted(self.candidates, key=lambda item: item.content_id))
        if not candidates or len(candidates) > self.bounds.max_candidates:
            raise MissingInputRetrievalBoundsError(
                "receipt candidate count is outside its declared bound"
            )
        if any(not isinstance(item, MissingInputCandidateNomination) for item in candidates):
            raise MissingInputRetrievalError("receipt candidates must be nominations")
        if len({item.content_id for item in candidates}) != len(candidates):
            raise MissingInputRetrievalError("receipt contains duplicate nominations")
        if any(item.candidate.roots != self.roots for item in candidates):
            raise MissingInputRetrievalBindingError(
                "candidate roots do not match receipt roots"
            )
        if any(
            item.candidate.requirement_id != self.requirement_id for item in candidates
        ):
            raise MissingInputRetrievalBindingError(
                "candidate requirement_id does not match receipt"
            )
        object.__setattr__(self, "candidates", candidates)
        expected = candidate_set_identity(tuple(item.candidate for item in candidates))
        if self.candidate_set_id != expected:
            raise MissingInputRetrievalBindingError(
                "candidate_set_id does not bind the complete candidate set"
            )
        roots: list[tuple[str, str]] = []
        for signal, root in self.signal_roots:
            normalized = _signal(signal)
            if not isinstance(root, str) or not root:
                raise MissingInputRetrievalBindingError(
                    "signal roots must be nonempty identities"
                )
            roots.append((normalized, root))
        roots.sort()
        if len({item[0] for item in roots}) != len(roots):
            raise MissingInputRetrievalBindingError("receipt contains duplicate signal roots")
        object.__setattr__(self, "signal_roots", tuple(roots))
        object.__setattr__(
            self, "vector_query_id", _text(self.vector_query_id, "vector_query_id", required=False)
        )
        object.__setattr__(
            self, "graph_id", _text(self.graph_id, "graph_id", required=False)
        )
        if self.semantic_authority is not False:
            raise MissingInputRetrievalBindingError(
                "retrieval receipts cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)

    @property
    def value_candidates(self) -> tuple[ValueProvenanceCandidate, ...]:
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
            "requirement_id": self.requirement_id,
            "bounds": self.bounds.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "candidate_set_id": self.candidate_set_id,
            "signal_roots": [
                {"signal": signal, "root_id": root} for signal, root in self.signal_roots
            ],
            "vector_query_id": self.vector_query_id,
            "graph_id": self.graph_id,
            "semantic_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MissingInputCandidateReceipt":
        allowed = {
            "schema",
            "content_id",
            "cid",
            "roots",
            "query",
            "requirement_id",
            "bounds",
            "candidates",
            "candidate_set_id",
            "signal_roots",
            "vector_query_id",
            "graph_id",
            "semantic_authority",
        }
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != cls.SCHEMA
            or set(payload).difference(allowed)
        ):
            raise MissingInputRetrievalError("unsupported missing-input candidate receipt payload")
        rows = payload.get("signal_roots", ())
        candidates = payload.get("candidates", ())
        if (
            not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes, bytearray))
            or not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes, bytearray))
        ):
            raise MissingInputRetrievalError(
                "receipt signal roots and candidates must be sequences"
            )
        signal_roots: list[tuple[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise MissingInputRetrievalError("receipt signal root row must be an object")
            signal_roots.append((str(row.get("signal", "")), str(row.get("root_id", ""))))
        roots = payload.get("roots")
        query = payload.get("query")
        bounds = payload.get("bounds")
        value = cls(
            roots=roots
            if isinstance(roots, PropagationAuthorityRoots)
            else PropagationAuthorityRoots.from_dict(roots),
            query=query
            if isinstance(query, MissingInputQuery)
            else MissingInputQuery.from_dict(query),
            requirement_id=payload.get("requirement_id", ""),
            bounds=bounds
            if isinstance(bounds, MissingInputRetrievalBounds)
            else MissingInputRetrievalBounds.from_dict(bounds),
            candidates=tuple(
                item
                if isinstance(item, MissingInputCandidateNomination)
                else MissingInputCandidateNomination.from_dict(item)
                for item in candidates
            ),
            candidate_set_id=payload.get("candidate_set_id", ""),
            signal_roots=tuple(signal_roots),
            vector_query_id=payload.get("vector_query_id", ""),
            graph_id=payload.get("graph_id", ""),
            semantic_authority=payload.get("semantic_authority", False),
        )
        _verify_record_identity(payload, value)
        return value


def _source_node(raw: Mapping[str, Any], query: MissingInputQuery) -> GraphNodeRef | None:
    value = raw.get("source_node", raw.get("node"))
    try:
        if isinstance(value, GraphNodeRef):
            return value
        if isinstance(value, Mapping):
            if "schema" in value:
                return GraphNodeRef.from_dict(value)
            return GraphNodeRef(
                node_id=str(value.get("node_id", value.get("id", "node:unknown"))),
                kind=str(value.get("kind", "symbol")),
                path=str(value.get("path", query.consumer_path or "unknown.py")),
                symbol_id=str(value.get("symbol_id", value.get("symbol", "symbol:unknown"))),
                artifact_id=str(value.get("artifact_id", value.get("blob_id", "blob:unknown"))),
                provenance=value.get("provenance", GraphProvenance.NOMINATED),
                extractor_id=str(value.get("extractor_id", "")),
            )
        if all(name in raw for name in ("node_id", "path", "symbol_id", "artifact_id")):
            return GraphNodeRef(
                node_id=str(raw["node_id"]),
                kind=str(raw.get("kind", "symbol")),
                path=str(raw["path"]),
                symbol_id=str(raw["symbol_id"]),
                artifact_id=str(raw["artifact_id"]),
                provenance=raw.get("provenance", GraphProvenance.NOMINATED),
                extractor_id=str(raw.get("extractor_id", "")),
            )
    except (KeyError, TypeError, ValueError, Exception):
        return None
    return None


def _expression_ref(raw: Mapping[str, Any], query: MissingInputQuery) -> str:
    for key in (
        "expression_ref",
        "expression",
        "name",
        "symbol",
        "symbol_id",
        "parameter_name",
        "field",
        "provider",
        "factory",
        "constructor",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    node = raw.get("source_node") or raw.get("node")
    if isinstance(node, GraphNodeRef) and node.symbol_id:
        return node.symbol_id
    if isinstance(node, Mapping) and node.get("symbol_id"):
        return str(node["symbol_id"])
    if isinstance(raw.get("row"), ChangeValueIndexRow):
        return _row_symbol(raw["row"])
    return f"expr:{query.parameter_name}:{_fingerprint(raw).split(':')[-1][:16]}"


def _type_ref(raw: Mapping[str, Any], query: MissingInputQuery) -> str:
    for key in ("type_ref", "type", "type_id", "return_type", "schema_type"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(raw.get("row"), ChangeValueIndexRow):
        return raw["row"].type_ref or query.type_ref
    return query.type_ref


def _infer_route(
    raw: Mapping[str, Any], signals: set[str]
) -> ConstructionRouteKind:
    supplied = raw.get("route", raw.get("construction_route", raw.get("route_kind")))
    if supplied:
        try:
            return _route(supplied)
        except MissingInputRetrievalError:
            pass
    if raw.get("new_behavior") is True or raw.get("new_type") is True or raw.get("synthesize") is True:
        return ConstructionRouteKind.NEW_BEHAVIOR
    if (
        raw.get("construct") is True
        or raw.get("factory") is True
        or raw.get("builder") is True
        or raw.get("constructor") is True
        or MissingInputSignal.FACTORY_BUILDER_CONSTRUCTOR.value in signals
        or MissingInputSignal.SCHEMA.value in signals
    ):
        return ConstructionRouteKind.CONSTRUCT
    if (
        raw.get("convert") is True
        or raw.get("conversion") is True
        or raw.get("adapter") is True
        or raw.get("conversion_ref")
    ):
        return ConstructionRouteKind.CONVERT
    if (
        raw.get("thread") is True
        or raw.get("thread_upward") is True
        or MissingInputSignal.REACHING_DEFINITION.value in signals
        and raw.get("available_locally") is False
    ):
        return ConstructionRouteKind.THREAD
    if MissingInputSignal.REACHING_DEFINITION.value in signals and raw.get("thread_hint") is True:
        return ConstructionRouteKind.THREAD
    return ConstructionRouteKind.REUSE


def _build_route(
    raw: Mapping[str, Any],
    expression_ref: str,
    signals: set[str],
    source_node: GraphNodeRef | None,
) -> ConstructionRouteCandidate:
    kind = _infer_route(raw, signals)
    return ConstructionRouteCandidate(
        route=kind,
        expression_ref=expression_ref,
        source_node_id="" if source_node is None else source_node.node_id,
        conversion_ref=str(raw.get("conversion_ref", "") or ""),
        factory_ref=str(raw.get("factory_ref", raw.get("factory", "")) or ""),
        dependency_refs=tuple(raw.get("dependency_refs", ()) or ()),
        notes=tuple(raw.get("route_notes", ()) or ()),
        semantic_authority=False,
    )


def _diagnostics(
    signal: str,
    raw: Mapping[str, Any],
    expected_roots: PropagationAuthorityRoots,
    vector_roots: tuple[str, str, str] | None,
) -> set[str]:
    reasons: set[str] = set()
    if raw.get("partial") is True or raw.get("complete") is False:
        reasons.add(REJECTION_PARTIAL)
    if _contains_body_or_secret(raw):
        reasons.add(REJECTION_BODY_OR_SECRET)
    if raw.get("forged") is True or raw.get("forged_history") is True or raw.get("history_reviewed") is False:
        reasons.add(REJECTION_FORGED)
    if raw.get("forbidden_config") is True or raw.get("env_allowed") is False or raw.get("policy_allowed") is False:
        reasons.add(REJECTION_FORBIDDEN_CONFIG_ENV)
    if raw.get("semantic_authority") is True:
        reasons.add(REJECTION_SEMANTIC_AUTHORITY_CLAIM)
    for key in _COMPATIBILITY_KEYS:
        if raw.get(key) is True:
            reasons.add(REJECTION_COMPATIBILITY_CLAIM)
            break
    for key in _WRITE_SCOPE_KEYS:
        value = raw.get(key)
        if value not in (None, (), [], ""):
            reasons.add(REJECTION_WRITE_SCOPE_CLAIM)
            break
    for key in _PLACEMENT_KEYS:
        value = raw.get(key)
        if value not in (None, (), [], "", False):
            reasons.add(REJECTION_PLACEMENT_CLAIM)
            break
    # Stale / cross-root bindings.
    for key in _ROOT_KEYS:
        if key in raw and raw[key] not in (None, "", getattr(expected_roots, key)):
            reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    candidate_roots = raw.get("roots")
    if isinstance(candidate_roots, PropagationAuthorityRoots):
        if candidate_roots != expected_roots:
            reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    elif isinstance(candidate_roots, Mapping):
        if any(
            key in candidate_roots
            and candidate_roots[key] not in (None, "", getattr(expected_roots, key))
            for key in _ROOT_KEYS
        ):
            reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    for alias_key, attr in (
        ("tree_id", "candidate_tree_id"),
        ("forest_id", "candidate_forest_id"),
    ):
        if alias_key in raw and raw[alias_key] not in (
            None,
            "",
            getattr(expected_roots, attr),
            getattr(expected_roots, "base_tree_id" if alias_key == "tree_id" else "base_forest_id"),
        ):
            reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    if signal == MissingInputSignal.VECTOR.value:
        try:
            score = raw.get("score", raw.get("score_millionths", 0))
            if not math.isfinite(float(score)):
                reasons.add(REJECTION_POISONED)
        except (TypeError, ValueError):
            reasons.add(REJECTION_POISONED)
        if raw.get("semantic_authority", False) is not False:
            reasons.add(REJECTION_POISONED)
        if raw.get("compatibility_claim", False) is not False:
            reasons.add(REJECTION_COMPATIBILITY_CLAIM)
        if vector_roots is not None:
            tree_id, config_id, model_id = vector_roots
            if raw.get("tree_id") not in (None, "", tree_id):
                reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
            if raw.get("config_id") not in (None, "", config_id):
                reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
            if raw.get("model_id") not in (None, "", model_id):
                reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
        binding = raw.get("binding")
        if isinstance(binding, Mapping) and vector_roots is not None:
            tree_id, config_id, model_id = vector_roots
            if binding.get("graph_root_id") not in (None, "", tree_id):
                reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
            if binding.get("configuration_id") not in (None, "", config_id):
                reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
            if binding.get("model_id") not in (None, "", model_id):
                reasons.add(REJECTION_STALE_OR_CROSS_ROOT)
    return reasons


def _row_symbol(row: ChangeValueIndexRow) -> str:
    return row.qualified_name or row.name or row.row_id


def _row_blob(row: ChangeValueIndexRow) -> str:
    sidecar = row.sidecar
    value = getattr(sidecar, "blob_identity", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"blob:{row.row_id}"


def _raw_from_item(item: Any) -> dict[str, Any]:
    if isinstance(item, ChangeValueHit):
        symbol = _row_symbol(item.row)
        return {
            "score": item.score,
            "rank": item.rank,
            "expression_ref": symbol,
            "type_ref": item.row.type_ref or "",
            "path": item.row.path,
            "index_id": item.index_id,
            "query_id": item.query_id,
            "source_node": {
                "node_id": f"node:{item.row.row_id}",
                "kind": str(getattr(item.row.kind, "value", item.row.kind)),
                "path": item.row.path,
                "symbol_id": symbol,
                "artifact_id": _row_blob(item.row),
                "provenance": GraphProvenance.NOMINATED.value,
            },
            "semantic_authority": item.semantic_authority,
            "compatibility_claim": item.compatibility_claim,
            "evidence_refs": (f"hit:{item.hit_id}",),
        }
    if isinstance(item, ChangeValueIndexRow):
        symbol = _row_symbol(item)
        return {
            "expression_ref": symbol,
            "type_ref": item.type_ref or "",
            "path": item.path,
            "source_node": {
                "node_id": f"node:{item.row_id}",
                "kind": str(getattr(item.kind, "value", item.kind)),
                "path": item.path,
                "symbol_id": symbol,
                "artifact_id": _row_blob(item),
                "provenance": GraphProvenance.NOMINATED.value,
            },
            "evidence_refs": (f"row:{item.row_id}",),
        }
    if isinstance(item, ValueProvenanceCandidate):
        return {
            "expression_ref": item.expression_ref,
            "type_ref": item.type_ref,
            "route": item.route.route.value,
            "source_node": item.source_node,
            "evidence_refs": item.signal_refs,
            "roots": item.roots,
            "information_content_ref": item.information_content_ref,
            "diagnostics": item.diagnostics,
        }
    if isinstance(item, ConstructionRouteCandidate):
        return {
            "expression_ref": item.expression_ref,
            "route": item.route.value,
            "factory_ref": item.factory_ref,
            "conversion_ref": item.conversion_ref,
            "dependency_refs": item.dependency_refs,
        }
    return _mapping(item)


class MissingInputCandidateRetriever:
    """Union bounded missing-input signal families into a diagnostic-only receipt."""

    def __init__(
        self,
        roots: PropagationAuthorityRoots,
        *,
        bounds: MissingInputRetrievalBounds | None = None,
    ) -> None:
        if not isinstance(roots, PropagationAuthorityRoots):
            raise MissingInputRetrievalBindingError("roots must be PropagationAuthorityRoots")
        self.roots = roots
        self.bounds = bounds or MissingInputRetrievalBounds()

    def retrieve(
        self,
        requirement: MissingInputRequirement,
        *,
        query: MissingInputQuery | None = None,
        consumer_path: str = "",
        consumer_node_id: str = "",
        consumer_context_refs: Sequence[str] = (),
        candidates_by_signal: Mapping[str, Any] | None = None,
        value_index: ChangeValueIndexSnapshot | None = None,
        vector_query: ChangeValueQuery | None = None,
        graph_id: str = "",
        **signal_candidates: Any,
    ) -> MissingInputCandidateReceipt:
        if not isinstance(requirement, MissingInputRequirement):
            raise MissingInputRetrievalBindingError(
                "requirement must be a typed MissingInputRequirement"
            )
        if requirement.roots != self.roots:
            raise MissingInputRetrievalBindingError(
                "requirement and retriever must share exact roots"
            )
        if query is None:
            query = MissingInputQuery.from_requirement(
                requirement,
                consumer_path=consumer_path
                or (consumer_node_id and f"node:{consumer_node_id}")
                or "consumer:unspecified",
                consumer_node_id=consumer_node_id,
                consumer_context_refs=consumer_context_refs
                or (("consumer:" + (consumer_path or consumer_node_id or "unspecified"),)),
            )
        if not isinstance(query, MissingInputQuery):
            raise MissingInputRetrievalBindingError("query must be MissingInputQuery")
        if query.roots != self.roots:
            raise MissingInputRetrievalBindingError("query roots do not match retriever roots")
        if query.requirement_id != requirement.requirement_id:
            raise MissingInputRetrievalBindingError(
                "query requirement_id does not match requirement"
            )

        vector_roots: tuple[str, str, str] | None = None
        # Each signal points at the immutable root that actually constrains it;
        # the enclosing PropagationAuthorityRoots still binds the complete
        # base/candidate forest/tree/overlay plus graph/index/model/config/
        # translator/toolchain/policy tuple for every replay.
        signal_roots: dict[str, str] = {
            MissingInputSignal.IN_SCOPE_SYMBOL.value: self.roots.graph_id,
            MissingInputSignal.RECEIVER_STATE.value: self.roots.graph_id,
            MissingInputSignal.CALLER_PARAMETER.value: self.roots.graph_id,
            MissingInputSignal.CONSTANT_DEFAULT.value: self.roots.graph_id,
            MissingInputSignal.REQUEST_SESSION_CONTEXT.value: self.roots.graph_id,
            MissingInputSignal.REACHING_DEFINITION.value: self.roots.graph_id,
            MissingInputSignal.CONFIG_ENV_PROVIDER.value: self.roots.policy_id,
            MissingInputSignal.DI_REGISTRY_PROVIDER.value: self.roots.graph_id,
            MissingInputSignal.FACTORY_BUILDER_CONSTRUCTOR.value: self.roots.graph_id,
            MissingInputSignal.SCHEMA.value: self.roots.index_id,
            MissingInputSignal.LINEAGE.value: self.roots.candidate_tree_id,
            MissingInputSignal.AUTHORITATIVE_SPEC_TEST.value: self.roots.policy_id,
            MissingInputSignal.LEXICAL_BM25.value: self.roots.index_id,
            MissingInputSignal.GRAPH.value: self.roots.graph_id,
            MissingInputSignal.VECTOR.value: self.roots.index_id,
        }
        query_id = ""
        bound_graph_id = graph_id or self.roots.graph_id
        if graph_id and graph_id != self.roots.graph_id:
            raise MissingInputRetrievalBindingError(
                "supplied graph_id does not match authority roots"
            )

        if value_index is not None:
            if not isinstance(value_index, ChangeValueIndexSnapshot):
                raise MissingInputRetrievalBindingError(
                    "value_index must be a canonical ChangeValueIndexSnapshot"
                )
            if value_index.tree_id not in (
                self.roots.candidate_tree_id,
                self.roots.base_tree_id,
            ):
                raise MissingInputRetrievalBindingError(
                    "value index tree does not bind receipt base/candidate tree roots"
                )
            if value_index.index_id != self.roots.index_id:
                raise MissingInputRetrievalBindingError(
                    "value index id does not match authority index_id"
                )
            if value_index.config.model_id != self.roots.model_id:
                raise MissingInputRetrievalBindingError(
                    "value index model does not match authority model_id"
                )
            if value_index.config.config_id != self.roots.config_id:
                raise MissingInputRetrievalBindingError(
                    "value index config does not match authority config_id"
                )
            vector_roots = (
                value_index.tree_id,
                value_index.config.config_id,
                value_index.config.model_id,
            )
            signal_roots[MissingInputSignal.VECTOR.value] = value_index.index_id

        if vector_query is not None:
            if not isinstance(vector_query, ChangeValueQuery):
                raise MissingInputRetrievalBindingError(
                    "vector_query must be a canonical ChangeValueQuery"
                )
            if vector_query.semantic_authority is not False:
                raise MissingInputRetrievalBindingError(
                    "vector query must be non-authoritative"
                )
            if vector_query.compatibility_claim is not False:
                raise MissingInputRetrievalBindingError(
                    "vector query cannot assert compatibility"
                )
            if value_index is None or (
                vector_query.tree_id,
                vector_query.index_id,
                vector_query.config_id,
            ) != (
                value_index.tree_id,
                value_index.index_id,
                value_index.config.config_id,
            ):
                raise MissingInputRetrievalBindingError(
                    "vector query does not bind the supplied value index"
                )
            if vector_query.missing_requirement_id not in (
                "",
                requirement.requirement_id,
                query.requirement_id,
            ):
                raise MissingInputRetrievalBindingError(
                    "vector query missing_requirement_id does not match requirement"
                )
            query_id = vector_query.query_id

        supplied = dict(candidates_by_signal or {})
        for name, value in signal_candidates.items():
            if value is not None:
                supplied[name] = value

        grouped: dict[str, list[Any]] = {}
        for raw_signal, value in supplied.items():
            signal = _signal(raw_signal)
            if isinstance(value, ChangeValueSearchResult):
                if (
                    signal != MissingInputSignal.VECTOR.value
                    or value.semantic_authority is not False
                    or value.complete is not True
                ):
                    raise MissingInputRetrievalBindingError(
                        "vector results must be complete, non-authoritative vector evidence"
                    )
                if value_index is not None and value.index_id != value_index.index_id:
                    raise MissingInputRetrievalBindingError(
                        "vector result index differs from value index"
                    )
                grouped.setdefault(signal, []).extend(value.hits)
                query_id = value.query.query_id
                continue
            if value is None:
                entries: tuple[Any, ...] = ()
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray, Mapping)
            ):
                entries = tuple(value)
            else:
                entries = (value,)
            if len(entries) > self.bounds.max_candidates_per_signal:
                raise MissingInputRetrievalBoundsError(
                    f"{signal} exceeds max_candidates_per_signal"
                )
            grouped.setdefault(signal, []).extend(entries)

        aggregate: dict[tuple[Any, ...], dict[str, Any]] = {}
        for signal in sorted(grouped):
            entries = grouped[signal]
            if len(entries) > self.bounds.max_candidates_per_signal:
                raise MissingInputRetrievalBoundsError(
                    f"{signal} exceeds max_candidates_per_signal"
                )
            for item in entries:
                raw = _raw_from_item(item)
                if not isinstance(raw, Mapping):
                    raw = {"value": raw}
                # Capture identity fields before redaction so expression refs
                # that merely contain substrings like "secret" are preserved.
                expression = _expression_ref(raw, query)
                type_ref = _type_ref(raw, query)
                source = _source_node(raw, query)
                had_body = _contains_body_or_secret(raw)
                safe_raw = _redact_payload(raw) if had_body else dict(raw)
                if not isinstance(safe_raw, Mapping):
                    safe_raw = {"value": safe_raw}
                # Re-bind compact identity after redaction of body/secret fields.
                safe_raw = dict(safe_raw)
                safe_raw["expression_ref"] = expression
                safe_raw["type_ref"] = type_ref
                if source is not None:
                    safe_raw["source_node"] = source
                key = (
                    expression,
                    type_ref,
                    "" if source is None else source.node_id,
                    "" if source is None else source.artifact_id,
                )
                if safe_raw.get("partial") is True or not expression:
                    key = key + (_fingerprint(safe_raw),)
                entry = aggregate.setdefault(
                    key,
                    {
                        "expression": expression,
                        "type_ref": type_ref,
                        "source": source,
                        "signals": set(),
                        "refs": {},
                        "reasons": set(),
                        "raw": [],
                        "routes": set(),
                    },
                )
                entry["signals"].add(signal)
                entry["refs"].setdefault(signal, []).extend(
                    _refs(safe_raw.get("evidence_refs", safe_raw.get("evidence_ref")), signal, safe_raw)
                )
                reasons = _diagnostics(signal, safe_raw, self.roots, vector_roots)
                if had_body:
                    reasons.add(REJECTION_BODY_OR_SECRET)
                entry["reasons"].update(reasons)
                entry["raw"].append(safe_raw)
                try:
                    entry["routes"].add(_infer_route(safe_raw, entry["signals"]).value)
                except MissingInputRetrievalError:
                    entry["reasons"].add(REJECTION_INVALID_PAYLOAD)

        if not aggregate:
            # Empty retrieval is a valid, explicit diagnostic rather than an
            # implicit winner.  The requirement id is only an audit anchor.
            raw = {
                "partial": True,
                "reason": "no_signal_candidates",
                "expression_ref": f"missing:{requirement.parameter_name}",
                "type_ref": requirement.type_ref,
            }
            aggregate[("empty", requirement.requirement_id)] = {
                "expression": raw["expression_ref"],
                "type_ref": requirement.type_ref,
                "source": None,
                "signals": set(),
                "refs": {},
                "reasons": {REJECTION_PARTIAL},
                "raw": [raw],
                "routes": set(),
            }
        if len(aggregate) > self.bounds.max_candidates:
            raise MissingInputRetrievalBoundsError(
                "unioned candidate set exceeds max_candidates; refusing partial union"
            )

        nominations: list[MissingInputCandidateNomination] = []
        for entry in aggregate.values():
            signals = set(entry["signals"])
            reasons = set(entry["reasons"])
            if len(entry["routes"]) > 1:
                reasons.add(REJECTION_CONFLICTING_ROUTES)
            raw = min(entry["raw"], key=_fingerprint)
            route = _build_route(
                raw, entry["expression"], signals, entry["source"]
            )
            # Preserve explicit multi-signal refs on the candidate itself.
            all_refs = tuple(
                sorted(
                    {ref for refs in entry["refs"].values() for ref in refs},
                    key=lambda ref: ref.content_id,
                )
            )
            candidate = ValueProvenanceCandidate(
                roots=self.roots,
                requirement_id=requirement.requirement_id,
                expression_ref=entry["expression"],
                type_ref=entry["type_ref"] or requirement.type_ref,
                route=route,
                source_node=entry["source"],
                information_content_ref=str(
                    raw.get("information_content_ref", requirement.information_content_ref)
                    or requirement.information_content_ref
                ),
                signal_refs=all_refs,
                diagnostics=tuple(sorted(reasons)),
                semantic_authority=False,
                compatibility_claim=False,
                placement_claim=False,
                write_paths=(),
            )
            nominations.append(
                MissingInputCandidateNomination(
                    candidate=candidate,
                    disposition=(
                        MissingInputCandidateDisposition.REJECTED
                        if reasons
                        else MissingInputCandidateDisposition.NOMINATED
                    ),
                    signal_evidence=tuple(
                        (
                            signal,
                            tuple(sorted(set(refs), key=lambda ref: ref.content_id)),
                        )
                        for signal, refs in entry["refs"].items()
                    ),
                    diagnostics=tuple(sorted(reasons)),
                    semantic_authority=False,
                )
            )
        nominations.sort(key=lambda item: item.content_id)
        candidates = tuple(nominations)
        return MissingInputCandidateReceipt(
            roots=self.roots,
            query=query,
            requirement_id=requirement.requirement_id,
            bounds=self.bounds,
            candidates=candidates,
            candidate_set_id=candidate_set_identity(
                tuple(item.candidate for item in candidates)
            ),
            signal_roots=tuple(signal_roots.items()),
            vector_query_id=query_id,
            graph_id=bound_graph_id,
            semantic_authority=False,
        )

    nominate = retrieve
    search = retrieve


def retrieve_missing_input_candidates(
    roots: PropagationAuthorityRoots,
    requirement: MissingInputRequirement,
    **kwargs: Any,
) -> MissingInputCandidateReceipt:
    """Stateless convenience entry point for the retrieval-only boundary."""
    bounds = kwargs.pop("bounds", None)
    return MissingInputCandidateRetriever(roots, bounds=bounds).retrieve(
        requirement, **kwargs
    )


__all__ = (
    "MISSING_INPUT_QUERY_SCHEMA",
    "VALUE_PROVENANCE_CANDIDATE_SCHEMA",
    "CONSTRUCTION_ROUTE_CANDIDATE_SCHEMA",
    "MISSING_INPUT_NOMINATION_SCHEMA",
    "MISSING_INPUT_RECEIPT_SCHEMA",
    "MISSING_INPUT_BOUNDS_SCHEMA",
    "MISSING_INPUT_SIGNAL_REF_SCHEMA",
    "PRODUCER_ID",
    "MAX_CANDIDATE_COUNT",
    "SIGNAL_FAMILIES",
    "MissingInputSignal",
    "ConstructionRouteKind",
    "MissingInputCandidateDisposition",
    "MissingInputRetrievalError",
    "MissingInputRetrievalBindingError",
    "MissingInputRetrievalBoundsError",
    "MissingInputSignalRef",
    "MissingInputRetrievalBounds",
    "MissingInputQuery",
    "ConstructionRouteCandidate",
    "ValueProvenanceCandidate",
    "MissingInputCandidateNomination",
    "MissingInputCandidateReceipt",
    "MissingInputCandidateRetriever",
    "retrieve_missing_input_candidates",
    "candidate_set_identity",
    "REJECTION_STALE_OR_CROSS_ROOT",
    "REJECTION_POISONED",
    "REJECTION_FORGED",
    "REJECTION_PARTIAL",
    "REJECTION_BODY_OR_SECRET",
    "REJECTION_COMPATIBILITY_CLAIM",
    "REJECTION_WRITE_SCOPE_CLAIM",
    "REJECTION_PLACEMENT_CLAIM",
    "REJECTION_SEMANTIC_AUTHORITY_CLAIM",
    "REJECTION_FORBIDDEN_CONFIG_ENV",
    "REJECTION_CONFLICTING_ROUTES",
    "REJECTION_INVALID_PAYLOAD",
)
