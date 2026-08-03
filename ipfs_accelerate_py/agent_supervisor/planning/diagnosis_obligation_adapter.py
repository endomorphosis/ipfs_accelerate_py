"""Translate checked Doctor findings into the shared obligation graph.

Interface: ``DiagnosisObligationBridge@1`` -> ``ObligationGraph@1``.

This module is deliberately an adapter, not a Doctor-specific planner.  It
lowers the checked records produced by :mod:`doctor_contract_adapters` and
:mod:`doctor_causal_localization` into the exact contracts implemented by
:mod:`obligation_graph_compiler`.

Only opaque identifiers and closed vocabularies participate in the formal
translation.  In particular, diagnostic ``message`` and ``details`` fields
are never copied into predicates, facts, producer rules, theorem references,
or candidate effects.  The resulting graph is proposal-only and carries no
proof, mutation, completion, or effect authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis import deterministic_doctor_contracts as det
from ..analysis import doctor_repository_diagnostics as diag
from ..analysis.doctor_causal_localization import (
    CausalLocalizationDisposition,
    DoctorCausalLocalizationReceipt,
)
from ..analysis.doctor_contract_adapters import (
    DIAGNOSIS_OBLIGATION_BRIDGE_INTERFACE,
    DIAGNOSIS_OBLIGATION_BRIDGE_SCHEMA,
    FINDING_BRIDGE_SCHEMA,
    ROOT_BRIDGE_SCHEMA,
    SNAPSHOT_BRIDGE_SCHEMA,
    DiagnosisObligationBridge,
    FindingBridge,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    canonical_json_bytes,
    content_identity,
)
from .obligation_graph_compiler import (
    OBLIGATION_GRAPH_SCHEMA,
    AssumptionBinding,
    AssumptionStatus,
    CompilationBounds,
    FactAuthority,
    FactTruth,
    InvalidationSelector,
    InvalidationSelectorKind,
    ObligationGraph,
    ObligationGraphCompiler,
    ObligationGraphDecision,
    ObservedFact,
    PredicatePolarity,
    ProducerRule,
    SemanticSupport,
    TaskCandidate,
    TypedIntent,
    TypedPredicate,
    obligation_id_for_predicate,
    obligation_id_for_producer,
)


DIAGNOSIS_OBLIGATION_ADAPTER_INTERFACE: Final[str] = (
    DIAGNOSIS_OBLIGATION_BRIDGE_INTERFACE
)
DIAGNOSIS_OBLIGATION_ADAPTER_VERSION: Final[int] = 1
DIAGNOSIS_OBLIGATION_COMPILATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/diagnosis-obligation-compilation@1"
)
CONTRACT_MISMATCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-mismatch-obligation-input@1"
)

MAX_FINDINGS: Final[int] = 4_096
MAX_REFERENCES: Final[int] = 16_384
MAX_RECORD_BYTES: Final[int] = 2_000_000

_SUPPORTED_FINDING_KINDS: Final[frozenset[str]] = frozenset(
    item.value
    for item in diag.FindingKind
    if item not in {diag.FindingKind.UNSUPPORTED, diag.FindingKind.COMPLETENESS}
)
_CONTRADICTION_MARKERS: Final[tuple[str, ...]] = (
    "contradict",
    "inconsisten",
    "conflict",
)
_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "source",
        "source_body",
        "source_text",
        "source_bytes",
        "content",
        "contents",
        "snippet",
        "code",
        "file_text",
        "raw_ast",
        "ast_body",
        "secret",
        "password",
        "token",
        "api_key",
        "private_key",
        "credential",
    }
)


class DiagnosisObligationAdapterError(ValueError):
    """A diagnosis cannot be translated without weakening its contracts."""


class DiagnosisObligationBoundsError(DiagnosisObligationAdapterError):
    """A bounded adapter input or output exceeded its declared limit."""


class DiagnosisObligationAuthorityError(DiagnosisObligationAdapterError):
    """Root, repository, snapshot, issue, or evidence authority mismatched."""


class DiagnosisObligationTamperError(DiagnosisObligationAdapterError):
    """A stored canonical identity did not recompute."""


class DiagnosisCompilationDisposition(str, Enum):
    """Fail-closed outcome of diagnosis lowering."""

    COMPILED = "compiled"
    REVIEW_REQUIRED = "review_required"
    ABSTAINED = "abstained"


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise DiagnosisObligationAdapterError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise DiagnosisObligationAdapterError(f"{name} must not be empty")
    if result and (
        any(char.isspace() for char in result)
        or "\x00" in result
        or len(result.encode("utf-8")) > 2_048
    ):
        raise DiagnosisObligationAdapterError(
            f"{name} must be an opaque compact identifier"
        )
    return result


def _ids(
    values: Sequence[Any] | None,
    name: str,
    *,
    required: bool = False,
    limit: int = MAX_REFERENCES,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise DiagnosisObligationAdapterError(f"{name} must be a sequence")
    else:
        raw = values
    if len(raw) > limit:
        raise DiagnosisObligationBoundsError(f"{name} exceeds its item bound")
    result = tuple(sorted({_identifier(item, name) for item in raw}))
    if required and not result:
        raise DiagnosisObligationAdapterError(f"{name} must not be empty")
    return result


def _mapping_ids(value: Mapping[str, Any] | None, name: str) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise DiagnosisObligationAdapterError(f"{name} must be an object")
    if len(value) > MAX_REFERENCES:
        raise DiagnosisObligationBoundsError(f"{name} exceeds its item bound")
    normalized = {
        _identifier(key, f"{name} key"): _identifier(item, f"{name} value")
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


def _assert_body_free(value: Any, name: str = "record") -> None:
    if isinstance(value, float):
        raise DiagnosisObligationAdapterError(f"{name} may not contain floats")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DiagnosisObligationAdapterError(
                    f"{name} has a non-string key"
                )
            normalized = key.casefold().replace("-", "_")
            if normalized in _FORBIDDEN_BODY_KEYS or any(
                normalized.endswith("_" + marker)
                for marker in _FORBIDDEN_BODY_KEYS
            ):
                raise DiagnosisObligationAdapterError(
                    f"{name} may not contain bodies or secrets"
                )
            _assert_body_free(item, name)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _assert_body_free(item, name)
    elif isinstance(value, (bytes, bytearray)):
        raise DiagnosisObligationAdapterError(f"{name} may not contain bytes")


def _stable_id(namespace: str, payload: Any) -> str:
    return f"{namespace}:{content_identity(payload)}"


def _verify_identity(payload: Mapping[str, Any], actual: str) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, "", actual):
        raise DiagnosisObligationTamperError(
            "diagnosis obligation compilation identity mismatch"
        )


def _enum(value: Any, cls: type[Enum], name: str) -> Any:
    if isinstance(value, cls):
        return value
    try:
        return cls(str(value))
    except (TypeError, ValueError) as exc:
        raise DiagnosisObligationAdapterError(
            f"{name} has an unsupported value"
        ) from exc


def _strategy_names(finding_kind: str) -> tuple[str, str]:
    """Return closed reviewed strategy classes, never model-authored effects."""

    if finding_kind in {
        "import",
        "name",
        "call_arity",
        "contract",
        "schema",
    }:
        return ("repair_contract_producer", "repair_contract_consumers")
    if finding_kind in {"syntax", "type"}:
        return ("repair_definition", "introduce_checked_adapter")
    if finding_kind in {
        "value",
        "dataflow",
        "effect",
        "resource",
        "state",
        "memory",
    }:
        return ("repair_semantics", "contain_compatibility_boundary")
    return ("repair_observed_contract", "repair_validation_boundary")


@dataclass(frozen=True)
class ContractMismatch:
    """Body-free semantic input shared by Planner and Doctor lowering.

    ``issue_ids``, evidence, and source references are provenance only.  The
    semantic identity is derived from the expected/observed contract IDs,
    typed kind, affected subjects, consumers, and frontiers.  This separation
    makes Planner and Doctor produce the same formal obligation identities for
    the same mismatch even though their evidence receipts differ.
    """

    repository_id: str
    current_root_id: str
    finding_kind: str
    expected_refs: tuple[str, ...]
    observed_refs: tuple[str, ...]
    subject_refs: tuple[str, ...] = ()
    consumer_refs: tuple[str, ...] = ()
    frontier_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    causal_slice_refs: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    issue_ids: tuple[str, ...] = ()
    diagnosis_complete: bool = True
    contradictory: bool = False
    approval_required: bool = False

    SCHEMA: ClassVar[str] = CONTRACT_MISMATCH_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self,
            "current_root_id",
            _identifier(self.current_root_id, "current_root_id"),
        )
        kind = _identifier(self.finding_kind, "finding_kind")
        object.__setattr__(self, "finding_kind", kind)
        for name in (
            "expected_refs",
            "observed_refs",
            "subject_refs",
            "consumer_refs",
            "frontier_refs",
            "evidence_refs",
            "causal_slice_refs",
            "source_refs",
            "issue_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        for name in (
            "diagnosis_complete",
            "contradictory",
            "approval_required",
        ):
            if not isinstance(getattr(self, name), bool):
                raise DiagnosisObligationAdapterError(
                    f"{name} must be a boolean"
                )

    @property
    def semantic_id(self) -> str:
        return _stable_id(
            "contract-mismatch",
            {
                "repository_id": self.repository_id,
                "current_root_id": self.current_root_id,
                "finding_kind": self.finding_kind,
                "expected_refs": list(self.expected_refs),
                "observed_refs": list(self.observed_refs),
                "subject_refs": list(self.subject_refs),
                "consumer_refs": list(self.consumer_refs),
                "frontier_refs": list(self.frontier_refs),
            },
        )

    @property
    def supported(self) -> bool:
        return (
            self.finding_kind in _SUPPORTED_FINDING_KINDS
            and bool(self.expected_refs)
            and bool(self.observed_refs)
            and self.diagnosis_complete
            and not self.contradictory
            and not self.approval_required
            and not self.frontier_refs
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "repository_id": self.repository_id,
            "current_root_id": self.current_root_id,
            "finding_kind": self.finding_kind,
            "expected_refs": list(self.expected_refs),
            "observed_refs": list(self.observed_refs),
            "subject_refs": list(self.subject_refs),
            "consumer_refs": list(self.consumer_refs),
            "frontier_refs": list(self.frontier_refs),
            "evidence_refs": list(self.evidence_refs),
            "causal_slice_refs": list(self.causal_slice_refs),
            "source_refs": list(self.source_refs),
            "issue_ids": list(self.issue_ids),
            "diagnosis_complete": self.diagnosis_complete,
            "contradictory": self.contradictory,
            "approval_required": self.approval_required,
            "semantic_id": self.semantic_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractMismatch":
        if not isinstance(payload, Mapping) or payload.get("schema") not in (
            None,
            cls.SCHEMA,
        ):
            raise DiagnosisObligationAdapterError(
                "contract mismatch has an unsupported schema"
            )
        allowed = {
            "schema",
            "repository_id",
            "current_root_id",
            "finding_kind",
            "expected_refs",
            "observed_refs",
            "subject_refs",
            "consumer_refs",
            "frontier_refs",
            "evidence_refs",
            "causal_slice_refs",
            "source_refs",
            "issue_ids",
            "diagnosis_complete",
            "contradictory",
            "approval_required",
            "semantic_id",
        }
        if set(payload).difference(allowed):
            raise DiagnosisObligationAdapterError(
                "contract mismatch contains unsupported fields"
            )
        _assert_body_free(payload, "contract mismatch")
        value = cls(
            repository_id=payload.get("repository_id", ""),
            current_root_id=payload.get("current_root_id", ""),
            finding_kind=payload.get("finding_kind", ""),
            expected_refs=tuple(payload.get("expected_refs") or ()),
            observed_refs=tuple(payload.get("observed_refs") or ()),
            subject_refs=tuple(payload.get("subject_refs") or ()),
            consumer_refs=tuple(payload.get("consumer_refs") or ()),
            frontier_refs=tuple(payload.get("frontier_refs") or ()),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            causal_slice_refs=tuple(payload.get("causal_slice_refs") or ()),
            source_refs=tuple(payload.get("source_refs") or ()),
            issue_ids=tuple(payload.get("issue_ids") or ()),
            diagnosis_complete=payload.get("diagnosis_complete", True),
            contradictory=payload.get("contradictory", False),
            approval_required=payload.get("approval_required", False),
        )
        claimed = payload.get("semantic_id", "")
        if claimed not in (None, "", value.semantic_id):
            raise DiagnosisObligationTamperError(
                "contract mismatch semantic identity mismatch"
            )
        return value


@dataclass(frozen=True)
class DiagnosisObligationRequest:
    """Typed request accepted by :class:`DiagnosisObligationAdapter`."""

    bridge: DiagnosisObligationBridge | FindingBridge | Mapping[str, Any]
    localizations: tuple[
        DoctorCausalLocalizationReceipt | Mapping[str, Any], ...
    ] = ()
    proof_requirement_refs: tuple[str, ...] = ()
    security_requirement_refs: tuple[str, ...] = ()
    validation_requirement_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "proof_requirement_refs",
            "security_requirement_refs",
            "validation_requirement_refs",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))


def _formal_projection(graph: ObligationGraph) -> dict[str, Any]:
    """Return the provenance-free formal content used for duality checks."""

    return {
        "current_root_id": graph.current_root_id,
        "predicates": sorted(
            (
                item.predicate_type,
                item.subject_ref,
                item.object_ref,
                item.polarity.value,
                item.support.value,
                tuple(item.assumption_refs),
                tuple(item.proof_requirement_refs),
                tuple(item.validation_requirement_refs),
            )
            for item in graph.predicates
        ),
        "facts": sorted(
            (
                item.predicate.predicate_type,
                item.predicate.subject_ref,
                item.predicate.object_ref,
                item.predicate.polarity.value,
                item.truth.value,
                item.authority.value,
            )
            for item in graph.facts
        ),
        "assumptions": sorted(
            (
                item.statement_ref,
                item.status.value,
                tuple(
                    sorted(
                        (selector.kind.value, selector.value_ref)
                        for selector in item.invalidation_selectors
                    )
                ),
            )
            for item in graph.assumptions
        ),
        "producers": sorted(
            (
                tuple(item.effect_predicate_ids),
                tuple(item.required_predicate_ids),
                tuple(item.assumption_refs),
                tuple(item.proof_requirement_refs),
                tuple(item.validation_requirement_refs),
                item.executable,
            )
            for item in graph.producers
        ),
    }


def formal_obligation_signature_for_graph(graph: ObligationGraph) -> str:
    """Identify formal obligations while excluding Doctor/Planner provenance."""

    if not isinstance(graph, ObligationGraph):
        raise DiagnosisObligationAdapterError("graph must be ObligationGraph")
    return _stable_id("formal-obligation-set", _formal_projection(graph))


@dataclass(frozen=True)
class DiagnosisObligationCompilation(CanonicalContract):
    """Body-free round-trippable receipt for one diagnosis translation."""

    SCHEMA: ClassVar[str] = DIAGNOSIS_OBLIGATION_COMPILATION_SCHEMA

    repository_id: str
    bridge_ids: tuple[str, ...]
    issue_ids: tuple[str, ...]
    semantic_issue_ids: tuple[str, ...]
    snapshot_ids: tuple[str, ...]
    authority_root_ids: Mapping[str, str]
    schema_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    causal_slice_ids: tuple[str, ...]
    frontier_ids: tuple[str, ...]
    graph: ObligationGraph
    disposition: DiagnosisCompilationDisposition | str
    desired_predicate_ids: tuple[str, ...]
    observed_fact_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    prohibition_obligation_ids: tuple[str, ...]
    impact_obligation_ids: tuple[str, ...]
    proof_obligation_ids: tuple[str, ...]
    security_obligation_ids: tuple[str, ...]
    validation_obligation_ids: tuple[str, ...]
    alternative_repair_subgoal_ids: tuple[str, ...]
    review_obligation_ids: tuple[str, ...] = ()
    abstention_obligation_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    formal_obligation_signature: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _identifier(self.repository_id, "repository_id")
        )
        for name in (
            "bridge_ids",
            "issue_ids",
            "semantic_issue_ids",
            "snapshot_ids",
            "schema_ids",
            "evidence_ids",
            "causal_slice_ids",
            "frontier_ids",
            "desired_predicate_ids",
            "observed_fact_ids",
            "assumption_ids",
            "prohibition_obligation_ids",
            "impact_obligation_ids",
            "proof_obligation_ids",
            "security_obligation_ids",
            "validation_obligation_ids",
            "alternative_repair_subgoal_ids",
            "review_obligation_ids",
            "abstention_obligation_ids",
            "reason_codes",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "authority_root_ids",
            _mapping_ids(self.authority_root_ids, "authority_root_ids"),
        )
        if not isinstance(self.graph, ObligationGraph):
            if isinstance(self.graph, Mapping):
                object.__setattr__(
                    self, "graph", ObligationGraph.from_dict(self.graph)
                )
            else:
                raise DiagnosisObligationAdapterError(
                    "graph must be ObligationGraph or a mapping"
                )
        object.__setattr__(
            self,
            "disposition",
            _enum(
                self.disposition,
                DiagnosisCompilationDisposition,
                "disposition",
            ),
        )
        signature = self.formal_obligation_signature or (
            formal_obligation_signature_for_graph(self.graph)
        )
        signature = _identifier(signature, "formal_obligation_signature")
        if signature != formal_obligation_signature_for_graph(self.graph):
            raise DiagnosisObligationTamperError(
                "formal obligation signature does not match graph"
            )
        object.__setattr__(self, "formal_obligation_signature", signature)
        self._validate_indexes()
        if len(canonical_json_bytes(self.to_dict())) > MAX_RECORD_BYTES:
            raise DiagnosisObligationBoundsError(
                "diagnosis obligation compilation exceeds byte bound"
            )

    @property
    def graph_id(self) -> str:
        return self.graph.graph_id

    @property
    def review_required(self) -> bool:
        return self.disposition in {
            DiagnosisCompilationDisposition.REVIEW_REQUIRED,
            DiagnosisCompilationDisposition.ABSTAINED,
        }

    @property
    def abstained(self) -> bool:
        return self.disposition is DiagnosisCompilationDisposition.ABSTAINED

    def _validate_indexes(self) -> None:
        predicate_ids = {item.predicate_id for item in self.graph.predicates}
        fact_ids = {item.fact_id for item in self.graph.facts}
        assumption_ids = {item.assumption_id for item in self.graph.assumptions}
        obligation_ids = {item.obligation_id for item in self.graph.nodes}
        if not set(self.desired_predicate_ids) <= predicate_ids:
            raise DiagnosisObligationAdapterError(
                "desired_predicate_ids references an unknown predicate"
            )
        if not set(self.observed_fact_ids) <= fact_ids:
            raise DiagnosisObligationAdapterError(
                "observed_fact_ids references an unknown fact"
            )
        if not set(self.assumption_ids) <= assumption_ids:
            raise DiagnosisObligationAdapterError(
                "assumption_ids references an unknown assumption"
            )
        for name in (
            "prohibition_obligation_ids",
            "impact_obligation_ids",
            "proof_obligation_ids",
            "security_obligation_ids",
            "validation_obligation_ids",
            "alternative_repair_subgoal_ids",
            "review_obligation_ids",
            "abstention_obligation_ids",
        ):
            if not set(getattr(self, name)) <= obligation_ids:
                raise DiagnosisObligationAdapterError(
                    f"{name} references an unknown obligation"
                )
        if self.disposition is DiagnosisCompilationDisposition.COMPILED:
            if self.graph.decision is not ObligationGraphDecision.READY:
                raise DiagnosisObligationAdapterError(
                    "compiled diagnosis requires a ready obligation graph"
                )
            if self.review_obligation_ids or self.abstention_obligation_ids:
                raise DiagnosisObligationAdapterError(
                    "compiled diagnosis cannot carry review/abstention obligations"
                )
        else:
            if not self.review_obligation_ids:
                raise DiagnosisObligationAdapterError(
                    "incomplete diagnosis must carry a review obligation"
                )
            if self.graph.decision is ObligationGraphDecision.READY:
                raise DiagnosisObligationAdapterError(
                    "review diagnosis cannot claim a ready graph"
                )
        if self.disposition is DiagnosisCompilationDisposition.ABSTAINED:
            if not self.abstention_obligation_ids:
                raise DiagnosisObligationAdapterError(
                    "abstention requires an abstention obligation"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DIAGNOSIS_OBLIGATION_ADAPTER_VERSION,
            "interface": DIAGNOSIS_OBLIGATION_ADAPTER_INTERFACE,
            "repository_id": self.repository_id,
            "bridge_ids": list(self.bridge_ids),
            "issue_ids": list(self.issue_ids),
            "semantic_issue_ids": list(self.semantic_issue_ids),
            "snapshot_ids": list(self.snapshot_ids),
            "authority_root_ids": dict(self.authority_root_ids),
            "schema_ids": list(self.schema_ids),
            "evidence_ids": list(self.evidence_ids),
            "causal_slice_ids": list(self.causal_slice_ids),
            "frontier_ids": list(self.frontier_ids),
            "graph": self.graph.to_dict(),
            "graph_id": self.graph.graph_id,
            "disposition": self.disposition.value,
            "desired_predicate_ids": list(self.desired_predicate_ids),
            "observed_fact_ids": list(self.observed_fact_ids),
            "assumption_ids": list(self.assumption_ids),
            "prohibition_obligation_ids": list(
                self.prohibition_obligation_ids
            ),
            "impact_obligation_ids": list(self.impact_obligation_ids),
            "proof_obligation_ids": list(self.proof_obligation_ids),
            "security_obligation_ids": list(self.security_obligation_ids),
            "validation_obligation_ids": list(self.validation_obligation_ids),
            "alternative_repair_subgoal_ids": list(
                self.alternative_repair_subgoal_ids
            ),
            "review_obligation_ids": list(self.review_obligation_ids),
            "abstention_obligation_ids": list(
                self.abstention_obligation_ids
            ),
            "reason_codes": list(self.reason_codes),
            "formal_obligation_signature": self.formal_obligation_signature,
            "authority": {
                "proof_authority": False,
                "security_attestation_authority": False,
                "effect_authority": False,
                "mutation_authority": False,
                "completion_authority": False,
                "candidate_generation_only": True,
            },
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DiagnosisObligationCompilation":
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.SCHEMA:
            raise DiagnosisObligationAdapterError(
                "diagnosis obligation compilation has an unsupported schema"
            )
        _assert_body_free(payload, "diagnosis obligation compilation")
        allowed = {
            "schema",
            "contract_version",
            "content_id",
            "cid",
            "interface",
            "repository_id",
            "bridge_ids",
            "issue_ids",
            "semantic_issue_ids",
            "snapshot_ids",
            "authority_root_ids",
            "schema_ids",
            "evidence_ids",
            "causal_slice_ids",
            "frontier_ids",
            "graph",
            "graph_id",
            "disposition",
            "desired_predicate_ids",
            "observed_fact_ids",
            "assumption_ids",
            "prohibition_obligation_ids",
            "impact_obligation_ids",
            "proof_obligation_ids",
            "security_obligation_ids",
            "validation_obligation_ids",
            "alternative_repair_subgoal_ids",
            "review_obligation_ids",
            "abstention_obligation_ids",
            "reason_codes",
            "formal_obligation_signature",
            "authority",
        }
        if set(payload).difference(allowed):
            raise DiagnosisObligationAdapterError(
                "diagnosis obligation compilation contains unsupported fields"
            )
        if payload.get("contract_version") not in (
            None,
            DIAGNOSIS_OBLIGATION_ADAPTER_VERSION,
        ):
            raise DiagnosisObligationAdapterError(
                "unsupported diagnosis obligation contract version"
            )
        if payload.get("interface") not in (
            None,
            DIAGNOSIS_OBLIGATION_ADAPTER_INTERFACE,
        ):
            raise DiagnosisObligationAdapterError(
                "unsupported diagnosis obligation interface"
            )
        authority = payload.get("authority")
        if authority is not None and authority != {
            "proof_authority": False,
            "security_attestation_authority": False,
            "effect_authority": False,
            "mutation_authority": False,
            "completion_authority": False,
            "candidate_generation_only": True,
        }:
            raise DiagnosisObligationAuthorityError(
                "diagnosis compilation cannot claim authority"
            )
        value = cls(
            repository_id=payload.get("repository_id", ""),
            bridge_ids=tuple(payload.get("bridge_ids") or ()),
            issue_ids=tuple(payload.get("issue_ids") or ()),
            semantic_issue_ids=tuple(payload.get("semantic_issue_ids") or ()),
            snapshot_ids=tuple(payload.get("snapshot_ids") or ()),
            authority_root_ids=payload.get("authority_root_ids") or {},
            schema_ids=tuple(payload.get("schema_ids") or ()),
            evidence_ids=tuple(payload.get("evidence_ids") or ()),
            causal_slice_ids=tuple(payload.get("causal_slice_ids") or ()),
            frontier_ids=tuple(payload.get("frontier_ids") or ()),
            graph=payload.get("graph") or {},
            disposition=payload.get("disposition", ""),
            desired_predicate_ids=tuple(
                payload.get("desired_predicate_ids") or ()
            ),
            observed_fact_ids=tuple(payload.get("observed_fact_ids") or ()),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            prohibition_obligation_ids=tuple(
                payload.get("prohibition_obligation_ids") or ()
            ),
            impact_obligation_ids=tuple(
                payload.get("impact_obligation_ids") or ()
            ),
            proof_obligation_ids=tuple(
                payload.get("proof_obligation_ids") or ()
            ),
            security_obligation_ids=tuple(
                payload.get("security_obligation_ids") or ()
            ),
            validation_obligation_ids=tuple(
                payload.get("validation_obligation_ids") or ()
            ),
            alternative_repair_subgoal_ids=tuple(
                payload.get("alternative_repair_subgoal_ids") or ()
            ),
            review_obligation_ids=tuple(
                payload.get("review_obligation_ids") or ()
            ),
            abstention_obligation_ids=tuple(
                payload.get("abstention_obligation_ids") or ()
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            formal_obligation_signature=payload.get(
                "formal_obligation_signature", ""
            ),
        )
        claimed_graph = payload.get("graph_id", "")
        if claimed_graph not in (None, "", value.graph.graph_id):
            raise DiagnosisObligationTamperError(
                "diagnosis obligation graph identity mismatch"
            )
        _verify_identity(payload, value.content_id)
        return value


# Compatibility name used by callers which refer to the output as a receipt.
DiagnosisObligationReceipt = DiagnosisObligationCompilation


@dataclass
class _GraphParts:
    predicates: dict[str, TypedPredicate] = field(default_factory=dict)
    facts: dict[str, ObservedFact] = field(default_factory=dict)
    assumptions: dict[str, AssumptionBinding] = field(default_factory=dict)
    producers: dict[str, ProducerRule] = field(default_factory=dict)
    candidates: dict[str, TaskCandidate] = field(default_factory=dict)
    desired_ids: set[str] = field(default_factory=set)
    observed_fact_ids: set[str] = field(default_factory=set)
    prohibition_predicate_ids: set[str] = field(default_factory=set)
    impact_predicate_ids: set[str] = field(default_factory=set)
    proof_predicate_ids: set[str] = field(default_factory=set)
    security_predicate_ids: set[str] = field(default_factory=set)
    validation_predicate_ids: set[str] = field(default_factory=set)
    alternative_pairs: list[tuple[str, str]] = field(default_factory=list)
    review_predicate_ids: set[str] = field(default_factory=set)
    abstention_predicate_ids: set[str] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)


def _merge_predicate(parts: _GraphParts, predicate: TypedPredicate) -> None:
    existing = parts.predicates.get(predicate.predicate_id)
    if existing is None:
        parts.predicates[predicate.predicate_id] = predicate
        return
    if (
        existing.semantic_key != predicate.semantic_key
        or existing.polarity is not predicate.polarity
        or existing.support is not predicate.support
        or existing.property_id != predicate.property_id
    ):
        raise DiagnosisObligationAdapterError(
            "stable predicate identity has conflicting definitions"
        )
    parts.predicates[predicate.predicate_id] = replace(
        existing,
        provenance_refs=tuple(
            sorted(set(existing.provenance_refs) | set(predicate.provenance_refs))
        ),
        assumption_refs=tuple(
            sorted(set(existing.assumption_refs) | set(predicate.assumption_refs))
        ),
        invalidation_selectors=tuple(
            {
                item.selector_id: item
                for item in (
                    *existing.invalidation_selectors,
                    *predicate.invalidation_selectors,
                )
            }.values()
        ),
        proof_requirement_refs=tuple(
            sorted(
                set(existing.proof_requirement_refs)
                | set(predicate.proof_requirement_refs)
            )
        ),
        validation_requirement_refs=tuple(
            sorted(
                set(existing.validation_requirement_refs)
                | set(predicate.validation_requirement_refs)
            )
        ),
    )


def _root_selector(spec: ContractMismatch) -> InvalidationSelector:
    return InvalidationSelector(
        selector_id=_stable_id(
            "invalidation-selector",
            {"kind": "root", "value": spec.current_root_id},
        ),
        kind=InvalidationSelectorKind.ROOT,
        value_ref=spec.current_root_id,
        provenance_refs=spec.source_refs or (spec.current_root_id,),
    )


def _evidence_selectors(spec: ContractMismatch) -> tuple[InvalidationSelector, ...]:
    return tuple(
        InvalidationSelector(
            selector_id=_stable_id(
                "invalidation-selector", {"kind": "evidence", "value": ref}
            ),
            kind=InvalidationSelectorKind.EVIDENCE,
            value_ref=ref,
            provenance_refs=(ref,),
        )
        for ref in spec.evidence_refs
    )


def _add_spec(
    parts: _GraphParts,
    spec: ContractMismatch,
    *,
    proof_requirement_refs: tuple[str, ...],
    security_requirement_refs: tuple[str, ...],
    validation_requirement_refs: tuple[str, ...],
) -> None:
    semantic_id = spec.semantic_id
    source_refs = spec.source_refs or spec.issue_ids or (semantic_id,)
    selectors = (_root_selector(spec), *_evidence_selectors(spec))
    supported = spec.supported
    support = (
        SemanticSupport.REVIEWED if supported else SemanticSupport.UNKNOWN
    )
    assumption_status = (
        AssumptionStatus.ACTIVE if supported else AssumptionStatus.UNKNOWN
    )

    root_assumption_id = _stable_id(
        "assumption",
        {"kind": "current_root", "root": spec.current_root_id},
    )
    parts.assumptions.setdefault(
        root_assumption_id,
        AssumptionBinding(
            assumption_id=root_assumption_id,
            statement_ref=spec.current_root_id,
            provenance_refs=(spec.current_root_id,),
            invalidation_selectors=(_root_selector(spec),),
            status=assumption_status,
        ),
    )
    assumption_ids = [root_assumption_id]
    for expected_ref in spec.expected_refs:
        assumption_id = _stable_id(
            "assumption",
            {
                "kind": "expected_contract_authoritative",
                "expected_ref": expected_ref,
                "root": spec.current_root_id,
            },
        )
        assumption_ids.append(assumption_id)
        parts.assumptions[assumption_id] = AssumptionBinding(
            assumption_id=assumption_id,
            statement_ref=expected_ref,
            provenance_refs=(expected_ref,),
            invalidation_selectors=selectors,
            status=assumption_status,
        )
    assumption_refs = tuple(sorted(assumption_ids))

    generated_proof_ref = _stable_id(
        "proof-requirement",
        {"mismatch": semantic_id, "property": "repair_equivalence"},
    )
    generated_security_ref = _stable_id(
        "security-requirement",
        {"mismatch": semantic_id, "property": "security_non_regression"},
    )
    generated_validation_ref = _stable_id(
        "validation-requirement",
        {"mismatch": semantic_id, "property": "contract_and_impact"},
    )
    proof_refs = tuple(
        sorted({generated_proof_ref, *proof_requirement_refs})
    )
    security_refs = tuple(
        sorted({generated_security_ref, *security_requirement_refs})
    )
    validation_refs = tuple(
        sorted(
            {
                generated_validation_ref,
                *validation_requirement_refs,
                *security_refs,
            }
        )
    )

    desired_behavior_ids: list[str] = []
    for expected_ref in spec.expected_refs or (
        _stable_id("missing-expected-contract", {"mismatch": semantic_id}),
    ):
        predicate = TypedPredicate(
            predicate_id=_stable_id(
                "predicate",
                {
                    "mismatch": semantic_id,
                    "type": "desired_contract_state",
                    "expected_ref": expected_ref,
                },
            ),
            predicate_type="desired_contract_state",
            subject_ref=expected_ref,
            object_ref=semantic_id,
            support=support,
            provenance_refs=tuple(sorted({expected_ref, *source_refs})),
            assumption_refs=assumption_refs,
            invalidation_selectors=selectors,
            proof_requirement_refs=proof_refs,
            validation_requirement_refs=validation_refs,
        )
        _merge_predicate(parts, predicate)
        parts.desired_ids.add(predicate.predicate_id)
        desired_behavior_ids.append(predicate.predicate_id)

    for observed_ref in spec.observed_refs or (
        _stable_id("missing-observed-contract", {"mismatch": semantic_id}),
    ):
        observed_predicate = TypedPredicate(
            predicate_id=_stable_id(
                "predicate",
                {
                    "mismatch": semantic_id,
                    "type": "observed_contract_state",
                    "observed_ref": observed_ref,
                },
            ),
            predicate_type="observed_contract_state",
            subject_ref=observed_ref,
            object_ref=semantic_id,
            support=SemanticSupport.REVIEWED,
            provenance_refs=tuple(sorted({observed_ref, *source_refs})),
            invalidation_selectors=selectors,
        )
        _merge_predicate(parts, observed_predicate)
        fact_id = _stable_id(
            "observed-fact",
            {"mismatch": semantic_id, "observed_ref": observed_ref},
        )
        parts.facts[fact_id] = ObservedFact(
            fact_id=fact_id,
            predicate=observed_predicate,
            truth=FactTruth.TRUE if spec.observed_refs else FactTruth.UNKNOWN,
            # The checked bridge establishes this opaque observation as a
            # bounded current-root fact.  It can satisfy only the distinct
            # ``observed_contract_state`` atom; it is never an expected
            # contract theorem or a repair effect.
            authority=FactAuthority.BOUNDED_OBSERVATION,
            provenance_refs=tuple(
                sorted({observed_ref, *spec.evidence_refs, *source_refs})
            ),
            current_root_id=spec.current_root_id,
            invalidation_selectors=selectors,
        )
        parts.observed_fact_ids.add(fact_id)

    def add_obligation_predicate(
        category: str,
        subject_ref: str,
        *,
        polarity: PredicatePolarity = PredicatePolarity.POSITIVE,
        proofs: tuple[str, ...] = proof_refs,
        validations: tuple[str, ...] = validation_refs,
    ) -> str:
        predicate = TypedPredicate(
            predicate_id=_stable_id(
                "predicate",
                {
                    "mismatch": semantic_id,
                    "type": category,
                    "subject_ref": subject_ref,
                    "polarity": polarity.value,
                },
            ),
            predicate_type=category,
            subject_ref=subject_ref,
            object_ref=semantic_id,
            polarity=polarity,
            support=support,
            provenance_refs=source_refs,
            assumption_refs=assumption_refs,
            invalidation_selectors=selectors,
            proof_requirement_refs=proofs,
            validation_requirement_refs=validations,
        )
        _merge_predicate(parts, predicate)
        parts.desired_ids.add(predicate.predicate_id)
        return predicate.predicate_id

    for prohibited in (
        f"unreviewed_effect:{semantic_id}",
        f"authority_escalation:{semantic_id}",
    ):
        predicate_id = add_obligation_predicate(
            "repair_prohibition",
            prohibited,
            polarity=PredicatePolarity.NEGATIVE,
        )
        parts.prohibition_predicate_ids.add(predicate_id)

    consumers = spec.consumer_refs or (f"impact_scope:{semantic_id}",)
    for consumer_ref in consumers:
        predicate_id = add_obligation_predicate(
            "impact_compatibility", consumer_ref
        )
        parts.impact_predicate_ids.add(predicate_id)

    proof_predicate_id = add_obligation_predicate(
        "proof_requirement", generated_proof_ref, proofs=proof_refs
    )
    parts.proof_predicate_ids.add(proof_predicate_id)
    security_predicate_id = add_obligation_predicate(
        "security_requirement",
        generated_security_ref,
        proofs=proof_refs,
        validations=validation_refs,
    )
    parts.security_predicate_ids.add(security_predicate_id)
    validation_predicate_id = add_obligation_predicate(
        "validation_requirement",
        generated_validation_ref,
        validations=validation_refs,
    )
    parts.validation_predicate_ids.add(validation_predicate_id)

    strategies = _strategy_names(spec.finding_kind)
    if supported:
        effect_ids = tuple(
            sorted(
                {
                    *desired_behavior_ids,
                    *parts.prohibition_predicate_ids,
                    *parts.impact_predicate_ids,
                    proof_predicate_id,
                    security_predicate_id,
                    validation_predicate_id,
                }
            )
        )
        # Scope the shared category sets to this mismatch before creating
        # effects; predicates from earlier findings are not effects of this
        # finding's repair.
        effect_ids = tuple(
            item
            for item in effect_ids
            if parts.predicates[item].object_ref == semantic_id
        )
        for strategy in strategies:
            producer_id = _stable_id(
                "repair-producer",
                {"mismatch": semantic_id, "strategy": strategy},
            )
            candidate_id = _stable_id(
                "repair-candidate",
                {"mismatch": semantic_id, "strategy": strategy},
            )
            producer = ProducerRule(
                producer_id=producer_id,
                effect_predicate_ids=effect_ids,
                provenance_refs=(
                    f"repair-strategy:{strategy}",
                    DIAGNOSIS_OBLIGATION_COMPILATION_SCHEMA,
                ),
                assumption_refs=assumption_refs,
                invalidation_selectors=selectors,
                proof_requirement_refs=proof_refs,
                validation_requirement_refs=validation_refs,
                task_candidate_ids=(candidate_id,),
                executable=True,
            )
            parts.producers[producer_id] = producer
            closes = tuple(
                obligation_id_for_producer(producer_id, effect_id)
                for effect_id in effect_ids
            )
            parts.candidates[candidate_id] = TaskCandidate(
                candidate_id=candidate_id,
                closes_obligation_ids=closes,
                producer_id=producer_id,
                provenance_refs=(
                    f"repair-strategy:{strategy}",
                    semantic_id,
                ),
            )
            primary_effect = desired_behavior_ids[0]
            parts.alternative_pairs.append((producer_id, primary_effect))
    else:
        review_id = add_obligation_predicate(
            "diagnosis_review", f"review:{semantic_id}"
        )
        abstention_id = add_obligation_predicate(
            "diagnosis_abstention", f"abstain:{semantic_id}"
        )
        parts.review_predicate_ids.add(review_id)
        parts.abstention_predicate_ids.add(abstention_id)
        for strategy in strategies:
            alternative_id = add_obligation_predicate(
                "repair_alternative_review",
                f"repair-strategy:{strategy}:{semantic_id}",
            )
            parts.alternative_pairs.append(("", alternative_id))
        if spec.contradictory:
            parts.reasons.add("contradictory_diagnosis")
        if not spec.diagnosis_complete:
            parts.reasons.add("diagnosis_incomplete")
        if spec.approval_required:
            parts.reasons.add("approval_required")
        if spec.frontier_refs:
            parts.reasons.add("open_frontier")
        if spec.finding_kind not in _SUPPORTED_FINDING_KINDS:
            parts.reasons.add("unsupported_finding_kind")
        if not spec.expected_refs:
            parts.reasons.add("missing_expected_contract")
        if not spec.observed_refs:
            parts.reasons.add("missing_observed_fact")


def _compile_specs(
    specs: Sequence[ContractMismatch],
    *,
    proof_requirement_refs: Sequence[str] = (),
    security_requirement_refs: Sequence[str] = (),
    validation_requirement_refs: Sequence[str] = (),
    bounds: CompilationBounds | Mapping[str, Any] | None = None,
) -> tuple[ObligationGraph, _GraphParts]:
    if not specs:
        raise DiagnosisObligationAdapterError(
            "at least one contract mismatch is required"
        )
    if len(specs) > MAX_FINDINGS:
        raise DiagnosisObligationBoundsError("finding count exceeds bound")
    repository_ids = {item.repository_id for item in specs}
    root_ids = {item.current_root_id for item in specs}
    if len(repository_ids) != 1 or len(root_ids) != 1:
        raise DiagnosisObligationAuthorityError(
            "contract mismatches must share one repository and current root"
        )
    proof_refs = _ids(proof_requirement_refs, "proof_requirement_refs")
    security_refs = _ids(
        security_requirement_refs, "security_requirement_refs"
    )
    validation_refs = _ids(
        validation_requirement_refs, "validation_requirement_refs"
    )
    parts = _GraphParts()
    for spec in sorted(specs, key=lambda item: item.semantic_id):
        _add_spec(
            parts,
            spec,
            proof_requirement_refs=proof_refs,
            security_requirement_refs=security_refs,
            validation_requirement_refs=validation_refs,
        )

    current_root_id = next(iter(root_ids))
    source_refs = tuple(
        sorted(
            {
                ref
                for spec in specs
                for ref in (*spec.source_refs, *spec.issue_ids)
            }
            or {item.semantic_id for item in specs}
        )
    )
    intent = TypedIntent(
        intent_id=_stable_id(
            "diagnosis-intent",
            {
                "semantic_mismatch_ids": sorted(
                    item.semantic_id for item in specs
                ),
                "current_root_id": current_root_id,
            },
        ),
        desired_predicates=tuple(
            parts.predicates[item] for item in sorted(parts.desired_ids)
        ),
        source_refs=source_refs,
        current_root_id=current_root_id,
        metadata={
            "origin": "diagnosis_obligation_adapter",
            "proposal_only": True,
            "effect_authority": False,
        },
    )
    graph = ObligationGraphCompiler(bounds=bounds).compile(
        intent,
        current_facts=tuple(parts.facts.values()),
        producers=tuple(parts.producers.values()),
        assumptions=tuple(parts.assumptions.values()),
        task_candidates=tuple(parts.candidates.values()),
        predicates=tuple(parts.predicates.values()),
        current_root_id=current_root_id,
    )
    return graph, parts


def compile_contract_mismatch_obligations(
    mismatch: ContractMismatch | Mapping[str, Any] | None = None,
    *,
    repository_id: str = "",
    current_root_id: str = "",
    finding_kind: str = "contract",
    expected_refs: Sequence[str] = (),
    observed_refs: Sequence[str] = (),
    subject_refs: Sequence[str] = (),
    consumer_refs: Sequence[str] = (),
    frontier_refs: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    causal_slice_refs: Sequence[str] = (),
    source_refs: Sequence[str] = (),
    issue_ids: Sequence[str] = (),
    diagnosis_complete: bool = True,
    contradictory: bool = False,
    approval_required: bool = False,
    proof_requirement_refs: Sequence[str] = (),
    security_requirement_refs: Sequence[str] = (),
    validation_requirement_refs: Sequence[str] = (),
    bounds: CompilationBounds | Mapping[str, Any] | None = None,
) -> ObligationGraph:
    """Compile a Planner- or Doctor-originated mismatch through one kernel."""

    if mismatch is not None:
        if any(
            (
                repository_id,
                current_root_id,
                expected_refs,
                observed_refs,
                subject_refs,
                consumer_refs,
                frontier_refs,
                evidence_refs,
                causal_slice_refs,
                source_refs,
                issue_ids,
            )
        ):
            raise DiagnosisObligationAdapterError(
                "supply mismatch or mismatch fields, not both"
            )
        normalized = (
            mismatch
            if isinstance(mismatch, ContractMismatch)
            else ContractMismatch.from_dict(mismatch)
        )
    else:
        normalized = ContractMismatch(
            repository_id=repository_id,
            current_root_id=current_root_id,
            finding_kind=finding_kind,
            expected_refs=tuple(expected_refs),
            observed_refs=tuple(observed_refs),
            subject_refs=tuple(subject_refs),
            consumer_refs=tuple(consumer_refs),
            frontier_refs=tuple(frontier_refs),
            evidence_refs=tuple(evidence_refs),
            causal_slice_refs=tuple(causal_slice_refs),
            source_refs=tuple(source_refs),
            issue_ids=tuple(issue_ids),
            diagnosis_complete=diagnosis_complete,
            contradictory=contradictory,
            approval_required=approval_required,
        )
    return _compile_specs(
        (normalized,),
        proof_requirement_refs=proof_requirement_refs,
        security_requirement_refs=security_requirement_refs,
        validation_requirement_refs=validation_requirement_refs,
        bounds=bounds,
    )[0]


class DiagnosisObligationAdapter:
    """Checked Doctor-to-Planner obligation adapter."""

    INTERFACE: ClassVar[str] = DIAGNOSIS_OBLIGATION_ADAPTER_INTERFACE

    def __init__(
        self,
        *,
        bounds: CompilationBounds | Mapping[str, Any] | None = None,
    ) -> None:
        self.bounds = bounds

    @staticmethod
    def _bridge(
        value: DiagnosisObligationBridge | FindingBridge | Mapping[str, Any],
    ) -> DiagnosisObligationBridge:
        if isinstance(value, DiagnosisObligationBridge):
            return value
        if isinstance(value, FindingBridge):
            return DiagnosisObligationBridge(
                repository_id=value.repository_id,
                finding_bridges=(value,),
                expected_contract_refs=value.expected_refs,
                observed_contract_refs=value.observed_refs,
                causal_slice_refs=value.causal_slice_refs,
                open_frontier_refs=value.open_frontier_refs,
            )
        if not isinstance(value, Mapping):
            raise DiagnosisObligationAdapterError(
                "bridge must be a checked diagnosis/finding bridge"
            )
        schema = value.get("schema")
        if schema == DIAGNOSIS_OBLIGATION_BRIDGE_SCHEMA:
            return DiagnosisObligationBridge.from_dict(value)
        if schema == FINDING_BRIDGE_SCHEMA:
            return DiagnosisObligationAdapter._bridge(
                FindingBridge.from_dict(value)
            )
        raise DiagnosisObligationAdapterError(
            "bridge has an unsupported schema"
        )

    @staticmethod
    def _localizations(
        values: (
            DoctorCausalLocalizationReceipt
            | Mapping[
                str,
                DoctorCausalLocalizationReceipt | Mapping[str, Any],
            ]
            | Sequence[
                DoctorCausalLocalizationReceipt | Mapping[str, Any]
            ]
        ),
    ) -> Mapping[str, DoctorCausalLocalizationReceipt]:
        expected_keys: tuple[str, ...] = ()
        if isinstance(values, DoctorCausalLocalizationReceipt):
            raw: Sequence[
                DoctorCausalLocalizationReceipt | Mapping[str, Any]
            ] = (values,)
        elif isinstance(values, Mapping):
            if values.get("schema") == DoctorCausalLocalizationReceipt.SCHEMA:
                raw = (values,)
            else:
                expected_keys = tuple(
                    _identifier(key, "localization key")
                    for key in values
                )
                raw = tuple(values.values())
        elif isinstance(values, Sequence) and not isinstance(
            values, (str, bytes, bytearray)
        ):
            raw = values
        else:
            raise DiagnosisObligationAdapterError(
                "localizations must be a receipt, sequence, or keyed mapping"
            )
        if len(raw) > MAX_FINDINGS:
            raise DiagnosisObligationBoundsError(
                "localization count exceeds bound"
            )
        result: dict[str, DoctorCausalLocalizationReceipt] = {}
        for item in raw:
            receipt = (
                item
                if isinstance(item, DoctorCausalLocalizationReceipt)
                else DoctorCausalLocalizationReceipt.from_dict(item)
            )
            key = receipt.diagnostic_finding_cid
            if key in result:
                raise DiagnosisObligationAdapterError(
                    "duplicate localization for diagnostic finding"
                )
            result[key] = receipt
        if expected_keys and set(expected_keys) != set(result):
            raise DiagnosisObligationAuthorityError(
                "localization mapping key does not match finding identity"
            )
        return MappingProxyType(result)

    def compile(
        self,
        bridge: (
            DiagnosisObligationRequest
            | DiagnosisObligationBridge
            | FindingBridge
            | Mapping[str, Any]
        ),
        *,
        localizations: (
            DoctorCausalLocalizationReceipt
            | Mapping[
                str,
                DoctorCausalLocalizationReceipt | Mapping[str, Any],
            ]
            | Sequence[
                DoctorCausalLocalizationReceipt | Mapping[str, Any]
            ]
        ) = (),
        proof_requirement_refs: Sequence[str] = (),
        security_requirement_refs: Sequence[str] = (),
        validation_requirement_refs: Sequence[str] = (),
    ) -> DiagnosisObligationCompilation:
        if isinstance(bridge, DiagnosisObligationRequest):
            if (
                localizations
                or proof_requirement_refs
                or security_requirement_refs
                or validation_requirement_refs
            ):
                raise DiagnosisObligationAdapterError(
                    "request object cannot be combined with compile keywords"
                )
            request = bridge
            bridge = request.bridge
            localizations = request.localizations
            proof_requirement_refs = request.proof_requirement_refs
            security_requirement_refs = request.security_requirement_refs
            validation_requirement_refs = request.validation_requirement_refs

        checked_bridge = self._bridge(bridge)
        if not checked_bridge.finding_bridges:
            raise DiagnosisObligationAdapterError(
                "diagnosis bridge contains no findings"
            )
        if len(checked_bridge.finding_bridges) > MAX_FINDINGS:
            raise DiagnosisObligationBoundsError("finding count exceeds bound")
        localization_by_finding = self._localizations(localizations)

        det_findings = tuple(
            item.materialize_deterministic()
            for item in checked_bridge.finding_bridges
        )
        root_cids = {item.roots.content_id for item in det_findings}
        repository_ids = {item.roots.repository_id for item in det_findings}
        tree_ids = {item.roots.tree_id for item in det_findings}
        if (
            repository_ids != {checked_bridge.repository_id}
            or len(root_cids) != 1
            or len(tree_ids) != 1
        ):
            raise DiagnosisObligationAuthorityError(
                "finding bridges do not share the bridge repository/root"
            )
        if checked_bridge.root_bridge is not None:
            expected_root_cid = (
                checked_bridge.root_bridge.deterministic_content_id
            )
            if root_cids != {expected_root_cid}:
                raise DiagnosisObligationAuthorityError(
                    "finding root does not match checked root bridge"
                )

        known_finding_ids = {
            item.diagnostic_finding_cid
            for item in checked_bridge.finding_bridges
        }
        if set(localization_by_finding).difference(known_finding_ids):
            raise DiagnosisObligationAuthorityError(
                "localization does not belong to this diagnosis bridge"
            )

        specs: list[ContractMismatch] = []
        evidence_ids: set[str] = set()
        causal_slice_ids: set[str] = set(checked_bridge.causal_slice_refs)
        frontier_ids: set[str] = set(checked_bridge.open_frontier_refs)
        semantic_issue_ids: set[str] = set()
        schema_ids: set[str] = {
            DIAGNOSIS_OBLIGATION_BRIDGE_SCHEMA,
            FINDING_BRIDGE_SCHEMA,
            OBLIGATION_GRAPH_SCHEMA,
            DIAGNOSIS_OBLIGATION_COMPILATION_SCHEMA,
        }
        snapshot_ids: set[str] = set()
        if checked_bridge.snapshot_bridge is not None:
            snapshot = checked_bridge.snapshot_bridge
            schema_ids.add(SNAPSHOT_BRIDGE_SCHEMA)
            snapshot_ids.update(
                {
                    snapshot.diagnostic_snapshot_cid,
                    snapshot.diagnostic_snapshot_id,
                    snapshot.deterministic_snapshot_id,
                    snapshot.deterministic_content_id,
                }
            )
        if checked_bridge.root_bridge is not None:
            schema_ids.add(ROOT_BRIDGE_SCHEMA)

        any_contradictory = False
        any_incomplete = False
        for finding_bridge, det_finding in zip(
            checked_bridge.finding_bridges, det_findings
        ):
            diagnostic = finding_bridge.materialize_diagnostic()
            localization = localization_by_finding.get(
                finding_bridge.diagnostic_finding_cid
            )
            local_complete = False
            contradictory = False
            consumers = set(det_finding.consumer_refs)
            frontiers = set(finding_bridge.open_frontier_refs)
            frontiers.update(det_finding.open_frontier_refs)
            local_evidence: set[str] = set(det_finding.observed_fact_refs)
            local_evidence.update(diagnostic.evidence_refs)
            local_slices: set[str] = set(finding_bridge.causal_slice_refs)
            issue_ids = {
                finding_bridge.issue_cid,
                finding_bridge.diagnostic_finding_cid,
                finding_bridge.deterministic_finding_id,
            }
            if localization is not None:
                if localization.repository_id != checked_bridge.repository_id:
                    raise DiagnosisObligationAuthorityError(
                        "localization repository does not match bridge"
                    )
                if (
                    checked_bridge.snapshot_bridge is not None
                    and localization.snapshot_cid
                    != checked_bridge.snapshot_bridge.diagnostic_snapshot_cid
                ):
                    raise DiagnosisObligationAuthorityError(
                        "localization snapshot does not match bridge"
                    )
                semantic_issue_ids.add(localization.issue_cid)
                issue_ids.add(localization.issue_cid)
                local_slices.add(localization.mismatch_slice.slice_cid)
                local_slices.add(localization.localization_cid)
                local_evidence.update(localization.exact_evidence_ids)
                local_evidence.update(
                    localization.mismatch_slice.evidence_ids
                )
                consumers.update(localization.mandatory_consumer_ids)
                consumers.update(
                    localization.mismatch_slice.mandatory_consumer_ids
                )
                frontiers.update(localization.open_frontier_refs)
                frontiers.update(
                    localization.mismatch_slice.open_frontier_refs
                )
                contradictory = any(
                    marker in reason.casefold()
                    for reason in localization.reason_codes
                    for marker in _CONTRADICTION_MARKERS
                )
                local_complete = (
                    localization.disposition
                    is CausalLocalizationDisposition.LOCALIZED
                    and localization.complete_frontier_accounting
                    and not frontiers
                    and not contradictory
                )
                schema_ids.update(
                    {
                        localization.SCHEMA,
                        localization.mismatch_slice.SCHEMA,
                    }
                )
            supported_disposition = (
                diagnostic.disposition is diag.FindingDisposition.SUPPORTED
                and det_finding.disposition
                is det.DoctorRepairDisposition.SUPPORTED
            )
            approval_required = (
                diagnostic.disposition
                is diag.FindingDisposition.APPROVAL_REQUIRED
                or det_finding.disposition
                is det.DoctorRepairDisposition.APPROVAL_REQUIRED
            )
            complete = (
                supported_disposition
                and local_complete
                and diagnostic.kind.value in _SUPPORTED_FINDING_KINDS
                and bool(finding_bridge.expected_refs)
                and bool(finding_bridge.observed_refs)
                and not approval_required
            )
            any_contradictory = any_contradictory or contradictory
            any_incomplete = any_incomplete or not complete
            evidence_ids.update(local_evidence)
            causal_slice_ids.update(local_slices)
            frontier_ids.update(frontiers)
            schema_ids.update(
                {
                    str(finding_bridge.diagnostic_payload.get("schema") or ""),
                    str(finding_bridge.deterministic_payload.get("schema") or ""),
                }
            )
            snapshot_ids.add(finding_bridge.snapshot_id)
            specs.append(
                ContractMismatch(
                    repository_id=checked_bridge.repository_id,
                    current_root_id=det_finding.roots.tree_id,
                    finding_kind=diagnostic.kind.value,
                    expected_refs=finding_bridge.expected_refs,
                    observed_refs=finding_bridge.observed_refs,
                    subject_refs=tuple(
                        sorted(
                            {
                                *det_finding.affected_symbol_refs,
                                *((diagnostic.symbol,) if diagnostic.symbol else ()),
                            }
                        )
                    ),
                    consumer_refs=tuple(consumers),
                    frontier_refs=tuple(frontiers),
                    evidence_refs=tuple(local_evidence),
                    causal_slice_refs=tuple(local_slices),
                    source_refs=(
                        checked_bridge.content_id,
                        finding_bridge.content_id,
                        *(
                            (localization.localization_cid,)
                            if localization is not None
                            else ()
                        ),
                    ),
                    issue_ids=tuple(issue_ids),
                    diagnosis_complete=complete,
                    contradictory=contradictory,
                    approval_required=approval_required,
                )
            )

        graph, parts = _compile_specs(
            specs,
            proof_requirement_refs=proof_requirement_refs,
            security_requirement_refs=security_requirement_refs,
            validation_requirement_refs=validation_requirement_refs,
            bounds=self.bounds,
        )
        if any_contradictory:
            disposition = DiagnosisCompilationDisposition.ABSTAINED
        elif any_incomplete:
            disposition = DiagnosisCompilationDisposition.REVIEW_REQUIRED
        else:
            disposition = DiagnosisCompilationDisposition.COMPILED

        authority_root_ids: dict[str, str] = {}
        first_roots = det_findings[0].roots
        for name in det.AUTHORITY_ROOT_FIELDS:
            value = getattr(first_roots, name)
            if value:
                authority_root_ids[name] = value
        authority_root_ids["deterministic_root_cid"] = first_roots.content_id
        if checked_bridge.root_bridge is not None:
            authority_root_ids["diagnostic_root_cid"] = (
                checked_bridge.root_bridge.diagnostic_content_id
            )

        def obligations(predicate_ids: set[str]) -> tuple[str, ...]:
            return tuple(
                obligation_id_for_predicate(item)
                for item in sorted(predicate_ids)
            )

        alternative_ids: list[str] = []
        for producer_id, effect_id in parts.alternative_pairs:
            if producer_id:
                alternative_ids.append(
                    obligation_id_for_producer(producer_id, effect_id)
                )
            else:
                alternative_ids.append(
                    obligation_id_for_predicate(effect_id)
                )

        bridge_ids = {
            checked_bridge.content_id,
            *(item.content_id for item in checked_bridge.finding_bridges),
        }
        if checked_bridge.snapshot_bridge is not None:
            bridge_ids.add(checked_bridge.snapshot_bridge.content_id)
        if checked_bridge.root_bridge is not None:
            bridge_ids.add(checked_bridge.root_bridge.content_id)

        return DiagnosisObligationCompilation(
            repository_id=checked_bridge.repository_id,
            bridge_ids=tuple(bridge_ids),
            issue_ids=tuple(
                {
                    issue
                    for spec in specs
                    for issue in spec.issue_ids
                }
            ),
            semantic_issue_ids=tuple(semantic_issue_ids),
            snapshot_ids=tuple(snapshot_ids),
            authority_root_ids=authority_root_ids,
            schema_ids=tuple(item for item in schema_ids if item),
            evidence_ids=tuple(evidence_ids),
            causal_slice_ids=tuple(causal_slice_ids),
            frontier_ids=tuple(frontier_ids),
            graph=graph,
            disposition=disposition,
            desired_predicate_ids=tuple(parts.desired_ids),
            observed_fact_ids=tuple(parts.observed_fact_ids),
            assumption_ids=tuple(parts.assumptions),
            prohibition_obligation_ids=obligations(
                parts.prohibition_predicate_ids
            ),
            impact_obligation_ids=obligations(parts.impact_predicate_ids),
            proof_obligation_ids=obligations(parts.proof_predicate_ids),
            security_obligation_ids=obligations(
                parts.security_predicate_ids
            ),
            validation_obligation_ids=obligations(
                parts.validation_predicate_ids
            ),
            alternative_repair_subgoal_ids=tuple(alternative_ids),
            review_obligation_ids=obligations(
                parts.review_predicate_ids
            ),
            abstention_obligation_ids=obligations(
                parts.abstention_predicate_ids
            ),
            reason_codes=tuple(parts.reasons),
        )

    compile_diagnosis = compile
    adapt = compile
    translate = compile


def compile_diagnosis_obligations(
    bridge: (
        DiagnosisObligationRequest
        | DiagnosisObligationBridge
        | FindingBridge
        | Mapping[str, Any]
    ),
    *,
    localizations: (
        DoctorCausalLocalizationReceipt
        | Mapping[
            str,
            DoctorCausalLocalizationReceipt | Mapping[str, Any],
        ]
        | Sequence[
            DoctorCausalLocalizationReceipt | Mapping[str, Any]
        ]
    ) = (),
    proof_requirement_refs: Sequence[str] = (),
    security_requirement_refs: Sequence[str] = (),
    validation_requirement_refs: Sequence[str] = (),
    bounds: CompilationBounds | Mapping[str, Any] | None = None,
) -> DiagnosisObligationCompilation:
    """Functional spelling for checked Doctor diagnosis lowering."""

    return DiagnosisObligationAdapter(bounds=bounds).compile(
        bridge,
        localizations=localizations,
        proof_requirement_refs=proof_requirement_refs,
        security_requirement_refs=security_requirement_refs,
        validation_requirement_refs=validation_requirement_refs,
    )


# Compatibility spelling emphasizing the adapter operation.
adapt_diagnosis_to_obligations = compile_diagnosis_obligations


def compile_diagnosis_obligation_graph(
    bridge: (
        DiagnosisObligationRequest
        | DiagnosisObligationBridge
        | FindingBridge
        | Mapping[str, Any]
    ),
    **kwargs: Any,
) -> ObligationGraph:
    """Return only the shared graph for callers which do not persist receipts."""

    return compile_diagnosis_obligations(bridge, **kwargs).graph


__all__ = [
    "CONTRACT_MISMATCH_SCHEMA",
    "DIAGNOSIS_OBLIGATION_ADAPTER_INTERFACE",
    "DIAGNOSIS_OBLIGATION_ADAPTER_VERSION",
    "DIAGNOSIS_OBLIGATION_COMPILATION_SCHEMA",
    "ContractMismatch",
    "DiagnosisCompilationDisposition",
    "DiagnosisObligationAdapter",
    "DiagnosisObligationAdapterError",
    "DiagnosisObligationAuthorityError",
    "DiagnosisObligationBoundsError",
    "DiagnosisObligationCompilation",
    "DiagnosisObligationReceipt",
    "DiagnosisObligationRequest",
    "DiagnosisObligationTamperError",
    "adapt_diagnosis_to_obligations",
    "compile_contract_mismatch_obligations",
    "compile_diagnosis_obligation_graph",
    "compile_diagnosis_obligations",
    "formal_obligation_signature_for_graph",
]
