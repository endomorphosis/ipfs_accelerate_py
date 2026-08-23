"""Exact causal evidence admission and nomination-only retrieval projection.

Retrieval proposes. Exact analysis disposes.  Doctor localization remains
report-only: a localization receipt never becomes federation graph authority,
policy, proof, independence, or completion.  Observational similarity cannot
prove cause or independence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar

from ..analysis.doctor_causal_localization import (
    CausalEvidence as DoctorCausalEvidence,
)
from ..analysis.doctor_causal_localization import (
    DoctorCausalLocalizationReceipt,
    doctor_kind_is_federation_nomination,
    federation_kind_for_doctor_evidence,
)
from ..task_sources.control_plane_contracts import content_identity
from .causal_graph import CausalGraphCommit, CausalGraphStore
from .contracts import (
    CausalEdge,
    CausalEvidence,
    CausalEvidenceKind,
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    _identifier,
    _integer,
    _text,
)

_SCHEMA_PREFIX = "ipfs_accelerate_py/agent-supervisor/causal-federation"
_EXACT_FEDERATION_KINDS = frozenset(
    kind
    for kind in CausalEvidenceKind
    if kind is not CausalEvidenceKind.RETRIEVAL_NOMINATION
)
_RETRIEVAL_METHODS = frozenset({"bm25", "vector", "kg", "lexical", "hybrid"})
_ADMISSION_SOURCES = frozenset(
    {
        "federation_native",
        "doctor_localization",
        "retrieval_candidate",
    }
)


class CausalEvidenceAdmissionError(FederationContractError):
    """Malformed causal-evidence admission input."""


class CausalEvidenceAuthorityError(FederationAuthorityError):
    """An attempt to mint authority from nomination or report-only analysis."""


@dataclass(frozen=True)
class RetrievalNominationBinding:
    """Exact release bindings required of every retrieval nomination."""

    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/retrieval-nomination-binding@1"

    index_revision: str
    source_cid: str
    tree_id: str
    method: str
    score_millionths: int
    partition_id: str = ""

    def __post_init__(self) -> None:
        _identifier(self.index_revision, "index_revision")
        _identifier(self.source_cid, "source_cid")
        _identifier(self.tree_id, "tree_id")
        method = _text(self.method, "method", maximum=64).casefold()
        if method not in _RETRIEVAL_METHODS:
            raise CausalEvidenceAdmissionError("retrieval method is not closed")
        object.__setattr__(self, "method", method)
        _integer(self.score_millionths, "score_millionths", maximum=1_000_000)
        _identifier(self.partition_id, "partition_id", required=False)

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "index_revision": self.index_revision,
                "source_cid": self.source_cid,
                "tree_id": self.tree_id,
                "method": self.method,
                "score_millionths": self.score_millionths,
                "partition_id": self.partition_id,
            }
        )


@dataclass(frozen=True)
class CausalEvidenceAdmission:
    """One federation evidence record plus the non-authoritative source trail."""

    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/causal-evidence-admission@1"

    evidence: CausalEvidence
    source_kind: str
    doctor_evidence_id: str = ""
    localization_cid: str = ""
    retrieval_binding: RetrievalNominationBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, CausalEvidence):
            raise CausalEvidenceAdmissionError("evidence must be a CausalEvidence")
        source = _identifier(self.source_kind, "source_kind")
        if source not in _ADMISSION_SOURCES:
            raise CausalEvidenceAdmissionError("admission source is not closed")
        object.__setattr__(self, "source_kind", source)
        _identifier(self.doctor_evidence_id, "doctor_evidence_id", required=False)
        _identifier(self.localization_cid, "localization_cid", required=False)
        if self.retrieval_binding is not None and not isinstance(
            self.retrieval_binding, RetrievalNominationBinding
        ):
            raise CausalEvidenceAdmissionError(
                "retrieval_binding must be a RetrievalNominationBinding"
            )
        _assert_admission_authority(self)

    @property
    def nomination_only(self) -> bool:
        return (
            not self.evidence.authoritative
            or self.evidence.evidence_kind is CausalEvidenceKind.RETRIEVAL_NOMINATION
        )


@dataclass(frozen=True)
class CausalEvidenceDispositionReceipt:
    """Exact-versus-nomination split after analysis disposes retrieval proposals."""

    SCHEMA: ClassVar[str] = f"{_SCHEMA_PREFIX}/causal-evidence-disposition@1"

    exact_evidence_ids: tuple[str, ...]
    nomination_evidence_ids: tuple[str, ...]
    rejected_evidence_ids: tuple[str, ...]
    localization_cid: str = ""
    federation_authority_admitted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exact_evidence_ids",
            _unique_ids(self.exact_evidence_ids, "exact_evidence_ids"),
        )
        object.__setattr__(
            self,
            "nomination_evidence_ids",
            _unique_ids(self.nomination_evidence_ids, "nomination_evidence_ids"),
        )
        object.__setattr__(
            self,
            "rejected_evidence_ids",
            _unique_ids(self.rejected_evidence_ids, "rejected_evidence_ids"),
        )
        _identifier(self.localization_cid, "localization_cid", required=False)
        if self.federation_authority_admitted is not False:
            raise CausalEvidenceAuthorityError(
                "analysis disposition cannot admit federation authority"
            )
        overlap = set(self.exact_evidence_ids) & set(self.nomination_evidence_ids)
        if overlap:
            raise CausalEvidenceAuthorityError(
                "an evidence identity cannot be both exact and nomination-only"
            )


def _unique_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(_identifier(item, name) for item in values)
    if len(set(result)) != len(result):
        raise CausalEvidenceAdmissionError(f"{name} contains duplicate identities")
    return result


def _assert_admission_authority(admission: CausalEvidenceAdmission) -> None:
    evidence = admission.evidence
    if (
        evidence.evidence_kind is CausalEvidenceKind.RETRIEVAL_NOMINATION
        and evidence.authoritative
    ):
        raise CausalEvidenceAuthorityError(
            "retrieval nomination cannot be authoritative causal evidence"
        )
    if admission.retrieval_binding is not None and evidence.authoritative:
        raise CausalEvidenceAuthorityError(
            "retrieval-bound evidence cannot be authoritative"
        )
    if (
        admission.source_kind == "retrieval_candidate"
        and evidence.evidence_kind is not CausalEvidenceKind.RETRIEVAL_NOMINATION
    ):
        raise CausalEvidenceAdmissionError(
            "retrieval candidates project only as retrieval nominations"
        )
    if admission.source_kind == "doctor_localization" and evidence.authoritative:
        raise CausalEvidenceAuthorityError(
            "doctor localization is report-only and cannot mint federation authority"
        )


def federation_kind_from_doctor(kind: Any) -> CausalEvidenceKind:
    """Map a doctor evidence kind onto the closed federation vocabulary."""

    name = federation_kind_for_doctor_evidence(kind)
    try:
        return CausalEvidenceKind(name)
    except ValueError as exc:
        raise CausalEvidenceAdmissionError(
            "doctor evidence kind is not a closed federation kind"
        ) from exc


def project_doctor_evidence(
    doctor_evidence: DoctorCausalEvidence,
    *,
    binding: FederationBinding,
    record_id: str,
    localization: DoctorCausalLocalizationReceipt | None = None,
) -> CausalEvidenceAdmission:
    """Project one doctor fact as federation evidence without granting authority.

    Exact doctor facts remain non-authoritative until a separate federation
    admission copies their identity under federation rules.  Nominations stay
    nomination-only.
    """

    if not isinstance(doctor_evidence, DoctorCausalEvidence):
        raise CausalEvidenceAdmissionError("doctor evidence has invalid type")
    if not isinstance(binding, FederationBinding):
        raise CausalEvidenceAdmissionError("binding must be a FederationBinding")
    kind = federation_kind_from_doctor(doctor_evidence.kind)
    nomination = (
        kind is CausalEvidenceKind.RETRIEVAL_NOMINATION
        or doctor_kind_is_federation_nomination(doctor_evidence.kind)
        or doctor_evidence.nomination_only
    )
    if localization is not None:
        if localization.federation_authority_admitted:
            raise CausalEvidenceAuthorityError(
                "doctor localization receipt cannot admit federation authority"
            )
        if doctor_evidence.evidence_id in localization.nomination_evidence_ids:
            nomination = True
        if doctor_evidence.evidence_id in localization.rejected_evidence_ids:
            raise CausalEvidenceAuthorityError(
                "rejected doctor evidence cannot be projected as causal fact"
            )
    evidence = CausalEvidence(
        record_id=_identifier(record_id, "record_id"),
        revision=1,
        binding=binding,
        evidence_kind=kind,
        evidence_ref=doctor_evidence.evidence_id,
        authoritative=False,
    )
    if nomination and evidence.evidence_kind is not CausalEvidenceKind.RETRIEVAL_NOMINATION:
        evidence = replace(
            evidence,
            evidence_kind=CausalEvidenceKind.RETRIEVAL_NOMINATION,
        )
    return CausalEvidenceAdmission(
        evidence=evidence,
        source_kind="doctor_localization",
        doctor_evidence_id=doctor_evidence.evidence_id,
        localization_cid="" if localization is None else localization.localization_cid,
    )


def project_retrieval_candidate(
    candidate: Any,
    *,
    binding: FederationBinding,
    record_id: str,
) -> CausalEvidenceAdmission:
    """Project a bound retrieval hit as nomination-only federation evidence."""

    if not isinstance(binding, FederationBinding):
        raise CausalEvidenceAdmissionError("binding must be a FederationBinding")
    node_id = _identifier(getattr(candidate, "node_id", ""), "candidate.node_id")
    source = _text(getattr(candidate, "source", ""), "candidate.source", maximum=64)
    index_root = _identifier(
        getattr(candidate, "index_root_id", "") or getattr(candidate, "index_revision", ""),
        "candidate.index_root_id",
    )
    snapshot = getattr(candidate, "binding", None)
    tree_id = _identifier(
        getattr(snapshot, "graph_id", "") or binding.repository_tree_ids[0],
        "candidate.tree_id",
    )
    partition_id = _identifier(
        getattr(snapshot, "partition_id", ""),
        "candidate.partition_id",
        required=False,
    )
    method = source.casefold()
    if method == "ast_symbol":
        method = "lexical"
    if method in {"dependency_neighborhood", "goal_coverage", "proof_gap"}:
        method = "kg"
    nomination = RetrievalNominationBinding(
        index_revision=index_root,
        source_cid=node_id,
        tree_id=tree_id,
        method=method,
        score_millionths=int(getattr(candidate, "score_millionths", 0) or 0),
        partition_id=partition_id,
    )
    evidence = CausalEvidence(
        record_id=_identifier(record_id, "record_id"),
        revision=1,
        binding=binding,
        evidence_kind=CausalEvidenceKind.RETRIEVAL_NOMINATION,
        evidence_ref=nomination.cid,
        authoritative=False,
    )
    return CausalEvidenceAdmission(
        evidence=evidence,
        source_kind="retrieval_candidate",
        retrieval_binding=nomination,
    )


def admit_exact_evidence(
    evidence: CausalEvidence,
    *,
    source_kind: str = "federation_native",
) -> CausalEvidenceAdmission:
    """Admit one exact federation evidence record under federation rules."""

    if not isinstance(evidence, CausalEvidence):
        raise CausalEvidenceAdmissionError("evidence must be a CausalEvidence")
    if evidence.evidence_kind not in _EXACT_FEDERATION_KINDS:
        raise CausalEvidenceAuthorityError(
            "retrieval nomination cannot be admitted as exact evidence"
        )
    if not evidence.authoritative:
        raise CausalEvidenceAdmissionError(
            "exact admission requires authoritative=True on the federation record"
        )
    if source_kind == "doctor_localization":
        raise CausalEvidenceAuthorityError(
            "doctor localization is report-only; copy the fact under federation rules"
        )
    return CausalEvidenceAdmission(evidence=evidence, source_kind=source_kind)


def admit_exact_from_doctor(
    doctor_evidence: DoctorCausalEvidence,
    *,
    binding: FederationBinding,
    record_id: str,
    localization: DoctorCausalLocalizationReceipt,
) -> CausalEvidenceAdmission:
    """Copy an exact doctor fact into federation authority after analysis disposes.

    The doctor receipt remains non-authoritative.  Federation creates a new
    exact record only when localization classified the fact as exact.
    """

    if not isinstance(localization, DoctorCausalLocalizationReceipt):
        raise CausalEvidenceAdmissionError("localization receipt is required")
    if localization.federation_authority_admitted:
        raise CausalEvidenceAuthorityError(
            "doctor localization receipt cannot admit federation authority"
        )
    if doctor_evidence.evidence_id not in localization.exact_evidence_ids:
        raise CausalEvidenceAuthorityError(
            "only exact doctor facts can be admitted as federation authority"
        )
    if doctor_kind_is_federation_nomination(doctor_evidence.kind):
        raise CausalEvidenceAuthorityError(
            "doctor nomination kinds cannot be admitted as exact evidence"
        )
    projected = project_doctor_evidence(
        doctor_evidence,
        binding=binding,
        record_id=record_id,
        localization=localization,
    )
    if projected.evidence.evidence_kind is CausalEvidenceKind.RETRIEVAL_NOMINATION:
        raise CausalEvidenceAuthorityError(
            "projected doctor nomination cannot be admitted as exact evidence"
        )
    exact = replace(projected.evidence, authoritative=True)
    return CausalEvidenceAdmission(
        evidence=exact,
        source_kind="federation_native",
        doctor_evidence_id=doctor_evidence.evidence_id,
        localization_cid=localization.localization_cid,
    )


def dispose_with_localization(
    localization: DoctorCausalLocalizationReceipt,
) -> CausalEvidenceDispositionReceipt:
    """Split exact, nomination, and rejected identities from a doctor receipt."""

    if not isinstance(localization, DoctorCausalLocalizationReceipt):
        raise CausalEvidenceAdmissionError("localization receipt is required")
    if localization.federation_authority_admitted:
        raise CausalEvidenceAuthorityError(
            "doctor localization receipt cannot admit federation authority"
        )
    return CausalEvidenceDispositionReceipt(
        exact_evidence_ids=tuple(localization.exact_evidence_ids),
        nomination_evidence_ids=tuple(localization.nomination_evidence_ids),
        rejected_evidence_ids=tuple(localization.rejected_evidence_ids),
        localization_cid=localization.localization_cid,
        federation_authority_admitted=False,
    )


def nominations_cannot_prove_independence(
    nomination_ids: Sequence[str],
    *,
    claimed_independent_ids: Sequence[str] = (),
) -> None:
    """Observational similarity cannot prove cause or independence."""

    _unique_ids(nomination_ids, "nomination_ids")
    claimed = _unique_ids(claimed_independent_ids, "claimed_independent_ids")
    if claimed:
        raise CausalEvidenceAuthorityError(
            "retrieval nominations cannot prove independence"
        )


def authoritative_evidence_ids(admissions: Sequence[CausalEvidenceAdmission]) -> tuple[str, ...]:
    """Return identities that may authorize a causal edge."""

    exact: list[str] = []
    for item in admissions:
        if not isinstance(item, CausalEvidenceAdmission):
            raise CausalEvidenceAdmissionError("admissions contain an invalid item")
        if item.nomination_only or not item.evidence.authoritative:
            continue
        exact.append(item.evidence.record_id)
    return tuple(exact)


class CausalEvidenceGateway:
    """Persist admitted exact or nomination-only evidence through the graph store."""

    INTERFACE: ClassVar[str] = "CausalEvidenceGateway@1"

    def __init__(self, store: CausalGraphStore) -> None:
        if not isinstance(store, CausalGraphStore):
            raise CausalEvidenceAdmissionError(
                "gateway requires a CausalGraphStore, not a database path"
            )
        self._store = store

    def record(
        self,
        admission: CausalEvidenceAdmission,
        *,
        federation_id: str,
        expected_graph_revision: int,
        owner_id: str,
        source_root: str,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        _assert_admission_authority(admission)
        return self._store.record_evidence(
            admission.evidence,
            federation_id=federation_id,
            expected_graph_revision=expected_graph_revision,
            owner_id=owner_id,
            source_root=source_root,
            idempotency_key=idempotency_key,
        )

    def record_edge(
        self,
        edge: CausalEdge,
        admissions: Sequence[CausalEvidenceAdmission],
        *,
        federation_id: str,
        expected_graph_revision: int,
        idempotency_key: str,
        fixed_point_group_id: str = "",
    ) -> CausalGraphCommit:
        admitted = {item.evidence.record_id: item for item in admissions}
        for evidence_id in edge.evidence_refs:
            admission = admitted.get(evidence_id)
            if admission is None:
                raise CausalEvidenceAdmissionError(
                    "causal edge evidence is not in the admission set"
                )
            if not edge.nomination_only and admission.nomination_only:
                raise CausalEvidenceAuthorityError(
                    "nomination-only evidence cannot authorize a causal edge"
                )
        return self._store.record_edge(
            edge,
            federation_id=federation_id,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            fixed_point_group_id=fixed_point_group_id,
        )


def doctor_kind_projection_inventory() -> Mapping[str, str]:
    """Return the closed doctor-kind to federation-kind name inventory."""

    from ..analysis.doctor_causal_localization import (
        FEDERATION_KIND_FOR_DOCTOR_KIND,
    )
    from ..analysis.doctor_causal_localization import (
        CausalEvidenceKind as DoctorKind,
    )

    return MappingProxyType(
        {kind.value: FEDERATION_KIND_FOR_DOCTOR_KIND[kind] for kind in DoctorKind}
    )


__all__ = (
    "CausalEvidenceAdmission",
    "CausalEvidenceAdmissionError",
    "CausalEvidenceAuthorityError",
    "CausalEvidenceDispositionReceipt",
    "CausalEvidenceGateway",
    "RetrievalNominationBinding",
    "admit_exact_evidence",
    "admit_exact_from_doctor",
    "authoritative_evidence_ids",
    "dispose_with_localization",
    "doctor_kind_projection_inventory",
    "federation_kind_from_doctor",
    "nominations_cannot_prove_independence",
    "project_doctor_evidence",
    "project_retrieval_candidate",
)
