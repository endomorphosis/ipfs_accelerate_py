"""Invariant candidate mining over admitted specification sources.

``InvariantMiner`` proposes invariant-kind properties only.  It reuses
specification mining's closed source schema, provenance/tier retention, and
fail-closed conflict rule.  Non-vacuity, mutant, and adversarial validation
are later independent obligations and are not performed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .contracts import (
    ArtifactBindings,
    ArtifactState,
    ConditionOperator,
    ExecutionTrajectory,
    ProcedureSpec,
    TaskFamily,
    _identifier,
    _nonnegative_int,
)
from .contracts import (
    InvariantCandidate as InvariantCandidateArtifact,
)
from .specification_mining import (
    AdmittedSource,
    EvidenceTier,
    MiningSource,
    PropertyKind,
    PropertyNomination,
    SourceKind,
    SpecificationCandidate,
    SpecificationCounterexample,
    SpecificationMiner,
    SpecificationMiningError,
)

INVARIANT_MINER_REVISION: Final[str] = "invariant-miner@1"


class InvariantMiningError(SpecificationMiningError):
    """Invariant candidates could not be mined from admitted sources."""


@dataclass(frozen=True)
class InvariantMiningResult:
    """Invariant-only mining output with candidate-status wire artifacts."""

    bindings: ArtifactBindings
    candidates: tuple[SpecificationCandidate, ...]
    refused: tuple[SpecificationCandidate, ...]
    counterexamples: tuple[SpecificationCounterexample, ...]
    invariant_artifacts: tuple[InvariantCandidateArtifact, ...]
    retained_source_kinds: tuple[SourceKind, ...]
    retained_evidence_tiers: tuple[EvidenceTier, ...]

    @property
    def upgraded_count(self) -> int:
        return 0


def _invariant_nomination(
    *,
    property_id: str,
    binding: str,
    operator: ConditionOperator,
    operand: object,
    evidence_cid: str,
) -> PropertyNomination:
    return PropertyNomination(
        property_kind=PropertyKind.INVARIANT,
        property_id=property_id,
        binding=binding,
        operator=operator,
        operand=operand,
        evidence_cid=evidence_cid,
        required=True,
    )


def project_invariant_sources(sources: Sequence[MiningSource]) -> tuple[AdmittedSource, ...]:
    """Project extra invariant nominations that hold across a source artifact."""

    extra: list[AdmittedSource] = []
    for item in sources:
        if isinstance(item, AdmittedSource):
            continue
        if isinstance(item, ProcedureSpec):
            provenance = item.content_id
            extra.append(
                AdmittedSource(
                    bindings=item.bindings,
                    source_id=f"invariant-runtime.{item.name}",
                    source_kind=SourceKind.RUNTIME_CHECK,
                    evidence_tier=EvidenceTier.RUNTIME_OBSERVATION,
                    provenance_cid=provenance,
                    artifact_cid=provenance,
                    nominations=(
                        _invariant_nomination(
                            property_id="invariant.scope-respected",
                            binding="procedure.scope_paths",
                            operator=ConditionOperator.SUBSET_OF,
                            operand=item.scope_paths,
                            evidence_cid=provenance,
                        ),
                        _invariant_nomination(
                            property_id="invariant.tree-current",
                            binding="bindings.tree_id",
                            operator=ConditionOperator.CURRENT,
                            operand=item.bindings.tree_id,
                            evidence_cid=provenance,
                        ),
                    ),
                )
            )
            continue
        if isinstance(item, ExecutionTrajectory):
            provenance = item.content_id
            extra.append(
                AdmittedSource(
                    bindings=item.bindings,
                    source_id=f"invariant-trace.{item.source_episode_cid}",
                    source_kind=SourceKind.RUNTIME_CHECK,
                    evidence_tier=EvidenceTier.RUNTIME_OBSERVATION,
                    provenance_cid=provenance,
                    artifact_cid=item.source_episode_cid,
                    nominations=(
                        _invariant_nomination(
                            property_id="invariant.contiguous-state-chain",
                            binding="trajectory.steps",
                            operator=ConditionOperator.EQUALS,
                            operand=tuple(
                                step.terminal_state_cid == nxt.initial_state_cid
                                for step, nxt in zip(
                                    item.steps, item.steps[1:], strict=False
                                )
                            ),
                            evidence_cid=provenance,
                        ),
                    ),
                )
            )
            continue
        if isinstance(item, TaskFamily):
            provenance = item.content_id
            extra.append(
                AdmittedSource(
                    bindings=item.bindings,
                    source_id=f"invariant-family.{item.name}",
                    source_kind=SourceKind.TYPE,
                    evidence_tier=EvidenceTier.TYPE_DECLARATION,
                    provenance_cid=provenance,
                    artifact_cid=provenance,
                    nominations=(
                        _invariant_nomination(
                            property_id="invariant.effect-ceiling",
                            binding="task_family.boundary.permitted_effect_classes",
                            operator=ConditionOperator.SUBSET_OF,
                            operand=tuple(
                                effect.value for effect in item.boundary.permitted_effect_classes
                            ),
                            evidence_cid=provenance,
                        ),
                    ),
                )
            )
    return tuple(extra)


class InvariantMiner:
    """Propose bounded invariant candidates; never certify or promote them."""

    def __init__(
        self,
        *,
        miner_revision: str = INVARIANT_MINER_REVISION,
        emitted_at_ms: int = 0,
    ) -> None:
        self.miner_revision = _identifier(miner_revision, "miner_revision")
        self.emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")
        self._specification_miner = SpecificationMiner(
            miner_revision=self.miner_revision,
            emitted_at_ms=self.emitted_at_ms,
        )

    def mine(self, sources: Sequence[MiningSource]) -> InvariantMiningResult:
        extra = project_invariant_sources(sources)
        result = self._specification_miner.mine((*sources, *extra))
        candidates = tuple(
            item for item in result.candidates if item.property_kind is PropertyKind.INVARIANT
        )
        refused = tuple(
            item for item in result.refused if item.property_kind is PropertyKind.INVARIANT
        )
        counterexamples = tuple(
            item
            for item in result.counterexamples
            if item.property_kind is PropertyKind.INVARIANT
        )
        artifacts = tuple(
            item.to_invariant_artifact(result.bindings, emitted_at_ms=self.emitted_at_ms)
            for item in candidates
        )
        if any(item.state is not ArtifactState.CANDIDATE for item in artifacts):
            raise InvariantMiningError("invariant candidates cannot leave candidate state")
        return InvariantMiningResult(
            bindings=result.bindings,
            candidates=candidates,
            refused=refused,
            counterexamples=counterexamples,
            invariant_artifacts=artifacts,
            retained_source_kinds=result.retained_source_kinds,
            retained_evidence_tiers=result.retained_evidence_tiers,
        )


__all__ = [
    "INVARIANT_MINER_REVISION",
    "InvariantCandidateArtifact",
    "InvariantMiner",
    "InvariantMiningError",
    "InvariantMiningResult",
    "project_invariant_sources",
]
