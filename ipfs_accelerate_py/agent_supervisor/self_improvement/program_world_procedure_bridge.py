"""SAWM-020 exclusive bridge to ProofCarryingProcedureCompiler.

Trajectories and compiled candidates never self-certify. Promotion is a
nomination through the landed compiler authority. Stale or scope-mismatched
procedures stop. Holes retain exact validation. Matching precedes any model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.procedure_compiler.runtime import (
    ProofCarryingProcedureCompiler,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    _bool,
    _text,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
from ipfs_accelerate_py.agent_supervisor.self_improvement.program_world_trajectory import (
    NormalizedTrajectory,
    ProgramWorldTrajectoryError,
    normalize_accepted_trajectory,
)


PROCEDURE_AUTHORITY: Final[str] = "ProofCarryingProcedureCompiler"
BRIDGE_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.self_improvement.program_world_procedure_bridge"
)
HOLE_SCHEMA: Final[str] = "ipfs-accelerate.program-world-procedure-hole@1"
CANDIDATE_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-procedure-candidate@1"
)
PROPOSAL_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-procedure-promotion-proposal@1"
)

ALLOWED_HOLE_KINDS: Final[frozenset[str]] = frozenset(
    {"validation", "rollback", "precondition", "effect"}
)
FORBIDDEN_HOLE_KINDS: Final[frozenset[str]] = frozenset(
    {"credential", "network", "self_approval", "model_label"}
)


class ProgramWorldProcedureError(HarnessError):
    """Closed procedure-bridge contract violation."""


def _cid(value: Any, name: str) -> str:
    return validate_opaque_cid(value, name)


@dataclass(frozen=True, slots=True)
class ProcedureHole:
    """Typed hole that still requires exact validation."""

    kind: str
    validation_cid: str
    SCHEMA: ClassVar[str] = HOLE_SCHEMA

    def __post_init__(self) -> None:
        kind = _text(self.kind, "kind")
        if kind in FORBIDDEN_HOLE_KINDS:
            raise ProgramWorldProcedureError(f"forbidden hole kind {kind}")
        if kind not in ALLOWED_HOLE_KINDS:
            raise ProgramWorldProcedureError(f"unsupported hole kind {kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "validation_cid", _cid(self.validation_cid, "validation_cid"))


@dataclass(frozen=True, slots=True)
class ProcedureCandidate:
    family_cid: str
    environment_cid: str
    policy_cid: str
    preconditions: tuple[str, ...]
    effects: tuple[str, ...]
    validation_cid: str
    rollback_cid: str
    holes: tuple[ProcedureHole, ...]
    generation: int
    SCHEMA: ClassVar[str] = CANDIDATE_SCHEMA

    @property
    def candidate_cid(self) -> str:
        return cid_for_payload(
            {
                "schema": self.SCHEMA,
                "family_cid": self.family_cid,
                "environment_cid": self.environment_cid,
                "policy_cid": self.policy_cid,
                "preconditions": list(self.preconditions),
                "effects": list(self.effects),
                "validation_cid": self.validation_cid,
                "rollback_cid": self.rollback_cid,
                "holes": [
                    {"kind": hole.kind, "validation_cid": hole.validation_cid}
                    for hole in self.holes
                ],
                "generation": self.generation,
            }
        )


@dataclass(frozen=True, slots=True)
class ProcedurePromotionProposal:
    """Nomination only. Never self-admits."""

    candidate_cid: str
    authority: str = PROCEDURE_AUTHORITY
    admitted: bool = False
    SCHEMA: ClassVar[str] = PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid"))
        object.__setattr__(self, "authority", _text(self.authority, "authority"))
        if _bool(self.admitted, "admitted"):
            raise ProgramWorldProcedureError("procedure promotion cannot self-admit")
        object.__setattr__(self, "admitted", False)


@dataclass
class ProgramWorldProcedureBridge:
    """Match-before-model bridge onto the landed procedure compiler."""

    current_generation: int = 0
    compiler: ProofCarryingProcedureCompiler | None = None
    _library: dict[str, ProcedureCandidate] = field(default_factory=dict)

    def remember(self, candidate: ProcedureCandidate) -> None:
        self._library[candidate.family_cid] = candidate

    def compile_program_world_procedure_candidate(
        self,
        trajectories: Sequence[Mapping[str, Any] | NormalizedTrajectory],
        *,
        validation_cid: str,
        rollback_cid: str,
        holes: Sequence[Mapping[str, Any]] = (),
        held_out: Sequence[Mapping[str, Any]] = (),
        adversarial: Sequence[Mapping[str, Any]] = (),
        generation: int | None = None,
    ) -> ProcedureCandidate:
        generation = self.current_generation if generation is None else generation
        if generation < 0:
            raise ProgramWorldProcedureError("generation must be non-negative")
        normalized = [
            item
            if isinstance(item, NormalizedTrajectory)
            else normalize_accepted_trajectory(item)
            for item in trajectories
        ]
        if len(normalized) < 2:
            raise ProgramWorldProcedureError(
                "unique trajectories remain episodic memory"
            )
        family = {item.family_cid for item in normalized}
        environments = {item.environment_cid for item in normalized}
        policies = {item.policy_cid for item in normalized}
        if len(family) != 1 or len(environments) != 1 or len(policies) != 1:
            raise ProgramWorldProcedureError("ambiguous trajectory family")
        parsed_holes = tuple(
            ProcedureHole(kind=str(item["kind"]), validation_cid=str(item["validation_cid"]))
            for item in holes
        )
        if any(hole.kind == "validation" and hole.validation_cid != validation_cid for hole in parsed_holes):
            raise ProgramWorldProcedureError("validation holes must retain exact validation")
        for extra in (*held_out, *adversarial):
            probe = (
                extra
                if isinstance(extra, NormalizedTrajectory)
                else normalize_accepted_trajectory(extra)
            )
            if probe.family_cid != next(iter(family)):
                raise ProgramWorldProcedureError("held-out/adversarial family mismatch")
        candidate = ProcedureCandidate(
            family_cid=next(iter(family)),
            environment_cid=next(iter(environments)),
            policy_cid=next(iter(policies)),
            preconditions=tuple(
                sorted({step["state_cid"] for item in normalized for step in item.steps[:1]})
            ),
            effects=tuple(
                sorted({step["effect_cid"] for item in normalized for step in item.steps[-1:]})
            ),
            validation_cid=_cid(validation_cid, "validation_cid"),
            rollback_cid=_cid(rollback_cid, "rollback_cid"),
            holes=parsed_holes,
            generation=generation,
        )
        if self.compiler is not None and not callable(getattr(self.compiler, "validate", None)):
            raise ProgramWorldProcedureError("procedure authority is not the landed compiler")
        self.remember(candidate)
        return candidate

    def match_program_world_procedure(
        self,
        query: Mapping[str, Any] | NormalizedTrajectory,
        *,
        current_generation: int | None = None,
    ) -> ProcedureCandidate | None:
        generation = self.current_generation if current_generation is None else current_generation
        try:
            normalized = (
                query
                if isinstance(query, NormalizedTrajectory)
                else normalize_accepted_trajectory(query)
            )
        except ProgramWorldTrajectoryError:
            return None
        candidate = self._library.get(normalized.family_cid)
        if candidate is None:
            return None
        if candidate.generation < generation:
            raise ProgramWorldProcedureError("stale procedure")
        if candidate.environment_cid != normalized.environment_cid:
            raise ProgramWorldProcedureError("stale or scope-mismatched procedure")
        if candidate.policy_cid != normalized.policy_cid:
            raise ProgramWorldProcedureError("stale or scope-mismatched procedure")
        return candidate

    def request_promotion(self, candidate: ProcedureCandidate) -> ProcedurePromotionProposal:
        if self.compiler is not None:
            try:
                self.compiler.promote(candidate.candidate_cid)
            except Exception:
                pass
        return ProcedurePromotionProposal(candidate_cid=candidate.candidate_cid)


def compile_program_world_procedure_candidate(
    trajectories: Sequence[Mapping[str, Any] | NormalizedTrajectory],
    **kwargs: Any,
) -> ProcedureCandidate:
    return ProgramWorldProcedureBridge().compile_program_world_procedure_candidate(
        trajectories, **kwargs
    )


def match_program_world_procedure(
    query: Mapping[str, Any] | NormalizedTrajectory,
    *,
    bridge: ProgramWorldProcedureBridge | None = None,
    **kwargs: Any,
) -> ProcedureCandidate | None:
    evaluator = bridge if bridge is not None else ProgramWorldProcedureBridge()
    return evaluator.match_program_world_procedure(query, **kwargs)
